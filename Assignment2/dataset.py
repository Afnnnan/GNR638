"""
Dataset loading, train/val splits, few-shot subsets, and corruption transforms
for GNR 638 Assignment 2.
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from PIL import ImageFilter, ImageEnhance
import config


# ─── Standard Transforms ────────────────────────────────────────────────────

def get_train_transform(img_size: int = config.IMG_SIZE):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


def get_val_transform(img_size: int = config.IMG_SIZE):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


# ─── Corruption Transforms ──────────────────────────────────────────────────

class GaussianNoise:
    """Add Gaussian noise to a tensor image."""
    def __init__(self, sigma: float = 0.1):
        self.sigma = sigma

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.sigma
        return torch.clamp(tensor + noise, 0.0, 1.0)


class MotionBlur:
    """Apply motion blur via a PIL filter before tensor conversion."""
    def __init__(self, kernel_size: int = 15):
        self.kernel_size = kernel_size

    def __call__(self, img):
        # Horizontal motion blur kernel
        kernel = [0] * self.kernel_size * self.kernel_size
        mid = self.kernel_size // 2
        for i in range(self.kernel_size):
            kernel[mid * self.kernel_size + i] = 1.0 / self.kernel_size
        return img.filter(ImageFilter.Kernel(
            size=(self.kernel_size, self.kernel_size),
            kernel=kernel,
        ))


class BrightnessShift:
    """Shift brightness of a PIL image."""
    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def __call__(self, img):
        return ImageEnhance.Brightness(img).enhance(self.factor)


def get_corruption_transform(corruption_type: str, img_size: int = config.IMG_SIZE, **kwargs):
    """
    Build a validation-time corruption transform.
    corruption_type: 'gaussian_noise', 'motion_blur', 'brightness_shift'
    """
    base_pre = [transforms.Resize((img_size, img_size))]
    to_tensor = [transforms.ToTensor()]
    normalize = [transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)]

    if corruption_type == "gaussian_noise":
        sigma = kwargs.get("sigma", 0.1)
        return transforms.Compose(
            base_pre + to_tensor + [GaussianNoise(sigma)] + normalize
        )
    elif corruption_type == "motion_blur":
        kernel_size = kwargs.get("kernel_size", 15)
        return transforms.Compose(
            base_pre + [MotionBlur(kernel_size)] + to_tensor + normalize
        )
    elif corruption_type == "brightness_shift":
        factor = kwargs.get("factor", 1.5)
        return transforms.Compose(
            base_pre + [BrightnessShift(factor)] + to_tensor + normalize
        )
    else:
        raise ValueError(f"Unknown corruption: {corruption_type}")


# ─── Dataset Construction ────────────────────────────────────────────────────

def get_dataset(root: str = config.DATA_ROOT, transform=None):
    """Load the AID dataset using ImageFolder."""
    return datasets.ImageFolder(root=root, transform=transform)


def get_train_val_indices(dataset, val_fraction: float = config.VAL_FRACTION, seed: int = config.SEED):
    """Stratified train/val split returning index arrays."""
    targets = np.array([s[1] for s in dataset.samples])
    indices = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(
        indices, test_size=val_fraction, stratify=targets, random_state=seed
    )
    return train_idx, val_idx


def get_fewshot_indices(dataset, indices, fraction: float, seed: int = config.SEED):
    """
    Sub-sample `fraction` of the given indices in a stratified manner.
    Returns the subset of indices.
    """
    if fraction >= 1.0:
        return indices
    targets = np.array([dataset.samples[i][1] for i in indices])
    _, subset_idx = train_test_split(
        indices, test_size=fraction, stratify=targets, random_state=seed
    )
    return subset_idx


def build_dataloaders(
    train_transform=None,
    val_transform=None,
    fewshot_fraction: float = 1.0,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    seed: int = config.SEED,
):
    """
    Build train and val DataLoaders from the AID dataset.
    Optionally subsample training data for few-shot experiments.
    """
    config.set_seed(seed)

    # We load dataset twice with different transforms
    full_dataset_train = get_dataset(transform=train_transform or get_train_transform())
    full_dataset_val   = get_dataset(transform=val_transform or get_val_transform())

    train_idx, val_idx = get_train_val_indices(full_dataset_train, seed=seed)

    # Few-shot sub-sampling on training set
    if fewshot_fraction < 1.0:
        train_idx = get_fewshot_indices(full_dataset_train, train_idx, fewshot_fraction, seed=seed)

    train_subset = Subset(full_dataset_train, train_idx)
    val_subset   = Subset(full_dataset_val, val_idx)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )

    class_names = full_dataset_train.classes
    print(f"[Dataset] Train: {len(train_subset)} | Val: {len(val_subset)} | "
          f"Classes: {len(class_names)} | Few-shot: {fewshot_fraction*100:.0f}%")

    return train_loader, val_loader, class_names


def build_corruption_loader(
    corruption_type: str,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    seed: int = config.SEED,
    **kwargs,
):
    """Build a DataLoader for the validation set with a specific corruption applied."""
    config.set_seed(seed)
    corrupt_transform = get_corruption_transform(corruption_type, **kwargs)
    full_dataset = get_dataset(transform=corrupt_transform)
    _, val_idx = get_train_val_indices(full_dataset, seed=seed)
    val_subset = Subset(full_dataset, val_idx)
    loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return loader
