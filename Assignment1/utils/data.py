from __future__ import annotations

import os
import random
import time
from array import array
from dataclasses import dataclass
from typing import Dict, Generator, List, Sequence, Tuple

import cv2

from dlframework.tensor import Tensor


@dataclass
class DatasetIndex:
    samples: List[Tuple[str, int]]
    label_to_name: Dict[int, str]
    name_to_label: Dict[str, int]
    indexing_time_sec: float


class ImageFolderDataset:
    def __init__(self, root: str, image_size: int = 32):
        self.root = root
        self.image_size = image_size
        self.index = self._build_index(root)

    def _build_index(self, root: str) -> DatasetIndex:
        t0 = time.time()
        class_names = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        class_names.sort()

        name_to_label = {name: i for i, name in enumerate(class_names)}
        label_to_name = {i: name for name, i in name_to_label.items()}

        samples: List[Tuple[str, int]] = []
        for name in class_names:
            class_dir = os.path.join(root, name)
            for fname in os.listdir(class_dir):
                lower = fname.lower()
                if lower.endswith(".png"):
                    samples.append((os.path.join(class_dir, fname), name_to_label[name]))

        return DatasetIndex(
            samples=samples,
            label_to_name=label_to_name,
            name_to_label=name_to_label,
            indexing_time_sec=time.time() - t0,
        )

    @property
    def num_classes(self) -> int:
        return len(self.index.label_to_name)

    @property
    def samples(self) -> List[Tuple[str, int]]:
        return self.index.samples

    def measure_decode_time(self) -> float:
        t0 = time.time()
        for path, _ in self.samples:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Failed to read image: {path}")
            _ = self._to_rgb_32(img)
        return time.time() - t0

    def _to_rgb_32(self, img):
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
            img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        return img

    def load_image_to_chw_array(
        self,
        path: str,
        augment: bool = False,
        rng: random.Random | None = None,
        augment_crop_pad: int = 0,
        normalize_mean: List[float] | None = None,
        normalize_std: List[float] | None = None,
    ) -> array:
        if rng is None:
            rng = random
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to read image: {path}")

        img = self._to_rgb_32(img)

        if augment and augment_crop_pad > 0:
            p = augment_crop_pad
            img = cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_REFLECT_101)
            oy = rng.randint(0, 2 * p)
            ox = rng.randint(0, 2 * p)
            img = img[oy : oy + self.image_size, ox : ox + self.image_size]

        if augment and rng.random() < 0.5:
            img = cv2.flip(img, 1)

        if normalize_mean is None:
            normalize_mean = [0.0, 0.0, 0.0]
        if normalize_std is None:
            normalize_std = [1.0, 1.0, 1.0]

        h, w, _ = img.shape
        # Convert HWC uint8 to normalized CHW float32 without using NumPy APIs directly.
        data = array("f", [0.0]) * (3 * h * w)
        for y in range(h):
            for x in range(w):
                px = img[y, x]
                r = (float(px[0]) / 255.0 - normalize_mean[0]) / normalize_std[0]
                g = (float(px[1]) / 255.0 - normalize_mean[1]) / normalize_std[1]
                b = (float(px[2]) / 255.0 - normalize_mean[2]) / normalize_std[2]
                idx = y * w + x
                data[idx] = r
                data[h * w + idx] = g
                data[2 * h * w + idx] = b
        return data


def split_train_val(samples: Sequence[Tuple[str, int]], val_ratio: float, seed: int) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    val_size = int(len(shuffled) * val_ratio)
    val_samples = shuffled[:val_size]
    train_samples = shuffled[val_size:]
    return train_samples, val_samples


def iter_batches(
    dataset: ImageFolderDataset,
    samples: Sequence[Tuple[str, int]],
    batch_size: int,
    shuffle: bool,
    augment: bool,
    seed: int,
    augment_crop_pad: int = 0,
    normalize_mean: List[float] | None = None,
    normalize_std: List[float] | None = None,
) -> Generator[Tuple[Tensor, List[int]], None, None]:
    order = list(samples)
    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(order)

    i = 0
    image_size = dataset.image_size
    c = 3
    per = c * image_size * image_size

    while i < len(order):
        chunk = order[i : i + batch_size]
        n = len(chunk)
        batch_data = array("f", [0.0]) * (n * per)
        labels: List[int] = []
        for j, (path, label) in enumerate(chunk):
            img = dataset.load_image_to_chw_array(
                path,
                augment=augment,
                rng=rng,
                augment_crop_pad=augment_crop_pad,
                normalize_mean=normalize_mean,
                normalize_std=normalize_std,
            )
            offset = j * per
            batch_data[offset : offset + per] = img
            labels.append(label)

        yield Tensor(batch_data, (n, c, image_size, image_size), requires_grad=False), labels
        i += batch_size
