"""
Model loading, freeze strategies, and feature extraction hooks
for GNR 638 Assignment 2.
Uses the timm library for pretrained CNN backbones.
"""
import torch
import torch.nn as nn
import timm
from collections import OrderedDict
import config


# ─── Model Factory ───────────────────────────────────────────────────────────

def get_model(
    model_name: str,
    num_classes: int = config.NUM_CLASSES,
    strategy: str = "linear_probe",
    pretrained: bool = True,
):
    """
    Load a pretrained CNN backbone and apply the specified freeze strategy.

    Args:
        model_name: one of config.MODEL_NAMES
        num_classes: number of output classes
        strategy: 'linear_probe' | 'last_block' | 'full' | 'selective'
        pretrained: whether to load pretrained weights

    Returns:
        model (nn.Module), model_info (dict)
    """
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    # Apply freeze strategy
    if strategy == "linear_probe":
        _freeze_all_except_classifier(model, model_name)
    elif strategy == "last_block":
        _freeze_except_last_block(model, model_name)
    elif strategy == "full":
        # All params trainable — nothing to freeze
        pass
    elif strategy == "selective":
        _freeze_selective(model, model_name)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    pct_trainable = 100.0 * trainable_params / total_params if total_params > 0 else 0

    model_info = {
        "model_name": model_name,
        "strategy": strategy,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "pct_trainable": pct_trainable,
    }

    print(f"[Model] {model_name} | Strategy: {strategy} | "
          f"Total: {total_params:,} | Trainable: {trainable_params:,} ({pct_trainable:.1f}%)")

    return model, model_info


# ─── Freeze Strategies ──────────────────────────────────────────────────────

def _get_classifier_param_names(model, model_name):
    """Return set of parameter name prefixes that belong to the classifier head."""
    if model_name == "resnet50":
        return {"fc."}
    elif model_name == "densenet121":
        return {"classifier."}
    elif model_name == "efficientnet_b0":
        return {"classifier."}
    else:
        # timm generic: check for 'head', 'classifier', 'fc'
        for attr in ["head", "classifier", "fc"]:
            if hasattr(model, attr):
                return {f"{attr}."}
        return set()


def _freeze_all_except_classifier(model, model_name):
    """Freeze all parameters except the final classifier."""
    classifier_prefixes = _get_classifier_param_names(model, model_name)
    for name, param in model.named_parameters():
        if not any(name.startswith(p) for p in classifier_prefixes):
            param.requires_grad = False


def _freeze_except_last_block(model, model_name):
    """Freeze everything except the last residual/dense block and classifier."""
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False

    classifier_prefixes = _get_classifier_param_names(model, model_name)

    if model_name == "resnet50":
        # Unfreeze layer4 + fc
        for name, param in model.named_parameters():
            if name.startswith("layer4.") or any(name.startswith(p) for p in classifier_prefixes):
                param.requires_grad = True

    elif model_name == "densenet121":
        # Unfreeze denseblock4 + transition layers after it + classifier
        for name, param in model.named_parameters():
            if ("denseblock4" in name or "norm5" in name or
                    any(name.startswith(p) for p in classifier_prefixes)):
                param.requires_grad = True

    elif model_name == "efficientnet_b0":
        # Unfreeze last 2 blocks (blocks.5, blocks.6) + classifier + conv_head + bn2
        for name, param in model.named_parameters():
            if (name.startswith("blocks.5.") or name.startswith("blocks.6.") or
                    name.startswith("conv_head.") or name.startswith("bn2.") or
                    any(name.startswith(p) for p in classifier_prefixes)):
                param.requires_grad = True

    else:
        raise ValueError(f"last_block not defined for {model_name}")


def _freeze_selective(model, model_name):
    """
    Freeze ~80% of backbone parameters, unfreeze ~20% + classifier.
    Strategy: unfreeze the deepest layers that accumulate to ~20% of backbone params.
    """
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False

    classifier_prefixes = _get_classifier_param_names(model, model_name)

    # Always unfreeze classifier
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in classifier_prefixes):
            param.requires_grad = True

    # Collect backbone parameters (non-classifier) in order
    backbone_params = []
    for name, param in model.named_parameters():
        if not any(name.startswith(p) for p in classifier_prefixes):
            backbone_params.append((name, param))

    total_backbone = sum(p.numel() for _, p in backbone_params)
    target = 0.20 * total_backbone  # unfreeze 20% of backbone

    # Unfreeze from the end (deepest layers first)
    unfrozen_count = 0
    for name, param in reversed(backbone_params):
        if unfrozen_count >= target:
            break
        param.requires_grad = True
        unfrozen_count += param.numel()

    actual_pct = 100.0 * unfrozen_count / total_backbone if total_backbone > 0 else 0
    print(f"  [Selective] Unfreezing {unfrozen_count:,} / {total_backbone:,} "
          f"backbone params ({actual_pct:.1f}%)")


# ─── Feature Extraction Hooks ───────────────────────────────────────────────

def get_layer_names_for_probing(model_name: str):
    """
    Return a dict mapping probe depth labels to layer name prefixes
    for extracting intermediate features (early, middle, final).
    """
    if model_name == "resnet50":
        return OrderedDict([
            ("early",  "layer1"),
            ("middle", "layer2"),
            ("mid_deep", "layer3"),
            ("final",  "layer4"),
        ])
    elif model_name == "densenet121":
        return OrderedDict([
            ("early",  "features.denseblock1"),
            ("middle", "features.denseblock2"),
            ("mid_deep", "features.denseblock3"),
            ("final",  "features.denseblock4"),
        ])
    elif model_name == "efficientnet_b0":
        return OrderedDict([
            ("early",  "blocks.1"),
            ("middle", "blocks.3"),
            ("mid_deep", "blocks.5"),
            ("final",  "blocks.6"),
        ])
    else:
        raise ValueError(f"Probing layers not defined for {model_name}")


class FeatureExtractor:
    """
    Register forward hooks to capture intermediate layer outputs.
    Usage:
        extractor = FeatureExtractor(model, model_name)
        output = model(input)
        features = extractor.get_features()   # dict of depth_label -> tensor
        extractor.remove_hooks()
    """
    def __init__(self, model, model_name):
        self.features = {}
        self.hooks = []
        layer_map = get_layer_names_for_probing(model_name)

        for depth_label, layer_prefix in layer_map.items():
            target_module = self._find_module(model, layer_prefix)
            if target_module is not None:
                hook = target_module.register_forward_hook(
                    self._make_hook(depth_label)
                )
                self.hooks.append(hook)
            else:
                print(f"  [Warning] Could not find layer: {layer_prefix}")

    def _find_module(self, model, prefix):
        """Navigate nested modules to find the target by dot-separated prefix."""
        parts = prefix.split(".")
        module = model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            elif part.isdigit() and isinstance(module, (nn.Sequential, nn.ModuleList)):
                module = module[int(part)]
            else:
                return None
        return module

    def _make_hook(self, label):
        def hook_fn(module, input, output):
            # Global average pool spatial dimensions if needed
            if output.dim() == 4:
                pooled = torch.nn.functional.adaptive_avg_pool2d(output, 1).flatten(1)
            else:
                pooled = output.flatten(1)
            self.features[label] = pooled.detach()
        return hook_fn

    def get_features(self):
        return self.features

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def clear_features(self):
        self.features.clear()
