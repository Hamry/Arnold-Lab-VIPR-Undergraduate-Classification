"""
Model Factory Module

Handles backbone loading and classifier construction for transfer learning.
Supports various torchvision architectures with automatic feature detection.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Optional
from torchvision import models


# ---------------------------------------------------------------------------
# Backbone Registry
# ---------------------------------------------------------------------------

BACKBONE_REGISTRY = {
    # ResNet family
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
    # DenseNet family
    "densenet121": models.densenet121,
    "densenet169": models.densenet169,
    "densenet201": models.densenet201,
    # VGG family
    "vgg16": models.vgg16,
    "vgg19": models.vgg19,
    # AlexNet
    "alexnet": models.alexnet,
    # Inception / GoogLeNet family
    "inception_v3": models.inception_v3,
    "googlenet": models.googlenet,
    # EfficientNet family
    "efficientnet_b0": models.efficientnet_b0,
    "efficientnet_b1": models.efficientnet_b1,
    "efficientnet_b2": models.efficientnet_b2,
    "efficientnet_b3": models.efficientnet_b3,
    "efficientnet_b4": models.efficientnet_b4,
    # Vision Transformer family
    "vit_b_16": models.vit_b_16,
    "vit_b_32": models.vit_b_32,
    "vit_l_16": models.vit_l_16,
    # Swin Transformer family
    "swin_t": models.swin_t,
    "swin_s": models.swin_s,
    "swin_b": models.swin_b,
    # ConvNeXt family
    "convnext_tiny": models.convnext_tiny,
    "convnext_small": models.convnext_small,
    "convnext_base": models.convnext_base,
}


# ---------------------------------------------------------------------------
# Feature Dimension Detection
# ---------------------------------------------------------------------------


def _get_classifier_attr(backbone_name):
    """
    Locate the classifier attribute name for a given architecture.

    Different architectures use different attribute names for their
    classification head. This function returns the attribute name.
    """
    if backbone_name.startswith("resnet"):
        return "fc"
    elif backbone_name.startswith("densenet"):
        return "classifier"
    elif backbone_name.startswith("vgg") or backbone_name == "alexnet":
        return "classifier"
    elif backbone_name in ("inception_v3", "googlenet"):
        return "fc"
    elif backbone_name.startswith("efficientnet"):
        return "classifier"
    elif backbone_name.startswith("vit"):
        return "heads"
    elif backbone_name.startswith("swin"):
        return "head"
    elif backbone_name.startswith("convnext"):
        return "classifier"
    else:
        raise ValueError(f"Unknown backbone architecture: {backbone_name}")


def get_feature_dim(model, backbone_name):
    """
    Auto-detect the output feature dimension of a backbone.

    Args:
        model: The loaded backbone model
        backbone_name: String identifier for the architecture

    Returns:
        int: Number of features output by the backbone
    """
    classifier_attr = _get_classifier_attr(backbone_name)
    classifier = getattr(model, classifier_attr)

    if backbone_name.startswith("resnet"):
        return classifier.in_features

    elif backbone_name.startswith("densenet"):
        return classifier.in_features

    elif backbone_name.startswith("vgg") or backbone_name == "alexnet":
        # VGG/AlexNet classifier is Sequential with Linear at index 6
        return classifier[6].in_features

    elif backbone_name in ("inception_v3", "googlenet"):
        return classifier.in_features

    elif backbone_name.startswith("efficientnet"):
        # EfficientNet classifier is Sequential([Dropout, Linear])
        return classifier[1].in_features

    elif backbone_name.startswith("vit"):
        # ViT heads is Sequential or just the head module
        if hasattr(classifier, "head"):
            return classifier.head.in_features
        return classifier[0].in_features

    elif backbone_name.startswith("swin"):
        return classifier.in_features

    elif backbone_name.startswith("convnext"):
        # ConvNeXt classifier is Sequential([LayerNorm, Flatten, Linear])
        return classifier[2].in_features

    raise ValueError(f"Cannot detect feature dim for: {backbone_name}")


# ---------------------------------------------------------------------------
# Classifier Construction
# ---------------------------------------------------------------------------


def build_classifier(feature_dim, hidden_dims, num_classes, dropout):
    """
    Construct a classifier head as a Sequential module.

    Architecture: Linear -> ReLU -> Dropout -> ... -> Linear(out)

    Args:
        feature_dim: Input features from backbone
        hidden_dims: List of hidden layer sizes (can be empty)
        num_classes: Number of output classes
        dropout: Dropout probability between layers

    Returns:
        nn.Sequential: The classifier head
    """
    layers = []
    in_dim = feature_dim

    for hidden_dim in hidden_dims:
        layers.extend(
            [
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ]
        )
        in_dim = hidden_dim

    layers.append(nn.Linear(in_dim, num_classes))

    return nn.Sequential(*layers)


def attach_classifier(model, backbone_name, classifier):
    """
    Replace the backbone's original classifier with a custom one.

    Args:
        model: The backbone model
        backbone_name: String identifier for the architecture
        classifier: The new classifier module to attach
    """
    if backbone_name.startswith("resnet"):
        model.fc = classifier

    elif backbone_name.startswith("densenet"):
        model.classifier = classifier

    elif backbone_name.startswith("vgg") or backbone_name == "alexnet":
        # VGG/AlexNet: replace Linear at index 6 in classifier Sequential
        model.classifier[6] = classifier

    elif backbone_name == "inception_v3":
        model.fc = classifier
        # Disable auxiliary logits to avoid tuple output during training
        model.aux_logits = False

    elif backbone_name == "googlenet":
        model.fc = classifier
        # Disable auxiliary logits
        model.aux_logits = False

    elif backbone_name.startswith("efficientnet"):
        model.classifier = nn.Sequential(
            nn.Dropout(p=model.classifier[0].p, inplace=True), classifier
        )

    elif backbone_name.startswith("vit"):
        model.heads = classifier

    elif backbone_name.startswith("swin"):
        model.head = classifier

    elif backbone_name.startswith("convnext"):
        # Preserve LayerNorm and Flatten, replace Linear
        model.classifier[2] = classifier

    else:
        raise ValueError(f"Cannot attach classifier to: {backbone_name}")


# ---------------------------------------------------------------------------
# Freezing Logic
# ---------------------------------------------------------------------------


def freeze_backbone(model, backbone_name):
    """
    Freeze all backbone parameters, leaving only the classifier trainable.

    Args:
        model: The full model with classifier attached
        backbone_name: String identifier for the architecture
    """
    classifier_attr = _get_classifier_attr(backbone_name)

    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze classifier
    classifier = getattr(model, classifier_attr)
    for param in classifier.parameters():
        param.requires_grad = True


def count_parameters(model):
    """
    Count total and trainable parameters in a model.

    Returns:
        tuple: (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ---------------------------------------------------------------------------
# Resolution Probe
# ---------------------------------------------------------------------------


@dataclass
class UnfreezeUnit:
    """
    An atomic group of backbone layers that share a resolution stage.

    Ordered output→input when returned by get_unfreeze_units(), so index 0
    is the stage closest to the classifier head (safest to unfreeze first).
    """

    stage_id: int
    module_names: List[str]
    parameter_count: int
    resolution: Optional[tuple] = None  # (H, W) for CNN stages; None for isotropic


def get_resolution_probe(model, input_shape=(3, 224, 224)):
    """
    Probe a model's spatial resolution stages using forward hooks.

    Attaches temporary hooks to every leaf module (no children), records
    4-D tensor outputs (B, C, H, W), and groups modules into stages
    whenever the spatial resolution (H, W) changes.

    Args:
        model: Any nn.Module backbone (with classifier still attached).
        input_shape: Tuple (C, H, W) for the dummy forward pass.

    Returns:
        List[dict]: Ordered input→output, each entry:
            {
                "stage_id": int,
                "resolution": (H, W),
                "module_names": List[str],  # dotted paths, e.g. "layer1.0.conv1"
                "parameter_count": int,
            }
    """
    model.eval()

    # Build id(module) → dotted name map for all named modules
    module_to_name = {id(m): name for name, m in model.named_modules()}

    resolution_map = []
    current_stage_names = []
    last_resolution = None
    stage_counter = [0]  # list so the closure can mutate it

    def hook_fn(module, _input, output):
        if not (isinstance(output, torch.Tensor) and output.dim() == 4):
            return

        res = (output.shape[2], output.shape[3])
        nonlocal last_resolution

        if last_resolution is not None and res != last_resolution:
            # Flush completed stage
            stage_params = sum(
                p.numel()
                for name in current_stage_names
                for p in model.get_submodule(name).parameters()
            )
            resolution_map.append(
                {
                    "stage_id": stage_counter[0],
                    "resolution": last_resolution,
                    "module_names": current_stage_names.copy(),
                    "parameter_count": stage_params,
                }
            )
            current_stage_names.clear()
            stage_counter[0] += 1

        last_resolution = res
        name = module_to_name.get(id(module), "")
        if name:
            current_stage_names.append(name)

    # Attach hooks only to leaf modules (avoids double-counting containers)
    hooks = []
    for module in model.modules():
        if len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(hook_fn))

    device = next(model.parameters()).device
    dummy = torch.zeros(1, *input_shape).to(device)
    with torch.no_grad():
        try:
            model(dummy)
        except Exception:
            pass  # Some models (e.g. Inception) may error on dummy; hooks still fired

    for h in hooks:
        h.remove()

    # Flush the final stage
    if current_stage_names and last_resolution is not None:
        stage_params = sum(
            p.numel()
            for name in current_stage_names
            for p in model.get_submodule(name).parameters()
        )
        resolution_map.append(
            {
                "stage_id": stage_counter[0],
                "resolution": last_resolution,
                "module_names": current_stage_names.copy(),
                "parameter_count": stage_params,
            }
        )

    return resolution_map


# Architectures whose intermediate activations are not 4-D spatial tensors.
# Values are dotted paths to their sequential block container.
_ISOTROPIC_REGISTRY = {
    "vit_b_16": "encoder.layers",
    "vit_b_32": "encoder.layers",
    "vit_l_16": "encoder.layers",
}


def get_unfreeze_units(model, backbone_name, input_shape=(3, 224, 224)):
    """
    Partition the backbone into UnfreezeUnit objects for dynamic unfreezing.

    For CNN/Swin backbones: uses the resolution probe to group layers by stage.
    For isotropic architectures (ViT): falls back to the isotropic registry,
    treating each transformer block as its own unit.

    The returned list is ordered **output → input** (index 0 = closest to head),
    mirroring the convention of get_backbone_blocks().

    Args:
        model: The full model (backbone + classifier attached).
        backbone_name: Registry key, e.g. "resnet50".
        input_shape: (C, H, W) tuple for the resolution probe.

    Returns:
        List[UnfreezeUnit]: Units ordered output→input.
    """
    classifier_attr = _get_classifier_attr(backbone_name)

    if backbone_name in _ISOTROPIC_REGISTRY:
        # Enumerate transformer blocks by index
        container_path = _ISOTROPIC_REGISTRY[backbone_name]
        container = model.get_submodule(container_path)
        units = []
        for i, block in enumerate(container):
            block_path = f"{container_path}.{i}"
            param_count = sum(p.numel() for p in block.parameters())
            units.append(
                UnfreezeUnit(
                    stage_id=i,
                    module_names=[block_path],
                    parameter_count=param_count,
                )
            )
        # Reverse so output-most block (last index) is first
        return list(reversed(units))

    # CNN / Swin path: use the resolution probe
    resolution_map = get_resolution_probe(model, input_shape)

    # Identify which module names belong to the classifier to skip them
    classifier_module = getattr(model, classifier_attr)
    classifier_ids = {id(m) for m in classifier_module.modules()}

    units = []
    for stage in resolution_map:
        # Filter out any classifier modules that appeared in this stage
        backbone_names = [
            name
            for name in stage["module_names"]
            if id(model.get_submodule(name)) not in classifier_ids
        ]
        if not backbone_names:
            continue
        param_count = sum(
            p.numel()
            for name in backbone_names
            for p in model.get_submodule(name).parameters()
        )
        # Skip parameter-free stages (e.g. pure pooling layers like avgpool)
        if param_count == 0:
            continue
        units.append(
            UnfreezeUnit(
                stage_id=stage["stage_id"],
                module_names=backbone_names,
                parameter_count=param_count,
                resolution=stage["resolution"],
            )
        )

    # Reverse to output→input order
    return list(reversed(units))


def thaw_units(model, units):
    """
    Set requires_grad=True for all parameters in the given UnfreezeUnits.

    Args:
        model: The model.
        units: List[UnfreezeUnit] to unfreeze.

    Returns:
        int: Total number of parameters unfrozen.
    """
    total_unfrozen = 0
    for unit in units:
        for name in unit.module_names:
            submodule = model.get_submodule(name)
            for param in submodule.parameters():
                if not param.requires_grad:
                    param.requires_grad = True
                    total_unfrozen += param.numel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"--- Unfreeze Triggered: Now {trainable:,} trainable parameters ---")
    return total_unfrozen


# ---------------------------------------------------------------------------
# Progressive Unfreezing
# ---------------------------------------------------------------------------


def get_backbone_blocks(model, backbone_name):
    """
    Dynamically identify atomic blocks via named_children().

    Returns list ordered from output→input (for thawing order).
    Blocks closest to the classifier head are first (safest to unfreeze).

    Args:
        model: The loaded model with classifier attached
        backbone_name: String identifier for the architecture

    Returns:
        list[str]: Block names ordered from output to input
    """
    classifier_attr = _get_classifier_attr(backbone_name)

    blocks = []
    for name, module in model.named_children():
        # Skip classifier head
        if name == classifier_attr:
            continue
        # Keep modules that have parameters
        if any(p.numel() > 0 for p in module.parameters()):
            blocks.append(name)

    # Reverse so output layers come first (thaw from output toward input)
    return list(reversed(blocks))


def thaw_backbone_percentage(model, backbone_name, percentage):
    """
    Unfreeze a percentage of backbone blocks from output toward input.

    Simply toggles requires_grad=True on target blocks. Should be called
    with monotonically increasing percentages during training.

    Args:
        model: The model with frozen backbone
        backbone_name: String identifier for the architecture
        percentage: Float 0.0-1.0 indicating fraction to unfreeze

    Returns:
        list[str]: Names of blocks that were unfrozen
    """
    if not 0.0 <= percentage <= 1.0:
        raise ValueError(f"Thaw percentage must be 0.0-1.0, got {percentage}")

    blocks = get_backbone_blocks(model, backbone_name)
    if not blocks:
        return []

    n_to_unfreeze = max(1, int(len(blocks) * percentage))
    blocks_to_thaw = blocks[:n_to_unfreeze]

    for block_name in blocks_to_thaw:
        block = getattr(model, block_name)
        for param in block.parameters():
            param.requires_grad = True

    return blocks_to_thaw


# ---------------------------------------------------------------------------
# Main Factory Function
# ---------------------------------------------------------------------------


def load_model(options, num_classes=4):
    """
    Build a complete model from configuration.

    Args:
        options: Config dict containing 'model' key with:
            - backbone: str, architecture name
            - pretrained: bool, load ImageNet weights
            - freeze_backbone: bool, freeze backbone parameters
            - classifier_hidden: list[int], hidden layer sizes
            - dropout: float, dropout probability
        num_classes: Number of output classes (default: 4)

    Returns:
        nn.Module: Complete model ready for training
    """
    model_opts = options["model"]
    backbone_name = model_opts["backbone"]

    backbone = _load_backbone(backbone_name, model_opts["pretrained"])
    feature_dim = get_feature_dim(backbone, backbone_name)

    classifier = build_classifier(
        feature_dim=feature_dim,
        hidden_dims=model_opts["classifier_hidden"],
        num_classes=num_classes,
        dropout=model_opts["dropout"],
    )
    attach_classifier(backbone, backbone_name, classifier)

    if model_opts["freeze_backbone"]:
        freeze_backbone(backbone, backbone_name)

    total, trainable = count_parameters(backbone)
    print(f"[Model] {backbone_name}: {total:,} params, {trainable:,} trainable")

    return backbone


def _load_backbone(backbone_name, pretrained):
    """
    Load a backbone model from the registry.

    Args:
        backbone_name: Key in BACKBONE_REGISTRY
        pretrained: Whether to load ImageNet weights

    Returns:
        nn.Module: The backbone model
    """
    if backbone_name not in BACKBONE_REGISTRY:
        available = ", ".join(sorted(BACKBONE_REGISTRY.keys()))
        raise ValueError(f"Unknown backbone '{backbone_name}'. Available: {available}")

    weights = "IMAGENET1K_V1" if pretrained else None
    model = BACKBONE_REGISTRY[backbone_name](weights=weights)

    return model
