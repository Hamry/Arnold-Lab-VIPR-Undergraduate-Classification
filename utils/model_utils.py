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
# Backbone Unfreeze Block Registry
# ---------------------------------------------------------------------------
# Maps backbone name → list of block specs ordered **input→output**.
# Each spec is either a str (single module path) or list[str] (multiple paths
# that form one indivisible unit, e.g. a stem with conv + bn).
# get_unfreeze_units() and get_backbone_blocks() reverse this to output→input.

BACKBONE_UNFREEZE_BLOCKS = {
    # ResNet family — initial stem + 4 residual stages
    "resnet50":  [["conv1", "bn1"], "layer1", "layer2", "layer3", "layer4"],
    "resnet101": [["conv1", "bn1"], "layer1", "layer2", "layer3", "layer4"],
    "resnet152": [["conv1", "bn1"], "layer1", "layer2", "layer3", "layer4"],

    # DenseNet family — stem + 4 (denseblock, transition) pairs + final norm
    "densenet121": [
        ["features.conv0", "features.norm0", "features.relu0", "features.pool0"],
        ["features.denseblock1", "features.transition1"],
        ["features.denseblock2", "features.transition2"],
        ["features.denseblock3", "features.transition3"],
        ["features.denseblock4", "features.norm5"],
    ],
    "densenet169": [
        ["features.conv0", "features.norm0", "features.relu0", "features.pool0"],
        ["features.denseblock1", "features.transition1"],
        ["features.denseblock2", "features.transition2"],
        ["features.denseblock3", "features.transition3"],
        ["features.denseblock4", "features.norm5"],
    ],
    "densenet201": [
        ["features.conv0", "features.norm0", "features.relu0", "features.pool0"],
        ["features.denseblock1", "features.transition1"],
        ["features.denseblock2", "features.transition2"],
        ["features.denseblock3", "features.transition3"],
        ["features.denseblock4", "features.norm5"],
    ],

    # VGG16 — 5 pooling stages; indices derived from features Sequential (no BN)
    # Stage boundaries (MaxPool2d at indices 4, 9, 16, 23, 30):
    "vgg16": [
        [f"features.{i}" for i in range(0, 5)],   # 2 conv + pool
        [f"features.{i}" for i in range(5, 10)],  # 2 conv + pool
        [f"features.{i}" for i in range(10, 17)], # 3 conv + pool
        [f"features.{i}" for i in range(17, 24)], # 3 conv + pool
        [f"features.{i}" for i in range(24, 31)], # 3 conv + pool
    ],
    # VGG19 — 5 pooling stages (MaxPool2d at indices 4, 9, 18, 27, 36)
    "vgg19": [
        [f"features.{i}" for i in range(0, 5)],
        [f"features.{i}" for i in range(5, 10)],
        [f"features.{i}" for i in range(10, 19)], # 4 conv + pool
        [f"features.{i}" for i in range(19, 28)], # 4 conv + pool
        [f"features.{i}" for i in range(28, 37)], # 4 conv + pool
    ],

    # AlexNet — 3 groups separated by MaxPool2d
    "alexnet": [
        [f"features.{i}" for i in range(0, 3)],  # conv1 + relu + pool
        [f"features.{i}" for i in range(3, 6)],  # conv2 + relu + pool
        [f"features.{i}" for i in range(6, 13)], # conv3-5 + relu + pool
    ],

    # Inception v3 — stem convs + inception modules
    "inception_v3": [
        "Conv2d_1a_3x3", "Conv2d_2a_3x3", "Conv2d_2b_3x3",
        "Conv2d_3b_1x1", "Conv2d_4a_3x3",
        "Mixed_5b", "Mixed_5c", "Mixed_5d",
        "Mixed_6a", "Mixed_6b", "Mixed_6c", "Mixed_6d", "Mixed_6e",
        "Mixed_7a", "Mixed_7b", "Mixed_7c",
    ],

    # GoogLeNet — stem convs + inception modules
    "googlenet": [
        "conv1", "conv2", "conv3",
        "inception3a", "inception3b",
        "inception4a", "inception4b", "inception4c", "inception4d", "inception4e",
        "inception5a", "inception5b",
    ],

    # EfficientNet — stem (features.0) + 7 MBConv stages + head conv (features.8)
    "efficientnet_b0": [f"features.{i}" for i in range(9)],
    "efficientnet_b1": [f"features.{i}" for i in range(9)],
    "efficientnet_b2": [f"features.{i}" for i in range(9)],
    "efficientnet_b3": [f"features.{i}" for i in range(9)],
    "efficientnet_b4": [f"features.{i}" for i in range(9)],

    # Vision Transformer — each transformer encoder block is its own unit
    # torchvision names them encoder_layer_0 … encoder_layer_N inside encoder.layers
    "vit_b_16": [f"encoder.layers.encoder_layer_{i}" for i in range(12)],
    "vit_b_32": [f"encoder.layers.encoder_layer_{i}" for i in range(12)],
    "vit_l_16": [f"encoder.layers.encoder_layer_{i}" for i in range(24)],

    # Swin Transformer — patch embed + 4 transformer stages + 3 patch merges + norm
    # features.0=PatchEmbed, features.1/3/5/7=SwinStage, features.2/4/6=PatchMerging
    "swin_t": [f"features.{i}" for i in range(8)] + ["norm"],
    "swin_s": [f"features.{i}" for i in range(8)] + ["norm"],
    "swin_b": [f"features.{i}" for i in range(8)] + ["norm"],

    # ConvNeXt — patchify stem + 4 ConvNeXt stages + 3 downsampling layers
    "convnext_tiny":  [f"features.{i}" for i in range(8)],
    "convnext_small": [f"features.{i}" for i in range(8)],
    "convnext_base":  [f"features.{i}" for i in range(8)],
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
# Unfreeze Units
# ---------------------------------------------------------------------------


@dataclass
class UnfreezeUnit:
    """
    An atomic group of backbone modules that are thawed together.

    Ordered output→input when returned by get_unfreeze_units(), so index 0
    is the block closest to the classifier head (safest to unfreeze first).
    """

    stage_id: int
    module_names: List[str]  # dotted paths resolvable via model.get_submodule()
    parameter_count: int


def get_unfreeze_units(model, backbone_name):
    """
    Partition the backbone into UnfreezeUnit objects for dynamic unfreezing.

    Looks up the hardcoded BACKBONE_UNFREEZE_BLOCKS registry for the given
    architecture. Units are ordered **output→input** (index 0 = closest to
    the classifier head, safest to unfreeze first).

    Parameter-free entries (e.g. pure pooling blocks) are silently skipped.

    Args:
        model: The full model (backbone + classifier attached).
        backbone_name: Registry key, e.g. "resnet152".

    Returns:
        List[UnfreezeUnit]: Units ordered output→input.

    Raises:
        ValueError: If backbone_name is not in BACKBONE_UNFREEZE_BLOCKS.
    """
    if backbone_name not in BACKBONE_UNFREEZE_BLOCKS:
        raise ValueError(
            f"No unfreeze block definition for '{backbone_name}'. "
            f"Add it to BACKBONE_UNFREEZE_BLOCKS."
        )

    block_specs = BACKBONE_UNFREEZE_BLOCKS[backbone_name]  # input→output order
    units = []
    for i, spec in enumerate(block_specs):
        module_names = [spec] if isinstance(spec, str) else list(spec)
        param_count = sum(
            p.numel()
            for name in module_names
            for p in model.get_submodule(name).parameters()
        )
        if param_count == 0:
            continue  # skip parameter-free entries (e.g. pooling layers)
        units.append(
            UnfreezeUnit(
                stage_id=i,
                module_names=module_names,
                parameter_count=param_count,
            )
        )

    # Reverse to output→input order (closest to head first)
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
    Return the backbone's unfreezeble blocks ordered output→input.

    Uses the BACKBONE_UNFREEZE_BLOCKS registry. Each element of the returned
    list is itself a list of module path strings belonging to that block
    (most blocks have a single path; grouped stems have multiple).
    Parameter-free entries are silently skipped.

    Args:
        model: The loaded model with classifier attached
        backbone_name: String identifier for the architecture

    Returns:
        list[list[str]]: Block path groups ordered from output to input
    """
    if backbone_name not in BACKBONE_UNFREEZE_BLOCKS:
        raise ValueError(f"No block definition for '{backbone_name}'.")

    specs = BACKBONE_UNFREEZE_BLOCKS[backbone_name]  # input→output
    blocks = []
    for spec in specs:
        module_names = [spec] if isinstance(spec, str) else list(spec)
        param_count = sum(
            p.numel()
            for name in module_names
            for p in model.get_submodule(name).parameters()
        )
        if param_count > 0:
            blocks.append(module_names)

    return list(reversed(blocks))  # output→input


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
        list[str]: Flat list of all module paths that were unfrozen
    """
    if not 0.0 <= percentage <= 1.0:
        raise ValueError(f"Thaw percentage must be 0.0-1.0, got {percentage}")

    blocks = get_backbone_blocks(model, backbone_name)  # list[list[str]], output→input
    if not blocks:
        return []

    n_to_unfreeze = max(1, int(len(blocks) * percentage))
    blocks_to_thaw = blocks[:n_to_unfreeze]

    for module_names in blocks_to_thaw:
        for name in module_names:
            for param in model.get_submodule(name).parameters():
                param.requires_grad = True

    return [name for module_names in blocks_to_thaw for name in module_names]


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
