"""
Model Factory Module

Handles backbone loading and classifier construction for transfer learning.
Supports various torchvision architectures with automatic feature detection.
"""

import torch
import torch.nn as nn
from torchvision import models


# ---------------------------------------------------------------------------
# Backbone Registry
# ---------------------------------------------------------------------------

BACKBONE_REGISTRY = {
    # ResNet family
    'resnet50': models.resnet50,
    'resnet101': models.resnet101,
    'resnet152': models.resnet152,

    # DenseNet family
    'densenet121': models.densenet121,
    'densenet169': models.densenet169,
    'densenet201': models.densenet201,

    # VGG family
    'vgg16': models.vgg16,
    'vgg19': models.vgg19,

    # AlexNet
    'alexnet': models.alexnet,

    # Inception / GoogLeNet family
    'inception_v3': models.inception_v3,
    'googlenet': models.googlenet,

    # EfficientNet family
    'efficientnet_b0': models.efficientnet_b0,
    'efficientnet_b1': models.efficientnet_b1,
    'efficientnet_b2': models.efficientnet_b2,
    'efficientnet_b3': models.efficientnet_b3,
    'efficientnet_b4': models.efficientnet_b4,

    # Vision Transformer family
    'vit_b_16': models.vit_b_16,
    'vit_b_32': models.vit_b_32,
    'vit_l_16': models.vit_l_16,

    # Swin Transformer family
    'swin_t': models.swin_t,
    'swin_s': models.swin_s,
    'swin_b': models.swin_b,

    # ConvNeXt family
    'convnext_tiny': models.convnext_tiny,
    'convnext_small': models.convnext_small,
    'convnext_base': models.convnext_base,
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
    if backbone_name.startswith('resnet'):
        return 'fc'
    elif backbone_name.startswith('densenet'):
        return 'classifier'
    elif backbone_name.startswith('vgg') or backbone_name == 'alexnet':
        return 'classifier'
    elif backbone_name in ('inception_v3', 'googlenet'):
        return 'fc'
    elif backbone_name.startswith('efficientnet'):
        return 'classifier'
    elif backbone_name.startswith('vit'):
        return 'heads'
    elif backbone_name.startswith('swin'):
        return 'head'
    elif backbone_name.startswith('convnext'):
        return 'classifier'
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

    if backbone_name.startswith('resnet'):
        return classifier.in_features

    elif backbone_name.startswith('densenet'):
        return classifier.in_features

    elif backbone_name.startswith('vgg') or backbone_name == 'alexnet':
        # VGG/AlexNet classifier is Sequential with Linear at index 6
        return classifier[6].in_features

    elif backbone_name in ('inception_v3', 'googlenet'):
        return classifier.in_features

    elif backbone_name.startswith('efficientnet'):
        # EfficientNet classifier is Sequential([Dropout, Linear])
        return classifier[1].in_features

    elif backbone_name.startswith('vit'):
        # ViT heads is Sequential or just the head module
        if hasattr(classifier, 'head'):
            return classifier.head.in_features
        return classifier[0].in_features

    elif backbone_name.startswith('swin'):
        return classifier.in_features

    elif backbone_name.startswith('convnext'):
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
        layers.extend([
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        ])
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
    if backbone_name.startswith('resnet'):
        model.fc = classifier

    elif backbone_name.startswith('densenet'):
        model.classifier = classifier

    elif backbone_name.startswith('vgg') or backbone_name == 'alexnet':
        # VGG/AlexNet: replace Linear at index 6 in classifier Sequential
        model.classifier[6] = classifier

    elif backbone_name == 'inception_v3':
        model.fc = classifier
        # Disable auxiliary logits to avoid tuple output during training
        model.aux_logits = False

    elif backbone_name == 'googlenet':
        model.fc = classifier
        # Disable auxiliary logits
        model.aux_logits = False

    elif backbone_name.startswith('efficientnet'):
        model.classifier = nn.Sequential(
            nn.Dropout(p=model.classifier[0].p, inplace=True),
            classifier
        )

    elif backbone_name.startswith('vit'):
        model.heads = classifier

    elif backbone_name.startswith('swin'):
        model.head = classifier

    elif backbone_name.startswith('convnext'):
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
    model_opts = options['model']
    backbone_name = model_opts['backbone']

    backbone = _load_backbone(backbone_name, model_opts['pretrained'])
    feature_dim = get_feature_dim(backbone, backbone_name)

    classifier = build_classifier(
        feature_dim=feature_dim,
        hidden_dims=model_opts['classifier_hidden'],
        num_classes=num_classes,
        dropout=model_opts['dropout']
    )
    attach_classifier(backbone, backbone_name, classifier)

    if model_opts['freeze_backbone']:
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
        available = ', '.join(sorted(BACKBONE_REGISTRY.keys()))
        raise ValueError(
            f"Unknown backbone '{backbone_name}'. Available: {available}"
        )

    weights = 'IMAGENET1K_V1' if pretrained else None
    model = BACKBONE_REGISTRY[backbone_name](weights=weights)

    return model
