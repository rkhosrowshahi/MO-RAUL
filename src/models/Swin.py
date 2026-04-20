"""
Swin Transformer wrapper for compatibility with MOMU training pipeline.
Provides same interface as ResNet (num_classes, imagenet, normalize,
forward_features, forward_classifier).

Uses torchvision Swin API: build with weights enum, then replace head.
"""

import torch
import torch.nn as nn
from torchvision.models import (
    swin_t,
    swin_s,
    swin_b,
    Swin_T_Weights,
    Swin_S_Weights,
    Swin_B_Weights,
)

from .ResNet import NormalizeByChannelMeanStd

_SWINS = {
    "swin_t": (swin_t, Swin_T_Weights.IMAGENET1K_V1),
    "swin_s": (swin_s, Swin_S_Weights.IMAGENET1K_V1),
    "swin_b": (swin_b, Swin_B_Weights.IMAGENET1K_V1),
}


class SwinTWrapper(nn.Module):
    """Wrapper adding normalize, forward_features, forward_classifier for MOMU compatibility."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.normalize = NormalizeByChannelMeanStd(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def forward(self, x):
        x = self.normalize(x)
        return self.backbone(x)

    def forward_features(self, x):
        x = self.normalize(x)
        x = self.backbone.features(x)
        x = self.backbone.norm(x)
        x = self.backbone.permute(x)
        x = self.backbone.avgpool(x)
        x = self.backbone.flatten(x)
        return x

    def forward_classifier(self, features):
        return self.backbone.head(features)


def _swin(name: str, num_classes: int = 1000, imagenet: bool = True, pretrained: bool = False, **kwargs):
    """
    Swin Transformer model (Tiny/Small/Base) for ImageNet-sized inputs (224x224).

    Args:
        name: One of "swin_t", "swin_s", "swin_b"
        num_classes: Number of output classes
        imagenet: Must be True for Swin (uses ImageNet normalization and 224x224 input)
        pretrained: If True, load ImageNet-pretrained weights
    """
    if not imagenet:
        raise ValueError("Swin Transformer requires imagenet=True (224x224 input)")

    ctor, weights_enum = _SWINS[name.lower()]
    weights = weights_enum if pretrained else None
    backbone = ctor(weights=weights, **kwargs)

    # Adjusted new head: replace classifier for target num_classes
    backbone.head = nn.Linear(backbone.head.in_features, num_classes)

    model = SwinTWrapper(backbone)
    return model


def swin_tiny(num_classes=1000, imagenet=True, pretrained=False, **kwargs):
    """Swin-T (Tiny) model for ImageNet-sized inputs (224x224)."""
    return _swin("swin_t", num_classes=num_classes, imagenet=imagenet, pretrained=pretrained, **kwargs)


def swin_small(num_classes=1000, imagenet=True, pretrained=False, **kwargs):
    """Swin-S (Small) model for ImageNet-sized inputs (224x224)."""
    return _swin("swin_s", num_classes=num_classes, imagenet=imagenet, pretrained=pretrained, **kwargs)


def swin_base(num_classes=1000, imagenet=True, pretrained=False, **kwargs):
    """Swin-B (Base) model for ImageNet-sized inputs (224x224)."""
    return _swin("swin_b", num_classes=num_classes, imagenet=imagenet, pretrained=pretrained, **kwargs)
