"""Shared TorchJD Jacobian descent for MOJD trainers (TorchJD 0.10+)."""

from __future__ import annotations

from collections.abc import Sequence

import torch.nn as nn
from torch import Tensor
from torchjd.aggregation import Aggregator
from torchjd.autojac import backward, jac_to_grad, mtl_backward
from torchjd.autojac._utils import as_checked_ordered_set, get_leaf_tensors


def has_split_forward(model: nn.Module) -> bool:
    """True if ``model`` exposes ``forward_features`` and ``forward_classifier`` (encoder + head)."""
    return callable(getattr(model, "forward_features", None)) and callable(
        getattr(model, "forward_classifier", None)
    )


def assert_split_forward_if_mtl(model: nn.Module, backward_mode: str) -> None:
    """Raise when MTL backward is requested but the model has no encoder/head split."""
    if backward_mode != "mtl":
        return
    if not has_split_forward(model):
        raise ValueError(
            "args.backward='mtl' requires model.forward_features and model.forward_classifier "
            "(e.g. ResNet / Swin in src/models)."
        )


def jd_backward(
    losses: Sequence[Tensor],
    model: nn.Module,
    aggregator: Aggregator,
    backward_mode: str,
    *,
    features: Sequence[Tensor] | None = None,
) -> None:
    """One Jacobian-descent step: ``full`` = full Jacobian + aggregate; ``mtl`` = trunk + aggregate."""
    mode = backward_mode if backward_mode in ("full", "mtl") else "full"
    if mode == "mtl":
        if features is None:
            raise ValueError("backward='mtl' requires features (e.g. per-batch forward_features).")
        features_seq = list(features)
        features_ = as_checked_ordered_set(features_seq, "features")
        shared_params = tuple(get_leaf_tensors(tensors=features_, excluded=[]))
        mtl_backward(losses, features_seq, shared_params=shared_params)
        jac_to_grad(shared_params, aggregator)
    else:
        params = tuple(model.parameters())
        backward(losses, inputs=params)
        jac_to_grad(params, aggregator)
