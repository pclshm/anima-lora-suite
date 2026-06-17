"""
Load / save an i2L predictor checkpoint.

A predictor is stored as a single ``.safetensors`` file whose **metadata**
carries the full :class:`I2LConfig` (architecture + the target layer spec) as
JSON under the ``i2l_config`` key, and whose tensors are the predictor's
``state_dict``. That makes the file self-describing: ``load_predictor`` rebuilds
the exact network without any side-car config.

    metadata = {
        "i2l_format": "1",
        "i2l_config": "<json of I2LConfig.to_dict()>",
    }

NOTE ON EXTERNAL CHECKPOINTS. The i2L predictors released with the paper
(ModelScope: ZImage-i2L-v2, KleinBase4B-i2L-v2, HidreamO1-i2L-v2) target
*other* backbones and use the DiffSynth-Studio key layout, not this one. There
is no publicly released i2L predictor for Anima. Importing a foreign checkpoint
would need an explicit key-mapping adapter; this loader handles checkpoints
written by *this* suite (including the bundled demo predictor).
"""

from __future__ import annotations

import json
import os
from typing import Dict

import torch
from safetensors.torch import load_file, save_file

from .model import I2LConfig, I2LModel

I2L_FORMAT = "1"


def save_predictor(model: I2LModel, path: str, extra_metadata: Dict[str, str] | None = None) -> None:
    """Write the predictor weights + embedded config to ``path``."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    metadata = {
        "i2l_format": I2L_FORMAT,
        "i2l_config": json.dumps(model.config.to_dict()),
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    sd = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(sd, path, metadata=metadata)


def _read_metadata(path: str) -> Dict[str, str]:
    """Read just the safetensors metadata header without loading tensors."""
    with open(path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_len))
    return header.get("__metadata__", {}) or {}


def load_config(path: str) -> I2LConfig:
    """Load only the :class:`I2LConfig` embedded in a predictor file."""
    meta = _read_metadata(path)
    raw = meta.get("i2l_config")
    if not raw:
        raise ValueError(
            f"{path} is not an i2L predictor (no 'i2l_config' metadata). "
            "It may be a plain LoRA, a base model, or a foreign i2L checkpoint."
        )
    return I2LConfig.from_dict(json.loads(raw))


def load_predictor(path: str, device: str = "cpu") -> I2LModel:
    """Rebuild an :class:`I2LModel` from a checkpoint and load its weights."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"predictor checkpoint not found: {path}")
    config = load_config(path)
    model = I2LModel(config)
    state = load_file(path, device=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(
            f"predictor checkpoint is missing {len(missing)} tensors "
            f"(e.g. {missing[:3]}) — architecture/config mismatch."
        )
    model.eval().to(device)
    return model
