"""
A tiny, randomly-initialised demo i2L predictor.

There is **no publicly released i2L predictor for Anima** (the paper ships
predictors for Z-Image, FLUX.2-klein, and Hidream-O1 — different backbones,
different key layout). Training one is exactly the heavy meta-training this
feature deliberately does *not* do. So that the image-to-LoRA pipeline is still
runnable, testable, and demonstrable end-to-end, this module builds a small
predictor with random weights.

What the demo predictor IS: a structurally faithful i2L network whose output is
a valid Anima LoRA (correct keys, ranks, and shapes) that loads in the editor,
analyzer, preview, and ComfyUI.

What it is NOT: trained. Its predicted LoRA encodes no learned style — the
weights are random, so the result will not stylise images the way a trained
predictor would. It is for wiring/validation and for trying the workflow, not
for quality. Swap in a real trained checkpoint to get real stylisation.
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch

from .layer_spec import DEFAULT_TARGET_SET, anima_layer_spec
from .model import I2LConfig, I2LModel


def build_demo_predictor(
    target_set: str = DEFAULT_TARGET_SET,
    blocks: Optional[Iterable[int]] = None,
    rank: int = 8,
    dim: int = 256,
    depth: int = 4,
    heads: int = 8,
    bottleneck: int = 32,
    image_embed_dim: int = 768,
    seed: int = 0,
) -> I2LModel:
    """Create a small random-weight predictor over the Anima layer spec.

    Defaults are deliberately modest so it builds quickly on CPU. ``blocks``
    defaults to all 28 AnimaBlocks; pass a subset (e.g. ``range(2)``) for an
    even tinier model in tests.
    """
    torch.manual_seed(seed)
    specs = anima_layer_spec(target_set=target_set, blocks=blocks)
    config = I2LConfig.for_specs(
        specs,
        dim=dim,
        depth=depth,
        heads=heads,
        rank=rank,
        bottleneck=bottleneck,
        image_embed_dim=image_embed_dim,
        encoder="fallback",
    )
    model = I2LModel(config)
    # Scale predicted matrices down so the demo LoRA is a gentle perturbation
    # rather than a wild one (the decoders' final expansion is zero-biased).
    with torch.no_grad():
        for dec in list(model.dec_A) + list(model.dec_B):
            dec.D.weight.mul_(0.1)
    model.eval()
    return model
