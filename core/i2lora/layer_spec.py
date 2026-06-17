"""
LoRA layer specification — *which* linear layers a predicted LoRA targets.

i2L (image-to-LoRA, Duan & Chen, "Compressing Image Style Training into a
Single Model Forward", arXiv:2606.13809) predicts a low-rank update

        W'_l = W_l + alpha_l * B_l A_l                                  (eq. 2)

for a fixed set of L selected linear layers {W_l}. To turn the predictor's
output into a *standard* Anima LoRA that the rest of this suite (the editor,
the analyzer, the live preview, ComfyUI, ...) already understands, every
predicted matrix pair must carry the key naming those tools recognise.

This module describes that target set. A :class:`LoRALayerSpec` records, for
one adapted layer:

    key     the LoRA module base name, e.g. ``lora_unet_blocks_0_self_attn_q_proj``
            (the suffixes ``.lora_down.weight`` / ``.lora_up.weight`` / ``.alpha``
            are appended when the state_dict is written)
    in_dim  the wrapped Linear's input features  (A is rank x in_dim)
    out_dim the wrapped Linear's output features (B is out_dim x rank)
    block   the AnimaBlock index (0..27) or None for non-block layers
    kind    one of {self_attn, cross_attn, mlp, llm_adapter, other}

The Anima dimensions below are not guesses — they are the real 2B config the
loader derives from the checkpoint (``core/anima/anima_models.py:get_dit_config``):
``model_channels`` (hidden size) = 2048, 28 blocks, cross-attention context
= 1024, MLP ratio 4 -> 8192. Attention is square (inner = hidden = 2048).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional


# --- Anima 2B dimensions (from get_dit_config, model_channels == 2048) -------

ANIMA_HIDDEN = 2048          # x_dim / model_channels
ANIMA_CROSS_CONTEXT = 1024   # crossattn_emb_channels (k/v come from text context)
ANIMA_MLP_HIDDEN = ANIMA_HIDDEN * 4   # GPT2FeedForward(x_dim, x_dim*mlp_ratio)
ANIMA_NUM_BLOCKS = 28

# Submodule layout of one AnimaBlock (see core/anima/anima_models.py:Block).
# Each tuple is (submodule_suffix, in_dim, out_dim). Names mirror the module
# attribute path with '.' -> '_', exactly as core.anima.lora_anima builds them.
_SELF_ATTN = [
    ("self_attn_q_proj", ANIMA_HIDDEN, ANIMA_HIDDEN),
    ("self_attn_k_proj", ANIMA_HIDDEN, ANIMA_HIDDEN),
    ("self_attn_v_proj", ANIMA_HIDDEN, ANIMA_HIDDEN),
    ("self_attn_output_proj", ANIMA_HIDDEN, ANIMA_HIDDEN),
]
_CROSS_ATTN = [
    ("cross_attn_q_proj", ANIMA_HIDDEN, ANIMA_HIDDEN),
    ("cross_attn_k_proj", ANIMA_CROSS_CONTEXT, ANIMA_HIDDEN),
    ("cross_attn_v_proj", ANIMA_CROSS_CONTEXT, ANIMA_HIDDEN),
    ("cross_attn_output_proj", ANIMA_HIDDEN, ANIMA_HIDDEN),
]
_MLP = [
    ("mlp_layer1", ANIMA_HIDDEN, ANIMA_MLP_HIDDEN),
    ("mlp_layer2", ANIMA_MLP_HIDDEN, ANIMA_HIDDEN),
]

# Named target sets, smallest -> largest. "Style transfer LoRAs" are usually
# attention-only; the paper itself adapts "selected linear layers".
TARGET_SETS = {
    "cross_attn": ["cross_attn"],
    "attn": ["self_attn", "cross_attn"],
    "attn+mlp": ["self_attn", "cross_attn", "mlp"],
}
DEFAULT_TARGET_SET = "attn"

_KIND_TABLE = {
    "self_attn": _SELF_ATTN,
    "cross_attn": _CROSS_ATTN,
    "mlp": _MLP,
}


@dataclass
class LoRALayerSpec:
    """One adapted linear layer the i2L predictor must emit A/B matrices for."""
    key: str
    in_dim: int
    out_dim: int
    kind: str = "other"
    block: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LoRALayerSpec":
        return cls(
            key=d["key"],
            in_dim=int(d["in_dim"]),
            out_dim=int(d["out_dim"]),
            kind=d.get("kind", "other"),
            block=(None if d.get("block") is None else int(d["block"])),
        )


def anima_layer_spec(
    target_set: str = DEFAULT_TARGET_SET,
    blocks: Optional[Iterable[int]] = None,
    prefix: str = "lora_unet",
) -> List[LoRALayerSpec]:
    """Build the Anima target layer list.

    Args:
        target_set: one of ``TARGET_SETS`` (``cross_attn`` / ``attn`` / ``attn+mlp``).
        blocks:     which AnimaBlock indices to adapt (default: all 28).
        prefix:     LoRA key prefix; ``lora_unet`` is the ComfyUI-compatible one
                    that ``core.anima.lora_anima`` and the editor expect.

    Returns the ordered list of :class:`LoRALayerSpec`. The order is stable
    (block, then submodule) so a saved predictor's queries line up across runs.
    """
    if target_set not in TARGET_SETS:
        raise ValueError(
            f"unknown target_set {target_set!r}; choose one of {sorted(TARGET_SETS)}"
        )
    block_ids = sorted(set(range(ANIMA_NUM_BLOCKS) if blocks is None else blocks))
    kinds = TARGET_SETS[target_set]

    specs: List[LoRALayerSpec] = []
    for b in block_ids:
        for kind in kinds:
            for suffix, din, dout in _KIND_TABLE[kind]:
                specs.append(
                    LoRALayerSpec(
                        key=f"{prefix}_blocks_{b}_{suffix}",
                        in_dim=din,
                        out_dim=dout,
                        kind=kind,
                        block=b,
                    )
                )
    return specs


def spec_to_list(specs: List[LoRALayerSpec]) -> List[dict]:
    return [s.to_dict() for s in specs]


def spec_from_list(items: List[dict]) -> List[LoRALayerSpec]:
    return [LoRALayerSpec.from_dict(d) for d in items]


def summarize_spec(specs: List[LoRALayerSpec]) -> Dict[str, int]:
    """Small human-readable rollup for logs / the UI."""
    blocks = sorted({s.block for s in specs if s.block is not None})
    kinds: Dict[str, int] = {}
    for s in specs:
        kinds[s.kind] = kinds.get(s.kind, 0) + 1
    return {
        "layers": len(specs),
        "blocks": len(blocks),
        "block_min": blocks[0] if blocks else -1,
        "block_max": blocks[-1] if blocks else -1,
        **{f"kind_{k}": v for k, v in kinds.items()},
    }
