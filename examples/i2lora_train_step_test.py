"""
Differentiable-injection / training-step smoke test for i2L.

Proves the mechanism a real Anima i2L trainer relies on: a predicted LoRA,
injected into a frozen backbone, produces a loss whose gradient flows back into
the *predictor* parameters (and not into the frozen base weights). Uses a tiny
stand-in backbone so it runs in a second on CPU — the same hooks apply to the
real Anima DiT (see docs/I2LORA_TRAINING.md).
"""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.i2lora import I2LConfig, I2LModel, LoRALayerSpec
from core.i2lora.inject import build_linear_map, injected_loras


class _Blk(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin1 = nn.Linear(d, d, bias=False)
        self.lin2 = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return self.lin2(torch.relu(self.lin1(x)))


class _FakeDiT(nn.Module):
    """Stand-in for the Anima DiT: Linears named blocks.<n>.lin{1,2}."""
    def __init__(self, d, n=2):
        super().__init__()
        self.blocks = nn.ModuleList(_Blk(d) for _ in range(n))

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


def main():
    print("=" * 72)
    print(" Anima i2L — differentiable injection / train-step test")
    print("=" * 72)

    d, n = 16, 2
    dit = _FakeDiT(d, n)
    for p in dit.parameters():          # freeze the backbone
        p.requires_grad_(False)

    # Layer spec whose keys match the fake DiT's Linears.
    mod_map = build_linear_map(dit)
    keys = sorted(mod_map)
    print(f"\n[+] Frozen backbone Linears: {keys}")
    specs = [LoRALayerSpec(key=k, in_dim=d, out_dim=d, kind="other") for k in keys]

    cfg = I2LConfig.for_specs(specs, dim=32, depth=2, heads=4, rank=4,
                              bottleneck=8, image_embed_dim=24, encoder="fallback")
    predictor = I2LModel(cfg)
    print(f"[+] Predictor: {predictor.num_layers} layers, "
          f"{predictor.num_parameters():,} params")

    # Fake reference-image tokens + a flow-matching-style regression target.
    image_tokens = torch.randn(10, cfg.image_embed_dim)
    x = torch.randn(4, d)
    target = torch.randn(4, d)

    # --- injection actually changes the output (the hooks fire) ---
    deltas = predictor(image_tokens)
    base_out = dit(x)
    with injected_loras(dit, deltas, alpha=cfg.rank) as info:
        lora_out = dit(x)
    assert info["applied"] == len(keys) and not info["missing"], info
    assert not torch.allclose(base_out, lora_out), "injection had no effect"
    print(f"[+] Injected {info['applied']} LoRAs; output changed  ✓")

    # --- hooks are removed on context exit (no lingering effect) ---
    assert torch.allclose(dit(x), base_out), "hooks leaked past the context"
    print("[+] Hooks removed cleanly on exit  ✓")

    # --- one optimisation step: loss -> grad into the predictor only ---
    opt = torch.optim.AdamW(predictor.parameters(), lr=1e-3)
    opt.zero_grad()
    deltas = predictor(image_tokens)
    with injected_loras(dit, deltas, alpha=cfg.rank):
        pred = dit(x)
    loss = torch.mean((pred - target) ** 2)
    loss.backward()

    grad_params = [p for p in predictor.parameters() if p.grad is not None
                   and p.grad.abs().sum() > 0]
    total = sum(1 for _ in predictor.parameters())
    print(f"[+] Predictor params receiving gradient: {len(grad_params)}/{total}")
    assert len(grad_params) >= total - 2, "gradient did not reach the predictor"
    # img_in and the decoders must be trained.
    assert predictor.img_in.weight.grad is not None
    assert predictor.dec_A[0].D.weight.grad is not None
    assert predictor.dec_B[0].D.weight.grad is not None
    # The frozen backbone must NOT have gradients.
    for p in dit.parameters():
        assert p.grad is None, "frozen backbone received a gradient"
    print("[+] Gradient reaches img_in + decoders; backbone stayed frozen  ✓")

    # --- a short optimisation run actually reduces the loss (it learns) ---
    before = loss.item()
    opt.step()
    after = before
    for _ in range(60):
        opt.zero_grad()
        with injected_loras(dit, predictor(image_tokens), alpha=cfg.rank):
            after_loss = torch.mean((dit(x) - target) ** 2)
        after_loss.backward()
        opt.step()
        after = after_loss.item()
    print(f"[+] Loss {before:.4f} -> {after:.4f} over 60 steps")
    assert after < before, "training did not reduce the loss"

    print("\n  All checks passed ✓\n")


if __name__ == "__main__":
    main()
