# Training an i2L predictor for Anima

This guide explains how to **meta-train** an image-to-LoRA predictor for Anima —
the step the inference integration ([`docs/I2LORA.md`](I2LORA.md)) deliberately
leaves out. After training you get a predictor checkpoint that
`python i2l.py create …` / the **Create from Images** panel turn into real
style LoRAs.

> **Scope.** The suite ships the *inference* path plus the one training
> primitive that is genuinely Anima-specific — differentiable LoRA injection
> (`core/i2lora/inject.py`, verified by `examples/i2lora_train_step_test.py`).
> It does **not** ship a turnkey training CLI: real training needs multiple GPUs,
> the Anima model weights, and a large style dataset, none of which belong in
> the repo. What follows is the complete recipe and a reference training loop
> you assemble from the building blocks already here.

---

## 1. What you are training (and what stays frozen)

You train **only** the predictor `G_φ` (the `I2LModel`): its image projection,
LoRA queries, transformer, and compressed decoders. Everything else is frozen:

| Component | Role in training | Frozen? |
| --------- | ---------------- | ------- |
| **i2L predictor** (`core.i2lora.I2LModel`) | predicts A/B from reference images | **trained** |
| **SigLIP2** image encoder | reference images → patch tokens | frozen |
| **Anima DiT** (`core.anima`) | the backbone the loss flows through | frozen |
| **Qwen-Image VAE** | target image → latent | frozen |
| **Qwen3** text encoder + LLMAdapter | prompt → cross-attention context | frozen |

The objective: the predicted LoRA, inserted into the frozen DiT, must let the
frozen backbone model **target images in the reference style** under the
standard flow-matching loss (paper eq. 6).

---

## 2. Requirements

**Hardware.** The paper trains each backbone for ~7 days on **8× A100** (lr
1e-5, global batch 8). You can train smaller/shorter for a weaker predictor, but
this is a multi-GPU, multi-day job. A single 24 GB GPU can train a *reduced*
predictor (fewer adapted layers, gradient checkpointing) slowly.

**Weights you must supply.**
- **Anima DiT**, **Qwen-Image VAE**, **Qwen3-0.6B** — the same three files the
  live preview uses (`setup_preview` + Model paths). Loaded via
  `core.anima.anima_utils`.
- **SigLIP2** — e.g. a `google/siglip2-*` checkpoint, loaded through 🤗
  `transformers` by `core.i2lora.encoder.SiglipImageEncoder`.

**Python extras.** `transformers`, `accelerate`, `pillow`, `einops`,
`sentencepiece` (all in `requirements-preview.txt`) plus a CUDA build of
`torch`.

---

## 3. Data — the part that decides quality

i2L's whole point is to learn *style*, not to copy reference content. That
requires **style-consistent, content-disjoint** training tuples (paper Sec. 3.3):

> Each tuple = {one or more **reference images** sharing a style} + {a **target
> image** in that same style **but different content**} + {a **prompt**
> describing the *target's* content}.

If reference and target share content, the predictor cheaply lowers the loss by
copying objects/identity (semantic leakage) instead of learning style.

- The paper uses **MegaStyle-1M** (arXiv:2604.08364) — ~1M tuples built for
  exactly this property. Use it if you can obtain it.
- **DIY:** group images by style (a creator, a render engine, a medium), then
  for each tuple draw references and a target from the *same style group* but
  ensure they depict *different subjects*. Caption the target's content with a
  VLM. Upscale to 1024².

A `Dataset` should yield, per item: `ref_images: list[PIL.Image]`,
`target_image: PIL.Image`, `prompt: str`.

---

## 4. The loss — Anima's rectified-flow convention

Anima is a **velocity** model and this suite's sampler integrates
`denoised = x − σ·v` with `σ ∈ [1,0]` (σ=1 noise → σ=0 clean; see
`core/preview/backends.py`). Train in that exact convention so the resulting
LoRA works in the preview and ComfyUI. For a clean latent `x1` and noise
`ε ∼ N(0,I)`:

```
σ ~ U(0,1)                         # per sample
x_σ      = (1−σ)·x1 + σ·ε          # noised latent (the DiT's input)
v_target = ε − x1                  # Anima velocity target
v_pred   = DiT(x_σ, σ, cross_emb)  # with the PREDICTED LoRA injected
L        = ‖ v_pred − v_target ‖²
```

(This is the paper's flow-matching loss, eq. 5–6, written in Anima's sign
convention. Check: `denoised = x_σ − σ·v_target = x1`, so a predictor trained
this way denoises to the styled target — exactly what `res_2m`/`euler` expect.)

Gradients flow `L → v_pred → injected A/B → decoders → transformer → queries`,
updating only `φ`.

---

## 5. The key mechanism: differentiable injection

At inference the suite **bakes** the LoRA into the DiT (`lora_anima.merge_to`,
in-place bf16) — no gradient path. For training, apply the predicted matrices as
an autograd-tracked additive path instead, with `core/i2lora/inject.py`:

```python
from core.i2lora import injected_loras, build_linear_map

module_map = build_linear_map(dit)          # build once, reuse every step
deltas = predictor(image_tokens)            # {key: {"A","B"}}, requires grad
with injected_loras(dit, deltas, alpha=predictor.config.rank, module_map=module_map):
    v_pred = dit(x_sigma, sigma, cross_emb, padding_mask=pad)
loss = ((v_pred.float() - v_target.float()) ** 2).mean()
loss.backward()                              # grads reach the predictor only
```

This is verified end-to-end (gradient reaches the predictor, frozen backbone
stays at `grad is None`, the loss goes down) by:

```bash
python examples/i2lora_train_step_test.py
```

**One LoRA per style.** i2L predicts a *single* LoRA per reference set, so a DiT
forward can only mix targets that share that style. Use micro-batch = one style
(optionally several same-style targets stacked in the batch dim) and reach the
paper's global batch via **gradient accumulation**.

---

## 6. Reference training loop

Assemble this from the pieces already in the repo. It is a starting point, not a
supported CLI — adapt paths, dataset, schedule, and parallelism to your setup.

```python
import torch
from torch.utils.data import DataLoader

from core.anima import anima_utils as au, strategy_anima as sa
from core.preview.backends import embed_cross_context
from core.i2lora import (
    anima_layer_spec, I2LConfig, I2LModel, save_predictor,
    injected_loras, build_linear_map,
)
from core.i2lora.encoder import SiglipImageEncoder

device, dtype = "cuda", torch.bfloat16

# --- frozen backbone ---------------------------------------------------------
qwen3, _ = au.load_qwen3_text_encoder(TE_PATH, dtype=dtype, device="cpu")
qwen3.eval().to(device)
dit = au.load_anima_dit(DIT_PATH, dtype=dtype, device="cpu").to(device)
vae, _, _, vae_scale = au.load_anima_vae(VAE_PATH, dtype=dtype, device="cpu")
vae.eval().to(device); vae_scale = [t.to(device) for t in vae_scale]
tok = sa.AnimaTokenizeStrategy(qwen3_path=TE_PATH, qwen3_max_length=1024)
enc = sa.AnimaTextEncodingStrategy()

for m in (dit, vae, qwen3):
    for p in m.parameters():
        p.requires_grad_(False)
dit.train()                          # enables gradient checkpointing branch
dit.enable_gradient_checkpointing()  # essential for VRAM

# --- frozen image encoder + trainable predictor ------------------------------
img_enc = SiglipImageEncoder(SIGLIP_PATH, device=device)
specs = anima_layer_spec(target_set="attn")     # try "cross_attn" first (cheapest)
cfg = I2LConfig.for_specs(
    specs, dim=1024, depth=8, heads=16, rank=16, bottleneck=64,
    image_embed_dim=img_enc.embed_dim, encoder="siglip2",
)
predictor = I2LModel(cfg).to(device).float()    # train the predictor in fp32
module_map = build_linear_map(dit)

opt = torch.optim.AdamW(predictor.parameters(), lr=1e-5, weight_decay=0.0)
loader = DataLoader(my_style_dataset, batch_size=1, shuffle=True,
                    collate_fn=lambda b: b[0])  # one style per item
ACCUM = 8                                        # → global batch 8

def to_latent(pil_image):
    import numpy as np
    a = torch.from_numpy(np.asarray(pil_image.convert("RGB"))).float() / 127.5 - 1.0
    x = a.permute(2, 0, 1)[None, :, None]        # (B=1, 3, T=1, H, W)
    return vae.encode(x.to(device, dtype), vae_scale)   # (1, 16, 1, H/8, W/8)

step = 0
for epoch in range(EPOCHS):
    for item in loader:
        with torch.no_grad():
            tokens = img_enc.encode(item["ref_images"])      # (Ntok, d_img)
            x1 = to_latent(item["target_image"]).float()
            cross = embed_cross_context(dit, tok, enc, qwen3, item["prompt"], device)

        deltas = predictor(tokens.to(device))
        sigma = torch.rand(x1.shape[0], device=device)
        eps = torch.randn_like(x1)
        s = sigma.view(-1, 1, 1, 1, 1)
        x_sigma = (1 - s) * x1 + s * eps
        v_target = eps - x1
        pad = torch.zeros(x1.shape[0], 1, x1.shape[-2], x1.shape[-1],
                          dtype=dtype, device=device)

        with injected_loras(dit, deltas, alpha=cfg.rank, module_map=module_map):
            with torch.autocast("cuda", dtype=dtype):
                v_pred = dit(x_sigma.to(dtype), sigma.to(dtype), cross, padding_mask=pad)

        loss = ((v_pred.float() - v_target) ** 2).mean() / ACCUM
        loss.backward()
        step += 1
        if step % ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            opt.step(); opt.zero_grad()

        if step % SAVE_EVERY == 0:
            save_predictor(predictor, f"anima_i2l_step{step}.safetensors")

save_predictor(predictor, "anima_i2l_final.safetensors")
```

Notes:
- `save_predictor` embeds the full `I2LConfig` (including `encoder: "siglip2"`
  and `image_embed_dim`) in the file, so `i2l.py` / the UI rebuild the right
  network and demand the matching SigLIP2 path automatically.
- `embed_cross_context` is the same prompt→context path the preview uses, so
  training and inference conditioning are identical.

---

## 7. Memory & throughput

- **Adapted-layer set.** Start with `target_set="cross_attn"` (4 layers/block),
  then `"attn"`, then `"attn+mlp"`. Fewer layers ⇒ fewer queries/decoders ⇒ less
  predictor memory and a smaller LoRA.
- **Gradient checkpointing** on the DiT (above) is the biggest VRAM lever; it
  needs `dit.train()`.
- **Mixed precision:** backbone in bf16 under autocast, predictor master weights
  in fp32. The injection hook computes in fp32 so gradients stay clean.
- **Gradient accumulation** to reach the effective batch the paper uses.
- **Resolution:** train at 512² to start; move to 1024² once stable.
- **Rank** 16 is a good default; higher rank = more capacity and a larger LoRA.

---

## 8. Validate the result

The output is a normal i2L predictor — exercise it with the inference path:

```bash
python i2l.py caps   --checkpoint anima_i2l_final.safetensors
python i2l.py create --checkpoint anima_i2l_final.safetensors \
    --siglip /path/to/siglip2 --images style_ref.png --output test_style.safetensors
```

Then load `test_style.safetensors` in the editor and use **Live Preview** (real
Anima backend) to see whether the style transfers while the prompt controls
content. Use the **impact analyzer** to confirm the predicted LoRA actually
touches the blocks you adapted.

---

## 9. Asymmetric guidance & multi-reference (free at inference)

You do **not** train these specially:
- **Gray neutral LoRA** for asymmetric CFG (paper Sec. 3.4) is just
  `G_φ(gray image)` — `predict_gray_lora` / `--gray`. A predictor trained on
  diverse styles yields a near-neutral LoRA for a flat gray input automatically.
- **Multi-reference fusion** falls out of token concatenation (eq. 1); to make
  the predictor robust to it, *sample a variable number of references per tuple*
  during training (paper: "we sample multiple references when available …while
  retaining single-image examples to support one-shot inference").

---

## 10. Common pitfalls

| Symptom | Likely cause |
| ------- | ------------ |
| Outputs copy reference *content*, ignore the prompt | training data not content-disjoint (semantic leakage) — fix the dataset |
| `KeyError: predicted layer(s) not found` during injection | predictor `layer_spec` doesn't match this DiT (wrong `target_set`/prefix) |
| Style barely transfers | rank too low, too few adapted layers, or undertrained |
| `i2l.py create` demands a SigLIP2 path | checkpoint was trained with `encoder="siglip2"` (correct) — pass `--siglip` |
| VRAM blows up | enable gradient checkpointing; shrink `target_set`/resolution; accumulate |

---

## See also

- [`docs/I2LORA.md`](I2LORA.md) — using a trained (or demo) predictor.
- `core/i2lora/inject.py` — the differentiable injection used above.
- `examples/i2lora_train_step_test.py` — minimal, verified train step.
- Paper: *Compressing Image Style Training into a Single Model Forward*
  (arXiv:2606.13809).
