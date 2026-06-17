# Create a LoRA from images — i2L (image-to-LoRA)

This suite can **predict an Anima style LoRA directly from reference images in a
single forward pass**, instead of training one. It is a faithful, standalone
implementation of:

> Zhongjie Duan, Yingda Chen. *Compressing Image Style Training into a Single
> Model Forward.* arXiv:2606.13809 (ModelScope / Alibaba).

The paper's method, **i2L**, replaces the expensive per-style LoRA training loop
with a meta-model that is trained once and then, at test time, maps one or more
reference images straight to LoRA weights.

```
reference images ──▶ frozen image encoder ──▶ transformer w/ LoRA queries
                                                      │
                                       compressed decoding heads
                                                      │
                                                      ▼
                                        a standard Anima .safetensors LoRA
```

---

## ⚠️ Read this first — you need a trained predictor

i2L is an **inference** architecture. It needs a *trained* predictor checkpoint
(the meta-model `G_φ`). Two honest caveats:

1. **No i2L predictor has been publicly released for Anima.** The paper releases
   predictors for *other* backbones — Z-Image, FLUX.2-klein, and Hidream-O1 — in
   the DiffSynth-Studio key layout, which is not Anima's. Training an Anima
   predictor is exactly the heavy meta-training (≈7 days on 8×A100 in the paper)
   that this feature deliberately does **not** do.

2. To let you exercise the whole workflow regardless, the suite ships a
   **demo predictor** (`python i2l.py make-demo …`) with *random* weights. Its
   output is a structurally valid Anima LoRA — correct keys, ranks, and shapes;
   it loads in the editor, analyzer, preview, and ComfyUI — but it is **not
   trained**, so it does **not** stylise. Use it for plumbing/validation and to
   learn the workflow. Drop in a real trained checkpoint for real results.

In short: **the pipeline is complete and runnable; the missing piece is trained
predictor weights, which only meta-training (out of scope here) can produce.**

---

## How it maps onto Anima

A predicted layer's matrices become a standard LoRA pair (eq. 2,
`W' = W + α·B·A`):

| tensor | shape | role |
| ------ | ----- | ---- |
| `<key>.lora_down.weight` | `(rank, in_dim)`  | `A` (down side) |
| `<key>.lora_up.weight`   | `(out_dim, rank)` | `B` (up side) |
| `<key>.alpha`            | scalar            | `α` (default = rank → gain 1) |

`<key>` uses the ComfyUI-compatible Anima naming the editor already recognises,
e.g. `lora_unet_blocks_7_cross_attn_k_proj`. The real Anima 2B dimensions are
used (hidden 2048, cross-context 1024, MLP 8192, 28 blocks). Which layers are
adapted is configurable via `target_set`:

| target_set | layers per block |
| ---------- | ---------------- |
| `cross_attn` | cross-attn q/k/v/out |
| `attn` (default) | self- + cross-attn q/k/v/out |
| `attn+mlp` | attention + both MLP layers |

---

## The image encoder

The paper uses a **frozen SigLIP2** encoder and keeps *patch* tokens (style is
distributed across local texture, palette, composition). Two encoders ship here:

- **SigLIP2** (`encoder: "siglip2"`) — the real thing, via 🤗 `transformers`.
  Needs `transformers` + the SigLIP2 weights (set the *SigLIP2 model path*). A
  trained predictor will expect this.
- **fallback** (`encoder: "fallback"`) — a deterministic, weight-free patch
  embedder so the pipeline runs with nothing but `torch` + `pillow`. It is
  **not** SigLIP2 and carries no learned semantics; the demo predictor uses it.

`pillow` is required to read reference images (it's in the preview extras).

---

## CLI

```bash
# 1) Make a demo predictor (random weights — workflow only, not quality)
python i2l.py make-demo demo_predictor.safetensors

# 2) Predict a LoRA from one or more reference images (multiple = style fusion)
python i2l.py create \
    --checkpoint demo_predictor.safetensors \
    --images ref1.png ref2.jpg \
    --output style_from_images.safetensors \
    --multiplier 1.0 \
    --gray                       # also write the gray neutral LoRA

# 3) Probe what a checkpoint/environment can do
python i2l.py caps --checkpoint demo_predictor.safetensors
```

`make-demo` flags: `--target-set {cross_attn,attn,attn+mlp}`, `--blocks N`
(limit to the first N AnimaBlocks for a tinier model), `--rank K`.

`create` flags: `--siglip PATH` (SigLIP2 weights, if the checkpoint needs them),
`--alpha`, `--device {cpu,cuda}`.

---

## Web UI

Open the editor and use the **Create from Images (i2L)** panel (the `へ` step):

1. **Predictor checkpoint** — path to a trained (or demo) `.safetensors` predictor.
2. **Reference images** — one path per line; multiple images are fused into one
   style LoRA.
3. **Multiplier** — LoRA gain (folded into the up/`B` side).
4. **Asymmetric guidance** — also writes a *gray neutral* LoRA next to the
   output (`…​.gray.safetensors`). Per Sec. 3.4, use the reference LoRA on the
   positive CFG branch and the gray LoRA on the negative to strengthen style.
5. **SigLIP2 model path** (advanced) — only if the checkpoint needs it; the
   status pill tells you.
6. **Output path**, then **Create LoRA from images**.

The pill shows readiness (`ready · CPU`, `set checkpoint`, `SigLIP2 needed`).
The result records the encoder used and warns if it's the (untrained) fallback.

---

## REST API

```bash
# What can i2L do here? (checkpoint summary, encoder, readiness)
curl -X POST http://localhost:7860/api/i2lora/capabilities \
     -H 'Content-Type: application/json' \
     -d '{"checkpoint": "/path/to/predictor.safetensors"}'

# Create a LoRA from images
curl -X POST http://localhost:7860/api/i2lora/create \
     -H 'Content-Type: application/json' \
     -d '{
       "checkpoint": "/path/to/predictor.safetensors",
       "image_paths": ["/path/ref1.png", "/path/ref2.jpg"],
       "output_path": "/path/style_from_images.safetensors",
       "siglip_path": "",
       "multiplier": 1.0,
       "gray": true
     }'
```

---

## Files

```
core/i2lora/
├── __init__.py        public surface
├── layer_spec.py      LoRALayerSpec + Anima target layer sets (real 2B dims)
├── model.py           I2LConfig + I2LModel (encoder→transformer→queries→decoders)
├── encoder.py         SigLIP2 encoder + deterministic fallback + image IO
├── checkpoint.py      self-describing predictor save/load (config in metadata)
├── predict.py         images → Anima LoRA state_dict (+ gray neutral, fusion)
├── capabilities.py    readiness probe (mirrors core/preview/capabilities.py)
└── demo.py            tiny random demo predictor

i2l.py                          CLI (make-demo / create / caps)
examples/i2lora_smoke_test.py   CPU end-to-end test
```

---

## What's faithful, and what's not

**Faithful to the paper:** the predictor architecture (frozen image encoder →
single-stream transformer fusing image tokens with learnable per-row/per-column
LoRA queries → 2L compressed `D·C` decoding heads), the row/column query↔matrix
alignment, multi-reference fusion by token concatenation, the gray-image neutral
LoRA for asymmetric guidance, and emitting an explicit standard LoRA.

**Not included:** the meta-*training* loop (flow-matching through the frozen
backbone over MegaStyle-1M) and any pretrained predictor weights for Anima.
Without trained weights the predictions are not meaningful — that's the nature
of an inference-only integration of a method whose value lives in its trained
meta-model.
