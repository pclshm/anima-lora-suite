# Anima LoRA Editor

A standalone, GUI-based **selective LoRA editor** for [CircleStone Labs' Anima](https://huggingface.co/circlestone-labs/Anima).
Toggle individual AnimaBlock layers on/off, scale each one independently, and
save out a new `.safetensors` LoRA that any Anima-compatible inference tool
(ComfyUI, etc.) can load directly.

No GPU required for editing — the editor runs entirely on CPU and never loads
the DiT, VAE, or text encoder. (The optional [live preview](#live-preview--clownsharksampler-standalone)
*can* load them on a GPU to render real samples, but that's strictly opt-in.)

> This editor focuses on the *post-training* problem: tuning a LoRA without
> re-training it.

---

## Features

- 28 individual AnimaBlock toggles (0–27) with per-block strength sliders.
- Separate controls for the **LLMAdapter** (Qwen3 → DiT bridge) and *other*
  weights (embeddings, time, finals).
- 12 preset masks — *Late Only*, *Mid Only*, *Skip Early*, *Peak Impact*,
  *Style Focus*, *Face Priority*, *Evens / Odds*, *All Off*, etc.
- Built-in **impact analyzer** — computes a Frobenius-norm score per block
  and colour-codes the checkboxes so you see at a glance which blocks
  carry the LoRA's weight.
- Architecture detection — warns you if the file you loaded is FLUX, SDXL,
  Z-Image, Qwen-Image, etc. instead of Anima.
- Original LoRA is *never* modified; output is always written to a new path.
- Original metadata preserved, with extra keys describing what was edited.
- **Live preview** — generate sample images right in the browser with a
  standalone, server-free re-implementation of RES4LYF's `ClownsharKSampler`
  (the RES / Refined Exponential Solver family). The preview reflects your
  *current* edits, so you can see a block toggle's effect before saving.
  **GPU-only** — needs a CUDA build of torch and external Anima model files;
  the editor itself runs fine without either.

---

## Install (from source)

### Prerequisites
- Python 3.10+ (3.12 recommended)
- A recent pip

### Windows
```bat
git clone <this repo>
cd anima-lora-editor
setup_env.bat
venv\Scripts\activate
python app.py
```

### Linux / macOS
```bash
git clone <this repo>
cd anima-lora-editor
chmod +x setup_env.sh
./setup_env.sh
source venv/bin/activate
python app.py
```

The setup script creates a local `venv/`, installs `flask`, `safetensors`,
`torch`, and friends, and leaves you ready for **editing**. By default pip
pulls the CPU build of torch — that's all the editor needs.

To enable **real live preview** you also need a **CUDA GPU**. Install a
CUDA build of `torch` into the venv *yourself* (the right wheel depends on
your driver / CUDA toolkit version) from
[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/),
e.g.:

```bash
# example for CUDA 12.8 — pick the index URL that matches your driver
pip install --force-reinstall --no-cache-dir torch \
    --index-url https://download.pytorch.org/whl/cu128
```

Then run `setup_preview.{bat,sh}` to add the rest of the generation deps
(transformers / accelerate / einops / sentencepiece). See
[Live preview](#live-preview--clownsharksampler-standalone).

Once launched, your browser will open to `http://localhost:7860/`. If it
doesn't, navigate there manually.

---

## Usage

1. **Paste a path** to an Anima LoRA `.safetensors` file in the *Input* panel
   and click **Inspect**. The analyzer scans the keys, classifies them, and
   colours the 28 block cells by Frobenius-norm impact.
2. **Pick a preset** or tune blocks individually. Each block has:
   - A checkbox (kept / dropped from the output)
   - A strength slider (−2.0 to +2.0; default 1.0)
   - An impact percentage (relative to the strongest block in this LoRA)
3. Toggle the **LLMAdapter** and **other weights** independently — useful if
   you want to drop the text-conditioning portion of the LoRA while keeping
   the visual side, or vice versa.
4. **Preview** (optional) — open the *Live Preview* panel, type a prompt, and
   click **Generate sample**. The image is produced by a standalone RES sampler
   (see below) using your current edit, so you can compare block configs by
   regenerating. No file is written; it previews the in-memory edit.
5. **Save** to a new path. The output is a fully-valid `.safetensors` file
   that ComfyUI / Anima inference will load like any other LoRA.

### Strength scaling, briefly
LoRAs decompose as `W = up @ down · (alpha / rank)`. Multiplying *either*
half by k scales the contribution by k. To avoid double-counting, this
editor scales only the `lora_up` / `lora_B` side (and any scalar tensors),
leaving `lora_down` / `lora_A` and `.alpha` untouched. The result is that
a per-block strength of `2.0` doubles that block's contribution — not
quadruples it.

---

## Live preview — `ClownsharKSampler`, standalone

The preview is **fully standalone**: no ComfyUI server, no external process. It
re-implements the RES (Refined Exponential Solver) sampler family that powers
RES4LYF's [`ClownsharKSampler`](https://github.com/ClownsharkBatwing/RES4LYF) as
plain, ComfyUI-independent `torch` (`core/preview/sampler.py`). The φ-function
coefficients and the `res_2m` step follow RES4LYF's `beta/phi_functions.py` and
`rk_method_beta.py`. Exposed knobs mirror the node: **sampler** (`res_2m`,
`res_2s`, `euler`, `euler_ancestral`), **scheduler**, **steps**, **cfg**, **eta**
(SDE noise), **seed**, and size.

Live Preview is **GPU-only** — there is no CPU stand-in:

| Mode | When | What you get |
| ---- | ---- | ------------ |
| **Real Anima** | CUDA GPU + model paths set | Actual Anima images: loads the DiT + WanVAE + Qwen3 text encoder, merges your edited LoRA in memory, samples with the RES solver, and VAE-decodes. |
| **Unavailable** | no CUDA, or model paths unset | A preview request returns a clear error explaining what's missing. Inspect/edit still work on CPU; only Live Preview needs the GPU stack. |

### Enabling real generation

The Anima model code (DiT, WanVAE, LLM adapter, LoRA merge machinery) ships
**vendored inside this repo** under `core/anima/` — no external checkout
required. You provide the CUDA torch wheel, the rest is one script.

**1. Install a CUDA build of `torch` into your venv.** The right wheel
depends on your driver / CUDA toolkit version; pick one from
[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).
Example:

```bash
pip install --force-reinstall --no-cache-dir torch \
    --index-url https://download.pytorch.org/whl/cu128
```

**2. Install the generation extras.** This step does *not* touch torch — it
just verifies CUDA torch is present and installs transformers / accelerate /
einops / sentencepiece on top:

```bat
setup_preview.bat        REM Windows
```
```bash
./setup_preview.sh       # Linux/macOS
```

Then open **Live Preview → Model paths** and point it at:

- **DiT checkpoint** — your Anima model `.safetensors`
- **VAE** — the Qwen-Image / WanVAE
- **Text encoder** — Qwen3-0.6B (the file or directory the loader uses)

The backend pill flips from `GPU required` to `Anima · <GPU>` once CUDA and
all three model paths resolve. Paths and preview settings are remembered in
your browser.

> Anima is a **rectified-flow** model (the DiT predicts a velocity). The RES
> solver works on a denoiser, related by `denoised = x − σ·v`; with that
> substitution the flow ODE is exactly the one the solver integrates, and the
> `euler` option reproduces the canonical Euler step — a built-in sanity check.

---

## Create a LoRA from images — i2L (image-to-LoRA)

Beyond *editing* LoRAs, the suite can **predict an Anima style LoRA directly
from reference images in a single forward pass** — a faithful, standalone
implementation of **i2L** from
[*Compressing Image Style Training into a Single Model Forward*](https://arxiv.org/abs/2606.13809)
(Duan & Chen, arXiv:2606.13809). A frozen image encoder + a transformer with
learnable LoRA queries + compressed decoding heads map one or more reference
images straight to LoRA weights — no per-style training loop.

> **⚠️ Needs a trained predictor checkpoint.** i2L is an *inference*
> architecture. No i2L predictor has been publicly released for Anima (the paper
> ships ones for Z-Image / FLUX.2-klein / Hidream-O1, in a different key
> layout). A bundled **demo predictor** (`python i2l.py make-demo …`) has
> *random* weights: it produces a structurally valid Anima LoRA so you can run
> the whole workflow, but it is **not trained** and will not stylise. Point it
> at a real trained checkpoint for real results.

```bash
# Make a random demo predictor, then create a LoRA from two reference images
python i2l.py make-demo demo_predictor.safetensors
python i2l.py create --checkpoint demo_predictor.safetensors \
    --images ref1.png ref2.jpg --output style_from_images.safetensors --gray
```

Or use the **Create from Images (i2L)** panel in the web UI (the `へ` step).
Multiple references fuse into one style LoRA; an optional *gray neutral* LoRA
supports the paper's asymmetric classifier-free guidance. The result is a
standard `.safetensors` the editor, analyzer, live preview, and ComfyUI all
load. Reference-image input needs `pillow`; the real SigLIP2 encoder needs
`transformers` + SigLIP2 weights (both in the preview extras). Full details:
[`docs/I2LORA.md`](docs/I2LORA.md).

---

## The Anima architecture, briefly

Anima is a 2 B parameter DiT derived from
[NVIDIA Cosmos-Predict2-2B-Text2Image](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Text2Image),
with 28 transformer blocks (`AnimaBlock` 0–27), each combining self-attention,
cross-attention, and an MLP. A dedicated `LLMAdapter` module bridges the
Qwen3-0.6B text encoder to the DiT — it's effectively a small transformer
on top of the text embeddings. The VAE is the Qwen-Image VAE.

LoRAs touch any combination of these:
- the 28 AnimaBlocks (recognised by `lora_unet_blocks_<N>_…` or `blocks.<N>.…`)
- the LLMAdapter (recognised by `llm_adapter` in the key)
- *other* tensors — embeddings, time, finals, possibly the patch embed

This editor classifies every key into one of those three buckets and
lets you keep or drop each independently.

---

## Customising the look — *the waifu background*

The UI comes with a beautiful default: a gradient mesh of sakura pink,
lavender, and gold, with twelve animated cherry-blossom petals drifting
across the screen.

To use **your own anime background image** (a wallpaper, a generated
image from Anima, anything you like):

1. Drop your image at one of:
   - `static/waifu-bg.png`
   - `static/waifu-bg.jpg`
   - `static/waifu-bg.webp`
2. Refresh the browser.

The CSS will pick it up automatically (it tries those three filenames in
order). The vignette and grain overlays will sit on top so foreground
text stays readable on any image.

If you want to tweak the colours, every accent lives in `:root` at the top
of `static/style.css` — sakura pink, gold, lavender, the impact bands.
Re-paint the whole UI by changing about ten variables.

---

## CLI options

```bash
python app.py --help
```

| Flag           | Default       | Notes                                  |
| -------------- | ------------- | -------------------------------------- |
| `--host`       | `127.0.0.1`   | Set to `0.0.0.0` for LAN access        |
| `--port`       | `7860`        | Standard Gradio-ish port               |
| `--no-browser` |               | Don't auto-open browser                |
| `--debug`      |               | Flask debug mode + verbose logs        |

---

## REST API

If you'd rather drive the editor from a script:

```bash
# Inspect
curl -X POST http://localhost:7860/api/inspect \
     -H 'Content-Type: application/json' \
     -d '{"lora_path": "/path/to/anima_lora.safetensors"}'

# Edit
curl -X POST http://localhost:7860/api/edit \
     -H 'Content-Type: application/json' \
     -d '{
       "lora_path": "/path/to/anima_lora.safetensors",
       "output_path": "/path/to/anima_lora_late_only.safetensors",
       "config": {
         "enabled_blocks": [19,20,21,22,23,24,25,26,27],
         "block_strengths": {"23": 1.5, "24": 1.5},
         "llm_adapter_enabled": false,
         "other_enabled": true,
         "global_strength": 1.0
       }
     }'

# Live preview (returns a PNG data-URI + metadata)
curl -X POST http://localhost:7860/api/preview \
     -H 'Content-Type: application/json' \
     -d '{
       "prompt": "a serene anime portrait, soft lighting",
       "sampler": "res_2m", "scheduler": "karras",
       "steps": 20, "cfg": 5.5, "eta": 0.5, "seed": 0,
       "width": 512, "height": 512,
       "lora_path": "/path/to/anima_lora.safetensors",
       "config": { "enabled_blocks": [19,20,21,22,23,24,25,26,27] },
       "model_paths": { "dit": "", "vae": "", "text_encoder": "" }
     }'

# What can the preview do here? (CUDA, active backend, sampler list)
curl http://localhost:7860/api/preview/capabilities
```

---

## File layout

```
anima-lora-editor/
├── app.py                  Flask backend (+ /api/preview, /api/i2lora endpoints)
├── i2l.py                  i2L CLI (make-demo / create / caps)
├── core/
│   ├── __init__.py
│   ├── detect.py           Key-pattern detection + architecture classifier
│   ├── editor.py           EditConfig + edit_lora()
│   ├── analyzer.py         Per-block Frobenius-norm impact scores
│   ├── presets.py          12 preset block masks
│   ├── anima/              Vendored Anima model code (DiT, WanVAE, LoRA merge)
│   ├── i2lora/             Image-to-LoRA (i2L): predict a LoRA from images
│   │   ├── layer_spec.py     Anima target layers (real 2B dims) + LoRA keys
│   │   ├── model.py          I2LModel: encoder→transformer→queries→decoders
│   │   ├── encoder.py        SigLIP2 encoder + weight-free fallback + image IO
│   │   ├── predict.py        images → Anima LoRA (+ gray neutral, fusion)
│   │   ├── checkpoint.py     self-describing predictor save/load
│   │   ├── capabilities.py   readiness probe
│   │   └── demo.py           tiny random demo predictor
│   └── preview/            Standalone live-preview package
│       ├── sampler.py        Vendored ClownsharKSampler RES solver (no ComfyUI)
│       ├── schedulers.py     karras / exponential / linear sigma schedules
│       ├── backends.py       Real Anima GPU backend (uses core/anima/)
│       ├── pipeline.py       edit -> encode -> sample -> decode -> PNG
│       ├── capabilities.py   What the preview can do (CUDA? backend?)
│       └── pngio.py          Minimal stdlib PNG encoder (no Pillow dep)
├── static/
│   ├── index.html          UI markup (incl. Live Preview panel)
│   ├── style.css           Anime-themed CSS (sakura petals, frosted glass)
│   ├── script.js           Frontend logic
│   └── waifu-bg.{png,jpg}  Optional — drop your own here
├── examples/
│   ├── smoke_test.py            Editor round-trip checks
│   ├── preview_smoke_test.py    Sampler + pipeline checks (CPU)
│   └── i2lora_smoke_test.py     Image-to-LoRA end-to-end checks (CPU)
├── requirements.txt
├── requirements-preview.txt     Optional generation extras (GPU)
├── setup_env.{bat,sh}           Install venv + base deps
├── setup_preview.{bat,sh}       Install generation extras (Live Preview, requires user-installed CUDA torch)
├── start_anima_editor.{bat,sh}      Launcher (runs editor under local venv)
└── README.md
```

---

## Caveats

- **Key format detection** is heuristic. The editor handles the kohya/
  sd-scripts format (`lora_unet_blocks_<N>_…`) and the diffusers/PEFT
  format (`transformer.blocks.<N>.…`). If you find a LoRA where blocks
  aren't recognised, run `python -c "from safetensors.torch import
  load_file; print(list(load_file('your.safetensors').keys())[:30])"`
  and file an issue with a sample of the keys.
- **Preview is GPU-only.** Without a CUDA build of torch *and* model paths
  set, Live Preview reports unavailable and returns a clear error if you try
  to generate. The editor itself never needs a GPU. To enable preview,
  install a CUDA torch wheel yourself
  (https://pytorch.org/get-started/locally/), run `setup_preview`, and set
  the model paths — or load the saved LoRA in ComfyUI / `anima_gen.py`.

---

## License

Apache-2.0.
