"""
Anima LoRA Editor — Flask backend.

A tiny REST API around core/ plus a static frontend. Run with:

    python app.py

then open http://localhost:7860 in a browser.

Endpoints:
    GET  /                      Serve the UI
    GET  /static/<path>         Static assets (css, js, images)
    GET  /api/health            { ok: true, num_blocks: 28 }
    GET  /api/presets           List of preset names and block sets
    POST /api/inspect           { lora_path } -> summary + per-block impact
    POST /api/validate          { lora_path, keyword, config } -> per-block keyword attribution
    POST /api/edit              { lora_path, output_path, config } -> info
    GET  /api/preview/capabilities   What the live preview can do (CUDA, backend)
    POST /api/preview           { prompt, ..., lora_path, config } -> PNG data-URI
    GET  /api/i2lora/capabilities    What image-to-LoRA can do (checkpoint, encoder)
    POST /api/i2lora/create     { checkpoint, image_paths, output_path } -> info
"""

import os
import json
import logging
import argparse
import webbrowser
from pathlib import Path
from threading import Timer

from flask import Flask, request, jsonify, send_from_directory, abort

from core import (
    ANIMA_NUM_BLOCKS,
    ANIMA_PRESETS,
    load_lora_state_dict,
    detect_architecture,
    analyze_lora,
    edit_lora,
    save_lora_state_dict,
    merge_loras,
    compress_state_dict,
    state_dict_nbytes,
    dtype_histogram,
    dominant_float_dtype,
    size_profile,
)
from core.detect import summarize_keys
from core.editor import EditConfig
from core.attribution import attribute_activation, attribute_cross_attn, cpu_cross_context

# ----------------------------------------------------------------------------

HERE = Path(__file__).parent.resolve()
STATIC_DIR = HERE / "static"

# Writable user content (uploaded/generated backgrounds) lives in a stable
# per-user location, overridable via ANIMA_EDITOR_DATA.
DATA_DIR = Path(os.environ.get("ANIMA_EDITOR_DATA") or (Path.home() / ".anima-lora-editor"))

app = Flask(
    __name__,
    static_folder=str(STATIC_DIR),
    static_url_path="/static",
)
app.config["JSON_SORT_KEYS"] = False
log = logging.getLogger("anima-editor")


# ----------------------------------------------------------------------------
# Static routes
# ----------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(str(STATIC_DIR), "index.html")


@app.route("/favicon.ico")
def favicon():
    fav = STATIC_DIR / "favicon.ico"
    if fav.exists():
        return send_from_directory(str(STATIC_DIR), "favicon.ico")
    return ("", 204)


# ----------------------------------------------------------------------------
# JSON API
# ----------------------------------------------------------------------------

@app.route("/api/health")
def api_health():
    return jsonify(ok=True, num_blocks=ANIMA_NUM_BLOCKS, version="0.1.0")


@app.route("/api/presets")
def api_presets():
    out = {}
    for name, blocks in ANIMA_PRESETS.items():
        if blocks is None:
            out[name] = None  # 'Custom' -> let UI handle
        else:
            out[name] = sorted(blocks)
    return jsonify(presets=out, num_blocks=ANIMA_NUM_BLOCKS)


@app.route("/api/inspect", methods=["POST"])
def api_inspect():
    """Load a LoRA from disk and return summary + impact scores."""
    body = request.get_json(force=True, silent=True) or {}
    path = body.get("lora_path", "").strip()
    if not path:
        return jsonify(error="lora_path is required"), 400
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return jsonify(error=f"file not found: {path}"), 404

    try:
        sd = load_lora_state_dict(path)
    except Exception as e:
        return jsonify(error=f"failed to load LoRA: {e}"), 400

    keys = list(sd.keys())
    arch = detect_architecture(keys)
    summary = summarize_keys(keys)
    impact = analyze_lora(sd)

    # Size / precision so the UI can show current state and project savings.
    try:
        file_size = os.path.getsize(path)
    except OSError:
        file_size = state_dict_nbytes(sd)
    payload_bytes = state_dict_nbytes(sd)

    return jsonify(
        path=path,
        detected_architecture=arch,
        is_anima=(arch == "ANIMA"),
        summary=summary,
        impact=impact,
        file_size_bytes=file_size,
        payload_bytes=payload_bytes,
        dtype=dominant_float_dtype(sd),
        dtype_histogram=dtype_histogram(sd),
        size_profile=size_profile(sd),
    )


def _edit_config_from_payload(cfg: dict) -> EditConfig:
    """Build an EditConfig from the JSON the UI sends (shared shape with /api/edit)."""
    enabled = set(int(b) for b in cfg.get("enabled_blocks", list(range(ANIMA_NUM_BLOCKS))))
    strengths = {int(k): float(v) for k, v in (cfg.get("block_strengths") or {}).items()}
    return EditConfig(
        enabled_blocks=enabled,
        block_strengths=strengths,
        llm_adapter_enabled=bool(cfg.get("llm_adapter_enabled", True)),
        llm_adapter_strength=float(cfg.get("llm_adapter_strength", 1.0)),
        other_enabled=bool(cfg.get("other_enabled", True)),
        other_strength=float(cfg.get("other_strength", 1.0)),
        global_strength=float(cfg.get("global_strength", 1.0)),
    )


def _compress_opts_from_payload(body: dict):
    """Pull compression options off the request. Returns (dtype, rank, energy) or
    (None, None, None) when nothing was requested. Raises ValueError on bad input."""
    c = body.get("compress") or {}
    dtype = (c.get("dtype") or "").strip() or None
    if dtype and dtype.lower() in ("keep", "same", "original", "none"):
        dtype = None
    rank = c.get("svd_rank")
    energy = c.get("svd_energy")
    rank = int(rank) if rank not in (None, "", 0, "0") else None
    if rank is not None and rank < 1:
        raise ValueError("svd_rank must be >= 1")
    energy = float(energy) if energy not in (None, "") else None
    if energy is not None and not (0 < energy < 1):
        raise ValueError("svd_energy must be between 0 and 1 (exclusive)")
    if dtype is None and rank is None and energy is None:
        return None, None, None
    return dtype, rank, energy


def _apply_compression(state_dict, body, metadata):
    """Apply requested compression to a (merged) state_dict in place of saving.

    Returns (new_sd, comp_info_or_None) and records params in ``metadata``.
    Raises ValueError if the options are invalid or unsupported by this torch."""
    dtype, rank, energy = _compress_opts_from_payload(body)
    if dtype is None and rank is None and energy is None:
        return state_dict, None
    new_sd, comp = compress_state_dict(
        state_dict, dtype=dtype, svd_rank=rank, svd_energy=energy
    )
    if dtype:
        metadata["anima_lora_editor.compress_dtype"] = dtype
    if rank is not None:
        metadata["anima_lora_editor.compress_svd_rank"] = str(rank)
    if energy is not None:
        metadata["anima_lora_editor.compress_svd_energy"] = str(energy)
    return new_sd, comp


@app.route("/api/validate", methods=["POST"])
def api_validate():
    """Keyword attribution — *which blocks does this keyword light up?*

    Pushes the keyword through the model (with the current edit applied) and
    returns a per-block score shaped like /api/inspect's ``impact``. Tries the
    faithful activation-delta on the GPU backend, falls back to a cheaper
    cross-attention projection (text-pathway only), and finally to the static
    prompt-free score if no Anima model is available — the chosen ``method`` and
    a ``note`` say which you're seeing.
    """
    body = request.get_json(force=True, silent=True) or {}
    path = os.path.expanduser((body.get("lora_path") or "").strip())
    keyword = (body.get("keyword") or "").strip()
    if not path:
        return jsonify(error="lora_path is required"), 400
    if not os.path.exists(path):
        return jsonify(error=f"file not found: {path}"), 404
    if not keyword:
        return jsonify(error="keyword is required"), 400

    try:
        sd = load_lora_state_dict(path)
    except Exception as e:
        return jsonify(error=f"failed to load LoRA: {e}"), 400

    try:
        edited_sd, _ = edit_lora(sd, _edit_config_from_payload(body.get("config") or {}))
    except (TypeError, ValueError) as e:
        return jsonify(error=f"bad config: {e}"), 400
    if not edited_sd:
        return jsonify(error="nothing to validate — all blocks disabled?"), 400

    model_paths = body.get("model_paths") or {}
    method, note, impact = _attribute_keyword(keyword, edited_sd, model_paths, sd)
    return jsonify(method=method, note=note, keyword=keyword, impact=impact)


def _attribute_keyword(keyword, edited_sd, model_paths, source_sd):
    """Pick the best available attribution tier. Returns (method, note, impact)."""
    from core.preview.pipeline import try_get_backend

    backend = try_get_backend(model_paths)
    if backend is not None:
        try:
            impact = attribute_activation(backend, keyword, edited_sd)
            return ("activation",
                    "Faithful: keyword run through the DiT; per-block change in "
                    "residual contribution with the LoRA applied vs. removed.",
                    impact)
        except Exception:
            log.exception("activation attribution failed; trying cross-attn")
            try:
                # Encode on the current (unmutated) weights — no lossy unmerge.
                import torch
                with torch.no_grad():
                    E = backend.encode(keyword, "")["pos"]
                impact = attribute_cross_attn(edited_sd, E)
                return ("cross_attn",
                        "Approximate: cross-attention projection only (text "
                        "pathway; self-attn/MLP not counted) — the full forward "
                        "pass was unavailable.",
                        impact)
            except Exception:
                log.exception("cross-attn (gpu) attribution failed; using static")

    else:
        # No GPU backend — try the CPU cross-context encoder (RAM-heavy load, but
        # no sampling, so the projection itself is cheap).
        try:
            enc = cpu_cross_context(model_paths)
            E = enc.cross_context(keyword)
            impact = attribute_cross_attn(edited_sd, E)
            return ("cross_attn_cpu",
                    "Approximate (CPU): cross-attention projection only — no GPU "
                    "backend, so self-attn/MLP effects aren't measured.",
                    impact)
        except Exception:
            log.exception("cpu cross-attn attribution failed; using static")

    return ("static",
            "Keyword attribution unavailable (no Anima model loaded) — showing "
            "the prompt-free impact score instead. Set the model paths under "
            "Live Preview to enable keyword attribution.",
            analyze_lora(source_sd))


@app.route("/api/preview/capabilities", methods=["GET", "POST"])
def api_preview_capabilities():
    """Report what the preview can do (CUDA? real Anima backend? samplers)."""
    from core.preview import preview_capabilities
    body = request.get_json(force=True, silent=True) or {}
    return jsonify(preview_capabilities(body.get("model_paths")))


@app.route("/api/preview", methods=["POST"])
def api_preview():
    """Generate a sample image from the *current* edit using ClownsharKSampler.

    Body mirrors the editor UI: prompt/seed/steps/cfg/sampler/eta/size, the
    source ``lora_path`` + ``config`` (same shape as /api/edit), and optional
    ``model_paths`` for the real backend. Returns a PNG data-URI + metadata.
    Requires a CUDA GPU + Anima weights; returns an error if they're unavailable
    (there is no CPU fallback).
    """
    from core.preview import PreviewConfig, generate_preview

    body = request.get_json(force=True, silent=True) or {}
    lora_path = os.path.expanduser((body.get("lora_path") or "").strip())
    if lora_path and not os.path.exists(lora_path):
        return jsonify(error=f"LoRA not found: {lora_path}"), 404

    # Multi-layer stack: combine several LoRAs in the preview. Each layer carries
    # its own edit config; expand ~paths and verify they exist up front.
    layers = []
    for spec in (body.get("loras") or []):
        p = os.path.expanduser((spec.get("lora_path") or "").strip())
        if not p:
            continue
        if not os.path.exists(p):
            return jsonify(error=f"LoRA not found: {p}"), 404
        layers.append({"lora_path": p, "edit": spec.get("config") or spec.get("edit") or {}})

    try:
        cfg = PreviewConfig(
            prompt=str(body.get("prompt", "")),
            negative=str(body.get("negative", "")),
            steps=int(body.get("steps", 20)),
            cfg=float(body.get("cfg", 5.5)),
            seed=int(body.get("seed", 0)),
            sampler=str(body.get("sampler", "res_2m")),
            scheduler=str(body.get("scheduler", "karras")),
            eta=float(body.get("eta", 0.5)),
            width=int(body.get("width", 512)),
            height=int(body.get("height", 512)),
            upscale=float(body.get("upscale", 1.0)),
            hires_steps=int(body.get("hires_steps", 12)),
            hires_denoise=float(body.get("hires_denoise", 0.5)),
            lora_path=lora_path,
            edit=body.get("config") or {},
            loras=layers,
            model_paths=body.get("model_paths") or {},
        )
    except (TypeError, ValueError) as e:
        return jsonify(error=f"bad preview config: {e}"), 400

    try:
        result = generate_preview(cfg)
    except Exception as e:  # surface model-load / sampling errors to the UI
        log.exception("preview generation failed")
        return jsonify(error=f"generation failed: {e}"), 500

    return jsonify(image=result.data_uri(), meta=result.meta)


def _edit_merge_layers(loras, out_path, body):
    """Edit each LoRA layer, merge the stack into one state_dict, and save it.

    Mirrors /api/edit's single-path response shape (``info``) so the UI renders
    it identically, with a couple of extra merge fields.
    """
    if not out_path:
        return jsonify(error="output_path is required"), 400

    edited, sources, arches, in_paths = [], 0, [], []
    for spec in loras:
        p = os.path.expanduser((spec.get("lora_path") or "").strip())
        if not p:
            continue
        if not os.path.exists(p):
            return jsonify(error=f"input file not found: {p}"), 404
        if os.path.abspath(p) == os.path.abspath(out_path):
            return jsonify(error="output_path must differ from every input"), 400
        try:
            sd = load_lora_state_dict(p)
        except Exception as e:
            return jsonify(error=f"failed to load LoRA {p}: {e}"), 400
        edit_cfg = _edit_config_from_payload(spec.get("config") or spec.get("edit") or {})
        new_sd, _ = edit_lora(sd, edit_cfg)
        if new_sd:
            edited.append(new_sd)
            sources += 1
            arches.append(detect_architecture(list(sd.keys())))
            in_paths.append(p)

    if not edited:
        return jsonify(error="no layers to merge — every layer empty or all blocks disabled?"), 400

    merged, minfo = merge_loras(edited)
    if not merged:
        return jsonify(error="nothing kept after merge"), 400

    arch = arches[0] if len(set(arches)) == 1 else "MIXED"
    metadata = {
        "anima_lora_editor": "1",
        "anima_lora_editor.merged_layers": str(sources),
        "anima_lora_editor.source_paths": " | ".join(in_paths),
        "anima_lora_editor.source_architecture": arch,
    }
    try:
        merged, comp = _apply_compression(merged, body, metadata)
    except ValueError as e:
        return jsonify(error=f"bad compression options: {e}"), 400
    try:
        save_lora_state_dict(merged, out_path, metadata=metadata)
    except Exception as e:
        return jsonify(error=f"failed to save: {e}"), 500

    info = {
        "merged_layers": sources,
        "input_paths": in_paths,
        "output_path": out_path,
        "output_tensor_count": len(merged),
        "modules_total": minfo["modules_total"],
        "modules_concatenated": minfo["modules_concatenated"],
        "modules_passthrough": minfo["modules_total"] - minfo["modules_concatenated"],
        "collisions": minfo["collisions"],
        "detected_architecture": arch,
        "saved": True,
    }
    if comp:
        info["compression"] = comp
    return jsonify(info=info)


@app.route("/api/edit", methods=["POST"])
def api_edit():
    """Apply an EditConfig and write a new safetensors file."""
    body = request.get_json(force=True, silent=True) or {}
    in_path = os.path.expanduser((body.get("lora_path") or "").strip())
    out_path = os.path.expanduser((body.get("output_path") or "").strip())
    cfg = body.get("config") or {}

    # Multi-layer save: edit each layer, then merge them all into one file.
    if body.get("loras"):
        return _edit_merge_layers(body.get("loras"), out_path, body)

    if not in_path or not os.path.exists(in_path):
        return jsonify(error=f"input file not found: {in_path}"), 404
    if not out_path:
        return jsonify(error="output_path is required"), 400
    if os.path.abspath(in_path) == os.path.abspath(out_path):
        return jsonify(error="output_path must differ from input"), 400

    # Build EditConfig from JSON payload
    try:
        enabled_blocks = set(int(b) for b in cfg.get("enabled_blocks", list(range(ANIMA_NUM_BLOCKS))))
        block_strengths = {int(k): float(v) for k, v in (cfg.get("block_strengths") or {}).items()}
        edit_cfg = EditConfig(
            enabled_blocks=enabled_blocks,
            block_strengths=block_strengths,
            llm_adapter_enabled=bool(cfg.get("llm_adapter_enabled", True)),
            llm_adapter_strength=float(cfg.get("llm_adapter_strength", 1.0)),
            other_enabled=bool(cfg.get("other_enabled", True)),
            other_strength=float(cfg.get("other_strength", 1.0)),
            global_strength=float(cfg.get("global_strength", 1.0)),
        )
    except (TypeError, ValueError) as e:
        return jsonify(error=f"bad config: {e}"), 400

    try:
        sd = load_lora_state_dict(in_path)
    except Exception as e:
        return jsonify(error=f"failed to load LoRA: {e}"), 400

    arch = detect_architecture(list(sd.keys()))
    new_sd, info = edit_lora(sd, edit_cfg)
    info["detected_architecture"] = arch
    info["input_path"] = in_path
    info["output_path"] = out_path

    if not new_sd:
        return jsonify(error="nothing kept — all blocks disabled?", info=info), 400

    metadata = {
        "anima_lora_editor": "1",
        "anima_lora_editor.global_strength": str(edit_cfg.global_strength),
        "anima_lora_editor.enabled_blocks": ",".join(str(b) for b in sorted(edit_cfg.enabled_blocks)),
        "anima_lora_editor.llm_adapter_enabled": str(edit_cfg.llm_adapter_enabled).lower(),
        "anima_lora_editor.other_enabled": str(edit_cfg.other_enabled).lower(),
        "anima_lora_editor.source_architecture": arch,
    }

    try:
        new_sd, comp = _apply_compression(new_sd, body, metadata)
    except ValueError as e:
        return jsonify(error=f"bad compression options: {e}", info=info), 400
    if comp:
        info["compression"] = comp

    try:
        save_lora_state_dict(new_sd, out_path, metadata=metadata)
    except Exception as e:
        return jsonify(error=f"failed to save: {e}", info=info), 500

    info["saved"] = True
    return jsonify(info=info)


# ----------------------------------------------------------------------------
# i2L — create a LoRA from reference images (image-to-LoRA, arXiv:2606.13809)
# ----------------------------------------------------------------------------

@app.route("/api/i2lora/capabilities", methods=["GET", "POST"])
def api_i2lora_capabilities():
    """Report what image-to-LoRA can do (checkpoint present? encoder? ready?)."""
    from core.i2lora import i2lora_capabilities
    body = request.get_json(force=True, silent=True) or {}
    ckpt = (body.get("checkpoint") or request.args.get("checkpoint") or "").strip()
    siglip = (body.get("siglip_path") or request.args.get("siglip_path") or "").strip()
    return jsonify(i2lora_capabilities(checkpoint=ckpt, siglip_path=siglip))


@app.route("/api/i2lora/create", methods=["POST"])
def api_i2lora_create():
    """Predict an Anima LoRA from reference images and write it to disk.

    Body: ``{ checkpoint, image_paths: [..], output_path, siglip_path?,
              multiplier?, alpha?, gray? }``. Reference images and the predictor
      checkpoint are server-side paths (same convention as /api/edit's
      lora_path). Returns an ``info`` dict; if ``gray`` is set, also writes the
      neutral gray LoRA alongside ``output_path`` and reports its path.
    """
    from core.i2lora import predict_lora_from_paths, make_metadata

    body = request.get_json(force=True, silent=True) or {}
    checkpoint = os.path.expanduser((body.get("checkpoint") or "").strip())
    out_path = os.path.expanduser((body.get("output_path") or "").strip())
    siglip = os.path.expanduser((body.get("siglip_path") or "").strip()) or None
    image_paths = [os.path.expanduser(p.strip())
                   for p in (body.get("image_paths") or []) if p and p.strip()]

    if not checkpoint or not os.path.exists(checkpoint):
        return jsonify(error=f"predictor checkpoint not found: {checkpoint}"), 404
    if not image_paths:
        return jsonify(error="at least one reference image is required"), 400
    missing = [p for p in image_paths if not os.path.exists(p)]
    if missing:
        return jsonify(error="reference image(s) not found: " + ", ".join(missing)), 404
    if not out_path:
        return jsonify(error="output_path is required"), 400

    try:
        multiplier = float(body.get("multiplier", 1.0))
        alpha = body.get("alpha")
        alpha = float(alpha) if alpha not in (None, "") else None
        gray = bool(body.get("gray", False))
    except (TypeError, ValueError) as e:
        return jsonify(error=f"bad options: {e}"), 400

    try:
        main, gray_res = predict_lora_from_paths(
            checkpoint=checkpoint, image_paths=image_paths, siglip_path=siglip,
            multiplier=multiplier, alpha=alpha, also_gray=gray,
        )
    except Exception as e:
        log.exception("i2lora create failed")
        return jsonify(error=f"prediction failed: {e}"), 500

    try:
        save_lora_state_dict(main.state_dict, out_path,
                             metadata=make_metadata(main.info, checkpoint, image_paths))
        info = dict(main.info)
        info["output_path"] = out_path
        info["saved"] = True
        if gray_res is not None:
            root, ext = os.path.splitext(out_path)
            gp = f"{root}.gray{ext or '.safetensors'}"
            save_lora_state_dict(gray_res.state_dict, gp,
                                 metadata=make_metadata(gray_res.info, checkpoint, []))
            info["gray_output_path"] = gp
    except Exception as e:
        return jsonify(error=f"failed to save: {e}"), 500

    return jsonify(info=info)


# ----------------------------------------------------------------------------
# Theme backgrounds — per-theme full-screen image, stored on disk
# ----------------------------------------------------------------------------
# Generated/uploaded backgrounds are full-screen images, far too large for the
# browser's localStorage quota, so they live as files under static/user-bg/ and
# the frontend only remembers the (small) URL per theme. Themes are allowlisted
# so the <theme> path segment can never be attacker-controlled traversal.

THEMES = ("light", "dark", "sakura", "kurenai")
BG_DIR = DATA_DIR / "user-bg"


@app.route("/static/user-bg/<path:fn>")
def user_bg(fn):
    """Serve persisted theme backgrounds from the writable data dir.

    They no longer live under static/ (so they survive in a frozen build), but
    the frontend still references them at /static/user-bg/<theme>.<ext>; this
    route takes precedence over the static handler for that prefix.
    """
    if not BG_DIR.exists():
        return ("", 404)
    return send_from_directory(str(BG_DIR), fn)
# data:image/<x>;base64, mime -> file extension we persist it under.
BG_EXTS = {"png": "png", "jpeg": "jpg", "jpg": "jpg", "webp": "webp"}
BG_MAX_BYTES = 25 * 1024 * 1024  # generous cap for a single full-screen image


def _bg_files(theme: str):
    """Existing background file(s) for a theme (any saved extension)."""
    return [BG_DIR / f"{theme}.{ext}" for ext in set(BG_EXTS.values())
            if (BG_DIR / f"{theme}.{ext}").exists()]


@app.route("/api/background", methods=["POST"])
def api_background_set():
    """Persist a data-URI image as the given theme's background.

    Body: ``{ theme, image: "data:image/png;base64,..." }``. Returns the
    cache-busted URL the frontend should point ``--waifu-image`` at.
    """
    import re
    import time as _time
    import base64 as _b64

    body = request.get_json(force=True, silent=True) or {}
    theme = str(body.get("theme", "")).strip()
    if theme not in THEMES:
        return jsonify(error=f"unknown theme: {theme!r}"), 400

    data_uri = body.get("image") or ""
    m = re.match(r"^data:image/([a-zA-Z0-9.+-]+);base64,(.+)$", data_uri, re.DOTALL)
    if not m:
        return jsonify(error="image must be a base64 data:image/* URI"), 400
    mime, b64 = m.group(1).lower(), m.group(2)
    ext = BG_EXTS.get(mime)
    if not ext:
        return jsonify(error=f"unsupported image type: {mime}"), 400

    try:
        raw = _b64.b64decode(b64, validate=True)
    except Exception:
        return jsonify(error="malformed base64 image data"), 400
    if not raw:
        return jsonify(error="empty image"), 400
    if len(raw) > BG_MAX_BYTES:
        return jsonify(error="image too large"), 413

    BG_DIR.mkdir(parents=True, exist_ok=True)
    # Remove any prior file (possibly a different extension) so themes stay 1:1.
    for old in _bg_files(theme):
        try: old.unlink()
        except OSError: pass

    dest = BG_DIR / f"{theme}.{ext}"
    dest.write_bytes(raw)
    return jsonify(url=f"/static/user-bg/{theme}.{ext}?v={int(_time.time())}")


@app.route("/api/background/clear", methods=["POST"])
def api_background_clear():
    """Delete the saved background for a theme (revert to the gradient mesh)."""
    body = request.get_json(force=True, silent=True) or {}
    theme = str(body.get("theme", "")).strip()
    if theme not in THEMES:
        return jsonify(error=f"unknown theme: {theme!r}"), 400
    for f in _bg_files(theme):
        try: f.unlink()
        except OSError: pass
    return jsonify(ok=True)


# ----------------------------------------------------------------------------
# CLI / launcher
# ----------------------------------------------------------------------------

def _open_browser(url: str, delay: float = 1.0):
    Timer(delay, lambda: webbrowser.open_new(url)).start()


def main():
    parser = argparse.ArgumentParser(description="Anima LoRA Editor")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true", help="don't auto-open browser")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )

    url = f"http://{args.host}:{args.port}/"
    print(f"\n  Anima LoRA Editor  —  {url}\n")
    if not args.no_browser:
        _open_browser(url)

    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)


if __name__ == "__main__":
    main()
