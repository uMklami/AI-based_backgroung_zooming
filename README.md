# AI-Based Selective Zoom

Zoom the background of a photo while keeping the subject (foreground) at its
original size and position — like a dolly-zoom for still images.

## Features

- **Automatic mode** — upload a photo, pick a zoom factor, done. The subject
  is detected, the background is reconstructed behind it, zoomed, and
  recomposited.
- **Zoom region selection** — draw a box on the background; the zoom anchors
  on (and is derived from) that region instead of the image center.
- **Smart select (click-to-select)** — click any object and MobileSAM
  (Segment Anything) returns a pixel-accurate mask; refine with
  include/exclude clicks, then add it to the foreground with one button.
- **Foreground painting** — brush over anything extra (a second person, an
  object) to keep it at original size alongside the auto-detected subject.
  Eraser, undo, and clear included.
- **AI super-resolution** — the zoomed background is enhanced with
  Real-ESRGAN (tiled, CPU-friendly) plus adaptive blending and sharpening.
- **Live progress** — per-step status while processing; before/after
  comparison slider on the result.
- Works from file upload or device camera; desktop and mobile.

## Pipeline

1. **Foreground extraction** — CarveKit (`ExtracForeground.py`) produces an
   RGBA matte of the subject; user-painted areas are merged in (`app.py`).
2. **Mask** — derived from the merged alpha channel.
3. **Background reconstruction** — pluggable backend (`inpainters.py`,
   `INPAINT_BACKEND` env var): **LaMa** (CPU default, fast) or **Stable
   Diffusion 2 inpainting** (GPU, synthesizes real content — much better
   quality; install `requirements-gpu.txt`). Size-adaptive mask dilation;
   only the masked region is blended back so untouched background keeps
   full detail.
4. **Zoom** — center- or region-anchored crop (`bgZoom.py`), upscaled with
   tiled Real-ESRGAN + Lanczos blend + unsharp mask (`upscaler.py`).
5. **Composite** — defringed alpha blend of the original-size foreground over
   the zoomed background (`composer.py`).

## Running locally

```bash
pip install -r requirements.txt
python app.py
# open http://localhost:10000
```

The first processed image is slow: CarveKit/LaMa models load and the
Real-ESRGAN weights (~5 MB) download to `./weights/`. The MobileSAM weights
(~40 MB) download to `./weights/mobile_sam.pt` the first time smart select is
used. Subsequent images are much faster.

## Reproducing the paper's results

Everything used for the evaluation in the paper is in `eval/`:

| Path | Contents |
|---|---|
| `eval/run_eval.py` | Original DUTS-TE evaluation; defines the 500-image test sample |
| `eval/rev1/` | Revision experiments (E0–E6), table and figure scripts |
| `eval/rev1/e0_build_synthzoom.py` | Generates the SynthZoom full-reference benchmark |
| `eval/rev1/sets/*.json` | Lists of the cross-domain images (AIM-500, DIS-TE, COD10K, ExDark, WildDepth) |
| `eval/rev1/results/*.csv` | Per-image results of every experiment |
| `eval/rev1/results/tables.txt` | Tables as reported in the paper |
| `eval/rev1/tags_duts_cast_shadow.json` | Manual cast-shadow tags for DUTS-TE |
| `supplementary/Supplementary_Table_S1.xlsx` | Supplementary Table S1: per-image results on DUTS-TE |

```bash
pip install -r eval/rev1/requirements-rev1.txt   # install PyTorch first
python eval/rev1/setup_rev1.py                   # download datasets and baselines
python eval/rev1/run_all.py --list               # show what is done / pending
python eval/rev1/run_all.py                      # run missing experiments, build tables and figures
```

Datasets and external baselines are stored outside the repository in
`~/selzoom_rev/` (override with `SELZOOM_REV`). The datasets (DUTS, AIM-500,
DIS5K, COD10K, ExDark, DIV2K, WildDepth) are not redistributed here; they are
publicly available from their authors under their own licenses. Saved
per-image output images are regenerated with `--save`.

## API

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Web app |
| `/process` | POST | multipart: `file`, `zoom` (1–4), optional `fg_mask` (PNG, white=keep), optional `zoom_rect` (JSON `{x,y,w,h}`, normalized). Returns `{job_id}` |
| `/status/<job_id>` | GET | Job progress: `{status, step, progress, result_url, ...}` |
| `/segment_init` | POST | multipart `file`; stores the editing image, returns `{seg_id}` |
| `/segment` | POST | JSON `{seg_id, points: [[x,y],...], labels: [1|0,...]}`; returns `{mask}` (RGBA data URL) |
| `/uploads/<f>`, `/results/<f>` | GET | Serve images |

## Deployment (Render)

`render.yaml` is configured for gunicorn (1 worker so models load once,
8 threads, 600s timeout). Use at least a **standard** instance — torch +
CarveKit + LaMa need the RAM. Note: on ephemeral disks the model weights
re-download after each deploy/restart.

## Licensing (commercial use)

All bundled models/libraries are commercial-friendly: LaMa (Apache-2.0),
CarveKit (Apache-2.0), MobileSAM (Apache-2.0, loaded directly — no AGPL
`ultralytics` dependency), Real-ESRGAN (BSD-3-Clause). The optional Stable
Diffusion 2 weights are CreativeML OpenRAIL-M: commercial use permitted with
use-based restrictions. Verify all licenses independently before launch.

## Tuning knobs

- `upscaler.py`: `AI_MIN_SCALE` (when AI kicks in), `BLEND_HI/BLEND_LO`
  (sharpness vs. hallucination), `SHARPEN_AMOUNT`, `TILE` size,
  `USE_AI_UPSCALE=0` env var to disable AI upscaling entirely.
- `app.py` step 3: dilation factor `0.02` (raise to `0.03` if inpaint smears
  near subject edges), LaMa working resolution `1024`.
- `composer.py`: defringe erosion iterations (currently 2px).
- `bgZoom.py`: `MAX_ZOOM` cap (currently 4x).
