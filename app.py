from flask import (Flask, render_template, request, send_from_directory,
                   url_for, jsonify, make_response)
import os
import io
import json
import base64
import threading
import cv2
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
from pathlib import Path
import uuid

from ExtracForeground import ForegroundExtractor
from inpainters import get_inpainter
from upscaler import SmartUpscaler

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path(app.config['RESULT_FOLDER']).mkdir(exist_ok=True)


# ------------------------------------------------------------------- CORS
# The Flutter web/mobile client runs on a different origin, so browsers
# block its requests without these headers. Set ALLOWED_ORIGIN to lock
# this down in production (default "*" is fine for local development).
ALLOWED_ORIGIN = os.environ.get('ALLOWED_ORIGIN', '*')


@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = ALLOWED_ORIGIN
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response


@app.route('/<path:_any>', methods=['OPTIONS'])
@app.route('/', methods=['OPTIONS'])
def cors_preflight(_any=None):
    """Answer the browser's preflight check."""
    return ('', 204)

# ---------------------------------------------------------------- processing

# Array-level implementation of Eqs. 1-13 lives in pipeline_core so the
# app and the evaluation scripts share exactly the same code.
import pipeline_core as core
refine_mask_edges = core.refine_mask_edges


class SelectiveZoomProcessor:
    def __init__(self):
        os.makedirs("images", exist_ok=True)
        os.makedirs("results", exist_ok=True)

        # Models are loaded once here
        self.foreground_extractor = ForegroundExtractor()
        self.inpainter = get_inpainter()  # LaMa or SD (INPAINT_BACKEND)
        self.upscaler = SmartUpscaler()
        self.cfg = core.Config()

    def process_image(self, input_path, zoom_factor=1.5, unique_id="",
                      fg_mask_path=None, zoom_rect=None, progress_cb=None,
                      remove_mask_path=None):
        """
        fg_mask_path:     optional user foreground mask (white = keep at original size)
        remove_mask_path: optional removal mask (white = erase from the photo)
        zoom_rect:        optional normalized {x,y,w,h} region to zoom toward
        progress_cb:      optional fn(step_key, percent)
        """
        def report(step, pct):
            if progress_cb:
                progress_cb(step, pct)

        # ---- Step 1: automatic foreground extraction
        report("foreground", 5)
        print("\n=== Step 1: Extracting Foreground ===")
        foreground = self.foreground_extractor.extract_foreground(input_path)
        report("foreground", 30)

        # ---- Step 2: merge user selection into foreground + build mask
        report("mask", 32)
        print("\n=== Step 2: Building Mask ===")
        orig_rgb = np.array(Image.open(input_path).convert("RGB"))
        h, w = orig_rgb.shape[:2]

        if foreground.shape[:2] != (h, w):
            foreground = cv2.resize(foreground, (w, h),
                                    interpolation=cv2.INTER_LANCZOS4)

        if fg_mask_path and os.path.exists(fg_mask_path):
            print("Merging user-selected foreground areas...")
            # PIL guarantees a 2D array; some cv2 builds return (h, w, 1)
            # from grayscale imread, which breaks boolean indexing
            try:
                user_mask = np.array(Image.open(fg_mask_path).convert("L"))
            except Exception as e:
                print(f"Could not read user mask ({e}); skipping merge")
                user_mask = None
            if user_mask is not None:
                user_mask = np.squeeze(user_mask)
                if user_mask.ndim != 2:
                    user_mask = user_mask[:, :, 0]
                if user_mask.shape != (h, w):
                    user_mask = cv2.resize(user_mask, (w, h),
                                           interpolation=cv2.INTER_NEAREST)
                user_bin = (user_mask > 127).astype(np.uint8) * 255

                # Snap hard selection edges (SAM/brush/box) to the true
                # object boundary and soften them naturally
                user_alpha = refine_mask_edges(orig_rgb, user_bin)

                # Take original pixels everywhere the refined matte has
                # coverage (so soft edge pixels have real colors, not
                # CarveKit leftovers)
                sel = user_alpha > 8
                fg_rgb = foreground[:, :, :3].copy()
                fg_rgb[sel] = orig_rgb[sel]
                alpha = core.fuse_mattes(foreground[:, :, 3], user_alpha)
                foreground = np.dstack((fg_rgb, alpha))

        # Object removal: erase selected areas from the scene entirely —
        # they join the inpaint mask but are excluded from the composite.
        # Removal wins over keep where the two overlap.
        remove_bin = None
        if remove_mask_path and os.path.exists(remove_mask_path):
            print("Applying object removal mask...")
            try:
                rm = np.array(Image.open(remove_mask_path).convert("L"))
                rm = np.squeeze(rm)
                if rm.ndim != 2:
                    rm = rm[:, :, 0]
                if rm.shape != (h, w):
                    rm = cv2.resize(rm, (w, h), interpolation=cv2.INTER_NEAREST)
                remove_bin = rm > 127
                alpha_ch = core.fuse_mattes(foreground[:, :, 3],
                                            remove_bin=remove_bin)
                foreground = np.dstack((foreground[:, :, :3], alpha_ch))
            except Exception as e:
                print(f"Could not read removal mask ({e}); skipping")
                remove_bin = None

        foreground_path = f"images/{unique_id}_foreground_output.png"
        self.foreground_extractor.save_foreground(foreground, foreground_path)

        mask_path = f"images/{unique_id}_mask.png"
        cv2.imwrite(mask_path, core.build_inpaint_mask(
            foreground[:, :, 3], remove_bin, core.Config(dilation_fixed=1)))
        report("mask", 38)

        # ---- Step 3: inpaint the subject out of the background
        # (Eqs. 4-5: adaptive dilation, LaMa at <= 1024 px, blend-back)
        report("inpaint", 40)
        print("\n=== Step 3: Inpainting Background ===")
        mask_dil = core.build_inpaint_mask(foreground[:, :, 3], remove_bin, self.cfg)
        inpainted = core.reconstruct_background(orig_rgb, mask_dil,
                                                self.inpainter, self.cfg)
        inpainted_path = f"images/{unique_id}_inpainted.png"
        Image.fromarray(inpainted).save(inpainted_path)
        report("inpaint", 70)

        # ---- Step 4: zoom the background (Eqs. 6-11). Region-anchored if
        # the user drew a rectangle, otherwise anchored at the subject's
        # ground-contact point so contact cues stay aligned. The crop is
        # taken at native resolution and super-resolved once.
        report("zoom", 72)
        print("\n=== Step 4: Zooming Background ===")
        if zoom_rect:
            zoom_factor, window = core.rect_to_window(h, w, zoom_rect)
        else:
            anchor = core.contact_point(foreground[:, :, 3])
            window = core.anchored_window(h, w, zoom_factor, anchor)
        zoomed = core.magnify(inpainted, window, self.upscaler, self.cfg)
        zoomed_path = f"images/{unique_id}_zoomed_image.jpg"
        Image.fromarray(zoomed).save(zoomed_path, quality=95)
        report("zoom", 88)

        # ---- Step 5: composite (Eqs. 12-13, defringed alpha)
        report("composite", 90)
        print("\n=== Step 5: Compositing Final Image ===")
        result = core.composite(foreground[:, :, :3], foreground[:, :, 3],
                                zoomed, self.cfg)
        output_path = f"results/{unique_id}_final_composite.png"
        Image.fromarray(result).save(output_path)

        report("composite", 98)
        print("\nProcessing completed successfully!")
        return output_path


# Lazy singleton: models load once, on first job
_processor = None
_processor_lock = threading.Lock()


def get_processor():
    global _processor
    with _processor_lock:
        if _processor is None:
            print("Loading models (first request only)...")
            _processor = SelectiveZoomProcessor()
    return _processor


# ------------------------------------------------------- smart select (SAM)

# seg_id -> stored image path for the editing session
seg_sessions = {}
seg_lock = threading.Lock()

_sam_predictor = None
_sam_embedded_id = None      # seg_id whose image embedding is currently loaded
_sam_lock = threading.Lock()
# seg_id -> cached image embedding, so every editing session pays for the
# image encoder once (at upload) and later clicks only run the decoder
_sam_cache = {}
SAM_CACHE_MAX = 16


def _sam_embed(seg_id, path):
    """Run the MobileSAM image encoder once and cache the embedding."""
    global _sam_embedded_id
    predictor = get_sam_predictor()
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise RuntimeError("Could not read session image")
    predictor.set_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    _sam_embedded_id = seg_id
    _sam_cache[seg_id] = (predictor.features, predictor.original_size,
                          predictor.input_size)
    while len(_sam_cache) > SAM_CACHE_MAX:
        _sam_cache.pop(next(iter(_sam_cache)))


def _sam_activate(seg_id, path):
    """Make seg_id's embedding current, re-encoding only on a cache miss."""
    global _sam_embedded_id
    predictor = get_sam_predictor()
    if _sam_embedded_id == seg_id:
        return predictor
    if seg_id in _sam_cache:
        feats, orig_size, in_size = _sam_cache[seg_id]
        predictor.features = feats
        predictor.original_size = orig_size
        predictor.input_size = in_size
        predictor.is_image_set = True
        _sam_embedded_id = seg_id
    else:
        _sam_embed(seg_id, path)
    return predictor

MOBILE_SAM_URL = ("https://github.com/ChaoningZhang/MobileSAM/raw/master/"
                  "weights/mobile_sam.pt")
MOBILE_SAM_PATH = os.path.join("weights", "mobile_sam.pt")


def get_sam_predictor():
    """Lazy-load MobileSAM (Apache-2.0, commercial-friendly) directly via
    the `mobile_sam` package — no AGPL dependencies. Install with:
    pip install git+https://github.com/ChaoningZhang/MobileSAM.git
    Weights (~40 MB) download automatically on first use."""
    global _sam_predictor
    if _sam_predictor is None:
        try:
            from mobile_sam import sam_model_registry, SamPredictor
        except ImportError as e:
            raise RuntimeError(
                f"Smart select needs MobileSAM (import failed: {e}). "
                "Run: pip install timm "
                "git+https://github.com/ChaoningZhang/MobileSAM.git")
        import torch
        if not os.path.exists(MOBILE_SAM_PATH):
            os.makedirs("weights", exist_ok=True)
            print(f"Downloading MobileSAM weights to {MOBILE_SAM_PATH} ...")
            import urllib.request
            urllib.request.urlretrieve(MOBILE_SAM_URL, MOBILE_SAM_PATH)
        print("Loading MobileSAM (first smart-select only)...")
        device = os.environ.get("SAM_DEVICE") or (
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else "cpu")
        sam = sam_model_registry["vit_t"](checkpoint=MOBILE_SAM_PATH)
        sam.to(device)
        sam.eval()
        _sam_predictor = SamPredictor(sam)
    return _sam_predictor


@app.route('/segment_init', methods=['POST'])
def segment_init():
    """Store the editing image once so per-click requests stay tiny."""
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify(error="No file provided"), 400
    file = request.files['file']
    if not allowed_file(file.filename):
        return jsonify(error="Unsupported file type"), 400

    seg_id = uuid.uuid4().hex
    path = os.path.join(app.config['UPLOAD_FOLDER'], f"{seg_id}_seg.png")
    file.save(path)
    with seg_lock:
        seg_sessions[seg_id] = path
    # Encode the image now, when it is loaded, so the first click is as
    # fast as every later one
    try:
        with _sam_lock:
            _sam_embed(seg_id, path)
    except Exception as e:
        print(f"SAM embedding deferred ({e})")
    return jsonify(seg_id=seg_id)


@app.route('/segment', methods=['POST'])
def segment():
    """Run SAM with point prompts; returns an RGBA mask (white where
    selected, transparent elsewhere) as a data URL."""
    data = request.get_json(silent=True) or {}
    seg_id = data.get('seg_id')
    points = data.get('points') or []
    labels = data.get('labels') or []
    bbox = data.get('bbox')  # optional [x1, y1, x2, y2]

    with seg_lock:
        path = seg_sessions.get(seg_id)
    if not path or not os.path.exists(path):
        return jsonify(error="Unknown segment session"), 404
    if len(points) != len(labels):
        return jsonify(error="Invalid points/labels"), 400
    if not points and not bbox:
        return jsonify(error="No prompt given"), 400
    if bbox is not None:
        try:
            bbox = [float(v) for v in bbox]
            assert len(bbox) == 4 and bbox[2] > bbox[0] and bbox[3] > bbox[1]
        except (ValueError, TypeError, AssertionError):
            return jsonify(error="Invalid bbox"), 400

    try:
        with _sam_lock:
            # Embedding was computed at upload; clicks only run the decoder
            predictor = _sam_activate(seg_id, path)

            pc = np.array(points, dtype=np.float32) if points else None
            pl = np.array(labels, dtype=np.float32) if points else None
            bx = np.array(bbox, dtype=np.float32) if bbox else None
            masks, scores, _ = predictor.predict(
                point_coords=pc, point_labels=pl, box=bx,
                multimask_output=False)

        if masks is None or len(masks) == 0 or not masks[0].any():
            return jsonify(error="No object found at that point"), 422

        mask = masks[0]  # bool, already at original image resolution
        with Image.open(path) as im:
            w, h = im.size
        m = (mask.astype(np.uint8)) * 255
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)

        rgba = np.dstack([np.full((h, w), 255, np.uint8)] * 3 + [m])
        buf = io.BytesIO()
        Image.fromarray(rgba, mode='RGBA').save(buf, 'PNG')
        b64 = base64.b64encode(buf.getvalue()).decode()
        return jsonify(mask=f"data:image/png;base64,{b64}")
    except RuntimeError as e:
        return jsonify(error=str(e)), 501
    except Exception as e:
        print(f"Segment error: {e}")
        return jsonify(error=f"Segmentation failed: {e}"), 500


# ---------------------------------------------------------------- job system

jobs = {}
jobs_lock = threading.Lock()


def update_job(job_id, **kw):
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(kw)


def run_job(job_id, input_path, zoom_factor, unique_id, fg_mask_path,
            zoom_rect, remove_mask_path=None):
    try:
        update_job(job_id, status="loading_models", step="models", progress=2)
        processor = get_processor()
        update_job(job_id, status="processing")

        def cb(step, pct):
            update_job(job_id, step=step, progress=pct)

        output_path = processor.process_image(
            input_path, zoom_factor, unique_id=unique_id,
            fg_mask_path=fg_mask_path, zoom_rect=zoom_rect, progress_cb=cb,
            remove_mask_path=remove_mask_path)

        result_filename = f"{unique_id}_processed.png"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        os.replace(output_path, result_path)

        update_job(job_id, status="done", progress=100, result=result_filename)
    except Exception as e:
        print(f"Job {job_id} failed: {e}")
        update_job(job_id, status="error", error=str(e))


def allowed_file(filename):
    return ('.' in filename and
            filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS'])


# ---------------------------------------------------------------- routes

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify(error="No file selected"), 400

    file = request.files['file']
    if not allowed_file(file.filename):
        return jsonify(error="Unsupported file type"), 400

    try:
        zoom_factor = float(request.form.get('zoom', 1.5))
        zoom_factor = max(1.0, min(zoom_factor, 4.0))
    except ValueError:
        return jsonify(error="Invalid zoom factor"), 400

    zoom_rect = None
    rect_raw = request.form.get('zoom_rect')
    if rect_raw:
        try:
            r = json.loads(rect_raw)
            zoom_rect = {k: float(r[k]) for k in ('x', 'y', 'w', 'h')}
        except (ValueError, KeyError, TypeError):
            return jsonify(error="Invalid zoom_rect"), 400

    unique_id = uuid.uuid4().hex
    filename = f"{unique_id}_{secure_filename(file.filename)}"
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(input_path)

    fg_mask_path = None
    if 'fg_mask' in request.files and request.files['fg_mask'].filename != '':
        fg_mask_path = os.path.join(app.config['UPLOAD_FOLDER'],
                                    f"{unique_id}_fgmask.png")
        request.files['fg_mask'].save(fg_mask_path)

    remove_mask_path = None
    if 'remove_mask' in request.files and request.files['remove_mask'].filename != '':
        remove_mask_path = os.path.join(app.config['UPLOAD_FOLDER'],
                                        f"{unique_id}_rmmask.png")
        request.files['remove_mask'].save(remove_mask_path)

    job_id = unique_id
    with jobs_lock:
        jobs[job_id] = {
            "status": "queued", "step": "queued", "progress": 0,
            "original": filename, "result": None, "error": None,
        }

    threading.Thread(
        target=run_job,
        args=(job_id, input_path, zoom_factor, unique_id, fg_mask_path,
              zoom_rect, remove_mask_path),
        daemon=True,
    ).start()

    return jsonify(job_id=job_id,
                   original=url_for('uploaded_file', filename=filename))


@app.route('/status/<job_id>')
def status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            return jsonify(error="Unknown job"), 404
        payload = dict(job)
    if payload.get("result"):
        payload["result_url"] = url_for('result_file', filename=payload["result"])
    payload["original_url"] = url_for('uploaded_file', filename=payload["original"])
    return jsonify(payload)


@app.route('/sw.js')
def service_worker():
    """Serve the service worker from root so its scope covers the whole app."""
    resp = make_response(send_from_directory('static', 'sw.js'))
    resp.headers['Content-Type'] = 'application/javascript'
    resp.headers['Service-Worker-Allowed'] = '/'
    resp.headers['Cache-Control'] = 'no-cache'
    return resp


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/results/<filename>')
def result_file(filename):
    response = make_response(
        send_from_directory(app.config['RESULT_FOLDER'], filename))
    response.headers['Content-Disposition'] = f'inline; filename="{filename}"'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    response.headers['Content-Type'] = 'image/png'
    return response


if __name__ == '__main__':
    Path("images").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    app.run(host="0.0.0.0", port=10000, threaded=True)
