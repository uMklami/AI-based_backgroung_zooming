"""
Smart upscaler: Real-ESRGAN (compact SRVGG net, CPU-friendly) with
automatic fallback to Lanczos if the model is unavailable or fails.

Sharpness pipeline:
  1. Tiled inference — any crop size goes through the network at FULL
     resolution (no detail-destroying pre-shrink), memory-bounded per tile.
  2. Adaptive AI/Lanczos blend — mostly-AI at mild zooms, more Lanczos at
     extreme zooms where the net starts hallucinating structure.
  3. Unsharp mask — final micro-contrast boost on the zoomed background.

No extra pip dependencies: uses plain torch (already required).
Weights (~4.7 MB) are downloaded once to ./weights/.
"""
import os
import urllib.request

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WEIGHTS_URL = ("https://github.com/xinntao/Real-ESRGAN/releases/download/"
               "v0.2.5.0/realesr-general-x4v3.pth")
WEIGHTS_DIR = "weights"
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "realesr-general-x4v3.pth")

# Below this scale factor Lanczos alone is used (still sharpened)
AI_MIN_SCALE = 1.25
# Absolute safety cap on crop pixels before SR. The paper states that
# SR sees native, never-resampled pixels, so this is set high enough
# (16 MP) that it only triggers on extreme inputs; below it, tiling
# alone bounds memory.
MAX_INPUT_PIXELS = 16_000_000
# Adaptive blend: AI share ramps from BLEND_HI (mild zoom) to BLEND_LO (4x)
BLEND_HI, BLEND_LO = 0.90, 0.65
# Unsharp mask strength / radius
SHARPEN_AMOUNT = 0.35
SHARPEN_SIGMA = 1.2
# Tiling
TILE = 384
TILE_OVERLAP = 16


def pick_device():
    """cuda > mps > cpu; SR_DEVICE env var overrides."""
    forced = os.environ.get("SR_DEVICE")
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def blend_weight(scale):
    """Eq. 10: w(z) = 0 below AI_MIN_SCALE, then 0.90 -> 0.65 linearly at 4x."""
    if scale < AI_MIN_SCALE:
        return 0.0
    t = min(max((scale - AI_MIN_SCALE) / (4.0 - AI_MIN_SCALE), 0.0), 1.0)
    return BLEND_HI - (BLEND_HI - BLEND_LO) * t


class SRVGGNetCompact(nn.Module):
    """Compact VGG-style SR network used by realesr-general-x4v3 (Real-ESRGAN)."""

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(nn.PReLU(num_parameters=num_feat))
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        return out + base


def tiled_scale(img, run_fn, scale=4, tile=TILE, overlap=TILE_OVERLAP):
    """
    Run `run_fn` (tile -> scale*tile upscale) over overlapping tiles and
    stitch with linear feathering. Keeps peak memory constant regardless
    of image size.
    """
    h, w = img.shape[:2]
    if h <= tile and w <= tile:
        return run_fn(img)

    out = np.zeros((h * scale, w * scale, 3), np.float32)
    weight = np.zeros((h * scale, w * scale, 1), np.float32)
    step = tile - overlap
    f = overlap * scale

    for y0 in range(0, max(h - overlap, 1), step):
        for x0 in range(0, max(w - overlap, 1), step):
            y1 = min(y0 + tile, h)
            x1 = min(x0 + tile, w)
            y0a = max(0, y1 - tile)
            x0a = max(0, x1 - tile)

            t = run_fn(img[y0a:y1, x0a:x1]).astype(np.float32)
            th, tw = t.shape[:2]

            wmask = np.ones((th, tw, 1), np.float32)
            if f > 0:
                ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
                if y0a > 0:
                    wmask[:f] *= ramp[:, None, None]
                if x0a > 0:
                    wmask[:, :f] *= ramp[None, :, None]
                if y1 < h:
                    wmask[-f:] *= ramp[::-1][:, None, None]
                if x1 < w:
                    wmask[:, -f:] *= ramp[::-1][None, :, None]

            oy, ox = y0a * scale, x0a * scale
            out[oy:oy + th, ox:ox + tw] += t * wmask
            weight[oy:oy + th, ox:ox + tw] += wmask

    out /= np.maximum(weight, 1e-6)
    return np.clip(out, 0, 255).astype(np.uint8)


def unsharp(img, amount=SHARPEN_AMOUNT, sigma=SHARPEN_SIGMA):
    """Eq. 11 (lambda = 0.35, sigma_u = 1.2 px)."""
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)


_unsharp_fn = unsharp


class SmartUpscaler:
    """upscale(img_rgb, target_w, target_h) -> RGB uint8 array of target size."""

    def __init__(self, enabled=None):
        if enabled is None:
            enabled = os.environ.get("USE_AI_UPSCALE", "1") != "0"
        self.enabled = enabled
        self.device = torch.device(pick_device())
        self._model = None
        self._load_failed = False

    # ---------- model management ----------
    def _get_model(self):
        if self._model is not None or self._load_failed:
            return self._model
        try:
            if not os.path.exists(WEIGHTS_PATH):
                os.makedirs(WEIGHTS_DIR, exist_ok=True)
                print(f"Downloading Real-ESRGAN weights to {WEIGHTS_PATH} ...")
                urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_PATH)
            model = SRVGGNetCompact()
            state = torch.load(WEIGHTS_PATH, map_location="cpu")
            if "params" in state:
                state = state["params"]
            model.load_state_dict(state, strict=True)
            model.eval().to(self.device)
            self._model = model
            print("Real-ESRGAN upscaler ready.")
        except Exception as e:
            print(f"AI upscaler unavailable ({e}); falling back to Lanczos.")
            self._load_failed = True
        return self._model

    def _run_model(self, tile_rgb):
        """Single tile through the network: uint8 RGB -> uint8 RGB (4x)."""
        x = torch.from_numpy(tile_rgb.astype(np.float32) / 255.0)
        x = x.permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            y = self._model(x)
        y = y.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        return (y * 255.0).round().astype(np.uint8)

    # ---------- public API ----------
    def sr4(self, img_rgb):
        """Tiled x4 super-resolution of an RGB uint8 image."""
        model = self._get_model()
        if model is None:
            h, w = img_rgb.shape[:2]
            return self._lanczos(img_rgb, 4 * w, 4 * h)
        h, w = img_rgb.shape[:2]
        if w * h > MAX_INPUT_PIXELS:
            fct = (MAX_INPUT_PIXELS / (w * h)) ** 0.5
            img_rgb = cv2.resize(img_rgb, (int(w * fct), int(h * fct)),
                                 interpolation=cv2.INTER_AREA)
        return tiled_scale(img_rgb, self._run_model, scale=4)

    def finish(self, ai, crop, target_w, target_h, mode="adaptive",
               unsharp=True, fixed=0.9):
        """Blend an SR result with the Lanczos upscale of the same crop
        (Eq. 10) and sharpen (Eq. 11)."""
        h, w = crop.shape[:2]
        scale = max(target_w / w, target_h / h)
        if mode == "sr":
            wgt = 1.0
        elif mode == "fixed":
            wgt = fixed
        else:
            wgt = blend_weight(scale)
        lan = self._lanczos(crop, target_w, target_h)
        out = cv2.addWeighted(ai, wgt, lan, 1.0 - wgt, 0) if wgt > 0 else lan
        return _unsharp_fn(out) if (unsharp and scale > 1.05) else out

    def upscale(self, img_rgb, target_w, target_h, mode="adaptive",
                unsharp=True, fixed=0.9):
        """Crop -> output size. mode: adaptive (paper default, Eq. 10),
        lanczos (w=0), sr (w=1) or fixed (w=fixed)."""
        h, w = img_rgb.shape[:2]
        scale = max(target_w / w, target_h / h)
        use_ai = (self.enabled and mode != "lanczos" and
                  (mode in ("sr", "fixed") or scale >= AI_MIN_SCALE))
        if use_ai and self._get_model() is not None:
            try:
                y = self.sr4(img_rgb)
                # 4x output -> exact target size; since z <= 4 this is
                # a reduction, so AREA resampling is used
                interp = (cv2.INTER_AREA if y.shape[1] > target_w
                          else cv2.INTER_LANCZOS4)
                ai = cv2.resize(y, (target_w, target_h), interpolation=interp)
                return self.finish(ai, img_rgb, target_w, target_h, mode,
                                   unsharp, fixed)
            except Exception as e:
                print(f"AI upscale failed ({e}); using Lanczos.")
        out = self._lanczos(img_rgb, target_w, target_h)
        return _unsharp_fn(out) if (unsharp and scale > 1.05) else out

    @staticmethod
    def _lanczos(img_rgb, target_w, target_h):
        return cv2.resize(img_rgb, (target_w, target_h),
                          interpolation=cv2.INTER_LANCZOS4)
