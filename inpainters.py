"""
Pluggable inpainting backends.

Select with the INPAINT_BACKEND environment variable:
  "lama" — LaMa (default on CPU): fast, light, decent quality
  "sd"   — Stable Diffusion 2 inpainting (GPU strongly recommended):
           synthesizes real content in the hole — kills the "smear" look
  "auto" — sd if CUDA is available, otherwise lama (default)

Both backends expose the same interface:
  inpainter(image_pil_rgb, mask_pil_l) -> PIL RGB image, same size
"""
import os

import torch
from PIL import Image

# The official stabilityai/stable-diffusion-2-inpainting repository has been
# withdrawn from Hugging Face; this community mirror carries the same weights
# (OpenRAIL++). Pinned to a revision for reproducibility; SD_MODEL_ID and
# SD_REVISION override.
SD_MODEL_ID = os.environ.get("SD_MODEL_ID", "sd2-community/stable-diffusion-2-inpainting")
SD_REVISION = os.environ.get("SD_REVISION", "5f74973cbb64c8568780732c17f43eb269d63a0d")
SD_MAX_SIDE = 768  # SD2 works best <= 768; result is blended back at full res

SD_PROMPT = ("empty scene background, seamless continuation of the "
             "surroundings, photorealistic, high detail")
SD_NEGATIVE = ("person, people, animal, object, text, watermark, "
               "artifacts, blurry, distorted")


class LamaInpainter:
    """Wrapper around simple-lama-inpainting (Apache-2.0)."""

    def __init__(self, device=None):
        from simple_lama_inpainting import SimpleLama
        # cuda > mps > cpu (LAMA_DEVICE overrides); MPS output matches CPU
        # to within one grey level and is ~15x faster on Apple silicon
        device = device or os.environ.get("LAMA_DEVICE")
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        print(f"Inpainting backend: LaMa ({device})")
        self._model = SimpleLama(device=torch.device(device))

    def __call__(self, image, mask):
        return self._model(image, mask)


class SDInpainter:
    """Stable Diffusion 2 inpainting via diffusers (OpenRAIL-M weights).
    Requires: pip install -r requirements-gpu.txt"""

    def __init__(self):
        try:
            from diffusers import StableDiffusionInpaintPipeline
        except ImportError:
            raise RuntimeError(
                "SD inpainting needs diffusers. Run: "
                "pip install -r requirements-gpu.txt")

        if torch.cuda.is_available():
            self.device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        if self.device == "cpu":
            print("WARNING: SD inpainting on CPU is extremely slow "
                  "(minutes per image). A GPU is strongly recommended.")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        print(f"Inpainting backend: Stable Diffusion 2 ({self.device})")
        self._pipe = StableDiffusionInpaintPipeline.from_pretrained(
            SD_MODEL_ID, revision=SD_REVISION, torch_dtype=dtype,
            use_safetensors=True)
        self._pipe = self._pipe.to(self.device)
        if self.device in ("cuda", "mps"):
            self._pipe.enable_attention_slicing()
        self._gen = torch.Generator("cpu")

    def __call__(self, image, mask):
        w, h = image.size

        # Work at SD's preferred resolution, multiples of 8
        scale = min(1.0, SD_MAX_SIDE / max(w, h))
        nw = max(64, int(round(w * scale / 8)) * 8)
        nh = max(64, int(round(h * scale / 8)) * 8)
        img_r = image.resize((nw, nh), Image.LANCZOS)
        mask_r = mask.resize((nw, nh), Image.NEAREST)

        out = self._pipe(
            prompt=SD_PROMPT,
            negative_prompt=SD_NEGATIVE,
            image=img_r,
            mask_image=mask_r,
            guidance_scale=7.5,
            num_inference_steps=30,
            generator=self._gen.manual_seed(int(os.environ.get("SD_SEED", "0"))),
        ).images[0]

        # Caller blends only the masked region back onto the sharp
        # original, so upscaling the result here is fine
        return out.resize((w, h), Image.LANCZOS)


def get_inpainter():
    backend = os.environ.get("INPAINT_BACKEND", "auto").lower()
    if backend == "auto":
        backend = "sd" if torch.cuda.is_available() else "lama"

    if backend == "sd":
        try:
            return SDInpainter()
        except Exception as e:
            print(f"SD backend unavailable ({e}); falling back to LaMa.")
            return LamaInpainter()
    return LamaInpainter()
