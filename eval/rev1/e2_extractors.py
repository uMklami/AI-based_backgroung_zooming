"""
E1 (guide): foreground extractor comparison -> Tables 7 and 8.

    python eval/rev1/e2_extractors.py duts|aim500 [--limit N] [--time N]

Variants (all at the same 1024 px working size):
  tracer_fba         ours: CarveKit Tracer-B7 + auto trimap + FBA (pipeline)
  u2net_fba          CarveKit U^2-Net + FBA (same matting, other segmenter)
  gfm                GFM r34_2b, TT, hybrid inference (AM-2k weights)
  tracer_vitmatte    Tracer-B7 mask -> same trimap generator -> ViTMatte-S
  sam2_auto          SAM 2.1-L with the bounding box of the Tracer-B7 mask
  sam2_auto_vitmatte SAM 2.1-L (auto box) -> trimap -> ViTMatte-S
  sam2_gt_vitmatte   SAM 2.1-L with the ground-truth box (oracle) -> ViTMatte-S

With --time N, every variant is instead timed on CPU over the first N
images (after 3 warm-up images) and parameter counts are recorded.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (EXT, cap_size, load_rgb, load_gray, duts_sample, set_items,  # noqa: E402
                    f1_iou, matting_metrics, write_csv, seed_all, accel_device)

VITMATTE_ID = "hustvl/vitmatte-small-composition-1k"
SAM2_ID = "facebook/sam2.1-hiera-large"


def nparams(m):
    return sum(p.numel() for p in m.parameters()) / 1e6


class Extractors:
    def __init__(self, dev):
        self.dev = dev
        from ExtracForeground import ForegroundExtractor
        from carvekit.api.high import HiInterface
        ck = "cuda" if dev == "cuda" else "cpu"                # FBA fails on MPS
        self.ours = ForegroundExtractor(device=ck)               # Tracer-B7 + FBA
        self.u2 = HiInterface(object_type="hairs-like", batch_size_seg=1,
                                  batch_size_matting=1, device=ck,
                                  seg_mask_size=640, matting_mask_size=2048,
                                  trimap_prob_threshold=10, trimap_dilation=10,
                                  trimap_erosion_iters=10, fp16=False)
        self.seg = self.ours.interface.segmentation_pipeline
        self.trimap = self.ours.interface.trimap_generator
        self._vm = self._sam = self._gfm = None

    # ---------------- models
    def vitmatte(self):
        if self._vm is None:
            from transformers import VitMatteForImageMatting, VitMatteImageProcessor
            self._vm = (VitMatteImageProcessor.from_pretrained(VITMATTE_ID),
                        VitMatteForImageMatting.from_pretrained(VITMATTE_ID).eval().to(self.dev))
        return self._vm

    def sam2(self):
        if self._sam is None:
            from transformers import Sam2Model, Sam2Processor
            self._sam = (Sam2Processor.from_pretrained(SAM2_ID),
                         Sam2Model.from_pretrained(SAM2_ID).eval().to(self.dev))
        return self._sam

    def gfm(self):
        if self._gfm is None:
            sys.path.insert(0, os.path.join(EXT, "GFM", "core"))
            from gfm import GFM

            class A:
                backbone = "r34_2b"
                rosta = "TT"
            model = GFM(A())
            sd = torch.load(os.path.join(EXT, "GFM", "models", "pretrained", "gfm_r2b_tt.zip"),
                            map_location="cpu", weights_only=False)
            model.load_state_dict(sd, strict=True)
            # GFM's pyramid pooling is not supported on MPS, so it runs on CPU
            self._gfm_dev = "cpu" if self.dev == "mps" else self.dev
            self._gfm = model.eval().to(self._gfm_dev)
        return self._gfm

    # ---------------- building blocks
    def seg_mask(self, rgb):
        """Tracer-B7 mask from the default extractor (PIL L, same size)."""
        return self.seg(images=[Image.fromarray(rgb)])[0]

    def trimap_from(self, rgb, mask_pil):
        return np.array(self.trimap(Image.fromarray(rgb), mask_pil))

    def run_vitmatte(self, rgb, trimap):
        proc, model = self.vitmatte()
        inp = proc(images=Image.fromarray(rgb), trimaps=Image.fromarray(trimap),
                   return_tensors="pt").to(self.dev)
        with torch.no_grad():
            a = model(**inp).alphas[0, 0].float().cpu().numpy()
        h, w = rgb.shape[:2]
        return (np.clip(a[:h, :w], 0, 1) * 255).astype(np.uint8)

    def run_sam2(self, rgb, box):
        proc, model = self.sam2()
        inp = proc(images=Image.fromarray(rgb), input_boxes=[[list(map(float, box))]],
                   return_tensors="pt").to(self.dev)
        with torch.no_grad():
            out = model(**inp, multimask_output=False)
        masks = proc.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"].cpu())[0]
        return (masks[0, 0].numpy() > 0).astype(np.uint8) * 255

    def run_gfm(self, rgb):
        """Hybrid test strategy of the GFM authors (1/3 glance, 1/2 focus)."""
        sys.path.insert(0, os.path.join(EXT, "GFM", "core"))
        from util import get_masked_local_from_global_test
        from skimage.transform import resize
        model = self.gfm()
        h, w = rgb.shape[:2]
        img = rgb.astype(np.float64) / 255.0

        def infer(ratio):
            nh = max(32, int(h * ratio) - int(h * ratio) % 32)
            nw = max(32, int(w * ratio) - int(w * ratio) % 32)
            s = resize(img, (nh, nw)) * 255.0
            t = torch.from_numpy(s.astype(np.float32)[None]).permute(0, 3, 1, 2)
            t = t.to(self._gfm_dev)
            with torch.no_grad():
                pg, pf, pfu = model(t)
            pg = pg.cpu().numpy()           # 1 x 3 x h x w, as GFM's helper expects
            pf = pf[0, 0].cpu().numpy()
            from util import gen_trimap_from_segmap_e2e
            return gen_trimap_from_segmap_e2e(pg), pf
        g1, _ = infer(1 / 3)
        _, f2 = infer(1 / 2)
        g1 = resize(g1, (h, w)) * 255.0
        f2 = resize(f2, (h, w))
        a = get_masked_local_from_global_test(g1, f2)
        return (np.clip(a, 0, 1) * 255).astype(np.uint8)

    # ---------------- variants
    def new_image(self, saved_alpha=None):
        """Per-image cache: the U^2-Net mask and the SAM 2 auto-box mask are
        shared between variants; our matte is reused from E1 if saved."""
        self._cache = {"ours": saved_alpha}

    def run(self, variant, rgb, gt=None):
        c = getattr(self, "_cache", {})
        if variant == "tracer_fba":
            if c.get("ours") is not None:
                return c["ours"]
            return self.ours.extract_array(rgb)[:, :, 3]
        if variant == "u2net_fba":
            out = np.array(self.u2([Image.fromarray(rgb).convert("RGBA")])[0])
            return out[:, :, 3]
        if variant == "gfm":
            return self.run_gfm(rgb)
        if "seg" not in c:
            c["seg"] = self.seg_mask(rgb)
        m = c["seg"]
        if variant == "tracer_vitmatte":
            return self.run_vitmatte(rgb, self.trimap_from(rgb, m))
        if variant.startswith("sam2"):
            src = np.array(gt) if "gt" in variant else np.array(m)
            ys, xs = np.nonzero(src > 127)
            if len(ys) == 0:
                return np.zeros(rgb.shape[:2], np.uint8)
            box = (xs.min(), ys.min(), xs.max(), ys.max())
            key = "sam_gt" if "gt" in variant else "sam_auto"
            if key not in c:
                c[key] = self.run_sam2(rgb, box)
            sm = c[key]
            if variant == "sam2_auto":
                return sm
            return self.run_vitmatte(rgb, self.trimap_from(rgb, Image.fromarray(sm)))
        raise ValueError(variant)

    def params(self):
        from carvekit.ml.wrap.u2net import U2NET  # noqa: F401
        return {
            "tracer_fba": nparams(self.ours.interface.u2net) + nparams(self.ours.interface.fba),
            "u2net_fba": nparams(self.u2.u2net) + nparams(self.u2.fba),
            "gfm": nparams(self.gfm()),
            "tracer_vitmatte": nparams(self.ours.interface.u2net) + nparams(self.vitmatte()[1]),
            "sam2_auto": nparams(self.ours.interface.u2net) + nparams(self.sam2()[1]),
            "sam2_auto_vitmatte": nparams(self.ours.interface.u2net) + nparams(self.sam2()[1]) + nparams(self.vitmatte()[1]),
            "sam2_gt_vitmatte": nparams(self.sam2()[1]) + nparams(self.vitmatte()[1]),
        }


VARIANTS = ["tracer_fba", "u2net_fba", "gfm", "tracer_vitmatte", "sam2_auto",
            "sam2_auto_vitmatte", "sam2_gt_vitmatte"]


def load(it):
    rgb = load_rgb(it["image"])
    gt = load_gray(it["mask"])
    return cap_size(rgb, gt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--time", type=int, default=0)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    args = ap.parse_args()
    seed_all()
    items = duts_sample() if args.set == "duts" else set_items(args.set)
    variants = args.variants.split(",")

    if args.time:
        torch.set_num_threads(os.cpu_count())
        ex = Extractors("cpu")
        rows = []
        prm = ex.params()
        for v in variants:
            ts = []
            for k, it in enumerate(items[:args.time + 3]):
                rgb, gt = load(it)
                t = time.perf_counter()
                ex.run(v, rgb, gt)
                if k >= 3:
                    ts.append((time.perf_counter() - t) * 1000)
            rows.append(dict(variant=v, cpu_ms_mean=np.mean(ts), cpu_ms_sd=np.std(ts),
                             params_m=prm[v], n=len(ts)))
            print(rows[-1], flush=True)
        write_csv(rows, f"e2_time_{args.set}.csv")
        return

    ex = Extractors(accel_device())
    if args.limit:
        items = items[:args.limit]
    rows = []
    from common import RESULTS
    e1_dir = os.path.join(RESULTS, "outputs", args.set)
    for i, it in enumerate(items, 1):
        rgb, gt = load(it)
        ap_ = os.path.join(e1_dir, it.get("name", os.path.basename(it["image"])) + "_alpha.png")
        ex.new_image(load_gray(ap_) if os.path.exists(ap_) else None)
        for v in variants:
            try:
                a = ex.run(v, rgb, gt)
                if a.shape != gt.shape:
                    a = cv2.resize(a, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
                r = dict(name=it.get("name", os.path.basename(it["image"])), variant=v)
                for k in ("type", "category"):
                    if k in it:
                        r[k] = it[k]
                r["f1"], r["iou"] = f1_iou(a, gt)
                if args.set == "aim500":
                    r.update(matting_metrics(a / 255.0, gt / 255.0))
                rows.append(r)
            except Exception as e:
                print(f"{v} failed on {it.get('name')}: {e}", flush=True)
        print(f"[{i}/{len(items)}] " + " ".join(
            f"{r['variant']}={r['f1']:.3f}" for r in rows[-len(variants):]), flush=True)
        if i % 20 == 0:
            write_csv(rows, f"e2_{args.set}.csv")
    print("saved", write_csv(rows, f"e2_{args.set}.csv"))


if __name__ == "__main__":
    main()
