"""
E1/E2/E3: default pipeline on a dataset, per image.
Fills Tables 3-6 (DUTS-TE), 9, 10 (left) and 11.

    python eval/rev1/e1_run_sets.py duts|aim500|dis_te|cod10k|exdark|wilddepth [--limit N]

For every image (long side capped at 1024 px, as in the paper):
  * segmentation F1/IoU of the automatic matte (if ground truth exists)
  * synthetic-hole inpainting PSNR/SSIM/LPIPS (hole placed in known
    background; for sets without masks the predicted matte, dilated by
    the Eq. 4 margin, is used to keep the hole off the subject)
  * downscale-upscale ablation at 1.8x: Lanczos, SR only (w = 1, no
    unsharp) and the adaptive blend of Eqs. 10-11 (paper default)
  * the full edit at z = 1.8 with contact anchoring and defringing, its
    deviation from the input, and (with ground truth) the oracle gap,
    leak ratio and carry-over ratio
  * per-module wall-clock times of the deployment path
  * attributes for Table 11: subject area, mean luminance, clutter
"""
import argparse
import os
import random
import sys
import time

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (Models, core, cap_size, load_rgb, load_gray, duts_sample,  # noqa: E402
                    set_items, f1_iou, lpips, full_psnr, full_ssim, leak_ratio,
                    carry_over_ratio, find_bg_hole, write_csv, RESULTS, SEED, sync,
                    seed_all)

Z = 1.8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save", action="store_true", help="save mattes/outputs")
    args = ap.parse_args()
    seed_all()

    items = duts_sample() if args.set == "duts" else set_items(args.set)
    if args.limit:
        items = items[:args.limit]
    m = Models()
    rng = random.Random(SEED)
    cfg = core.Config()
    out_dir = os.path.join(RESULTS, "outputs", args.set)
    os.makedirs(out_dir, exist_ok=True)
    rows = []

    for i, it in enumerate(items, 1):
        try:
            rgb = load_rgb(it["image"])
            gt = load_gray(it["mask"]) if it.get("mask") else None
            rgb, gt = cap_size(rgb, gt)
            h, w = rgb.shape[:2]
            r = dict(name=it.get("name", os.path.basename(it["image"])), h=h, w=w)
            for k in ("type", "category", "subset", "species", "source"):
                if k in it:
                    r[k] = it[k]

            # ---- extraction (automatic matte)
            t0 = time.perf_counter()
            fg = m.extractor.extract_array(rgb)
            r["t_extract"] = time.perf_counter() - t0
            if fg.shape[:2] != (h, w):
                fg = cv2.resize(fg, (w, h), interpolation=cv2.INTER_LANCZOS4)
            F, A = fg[:, :, :3], fg[:, :, 3]
            Image.fromarray(A).save(os.path.join(out_dir, r["name"] + "_alpha.png"))
            if gt is not None:
                r["f1"], r["iou"] = f1_iou(A, gt)
                r["gt_area"] = float((gt > 127).mean())
            r["pred_area"] = float((A > 127).mean())

            # ---- attributes
            ref_mask = (gt > 127) if gt is not None else (A > 127)
            ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)[:, :, 0]
            r["luminance"] = float(ycc.mean() / 255.0)
            edges = cv2.Canny(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 100, 200) > 0
            bgm = ~cv2.dilate(ref_mask.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
            r["clutter"] = float(edges[bgm].mean()) if bgm.any() else np.nan

            # ---- synthetic-hole inpainting (evaluation only)
            if gt is not None:
                keep_out = gt > 127
            else:
                keep_out = core.build_inpaint_mask(A, None, cfg) > 0
            hole = find_bg_hole(keep_out, rng=rng)
            if hole:
                y, x, rh, rw, frac = hole
                hm = np.zeros((h, w), np.uint8)
                hm[y:y + rh, x:x + rw] = 255
                inp = np.array(m.lama(Image.fromarray(rgb), Image.fromarray(hm)))[:h, :w]
                a, b = inp[y:y + rh, x:x + rw], rgb[y:y + rh, x:x + rw]
                r["inp_psnr"] = full_psnr(a, b)
                r["inp_ssim"] = full_ssim(a, b)
                r["inp_lpips"] = lpips(a, b)
                r["hole_frac"] = frac

            # ---- downscale-upscale ablation at 1.8x (evaluation only)
            small = cv2.resize(rgb, (int(w / Z), int(h / Z)), interpolation=cv2.INTER_AREA)
            lan = cv2.resize(small, (w, h), interpolation=cv2.INTER_LANCZOS4)
            sr = m.upscaler.upscale(small, w, h, mode="sr", unsharp=False)
            ad = m.upscaler.upscale(small, w, h, mode="adaptive", unsharp=True)
            for k, v in (("lan", lan), ("sr", sr), ("ad", ad)):
                r[f"up_{k}_psnr"] = full_psnr(v, rgb)
                r[f"up_{k}_lpips"] = lpips(v, rgb)

            # ---- the edit itself (deployment path), timed per stage
            t0 = time.perf_counter()
            M = core.build_inpaint_mask(A, None, cfg)
            B = core.reconstruct_background(rgb, M, m.lama, cfg)
            t1 = time.perf_counter()
            p = core.contact_point(A)
            win = core.anchored_window(h, w, Z, p)
            Bz = core.magnify(B, win, m.upscaler, cfg)
            sync()
            t2 = time.perf_counter()
            C = core.composite(F, A, Bz, cfg)
            t3 = time.perf_counter()
            r["t_inpaint"], r["t_zoom"], r["t_comp"] = t1 - t0, t2 - t1, t3 - t2
            r["t_total"] = r["t_extract"] + t3 - t0
            r["dev_psnr"] = full_psnr(C, rgb)
            r["dev_ssim"] = full_ssim(C, rgb)
            r["dev_lpips"] = lpips(C, rgb)
            if p is not None:
                r["p_x"], r["p_y"] = p
                r["drift_center_px"] = core.contact_drift(h, w, Z, p)
            Image.fromarray(C).save(os.path.join(out_dir, r["name"] + "_comp.jpg"), quality=92)
            if args.save:
                Image.fromarray(B).save(os.path.join(out_dir, r["name"] + "_bg.jpg"), quality=92)
                Image.fromarray(Bz).save(os.path.join(out_dir, r["name"] + "_bgz.jpg"), quality=92)

            # ---- oracle composite with the ground-truth matte
            if gt is not None:
                r["leak"] = leak_ratio(gt, M)
                r["carry"] = carry_over_ratio(A, gt)
                Mg = core.build_inpaint_mask(gt, None, cfg)
                Bg = core.reconstruct_background(rgb, Mg, m.lama, cfg)
                pg = core.contact_point(gt)
                wg = core.anchored_window(h, w, Z, pg)
                Bzg = core.magnify(Bg, wg, m.upscaler, cfg)
                Cg = core.composite(rgb, gt, Bzg, cfg)
                # Compare under the same crop window so the gap measures
                # what the matte error does to the image (ghosts, pasted
                # background, halos) rather than a global shift; the
                # anchor displacement is reported separately.
                Cw = core.composite(rgb, A, core.magnify(B, wg, m.upscaler, cfg), cfg)
                r["oracle_gap"] = lpips(Cw, Cg)
                if p is not None and pg is not None:
                    r["anchor_shift_px"] = float(np.hypot(p[0] - pg[0], p[1] - pg[1]))
                Image.fromarray(Cg).save(os.path.join(out_dir, r["name"] + "_oracle.jpg"), quality=92)

            rows.append(r)
            print(f"[{i}/{len(items)}] {r['name']} f1={r.get('f1', float('nan')):.3f} "
                  f"t={r['t_total']:.2f}s", flush=True)
        except Exception as e:
            print(f"[{i}/{len(items)}] FAILED {it.get('name')}: {e}", flush=True)
        if i % 25 == 0:
            write_csv(rows, f"e1_{args.set}.csv")
    print("saved", write_csv(rows, f"e1_{args.set}.csv"))


if __name__ == "__main__":
    main()
