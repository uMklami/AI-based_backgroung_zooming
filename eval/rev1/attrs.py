"""
Image attributes for Tables 9 and 11 that are not computed in E1.

  indoor/outdoor  zero-shot CLIP ViT-B/32 with the prompts
                  "a photo taken indoors" / "a photo taken outdoors"
  multi           >= 2 ground-truth components each covering >= 0.5% of
                  the frame (multiple salient objects), sets with masks

    python eval/rev1/attrs.py
"""
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import duts_sample, set_items, load_gray, write_csv, accel_device  # noqa: E402

SETS = [("duts", "DUTS-TE"), ("aim500", "AIM-500"), ("dis_te", "DIS-TE"),
        ("cod10k", "COD10K-test"), ("exdark", "ExDark"), ("wilddepth", "WildDepth")]


def main():
    from transformers import CLIPModel, CLIPProcessor
    dev = accel_device()
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().to(dev)
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    prompts = ["a photo taken indoors", "a photo taken outdoors"]
    io_rows, mc_rows = [], []
    for key, label in SETS:
        items = duts_sample() if key == "duts" else set_items(key)
        for it in items:
            name = it.get("name", os.path.basename(it["image"]))
            im = Image.open(it["image"]).convert("RGB")
            inp = proc(text=prompts, images=im, return_tensors="pt", padding=True).to(dev)
            with torch.no_grad():
                p = model(**inp).logits_per_image.softmax(-1)[0].cpu().numpy()
            io_rows.append(dict(set=label, name=name, indoor=int(p[0] > p[1]), p_indoor=float(p[0])))
            if it.get("mask"):
                g = load_gray(it["mask"]) > 127
                n, lab, st, _ = cv2.connectedComponentsWithStats(g.astype(np.uint8))
                big = (st[1:, cv2.CC_STAT_AREA] >= 0.005 * g.size).sum()
                mc_rows.append(dict(set=label, name=name, multi=int(big >= 2), n_comp=int(big)))
        print(label, len(items), flush=True)
    write_csv(io_rows, "attr_indoor_outdoor.csv")
    write_csv(mc_rows, "attr_multi.csv")


if __name__ == "__main__":
    main()
