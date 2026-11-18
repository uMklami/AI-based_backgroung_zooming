"""
Runtime, memory, throughput and latency -> Tables 18, 19, 20.

    python eval/rev1/e6_runtime.py modules|scaling|latency

"GPU" = CUDA if available, otherwise Apple MPS; "CPU" = same machine with
all models forced to CPU. CarveKit's FBA Matting does not run on MPS, so
on Apple machines foreground extraction stays on CPU in both columns.
The script records the hardware it ran on in results/e6_hardware.json.

Protocol: 30 DIV2K validation images (0851-0880) resized to the stated
long side; 3 warm-up images; device synchronisation before every timer
read; evaluation-only work excluded; models loaded beforehand.
Memory is the peak MPS driver allocation during the module (sampled every
5 ms), i.e. accelerator memory; CPU-only modules show "-".
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DATA, PROJECT, RESULTS, core, PeakMem, write_csv, load_rgb,  # noqa: E402
                    accel_device)

GPU = accel_device()

N, WARM = 30, 3


def images(long_side=1024):
    names = [f"{i:04d}.png" for i in range(851, 851 + N + WARM)]
    out = []
    for n in names:
        im = load_rgb(os.path.join(DATA, "DIV2K_valid_HR", n))
        h, w = im.shape[:2]
        s = long_side / max(h, w)
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
        out.append(cv2.resize(im, (int(round(w * s)), int(round(h * s))), interpolation=interp))
    return out


def sync(dev):
    if dev == "mps":
        torch.mps.synchronize()
    elif dev == "cuda":
        torch.cuda.synchronize()


class Stack:
    def __init__(self, dev):
        self.dev = dev
        os.environ["SR_DEVICE"] = dev
        from ExtracForeground import ForegroundExtractor
        from inpainters import LamaInpainter
        from upscaler import SmartUpscaler
        from mobile_sam import sam_model_registry, SamPredictor
        self.ex = ForegroundExtractor(device="cuda" if dev == "cuda" else "cpu")
        self.lama = LamaInpainter(device=dev)
        self.up = SmartUpscaler()
        self.up.device = torch.device(dev)
        sam = sam_model_registry["vit_t"](checkpoint=os.path.join(PROJECT, "weights", "mobile_sam.pt"))
        sam.to(dev).eval()
        self.sam = SamPredictor(sam)

    def automatic(self, rgb, z=2.0):
        fg = self.ex.extract_array(rgb)
        return core.run_pipeline(rgb, fg[:, :, :3], fg[:, :, 3], z, self.lama, self.up)


def measure(fn, dev, gpu_module=True):
    sync(dev)
    if gpu_module and dev in ("mps", "cuda"):
        with PeakMem(dev) as pm:
            t = time.perf_counter()
            out = fn()
            sync(dev)
            dt = time.perf_counter() - t
        return out, dt * 1000, pm.gb
    t = time.perf_counter()
    out = fn()
    sync(dev)
    return out, (time.perf_counter() - t) * 1000, np.nan


def exp_modules():
    rows = []
    imgs = images(1024)
    for dev in (GPU, "cpu"):
        st = Stack(dev)
        for k, rgb in enumerate(imgs):
            h, w = rgb.shape[:2]
            r = dict(device=dev, k=k)
            fg, r["extract_ms"], r["extract_gb"] = measure(lambda: st.ex.extract_array(rgb), dev,
                                                           dev == "cuda")
            A = fg[:, :, 3]
            _, r["sam_enc_ms"], r["sam_enc_gb"] = measure(lambda: st.sam.set_image(rgb), dev)
            ys, xs = np.nonzero(A > 127)
            box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], np.float32) if len(ys) else np.array([w / 4, h / 4, 3 * w / 4, 3 * h / 4], np.float32)
            dec = []
            for j in range(5):
                bj = box + np.array([-j, -j, j, j], np.float32)
                (mk, _, _), t_ms, g = measure(lambda: st.sam.predict(box=bj, multimask_output=False), dev)
                dec.append(t_ms)
            r["sam_dec_ms"] = float(np.mean(dec))
            K = mk[0].astype(np.uint8) * 255
            _, r["refine_ms"], _ = measure(lambda: core.fuse_mattes(A, core.refine_mask_edges(rgb, K)), dev, False)
            cfg = core.Config()

            def inpaint():
                M = core.build_inpaint_mask(A, None, cfg)
                return core.reconstruct_background(rgb, M, st.lama, cfg)
            B, r["inpaint_ms"], r["inpaint_gb"] = measure(inpaint, dev)
            win = core.anchored_window(h, w, 2.0, core.contact_point(A))
            Bz, r["zoom_ms"], r["zoom_gb"] = measure(lambda: core.magnify(B, win, st.up, cfg), dev)
            _, r["comp_ms"], _ = measure(lambda: core.composite(fg[:, :, :3], A, Bz, cfg), dev, False)
            r["total_ms"] = r["extract_ms"] + r["inpaint_ms"] + r["zoom_ms"] + r["comp_ms"]
            rows.append(r)
            print(dev, k, {kk: round(v, 1) for kk, v in r.items() if kk.endswith("ms")}, flush=True)
        # throughput: whole automatic pipeline, 1 and 2 concurrent workers
        test = imgs[WARM:]
        for workers in (1, 2):
            t = time.perf_counter()
            with ThreadPoolExecutor(workers) as ex:
                list(ex.map(st.automatic, test))
            sync(dev)
            rows.append(dict(device=dev, k=-workers, throughput=len(test) / (time.perf_counter() - t),
                             workers=workers))
            print(dev, "throughput", workers, rows[-1]["throughput"], flush=True)
        # whole-pipeline peak memory
        with PeakMem(dev) as pm:
            for rgb in test[:10]:
                st.automatic(rgb)
            sync(dev)
        rows.append(dict(device=dev, k=-9, total_gb=pm.gb if dev != "cpu" else np.nan))
        del st
    write_csv(rows, "e6_modules.csv")


def exp_scaling():
    rows = []
    st = Stack(GPU)
    for L in (512, 1024, 2048, 4096):
        imgs = images(L)[:13]
        for k, rgb in enumerate(imgs):
            with PeakMem(GPU) as pm:
                t = time.perf_counter()
                st.automatic(rgb, 2.0)
                sync(GPU)
                dt = time.perf_counter() - t
            rows.append(dict(kind="res", L=L, z=2.0, k=k, total_ms=dt * 1000, gb=pm.gb))
            print("res", L, k, round(dt, 2), round(pm.gb, 2), flush=True)
    imgs = images(1024)[:13]
    for z in (1.25, 1.8, 2.5, 4.0):
        for k, rgb in enumerate(imgs):
            h, w = rgb.shape[:2]
            fg = st.ex.extract_array(rgb)
            A = fg[:, :, 3]
            cfg = core.Config()
            t0 = time.perf_counter()
            B = core.reconstruct_background(rgb, core.build_inpaint_mask(A, None, cfg), st.lama, cfg)
            sync(GPU)
            t1 = time.perf_counter()
            Bz = core.magnify(B, core.anchored_window(h, w, z, core.contact_point(A)), st.up, cfg)
            sync(GPU)
            t2 = time.perf_counter()
            core.composite(fg[:, :, :3], A, Bz, cfg)
            t3 = time.perf_counter()
            rows.append(dict(kind="zoom", L=1024, z=z, k=k, zoom_ms=(t2 - t1) * 1000,
                             rest_ms=(t1 - t0 + t3 - t2) * 1000))
            print("zoom", z, k, round((t2 - t1) * 1000), flush=True)
    write_csv(rows, "e6_scaling.csv")


# ------------------------------------------------------------ latency (HTTP)
def post_multipart(url, fields, files):
    import uuid as _u
    b = _u.uuid4().hex
    body = bytearray()
    for k, v in fields.items():
        body += f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
    for k, (fn, data, ct) in files.items():
        body += (f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"; filename=\"{fn}\"\r\n"
                 f"Content-Type: {ct}\r\n\r\n").encode() + data + b"\r\n"
    body += f"--{b}--\r\n".encode()
    req = urllib.request.Request(url, data=bytes(body),
                                 headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def post_json(url, obj):
    req = urllib.request.Request(url, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def exp_latency():
    rows = []
    imgs = images(1024)[:13]
    for dev in (GPU, "cpu"):
        env = dict(os.environ, SR_DEVICE=dev, LAMA_DEVICE=dev, SAM_DEVICE=dev,
                   EXTRACT_DEVICE=dev, INPAINT_BACKEND="lama")
        if dev == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""       # nothing may fall back to the GPU
        port = "10017"
        srv = subprocess.Popen([sys.executable, "-c",
                                f"import app; app.app.run(host='127.0.0.1', port={port}, threaded=True)"],
                               cwd=PROJECT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        base = f"http://127.0.0.1:{port}"
        for _ in range(120):
            try:
                urllib.request.urlopen(base + "/", timeout=2)
                break
            except Exception:
                time.sleep(1)
        try:
            for k, rgb in enumerate(imgs):
                buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))[1].tobytes()
                t = time.perf_counter()
                seg = post_multipart(base + "/segment_init", {}, {"file": ("x.jpg", buf, "image/jpeg")})
                t_init = time.perf_counter() - t
                h, w = rgb.shape[:2]
                pts = []
                for j in range(5):
                    t = time.perf_counter()
                    post_json(base + "/segment", dict(seg_id=seg["seg_id"],
                                                      points=[[w // 2 + 5 * j, h // 2]], labels=[1]))
                    pts.append(time.perf_counter() - t)
                t = time.perf_counter()
                job = post_multipart(base + "/process", {"zoom": "2.0"},
                                     {"file": ("x.jpg", buf, "image/jpeg")})
                while True:
                    with urllib.request.urlopen(base + f"/status/{job['job_id']}") as r:
                        st = json.load(r)
                    if st["status"] in ("done", "error"):
                        break
                    time.sleep(0.05)
                t_proc = time.perf_counter() - t
                with urllib.request.urlopen(base + st["result_url"]) as r:
                    r.read()
                t_total = time.perf_counter() - t
                rows.append(dict(device=dev, k=k, init_s=t_init, prompt_ms=np.mean(pts) * 1000,
                                 prompt_ms_list=json.dumps([round(p * 1000, 1) for p in pts]),
                                 process_s=t_proc, result_s=t_total, status=st["status"]))
                print(rows[-1], flush=True)
        finally:
            srv.terminate()
            srv.wait()
    write_csv(rows, "e6_latency.csv")


def hardware():
    import platform
    info = dict(platform=platform.platform(), processor=platform.processor(),
                python=platform.python_version(), torch=torch.__version__, gpu=GPU)
    if GPU == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_mem_gb"] = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        info["cuda"] = torch.version.cuda
    with open(os.path.join(RESULTS, "e6_hardware.json"), "w") as f:
        json.dump(info, f, indent=1)


if __name__ == "__main__":
    hardware()
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    a = ap.parse_args()
    torch.manual_seed(0)
    globals()["exp_" + a.exp]()
