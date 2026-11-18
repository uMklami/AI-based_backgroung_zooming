"""
Aggregate the per-image CSVs in results/ into the numbers reported in the
paper. Prints every table and writes results/summary.json.

    python eval/rev1/make_tables.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
S = {}


PSNR_CAP = 50.0   # dB; exact reconstructions of flat regions give inf / ~99 dB


def rd(name):
    p = os.path.join(R, name)
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    for c in d.columns:
        if "psnr" in c and c not in ("up_lan_psnr", "up_sr_psnr", "up_ad_psnr"):
            d[c] = d[c].clip(upper=PSNR_CAP)
    return d


def dist(x):
    x = pd.Series(x).dropna()
    return dict(mean=x.mean(), sd=x.std(), median=x.median(), min=x.min(), max=x.max(), n=len(x))


def show(title, obj):
    print(f"\n==== {title}")
    if isinstance(obj, pd.DataFrame):
        print(obj.round(4).to_string())
    else:
        print(json.dumps(obj, indent=1, default=float))


# ------------------------------------------------ Tables 3-6 (DUTS-TE, E1)
d = rd("e1_duts.csv")
if d is not None:
    t3 = {k: dist(d[k]) for k in ("iou", "f1")}
    t3["share_f1_ge_0.9"] = float((d.f1 >= 0.9).mean())
    t3["share_f1_lt_0.5"] = float((d.f1 < 0.5).mean())
    t3["n_f1_lt_0.5"] = int((d.f1 < 0.5).sum())
    t4 = {k: dist(d[k]) for k in ("inp_psnr", "inp_ssim", "inp_lpips")}
    t5 = {}
    for k in ("lan", "sr", "ad"):
        t5[k] = dict(psnr=dist(d[f"up_{k}_psnr"]), lpips=dist(d[f"up_{k}_lpips"]))
    t5["sr_better_lpips_than_lan"] = float((d.up_sr_lpips < d.up_lan_lpips).mean())
    t5["ad_better_lpips_than_lan"] = float((d.up_ad_lpips < d.up_lan_lpips).mean())
    t5["lan_better_psnr_than_ad"] = float((d.up_lan_psnr > d.up_ad_psnr).mean())
    t5["lan_better_psnr_than_sr"] = float((d.up_lan_psnr > d.up_sr_psnr).mean())
    t6 = {k: dist(d[k]) for k in ("dev_lpips", "dev_ssim", "dev_psnr")}
    S.update(t3=t3, t4=t4, t5=t5, t6=t6)
    S["duts_size"] = dict(long_side=dist(d[["h", "w"]].max(axis=1)),
                          share_gt_1024=float((d[["h", "w"]].max(axis=1) >= 1024).mean()))
    S["duts_misc"] = dict(leak=dist(d.leak), carry=dist(d.carry), oracle_gap=dist(d.oracle_gap),
                          anchor_shift=dist(d.anchor_shift_px), drift_center=dist(d.drift_center_px))
    for k in ("t3", "t4", "t5", "t6", "duts_size", "duts_misc"):
        show(k, S[k])

    # Table 10 left: bins by F1
    bins = [(0.95, 1.01), (0.90, 0.95), (0.80, 0.90), (0.50, 0.80), (0.0, 0.50)]
    rows = []
    for lo, hi in bins:
        m = (d.f1 >= lo) & (d.f1 < hi)
        rows.append(dict(bin=f"[{lo},{min(hi, 1.0)})", n=int(m.sum()), leak=d.leak[m].mean(),
                         carry=d.carry[m].mean(), oracle_gap=d.oracle_gap[m].mean()))
    t10l = pd.DataFrame(rows)
    from scipy.stats import spearmanr
    rho, pv = spearmanr(d.f1, d.oracle_gap, nan_policy="omit")
    S["t10_left"] = t10l.to_dict("records")
    S["t10_spearman"] = dict(rho=rho, p=pv)
    show("t10 left", t10l)
    show("spearman", S["t10_spearman"])

    # Table 9: challenging categories (DUTS-based rows)
    import json as _j
    tags = _j.load(open(os.path.join(HERE, "tags_duts_cast_shadow.json")))
    sh = d.name.isin(tags["names"])
    S["t9_shadow"] = dict(n=int(sh.sum()), f1=d.f1[sh].mean(), leak=d.leak[sh].mean(),
                          carry=d.carry[sh].mean(), oracle_gap=d.oracle_gap[sh].mean())
    if "n_comp" in d:
        pass

# ------------------------------------------------ Table 9 / 11: other sets
sets = {"aim500": "AIM-500", "dis_te": "DIS-TE", "cod10k": "COD10K-test",
        "exdark": "ExDark", "wilddepth": "WildDepth"}
cross = []
frames = []
for key, label in [("duts", "DUTS-TE")] + list(sets.items()):
    d2 = rd(f"e1_{key}.csv")
    if d2 is None:
        continue
    d2["set"] = label
    frames.append(d2)
    row = dict(set=label, n=len(d2))
    for c in ("f1", "inp_psnr", "inp_lpips", "oracle_gap", "leak", "carry"):
        row[c] = d2[c].mean() if c in d2 else np.nan
    row["sr_gain"] = (d2.up_lan_lpips - d2.up_ad_lpips).mean()
    row["f1_lt_0.5"] = (d2.f1 < 0.5).mean() if "f1" in d2 else np.nan
    cross.append(row)
    if key == "aim500" and "type" in d2:
        stm = d2[d2.type == "STM"]
        S["t9_stm"] = dict(n=len(stm), f1=stm.f1.mean(), leak=stm.leak.mean(),
                           carry=stm.carry.mean(), oracle_gap=stm.oracle_gap.mean())
if cross:
    S["t11_sources"] = cross
    show("t11 sources", pd.DataFrame(cross))
    allf = pd.concat(frames, ignore_index=True)
    area = allf.gt_area.fillna(allf.pred_area)
    attrs = {
        "small (<5%)": area < 0.05, "medium (5-25%)": (area >= 0.05) & (area <= 0.25),
        "large (>25%)": area > 0.25,
        "normal light": (allf.luminance >= 0.25) & (allf.set != "ExDark"),
        "low light": (allf.luminance < 0.25) | (allf.set == "ExDark"),
        "low clutter": allf.clutter <= allf.clutter.median(),
        "high clutter": allf.clutter > allf.clutter.median(),
    }
    io = rd("attr_indoor_outdoor.csv")
    if io is not None:
        allf = allf.merge(io[["set", "name", "indoor"]], on=["set", "name"], how="left")
        attrs["indoor"] = allf.indoor == 1
        attrs["outdoor"] = allf.indoor == 0
    rows = []
    for k, m in attrs.items():
        m = m.values if hasattr(m, "values") else m
        sub = allf[m]
        rows.append(dict(attr=k, n=len(sub), f1=sub.f1.mean(), inp_psnr=sub.inp_psnr.mean(),
                         inp_lpips=sub.inp_lpips.mean(),
                         sr_gain=(sub.up_lan_lpips - sub.up_ad_lpips).mean(),
                         oracle_gap=sub.oracle_gap.mean()))
    S["t11_attrs"] = rows
    S["clutter_median"] = float(allf.clutter.median())
    show("t11 attributes", pd.DataFrame(rows))
    for key, label in (("dis_te", "DIS-TE"), ("cod10k", "COD10K-test")):
        sub = allf[allf.set == label]
        if len(sub):
            S[f"t9_{key}"] = dict(n=len(sub), f1=sub.f1.mean(), leak=sub.leak.mean(),
                                  carry=sub.carry.mean(), oracle_gap=sub.oracle_gap.mean())
    # multiple objects (>= 2 GT components >= 0.5% of the frame), DUTS + AIM
    mc = rd("attr_multi.csv")
    if mc is not None:
        mm = allf.merge(mc, on=["set", "name"], how="inner")
        mm = mm[(mm.multi == 1) & mm.set.isin(["DUTS-TE", "AIM-500"])]
        S["t9_multi"] = dict(n=len(mm), f1=mm.f1.mean(), leak=mm.leak.mean(),
                             carry=mm.carry.mean(), oracle_gap=mm.oracle_gap.mean())
    show("t9", {k: v for k, v in S.items() if k.startswith("t9")})

# ------------------------------------------------ Tables 7-8 (E2)
for key in ("duts", "aim500"):
    d2 = rd(f"e2_{key}.csv")
    if d2 is None:
        continue
    cols = ["f1", "iou"] + (["sad", "mse", "grad", "conn"] if key == "aim500" else [])
    g = d2.groupby("variant")[cols].mean()
    g["n"] = d2.groupby("variant").size()
    S[f"t7_{key}"] = g.reset_index().to_dict("records")
    show(f"t7 {key}", g)
    if key == "aim500":
        t8 = d2.pivot_table(index="variant", columns="type", values="sad", aggfunc="mean")
        S["t8"] = t8.reset_index().to_dict("records")
        show("t8 SAD by type", t8)
tt = rd("e2_time_aim500.csv")
if tt is not None:
    S["t7_time"] = tt.to_dict("records")
    show("t7 time", tt)

# ------------------------------------------------ SynthZoom (E5)
d2 = rd("e5_mask.csv")
if d2 is not None:
    g = d2.groupby("config")[["hole_psnr", "hole_lpips", "ghost", "lpips", "bnd_lpips", "t_inp_ms"]].mean()
    S["t12"] = g.reset_index().to_dict("records")
    show("t12 mask", g)
d2 = rd("e5_zoom.csv")
if d2 is not None:
    g = d2.pivot_table(index="z", columns="mode", values=["lpips", "psnr", "t_ms"], aggfunc="mean")
    S["t13"] = {str(k): v for k, v in g.to_dict().items()}
    show("t13 zoom", g)
    best = d2.groupby(["z", "mode"]).lpips.mean().unstack().idxmin(axis=1)
    show("t13 best by z", best.to_dict())
d2 = rd("e5_stage.csv")
if d2 is not None:
    g = d2.groupby("config")[["psnr", "ssim", "lpips", "hole_lpips", "bnd_lpips", "drift_px", "t_ms"]].mean()
    if "f1" in d2:
        g["f1"] = d2.groupby("config").f1.mean()
    g["n"] = d2.groupby("config").size()
    S["t14"] = g.reset_index().to_dict("records")
    show("t14 stage", g)
d2 = rd("e5_order.csv")
if d2 is not None:
    g = d2.groupby("order")[["psnr", "lpips", "t_ms", "mem_gb"]].agg(["mean"])
    g.columns = [c[0] for c in g.columns]
    S["t15"] = g.reset_index().to_dict("records")
    show("t15 order", g)
d2 = rd("e5_comp.csv")
if d2 is not None:
    g = d2.groupby("method")[["bnd_psnr", "bnd_lpips", "lpips", "de00", "t_ms"]].mean()
    S["t16"] = g.reset_index().to_dict("records")
    show("t16 comp", g)
d2 = rd("e5_res.csv")
if d2 is not None:
    g = d2.groupby("L")[["hole_psnr", "hole_lpips", "full_lpips", "t_ms"]].mean()
    S["t17"] = g.reset_index().to_dict("records")
    show("t17 res", g)
d2 = rd("e5_perturb.csv")
if d2 is not None:
    bins = [(0.95, 1.01), (0.90, 0.95), (0.80, 0.90), (0.50, 0.80), (0.0, 0.50)]
    rows = []
    for lo, hi in bins:
        m = (d2.f1 >= lo) & (d2.f1 < hi) & (d2.kind != "none")
        rows.append(dict(bin=f"[{lo},{min(hi, 1.0)})", n=int(m.sum()), lpips=d2.lpips[m].mean(),
                         hole_lpips=d2.hole_lpips[m].mean(), bnd_lpips=d2.bnd_lpips[m].mean(),
                         ghost=d2.ghost[m].mean()))
    S["t10_right"] = rows
    show("t10 right", pd.DataFrame(rows))
    g = d2.groupby(["kind", "level"])[["f1", "lpips", "bnd_lpips", "hole_lpips", "ghost", "anchor_shift_px"]].mean()
    S["perturb_by_kind"] = g.reset_index().to_dict("records")
    show("perturb by kind", g)

# ------------------------------------------------ Runtime (E6)
d2 = rd("e6_modules.csv")
if d2 is not None:
    per = d2[d2.k >= 3]
    cols = [c for c in per.columns if c.endswith("_ms") or c.endswith("_gb")]
    g = per.groupby("device")[cols].agg(["mean", "std"])
    S["t18"] = {f"{a}|{b}|{c}": v for (a, b), s in g.items() for c, v in s.items()} if False else None
    S["t18_mean"] = per.groupby("device")[cols].mean().to_dict("index")
    S["t18_sd"] = per.groupby("device")[cols].std().to_dict("index")
    S["t18_thr"] = d2[d2.k < 0][["device", "workers", "throughput", "total_gb"]].to_dict("records")
    show("t18 mean", per.groupby("device")[cols].mean().T)
    show("t18 sd", per.groupby("device")[cols].std().T)
    show("t18 thr", d2[d2.k < 0])
d2 = rd("e6_scaling.csv")
if d2 is not None:
    r = d2[(d2.kind == "res") & (d2.k >= 3)].groupby("L")[["total_ms", "gb"]].mean()
    z = d2[(d2.kind == "zoom") & (d2.k >= 3)].groupby("z")[["zoom_ms", "rest_ms"]].mean()
    S["t19_res"] = r.reset_index().to_dict("records")
    S["t19_zoom"] = z.reset_index().to_dict("records")
    show("t19 res", r)
    show("t19 zoom", z)
d2 = rd("e6_latency.csv")
if d2 is not None:
    per = d2[d2.k >= 3]
    lat = {}
    for dev, g in per.groupby("device"):
        pl = np.concatenate([json.loads(x) for x in g.prompt_ms_list])
        lat[dev] = dict(init_med=g.init_s.median(), init_p95=g.init_s.quantile(0.95),
                        prompt_med=np.median(pl), prompt_p95=np.percentile(pl, 95),
                        proc_med=g.process_s.median(), proc_p95=g.process_s.quantile(0.95),
                        result_med=g.result_s.median(), result_p95=g.result_s.quantile(0.95),
                        n=len(g))
    S["t20"] = lat
    show("t20", lat)

with open(os.path.join(R, "summary.json"), "w") as f:
    json.dump(S, f, indent=1, default=float)
print("\nwrote summary.json")
