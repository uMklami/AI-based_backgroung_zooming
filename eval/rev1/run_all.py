"""
Run every revision experiment that is not yet complete, then build the
tables and figures. Completed results (CSV with the expected number of
rows) are skipped, so this can be interrupted and re-run at any time.

    python eval/rev1/setup_rev1.py      # once
    python eval/rev1/run_all.py         # everything else
    python eval/rev1/run_all.py --list  # show what is done / pending

Logs go to SELZOOM_REV/logs (default ~/selzoom_rev/logs).
Runtime experiments (E6) run last, on an otherwise idle machine.
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(HERE, "results")
LOGS = os.path.join(os.environ.get("SELZOOM_REV", os.path.expanduser("~/selzoom_rev")), "logs")

# (name, csv, expected rows, script, args)
STEPS = [
    ("E1 DUTS-TE", "e1_duts.csv", 500, "e1_run_sets.py", ["duts", "--save"]),
    ("E1 AIM-500", "e1_aim500.csv", 500, "e1_run_sets.py", ["aim500", "--save"]),
    ("E1 DIS-TE", "e1_dis_te.csv", 400, "e1_run_sets.py", ["dis_te", "--save"]),
    ("E1 COD10K", "e1_cod10k.csv", 300, "e1_run_sets.py", ["cod10k", "--save"]),
    ("E1 ExDark", "e1_exdark.csv", 150, "e1_run_sets.py", ["exdark", "--save"]),
    ("E1 WildDepth", "e1_wilddepth.csv", 142, "e1_run_sets.py", ["wilddepth", "--save"]),
    ("Attributes", "attr_indoor_outdoor.csv", 1992, "attrs.py", []),
    ("E2 extractors AIM-500", "e2_aim500.csv", 3500, "e2_extractors.py", ["aim500"]),
    ("E2 extractors DUTS-TE", "e2_duts.csv", 3500, "e2_extractors.py", ["duts"]),
    ("E2 extractor timing (CPU)", "e2_time_aim500.csv", 7, "e2_extractors.py", ["aim500", "--time", "30"]),
    ("E5 zoom sweep", "e5_zoom.csv", 4800, "e5_synth_ablations.py", ["zoom"]),
    ("E5 processing order", "e5_order.csv", 800, "e5_synth_ablations.py", ["order"]),
    ("E5 composition (GT matte)", "e5_comp.csv", 800, "e5_synth_ablations.py", ["comp"]),
    ("E5 mask size/shape", "e5_mask.csv", 2000, "e5_synth_ablations.py", ["mask"]),
    ("E5 perturbation", "e5_perturb.csv", 3000, "e5_synth_ablations.py", ["perturb"]),
    ("E5 working resolution", "e5_res.csv", 1000, "e5_synth_ablations.py", ["res"]),
    ("E5 stage ablation", "e5_stage.csv", 2600, "e5_synth_ablations.py", ["stage"]),
    ("E5 composition (auto matte)", "e5_comp_auto.csv", 800, "e5_synth_ablations.py", ["comp_auto"]),
    ("E6 modules", "e6_modules.csv", 70, "e6_runtime.py", ["modules"]),
    ("E6 scaling", "e6_scaling.csv", 104, "e6_runtime.py", ["scaling"]),
    ("E6 latency", "e6_latency.csv", 26, "e6_runtime.py", ["latency"]),
]


def rows(csv):
    p = os.path.join(RES, csv)
    if not os.path.exists(p):
        return 0
    with open(p) as f:
        return max(0, sum(1 for _ in f) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", default=None, help="substring filter on step names")
    a = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    for name, csv, n, script, args in STEPS:
        have = rows(csv)
        done = have >= n
        if a.list:
            print(f"{'DONE   ' if done else 'PENDING'} {name:32s} {have}/{n}  ({csv})")
            continue
        if done or (a.only and a.only.lower() not in name.lower()):
            continue
        log = os.path.join(LOGS, csv.replace(".csv", ".log"))
        print(f"[run_all] {name} -> {log}", flush=True)
        t = time.time()
        with open(log, "w") as f:
            r = subprocess.call([sys.executable, os.path.join(HERE, script)] + args,
                                cwd=PROJECT, stdout=f, stderr=subprocess.STDOUT)
        print(f"[run_all]   exit {r}, {rows(csv)}/{n} rows, {(time.time() - t) / 60:.1f} min",
              flush=True)
    if not a.list:
        subprocess.call([sys.executable, os.path.join(HERE, "make_tables.py")], cwd=PROJECT,
                        stdout=open(os.path.join(RES, "tables.txt"), "w"), stderr=subprocess.STDOUT)
        subprocess.call([sys.executable, os.path.join(HERE, "make_figures.py"), "all"], cwd=PROJECT)
        print("[run_all] tables -> eval/rev1/results/tables.txt and summary.json; "
              "figures -> BG-Zooming/media/rev1_*.png")


if __name__ == "__main__":
    main()
