#!/usr/bin/env python
"""RC-3: multi-generation confidence for RAP registration.

Idea: RAP sampling is stochastic. Running k independent generations of the
SAME pair and measuring the inter-generation pose dispersion turns RAP's
sampling randomness into a free uncertainty estimate. This script tests:
  Q1: does dispersion correlate with the edge's true relative-pose error
      (vs GT)? (RC-1 found overlap/mean_dist correlate only weakly,
      r = -0.20 / +0.25.)
  Q2: does dispersion separate TRUE loops from FALSE loops better than the
      overlap gate (esp. the 0.44-0.50 overlap urban-street traps)?
  Q3: does a dispersion-scaled PGO noise model (and/or the best-of-k
      estimate) beat the current relfloor+Huber default
      (KITTI-02 26.423 m, POSS 1.757 m golden)?

Phase "sample" (GPU, WSL conda env rap-inference): k generations per pair,
re-seeded per generation for reproducibility, resumable JSON cache.
Phase "analyze" (CPU): correlations, gate separation, PGO variants.

Run:
  python scripts/eval_multigen_confidence.py                 # both phases
  python scripts/eval_multigen_confidence.py --phase sample  # GPU only
  python scripts/eval_multigen_confidence.py --phase analyze # CPU only
"""

import argparse
import json
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # eval_pgo_robust

KITTI_SEQ_DIR = "/mnt/e/githouse/Datasets/data_odometry_velodyne/dataset/sequences/02"
POSS_VEL_DIR = os.path.join(REPO_ROOT, "data", "poss", "dataset",
                            "sequences", "00", "velodyne")
POSS_OFFSET = 3850  # live POSS indices are local; scan file = local + 3850

CACHE_DEFAULT = os.path.join(REPO_ROOT, "output_multigen", "multigen_samples.json")


# ----------------------------------------------------------------------
# Pair lists
# ----------------------------------------------------------------------
def pair_lists():
    acc = json.load(open(os.path.join(
        REPO_ROOT, "output_ros2", "step3_kitti02", "accepted_loops.json")))
    false = json.load(open(os.path.join(
        REPO_ROOT, "output_ros2", "step3_kitti02_bigbudget",
        "accepted_loops.json")))
    poss = json.load(open(os.path.join(
        REPO_ROOT, "output_ros2", "step3", "accepted_loops.json")))
    pairs = []
    for lc in acc:
        pairs.append({"ds": "kitti02_true", "i": lc["i"], "j": lc["j"]})
    for lc in false:
        pairs.append({"ds": "kitti02_false", "i": lc["i"], "j": lc["j"]})
    for lc in poss:
        pairs.append({"ds": "poss_true", "i": lc["i"], "j": lc["j"]})
    return pairs


def load_scan(ds, idx):
    if ds.startswith("kitti02"):
        path = os.path.join(KITTI_SEQ_DIR, "velodyne", f"{idx:06d}.bin")
    else:
        path = os.path.join(POSS_VEL_DIR, f"{idx + POSS_OFFSET:06d}.bin")
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]


# ----------------------------------------------------------------------
# Phase 1: multi-generation sampling (GPU)
# ----------------------------------------------------------------------
def phase_sample(args):
    import torch
    from rap_slam.rap_backend import RAPBackend

    pairs = pair_lists()
    cache = {}
    if os.path.exists(args.cache) and not args.resample:
        saved = json.load(open(args.cache))
        cache = {g["key"]: g for g in saved["entries"]}
    print(f"pairs: {len(pairs)} total, {len(cache)} already cached, k={args.k}")

    rap = RAPBackend()  # defaults: rap_12 ckpt, voxel 0.4, 4000 pts, spinnet
    entries = list(cache.values())
    t0 = time.time()
    n_done = 0
    for p in pairs:
        key = f"{p['ds']}|{p['i']}|{p['j']}"
        if key in cache:
            n_done += 1
            continue
        sa = load_scan(p["ds"], p["i"])
        sb = load_scan(p["ds"], p["j"])
        gens = []
        for g in range(args.k):
            # Reproducible per-generation re-seed (RNG was seeded once in
            # RAPBackend.__init__; consecutive calls would also differ, but
            # explicit seeds make the cache reproducible).
            sd = args.seed_base * 100003 + p["i"] * 977 + p["j"] * 31 + g
            torch.manual_seed(sd)
            torch.cuda.manual_seed_all(sd)
            np.random.seed(sd % (2**31 - 1))
            T_rel, diag = rap.register(sa, sb)
            gens.append({
                "T_rel": np.asarray(T_rel).tolist(),
                "overlap": float(diag["overlap_ratio"]),
                "mean_dist": float(diag["mean_dist"]),
                "t_inference": float(diag["t_inference"]),
            })
        entries.append({"key": key, "ds": p["ds"], "i": p["i"], "j": p["j"],
                        "gens": gens})
        n_done += 1
        if n_done % 10 == 0 or n_done == len(pairs):
            el = time.time() - t0
            eta = el / max(n_done - len(cache), 1) * (len(pairs) - n_done)
            print(f"  [{n_done}/{len(pairs)}] {key} done | "
                  f"elapsed {el/60:.1f} min, ETA {eta/60:.1f} min", flush=True)
            _save_cache(args.cache, entries)

    _save_cache(args.cache, entries)
    print(f"sample phase done: {len(entries)} entries -> {args.cache}")


def _save_cache(path, entries):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"entries": entries}, f)
    os.replace(tmp, path)


# ----------------------------------------------------------------------
# Dispersion metrics
# ----------------------------------------------------------------------
def rot_angle_deg(R):
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def dispersion(gens):
    """Mean pairwise inter-generation disagreement.

    disp_t: mean pairwise |t_a - t_b| over the k(k-1)/2 generation pairs (m)
    disp_r: mean pairwise angular distance R_a^-1 R_b (deg)
    Also returns best-of-k index (max overlap, tie -> min mean_dist).
    """
    Ts = [np.asarray(g["T_rel"]) for g in gens]
    k = len(Ts)
    dt, dr = [], []
    for a in range(k):
        for b in range(a + 1, k):
            dt.append(float(np.linalg.norm(Ts[a][:3, 3] - Ts[b][:3, 3])))
            dr.append(rot_angle_deg(Ts[a][:3, :3].T @ Ts[b][:3, :3]))
    ov = [g["overlap"] for g in gens]
    md = [g["mean_dist"] for g in gens]
    best = max(range(k), key=lambda x: (ov[x], -md[x]))
    return {
        "disp_t": float(np.mean(dt)), "disp_r": float(np.mean(dr)),
        "best_idx": int(best),
        "overlap_best": float(ov[best]), "mean_dist_best": float(md[best]),
        "overlap_mean": float(np.mean(ov)), "overlap_min": float(np.min(ov)),
        "mean_dist_mean": float(np.mean(md)),
        "T_rel_best": Ts[best].tolist(),
    }


# ----------------------------------------------------------------------
# Phase 2: analysis (CPU)
# ----------------------------------------------------------------------
def phase_analyze(args):
    from scipy.stats import spearmanr
    import eval_pgo_robust as epr
    from rap_slam import evaluate, real_dataset

    saved = json.load(open(args.cache))
    entries = saved["entries"]
    for e in entries:
        e.update(dispersion(e["gens"]))
    print(f"loaded {len(entries)} sampled pairs "
          f"(k={len(entries[0]['gens'])} generations)")

    # GT for KITTI-02 (full seq, calib applied) and POSS (offset segment).
    kitti_root = "/mnt/e/githouse/Datasets/data_odometry_velodyne/dataset"
    gt_k = evaluate.align_to_first_pose(list(real_dataset.load_gt_poses(
        os.path.join(kitti_root, "poses/02.txt"),
        os.path.join(kitti_root, "sequences/02/calib.txt"),
        start=0, end=4660, apply_calib=True)))
    poss_root = os.path.join(REPO_ROOT, "data", "poss", "dataset")
    gt_p = evaluate.align_to_first_pose(list(real_dataset.load_gt_poses(
        os.path.join(poss_root, "poses", "00.txt"),
        None, start=POSS_OFFSET, end=POSS_OFFSET + 1418, apply_calib=False)))

    def gt_err(e):
        gt = gt_p if e["ds"] == "poss_true" else gt_k
        T_gt = np.linalg.inv(gt[e["i"]]) @ gt[e["j"]]
        terrs, rerrs = [], []
        for g in e["gens"]:
            dG = np.linalg.inv(T_gt) @ np.asarray(g["T_rel"])
            terrs.append(float(np.linalg.norm(dG[:3, 3])))
            rerrs.append(rot_angle_deg(dG[:3, :3]))
        dGb = np.linalg.inv(T_gt) @ np.asarray(e["T_rel_best"])
        return (float(np.mean(terrs)), float(np.mean(rerrs)),
                float(np.linalg.norm(dGb[:3, 3])))

    for e in entries:
        e["te_mean"], e["re_mean"], e["te_best"] = gt_err(e)

    kt = [e for e in entries if e["ds"] == "kitti02_true"]
    kf = [e for e in entries if e["ds"] == "kitti02_false"]

    # ------------------------------------------------------------------
    print("\n=== Q1: dispersion vs true edge error (KITTI-02, "
          f"{len(kt)} accepted edges) ===")
    te = np.array([e["te_mean"] for e in kt])
    for name, x in (("disp_t", [e["disp_t"] for e in kt]),
                    ("disp_r", [e["disp_r"] for e in kt]),
                    ("overlap_mean", [e["overlap_mean"] for e in kt]),
                    ("mean_dist_mean", [e["mean_dist_mean"] for e in kt])):
        rho, p = spearmanr(x, te)
        print(f"  spearman({name:>14s}, trans_err) = {rho:+.3f} (p={p:.2e})")
    te_b = np.array([e["te_best"] for e in kt])
    print(f"  true trans err: single-gen mean {te.mean():.3f} m vs "
          f"best-of-k {te_b.mean():.3f} m (median {np.median(te):.3f} -> "
          f"{np.median(te_b):.3f})")

    # ------------------------------------------------------------------
    print(f"\n=== Q2: gate separation — {len(kt)} true vs {len(kf)} false ===")
    for name, f in (("overlap (live single-gen style: best-of-k)",
                     lambda e: e["overlap_best"]),
                    ("overlap_min across gens", lambda e: e["overlap_min"]),
                    ("disp_t", lambda e: e["disp_t"]),
                    ("disp_r", lambda e: e["disp_r"]),
                    ("mean_dist_best", lambda e: e["mean_dist_best"])):
        tv = np.array([f(e) for e in kt])
        fv = np.array([f(e) for e in kf])
        print(f"  {name:>38s}: TRUE [{tv.min():.3f}..{tv.max():.3f}] "
              f"FALSE [{fv.min():.3f}..{fv.max():.3f}]")
    # False-pair detail: does dispersion flag the high-overlap traps?
    print("  false pairs (overlap_best, disp_t, disp_r, GT trans err):")
    for e in sorted(kf, key=lambda x: -x["overlap_best"]):
        print(f"    ({e['i']:5d},{e['j']:5d}) ov {e['overlap_best']:.3f} "
              f"disp_t {e['disp_t']:.3f} m disp_r {e['disp_r']:.2f} deg "
              f"GT err {e['te_mean']:.1f} m")

    # ------------------------------------------------------------------
    print("\n=== Q3: PGO with dispersion-scaled noise / best-of-k ===")
    import gtsam
    from gtsam import noiseModel

    def run_dataset(tag, odom_path, save_dir, gt_full):
        odom_poses = epr.load_tum(odom_path)
        keyframes = json.load(open(os.path.join(save_dir, "keyframes.json")))
        loops = json.load(open(os.path.join(save_dir, "accepted_loops.json")))
        gt_rel = evaluate.align_to_first_pose(list(gt_full))
        n = len(odom_poses)
        m_odom = evaluate.compute_ape_rpe(
            gt_rel[:n], evaluate.align_to_first_pose(odom_poses))
        print(f"  [{tag}] odom APE {m_odom['ape_rmse']:.3f} m, "
              f"{len(loops)} loop edges")

        disp_map = {}
        for e in entries:
            disp_map[(e["i"], e["j"])] = e

        def huber(k, base_fn):
            def f(lc):
                return noiseModel.Robust.Create(
                    noiseModel.mEstimator.Huber(k), base_fn(lc))
            return f

        def relfloor(lc):  # current default, reproduces 26.423 / 1.757
            md = float(lc.get("mean_dist", 0.5))
            s = float(np.clip(md / 1.0, 1.0, 3.0))
            return noiseModel.Diagonal.Sigmas(
                np.array(epr.LOOP_SIGMAS_BASE) * s)

        def make_disp_rel(ref, cap):
            def f(lc):
                e = disp_map.get((lc["i"], lc["j"]))
                s = 1.0 if e is None else float(np.clip(e["disp_t"] / ref,
                                                        1.0, cap))
                return noiseModel.Diagonal.Sigmas(
                    np.array(epr.LOOP_SIGMAS_BASE) * s)
            return f

        def make_comb(md_ref, d_ref, cap):
            def f(lc):
                e = disp_map.get((lc["i"], lc["j"]))
                s_md = float(np.clip(float(lc.get("mean_dist", 0.5)) / md_ref,
                                     1.0, cap))
                s_d = 1.0 if e is None else float(np.clip(
                    e["disp_t"] / d_ref, 1.0, cap))
                return noiseModel.Diagonal.Sigmas(
                    np.array(epr.LOOP_SIGMAS_BASE) * max(s_md, s_d))
            return f

        # best-of-k loops: replace T_rel/overlap/mean_dist by best generation
        loops_bk = []
        for lc in loops:
            lc2 = dict(lc)
            e = disp_map.get((lc["i"], lc["j"]))
            if e is not None:
                lc2["T_rel"] = e["T_rel_best"]
                lc2["mean_dist"] = e["mean_dist_best"]
                lc2["overlap"] = e["overlap_best"]
            loops_bk.append(lc2)

        results = {}
        disp_refs = (0.05, 0.1, 0.2) if tag.startswith("kitti") else (0.05,)
        variants = {"relfloor+huber (default)": huber(1.0, relfloor)}
        for r in disp_refs:
            for c in (3.0, 5.0):
                variants[f"disp_rel r{r}c{c}+huber"] = huber(
                    1.0, make_disp_rel(r, c))
        variants["comb md1.0+d0.1 c3+huber"] = huber(
            1.0, make_comb(1.0, 0.1, 3.0))
        for name, fn in variants.items():
            optimized, ne = epr.run_pgo(odom_poses, keyframes, loops, fn)
            m = epr.full_traj_ape(odom_poses, keyframes, optimized, gt_rel)
            results[name] = m["ape_rmse"]
            print(f"    {name:>34s}: APE rmse {m['ape_rmse']:7.3f} m")
        # best-of-k estimate with the two safest weightings
        for wname, fn in (("relfloor+huber", huber(1.0, relfloor)),
                          ("disp_rel r0.1c3+huber",
                           huber(1.0, make_disp_rel(0.1, 3.0)))):
            optimized, ne = epr.run_pgo(odom_poses, keyframes, loops_bk, fn)
            m = epr.full_traj_ape(odom_poses, keyframes, optimized, gt_rel)
            results[f"best-of-k T_rel + {wname}"] = m["ape_rmse"]
            print(f"    {'best-of-k T_rel + ' + wname:>34s}: "
                  f"APE rmse {m['ape_rmse']:7.3f} m")
        return results

    res_k = run_dataset(
        "kitti02",
        os.path.join(REPO_ROOT, "output_ros2", "live_poses_kitti02.txt"),
        os.path.join(REPO_ROOT, "output_ros2", "step3_kitti02"),
        list(real_dataset.load_gt_poses(
            os.path.join(kitti_root, "poses/02.txt"),
            os.path.join(kitti_root, "sequences/02/calib.txt"),
            start=0, end=4660, apply_calib=True)))
    res_p = run_dataset(
        "poss (golden)",
        os.path.join(REPO_ROOT, "output_ros2", "live_poses_step3.txt"),
        os.path.join(REPO_ROOT, "output_ros2", "step3"),
        list(real_dataset.load_gt_poses(
            os.path.join(poss_root, "poses", "00.txt"),
            None, start=POSS_OFFSET, end=POSS_OFFSET + 1418,
            apply_calib=False)))

    out = os.path.join(os.path.dirname(args.cache),
                       "multigen_analysis.json")
    summary = {
        "k": len(entries[0]["gens"]),
        "q1_spearman_vs_trans_err": {
            n: float(spearmanr([e[f] for e in kt],
                               [e["te_mean"] for e in kt])[0])
            for n, f in (("disp_t", "disp_t"), ("disp_r", "disp_r"),
                         ("overlap_mean", "overlap_mean"),
                         ("mean_dist_mean", "mean_dist_mean"))},
        "best_of_k_gain": {
            "single_gen_mean_te": float(np.mean([e["te_mean"] for e in kt])),
            "best_of_k_mean_te": float(np.mean([e["te_best"] for e in kt]))},
        "pgo_kitti02": res_k, "pgo_poss": res_p,
        "cost_per_pair_s": float(np.mean([
            g["t_inference"] for e in entries for g in e["gens"]])),
    }
    with open(out, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\nanalysis saved to {out}")
    t_inf = summary["cost_per_pair_s"]
    print(f"mean RAP inference {t_inf:.2f} s/gen -> "
          f"k={len(entries[0]['gens'])} costs ~{t_inf*len(entries[0]['gens']):.1f} s/pair")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["sample", "analyze", "both"],
                    default="both")
    ap.add_argument("--k", type=int, default=4, help="generations per pair")
    ap.add_argument("--seed-base", type=int, default=42)
    ap.add_argument("--cache", default=CACHE_DEFAULT)
    ap.add_argument("--resample", action="store_true")
    args = ap.parse_args()
    if args.phase in ("sample", "both"):
        phase_sample(args)
    if args.phase in ("analyze", "both"):
        phase_analyze(args)


if __name__ == "__main__":
    main()
