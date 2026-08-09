#!/usr/bin/env python
"""Direction B (RAP-CORE, owner-approved 2026-08-09, probe-first kill-fast):
place recognition from RAP's DiT encoder features vs Scan Context.

Pools RAP's PointCloudDiT per-point transformer features into per-keyframe
global descriptors (RAPBackend.extract_features — ONE t=0 forward pass with
x = cond, no flow sampling, deterministic per scan) and evaluates retrieval
separability against Scan Context.

Arms:
  rap-meanmax  : DiT transformer features, mean+max pooled (1024-D)
  rap-mean     : DiT transformer features, mean pooled (512-D)
  spinnet-mean : miniSpinNet 32-D input features, mean pooled (free byproduct
                 of the same preprocess)
  sc           : Scan Context baseline (20x60, shift-invariant distance)

Datasets:
  poss    — 82 keyframes (output_prods/poss products), true pairs = GT dist
            < 3 m & separation >= 200 (eval_loop_retrieval.py convention;
            SC reference: true 0.060/0.106 vs false >= 0.347)
  kitti02 — 812 keyframes (output_ros2/step3_kitti02 live products), true
            pairs = GT dist < 10 m & separation >= 200 (the known 27+ true
            revisit pairs; ALSO reports the strict < 3 m variant). This is
            the dataset where SC's false hits on repetitive urban streets
            exhausted the old retrieval budget — the real test.

Metrics per arm over all candidate pairs (sep >= 200):
  * separability: max true-pair distance vs min false-pair distance + margin
  * AUC (probability a true pair ranks closer than a false pair)
  * best-F1 threshold with precision / recall at that threshold

Descriptors are cached to npz so the analysis reruns GPU-free
(--reuse-cache). Run in the conda env:
  /root/miniconda3/envs/rap-inference/bin/python scripts/eval_rap_retrieval.py \
      --datasets poss,kitti02
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from rap_slam import real_dataset, scan_context as sc_mod  # noqa: E402

logger = logging.getLogger("eval_rap_retrieval")

KITTI_ROOT = "/mnt/e/githouse/Datasets/data_odometry_velodyne/dataset"
POSS_ROOT = "/mnt/e/githouse/RAP-SLAM/data/poss/dataset"

DATASETS = {
    "poss": dict(
        keyframes="output_prods/poss/keyframes.json",
        poses_file=os.path.join(POSS_ROOT, "poses/00.txt"),
        calib_file=os.path.join(POSS_ROOT, "sequences/00/calib.txt"),
        apply_calib=False,
        velodyne=os.path.join(POSS_ROOT, "sequences/00/velodyne"),
        start=3850, end=5268, file_offset=3850,
        true_dists=[3.0],
    ),
    "kitti02": dict(
        keyframes="output_ros2/step3_kitti02/keyframes.json",
        poses_file=os.path.join(KITTI_ROOT, "poses/02.txt"),
        calib_file=os.path.join(KITTI_ROOT, "sequences/02/calib.txt"),
        apply_calib=True,
        velodyne=os.path.join(KITTI_ROOT, "sequences/02/velodyne"),
        start=0, end=4660, file_offset=0,
        true_dists=[10.0, 3.0],
    ),
}


def parse_args():
    p = argparse.ArgumentParser(description="RAP-feature retrieval probe")
    p.add_argument("--datasets", default="poss,kitti02")
    p.add_argument("--loop-sep", type=int, default=200)
    p.add_argument("--sc-max-radius", type=float, default=80.0)
    p.add_argument("--sc-rings", type=int, default=20)
    p.add_argument("--sc-sectors", type=int, default=60)
    p.add_argument("--scan-min-range", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reuse-cache", action="store_true")
    p.add_argument("--output", default=os.path.join(REPO_ROOT,
                                                    "output_rap_retrieval"))
    return p.parse_args()


def load_scan(cfg, idx, min_range):
    path = os.path.join(cfg["velodyne"],
                        f"{idx + cfg['file_offset']:06d}.bin")
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]
    if min_range > 0:
        r = np.linalg.norm(pts, axis=1)
        pts = pts[r >= min_range]
    return pts.astype(np.float32)


def compute_descriptors(name, cfg, keyframes, args):
    """Per keyframe: rap-meanmax / rap-mean DiT descriptors, spinnet-mean,
    and Scan Context. Cached to npz."""
    cache = os.path.join(args.output, f"{name}_descs.npz")
    if args.reuse_cache and os.path.exists(cache):
        z = np.load(cache)
        logger.info(f"{name}: loaded cached descriptors ({cache})")
        return {"rap_meanmax": z["rap_meanmax"], "rap_mean": z["rap_mean"],
                "spinnet_mean": z["spinnet_mean"], "sc": z["sc"]}

    from rap_slam.rap_backend import RAPBackend
    rap = RAPBackend(seed=args.seed)
    n = len(keyframes)
    rap_meanmax = np.zeros((n, 1024), dtype=np.float32)
    rap_mean = np.zeros((n, 512), dtype=np.float32)
    spinnet_mean = np.zeros((n, 32), dtype=np.float32)
    scs = np.zeros((n, args.sc_rings, args.sc_sectors), dtype=np.float32)
    t0 = time.time()
    for k, idx in enumerate(keyframes):
        pts = load_scan(cfg, idx, args.scan_min_range)
        desc, diag = rap.extract_features(pts, pooling="meanmax")
        rap_meanmax[k] = desc
        rap_mean[k] = desc[:512]  # mean half of the mean+max concat
        spinnet_mean[k] = diag["spinnet_mean"]
        scs[k] = sc_mod.make_scan_context(
            pts, n_rings=args.sc_rings, n_sectors=args.sc_sectors,
            max_radius=args.sc_max_radius)
        if (k + 1) % 25 == 0 or k + 1 == n:
            dt = (time.time() - t0) / (k + 1)
            logger.info(f"  {name}: {k + 1}/{n} descriptors "
                        f"({dt:.2f} s/keyframe)")
    os.makedirs(args.output, exist_ok=True)
    np.savez(cache, rap_meanmax=rap_meanmax, rap_mean=rap_mean,
             spinnet_mean=spinnet_mean, sc=scs,
             keyframes=np.array(keyframes))
    logger.info(f"{name}: cached descriptors -> {cache}")
    return {"rap_meanmax": rap_meanmax, "rap_mean": rap_mean,
            "spinnet_mean": spinnet_mean, "sc": scs}


def cosine_dists(desc):
    """(K, D) rows (re-normalized here) -> condensed cosine distances."""
    desc = desc / (np.linalg.norm(desc, axis=1, keepdims=True) + 1e-12)
    n = desc.shape[0]
    sim = desc @ desc.T
    iu = np.triu_indices(n, k=1)
    return 1.0 - sim[iu], iu


def pair_stats(dists, iu, keyframes, gt_pos, sep, true_dist):
    """Separability / AUC / best-F1 for one arm at one GT true threshold."""
    i, j = iu
    kf = np.asarray(keyframes)
    mask = (kf[j] - kf[i]) >= sep
    i, j, d = i[mask], j[mask], dists[mask]
    gd = np.linalg.norm(gt_pos[kf[i]] - gt_pos[kf[j]], axis=1)
    truth = gd < true_dist
    n_true = int(truth.sum())
    if n_true == 0:
        return {"n_true": 0}
    t_d, f_d = d[truth], d[~truth]
    # AUC = P(d_true < d_false), direct pair counting via sorted false dists
    f_sorted = np.sort(f_d)
    n_right = np.searchsorted(f_sorted, t_d, "right")  # false <= t
    n_greater = len(f_d) - n_right                     # false strictly > t
    n_tie = n_right - np.searchsorted(f_sorted, t_d, "left")
    auc = float((n_greater.sum() + 0.5 * n_tie.sum()) / (n_true * len(f_d)))
    # best F1 via sorted arrays + searchsorted (fast on 300k+ pairs)
    t_sorted = np.sort(t_d)
    cand = np.quantile(d, np.linspace(0, 1, 2001))
    tp = np.searchsorted(t_sorted, cand, "right")
    fp = np.searchsorted(f_sorted, cand, "right")
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / n_true
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    k = int(np.argmax(f1))
    best = {"f1": float(f1[k]), "thresh": float(cand[k]),
            "precision": float(prec[k]), "recall": float(rec[k])}
    return {
        "n_true": n_true,
        "max_true_dist": float(t_d.max()),
        "med_true_dist": float(np.median(t_d)),
        "min_false_dist": float(f_d.min()),
        "margin": float(f_d.min() - t_d.max()),
        "auc": float(auc),
        "best_f1": best,
    }


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    t0_all = time.time()
    results = {}

    for name in args.datasets.split(","):
        cfg = DATASETS[name]
        keyframes = json.load(open(os.path.join(REPO_ROOT, cfg["keyframes"])))
        gt = list(real_dataset.load_gt_poses(
            cfg["poses_file"], cfg["calib_file"], start=cfg["start"],
            end=cfg["end"], apply_calib=cfg["apply_calib"]))
        gt_pos = np.array([T[:3, 3] for T in gt])
        logger.info(f"{name}: {len(keyframes)} keyframes, GT {len(gt)} poses")

        t0 = time.time()
        descs = compute_descriptors(name, cfg, keyframes, args)
        t_desc = time.time() - t0

        res = {"n_keyframes": len(keyframes), "t_desc_s": round(t_desc, 1),
               "t_desc_per_kf_s": round(t_desc / len(keyframes), 3),
               "arms": {}}
        for arm in ("rap_meanmax", "rap_mean", "spinnet_mean"):
            d, iu = cosine_dists(descs[arm])
            for td in cfg["true_dists"]:
                st = pair_stats(d, iu, keyframes, gt_pos, args.loop_sep, td)
                res["arms"].setdefault(arm, {})[f"true<{td}m"] = st
                logger.info(
                    f"  {name} {arm} true<{td} m: n={st.get('n_true')} "
                    f"max_true={st.get('max_true_dist', float('nan')):.3f} "
                    f"min_false={st.get('min_false_dist', float('nan')):.3f} "
                    f"margin={st.get('margin', float('nan')):.3f} "
                    f"AUC={st.get('auc', float('nan')):.4f} "
                    f"bestF1={st.get('best_f1', {}).get('f1', float('nan')):.3f}"
                    f" @ {st.get('best_f1', {}).get('thresh', float('nan')):.3f}"
                    f" (P {st.get('best_f1', {}).get('precision', float('nan')):.2f}"
                    f" R {st.get('best_f1', {}).get('recall', float('nan')):.2f})")
        # Scan Context arm (shift-invariant distance, vectorized per query)
        scs = descs["sc"]
        n = len(keyframes)
        pairs = [(a, b) for b in range(n) for a in range(b)]
        d_sc = np.zeros(len(pairs))
        for b in range(1, n):
            d_b, _ = sc_mod.sc_distances_batch(scs[:b], scs[b])
            base = b * (b - 1) // 2
            d_sc[base:base + b] = d_b
        iu = (np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs]))
        for td in cfg["true_dists"]:
            st = pair_stats(d_sc, iu, keyframes, gt_pos, args.loop_sep, td)
            res["arms"].setdefault("sc", {})[f"true<{td}m"] = st
            logger.info(
                f"  {name} sc true<{td} m: n={st.get('n_true')} "
                f"max_true={st.get('max_true_dist', float('nan')):.3f} "
                f"min_false={st.get('min_false_dist', float('nan')):.3f} "
                f"margin={st.get('margin', float('nan')):.3f} "
                f"AUC={st.get('auc', float('nan')):.4f} "
                f"bestF1={st.get('best_f1', {}).get('f1', float('nan')):.3f}")
        results[name] = res

    print("\n===== SUMMARY (separability: max_true / min_false, AUC, best F1) =====")
    for name, res in results.items():
        print(f"\n[{name}] {res['n_keyframes']} keyframes "
              f"({res['t_desc_per_kf_s']} s/kf descriptor cost)")
        for arm, per in res["arms"].items():
            for td_key, st in per.items():
                if st.get("n_true", 0) == 0:
                    print(f"  {arm:12s} {td_key}: no true pairs")
                    continue
                bf = st["best_f1"]
                print(f"  {arm:12s} {td_key}: true<={st['max_true_dist']:.3f} "
                      f"false>={st['min_false_dist']:.3f} "
                      f"margin={st['margin']:+.3f} AUC={st['auc']:.4f} "
                      f"F1={bf['f1']:.3f} (P {bf['precision']:.2f} "
                      f"R {bf['recall']:.2f} @ {bf['thresh']:.3f})")
    results["config"] = {k: v for k, v in vars(args).items()}
    results["total_runtime_s"] = round(time.time() - t0_all, 1)
    out = os.path.join(args.output, "summary.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1, default=str)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
