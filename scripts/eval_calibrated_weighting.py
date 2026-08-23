"""P2 — calibrated generative uncertainty: fit a conditional error model
p(edge error | RAP diagnostics) and validate zero-adjustment transfer.

Data (all saved products, no GPU):
  FIT (KITTI-02 + POSS):
    - kitti02 live: 182 accepted edges (output_ros2/step3_kitti02) +
      multi-generation dispersion from output_multigen/multigen_samples.json
      (4 gens/pair -> disp_t/disp_r/overlap_min; RC-3 products)
    - poss: offline 4 (output_prods/poss) + live 2 (output_ros2/step3) +
      possb 12 (output_possb_smoke/summary.json, analysis-only: no T_rel)
  TRANSFER (zero-adjustment): kitti00_bb 40, kitti05 12, kitti06 offline 12,
    kitti06 live 114.

Model: per-edge noise sigma for PGO as a function of RAP mean_dist (and
optionally overlap / dispersion):
    sigma_t = a_t * mean_dist^b_t   (log-log LSQ on gt trans err, rescaled
                                     to 68.3% in-sample coverage)
    sigma_r = a_r * mean_dist^b_r   (same on gt rot err)
Benchmarks compared on identical edges: fixed (base sigmas 0.20 m / 0.03 rad),
relfloor (current default), adaptive (RC-1 alternative), calibrated (this).

Calibration metrics per scheme: empirical coverage at 1/2/3 sigma (target
68.3/95.4/99.7 under a half-normal-ish reading) and mean NLL.

Phase pgo: re-run every golden with loop_weighting in {relfloor, calibrated}
(same NMS setting as each golden's canonical run) and compare APE.

Run (WSL2 conda rap-inference; phase fit is CPU-numpy, pgo needs gtsam):
  python scripts/eval_calibrated_weighting.py --phase all
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from rap_slam import real_dataset  # noqa: E402
from eval_paper_metrics import (  # noqa: E402
    POSS_SEGMENTS, gt_for_live, gt_from_summary)
from rerun_pgo_saved import load_tum, mat2quat  # noqa: E402

logger = logging.getLogger("eval_calibrated_weighting")

OUT = os.path.join(REPO_ROOT, "output_calibrated")

# PGO base noise (rap_slam/pose_graph.py defaults): rot 0.03 rad, trans 0.20 m
BASE_SIGMA_T = 0.20
BASE_SIGMA_R = 0.03


# ---------------------------------------------------------------------------
# edge dataset assembly
# ---------------------------------------------------------------------------

def _edge_errors(T_est: np.ndarray, T_gt: np.ndarray):
    dG = np.linalg.inv(T_gt) @ T_est
    te = float(np.linalg.norm(dG[:3, 3]))
    re = float(np.rad2deg(np.arccos(np.clip(
        (np.trace(dG[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))))
    return te, re


def _dispersion(gens: List[dict]):
    """Mean pairwise translation/rotation spread across generations (m/deg)."""
    Ts = [np.asarray(g["T_rel"], dtype=float) for g in gens]
    if len(Ts) < 2:
        return None, None
    dt, dr = [], []
    for a in range(len(Ts)):
        for b in range(a + 1, len(Ts)):
            d = Ts[a] @ np.linalg.inv(Ts[b])
            dt.append(float(np.linalg.norm(d[:3, 3])))
            dr.append(float(np.rad2deg(np.arccos(np.clip(
                (np.trace(d[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)))))
    return float(np.mean(dt)), float(np.mean(dr))


def assemble_edges() -> List[dict]:
    edges: List[dict] = []

    def add(ds, i, j, overlap, mean_dist, te, re, gt_dist, T_rel=None,
            extra=None):
        e = {"ds": ds, "i": int(i), "j": int(j), "overlap": float(overlap),
             "mean_dist": float(mean_dist), "gt_te": te, "gt_re": re,
             "gt_dist": gt_dist}
        if T_rel is not None:
            e["T_rel"] = np.asarray(T_rel, dtype=float).tolist()
        if extra:
            e.update(extra)
        edges.append(e)

    def add_from_loops(ds, loops_path, gt):
        for lc in json.load(open(loops_path)):
            T_est = np.asarray(lc["T_rel"], dtype=float)
            T_gt = np.linalg.inv(gt[lc["i"]]) @ gt[lc["j"]]
            te, re = _edge_errors(T_est, T_gt)
            gd = float(np.linalg.norm(
                gt[lc["i"]][:3, 3] - gt[lc["j"]][:3, 3]))
            add(ds, lc["i"], lc["j"], lc["overlap"], lc["mean_dist"],
                te, re, gd, T_rel=T_est)

    # --- FIT: KITTI-02 live (182) ---
    gt02 = gt_for_live("kitti", "02", 4661)
    add_from_loops("kitti02", os.path.join(
        REPO_ROOT, "output_ros2/step3_kitti02/accepted_loops.json"), gt02)
    # POSS offline (4) + live (2)
    poss_off_dir = os.path.join(REPO_ROOT, "output_prods/poss")
    gt_poss_off = gt_from_summary(poss_off_dir)
    add_from_loops("poss_off", os.path.join(poss_off_dir,
                                            "accepted_loops.json"),
                   gt_poss_off)
    gt_poss_live = gt_for_live("poss", "A", 1419)
    add_from_loops("poss_live", os.path.join(
        REPO_ROOT, "output_ros2/step3/accepted_loops.json"), gt_poss_live)
    # POSS-B smoke (12, analysis-only: summary.json stores gt errs, no T_rel)
    sb = json.load(open(os.path.join(REPO_ROOT,
                                     "output_possb_smoke/summary.json")))
    for lc in sb["accepted_loops"]:
        add("possb", lc["i"], lc["j"], lc["overlap"],
            lc.get("mean_dist", np.nan), float(lc["gt_trans_err"]),
            float(lc["gt_rot_err_deg"]), np.nan)

    # --- TRANSFER: 00_bb / 05 / 06 offline + 06 live ---
    for ds, d in (("kitti00", "output_prods/kitti00_bb"),
                  ("kitti05", "output_prods/kitti05"),
                  ("kitti06_off", "output_prods/kitti06")):
        full = os.path.join(REPO_ROOT, d)
        add_from_loops(ds, os.path.join(full, "accepted_loops.json"),
                       gt_from_summary(full))
    gt06 = gt_for_live("kitti", "06", 1101)
    add_from_loops("kitti06_live", os.path.join(
        REPO_ROOT, "output_ros2/step3_kitti06/accepted_loops.json"), gt06)

    # --- multigen dispersion features (RC-3 products), keyed by (ds, i, j)
    mg = json.load(open(os.path.join(
        REPO_ROOT, "output_multigen/multigen_samples.json")))["entries"]
    lut = {}
    for e in mg:
        disp_t, disp_r = _dispersion(e["gens"])
        ovs = [g["overlap"] for g in e["gens"]]
        mds = [g["mean_dist"] for g in e["gens"]]
        lut[(e["ds"], e["i"], e["j"])] = {
            "disp_t": disp_t, "disp_r": disp_r,
            "overlap_min": float(np.min(ovs)),
            "md_gen_mean": float(np.mean(mds)),
        }
    n_hit = 0
    for e in edges:
        key = None
        if e["ds"] == "kitti02":
            key = ("kitti02_true", e["i"], e["j"])
            if key not in lut:
                key = ("kitti02_false", e["i"], e["j"])
        elif e["ds"] == "poss_live":
            key = ("poss_true", e["i"], e["j"])
        if key in lut:
            e.update(lut[key])
            n_hit += 1
    logger.info(f"dispersion features attached to {n_hit}/{len(edges)} edges")
    return edges


def load_poses_auto(path: str) -> List[np.ndarray]:
    """TUM (8 cols, live) or KITTI 3x4 (12 cols, offline) pose file."""
    rows = np.loadtxt(path)
    if rows.ndim == 1:
        rows = rows[None, :]
    if rows.shape[1] == 12:
        poses = []
        for r in rows:
            T = np.eye(4)
            T[:3] = r.reshape(3, 4)
            poses.append(T)
        return poses
    _, poses = load_tum(path)
    return poses


# ---------------------------------------------------------------------------
# sigma schemes
# ---------------------------------------------------------------------------

def sigma_fixed(md, ov):
    return BASE_SIGMA_T, BASE_SIGMA_R


def sigma_relfloor(md, ov, ref=1.0, cap=3.0):
    s = float(np.clip(md / ref, 1.0, cap))
    return BASE_SIGMA_T * s, BASE_SIGMA_R * s


def sigma_adaptive(md, ov, alpha_t=1.0, alpha_r=0.15):
    return (float(np.clip(alpha_t * md, 0.10, 3.0)),
            float(np.clip(alpha_r * md, 0.05, 1.0)))


def sigma_calibrated(md, ov, p):
    st = p["a_t"] * float(md) ** p["b_t"]
    sr = p["a_r"] * float(md) ** p["b_r"]
    return (float(np.clip(st, p["floor_t"], p["cap_t"])),
            float(np.clip(sr, p["floor_r"], p["cap_r"])))


def fit_calibrated(fit_edges: List[dict]) -> dict:
    """log-log LSQ of gt error ~ mean_dist, rescaled to 68.3% coverage."""
    md = np.array([e["mean_dist"] for e in fit_edges])
    te = np.array([e["gt_te"] for e in fit_edges])
    re_ = np.array([e["gt_re"] for e in fit_edges])  # degrees

    def fit_one(err):
        err = np.maximum(err, 1e-3)  # avoid log(0) on near-perfect edges
        x, y = np.log(md), np.log(err)
        b, a = np.polyfit(x, y, 1)
        a = float(np.exp(a))
        # coverage rescale: find c s.t. P(err <= c * a * md^b) = 68.3%
        pred = a * md ** b
        ratio = err / pred
        c = float(np.quantile(ratio, 0.683))
        return a * c, float(b)

    a_t, b_t = fit_one(te)
    a_r, b_r = fit_one(re_)
    # store rotation params in RADIANS (gtsam convention)
    d2r = np.pi / 180.0
    p = {"a_t": a_t, "b_t": b_t, "a_r": a_r * d2r, "b_r": b_r,
         # floors/caps: keep sigmas inside the observed sensible range
         "floor_t": float(max(np.quantile(te, 0.1) * 0.5, 0.05)),
         "cap_t": float(np.quantile(te, 0.99) * 3.0),
         "floor_r": float(max(np.quantile(re_, 0.1) * 0.5 * d2r,
                              np.deg2rad(0.05))),
         "cap_r": float(np.quantile(re_, 0.99) * 3.0 * d2r)}
    logger.info(f"fit on {len(fit_edges)} edges: sigma_t = "
                f"{p['a_t']:.3f} * md^{p['b_t']:.3f}  "
                f"[{p['floor_t']:.3f}, {p['cap_t']:.3f}] m; sigma_r = "
                f"{p['a_r'] / d2r:.3f} * md^{p['b_r']:.3f} deg "
                f"[{p['floor_r'] / d2r:.3f}, {p['cap_r'] / d2r:.3f}]")
    return p


def coverage_report(edges: List[dict], sigma_fn, p: Optional[dict] = None):
    """Empirical coverage at 1/2/3 sigma + Gaussian NLL (per dim pair)."""
    zt, zr = [], []
    for e in edges:
        st, sr = sigma_fn(e["mean_dist"], e["overlap"], p) if p else \
            sigma_fn(e["mean_dist"], e["overlap"])
        zt.append(e["gt_te"] / st)
        zr.append(e["gt_re"] / np.rad2deg(sr))
    zt, zr = np.array(zt), np.array(zr)
    # half-normal thresholds: |z| < 1.347 (68.3%), 2.448 (95.4%), 3.478 (99.7%)
    def cov(z, q):
        return float(np.mean(z < q))
    out = {}
    for name, z in (("t", zt), ("r", zr)):
        out[f"cov1_{name}"] = cov(z, 1.347)
        out[f"cov2_{name}"] = cov(z, 2.448)
        out[f"cov3_{name}"] = cov(z, 3.478)
    out["z_t_median"] = float(np.median(zt))
    out["z_r_median"] = float(np.median(zr))
    return out


# ---------------------------------------------------------------------------
# phase pgo: golden regression under a chosen weighting
# ---------------------------------------------------------------------------

GOLDENS = {
    # name: (odom TUM, keyframes.json, accepted_loops.json, source, seq, nms)
    "poss_off": ("output_prods/poss/odom_poses.txt",
                 "output_prods/poss/keyframes.json",
                 "output_prods/poss/accepted_loops.json", None, None, 0.0),
    "kitti06_off": ("output_prods/kitti06/odom_poses.txt",
                    "output_prods/kitti06/keyframes.json",
                    "output_prods/kitti06/accepted_loops.json",
                    None, None, 0.0),
    "kitti05": ("output_prods/kitti05/odom_poses.txt",
                "output_prods/kitti05/keyframes.json",
                "output_prods/kitti05/accepted_loops.json", None, None, 0.0),
    "kitti00_bb": ("output_prods/kitti00_bb/odom_poses.txt",
                   "output_prods/kitti00_bb/keyframes.json",
                   "output_prods/kitti00_bb/accepted_loops.json",
                   None, None, 0.0),
    "kitti02_live": ("output_ros2/live_poses_kitti02.txt",
                     "output_ros2/step3_kitti02/keyframes.json",
                     "output_ros2/step3_kitti02/accepted_loops.json",
                     "kitti", "02", 10.0),
    "kitti06_live": ("output_ros2/live_poses_kitti06.txt",
                     "output_ros2/step3_kitti06/keyframes.json",
                     "output_ros2/step3_kitti06/accepted_loops.json",
                     "kitti", "06", 10.0),
    "poss_live": ("output_ros2/live_poses_step3.txt",
                  "output_ros2/step3/keyframes.json",
                  "output_ros2/step3/accepted_loops.json", "poss", "A", 10.0),
}


def run_golden(name: str, weighting: str, calib: Optional[dict]) -> dict:
    from rap_slam import pose_graph
    odom_f, kf_f, loops_f, source, seq, nms = GOLDENS[name]
    odom = load_poses_auto(os.path.join(REPO_ROOT, odom_f))
    keyframes = json.load(open(os.path.join(REPO_ROOT, kf_f)))
    loops = json.load(open(os.path.join(REPO_ROOT, loops_f)))
    for lc in loops:
        lc["T_rel"] = np.asarray(lc["T_rel"], dtype=float)
    kwargs = dict(loop_weighting=weighting, robust_kernel="huber",
                  loop_nms_width=nms)
    if weighting == "calibrated" and calib:
        kwargs["calib_params"] = calib
    optimized = pose_graph.optimize_pose_graph(odom, keyframes, loops,
                                               **kwargs)
    full = pose_graph.propagate_to_full_trajectory(odom, keyframes, optimized)
    if source:  # live
        gt = gt_for_live(source, seq, len(odom))
    else:       # offline
        gt = gt_from_summary(os.path.dirname(os.path.join(REPO_ROOT,
                                                          odom_f)))
    from eval_paper_metrics import evaluate_pair
    return evaluate_pair(gt, full)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def fmt_cov(c):
    return (f"1σ {100*c['cov1_t']:.0f}/{100*c['cov1_r']:.0f} "
            f"2σ {100*c['cov2_t']:.0f}/{100*c['cov2_r']:.0f} "
            f"3σ {100*c['cov3_t']:.0f}/{100*c['cov3_r']:.0f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", default="all", choices=["fit", "pgo", "all"])
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s: "
                               "%(message)s", datefmt="%H:%M:%S")
    os.makedirs(args.out, exist_ok=True)

    edges = assemble_edges()
    with open(os.path.join(args.out, "edge_dataset.json"), "w") as f:
        json.dump(edges, f)
    by_ds = {}
    for e in edges:
        by_ds.setdefault(e["ds"], []).append(e)
    logger.info("edge dataset: " + ", ".join(
        f"{k}={len(v)}" for k, v in sorted(by_ds.items())))

    fit_edges = [e for e in edges
                 if e["ds"] in ("kitti02", "poss_off", "poss_live", "possb")
                 and np.isfinite(e["mean_dist"])]
    transfer = {k: v for k, v in by_ds.items()
                if k in ("kitti00", "kitti05", "kitti06_off",
                         "kitti06_live")}

    calib = fit_calibrated(fit_edges)

    # Dispersion-augmented model (in-sample only): sigma_t = a * disp_t^b.
    # Dispersion requires extra RAP sampling passes (RC-3), so it cannot
    # transfer to the non-multigen validation sets — we quantify its
    # in-sample calibration gain over the mean_dist model here.
    disp_report = None
    mg_mask = np.array([np.isfinite(e.get("disp_t", np.nan))
                        and e["disp_t"] > 0 for e in fit_edges])
    if mg_mask.sum() > 30:
        sub = [e for e, m in zip(fit_edges, mg_mask) if m]
        dx = np.log(np.array([e["disp_t"] for e in sub]))
        dy = np.log(np.maximum(np.array([e["gt_te"] for e in sub]), 1e-3))
        b_d, a_d = np.polyfit(dx, dy, 1)
        pred = np.exp(a_d) * np.exp(dx) ** b_d
        c_d = float(np.quantile(dy - np.log(pred), 0.683))
        a_d = float(np.exp(a_d + c_d))
        z = np.exp(dy) / (a_d * np.exp(dx) ** b_d)
        disp_report = {
            "a_t": a_d, "b_t": float(b_d), "n": int(mg_mask.sum()),
            "cov1": float(np.mean(z < 1.347)),
            "cov2": float(np.mean(z < 2.448)),
            "cov3": float(np.mean(z < 3.478)),
        }
        logger.info(f"dispersion model: sigma_t = {a_d:.3f} * disp_t^"
                    f"{b_d:.3f}, cov 1/2/3σ = {disp_report['cov1']:.2f}/"
                    f"{disp_report['cov2']:.2f}/{disp_report['cov3']:.2f} "
                    f"(n={mg_mask.sum()})")
        # same-subset coverage of the mean_dist model for comparison
        zs = np.array([e["gt_te"] for e in sub]) / np.array(
            [sigma_calibrated(e["mean_dist"], e["overlap"], calib)[0]
             for e in sub])
        disp_report["md_model_cov1_same_subset"] = float(
            np.mean(zs < 1.347))

    # feature-value check: log-log LSQ R^2 for candidate predictors
    md = np.array([e["mean_dist"] for e in fit_edges])
    ov = np.array([e["overlap"] for e in fit_edges])
    te = np.log(np.maximum(np.array([e["gt_te"] for e in fit_edges]), 1e-3))
    feats = {"ln_md": np.log(md), "overlap": ov}
    disp = np.array([e.get("disp_t", np.nan) for e in fit_edges])
    if np.isfinite(disp).sum() > 30:
        feats["ln_disp_t"] = np.log(disp[np.isfinite(disp)])
    feat_report = {}
    for fn, x in feats.items():
        if fn == "ln_disp_t":
            m = np.isfinite(disp)
            x = np.log(np.maximum(disp[m], 1e-4))
            y = te[m]
        else:
            m = np.isfinite(x) & np.isfinite(te)
            x, y = x[m], te[m]
        r = np.corrcoef(x, y)[0, 1]
        feat_report[fn] = {"pearson_r_vs_ln_te": float(r), "n": int(m.sum())}
    logger.info(f"feature value vs ln(gt_te): {feat_report}")

    schemes = {"fixed": (sigma_fixed, None),
               "relfloor": (sigma_relfloor, None),
               "adaptive": (sigma_adaptive, None),
               "calibrated": (sigma_calibrated, calib)}
    report = {"fit_set": {}, "transfer": {}, "features": feat_report,
              "calib_params": calib, "dispersion_model": disp_report}
    for name, (fn, p) in schemes.items():
        report["fit_set"][name] = coverage_report(fit_edges, fn, p)
        for tds, tedges in transfer.items():
            report["transfer"].setdefault(tds, {})[name] = \
                coverage_report(tedges, fn, p)

    lines = ["# P2 — calibrated generative uncertainty: fit + transfer", ""]
    d2r = np.pi / 180.0
    lines.append(f"Fit set: {len(fit_edges)} edges (KITTI-02 182 + POSS 18). "
                 f"sigma_t = {calib['a_t']:.3f} · mean_dist^{calib['b_t']:.3f} "
                 f"m (clip [{calib['floor_t']:.3f}, {calib['cap_t']:.2f}]); "
                 f"sigma_r = {calib['a_r'] / d2r:.3f} · "
                 f"mean_dist^{calib['b_r']:.3f} deg "
                 f"(clip [{calib['floor_r'] / d2r:.3f}, "
                 f"{calib['cap_r'] / d2r:.2f}]). "
                 "Coverage target: 68/95/99.7 % at 1/2/3 σ.")
    lines.append("")
    lines.append("| set | scheme | cov t 1/2/3σ [%] | cov r 1/2/3σ [%] |")
    lines.append("|---|---|---|---|")
    for name in schemes:
        c = report["fit_set"][name]
        lines.append(f"| fit (02+POSS) | {name} | "
                     f"{100*c['cov1_t']:.0f}/{100*c['cov2_t']:.0f}/"
                     f"{100*c['cov3_t']:.0f} | "
                     f"{100*c['cov1_r']:.0f}/{100*c['cov2_r']:.0f}/"
                     f"{100*c['cov3_r']:.0f} |")
    for tds in transfer:
        for name in schemes:
            c = report["transfer"][tds][name]
            lines.append(f"| {tds} | {name} | "
                         f"{100*c['cov1_t']:.0f}/{100*c['cov2_t']:.0f}/"
                         f"{100*c['cov3_t']:.0f} | "
                         f"{100*c['cov1_r']:.0f}/{100*c['cov2_r']:.0f}/"
                         f"{100*c['cov3_r']:.0f} |")
    lines.append("")
    if disp_report:
        lines.append(
            f"Dispersion model (in-sample, n={disp_report['n']}): "
            f"sigma_t = {disp_report['a_t']:.3f} · disp_t^"
            f"{disp_report['b_t']:.3f}, 1σ coverage "
            f"{100 * disp_report['cov1']:.0f} % vs mean_dist-model "
            f"{100 * disp_report['md_model_cov1_same_subset']:.0f} % on the "
            "same subset. Dispersion is the only diagnostic with per-edge "
            "signal but requires extra RAP sampling passes (RC-3) and "
            "showed no PGO gain there — so the transferable calibrated "
            "model stays mean_dist-based (near-constant).")
    lines.append("")

    if args.phase in ("pgo", "all"):
        lines.append("## Golden PGO regression (APE unaligned / aligned [m])")
        lines.append("")
        lines.append("| golden | nms | relfloor (current) | calibrated |")
        lines.append("|---|---|---|---|")
        report["pgo"] = {}
        for name, g in GOLDENS.items():
            row = {}
            for w in ("relfloor", "calibrated"):
                m = run_golden(name, w, calib)
                row[w] = m
                logger.info(f"[{name}][{w}] APE {m['ape_unaligned_rmse']:.3f}"
                            f" / {m['ape_aligned_rmse']:.3f} m")
            report["pgo"][name] = row
            lines.append(f"| {name} | {g[5]:.0f} | "
                         f"{row['relfloor']['ape_unaligned_rmse']:.3f} / "
                         f"{row['relfloor']['ape_aligned_rmse']:.3f} | "
                         f"{row['calibrated']['ape_unaligned_rmse']:.3f} / "
                         f"{row['calibrated']['ape_aligned_rmse']:.3f} |")

    with open(os.path.join(args.out, "fit_report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(args.out, "fit_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Wrote {args.out}/fit_report.{{md,json}}")


if __name__ == "__main__":
    main()
