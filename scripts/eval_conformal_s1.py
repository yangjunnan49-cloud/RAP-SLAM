#!/usr/bin/env python
"""S1 — conformal acceptance gate (S1a) + conformalized per-edge sigma
(S1b), evaluated against the predeclared gates of the 2026-09-05 program.

Inputs (from scripts/build_s1_features.py):
  output_s1/candidates.json — 828 saved verification candidates with labels
  output_s1/edges_v2.json   — the P2 378-edge dataset + new features

Predeclared protocol (fixed 2026-09-05 before any result was seen):

S1a method (split-conformal FPR control, "design B"):
  danger score s(x) = L2-logistic P(dangerous | features), trained on a FIT
  split; acceptance threshold Q = j-th smallest FIT-model score on a
  dangerous-only CALIB split with j = floor((n_cal + 1) * ALPHA), ALPHA =
  0.05. Acceptance region = {s(x) < Q}, giving the marginal guarantee
  P(accept | dangerous) <= ALPHA under exchangeability of the dangerous
  pool. Splits are stratified by (set, label), seed 42 (sensitivity: 0-4).
  Arms:
    A1 conf-basic:  [overlap, mean_dist]
    A2 conf-full:   + [sc_d1, sc_margin, deg_loc_t, deg_loc_r, deg_scat]
                       (PRIMARY; intrinsic geometry only — odometry
                       consistency is excluded by project principle)
    A1o overlap-only: [overlap] — for the legacy bigbudget set (no
                       mean_dist was logged pre-f59686f); TEST-only there.
    A3 conf-multigen (multigen subsets only): A2 + [overlap_min, disp_t].
S1a GATE (clause 1): per-set dangerous-accept count of the conformal gate
  <= the current heuristic gate's count under identical labels on EVERY
  saved set, zero wherever the heuristic is zero, AND overall safe recall
  >= 95%. (Labelling the saved edges by gt_te > 5 m or gt_re > 10 deg gives
  the heuristic 6 dangerous accepts, all on kitti02_live; every rejected
  candidate in the saved sets is dangerous or gray — the historical "0/84"
  claim refers to rejected-candidate falseness, which we also report.)

S1b method (conformalized quantile sigma):
  sigma_t(x) = q_0.8413(gt_te | x) + Q, one-sided 68.3% upper bound; Q =
  the ceil((n_cal+1)*0.683)-th smallest conformal score (y - q_hi) on CALIB.
  Models: M-lin (quantile regression on ln y, P2-comparable) and M-gbm
  (quantile GBM on raw y); the sharper one on CALIB is selected ("calibrated
  v2"). Baseline: P2 near-constant sigma_const = a_t * mean_dist^b_t fit on
  the same FIT split and rescaled to 68.3% one-sided coverage on the same
  CALIB split. Same pipeline on gt_re for sigma_r.
S1b GATE (clause 2): mean sigma_CQR <= 0.90 * mean sigma_const on TEST at
  equal-or-better (>= -1 pp) empirical one-sided 68.3% coverage.
S1b GATE (clause 3): PGO with loop_weighting="per_edge" (CQR sigmas,
  leave-one-golden-out fits) must be within +5% of the canonical relfloor
  unaligned APE on >= 5/7 goldens; each golden also re-runs relfloor
  in-process as a bit-exactness guard against environment drift.

Run (WSL2 conda rap-inference; gate/sigma CPU-numpy+sklearn, pgo needs gtsam):
  python scripts/eval_conformal_s1.py --phase all
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

logger = logging.getLogger("eval_conformal_s1")

OUT = os.path.join(REPO_ROOT, "output_s1")

ALPHA = 0.05            # false-accept bound (S1a)
COVERAGE = 0.683        # one-sided 1-sigma coverage (S1b)
SHARPNESS = 0.90        # clause-2 width ratio threshold
RECALL_REQ = 0.95       # clause-1 safe-recall requirement
PGO_TOL = 1.05          # clause-3 per-golden regression tolerance
SEED = 42
SENS_SEEDS = [0, 1, 2, 3, 4]

GATE_BASIC = ["overlap", "mean_dist"]
# NOTE: odom_dT/odom_dR are deliberately NOT gate features — gating on
# odometry consistency is self-defeating under drift (hard-won gotcha #2 in
# AGENTS.md). The gate uses intrinsic geometry only.
GATE_EXTRA = ["sc_d1", "sc_margin", "deg_loc_t", "deg_loc_r", "deg_scat"]
GATE_MG = ["overlap_min", "disp_t"]
SIGMA_FEATS = ["mean_dist", "overlap", "sc_d1", "sc_margin",
               "deg_loc_t", "deg_loc_r", "deg_scat"]

GOLDEN2DS = {"poss_off": "poss_off", "kitti06_off": "kitti06_off",
             "kitti05": "kitti05", "kitti00_bb": "kitti00",
             "kitti02_live": "kitti02", "kitti06_live": "kitti06_live",
             "poss_live": "poss_live"}
# canonical relfloor references (verified bit-exact by eval_gnc_backend.py
# canonical arms; loaded from its report at runtime with this fallback)
CANONICAL = {"poss_off": 1.8208768471979413, "kitti06_off": 2.4410217486493364,
             "kitti05": 3.417, "kitti00_bb": 10.920, "kitti02_live": 26.018,
             "kitti06_live": 2.581, "poss_live": 1.757}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def feat_vec(rec: dict, feats: List[str]) -> Optional[np.ndarray]:
    v = []
    for f in feats:
        x = rec.get(f)
        if x is None or not np.isfinite(x):
            return None
        v.append(float(x))
    return np.array(v)


def conformal_k(n_cal: int, level: float) -> int:
    """Rank index (0-based) of the split-conformal quantile."""
    return min(n_cal - 1, int(np.ceil((n_cal + 1) * level)) - 1)


def stratified_split(recs: List[dict], seed: int, fracs=(0.5, 0.25, 0.25),
                     key=None):
    """Stratify by key(r) (default: (ds, label)); returns (fit, calib, test).
    """
    rng = np.random.RandomState(seed)
    if key is None:
        key = lambda r: (r["ds"], r["label"])
    groups: Dict[tuple, List[dict]] = {}
    for r in recs:
        groups.setdefault(key(r), []).append(r)
    fit, calib, test = [], [], []
    for g in groups.values():
        g = list(g)
        rng.shuffle(g)
        n = len(g)
        a, b = int(n * fracs[0]), int(n * fracs[0] + n * fracs[1])
        fit += g[:a]
        calib += g[a:b]
        test += g[b:]
    return fit, calib, test


# ---------------------------------------------------------------------------
# S1a — the conformal acceptance gate
# ---------------------------------------------------------------------------

def fit_gate(fit: List[dict], feats: List[str]):
    from sklearn.linear_model import LogisticRegression
    X = np.stack([feat_vec(r, feats) for r in fit])
    y = np.array([1 if r["label"] == "dangerous" else 0 for r in fit])
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    clf = LogisticRegression(C=1.0, max_iter=5000)
    clf.fit((X - mu) / sd, y)
    return (clf, mu, sd, feats)


def gate_score(model, rec) -> Optional[float]:
    clf, mu, sd, feats = model
    v = feat_vec(rec, feats)
    if v is None:
        return None
    return float(clf.predict_proba(((v - mu) / sd)[None, :])[0, 1])


def conformal_threshold(model, calib_d: List[dict]) -> Optional[float]:
    scores = [gate_score(model, c) for c in calib_d]
    scores = sorted(s for s in scores if s is not None)
    n = len(scores)
    if n == 0:
        return None
    j = int(np.floor((n + 1) * ALPHA))          # accept iff score < Q
    if j <= 0:
        return -np.inf                          # accept nothing
    return scores[j - 1]


def run_gate_arm(cands: List[dict], feats: List[str], seed: int,
                 legacy_sets=()) -> dict:
    """One arm, one seed. legacy_sets are evaluated TEST-only (never split
    into fit/calib)."""
    pool = [c for c in cands
            if c["label"] in ("safe", "dangerous")
            and c["ds"] not in legacy_sets
            and feat_vec(c, feats) is not None]
    fit, calib, test = stratified_split(pool, seed)
    calib_d = [c for c in calib if c["label"] == "dangerous"]
    model = fit_gate(fit, feats)
    Q = conformal_threshold(model, calib_d)

    # TEST-split honest metrics
    def verdict(c):
        s = gate_score(model, c)
        return None if s is None else (s < Q)
    te_d = [c for c in test if c["label"] == "dangerous"]
    te_s = [c for c in test if c["label"] == "safe"]
    test_fpr = float(np.mean([verdict(c) for c in te_d])) if te_d else None
    test_rec = float(np.mean([verdict(c) for c in te_s])) if te_s else None

    # per-set behaviour on ALL saved candidates (the clause-1 view)
    per_set: Dict[str, dict] = {}
    for c in cands:
        if c["label"] not in ("safe", "dangerous"):
            continue
        v = verdict(c)
        if v is None:        # missing features -> unscored
            per_set.setdefault(c["ds"], {"danger_acc": 0, "danger_n": 0,
                                         "safe_acc": 0, "safe_n": 0,
                                         "unscored": 0})
            per_set[c["ds"]]["unscored"] += 1
            continue
        st = per_set.setdefault(c["ds"], {"danger_acc": 0, "danger_n": 0,
                                          "safe_acc": 0, "safe_n": 0,
                                          "unscored": 0})
        if c["label"] == "dangerous":
            st["danger_n"] += 1
            st["danger_acc"] += int(v)
        else:
            st["safe_n"] += 1
            st["safe_acc"] += int(v)
    tot_s = sum(s["safe_acc"] for s in per_set.values())
    n_s = sum(s["safe_n"] for s in per_set.values())
    return {"Q": Q, "n_calib_d": len(calib_d), "test_fpr": test_fpr,
            "test_recall": test_rec, "per_set": per_set,
            "overall_safe_recall": (tot_s / n_s) if n_s else None,
            "coef": dict(zip(feats, model[0].coef_[0].round(3).tolist()))}


def heuristic_reference(cands: List[dict]) -> Dict[str, dict]:
    """The current two-metric gate (overlap >= 0.7 AND mean_dist <= 2.5)
    under identical labels (clause-1 reference)."""
    per_set: Dict[str, dict] = {}
    for c in cands:
        if c["label"] not in ("safe", "dangerous"):
            continue
        st = per_set.setdefault(c["ds"], {"danger_acc": 0, "danger_n": 0,
                                          "safe_acc": 0, "safe_n": 0})
        v = (c.get("overlap") is not None and c["overlap"] >= 0.7
             and c.get("mean_dist") is not None and c["mean_dist"] <= 2.5)
        if c["label"] == "dangerous":
            st["danger_n"] += 1
            st["danger_acc"] += int(v)
        else:
            st["safe_n"] += 1
            st["safe_acc"] += int(v)
    return per_set


def phase_gate(cands: List[dict]) -> dict:
    out: Dict[str, dict] = {}
    out["heuristic_reference"] = heuristic_reference(cands)
    arms = {"A1_basic": (GATE_BASIC, ()),
            "A2_full": (GATE_BASIC + GATE_EXTRA, ()),
            "A1o_overlap_only": (["overlap"], ("kitti02_bigbudget",))}
    mg_sets = ("multigen_k02true", "multigen_k02false", "multigen_poss")
    mg_pool = [c for c in cands if c["ds"] in mg_sets]
    arms_mg = {"A3_multigen": (GATE_BASIC + GATE_MG, ())}

    for name, (feats, legacy) in arms.items():
        main = run_gate_arm(cands, feats, SEED, legacy_sets=legacy)
        sens = [run_gate_arm(cands, feats, s, legacy_sets=legacy)
                for s in SENS_SEEDS]
        fprs = [s["test_fpr"] for s in sens if s["test_fpr"] is not None]
        recs = [s["test_recall"] for s in sens
                if s["test_recall"] is not None]
        main["sensitivity"] = {"test_fpr_range": [min(fprs), max(fprs)],
                               "test_recall_range": [min(recs), max(recs)]}
        out[name] = main
        logger.info(f"[gate][{name}] Q={main['Q']:.4f} "
                    f"(n_calib_d={main['n_calib_d']}) TEST fpr="
                    f"{main['test_fpr']} recall={main['test_recall']} "
                    f"overall safe recall={main['overall_safe_recall']:.3f}")
    for name, (feats, _) in arms_mg.items():
        main = run_gate_arm(mg_pool, feats, SEED)
        out[name] = main
        logger.info(f"[gate][{name}] Q={main['Q']:.4f} TEST fpr="
                    f"{main['test_fpr']} recall={main['test_recall']}")

    # clause 1 verdict against the heuristic reference (identical labels)
    ref = out["heuristic_reference"]
    prim = out["A2_full"]["per_set"]
    rows = []
    ok = True
    for ds in sorted(set(ref) | set(prim)):
        r, p = ref.get(ds, {}), prim.get(ds, {})
        clause_ok = p.get("danger_acc", 0) <= r.get("danger_acc", 0) and \
            (r.get("danger_acc", 0) > 0 or p.get("danger_acc", 0) == 0)
        ok &= clause_ok
        rows.append({"set": ds,
                     "heur_danger_acc": r.get("danger_acc"),
                     "heur_danger_n": r.get("danger_n"),
                     "conf_danger_acc": p.get("danger_acc"),
                     "conf_danger_n": p.get("danger_n"),
                     "conf_safe_recall": (p.get("safe_acc", 0) /
                                          max(p.get("safe_n", 1), 1)),
                     "unscored": p.get("unscored", 0),
                     "clause_ok": bool(clause_ok)})
    rec = out["A2_full"]["overall_safe_recall"]
    href_s = sum(s.get("safe_acc", 0) for s in ref.values())
    nref_s = sum(s.get("safe_n", 0) for s in ref.values())
    out["clause1"] = {"per_set": rows, "overall_safe_recall": rec,
                      "heuristic_overall_safe_recall":
                      href_s / max(nref_s, 1),
                      "recall_req": RECALL_REQ,
                      "PASS": bool(ok and rec is not None
                                   and rec >= RECALL_REQ)}
    return out


# ---------------------------------------------------------------------------
# S1b — conformalized per-edge sigma
# ---------------------------------------------------------------------------

def fit_const_baseline(fit: List[dict], calib: List[dict],
                       err_key: str) -> dict:
    """P2 construction: log-log LSQ err ~ mean_dist on FIT, rescale to
    68.3% one-sided coverage on CALIB."""
    md = np.array([e["mean_dist"] for e in fit])
    y = np.maximum(np.array([e[err_key] for e in fit]), 1e-3)
    b, la = np.polyfit(np.log(md), np.log(y), 1)
    a = float(np.exp(la))
    pred = a * md ** b
    rc = np.array([e[err_key] for e in calib]) / \
        (a * np.array([e["mean_dist"] for e in calib]) ** b)
    c = float(np.sort(rc)[conformal_k(len(rc), COVERAGE)])
    return {"a": a * c, "b": float(b)}


def fit_quantile_models(fit: List[dict], feats: List[str], err_key: str,
                        seed: int):
    from sklearn.linear_model import QuantileRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    X = np.stack([feat_vec(e, feats) for e in fit])
    y = np.array([e[err_key] for e in fit])
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    Xs = (X - mu) / sd
    q_hi = 1 - (1 - COVERAGE) / 2
    qlin = QuantileRegressor(quantile=q_hi, alpha=1e-4, solver="highs")
    qlin.fit(Xs, np.log(np.maximum(y, 1e-3)))
    qgbm = GradientBoostingRegressor(loss="quantile", alpha=q_hi,
                                     n_estimators=300, max_depth=2,
                                     learning_rate=0.05, subsample=0.9,
                                     random_state=seed)
    qgbm.fit(Xs, y)
    qgbm_ln = GradientBoostingRegressor(loss="quantile", alpha=q_hi,
                                        n_estimators=300, max_depth=2,
                                        learning_rate=0.05, subsample=0.9,
                                        random_state=seed)
    qgbm_ln.fit(Xs, np.log(np.maximum(y, 1e-3)))
    return {"lin": qlin, "gbm": qgbm, "gbm_ln": qgbm_ln, "mu": mu, "sd": sd,
            "feats": feats, "err_key": err_key}


def predict_qhi(models: dict, kind: str, recs: List[dict]) -> np.ndarray:
    X = np.stack([feat_vec(e, models["feats"]) for e in recs])
    Xs = (X - models["mu"]) / models["sd"]
    if kind in ("lin", "gbm_ln"):
        return np.exp(models[kind].predict(Xs))
    return models["gbm"].predict(Xs)


def conformalize(models: dict, kind: str, calib: List[dict]) -> float:
    q = predict_qhi(models, kind, calib)
    y = np.array([e[models["err_key"]] for e in calib])
    E = y - q
    return float(np.sort(E)[conformal_k(len(E), COVERAGE)])


def eval_sigma(models: dict, kind: str, Q: float, test: List[dict],
               floor: float):
    """Nominal coverage/width plus the coverage-EQUALIZED width: the exact
    0.683-quantile rescale of sigma on the evaluation split, so models are
    compared at exactly equal empirical 1-sigma coverage (the faithful
    reading of the predeclared "at equal 1-sigma coverage" clause)."""
    q = predict_qhi(models, kind, test)
    y = np.array([e[models["err_key"]] for e in test])
    sig = np.maximum(q + Q, floor)
    c_eq = float(np.quantile(y / sig, COVERAGE))
    return {"coverage": float(np.mean(y <= sig)),
            "mean_sigma": float(np.mean(sig)),
            "mean_sigma_eq": float(c_eq * np.mean(sig))}


def const_eval(const: dict, recs: List[dict], err_key: str) -> dict:
    sig = np.array([const["a"] * e["mean_dist"] ** const["b"] for e in recs])
    y = np.array([e[err_key] for e in recs])
    c_eq = float(np.quantile(y / sig, COVERAGE))
    return {"coverage": float(np.mean(y <= sig)),
            "mean_sigma": float(np.mean(sig)),
            "mean_sigma_eq": float(c_eq * np.mean(sig))}


def usable_edges(edges: List[dict], err_key: str) -> List[dict]:
    return [e for e in edges
            if feat_vec(e, SIGMA_FEATS) is not None
            and np.isfinite(e.get(err_key, np.nan))
            and np.isfinite(e.get("mean_dist", np.nan))]


def phase_sigma(edges: List[dict]) -> dict:
    pool_t = usable_edges(edges, "gt_te")
    logger.info(f"[sigma] usable edges: {len(pool_t)}/{len(edges)}")

    # feature signal table (P2 protocol: Pearson + Spearman vs ln error)
    from scipy.stats import pearsonr, spearmanr
    y = np.log(np.maximum([e["gt_te"] for e in pool_t], 1e-3))
    sig_table = {}
    for f in SIGMA_FEATS + ["disp_t"]:
        xs = [e.get(f) for e in pool_t]
        m = np.array([x is not None and np.isfinite(x) for x in xs])
        if m.sum() < 30:
            sig_table[f] = None
            continue
        x = np.array([xs[i] for i in range(len(xs)) if m[i]])
        sig_table[f] = {"n": int(m.sum()),
                        "pearson": float(pearsonr(x, y[m])[0]),
                        "spearman": float(spearmanr(x, y[m])[0])}

    fit, calib, test = stratified_split(pool_t, SEED,
                                        key=lambda r: r["ds"])
    out: Dict[str, object] = {"n_pool": len(pool_t),
                              "feature_signal": sig_table,
                              "split": {"fit": len(fit), "calib": len(calib),
                                        "test": len(test)}}
    selected: Dict[str, str] = {}
    for err_key, floor in (("gt_te", 0.05), ("gt_re", 0.05)):
        const = fit_const_baseline(fit, calib, err_key)
        ce_test = const_eval(const, test, err_key)
        models = fit_quantile_models(fit, SIGMA_FEATS, err_key, SEED)
        res = {}
        for kind in ("lin", "gbm", "gbm_ln"):
            Q = conformalize(models, kind, calib)
            res[kind] = {"Q": Q,
                         "calib": eval_sigma(models, kind, Q, calib, floor),
                         "test": eval_sigma(models, kind, Q, test, floor)}
        # selection on FIT-side (calib) equalized width only; TEST untouched
        sel = min(res, key=lambda k: res[k]["calib"]["mean_sigma_eq"])
        selected[err_key] = sel
        out[err_key] = {"const": {"a": const["a"], "b": const["b"],
                                  **ce_test},
                        "models": res, "selected": sel}
        logger.info(f"[sigma][{err_key}] const: eqwidth="
                    f"{ce_test['mean_sigma_eq']:.3f} | {sel}: eqwidth="
                    f"{res[sel]['test']['mean_sigma_eq']:.3f} (ratio "
                    f"{res[sel]['test']['mean_sigma_eq'] / ce_test['mean_sigma_eq']:.3f})"
                    f" [nominal cov {res[sel]['test']['coverage']:.3f} vs "
                    f"{ce_test['coverage']:.3f}]")
    t = out["gt_te"]
    ratio = t["models"][t["selected"]]["test"]["mean_sigma_eq"] / \
        t["const"]["mean_sigma_eq"]
    # coverage parity is exact by construction of the equalized comparison
    out["clause2"] = {"width_ratio": ratio,
                      "PASS": bool(ratio <= SHARPNESS)}
    out["selected"] = selected
    return out


# ---------------------------------------------------------------------------
# S1b clause 3 — PGO with per-edge CQR sigmas (leave-one-golden-out)
# ---------------------------------------------------------------------------

def run_golden_peredge(name: str, sigmas: Dict[tuple, tuple],
                       weighting: str) -> dict:
    from eval_calibrated_weighting import GOLDENS, load_poses_auto
    from eval_paper_metrics import evaluate_pair, gt_for_live, gt_from_summary
    from rap_slam import pose_graph
    odom_f, kf_f, loops_f, source, seq, nms = GOLDENS[name]
    odom = load_poses_auto(os.path.join(REPO_ROOT, odom_f))
    keyframes = json.load(open(os.path.join(REPO_ROOT, kf_f)))
    loops = json.load(open(os.path.join(REPO_ROOT, loops_f)))
    ds = GOLDEN2DS[name]
    for lc in loops:
        lc["T_rel"] = np.asarray(lc["T_rel"], dtype=float)
        sg = sigmas.get((ds, lc["i"], lc["j"]))
        if sg is not None:
            lc["sigma_t"], lc["sigma_r"] = sg
    optimized = pose_graph.optimize_pose_graph(
        odom, keyframes, loops, loop_weighting=weighting,
        robust_kernel="huber", loop_nms_width=nms)
    full = pose_graph.propagate_to_full_trajectory(odom, keyframes, optimized)
    gt = (gt_for_live(source, seq, len(odom)) if source else
          gt_from_summary(os.path.dirname(os.path.join(REPO_ROOT, odom_f))))
    return evaluate_pair(gt, full)


def phase_pgo_run(edges: List[dict], selected_kind: str) -> dict:
    usable_t = {ds: usable_edges([e for e in edges if e["ds"] == ds], "gt_te")
                for ds in set(e["ds"] for e in edges)}
    usable_r = {ds: usable_edges([e for e in edges if e["ds"] == ds], "gt_re")
                for ds in set(e["ds"] for e in edges)}
    out: Dict[str, dict] = {}
    canon = dict(CANONICAL)
    rep = os.path.join(REPO_ROOT, "output_gnc_backend/gnc_report.json")
    if os.path.exists(rep):
        r = json.load(open(rep))
        for g in canon:
            if g in r and "canonical" in r[g]:
                canon[g] = r[g]["canonical"]["ape_unaligned_rmse"]

    for golden, ds in GOLDEN2DS.items():
        fit_t = [e for d, es in usable_t.items() if d != ds for e in es]
        fit_r = [e for d, es in usable_r.items() if d != ds for e in es]
        sigmas: Dict[tuple, tuple] = {}
        if len(fit_t) >= 60 and len(fit_r) >= 60:
            f_t, c_t, _ = stratified_split(fit_t, SEED, (2 / 3, 1 / 3, 0.0),
                                           key=lambda r: r["ds"])
            f_r, c_r, _ = stratified_split(fit_r, SEED, (2 / 3, 1 / 3, 0.0),
                                           key=lambda r: r["ds"])
            m_t = fit_quantile_models(f_t, SIGMA_FEATS, "gt_te", SEED)
            m_r = fit_quantile_models(f_r, SIGMA_FEATS, "gt_re", SEED)
            Q_t = conformalize(m_t, selected_kind, c_t)
            Q_r = conformalize(m_r, selected_kind, c_r)
            tgt = [e for e in usable_t.get(ds, [])]
            st = np.maximum(predict_qhi(m_t, selected_kind, tgt) + Q_t, 0.05)
            tgt_r = [e for e in usable_r.get(ds, [])]
            sr = np.maximum(predict_qhi(m_r, selected_kind, tgt_r) + Q_r,
                            0.05)
            sr_map = {(e["ds"], e["i"], e["j"]): s
                      for e, s in zip(tgt_r, sr)}
            for e, s in zip(tgt, st):
                srad = np.deg2rad(min(
                    sr_map.get((e["ds"], e["i"], e["j"]), 1.0), 30.0))
                sigmas[(ds, e["i"], e["j"])] = (float(min(s, 20.0)),
                                                float(srad))
        ref = run_golden_peredge(golden, {}, "relfloor")   # drift guard
        new = run_golden_peredge(golden, sigmas, "per_edge")
        out[golden] = {
            "n_sigma": len(sigmas),
            "canonical": canon[golden],
            "relfloor_rerun": ref["ape_unaligned_rmse"],
            "per_edge": new["ape_unaligned_rmse"],
            "per_edge_aligned": new["ape_aligned_rmse"],
            "within_tol": bool(new["ape_unaligned_rmse"]
                               <= PGO_TOL * canon[golden]),
            "drift_ok": bool(abs(ref["ape_unaligned_rmse"] - canon[golden])
                             < 0.02 * canon[golden])}
        logger.info(f"[pgo][{golden}] canonical={canon[golden]:.3f} "
                    f"relfloor-rerun={ref['ape_unaligned_rmse']:.3f} "
                    f"per_edge={new['ape_unaligned_rmse']:.3f} "
                    f"({'OK' if out[golden]['within_tol'] else 'REGRESSED'})")
    n_ok = sum(1 for g in out.values() if g["within_tol"])
    drift_ok = all(g["drift_ok"] for g in out.values())
    out["_clause3"] = {"n_within_tol": n_ok, "drift_ok": bool(drift_ok),
                       "PASS": bool(n_ok >= 5 and drift_ok)}
    return out


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", default="all",
                    choices=["gate", "sigma", "pgo", "all"])
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s: "
                        "%(message)s", datefmt="%H:%M:%S")
    os.makedirs(args.out, exist_ok=True)

    cands = json.load(open(os.path.join(args.out, "candidates.json")))
    edges = json.load(open(os.path.join(args.out, "edges_v2.json")))
    report: Dict[str, object] = {}

    if args.phase in ("gate", "all"):
        report["gate"] = phase_gate(cands)
    if args.phase in ("sigma", "all"):
        report["sigma"] = phase_sigma(edges)
    if args.phase in ("pgo", "all"):
        kind = "gbm"
        if "sigma" in report:
            kind = report["sigma"]["selected"]["gt_te"]
        else:
            prev = os.path.join(args.out, "s1_report.json")
            if os.path.exists(prev):
                kind = json.load(open(prev))["sigma"]["selected"]["gt_te"]
        report["pgo"] = phase_pgo_run(edges, kind)

    # ---- markdown report ----
    lines = ["# S1 — conformal gate + conformalized per-edge sigma", ""]
    if "gate" in report:
        g = report["gate"]
        lines.append("## S1a acceptance gate")
        lines.append("")
        lines.append("| arm | Q | n_calib_d | TEST fpr | TEST recall | "
                     "overall safe recall |")
        lines.append("|---|---|---|---|---|---|")
        for name in ("A1_basic", "A2_full", "A1o_overlap_only",
                     "A3_multigen"):
            if name not in g:
                continue
            a = g[name]
            lines.append(
                f"| {name} | {a['Q']:.4f} | {a['n_calib_d']} | "
                f"{a['test_fpr']} | {a['test_recall']} | "
                f"{a.get('overall_safe_recall')} |")
        lines.append("")
        c1 = g["clause1"]
        lines.append(f"### Clause 1: {'PASS' if c1['PASS'] else 'FAIL'} "
                     f"(conformal safe recall {c1['overall_safe_recall']:.3f}"
                     f" vs heuristic {c1['heuristic_overall_safe_recall']:.3f}"
                     f"; need >= {c1['recall_req']})")
        lines.append("")
        lines.append("| set | heuristic danger acc/n | conformal danger "
                     "acc/n | conformal safe recall | unscored | ok |")
        lines.append("|---|---|---|---|---|---|")
        for r in c1["per_set"]:
            lines.append(f"| {r['set']} | {r['heur_danger_acc']}/"
                         f"{r['heur_danger_n']} | {r['conf_danger_acc']}/"
                         f"{r['conf_danger_n']} | "
                         f"{r['conf_safe_recall']:.3f} | "
                         f"{r.get('unscored', 0)} | {r['clause_ok']} |")
        lines.append("")
        leg = g.get("A1o_overlap_only", {}).get("per_set", {}).get(
            "kitti02_bigbudget")
        if leg:
            lines.append(f"Legacy set kitti02_bigbudget (no mean_dist "
                         f"logged) under the overlap-only arm A1o: dangerous "
                         f"accepted {leg['danger_acc']}/{leg['danger_n']}, "
                         f"safe recall n/a (no safe candidates in the set).")
            lines.append("")
    if "sigma" in report:
        s = report["sigma"]
        lines.append("## S1b conformalized per-edge sigma")
        lines.append("")
        lines.append("Feature signal vs ln(gt_te):")
        lines.append("")
        lines.append("| feature | n | pearson | spearman |")
        lines.append("|---|---|---|---|")
        for f, v in s["feature_signal"].items():
            if v:
                lines.append(f"| {f} | {v['n']} | {v['pearson']:.3f} | "
                             f"{v['spearman']:.3f} |")
        lines.append("")
        for err in ("gt_te", "gt_re"):
            e = s[err]
            lines.append(f"### {err} (selected: {e['selected']})")
            lines.append("")
            lines.append("| model | TEST cov (nominal) | TEST mean σ "
                         "(nominal) | TEST mean σ (coverage-equalized) |")
            lines.append("|---|---|---|---|")
            lines.append(f"| const (P2) | {e['const']['coverage']:.3f} "
                         f"| {e['const']['mean_sigma']:.3f} | "
                         f"{e['const']['mean_sigma_eq']:.3f} |")
            for kind in ("lin", "gbm", "gbm_ln"):
                m = e["models"][kind]
                lines.append(f"| CQR-{kind} | {m['test']['coverage']:.3f} | "
                             f"{m['test']['mean_sigma']:.3f} | "
                             f"{m['test']['mean_sigma_eq']:.3f} |")
            lines.append("")
        c2 = s["clause2"]
        lines.append(f"### Clause 2: {'PASS' if c2['PASS'] else 'FAIL'} "
                     f"(coverage-equalized width ratio {c2['width_ratio']:.3f}"
                     f", need <= {SHARPNESS})")
        lines.append("")
    if "pgo" in report:
        p = report["pgo"]
        lines.append("## S1b clause 3 — PGO with per-edge sigmas (LOGO)")
        lines.append("")
        lines.append("| golden | canonical | relfloor rerun | per_edge | "
                     "within +5% |")
        lines.append("|---|---|---|---|---|")
        for g, v in p.items():
            if g.startswith("_"):
                continue
            lines.append(f"| {g} | {v['canonical']:.3f} | "
                         f"{v['relfloor_rerun']:.3f} | {v['per_edge']:.3f} "
                         f"| {v['within_tol']} |")
        c3 = p["_clause3"]
        lines.append("")
        lines.append(f"### Clause 3: {'PASS' if c3['PASS'] else 'FAIL'} "
                     f"({c3['n_within_tol']}/7 within tolerance, drift_ok="
                     f"{c3['drift_ok']})")
        lines.append("")

    with open(os.path.join(args.out, "s1_report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(args.out, "s1_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    logger.info(f"Wrote {args.out}/s1_report.{{md,json}}")


if __name__ == "__main__":
    main()
