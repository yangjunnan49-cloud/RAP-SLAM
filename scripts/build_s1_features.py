#!/usr/bin/env python
"""S1 prep — build the candidate universe (S1a) and enrich the P2 edge
dataset (S1b) with two NEW pre-acceptance features:

  1. retrieval-ambiguity margin (sc_margin): for candidate (i, j),
         sc_margin = d2 - d1
     where d1 = sc_distance(desc_j, desc_i) and d2 = the best SC distance
     from desc_j to any OTHER keyframe at a DISTINCT site (frame-index
     distance from i >= SITE_EXCL = 100). Small/negative margin means the
     retrieval was ambiguous (repetitive geometry — the P3c warp regime).
     Also stores sc_d1 itself.
  2. degeneracy / localizability (X-ICP-flavoured, Tuna et al. T-RO 2023):
     per scan, voxel-downsample (0.4 m) + 15-NN PCA normals + point-to-plane
     information matrices. Stores per endpoint and per pair (min over the
     two endpoints = the weaker link governs):
         loc_t = lambda_min(sum n n^T) / N                (translational)
         loc_r = lambda_min(sum (p/r_bar x n)(...)^T) / N (rotational)
         scat  = lambda_3 / lambda_1 of the covariance spectrum
     plus pair-level minima deg_loc_t / deg_loc_r / deg_scat.

Outputs (output_s1/):
  candidates.json  — S1a universe: every saved verification candidate
      (accepted + rejected) from the live logs (kitti02 final run,
      gatecheck, bigbudget; kitti06; poss), the re-scored acceptance-gate
      pairs, and the multigen entries — with features and GT labels.
      Label rule (predeclared): gt_te known -> "safe" iff gt_te <= 5 m and
      gt_re <= 10 deg, else "dangerous"; gt_te unknown -> "safe" iff
      gt_dist <= 10 m, "dangerous" iff gt_dist > 20 m, else "gray".
  edges_v2.json    — the P2 378-edge dataset joined with the new features.

Caches (output_s1/cache/): per-dataset SC descriptor stacks (.npz) and
per-scan degeneracy features (.json) so re-runs are cheap.

Run (WSL2 conda rap-inference, CPU):
  python scripts/build_s1_features.py
"""

import json
import logging
import os
import re
import sys
import time
from typing import Dict, List

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from rap_slam import real_dataset, scan_context  # noqa: E402
from eval_paper_metrics import POSS_SEGMENTS, gt_for_live  # noqa: E402
from eval_streaming_retrieval import streaming_keyframes  # noqa: E402
from rerun_pgo_saved import load_tum  # noqa: E402

logger = logging.getLogger("build_s1_features")

OUT = os.path.join(REPO_ROOT, "output_s1")
CACHE = os.path.join(OUT, "cache")

KITTI_ROOT = "/mnt/e/githouse/Datasets/data_odometry_velodyne/dataset"
SITE_EXCL = 100          # frames: distinct-site exclusion for the margin
VOXEL = 0.4              # m, matches the RAP backend downsample
RANGE_MIN, RANGE_MAX = 1.5, 80.0


# ---------------------------------------------------------------------------
# dataset registry
# ---------------------------------------------------------------------------

def _kitti_paths(seq: str):
    seq_dir = os.path.join(KITTI_ROOT, "sequences", seq)
    return (os.path.join(seq_dir, "velodyne"),
            os.path.join(KITTI_ROOT, "poses", f"{seq}.txt"),
            os.path.join(seq_dir, "calib.txt"))


def _load_bin(path: str) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]


class DatasetSpec:
    def __init__(self, name, source, seq, keyframes, gt, frame_offset=0):
        self.name = name
        self.source = source          # "kitti" | "poss"
        self.seq = seq
        self.keyframes = sorted(int(k) for k in keyframes)
        self.gt = gt
        self.frame_offset = frame_offset  # poss segment-relative -> file idx

    def load_scan(self, frame: int) -> np.ndarray:
        fidx = frame + self.frame_offset
        if self.source == "kitti":
            vd, _, _ = _kitti_paths(self.seq)
            return _load_bin(os.path.join(vd, f"{fidx:06d}.bin"))
        return _load_bin(os.path.join(
            REPO_ROOT, "data/poss/dataset/sequences/00/velodyne/"
            f"{fidx:06d}.bin"))

    def gt_dist(self, i: int, j: int) -> float:
        return float(np.linalg.norm(self.gt[i][:3, 3] - self.gt[j][:3, 3]))


def _load_kitti_gt(seq: str, n: int) -> List[np.ndarray]:
    _, poses, calib = _kitti_paths(seq)
    return list(real_dataset.load_gt_poses(poses, calib, start=0, end=n - 1,
                                           apply_calib=True))


def _kf_json(path: str) -> List[int]:
    return [int(k) for k in json.load(open(path))]


def build_registry() -> Dict[str, DatasetSpec]:
    reg: Dict[str, DatasetSpec] = {}

    reg["kitti02"] = DatasetSpec(
        "kitti02", "kitti", "02",
        _kf_json(os.path.join(REPO_ROOT,
                              "output_ros2/step3_kitti02/keyframes.json")),
        _load_kitti_gt("02", 4661))

    for seq, n, kf_path, name in (
            ("06", 1101, "output_ros2/step3_kitti06/keyframes.json",
             "kitti06_live"),
            ("06", 1101, "output_prods/kitti06/keyframes.json", "kitti06_off"),
            ("05", 2761, "output_prods/kitti05/keyframes.json", "kitti05"),
            ("00", 4541, "output_prods/kitti00_bb/keyframes.json", "kitti00")):
        reg[name] = DatasetSpec(name, "kitti", seq,
                                _kf_json(os.path.join(REPO_ROOT, kf_path)),
                                _load_kitti_gt(seq, n))

    # POSS segment A: offline + live products share the same scans; frame
    # indices are segment-relative (file = 3850 + idx).
    start_a, _ = POSS_SEGMENTS["A"]
    gt_poss = gt_for_live("poss", "A", 1419)
    reg["poss_off"] = DatasetSpec(
        "poss_off", "poss", "A",
        _kf_json(os.path.join(REPO_ROOT, "output_prods/poss/keyframes.json")),
        gt_poss, frame_offset=start_a)
    reg["poss_live"] = DatasetSpec(
        "poss_live", "poss", "A",
        _kf_json(os.path.join(REPO_ROOT,
                              "output_ros2/step3/keyframes.json")),
        gt_poss, frame_offset=start_a)

    # POSS segment B: the smoke run saved no keyframes.json — reconstruct
    # with the streaming keyframe rule on the saved odometry poses.
    start_b, _ = POSS_SEGMENTS["B"]
    _, odom_b = load_tum(os.path.join(REPO_ROOT,
                                      "output_possb_smoke/odom_poses.txt"))
    kf_b = streaming_keyframes(odom_b, 6.0, 12.0)
    reg["possb"] = DatasetSpec("possb", "poss", "B", kf_b, None,
                               frame_offset=start_b)
    logger.info(f"possb keyframes reconstructed: {len(kf_b)}")
    return reg


# ---------------------------------------------------------------------------
# candidate parsing (live logs + re-scored sets)
# ---------------------------------------------------------------------------

PAT_NEW = re.compile(
    r"pair \(\s*(\d+),\s*(\d+)\): overlap=([\d.]+) mean_dist=([\d.]+) m "
    r"odom-consistency dT=([\d.]+) m dR=([\d.]+) deg .*? => (ACCEPT|reject)")
PAT_OLD = re.compile(
    r"pair \(\s*(\d+),\s*(\d+)\): overlap=([\d.]+) "
    r"odom-consistency dT=([\d.]+) m dR=([\d.]+) deg .*? => (ACCEPT|reject)")
PAT_AG = re.compile(
    r"\(\s*(\d+),\s*(\d+)\)\s+(TRUE|FALSE)\s+([\d.]+)\s+([\d.]+)\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)")


def parse_log(path: str, pat) -> List[dict]:
    out = []
    for line in open(path, encoding="utf-8", errors="ignore"):
        m = pat.search(line)
        if not m:
            continue
        g = m.groups()
        if pat is PAT_NEW:
            out.append({"i": int(g[0]), "j": int(g[1]),
                        "overlap": float(g[2]), "mean_dist": float(g[3]),
                        "odom_dT": float(g[4]), "odom_dR": float(g[5]),
                        "verdict": "accept" if g[6] == "ACCEPT" else "reject"})
        else:
            out.append({"i": int(g[0]), "j": int(g[1]),
                        "overlap": float(g[2]), "mean_dist": None,
                        "odom_dT": float(g[3]), "odom_dR": float(g[4]),
                        "verdict": "accept" if g[5] == "ACCEPT" else "reject"})
    return out


def assemble_candidates(reg: Dict[str, DatasetSpec]) -> List[dict]:
    cands: List[dict] = []

    def add(ds, rec):
        rec = dict(rec)
        rec["ds"] = ds
        cands.append(rec)

    # 1) live logs (kitti02 final clean run, gatecheck, bigbudget, poss, 06)
    for ds, log, pat in (
            ("kitti02", "loop_node_kitti02.log", PAT_NEW),
            ("kitti02_gatecheck", "loop_node_kitti02_gatecheck.log", PAT_NEW),
            ("kitti02_bigbudget", "loop_node_kitti02_bigbudget.log", PAT_OLD),
            ("poss_live", "loop_node.log", PAT_NEW),
            ("kitti06_live", "loop_node_kitti06.log", PAT_NEW)):
        for rec in parse_log(os.path.join(REPO_ROOT, "output_ros2", log), pat):
            add(ds, rec)

    # kitti06 live: the surviving log is a 43-accept fragment of the
    # canonical 114-accept run -> union with the canonical product.
    canon06 = json.load(open(os.path.join(
        REPO_ROOT, "output_ros2/step3_kitti06/accepted_loops.json")))
    have06 = {(c["i"], c["j"]) for c in cands if c["ds"] == "kitti06_live"
              and c["verdict"] == "accept"}
    for lc in canon06:
        if (lc["i"], lc["j"]) not in have06:
            add("kitti06_live",
                {"i": lc["i"], "j": lc["j"], "overlap": lc["overlap"],
                 "mean_dist": lc.get("mean_dist"),
                 "odom_dT": lc.get("odom_trans_err"),
                 "odom_dR": lc.get("odom_rot_err_deg"),
                 "verdict": "accept"})

    # 2) re-scored acceptance-gate pairs (batch RAP, GT-errored)
    for line in open(os.path.join(REPO_ROOT,
                                  "output_ros2/eval_acceptance_gate.log"),
                     encoding="utf-8", errors="ignore"):
        m = PAT_AG.search(line)
        if m:
            g = m.groups()
            add("acceptgate", {"i": int(g[0]), "j": int(g[1]),
                               "overlap": float(g[3]),
                               "mean_dist": float(g[4]),
                               "gt_dist": float(g[5]),
                               "gt_te": float(g[6]), "gt_re": float(g[7]),
                               "verdict": "accept" if g[2] == "TRUE"
                               else "reject",
                               "ds_ref": "kitti02"})

    # 3) multigen entries (4 gens/pair -> dispersion + min-overlap features)
    mg = json.load(open(os.path.join(
        REPO_ROOT, "output_multigen/multigen_samples.json")))["entries"]
    ds_map = {"kitti02_true": ("multigen_k02true", "kitti02"),
              "kitti02_false": ("multigen_k02false", "kitti02"),
              "poss_true": ("multigen_poss", "poss_live")}
    for e in mg:
        ovs = [g["overlap"] for g in e["gens"]]
        mds = [g["mean_dist"] for g in e["gens"]]
        Ts = [np.asarray(g["T_rel"], dtype=float) for g in e["gens"]]
        dt, dr = [], []
        for a in range(len(Ts)):
            for b in range(a + 1, len(Ts)):
                d = Ts[a] @ np.linalg.inv(Ts[b])
                dt.append(float(np.linalg.norm(d[:3, 3])))
                dr.append(float(np.rad2deg(np.arccos(np.clip(
                    (np.trace(d[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)))))
        ds_name, ds_ref = ds_map[e["ds"]]
        add(ds_name,
            {"i": e["i"], "j": e["j"], "overlap": float(np.max(ovs)),
             "mean_dist": float(np.min(mds)),
             "verdict": "accept" if "true" in e["ds"] else "reject",
             "overlap_min": float(np.min(ovs)),
             "overlap_mean": float(np.mean(ovs)),
             "disp_t": float(np.mean(dt)), "disp_r": float(np.mean(dr)),
             "ds_ref": ds_ref})
    return cands


# ---------------------------------------------------------------------------
# GT labels
# ---------------------------------------------------------------------------

# candidate-set name -> registry name for GT / features
CAND_REF = {"kitti02": "kitti02", "kitti02_gatecheck": "kitti02",
            "kitti02_bigbudget": "kitti02", "poss_live": "poss_live",
            "kitti06_live": "kitti06_live", "acceptgate": "kitti02",
            "multigen_k02true": "kitti02", "multigen_k02false": "kitti02",
            "multigen_poss": "poss_live"}
# registry name -> P2 edge-dataset ds name (for gt_te join)
EDGE_DS = {"kitti02": "kitti02", "poss_live": "poss_live",
           "kitti06_live": "kitti06_live"}


def attach_labels(cands: List[dict], reg: Dict[str, DatasetSpec]) -> None:
    edges = json.load(open(os.path.join(
        REPO_ROOT, "output_calibrated/edge_dataset.json")))
    lut = {(e["ds"], e["i"], e["j"]): e for e in edges}
    ag = [c for c in cands if c["ds"] == "acceptgate"]

    for c in cands:
        ref = CAND_REF[c["ds"]]
        spec = reg[ref]
        if spec.gt is not None and "gt_dist" not in c:
            try:
                c["gt_dist"] = spec.gt_dist(c["i"], c["j"])
            except IndexError:
                c["gt_dist"] = None
        # join known GT errors from the P2 edge dataset
        ed = EDGE_DS.get(ref)
        if ed and (ed, c["i"], c["j"]) in lut:
            e = lut[(ed, c["i"], c["j"])]
            c.setdefault("gt_te", e["gt_te"])
            c.setdefault("gt_re", e["gt_re"])
        # bigbudget's 11 accepts == the acceptgate FALSE pairs (same (i, j))
        if c["ds"] == "kitti02_bigbudget" and "gt_te" not in c:
            for c2 in ag:
                if c2["i"] == c["i"] and c2["j"] == c["j"]:
                    c["gt_te"], c["gt_re"] = c2["gt_te"], c2["gt_re"]
                    break
        te, re_, gd = c.get("gt_te"), c.get("gt_re"), c.get("gt_dist")
        if te is not None and re_ is not None:
            c["label"] = ("safe" if (te <= 5.0 and re_ <= 10.0)
                          else "dangerous")
        elif gd is not None:
            c["label"] = ("safe" if gd <= 10.0 else
                          "dangerous" if gd > 20.0 else "gray")
        else:
            c["label"] = "gray"


# ---------------------------------------------------------------------------
# feature computation (cached)
# ---------------------------------------------------------------------------

def _voxel_down(pts: np.ndarray, voxel: float) -> np.ndarray:
    keys = np.floor(pts / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[np.sort(idx)]


def degeneracy_features(pts_raw: np.ndarray) -> dict:
    from scipy.spatial import cKDTree
    p = np.asarray(pts_raw, dtype=np.float64)
    r3 = np.linalg.norm(p, axis=1)
    p = p[(r3 >= RANGE_MIN) & (r3 <= RANGE_MAX)]
    if len(p) < 200:
        return {"n": int(len(p))}
    q = _voxel_down(p, VOXEL)
    if len(q) < 100:
        return {"n": int(len(q))}
    c = q - q.mean(axis=0)
    ev = np.linalg.eigvalsh((c.T @ c) / len(q))[::-1]
    scat = float(ev[2] / max(ev[0], 1e-12))
    tree = cKDTree(q)
    _, nn = tree.query(q, k=min(16, len(q)))
    loc = q[nn] - q[:, None, :]
    cov = np.einsum("nki,nkj->nij", loc, loc) / loc.shape[1]
    _, v = np.linalg.eigh(cov)
    normals = v[:, :, 0]                    # smallest-eigenvalue direction
    r_bar = max(float(np.linalg.norm(q, axis=1).mean()), 1.0)
    cr = np.cross(q / r_bar, normals)
    Ht = np.einsum("ni,nj->ij", normals, normals)
    Hr = np.einsum("ni,nj->ij", cr, cr)
    n = float(len(q))
    return {"n": int(n), "scat": scat,
            "loc_t": float(np.linalg.eigvalsh(Ht)[0] / n),
            "loc_r": float(np.linalg.eigvalsh(Hr)[0] / n)}


def compute_features(reg: Dict[str, DatasetSpec],
                     needed: Dict[str, set]) -> None:
    os.makedirs(CACHE, exist_ok=True)
    for name, frames in sorted(needed.items()):
        spec = reg[name]
        deg_path = os.path.join(CACHE, f"deg_{name}.json")
        sc_path = os.path.join(CACHE, f"sc_{name}.npz")
        deg = json.load(open(deg_path)) if os.path.exists(deg_path) else {}
        scs: Dict[int, np.ndarray] = {}
        if os.path.exists(sc_path):
            z = np.load(sc_path)
            scs = {int(k[1:]): z[k] for k in z.files}
        t0 = time.time()
        todo = sorted(f for f in frames if str(f) not in deg or f not in scs)
        logger.info(f"[{name}] {len(todo)} scans to process "
                    f"(cached: {len(scs)} SC, {len(deg)} deg)")
        for k, f in enumerate(todo):
            pts = spec.load_scan(f)
            if f not in scs:
                scs[f] = scan_context.make_scan_context(pts)
            if str(f) not in deg:
                deg[str(f)] = degeneracy_features(pts)
            if (k + 1) % 200 == 0:
                logger.info(f"[{name}] {k + 1}/{len(todo)} "
                            f"({time.time() - t0:.0f} s)")
        np.savez_compressed(sc_path, **{f"f{k2}": v for k2, v
                                        in scs.items()})
        json.dump(deg, open(deg_path, "w"))
        logger.info(f"[{name}] cached {len(scs)} SC + {len(deg)} deg "
                    f"({time.time() - t0:.0f} s)")


def load_cache(name: str):
    z = np.load(os.path.join(CACHE, f"sc_{name}.npz"))
    scs = {int(k[1:]): z[k] for k in z.files}
    deg = json.load(open(os.path.join(CACHE, f"deg_{name}.json")))
    return scs, deg


def attach_sc_margin(rec: dict, spec: DatasetSpec,
                     scs: Dict[int, np.ndarray]) -> None:
    i, j = rec["i"], rec["j"]
    if i not in scs or j not in scs:
        rec["sc_d1"] = rec["sc_margin"] = None
        return
    db_frames = [f for f in spec.keyframes
                 if f in scs and abs(f - i) >= SITE_EXCL]
    if not db_frames:
        rec["sc_d1"] = rec["sc_margin"] = None
        return
    d1, _ = scan_context.sc_distance(scs[i], scs[j])
    stack = np.stack([scs[f] for f in db_frames])
    dists, _ = scan_context.sc_distances_batch(stack, scs[j])
    rec["sc_d1"] = float(d1)
    rec["sc_margin"] = float(np.min(dists)) - float(d1)


def attach_deg(rec: dict, deg: dict) -> None:
    a, b = deg.get(str(rec["i"]), {}), deg.get(str(rec["j"]), {})
    for k in ("loc_t", "loc_r", "scat"):
        va, vb = a.get(k), b.get(k)
        rec[f"deg_{k}_i"], rec[f"deg_{k}_j"] = va, vb
        rec[f"deg_{k}"] = (min(va, vb) if va is not None and vb is not None
                           else None)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s: "
                        "%(message)s", datefmt="%H:%M:%S")
    os.makedirs(OUT, exist_ok=True)
    reg = build_registry()
    logger.info("registry: " + ", ".join(
        f"{k}({len(v.keyframes)} kf)" for k, v in reg.items()))

    cands = assemble_candidates(reg)
    by_ds = {}
    for c in cands:
        by_ds.setdefault(c["ds"], []).append(c)
    logger.info("candidates: " + ", ".join(
        f"{k}={len(v)}" for k, v in sorted(by_ds.items())))

    attach_labels(cands, reg)
    n_lab = {}
    for c in cands:
        n_lab[c["label"]] = n_lab.get(c["label"], 0) + 1
    logger.info(f"labels: {n_lab}")

    edges = json.load(open(os.path.join(
        REPO_ROOT, "output_calibrated/edge_dataset.json")))
    edge_ref = {"kitti02": "kitti02", "poss_off": "poss_off",
                "poss_live": "poss_live", "possb": "possb",
                "kitti00": "kitti00", "kitti05": "kitti05",
                "kitti06_off": "kitti06_off", "kitti06_live": "kitti06_live"}

    needed: Dict[str, set] = {k: set(v.keyframes) for k, v in reg.items()}
    for c in cands:
        needed[CAND_REF[c["ds"]]].update((c["i"], c["j"]))
    for e in edges:
        needed[edge_ref[e["ds"]]].update((e["i"], e["j"]))
    logger.info("scans needed: " + ", ".join(
        f"{k}={len(v)}" for k, v in sorted(needed.items())))
    compute_features(reg, needed)

    caches = {name: load_cache(name) for name in needed}
    for c in cands:
        ref = CAND_REF[c["ds"]]
        scs, deg = caches[ref]
        attach_sc_margin(c, reg[ref], scs)
        attach_deg(c, deg)
    for e in edges:
        ref = edge_ref[e["ds"]]
        scs, deg = caches[ref]
        attach_sc_margin(e, reg[ref], scs)
        attach_deg(e, deg)

    with open(os.path.join(OUT, "candidates.json"), "w") as f:
        json.dump(cands, f)
    with open(os.path.join(OUT, "edges_v2.json"), "w") as f:
        json.dump(edges, f)
    n_feat = sum(1 for c in cands if c.get("sc_margin") is not None)
    logger.info(f"Wrote {OUT}/candidates.json ({len(cands)}; "
                f"{n_feat} with sc features) + edges_v2.json ({len(edges)})")


if __name__ == "__main__":
    main()
