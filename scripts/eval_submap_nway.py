#!/usr/bin/env python
"""Direction A (RAP-CORE, owner-approved 2026-08-09): submap-level N-way SLAM.

RC-4 (scripts/eval_hier_nway.py) jointly registered the raw single scans of
each loop cluster — no APE gain on KITTI-02/POSS. This script evaluates the
approved next step: register SUBMAPS (odometry-accumulated local scans around
each keyframe) instead of raw scans, giving RAP denser geometry and larger
overlap context, then run PGO over the submap graph.

Arms per dataset/seed:
  pairwise       : accepted RAP edges as-is + PGO (baseline, relfloor+huber
                   defaults — reproduces the offline pipeline result)
  nway_raw       : RC-4 arm — ONE register_multi per cluster over the members'
                   RAW scans; accepted edges' T_rel replaced by the
                   cluster-consistent inv(T_i) @ T_j; PGO
  submap         : same, but parts are SUBMAPS (--submap-half-width scans to
                   each side of the keyframe, accumulated with odometry
                   relatives, voxel-downsampled)
  submap_clique  : reuses the submap joint registration, but ALL intra-cluster
                   member pairs (sep >= 200, like loops) are added as
                   constraints (cycle-consistent clique)

HAZARD GUARD (learned the hard way on kitti06_live, 2026-08-09): a degree-
trimmed street-CHAIN cluster can warp the joint registration (97-member
cluster -> 16 parts: layout rmse 67-69 m), whose relatives are self-
consistent garbage; ungated clique edges from it blew PGO to 135.7 m
(baseline 2.581). BOTH replaced and clique edges therefore pass through an
intrinsic CATASTROPHE re-gate (overlap >= 0.5 AND mean_dist <= 2.0 m
measured on the preprocessed parts under the candidate T_rel — NOT the
strict 0.7/2.5 retrieval gate, which misfires on submap parts; empirical
basis scripts/eval_submap_regate.py: good edges ov >= 0.588 / md <= 1.75,
warped edges ov <= 0.486 / md >= 2.14). Chain-aware batch composition
(sliding windows over the chain instead of degree-trim) is the remaining
follow-up.

Metrics per arm:
  * full-trajectory APE (evo, first-pose frame) — primary axis
  * map consistency (owner's second axis): symmetric truncated Chamfer
    distance to the GT-posed map on the same scans + mean map entropy (MME)
  * loop-site alignment: symmetric NN stats of accepted-loop scan pairs at
    corrected poses (cycle consistency measured on the map)
  * per-cluster oracle SE(3) layout fit to GT (warp check, as in RC-4)

Datasets: offline pipeline products (run_pipeline.py dumps keyframes.json +
accepted_loops.json since 2026-08-09): poss / kitti06 / kitti05 / kitti00_bb.

Run in the conda env:
  /root/miniconda3/envs/rap-inference/bin/python scripts/eval_submap_nway.py \
      --datasets poss,kitti06,kitti05,kitti00_bb --seeds 42,43
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

from rap_slam import evaluate, pose_graph, real_dataset, sequence  # noqa: E402

logger = logging.getLogger("eval_submap_nway")

KITTI_ROOT = "/mnt/e/githouse/Datasets/data_odometry_velodyne/dataset"
POSS_ROOT = "/mnt/e/githouse/RAP-SLAM/data/poss/dataset"

# Dataset registry: products dir (output_prods/<name>), GT poses, velodyne
# dir, frame range, and the offset between product indices and .bin filenames
# (POSS products are segment-relative; files are absolute).
DATASETS = {
    "poss": dict(
        prods="output_prods/poss",
        poses_file=os.path.join(POSS_ROOT, "poses/00.txt"),
        calib_file=os.path.join(POSS_ROOT, "sequences/00/calib.txt"),
        apply_calib=False,
        velodyne=os.path.join(POSS_ROOT, "sequences/00/velodyne"),
        start=3850, end=5268, file_offset=3850, map_stride=3,
    ),
    "kitti06": dict(
        prods="output_prods/kitti06",
        poses_file=os.path.join(KITTI_ROOT, "poses/06.txt"),
        calib_file=os.path.join(KITTI_ROOT, "sequences/06/calib.txt"),
        apply_calib=True,
        velodyne=os.path.join(KITTI_ROOT, "sequences/06/velodyne"),
        start=0, end=1100, file_offset=0, map_stride=3,
    ),
    "kitti05": dict(
        prods="output_prods/kitti05",
        poses_file=os.path.join(KITTI_ROOT, "poses/05.txt"),
        calib_file=os.path.join(KITTI_ROOT, "sequences/05/calib.txt"),
        apply_calib=True,
        velodyne=os.path.join(KITTI_ROOT, "sequences/05/velodyne"),
        start=0, end=2760, file_offset=0, map_stride=5,
    ),
    "kitti00_bb": dict(
        prods="output_prods/kitti00_bb",
        poses_file=os.path.join(KITTI_ROOT, "poses/00.txt"),
        calib_file=os.path.join(KITTI_ROOT, "sequences/00/calib.txt"),
        apply_calib=True,
        velodyne=os.path.join(KITTI_ROOT, "sequences/00/velodyne"),
        start=0, end=4540, file_offset=0, map_stride=5,
    ),
    # KITTI-06 LIVE products (ROS 2 Step-3 stack): 114 accepted edges all at
    # the single dense revisit site — the offline budget-capped edge set is
    # 12 DISJOINT 2-member pairs (no >=3 clusters, all arms degenerate), so
    # the dense live edge set is the only place KITTI-06 exercises N-way.
    # Live poses are TUM format; baseline matches the live default
    # loop_nms_width=10 (dense-edge pathology fix, 2026-08-08).
    "kitti06_live": dict(
        prods="output_ros2/step3_kitti06",
        odom_poses="output_ros2/live_poses_kitti06.txt",
        odom_format="tum",
        poses_file=os.path.join(KITTI_ROOT, "poses/06.txt"),
        calib_file=os.path.join(KITTI_ROOT, "sequences/06/calib.txt"),
        apply_calib=True,
        velodyne=os.path.join(KITTI_ROOT, "sequences/06/velodyne"),
        start=0, end=1100, file_offset=0, map_stride=3,
        loop_nms_width=10.0,
    ),
}


def parse_args():
    p = argparse.ArgumentParser(description="Submap-level N-way SLAM eval")
    p.add_argument("--datasets", default="poss,kitti06,kitti05,kitti00_bb")
    p.add_argument("--arms", default="nway_raw,submap,submap_clique")
    p.add_argument("--seeds", default="42,43")
    p.add_argument("--min-cluster", type=int, default=3,
                   help="clusters smaller than this keep pairwise measurements")
    p.add_argument("--max-parts", type=int, default=16,
                   help="max parts per register_multi call (bigger clusters "
                        "are split per --compose)")
    p.add_argument("--compose", choices=["degree", "chain"], default="degree",
                   help="batch composition for clusters bigger than "
                        "--max-parts. degree: keep the N highest-degree "
                        "members (legacy — SHATTERS street chains: on "
                        "kitti06_live's 97-member chain it left ZERO edges "
                        "with both endpoints covered and warped the joint "
                        "registration). chain: overlapping sliding windows "
                        "over the PCA-1 chain order (chain-preserving)")
    p.add_argument("--chain-overlap", type=int, default=0,
                   help="overlap between chain windows; 0 = max_parts//2")
    p.add_argument("--points-budget", type=int, default=30000,
                   help="total points per cluster call; per-part cap = "
                        "min(4000, budget // n_parts)")
    p.add_argument("--submap-half-width", type=int, default=4,
                   help="submap = scans [kf-w .. kf+w] accumulated with "
                        "odometry (default 4 -> 9 scans)")
    p.add_argument("--submap-voxel", type=float, default=0.2,
                   help="pre-voxel size for the accumulated submap (m)")
    p.add_argument("--submap-max-points", type=int, default=30000,
                   help="cap on submap points before register_multi")
    p.add_argument("--scan-min-range", type=float, default=1.5)
    p.add_argument("--scan-max-range", type=float, default=80.0)
    p.add_argument("--map-voxel", type=float, default=0.4)
    p.add_argument("--map-cap", type=int, default=2000000)
    p.add_argument("--map-seeds", default="42",
                   help="seeds that also get map-consistency metrics")
    p.add_argument("--skip-maps", action="store_true")
    p.add_argument("--reuse-cache", action="store_true")
    p.add_argument("--output", default=os.path.join(REPO_ROOT,
                                                    "output_submap_nway"))
    return p.parse_args()


# ------------------------------------------------------------------ helpers
def load_tum(path):
    """TUM format (timestamp tx ty tz qx qy qz qw) -> list of 4x4."""
    rows = np.loadtxt(path)
    poses = []
    for r in rows:
        qx, qy, qz, qw = r[4:8]
        n = qx * qx + qy * qy + qz * qz + qw * qw
        s = 2.0 / n
        R = np.array([
            [1 - s * (qy * qy + qz * qz), s * (qx * qy - qz * qw),
             s * (qx * qz + qy * qw)],
            [s * (qx * qy + qz * qw), 1 - s * (qx * qx + qz * qz),
             s * (qy * qz - qx * qw)],
            [s * (qx * qz - qy * qw), s * (qy * qz + qx * qw),
             1 - s * (qx * qx + qy * qy)]])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = r[1:4]
        poses.append(T)
    return poses


def _rot_err_deg(R):
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def rel_err(T_est, T_i_gt, T_j_gt):
    dG = np.linalg.inv(np.linalg.inv(T_i_gt) @ T_j_gt) @ T_est
    return float(np.linalg.norm(dG[:3, 3])), _rot_err_deg(dG[:3, :3])


def umeyama_se3(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    H = (src - mu_s).T @ (dst - mu_d)
    U, _, Vt = np.linalg.svd(H)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ D @ U.T
    return R, mu_d - R @ mu_s


def mine_clusters(loops):
    """Union-find over loop endpoints -> list of dicts (id, members, edges)."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for lc in loops:
        ra, rb = find(lc["i"]), find(lc["j"])
        if ra != rb:
            parent[ra] = rb
    comps = {}
    for k, lc in enumerate(loops):
        comps.setdefault(find(lc["i"]), []).append(k)
    clusters = []
    for cid, edge_ids in enumerate(sorted(comps.values(), key=lambda e: -len(e))):
        members = sorted({s for k in edge_ids
                          for s in (loops[k]["i"], loops[k]["j"])})
        deg = {}
        for k in edge_ids:
            for s in (loops[k]["i"], loops[k]["j"]):
                deg[s] = deg.get(s, 0) + 1
        clusters.append({"id": cid, "members": members, "edges": edge_ids,
                         "deg": deg})
    return clusters


def chain_order(members, odom_poses):
    """Order cluster members along the dominant spatial axis.

    Street-chain clusters (a road revisited in the same or the opposite
    direction) need chain-preserving batching: sorting members by their
    PCA-1 projection of the odometry positions interleaves the passes in
    spatial order, so sliding windows cover contiguous street segments and
    loop edges — spatially adjacent by construction — keep both endpoints
    inside one window.
    """
    pos = np.array([odom_poses[m][:3, 3] for m in members], dtype=float)
    c = pos - pos.mean(axis=0, keepdims=True)
    v1 = np.linalg.svd(c, full_matrices=False)[2][0]
    s = c @ v1
    return [m for _, m in sorted(zip(s.tolist(), members))]


def compose_batches(members, deg, odom_poses, max_parts, compose,
                    chain_overlap=0):
    """Split a cluster's members into <=max_parts batches for register_multi.

    Returns a list of sorted member lists.
      degree: single batch of the highest-degree members (legacy).
      chain:  overlapping sliding windows (size max_parts, stride
              max_parts - overlap) over the PCA-1 chain order. Any member
              pair within (max_parts - overlap) in chain order — in
              particular spatially adjacent loop endpoints — shares a
              window, so replacement edges always get coverage.
    """
    if len(members) <= max_parts:
        return [sorted(members)]
    if compose == "degree":
        return [sorted(sorted(members, key=lambda s: -deg[s])[:max_parts])]
    order = chain_order(members, odom_poses)
    overlap = chain_overlap if chain_overlap > 0 else max_parts // 2
    stride = max(1, max_parts - overlap)
    last = len(order) - max_parts
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)  # tail window, full size
    return [sorted(order[s:s + max_parts]) for s in starts]


def full_traj_ape(odom_poses, keyframes, optimized, gt_rel):
    opt = evaluate.align_to_first_pose(
        pose_graph.propagate_to_full_trajectory(odom_poses, keyframes,
                                                optimized))
    return evaluate.compute_ape_rpe(gt_rel, opt)["ape_rmse"]


# ------------------------------------------------------------------ dataset
class Dataset:
    def __init__(self, name, cfg, args):
        self.name = name
        self.cfg = cfg
        self.nms = float(cfg.get("loop_nms_width", 0.0))
        odom_path = os.path.join(
            REPO_ROOT, cfg.get("odom_poses",
                               os.path.join(cfg["prods"], "odom_poses.txt")))
        if cfg.get("odom_format", "kitti") == "tum":
            self.odom_poses = load_tum(odom_path)
        else:
            self.odom_poses = [T for T in sequence.load_kitti_poses(odom_path)]
        prods = os.path.join(REPO_ROOT, cfg["prods"])
        self.keyframes = json.load(open(os.path.join(prods, "keyframes.json")))
        self.loops = json.load(open(os.path.join(prods, "accepted_loops.json")))
        for lc in self.loops:
            lc["T_rel"] = np.array(lc["T_rel"])
        self.gt = list(real_dataset.load_gt_poses(
            cfg["poses_file"], cfg["calib_file"],
            start=cfg["start"], end=cfg["end"],
            apply_calib=cfg["apply_calib"]))
        n = len(self.odom_poses)
        assert len(self.gt) == n, f"{name}: GT {len(self.gt)} != odom {n}"
        self.gt_rel = evaluate.align_to_first_pose(self.gt)
        self.odom_rel = evaluate.align_to_first_pose(self.odom_poses)
        self.ape_odom = evaluate.compute_ape_rpe(
            self.gt_rel, self.odom_rel)["ape_rmse"]
        self.clusters = mine_clusters(self.loops)
        self._scan_cache = {}
        self.args = args
        logger.info(f"{name}: frames={n} keyframes={len(self.keyframes)} "
                    f"loops={len(self.loops)} clusters="
                    f"{[(len(c['members']), len(c['edges'])) for c in self.clusters]}"
                    f" odom APE={self.ape_odom:.3f} m")

    def load_scan(self, idx):
        """Raw scan (N, 3) with the pipeline's range filter, cached."""
        if idx not in self._scan_cache:
            path = os.path.join(
                self.cfg["velodyne"],
                f"{idx + self.cfg['file_offset']:06d}.bin")
            pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]
            r = np.linalg.norm(pts, axis=1)
            keep = np.ones(len(pts), dtype=bool)
            if self.args.scan_min_range > 0:
                keep &= r >= self.args.scan_min_range
            if self.args.scan_max_range > 0:
                keep &= r <= self.args.scan_max_range
            self._scan_cache[idx] = pts[keep].astype(np.float32)
        return self._scan_cache[idx]

    def drop_scan_cache(self):
        self._scan_cache = {}

    def build_submap(self, center, half_width, voxel, max_points, rng):
        """Accumulate scans [center-w .. center+w] into the center frame via
        odometry relatives, voxel-downsample, cap points."""
        import open3d as o3d
        T_c_inv = np.linalg.inv(self.odom_poses[center])
        lo = max(0, center - half_width)
        hi = min(len(self.odom_poses) - 1, center + half_width)
        clouds = []
        for s in range(lo, hi + 1):
            pts = self.load_scan(s)
            T_rel = T_c_inv @ self.odom_poses[s]
            clouds.append(((T_rel[:3, :3] @ pts.T).T
                           + T_rel[:3, 3]).astype(np.float32))
        sub = np.concatenate(clouds)
        if voxel > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(sub.astype(np.float64))
            pcd = pcd.voxel_down_sample(voxel)
            sub = np.asarray(pcd.points, dtype=np.float32)
        if len(sub) > max_points:
            sel = rng.choice(len(sub), max_points, replace=False)
            sub = sub[sel]
        return sub


# ------------------------------------------------------------------ metrics
def build_map_points(dataset, scan_ids, poses, voxel, cap, chunk=80):
    """Merge scans at the given poses; chunked voxel-downsample; cap."""
    import open3d as o3d
    parts = []
    for c0 in range(0, len(scan_ids), chunk):
        clouds = []
        for s in scan_ids[c0:c0 + chunk]:
            pts = dataset.load_scan(s)
            T = poses[s]
            clouds.append(((T[:3, :3] @ pts.T).T + T[:3, 3]).astype(np.float32))
        sub = np.concatenate(clouds)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(sub.astype(np.float64))
        pcd = pcd.voxel_down_sample(voxel)
        parts.append(np.asarray(pcd.points, dtype=np.float32))
        dataset.drop_scan_cache()
    pts = np.concatenate(parts)
    if len(pts) > cap:
        rng = np.random.default_rng(0)
        pts = pts[rng.choice(len(pts), cap, replace=False)]
    return pts


def chamfer_to_gt(est_pts, gt_pts, n_query=30000, cap_dist=2.0, recall_at=0.5):
    """Symmetric truncated Chamfer + recall: est map vs GT-posed map."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(0)
    q1 = est_pts[rng.choice(len(est_pts), min(n_query, len(est_pts)),
                            replace=False)]
    q2 = gt_pts[rng.choice(len(gt_pts), min(n_query, len(gt_pts)),
                           replace=False)]
    d1, _ = cKDTree(gt_pts).query(q1, k=1)
    d2, _ = cKDTree(est_pts).query(q2, k=1)
    cham = 0.5 * (float(np.minimum(d1, cap_dist).mean())
                  + float(np.minimum(d2, cap_dist).mean()))
    rec = 0.5 * (float(np.mean(d1 < recall_at)) + float(np.mean(d2 < recall_at)))
    return cham, rec


def mean_map_entropy(pts, n_query=20000, k=10):
    """Mean map entropy (Droeschel & Behnke): mean over sampled points of the
    differential entropy 0.5*ln(det(2*pi*e*Cov)) of the k-NN neighborhood."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(0)
    q = pts[rng.choice(len(pts), min(n_query, len(pts)), replace=False)]
    _, idx = cKDTree(pts).query(q, k=k + 1)
    nb = pts[idx[:, 1:]]                              # (Q, k, 3)
    nb = nb - nb.mean(axis=1, keepdims=True)
    cov = np.einsum("qki,qkj->qij", nb, nb) / nb.shape[1]
    cov += np.eye(3)[None] * 1e-6
    sign, logdet = np.linalg.slogdet(cov)
    h = 0.5 * (3.0 * np.log(2.0 * np.pi * np.e) + logdet)
    return float(np.mean(h))


def prep_for_metrics(pts, cap=4000):
    """Voxel-0.4 downsample + random cap, matching RAP preprocess scale."""
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd = pcd.voxel_down_sample(0.4)
    ds = np.asarray(pcd.points, dtype=np.float32)
    if len(ds) > cap:
        rng = np.random.default_rng(0)
        ds = ds[rng.choice(len(ds), cap, replace=False)]
    return ds


def loop_site_alignment(dataset, loops, opt_poses_rel):
    """Median overlap / mean_dist of accepted-loop scan pairs at corrected
    poses (map-level cycle consistency at revisit sites)."""
    from rap_slam.rap_backend import RAPBackend
    ovs, mds = [], []
    for lc in loops:
        i, j = lc["i"], lc["j"]
        ds_i = prep_for_metrics(dataset.load_scan(i))
        ds_j = prep_for_metrics(dataset.load_scan(j))
        T_rel = np.linalg.inv(opt_poses_rel[i]) @ opt_poses_rel[j]
        ov, md = RAPBackend._alignment_metrics(ds_i, ds_j, T_rel, thresh=1.0)
        ovs.append(ov)
        mds.append(md)
    dataset.drop_scan_cache()
    return (float(np.median(ovs)) if ovs else None,
            float(np.median(mds)) if mds else None)


# ------------------------------------------------------------------ main
def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    t0_all = time.time()

    names = args.datasets.split(",")
    arm_list = args.arms.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    map_seeds = {int(s) for s in args.map_seeds.split(",")}

    datasets = {}
    for name in names:
        datasets[name] = Dataset(name, DATASETS[name], args)

    # ------------------------------------------------ baseline pairwise PGO
    results = {}
    for name, D in datasets.items():
        opt_kf = pose_graph.optimize_pose_graph(D.odom_poses, D.keyframes,
                                                D.loops, loop_nms_width=D.nms)
        D.opt_rel_pairwise = evaluate.align_to_first_pose(
            pose_graph.propagate_to_full_trajectory(D.odom_poses,
                                                    D.keyframes, opt_kf))
        m_pgo = evaluate.compute_ape_rpe(D.gt_rel, D.opt_rel_pairwise)["ape_rmse"]
        results[name] = {"odometry_ape": D.ape_odom, "pairwise_pgo_ape": m_pgo,
                         "n_loops": len(D.loops),
                         "n_clusters": len(D.clusters), "seeds": {}}
        logger.info(f"{name}: odometry {D.ape_odom:.3f} m | pairwise-PGO "
                    f"{m_pgo:.3f} m")

    # ------------------------------------------------ submaps (seed-indep.)
    need_submaps = any(a in arm_list for a in ("submap", "submap_clique"))
    submaps = {name: {} for name in names}
    if need_submaps:
        for name, D in datasets.items():
            big = [c for c in D.clusters
                   if len(c["members"]) >= args.min_cluster]
            members = sorted({s for c in big for s in c["members"]})
            rng = np.random.default_rng(0)
            t0 = time.time()
            for m in members:
                submaps[name][m] = D.build_submap(
                    m, args.submap_half_width, args.submap_voxel,
                    args.submap_max_points, rng)
            D.drop_scan_cache()
            logger.info(f"{name}: built {len(members)} submaps "
                        f"(w=±{args.submap_half_width}, "
                        f"pts med={int(np.median([len(submaps[name][m]) for m in members])) if members else 0}) "
                        f"in {time.time() - t0:.1f} s")

    # ------------------------------------------------ arms x seeds
    rap = None
    for seed in seeds:
        logger.info("=" * 70)
        logger.info(f"seed {seed}")
        for name, D in datasets.items():
            big = [c for c in D.clusters
                   if len(c["members"]) >= args.min_cluster]
            seed_res = {}
            # joint registrations for this dataset/seed, per part-type
            T_parts_cache = {}  # (arm_parttype, cluster_id) -> (idx, T_parts)
            for arm in arm_list:
                loops_new = [dict(lc) for lc in D.loops]
                edge_stats = {}
                n_rep = 0
                n_clique = 0
                n_regate = 0
                for c in big:
                    members, deg = c["members"], c["deg"]
                    batches = compose_batches(members, deg, D.odom_poses,
                                              args.max_parts, args.compose,
                                              args.chain_overlap)
                    ptype = ("raw" if arm == "nway_raw" else "submap")
                    # ---- register every batch (cached per batch) ----------
                    # Single-batch clusters keep the legacy cache filename
                    # (no _b suffix) so --reuse-cache still finds them.
                    batch_data = []  # (idx, T_parts, layout_rmse) per batch
                    for bi, batch in enumerate(batches):
                        key = (ptype, c["id"], bi)
                        if key not in T_parts_cache:
                            parts = ([D.load_scan(s) for s in batch]
                                     if ptype == "raw"
                                     else [submaps[name][s] for s in batch])
                            pps = min(4000, args.points_budget // len(batch))
                            suffix = "" if len(batches) == 1 else f"_b{bi}"
                            cache = os.path.join(
                                args.output,
                                f"{name}_{ptype}_c{c['id']}{suffix}"
                                f"_seed{seed}.npz")
                            if args.reuse_cache and os.path.exists(cache):
                                z = np.load(cache)
                                T_parts = [z["T_parts"][k]
                                           for k in range(len(z["T_parts"]))]
                                assert [int(k) for k in z["members"]] == batch
                                t_infer = -1.0
                            else:
                                if rap is None or rap.seed != seed:
                                    from rap_slam.rap_backend import RAPBackend
                                    rap = RAPBackend(seed=seed)
                                T_parts, diag = rap.register_multi(
                                    parts, max_points_per_scan=pps)
                                t_infer = diag["t_inference"]
                                np.savez(cache, T_parts=np.stack(T_parts),
                                         members=np.array(batch))
                            idx = {s: k for k, s in enumerate(batch)}

                            # oracle layout warp check
                            pos = np.array([T_parts[idx[s]][:3, 3]
                                            for s in batch])
                            gtpos = np.array([D.gt[s][:3, 3] for s in batch])
                            R_f, t_f = umeyama_se3(pos, gtpos)
                            resid = np.linalg.norm(
                                (R_f @ pos.T).T + t_f - gtpos, axis=1)
                            layout_rmse = float(np.sqrt(np.mean(resid ** 2)))
                            logger.info(
                                f"  {name} {arm} cluster {c['id']}{suffix or ' (single batch)'}: "
                                f"{len(members)}->{len(batch)} parts"
                                f"{f'/{len(batches)} batches' if suffix else ''}, "
                                f"{pps} pts/part, infer {t_infer:.1f} s, "
                                f"layout rmse {layout_rmse:.2f} m")
                            T_parts_cache[key] = (idx, T_parts, layout_rmse)
                        batch_data.append(T_parts_cache[key])

                    # replace accepted edges with cluster-consistent
                    # relatives — but RE-GATED intrinsically (Direction A
                    # hazard found on kitti06_live 2026-08-09: a degree-
                    # trimmed 97-member street-CHAIN cluster warps the joint
                    # registration (layout rmse 67-69 m), and its relatives
                    # are self-consistent garbage; ungated clique edges blew
                    # PGO to 135.7 m). The re-gate is a CATASTROPHE detector,
                    # NOT the strict 0.7/2.5 retrieval gate — submap parts
                    # have systematically lower overlap than raw scans, and
                    # the strict gate misfired on marginal-but-real edges
                    # (POSS submap 1.705 -> 1.838). Empirical basis
                    # (scripts/eval_submap_regate.py, 48 edges, seed 42):
                    # good edges (GT err < 1.25 m): ov >= 0.588, md <= 1.75;
                    # warped edges (GT err 184-280 m): ov <= 0.486,
                    # md >= 2.14 -> gate ov >= 0.5 AND md <= 2.0 sits in
                    # both gaps. Warps fail BOTH criteria on 5/7 edges.
                    from rap_slam.rap_backend import RAPBackend
                    batch_members = sorted({s for b in batches for s in b})
                    ds_parts = {s: (prep_for_metrics(D.load_scan(s))
                                    if ptype == "raw"
                                    else prep_for_metrics(submaps[name][s]))
                                for s in batch_members}
                    for k in c["edges"]:
                        i, j = D.loops[k]["i"], D.loops[k]["j"]
                        # first batch containing BOTH endpoints (chain
                        # windows guarantee coverage for chain-adjacent
                        # loop endpoints)
                        hit = next(((idx, T_parts)
                                    for idx, T_parts, _ in batch_data
                                    if i in idx and j in idx), None)
                        if hit is None:
                            continue
                        idx, T_parts = hit
                        T_new = (np.linalg.inv(T_parts[idx[i]])
                                 @ T_parts[idx[j]])
                        ov, md = RAPBackend._alignment_metrics(
                            ds_parts[i], ds_parts[j], T_new, thresh=1.0)
                        if ov < 0.5 or md > 2.0:
                            n_regate += 1
                            continue
                        loops_new[k]["T_rel"] = T_new
                        n_rep += 1
                        te_n, re_n = rel_err(loops_new[k]["T_rel"],
                                             D.gt[i], D.gt[j])
                        te_p, re_p = rel_err(D.loops[k]["T_rel"],
                                             D.gt[i], D.gt[j])
                        edge_stats[k] = {"pair": (i, j),
                                         "cluster": c["id"],
                                         "pw_te": te_p, "nw_te": te_n,
                                         "pw_re": re_p, "nw_re": re_n}

                    # clique arm: add ALL intra-cluster member pairs through
                    # the SAME acceptance gate (ungated clique edges from a
                    # warped registration blew kitti06_live PGO to 135.7 m).
                    # Only pairs separated like loops — near-temporal members
                    # are already well constrained by odometry, and RC-5
                    # showed RAP relatives are noisier than KISS-ICP there.
                    # With chain windows, only INTRA-batch pairs are added
                    # (the batch is the jointly-registered unit); pairs in
                    # the window overlap appear in two batches — dedupe via
                    # `have`.
                    if arm == "submap_clique":
                        have = {tuple(sorted((e["i"], e["j"])))
                                for e in loops_new}
                        for idx, T_parts, _ in batch_data:
                            trimmed = sorted(idx.keys())
                            for ai in range(len(trimmed)):
                                for bi in range(ai + 1, len(trimmed)):
                                    i, j = trimmed[ai], trimmed[bi]
                                    if j - i < 200:
                                        continue
                                    if (i, j) in have:
                                        continue
                                    T_rel = (np.linalg.inv(T_parts[idx[i]])
                                             @ T_parts[idx[j]])
                                    ov, md = RAPBackend._alignment_metrics(
                                        ds_parts[i], ds_parts[j], T_rel,
                                        thresh=1.0)
                                    if ov < 0.5 or md > 2.0:
                                        n_regate += 1
                                        continue
                                    te, re = rel_err(T_rel, D.gt[i], D.gt[j])
                                    loops_new.append({
                                        "i": i, "j": j, "T_rel": T_rel,
                                        "overlap": ov, "mean_dist": md,
                                        "gt_trans_err": te, "gt_rot_err_deg": re,
                                        "source": "submap_clique"})
                                    have.add((i, j))
                                    n_clique += 1
                opt_kf = pose_graph.optimize_pose_graph(
                    D.odom_poses, D.keyframes, loops_new,
                    loop_nms_width=D.nms)
                m_arm = full_traj_ape(D.odom_poses, D.keyframes, opt_kf,
                                      D.gt_rel)
                te_pw = [v["pw_te"] for v in edge_stats.values()]
                te_nw = [v["nw_te"] for v in edge_stats.values()]
                logger.info(
                    f"  {name} {arm}: APE {m_arm:.3f} m "
                    f"(baseline {results[name]['pairwise_pgo_ape']:.3f} m) | "
                    f"edges replaced {len(edge_stats)}"
                    f" re-gated {n_regate}"
                    + (f" +{n_clique} clique" if arm == "submap_clique" else "")
                    + (f" | intrinsic med {np.median(te_pw):.2f} -> "
                       f"{np.median(te_nw):.2f} m" if te_pw else ""))
                seed_res[arm] = {
                    "ape": m_arm,
                    "n_edges_replaced": len(edge_stats),
                    "n_regated": n_regate,
                    "n_clique_added": n_clique if arm == "submap_clique" else 0,
                    "edge_pw_te_med": float(np.median(te_pw)) if te_pw else None,
                    "edge_nw_te_med": float(np.median(te_nw)) if te_nw else None,
                }

                # ---------------- map-consistency metrics (selected seeds)
                if seed in map_seeds and not args.skip_maps:
                    opt_rel = evaluate.align_to_first_pose(
                        pose_graph.propagate_to_full_trajectory(
                            D.odom_poses, D.keyframes, opt_kf))
                    t0 = time.time()
                    scan_ids = list(range(0, len(D.odom_poses),
                                          D.cfg["map_stride"]))
                    map_est = build_map_points(D, scan_ids, opt_rel,
                                               args.map_voxel, args.map_cap)
                    if not hasattr(D, "_maps"):
                        D._maps = {}
                        D._maps["gt"] = build_map_points(
                            D, scan_ids, D.gt_rel, args.map_voxel,
                            args.map_cap)
                    cham, rec = chamfer_to_gt(map_est, D._maps["gt"])
                    mme = mean_map_entropy(map_est)
                    ov_med, md_med = loop_site_alignment(D, D.loops, opt_rel)
                    seed_res[arm].update({
                        "map_chamfer": cham, "map_recall_0.5m": rec,
                        "map_mme": mme, "loopsite_overlap_med": ov_med,
                        "loopsite_mean_dist_med": md_med,
                    })
                    logger.info(
                        f"    map: chamfer {cham:.3f} m recall@0.5m {rec:.3f} "
                        f"MME {mme:.3f} | loop-site overlap {ov_med:.3f} "
                        f"mean_dist {md_med:.3f} m "
                        f"({time.time() - t0:.0f} s)")
            # baseline map metrics once per dataset (odom + pairwise)
            if seed in map_seeds and not args.skip_maps and "maps" not in results[name]:
                maps_m = {}
                for label, poses_src in (("odometry", D.odom_rel),
                                         ("pairwise", D.opt_rel_pairwise)):
                    scan_ids = list(range(0, len(D.odom_poses),
                                          D.cfg["map_stride"]))
                    mp = build_map_points(D, scan_ids, poses_src,
                                          args.map_voxel, args.map_cap)
                    cham, rec = chamfer_to_gt(mp, D._maps["gt"])
                    mme = mean_map_entropy(mp)
                    ov_med, md_med = loop_site_alignment(D, D.loops, poses_src)
                    maps_m[label] = {"map_chamfer": cham,
                                     "map_recall_0.5m": rec, "map_mme": mme,
                                     "loopsite_overlap_med": ov_med,
                                     "loopsite_mean_dist_med": md_med}
                    logger.info(
                        f"    map [{label}]: chamfer {cham:.3f} m "
                        f"recall@0.5m {rec:.3f} MME {mme:.3f} | loop-site "
                        f"overlap {ov_med:.3f} mean_dist {md_med:.3f} m")
                results[name]["maps"] = maps_m
            results[name]["seeds"][str(seed)] = seed_res

    # ------------------------------------------------ summary
    print("\n===== SUMMARY (APE rmse, m) =====")
    for name in names:
        r = results[name]
        print(f"\n[{name}] odometry {r['odometry_ape']:.3f} | pairwise-PGO "
              f"{r['pairwise_pgo_ape']:.3f}")
        for arm in arm_list:
            apes = [r["seeds"][s][arm]["ape"] for s in r["seeds"]]
            print(f"  {arm:14s}: {['%.3f' % a for a in apes]} "
                  f"(med {float(np.median(apes)):.3f})")
        if "maps" in r:
            for label, mm in r["maps"].items():
                print(f"  map[{label}]: chamfer {mm['map_chamfer']:.3f} "
                      f"MME {mm['map_mme']:.3f}")
        for s in r["seeds"]:
            for arm in arm_list:
                mm = r["seeds"][s][arm]
                if "map_chamfer" in mm:
                    print(f"  map[{arm} seed {s}]: chamfer "
                          f"{mm['map_chamfer']:.3f} MME {mm['map_mme']:.3f} "
                          f"loop-site ov {mm['loopsite_overlap_med']:.3f} "
                          f"md {mm['loopsite_mean_dist_med']:.3f}")
    results["config"] = {k: v for k, v in vars(args).items()}
    results["total_runtime_s"] = round(time.time() - t0_all, 1)
    out = os.path.join(args.output, "summary.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1, default=str)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
