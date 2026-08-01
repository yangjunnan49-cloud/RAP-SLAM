#!/usr/bin/env python
"""RC-3: hierarchical N-way backend — joint registration within loop clusters,
PGO stitching across clusters.

Motivation (from the B1/B2/B3 N-way comparison on POSS): small loop-batch
joint registration is near-perfect and its cycle-consistent constraints beat
independent pairwise constraints in PGO, but one-shot dense batches warp the
global layout. The hierarchical design registers each loop cluster (connected
component of the accepted loop-edge graph) separately — keeping batches in
the small, well-conditioned regime — and lets GTSAM PGO stitch the trajectory
across clusters.

Experiment on the saved live products (KITTI 02: 182 edges -> 12 clusters;
POSS: 2 edges -> 1 cluster):
  baseline : pairwise RAP constraints + PGO (current defaults relfloor+huber)
  hier     : for every cluster with >= --min-cluster members, replace each
             accepted edge's T_rel by the cluster-consistent relative
             inv(T_i) @ T_j from ONE register_multi call per cluster
  multi-seed: repeat with several RAP seeds to expose sampling variance

Also reports, per replaced edge, the intrinsic T_rel error vs GT (pairwise
vs N-way), and per cluster the oracle SE(3) layout fit to GT (warp check).

Run in the conda env:
  /root/miniconda3/envs/rap-inference/bin/python scripts/eval_hier_nway.py \
      --dataset kitti02 --seeds 42,43,44
"""

import argparse
import json
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from rap_slam import evaluate, pose_graph, real_dataset  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Hierarchical N-way backend eval")
    p.add_argument("--dataset", choices=["kitti02", "poss"], default="kitti02")
    p.add_argument("--kitti-root",
                   default="/mnt/e/githouse/Datasets/data_odometry_velodyne/dataset")
    p.add_argument("--poss-root", default="/mnt/e/githouse/RAP-SLAM/data/poss/dataset")
    p.add_argument("--min-cluster", type=int, default=3,
                   help="clusters smaller than this keep pairwise measurements")
    p.add_argument("--max-parts", type=int, default=16,
                   help="trim bigger clusters to the N highest-degree scans")
    p.add_argument("--points-budget", type=int, default=30000,
                   help="total points per cluster call; per-scan cap = "
                        "min(4000, budget // n_parts)")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--reuse-cache", action="store_true")
    p.add_argument("--output", default=os.path.join(REPO_ROOT, "output_hier_nway"))
    return p.parse_args()


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
    for cid, edge_ids in enumerate(sorted(comps.values(),
                                          key=lambda e: -len(e))):
        members = sorted({s for k in edge_ids
                          for s in (loops[k]["i"], loops[k]["j"])})
        deg = {}
        for k in edge_ids:
            for s in (loops[k]["i"], loops[k]["j"]):
                deg[s] = deg.get(s, 0) + 1
        clusters.append({"id": cid, "members": members, "edges": edge_ids,
                         "deg": deg})
    return clusters


def full_traj_ape(odom_poses, keyframes, optimized, gt_rel):
    opt = evaluate.align_to_first_pose(
        pose_graph.propagate_to_full_trajectory(odom_poses, keyframes,
                                                optimized))
    return evaluate.compute_ape_rpe(gt_rel, opt)["ape_rmse"]


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    t0 = time.time()

    # ------------------------------------------------ dataset products
    if args.dataset == "kitti02":
        save_dir = os.path.join(REPO_ROOT, "output_ros2", "step3_kitti02")
        odom_path = os.path.join(REPO_ROOT, "output_ros2",
                                 "live_poses_kitti02.txt")
        gt = list(real_dataset.load_gt_poses(
            os.path.join(args.kitti_root, "poses/02.txt"),
            os.path.join(args.kitti_root, "sequences/02/calib.txt"),
            start=0, end=4660, apply_calib=True))
        seq_dir = os.path.join(args.kitti_root, "sequences/02")
        file_offset = 0
    else:
        save_dir = os.path.join(REPO_ROOT, "output_ros2", "step3")
        odom_path = os.path.join(REPO_ROOT, "output_ros2",
                                 "live_poses_step3.txt")
        gt = list(real_dataset.load_gt_poses(
            os.path.join(args.poss_root, "poses", "00.txt"),
            os.path.join(args.poss_root, "sequences", "00", "calib.txt"),
            start=3850, end=5268, apply_calib=False))
        seq_dir = os.path.join(args.poss_root, "sequences", "00")
        file_offset = 3850  # live products use segment-relative indices

    odom_poses = load_tum(odom_path)
    keyframes = json.load(open(os.path.join(save_dir, "keyframes.json")))
    loops = json.load(open(os.path.join(save_dir, "accepted_loops.json")))
    for lc in loops:
        lc["T_rel"] = np.array(lc["T_rel"])
    print(f"{args.dataset}: frames={len(odom_poses)} keyframes={len(keyframes)} "
          f"loop edges={len(loops)}")

    gt_rel = evaluate.align_to_first_pose(gt)
    odom_rel = evaluate.align_to_first_pose(odom_poses)
    m_odom = evaluate.compute_ape_rpe(gt_rel, odom_rel)["ape_rmse"]

    # ------------------------------------------------ baseline pairwise PGO
    opt_kf = pose_graph.optimize_pose_graph(odom_poses, keyframes, loops)
    m_pgo = full_traj_ape(odom_poses, keyframes, opt_kf, gt_rel)
    print(f"odometry APE {m_odom:.3f} m | pairwise-PGO APE {m_pgo:.3f} m "
          f"(defaults relfloor+huber)")

    # ------------------------------------------------ clusters
    clusters = mine_clusters(loops)
    big = [c for c in clusters if len(c["members"]) >= args.min_cluster]
    print(f"clusters: {len(clusters)} total, {len(big)} with "
          f">= {args.min_cluster} members")
    for c in clusters:
        print(f"  cluster {c['id']}: {len(c['members'])} members, "
              f"{len(c['edges'])} edges, span "
              f"[{c['members'][0]}..{c['members'][-1]}]")

    needed = sorted({s for c in big for s in c["members"]})
    scans = {}
    for s in needed:
        scans[s] = np.fromfile(
            os.path.join(seq_dir, "velodyne", f"{s + file_offset:06d}.bin"),
            dtype=np.float32).reshape(-1, 4)[:, :3]
    print(f"loaded {len(scans)} cluster scans")

    seeds = [int(s) for s in args.seeds.split(",")]
    results = {"dataset": args.dataset, "odometry_ape": m_odom,
               "pairwise_pgo_ape": m_pgo, "seeds": {}}

    rap = None
    for seed in seeds:
        print(f"\n================ seed {seed} ================")
        hier_loops = [dict(lc) for lc in loops]
        edge_stats = {}
        for c in big:
            members, deg = c["members"], c["deg"]
            trimmed = members
            if len(members) > args.max_parts:
                trimmed = sorted(sorted(members, key=lambda s: -deg[s])
                                 [:args.max_parts])
            pps = min(4000, args.points_budget // len(trimmed))
            cache = os.path.join(
                args.output, f"{args.dataset}_c{c['id']}_seed{seed}.npz")
            if args.reuse_cache and os.path.exists(cache):
                z = np.load(cache)
                T_parts = [z["T_parts"][k] for k in range(len(z["T_parts"]))]
                assert [int(k) for k in z["members"]] == trimmed
                t_infer = -1.0
            else:
                if rap is None or rap.seed != seed:
                    from rap_slam.rap_backend import RAPBackend
                    rap = RAPBackend(seed=seed)
                T_parts, diag = rap.register_multi(
                    [scans[s] for s in trimmed], max_points_per_scan=pps)
                t_infer = diag["t_inference"]
                np.savez(cache, T_parts=np.stack(T_parts),
                         members=np.array(trimmed))
            idx = {s: k for k, s in enumerate(trimmed)}

            # oracle layout warp check (needs >= 3 points)
            pos = np.array([T_parts[idx[s]][:3, 3] for s in trimmed])
            gtpos = np.array([gt[s][:3, 3] for s in trimmed])
            R_f, t_f = umeyama_se3(pos, gtpos)
            resid = np.linalg.norm((R_f @ pos.T).T + t_f - gtpos, axis=1)

            n_rep = 0
            for k in c["edges"]:
                i, j = loops[k]["i"], loops[k]["j"]
                if i in idx and j in idx:
                    hier_loops[k]["T_rel"] = (np.linalg.inv(T_parts[idx[i]])
                                              @ T_parts[idx[j]])
                    n_rep += 1
                    te_n, re_n = rel_err(hier_loops[k]["T_rel"], gt[i], gt[j])
                    te_p, re_p = rel_err(loops[k]["T_rel"], gt[i], gt[j])
                    edge_stats[k] = {"pair": (i, j), "cluster": c["id"],
                                     "pw_te": te_p, "nw_te": te_n,
                                     "pw_re": re_p, "nw_re": re_n}
            print(f"  cluster {c['id']}: {len(members)}->{len(trimmed)} parts, "
                  f"{pps} pts/scan, infer {t_infer:.1f} s, layout rmse "
                  f"{float(np.sqrt(np.mean(resid**2))):.2f} m, "
                  f"{n_rep}/{len(c['edges'])} edges replaced")

        opt_kf_h = pose_graph.optimize_pose_graph(odom_poses, keyframes,
                                                  hier_loops)
        m_h = full_traj_ape(odom_poses, keyframes, opt_kf_h, gt_rel)
        te_pw = [v["pw_te"] for v in edge_stats.values()]
        te_nw = [v["nw_te"] for v in edge_stats.values()]
        re_pw = [v["pw_re"] for v in edge_stats.values()]
        re_nw = [v["nw_re"] for v in edge_stats.values()]
        better = sum(1 for v in edge_stats.values() if v["nw_te"] < v["pw_te"])
        print(f"  edges replaced: {len(edge_stats)} | intrinsic T err vs GT: "
              f"pairwise med {np.median(te_pw):.2f} m -> N-way med "
              f"{np.median(te_nw):.2f} m (rot {np.median(re_pw):.1f} -> "
              f"{np.median(re_nw):.1f} deg), N-way better on "
              f"{better}/{len(edge_stats)}")
        print(f"  >>> hier-PGO APE {m_h:.3f} m  (baseline {m_pgo:.3f} m)")
        results["seeds"][str(seed)] = {
            "hier_pgo_ape": m_h,
            "n_edges_replaced": len(edge_stats),
            "edge_pw_te_med": float(np.median(te_pw)) if te_pw else None,
            "edge_nw_te_med": float(np.median(te_nw)) if te_nw else None,
            "edges": {str(k): {kk: (round(vv, 3) if isinstance(vv, float)
                                    else vv)
                               for kk, vv in v.items()}
                      for k, v in edge_stats.items()}}

    apes = [v["hier_pgo_ape"] for v in results["seeds"].values()]
    print("\n===== SUMMARY =====")
    print(f"odometry {m_odom:.3f} | pairwise-PGO {m_pgo:.3f} | "
          f"hier-PGO per seed {['%.3f' % a for a in apes]} "
          f"(med {float(np.median(apes)):.3f}, spread "
          f"{max(apes) - min(apes):.3f})")
    results["total_runtime_s"] = round(time.time() - t0, 1)
    out = os.path.join(args.output, f"summary_{args.dataset}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1, default=str)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
