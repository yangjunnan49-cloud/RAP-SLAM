"""Offline pipeline arguments; numerical defaults are preserved from A3 before."""
import argparse
from .runtime_paths import default_paths


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="RAP-SLAM offline SLAM pipeline")
    paths = default_paths()
    p.add_argument("--map", default=paths["map"])
    p.add_argument("--rap-root", default=paths["rap_root"])
    p.add_argument("--output", default=paths["output"])
    p.add_argument("--n-poses", type=int, default=100)
    p.add_argument("--radius", type=float, default=80.0)
    p.add_argument("--center", type=float, nargs=2, default=[250.0, -30.0])
    p.add_argument("--sensor-range", type=float, default=50.0)
    p.add_argument("--points-range", type=int, nargs=2, default=[2000, 4000])
    p.add_argument("--noise-sigma", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--kf-trans", type=float, default=6.0, help="keyframe translation threshold (m)")
    p.add_argument("--kf-rot", type=float, default=12.0, help="keyframe rotation threshold (deg)")
    p.add_argument("--loop-dist", type=float, default=15.0, help="loop candidate distance threshold (m)")
    p.add_argument("--loop-sep", type=int, default=5, help="loop candidate min scan separation")
    p.add_argument("--max-candidates", type=int, default=12)
    p.add_argument("--loop-method", choices=["proximity", "sc", "both"], default="both",
                   help="loop-candidate retrieval: odometry proximity, "
                        "Scan Context global place recognition, or both merged")
    p.add_argument("--sc-dist-thresh", type=float, default=0.25,
                   help="Scan Context distance acceptance threshold")
    p.add_argument("--sc-max-radius", type=float, default=80.0)
    p.add_argument("--sc-rings", type=int, default=20)
    p.add_argument("--sc-sectors", type=int, default=60)
    p.add_argument("--sc-shortlist", type=int, default=10,
                   help="ring-key shortlist size per query")
    p.add_argument("--rap-steps", type=int, default=5, help="RAP inference sampling steps")
    p.add_argument("--rap-max-points", type=int, default=4000)
    p.add_argument("--rap-voxel", type=float, default=0.4)
    p.add_argument("--trans-consistency", type=float, default=5.0)
    p.add_argument("--rot-consistency", type=float, default=25.0)
    p.add_argument("--min-overlap", type=float, default=0.7)
    p.add_argument("--max-mean-dist", type=float, default=2.5,
                   help="loop acceptance: max mean symmetric NN distance (m); 0 disables")
    # RC-3 (2026-08-08): optional multi-generation consistency gate
    # (eval_multigen_confidence.py). k=1 keeps the single-generation gate.
    p.add_argument("--multigen-k", type=int, default=1,
                   help="RAP generations per candidate; >1 switches the gate "
                        "to cross-generation consistency (overlap_min, disp_r)")
    p.add_argument("--min-overlap-min", type=float, default=0.4,
                   help="multigen gate: min overlap across generations")
    p.add_argument("--max-disp-r", type=float, default=30.0,
                   help="multigen gate: max inter-generation rotation "
                        "dispersion (deg); catches bistable yaw flips")
    # S1a arm (2026-09-05, output_s1/s1_report.md): conformal acceptance
    # gate with a finite-sample false-accept bound (alpha=0.05). Default
    # "heuristic" keeps the two-metric gate (canonical goldens unchanged).
    p.add_argument("--gate-mode", choices=["heuristic", "conformal"],
                   default="heuristic",
                   help="conformal = ConformalGate A1 (data/conformal_gate/"
                        "A1.json) decides acceptance from overlap+mean_dist")
    p.add_argument("--gate-model", default=None,
                   help="explicit ConformalGate JSON path (overrides the "
                        "default A1 model when --gate-mode conformal)")
    # RAP-CORE PGO (2026-08-01): floor-relative adaptive sigmas from RAP
    # mean_dist + Huber; best cross-dataset combo (eval_pgo_robust.py:
    # KITTI-02 27.387 -> 26.423 m, POSS bit-identical 1.757 m).
    p.add_argument("--loop-weighting", choices=["fixed", "adaptive", "relfloor"],
                   default="relfloor",
                   help="fixed: uniform loop sigmas; adaptive: absolute sigmas "
                        "from mean_dist; relfloor: loosen only edges sloppier "
                        "than --relfloor-ref (cross-dataset safe)")
    p.add_argument("--relfloor-ref", type=float, default=1.0,
                   help="reference RAP mean_dist (m) for relfloor weighting")
    p.add_argument("--relfloor-cap", type=float, default=3.0,
                   help="max sigma loosening factor for relfloor weighting")
    p.add_argument("--loop-nms-width", type=float, default=0.0,
                   help="(i,j)-space NMS width (scans) on accepted loop edges "
                        "before PGO; 0 = off. Live default is 10 (dense "
                        "same-site clusters); offline is budget-capped, so "
                        "off keeps the golden regressions bit-exact")
    p.add_argument("--robust-kernel", choices=["none", "huber", "dcs"],
                   default="huber", help="robust mEstimator on loop edges")
    p.add_argument("--huber-k", type=float, default=1.0)
    p.add_argument("--dcs-phi", type=float, default=10.0)
    p.add_argument("--no-rap-features", action="store_true",
                   help="disable miniSpinNet features (zero features)")
    p.add_argument("--force-regenerate", action="store_true")
    p.add_argument("--skip-loop-closure", action="store_true")
    # --- real sequence (KITTI-format: KITTI odometry / POSS) ---
    p.add_argument("--source", choices=["sim", "real", "spires"], default="sim",
                   help="sim = simulated sequence from an aggregated map; "
                        "real = KITTI-format dataset segment; "
                        "spires = Oxford Spires (Hesai QT64) sequence")
    p.add_argument("--real-root", default=paths["real_root"],
                   help="dataset root with dataset/poses and dataset/sequences")
    p.add_argument("--real-seq", default="00")
    p.add_argument("--real-start", type=int, default=3850)
    p.add_argument("--real-end", type=int, default=5268)
    p.add_argument("--real-stride", type=int, default=1)
    p.add_argument("--real-apply-calib", action="store_true",
                   help="apply Tr velo<->cam conversion (KITTI convention; "
                        "POSS poses are already in the LiDAR/world frame)")
    # [meeting-20260921] default flipped to True on the meeting branch: the
    # 0.205-deg correction matches the CT-ICP/MULLS/Traj-LO/KISS KITTI
    # convention and is our adopted reporting arm (S0b). main keeps default False.
    p.add_argument("--apply-kitti-205", dest="apply_kitti_205",
                   action="store_true", default=True,
                   help="apply the 0.205-deg HDL-64 intrinsic vertical-angle "
                        "correction (CT-ICP/MULLS/Traj-LO KITTI convention); "
                        "only legal together with --real-apply-calib")
    p.add_argument("--no-apply-kitti-205", dest="apply_kitti_205",  # [meeting-20260921]
                   action="store_false",
                   help="disable the 0.205-deg correction (pre-meeting default)")
    # --- Oxford Spires (P4 cross-sensor zero-shot) ---
    p.add_argument("--spires-root",
                   default=paths["spires_root"],
                   help="root with sequences/<seq>/raw/lidar-clouds/*.pcd "
                        "and gt/<seq>.txt")
    p.add_argument("--spires-seq", default="2024-03-12-keble-college-02")
    p.add_argument("--spires-start", type=int, default=0)
    p.add_argument("--spires-end", type=int, default=None)
    p.add_argument("--allow-odom-fallback", action="store_true",
                   help="Explicitly permit Open3D if KISS-ICP initialization fails")
    # [meeting-20260921] front-end range/voxel defaults aligned to the official
    # KISS-SLAM/KISS-ICP KITTI config (max 100 m, min 0 m, mapping voxel 1.0 m);
    # main keeps the POSS-tuned 80/1.5/0.8 set.
    p.add_argument("--odom-max-range", type=float, default=100.0)  # [meeting-20260921] was 80.0
    p.add_argument("--odom-min-range", type=float, default=0.0)  # [meeting-20260921] new
    p.add_argument("--odom-voxel", type=float, default=1.0)  # [meeting-20260921] new
    p.add_argument("--deskew", action="store_true",
                   help="synthesize per-point azimuth timestamps and enable "
                        "KISS-ICP motion compensation (real spinning-LiDAR "
                        "scans only; simulated scans have no distortion)")
    p.add_argument("--rap-ckpt", default=None, help="RAP checkpoint (default: weights root/rap_model_12.ckpt)")
    p.add_argument("--spinnet-ckpt", default=None, help="miniSpinNet checkpoint (default: weights root/mini_spinnet_t.pth)")
    return p.parse_args(argv)

