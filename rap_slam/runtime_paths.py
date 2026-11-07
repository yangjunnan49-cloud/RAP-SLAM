"""Dependency-free path defaults and preflight for current runtime entries."""
import os
from pathlib import Path

SLAM_ROOT = Path(__file__).resolve().parents[1]


def default_paths():
    workspace = Path(os.environ.get("RAP_WORKSPACE", str(SLAM_ROOT.parent))).expanduser().resolve()
    rap = Path(os.environ.get("RAP_ROOT", str(workspace / "RAP-main"))).expanduser().resolve()
    weights = Path(os.environ.get("RAP_WEIGHTS_ROOT", str(rap / "weights"))).expanduser().resolve()
    datasets = Path(os.environ.get("RAP_DATASETS_ROOT", str(workspace / "Datasets"))).expanduser().resolve()
    output = Path(os.environ.get("RAP_OUTPUT_ROOT", str(workspace / "RAP-runs"))).expanduser().resolve()
    return {"rap_root": str(rap), "map": str(rap / "demo_example_data/map_kitti/kitti00.ply"),
            "output": str(output / "slam"), "real_root": str(SLAM_ROOT / "data/poss"),
            "spires_root": str(datasets / "oxford_spires"),
            "rap_ckpt": os.environ.get("RAP_BACKEND_CHECKPOINT", str(weights / "rap_model_12.ckpt")),
            "spinnet_ckpt": os.environ.get("RAP_SPINNET_CHECKPOINT", str(weights / "mini_spinnet_t.pth"))}


def backend_paths(rap_root=None, ckpt_path=None, spinnet_ckpt=None, feature_extraction_on=True):
    defaults = default_paths()
    rap = Path(rap_root or defaults["rap_root"]).expanduser().resolve()
    # Explicit repository roots select their own weights unless separately configured.
    weights = Path(os.environ.get("RAP_WEIGHTS_ROOT", str(rap / "weights"))).expanduser().resolve()
    ckpt = Path(ckpt_path or os.environ.get("RAP_BACKEND_CHECKPOINT", str(weights / "rap_model_12.ckpt"))).expanduser().resolve()
    spinnet = Path(spinnet_ckpt or os.environ.get("RAP_SPINNET_CHECKPOINT", str(weights / "mini_spinnet_t.pth"))).expanduser().resolve()
    required = [(rap / "config/RAP_inference.yaml", "RAP inference config"), (ckpt, "RAP checkpoint")]
    if feature_extraction_on:
        required.append((spinnet, "miniSpinNet checkpoint"))
    for path, role in required:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {role}: {path}")
    if not (rap / "rectified_point_flow").is_dir():
        raise FileNotFoundError(f"Missing RAP model package: {rap / 'rectified_point_flow'}")
    return str(rap), str(ckpt), str(spinnet)


def validate_pipeline_paths(args):
    """Check selected paths, not dataset completeness or scientific admissibility."""
    errors = []

    def need(path, directory, role):
        path = Path(path).expanduser()
        if not (path.is_dir() if directory else path.is_file()):
            errors.append(f"Missing {role}: {path.resolve()}")

    output = Path(args.output).expanduser()
    if output.exists() or output.is_symlink():
        errors.append(f"Output already exists; choose a new output: {output.resolve()}")
    if args.source == "sim":
        need(args.map, False, "simulation map")
    elif args.source == "real":
        root = Path(args.real_root) / "dataset"
        seq = root / "sequences" / args.real_seq
        need(root / "poses" / (args.real_seq + ".txt"), False, "reference poses")
        need(seq / "velodyne", True, "LiDAR scans")
        if args.real_apply_calib:
            need(seq / "calib.txt", False, "LiDAR/camera calibration")
    else:
        root = Path(args.spires_root)
        need(root / "sequences" / args.spires_seq / "raw/lidar-clouds", True, "Spires scans")
        need(root / "gt" / (args.spires_seq + ".txt"), False, "Spires reference poses")
    if not args.skip_loop_closure:
        try:
            backend_paths(args.rap_root, args.rap_ckpt, args.spinnet_ckpt, not args.no_rap_features)
        except FileNotFoundError as exc:
            errors.append(str(exc))
        if args.gate_mode == "conformal":
            need(args.gate_model or SLAM_ROOT / "data/conformal_gate/A1.json", False, "gate model")
    return errors
