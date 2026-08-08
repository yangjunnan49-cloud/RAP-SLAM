"""In-process wrapper around the RAP (Rectified Point Flow) registration model.

The model is loaded ONCE and reused for every pairwise registration call.

Replicated from RAP's demo.py / sample.py inference path (RAP repo is imported
read-only via sys.path):
  1. Per-scan preprocessing: voxel downsample, statistical outlier removal,
     random cap, miniSpinNet 32-D feature extraction (as in demo.py).
  2. Sample normalization identical to PointCloudDataset._transform() in
     val/test mode (no augmentation): primary-part centering, uniform scaling,
     per-part centering, anchor-part offset.
  3. RectifiedPointFlow.sample_rectified_flow() generates the registered cloud;
     fit_transformations() fits per-part rigid transforms (scaled space).
  4. The per-part transforms are mapped back to the original scan frames, from
     which the relative transform between the two scans follows exactly.

Convention: register(points_a, points_b) returns T_rel = T_a^{-1} @ T_b, i.e.
the pose of scan b expressed in scan a's frame (maps points from frame b into
frame a) — the same convention as GTSAM BetweenFactor measurements.
"""

import logging
import os
import sys
import time
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_RAP_ENV_READY = False


def _setup_rap_imports(rap_root: str):
    """Add the RAP repo to sys.path (read-only usage)."""
    global _RAP_ENV_READY
    if _RAP_ENV_READY:
        return
    if rap_root not in sys.path:
        sys.path.insert(0, rap_root)
    dsp = os.path.join(rap_root, "dataset_process")
    if dsp not in sys.path:
        sys.path.insert(0, dsp)
    _RAP_ENV_READY = True


class RAPBackend:
    """Pairwise point-cloud registration with the RAP flow-matching model."""

    def __init__(
        self,
        rap_root: str = "/mnt/e/githouse/RAP-main",
        ckpt_path: Optional[str] = None,
        spinnet_ckpt: Optional[str] = None,
        sampling_steps: int = 5,
        voxel_size: float = 0.4,
        max_points_per_scan: int = 4000,
        des_r: float = 5.0,
        remove_outliers: bool = True,
        feature_extraction_on: bool = True,
        seed: int = 42,
    ):
        _setup_rap_imports(rap_root)
        self.rap_root = rap_root
        self.ckpt_path = ckpt_path or os.path.join(rap_root, "weights", "rap_model_12.ckpt")
        self.spinnet_ckpt = spinnet_ckpt or os.path.join(rap_root, "weights", "mini_spinnet_t.pth")
        self.sampling_steps = int(sampling_steps)
        self.voxel_size = float(voxel_size)
        self.max_points_per_scan = int(max_points_per_scan)
        self.des_r = float(des_r)
        self.remove_outliers = bool(remove_outliers)
        self.feature_extraction_on = bool(feature_extraction_on)
        self.seed = int(seed)

        import torch
        import hydra
        from rectified_point_flow.utils import load_checkpoint_for_module

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        t0 = time.time()
        overrides = [
            "model=rap_12",
            f"model.inference_sampling_steps={self.sampling_steps}",
            "model.rigidity_forcing=true",
            "model.n_generations=1",
            "model.save_results=false",
        ]
        with hydra.initialize_config_dir(config_dir=os.path.join(rap_root, "config"),
                                         version_base="1.3"):
            cfg = hydra.compose(config_name="RAP_inference", overrides=overrides)
        self.model = hydra.utils.instantiate(cfg.model)
        load_checkpoint_for_module(self.model, self.ckpt_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"RAP model loaded from {self.ckpt_path} "
                    f"({time.time() - t0:.1f} s, device={self.device})")

        self.feature_extractor = None
        if self.feature_extraction_on:
            from dataset_process.extract_sample_features import FeatureExtractor
            model_config = {
                "num_points_per_patch": 512,
                "is_aligned_to_global_z": True,
                "des_r": self.des_r,
            }
            self.feature_extractor = FeatureExtractor(
                model_config=model_config,
                des_r=self.des_r,
                is_aligned_to_global_z=True,
                checkpoint_path=self.spinnet_ckpt,
                device="auto",
            )
            logger.info(f"miniSpinNet feature extractor loaded (des_r={self.des_r} m)")

    # ------------------------------------------------------------------
    # Preprocessing (mirrors demo.py: voxel downsample + outlier removal +
    # point cap + miniSpinNet features)
    # ------------------------------------------------------------------
    def preprocess(self, points: np.ndarray,
                   max_points: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Downsample one scan and extract per-point features.

        Args:
            points: (N, 3) scan in its own sensor frame (meters).
            max_points: optional override of self.max_points_per_scan
                (use fewer points per scan for large N-way batches).

        Returns:
            ds_points: (M, 3) float32 downsampled points (original frame).
            features: (M, 32) float32 miniSpinNet features (zeros if disabled).
        """
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        if self.voxel_size > 0:
            pcd = pcd.voxel_down_sample(self.voxel_size)
        if self.remove_outliers:
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.5)
        ds = np.asarray(pcd.points, dtype=np.float32)

        cap = self.max_points_per_scan if max_points is None else int(max_points)
        if len(ds) > cap:
            rng = np.random.default_rng(self.seed)
            sel = rng.choice(len(ds), cap, replace=False)
            ds = ds[sel]
        if len(ds) < 200:
            raise ValueError(f"Too few points after preprocessing: {len(ds)} (< 200)")

        if self.feature_extractor is not None:
            result = self.feature_extractor.extract_features(ds)
            features = np.asarray(result["features"], dtype=np.float32)
        else:
            features = np.zeros((len(ds), 32), dtype=np.float32)
        return ds, features

    # ------------------------------------------------------------------
    # Normalization identical to PointCloudDataset._transform (val mode).
    # Returns cond (model input), plus everything needed to undo the transform.
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_parts(parts_points):
        """Replicates RAP's dataset normalization for a 2-part sample.

        Returns:
            cond: (TP, 3) float32 model-input point cloud (scaled space).
            scale: float, uniform scale factor.
            offsets_c: list of (3,) arrays c_i with cond_i * scale = orig_i - c_i.
            points_per_part: (2,) int64.
            anchor_idx: index of the anchor (primary) part.
        """
        counts = np.array([len(p) for p in parts_points])
        offsets = np.concatenate([[0], np.cumsum(counts)])
        pts_gt = np.concatenate(parts_points).astype(np.float64)

        primary_idx = int(np.argmax(counts))
        p_st, p_ed = offsets[primary_idx], offsets[primary_idx + 1]
        primary_trans = pts_gt[p_st:p_ed].mean(axis=0)

        # Global normalization (rot_global = I in val mode).
        pts_gt = pts_gt - primary_trans
        scale = float(np.max(np.abs(pts_gt[p_st:p_ed]))) * 1.5
        pts_gt /= scale
        gt_trans = pts_gt.mean(axis=0)
        pts_gt = pts_gt - gt_trans

        cond = np.empty_like(pts_gt)
        offsets_c = []
        for i in range(len(parts_points)):
            st, ed = offsets[i], offsets[i + 1]
            if i == primary_idx:
                # Anchor part: offset by gt_trans, trans_i = -gt_trans.
                cond[st:ed] = pts_gt[st:ed] + gt_trans
                c_i = primary_trans
            else:
                trans_i = pts_gt[st:ed].mean(axis=0)
                cond[st:ed] = pts_gt[st:ed] - trans_i
                c_i = primary_trans + gt_trans * scale + trans_i * scale
            offsets_c.append(c_i)

        return (cond.astype(np.float32), scale, offsets_c,
                counts.astype(np.int64), primary_idx)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register(self, points_a: np.ndarray, points_b: np.ndarray,
                 return_registered: bool = False):
        """Register two unposed scans.

        Args:
            points_a, points_b: (N, 3) scans, each in its own sensor frame.
            return_registered: also return the registered clouds (debug).

        Returns:
            T_rel: (4, 4) transform = T_a^{-1} @ T_b (pose of b in a's frame).
            diagnostics: dict with overlap ratio, point counts, timing, etc.
        """
        import torch
        from rectified_point_flow.procrustes import fit_transformations

        t_start = time.time()
        ds_a, feat_a = self.preprocess(points_a)
        ds_b, feat_b = self.preprocess(points_b)
        t_prep = time.time() - t_start

        parts_points = [ds_a, ds_b]
        cond, scale, offsets_c, counts, anchor_idx = self._normalize_parts(parts_points)
        features = np.concatenate([feat_a, feat_b]).astype(np.float32)
        n_a, n_b = int(counts[0]), int(counts[1])
        tp = n_a + n_b

        # Anchor indices: True for anchor-part points.
        anchor_indices = np.zeros(tp, dtype=bool)
        if anchor_idx == 0:
            anchor_indices[:n_a] = True
        else:
            anchor_indices[n_a:] = True

        batch = {
            "pointclouds": torch.from_numpy(cond).to(self.device),
            "pointclouds_gt": torch.from_numpy(cond).to(self.device),  # unused by sampler
            "features": torch.from_numpy(features).to(self.device),
            "scales": torch.tensor([scale], dtype=torch.float32, device=self.device),
            "anchor_indices": torch.from_numpy(anchor_indices).to(self.device),
            "points_per_part": torch.from_numpy(counts[None, :]).to(self.device),
            "cu_seqlens": torch.tensor([0, tp], dtype=torch.int64, device=self.device),
        }

        t_infer = time.time()
        with torch.inference_mode(), torch.autocast(device_type=self.device.type,
                                                    dtype=torch.float16,
                                                    enabled=self.device.type == "cuda"):
            result = self.model.sample_rectified_flow(
                batch, None, return_tarjectory=True)
            pred = result["end_point_trajectory"][-1]  # (TP, 3), scaled space
            cu_seqlens_batch = batch["cu_seqlens"].to(torch.int32)
            rotations, translations = fit_transformations(
                batch["pointclouds"], pred, batch["points_per_part"], cu_seqlens_batch)
        t_infer = time.time() - t_infer

        # Per-part transforms from original scan frame -> registered frame (meters).
        # cond_i * scale = orig_i - c_i  =>  p_reg = R_i (orig_i - c_i) + t_i * scale
        rots = rotations[0].float().cpu().numpy()      # (2, 3, 3)
        trans = translations[0].float().cpu().numpy()  # (2, 3)
        T_parts = []
        for i in range(2):
            T = np.eye(4)
            T[:3, :3] = rots[i]
            T[:3, 3] = trans[i] * scale - rots[i] @ offsets_c[i]
            T_parts.append(T)

        # Relative transform: pose of b expressed in a's frame.
        T_rel = np.linalg.inv(T_parts[0]) @ T_parts[1]

        # Alignment diagnostics on the downsampled clouds (in meters).
        overlap, mean_dist = self._alignment_metrics(ds_a, ds_b, T_rel,
                                                     thresh=1.0)

        diag = {
            "n_a": n_a, "n_b": n_b,
            "scale": scale,
            "anchor_idx": anchor_idx,
            "overlap_ratio": overlap,
            "mean_dist": mean_dist,
            "t_preprocess": t_prep,
            "t_inference": t_infer,
            "t_total": time.time() - t_start,
        }
        if return_registered:
            diag["T_parts"] = T_parts
            diag["ds_a"] = ds_a
            diag["ds_b"] = ds_b
        return T_rel, diag

    # ------------------------------------------------------------------
    # Multi-generation consistency verification (RC-3, 2026-08-08).
    # ------------------------------------------------------------------
    def register_consistent(self, points_a: np.ndarray, points_b: np.ndarray,
                            k: int = 3):
        """Register the same pair k times and measure cross-generation
        consistency. RAP sampling is stochastic; TRUE loops reproduce
        across generations (min overlap 0.66-0.98 on KITTI-02) while FALSE
        loops do not (min overlap 0.03-0.16, sometimes one lucky generation
        reaching 0.64, plus 90-deg bistable yaw flips). Returns the
        best-of-k result (max overlap, tie -> min mean_dist) with added
        diagnostics: overlap_min, overlap_all, mean_dist_all, disp_t (m),
        disp_r (deg). See scripts/eval_multigen_confidence.py."""
        Ts, ovs, mds = [], [], []
        t_infer = t_total = 0.0
        for _ in range(int(k)):
            T_rel, diag = self.register(points_a, points_b)
            Ts.append(T_rel)
            ovs.append(float(diag["overlap_ratio"]))
            mds.append(float(diag["mean_dist"]))
            t_infer += diag["t_inference"]
            t_total += diag["t_total"]
        best = max(range(len(Ts)), key=lambda x: (ovs[x], -mds[x]))
        dt, dr = [], []
        for a in range(len(Ts)):
            for b in range(a + 1, len(Ts)):
                dt.append(float(np.linalg.norm(Ts[a][:3, 3] - Ts[b][:3, 3])))
                dR = Ts[a][:3, :3].T @ Ts[b][:3, :3]
                c = (np.trace(dR) - 1.0) / 2.0
                dr.append(float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))))
        diag = {
            "overlap_ratio": ovs[best],
            "mean_dist": mds[best],
            "overlap_min": float(np.min(ovs)),
            "overlap_all": ovs,
            "mean_dist_all": mds,
            "disp_t": float(np.mean(dt)) if dt else 0.0,
            "disp_r": float(np.mean(dr)) if dr else 0.0,
            "n_generations": int(k),
            "t_preprocess": 0.0,
            "t_inference": t_infer,
            "t_total": t_total,
        }
        return Ts[best], diag

    # ------------------------------------------------------------------
    # N-way joint registration (RAP's native multi-part capability).
    # ------------------------------------------------------------------
    def register_multi(self, scans_points: list,
                       max_points_per_scan: Optional[int] = None):
        """Jointly register N unposed scans in a single RAP call.

        All scans are normalized together and transported by one rectified-
        flow sampling run, so every part lands in ONE common frame — the
        resulting per-part poses are cycle-consistent by construction.

        Args:
            scans_points: list of (N_i, 3) scans, each in its own sensor frame.
            max_points_per_scan: optional per-scan point cap override (use
                fewer points for large batches; global attention is O(TP^2)).

        Returns:
            T_parts: list of (4, 4) transforms; T_parts[i] maps scan i from
                its own frame into the common registered frame, i.e. the
                pose of scan i in the RAP common frame. Relative pose of j
                in i's frame: inv(T_i) @ T_j.
            diagnostics: dict with n_parts, per-part point counts, anchor
                index, total points, and timing.
        """
        import torch
        from rectified_point_flow.procrustes import fit_transformations

        n_parts = len(scans_points)
        if n_parts < 2:
            raise ValueError("register_multi needs at least 2 scans")

        t_start = time.time()
        ds_list, feat_list = [], []
        for pts in scans_points:
            ds, feat = self.preprocess(pts, max_points=max_points_per_scan)
            ds_list.append(ds)
            feat_list.append(feat)
        t_prep = time.time() - t_start

        cond, scale, offsets_c, counts, anchor_idx = self._normalize_parts(ds_list)
        features = np.concatenate(feat_list).astype(np.float32)
        tp = int(counts.sum())
        offsets = np.concatenate([[0], np.cumsum(counts)])

        anchor_indices = np.zeros(tp, dtype=bool)
        anchor_indices[offsets[anchor_idx]:offsets[anchor_idx + 1]] = True

        batch = {
            "pointclouds": torch.from_numpy(cond).to(self.device),
            "pointclouds_gt": torch.from_numpy(cond).to(self.device),  # unused by sampler
            "features": torch.from_numpy(features).to(self.device),
            "scales": torch.tensor([scale], dtype=torch.float32, device=self.device),
            "anchor_indices": torch.from_numpy(anchor_indices).to(self.device),
            "points_per_part": torch.from_numpy(counts[None, :]).to(self.device),
            "cu_seqlens": torch.tensor([0, tp], dtype=torch.int64, device=self.device),
        }

        t_infer = time.time()
        with torch.inference_mode(), torch.autocast(device_type=self.device.type,
                                                    dtype=torch.float16,
                                                    enabled=self.device.type == "cuda"):
            result = self.model.sample_rectified_flow(
                batch, None, return_tarjectory=True)
            pred = result["end_point_trajectory"][-1]  # (TP, 3), scaled space
            cu_seqlens_batch = batch["cu_seqlens"].to(torch.int32)
            rotations, translations = fit_transformations(
                batch["pointclouds"], pred, batch["points_per_part"], cu_seqlens_batch)
        t_infer = time.time() - t_infer

        # Per-part transforms from original scan frame -> common frame (meters).
        rots = rotations[0].float().cpu().numpy()      # (N, 3, 3)
        trans = translations[0].float().cpu().numpy()  # (N, 3)
        T_parts = []
        for i in range(n_parts):
            T = np.eye(4)
            T[:3, :3] = rots[i]
            T[:3, 3] = trans[i] * scale - rots[i] @ offsets_c[i]
            T_parts.append(T)

        diag = {
            "n_parts": n_parts,
            "counts": counts.tolist(),
            "total_points": tp,
            "anchor_idx": int(anchor_idx),
            "scale": scale,
            "t_preprocess": t_prep,
            "t_inference": t_infer,
            "t_total": time.time() - t_start,
        }
        return T_parts, diag

    @staticmethod
    def _alignment_metrics(ds_a: np.ndarray, ds_b: np.ndarray, T_rel: np.ndarray,
                           thresh: float = 1.0) -> Tuple[float, float]:
        """Symmetric NN statistics of the aligned clouds.

        Returns:
            overlap: fraction of points with a mutual nearest neighbor
                within thresh (m) — mean of both directions.
            mean_dist: mean symmetric nearest-neighbor distance (m) of the
                same distance field. Separates true from false registrations
                far more sharply than overlap (KITTI 02: true 0.39-1.45 m,
                false 3.7-22.1 m).
        """
        from scipy.spatial import cKDTree
        if len(ds_a) == 0 or len(ds_b) == 0:
            return 0.0, float("inf")
        b_in_a = (T_rel[:3, :3] @ ds_b.T).T + T_rel[:3, 3]
        tree_a = cKDTree(ds_a)
        tree_b = cKDTree(b_in_a)
        d_ba, _ = tree_a.query(b_in_a, k=1)
        d_ab, _ = tree_b.query(ds_a, k=1)
        overlap = 0.5 * (float(np.mean(d_ba < thresh)) +
                         float(np.mean(d_ab < thresh)))
        mean_dist = 0.5 * (float(np.mean(d_ba)) + float(np.mean(d_ab)))
        return overlap, mean_dist

    @classmethod
    def _overlap_ratio(cls, ds_a: np.ndarray, ds_b: np.ndarray, T_rel: np.ndarray,
                       thresh: float = 1.0) -> float:
        """Fraction of points with a mutual nearest neighbor within thresh (m)."""
        return cls._alignment_metrics(ds_a, ds_b, T_rel, thresh)[0]
