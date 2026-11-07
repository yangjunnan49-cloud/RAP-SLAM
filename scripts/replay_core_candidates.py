"""Fresh loop-stage replay with frozen saved frontends and one fixed protocol.

This does not rerun odometry. Every newly proposed candidate is measured and logged.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import uuid
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from rap_slam import scan_context,loop_closure,pose_graph,sequence
from rap_slam.verification_log import VerificationLog
from rap_slam.measurement import historical_margin,pair_id
import eval_paper_metrics as metrics


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class LazyScans:
    def __init__(self,paths):self.paths=paths;self.hashes={}
    def __getitem__(self,index):
        path=self.paths[index];raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        if index in self.hashes and self.hashes[index]!=digest:raise ValueError('Input scan changed during run')
        self.hashes[index]=digest
        xyz=np.frombuffer(raw,dtype=np.float32).reshape(-1,4)[:,:3]
        return xyz[np.linalg.norm(xyz,axis=1)>=1.5]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    (out/'INCOMPLETE').write_text('Loop-stage replay running\n')
    protocol=dict(frontend='frozen saved odometry and keyframes',budget=12,loop_method='both',
                  proximity_m=15,temporal_separation=200,sc_threshold=.25,sc_shortlist=10,
                  seed_per_sequence=42,generations=1,overlap_min=.7,mean_dist_max=2.5,
                  pgo='relfloor+Huber',nms=0,source_preprocessing='minimum range 1.5m, no KITTI205',
                  limitation='Saved frontend frame binding assumes historical complete config range. This is not a full end-to-end rerun or independent calibration sample.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    import torch
    from rap_slam.rap_backend import RAPBackend
    backend=RAPBackend()
    provenance=dict(checkpoint_sha256=sha(backend.ckpt_path),spinnet_sha256=sha(backend.spinnet_ckpt),
                    precision='ieee_float32_matmul_scope',seed_policy='shared backend RNG in recorded candidate order; reset42 per sequence',
                    source_sha256={name:sha(ROOT/name) for name in ['rap_slam/rap_backend.py','rap_slam/rap_precision.py',
                        'rap_slam/measurement.py','rap_slam/verification_log.py','rap_slam/loop_closure.py',
                        'rap_slam/scan_context.py','rap_slam/pose_graph.py','scripts/replay_core_candidates.py']})
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    reports=[]
    for name,directory in [('poss_off','poss'),('kitti06_off','kitti06'),('kitti05','kitti05'),('kitti00','kitti00_bb')]:
        parent=ROOT/'output_prods'/directory;dest=out/name;dest.mkdir()
        cfg=json.loads((parent/'summary.json').read_text())['config']
        odom=metrics.load_kitti_format(str(parent/'odom_poses.txt'));gt=metrics.gt_from_summary(str(parent))
        if len(odom)!=len(gt):raise ValueError('Frozen frontend/GT length mismatch')
        kfs=json.loads((parent/'keyframes.json').read_text())
        source='kitti' if cfg['real_apply_calib'] else 'poss';seq=cfg['real_seq']
        ids=cfg['real_start']+np.arange(len(odom))*cfg['real_stride']
        base=Path(cfg['real_root'])/'dataset';scan_dir=base/'sequences'/seq/'velodyne'
        scans=LazyScans([scan_dir/f'{frame:06d}.bin' for frame in ids])
        sources={str(parent/file):sha(parent/file) for file in ['summary.json','odom_poses.txt','keyframes.json','accepted_loops.json','optimized_poses.txt']}
        sources.update({str(base/'poses'/f'{seq}.txt'):sha(base/'poses'/f'{seq}.txt'),
                        str(base/'sequences'/seq/'calib.txt'):sha(base/'sequences'/seq/'calib.txt')})
        (dest/'parents.json').write_text(json.dumps(sources,indent=2))
        print(f'{name}: descriptors for {len(kfs)} frozen keyframes',flush=True)
        descriptors=scan_context.build_descriptors(scans,kfs)
        prox=loop_closure.find_candidates(odom,kfs,dist_thresh=15,min_separation=200,max_candidates=12)
        sc=scan_context.find_candidates_sc(descriptors,kfs,min_separation=200,dist_thresh=.25,shortlist=10,max_candidates=12)
        candidates=loop_closure.merge_candidates(prox,sc,max_candidates=12)
        (dest/'candidates.json').write_text(json.dumps(dict(proximity=prox,scan_context=sc,merged=candidates),indent=2))
        global_desc={int(ids[i]):d['sc'] for i,d in descriptors.items()}
        features=[]
        for i,j in candidates:
            features.append(dict(physical_pair_id=pair_id(source,seq,[int(ids[i]),int(ids[j])]),
                **historical_margin(int(ids[i]),int(ids[j]),[int(ids[k]) for k in kfs],global_desc)))
        (dest/'candidate_features.json').write_text(json.dumps(features,indent=2))
        log=VerificationLog(dest/'verification_measurements.jsonl',str(uuid.uuid4()),source,seq,ids,gt,
                            dict(provenance,frame_binding='explicit_source_ids_with_legacy_frontend_range_assumption'))
        torch.manual_seed(42);np.random.seed(42)
        accepted=loop_closure.verify_candidates(candidates,scans,odom,backend,gt_poses=gt,measurement_log=log)
        if log.summary()['failed'] or log.summary()['started']:raise RuntimeError('Incomplete candidate verification')
        (dest/'accepted_loops.json').write_text(json.dumps([{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in e.items()} for e in accepted],indent=2))
        (dest/'source_frames.json').write_text(json.dumps(dict(source_frame_ids=ids.tolist(),scan_files=[str(p) for p in scans.paths],timestamps=None),indent=2))
        (dest/'scan_hashes.json').write_text(json.dumps({str(scans.paths[i]):h for i,h in scans.hashes.items()},indent=2))
        optimized=pose_graph.optimize_pose_graph(odom,kfs,accepted,loop_weighting='relfloor',robust_kernel='huber',loop_nms_width=0)
        full=pose_graph.propagate_to_full_trajectory(odom,kfs,optimized)
        sequence.save_kitti_poses(np.stack(full),str(dest/'optimized_poses.txt'))
        current=metrics.evaluate_pair(gt,full,ids);baseline=metrics.evaluate_pair(gt,odom,ids)
        old=metrics.evaluate_pair(gt,metrics.load_kitti_format(str(parent/'optimized_poses.txt')),ids)
        report=dict(name=name,candidates=len(candidates),counts=log.summary(),
            correctness=dict(Counter(r['labels']['label'] for r in log.measurements.values())),
            odometry=baseline,legacy_pgo=old,fresh_pgo=current,
            comparison_limit='Fresh protocol has fixed budget12; parent KITTI00 big-budget protocol differs. No claim of isolating only precision for that parent.')
        (dest/'metrics.json').write_text(json.dumps(report,indent=2));reports.append(report)
        for path,digest in sources.items():assert sha(path)==digest
        print(f"{name}: {len(accepted)}/{len(candidates)} accepted; first APE {baseline['ape_unaligned_rmse']:.3f} -> {current['ape_unaligned_rmse']:.3f}",flush=True)
    (out/'summary.json').write_text(json.dumps(dict(protocol=protocol,sequences=reports),indent=2))
    lines=['# Fresh loop-stage replay on frozen frontends','',
           'Uniform budget 12; one fresh registration per merged candidate; heuristic gate; NMS 0; relfloor + Huber. Odometry and keyframes are reused, not rerun.','',
           '| Sequence | Candidates | Accepted | Odom first APE m | Fresh first APE m | Odom aligned APE m | Fresh aligned APE m |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for r in reports:
        a,b=r['odometry'],r['fresh_pgo']
        lines.append(f"| {r['name']} | {r['candidates']} | {r['counts']['accepted']} | {a['ape_unaligned_rmse']:.4f} | {b['ape_unaligned_rmse']:.4f} | {a['ape_aligned_rmse']:.4f} | {b['ape_aligned_rmse']:.4f} |")
    lines+=['','These naturally proposed candidates are budget-filtered and share sites/scans. They are not an independent calibration/test population. All individual events, source identities and measurements are retained.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n');(out/'INCOMPLETE').unlink()

if __name__=='__main__':main()
