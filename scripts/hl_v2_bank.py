"""Immutable HL-V2 K4 bank generation; no GT/strata consumed."""
import argparse
import hashlib,json,random,sys,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from hl0_preflight import request_rng,score
from rap_slam.rap_backend import RAPBackend
from rap_slam.measurement import rotation_validity
MAIN=Path('/mnt/e/githouse/RAP-main');BASE=MAIN/'docs/maintenance/2026-09-14/hl-v2'
DATA=BASE/'data-expansion-01'

def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def seed(seq,i,j,g):return int(hashlib.sha256(f'HL-V2|hypothesis|kitti|{seq}|{i}|{j}|{g}'.encode()).hexdigest()[:8],16)%(2**31)

def setup_backend():
    import torch
    b=RAPBackend()
    from rectified_point_flow.procrustes import fit_transformations
    from rectified_point_flow.eval.metrics import compute_rigidity_rmse
    original=b.model.sample_rectified_flow
    # Instance-only observation hook. Reuses original unmodified generated trajectory.
    def observed(batch,*args,**kwargs):
        result=original(batch,*args,**kwargs);vals=[]
        for pred in result['end_point_trajectory']:
            cu=batch['cu_seqlens'].to(torch.int32)
            r,t=fit_transformations(batch['pointclouds'],pred,batch['points_per_part'],cu)
            vals.append(compute_rigidity_rmse(batch['pointclouds'],pred,r,t,batch['points_per_part'],cu,batch['scales']))
        b.last_rigidity=float(torch.stack(vals).mean().item())
        return result
    b.model.sample_rectified_flow=observed
    original_features=b.feature_extractor.extract_features
    def checked_features(*args,**kwargs):
        result=original_features(*args,**kwargs);f=np.asarray(result['features'])
        if not np.isfinite(f).all() or not np.any(f):raise RuntimeError('Invalid or all-zero learned features')
        return result
    b.feature_extractor.extract_features=checked_features
    b.unobserved_sampler=original
    return b

def main(role,mode):
    import torch,platform
    inputs=DATA/f'{role}_inputs.json';frozen=json.loads((DATA/'cohort_freeze.json').read_text())
    assert sha(inputs)==frozen['hashes'][inputs.name]
    if role=='confirmation':
        frozen_discovery=json.loads((BASE/'pilot-01/discovery_frozen.json').read_text())
        assert all(sha(BASE/'pilot-01/discovery'/p)==h for p,h in frozen_discovery['hashes'].items())
    out=BASE/('preflight-01' if mode=='preflight' else 'pilot-01')/role
    out.mkdir(parents=True,exist_ok=False);(out/'INCOMPLETE').write_text('K4 bank in progress; do not treat partial results as a cohort\n');(out/'bank').mkdir()
    rows=json.loads(inputs.read_text());rows=rows[:10] if mode=='preflight' else rows
    save(out/'inputs.json',rows)
    save(out/'environment.json',dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),platform=platform.platform(),role=role,mode=mode))
    start=time.perf_counter();b=setup_backend();geom=RAPBackend.__new__(RAPBackend)
    geom.voxel_size=.4;geom.remove_outliers=True;geom.max_points_per_scan=4000;geom.seed=42;geom.feature_extractor=None
    meta={s:json.loads((DATA/s/'data.json').read_text()) for s in sorted({r['sequence'] for r in rows})};hashes={};timings=[]
    def scan(seq,i):
        r=meta[seq]['scans'][i];raw=Path(r['path']).read_bytes();digest=hashlib.sha256(raw).hexdigest();assert digest==r['sha256'];hashes[r['path']]=digest
        p=np.frombuffer(raw,dtype=np.float32).reshape(-1,4)[:,:3];return p[np.linalg.norm(p,axis=1)>=1.5]
    first=None
    def generate(r,g):
        seq,i,j=r['sequence'],r['source_i'],r['source_j'];s=seed(seq,i,j,g)
        with request_rng(s,b):T,diag=b.register(scan(seq,i),scan(seq,j))
        return dict(sequence=seq,source_i=i,source_j=j,generation=g,seed=s,T=T.tolist(),diagnostics=diag,validity=rotation_validity(T),rigidity_rmse_m=b.last_rigidity)
    for idx,r in enumerate(rows):
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter();supp=[]
        for key in ['support_i','support_j']:
            supp.append([(np.array(s['local_pose']),geom.preprocess(scan(r['sequence'],s['source_id']))[0]) for s in r[key]])
        for g in range(4):
            try:
                result=generate(r,g);result['support']=score(np.array(result['T']),*supp) if result['validity']['valid'] else None
            except torch.cuda.OutOfMemoryError:
                save(out/'engineering_failure.json',dict(type='CUDA_OOM',pair=idx,generation=g));raise
            except Exception as e:
                result=dict(sequence=r['sequence'],source_i=r['source_i'],source_j=r['source_j'],generation=g,seed=seed(r['sequence'],r['source_i'],r['source_j'],g),failure=repr(e),validity={'valid':False},support=None)
            name=f"{r['sequence']}_{r['source_i']}_{r['source_j']}_{g}.json";save(out/'bank'/name,result)
            if first is None:first=(r,result)
        torch.cuda.synchronize();timings.append(dict(pair_index=idx,wall_s=time.perf_counter()-t,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()))
        if idx%10==0:print(role,mode,idx+1,'/',len(rows),'wall',timings[-1]['wall_s'],flush=True)
    if mode=='preflight':
        r,original=first;assert 'T' in original;repeat=generate(r,0);delta=float(np.max(np.abs(np.array(repeat['T'])-original['T'])))
        # Observability hook must not change the generated transformation.
        hooked=b.model.sample_rectified_flow;b.model.sample_rectified_flow=b.unobserved_sampler
        try:plain=generate(r,0)
        finally:b.model.sample_rectified_flow=hooked
        hookdelta=float(np.max(np.abs(np.array(plain['T'])-original['T'])));assert delta<=1e-6 and hookdelta<=1e-6
        save(out/'rng_hook_validation.json',dict(reordered_max_delta=delta,hook_vs_original_max_delta=hookdelta,tolerance=1e-6,passed=True))
    save(out/'timings.json',dict(pairs=timings,wall_s=time.perf_counter()-start,profile_pair_wall_s=sum(t['wall_s'] for t in timings)))
    save(out/'input_hashes.json',hashes)
    save(out/'provenance.json',dict(input_manifest_sha256=sha(inputs),runner_sha256=sha(__file__),backend_sha256=sha(ROOT/'rap_slam/rap_backend.py'),checkpoint_sha256=sha(b.ckpt_path),spinnet_sha256=sha(b.spinnet_ckpt),rigidity='Official average per-step rigidity RMSE using original model functions on returned endpoint trajectory; additional named selector R4; F02 IEEE scope retained'))
    save(out/'bank_hashes.json',{p.name:sha(p) for p in sorted((out/'bank').glob('*.json'))})
    print('Bank complete',out,len(rows)*4,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--role',choices=['discovery','confirmation'],required=True);p.add_argument('--mode',choices=['preflight','pilot'],required=True);a=p.parse_args();main(a.role,a.mode)