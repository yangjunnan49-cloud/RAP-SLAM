"""HL-V1 bounded geometry, causal-support and request-local RNG feasibility."""
import contextlib
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import traceback
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from rap_slam.measurement import rotation_validity, transform_label
import eval_paper_metrics as metrics
OUT=Path('/mnt/e/githouse/RAP-main/docs/maintenance/2026-09-14/hl-v1/preflight-01')

def save(name,obj):
    p=OUT/name
    with p.open('x') as f: json.dump(obj,f,indent=2,allow_nan=False)

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pose(r,t):
    v=np.eye(4);v[:3,:3]=Rotation.from_rotvec(r).as_matrix();v[:3,3]=t;return v

def move(T,p): return p@T[:3,:3].T+T[:3,3]

def support(anchor,other,kfs,cumulative,timestamps,odom):
    chosen=[]
    for target in (2.,5.):
        eligible=[s for s in kfs if s<anchor and s!=other and cumulative[anchor]-cumulative[s]>=target]
        if not eligible: continue
        s=max(eligible)
        if timestamps[anchor]-timestamps[s]<=5 and s not in [x['source_id'] for x in chosen]:
            chosen.append(dict(source_id=s,timestamp=float(timestamps[s]),age_s=float(timestamps[anchor]-timestamps[s]),
                travelled_m=float(cumulative[anchor]-cumulative[s]),local_pose=(np.linalg.inv(odom[anchor])@odom[s]).tolist()))
    return chosen

def score(T,left,right):
    combos=[]
    for A,P in left:
        for B,Q in right:
            if not len(P) or not len(Q): return None
            x,y=move(A,P),move(T@B,Q)
            d=cKDTree(y).query(x,workers=1)[0]; e=cKDTree(x).query(y,workers=1)[0]
            if not np.isfinite(d).all() or not np.isfinite(e).all(): return None
            combos.append(dict(score=float((np.minimum(d,3).mean()+np.minimum(e,3).mean())/2),
                median=float(np.median(np.r_[d,e])),p90=float(np.percentile(np.r_[d,e],90)),
                coverage_1m=float(((d<=1).mean()+(e<=1).mean())/2)))
    if len(combos)!=4:return None
    return dict(score=float(np.mean([c['score'] for c in combos])),combinations=combos)

@contextlib.contextmanager
def request_rng(seed,backend):
    import torch
    py=random.getstate();npstate=np.random.get_state();old=backend.seed
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        try:
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            backend.seed=seed
            yield
        finally:
            random.setstate(py);np.random.set_state(npstate);backend.seed=old

def request_seed(i,j,g):
    return int(hashlib.sha256(f'HL-V1|hypothesis|kitti|05|{i}|{j}|{g}'.encode()).hexdigest()[:8],16)%(2**31)

def geometry_checks():
    poses=[pose([.1,.2,.3],[1,2,3]),pose([-.3,.2,.1],[8,-4,2]),pose([.2,-.2,.4],[-2,4,7]),pose([-.1,.3,-.2],[4,5,8])]
    I,J,S,U=poses; A=np.linalg.inv(I)@S;B=np.linalg.inv(J)@U;T=np.linalg.inv(I)@J
    P=np.random.default_rng(19).normal(size=(200,3));Q=move(np.linalg.inv(U)@S,P)
    error=float(np.max(np.abs(move(A,P)-move(T@B,Q))))
    equivalent=float(np.max(np.abs(np.linalg.inv(A)@T@B-np.linalg.inv(S)@U)))
    assert max(error,equivalent)<1e-12
    for rte,rre in [(0.,0.),(.99,4.99),(1.01,4.99),(.99,5.01)]:
        estimate=T@pose([0,0,np.deg2rad(rre)],[rte,0,0]);lab=transform_label(estimate,T)
        assert abs(lab['gt_translation_error_m']-rte)<1e-10 and abs(lab['gt_rotation_error_deg']-rre)<1e-10
    reflection=np.eye(4);reflection[0,0]=-1;assert not rotation_validity(reflection)['valid']
    assert score(T,[(A,P),(A,P)],[(B,Q),(B,Q)])['score']<1e-12
    assert score(T,[(A,P[:0]),(A,P)],[(B,Q),(B,Q)]) is None
    # Distinct target crossings, duplicate target, causality and age boundary.
    O=np.repeat(np.eye(4)[None],8,axis=0);O[:,0,3]=np.arange(8)
    assert [s['source_id'] for s in support(7,0,list(range(8)),np.arange(8),np.arange(8)*.1,O)]==[5,2]
    assert len(support(7,0,[0,1,7],np.arange(8),np.arange(8)*.1,O))==1
    assert not support(7,0,[0,1,7],np.arange(8),np.arange(8)*2,O)
    save('geometry_validation.json',dict(transform_point_error=error,transform_composition_error=equivalent,
        strict_threshold_tests=4,reflection_rejected=True,empty_support_invalid=True,support_boundary_tests=3))

def main():
    started=time.perf_counter();geometry_checks()
    data={seq:json.loads((OUT/f'data_{seq}.json').read_text()) for seq in ['05','06']}
    gt_stats=[];eligible_anchors={};eligible_pairs=[]
    for seq,d in data.items():
        parent=Path(d['parent'])/'pipeline';odom=np.array(metrics.load_kitti_format(str(parent/'odom_poses.txt')))
        gt=np.array(metrics.gt_from_summary(str(parent)));assert len(gt)==len(odom)
        validity=[rotation_validity(T,1e-4) for T in gt];assert all(v['valid'] for v in validity)
        # Text-rounding sensitivity: SVD projection only in this diagnostic, never repair saved hypotheses.
        delta=[]
        for T in gt:
            u,s,v=np.linalg.svd(T[:3,:3]); delta.append(float(np.max(np.abs(T[:3,:3]-u@v))))
        gt_stats.append(dict(sequence=seq,max_orthogonality=max(v['orthogonality'] for v in validity),max_det_error=max(v['det_error'] for v in validity),max_projection_element_change=max(delta),
            interpretation='Serialized GT rigidity is adequate for 1m/5deg numeric labels; this does not bound physical GT measurement uncertainty.'))
        cumulative=np.r_[0,np.cumsum(np.linalg.norm(np.diff(odom[:,:3,3],axis=0),axis=1))]
        kfs=d['keyframes'];ts=d['timestamps'];supports={i:support(i,-1,kfs,cumulative,ts,odom) for i in kfs}
        eligible_anchors[seq]=dict(total=len(kfs),two_supports=sum(len(s)==2 for s in supports.values()),zero_supports=sum(len(s)==0 for s in supports.values()),one_support=sum(len(s)==1 for s in supports.values()))
        save(f'support_metadata_{seq}.json',supports)
        if seq=='05':
            for j in kfs:
                for i in kfs:
                    if j-i<200:continue
                    distance=float(np.linalg.norm(odom[i,:3,3]-odom[j,:3,3]))
                    if distance>30:continue
                    a=support(i,j,kfs,cumulative,ts,odom);b=support(j,i,kfs,cumulative,ts,odom)
                    eligible_pairs.append(dict(source_i=i,source_j=j,distance_m=distance,support_i=a,support_j=b,eligible=len(a)==2 and len(b)==2,
                        sampling_hash=hashlib.sha256(f'HL-V1|kitti|05|{i}|{j}'.encode()).hexdigest()))
    save('gt_precision.json',gt_stats);save('support_availability.json',eligible_anchors)
    save('profile_candidate_registry.json',eligible_pairs)
    selected=sorted([p for p in eligible_pairs if p['eligible']],key=lambda p:p['sampling_hash'])[:10]
    save('profile_pairs.json',selected)
    print('Support availability',eligible_anchors,'eligible profile pairs',sum(p['eligible'] for p in eligible_pairs),flush=True)
    if len(selected)<10:
        save('resource_blocker.json',dict(required_pairs=10,available_pairs=len(selected),decision='INCONCLUSIVE',reason='Insufficient eligible profile pairs under fixed causal support rule'))
        return
    import torch,open3d,platform
    from rap_slam.rap_backend import RAPBackend
    save('environment.json',dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),open3d=open3d.__version__,platform=platform.platform(),threads=torch.get_num_threads()))
    backend=RAPBackend();geom=RAPBackend.__new__(RAPBackend)
    geom.voxel_size=.4;geom.remove_outliers=True;geom.max_points_per_scan=4000;geom.seed=42;geom.feature_extractor=None
    save('runner_identity.json',dict(path=str(Path(__file__).resolve()),sha256=sha(__file__),backend_sha256=sha(ROOT/'rap_slam/rap_backend.py'),checkpoint_sha256=sha(backend.ckpt_path),spinnet_sha256=sha(backend.spinnet_ckpt)))
    def scan(i):
        xyz=np.fromfile(data['05']['scans'][i]['path'],dtype=np.float32).reshape(-1,4)[:,:3]
        return xyz[np.linalg.norm(xyz,axis=1)>=1.5]
    def generate(i,j,g):
        seed=request_seed(i,j,g)
        with request_rng(seed,backend): T,diag=backend.register(scan(i),scan(j))
        return dict(source_i=i,source_j=j,generation=g,seed=seed,T=T.tolist(),diagnostics=diag,validity=rotation_validity(T))
    (OUT/'bank').mkdir(exist_ok=False);profiles=[];bank={}
    for number,p in enumerate(selected):
        i,j=p['source_i'],p['source_j'];torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
        supportsets=[]
        for key in ['support_i','support_j']:
            supportsets.append([(np.array(s['local_pose']),geom.preprocess(scan(s['source_id']))[0]) for s in p[key]])
        prep=time.perf_counter()-t; rows=[]
        for g in range(4):
            row=generate(i,j,g);row['support']=score(np.array(row['T']),*supportsets) if row['validity']['valid'] else None
            save(f'bank/{i}_{j}_{g}.json',row);bank[(i,j,g)]=row;rows.append(row)
        torch.cuda.synchronize();elapsed=time.perf_counter()-t
        profiles.append(dict(source_i=i,source_j=j,wall_s=elapsed,support_preprocess_s=prep,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),valid_generations=sum(r['validity']['valid'] for r in rows)))
        save(f'profile_{number:02d}.json',profiles[-1]);print('Profile',number+1,profiles[-1],flush=True)
    # Repeat first request alone after the profile; then reverse a two-request order.
    keys=list(bank)[:2];checks=[]
    for key in [keys[0],keys[1],keys[0]]:
        repeat=generate(*key);original=bank[key];delta=float(np.max(np.abs(np.array(repeat['T'])-original['T'])))
        cached=json.loads((OUT/f'bank/{key[0]}_{key[1]}_{key[2]}.json').read_text())
        assert cached==original
        checks.append(dict(request=list(key),max_transform_element_delta=delta,exact=delta==0,cache_exact=True))
    save('rng_validation.json',dict(tolerance=1e-6,passed=all(c['max_transform_element_delta']<=1e-6 for c in checks),checks=checks,scope='One isolated repeat and reversed two-request order; exact saved-bank cache reload; CPU/Python/CUDA RNG states restored'))
    save('resource_summary.json',dict(pairs=len(profiles),generations=40,profiles=profiles,total_wall_s=time.perf_counter()-started,profile_wall_s=sum(p['wall_s'] for p in profiles),cuda_oom=False))

if __name__=='__main__':
    try:main()
    except Exception:
        (OUT/'failure.txt').write_text(traceback.format_exc());raise