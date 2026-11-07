"""HL-V2 real-data bridge: complete sources, broader candidates, causal raw supports."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from rap_slam.rap_backend import RAPBackend
from rap_slam.odometry import KissOdometry
from rap_slam.keyframes import select_keyframes
from rap_slam import scan_context,real_dataset
from hl0_preflight import support
from hl1_cohort_audit import coverage,stratum,sample,STRATA
MAIN=Path('/mnt/e/githouse/RAP-main');OUT=MAIN/'docs/maintenance/2026-09-14/hl-v2/data-expansion-01'
DATA=Path('/mnt/e/githouse/Datasets/data_odometry_velodyne/dataset')
COUNTS={'00':4541,'02':4661,'05':2761,'07':1101,'08':4071,'09':1591}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def pairhash(seq,i,j):return hashlib.sha256(f'HL-V2|kitti|{seq}|{i}|{j}'.encode()).hexdigest()
def run(seq):
    t=time.perf_counter();out=OUT/seq;out.mkdir(exist_ok=False);(out/'INCOMPLETE').write_text('Data bridge in progress\n')
    n=COUNTS[seq];base=DATA/'sequences'/seq;ts=np.loadtxt(base/'times.txt');assert len(ts)==n and (np.diff(ts)>0).all()
    paths=sorted((base/'velodyne').glob('*.bin'));assert [int(p.stem) for p in paths]==list(range(n))
    gt=np.array(real_dataset.load_gt_poses(str(DATA/'poses'/f'{seq}.txt'),str(base/'calib.txt'),start=0,end=n-1,apply_calib=True));assert len(gt)==n
    parent=MAIN/f'docs/maintenance/2026-09-11/full-kitti{seq}-01'
    odom=None;parent_hashes={}
    if seq in ['00','05']:
        assert not (parent/'INCOMPLETE').exists()
        manifest=json.loads((parent/'input_manifest.json').read_text())
        for p,h in manifest.items():assert sha(p)==h
        source=json.loads((parent/'pipeline/source_frames.json').read_text());assert source['source_frame_ids']==list(range(n)) and np.array_equal(source['timestamps'],ts)
        odom=np.repeat(np.eye(4)[None],n,axis=0);odom[:,:3,:]=np.loadtxt(parent/'pipeline/odom_poses.txt').reshape(n,3,4)
        kfs=json.loads((parent/'pipeline/keyframes.json').read_text())
        parent_hashes={str(parent/'pipeline'/p):sha(parent/'pipeline'/p) for p in ['odom_poses.txt','keyframes.json','source_frames.json','summary.json']}
    runner=KissOdometry() if odom is None else None;scan_rows=[]
    for i,p in enumerate(paths):
        raw=p.read_bytes();v=np.frombuffer(raw,dtype=np.float32);assert len(v)>0 and len(v)%4==0 and np.isfinite(v).all()
        scan_rows.append(dict(source_id=i,path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
        if runner is not None:runner.register(v.reshape(-1,4)[:,:3])
        if i%500==0:print(seq,'source/odometry',i,'/',n,flush=True)
    if runner is not None:odom=np.array(runner.poses);kfs=select_keyframes(odom)
    np.save(out/'odom.npy',odom);np.save(out/'evaluator_gt.npy',gt)
    save(out/'data.json',dict(dataset='kitti',sequence=seq,source_ids=list(range(n)),timestamps=ts.tolist(),scans=scan_rows,keyframes=kfs,
        parent_hashes=parent_hashes,metadata_sha256={str(p):sha(p) for p in [base/'times.txt',base/'calib.txt',DATA/'poses'/f'{seq}.txt']},
        frontend='frozen corrected parent' if runner is None else 'fresh KISS-ICP60/1.5/0.8/no-deskew, keyframes6m12deg forcedlast'))
    geom=RAPBackend.__new__(RAPBackend);geom.voxel_size=.4;geom.remove_outliers=True;geom.max_points_per_scan=4000;geom.seed=42;geom.feature_extractor=None
    desc={};clouds={}
    for nidx,i in enumerate(kfs):
        v=np.fromfile(paths[i],dtype=np.float32).reshape(-1,4)[:,:3];v=v[np.linalg.norm(v,axis=1)>=1.5]
        s=scan_context.make_scan_context(v);desc[i]=dict(sc=s,rk=scan_context.make_ring_key(s));clouds[i]=geom.preprocess(v)[0]
        if nidx%150==0:print(seq,'descriptor/geometry',nidx,'/',len(kfs),flush=True)
    np.savez_compressed(out/'label_geometry.npz',**{str(i):p for i,p in clouds.items()})
    np.savez_compressed(out/'descriptors.npz',ids=kfs,sc=np.stack([desc[i]['sc'] for i in kfs]))
    cum=np.r_[0,np.cumsum(np.linalg.norm(np.diff(odom[:,:3,3],axis=0),axis=1))]
    supports={i:support(i,-1,list(range(n)),cum,ts,odom) for i in kfs}
    save(out/'support_metadata.json',supports)
    queries=[];registry=[]
    for j in kfs:
        past=[i for i in kfs if j-i>=200]
        ring=sorted([(i,float(np.linalg.norm(desc[i]['rk']-desc[j]['rk']))) for i in past],key=lambda v:(v[1],v[0]))
        near=sorted([(i,float(np.linalg.norm(odom[i,:3,3]-odom[j,:3,3]))) for i in past],key=lambda v:(v[1],v[0]))
        short=ring[:50];dd,ss=scan_context.sc_distances_batch(np.stack([desc[i]['sc'] for i,_ in short]),desc[j]['sc']) if short else ([],[])
        scored=sorted([dict(source_i=i,ring_distance=rd,ring_rank=rank+1,sc_distance=float(d),shift=int(s)) for rank,((i,rd),d,s) in enumerate(zip(short,dd,ss))],key=lambda r:(r['sc_distance'],r['source_i']))
        for rank,r in enumerate(scored):r['sc_rank']=rank+1;r['passes_v1_cutoff']=r['sc_distance']<=.25
        sc=scored[:20];prox=[i for i,d in near if d<=30][:5];bands=[]
        for band in range(10):
            lo=30+band*5;hi=lo+5;pool=[i for i,d in near if d>=lo and (d<hi or band==9 and d<=hi)]
            if pool:bands.append(dict(source_i=min(pool,key=lambda i:pairhash(seq,i,j)),low=lo,high=hi))
        close=[r['source_i'] for r in sc if r['sc_distance']<=sc[0]['sc_distance']+.02] if sc else []
        amb=len(close)>=2 and max(close)-min(close)>=200
        queries.append(dict(source_j=j,all_ring=ring,all_odom=near,sc_shortlist=scored,sc_retained=sc,proximity=prox,bands=bands,ambiguous=amb,ambiguous_ids=close))
        scmap={r['source_i']:r for r in sc};bmap={r['source_i']:r for r in bands};distmap=dict(near)
        for i in sorted(set(scmap)|set(prox)|set(bmap)):
            a=supports[i];b=supports[j]
            assert not ({i,j}&{s['source_id'] for s in a+b})
            registry.append(dict(source_i=i,source_j=j,sequence=seq,sources=(['sc'] if i in scmap else [])+(['proximity'] if i in prox else [])+(['offset_band'] if i in bmap else []),
                sc=scmap.get(i),odom_distance=distmap[i],band=bmap.get(i),support_i=a,support_j=b,supported=len(a)==2 and len(b)==2,sampling_hash=pairhash(seq,i,j)))
    save(out/'queries_evaluator.json',queries);save(out/'candidate_registry.json',registry)
    qmap={q['source_j']:q for q in queries};labels=[]
    for index,r in enumerate(registry):
        i,j=r['source_i'],r['source_j'];T=np.linalg.inv(gt[i])@gt[j];o,a,b=coverage(clouds[i],clouds[j],T);distance=float(np.linalg.norm(T[:3,3]))
        k=stratum('sc' in r['sources'],distance,o,qmap[j]['ambiguous']);labels.append(dict(r,stratum=k,GT_overlap=o,GT_distance_m=distance,GT_coverage_i=a,GT_coverage_j=b))
        if index%1000==0:print(seq,'labelled',index,'/',len(registry),flush=True)
    selected=sample(labels);save(out/'evaluator_sampling.json',labels);save(out/'selected_evaluator.json',selected)
    save(out/'selected_inputs.json',[{k:r[k] for k in ['sequence','source_i','source_j','support_i','support_j']} for r in selected])
    strata=[dict(stratum=k,population=sum(r['stratum']==k for r in labels),supported=sum(r['stratum']==k and r['supported'] for r in labels),selected=sum(r['stratum']==k for r in selected)) for k in STRATA]
    summary=dict(sequence=seq,frames=n,keyframes=len(kfs),candidates=len(registry),supported_candidates=sum(r['supported'] for r in registry),strata=strata,selected=len(selected),adequate=len(selected)>=60 and all(r['selected']>=10 for r in strata),headroom=all(r['selected']>=20 for r in strata),wall_s=time.perf_counter()-t,hypotheses=0,source_sha256=sha(__file__))
    save(out/'adequacy.json',summary);print(json.dumps(summary),flush=True)
    # Full raw source bytes were validated before use; retain hashes and exclusive products.
    (out/'DATA_READY').write_text('Data-only audit computed; independent validation pending.\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sequence',choices=list(COUNTS),required=True);args=p.parse_args()
    try:run(args.sequence)
    except Exception:
        (OUT/f'failure_{args.sequence}.txt').write_text(traceback.format_exc());raise