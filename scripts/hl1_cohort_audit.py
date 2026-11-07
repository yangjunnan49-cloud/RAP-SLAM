"""HL-V1 data-only candidate/stratum adequacy audit; no hypothesis inference."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from rap_slam import scan_context
from rap_slam.rap_backend import RAPBackend
import eval_paper_metrics as metrics
from hl0_preflight import support,move
MAIN=Path('/mnt/e/githouse/RAP-main')
PREFLIGHT=MAIN/'docs/maintenance/2026-09-14/hl-v1/preflight-01'
PILOT=MAIN/'docs/maintenance/2026-09-14/hl-v1/pilot-01'
STRATA=['confuser','ambiguous_positive','low_overlap','ordinary']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(path,obj):
    with path.open('x') as f:json.dump(obj,f,indent=2,allow_nan=False)
def coverage(P,Q,T):
    if len(P)==0 or len(Q)==0:raise ValueError('Empty GT overlap cloud')
    Q=move(T,Q)
    a=float((cKDTree(Q).query(P,workers=1)[0]<=1).mean())
    b=float((cKDTree(P).query(Q,workers=1)[0]<=1).mean())
    return (a+b)/2,a,b

def stratum(sc,distance,overlap,ambiguous):
    if sc and distance>20 and overlap<.05:return 'confuser'
    if overlap>=.30 and ambiguous:return 'ambiguous_positive'
    if .10<=overlap<.30:return 'low_overlap'
    if overlap>=.50:return 'ordinary'
    return None

def sample(rows):
    selected=[];qcounts=Counter()
    for kind in STRATA:
        chosen=0
        for r in sorted([r for r in rows if r['stratum']==kind],key=lambda r:r['sampling_hash']):
            if not r['supported']:r['sampling_reason']='unsupported';continue
            if qcounts[r['source_j']]>=3:r['sampling_reason']='query_cap';continue
            if chosen>=25:r['sampling_reason']='stratum_quota';continue
            r['sampling_reason']='selected';selected.append(r);qcounts[r['source_j']]+=1;chosen+=1
    for r in rows:
        if r['stratum'] is None:r['sampling_reason']='overlap_gap'
    return selected

def checks():
    assert stratum(True,21,.049,True)=='confuser'
    assert stratum(True,20,.049,True) is None
    assert stratum(False,21,.049,True) is None
    assert stratum(True,21,.30,True)=='ambiguous_positive'
    assert stratum(False,0,.10,False)=='low_overlap'
    assert stratum(False,0,.30,False) is None
    assert stratum(False,0,.50,False)=='ordinary'
    x=np.array([[0.,0,0],[10,0,0]])
    assert coverage(x,x,np.eye(4))==(1.,1.,1.)
    assert coverage(x,x+100,np.eye(4))==(0.,0.,0.)
    rows=[dict(source_j=99,stratum=k,supported=True,sampling_hash=f'{i:02d}') for i,k in enumerate(['confuser']*2+['ambiguous_positive']*2+['ordinary'])]
    picked=sample(rows);assert len(picked)==3 and [r['stratum'] for r in picked]==['confuser','confuser','ambiguous_positive']
    rows=[dict(source_j=i,stratum='ordinary',supported=i!=0,sampling_hash=f'{i:02d}') for i in range(28)]
    assert len(sample(rows))==25 and rows[0]['sampling_reason']=='unsupported' and rows[-1]['sampling_reason']=='stratum_quota'
    return dict(stratum_boundary_cases=7,coverage_cases=2,sampling_cases=2,passed=True)

def run(seq):
    start=time.perf_counter()
    if seq=='05':
        PILOT.mkdir(parents=True,exist_ok=False)
        (PILOT/'INCOMPLETE').write_text('Data adequacy audit in progress; no H1/H2 decision.\n')
        protocol=MAIN/'docs/maintenance/2026-09-14/HARD_LOOP_EXPERIMENT_PROTOCOL_V1.md'
        (PILOT/'PROTOCOL.md').write_bytes(protocol.read_bytes())
        save(PILOT/'execution_plan.json',dict(protocol_sha256=sha(protocol),stage='HL1 adequacy before inference',
            order=['05 discovery, freeze report','06 data adequacy only after discovery freeze'],
            insufficient_action='INCONCLUSIVE; do not generate hypotheses or tune protocol after failed cohort minima',
            tie_order='distance ascending then source ID ascending',geometric_cap_seed=42,
            overlap_preprocess='minrange1.5, voxel0.4, statistical outlier20/2.5, cap4000, no learned features',
            full_population='All temporal-eligible historical ring/proximity distances plus shortlisted SC values persisted; registry union of up-to20 SC<=.25 and up-to5 proximity<=30 per query',
            ambiguous='At least two retained historical SC candidates within .02 of best SC distance and source separation>=200; best counts among the two'))
        save(PILOT/'unit_validation.json',checks())
    else:
        frozen=json.loads((PILOT/'discovery_frozen.json').read_text())
        assert all(sha(PILOT/'05'/p)==h for p,h in frozen['hashes'].items())
    out=PILOT/seq;out.mkdir(exist_ok=False)
    (out/'INCOMPLETE').write_text('Candidate and stratum audit in progress.\n')
    save(out/'environment.json',dict(python=sys.version,numpy=np.__version__,platform=platform.platform(),stage='CPU data-only, no CUDA model loaded',started_utc=datetime.now(timezone.utc).isoformat()))
    d=json.loads((PREFLIGHT/f'data_{seq}.json').read_text());parent=Path(d['parent'])/'pipeline'
    for p,h in d['parent_files'].items():assert sha(p)==h
    assert sha(parent/'odom_poses.txt')==d['odometry_sha256']
    odom=np.array(metrics.load_kitti_format(str(parent/'odom_poses.txt')));gt=np.array(metrics.gt_from_summary(str(parent)))
    kfs=d['keyframes'];ts=d['timestamps'];ids=d['source_ids']
    assert ids==list(range(2761 if seq=='05' else 1101))
    cumulative=np.r_[0,np.cumsum(np.linalg.norm(np.diff(odom[:,:3,3],axis=0),axis=1))]
    geom=RAPBackend.__new__(RAPBackend);geom.voxel_size=.4;geom.remove_outliers=True;geom.max_points_per_scan=4000;geom.seed=42;geom.feature_extractor=None
    desc={};clouds={};input_hashes={}
    for n,i in enumerate(kfs):
        row=d['scans'][i];p=Path(row['path']);raw=p.read_bytes();digest=hashlib.sha256(raw).hexdigest();assert digest==row['sha256']
        input_hashes[str(p)]=digest
        xyz=np.frombuffer(raw,dtype=np.float32).reshape(-1,4)[:,:3];xyz=xyz[np.linalg.norm(xyz,axis=1)>=1.5]
        sc=scan_context.make_scan_context(xyz,20,60,80)
        desc[i]=dict(sc=sc,ring_key=scan_context.make_ring_key(sc));clouds[i]=geom.preprocess(xyz)[0]
        if n%75==0:print(seq,'descriptors/preprocessing',n,'/',len(kfs),flush=True)
    np.savez_compressed(out/'descriptors.npz',source_ids=kfs,sc=np.stack([desc[i]['sc'] for i in kfs]))
    # Persist the exact label geometry to support independent overlap validation.
    np.savez_compressed(out/'label_geometry.npz',**{str(i):p for i,p in clouds.items()})
    registry=[];queries=[]
    for j in kfs:
        past=[i for i in kfs if j-i>=200]
        ring=sorted([(i,float(np.linalg.norm(desc[i]['ring_key']-desc[j]['ring_key']))) for i in past],key=lambda v:(v[1],v[0]))
        near=sorted([(i,float(np.linalg.norm(odom[i,:3,3]-odom[j,:3,3]))) for i in past],key=lambda v:(v[1],v[0]))
        short=ring[:20]
        dist,shifts=scan_context.sc_distances_batch(np.stack([desc[i]['sc'] for i,_ in short]),desc[j]['sc']) if short else ([],[])
        scored=sorted([dict(source_i=i,ring_rank=rank+1,ring_distance=rd,sc_distance=float(dd),shift=int(ss)) for rank,((i,rd),dd,ss) in enumerate(zip(short,dist,shifts))],key=lambda v:(v['sc_distance'],v['source_i']))
        for rank,r in enumerate(scored):r['sc_rank']=rank+1
        accepted=[r for r in scored if r['sc_distance']<=.25][:20]
        prox=[(i,x) for i,x in near if x<=30][:5]
        close=[r['source_i'] for r in accepted if r['sc_distance']<=accepted[0]['sc_distance']+.02] if accepted else []
        ambiguous=len(close)>=2 and max(close)-min(close)>=200
        queries.append(dict(source_j=j,all_ring_ranked=ring,all_proximity_ranked=near,sc_shortlist=scored,sc_retained=[r['source_i'] for r in accepted],proximity_retained=[i for i,_ in prox],ambiguous_competitor_ids=close,ambiguous=ambiguous))
        scmap={r['source_i']:r for r in accepted};pmap={i:(rank+1,x) for rank,(i,x) in enumerate(prox)}
        for i in sorted(set(scmap)|set(pmap)):
            a=support(i,j,kfs,cumulative,ts,odom);b=support(j,i,kfs,cumulative,ts,odom)
            registry.append(dict(source_i=i,source_j=j,sources=(['sc'] if i in scmap else [])+(['proximity'] if i in pmap else []),
                sc=scmap.get(i),proximity=None if i not in pmap else dict(rank=pmap[i][0],distance_m=pmap[i][1]),
                support_i=a,support_j=b,supported=len(a)==2 and len(b)==2,
                sampling_hash=hashlib.sha256(f'HL-V1|kitti|{seq}|{i}|{j}'.encode()).hexdigest()))
    save(out/'query_population_evaluator.json',queries)
    save(out/'candidate_registry.json',registry)
    qmap={q['source_j']:q for q in queries};labels=[]
    for n,r in enumerate(registry):
        i,j=r['source_i'],r['source_j'];T=np.linalg.inv(gt[i])@gt[j];o,a,b=coverage(clouds[i],clouds[j],T);distance=float(np.linalg.norm(T[:3,3]))
        kind=stratum('sc' in r['sources'],distance,o,qmap[j]['ambiguous'])
        labels.append(dict(r,GT_overlap=o,GT_coverage_i=a,GT_coverage_j=b,GT_distance_m=distance,stratum=kind))
        if n%500==0:print(seq,'GT strata',n,'/',len(registry),flush=True)
    selected=sample(labels)
    save(out/'evaluator_sampling.json',labels)
    save(out/'selected_evaluator.json',selected)
    # Generation inputs contain no GT overlap, strata, ambiguity proxy or labels.
    fields=['source_i','source_j','support_i','support_j']
    save(out/'selected_inputs.json',[{k:r[k] for k in fields} for r in selected])
    rows=[]
    for k in STRATA:
        group=[r for r in labels if r['stratum']==k]
        rows.append(dict(stratum=k,population=len(group),supported=sum(r['supported'] for r in group),selected=sum(r['sampling_reason']=='selected' for r in group),reasons=dict(Counter(r['sampling_reason'] for r in group))))
    adequate=len(selected)>=60 and all(r['selected']>=10 for r in rows)
    summary=dict(sequence=seq,candidates=len(registry),sc=sum('sc' in r['sources'] for r in registry),proximity=sum('proximity' in r['sources'] for r in registry),supported_candidates=sum(r['supported'] for r in registry),selected=len(selected),strata=rows,exclusions=dict(Counter(r['sampling_reason'] for r in labels)),
        adequate=adequate,decision='READY_FOR_INFERENCE' if adequate else 'INCONCLUSIVE',H1='NOT EVALUATED',H2='NOT EVALUATED',generations=0,elapsed_wall_s=time.perf_counter()-start,gpu_inference_s=0.,completed_utc=datetime.now(timezone.utc).isoformat())
    save(out/'adequacy.json',summary)
    save(out/'input_hashes.json',input_hashes)
    sourcepaths=[Path(__file__).resolve(),ROOT/'scripts/hl0_preflight.py',ROOT/'rap_slam/scan_context.py',ROOT/'rap_slam/rap_backend.py',ROOT/'scripts/eval_paper_metrics.py',ROOT/'rap_slam/real_dataset.py',ROOT/'rap_slam/trajectory_protocol.py']
    save(out/'provenance.json',dict(preflight_data_sha256=sha(PREFLIGHT/f'data_{seq}.json'),source_sha256={str(p):sha(p) for p in sourcepaths},parent=d['parent'],parent_files=d['parent_files'],checkpoint_identity=json.loads((parent/'inference_provenance.json').read_text()),note='Checkpoints not loaded in data-only adequacy audit'))
    for p,h in input_hashes.items():assert sha(p)==h
    lines=[f'# KITTI {seq} HL1 data adequacy report','',f"Decision: **{summary['decision']}**. Selected {len(selected)} pairs; requires >=60 total and >=10 in each stratum.",'','| Stratum | Population | Supported | Selected |','|---|---:|---:|---:|']
    lines += [f"| {r['stratum']} | {r['population']} | {r['supported']} | {r['selected']} |" for r in rows]
    lines += ['',f"Full retained candidate union: {len(registry)}. Supported before strata/caps: {summary['supported_candidates']}.",f"Exclusion reasons: {summary['exclusions']}.",'','Candidate construction follows the frozen ring shortlist20, SC<=0.25, proximity30m/top5, separation200 and exact support2m/5m/age5s rules. Same-query cap3 applies across priority-ordered strata. All raw temporal-eligible ring/proximity distances and shortlisted SC distances are persisted. Label geometry uses fixed seed42, voxel0.4, outlier20/2.5, cap4000 and minrange1.5.', '', 'No RAP inference, selector score, oracle gap or training was run. The sample-size gate is checked before inference; inadequate data are not evidence that H1 or H2 is false. No relaxed threshold or synthetic replacement. These are already observed development sequences.', '',f"CPU data audit wall time {summary['elapsed_wall_s']:.3f}s; RAP GPU time 0s. Human labour not independently instrumented.",'','Selected inputs are separated from evaluator-only GT/stratum records. Visual examples are diagnostic and cannot change membership. Independent validation and visual audit are required before removing INCOMPLETE.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sequence',choices=['05','06'],required=True);args=p.parse_args()
    try:run(args.sequence)
    except Exception:
        if PILOT.exists():
            with (PILOT/f'failure_{args.sequence}.txt').open('x') as f:f.write(traceback.format_exc())
        raise