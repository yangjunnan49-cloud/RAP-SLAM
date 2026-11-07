"""Independent validation of HL1 data registry/sampling and diagnostic plots."""
import argparse
from collections import Counter
import json
import hashlib
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from rap_slam.scan_context import sc_distance
import eval_paper_metrics as metrics
MAIN=Path('/mnt/e/githouse/RAP-main');PILOT=MAIN/'docs/maintenance/2026-09-14/hl-v1/pilot-01'
KINDS=['confuser','ambiguous_positive','low_overlap','ordinary']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def validate(seq):
    out=PILOT/seq
    def read(n):return json.loads((out/n).read_text())
    reg=read('candidate_registry.json');labels=read('evaluator_sampling.json');sel=read('selected_evaluator.json');inputs=read('selected_inputs.json');queries=read('query_population_evaluator.json')
    assert len(reg)==len(labels) and len({(r['source_i'],r['source_j']) for r in reg})==len(reg)
    d=json.loads((MAIN/f'docs/maintenance/2026-09-14/hl-v1/preflight-01/data_{seq}.json').read_text())
    parent=Path(d['parent'])/'pipeline';odom=np.array(metrics.load_kitti_format(str(parent/'odom_poses.txt')));gt=np.array(metrics.gt_from_summary(str(parent)))
    travel=np.r_[0,np.cumsum(np.linalg.norm(np.diff(odom[:,:3,3],axis=0),axis=1))];kfs=d['keyframes'];ts=d['timestamps']
    desc=np.load(out/'descriptors.npz');D=dict(zip(desc['source_ids'],desc['sc']));union=set();scmax=0.
    for q in queries:
        j=q['source_j'];expected={i for i in kfs if j-i>=200}
        assert {i for i,_ in q['all_ring_ranked']}==expected and {i for i,_ in q['all_proximity_ranked']}==expected
        assert q['all_ring_ranked']==sorted(q['all_ring_ranked'],key=lambda x:(x[1],x[0]))
        assert q['all_proximity_ranked']==sorted(q['all_proximity_ranked'],key=lambda x:(x[1],x[0]))
        assert {r['source_i'] for r in q['sc_shortlist']}=={i for i,_ in q['all_ring_ranked'][:20]}
        for r in q['sc_shortlist']:
            actual,shift=sc_distance(D[r['source_i']],D[j]);delta=abs(actual-r['sc_distance']);scmax=max(scmax,delta)
            assert delta<1e-6
        expected_sc=[r['source_i'] for r in q['sc_shortlist'] if r['sc_distance']<=.25][:20]
        expected_prox=[i for i,x in q['all_proximity_ranked'] if x<=30][:5]
        assert expected_sc==q['sc_retained'] and expected_prox==q['proximity_retained']
        union.update((i,j) for i in set(expected_sc)|set(expected_prox))
    assert union=={(r['source_i'],r['source_j']) for r in reg}
    for r in reg:
        i,j=r['source_i'],r['source_j'];assert j-i>=200
        for anchor,other,key in [(i,j,'support_i'),(j,i,'support_j')]:
            expected=[]
            for target in [2.,5.]:
                for s in reversed(kfs):
                    if s>=anchor or s==other:continue
                    if travel[anchor]-travel[s]>=target:
                        if ts[anchor]-ts[s]<=5 and s not in expected:expected.append(s)
                        break
            assert expected==[s['source_id'] for s in r[key]]
            for s in r[key]:assert np.allclose(s['local_pose'],np.linalg.inv(odom[anchor])@odom[s['source_id']],rtol=0,atol=1e-12)
    # Independent priority/quota/cap replay with no generated outcomes.
    expected=[];counts=Counter()
    for k in KINDS:
        quota=0
        for r in sorted([r for r in labels if r['stratum']==k],key=lambda r:r['sampling_hash']):
            i,j=r['source_i'],r['source_j'];assert r['sampling_hash']==hashlib.sha256(f'HL-V1|kitti|{seq}|{i}|{j}'.encode()).hexdigest()
            if r['supported'] and quota<25 and counts[j]<3:expected.append((i,j));quota+=1;counts[j]+=1
    assert expected==[(r['source_i'],r['source_j']) for r in sel]
    assert all(set(r)=={'source_i','source_j','support_i','support_j'} for r in inputs)
    assert [(r['source_i'],r['source_j']) for r in inputs]==expected
    geometry=np.load(out/'label_geometry.npz');examples=[]
    for k in KINDS:
        pool=[r for r in sel if r['stratum']==k] or [r for r in labels if r['stratum']==k]
        if pool:examples.append(sorted(pool,key=lambda r:r['sampling_hash'])[0])
    fig,axes=plt.subplots(2,2,figsize=(12,10));audit=[]
    for ax,k in zip(axes.flat,KINDS):
        pool=[r for r in examples if r['stratum']==k]
        if not pool:ax.text(.5,.5,'No population example',ha='center');ax.set_title(k);continue
        r=pool[0];i,j=r['source_i'],r['source_j'];P=geometry[str(i)];Q=geometry[str(j)];T=np.linalg.inv(gt[i])@gt[j];Q=Q@T[:3,:3].T+T[:3,3]
        # Exact independent chunked all-distance overlap, no KD-tree reuse.
        nearP=np.concatenate([cdist(P[a:a+200],Q).min(axis=1)<=1 for a in range(0,len(P),200)])
        nearQ=np.concatenate([cdist(Q[a:a+200],P).min(axis=1)<=1 for a in range(0,len(Q),200)])
        o=float((nearP.mean()+nearQ.mean())/2);assert abs(o-r['GT_overlap'])<1e-12
        ax.scatter(P[:,0],P[:,1],s=.35,c='#1764aa',alpha=.5,label=f'i={i}')
        ax.scatter(Q[:,0],Q[:,1],s=.35,c='#d65f19',alpha=.5,label=f'j={j} aligned by GT')
        ax.set_aspect('equal');ax.set_xlabel('Anchor i x (m)');ax.set_ylabel('Anchor i y (m)');ax.legend(markerscale=6,fontsize=8)
        ax.set_title(f"{k}: O_GT={o:.3f}\n{r['sampling_reason']}; support={r['supported']}")
        audit.append(dict(stratum=k,source_i=i,source_j=j,GT_overlap=o,selected=r['sampling_reason']=='selected',visual_disagreement='pending human/agent image inspection',membership_changed=False))
    fig.suptitle(f'KITTI {seq}: fixed-label examples, GT-aligned XY projection; no outcome filtering')
    fig.tight_layout();fig.savefig(out/'strata_examples.png',dpi=150);plt.close(fig)
    save(out/'visual_examples.json',audit)
    save(out/'independent_validation.json',dict(candidate_union_exact=True,support_reconstructed=True,sampling_reconstructed=True,no_gt_fields_in_generation_inputs=True,sc_scalar_max_delta=scmax,bruteforce_overlap_examples=len(audit),selected=len(sel)))
    print(seq,'validated',len(reg),'candidates',len(sel),'selected; SC delta',scmax,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sequence',choices=['05','06'],required=True);validate(p.parse_args().sequence)