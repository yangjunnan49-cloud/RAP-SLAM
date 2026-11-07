"""Bounded independent checks for HL-V2 data expansion."""
import argparse
from collections import Counter
import hashlib,json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.distance import cdist
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from rap_slam.scan_context import sc_distance
from hl2_data_bridge import OUT,STRATA,pairhash,save

def validate(seq):
    out=OUT/seq
    def read(p):return json.loads((out/p).read_text())
    d=read('data.json');queries=read('queries_evaluator.json');reg=read('candidate_registry.json');labels=read('evaluator_sampling.json');sel=read('selected_evaluator.json')
    odom=np.load(out/'odom.npy');gt=np.load(out/'evaluator_gt.npy');n=len(odom);ts=d['timestamps'];kfs=d['keyframes'];cum=np.r_[0,np.cumsum(np.linalg.norm(np.diff(odom[:,:3,3],axis=0),axis=1))]
    desc=np.load(out/'descriptors.npz');D=dict(zip(desc['ids'],desc['sc']));union=set();maxdelta=0.;boundary=1.;qmap={q['source_j']:q for q in queries}
    for q in queries:
        j=q['source_j'];past={i for i in kfs if j-i>=200};assert past=={i for i,_ in q['all_ring']}=={i for i,_ in q['all_odom']}
        assert {r['source_i'] for r in q['sc_shortlist']}=={i for i,_ in q['all_ring'][:50]}
        assert q['sc_retained']==q['sc_shortlist'][:20]
        for r in q['sc_retained'][:2]:
            actual,_=sc_distance(D[r['source_i']],D[j]);maxdelta=max(maxdelta,abs(actual-r['sc_distance']));assert abs(actual-r['sc_distance'])<1e-6
        assert q['proximity']==[i for i,x in q['all_odom'] if x<=30][:5]
        for band in q['bands']:
            lo,hi=band['low'],band['high'];pool=[i for i,x in q['all_odom'] if x>=lo and (x<hi or hi==80 and x<=hi)]
            assert band['source_i']==min(pool,key=lambda i:pairhash(seq,i,j))
        union.update((i,j) for i in {r['source_i'] for r in q['sc_retained']}|set(q['proximity'])|{r['source_i'] for r in q['bands']})
    assert union=={(r['source_i'],r['source_j']) for r in reg} and len(union)==len(reg)==len(labels)
    # Reconstruct support independently with search on monotone cumulative distance.
    for anchor,rows in read('support_metadata.json').items():
        anchor=int(anchor);expected=[]
        for target in [2.,5.]:
            s=int(np.searchsorted(cum[:anchor],cum[anchor]-target,side='right')-1)
            if s>=0 and ts[anchor]-ts[s]<=5 and s not in expected:expected.append(s)
        assert expected==[s['source_id'] for s in rows]
        for s in rows:assert np.allclose(s['local_pose'],np.linalg.inv(odom[anchor])@odom[s['source_id']],atol=1e-12,rtol=0)
    for r in labels:
        o=r['GT_overlap'];distance=r['GT_distance_m'];j=r['source_j']
        kind='confuser' if 'sc' in r['sources'] and distance>20 and o<.05 else 'ambiguous_positive' if o>=.30 and qmap[j]['ambiguous'] else 'low_overlap' if .10<=o<.30 else 'ordinary' if o>=.50 else None
        assert kind==r['stratum']
    expected=[];counts=Counter()
    for kind in STRATA:
        quota=0
        for r in sorted([r for r in labels if r['stratum']==kind],key=lambda r:r['sampling_hash']):
            assert r['sampling_hash']==pairhash(seq,r['source_i'],r['source_j'])
            if r['supported'] and counts[r['source_j']]<3 and quota<25:expected.append((r['source_i'],r['source_j']));counts[r['source_j']]+=1;quota+=1
    assert expected==[(r['source_i'],r['source_j']) for r in sel]
    assert all(set(r)=={'sequence','source_i','source_j','support_i','support_j'} for r in read('selected_inputs.json'))
    geom=np.load(out/'label_geometry.npz');examples=[]
    for kind in STRATA:
        rows=[r for r in sel if r['stratum']==kind]
        for r in rows[:2]:
            i,j=r['source_i'],r['source_j'];P=geom[str(i)];Q=geom[str(j)];T=np.linalg.inv(gt[i])@gt[j];Q=Q@T[:3,:3].T+T[:3,3]
            a=np.concatenate([cdist(P[k:k+200],Q).min(axis=1)<=1 for k in range(0,len(P),200)]).mean()
            b=np.concatenate([cdist(Q[k:k+200],P).min(axis=1)<=1 for k in range(0,len(Q),200)]).mean();assert abs((a+b)/2-r['GT_overlap'])<1e-12
            examples.append(dict(stratum=kind,source_i=i,source_j=j,overlap=r['GT_overlap']))
    save(out/'validation.json',dict(passed=True,union_exact=True,support_raw_search_exact=True,strata_exact=True,sampling_exact=True,GT_free_inputs=True,sc_max_delta=maxdelta,bruteforce_examples=examples))
    print(seq,'independently validated',len(reg),'pairs',len(sel),'selected',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sequence',required=True);validate(p.parse_args().sequence)