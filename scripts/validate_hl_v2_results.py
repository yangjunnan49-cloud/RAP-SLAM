"""Independent selector/label/gate validation for HL-V2 saved evidence."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from evaluate_hl_v2 import selectors
from hl_v2_bank import BASE,DATA,save,sha

def synthetic():
    def row(g,valid=True,overlap=.8,distance=1.,support=.2):return dict(generation=g,validity={'valid':valid},diagnostics={'overlap_ratio':overlap,'mean_dist':distance},support=None if support is None else {'score':support},rigidity_rmse_m=.1)
    bank=[row(0,False),row(1,overlap=.8,distance=2),row(2,overlap=.8,distance=1,support=None),row(3,overlap=.7,support=.1)]
    v=selectors(bank);assert v['B1'] is None and v['B4']['generation']==2 and v['S4']['generation']==3 and v['R4']['generation']==1 and v['REJ'] is None
    assert all(v is None for v in selectors([row(g,False) for g in range(4)]).values())
    v=selectors([row(g) for g in range(4)]);assert all(r is None or r['generation']==0 for r in v.values())
    return True

def validate(role):
    assert synthetic();out=BASE/'pilot-01'/role
    decisions=json.loads((out/'evaluator_only_decisions.json').read_text());oracle=json.loads((out/'oracle_only.json').read_text());metrics=json.loads((out/'metrics.json').read_text())
    gts={s:np.load(DATA/s/'evaluator_gt.npy') for s in {r['sequence'] for r in decisions}}
    bankhash=json.loads((out/'bank_hashes.json').read_text());assert all(sha(out/'bank'/p)==h for p,h in bankhash.items())
    checked=0;maxte=0.;maxre=0.;joined=[]
    for d,o in zip(decisions,oracle):
        seq,i,j=d['sequence'],d['source_i'],d['source_j'];target=np.linalg.inv(gts[seq][i])@gts[seq][j];bank=[json.loads((out/'bank'/f'{seq}_{i}_{j}_{g}.json').read_text()) for g in range(4)];valid=[r for r in bank if r['validity']['valid']]
        values=dict(d['methods'],OR4=o['OR4']);independent={}
        for r in valid:
            T=np.array(r['T']);te=float(np.linalg.norm(T[:3,3]-target[:3,3]));re=float(np.rad2deg(Rotation.from_matrix(target[:3,:3].T@T[:3,:3]).magnitude()))
            independent[r['generation']]=(te,re,max(te,re/5))
        picks=dict(B1=0 if bank[0]['validity']['valid'] else None,B4=sorted(valid,key=lambda r:(-r['diagnostics']['overlap_ratio'],r['diagnostics']['mean_dist'],r['generation']))[0]['generation'] if valid else None,
            R4=sorted(valid,key=lambda r:(r['rigidity_rmse_m'],r['generation']))[0]['generation'] if valid else None,REJ=None)
        supported=[r for r in valid if r.get('support') is not None];picks['S4']=sorted(supported,key=lambda r:(r['support']['score'],r['generation']))[0]['generation'] if supported else None
        picks['OR4']=min(independent,key=lambda g:(independent[g][2],g)) if independent else None
        for m,g in picks.items():
            v=values[m];assert v['generation']==g
            if g is None:assert v['failed'] and not v['accepted'] and v['clipped_error']==5;continue
            te,re,e=independent[g];maxte=max(maxte,abs(te-v['RTE_m']));maxre=max(maxre,abs(re-v['RRE_deg']));assert abs(te-v['RTE_m'])<1e-3 and abs(re-v['RRE_deg'])<1e-3
            assert bool(te<=1 and re<=5)==v['strict']
            r=bank[g];assert v['accepted']==(r['diagnostics']['overlap_ratio']>=.7 and r['diagnostics']['mean_dist']<=2.5)
            checked+=1
        joined.append(dict(d,methods=values))
    for m in metrics['overall']:
        v=metrics['overall'][m];positive=[r for r in joined if r['stratum']!='confuser'];vals=[r['methods'][m] for r in joined]
        assert v['attempts']==len(joined) and v['positives']==len(positive)
        assert v['strict_positive_successes']==sum(r['methods'][m]['strict'] for r in positive)
        assert v['accepted']==sum(r['accepted'] for r in vals)
        assert v['incorrect_accepted']==sum(r['accepted'] and not r['strict'] for r in vals)
        assert abs(v['mean_clipped_positive_error']-np.mean([r['methods'][m]['clipped_error'] for r in positive]))<1e-12
    b,o,s=[metrics['overall'][m] for m in ['B4','OR4','S4']]
    rg=o['selection_recall']-b['selection_recall'];eg=(b['mean_clipped_positive_error']-o['mean_clipped_positive_error'])/b['mean_clipped_positive_error'] if b['mean_clipped_positive_error'] else 0
    assert metrics['H1']==(rg>=.05 or eg>=.15)
    new=[r for r in joined if r['methods']['S4']['incorrect_accepted'] and not r['methods']['B4']['incorrect_accepted']]
    assert len(new)==len(metrics['new_incorrect_pairs'])
    assert metrics['H2']==bool(metrics['H1'] and metrics['H2_gap_recovery'] is not None and metrics['H2_gap_recovery']>=.25 and metrics['ordinary_recall_loss']<=.02 and not new)
    save(out/'independent_validation.json',dict(passed=True,synthetic_failure_tie_cases=True,checked_selections=checked,all_bank_hashes=True,independent_pose_error_max_translation_delta=maxte,independent_pose_error_max_rotation_delta=maxre,denominators_and_gate_recomputed=True))
    print(role,'independent metric/selector validation passed',checked,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--role',choices=['discovery','confirmation']);p.add_argument('--synthetic-only',action='store_true');a=p.parse_args()
    if a.synthetic_only:print('Synthetic selector failure/tie checks',synthetic())
    else:validate(a.role)