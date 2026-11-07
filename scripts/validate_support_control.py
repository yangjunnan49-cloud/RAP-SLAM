"""Independent SC-V1.1 saved-bank selectors, geometry, metrics and decision checks."""
import json,hashlib,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
MAIN=Path('/mnt/e/githouse/RAP-main');BASE=MAIN/'docs/maintenance/2026-09-14/support-control-02'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def validate(role):
    out=BASE/role;rows=read(out/'evaluator_only_decisions.json');met=read(out/'metrics.json')
    ctr=read(out/'controls.json');g4=read(out/'evaluator_only_G4.json')
    data=MAIN/('docs/maintenance/2026-09-14/hl-v2/data-expansion-01' if role=='discovery' else 'docs/maintenance/2026-09-14/support-control-01/data')
    gt=np.load(data/('00' if role=='discovery' else '08')/'evaluator_gt.npy')
    inputs=read(out/'inputs.json');assert len(inputs)==len(rows)==100
    expected_labels=read(data/f'{role}_evaluator.json')
    assert [(r['sequence'],r['source_i'],r['source_j'],r['stratum']) for r in rows]==[(r['sequence'],r['source_i'],r['source_j'],r['stratum']) for r in expected_labels]
    hashes=read(out/'bank_hashes.json');assert len(hashes)==400
    assert all(sha(out/'bank'/p)==h for p,h in hashes.items())
    checked=0;te_delta=0.;re_delta=0.
    for row in rows:
        seq,i,j=row['sequence'],row['source_i'],row['source_j'];key=f'{seq}_{i}_{j}'
        bank=[read(out/'bank'/f'{key}_{g}.json') for g in range(4)]
        valid=[b for b in bank if b['validity']['valid']];pose={}
        target=np.linalg.inv(gt[i])@gt[j]
        for g,b in enumerate(bank):
            prefix='HL-V2' if role=='discovery' else 'SC-V1'
            expected=int(hashlib.sha256(f'{prefix}|hypothesis|kitti|{seq}|{i}|{j}|{g}'.encode()).hexdigest()[:8],16)%(2**31)
            assert b['generation']==g and b['seed']==expected
            c=ctr[f'{key}_{g}'];assert c['query_points']['P4x2']==c['query_points']['S4']==2*c['query_points']['P4']
            assert c['scores']['P4']==c['scores']['P4x2']
            if not b['validity']['valid']:continue
            T=np.array(b['T']);assert np.max(np.abs(T[3]-[0,0,0,1]))<1e-8
            assert np.linalg.norm(T[:3,:3].T@T[:3,:3]-np.eye(3))<1e-4 and abs(np.linalg.det(T[:3,:3])-1)<1e-4
            te=float(np.linalg.norm(T[:3,3]-target[:3,3]))
            re=float(np.rad2deg(Rotation.from_matrix(target[:3,:3].T@T[:3,:3]).magnitude()))
            pose[g]=(te,re,max(te,re/5))
        picks={'B1':0 if bank[0]['validity']['valid'] else None,'REJ':None,
         'B4':min(valid,key=lambda b:(-b['diagnostics']['overlap_ratio'],b['diagnostics']['mean_dist'],b['generation']))['generation'] if valid else None,
         'R4':min(valid,key=lambda b:(b['rigidity_rmse_m'],b['generation']))['generation'] if valid else None,
         'OR4':min(pose,key=lambda g:(pose[g][2],g)) if pose else None}
        for m in ['A4','S4','P4','P4x2','G4']:
            candidates=[]
            for b in valid:
                g=b['generation'];value=g4[f'{key}_{g}']['score'] if m=='G4' else ctr[f'{key}_{g}']['scores'][m]
                if value is not None:
                    assert np.isfinite(value) and 0<=value<=3
                    candidates.append((value,g))
            picks[m]=min(candidates)[1] if candidates else None
        for m,g in picks.items():
            v=row['methods'][m];assert v['generation']==g;checked+=1
            if g is None:
                assert v['failed'] and not v['accepted'] and not v['strict'] and v['clipped_error']==5
                continue
            te,re,e=pose[g];te_delta=max(te_delta,abs(te-v['RTE_m']));re_delta=max(re_delta,abs(re-v['RRE_deg']))
            assert abs(te-v['RTE_m'])<1e-3 and abs(re-v['RRE_deg'])<1e-3
            assert v['strict']==(te<=1 and re<=5)
            assert abs(v['clipped_error']-min(5,v['error']))<1e-12
            b=bank[g];assert v['accepted']==(b['diagnostics']['overlap_ratio']>=.7 and b['diagnostics']['mean_dist']<=2.5)
            assert v['incorrect_accepted']==(v['accepted'] and not v['strict'])
    for group,table in [(rows,met['overall'])]+[([r for r in rows if r['stratum']==s],t) for s,t in met['by_stratum'].items()]:
        pos=[r for r in group if r['stratum']!='confuser']
        for m,v in table.items():
            vals=[r['methods'][m] for r in group]
            assert v['attempts']==len(group) and v['positives']==len(pos)
            assert v['accepted']==sum(x['accepted'] for x in vals)
            assert v['incorrect_accepted']==sum(x['incorrect_accepted'] for x in vals)
            assert v['failures']==sum(x['failed'] for x in vals)
            assert v['strict_positive_successes']==sum(r['methods'][m]['strict'] for r in pos)
            assert v['correct_positive_accepts']==sum(r['methods'][m]['strict'] and r['methods'][m]['accepted'] for r in pos)
            if pos:
                assert abs(v['mean_clipped_positive_error']-sum(r['methods'][m]['clipped_error'] for r in pos)/len(pos))<1e-12
                assert v['selection_recall']==v['strict_positive_successes']/len(pos)
    err=lambda m:met['overall'][m]['mean_clipped_positive_error']
    gap=err('A4')-err('OR4');rec=(err('A4')-err('S4'))/gap if gap>0 else None
    diff=sum(r['stratum']!='confuser' and abs(r['methods']['A4']['clipped_error']-r['methods']['OR4']['clipped_error'])>=.1 for r in rows)
    assert diff==met['discrepant_positives'] and gap==met['A4_oracle_gap'] and rec==met['gap_recovery']
    loss=met['by_stratum']['ordinary']['A4']['selection_recall']-met['by_stratum']['ordinary']['S4']['selection_recall']
    assert loss==met['ordinary_recall_loss']
    bad={}
    for m in ['A4','R4']:
        bad[m]=[dict(sequence=r['sequence'],source_i=r['source_i'],source_j=r['source_j']) for r in rows if r['methods']['S4']['incorrect_accepted'] and not r['methods'][m]['incorrect_accepted']]
        assert bad[m]==met['new_incorrect_pairs'][m]
    conditions=dict(recovery_25percent=rec is not None and rec>=.25,no_worse_than_R4=err('S4')<=err('R4'),
       ordinary_loss_at_most_2pp=loss<=.02,no_new_incorrect_A4=not bad['A4'],no_new_incorrect_R4=not bad['R4'])
    assert conditions==met['conditions']
    expected='INCONCLUSIVE_HEADROOM' if gap<=0 or diff<10 else 'NO_SUPPORT_SIGNAL_UNDER_FIXED_SCORE' if not all(conditions.values()) else 'SUPPORT_AND_VIEW_AGGREGATION_SIGNAL' if err('S4')<err('P4') and err('S4')<err('P4x2') else 'SUPPORT_BUT_NO_VIEW_AGGREGATION_ADVANTAGE'
    assert met['adequate'] and met['decision']==expected and met['neural_expansion']=='STOPPED'
    if role=='discovery':
        old=MAIN/'docs/maintenance/2026-09-14/hl-v2/pilot-01/discovery'
        assert all(sha(out/'bank'/p)==sha(old/'bank'/p) for p in hashes)
        oldrows=read(old/'evaluator_only_decisions.json');omap={(r['source_i'],r['source_j']):r for r in oldrows}
        for r in rows:
            for m in ['B1','B4','R4','S4']:
                assert r['methods'][m]==omap[(r['source_i'],r['source_j'])]['methods'][m]
    with (out/'independent_validation.json').open('x') as f:
        json.dump(dict(passed=True,checked_selections=checked,checked_banks=len(hashes),
           max_translation_delta=te_delta,max_rotation_delta=re_delta,all_strata_counts=True,
           original_discovery_methods_exact=role=='discovery',decision_recomputed=True,source_sha256=sha(Path(__file__))),f,indent=2)
    print(role,'independently verified',checked,'selections',flush=True)
if __name__=='__main__':validate(sys.argv[1])
