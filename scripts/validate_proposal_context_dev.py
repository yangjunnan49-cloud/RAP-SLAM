"""Independent PG-DEV-V1 selector, geometry, gate and decision audit."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rap_slam.rap_backend import RAPBackend
MAIN=Path('/mnt/e/githouse/RAP-main')
BASE=MAIN/'docs/maintenance/2026-09-15/proposal-context-dev-01'
SC=MAIN/'docs/maintenance/2026-09-14/support-control-02'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def validate(seq):
    out=BASE/seq;role='discovery' if seq=='00' else 'confirmation';parent=SC/role
    data=MAIN/('docs/maintenance/2026-09-14/hl-v2/data-expansion-01' if seq=='00' else 'docs/maintenance/2026-09-14/support-control-01/data')
    inputs=read(out/'inputs.json');assert inputs==read(data/f'{role}_inputs.json')
    labels=read(data/f'{role}_evaluator.json');rows=read(out/'evaluator_only_decisions.json')
    gt=np.load(data/seq/'evaluator_gt.npy');meta=read(data/seq/'data.json');ctr=read(parent/'controls.json')
    met=read(out/'metrics.json');gates=read(out/'gates.json');counts={};checks=0;delta=0.;gdelta=0.
    geom=RAPBackend.__new__(RAPBackend)
    geom.voxel_size=.4;geom.remove_outliers=True;geom.max_points_per_scan=4000;geom.seed=42;geom.feature_extractor=None
    cache={}
    def points(i):
        if i not in cache:
            m=meta['scans'][i];p=np.fromfile(m['path'],dtype=np.float32).reshape(-1,4)[:,:3]
            cache[i]=geom.preprocess(p[np.linalg.norm(p,axis=1)>=1.5])[0]
        return cache[i]
    assert len(rows)==len(labels)==len(inputs)==100
    for row,label,r in zip(rows,labels,inputs):
        assert all(row[k]==label[k]==r[k] for k in ['sequence','source_i','source_j'])
        assert row['stratum']==label['stratum'];counts[row['stratum']]=counts.get(row['stratum'],0)+1
        i,j=r['source_i'],r['source_j'];key=f'{seq}_{i}_{j}';target=np.linalg.solve(gt[i],gt[j])
        x,y=points(i),points(j)
        for arm,root in [('A',parent),('P',out)]:
            bank=[read(root/'bank'/f'{key}_{k}.json') for k in range(4)]
            valid=[];pose={};accept={}
            for k,b in enumerate(bank):
                prefix='HL-V2' if seq=='00' else 'SC-V1'
                seed=int(hashlib.sha256(f'{prefix}|hypothesis|kitti|{seq}|{i}|{j}|{k}'.encode()).hexdigest()[:8],16)%(2**31)
                assert (b['seed'],b['generation'],b['source_i'],b['source_j'],b['sequence'])==(seed,k,i,j,seq)
                if not b['validity']['valid']:continue
                T=np.asarray(b['T']);R=T[:3,:3]
                assert np.linalg.norm(R.T@R-np.eye(3))<1e-4 and abs(np.linalg.det(R)-1)<1e-4
                assert np.max(np.abs(T[3]-[0,0,0,1]))<1e-8
                assert 200<=b['diagnostics']['n_a']<=4000 and 200<=b['diagnostics']['n_b']<=4000
                te=float(np.linalg.norm(T[:3,3]-target[:3,3]))
                re=float(np.degrees(Rotation.from_matrix(target[:3,:3].T@R).magnitude()))
                pose[k]=(te,re,max(te,re/5));valid.append(k)
                yy=(np.c_[y,np.ones(len(y))]@T.T)[:,:3]
                d=cKDTree(x).query(yy)[0];e=cKDTree(yy).query(x)[0]
                ov=float(((d<1).mean()+(e<1).mean())/2);md=float((d.mean()+e.mean())/2)
                saved=gates[key][arm][k];gdelta=max(gdelta,abs(ov-saved['overlap_ratio']),abs(md-saved['mean_dist']))
                assert abs(ov-saved['overlap_ratio'])<1e-12 and abs(md-saved['mean_dist'])<1e-10
                accept[k]=ov>=.7 and md<=2.5;assert saved['accepted']==accept[k]
            picks={'1':0 if 0 in valid else None,
                   'R4':min(valid,key=lambda k:(bank[k]['rigidity_rmse_m'],k)) if valid else None,
                   'OR4':min(valid,key=lambda k:(pose[k][2],k)) if valid else None}
            if arm=='A':picks['S4']=min(valid,key=lambda k:(ctr[f'{key}_{k}']['scores']['S4'],k)) if valid else None
            for kind,k in picks.items():
                m=arm+'1' if kind=='1' else arm+'-'+kind;v=row['methods'][m];checks+=1
                assert v['generation']==k
                if k is None:
                    assert v['failed'] and not v['strict'] and not v['accepted'] and v['clipped_error']==5;continue
                te,re,error=pose[k];delta=max(delta,abs(te-v['RTE_m']),abs(re-v['RRE_deg']))
                assert abs(te-v['RTE_m'])<1e-3 and abs(re-v['RRE_deg'])<1e-3
                assert v['strict']==(te<=1 and re<=5) and v['legacy_safe']==(te<=5 and re<=10)
                assert v['accepted']==accept[k] and v['incorrect_accepted']==(accept[k] and not v['strict'])
                assert abs(v['clipped_error']-min(5,v['error']))<1e-12
    assert len(counts)==4 and set(counts.values())=={25}
    for subset,table in [(rows,met['overall'])]+[([r for r in rows if r['stratum']==s],t) for s,t in met['by_stratum'].items()]:
        pos=[r for r in subset if r['stratum']!='confuser']
        for m,v in table.items():
            assert v['attempts']==len(subset) and v['positives']==len(pos)
            for field in ['accepted','incorrect_accepted','failed']:
                assert v['failures' if field=='failed' else field]==sum(r['methods'][m][field] for r in subset)
            strict=sum(r['methods'][m]['strict'] for r in pos)
            assert v['strict_positive_successes']==strict
            assert v['correct_positive_accepts']==sum(r['methods'][m]['strict'] and r['methods'][m]['accepted'] for r in pos)
            if pos:
                assert v['selection_recall']==strict/len(pos)
                assert abs(v['mean_clipped_positive_error']-np.mean([r['methods'][m]['clipped_error'] for r in pos]))<1e-12
    recall=lambda m:sum(r['methods'][m]['strict'] for r in rows if r['stratum']!='confuser')/75
    cond=dict(oracle_gain_5pp=recall('P-OR4')-recall('A-OR4')>=.05,
        executable_gain_3pp=recall('P-R4')-recall('A-R4')>=.03,no_worse_than_S4=recall('P-R4')>=recall('A-S4'))
    for m in ['A-R4','A-S4']:
        loss=sum(int(r['methods'][m]['strict'])-int(r['methods']['P-R4']['strict']) for r in rows if r['stratum']=='ordinary')/25
        cond['ordinary_loss_'+m]=loss<=.02
        bad=[f"{r['source_i']}_{r['source_j']}" for r in rows if r['methods']['P-R4']['incorrect_accepted'] and not r['methods'][m]['incorrect_accepted']]
        assert bad==met['new_incorrect_pairs'][m];cond['no_new_incorrect_'+m]=not bad
    timing=read(out/'timings.json');sample=timing['pairs'][:10]
    assert all(p['timing_sample']==(k<10) for k,p in enumerate(timing['pairs']))
    ratio=sum(p['P_R4_full_s'] for p in sample)/sum(p['A_S4_full_s'] for p in sample)
    assert ratio==met['sampled_cost']['ratio_P_to_A_S4'];cond['sampled_cost']=ratio<=1.25
    assert cond==met['conditions']
    assert met['decision']==('DEVELOPMENT_SIGNAL' if all(cond.values()) else 'NO_DEVELOPMENT_CONTEXT_SIGNAL')
    for root,manifest in [(out/'bank',out/'bank_hashes.json'),(parent/'bank',parent/'bank_hashes.json')]:
        hashes=read(manifest);assert len(hashes)==400
        for p,h in hashes.items():assert sha(root/p)==h
    for p,h in read(out/'input_hashes.json').items():assert sha(Path(p))==h
    for p,h in read(BASE/'execution_freeze.json')['hashes'].items():assert sha(Path(p))==h
    save(out/'independent_validation.json',dict(passed=True,selections=checks,bank_files=800,gate_checks=800,
        max_pose_metric_delta=delta,max_gate_delta=gdelta,strata_counts=counts,decision_recomputed=True,
        source_sha256=sha(Path(__file__))))
    (out/'INCOMPLETE').unlink()
    if seq=='00':save(BASE/'00_frozen.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print(seq,'validated',checks,'selections;800 gates',flush=True)
if __name__=='__main__':validate(sys.argv[1])
