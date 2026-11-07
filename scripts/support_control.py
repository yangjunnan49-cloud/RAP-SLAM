"""SC-V1.1 frozen-bank observation/score controls; no training or production edits."""
import argparse, hashlib, json, sys, time, shutil
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from hl0_preflight import score, move, request_rng
from hl_v2_bank import setup_backend
from rap_slam.rap_backend import RAPBackend
from rap_slam.measurement import transform_label, rotation_validity
MAIN=Path('/mnt/e/githouse/RAP-main')
BASE=MAIN/'docs/maintenance/2026-09-14/support-control-02'
OLD=MAIN/'docs/maintenance/2026-09-14/hl-v2'
SC0=MAIN/'docs/maintenance/2026-09-14/support-control-01/data'
METHODS=['B1','B4','R4','A4','S4','P4','P4x2','G4','OR4','REJ']
STRATA=['confuser','ambiguous_positive','low_overlap','ordinary']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def seed(seq,i,j,g):return int(hashlib.sha256(f'SC-V1|hypothesis|kitti|{seq}|{i}|{j}|{g}'.encode()).hexdigest()[:8],16)%(2**31)
def pooled(T,left,right):
    if any(not len(p) for _,p in left+right):return None
    x=np.concatenate([move(a,p) for a,p in left])
    y=np.concatenate([move(T@b,q) for b,q in right])
    d=cKDTree(y).query(x,workers=1)[0];e=cKDTree(x).query(y,workers=1)[0]
    if not np.isfinite(d).all() or not np.isfinite(e).all():return None
    return float((np.minimum(d,3).mean()+np.minimum(e,3).mean())/2)
def scalar(T,left,right):
    v=score(T,left,right)
    return None if v is None else v['score']
def pick(bank,field):
    good=[b for b in bank if b['validity']['valid'] and b['scores'].get(field) is not None]
    return min(good,key=lambda b:(b['scores'][field],b['generation'])) if good else None
def synthetic():
    I=np.eye(4);T=I.copy();T[:3,3]=[2,-1,.5]
    p=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,2]])
    q=move(np.linalg.inv(T),p);left=[(I,p),(I,p)];right=[(I,q),(I,q)]
    assert scalar(T,left,right)<1e-12 and pooled(T,left,right)<1e-12
    A=I.copy();A[:3,3]=[.3,.7,-.2]
    moved=[(A,move(np.linalg.inv(A),p))]*2
    assert scalar(T,moved,right)<1e-12
    wrong=I.copy();wrong[:3,3]=[.4,-.1,.3]
    x=p;y=move(wrong,q);d=cdist(x,y)
    ref=float((np.minimum(d.min(1),3).mean()+np.minimum(d.min(0),3).mean())/2)
    assert abs(scalar(wrong,left,right)-ref)<1e-12
    assert abs(pooled(wrong,left,right)-ref)<1e-12
    assert pooled(I,[(I,np.empty((0,3)))],right) is None
    assert scalar(I,[(I,np.empty((0,3)))]*2,right) is None
    tie=[dict(generation=g,validity={'valid':True},scores={'A4':1.}) for g in [3,1,0,2]]
    assert pick(tie,'A4')['generation']==0 and pick([],'A4') is None
    save(BASE/'synthetic_validation.json',dict(passed=True,transport=True,
         duplicate_anchor=True,pooled_bruteforce=True,empty=True,tie=True))
def aggregate(rows,m):
    v=[r['methods'][m] for r in rows];pos=[r for r in rows if r['stratum']!='confuser']
    conf=[r for r in rows if r['stratum']=='confuser']
    n=len(pos);acc=sum(x['accepted'] for x in v);bad=sum(x['incorrect_accepted'] for x in v)
    success=sum(r['methods'][m]['strict'] for r in pos)
    correct=sum(r['methods'][m]['accepted'] and r['methods'][m]['strict'] for r in pos)
    return dict(attempts=len(rows),positives=n,strict_positive_successes=success,
      selection_recall=success/n if n else None,correct_positive_accepts=correct,
      correct_accept_recall=correct/n if n else None,accepted=acc,incorrect_accepted=bad,
      incorrect_per_accepted=bad/acc if acc else None,incorrect_per_attempt=bad/len(rows) if rows else None,
      confusers=len(conf),accepted_confusers=sum(r['methods'][m]['accepted'] for r in conf),
      confuser_accept_rate=sum(r['methods'][m]['accepted'] for r in conf)/len(conf) if conf else None,
      mean_clipped_positive_error=float(np.mean([r['methods'][m]['clipped_error'] for r in pos])) if n else None,
      failures=sum(x['failed'] for x in v),legacy_safe=sum(x['legacy_safe'] for x in v),
      RTE_p90=float(np.percentile([x['RTE_m'] for x in v if x['RTE_m'] is not None],90)) if any(x['RTE_m'] is not None for x in v) else None,
      RRE_p90=float(np.percentile([x['RRE_deg'] for x in v if x['RRE_deg'] is not None],90)) if any(x['RRE_deg'] is not None for x in v) else None)
def evaluate(role):
    out=BASE/role;data=OLD/'data-expansion-01' if role=='discovery' else SC0
    labels=read(data/f'{role}_evaluator.json');gt=np.load(data/('00' if role=='discovery' else '08')/'evaluator_gt.npy')
    controls=read(out/'controls.json');g4=read(out/'evaluator_only_G4.json');joined=[];labs_all=[]
    for r in labels:
        seq,i,j=r['sequence'],r['source_i'],r['source_j'];key=f'{seq}_{i}_{j}'
        bank=[read(out/'bank'/f'{key}_{g}.json') for g in range(4)]
        good=[b for b in bank if b['validity']['valid']];Tgt=np.linalg.inv(gt[i])@gt[j];labs={}
        for b in good:
            g=b['generation'];lab=transform_label(b['T'],Tgt);rte=lab['gt_translation_error_m'];rre=lab['gt_rotation_error_deg']
            labs[g]=dict(RTE_m=rte,RRE_deg=rre,error=max(rte,rre/5),clipped_error=min(5,max(rte,rre/5)),
                         strict=rte<=1 and rre<=5,legacy_safe=lab['label']=='safe')
            labs_all.append(dict(pair=key,generation=g,**labs[g]))
        for b in bank:b['scores']=dict(controls[f'{key}_{b["generation"]}']['scores'],G4=g4[f'{key}_{b["generation"]}']['score'])
        choices=dict(B1=bank[0] if bank[0]['validity']['valid'] else None,
             B4=min(good,key=lambda b:(-b['diagnostics']['overlap_ratio'],b['diagnostics']['mean_dist'],b['generation'])) if good else None,
             R4=min(good,key=lambda b:(b['rigidity_rmse_m'],b['generation'])) if good else None,
             REJ=None,OR4=min(good,key=lambda b:(labs[b['generation']]['error'],b['generation'])) if good else None)
        choices.update({m:pick(bank,m) for m in ['A4','S4','P4','P4x2','G4']})
        values={}
        for m in METHODS:
            b=choices[m]
            lab=labs[b['generation']] if b is not None else dict(RTE_m=None,RRE_deg=None,error=5.,clipped_error=5.,strict=False,legacy_safe=False)
            accepted=b is not None and b['diagnostics']['overlap_ratio']>=.7 and b['diagnostics']['mean_dist']<=2.5
            values[m]=dict(generation=b['generation'] if b is not None else None,accepted=accepted,
                 incorrect_accepted=accepted and not lab['strict'],failed=b is None,**lab)
        joined.append(dict(sequence=seq,source_i=i,source_j=j,stratum=r['stratum'],methods=values))
    allm={m:aggregate(joined,m) for m in METHODS}
    strata={s:{m:aggregate([r for r in joined if r['stratum']==s],m) for m in METHODS} for s in STRATA}
    gap=allm['A4']['mean_clipped_positive_error']-allm['OR4']['mean_clipped_positive_error']
    diff=sum(r['stratum']!='confuser' and abs(r['methods']['A4']['clipped_error']-r['methods']['OR4']['clipped_error'])>=.10 for r in joined)
    recovery=(allm['A4']['mean_clipped_positive_error']-allm['S4']['mean_clipped_positive_error'])/gap if gap>0 else None
    loss=strata['ordinary']['A4']['selection_recall']-strata['ordinary']['S4']['selection_recall']
    harmful={b:[dict(sequence=r['sequence'],source_i=r['source_i'],source_j=r['source_j']) for r in joined
                  if r['methods']['S4']['incorrect_accepted'] and not r['methods'][b]['incorrect_accepted']] for b in ['A4','R4']}
    helpful={b:[dict(sequence=r['sequence'],source_i=r['source_i'],source_j=r['source_j']) for r in joined
                  if not r['methods']['S4']['incorrect_accepted'] and r['methods'][b]['incorrect_accepted']] for b in ['A4','R4']}
    conditions=dict(recovery_25percent=recovery is not None and recovery>=.25,
       no_worse_than_R4=allm['S4']['mean_clipped_positive_error']<=allm['R4']['mean_clipped_positive_error'],
       ordinary_loss_at_most_2pp=loss<=.02,no_new_incorrect_A4=not harmful['A4'],no_new_incorrect_R4=not harmful['R4'])
    adequate=len(joined)>=60 and all(sum(r['stratum']==s for r in joined)>=10 for s in STRATA)
    signal=all(conditions.values());view=all(allm['S4']['mean_clipped_positive_error']<allm[m]['mean_clipped_positive_error'] for m in ['P4','P4x2'])
    decision='INCONCLUSIVE_DATA' if not adequate else 'INCONCLUSIVE_HEADROOM' if gap<=0 or diff<10 else 'NO_SUPPORT_SIGNAL_UNDER_FIXED_SCORE' if not signal else 'SUPPORT_AND_VIEW_AGGREGATION_SIGNAL' if view else 'SUPPORT_BUT_NO_VIEW_AGGREGATION_ADVANTAGE'
    save(out/'evaluator_only_decisions.json',joined);save(out/'evaluator_only_generation_labels.json',labs_all)
    result=dict(role=role,decision=decision,overall=allm,by_stratum=strata,adequate=adequate,discrepant_positives=diff,
       A4_oracle_gap=gap,gap_recovery=recovery,ordinary_recall_loss=loss,conditions=conditions,
       new_incorrect_pairs=harmful,resolved_incorrect_pairs=helpful,neural_expansion='STOPPED')
    save(out/'metrics.json',result)
    lines=[f'# SC-V1.1 {role}','',f'Decision: **{decision}**. Neural expansion remains stopped.','',
       '| Method | Strict success | Correct accepts | Incorrect/accepted | Clipped positive error |',
       '|---|---:|---:|---:|---:|']
    for m,v in allm.items():lines.append(f"| {m} | {v['strict_positive_successes']}/{v['positives']} | {v['correct_positive_accepts']} | {v['incorrect_accepted']}/{v['accepted']} | {v['mean_clipped_positive_error']:.9f} |")
    lines+=['',f'A4/oracle gap={gap}; discrepant positives={diff}; S4 recovery={recovery}; ordinary recall loss={loss}.',
      '',f'Conditions: {conditions}.','',
      'G4 and OR4 are evaluator-only diagnostics. No GT enters executable selectors. All arms use the same immutable bank; full per-stratum and pair transitions are in metrics/evaluator files. No independent-domain or novelty claim.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(role,decision,'recovery',recovery,'conditions',conditions,flush=True)
def run(role):
    import torch
    freeze=read(BASE/'execution_freeze.json')
    assert sha(BASE/'ANNEX.md')==freeze['annex_sha256']
    assert all(sha(p)==h for p,h in freeze['source_hashes'].items())
    assert all(sha(p)==h for p,h in freeze['input_hashes'].items())
    if role=='confirmation':
        f=read(BASE/'discovery_frozen.json')
        assert all(sha(BASE/'discovery'/p)==h for p,h in f['hashes'].items())
    out=BASE/role;out.mkdir(exist_ok=False);(out/'INCOMPLETE').write_text('Scoring/bank/independent validation pending\n');(out/'bank').mkdir()
    data=OLD/'data-expansion-01' if role=='discovery' else SC0
    rows=read(data/f'{role}_inputs.json');save(out/'inputs.json',rows)
    seq='00' if role=='discovery' else '08';meta=read(data/seq/'data.json')
    geom=RAPBackend.__new__(RAPBackend);geom.voxel_size=.4;geom.remove_outliers=True;geom.max_points_per_scan=4000;geom.seed=42;geom.feature_extractor=None
    cloud={};raw_hash={};controls={};timings=[];start=time.monotonic()
    b=setup_backend() if role=='confirmation' else None
    if b is not None:
        provenance=read(OLD/'pilot-01/discovery/provenance.json')
        assert sha(b.ckpt_path)==provenance['checkpoint_sha256']
        assert sha(b.spinnet_ckpt)==provenance['spinnet_sha256']
    def raw(i):
        r=meta['scans'][i];v=Path(r['path']).read_bytes();h=hashlib.sha256(v).hexdigest();assert h==r['sha256'];raw_hash[r['path']]=h
        p=np.frombuffer(v,dtype=np.float32).reshape(-1,4)[:,:3];return p[np.linalg.norm(p,axis=1)>=1.5]
    def points(i):
        if i not in cloud:cloud[i]=geom.preprocess(raw(i))[0]
        return cloud[i]
    old_delta=[];brute=[]
    for idx,r in enumerate(rows):
        if time.monotonic()-start>7200:raise RuntimeError('Two-hour cap')
        i,j=r['source_i'],r['source_j'];key=f'{seq}_{i}_{j}';t=time.perf_counter()
        left=[(np.array(s['local_pose']),points(s['source_id'])) for s in r['support_i']]
        right=[(np.array(s['local_pose']),points(s['source_id'])) for s in r['support_j']]
        anchors=[(np.eye(4),points(i))]*2,[(np.eye(4),points(j))]*2
        prep=time.perf_counter()-t;genwall=0.;torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        for g in range(4):
            name=f'{key}_{g}';bankpath=out/'bank'/f'{name}.json'
            if role=='discovery':
                parent=OLD/'pilot-01/discovery/bank'/f'{name}.json'
                shutil.copyfile(parent,bankpath);item=read(bankpath)
            else:
                t=time.perf_counter()
                try:
                    with request_rng(seed(seq,i,j,g),b):T,diag=b.register(raw(i),raw(j))
                    item=dict(sequence=seq,source_i=i,source_j=j,generation=g,seed=seed(seq,i,j,g),
                      T=T.tolist(),diagnostics=diag,validity=rotation_validity(T),rigidity_rmse_m=b.last_rigidity)
                except torch.cuda.OutOfMemoryError:raise
                except Exception as exc:raise RuntimeError(f'Generation engineering failure {name}: {exc}') from exc
                torch.cuda.synchronize();genwall+=time.perf_counter()-t
                save(bankpath,item)
            scores={};cost={}
            T=np.array(item['T']) if item['validity']['valid'] else None
            for m,fn,args in [('A4',scalar,anchors),('S4',scalar,(left,right)),('P4',pooled,(left,right))]:
                t=time.perf_counter();scores[m]=fn(T,*args) if T is not None else None;cost[m]=time.perf_counter()-t
            t=time.perf_counter()
            p1=pooled(T,left,right) if T is not None else None;p2=pooled(T,left,right) if T is not None else None
            cost['P4x2']=time.perf_counter()-t;assert p1==p2==scores['P4'];scores['P4x2']=p1
            q=sum(len(p) for _,p in left+right)
            controls[name]=dict(scores=scores,score_wall_s=cost,
                query_points=dict(A4=4*(len(points(i))+len(points(j))),S4=2*q,P4=q,P4x2=2*q),
                support_point_counts=[len(p) for _,p in left+right],anchor_point_counts=[len(points(i)),len(points(j))])
            if role=='discovery':
                delta=abs(scores['S4']-item['support']['score']);old_delta.append(delta);assert delta<1e-10
            if idx==0 and g==0 and T is not None:
                x=np.concatenate([move(a,p) for a,p in left]);y=np.concatenate([move(T@a,p) for a,p in right])
                d=np.concatenate([cdist(x[k:k+128],y).min(1) for k in range(0,len(x),128)])
                e=np.concatenate([cdist(y[k:k+128],x).min(1) for k in range(0,len(y),128)])
                reference=float((np.minimum(d,3).mean()+np.minimum(e,3).mean())/2)
                assert abs(reference-scores['P4'])<1e-10;brute.append(abs(reference-scores['P4']))
        peak=torch.cuda.max_memory_allocated();assert peak<=12*2**30
        timings.append(dict(pair=key,preprocess_wall_s=prep,generation_wall_s=genwall,peak_allocated=peak,peak_reserved=torch.cuda.max_memory_reserved()))
        if role=='discovery' and idx==9:
            save(out/'preflight_validation.json',dict(passed=True,hypotheses=40,max_saved_S4_delta=max(old_delta),pooled_bruteforce_delta=brute,
                original_observer_evidence='Inherited unchanged official observer and frozen HL-V2 preflight RNG/hook evidence; no new discovery generation'))
        if idx%10==0:print(role,idx+1,'/100',flush=True)
        if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>20*2**30:raise RuntimeError('Artifact cap')
    save(out/'controls.json',controls);save(out/'input_hashes.json',raw_hash)
    save(out/'bank_hashes.json',{p.name:sha(p) for p in sorted((out/'bank').glob('*.json'))})
    # Evaluator-only GT transport is performed after executable score persistence.
    gt=np.load(data/seq/'evaluator_gt.npy');g4={}
    for r in rows:
        i,j=r['source_i'],r['source_j'];key=f'{seq}_{i}_{j}'
        left=[(np.linalg.inv(gt[i])@gt[s['source_id']],points(s['source_id'])) for s in r['support_i']]
        right=[(np.linalg.inv(gt[j])@gt[s['source_id']],points(s['source_id'])) for s in r['support_j']]
        for g in range(4):
            name=f'{key}_{g}';item=read(out/'bank'/f'{name}.json');t=time.perf_counter()
            v=scalar(np.array(item['T']),left,right) if item['validity']['valid'] else None
            g4[name]=dict(score=v,wall_s=time.perf_counter()-t)
    save(out/'evaluator_only_G4.json',g4)
    save(out/'timings.json',dict(pairs=timings,wall_s=time.monotonic()-start,
         controls_wall_s={m:sum(v['score_wall_s'][m] for v in controls.values()) for m in ['A4','S4','P4','P4x2']},
         G4_wall_s=sum(x['wall_s'] for x in g4.values()),new_generations=400 if role=='confirmation' else 0,
         scope='Raw reads, geometric preprocessing, generation and control scoring; CPU scores monotonic wall, generation CUDA synchronized'))
    save(out/'score_validation.json',dict(passed=True,pooled_bruteforce_delta=brute,
         inherited_S4_max_delta=max(old_delta) if old_delta else None,valid_hypotheses=sum(read(p)['validity']['valid'] for p in (out/'bank').glob('*.json'))))
    evaluate(role)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['synthetic','discovery','confirmation']);a=p.parse_args()
    synthetic() if a.mode=='synthetic' else run(a.mode)
