"""PG-DEV-V1 frozen proposal-context screen; no production changes."""
import argparse, hashlib, json, sys, time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from hl_v2_bank import setup_backend
from hl0_preflight import request_rng, move, score
from rap_slam.rap_backend import RAPBackend
from rap_slam.measurement import rotation_validity, transform_label
from support_control import aggregate, STRATA
MAIN=Path('/mnt/e/githouse/RAP-main')
BASE=MAIN/'docs/maintenance/2026-09-15/proposal-context-dev-01'
SC=MAIN/'docs/maintenance/2026-09-14/support-control-02'
OLD=MAIN/'docs/maintenance/2026-09-14/hl-v2'
SC0=MAIN/'docs/maintenance/2026-09-14/support-control-01/data'
METHODS=['A1','A-R4','A-S4','A-OR4','P1','P-R4','P-OR4']
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    with Path(p).open('x') as f: json.dump(v,f,indent=2,allow_nan=False)
def paths(seq):
    role='discovery' if seq=='00' else 'confirmation'
    data=OLD/'data-expansion-01' if seq=='00' else SC0
    return role,data,SC/role
def geometry():
    g=RAPBackend.__new__(RAPBackend)
    g.voxel_size=.4;g.remove_outliers=True;g.max_points_per_scan=4000;g.seed=42;g.feature_extractor=None
    return g
def assemble(r,side,raw):
    anchor=r['source_'+side];ss=r['support_'+side]
    assert len(ss)==2 and len({s['source_id'] for s in ss})==2
    assert all(s['source_id']<anchor and 0<s['age_s']<=5 for s in ss)
    return np.concatenate([raw(anchor)]+[move(np.asarray(s['local_pose']),raw(s['source_id'])) for s in ss])
def verify_freeze():
    for p,h in read(BASE/'execution_freeze.json')['hashes'].items(): assert sha(p)==h,p
def freeze():
    counts={}
    for root in [OLD/'pilot-01',SC]:
        manifest=read(root/'artifact_hashes.json')
        for p,h in manifest.items(): assert sha(root/p)==h,p
        counts[str(root)]=len(manifest)
    old=read(SC/'execution_freeze.json')
    for group in ['source_hashes','input_hashes']:
        for p,h in old[group].items(): assert sha(p)==h,p
    files=[BASE/'PROTOCOL.md',Path(__file__),ROOT/'scripts/validate_proposal_context_dev.py']
    files += [Path(p) for group in ['source_hashes','input_hashes'] for p in old[group]]
    files += [SC/'artifact_hashes.json',OLD/'pilot-01/artifact_hashes.json']
    for seq in ['00','08']:
        role,data,parent=paths(seq);rows=read(data/f'{role}_inputs.json')
        assert len(rows)==100 and all(set(r)=={'sequence','source_i','source_j','support_i','support_j'} for r in rows)
        files += [parent/'controls.json',parent/'bank_hashes.json']
    rng=np.random.default_rng(7);p=rng.normal(size=(100,3))
    a=np.eye(4);s=np.eye(4);b=np.eye(4)
    a[:3,:3]=Rotation.from_rotvec([.2,-.3,.1]).as_matrix();a[:3,3]=[2,-3,4]
    s[:3,:3]=Rotation.from_rotvec([-.3,.4,.2]).as_matrix();s[:3,3]=[-2,1,5]
    b[:3,:3]=Rotation.from_rotvec([.1,.2,-.4]).as_matrix();b[:3,3]=[3,4,-2]
    actual=move(np.linalg.inv(a)@s,p)
    reference=np.linalg.solve(a,np.c_[move(s,p),np.ones(len(p))].T).T[:,:3]
    assert np.max(np.abs(actual-reference))<1e-12
    assert np.max(np.abs(move(np.linalg.inv(a)@b,move(np.linalg.inv(b)@s,p))-actual))<1e-12
    q=geometry().preprocess(rng.normal(size=(30000,3))*10)[0]
    assert 200<=len(q)<=4000
    save(BASE/'synthetic_validation.json',dict(passed=True,transport_delta=float(np.max(np.abs(actual-reference))),capped_points=len(q)))
    files.append(BASE/'synthetic_validation.json')
    save(BASE/'execution_freeze.json',dict(frozen_utc=datetime.now(timezone.utc).isoformat(),
         hashes={str(p):sha(p) for p in sorted(set(files))},parent_counts=counts,
         scope='Both sequences observed development; no new model outputs yet'))
    print('Freeze and geometry validation passed',counts,flush=True)
def choose(bank,kind,scores=None):
    good=[x for x in bank if x['validity']['valid']]
    if not good:return None
    if kind=='1':return bank[0] if bank[0]['validity']['valid'] else None
    if kind=='R4':return min(good,key=lambda x:(x['rigidity_rmse_m'],x['generation']))
    return min(good,key=lambda x:(scores[x['generation']],x['generation']))
def gate(g,anchors,item):
    if item is None:return dict(overlap_ratio=None,mean_dist=None,accepted=False)
    ov,md=g._alignment_metrics(*anchors,np.asarray(item['T']),thresh=1.)
    return dict(overlap_ratio=ov,mean_dist=md,accepted=ov>=.7 and md<=2.5)
def run(seq):
    import torch
    verify_freeze()
    if seq=='08':
        for p,h in read(BASE/'00_frozen.json').items():assert sha(BASE/'00'/p)==h
    out=BASE/seq;out.mkdir(exist_ok=False);(out/'bank').mkdir()
    (out/'INCOMPLETE').write_text('Generation/evaluation/independent validation pending\n')
    role,data,parent=paths(seq);rows=read(data/f'{role}_inputs.json');save(out/'inputs.json',rows)
    meta=read(data/seq/'data.json');controls=read(parent/'controls.json')
    g=geometry();start=time.monotonic();b=setup_backend()
    provenance=read(OLD/'pilot-01/discovery/provenance.json')
    assert sha(b.ckpt_path)==provenance['checkpoint_sha256']
    assert sha(b.spinnet_ckpt)==provenance['spinnet_sha256']
    assert b.max_points_per_scan==4000 and b.voxel_size==.4
    used={};profiles=[];gates={};reproductions=[]
    prior=read(BASE/'00/timings.json')['process_wall_s'] if seq=='08' else 0
    def raw_reader():
        cache={}
        def raw(i):
            if i not in cache:
                m=meta['scans'][i];v=Path(m['path']).read_bytes()
                p=np.frombuffer(v,dtype=np.float32).reshape(-1,4)[:,:3]
                assert np.isfinite(p).all()
                cache[i]=p[np.linalg.norm(p,axis=1)>=1.5]
            return cache[i]
        return raw
    def generate(x,y,old):
        with request_rng(old['seed'],b):T,diag=b.register(x,y)
        assert diag['n_a']<=4000 and diag['n_b']<=4000
        return dict(sequence=seq,source_i=old['source_i'],source_j=old['source_j'],
            generation=old['generation'],seed=old['seed'],T=T.tolist(),
            diagnostics=diag,validity=rotation_validity(T),rigidity_rmse_m=b.last_rigidity)
    for idx,r in enumerate(rows):
        if time.monotonic()-start+prior>7200:raise RuntimeError('Process time cap')
        key=f"{seq}_{r['source_i']}_{r['source_j']}"
        old=[read(parent/'bank'/f'{key}_{k}.json') for k in range(4)]
        ids=[r['source_i'],r['source_j']]+[s['source_id'] for side in ['i','j'] for s in r['support_'+side]]
        for i in ids:
            m=meta['scans'][i]
            if m['path'] not in used:
                h=sha(m['path']);assert h==m['sha256'];used[m['path']]=h
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        record=dict(pair=key,timing_sample=idx<10)
        for arm in (['A','P'] if idx%2==0 else ['P','A']):
            if arm=='A' and idx>=10:continue
            raw=raw_reader();torch.cuda.synchronize();t=time.perf_counter()
            if arm=='P':
                x=assemble(r,'i',raw);y=assemble(r,'j',raw)
                record['pooled_raw_counts']=[len(x),len(y)];tt=time.perf_counter()
                fresh=[generate(x,y,v) for v in old]
                torch.cuda.synchronize();record['P_generation_s']=time.perf_counter()-tt
                anchors=[g.preprocess(raw(r['source_'+side]))[0] for side in ['i','j']]
                gate(g,anchors,choose(fresh,'R4'))
                torch.cuda.synchronize();record['P_R4_full_s']=time.perf_counter()-t
                gates[key]={'P':[gate(g,anchors,v if v['validity']['valid'] else None) for v in fresh],
                            'A':[gate(g,anchors,v if v['validity']['valid'] else None) for v in old]}
            else:
                x,y=raw(r['source_i']),raw(r['source_j'])
                replay=[generate(x,y,v) for v in old];torch.cuda.synchronize()
                anchors=[g.preprocess(raw(r['source_'+side]))[0] for side in ['i','j']]
                gate(g,anchors,choose(replay,'R4'))
                record['A_R4_full_s']=time.perf_counter()-t
                support=[[(np.asarray(s['local_pose']),g.preprocess(raw(s['source_id']))[0])
                          for s in r['support_'+side]] for side in ['i','j']]
                vals=[score(np.asarray(v['T']),*support)['score'] for v in replay]
                gate(g,anchors,choose(replay,'S4',vals));torch.cuda.synchronize()
                record['A_S4_full_s']=time.perf_counter()-t
                for k,v in enumerate(replay):
                    delta=float(np.max(np.abs(np.asarray(v['T'])-old[k]['T'])))
                    assert delta<=1e-6,(key,k,delta)
                    assert abs(vals[k]-controls[f'{key}_{k}']['scores']['S4'])<=1e-8
                    reproductions.append(dict(pair=key,generation=k,transform_delta=delta))
        for k,v in enumerate(fresh):save(out/'bank'/f'{key}_{k}.json',v)
        record['peak_allocated']=torch.cuda.max_memory_allocated()
        record['peak_reserved']=torch.cuda.max_memory_reserved()
        assert record['peak_allocated']<=12*2**30;profiles.append(record)
        if idx==9:
            assert all(read(p)['validity']['valid'] for p in (out/'bank').glob('*.json'))
            save(out/'preflight_validation.json',dict(passed=True,hypotheses=40,replays=40,
                 max_anchor_delta=max(x['transform_delta'] for x in reproductions),no_gt_loaded=True))
        if idx%10==0:print(seq,idx+1,'/100',round(time.monotonic()-start,1),'s',flush=True)
    save(out/'gates.json',gates);save(out/'input_hashes.json',used);save(out/'reproduction.json',reproductions)
    sample=[p for p in profiles if p['timing_sample']]
    cost={k:sum(p[k] for p in sample) for k in ['A_R4_full_s','A_S4_full_s','P_R4_full_s']}
    cost['ratio_P_to_A_S4']=cost['P_R4_full_s']/cost['A_S4_full_s']
    save(out/'timings.json',dict(pairs=profiles,sampled_cost=cost,process_wall_s=time.monotonic()-start,
          new_pooled_generations=400,timing_anchor_replays=40,scope='Ten fixed pairs; warm OS cache; fresh per-arm raw cache; model load/evidence I/O excluded'))
    assert sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file())<20*2**30
    save(out/'bank_hashes.json',{p.name:sha(p) for p in sorted((out/'bank').glob('*.json'))})
    evaluate(seq)
def evaluate(seq):
    role,data,parent=paths(seq);out=BASE/seq
    labels=read(data/f'{role}_evaluator.json');gt=np.load(data/seq/'evaluator_gt.npy')
    controls=read(parent/'controls.json');gates=read(out/'gates.json');joined=[]
    for r in labels:
        i,j=r['source_i'],r['source_j'];key=f'{seq}_{i}_{j}';values={}
        for arm,root in [('A',parent),('P',out)]:
            bank=[read(root/'bank'/f'{key}_{k}.json') for k in range(4)];labs={}
            for k,v in enumerate(bank):
                if v['validity']['valid']:
                    lab=transform_label(v['T'],np.linalg.inv(gt[i])@gt[j])
                    t,a=lab['gt_translation_error_m'],lab['gt_rotation_error_deg']
                    labs[k]=dict(RTE_m=t,RRE_deg=a,error=max(t,a/5),clipped_error=min(5,max(t,a/5)),
                        strict=t<=1 and a<=5,legacy_safe=lab['label']=='safe')
            for kind in (['1','R4','S4','OR4'] if arm=='A' else ['1','R4','OR4']):
                scores={k:lab['error'] for k,lab in labs.items()} if kind=='OR4' else {k:controls[f'{key}_{k}']['scores']['S4'] for k in range(4)} if kind=='S4' else None
                v=choose(bank,kind,scores);k=v['generation'] if v else None
                lab=labs[k] if k is not None else dict(RTE_m=None,RRE_deg=None,error=5.,clipped_error=5.,strict=False,legacy_safe=False)
                accepted=gates[key][arm][k]['accepted'] if k is not None else False
                name=arm+'1' if kind=='1' else arm+'-'+kind
                values[name]=dict(generation=k,accepted=accepted,incorrect_accepted=accepted and not lab['strict'],failed=k is None,**lab)
        joined.append(dict(sequence=seq,source_i=i,source_j=j,stratum=r['stratum'],methods=values))
    overall={m:aggregate(joined,m) for m in METHODS}
    strata={s:{m:aggregate([r for r in joined if r['stratum']==s],m) for m in METHODS} for s in STRATA}
    harmful={m:[f"{r['source_i']}_{r['source_j']}" for r in joined if r['methods']['P-R4']['incorrect_accepted'] and not r['methods'][m]['incorrect_accepted']] for m in ['A-R4','A-S4']}
    conditions=dict(oracle_gain_5pp=overall['P-OR4']['selection_recall']-overall['A-OR4']['selection_recall']>=.05,
        executable_gain_3pp=overall['P-R4']['selection_recall']-overall['A-R4']['selection_recall']>=.03,
        no_worse_than_S4=overall['P-R4']['selection_recall']>=overall['A-S4']['selection_recall'])
    for m in ['A-R4','A-S4']:
        conditions['ordinary_loss_'+m]=strata['ordinary'][m]['selection_recall']-strata['ordinary']['P-R4']['selection_recall']<=.02
        conditions['no_new_incorrect_'+m]=not harmful[m]
    cost=read(out/'timings.json')['sampled_cost'];conditions['sampled_cost']=cost['ratio_P_to_A_S4']<=1.25
    save(out/'evaluator_only_decisions.json',joined)
    result=dict(sequence=seq,role='observed_development',overall=overall,by_stratum=strata,
        conditions=conditions,new_incorrect_pairs=harmful,sampled_cost=cost,
        decision='DEVELOPMENT_SIGNAL' if all(conditions.values()) else 'NO_DEVELOPMENT_CONTEXT_SIGNAL')
    save(out/'metrics.json',result);print(seq,result['decision'],conditions,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','00','08']);a=p.parse_args()
    freeze() if a.mode=='freeze' else run(a.mode)
