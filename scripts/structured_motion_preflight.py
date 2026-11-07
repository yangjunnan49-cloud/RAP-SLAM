"""F1 engineering preparation/run; run consumes sanitized inputs, never GT."""
import hashlib,json,sys,time,random,zipfile
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
MAIN=ROOT.parent/'RAP-main'
BASE=MAIN/'docs/maintenance/2026-09-26/f1-structured-preflight-01'
DATA=MAIN/'docs/maintenance/2026-09-14/hl-v2/data-expansion-01'
PAIRS=[(24,629,'confuser'),(210,1656,'ambiguous_positive'),(422,3496,'low_overlap'),(956,3859,'ordinary')]

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for v in iter(lambda:f.read(2**20),b''):h.update(v)
    return h.hexdigest()
def save(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def prepare():
    frozen=read(DATA/'cohort_freeze.json');rows=read(DATA/'discovery_inputs.json');labs=read(DATA/'discovery_evaluator.json')
    for name in ['discovery_inputs.json']:
        assert sha(DATA/name)==frozen['hashes'][name]
    for name,h in read(DATA/'artifact_hashes.json').items():assert sha(DATA/name)==h,name
    meta=read(DATA/'00/data.json')
    for p,h in meta['parent_hashes'].items():assert sha(p)==h
    op=next(p for p in meta['parent_hashes'] if p.endswith('/odom_poses.txt'))
    odom=np.tile(np.eye(4),(4541,1,1));odom[:,:3,:]=np.loadtxt(op).reshape(-1,3,4)
    chosen=[];hashes={}
    for idx,(i,j,s) in enumerate(PAIRS):
        first=next(r for r in labs if r['stratum']==s)
        assert (first['source_i'],first['source_j'])==(i,j)
        r=next(r for r in rows if (r['source_i'],r['source_j'])==(i,j))
        scans=[];motions=[]
        for side in ['i','j']:
            anchor=r['source_'+side]
            scans.append(meta['scans'][anchor]);motions.append(np.eye(4).tolist())
            assert len(r['support_'+side])==2
            for sup in r['support_'+side]:
                k=sup['source_id'];assert k<anchor and 0<sup['age_s']<=5
                L=np.linalg.inv(odom[anchor])@odom[k]
                assert np.max(np.abs(L-np.asarray(sup['local_pose'])))<1e-10
                scans.append(meta['scans'][k]);motions.append(sup['local_pose'])
        for x in scans:
            assert sha(x['path'])==x['sha256'];hashes[x['path']]=x['sha256']
        chosen.append(dict(key=f'00_{i}_{j}',seed=260926+idx,scans=scans,motions=motions))
    save(BASE/'inputs.json',chosen)
    files=[BASE/'PROTOCOL.md',BASE/'inputs.json',BASE/'synthetic_validation.json',MAIN/'scripts/maintenance/validate_f1_structured.py',Path(__file__),ROOT/'rap_slam/structured_motion.py',
           ROOT/'rap_slam/rap_backend.py',ROOT/'rap_slam/rap_precision.py',ROOT/'scripts/hl0_preflight.py']
    files+=list((MAIN/'rectified_point_flow').rglob('*.py'))+list((MAIN/'config').rglob('*.yaml'))
    files+=list((MAIN/'dataset_process').rglob('*.py'))
    sources={str(p):sha(p) for p in files}
    for name in ['rap_model_12.ckpt','mini_spinnet_t.pth']:
        p=MAIN/'weights'/name;sources[str(p)]=sha(p)
    with zipfile.ZipFile(BASE/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,str(p.relative_to(ROOT.parent)))
    save(BASE/'freeze.json',dict(utc=datetime.now(timezone.utc).isoformat(),sources=sources,raw=hashes,
        parent_hashes={str(DATA/n):sha(DATA/n) for n in ['discovery_inputs.json','discovery_evaluator.json','cohort_freeze.json','00/data.json']},
        pairs=[dict(i=i,j=j,stratum=s) for i,j,s in PAIRS],motion_source=op,motion_source_sha256=sha(op)))
    print('F1 frozen',len(chosen),'pairs',len(hashes),'raw scans',flush=True)

def check_freeze():
    f=read(BASE/'freeze.json')
    for group in ['sources','raw']:
        for p,h in f[group].items():assert sha(p)==h,p


def run():
    import torch
    from rap_slam.rap_backend import RAPBackend
    from rap_slam.rap_precision import ieee_matmul
    from rap_slam.structured_motion import build_batch,group_coordinates,project_groups,decode_groups,decode_parts,infer
    from rap_slam.measurement import rotation_validity
    from hl0_preflight import request_rng
    check_freeze();out=BASE/'attempt-01';out.mkdir(exist_ok=False)
    (out/'INCOMPLETE').write_text('Engineering checks pending\n')
    start=time.monotonic();torch.cuda.reset_peak_memory_stats();b=RAPBackend();assert b.device.type=='cuda'
    native_method=b.model.sample_rectified_flow
    records=[]
    def budget():
        assert time.monotonic()-start<=1800,'time cap'
        assert torch.cuda.max_memory_allocated()<=12*2**30,'memory cap'
    def states():return (torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone(),random.getstate(),np.random.get_state(),b.seed,torch.get_float32_matmul_precision())
    def same_state(a,c):
        assert torch.equal(a[0],c[0]) and torch.equal(a[1],c[1]) and a[2]==c[2]
        assert a[3][0]==c[3][0] and np.array_equal(a[3][1],c[3][1]) and a[3][2:]==c[3][2:]
        assert a[4:]==c[4:]
    before=states()
    try:
        with request_rng(99,b),ieee_matmul():
            torch.rand(3,device=b.device);np.random.rand();random.random()
            raise RuntimeError('intentional restoration test')
    except RuntimeError as e:assert str(e)=='intentional restoration test'
    same_state(before,states())
    for idx,r in enumerate(read(BASE/'inputs.json')):
        budget();dest=out/r['key'];dest.mkdir();points=[];features=[]
        state=states()
        with request_rng(r['seed'],b),ieee_matmul(),torch.inference_mode():
            b.seed=42
            for scan,cap in zip(r['scans'],[2000,1000,1000,2000,1000,1000]):
                p=np.fromfile(scan['path'],np.float32).reshape(-1,4)[:,:3]
                p=p[np.linalg.norm(p,axis=1)>=1.5]
                x,f=b.preprocess(p,max_points=cap)
                assert len(x)==cap and np.isfinite(f).all() and np.any(f)
                points.append(x);features.append(f)
        same_state(state,states());budget()
        L=np.asarray(r['motions']);batch,scale,centers,counts=build_batch(b,points,features)
        q=group_coordinates(points,L,centers,scale,b.device)
        with request_rng(r['seed'],b):noise=torch.randn_like(batch['pointclouds'])
        assert np.max(np.abs(batch['pointclouds'].cpu().numpy()*scale+np.repeat(centers,counts,axis=0)-np.concatenate(points)))<1e-4
        np.savez_compressed(dest/'inputs.npz',points=np.concatenate(points),features=np.concatenate(features),counts=counts,
            cond=batch['pointclouds'].cpu().numpy(),noise=noise.cpu().numpy(),centers=centers,scale=scale,motions=L,group_source=q.cpu().numpy())
        def calc(batch_,noise_,q_=None):
            state_=states();torch.cuda.synchronize();t=time.monotonic()
            with request_rng(r['seed'],b):value=infer(b,batch_,noise_,q_,4000 if q_ is not None else None)
            torch.cuda.synchronize();wall=time.monotonic()-t;same_state(state_,states());budget()
            assert b.model.sample_rectified_flow==native_method
            return value,wall
        native,nt=calc(batch,noise)
        with request_rng(r['seed'],b),ieee_matmul(),torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.float16):
            reference=b.model.sample_rectified_flow(batch,None,x_1=noise,return_tarjectory=True)
        delta=max(float((native[k]-reference[k]).abs().max()) for k in ['end_point_trajectory','trajectory'])
        assert delta<=1e-6,('native reproduction',delta)
        structured,st=calc(batch,noise,q)
        def record(arm,result,bb,cc,ss,wall,group=False,group_q=None,motions=None):
            raw=result['end_point_trajectory'][-1]
            with ieee_matmul(),torch.inference_mode():
                raw_T=decode_parts(bb,raw,cc,ss)
                if group:
                    final,g=project_groups(raw,group_q,4000);Tg=decode_groups(g,cc,ss)
                    fitted=decode_parts(bb,final,cc,ss)
                    residual=max(float(np.max(np.abs(np.linalg.inv(fitted[0 if k<3 else 3])@fitted[k]-motions[k]))) for k in range(6))
                    assert residual<=1e-4,('constraint',residual)
                    T=np.linalg.inv(Tg[0])@Tg[1]
                else:
                    final=result['projected_clean_trajectory'][-1];Tg=None;residual=None
                    T=np.linalg.inv(raw_T[0])@raw_T[3 if len(raw_T)==6 else 1]
                for t in list(raw_T)+[T]:assert rotation_validity(t)['valid']
                if Tg is not None:
                    for t in Tg:assert rotation_validity(t)['valid']
            arrays={k:v.detach().float().cpu().numpy() for k,v in result.items()}
            assert all(np.isfinite(v).all() for v in arrays.values())
            np.savez_compressed(dest/f'{arm}.npz',**arrays,decoded_final=final.cpu().numpy(),raw_poses=raw_T,
                T=T,group_poses=np.empty((0,4,4)) if Tg is None else Tg)
            item=dict(pair=r['key'],arm=arm,T=T.tolist(),wall_s=wall,constraint_matrix_residual=residual,
                point_counts=bb['points_per_part'][0].cpu().tolist(),native_max_delta=delta if arm=='N' else None)
            records.append(item);return item
        record('N',native,batch,centers,scale,nt)
        torch.cuda.synchronize();t=time.monotonic()
        with ieee_matmul(),torch.inference_mode():project_groups(native['end_point_trajectory'][-1],q,4000)
        torch.cuda.synchronize();ft=time.monotonic()-t
        record('F',native,batch,centers,scale,ft,True,q,L)
        record('S',structured,batch,centers,scale,st,True,q,L)
        transported=[p@l[:3,:3].T+l[:3,3] for p,l in zip(points,L)]
        pooled=[np.concatenate(transported[:3]).astype(np.float32),np.concatenate(transported[3:]).astype(np.float32)]
        pb,ps,pc,pn=build_batch(b,pooled,[np.concatenate(features[:3]),np.concatenate(features[3:])])
        pred,pt=calc(pb,noise);record('P',pred,pb,pc,ps,pt)
        ab,ass,ac,an=build_batch(b,[points[0],points[3]],[features[0],features[3]])
        ap,at=calc(ab,torch.cat([noise[:2000],noise[4000:6000]]));record('A',ap,ab,ac,ass,at)
        if idx==3:
            replay,rt=calc(batch,noise,q)
            replay_delta=float((replay['end_point_trajectory']-structured['end_point_trajectory']).abs().max())
            assert replay_delta<=1e-6
            save(dest/'replay.json',dict(max_delta=replay_delta,wall_s=rt))
            for sign in [-1,1]:
                a=np.deg2rad(sign);D=np.eye(4);D[:3,:3]=[[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]];D[0,3]=sign*.1
                pert=L.copy()
                for k in [1,2,4,5]:pert[k]=D@L[k]
                pq=group_coordinates(points,pert,centers,scale,b.device)
                pr,pwall=calc(batch,noise,pq)
                record('Sminus' if sign<0 else 'Splus',pr,batch,centers,scale,pwall,True,pq,pert)
                np.save(dest/('motions-minus.npy' if sign<0 else 'motions-plus.npy'),pert)
        budget();print('F1 pair complete',r['key'],'native delta',delta,'wall',round(time.monotonic()-start,2),flush=True)
    save(out/'records.json',records)
    save(out/'execution.json',dict(wall_s=time.monotonic()-start,peak_allocated=torch.cuda.max_memory_allocated(),
        peak_reserved=torch.cuda.max_memory_reserved(),no_hooks_installed=True,restoration_on_exception=True,
        gt_loaded_by_runner=False,torch=torch.__version__,gpu=torch.cuda.get_device_name(),completed_utc=datetime.now(timezone.utc).isoformat()))
    check_freeze();assert sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<2*2**30
    save(out/'artifact_hashes.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='INCOMPLETE'})
    print('F1 execution complete; independent validation pending',flush=True)
if __name__=='__main__':{'prepare':prepare,'run':run}[sys.argv[1]]()