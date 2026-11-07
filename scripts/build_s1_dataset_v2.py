"""Build a bounded, fresh M3 pilot with measurement-level labels and split manifests.

This is a case-control engineering pilot, not an unbiased retrieval evaluation.
Every run uses a new output directory, independent request seeds, and new caches.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'scripts')]
from rap_slam import real_dataset, scan_context
from rap_slam.measurement import (stable_id, make_measurement, best_generation,
                                  historical_margin, grouped_split)
from build_s1_features import degeneracy_features


def sha(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True)
    parser.add_argument('--seed',type=int,default=20260910)
    args=parser.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    run_id=f'{out.name}-{uuid.uuid4().hex}'
    (out/'INCOMPLETE').write_text('Pilot generation in progress\n')
    mg_path=ROOT/'output_multigen/multigen_samples.json'
    entries=json.loads(mg_path.read_text())['entries']
    selections=[]
    for ds,count,source,seq,offset in [('kitti02_false',11,'kitti','02',0),
                                       ('kitti02_true',9,'kitti','02',0),
                                       ('poss_true',2,'poss','00',3850)]:
        pool=sorted([e for e in entries if e['ds']==ds],key=lambda e:stable_id([args.seed,e['key']]))[:count]
        for e in pool:
            selections.append(dict(source=source,sequence=seq,ids=[e['i']+offset,e['j']+offset],
                                   legacy_key=e['key'],selection_bucket=ds))
    kp=ROOT/'output_prods/kitti06/accepted_loops.json'
    for e in json.loads(kp.read_text())[:2]:
        selections.append(dict(source='kitti',sequence='06',ids=[e['i'],e['j']],
                               legacy_key=f"kitti06_off|{e['i']}|{e['j']}",selection_bucket='legacy_accepted'))
    assert len(selections)==24 and len({(s['source'],s['sequence'],tuple(s['ids'])) for s in selections})==24, 'Pilot requires 24 distinct physical pairs'
    (out/'selection_manifest.json').write_text(json.dumps(dict(seed=args.seed,run_id=run_id,
        design='24 preselected historical case-control pairs, two fresh generations each; selected-generation export for group evaluation',
        source_hashes={str(mg_path):sha(mg_path),str(kp):sha(kp)},pairs=selections),indent=2))
    paths={('kitti','02'):ROOT/'output_ros2/step3_kitti02/keyframes.json',
           ('kitti','06'):ROOT/'output_prods/kitti06/keyframes.json',
           ('poss','00'):ROOT/'output_ros2/step3/keyframes.json'}
    datasets={}; input_files=[]
    for key,kf_path in paths.items():
        source,seq=key
        base=Path('/mnt/e/githouse/Datasets/data_odometry_velodyne/dataset') if source=='kitti' else ROOT/'data/poss/dataset'
        seq_dir=base/'sequences'/seq
        gt_path=base/'poses'/f'{seq}.txt';calib=seq_dir/'calib.txt'
        gt=real_dataset.load_gt_poses(str(gt_path),str(calib),apply_calib=source=='kitti')
        offset=3850 if source=='poss' else 0
        pairs=[s['ids'] for s in selections if (s['source'],s['sequence'])==key]
        keyframes=sorted(int(k)+offset for k in json.loads(kf_path.read_text()))
        needed={i for pair in pairs for i in pair}|{f for f in keyframes if f<max(j for i,j in pairs)}
        descriptors={};clouds={};deg={}
        endpoints={i for pair in pairs for i in pair}
        for n,frame in enumerate(sorted(needed)):
            path=seq_dir/'velodyne'/f'{frame:06d}.bin'
            raw=np.fromfile(path,dtype=np.float32).reshape(-1,4)[:,:3]
            input_files.append(dict(path=str(path),sha256=sha(path),source_frame_id=frame,dataset=source,sequence=seq))
            descriptors[frame]=scan_context.make_scan_context(raw)
            if frame in endpoints:
                clouds[frame]=raw[np.linalg.norm(raw,axis=1)>=1.5]
                deg[frame]=degeneracy_features(raw)
            if n%100==0:print(f'{source}/{seq}: descriptors {n+1}/{len(needed)}',flush=True)
        datasets[key]=dict(gt=gt,keyframes=keyframes,descriptors=descriptors,clouds=clouds,deg=deg)
        input_files.extend(dict(path=str(p),sha256=sha(p)) for p in [gt_path,calib,kf_path])
        np.savez_compressed(out/f'sc_{source}_{seq}.npz',**{f'f{i}':v for i,v in descriptors.items()})
    (out/'input_manifest.json').write_text(json.dumps(input_files,indent=2))
    import torch
    from rap_slam.rap_backend import RAPBackend
    backend=RAPBackend()
    provenance=dict(precision='ieee_float32_matmul_scope',seed_policy='sha256(root_seed,physical_sequence,ordered_pair,generation)',
                    checkpoint_sha256=sha(backend.ckpt_path),spinnet_sha256=sha(backend.spinnet_ckpt),
                    source_sha256={p:sha(ROOT/p) for p in ['rap_slam/rap_backend.py','rap_slam/rap_precision.py','rap_slam/measurement.py','rap_slam/scan_context.py','rap_slam/real_dataset.py','scripts/build_s1_features.py','scripts/build_s1_dataset_v2.py']},
                    torch=torch.__version__,sampling_steps=5,max_points_per_scan=4000,voxel_m=.4,
                    preprocessing='range >=1.5m, backend outlier filter and features enabled; no KITTI205',
                    sc_provenance='Recomputed raw-scan descriptors and saved keyframes, full causal pool; historical budget/shortlist not claimed')
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    records=[];selected=[];legacy_differences=[]
    for count,s in enumerate(selections):
        source,seq=s['source'],s['sequence'];i,j=s['ids'];data=datasets[(source,seq)]
        gt_rel=np.linalg.inv(data['gt'][i])@data['gt'][j]
        feature=historical_margin(i,j,data['keyframes'],data['descriptors'])
        for name in ['loc_t','loc_r','scat']:
            values=[data['deg'][f].get(name) for f in [i,j]]
            feature[f'deg_{name}']=None if None in values else min(values)
        generations=[];pair_records=[]
        for generation in range(2):
            seed=int(stable_id([args.seed,source,seq,i,j,generation])[:8],16)%(2**31)
            torch.manual_seed(seed);np.random.seed(seed)
            transform,diag=backend.register(data['clouds'][i],data['clouds'][j])
            g=dict(T_rel=transform,overlap=diag['overlap_ratio'],mean_dist=diag['mean_dist'])
            record=make_measurement(run_id,source,seq,[i,j],generation,g,gt_rel,
                                    dict(seed=seed,provenance_file='provenance.json'))
            if record['labels']['label']=='unlabeled':raise ValueError(record['labels'])
            record.update(feature);record['selection_bucket']=s['selection_bucket'];record['legacy_pair_key']=s['legacy_key']
            record['heuristic_accept']=record['overlap']>=.7 and record['mean_dist']<=2.5
            generations.append(g);pair_records.append(record)
        chosen=best_generation(generations)
        for index,record in enumerate(pair_records):record['selected_generation']=index==chosen
        records.extend(pair_records);selected.append(pair_records[chosen])
        legacy_differences.append(dict(pair=s['legacy_key'],selected_generation=chosen,
            mixed_metric_would_differ=generations[chosen]['mean_dist']!=min(g['mean_dist'] for g in generations)))
        with (out/'measurements.jsonl').open('a') as f:
            for record in pair_records:f.write(json.dumps(record,allow_nan=False)+'\n')
        print(f"pair {count+1}/{len(selections)} {source}/{seq} {i},{j}: {pair_records[chosen]['labels']['label']}",flush=True)
    split=grouped_split(selected,seed=str(args.seed))
    (out/'selected_measurements.json').write_text(json.dumps(selected,indent=2,allow_nan=False))
    (out/'splits.json').write_text(json.dumps(split,indent=2))
    split_counts={name:dict(Counter(r['labels']['label'] for r in selected if r['measurement_id'] in ids)) for name,ids in split['measurement_ids'].items()}
    report=dict(status='passed',measurements=len(records),physical_pairs=len(selected),
                labels_all=dict(Counter(r['labels']['label'] for r in records)),
                labels_selected=dict(Counter(r['labels']['label'] for r in selected)),
                split_label_counts=split_counts,missing_sc_margin=sum(r['sc_margin'] is None for r in selected),
                legacy_mixing_comparison=legacy_differences,
                heuristic_selected_confusion=dict(Counter(f"{r['labels']['label']}:{'accept' if r['heuristic_accept'] else 'reject'}" for r in selected)),
                limitation='Case-control pilot with correlated physical sites; group split excludes identical/reversed pairs only. No calibration model fit or population guarantee.',
                next='Expand predeclared physical-site/sequence groups before statistical model fitting; retain selected generation only as the unit per pair')
    (out/'quality_report.json').write_text(json.dumps(report,indent=2))
    (out/'INCOMPLETE').unlink()
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
