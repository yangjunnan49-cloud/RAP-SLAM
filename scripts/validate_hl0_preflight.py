"""Independent bounded validation of the persisted HL0 bank, no pilot selection."""
import json
from pathlib import Path
import sys
import time
import random
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import hl0_preflight as h
from rap_slam.rap_backend import RAPBackend
from rap_slam.measurement import rotation_validity,transform_label
p=json.loads((h.OUT/'profile_pairs.json').read_text())[0];i,j=p['source_i'],p['source_j']
d=json.loads((h.OUT/'data_05.json').read_text())
def scan(k):
    xyz=np.fromfile(d['scans'][k]['path'],dtype=np.float32).reshape(-1,4)[:,:3]
    return xyz[np.linalg.norm(xyz,axis=1)>=1.5]
backend=RAPBackend()
py=random.getstate();ns=np.random.get_state();cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state();old_seed=backend.seed
with h.request_rng(h.request_seed(i,j,0),backend): T,diag=backend.register(scan(i),scan(j))
restored=py==random.getstate() and np.array_equal(ns[1],np.random.get_state()[1]) and ns[2:]==np.random.get_state()[2:] and torch.equal(cpu,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state()) and old_seed==backend.seed
original=json.loads((h.OUT/f'bank/{i}_{j}_0.json').read_text());delta=float(np.max(np.abs(T-np.array(original['T']))))
assert restored and delta<=1e-6
h.save('rng_fresh_process.json',dict(max_transform_element_delta=delta,exact=delta==0,caller_states_restored=restored,source_i=i,source_j=j,generation=0))
labels=[];counts=[]
gt=np.array(h.metrics.gt_from_summary(str(Path(d['parent'])/'pipeline')))
for p in json.loads((h.OUT/'profile_pairs.json').read_text()):
    i,j=p['source_i'],p['source_j'];target=np.linalg.inv(gt[i])@gt[j]
    assert rotation_validity(target,1e-4)['valid']
    for g in range(4):
        r=json.loads((h.OUT/f'bank/{i}_{j}_{g}.json').read_text())
        assert r['seed']==h.request_seed(i,j,g) and r['validity']['valid'] and r['support'] is not None
        lab=transform_label(r['T'],target)
        rte,rre=lab['gt_translation_error_m'],lab['gt_rotation_error_deg']
        labels.append(dict(source_i=i,source_j=j,generation=g,RTE_m=rte,RRE_deg=rre,strict_success=rte<=1 and rre<=5,legacy_label=lab['label'],GT_relative=target.tolist()))
        counts.append(r['validity'])
h.save('evaluator_only_profile_labels.json',labels)
h.save('bank_validation.json',dict(generations=len(labels),valid_se3=len(counts),valid_support_scores=len(counts),max_orthogonality=max(v['orthogonality'] for v in counts),max_det_error=max(v['det_error'] for v in counts),strict_labels='Evaluator-only feasibility outputs; no B1/B4/S4/oracle pilot result or gate computed',fresh_process_repeat=True))
h.save('bank_hashes.json',{p.name:h.sha(p) for p in sorted((h.OUT/'bank').glob('*.json'))})
print('Independent validation complete',delta,restored,flush=True)