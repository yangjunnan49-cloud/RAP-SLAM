"""HL-V2 evaluator-only labels, paired metrics and unchanged H1/H2 gates."""
import argparse,hashlib,json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from rap_slam.measurement import transform_label,rotation_validity
from hl_v2_bank import BASE,DATA,save,sha
METHODS=['B1','B4','S4','R4','OR4','REJ'];STRATA=['confuser','ambiguous_positive','low_overlap','ordinary']
def ratio(a,b):return a/b if b else None

def selectors(bank):
    valid=[r for r in bank if r['validity']['valid']]
    support=[r for r in valid if r.get('support') is not None]
    return dict(B1=bank[0] if bank[0]['validity']['valid'] else None,
        B4=min(valid,key=lambda r:(-r['diagnostics']['overlap_ratio'],r['diagnostics']['mean_dist'],r['generation'])) if valid else None,
        S4=min(support,key=lambda r:(r['support']['score'],r['generation'])) if support else None,
        R4=min(valid,key=lambda r:(r['rigidity_rmse_m'],r['generation'])) if valid else None,REJ=None)

def main(role):
    out=BASE/'pilot-01'/role;rows=json.loads((out/'inputs.json').read_text());bankhash=json.loads((out/'bank_hashes.json').read_text())
    assert all(sha(out/'bank'/p)==h for p,h in bankhash.items())
    labels=json.loads((DATA/f'{role}_evaluator.json').read_text());labelmap={(r['sequence'],r['source_i'],r['source_j']):r for r in labels}
    gt={s:np.load(DATA/s/'evaluator_gt.npy') for s in {r['sequence'] for r in rows}}
    decisions=[];oracle=[];generation_labels=[]
    for r in rows:
        seq,i,j=r['sequence'],r['source_i'],r['source_j'];Tgt=np.linalg.inv(gt[seq][i])@gt[seq][j]
        bank=[json.loads((out/'bank'/f'{seq}_{i}_{j}_{g}.json').read_text()) for g in range(4)]
        choices=selectors(bank);labs={}
        for g,b in enumerate(bank):
            if b['validity']['valid']:
                assert rotation_validity(b['T'])['valid'];lab=transform_label(b['T'],Tgt);rte=lab['gt_translation_error_m'];rre=lab['gt_rotation_error_deg'];e=max(rte,rre/5)
                labs[g]=dict(RTE_m=rte,RRE_deg=rre,error=e,clipped_error=min(5,e),strict=rte<=1 and rre<=5,legacy_safe=lab['label']=='safe')
                generation_labels.append(dict(sequence=seq,source_i=i,source_j=j,generation=g,**labs[g]))
        best=min(labs,key=lambda g:(labs[g]['error'],g)) if labs else None;choices['OR4']=bank[best] if best is not None else None
        values={}
        for m,b in choices.items():
            lab=labs[b['generation']] if b is not None else dict(RTE_m=None,RRE_deg=None,error=5.,clipped_error=5.,strict=False,legacy_safe=False)
            accepted=b is not None and b['diagnostics']['overlap_ratio']>=.7 and b['diagnostics']['mean_dist']<=2.5
            values[m]=dict(generation=None if b is None else b['generation'],accepted=accepted,incorrect_accepted=accepted and not lab['strict'],failed=b is None,**lab)
        decisions.append(dict(sequence=seq,source_i=i,source_j=j,stratum=labelmap[(seq,i,j)]['stratum'],methods={m:v for m,v in values.items() if m!='OR4'}))
        oracle.append(dict(sequence=seq,source_i=i,source_j=j,OR4=values['OR4']))
    save(out/'evaluator_only_generation_labels.json',generation_labels);save(out/'evaluator_only_decisions.json',decisions);save(out/'oracle_only.json',oracle)
    # Join only within the offline evaluator, never an inference selector.
    joined=[dict(r,methods=dict(r['methods'],OR4=o['OR4'])) for r,o in zip(decisions,oracle)]
    def aggregate(group,m):
        positives=[r for r in group if r['stratum']!='confuser'];confusers=[r for r in group if r['stratum']=='confuser'];v=[r['methods'][m] for r in group]
        success=sum(r['methods'][m]['strict'] for r in positives);accepted=sum(x['accepted'] for x in v);incorrect=sum(x['incorrect_accepted'] for x in v);correctaccept=sum(r['methods'][m]['accepted'] and r['methods'][m]['strict'] for r in positives)
        return dict(attempts=len(group),positives=len(positives),strict_positive_successes=success,selection_recall=ratio(success,len(positives)),correct_positive_accepts=correctaccept,correct_accept_recall=ratio(correctaccept,len(positives)),accepted=accepted,incorrect_accepted=incorrect,incorrect_per_accepted=ratio(incorrect,accepted),incorrect_per_attempt=ratio(incorrect,len(group)),confusers=len(confusers),accepted_confusers=sum(r['methods'][m]['accepted'] for r in confusers),confuser_accept_rate=ratio(sum(r['methods'][m]['accepted'] for r in confusers),len(confusers)),mean_clipped_positive_error=float(np.mean([r['methods'][m]['clipped_error'] for r in positives])) if positives else None,failures=sum(x['failed'] for x in v),legacy_safe=sum(x['legacy_safe'] for x in v),RTE_p90=float(np.percentile([x['RTE_m'] for x in v if x['RTE_m'] is not None],90)) if any(x['RTE_m'] is not None for x in v) else None,RRE_p90=float(np.percentile([x['RRE_deg'] for x in v if x['RRE_deg'] is not None],90)) if any(x['RRE_deg'] is not None for x in v) else None)
    overall={m:aggregate(joined,m) for m in METHODS};bystratum={s:{m:aggregate([r for r in joined if r['stratum']==s],m) for m in METHODS} for s in STRATA}
    discrepant=sum(r['stratum']!='confuser' and (r['methods']['OR4']['strict']!=r['methods']['B4']['strict'] or abs(r['methods']['OR4']['error']-r['methods']['B4']['error'])>=.10) for r in joined)
    adequate=len(rows)>=60 and all(sum(r['stratum']==s for r in joined)>=10 for s in STRATA) and discrepant>=10
    b,o,s=overall['B4'],overall['OR4'],overall['S4'];recallgain=o['selection_recall']-b['selection_recall'];errorgain=b['mean_clipped_positive_error']-o['mean_clipped_positive_error'];reduction=ratio(errorgain,b['mean_clipped_positive_error'])
    recallpass=recallgain>=.05;errorpass=reduction is not None and reduction>=.15;h1=recallpass or errorpass
    metric='recall' if recallpass else 'clipped_error' if errorpass else None
    recovery=ratio(s['selection_recall']-b['selection_recall'],recallgain) if metric=='recall' else ratio(b['mean_clipped_positive_error']-s['mean_clipped_positive_error'],errorgain) if metric=='clipped_error' else None
    loss=bystratum['ordinary']['B4']['selection_recall']-bystratum['ordinary']['S4']['selection_recall']
    harmful=[dict(sequence=r['sequence'],source_i=r['source_i'],source_j=r['source_j']) for r in joined if r['methods']['S4']['incorrect_accepted'] and not r['methods']['B4']['incorrect_accepted']]
    helpful=[dict(sequence=r['sequence'],source_i=r['source_i'],source_j=r['source_j']) for r in joined if r['methods']['B4']['incorrect_accepted'] and not r['methods']['S4']['incorrect_accepted']]
    h2=h1 and recovery is not None and recovery>=.25 and loss<=.02 and not harmful
    decision='INCONCLUSIVE' if not adequate else 'STOP_RANKING' if not h1 else 'STOP_NEURAL_EXPANSION' if not h2 else 'GO'
    result=dict(role=role,overall=overall,by_stratum=bystratum,discrepant_positive_pairs=discrepant,adequacy=adequate,H1=h1,H2=h2,H1_recall_gain=recallgain,H1_relative_error_reduction=reduction,H2_metric=metric,H2_gap_recovery=recovery,ordinary_recall_loss=loss,new_incorrect_pairs=harmful,resolved_incorrect_pairs=helpful,decision=decision,scope='Development engineering gates; discovery decision is diagnostic, confirmation decision governs')
    save(out/'metrics.json',result)
    lines=[f'# HL-V2 {role} results','',f"Decision: **{decision}**; adequacy={adequate}, discrepant positives={discrepant}.",'','| Method | Positive success/75 | Accepted | Incorrect accepted | Mean clipped positive error |','|---|---:|---:|---:|---:|']
    lines += [f"| {m} | {v['strict_positive_successes']}/{v['positives']} | {v['accepted']} | {v['incorrect_accepted']} | {v['mean_clipped_positive_error']:.6f} |" for m,v in overall.items()]
    lines += ['',f'H1 recall gain={recallgain:.6f}; relative clipped-error reduction={reduction}; H2 metric={metric}, recovery={recovery}; ordinary recall loss={loss}; newly incorrect pairs={len(harmful)}.', '', 'All methods share one immutable K4 bank. OR4 is an evaluator-only upper bound; R4 uses official average per-step rigidity RMSE. Ungated recall and gated acceptance remain separate. Full counts, failure denominators, continuous errors, all strata and harmful/helpful transitions are in metrics.json and evaluator-only decisions. No significance or independent-domain claim. No training was performed. Discovery must freeze before confirmation inference.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['overall','by_stratum']}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--role',choices=['discovery','confirmation'],required=True);main(p.parse_args().role)