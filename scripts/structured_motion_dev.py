"""F2-V1 structured-motion generation development screen. No production changes.

Modes: prepare (verify parents, fix input identities, freeze) then 00 / 08
(generation bank, common gate, evaluator-only metrics). Implements the frozen
protocol in docs/maintenance/2026-09-27/f2-structured-dev-01/PROTOCOL.md.
No GT or stratum labels are read outside prepare (fixed-ID membership only)
and evaluate (after the sequence generation bank is hash-frozen).
"""
import argparse, hashlib, json, random, sys, time, zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
MAIN = ROOT.parent / 'RAP-main'
BASE = MAIN / 'docs/maintenance/2026-09-27/f2-structured-dev-01'
D00 = MAIN / 'docs/maintenance/2026-09-14/hl-v2/data-expansion-01'
D08 = MAIN / 'docs/maintenance/2026-09-14/support-control-01/data'
CAPS = [2000, 1000, 1000, 2000, 1000, 1000]
ARMS = ['N', 'F', 'S', 'P', 'A']
SPLIT = 4000
WALL_CAP_S = 4 * 3600
VRAM_CAP = 12 * 2**30
ARTIFACT_CAP = 20 * 2**30


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for v in iter(lambda: f.read(2**20), b''):
            h.update(v)
    return h.hexdigest()


def save(p, v):
    with Path(p).open('x') as f:
        json.dump(v, f, indent=2, allow_nan=False)


def genseed(seq, i, j, k):
    return int(hashlib.sha256(f'F2|structured-motion|kitti|{seq}|{i}|{j}|{k}'.encode()).hexdigest()[:8], 16) % (2**31)


def datadir(seq):
    return D00 if seq == '00' else D08


def role(seq):
    return 'discovery' if seq == '00' else 'confirmation'


def prepare():
    """Verify immutable parents, fix every input identity, then freeze."""
    f00 = read(D00 / 'cohort_freeze.json')
    assert sha(D00 / 'discovery_inputs.json') == f00['hashes']['discovery_inputs.json']
    for name, h in read(D00 / 'artifact_hashes.json').items():
        assert sha(D00 / name) == h, name
    f08 = read(D08 / 'cohort_freeze.json')
    assert sha(D08 / 'confirmation_inputs.json') == f08['hashes']['confirmation_inputs.json']
    for name, h in read(D08 / 'artifact_hashes.json').items():
        assert sha(D08 / name) == h, name
    used = {}
    inputs = {}
    max_motion_delta = 0.0
    for seq in ['00', '08']:
        d = datadir(seq)
        meta = read(d / seq / 'data.json')
        odom = np.load(d / seq / 'odom.npy')
        rows = read(d / f'{role(seq)}_inputs.json')
        labels = read(d / f'{role(seq)}_evaluator.json')  # fixed-ID/stratum membership only; no outcomes exist yet
        assert len(rows) == 100 and len(labels) == 100
        from collections import Counter
        counts = Counter(r['stratum'] for r in labels)
        assert counts == {'confuser': 25, 'ambiguous_positive': 25, 'low_overlap': 25, 'ordinary': 25}, counts
        strata_order = [r['stratum'] for r in labels]
        assert [(r['source_i'], r['source_j']) for r in rows] == [(r['source_i'], r['source_j']) for r in labels]
        ordinary_keys = [f"{seq}_{r['source_i']}_{r['source_j']}" for r, s in zip(rows, strata_order) if s == 'ordinary']
        out_rows = []
        for r, st in zip(rows, strata_order):
            i, j = r['source_i'], r['source_j']
            key = f"{seq}_{i}_{j}"
            scans = []
            motions = []
            for side in ['i', 'j']:
                anchor = r['source_' + side]
                ss = r['support_' + side]
                assert len(ss) == 2 and len({s['source_id'] for s in ss}) == 2
                scans.append(dict(meta['scans'][anchor], source_id=anchor))
                motions.append(np.eye(4).tolist())
                for sp in ss:
                    k = sp['source_id']
                    assert k < anchor and 0 < sp['age_s'] <= 5
                    L = np.linalg.inv(odom[anchor]) @ odom[k]
                    delta = float(np.max(np.abs(L - np.asarray(sp['local_pose']))))
                    max_motion_delta = max(max_motion_delta, delta)
                    assert delta < 1e-8, (key, k, delta)
                    scans.append(dict(meta['scans'][k], source_id=k))
                    motions.append(sp['local_pose'])
            for x in scans:
                if x['path'] not in used:
                    h = sha(x['path'])
                    assert h == x['sha256'], x['path']
                    used[x['path']] = h
            out_rows.append(dict(key=key, sequence=seq, source_i=i, source_j=j,
                                 seeds=[genseed(seq, i, j, k) for k in range(4)],
                                 scans=scans, motions=motions))
        save(BASE / f'inputs_{seq}.json', out_rows)
        inputs[seq] = dict(pairs=len(out_rows), sensitivity_keys=ordinary_keys[:5])
    files = [BASE / 'PROTOCOL.md', BASE / 'synthetic_validation.json', Path(__file__),
             MAIN / 'scripts/maintenance/validate_f2_structured.py',
             ROOT / 'rap_slam/structured_motion.py', ROOT / 'rap_slam/rap_backend.py',
             ROOT / 'rap_slam/rap_precision.py', ROOT / 'rap_slam/measurement.py',
             ROOT / 'rap_slam/real_dataset.py', ROOT / 'scripts/hl0_preflight.py',
             ROOT / 'scripts/support_control.py']
    files += list((MAIN / 'rectified_point_flow').rglob('*.py'))
    files += list((MAIN / 'config').rglob('*.yaml'))
    files += list((MAIN / 'dataset_process').rglob('*.py'))
    sources = {str(p): sha(p) for p in files}
    for name in ['rap_model_12.ckpt', 'mini_spinnet_t.pth']:
        p = MAIN / 'weights' / name
        sources[str(p)] = sha(p)
    with zipfile.ZipFile(BASE / 'source_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, str(p.relative_to(ROOT.parent)))
    save(BASE / 'freeze.json', dict(
        utc=datetime.now(timezone.utc).isoformat(), protocol='F2-V1', sources=sources, raw=used,
        parents={str(p): sha(p) for p in [D00 / 'discovery_inputs.json', D00 / 'discovery_evaluator.json',
                                          D00 / 'cohort_freeze.json', D00 / 'artifact_hashes.json',
                                          D00 / '00/data.json', D00 / '00/odom.npy', D00 / '00/evaluator_gt.npy',
                                          D08 / 'confirmation_inputs.json', D08 / 'confirmation_evaluator.json',
                                          D08 / 'cohort_freeze.json', D08 / 'artifact_hashes.json',
                                          D08 / '08/data.json', D08 / '08/odom.npy', D08 / '08/evaluator_gt.npy']},
        inputs={s: sha(BASE / f'inputs_{s}.json') for s in ['00', '08']},
        sensitivity_keys={s: inputs[s]['sensitivity_keys'] for s in inputs},
        max_local_motion_recheck_delta=max_motion_delta,
        scope='200 observed-development pairs (100 KITTI00 + 100 KITTI08), K4, arms N/F/S/P/A; frozen before generation'))
    print('F2 frozen:', len(used), 'distinct raw scans, max local-motion recheck delta', max_motion_delta, flush=True)


def check_freeze():
    f = read(BASE / 'freeze.json')
    for group in ['sources', 'raw', 'parents']:
        for p, h in f[group].items():
            assert sha(p) == h, p
    for s, h in f['inputs'].items():
        assert sha(BASE / f'inputs_{s}.json') == h, s


def run(seq):
    import torch
    from rap_slam.rap_backend import RAPBackend
    from rap_slam.rap_precision import ieee_matmul
    from rap_slam.structured_motion import (build_batch, group_coordinates, project_groups,
                                            decode_groups, decode_parts, infer)
    from rap_slam.measurement import rotation_validity
    from rap_slam.real_dataset import kitti_vertical_correction
    from hl0_preflight import request_rng
    from rectified_point_flow.procrustes import fit_transformations
    from rectified_point_flow.eval.metrics import compute_rigidity_rmse
    check_freeze()
    if seq == '08':
        for p, h in read(BASE / '00_frozen.json').items():
            assert sha(BASE / '00' / p) == h, p
        prior = read(BASE / '00' / 'execution.json')['process_wall_s']
    else:
        prior = 0.0
    out = BASE / seq
    out.mkdir(exist_ok=False)
    (out / 'INCOMPLETE').write_text('Generation/evaluation/independent validation pending\n')
    rows = read(BASE / f'inputs_{seq}.json')
    sens_keys = set(read(BASE / 'freeze.json')['sensitivity_keys'][seq])
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    b = RAPBackend()
    assert b.device.type == 'cuda'
    provenance = read(MAIN / 'docs/maintenance/2026-09-14/hl-v2/pilot-01/discovery/provenance.json')
    assert sha(b.ckpt_path) == provenance['checkpoint_sha256']
    assert sha(b.spinnet_ckpt) == provenance['spinnet_sha256']
    assert b.max_points_per_scan == 4000 and b.voxel_size == .4
    g = RAPBackend.__new__(RAPBackend)
    g.voxel_size = .4; g.remove_outliers = True; g.max_points_per_scan = 4000; g.seed = 42; g.feature_extractor = None
    native_method = b.model.sample_rectified_flow

    def states():
        return (torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone(), random.getstate(),
                np.random.get_state(), b.seed, torch.get_float32_matmul_precision())

    def same_state(a, c):
        assert torch.equal(a[0], c[0]) and torch.equal(a[1], c[1]) and a[2] == c[2]
        assert a[3][0] == c[3][0] and np.array_equal(a[3][1], c[3][1]) and a[3][2:] == c[3][2:]
        assert a[4:] == c[4:]

    before = states()
    try:
        with request_rng(99, b), ieee_matmul():
            torch.rand(3, device=b.device); np.random.rand(); random.random()
            raise RuntimeError('intentional restoration test')
    except RuntimeError as e:
        assert str(e) == 'intentional restoration test'
    same_state(before, states())

    used = {}
    records = []
    gates = {}
    sens_records = []
    native_delta = None
    replay_delta = None

    def load_scan(entry):
        v = Path(entry['path']).read_bytes()
        h = hashlib.sha256(v).hexdigest()
        assert h == entry['sha256'], entry['path']
        used[entry['path']] = h
        p = np.frombuffer(v, dtype=np.float32).reshape(-1, 4)[:, :3]
        p = p[np.linalg.norm(p, axis=1) >= 1.5]
        assert len(p) and np.isfinite(p).all()
        return np.ascontiguousarray(kitti_vertical_correction(p), dtype=np.float32)

    def rigidity_steps(bb, traj):
        vals = []
        with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False), ieee_matmul():
            cu = bb['cu_seqlens'].to(torch.int32)
            for pred in traj.float():
                r_, t_ = fit_transformations(bb['pointclouds'], pred, bb['points_per_part'], cu)
                vals.append(float(compute_rigidity_rmse(bb['pointclouds'], pred, r_, t_,
                                                        bb['points_per_part'], cu, bb['scales']).item()))
        return vals

    def decode_record(arm, k, result, bb, cc, ss, wall, group=False, group_q=None, motions=None):
        """Decode one generation; enforced-geometry failure is an engineering stop."""
        raw = result['end_point_trajectory'][-1]
        arrays = {kk: vv.detach().float().cpu().numpy() for kk, vv in result.items()}
        if not all(np.isfinite(v).all() for v in arrays.values()):
            return dict(pair=key, arm=arm, generation=k, seed=r['seeds'][k], valid=False,
                        rigidity_rmse_m=None, T=None, wall_s=wall, constraint_matrix_residual=None)
        with ieee_matmul(), torch.inference_mode():
            raw_T = decode_parts(bb, raw, cc, ss)
            if group:
                final, gp = project_groups(raw, group_q, SPLIT)
                Tg = decode_groups(gp, cc, ss)
                fitted = decode_parts(bb, final, cc, ss)
                residual = max(float(np.max(np.abs(np.linalg.inv(fitted[0 if m < 3 else 3]) @ fitted[m] - motions[m])))
                               for m in range(6))
                assert residual <= 1e-4, ('constraint', key, arm, k, residual)
                T = np.linalg.inv(Tg[0]) @ Tg[1]
            else:
                final = result['projected_clean_trajectory'][-1]
                Tg = None
                residual = None
                T = np.linalg.inv(raw_T[0]) @ raw_T[3 if len(raw_T) == 6 else 1]
        poses = list(raw_T) + [T] + ([] if Tg is None else list(Tg))
        valid = all(rotation_validity(x)['valid'] for x in poses)
        rig = rigidity_steps(bb, result['end_point_trajectory'])
        np.savez_compressed(dest / f'{arm}{k}.npz',
                            end_point_trajectory=arrays['end_point_trajectory'],
                            projected_clean_trajectory=arrays['projected_clean_trajectory'],
                            trajectory=arrays['trajectory'],
                            decoded_final=final.detach().float().cpu().numpy(),
                            raw_poses=raw_T, group_poses=np.empty((0, 4, 4)) if Tg is None else Tg,
                            T=T, rigidity_steps=np.asarray(rig))
        return dict(pair=key, arm=arm, generation=k, seed=r['seeds'][k], valid=bool(valid),
                    rigidity_rmse_m=float(np.mean(rig)), T=T.tolist(), wall_s=wall,
                    constraint_matrix_residual=residual)

    def generate(arm, k, bb, noise_k, gsrc=None, split=None):
        state = states()
        torch.cuda.synchronize()
        t = time.monotonic()
        with request_rng(r['seeds'][k], b):
            result = infer(b, bb, noise_k, gsrc, split)
        torch.cuda.synchronize()
        wall = time.monotonic() - t
        same_state(state, states())
        assert b.model.sample_rectified_flow == native_method
        return result, wall

    for idx, r in enumerate(rows):
        if time.monotonic() - start + prior > WALL_CAP_S:
            raise RuntimeError('Four-hour GPU-process cap')
        key = r['key']
        dest = out / key
        dest.mkdir()
        scans = [load_scan(e) for e in r['scans']]
        state = states()
        with request_rng(r['seeds'][0], b), ieee_matmul(), torch.inference_mode():
            b.seed = 42
            points, features = [], []
            for p, cap in zip(scans, CAPS):
                x, f = b.preprocess(p, max_points=cap)
                assert len(x) == cap and np.isfinite(f).all() and np.any(f), ('allocation', key)
                points.append(x)
                features.append(f)
        same_state(state, states())
        L = np.asarray(r['motions'])
        batch, scale, centers, counts = build_batch(b, points, features)
        q = group_coordinates(points, L, centers, scale, b.device)
        noises = []
        for k in range(4):
            with request_rng(r['seeds'][k], b):
                noises.append(torch.randn_like(batch['pointclouds']))
        assert np.max(np.abs(batch['pointclouds'].cpu().numpy() * scale
                             + np.repeat(centers, counts, axis=0) - np.concatenate(points))) < 1e-4
        np.savez_compressed(dest / 'inputs.npz', points=np.concatenate(points), counts=counts,
                            cond=batch['pointclouds'].cpu().numpy(),
                            noise=torch.stack(noises).cpu().numpy(), centers=centers, scale=scale,
                            motions=L, group_source=q.cpu().numpy())
        transported = [p @ l[:3, :3].T + l[:3, 3] for p, l in zip(points, L)]
        pooled = [np.concatenate(transported[:3]).astype(np.float32),
                  np.concatenate(transported[3:]).astype(np.float32)]
        pb, ps, pc, pn = build_batch(b, pooled, [np.concatenate(features[:3]), np.concatenate(features[3:])])
        ab, ass, ac, an = build_batch(b, [points[0], points[3]], [features[0], features[3]])
        assert pn.tolist() == [4000, 4000] and an.tolist() == [2000, 2000]
        natives = {}
        order = ['N', 'S', 'P', 'A'] if idx % 2 == 0 else ['A', 'P', 'S', 'N']
        for arm in order:
            for k in range(4):
                if arm == 'N':
                    result, wall = generate('N', k, batch, noises[k])
                    natives[k] = result
                    if seq == '00' and idx == 0 and k == 0:
                        with request_rng(r['seeds'][0], b), ieee_matmul(), torch.inference_mode(), \
                                torch.autocast(device_type='cuda', dtype=torch.float16):
                            reference = b.model.sample_rectified_flow(batch, None, x_1=noises[0],
                                                                      return_tarjectory=True)
                        native_delta = max(float((result[kk] - reference[kk]).abs().max())
                                           for kk in ['end_point_trajectory', 'trajectory'])
                        assert native_delta <= 1e-6, ('native reproduction', native_delta)
                    rec = decode_record('N', k, result, batch, centers, scale, wall)
                    records.append(rec)
                    torch.cuda.synchronize()
                    t = time.monotonic()
                    with ieee_matmul(), torch.inference_mode():
                        project_groups(result['end_point_trajectory'][-1], q, SPLIT)
                    torch.cuda.synchronize()
                    fwall = time.monotonic() - t
                    records.append(decode_record('F', k, result, batch, centers, scale, fwall,
                                                 group=True, group_q=q, motions=L))
                elif arm == 'S':
                    result, wall = generate('S', k, batch, noises[k], q, SPLIT)
                    records.append(decode_record('S', k, result, batch, centers, scale, wall,
                                                 group=True, group_q=q, motions=L))
                elif arm == 'P':
                    result, wall = generate('P', k, pb, noises[k])
                    records.append(decode_record('P', k, result, pb, pc, ps, wall))
                else:
                    anoise = torch.cat([noises[k][:2000], noises[k][4000:6000]])
                    result, wall = generate('A', k, ab, anoise)
                    records.append(decode_record('A', k, result, ab, ac, ass, wall))
        if seq == '00' and idx == 0:
            replay, rwall = generate('S', 0, batch, noises[0], q, SPLIT)
            replay_delta = float((replay['end_point_trajectory'] - torch.from_numpy(
                np.load(dest / 'S0.npz')['end_point_trajectory']).cuda()).abs().max())
            assert replay_delta <= 1e-6, ('replay', replay_delta)
            save(dest / 'replay.json', dict(max_delta=replay_delta, wall_s=rwall))
        if key in sens_keys:
            for sign in [-1, 1]:
                a = np.deg2rad(sign)
                D = np.eye(4)
                D[:3, :3] = [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
                D[0, 3] = sign * .1
                pert = L.copy()
                for m in [1, 2, 4, 5]:
                    pert[m] = D @ L[m]
                pq = group_coordinates(points, pert, centers, scale, b.device)
                np.save(dest / ('motions-minus.npy' if sign < 0 else 'motions-plus.npy'), pert)
                for k in range(4):
                    result, wall = generate('S', k, batch, noises[k], pq, SPLIT)
                    rec = decode_record('Sminus' if sign < 0 else 'Splus', k, result, batch,
                                        centers, scale, wall, group=True, group_q=pq, motions=pert)
                    records.append(rec)
                    sens_records.append(rec)
        anchors = [g.preprocess(scans[0])[0], g.preprocess(scans[3])[0]]
        assert 200 <= len(anchors[0]) <= 4000 and 200 <= len(anchors[1]) <= 4000
        gates[key] = {}
        for rec in [x for x in records if x['pair'] == key and x['arm'] in ARMS]:
            if rec['valid']:
                ov, md = g._alignment_metrics(*anchors, np.asarray(rec['T']), thresh=1.)
                entry = dict(overlap_ratio=float(ov), mean_dist=float(md),
                             accepted=bool(ov >= .7 and md <= 2.5))
            else:
                entry = dict(overlap_ratio=None, mean_dist=None, accepted=False)
            gates[key].setdefault(rec['arm'], {})[str(rec['generation'])] = entry
        if idx % 10 == 0:
            print(seq, idx + 1, '/100 wall', round(time.monotonic() - start, 1), flush=True)
        if seq == '00' and idx == 9:
            save(out / 'preflight_validation.json', dict(
                passed=True, pairs=10, generations=160,
                native_max_delta=native_delta, replay_max_delta=replay_delta,
                records_so_far=len(records), no_gt_loaded=True))
        peak = torch.cuda.max_memory_allocated()
        assert peak <= VRAM_CAP, ('VRAM cap', peak)
    save(out / 'records.json', records)
    save(out / 'gates.json', gates)
    save(out / 'input_hashes.json', used)
    sens_summary = {}
    for signame in ['Sminus', 'Splus']:
        deltas = []
        for rec in sens_records:
            if rec['arm'] != signame or not rec['valid']:
                continue
            base = next(x for x in records if x['pair'] == rec['pair'] and x['arm'] == 'S'
                        and x['generation'] == rec['generation'])
            if not base['valid']:
                continue
            from rap_slam.measurement import transform_label
            lab = transform_label(np.asarray(rec['T']), np.asarray(base['T']))
            deltas.append(dict(pair=rec['pair'], generation=rec['generation'],
                               translation_m=lab['gt_translation_error_m'],
                               rotation_deg=lab['gt_rotation_error_deg']))
        sens_summary[signame] = deltas
    save(out / 'sensitivity.json', dict(
        scope='Fixed support-only +/-0.10m x / +/-1deg z perturbations, first 5 ordinary pairs, S arm K4; '
              'sensitivity observations only, no accuracy claim',
        pairs=sorted(sens_keys), deltas=sens_summary))
    peak_alloc = torch.cuda.max_memory_allocated()
    wall = time.monotonic() - start
    save(out / 'execution.json', dict(
        process_wall_s=wall, cumulative_wall_s=wall + prior, peak_allocated=peak_alloc,
        peak_reserved=torch.cuda.max_memory_reserved(), generations=sum(1 for x in records if x['arm'] in ARMS),
        sensitivity_generations=len(sens_records), native_max_delta=native_delta,
        replay_max_delta=replay_delta, no_hooks_installed=True, restoration_on_exception=True,
        gt_access='evaluator_only_after_generation_hash_freeze',
        torch=torch.__version__, gpu=torch.cuda.get_device_name(),
        completed_utc=datetime.now(timezone.utc).isoformat()))
    save(BASE / f'{seq}_frozen.json', {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*'))
                                       if p.is_file() and p.name != 'INCOMPLETE'})
    evaluate(seq)
    if seq == '08':
        save(BASE / 'artifact_hashes.json', {str(p.relative_to(BASE)): sha(p) for p in sorted(BASE.rglob('*'))
                                             if p.is_file() and p.name != 'INCOMPLETE'
                                             and p.name != 'artifact_hashes.json'})
    print('F2', seq, 'generation+evaluation complete; independent validation pending', flush=True)


def evaluate(seq):
    """Evaluator-only: loads GT and strata strictly after the generation bank freeze."""
    from rap_slam.measurement import transform_label
    from support_control import aggregate, STRATA
    out = BASE / seq
    records = read(out / 'records.json')
    gates = read(out / 'gates.json')
    labels = read(datadir(seq) / f'{role(seq)}_evaluator.json')
    gt = np.load(datadir(seq) / seq / 'evaluator_gt.npy')
    bypair = {}
    for rec in records:
        if rec['arm'] in ARMS:
            bypair.setdefault(rec['pair'], {}).setdefault(rec['arm'], {})[rec['generation']] = rec
    joined = []
    for row in labels:
        i, j = row['source_i'], row['source_j']
        key = f'{seq}_{i}_{j}'
        Tgt = np.linalg.inv(gt[i]) @ gt[j]
        methods = {}
        for arm in ARMS:
            gens = bypair[key][arm]
            labs = {}
            for k, rec in gens.items():
                if rec['valid']:
                    lab = transform_label(np.asarray(rec['T']), Tgt)
                    t, a = lab['gt_translation_error_m'], lab['gt_rotation_error_deg']
                    labs[k] = dict(RTE_m=t, RRE_deg=a, error=max(t, a / 5),
                                   clipped_error=min(5, max(t, a / 5)),
                                   strict=bool(t <= 1 and a <= 5), legacy_safe=lab['label'] == 'safe')
            good = [k for k, rec in gens.items() if rec['valid']]
            sel = min(good, key=lambda k: (gens[k]['rigidity_rmse_m'], k)) if good else None
            lab = labs[sel] if sel is not None else dict(RTE_m=None, RRE_deg=None, error=5.,
                                                         clipped_error=5., strict=False, legacy_safe=False)
            accepted = gates[key][arm][str(sel)]['accepted'] if sel is not None else False
            methods[arm] = dict(generation=sel, accepted=accepted,
                                incorrect_accepted=bool(accepted and not lab['strict']),
                                failed=sel is None, **lab)
            osel = min(labs, key=lambda k: (labs[k]['error'], k)) if labs else None
            olab = labs[osel] if osel is not None else dict(RTE_m=None, RRE_deg=None, error=5.,
                                                            clipped_error=5., strict=False, legacy_safe=False)
            oacc = gates[key][arm][str(osel)]['accepted'] if osel is not None else False
            methods[arm + '-OR4'] = dict(generation=osel, accepted=oacc,
                                         incorrect_accepted=bool(oacc and not olab['strict']),
                                         failed=osel is None, **olab)
        joined.append(dict(sequence=seq, source_i=i, source_j=j, stratum=row['stratum'], methods=methods))
    names = ARMS + [a + '-OR4' for a in ARMS]
    overall = {m: aggregate(joined, m) for m in names}
    strata = {s: {m: aggregate([r for r in joined if r['stratum'] == s], m) for m in names} for s in STRATA}
    conditions = {}
    for tag, other in [('C1_vs_N', 'N'), ('C2_vs_F', 'F'), ('C3_vs_P', 'P')]:
        conditions[tag + '_recall_3pp'] = overall['S']['selection_recall'] - overall[other]['selection_recall'] >= .03
        conditions[tag + '_error_5pct'] = overall['S']['mean_clipped_positive_error'] <= .95 * overall[other]['mean_clipped_positive_error']
    conditions['C4_no_more_incorrect_than_N'] = overall['S']['incorrect_accepted'] <= overall['N']['incorrect_accepted']
    conditions['C4_ordinary_loss_2pp'] = strata['ordinary']['N']['selection_recall'] - strata['ordinary']['S']['selection_recall'] <= .02
    seq_signal = all(conditions.values())
    save(out / 'evaluator_only_decisions.json', joined)
    save(out / 'metrics.json', dict(sequence=seq, role='observed_development', overall=overall,
                                    by_stratum=strata, conditions=conditions,
                                    sequence_signal=seq_signal,
                                    note='OR4 rows are evaluator-only diagnostics; success metrics are gate-independent '
                                         'strict recall of the common R4 selection; incorrect accepts use the common gate'))
    print(seq, 'sequence_signal', seq_signal, conditions, flush=True)
    if seq == '08':
        m00 = read(BASE / '00/metrics.json')
        e00 = read(BASE / '00/execution.json')
        e08 = read(out / 'execution.json')
        c5 = bool(e00['cumulative_wall_s'] <= WALL_CAP_S and e08['cumulative_wall_s'] <= WALL_CAP_S
                  and e00['peak_allocated'] <= VRAM_CAP and e08['peak_allocated'] <= VRAM_CAP
                  and e00['native_max_delta'] <= 1e-6 and e00['replay_max_delta'] <= 1e-6)
        both = m00['sequence_signal'] and seq_signal
        save(BASE / 'decision.json', dict(
            protocol='F2-V1', decision='DEVELOPMENT_MECHANISM_SIGNAL' if (both and c5) else 'NO_DEVELOPMENT_MECHANISM_SIGNAL',
            sequence_signals={'00': m00['sequence_signal'], '08': seq_signal},
            conditions={'00': m00['conditions'], '08': conditions, 'C5_engineering': c5},
            scope='Observed development cohorts only; a signal licenses only a separately frozen fresh-confirmation '
                  'protocol; no production/default change; HL/SC/PG stops and manuscript pause persist'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['prepare', '00', '08'])
    a = p.parse_args()
    prepare() if a.mode == 'prepare' else run(a.mode)
