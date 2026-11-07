"""M3 measurement identity, same-generation labels, historical features and splits."""
import hashlib
import json
import numpy as np


def stable_id(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def pair_id(dataset, sequence, source_frame_ids):
    ids = list(source_frame_ids)
    if len(ids) != 2 or any(not isinstance(i, (int, np.integer)) or i < 0 for i in ids) or ids[0] == ids[1]:
        raise ValueError('A physical pair requires two distinct nonnegative integer source frame IDs')
    return stable_id([dataset, str(sequence), sorted(map(int, ids))])


def best_generation(generations):
    if not generations:
        raise ValueError('No generations')
    for g in generations:
        if not np.isfinite([g['overlap'], g['mean_dist']]).all():
            raise ValueError('Nonfinite generation diagnostics')
        if not 0 <= g['overlap'] <= 1 or g['mean_dist'] < 0:
            raise ValueError('Invalid generation diagnostics')
    return max(range(len(generations)), key=lambda i: (generations[i]['overlap'], -generations[i]['mean_dist']))


def rotation_validity(transform, tolerance=1e-5):
    t = np.asarray(transform, dtype=float)
    if t.shape != (4, 4) or not np.isfinite(t).all():
        return {'valid': False, 'reason': 'nonfinite_or_malformed_transform'}
    r = t[:3, :3]
    orth = float(np.max(np.abs(r.T @ r - np.eye(3))))
    det = float(abs(np.linalg.det(r) - 1))
    valid = max(orth, det) <= tolerance and np.allclose(t[3], [0,0,0,1], atol=tolerance, rtol=0)
    return {'valid': bool(valid), 'orthogonality': orth, 'det_error': det,
            'reason': None if valid else 'nonrigid_transform'}


def transform_label(transform, gt_relative):
    """Pose correctness and place proximity are separate labels.

    atan2 rotation angle avoids acos sensitivity near identity after validity checks.
    GT text precision gets a 1e-4 tolerance; no historical estimate is projected.
    """
    validity = rotation_validity(transform)
    if not validity['valid']:
        return dict(label='unlabeled', reason=validity['reason'], validity=validity)
    if gt_relative is None or not rotation_validity(gt_relative, 1e-4)['valid']:
        return dict(label='unlabeled', reason='missing_or_invalid_GT', validity=validity)
    gt = np.asarray(gt_relative)
    error = np.linalg.inv(gt) @ np.asarray(transform)
    r = error[:3,:3]
    sine = np.linalg.norm([r[2,1]-r[1,2],r[0,2]-r[2,0],r[1,0]-r[0,1]]) / 2
    angle = float(np.degrees(np.arctan2(sine, np.clip((np.trace(r)-1)/2,-1,1))))
    te = float(np.linalg.norm(error[:3,3]))
    distance = float(np.linalg.norm(gt[:3,3]))
    return dict(label='safe' if te <= 5 and angle <= 10 else 'dangerous',
                gt_translation_error_m=te, gt_rotation_error_deg=angle,
                place_distance_m=distance,
                place_label='near' if distance <= 10 else 'far' if distance > 20 else 'gray',
                label_protocol='pose_error_5m_10deg_v2', validity=validity)


def make_measurement(run_id, dataset, sequence, source_frame_ids, generation_id,
                     generation, gt_relative, provenance, invocation_id=None):
    if not isinstance(run_id, str) or not run_id or not isinstance(generation_id, (int, np.integer)) or generation_id < 0:
        raise ValueError('Measurement requires a run ID and nonnegative integer generation ID')
    pid = pair_id(dataset, sequence, source_frame_ids)
    best_generation([generation])  # Validate this same observation's diagnostics.
    identity = dict(run_id=run_id, physical_pair_id=pid,
                    source_frame_ids=list(map(int,source_frame_ids)), generation_id=int(generation_id))
    if invocation_id is not None:
        identity["invocation_id"] = str(invocation_id)
    return dict(measurement_id=stable_id(identity), **identity, dataset=dataset,
                physical_sequence=f'{dataset}/{sequence}',
                T_rel=np.asarray(generation['T_rel']).tolist(),
                overlap=float(generation['overlap']), mean_dist=float(generation['mean_dist']),
                labels=transform_label(generation['T_rel'], gt_relative), provenance=provenance)


def historical_margin(candidate, query, keyframes, descriptors, site_exclusion=100,
                      temporal_separation=200):
    """Full reconstructed historical keyframe pool, not an inferred old shortlist."""
    from .scan_context import sc_distance, sc_distances_batch
    if candidate >= query:
        raise ValueError('Candidate must precede query for historical features')
    eligible = sorted(set(int(f) for f in keyframes if f < query and query-f >= temporal_separation and abs(f-candidate) >= site_exclusion and f != candidate))
    required = set(eligible) | {candidate,query}
    if required - descriptors.keys():
        raise ValueError('Incomplete historical descriptor pool; do not silently subset it')
    d1 = float(sc_distance(descriptors[candidate],descriptors[query])[0])
    if eligible:
        distances, _ = sc_distances_batch(np.stack([descriptors[f] for f in eligible]), descriptors[query])
        d2 = float(np.min(distances))
    else:
        d2 = None
    return dict(sc_d1=d1, sc_margin=None if d2 is None else d2-d1,
                sc_competitor_ids=eligible, sc_pool_id=stable_id(eligible),
                sc_feature_protocol='full_historical_keyframes_sep200_site100_v2',
                sc_missing_reason='no_distinct_historical_site' if d2 is None else None)


def grouped_split(records, seed='m3-v2', group_key='physical_pair_id'):
    """Deterministic label-independent 50/25/25 partition; no row-wise shuffle."""
    if group_key not in ('physical_pair_id','physical_sequence'):
        raise ValueError('Unsupported grouping')
    assignments = {}
    splits = dict(fit=[],calibration=[],test=[])
    for record in records:
        key = record[group_key]
        bucket = int(stable_id([seed,key])[:16],16) % 100
        split = 'fit' if bucket < 50 else 'calibration' if bucket < 75 else 'test'
        assignments[key] = split
        splits[split].append(record['measurement_id'])
    all_ids = [i for rows in splits.values() for i in rows]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError('Duplicate measurement identity')
    return dict(group_key=group_key,seed=seed,assignments=assignments,measurement_ids=splits,
                scope='within-sequence diagnostic; overlapping sites may remain' if group_key=='physical_pair_id' else 'physical sequence holdout')
