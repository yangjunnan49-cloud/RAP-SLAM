"""TR0 passive frozen-model token extraction; isolated from production defaults."""
import hashlib
import random
import numpy as np
import torch
from rap_slam.rap_precision import ieee_matmul


def snapshot():
    """Capture all runtime RNGs without drawing a sample."""
    ns = np.random.get_state()
    np_bytes = ns[0].encode() + ns[1].tobytes() + repr(ns[2:]).encode()
    return dict(cpu=torch.get_rng_state().cpu().numpy().copy(),
                cuda=torch.cuda.get_rng_state(0).cpu().numpy().copy(),
                python=hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
                numpy=hashlib.sha256(np_bytes).hexdigest())


def same_rng(a, b):
    return all(np.array_equal(a[k], b[k]) for k in ('cpu', 'cuda')) and all(a[k] == b[k] for k in ('python', 'numpy'))


def extract(model, batch, noise, export=False, hook=False):
    """Use existing sampler interface; hooks neither replace nor modify tensors."""
    assert not model.training and model.inference_sampling_steps == 5
    assert model.inference_sampler == 'euler' and model.rigidity_forcing
    assert model.timestep_sampling == 'u_shaped'
    assert model.flow_model.embed_dim == 512 and len(model.flow_model.transformer_layers) == 12
    versions = tuple((p._version, p.requires_grad) for p in model.parameters())
    modes = tuple(m.training for m in model.modules())
    trace = dict(inputs=[], times=[], last={})
    handles = []

    def pre(module, args, kw):
        assert not args
        trace['inputs'].append(kw['x'].detach().clone())
        trace['times'].append(kw['timesteps'].detach().clone())
        if len(trace['inputs']) == 5:
            trace['last'] = {k: kw[k].detach().clone() for k in
                ('cond_coord', 'local_features', 'scales', 'anchor_indices', 'cu_seqlens_batch', 'cu_seqlens_part')}
            assert kw['return_transformer_features'] is True

    def post(module, args, result):
        if len(trace['inputs']) == 5:
            assert 'tokens' not in trace
            trace['tokens'] = result.detach().float().clone()

    if hook:
        assert export
        handles.append(model.flow_model.register_forward_pre_hook(pre, with_kwargs=True))
        handles.append(model.flow_model.transformer_layers[-1].register_forward_hook(post))
    before = snapshot()
    try:
        with ieee_matmul(), torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            result = model.sample_rectified_flow(batch, None, x_1=noise,
                return_tarjectory=True, return_transformer_features=export)
        torch.cuda.synchronize()
        after = snapshot()
    finally:
        for handle in handles:
            handle.remove()
    assert versions == tuple((p._version, p.requires_grad) for p in model.parameters())
    assert modes == tuple(m.training for m in model.modules())
    native = result['trajectory'] if export else result
    assert set(native) == {'trajectory', 'end_point_trajectory'}
    arrays = {k: v.detach().float().cpu().numpy() for k, v in native.items()}
    if export:
        arrays['tokens'] = result['transformer_features'].detach().float().cpu().numpy().reshape(3, 1000, 512)
    if hook:
        assert len(trace['inputs']) == 5 and len(trace['times']) == 5
        arrays['hook_tokens'] = trace['tokens'].cpu().numpy().reshape(3, 1000, 512)
        arrays['flow_inputs'] = torch.stack(trace['inputs']).float().cpu().numpy()
        arrays['flow_times'] = torch.stack(trace['times']).float().cpu().numpy()
        for k, v in trace['last'].items():
            arrays['hook_' + k] = v.cpu().numpy()
    for when, state in [('before', before), ('after', after)]:
        for k in ('cpu', 'cuda'):
            arrays['rng_' + when + '_' + k] = state[k]
    assert all(np.isfinite(v).all() for v in arrays.values())
    return arrays, dict(before={k: before[k] for k in ('python', 'numpy')},
                        after={k: after[k] for k in ('python', 'numpy')},
                        hooked_flow_calls=len(trace['inputs']) if hook else None,
                        parameters_and_modes_unchanged=True)
