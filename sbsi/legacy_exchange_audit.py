"""Instrument a frozen sampler without changing its native-exchange decisions.

Loaded by file path beside the legacy package. No production sampler imports
this module. Exact replay comparison is mandatory before counts are trusted.
"""
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

COUNT_FIELDS = ('attempted', 'old_accepted', 'proposed_off_support',
    'old_accepted_off_support', 'proposed_target_into_zero',
    'old_accepted_target_into_zero', 'proposed_selected_neighbour_zero',
    'old_accepted_selected_neighbour_zero')


def file_digest(path):
    h = sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    def invalid(value):
        raise ValueError(f'nonfinite JSON value: {value}')
    return json.loads(Path(path).read_text(), parse_constant=invalid)


class LegacyExchangeAudit:
    """Pair old-target/new-neighbour helper calls within each actual mutation."""
    def __init__(self, observations, fixed_helper):
        self.observations = list(observations)
        self.fixed_helper = fixed_helper
        self.counts = {}
        self.frame = None
        self.proposal_calls = 0
        self.support_calls = 0
        self.last_state = self.last_populations = self.last_rng = None

    @contextmanager
    def instrument(self, target_module, exchange_module, population_class):
        original_helper = exchange_module.numerical_over_analytic
        original_proposal = exchange_module.propose_native_exchange
        original_alias = target_module.propose_native_exchange
        original_mutate = target_module.GalSBIParticleTarget.mutate
        original_population_mutate = population_class.mutate
        original_reweight = population_class.reweight_resample
        if original_alias is not original_proposal:
            raise ValueError('unexpected legacy proposal alias')

        def helper(prior, data):
            # Return this exact object; the fixed correction is diagnostic only.
            old = original_helper(prior, data)
            if self.frame is None or self.frame.get('calls') is None:
                raise ValueError('native correction outside an audited proposal')
            fixed, masks = self.fixed_helper(prior, data, return_support_audit=True)
            legacy = np.asarray(masks['legacy_finite'])
            below = np.asarray(masks['below_uniform_clamp'])
            if (legacy.shape != old.shape or below.shape != old.shape
                    or legacy.dtype != bool or below.dtype != bool
                    or not np.array_equal(legacy, np.isfinite(old))
                    or not np.array_equal(old[~below], fixed[~below])):
                raise ValueError('diagnostic does not reproduce legacy finite-path correction')
            self.frame['calls'].append((legacy.copy(), below.copy()))
            self.support_calls += 1
            return old

        def proposal(prior, scene, probability, rng, **kwargs):
            frame = self.frame
            if frame is None or frame['proposal_seen']:
                raise ValueError('unexpected or repeated proposal within mutation')
            if len(scene.counts) != len(frame['rows']):
                raise ValueError('legacy finite-prior subset no longer matches proposal')
            frame['proposal_seen'] = True
            frame['calls'] = []
            moved, ratio, active = original_proposal(prior, scene, probability, rng, **kwargs)
            slots = np.flatnonzero(scene.counts > 0)
            if not np.array_equal(active, scene.counts > 0):
                raise ValueError('legacy active exchange convention changed')
            calls = frame['calls']
            if len(calls) != (2 if len(slots) else 0):
                raise ValueError('expected paired old-target/new-neighbour corrections')
            if len(slots):
                (old_finite, old_below), (new_finite, new_below) = calls
                if old_finite.shape != slots.shape or new_finite.shape != slots.shape:
                    raise ValueError('paired native corrections lost row alignment')
                eligible = old_finite & new_finite & np.isfinite(ratio[slots])
                rows = frame['rows'][slots]
                frame['old_below'][rows] = eligible & old_below
                frame['new_below'][rows] = eligible & new_below
            frame['calls'] = None
            self.proposal_calls += 1
            return moved, ratio, active

        def mutate(target, state, beta, rng, **kwargs):
            block = kwargs.get('block')
            if block not in ('exchange', 'exchange_relocate'):
                return original_mutate(target, state, beta, rng, **kwargs)
            if self.frame is not None:
                raise ValueError('nested target mutation is unsupported')
            owner = np.asarray(target.owner)
            if (owner.shape != (len(state.x),) or np.any(owner < 0)
                    or np.any(owner >= len(self.observations))):
                raise ValueError('original observation ownership required')
            frame = dict(rows=np.flatnonzero(np.isfinite(state.log_prior)),
                proposal_seen=False, calls=None, old_below=np.zeros(len(owner), bool),
                new_below=np.zeros(len(owner), bool))
            self.frame = frame
            try:
                moved, accepted, active = original_mutate(target, state, beta, rng, **kwargs)
                if frame['proposal_seen'] != bool(len(frame['rows'])):
                    raise ValueError('missing legacy proposal instrumentation')
                if accepted.shape != owner.shape or active.shape != owner.shape:
                    raise ValueError('legacy decisions lost particle alignment')
                old, new = frame['old_below'], frame['new_below']
                violation = old | new
                for index in np.unique(owner):
                    use = owner == index
                    key = (float(beta), int(index), block)
                    row = self.counts.setdefault(key, dict.fromkeys(COUNT_FIELDS, 0))
                    values = dict(attempted=active, old_accepted=accepted,
                        proposed_off_support=violation, old_accepted_off_support=accepted & violation,
                        proposed_target_into_zero=old, old_accepted_target_into_zero=accepted & old,
                        proposed_selected_neighbour_zero=new,
                        old_accepted_selected_neighbour_zero=accepted & new)
                    for name, mask in values.items():
                        row[name] += int(np.count_nonzero(mask & use))
                return moved, accepted, active
            finally:
                self.frame = None

        def remember(populations, state, rng):
            # References only: no extra RNG use or changes to sampler objects.
            self.last_populations, self.last_state, self.last_rng = populations, state, rng

        def population_mutate(populations, state, beta, rng, **kwargs):
            result = original_population_mutate(populations, state, beta, rng, **kwargs)
            remember(populations, result[0], rng)
            return result

        def reweight(populations, state, delta, stage, rng):
            result = original_reweight(populations, state, delta, stage, rng)
            remember(populations, result[0], rng)
            return result

        exchange_module.numerical_over_analytic = helper
        target_module.propose_native_exchange = proposal
        target_module.GalSBIParticleTarget.mutate = mutate
        population_class.mutate = population_mutate
        population_class.reweight_resample = reweight
        try:
            yield self
        finally:
            exchange_module.numerical_over_analytic = original_helper
            target_module.propose_native_exchange = original_alias
            target_module.GalSBIParticleTarget.mutate = original_mutate
            population_class.mutate = original_population_mutate
            population_class.reweight_resample = original_reweight

    def records(self):
        return [dict(beta=beta, observation_index=index, observation=self.observations[index],
                     block=block, **values)
                for (beta, index, block), values in sorted(self.counts.items())]

    def validate_progress(self, progress):
        """Reconcile every stage/observation exchange count, including zero rows."""
        seen = set()
        for stage in progress['stages']:
            beta = float(stage['beta'])
            for block, values in stage['acceptance'].items():
                if block not in ('exchange', 'exchange_relocate'):
                    continue
                if (len(values['attempted']) != len(self.observations)
                        or len(values['accepted']) != len(self.observations)):
                    raise ValueError('stage counts have incompatible observation ownership')
                for index in range(len(self.observations)):
                    key = (beta, index, block)
                    row = self.counts.setdefault(key, dict.fromkeys(COUNT_FIELDS, 0))
                    if (row['attempted'] != values['attempted'][index]
                            or row['old_accepted'] != values['accepted'][index]):
                        raise ValueError(f'audited exchange counts disagree with full progress: {key}')
                    seen.add(key)
        if set(self.counts)-seen:
            raise ValueError('audited exchanges absent from complete saved progress')


def compare_complete_outputs(original, replay):
    """Fail closed on any scientific difference, missing payload, or unknown file.

    Only explicit timing, scheduler execution identifiers and their verified
    protocol digest may differ. Archive bytes AND every member's dtype, shape,
    and raw bytes must match. Original RNG artifacts, if present, are required.
    """
    original, replay = Path(original), Path(replay)
    required = {'protocol.json', 'progress.json', 'result.json', 'particles.npz', 'scenes.npz'}
    optional = {'initial_environment.npz', 'rng_state.json', 'final_rng_state.json'}
    found = {p.name for p in original.iterdir()}
    if not required <= found or found-(required | optional):
        raise ValueError(f'missing or unknown original artifacts: {sorted(found ^ required)}')
    if {p.name for p in replay.iterdir()} != found:
        raise ValueError('complete original/replay artifact sets differ')
    protocols = [read_json(p/'protocol.json') for p in (original, replay)]
    results = [read_json(p/'result.json') for p in (original, replay)]
    for root, result, protocol in zip((original, replay), results, protocols):
        if result.get('status') not in ('completed', 'completed_with_extinction'):
            raise ValueError('completed original and replay receipts required')
        for name in ('protocol', 'particles', 'scenes'):
            suffix = '.json' if name == 'protocol' else '.npz'
            if result[name+'_sha256'] != file_digest(root/(name+suffix)):
                raise ValueError(f'{root.name}: invalid recorded {name} artifact hash')
        informed = 'informed_environment' in protocol
        if informed != ('initial_environment.npz' in found):
            raise ValueError('informed initialization payload is missing or unexpected')
        if informed and (result['informed_environment_initialization']['sha256']
                         != file_digest(root/'initial_environment.npz')):
            raise ValueError('invalid initial environment artifact hash')
    arrays = {}
    native = ('type', 'z', 'M_B', 'c', 'R_kpc', 'n', 'a', 'psi', 'm_r', 'm_i', 'M_i')
    environment = ('log_tilt', 'valid', 'augmented', 'raw_coordinates', 'raw_types',
                   'raw_offsets', 'insertion_index', 'classification')
    required_arrays = {
        'particles.npz': {'coordinates', 'summaries', 'log_prior', 'log_kernel',
            'scores', 'ancestor_labels', 'extinct', 'log_normalizer'},
        'scenes.npz': {'indptr', 'offset_arcsec'} | {prefix+key
            for prefix in ('target_', 'neighbour_') for key in native}}
    if 'informed_environment' in protocols[0]:
        required_arrays['particles.npz'].update('environment_'+key for key in environment)
        required_arrays['initial_environment.npz'] = set(environment) | {'target_coordinates', 'log_potential'}
    for name in sorted(found):
        if not name.endswith('.npz'):
            continue
        with np.load(original/name, allow_pickle=False) as a, np.load(replay/name, allow_pickle=False) as b:
            if not required_arrays[name] <= set(a.files):
                raise ValueError(f'{name}: incomplete original scientific payload')
            if set(a.files) != set(b.files):
                raise ValueError(f'{name}: array payload fields differ')
            for key in a.files:
                left, right = a[key], b[key]
                if (left.dtype != right.dtype or left.shape != right.shape
                        or left.tobytes() != right.tobytes()):
                    raise ValueError(f'{name}:{key}: replay is not bitwise identical')
            arrays[name] = sorted(a.files)
        if file_digest(original/name) != file_digest(replay/name):
            raise ValueError(f'{name}: archive bytes differ despite matching members')
    normalized = []
    scheduler = []
    for protocol, result, root in zip(protocols, results, (original, replay)):
        protocol, result = deepcopy(protocol), deepcopy(result)
        runtime = protocol['runtime']
        scheduler.append({k: runtime.pop(k, None) for k in ('slurm_job_id', 'slurm_node_list')})
        result.pop('timing_seconds')
        result['final_stage'].pop('elapsed_seconds')
        result.pop('protocol_sha256')  # Verified above; scheduler identifiers differ.
        progress = read_json(root/'progress.json')
        if (progress['requested'] != protocol['stages']
                or len(progress['stages']) != protocol['stages']
                or [s['stage'] for s in progress['stages']] != list(range(1, protocol['stages']+1))
                or [s['beta'] for s in progress['stages']] != protocol['betas'][1:]):
            raise ValueError('complete original and replay stage histories required')
        for stage in progress['stages']:
            stage.pop('elapsed_seconds')
        normalized.append((protocol, result, progress))
    for index, name in enumerate(('protocol', 'result', 'complete stage history')):
        if normalized[0][index] != normalized[1][index]:
            raise ValueError(f'{name}: scientific replay metadata differs')
    rng_files = sorted(found & {'rng_state.json', 'final_rng_state.json'})
    for name in rng_files:
        if read_json(original/name) != read_json(replay/name):
            raise ValueError(f'{name}: original persisted RNG state differs')
    return dict(status='all_persisted_outputs_bitwise_equal', array_fields=arrays,
        artifacts={name: dict(original_sha256=file_digest(original/name),
                             replay_sha256=file_digest(replay/name)) for name in sorted(found)},
        scheduler_execution_identifiers=scheduler, original_rng_artifacts=rng_files,
        original_rng_verified=bool(rng_files),
        limits='Original drivers omit intermediate full states and final RNG; their absence is not filled by matching terminal states. This audit reruns every legacy transition and reconciles all saved stage history.')
