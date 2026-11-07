"""Append-only candidate and generation evidence for offline verification."""
import json
from pathlib import Path
import numpy as np
from .measurement import make_measurement, stable_id
from .trajectory_protocol import validate_ids


class RecordingError(RuntimeError):
    """Evidence loss is fatal, never swallowed as a registration failure."""


class VerificationLog:
    def __init__(self, path, run_id, dataset, sequence, source_frame_ids, gt_poses=None,
                 provenance=None):
        self.path = Path(path)
        self.run_id, self.dataset, self.sequence = run_id, dataset, str(sequence)
        self.ids = validate_ids(source_frame_ids)
        self.gt = None if gt_poses is None else np.asarray(gt_poses)
        if self.gt is not None and self.gt.shape != (len(self.ids),4,4):
            raise ValueError('GT and source-frame manifest lengths differ')
        self.provenance = provenance or {}
        self.attempts = []
        self.measurements = {}
        # Refuse both old output files and attempts to reopen an active log.
        with self.path.open('x', encoding='utf-8') as f:
            f.write(json.dumps(dict(event='run',run_id=run_id,dataset=dataset,sequence=self.sequence,
                                    source_frame_ids=self.ids.tolist(),provenance=self.provenance),allow_nan=False)+'\n')

    def _write(self, event):
        try:
            with self.path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(event, allow_nan=False)+'\n')
                f.flush()
        except Exception as error:
            raise RecordingError('Cannot persist verification evidence') from error

    def start(self, i, j):
        if not (0 <= i < len(self.ids) and 0 <= j < len(self.ids)) or i == j:
            raise ValueError('Candidate indices outside source frame manifest')
        attempt=dict(attempt_id=stable_id([self.run_id,len(self.attempts)]),i=int(i),j=int(j),
                     source_frame_ids=[int(self.ids[i]),int(self.ids[j])],measurement_ids=[],
                     status='started')
        self._write(dict(event='candidate_started',**attempt))
        self.attempts.append(attempt)
        return attempt

    def generation(self, attempt, generation_id, transform, diag):
        if generation_id != len(attempt['measurement_ids']):
            raise RecordingError('Generation order is incomplete or duplicated')
        i,j=attempt['i'],attempt['j']
        gt=None if self.gt is None else np.linalg.inv(self.gt[i])@self.gt[j]
        generation=dict(T_rel=transform,overlap=diag['overlap_ratio'],mean_dist=diag['mean_dist'])
        record=make_measurement(self.run_id,self.dataset,self.sequence,attempt['source_frame_ids'],
                                generation_id,generation,gt,self.provenance,
                                invocation_id=attempt['attempt_id'])
        self._write(dict(event='generation',attempt_id=attempt['attempt_id'],measurement=record))
        attempt['measurement_ids'].append(record['measurement_id'])
        self.measurements[record['measurement_id']] = record
        if not record['labels']['validity']['valid']:
            raise ValueError('Nonrigid RAP transform; measurement logged, candidate failed')
        return record['measurement_id']

    def decision(self, attempt, selected_generation, accepted, gate, transform, diag):
        ids=attempt['measurement_ids']
        if not (0 <= selected_generation < len(ids)):
            raise RecordingError('Selected generation missing from recorded measurements')
        status='accepted' if accepted else 'rejected'
        chosen=ids[selected_generation]
        record=self.measurements[chosen]
        if (not np.array_equal(record['T_rel'],transform) or
                record['overlap'] != diag['overlap_ratio'] or
                record['mean_dist'] != diag['mean_dist']):
            raise RecordingError('Selected transform/diagnostics differ from recorded generation')
        self._write(dict(event='candidate_finished',attempt_id=attempt['attempt_id'],
                         status=status,selected_generation=selected_generation,
                         selected_measurement_id=chosen,gate=gate))
        attempt.update(status=status,selected_measurement_id=chosen)
        return chosen

    def failure(self, attempt, error):
        self._write(dict(event='candidate_finished',attempt_id=attempt['attempt_id'],status='failed',
                         error_type=type(error).__name__,error=str(error),
                         completed_measurement_ids=attempt['measurement_ids']))
        attempt['status']='failed'

    def summary(self):
        return {status:sum(a['status']==status for a in self.attempts)
                for status in ['started','accepted','rejected','failed']}
