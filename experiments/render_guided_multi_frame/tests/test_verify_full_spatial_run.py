"""CPU artifact audits distinguish real feature refits from replay and no-ops."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


MODULE = Path(__file__).resolve().parents[1] / 'verify_full_spatial_run.py'
SPEC = importlib.util.spec_from_file_location('verify_full_spatial_run', MODULE)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class FullSpatialRunAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.pair = self.root / 'pairs' / '1_2'
        self.pair.mkdir(parents=True)
        (self.root / 'logs').mkdir()
        (self.root / 'logs' / '1_2_estimate.log').write_text('Fresh feature extraction and estimation\n')
        plane = dict(valid=True, normal=[0., -1., 0.], offset=1.5, camera_height=1.5, confidence=.8)
        gate = dict(enabled=True, applied=True, excluded_anchors=2, downweighted_anchors=1,
                    background_filtered=False, plane=plane)
        render = dict(optimized_parameter='object_logits', motion_probabilities_optimized=False,
                      iterations_completed=30, nonzero_gradient_steps=30, render_passes=64,
                      frames=[1, 2], weights=dict(height=.15, connectivity=.08),
                      spatial_constraints=dict(height=dict(enabled=True), connectivity=dict(enabled=True)),
                      initial_losses=dict(total=.3, height=.1), final_losses=dict(total=.2, height=.05))
        models = [dict(object_id=0, valid=True, source='local_0', height_gating_applied=False,
                       height_excluded_anchors=0, height_downweighted_anchors=0, height_gated_refit=False),
                  dict(object_id=1, valid=True, source='local_1', height_gating_applied=True,
                       height_excluded_anchors=2, height_downweighted_anchors=1, height_gated_refit=True),
                  dict(object_id=2, valid=False, height_gating_applied=True, height_gated_refit=False)]
        self.row = dict(first=1, second=2, result_path='pairs/1_2',
                        object_window=dict(feature_height_gate=gate, models=models, render_optimization=render))
        self.state = dict(frames=[1, 2], config=dict(joint=dict(freeze_motion_ownership=True),
                         object_window=dict(height=dict(enabled=True), connectivity=dict(enabled=True),
                                            render=dict(enabled=True))), pairs=[self.row])
        self._write_state()
        (self.pair / 'summary.json').write_text(json.dumps(dict(initial_matches=20, initial_3d_anchors=12)))
        self.q = np.array([[.9, .1], [.2, .8], [.6, .4]])
        self.raw_H = np.repeat(np.eye(4)[None], 2, axis=0)
        self.raw_H[1, 0, 3] = 1.
        self._write_motion(self.q)
        np.savez(self.pair / 'gaussian_state.npz', H=self.raw_H, q=self.q, local_to_object=[0, 1])
        object_H = np.concatenate((self.raw_H.copy(), np.full((1, 4, 4), np.nan)))
        object_H[1, 0, 3] = .9
        np.savez(self.pair / 'object_state.npz', gaussian_ids=[0, 1, 2], object_ids=[0, 1, 2],
                 H_object=object_H, object_model_valid=[True, True, False], q_motion_initial=self.q,
                 object_probabilities=np.column_stack((self.q, np.zeros(3))), ground_plane_valid=True,
                 ground_plane_normal=plane['normal'], ground_plane_offset=1.5, ground_plane_confidence=.8)

    def _write_state(self):
        (self.root / 'sequence_state.json').write_text(json.dumps(self.state))
        (self.pair / 'temporal_state.json').write_text(json.dumps(self.row))

    def _write_motion(self, final):
        np.savez(self.pair / 'pairwise_state.npz', H_final=self.raw_H, q_gaussian_sparse=self.q,
                 q_gaussian_joint=self.q, q_gaussian_final=final)

    def test_full_run_reports_filtering_refit_deltas_and_allows_missing_models(self):
        result = audit.verify(self.root)
        self.assertTrue(result['passed'], result['checks'])
        self.assertEqual(result['refitted_nonbackground_models'], 1)
        self.assertEqual(result['excluded_anchor_observations'], 2)
        self.assertEqual(result['downweighted_anchor_observations'], 1)
        delta = result['pairs'][0]['pose_comparisons'][1]['mapped_raw_motion_deltas'][0]
        self.assertAlmostEqual(delta['translation_m'], .1)
        self.assertTrue((self.root / 'full_spatial_verification.json').is_file())

    def test_replay_is_rejected_despite_valid_renderer_gradients(self):
        self.state['replay'] = True
        self._write_state()
        result = audit.verify(self.root)
        self.assertFalse(result['passed'])
        self.assertFalse(result['checks']['full_frontend_not_replay'])

    def test_supported_nonbackground_without_refit_is_rejected(self):
        self.row['object_window']['models'][1]['height_gated_refit'] = False
        self._write_state()
        result = audit.verify(self.root)
        self.assertFalse(result['passed'])
        self.assertFalse(result['pairs'][0]['checks']['supported_nonbackground_models_refitted'])

    def test_changed_local_motion_probabilities_are_rejected(self):
        changed = self.q.copy()
        changed[0] = [.5, .5]
        self._write_motion(changed)
        result = audit.verify(self.root)
        self.assertFalse(result['passed'])
        self.assertFalse(result['pairs'][0]['checks']['original_motion_probabilities_frozen'])

    def test_missing_fresh_estimation_log_is_rejected(self):
        (self.root / 'logs' / '1_2_estimate.log').unlink()
        result = audit.verify(self.root)
        self.assertFalse(result['passed'])
        self.assertFalse(result['pairs'][0]['checks']['frontend_artifacts_present'])


if __name__ == '__main__':
    unittest.main()
