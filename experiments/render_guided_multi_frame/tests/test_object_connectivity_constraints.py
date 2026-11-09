"""CPU checks for evidence-selected cores and revisable connectivity losses."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


_path = Path(__file__).resolve().parents[1] / 'object_connectivity_constraints.py'
_spec = importlib.util.spec_from_file_location('object_connectivity_constraints', _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
build_connectivity_state = _module.build_connectivity_state
update_connectivity_targets = _module.update_connectivity_targets


def snapshot(ids, uv, costs, tracks=None, xyz=None, colors=None):
    ids = np.asarray(ids, np.int64)
    uv = np.asarray(uv, float)
    xyz = (np.column_stack((uv[:, 0]*.1, uv[:, 1]*.1, np.full(len(ids), 5.)))
           if xyz is None else np.asarray(xyz, float))
    return dict(gaussian_ids=ids, object_ids=np.array([0, 1]),
                costs=np.asarray(costs, float), weights=np.ones((len(ids), 2)),
                track_ids=np.full(len(ids), -1) if tracks is None else np.asarray(tracks),
                payload=dict(uv=uv, xyz=xyz,
                             color=np.zeros((len(ids), 3)) if colors is None else np.asarray(colors)))


def window(snapshots):
    gids = np.unique(np.concatenate([snap['gaussian_ids'] for snap in snapshots]))
    return SimpleNamespace(_active_gids=gids, object_ids=[0, 1],
                           _active={(i, i+1): snap for i, snap in enumerate(snapshots)})


def scene(background_count=12, contiguous=False, untracked_background_core=False):
    count = 3+background_count
    x = np.r_[np.arange(3), np.arange(background_count)+(3 if contiguous else 10)]
    uv = np.column_stack((x, np.zeros(count)))
    costs = np.vstack((np.tile([.9, .05], (3, 1)),
                       np.tile([.99, .01] if untracked_background_core else [.05, .9],
                               (background_count, 1))))
    tracks = np.r_[np.arange(3), np.full(background_count, -1)]
    snap = snapshot(np.arange(count), uv, costs, tracks)
    copies = 3 if untracked_background_core else 2
    return window([snap]*copies)


class ObjectConnectivityTests(unittest.TestCase):
    def test_large_wrong_background_block_is_not_selected_by_probability_or_area(self):
        state = build_connectivity_state(scene(background_count=40))
        np.testing.assert_array_equal(state['core_by_object'][1], [0, 1, 2])
        p = np.tile([.01, .99], (43, 1))
        result = update_connectivity_targets(state, p)
        np.testing.assert_array_equal(result['penalty'][:3, 1], 0)
        self.assertTrue(np.all(result['penalty'][3:, 1] > 0))
        self.assertFalse(result['report']['forces_background_assignment'])

    def test_actual_track_motion_support_wins_over_larger_untracked_false_island(self):
        state = build_connectivity_state(scene(40, untracked_background_core=True))
        np.testing.assert_array_equal(state['core_by_object'][1], [0, 1, 2])
        self.assertEqual(state['diagnostics']['objects']['1']['rejected_core_candidates'], 40)

    def test_background_is_exempt_from_connectivity_and_anchors(self):
        state = build_connectivity_state(scene())
        result = update_connectivity_targets(state, np.tile([.05, .95], (15, 1)))
        np.testing.assert_array_equal(result['penalty'][:, 0], 0)
        self.assertFalse(np.any(result['seed_columns'] == 0))
        self.assertEqual(result['report']['objects']['0']['action'], 'background_exempt')

    def test_shared_id_bridges_partial_views_without_cross_camera_xyz_distance(self):
        first = snapshot([0, 1, 2], [[0, 0], [1, 0], [2, 0]], [[.9, .05]]*3,
                         tracks=[10, 11, 12])
        # Different camera coordinate origins must never be concatenated into
        # the metric graph. A shared visible ID is the temporal bridge.
        last = snapshot([2, 3, 4], [[0, 0], [1, 0], [2, 0]], [[.9, .05]]*3,
                        xyz=[[100., 0, 5], [100.1, 0, 5], [100.2, 0, 5]])
        state = build_connectivity_state(window([first, first, last]))
        result = update_connectivity_targets(state, np.tile([.05, .95], (5, 1)))
        np.testing.assert_array_equal(result['penalty'][:, 1], 0)
        self.assertEqual(result['report']['objects']['1']['protected_count'], 5)
        self.assertTrue(np.any(np.all(state['edges'] == [2, 3], axis=1)))

    def test_no_reliable_core_does_not_delete_or_penalize_any_object(self):
        original = scene()
        state = build_connectivity_state(window([next(iter(original._active.values()))]))
        self.assertNotIn(1, state['core_by_object'])
        result = update_connectivity_targets(state, np.tile([.01, .99], (15, 1)))
        np.testing.assert_array_equal(result['penalty'], 0)
        self.assertEqual(len(result['seed_rows']), 0)
        self.assertEqual(result['report']['objects']['1']['action'], 'skip_no_reliable_core')

    def test_unobserved_high_probability_island_is_not_declared_disconnected(self):
        current = scene(background_count=3)
        for snap in current._active.values():
            snap['costs'][3:] = np.nan
        state = build_connectivity_state(current)
        result = update_connectivity_targets(state, np.tile([.01, .99], (6, 1)))
        np.testing.assert_array_equal(result['penalty'][3:], 0)

    def test_car_road_weak_bridge_is_rejected_by_observed_motion_compatibility(self):
        state = build_connectivity_state(scene(background_count=8, contiguous=True))
        self.assertTrue(np.any(np.all(state['edges'] == [2, 3], axis=1)))
        result = update_connectivity_targets(state, np.tile([.01, .99], (11, 1)))
        np.testing.assert_array_equal(result['penalty'][:3, 1], 0)
        self.assertTrue(np.all(result['penalty'][3:, 1] > 0))

    def test_depth_discontinuity_blocks_edges_but_body_color_change_is_soft(self):
        uv = np.column_stack((np.arange(6), np.zeros(6)))
        xyz = np.column_stack((np.arange(6)*.1, np.zeros(6), [5, 5, 5, 10, 10, 10]))
        color = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]]*2)
        snap = snapshot(np.arange(6), uv, [[.9, .05]]*6, tracks=np.arange(6), xyz=xyz, colors=color)
        state = build_connectivity_state(window([snap, snap]))
        self.assertTrue(np.any(np.all(state['edges'] == [0, 1], axis=1)))
        self.assertFalse(np.any(np.all(state['edges'] == [2, 3], axis=1)))
        self.assertGreater(float(state['adjacency'][0, 1]), 0.)

    def test_known_impossible_height_rejects_core_but_missing_plane_does_not(self):
        current = scene()
        height = np.full(15, 10.)
        state = build_connectivity_state(current, height_info=dict(height=height, valid=np.ones(15, bool)))
        self.assertNotIn(1, state['core_by_object'])
        state = build_connectivity_state(current, height_info=dict(height=height, valid=np.zeros(15, bool)))
        np.testing.assert_array_equal(state['core_by_object'][1], [0, 1, 2])
        state = build_connectivity_state(current, height_info=dict(
            height=np.ones(15), valid=np.ones(15, bool), allowed=np.zeros(15, bool)))
        self.assertNotIn(1, state['core_by_object'])

    def test_inherited_labels_and_init_are_not_motion_evidence(self):
        current = scene()
        for snap in current._active.values():
            snap['object_labels'] = np.ones(15, int)
            snap['q_init'] = np.tile([.001, .999], (15, 1))
            snap['costs'][:] = [.2, .2]
        state = build_connectivity_state(current)
        self.assertFalse(state['core_by_object'])

    def test_targets_are_differentiable_without_mutating_probabilities_or_snapshot(self):
        import torch
        current = scene()
        before = next(iter(current._active.values()))['costs'].copy()
        state = build_connectivity_state(current)
        logits = torch.zeros((15, 2), requires_grad=True)
        probabilities = torch.softmax(logits, dim=1)
        targets = update_connectivity_targets(state, probabilities.detach().numpy())
        losses = _module.connectivity_losses(probabilities, targets)
        (losses['disconnected']+losses['anchor']).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.all(logits.grad[:3, 1] < 0))
        self.assertTrue(torch.all(logits.grad[3:, 1] > 0))
        np.testing.assert_array_equal(next(iter(current._active.values()))['costs'], before)
        np.testing.assert_allclose(probabilities.detach().numpy(), .5)

    def test_invalid_probability_input_is_rejected(self):
        state = build_connectivity_state(scene())
        with self.assertRaises(ValueError):
            update_connectivity_targets(state, np.ones((15, 2)))


if __name__ == '__main__':
    unittest.main()
