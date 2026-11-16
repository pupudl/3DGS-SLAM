"""Bounded, camera-local endpoint links never label the intervening surface."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

EXPERIMENT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT))
from ownership.core_evidence_links import build_core_evidence_links
from ownership.object_connectivity_constraints import build_connectivity_state, update_connectivity_targets
from ownership.final_connectivity import finalize_connectivity
from scipy import sparse


def snapshot(uv=None, xyz=None, tracks=(10, 11, 12), colors=None):
    uv = np.column_stack((np.arange(7), np.zeros(7))) if uv is None else np.asarray(uv, float)
    n = len(uv)
    xyz = (np.column_stack((uv[:, 0]*.05, uv[:, 1]*.05, np.full(n, 5.)))
           if xyz is None else np.asarray(xyz, float))
    ids = np.full(n, -1)
    ids[[0, 3, 6]] = tracks
    costs = np.tile([.05, .9], (n, 1))
    costs[[0, 3, 6]] = [.9, .05]
    return dict(gaussian_ids=np.arange(n), object_ids=np.array([0, 1]),
                costs=costs, weights=np.ones((n, 2)), track_ids=ids,
                payload=dict(uv=uv, xyz=xyz, color=np.zeros((n, 3)) if colors is None else colors))


def window(*snapshots):
    return SimpleNamespace(_active_gids=np.arange(len(snapshots[0]['gaussian_ids'])),
                           _active={(i, i+1): s for i, s in enumerate(snapshots)})


def probe(current, **settings):
    return build_core_evidence_links(current, {1: np.array([0, 3, 6])},
                                     dict(core_evidence_links=settings))


class CoreEvidenceLinkTests(unittest.TestCase):
    def test_connects_reliable_endpoints_without_labeling_intermediate_background(self):
        snap = snapshot()
        before = snap['costs'].copy()
        result = probe(window(snap, snap))
        self.assertEqual(result['edges_by_object'][1].shape, (3, 2))
        self.assertTrue(np.all(np.isin(result['edges_by_object'][1], [0, 3, 6])))
        np.testing.assert_array_equal(snap['costs'], before)
        self.assertFalse(result['report']['intermediate_rows_assigned'])
        self.assertEqual(result['report']['objects']['1']['qualified_groups'][0]['independent_tracks'], 3)

    def test_same_track_cannot_certify_three_gaussians(self):
        result = probe(window(snapshot(tracks=(10, 10, 10))))
        self.assertEqual(len(result['edges_by_object'][1]), 0)

    def test_changing_track_ids_on_one_endpoint_does_not_inflate_independence(self):
        first = snapshot(tracks=(10, 10, 11))
        second = snapshot(tracks=(10, 10, 12))
        result = probe(window(first, second))
        self.assertEqual(len(result['edges_by_object'][1]), 0)
        self.assertEqual(result['report']['objects']['1']['local_groups'][0]['independent_tracks'], 2)

    def test_track_observation_without_positive_motion_evidence_does_not_certify(self):
        snap = snapshot()
        snap['costs'][3] = [.05, .9]
        result = probe(window(snap))
        self.assertEqual(len(result['edges_by_object'][1]), 0)

    def test_path_budget_counts_geometric_length_not_only_endpoint_chord(self):
        xyz = np.column_stack((np.arange(7)*.1, (np.arange(7) % 2)*.2, np.full(7, 5.)))
        result = probe(window(snapshot(xyz=xyz)), max_path_length_m=.5)
        self.assertEqual(len(result['edges_by_object'][1]), 0)

    def test_geometry_and_height_breaks_are_not_bypassed(self):
        snap = snapshot()
        snap['payload']['xyz'][2, 2] = 10.
        result = probe(window(snap))
        self.assertEqual(len(result['edges_by_object'][1]), 0)
        snap = snapshot()
        height = np.ones(7, bool)
        height[2] = False
        result = build_core_evidence_links(window(snap), {1: np.array([0, 3, 6])}, height_keep=height)
        self.assertEqual(len(result['edges_by_object'][1]), 0)

    def test_body_color_change_does_not_cut_geometric_path(self):
        color = np.tile([0., 1.], 4)[:7, None]*np.ones((7, 3))
        result = probe(window(snapshot(colors=color)))
        self.assertEqual(len(result['edges_by_object'][1]), 3)

    def test_distinct_camera_coordinate_origins_are_never_concatenated(self):
        first, second = snapshot(), snapshot()
        second['payload']['xyz'][:, 0] += 100.
        result = probe(window(first, second))
        self.assertEqual(len(result['edges_by_object'][1]), 3)

    def test_source_verified_edges_join_through_stable_ids_across_views(self):
        first, second = snapshot(), snapshot()
        first['payload']['xyz'][4, 2] = 10.  # Only the left pair has a local path.
        second['payload']['xyz'][2, 2] = 10.  # Only the right pair has a local path.
        result = probe(window(first, second))
        np.testing.assert_array_equal(result['edges_by_object'][1], [[0, 3], [3, 6]])
        self.assertEqual(result['report']['objects']['1']['qualified_groups'][0]['independent_tracks'], 3)

    def test_common_full_component_cannot_use_a_detour_outside_local_bbox(self):
        uv = [(0, 0), (0, 1), (0, 2), (3, 0), (3, 1), (3, 2), (6, 0)]
        uv += [(x, y) for x in (0, 3, 6) for y in range(3, 7)]
        uv += [(6, 1), (6, 2), (1, 6), (2, 6), (4, 6), (5, 6)]
        snap = snapshot(uv=uv)
        self.assertEqual(len(probe(window(snap), bbox_padding_px=4.)['edges_by_object'][1]), 0)
        self.assertEqual(len(probe(window(snap), bbox_padding_px=6.)['edges_by_object'][1]), 3)

    def test_ambiguous_rounded_binding_does_not_create_a_shortcut(self):
        snap = snapshot()
        snap['payload']['uv'][1] = snap['payload']['uv'][0]
        result = probe(window(snap))
        self.assertEqual(len(result['edges_by_object'][1]), 0)

    def test_roi_cannot_hide_an_ambiguous_binding_just_outside_its_boundary(self):
        uv = np.vstack((np.column_stack((np.arange(7), np.zeros(7))), [[6.4, 0.]]))
        snap = snapshot(uv=uv)
        result = probe(window(snap), bbox_padding_px=0.)
        # Pixel 6 is ambiguous in the full image. Cropping at x=6 must not
        # discard its x=6.4 counterpart and turn that core into a unique node.
        self.assertEqual(len(result['edges_by_object'][1]), 0)

    def test_far_apart_core_chain_does_not_exceed_cluster_pixel_bound(self):
        snap = snapshot()
        result = probe(window(snap), max_pixel_distance=4.)
        # Adjacent core pairs are 3px apart, but their 6px chain is not a 4px cluster.
        self.assertEqual(len(result['edges_by_object'][1]), 0)

    def test_object_edges_remain_separate(self):
        snap = snapshot()
        snap['object_ids'] = np.array([0, 1, 2])
        snap['costs'] = np.column_stack((snap['costs'], np.full(7, 2.)))
        snap['weights'] = np.ones((7, 3))
        result = build_core_evidence_links(window(snap), {1: [0, 3, 6], 2: [0, 3, 6]})
        self.assertEqual(len(result['edges_by_object'][1]), 3)
        self.assertEqual(len(result['edges_by_object'][2]), 0)


class CoreEvidenceLinkIntegrationTests(unittest.TestCase):
    def test_four_fragmented_reliable_candidates_become_a_linked_core(self):
        snap = snapshot()
        candidates = np.array([0, 2, 4, 6])
        snap['costs'][:] = [.05, .9]
        snap['costs'][candidates] = [.9, .05]
        snap['track_ids'][:] = -1
        snap['track_ids'][[0, 4, 6]] = [10, 11, 12]
        current = window(snap, snap, snap)
        current.object_ids = np.array([0, 1])
        before = build_connectivity_state(current)
        self.assertEqual(before['diagnostics']['objects']['1']['action'], 'skip_fragmented_core')
        self.assertEqual(before['diagnostics']['objects']['1']['candidate_count'], 4)
        linked = build_connectivity_state(current, dict(core_evidence_links=dict(enabled=True)))
        np.testing.assert_array_equal(linked['core_by_object'][1], candidates)
        self.assertEqual(linked['diagnostics']['objects']['1']['action'], 'constrain_linked_core')
        self.assertEqual(len(linked['core_link_edges'][1]), 6)
        np.testing.assert_array_equal(linked['object_core_groups'][1][0], candidates)
        # The ordinary graph still has only adjacent source-pixel edges.
        self.assertEqual(linked['adjacency'][0, 2], 0.)
        self.assertGreater(linked['adjacency_by_object'][1][0, 2], 0.)
        self.assertEqual(before['adjacency'].nnz, linked['adjacency'].nnz)

    def state(self):
        n = 7
        left, right = np.arange(n-1), np.arange(1, n)
        base = sparse.csr_matrix((np.ones(2*len(left)),
                                  (np.r_[left, right], np.r_[right, left])), shape=(n, n))
        endpoint = np.array([[0, 2], [2, 4], [4, 6]])
        extra = sparse.csr_matrix((np.ones(2*len(endpoint)),
                                   (np.r_[endpoint[:, 0], endpoint[:, 1]],
                                    np.r_[endpoint[:, 1], endpoint[:, 0]])), shape=(n, n))
        return dict(active_gids=np.arange(n), object_ids=np.array([0, 1, 2]),
                    adjacency=base, edges=np.column_stack((left, right)),
                    adjacency_by_object={1: base.maximum(extra)},
                    observed=np.ones((n, 3), bool), compatible=np.ones((n, 3), bool),
                    height_keep=np.ones(n, bool), mean_advantage=np.ones((n, 3)),
                    core_by_object={1: np.array([0, 2, 4, 6]), 2: np.array([0, 2, 4, 6])},
                    options={}, diagnostics=dict(objects={
                        '0': dict(action='background_exempt'),
                        '1': dict(action='constrain_linked_core'), '2': dict(action='constrain')}))

    def test_other_object_cannot_borrow_endpoint_links_in_soft_or_final_graphs(self):
        state = self.state()
        p = np.tile([.99, .005, .005], (7, 1))
        p[[0, 2, 4, 6]] = [.005, .005, .99]
        result = finalize_connectivity(state, p, assignment_support=np.ones_like(p, bool),
                                       min_probability=.6, min_margin=.1)
        self.assertTrue(result['disabled_columns'][2])
        self.assertTrue(np.all(result['labels'] == 0))
        state['core_by_object'][2] = np.array([0])
        targets = update_connectivity_targets(state, p)
        self.assertTrue(np.all(targets['penalty'][[2, 4, 6], 2] > 0))

    def test_background_core_cannot_bridge_three_remaining_owned_cores(self):
        state = self.state()
        p = np.tile([.99, .005, .005], (7, 1))
        p[[0, 4, 6]] = [.005, .99, .005]
        # GID 2 is the old bridge between GID 0 and the GID 4/6 group, but
        # it now belongs to background. Three owned anchors alone are not
        # enough to invent a path through that stale background core.
        result = finalize_connectivity(state, p, assignment_support=np.ones_like(p, bool),
                                       min_probability=.6, min_margin=.1)
        self.assertTrue(result['disabled_columns'][1])
        self.assertTrue(np.all(result['labels'] == 0))

    def test_certified_group_retains_three_owned_endpoints_after_tree_hub_becomes_background(self):
        state = self.state()
        state['object_core_groups'] = {1: [np.array([0, 2, 4, 6])]}
        p = np.tile([.99, .005, .005], (7, 1))
        p[[0, 4, 6]] = [.005, .99, .005]
        result = finalize_connectivity(state, p, assignment_support=np.ones_like(p, bool),
                                       min_probability=.6, min_margin=.1)
        np.testing.assert_array_equal(result['labels'], [1, 0, 0, 0, 1, 0, 1])
        self.assertFalse(result['disabled_columns'][1])
        group = result['report']['objects']['1']['structural_owned_core_groups'][0]
        self.assertEqual(group['owned_gaussian_ids'], [0, 4, 6])
        self.assertEqual(group['owned_core_count'], 3)
        self.assertFalse(group['intermediate_rows_assigned'])
        # Physical path interiors and the old tree hub remain background.
        self.assertTrue(np.all(result['labels'][[1, 2, 3, 5]] == 0))

    def test_unknown_member_never_participates_in_certified_group_edge(self):
        state = self.state()
        state['object_core_groups'] = {1: [np.array([0, 2, 4, 6])]}
        p = np.tile([.99, .005, .005], (7, 1))
        p[[0, 4, 6]] = [.005, .99, .005]
        p[2] = [.445, .55, .005]
        result = finalize_connectivity(state, p, assignment_support=np.ones_like(p, bool),
                                       min_probability=.6, min_margin=.1)
        self.assertEqual(result['labels'][2], -1)
        group = result['report']['objects']['1']['structural_owned_core_groups'][0]
        self.assertEqual(group['owned_gaussian_ids'], [0, 4, 6])

    def test_certified_group_cannot_save_an_object_with_only_two_owned_cores(self):
        state = self.state()
        state['object_core_groups'] = {1: [np.array([0, 2, 4, 6])]}
        p = np.tile([.99, .005, .005], (7, 1))
        p[[0, 6]] = [.005, .99, .005]
        result = finalize_connectivity(state, p, assignment_support=np.ones_like(p, bool),
                                       min_probability=.6, min_margin=.1)
        self.assertTrue(result['disabled_columns'][1])
        self.assertTrue(np.all(result['labels'] == 0))

    def test_another_object_cannot_borrow_a_certified_structural_group(self):
        state = self.state()
        state['object_core_groups'] = {1: [np.array([0, 2, 4, 6])]}
        p = np.tile([.99, .005, .005], (7, 1))
        p[[0, 2, 4, 6]] = [.005, .005, .99]
        result = finalize_connectivity(state, p, assignment_support=np.ones_like(p, bool),
                                       min_probability=.6, min_margin=.1)
        self.assertTrue(result['disabled_columns'][2])
        self.assertTrue(np.all(result['labels'] == 0))


if __name__ == '__main__':
    unittest.main()
