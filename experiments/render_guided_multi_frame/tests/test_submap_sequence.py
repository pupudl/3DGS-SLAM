"""Sequence-entry integration for fixed submaps and shared object lifecycles."""
import sys
from pathlib import Path

_EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(_EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_ROOT))

import json
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import test_streaming_sequence as synthetic


class SubmapSequenceTests(unittest.TestCase):
    def fixture(self, active_budget=8):
        fixture = synthetic.StreamingSequenceTests()
        fixture.setUp()
        self.addCleanup(fixture.temporary.cleanup)
        fixture.cfg['gaussian_map'] = dict(
            enabled=True, observation_stride=1, initial_scale_multiplier=1.,
            association_distance=.2, voxel_size=.08, min_probability=.6,
            max_active_gaussians=active_budget, max_samples_per_frame=16,
            max_new_gaussians_per_frame=8, checkpoint_interval=3,
            snapshot_scope='full_visible_map',
            submaps=dict(enabled=True, frames=12, overlap_frames=1,
                         max_boundary_landmarks=16),
            active_selection=dict(enabled=True, tile_size_px=16),
            optimization=dict(backend='torch_reference', device='cpu', iterations=1,
                              downsample=1, verbose=False))
        return fixture

    def test_real_sequence_entry_archives_submaps_and_preserves_global_objects(self):
        fixture = self.fixture()
        output = fixture.root / 'submap_sequence'
        from mapping.gaussian_submaps import SubmapGaussianMapBackend
        managers = []

        def create_manager(*args, **kwargs):
            manager = SubmapGaussianMapBackend(*args, **kwargs)
            managers.append(manager)
            return manager

        with patch('frontend.strict_two_frame.run', side_effect=fixture.frontend), \
                patch('mapping.gaussian_submaps.SubmapGaussianMapBackend', side_effect=create_manager):
            state = synthetic.runner.run_sequence(fixture.cfg, list(range(36)), output)
        self.assertEqual(state['status'], 'complete')
        self.assertEqual(len(state['pairs']), 35)
        self.assertEqual(len(managers), 1)
        self.assertIsNone(managers[0].backend)
        self.assertIsNone(managers[0]._boundary)
        report = state['gaussian_map']
        self.assertTrue(report['finalized'])
        self.assertEqual(report['updates'], 35)
        self.assertEqual(report['resident_submaps'], 0)
        manifest = json.loads((output/'gaussian_map/submaps/manifest.json').read_text())
        entries = manifest['submaps']
        self.assertEqual([entry['frame_ids'] for entry in entries],
                         [list(range(0, 12)), list(range(11, 23)),
                          list(range(22, 34)), list(range(33, 36))])
        self.assertEqual([entry['partial'] for entry in entries], [False, False, False, True])
        self.assertTrue(all(entry['state'] == 'closed' for entry in entries))
        self.assertTrue(all(entry['max_total_gaussians'] >= 3000000 for entry in entries))
        self.assertEqual(manifest['resident_submaps'], 0)
        self.assertTrue(manifest['global_object_ids'])
        self.assertFalse(manifest['object_lifecycle_resets_at_boundary'])
        dynamic = sorted((record for oid, record in state['objects'].items()
                          if oid != '0' and record['confirmed']),
                         key=lambda record: record['first_observed_frame'])
        self.assertEqual(len(dynamic), 2)
        first, second = dynamic
        self.assertEqual((first['first_observed_frame'], first['last_observed_frame']), (10, 16))
        self.assertEqual((second['first_observed_frame'], second['last_observed_frame']), (22, 26))
        self.assertEqual(first['status'], 'retired')
        self.assertEqual(second['status'], 'retired')
        self.assertNotEqual(first['object_id'], second['object_id'])
        self.assertEqual(entries[0]['object_status_at_close'][str(first['object_id'])], 'active')
        self.assertEqual(manifest['objects'][str(first['object_id'])]['status'], 'retired')
        from mapping.persistent_gaussian_map import PersistentGaussianMap
        for entry in entries[:2]:
            store = PersistentGaussianMap.load((output/entry['checkpoint']).parent)
            self.assertEqual(len(store.ids_for_object(first['object_id'])), 4)
        trajectory = [json.loads(line) for line in (output/'gaussian_map/trajectory.jsonl').read_text().splitlines()]
        self.assertEqual(len(trajectory), 35)
        for previous, current in zip(trajectory, trajectory[1:]):
            np.testing.assert_allclose(previous['camera_to_world_second'], current['camera_to_world_first'], atol=1e-6)
            if previous['submap_id'] != current['submap_id']:
                shared = previous['object_to_world_second'].keys() & current['object_to_world_first'].keys()
                for owner in shared:
                    np.testing.assert_allclose(previous['object_to_world_second'][owner],
                                               current['object_to_world_first'][owner], atol=1e-6)
        seed = json.loads((output/'gaussian_map/submaps/submap_000001/boundary_seed.json').read_text())
        self.assertEqual(seed['frame'], 11)
        self.assertGreaterEqual(seed['tracked_anchor_count'], 4)
        from tools.visualization.visualize_gaussian_map import discover_map_snapshots
        snapshots = discover_map_snapshots(output)
        self.assertEqual(len(snapshots), 36)
        shared = next(path for path in snapshots if path.name == 'frame_0000000011.npz')
        self.assertEqual(shared.parent.parent.name, 'submap_000001')
        for frame, present in ((11, True), (16, True), (17, False)):
            snapshot = next(path for path in snapshots if path.name == f'frame_{frame:010d}.npz')
            with np.load(snapshot) as saved:
                self.assertEqual(first['object_id'] in saved['object_ids'], present)
                self.assertEqual(str(saved['snapshot_scope']), 'full_visible_map')
        for pair in state['pairs']:
            local = json.loads((output/pair['result_path']/'temporal_state.json').read_text())['gaussian_map']
            self.assertLessEqual(local['active_map_gaussians'], 8)
            self.assertIn('active_selection', local)
        scene = json.loads((output/'gaussian_map/global_scene.json').read_text())
        self.assertEqual(scene['format'], 'global_gaussian_submap_scene_v1')
        self.assertEqual(len(scene['chunks']), 4)
        self.assertEqual({row['object_id'] for row in scene['dynamic_objects']},
                         {first['object_id'], second['object_id']})
        from tools.validation.verify_streaming_run import verify
        from tools.validation.verify_gaussian_map_run import verify as verify_map
        streaming = verify(output)
        self.assertTrue(streaming['passed'], streaming)
        mapping = verify_map(output)
        self.assertTrue(mapping['passed'], mapping)
        images = fixture.root/'dataset/data_2d_raw/synthetic/image_00/data_rect'
        images.mkdir(parents=True)
        for frame in range(36):
            cv2.imwrite(str(images/f'{frame:010d}.png'), fixture.frame(frame)[1])
        from tools.visualization.visualize_objects import visualize
        visual = visualize(output, width=64)
        self.assertEqual(len(visual['scene']['views']), 36)
        self.assertTrue((output/'gaussian_map/visualization/index.html').is_file())
        from tools.export_gaussian_submaps import export_submaps
        ply = output/'gaussian_map/background.ply'
        exported = export_submaps(output, background_ply=ply)
        self.assertTrue(ply.is_file())
        self.assertGreater(exported['background_ply']['gaussians'], 0)
        self.assertGreater(exported['background_ply']['exact_duplicate_seeds_omitted'], 0)

    def test_full_visible_snapshots_keep_points_omitted_from_training(self):
        fixture = self.fixture(active_budget=6)
        output = fixture.root/'small_training_budget'
        with patch('frontend.strict_two_frame.run', side_effect=fixture.frontend):
            state = synthetic.runner.run_sequence(fixture.cfg, list(range(20)), output)
        reports = [json.loads((output/pair['result_path']/'temporal_state.json').read_text())['gaussian_map']
                   for pair in state['pairs']]
        self.assertTrue(any(row['deferred_visible_gaussians'] > 0 for row in reports))
        self.assertTrue(all(row['active_map_gaussians'] <= 6 for row in reports))
        from tools.visualization.visualize_gaussian_map import discover_map_snapshots
        snapshot = next(path for path in discover_map_snapshots(output)
                        if path.name == 'frame_0000000014.npz')
        with np.load(snapshot) as saved:
            self.assertEqual(len(saved['map_ids']), 8)
            self.assertLessEqual(int(saved['optimized_in_last_solve'].sum()), 6)
            self.assertEqual(str(saved['snapshot_scope']), 'full_visible_map')
        from tools.validation.verify_gaussian_map_run import verify
        audit = verify(output)
        self.assertTrue(audit['passed'], audit)


if __name__ == '__main__':
    unittest.main()
