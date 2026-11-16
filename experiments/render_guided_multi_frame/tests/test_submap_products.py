"""Archived map browsing/export/audit preserve boundaries and moving owners."""
import sys
from pathlib import Path

_EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(_EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_ROOT))

from copy import deepcopy
import json
import tempfile
import unittest

import cv2
import numpy as np

HERE = Path(__file__).resolve().parents[1]
from tools.export_gaussian_submaps import export_submaps
from mapping.gaussian_submaps import partition_submaps
from mapping.persistent_gaussian_map import PersistentGaussianMap
from tools.validation.verify_gaussian_map_run import WIDTHS, verify
from tools.visualization.visualize_gaussian_map import discover_map_snapshots, prepare_snapshot, visualize_map


class SubmapProductsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.frames = list(range(5))
        self.cfg = dict(dataset=dict(data_root=str(self.root / 'dataset'), sequence='fixture_sync'),
                        gaussian_map=dict(enabled=True, window_frames=3, max_active_gaussians=2,
                                          optimization=dict(backend='torch_reference', iterations=0),
                                          submaps=dict(frames=3, overlap_frames=1)))
        self.state = dict(status='complete', frames=self.frames, config=self.cfg, pairs=[],
                          objects={str(oid): dict(status='active', confirmed=True,
                                                first_observed_frame=0, last_observed_frame=4) for oid in (0, 1)})
        self.manifest = dict(format='fixed_gaussian_submaps', format_version=1,
                             frames_per_submap=3, overlap_frames=1, requested_frame_ids=self.frames,
                             planned_submaps=2, processed_pairs=4, finalized=True,
                             global_object_ids=True, object_lifecycle_resets_at_boundary=False,
                             coordinate_system='background_world;dynamic_object_canonical',
                             world_from_submap=np.eye(4).tolist(), resident_submaps=0, submaps=[])
        self.K = np.asarray([[16., 0., 8.5], [0., 16., 8.5], [0., 0., 1.]])
        image_dir = self.root / 'dataset/data_2d_raw/fixture_sync/image_00/data_rect'
        image_dir.mkdir(parents=True)
        for frame in self.frames:
            cv2.imwrite(str(image_dir / f'{frame:010d}.png'), np.full((17, 17, 3), 32, np.uint8))
        previous = None
        global_trajectory = []
        for specification in partition_submaps(self.frames, frames=3):
            sid = specification['submap_id']
            directory = self.root / f'gaussian_map/submaps/submap_{sid:06d}'
            (directory / 'frames').mkdir(parents=True)
            store = PersistentGaussianMap(dict(association_distance=.01))
            if previous is None:
                store.integrate(0, [[-.3, 0., 3.], [.3, 0., 3.]], [[0., 1., 0.]]*2, frame_id=0)
                store.integrate(1, [[0., 0., 2.]], [[1., 0., 0.]], frame_id=0)
                seed_count = 0
            else:
                old_ids, new_ids = [], []
                for mid, oid in ((0, 0), (2, 1)):
                    imported = store.import_rows(previous.select(np.asarray([mid])), oid)
                    old_ids.append(mid)
                    new_ids.append(int(imported[0]))
                store.integrate(0, [[.6, 0., 3.]], [[0., 1., 0.]], frame_id=2)
                (directory / 'boundary_seed.json').write_text(json.dumps(dict(
                    source_submap_id=0, destination_submap_id=1, frame=2,
                    source_map_ids=old_ids, destination_map_ids=new_ids,
                    geometry='same_global_owner_canonical_coordinates', tracked_anchor_count=2)))
                seed_count = 2
            for oid in (0, 1):
                store.set_status(oid, 'active', last_seen_frame=specification['frame_ids'][-1])
            store.save(directory / 'map')
            entry = dict(specification, state='closed', processed_pairs=2,
                         directory=directory.relative_to(self.root).as_posix(),
                         checkpoint=(directory / 'map/manifest.json').relative_to(self.root).as_posix(),
                         snapshots=(directory / 'frames').relative_to(self.root).as_posix(),
                         trajectory=(directory / 'trajectory.jsonl').relative_to(self.root).as_posix(),
                         world_from_submap=np.eye(4).tolist(), map_id_namespace=f'submap_{sid:06d}',
                         live_gaussians=3, allocated_gaussians=3,
                         object_status_at_close={'0': 'active', '1': 'active'})
            self.manifest['submaps'].append(entry)
            batch = store.select(np.arange(3), include_optimizer=False)
            for frame in specification['frame_ids']:
                camera, owners = self.poses(frame)
                snapshot = {key: batch[key] for key in (*WIDTHS, 'map_ids')}
                snapshot.update(frame=np.asarray(frame), K=self.K, map_revision=np.asarray(store.revision),
                                object_labels=batch['object_ids'], object_ids=np.asarray([0, 1]),
                                object_to_camera=np.asarray([np.linalg.inv(camera) @ np.asarray(owners[str(oid)]) for oid in (0, 1)]),
                                snapshot_scope=np.asarray('full_visible_map'))
                np.savez_compressed(directory / f'frames/frame_{frame:010d}.npz', **snapshot)
            rows = []
            for index, (first, second) in enumerate(zip(specification['frame_ids'][:-1], specification['frame_ids'][1:])):
                a, pa = self.poses(first)
                b, pb = self.poses(second)
                row = dict(first=first, second=second, submap_id=sid,
                           camera_to_world_first=a.tolist(), camera_to_world_second=b.tolist(),
                           object_to_world_first=pa, object_to_world_second=pb)
                rows.append(row)
                global_trajectory.append(row)
                relative = f'pairs/{first}_{second}'
                pair_dir = self.root / relative
                pair_dir.mkdir(parents=True)
                self.state['pairs'].append(dict(first=first, second=second, result_path=relative))
                optimization = dict(optimized_parameters=list(WIDTHS), iterations_completed=0,
                                    ownership_optimized=False, camera_poses_optimized=False,
                                    object_poses_optimized=False, track_count=0, track_3d_count=0)
                report = dict(submap_id=sid, total_map_gaussians=3, allocated_map_gaussians=3,
                              inserted_gaussians=3-seed_count if index == 0 else 0,
                              active_map_gaussians=2, deferred_visible_gaussians=1,
                              fixed_object_ownership=True, camera_poses_optimized=False,
                              object_poses_optimized=False, optimization=optimization,
                              window_frames=specification['frame_ids'][:index+2])
                life = dict(source=dict(frame=first, observed_object_ids=[0, 1]),
                            target=dict(frame=second, observed_object_ids=[0, 1]))
                (pair_dir / 'temporal_state.json').write_text(json.dumps(dict(
                    gaussian_map=report, object_window=dict(lifecycle=life))))
                models = np.asarray([np.linalg.inv(b) @ np.asarray(pb[str(oid)]) @
                                     np.linalg.inv(np.asarray(pa[str(oid)])) @ a for oid in (0, 1)])
                np.savez_compressed(pair_dir / 'object_state.npz', object_ids=np.asarray([0, 1]),
                                    H_object=models, object_model_valid=np.ones(2, bool))
            (directory / 'trajectory.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
            previous = store
        self.manifest['global_stored_gaussians'] = 6
        self.save_manifest()
        (self.root / 'gaussian_map/trajectory.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in global_trajectory))
        (self.root / 'sequence_state.json').write_text(json.dumps(self.state))

    def poses(self, frame):
        camera = np.eye(4)
        camera[0, 3] = .01*frame
        owner = np.eye(4)
        owner[0, 3] = .05*frame
        return camera, {'0': np.eye(4).tolist(), '1': owner.tolist()}

    def save_manifest(self):
        (self.root / 'gaussian_map/submaps/manifest.json').write_text(json.dumps(self.manifest))

    def test_audit_two_archived_chunks_and_full_visible_snapshot_exceeds_training_budget(self):
        before = {path: path.stat().st_mtime_ns for path in self.root.rglob('*') if path.is_file()}
        report = verify(self.root)
        self.assertTrue(report['passed'], report)
        self.assertEqual(len(report['submaps']), 2)
        self.assertEqual(len(report['snapshots']), 6)
        self.assertTrue(report['checks']['snapshot_training_budget_separated'])
        after = {path: path.stat().st_mtime_ns for path in self.root.rglob('*') if path.is_file()}
        self.assertEqual(before, after)

    def test_strict_boundary_and_global_camera_corruption_are_rejected(self):
        saved = deepcopy(self.manifest)
        self.manifest['submaps'][0]['frame_ids'] = [0, 1]
        self.save_manifest()
        self.assertFalse(verify(self.root)['checks']['strict_submap_partition'])
        self.manifest = saved
        self.save_manifest()
        trajectory_path = self.root / 'gaussian_map/trajectory.jsonl'
        rows = [json.loads(line) for line in trajectory_path.read_text().splitlines()]
        rows[2]['camera_to_world_first'][0][3] += 1
        trajectory_path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        self.assertFalse(verify(self.root)['checks']['camera_boundary_continuity'])

    def test_shared_frame_browse_uses_next_submap_and_continuous_global_object(self):
        paths = discover_map_snapshots(self.root)
        self.assertEqual(len(paths), 5)
        self.assertIn('submap_000001', str(paths[2]))
        with np.load(paths[-1]) as saved:
            _, _, geometry = prepare_snapshot({key: saved[key] for key in saved.files})
        np.testing.assert_allclose(geometry['xyz'][geometry['labels'] == 1], [[.16, 0., 2.]], atol=1e-6)
        page = visualize_map(self.root, width=24)
        self.assertEqual([row['frame'] for row in page['scene']['views']], self.frames)
        self.assertEqual([row['object_id'] for row in page['objects']], [0, 1])
        self.assertEqual(len(page['objects'][1]['views']), 5)
        self.assertIn('submap_000001', page['scene']['views'][2]['snapshot'])

    def test_global_index_and_streaming_background_export_deduplicate_seeds_exclude_dynamic(self):
        output = self.root / 'gaussian_map/stitched_background.ply'
        index = export_submaps(self.root, background_ply=output)
        self.assertEqual(index['background_ply']['gaussians'], 3)
        self.assertEqual(index['background_ply']['exact_duplicate_seeds_omitted'], 2)
        self.assertEqual(index['dynamic_objects'], [dict(object_id=1, submap_ids=[0, 1])])
        content = output.read_bytes()
        header = content.split(b'end_header\n')[0].decode()
        self.assertIn('element vertex 3', header)
        self.assertIn('property uint local_map_id', header)
        self.assertTrue((self.root / 'gaussian_map/global_scene.json').is_file())

    def test_manifest_cannot_read_snapshots_outside_run(self):
        self.manifest['submaps'][0]['snapshots'] = '../elsewhere'
        self.save_manifest()
        with self.assertRaises(ValueError):
            discover_map_snapshots(self.root)


if __name__ == '__main__':
    unittest.main()
