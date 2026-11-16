"""CPU checks of persistent geometry, association and checkpoint continuity."""
import sys
from pathlib import Path

_EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(_EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_ROOT))

import importlib.util
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation


_path = Path(__file__).resolve().parents[1] / "mapping/persistent_gaussian_map.py"
_spec = importlib.util.spec_from_file_location("persistent_gaussian_map", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
PersistentGaussianMap = _module.PersistentGaussianMap
PARAMETER_WIDTHS = _module.PARAMETER_WIDTHS


def rgb(n, color=(.3, .4, .5)):
    return np.tile(np.asarray(color, np.float32), (n, 1))


class PersistentGaussianMapTests(unittest.TestCase):
    def make_map(self):
        return PersistentGaussianMap(dict(association_distance=.1, color_distance=.2,
                                          voxel_size=.04, initial_scale=.03))

    def test_initialization_preserves_canonical_geometry_and_parameter_schema(self):
        store = self.make_map()
        points = np.asarray([[1., 2., 3.], [2., 3., 4.]])
        result = store.integrate(4, points, rgb(2), scales=[.02, .04],
                                 rotations=[[2., 0., 0., 0.], [0., 0., 0., 3.]],
                                 opacity=[.25, .75], frame_id=10)
        self.assertEqual((result["inserted"], result["matched"], result["skipped"]), (2, 0, 0))
        np.testing.assert_array_equal(result["map_ids"], [0, 1])
        batch = store.select([1, 0])
        np.testing.assert_array_equal(batch["object_ids"], [4, 4])
        np.testing.assert_array_equal(batch["xyz"], points[::-1])
        np.testing.assert_allclose(np.exp(batch["log_scales"]), [[.04] * 3, [.02] * 3], rtol=1e-6)
        np.testing.assert_allclose(batch["rotations"], [[0., 0., 0., 1.], [1., 0., 0., 0.]])
        np.testing.assert_allclose(1 / (1 + np.exp(-batch["opacity_logits"])), [[.75], [.25]], rtol=1e-6)
        self.assertEqual(store.to_dict()["objects"]["4"]["first_frame"], 10)
        for name, width in PARAMETER_WIDTHS.items():
            self.assertEqual(batch[name].shape, (2, width))
            self.assertEqual(batch["optimizer_state"]["params"][name]["row_steps"].shape, (2, 1))

    def test_same_batch_and_later_observations_share_map_rows_without_reset(self):
        store = self.make_map()
        # Neighbours across voxel boundaries must still deduplicate.
        points = [[.039, 0., 2.], [.041, 0., 2.], [.3, 0., 2.]]
        first = store.integrate(0, points, rgb(3), frame_id=0)
        np.testing.assert_array_equal(first["map_ids"], [0, 0, 1])
        self.assertEqual(first["inserted"], 2)
        original = store.select([0, 1])
        learned_xyz = original["xyz"] + [0., .02, 0.]
        store.commit([0, 1], dict(xyz=learned_xyz, colors=rgb(2, (.32, .4, .5)),
                                 log_scales=np.full((2, 3), -3., np.float32)))
        repeated = store.integrate(0, np.asarray(points) + [0., .01, 0.], rgb(3), frame_id=2)
        np.testing.assert_array_equal(repeated["map_ids"], [0, 0, 1])
        self.assertEqual(repeated["inserted"], 0)
        self.assertEqual(len(store), 2)
        after = store.select([0, 1])
        np.testing.assert_array_equal(after["xyz"], learned_xyz.astype(np.float32))
        np.testing.assert_array_equal(after["initial_xyz"], original["xyz"])
        np.testing.assert_array_equal(after["initial_log_scales"], original["log_scales"])
        np.testing.assert_array_equal(after["observation_count"], [4, 2])
        np.testing.assert_array_equal(after["last_seen_frame"], [2, 2])

    def test_association_checks_color_and_owner_and_falls_back_from_bad_hints(self):
        store = self.make_map()
        initial = store.integrate(0, [[0., 0., 2.], [.05, 0., 2.]],
                                  [[1., 0., 0.], [0., 1., 0.]])
        np.testing.assert_array_equal(initial["map_ids"], [0, 1])
        # Closest geometric neighbour has the wrong color; choose the second.
        sample = store.integrate(0, [[.01, 0., 2.]], [[0., 1., 0.]], existing_map_ids=[0])
        np.testing.assert_array_equal(sample["map_ids"], [1])
        dynamic = store.integrate(8, [[.01, 0., 2.]], [[0., 1., 0.]], existing_map_ids=[1])
        np.testing.assert_array_equal(dynamic["map_ids"], [2])
        np.testing.assert_array_equal(store.select([1, 2])["object_ids"], [0, 8])
        # A geometrically stale hint cannot attach another surface to ID 2.
        other = store.integrate(8, [[2., 0., 2.]], [[0., 1., 0.]], existing_map_ids=[2])
        np.testing.assert_array_equal(other["map_ids"], [3])
        np.testing.assert_array_equal(store.ids_for_object(8), [2, 3])

    def test_learned_appearance_cannot_turn_unchanged_measurements_into_new_surfaces(self):
        store = self.make_map()
        points = np.asarray([[0., 0., 2.], [.4, 0., 2.]], np.float32)
        measured_colors = np.asarray([[.8, .1, .2], [.1, .7, .3]], np.float32)
        first = store.integrate(0, points, measured_colors)
        learned_colors = np.asarray([[.1, .8, .9], [.9, .1, .8]], np.float32)
        store.commit(first['map_ids'], dict(colors=learned_colors))
        # Both ordinary matching and explicit ID hints use raw color anchors.
        for hints in (None, first['map_ids']):
            matched = store.integrate(0, points, measured_colors, existing_map_ids=hints)
            np.testing.assert_array_equal(matched['map_ids'], first['map_ids'])
            self.assertEqual(matched['inserted'], 0)
        np.testing.assert_array_equal(store.select([0, 1])['colors'], learned_colors)
        np.testing.assert_array_equal(store.select([0, 1])['association_colors'], measured_colors)
        with tempfile.TemporaryDirectory() as directory:
            store.save(directory)
            restored = PersistentGaussianMap.load(directory)
            matched = restored.integrate(0, points, measured_colors)
            np.testing.assert_array_equal(matched['map_ids'], first['map_ids'])
            self.assertEqual(matched['inserted'], 0)
            np.testing.assert_array_equal(restored.select([0, 1])['association_colors'], measured_colors)
            np.testing.assert_array_equal(restored.select([0, 1])['colors'], learned_colors)

    def test_budget_keeps_association_and_leaves_no_allocated_id_holes(self):
        store = self.make_map()
        store.integrate(0, [[0., 0., 2.]], rgb(1))
        result = store.integrate(0, [[0., 0., 2.], [1., 0., 2.], [1.01, 0., 2.], [2., 0., 2.]],
                                 rgb(4), max_new=1)
        np.testing.assert_array_equal(result["map_ids"], [0, 1, 1, -1])
        self.assertEqual((result["inserted"], result["matched"], result["skipped"]), (1, 2, 1))
        frozen = store.integrate(0, [[1., 0., 2.], [2., 0., 2.]], rgb(2), max_new=0)
        np.testing.assert_array_equal(frozen["map_ids"], [1, -1])
        self.assertEqual(store.next_id, 2)
        next_row = store.integrate(3, [[0., 0., 2.]], rgb(1), max_new=1)
        np.testing.assert_array_equal(next_row["map_ids"], [2])
        self.assertEqual(len(store), 3)

    def test_reordered_optimizer_rows_preserve_moments_and_immutable_priors(self):
        store = self.make_map()
        store.integrate(0, [[0., 0., 2.], [1., 0., 2.]], rgb(2))
        store.integrate(7, [[0., 0., 2.]], rgb(1))
        before = store.select([2, 0])
        state = dict(map_ids=np.asarray([0, 2]), params={
            name: dict(exp_avg=np.tile([[1.], [3.]], (1, width)),
                       exp_avg_sq=np.tile([[2.], [4.]], (1, width)),
                       row_steps=np.asarray([[5], [9]], np.int64))
            for name, width in PARAMETER_WIDTHS.items()})
        store.commit([2, 0], dict(xyz=before["xyz"] + 1., colors=[[2., -.1, .5], [.1, .2, .3]],
                                 rotations=[[0., 2., 0., 0.], [2., 0., 0., 0.]]), state)
        after = store.select([0, 1, 2])
        np.testing.assert_array_equal(after["optimizer_state"]["params"]["xyz"]["row_steps"], [[5], [0], [9]])
        np.testing.assert_array_equal(after["optimizer_state"]["params"]["colors"]["exp_avg"][:, 0], [1., 0., 3.])
        np.testing.assert_array_equal(store.select([2])["colors"], [[1., 0., .5]])
        np.testing.assert_array_equal(store.select([2, 0])["initial_xyz"], before["initial_xyz"])
        # Selection returns independent copies, including optimizer moments.
        after["xyz"][:] = -100
        after["optimizer_state"]["params"]["xyz"]["exp_avg"][:] = -100
        self.assertGreater(store.select([0])["xyz"][0, 0], -100)
        self.assertEqual(store.select([0])["optimizer_state"]["params"]["xyz"]["exp_avg"][0, 0], 1.)

    def test_invalid_commit_is_atomic_and_owner_cannot_be_edited(self):
        store = self.make_map()
        store.integrate(2, [[0., 0., 2.]], rgb(1))
        original = store.select([0])["xyz"]
        with self.assertRaises(ValueError):
            store.commit([0], dict(xyz=original + 1., rotations=[[0., 0., 0., 0.]]))
        np.testing.assert_array_equal(store.select([0])["xyz"], original)
        with self.assertRaises(ValueError):
            store.commit([0], dict(object_ids=np.asarray([3])))
        with self.assertRaises(ValueError):
            store.select([0, 0])
        with self.assertRaises(KeyError):
            store.select([1])

    def test_archive_keeps_history_and_new_object_never_reuses_ids(self):
        store = self.make_map()
        store.integrate(1, [[0., 0., 2.]], rgb(1), frame_id=10)
        store.integrate(1, [[.01, 0., 2.]], rgb(1), frame_id=16)
        store.set_status(1, "missing")
        store.set_status(1, "archived", last_seen_frame=16)
        archived = store.export_object(1)
        self.assertEqual(archived["metadata"]["last_frame"], 16)
        self.assertEqual(store.get_status(1), "archived")
        new = store.integrate(2, [[0., 0., 2.]], rgb(1), frame_id=20)
        np.testing.assert_array_equal(new["map_ids"], [1])
        np.testing.assert_array_equal(store.export_object(1)["xyz"], archived["xyz"])
        store.set_status(2, "retired", last_seen_frame=21)
        self.assertEqual(store.get_status(2), "retired")

    def test_checkpoint_roundtrip_includes_all_parameters_and_adam_history(self):
        store = self.make_map()
        store.integrate(0, [[0., 0., 2.], [1., 0., 2.]], rgb(2), frame_id=0)
        store.integrate(4, [[0., 0., 3.]], rgb(1), frame_id=10)
        selected = store.select([2, 0])
        for values in selected["optimizer_state"]["params"].values():
            values["exp_avg"][:] = .2
            values["exp_avg_sq"][:] = .4
            values["row_steps"][:] = 8
        store.commit([2, 0], dict(xyz=selected["xyz"] + .01), selected["optimizer_state"])
        store.set_status(4, "archived", last_seen_frame=16)
        before = store.select([2, 0, 1])
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = store.save(directory)
            self.assertTrue(manifest_path.is_file())
            restored = PersistentGaussianMap.load(directory)
            self.assertEqual(restored.to_dict(), store.to_dict())
            self.assertEqual(restored.association_distance, .1)
            after = restored.select([2, 0, 1])
            for name in (*PARAMETER_WIDTHS, "initial_xyz", "initial_log_scales", "association_colors", "map_ids", "object_ids"):
                np.testing.assert_array_equal(after[name], before[name])
            for name in PARAMETER_WIDTHS:
                for key in ("exp_avg", "exp_avg_sq", "row_steps"):
                    np.testing.assert_array_equal(after["optimizer_state"]["params"][name][key],
                                                  before["optimizer_state"]["params"][name][key])
            # cKDTree is rebuilt lazily from restored learned coordinates.
            matched = restored.integrate(4, [[.01, .01, 3.01]], rgb(1), max_new=0)
            np.testing.assert_array_equal(matched["map_ids"], [2])
            inserted = restored.integrate(8, [[0., 0., 2.]], rgb(1))
            np.testing.assert_array_equal(inserted["map_ids"], [3])

    def test_checkpoint_writes_only_changed_blocks_and_metadata_without_array_rewrite(self):
        store = self.make_map()
        for oid in (0, 1):
            store.integrate(oid, [[0., 0., 2.]], rgb(1))
        with tempfile.TemporaryDirectory() as directory:
            store.save(directory)
            path = Path(directory) / "manifest.json"
            first = json.loads(path.read_text())["objects"]
            store.set_status(1, "archived")
            store.save(directory)
            second = json.loads(path.read_text())["objects"]
            self.assertEqual(first["1"]["filename"], second["1"]["filename"])
            self.assertEqual(second["1"]["status"], "archived")
            store.commit([0], dict(xyz=np.asarray([[.1, 0., 2.]])))
            store.save(directory)
            third = json.loads(path.read_text())["objects"]
            self.assertNotEqual(second["0"]["filename"], third["0"]["filename"])
            self.assertEqual(second["1"]["filename"], third["1"]["filename"])
            self.assertFalse((Path(directory) / second["0"]["filename"]).exists())
            self.assertEqual(len(list(Path(directory).glob("*.npz"))), 2)

    def test_interrupted_checkpoint_keeps_old_manifest_consistent(self):
        store = self.make_map()
        for oid in (0, 1):
            store.integrate(oid, [[0., 0., 2.]], rgb(1))
        with tempfile.TemporaryDirectory() as directory:
            store.save(directory)
            old = store.select([0, 1])["xyz"]
            store.commit([0, 1], dict(xyz=old + 1.))
            original_save = store._atomic_npz
            calls = []

            def interrupted_save(path, values):
                calls.append(path)
                if len(calls) == 2:
                    raise OSError("simulated interrupted write")
                original_save(path, values)

            with patch.object(store, "_atomic_npz", side_effect=interrupted_save):
                with self.assertRaises(OSError):
                    store.save(directory)
            restored = PersistentGaussianMap.load(directory)
            np.testing.assert_array_equal(restored.select([0, 1])["xyz"], old)

    def test_empty_shapes_and_input_validation(self):
        store = self.make_map()
        empty = store.integrate(0, np.empty((0, 3)), np.empty((0, 3)))
        self.assertEqual(empty["map_ids"].shape, (0,))
        batch = store.select([])
        self.assertEqual(batch["xyz"].shape, (0, 3))
        self.assertEqual(batch["rotations"].shape, (0, 4))
        self.assertEqual(store.object_ids, [])
        for kwargs in (dict(scales=0.), dict(opacity=1.), dict(max_new=-1),
                       dict(rotations=[[0., 0., 0., 0.]]), dict(existing_map_ids=[1.5])):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                store.integrate(0, [[0., 0., 2.]], rgb(1), **kwargs)
        with self.assertRaises(ValueError):
            store.integrate(0, [[float("nan"), 0., 2.]], rgb(1))
        with self.assertRaises(ValueError):
            store.integrate(0, [[0., 0., 2.]], [[1.2, 0., 0.]])
        self.assertEqual(len(store), 0)
        self.assertEqual(store.object_ids, [])

    def test_reassignment_preserves_world_centres_covariances_priors_and_stable_ids(self):
        store = self.make_map()
        old_q = Rotation.from_euler("xyz", [[.4, .2, -.1], [-.2, .7, .5]]).as_quat()[:, [3, 0, 1, 2]]
        store.integrate(0, [[.2, -.4, 2.], [1., .1, 3.]], rgb(2),
                        scales=[[.02, .06, .03], [.1, .04, .07]], rotations=old_q, frame_id=12)
        before = store.select([1, 0])
        for state in before["optimizer_state"]["params"].values():
            state["exp_avg"][:] = .3
            state["exp_avg_sq"][:] = .7
            state["row_steps"][:] = 11
        store.commit([1, 0], dict(xyz=before["xyz"] + [.01, -.02, .03]), before["optimizer_state"])
        before = store.select([1, 0])
        old_pose, new_pose = np.eye(4), np.eye(4)
        old_pose[:3, :3] = Rotation.from_euler("xyz", [.2, -.1, .4]).as_matrix()
        old_pose[:3, 3] = [3., -1., .5]
        new_pose[:3, :3] = Rotation.from_euler("xyz", [-.3, .5, -.2]).as_matrix()
        new_pose[:3, 3] = [-2., 4., 1.]
        transform = np.linalg.inv(new_pose) @ old_pose
        # Prime the old association tree; moving these rows must invalidate it.
        store.integrate(0, before["xyz"], rgb(2), max_new=0)
        event = store.reassign([1, 0], 8, transform, frame_id=14)
        after = store.select([1, 0])
        np.testing.assert_array_equal(after["object_ids"], [8, 8])
        np.testing.assert_array_equal(after["map_ids"], [1, 0])
        self.assertEqual((len(store), store.next_id, store.revision), (2, 2, 1))
        self.assertEqual(event["old_object_ids"], [0, 0])
        self.assertEqual(event["new_object_id"], 8)
        for name in ("xyz", "initial_xyz"):
            world_before = before[name] @ old_pose[:3, :3].T + old_pose[:3, 3]
            world_after = after[name] @ new_pose[:3, :3].T + new_pose[:3, 3]
            np.testing.assert_allclose(world_after, world_before, atol=8e-7)
        def world_covariance(batch, pose):
            basis = Rotation.from_quat(batch["rotations"][:, [1, 2, 3, 0]]).as_matrix()
            basis = pose[:3, :3] @ basis
            scaled = basis * np.exp(2 * batch["log_scales"])[:, None, :]
            return scaled @ np.swapaxes(basis, 1, 2)
        np.testing.assert_allclose(world_covariance(after, new_pose), world_covariance(before, old_pose),
                                   rtol=3e-6, atol=1e-8)
        for name in ("colors", "opacity_logits", "log_scales", "initial_log_scales", "association_colors"):
            np.testing.assert_array_equal(after[name], before[name])
        for name, state in after["optimizer_state"]["params"].items():
            for key in state:
                expected = np.zeros_like(state[key]) if name in ("xyz", "rotations") else before["optimizer_state"]["params"][name][key]
                np.testing.assert_array_equal(state[key], expected)
        self.assertEqual(len(store.ids_for_object(0)), 0)
        # Both ordinary matching and stale old-owner hints use new ownership.
        matched = store.integrate(8, after["xyz"], rgb(2), existing_map_ids=[1, 0], max_new=0)
        np.testing.assert_array_equal(matched["map_ids"], [1, 0])

    def test_reassignment_validates_all_transforms_before_edits_and_supports_mixed_owners(self):
        store = self.make_map()
        store.integrate(0, [[0., 0., 2.], [1., 0., 2.]], rgb(2))
        store.integrate(2, [[0., 0., 3.]], rgb(1))
        before = store.select([0, 1, 2])
        bad = np.tile(np.eye(4), (2, 1, 1))
        bad[1, 0, 0] = 2.
        with self.assertRaises(ValueError):
            store.reassign([0, 2], 4, bad)
        self.assertEqual(store.revision, 0)
        np.testing.assert_array_equal(store.select([0, 1, 2])["xyz"], before["xyz"])
        transforms = np.tile(np.eye(4), (2, 1, 1))
        transforms[:, :3, 3] = [[1., 0., 0.], [0., 2., 0.]]
        event = store.reassign([0, 2], 4, transforms)
        self.assertEqual(event["old_object_ids"], [0, 2])
        np.testing.assert_array_equal(store.select([0, 1, 2])["object_ids"], [4, 0, 4])
        np.testing.assert_array_equal(store.select([0, 2])["xyz"], [[1., 0., 2.], [0., 2., 3.]])
        same = store.reassign([0, 2], 4, np.eye(4))
        self.assertEqual(same["map_ids"], [])
        self.assertEqual(store.revision, 1)
        # Returned journal dictionaries cannot mutate persistent provenance.
        event["map_ids"].clear()
        self.assertEqual(store.ownership_events[0]["map_ids"], [0, 2])

    def test_split_clone_and_prune_keep_live_routes_and_monotonic_ids(self):
        store = self.make_map()
        store.integrate(0, [[0., 0., 2.], [1., 0., 2.]], rgb(2), frame_id=8)
        store.integrate(4, [[0., 0., 3.]], rgb(1), frame_id=10)
        parent = store.select([0, 2])
        for state in parent["optimizer_state"]["params"].values():
            state["exp_avg"][:] = .4
            state["exp_avg_sq"][:] = .7
            state["row_steps"][:] = 9
        store.commit([0, 2], {}, parent["optimizer_state"])
        children = store.append_from_parent([2, 0, 0], dict(
            xyz=[[0., .01, 3.], [-.01, 0., 2.], [.01, 0., 2.]],
            log_scales=np.full((3, 3), -4., np.float32)), frame_id=12, reason="split")
        np.testing.assert_array_equal(children, [3, 4, 5])
        child = store.select(children)
        np.testing.assert_array_equal(child["object_ids"], [4, 0, 0])
        np.testing.assert_array_equal(child["initial_xyz"], child["xyz"])
        np.testing.assert_array_equal(child["initial_log_scales"], child["log_scales"])
        for state in child["optimizer_state"]["params"].values():
            for values in state.values():
                self.assertFalse(np.any(values))
        removed = store.remove([0, 2], frame_id=12, reason="split_parent")
        self.assertEqual(removed["old_object_ids"], [0, 4])
        self.assertEqual((len(store), store.next_id, store.revision), (4, 6, 2))
        np.testing.assert_array_equal(store.is_live([-1, 0, 1, 2, 3, 5, 99]), [False, False, True, False, True, True, False])
        np.testing.assert_array_equal(store.ids_for_object(0), [1, 4, 5])
        np.testing.assert_array_equal(store.ids_for_object(4), [3])
        with self.assertRaises(KeyError):
            store.select([0])
        with self.assertRaises(KeyError):
            store.commit([2], {})
        # A stale hint to a pruned ID never resurrects it.
        fresh = store.integrate(0, [[2., 0., 2.]], rgb(1), existing_map_ids=[0])
        np.testing.assert_array_equal(fresh["map_ids"], [6])
        self.assertEqual(store.to_dict()["pruned_count"], 2)
        self.assertEqual(store.next_revision, 3)

    def test_checkpoint_restores_reassigned_and_pruned_rows_and_event_revisions(self):
        store = self.make_map()
        store.integrate(0, [[0., 0., 2.], [1., 0., 2.]], rgb(2), frame_id=5)
        transform = np.eye(4)
        transform[:3, 3] = [1., 2., 3.]
        store.reassign([1], 7, transform, frame_id=10)
        store.append_from_parent([1], frame_id=11)
        store.remove([0], frame_id=12)
        with tempfile.TemporaryDirectory() as directory:
            store.save(directory)
            restored = PersistentGaussianMap.load(directory)
            self.assertEqual(restored.to_dict(), store.to_dict())
            self.assertEqual(restored.ownership_events, store.ownership_events)
            self.assertEqual(restored.topology_events, store.topology_events)
            np.testing.assert_array_equal(restored.is_live([0, 1, 2]), [False, True, True])
            for key in (*PARAMETER_WIDTHS, "object_ids", "initial_xyz", "initial_log_scales"):
                np.testing.assert_array_equal(restored.select([2, 1])[key], store.select([2, 1])[key])
            inserted = restored.integrate(0, [[0., 0., 2.]], rgb(1))
            np.testing.assert_array_equal(inserted["map_ids"], [3])
            restored.remove([1], frame_id=13)
            self.assertEqual(restored.topology_events[-1]["revision"], 4)

    def test_version_one_checkpoint_loads_without_edit_journal(self):
        store = self.make_map()
        store.integrate(0, [[0., 0., 2.]], rgb(1))
        with tempfile.TemporaryDirectory() as directory:
            path = store.save(directory)
            manifest = json.loads(path.read_text())
            manifest["format_version"] = 1
            for key in ("live_count", "allocated_count", "pruned_ids", "revision", "next_revision", "ownership_events", "topology_events"):
                manifest.pop(key)
            path.write_text(json.dumps(manifest))
            restored = PersistentGaussianMap.load(directory)
            self.assertEqual((len(restored), restored.next_id, restored.revision), (1, 1, 0))

    def test_topology_validation_is_atomic_and_all_pruned_checkpoint_can_restart(self):
        store = self.make_map()
        store.integrate(3, [[0., 0., 2.]], rgb(1))
        for operation in (lambda: store.append_from_parent([0], dict(xyz=[[np.nan, 0., 2.]])),
                          lambda: store.append_from_parent([0], dict(rotations=[[0., 0., 0., 0.]])),
                          lambda: store.append_from_parent([8]),
                          lambda: store.remove([0, 0])):
            with self.assertRaises((ValueError, KeyError)):
                operation()
            self.assertEqual((len(store), store.next_id, store.revision), (1, 1, 0))
        store.remove([0], frame_id=10)
        with tempfile.TemporaryDirectory() as directory:
            store.save(directory)
            restored = PersistentGaussianMap.load(directory)
            self.assertEqual((len(restored), restored.next_id), (0, 1))
            self.assertEqual(restored.select([])["xyz"].shape, (0, 3))
            inserted = restored.integrate(4, [[0., 0., 2.]], rgb(1))
            np.testing.assert_array_equal(inserted["map_ids"], [1])

    def test_checkpoint_rejects_tampered_ownership_and_topology_provenance(self):
        store = self.make_map()
        store.integrate(0, [[0., 0., 2.]], rgb(1))
        store.reassign([0], 3, np.eye(4), frame_id=10)
        with tempfile.TemporaryDirectory() as directory:
            path = store.save(directory)
            original = json.loads(path.read_text())
            for change in (lambda value: value["ownership_events"][0].update(new_object_id=7),
                           lambda value: value.update(revision=2),
                           lambda value: value["ownership_events"][0]["transforms"][0][0].__setitem__(0, 2.)):
                manifest = json.loads(json.dumps(original))
                change(manifest)
                path.write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    PersistentGaussianMap.load(directory)

    def test_shared_id_observations_update_support_only_for_repeated_and_reassigned_ids(self):
        store = self.make_map()
        store.integrate(0, [[0., 0., 2.], [1., 0., 2.]], rgb(2), frame_id=5)
        store.reassign([1], 3, np.eye(4), frame_id=8)
        before = store.select([0, 1])
        for state in before["optimizer_state"]["params"].values():
            state["exp_avg"][:] = .1
            state["exp_avg_sq"][:] = .3
            state["row_steps"][:] = 7
        store.commit([0, 1], dict(xyz=before["xyz"] + [2., 0., 0.]), before["optimizer_state"])
        before = store.select([0, 1])
        revision = store.revision
        result = store.record_observations([1, 0, 1], 10)
        self.assertEqual(result, dict(observations=3, gaussians=2))
        after = store.select([0, 1])
        np.testing.assert_array_equal(after["observation_count"], before["observation_count"] + [1, 2])
        np.testing.assert_array_equal(after["last_seen_frame"], [10, 10])
        for name in (*PARAMETER_WIDTHS, "initial_xyz", "initial_log_scales", "association_colors"):
            np.testing.assert_array_equal(after[name], before[name])
        for name, state in after["optimizer_state"]["params"].items():
            for key in state:
                np.testing.assert_array_equal(state[key], before["optimizer_state"]["params"][name][key])
        self.assertEqual((len(store), store.next_id, store.revision), (2, 2, revision))
        self.assertEqual(store.to_dict()["objects"]["3"]["last_frame"], 10)
        # Out-of-order supported observations cannot decrease last-seen times.
        store.record_observations([1], 9)
        self.assertEqual(store.select([1])["last_seen_frame"][0], 10)
        self.assertEqual(store.record_observations([], 10), dict(observations=0, gaussians=0))
        with tempfile.TemporaryDirectory() as directory:
            store.save(directory)
            restored = PersistentGaussianMap.load(directory)
            np.testing.assert_array_equal(restored.select([0, 1])["observation_count"], [2, 4])

    def test_invalid_shared_id_observations_do_not_partially_update_support(self):
        store = self.make_map()
        store.integrate(0, [[0., 0., 2.], [1., 0., 2.]], rgb(2), frame_id=5)
        store.remove([1])
        before = store.select([0])
        for ids, frame in (([0, 1], 10), ([0, 2], 10), ([0, -1], 10), ([0.5], 10),
                           ([0], None), ([0], True), ([0], -1)):
            with self.subTest(ids=ids, frame=frame), self.assertRaises((ValueError, KeyError)):
                store.record_observations(ids, frame)
            after = store.select([0])
            np.testing.assert_array_equal(after["observation_count"], before["observation_count"])
            np.testing.assert_array_equal(after["last_seen_frame"], before["last_seen_frame"])


if __name__ == "__main__":
    unittest.main()
