"""CPU checks for independent replay ablations and honest motion provenance."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
SPEC = importlib.util.spec_from_file_location('spatial_replay_options', HERE / 'replay_object_window.py')
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


class ReplaySpatialOptionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.cfg = dict(dataset=dict(data_root=str(self.root / 'dataset'), sequence='test'),
                        object_window=dict(enabled=True))
        (self.source / 'config.yaml').write_text(yaml.safe_dump(self.cfg))
        (self.source / 'sequence_state.json').write_text(json.dumps(dict(
            frames=[0, 1], pairs=[], objects={'0': {}})))

    def run_replay(self, name, **kwargs):
        with patch.object(replay.stereo_io, 'read_calibration', return_value=(np.eye(3), .5)):
            return replay.run_replay(self.source, self.root / name, visualize_results=False,
                                     render_optimize=False, reuse_saved_objects=True, **kwargs)

    def test_new_defaults_enable_spatial_terms_without_claiming_feature_motion_refit(self):
        original = (self.source / 'config.yaml').read_bytes()
        result = self.run_replay('defaults')
        options = result['config']['object_window']
        self.assertTrue(options['height']['enabled'])
        self.assertTrue(options['connectivity']['enabled'])
        self.assertEqual(options['height']['weight'], .15)
        self.assertEqual(options['connectivity']['weight'], .08)
        self.assertFalse(result['replay']['object_motion_fitting_rerun'])
        self.assertFalse(result['replay']['feature_height_model_fitting_enabled'])
        self.assertFalse(result['replay']['feature_height_model_refit_applied'])
        self.assertIn('saved object motions', result['replay']['feature_height_model_fitting_note'])
        self.assertEqual((self.source / 'config.yaml').read_bytes(), original)

    def test_spatial_ablations_are_independent_and_preserve_explicit_config(self):
        for name, flags, expected in (
                ('no_height', dict(no_height_constraint=True), (False, True)),
                ('no_connectivity', dict(no_connectivity_constraint=True), (True, False)),
                ('neither', dict(no_height_constraint=True, no_connectivity_constraint=True), (False, False))):
            with self.subTest(name=name):
                state = self.run_replay(name, **flags)
                options = state['config']['object_window']
                self.assertEqual((options['height']['enabled'], options['connectivity']['enabled']), expected)
                self.assertEqual((state['replay']['height_constraint_enabled'],
                                  state['replay']['connectivity_constraint_enabled']), expected)
        self.cfg['object_window']['height'] = dict(enabled=False)
        (self.source / 'config.yaml').write_text(yaml.safe_dump(self.cfg))
        state = self.run_replay('configured_off')
        self.assertFalse(state['config']['object_window']['height']['enabled'])
        self.assertTrue(state['config']['object_window']['connectivity']['enabled'])
        self.assertEqual(state['config']['object_window']['height']['weight'], .15)


if __name__ == '__main__':
    unittest.main()
