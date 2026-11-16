"""Real corrected-run core eligibility, without inventing transient posteriors."""
from pathlib import Path
import sys,json,unittest

import numpy as np

EXP=Path(__file__).resolve().parents[1]
if str(EXP) not in sys.path:sys.path.insert(0,str(EXP))
from tracking.object_state import ObjectRegistry
from frontend.hybrid_motion import motion_costs,motion_probabilities


class FinalCoreIdentityLedgerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with np.load(Path(__file__).parent/'fixtures/red_car_final_core_identity_4920_4922.npz') as a:
            cls.f={k:a[k] for k in a.files}

    def inputs(self):
        f=self.f;motion={'hybrid':json.loads(str(f['hybrid_cfg_json']))}
        r=ObjectRegistry({'sequence':{'streaming':True},'object_window':{'enabled':True},
            'tracking':{'motion_identity':{'enabled':True,'geometric_identity_ledger':True}},
            'motion':motion})
        for _ in range(24):r._create_object(4920)
        previous={k:f['previous_'+k].copy() for k in ['x0','x1','matches','confidence']}
        r._remember_geometric_identity(4920,4922,f['previous_models'],f['previous_labels'],
            f['previous_track_ids'],dict(enumerate(map(int,f['previous_local_to_object']))),
            previous,f['K'],motion)
        r._begin_pair(4922)
        r._identity_core_pair=(4922,4924)
        r._identity_cores={local:{tuple(map(float,m)) for m in f['current_initial_matches'][
            f['current_initial_labels']==local]} for local in range(len(f['current_models']))}
        current={k:f['current_'+k].copy() for k in ['x0','x1','matches','confidence']}
        return r,current,motion

    def evaluate(self,r,current,motion):
        f=self.f;cost=motion_costs(current,f['current_models'],f['K'],motion['hybrid'])
        q,_=motion_probabilities(cost,motion['hybrid'])
        return r._motion_identity_gate(f['current_models'],f['current_labels'],q,
            f['current_track_ids'],current,f['K'],motion)

    def test_final_revalidated_old_tracks_can_add_identity_without_diluting_discovery(self):
        r,current,motion=self.inputs();forbidden,report=self.evaluate(r,current,motion)
        # No original transient posterior is available in this fixture. This
        # establishes an admissible same-ID geometric set, not the historical
        # final mapping or an E2E result under unknown foreground conflicts.
        self.assertEqual(r.track_objects,{})
        support=report['identity_support']['19']['1']
        self.assertEqual(support['discovery_core_tracks'],1)
        self.assertEqual(support['additional_final_core_track_ids'],[991,1448])
        self.assertEqual(support['reliable_identity_tracks'],3)
        self.assertNotIn((1,19),forbidden)
        # Geometric gain/candidate support remains the original discovery set.
        self.assertEqual(report['core_evidence']['1']['discovered_matches'],29)
        self.assertEqual(report['core_evidence']['1']['reliable_unique_support'],25)
        accepted={v['track_id']:v for v in report['geometric_identity_ledger']['accepted_tracks']}
        for track in [991,1448]:
            self.assertEqual(accepted[track]['reason'],'accepted_final_core_outside_discovery')
            self.assertFalse(accepted[track]['in_initial_discovery_core'])

    def test_explicit_other_foreground_conflict_still_rejects(self):
        r,current,motion=self.inputs();r.track_objects[1448]=13
        forbidden,report=self.evaluate(r,current,motion)
        self.assertIn((1,19),forbidden)
        self.assertEqual(report['identity_support']['19']['1']['reliable_identity_tracks'],2)
        rejected={v['track_id']:v for v in report['geometric_identity_ledger']['rejected_tracks']}
        self.assertEqual(rejected[1448]['reason'],'current_foreground_conflict')
        self.assertEqual(r.track_objects[1448],13)

    def test_unreliable_old_track_rejection_is_explicit(self):
        r,current,motion=self.inputs();track=1069
        _,report=self.evaluate(r,current,motion)
        self.assertEqual(int(self.f['current_labels'][self.f['current_track_ids']==track][0]),-1)
        rejected={v['track_id']:v for v in report['geometric_identity_ledger']['rejected_tracks']}
        self.assertEqual(rejected[track]['reason'],'current_core_unreliable')

    def test_ledger_refresh_uses_final_core_instead_of_initial_unknown_label(self):
        r,current,motion=self.inputs();f=self.f
        r._remember_geometric_identity(4922,4924,f['current_models'],f['current_labels'],
            f['current_track_ids'],dict(enumerate(map(int,f['current_local_to_object']))),
            current,f['K'],motion)
        for track in [991,1448]:
            self.assertEqual(r._geometric_identity_tracks[track],20)
        self.assertNotIn(1069,r._geometric_identity_tracks)


if __name__=='__main__':unittest.main()
