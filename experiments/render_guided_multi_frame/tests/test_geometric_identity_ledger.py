"""Adjacent identity continuation uses fresh geometry, without changing labels."""
from pathlib import Path
import sys
import json
import unittest
from copy import deepcopy

import numpy as np

EXP=Path(__file__).resolve().parents[1]
if str(EXP) not in sys.path:sys.path.insert(0,str(EXP))
from tracking.object_state import ObjectRegistry
from frontend.hybrid_motion import _core_labels, motion_costs, motion_probabilities


class GeometricIdentityLedgerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with np.load(Path(__file__).parent/'fixtures/red_car_identity_4920_4922.npz') as a:
            cls.fixture={k:a[k] for k in a.files}

    def registry(self,enabled=True):
        f=self.fixture;motion={'hybrid':json.loads(str(f['hybrid_cfg_json']))}
        cfg={'sequence':{'streaming':True},'object_window':{'enabled':True},
             'tracking':{'prior_weight':.25,'motion_identity':{
                 'enabled':True,'geometric_identity_ledger':enabled,
                 'split_background_on_independent_core':True}},'motion':motion}
        r=ObjectRegistry(cfg)
        for _ in range(23):r._create_object(4920)
        prev={k:f['previous_'+k].copy() for k in ['x0','x1','matches','confidence']}
        mapping=dict(enumerate(map(int,f['previous_local_to_object'])))
        r._remember_geometric_identity(4920,4922,f['previous_models'],f['previous_labels'],
            f['previous_track_ids'],mapping,prev,f['K'],motion)
        car=f['previous_track_ids'][f['previous_labels']==1]
        # The archive records two reliable identity votes in current local1
        # and one in local3. Its transient exact per-track posteriors were not
        # archived; emulate that measured count using real matched core IDs.
        r.track_objects={int(t):0 for t in f['previous_track_ids'] if t>=0}
        r.track_objects.update({1657:16,54:16,1679:16})
        r.track_probabilities={t:{oid:1.} for t,oid in r.track_objects.items()}
        current={k:f['current_'+k].copy() for k in ['x0','x1','matches','confidence']}
        r._begin_pair(4922)
        return r,current,f['current_models'].copy(),f['current_labels'].copy(),f['current_track_ids'].copy(),motion

    def view(self,r,a,models,labels,tracks,motion):
        return r._identity_track_objects(models,labels,tracks,a,self.fixture['K'],motion)

    def test_recorded_red_car_three_geometric_tracks_recover_two_posterior_votes(self):
        r,a,models,labels,tracks,motion=self.registry()
        old=r._eligible_track_objects();rows=labels==1
        self.assertEqual(sum(old.get(int(t))==16 for t in tracks[rows]),2)
        view,report=self.view(r,a,models,labels,tracks,motion)
        self.assertEqual(sum(view.get(int(t))==16 for t in tracks[rows]),3)
        self.assertIn(992,report['revalidated_track_ids'])
        q,_=motion_probabilities(motion_costs(a,models,self.fixture['K'],motion['hybrid']),motion['hybrid'])
        forbidden,_=r._motion_identity_gate(models,labels,q,tracks,a,self.fixture['K'],motion)
        self.assertIn((1,0),forbidden)  # The existing background conflict gate remains active.
        self.assertNotIn((1,16),forbidden)
        original=deepcopy(r.track_objects)
        prepared,core,q,init=r.prepare_pair(4922,4924,a,self.fixture['K'],motion,
                                           list(models),labels,tracks)
        self.assertEqual(init['initial_local_to_object'][1],16)
        final=r.finish_pair(4922,4924,prepared,q,tracks,0,'red_fixture',labels=core,
                            anchors=a,K=self.fixture['K'],motion_cfg=motion)
        self.assertEqual(final['1'],16)
        self.assertEqual(r.track_objects[992],original[992])
        self.assertEqual(r.track_probabilities[992],{0:1.})
        self.assertEqual(r._geometric_identity_pair,(4922,4924))

    def test_disabling_ledger_retains_the_recorded_identity_rejection(self):
        r,a,models,labels,tracks,motion=self.registry(False)
        q,_=motion_probabilities(motion_costs(a,models,self.fixture['K'],motion['hybrid']),motion['hybrid'])
        forbidden,report=r._motion_identity_gate(models,labels,q,tracks,a,self.fixture['K'],motion)
        self.assertIn((1,16),forbidden)
        rejection=next(v for v in report['rejected'] if v['local_model']==1 and v['object_id']==16)
        self.assertEqual(rejection['identity_support'],2)
        self.assertEqual(rejection['reason'],'insufficient_compatible_identity_support')

    def test_conflicting_current_foreground_identity_is_not_overwritten(self):
        r,a,models,labels,tracks,motion=self.registry();r.track_objects[992]=17
        view,report=self.view(r,a,models,labels,tracks,motion)
        self.assertEqual(view[992],17)
        self.assertIn(992,report['conflicting_track_ids'])
        self.assertEqual(sum(view.get(int(t))==16 for t in tracks[labels==1]),2)

    def test_current_metric_failure_cannot_supply_a_ledger_vote(self):
        r,a,models,labels,tracks,motion=self.registry()
        i=int(np.flatnonzero(tracks==992)[0]);a['x1'][i]+=[10.,0.,10.]
        view,_=self.view(r,a,models,labels,tracks,motion)
        self.assertEqual(view[992],0)
        self.assertEqual(sum(view.get(int(t))==16 for t in tracks[labels==1]),2)

    def test_unknown_current_rows_cannot_supply_a_ledger_vote(self):
        r,a,models,labels,tracks,motion=self.registry();labels[tracks==992]=-1
        view,_=self.view(r,a,models,labels,tracks,motion)
        self.assertEqual(view[992],0)

    def test_conflicting_duplicate_track_rows_cannot_cast_a_ledger_vote(self):
        r,a,models,labels,tracks,motion=self.registry()
        tracks[tracks==1679]=992  # Same track now claims two exclusive motions.
        view,_=self.view(r,a,models,labels,tracks,motion)
        self.assertEqual(view[992],0)

    def test_skipped_pair_expires_all_geometric_identity(self):
        r,a,models,labels,tracks,motion=self.registry();r._begin_pair(4924)
        self.assertEqual(r._geometric_identity_tracks,{})
        self.assertIsNone(r._geometric_identity_pair)
        view,_=self.view(r,a,models,labels,tracks,motion)
        self.assertEqual(view[992],0)

    def test_retention_retirement_and_finalization_prune_geometry(self):
        r,*_=self.registry();r.retain_tracks([54,992])
        self.assertEqual(set(r._geometric_identity_tracks),{54,992})
        r.discard_empty_objects(4922,[16])
        self.assertEqual(r._geometric_identity_tracks,{})
        r,*_=self.registry();r.finalize_sequence()
        self.assertEqual(r._geometric_identity_tracks,{})
        self.assertIsNone(r._geometric_identity_pair)

    def test_unknown_previous_rows_and_fewer_than_six_points_do_not_renew(self):
        r,a,models,labels,tracks,motion=self.registry()
        # Even a strong conditional probability cannot mint geometric history.
        labels[labels==1]=-1
        r._remember_geometric_identity(4922,4924,models,labels,tracks,
            {i:16 for i in range(len(models))},a,self.fixture['K'],motion)
        self.assertNotIn(992,r._geometric_identity_tracks)
        few=np.flatnonzero(self.fixture['current_labels']==1)[:5]
        short={k:v[few] for k,v in a.items()}
        r._remember_geometric_identity(4922,4924,models,np.ones(5,int),tracks[few],
            {i:16 for i in range(len(models))},short,self.fixture['K'],motion)
        self.assertEqual(r._geometric_identity_tracks,{})


if __name__=='__main__':unittest.main()
