"""CPU integration checks of render initialization, not renderer accuracy."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

EXP=Path(__file__).resolve().parents[1]
if str(EXP) not in sys.path:sys.path.insert(0,str(EXP))
from ownership.object_window import ObjectWindow


class MeasurementInitializationTests(unittest.TestCase):
    def config(self,mode=None,render=True):
        options={'enabled':render}
        if mode is not None:options['initialization']=mode
        return {'sequence':{'streaming':True},'object_window':{
            'window_frames':4,'window_step':3,'temperature':.25,
            'prior_weight':.05,'prior_floor':.001,'render':options}}

    def update(self,window,first,oids,costs,*,fresh=False):
        count=len(oids)
        payload={'xyz':np.array([[0.,0.,5.]]),'uv':np.array([[0.,0.]]),
                 'color':np.ones((1,3)),'scale':np.full((1,3),.1),'K':np.eye(3),
                 'H_object':np.repeat(np.eye(4)[None],count,axis=0)}
        observations={k:np.zeros((2,2,3),np.uint8) for k in ['source','target']}
        observations.update({k:np.full((2,2),5.) for k in ['source_depth','target_depth']})
        observations.update({k:np.ones((2,2)) for k in ['source_confidence','target_confidence']})
        q=np.array([[.01 if oid==0 else .99 for oid in oids]]) if count>1 else np.ones((1,1))
        return window.update(first,first+1,np.array([42]),np.array(oids),q,np.asarray([costs]),
            payload=payload,render_observations=observations,q_motion_initial=np.array([[.2,.3,.5]]))

    def sequence(self,mode=None,*,render=True,reverse=False,unavailable=False):
        window=ObjectWindow(self.config(mode,render));captured=[];collected=[]
        def solver(current,cfg,*,priors):
            initial=current._render_initial_probabilities.copy()
            captured.append({'initial':initial,'priors':priors.copy(),
                             'energy':current._render_energy.copy(),
                             'ids':list(current.object_ids)})
            # The initializer itself supplies no new visibility/fit support.
            support=np.zeros_like(initial,bool) if unavailable else np.ones_like(initial,bool)
            return {'probabilities':initial,'render_support':support,
                    'report':{'render_passes':0,'iterations_completed':0}}
        with patch('ownership.object_render_optimizer.optimize_render_window',side_effect=solver):
            self.update(window,0,[0],[.01])
            collected.append(window._active_probabilities.copy())
            for first in [1,2]:
                ids=[3,0] if reverse else [0,3]
                costs=[(.01 if oid==3 else .99) for oid in ids]
                if unavailable:costs=[np.nan]*len(ids)
                result=self.update(window,first,ids,costs)
                if first==1:collected.append(window._active_probabilities.copy())
        return window,result,captured,collected

    def test_new_object_can_start_from_fresh_window_evidence(self):
        measured,result,captured,_=self.sequence('measurement')
        baseline,analytic,_,_=self.sequence(render=False)
        initial=captured[0]['initial']
        np.testing.assert_allclose(initial,analytic['object_probabilities'],atol=1e-14)
        self.assertGreater(initial[0,measured.object_ids.index(3)],.99)
        self.assertEqual(result['report']['render_optimization']['initialization'],'measurement')
        # The first prior remains background: it did not become this proposal.
        np.testing.assert_allclose(captured[0]['priors'],[[1/1.001,.001/1.001]])
        np.testing.assert_array_equal(measured._initial[42],[1.])
        for snapshot in measured._active.values():
            np.testing.assert_array_equal(snapshot['q_motion_initial'],[[.2,.3,.5]])

    def test_previous_default_remains_compatible_and_collecting_is_unchanged(self):
        previous,_,old,old_collecting=self.sequence()
        explicit,_,explicit_capture,explicit_collecting=self.sequence('previous')
        measured,_,new,new_collecting=self.sequence('measurement')
        np.testing.assert_array_equal(old[0]['initial'],explicit_capture[0]['initial'])
        self.assertGreater(old[0]['initial'][0,previous.object_ids.index(0)],.99)
        for old_state,new_state,explicit_state in zip(old_collecting,new_collecting,explicit_collecting):
            np.testing.assert_array_equal(old_state,new_state)
            np.testing.assert_array_equal(old_state,explicit_state)
        np.testing.assert_array_equal(old[0]['priors'],new[0]['priors'])
        np.testing.assert_array_equal(old[0]['energy'],new[0]['energy'])

    def test_local_candidate_column_order_does_not_change_global_initialization(self):
        left,_,a,_=self.sequence('measurement')
        right,_,b,_=self.sequence('measurement',reverse=True)
        columns=[right.object_ids.index(oid) for oid in left.object_ids]
        np.testing.assert_allclose(a[0]['initial'],b[0]['initial'][:,columns],atol=1e-14)
        np.testing.assert_allclose(a[0]['priors'],b[0]['priors'][:,columns],atol=1e-14)

    def test_missing_costs_are_not_zero_error_support_for_new_candidates(self):
        window,result,captured,_=self.sequence('measurement',unavailable=True)
        baseline,analytic,_,_=self.sequence(render=False,unavailable=True)
        np.testing.assert_array_equal(captured[0]['energy'],np.zeros((1,2)))
        np.testing.assert_allclose(captured[0]['initial'],analytic['object_probabilities'],atol=1e-14)
        self.assertGreater(captured[0]['initial'][0,window.object_ids.index(0)],.5)
        self.assertLess(captured[0]['initial'][0,window.object_ids.index(3)],.5)
        np.testing.assert_array_equal(analytic['object_labels'],[-1])
        np.testing.assert_array_equal(result['object_labels'],[-1])


if __name__=='__main__':unittest.main()
