"""CPU checks for selective solver proposals and unchanged hard constraints."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from scipy import sparse
EXP=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(EXP))
from ownership.object_connectivity_constraints import condition_core_initialization,no_core_background_policy
from ownership.object_spatial_constraints import prepare_spatial_constraints
from ownership.final_connectivity import finalize_connectivity


def scene():
    p=np.array([[.1,.2,.4,.3],[.7,.1,.1,.1],[.1,.5,.2,.2],[.1,.2,.1,.6],[.4,0,.4,.2]])
    state=dict(active_gids=np.arange(5),object_ids=np.array([0,1,3,8]),
        core_by_object={1:np.array([0,1,2]),3:np.array([0,1,2])},
        object_core_groups={3:[np.array([0,1,2])]},
        options=dict(initialization_conditioning='certified_core_winner',core_min_points=3,
            enabled=True,no_core_to_background=True,final_hard_assignment=True),diagnostics={},
        observed=np.ones(p.shape,bool),compatible=np.ones(p.shape,bool),
        adjacency=sparse.csr_matrix(np.ones((5,5))-np.eye(5)))
    return state,p


class CoreInitializationTests(unittest.TestCase):
    def test_only_certified_original_winner_rows_are_conditioned(self):
        state,p=scene();original=p.copy()
        result=condition_core_initialization(state,p);q=result['probabilities']
        np.testing.assert_allclose(q[0],[1/7,2/7,4/7,0])
        np.testing.assert_array_equal(q[1:],p[1:])
        np.testing.assert_array_equal(p,original)
        self.assertEqual(result['report']['eligible_object_ids'],[3])
        self.assertEqual(result['report']['selected_count'],1)
        self.assertAlmostEqual(result['report']['removed_probability_mass'],.3)
        self.assertEqual(result['report']['zero_mass_count'],0)
        # Background remains among the supported competitors.
        self.assertGreater(q[0,0],0)

    def test_native_core_without_certified_group_gets_no_special_proposal(self):
        state,p=scene();state['object_core_groups']={}
        result=condition_core_initialization(state,p)
        np.testing.assert_array_equal(result['probabilities'],p)
        self.assertEqual(result['report']['eligible_object_ids'],[])

    def test_invalid_small_duplicate_or_foreign_groups_do_not_qualify(self):
        for group in ([0,1],[0,0,1],[0,1,4],[-1,0,1],[0.,1.,2.]):
            with self.subTest(group=group):
                state,p=scene();state['object_core_groups']={3:[np.array(group)]}
                np.testing.assert_array_equal(condition_core_initialization(state,p)['probabilities'],p)

    def test_background_group_never_selects_background_winners(self):
        state,p=scene();state['core_by_object'][0]=np.arange(5)
        state['object_core_groups'][0]=[np.arange(5)]
        q=condition_core_initialization(state,p)['probabilities']
        np.testing.assert_array_equal(q[1],p[1]);np.testing.assert_array_equal(q[4],p[4])

    def test_ties_use_smallest_object_id_and_are_column_permutation_invariant(self):
        state,p=scene();p[2]=[.1,.4,.4,.1] # Native O1 beats certified O3 in a tie.
        a=condition_core_initialization(state,p)['probabilities'];permutation=[3,2,1,0]
        other={**state,'object_ids':state['object_ids'][permutation]}
        b=condition_core_initialization(other,p[:,permutation])['probabilities']
        np.testing.assert_array_equal(a,b[:,np.argsort(permutation)])
        np.testing.assert_array_equal(a[2],p[2]);np.testing.assert_array_equal(a[4],p[4])

    def test_default_none_is_exact_copy_and_unknown_modes_are_rejected(self):
        state,p=scene();state['options'].pop('initialization_conditioning')
        q=condition_core_initialization(state,p)['probabilities']
        np.testing.assert_array_equal(q,p);self.assertIsNot(q,p)
        state['options']['initialization_conditioning']='all_winners'
        with self.assertRaises(ValueError):condition_core_initialization(state,p)

    def test_no_core_and_final_owned_core_constraints_remain_effective(self):
        state,p=scene();q=condition_core_initialization(state,p)['probabilities']
        fallback=no_core_background_policy(state,q)
        np.testing.assert_array_equal(fallback['probabilities'][3],[1,0,0,0])
        np.testing.assert_array_equal(fallback['probabilities'][:,3],0.)
        final=finalize_connectivity(state,fallback['probabilities'],assignment_support=np.ones_like(p,bool),
            min_probability=.6,min_margin=.1,hard_background_mask=np.array([True,False,False,False,False]))
        np.testing.assert_array_equal(final['probabilities'][0],[1,0,0,0])
        self.assertIn(3,final['report']['disabled_object_ids'])

    def prepare(self,changes=None):
        state,p=scene();cfg=dict(render=dict(enabled=True,initialization='measurement'),
            connectivity=state['options'].copy())
        if changes:
            for section,key,value in changes:cfg[section][key]=value
        state['options']=cfg['connectivity']
        initial=np.array([.9,.05,.04,.01]);energy=np.arange(20.).reshape(5,4)
        fixed_prior=initial.copy();fixed_energy=energy.copy();fixed_proposal=p.copy()
        window=SimpleNamespace(_active_gids=np.arange(5),_active={},object_ids=[0,1,3,8],
            _render_initial_probabilities=p,_initial={7:initial},_render_energy=energy)
        with patch('ownership.object_connectivity_constraints.build_connectivity_state',return_value=state), \
             patch('ownership.object_connectivity_constraints.update_connectivity_targets',return_value={'report':{}}):
            result=prepare_spatial_constraints(window,cfg)
        np.testing.assert_array_equal(window._initial[7],fixed_prior)
        np.testing.assert_array_equal(window._render_energy,fixed_energy)
        np.testing.assert_array_equal(window._render_initial_probabilities,fixed_proposal)
        return result,state,p

    def test_prepare_conditions_before_original_fallback_without_mutating_first_prior(self):
        result,state,p=self.prepare();q=condition_core_initialization(state,p)['probabilities']
        expected=no_core_background_policy(state,q)
        np.testing.assert_allclose(result['no_core_initial_probabilities'],expected['probabilities'])
        self.assertTrue(result['report']['connectivity']['initialization_conditioning']['enabled'])

    def test_missing_any_render_or_hard_constraint_precondition_disables_proposal(self):
        cases=[('render','enabled',False),('render','initialization','previous'),
               ('connectivity','no_core_to_background',False),('connectivity','final_hard_assignment',False)]
        for change in cases:
            with self.subTest(change=change):
                result,state,p=self.prepare([change])
                np.testing.assert_allclose(result['no_core_initial_probabilities'],
                    no_core_background_policy(state,p)['probabilities'])
                self.assertFalse(result['report']['connectivity']['initialization_conditioning']['enabled'])
        result,_,_=self.prepare([('connectivity','enabled',False)])
        self.assertIsNone(result['no_core_initial_probabilities'])


if __name__=='__main__':unittest.main()
