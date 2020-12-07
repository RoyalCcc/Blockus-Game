"""
This file is for test blockus_data.py
"""
import numpy as np
import torch as tr
import unittest as ut
import blockus_data as bd
import blockus_game as bg

class BlockusDataTestCase(ut.TestCase):

    def test_encode(self):
        state = bg.initial_state(board_size=2)
        actual = bd.encode(state)
        expected = tr.zeros(3,2,2)
        expected[0,:,:] = 1.
        self.assertTrue(tr.allclose(actual, expected))
        
        state = state.perform(state.valid_actions()[0])
        actual = bd.encode(state)
        expected[:2,0,0] = tr.tensor([0.,1.])
        self.assertTrue(tr.allclose(actual, expected))

        state = state.perform(state.valid_actions()[0])
        actual = bd.encode(state)
        expected[[0, 2],1,1] = tr.tensor([0.,1.])
        self.assertTrue(tr.allclose(actual, expected))

    def test_get_batch(self):
        inputs, outputs = bd.get_batch(
            board_size=2, polyomino_size=2, num_games=1, num_rollouts=1, max_depth=2,
            choose_method=lambda node: node.children()[0])
        self.assertTrue(tr.allclose(outputs, tr.zeros(6,1)))
        
        inputs, outputs = bd.get_batch(
            board_size=2, polyomino_size=2, num_games=1, num_rollouts=7, max_depth=2,
            choose_method=lambda node: node.children()[np.argmin(node.get_visit_counts())])
        self.assertTrue(tr.allclose(outputs, tr.tensor([[-2/3,1/2,1/2,-1,0]]).t()))
        expected = tr.zeros(3,2,2)
        expected[0] = 1.
        expected[:2,0,0] = tr.tensor([0, 1])
        self.assertTrue(tr.allclose(inputs[0], expected))
        
if __name__ == "__main__":    
    
    test_suite = ut.TestLoader().loadTestsFromTestCase(BlockusDataTestCase)
    res = ut.TextTestRunner(verbosity=2).run(test_suite)
    num, errs, fails = res.testsRun, len(res.errors), len(res.failures)
    print("score: %d of %d (%d errors, %d failures)" % (num - (errs+fails), num, errs, fails))
    

