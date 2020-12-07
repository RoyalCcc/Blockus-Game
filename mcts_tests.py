"""
This file is for test mcts.py
"""
import numpy as np
import torch as tr
import unittest as ut
import mcts as ms
import blockus_game as bg

class TestState(object):
    def __init__(self, is_max_players_turn):
        self.is_max_players_turn = is_max_players_turn

def make_state(max_player):
    return TestState(lambda : max_player)

def make_node(max_player, visit_count, score_total):
    node = ms.Node(make_state(max_player))
    node.score_total = score_total
    node.visit_count = visit_count    
    return node

def make_nodes(Mp, Np, Wp, M, N, W):
    node = make_node(Mp, Np, Wp)
    node.child_list = [make_node(M[c], N[c], W[c]) for c in range(len(M))]
    return node
    
class MCTSTestCase(ut.TestCase):

    def test_get_visit_counts(self):
        parent = make_nodes(True, 9, 0, [False, False], [6, 3], [1, 4])
        self.assertTrue(np.allclose(
            parent.get_visit_counts(),
            np.array([6, 3])))

    def test_get_score_estimates(self):
        parent = make_nodes(True, 10, 0, [False, False], [3, 7], [1, 4])
        self.assertTrue(np.allclose(parent.get_score_estimates(), np.array([1/3, 4/7])))

        parent = make_nodes(False, 10, 0, [True, True], [3, 7], [1, 4])
        self.assertTrue(np.allclose(parent.get_score_estimates(), -np.array([1/3, 4/7])))

        parent = make_nodes(False, 3, 0, [True, True], [3, 0], [2, 0])
        self.assertTrue(np.allclose(parent.get_score_estimates(), -np.array([2/3, 0])))

    def test_make_child_list(self):
        node = ms.Node(bg.initial_state(board_size=2, polyomino_size=2))
        self.assertTrue(node.child_list is None)
        node.make_child_list()
        self.assertTrue(len(node.child_list)  == 3)
        for child in node.children(): self.assertTrue(child.depth == 1)

        node = ms.Node(bg.initial_state(board_size=2, polyomino_size=2))
        self.assertTrue(node.child_list is None)
        child_list = node.children()
        self.assertTrue(len(child_list) == 3)
        self.assertTrue(child_list == node.child_list)

        for c,Ngc in enumerate([3, 2, 2]):
            child = child_list[c]
            self.assertTrue(child.child_list is None)
            child.children()
            self.assertTrue(len(child.child_list) == Ngc)

    def test_puct_probs(self):

        parent = make_nodes(True, 10, 0, [False, False], [3, 7], [1, 4])
        probs = ms.puct_probs(parent)
        self.assertTrue(np.allclose(probs, np.array([0.49716987, 0.50283013])))

        parent = make_nodes(False, 10, 0, [True, True], [3, 7], [1, 4])
        probs = ms.puct_probs(parent)
        self.assertTrue(np.allclose(probs, np.array([0.6141688, 0.3858312])))

        parent = make_nodes(False, 3, 0, [True, True], [3, 0], [2, 0])
        probs = ms.puct_probs(parent)
        self.assertTrue(np.allclose(probs, np.array([0.22177166, 0.77822834])))
        
if __name__ == "__main__":    
    
    test_suite = ut.TestLoader().loadTestsFromTestCase(MCTSTestCase)
    res = ut.TextTestRunner(verbosity=2).run(test_suite)
    num, errs, fails = res.testsRun, len(res.errors), len(res.failures)
    print("score: %d of %d (%d errors, %d failures)" % (num - (errs+fails), num, errs, fails))
    
