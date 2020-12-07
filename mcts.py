"""
/////////////////////////////////////////////////
// mcts.py                                     //
// Author: Chen Luo                            //
// SUID:230553092                              //
// Last Change Time:11/18/2020                 //
// Copyright @ 2020 Chen. All rights reserved. //
/////////////////////////////////////////////////
"""
import numpy as np

def uniform(node):
    c = np.random.choice(len(node.children()))
    return node.children()[c]

def puct(node):
    c = np.random.choice(len(node.children()), p=puct_probs(node))
    return node.children()[c]

def puct_probs(node):
    n = node.get_visit_counts()
    q = node.get_score_estimates()
    res = []
    for i in range(len(n)):
        prob = q[i] + np.sqrt((np.log(node.visit_count + 1)/(n[i]+1)))
        res.append(prob)
    res = np.exp(res) / np.sum(np.exp(res))
    return res

class Node(object):
    def __init__(self, state, depth = 0, choose_method=uniform):
        self.state = state
        self.child_list = None
        self.visit_count = 0
        self.score_total = 0
        self.depth = depth
        self.choose_method = choose_method
    def make_child_list(self):
        self.child_list = []
        for action in self.state.valid_actions():
            state = self.state.perform(action)
            node = Node(state, self.depth+1)
            self.child_list.append(node)
        return self.child_list
    def children(self):
        if self.child_list is None: self.make_child_list()
        return self.child_list
    def get_score_estimates(self):
        res = []
        for i in range(len(self.children())):
            if self.child_list[i].visit_count == 0:
                res.append(0)
            else:
                score=self.child_list[i].score_total/self.child_list[i].visit_count
                if not self.state.is_max_players_turn():
                    score = 0 - score
                res.append(score)
        return np.asarray(res)
    def get_visit_counts(self):
        res = []
        for i in range(len(self.children())):
            res.append(self.child_list[i].visit_count)
        return np.asarray(res)
    def choose_child(self):
        return self.choose_method(self)

def rollout(node, max_depth=None):
    if node.depth == max_depth or node.state.is_leaf():
        result = node.state.score_for_max_player()
    else:
        result = rollout(node.choose_child(), max_depth)
    node.visit_count += 1
    node.score_total += result
    return result

def decide_action(state, num_rollouts, choose_method=puct, max_depth=10, verbose=False):
    node = Node(state, choose_method=choose_method)
    for n in range(num_rollouts):
        if verbose and n % 10 == 0: print("Rollout %d of %d..." % (n+1, num_rollouts))
        rollout(node, max_depth=max_depth)
    return np.argmax(node.get_score_estimates()), node

