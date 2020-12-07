"""
This file is for test blockus_net.py
"""
import numpy as np
import torch as tr
import unittest as ut
import blockus_net as bn

class BlockusNetTestCase(ut.TestCase):

    def test_blockus_net1(self):
        net = bn.BlockusNet1(board_size=7)
        actual_shapes = set([tuple(param.shape) for param in net.parameters()])
        expected_shapes = set([(1, 147), (1,)])
        self.assertTrue(actual_shapes == expected_shapes)

    def test_calculate_loss1(self):
        net = tr.nn.Linear(8,1)
        net.weight.data[:] = 1
        net.bias.data[:] = 1
        x = tr.ones((3,8))
        y_targ = tr.zeros((3,1))
        y, e = bn.calculate_loss(net, x, y_targ)
        self.assertTrue(tr.allclose(y, 9*tr.ones((3,1))))
        self.assertTrue(tr.allclose(e, tr.tensor(3.*9**2)))

    def test_calculate_loss2(self):
        net = bn.BlockusNet1(board_size=7)
        for param in net.parameters(): param.data[:] = 1
        x = tr.ones((2,3,7,7))
        y_targ = tr.zeros((2,1))
        y, e = bn.calculate_loss(net, x, y_targ)
        self.assertTrue(tr.allclose(e, tr.tensor(43808.000000)))

    def test_optimization_step(self):
        net = bn.BlockusNet1(board_size=7)
        for param in net.parameters():
            param.data[:] = 1
            param.grad = tr.ones(param.data.shape)
        optimizer = tr.optim.Adam(net.parameters())
        x = tr.ones((2,3,7,7))
        y_targ = tr.zeros((2,1))
        _, e = bn.optimization_step(optimizer, net, x, y_targ)
        self.assertTrue(tr.allclose(e, tr.tensor(43808.000000)))
        actual_sum = sum([param.grad.sum() for param in net.parameters()])
        expected_sum = tr.tensor(87616.000000)
        self.assertTrue(tr.allclose(actual_sum, expected_sum))
        
if __name__ == "__main__":    
    
    test_suite = ut.TestLoader().loadTestsFromTestCase(BlockusNetTestCase)
    res = ut.TextTestRunner(verbosity=2).run(test_suite)
    num, errs, fails = res.testsRun, len(res.errors), len(res.failures)
    print("score: %d of %d (%d errors, %d failures)" % (num - (errs+fails), num, errs, fails))
    
