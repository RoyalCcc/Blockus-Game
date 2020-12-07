"""
/////////////////////////////////////////////////
// blockus_net.py                              //
// Author: Chen Luo                            //
// SUID:230553092                              //
// Last Change Time:11/18/2020                 //
// Copyright @ 2020 Chen. All rights reserved. //
/////////////////////////////////////////////////
"""
import numpy as np
import torch as tr
from torch.nn import Sequential, Conv2d, Linear, Flatten, LeakyReLU, Tanh

def BlockusNet1(board_size):
    model = Sequential(
        Flatten(),
        Linear(3*board_size*board_size,1)
    )
    return model

def calculate_loss(net, x, y_targ):
    output = net(x)
    e = tr.sum((output-y_targ)*(output-y_targ))
    return (output,e)

def optimization_step(optimizer, net, x, y_targ):
    optimizer.zero_grad()
    y, e = calculate_loss(net, x, y_targ)
    e.backward()
    optimizer.step()
    return (y,e)

if __name__ == "__main__":

    board_size = 6
    net = BlockusNet1(board_size=board_size)
    print(net)

    import pickle as pk
    with open("data%d.pkl" % board_size,"rb") as f: (x, y_targ) = pk.load(f)

    # Optimization loop
    optimizer = tr.optim.Adam(net.parameters())
    train_loss, test_loss = [], []
    shuffle = np.random.permutation(range(len(x)))
    split = 10
    train, test = shuffle[:-split], shuffle[-split:]
    for epoch in range(5000):
        y_train, e_train = optimization_step(optimizer, net, x[train], y_targ[train])
        y_test, e_test = calculate_loss(net, x[test], y_targ[test])
        if epoch % 10 == 0: print("%d: %f (%f)" % (epoch, e_train.item(), e_test.item()))
        train_loss.append(e_train.item() / (len(shuffle)-split))
        test_loss.append(e_test.item() / split)
    
    tr.save(net.state_dict(), "model%d.pth" % board_size)
    
    import matplotlib.pyplot as pt
    pt.plot(train_loss,'b-')
    pt.plot(test_loss,'r-')
    pt.legend(["Train","Test"])
    pt.xlabel("Iteration")
    pt.ylabel("Average Loss")
    pt.show()
    
    pt.plot(y_train.detach().numpy(), y_targ[train].detach().numpy(),'bo')
    pt.plot(y_test.detach().numpy(), y_targ[test].detach().numpy(),'ro')
    pt.legend(["Train","Test"])
    pt.xlabel("Actual output")
    pt.ylabel("Target output")
    pt.show()

    

