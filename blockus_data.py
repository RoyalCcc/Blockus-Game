"""
/////////////////////////////////////////////////
// blockus_data.py                             //
// Author: Chen Luo                            //
// SUID:230553092                              //
// Last Change Time:11/18/2020                 //
// Copyright @ 2020 Chen. All rights reserved. //
/////////////////////////////////////////////////
"""
import itertools as it
import blockus_game as bg
import mcts as mcts
import torch as tr

def generate(board_size=6, polyomino_size=5, num_games=2, num_rollouts=10, max_depth=4, choose_method=None):

    if choose_method is None: choose_method = mcts.puct

    data = []    
    for game in range(num_games):
    
        state = bg.initial_state(board_size, polyomino_size)
        for turn in it.count():
            print("game %d, turn %d..." % (game, turn))
    
            # Stop when game is over
            if state.is_leaf(): break
    
            # Act immediately if only one action available
            valid_actions = state.valid_actions()
            if len(valid_actions) == 1:
                state = state.perform(valid_actions[0])
                continue
            
            # Otherwise, use MCTS
            a, node = mcts.decide_action(state, num_rollouts, choose_method, max_depth)
            state = node.children()[a].state
            
            # Add child states and their values to the data
            Q = node.get_score_estimates()
            for c,child in enumerate(node.children()):
                data.append((child.state, Q[c]))

    return data

def encode(state):
    expected = tr.zeros(3, len(state.board), len(state.board[0]))
    for r in range(len(state.board)):
        for c in range(len(state.board[0])):
            s = state.board[r][c]
            expected[s][r][c] = 1
    return expected

def get_batch(board_size=6, polyomino_size=5, num_games=2, num_rollouts=50, max_depth=6, choose_method=None):
    data = generate(board_size,polyomino_size,num_games,num_rollouts,max_depth,choose_method)
    input = tr.zeros(len(data),3,board_size,board_size)
    output = tr.zeros(len(data),0)
    for i in range(len(data)):
        e_state = encode(data[i][0])
        input[i] = e_state
        output[i] = data[i][1]
    return (input, output)

if __name__ == "__main__":
    
    board_size, num_games = 6, 50
    inputs, outputs = get_batch(board_size, num_games=num_games)

    import pickle as pk
    with open("data%d.pkl" % board_size, "wb") as f: pk.dump((inputs, outputs), f)

