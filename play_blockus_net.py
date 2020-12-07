import itertools as it
import numpy as np
import torch as tr
import blockus_game as bg
import mcts as mcts
import blockus_data as bd
import blockus_net as bn

def get_int(prompt, low, high):
    valid = list(map(str, range(low,high)))
    while True:
        inp = input(prompt + " (%d to %d): " % (low, high-1))
        if inp in valid: return int(inp)
        print("Invalid choice...")

board_size = 6
net = bn.BlockusNet1(board_size)
net.load_state_dict(tr.load("model%d.pth" % board_size))

def nn_puct(node):
    with tr.no_grad():
        x = tr.stack(tuple(map(bd.encode, [child.state for child in node.children()])))
        y = net(x)
        probs = tr.softmax(y.flatten(), dim=0)
        a = np.random.choice(len(probs), p=probs.detach().numpy())
    return node.children()[a]

if __name__ == "__main__":
    
    state = bg.initial_state(board_size = 6)
    for step in it.count():
        print(state)
        print("Step %d" % step)
        
        # Stop when game is over
        if state.is_leaf(): break

        # Act immediately if only one action available
        valid_actions = state.valid_actions()
        if len(valid_actions) == 1:
            state = state.perform(valid_actions[0])
            continue

        # Otherwise, if it is the AI's turn (min), run MCTS to decide its action
        if not state.is_max_players_turn():
            a, node = mcts.decide_action(state,
                choose_method=nn_puct,
                num_rollouts=100, max_depth = 10, verbose=True)
            state = node.children()[a].state
            continue

        # Otherwise, get next move from user
        valid_polyominoes, valid_poses, valid_rows, valid_cols = zip(*valid_actions)
        while True: # repeat until user chooses a valid action

            # get polyomino choice
            p = get_int("Choose a piece", 0, len(state.hands[state.turn]))
            polyomino = state.hands[state.turn][p]
            if polyomino not in valid_polyominoes:
                print("No valid actions, try again")
                continue

            # get pose choice
            chars = np.array([" ", bg.RED, bg.BLUE])
            s = "Poses for piece %d:\n" % p
            pose = np.full((max(polyomino.array.shape), 80), " ")
            offset = 0
            for p, arr in enumerate(polyomino.poses):
                if offset + 2*arr.shape[1] + 2 > 80:
                    s += "\n".join(["".join(row) for row in pose]) + "\n"
                    pose = np.full((max(polyomino.array.shape), 80), " ")
                    offset = 0
                num = "%d" % p
                pose[0,offset:offset+len(num)] = list(num)
                offset += len(num)
                pose[:arr.shape[0],offset:offset+2*arr.shape[1]] = np.repeat(chars[[0,state.turn]][arr], 2, axis=1)
                offset += 2*arr.shape[1]+1
            s += "\n".join(["".join(row) for row in pose]) + "\n"
            print(s)
            pose = get_int("Choose a pose", 0, len(polyomino.poses))
            if (polyomino, pose) not in zip(valid_polyominoes, valid_poses):
                print("No valid actions, try again")
                continue
            
            # get position choice
            chars = np.array([bg.OPEN, bg.RED, bg.BLUE])
            valid_board = np.repeat(chars[state.board], 2, axis=1)
            valid_positions = []
            for (row, col) in it.product(range(state.size), repeat=2):
                if (polyomino, pose, row, col) in valid_actions:
                    valid_positions.append((row, col))
                    num = "%2d" % (len(valid_positions)-1)
                    valid_board[row,2*col:2*col+2] = list(num)
            print("Valid (upper, left) positions for pose %d:" % pose)
            print("\n".join(["".join(row) for row in valid_board]) + "\n")
            a = get_int("Choose a position", 0, len(valid_positions))
            row, col = valid_positions[a]
    
            # action was valid, exit the busy loop
            break

        # perform selected action
        action = polyomino, pose, row, col
        state = state.perform(action)

    print("Game over!")
    




