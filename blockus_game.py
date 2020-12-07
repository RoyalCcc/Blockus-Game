import itertools as it
import numpy as np
from polyomino import polyominoes

OPEN = "\u2591"
RED = "\u2592"
BLUE = "\u2593"

class State(object):
    def __init__(self, size, turn, hands, plays, board=None):

        self.size = size # side length of board
        self.turn = turn # current player integer (1, 2, ...)

        # hand is list of polyominoes
        # play is list of (row, col, posed_array), row/col are upper left
        self.hands = {player: list(hand) for (player, hand) in hands.items()}
        self.plays = {player: list(play) for (player, play) in plays.items()}

        # board uses 1-based player indices at claimed cells, 0 at open cells
        if board is None: board = np.zeros((self.size,self.size), dtype=np.int64)
        self.board = board.copy()
        
        # cache valid actions
        self.valid_action_list = None

    def __str__(self):

        chars = np.array([OPEN, RED, BLUE])
        # board string
        s = "\n".join(["".join(row) for row in np.repeat(chars[self.board], 2, axis=1)]) + "\n"

        # hand strings
        chars = np.array([" ", RED, BLUE])
        for player in sorted(self.hands.keys()):
            s += "\n"
            hand = np.full((3, 80), " ")
            offset = 0
            for p, polyomino in enumerate(self.hands[player]):
                arr = polyomino.array
                if offset + 2*arr.shape[1] + 2 > 80:
                    s += "\n".join(["".join(row) for row in hand]) + "\n"
                    hand = np.full((3, 80), " ")
                    offset = 0
                num = "%d" % p
                hand[0,offset:offset+len(num)] = list(num)
                offset += len(num)
                hand[:arr.shape[0],offset:offset+2*arr.shape[1]] = np.repeat(chars[[0,player]][arr], 2, axis=1)
                offset += 2*arr.shape[1]+1
            s += "\n".join(["".join(row) for row in hand]) + "\n"
        s += "Player 1 score = %d\n" % self.score_for_max_player()
        s += "Turn: Player %d (%s)" % (self.turn, chars[self.turn]*2)

        return s

    def copy(self):
         return State(self.size, self.turn, self.hands, self.plays, self.board)

    def is_leaf(self):
        if self.valid_actions() != [None]: return False
        else:
            # check if other player also has no moves
            other = self.perform(action=None)
            return other.valid_actions() == [None]

    def score_for_max_player(self):
        return (self.board == 1).sum() - (self.board == 2).sum()
    
    def is_max_players_turn(self):
        return (self.turn == 1)

    def valid_actions(self):
        # action is (polyomino, p, row, col) tuple
        # p is pose index
        
        # use cached results if available
        if self.valid_action_list is not None:
            return self.valid_action_list
        
        # make padded board for simpler indexing
        padded = np.zeros((self.size+4, self.size+4), dtype=int)
        padded[:self.size, :self.size] = self.board
        
        # get cells within board limits
        board_cells = np.zeros(padded.shape, dtype=bool)
        board_cells[:self.size, :self.size] = True
        
        # get open cells
        open_cells = (padded == 0)
        
        # get cells with current player's color
        current = (padded == self.turn)
        
        # get cells that share sides with current player's color
        sides = np.zeros(padded.shape, dtype=bool)
        sides[:,:-1] |= current[:,+1:]
        sides[:,+1:] |= current[:,:-1]
        sides[:-1,:] |= current[+1:,:]
        sides[+1:,:] |= current[:-1,:]
        
        # get cells that share corners with current player's color
        corners = np.zeros(padded.shape, dtype=bool)
        corners[:-1,:-1] |= current[+1:,+1:]
        corners[:-1,+1:] |= current[+1:,:-1]
        corners[+1:,:-1] |= current[:-1,+1:]
        corners[+1:,+1:] |= current[:-1,:-1]
        
        # include starting positions
        if self.turn == 1: corners[0,0] = True # upper left
        if self.turn == 2: corners[self.size-1,self.size-1] = True # lower right

        # iterate over possible polyomino poses
        actions = []
        for polyomino in self.hands[self.turn]:
            for p, posed_offsets in enumerate(polyomino.posed_offsets):
                
                # positions fully on board
                on_board = np.stack([
                    board_cells[dr:self.size+dr, dc:self.size+dc]
                    for dr, dc in posed_offsets]).all(axis=0)

                # positions fully open
                fully_open = np.stack([
                    open_cells[dr:self.size+dr, dc:self.size+dc]
                    for dr, dc in posed_offsets]).all(axis=0)

                # positions with no adjacent sides
                no_sides = np.stack([
                    ~sides[dr:self.size+dr, dc:self.size+dc]
                    for dr, dc in posed_offsets]).all(axis=0)

                # positions with any adjacent corners
                any_corners = np.stack([
                    corners[dr:self.size+dr, dc:self.size+dc]
                    for dr, dc in posed_offsets]).any(axis=0)
                
                # isolate valid positions
                valid_positions = on_board & fully_open & no_sides & any_corners
                
                # append valid actions
                actions.extend([
                    (polyomino, p, row, col)
                    for row, col in zip(*np.nonzero(valid_positions))])

        # special case when no moves left for current player
        if len(actions) == 0: actions = [None]

        # cache results
        self.valid_action_list = actions
        
        # return
        return actions

    def perform(self, action):
        # action=None skips current player
        new_state = self.copy()
        new_state.turn = (self.turn % len(self.plays)) + 1

        if action is not None:
            polyomino, p, row, col = action
            posed = polyomino.poses[p]
            new_state.hands[self.turn].remove(polyomino)
            new_state.plays[self.turn].append((row, col, posed))
            for dr, dc in polyomino.posed_offsets[p]:
                new_state.board[row+dr, col+dc] = self.turn

        return new_state

def initial_state(board_size=10, polyomino_size=None):
    if polyomino_size is None: polyomino_size = len(polyominoes)
    hands = {}
    for player in [1, 2]:
        hands[player] = []
        for size in range(polyomino_size):
            hands[player].extend(polyominoes[size])
    plays = {1: [], 2: []}
    return State(size=board_size, turn=1, hands=hands, plays=plays)

if __name__ == "__main__":
    
    state = initial_state()
    while not state.is_leaf():
        print(state)
        actions = state.valid_actions()
        state = state.perform(actions[0])

