
import numpy as np
import itertools as it

class Polyomino(object):
    def __init__(self, array):
        self.array = array
        self.poses = [] # list of posed arrays
        self.posed_offsets = []
        pose_set = set()
        for num_rotations, flip in it.product(range(4), [False, True]):
            posed = np.rot90(array, k=num_rotations)
            if flip: posed = np.fliplr(posed)
            if str(posed) not in pose_set:
                pose_set.add(str(posed))
                self.poses.append(posed)
                self.posed_offsets.append(list(zip(*np.nonzero(posed))))

polyominoes = [
    [
        Polyomino(np.array([[1]]))
    ],[
        Polyomino(np.array([[1, 1]]))
    ],[
        Polyomino(np.array([[1, 1, 1]])),
        Polyomino(np.array([[1, 1],
                           [1, 0]]))
    ],[
        Polyomino(np.array([[1, 1, 1, 1]])),
        Polyomino(np.array([[1, 1, 1],
                           [1, 0, 0]])),
        Polyomino(np.array([[1, 1, 1],
                           [0, 1, 0]])),
        Polyomino(np.array([[1, 1, 0],
                           [0, 1, 1]])),
        Polyomino(np.array([[1, 1],
                           [1, 1]])),
    ],[
        Polyomino(np.array([[1, 1, 1, 1, 1]])),
        Polyomino(np.array([[1, 1, 1, 1],
                           [1, 0, 0, 0]])),
        Polyomino(np.array([[1, 1, 1, 1],
                           [0, 1, 0, 0]])),
        Polyomino(np.array([[1, 1, 0, 0],
                           [0, 1, 1, 1]])),
        Polyomino(np.array([[1, 1, 1],
                           [0, 1, 1]])),
        Polyomino(np.array([[1, 1, 1],
                           [1, 0, 1]])),
        Polyomino(np.array([[1, 1, 1],
                           [1, 0, 0],
                           [1, 0, 0]])),
        Polyomino(np.array([[1, 1, 1],
                           [0, 1, 0],
                           [0, 1, 0]])),
        Polyomino(np.array([[0, 1, 1],
                           [0, 1, 0],
                           [1, 1, 0]])),
        Polyomino(np.array([[0, 1, 1],
                           [1, 1, 0],
                           [1, 0, 0]])),
        Polyomino(np.array([[0, 1, 0],
                           [1, 1, 1],
                           [1, 0, 0]])),
        Polyomino(np.array([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]])),
    ],
]

if __name__ == "__main__":
    
    print(polyominoes[2][0].array)
    print(polyominoes[2][0].poses)

