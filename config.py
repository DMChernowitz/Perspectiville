import numpy as np

proj_array = np.array([[1., 0.], [0.5, 0.5], [0., 1.]])
camera_ratio = np.array([1, -2, 1])

path_weights = {
    "r": 5,  # moving right
    "l": 5,  # moving left
    "b": 5,  # moving back
    "f": 5,  # moving front
    "ru": 2,  # moving right and up
    "rd": 5,  # moving right and down
    "lu": 5,  # moving left and up
    "ld": 2,  # moving left and down
    "bu": 5,  # moving back and up
    "bd": 2,  # moving back and down
    "fu": 2,  # moving front and up
    "fd": 5,  # moving front and down
}

unit_cell_paths = [[[(0,1)],[]],[[],[(1,0)]],[[],[]]]#[[[(1,0)],[(1,-1)]],[[(1,-1)],[]],[[],[(0,1)]]]