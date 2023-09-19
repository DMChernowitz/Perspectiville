import numpy as np

proj_array = np.array([[1., 0.], [0.5, 0.5], [0., 1.]])
camera_ratio = np.array([1, -2, 1])

n_colors = 3

def get_poly_colors():
    modulator = {
        "front": 0.7,
        "side": 0.5,
        "top": 0.9
    }
    modulator = modulator | {f"slant_{n}": 0.85-0.05*n for n in range(6)}
    modulator = modulator | {f"tnals_{n}": 0.65-0.05*n for n in range(6)}

    color_group_bases = {
        0: np.array([0, 0, 1]),  # path is blue
        1: np.array([1, 1, 0]),  # windows are yellow
        2: np.array([1, 0, 0]),  # roof is red
        3: np.array([0, 1, 0]),  # target is green
        4: np.array([0.9, 0.9, 0.9]),  # buildings are rgay
        5: np.array([0.75, 0.75, 0.75]),  # buildings are rgay
        6: np.array([0.4, 0.4, 0.4]),  # buildings are rgay
        # "top": (0.8,0,0),
        # "front": (0.65,0,0),
        # "side": (0.4,0,0),
        # "mtop2": (0.8,0,0),
        # "mtop": (0.8,0.8,0.8),
        # "mfront": (0.65,0.65,0.65),
        # "mside": (0.4,0.4,0.4),
        # "away": 0.3,
        # "slant": 0.7,
        # "toward": 0.6,
        # "skew": 0.3,
        # "target": (0.7,0.6,0)
    }

    kron_dict = {(c,m): tuple(C*M) for c,C in color_group_bases.items() for m,M in modulator.items()}

    return kron_dict


#relative weights of the directions
path_weights = {k: 5 for k in ["r", "l", "b", "f", "rd", "lu", "bu", "fd"]} | {k: 2 for k in ["ru", "ld", "bd", "fu"]}

curvature_weights = {
    "inertia": 3,  # relative prob to keep going straight
    "updown": 2,  # relative prob to start moving up or down
    "corner": 1  # relative prob to turn left or right
}

unit_cell_paths = [[[(0,1)],[]],[[],[(1,0)]],[[],[]]]#[[[(1,0)],[(1,-1)]],[[(1,-1)],[]],[[],[(0,1)]]]

roof_style_probs = {"flat": 1, "slant_front": 1}