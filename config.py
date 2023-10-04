import numpy as np
import matplotlib as mpl
import colorsys

proj_array = np.array([[1., 0.], [0.5, 0.5], [0., 1.]])
camera_ratio = np.array([1, -2, 1])

n_colors = 3

ns = 5 #n steps per flight of stairs

def darken(ori: tuple, light_retention: float, hue_target: float, color_retention: float=None) -> tuple:
    #light_retention : 1: no change
    #light_retention : 0: black
    h,l,s = colorsys.rgb_to_hls(*ori)
    hue_arc = hue_target-h
    move = min([hue_arc,1-hue_arc],key=abs)
    if color_retention is None:
        dsat = 0.2*(light_retention+4)
        dhue = 0.2*(1-light_retention)*move
    else:
        dsat = color_retention+light_retention*(1-color_retention)
        dhue = (1-color_retention)*(1-light_retention)*move
    return colorsys.hls_to_rgb(h+dhue,(1-abs(move))*light_retention*l,dsat*s+1-dsat)

def get_poly_colors(windows_bright=True):
    modulator = {
        "front": 0.7,
        "side": 0.5,
        "top": 0.9
    }
    modulator = modulator | {f"slant_{n}": 0.85-0.07*n for n in range(5)} | {f"tnals_{n}": 0.65-0.07*n for n in range(5)}
    modulator = modulator | {f"slope_{n}": 0.67-0.07*n for n in range(5)}

    color_group_bases = {
        0: np.array([0, 0, 1]),  # path is blue
        1: np.array([1, 1, 0]),  # windows are yellow
        2: np.array([1, 0, 0]),  # roof is red
        #3: np.array([0, 1, 0]),  # target is green
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

    cmapname = np.random.choice(["bone"]) #"pink","coolwarm",

    cmap = lambda cc: np.array(mpl.colormaps[cmapname](cc)[:3])
    color_group_bases2 = {
        0: cmap(0.85),
        1: cmap(0.95),
        2: cmap(0.5),
        4: cmap(0.2),
        5: cmap(0.4),
        6: cmap(0.6)
    }

    # keys = list(color_group_bases)
    # color_group_bases = {}
    # colors = []

    # for key in keys:
    #     while ((v := cmap(np.random.random())) in colors):
    #         pass
    #     colors.append(v)
    #     color_group_bases[key] = np.array(v[:3])

    hue_target = np.random.random()

    kron_dict = {
        (c,m): C if (c==1 and windows_bright) else darken(ori=C,light_retention=M,hue_target=hue_target, color_retention=0.9)
        for c,C in color_group_bases.items() for m,M in modulator.items()
    }

    return kron_dict


#relative weights of the directions
path_weights = {k: 5 for k in ["r", "l", "b", "f", "rd", "lu", "bu", "fd"]} | {k: 2 for k in ["ru", "ld", "bd", "fu"]}

curvature_weights = {
    "inertia": 3,  # relative prob to keep going straight
    "updown": 2,  # relative prob to start moving up or down
    "corner": 1  # relative prob to turn left or right
}

unit_cell_paths = [[[(0,1)],[]],[[],[(1,0)]],[[],[]]]#[[[(1,0)],[(1,-1)]],[[(1,-1)],[(0,0)]],[[],[(0,1)]]]

roof_style_probs = {"slant_front": 2, "slope_side": 2, "castle": 1}