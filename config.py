import numpy as np
import matplotlib as mpl
import colorsys

from palettable.cartocolors.diverging import Earth_7, Fall_7_r, TealRose_7_r
from palettable.colorbrewer.diverging import BrBG_7, RdBu_7, RdBu_7_r, RdGy_7, RdYlBu_7, Spectral_7, RdGy_6
from palettable.lightbartlein.diverging import BlueDarkRed18_7_r,BlueGray_7, BlueOrangeRed_7_r, BlueDarkOrange18_7_r
from palettable.cmocean.diverging import Curl_7_r, Delta_7
from palettable.cmocean.sequential import Oxy_7
from palettable.cartocolors.qualitative import Antique_6, Bold_6


palettable_cmaps = [Earth_7, BrBG_7, RdBu_7, RdBu_7_r, RdGy_7, RdYlBu_7, BlueDarkRed18_7_r,BlueGray_7, BlueOrangeRed_7_r, BlueDarkOrange18_7_r, Curl_7_r, Delta_7, Oxy_7, Spectral_7, Fall_7_r, TealRose_7_r]



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
    elif color_retention < 1:
        dsat = color_retention+light_retention*(1-color_retention)
        dhue = (1-color_retention)*(1-light_retention)*move
    else:
        move = 0
        dhue = 0
        dsat = 1
    return colorsys.hls_to_rgb(h+dhue,(1-abs(move))*light_retention*l,dsat*s+1-dsat)

def get_poly_colors(windows_bright=True, default=False):
    modulator = {
        "front": 0.7,
        "side": 0.5,
        "top": 0.9
    }
    modulator = modulator | {f"slant_{n}": 0.85-0.07*n for n in range(5)} | {f"tnals_{n}": 0.65-0.07*n for n in range(5)}
    modulator = modulator | {f"slope_{n}": 0.67-0.07*n for n in range(5)}

    default_scheme = [
        (0, 0, 1),  # path is blue
        (1, 1, 0),  # windows are yellow
        (1, 0, 0),  # roof is red
        (1, 1, 1),  # background is white
        (0.9, 0.9, 0.9),  # buildings are rgay
        (0.75, 0.75, 0.75),  # buildings are rgay
        (0.4, 0.4, 0.4),  # buildings are rgay
    ]

    # utils for parsing colormaps
    redthird = [1, 2, 0, 3, 4, 5, 6]
    def tuplify(cmap):
        return [tuple(u) for u in np.array(cmap.colors)[redthird, :] / 256]
    def _sc(cc):
        return [tuple(np.array(ee) / 256) for ee in cc]

    bo = mpl.colormaps["bone"]
    pa = mpl.colormaps["Pastel1"]

    schemes = [[bo(0.8), (1, 1, 0), bo(0.6), (1, 1, 1), bo(0.2), bo(0.3), bo(0.4)],
               [pa(0.7), pa(3 / 16), pa(1 / 16), pa(10 / 16), pa(5 / 16), pa(7 / 16), pa(8 / 16)],
               _sc(Earth_7.colors),
               _sc(Antique_6.colors[:3]) + [(1, 1, 1)] + _sc(Antique_6.colors[3:]),
               _sc(Bold_6.colors[2:3]) + _sc(Bold_6.colors[3:5]) + [(1, 1, 1)] + _sc(Bold_6.colors[:2]) + _sc(Bold_6.colors[5:]),
               _sc(RdGy_6.colors[1:2]) + [(1, 0.9, 0)] + _sc(RdGy_6.colors[:1]) + [(1, 1, 1)] + _sc(RdGy_6.colors[3:6:2]) + _sc(BrBG_7.colors[6:])
               ]
    schemes.extend([tuplify(cm) for cm in palettable_cmaps])

    hue_target = np.random.random()

    chosen_scheme = default_scheme if default else schemes[np.random.randint(len(schemes))]
    color_retention = 1 if default else 0.9

    kron_dict = {
        (c,m): C if (c == 1 and windows_bright)
        else
        darken(
            ori=C[:3],
            light_retention=M,
            hue_target=hue_target,
            color_retention=color_retention
        )
        for
        c,C in enumerate(chosen_scheme)
        for
        m,M in modulator.items()
    }

    return kron_dict


#relative weights of the directions
path_weights = {k: 5 for k in ["r", "l", "b", "f", "rd", "lu", "bu", "fd"]} | {k: 2 for k in ["ru", "ld", "bd", "fu"]}

curvature_weights = {
    "inertia": 3,  # relative prob to keep going straight
    "updown": 2,  # relative prob to start moving up or down
    "corner": 1  # relative prob to turn left or right
}

unit_cell_paths = [
    [[(0, 1)], []],
    [[(-1, 1)], []],
    [[(0,0), (0, 1)], []]
]
#  [[[(0,1)],[]],[[],[(1,0)]],[[],[]]]#

roof_style_probs = {"slant_front": 2, "slope_side": 2, "castle": 1}