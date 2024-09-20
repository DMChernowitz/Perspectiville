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

hatch_bool = False #whether to color with hatches, (True) or colors (False)

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
    invret = min(light_retention,1/light_retention)  # always <=1

    if color_retention is None:
        dsat = 0.2*(invret+4)  # dsat is a mixing coef, not a delta
        dhue = 0.2*(1-invret)*move
    elif color_retention < 1:
        dsat = color_retention+invret*(1-color_retention)
        dhue = (1-color_retention)*(1-invret)*move
    else:
        move = 0
        dhue = 0
        dsat = 1
    if light_retention < 1:
        new_light = light_retention*l
    else:
        dhue -= 1
        dark = (1-l)*invret
        new_light = 1-dark
    return colorsys.hls_to_rgb(h+dhue,(1-abs(move))*new_light,dsat*s+1-dsat)

def sb(t):
    return tuple(np.array(t)/256)

bcg_cols = {
    (0, "side"): sb((41, 94, 126)),  # true blue
    (0, "top"): sb((48, 193, 215)),  # bright blue
    (0, "front"): sb((45, 144, 170)),  # interpolate

    (1, "top"): sb((212, 223, 51)),  # yellow
    (1, "side"): sb((212, 223, 51)),  # yellow
    (1, "front"): sb((212, 223, 51)),  # yellow

    (2, "front"): sb((103, 15, 49)),  # cranberry
    (2, "top"): sb((231, 28, 87)),  # magenta

    (3, "top"): sb((256, 256, 256)),  # white

    (4, "top"): sb((62, 173, 146)),  # green light
    (4, "front"): sb((25, 122, 86)),  # jade
    (4, "side"): sb((3, 82, 45)),  # dark green

    (5, "top"): sb((200, 200, 200)),  # light gray
    (5, "front"): sb((154, 154, 154)),  # gray
    (5, "side"): sb((110, 111, 115)),  # dark gray

    (6, "top"): sb((106, 214, 90)),  # light green
    (6, "front"): sb((41, 186, 116)),  # bcg green
    (6, "side"): sb((35, 89, 80)),  # dark green
}
for n in range(6):
    bcg_cols[(1, f"slant_{n}")] = bcg_cols[(1, "top")]
    bcg_cols[(1, f"slope_{n}")] = bcg_cols[(1, "top")]
    bcg_cols[(1, f"tnals_{n}")] = bcg_cols[(1, "top")]

    a1 = 0.2*n
    a2 = 0.2*(5-n)

    bcg_cols[(2, f"slant_{n}")] = tuple(a2*np.array(bcg_cols[(2, "front")])+a1*np.array(bcg_cols[(2, "top")]))
    bcg_cols[(2, f"slope_{n}")] = bcg_cols[(2, f"slant_{n}")]
    bcg_cols[(2, f"tnals_{n}")] = darken(bcg_cols[(2, f"slant_{n}")], hue_target=1, light_retention=0.8, color_retention=1)

def get_poly_colors(windows_bright=True, default=False):
    modulator = {
        "front": 1,
        "side": 0.5,
        "top": 1.5
    }
    modulator = modulator | {f"slant_{n}": 1.4-0.2*n for n in range(5)} | {f"tnals_{n}": 1-0.2*n for n in range(5)}
    modulator = modulator | {f"slope_{n}": 1.3-0.2*n for n in range(5)}

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


    hue_target = np.random.random()

    if isinstance(default,bool) and default:
        chosen_scheme = default_scheme
    elif isinstance(default, dict) or isinstance(default, list):

        if isinstance(default,dict):
            default = default.values()

        if isinstance(default[0],str):

            def h2r(h):
                return tuple(int(h[i:i+2], 16)/255 for i in (0, 2, 4))

            default = [h2r(def_col) for def_col in default]

        chosen_scheme = default
    else:
        schemes = get_schemes()
        chosen_scheme = schemes[np.random.randint(len(schemes))]
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


def get_schemes():
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
               _sc(Bold_6.colors[2:3]) + _sc(Bold_6.colors[3:5]) + [(1, 1, 1)] + _sc(Bold_6.colors[:2]) + _sc(
                   Bold_6.colors[5:]),
               _sc(RdGy_6.colors[1:2]) + [(1, 0.9, 0)] + _sc(RdGy_6.colors[:1]) + [(1, 1, 1)] + _sc(
                   RdGy_6.colors[3:6:2]) + _sc(BrBG_7.colors[6:])
               ]
    schemes.extend([tuplify(cm) for cm in palettable_cmaps])
    return schemes


#relative weights of the directions
path_weights = {k: 5 for k in ["r", "l", "b", "f", "rd", "lu", "bu", "fd"]} | {k: 2 for k in ["ru", "ld", "bd", "fu"]}

curvature_weights = {
    "inertia": 3,  # relative prob to keep going straight
    "updown": 2,  # relative prob to start moving up or down
    "corner": 1  # relative prob to turn left or right
}

# unit_cell_paths = [[[]]]

# unit_cell_paths = [
#     [[],[],[]],
#     [[(-1,1),(-1,0)],[],[(-1,-1)]]
# ]


unit_cell_paths = [
    [[ (1,0)], []],
    [[(-1, 1)], []],
    [[], [(-1, 0)]]
]
#  [[[(0,1)],[]],[[],[(1,0)]],[[],[]]]#

roof_style_probs = {"slant_front": 2, "slope_side": 2, "castle": 1}