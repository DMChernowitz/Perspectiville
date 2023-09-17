import numpy as np
from math import gcd

from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
from classes import Poly
from config import path_weights

def scan_wall(wall,j,k):
    h_ribs_front = sorted(wall, key=lambda x: (x[k], x[j]))
    z0 = None
    running_gcd = 0
    l = 0
    xmin = min([w[j] for w in wall])
    for w in h_ribs_front:
        if w[k] != z0:
            running_gcd = gcd(l,running_gcd,w[j]-xmin)
            l = 0
            z0 = w[k]
            x0 = w[j]
        elif w[j] != x0:
            running_gcd = gcd(l,running_gcd,w[j]-x0)
            l = 0
            x0 = w[j]
        if running_gcd == 1:
            return 1
        l += 1
        x0 += 1
    return gcd(l,running_gcd)

def divisors(N):
    return [j for j in range(1, N + 1) if (N % j) == 0]

def minimize_pts(coords):
    """removes redundant waypoints in a 2d polygon"""
    two_d = np.array(coords)
    diffs = two_d[1:]-two_d[:-1]
    new_ind = [h for h in range(len(diffs)) if diffs[h-1,0]*diffs[h,1]-diffs[h,0]*diffs[h-1,1]]
    return two_d[new_ind]

def get_fixed_coord(p: np.array) -> tuple:
    """

    :param p: polygon as np.aray([point1,point2,point3, etc]). point1 = [x_0,y_0,z_0]
    :return: tuple, first value [0,1,2] for [x,y,z], second the value of that coord
        that is constant over the polygon
    """
    for h,j in enumerate(sum(abs(p[:3] - p[0]))):
        if not j:
            break
    return h,p[0,h]

def join_polygons(polys: list) -> list:
    """
    Take a list of flat polygons that are in the same 3d plane, join them, if possible
    :param polys: list of np.array objects of dim (>2,3)
    :return: list of np.arrays of shorter or equal length.
    """
    comps = [(1, 2), (0, 2), (0, 1)]
    fixed_1 = get_fixed_coord(polys[0].coords)
    parsed_polys  = [Polygon(p.coords[:,comps[fixed_1[0]]]) for p in polys]

    unified = mapping(unary_union(parsed_polys))

    if unified["type"] == "Polygon":
        new_coords = [unified["coordinates"]]
    elif unified["type"] == "MultiPolygon":
        new_coords = unified["coordinates"]

    return [
        Poly(coords=np.insert(minimize_pts(u[0]), fixed_1[0], fixed_1[1], axis=1),
             orientation=polys[0].orientation,
             color_group=polys[0].color_group) for u in new_coords
    ]


def choose_next_weighted(pos_data):
    weights = [path_weights[p[1]] for p in pos_data]
    cum = np.cumsum(weights)
    needle = sum(weights)*np.random.random()
    for h in range(len(weights)):
        if cum[h] > needle:
            return pos_data[h]

def above(pos):
    return tuple(np.array([0, 0, 1]) + pos)


cube_blocking = {
    0: 9,
    1: 10,
    2: 11,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    11: 11
}


directions = { #encoding
    "s": [[0, 0, 0]],  # start
    "r": [[1, 0, 0]],  # moving right
    "l": [[-1, 0, 0]],  # moving left
    "b": [[0, 1, 0]],  # moving back
    "f": [[0, -1, 0]],  # moving front
    "ru": [[1, 0, 0],[1,0,1]],  # moving right and up
    "rd": [[1, 0, -1],[1,0,0]],  #moving right and down
    "lu": [[-1, 0, 0],[-1,0,1]],  # moving left and up
    "ld": [[-1, 0, -1],[-1,0,0]],  # moving left and down
    "bu": [[0, 1, 0],[0,1,1]],  # moving back and up
    "bd": [[0, 1, -1],[0,1,0]],  # moving back and down
    "fu": [[0, -1, 0],[0,1,1]],  # moving front and up
    "fd": [[0, -1, -1],[0,-1,0]]  # moving front and down
}

dir_to_fill = {
    "s": 1,  # start
    "r": 3,  # moving right
    "l": 3,  # moving left
    "b": 3,  # moving back
    "f": 3,  # moving front
    "ru": 5,  # moving right and up
    "rd": 6,  # moving right and down
    "lu": 6,  # moving left and up
    "ld": 5,  # moving left and down
    "bu": 7,  # moving back and up
    "bd": 8,  # moving back and down
    "fu": 8,  # moving front and up
    "fd": 7,  # moving front and down
}

transitions = {#can go from this direction to these
    "s": ["r", "l", "b", "f", "ru", "rd", "lu", "ld", "bu", "bd", "fu", "fd"],  # start
    "r": ["r", "b", "f", "ru", "rd", "bu", "bd", "fu", "fd"],  # moving right
    "l": ["l", "b", "f", "lu", "ld", "bu", "bd", "fu", "fd"],  # moving left
    "b": ["r", "l", "b", "ru", "rd", "lu", "ld", "bu", "bd"],  # moving back
    "f": ["r", "l", "f", "ru", "rd", "lu", "ld", "fu", "fd"],  # moving front
    "ru": ["r", "ru"],  # moving right and up
    "rd": ["r", "rd"],  # moving right and down
    "lu": ["l", "lu"],  # moving left and up
    "ld": ["l", "ld"],  # moving left and down
    "bu": ["b", "bu"],  # moving back and up
    "bd": ["b", "bd"],  # moving back and down
    "fu": ["f", "fu"],  # moving front and up
    "fd": ["f", "fd"],  # moving front and down
}

transition_directions = {key: {i: np.array(directions[i]) for i in obj} for (key,obj) in transitions.items()}


# def find_clusters(airspace,codes):
#     Lx,Ly,Lz = airspace.shape
#     cluster_hash = dict()
#     cluster_map = []
#     lookback = list(itertools.product(range(-1,1),range(-1,2),range(-1,2)))
#     lookback = [l for l in lookback if l not in [(0,0,0),(0,1,0),(0,1,1)]]
#     for ix,iy,iz in itertools.product(range(1,Lx-1), range(1,Ly-1), range(1,Lz)):
#         if airspace[(ix,iy,iz)] in codes:
#             unknown = True
#             for dx,dy,dz in lookback:
#                 if (ix+dx,iy+dy,iz+dz) in cluster_hash:
#                     unknown = False
#                     cn = cluster_hash[(ix+dx,iy+dy,iz+dz)]
#                     cluster_hash[(ix,iy,iz)] = cn
#                     cluster_map[cn].append((ix,iy,iz))
#                     break
#             if unknown:
#                 cluster_hash[(ix, iy, iz)] = len(cluster_map)
#                 cluster_map.append([(ix,iy,iz)])
#
#     return cluster_map, cluster_hash