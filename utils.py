import numpy as np
from math import gcd

from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
from classes import Poly
from config import path_weights, curvature_weights
from matplotlib.path import Path
from matplotlib.patches import PathPatch

from shapely.geometry.polygon import orient

# Plots a Polygon to pyplot `ax`
def patch_polygon(poly, **kwargs):

    opoly = orient(poly)
    path = Path.make_compound_path(
        Path(np.asarray(opoly.exterior.coords)[:, :2]),
        *[Path(np.asarray(ring.coords)[:, :2]) for ring in opoly.interiors])

    return PathPatch(path, **kwargs)



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

    parsed_polys = [Polygon(p.coords[:,comps[fixed_1[0]]]) for p in polys]

    unified = mapping(unary_union(parsed_polys))

    if unified["type"] == "Polygon":
        new_coords = [unified["coordinates"]]
    elif unified["type"] == "MultiPolygon":
        new_coords = unified["coordinates"]


    def restore_dim(u):
        return np.insert(minimize_pts(u), fixed_1[0], fixed_1[1], axis=1)

    merged_polys = []

    for merged_coords in new_coords:
        holes = [restore_dim(u) for u in merged_coords[1:]]
        merged_polys.append(
            Poly(coords=restore_dim(merged_coords[0]),
            orientation=polys[0].orientation,
            color_group=polys[0].color_group,
            holes=holes)
        )


    return merged_polys


def choose_next_weighted(s0,pos_data):
    weights = [weight_transition(s0,p[1])*path_weights[p[1]] for p in pos_data]
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
    11: 11,
    13: 13
}


directions = { #encoding
    "s": [[0, 0, 0]],  # start
    "r": [[1, 0, 0]],  # moving right
    "l": [[-1, 0, 0]],  # moving left
    "b": [[0, 1, 0]],  # moving back
    "f": [[0, -1, 0]],  # moving front
    "ru": [[1, 0, 0], [1,0,1]],  # moving right and up
    "rd": [[1, 0, -1], [1,0,0]],  #moving right and down
    "lu": [[-1, 0, 0], [-1,0,1]],  # moving left and up
    "ld": [[-1, 0, -1], [-1,0,0]],  # moving left and down
    "bu": [[0, 1, 0], [0,1,1]],  # moving back and up
    "bd": [[0, 1, -1], [0,1,0]],  # moving back and down
    "fu": [[0, -1, 0], [0,-1,1]],  # moving front and up
    "fd": [[0, -1, -1], [0,-1,0]]  # moving front and down
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

def weight_transition(s0,s1):
    if s0 == s1:
        return curvature_weights["inertia"]
    if (s01 := s0+s1).count("u") == 1 or s01.count("d") == 1:
        return curvature_weights["updown"]
    return curvature_weights["corner"]

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


def get_y_blocks(roof_hash: dict, roof_dir: int) -> tuple:
    """
    take a dict of (x,y,z) keys and height (int) values,
    and output two lists: roof position dicts, and
    coordinates of extra building blocks
    Also, don't look at me, I'm hideous
    :param roof_hash: single value of self.roof_hash from escherville
    :param roof_dir: 0 for roofs with a front, 1 for roofs with a side.
    :return:
    """
    target = [pos for pos in roof_hash if roof_hash[pos]<0]

    coords = np.array(list(roof_hash))
    front_left = coords.min(axis=0)
    back_right = coords.max(axis=0)

    if target:
        dy_enforce = True
        dy_limit = back_right[1]-max([t[1] for t in target])
    else:
        dy_enforce = False

    xmin = front_left[0]
    xmax = back_right[0]
    zmin = front_left[2]
    max_height = max(roof_hash.values())
    extra_blocks = []
    max_dz = 0
    roofing_areas = []
    new_flats = []
    for dz in range(max_height):
        extra_roofs = []
        z = zmin + dz
        for dy in range(back_right[1]-front_left[1]+1):
            y = back_right[1]-dy
            fly = True
            if dy_enforce and dy>=dy_limit:
                fly = False #dont build beyond escher targets and sources
            for dx in range(1+(xmax-xmin)//2):
                xl = xmin+dx
                xr = xmax-dx
                pl = (xl,y,z)
                pr = (xr,y,z)
                #for left point, right point:
                support = (dz==0 and pl in roof_hash) or (xl,y,z-1) in extra_blocks,  (dz==0 and pr in roof_hash) or (xr,y,z-1) in extra_blocks
                backing = (dy==0 or (xl,y+1,z) in extra_blocks), (dy==0 or (xr,y+1,z) in extra_blocks)
                hl = roof_hash.get((xl,y,zmin),0)
                hr = roof_hash.get((xr,y,zmin),0)
                skyspace = hl > dz, hr > dz
                if fly and all(support) and all(backing) and all(skyspace) and (np.random.randint(3+dy)>0) and (dz+1<max_height):
                    extra_blocks.extend([pl,pr])
                    max_dz = max(max_dz,dz)
                elif fly and all(support) and all(skyspace):
                    extra_roofs.extend([(xl,y,hl-dz),(xr,y,hr-dz)])
                elif dz: # new flat roofs above old roof level
                    if support[0] and skyspace[0]:
                        new_flats.append(pl)
                    if support[1] and skyspace[1]:
                        new_flats.append(pr)
        for roof_patch in get_roof_patches(extra_roofs,j=roof_dir):
            displace = np.array(roof_patch["botleft"]+[z])
            droof = np.array([roof_patch["droof"]])
            roofing_areas.append({
                "displace": displace,
                "droof": droof,
            })
        if dz > max_dz:  # gap
            break
    return roofing_areas,extra_blocks,new_flats


def get_roof_patches(squares: list, j: int):
    if not squares:
        return []

    #square is (x,y,height)
    if j==2: #castle walls, remove using trim_poly_logic.
        return [{"botleft": list(s[:2]), "droof": [1,1,1]} for s in squares]

    #j is 0 or 1
    squares = sorted(squares,key = lambda x: (x[1-j],x[j]))
    left = squares[0]
    right = squares[0]
    curheight = left[2]
    rods = {}
    for h,square in enumerate(squares[1:]):
        if square[j] != squares[h][j]+1 or square[1-j] != squares[h][1-j]:
            rods.setdefault((left[j],right[j]),[]).append((left[1-j],curheight))
            left = square
            curheight = left[2]
        right = square
        curheight = min(curheight,right[2])
    rods.setdefault((left[j], right[j]), []).append((left[1-j],curheight))
    roof_patches = []
    ord = 1-2*j

    def bottle_patch(_key, _l, njm, _w, _ch):
        return {
            "botleft": [_key[0], njm][::ord],
            "droof": [_w, _l][::ord] + [_ch]
        }

    for key, val in rods.items():
        w = key[1]-key[0]+1
        l = 1
        not_j_min = val[0][0]
        curheight = val[0][1]
        for h,v in enumerate(val[1:]):
            if val[h][0] != v[0]-1:
                roof_patches.append(bottle_patch(key, l, not_j_min, w, curheight))
                not_j_min = v[0]
                curheight = v[1]
                l = 0
            l += 1
            curheight = min(curheight,v[1])
        roof_patches.append(bottle_patch(key, l, not_j_min, w, curheight))
    return roof_patches

def rel_choice(D: dict):
    tot = np.cumsum(list(D.values()))
    needle = np.random.randint(tot[-1])
    for k,t in zip(D,tot):
        if needle < t:
            return k




import matplotlib as mpl

labels = ["path","window","roof","bg","build_1","build_2","build_3"]



def show_color_scheme(color_list):
    nrows = len(color_list)
    figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
    fig, axs = plt.subplots(nrows=nrows + 1, figsize=(6.4, figh))
    fig.subplots_adjust(top=1 - 0.35 / figh, bottom=0.15 / figh,
                        left=0.2, right=0.99)
    h=0
    for col, ax in zip(color_list,axs):
        for j,c in enumerate(col):
            ax.fill([j,j,j+1,j+1],[0,1,1,0],color=c)

        ax.text(-0.01, 0.5, str(h), va='center', ha='right', fontsize=10,
                transform=ax.transAxes)
        h+=1

    for j,lab in enumerate(labels):
        axs[-1].text(j/7.5+0.11, -0.5, lab, va='top', ha='center', fontsize=10,
                transform=ax.transAxes)

    for ax in axs:
        ax.set_axis_off()










