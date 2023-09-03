# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import collections as mc
from shapely.geometry import Polygon, mapping, LineString
from shapely.ops import unary_union
import itertools
import copy
import functools
import networkx as nx


from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection


e_tol = 10e-5

#practice polygons
p_0 = np.array([[2,2,2,2],[1,2,2,1],[3,3,4,4]]).T
p_1 = np.array([[1,2,2,1],[2,2,2,2],[3,3,4,4]]).T
p_2 = np.array([[1,2,2,1],[3,3,4,4],[2,2,2,2]]).T
p_3 = np.array([[1.5,1.8,1.8,1.5],[2,2,2,2],[3.5,3.5,4.5,4.5]]).T


my_graph = {1: [2], 2: [1, 5], 3: [4], 4: [3, 5], 5: [6], 6: [7], 7: [8], 8: [6, 9], 9: []}


fat_graph = {0:[1,2,3,4],1:[5],2:[6],3:[7],4:[8],5:[10],6:[10],7:[10],8:[10],9:[5],10:[9,11],11:[0,10]}
pat_graph = {0:[1,2],1:[2],2:[3,5],3:[1],4:[2],5:[0,4]}




bah = nx.DiGraph()
bah.add_edges_from([(v,k) for v,kk in fat_graph.items() for k in kk])
print(list(nx.simple_cycles(bah))) #just find most used edge in simple cycle, remove it. Remove all cycles that use that edge, then repeat.






class Poly:
    """
    will have coordinates, sorting function, can hold which edges to draw, and comparison.
    """

    def __init__(
            self, coords: np.array,
            color_group: int = None,
            orientation: str = None,
            merges: bool = False,
            camera_ratio: np.array = np.array([1, -2, 1])
    ):
        self.coords = coords
        self.color_group = color_group  # 0: path, 1: window, 2: roof, 3+: buildings
        self.orientation = orientation  # top, side, front, slant_{i}
        self.merges = merges
        self._edges = None
        self.normal_vec = None
        self.d_const = None
        self._camera_ratio = camera_ratio
        self._close_far = None

    @property
    def edges(self):
        if self._edges == None:
            self._edges = [self.coords[[j, j + 1]] for j in range(-1, len(self.coords) - 1)]
        return self._edges

    @edges.setter
    def edges(self, new_edges):
        self._edges = new_edges

    def __eq__(self, other_poly):
        return self.custom_zorder(other_poly) == 0

    def __lt__(self, other_poly):
        return self.custom_zorder(other_poly) < 0

    def custom_zorder(self, other_poly):

        #first: simply distance, if indecisive, then old_custom_order, if indecisive, use planes

        if self.normal_vec is None:
            self.get_plane()
            self._recalculate_perspective()

        margin = 6  # significant digits, if points are within 1e-margin of the plane, consider them on the plane.

        sides = np.sign(np.round(other_poly.coords.dot(self.normal_vec)+self.d_const,margin))

        if (-1 in sides and 1 in sides) or (sumsides := sum(sides)) == 0:
            if self.close_far == other_poly.close_far:
                wrong_answer = 0
            else:
                wrong_answer = 2*int(self.close_far > other_poly.close_far)-1
        else:
            wrong_answer = self.forward_backward*sumsides

        real_order = old_custom_zorder(self.coords,other_poly.coords)

        if np.sign(real_order) != np.sign(wrong_answer):
            self.why = 9

        return real_order


    def get_plane(self):
        self.normal_vec = np.cross(*(self.coords[1:3] - self.coords[0]))
        self.d_const = -self.normal_vec.dot(self.coords[0])


    @property
    def camera_ratio(self):
        return self._camera_ratio

    @camera_ratio.setter
    def camera_ratio(self, new_ratio):
        self._camera_ratio = new_ratio
        self._recalculate_perspective()

    def _recalculate_perspective(self):
        self.forward_backward = np.sign(self.normal_vec.dot(self._camera_ratio))
        self._close_far = None

    @property
    def close_far(self):
        if self._close_far is None:
            pts = self.coords.dot(self._camera_ratio)
            self._close_far = min(pts), max(pts)
        return self._close_far


    @property
    def merge_tuple(self):
        return self.color_group,self.orientation

# def get_polygon_normal(pg: np.array) -> np.array:
#     #can parallelize
#     scaled_normal = np.cross(pg[:,1]-pg[:,0],pg[:,2]-pg[:,1])
#     return scaled_normal/np.linalg.norm(scaled_normal,2)

class PolyGraph:
    def __init__(self, polygons: list):
        self.polygons = polygons


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
    # if (fixed_1 := get_fixed_coord(p1)) != get_fixed_coord(p2):
    #     return p1,p2
    #this should be done for each poly, sorted, and hashed.
    fixed_1 = get_fixed_coord(polys[0].coords)
    parsed_polys  = [Polygon(p.coords[:,comps[fixed_1[0]]]) for p in polys]

    unified = unary_union(parsed_polys)

    polys_with_edges = []

    for parsed_poly,poly in zip(parsed_polys,polys):
        keep_lines = mapping(unified.boundary.intersection(LineString(parsed_poly.boundary)))
        if keep_lines["type"] == "GeometryCollection":
            keep_lines_iter = [u["coordinates"] for u in keep_lines["geometries"] if u["type"] == "LineString"]
        elif keep_lines["type"] == "MultiLineString":
            keep_lines_iter = keep_lines["coordinates"]
        elif keep_lines["type"] == "LineString":
            keep_lines_iter = [keep_lines["coordinates"]]
        bdys= [np.insert(u, fixed_1[0], fixed_1[1], axis=1) for u in keep_lines_iter]
        poly.edges = bdys #class has an attribute for this
        polys_with_edges.append(poly)

    return polys_with_edges

camera_ratio = np.array([1,-2,1])
ns = 5 #n steps

cube_filling_building = {
    0: [], # empty space
    1: [  # building roof
        Poly(coords=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]]),
             orientation="top",
             merges=True,
             color_group=4
             ),
        Poly(coords=np.array([[0,0,-0.2],[1,0,-0.2],[1,0,0],[0,0,0]]),
             orientation="front",
             merges=True,
             color_group=4),
        Poly(coords=np.array([[1,0,-0.2],[1,1,-0.2],[1,1,0],[1,0,0]]),
             orientation="side",
             merges=True,
             color_group=4)
    ],
    2: [  # building center
        Poly(coords=np.array([[0,0,0],[1,0,0],[1,0,1],[0,0,1]]),
             orientation="front",
             merges=True,
             color_group=4),
        Poly(coords=np.array([[1,0,0],[1,1,0],[1,1,1],[1,0,1]]),
             orientation="side",
             merges=True,
             color_group=4)
    ],
    3: [  # flat path
        Poly(coords=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]]),
             orientation="top",
             merges=True,
             color_group=0),
        Poly(coords=np.array([[0,0,0.01],[0.1,0,0.01],[0.1,0,0.2],[0.9,0,0.2],[0.9,0,0.01],[1,0,0.01],[1,0,0.3],[0,0,0.3]]),
             orientation="front",
             merges=False,
             color_group=0),
        Poly(coords=np.array([[0,1,0.01],[0.1,1,0.01],[0.1,1,0.2],[0.9,1,0.2],[0.9,1,0.01],[1,1,0.01],[1,1,0.3],[0,1,0.3]]),
             orientation="front",
             merges=False,
             color_group=0),
        Poly(coords=np.array([[0,0,0.01],[0,0.1,0.01],[0,0.1,0.2],[0,0.9,0.2],[0,0.9,0.01],[0,1,0.01],[0,1,0.3],[0,0,0.3]]),
             orientation="side",
             merges=False,
             color_group=0),
        Poly(coords=np.array([[1,0,0.01],[1,0.1,0.01],[1,0.1,0.2],[1,0.9,0.2],[1,0.9,0.01],[1,1,0.01],[1,1,0.3],[1,0,0.3]]),
             orientation="side",
             merges=False,
             color_group=0)
    ],
    4:  [  # building roof, target
        Poly(coords=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]]),
             orientation="top",
             merges=False,
             color_group=3),
        Poly(coords=np.array([[0, 0, -0.2], [1, 0, -0.2], [1, 0, 0], [0, 0, 0]]),
             orientation="front",
             merges=True,
             color_group=4),
        Poly(coords=np.array([[1, 0, -0.2], [1, 1, -0.2], [1, 1, 0], [1, 0, 0]]),
             orientation="side",
             merges=True,
             color_group=4)
    ],
    5: [  # up to right / down to left
        Poly(coords=np.array([*[[(j//2)/ns,0,((j+1)//2)/ns] for j in range(2*ns)],[1,0,1],[1,0,0]]),
             orientation="front",
             merges=True,
             color_group=4),
        *[
            Poly(coords=np.array([[j/ns,0,(j+1)/ns],[(j+1)/ns,0,(j+1)/ns],[(j+1)/ns,1,(j+1)/ns],[j/ns,1,(j+1)/ns]]),
                 orientation="top",
                 merges=False,
                 color_group=0)
            for j in range(ns)
        ]
    ],
    6: [  # down to right / up to left
        Poly(coords=np.array([*[[((j+1)//2)/ns,0,1-(j//2)/ns] for j in range(2*ns)],[1,0,0],[0,0,0]]),
             orientation="front",
             merges=True,
             color_group=4),
        *[
            Poly(coords=np.array([[j/ns,0,1-j/ns],[(j+1)/ns,0,1-j/ns],[(j+1)/ns,1,1-j/ns],[j/ns,1,1-j/ns]]),
                 orientation="top",
                 merges=False,
                 color_group=0)
            for j in range(ns)
        ],
        *[
            Poly(coords=np.array([[(j+1)/ns,0,1-j/ns],[(j+1)/ns,0,1-(j+1)/ns],[(j+1)/ns,1,1-(j+1)/ns],[(j+1)/ns,1,1-j/ns]]),
                 orientation="side",
                 merges=False,
                 color_group=0)
            for j in range(ns)
        ]
    ],
    7: [  # up to back / down to front
        Poly(coords=np.array([*[[1,(j//2)/ns,((j+1)//2)/ns] for j in range(2*ns)],[1,1,1],[1,1,0]]),
             orientation="side",
             merges=True,
             color_group=4),
        *[
            Poly(coords=np.array([[0,j/ns,j/ns],[1,j/ns,j/ns],[1,j/ns,(j+1)/ns],[0,j/ns,(j+1)/ns]]),
                 orientation="front",
                 merges=False,
                 color_group=0)
            for j in range(ns)
        ],
        *[
            Poly(coords=np.array([[0,j/ns,(j+1)/ns],[1,j/ns,(j+1)/ns],[1,(j+1)/ns,(j+1)/ns],[0,(j+1)/ns,(j+1)/ns]]),
                 orientation="top",
                 merges=False,
                 color_group=0)
            for j in range(ns)
        ]
    ],
    8: [  # down to back / up to front
        Poly(coords=np.array([*[[1,((j+1)//2)/ns,1-(j//2)/ns] for j in range(2*ns)],[1,1,0],[1,0,0]]),
             orientation="side",
             merges=True,
             color_group=4),
        *[
            Poly(coords=np.array([[0,j/ns,1-j/ns],[1,j/ns,1-j/ns],[1,(j+1)/ns,1-j/ns],[0,(j+1)/ns,1-j/ns]]),
                 orientation="top",
                 merges=False,
                 color_group=0)
            for j in range(ns)
        ]
    ],
    9: [],  # boundary, can't be built on.
}

cube_filling_building[10] = cube_filling_building[1]  # building roof: cannot be replaced
cube_filling_building[11] = cube_filling_building[2]  # building center, cannot be replaced

cube_filling_free = copy.deepcopy(cube_filling_building)
cube_filling_free[5][0] = Poly(coords=np.array(list(cube_filling_building[5][0].coords[:-1])+[[1,0,1-1/ns],[1/ns,0,0]]),
                              orientation="front",
                              merges=False,
                              color_group=0)
cube_filling_free[6][0] = Poly(coords=np.array(list(cube_filling_building[6][0].coords[:-1])+[[1-1/ns,0,0],[0,0,1-1/ns]]),
                               orientation="front",
                               merges=False,
                               color_group=0)
cube_filling_free[7][0] = Poly(coords=np.array(list(cube_filling_building[7][0].coords[:-1])+[[1,1,1-1/ns],[1,1/ns,0]]),
                               orientation="side",
                               merges=False,
                               color_group=0)
cube_filling_free[8][0] = Poly(coords=np.array(list(cube_filling_building[8][0].coords[:-1])+[[1,1-1/ns,0],[1,0,1-1/ns]]),
                               orientation="side",
                               merges=False,
                               color_group=0)

cube_filling_building[5].append(
    Poly(coords=np.array([[1,0,0],[1,1,0],[1,1,1],[1,0,1]]),
         orientation="side",
         merges=True,
         color_group=4)
)
cube_filling_building[8].append(
    Poly(coords=np.array([[0,0,0],[1,0,0],[1,0,1],[0,0,1]]),
         orientation="front",
         merges=True,
         color_group=4)
)

def get_poly_colors():
    modulator = {
        "front": 0.7,
        "side": 0.5,
        "top": 0.9
    }
    color_group_bases = {
        0: np.array([0, 0, 1]),  # path is blue
        1: np.array([1, 0.8, 0]),  # windows are yellow
        2: np.array([1, 0, 0]),  # roof is red
        3: np.array([0, 1, 0]),  # target is green
        4: np.array([0.8, 0.8, 0.8]),  # buildings are rgay
        5: np.array([0.9, 0.9, 0.9]),  # buildings are rgay
        6: np.array([0.7, 0.7, 0.7]),  # buildings are rgay
        7: np.array([0.6, 0.6, 0.6]),  # buildings are rgay
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

print(get_poly_colors())


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

def above(pos):
    return tuple(np.array([0, 0, 1]) + pos)

    # all sizes are unit.

class Escherville:
    def __init__(self,Nx: int, Ny: int, building_period: int = 4, building_height: int = 4, camera_ratio: np.array = np.array([1,-2,1])):
        self.Nx = Nx
        self.Ny = Ny
        self.building_period = building_period
        self.building_height = building_height
        self.Lx = Nx * building_period
        self.Ly = Ny * building_period
        self.Lz = 7  # maximum build height
        # create array to hold state of the world.
        self.airspace = np.zeros((self.Lx + 2, self.Ly + 2, self.Lz + 2)) + 9
        self.airspace[1:-1, 1:-1, 1:-1] = 0
        hbp = building_period // 2
        for dx in range(hbp):
            for dy in range(hbp):
                self.airspace[hbp // 2 + 1 + dx::building_period, hbp // 2 + 1 + dy::building_period, building_height] = 1
                self.airspace[hbp // 2 + 1 + dx::building_period, hbp // 2 + 1 + dy::building_period, 1:building_height] = 2
        possible_x = sum([list(np.arange(self.Lx)[hbp // 2 + d + 1::building_period]) for d in range(hbp)], [])
        possible_y = sum([list(np.arange(self.Ly)[hbp // 2 + d + 1::building_period]) for d in range(hbp)], [])
        self.possible_starts = [(a, b, building_height) for a in possible_x for b in possible_y]

        self.succesful_paths = None

        self.camera_ratio = camera_ratio



    def pathfinder(self,n_path_attempts):

        if self.succesful_paths is None:
            self.succesful_paths = 0

        while self.succesful_paths < n_path_attempts:
            _airspace = copy.deepcopy(self.airspace)
            _possible_starts = copy.deepcopy(self.possible_starts)
            pos = (0,0,0)
            while _airspace[pos] not in [1,3]:
                pos = _possible_starts[np.random.randint(len(_possible_starts))]

            t_dist = 0
            while t_dist < 5 or t_dist > 10:
                target = _possible_starts[np.random.randint(len(_possible_starts))]
                if _airspace[target] not in [1,3]:
                    t_dist = 0
                else:
                    t_dist = sum(abs(np.array(pos)-target))

            target_diag = [_ta for _ta in self.get_hyperdiag(target) if _ta[2] >= self.building_height]

            _airspace[pos] = 4
            _airspace[target] = 4

            state = "s" #no direction yet
            pathlen = 0
            while pathlen < 10 and not (pos in target_diag and state in ["r","l","b","f"]):
                #possible positions
                pos_pos_data = [([tuple(pos+_d) for _d in _dir],_state) for _state,_dir in transition_directions[state].items()]

                #filter for positions that can still be built into
                pos_pos_data = [p for p in pos_pos_data if all([_airspace[_p]<3 for _p in p[0]])]

                if len(pos_pos_data) == 0:
                    break

                #choose next step
                new_pos_data = pos_pos_data[np.random.randint(len(pos_pos_data))]
                #fill in master array cell
                fill = dir_to_fill[new_pos_data[1]]

                pos = new_pos_data[0][0]

                _airspace[pos] = fill

                if fill == 3:
                    _possible_starts.append(pos)
                if fill in [5,6,7,8]: #stairs
                    ab = above(pos)
                    #don't put stuff directly above stairs
                    _airspace[ab] = 9

                #prevent too many crossings in one line of sight
                hyperdiag = self.get_hyperdiag(pos)
                if sum([_airspace[dp] in [3,5,6,7,8] for dp in hyperdiag]) == 2:
                    for dp in hyperdiag:
                        _airspace[dp] = cube_blocking[_airspace[dp]]

                state = new_pos_data[1]

                if "u" in state:
                    pos = above(pos)

                pathlen += 1

            if (pos in target_diag and state in ["r","l","b","f"]):
                self.airspace = _airspace
                self.possible_starts = _possible_starts
                self.succesful_paths += 1

        self.project_airspace(cabinet=True)

    def get_hyperdiag(self, intersection):
        # line of sight is along +x,-2y,+z

        arrshape = self.airspace.shape

        lowers, uppers = [], []

        for j in [0, 1, 2]:
            bdys = sorted([-intersection[j] / self.camera_ratio[j], (arrshape[j] - intersection[j] - 1) / self.camera_ratio[j]])
            lowers.append(bdys[0])
            uppers.append(bdys[1])

        return [tuple(intersection + k * self.camera_ratio) for k in range(int(max(lowers)), int(min(uppers)) + 1)]

    def project_airspace(self,cabinet=True):

        n_colors = 4

        #redesign cluster finder
        cluster_map, self.cluster_hash = find_clusters(self.airspace, [2, 11])
        self.building_colors = np.random.randint(4, n_colors + 4, len(cluster_map))

        self.group_colors = []  # (.color_group,.orientation) format
        self.polygons = []
        for ix, iy, iz in itertools.product(range(1, self.Lx+1), range(1, self.Ly+1), range(1, self.Lz+1)):
            self.trim_polys(ix=ix,iy=iy,iz=iz)

        orientation_fixed_dim = {
            "top": 2,
            "side": 0,
            "front": 1,
        }

        self.joined_polys = []

        for merge_tuple in self.group_colors:
            d = orientation_fixed_dim[merge_tuple[1]]
            cardinal_polys = sorted([p for p in self.polygons if p.merge_tuple == merge_tuple], key=lambda x: x.coords[0, d])
            plane_coord = cardinal_polys[0].coords[0, d]
            this_slice = []
            for _poly in cardinal_polys:
                if _poly.coords[0, d] != plane_coord:
                    self.joined_polys.extend(join_polygons(this_slice))
                    this_slice = []
                    plane_coord = _poly.coords[0, d]
                this_slice.append(_poly)
            self.joined_polys.extend(join_polygons(this_slice))

        self.joined_polys.extend([p for p in self.polygons if not p.merges])

        fig = plt.figure(figsize=(200, 80))

        flat_proj = np.array([[1., 0.], [pers_x, pers_z], [0., 1.]])

        poly_colors = get_poly_colors()

        if cabinet:
            # joined_polys.sort(key = lambda x: tuple((x[0].mean(axis=0)*dxyz)[[1,2,0]]))
            #just sort themselves.
            #self.joined_polys = sorted(joined_polys, key=functools.cmp_to_key(custom_zorder))
            # lambda x: (max(x[0].dot(dxyz)),min(x[0].dot(dxyz))))
            # polygons.sort(key=lambda x: (max(x[0].dot(dxyz)),min(x[0].dot(dxyz))))
            self.joined_polys.sort()
            plt.axis('equal')
        else:
            ax = Axes3D(fig, auto_add_to_figure=False)
            ax.set_xlim3d(0, self.Lx)
            ax.set_ylim3d(0, self.Ly)
            ax.set_zlim3d(0, self.Lz)
            ax.set_proj_type("ortho")
            fig.add_axes(ax)
        for _poly in self.joined_polys:
            c = poly_colors[_poly.merge_tuple]
            if cabinet:
                x, y = _poly.coords.dot(flat_proj).T
                plt.fill(x, y, facecolor=c, edgecolor=c, zorder=1, lw=1)
                proj_lines = np.array(_poly.edges).dot(flat_proj).T.tolist()
                plt.plot(*proj_lines, color="k", zorder=1, solid_capstyle="round")  # , lw=1.5, ms=1, marker= "o")
            else:
                pol = Poly3DCollection([_poly.coords])
                linen = Line3DCollection(_poly.edges)
                linen.set_color("k")
                pol.set_color(mpl.colors.rgb2hex(c))
                pol.set_edgecolor(c)
                ax.add_collection3d(pol)
                ax.add_collection3d(linen)

        plt.show()

    def trim_polys(self, ix, iy, iz):

        # put all this logic into one function.
        cur_fill_code = self.airspace[(ix, iy, iz)]
        if cur_fill_code in [2, 11]:
            color = self.building_colors[self.cluster_hash[(ix, iy, iz)]]
        else:
            color = None

        if self.airspace[(ix, iy, iz - 1)] in [2, 11]:  # inside building
            cur_fill = cube_filling_building[cur_fill_code]
            # here have to pass the cluster
        else:
            cur_fill = cube_filling_free[cur_fill_code]

        #remove hidden faces
        if cur_fill_code == 3:
            cut_cur_fill = [cur_fill[0]]
            if self.airspace[(ix, iy - 1, iz)] in [0, 9] and self.airspace[(ix, iy - 1, iz - 1)] != 7:
                cut_cur_fill.append(cur_fill[1])
            if self.airspace[(ix, iy + 1, iz)] in [0, 9] and self.airspace[(ix, iy + 1, iz - 1)] != 8:
                cut_cur_fill.append(cur_fill[2])
            if self.airspace[(ix - 1, iy, iz)] in [0, 9] and self.airspace[(ix - 1, iy, iz - 1)] != 5:
                cut_cur_fill.append(cur_fill[3])
            if self.airspace[(ix + 1, iy, iz)] in [0, 9] and self.airspace[(ix + 1, iy, iz - 1)] != 6:
                cut_cur_fill.append(cur_fill[4])
            cur_fill = cut_cur_fill
        elif cur_fill_code in [2, 11]:
            cut_cur_fill = []
            if self.airspace[(ix, iy - 1, iz)] not in [2, 11]:
                cut_cur_fill.append(cur_fill[0])
            if self.airspace[(ix + 1, iy, iz)] not in [2, 11]:
                cut_cur_fill.append(cur_fill[1])
            cur_fill = cut_cur_fill
        elif cur_fill_code in [1, 4, 10]:  # roof, no building below
            cut_cur_fill = [cur_fill[0]]
            if self.airspace[(ix, iy, iz - 1)] not in [2, 11]:
                if not (self.airspace[(ix, iy - 1, iz - 1)] in [2, 11, 0, 9] or
                        self.airspace[(ix, iy - 1, iz)] in [1, 4, 10]):
                    cut_cur_fill.append(cur_fill[1])
                if not (self.airspace[(ix + 1, iy, iz - 1)] in [2, 11, 0, 9] or
                        self.airspace[(ix + 1, iy, iz)] in [1, 4, 10]):
                    cut_cur_fill.append(cur_fill[2])
            cur_fill = cut_cur_fill

        #modify and finalize the polygon
        displacement = np.array([ix, iy, iz])
        for _poly in cur_fill:
            new_poly = copy.deepcopy(_poly)
            if new_poly.color_group == 4 and color is not None: #recolor buildings
                new_poly.color_group = color
            new_poly.coords += displacement
            self.polygons.append(new_poly)
            if new_poly.merges:
                _key = (new_poly.color_group,new_poly.orientation)
                if _key not in self.group_colors:
                    self.group_colors.append(_key)


# def get_contour_lines(poly):
#     return [poly[[j,j+1]] for j in range(-1,len(poly)-1)]


pers_x = -camera_ratio[0]/camera_ratio[1]
pers_z = -camera_ratio[2]/camera_ratio[1]

def old_custom_zorder(poly1, poly2):
    tolerance = 1e-6
    min1, max1 = poly1.min(axis=0), poly1.max(axis=0)
    min2, max2 = poly2.min(axis=0), poly2.max(axis=0)
    dim_order = [int(min1[j]+tolerance > max2[j]) - int(max1[j] < min2[j] + tolerance) for j in [0,1,2]]
    dim_order[1] *= -1 #y-coord is reversed
    if sum([a == 0 for a in dim_order]) > 1:
        return sum(dim_order)
    else:
        #use the other trick with planes.
        d1 = poly1.dot(camera_ratio)
        d2 = poly2.dot(camera_ratio)
        dd1 = (max(d1),min(d1))
        dd2 = (max(d2),min(d2))
        if dd1 == dd2:
            return 0
        else:
            return 2*(dd1 > dd2) -1

def customer_zorder(coords,other_coords):

    margin = 6 #significant digits, if points are this close to the plane, consider them on the plane.

    uv = None

    if uv is None:
        uv = coords[1:3] - coords[0] + 0.

        sign_camera = np.sign(np.linalg.det(np.insert(uv, 2, camera_ratio, 0)))
        uv[0] *= sign_camera

        uv = uv[np.newaxis, :, :]

    sign_distances = np.sign(np.round(np.linalg.det(
        np.insert(uv * np.ones((len(other_coords), 1, 1)), 2, other_coords - coords[0], 1)
    ), margin))

    if -1 in sign_distances and 1 in sign_distances:
        return 0  # points on both sides of the plane
    else:
        return -sum(sign_distances)









def get_hyperdiag(arrshape,intersection):
    #line of sight is along +x,-2y,+z

    lowers,uppers = [],[]

    for j in [0,1,2]:
        bdys = sorted([-intersection[j]/camera_ratio[j],(arrshape[j]-intersection[j]-1)/camera_ratio[j]])
        lowers.append(bdys[0])
        uppers.append(bdys[1])

    return [tuple(intersection+k*camera_ratio) for k in range(int(max(lowers)),int(min(uppers))+1)]

def find_clusters(airspace,codes):
    Lx,Ly,Lz = airspace.shape
    cluster_hash = dict()
    cluster_map = []
    lookback = list(itertools.product(range(-1,1),range(-1,2),range(-1,2)))
    lookback = [l for l in lookback if l not in [(0,0,0),(0,1,0),(0,1,1)]]
    for ix,iy,iz in itertools.product(range(1,Lx-1), range(1,Ly-1), range(1,Lz)):
        if airspace[(ix,iy,iz)] in codes:
            unknown = True
            for dx,dy,dz in lookback:
                if (ix+dx,iy+dy,iz+dz) in cluster_hash:
                    unknown = False
                    cn = cluster_hash[(ix+dx,iy+dy,iz+dz)]
                    cluster_hash[(ix,iy,iz)] = cn
                    cluster_map[cn].append((ix,iy,iz))
                    break
            if unknown:
                cluster_hash[(ix, iy, iz)] = len(cluster_map)
                cluster_map.append([(ix,iy,iz)])

    return cluster_map, cluster_hash





mce = Escherville(3,3)
mce.pathfinder(5)



class cityscape:
    def __init__(self,d):
        self.d = d
        self.dx = 20
        self.dy = 10
        self.dz = 40

        self.DX = np.array([0,self.dx,self.dx,0])
        self.DZ = np.array([0,0,self.dz,self.dz])

        self.DY1 = np.array([0,self.dy,self.dy,0])
        self.DY2 = np.array([0,0,self.dy,self.dy])


        self.cols = ['navy', 'royalblue', 'silver']

        self.ecg = self.random_col_gen()
        # for flat projection
        self.pers_x = 1
        self.pers_z = 0.8

        self.flat = True if d > 999 else False

        self.dark = np.array([0.5,0.5,0.5,1])
        self.window_color = np.array([0.87,1,1,1])
        self.dark_window = self.make_dark(self.window_color)

        self.cmap = mpl.cm.get_cmap('Set1')
        self.bleiswijk = [(1.,0.,0.,1.),(1.,1.,0.,1.),(0.5,1.,0.,1.),(0.4,0.8,1.,1.),(0.,0.25,0.6,1.),(1.,0.4,0.8,1.)]
        self.rainbow_gen = self.rainbow()

    def rainbow(self):
        while True:
            for b in self.bleiswijk:
                yield b

    def eternal_col_gen(self):
        while True:
            for c in self.cols:
                yield c

    def make_dark(self,rgba):
        return tuple(self.dark * rgba)

    def make_light(self,rgba):
        return tuple(rgba + (1 - rgba) * self.dark)

    def random_col_gen(self):
        cmap = mpl.cm.get_cmap('Pastel2')

        while True:
            rgba = np.array(cmap(np.random.uniform()))
            yield self.make_dark(rgba)
            yield tuple(rgba)
            yield self.make_light(rgba)

    def next_color_gen(self):
        self.nowcol = np.array(self.cmap(np.random.uniform()))
        self.nowcol = np.array(self.bleiswijk[np.random.randint(6)])
        #self.nowcol = np.array(next(self.rainbow_gen))
        self.darkcol = self.make_dark(self.nowcol)
        self.lightcol = self.make_light(self.nowcol)


    def generate_windows(self,axy,bxy,nxy,az,bz,nz,xy0,z0,xy='x'):
        ux = 1/(nxy*(axy+bxy)+bxy)
        uy = 1/(nz*(az+bz)+bz)
        if xy == 'x':
            DXY = self.DX
            dxy = self.dx
        else:
            DXY = self.DY1
            dxy = self.dy
        base_xy = DXY*ux*axy
        base_z = self.DZ*uy*az
        for hxy in range(nxy):
            for hz in range(nz):
                yield xy0+(bxy+(axy+bxy)*hxy)*ux*dxy+base_xy, z0+(bz+(az+bz)*hz)*uy*self.dz+base_z


    def project(self,x,y,z):
        reciprocal = 1/(y+self.d)
        return x*self.d*reciprocal,z*self.d*reciprocal

    def flat_project(self,x,y,z):
        return x+self.pers_x*y,z+self.pers_z*y


    def roofstyle_1(self,x0,y0,z0,c):
        xx = np.array([x0]*4*c)
        yy = np.array([y0]*4*c)

        predz = np.array([0, self.dz] +[self.dz,0.5*self.dz,0.5*self.dz,self.dz]*(c-1)+ [self.dz,0])

        DZ = predz * 0.05*c + self.dz + z0

        jag_x = np.linspace(x0,x0+self.dx,2*c)
        DX = np.array([x for x in jag_x for _ in (0, 1)])

        jag_y = np.linspace(y0,y0+self.dy,2*c)
        DY = np.array([y for y in jag_y for _ in (0, 1)])

        front = [DX,yy,DZ,self.nowcol]
        side = [xx,DY,DZ,self.darkcol]
        back = [DX,yy+self.dy,DZ,self.nowcol]
        otherside = [xx+self.dx,DY,DZ,self.darkcol]

        # front = [self.DX+x0,np.array([y0]*4),DZ,self.nowcol]
        # side = [xx,self.DY1+y0,DZ,self.darkcol]
        # back = [self.DX+x0,np.array([y0]*4)+self.dy,DZ,self.nowcol]
        # otherside = [xx+self.dx,self.DY1+y0,DZ,self.darkcol]
        for s in [back,otherside,side,front]:
            yield s

    def roofstyle_2(self,x0,y0,z0,c):
        delta_y = self.dy/np.random.randint(2,5)
        z = z0+self.dz
        zup = z0+self.dz*(1+0.1*c)
        front_x = np.array([x0,x0+0.5*self.dx,x0+self.dx])
        front_y = np.array([y0]*3)
        front_z =  np.array([z,zup,z])
        front = [front_x,front_y,front_z,self.nowcol]
        #back = [front_x,front_y+self.dy,front_z,self.nowcol]
        roof_x_1 = np.array([x0, x0 + 0.5 * self.dx, x0 + 0.5 * self.dx, x0])
        roof_x_2 = np.array([x0+self.dx, x0 + 0.5 * self.dx, x0 + 0.5 * self.dx, x0+self.dx])
        roof_z = np.array([z,zup,zup,z])

        if x0+self.dx < 0 or self.flat:
            first = roof_x_1
            second = roof_x_2
        else:
            first = roof_x_2
            second = roof_x_1

        col = self.darkcol
        for roof_x in [first,second]:
            for y in np.arange(y0,y0+self.dy,delta_y):
                roof_y = np.array([y,y,y+delta_y,y+delta_y])
                yield [roof_x,roof_y,roof_z,col]
            col = self.lightcol
        yield front

    def roofstyle_3(self,x0,y0,z0,cc):
        for c in [cc,np.random.randint(1,5)]:
            x = x0 + self.dx*0.25*(1+2*((c-1)//2))
            y = y0 + self.dy*0.25*(1+2*(c % 2))

            width = 0.1

            DX = self.DX*width + x
            DZ = self.DZ*0.2 + z0 + self.dz
            DY1 = self.DY1*width + y
            DY2 = self.DY2*width + y

            front = [DX,np.array([y]*4),DZ,self.nowcol]

            side = [np.array([x]*4),DY1,DZ,self.darkcol]
            if x+self.dx*width < 0 or self.flat:
                side[0] += self.dx*width

            top = [DX,DY2,np.array([z0+self.dz]*4,dtype=float),self.lightcol]
            if z0+self.dz*0.2 < 0:
                top[2] += self.dz*0.2

            yield side
            yield front
            yield top


    def building(self,x0,y0,z0):

        self.next_color_gen()

        DX = self.DX + x0
        DZ = self.DZ + z0

        DY1 = self.DY1 + y0
        DY2 = self.DY2 + y0

        front = [DX,np.array([y0]*4),DZ,self.nowcol]

        side = [np.array([x0]*4),DY1,DZ,self.darkcol]
        if x0+self.dx < 0 or self.flat:
            side[0] += self.dx

        top = [DX,DY2,np.array([z0]*4),self.lightcol]
        if z0+self.dz < 0:
            top[2] += self.dz




        yield side
        az,bz,nz,ay,by,ny = np.random.randint(1,6,6)
        for window in self.generate_windows(az,bz,nz,ay,by,ny,y0,z0,'z'):
            yield [side[0],window[0],window[1],self.dark_window]

        yield front
        yield top

        #ax,bx,nx,ay,by,ny = np.random.randint(1,6,6)
        for window in self.generate_windows(az,bz,nz,ay,by,ny,x0,z0):
            yield [window[0],front[1],window[1],self.window_color]

        c = np.random.randint(1, 5)
        roofint = np.random.randint(4)
        if roofint:
            rooffunc = [self.roofstyle_1,self.roofstyle_2,self.roofstyle_3][roofint-1]
            for s in rooffunc(x0,y0,z0,c):
                yield s

    def grid(self,nx,mx,ny,z):
        z0 = -z*self.dz
        for y0 in self.dy*np.arange(2*ny,0,-2):
            xx = self.dx*np.arange(2*nx-.5,2*mx+.5,2)
            for x0 in sorted(xx,key= lambda x: -abs(x)):
                for facets in self.building(x0,y0,z0):
                    yield facets

    def plotall(self,nx,mx,ny,z):
        mapper = self.flat_project if self.flat else self.project
        plt.figure(figsize=(80, 32))
        plt.axis('equal')
        for face in self.grid(nx,mx,ny,z):
            # if face[3] == 0:
            #     fc = self.window_color
            # elif face[3] == 1:
            #     fc = self.dark_window
            # else:
            #     fc = next(self.ecg)
            x,y = mapper(face[0],face[1],face[2])
            plt.fill(x,y,facecolor=face[3],edgecolor='k')
        #plt.savefig('perspectiville_'+self.cmap.name+'_'+str(self.d)+'.png', dpi=300)
        plt.savefig('perspectiville_rainbow_' + str(self.d) + '.png', dpi=300)
        plt.show()


# wise to increase the canvas size with the number of buildings depicted. Then the black lines will have
#a nice weight.
if __name__ == '__main__':
    pass
    #tester = cityscape(9)
    #tester.plotall(-12,12,18,4)

    #
    # tester = cityscape(1001)
    # tester.plotall(-10,10,20,4)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
