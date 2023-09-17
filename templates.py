import numpy as np
import copy

from classes import Poly

ns = 5 #n steps


cube_filling_building = {
    0: [], # empty space
    1: [  # building roof
        Poly(coords=np.array([[0,0,0],[100,0,0],[100,100,0],[0,100,0]]),
             orientation="top",
             merges=True,
             color_group=4
             ),
        Poly(coords=np.array([[0,0,-20],[100,0,-20],[100,0,0],[0,0,0]]),
             orientation="front",
             merges=True,
             color_group=4),
        Poly(coords=np.array([[100,0,-20],[100,100,-20],[100,100,0],[100,0,0]]),
             orientation="side",
             merges=True,
             color_group=4)
    ],
    2: [  # building center
        Poly(coords=np.array([[0,0,0],[100,0,0],[100,0,100],[0,0,100]]),
             orientation="front",
             merges=True,
             color_group=4),
        Poly(coords=np.array([[100,0,0],[100,100,0],[100,100,100],[100,0,100]]),
             orientation="side",
             merges=True,
             color_group=4)
    ],
    3: [  # flat path
        Poly(coords=np.array([[0,0,0],[100,0,0],[100,100,0],[0,100,0]]),
             orientation="top",
             merges=True,
             color_group=0),
        Poly(coords=np.array([[0,0,2],[10,0,0],[10,0,20],[90,0,20],[90,0,0],[100,0,0],[100,0,30],[0,0,30]]),
             orientation="front",
             merges=False,
             color_group=0),
        Poly(coords=np.array([[0,100,0],[10,100,0],[10,100,20],[90,100,20],[90,100,0],[100,100,0],[100,100,30],[0,100,30]]),
             orientation="front",
             merges=False,
             color_group=0),
        Poly(coords=np.array([[0,0,0],[0,10,0],[0,10,20],[0,90,20],[0,90,0],[0,100,0],[0,100,30],[0,0,30]]),
             orientation="side",
             merges=False,
             color_group=0),
        Poly(coords=np.array([[100,0,0],[100,10,0],[100,10,20],[100,90,20],[100,90,0],[100,100,0],[100,100,30],[100,0,30]]),
             orientation="side",
             merges=False,
             color_group=0)
    ],
    4:  [  # building roof, target
        Poly(coords=np.array([[0,0,0],[100,0,0],[100,100,0],[0,100,0]]),
             orientation="top",
             merges=False,
             color_group=3),
        Poly(coords=np.array([[0, 0, -20], [100, 0, -20], [100, 0, 0], [0, 0, 0]]),
             orientation="front",
             merges=True,
             color_group=4),
        Poly(coords=np.array([[100, 0, -20], [100, 100, -20], [100, 100, 0], [100, 0, 0]]),
             orientation="side",
             merges=True,
             color_group=4)
    ],
    5: [  # up to right / down to left
        Poly(coords=np.array([*[[100*(j//2)/ns,0,100*((j+1)//2)/ns] for j in range(2*ns)],[100,0,100],[100,0,0]]),
             orientation="front",
             merges=True,
             color_group=4),
        *[
            Poly(coords=np.array([
                [100*j/ns,0,100*(j+1)/ns],
                [100*(j+1)/ns,0,100*(j+1)/ns],
                [100*(j+1)/ns,100,100*(j+1)/ns],
                [100*j/ns,100,100*(j+1)/ns]
            ]),
                 orientation="top",
                 merges=False,
                 color_group=0)
            for j in range(ns)
        ]
    ],
    6: [  # down to right / up to left
        Poly(coords=np.array([*[[100*((j+1)//2)/ns,0,100-100*(j//2)/ns] for j in range(2*ns)],[100,0,0],[0,0,0]]),
             orientation="front",
             merges=True,
             color_group=4),
        *[
            Poly(coords=np.array([
                [100*j/ns,0,100-100*j/ns],
                [100*(j+1)/ns,0,100-100*j/ns],
                [100*(j+1)/ns,100,100-100*j/ns],
                [100*j/ns,100,100-100*j/ns]
            ]),
                 orientation="top",
                 merges=False,
                 color_group=0)
            for j in range(ns)
        ],
        *[
            Poly(coords=np.array([
                [100*(j+1)/ns,0,100-100*j/ns],
                [100*(j+1)/ns,0,100-100*(j+1)/ns],
                [100*(j+1)/ns,100,100-100*(j+1)/ns],
                [100*(j+1)/ns,100,100-100*j/ns]
            ]),
                 orientation="side",
                 merges=False,
                 color_group=0)
            for j in range(ns)
        ]
    ],
    7: [  # up to back / down to front
        Poly(coords=np.array([*[[100,100*(j//2)/ns,100*((j+1)//2)/ns] for j in range(2*ns)],[100,100,100],[100,100,0]]),
             orientation="side",
             merges=True,
             color_group=4),
        *[
            Poly(coords=np.array([
                [0,100*j/ns,100*j/ns],
                [100,100*j/ns,100*j/ns],
                [100,100*j/ns,100*(j+1)/ns],
                [0,100*j/ns,100*(j+1)/ns]
            ]),
                 orientation="front",
                 merges=False,
                 color_group=0)
            for j in range(ns)
        ],
        *[
            Poly(coords=np.array([
                [0,100*j/ns,100*(j+1)/ns],
                [100,100*j/ns,100*(j+1)/ns],
                [100,100*(j+1)/ns,100*(j+1)/ns],
                [0,100*(j+1)/ns,100*(j+1)/ns]
            ]),
                 orientation="top",
                 merges=False,
                 color_group=0)
            for j in range(ns)
        ]
    ],
    8: [  # down to back / up to front
        Poly(coords=np.array([*[[100,100*((j+1)//2)/ns,100-100*(j//2)/ns] for j in range(2*ns)],[100,100,0],[100,0,0]]),
             orientation="side",
             merges=True,
             color_group=4),
        *[
            Poly(coords=np.array([
                [0,100*j/ns,100-100*j/ns],
                [100,100*j/ns,100-100*j/ns],
                [100,100*(j+1)/ns,100-100*j/ns],
                [0,100*(j+1)/ns,100-100*j/ns]
            ]),
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
cube_filling_free[5][0] = Poly(coords=np.array(list(cube_filling_building[5][0].coords[:-1])+[[100,0,100-100/ns],[100/ns,0,0]]),
                              orientation="front",
                              merges=False,
                              color_group=0)
cube_filling_free[6][0] = Poly(coords=np.array(list(cube_filling_building[6][0].coords[:-1])+[[100-100/ns,0,0],[0,0,100-100/ns]]),
                               orientation="front",
                               merges=False,
                               color_group=0)
cube_filling_free[7][0] = Poly(coords=np.array(list(cube_filling_building[7][0].coords[:-1])+[[100,100,100-100/ns],[100,100/ns,0]]),
                               orientation="side",
                               merges=False,
                               color_group=0)
cube_filling_free[8][0] = Poly(coords=np.array(list(cube_filling_building[8][0].coords[:-1])+[[100,100-100/ns,0],[100,0,100-100/ns]]),
                               orientation="side",
                               merges=False,
                               color_group=0)

cube_filling_building[5].append(
    Poly(coords=np.array([[100,0,0],[100,100,0],[100,100,100],[100,0,100]]),
         orientation="side",
         merges=True,
         color_group=4)
)
cube_filling_building[8].append(
    Poly(coords=np.array([[0,0,0],[100,0,0],[100,0,100],[0,0,100]]),
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
        1: np.array([1, 1, 0]),  # windows are yellow
        2: np.array([1, 0, 0]),  # roof is red
        3: np.array([0, 1, 0]),  # target is green
        4: np.array([1, 1, 1]),  # buildings are rgay
        5: np.array([0.7, 0.7, 0.7]),  # buildings are rgay
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

def get_window(nx,ny,fx,fy):
    """
    return a list of 2d window poly coordinates. all params in [0,1,2]
    :param nx: number of windows in x dir
    :param ny: number of windows in y dir
    :param fx: thickness of mullion in x dir
    :param fy: thickness of mullion in y dir
    :return: list of len nx*ny of windows
    """

    def ser(_n,_f):
        _di = {
            0: [[10+14*_f,90-14*_f]],
            1: [[10+6*_f,50-6*_f],[50+6*_f,90-6*_f]],
            2: [[10+4*_f, 36-4*_f], [36+4*_f,64-4*_f], [64+4*_f,90-4*_f]],
        }
        return _di[_n]

    for f in [fx,fy,nx,ny]:
        if f < 0 or f > 2:
            raise ValueError("all args in [0,1,2]")

    pxl = ser(nx,fx)
    pyl = ser(ny,fy)

    lap = [(0,0),(0,1),(1,1),(1,0)]

    return [np.array([[px[j],py[k]] for j,k in lap]) for px in pxl for py in pyl]


#practice polygons
p_0 = np.array([[2,2,2,2],[1,2,2,1],[3,3,4,4]]).T
p_1 = np.array([[1,2,2,1],[2,2,2,2],[3,3,4,4]]).T
p_2 = np.array([[1,2,2,1],[3,3,4,4],[2,2,2,2]]).T
p_3 = np.array([[1.5,1.8,1.8,1.5],[2,2,2,2],[3.5,3.5,4.5,4.5]]).T


my_graph = {1: [2], 2: [1, 5], 3: [4], 4: [3, 5], 5: [6], 6: [7], 7: [8], 8: [6, 9], 9: []}


fat_graph = {0:[1,2,3,4],1:[5],2:[6],3:[7],4:[8],5:[10],6:[10],7:[10],8:[10],9:[5],10:[9,11],11:[0,10]}
pat_graph = {0:[1,2],1:[2],2:[3,5],3:[1],4:[2],5:[0,4]}