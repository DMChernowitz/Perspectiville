import numpy as np
import copy

from classes import Poly

from config import ns


dd = 100//ns

def front_to_side(arr):
    return arr[:,(1,0,2)]+[[100,0,0]]

def get_castle_walls():
    polylist = []
        # front
    polylist.append(Poly(
        coords = sum([[[j*dd,0,25*((j+1)%2+1)],[(j+1)*dd,0,25*((j+1)%2+1)]] for j in range(ns)],[[0,0,0]])+[[100,0,0]],
        orientation="front",
        merges=True,
        color_group=4)
    )
    # back
    polylist.append(copy.deepcopy(polylist[0]))
    polylist[-1].coords += [0,100,0]
    # left
    polylist.append(copy.deepcopy(polylist[0]))
    polylist[-1].coords = front_to_side(polylist[-1].coords)
    polylist[-1].orientation = "side"
    # right
    polylist.append(copy.deepcopy(polylist[-1]))
    polylist[-1].coords -= [100,0,0]
    return polylist


cube_filling_building = {
    0: [], # empty space
    1: [  # building roof
        Poly(coords=(lining_front := np.array([[0,0,-20],[100,0,-20],[100,0,0],[0,0,0]])),
             orientation="front",
             merges=True,
             color_group=4),
        Poly(coords=front_to_side(lining_front),
             orientation="side",
             merges=True,
             color_group=4),
        Poly(coords=(top_plate := np.array([[0, 0, 0], [100, 0, 0], [100, 100, 0], [0, 100, 0]])),
             orientation="top",
             merges=True,
             color_group=4
             ),
    ],
    2: [  # building center
        Poly(coords=(front_wall := np.array([[0,0,0],[100,0,0],[100,0,100],[0,0,100]])),
             orientation="front",
             merges=True,
             color_group=4),
        Poly(coords=front_to_side(front_wall),
             orientation="side",
             merges=True,
             color_group=4)
    ],
    3: [  # flat path
        Poly(coords=top_plate,
             orientation="top",
             merges=True,
             color_group=0),
        Poly(coords=(rails := np.array([[0,0,0],[10,0,0],[10,0,20],[90,0,20],[90,0,0],[100,0,0],[100,0,30],[0,0,30]])),
             orientation="front",
             merges=False,
             color_group=0),
        Poly(coords= rails + [[0,100,0]],
             orientation="front",
             merges=False,
             color_group=0),
        Poly(coords=front_to_side(rails),
             orientation="side",
             merges=False,
             color_group=0),
        Poly(coords=front_to_side(rails) - [[100, 0, 0]],
             orientation="side",
             merges=False,
             color_group=0)
    ],
    4:  [],  # building roof, target, used to be same as 1
    5: [  # up to right / down to left
        Poly(coords=(stairs:=np.array([*[[dd*(j//2),0,dd*((j+1)//2)] for j in range(2*ns)],[100,0,100],[100,0,0]])),
             orientation="front",
             merges=True,
             color_group=4),
        *[
            Poly(coords=np.array([
                [dd*j,0,dd*(j+1)],
                [dd*(j+1),0,dd*(j+1)],
                [dd*(j+1),100,dd*(j+1)],
                [dd*j,100,dd*(j+1)]
            ]),
                 orientation="top",
                 merges=False,
                 color_group=0)
            for j in range(ns)
        ]
    ],
    6: [  # down to right / up to left
        Poly(coords=(stairs2 := np.array([*[[dd*((j+1)//2),0,100-dd*(j//2)] for j in range(2*ns)],[100,0,0],[0,0,0]])),
             orientation="front",
             merges=True,
             color_group=4),
        *[
            Poly(coords=np.array([
                [dd*j,0,100-dd*j],
                [dd*(j+1),0,100-dd*j],
                [dd*(j+1),100,100-dd*j],
                [dd*j,100,100-dd*j]
            ]),
                 orientation="top",
                 merges=False,
                 color_group=0)
            for j in range(ns)
        ],
        *[
            Poly(coords=np.array([
                [dd*(j+1),0,100-dd*j],
                [dd*(j+1),0,100-dd*(j+1)],
                [dd*(j+1),100,100-dd*(j+1)],
                [dd*(j+1),100,100-dd*j]
            ]),
                 orientation="side",
                 merges=False,
                 color_group=0)
            for j in range(ns)
        ]
    ],
    7: [  # up to back / down to front
        Poly(coords=front_to_side(stairs),
             orientation="side",
             merges=True,
             color_group=4),
        *[
            Poly(coords=np.array([
                [0,dd*j,dd*j],
                [100,dd*j,dd*j],
                [100,dd*j,dd*(j+1)],
                [0,dd*j,dd*(j+1)]
            ]),
                 orientation="front",
                 merges=False,
                 color_group=0)
            for j in range(ns)
        ],
        *[
            Poly(coords=np.array([
                [0,dd*j,dd*(j+1)],
                [100,dd*j,dd*(j+1)],
                [100,dd*(j+1),dd*(j+1)],
                [0,dd*(j+1),dd*(j+1)]
            ]),
                 orientation="top",
                 merges=False,
                 color_group=0)
            for j in range(ns)
        ]
    ],
    8: [  # down to back / up to front
        Poly(coords=front_to_side(stairs2),
             orientation="side",
             merges=True,
             color_group=4),
        *[
            Poly(coords=np.array([
                [0,dd*j,100-dd*j],
                [100,dd*j,100-dd*j],
                [100,dd*(j+1),100-dd*j],
                [0,dd*(j+1),100-dd*j]
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
cube_filling_building[12] = cube_filling_building[1][:2] # slanted/sloping roof
cube_filling_building[13] = cube_filling_building[3]  # path but also an origin
cube_filling_building[14] = copy.deepcopy(cube_filling_building[1]) + get_castle_walls() #castle



cube_filling_free = copy.deepcopy(cube_filling_building)
cube_filling_free[5][0] = Poly(coords=np.array(list(cube_filling_building[5][0].coords[:-1])+[[100,0,100-dd],[dd,0,0]]),
                              orientation="front",
                              merges=False,
                              color_group=0)
cube_filling_free[6][0] = Poly(coords=np.array(list(cube_filling_building[6][0].coords[:-1])+[[100-dd,0,0],[0,0,100-dd]]),
                               orientation="front",
                               merges=False,
                               color_group=0)
cube_filling_free[7][0] = Poly(coords=np.array(list(cube_filling_building[7][0].coords[:-1])+[[100,100,100-dd],[100,dd,0]]),
                               orientation="side",
                               merges=False,
                               color_group=0)
cube_filling_free[8][0] = Poly(coords=np.array(list(cube_filling_building[8][0].coords[:-1])+[[100,100-dd,0],[100,0,100-dd]]),
                               orientation="side",
                               merges=False,
                               color_group=0)

cube_filling_building[5].append(
    Poly(coords=front_to_side(front_wall),
         orientation="side",
         merges=True,
         color_group=4)
)
cube_filling_building[8].append(
    Poly(coords=front_wall,
         orientation="front",
         merges=True,
         color_group=4)
)




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


def get_chimney_bds(droof: np.array, j: int, dd: int):
    """
    :param droof roof scale vector
    :param j: 0 for slant_front, 1 for slope_side
    :param dd: width of stairs e.d.
    :return: horizontal and vertical coords of bottom of chimney
    """

    dx,dz = droof[0,(j,2)]
    unit = 100
    center = (3-2*j)*dx*unit//4

    width = dd//(2*dx)+bool(dd%(2*dx))
    stop = (center-width*dx)//(2*dx)

    xl = stop*2*dx
    xr = (stop+width)*2*dx

    zl = 2*stop*dz
    zr = 2*(stop+width)*dz
    if j == 0:
        zl = dz*unit - zl
        zr = dz*unit - zr

    return xl,xr,zl,zr

def get_chimneys(droof,j):
    polylist = []
    xl, xr, zl, zr = get_chimney_bds(droof, j, dd)

    z_top = droof[0][2]*50
    for yl in range(50 - dd // 2, 100 * droof[0][1-j], 100):
        for dy in [0, dd]:
            polylist.append(Poly(
                coords=[[xl, yl + dy, zl], [xr, yl + dy, zr], [xr, yl + dy, z_top], [xl, yl + dy, z_top]],
                orientation=["front", "side"][j],
                merges=False,
                color_group=4)
            )
        for _x, _z in [(xl, zl), (xr, zr)]:
            polylist.append(Poly(
                coords=[[_x, yl, _z], [_x, yl + dd, _z], [_x, yl + dd, z_top], [_x, yl, z_top]],
                orientation=["side", "front"][j],
                merges=False,
                color_group=4)
            )
    if j:
        for _poly in polylist:
            _poly.coords = _poly.coords[:,(1,0,2)]
    return polylist

def get_inset_outset(mode: str, dw: int, dl: int, jdir: int, droof: np.array) -> dict:
    """
    get polys for insets and outsets
    :param dw: distance along slant hori and vert
    :param dl: distance perpendicular to slant
    :param jdir: 0 for slant_front, or 1, for slope_side orientation
    :return: dict with unscaled polys, gap (if mode=="inset), and window polys.
    """

    #standard is for j=1, can rotate for j=0
    polylist = []
    d_win = 6 #window mullion
    unit = 100
    xl,xr = dl, unit-dl
    yf,yb = dw, unit//2-dw
    depth,height = (yb,yf) if mode == "inset" else (yf,yb)
    length = xl if mode=="inset" else xr

    def rot_and_scale(imp):
        ar = np.array(imp)
        return droof*(ar if jdir else ar[:,(1,0,2)]*[[-1,-1,1]]+[[unit,unit,0]])

    polylist.append(Poly(
            coords=rot_and_scale([[xl,depth,yf],[xr,depth,yf],[xr,depth,yb],[xl,depth,yb]]),
            orientation=["side","front"][jdir],
            merges=False,
            color_group=4)
    )
    polylist.append(Poly(
            coords=rot_and_scale([[xl,yf,height],[xr,yf,height],[xr,yb,height],[xl,yb,height]]),
            orientation="top",
            merges=False,
            color_group=4)
    )
    polylist.append(Poly(
        coords=rot_and_scale([[length,depth,height],[length,yb,yb],[length,yf,yf]]),
        orientation=["front","side"][jdir],
        merges=False,
        color_group=4)
    )
    polylist.append(Poly(
            coords=rot_and_scale([
                [xl+d_win,depth,yf+d_win],
                [xr-d_win,depth,yf+d_win],
                [xr-d_win,depth,yb-d_win],
                [xl+d_win,depth,yb-d_win]
            ]) + polylist[0].camera_ratio,
            orientation=["front","side"][jdir],
            merges=False,
            color_group=1)
    )
    holes = [rot_and_scale([[xl,yf,yf],[xr,yf,yf],[xr,yb,yb],[xl,yb,yb]])] if mode == "inset" else []
    return {"polylist": polylist, "holes": holes}



def get_roof_template(orient, m, droof):
    """
    :param orient: orientation of the roof
    :param m: style enum; 0: nothing, 1 & 2: tiles, 3: inset, 4: outset, 5: chimneys
    :param droof:
    :return:
    """
    jdir = 0 if orient == "slant_front" else 1

    slant_nr = str(int(3*np.arctan(droof[0][2] / (2*droof[0][jdir]))))

    polylist = []
    if m in [1,2]:
        l = [None,2,5][m]
    else:
        l = 1
    w = 100 // l

    if orient == "slant_front":
        polylist.append(Poly(
            coords=droof*[[0,0,0],[100,0,0],[50,0,50]],
            orientation="front",
            merges=True,
            color_group=4)
        )
        for j in range(l):
            polylist.append(Poly(
                coords=droof*[[100,w*j,0],[100,w*(j+1),0],[50,w*(j+1),50],[50,w*j,50]],
                orientation=f"slant_{slant_nr}",
                merges=False,
                color_group=2)
            )
            polylist.append(Poly(
                coords=droof*[[0,w*j,0],[0,w*(j+1),0],[50,w*(j+1),50],[50,w*j,50]],
                orientation=f"tnals_{slant_nr}",
                merges=False,
                color_group=2)
            )
        if droof[0][0] >= 2 and m in [1,2]:
            dw = 3 * np.random.randint(3-m, 4)
            dl = 5 * np.random.randint(3-m, 4)
            for j in range(l): #add skylights
                polylist.append(Poly(
                    coords=droof * [
                        [100-dl, w * j+dw, dl],
                        [100-dl, w * (j + 1)-dw, dl],
                        [50+dl, w * (j + 1)-dw, 50-dl],
                        [50+dl, w * j+dw, 50-dl]
                    ] + polylist[0].camera_ratio,
                    orientation=f"slant_{slant_nr}",
                    merges=False,
                    color_group=1)
                )
                polylist.append(Poly(
                    coords=droof * [
                        [dl, w * j+dw, dl],
                        [dl, w * (j + 1)-dw, dl],
                        [50-dl, w * (j + 1)-dw, 50-dl],
                        [50-dl, w * j+dw, 50-dl]
                    ] + polylist[0].camera_ratio,
                    orientation=f"tnals_{slant_nr}",
                    merges=False,
                    color_group=1)
                )

    if orient == "slope_side":
        polylist.append(Poly(
            coords=droof*[[100,0,0],[100,100,0],[100,50,50]],
            orientation="side",
            merges=True,
            color_group=4)
        )
        for j in range(l):
            polylist.append(Poly(
                coords=droof*[[w*j,0,0],[w*(j+1),0,0],[w*(j+1),50,50],[w*j,50,50]],
                orientation=f"slope_{slant_nr}",
                merges=False,
                color_group=2)
            )
        if droof[0][1] >= 2 and m in [1,2]:
            dw = 3 * np.random.randint(3-m, 4)
            dl = 5 * np.random.randint(3-m, 4)
            for j in range(l):
                polylist.append(Poly(
                    coords=droof * [
                        [w * j+dw, dl, dl],
                        [w * (j + 1)-dw, dl, dl],
                        [w * (j + 1)-dw, 50-dl, 50-dl],
                        [w * j+dw, 50-dl, 50-dl]
                    ] + polylist[0].camera_ratio,
                    orientation=f"slope_{slant_nr}",
                    merges=False,
                    color_group=1)
                )

    if m in [3,4]:
        mode = "inset" if m == 3 else "outset"
        dw = 4*np.random.randint(2,4)
        dl = 10*np.random.randint(1,4)
        pieces_dict = get_inset_outset(mode=mode,dw=dw,dl=dl,jdir=jdir,droof=droof)
        polylist[1].holes = pieces_dict["holes"]
        polylist.extend(pieces_dict["polylist"])
    if m == 5: #chimney
        polylist.extend(get_chimneys(droof,jdir))

    return polylist

def get_single_chimney():
    unit = 100
    height = np.random.randint(2,4)*unit//2

    polylist = []

    if np.random.randint(3)==0:  # make a housy
        xwidth = unit-dd
        xl = dd//2
        ywidth = unit-2*dd
        yl = dd
        polylist.append(Poly(
            coords=[[xl, yl, 0], [xl + xwidth, yl, 0], [xl + xwidth, yl, height],
                    [xl, yl, height]],
            orientation="front",
            merges=False,
            color_group=4)
        )
        polylist.append(Poly(
            coords=[[xl + xwidth, yl, 0], [xl + xwidth, yl + ywidth, 0], [xl + xwidth, yl + ywidth, height],
                    [xl + xwidth, yl, height]],
            orientation="side",
            merges=False,
            color_group=4)
        )
        polylist.append(Poly(
            coords=[[xl, yl, height], [xl + xwidth, yl, height], [xl + xwidth, yl + ywidth, height],
                    [xl, yl + ywidth, height]],
            orientation="top",
            merges=False,
            color_group=4)
        )
        polylist.append(Poly(
            coords=[[xl+dd, yl, 0], [xl + xwidth-dd, yl, 0], [xl + xwidth-dd, yl, height-dd],
                    [xl+dd, yl, height-dd]]+polylist[0].camera_ratio,
            orientation="slope_4",
            merges=False,
            color_group=2)
        )

    else:  # make a chimney
        width = dd
        xl = unit // 2 - width // 2
        for dy in [0, width]:
            polylist.append(Poly(
                coords=[[xl, xl + dy, 0], [xl + width, xl + dy, 0], [xl + width, xl + dy, height],
                        [xl, xl + dy, height]],
                orientation="front",
                merges=False,
                color_group=4)
            )
        for dx in [0, width]:
            polylist.append(Poly(
                coords=[[xl + dx, xl, 0], [xl + dx, xl + width, 0], [xl + dx, xl + width, height],
                        [xl + dx, xl, height]],
                orientation="side",
                merges=False,
                color_group=4)
            )


    return polylist


#practice polygons
p_0 = np.array([[2,2,2,2],[1,2,2,1],[3,3,4,4]]).T
p_1 = np.array([[1,2,2,1],[2,2,2,2],[3,3,4,4]]).T
p_2 = np.array([[1,2,2,1],[3,3,4,4],[2,2,2,2]]).T
p_3 = np.array([[1.5,1.8,1.8,1.5],[2,2,2,2],[3.5,3.5,4.5,4.5]]).T


my_graph = {1: [2], 2: [1, 5], 3: [4], 4: [3, 5], 5: [6], 6: [7], 7: [8], 8: [6, 9], 9: []}


fat_graph = {0:[1,2,3,4],1:[5],2:[6],3:[7],4:[8],5:[10],6:[10],7:[10],8:[10],9:[5],10:[9,11],11:[0,10]}
pat_graph = {0:[1,2],1:[2],2:[3,5],3:[1],4:[2],5:[0,4]}