import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from shapely.geometry import Polygon, mapping
import copy

from config import camera_ratio, proj_array

class Poly:
    """
    will have coordinates, sorting function, can hold which edges to draw, and comparison.
    """

    def __init__(
            self, coords: np.array,
            color_group: int = None,
            orientation: str = None,
            merges: bool = False,
            camera_ratio: np.array = camera_ratio,
            proj_array: np.array = proj_array,
            holes: list = None,
    ):
        if not type(coords) == np.array:
            coords = np.array(coords)
        self.coords = coords
        self.color_group = color_group  # 0: path, 1: window, 2: roof, 3+: buildings
        self.orientation = orientation  # top, side, front, slant_{i}
        self.merges = merges
        if holes is None:
            holes = []
        self.holes = [h if type(h) == np.array else np.array(h) for h in holes]

        # internal variables:
        # self._edges = None
        self._normal_vec = None
        self._d_const = None
        self._camera_ratio = camera_ratio
        self._min_max_from_view = None
        self._projection = None
        self._proj_array = proj_array
        self._flat_shapely = None
        self._camera_side = None
        self._mm_uv = None
        self._hole_projections = None

    def distance_to_plane(self, _co):
        return -(self.d_const + _co.dot(self.normal_vec)) / self.normal_vec.dot(self.camera_ratio)

    def subtract(self,other_poly_list):
        original = copy.deepcopy(self.flat_shapely)
        for other_poly in other_poly_list:
            behind_infront = self.order_us(other_poly)
            if behind_infront == (True,False):
                original = original.difference(other_poly.flat_shapely)
            elif behind_infront == (True,True):
                #diff = list(self.flat_shapely.difference(other_poly.flat_shapely).geoms)
                diff = self.flat_shapely.intersection(other_poly.flat_shapely)
                if hasattr(diff,"geoms"):
                    diff_list = list(diff.geoms)
                else:
                    diff_list = [diff]
                for _partpoly in diff_list:
                    _co = np.insert(_partpoly.exterior.coords.xy,1,0,0).T[:-1]
                    if sum(self.distance_to_plane(_co)) < sum(other_poly.distance_to_plane(_co)): #other is closer to camera
                        original = original.difference(_partpoly)
                    if list(_partpoly.interiors):
                        print("missed an interior")
                #get components of it with mapping
                #find which ones are in front, with normal vec logic
                #remove those from original.
            else:
                print("Shouldn't be cutting this")

        if hasattr(original,"geoms"):
            return list(original.geoms)
        return [original]
        # mor = mapping(original)
        # if mor["type"] == "Polygon":
        #     return [np.array(u[:-1]) for u in mor["coordinates"]]
        # else:
        #     return [np.array(u[0])[:-1] for u in mor["coordinates"]]

    @property
    def projection(self):
        if self._projection is None:
            self._projection = self.coords.dot(self.proj_array)
        return self._projection

    @property
    def proj_array(self):
        if self._proj_array is None:
            pers_x = -self._camera_ratio[0] / self._camera_ratio[1]
            pers_z = -self._camera_ratio[2] / self._camera_ratio[1]
            self._proj_array = np.array([[1., 0.], [pers_x, pers_z], [0., 1.]])
        return self._proj_array

    @property
    def hole_projections(self):
        if self._hole_projections is None:
            self._hole_projections = [hole.dot(self.proj_array) for hole in self.holes]
        return self._hole_projections

    @property
    def flat_shapely(self) -> Polygon:
        if self._flat_shapely is None:
            self._flat_shapely = Polygon(self.projection, holes = self.hole_projections)
        return self._flat_shapely

    @property
    def normal_vec(self) -> np.array:
        if self._normal_vec is None:
            self._normal_vec = np.cross(*(self.coords[1:3] - self.coords[0]))
        return self._normal_vec

    @property
    def d_const(self) -> float:
        if self._d_const is None:
            self._d_const = -self.normal_vec.dot(self.coords[0])
        return self._d_const

    def order_us(self,other_poly) -> tuple:  # me->other, other->me
        if self.flat_shapely.intersection(other_poly.flat_shapely).area:
            if self.min_max_from_view[1] <= other_poly.min_max_from_view[0]:
                return True, False
            elif self.min_max_from_view[0] >= other_poly.min_max_from_view[1]:
                return False, True
            else:  # no clear ordering based on distance
                signal = self.cut_plane(other_poly.coords)
                me_behind = 1 in signal
                me_infront = -1 in signal
                if me_behind and me_infront:
                    other_signal = other_poly.cut_plane(self.coords)
                    return -1 in other_signal, 1 in other_signal
                return me_behind, me_infront
        return False, False

    def cut_plane(self, other_poly_coords):
        return np.sign(np.round(other_poly_coords.dot(self.normal_vec) + self.d_const, 6)) * self.camera_side

    @property
    def mm_uv(self):
        if self._mm_uv is None:
            minu,minv = self.projection.min(axis=0)
            maxu,maxv = self.projection.max(axis=0)
            self._mm_uv = minu,maxu,minv,maxv
        return self._mm_uv


    @property
    def camera_ratio(self):
        return self._camera_ratio

    @camera_ratio.setter
    def camera_ratio(self, new_ratio):
        self._camera_ratio = new_ratio
        self._min_max_from_view = None
        self._proj_array = None
        self._projection = None
        self._camera_side = None
        self._mm_uv = None
        self._flat_shapely = None
        self._hole_projections = None


    @property
    def min_max_from_view(self):
        if self._min_max_from_view is None:
            pts = self.coords.dot(self._camera_ratio)
            self._min_max_from_view = min(pts), max(pts)
        return self._min_max_from_view

    @property
    def camera_side(self):
        if self._camera_side is None:
            self._camera_side = np.sign(self.camera_ratio.dot(self.normal_vec))
        return self._camera_side

    @property
    def merge_tuple(self):
        return self.color_group,self.orientation


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