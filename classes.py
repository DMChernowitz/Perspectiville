import numpy as np
from enum import Enum

class ColorGroups(Enum):
    path = 0
    window = 1
    roof = 2
    background = 3
    building_1 = 4
    building_2 = 5
    building_3 = 6


from shapely.geometry import Polygon
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

    def scoot(self, dvector: np.array):
        """
        Move the polygons 3d coordinates and its holes
        :param dvector: displacement vector
        :return: none
        """
        if dvector.shape == (3,):
            dvector = dvector[np.newaxis,:]
        elif dvector.shape != (1,3):
            raise ValueError("Incorrect dimension")
        self.coords += dvector
        for _hole in self.holes:
            _hole += dvector

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
                    if not hasattr(_partpoly,"exterior"):
                        continue
                    _co = np.insert(_partpoly.exterior.coords.xy,1,0,0).T[:-1]
                    if sum(self.distance_to_plane(_co)) < sum(other_poly.distance_to_plane(_co)): #other is closer to camera
                        original = original.difference(_partpoly)
                    if list(_partpoly.interiors):
                        print("missed an interior")
                    if original.area == 0:
                        print("reduced to nothing")
                        return []
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

