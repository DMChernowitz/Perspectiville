# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
from math import gcd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import PatchCollection


import itertools
import copy

import networkx as nx
from ortools.sat.python import cp_model

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from utils import (
    join_polygons,
    choose_next_weighted,
    above,
    transition_directions,
    dir_to_fill,
    cube_blocking,
    divisors,
    scan_wall,
    patch_polygon,
    get_y_blocks,
    rel_choice
)
from templates import cube_filling_building, cube_filling_free, get_window, get_roof_template, get_single_chimney
from classes import Poly

from config import unit_cell_paths, get_poly_colors, roof_style_probs, n_colors


seed = np.random.randint(100000)

print("seed:", seed)

np.random.seed(seed)


#all sizes are in 100 units

class Escherville:
    def __init__(self,Nx: int, Ny: int, building_period: int = 4, building_height: int = 4, camera_ratio: np.array = np.array([1,-2,1])):
        self.Nx = Nx
        self.Ny = Ny
        self.building_period = building_period
        self.building_height = building_height
        self.Lx = Nx * building_period
        self.Ly = Ny * building_period
        self.Lz = building_height+3  # maximum build height
        # create array to hold state of the world.
        self.airspace = np.zeros((self.Lx + 2, self.Ly + 2, self.Lz + 2)) + 9
        self.airspace[1:-1, 1:-1, 1:-1] = 0
        self.hbp = building_period // 2
        for dx in range(self.hbp):
            for dy in range(self.hbp):
                self.airspace[self.hbp // 2 + 1 + dx::building_period, self.hbp // 2 + 1 + dy::building_period, building_height] = 1
                self.airspace[self.hbp // 2 + 1 + dx::building_period, self.hbp // 2 + 1 + dy::building_period, 1:building_height] = 2
        self.x_building = self.get_building_locs(self.Lx)
        self.y_building = self.get_building_locs(self.Ly)

        self.camera_ratio = camera_ratio

        self.merged_polys = [] #have been joined
        self.unjoined_polys = [] #need to be joined
        self.finished_polys = [] #don't need to be joined

        self.polygraph = nx.DiGraph()

        #to store the blocks that make up each building at the end
        self.windows_hash = {n: {"side":[], "front":[]} for n in range(self.Nx*self.Ny)}
        self.little_windows_hash = copy.deepcopy(self.windows_hash)
        #first label: building nr, second, roof loc, points to possible build height.
        self.roof_hash = {n: {} for n in range(self.Nx*self.Ny)}

        self.genz = self.get_path_sources_targets(unit_cell_paths)

        # (.color_group,.orientation) format
        self.group_colors = [(cg,ori) for cg in [0]+list(range(4,4+n_colors)) for ori in ["front","side","top"]]

        #how often to try to plot a path
        self.n_path_attempts = 10000

        self.building_colors = None
        self.task_stack = []
        self.freeze_joined = False

    def queue_poly(self,poly):
        if poly.merges:
            self.unjoined_polys.append(poly)
        else:
            self.finished_polys.append(poly)

    def get_task_stack(self):
        #each task has a name (method name), args to input to the method, and a bool, whether to plot after.

        self.task_stack.append({"task": "set_building_colors", "args": [], "plot": True})


        for path_nr in range(len(self.genz)):
            self.task_stack.append({"task": "add_one_escher_path", "args": [path_nr], "plot": True})

        self.task_stack.append({"task": "scope_roofs", "args": [], "plot": False})

        for cluster_nr in range(self.Nx*self.Ny):
            self.task_stack.append({"task": "roofer", "args": [cluster_nr], "plot": True})

        #from this step, can keep the joined_polys. i.e. add to scope_walls a flag: freeze joined..
        self.task_stack.append({"task": "scope_walls", "args": [], "plot": False})

        for cluster_nr in range(self.Nx*self.Ny):
            self.task_stack.append({"task": "stretch_windows", "args": [cluster_nr], "plot": True})

        self.task_stack.append({"task": "set_poly_colors", "args": [], "plot": True})


    def set_building_colors(self):
        self.building_colors = np.random.randint(4, n_colors + 4, self.Nx*self.Ny)

    def get_path_sources_targets(self,unit_cell):
        ux = len(unit_cell)
        if ux:
            uy = len(unit_cell[0])
        else:
            return []

        genz = []

        for bx in range(0,self.Nx,ux):
            for by in range(0,self.Ny,uy):
                for hx in range(ux):
                    sx = bx + hx  # source building address
                    if sx < self.Nx:
                        for hy in range(uy):
                            sy = by+hy  # source building address
                            if sy < self.Ny:
                                for dxy in unit_cell[hx][hy]:
                                    tx = sx + dxy[0]  # target building address
                                    if tx >= 0 and tx<self.Nx:
                                        ty = sy + dxy[1]  # target building address
                                        if ty >= 0 and ty < self.Ny:
                                            genz.append(self.target_source_gen(sx,sy,tx,ty))
                            else:
                                break
                    else:
                        break

        return genz

    def target_source_gen(self,sx,sy,tx,ty):
        can_start_path = [1,3,4]
        dixy = self.hbp // 2 + 1+self.building_period*np.array([sx,sy,tx,ty])
        while True:
            sr = np.random.randint(0,self.hbp,2) + dixy[:2]
            source_indices = (sr[0], sr[1], self.building_height)
            if self.airspace[source_indices] in can_start_path:
                tr = np.random.randint(0,self.hbp,2) + dixy[2:]
                target_indices = (tr[0], tr[1], self.building_height)
                if self.airspace[target_indices] in can_start_path:
                    yield source_indices,target_indices

    def get_building_locs(self,L):
        return sorted(sum([list(range(self.hbp // 2 + d + 1,L,self.building_period)) for d in range(self.hbp)], []))

    def get_cluster(self,ix,iy):
        return ((ix-1)//self.building_period)*self.Ny+(iy-1)//self.building_period

    def scope_roofs(self):
        for ix, iy, iz in self.all_xyz():
            pos = (ix,iy,iz)
            cur_cluster = self.get_cluster(ix, iy)
            if self.airspace[pos] in [1,10]: #building roof:
                for dz in range(1,self.Lz-iz+1):
                    if self.airspace[ix,iy,iz+dz] not in [0,9]:
                        break
                self.roof_hash[cur_cluster][pos] = dz
            if self.airspace[pos] in [4]: #target #could add 13 to free up start of path here
                self.roof_hash[cur_cluster][pos] = -1

    def roofer(self, cluster_nr):
        #choose roof styles
        full_roof = self.hbp * self.hbp

        if sum([v > 0 for v in self.roof_hash[cluster_nr].values()]) == full_roof:
            #don't allow castle
            roof_strat = rel_choice({i:roof_style_probs[i] for i in roof_style_probs if i!='castle'})
            max_height = min(self.roof_hash[cluster_nr].values())
            # add extra windows, or chimnies. Different from other, more detail
            roofing_areas = [{
                "displace": np.array(list(self.roof_hash[cluster_nr])).min(axis=0),
                "droof": np.array([[self.hbp, self.hbp, np.random.randint(1, 1 + max_height)]])
            }]

            m = np.random.randint(1, 6)
            rh = np.random.randint(1, min(roofing_areas[0]["droof"][0][2], self.hbp) + 1)
            roofing_areas[0]["droof"][0][2] = 2 * rh
        else:
            roof_strat = rel_choice(roof_style_probs)
            jdir = 0 if roof_strat == "slant_front" else 1
            roofing_areas, extra_blocks, new_flats = get_y_blocks(self.roof_hash[cluster_nr], 1-jdir)
            for loc in new_flats:
                self.airspace[loc] = 1
            for loc in extra_blocks:
                self.airspace[loc] = 2

            m = 0
            max_slant = min([rf["droof"][0][2] / rf["droof"][0][jdir] for rf in roofing_areas] + [10])
            for rf in roofing_areas:
                rf["droof"][0][2] = round(max(1, 2 * rf["droof"][0][jdir] * max_slant))


        aircode = 14 if roof_strat == "castle" else 12
        for rf in roofing_areas:
            for xb in range(rf["displace"][0], rf["displace"][0] + rf["droof"][0][0]):
                for yb in range(rf["displace"][1], rf["displace"][1] + rf["droof"][0][1]):
                    self.airspace[xb, yb, rf["displace"][2]] = aircode
            if roof_strat in ["slant_front", "slope_side"]:
                for _roof_poly in get_roof_template(roof_strat, m=m, droof=rf["droof"]):
                    _roof_poly.coords += (dv := 100 * rf["displace"][np.newaxis, :])
                    for _h in _roof_poly.holes:
                        _h += dv
                    if _roof_poly.color_group == 4:
                        _roof_poly.color_group = self.building_colors[cluster_nr]
                    #self.polygons.append(_roof_poly)
                    self.queue_poly(_roof_poly)

        self.add_chimneys_and_housies(cluster_nr)

    def add_chimneys_and_housies(self, cluster_nr):

        def get_check(_kl):
            hd = self.get_hyperdiag(_kl)
            bla = all([self.airspace[hd[jk]] in [0,1,9,10] for jk in range(hd.index(_kl) + 1, len(hd))])
            return bla

        for kl,rh in self.roof_hash[cluster_nr].items():
            if rh > 0 and self.airspace[kl] in [1, 10]:
                ix,iy,iz = kl
                if all([get_check(_kl) for _kl in [kl,(ix+1,iy,iz),(ix,iy-1,iz),(ix,iy+1,iz-1),(ix,iy,iz-1)]]):
                    if np.random.randint(self.building_period) == 0:
                        loc = np.array([self.get_hyperdiag(kl)[-1]])
                        for _poly in get_single_chimney():
                            _poly.coords += 100*loc
                            if _poly.color_group == 4:
                                _poly.color_group = self.building_colors[cluster_nr]
                            #self.polygons.append(_poly)
                            self.queue_poly(_poly)

    def scope_walls(self):
        if self.freeze_joined:
            return None

        ly = self.hbp//2 + 1
        lx = ly+self.hbp-1
        for ix, iy, iz in self.all_xyz():
            pos = (ix,iy,iz)
            cur_cluster = self.get_cluster(ix, iy)
            if self.airspace[pos] in [2,11]: #building body
                if self.airspace[ix,iy-1,iz] in [0,1,3,4,7,9,10,13,14]:
                    if iy % self.building_period == ly: #original walls
                        self.windows_hash[cur_cluster]["front"].append(pos)
                    else:  # towers inside
                        self.little_windows_hash[cur_cluster]["front"].append(pos)
                if self.airspace[ix+1,iy,iz] in [0,1,3,4,6,9,10,13,14]:
                    if ix % self.building_period == lx: #visible side
                        self.windows_hash[cur_cluster]["side"].append(pos)
                    else:
                        self.little_windows_hash[cur_cluster]["side"].append(pos)

        #don't need to redo this, airspace won't change anymore
        self.freeze_joined = True


    def gcd_walls(self,cluster_number):

        #horizontal gcd:
        h_gcd = scan_wall(wall=self.windows_hash[cluster_number]["front"], j=0, k=2)
        if h_gcd != 1:
            h_gcd = gcd(h_gcd,scan_wall(wall=self.windows_hash[cluster_number]["side"], j=1, k=2))

        #vertical gcd:
        v_gcd = scan_wall(wall=self.windows_hash[cluster_number]["front"], j=2, k=0)
        if v_gcd != 1:
            v_gcd = gcd(v_gcd,scan_wall(wall=self.windows_hash[cluster_number]["side"], j=2, k=1))

        return h_gcd,v_gcd

    def stretch_windows(self,cluster_number):

        h_gcd, v_gcd = self.gcd_walls(cluster_number)

        h_period = np.random.choice(divisors(h_gcd))
        v_period = np.random.choice(divisors(v_gcd))

        window_template = get_window(*list(np.random.randint(0,3,4)))

        #window template is list of windows on 100*100 total u,v plane
        #list of np.arrays
        scale_vec = np.array([h_period,v_period])
        window_tile = [w*scale_vec[np.newaxis,:] for w in window_template]

        windows_list = []

        #front:
        front_array = np.array(self.windows_hash[cluster_number]["front"])
        xmin,y,zmin = front_array.min(axis=0)
        xmax,y,zmax = front_array.max(axis=0)
        for x_tile in range(xmin,xmax+1,h_period):
            for z_tile in range(zmin,zmax+1,v_period):
                if (x_tile,y,z_tile) in self.windows_hash[cluster_number]["front"]:
                    displace = 100*np.array([[x_tile,y,z_tile]])+self.camera_ratio
                    for win in window_tile:
                        windows_list.append(
                            Poly(
                                coords=np.insert(win,1,0,axis=1)+displace,
                                color_group=1,
                                orientation="front"
                            )
                        )


        #side:
        side_array = np.array(self.windows_hash[cluster_number]["side"])
        x,ymin,zmin = side_array.min(axis=0)
        x,ymax,zmax = side_array.max(axis=0)
        for y_tile in range(ymin,ymax+1,h_period):
            for z_tile in range(zmin,zmax+1,v_period):
                if (x,y_tile,z_tile) in self.windows_hash[cluster_number]["side"]:
                    displace = 100*np.array([[x+1,y_tile,z_tile]])+self.camera_ratio
                    for win in window_tile:
                        windows_list.append(
                            Poly(
                                coords=np.insert(win,0,0,axis=1)+displace,
                                color_group=1,
                                orientation="side"
                            )
                        )

        # little ones:
        for side,jj in [("front",1),("side",0)]:
            for _x,_y,_z in self.little_windows_hash[cluster_number][side]:
                displace = 100 * np.array([[_x+1-jj,_y,_z]]) + self.camera_ratio
                for win in window_template:
                    windows_list.append(
                        Poly(
                            coords=np.insert(win, jj, 0, axis=1) + displace,
                            color_group=1,
                            orientation=side
                        )
                    )

        self.finished_polys.extend(windows_list)

    def pathfinder(self):
        # can project airspace after any of these steps.
        self.set_building_colors()

        for path_nr in range(len(self.genz)):
            self.add_one_escher_path(path_nr)

        self.scope_roofs()  # no point projecting after this step

        for cluster_nr in range(self.Nx*self.Ny):
            self.roofer(cluster_nr)

        self.scope_walls()  # no point projecting after this step

        for cluster_nr in range(self.Nx*self.Ny):
            self.stretch_windows(cluster_nr)

        self.set_poly_colors()

        self.project_airspace(cabinet=True)

    def add_one_escher_path(self, path_number):
        st_gen = self.genz[path_number]
        for _ in range(self.n_path_attempts):
            _airspace = copy.deepcopy(self.airspace)

            pos, target = next(st_gen)

            target_diag = [_ta for _ta in self.get_hyperdiag(target) if _ta[2] >= self.building_height]

            _airspace[pos] = 13
            _airspace[target] = 4

            state = "s"  # no direction yet
            pathlen = 0
            while pathlen < 5 * self.building_period and not (pos in target_diag and state in ["r", "l", "b", "f"]):
                # possible positions
                pos_pos_data = [([tuple(pos + _d) for _d in _dir], _state) for _state, _dir in
                                transition_directions[state].items()]

                # filter for positions that can still be built into
                pos_pos_data = [p for p in pos_pos_data if all([_airspace[_p] < 3 for _p in p[0]])]

                if len(pos_pos_data) == 0:
                    break

                # choose next step
                new_pos_data = choose_next_weighted(state, pos_pos_data)
                # fill in master array cell
                fill = dir_to_fill[new_pos_data[1]]

                pos = new_pos_data[0][0]

                _airspace[pos] = fill

                # if fill == 3:
                #     _possible_starts.append(pos)
                if fill in [5, 6, 7, 8]:  # stairs
                    # don't put stuff directly above stairs
                    _airspace[above(pos)] = 9

                # prevent too many crossings in one line of sight
                hyperdiag = self.get_hyperdiag(pos)
                if sum([_airspace[dp] in [3, 5, 6, 7, 8, 13] for dp in hyperdiag]) == 2:
                    for dp in hyperdiag:
                        _airspace[dp] = cube_blocking[_airspace[dp]]

                state = new_pos_data[1]

                if "u" in state:
                    pos = above(pos)

                pathlen += 1

            if (pos in target_diag and state in ["r", "l", "b", "f"]):
                self.airspace = _airspace
                break

    def get_hyperdiag(self, intersection):
        # line of sight is along +x,-2y,+z

        arrshape = self.airspace.shape

        lowers, uppers = [], []

        for j in [0, 1, 2]:
            bdys = sorted([-intersection[j] / self.camera_ratio[j], (arrshape[j] - intersection[j] - 1) / self.camera_ratio[j]])
            lowers.append(bdys[0])
            uppers.append(bdys[1])

        return [tuple(intersection + k * self.camera_ratio) for k in range(int(max(lowers)), int(min(uppers)) + 1)]

    def set_poly_colors(self):
        self.poly_colors = get_poly_colors()

    def project_airspace(self,cabinet=True):

        self.trim_polys()

        #can put them in joined_polys to get a line between roof and building
        #self.polygons.extend(self.roof_items)

        self.join_these_polys()

        #have self.pre-join polys
        #have polys from trim_polys as local var
        #have self.post-join polys
        self.joined_polys = self.finished_polys + self.merged_polys


        fig, ax = plt.subplots(figsize=(20, 10))

        if cabinet:

            self.topoplot(ax)


        else:

            ax = Axes3D(fig, auto_add_to_figure=False)
            ax.set_xlim3d(0, 100*self.Lx)
            ax.set_ylim3d(0, 100*self.Ly)
            ax.set_zlim3d(0, 100*self.Lz)
            ax.set_proj_type("ortho")
            fig.add_axes(ax)
            for _poly in self.joined_polys:
                c = self.poly_colors[_poly.merge_tuple]
                pol = Poly3DCollection([_poly.coords])
                # linen = Line3DCollection(_poly.edges)
                # linen.set_color("k")
                pol.set_color(mpl.colors.rgb2hex(c))
                pol.set_edgecolor("k")
                ax.add_collection3d(pol)
                # ax.add_collection3d(linen)

        plt.savefig(f"Escherville_{seed}.png",dpi=300)
        plt.show()

    def topoplot(self, ax):

        self.construct_graph()  # fills in self.polygraph

        to_break = []
        for sg in nx.strongly_connected_components(self.polygraph):
            if len(sg) > 1:
                subgraph = self.polygraph.subgraph(sg)
                model = cp_model.CpModel()
                cancut = {edge: model.NewBoolVar(str(edge)) for edge in subgraph.edges}

                # don't need this if we _can_ have polys with holes in them.
                # for edge in subgraph.edges:
                #     if self.joined_polys[edge[0]].flat_shapely.contains(self.joined_polys[edge[1]].flat_shapely):
                #         model.Add(cancut[edge] == 0)  # can't plot nontrivial topologies

                for cycle in nx.simple_cycles(subgraph):
                    model.Add(sum([cancut[(cycle[j - 1], cycle[j])] for j in range(len(cycle))]) > 0)
                model.Minimize(sum([cancut[k] for k in cancut]))
                solver = cp_model.CpSolver()
                solver.Solve(model)
                to_break.extend([edge for edge in subgraph.edges if solver.Value(cancut[edge])])
        # print([
        #     len(list(nx.simple_cycles(self.polygraph.subgraph(c))))
        #     for c in sorted(nx.strongly_connected_components(self.polygraph), key=len, reverse=True)
        # ])
        print("break:", to_break)

        self.polygraph.remove_edges_from(to_break)
        break_dict = {}
        for edge in to_break:
            break_dict.setdefault(edge[0], []).append(edge[1])
        patches = []
        facecolors = []

        for ind in nx.topological_sort(self.polygraph):
            _poly = self.joined_polys[ind]

            if ind in break_dict:
                flat_polys = _poly.subtract([self.joined_polys[k] for k in break_dict[ind]])
            else:
                flat_polys = [_poly.flat_shapely]

            c = self.poly_colors[_poly.merge_tuple]

            patches.extend([patch_polygon(fs) for fs in flat_polys])
            facecolors.extend([c] * len(flat_polys))

            # for fp in flat_polys:
            #     x, y = fp.T
            #     plt.fill(x, y, facecolor=c, edgecolor="k", zorder=1, lw=1)
        pa = PatchCollection(patches, facecolors=facecolors,
                             edgecolors="k", lw=1, joinstyle="round", capstyle="round")
        ax.add_collection(pa)
        ax.autoscale_view()

    def all_xyz(self):
        return itertools.product(range(1, self.Lx + 1), range(1, self.Ly + 1), range(1, self.Lz + 1))

    def join_these_polys(self):

        if self.freeze_joined and self.merged_polys:
            return None
        self.merged_polys = []

        orientation_fixed_dim = {
            "top": 2,
            "side": 0,
            "front": 1,
        }
        for merge_tuple in self.group_colors:
            d = orientation_fixed_dim[merge_tuple[1]]
            cardinal_polys = sorted([p for p in self.unjoined_polys if p.merge_tuple == merge_tuple],
                                    key=lambda x: x.coords[0, d])
            if not cardinal_polys:
                continue
            plane_coord = cardinal_polys[0].coords[0, d]
            this_slice = []
            for _poly in cardinal_polys:
                if _poly.coords[0, d] != plane_coord:
                    self.merged_polys.extend(join_polygons(this_slice))
                    this_slice = []
                    plane_coord = _poly.coords[0, d]
                this_slice.append(_poly)
            self.merged_polys.extend(join_polygons(this_slice))
        #self.joined_polys.extend([p for p in self.polygons if not p.merges])

    def construct_graph(self):
        #sort by left bdy of bounding box
        self.joined_polys.sort(key= lambda x: x.mm_uv[0])

        self.polygraph.add_nodes_from(range(len(self.joined_polys)))

        minmax_uv = np.array([p.mm_uv for p in self.joined_polys])
        #mmuv = min u, max u, min v max v
        max_du,max_dv = (minmax_uv[:,1::2]-minmax_uv[:,::2]).max(axis=0)
        # construct graph edges

        for j in range(len(self.joined_polys)):
            for k in range(j + 1, len(self.joined_polys)):
                j_mmuv = self.joined_polys[j].mm_uv
                k_mmuv = self.joined_polys[k].mm_uv
                if j_mmuv[1] > k_mmuv[0] and j_mmuv[0] < k_mmuv[1] and j_mmuv[3] > k_mmuv[2] and j_mmuv[2] < k_mmuv[3]:
                    self.add_joined_polys_to_graph(j, k)
                if j_mmuv[0] > k_mmuv[0] + max_du:
                    break

    def add_joined_polys_to_graph(self, j, k):
        back, forth = self.joined_polys[j].order_us(self.joined_polys[k])  # me->other, other->me
        if back:
            self.polygraph.add_edge(j, k)
        if forth:
            self.polygraph.add_edge(k, j)

    def trim_polys(self):

        for ix, iy, iz in self.all_xyz():
            # put all this logic into one function.
            cur_fill_code = self.airspace[(ix, iy, iz)]

            cluster_nr = self.get_cluster(ix,iy)
            color_nr = self.building_colors[cluster_nr]

            if self.airspace[(ix, iy, iz - 1)] in [2, 11]:  # inside building
                cur_fill = cube_filling_building[cur_fill_code]
            else:
                cur_fill = cube_filling_free[cur_fill_code]


            #remove hidden faces
            if cur_fill_code in [3,13]: #flat path and source flat path
                cut_cur_fill = [cur_fill[0]]
                cut_cur_fill.extend(self.get_fences(cur_fill[1:], ix, iy, iz, code=cur_fill_code))
                cur_fill = cut_cur_fill
            elif cur_fill_code in [2, 11]:
                cut_cur_fill = []
                if self.airspace[(ix, iy - 1, iz)] not in [2, 11]:
                    cut_cur_fill.append(cur_fill[0])
                if self.airspace[(ix + 1, iy, iz)] not in [2, 11]:
                    cut_cur_fill.append(cur_fill[1])
                cur_fill = cut_cur_fill
            elif cur_fill_code in [1, 10, 12, 14]:  # roof, no building below
                cut_cur_fill = []
                if self.airspace[(ix, iy, iz - 1)] not in [2, 11]:
                    if not (self.airspace[(ix, iy - 1, iz - 1)] in [2, 11] or
                            self.airspace[(ix, iy - 1, iz)] in [1, 4, 10]):
                        cut_cur_fill.append(cur_fill[0])
                    if not (self.airspace[(ix + 1, iy, iz - 1)] in [2, 11] or
                            self.airspace[(ix + 1, iy, iz)] in [1, 4, 10]):
                        cut_cur_fill.append(cur_fill[1])
                if cur_fill_code in [1,10,14]:
                    cut_cur_fill.append(cur_fill[2])
                if cur_fill_code == 14:
                    cut_cur_fill.extend(self.get_fences(cur_fill[3:], ix, iy, iz, code=14))
                cur_fill = cut_cur_fill

            #modify and finalize the polygon
            displacement = np.array([ix, iy, iz])*100
            for _poly in cur_fill:
                new_poly = copy.deepcopy(_poly)
                if new_poly.color_group == 4: #recolor buildings
                    new_poly.color_group = color_nr
                new_poly.coords += displacement
                #self.polygons.append(new_poly)
                self.queue_poly(new_poly)

    def get_fences(self, cur_fill, ix, iy, iz, code):
        cut_cur_fill = []
        extra = [14] if code in [3,13] else [5,6,7,8] #might have to remove these four...
        fence_off = [0, 1, 9, 10] + extra
        if self.airspace[(ix, iy - 1, iz)] in fence_off and self.airspace[(ix, iy - 1, iz - 1)] != 7:
            cut_cur_fill.append(cur_fill[0])
        if self.airspace[(ix, iy + 1, iz)] in fence_off and self.airspace[(ix, iy + 1, iz - 1)] != 8:
            cut_cur_fill.append(cur_fill[1])
        if self.airspace[(ix + 1, iy, iz)] in fence_off and self.airspace[(ix + 1, iy, iz - 1)] != 6:
            cut_cur_fill.append(cur_fill[2])
        if self.airspace[(ix - 1, iy, iz)] in fence_off and self.airspace[(ix - 1, iy, iz - 1)] != 5:
            cut_cur_fill.append(cur_fill[3])
        return cut_cur_fill


# pers_x = -camera_ratio[0]/camera_ratio[1]
# pers_z = -camera_ratio[2]/camera_ratio[1]

# def old_custom_zorder(poly1, poly2):
#     tolerance = 1e-6
#     min1, max1 = poly1.min(axis=0), poly1.max(axis=0)
#     min2, max2 = poly2.min(axis=0), poly2.max(axis=0)
#     dim_order = [int(min1[j]+tolerance > max2[j]) - int(max1[j] < min2[j] + tolerance) for j in [0,1,2]]
#     dim_order[1] *= -1 #y-coord is reversed
#     if sum([a == 0 for a in dim_order]) > 1:
#         return sum(dim_order)
#     else:
#         #use the other trick with planes.
#         d1 = poly1.dot(camera_ratio)
#         d2 = poly2.dot(camera_ratio)
#         dd1 = (max(d1),min(d1))
#         dd2 = (max(d2),min(d2))
#         if dd1 == dd2:
#             return 0
#         else:
#             return 2*(dd1 > dd2) -1

# def customer_zorder(coords,other_coords):
#
#     margin = 6 #significant digits, if points are this close to the plane, consider them on the plane.
#
#     uv = None
#
#     if uv is None:
#         uv = coords[1:3] - coords[0] + 0.
#
#         sign_camera = np.sign(np.linalg.det(np.insert(uv, 2, camera_ratio, 0)))
#         uv[0] *= sign_camera
#
#         uv = uv[np.newaxis, :, :]
#
#     sign_distances = np.sign(np.round(np.linalg.det(
#         np.insert(uv * np.ones((len(other_coords), 1, 1)), 2, other_coords - coords[0], 1)
#     ), margin))
#
#     if -1 in sign_distances and 1 in sign_distances:
#         return 0  # points on both sides of the plane
#     else:
#         return -sum(sign_distances)



# def get_hyperdiag(arrshape,intersection):
#     #line of sight is along +x,-2y,+z
#
#     lowers,uppers = [],[]
#
#     for j in [0,1,2]:
#         bdys = sorted([-intersection[j]/camera_ratio[j],(arrshape[j]-intersection[j]-1)/camera_ratio[j]])
#         lowers.append(bdys[0])
#         uppers.append(bdys[1])
#
#     return [tuple(intersection+k*camera_ratio) for k in range(int(max(lowers)),int(min(uppers))+1)]

# from shapely.geometry import Polygon
#
# holy = Polygon([[0,4],[4,4],[4,0],[0,0]][::-1],holes=[[[2,2],[2,3],[3,3],[3,2]]])
# moly = Polygon([[10,4],[14,4],[14,0],[10,0]])
# zoly = Polygon([[1,2.5],[2.5,2.5],[2.5,1],[1,1]])

mce = Escherville(3,4, building_period=8, building_height=8)
mce.pathfinder()






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
