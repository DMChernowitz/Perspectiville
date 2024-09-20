from main import Escherville
from config import bcg_cols
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np

from utils import Dir2Fil, Directions, above



class Painter:
    def __init__(self,Nx: int, Ny: int, building_period: int = 8, building_height: int = 8):
        self.seed = np.random.randint(100000)
        print("seed:", self.seed)
        np.random.seed(self.seed)

        self.villist = [Escherville(Nx=Nx,Ny=Ny,building_period=building_period,building_height=building_height)]
        self.fig, self.ax = plt.subplots(figsize=(20, 10))
        self.keep()
        self.save_count = 0


    def redo(self):
        # retry the last generation step
        self.villist[-1] = deepcopy(self.villist[-2])
        self._task_and_plot()

    def keep(self):
        # accept the last generation step and propose the next
        if self.villist[-1].task_stack:
            self.villist.append(deepcopy(self.villist[-1]))
            self._task_and_plot()

    def _task_and_plot(self):
        plotbool = self.villist[-1].next_task()
        if plotbool:
            self.ax.clear()
            self.villist[-1].project_airspace(self.ax)
            self.fig.show()
        else:
            self._task_and_plot()

    def undo(self):
        self.villist.pop(-1)
        self.redo()

    def recolor(self, default=False):
        self.villist[-1].set_poly_colors(default=default)
        self.villist[-1].plot_ville(self.ax)

    def save(self):
        self.fig.savefig(f"Escherville_{self.seed}_{self.save_count}.png",dpi=400, bbox_inches='tight')
        self.save_count += 1

    def complete(self):
        self.villist[-1].merged_polys = []
        while self.villist[-1].task_stack:
            self.villist[-1].next_task()
        self.villist[-1].project_airspace(self.ax)
        self.fig.show()

    def bcg_color(self):
        self.villist[-1].poly_colors = bcg_cols
        self.villist[-1].plot_ville(self.ax)

    def set_pointer(self,x,y,z,code):
        self.villist.append(deepcopy(self.villist[-1]))
        self.villist[-1].pointer = (x,y,z)
        fill = getattr(Dir2Fil, code)
        self.villist[-1].airspace[self.villist[-1].pointer] = fill
        self.ax.clear()
        self.villist[-1].project_airspace(self.ax)
        self.fig.show()
        if "u" in code:
            self.villist[-1].pointer = above(self.villist[-1].pointer)


    def step(self, code: str, repeat=1):
        self.villist.append(deepcopy(self.villist[-1]))
        for _ in range(repeat):
            self._pave(code)
        self.ax.clear()
        self.villist[-1].project_airspace(self.ax)
        self.fig.show()


    def _pave(self,code):

        self.villist[-1].pointer = tuple(getattr(Directions,code)+self.villist[-1].pointer)

        fill = getattr(Dir2Fil, code)

        self.villist[-1].airspace[self.villist[-1].pointer] = fill

        # if fill == 3:
        #     _possible_starts.append(pos)
        if fill in [5, 6, 7, 8]:  # stairs
            # don't put stuff directly above stairs
            self.villist[-1].airspace[above(self.villist[-1].pointer)] = 9

        if "u" in code:
            self.villist[-1].pointer = above(self.villist[-1].pointer)

    def anchor_target(self):
        hyperdiag = self.villist[-1].get_hyperdiag(self.villist[-1].pointer)
        for hd in hyperdiag:
            if self.villist[-1].airspace[hd] in [1,10]:
                self.villist[-1].airspace[hd] = 4
                break

    def undo_step(self):
        self.villist.pop(-1)
        self.ax.clear()
        self.villist[-1].project_airspace(self.ax)
        self.fig.show()