Import painter from jupyter_classes

init Painter(N,M, building_period, building_heigt) for the grid of buildings
N: number of rows
M: number of columns
building_period: number of cells in x and y direction before building repeats
building_heigt: number of cells in z direction of the building to touch the sky

use -Painter.keep() to keep the current grid with its base colorings,
moves on to adding paths, later to window styles
use -Painter.redo() to retry the last step
use -Painter.undo() to undo the last step and go back one
use -Painter.complete() to automatically move to the final step
Then -Painter.recolor() will cycle through the building color presets.

Can also build your own path with the
- Painter.set_pointer(x,y,z, code) to set the pointer to a specific location
    code chooses the type of path to be drawn. "l", "r", "f", "b" for left, right, forward, back
    "lu", "ru", "fu", "bu" for left up, right up, forward up, back up
    "ld", "rd", "fd", "bd" for left down, right down, forward down, back down (I think)
- Painter.step(code): from current pointer location, move one step in the direction of the pointer
- Can also change the targets of the paths (special begin and end points), on which hyperdiagonal.
- Painter.undo_step() to undo the last step