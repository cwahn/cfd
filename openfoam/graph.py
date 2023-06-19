from csv import reader, writer
from typing import List
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from more_itertools import consume, side_effect, transpose

from openfoam.optimizer import T, Theta


def to_csv(
    file_path: str, 
    colomns: List[str]):

    if os.path.isfile(file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()
            if len(lines) >= 1 or line[0].strip() != ",".join(colomns).strip() :
                raise Exception(f"File exist: {file_path}")

    line = ",".join(colomns)
    with open(file_path, "w") as f:
        f.write(line + "\n")

    def logger(i:int, ts: List[T], y: float, pds: List[float]):
        t_inners = list(map(lambda t: t.inner, ts))
        # t_inners = ts
        row = t_inners + [y] + pds
        
        with open(file_path, "a") as f:
            wo = writer(f)
            wo.writerow(row)
    
    return logger
            
def plot_csv(
    file_path: str,
    title: str,
):
    lines = []
    with open(file_path, "r") as f:
        lines = list(reader(f))

    header = lines[0]
    rows = lines[1:]
    cols = transpose(rows)

    fig, ax = plt.subplots()

    twins = list(map(lambda _: ax.twinx(), header))

    n = len(header)
    colormap = plt.cm.viridis
    colors = list(map(lambda x: colormap(int(x * colormap.N/n)), range(n)))

    idxs = range(len(rows))

    positions = list(map(lambda i: 1 + i * 0.15, range(n)))
    consume(side_effect(lambda t: t[0].spines.right.set_position(("axes", t[1])), zip(twins, positions)))

    ps = list(map(lambda t: (t[0].plot(idxs, t[1], color=t[2], label=t[3]))[0], zip(twins, cols, colors, header)))

    ax.set_xlabel("iteration")
    consume(side_effect(lambda t: t[0].set_ylabel(t[1]), zip(twins, header)))

    tkw = {
        "size": 4,
        "width": 1.5
    }
    ax.tick_params(axis='x', **tkw)
    consume(side_effect(lambda t: t[0].tick_params(axis="y", colors=t[1].get_color(), **tkw), zip(twins, ps)))

    ax.legend(handles=ps)
    ax.set_title(title)

    def animate(frame_num):
        lines = []
        with open(file_path, "r") as f:
            lines = list(reader(f))

        rows = lines[1:]
        cols = transpose(rows)
        idxs = range(len(rows))

        ps = list(map(lambda t: (t[0].plot(idxs, t[1], color=t[2], label=t[3]))[0], zip(twins, cols, colors, header)))
        return ps
    
    anim = FuncAnimation(fig, animate, interval=20)

    plt.show()