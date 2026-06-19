#!/bin/python3
import argparse

import matplotlib.pyplot as plt
import numpy as np
import plot_utils.config as conf

from semtex_fieldio import Mesh
from semtex_fieldplot.plot_mesh import plot_mesh_xy, plot_mesh_xy_symm


def main():
    parser = argparse.ArgumentParser(description="Plot mesh of Semtex .msh file.")
    parser.add_argument("--mesh", type=str, help="Path to Semtex .msh file")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=[5, 3])

    conf.set_font_sizes({"font_size": 32})
    conf.set_plot_style()

    mesh = Mesh.from_file(args.mesh)
    plot_mesh_xy(ax, mesh)

    ax.set_aspect("equal")
    # ax.set_title("Mesh in xy-plane")
    ax.set_xlim([-0.5, 1])
    ax.set_ylim([0, 0.5])
    ax.set_yticks(np.arange(0, 0.6, 0.1))
    plt.xlabel(r"$x$")
    plt.ylabel(r"$r$")

    plt.show()


if __name__ == "__main__":
    main()
