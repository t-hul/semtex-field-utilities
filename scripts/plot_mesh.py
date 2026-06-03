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
    # plot_mesh_xy_symm(ax, mesh)

    # x_target = 0
    # mask, r_target = mesh.axial_node_mask_with_r(x_target, tol=0.25)
    # ax.plot(
    #     mesh.xy[..., 0][mask],
    #     mesh.xy[..., 1][mask],
    #     color="red",
    #     ls="",
    #     marker="o",
    #     fillstyle="none",
    # )
    # ax.plot(np.full_like(r_target, x_target), r_target, color="red", ls="", marker="x")
    #
    # x_target = 3
    # mask, r_target = mesh.axial_node_mask_with_r(x_target, tol=0.25)
    # ax.plot(
    #     mesh.xy[..., 0][mask],
    #     mesh.xy[..., 1][mask],
    #     color="blue",
    #     ls="",
    #     marker="o",
    #     fillstyle="none",
    # )
    # ax.plot(np.full_like(r_target, x_target), r_target, color="blue", ls="", marker="x")

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
