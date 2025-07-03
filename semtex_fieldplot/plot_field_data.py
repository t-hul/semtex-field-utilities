import matplotlib.pyplot as plt
import numpy as np


def plot_field_contour(ax, mesh, field_zplane, levels=30, cmap="viridis"):
    nel, ns, nr = mesh.geometry.nel, mesh.geometry.ns, mesh.geometry.nr

    # Extract x, y coordinates
    x = mesh.xy[..., 0].flatten()  # shape: (nel*ns*nr,)
    y = mesh.xy[..., 1].flatten()
    z = field_zplane.flatten()

    contour = ax.tricontourf(x, y, z, levels=levels, cmap=cmap)
    ax.set_aspect("equal")
    ax.set_title("Field Contour at z-plane")
    plt.colorbar(contour, ax=ax, label="Field value")
    plt.xlabel("x")
    plt.ylabel("y")
    # plt.grid(True)


def plot_field_contour_dual(
    ax, mesh, field_zplane_1, field_zplane_2, levels=30, cmap="viridis"
):
    nel, ns, nr = mesh.geometry.nel, mesh.geometry.ns, mesh.geometry.nr

    # Extract x, y coordinates
    x = mesh.xy[..., 0].flatten()  # shape: (nel*ns*nr,)
    y = mesh.xy[..., 1].flatten()
    z1 = field_zplane_1.flatten()
    z2 = field_zplane_2.flatten()
    x = np.concatenate([x, x])
    y = np.concatenate([y, -y])
    z = np.concatenate([z1, z2])

    contour = ax.tricontourf(x, y, z, levels=levels, cmap=cmap)
    ax.set_aspect("equal")
    ax.set_title("Field Contour at z-plane")
    plt.colorbar(contour, ax=ax, label="Field value")
    plt.xlabel("x")
    plt.ylabel("y")
    # plt.grid(True)
