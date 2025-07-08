import matplotlib.pyplot as plt
import numpy as np

from semtex_fieldio import Mesh
from semtex_fieldplot.plot_mesh import plot_mesh_xy, plot_mesh_xy_symm

fig, ax = plt.subplots()

mesh = Mesh.from_file("tests/data/koal1.msh")
plot_mesh_xy(ax, mesh)
plot_mesh_xy_symm(ax, mesh)

x_target = 0
mask, r_target = mesh.axial_node_mask_with_r(x_target, tol=0.25)
ax.plot(
    mesh.xy[..., 0][mask],
    mesh.xy[..., 1][mask],
    color="red",
    ls="",
    marker="o",
    fillstyle="none",
)
ax.plot(np.full_like(r_target, x_target), r_target, color="red", ls="", marker="x")

x_target = 3
mask, r_target = mesh.axial_node_mask_with_r(x_target, tol=0.25)
ax.plot(
    mesh.xy[..., 0][mask],
    mesh.xy[..., 1][mask],
    color="blue",
    ls="",
    marker="o",
    fillstyle="none",
)
ax.plot(np.full_like(r_target, x_target), r_target, color="blue", ls="", marker="x")

ax.set_aspect("equal")
ax.set_title("Mesh in xy-plane")
ax.set_xlim([-1, 5])
plt.xlabel("x")
plt.ylabel("r")

plt.show()
