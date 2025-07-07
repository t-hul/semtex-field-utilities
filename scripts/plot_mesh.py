import matplotlib.pyplot as plt

from semtex_fieldio import Mesh
from semtex_fieldplot.plot_mesh import plot_mesh_xy, plot_mesh_xy_symm

ax, fig = plt.subfigures()

mesh = Mesh.from_file("tests/data/koal1.msh")
plot_mesh_xy(mesh)
plot_mesh_xy_symm(mesh)

ax.set_aspect("equal")
ax.set_title("Mesh in xy-plane")
plt.xlabel("x")
plt.ylabel("r")
