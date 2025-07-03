from semtex_fieldio import Mesh
from semtex_fieldplot.plot_mesh import plot_mesh_xy, plot_mesh_xy_symm

mesh = Mesh.from_file("tests/data/koal1.msh")
plot_mesh_xy(mesh)
plot_mesh_xy_symm(mesh)
