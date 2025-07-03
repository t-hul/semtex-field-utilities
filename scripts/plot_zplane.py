import matplotlib.pyplot as plt

from semtex_fieldio import Fieldfile, Mesh
from semtex_fieldplot.plot_field_data import plot_field_contour_dual
from semtex_fieldplot.plot_mesh import plot_mesh_xy_symm

mesh = Mesh.from_file("tests/data/koal1.msh")
ff = Fieldfile("tests/data/koal1.1200.chk", "r")

if ff.geometry != mesh.geometry:
    raise RuntimeError("Geometry mismatch")

z_idx_1 = 0
z_idx_2 = (z_idx_1 + ff.geometry.nz // 2) % ff.geometry.nz
print(f"Plotting planes {z_idx_1} and {z_idx_2}")
data1 = ff.read_zplane(z_idx=z_idx_1, field_names=["u"])
data2 = ff.read_zplane(z_idx=z_idx_2, field_names=["u"])

fig, ax = plt.subplots()
plot_field_contour_dual(ax, mesh, data1[0], data2[0])
plot_mesh_xy_symm(ax, mesh, only_elements=True)

plt.show()
