import matplotlib.pyplot as plt

from semtex_fieldio import Fieldfile, Mesh
from semtex_fieldplot import interpolation

mesh = Mesh.from_file("tests/data/koal1.msh")
ff = Fieldfile("tests/data/koal1.1200.chk", "r")

if ff.geometry != mesh.geometry:
    raise RuntimeError("Geometry mismatch")

x_target = 0
print(f"Plotting axial slice at x = {x_target}")
mask, r_target = mesh.axial_node_mask_with_r(x_target, tol=0.25)
triang = mesh.triangulate_axial_slice(r_target)

data = ff.read_fields_masked(["u"], mask)
interp_data = interpolation.interpolate_to_axial_slice(
    data, mesh, mask, x_target, r_target
).flatten()  # reshape(1, ff.hdr.geometry.nz * len(r_target))

fig, ax = plt.subplots()
plt.tricontourf(triang, interp_data, cmap="plasma", levels=30)
ax.set_aspect("equal")

plt.show()
