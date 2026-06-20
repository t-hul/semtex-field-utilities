from semtex_fieldio.mesh import Mesh
from semtex_fieldio.fieldfile import Fieldfile
from pathlib import Path


mesh_path = Path("../../tests/data/koal1.msh")
mesh = Mesh.from_file(mesh_path)

data_path = Path("../../tests/data/koal1.1200.chk")
fieldfile = Fieldfile(data_path, "r")
fieldfile.read_all_data()

vtk_file = semtex_to_vtk_unstructured_grid(mesh, fieldfile)

write_vtu(vtk_file, "../../tests/tmp/koal1.1200.vtu")
