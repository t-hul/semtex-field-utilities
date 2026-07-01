import numpy as np
from paraview.util.vtkAlgorithm import smproxy, smproperty, smdomain
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.vtkCommonDataModel import vtkMultiBlockDataSet, vtkStructuredGrid
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.util.numpy_support import numpy_to_vtk
from pathlib import Path


from semtex_fieldio.mesh import Mesh
from semtex_fieldio.fieldfile import Fieldfile
from semtex_fieldio.vtk_export_helpers import mesh_to_block_arrays, field_to_block_point_data


@smproxy.reader(
    name="SemtexFieldReader",
    label="Semtex Field Reader",
    extensions=["fld", "chk"],
    file_description="Semtex field files",
)
class SemtexFieldReader(VTKPythonAlgorithmBase):
    def __init__(self):
        super().__init__(
            nInputPorts=0,
            nOutputPorts=1,
            outputType="vtkMultiBlockDataSet",
        )
        self._filename = None
        self._mesh_filename = None
        self._wrap_z = True

    @smproperty.stringvector(name="FileName")
    @smdomain.filelist()
    def SetFileName(self, filename):
        self._filename = filename
        self.Modified()

    @smproperty.intvector(name="WrapZ", default_values=1)
    @smdomain.xml("""
                  <BooleanDomain name="bool" />
                  """)
    def SetWrapZ(self, value):
        self._wrap_z = bool(value)
        self.Modified()

    # @smproperty.stringvector(name="MeshFileName")
    # @smdomain.filelist()
    # def SetMeshFileName(self, filename):
    #     self._mesh_filename = filename
    #     self.Modified()
    #

    def RequestData(self, request, inInfo, outInfo):
        output = vtkMultiBlockDataSet.GetData(outInfo, 0)

        mesh_filename = self._mesh_filename
        if mesh_filename is None or mesh_filename == "None":
            mesh_filename = detect_mesh_file(self._filename)
        mesh = Mesh.from_file(mesh_filename)
        fieldfile = Fieldfile(self._filename, "r")

        blocks = semtex_to_vtk_multiblock(mesh, fieldfile, self._wrap_z)

        output.ShallowCopy(blocks)
        return 1


def make_structured_block(x, y, z, point_data):
    """Create a pure VTK StructuredGrid from coordinate arrays.

    x, y, z must have shape (nz, ns, nr), which than flattend (nr changes fastest).
    vtkStructuredGrid expects the set dimensions in opposite order (first changes fastest).
    """
    if x.shape != y.shape or x.shape != z.shape:
        raise ValueError("x, y, z must have identical shapes.")

    nz, ns, nr = x.shape

    grid = vtkStructuredGrid()
    grid.SetDimensions(nr, ns, nz)

    points_np = np.column_stack(
        [
            x.ravel(order="C"),
            y.ravel(order="C"),
            z.ravel(order="C"),
        ]
    )

    vtk_points = vtkPoints()
    vtk_points.SetData(numpy_to_vtk(points_np, deep=True))
    grid.SetPoints(vtk_points)

    for name, values in point_data.items():
        values = np.asarray(values)

        if values.size != nr * ns * nz:
            raise ValueError(
                f"Field {name!r} has {values.size} values, "
                f"expected {nr * ns * nz}."
            )

        vtk_array = numpy_to_vtk(values, deep=True)
        vtk_array.SetName(name)
        grid.GetPointData().AddArray(vtk_array)

    return grid


def semtex_to_vtk_multiblock(mesh, fieldfile, wrap_z):
    multiblock = vtkMultiBlockDataSet()
    multiblock.SetNumberOfBlocks(mesh.geometry.nel)

    for element_id in range(mesh.geometry.nel):
        x, y, z = mesh_to_block_arrays(mesh, element_id, wrap_z=wrap_z)
        point_data = field_to_block_point_data(fieldfile, element_id, wrap_z=wrap_z)

        block = make_structured_block(x, y, z, point_data)

        multiblock.SetBlock(element_id, block)
        multiblock.GetMetaData(element_id).Set(
            vtkMultiBlockDataSet.NAME(),
            f"element_{element_id}",
        )

    return multiblock


def detect_mesh_file(field_name: str | Path) -> Path | None:
    field_path = Path(field_name)
    session_name = field_path.stem.split('.')[0]
    mesh_path = field_path.with_name(f"{session_name}.msh")
    if not mesh_path.exists():
        raise FileNotFoundError(
            f"Could not find mesh file '{mesh_path.name}'"
            f"for field file '{field_path.name}'"
        )
    return mesh_path
