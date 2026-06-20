import numpy as np
import pyvista as pv
from .mesh import Mesh
from .fieldfile import Fieldfile


def mesh_to_vtk_points(mesh: Mesh) -> np.ndarray:
    ...


def mesh_to_vtk_cells(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    ...


def field_to_point_data(fieldfile: Fieldfile) -> dict[str, np.ndarray]:
    ...


def semtex_to_unstructured_grid(mesh: Mesh, fieldfile: Fieldfile) -> pv.UnstructuredGrid:
    points = mesh_to_vtk_points(mesh)
    cells, celltypes = mesh_to_vtk_cells(mesh)

    grid = pv.UnstructuredGrid(cells, celltypes, points)

    for name, values in field_to_point_data(fieldfile).items():
        grid.point_data[name] = values

    return grid
