from __future__ import annotations

import numpy as np
import pyvista as pv

from .mesh import Mesh
from .fieldfile import Fieldfile


def mesh_to_vtk_points(mesh: Mesh) -> np.ndarray:
    """Convert Semtex mesh points to VTK Cartesian points.

    Returns
    -------
    points : ndarray, shape (nz * nel * ns * nr, 3)
        Point order is z-major, then element, then s, then r.
    """
    xy = mesh.xy
    theta = mesh.theta[:-1]

    nel, ns, nr, _ = xy.shape
    nz = len(theta)

    points = np.empty((nz, nel, ns, nr, 3), dtype=float)

    x = xy[..., 0]
    r = xy[..., 1]

    for k, th in enumerate(theta):
        points[k, ..., 0] = x
        points[k, ..., 1] = r * np.cos(th)
        points[k, ..., 2] = r * np.sin(th)

    return points.reshape(-1, 3)


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
