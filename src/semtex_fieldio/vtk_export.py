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
    """Build linearized VTK cells.

    For nz == 1:
        each Semtex element becomes (ns-1)*(nr-1) QUAD cells.

    For nz > 1:
        each Semtex element becomes (nz)*(ns-1)*(nr-1) HEXAHEDRON cells,
        with periodic wrap in theta.
    """
    geo = mesh.geometry
    nr = geo.nr
    ns = geo.ns
    nz = geo.nz
    nel = geo.nel

    def _node_id(k, e, j, i):
        return ((k * nel + e) * ns + j) * nr + i

    cell_list = []

    if nz < 1:
        raise ValueError(f"Need at least 1 z-plane, got {nz}")
    elif nz == 1:
        ncells = nel * (ns - 1) * (nr - 1)
        for e in range(nel):
            for j in range(ns - 1):
                for i in range(nr - 1):
                    p0 = _node_id(0, e, j, i)
                    p1 = _node_id(0, e, j, i + 1)
                    p2 = _node_id(0, e, j + 1, i + 1)
                    p3 = _node_id(0, e, j + 1, i)

                    cell_list.append([4, p0, p1, p2, p3])
        if len(cell_list) == ncells:
            cells = np.array(cell_list).ravel()
        else:
            raise RuntimeError(
                f"Number of created cells {len(cell_list)} does not match expected number {ncells}")
        celltypes = np.full(ncells, pv.CellType.QUAD)
    else:
        ncells = nz * nel * (ns - 1) * (nr - 1)
        for k in range(nz):
            k_next = (k + 1) % nz

            for e in range(nel):
                for j in range(ns - 1):
                    for i in range(nr - 1):
                        p0 = _node_id(k, e, j, i)
                        p1 = _node_id(k, e, j, i + 1)
                        p2 = _node_id(k, e, j + 1, i + 1)
                        p3 = _node_id(k, e, j + 1, i)
                        p4 = _node_id(k_next, e, j, i)
                        p5 = _node_id(k_next, e, j, i + 1)
                        p6 = _node_id(k_next, e, j + 1, i + 1)
                        p7 = _node_id(k_next, e, j + 1, i)

                    cell_list.append([8, p0, p1, p2, p3, p4, p5, p6, p7])
        if len(cell_list) == ncells:
            cells = np.array(cell_list).ravel()
        else:
            raise RuntimeError(
                f"Number of created cells {len(cell_list)} does not match expected number {ncells}")
        celltypes = np.full(ncells, pv.CellType.HEXAHEDRON)

    return cells, celltypes


def field_to_point_data(fieldfile: Fieldfile) -> dict[str, np.ndarray]:
    """Return field arrays aligned with mesh_to_vtk_points ordering."""


def semtex_to_unstructured_grid(mesh: Mesh, fieldfile: Fieldfile) -> pv.UnstructuredGrid:
    points = mesh_to_vtk_points(mesh)
    cells, celltypes = mesh_to_vtk_cells(mesh)

    grid = pv.UnstructuredGrid(cells, celltypes, points)

    for name, values in field_to_point_data(fieldfile).items():
        grid.point_data[name] = values

    return grid
