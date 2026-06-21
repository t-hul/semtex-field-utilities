from __future__ import annotations

import numpy as np
import pyvista as pv
from pathlib import Path

from .mesh import Mesh
from .fieldfile import Fieldfile


def mesh_to_vtk_points(mesh: Mesh, like_tec: bool = False) -> np.ndarray:
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
        if like_tec:
            points[k, ..., 0] = -x
            points[k, ..., 1] = r * np.sin(th)
            points[k, ..., 2] = r * np.cos(th)
        else:
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
    if fieldfile.data is None:
        fieldfile.read_all_data()
    return fieldfile.get_data_dict(fieldfile.data)


def semtex_to_unstructured_grid(mesh: Mesh, fieldfile: Fieldfile) -> pv.UnstructuredGrid:
    points = mesh_to_vtk_points(mesh)
    cells, celltypes = mesh_to_vtk_cells(mesh)

    grid = pv.UnstructuredGrid(cells, celltypes, points)

    for name, values in field_to_point_data(fieldfile).items():
        if values.shape[0] != grid.n_points:
            raise ValueError(
                f"Field {name} has {values.shape[0]} values, "
                f"but grid has {grid.n_points} points.")
        grid.point_data[name] = values

    return grid


def write_vtu(grid: pv.UnstructuredGrid, filename: str | Path):
    """Write PyVista unstructured grid to a VTU file"""
    filename = Path(filename)

    if filename.suffix != ".vtu":
        filename = filename.with_suffix(".vtu")
    filename.parent.mkdir(parents=True, exist_ok=True)

    grid.save(filename)


def mesh_to_structured_block(mesh: Mesh, element_id: int) -> pv.StructuredGrid:
    """Convert Semtex mesh points to VTK Cartesian points of a given element
    and create a structured grid of the element x nz planes.
    """
    xy = mesh.xy
    theta = mesh.theta[:-1]

    _, ns, nr, _ = xy.shape
    nz = len(theta)

    points = np.empty((nz, ns, nr, 3), dtype=float)

    x = xy[element_id, ..., 0]
    r = xy[element_id, ..., 1]

    for k, th in enumerate(theta):
        if like_tec:
            points[k, ..., 0] = -x
            points[k, ..., 1] = r * np.sin(th)
            points[k, ..., 2] = r * np.cos(th)
        else:
            points[k, ..., 0] = x
            points[k, ..., 1] = r * np.cos(th)
            points[k, ..., 2] = r * np.sin(th)

    return pv.StructuredGrid(points[:, 0], points[:, 1], points[:, 2])


def field_to_block_point_data(fieldfile: Fieldfile, element_id: int):
    ...


def semtex_to_multiblock(mesh: Mesh, fieldfile: Fieldfile) -> pv.MultiBlock:
    ...


def write_vtm(blocks: pv.MultiBlock, filename: str | Path):
    """Write PyVista multiblock to a VTM file"""
    filename = Path(filename)

    if filename.suffix != ".vtm":
        filename = filename.with_suffix(".vtm")
    filename.parent.mkdir(parents=True, exist_ok=True)

    blocks.save(filename)
