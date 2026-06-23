from __future__ import annotations

import numpy as np
from pathlib import Path

from .mesh import Mesh
from .fieldfile import Fieldfile


def mesh_to_block_arrays(mesh: Mesh, element_id: int, like_tec: bool = False, wrap_z: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert Semtex mesh points to VTK Cartesian points of a given element."""
    xy = mesh.xy
    if wrap_z:
        theta = mesh.theta
    else:
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

    points = points.transpose(2, 1, 0, 3)
    return points[..., 0], points[..., 1], points[..., 2]


def field_to_block_point_data(fieldfile: Fieldfile, element_id: int, wrap_z: bool = False):
    """Return field arrays for a given element aligned with mesh_to_structured_block ordering."""
    block_data = fieldfile.read_element(element_id, wrap_z=wrap_z)
    return fieldfile.get_data_dict(block_data)
