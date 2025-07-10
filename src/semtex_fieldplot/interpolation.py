import logging

import numpy as np
from scipy.interpolate import griddata

from semtex_fieldio import Mesh

logger = logging.getLogger(__name__)


def interpolate_to_axial_slice(
    masked_data: np.ndarray,
    mesh: Mesh,
    mask: np.ndarray,
    x_pos: float,
    y_target: np.ndarray,
) -> np.ndarray:
    """Interpolate masked data from mesh to target coordinates x_target, r_target"""
    logger.info("Interpolating axial slice data")
    x = mesh.xy[..., 0][mask]
    y = mesh.xy[..., 1][mask]
    points_known = np.column_stack((x, y))

    x_target = np.full_like(y_target, x_pos)
    points_target = np.column_stack((x_target, y_target))
    if np.all(np.isclose(points_known, points_target)):
        logger.info(f"shape(interp_data): {masked_data.shape}")
        return masked_data

    nfields, nz, n_select = masked_data.shape

    interp_data = np.empty((nfields, nz, len(y_target)), dtype=np.float64)
    for field_idx in range(nfields):
        for z_idx in range(nz):
            interp_data[field_idx, z_idx, :] = griddata(
                points_known, masked_data[field_idx, z_idx, :], points_target
            )

    logger.info(f"shape(interp_data): {interp_data.shape}")
    return interp_data  # shape: (nfields, nz, len(y_target))
