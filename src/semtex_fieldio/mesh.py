import logging
from dataclasses import dataclass, field
from typing import Optional

import matplotlib.tri as tri
import numpy as np

from .geometry import Geometry

logger = logging.getLogger(__name__)


@dataclass
class Mesh:
    geometry: Geometry
    xy: np.ndarray  # shape: (nel, ns, nr, 2)
    theta: np.ndarray  # shape: (nz+1,)
    _triangulation: Optional[tri.Triangulation] = field(
        default=None, init=False, repr=False, compare=False
    )
    _is_dual: bool = False

    @classmethod
    def from_file(cls, path: str) -> "Mesh":
        with open(path, "r") as f:
            # Parse header line: "nr ns nz nel"
            header = f.readline().strip()
            try:
                nr, ns, nz, nel = map(int, header.split()[:4])
            except Exception as e:
                raise ValueError(
                    f"Failed to parse geometry from header line: {header}"
                ) from e

            geometry = Geometry(nr, ns, nz, nel)
            n_points_per_plane = nel * ns * nr

            # Read x, y coordinates (assume each line is: x y)
            xy_data = np.loadtxt(f, max_rows=n_points_per_plane)
            if xy_data.shape != (n_points_per_plane, 2):
                raise ValueError(
                    f"Expected ({n_points_per_plane}, 2) xy values, got {xy_data.shape}"
                )

            # Reshape to (nel, ns, nr, 2)
            xy = xy_data.reshape(nel, ns, nr, 2)

            # Read theta values for each z-plane
            theta_data = np.loadtxt(f)
            if theta_data.size != nz + 1:
                raise ValueError(
                    f"Expected {nz + 1} theta values, got {theta_data.size}"
                )

        return cls(geometry=geometry, xy=xy, theta=theta_data)

    def triangulate(self, dual=False, **kwargs) -> None:
        logger.info(f"Triangulating {'dual' if dual else 'single'} plane mesh")
        x = self.xy[..., 0].flatten()  # shape: (nel*ns*nr,)
        y = self.xy[..., 1].flatten()
        triangles = self._build_element_triangles()
        logger.debug(f"created triangles: {triangles.shape}")
        if dual:
            offset = len(x)
            x = np.concatenate([x, x])
            y = np.concatenate([y, -y])
            # add mirrored triangles, add offset of duplicated nodes
            triangles = np.concatenate([triangles, triangles[:, [0, 2, 1]] + offset])
            logger.debug(f"duplicated triangles: {triangles.shape}")
        self._triangulation = tri.Triangulation(x, y, triangles=triangles, **kwargs)
        self._is_dual = dual

        # Mask poorly shaped or boundary triangles
        # mask = tri.TriAnalyzer(self._triangulation).get_flat_tri_mask(0.01)
        # self._triangulation.set_mask(mask)

    def get_triangulation(self, dual=False) -> tri.Triangulation:
        if self._triangulation is None:
            self.triangulate(dual=dual)
        elif self._is_dual != dual:
            self.triangulate(dual=dual)
        return self._triangulation

    def _build_element_triangles(self) -> np.array:
        triangles = []

        nr = self.geometry.nr
        ns = self.geometry.ns

        for e in range(self.geometry.nel):
            base = e * nr * ns
            for j in range(ns - 1):
                for i in range(nr - 1):
                    # local node indices within the element, anti-clockwise
                    n0 = base + j * nr + i
                    n1 = base + j * nr + i + 1
                    n2 = base + (j + 1) * nr + i + 1
                    n3 = base + (j + 1) * nr + i

                    triangles.append([n0, n1, n2])
                    triangles.append([n0, n2, n3])

        return np.array(triangles)

    def axial_node_mask_with_r(
        self, x_target: float, tol: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return a boolean mask selecting nodes that bracket x_target per (element, s) line,
        anlong with their corresponding radial positions (exact or interpolated).

        Parameters
        ----------
        x_target : float
            Target axial position.
        tol : float
            Broad tolerance to pre-select candidate nodes.

        Returns
        -------
        mask : np.ndarray of shape (nel, ns, nr)
            Boolean mask selecting 1 or 2 nodes per (element, s) line
        r_target : np.ndarray of shape (n_selected,)
            Radial positions (either exact or interpolated) of selected nodes.
        """
        x = self.xy[..., 0]  # shape: (nel, ns, nr)
        r = self.xy[..., 1]  # shape: (nel, ns, nr)
        nel, ns, nr = x.shape
        mask = np.zeros_like(x, dtype=bool)
        r_target_list = []

        for e in range(nel):
            for s in range(ns):
                x_line = x[e, s, :]
                r_line = r[e, s, :]
                diffs = x_line - x_target

                in_tol = np.abs(diffs) < tol
                if not np.any(in_tol):
                    continue

                # Check for exact match
                exact = np.isclose(diffs, 0.0)
                if np.any(exact):
                    r_idx = np.where(exact)
                    mask[e, s, r_idx] = True
                    r_target_list.append(r_line[r_idx])
                    continue

                # Otherwise, find one node below and one above
                below = np.where(diffs < 0)[0]
                above = np.where(diffs > 0)[0]

                if below.size == 0 or above.size == 0:
                    continue  # can not interpolate - x_target not bracketed

                # Closest below and above index
                i_below = below[np.argmin(np.abs(diffs[below]))]
                i_above = above[np.argmin(np.abs(diffs[above]))]

                # Linear interpolation
                x0, x1 = x_line[i_below], x_line[i_above]
                r0, r1 = r_line[i_below], r_line[i_above]
                weight = (x_target - x0) / (x1 - x0)
                r_interp = r0 * (1 - weight) + r1 * weight

                mask[e, s, i_below] = True
                mask[e, s, i_above] = True
                r_target_list.append(r_interp)

        r_target = np.array(r_target_list)
        return mask, r_target
