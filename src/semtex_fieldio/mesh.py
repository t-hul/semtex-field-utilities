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
    _deduped_triangulation: Optional[tri.Triangulation] = field(
        default=None, init=False, repr=False, compare=False
    )
    _dedup_indices: Optional[np.ndarray] = None
    _is_dual: bool = False
    _x_normalized: bool = False
    _y_normalized: bool = False

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
        mask = tri.TriAnalyzer(self._triangulation).get_flat_tri_mask(0.01)
        self._triangulation.set_mask(mask)

    def get_triangulation(self, dual=False) -> tri.Triangulation:
        if self._triangulation is None:
            self.triangulate(dual=dual)
        elif self._is_dual != dual:
            self.triangulate(dual=dual)
        return self._triangulation

    def get_deduplicated_triangulation(self, dual=False) -> tri.Triangulation:
        """
        Returns a deduplicated meridional triangulation suitable for spatial queries.
        """
        if self._deduped_triangulation is not None and self._is_dual == dual:
            return self._deduped_triangulation, self._dedup_indices

        points = self.xy[..., :2].reshape(-1, 2)  # nel*ns*nr, 2
        if dual:
            offset = points.shape[0]
            points = np.concatenate(
                [points, np.stack([points[:, 0], -points[:, 1]], axis=1)]
            )
        points = np.round(points, decimals=10)
        unique_points, dedup_indices, inverse_indices = np.unique(
            points, axis=0, return_index=True, return_inverse=True
        )
        self._dedup_indices = dedup_indices
        raw_triangles = self._build_element_triangles()
        if dual:
            # add mirrored triangles, add offset of duplicated nodes
            raw_triangles = np.concatenate(
                [raw_triangles, raw_triangles[:, [0, 2, 1]] + offset]
            )
        triangles_deduped = inverse_indices[raw_triangles]

        self._deduped_triangulation = tri.Triangulation(
            unique_points[:, 0], unique_points[:, 1], triangles_deduped
        )
        self._is_dual = dual

        # Mask poorly shaped or boundary triangles
        # mask = tri.TriAnalyzer(triang).get_flat_tri_mask(0.01)
        # triang.set_mask(mask)

        return self._deduped_triangulation, dedup_indices

    def triangulate_axial_slice(
        self, x_target: float, r_target: np.ndarray
    ) -> tri.Triangulation:
        theta = self.theta[:-1]  # shape: (nz,)
        nz = len(theta)
        nr = len(r_target)
        logger.debug(f"shape(r_target): {r_target.shape}")

        # Structured grid (theta varies along axis=0, r along axis=1)
        Theta, R = np.meshgrid(theta, r_target, indexing="ij")  # shape: (nz, nr)

        Y = R * np.cos(Theta)
        Z = R * np.sin(Theta)

        # Flatten for triangulation
        y_flat = Y.flatten()
        z_flat = Z.flatten()

        # Structured grid triangulation
        triangles = []
        for i in range(nz):
            i_next = (i + 1) % nz  # wrap around in 0
            for j in range(nr - 1):
                n0 = i * nr + j
                n1 = i_next * nr + j
                n2 = i_next * nr + (j + 1)
                n3 = i * nr + (j + 1)
                triangles.append([n0, n1, n2])
                triangles.append([n0, n2, n3])

        triangles = np.array(triangles)
        triang = tri.Triangulation(y_flat, z_flat, triangles=triangles)

        # for x < 1 check if triangles are between elements, use meridional triangulation
        if x_target < 1:
            meridional_triang = self.get_deduplicated_triangulation()
            logger.info("Got meridional triang")
            finder = meridional_triang.get_trifinder()
            logger.info("Got triFinder")

            # center radius
            R = R.flatten()
            rc = (R[triangles[:, 0]] + R[triangles[:, 1]] + R[triangles[:, 2]]) / 3

            inside = (
                finder(np.full_like(rc, x_target), rc) >= 0
            )  # returns -1 if point is outside
            mask = ~inside

            logger.info(f"Masking {np.count_nonzero(mask)} triangles")
            triang.set_mask(mask)

        logger.info(f"Created triangulation of axial slice with {len(y_flat)} points")
        return triang

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
                    # skip if already in list (e.g element edges)
                    if r_line[r_idx] in r_target_list:
                        continue
                    mask[e, s, r_idx] = True
                    r_target_list.append(r_line[r_idx])
                    logger.debug(f"exact match at r[{r_idx}] = {r_line[r_idx]}")
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
                logger.debug(f"interpolated at r[{i_below} .. {i_above}] = {r_interp}")

        r_target = np.array(r_target_list)
        return mask, r_target

    def normalize(self, axial_scale=None, radius_scale=None, **kwargs):
        if self._x_normalized and self._y_normalized:
            return
        if axial_scale is None and radius_scale is None:
            return
        if axial_scale is not None and not self._x_normalized:
            self.xy[..., 0] /= axial_scale
            self._x_normalized = True

        if radius_scale is not None and not self._y_normalized:
            self.xy[..., 1] /= radius_scale
            self._y_normalized = True

    def is_normalized(self, dir="both"):
        if dir == "both":
            return self._x_normalized and self._y_normalized
        if dir in ["x", "axial"]:
            return self._x_normalized
        if dir == ["y", "r", "radius"]:
            return self._y_normalized

    def get_xy_mask(
        self, xlim: tuple[float, float], ylim: tuple[float, float]
    ) -> np.ndarray:
        """
        Create a mask for points within the given x and y limits.

        Parameters
        ----------
        xlim : tuple of float
            (xmin, xmax) range for x.
        ylim : tuple of float
            (ymin, ymax) range for y.

        Returns
        -------
        mask : np.ndarray of bool
            Boolean mask with shape (nel*ns*nr,), True if point lies within limits.
        """
        x = self.xy[..., 0].ravel()
        y = self.xy[..., 1].ravel()

        mask_x = (x >= xlim[0]) & (x <= xlim[1])
        mask_y = (y >= ylim[0]) & (y <= ylim[1])
        if self._is_dual:
            return np.concatenate([mask_x & mask_y, mask_x & mask_y])
        return mask_x & mask_y
