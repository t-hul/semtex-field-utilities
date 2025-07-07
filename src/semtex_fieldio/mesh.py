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
