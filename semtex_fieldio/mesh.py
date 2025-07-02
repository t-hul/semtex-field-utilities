from dataclasses import dataclass

import numpy as np

from .geometry import Geometry


@dataclass
class Mesh:
    geometry: Geometry
    xy: np.ndarray  # shape: (nel, ns, nr, 2)
    theta: np.ndarray  # shape: (nz,)

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
            if theta_data.size != nz:
                raise ValueError(f"Expected {nz} theta values, got {theta_data.size}")

        return cls(geometry=geometry, xy=xy, theta=theta_data)
