import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .geometry import Geometry

logger = logging.getLogger(__name__)


@dataclass
class WallMesh:
    """Reduced Semtex wall mesh.

    The wall mesh uses the same first header line as a regular Semtex mesh:
    ``nr ns nz nel``. It then stores one ``x y`` coordinate per wall point in
    one z-plane. Some files additionally contain ``nz + 1`` theta/z values at
    the end; z-averaged files may omit this list.
    """

    geometry: Geometry
    xy: np.ndarray  # shape: (nel, ns, nr, 2)
    theta: Optional[np.ndarray] = None  # shape: (nz + 1,), if present

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "WallMesh":
        path = Path(path)
        with path.open("r") as f:
            header = f.readline().strip()
            try:
                nr, ns, nz, nel = map(int, header.split()[:4])
            except Exception as e:
                raise ValueError(
                    f"Failed to parse geometry from header line: {header}"
                ) from e

            geometry = Geometry(nr=nr, ns=ns, nz=nz, nel=nel)
            n_points_per_plane = nel * ns * nr

            xy_data = np.loadtxt(f, max_rows=n_points_per_plane)
            if xy_data.ndim == 1:
                xy_data = xy_data.reshape(1, -1)
            if xy_data.shape != (n_points_per_plane, 2):
                raise ValueError(
                    f"Expected ({n_points_per_plane}, 2) xy values in {path}, "
                    f"got {xy_data.shape}"
                )

            xy = xy_data.reshape(nel, ns, nr, 2)

            # Optional theta/z list. np.loadtxt warns on empty input in some
            # NumPy versions, so read the remaining text first.
            remaining = f.read().strip()
            if remaining:
                theta_data = np.fromstring(remaining, sep=" ")
                if theta_data.size != nz + 1:
                    raise ValueError(
                        f"Expected either no trailing theta values or {nz + 1}, "
                        f"got {theta_data.size} in {path}"
                    )
                theta = theta_data
            else:
                theta = None

        logger.info("Read wall mesh %s with geometry %s", path, geometry)
        return cls(geometry=geometry, xy=xy, theta=theta)

    @property
    def npoints_per_plane(self) -> int:
        return self.geometry.nel * self.geometry.ns * self.geometry.nr

    def has_compatible_field_geometry(self, field_geometry: Geometry) -> bool:
        """Return True for matching wall points and full-z or z-averaged fields."""
        same_wall_points = (
            self.geometry.nr == field_geometry.nr
            and self.geometry.ns == field_geometry.ns
            and self.geometry.nel == field_geometry.nel
        )
        same_z = self.geometry.nz == field_geometry.nz
        averaged_z = field_geometry.nz == 1
        return same_wall_points and (same_z or averaged_z)

    def select_line(
        self,
        *,
        constant: str,
        value: float,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        atol: float = 1.0e-8,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select a straight wall section.

        Parameters
        ----------
        constant:
            ``"x"`` for a vertical section or ``"y"`` for a horizontal section.
        value:
            Constant coordinate value.
        min_value, max_value:
            Bounds along the varying coordinate. For ``constant="y"`` these are
            x-bounds; for ``constant="x"`` they are y-bounds.
        atol:
            Absolute tolerance used for matching the constant coordinate.

        Returns
        -------
        idx:
            Flat point indices matching the selected wall section, sorted along
            the varying coordinate.
        coord:
            Varying coordinate values suitable for the x-axis of a line plot.
        xy:
            Selected ``x y`` coordinates, sorted like ``idx``.
        """
        constant = constant.lower()
        if constant not in {"x", "y"}:
            raise ValueError("line.constant must be either 'x' or 'y'")

        xy_flat = self.xy.reshape(-1, 2)
        const_col = 0 if constant == "x" else 1
        vary_col = 1 if constant == "x" else 0

        mask = np.isclose(xy_flat[:, const_col], value, atol=atol)
        varying = xy_flat[:, vary_col]

        if min_value is not None:
            mask &= varying >= min_value
        if max_value is not None:
            mask &= varying <= max_value

        idx = np.where(mask)[0]
        if idx.size == 0:
            raise ValueError(
                f"No wall points found for {constant}={value} "
                f"within [{min_value}, {max_value}] using atol={atol}"
            )

        order = np.argsort(varying[idx])
        idx = idx[order]
        return idx, varying[idx], xy_flat[idx]

    def select_line_from_config(
        self, line_config: dict
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select a wall section from the YAML ``line`` block."""
        if not line_config:
            raise ValueError("Missing required 'line' configuration block")
        if "constant" not in line_config or "value" not in line_config:
            raise ValueError("line config requires 'constant' and 'value'")

        return self.select_line(
            constant=line_config["constant"],
            value=float(line_config["value"]),
            min_value=(
                float(line_config["min"]) if line_config.get("min") is not None else None
            ),
            max_value=(
                float(line_config["max"]) if line_config.get("max") is not None else None
            ),
            atol=float(line_config.get("atol", 1.0e-8)),
        )
