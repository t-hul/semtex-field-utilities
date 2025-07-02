import logging
from pathlib import Path
from typing import BinaryIO, Optional, Union

import numpy as np

from .geometry import Geometry
from .header import Header

logger = logging.getLogger(__name__)


class Fieldfile:
    def __init__(
        self, fname: Union[str, Path], mode: str, header: Optional[Header] = None
    ):
        if mode not in ("r", "w"):
            raise ValueError("mode must be 'r' (read) or 'w' (write)")

        self.fname = Path(fname)
        self.mode = mode
        self.data: np.ndarray | None = None

        if mode == "r":
            with self.fname.open("rb") as f:
                logger.info(f"Reading header from {self.fname.resolve()}")
                self.hdr = Header(Geometry())
                self.hdr.read(f)
                self._update_geometry_info()
                # delay reading until explicitly requested
        elif mode == "w":
            if not header:
                raise ValueError("Need header when writing file")
            self.hdr = header
            self._update_geometry_info()
            with self.fname.open("wb") as f:
                logger.info(f"Writing header to {self.fname.resolve()}")
                self.hdr.write(f)
                self._f = f  # if keep_open is needed
        else:
            ValueError("Unsupported mode")

    def _update_geometry_info(self) -> None:
        self.fields = self.hdr.fields
        self.geometry = self.hdr.geometry
        self.nflds = len(self.fields)
        self.npoints = (
            self.geometry.nr * self.geometry.ns * self.geometry.nz * self.geometry.nel
        )
        self.ntotf = self.npoints * self.nflds

    def _print_geometry_info(self) -> None:
        logger.debug("Geometry info:")
        logger.debug(repr(self.fields))
        logger.debug(repr(self.geometry))
        logger.debug(f"nfields: {self.nflds}")
        logger.debug(f"npoints: {self.npoints}")
        logger.debug(f"ntotf: {self.ntotf}")

    def read_all_data(self) -> np.ndarray:
        """Read all float64 field data into memory and reshape."""
        with self.fname.open("rb") as f:
            # Skip header
            for _ in range(10):
                f.readline()

            logger.info(f"Reading data from {self.fname.resolve()}")
            buf = f.read()
            flat = np.frombuffer(buf, dtype=np.float64)

            if flat.size != self.ntotf:
                raise ValueError(f"Expected {self.ntotf} values, got {flat.size}.")

            self.data = flat.reshape((self.nflds, self.npoints))
            logger.info(
                f"Loaded {self.data.shape[0]} fields with {self.data.shape[1]} points."
            )
            return self.data

    def write(self, data: np.ndarray, keep_open: bool = False) -> None:
        """Write float64 field data to file."""
        if data.dtype != np.float64:
            raise TypeError("Expected float64 data")
        mode = "ab" if keep_open else "r+b"
        with self.fname.open(mode) as f:
            logger.info(f"Writing data to {self.fname.resolve()}")
            f.seek(0, 2)
            data.tofile(f)

    def write_fields(self, fields: list[np.ndarray]) -> None:
        flat_data = np.concatenate([field.flatten() for field in fields])
        self.write(flat_data)

    def __getitem__(self, field_name: str) -> np.ndarray:
        if self.data is None:
            raise RuntimeError("No data loaded")
        idx = self.field_index(field_name)
        return self.data[idx]

    def field_index(self, name: str) -> int:
        return self.fields.index(name)

    def alloc(self) -> np.ndarray:
        """Allocate and return a flat zero-initialized field array."""
        return np.zeros(self.ntotf, dtype=np.float64)

    def reshape_elementwise(self) -> np.ndarray:
        """Reshape to (nflds, nz, nel, ns, nr) for element-wise access."""
        if self.data is None:
            raise RuntimeError("No data loaded")
        return self.data.reshape(
            (
                self.nflds,
                self.geometry.nz,
                self.geometry.nel,
                self.geometry.ns,
                self.geometry.nr,
            )
        )

    def has_equal_geometry(self, other_geo: Geometry) -> bool:
        equal, _ = self.geometry == other_geo
        return equal
