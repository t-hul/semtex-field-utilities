from typing import BinaryIO

import numpy as np

from .geometry import Geometry
from .header import Header


class Fieldfile:
    def __init__(self, fname: str, mode: str, header: Header | None = None):
        if mode not in ("r", "w"):
            raise ValueError("mode must be 'r' (read) or 'w' (write)")

        self.fname = fname
        self.mode = mode
        self.data: np.ndarray | None = None

        if mode == "r":
            with open(fname, "rb") as f:
                self.hdr = Header(Geometry())
                self.hdr.read(f)
                self._read_data(f)
        elif mode == "w":
            if not header:
                raise ValueError("Need header when writing file")
            self.hdr = header
            with open(fname, "wb") as f:
                f.write(str(self.header).encode("ascii"))
                self._f = f  # if keep_open is needed
        else:
            ValueError("Unsupported mode")

        self.fields = self.hdr.fields
        self.geometry = self.hdr.geometry
        self.nflds = len(self.fields)

        self.npoints = (
            self.geometry.nr * self.geometry.ns * self.geometry.nz * self.geometry.nel
        )
        self.ntotf = self.npoints * self.nflds

    def _read_data(self, f: BinaryIO) -> None:
        """Read float64 field data from file into a (nflds, ntot) array."""
        buf = f.read()
        flat = np.frombuffer(buf, dtype=np.float64)

        if flat.size != self.ntotf:
            raise ValueError(f"Expected {self.ntotf} values, got {flat.size}.")
        self.data = flat.reshape((self.nflds, self.npoints))

    def write(self, data: np.ndarray, keep_open: bool = False) -> None:
        """Write float64 field data to file."""
        if data.dtype != np.float64:
            raise TypeError("Expected float64 data")
        with open(self.fname, "ab" if keep_open else "wb") as f:
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
