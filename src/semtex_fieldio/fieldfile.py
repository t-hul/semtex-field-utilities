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

    def __getitem__(self, field_name: str) -> np.ndarray:
        if self.data is None:
            raise RuntimeError("No data loaded")
        idx = self.field_index(field_name)
        return self.data[idx]

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

    def _offset(self, field_idx: int = 0, z_idx: int = 0) -> int:
        """Byte offset for a specific field and z-plane."""
        nz = self.geometry.nz
        nel = self.geometry.nel
        ns = self.geometry.ns
        nr = self.geometry.nr
        ntot_z = nel * ns * nr
        if z_idx >= nz:
            raise ValueError(f"z_idx {z_idx} exceeds number of planes {nz}")
        return (field_idx * self.npoints + z_idx * ntot_z) * 8  # bytes

    def _skip_header(self, f: BinaryIO) -> int:
        for _ in range(10):
            f.readline()
        return f.tell()

    def read_all_data(self) -> np.ndarray:
        """Read all float64 field data into memory and reshape."""
        with self.fname.open("rb") as f:
            self._skip_header(f)

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

    def read_fields(self, field_names: list[str]) -> np.ndarray:
        """Read entire fields for the given field names (all z-planes)."""
        result = []
        ntot = self.npoints

        with self.fname.open("rb") as f:
            hdr_off = self._skip_header(f)

            for name in field_names:
                logger.info(f"Reading field {name} from {self.fname.resolve()}")
                idx = self.field_index(name)
                offset = self._offset(field_idx=idx, z_idx=0)
                f.seek(offset + hdr_off)
                buf = f.read(ntot * 8)
                result.append(np.frombuffer(buf, dtype=np.float64))

        return np.stack(result)

    def read_fields_masked(
        self, field_names: list[str], mask: np.ndarray
    ) -> np.ndarray:
        """Read masked node values for all z-planes from selected fields."""
        nel, ns, nr = mask.shape
        if (self.geometry.nel, self.geometry.ns, self.geometry.nr) != (nel, ns, nr):
            raise ValueError("Mask shape does not match fieldfile geometry")

        nz = self.geometry.nz
        nfields = len(field_names)
        n_selected = np.count_nonzero(mask)

        if n_selected == 0:
            raise ValueError("Mask does not select any nodes")

        data_selected = np.empty((nfields, nz, n_selected), dtype=np.float64)

        for z_idx in range(nz):
            plane_data = self.read_zplane(
                z_idx, field_names
            )  # shape: (nfields, nel*ns*nr)
            for i, field_data in enumerate(plane_data):
                field_reshaped = field_data.reshape((nel, ns, nr))
                data_selected[i, z_idx] = field_reshaped[mask]

        return data_selected  # shape: (nfields, nz, n_selected)

    def read_zplane(
        self, z_idx: int, field_names: Optional[list[str]] = None
    ) -> np.ndarray:
        """Read a specific z-plane for all or selected fields."""
        if field_names is None:
            field_names = self.fields

        ntot_z = self.geometry.nel * self.geometry.ns * self.geometry.nr
        result = []

        with self.fname.open("rb") as f:
            hdr_off = self._skip_header(f)

            for name in field_names:
                idx = self.field_index(name)
                offset = self._offset(field_idx=idx, z_idx=z_idx)
                f.seek(offset + hdr_off)
                buf = f.read(ntot_z * 8)
                result.append(np.frombuffer(buf, dtype=np.float64))
                if z_idx < 10 or self.geometry.nz - z_idx < 10:
                    logger.debug(
                        f"Read plane {z_idx} of field {name} with {ntot_z} bytes"
                    )

        return np.stack(result)  # shape: (nfields, ntot_z)

    def get_data_dict(
        self, data: np.ndarray, field_names: Optional[list[str]] = None
    ) -> dict[np.ndarray]:
        if field_names is None:
            field_names = self.fields
        if len(field_names) != data.shape[0]:
            raise ValueError(
                f"Number of field names ({len(field_names)}) not equal to data.shape[0] ({data.shape[0]})"
            )
        data_dict = {}
        for i, key in enumerate(field_names):
            data_dict[key] = data[i, ...]
        logger.info("Created data dict")
        return data_dict

    def memory_map(self) -> np.memmap:
        """Memory-map the field data section of the file (read-only)."""
        # Estimate header size by reading 10 lines
        with self.fname.open("rb") as f:
            hdr_off = self._skip_header(f)

        # Confirm size matches expected total
        expected_bytes = self.ntotf * 8  # float64
        actual_bytes = self.fname.stat().st_size - hdr_off
        if actual_bytes < expected_bytes:
            raise ValueError(
                f"File too small for expected data. Missing {expected_bytes - actual_bytes} bytes."
            )

        shape = (self.nflds, self.npoints)
        return np.memmap(
            self.fname,
            dtype="float64",
            mode="r",
            offset=hdr_off,
            shape=shape,
        )

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
