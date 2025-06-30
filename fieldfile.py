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

    def write(self, data, keep_open=False):
        """write field data to file"""
        if data.dtype != np.dtype("float64"):
            raise TypeError("need float64 data")
        data.tofile(self.f)
        if not keep_open:
            self.f.close()

    def write_fields(self, fields):
        self.write(np.vstack(the_field.flatten() for the_field in fields))

    def __getitem__(self, fieldname):
        return self.data[self.field_index(fieldname)]

    # --------------------------------------------------------------------------
    def read(self):
        "read field data from file. Float64 data expected."
        buf = self.f.read()
        self.data = np.fromstring(buf, np.float64, count=-1).reshape(
            self.nflds, self.ntot
        )
        return self.data
        # return np.fromfile(self.f, 'd')
        self.close()

    def alloc(self):
        """allocate and return data storage"""
        #        if not fields: fields = self.hdr.fields
        return np.zeros((self.ntotf), dtype="float64")

    # --------------------------------------------------------------------------
    def close(self):
        self.f.close()

    def field_index(self, needle):
        return self.fields.index(needle)

    def is_mesh_compatible(self, mesh):
        return mesh.geometry == self.hdr.geometry


# ----------------------------------------------------------
def convert():
    """convert fieldfile to ASCII, like utility/convert.C"""
    ff = Fieldfile("example.fld", "r")
    ff.hdr.write(sys.stdout)
    data = ff.read()

    for i in range(ff.ntot):
        for field in range(ff.nflds):
            print("%g" % data[i, field])


#        print


def element_wise():
    """demonstrates element-wise access
    NB: Fieldfile.read() expects double precision (Float64) data, as is standard for
        semtex field files. No checks done for single precision or funny byte order.
    """
    ff = Fieldfile("example.fld", "r")
    data = ff.read()
    # elmt_wise = data.reshape((ff.nr, ff.ns, ff.nel, ff.nz, ff.nflds))
    elmt_wise = data.reshape((ff.nflds, ff.nz, ff.nel, ff.ns, ff.nr))  # works!
    #    print data
    #    - of nr * ns nodes
    #    - of 11th element
    #    - of first z-plane
    #    - of second field (in this case, v)
    print(elmt_wise[1, 0, 10, :, :])


if __name__ == "__main__":
    element_wise()
    # convert()
    # main()
