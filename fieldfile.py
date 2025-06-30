from typing import BinaryIO

import numpy as np

from .geometry import Geometry
from .header import Header


class Fieldfile:
    def __init__(self, fname, state, header=None):

        if state == "r":
            self.f = open(fname, "rb")
            self.hdr = Header(Geometry())
            self.hdr.read(self.f)

        elif state == "w":
            if not header:
                raise ValueError("Need header when writing file")
            self.hdr = header
            self.f = open(fname, "w")
            self.hdr.write(self.f)

        # FIXME: this design sucks. Shouldnt replicate data.
        #        Get rid of hdr?
        self.nr = self.hdr.geometry.nr
        self.ns = self.hdr.geometry.ns
        self.nz = self.hdr.geometry.nz
        self.nel = self.hdr.geometry.nel

        self.nrns = self.hdr.geometry.nr * self.hdr.geometry.ns
        self.nxy = self.nrns * self.hdr.geometry.nel
        self.ntot = self.nxy * self.hdr.geometry.nz
        self.nflds = len(self.hdr.fields)
        self.ntotf = self.ntot * self.nflds

        self.data = None

        # -- create list of field variables
        self.fields = [f for f in self.hdr.fields]

    # --------------------------------------------------------------------------
    #    def read(self):
    #        bin = array.array('d')
    #        bin.read(self.f, 1)
    #        return bin[0]

    # --------------------------------------------------------------------------
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
