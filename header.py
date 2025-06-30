from dataclasses import dataclass
from typing import BinaryIO

from .geometry import Geometry


class Header:
    #    def __init__(self, nr=3, ns=3, nz=1, nel=1, geometry=None, fields="", format="binary"):
    def __init__(self, geometry, fields="", format="binary"):
        self.session = ""
        self.created = ""
        # if not geometry: geometry = Geometry(nr, ns, nz, nel)
        self.geometry = geometry
        self.step = 0
        self.time = 0.0
        self.dt = 0.0
        self.kinvis = 1.0
        self.beta = 1.0
        self.fields = fields
        self.format = format

    def read(self, f):
        hdr = []
        for i in range(0, 9):
            #            print(i)
            line = f.readline().decode("ascii")
            hdr.append(line.split())
        hdr.append(f.readline().decode("ascii"))

        self.session = hdr[0][0]
        self.created = hdr[1][0]
        self.geometry = Geometry(
            int(hdr[2][0]), int(hdr[2][1]), int(hdr[2][2]), int(hdr[2][3])
        )
        self.step = int(hdr[3][0])
        self.time = float(hdr[4][0])
        self.dt = float(hdr[5][0])
        self.kinvis = float(hdr[6][0])
        self.beta = float(hdr[7][0])
        rawFields = hdr[8][0]
        for char in "[,]":
            rawFields = rawFields.replace(char, "")
        self.fields = rawFields
        self.format = hdr[9][:-1]

    def write(self, f):
        f.write(str(self))

    def __str__(self):
        out = []
        out.append("%-25s Session" % self.session)
        out.append("%-25s Created" % self.created)
        out.append(
            "%-4i %-4i %-4i %-6i     Nr, Ns, Nz, Elements"
            % (self.geometry.nr, self.geometry.ns, self.geometry.nz, self.geometry.nel)
        )
        out.append("%-25i Step" % self.step)
        out.append("%-25g Time" % self.time)
        out.append("%-25g Time step" % self.dt)
        out.append("%-25g Kinvis" % self.kinvis)
        out.append("%-25g Beta" % self.beta)
        out.append("%-25s Fields written" % self.fields)
        out.append("%-25s Format" % self.format)
        return "\n".join(out) + "\n"
