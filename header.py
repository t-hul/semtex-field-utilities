from dataclasses import dataclass
from typing import BinaryIO

from .geometry import Geometry


@dataclass
class Header:
    geometry: Geometry
    fields: str = ""
    format: str = "binary"
    session: str = ""
    created: str = ""
    step: int = 0
    time: float = 0.0
    dt: float = 0.0
    kinvis: float = 1.0
    beta: float = 1.0

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
