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

    def read(self, f: BinaryIO) -> None:
        hdr_lines = [f.readline().decode("ascii") for _ in range(10)]

        self.session = hdr_lines[0].strip().split()[0]
        self.created = hdr_lines[1].strip().split()[0]

        nr, ns, nz, nel = map(int, hdr_lines[2].strip().split())
        self.geometry = Geometry(nr, ns, nz, nel)

        self.step = int(hdr_lines[3].strip().split()[0])
        self.time = float(hdr_lines[4].strip().split()[0])
        self.dt = float(hdr_lines[5].split().strip()[0])
        self.kinvis = float(hdr_lines[6].split().strip()[0])
        self.beta = float(hdr_lines[7].split().strip()[0])

        self.fields = hdr_lines[8].strip().replace("[", "").replace("]", "")
        self.format = hdr_lines[9].strip()

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
