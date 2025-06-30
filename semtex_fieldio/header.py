from dataclasses import dataclass, field
from typing import BinaryIO

from .geometry import Geometry


@dataclass
class Header:
    geometry: Geometry
    fields: list[str] = field(default_factory=list)
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

        nr, ns, nz, nel = map(int, hdr_lines[2].strip().split()[:4])
        self.geometry = Geometry(nr, ns, nz, nel)

        self.step = int(hdr_lines[3].strip().split()[0])
        self.time = float(hdr_lines[4].strip().split()[0])
        self.dt = float(hdr_lines[5].strip().split()[0])
        self.kinvis = float(hdr_lines[6].strip().split()[0])
        self.beta = float(hdr_lines[7].strip().split()[0])

        self.fields = self._extract_fields(hdr_lines[8].strip())
        self.format = hdr_lines[9].strip()

    @staticmethod
    def _extract_fields(fields_line: str) -> list[str]:
        if fields_line.startswith("["):
            if "]" not in fields_line:
                raise ValueError("Closing bracket ']' of fields string not found")
            return fields_line[1:].split("]")[0].replace(",", " ").split()
        elif fields_line.split()[1] == "Fields":
            return [char for char in fields_line.split()[0]]
        else:
            raise ValueError(f"Format of fields_line not recognized: {fields_line}")

    def write(self, f: BinaryIO) -> None:
        f.write(str(self).encode("ascii"))

    def __str__(self) -> str:
        lines = [
            f"{self.session:<25} Session",
            f"{self.created:<25} Created",
            f"{self.geometry.nr:<4} {self.geometry.ns:<4} {self.geometry.nz:<4} \
                    {self.geometry.nel:<6}     Nr, Ns, Nz, Elements",
            f"{self.step:<25} Step",
            f"{self.time:<25g} Time",
            f"{self.dt:<25g} Time step",
            f"{self.kinvis:<25g} Kinvis",
            f"{self.beta:<25g} Beta",
            f"[{', '.join(self.fields)}] Fields written",
            f"{self.format:<25} Format",
        ]
        return "\n".join(lines) + "\n"
