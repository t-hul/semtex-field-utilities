from dataclasses import dataclass


@dataclass
class Geometry:
    nr: int = 3
    ns: int = 3
    nz: int = 1
    nel: int = 1

    def __str__(self) -> str:
        return f"{self.nr} {self.ns} {self.nz} {self.nel}"

    def __eq__(self, other: object) -> tuple[bool, str]:
        if not isinstance(other, Geometry):
            return False, "Not a Geometry instance"

        is_equal = True
        msg = []

        if self.nr != other.nr:
            msg.append("nr differs")
            is_equal = False
        if self.ns != other.ns:
            msg.append("ns differs")
            is_equal = False
        if self.nz != other.nz:
            msg.append("nz differs")
            is_equal = False
        if self.nel != other.nel:
            msg.append("nel differs")
            is_equal = False

        return is_equal, ", ".join(msg)
