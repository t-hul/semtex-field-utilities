from pathlib import Path

import numpy as np

from semtex_fieldio.fieldfile import Fieldfile
from semtex_fieldio.wallmesh import WallMesh


TRACTION_FIELD_LABELS = {
    "n": "normal traction",
    "t": "tangential traction",
    "s": "spanwise traction",
    "h": "scalar gradient",
    "m": "traction magnitude",
    "d": "wall-normal element size",
}


def load_wall_traction_line(
    traction_file,
    wall_mesh_file,
    line,
    fields=None,
    plane_index=0,
):
    """
    Load wall traction data along a straight wall section.

    Returns a dict usable by profile plotting functions.

    Example return:
        {
            "x": array(...),
            "n": array(...),
            "t": array(...),
            "m": array(...),
        }
    """
    wall_mesh = WallMesh.from_file(wall_mesh_file)
    trc = Fieldfile(traction_file, "r")

    if fields is None:
        fields = trc.fields

    invalid = set(fields) - set(TRACTION_FIELD_LABELS)
    if invalid:
        raise ValueError(
            f"Invalid traction field(s): {sorted(invalid)}. "
            f"Valid fields are {sorted(TRACTION_FIELD_LABELS)}"
        )

    if not wall_mesh.has_compatible_field_geometry(trc.geometry):
        raise ValueError(
            f"Wall mesh geometry {wall_mesh.geometry} is not compatible "
            f"with traction file geometry {trc.geometry}"
        )

    idx, _, xy = wall_mesh.select_line(
        constant=line["constant"],
        value=line["value"],
        min_value=line.get("min"),
        max_value=line.get("max"),
        atol=line.get("atol", 1.0e-8),
    )

    if line["constant"] == "y":
        coord_name = "x"
        coord = xy[:, 0]
    elif line["constant"] == "x":
        coord_name = "radius"
        coord = xy[:, 1]
    else:
        raise ValueError("line['constant'] must be either 'x' or 'y'")

    data = trc.read_zplane(plane_index, field_names=fields)

    data_dict = {coord_name: coord}
    for i, field in enumerate(fields):
        data_dict[field] = data[i, idx]

    return data_dict
