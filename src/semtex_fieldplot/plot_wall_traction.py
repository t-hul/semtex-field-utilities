import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plot_utils.config as conf
import plot_utils.processor as proc
import plot_utils.utils as utils

from semtex_fieldio.fieldfile import Fieldfile
from semtex_fieldio.wallmesh import WallMesh

logger = logging.getLogger(__name__)

VALID_TRACTION_FIELDS = {"n", "t", "s", "h", "m", "d"}

TRACTION_FIELD_LABELS = {
    "n": "normal traction",
    "t": "tangential traction",
    "s": "spanwise traction",
    "h": "scalar gradient",
    "m": "traction magnitude",
    "d": "wall-normal element size",
}


def _validate_traction_fields(requested_fields, available_fields):
    invalid = set(requested_fields) - VALID_TRACTION_FIELDS
    if invalid:
        raise ValueError(
            f"Invalid traction field(s): {sorted(invalid)}. "
            f"Valid fields are: {sorted(VALID_TRACTION_FIELDS)}"
        )

    missing = set(requested_fields) - set(available_fields)
    if missing:
        raise ValueError(
            f"Requested traction field(s) not found in file: {sorted(missing)}. "
            f"Available fields are: {available_fields}"
        )


def _axis_label(field_name, config):
    if config.get("normalize"):
        return proc.get_axis_label(
            field_name, config.get("normalize"), config.get("use_plus")
        )
    return TRACTION_FIELD_LABELS.get(field_name, field_name)


def extract_wall_line(wall_mesh, field_zplane, line_config):
    """Extract values along a configured straight wall section."""
    idx, coord, xy = wall_mesh.select_line_from_config(line_config)
    return coord, field_zplane[idx], xy


def plot_wall_line(ax, coord, values, *, field_name, line_config, label=None, **kwargs):
    """Plot one traction field along the selected wall section."""
    ax.plot(coord, values, label=label or field_name, **kwargs)

    constant = line_config["constant"].lower()
    if constant == "y":
        ax.set_xlabel(r"$x$")
    elif constant == "x":
        ax.set_xlabel(r"$y$")
    else:
        raise ValueError("line.constant must be either 'x' or 'y'")

    ax.set_ylabel(label or field_name)
    ax.grid(True)


def plot_wall_traction_from_config(file_names, config, save_path):
    """Plot wall-traction line profiles from ``*.trc`` files."""
    wall_mesh_file = config.get("wall_mesh_file")
    if not wall_mesh_file:
        raise ValueError("wall_traction plots require 'wall_mesh_file'")

    line_config = config.get("line")
    if not line_config:
        raise ValueError("wall_traction plots require a 'line' config block")

    z_idx = config.get("plane_index", 0)
    figure_name = config["name"]
    wall_mesh = WallMesh.from_file(wall_mesh_file)

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    for file_idx, fname in enumerate(file_names):
        if Path(fname).suffix and Path(fname).suffix != ".trc":
            logger.warning("Expected a .trc traction file, got %s", fname)

        ff = Fieldfile(fname, "r")
        if not wall_mesh.has_compatible_field_geometry(ff.geometry):
            raise RuntimeError(
                f"Wall mesh geometry {wall_mesh.geometry} is incompatible with "
                f"traction field geometry {ff.geometry}"
            )

        field_list = config.get("fields") or ff.fields
        _validate_traction_fields(field_list, ff.fields)

        if ff.geometry.nz == 1:
            z_idx_to_read = 0
        else:
            z_idx_to_read = z_idx

        data = ff.read_zplane(z_idx_to_read, field_names=field_list)
        data_dict = ff.get_data_dict(data, field_names=field_list)

        scaling_params = utils.get_scaling_parameters(config, None, 1)
        data_dict = proc.normalize_data(
            data_dict, config.get("normalize"), **scaling_params
        )[0]

        fig, axs = plt.subplots(
            len(field_list),
            1,
            sharex=True,
            figsize=config.get("fig_size", [8, 6]),
            dpi=config.get("dpi", 100),
            layout="tight",
        )
        axs = np.atleast_1d(axs)

        for i, field in enumerate(field_list):
            label = _axis_label(field, config)
            coord, values, _ = extract_wall_line(
                wall_mesh, data_dict[field], line_config
            )
            plot_wall_line(
                axs[i], coord, values, field_name=field, line_config=line_config, label=label
            )
            conf.set_axis_limits(axs[i], config)
            if i < len(field_list) - 1:
                axs[i].set_xlabel("")

        title = config.get("title")
        if isinstance(title, list):
            fig.suptitle(rf"{title[file_idx]}")
        elif isinstance(title, str):
            fig.suptitle(rf"{title}")

        fig.savefig(save_path / f"wall_traction_{figure_name}_{file_idx}.pdf", format="pdf")
        fig.savefig(save_path / f"wall_traction_{figure_name}_{file_idx}")
        plt.close(fig)
