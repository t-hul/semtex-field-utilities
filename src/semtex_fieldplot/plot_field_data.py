import logging

import matplotlib.pyplot as plt
import numpy as np
import plot_utils.config as conf
import plot_utils.processor as proc
import plot_utils.utils as utils
from mpl_toolkits.axes_grid1 import make_axes_locatable

from semtex_fieldio.fieldfile import Fieldfile
from semtex_fieldio.mesh import Mesh

from .plot_mesh import plot_mesh_xy, plot_mesh_xy_symm

logger = logging.getLogger(__name__)


def plot_field_contour(
    ax, mesh, field_zplane_1, field_zplane_2=None, levels=30, cmap="viridis", label=None
):
    logger.info(f"Plotting field data with label {label}")
    nel, ns, nr = mesh.geometry.nel, mesh.geometry.ns, mesh.geometry.nr

    z = field_zplane_1.flatten()
    logger.debug(f"len(z1): {len(z)}")
    try:
        if field_zplane_2 is not None:
            z2 = field_zplane_2.flatten()
            z = np.concatenate([z, z2])
            triang = mesh.get_triangulation(dual=True)
        else:
            triang = mesh.get_triangulation()
    except:
        raise RuntimeError("Error getting triangulation")

    logger.debug(f"len(z): {len(z)}")
    contour = ax.tricontourf(triang, z, levels=levels, cmap=cmap)
    ax.set_aspect("equal")
    # ax.set_title("Field Contour at z-plane")

    # Create colorbar axis that matches the height of the main axis
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$r$")

    # plt.grid(True)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    plt.colorbar(contour, cax=cax, label=rf"${label}$")


def plot_from_config(file_names, config, save_path):
    mesh_name = config.get("mesh_file")
    z_idx = config.get("plane_index", 0)

    figure_name = config["name"]

    mesh = Mesh.from_file(mesh_name)

    for file_idx, fname in enumerate(file_names):
        ff = Fieldfile(fname, "r")
        field_list = config.get("fields")
        if field_list:
            field_list = [f for f in field_list if f in ff.fields]
        scaling_params = utils.get_scaling_parameters(config, None, 1)

        fig, axs = plt.subplots(
            len(field_list),
            1,
            sharex=True,
            figsize=config.get("fig_size", [8, 6]),
            dpi=config.get("dpi", 100),
            layout="tight",
        )

        if ff.geometry.nz == 1:
            mesh.geometry.nz = 1
        if ff.geometry != mesh.geometry:
            raise RuntimeError("Geometry mismatch")

        data1 = ff.read_zplane(z_idx, field_names=field_list)
        data_dict = ff.get_data_dict(data1, field_names=field_list)
        data_dict = proc.normalize_data(
            data_dict, config.get("normalize"), **scaling_params
        )[0]
        data1 = np.vstack([data_dict[name] for name in field_list])
        cmap = config.get("color_map", "viridis")

        if config.get("dual_plane", False):
            if ff.geometry.nz == 1:
                data2 = data1
            else:
                z_idx_2 = (z_idx + ff.geometry.nz // 2) % ff.geometry.nz
                data2 = ff.read_zplane(z_idx_2, field_names=field_list)
                data_dict2 = ff.get_data_dict(data2, field_names=field_list)
                data_dict2 = proc.normalize_data(
                    data_dict2, config.get("normalize"), **scaling_params
                )[0]
                data2 = np.vstack([data_dict2[name] for name in field_list])
        else:
            data2 = None

        for i, field in enumerate(field_list):
            if data2 is not None:
                d2 = data2[i]
            else:
                d2 = None

            if config.get("normalize"):
                label = proc.get_axis_label(
                    field, config.get("normalize"), config.get("use_plus")
                )
            else:
                label = field
            plot_field_contour(
                axs[i],
                mesh,
                data1[i],
                field_zplane_2=d2,
                label=label,
                cmap=cmap,
            )

            conf.set_axis_limits(axs[i], config)

            if config.get("plot_mesh"):
                plot_mesh_xy_symm(axs[i], mesh, only_elements=True)
            if i < len(field_list) - 1:
                axs[i].set_xlabel("")

        if isinstance(config.get("title"), list):
            plt.title(rf"{config.get('title')[file_idx]}")

        plt.savefig(f"{save_path}/contour_{figure_name}_{file_idx}.pdf", format="pdf")
        plt.savefig(f"{save_path}/contour_{figure_name}_{file_idx}")
        plt.close()
