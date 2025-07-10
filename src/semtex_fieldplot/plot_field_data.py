import logging

import matplotlib.pyplot as plt
import numpy as np
import plot_utils.config as conf
import plot_utils.processor as proc
import plot_utils.utils as utils
from mpl_toolkits.axes_grid1 import make_axes_locatable

from semtex_fieldio.fieldfile import Fieldfile
from semtex_fieldio.mesh import Mesh
from semtex_fieldplot import interpolation
from semtex_fieldplot.plot_mesh import plot_mesh_xy, plot_mesh_xy_symm

logger = logging.getLogger(__name__)


def plot_field_meridional_contour(
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


def plot_field_axial_contour(
    ax, triang, field_data, levels=30, cmap="viridis", label=None
):
    logger.info(f"Plotting field data with label {label}")

    z = field_data.flatten()

    contour = ax.tricontourf(triang, z, levels=levels, cmap=cmap)
    ax.set_aspect("equal")
    # ax.set_title("Field Contour at z-plane")

    # Create colorbar axis that matches the height of the main axis
    ax.set_xlabel(r"$y$")
    ax.set_ylabel(r"$z$")

    # plt.grid(True)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(contour, cax=cax, label=rf"${label}$")


def plot_meridional_planes_for_file(
    ff: Fieldfile, mesh: Mesh, config: dict, field_list: list[str], axs
):
    if ff.geometry.nz == 1:
        mesh.geometry.nz = 1
    if ff.geometry != mesh.geometry:
        raise RuntimeError("Geometry mismatch")

    # Load data
    z_idx = config.get("plane_index", 0)
    data1 = ff.read_zplane(z_idx, field_names=field_list)
    data_dict = ff.get_data_dict(data1, field_names=field_list)

    # Normalization
    scaling_params = utils.get_scaling_parameters(config, None, 1)
    data_dict = proc.normalize_data(
        data_dict, config.get("normalize"), **scaling_params
    )[0]
    data1 = np.vstack([data_dict[name] for name in field_list])

    cmap = config.get("color_map", "viridis")

    # Process second plane
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

    # Plot fields
    for i, field in enumerate(field_list):
        if data2 is not None:
            d2 = data2[i]
        else:
            d2 = None

        if config.get("normalize"):
            label = proc.get_axis_label(
                field, config.get("normalize"), config.get("use_plus")
            )
            if not config.get("usetex", True):
                label = label.strip("$")
        else:
            label = field
        plot_field_meridional_contour(
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


def plot_axial_planes_for_file(
    ff: Fieldfile, mesh: Mesh, config: dict, field_list: list[str], axs, x_target=None
):
    if ff.geometry != mesh.geometry:
        raise RuntimeError("Geometry mismatch")

    if x_target is None:
        x_target = config.get("x_slice")
        if isinstance(x_target, list):
            x_target = x_target[0]
            logger.warning(f"Choosing first value of x_slice list: {x_target}")
    if x_target is None:
        raise ValueError("x_slice not defined in config")
    logger.info(f"Plotting axial slice at x = {x_target}")

    mask, r_target = mesh.axial_node_mask_with_r(x_target, tol=0.25)
    triang = mesh.triangulate_axial_slice(r_target)

    # Load data
    data = ff.read_fields_masked(field_list, mask)
    print(f"shape(data): {data.shape}")
    data_dict = ff.get_data_dict(data, field_names=field_list)

    # Normalization
    scaling_params = utils.get_scaling_parameters(config, None, 1)
    data_dict = proc.normalize_data(
        data_dict, config.get("normalize"), **scaling_params
    )[0]
    data = np.stack([data_dict[name] for name in field_list])
    data = interpolation.interpolate_to_axial_slice(
        data, mesh, mask, x_target, r_target
    )  # .reshape(   len(field_list), -1)  # reshape(nfields, ff.hdr.geometry.nz * len(r_target))
    print(f"shape(data): {data.shape}")

    cmap = config.get("color_map", "viridis")

    # Plot fields
    for i, field in enumerate(field_list):
        logger.info(f"Plotting field {field}")
        if config.get("normalize"):
            label = proc.get_axis_label(
                field, config.get("normalize"), config.get("use_plus")
            )
            if not config.get("usetex", True):
                label = label.strip("$")
        else:
            label = field
        plot_field_axial_contour(
            axs[i],
            triang,
            data[i],
            label=label,
            cmap=cmap,
        )

        conf.set_axis_limits(axs[i], config)

        if config.get("plot_mesh"):
            pass
            # plot_axial_mesh(axs[i], mesh, only_elements=True)


def plot_figure(fname, config, mesh, save_path, fig_idx, slice_type, x_target=None):
    ff = Fieldfile(fname, "r")
    figure_name = config["name"]
    field_list = config.get("fields")
    if field_list:
        field_list = [f for f in field_list if f in ff.fields]

    if slice_type == "meridional":
        n_cols = 1
    elif slice_type == "axial":
        n_cols = 2
    n_rows = (len(field_list) + n_cols - 1) // n_cols
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        sharex=True,
        figsize=config.get("fig_size", [8, 6]),
        dpi=config.get("dpi", 100),
        layout="tight",
    )
    axs = axs.flatten()
    if isinstance(config.get("title"), list):
        axs[0].set_title(rf"{config.get('title')[fig_idx]}")

    if slice_type == "meridional":
        plot_meridional_planes_for_file(ff, mesh, config, field_list, axs)
    elif slice_type == "axial":
        plot_axial_planes_for_file(ff, mesh, config, field_list, axs, x_target)

    for j in range(len(field_list), n_rows * n_cols):
        fig.delaxes(axs[j])

    plt.savefig(
        f"{save_path}/contour_{slice_type}_{figure_name}_{fig_idx}.pdf",
        format="pdf",
    )
    plt.savefig(f"{save_path}/contour_{slice_type}_{figure_name}_{fig_idx}")
    plt.close()


def plot_from_config(file_names, config, save_path):
    mesh_name = config.get("mesh_file")

    mesh = Mesh.from_file(mesh_name)

    slice_type = config.get("slice", "meridional")
    allowed_slices = ["meridional", "axial"]
    if slice_type not in allowed_slices:
        raise ValueError(
            f"Unsupported slice: {slice_type}. Choose from: {allowed_slices}"
        )

    if slice_type == "meridional" or len(file_names) > 1:
        for file_idx, fname in enumerate(file_names):
            plot_figure(fname, config, mesh, save_path, file_idx, slice_type)
    elif slice_type == "axial" and len(file_names) == 1:
        x_target = config.get("x_slice")
        fname = file_names[0]
        if isinstance(x_target, list):
            for x_idx, x_t in enumerate(x_target):
                plot_figure(
                    fname, config, mesh, save_path, x_idx, slice_type, x_target=x_t
                )
