import logging
from typing import Optional  # , Tuple, Union

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import plot_utils.config as conf
import plot_utils.processor as proc
import plot_utils.utils as utils
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from plot_utils.plotter import plot_additions

from semtex_fieldio.fieldfile import Fieldfile
from semtex_fieldio.mesh import Mesh
from semtex_fieldplot import interpolation
from semtex_fieldplot.plot_mesh import plot_mesh_xy, plot_mesh_xy_symm

# from mpl_toolkits.axes_grid1.inset_locator import inset_axes


logger = logging.getLogger(__name__)


def plot_field_meridional_contour(
    ax,
    mesh: Mesh,
    field_zplane_1: np.ndarray,
    field_zplane_2=None,
    levels=30,
    z_lim=[None, None],
    cmap="viridis",
    label=None,
    auto_range_x_lim=None,
    auto_range_y_lim=None,
):
    logger.info(f"Plotting field data with label {label}")
    nel, ns, nr = mesh.geometry.nel, mesh.geometry.ns, mesh.geometry.nr

    z = field_zplane_1.flatten()
    logger.debug(f"len(z1): {len(z)}")
    try:
        if field_zplane_2 is not None:
            z2 = field_zplane_2.flatten()
            z = np.concatenate([z, z2])
            # triang = mesh.get_triangulation(dual=True)
            triang, dedup_indices = mesh.get_deduplicated_triangulation(dual=True)
        else:
            # triang = mesh.get_triangulation()
            triang, dedup_indices = mesh.get_deduplicated_triangulation()
    except Exception as e:
        raise RuntimeError(f"Error getting triangulation: {e}")

    logger.debug(f"len(z): {len(z)}")
    levels, norm, ticks, extend = apply_z_limits(
        ax,
        mesh,
        z,
        z_lim,
        levels,
        auto_range_x_lim=auto_range_x_lim,
        auto_range_y_lim=auto_range_y_lim,
    )
    contour = ax.tricontourf(
        triang, z[dedup_indices], levels=levels, cmap=cmap, norm=norm, extend=extend
    )
    ax.set_aspect("equal")
    # ax.set_title("Field Contour at z-plane")

    # Create colorbar axis that matches the height of the main axis
    # plt.grid(True)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    ax.figure.colorbar(
        contour, cax=cax, label=rf"${label}$", ticks=ticks, extend=extend
    )


def plot_field_axial_contour(
    ax,
    triang,
    field_data: np.ndarray,
    levels=30,
    z_lim=[None, None],
    cmap="viridis",
    label=None,
):
    logger.info(f"Plotting field data with label {label}")

    z = field_data.flatten()

    levels, norm, ticks, extend = apply_z_limits(z, z_lim, levels)
    contour = ax.tricontourf(
        triang, z, levels=levels, cmap=cmap, norm=norm, extend=extend
    )
    ax.set_aspect("equal")
    # ax.set_title("Field Contour at z-plane")

    # Create colorbar axis that matches the height of the main axis
    ax.set_xlabel(r"$y$")
    ax.set_ylabel(r"$z$")

    # plt.grid(True)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(contour, cax=cax, label=rf"${label}$", ticks=ticks, extend=extend)


def plot_meridional_planes_for_file(
    ff: Fieldfile, mesh: Mesh, config: dict, field_list: list[str], axs
):
    if ff.geometry.nz == 1:
        mesh.geometry.nz = 1
    if ff.geometry != mesh.geometry:
        raise RuntimeError("Geometry mismatch")

    # Load data
    load_fields = utils.get_needed_fields(field_list, ff.fields)
    z_idx = config.get("plane_index", 0)
    data1 = ff.read_zplane(z_idx, field_names=load_fields)
    data_dict = ff.get_data_dict(data1, field_names=load_fields)

    # Normalization
    # aspect_ratio = 1.0
    scaling_params = utils.get_scaling_parameters(config, None, 1)
    # mesh.normalize(**scaling_params)
    # if (
    #     scaling_params["axial_scale"][0] is not None
    #     and scaling_params["radius_scale"][0] is not None
    # ):
    #     aspect_ratio = (
    #         scaling_params["radius_scale"][0] / scaling_params["axial_scale"][0]
    #     )
    data_dict = proc.normalize_data(
        data_dict,
        config.get("normalize"),
        quantity_type=config.get("quantity_type"),
        **scaling_params,
    )[0]
    data_dict = proc.append_calculated_data(data_dict, **scaling_params)[0]
    data1 = np.vstack([data_dict[name] for name in field_list])

    cmap = config.get("color_map", "viridis")

    # Process second plane
    if config.get("dual_plane", False):
        if ff.geometry.nz == 1:
            data2 = data1
            data_dict2 = data_dict
        else:
            z_idx_2 = (z_idx + ff.geometry.nz // 2) % ff.geometry.nz
            data2 = ff.read_zplane(z_idx_2, field_names=load_fields)
            data_dict2 = ff.get_data_dict(data2, field_names=load_fields)
            data_dict2 = proc.normalize_data(
                data_dict2, config.get("normalize"),
                quantity_type=config.get("quantity_type"), **scaling_params
            )[0]
            data_dict2 = proc.append_calculated_data(data_dict2, **scaling_params)[0]
            data2 = np.vstack([data_dict2[name] for name in field_list])
    else:
        data2 = None
        data_dict2 = None
        if not field_list and config.get("plot_mesh"):
            plot_mesh_xy(axs[0], mesh, only_elements=True)

    # Plot fields
    for i, field in enumerate(field_list):
        if data2 is not None:
            d2 = data2[i]
        else:
            d2 = None

        if config.get("normalize"):
            label = proc.get_axis_label(
                field,
                config.get("normalize"),
                config.get("use_plus"),
                quantity_type=config.get("quantity_type"),
            )
            # mathtex from matplotlib automatically adds '$', remove manual ones
            if not config.get("usetex", True):
                label = label.strip("$")
        else:
            label = field

        z_lim = utils.get_z_limits(config, field)

        levels = config.get("levels", 30)
        plot_field_meridional_contour(
            axs[i],
            mesh,
            data1[i],
            field_zplane_2=d2,
            label=label,
            z_lim=z_lim,
            cmap=cmap,
            levels=levels,
            auto_range_x_lim=utils.list_to_tuple(config.get("auto_range_x_lim")),
            auto_range_y_lim=utils.list_to_tuple(config.get("auto_range_y_lim")),
        )

        conf.set_axis_limits(axs[i], config)
        if scaling_params["axial_scale"][0] is not None:
            axs[i].set_xlabel(r"$x / L_\mathrm{c}$")
            x_scale = scaling_params["axial_scale"][0]
            axs[i].xaxis.set_major_formatter(
                FuncFormatter(lambda v, _: f"{v/x_scale:g}")
            )
            axs[i].xaxis.set_major_locator(MultipleLocator(base=0.25 * x_scale))
        else:
            axs[i].set_xlabel(r"$x$")
        if scaling_params["radius_scale"][0] is not None:
            axs[i].set_ylabel(r"$r / R$")
            y_scale = scaling_params["radius_scale"][0]
            axs[i].yaxis.set_major_formatter(
                FuncFormatter(lambda v, _: f"{v/y_scale:g}")
            )
            axs[i].yaxis.set_major_locator(MultipleLocator(base=0.5 * y_scale))
        else:
            axs[i].set_ylabel(r"$r$")

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
    triang = mesh.triangulate_axial_slice(x_target, r_target)

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
                field,
                config.get("normalize"),
                config.get("use_plus"),
                quantity_type=config.get("quantity_type"),
            )
            if not config.get("usetex", True):
                label = label.strip("$")
        else:
            label = field
        z_lim = utils.get_z_limits(config, field)
        levels = config.get("levels", 30)
        plot_field_axial_contour(
            axs[i], triang, data[i], label=label, z_lim=z_lim, cmap=cmap, levels=levels
        )

        conf.set_axis_limits(axs[i], config)

        if config.get("plot_mesh"):
            pass
            # plot_axial_mesh(axs[i], mesh, only_elements=True)


def plot_figure(fname, config, mesh, save_path, fig_idx, slice_type, x_target=None):
    ff = Fieldfile(fname, "r")
    figure_name = config["name"]
    config_fields = config.get("fields")
    field_list = []
    if config_fields is None or config_fields == "all":
        field_list = ff.fields
    else:
        field_list = utils.expand_field_list(config_fields, ff.fields)

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
    if isinstance(axs, plt.Axes):
        axs = np.array([axs])
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


def apply_z_limits(
    ax,
    mesh,
    z,
    z_lim=[None, None],
    levels=30,
    ticks=5,
    auto_range_x_lim=None,
    auto_range_y_lim=None,
):
    """Returns tuple of levels, norm, ticks"""
    extend = "neither"
    if isinstance(z_lim, list):
        z_min = z_lim[0]
        z_max = z_lim[1]
        masked_min = masked_max = 0
        if z_min is None or z_max is None:
            if not auto_range_x_lim:
                auto_range_x_lim = ax.get_xlim()
            if not auto_range_y_lim:
                auto_range_y_lim = ax.get_ylim()
            mask = mesh.get_xy_mask(auto_range_x_lim, auto_range_y_lim)
            masked_min = np.min(z[mask])
            masked_max = np.max(z[mask])
        if z_min is None:
            z_min = np.round(masked_min, 1)
        if z_max is None:
            z_max = np.round(masked_max, 1)
        if z_min >= z_max:
            for i in range(2, 10):
                z_min = np.round(masked_min, i)
                z_max = np.round(masked_max, i)
                if z_min < z_max:
                    break
            z_min = np.round(masked_min - 0.5 * 0.1**i, i)
            z_max = np.round(masked_max + 0.5 * 0.1**i, i)
        else:
            if z_lim[0] is None:
                z_min = np.round(masked_min - 0.05, 1)
            if z_lim[1] is None:
                z_max = np.round(masked_max + 0.05, 1)

        if z_min < 0 and z_max > 0:
            z_norm = np.max(np.abs([z_min, z_max]))
            norm = colors.Normalize(vmin=-z_norm, vmax=z_norm, clip=False)
        else:
            norm = colors.Normalize(vmin=z_min, vmax=z_max, clip=False)
        levels = np.linspace(z_min, z_max, levels)
        ticks = np.linspace(z_min, z_max, 5)
        if np.min(z) < z_min:
            extend = "min"
            if np.max(z) > z_max:
                extend = "both"
        elif np.max(z) > z_max:
            extend = "max"
    else:
        norm = None
        ticks = None

    return levels, norm, ticks, extend


