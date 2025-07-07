import matplotlib.pyplot as plt


def plot_mesh_xy(ax, mesh):

    for e in range(mesh.geometry.nel):
        block = mesh.xy[e]  # shape: (ns, nr, 2)

        # Plot lines in 'r' direction (each s)
        for s in range(mesh.geometry.ns):
            x = block[s, :, 0]
            y = block[s, :, 1]
            if s == 0 or s == mesh.geometry.ns - 1:
                linewidth = 1
                color = "black"
            else:
                linewidth = 0.5
                color = "gray"
            ax.plot(x, y, color=color, linewidth=linewidth)

        # Plot lines in 's' direction (each r)
        for r in range(mesh.geometry.nr):
            x = block[:, r, 0]
            y = block[:, r, 1]
            if r == 0 or r == mesh.geometry.nr - 1:
                linewidth = 1
                color = "black"
            else:
                linewidth = 0.5
                color = "gray"
            ax.plot(x, y, color=color, linewidth=linewidth)


def plot_mesh_xy_symm(ax, mesh, only_elements=False):

    for e in range(mesh.geometry.nel):
        block = mesh.xy[e]  # shape: (ns, nr, 2)

        # Plot lines in 'r' direction (each s)
        for s in range(mesh.geometry.ns):
            x = block[s, :, 0]
            y = block[s, :, 1]
            if s == 0 or s == mesh.geometry.ns - 1:
                linewidth = 1
                color = "black"
            else:
                linewidth = 0.5
                color = "gray"
                if only_elements:
                    continue
            ax.plot(x, y, color=color, linewidth=linewidth)
            ax.plot(x, -y, color=color, linewidth=linewidth)

        # Plot lines in 's' direction (each r)
        for r in range(mesh.geometry.nr):
            x = block[:, r, 0]
            y = block[:, r, 1]
            if r == 0 or r == mesh.geometry.nr - 1:
                linewidth = 1
                color = "black"
            else:
                linewidth = 0.5
                color = "gray"
                if only_elements:
                    continue
            ax.plot(x, y, color=color, linewidth=linewidth)
            ax.plot(x, -y, color=color, linewidth=linewidth)
