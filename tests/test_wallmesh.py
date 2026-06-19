import numpy as np
from semtex_fieldio.wallmesh import WallMesh


def test_read_wallmesh():
    wm = WallMesh.from_file("tests/data/RIK.wmsh")

    assert wm.geometry.nr == 11
    assert wm.geometry.ns == 1
    assert wm.geometry.nel == 435

    assert wm.xy.shape == (435, 1, 11, 2)
    assert len(wm.theta) == 721


def test_read_wallmesh_z1():
    wm = WallMesh.from_file("tests/data/RIK_z1.wmsh")

    assert wm.theta is None


def test_select_y_constant():
    wm = WallMesh.from_file("tests/data/RIK.wmsh")
    ixd, coord, xy = wm.select_line(
        constant="y",
        value=14.0,
        min_value=-40,
        max_value=60
    )

    assert np.allclose(xy[:, 1], 14.0)
    assert np.all(np.diff(xy[:, 0]) >= 0)
    assert np.all(np.equal(coord, xy[:, 0]))


def test_select_x_constant():
    wm = WallMesh.from_file("tests/data/RIK.wmsh")
    ixd, coord, xy = wm.select_line(
        constant="x",
        value=60.0,
        min_value=0,
        max_value=14
    )

    assert np.allclose(xy[:, 0], 60.0)
    assert np.all(np.diff(xy[:, 1]) >= 0)
    assert np.all(np.equal(coord, xy[:, 1]))
