import os
import shutil
import subprocess

import numpy as np

from semtex_fieldio.fieldfile import Fieldfile
from semtex_fieldio.geometry import Geometry
from semtex_fieldio.header import Header


def test_basic_fieldfile_roundtrip(tmp_path):
    fname = tmp_path / "test.fld"

    geom = Geometry(nr=2, ns=2, nz=1, nel=1)
    fields = ["u", "v"]
    header = Header(geometry=geom, fields=fields, format="binary")
    header.session = "test_session"
    header.created = "2025-01-01"

    data = np.arange(8, dtype=np.float64)

    ff_write = Fieldfile(fname, "w", header=header)
    ff_write._print_geometry_info()
    ff_write.write(data)

    ff_read = Fieldfile(fname, "r")
    ff_read.read_all_data()
    ff_read._print_geometry_info()

    assert ff_read.hdr.session == "test_session"
    assert ff_read.hdr.geometry == geom
    assert ff_read.fields == fields
    assert ff_read.data is not None
    assert ff_read.data.shape == (2, 4)

    assert np.allclose(ff_read["u"], [0, 1, 2, 3])
    assert np.allclose(ff_read["v"], [4, 5, 6, 7])

    elmt_data = ff_read.reshape_elementwise()
    assert elmt_data.shape == (2, 1, 1, 2, 2)


def test_read_checkpoint(tmp_path):
    testdata_dir = os.path.join(os.path.dirname(__file__), "data")
    file_name = "koal1.1200.chk"
    copy_path = os.path.join(testdata_dir, file_name)
    orig_file = tmp_path / "original.chk"
    shutil.copy(copy_path, orig_file)

    ff_read = Fieldfile(orig_file, "r")
    assert ff_read.hdr.session == "koal1"
    assert ff_read.hdr.created == "Fri May 03 16:20:23 2024"
    geom = Geometry(nr=11, ns=11, nz=96, nel=30)
    assert ff_read.has_equal_geometry(geom)
    assert ff_read.hdr.step == 400000
    assert ff_read.hdr.time == 1200
    assert ff_read.hdr.dt == 0.0005
    assert ff_read.hdr.kinvis == 0.0001
    assert ff_read.hdr.beta == 1
    assert ff_read.fields == ["u", "v", "w", "p"]
    assert ff_read.hdr.format == "binary IEEE little-endian"

    data = ff_read.read_all_data()

    new_file = tmp_path / "copy.chk"
    ff_write = Fieldfile(new_file, "w", header=ff_read.hdr)
    ff_write.write(data)

    # Use `diff` to compare file byte-by-byte
    result = subprocess.run(
        ["diff", "-q", str(orig_file), str(new_file)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert (
        result.returncode == 0
    ), f"Files differ:\n{result.stdout.decode()}\n{result.stderr.decode()}"


def test_read_fields(tmp_path):
    fname = tmp_path / "test.fld"

    geom = Geometry(nr=2, ns=2, nz=5, nel=1)
    fields = ["u", "v", "w", "p"]
    header = Header(geometry=geom, fields=fields, format="binary")
    header.session = "test_session"
    header.created = "2025-01-01"

    write_data = np.arange(2 * 2 * 5 * 1 * 4, dtype=np.float64)

    ff_write = Fieldfile(fname, "w", header=header)
    ff_write.write(write_data)
    ff = Fieldfile(fname, "r")
    data = ff.read_fields(["v", "w"])  # replace with actual field names in file

    assert data.shape[0] == 2  # v and w
    assert data.shape[1] == ff.npoints
    assert data.dtype == "float64"

    # Optional content test
    v_data = data[0]
    assert np.all(np.isfinite(v_data))  # sanity check

    all_data = ff.read_all_data()
    print(ff["v"][10])
    print(data[0][10])
    assert np.allclose(ff["v"], data[0])
    assert np.allclose(ff["w"], data[1])
    assert np.allclose(all_data[1], data[0])
    assert np.allclose(all_data[2], data[1])


def test_read_zplane(tmp_path):
    testdata_dir = os.path.join(os.path.dirname(__file__), "data")
    file_name = "koal1.1200.chk"
    copy_path = os.path.join(testdata_dir, file_name)
    orig_file = tmp_path / "original.chk"
    shutil.copy(copy_path, orig_file)

    ff = Fieldfile(orig_file, "r")
    z_data = ff.read_zplane(z_idx=10, field_names=["u", "w"])  # example fields

    assert z_data.shape[0] == 2  # 2 fields
    assert z_data.shape[1] == ff.geometry.nel * ff.geometry.ns * ff.geometry.nr
    assert z_data.dtype == "float64"

    # Check values are bounded/floating
    assert np.all(np.isfinite(z_data))

    mm = ff.memory_map()

    ff.read_all_data()
    print(ff["w"][100])
    print(z_data[1][100])
    assert np.allclose(ff.reshape_elementwise()[0][10].flatten(), z_data[0])
    field3_z5 = mm[3].reshape(
        ff.geometry.nz, ff.geometry.nel * ff.geometry.ns * ff.geometry.nr
    )[5]
    z_data_2 = ff.read_zplane(z_idx=5, field_names=["p"])
    assert np.allclose(field3_z5, z_data_2)
