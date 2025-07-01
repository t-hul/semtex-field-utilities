import subprocess

import numpy as np

from semtex_fieldio import Fieldfile, Geometry, Header


def create_dummy_fld(path, fields, shape, value, step):
    geom = Geometry(nr=shape[0], ns=shape[1], nz=shape[2], nel=shape[3])
    hdr = Header(geometry=geom, fields=fields, step=step)
    ff = Fieldfile(path, "w", hdr)
    ntot = geom.nr * geom.ns * geom.nz * geom.nel
    data = np.full((len(fields), ntot), value, dtype=np.float64)
    ff.write(data)
    return data


def read_field_data(path):
    ff = Fieldfile(path, "r")
    return ff.data, ff.hdr.step


def test_weighted_add_and_subtract(tmp_path):
    # Setup
    shape = (2, 2, 1, 1)  # nr, ns, nz, nel
    fields = ["A", "B"]

    file1 = tmp_path / "file1.fld"
    file2 = tmp_path / "file2.fld"
    out_add = tmp_path / "add.fld"
    out_sub = tmp_path / "sub.fld"

    data1 = create_dummy_fld(file1, fields, shape, value=2.0, step=10)
    data2 = create_dummy_fld(file2, fields, shape, value=1.0, step=4)

    # Add
    subprocess.run(
        ["python", "scripts/avg_ops.py", str(file1), str(out_add), "--add", str(file2)],
        check=True,
    )
    data_add, step_add = read_field_data(out_add)
    expected_add = (2.0 * 10 + 1.0 * 4) / (10 + 4)
    assert np.allclose(data_add, expected_add)
    assert step_add == 14

    # Subtract
    subprocess.run(
        [
            "python",
            "scripts/avg_ops.py",
            str(file1),
            str(out_sub),
            "--subtract",
            str(file2),
        ],
        check=True,
    )
    data_sub, step_sub = read_field_data(out_sub)
    expected_sub = (2.0 * 10 - 1.0 * 4) / (10 - 4)
    assert np.allclose(data_sub, expected_sub)
    assert step_sub == 6
