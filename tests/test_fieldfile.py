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

    ff_write = Fieldfile(str(fname), "w", header=header)
    ff_write._print_geometry_info()
    ff_write.write(data)

    ff_read = Fieldfile(str(fname), "r")
    ff_read._print_geometry_info()

    assert ff_read.hdr.session == "test_session"
    assert ff_read.hdr.geometry == geom
    assert ff_read.fields == fields
    assert ff_read.data.shape == (2, 4)

    assert np.allclose(ff_read["u"], [0, 1, 2, 3])
    assert np.allclose(ff_read["v"], [4, 5, 6, 7])

    elmt_data = ff_read.reshape_elementwise()
    assert elmt_data.shape == (2, 1, 1, 2, 2)


def test_read_checkpoint():
    fname = "./tests/data/koal1.1200.chk"

    ff_read = Fieldfile(fname, "r")
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
    assert ff_read.hdr.format == "binary"

    ff_write = Fieldfile(fname + "_copy", "w", header=ff_read.hdr)
    ff_write.write(ff_read.data)
