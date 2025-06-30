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
