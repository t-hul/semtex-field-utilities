from unittest import TestCase
from semtex_fieldio.mesh import Mesh, MeshFileError
from pathlib import Path


class TestMeshFromFile(TestCase):
    def test_valid_mesh(self):
        mesh_path = Path(__file__).parent / "data" / "koal1.msh"
        mesh = Mesh.from_file(mesh_path)
        self.assertEqual(mesh.geometry.nr, 11)
        self.assertEqual(mesh.geometry.ns, 11)
        self.assertEqual(mesh.geometry.nz, 96)
        self.assertEqual(mesh.geometry.nel, 30)

        self.assertEqual(mesh.xy.shape, (mesh.geometry.nel,
                         mesh.geometry.ns, mesh. geometry.nr, 2))
        self.assertEqual(mesh.xy[0, 0, 0, 0], -3.14159)
        self.assertEqual(mesh.xy[10, 6, 7, 1], 0.3909139903384115)
        self.assertEqual(mesh.theta.shape, (mesh.geometry.nz + 1,))
        self.assertEqual(mesh.theta[37], 2.421644337142132)

    def test_wrong_header_nr(self):
        mesh_path = Path(__file__).parent / "data" / "koal1_wrong_header_nr.msh"
        self.assertRaisesRegex(
            MeshFileError, "Failed to read theta/Z data", Mesh.from_file, mesh_path)

    def test_wrong_header_ns(self):
        mesh_path = Path(__file__).parent / "data" / "koal1_wrong_header_ns.msh"
        self.assertRaisesRegex(MeshFileError, "Failed to read XY data",
                               Mesh.from_file, mesh_path)

    def test_wrong_header_nz(self):
        mesh_path = Path(__file__).parent / "data" / "koal1_wrong_header_nz.msh"
        self.assertRaisesRegex(
            MeshFileError, "Expected 81 theta values", Mesh.from_file, mesh_path)

    def test_wrong_header_nel(self):
        mesh_path = Path(__file__).parent / "data" / "koal1_wrong_header_nel.msh"
        self.assertRaisesRegex(MeshFileError, "Failed to read XY data",
                               Mesh.from_file, mesh_path)

    def test_truncated_mesh(self):
        mesh_path = Path(__file__).parent / "data" / "koal1_truncated.msh"
        self.assertRaisesRegex(
            MeshFileError, "xy values", Mesh.from_file, mesh_path)
