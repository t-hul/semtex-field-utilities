from unittest import TestCase
from semtex_fieldio.mesh import Mesh

class TestMeshFromFile(TestCase):
    def test_valid_mesh(self):
        mesh_path = "tests/data/koal1.msh"
        mesh = Mesh.from_file(mesh_path)
        self.assertEqual(mesh.geometry.nr, 11)
        self.assertEqual(mesh.geometry.ns, 11)
        self.assertEqual(mesh.geometry.nz, 96)
        self.assertEqual(mesh.geometry.nel, 30)

        self.assertEqual(mesh.xy.shape, (mesh.geometry.nel, mesh.geometry.ns, mesh. geometry.nr, 2))
        self.assertEqual(mesh.xy[0,0,0,0], -3.14159)
        self.assertEqual(mesh.xy[10,6,7,1], 0.3909139903384115)
        self.assertEqual(mesh.theta.shape, (mesh.geometry.nz + 1,))
        self.assertEqual(mesh.theta[37], 2.421644337142132)
