# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.


import os


import numpy as np
import matplotlib.pyplot as plt


import tofu as tf


_PATH_HERE = os.path.dirname(__file__)
_PATH_TESTDATA = os.path.join(
    os.path.dirname(_PATH_HERE),
    'tests',
    'tests06_mesh',
    'test_data',
)
_PFE_TESTDATA = os.path.join(_PATH_TESTDATA, 'mesh_triangular_WEST_eq.txt')


# #############################################################################
# #############################################################################
#               Benchmark of Mesh2D routines
#                   High level routines
# #############################################################################


class HighLevel:
    """ Benchmark the geometry-oriented routines

    In particular:
        - camera creation (ray-tracing)
        - solid angle computing

    """

    # -----------------------------
    # Attributes reckognized by asv

    # time before benchmark is killed
    timeout = 30
    repeat = (1, 10, 20.0)
    sample_time = 0.100

    # -------------------------------------------------------
    # Setup and teardown, run before / after benchmark methods

    def setup_cache(self):

        # prepare input dict for a cam2d
        dcam2d = {
            'pinhole': [8.38, 0., 0.],
            'orientation': [-7*np.pi/8, np.pi/6, 0],
            'focal': 0.08,
            'sensor_nb': 200,
            'sensor_size': 0.2,
            'Diag': 'SXR',
            'Exp': 'WEST',
            'Name': 'cam2',
        }

        # prepare input dict for mesh2d rect
        dmesh2drect = {
            'res': 0.10,
            'deg': 2,
            'key': 'rect1',
        }

        # prepare input dict for mesh2d tri
        out = np.loadtxt(_PFE_TESTDATA)
        dmesh2dtri = {
            'knots': out[1:int(out[0, 0])+1, :2],
            'cents': out[int(out[0, 0])+1:, :].astype(int),
            'deg': 1,
            'key': 'tri1',
        }

        return dcam2d, dmesh2drect, dmesh2dtri

    def setup(self, out):
        """ run before each benchmark method, out from setup_cache  """
        self.conf = tf.load_config('ITER')
        self.mesh2d = tf.data.Mesh2D()
        self.mesh2d.add_mesh(crop_poly=self.conf, **out[1])
        self.mesh2d.add_mesh(**out[2])
        self.cam = tf.geom.utils.create_CamLOS1D(config=self.conf, **out[0])

    def teardown(self, out):
        pass

    # -------------------------------------
    # benchmarks methods for geometry tools

    # Mesh2D - rect
    def time_00_mesh2d_rect_bs2(self, out):
        mesh = tf.data.Mesh2D()
        mesh.add_mesh(crop_poly=self.conf, **out[1])

    # Mesh2D - tri
    def time_01_mesh2dtri_bs1(self, out):
        mesh = tf.data.Mesh2D()
        mesh.add_mesh(**out[2])

    # Geometry matrix - rect
    def time_02_geommatrix_rect(self, out):
        self.mesh2d.add_geometry_matrix(
            key='rect1-bs2', cam=self.cam, res=0.01, verb=False,
        )
