# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.


import numpy as np
import matplotlib.pyplot as plt


import tofu as tf


# #############################################################################
# #############################################################################
#               Benchmark of Mesh2D routines
#                   High level routines
# #############################################################################


class Mesh2D_HighLevel:
    """ Benchmark the geometry-oriented routines

    In particular:
        - camera creation (ray-tracing)
        - solid angle computing

    """

    # -----------------------------
    # Attributes reckognized by asv

    # time before benchmark is killed
    timeout = 60

    # -------------------------------------------------------
    # Setup and teardown, run before / after benchmark methods

    def setup(self):

        # Load a configuration
        self.conf = tf.load_config('ITER')

        # input dict for a cam1d
        dcam1d = {
            'pinhole': [8.38/np.sqrt(2.), 8.38/np.sqrt(2.), 0.],
            'orientation': [-np.pi, 0., 0],
            'focal': 0.08,
            'sensor_nb': 100,
            'sensor_size': 0.3,
            'config': self.conf,
            'Diag': 'SXR',
            'Exp': 'WEST',
            'Name': 'cam1',
        }
        self.cam1d = tf.geom.utils.create_CamLOS1D(**dcam1d)

        # prepare input dict for a cam2d
        dcam2d = {
            'pinhole': [8.38, 0., 0.],
            'orientation': [-7*np.pi/8, np.pi/6, 0],
            'focal': 0.08,
            'sensor_nb': 400,
            'sensor_size': 0.2,
            'config': self.conf,
            'Diag': 'SXR',
            'Exp': 'WEST',
            'Name': 'cam2',
        }
        self.cam2d = tf.geom.utils.create_CamLOS1D(**dcam2d)

        # prepare input dict for mesh2d
        self.dmesh2drect = {
            'crop_poly': self.conf,
            'res': 0.10,
            'deg': 2,
        }
        # dmesh2dtri = {
            # 'knots': ,
            # 'cents': ,
            # 'key': 'tri1',
        # }
        self.mesh2d = tf.data.Mesh2D()
        self.mesh2d.add_mesh(key='rect1', **self.dmesh2drect)


    def teardown(self):
        pass

    # -------------------------------------
    # benchmarks methods for geometry tools

    # Mesh2D - rect
    def time_00_mesh2d_rect_bs2(self):
        self.mesh2d.add_mesh(key='temp', **self.dmesh2drect)

    def peakmem_00_mesh2d_rect_bs2(self):
        self.mesh2d.add_mesh(key='temp', **self.dmesh2drect)

    # Mesh2D - tri
    # def time_00_mesh2dtri_bs1(self):
        # self.mesh2d.add_mesh(**self.dmesh2dtri)

    # def peakmem_mesh2dtri_bs1(self):
        # self.mesh2d.add_mesh(**self.dmesh2dtri)

    # Geometry matrix - rect
    def time_02_geommatrix_rect(self):
        self.mesh2d.add_geometry_matrix(
            key='rect1-bs2', cam=self.cam1d, res=0.01, verb=False,
        )

    def peakmem_02_geommatrix_rect(self):
        self.mesh2d.add_geometry_matrix(
            key='rect1-bs2', cam=self.cam1d, res=0.01, verb=False,
        )
