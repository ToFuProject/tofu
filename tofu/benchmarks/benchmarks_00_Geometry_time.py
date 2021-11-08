# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.


import numpy as np
import matplotlib.pyplot as plt


import tofu as tf


# #############################################################################
# #############################################################################
#               Benchmark of geometry routines
#                   High level routines
# #############################################################################


class Geometry_HighLevel:
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

        # prepare input dict for a cam1d
        self.dcam1d = {
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

        # prepare input dict for a cam2d
        self.dcam2d = {
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

        # prepare input dict for particle solid angle toroidal integration
        self.dpart = {
            'part_traj': np.array([
                [6., 0., 0.], [6., 0.01, -4],
            ]).T,
            'part_radius': np.array([10e-6, 10e-6]),
            'resolution': 0.3,
            'DPhi': [-np.pi/2, np.pi/2],
            'vmax': False,
            'approx': False,
        }

    def teardown(self):
        pass

    # -------------------------------------
    # benchmarks methods for geometry tools

    # CAMLOS1D
    def time_camlos1d(self):
        cam = tf.geom.utils.create_CamLOS1D(**self.dcam1d)

    # CAMLOS2D
    def time_camlos2d(self):
        cam = tf.geom.utils.create_CamLOS2D(**self.dcam2d)

    # Solid angle toroidal integral for particle
    def time_solidangle_part(self):
        (
            ptsRZ, sang, indices, reseff, dax,
        ) = self.conf.calc_solidangle_particle_integrated(**self.dpart)
        plt.close('all')
