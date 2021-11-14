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
        """ setup_cache caches setup data and is un only once for all

        It should either return data or save a file
        Data returned is fed to setup(), teardown() and all benchmarks

        """

        # prepare input dict for a cam1d
        dcam1d = {
            'pinhole': [8.38/np.sqrt(2.), 8.38/np.sqrt(2.), 0.],
            'orientation': [-np.pi, 0., 0],
            'focal': 0.08,
            'sensor_nb': 100,
            'sensor_size': 0.3,
            'Diag': 'SXR',
            'Exp': 'WEST',
            'Name': 'cam1',
        }

        # prepare input dict for a cam2d
        dcam2d = {
            'pinhole': [8.38, 0., 0.],
            'orientation': [-7*np.pi/8, np.pi/6, 0],
            'focal': 0.08,
            'sensor_nb': 400,
            'sensor_size': 0.2,
            'Diag': 'SXR',
            'Exp': 'WEST',
            'Name': 'cam2',
        }

        # prepare input dict for particle solid angle toroidal integration
        dpart = {
            'part_traj': np.array([
                [6., 0., 0.], [6., 0.01, -4],
            ]).T,
            'part_radius': np.array([10e-6, 10e-6]),
            'resolution': 0.3,
            'DPhi': [-np.pi/2, np.pi/2],
            'vmax': False,
            'approx': False,
            'plot': False,
        }
        return dcam1d, dcam2d, dpart

    def setup(self, out):
        """ run before each benchmark method, out from setup_cache  """
        self.conf = tf.load_config('ITER')

    def teardown(self, out):
        """ run after each benchmark method, out from setup_cache  """
        pass

    # -------------------------------------
    # benchmarks methods for geometry tools

    # CAMLOS1D
    def peakmem_camlos1d(self, out):
        cam = tf.geom.utils.create_CamLOS1D(config=self.conf, **out[0])

    # CAMLOS2D
    def peakmem_camlos2d(self, out):
        cam = tf.geom.utils.create_CamLOS2D(config=self.conf, **out[1])

    # Solid angle toroidal integral for particle
    def peakmem_solidangle_part(self, out):
        (
            ptsRZ, sang, indices, reseff,
        ) = self.conf.calc_solidangle_particle_integrated(**out[2])
