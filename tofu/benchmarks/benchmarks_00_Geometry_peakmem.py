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
    timeout = 60
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
            'sensor_nb': 300,
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

        # time for signal
        t = np.linspace(0, 10, 11)

        return dcam1d, dcam2d, dpart, t

    def setup(self, out):
        """ run before each benchmark method, out from setup_cache  """
        self.conf = tf.load_config('ITER')
        self.ves = self.conf.Ves.InnerV0

        def emiss(pts, t=None):
            R = np.hypot(pts[0, :], pts[1, :])
            Z = pts[2, :]
            ee = np.exp(-(R-6)**2/1**2 - Z**2/2**2)
            if np.isscalar(t):
                ee = ee + 0.1*np.cos(t)*ee
            elif isinstance(t, np.ndarray):
                ee = ee[None, :] + 0.1*np.cos(t)[:, None]*ee
            return ee

        self.cam2d = tf.geom.utils.create_CamLOS2D(config=self.conf, **out[1])
        self.emiss = emiss

    def teardown(self, out):
        """ run after each benchmark method, out from setup_cache  """
        pass

    # -------------------------------------
    # benchmarks methods for geometry tools

    # conf sampling
    def peakmem_00_conf_sample_cross(self, out):
        ptsRZ, dS, ind, reseff = self.ves.get_sampleCross(
            res=0.02,
            domain=[None, [-2, 2]],
            resMode='abs',
            ind=None,
            mode='flat',        # imshow bugged?
        )

    # conf sampling
    def peakmem_01_conf_sample_surface(self, out):
        pts, dS, ind, [reseff_cross, reseff_phi] = self.ves.get_sampleS(
            res=0.02,
            domain=[None, None, [0, np.pi/2.]],
            resMode='abs',
            ind=None,
            offsetIn=0.,
            returnas='(X, Y, Z)',
            Ind=None,
        )

    # conf sampling
    def peakmem_02_conf_sample_volume(self, out):
        pts, dV, ind, [resR, resZ, resPhi] = self.ves.get_sampleV(
            res=0.05,
            domain=[None, [0, None], [0, np.pi/4.]],
            resMode='abs',
            ind=None,
            returnas='(X, Y, Z)',
            algo='new',
        )

    # CAMLOS1D
    def peakmem_03_camlos1d(self, out):
        cam = tf.geom.utils.create_CamLOS1D(config=self.conf, **out[0])

    # CAMLOS2D
    def peakmem_04_camlos2d(self, out):
        cam = tf.geom.utils.create_CamLOS2D(config=self.conf, **out[1])

    # calc_signal
    def peakmem_05_camlos2d_calcsignal_calls(self, out):
        sig, units = self.cam2d.calc_signal(
            self.emiss,
            t=out[3],
            res=0.01,
            resMode='abs',
            method='sum',
            minimize='calls',
            plot=False,
        )

    # calc_signal
    def peakmem_06_camlos2d_calcsignal_hybrid(self, out):
        sig, units = self.cam2d.calc_signal(
            self.emiss,
            t=out[3],
            res=0.01,
            resMode='abs',
            method='sum',
            minimize='hybrid',
            plot=False,
        )

    # calc_signal
    def peakmem_07_camlos2d_calcsignal_mem(self, out):
        sig, units = self.cam2d.calc_signal(
            self.emiss,
            t=out[3],
            res=0.01,
            resMode='abs',
            method='sum',
            minimize='memory',
            plot=False,
        )

    # Solid angle toroidal integral for particle
    def peakmem_08_solidangle_part(self, out):
        (
            ptsRZ, sang, indices, reseff,
        ) = self.conf.calc_solidangle_particle_integrated(**out[2])
