# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.


import os


import numpy as np
import matplotlib.pyplot as plt


_PATH_HERE = os.path.dirname(__file__)
_PATH_TOFU = os.path.dirname(os.path.dirname(_PATH_HERE))

import tofu as tf


_PATH_TESTDATA_01 = os.path.join(
    os.path.dirname(_PATH_HERE),
    'tests',
    'tests01_geom',
    'test_data',
)
_PATH_TESTDATA_04 = os.path.join(
    os.path.dirname(_PATH_HERE),
    'tests',
    'tests04_spectro',
    'test_data',
)
_PFE_TESTDATA = os.path.join(_PATH_TESTDATA_04, 'spectral_fit.npz')
_PFE_DET = os.path.join(_PATH_TESTDATA_01, 'det37_CTVD_incC4_New.npz')
_PFE_CRYST = os.path.join(
    _PATH_TESTDATA_04,
    'TFG_CrystalBragg_ExpWEST_DgXICS_ArXVII_sh00000_Vers1.5.0.npz',
)


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
    timeout = 500
    repeat = (1, 10, 20.0)
    sample_time = 0.100

    # -------------------------------------------------------
    # Setup and teardown, run before / after benchmark methods

    def setup_cache(self):
        return 0

    def setup(self, out):
        """ run before each benchmark method, out from setup_cache  """
        out = dict(np.load(_PFE_TESTDATA, allow_pickle=True))
        self.lamb = out['lamb']
        self.var = out['var']
        self.dlines = out['dlines'].tolist()
        self.spect2d = out['spect2d']

        self.conf0 = tf.load_config('WEST-V0')
        self.cryst = tf.load(_PFE_CRYST)
        self.det = dict(np.load(_PFE_DET, allow_pickle=True))
        self.xixj_lim = [
            [-0.041882, 0.041882], 0.1 + 100*172.e-6*np.r_[-0.5, 0.5]
        ]

    def teardown(self, out):
        pass

    # -------------------------------------
    # benchmarks methods for geometry tools

    # Mesh2D - rect
    def time_00_fit1d(self, out):

        # Define constraint dict
        dconst = {
            'amp': {'a1': ['a', 'd']},
            'width': 'group',
            'shift': {
                'a': {'key': 's1', 'coef': 1., 'offset': 0.},
                'b': {'key': 's1', 'coef': 1., 'offset': 0.},
                'c': {'key': 's2', 'coef': 2., 'offset': 0.},
                'd': {'key': 's3', 'coef': 1., 'offset': 0.001e-10},
            },
            'double': True,
            'symmetry': False,
        }

        dx0 = {
            'width': 1.,
            'shift': {
                's1': 0.,
                's2': 1.,
            },
            'dratio': 0,
            'dshift': 0,
        }

        domain = {
            'lamb': [
                [3.94e-10, 3.952e-10],
                (3.95e-10, 3.956e-10),
                [3.96e-10, 4e-10],
            ],
        }

        data = self.spect2d[:5, :]
        pos = True
        focus = 'a'
        dconstants = {'shift': {'s1': 0}}

        dinput = tf.spectro.fit1d_dinput(
            dlines=self.dlines,
            dconstraints=dconst,
            dconstants=dconstants,
            dprepare=None,
            data=data,
            lamb=self.lamb,
            mask=None,
            domain=domain,
            pos=pos,
            subset=None,
            same_spectrum=None,
            nspect=None,
            same_spectrum_dlamb=None,
            focus=focus,
            valid_fraction=0.28,     # fraction of pixels ok per time step
            valid_nsigma=0,         # S/N ratio for each pixel
            focus_half_width=None,
            valid_return_fract=None,
            dscales=None,
            dx0=dx0,
            dbounds=None,
            # defconst=None,
        )

        dfit1d = tf.spectro.fit1d(
            dinput=dinput,
            method=None,
            Ti=None,
            chain=True,
            jac='dense',
            verbose=False,
            plot=False,
        )

    def time_01_fit2d(self, out):
        # Define constraint dict
        dconst = {
            'amp': {'a1': ['a', 'd']},
            'width': 'group',
            'shift': {
                'a': {'key': 's1', 'coef': 1., 'offset': 0.},
                'b': {'key': 's1', 'coef': 1., 'offset': 0.},
                'c': {'key': 's2', 'coef': 2., 'offset': 0.},
                'd': {'key': 's3', 'coef': 1., 'offset': 0.001e-10},
            },
            'double': False,
            'symmetry': False,
        }

        dx0 = {
            'width': 1.,
            'shift': {
                's1': 0.,
                's2': 1.,
            },
            'dratio': 0,
            'dshift': 0,
        }

        domain = {
            'lamb': [
                [3.94e-10, 3.952e-10],
                (3.95e-10, 3.956e-10),
                [3.96e-10, 4e-10],
            ],
        }

        data = self.spect2d
        pos = True
        focus = 'a'
        dconstants = {'shift': {'s1': 0}}

        dinput = tf.spectro.fit2d_dinput(
            dlines=self.dlines,
            dconstraints=dconst,
            dconstants=dconstants,
            dprepare=None,
            data=data,
            lamb=self.lamb,
            phi=np.arange(0, data.shape[0]),
            binning=None,
            mask=None,
            domain=domain,
            pos=pos,
            focus=focus,
            valid_fraction=0.28,     # fraction of pixels ok per time step
            valid_nsigma=0,         # S/N ratio for each pixel
            focus_half_width=None,
            valid_return_fract=None,
            dscales=None,
            dx0=dx0,
            dbounds=None,
            # defconst=None,
        )

        dfit2d = tf.spectro.fit2d(
            dinput=dinput,
            method=None,
            Ti=None,
            chain=True,
            jac='dense',
            xtol=1e-6,
            ftol=1e-6,
            gtol=1e-6,
            verbose=False,
            plot=False,
        )


    def time_02_get_plasmadomain_at_lamb(self, out):
        pts, lambok = self.cryst.get_plasmadomain_at_lamb(
            det=self.det,
            lamb=[3.94e-10, 4.e-10],
            res=[0.005, 0.005, 0.01],
            config=self.conf0,
            domain=[None, [-0.36, -0.22], [-4*np.pi/5., -np.pi/2.]],
            xixj_lim=self.xixj_lim,
            plot=False,
        )
