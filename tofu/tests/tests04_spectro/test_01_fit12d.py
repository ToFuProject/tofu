
"""
This module contains tests for tofu.geom in its structured version
"""

# Built-in
import os
import sys
import warnings
import itertools as itt

# Standard
import numpy as np
import scipy.constants as scpct
import matplotlib.pyplot as plt

# tofu-specific
from tofu import __version__
import tofu.spectro as tfs

_here = os.path.abspath(os.path.dirname(__file__))
VerbHead = 'tofu.spectro.fit12d'


#######################################################
#
#     Setup and Teardown
#
#######################################################


def setup_module(module):
    print("")   # this is to get a newline after the dots
    LF = os.listdir(_here)
    lss = ['TFD_', 'Test', '.npz']
    LF = [lf for lf in LF if all([ss in lf for ss in lss])]
    LF = [
        lf for lf in LF
        if not lf[lf.index('_Vv')+2:lf.index('_U')] == __version__
    ]
    print("Removing the following previous test files:")
    print(LF)
    for lf in LF:
        os.remove(os.path.join(_here, lf))
    # print("setup_module before anything in this file")


def teardown_module(module):
    # os.remove(VesTor.Id.SavePath + VesTor.Id.SaveName + '.npz')
    # os.remove(VesLin.Id.SavePath + VesLin.Id.SaveName + '.npz')
    # print("teardown_module after everything in this file")
    # print("") # this is to get a newline
    LF = os.listdir(_here)
    lss = ['TFD_', 'Test', '.npz']
    LF = [lf for lf in LF if all([ss in lf for ss in lss])]
    LF = [
        lf for lf in LF
        if lf[lf.index('_Vv')+2:lf.index('_U')] == __version__
    ]
    print("Removing the following test files:")
    print(LF)
    for lf in LF:
        os.remove(os.path.join(_here, lf))
    pass


# def my_setup_function():
#    print ("my_setup_function")

# def my_teardown_function():
#    print ("my_teardown_function")

# @with_setup(my_setup_function, my_teardown_function)
# def test_numbers_3_4():
#    print 'test_numbers_3_4  <============================ actual test code'
#    assert multiply(3,4) == 12

# @with_setup(my_setup_function, my_teardown_function)
# def test_strings_a_3():
#    print 'test_strings_a_3  <============================ actual test code'
#    assert multiply('a',3) == 'aaa'


#######################################################
#
#     Creating Ves objects and testing methods
#
#######################################################


class Test01_DataCollection(object):

    @classmethod
    def setup_class(cls):

        nlamb = 100
        lamb = np.linspace(3.94, 4, nlamb)*1e-10
        var = np.linspace(-25, 25, 51)

        def bck(lamb=None, offset=None, slope=None):
            return offset + (lamb-lamb.min())*slope/(lamb.max()-lamb.min())

        def gauss(lamb=None, lamb0=None, sigma=None, delta=None, amp=None):
            return amp * np.exp(-(lamb-lamb0-delta)**2/(2*sigma**2))

        def noise(lamb=None, amp=None, freq=None, phase=None):
            return amp*np.sin(
                lamb*freq*2.*np.pi/(lamb.max()-lamb.min()) + phase
            )

        dlines = {
            'a': {
                'lambda0': 3.95e-10,
                'delta': 0.001e-10,
                'sigma': 0.002e-10,
                'amp': 1.,
                'noise': 0.01,
                'group': 0,
            },
            'b': {
                'lambda0': 3.97e-10,
                'delta': -0.001e-10,
                'sigma': 0.0015e-10,
                'amp': 0.5,
                'noise': 0.001,
                'group': 1,
            },
            'c': {
                'lambda0': 3.975e-10,
                'delta': 0.001e-10,
                'sigma': 0.001e-10,
                'amp': 0.6,
                'noise': 0.005,
                'group': 0,
            },
            'd': {
                'lambda0': 3.99e-10,
                'delta': 0.002e-10,
                'sigma': 0.002e-10,
                'amp': 0.8,
                'noise': 0.01,
                'group': 1,
            },
        }

        spect2d = bck(
            lamb=lamb[None, :],
            offset=0.1*np.exp(-(var[:, None]-25)**2/10**2),
            slope=0.001,
        )
        spect2d += noise(lamb=lamb[None, :], amp=0.01, freq=10, phase=0.)

        for ii, k0 in enumerate(dlines.keys()):
            spect2d += gauss(
                lamb=lamb[None, :],
                amp=dlines[k0]['amp'] * np.exp(-var[:, None]**2/20**2),
                lamb0=dlines[k0]['lambda0'],
                sigma=dlines[k0]['sigma']*(
                    1 + 2*(ii/len(dlines))*np.cos(var[:, None]*2*np.pi/50)
                ),
                delta=dlines[k0]['delta']*(
                    1 + 2*(ii/len(dlines))*np.sin(
                        var[:, None]*2*np.pi*(len(dlines)-ii)/50
                    )
                ),
            )
            spect2d += noise(
                lamb=lamb[None, :],
                amp=dlines[k0]['noise'] * np.exp(-var[:, None]**2/10**2),
                freq=10*(len(dlines)-ii),
                phase=ii,
            )

        mask = np.repeat((np.abs(var-15) < 3)[:, None], nlamb, axis=1)

        # Plot spect 2d
        # fig = plt.figure(figsize=(12, 10));
        # ax0 = fig.add_axes([0.05, 0.1, 0.4, 0.8])
        # ax0.set_xlabel(r'$\lambda$ (m)')
        # ax0.set_ylabel(r'$y$ (m)')
        # extent = (lamb.min(), lamb.max(), var.min(), var.max())
        # ax0.imshow(spect2d, extent=extent, aspect='auto', origin='lower');
        # sp2bis = np.copy(spect2d)
        # sp2bis[mask] = np.nan
        # ax1 = fig.add_axes([0.55, 0.1, 0.4, 0.8])
        # ax1.set_xlabel(r'$\lambda$ (m)')
        # ax1.set_ylabel(r'$y$ (m)')
        # ax1.imshow(sp2bis, extent=extent, aspect='auto', origin='lower');
        # plt.ion()
        # plt.show();
        # import pdb; pdb.set_trace()     # DB

        cls.lamb = lamb
        cls.var = var
        cls.dlines = dlines
        cls.spect2d = spect2d
        cls.mask = mask
        cls.ldinput1d = []
        cls.ldfit1d = []
        cls.ldex1d = []
        cls.ldinput2d = []
        cls.ldfit2d = []
        cls.ldex2d = []

    @classmethod
    def setup(self):
        pass

    def teardown(self):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def test01_fit1d_dinput(self):
        defconst = {
            'amp': False,
            'width': False,
            'shift': False,
            'double': False,
            'symmetry': False,
        }

        # Define constraint dict
        ldconst = [
            {
                'amp': {'a1': ['a', 'd']},
                'width': 'group',
                'shift': {
                    'a': {'key': 's1', 'coef': 1., 'offset': 0.},
                    'b': {'key': 's1', 'coef': 1., 'offset': 0.},
                    'c': {'key': 's2', 'coef': 2., 'offset': 0.},
                    'd': {'key': 's3', 'coef': 1., 'offset': 0.001e-10},
                },
                'double': True,
                'symmetry': True,
            },
            {
                'amp': False,
                'width': 'group',
                'shift': {
                    'a': {'key': 's1', 'coef': 1., 'offset': 0.},
                    'c': {'key': 's2', 'coef': 2., 'offset': 0.},
                    'd': {'key': 's2', 'coef': 1., 'offset': 0.001e-10},
                },
                'double': False,
                'symmetry': False,
            },
            {
                'amp': False,
                'width': 'group',
                'shift': {
                    'a': {'key': 's1', 'coef': 1., 'offset': 0.},
                    'c': {'key': 's2', 'coef': 2., 'offset': 0.},
                    'd': {'key': 's2', 'coef': 1., 'offset': 0.001e-10},
                },
                'double': False,
                'symmetry': False,
            },
        ]

        ldx0 = [
            None,
            {
                # 'amp': {''},
                'width': 1.,
                'shift': {
                    's1': 0.,
                    's2': 1.,
                },
                'dratio': 0,
                'dshift': 0,
            }
        ]

        ldomain = [
            None,
            {
                'lamb': [
                    [3.94e-10, 3.952e-10],
                    (3.95e-10, 3.956e-10),
                    [3.96e-10, 4e-10],
                ],
            },
        ]

        ldata = [
            self.spect2d[5, :],
            self.spect2d[:5, :],
        ]
        lpos = [False, True]

        lfocus = [None, 'a', [3.94e-10, 3.96e-10]]

        ldconstants = [None, {'shift': {'s1': 0}}]

        combin = [ldconst, ldx0, ldomain, ldata, lpos, lfocus, ldconstants]
        for comb in itt.product(*combin):
            dinput = tfs.fit1d_dinput(
                dlines=self.dlines,
                dconstraints=comb[0],
                dconstants=comb[6],
                dprepare=None,
                data=np.copy(comb[3]),
                lamb=self.lamb,
                mask=None,
                domain=comb[2],
                pos=comb[4],
                subset=None,
                same_spectrum=None,
                nspect=None,
                same_spectrum_dlamb=None,
                focus=comb[5],
                valid_fraction=0.28,     # fraction of pixels ok per time step
                valid_nsigma=0,         # S/N ratio for each pixel
                focus_half_width=None,
                valid_return_fract=None,
                dscales=None,
                dx0=comb[1],
                dbounds=None,
                defconst=defconst,
            )
            self.ldinput1d.append(dinput)

    def test02_funccostjac_1d(self):
        func = tfs._fit12d_funccostjac.multigausfit1d_from_dlines_funccostjac
        for ii, dd in enumerate(self.ldinput1d):
            func_detail, func_cost, func_jac = func(
                lamb=dd['dprepare']['lamb'], dinput=dd,
                dind=dd['dind'], jac='dense',
            )

            # Get x0
            x0 = tfs._fit12d._dict2vector_dscalesx0bounds(
                dd=dd['dx0'], dd_name='dx0', dinput=dd,
            )
            scales = tfs._fit12d._dict2vector_dscalesx0bounds(
                dd=dd['dscales'], dd_name='dscales', dinput=dd,
            )

            y0 = func_detail(x0[0, :], scales=scales[0, :])
            y1 = func_cost(
                x0[0, :],
                scales=scales[0, :],
                data=dd['dprepare']['data'][0, :],
            )

            # check consistency between func_detail and func_cost
            assert np.allclose(
                np.sum(y0, axis=1) - dd['dprepare']['data'][0, :],
                y1,
                equal_nan=True,
            )

    def test03_fit1d(self):
        lchain = [False, True]
        for ii, dd in enumerate(itt.product(self.ldinput1d, lchain)):
            dfit1d = tfs.fit1d(
                dinput=dd[0],
                method=None,
                Ti=None,
                chain=dd[1],
                jac='dense',
                verbose=None,
                plot=False,
            )
            assert np.sum(dfit1d['validity'] < 0) == 0
            self.ldfit1d.append(dfit1d)

    def test04_fit1d_dextract(self):
        for ii, dd in enumerate(self.ldfit1d):
            dex = tfs.fit1d_extract(
                dfit1d=dd,
                ratio=('a', 'c'),
                pts_lamb_detail=True,
            )
            self.ldex1d.append(dex)

    def test05_fit1d_plot(self):
        lwar = []
        for ii, dd in enumerate(self.ldex1d):
            try:
                # For a yet unknown reason, this particular test crahses on
                # Windows only due to figure creation at
                # tfs._plot.plot_fit1d(): line 337
                # already investigated: reducing figure size and early closing
                # No more ideas...
                # This link suggests it may have something to do with 
                # inches => pixels conversion of figure size...
                # https://github.com/matplotlib/matplotlib/issues/14225
                if 'win' not in sys.platform.lower():
                    dax = tfs._plot.plot_fit1d(
                        dfit1d=self.ldfit1d[ii],
                        dextract=dd,
                        annotate=self.ldfit1d[ii]['dinput']['keys'][0],
                        fs=(4, 4),
                    )
            except Exception as err:
                lwar.append((ii, str(err)))
            finally:
                plt.close('all')

        if len(lwar) > 0:
            msg = (
                "\nThe ({}/{}) following fit1d plots failed:\n".format(
                    len(lwar), len(self.ldex1d),
                )
                + "\n".join(["\t- {}: {}".format(ww[0], ww[1]) for ww in lwar])
            )
            warnings.warn(msg)

    def test06_fit2d_dinput(self):
        defconst = {
            'amp': False,
            'width': False,
            'shift': False,
            'double': False,
            'symmetry': False,
        }

        # Define constraint dict
        ldconst = [
            {
                'amp': {'a1': ['a', 'd']},
                'width': 'group',
                'shift': {
                    'a': {'key': 's1', 'coef': 1., 'offset': 0.},
                    'b': {'key': 's1', 'coef': 1., 'offset': 0.},
                    'c': {'key': 's2', 'coef': 2., 'offset': 0.},
                    'd': {'key': 's3', 'coef': 1., 'offset': 0.001e-10},
                },
                'double': True,
                'symmetry': True,
            },
            {
                'amp': False,
                'width': 'group',
                'shift': {
                    'a': {'key': 's1', 'coef': 1., 'offset': 0.},
                    'c': {'key': 's2', 'coef': 2., 'offset': 0.},
                    'd': {'key': 's2', 'coef': 1., 'offset': 0.001e-10},
                },
                'double': False,
                'symmetry': False,
            },
            {
                'amp': False,
                'width': 'group',
                'shift': {
                    'a': {'key': 's1', 'coef': 1., 'offset': 0.},
                    'c': {'key': 's2', 'coef': 2., 'offset': 0.},
                    'd': {'key': 's2', 'coef': 1., 'offset': 0.001e-10},
                },
                'double': False,
                'symmetry': False,
            },
        ]

        ldx0 = [
            None,
            {
                # 'amp': {''},
                'width': 1.,
                'shift': {
                    's1': 0.,
                    's2': 1.,
                },
                'dratio': 0,
                'dshift': 0,
            }
        ]

        ldomain = [
            None,
            {
                'lamb': [
                    [3.94e-10, 3.952e-10],
                    (3.95e-10, 3.956e-10),
                    [3.96e-10, 4e-10],
                ],
            },
        ]

        lpos = [False, True]
        lmask = [None, self.mask]

        lfocus = [None, 'a', [3.94e-10, 3.96e-10]]

        ldconstants = [None, {'shift': {'s1': 0}}]

        combin = [ldconst, ldx0, ldomain, lmask, lpos, lfocus, ldconstants]
        for comb in itt.product(*combin):
            dinput = tfs.fit2d_dinput(
                dlines=self.dlines,
                dconstraints=comb[0],
                dconstants=comb[6],
                dprepare=None,
                data=np.copy(self.spect2d),
                lamb=self.lamb,
                phi=self.var,
                mask=comb[3],
                nbsplines=5,
                domain=comb[2],
                pos=comb[4],
                subset=None,
                focus=comb[5],
                valid_fraction=0.28,     # fraction of pixels ok per time step
                valid_nsigma=0,         # S/N ratio for each pixel
                focus_half_width=None,
                valid_return_fract=None,
                dscales=None,
                dx0=comb[1],
                dbounds=None,
                defconst=defconst,
            )
            self.ldinput2d.append(dinput)

    """
    def test07_funccostjac_2d(self):
        func = tfs._fit12d_funccostjac.multigausfit2d_from_dlines_funccostjac
        for ii, dd in enumerate(self.ldinput2d):
            func_detail, func_cost, func_jac = func(
                lamb=dd['dprepare']['lamb'],
                phi=dd['dprepare']['phi'],
                dinput=dd,
                dind=dd['dind'], jac='dense',
            )

            # Get x0
            x0 = tfs._fit12d._dict2vector_dscalesx0bounds(
                dd=dd['dx0'], dd_name='dx0', dinput=dd,
            )
            scales = tfs._fit12d._dict2vector_dscalesx0bounds(
                dd=dd['dscales'], dd_name='dscales', dinput=dd,
            )

            y0 = func_detail(x0[0, :], scales=scales[0, :])
            y1 = func_cost(
                x0[0, :],
                scales=scales[0, :],
                data=dd['dprepare']['data'][0, :],
            )

            # check consistency between func_detail and func_cost
            assert np.allclose(
                np.sum(y0, axis=1) - dd['dprepare']['data'][0, :],
                y1,
                equal_nan=True,
            )
    """
