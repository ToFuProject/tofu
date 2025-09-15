
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
import tofu as tf
import tofu.spectro as tfs


from .test_data._spectral_constraints import _DLINES_TOT, _DLINES
from .test_data._spectral_constraints import _DCONSTRAINTS, _DCONSTANTS
from .test_data._spectral_constraints import _DOMAIN, _FOCUS, _BINNING
from .test_data._spectral_constraints import _DSCALES, _DX0, _DBOUNDS


_here = os.path.abspath(os.path.dirname(__file__))
VerbHead = 'tofu.spectro.fit12d'
_PATH_TEST_DATA = os.path.join(_here, 'test_data')


# #############################################################################
#                           Default values
# #############################################################################


_DLINES_LOCAL = {
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


#######################################################
#
#        saving input spect
#       (To be run manually)
#######################################################


def _save_test_spect(path=_PATH_TEST_DATA, dlines=_DLINES_LOCAL, save=True):

    nlamb = 100
    lamb = np.linspace(3.94, 4, nlamb)*1e-10
    var = np.linspace(-25, 25, 51)
    dvar = var[-1] - var[0]

    def bck(lamb=None, offset=None, slope=None):
        return offset + (lamb-lamb.min())*slope/(lamb.max()-lamb.min())

    def gauss(lamb=None, lamb0=None, sigma=None, delta=None, amp=None):
        return amp * np.exp(-(lamb-lamb0-delta)**2/(2*sigma**2))

    def noise(lamb=None, amp=None, freq=None, phase=None):
        return amp*np.sin(
            lamb*freq*2.*np.pi/(lamb.max()-lamb.min()) + phase
        )

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
                1 + 2*(ii/len(dlines))*np.cos(var[:, None]*2*np.pi/dvar)
            ),
            delta=dlines[k0]['delta']*(
                1 + 2*(ii/len(dlines))*np.sin(
                    var[:, None]*2*np.pi*(len(dlines)-ii)/dvar
                )
            ),
        )
        spect2d += noise(
            lamb=lamb[None, :],
            amp=dlines[k0]['noise'] * np.exp(-var[:, None]**2/10**2),
            freq=10*(len(dlines)-ii),
            phase=ii,
        )

    # --------
    # save
    if save:
        pfe = os.path.join(path, 'spectral_fit.npz')
        np.savez(pfe, lamb=lamb, var=var, dlines=dlines, spect2d=spect2d)


def _load_test_spect(path=_PATH_TEST_DATA):
    pfe = os.path.join(path, 'spectral_fit.npz')
    out = dict(np.load(pfe, allow_pickle=True))
    return out['lamb'], out['var'], out['dlines'].tolist(), out['spect2d']


# #############################################################################
#
#     Creating Ves objects and testing methods
#
# #############################################################################


class Test01_ProofOfPrinciple(object):

    @classmethod
    def setup_class(cls):

        # load test data
        lamb, var, dlines, spect2d = _load_test_spect()
        mask = np.repeat((np.abs(var-15) > 3)[:, None], lamb.size, axis=1)

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

        # Define constraint dict
        defconst = {
            'amp': False,
            'width': False,
            'shift': False,
            'double': False,
            'symmetry': False,
        }

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
        ]

        ldx0 = [
            None,
            {
                # 'amp': {''},
                'width': 1.,
                'shift': {
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
            spect2d[5, :],
            spect2d[5:8, :],
        ]

        lfocus = [None, 'b', [3.94e-10, 3.96e-10]]

        ldconstants = [None, {'shift': {'s2': 0}}]

        cls.lamb = lamb
        cls.var = var
        cls.dlines = dlines
        cls.spect2d = spect2d
        cls.mask = mask
        cls.defconst = defconst
        cls.ldconst = ldconst
        cls.ldx0 = ldx0
        cls.ldomain = ldomain
        cls.ldata = ldata
        cls.lfocus = lfocus
        cls.ldconstants = ldconstants
        cls.ldinput1d = []
        cls.ldinput1d_run = []
        cls.ldfit1d = []
        cls.ldex1d = []
        cls.ldinput2d = []
        cls.ldinput2d_run = []
        cls.ldfit2d = []
        cls.ldex2d = []

    @classmethod
    def setup_method(self):
        pass

    def teardown_method(self):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def test01_fit1d_dinput(self, full=None):
        """ Build the input dict for fitting a 1d spectrum
        """

        combin = [
            self.ldconst, self.ldx0, self.ldomain, self.ldata,
            self.lfocus, self.ldconstants,
        ]
        nn = int(np.prod([len(cc) for cc in combin]))
        for ii, comb in enumerate(itt.product(*combin)):

            pos = ii % 2 == 0
            dinput = tfs.fit1d_dinput(
                dlines=self.dlines,
                dconstraints=comb[0],
                dconstants=comb[5],
                dprepare=None,
                data=np.copy(comb[3]),
                lamb=self.lamb,
                mask=None,
                domain=comb[2],
                pos=pos,
                subset=None,
                same_spectrum=None,
                nspect=None,
                same_spectrum_dlamb=None,
                focus=comb[4],
                valid_fraction=0.28,     # fraction of pixels ok per time step
                valid_nsigma=0,         # S/N ratio for each pixel
                focus_half_width=None,
                valid_return_fract=None,
                dscales=None,
                dx0=comb[1],
                dbounds=None,
                defconst=self.defconst,
            )
            self.ldinput1d.append(dinput)

            # run (fit1d) only for some cases
            run = False
            if comb[1] is None and comb[5] is not None and comb[4] != 'b':
                run = True
            self.ldinput1d_run.append(run)

    def test02_funccostjac_1d(self):
        """ check that tofu properly returns 3 functions for fitting 1d spectra

        func_detail: should return all components of a spectrum
        func_cost: shoud return only the total minus the original data
        func_jac: should return the jacobian

        in principle: sum(func_detail) == func_cost(data=0)

        """

        func = tfs._fit12d_funccostjac.multigausfit1d_from_dlines_funccostjac
        for ii, dd in enumerate(self.ldinput1d):
            func_detail, func_cost, func_jac = func(
                lamb=dd['dprepare']['lamb'],
                dinput=dd,
                dind=dd['dind'],
                jac='dense',
            )

            # x0
            x0 = tfs._fit12d_dinput._dict2vector_dscalesx0bounds(
                dd=dd['dx0'], dd_name='dx0', dinput=dd,
            )

            # scales
            scales = tfs._fit12d_dinput._dict2vector_dscalesx0bounds(
                dd=dd['dscales'], dd_name='dscales', dinput=dd,
            )

            # y0
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

    def test03_fit1d(self, strict=None, verb=None):
        """ Actually run the 1d spectrum fitting routine,

        """
        if verb:
            nprod = np.array(self.ldinput1d_run).sum()
            nn = len(f'\tspectrum {nprod} / {nprod}')

        for ii, inn in enumerate(np.array(self.ldinput1d_run).nonzero()[0]):

            if verb:
                msg = f"\tspectrum {ii+1} / {nprod}".ljust(nn)
                print(msg, end='\r', flush=True)

            chain = ii % 2 == 0
            dfit1d = tfs.fit1d(
                dinput=self.ldinput1d[inn],
                method=None,
                Ti=None,
                chain=chain,
                jac='dense',
                verbose=False,
                strict=strict,
                plot=False,
            )
            assert np.sum(dfit1d['validity'] < 0) == 0
            self.ldfit1d.append(dfit1d)

    def test04_fit1d_dextract(self):
        """ Extract dict of output from fitted 1d spectra
        """
        for ii, dd in enumerate(self.ldfit1d):
            dex = tfs.fit1d_extract(
                dfit1d=dd,
                ratio=('a', 'c'),
                sol_total=True,
                sol_detail=ii % 2 == 0,
            )
            self.ldex1d.append(dex)

    def test05_fit1d_plot(self, warn=True):
        lwar = []
        for ii, dd in enumerate(self.ldex1d):
            try:
                # For a yet unknown reason, this particular test crashes on
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
                if warn:
                    lwar.append((ii, str(err)))
                else:
                    raise err
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
        """ Build the input dict for fitting a 2d spectrum
        """
        combin = [
            self.ldconst, self.ldx0, self.ldomain,
            self.lfocus, self.ldconstants,
        ]
        ntot = (
            len(self.ldconst) * len(self.ldx0) * len(self.ldomain)
            * len(self.lfocus) * len(self.ldconstants)
        )
        for ii, comb in enumerate(itt.product(*combin)):

            # additional parameters
            pos = ii % 2 == 0
            mask = self.mask if ii % 3 == 0 else None
            binning = False if ii % 2 == 0 else {'lamb': 85, 'phi': 40}

            # set dinput
            dinput = tfs.fit2d_dinput(
                dlines=self.dlines,
                dconstraints=comb[0],
                dconstants=comb[4],
                dprepare=None,
                data=np.copy(self.spect2d),
                lamb=self.lamb,
                phi=self.var,
                mask=mask,
                nbsplines=5,
                domain=comb[2],
                pos=pos,
                subset=None,
                binning=binning,
                focus=comb[3],
                valid_fraction=0.28,     # fraction of pixels ok per time step
                valid_nsigma=0.2,         # S/N ratio for each pixel
                focus_half_width=None,
                valid_return_fract=None,
                dscales=None,
                dx0=comb[1],
                dbounds=None,
                defconst=self.defconst,
            )
            self.ldinput2d.append(dinput)

            c0 = (
                comb[1] == self.ldx0[0]
                and comb[2] == self.ldomain[1]
                and comb[3] == self.lfocus[0]
                and comb[4] == self.ldconstants[0]
            )
            if c0:
                self.ldinput2d_run.append(ii)

    def test07_plot_dinput2d(self):
        for ii, dd in enumerate(self.ldinput2d):
            dax = tfs._plot.plot_dinput2d(dinput=dd)
        plt.close('all')

    def test08_funccostjac_2d(self):
        """ check that tofu properly returns 3 functions for fitting 1d spectra

        func_detail: should return all components of a spectrum
        func_sum: shoud return the total spectrum
        func_cost: shoud return the total spectrum minus the original data
        func_jac: should return the jacobian

        in principle: sum(func_detail()) == func_sum() == func_cost(data=0)

        """
        func = tfs._fit12d_funccostjac.multigausfit2d_from_dlines_funccostjac
        for ii, dd in enumerate(self.ldinput2d):

            if ii < 24:
                continue

            lamb_flat = dd['dprepare']['lamb'].ravel()
            phi_flat = dd['dprepare']['phi'].ravel()
            data_flat = dd['dprepare']['data'][0, ...].ravel()
            func_detail, func_sum, func_cost, func_jac = func(
                phi_flat=phi_flat,
                dinput=dd,
                dind=dd['dind'],
                jac='dense',
            )

            # x0
            x0 = tfs._fit12d_dinput._dict2vector_dscalesx0bounds(
                dd=dd['dx0'], dd_name='dx0', dinput=dd,
            )

            # scales
            scales = tfs._fit12d_dinput._dict2vector_dscalesx0bounds(
                dd=dd['dscales'], dd_name='dscales', dinput=dd,
            )

            # lambrel qnd lambn
            lambrel_flat = lamb_flat - dd['lambmin_bck']
            lambn_flat = lamb_flat[..., None] / dd['lines'][None, :]

            # dy0 vs dy1
            y0 = func_detail(
                x0[0, :],
                lamb=lamb_flat,
                phi=phi_flat,
                scales=scales[0, :],
            )
            indnan = np.all(np.all(np.isnan(y0), axis=-1), axis=-1)
            dy0 = np.nansum(np.nansum(y0, axis=-1), axis=-1) - data_flat
            # beware, nansum returns 0 for all-nans in numpy >= 1.9.0
            dy0[indnan] = np.nan
            dy1 = func_sum(
                x0[0, :],
                lamb=lamb_flat,
                phi=phi_flat,
                scales=scales[0, :],
            ) - data_flat
            dy2 = func_cost(
                x0[0, :],
                scales=scales[0, :],
                phi_flat=phi_flat,
                lambrel_flat=lambrel_flat,
                lambn_flat=lambn_flat,
                data_flat=data_flat,
                indok_flat=np.ones((phi_flat.size,), dtype=bool)
            )
            # check consistency between func_detail and func_cost
            assert np.sum(np.isfinite(dy0)) == np.sum(np.isfinite(dy1))
            assert np.sum(np.isfinite(dy0)) == np.sum(np.isfinite(dy2))
            assert np.allclose(dy0, dy1, equal_nan=True)
            assert np.allclose(dy0, dy2, equal_nan=True)

    def test09_fit2d(self, strict=None, verb=False):
        """ Actually run the 2d spectrum fitting routine,

        """
        for ii, ij in enumerate(self.ldinput2d_run):
            din = self.ldinput2d[ij]
            chain = ii % 2 == 0

            dfit2d = tfs.fit2d(
                dinput=din,
                method=None,
                Ti=None,
                chain=chain,
                jac='dense',
                verbose=verb,
                strict=strict,
                plot=False,
            )
            assert np.sum(dfit2d['validity'] < 0) == 0
            self.ldfit2d.append(dfit2d)

    def test09_fit2d_dextract(self):
        """ Extract dict of output from fitted 2d spectra
        """
        for ii, dd in enumerate(self.ldfit2d):
            dex = tfs.fit2d_extract(
                dfit2d=dd,
                ratio=('a', 'c'),
                sol_total=True,
                sol_detail=ii % 2 == 0,
            )
            self.ldex2d.append(dex)

    def test10_fit2d_plot(self, warn=True):
        lwar = []
        for ii, dd in enumerate(self.ldex2d):
            try:
                # For a yet unknown reason, this particular test crashes on
                # Windows only due to figure creation at
                # tfs._plot.plot_fit1d(): line 337
                # already investigated: reducing figure size and early closing
                # No more ideas...
                # This link suggests it may have something to do with
                # inches => pixels conversion of figure size...
                # https://github.com/matplotlib/matplotlib/issues/14225
                if 'win' not in sys.platform.lower():
                    dax = tfs._plot.plot_fit2d(
                        dfit2d=self.ldfit2d[ii],
                        dextract=dd,
                        annotate=self.ldfit2d[ii]['dinput']['keys'][0],
                        fs=(4, 4),
                    )
            except Exception as err:
                if warn:
                    lwar.append((ii, str(err)))
                else:
                    raise err
            finally:
                plt.close('all')


class Test02_RealisticWESTCase(object):

    def test00_load_dinput_dfit_dextract(
        self,
        verb=None,
        # non-default parameters
        symmetry=None,
        mask=None,
        domain=None,
        dconstants=None,
        binning=None,
        dscales=None,
        dx0=None,
        dbounds=None,
        focus=None,
        valid_fraction=None,
        valid_nsigma=None,
        nbsplines=None,
        tol=None,
        # debug
        plot_dinput=False,
        vmin=None,
        vmax=None,
        # extract
        sol_detail=None,
        phi_prof=None,
        phi_npts=None,
    ):

        # load test data
        shot = 54046
        dout = dict(np.load(
            os.path.join(_PATH_TEST_DATA, f'west_{shot}_xics.npz'),
            allow_pickle=True,
        ))

        lk = ['data', 't', 'indt', 'nt', 'names', 'dunits', 'dbonus']
        data, t, indt, nt, names, dunits, dbonus = [dout[k0] for k0 in lk]
        dunits = dunits.tolist()
        dbonus = dbonus.tolist()

        # reshape data into a (nt, nxi, nxj) array
        nt = t.size
        nxi, nxj = dbonus['nH'], dbonus['nV']
        data = data.reshape((nt, nxj, nxi)).swapaxes(1, 2)

        # ----------------
        # Get crystal and det

        det = dict(np.load(
            os.path.join(_PATH_TEST_DATA, 'det37_CTVD_incC4_New.npz'),
            allow_pickle=True,
        ))

        cryst = tf.load(os.path.join(
            _PATH_TEST_DATA,
            'TFG_CrystalBragg_ExpWEST_DgXICS_ArXVII_sh00000_Vers1.5.0.npz',
        ))
        angle = 1.3124
        # tlim2 = [32, 46]
        cryst.move(angle*np.pi/180.)

        # get lamb / phi
        xi = 172e-6*(np.arange(0, 487) - (487 - 1)/2.)
        xj = 172e-6*(np.arange(0, 1467) - (1467 - 1)/2.)
        bragg, phi, lamb = cryst.get_lambbraggphi_from_ptsxixj_dthetapsi(
            det=det, xi=xi, xj=xj, grid=True,
        )

        # computed from a unique point on the crystal (the summit)
        lamb, bragg, phi = lamb[..., 0], bragg[..., 0], phi[..., 0]

        # ------------------------------------
        # Start data treatment: dict of inputs

        # get dict of constraints
        (
            dlines, dconstraints, dconstants, domain, focus, defconst,
            valid_fraction, valid_nsigma, binning,
            dscales, dx0, dbounds,
            nbsplines, deg, mask, tol,
        ) = get_constraints(
            shot=shot,
            cryst=cryst,
            det=det,
            # non-default parameters
            symmetry=symmetry,
            mask=mask,
            domain=domain,
            dconstants=dconstants,
            binning=binning,
            dscales=dscales,
            dx0=dx0,
            dbounds=dbounds,
            focus=focus,
            valid_fraction=valid_fraction,
            valid_nsigma=valid_nsigma,
            nbsplines=nbsplines,
            tol=tol,
        )

        # format into a unique input dict
        dinput = tf.spectro.fit2d_dinput(
            dlines=dlines,
            dconstraints=dconstraints,
            dconstants=dconstants,
            dprepare=None,
            deg=deg,
            nbsplines=nbsplines,
            knots=None,
            data=data,
            lamb=lamb,
            phi=phi,
            mask=mask,
            domain=domain,
            pos=None,
            subset=None,
            binning=binning,
            cent_fraction=None,
            focus=focus,
            valid_fraction=valid_fraction,
            valid_nsigma=valid_nsigma,
            focus_half_width=None,
            valid_return_fract=None,
            dscales=dscales,
            dx0=dx0,
            dbounds=dbounds,
            nxi=nxi,
            nxj=nxj,
            lphi=None,
            lphi_tol=None,
            defconst=defconst,
        )

        # --------------
        # Optional debug

        if plot_dinput is not False:
            ldax = tf.spectro._plot.plot_dinput2d(
                dinput=dinput,
                indspect=plot_dinput,
                vmin=vmin,
                vmax=vmax,
            )
            return dinput, ldax

        # ------------------------------------
        # Perform main data treatment: fit

        dfit2d = tf.spectro.fit2d(
            dinput=dinput,
            method=None,
            tr_solver=None,
            tr_options=None,
            xtol=tol,
            ftol=tol,
            gtol=tol,
            max_nfev=None,
            loss=None,
            chain=True,
            jac=None,
            verbose=verb,
            strict=True,    # to make sure a corrupted time step doesn't stop
            save=False,
            plot=False,
        )

        # ------------------------------------
        # Finish data treatment: extract quantities

        dextract = tf.spectro.fit2d_extract(
            dfit2d=dfit2d,
            bck=True,
            amp=True,
            ratio=[('ArXVII_w_Bruhns', 'ArXVI_k_Adhoc200408')],
            Ti=True,
            width=True,
            vi=True,
            shift=True,
            sol_total=True,
            sol_detail=sol_detail,
            sol_lamb_phi=None,
            phi_prof=phi_prof,
            phi_npts=phi_npts,
            vs_nbs=None,
        )

        return dinput, dfit2d, dextract, t


def get_constraints(
    shot=None,
    cryst=None,
    det=None,
    # non-default parameters
    symmetry=None,
    mask=None,
    domain=None,
    dconstants=None,
    binning=None,
    dscales=None,
    dx0=None,
    dbounds=None,
    focus=None,
    valid_fraction=None,
    valid_nsigma=None,
    nbsplines=None,
    tol=None,
):

    crystname = 'ArXVII'

    # dlines
    dlines = {
        k0: dict(_DLINES_TOT[k0])
        for k0 in _DLINES[crystname]
    }

    # dconstraints
    dconstraints = dict(_DCONSTRAINTS[crystname])
    if symmetry is not None:
        dconstraints['symmetry'] = symmetry

    # dconstants
    if dconstants is False:
        dconstants = None
    elif dconstants is None:
        dconstants = dict(_DCONSTANTS[crystname])

    # domain
    if domain is False:
        domain = None
    elif domain is None:
        if _DOMAIN[crystname] is not None:
            domain = dict(_DOMAIN[crystname])

    # binning
    if binning is False:
        binning = None
    elif binning is None:
        if _BINNING[crystname] is not None:
            binning = dict(_BINNING[crystname])

    # dscales
    if dscales is False:
        dscales = None
    elif dscales is None:
        if _DSCALES[crystname] is not None:
            dscales = dict(_DSCALES[crystname])

    # dx0
    if dx0 is False:
        dx0 = None
    elif dx0 is None:
        if _DX0[crystname] is not None:
            dx0 = dict(_DX0[crystname])

    # dbounds
    if dbounds is False:
        dbounds = False
    elif dbounds is None:
        if _DBOUNDS[crystname] is not None:
            dbounds = dict(_DBOUNDS[crystname])

    # focus
    if focus is False:
        focus = None
    elif focus is None:
        if _FOCUS[crystname] is not None:
            focus = _FOCUS[crystname]

    # valid
    if valid_fraction is None:
        valid_fraction = 0.6
    if valid_nsigma is None:
        valid_nsigma = 4

    # nbsplines
    if nbsplines is None:
        nbsplines = 15

    # defconst
    defconst = {
        'bck_amp': False, 'bck_rate': False, 'amp': False,
        'width': False, 'shift': False, 'double': False,
    }

    # deg
    deg = 2

    # mask
    # (nxj, nxi) array
    # False where corrupt pixels
    if mask is None:
        pfe = os.path.join(_PATH_TEST_DATA, '_mask_54041.npz')
        mask = dict(np.load(pfe, allow_pickle=True))
        mask = ~np.any(mask['ind'], axis=0).T

    # tol
    if tol is None:
        tol = 1.e-2

    return (
        dlines, dconstraints, dconstants, domain, focus, defconst,
        valid_fraction, valid_nsigma, binning,
        dscales, dx0, dbounds,
        nbsplines, deg, mask, tol,
    )
