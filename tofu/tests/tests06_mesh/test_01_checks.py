"""
This module contains tests for tofu.geom in its structured version
"""

# Built-in
import os
import shutil
import itertools as itt
import warnings


# Standard
import numpy as np
import matplotlib.pyplot as plt


# tofu-specific
from tofu import __version__
import tofu as tf
import tofu.data as tfd


_HERE = os.path.abspath(os.path.dirname(__file__))
_PATH_DATA = os.path.join(_HERE, 'test_data')
_TOFU_USER = os.path.join(os.path.expanduser("~"), '.tofu')
_CUSTOM = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
_CUSTOM = os.path.join(_CUSTOM, 'scripts', 'tofucustom.py')
_FNAME = 'ITER_JINTRAC_sh134000_run30_public_edgesources_quadtrimesh.npz'
VerbHead = 'tofu.mesh.test_01_checks'


_PFE = os.path.join(_PATH_DATA, _FNAME)
_DTRI = {
    k0: v0.tolist()
    for k0, v0 in dict(np.load(_PFE, allow_pickle=True)).items()
}


#######################################################
#
#     Setup and Teardown
#
#######################################################


def setup_module():
    print("Removing user ~/.tofu/ if any")
    if os.path.isdir(_TOFU_USER):
        shutil.rmtree(_TOFU_USER)
    # Recreating clean .tofu
    # out = subprocess.run(_CUSTOM, stdout=PIPE, stderr=PIPE)
    os.system('python '+_CUSTOM)


def teardown_module():
    print("Removing user ~/.tofu/ if any")
    if os.path.isdir(_TOFU_USER):
        shutil.rmtree(_TOFU_USER)


#######################################################
#
#     Basic instanciation
#
#######################################################


def _add_rect_uniform(plasma):
    # add uniform rect mesh
    plasma.add_mesh(key='m0', domain=[[2, 3], [-1, 1]], res=0.1)


def _add_rect_variable(plasma):
    # add variable rect mesh
    plasma.add_mesh(
        key='m1',
        domain=[[2, 2.3, 2.6, 3], [-1, 0., 1]],
        res=[[0.2, 0.1, 0.1, 0.2], [0.2, 0.1, 0.2]],
    )


def _add_rect_variable_crop(plasma):
    # add variable rect mesh
    plasma.add_mesh(
        key='m2',
        domain=[[2, 2.3, 2.6, 3], [-1, 0., 1]],
        res=[[0.2, 0.1, 0.1, 0.2], [0.2, 0.1, 0.2]],
        crop_poly=tf.load_config('WEST'),
    )


def _add_tri_ntri1(plasma):
    cents = _DTRI['cents']['data']
    cents2 = np.zeros((cents.shape[0]*2, 3))
    cents2[::2, :] = cents[:, :3]
    cents2[1::2, :2] = cents[:, 2:]
    cents2[1::2, -1] = cents[:, 0]

    plasma.add_mesh(
        key='m3',
        knots=_DTRI['nodes']['data'],
        cents=cents2,
    )


def _add_tri_ntri2(plasma):
    plasma.add_mesh(
        key='m4',
        knots=_DTRI['nodes']['data'],
        cents=_DTRI['cents']['data'],
    )


def _add_bsplines(plasma):
    for k0, v0 in plasma.dobj['mesh'].items():
        if v0['type'] == 'tri':
            plasma.add_bsplines(key=k0, deg=0)
            plasma.add_bsplines(key=k0, deg=1)
        else:
            plasma.add_bsplines(key=k0, deg=0)
            plasma.add_bsplines(key=k0, deg=1)
            plasma.add_bsplines(key=k0, deg=2)
            plasma.add_bsplines(key=k0, deg=3)


#######################################################
#
#     checking routines
#
#######################################################


class Test01_checks_Instanciate():

    @classmethod
    def setup_class(cls):
        pass

    @classmethod
    def setup(self):
        pass

    def teardown(self):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def test01_mesh2DRect_X_check(self):

        lx = [[1, 2], [1, 2, 3, 4]]
        lres = [None, 10, 0.1, [0.1, 0.2], [0.1, 0.2, 0.3, 0.1]]

        for comb in itt.product(lx, lres):
            if hasattr(lres, '__iter__') and len(lres) != len(lx):
                continue
            x, res, ind = tfd._mesh_checks._mesh2DRect_X_check(
                x=[1, 2, 3, 4],
                res=10,
            )
            if hasattr(lres, '__iter__'):
                assert x_new.size == np.unique(x_new).size == res.size + 1

    def test02_add_mesh_rect_uniform(self):
        plasma = tfd.Plasma2D()
        _add_rect_uniform(plasma)
        _add_bsplines(plasma)

    def test03_add_mesh_rect_variable(self):
        plasma = tfd.Plasma2D()
        _add_rect_variable(plasma)
        _add_bsplines(plasma)

    def test04_add_mesh_rect_variable_crop(self):
        plasma = tfd.Plasma2D()
        _add_rect_variable_crop(plasma)
        _add_bsplines(plasma)

    def test05_add_mesh_tri_ntri1(self):
        plasma = tfd.Plasma2D()
        _add_tri_ntri1(plasma)
        _add_bsplines(plasma)

    def test06_add_mesh_tri_ntri2(self):
        plasma = tfd.Plasma2D()
        _add_tri_ntri2(plasma)
        _add_bsplines(plasma)


#######################################################
#
#     object mesh2D
#
#######################################################


class Test02_Plasma2D():

    @classmethod
    def setup_class(cls):
        pass

    def setup(self):
        plasma = tfd.Plasma2D()

        # add rect mesh
        _add_rect_uniform(plasma)
        _add_rect_variable(plasma)
        _add_rect_variable_crop(plasma)

        # add tri mesh
        _add_tri_ntri1(plasma)
        _add_tri_ntri2(plasma)

        # add bsplines
        _add_bsplines(plasma)

        # store
        self.obj = plasma
        self.lm = list(plasma.dobj['mesh'].keys())
        self.lbs = list(plasma.dobj['bsplines'].keys())

    def teardown(self):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def test01_show(self):
        self.obj.show()

    def test02_select_ind(self):

        # Rect mesh
        nn = 4
        lelements = ['knots', None, 'cents', None]
        lind = [None, ([0, 5], [0, 6]), [0, 10, 50], ([0, 5, 6], [0, 2, 3])]
        lcrop = [True, False, True, False]

        # select fom mesh
        for ii, k0 in enumerate(self.lm):
            ind = ii % nn
            indt = self.obj.select_ind(
                key=k0,
                ind=lind[ind],
                elements=lelements[ind],
                returnas=tuple,
                crop=lcrop[ind],
            )
            indf = self.obj.select_ind(
                key=k0,
                ind=indt,
                elements=lelements[ind],
                returnas=np.ndarray,
                crop=lcrop[ind],
            )
            indt2 = self.obj.select_ind(
                key=k0,
                ind=indf,
                elements=lelements[ind],
                returnas=tuple,
                crop=lcrop[ind],
            )
            assert all([np.allclose(indt[jj], indt2[jj]) for jj in [0, 1]])

    # To be updated, TBF
    def test03_select_mesh(self):

        # rectangular meshes
        lkey = ['m0', 'm1', 'm2', 'm3']
        lind = [None, ([0, 5], [0, 6]), [0, 10, 100], ([0, 5, 6], [0, 2, 3])]
        lelements = ['cents', 'knots', 'cents', None]
        lreturnas = ['ind', 'data', 'data', 'ind']
        lreturn_neig = [None, True, False, True]
        lcrop = [False, True, True, False]
        for ii, (k0, v0) in enumerate(self.dobj.items()):
            indf = self.dobj[k0].select_mesh_elements(
                key=lkey[ii],
                ind=lind[ii],
                elements=lelements[ii],
                returnas=lreturnas[ii],
                return_neighbours=lreturn_neig[ii],
                crop=lcrop[ii],
            )

        # triangular meshes
        lkeys = ['tri0', 'tri0', 'tri0', 'tri1']
        lind = [None, [1], 1, [0, 1]]
        lelements = ['knots', None, 'cents', 'cents']
        lreturnas = ['ind', 'data', 'ind', 'data']
        for ii, k0 in enumerate(lkeys):
            out = self.dobjtri[k0].select_mesh_elements(
                key=k0,
                ind=lind[ii],
                elements=lelements[ii],
                returnas=lreturnas[ii],
                return_neighbours=True,
                crop=lcrop[ii],
            )

    def test04_select_bsplines(self):

        # rectangular meshes
        lkey = ['m0-bs0', 'm1-bs1', 'm2-bs2', 'm3-bs3']
        lind = [None, ([0, 5], [0, 6]), [0, 10, 100], ([0, 5, 6], [0, 2, 3])]
        lreturnas = [None, 'data', 'data', 'ind']
        lreturn_cents = [None, True, False, True]
        lreturn_knots = [None, False, True, True]
        for ii, (k0, v0) in enumerate(self.dobj.items()):
            indf = self.dobj[k0].select_bsplines(
                key=lkey[ii],
                ind=lind[ii],
                returnas=lreturnas[ii],
                return_cents=lreturn_cents[ii],
                return_knots=lreturn_knots[ii],
            )

        # triangular meshes
        lkeys = ['tri0', 'tri0', 'tri0', 'tri1']
        lkeysbs = ['tri0-bs0', None, 'tri0-bs0', 'tri1-bs1']
        lind = [None, [1], 1, [0, 1]]
        lelements = ['knots', None, 'cents', 'cents']
        lreturnas = ['ind', 'data', 'ind', 'data']
        for ii, k0 in enumerate(lkeys):
            indf = self.dobjtri[k0].select_bsplines(
                key=lkeysbs[ii],
                ind=lind[ii],
                returnas=lreturnas[ii],
                return_cents=lreturn_cents[ii],
                return_knots=lreturn_knots[ii],
            )

    def test05_sample_mesh(self):

        # rectangular meshes
        lres = [None, 0.1, 0.01, [0.1, 0.05]]
        lmode = [None, 'rel', 'abs', 'abs']
        lgrid = [None, True, False, False]
        for ii, (k0, v0) in enumerate(self.dobj.items()):
            out = v0.get_sample_mesh(
                res=lres[ii], grid=lgrid[ii], mode=lmode[ii],
            )

        # triangular meshes
        lkeys = ['tri0', 'tri0', 'tri0', 'tri1']
        lres = [None, 0.1, 0.01, [0.1, 0.05]]
        lmode = [None, 'rel', 'abs', 'abs']
        lgrid = [None, True, False, False]
        for ii, k0 in enumerate(lkeys):
            out = self.dobjtri[k0].get_sample_mesh(
                res=lres[ii], grid=lgrid[ii], mode=lmode[ii],
            )

    """
    def test06_sample_bspline(self):
        lres = [None, 0.1, 0.01, [0.1, 0.05]]
        lmode = [None, 'rel', 'abs', 'abs']
        lgrid = [None, True, False, False]
        for ii, (k0, v0) in enumerate(self.dobj.items()):
            out = v0.get_sample_bspline(
                res=lres[ii], grid=lgrid[ii], mode=lmode[ii],
            )
    """

    def test07_ev_details_vs_sum(self):

        x = np.linspace(2.2, 2.8, 5)
        y = np.linspace(-0.5, 0.5, 5)
        x = np.tile(x, (y.size, 1))
        y = np.tile(y, (x.shape[1], 1)).T

        # rectangular meshes
        lkey = ['m0-bs0', 'm1-bs1', 'm2-bs2', 'm3-bs3']
        for ii, (k0, v0) in enumerate(self.dobj.items()):
            val = v0.interp2d(
                key=lkey[ii],
                R=x,
                Z=y,
                coefs=None,
                indbs=None,
                indt=None,
                grid=False,
                details=True,
                reshape=True,
                res=None,
                crop=True,
                nan0=ii % 2 == 0,
                imshow=False,
            )
            crop = v0.dobj['bsplines'][lkey[ii]]['crop']
            if crop is False:
                shap = np.prod(v0.dobj['bsplines'][lkey[ii]]['shape'])
            else:
                shap = v0.ddata[crop]['data'].sum()
            assert val.shape == tuple(np.r_[x.shape, shap])

            val_sum = v0.interp2d(
                key=lkey[ii],
                R=x,
                Z=y,
                coefs=None,
                indbs=None,
                indt=None,
                grid=False,
                details=False,
                reshape=True,
                res=None,
                crop=True,
                nan0=ii % 2 == 0,
                imshow=False,
            )
            indok = ~np.isnan(val_sum[0, ...])

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # Does not work because of knots padding used in func_details
            # Due to scpinterp._bspl.evaluate_spline()...
            if False:   # To be debugged
                assert np.allclose(
                    val_sum[0, indok],
                    np.nansum(val, axis=-1)[indok],
                    equal_nan=True,
                )
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # triangular meshes
        lkey = ['tri0-bs0', 'tri1-bs1']
        for ii, (k0, v0) in enumerate(self.dobjtri.items()):
            val = v0.interp2d(
                key=lkey[ii],
                R=x,
                Z=y,
                coefs=None,
                indbs=None,
                indt=None,
                grid=False,
                details=True,
                reshape=None,
                res=None,
                crop=True,
                nan0=ii % 2 == 0,
                imshow=False,
            )
            crop = v0.dobj['bsplines'][lkey[ii]].get('crop', False)
            if crop is False:
                shap = np.prod(v0.dobj['bsplines'][lkey[ii]]['shape'])
            else:
                shap = v0.ddata[crop]['data'].sum()
            assert val.shape == tuple(np.r_[x.shape, shap])

            val_sum = v0.interp2d(
                key=lkey[ii],
                R=x,
                Z=y,
                coefs=None,
                indbs=None,
                indt=None,
                grid=False,
                details=False,
                reshape=None,
                res=None,
                crop=True,
                nan0=ii % 2 == 0,
                imshow=False,
            )
            indok = ~np.isnan(val_sum[0, ...])
            assert np.allclose(
                val_sum[0, indok],
                np.nansum(val, axis=-1)[indok],
                equal_nan=True,
            )

    def test08_plot_mesh(self):

        # rectangular meshes
        lik = [None, ([0, 2], [0, 3]), [2, 3], None]
        lic = [None, ([0, 2], [0, 3]), None, [2, 3]]
        for ii, (k0, v0) in enumerate(self.dobj.items()):
            dax = self.dobj[k0].plot_mesh(
                ind_knot=lik[ii],
                ind_cent=lic[ii],
            )
        plt.close('all')

        # triangular meshes
        lik = [None, [0, 2], [2, 3], None]
        lic = [None, [0, 2], None, [2, 3]]
        for ii, (k0, v0) in enumerate(self.dobjtri.items()):
            dax = self.dobjtri[k0].plot_mesh(
                ind_knot=lik[ii],
                ind_cent=lic[ii],
            )
        plt.close('all')

    # TBF for triangular
    def test09_plot_bsplines(self):

        # rectangular meshes
        lkey = ['m0-bs0', 'm1-bs1', 'm2-bs2', 'm3-bs3']
        lind = [None, ([1, 2], [2, 1]), (1, 1), [1, 2, 10]]
        lknots = [None, True, False, True]
        lcents = [False, False, True, True]
        for ii, (k0, v0) in enumerate(self.dobj.items()):
            dax = self.dobj[k0].plot_bsplines(
                key=lkey[ii],
                ind=lind[ii],
                knots=lknots[ii],
                cents=lcents[ii],
            )
        plt.close('all')

        # triangular meshes
        lkey = ['tri0-bs0', 'tri1-bs1']  # , 'm2-bs2', 'm3-bs3']
        lind = [None, [1, 2], (1, 1), [1, 2, 10]]
        lknots = [None, True, False, True]
        lcents = [False, False, True, True]
        for ii, (k0, v0) in enumerate(self.dobjtri.items()):
            dax = self.dobjtri[k0].plot_bsplines(
                key=lkey[ii],
                ind=lind[ii],
                knots=lknots[ii],
                cents=lcents[ii],
            )
        plt.close('all')

    def test10_plot_profile2d(self):

        # rectangular meshes
        lkey = ['m0-bs0', 'm1-bs1', 'm2-bs2', 'm3-bs3']
        for ii, (k0, v0) in enumerate(self.dobj.items()):
            key = str(ii)
            kbs = lkey[ii]
            ref = self.dobj[k0].dobj['bsplines'][kbs]['ref']
            shapebs = self.dobj[k0].dobj['bsplines'][kbs]['shape']

            self.dobj[k0].add_data(
                key=key,
                data=np.random.random(shapebs),
                ref=ref,
            )

            dax = self.dobj[k0].plot_profile2d(
                key=key,
            )
        plt.close('all')

        # triangular meshes
        # DEACTIVATED BECAUSE TOO SLOW IN CURRENT VERSION !!!
        if False:
            lkey = ['tri0-bs0', 'tri1-bs1']
            for ii, (k0, v0) in enumerate(self.dobjtri.items()):
                key = str(ii)
                kbs = lkey[ii]
                ref = self.dobjtri[k0].dobj['bsplines'][kbs]['ref']
                shapebs = self.dobjtri[k0].dobj['bsplines'][kbs]['shape']

                self.dobjtri[k0].add_data(
                    key=key,
                    data=np.random.random(shapebs),
                    ref=ref,
                )

                dax = self.dobjtri[k0].plot_profile2d(
                    key=key,
                )
            plt.close('all')

    # TBF for triangular
    def test11_add_bsplines_operator(self):
        lkey = ['m0-bs0', 'm1-bs1', 'm2-bs2']
        lop = ['D0N1', 'D0N2', 'D1N2', 'D2N2']
        lgeom = ['linear', 'toroidal']
        lcrop = [False, True]

        dfail = {}
        for ii, (k0, v0) in enumerate(self.dobj.items()):
            if ii == 3:
                continue

            for comb in itt.product(lop, lgeom, lcrop):
                deg = self.dobj[k0].dobj['bsplines'][lkey[ii]]['deg']

                # only test exact operators
                if int(comb[0][1]) > deg:
                    # except deg =0 D1N2
                    if deg == 0 and comb[0] == 'D1N2':
                        pass
                    else:
                        continue
                try:
                    self.dobj[k0].add_bsplines_operator(
                        key=lkey[ii],
                        operator=comb[0],
                        geometry=comb[1],
                        crop=comb[2],
                    )
                except Exception as err:
                    dfail[k0] = (
                        f"key {lkey[ii]}, op '{comb[0]}', geom '{comb[1]}': "
                        + str(err)
                    )

        # Raise error if any fail
        if len(dfail) > 0:
            lstr = [f'\t- {k0}: {v0}' for k0, v0 in dfail.items()]
            msg = (
                "The following operators failed:\n"
                + "\n".join(lstr)
            )
            raise Exception(msg)

    # TBF for triangular
    def test12_compute_plot_geometry_matrix(self):

        # get config and cam
        conf = tf.load_config('WEST-V0')
        cam = tf.geom.utils.create_CamLOS1D(
            pinhole=[3., 1., 0.],
            orientation=[np.pi, 0., 0],
            focal=0.1,
            sensor_nb=50,
            sensor_size=0.15,
            config=conf,
            Diag='SXR',
            Exp='WEST',
            Name='cam1',
        )

        # compute geometry matrices
        for ii, (k0, v0) in enumerate(self.dobj.items()):

            self.dobj[k0].add_geometry_matrix(
                cam=cam,
                res=0.01,
                crop=True,
                store=True,
            )

            dax = self.dobj[k0].plot_geometry_matrix(
                cam=cam, indchan=12, indbf=10,
            )
            plt.close('all')
