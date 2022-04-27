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
            if self.obj.dobj['mesh'][k0]['type'] == 'rect':
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

            elif ind not in [1, 3]:
                indt = self.obj.select_ind(
                    key=k0,
                    ind=lind[ind],
                    elements=lelements[ind],
                    returnas=int,
                )

    def test03_select_mesh(self):

        lind0 = [None, ([0, 5], [0, 6]), [0, 10, 100], ([0, 5, 6], [0, 2, 3])]
        lind1 = [None, [1], 1, [0, 1]]
        lelem = [None, 'cents', 'knots']
        for ii, k0 in enumerate(self.lm):

            if self.obj.dobj['mesh'][k0]['type'] == 'rect':
                lind = lind0
            else:
                lind = lind1

            out = self.obj.select_mesh_elements(
                key=k0,
                ind=lind[ii%len(lind)],
                elements=lelem[ii%3],
                returnas='ind' if ii%2 == 0 else 'data',
                return_neighbours=None if ii == 0 else bool(ii%2),
                crop=ii%3 == 1,
            )

    def test04_select_bsplines(self):

        lind0 = [None, ([0, 5], [0, 6]), [0, 10, 100], ([0, 5, 6], [0, 2, 3])]
        lind1 = [None, [1], 1, [0, 1]]
        for ii, k0 in enumerate(self.lbs):

            km = self.obj.dobj['bsplines'][k0]['mesh']
            if self.obj.dobj['mesh'][km]['type'] == 'rect':
                lind = lind0
            else:
                lind = lind1

            out = self.obj.select_bsplines(
                key=k0,
                ind=lind[ii%len(lind)],
                returnas='ind' if ii%3 == 0 else 'data',
                return_cents=None if ii == 1 else bool(ii%3),
                return_knots=None if ii == 2 else bool(ii%2),
            )

    def test05_sample_mesh(self):

        lres = [None, 0.1, 0.01, [0.1, 0.05]]
        lmode = [None, 'rel', 'abs']
        lgrid = [None, True, False]
        for ii, k0 in enumerate(self.lm):
            out = self.obj.get_sample_mesh(
                key=k0,
                res=lres[ii%len(lres)],
                mode=lmode[ii%len(lmode)],
                grid=lgrid[ii%len(lgrid)],
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

        for ii, k0 in enumerate(self.lbs):
            val = self.obj.interpolate_profile2d(
                key=k0,
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
            crop = self.obj.dobj['bsplines'][k0].get('crop', False)
            if crop is False:
                shap = np.prod(self.obj.dobj['bsplines'][k0]['shape'])
            else:
                shap = self.obj.ddata[crop]['data'].sum()
            assert val.shape == tuple(np.r_[x.shape, shap])

            val_sum = self.obj.interpolate_profile2d(
                key=k0,
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
            # Does not work for rect mesh
            # because of knots padding used in func_details
            # Due to scpinterp._bspl.evaluate_spline()...
            km = self.obj.dobj['bsplines'][k0]['mesh']
            if self.obj.dobj['mesh'][km]['type'] == 'tri':   # To be debugged
                assert np.allclose(
                    val_sum[0, indok],
                    np.nansum(val, axis=-1)[indok],
                    equal_nan=True,
                )
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    def test08_plot_mesh(self):
        lik0 = [None, ([0, 2], [0, 3]), [2, 3], None]
        lic0 = [None, ([0, 2], [0, 3]), None, [2, 3]]
        lik1 = [None, [0, 2], [2, 3], None]
        lic1 = [None, [0, 2], None, [2, 3]]
        for ii, k0 in enumerate(self.lm):

            lik = lik0 if self.obj.dobj['mesh'][k0]['type'] == 'rect' else lik1
            lic = lic0 if self.obj.dobj['mesh'][k0]['type'] == 'rect' else lic1

            dax = self.obj.plot_mesh(
                key=k0,
                ind_knot=lik[ii%len(lik)],
                ind_cent=lic[ii%len(lic)],
            )
        plt.close('all')

    def test09_plot_bsplines(self):

        li0 = [None, ([1, 2], [2, 1]), (1, 1), [1, 2, 10]]
        li1 = [None, [1, 2], (1, 1), [1, 2, 10]]
        for ii, k0 in enumerate(self.lbs):

            km = self.obj.dobj['bsplines'][k0]['mesh']
            li = li0 if self.obj.dobj['mesh'][km]['type'] == 'rect' else li1

            dax = self.obj.plot_bsplines(
                key=k0,
                ind=li[ii%len(li)],
                knots=bool(ii%3),
                cents=bool(ii%2),
            )
        plt.close('all')

    # def test10_plot_profile2d(self):

        # # rectangular meshes
        # lkey = ['m0-bs0', 'm1-bs1', 'm2-bs2', 'm3-bs3']
        # for ii, (k0, v0) in enumerate(self.dobj.items()):
            # key = str(ii)
            # kbs = lkey[ii]
            # ref = self.dobj[k0].dobj['bsplines'][kbs]['ref']
            # shapebs = self.dobj[k0].dobj['bsplines'][kbs]['shape']

            # self.dobj[k0].add_data(
                # key=key,
                # data=np.random.random(shapebs),
                # ref=ref,
            # )

            # dax = self.dobj[k0].plot_profile2d(
                # key=key,
            # )
        # plt.close('all')

        # # triangular meshes
        # # DEACTIVATED BECAUSE TOO SLOW IN CURRENT VERSION !!!
        # if False:
            # lkey = ['tri0-bs0', 'tri1-bs1']
            # for ii, (k0, v0) in enumerate(self.dobjtri.items()):
                # key = str(ii)
                # kbs = lkey[ii]
                # ref = self.dobjtri[k0].dobj['bsplines'][kbs]['ref']
                # shapebs = self.dobjtri[k0].dobj['bsplines'][kbs]['shape']

                # self.dobjtri[k0].add_data(
                    # key=key,
                    # data=np.random.random(shapebs),
                    # ref=ref,
                # )

                # dax = self.dobjtri[k0].plot_profile2d(
                    # key=key,
                # )
            # plt.close('all')

    def test11_add_bsplines_operator(self):
        lkey = ['m0-bs0', 'm1-bs1', 'm2-bs2']
        lop = ['D0N1', 'D0N2', 'D1N2', 'D2N2']
        lgeom = ['linear', 'toroidal']
        lcrop = [False, True]

        dfail = {}
        for ii, k0 in enumerate(self.lbs):

            km = self.obj.dobj['bsplines'][k0]['mesh']
            if self.obj.dobj['mesh'][km]['type'] == 'tri':
                continue

            for comb in itt.product(lop, lgeom, lcrop):
                deg = self.obj.dobj['bsplines'][k0]['deg']

                if deg == 3 and comb[0] in ['D0N1', 'D0N2', 'D1N2', 'D2N2']:
                    continue

                # only test exact operators
                if int(comb[0][1]) > deg:
                    # except deg = 0 D1N2
                    if deg == 0 and comb[0] == 'D1N2':
                        pass
                    else:
                        continue
                try:
                    self.obj.add_bsplines_operator(
                        key=k0,
                        operator=comb[0],
                        geometry=comb[1],
                        crop=comb[2],
                    )
                except Exception as err:
                    dfail[k0] = (
                        f"key {k0}, op '{comb[0]}', geom '{comb[1]}': "
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
        for ii, k0 in enumerate(self.lbs):

            self.obj.add_geometry_matrix(
                key=k0,
                cam=cam,
                res=0.01,
                crop=True,
                store=True,
            )

        # plot geometry matrices
        for ii, k0 in enumerate(self.obj.dobj['matrix']):
            dax = self.obj.plot_geometry_matrix(
                key=k0,
                cam=cam,
                indchan=12,
                indbf=10,
            )
        plt.close('all')
