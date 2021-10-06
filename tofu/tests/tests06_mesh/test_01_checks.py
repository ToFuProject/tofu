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


_here = os.path.abspath(os.path.dirname(__file__))
VerbHead = 'tofu.mesh.test_01_checks'
_TOFU_USER = os.path.join(os.path.expanduser("~"), '.tofu')
_CUSTOM = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
_CUSTOM = os.path.join(_CUSTOM, 'scripts', 'tofucustom.py')


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
#     checking routines
#
#######################################################


class Test01_checks():

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


#######################################################
#
#     object mesh2D
#
#######################################################


class Test02_Mesh2DRect():

    @classmethod
    def setup_class(cls):
        pass

    def setup(self):
        self.dobj = {
            'm0': tfd.Mesh2DRect(),
            'm1': tfd.Mesh2DRect(),
            'm2': tfd.Mesh2DRect(),
            'm3': None,
        }

        # add mesh
        ldomain = [
            [[2, 3], [-1, 1]],
            [[2, 2.3, 2.6, 3], [-1, 0., 1]],
            [[2, 3], [-1, 0, 1]],
        ]
        lres = [
            0.1,
            [[0.2, 0.1, 0.1, 0.2], [0.2, 0.1, 0.2]],
            [0.1, [0.2, 0.1, 0.2]],
        ]

        for ii, (k0, v0) in enumerate(self.dobj.items()):
            if k0 != 'm3':
                self.dobj[k0].add_mesh(
                    domain=ldomain[ii],
                    res=lres[ii],
                    key=k0,
                )
            else:
                self.dobj[k0] = tfd.Mesh2DRect.from_Config(
                    tf.load_config('WEST'),
                    res=0.1,
                    key=k0,
                )

        # add splines
        for ii, (k0, v0) in enumerate(self.dobj.items()):
            self.dobj[k0].add_bsplines(deg=ii)

    def teardown(self):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def test01_get_summary(self):
        for ii, (k0, v0) in enumerate(self.dobj.items()):
            self.dobj[k0].get_summary()

    def test02_select_ind(self):
        lkey = ['m0', 'm1-bs1', 'm2', 'm3-bs3']
        lelements = ['cents', None, 'knots', None]
        lind = [None, ([0, 5], [0, 6]), [0, 10, 100], ([0, 5, 6], [0, 2, 3])]
        lcrop = [True, False, True, False]
        for ii, (k0, v0) in enumerate(self.dobj.items()):
            indt = self.dobj[k0].select_ind(
                key=lkey[ii],
                ind=lind[ii],
                elements=lelements[ii],
                returnas=tuple,
                crop=lcrop[ii],
            )
            indf = self.dobj[k0].select_ind(
                key=lkey[ii],
                ind=indt,
                elements=lelements[ii],
                returnas=np.ndarray,
                crop=lcrop[ii],
            )
            indt2 = self.dobj[k0].select_ind(
                key=lkey[ii],
                ind=indf,
                elements=lelements[ii],
                returnas=tuple,
                crop=lcrop[ii],
            )
            assert all([np.allclose(indt[ii], indt2[ii]) for ii in [0, 1]])

    def test03_select_mesh(self):
        lkey = ['m0', 'm1', 'm2', 'm3']
        lind = [None, ([0, 5], [0, 6]), [0, 10, 100], ([0, 5, 6], [0, 2, 3])]
        lelements = ['cents', 'cents', 'knots', None]
        lreturnas = ['ind', 'data', 'data', 'ind']
        lreturn_neig = [None, True, False, True]
        lcrop = [False, True, False, False]
        for ii, (k0, v0) in enumerate(self.dobj.items()):
            indf = self.dobj[k0].select_mesh_elements(
                key=lkey[ii],
                ind=lind[ii],
                elements=lelements[ii],
                returnas=lreturnas[ii],
                return_neighbours=lreturn_neig[ii],
                crop=lcrop[ii],
            )

    def test04_select_bsplines(self):
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

    def test05_sample_mesh(self):
        lres = [None, 0.1, 0.01, [0.1, 0.05]]
        lmode = [None, 'rel', 'abs', 'abs']
        lgrid = [None, True, False, False]
        for ii, (k0, v0) in enumerate(self.dobj.items()):
            out = v0.get_sample_mesh(
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

    def test07_plot_mesh(self):
        lik = [None, ([0, 2], [0, 3]), [2, 3], None]
        lic = [None, ([0, 2], [0, 3]), None, [2, 3]]
        for ii, (k0, v0) in enumerate(self.dobj.items()):
            dax = self.dobj[k0].plot_mesh(
                ind_knot=lik[ii],
                ind_cent=lic[ii],
            )
        plt.close('all')

    def test08_plot_bsplines(self):
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

    def test09_plot_profile2d(self):
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

    def test10_add_bsplines_operator(self):
        lkey = ['m0-bs0', 'm1-bs1', 'm2-bs2']
        lop = ['D0', 'D0N2', 'D1N2', 'D2N2']
        lgeom = ['linear', 'toroidal']
        lcrop = [False, True]
        for ii, (k0, v0) in enumerate(self.dobj.items()):
            if ii == 3:
                continue

            for comb in itt.product(lop, lgeom, lcrop):
                deg = self.dobj[k0].dobj['bsplines'][lkey[ii]]['deg']
                if int(comb[0][1]) > deg:
                    continue
                self.dobj[k0].add_bsplines_operator(
                    key=lkey[ii],
                    operator=comb[0],
                    geometry=comb[1],
                    crop=comb[2],
                )

    def test11_compute_plot_geometry_matrix(self):

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

            mat = self.dobj[k0].compute_geometry_matrix(
                cam=cam, res=0.01, crop=True,
            )

            dax = mat.plot_geometry_matrix(cam=cam, indchan=12, indbf=100)
            plt.close('all')
