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


class Test01_Inversions():

    @classmethod
    def setup_class(cls):
        pass

    def setup_method(self):

        # create conf and cam
        conf0 = tf.load_config('WEST-V0')

        coll = tf.data.Collection()

        # add camera
        npix = 10
        coll.add_camera_pinhole(
            key='camH',
            key_diag='d0',
            x=3.0,
            y=1.,
            z=0.3,
            pinhole_size=0.01,
            focal=0.1,
            pix_size=0.1,
            pix_nb=npix,
            theta=-5*np.pi/6,
            dphi=0,
            tilt=0,
            config=conf0,
        )

        coll.add_camera_pinhole(
            key='camV',
            key_diag='d0',
            x=3.0,
            y=1.,
            z=-0.3,
            pinhole_size=0.01,
            focal=0.1,
            pix_size=0.1,
            pix_nb=npix,
            theta=5*np.pi/6,
            dphi=0,
            tilt=0,
            config=conf0,
        )

        coll.add_camera_pinhole(
            key='cam2',
            cam_type='2d',
            key_diag='d1',
            x=3.0,
            y=1.,
            z=-0.3,
            pinhole_size=0.01,
            focal=0.1,
            pix_size=0.1,
            pix_nb=[10, 5],
            theta=5*np.pi/6,
            dphi=0,
            tilt=0,
            config=conf0,
        )

        # mesh rect deg 1 and 2
        coll.add_mesh(
            crop_poly=conf0,
            key='m1',
            res=0.10,
            deg=0,
        )
        coll.add_bsplines(deg=1)
        coll.add_bsplines(deg=2)

        # add 2d radius for polar mesh
        kR, kZ = coll.dobj['mesh']['m1']['knots']
        R, Z = coll.ddata[kR]['data'], coll.ddata[kZ]['data']
        nR, nZ = R.size, Z.size
        R = np.repeat(R[:, None], nZ, axis=1)
        Z = np.repeat(Z[None, :], nR, axis=0)
        rad2d = ((R-2.4)/0.4)**2 + (Z/0.5)**2
        krad = 'rad2d'
        coll.add_data(key=krad, data=rad2d, ref='m1-bs1')

        # mesh polar deg 1 and 2
        coll.add_mesh_polar(
            key='m2',
            radius=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.],
            radius2d=krad,
            deg=0,
        )
        coll.add_bsplines(key='m2', deg=1)
        coll.add_bsplines(key='m2', deg=2)

        # add geometry matrices
        # coll.add_ref(key='chan', data=chan, group='chan')
        coll.add_geometry_matrix(key_diag='d0', key_bsplines='m1-bs0')
        coll.add_geometry_matrix(key_diag='d0', key_bsplines='m1-bs1')
        coll.add_geometry_matrix(key_diag='d1', key_bsplines='m1-bs2')
        coll.add_geometry_matrix(key_diag='d0', key_bsplines='m2-bs0')
        coll.add_geometry_matrix(key_diag='d0', key_bsplines='m2-bs1')
        coll.add_geometry_matrix(key_diag='d1', key_bsplines='m2-bs2')

        # add emiss
        t0 = np.array([0, 1])
        kap = coll.dobj['bsplines']['m2-bs1']['apex'][0]
        rad = coll.ddata[kap]['data']
        emiss = (
            np.exp(-(rad)**2/0.2**2)
            + 0.1*np.cos(t0)[:, None]*np.exp(-rad**2/0.05**2)
        )

        coll.add_ref(key='nt0', size=2)
        coll.add_data(key='t0', data=t0, dim='time', ref='nt0')
        coll.add_data(
            key='emiss',
            data=emiss,
            ref=('nt0', 'm2-bs1'),
            units='W/m3/sr',
        )

        # add synthetic data
        coll.compute_diagnostic_signal(
            key='s0',
            key_diag='d0',
            key_integrand='emiss',
            res=0.01,
        )

        coll.compute_diagnostic_signal(
            key='s1',
            key_diag='d1',
            key_integrand='emiss',
            res=0.01,
        )

        self.coll = coll

    def teardown_method(self):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def test01_run_all_and_plot(self):

        dalgo = tf.data.get_available_inversions_algo(returnas=dict)
        lstore = [True, False]

        # running
        for kmat in self.coll.dobj['geom matrix'].keys():

            kbs = self.coll.dobj['geom matrix'][kmat]['bsplines']
            kd = self.coll.dobj['geom matrix'][kmat]['diagnostic']
            km = self.coll.dobj['bsplines'][kbs]['mesh']
            mtype = self.coll.dobj['mesh'][km]['type']
            deg = self.coll.dobj['bsplines'][kbs]['deg']

            if deg in [0, 1]:
                lop = ['D1N2']
            else:
                lop = ['D1N2', 'D2N2']

            for comb in itt.product(dalgo.keys(), lop, lstore):

                if comb[0] == 'algo5':
                    continue

                if comb[2] == 'D2N2' and deg != 2:
                    continue

                if 'mfr' in comb[0].lower() and deg != 0:
                    continue

                algofam = dalgo[comb[0]]['family']
                if algofam == 'Non-regularized' and mtype != 'polar':
                    continue
                if algofam != 'Non-regularized' and mtype == 'polar':
                    continue


                kdat = 's0' if kd == 'd0' else 's1'
                try:

                    self.coll.add_inversion(
                        algo=comb[0],
                        key_matrix=kmat,
                        key_data=kdat,
                        sigma=0.10,
                        operator=comb[1],
                        store=comb[2],
                        conv_crit=1.e-2,
                        kwdargs={'tol': 1.e-2, 'maxiter': 100},
                        verb=0,
                    )
                    ksig = f'{kdat}-sigma'
                    if ksig in self.coll.ddata.keys():
                        self.coll.remove_data(ksig)

                except Exception as err:
                    c0 = (
                        dalgo[comb[0]]['source'] == 'tomotok'
                        and comb[1] == 'D1N2'
                        and kmat == 'matrix0'
                    )
                    if c0:
                        # Discrete gradient seem to be not positive-definite
                        # To be investigated...
                        pass
                    else:
                        raise err

        # plotting
        linv = list(self.coll.dobj['inversions'].keys())[::7]
        for kinv in linv:
            dax = self.coll.plot_inversion(key=kinv, res=0.1)

        plt.close('all')
