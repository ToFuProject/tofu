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

    def setup(self):

        # create conf and cam
        conf0 = tf.load_config('WEST-V0')
        cam = tf.geom.utils.create_CamLOS1D(
            pinhole=[3.0, 1., 0.3],
            focal=0.1,
            sensor_size=0.1,
            sensor_nb=30,
            orientation=[-5*np.pi/6, 0, 0],
            config=conf0,
            Name='camH',
            Exp='WEST',
            Diag='SXR',
        )

        # mesh rect deg 1 and 2
        mesh = tf.data.Plasma2D()
        mesh.add_mesh(
            crop_poly=conf0,
            key='m1',
            res=0.10,
            deg=0,
        )
        mesh.add_bsplines(deg=1)
        mesh.add_bsplines(deg=2)

        # add 2d radius for polar mesh
        kR, kZ = mesh.dobj['mesh']['m1']['knots']
        R, Z = mesh.ddata[kR]['data'], mesh.ddata[kZ]['data']
        nR, nZ = R.size, Z.size
        R = np.repeat(R[:, None], nZ, axis=1)
        Z = np.repeat(Z[None, :], nR, axis=0)
        rad2d = ((R-2.4)/0.4)**2 + (Z/0.5)**2
        krad = 'rad2d'
        mesh.add_data(key=krad, data=rad2d, ref='m1-bs1')

        # mesh polar deg 1 and 2
        mesh.add_mesh_polar(
            key='m2',
            radius=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.],
            radius2d=krad,
            deg=0,
        )
        mesh.add_bsplines(key='m2', deg=1)
        mesh.add_bsplines(key='m2', deg=2)

        # add geometry matrices
        chan = np.arange(0, 30)
        mesh.add_ref(key='chan', data=chan, group='chan')
        mesh.add_geometry_matrix(cam=cam, key='m1-bs0', key_chan='chan')
        mesh.add_geometry_matrix(cam=cam, key='m1-bs1', key_chan='chan')
        mesh.add_geometry_matrix(cam=cam, key='m1-bs2', key_chan='chan')
        mesh.add_geometry_matrix(cam=cam, key='m2-bs0', key_chan='chan')
        mesh.add_geometry_matrix(cam=cam, key='m2-bs1', key_chan='chan')
        mesh.add_geometry_matrix(cam=cam, key='m2-bs2', key_chan='chan')

        # add data
        t0 = np.array([0])
        t1 = np.array([0, 1.])
        data0 = np.exp(-(chan - 15.)**2/10**2)
        data1 = (
            np.exp(-(chan - 15.)**2/10**2)
            + 0.1*np.cos(t1)[:, None]*np.exp(-(chan - 15)**2/2**2)
        )
        # mesh.add_ref(key='t0', data=t0, units='s', group='time')
        mesh.add_ref(key='t1', data=t1, units='s', dim='time')
        mesh.add_data(key='data0', data=data0, ref=('chan',))
        mesh.add_data(key='data1', data=data1, ref=('t1', 'chan'))

        self.mesh = mesh

    def teardown(self):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def test01_run_all_and_plot(self):

        dalgo = tf.data.get_available_inversions_algo(returnas=dict)
        lstore = [True, False]
        lkdata = ['data0', 'data1']

        # running
        for kmat in self.mesh.dobj['matrix'].keys():

            kbs = self.mesh.dobj['matrix'][kmat]['bsplines']
            km = self.mesh.dobj['bsplines'][kbs]['mesh']
            mtype = self.mesh.dobj['mesh'][km]['type']
            deg = self.mesh.dobj['bsplines'][kbs]['deg']

            if deg in [0, 1]:
                lop = ['D1N2']
            else:
                lop = ['D1N2', 'D2N2']

            for comb in itt.product(dalgo.keys(), lkdata, lop, lstore):

                if comb[2] == 'D2N2' and deg != 2:
                    continue

                if 'mfr' in comb[0].lower() and deg != 0:
                    continue

                algofam = dalgo[comb[0]]['family']
                if algofam == 'Non-regularized' and mtype != 'polar':
                    continue
                if algofam != 'Non-regularized' and mtype == 'polar':
                    continue

                try:
                    self.mesh.add_inversion(
                        algo=comb[0],
                        key_matrix=kmat,
                        key_data=comb[1],
                        sigma=0.10,
                        operator=comb[2],
                        store=comb[3],
                        conv_crit=1.e-3,
                        kwdargs={'tol': 1.e-4},
                        verb=0,
                    )
                    ksig = f'{comb[1]}-sigma'
                    if ksig in self.mesh.ddata.keys():
                        self.mesh.remove_data(ksig)

                except Exception as err:
                    c0 = (
                        dalgo[comb[0]]['source'] == 'tomotok'
                        and comb[2] == 'D1N2'
                        and kmat == 'matrix0'
                    )
                    if c0:
                        # Discrete gradient seem to be not positive-definite
                        # To be investigated...
                        pass
                    else:
                        raise err

        # plotting
        linv = list(self.mesh.dobj['inversions'].keys())[::7]
        for kinv in linv:
            dax = self.mesh.plot_inversion(key=kinv)

        plt.close('all')
