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

        # mesh deg 1 and 2
        mesh = tf.data.Mesh2DRect.from_Config(
            config=conf0,
            key='try1',
            res=0.10,
            deg=1,
        )
        mesh.add_bsplines(deg=2)

        # add geometry matrices
        chan = np.arange(0, 30)
        mesh.add_ref(key='chan', data=chan, group='chan')
        mesh.add_geometry_matrix(cam=cam, key='try1-bs1', key_chan='chan')
        mesh.add_geometry_matrix(cam=cam, key='try1-bs2', key_chan='chan')

        # add data
        t0 = np.array([0])
        t1 = np.array([0, 1.])
        data0 = np.exp(-(chan - 15.)**2/10**2)
        data1 = (
            np.exp(-(chan - 15.)**2/10**2)
            + 0.1*np.cos(t1)[:, None]*np.exp(-(chan - 15)**2/2**2)
        )
        # mesh.add_ref(key='t0', data=t0, units='s', group='time')
        mesh.add_ref(key='t1', data=t1, units='s', group='time')
        mesh.add_data(key='data0', data=data0, ref=('chan',))
        mesh.add_data(key='data1', data=data1, ref=('t1', 'chan'))

        self.mesh = mesh

    def teardown(self):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def test01_run_all_and_plot(self):

        lalgo = tf.data._mesh._inversions_comp._inversions_checks._LALGO
        lstore = [True, False]
        lkdata = ['data0', 'data1']

        # running
        for kmat in ['matrix0', 'matrix1']:

            lop = ['D1N2'] if kmat == 'matrix0' else ['D1N2', 'D2N2']

            for comb in itt.product(lalgo, lkdata, lop, lstore):
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

        # plotting
        linv = list(self.mesh.dobj['inversions'].keys())[::7]
        for kinv in linv:
            dax = self.mesh.plot_inversion(key=kinv)

        plt.close('all')
