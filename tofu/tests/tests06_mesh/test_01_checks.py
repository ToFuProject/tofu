"""
This module contains tests for tofu.geom in its structured version
"""

# Built-in
import os
import shutil
import itertools as itt
import warnings


# Standard


# tofu-specific
from tofu import __version__
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
#     Creating Ves objects and testing methods
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
