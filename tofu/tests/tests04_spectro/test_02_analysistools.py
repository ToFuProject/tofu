
"""
This module contains tests for tofu.geom in its structured version
"""

# Built-in
import os
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
VerbHead = 'tofu.spectro'


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

# @with_setup_method(my_setup_function, my_teardown_function)
# def test_numbers_3_4():
#    print 'test_numbers_3_4  <============================ actual test code'
#    assert multiply(3,4) == 12

# @with_setup_method(my_setup_function, my_teardown_function)
# def test_strings_a_3():
#    print 'test_strings_a_3  <============================ actual test code'
#    assert multiply('a',3) == 'aaa'


#######################################################
#
#     Creating Ves objects and testing methods
#
#######################################################


class Test01_LocalExtrema(object):

    @classmethod
    def setup_class(cls):

        pfe = os.path.join(_here, 'test_data', 'UV_spectra_sh55506.npz')
        out = dict(np.load(pfe, allow_pickle=True))
        cls.lamb = out['lamb']
        cls.data = out['data']

    @classmethod
    def setup_method(self):
        pass

    def teardown_method(self):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def test01_get_local_extrema(self):

        lmethod = ['find_peaks', 'bspline']
        lreturnas = [bool, float]
        lreturn_min = [True, False]
        lreturn_prom = [True, False]

        for comb in itt.product(lmethod, lreturnas, lreturn_min, lreturn_prom):
            out = tfs.get_localextrema_1d(
                data=self.data[:10, :], lamb=self.lamb,
                width=None, weights=None,
                method=comb[0], returnas=comb[1],
                return_minima=comb[2],
                return_prominence=comb[3],
                return_width=None,
            )
