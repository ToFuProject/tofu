
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


class Test01_RockingCurve(object):

    @classmethod
    def setup_class(cls):

        cls.lc = [
            k0 for k0 in tfs._rockingcurve_def._DCRYST.keys()
            if 'xxx' not in k0.lower()
        ]

    def setup_method(self):
        pass

    def teardown_method(self):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def test01_rockingcurve_compute(self):

        for k0 in self.lc:
            dout = tfs.compute_rockingcurve(
                crystal=k0,
                lamb=np.r_[3.969067],
                miscut=False,
                alpha_limits=None,
                therm_exp=False,
                temp_limits=None,
                # Plot
                plot_therm_exp=False,
                plot_asf=False,
                plot_power_ratio=False,
                plot_asymmetry=False,
                plot_cmaps=False,
                # return
                returnas=dict,
            )
