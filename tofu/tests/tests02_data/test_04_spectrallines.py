
"""
This module contains tests for tofu.geom in its structured version
"""


# Built-in
import os
import warnings


# Standard
import numpy as np
import scipy.constants as scpct
import matplotlib.pyplot as plt


# tofu-specific
from tofu import __version__
import tofu.data as tfd
import tofu.utils as tfu

_here = os.path.abspath(os.path.dirname(__file__))
VerbHead = 'tofu.data.SpectralLines'


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


# #############################################################################
# #############################################################################
#           Specific to SpectralLines
# #############################################################################


class Test01_SpectralLines(object):

    @classmethod
    def setup_class(cls, Name='data1',  SavePath='./', verb=False):
        cls.sl = tfd.SpectralLines.from_openadas(
            lambmin=3.94e-10,
            lambmax=4e-10,
            element=['Ar', 'W'],
        )

    @classmethod
    def setup_method(self):
        pass

    def teardown_method(self):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def test01_add_from_openadas(self):
        lines = self.sl.dobj['lines']
        self.sl.add_from_openadas(
            lambmin=3.90e-10,
            lambmax=3.96e-10,
            element='W',
        )
        assert all([k0 in self.sl.dobj['lines'].keys() for k0 in lines.keys()])

    def test02_sortby(self):
        self.sl.sortby(param='lambda0', which='lines')
        self.sl.sortby(param='ion', which='lines')

    def test03_convert_lines(self):
        self.sl.convert_lines(units='Hz')

    def test04_calc_pec(self):
        ne = np.r_[1e15, 1e18, 1e21]
        Te = np.r_[1e3, 2e3, 3e3, 4e3, 5e3]
        dpec = self.sl.calc_pec(ne=ne, Te=Te[:ne.size], grid=False)
        dpec = self.sl.calc_pec(
            key='Ar16_9_oa_pec40_cl', ne=ne, Te=Te[:ne.size], grid=False,
        )
        dpec = self.sl.calc_pec(ne=ne, Te=Te, grid=True)
        dpec = self.sl.calc_pec(
            key='Ar16_9_oa_pec40_cl', ne=ne, Te=Te[:ne.size], grid=False,
        )

    def test05_calc_intensity(self):
        ne = np.r_[1e15, 1e18, 1e21]
        Te = np.r_[1e3, 2e3, 3e3, 4e3, 5e3]

        concentration = np.r_[0.1, 0.2, 0.3]
        dint = self.sl.calc_intensity(
            ne=ne, Te=Te[:ne.size], concentration=concentration, grid=False,
        )

        key = ['Ar16_9_oa_pec40_cl']
        concentration = {k0: np.r_[0.1, 0.2, 0.3] for k0 in key}
        dint = self.sl.calc_intensity(
            key=key,
            ne=ne, Te=Te[:ne.size], concentration=concentration, grid=False,
        )

        concentration = np.random.random((ne.size, Te.size))
        dint = self.sl.calc_intensity(
            ne=ne, Te=Te, concentration=concentration, grid=True,
        )

        key = ['Ar16_9_oa_pec40_cl']
        concentration = {k0: concentration for k0 in key}
        dint = self.sl.calc_intensity(
            key=key,
            ne=ne, Te=Te, concentration=concentration, grid=True,
        )

    def test06_plot_spectral_lines(self):
        ax = self.sl.plot_spectral_lines()
        plt.close('all')

    def test07_plot_pec_single(self):
        Te = 1.e3
        ne = 1.e20
        ax = self.sl.plot_pec_single(Te=Te, ne=ne)

    # def test08_plot_pec(self):
        # Te = np.linspace(1, 7, 7)*1e3
        # ne = np.logspace(15, 21, 7)
        # ax = self.sl.plot_pec(Te=1e3, ne=ne)
        # ax = self.sl.plot_pec(Te=Te, ne=1e19)
        # ax = self.sl.plot_pec(Te=Te, ne=ne)
        # plt.close('all')
