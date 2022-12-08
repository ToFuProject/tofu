"""
This module contains tests for tofu.geom in its structured version
"""

# Built-in
import os
import warnings
import shutil
import subprocess
from subprocess import PIPE
import itertools as itt

# Standard
import numpy as np
import matplotlib.pyplot as plt

# tofu-specific
from tofu import __version__
import tofu.openadas2tofu as tfoa

_here = os.path.abspath(os.path.dirname(__file__))
VerbHead = 'tofu.openadas2tofu.test_03_core'
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


class Test01_openadas(object):

    @classmethod
    def setup_class(cls):
        pass

    @classmethod
    def setup_method(self):
        pass

    def teardown_method(self):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def test01_search_online(self):

        # Search by searchstr
        out = tfoa.step01_search_online(
            'ar+16 ADF15',
            verb=True,
            returnas=False,
        )
        assert out is None

        out = tfoa.step01_search_online(
            'ar+16 ADF11',
            verb=False,
            returnas=np.ndarray,
        )
        assert isinstance(out, np.ndarray)

        # Search by wavelength
        lret = [None, str, np.ndarray]
        lverb = [True, False]
        lelement = [None, 'ar', ['ar', 'w'], ('w',)]
        lcharge = [None, 14, [15, 16], (16,)]
        lres = ['transition', 'file']
        for comb in itt.product(lret, lverb, lelement, lcharge, lres):
            out = tfoa.step01_search_online_by_wavelengthA(
                lambmin=3.94, lambmax=4.,
                returnas=comb[0], verb=comb[1],
                element=comb[2], charge=comb[3],
                resolveby=comb[4],
            )
            assert out is comb[0] or isinstance(out, comb[0])

    def test02_download(self):
        out = tfoa.step02_download(
            filename='/adf15/pec40][ar/pec40][ar_ic][ar16.dat',
            update=False,
            verb=True,
            returnas=False,
        )
        assert out is None

        out = tfoa.step02_download(
            filename='/adf15/pec40][ar/pec40][ar_ic][ar16.dat',
            update=True,
            verb=False,
            returnas=str,
        )
        assert isinstance(out, str)

        out = tfoa.step02_download_all(
            lambmin=3.94,
            lambmax=4,
            element='Mo',
            update=False,
            verb=True,
        )
        assert out is None

        lf = [
            '/adf15/pec40][ar/pec40][ar_ic][ar16.dat',
            '/adf11/scd74/scd74_ar.dat',
            '/adf15/pec40][w/pec40][w_ls][w64.dat',
            '/adf11/plt41/plt41_xe.dat',
        ]
        out = tfoa.step02_download_all(
            files=lf,
            update=False,
            verb=True,
        )
        assert out is None

        out = tfoa.step02_download_all(
            searchstr='ar+16 ADF15',
            include_partial=False,
            update=False,
            verb=True,
        )
        assert out is None

    def test03_readfiles(self):
        out = tfoa.step03_read(
            '/adf15/pec40][ar/pec40][ar_ic][ar16.dat',
        )
        assert isinstance(out, dict)

        out = tfoa.step03_read(
            'adf11/plt41/plt41_xe.dat',
        )
        assert isinstance(out, dict)

        out = tfoa.step03_read_all(
            element='ar',
            charge=16,
            typ1='adf15',
            lambmin=3.94,
            lambmax=4.,
        )
        assert isinstance(out, dict)

        out = tfoa.step03_read_all(
            element='ar',
            charge=16,
            typ1='adf11',
            typ2='scd',
        )
        assert isinstance(out, dict)

        out = tfoa.step03_read_all(
            element='ar',
            charge=16,
            typ1='adf11',
            typ2='plt',
        )
        assert isinstance(out, dict)

    def test04_clear_downloads(self):
        tfoa.clear_downloads()
