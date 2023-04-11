"""
This module contains tests for tofu.geom in its structured version
"""

# Built-in
import os
import shutil
import itertools as itt

# tofu-specific
import tofu.nist2tofu as tfn

_here = os.path.abspath(os.path.dirname(__file__))
VerbHead = 'tofu.nist2tofu.test_03_core'
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


class Test01_nist(object):

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

    def test01_search_online_by_wavelengthA(self):

        llambmin = [None, 3.94]
        llambmax = [None, 4.]
        lion = [None, 'H', ['ar', 'W44+']]

        lcache_from = [False, True]
        ldatacol = [False, True]

        # Search by searchstr
        llcomb = [
            llambmin, llambmax, lion,
            lcache_from,
            ldatacol,
        ]
        ii, itot = -1, 2*2*3*2*2
        for comb in itt.product(*llcomb):
            ii += 1
            if all([vv is None for vv in comb[:2]]):
                continue
            if comb[2] == 'H' and comb[1] is None:
                continue
            if comb[2] == 'H' and all([vv is not None for vv in comb[:2]]):
                continue
            if any([vv is None for vv in comb[:2]]) and comb[2] != 'H':
                continue
            print(f'{ii} / {itot}  -  {comb}')

            # out = tfn.step01_search_online_by_wavelengthA(
                # lambmin=comb[0],
                # lambmax=comb[1],
                # ion=comb[2],
                # verb=True,
                # return_dout=True,
                # return_dsources=True,
                # cache_from=comb[3],
                # cache_info=True,
                # format_for_DataStock=comb[4],
                # create_custom=True,
            # )
            # del out

    def test02_clear_cache(self):
        tfn.clear_cache()
