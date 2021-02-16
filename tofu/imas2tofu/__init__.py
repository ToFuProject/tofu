# -*- coding: utf-8 -*-
#! /usr/bin/python
"""
The imas-compatibility module of tofu

"""
import warnings
import traceback
import itertools as itt

try:
    try:
        from tofu.imas2tofu._core import *
        from tofu.imas2tofu._mat2ids2calc import *
    except Exception:
        from ._core import *
        from ._mat2ids2calc import *
except Exception as err:
    if str(err) == 'imas not available':
        msg = ""
        msg += "\n\nIMAS python API issue\n"
        msg += "imas could not be imported into tofu ('import imas' failed):\n"
        msg += "  - it may not be installed (optional dependency)\n"
        msg += "  - or you have loaded the wrong working environment\n\n"
        msg += "    => the optional sub-package tofu.imas2tofu is not usable\n"
    else:
        msg = str(traceback.format_exc())
        msg += "\n\n    => the optional sub-package tofu.imas2tofu is not usable\n"
    raise Exception(msg)


# -----------------------------------------------
#   Check IMAS version vs latest available in linux modules
# -----------------------------------------------

_KEYSTR = 'IMAS/'


# extract all IMAS versions from a str returned by modules
def extractIMAS(ss, keystr=_KEYSTR):
    if keystr not in ss:
        raise Exception
    ls = ss[ss.index(keystr):].split('\n')
    ls = itt.chain.from_iterable([s.split(' ') for s in ls])
    ls = [s for s in ls if keystr in s]
    ls = [s[len(keystr):s.index('(')] if '(' in s else s[len(keystr):]
          for s in ls]
    return sorted(ls)


# Compare current and latest available IMAS versions
def check_IMAS_version(verb=True, keystr=_KEYSTR):
    import subprocess

    # Get currently loaded IMAS
    cmd = "module list"
    proc = subprocess.run(cmd, check=True, shell=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lcur = extractIMAS(proc.stdout.decode(), keystr=keystr)
    if len(lcur) != 1:
        msg = ("You seem to have no / several IMAS version loaded:\n"
               + "\t- module list: {}".format(lcur))
        raise Exception(msg)

    # Get all available IMAS
    cmd = "module av IMAS"
    proc = subprocess.run(cmd, check=True, shell=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lav = extractIMAS(proc.stdout.decode(), keystr=keystr)
    if len(lav) == 0:
        msg = "There is not available IMAS version"
        raise Exception(msg)

    # Compare and warn
    if lcur[0] not in lav:
        msg = "The current IMAS version is not available!"
        raise Exception(msg)

    msg = None
    c0 = (lav.index(lcur[0]) != len(lav)-1
          and lcur[0]+'.bak' != lav[-1])
    if c0:
        msg = ("\nYou do not seem to be using the latest IMAS version:\n"
               + "'module list' vs 'module av IMAS' suggests:\n"
               + "\t- Current version: {}\n".format(lcur[0])
               + "\t- Latest version : {}".format(lav[-1]))
        warnings.warn(msg)
    return lcur[0], lav


# Try comparing and warning
try:
    _, _ = check_IMAS_version(verb=True)
except Exception as err:
    # This warning is an optional luxury, should not block anything
    pass


__all__ = ['MultiIDSLoader', 'load_Config', 'load_Plasma2D',
           'load_Cam', 'load_Data']
del warnings, traceback, itt, _KEYSTR
