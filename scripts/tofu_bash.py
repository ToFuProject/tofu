#!/usr/bin/env python

# Built-in
import sys
import os
import argparse


_HERE = os.path.abspath(os.path.dirname(__file__))


# import parser dict
sys.path.insert(1, _HERE)
from _dparser import _DPARSER
_ = sys.path.pop(1)

_TOFUPATH = os.path.dirname(_HERE)
_ENTRYPOINTS_PATH = os.path.join(_TOFUPATH, 'tofu', 'entrypoints')


###################################################
###################################################
#       default values
###################################################


_LOPTIONS = ['--version', 'custom', 'plot', 'calc']
_LOPSTRIP = [ss.strip('--') for ss in _LOPTIONS]


###################################################
###################################################
#       function
###################################################


def tofu_bash(option=None, ddef=None, **kwdargs):
    """ Print tofu version and / or store in environment variable """

    # --------------
    # Check inputs
    if option not in _LOPSTRIP:
        msg = ("Provided option is not acceptable:\n"
               + "\t- available: {}\n".format(_LOPSTRIP)
               + "\t- provided:  {}".format(option))
        raise Exception(msg)

    # --------------
    # call corresponding bash command
    if option == 'version':
        sys.path.insert(1, _HERE)
        import tofuversion
        _ = sys.path.pop(1)
        tofuversion.get_version(ddef=ddef, **kwdargs)

    elif option == 'custom':
        sys.path.insert(1, _HERE)
        import tofucustom
        _ = sys.path.pop(1)
        tofucustom.custom(ddef=ddef, **kwdargs)

    elif option == 'plot':
        sys.path.insert(1, _ENTRYPOINTS_PATH)
        import tofuplot
        _ = sys.path.pop(1)
        tofuplot.call_tfloadimas(ddef=ddef, **kwdargs)

    elif option == 'calc':
        sys.path.insert(1, _ENTRYPOINTS_PATH)
        import tofucalc
        _ = sys.path.pop(1)
        tofucalc.call_tfcalcimas(ddef=ddef, **kwdargs)


###################################################
###################################################
#       bash call (main)
###################################################


def main():
    # Parse input arguments
    msg = """ Get tofu version from bash optionally set an enviroment variable

    If run from a git repo containing tofu, simply returns git describe
    Otherwise reads the tofu version stored in tofu/version.py

    """

    # Instanciate parser
    parser = argparse.ArgumentParser(description=msg)

    # Define input arguments
    parser.add_argument('option',
                        nargs='?',
                        type=str,
                        default='None')
    parser.add_argument('-v', '--version',
                        help='get tofu current version',
                        required=False,
                        action='store_true')
    parser.add_argument('kwd', nargs='?', type=str, default='None')

    if sys.argv[1] not in _LOPTIONS:
        msg = ("Provided option is not acceptable:\n"
               + "\t- available: {}\n".format(_LOPTIONS)
               + "\t- provided:  {}".format(sys.argv[1]))
        raise Exception(msg)
    if len(sys.argv) > 2:
        if any([ss in sys.argv[2:] for ss in _LOPTIONS]):
            lopt = [ss for ss in sys.argv[1:] if ss in _LOPTIONS]
            msg = ("Only one option can be provided!\n"
                   + "\t- provided: {}".format(lopt))
            raise Exception(msg)

    option = sys.argv[1].strip('--')
    ddef, parser = _DPARSER[option]()
    if len(sys.argv) > 2:
        kwdargs = dict(parser.parse_args(sys.argv[2:])._get_kwargs())
    else:
        kwdargs = {}

    # Call function
    tofu_bash(option=option, ddef=ddef, **kwdargs)


if __name__ == '__main__':
    main()
