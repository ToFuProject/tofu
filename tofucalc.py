#!/usr/bin/env python

# Built-in
import sys
import os
import argparse

# Generic
import matplotlib.pyplot as plt
plt.switch_backend('Qt5Agg')
plt.ioff()

# tofu
# test if in a tofu git repo
_HERE = os.path.abspath(os.path.dirname(__file__))
istofugit = False
if '.git' in _HERE and 'tofu' in _HERE:
    istofugit = True

if istofugit:
    # Make sure we load the corresponding tofu
    sys.path.insert(1,_HERE)
    import tofu as tf
    from tofu.imas2tofu import MultiIDSLoader
    _ = sys.path.pop(1)
else:
    import tofu as tf
    from tofu.imas2tofu import MultiIDSLoader
tforigin = tf.__file__
tfversion = tf.__version__

# if tf.__version__ < '1.4.1':
    # msg = "tofuplot only works with tofu >= 1.4.1"
    # raise Exception(msg)

if 'imas2tofu' not in dir(tf):
    msg = "imas does not seem to be available\n"
    msg += "  => tf.imas2tofu not available\n"
    msg += "  => tofuplot not available"
    raise Exception(msg)



###################################################
###################################################
#       default values
###################################################

_RUN = 0
_USER = 'imas_public'
_TOKAMAK = 'west'
_VERSION = '3'
_LIDS_DIAG = MultiIDSLoader._lidsdiag
_LIDS = _LIDS_DIAG
_T0 = 'IGNITRON'
_SHAREX = False
_BCK = True

###################################################
###################################################
#       function
###################################################

def _get_exception(q, ids, qtype='quantity'):
    msg = MultiIDSLoader._shortcuts(ids=ids,
                                    verb=False, return_=True)
    col = ['ids', 'shortcut', 'long version']
    msg = MultiIDSLoader._getcharray(msg, col)
    msg = """\nArgs quantity and quant_X must be valid tofu shortcuts
    to quantities in ids %s\n\n"""
    msg += "Available shortcuts are:\n"""%ids + msg
    msg += "\n\nProvided:\n    - %s: %s\n"%(qtype,str(qq))
    raise Exception(msg)


def call_tfcalcimas(shot=None, run=_RUN, user=_USER,
                    tokamak=_TOKAMAK, version=_VERSION,
                    ids=None, t0=_T0,
                    plot_compare=True, Brightness=None,
                    res=None, interp_t=None,
                    sharex=_SHAREX, indch=None, indch_auto=None,
                    background=_BCK):

    if t0.lower() == 'none':
        t0 = None

    tf.calc_from_imas(shot=shot, run=run, user=user,
                      tokamak=tokamak, version=version,
                      ids=ids, indch=indch, indch_auto=indch_auto,
                      plot_compare=plot_compare,
                      Brightness=Brightness, res=res, interp_t=interp_t,
                      t0=t0, plot=True, sharex=sharex, bck=background)

    plt.show(block=True)



###################################################
###################################################
#       bash call (main)
###################################################

def _str2bool(v):
    if isinstance(v,bool):
        return v
    elif v.lower() in ['yes','true','y','t','1']:
        return True
    elif v.lower() in ['no','false','n','f','0']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected !')


if __name__ == '__main__':

    # Parse input arguments
    msg = """Fast interactive visualization tool for diagnostics data in
    imas

    This is merely a wrapper around the function tofu.calc_from_imas()
    It calculates synthetic signal (from imas) and displays it from the following
    ids:
        %s
    """%repr(_LIDS)
    parser = argparse.ArgumentParser(description = msg)

    parser.add_argument('-s', '--shot', type=int,
                        help='shot number', required=True)
    msg = 'username of the DB where the datafile is located'
    parser.add_argument('-u','--user',help=msg, required=False, default=_USER)
    msg = 'tokamak name of the DB where the datafile is located'
    parser.add_argument('-t','--tokamak',help=msg, required=False,
                        default=_TOKAMAK)
    parser.add_argument('-r','--run',help='run number',
                        required=False, type=int, default=_RUN)
    parser.add_argument('-v','--version',help='version number',
                        required=False, type=str, default=_VERSION)

    msg = "ids from which to load diagnostics data, can be:\n%s"%repr(_LIDS)
    parser.add_argument('-i', '--ids', type=str, required=True,
                        help=msg, nargs='+', choices=_LIDS)
    parser.add_argument('-B', '--Brightness', type=bool, required=False,
                        help='Whether to express result as brightness',
                        default=None)
    parser.add_argument('-res', '--res', type=float, required=False,
                        help='Space resolution for the LOS-discretization',
                        default=None)
    parser.add_argument('-t0', '--t0', type=str, required=False,
                        help='Reference time event setting t = 0', default=_T0)
    parser.add_argument('-ich', '--indch', type=int, required=False,
                        help='indices of channels to be loaded',
                        nargs='+', default=None)
    parser.add_argument('-ichauto', '--indch_auto', type=bool, required=False,
                        help='automatically determine indices of channels to be loaded',
                        default=True)
    parser.add_argument('-sx', '--sharex', type=_str2bool, required=False,
                        help='Should X axis be shared between diagnostics ids ?',
                        default=_SHAREX, const=True, nargs='?')
    parser.add_argument('-bck', '--background', type=_str2bool, required=False,
                        help='Plot data enveloppe as grey background ?',
                        default=_BCK, const=True, nargs='?')

    args = parser.parse_args()

    # Call wrapper function
    call_tfcalcimas(**dict(args._get_kwargs()))
