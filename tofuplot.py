#!/usr/bin/env python

# Built-in
import argparse

# Generic
import matplotlib.pyplot as plt
plt.switch_backend('Qt5Agg')
plt.ioff()

# tofu-specific
import tofu as tf

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
_LIDS_DIAG = ['ece', 'reflectometer_profile', 'interferometer',
              'bolometer', 'soft_x_rays', 'bremsstrahlung_visible',
              'spectrometer_visible']
_LIDS_PLASMA = ['core_profiles']
_LIDS = _LIDS_DIAG + _LIDS_PLASMA

###################################################
###################################################
#       function
###################################################

def call_tfloadimas(shot=None, run=_RUN, user=_USER, tokamak=_TOKAMAK,
                    ids=None, quantity=None, quant_X=None):

    if ids in _LIDS_DIAG:
        tf.load_from_imas(shot=shot, run=run, user=user, tokamak=tokamak,
                          ids=ids, plot=True)
    else:
        c0 = quantity is None or quant_X is None
        if c0:
            msg = "quantity and quant_X must be provided to plot a plasma profile!"
            raise Exception(msg)
        dq = tf.imas2tofu.MultiIDSLoader._dshort[ids]
        lk = sorted(dq.keys())
        c1 = quantity not in lk or quant_X not in lk
        if c1:
            msg = tf.imas2tofu.MultiIDSLoader._shortcuts(ids=ids, verb=False,
                                                         return_=True)
            col = ['ids', 'shortcut', 'long version']
            msg = obj._getcharray(msg, col)
            msg = """\nArgs quantity and quant_X must be valid tofu shortcuts
            to quantities in ids %s\n\nAvailable shortcuts are:\n"""%ids + msg
            raise Exception(msg)

        tf.load_from_imas(shot=shot, run=run, user=user, tokamak=tokamak,
                          ids=ids, plot_sig=quantity, plot_X=quant_X, plot=True)
    plt.show(block=True)



###################################################
###################################################
#       bash call (main)
###################################################


if __name__ == '__main__':

    # Parse input arguments
    msg = """Fast interactive visualization tool for diagnostics data in
    imas

    This is merely a wrapper around the function tofu.load_from_imas()
    It loads (from imas) and displays diagnostics data from the following
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

    msg = "ids from which to load diagnostics data, can be:\n%s"%repr(_LIDS)
    parser.add_argument('-i', '--ids', type=str, required=True,
                        help=msg, choices=_LIDS)
    parser.add_argument('-q', '--quantity', type=str, required=False,
                        help='Desired quantity from the plasma ids', default=None)
    parser.add_argument('-qX', '--quant_X', type=str, required=False,
                        help='Quantity from the plasma ids to use for abscissa', default=None)

    args = parser.parse_args()

    # Call wrapper function
    call_tfloadimas(**dict(args._get_kwargs()))
