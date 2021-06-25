#!/usr/bin/env python

# Built-in
import sys
import os


# Generic
import matplotlib.pyplot as plt
plt.switch_backend('Qt5Agg')
plt.ioff()


# tofu
# test if in a tofu git repo
_HERE = os.path.abspath(os.path.dirname(__file__))
_TOFUPATH = os.path.dirname(os.path.dirname(_HERE))
istofugit = False
if '.git' in os.listdir(_TOFUPATH) and 'tofu' in _TOFUPATH:
    istofugit = True

if istofugit:
    # Make sure we load the corresponding tofu
    sys.path.insert(1, _TOFUPATH)
    import tofu as tf
    _ = sys.path.pop(1)
else:
    import tofu as tf


# import parser dict
sys.path.insert(1, _TOFUPATH)
from scripts._dparser import _DPARSER
_ = sys.path.pop(1)


# tforigin = tf.__file__
# tfversion = tf.__version__
# print(tforigin, tfversion)


if 'imas2tofu' not in dir(tf):
    msg = ("imas does not seem to be available\n"
           + "  => tf.imas2tofu not available\n"
           + "  => tofuplot not available")
    raise Exception(msg)


###################################################
###################################################
#       default values
###################################################


_DCONVERT = {
    'background': 'bck',
    'quantity': 'plot_sig',
    'X': 'plot_X',
}


###################################################
###################################################
#       function
###################################################


def call_tfloadimas(shot=None, run=None, user=None,
                    database=None, version=None, extra=None,
                    ids=None, quantity=None, X=None, t0=None, tlim=None,
                    sharex=None, indch=None, indch_auto=None,
                    background=None, t=None, ddef=None,
                    config=None, tosoledge3x=None,
                    mag_init_pts=None, mag_sep_dR=None, mag_sep_nbpts=None):

    # --------------
    # Check inputs
    kwd = locals()
    for k0 in set(ddef.keys()).intersection(kwd.keys()):
        if kwd[k0] is None:
            kwd[k0] = ddef[k0]
    for k0 in set(_DCONVERT.keys()).intersection(kwd.keys()):
        kwd[_DCONVERT[k0]] = kwd[k0]
        del kwd[k0]
    del kwd['ddef']

    if kwd['tlim'] is not None and len(kwd['tlim']) == 1:
        kwd['tlim'] = [kwd['tlim'], None]
    if kwd['tlim'] is not None and len(kwd['tlim']) != 2:
        msg = ("tlim must contain 2 limits:\n"
               + "\t- provided: {}".format(kwd['tlim']))
        raise Exception(msg)

    # --------------
    # run
    tf.load_from_imas(plot=True, **kwd)
    plt.show(block=True)


###################################################
###################################################
#       bash call (main)
###################################################


# if __name__ == '__main__':
def main():
    # Parse input arguments
    # Instanciate parser
    ddef, parser = _DPARSER['plot']()

    # Parse arguments
    args = parser.parse_args()

    # Call wrapper function
    call_tfloadimas(ddef=ddef, **dict(args._get_kwargs()))


# Add this to make sure it remains executable even without install
if __name__ == '__main__':
    main()
