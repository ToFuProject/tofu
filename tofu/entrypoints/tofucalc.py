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
           + "  => tofucalc not available")
    raise Exception(msg)


###################################################
###################################################
#       default values
###################################################


_DCONVERT = {
    'background': 'bck'
}


###################################################
###################################################
#       function
###################################################


def call_tfcalcimas(shot=None, run=None, user=None,
                    database=None, version=None,
                    database_eq=None, user_eq=None,
                    shot_eq=None, run_eq=None,
                    database_prof=None, user_prof=None,
                    shot_prof=None, run_prof=None,
                    ids=None, t0=None, tlim=None, extra=None,
                    plot_compare=True, Brightness=None,
                    res=None, interp_t=None, coefs=None,
                    sharex=None, indch=None, indch_auto=None,
                    input_file=None, output_file=None,
                    background=None, ddef=None):

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
    tf.calc_from_imas(plot=None, **kwd)
    plt.show(block=True)


###################################################
###################################################
#       bash call (main)
###################################################


# if __name__ == '__main__':
def main():

    # Instanciate parser
    ddef, parser = _DPARSER['calc']()

    # Parse arguments
    args = parser.parse_args()

    # Call wrapper function
    call_tfcalcimas(ddef=ddef, **dict(args._get_kwargs()))


# Add this to make sure it remains executable even without install
if __name__ == '__main__':
    main()
