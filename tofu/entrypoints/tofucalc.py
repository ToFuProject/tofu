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
_TOFUPATH = os.path.dirname(os.path.dirname(_HERE))
istofugit = False
if '.git' in os.listdir(_TOFUPATH) and 'tofu' in _TOFUPATH:
    istofugit = True

if istofugit:
    # Make sure we load the corresponding tofu
    sys.path.insert(1, _TOFUPATH)
    import tofu as tf
    from tofu.imas2tofu import MultiIDSLoader
    _ = sys.path.pop(1)
else:
    import tofu as tf
    from tofu.imas2tofu import MultiIDSLoader

# import parser dict
sys.path.insert(1, _TOFUPATH)
from scripts._dparser import _DPARSER
_ = sys.path.pop(1)


# default parameters
pfe = os.path.join(os.path.expanduser('~'), '.tofu', '_scripts_def.py')
if os.path.isfile(pfe):
    # Make sure we load the user-specific file
    # sys.path method
    # sys.path.insert(1, os.path.join(os.path.expanduser('~'), '.tofu'))
    # import _scripts_def as _defscripts
    # _ = sys.path.pop(1)
    # importlib method
    import importlib.util
    spec = importlib.util.spec_from_file_location("_defscripts", pfe)
    _defscripts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_defscripts)
else:
    try:
        import tofu.entrypoints._def as _defscripts
    except Exception as err:
        from . import _def as _defscripts

tforigin = tf.__file__
tfversion = tf.__version__
print(tforigin, tfversion)

if 'imas2tofu' not in dir(tf):
    msg = "imas does not seem to be available\n"
    msg += "  => tf.imas2tofu not available\n"
    msg += "  => tofuplot not available"
    raise Exception(msg)



###################################################
###################################################
#       default values
###################################################


# _DDEF = {
    # # User-customizable
    # 'run': _defscripts._TFCALC_RUN,
    # 'user': _defscripts._TFCALC_USER,
    # 'tokamak': _defscripts._TFCALC_TOKAMAK,
    # 'version': _defscripts._TFCALC_VERSION,
    # 't0': _defscripts._TFCALC_T0,
    # 'tlim': None,
    # 'sharex': _defscripts._TFCALC_SHAREX,
    # 'bck': _defscripts._TFCALC_BCK,
    # 'extra': _defscripts._TFCALC_EXTRA,
    # 'indch_auto': _defscripts._TFCALC_INDCH_AUTO,
    # 'coefs': None,

    # # Non user-customizable
    # 'lids_diag': MultiIDSLoader._lidsdiag,
    # 'lids': MultiIDSLoader._lidsdiag,
# }


_DCONVERT = {
    'background': 'bck'
}


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


def call_tfcalcimas(shot=None, run=None, user=None,
                    tokamak=None, version=None,
                    tokamak_eq=None, user_eq=None,
                    shot_eq=None, run_eq=None,
                    tokamak_prof=None, user_prof=None,
                    shot_prof=None, run_prof=None,
                    ids=None, t0=None, tlim=None, extra=None,
                    plot_compare=True, Brightness=None,
                    res=None, interp_t=None, coefs=None,
                    sharex=None, indch=None, indch_auto=None,
                    input_file=None, output_file=None,
                    background=None):

    # --------------
    # Check inputs
    kwd = locals()
    for k0 in set(_DDEF.keys()).intersection(kwd.keys()):
        if kwd[k0] is None:
            kwd[k0] = _DDEF[k0]
    for k0 in set(_DCONVERT.keys()).intersection(kwd.keys()):
        kwd[_DCONVERT[k0]] = kwd[k0]
        del kwd[k0]

    if isinstance(kwd['t0'], str) and kwd['t0'].lower() == 'none':
        kwd['t0'] = None
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
    # Parse input arguments
    msg = """Fast interactive visualization tool for diagnostics data in
    imas

    This is merely a wrapper around the function tofu.calc_from_imas()
    It calculates synthetic signal (from imas) and displays it from the following
    ids:
        %s
    """%repr(_DDEF['lids'])

    parser = _DPARSER['calc'](_DDEF, msg)
    args = parser.parse_args()

    # Call wrapper function
    call_tfcalcimas(**dict(args._get_kwargs()))


# Add this to make sure it remains executable even without install
if __name__ == '__main__':
    main()
