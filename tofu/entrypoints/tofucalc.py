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


# User-customizable
_RUN = _defscripts._TFCALC_RUN
_USER = _defscripts._TFCALC_USER
_TOKAMAK = _defscripts._TFCALC_TOKAMAK
_VERSION = _defscripts._TFCALC_VERSION
_T0 = _defscripts._TFCALC_T0
_TLIM = None
_SHAREX = _defscripts._TFCALC_SHAREX
_BCK = _defscripts._TFCALC_BCK
_EXTRA = _defscripts._TFCALC_EXTRA
_INDCH_AUTO = _defscripts._TFCALC_INDCH_AUTO
_COEFS = None

# Non user-customizable
_LIDS_DIAG = MultiIDSLoader._lidsdiag
_LIDS = _LIDS_DIAG


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
                    tokamak_eq=None, user_eq=None,
                    shot_eq=None, run_eq=None,
                    tokamak_prof=None, user_prof=None,
                    shot_prof=None, run_prof=None,
                    ids=None, t0=_T0, tlim=_TLIM, extra=_EXTRA,
                    plot_compare=True, Brightness=None,
                    res=None, interp_t=None, coefs=_COEFS,
                    sharex=_SHAREX, indch=None, indch_auto=_INDCH_AUTO,
                    input_file=None, output_file=None,
                    background=_BCK):

    if t0.lower() == 'none':
        t0 = None
    if tlim is not None and len(tlim) == 1:
        tlim = [tlim, None]
    if tlim is not None and len(tlim) != 2:
        msg = ("tlim must contain 2 limits:\n"
               + "\t- provided: {}".format(tlim))
        raise Exception(msg)

    tf.calc_from_imas(shot=shot, run=run, user=user,
                      tokamak=tokamak, version=version,
                      tokamak_eq=tokamak_eq, user_eq=user_eq,
                      shot_eq=shot_eq, run_eq=run_eq,
                      tokamak_prof=tokamak_prof, user_prof=user_prof,
                      shot_prof=shot_prof, run_prof=run_prof,
                      ids=ids, indch=indch, indch_auto=indch_auto,
                      plot_compare=plot_compare, extra=extra, coefs=coefs,
                      tlim=tlim,
                      Brightness=Brightness, res=res, interp_t=interp_t,
                      input_file=input_file, output_file=output_file,
                      t0=t0, plot=None, sharex=sharex, bck=background)

    plt.show(block=True)



###################################################
###################################################
#       bash call (main)
###################################################


def _str2bool(v):
    if isinstance(v, bool):
        return v
    elif v.lower() in ['yes', 'true', 'y', 't', '1']:
        return True
    elif v.lower() in ['no', 'false', 'n', 'f', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected !')


def _str2tlim(v):
    c0 = (v.isdigit()
          or ('.' in v
              and len(v.split('.')) == 2
              and all([vv.isdigit() for vv in v.split('.')])))
    if c0 is True:
        v = float(v)
    elif v.lower() == 'none':
        v = None
    return v


# if __name__ == '__main__':
def main():
    # Parse input arguments
    msg = """Fast interactive visualization tool for diagnostics data in
    imas

    This is merely a wrapper around the function tofu.calc_from_imas()
    It calculates synthetic signal (from imas) and displays it from the following
    ids:
        %s
    """%repr(_LIDS)
    parser = argparse.ArgumentParser(description = msg)

    # Main idd parameters
    parser.add_argument('-s', '--shot', type=int,
                        help='shot number', required=True)
    msg = 'username of the DB where the datafile is located'
    parser.add_argument('-u', '--user',
                        help=msg, required=False, default=_USER)
    msg = 'tokamak name of the DB where the datafile is located'
    parser.add_argument('-tok', '--tokamak', help=msg, required=False,
                        default=_TOKAMAK)
    parser.add_argument('-r', '--run', help='run number',
                        required=False, type=int, default=_RUN)
    parser.add_argument('-v', '--version', help='version number',
                        required=False, type=str, default=_VERSION)

    # Equilibrium idd parameters
    parser.add_argument('-s_eq', '--shot_eq', type=int,
                        help='shot number for equilibrium, defaults to -s',
                        required=False, default=None)
    msg = 'username for the equilibrium, defaults to -u'
    parser.add_argument('-u_eq', '--user_eq',
                        help=msg, required=False, default=None)
    msg = 'tokamak for the equilibrium, defaults to -tok'
    parser.add_argument('-tok_eq', '--tokamak_eq',
                        help=msg, required=False, default=None)
    parser.add_argument('-r_eq', '--run_eq',
                        help='run number for the equilibrium, defaults to -r',
                        required=False, type=int, default=None)

    # Profile idd parameters
    parser.add_argument('-s_prof', '--shot_prof', type=int,
                        help='shot number for profiles, defaults to -s',
                        required=False, default=None)
    msg = 'username for the profiles, defaults to -u'
    parser.add_argument('-u_prof', '--user_prof',
                        help=msg, required=False, default=None)
    msg = 'tokamak for the profiles, defaults to -tok'
    parser.add_argument('-tok_prof', '--tokamak_prof',
                        help=msg, required=False, default=None)
    parser.add_argument('-r_prof', '--run_prof',
                        help='run number for the profiles, defaults to -r',
                        required=False, type=int, default=None)

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
    parser.add_argument('-tl', '--tlim', type=_str2tlim,
                        required=False,
                        help='limits of the time interval',
                        nargs='+', default=_TLIM)
    parser.add_argument('-c', '--coefs', type=float, required=False,
                        help='Corrective coefficient, if any',
                        default=_COEFS)
    parser.add_argument('-ich', '--indch', type=int, required=False,
                        help='indices of channels to be loaded',
                        nargs='+', default=None)
    parser.add_argument('-ichauto', '--indch_auto', type=bool, required=False,
                        help='automatically determine indices of channels to be loaded',
                        default=_INDCH_AUTO)
    parser.add_argument('-e', '--extra', type=_str2bool, required=False,
                        help='If True loads separatrix and heating power',
                        default=_EXTRA)
    parser.add_argument('-sx', '--sharex', type=_str2bool, required=False,
                        help='Should X axis be shared between diagnostics ids ?',
                        default=_SHAREX, const=True, nargs='?')
    parser.add_argument('-if', '--input_file', type=str, required=False,
                        help='mat file from which to load core_profiles',
                        default=None)
    parser.add_argument('-of', '--output_file', type=str, required=False,
                        help='mat file into which to save synthetic signal',
                        default=None)
    parser.add_argument('-bck', '--background', type=_str2bool, required=False,
                        help='Plot data enveloppe as grey background ?',
                        default=_BCK, const=True, nargs='?')

    args = parser.parse_args()

    # Call wrapper function
    call_tfcalcimas(**dict(args._get_kwargs()))


# Add this to make sure it remains executable even without install
if __name__ == '__main__':
    main()
