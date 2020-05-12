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
    msg = ("imas does not seem to be available\n"
           + "  => tf.imas2tofu not available\n"
           + "  => tofuplot not available")
    raise Exception(msg)



###################################################
###################################################
#       default values
###################################################


# User-customizable
_RUN = _defscripts._TFPLOT_RUN
_USER = _defscripts._TFPLOT_USER
_TOKAMAK = _defscripts._TFPLOT_TOKAMAK
_VERSION = _defscripts._TFPLOT_VERSION
_T0 = _defscripts._TFPLOT_T0
_TLIM = None
_SHAREX = _defscripts._TFPLOT_SHAREX
_BCK = _defscripts._TFPLOT_BCK
_EXTRA = _defscripts._TFPLOT_EXTRA
_INDCH_AUTO = _defscripts._TFPLOT_INDCH_AUTO

# Not user-customizable
_LIDS_DIAG = MultiIDSLoader._lidsdiag
_LIDS_PLASMA = tf.imas2tofu.MultiIDSLoader._lidsplasma
_LIDS = _LIDS_DIAG + _LIDS_PLASMA + tf.utils._LIDS_CUSTOM


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


def call_tfloadimas(shot=None, run=_RUN, user=_USER,
                    tokamak=_TOKAMAK, version=_VERSION, extra=_EXTRA,
                    ids=None, quantity=None, X=None, t0=_T0, tlim=_TLIM,
                    sharex=_SHAREX, indch=None, indch_auto=_INDCH_AUTO,
                    background=_BCK, t=None, dR_sep=None, init=None):

    lidspla = [ids_ for ids_ in ids if ids_ in _LIDS_PLASMA]
    if t0.lower() == 'none':
        t0 = None
    if tlim is not None and len(tlim) == 1:
        tlim = [tlim, None]
    if tlim is not None and len(tlim) != 2:
        msg = ("tlim must contain 2 limits:\n"
               + "\t- provided: {}".format(tlim))
        raise Exception(msg)

    tf.load_from_imas(shot=shot, run=run, user=user,
                      tokamak=tokamak, version=version,
                      ids=ids, indch=indch, indch_auto=indch_auto,
                      plot_sig=quantity, plot_X=X, extra=extra,
                      t0=t0, plot=True, sharex=sharex, bck=background,
                      t=t, tlim=tlim, dR_sep=dR_sep, init=init)

    plt.show(block=True)



###################################################
###################################################
#       bash call (main)
###################################################

def _str2bool(v):
    if isinstance(v, bool):
        return v
    elif v.lower() in ['yes','true','y','t','1']:
        return True
    elif v.lower() in ['no','false','n','f','0']:
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

    This is merely a wrapper around the function tofu.load_from_imas()
    It loads (from imas) and displays diagnostics data from the following
    ids:
        %s
    """%repr(_LIDS)
    parser = argparse.ArgumentParser(description = msg)

    parser.add_argument('-s', '--shot', type=int,
                        help='shot number', required=False, nargs='+')
    msg = 'username of the DB where the datafile is located'
    parser.add_argument('-u','--user',help=msg, required=False, default=_USER)
    msg = 'tokamak name of the DB where the datafile is located'
    parser.add_argument('-tok', '--tokamak', help=msg, required=False,
                        default=_TOKAMAK)
    parser.add_argument('-r', '--run', help='run number',
                        required=False, type=int, default=_RUN)
    parser.add_argument('-v', '--version', help='version number',
                        required=False, type=str, default=_VERSION)

    msg = "ids from which to load diagnostics data, can be:\n%s"%repr(_LIDS)
    parser.add_argument('-i', '--ids', type=str, required=True,
                        help=msg, nargs='+', choices=_LIDS)
    parser.add_argument('-q', '--quantity', type=str, required=False,
                        help='Desired quantity from the plasma ids',
                        nargs='+', default=None)
    parser.add_argument('-X', '--X', type=str, required=False,
                        help='Quantity from the plasma ids to use for abscissa',
                        nargs='+', default=None)
    parser.add_argument('-t0', '--t0', type=str, required=False,
                        help='Reference time event setting t = 0', default=_T0)
    parser.add_argument('-t', '--t', type=float, required=False,
                        help='Input time when needed')
    parser.add_argument('-tl', '--tlim', type=_str2tlim,
                        required=False,
                        help='limits of the time interval',
                        nargs='+', default=_TLIM)
    parser.add_argument('-dR_sep', '--dR_sep', type=float, required=False,
                        help='Distance to separatrix from r_ext to plot'
                        + ' 10 magnetic field lines')
    parser.add_argument('-init', '--init', type=float, required=False, nargs=3,
                        help='Manual coordinates of point that a RED magnetic'
                        + ' field line will cross on graphics,'
                        + ' give coordinates as: R [m], Phi [rad], Z [m]')
    parser.add_argument('-ich', '--indch', type=int, required=False,
                        help='indices of channels to be loaded',
                        nargs='+', default=None)
    parser.add_argument('-ichauto', '--indch_auto', type=_str2bool, required=False,
                        help='automatically determine indices of'
                        + ' channels to be loaded', default=_INDCH_AUTO)
    parser.add_argument('-e', '--extra', type=_str2bool, required=False,
                        help='If True loads separatrix and heating power',
                        default=_EXTRA)
    parser.add_argument('-sx', '--sharex', type=_str2bool, required=False,
                        help='Should X axis be shared between diagnostics ids ?',
                        default=_SHAREX, const=True, nargs='?')
    parser.add_argument('-bck', '--background', type=_str2bool, required=False,
                        help='Plot data enveloppe as grey background ?',
                        default=_BCK, const=True, nargs='?')

    args = parser.parse_args()

    # Call wrapper function
    call_tfloadimas(**dict(args._get_kwargs()))


# Add this to make sure it remains executable even without install
if __name__ == '__main__':
    main()
