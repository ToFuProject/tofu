import sys
import os
import getpass
import argparse


# tofu
# test if in a tofu git repo
_HERE = os.path.abspath(os.path.dirname(__file__))
_TOFUPATH = os.path.dirname(_HERE)


def get_mods():
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
    return tf, MultiIDSLoader, _defscripts


# #############################################################################
#       utility functions
# #############################################################################


def _str2bool(v):
    if isinstance(v, bool):
        return v
    elif v.lower() in ['yes', 'true', 'y', 't', '1']:
        return True
    elif v.lower() in ['no', 'false', 'n', 'f', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected!')


def _str2boolstr(v):
    if isinstance(v, bool):
        return v
    elif isinstance(v, str):
        if v.lower() in ['yes', 'true', 'y', 't', '1']:
            return True
        elif v.lower() in ['no', 'false', 'n', 'f', '0']:
            return False
        elif v.lower() == 'none':
            return None
        else:
            return v
    else:
        raise argparse.ArgumentTypeError('Boolean, None or str expected!')


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


# #############################################################################
#       Parser for tofu version
# #############################################################################


def parser_version():
    msg = """ Get tofu version from bash optionally set an enviroment variable

    If run from a git repo containing tofu, simply returns git describe
    Otherwise reads the tofu version stored in tofu/version.py

    """
    ddef = {
        'path': os.path.join(_TOFUPATH, 'tofu'),
        'envvar': False,
        'verb': True,
        'warn': True,
        'force': False,
        'name': 'TOFU_VERSION',
    }

    # Instanciate parser
    parser = argparse.ArgumentParser(description=msg)

    # Define input arguments
    parser.add_argument('-p', '--path',
                        type=str,
                        help='tofu source directory where version.py is found',
                        required=False, default=ddef['path'])
    parser.add_argument('-v', '--verb',
                        type=_str2bool,
                        help='flag indicating whether to print the version',
                        required=False, default=ddef['verb'])
    parser.add_argument('-ev', '--envvar',
                        type=_str2boolstr,
                        help='name of the environment variable to set, if any',
                        required=False, default=ddef['envvar'])
    parser.add_argument('-w', '--warn',
                        type=_str2bool,
                        help=('flag indicatin whether to print a warning when'
                              + 'the desired environment variable (envvar)'
                              + 'already exists'),
                        required=False, default=ddef['warn'])
    parser.add_argument('-f', '--force',
                        type=_str2bool,
                        help=('flag indicating whether to force the update of '
                              + 'the desired environment variable (envvar)'
                              + ' even if it already exists'),
                        required=False, default=ddef['force'])

    return ddef, parser


# #############################################################################
#       Parser for tofu custom
# #############################################################################


def parser_custom():
    msg = """ Create a local copy of tofu default parameters

    This creates a local copy, in your home, of tofu default parameters
    A directory .tofu is created in your home directory
    In this directory, modules containing default parameters are copied
    You can then customize them without impacting other users

    """
    _USER = getpass.getuser()
    _USER_HOME = os.path.expanduser('~')

    ddef = {
        'target': os.path.join(_USER_HOME, '.tofu'),
        'source': os.path.join(_TOFUPATH, 'tofu'),
        'files': [
            '_imas2tofu_def.py',
            '_entrypoints_def.py',
        ],
        'directories': [
            'openadas2tofu',
            'nist2tofu',
            os.path.join('nist2tofu', 'ASD'),
        ],
    }

    # Instanciate parser
    parser = argparse.ArgumentParser(description=msg)

    # Define input arguments
    parser.add_argument('-s', '--source',
                        type=str,
                        help='tofu source directory',
                        required=False,
                        default=ddef['source'])
    parser.add_argument('-t', '--target',
                        type=str,
                        help=('directory where .tofu/ should be created'
                              + ' (default: {})'.format(ddef['target'])),
                        required=False,
                        default=ddef['target'])
    parser.add_argument('-f', '--files',
                        type=str,
                        help='list of files to be copied',
                        required=False,
                        nargs='+',
                        default=ddef['files'],
                        choices=ddef['files'])
    return ddef, parser


# #############################################################################
#       Parser for tofu plot
# #############################################################################


def parser_plot():

    tf, MultiIDSLoader, _defscripts = get_mods()

    _LIDS_CONFIG = MultiIDSLoader._lidsconfig
    _LIDS_DIAG = MultiIDSLoader._lidsdiag
    _LIDS_PLASMA = tf.imas2tofu.MultiIDSLoader._lidsplasma
    _LIDS = _LIDS_CONFIG + _LIDS_DIAG + _LIDS_PLASMA + tf.utils._LIDS_CUSTOM

    msg = """Fast interactive visualization tool for diagnostics data in
    imas

    This is merely a wrapper around the function tofu.load_from_imas()
    It loads (from imas) and displays diagnostics data from the following
    ids:
        {}
    """.format(_LIDS)

    ddef = {
        # User-customizable
        'run': _defscripts._TFPLOT_RUN,
        'user': _defscripts._TFPLOT_USER,
        'database': _defscripts._TFPLOT_DATABASE,
        'version': _defscripts._TFPLOT_VERSION,
        't0': _defscripts._TFPLOT_T0,
        'tlim': None,
        'sharex': _defscripts._TFPLOT_SHAREX,
        'bck': _defscripts._TFPLOT_BCK,
        'extra': _defscripts._TFPLOT_EXTRA,
        'indch_auto': _defscripts._TFPLOT_INDCH_AUTO,

        'config': _defscripts._TFPLOT_CONFIG,
        'tosoledge3x': _defscripts._TFPLOT_TOSOLEDGE3X,

        'mag_sep_nbpts': _defscripts._TFPLOT_MAG_SEP_NBPTS,
        'mag_sep_dR': _defscripts._TFPLOT_MAG_SEP_DR,
        'mag_init_pts': _defscripts._TFPLOT_MAG_INIT_PTS,

        # Non user-customizable
        'lids_plasma': _LIDS_PLASMA,
        'lids_diag': _LIDS_DIAG,
        'lids': _LIDS,
    }

    parser = argparse.ArgumentParser(description=msg)

    msg = 'config name to be loaded'
    parser.add_argument('-c', '--config', help=msg,
                        required=False, type=str,
                        default=ddef['config'])
    msg = 'path in which to save the tofu config in SOLEDGE3X format'
    parser.add_argument('-tse3x', '--tosoledge3x', help=msg,
                        required=False, type=str,
                        default=ddef['tosoledge3x'])
    parser.add_argument('-s', '--shot', type=int,
                        help='shot number', required=False, nargs='+')
    msg = 'username of the DB where the datafile is located'
    parser.add_argument('-u', '--user', help=msg, required=False,
                        default=ddef['user'])
    msg = 'database name where the datafile is located'
    parser.add_argument('-db', '--database', help=msg, required=False,
                        default=ddef['database'])
    parser.add_argument('-r', '--run', help='run number',
                        required=False, type=int,
                        default=ddef['run'])
    parser.add_argument('-v', '--version', help='version number',
                        required=False, type=str,
                        default=ddef['version'])

    msg = ("ids from which to load diagnostics data,"
           + " can be:\n{}".format(ddef['lids']))
    parser.add_argument('-i', '--ids', type=str, required=True,
                        help=msg, nargs='+', choices=ddef['lids'])
    parser.add_argument('-q', '--quantity', type=str, required=False,
                        help='Desired quantity from the plasma ids',
                        nargs='+', default=None)
    parser.add_argument('-X', '--X', type=str, required=False,
                        help='Quantity from the plasma ids for abscissa',
                        nargs='+', default=None)
    parser.add_argument('-t0', '--t0', type=_str2boolstr, required=False,
                        help='Reference time event setting t = 0',
                        default=ddef['t0'])
    parser.add_argument('-t', '--t', type=float, required=False,
                        help='Input time when needed')
    parser.add_argument('-tl', '--tlim', type=_str2tlim,
                        required=False,
                        help='limits of the time interval',
                        nargs='+', default=ddef['tlim'])
    parser.add_argument('-ich', '--indch', type=int, required=False,
                        help='indices of channels to be loaded',
                        nargs='+', default=None)
    parser.add_argument('-ichauto', '--indch_auto', type=_str2bool,
                        required=False,
                        help='automatically determine indices of'
                        + ' channels to be loaded', default=ddef['indch_auto'])
    parser.add_argument('-e', '--extra', type=_str2bool, required=False,
                        help='If True loads separatrix and heating power',
                        default=ddef['extra'])
    parser.add_argument('-sx', '--sharex', type=_str2bool, required=False,
                        help='Should X axis be shared between diag ids ?',
                        default=ddef['sharex'], const=True, nargs='?')
    parser.add_argument('-bck', '--background', type=_str2bool, required=False,
                        help='Plot data enveloppe as grey background ?',
                        default=ddef['bck'], const=True, nargs='?')

    parser.add_argument('-mag_sep_dR', '--mag_sep_dR', type=float,
                        required=False,
                        default=ddef['mag_sep_dR'],
                        help='Distance to separatrix from r_ext to plot'
                        + ' magnetic field lines')
    parser.add_argument('-mag_sep_nbpts', '--mag_sep_nbpts', type=int,
                        required=False,
                        default=ddef['mag_sep_nbpts'],
                        help=('Number of mag. field lines to plot '
                              + 'from separatrix'))
    parser.add_argument('-mag_init_pts', '--mag_init_pts',
                        type=float, required=False, nargs=3,
                        default=ddef['mag_init_pts'],
                        help='Manual coordinates of point that a RED magnetic'
                        + ' field line will cross on graphics,'
                        + ' give coordinates as: R [m], Phi [rad], Z [m]')
    return ddef, parser


# #############################################################################
#       Parser for tofu calc
# #############################################################################


def parser_calc():

    tf, MultiIDSLoader, _defscripts = get_mods()

    _LIDS_DIAG = MultiIDSLoader._lidsdiag
    _LIDS_PLASMA = tf.imas2tofu.MultiIDSLoader._lidsplasma
    _LIDS = _LIDS_DIAG + _LIDS_PLASMA + tf.utils._LIDS_CUSTOM

    # Parse input arguments
    msg = """Fast interactive visualization tool for diagnostics data in
    imas

    This is merely a wrapper around the function tofu.calc_from_imas()
    It calculates and dsplays synthetic signal (from imas) from the following
    ids:
        {}
    """.format(_LIDS)

    ddef = {
        # User-customizable
        'run': _defscripts._TFCALC_RUN,
        'user': _defscripts._TFCALC_USER,
        'database': _defscripts._TFCALC_DATABASE,
        'version': _defscripts._TFCALC_VERSION,
        't0': _defscripts._TFCALC_T0,
        'tlim': None,
        'sharex': _defscripts._TFCALC_SHAREX,
        'bck': _defscripts._TFCALC_BCK,
        'extra': _defscripts._TFCALC_EXTRA,
        'indch_auto': _defscripts._TFCALC_INDCH_AUTO,
        'coefs': None,

        # Non user-customizable
        'lids_plasma': _LIDS_PLASMA,
        'lids_diag': _LIDS_DIAG,
        'lids': _LIDS,
    }

    parser = argparse.ArgumentParser(description=msg)

    # Main idd parameters
    parser.add_argument('-s', '--shot', type=int,
                        help='shot number', required=True)
    msg = 'username of the DB where the datafile is located'
    parser.add_argument('-u', '--user',
                        help=msg, required=False, default=ddef['user'])
    msg = 'database name where the datafile is located'
    parser.add_argument('-db', '--database', help=msg, required=False,
                        default=ddef['database'])
    parser.add_argument('-r', '--run', help='run number',
                        required=False, type=int, default=ddef['run'])
    parser.add_argument('-v', '--version', help='version number',
                        required=False, type=str, default=ddef['version'])

    # Equilibrium idd parameters
    parser.add_argument('-s_eq', '--shot_eq', type=int,
                        help='shot number for equilibrium, defaults to -s',
                        required=False, default=None)
    msg = 'username for the equilibrium, defaults to -u'
    parser.add_argument('-u_eq', '--user_eq',
                        help=msg, required=False, default=None)
    msg = 'database name for the equilibrium, defaults to -tok'
    parser.add_argument('-db_eq', '--database_eq',
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
    msg = 'database name for the profiles, defaults to -tok'
    parser.add_argument('-db_prof', '--database_prof',
                        help=msg, required=False, default=None)
    parser.add_argument('-r_prof', '--run_prof',
                        help='run number for the profiles, defaults to -r',
                        required=False, type=int, default=None)

    msg = ("ids from which to load diagnostics data,"
           + " can be:\n{}".format(ddef['lids']))
    parser.add_argument('-i', '--ids', type=str, required=True,
                        help=msg, nargs='+', choices=ddef['lids'])
    parser.add_argument('-B', '--Brightness', type=bool, required=False,
                        help='Whether to express result as brightness',
                        default=None)
    parser.add_argument('-res', '--res', type=float, required=False,
                        help='Space resolution for the LOS-discretization',
                        default=None)
    parser.add_argument('-t0', '--t0', type=_str2boolstr, required=False,
                        help='Reference time event setting t = 0',
                        default=ddef['t0'])
    parser.add_argument('-tl', '--tlim', type=_str2tlim,
                        required=False,
                        help='limits of the time interval',
                        nargs='+', default=ddef['tlim'])
    parser.add_argument('-c', '--coefs', type=float, required=False,
                        help='Corrective coefficient, if any',
                        default=ddef['coefs'])
    parser.add_argument('-ich', '--indch', type=int, required=False,
                        help='indices of channels to be loaded',
                        nargs='+', default=None)
    parser.add_argument('-ichauto', '--indch_auto', type=bool, required=False,
                        help=('automatically determine indices '
                              + 'of channels to be loaded'),
                        default=ddef['indch_auto'])
    parser.add_argument('-e', '--extra', type=_str2bool, required=False,
                        help='If True loads separatrix and heating power',
                        default=ddef['extra'])
    parser.add_argument('-sx', '--sharex', type=_str2bool, required=False,
                        help='Should X axis be shared between diag ids?',
                        default=ddef['sharex'], const=True, nargs='?')
    parser.add_argument('-if', '--input_file', type=str, required=False,
                        help='mat file from which to load core_profiles',
                        default=None)
    parser.add_argument('-of', '--output_file', type=str, required=False,
                        help='mat file into which to save synthetic signal',
                        default=None)
    parser.add_argument('-bck', '--background', type=_str2bool, required=False,
                        help='Plot data enveloppe as grey background ?',
                        default=ddef['bck'], const=True, nargs='?')
    return ddef, parser


# #############################################################################
#       Parser dict
# #############################################################################


_DPARSER = {
    'version': parser_version,
    'custom': parser_custom,
    'plot': parser_plot,
    'calc': parser_calc,
}
