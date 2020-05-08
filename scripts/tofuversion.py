#!/usr/bin/env python

# Built-in
import os
import argparse
import warnings


###################################################
###################################################
#       default values
###################################################


# _TOFUPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_TOFUPATH = os.path.join(os.path.join(os.path.dirname(__file__), '..'), 'tofu')
_DBOOL = {'verb': True, 'warn': True, 'force': False}
_ENVVAR = False
_NAME = 'TOFU_VERSION'


###################################################
###################################################
#       function
###################################################


def get_version(verb=None, envvar=None,
                path=_TOFUPATH, warn=None, force=None):
    """ Print tofu version and / or store in environment variable """

    # --------------
    # Check inputs

    # verb, warn, force
    dbool = {'verb': verb, 'warn': warn, 'force': force}
    for k0, v0 in dbool.items():
        if v0 is None:
            dbool[k0] = _DBOOL[k0]
        if not isinstance(dbool[k0], bool):
            msg = ("Arg {} must be a bool\n".format(k0)
                   + "\t- provided: {}".format(dbool[k0]))
            raise Exception(msg)

    # envvar
    if envvar is None:
        envvar = _ENVVAR
    if isinstance(envvar, bool):
        if envvar is True:
            envvar = _NAME
    elif isinstance(envvar, str):
        envvar = envvar.upper()
    else:
        msg = ("Arg envvar must be either:\n"
               + "\t- None:   set to default ({})\n".format(_ENVVAR)
               + "\t- False:  no setting of environment variable\n"
               + "\t- True:   set default env. variable ({})\n".format(_NAME)
               + "\t- str:    set provided env. variable\n\n"
               + " you provided: {}".format(envvar))
        raise Exception(msg)

    if envvar is not False and envvar in os.environ.keys():
        if dbool['warn'] is True:
            msg = ("Chosen environment variable name already exists:\n"
                   + "\t- {}: {}\n".format(envvar, os.environ[envvar])
                   + "  => Are you sure you want to change it?\n"
                   + "       use force=True to overwrite {}\n".format(envvar)
                   + "       use warn=False to disable this warning")
            warnings.warn(msg)

    # --------------
    # Fetch version from git tags, and write to version.py
    # Also, when git is not available (PyPi package), use stored version.py
    pfe = os.path.join(path, 'version.py')
    if not os.path.isfile(pfe):
        msg = ("It seems your current tofu install has no version.py:\n"
               "\t- looked for: {}".format(pfe))
        raise Exception(msg)

    # --------------
    # Read file
    with open(pfe, 'r') as fh:
        version = fh.read().strip().split("=")[-1].replace("'", '')
    version = version.lower().replace('v', '').replace(' ', '')

    # --------------
    # Outputs
    if dbool['verb'] is True:
        print(version)
    if envvar is not False:
        c0 = ((envvar in os.environ.keys() and dbool['force'] is True)
              or envvar not in os.environ.keys())
        if c0 is True:
            os.environ[envvar] = version


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
        raise argparse.ArgumentTypeError('Boolean or str expected!')


def main():
    # Parse input arguments
    msg = """ Get tofu version from bash optionally set an enviroment variable

    If run from a git repo containing tofu, simply returns git describe
    Otherwise reads the tofu version stored in tofu/version.py

    """

    # Instanciate parser
    parser = argparse.ArgumentParser(description=msg)

    # Define input arguments
    parser.add_argument('-p', '--path',
                        type=str,
                        help='tofu source directory where version.py is found',
                        required=False,
                        default=_TOFUPATH)
    parser.add_argument('-v', '--verb',
                        type=_str2bool,
                        help='flag indicating whether to print the version',
                        required=False,
                        default=_DBOOL['verb'])
    parser.add_argument('-ev', '--envvar',
                        type=_str2boolstr,
                        help='name of the environment variable to set, if any',
                        required=False,
                        default=_ENVVAR)
    parser.add_argument('-w', '--warn',
                        type=_str2bool,
                        help=('flag indicating whether to print a warning when'
                              + ' the desired environment variable (envvar) '
                              + 'already exists'),
                        required=False,
                        default=_DBOOL['warn'])
    parser.add_argument('-f', '--force',
                        type=_str2bool,
                        help=('flag indicating whether to force the update of '
                              + ' the desired environment variable (envvar) '
                              + 'even if it already exists'),
                        required=False,
                        default=_DBOOL['force'])

    # Parse arguments
    args = parser.parse_args()

    # Call function
    get_version(**dict(args._get_kwargs()))


if __name__ == '__main__':
    main()
