#!/usr/bin/env python

# Built-in
import os
import getpass
from shutil import copyfile
import argparse


###################################################
###################################################
#       default values
###################################################


_SOURCE = os.path.join(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..')), 'tofu')
_USER = getpass.getuser()
_USER_HOME = os.path.expanduser('~')
_TARGET = os.path.join(_USER_HOME, '.tofu')
_LF = ['_imas2tofu_def.py', '_entrypoints_def.py']
_LD = ['openadas2tofu']


###################################################
###################################################
#       function
###################################################


def custom(target=_TARGET, source=_SOURCE,
           files=_LF, directories=_LD):

    # Caveat (up to now only relevant for _TARGET)
    if target != _TARGET:
        msg = ""
        raise Exception(msg)

    # Check files
    if isinstance(files, str):
        files = [files]
    if not isinstance(files, list) or any([ff not in _LF for ff in files]):
        msg = "All files should be in {}".format(_LF)
        raise Exception(msg)

    # Try creating directory and copying modules
    try:
        # Create .tofu/ if non-existent
        if not os.path.isdir(target):
            os.mkdir(target)

        # Create directories
        for dd in directories:
            if not os.path.isdir(os.path.join(target, dd)):
                os.mkdir(os.path.join(target, dd))

        # Copy files
        for ff in files:
            mod, f0 = ff.split('_')[1:]
            copyfile(os.path.join(source, mod, '_'+f0),
                     os.path.join(target, ff))

        msg = ("A local copy of default tofu parameters is now in:\n"
               + "\t{}/\n".format(target)
               + "You can edit it to spice up your tofu")
        print(msg)

    except Exception as err:
        msg = (str(err) + '\n\n'
               + "A problem occured\n"
               + "tofu-custom tried to create a local directory .tofu/ in "
               + "your home {}\n".format(target)
               + "But it could not, check the error message above to debug\n"
               + "Most frequent cause is a permission issue")
        raise Exception(msg)


###################################################
###################################################
#       bash call (main)
###################################################

def main():
    # Parse input arguments
    msg = """ Create a local copy of tofu default parameters

    This creates a local copy, in your home, of tofu default parameters
    A directory .tofu is created in your home directory
    In this directory, modules containing default parameters are copied
    You can then customize them without impacting other users

    """

    # Instanciate parser
    parser = argparse.ArgumentParser(description=msg)

    # Define input arguments
    parser.add_argument('-s', '--source',
                        type=str,
                        help='tofu source directory',
                        required=False,
                        default=_SOURCE)
    parser.add_argument('-t', '--target',
                        type=str,
                        help=('directory where .tofu/ should be created'
                              + ' (default: {})'.format(_TARGET)),
                        required=False,
                        default=_TARGET)
    parser.add_argument('-f', '--files',
                        type=str,
                        help='list of files to be copied',
                        required=False,
                        nargs='+',
                        default=_LF,
                        choices=_LF)

    # Parse arguments
    args = parser.parse_args()

    # Call function
    custom(**dict(args._get_kwargs()))


if __name__ == '__main__':
    main()
