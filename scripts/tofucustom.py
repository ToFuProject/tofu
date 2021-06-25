#!/usr/bin/env python

# Built-in
import sys
import os
from shutil import copyfile


_HERE = os.path.abspath(os.path.dirname(__file__))


# import parser dict
sys.path.insert(1, _HERE)
from _dparser import _DPARSER
_ = sys.path.pop(1)


###################################################
###################################################
#       function
###################################################


def custom(
    target=None, source=None,
    files=None, directories=None, ddef=None,
):

    # --------------
    # Check inputs
    kwd = locals()
    for k0 in set(ddef.keys()).intersection(kwd.keys()):
        if kwd[k0] is None:
            kwd[k0] = ddef[k0]
    target, source = kwd['target'], kwd['source']
    files, directories = kwd['files'], kwd['directories']

    # Caveat (up to now only relevant for _TARGET)
    if target != ddef['target']:
        msg = ""
        raise Exception(msg)

    # Check files
    if isinstance(files, str):
        files = [files]
    c0 = (not isinstance(files, list)
          or any([ff not in ddef['files'] for ff in files]))
    if c0 is True:
        msg = "All files should be in {}".format(ddef['files'])
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

        msg = (
            "A local copy of default tofu parameters is now in:\n"
            + "\t{}/\n".format(target)
            + "You can edit it to spice up your tofu"
        )
        print(msg)

    except Exception as err:
        msg = (
            str(err) + '\n\n'
            + "A problem occured\n"
            + "tofu-custom tried to create a local directory .tofu/ in "
            + "your home {}\n".format(target)
            + "But it could not, check the error message above to debug\n"
            + "Most frequent cause is a permission issue"
        )
        raise Exception(msg)


###################################################
###################################################
#       bash call (main)
###################################################


def main():
    # Parse input arguments
    # Instanciate parser
    ddef, parser = _DPARSER['custom']()

    # Parse arguments
    args = parser.parse_args()

    # Call function
    custom(ddef=ddef, **dict(args._get_kwargs()))


if __name__ == '__main__':
    main()
