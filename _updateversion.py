#!/usr/bin/env/python
# coding=utf-8

import os
import subprocess

_HERE = os.path.abspath(os.path.dirname(__file__))


def updateversion(path=_HERE):
    # Fetch version from git tags, and write to version.py
    # Also, when git is not available (PyPi package), use stored version.py
    version_py = os.path.join(path, 'tofu', 'version.py')
    try:
        version_git = subprocess.check_output(["git",
                                               "describe"]).rstrip().decode()
    except subprocess.CalledProcessError:
        with open(version_py, 'r') as fh:
            version_git = fh.read().strip().split("=")[-1].replace("'", '')
    version_git = version_git.lower().replace('v', '').replace(' ', '')

    version_msg = "# Do not edit, pipeline versioning governed by git tags!"
    with open(version_py, "w") as fh:
        msg = "{0}__version__ = '{1}'{0}".format(os.linesep, version_git)
        fh.write(version_msg + msg)
    return version_git
