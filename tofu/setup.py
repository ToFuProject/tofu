#!/usr/bin/env python


import sys

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('tofu', parent_package, top_path)
    config.add_subpackage('distutils')
    #config.add_subpackage('testing')
    #config.add_subpackage('f2py')

    config.add_subpackage('core')
    config.add_subpackage('geom')
    config.add_subpackage('mesh')
    config.add_subpackage('matcomp')
    config.add_subpackage('data')
    config.add_subpackage('inv')

    #config.add_subpackage('doc')
    #if sys.version_info[0] < 3:            # Check python version, add version-specific packages if necessary (inspired from scipy/__init__.py)
    #    config.add_subpackage('weave')
    #config.add_data_dir('doc')
    #config.add_data_dir('tests')
    config.make_config_py() # installs __config__.py
    return config



""" # From scipy

    config.add_subpackage('_build_utils')
"""



if __name__ == '__main__':
    print('This is the wrong setup.py file to run')

