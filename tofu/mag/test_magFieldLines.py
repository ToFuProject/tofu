# -*- coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
# Also if needed: retab
'''
    TEST magFieldLines
'''
from __future__ import (unicode_literals, absolute_import,  \
                        print_function, division)
import doctest
import os
import sys

if __name__ == '__main__':
    #print('path 1 =', sys.path)
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    #print('path 2 =', sys.path)
    #import mag
    from mag import magFieldLines
    path = sys.path.pop(0)
    doctest.testmod(magFieldLines, verbose=True)
