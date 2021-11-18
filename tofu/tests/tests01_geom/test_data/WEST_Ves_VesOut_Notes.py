#!/usr/bin/env python

import os
import argparse

import numpy as np

_save = True
_here = os.path.abspath(os.path.dirname(__file__))
_Exp, _Cls, _name = os.path.split(__file__)[1].split('_')[:3]
assert not any([any([ss in s for ss in ['Notes','.']])
               for s in [_Exp, _Cls, _name]])

def get_notes():

    # Notes from creoView (-X,Z,Y)
    notes = {'C': np.r_[2.465, 0.],
             'r_in': 3.162/2., # r_out for later use (thick)
             'r_out': 3.292/2.}
    return notes

def make_Poly(save=_save, path=_here):

    notes = get_notes()

    C = notes['C']
    nP = 100
    theta = np.linspace(0.,2*np.pi, nP, endpoint=False)
    P = np.array([C[0]+notes['r_out']*np.cos(theta),
                  C[1]+notes['r_out']*np.sin(theta)])

    if save:
        cstr = '%s_%s_%s'%(_Exp,_Cls,_name)
        pathfilext = os.path.join(path, cstr+'_V0.txt')
        np.savetxt(pathfilext, P)
    return P, notes



if __name__=='__main__':

    # Parse input arguments
    msg = 'Launch creation of polygons txt from bash'
    parser = argparse.ArgumentParser(description = msg)

    parser.add_argument('-save', type=bool, help='save ?', default=_save)
    parser.add_argument('-path', type=str, help='saving path ?', default=_here)

    args = parser.parse_args()

    # Call wrapper function
    make_Poly(save=args.save, path=args.path)
