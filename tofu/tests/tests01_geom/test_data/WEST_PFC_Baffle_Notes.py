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

    notes = {'DPhi':{}, 'dPhi':{}}
    # Total length
    notes['DL'] = 405.107

    # Toroidal width
    notes['DPhi']['In'] = 102.951
    notes['DPhi']['Out'] = 120.538

    # Toroidal gap
    notes['dPhi']['In'] = 1.000
    notes['dPhi']['Out'] = 1.000

    # Height
    notes['DZ'] = 21.000

    # Number in toroidal direction
    notes['nbPhi'] = 12*12

    # sample (X,Z,Y)
    notes['PolyRZ'] = [[2.58000, -8.5940e-01],#0, V1
                       [2.58000, -8.3940e-01],  # V1
                       [2.61500, -8.3940e-01],
                       [2.61500, -7.3040e-01],
                       [2.47200, -7.3040e-01],
                       [2.46200, -7.1340e-01],  # V1
                       [2.46200, -6.9340e-01],
                       [2.38950, -6.9340e-01],
                       [2.38670, -6.9300e-01],  # V1
                       [2.38420, -6.9170e-01],  # V1
                       [2.38220, -6.8970e-01],#10, V1
                       [2.38090, -6.8720e-01],
                       [2.38050, -6.8440e-01],  # V1
                       [2.38090, -6.8160e-01],
                       [2.38220, -6.7910e-01],  # V1
                       [2.38420, -6.7710e-01],  # V1
                       [2.38670, -6.7580e-01],  # V1
                       [2.38950, -6.7540e-01],
                       [2.46525, -6.7540e-01],  # V1
                       [2.61767, -6.7540e-01],  # V1
                       [2.72657, -6.7540e-01],#20, V1
                       [2.78560, -6.7540e-01],
                       [2.78560, -6.9340e-01],
                       [2.75000, -6.9340e-01],  # V1
                       [2.75000, -7.0840e-01],  # V1
                       [2.69800, -7.0840e-01],  # V1
                       [2.69500, -7.3040e-01],
                       [2.63500, -7.3040e-01],
                       [2.63500, -8.3940e-01],
                       [2.67000, -8.3940e-01],  # V1
                       [2.67000, -8.5940e-01]]#30, V1

    notes['PolyRZ'] = np.asarray(notes['PolyRZ'],dtype=float)*1.e3
    notes['ind_V1'] = np.array([0,1,5,8,9,10,12,14,15,16,
                                18,19,20,23,24,25,29,30],dtype=int)

    for kk in notes.keys():
        if type(notes[kk]) is dict:
            notes[kk]['In'] = notes[kk]['In']*1.e-3
            notes[kk]['Out'] = notes[kk]['Out']*1.e-3
        elif not ('nb' in kk or 'ind' in kk):
            notes[kk] = notes[kk]*1.e-3
    return notes


def make_Poly(save=_save, path=_here):

    notes = get_notes()

    Poly = notes['PolyRZ']
    ind0 = np.ones((Poly.shape[0],),dtype=bool)
    ind0[notes['ind_V1']] = False
    Poly0 = Poly[ind0,:]

    if save:
        cstr = '%s_%s_%s'%(_Exp,_Cls,_name)
        pathfilext = os.path.join(path, cstr+'_V0.txt')
        np.savetxt(pathfilext, Poly0)
        pathfilext = os.path.join(path, cstr+'_V1.txt')
        np.savetxt(pathfilext, Poly)
    return Poly0.T, Poly.T, notes



if __name__=='__main__':

    # Parse input arguments
    msg = 'Launch creation of polygons txt from bash'
    parser = argparse.ArgumentParser(description = msg)

    parser.add_argument('-save', type=bool, help='save ?', default=_save)
    parser.add_argument('-path', type=str, help='saving path ?', default=_here)

    args = parser.parse_args()

    # Call wrapper function
    make_Poly(save=args.save, path=args.path)
