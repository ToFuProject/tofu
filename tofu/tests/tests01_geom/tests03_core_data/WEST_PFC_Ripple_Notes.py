#!/usr/bin/env python

# Built-in
import os
import argparse

# Common
import numpy as np


_save = True
_here = os.path.abspath(os.path.dirname(__file__))
_Exp, _Cls, _name = os.path.split(__file__)[1].split('_')[:3]
assert not any([any([ss in s for ss in ['Notes','.']])
               for s in [_Exp, _Cls, _name]])

def get_notes():
    """ By convention : D is a length of the element, d is a gap  """

    notes = {'DPhi':{}}
    # Toroidal width (mm, inner outer)
    notes['DPhi']['In'] = 44.000
    notes['DPhi']['Out'] = 44.000

    notes['nbtor'] = 6

    notes['DL'] = 284.000
    notes['DZ'] = 30.023

    # sampleXZY
    notes['sampleXZY'] = [[-1218.000, 836.000, 2109.638],
                          [-1218.000, 805.994, 2109.638],
                          [-1357.000, 805.994, 2350.393],
                          [-1358.161, 806.468, 2352.403],
                          [-1359.121, 807.751, 2354.067],
                          [-1359.763, 809.674, 2355.180],
                          [-1360.000, 811.994, 2355.589],
                          [-1360.000, 827.000, 2355.589]]
    notes['sampleXZY'] = np.array(notes['sampleXZY'])

    for kk in notes.keys():
        if type(notes[kk]) is dict:
            notes[kk]['In'] = notes[kk]['In']*1.e-3
            notes[kk]['Out'] = notes[kk]['Out']*1.e-3
        elif not 'nb' in kk:
            notes[kk] = notes[kk]*1.e-3
    return notes



def make_Poly(save=_save, path=_here):
    notes = get_notes()

    Poly = np.array([np.hypot(notes['sampleXZY'][:,0],notes['sampleXZY'][:,2]),
                     notes['sampleXZY'][:,1]])
    Polymin = Poly[:,[0,1,2,6,7]]

    if save:
        cstr = '%s_%s_%s'%(_Exp,_Cls,_name)
        pathfilext = os.path.join(path, cstr+'_V0.txt')
        np.savetxt(pathfilext, Polymin.T)
        pathfilext = os.path.join(path, cstr+'_V1.txt')
        np.savetxt(pathfilext, Poly.T)
    return Poly, notes


if __name__=='__main__':

    # Parse input arguments
    msg = 'Launch creation of polygons txt from bash'
    parser = argparse.ArgumentParser(description = msg)

    parser.add_argument('-save', type=bool, help='save ?', default=_save)
    parser.add_argument('-path', type=str, help='saving path ?', default=_here)

    args = parser.parse_args()

    # Call wrapper function
    make_Poly(save=args.save, path=args.path)
