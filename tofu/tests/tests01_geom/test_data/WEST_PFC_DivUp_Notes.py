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
    notes['DL'] = 449.000

    # Toroidal width
    notes['DPhi']['In'] = 26.151
    notes['DPhi']['Out'] = 31.887

    # Toroidal gap
    notes['dPhi']['In'] = 0.681
    notes['dPhi']['Out'] = 0.680

    # Height
    notes['DZ'] = 41.562

    # Number in toroidal direction
    notes['nbPhi'] = 12*38

    # sample (X,Z,Y)
    notes['sampleXZY'] = [[-1917.967, 640.060, -293.245],
                          [-1932.817, 601.230, -299.739],
                          [-2167.242, 697.010, -335.545],
                          [-2168.645, 697.479, -335.689],
                          [-2198.613, 709.727, -340.266],
                          [-2201.495, 711.408, -340.635],
                          [-2344.246, 769.769, -362.438],
                          [-2329.510, 808.258, -356.103]]
    notes['sampleXZY'] = np.asarray(notes['sampleXZY'],dtype=float)

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
    Poly0 = Poly[:,[0,1,-2,-1]]
    PM = np.mean(Poly[:,[2,5]],axis=1)[:,np.newaxis]
    Poly1 = np.concatenate((Poly0[:,:2], PM,Poly0[:,2:]),axis=1)

    if save:
        cstr = '%s_%s_%s'%(_Exp,_Cls,_name)
        pathfilext = os.path.join(path, cstr+'_V0.txt')
        np.savetxt(pathfilext, Poly0.T)
        pathfilext = os.path.join(path, cstr+'_V1.txt')
        np.savetxt(pathfilext, Poly1.T)
        pathfilext = os.path.join(path, cstr+'_V2.txt')
        np.savetxt(pathfilext, Poly.T)
    return Poly0, Poly1, Poly, notes



if __name__=='__main__':

    # Parse input arguments
    msg = 'Launch creation of polygons txt from bash'
    parser = argparse.ArgumentParser(description = msg)

    parser.add_argument('-save', type=bool, help='save ?', default=_save)
    parser.add_argument('-path', type=str, help='saving path ?', default=_here)

    args = parser.parse_args()

    # Call wrapper function
    make_Poly(save=args.save, path=args.path)
