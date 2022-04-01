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

    notes = {'DPhi':{}, 'dPhi':{}}
    # Toroidal width (mm, inner outer)
    notes['DPhi']['In'] = 25.508
    notes['DPhi']['Out'] = 32.908

    # Inter tiles distance (mm, uniform)
    notes['dl'] = 1.500

    # Poloidal/Radial total length (mm)
    notes['DL'] = 573.995

    # Number of tiles radially
    notes['nb'] = 2
    notes['nbPhi'] = 38*12

    # Radial length of a tile (mm)
    notes['Dl'] = np.array([316.495,256.000])

    # Vertical height of tiles (mm, uniform)
    notes['DZ'] = 25.538

    # Toroidal space between needles (mm, inner outer)
    notes['dPhi']['In'] = 0.666
    notes['dPhi']['Out'] = 0.666

    # (X,Z,Y) polygon of one needle (mm) !!!!!! (X,Z,Y)
    # 1 mm should be added towards Z>0 in the direction normal to the divertor's upper surface
    notes['sampleXZY'] = [[-440.221, -606.218, 1854.847],
                          [-440.163, -581.217, 1854.860],
                          [-440.748, -579.362, 1857.546],
                          [-506.714, -694.150, 2133.992],
                          [-506.951, -694.466, 2134.837],
                          [-510.087, -699.924, 2147.977],
                          [-510.336, -702.527, 2149.054],
                          [-508.309, -724.780, 2140.295],
                          [-508.995, -721.633, 2143.101],
                          [-510.684, -703.089, 2150.401],
                          [-511.270, -701.984, 2152.937],
                          [-514.021, -706.772, 2164.465],
                          [-514.742, -708.491, 2167.329],
                          [-565.491, -796.886, 2380.007],
                          [-565.707, -799.489, 2381.092],
                          [-563.726, -821.241, 2372.530]]
    notes['sampleXZY'] = np.array(notes['sampleXZY'])

    for kk in notes.keys():
        if type(notes[kk]) is dict:
            notes[kk]['In'] = notes[kk]['In']*1.e-3
            notes[kk]['Out'] = notes[kk]['Out']*1.e-3
        elif not 'nb' in kk:
            notes[kk] = notes[kk]*1.e-3
    return notes


def _get_inter(D0,u0,D1,u1):
    k = -np.cross(D0-D1,u1)/np.cross(u0,u1)
    return D0 + k*u0

def make_Poly(save=_save, path=_here):
    notes = get_notes()

    R = np.hypot(notes['sampleXZY'][:,0],notes['sampleXZY'][:,2])
    Z = notes['sampleXZY'][:,1]

    d = np.sqrt(np.diff(R)**2+np.diff(Z)**2)
    indup = np.argmax(d)
    e1 = np.array([R[indup+1]-R[indup], Z[indup+1]-Z[indup]])
    e1 = e1/np.linalg.norm(e1)
    e2 = np.r_[-e1[1], e1[0]]

    P0 = (np.array([R[0],Z[0]])-0.01*e2)[:,np.newaxis]
    PEnd = (np.array([R[-1],Z[-1]])-0.01*e2)[:,np.newaxis]
    Poly = np.array([R[1:-1],Z[1:-1]])
    Poly = np.concatenate((P0,Poly,PEnd),axis=1)
    Poly0 = Poly[:,[0,2,-3,-1]]

    # Making Poly1
    D0 = Poly0[:,1]
    u0 = Poly0[:,-2]-D0
    k0 = -np.sum((D0-Poly[:,5])*u0)/np.linalg.norm(u0)**2
    k1 = -np.sum((D0-Poly[:,10])*u0)/np.linalg.norm(u0)**2
    P0 = (D0 + k0*u0)[:,np.newaxis]
    P1 = (D0 + k1*u0)[:,np.newaxis]
    D0, D1 = Poly[:,5], Poly[:,10]
    u0, u1 = Poly[:,6]-D0, Poly[:,9]-D1
    PI = _get_inter(D0,u0,D1,u1)[:,np.newaxis]
    Poly1 = np.concatenate((Poly0[:,:2], P0,PI,P1, Poly0[:,2:]),axis=1)

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
