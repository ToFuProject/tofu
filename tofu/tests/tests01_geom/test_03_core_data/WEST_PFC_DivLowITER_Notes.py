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
    notes['DPhi']['In'] = 26.370
    notes['DPhi']['Out'] = 31.929

    # Inter tiles distance (mm, uniform)
    notes['dl'] = 0.500

    # Poloidal/Radial total length (mm)
    notes['DL'] = 437.000

    # Number of tiles radially
    notes['nb'] = 35
    notes['nbPhi'] = 19*2*12

    # Radial length of a tile (mm, uniform)
    notes['Dl'] = 12.000

    # Vertical height of tiles (mm, uniform)
    notes['DZ'] = 26.000

    # Toroidal space between needles (mm, inner outer)
    notes['dPhi']['In'] = 0.588
    notes['dPhi']['Out'] = 0.612

    # (X,Z,Y) polygon of one needle (mm) !!!!!! (X,Z,Y)
    # 1 mm should be added towards Z>0 in the direction normal to the divertor's upper surface
    notes['sampleXZY'] = [[-759.457, -625.500, -1797.591], # Old start point
                          [-759.603, -624.572, -1797.936], # Only for pattern
                          [-772.277, -620.864, -1794.112],
                          [-761.681, -610.036, -1769.498], # Computed,tube/plane
                          [-761.895, -620.231, -1764.921],
                          [-751.095, -609.687, -1741.154],
                          [-755.613, -580.944, -1751.852],
                          [-766.413, -591.488, -1775.620], # Edge of plane
                          [-763.902, -596.129, -1774.659], # Computed,tube/plane
                          [-774.498, -606.956, -1799.274], # Middle top of tube
                          [-763.246, -601.395, -1806.563],
                          [-767.575, -605.891, -1816.813],
                          [-763.932, -629.068, -1808.186],
                          [-764.112, -629.255, -1808.613],
                          [-767.755, -606.078, -1817.240],
                          [-772.084, -610.573, -1827.490],
                          [-768.441, -633.750, -1818.863],
                          [-768.622, -633.938, -1819.290],
                          [-772.265, -610.760, -1827.917],
                          [-776.594, -615.256, -1838.167],
                          [-772.950, -638.433, -1829.540],
                          [-773.131, -638.620, -1829.967],
                          [-776.774, -615.443, -1838.594],
                          [-781.103, -619.938, -1848.844],
                          [-777.460, -643.115, -1840.217],
                          [-777.640, -643.303, -1840.644],
                          [-781.283, -620.126, -1849.271],
                          [-785.612, -624.621, -1859.520],
                          [-781.969, -647.798, -1850.894],
                          [-782.149, -647.985, -1851.321],
                          [-785.793, -624.808, -1859.948],
                          [-790.122, -629.303, -1870.197],
                          [-786.478, -652.481, -1861.571],
                          [-786.659, -652.668, -1861.998],
                          [-790.302, -629.491, -1870.624],
                          [-794.631, -633.986, -1880.874],
                          [-790.988, -657.163, -1872.248],
                          [-791.168, -657.351, -1872.675],
                          [-794.811, -634.173, -1881.301]]
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

    R = np.hypot(notes['sampleXZY'][:,0],notes['sampleXZY'][:,2])
    Z = notes['sampleXZY'][:,1]

    d = np.sqrt(np.diff(R)**2+np.diff(Z)**2)
    indup = (np.abs(d-notes['Dl'])<1.e-6).nonzero()[0]
    e1 = np.array([R[indup+1]-R[indup], Z[indup+1]-Z[indup]])
    e1 = np.mean(e1,axis=1)
    e1 = e1/np.linalg.norm(e1)
    e2 = np.r_[-e1[1], e1[0]]

    nP = 4
    P0 = np.array([R[[0,2,3,4,5,6,7,8,9]],Z[[0,2,3,4,5,6,7,8,9]]])
    PEnd = P0[:,0:1] + e1[:,np.newaxis]*notes['DL']
    El = np.array([R[[1,10,11,12]],Z[[1,10,11,12]]])
    k = np.arange(0,notes['nb'])
    l = np.repeat(k*(notes['Dl']+notes['dl']), nP)
    Poly = np.tile(El,notes['nb']) + e1[:,np.newaxis]*l[np.newaxis,:]
    Poly = np.concatenate((P0,Poly[:,1:],PEnd),axis=1)
    Poly0 = Poly[:,[0,1,2,3,4,5,6,7,8,9,-3,-1]]

    # Poly1
    indI0 = np.arange(10,Poly.shape[1]-6,4)
    PI = (Poly[:,indI0] + Poly[:,indI0+3])/2.
    Poly1 = np.concatenate((Poly0[:,:9],PI,Poly0[:,-3:]),axis=1)

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
