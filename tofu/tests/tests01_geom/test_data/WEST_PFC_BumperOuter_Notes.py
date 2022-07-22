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
    notes = {'DPhi':{}, 'dPhi':{}}
    # Total toiroidal width in equatorial plane
    notes['DPhi'] = 346.756
    # Toroidal gap between central tiles in equatorial plane
    notes['dPhi'] = 1.500
    # Total vertical height
    notes['DZ'] = 1047.517
    # Poloidal gap between central tiles
    notes['dl'] = 2.249
    # Radial width
    notes['DR'] = 134.678

    # sampleXZY
    notes['sampleXZY'] = [[2159.632, 523.758, 2030.517],#0
                          [2149.423, 514.993, 2020.128],
                          [2148.227, 508.130, 2018.912],
                          [2154.559, 442.235, 2071.259],    # h
                          [2166.380, 414.998, 2085.893],    # h
                          [2178.412, 385.686, 2095.532],    # h
                          [2206.053, 371.041, 2077.756],    # e1
                          [2210.863, 369.335, 2082.651],
                          [2221.072, 378.101, 2093.040],
                          [2222.233, 374.874, 2094.221],
                          [2211.288, 368.155, 2083.083],#10
                          [2209.454, 361.586, 2081.216],    # e2
                          [2192.106, 347.606, 2109.467],    # h
                          [2201.248, 317.292, 2118.769],    # h
                          [2210.389, 286.979, 2128.072],    # h
                          [2234.025, 280.107, 2106.220],    # e1
                          [2230.026, 271.564, 2115.303],    # i
                          [2234.631, 268.948, 2119.988],
                          [2245.923, 275.667, 2130.785],
                          [2246.768, 272.245, 2131.645],
                          [2234.940, 267.696, 2120.303],#20
                          [2232.501, 261.541, 2117.821],    # i
                          [2240.819, 252.580, 2113.134],    # e2
                          [2220.353, 246.610, 2138.212],    # h
                          [2226.542, 214.813, 2144.510],    # h
                          [2232.731, 183.016, 2150.808],    # h
                          [2257.455, 167.114, 2130.063],    # e1
                          [2261.769, 163.634, 2134.453],
                          [2273.250, 168.183, 2146.136],
                          [2273.763, 164.630, 2146.658],
                          [2261.957, 162.335, 2134.644],#30
                          [2258.958, 156.705, 2131.592],    # e2
                          [2238.784, 141.097, 2156.967],    # h
                          [2241.907, 108.399, 2160.145],    # h
                          [2245.030, 075.701, 2163.323],    # h
                          [2268.232, 059.603, 2141.030],    # e1
                          [2272.178, 055.324, 2145.045],
                          [2283.984, 057.619, 2157.059],
                          [2284.156, 054.000, 2157.234],
                          [2272.241, 054.000, 2145.109],
                          [2268.736, 049.000, 2141.543],#40 e2
                          [2247.062, 033.000, 2165.391],    # h
                          [2272.659, 011.500, 2137.688],    # i
                          [2247.062, 000.000, 2165.391],    # h
                          [2247.062,-033.000, 2165.391],    # h
                          [2268.736,-049.000, 2141.543],    # e1
                          [2272.241,-054.000, 2145.109],
                          [2284.156,-054.000, 2157.234],
                          [2283.984,-057.619, 2157.059],
                          [2272.178,-055.324, 2145.045],
                          [2268.232,-059.603, 2141.030],#50 e2
                          [2245.030,-075.701, 2163.323],    # h
                          [2241.907,-108.399, 2160.145],    # h
                          [2238.784,-141.097, 2156.967],    # h
                          [2258.958,-156.705, 2131.592],    # e1
                          [2261.957,-162.335, 2134.644],
                          [2273.763,-164.630, 2146.658],
                          [2273.250,-168.183, 2146.136],
                          [2261.769,-163.634, 2134.453],
                          [2257.455,-167.114, 2130.063],#59 e2
                          [2232.731,-183.016, 2150.808],    # h
                          [2226.542,-214.813, 2144.510],    # h
                          [2220.353,-246.610, 2138.212],    # h
                          [2240.819,-252.580, 2113.134],    # e1
                          [2232.501,-261.541, 2117.821],    # i
                          [2234.940,-267.696, 2120.303],
                          [2246.768,-272.245, 2131.645],
                          [2245.923,-275.667, 2130.785],
                          [2234.631,-268.948, 2119.988],
                          [2230.026,-271.564, 2115.303],    # i
                          [2234.025,-280.107, 2106.220],#70 e2
                          [2210.389,-286.979, 2128.072],    # h
                          [2201.248,-317.292, 2118.769],    # h
                          [2192.106,-347.606, 2109.467],    # h
                          [2209.454,-361.586, 2081.216],    # e1
                          [2211.288,-368.155, 2083.083],
                          [2222.233,-374.874, 2094.221],
                          [2221.072,-378.101, 2093.040],
                          [2210.863,-369.335, 2082.651],
                          [2206.053,-371.041, 2077.756],#79 e2
                          [2178.412,-385.686, 2095.532],    # h
                          [2166.486,-413.961, 2083.395],    # h
                          [2154.559,-442.235, 2071.259],    # h
                          [2148.227,-508.130, 2018.912],
                          [2149.423,-514.993, 2020.128],
                          [2159.632,-523.758, 2030.517],
                          [2152.686,-542.612, 2243.735],    # Back
                          [2452.439,-057.045, 2300.172],
                          [3550.000,-057.150, 0000.000],
                          [3550.000, 057.150, 0000.000],
                          [2452.118, 056.737, 2302.204],#90
                          [2151.698, 544.667, 2242.729]]
    notes['sampleXZY'] = np.array(notes['sampleXZY'])
    notes['ind_h'] = [3,4,5,12,13,14,23,24,25,32,33,34,41,43,44,51,52,53,
                      60,61,62,71,72,73,80,81,82]
    notes['ind_Back'] = [86,87,88,89,90,91]
    notes['ind_i'] = [16,21,42,64,69]
    notes['ind_e1'] = [6,15,26,35,45,54,63,74]
    notes['ind_e2'] = [11,22,31,40,50,59,70,79]


    for kk in notes.keys():
        if type(notes[kk]) is dict:
            notes[kk]['In'] = notes[kk]['In']*1.e-3
            notes[kk]['Out'] = notes[kk]['Out']*1.e-3
        elif not 'nb' in kk and not 'ind' in kk:
            notes[kk] = notes[kk]*1.e-3
    return notes

def _get_intersect(D0,u0,D1,u1):
    k = -np.cross(D0-D1,u1)/np.cross(u0,u1)
    return D0 + k*u0


def make_Poly(save=_save, path=_here):
    notes = get_notes()

    Poly = np.array([np.hypot(notes['sampleXZY'][:,0],notes['sampleXZY'][:,2]),
                     notes['sampleXZY'][:,1]])
    # Finish V2
    ind = np.zeros((Poly.shape[1],),dtype=bool)
    ind[notes['ind_h']] = True
    ind[notes['ind_i']] = True
    nind = np.arange(0,Poly.shape[1])
    Polybis = Poly.copy()
    for ii in range(0,len(notes['ind_e1'])+1):
        i0 = notes['ind_e2'][ii-1] if ii>0 else 2
        i1 = notes['ind_e1'][ii] if ii<len(notes['ind_e1']) else 83
        indi = ind & (nind>i0) & (nind<i1)
        D0 = Poly[:,i0]
        u = Poly[:,i1]-D0
        un2 = np.linalg.norm(u)**2
        k = -np.sum((D0[:,np.newaxis]-Poly[:,indi])*u[:,np.newaxis],axis=0)/un2
        Polybis[:,indi] = D0[:,np.newaxis] + k[np.newaxis,:]*u[:,np.newaxis]

    # Make V0 and V1
    ind0 = np.ones((Poly.shape[1],),dtype=bool)
    ind0[notes['ind_h']] = False
    ind0[notes['ind_i']] = False
    ind0[[0,1,84,85]] = False
    Poly0 = Poly[:,ind0]
    inde1 = np.zeros((Poly.shape[1],),dtype=bool)
    inde2 = np.zeros((Poly.shape[1],),dtype=bool)
    inde1[notes['ind_e1']] = True
    inde2[notes['ind_e2']] = True
    inde1 = inde1[ind0].nonzero()[0]
    inde2 = inde2[ind0].nonzero()[0]
    p0 = [Poly0[:,:inde1[0]]]
    p1 = [Poly0[:,:inde1[0]]]
    for ii in range(0,len(inde1)):
        D0, D1 = Poly0[:,inde1[ii]],        Poly0[:,inde2[ii]]
        u0, u1 = D0-Poly0[:,inde1[ii]-1],   D1-Poly0[:,inde2[ii]+1]
        p0.append(_get_intersect(D0,u0,D1,u1)[:,np.newaxis])
        u0, u1 = Poly0[:,inde1[ii]+1]-D0,   Poly0[:,inde2[ii]-1]-D1
        p1.append(np.vstack([D0,_get_intersect(D0,u0,D1,u1),D1]).T)

    p0.append(Poly0[:,inde2[ii]+1:])
    p1.append(Poly0[:,inde2[ii]+1:])
    Poly0 = np.concatenate(tuple(p0),axis=1)
    Poly1 = np.concatenate(tuple(p1),axis=1)

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
