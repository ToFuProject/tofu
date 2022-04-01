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

    notes = {}
    # Number of tiles vertically
    notes['nbZ'] = 10
    # Number of tiles toroidally
    notes['nbPhi'] = 3
    # Vertical average gap
    notes['dZ'] = 3.842
    # Toroidal average gap
    notes['dPhi'] = 1.500

    # Total toroidal width in equatorial plane
    notes['DPhi'] = 269.000
    # Total height
    notes['DZ'] = 1189.530

    # Radius of holes
    notes['r_outer'] = 6.000
    notes['r_inner'] = 4.000

    # sampleXZY
    notes['sampleXZY'] = [[-939.226, 592.992, 1626.787],#0
                          [-940.405, 594.553, 1628.830],
                          [-940.557, 594.491, 1629.092],
                          [-942.429, 591.177, 1632.335],
                          [-948.038, 557.622, 1642.051],
                          [-948.117, 555.038, 1642.187],
                          [-945.686, 531.169, 1637.977],
                          [-942.393, 498.833, 1632.273],
                          [-939.100, 466.497, 1626.569],
                          [-937.503, 450.819, 1623.803],
                          [-934.555, 446.918, 1618.696],#10
                          [-925.736, 450.510, 1603.421],
                          [-925.323, 445.938, 1602.706],
                          [-934.213, 443.137, 1618.105],
                          [-936.294, 437.419, 1621.708],
                          [-935.048, 421.614, 1619.551],
                          [-932.480, 389.017, 1615.103],
                          [-929.912, 356.419, 1610.655],
                          [-928.667, 340.614, 1608.498],
                          [-925.808, 336.453, 1603.547],
                          [-916.918, 339.255, 1588.148],#20
                          [-916.598, 334.511, 1587.595],
                          [-925.542, 332.506, 1603.086],
                          [-927.748, 326.981, 1606.907],
                          [-926.858, 311.080, 1605.365],
                          [-925.021, 278.286, 1602.183],
                          [-923.183, 245.491, 1599.000],
                          [-922.293, 229.590, 1597.458],
                          [-919.530, 225.178, 1592.672],
                          [-910.586, 227.182, 1577.181],
                          [-910.379, 222.555, 1576.822],#30
                          [-919.359, 221.351, 1592.376],
                          [-921.686, 216.028, 1596.407],
                          [-921.151, 200.064, 1595.480],
                          [-920.047, 167.138, 1593.569],
                          [-918.944, 134.211, 1591.658],
                          [-918.409, 118.247, 1590.731],
                          [-915.747, 113.593, 1586.121],
                          [-906.768, 114.797, 1570.567],
                          [-906.661, 110.014, 1570.383],
                          [-915.659, 109.615, 1585.967],#40
                          [-918.103, 104.505, 1590.200],
                          [-917.925, 088.509, 1589.892],
                          [-917.558, 055.517, 1589.258],
                          [-917.192, 022.525, 1588.623],
                          [-917.014, 006.529, 1588.315],
                          [-914.459, 001.641, 1583.890],
                          [-905.461, 002.041, 1568.305],
                          [-905.462,-002.605, 1568.306],
                          [-914.459,-002.201, 1583.890],
                          [-917.015,-007.087, 1588.316],#50
                          [-917.194,-023.083, 1588.627],
                          [-917.565,-056.075, 1589.268],
                          [-917.935,-089.067, 1589.910],
                          [-918.114,-105.063, 1590.221],
                          [-915.671,-110.174, 1585.989],
                          [-906.673,-110.578, 1570.404],
                          [-906.780,-115.360, 1570.590],
                          [-915.760,-114.152, 1586.143],
                          [-918.422,-118.805, 1590.754],
                          [-918.959,-134.769, 1591.684],#60
                          [-920.066,-167.695, 1593.601],
                          [-921.173,-200.620, 1595.519],
                          [-921.710,-216.584, 1596.448],
                          [-919.383,-221.908, 1592.419],
                          [-910.404,-223.116, 1576.865],
                          [-910.611,-227.744, 1577.225],
                          [-919.555,-225.735, 1592.716],
                          [-922.318,-230.146, 1597.502],
                          [-923.211,-246.046, 1599.048],
                          [-925.052,-278.840, 1602.236],#70
                          [-926.892,-311.634, 1605.425],
                          [-927.785,-327.534, 1606.971],
                          [-925.580,-333.061, 1603.151],
                          [-916.636,-335.069, 1587.660],
                          [-916.956,-339.813, 1588.214],
                          [-925.846,-337.007, 1603.612],
                          [-928.705,-341.167, 1608.564],
                          [-929.952,-356.971, 1610.724],
                          [-932.524,-389.568, 1615.178],
                          [-935.096,-422.165, 1619.633],#80
                          [-936.343,-437.969, 1621.793],
                          [-934.263,-443.687, 1618.191],
                          [-925.373,-446.493, 1602.793],
                          [-925.786,-451.065, 1603.508],
                          [-934.604,-447.468, 1618.782],
                          [-937.554,-451.369, 1623.890],
                          [-939.152,-467.046, 1626.659],
                          [-942.449,-499.381, 1632.369],
                          [-945.746,-531.715, 1638.079],
                          [-948.179,-555.583, 1642.294],#90
                          [-948.101,-558.166, 1642.159],
                          [-942.495,-591.724, 1632.449],
                          [-940.623,-595.039, 1629.207],
                          [-939.292,-593.541, 1626.902]]
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

    Poly = np.array([np.hypot(notes['sampleXZY'][:,0],notes['sampleXZY'][:,2]),
                     notes['sampleXZY'][:,1]])
    RMin = 1.775#np.min(Poly[0,:])-0.01
    P0 = np.array([[RMin],[Poly[1,0]]])
    PEnd = np.array([[RMin],[Poly[1,-1]]])
    Poly = np.concatenate((P0,Poly,PEnd),axis=1)
    ind0 = [0,4,6,-6,-4,-1]
    indt = [10,15,19,24,28,33,37,42,46,51,55,60,
            64,69,73,78,82,87]
    p0, p1 = [Poly[:,ind0[:3]]], [Poly[:,ind0[:3]]]
    for ii in range(0,len(indt),2):
        D0, D1 = Poly[:,indt[ii]], Poly[:,indt[ii+1]]
        u0, u1 = D0-Poly[:,indt[ii]-1], D1-Poly[:,indt[ii+1]+1]
        p0.append(_get_inter(D0,u0,D1,u1)[:,np.newaxis])
        u0, u1 = Poly[:,indt[ii]+1]-D0, Poly[:,indt[ii+1]-1]-D1
        p1.append(np.vstack((D0,_get_inter(D0,u0,D1,u1),D1)).T)
    p0.append(Poly[:,ind0[3:]])
    p1.append(Poly[:,ind0[3:]])
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
