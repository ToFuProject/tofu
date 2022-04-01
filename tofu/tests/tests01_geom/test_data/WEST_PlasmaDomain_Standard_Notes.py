#!/usr/bin/env python

# Built-in
import os
import argparse

# Common
import numpy as np

# tofu_west
from WEST_PFC_BumperInner_Notes import make_Poly as InBump
#from tofu_west.VesStruct.Inputs.WEST_Struct_VDE_Notes import make_Poly as VDE
from WEST_PFC_Ripple_Notes import make_Poly as Ripple
from WEST_PFC_DivLowGC_Notes import make_Poly as LowDiv
from WEST_PFC_DivUp_Notes import make_Poly as UpDiv
#from tofu_west.VesStruct.Inputs.WEST_Struct_Baffle_Notes import make_Poly as Baffle



_save = True
_here = os.path.abspath(os.path.dirname(__file__))
_Exp, _Cls, _name = os.path.split(__file__)[1].split('_')[:3]
assert not any([any([ss in s for ss in ['Notes','.']])
               for s in [_Exp, _Cls, _name]])


def get_notes_UpDivSupport():
    notes = {}
    # sampleXZY
    notes['sampleXZY'] = [[780.663, 799.580, -2162.398],
                          [665.129, 663.100, -1844.969],
                          [662.219, 661.100, -1819.433],
                          [699.823, 661.100, -1744.892]]
    notes['sampleXZY'] = np.array(notes['sampleXZY'])
    for kk in notes.keys():
        if type(notes[kk]) is dict:
            notes[kk]['In'] = notes[kk]['In']*1.e-3
            notes[kk]['Out'] = notes[kk]['Out']*1.e-3
        elif not 'nb' in kk:
            notes[kk] = notes[kk]*1.e-3
    return notes


def get_notes_PEI():

    notes = {}
    # Number of panles in toroidal direction
    notes['nbPhi'] = 30
    # Toroidal width in equatorial plane
    notes['DPhi'] = 343.653

    # sampleXZY
    notes['sampleXZY'] = [[1323.509, 648.510, -1332.719],
                          [1312.213, 555.372, -1312.213],
                          [1261.223, 442.976, -1323.715],
                          [1239.370, 155.735, -1299.672],
                          [1532.443, 049.922,  -921.707],
                          [1311.454, 000.000, -1217.927],
                          [1532.443,-049.922,  -921.707],
                          [1239.370,-155.735, -1299.672],
                          [1261.223,-442.976, -1325.715],
                          [1312.213,-555.372, -1312.213],
                          [1292.845,-648.518, -1347.774]]
    notes['sampleXZY'] = np.array(notes['sampleXZY'])
    for kk in notes.keys():
        if type(notes[kk]) is dict:
            notes[kk]['In'] = notes[kk]['In']*1.e-3
            notes[kk]['Out'] = notes[kk]['Out']*1.e-3
        elif not 'nb' in kk:
            notes[kk] = notes[kk]*1.e-3
    return notes

def get_notes_LowDivSupport():
    notes = {}
    # sampleXZY
    notes['sampleXZY'] = [[1861.032, -658.200, 266.381],
                          [1906.785, -658.200, 336.218],
                          [1931.405, -660.200, 340.559],
                          [2262.225, -795.922, 398.891]]
    notes['sampleXZY'] = np.array(notes['sampleXZY'])
    for kk in notes.keys():
        if type(notes[kk]) is dict:
            notes[kk]['In'] = notes[kk]['In']*1.e-3
            notes[kk]['Out'] = notes[kk]['Out']*1.e-3
        elif not 'nb' in kk:
            notes[kk] = notes[kk]*1.e-3
    return notes

def get_notes_LFS():
    notes = {}
    # sampleXZY
    notes['sampleXZY'] = [[2415.303, -797.969, -1325.194],
                          [2467.984, -779.176, -1355.609],
                          [2517.375, -750.879, -1384.125],
                          [2562.071, -717.338, -1409.930],
                          [2604.414, -680.461, -1434.377],
                          [2644.704, -639.931, -1457.639],
                          [2682.678, -595.887, -1479.563],
                          [2716.153, -551.311, -1498.890],
                          [2747.786, -502.835, -1517.153],
                          [2777.383, -450.254, -1534.241],
                          [2802.909, -396.912, -1554.978],
                          [2824.881, -342.348, -1561.664],
                          [2844.618, -282.957, -1573.059],
                          [2859.326, -226.839, -1581.551],
                          [2871.246, -167.234, -1588.432],
                          [2879.653, -105.919, -1593.286],
                          [2884.292, -043.463, -1596.035],
                          [2885.066,  015.167, -1596.412],
                          [2882.103,  080.155, -1594.701],
                          [2876.169,  133.956, -1591.275],
                          [2864.513,  195.563, -1587.924],
                          [2850.877,  260.786, -1576.673],
                          [2834.388,  315.143, -1567.152],
                          [2813.764,  370.994, -1555.245],
                          [2786.949,  431.177, -1539.764],
                          [2763.249,  476.096, -1526.080],
                          [2731.774,  528.094, -1507.908],
                          [2698.419,  575.507, -1488.651],
                          [2660.113,  622.771, -1466.535],
                          [2620.738,  664.693, -1443.802],
                          [2579.993,  702.297, -1420.278],
                          [2537.090,  736.594, -1395.507],
                          [2486.115,  771.280, -1366.077],
                          [2444.330,  795.360, -1341.952],
                          [2392.328,  820.525, -1311.929],
                          [2342.744,  839.444, -1283.302],
                          [2294.217,  853.544, -1255.285],
                          [2244.367,  863.562, -1226.504],
                          [2192.957,  869.452, -1196.822]]
    notes['sampleXZY'] = np.array(notes['sampleXZY'])
    for kk in notes.keys():
        if type(notes[kk]) is dict:
            notes[kk]['In'] = notes[kk]['In']*1.e-3
            notes[kk]['Out'] = notes[kk]['Out']*1.e-3
        elif not 'nb' in kk:
            notes[kk] = notes[kk]*1.e-3
    return notes


def make_Poly(save=_save, path=_here):

    nup = get_notes_UpDivSupport()
    nin = get_notes_PEI()
    nlow = get_notes_LowDivSupport()
    nout = get_notes_LFS()

    Rup = np.hypot(nup['sampleXZY'][:,0],nup['sampleXZY'][:,2])
    Zup = nup['sampleXZY'][:,1]
    Rin = np.hypot(nin['sampleXZY'][:,0],nin['sampleXZY'][:,2])
    Zin = nin['sampleXZY'][:,1]
    Rlow = np.hypot(nlow['sampleXZY'][:,0],nlow['sampleXZY'][:,2])
    Zlow = nlow['sampleXZY'][:,1]
    Rout = np.hypot(nout['sampleXZY'][:,0],nout['sampleXZY'][:,2])
    Zout = nout['sampleXZY'][:,1]

    indneg = (Zout>0).nonzero()[0][0]
    Rout = np.r_[Rout[:indneg], 3.2977, Rout[indneg:]]
    Zout = np.r_[Zout[:indneg],     0., Zout[indneg:]]

    Poly = np.array([np.r_[Rup,Rin,Rlow,Rout], np.r_[Zup,Zin,Zlow,Zout]])

    # Make V0 (inc. inner bumpers, up/low div, VDE/ripple and baffle)
    PIn = InBump()[0]
    PRip = Ripple()[0]
    PLD = LowDiv()[0]
    PUD = UpDiv()[0][:,::-1]
    PVDE = np.loadtxt(os.path.join(path,'WEST_PFC_VDE_V0.txt')).T
    PBa = np.loadtxt(os.path.join(path,'WEST_PFC_Baffle_V0.txt')).T
    ind = (Poly[0,:]>2.75) & (Poly[1,:]<0.78)
    Poly0 = [PVDE[:,:2],PUD[:,1:-1],PIn[:,2:-2],PLD[:,1:-1],
             PBa[:,6:-7],Poly[:,ind]]
    Poly0 = np.concatenate(tuple(Poly0),axis=1)

    # Poly1
    ind0 = (Poly[0,:]<1.9) & (Poly[1,:]>-0.6) & (Poly[1,:]<0.6)
    ind = Poly[0,:]>2.47
    Poly1 = [PUD[:,1:-1],Poly[:,ind0],PLD[:,1:-1],
             PBa[:,2:-2],Poly[:,ind]]
    Poly1 = np.concatenate(tuple(Poly1),axis=1)

    if save:
        cstr = '%s_%s'%(_Exp,_Cls)
        pathfilext = os.path.join(path, cstr+'_V0.txt')
        np.savetxt(pathfilext, Poly0.T)
        pathfilext = os.path.join(path, cstr+'_V1.txt')
        np.savetxt(pathfilext, Poly1.T)
        pathfilext = os.path.join(path, cstr+'_V2.txt')
        np.savetxt(pathfilext, Poly.T)
    return Poly0, Poly1, Poly, (nup,nin,nlow,nout)



if __name__=='__main__':

    # Parse input arguments
    msg = 'Launch creation of polygons txt from bash'
    parser = argparse.ArgumentParser(description = msg)

    parser.add_argument('-save', type=bool, help='save ?', default=_save)
    parser.add_argument('-path', type=str, help='saving path ?', default=_here)

    args = parser.parse_args()

    # Call wrapper function
    make_Poly(save=args.save, path=args.path)
