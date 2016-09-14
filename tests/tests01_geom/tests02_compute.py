"""
This module contains tests for tofu.geom in its structured version
"""

# External modules
import os
import numpy as np
import matplotlib.pyplot as plt


# Nose-specific
from nose import with_setup # optional


# Importing package tofu.geom
import tofu.defaults as tfd
import tofu.pathfile as tfpf
from tofu.geom import _compute as _tfg_c


Root = tfpf.Find_Rootpath()


#######################################################
#
#     Setup and Teardown
#
#######################################################

def setup_module(module):
    print ("") # this is to get a newline after the dots
    #print ("setup_module before anything in this file")

def teardown_module(module):
    #os.remove(VesTor.Id.SavePath + VesTor.Id.SaveName + '.npz')
    #os.remove(VesLin.Id.SavePath + VesLin.Id.SaveName + '.npz')
    #print ("teardown_module after everything in this file")
    #print ("") # this is to get a newline
    pass


#def my_setup_function():
#    print ("my_setup_function")

#def my_teardown_function():
#    print ("my_teardown_function")

#@with_setup(my_setup_function, my_teardown_function)
#def test_numbers_3_4():
#    print 'test_numbers_3_4  <============================ actual test code'
#    assert multiply(3,4) == 12

#@with_setup(my_setup_function, my_teardown_function)
#def test_strings_a_3():
#    print 'test_strings_a_3  <============================ actual test code'
#    assert multiply('a',3) == 'aaa'






#######################################################
#
#      Ves functions
#
#######################################################


thet = np.linspace(0., 2.*np.pi, 101)
Poly = np.array([2.+1.*np.cos(thet), 0.+1.*np.sin(thet)])


def test01_Ves_set_Poly(Poly=Poly):
    Poly, NP, P1Max, P1Min, P2Max, P2Min, BaryP, BaryL, Surf, BaryS, DLong, Vol, BaryV, Vect, Vin = _tfg_c._Ves_set_Poly(Poly, 'C', 'Lin', DLong=[-1.,1.], Clock=False)
    Poly, NP, P1Max, P1Min, P2Max, P2Min, BaryP, BaryL, Surf, BaryS, DLong, Vol, BaryV, Vect, Vin = _tfg_c._Ves_set_Poly(Poly, 'C', 'Tor', DLong=None, Clock=True)
    Poly, NP, P1Max, P1Min, P2Max, P2Min, BaryP, BaryL, Surf, BaryS, DLong, Vol, BaryV, Vect, Vin = _tfg_c._Ves_set_Poly(Poly, 'F', 'Lin', DLong=[-1.,1.], Clock=False)
    Poly, NP, P1Max, P1Min, P2Max, P2Min, BaryP, BaryL, Surf, BaryS, DLong, Vol, BaryV, Vect, Vin = _tfg_c._Ves_set_Poly(Poly, 'F', 'Tor', DLong=None, Clock=True)

    # Testing Types and format
    assert type(Poly) is np.ndarray and Poly.ndim==2 and Poly.shape[0]==2 and Poly.dtype.name=='float64'
    assert type(NP) is int and NP==Poly.shape[1]
    assert all([type(pp) is np.ndarray and pp.shape==(2,) for pp in [P1Max, P1Min, P2Max, P2Min, BaryP, BaryL, BaryS]])
    assert BaryV is None or type(BaryV) is np.ndarray and BaryV.shape==(2,)
    assert type(Surf) is float
    assert Vol is None or type(Vol) is float
    assert DLong is None or (hasattr(DLong,'__iter__') and len(DLong)==2)
    assert type(Vect) is np.ndarray and Vect.ndim==2 and Vect.shape[0]==2
    assert type(Vin) is np.ndarray and Vin.ndim==2 and Vin.shape[0]==2

    # Testing values
    assert NP==102
    assert np.all(np.abs(P1Max-[3.,0.])<1.e-3) and np.all(np.abs(P1Min-[1.,0.])<1.e-3) and np.all(np.abs(P2Min-[2.,-1.])<1.e-3) and np.all(np.abs(P2Max-[2.,1.])<1.e-3)
    assert np.all(np.abs(BaryP-[2.,0.])<1.e-2) and np.all(np.abs(BaryL-[2.,0.])<1.e-3) and np.all(np.abs(BaryS-[2.,0.])<1.e-3) and np.all(np.abs(BaryV-[2.125,0.])<1.e-3)
    assert np.abs(Surf-np.pi)<1.e-2 and np.abs(Vol-6.279)<1.e-2


def test02_Ves_isInside(Poly=Poly):
    Pts = np.array([[0.,0.,0.,0.],[0.,1.5,2.5,3.5],[0.,0.,0.,0.]])
    ind1 = _tfg_c._Ves_isInside(Poly, 'Lin', [-1.,1.], Pts, In='(X,Y,Z)')
    ind2 = _tfg_c._Ves_isInside(Poly, 'Tor', None, Pts[1:,:], In='(R,Z)')
    assert np.all(ind1==[False,True,True,False])
    assert np.all(ind1==ind2)


def test03_Ves_get_InsideConvexPoly(Poly=Poly):
    Poly, NP, P1Max, P1Min, P2Max, P2Min, BaryP, BaryL, Surf, BaryS, DLong, Vol, BaryV, Vect, Vin = _tfg_c._Ves_set_Poly(Poly, 'C', 'Lin', DLong=[-1.,1.], Clock=False)
    Poly1 = _tfg_c._Ves_get_InsideConvexPoly(Poly, P2Min, P2Max, BaryS, RelOff=tfd.TorRelOff, ZLim='Def', Spline=True, Splprms=tfd.TorSplprms, NP=tfd.TorInsideNP, Plot=False, Test=True)
    ind1 = _tfg_c._Ves_isInside(Poly, 'Lin', [-1.,1.], np.array([np.zeros((Poly1.shape[1])),Poly1[0,:],Poly1[1,:]]), In='(X,Y,Z)')
    Poly, NP, P1Max, P1Min, P2Max, P2Min, BaryP, BaryL, Surf, BaryS, DLong, Vol, BaryV, Vect, Vin = _tfg_c._Ves_set_Poly(Poly, 'C', 'Tor', DLong=None, Clock=False)
    Poly2 = _tfg_c._Ves_get_InsideConvexPoly(Poly, P2Min, P2Max, BaryS, RelOff=tfd.TorRelOff, ZLim='Def', Spline=True, Splprms=tfd.TorSplprms, NP=tfd.TorInsideNP, Plot=False, Test=True)
    ind2 = _tfg_c._Ves_isInside(Poly, 'Tor', None, Poly2, In='(R,Z)')
    assert type(Poly1) is np.ndarray and Poly1.ndim==2 and Poly1.shape==(2,tfd.TorInsideNP+1)
    assert type(Poly2) is np.ndarray and Poly2.ndim==2 and Poly2.shape==(2,tfd.TorInsideNP+1)
    assert np.all(ind1) and np.all(ind2)


def test04_Ves_get_MeshCrossSection(Poly=Poly):
    Poly, NP, P1Max, P1Min, P2Max, P2Min, BaryP, BaryL, Surf, BaryS, DLong, Vol, BaryV, Vect, Vin = _tfg_c._Ves_set_Poly(Poly, 'C', 'Lin', DLong=[-1.,1.], Clock=False)
    Pts, X1, X2, NumX1, NumX2 = _tfg_c._Ves_get_MeshCrossSection(P1Min, P1Max, P2Min, P2Max, Poly, 'Lin', DLong=[-1.,1.], CrossMesh=[0.01,0.01], CrossMeshMode='abs', Test=True)
    Poly, NP, P1Max, P1Min, P2Max, P2Min, BaryP, BaryL, Surf, BaryS, DLong, Vol, BaryV, Vect, Vin = _tfg_c._Ves_set_Poly(Poly, 'C', 'Tor', DLong=None, Clock=False)
    Pts, X1, X2, NumX1, NumX2 = _tfg_c._Ves_get_MeshCrossSection(P1Min, P1Max, P2Min, P2Max, Poly, 'Tor', DLong=None, CrossMesh=[0.01,0.01], CrossMeshMode='rel', Test=True)
    assert type(Pts) is np.ndarray and Pts.ndim==2 and Pts.shape[0]==2
    assert type(NumX1) is int and type(NumX2) is int
    assert type(X1) is np.ndarray and X1.shape==(NumX1,)
    assert type(X2) is np.ndarray and X2.shape==(NumX2,)









#######################################################
#
#      LOS functions
#
#######################################################




