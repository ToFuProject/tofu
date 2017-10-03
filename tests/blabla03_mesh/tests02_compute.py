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
import tofu.mesh._compute as _tfm_c
import tofu.mesh._bsplines_cy as _tfm_bs


Root = tfpf.Find_Rootpath()
Addpath = '/tests/test02_mesh/'


#######################################################
#
#     Setup and Teardown
#
#######################################################

def setup_module(module):
    print ("") # this is to get a newline after the dots
    print ("--------------------------------------------")
    print ("--------------------------------------------")
    print ("        test02_compute")
    print ("")

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
#     Mesh
#
#######################################################




############################################
#####     Computing functions for Mesh1D
############################################


def test01_Mesh1D_set_Knots():
    Knots = [1,2,3,4,5,6]
    NKnots, Knots, NCents, Cents, Lengths, Length, Bary, Cents_Knotsind, Knots_Centsind = _tfm_c._Mesh1D_set_Knots(Knots)

    # Testing types and formats
    assert type(NKnots) is int
    assert type(Knots) is np.ndarray and Knots.ndim==1 and Knots.dtype.name=='float64'
    assert type(NCents) is int
    assert type(Cents) is np.ndarray and Cents.ndim==1 and Cents.dtype.name=='float64'
    assert type(Lengths) is np.ndarray and Lengths.ndim==1 and Lengths.dtype.name=='float64'
    assert type(Length) is float
    assert type(Bary) is float
    assert type(Cents_Knotsind) is np.ndarray and Cents_Knotsind.ndim==2 and Cents_Knotsind.dtype.name=='int64'
    assert type(Knots_Centsind) is np.ndarray and Knots_Centsind.ndim==2 and Knots_Centsind.dtype.name=='float64'

    # Testing values
    assert NKnots==6 and np.all(Knots==[1,2,3,4,5,6])
    assert NCents==5 and np.all(Cents==[1.5,2.5,3.5,4.5,5.5])
    assert np.all(Lengths==1.) and Length==5.
    assert Bary==3.5
    assert np.all(Cents_Knotsind==np.array([[0,1,2,3,4],[1,2,3,4,5]]))
    assert np.isnan(Knots_Centsind[0,0]) and np.isnan(Knots_Centsind[1,-1]) and Knots_Centsind[1,0]==0 and Knots_Centsind[0,-1]==4 and np.all(Knots_Centsind[:,1:-1]==np.array([[0,1,2,3],[1,2,3,4]]))



def test02_Mesh1D_sample():
    Knots = [0.,10.,110.]
    xx0 = _tfm_c._Mesh1D_sample(Knots, Sub=1., SubMode='abs', Test=True)
    xx1 = _tfm_c._Mesh1D_sample(Knots, Sub=0.1, SubMode='rel', Test=True)
    assert type(xx0) is np.ndarray and xx0.ndim==1 and np.all(xx0==np.linspace(0.,110.,111))
    assert type(xx1) is np.ndarray and xx1.ndim==1 and np.all(np.abs(xx1-np.unique(np.append(np.linspace(0.,10.,11), np.linspace(10.,110.,11))))<1.e-12)




############################################
#####     Computing functions for Mesh2D
############################################


K1 = np.array([1,2,3])
C1 = np.array([1.5,2.5])
K2 = np.array([10,20,30])
C2 = np.array([15,25])
Knots = np.array([np.tile(K1,(3,1)).flatten(),np.tile(K2,(3,1)).T.flatten()])
Cents = np.array([np.tile(C1,(2,1)).flatten(),np.tile(C2,(2,1)).T.flatten()])
NCents, NKnots = 4, 9



def test03_Mesh2D_set_Cents_Knotsind(NCents=NCents, K1=K1, K2=K2, Cents=Cents, Knots=Knots):
    ind = _tfm_c._Mesh2D_set_Cents_Knotsind(NCents, K1, K2, Cents, Knots)

    # Testing types and formats
    assert type(ind) is np.ndarray and ind.shape==(4,NCents) and ind.dtype.name=='int64'
    # Testing values
    assert np.all(ind==np.array([[0,1,3,4],[1,2,4,5],[4,5,7,8],[3,4,6,7]])), "Indices of knots surrounding each center should be given in counter-clockwise order !"


def test04_Mesh2D_set_Knots_Centsind(NKnots=NKnots, C1=C1, C2=C2, Cents=Cents, NCents=NCents):
    ind = _tfm_c._Mesh2D_set_Knots_Centsind(NKnots, C1, C2, Knots, Cents, NCents)
    # Testing types and formats
    assert type(ind) is np.ndarray and ind.shape==(4,NKnots) and ind.dtype.name=='int64'
    # Testing values
    assert np.all(ind==np.array([[-40,-40,-40,-40,0,1,-40,2,3],[-40,-40,-40,0,1,-40,2,3,-40],[0,1,-40,2,3,-40,-40,-40,-40],[-40,0,1,-40,2,3,-40,-40,-40]])), "Indices should be given on counter-clockwise order and indices of non-existing centers should be negative integers (-10*NCents) !"



def test05_Mesh2D_set_SurfVolBary(Knots=Knots, Cents=Cents, K1=K1, K2=K2, NCents=NCents):
    Cents_Knotsind = _tfm_c._Mesh2D_set_Cents_Knotsind(NCents, K1, K2, Cents, Knots)

    Surfs, Surf, VolAngs, VolAng, BaryS, BaryV, CentsV = _tfm_c._Mesh2D_set_SurfVolBary(Knots, Cents_Knotsind, Cents, VType='Lin')
    # Testing types and formats
    assert type(Surfs) is np.ndarray and Surfs.shape==(NCents,) and Surfs.dtype.name=='float64'
    assert type(Surf) is float
    assert type(BaryS) is np.ndarray and BaryS.shape==(2,) and BaryS.dtype.name=='float64'  # Shape changed to (2,)
    assert all([ss is None for ss in [VolAngs,VolAng,BaryV,CentsV]])
    # Testing values
    assert np.all(Surfs==10.) and Surf==40
    assert np.all(BaryS==np.array([2.,20.]))

    Surfs, Surf, VolAngs, VolAng, BaryS, BaryV, CentsV = _tfm_c._Mesh2D_set_SurfVolBary(Knots, Cents_Knotsind, Cents, VType='Tor')
    # Testing types and formats
    assert type(Surfs) is np.ndarray and Surfs.shape==(NCents,) and Surfs.dtype.name=='float64'
    assert type(Surf) is float
    assert type(VolAngs) is np.ndarray and VolAngs.shape==(NCents,) and VolAngs.dtype.name=='float64'
    assert type(VolAng) is float
    assert type(BaryS) is np.ndarray and BaryS.shape==(2,) and BaryS.dtype.name=='float64'  # Shape changed to (2,)
    assert type(BaryV) is np.ndarray and BaryV.shape==(2,) and BaryV.dtype.name=='float64'  # Shape changed to (2,)
    assert type(CentsV) is np.ndarray and CentsV.shape==(2,NCents) and CentsV.dtype.name=='float64'
    # Testing values
    assert np.all(Surfs==10.) and Surf==40
    assert np.all(VolAngs[::2]==15.) and  np.all(VolAngs[1::2]==25.) and VolAng==80.
    assert np.all(BaryS==np.array([2.,20.]))
    assert BaryV[0]==13./6. and BaryV[1]==20.
    assert np.all(CentsV==np.array([[14./9.,2.*19./15.,14./9.,2.*19/15.],[15.,15.,25.,25.]]))


def test06_Mesh2D_set_BoundPoly(Knots=Knots, NCents=NCents, K1=K1, K2=K2, Cents=Cents):
    Cents_Knotsind = _tfm_c._Mesh2D_set_Cents_Knotsind(NCents, K1, K2, Cents, Knots)
    BoundPol = _tfm_c._Mesh2D_set_BoundPoly(Knots, Cents_Knotsind, NCents)

    # Testing types and formats
    assert type(BoundPol) is np.ndarray and BoundPol.ndim==2 and BoundPol.shape[0]==2 and BoundPol.dtype.name=='float64'
    # Testing values
    assert np.all(BoundPol==np.array([[3.,3.,1.,1.,1.,3.,3.],[20.,10.,10.,20.,30.,30.,20.]]))


def test07_Mesh2D_get_SubMeshPolygon(Knots=Knots, NCents=NCents, K1=K1, K2=K2, Cents=Cents):
    Cents_Knotsind = _tfm_c._Mesh2D_set_Cents_Knotsind(NCents, K1, K2, Cents, Knots)
    Poly = np.array([[0.,4.,2.5,0.],[0.,0.,22.,22.]])

    indIn = _tfm_c._Mesh2D_get_SubMeshPolygon(Cents, Knots, Cents_Knotsind, Poly, InMode='Cents')
    # Testing types and formats
    assert type(indIn) is np.ndarray and indIn.shape==(NCents,) and indIn.dtype.name=='bool'
    # Testing values
    assert np.all(indIn==np.array([1,1,0,0],dtype=bool))

    indIn = _tfm_c._Mesh2D_get_SubMeshPolygon(Cents, Knots, Cents_Knotsind, Poly, InMode='Knots', NLim=2)
    # Testing types and formats
    assert type(indIn) is np.ndarray and indIn.shape==(NCents,) and indIn.dtype.name=='bool'
    # Testing values
    assert np.all(indIn==np.array([1,1,1,0],dtype=bool))



def test08_Mesh2D_get_CentBckg(C1=C1, C2=C2, Cents=Cents):
    Cents = np.copy(Cents[:,:-1])   # Remove one mesh element
    NC1, NC2, NC = C1.size, C2.size, Cents.shape[1]
    CentsBck, indCentBckInMesh, NumCentBck  = _tfm_c._Mesh2D_get_CentBckg(NC1, NC2, C1, C2, Cents, NC)

    # Testing types and formats
    assert type(CentsBck) is np.ndarray and CentsBck.ndim==2 and CentsBck.shape==(2,NC1*NC2) and CentsBck.dtype.name=='float64'
    assert type(indCentBckInMesh) is np.ndarray and indCentBckInMesh.ndim==1 and indCentBckInMesh.shape==(NC1*NC2,) and indCentBckInMesh.dtype.name=='bool'
    assert type(NumCentBck) is np.ndarray and NumCentBck.ndim==1 and NumCentBck.shape==(NC1*NC2,) and NumCentBck.dtype.name=='float64'   # int + NaN

    # Testing values
    assert np.all(CentsBck==np.array([[1.5,2.5,1.5,2.5],[15.,15.,25.,25.]]))
    assert np.all(indCentBckInMesh==[True,True,True,False])
    assert np.all(NumCentBck[:-1]==[0.,1.,2.]) and np.isnan(NumCentBck[-1])



def test09_Mesh2D_get_KnotsBckg(K1=K1, K2=K2, Knots=Knots):
    Knots = np.copy(Knots[:,:-1])   # Equivalent to removing one mesh element
    NK1, NK2, NK = K1.size, K2.size, Knots.shape[1]
    KnotsBck, indKnotsBckInMesh, NumKnotBck = _tfm_c._Mesh2D_get_KnotsBckg(NK1, NK2, K1, K2, Knots, NK)

    # Testing types and formats
    assert type(KnotsBck) is np.ndarray and KnotsBck.ndim==2 and KnotsBck.shape==(2,NK1*NK2) and KnotsBck.dtype.name=='float64'
    assert type(indKnotsBckInMesh) is np.ndarray and indKnotsBckInMesh.ndim==1 and indKnotsBckInMesh.shape==(NK1*NK2,) and indKnotsBckInMesh.dtype.name=='bool'
    assert type(NumKnotBck) is np.ndarray and NumKnotBck.ndim==1 and NumKnotBck.shape==(NK1*NK2,) and NumKnotBck.dtype.name=='float64'   # int + NaN

    # Testing values
    assert np.all(KnotsBck==np.array([[1.,2.,3.,1.,2.,3.,1.,2.,3.],[10.,10.,10.,20.,20.,20.,30.,30.,30.]]))
    assert np.all(indKnotsBckInMesh==[True,True,True,True,True,True,True,True,False])
    assert np.all(NumKnotBck[:-1]==[0.,1.,2.,3.,4.,5.,6.,7.]) and np.isnan(NumKnotBck[-1])


def test10_Mesh2D_isInside(Knots=Knots, NCents=NCents, K1=K1, K2=K2, Cents=Cents):
    Cents_Knotsind = _tfm_c._Mesh2D_set_Cents_Knotsind(NCents, K1, K2, Cents, Knots)
    BoundPoly = _tfm_c._Mesh2D_set_BoundPoly(Knots, Cents_Knotsind, NCents)
    Pts2D = np.array([[0.,1.5,2.5,5.],[0.,15.,25.,50.]])
    ind = _tfm_c._Mesh2D_isInside(Pts2D, BoundPoly)

    # Testing types and formats
    assert type(ind) is np.ndarray and ind.ndim==1 and ind.shape==(Pts2D.shape[1],) and ind.dtype.name=='bool'

    # Testing values
    assert np.all(ind==[False,True,True,False])


def test11_Mesh2D_sample(Knots=Knots, NCents=NCents, K1=K1, K2=K2, Cents=Cents):
    Cents_Knotsind = _tfm_c._Mesh2D_set_Cents_Knotsind(NCents, K1, K2, Cents, Knots)
    BoundPoly = _tfm_c._Mesh2D_set_BoundPoly(Knots, Cents_Knotsind, NCents)

    Pts0 = _tfm_c._Mesh2D_sample(Knots, Sub=0.1, SubMode='rel', BoundPoly=BoundPoly, Test=True)
    Pts1 = _tfm_c._Mesh2D_sample(Knots, Sub=(0.1,0.2), SubMode='rel', BoundPoly=None, Test=True)
    Pts2 = _tfm_c._Mesh2D_sample(Knots, Sub=(0.01,0.05), SubMode='abs', BoundPoly=BoundPoly, Test=True)

    # Testing types and formats
    assert type(Pts0) is np.ndarray and Pts0.ndim==2 and Pts0.shape[0]==2
    assert type(Pts1) is np.ndarray and Pts1.ndim==2 and Pts1.shape[0]==2
    assert type(Pts2) is np.ndarray and Pts2.ndim==2 and Pts2.shape[0]==2


############################################
#####     Computing functions for LBF1D
############################################



def test12_LBF1D_get_Coefs():
    Knots = np.linspace(0.,20.,11)
    LFunc = _tfm_bs.BSpline_LFunc(2, Knots, Deriv=0, Test=True)[0]
    NFunc = len(LFunc)
    xx = np.linspace(0.,20.,101)
    yy = np.exp(-(xx-10.)**2/8.) + 2.*np.exp(-(xx-13.)**2/0.5)
    ff = lambda x: np.exp(-(x-10.)**2/8.) + 2.*np.exp(-(x-13.)**2/0.5)

    Coefs0, res0 = _tfm_c._LBF1D_get_Coefs(LFunc, NFunc, Knots, xx=xx, yy=yy, ff=None, Sub=None, SubMode=None, Test=True)
    Coefs1, res1 = _tfm_c._LBF1D_get_Coefs(LFunc, NFunc, Knots, xx=None, yy=None, ff=ff, Sub=0.1, SubMode='rel', Test=True)
    for (cc,rr) in [(Coefs0,res0),(Coefs1, res1)]:
        assert type(cc) is np.ndarray and cc.shape==(NFunc,)
        assert type(rr) is np.ndarray










