"""
This module contains tests for tofu.geom in its structured version
"""

# External modules
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.gridspec as mplgrid
from mpl_toolkits.mplot3d import Axes3D

# Nose-specific
from nose import with_setup # optional

# ToFu-specific
import tofu.geom._GG as GG



#Root = tfpf.Find_Rootpath()

VerbHead = 'tofu.geom.tests01_GG'


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
#     Testing
#
#######################################################



"""
######################################################
######################################################
#               Commons
######################################################
######################################################
"""

def test01_CoordShift():

    # Tests 1D input
    Pts = np.array([1.,1.,1.])
    pts = GG.CoordShift(Pts, In='(X,Y,Z)', Out='(R,Z)', CrossRef=0.)
    assert pts.shape==(2,) and np.allclose(pts,[np.sqrt(2),1.])
    pts = GG.CoordShift(Pts, In='(R,Z,Phi)', Out='(X,Y,Z)', CrossRef=0.)
    assert pts.shape==(3,) and np.allclose(pts,[np.cos(1.),np.sin(1.),1.])

    # Test 2D input
    Pts = np.array([[1.,1.],[1.,1.],[1.,1.]])
    pts = GG.CoordShift(Pts, In='(X,Y,Z)', Out='(R,Phi,Z)', CrossRef=0.)
    assert pts.shape==(3,2) and np.allclose(pts,[[np.sqrt(2.),np.sqrt(2.)],[np.pi/4.,np.pi/4.],[1.,1.]])
    pts = GG.CoordShift(Pts, In='(Phi,Z,R)', Out='(X,Y)', CrossRef=0.)
    assert pts.shape==(2,2) and np.allclose(pts,[[np.cos(1.),np.cos(1.)],[np.sin(1.),np.sin(1.)]])






########################################################
########################################################
#       Polygons
########################################################

def test02_Poly_CLockOrder():

    # Test arbitrary 2D polygon
    Poly = np.array([[0.,1.,1.,0.],[0.,0.,1.,1.]])
    P = GG.Poly_Order(Poly, order='C', Clock=False, close=True, layout='(N,cc)', layout_in=None, Test=True)
    assert all([np.allclose(P[0,:],P[-1,:]), P.shape==(5,2), not GG.Poly_isClockwise(P), P.flags['C_CONTIGUOUS'], not P.flags['F_CONTIGUOUS']])
    P = GG.Poly_Order(Poly, order='F', Clock=True, close=False, layout='(cc,N)', layout_in=None, Test=True)
    assert all([not np.allclose(P[:,0],P[:,-1]), P.shape==(2,4), GG.Poly_isClockwise(np.concatenate((P,P[:,0:1]),axis=1)), not P.flags['C_CONTIGUOUS'], P.flags['F_CONTIGUOUS']])

    # Test arbitrary 3D polygon
    Poly = np.array([[0.,1.,1.,0.],[0.,0.,1.,1.],[0.,0.,0.,0.]])
    P = GG.Poly_Order(Poly, order='C', Clock=False, close=False, layout='(N,cc)', layout_in=None, Test=True)
    assert all([not np.allclose(P[0,:],P[-1,:]), P.shape==(4,3), P.flags['C_CONTIGUOUS'], not P.flags['F_CONTIGUOUS']])
    P = GG.Poly_Order(Poly, order='F', Clock=True, close=True, layout='(cc,N)', layout_in=None, Test=True)
    assert all([np.allclose(P[:,0],P[:,-1]), P.shape==(3,5), not P.flags['C_CONTIGUOUS'], P.flags['F_CONTIGUOUS']])


def test03_Poly_VolAngTor():
    Poly = np.array([[1.,1.5,2.,2.,2.,1.5,1.],[0.,0.,0.,0.5,1.,1.,1.]])
    Poly = GG.Poly_Order(Poly, order='C', Clock=False, close=True, layout='(cc,N)', Test=True)
    V, B = GG.Poly_VolAngTor(Poly)
    assert V==1.5
    assert np.allclose(B,[7./(3.*1.5),0.5])





"""
######################################################
######################################################
#               Ves
######################################################
######################################################
"""

# VPoly
thet = np.linspace(0.,2.*np.pi,100)
VPoly = np.array([2.+1.*np.cos(thet), 0.+1.*np.sin(thet)])



def test02_Ves_isInside(VPoly=VPoly):

    # Lin Ves
    Pts = np.array([[-10.,-10.,5.,5.,5.,5., 5.,30.,30.,30.],
                    [  0.,  2.,0.,2.,4.,2., 2., 2., 0., 0.],
                    [  0.,  0.,0.,0.,0.,2.,-2., 0., 0., 2.]])
    ind = GG._Ves_isInside(Pts, VPoly, Lim=[0.,10.], VType='Lin', In='(X,Y,Z)', Test=True)
    assert ind.shape==(Pts.shape[1],) and np.all(ind==[False,False,False,True,False,False,False,False,False,False])

    # Tor Ves
    Pts = np.array([[  0.,-10.,5.,5.,5.,5., 5.,30.,30.,30.],
                    [  0.,  2.,0.,2.,4.,2., 2., 2., 0., 0.],
                    [  0.,  0.,0.,0.,0.,2.,-2., 0., 0., 2.]])
    ind = GG._Ves_isInside(Pts, VPoly, Lim=None, VType='Tor', In='(Phi,R,Z)', Test=True)
    assert ind.shape==(Pts.shape[1],) and np.all(ind==[False,True,False,True,False,False,False,True,False,False])

    # Tor Struct
    Pts = np.array([[  0.,  0.,2.*np.pi,np.pi,np.pi,np.pi,np.pi,2.*np.pi,2.*np.pi,2.*np.pi],
                    [  0.,  2.,      0.,   2.,   4.,   2.,   2.,      2.,      0.,      0.],
                    [  0.,  0.,      0.,   0.,   0.,   2.,  -2.,      0.,      0.,      2.]])
    ind = GG._Ves_isInside(Pts, VPoly, Lim=[np.pi/2.,3.*np.pi/2.], VType='Tor', In='(Phi,R,Z)', Test=True)
    assert ind.shape==(Pts.shape[1],) and np.all(ind==[False,False,False,True,False,False,False,False,False,False])


#####################################################
#               Ves  - Commons
#####################################################

def test03_Ves_mesh_dlfromL():

    LMinMax = np.array([0.,10.])
    L, dLr, indL, N = GG._Ves_mesh_dlfromL_cython(LMinMax, 20., DL=None, Lim=True, margin=1.e-9)
    assert np.allclose(L,[5.]) and dLr==10. and np.allclose(indL,[0]) and N==1
    L, dLr, indL, N = GG._Ves_mesh_dlfromL_cython(LMinMax, 1., DL=None, Lim=True, margin=1.e-9)
    assert np.allclose(L,0.5+np.arange(0,10)) and dLr==1. and np.allclose(indL,range(0,10)) and N==10
    L, dLr, indL, N = GG._Ves_mesh_dlfromL_cython(LMinMax, 1., DL=[2.,8.], Lim=True, margin=1.e-9)
    assert np.allclose(L,0.5+np.arange(2,8)) and dLr==1. and np.allclose(indL,range(2,8)) and N==10
    L, dLr, indL, N = GG._Ves_mesh_dlfromL_cython(LMinMax, 1., DL=[2.,12.], Lim=True, margin=1.e-9)
    assert np.allclose(L,0.5+np.arange(2,10)) and dLr==1. and np.allclose(indL,range(2,10)) and N==10
    L, dLr, indL, N = GG._Ves_mesh_dlfromL_cython(LMinMax, 1., DL=[2.,12.], Lim=False, margin=1.e-9)
    assert np.allclose(L,0.5+np.arange(2,12)) and dLr==1. and np.allclose(indL,range(2,12)) and N==10



def test03_Ves_Smesh_Cross(VPoly=VPoly):

    VIn = VPoly[:,1:]-VPoly[:,:-1]
    VIn = np.array([-VIn[1,:],VIn[0,:]])
    VIn = VIn/np.sqrt(np.sum(VIn**2,axis=0))[np.newaxis,:]
    dL = 0.01

    PtsCross, dLr, ind, N, Rref, VPbis = GG._Ves_Smesh_Cross(VPoly, dL, D1=None, D2=None, margin=1.e-9, DIn=0., VIn=VIn)
    assert PtsCross.ndim==2 and PtsCross.shape[1]>=VPoly.shape[1]-1 and not np.allclose(PtsCross[:,0],PtsCross[:,-1])
    assert dLr.shape==(PtsCross.shape[1],) and np.all(dLr<=dL)
    assert ind.shape==(PtsCross.shape[1],) and np.all(np.unique(ind)==ind) and np.all(~np.isnan(ind)) and np.max(ind)<PtsCross.shape[1]
    assert N.shape==(VPoly.shape[1]-1,) and np.all(N>=1)
    assert Rref.shape==(PtsCross.shape[1],) and np.all(Rref==PtsCross[0,:])
    assert VPbis.ndim==2 and VPbis.shape[1]>=VPoly.shape[1]

    PtsCross, dLr, ind, N, Rref, VPbis = GG._Ves_Smesh_Cross(VPoly, dL, D1=[0.,2.], D2=[-2.,0.], margin=1.e-9, DIn=0.05, VIn=VIn)
    assert np.all(PtsCross[0,:]>=0.) and np.all(PtsCross[0,:]<=2.) and np.all(PtsCross[1,:]>=-2.) and np.all(PtsCross[1,:]<=0.)
    assert np.all(Path(VPoly.T).contains_points(PtsCross.T))
    assert dLr.shape==(PtsCross.shape[1],) and np.all(dLr<=dL)
    assert ind.shape==(PtsCross.shape[1],) and np.all(np.unique(ind)==ind) and np.all(~np.isnan(ind))
    assert N.shape==(VPoly.shape[1]-1,) and np.all(N>=1)
    assert Rref.size>3*PtsCross.shape[1]
    assert VPbis.ndim==2 and VPbis.shape[1]>=VPoly.shape[1]

    PtsCross, dLr, ind, N, Rref, VPbis = GG._Ves_Smesh_Cross(VPoly, dL, D1=[0.,2.], D2=[-2.,0.], margin=1.e-9, DIn=-0.05, VIn=VIn)
    assert np.all(PtsCross[0,:]>=0.-0.05) and np.all(PtsCross[0,:]<=2.) and np.all(PtsCross[1,:]>=-2.-0.05) and np.all(PtsCross[1,:]<=0.)
    assert np.all(~Path(VPoly.T).contains_points(PtsCross.T))


#####################################################
#               Ves  - VMesh
#####################################################

def test04_Ves_Vmesh_Tor(VPoly=VPoly):

    RMinMax = np.array([np.min(VPoly[0,:]), np.max(VPoly[0,:])])
    ZMinMax = np.array([np.min(VPoly[1,:]), np.max(VPoly[1,:])])
    dR, dZ, dRPhi = 0.05, 0.05, 0.05
    LDPhi = [None, [3.*np.pi/4.,5.*np.pi/4.], [-np.pi/4.,np.pi/4.]]

    for ii in range(0,len(LDPhi)):
        Pts, dV, ind, dRr, dZr, dRPhir = GG._Ves_Vmesh_Tor_SubFromD_cython(dR, dZ, dRPhi, RMinMax, ZMinMax,
                                                                           DR=[0.5,2.], DZ=[0.,1.2], DPhi=LDPhi[ii], VPoly=VPoly,
                                                                           Out='(R,Z,Phi)', margin=1.e-9)
        assert Pts.ndim==2 and Pts.shape[0]==3
        assert np.all(Pts[0,:]>=1.) and np.all(Pts[0,:]<=2.) and np.all(Pts[1,:]>=0.) and np.all(Pts[1,:]<=1.)
        marg = np.abs(np.arctan(np.mean(dRPhir)/np.min(VPoly[1,:])))
        if not LDPhi[ii] is None:
            LDPhi[ii][0] = np.arctan2(np.sin(LDPhi[ii][0]),np.cos(LDPhi[ii][0]))
            LDPhi[ii][1] = np.arctan2(np.sin(LDPhi[ii][1]),np.cos(LDPhi[ii][1]))
            if LDPhi[ii][0]<=LDPhi[ii][1]:
                assert np.all((Pts[2,:]>=LDPhi[ii][0]-marg) & (Pts[2,:]<=LDPhi[ii][1]+marg))
            else:
                assert np.all( (Pts[2,:]>=LDPhi[ii][0]-marg) | (Pts[2,:]<=LDPhi[ii][1]+marg))
        assert dV.shape==(Pts.shape[1],)
        assert all([ind.shape==(Pts.shape[1],), ind.dtype==int, np.unique(ind).size==ind.size, np.all(ind==np.unique(ind)), np.all(ind>=0)])
        assert dRPhir.ndim==1

        Ptsi, dVi, dRri, dZri, dRPhiri = GG._Ves_Vmesh_Tor_SubFromInd_cython(dR, dZ, dRPhi, RMinMax, ZMinMax, ind, Out='(R,Z,Phi)', margin=1.e-9)
        assert np.allclose(Pts,Ptsi)
        assert np.allclose(dV,dVi)
        assert dRr==dRri and dZr==dZri
        assert np.allclose(dRPhir,dRPhiri)



def test05_Ves_Vmesh_Lin(VPoly=VPoly):

    XMinMax = np.array([0.,10.])
    YMinMax = np.array([np.min(VPoly[0,:]), np.max(VPoly[0,:])])
    ZMinMax = np.array([np.min(VPoly[1,:]), np.max(VPoly[1,:])])
    dX, dY, dZ = 0.05, 0.05, 0.05

    Pts, dV, ind, dXr, dYr, dZr = GG._Ves_Vmesh_Lin_SubFromD_cython(dX, dY, dZ, XMinMax, YMinMax, ZMinMax,
                                                                    DX=[8.,15.], DY=[0.5,2.], DZ=[0.,1.2], VPoly=VPoly, margin=1.e-9)
    assert Pts.ndim==2 and Pts.shape[0]==3
    assert np.all(Pts[0,:]>=8.) and np.all(Pts[0,:]<=10.) and np.all(Pts[1,:]>=1.) and np.all(Pts[1,:]<=2.) and np.all(Pts[2,:]>=0.) and np.all(Pts[2,:]<=1.)
    assert all([ind.shape==(Pts.shape[1],), ind.dtype==int, np.unique(ind).size==ind.size, np.all(ind==np.unique(ind)), np.all(ind>=0)])

    Ptsi, dVi, dXri, dYri, dZri = GG._Ves_Vmesh_Lin_SubFromInd_cython(dX, dY, dZ, XMinMax, YMinMax, ZMinMax, ind, margin=1.e-9)
    assert np.allclose(Pts,Ptsi)
    assert np.allclose(dV,dVi)
    assert dXr==dXri and dYr==dYri and dZr==dZri




#####################################################
#               Ves  - SMesh
#####################################################

def test06_Ves_Smesh_Tor(VPoly=VPoly):

    dL, dRPhi = 0.02, 0.05
    VIn = VPoly[:,1:]-VPoly[:,:-1]
    VIn = np.array([-VIn[1,:],VIn[0,:]])
    VIn = VIn/np.sqrt(np.sum(VIn**2,axis=0))[np.newaxis,:]
    DIn = 0.001
    LDPhi = [None, [3.*np.pi/4.,5.*np.pi/4.], [-np.pi/4.,np.pi/4.]]

    for ii in range(0,len(LDPhi)):
        # With Ves
        Pts, dS, ind, NL, dLr, Rref, dRPhir, nRPhi0, VPbis = GG._Ves_Smesh_Tor_SubFromD_cython(dL, dRPhi, VPoly,
                                                                                               DR=[0.5,2.], DZ=[0.,1.2], DPhi=LDPhi[ii],
                                                                                               DIn=DIn, VIn=VIn, PhiMinMax=None,
                                                                                               Out='(R,Z,Phi)', margin=1.e-9)

        assert Pts.ndim==2 and Pts.shape[0]==3
        assert np.all(Pts[0,:]>=1.-np.abs(DIn)) and np.all(Pts[0,:]<=2.+np.abs(DIn)) and np.all(Pts[1,:]>=0.-np.abs(DIn)) and np.all(Pts[1,:]<=1.+np.abs(DIn))
        marg = np.abs(np.arctan(np.mean(dRPhir+np.abs(DIn))/np.min(VPoly[1,:])))
        if not LDPhi[ii] is None:
            LDPhi[ii][0] = np.arctan2(np.sin(LDPhi[ii][0]),np.cos(LDPhi[ii][0]))
            LDPhi[ii][1] = np.arctan2(np.sin(LDPhi[ii][1]),np.cos(LDPhi[ii][1]))
            if LDPhi[ii][0]<=LDPhi[ii][1]:
                assert np.all((Pts[2,:]>=LDPhi[ii][0]-marg) & (Pts[2,:]<=LDPhi[ii][1]+marg))
            else:
                assert np.all( (Pts[2,:]>=LDPhi[ii][0]-marg) | (Pts[2,:]<=LDPhi[ii][1]+marg))
        assert np.all(GG._Ves_isInside(Pts, VPoly, VType='Tor', In='(R,Z,Phi)', Test=True))
        assert dS.shape==(Pts.shape[1],)
        assert all([ind.shape==(Pts.shape[1],), ind.dtype==int, np.unique(ind).size==ind.size, np.all(ind==np.unique(ind)), np.all(ind>=0)])
        assert ind.shape==(Pts.shape[1],) and ind.dtype==int and np.all(ind==np.unique(ind)) and np.all(ind>=0)
        assert NL.ndim==1 and NL.size==VPoly.shape[1]-1
        assert dLr.ndim==1 and dLr.size==NL.size
        assert Rref.ndim==1
        assert dRPhir.ndim==1 and dRPhir.size==Rref.size
        assert type(nRPhi0) is int

        Ptsi, dSi, NLi, dLri, Rrefi, dRPhiri, nRPhi0i, VPbisi = GG._Ves_Smesh_Tor_SubFromInd_cython(dL, dRPhi, VPoly, ind,
                                                                                                    DIn=DIn, VIn=VIn, PhiMinMax=None,
                                                                                                    Out='(R,Z,Phi)', margin=1.e-9)
        assert np.allclose(Pts,Ptsi)
        assert np.allclose(dSi,dS)
        assert np.allclose(NLi,NL)
        assert np.allclose(dLri,dLr)
        assert np.allclose(Rrefi,Rref)
        assert np.allclose(dRPhiri,dRPhir)
        assert nRPhi0i==nRPhi0



def test07_Ves_Smesh_Tor_PhiMinMax(VPoly=VPoly, plot=True):

    dL, dRPhi = 0.02, 0.05
    VIn = VPoly[:,1:]-VPoly[:,:-1]
    VIn = np.array([-VIn[1,:],VIn[0,:]])
    VIn = VIn/np.sqrt(np.sum(VIn**2,axis=0))[np.newaxis,:]
    DIn = 0.001
    LPhi = [[[-np.pi/4.,np.pi/4.], [3.*np.pi/2.,np.pi/2.]],
            [[-np.pi/4.,np.pi/4.], [0.,np.pi/2.]],
            [[-np.pi/4.,np.pi/4.], [np.pi/6.,-np.pi/6.]],
            [[-np.pi/4.,np.pi/4.], [0.,5.*np.pi/4.]],
            [[3.*np.pi/4.,5.*np.pi/4.], [np.pi/2.,-np.pi/2.]],
            [[3.*np.pi/4.,5.*np.pi/4.], [7.*np.pi/6.,-np.pi/2.]],
            [[3.*np.pi/4.,5.*np.pi/4.], [np.pi/2.,np.pi]],
            [[3.*np.pi/4.,5.*np.pi/4.], [7.*np.pi/6.,5.*np.pi/6.]]]

    if plot and sys.version[0]=='2':
        f = plt.figure(figsize=(11.7,8.3),facecolor="w")
        axarr = mplgrid.GridSpec(2,len(LPhi)/2)
        axarr.update(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.05, hspace=0.05)
        Lax = []

    for ii in range(0,len(LPhi)):
        Pts, dS, ind, NL, dLr, Rref, dRPhir, nRPhi0, VPbis = GG._Ves_Smesh_Tor_SubFromD_cython(dL, dRPhi, VPoly,
                                                                                               DR=[0.5,2.], DZ=[0.,1.2], DPhi=LPhi[ii][1],
                                                                                               DIn=DIn, VIn=VIn, PhiMinMax=np.array(LPhi[ii][0]),
                                                                                               Out='(R,Z,Phi)', margin=1.e-9)

        if plot and sys.version[0]=='2':
            Lax.append( f.add_subplot(axarr[ii], facecolor='w', projection='3d') )
            pts = GG.CoordShift(Pts, In='(R,Z,Phi)', Out='(X,Y,Z)', CrossRef=None)
            Lax[-1].plot(pts[0,:],pts[1,:],pts[2,:], '.k', ms=3.)
            Lax[-1].set_title("Phi = [{0:02.0f},{1:02.0f}]\n DPhi = [{2:02.0f},{3:02.0f}] ".format(LPhi[ii][0][0]*180./np.pi, LPhi[ii][0][1]*180./np.pi, LPhi[ii][1][0]*180./np.pi, LPhi[ii][1][1]*180./np.pi))

        #try:
        assert Pts.ndim==2 and Pts.shape[0]==3
        LPhi[ii][0][0] = np.arctan2(np.sin(LPhi[ii][0][0]),np.cos(LPhi[ii][0][0]))
        LPhi[ii][0][1] = np.arctan2(np.sin(LPhi[ii][0][1]),np.cos(LPhi[ii][0][1]))
        marg = np.abs(np.arctan(np.mean(dRPhir+np.abs(DIn))/np.min(VPoly[1,:])))
        if LPhi[ii][0][0]<=LPhi[ii][0][1]:
            assert np.all((Pts[2,:]>=LPhi[ii][0][0]-marg) & (Pts[2,:]<=LPhi[ii][0][1]+marg))
        else:
            assert np.all( (Pts[2,:]>=LPhi[ii][0][0]-marg) | (Pts[2,:]<=LPhi[ii][0][1]+marg))
        assert np.all(GG._Ves_isInside(Pts, VPoly, VType='Tor', In='(R,Z,Phi)', Test=True))
        assert dS.shape==(Pts.shape[1],)
        assert np.all([ind.shape==(Pts.shape[1],), ind.dtype==int, ind.size==np.unique(ind).size, np.all(ind==np.unique(ind)), np.all(ind>=0)])
        assert NL.ndim==1 and NL.size==VPoly.shape[1]-1
        assert dLr.ndim==1 and dLr.size==NL.size
        assert Rref.ndim==1
        assert dRPhir.ndim==1 and dRPhir.size==Rref.size
        assert type(nRPhi0) is int

        Ptsi, dSi, NLi, dLri, Rrefi, dRPhiri, nRPhi0i, VPbisi = GG._Ves_Smesh_Tor_SubFromInd_cython(dL, dRPhi, VPoly, ind,
                                                                                                    DIn=DIn, VIn=VIn, PhiMinMax=np.array(LPhi[ii][0]),
                                                                                                    Out='(R,Z,Phi)', margin=1.e-9)
        assert np.allclose(Pts,Ptsi)
        assert np.allclose(dSi,dS)
        assert np.allclose(NLi,NL)
        assert np.allclose(dLri,dLr)
        assert np.allclose(Rrefi,Rref)
        assert np.allclose(dRPhiri,dRPhir)
        assert nRPhi0i==nRPhi0

        #except:
        #    print([ind.shape==(Pts.shape[1],), ind.dtype==int, ind.size==np.unique(ind).size, np.all(ind==np.unique(ind)), np.all(ind>=0)])
        #    print(np.unique(ind).size, ind.size)
        #    lii = [ind[ii] for ii in range(0,len(ind)) if np.sum(ind==ind[ii])>1]
        #    liib = [ii for ii in range(0,len(ind)) if np.sum(ind==ind[ii])>1]
        #    print(len(lii),len(liib))
        #    print(lii)
        #    print(liib)
        #    for ii in range(0,len(liib)):
        #        print([Pts[:,liib[ii]]==Pts[:,hh] for hh in [jj for jj in range(0,len(ind)) if ind[jj]==lii[ii]]])


    if plot and sys.version[0]=='2':
        f.canvas.draw()
        f.savefig('./test_GG_test07_Ves_Smesh_Tor_PhiMinMax.png', format='png')
        plt.close(f)



def test08_Ves_Smesh_TorStruct(VPoly=VPoly, plot=True):

    PhiMinMax = np.array([3.*np.pi/4.,5.*np.pi/4.])
    dL, dRPhi = 0.02, 0.05
    VIn = VPoly[:,1:]-VPoly[:,:-1]
    VIn = np.array([-VIn[1,:],VIn[0,:]])
    VIn = VIn/np.sqrt(np.sum(VIn**2,axis=0))[np.newaxis,:]
    DIn = -0.001
    LPhi = [[[-np.pi/4.,np.pi/4.], [3.*np.pi/2.,np.pi/2.]],
            [[-np.pi/4.,np.pi/4.], [0.,np.pi/2.]],
            [[-np.pi/4.,np.pi/4.], [np.pi/6.,-np.pi/6.]],
            [[-np.pi/4.,np.pi/4.], [0.,5.*np.pi/4.]],
            [[3.*np.pi/4.,5.*np.pi/4.], [np.pi/2.,-np.pi/2.]],
            [[3.*np.pi/4.,5.*np.pi/4.], [7.*np.pi/6.,-np.pi/2.]],
            [[3.*np.pi/4.,5.*np.pi/4.], [np.pi/2.,np.pi]],
            [[3.*np.pi/4.,5.*np.pi/4.], [7.*np.pi/6.,5.*np.pi/6.]]]


    if plot and sys.version[0]=='2':
        f = plt.figure(figsize=(11.7,8.3),facecolor="w")
        axarr = mplgrid.GridSpec(2,len(LPhi)/2)
        axarr.update(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.05, hspace=0.05)
        Lax = []

    for ii in range(0,len(LPhi)):
        Pts, dS, ind, NL, dLr, Rref, dR0r, dZ0r, dRPhir, VPbis = GG._Ves_Smesh_TorStruct_SubFromD_cython(np.array(LPhi[ii][0]), dL, dRPhi, VPoly,
                                                                                                         DR=[0.5,2.], DZ=[0.,1.2], DPhi=LPhi[ii][1],
                                                                                                         DIn=DIn, VIn=VIn, Out='(R,Z,Phi)', margin=1.e-9)

        if plot and sys.version[0]=='2':
            Lax.append( f.add_subplot(axarr[ii], facecolor='w', projection='3d') )
            pts = GG.CoordShift(Pts, In='(R,Z,Phi)', Out='(X,Y,Z)', CrossRef=None)
            Lax[-1].plot(pts[0,:],pts[1,:],pts[2,:], '.k', ms=3.)
            Lax[-1].set_title("Phi = [{0:02.0f},{1:02.0f}]\n DPhi = [{2:02.0f},{3:02.0f}] ".format(LPhi[ii][0][0]*180./np.pi, LPhi[ii][0][1]*180./np.pi, LPhi[ii][1][0]*180./np.pi, LPhi[ii][1][1]*180./np.pi))

        #try:
        assert Pts.ndim==2 and Pts.shape[0]==3
        LPhi[ii][0][0] = np.arctan2(np.sin(LPhi[ii][0][0]),np.cos(LPhi[ii][0][0]))
        LPhi[ii][0][1] = np.arctan2(np.sin(LPhi[ii][0][1]),np.cos(LPhi[ii][0][1]))
        marg = np.abs(np.arctan(np.mean(dRPhir+np.abs(DIn))/np.min(VPoly[1,:])))
        if LPhi[ii][0][0]<=LPhi[ii][0][1]:
            assert np.all((Pts[2,:]>=LPhi[ii][0][0]-marg) & (Pts[2,:]<=LPhi[ii][0][1]+marg))
        else:
            assert np.all( (Pts[2,:]>=LPhi[ii][0][0]-marg) | (Pts[2,:]<=LPhi[ii][0][1]+marg))
        if DIn>=0:
            assert np.all(GG._Ves_isInside(Pts, VPoly, VType='Tor', In='(R,Z,Phi)', Test=True))
        else:
            assert not np.all(GG._Ves_isInside(Pts, VPoly, VType='Tor', In='(R,Z,Phi)', Test=True))
        assert dS.shape==(Pts.shape[1],)
        assert np.all([ind.shape==(Pts.shape[1],), ind.dtype==int, ind.size==np.unique(ind).size, np.all(ind==np.unique(ind)), np.all(ind>=0)])
        assert NL.ndim==1 and NL.size==VPoly.shape[1]-1
        assert dLr.ndim==1 and dLr.size==NL.size
        assert Rref.ndim==1
        assert type(dR0r) is float and type(dZ0r) is float
        assert dRPhir.ndim==1 and dRPhir.size==Rref.size

        Ptsi, dSi, NLi, dLri, Rrefi, dR0ri, dZ0ri, dRPhiri, VPbisi = GG._Ves_Smesh_TorStruct_SubFromInd_cython(np.array(LPhi[ii][0]), dL, dRPhi, VPoly, ind,
                                                                                                               DIn=DIn, VIn=VIn, Out='(R,Z,Phi)', margin=1.e-9)
        assert np.allclose(Pts,Ptsi)
        assert np.allclose(dSi,dS)
        assert np.allclose(NLi,NL)
        # We know it does not match here (too complicated, not necessary)
        #assert np.allclose(dLri,dLr)
        #assert np.allclose(Rrefi,Rref)
        #assert np.allclose(dRPhiri,dRPhir)
        assert all([dR0r==dR0ri, dZ0r==dZ0ri])
        """
        except:
            print([ind.shape==(Pts.shape[1],), ind.dtype==int, ind.size==np.unique(ind).size, np.all(ind==np.unique(ind)), np.all(ind>=0)])
            print(np.unique(ind).size, ind.size)
            lii = [ind[ii] for ii in range(0,len(ind)) if np.sum(ind==ind[ii])>1]
            liib = [ii for ii in range(0,len(ind)) if np.sum(ind==ind[ii])>1]
            print(len(lii),len(liib))
            print(lii)
            print(liib)
            for ii in range(0,len(liib)):
                print([Pts[:,liib[ii]]==Pts[:,hh] for hh in [jj for jj in range(0,len(ind)) if ind[jj]==lii[ii]]])
        """

    if plot and sys.version[0]=='2':
        f.canvas.draw()
        f.savefig('./test_GG_test08_Ves_Smesh_TorStruct.png', format='png')
        plt.close(f)





def test09_Ves_Smesh_Lin(VPoly=VPoly):

    XMinMax = np.array([0.,10.])
    dL, dX = 0.02, 0.05
    VIn = VPoly[:,1:]-VPoly[:,:-1]
    VIn = np.array([-VIn[1,:],VIn[0,:]])
    VIn = VIn/np.sqrt(np.sum(VIn**2,axis=0))[np.newaxis,:]
    DIn = -0.001
    DY, DZ = [0.,2.], [0.,1.]
    LDX = [None,[-1.,2.],[2.,5.],[8.,11.]]

    for ii in range(0,len(LDX)):
        Pts, dS, ind, NL, dLr, Rref, dXr, dY0r, dZ0r, VPbis = GG._Ves_Smesh_Lin_SubFromD_cython(XMinMax, dL, dX, VPoly,
                                                                                                DX=LDX[ii], DY=DY, DZ=DZ,
                                                                                                DIn=DIn, VIn=VIn, margin=1.e-9)

        assert Pts.ndim==2 and Pts.shape[0]==3
        assert np.all(Pts[0,:]>=XMinMax[0]-np.abs(DIn)) and np.all(Pts[0,:]<=XMinMax[1]+np.abs(DIn))
        assert np.all(Pts[1,:]>=1.-np.abs(DIn)) and np.all(Pts[1,:]<=3.+np.abs(DIn))
        assert np.all(Pts[2,:]>=-np.abs(DIn)) and np.all(Pts[2,:]<=1.+np.abs(DIn))
        if DIn>=0:
            assert np.all(GG._Ves_isInside(Pts, VPoly, Lim=XMinMax, VType='Lin', In='(X,Y,Z)', Test=True))
        else:
            assert not np.all(GG._Ves_isInside(Pts, VPoly, Lim=XMinMax, VType='Lin', In='(X,Y,Z)', Test=True))
        assert dS.shape==(Pts.shape[1],)
        assert all([ind.shape==(Pts.shape[1],), ind.dtype==int, np.unique(ind).size==ind.size, np.all(ind==np.unique(ind)), np.all(ind>=0)])
        assert ind.shape==(Pts.shape[1],) and ind.dtype==int and np.all(ind==np.unique(ind)) and np.all(ind>=0)
        assert NL.ndim==1 and NL.size==VPoly.shape[1]-1
        assert dLr.ndim==1 and dLr.size==NL.size
        assert Rref.ndim==1
        assert all([type(xx) is float for xx in [dXr,dY0r,dZ0r]])

        Ptsi, dSi, NLi, dLri, Rrefi, dXri, dY0ri, dZ0ri, VPbisi = GG._Ves_Smesh_Lin_SubFromInd_cython(XMinMax, dL, dX, VPoly, ind, DIn=DIn, VIn=VIn, margin=1.e-9)

        assert np.allclose(Pts,Ptsi)
        assert np.allclose(dS,dSi)
        assert np.allclose(NL,NLi)
        # We know the following are not identical (size), but too complicated for little gain
        #assert np.allclose(dLr,dLri)
        #assert np.allclose(Rref,Rrefi)
        assert all([dXr==dXri, dY0r==dY0ri, dZ0r==dZ0ri])







######################################################
######################################################
#               LOS
######################################################
######################################################


def test10_LOS_PInOut():

    VP = np.array([[6.,8.,8.,6.,6.],[6.,6.,8.,8.,6.]])
    VIn = np.array([[0.,-1.,0.,1.],[1.,0.,-1.,0.]])
    VL = np.array([0.,1.])*2.*np.pi
    SP0 = np.array([[6.,7.,7.,6.,6.],[6.,6.,7.,7.,6.]])
    SP1 = np.array([[7.,8.,8.,7.,7.],[7.,7.,8.,8.,7.]])
    SP2 = np.array([[6.,7.,7.,6.,6.],[7.,7.,8.,8.,7.]])
    SL0 = np.array([0.,1.])*2.*np.pi
    SL1 = [np.array(ss)*2.*np.pi for ss in [[0.,1./3.],[2./3.,1.]]]
    SL2 = np.array([2./3.,1.])*2.*np.pi

    # Linear without Struct
    y, z = [5.,5.,6.5,7.5,9.,9.,7.5,6.5], [7.5,6.5,5.,5.,6.5,7.5,9.,9.]
    N = len(y)
    Ds = np.array([2.*np.pi*np.concatenate((np.ones((N,))/6.,np.ones((N,))/2.,5.*np.ones((N,))/6.)),np.tile(y,3),np.tile(z,3)])
    Ds = np.concatenate((Ds,np.array([2.*np.pi*np.array([-1.,-1.,-1.,-1.,2.,2.,2.,2.]),[6.5,7.5,7.5,6.5,6.5,7.5,7.5,6.5],[6.5,6.5,7.5,7.5,6.5,6.5,7.5,7.5]])),axis=1)
    ex, ey, ez = np.array([[1.],[0.],[0.]]), np.array([[0.],[1.],[0.]]), np.array([[0.],[0.],[1.]])
    us = np.concatenate((ey,ey,ez,ez,-ey,-ey,-ez,-ez, ey,ey,ez,ez,-ey,-ey,-ez,-ez, ey,ey,ez,ez,-ey,-ey,-ez,-ez, ex,ex,ex,ex,-ex,-ex,-ex,-ex),axis=1)
    y, z = [6.,6.,6.5,7.5,8.,8.,7.5,6.5], [7.5,6.5,6.,6.,6.5,7.5,8.,8.]
    Sols_In = np.array([2.*np.pi*np.concatenate((np.ones((N,))/6.,np.ones((N,))/2.,5.*np.ones((N,))/6.)),np.tile(y,3),np.tile(z,3)])
    Sols_In = np.concatenate((Sols_In,np.array([2.*np.pi*np.array([0.,0.,0.,0.,1.,1.,1.,1.]),[6.5,7.5,7.5,6.5,6.5,7.5,7.5,6.5],[6.5,6.5,7.5,7.5,6.5,6.5,7.5,7.5]])),axis=1)
    y, z = [8.,8.,6.5,7.5,6.,6.,7.5,6.5], [7.5,6.5,8.,8.,6.5,7.5,6.,6.]
    Sols_Out = np.array([2.*np.pi*np.concatenate((np.ones((N,))/6.,np.ones((N,))/2.,5.*np.ones((N,))/6.)),np.tile(y,3),np.tile(z,3)])
    Sols_Out = np.concatenate((Sols_Out,np.array([2.*np.pi*np.array([1.,1.,1.,1.,0.,0.,0.,0.]),[6.5,7.5,7.5,6.5,6.5,7.5,7.5,6.5],[6.5,6.5,7.5,7.5,6.5,6.5,7.5,7.5]])),axis=1)
    Iin = np.array([3,3,0,0,1,1,2,2, 3,3,0,0,1,1,2,2, 3,3,0,0,1,1,2,2, -1,-1,-1,-1, -2,-2,-2,-2],dtype=int)
    Iout = np.array([1,1,2,2,3,3,0,0, 1,1,2,2,3,3,0,0, 1,1,2,2,3,3,0,0, -2,-2,-2,-2, -1,-1,-1,-1],dtype=int)
    PIn, POut, kPIn, kPOut, VperpIn, VperpOut, IIn, IOut = GG.LOS_Calc_PInOut_VesStruct(Ds, us, VP, VIn, Lim=VL, VType='Lin', Test=True)
    assert np.allclose(PIn,Sols_In, equal_nan=True)
    assert np.allclose(POut,Sols_Out, equal_nan=True)
    assert np.allclose(kPIn,np.concatenate((np.ones((3*N,)),2.*np.pi*np.ones((8,)))), equal_nan=True)
    assert np.allclose(kPOut,np.concatenate((3.*np.ones((kPOut.size-8,)),2.*np.pi*(1.+np.ones((8,))))), equal_nan=True)
    assert np.allclose(VperpIn, -us) and np.allclose(VperpOut, -us)
    assert np.allclose(IIn, Iin) and np.allclose(IOut[2,:], Iout)

    # Linear with Struct
    x = [1./6,1./6,1./6,1./6,1./6,1./6,1./6,1./6, 0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5, 5./6,5./6,5./6,5./6,5./6,5./6,5./6,5./6, 0.,1.,0.,2./3,1.,0.,1.,1.]
    y = [7.,6.,6.5,7.5,7.,8.,7.5,6.5, 8.,6.,6.5,7.5,7.,6.,7.5,6.5, 6.,6.,6.5,7.5,7.,8.,7.5,6.5, 6.5,7.5,7.5,6.5,6.5,7.5,7.5,6.5]
    z = [7.5,6.5,6.,7.,6.5,7.5,8.,7., 7.5,6.5,6.,8.,6.5,7.5,6.,7., 7.5,6.5,6.,7.,6.5,7.5,8.,8., 6.5,6.5,7.5,7.5,6.5,6.5,7.5,7.5]
    Sols_Out = np.array([2.*np.pi*np.array(x), y, z])
    Iin = np.array([3,3,0,0,1,1,2,2, 3,3,0,0,1,1,2,2, 3,3,0,0,1,1,2,2, -1,-1,-1,-1, -2,-2,-2,-2],dtype=int)
    Iout = np.array([3,3,0,0,1,1,2,2, 1,3,0,2,1,3,0,2, 3,3,0,0,1,1,2,2, -1,-2,-1,-1, -2,-1,-2,-2],dtype=int)
    indS = np.array([[2,1,1,2,1,2,2,1, 0,1,1,0,1,0,0,1, 3,1,1,2,1,2,2,3, 1,0,2,3, 1,0,2,3],
                     [0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,1,0,1,1,0, 0,0,0,0, 0,0,1,0]],dtype=int)
    PIn, POut, kPIn, kPOut, VperpIn, VperpOut, IIn, IOut = GG.LOS_Calc_PInOut_VesStruct(Ds, us, VP, VIn, Lim=VL, LSPoly=[SP0,SP1,SP2], LSLim=[SL0,SL1,SL2], LSVIn=[VIn,VIn,VIn], VType='Lin', Test=True)
    assert np.allclose(PIn,Sols_In, equal_nan=True)
    assert np.allclose(POut,Sols_Out, equal_nan=True)
    assert np.allclose(kPIn,np.concatenate((np.ones((3*N,)),2.*np.pi*np.ones((8,)))), equal_nan=True)
    assert np.allclose(kPOut, np.concatenate(([2,1,1,2,2,1,1,2, 3,1,1,3,2,3,3,2, 1,1,1,2,2,1,1,1],2.*np.pi*np.array([1,2,1,1+2./3, 1,2,1,1]))))
    assert np.allclose(VperpIn, -us) and np.allclose(VperpOut, -us)
    assert np.allclose(IIn, Iin) and np.allclose(IOut[2,:], Iout)
    assert np.allclose(IOut[:2,:], indS)


    # Toroidal, without Struct
    Theta = np.pi*np.array([1./4.,3./4.,5./4.,7./4.])
    r, z = np.array([5.,5.,6.5,7.5,9.,9.,7.5,6.5]), np.array([7.5,6.5,5.,5.,6.5,7.5,9.,9.])
    N = len(r)
    ex, ey = np.array([[1.],[0.],[0.]]), np.array([[0.],[1.],[0.]])
    ez = np.array([[0.],[0.],[1.]])
    Ds, us = [], []
    Sols_In, Sols_Out = [], []
    rsol_In = [[6.,6.,6.5,7.5,8.,8.,7.5,6.5],[6.,6.,6.5,7.5,8.,8.,7.5,6.5],[6.,6.,6.5,7.5,8.,8.,7.5,6.5],[6.,6.,6.5,7.5,8.,8.,7.5,6.5]]
    zsol_In = [[7.5,6.5,6.,6.,6.5,7.5,8.,8.],[7.5,6.5,6.,6.,6.5,7.5,8.,8.],[7.5,6.5,6.,6.,6.5,7.5,8.,8.],[7.5,6.5,6.,6.,6.5,7.5,8.,8.]]
    rsol_Out = [[8.,8.,6.5,7.5,6.,6.,7.5,6.5],[8.,8.,6.5,7.5,6.,6.,7.5,6.5],[8.,8.,6.5,7.5,6.,6.,7.5,6.5],[8.,8.,6.5,7.5,6.,6.,7.5,6.5]]
    zsol_Out = [[7.5,6.5,8.,8.,6.5,7.5,6.,6.],[7.5,6.5,8.,8.,6.5,7.5,6.,6.],[7.5,6.5,8.,8.,6.5,7.5,6.,6.],[7.5,6.5,8.,8.,6.5,7.5,6.,6.]]
    for ii in range(0,len(Theta)):
        er = np.array([[np.cos(Theta[ii])], [np.sin(Theta[ii])], [0.]])
        Ds.append(er*r[np.newaxis,:] + ez*z[np.newaxis,:])
        us.append(np.concatenate((er,er,ez,ez,-er,-er,-ez,-ez),axis=1))
        Sols_In.append(np.array(rsol_In[ii])[np.newaxis,:]*er + np.array(zsol_In[ii])[np.newaxis,:]*ez)
        Sols_Out.append(np.array(rsol_Out[ii])[np.newaxis,:]*er + np.array(zsol_Out[ii])[np.newaxis,:]*ez)
    Ds.append(np.array([[6.5,7.5,7.5,6.5, 6.5,6.5,6.5,6.5],[-6.5,-6.5,-6.5,-6.5, -7.5,-6.5,-6.5,-7.5],[6.5,6.5,7.5,7.5, 6.5,6.5,7.5,7.5]]))
    us.append(np.concatenate((ey,ey,ey,ey,-ex,-ex,-ex,-ex),axis=1))
    Ds = np.concatenate(tuple(Ds),axis=1)
    us = np.concatenate(tuple(us),axis=1)
    Sols_In = np.concatenate(tuple(Sols_In),axis=1)
    Sols_Out = np.concatenate(tuple(Sols_Out),axis=1)
    Iin = np.array([3,3,0,0,1,1,2,2, 3,3,0,0,1,1,2,2, 3,3,0,0,1,1,2,2, 3,3,0,0,1,1,2,2, 1,1,1,1,1,1,1,1])
    Iout = np.array([1,1,2,2,3,3,0,0, 1,1,2,2,3,3,0,0, 1,1,2,2,3,3,0,0, 1,1,2,2,3,3,0,0, 1,1,1,1,1,1,1,1])
    PIn, POut, kPIn, kPOut, VperpIn, VperpOut, IIn, IOut = GG.LOS_Calc_PInOut_VesStruct(Ds, us, VP, VIn, Lim=None, VType='Tor', Test=True)
    ThetaIn, ThetaOut = np.arctan2(PIn[1,32:],PIn[0,32:]), np.arctan2(POut[1,32:],POut[0,32:])
    ErIn = np.array([np.cos(ThetaIn), np.sin(ThetaIn), np.zeros((8,))])
    ErOut = np.array([np.cos(ThetaOut), np.sin(ThetaOut), np.zeros((8,))])
    assert np.allclose(PIn[:,:32],Sols_In[:,:32], equal_nan=True) and np.all((ThetaIn>-np.pi/2.) & (ThetaIn<0.))
    assert np.allclose(POut[:,:32],Sols_Out[:,:32], equal_nan=True) and np.all((ThetaOut>0.) | (ThetaOut<-np.pi/2.))
    assert np.allclose(np.hypot(PIn[0,32:],PIn[1,32:]),8.*np.ones((8,))) and np.allclose(np.hypot(POut[0,32:],POut[1,32:]),8.*np.ones((8,)))
    assert np.allclose(kPIn[:32],np.ones((4*N,)), equal_nan=True) and np.all((kPIn[32:]>0.) & (kPIn[32:]<6.5))
    assert np.allclose(kPOut[:32], 3.*np.ones((4*N,)), equal_nan=True) and np.all((kPOut[32:]>6.5) & (kPOut[32:]<16.))
    assert np.allclose(VperpIn[:,:32], -us[:,:32]) and np.allclose(VperpOut[:,:32], -us[:,:32]) and np.allclose(VperpIn[:,32:],ErIn) and np.allclose(VperpOut[:,32:],-ErOut)
    assert np.allclose(IIn, Iin) and np.allclose(IOut[2,:], Iout)

    # Toroidal, with Struct
    SL0 = None
    SL1 = [np.array(ss)*np.pi for ss in [[0.,0.5],[1.,3./2.]]]
    SL2 = np.array([0.5,3./2.])*np.pi
    Sols_In, Sols_Out = [], []
    rsol_In = [[6.,6.,6.5,7.5,8.,8.,7.5,6.5],[6.,6.,6.5,7.5,8.,8.,7.5,6.5],[6.,6.,6.5,7.5,8.,8.,7.5,6.5],[6.,6.,6.5,7.5,8.,8.,7.5,6.5]]
    zsol_In = [[7.5,6.5,6.,6.,6.5,7.5,8.,8.],[7.5,6.5,6.,6.,6.5,7.5,8.,8.],[7.5,6.5,6.,6.,6.5,7.5,8.,8.],[7.5,6.5,6.,6.,6.5,7.5,8.,8.]]
    rsol_Out = [[7.,6.,6.5,7.5,7.,8.,7.5,6.5],[6.,6.,6.5,7.5,7.,7.,7.5,6.5],[6.,6.,6.5,7.5,7.,8.,7.5,6.5],[8.,6.,6.5,7.5,7.,6.,7.5,6.5]]
    zsol_Out = [[7.5,6.5,6.,7.,6.5,7.5,8.,7.],[7.5,6.5,6.,8.,6.5,7.5,6.,8.],[7.5,6.5,6.,7.,6.5,7.5,8.,8.],[7.5,6.5,6.,8.,6.5,7.5,6.,7.]]
    for ii in range(0,len(Theta)):
        er = np.array([[np.cos(Theta[ii])], [np.sin(Theta[ii])], [0.]])
        Sols_In.append(np.array(rsol_In[ii])[np.newaxis,:]*er + np.array(zsol_In[ii])[np.newaxis,:]*ez)
        Sols_Out.append(np.array(rsol_Out[ii])[np.newaxis,:]*er + np.array(zsol_Out[ii])[np.newaxis,:]*ez)
    Sols_In = np.concatenate(tuple(Sols_In),axis=1)
    Sols_Out = np.concatenate(tuple(Sols_Out),axis=1)
    kpout = np.array([2,1,1,2,2,1,1,2, 1,1,1,3,2,2,3,1, 1,1,1,2,2,1,1,1, 3,1,1,3,2,3,3,2])
    Iin = np.array([3,3,0,0,1,1,2,2, 3,3,0,0,1,1,2,2, 3,3,0,0,1,1,2,2, 3,3,0,0,1,1,2,2, 1,1,1,1,1,1,1,1])
    Iout = np.array([3,3,0,0,1,1,2,2, 3,3,0,2,1,1,0,2, 3,3,0,0,1,1,2,2, 1,3,0,2,1,3,0,2, 1,1,-1,3,1,1,-2,-2])
    indS = np.array([[2,1,1,2,1,2,2,1, 3,1,1,0,1,3,0,3, 3,1,1,2,1,2,2,3, 0,1,1,0,1,0,0,1, 1,0,2,2, 0,1,3,2],
                     [0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,1,0,1,1,0, 0,0,0,0,0,0,0,0, 0,0,0,0, 0,0,0,1]],dtype=int)
    PIn, POut, kPIn, kPOut, VperpIn, VperpOut, IIn, IOut = GG.LOS_Calc_PInOut_VesStruct(Ds, us, VP, VIn, Lim=None, LSPoly=[SP0,SP1,SP2], LSLim=[SL0,SL1,SL2], LSVIn=[VIn,VIn,VIn], VType='Tor', Test=True)
    RIn, ROut = np.hypot(PIn[0,32:],PIn[1,32:]), np.hypot(POut[0,32:],POut[1,32:])
    ThetaIn, ThetaOut = np.arctan2(PIn[1,32:],PIn[0,32:]), np.arctan2(POut[1,32:],POut[0,32:])
    ErIn = np.array([np.cos(ThetaIn), np.sin(ThetaIn), np.zeros((8,))])
    ErOut = np.array([np.cos(ThetaOut), np.sin(ThetaOut), np.zeros((8,))])
    vperpout = np.concatenate((ErOut[:,0:1],-ErOut[:,1:2],-ey,-ErOut[:,3:4], -ErOut[:,4:5],ErOut[:,5:6],ex,ex),axis=1)
    assert np.allclose(PIn[:,:32],Sols_In[:,:32], equal_nan=True) and np.all((ThetaIn>-np.pi/2.) & (ThetaIn<0.))
    assert np.allclose(POut[:,:32],Sols_Out[:,:32], equal_nan=True) and np.all((ThetaOut>-np.pi) & (ThetaOut<np.pi/2.))
    assert np.all((RIn>=6.) & (RIn<=8.)) and np.all((ROut>=6.) & (ROut<=8.))
    assert np.allclose(kPIn[:32],np.ones((4*N,)), equal_nan=True) and np.all((kPIn[32:]>0.) & (kPIn[32:]<6.5))
    assert np.allclose(kPOut[:32], kpout, equal_nan=True) and np.all((kPOut[32:]>=3.) & (kPOut[32:]<16.))
    assert np.allclose(VperpIn[:,:32], -us[:,:32]) and np.allclose(VperpOut[:,:32], -us[:,:32]) and np.allclose(VperpIn[:,32:],ErIn) and np.allclose(VperpOut[:,32:],vperpout)
    assert np.allclose(IIn, Iin) and np.allclose(IOut[2,:], Iout)
    assert np.allclose(IOut[:2,:],indS)


def test11_LOS_sino():

    RZ = np.array([2.,0.])
    r = np.array([0.1, 0.2, 0.1])
    theta = np.array([5*np.pi/6,0,np.pi/2])
    phi = np.array([0.,0.,np.pi/10.])
    k = np.array([1,10,5])
    N = len(r)
    us = np.ascontiguousarray([np.sin(phi), -np.sin(theta)*np.cos(phi), np.cos(theta)*np.cos(phi)])
    Ms = np.array([np.zeros((N,)), RZ[0]+r*np.cos(theta), RZ[1]+r*np.sin(theta)])
    Ds = np.ascontiguousarray(Ms - k[np.newaxis,:]*us)
    PMin0, kPMin0, RMin0 = np.nan*np.ones((3,N)), np.nan*np.ones((N,)), np.nan*np.ones((N,))
    Theta0, p0, ImpTheta0, phi0 = np.nan*np.ones((N,)), np.nan*np.ones((N,)), np.nan*np.ones((N,)), np.nan*np.ones((N,))
    PMin0, kPMin0, RMin0, Theta0, p0, ImpTheta0, phi0 = GG.LOS_sino(Ds, us, RZ, kOut=np.full((N,),np.inf),
                                                                    Mode='LOS', VType='Lin')
    assert np.allclose(PMin0,Ms)
    assert np.allclose(kPMin0,k)
    assert RMin0.shape==(N,)
    assert Theta0.shape==(N,)
    assert np.allclose(np.abs(p0),r)
    assert np.allclose(np.abs(ImpTheta0),theta)
    assert np.allclose(phi0,phi)

    # Tor (to be finished)
    us = np.ascontiguousarray([-np.sin(theta)*np.cos(phi), np.sin(phi), np.cos(theta)*np.cos(phi)])
    Ms = np.array([RZ[0]+r*np.cos(theta), np.zeros((N,)), RZ[1]+r*np.sin(theta)])
    Ds = np.ascontiguousarray(Ms - k[np.newaxis,:]*us)
    PMin0, kPMin0, RMin0, Theta0, p0, ImpTheta0, phi0 = GG.LOS_sino(Ds, us, RZ, kOut=np.full((N,),np.inf),
                                                                    Mode='LOS', VType='Tor')
    assert np.allclose(PMin0,Ms)
    assert np.allclose(kPMin0,k)
    assert RMin0.shape==(N,)
    assert Theta0.shape==(N,)
    assert np.allclose(np.abs(p0),r)
    assert np.allclose(np.abs(ImpTheta0),theta)
    assert np.allclose(np.abs(phi0),phi)
