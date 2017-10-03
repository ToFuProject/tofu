"""
This module contains tests for tofu.geom in its structured version
"""

# External modules
import os
import numpy as np
import math
import matplotlib.pyplot as plt


# Nose-specific
from nose import with_setup # optional


# Importing package tofu.geom
import tofu.defaults as tfd
import tofu.pathfile as tfpf
import tofu.mesh._bsplines_cy as _tfm_bs


Root = tfpf.Find_Rootpath()
Addpath = '/tests/tests02_mesh/'


#######################################################
#
#     Setup and Teardown
#
#######################################################

def setup_module(module):
    print ("") # this is to get a newline after the dots
    print ("--------------------------------------------")
    print ("--------------------------------------------")
    print ("        test01_bsplines")
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
#     1D bsplines
#
#######################################################



def test01_Nj(Nbx=1000, xout=0.2):

    knt = [0.,10.,20.,30.,40.,50.,60.]
    Dk=knt[-1]-knt[0]
    x = np.linspace(knt[0]-xout*Dk, knt[-1]+xout*Dk, Nbx)

    func = _tfm_bs._Nj0(knt[1],knt[2])
    ind0, ind1 = (x<knt[1])|(x>knt[2]), (x>knt[1])&(x<knt[2])
    assert hasattr(func,'__call__')
    assert np.all(func(x[ind0])==0.) and np.all(func(x[ind1])==1.)

    func, funcD1 = _tfm_bs._Nj1(knt[1],knt[2],knt[3]), _tfm_bs._Nj1D1(knt[1],knt[2],knt[3])
    ind0, ind1, ind2 = (x<knt[1])|(x>knt[3]), (x>knt[1])&(x<knt[2]), (x>knt[2])&(x<knt[3])
    assert hasattr(func,'__call__') and hasattr(funcD1,'__call__')
    assert np.all(func(x[ind0])==0.) and np.all(func(x[ind1|ind2])>=0.) and np.all(func(x[ind1|ind2])<=1.)
    assert np.all(funcD1(x[ind0])==0.) and np.all(funcD1(x[ind1])==1./(knt[2]-knt[1])) and np.all(funcD1(x[ind2])==-1./(knt[3]-knt[2]))

    func, funcD1, funcD2 = _tfm_bs._Nj2(knt[1],knt[2],knt[3],knt[4]), _tfm_bs._Nj2D1(knt[1],knt[2],knt[3],knt[4]), _tfm_bs._Nj2D2(knt[1],knt[2],knt[3],knt[4])
    ind0, ind1, ind2, ind3 = (x<knt[1])|(x>knt[4]), (x>knt[1])&(x<knt[2]), (x>knt[2])&(x<knt[3]), (x>knt[3])&(x<knt[4])
    assert hasattr(func,'__call__') and hasattr(funcD1,'__call__') and hasattr(funcD2,'__call__')
    assert np.all(func(x[ind0])==0.) and np.all(func(x[ind1|ind2|ind3])>=0.) and np.all(func(x[ind1|ind2|ind3])<=1.)
    assert np.all(funcD1(x[ind0])==0.) and np.all(funcD1(x[ind1])>=0.) and np.all(funcD1(x[ind3])<=0.)
    assert np.all(funcD2(x[ind0])==0.) and np.all(funcD2(x[ind1])>=0.) and np.all(funcD2(x[ind2])<=0.) and np.all(funcD2(x[ind3])>=0.)

    func, funcD1 = _tfm_bs._Nj3(knt[1],knt[2],knt[3],knt[4],knt[5]), _tfm_bs._Nj3D1(knt[1],knt[2],knt[3],knt[4],knt[5])
    funcD2, funcD3 = _tfm_bs._Nj3D2(knt[1],knt[2],knt[3],knt[4],knt[5]), _tfm_bs._Nj3D3(knt[1],knt[2],knt[3],knt[4],knt[5])
    ind0, ind1, ind2, ind3, ind4 = (x<knt[1])|(x>knt[5]), (x>knt[1])&(x<knt[2]), (x>knt[2])&(x<knt[3]), (x>knt[3])&(x<knt[4]), (x>knt[4])&(x<knt[5])
    assert hasattr(func,'__call__') and hasattr(funcD1,'__call__') and hasattr(funcD2,'__call__') and hasattr(funcD3,'__call__')
    assert np.all(func(x[ind0])==0.) and np.all(func(x[ind1|ind2|ind3|ind4])>=0.) and np.all(func(x[ind1|ind2|ind3|ind4])<=1.)
    assert np.all(funcD1(x[ind0])==0.) and np.all(funcD1(x[ind1|ind2])>=0.)
    assert np.all(funcD2(x[ind0])==0.) and np.all(funcD2(x[ind1|ind4])>=0.)
    assert np.all(funcD3(x[ind0])==0.) and np.all(funcD3(x[ind1|ind3])>=0.) and np.all(funcD3(x[ind2|ind4])<=0.)



def test02_BSpline_LFunc(Nbx=1000, xout=0.2):

    knt = np.asarray([0.,10.,20.,30.,40.,50.,60.])
    Dk=knt[-1]-knt[0]
    x = np.linspace(knt[0]-xout*Dk, knt[-1]+xout*Dk, Nbx)

    Deg, Modes = [0,1,2,3], ['','scp','','scp']
    Deriv = [range(0,dd+1) for dd in Deg]
    f, axarr = plt.subplots(len(Deg), max([len(dd) for dd in Deriv]))
    plt.subplots_adjust(left=0.03, bottom=0.03, right=0.99, top=0.98, wspace=0.10, hspace=0.08)
    for ii in range(0,len(Deg)):
        for jj in range(0,len(Deriv[ii])):
            LFunc, Func_Knotsind, Func_Centsind, Knots_Funcind, Cents_Funcind, MaxPos, scp_Lkntsf, scp_Lcoeff = _tfm_bs.BSpline_LFunc(Deg[ii], knt, Deriv=Deriv[ii][jj], Mode=Modes[ii], Test=True)
            assert type(LFunc) is list and all([hasattr(ff,'__call__') for ff in LFunc])
            assert all([type(aa) is np.ndarray and aa.ndim==2 for aa in [Func_Knotsind,Func_Centsind,Knots_Funcind,Cents_Funcind]])
            assert type(MaxPos) is np.ndarray and MaxPos.ndim==1
            assert scp_Lkntsf is None or (type(scp_Lkntsf) is tuple and len(scp_Lkntsf)==len(LFunc))
            assert scp_Lcoeff is None or (type(scp_Lcoeff) is tuple and len(scp_Lcoeff)==len(LFunc))
            for ff in LFunc:
                axarr[ii,jj].plot(x, ff(x))
            axarr[ii,jj].set_title(r"Deg "+str(Deg[ii])+" - Deriv"+str(Deriv[ii][jj])+"  Mode="+Modes[ii], size=8)
            axarr[ii,jj].set_xticklabels(axarr[ii,jj].get_xticks(), size=6)
            axarr[ii,jj].set_yticklabels(axarr[ii,jj].get_yticks(), size=6)
            if ii==len(Deg)-1:
                axarr[ii,jj].set_xticklabels(knt, size=6)
            else:
                axarr[ii,jj].set_xticks([])
            if jj==0:
                axarr[ii,jj].set_yticklabels(axarr[ii,jj].get_yticks(), size=6)
            else:
                axarr[ii,jj].set_yticks([])
            axarr[ii,jj].set_xlim(knt[0],knt[-1])
        for jj in range(len(Deriv[ii]),len(axarr[ii,:])):
            axarr[ii,jj].set_xticks([])
            axarr[ii,jj].set_yticks([])

    f.savefig(Root+Addpath+'tests01_bsplines_test02_BSpline_LFunc.pdf', format='pdf')



def test03_BSpline_TotFunc(Nbx=1000, xout=0.2, Mode='scp'):

    knt = np.asarray([0.,10.,20.,30.,40.,50.,60.])
    Dk=knt[-1]-knt[0]
    x = np.linspace(knt[0]-xout*Dk, knt[-1]+xout*Dk, Nbx)

    Deg, Modes = [0,1,2,3], ['','scp','','scp']
    Deriv = [['D0','D0N2','D0ME'],['D0','D0N2','D0ME','D1','D1N2','D1FI'], ['D0','D0N2','D0ME','D1','D1N2','D1FI','D2','D2N2'], ['D0','D0N2','D0ME','D1','D1N2','D1FI','D2','D2N2','D3','D3N2']]
    f, axarr = plt.subplots(len(Deg), max([len(dd) for dd in Deriv]))
    plt.subplots_adjust(left=0.03, bottom=0.03, right=0.99, top=0.98, wspace=0.10, hspace=0.08)
    for ii in range(0,len(Deg)):
        for jj in range(0,len(Deriv[ii])):
            Func = _tfm_bs.BSpline_TotFunc(Deg[ii], knt, Deriv=Deriv[ii][jj], Coefs=1., thr=1.e-8, thrmode='rel', Abs=True, Mode=Modes[ii], Test=True)
            assert hasattr(Func,'__call__')
            axarr[ii,jj].plot(x, Func(x))
            axarr[ii,jj].set_title(r"Deg "+str(Deg[ii])+" - "+Deriv[ii][jj], size=8)
            axarr[ii,jj].set_xticklabels(axarr[ii,jj].get_xticks(), size=6)
            axarr[ii,jj].set_yticklabels(axarr[ii,jj].get_yticks(), size=6)
            if ii==len(Deg)-1:
                axarr[ii,jj].set_xticklabels(knt, size=6)
            else:
                axarr[ii,jj].set_xticks([])
            if jj==0:
                axarr[ii,jj].set_yticklabels(axarr[ii,jj].get_yticks(), size=6)
            else:
                axarr[ii,jj].set_yticks([])
            axarr[ii,jj].set_xlim(knt[0],knt[-1])
        for jj in range(len(Deriv[ii]),len(axarr[ii,:])):
            axarr[ii,jj].set_xticks([])
            axarr[ii,jj].set_yticks([])

    f.savefig(Root+Addpath+'tests01_bsplines_test03_BSpline_TotFunc.pdf', format='pdf')




def test04_Calc_BF1D_Weights():
    knt = np.asarray([0.,10.,20.,30.,40.,50.,60.])
    LFunc = _tfm_bs.BSpline_LFunc(3, knt, Deriv=0, Mode='scp', Test=True)[0]
    Pts = np.array([5.,15.,35.,50.])
    Wgh = _tfm_bs.Calc_BF1D_Weights(LFunc, Pts)
    assert type(Wgh) is np.ndarray and Wgh.shape==(Pts.size,len(LFunc))


def test05_get_NptsGaussFromDegDeriv():
    N0 = _tfm_bs.get_NptsGaussFromDegDeriv(1, 'D0', 0, Mode='Surf')
    N1 = _tfm_bs.get_NptsGaussFromDegDeriv(2, 'D1N2', 1, Mode='Vol')
    N2 = _tfm_bs.get_NptsGaussFromDegDeriv(3, 'D1FI', 1, Mode='Vol')
    N3 = _tfm_bs.get_NptsGaussFromDegDeriv(3, 'D3N2', 3, Mode='Surf')
    assert all([type(n) is int for n in [N0,N1,N2,N3]])
    assert all([N0==1,N1==2,N2==4,N3==1]), str([N0,N1,N2,N3])+" instead of "+str([1,2,4,1])



def test06_get_IntQuadPts(cnp.ndarray[cnp.float64_t, ndim=1] Knots, Deg, Deriv, intDeriv, Mode='Surf', N=None):
    NKnots = 50
    Knots = np.unique(np.random.random(NKnots))+1.
    par = [(2,0,0,'Surf'), (3,'D1FI',1,'Surf'), (3,'D3N2',3,'Vol'), (2,'D2N2',2,'Surf')]
    for ii in range(0,len(par)):
        pts, w, aa, N = _tfm_bs.get_IntQuadPts(Knots, par[ii][0], par[ii][1], par[ii][2], Mode=par[ii][3], N=None)
        assert type(pts) is np.ndarray and pts.shape==(N,NKnots-1)
        assert type(w) is np.ndarray and w.shape==pts.shape
        assert type(aa) is np.ndarray and aa.shape==(NKnots-1,)
        assert type(N) is int



def test07_Calc_1D_IntVal_Quad(Coefs=1., cnp.ndarray[cnp.float64_t, ndim=1] Knots=np.linspace(0.,1000,1001), int Deg=2, Deriv=0, str Mode='Vol', LFunc=None, LFunc_Mode='scp', quad_pts=None, quad_w=None, quad_aa=None,
        thr=1.e-8, thrmode='rel', Abs=True, N=None, Test=True):

    par = []
    for ii in range(0,len(par)):
        Int = _tfm_bs.Calc_1D_IntVal_Quad(Coefs=1., Knots=np.linspace(0.,1000,1001), Deg=2, Deriv=0, Mode='Vol', LFunc=None, LFunc_Mode='scp', quad_pts=None, quad_w=None, quad_aa=None,
                                          thr=1.e-8, thrmode='rel', Abs=True, N=None, Test=True)









def test06_Calc_1D_LinIntOp():

    Deg = [0,1,2,3]
    Deriv = [0,1,2,3,'D0','D1','D2','D3','D0N2','D1N2','D2N2','D3N2']
    Knots = np.unique(np.random.random(50))+1.

    for ii in range(0,len(Deg)):
        for jj in range(0,len(Deriv)):
            intDeriv = int(Deriv[1]) if type(Deriv) is str else Deriv
            if intDeriv<=Deg[ii] and not (Deg[ii]==3 and Deriv in ['D0N2','D1N2']):
                if type(Deriv) is str and 'N' in Deriv:
                    N = int(Deg[ii]-intDeriv+1+1) if Mode=='Surf' else int(Deg[ii]-intDeriv+2+1)
                else:
                    N = int(math.ceil((Deg[ii]-intDeriv+1.)/2.)) if Mode=='Surf' else int(math.ceil((Deg[ii]-intDeriv+2)/2.))
                (pts, w, aa) = bspcy._get_Quad_GaussLegendre_1D(Knots, N)
                LFunc = _tfm_bs.BSpline_LFunc(Deg[ii], Knots, Deriv=intDeriv, Mode='scp', Test=True)[0]
                NF = len(LFunc)

                A0, m0 = _tfm_bs.Calc_1D_LinIntOp(Knots=Knots, Deg=Deg[ii], Deriv=Deriv[jj], Method='exact', Mode='Surf', LFunc=None, LFunc_Mode='scp', quad_pts=None, quad_w=None, quad_aa=None, Sparse=True, SpaFormat=None, Test=True)
                A1, m1 = _tfm_bs.Calc_1D_LinIntOp(Knots=Knots, Deg=Deg[ii], Deriv=Deriv[jj], Method='quad', Mode='Surf', LFunc=None, LFunc_Mode='', quad_pts=None, quad_w=None, quad_aa=None, Sparse=True, SpaFormat=None, Test=True)
                A2, m2 = _tfm_bs.Calc_1D_LinIntOp(Knots=Knots, Deg=Deg[ii], Deriv=Deriv[jj], Method='quad', Mode='Surf', LFunc=LFunc, LFunc_Mode='scp', quad_pts=pts, quad_w=w, quad_aa=aa, Sparse=True, SpaFormat=None, Test=True)

                assert all([type(m) is int and m in [0,1] for m in [m0,m1,m2]])
                assert all([type(A) in [np.ndarray,] and A.shape in [(NF,),(NF,NF)] for A in [A0,A1,A2]])
                assert A0.shape==A1.shape and A0.shape==A2.shape
                assert np.all(np.abs(A0-A1)<=1.e-10), "Different computations for Deg="+str(Deg[ii])+" Deriv="+str(Deriv)
                assert np.all(np.abs(A0-A2)<=1.e-10), "Different computations for Deg="+str(Deg[ii])+" Deriv="+str(Deriv)


















