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
import tofu.Eq._compute as _tfEq_c


Root = tfpf.Find_Rootpath()
Addpath = '/tests/test03_Eq/'


#######################################################
#
#     Setup and Teardown
#
#######################################################

def setup_module(module):
    print ("") # this is to get a newline after the dots
    print ("--------------------------------------------")
    print ("--------------------------------------------")
    print ("        test01_compute")
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






############################################
#####     Computing functions for Eq2D
############################################



def test01_correctRef():
    Ref = 'vol'
    MagAx = np.array([[1.7,0.],[1.7,0.],[1.7,0.]])
    PtsCross = np.array([[1.5,1.6,1.7,1.8,1.9],[0.,0.,0.,0.,0.]])
    Nt, NP = MagAx.shape[0], PtsCross.shape[1]
    RefV0 = np.array([[1.,1.,1.,1.,1.],[1.,1.,1.,1.,1.],[1.,1.,1.,1.,1.]])
    RefV0[:,1:3] = np.nan
    RefV1 = _tfEq_c._correctRef(Ref, RefV0, Nt, PtsCross, MagAx)
    assert np.all(RefV1-np.array([[1.,1.,0.,1.,1.],[1.,1.,0.,1.,1.],[1.,1.,0.,1.,1.]])<1.e-14)



def test02_interp_Quant():
    Tab_t = np.array([1.,2.,3.])
    Tab_Pts = np.array([[1.5,1.6,1.7,1.8,1.5,1.6,1.7,1.8,1.5,1.6,1.7,1.8,1.5,1.6,1.7,1.8],[-1.,-1.,-1.,-1.,0.,0.,0.,0.,1.,1.,1.,1.,2.,2.,2.,2.]])
    Tab_vPts = np.array([[1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.],[1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.],[1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.]])
    Pts = np.array([[1.55,1.75,1.85],[-0.5,0.,0.5]])
    Nt, NP = Tab_t.size, Pts.shape[1]

    LQ = _tfEq_c._interp_Quant(Tab_t, Tab_Pts, [Tab_vPts], Pts, LQuant='rho_p', indt=0, t=None, deg=3, Test=True)
    assert type(LQ) is np.ndarray and LQ.shape==(1,NP)
    LQ = _tfEq_c._interp_Quant(Tab_t, Tab_Pts, [Tab_vPts], Pts, LQuant=['rho_p'], indt=None, t=2., deg=3, Test=True)
    assert type(LQ) is np.ndarray and LQ.shape==(1,NP)
    LQ = _tfEq_c._interp_Quant(Tab_t, Tab_Pts, [Tab_vPts,Tab_vPts], Pts, LQuant=['rho_p','q'], indt=None, t=None, deg=3, Test=True)
    assert type(LQ) is dict and all([ss in LQ.keys() for ss in ['rho_p','q']]) and all([type(LQ[ss]) is np.ndarray and LQ[ss].shape==(Nt,NP) for ss in LQ.keys()])



def test03_get_rgrad():
    Tab_t = np.array([1.,2.,3.])
    Tab_Pts = np.array([[1.5,1.6,1.7,1.8,1.5,1.6,1.7,1.8,1.5,1.6,1.7,1.8,1.5,1.6,1.7,1.8],[-1.,-1.,-1.,-1.,0.,0.,0.,0.,1.,1.,1.,1.,2.,2.,2.,2.]])
    Tab_vPts = np.array([[1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.],[1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.],[1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.]])
    Pts = np.array([[1.55,1.75,1.85],[-0.5,0.,0.5]])
    Nt, NP = Tab_t.size, Pts.shape[1]

    rgrad = _tfEq_c._get_rgrad(Pts, Tab_Pts, Tab_vPts, Tab_t, indt=0, t=None, Test=True)
    assert type(rgrad) is np.ndarray and rgrad.shape==(2,NP)
    rgrad = _tfEq_c._get_rgrad(Pts, Tab_Pts, Tab_vPts, Tab_t, indt=None, t=2., Test=True)
    assert type(rgrad) is np.ndarray and rgrad.shape==(2,NP)
    rgrad = _tfEq_c._get_rgrad(Pts, Tab_Pts, Tab_vPts, Tab_t, indt=None, t=None, Test=True)
    assert type(rgrad) is list and all([rg.shape==(2,NP) for rg in rgrad])


















