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
import tofu.plugins.AUG.Eq as tfaugEq


Root = tfpf.Find_Rootpath()
Addpath = '/tests/tests10_plugins/tests01_AUG/'

VerbHead = 'tfaugEq.'


#######################################################
#
#     Setup and Teardown
#
#######################################################

def setup_module(module):
    print ("") # this is to get a newline after the dots
    print ("--------------------------------------------")
    print ("--------------------------------------------")
    print ("        test02_Eq")

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
#     Eq2D
#
#######################################################


class test01_data:

    @classmethod
    def setup_class(cls, shot=30532, t=[2.0,2.01], Quants=['MagAx','Sep','q','jp','rho_p','rho_t','surf','vol','BTX','BRY','BZ'], Diag='EQH', Ves=Root+'/tests/tests10_plugins/tests01_AUG/TFG_VesTor_AUG_Test_sh0.npz'):
        print ("")
        print "--------- "+VerbHead+cls.__name__

        MeshPar = {'CrossMesh':[0.01,0.01], 'CrossMeshMode':'abs'}
        Eq1 = tfaugEq.get_Equilibrium(shot, t, Pts='Ves', MeshPar=MeshPar, Quants=Quants, Diag=Diag, Object=True, Name='Test1', Ves=Ves, SavePath=Root+Addpath, save=False, dtime=None, dtimeIn=False, Test=True)
        Eq2 = tfaugEq.get_Equilibrium(shot, t, Pts=None, MeshPar=None, Quants=Quants, Diag=Diag, Object=False, Name='Test2', Ves=None, SavePath=Root+Addpath, save=False, dtime=None, dtimeIn=False, Test=True)
        Eq3 = tfaugEq.get_Equilibrium(shot, t, Pts='Ves', MeshPar=MeshPar, Quants=Quants, Diag=Diag, Object=False, Name='Test3', Ves=Ves, SavePath=Root+Addpath, save=False, dtime=None, dtimeIn=False, Test=True)
        cls.Obj = tfaugEq.get_Equilibrium(shot, t, Pts=None, MeshPar=None, Quants=Quants, Diag=Diag, Object=True, Name=None, Ves=None, SavePath=Root+Addpath, save=False, dtime=None, dtimeIn=False, Test=True)

    @classmethod
    def teardown_class(cls):
        #print ("teardown_class() after any methods in this class")
        pass

    def setup(self):
        #print ("TestUM:setup() before each test method")
        pass

    def teardown(self):
        #print ("TestUM:teardown() after each test method")
        pass

    def test01_saveload(self):
        self.Obj.save()
        Eq = tfpf.Open(self.Obj.Id.SavePath + self.Obj.Id.SaveName + '.npz')
        os.remove(self.Obj.Id.SavePath + self.Obj.Id.SaveName + '.npz')


















