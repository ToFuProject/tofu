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
import tofu.plugins.AUG.SXR as tfaugSXR


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
    print ("        test01_SXR")

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


class test01_geom:

    @classmethod
    def setup_class(cls, shot=30532):
        print ("")
        print "--------- "+VerbHead+cls.__name__

        cls.shot = shot

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

    def test01_get_GeomFromShot(self):
        Change, Array = tfaugSXR.geom.get_GeomFromShot( shot_init=29038, shot_end=29100, Ds=10, Chan='CamHeads', Verb=True, save=False )

        # Output format
        assert type(Change) is dict
        assert type(Array) is np.ndarray

        # Values
        assert Change.keys()==[29043]


    def test02_create(self):
        LGD = tfaugSXR.geom.create(shot=self.shot, VesName='V1', SavePathObj=None, forceshot=False, overwrite=False, save=False,
                                   CalcEtend=True, CalcSpanImp=True, CalcCone=True, CalcPreComp=False, Calc=True, Verb=True,
                                   Etend_Method='simps', Etend_RelErr=1.e-3, Etend_dX12=[0.05,0.05], Etend_dX12Mode='rel', Etend_Ratio=0.02, Colis=True, LOSRef='Cart',
                                   Cone_DRY=0.005, Cone_DXTheta=np.pi/512., Cone_DZ=0.005, Cone_NPsi=20, Cone_Nk=60)

    def test03_load(self):
        LGD1 = tfaugSXR.geom.load(Cams=['L','K1'], shot=np.inf, SavePathObj=None, sort=False)
        LGD2 = tfaugSXR.geom.load(Cams=['L','K1'], shot=np.inf, SavePathObj=None, sort=True)
        assert LGD1[0].Id.Name==LGD2[1].Id.Name





class test02_data:

    @classmethod
    def setup_class(cls, shot=30532):
        print ("")
        print "--------- "+VerbHead+cls.__name__

        cls.shot = shot

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

    def test01_load(self):
        Names = ['F_015','G_020','H_015','I_052','J_050','K_020','L_017','M_021']

        # Without Tofu output
        SXR0, t0, Names0 = tfaugSXR.data.load(shot=self.shot, Names=None, Mode='SSX', Dt=[2.,3.], NoGeom=False, Tofu=False, Verb=False, Test=True)
        SXR1, t1, Names1 = tfaugSXR.data.load(shot=self.shot, Names=Names, Mode='SSX', Dt=None, NoGeom=True, Tofu=False, Verb=False, Test=True)
        SXR2, t2, Names2 = tfaugSXR.data.load(shot=self.shot, Names=Names, Mode='SX', Join=True, Dt=[2.9,3.], tRef='fmax', Method='interp', NoGeom=True, Tofu=False, Verb=False, Test=True)
        SXR3, t3, Names3 = tfaugSXR.data.load(shot=self.shot, Names=Names, Mode='SX', Join=False, Dt=[2.9,3.], tRef='fmin', Method='interp', NoGeom=True, Tofu=False, Verb=False, Test=True)

        # Output format
        assert all([type(names) is list and all([type(ss) is str for ss in names]) for names in [Names0,Names1,Names2,Names3]])
        assert all([type(sxr) is np.ndarray and sxr.ndim==2 for sxr in [SXR0,SXR1,SXR2]])
        assert all([type(t) is np.ndarray and t.ndim==1 for t in [t0,t1,t2]])
        assert all([sxr.shape[0]==t.size for (sxr,t) in [(SXR0,t0),(SXR1,t1),(SXR2,t2)]])
        assert type(SXR3) is list and len(SXR3)==len(Names3)
        assert type(t3) is list and len(t3)==len(Names3)

        # Values
        assert all([names==Names for names in [Names1,Names2,Names3]])

        # With Tofu output
        pre0 = tfaugSXR.data.load(shot=self.shot, Names=None, Mode='SSX', Dt=[2.,3.], NoGeom=False, Tofu=True, Verb=False, Test=True)
        pre1 = tfaugSXR.data.load(shot=self.shot, Names=Names, Mode='SSX', Dt=None, NoGeom=True, Tofu=True, Verb=False, Test=True)
        pre2 = tfaugSXR.data.load(shot=self.shot, Names=Names, Mode='SX', Join=True, Dt=[2.9,3.], tRef='fmax', Method='interp', NoGeom=True, Tofu=True, Verb=False, Test=True)
        pre3 = tfaugSXR.data.load(shot=self.shot, Names=Names, Mode='SX', Join=False, Dt=[2.9,3.], tRef='fmin', Method='interp', NoGeom=True, Tofu=True, Verb=False, Test=True)

        # Output format
        assert all([pre.Chans==names for (pre,names) in [(pre0,Names0),(pre1,Names1),(pre2,Names2)]])
        assert all([np.all(pre.data==sxr) for (pre,sxr) in [(pre0,sxr0),(pre1,sxr1),(pre2,sxr2)]])
        assert all([np.all(pre.t==t) for (pre,t) in [(pre0,t0),(pre1,t1),(pre2,t2)]])








