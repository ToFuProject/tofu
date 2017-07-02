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
import tofu.treat as tft


Root = tfpf.Find_Rootpath()
Addpath = '/tests/tests02_treat/'

VerbHead = 'tft.'


#######################################################
#
#     Setup and Teardown
#
#######################################################

def setup_module(module):
    print ("") # this is to get a newline after the dots
    print ("--------------------------------------------")
    print ("--------------------------------------------")
    print ("        test01_core")

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
#     PreData class
#
#######################################################


data = np.random.rand(100000).reshape(1000,100)
t = np.linspace(0.,10.,data.shape[0])
Names = ['D{0:02.0f}'.format(ii) for ii in range(0,data.shape[1])]
SavePath = Root + Addpath

class test01_PreData:

    @classmethod

    def setup_class(cls, data=data, t=t, Names=Names, SavePath=SavePath):
        print ("")
        print("--------- "+VerbHead+cls.__name__)
        cls.Obj = tft.PreData(data, t=t, Chans=Names, Exp='Misc', Diag='Misc', shot=0, SavePath=SavePath)


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

    def test01_set_Dt(self):
        self.Obj.set_Dt(Dt=[2.,3.])
        self.Obj.set_Dt(Dt=[0.,10.])

    def test02_set_Resamp(self):
        t = np.linspace(2.,8.,100)
        self.Obj.set_Resamp(t=t, f=None, Method='interp', interpkind='linear')
        assert self.Obj.t.size==100 and self.Obj.data.shape==(100,100)
        self.Obj.set_Resamp(t=None, f=50., Method='movavrg', interpkind='linear')
        assert self.Obj.t.size==500 and self.Obj.data.shape==(500,100)

    def test03_OutIn(self):
        self.Obj.Out_add(['D05','D30','D55','D70'], LCrit=['Name'])
        assert self.Obj.data.shape[1]==96 and len(self.Obj.Chans)==96 and not any([ss in self.Obj.Chans for ss in ['D05','D30','D55','D70']])
        self.Obj.In_add(['D30'], LCrit=['Name'])
        assert self.Obj.data.shape[1]==97 and len(self.Obj.Chans)==97 and 'D30' in self.Obj.Chans
        l = self.Obj.In_list()
        assert l==self.Obj.Chans

    def test04_Corr(self):
        self.Obj.Out_add(['D05','D30','D55','D70'], LCrit=['Name'])
        assert self.Obj.data.shape[1]==96 and len(self.Obj.Chans)==96 and not any([ss in self.Obj.Chans for ss in ['D05','D30','D55','D70']])
        self.Obj.In_add(['D30'], LCrit=['Name'])
        assert self.Obj.data.shape[1]==97 and len(self.Obj.Chans)==97 and 'D30' in self.Obj.Chans
        l = self.Obj.In_list()
        assert l==self.Obj.Chans

    def test05_select(self):
        ind = self.Obj.select(Val=['D00','D10','D50','D85'], Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='In', Out=bool, ToIn=False)
        assert ind.shape==(100,) and np.sum(ind)==4
        ind = self.Obj.select(Val=['D00','D10','D50','D85'], Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='In', Out=bool, ToIn=True)
        assert ind.shape==(97,) and np.sum(ind)==4
        ind0 = self.Obj.select(Val=['D00','D10','D50','D85'], Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='In', Out=int, ToIn=False)
        ind1 = self.Obj.select(Val=['D00','D10','D50','D85'], Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='In', Out=int, ToIn=True)
        assert ind0.size==4 and ind1.size==4 and np.all(ind1<=ind0)
        ind = self.Obj.select(Val=['D00','D10','D50','D85'], Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='In', Out='Name', ToIn=True)
        assert type(ind) is list and ind==['D00','D10','D50','D85']

    def test06_interp(self):
        s0 = np.copy(self.Obj.data[np.argmin(np.abs(self.Obj.t-1.1)),0:2])
        s1 = np.copy(self.Obj.data[np.argmin(np.abs(self.Obj.t-2.5)),:])
        s2 = np.copy(self.Obj.data[np.argmin(np.abs(self.Obj.t-6.2)),2])
        self.Obj.interp(lt=[1.1,2.5,6.2], lNames=[['D00','D01'],'All','D02'])
        assert not np.all(s0==self.Obj.data[np.argmin(np.abs(self.Obj.t-1.1)),0:2])
        assert not np.all(s1==self.Obj.data[np.argmin(np.abs(self.Obj.t-2.5)),:])
        assert not s2==self.Obj.data[np.argmin(np.abs(self.Obj.t-6.2)),2]

    def test07_substract_Dt(self):
        ind = np.argmin(np.abs(self.Obj.t-5.5))
        self.Obj.substract_Dt(tsub=self.Obj.t[ind])
        assert np.all(self.Obj.data[ind,:]==0.)

    def test08_set_fft(self):
        self.Obj.set_fft(DF=[10.,12.], Harm=False, DFEx=None, HarmEx=True)
        self.Obj.set_fft(DF=[10.,12.], Harm=True, DFEx=[14.,15.], HarmEx=True)

    def test09_set_PhysNoise(self):
        self.Obj.set_PhysNoise(Method='svd', Modes=list(range(0,8)), DF=None, DFEx=None, Harm=True, HarmEx=True, Deg=0, Nbin=3, LimRatio=0.05, Plot=False)
        self.Obj.set_PhysNoise(Method='fft', Modes=None, DF=[10.,12.], DFEx=None, Harm=True, HarmEx=True, Deg=0, Nbin=3, LimRatio=0.05, Plot=False)


    def test10_plot(self):
        Lax = self.Obj.plot()
        plt.close('all')

    def test11_plot_svd(self):
        Lax = self.Obj.plot_svd(Modes=10, NRef=4)
        plt.close('all')

    def test12_plot_fft(self):
        Lax = self.Obj.plot_fft(Val=['D19','D48'], Crit='Name', V='simple', SpectNorm=True)
        plt.close('all')

    def test13_saveload(self):
        print(self.Obj.Id.LObj)
        self.Obj.save()
        obj = tfpf.Open(self.Obj.Id.SavePath + self.Obj.Id.SaveName + '.npz')
        os.remove(self.Obj.Id.SavePath + self.Obj.Id.SaveName + '.npz')





















