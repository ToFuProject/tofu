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
import tofu.Eq as tfEq


Root = tfpf.Find_Rootpath()
Addpath = '/tests/tests03_Eq/'

VerbHead = 'tfEq.'


#######################################################
#
#     Setup and Teardown
#
#######################################################

def setup_module(module):
    print ("") # this is to get a newline after the dots
    print ("--------------------------------------------")
    print ("--------------------------------------------")
    print ("        test02_core")

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


VesTor = tfpf.Open(Root+'/tests/tests01_geom/TFG_VesTor_AUG_Test_sh0.npz')
VesLin = tfpf.Open(Root+'/tests/tests01_geom/TFG_VesLin_AUG_Test_sh0.npz')



#######################################################
#
#     Eq2D
#
#######################################################


Tab_t = np.array([1.,2.,3.])
Tab_Pts = np.array([[1.5,1.6,1.7,1.8,1.5,1.6,1.7,1.8,1.5,1.6,1.7,1.8,1.5,1.6,1.7,1.8],[-1.,-1.,-1.,-1.,0.,0.,0.,0.,1.,1.,1.,1.,2.,2.,2.,2.]])
Tab_vPts = np.array([[1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.],[1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.],[1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.]])
Nt = Tab_t.size


class test01_Eq2D:

    @classmethod
    def setup_class(cls, Tab_t=Tab_t, Tab_Pts=Tab_Pts, Tab_vPts=Tab_vPts, Nt=Nt):
        print ("")
        print "--------- "+VerbHead+cls.__name__

        MagAx = np.tile(np.array([1.7,0.]),(Nt,1))
        Sep = [np.array([[1.4,2.,2.,1.4],[-2.,-2.,2.,2.]]) for ii in range(0,Nt)]
        rho_p = {'vRef':np.tile(np.array([0.,1.]),(Nt,1)), 'vPts':Tab_vPts}
        cls.Obj = tfEq.Eq2D('Test', Tab_Pts, t=Tab_t, MagAx=MagAx, Sep=Sep, rho_p=rho_p, Ref='rho_p',
                      Type='Tor', Exp='Test', shot=0, Diag='Test', dtime=None, dtimeIn=False, SavePath=Root+Addpath)

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

    def test01_attributes(self):
        assert all([hasattr(self.Obj,aa) for aa in ['Id', 'Type', 'shot', 'Diag',
                                                    'PtsCross', 'NP', 't', 'Nt',
                                                    'MagAx', 'Sep', 'pf', 'tf', 'rho_p', 'rho_t', 'surf', 'vol', 'q', 'jp', 'theta', 'thetastar', 'BTX', 'BRY', 'BZ', 'Tabs_vPts', 'Ref', 'NRef',
                                                    'interp','get_RadDir','plot','plot_vs','save']])
        # Testing types and formats
        assert type(self.Obj.PtsCross) is np.ndarray and self.Obj.PtsCross.ndim==2 and self.Obj.PtsCross.shape[0]==2 and self.Obj.PtsCross.dtype.name=='float64'
        assert type(self.Obj.NP) is int and self.Obj.NP==self.Obj.PtsCross.shape[1]
        assert type(self.Obj.t) is np.ndarray and self.Obj.t.ndim==1 and self.Obj.t.dtype.name=='float64'
        assert type(self.Obj.Nt) is int and self.Obj.Nt==self.Obj.t.size
        assert self.Obj.MagAx is None or (type(self.Obj.MagAx) is np.ndarray and self.Obj.MagAx.ndim==2 and self.Obj.MagAx.shape==(self.Obj.Nt,2) and self.Obj.MagAx.dtype.name=='float64')
        assert self.Obj.Sep is None or (type(self.Obj.Sep) is list and len(self.Obj.Sep)==self.Obj.Nt and all([type(ss) is np.ndarray and ss.ndim==2 and ss.shape[0]==2 and ss.dtype.name=='float64' for ss in self.Obj.Sep]))
        assert type(self.Obj.Tabs_vPts) is list and all([type(ss) is str for ss in self.Obj.Tabs_vPts])
        for ss in self.Obj.Tabs_vPts:
            vv = eval('self.Obj.'+ss)
            assert vv is None or (type(vv) is np.ndarray and vv.ndim==2 and vv.shape==(self.Obj.Nt,self.Obj.NP) and vv.dtype.name=='float64')
        assert type(self.Obj.Ref) is str and self.Obj.Ref in self.Obj.Tabs_vPts and eval('self.Obj.'+self.Obj.Ref) is not None
        assert type(self.Obj.NRef) is int and self.Obj._Tab[self.Obj.Ref]['vRef'].shape[1]==self.Obj.NRef
        assert hasattr(self.Obj.interp, '__call__')
        assert hasattr(self.Obj.get_RadDir, '__call__')
        assert hasattr(self.Obj.plot, '__call__')
        assert hasattr(self.Obj.plot_vs, '__call__')
        assert hasattr(self.Obj.save, '__call__')

        # Testing values
        assert np.all(self.Obj.PtsCross==np.array([[1.5,1.6,1.7,1.8,1.5,1.6,1.7,1.8,1.5,1.6,1.7,1.8,1.5,1.6,1.7,1.8],[-1.,-1.,-1.,-1.,0.,0.,0.,0.,1.,1.,1.,1.,2.,2.,2.,2.]]))
        assert np.all(self.Obj.t==np.array([1.,2.,3.]))
        assert np.all(self.Obj.MagAx==np.tile(np.array([1.7,0.]),(Nt,1)))
        assert np.all([ss==np.array([[1.4,2.,2.,1.4],[-2.,-2.,2.,2.]]) for ss in self.Obj.Sep])
        assert np.all(self.Obj.rho_p==np.array([[1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.],[1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.],[1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.]]))

    def test02_interp(self):
        Pts = np.array([[1.55,1.75,1.85],[-0.5,0.,0.5]])
        dQuant = self.Obj.interp(Pts, Quant='rho_p', indt=0, t=None, deg=3, Test=True)
        dQuant = self.Obj.interp(Pts, Quant=['rho_p'], indt=None, t=2., deg=3, Test=True)
        dQuant = self.Obj.interp(Pts, Quant=['rho_p','rho_t'], indt=None, t=None, deg=3, Test=True)


    def test03_get_RadDir(self):
        Pts = np.array([[1.55,1.75,1.85],[-0.5,0.,0.5]])
        rgrad = self.Obj.get_RadDir(Pts, indt=0, t=None, Test=True)
        rgrad = self.Obj.get_RadDir(Pts, indt=None, t=2., Test=True)
        rgrad = self.Obj.get_RadDir(Pts, indt=None, t=None, Test=True)


    def test04_plot(self):
        ax0 = self.Obj.plot(V='static', ax=None, Quant=['rho_p','MagAx','Sep','pol'], plotfunc='imshow', lvls=[1.,1.5,2.], indt=None, t=None, Cdict=None, RadDict=None, PolDict=None, MagAxDict=None, SepDict=None, LegDict=tfd.TorLegd,
                            ZRef='MagAx', NP=100, Ratio=-0.1, ylim=None, NaNout=True, Abs=True, clab=True, cbar=False, draw=True, a4=False, Test=True)
        ax1 = self.Obj.plot(V='static', ax=None, Quant=['q','MagAx','Sep','pol'], plotfunc='contour', lvls=[1.,1.5,2.], indt=None, t=None, Cdict=None, RadDict=None, PolDict=None, MagAxDict=None, SepDict=None, LegDict=tfd.TorLegd,
                            ZRef='MagAx', NP=100, Ratio=-0.1, ylim=None, NaNout=True, Abs=True, clab=True, cbar=False, draw=True, a4=False, Test=True)
        ax2 = self.Obj.plot(V='inter', ax=None, Quant=['rho_p','MagAx','Sep','pol'], plotfunc='scatter', lvls=[1.,1.5,2.], indt=None, t=None, Cdict=None, RadDict=None, PolDict=None, MagAxDict=None, SepDict=None, LegDict=tfd.TorLegd,
                            ZRef='MagAx', NP=100, Ratio=-0.1, ylim=None, NaNout=True, Abs=True, clab=True, cbar=False, draw=True, a4=False, Test=True)
        ax3 = self.Obj.plot(V='inter', ax=None, Quant=['q','MagAx','Sep','pol'], plotfunc='contourf', lvls=[1.,1.5,2.], indt=None, t=None, Cdict=None, RadDict=None, PolDict=None, MagAxDict=None, SepDict=None, LegDict=tfd.TorLegd,
                            ZRef='MagAx', NP=100, Ratio=-0.1, ylim=None, NaNout=True, Abs=True, clab=True, cbar=False, draw=True, a4=False, Test=True)
        plt.close('all')

    def test05_plot_vs(self):
        ax = self.Obj.plot_vs(ax=None, Qy='rho_p', Qx='rho_p', indt=0, t=None, Dict=None, xlim=None, ylim=None, Abs=True, LegDict=None, draw=True, a4=False, Test=True)
        ax = self.Obj.plot_vs(ax=None, Qy='rho_p', Qx='rho_p', indt=None, t=2., Dict=None, xlim=None, ylim=None, Abs=True, LegDict=None, draw=True, a4=False, Test=True)
        ax = self.Obj.plot_vs(ax=None, Qy='rho_p', Qx='rho_p', indt=None, t=None, Dict=None, xlim=None, ylim=None, Abs=True, LegDict=None, draw=True, a4=False, Test=True)
        ax = self.Obj.plot_vs(ax=None, Qy='q', Qx='rho_p', indt=None, t=None, Dict=None, xlim=None, ylim=None, Abs=True, LegDict=None, draw=True, a4=False, Test=True)
        plt.close('all')

    def test06_saveload(self):
        self.Obj.save()
        Eq = tfpf.Open(self.Obj.Id.SavePath + self.Obj.Id.SaveName + '.npz')
        os.remove(self.Obj.Id.SavePath + self.Obj.Id.SaveName + '.npz')


















