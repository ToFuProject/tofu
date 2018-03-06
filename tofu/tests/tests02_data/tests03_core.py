"""
This module contains tests for tofu.geom in its structured version
"""

# External modules
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings as warn

# Nose-specific
from nose import with_setup # optional


# Importing package tofu.geom
from tofu import __version__
import tofu.defaults as tfd
import tofu.pathfile as tfpf
import tofu.utils as tfu
import tofu.geom as tfg
import tofu.data as tfd


here = os.path.abspath(os.path.dirname(__file__))
VerbHead = 'tofu.data.tests03_core'


#######################################################
#
#     Setup and Teardown
#
#######################################################

def setup_module(module):
    print("") # this is to get a newline after the dots
    LF = os.listdir(here)
    LF = [lf for lf in LF if all([ss in lf for ss in ['TFD_','Test','.npz']])]
    LF = [lf for lf in LF if not lf[lf.index('_Vv')+2:lf.index('_U')]==__version__]
    print("Removing the following previous test files:")
    print (LF)
    for lf in LF:
        os.remove(os.path.join(here,lf))
    #print("setup_module before anything in this file")

def teardown_module(module):
    #os.remove(VesTor.Id.SavePath + VesTor.Id.SaveName + '.npz')
    #os.remove(VesLin.Id.SavePath + VesLin.Id.SaveName + '.npz')
    #print("teardown_module after everything in this file")
    #print("") # this is to get a newline
    LF = os.listdir(here)
    LF = [lf for lf in LF if all([ss in lf for ss in ['TFD_','Test','.npz']])]
    LF = [lf for lf in LF if lf[lf.index('_Vv')+2:lf.index('_U')]==__version__]
    print("Removing the following test files:")
    print (LF)
    for lf in LF:
        os.remove(os.path.join(here,lf))
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
#     Creating Ves objects and testing methods
#
#######################################################

def emiss(Pts, t=None):
    R = np.hypot(Pts[0,:],Pts[1,:])
    Z = Pts[2,:]
    e0 = np.exp(-(R-2.5)**2/1. - Z**2/1.)
    e1 = np.exp(-(R-2.5)**2/0.3 - Z**2/0.3)
    if t is None:
        e = e0 + 0.8*e1
    else:
        e = e0[np.newaxis,:] + 0.8*(np.cos(t)[:,np.newaxis])*e1[np.newaxis,:]
    return e


class Test01_Data1D:

    @classmethod
    def setup_class(cls):
        thet = np.linspace(0,2.*np.pi,100)
        P = np.array([2.4 + 0.8*np.cos(thet),0.8*np.sin(thet)])
        V = tfg.Ves('Test', P, Exp='Test', SavePath=here)
        N = 10
        Ds = np.array([3.*np.ones(N,), np.zeros((N,)), np.linspace(-0.5,0.5,N)])
        A = np.r_[2.5,0,0]
        us = A[:,np.newaxis]-Ds
        d0 = dict(Name=['C0-{0}'.format(ii) for ii in range(0,N)])
        d1 = dict(Name=['C1-{0}'.format(ii) for ii in range(0,N)])
        C0 = tfg.LOSCam1D('C0',( Ds,us), Ves=V,
                          Exp='Test', Diag='Test', dchans=d0, SavePath=here)
        C1 = tfg.LOSCam1D('C1', (Ds,us), Ves=V,
                          Exp='Test', Diag='Test', dchans=d1, SavePath=here)
        V.save()
        C0.save()
        C1.save()
        t = np.linspace(0,10,20)
        sig00 = C0.calc_signal(emiss, t=None, dl=0.01, method='sum',
                               plot=False,out='')
        sig01 = C0.calc_signal(emiss, t=t, dl=0.01, method='sum',
                               plot=False,out='')
        sig10 = C1.calc_signal(emiss, t=None, dl=0.01, method='sum',
                               plot=False,out='')
        sig11 = C1.calc_signal(emiss, t=t, dl=0.01, method='sum',
                               plot=False,out='')
        sig20 = np.concatenate((sig00,sig10))
        sig21 = np.concatenate((sig01,sig11),axis=1)
        cls.LObj = [tfd.Data1D(sig00, Id='0', SavePath=here),
                   tfd.Data1D(sig01, t=t, Id='1', SavePath=here),
                   tfd.Data1D(sig00, LCam=C0, Id='2', SavePath=here),
                   tfd.Data1D(sig01, t=t, LCam=C0, Id='3', SavePath=here),
                   tfd.Data1D(sig20, LCam=[C0,C1], Id='4', SavePath=here),
                   tfd.Data1D(sig21, t=t, LCam=[C0,C1], Id='5', SavePath=here)]

    @classmethod
    def teardown_class(cls):
        pass

    def setup(self):
        #print("TestUM:setup() before each test method")
        pass

    def teardown(self):
        #print("TestUM:teardown() after each test method")
        pass

    def test01_dchans(self):
        for ii in range(0,len(self.LObj)):
            oo = self.LObj[ii]
            out0 = oo.dchans()
            out1 = oo.dchans(key='Name')
            if oo.geom is None:
                assert out0 is None and out1 is None
            else:
                lK = list(oo.dchans().keys())
                assert type(out0) is dict and type(out1) is np.ndarray
                assert all([ss in out0.keys() for ss in lK])
                assert all([len(out0[ss])==oo.Ref['nch'] for ss in lK])
                assert len(out1)==oo.Ref['nch']

    def test02_select_t(self):
        for ii in range(0,len(self.LObj)):
            oo = self.LObj[ii]
            ind = oo.select_t(t=None, out=bool)
            assert ind.sum()==oo.Ref['nt']
            ind = oo.select_t(t=5, out=bool)
            if oo.Ref['t'] is None:
                assert ind.sum()==oo.Ref['nt']
            else:
                assert ind.sum()==1
            ind = oo.select_t(t=[1,4], out=bool)
            if oo.Ref['t'] is None:
                assert ind.sum()==oo.Ref['nt']
            else:
                assert np.all((oo.t[ind]>=1.) & (oo.t[ind]<=4))

    def test03_set_indt(self):
        for ii in range(0,len(self.LObj)):
            oo = self.LObj[ii]
            oo.set_indt(t=[2,3])
            if oo.Ref['t'] is None:
                assert oo.indt.sum()==oo.Ref['nt']
            else:
                assert np.all((oo.t>=2) & (oo.t<=3))
            oo.set_indt(indt=list(range(0,min(4,oo.Ref['nt']))))
            assert oo.nt == 4 or oo.nt==1
            oo.set_indt()
            assert oo.nt == oo.Ref['nt']

    def test04_select_ch(self):
        for ii in range(0,len(self.LObj)):
            oo = self.LObj[ii]
            if oo.geom is not None:
                ind = oo.select_ch(touch='Ves', out=bool)
                assert ind.sum()==oo.Ref['nch']
            if oo.Ref['dchans'] not in [None,{}] :
                ind =oo.select_ch(key='Name',val=['C0-0','C1-0'],log='any',out=bool)
                assert ind.sum() in [1,2]

    def test05_set_indch(self):
        for ii in range(0,len(self.LObj)):
            oo = self.LObj[ii]
            if oo.geom is not None:
                oo.set_indch(touch='Ves')
                assert oo.indch.sum()==oo.Ref['nch']
            if oo.Ref['dchans'] not in [None,{}] :
                oo.set_indch(key='Name',val=['C0-0','C1-0'],log='any')
                assert oo.indch.sum() in [1,2]
            oo.set_indch(indch=list(range(0,min(5,oo.Ref['nch']))))
            assert oo.indch.sum() in [oo.Ref['nch'],5]

    def test06_set_data0(self):
        for ii in range(0,len(self.LObj)):
            oo = self.LObj[ii]
            # Re-initialise
            oo.set_indt()
            oo.set_indch()
            if oo.Ref['nt']>1:
                oo.set_data0(data0=oo.Ref['data'][0,:])
                assert oo.data0['indt'] is None and oo.data0['Dt'] is None
                assert np.allclose(oo.data[0,:],0.)
                oo.set_data0(indt=[1,2,6,8,9])
                assert oo.data0['indt'].sum()==5
                assert oo.data0['data'].size==oo.Ref['nch']
                if oo.t is not None:
                    oo.set_data0(Dt=[2,3])
                    assert oo.data0['Dt'][0]>=2. and oo.data0['Dt'][1]<=3.
                    assert oo.data0['data'].size==oo.Ref['nch']
                oo.set_data0()
                assert oo.data0['data'] is None
                assert np.allclose(oo.data,oo.Ref['data'])

    def test07_plot(self):
        for ii in range(0,len(self.LObj)):
            oo = self.LObj[ii]
            dax, KH = oo.plot(key=None, Max=None)
            dax, KH = oo.plot(key='Name', Max=2)
            plt.close('all')

    def test08_tofromdict(self):
        for ii in range(0,len(self.LObj)):
            oo = self.LObj[ii]
            dd = oo._todict()
            if oo.Id.Cls=='Data1D':
                oo = tfd.Data1D(fromdict=dd)
            else:
                oo = tfd.Data2D(fromdict=dd)
            assert dd==oo._todict(), "Unequal to and from dict !"

    def test09_saveload(self):
        for ii in range(0,len(self.LObj)):
            oo = self.LObj[ii]
            dd = oo._todict()
            oo.save(Print=False)
            PathFileExt = os.path.join(oo.Id.SavePath,
                                       oo.Id.SaveName+'.npz')
            obj = tfpf.Open(PathFileExt, Print=False)
    # Just to check the loaded version works fine
            do = obj._todict()
            assert tfu.dict_cmp(dd,do)
            os.remove(PathFileExt)





class Test01_Data2D(Test01_Data1D):
    @classmethod
    def setup_class(cls):
        thet = np.linspace(0,2.*np.pi,100)
        P = np.array([2.4 + 0.8*np.cos(thet),0.8*np.sin(thet)])
        V = tfg.Ves('Test', P, Exp='Test', SavePath=here)
        N = 5
        Ds, us = tfu.create_CamLOS2D([3.5,0.,0.], 0.1, (0.05,0.05), (N,N),
                                     nIn=[-1,0.,0.], e1=None, e2=None,
                                     VType='Tor')
        d0 = dict(Name=['C0-{0}'.format(ii) for ii in range(0,N**2)])
        d1 = dict(Name=['C1-{0}'.format(ii) for ii in range(0,N**2)])
        C0 = tfg.LOSCam2D('C0', (Ds,us), Ves=V,
                          Exp='Test', Diag='Test', dchans=d0, SavePath=here)
        V.save()
        C0.save()
        t = np.linspace(0,10,20)
        sig00 = C0.calc_signal(emiss, t=None, dl=0.01, method='sum',
                               plot=False, out='')
        sig01 = C0.calc_signal(emiss, t=t, dl=0.01, method='sum',
                               plot=False, out='')
        cls.LObj = [tfd.Data2D(sig00, Id='0', SavePath=here),
                    tfd.Data2D(sig01, t=t, Id='1', SavePath=here),
                    tfd.Data2D(sig00, LCam=C0, Id='2', SavePath=here),
                    tfd.Data2D(sig01, t=t, LCam=C0, Id='3', SavePath=here)]

    def test07_plot(self):
        for ii in range(0,len(self.LObj)):
            oo = self.LObj[ii]
            if oo._X12 is not None and oo.geom is not None:
                dax, KH = oo.plot(key=None, Max=None)
                dax, KH = oo.plot(key='Name', Max=2)
                plt.close('all')
