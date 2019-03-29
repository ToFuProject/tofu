"""
This module contains tests for tofu.geom in its structured version
"""

# Built-in
import os

# Standard
import numpy as np
import matplotlib.pyplot as plt

# Nose-specific
from nose import with_setup # optional

# tofu-specific
from tofu import __version__
import tofu.pathfile as tfpf
import tofu.geom as tfg
import tofu.data as tfd


_here = os.path.abspath(os.path.dirname(__file__))
VerbHead = 'tofu.data.tests03_core'


#######################################################
#
#     Setup and Teardown
#
#######################################################

def setup_module(module):
    print("") # this is to get a newline after the dots
    LF = os.listdir(_here)
    LF = [lf for lf in LF if all([ss in lf for ss in ['TFD_','Test','.npz']])]
    LF = [lf for lf in LF if not lf[lf.index('_Vv')+2:lf.index('_U')]==__version__]
    print("Removing the following previous test files:")
    print (LF)
    for lf in LF:
        os.remove(os.path.join(_here,lf))
    #print("setup_module before anything in this file")

def teardown_module(module):
    #os.remove(VesTor.Id.SavePath + VesTor.Id.SaveName + '.npz')
    #os.remove(VesLin.Id.SavePath + VesLin.Id.SaveName + '.npz')
    #print("teardown_module after everything in this file")
    #print("") # this is to get a newline
    LF = os.listdir(_here)
    LF = [lf for lf in LF if all([ss in lf for ss in ['TFD_','Test','.npz']])]
    LF = [lf for lf in LF if lf[lf.index('_Vv')+2:lf.index('_U')]==__version__]
    print("Removing the following test files:")
    print (LF)
    for lf in LF:
        os.remove(os.path.join(_here,lf))
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

def emiss(pts, t=None):
    R = np.hypot(pts[0,:],pts[1,:])
    Z = pts[2,:]
    r = np.hypot(R-2.4, Z)
    e0 = np.exp(-r**2/0.25)
    e1 = np.exp(-r**2/0.1)
    e2 = np.exp(-(r-0.4)**2/0.1)
    if t is None:
        emiss = e0 + 0.1*e1
    else:
        emiss = (e0[None,:]
                 + 0.5*np.cos(t)[:,None]*e1[None,:]
                 + 0.1*np.sin(2*t)[:,None]*e2[None,:])
    return emiss




class Test01_DataCam12D(object):


    def _create_cams(nch, lconf, ld, SavePath='./'):
        c0 = tfg.utils.create_CamLOS1D(P=[4.5,0,0], F=0.1, N12=nch, D12=0.05,
                                       angs=[-np.pi,np.pi/10.,0.],
                                       config=lconf[0],
                                       Diag='Test', Name='Test',
                                       SavePath=SavePath)
        c1 = tfg.utils.create_CamLOS2D(P=[4.5,0,0], F=0.1,
                                       N12=[int(1.5*nch),nch],
                                       D12=[0.075,0.05],
                                       angs=[-np.pi,np.pi/10.,0.],
                                       config=lconf[1],
                                       Diag='Test', Name='Test',
                                       SavePath=SavePath)
        return [c0, c1]

    @classmethod
    def setup_class(cls, nch=30, nt=50, SavePath='./'):

        # time vector
        t = np.linspace(0, 10, nt)

        # Configs
        conf0 = tfg.utils.create_config(case='B2')
        conf1 = tfg.utils.create_config(case='B3')

        # dchans and cams
        d0 = dict(Name=['C0-{0}'.format(ii) for ii in range(0,nch)])
        d1 = dict(Name=['C1-{0}'.format(ii) for ii in range(0,nch)])
        lc = cls._create_cams(nch, [conf0, conf1], [d0, d1], SavePath=SavePath)

        # -------
        # dextra
        nteq = nt // 2
        teq = np.linspace(t.min(), t.max(), nteq)
        teq2 = np.copy(teq) - 0.01
        Ax = np.array([2.4+0.1*np.cos(teq2), 0.1*np.sin(teq2)]).T
        Ax2 = np.array([2.4+0.1*np.cos(teq2/2.), 0.1*np.sin(teq2/2.)]).T
        Sep = (Ax[:,:,None]
               + 0.4*np.array([[-1,1,1,-1],[-1,-1,1,1]])[None,:,:])
        Sep2 = (Ax2[:,:,None]
                + 0.3*np.array([[-1,1,1,-1],[-1,-1,1,1]])[None,:,:])

        n1, n2 = 40, 60
        x1, x2 = np.linspace(2,3,n1), np.linspace(-0.8,0.8,n2)
        dx1, dx2 = (x1[1]-x1[0])/2., (x2[1]-x2[0])/2
        extent = (x1[0]-dx1, x1[-1]+dx1, x2[0]-dx2, x2[-1]+dx2)
        pts = np.array([np.tile(x1,n2), np.zeros((n1*n2,)), np.repeat(x2,n1)])
        emis = emiss(pts, t=teq2).reshape(nteq, n2, n1)
        dextra0 = {'pouet':{'t':teq, 'c':'k', 'data':np.sin(teq),
                            'units':'a.u.' , 'label':'pouet'},
                   'Ax':{'t':teq2, 'data2D':Ax},
                   'Sep':{'t':teq2, 'data2D':Sep},
                   'map':{'t':teq2, 'data2D':emis, 'extent':extent}}
        dextra1 = {'pouet':{'t':teq, 'c':'k', 'data':np.cos(teq),
                            'units':'a.u.' , 'label':'pouet'},
                   'Ax':{'t':teq2, 'data2D':Ax2},
                   'Sep':{'t':teq2, 'data2D':Sep2}}

        # -------
        # signal as Data from lcams
        lm = ['sum', 'simps']
        lData = [lc[ii].calc_signal(emiss, t=t,
                                    res=0.01, method=lm[ii], plot=False)
                 for ii in range(0,len(lc))]

        # Adding concatenated sig / data and without lcam
        sig = np.concatenate([dd.data for dd in lData[:2]], axis=1)
        lData += [tfd.DataCam1D(data=sig, Name='All',
                                Diag='Test', Exp=conf0.Id.Exp, config=conf0)]
        dX12 = lc[1].dX12
        lData += [tfd.DataCam2D(data=lData[1].data, dX12=dX12, Name='c1nocam',
                                Diag='Test', Exp=conf0.Id.Exp)]

        # Setting dchans
        for ii in range(0,len(lData)):
            if ii % 2 == 0:
                lData[ii].set_dchans({'Name':['c%s'%jj for jj in
                                              range(0,lData[ii].nch)]})

        # Setting dextra
        for ii in range(0,len(lData)):
            de = dextra0 if ii % 2 == 0 else dextra1
            lData[ii].set_dextra(dextra=de)

        # Storing
        cls.lobj = lData

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
        for ii in range(0,len(self.lobj)):
            oo = self.lobj[ii]
            out0 = oo.dchans()
            if out0 is not None:
                assert type(out0) is dict
                lk = list(out0.keys())
                if len(lk)>0:
                    out1 = oo.dchans(key=lk[0])
                    assert type(out1) is np.ndarray
                assert all([len(out0[ss])==oo.ddataRef['nch'] for ss in lk])

    def test02_select_t(self):
        for oo in self.lobj:
            ind = oo.select_t(t=None, out=bool)
            assert ind.sum()==oo.ddataRef['nt']
            ind = oo.select_t(t=5, out=bool)
            if oo.ddataRef['t'] is None:
                assert ind.sum()==oo.ddataRef['nt']
            else:
                assert ind.sum()==1
            ind = oo.select_t(t=[1,4], out=bool)
            if oo.ddataRef['t'] is None:
                assert ind.sum()==oo.ddataRef['nt']
            else:
                assert np.all((oo.t[ind]>=1.) & (oo.t[ind]<=4))

    def test03_set_indt(self):
        for ii in range(0,len(self.lobj)):
            oo = self.lobj[ii]
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
        for ii in range(0,len(self.lobj)):
            oo = self.lobj[ii]
            if oo.geom is not None and oo.geom['LCam'] is not None:
                ind = oo.select_ch(touch='Ves', out=bool)
                assert ind.sum()==oo.Ref['nch']
            if oo.Ref['dchans'] not in [None,{}] :
                ind =oo.select_ch(key='Name',val=['C0-0','C1-0'],log='any',out=bool)
                assert ind.sum() in [1,2]

    def test05_set_indch(self):
        for ii in range(0,len(self.lobj)):
            oo = self.lobj[ii]
            if oo.geom is not None and oo.geom['LCam'] is not None:
                oo.set_indch(touch='Ves')
                assert oo.indch.sum()==oo.Ref['nch']
            if oo.Ref['dchans'] not in [None,{}] :
                oo.set_indch(key='Name',val=['C0-0','C1-0'],log='any')
                assert oo.indch.sum() in [1,2]
            oo.set_indch(indch=list(range(0,min(5,oo.Ref['nch']))))
            assert oo.indch.sum() in [oo.Ref['nch'],5]

    def test06_set_data0(self):
        for ii in range(0,len(self.lobj)):
            oo = self.lobj[ii]
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

    def test07_operators(self):
        o0 = self.lobj[-1]
        o1 = 100.*(o0-0.1*o0)

    def test08_plot(self):
        connect = (hasattr(plt.get_current_fig_manager(),'toolbar')
                   and plt.get_current_fig_manager().toolbar is not None)
        for ii in range(0,len(self.lobj)):
            oo = self.lobj[ii]
            KH = oo.plot(key=None, ntMax=4, nchMax=2, fs=None,
                         dmargin=dict(left=0.06, right=0.9),
                         connect=connect, wintit='test', tit='AHAH')
            KH = oo.plot(key='Name', draw=False, dmargin=None, connect=connect)
        plt.close('all')

    def test09_compare(self):
        if self.__class__ is Test02_Data2D:
            return
        connect = (hasattr(plt.get_current_fig_manager(),'toolbar')
                   and plt.get_current_fig_manager().toolbar is not None)
        o0 = self.lobj[0]
        for ii in range(1,len(self.lobj)):
            oo = self.lobj[ii]
            KH = oo.plot_compare(o0, connect=connect)
        plt.close('all')

    def test10_combine(self):
        if self.__class__ is Test02_Data2D:
            return
        connect = (hasattr(plt.get_current_fig_manager(),'toolbar')
                   and plt.get_current_fig_manager().toolbar is not None)
        o0 = self.lobj[0]
        for ii in range(1,len(self.lobj)):
            oo = self.lobj[ii]
            KH = oo.plot_combine(o0, connect=connect)
        plt.close('all')

    def test11_tofromdict(self):
        for ii in range(0,len(self.lobj)):
            oo = self.lobj[ii]
            dd = oo._todict()
            if oo.Id.Cls=='Data1D':
                oo = tfd.Data1D(fromdict=dd)
            else:
                oo = tfd.Data2D(fromdict=dd)
            assert dd==oo._todict(), "Unequal to and from dict !"

    def test12_saveload(self):
        for ii in range(0,len(self.lobj)):
            oo = self.lobj[ii]
            dd = oo._todict()
            pfe = oo.save(verb=False, return_pfe=True)
            obj = tfpf.Open(pfe, Print=False)
            # Just to check the loaded version works fine
            do = obj._todict()
            assert tfu.dict_cmp(dd,do)
            os.remove(pfe)




"""
class Test02_Data2D(Test01_Data1D):
    @classmethod
    def setup_class(cls):
        thet = np.linspace(0,2.*np.pi,100)
        P = np.array([2.4 + 0.8*np.cos(thet),0.8*np.sin(thet)])
        V = tfg.Ves(Name='Test', Poly=P, Exp='Dummy', SavePath=here)
        N = 5
        Ds, us = tfg.utils.compute_CamLOS2D_pinhole([3.5,0.,0.], 0.1,
                                                    (0.05,0.05), (N,N),
                                                    angs=None,
                                                    nIn=[-1,0.,0.],
                                                    VType='Tor',
                                                    return_Du=True)
        d0 = dict(Name=['C0-{0}'.format(ii) for ii in range(0,N**2)])
        d1 = dict(Name=['C1-{0}'.format(ii) for ii in range(0,N**2)])
        config = tfg.Config(Name="Conf", lStruct=[V])
        C0 = tfg.CamLOS2D(Name="Dummy", dgeom=(Ds,us), config=config,
                          Exp=config.Id.Exp, Diag='Test', dchans=d0, SavePath=here)
        V.save()
        C0.save()
        t = np.linspace(0,10,20)
        sig00 = C0.calc_signal(emiss, t=None, res=0.01, method='sum',
                               plot=False)
        sig01 = C0.calc_signal(emiss, t=t, res=0.01, method='sum',
                               plot=False)
        cls.lobj = [tfd.Data2D(sig00, Id='0', SavePath=here),
                    tfd.Data2D(sig01, t=t, Id='1', SavePath=here),
                    tfd.Data2D(sig01, t=t, Ves=V, LStruct=C0.LStruct,
                               Id='1', SavePath=here),
                    tfd.Data2D(sig00, LCam=C0, Id='2', SavePath=here),
                    tfd.Data2D(sig01, t=t, LCam=C0, Id='3', SavePath=here)]

    def test08_plot(self):
        for ii in range(0,len(self.lobj)):
            oo = self.lobj[ii]
            if oo._X12 is not None and oo.geom is not None:
                oo.set_indch()
                KH = oo.plot(key=None, Max=None, fs=None,
                             invert=True, vmin=0, wintit='test', tit='AHAH',
                             dmargin=dict(left=0.05,right=0.9))
                KH = oo.plot(key='Name', Max=2, fs=(13,5),
                             normt=True, dmargin=None)
        plt.close('all')
"""
