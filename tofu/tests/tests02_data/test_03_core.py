"""
This module contains tests for tofu.geom in its structured version
"""

# Built-in
import os
import warnings

# Standard
import numpy as np
import matplotlib.pyplot as plt

# tofu-specific
from tofu import __version__
import tofu.utils as tfu
import tofu.geom as tfg
import tofu.data as tfd


_here = os.path.abspath(os.path.dirname(__file__))
VerbHead = 'tofu.data.test_03_core'


#######################################################
#
#     Setup and Teardown
#
#######################################################

def setup_module():
    print("") # this is to get a newline after the dots
    LF = os.listdir(_here)
    LF = [lf for lf in LF if all([ss in lf for ss in ['TFD_','Test','.npz']])]
    LF = [lf for lf in LF if not lf[lf.index('_Vv')+2:lf.index('_U')]==__version__]
    print("Removing the following previous test files:")
    print (LF)
    for lf in LF:
        os.remove(os.path.join(_here,lf))
    #print("setup_module before anything in this file")

def teardown_module():
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


    @staticmethod
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
    def setup_class(cls, nch=30, nt=50, SavePath='./', verb=False):

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
                                    res=0.01, method=lm[ii], plot=False)[0]
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
        cls.t = t

        # Saving for intermediate use
        lpfe = []
        for oo in cls.lobj:
            if oo._dgeom['config'] is not None:
                lpfe.append( oo._dgeom['config'].save(return_pfe=True,
                                                      verb=verb) )
            if oo._dgeom['lCam'] is not None:
                for cc in oo._dgeom['lCam']:
                    lpfe.append( cc.save(return_pfe=True, verb=verb) )

        cls.lpfe = lpfe

    @classmethod
    def setup_method(self):
        pass
    def teardown_method(self):
        pass

    @classmethod
    def teardown_class(cls):
        for pfe in set(cls.lpfe):
            os.remove(pfe)
        pass

    def test01_set_dchans(self):
        for ii in range(0,len(self.lobj)):
            oo = self.lobj[ii]
            out0 = oo.dchans()
            if out0 is not None:
                assert type(out0) is dict
                lk = list(out0.keys())
                if len(lk)>0:
                    out1 = oo.dchans(key=lk[0])
                    assert type(out1) is np.ndarray
                    dch2 = {'dch2':['abc' for ii in range(0,oo.ddataRef['nch'])]}
                    oo.set_dchans(dch2, method='update')
                assert all([len(out0[ss])==oo.ddataRef['nch'] for ss in lk])

    def test02_set_dextra(self):
        for ii in range(0,len(self.lobj)):
            oo = self.lobj[ii]
            out0 = oo.dextra
            if out0 is not None:
                assert type(out0) is dict
                t = np.linspace(0,10,10)
                dd = {'pouet': {'t':t, 'data':t,
                                'label':'pouet', 'units':'pouet units'}}
                oo.set_dextra(dd, method='update')

    def test03_select_t(self):
        for oo in self.lobj:
            ind = oo.select_t(t=None, out=bool)
            assert ind.sum()==oo.ddataRef['nt']
            ind = oo.select_t(t=5, out=bool)
            assert ind.sum() == 1
            ind = oo.select_t(t=[1,4], out=bool)
            assert np.all((oo.t[ind]>=1.) & (oo.t[ind]<=4))

    def test04_select_ch(self):
        for oo in self.lobj:
            if oo.dgeom['lCam'] is not None:
                name = [(ii, k) for ii, k in
                        enumerate(oo.config.dStruct['lorder'])
                        if 'Ves' in k or 'PlasmaDomain' in k]
                # assert len(name) == 1  # There can be several Ves now
                ind = oo.select_ch(touch=name[0][1], out=bool)
                assert ind.sum() > 0, (ind.sum(), ind)
                assert np.allclose(ind, oo.select_ch(touch=name[0][0],
                                                     out=bool))
            if len(oo.dchans().keys()) > 0:
                ind = oo.select_ch(key='Name', val=['c0','c10'],
                                   log='any', out=bool)
                assert ind.sum() == 2, (ind.sum(), ind)

    def test05_set_dtreat_indt(self):
        for oo in self.lobj:
            oo.set_dtreat_indt(t=[2,3])
            assert np.all((oo.t>=2) & (oo.t<=3))
            oo.set_dtreat_indt(indt=list(range(0,min(4,oo.ddataRef['nt']))))
            assert oo.nt == 4 or oo.nt==1
            oo.set_dtreat_indt()
            assert oo.nt == oo.ddataRef['nt']

    def test06_set_dtreat_indch(self):
        for oo in self.lobj:
            oo.set_dtreat_indch(indch = range(0,10))
            assert oo.dtreat['indch'].sum() == 10

    def test07_set_dtreat_mask(self):
        for oo in self.lobj:
            # Re-initialise
            oo.set_dtreat_indch()
            # set mask
            mask = np.arange(0,oo.ddataRef['nch'],10)
            oo.set_dtreat_mask(ind=mask, val=np.nan)
            nbnan = np.sum(np.any(np.isnan(oo.data), axis=0))
            assert nbnan >= mask.size, [oo.ddataRef['nch'], nbnan]

    def test08_dtreat_set_data0(self):
        for oo in self.lobj:
            # Re-initialise
            oo.set_dtreat_indt()
            oo.set_dtreat_mask()

            oo.set_dtreat_data0( data0 = oo.data[0,:] )
            assert oo.dtreat['data0-indt'] is None
            assert oo.dtreat['data0-Dt'] is None
            assert np.allclose(oo.data[0,:],0.), oo.data[0,:]

            oo.set_dtreat_data0(indt=[1,2,6,8,9])
            assert oo.dtreat['data0-indt'].sum() == 5
            assert oo.dtreat['data0-data'].size == oo.ddataRef['nch']

            oo.set_dtreat_data0(Dt=[2,3])
            assert oo.dtreat['data0-Dt'][0] >= 2. and oo.dtreat['data0-Dt'][1] <= 3.
            assert oo.dtreat['data0-data'].size == oo.ddataRef['nch']

            oo.set_dtreat_data0()
            assert oo.dtreat['data0-data'] is None
            assert np.allclose(oo.data, oo.ddataRef['data'])

    def test09_dtreat_set_interp_indt(self):
        for oo in self.lobj:
            ind = np.arange(0,oo.nt,10)
            oo.set_dtreat_interp_indt( ind )
            assert oo._dtreat['interp-indt'].sum() == ind.size

            ind = dict([(ii, np.arange(0,oo.nt,5)) for ii in range(0,oo.nch,3)])
            oo.set_dtreat_interp_indt( ind )
            assert type(oo._dtreat['interp-indt']) is dict

            oo.set_dtreat_interp_indt()
            assert oo._dtreat['interp-indt'] is None

    def test10_dtreat_set_interp_indch(self):
        for oo in self.lobj:
            ind = np.arange(0, oo.nch, 10, dtype=int)
            oo.set_dtreat_interp_indch( ind )
            assert oo._dtreat['interp-indch'].sum() == ind.size

            ind = dict([(ii, np.arange(0,oo.nch,5)) for ii in range(0,oo.nt,3)])
            oo.set_dtreat_interp_indch( ind )
            assert type(oo._dtreat['interp-indch']) is dict

            oo.set_dtreat_interp_indch()
            assert oo._dtreat['interp-indch'] is None

    def test11_streat_set_dfit(self):
        for oo in self.lobj:
            oo.set_dtreat_dfit()

    def test12_streat_set_interpt(self):
        t = np.linspace(self.t[0]-0.1, self.t[-1]+0.5, 100)
        for oo in self.lobj:
            oo.set_dtreat_interpt(t)

    def test13_clear_ddata(self):
        for ii in range(0,len(self.lobj)):
            if ii%2 == 0:
                self.lobj[ii].clear_ddata()

    def test14_clear_dtreat(self):
        for ii in range(0,len(self.lobj)):
            if ii%2 == 1:
                self.lobj[ii].clear_dtreat(force=True)


    def test15_plot(self):
        for oo in self.lobj:
            kh = oo.plot(key=None, ntMax=4, nchMax=2, fs=None,
                         dmargin=dict(left=0.06, right=0.9),
                         wintit='test', tit='AHAH')
        plt.close('all')

    def test16_compare(self):
        for oo in self.lobj:
            kh = oo.plot_compare(oo)
        plt.close('all')

    def test17_plot_combine(self):
        for ii in range(1,len(self.lobj)):
            kh = self.lobj[ii].plot_combine(self.lobj[ii-1])
        plt.close('all')

    def test18_spectrogram(self):
        for oo in self.lobj:
            kh = oo.plot_spectrogram(warn=False)
        plt.close('all')

    def test19_plot_svd(self):
        for oo in self.lobj:
            kh = oo.plot_svd()
        plt.close('all')

    def test20_copy_equal(self):
        for oo in self.lobj:
            obj = oo.copy()
            assert obj == oo

    def test21_get_nbytes(self):
        for oo in self.lobj:
            nb, dnb = oo.get_nbytes()

    def test22_strip_nbytes(self, verb=False):
        lok = self.lobj[0].__class__._dstrip['allowed']
        nb = np.full((len(lok),), np.nan)
        for oo in self.lobj:
            for ii in lok:
                oo.strip(ii, verb=verb)
                nb[ii] = oo.get_nbytes()[0]
            assert np.all(np.diff(nb)<=0.), nb
            for ii in lok[::-1]:
                oo.strip(ii, verb=verb)

    def test23_saveload(self, verb=False):
        for oo in self.lobj:
            pfe = oo.save(deep=False, verb=verb, return_pfe=True)
            obj = tfu.load(pfe, verb=verb)
            # Just to check the loaded version works fine
            assert oo == obj
            os.remove(pfe)





class Test02_DataCam12DSpectral(Test01_DataCam12D):

    @classmethod
    def setup_class(cls, nch=30, nt=50, SavePath='./', verb=False):

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

        # lamb
        nlamb = 100
        lamb = np.linspace(10,20,nlamb)
        flamb = np.exp(-(lamb-12)**2/0.1) + 0.4*np.exp(-(lamb-16)**2/0.5)

        # -------
        # signal as Data from lcams
        lm = ['sum', 'simps']
        lData = [None for ii in range(0,len(lc))]
        for ii in range(0,len(lc)):
            sig = lc[ii].calc_signal(emiss, t=t, res=0.01, method=lm[ii],
                                     plot=False, returnas=np.ndarray)[0]
            sig = sig[:,:,None]*flamb[None,None,:]
            cla = eval('tfd.DataCam%sDSpectral'%('2' if lc[ii]._is2D() else '1'))
            data = cla(data=sig, Name='All', Diag='Test',
                       Exp=conf0.Id.Exp, lCam=lc[ii], t=t,
                       lamb=lamb)
            lData[ii] = data

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
        cls.t = t

        # Saving for intermediate use
        lpfe = []
        for oo in cls.lobj:
            if oo._dgeom['config'] is not None:
                lpfe.append( oo._dgeom['config'].save(return_pfe=True,
                                                      verb=verb) )
            if oo._dgeom['lCam'] is not None:
                for cc in oo._dgeom['lCam']:
                    lpfe.append( cc.save(return_pfe=True, verb=verb) )

        cls.lpfe = lpfe

    def test08_dtreat_set_data0(self):
        for oo in self.lobj:
            # Re-initialise
            oo.set_dtreat_indt()
            oo.set_dtreat_mask()

            oo.set_dtreat_data0( data0 = oo.data[0,:,:] )
            assert oo.dtreat['data0-indt'] is None
            assert oo.dtreat['data0-Dt'] is None
            assert np.allclose(oo.data[0,:,:],0.), oo.data[0,:,:]

            oo.set_dtreat_data0(indt=[1,2,6,8,9])
            assert oo.dtreat['data0-indt'].sum() == 5
            assert oo.dtreat['data0-data'].shape == (oo.ddataRef['nch'],
                                                     oo.ddataRef['nlamb'])

            oo.set_dtreat_data0(Dt=[2,3])
            assert oo.dtreat['data0-Dt'][0] >= 2. and oo.dtreat['data0-Dt'][1] <= 3.
            assert oo.dtreat['data0-data'].shape == (oo.ddataRef['nch'],
                                                     oo.ddataRef['nlamb'])

            oo.set_dtreat_data0()
            assert oo.dtreat['data0-data'] is None
            assert np.allclose(oo.data, oo.ddataRef['data'])

    def test17_plot_combine(self):
        pass

    def test18_spectrogram(self):
        pass

    def test19_plot_svd(self):
        pass





"""
    def test12_operators(self):
        o0 = self.lobj[-1]
        o1 = 100.*(o0-0.1*o0)

    def test15_combine(self):
        if self.__class__ is Test02_Data2D:
            return
        connect = (hasattr(plt.get_current_fig_manager(),'toolbar')
                   and plt.get_current_fig_manager().toolbar is not None)
        o0 = self.lobj[0]
        for ii in range(1,len(self.lobj)):
            oo = self.lobj[ii]
            KH = oo.plot_combine(o0, connect=connect)
        plt.close('all')

"""



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
