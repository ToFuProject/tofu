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
import tofu.mesh as tfm


Root = tfpf.Find_Rootpath()
Addpath = '/tests/tests02_mesh/'

VerbHead = 'tfm.'


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
#     Helper functions
#
#######################################################


X1, X2 = 0., 10.
Res1, Res2 = 1., 3.
DRes = 0.001
Mode = 'Larger' # or 'Both'

KnotsL = [(0.,10.),(10.,20.)]
ResL = [(1.,3.),(3.,1.)]
Concat = True
Tol = 1e-14


class test01_Helper:

    @classmethod
    def setup_class(cls, X1=X1, X2=X2, Res1=Res1, Res2=Res2, DRes=DRes, KnotsL=KnotsL, ResL=ResL, Mode=Mode, Concat=Concat, Tol=Tol):
        print ("")
        print "--------- "+VerbHead+cls.__name__
        cls.X1, cls.X2 = X1, X2
        cls.Res1, cls.Res2 = Res1, Res2
        cls.DRes = DRes
        cls.KnotsL = KnotsL
        cls.ResL = ResL
        cls.Mode = Mode
        cls.Concat = Concat
        cls.Tol = Tol

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

    def test01_LinMesh(self):
        xn, Res1, Res2 = tfm.LinMesh(self.X1, self.X2, self.Res1, self.Res2, DRes=self.DRes, Mode=self.Mode, Test=True)

        # Testing types and formats
        assert type(xn) is np.ndarray and xn.ndim==1 and xn.dtype.name=='float64'
        assert type(Res1) is float
        assert type(Res2) is float

        # Testing values
        assert np.all(xn==[0.,1.,2.5,4.5,7.,10.])
        assert Res1==1.
        assert Res2==3.


    def test02_LinMesh_List(self):
        X, Res = tfm.LinMesh_List(self.KnotsL, self.ResL, DRes=self.DRes, Mode=self.Mode, Test=True, Concat=self.Concat, Tol=self.Tol)

        # Testing types and formats
        assert type(X) is np.ndarray and X.ndim==1 and X.dtype.name=='float64'
        assert type(Res) is list and all([hasattr(rr,'__getitem__') and len(rr)==2 for rr in Res])

        # Testing values
        assert np.all(X==[0.,1.,2.5,4.5,7.,10.,13.,15.5,17.5,19.,20.])
        assert Res==[(1.,3.),(3.,1.)]




#######################################################
#
#     Mesh
#
#######################################################


class test02_Mesh1D:

    @classmethod
    def setup_class(cls, KnotsL=KnotsL, ResL=ResL, Mode=Mode, Concat=Concat, Tol=Tol):
        print ("")
        print "--------- "+VerbHead+cls.__name__

        X, Res = tfm.LinMesh_List(KnotsL, ResL, DRes=DRes, Mode=Mode, Test=True, Concat=Concat, Tol=Tol)
        cls.Obj = tfm.Mesh1D('Test', X, Exp='Test', shot=0, dtime=None, dtimeIn=False, SavePath=Root+Addpath)

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
        assert all([hasattr(self.Obj,aa) for aa in ['Id','Knots','NKnots','Cents','NCents','Bary','Lengths','Length',
                                                    'sample','plot','plot_Res','save']])
        # Testing types and formats
        assert type(self.Obj.Knots) is np.ndarray and self.Obj.Knots.ndim==1 and self.Obj.Knots.dtype.name=='float64'
        assert type(self.Obj.NKnots) is int
        assert type(self.Obj.Cents) is np.ndarray and self.Obj.Cents.ndim==1 and self.Obj.Cents.dtype.name=='float64'
        assert type(self.Obj.NCents) is int
        assert type(self.Obj.Bary) is float
        assert type(self.Obj.Lengths) is np.ndarray and self.Obj.Lengths.ndim==1 and self.Obj.Lengths.dtype.name=='float64'
        assert type(self.Obj.Length) is float

        # Testing values
        assert np.all(self.Obj.Knots==[0.,1.,2.5,4.5,7.,10.,13.,15.5,17.5,19.,20.])
        assert self.Obj.NKnots==11
        assert np.all(self.Obj.Cents==[0.5,1.75,3.5,5.75,8.5,11.5,14.25,16.5,18.25,19.5])
        assert self.Obj.NCents==10
        assert self.Obj.Bary==10.
        assert np.all(self.Obj.Lengths==[1.,1.5,2.,2.5,3.,3.,2.5,2.,1.5,1.])
        assert self.Obj.Length==20.

    def test02_sample(self):
        xx = self.Obj.sample(Sub=0.1, SubMode='rel', Test=True)

    def test03_plot(self):
        ax0 = self.Obj.plot(y=0., Elt='KCN', ax=None, draw=False, a4=False, Test=True)
        ax1 = self.Obj.plot(y=10., Elt='K', ax=None, draw=False, a4=False, Test=True)
        plt.close('all')

    def test04_plot_Res(self):
        ax = self.Obj.plot_Res(ax=None, draw=False, a4=False, Test=True)
        plt.close('all')

    def test05_saveload(self):
        self.Obj.save()
        Los = tfpf.Open(self.Obj.Id.SavePath + self.Obj.Id.SaveName + '.npz')
        os.remove(self.Obj.Id.SavePath + self.Obj.Id.SaveName + '.npz')





VesTor = tfpf.Open(Root+'/tests/tests01_geom/TFG_VesTor_AUG_Test_sh0.npz')
VesLin = tfpf.Open(Root+'/tests/tests01_geom/TFG_VesLin_AUG_Test_sh0.npz')

class test03_Mesh2D:

    @classmethod
    def setup_class(cls, Ves=VesTor, DRes=DRes, Mode=Mode, Tol=Tol):
        print ("")
        print "--------- "+VerbHead+cls.__name__

        D1, D2 = Ves._P1Max[0]-Ves._P1Min[0], Ves._P2Max[1]-Ves._P2Min[1]
        K1L = [(Ves._P1Min[0],Ves._P1Min[0]+0.3*D1),(Ves._P1Min[0]+0.3*D1,Ves._P1Min[0]+0.7*D1),(Ves._P1Min[0]+0.7*D1,Ves._P1Max[0])]
        K2L = [(Ves._P2Min[1],Ves._P2Min[1]+0.35*D2),(Ves._P2Min[1]+0.35*D2,Ves._P2Min[1]+0.65*D2),(Ves._P2Min[1]+0.65*D2,Ves._P2Max[1])]
        R1L = [(0.05,0.03),(0.03,0.03),(0.03,0.05)]
        R2L = [(0.10,0.03),(0.03,0.03),(0.03,0.08)]

        X1, R1 = tfm.LinMesh_List(K1L, R1L, DRes=DRes, Mode=Mode, Test=True, Concat=True, Tol=Tol)
        X2, R2 = tfm.LinMesh_List(K2L, R2L, DRes=DRes, Mode=Mode, Test=True, Concat=True, Tol=Tol)
        objT = tfm.Mesh2D('Test', [X1,X2], Type='Tor', Exp='Test', shot=0, dtime=None, dtimeIn=False, SavePath=Root+Addpath)
        objL = tfm.Mesh2D('Test', [X1,X2], Type='Lin', Exp='Test', shot=0, dtime=None, dtimeIn=False, SavePath=Root+Addpath)
        cls.LObj = [objT,objL]
        cls.PolyVes = VesTor.Poly

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
        for oo in self.LObj:
            assert all([hasattr(oo,aa) for aa in ['Id','Knots','NKnots','Cents','NCents','MeshX1','MeshX2','SubMesh','BaryS','Surfs','Surf','VolAngs','VolAng','BaryV','CentsV',
                                                        'add_SubMesh','isInside','sample','plot','plot_Res','save']])
            # Testing types and formats
            assert type(oo.Knots) is np.ndarray and oo.Knots.ndim==2 and oo.Knots.shape[0]==2 and oo.Knots.dtype.name=='float64'
            assert type(oo.NKnots) is int
            assert type(oo.Cents) is np.ndarray and oo.Cents.ndim==2 and oo.Cents.shape[0]==2 and oo.Cents.dtype.name=='float64'
            assert type(oo.NCents) is int
            assert type(oo.BaryS) is np.ndarray and oo.BaryS.shape==(2,) and oo.BaryS.dtype.name=='float64'
            assert oo.BaryV is None or (type(oo.BaryV) is np.ndarray and oo.BaryV.shape==(2,) and oo.BaryV.dtype.name=='float64')
            assert type(oo.MeshX1) is tfm.Mesh1D
            assert type(oo.MeshX2) is tfm.Mesh1D
            assert type(oo.SubMesh) is dict
            assert type(oo.Surfs) is np.ndarray and oo.Surfs.shape==(oo.NCents,) and oo.Surfs.dtype.name=='float64'
            assert type(oo.Surf) is float
            assert oo.VolAngs is None or (type(oo.VolAngs) is np.ndarray and oo.VolAngs.shape==(oo.NCents,) and oo.VolAngs.dtype.name=='float64')
            assert oo.VolAng is None or type(oo.VolAng) is float
            assert oo.CentsV is None or (type(oo.CentsV) is np.ndarray and oo.CentsV.shape==(2,oo.NCents) and oo.CentsV.dtype.name=='float64')

        assert self.LObj[0].BaryV is not None and self.LObj[1].BaryV is None
        assert self.LObj[0].VolAngs is not None and self.LObj[1].VolAngs is None
        assert self.LObj[0].VolAng is not None and self.LObj[1].VolAng is None
        assert self.LObj[0].CentsV is not None and self.LObj[1].CentsV is None

    def test02_add_SubMesh(self):
        for oo in self.LObj:
            oo.add_SubMesh(ind=np.arange(100,oo.NCents-200), InMode='Cents', NLim=1, Name='ind')
            oo.add_SubMesh(Poly=self.PolyVes, InMode='Cents', NLim=1, Name='Poly')

    def test03_isInside(self, NR=10, NZ=20):
        PtsR = np.linspace(np.nanmin(self.PolyVes[0,:]),np.nanmax(self.PolyVes[0,:]),NR)
        PtsZ = np.linspace(np.nanmin(self.PolyVes[1,:]),np.nanmax(self.PolyVes[1,:]),NZ)
        Pts2D = np.array([np.tile(PtsR,(NZ,1)).flatten(), np.tile(PtsZ,(NR,1)).T.flatten()])
        for oo in self.LObj:
            ind = oo.isInside(Pts2D)

    def test04_sample(self):
        for oo in self.LObj:
            Pts = oo.sample(Sub=(0.1,0.05), SubMode='rel', Test=True)


    def test05_plot(self):
        for oo in self.LObj:
            ax1 = oo.plot(ax=None, Elt='MBgKCBsBv', indKnots=None, indCents=None, SubMesh=None, draw=False, a4=False, Test=True)
            ax2 = oo.plot(ax=None, Elt='MBgKCBsBv', indKnots=None, indCents=range(10,1000,10), SubMesh=None, draw=False, a4=False, Test=True)
            ax3 = oo.plot(ax=None, Elt='MBgKCBsBv', indKnots=None, indCents=None, SubMesh='ind', draw=False, a4=False, Test=True)
        #plt.close('all')   # Creates an error during test05_plot_Res() for some reason... bug ?   (">> can't invoke "event" command:  application has been destroyed")


    def test06_plot_Res(self):
        axA = self.LObj[0].plot_Res(Elt='MBgV', SubMesh=None, Leg=None, draw=False, a4=False, Test=True)
        axA = self.LObj[1].plot_Res(Elt='MBgS', SubMesh=None, Leg=None, draw=False, a4=False, Test=True)
        axB = self.LObj[0].plot_Res(Elt='MBgV', SubMesh='Poly', Leg=None, draw=False, a4=False, Test=True)
        axB = self.LObj[1].plot_Res(Elt='MBgS', SubMesh='Poly', Leg=None, draw=False, a4=False, Test=True)
        plt.close('all')


    def test07_saveload(self):
        for oo in self.LObj:
            oo.save()
            OO = tfpf.Open(oo.Id.SavePath + oo.Id.SaveName + '.npz')
            os.remove(oo.Id.SavePath + oo.Id.SaveName + '.npz')






#######################################################
#
#     LBF1D
#
#######################################################



class test04_LBF1D:

    @classmethod
    def setup_class(cls):
        print ("")
        print "--------- "+VerbHead+cls.__name__

        cls.Knots = np.linspace(0.,20.,11)
        LDeg = [0,1,2,3]
        cls.LObj = [tfm.LBF1D('Test', cls.Knots, dd, Exp='Test', shot=0, dtime=None, dtimeIn=False, SavePath=Root+Addpath) for dd in LDeg]

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
        for oo in self.LObj:
            assert all([hasattr(oo,aa) for aa in ['Id','Type','Mesh','Deg','LFunc','NFunc',
                                                  'get_TotFunc','get_TotVal','get_Coefs','get_IntOp','get_IntVal','plot','plot_Ind','save']])
            # Testing types and formats
            assert type(oo.Mesh) is tfm.Mesh1D
            assert type(oo.Deg) is int
            assert type(oo.LFunc) is list and all([hasattr(ff,'__call__') for ff in oo.LFunc])
            assert type(oo.NFunc) is int and len(oo.LFunc)==oo.NFunc

            # Testing values
            assert np.all(oo.Mesh.Knots==self.Knots)
            assert oo.NFunc==len(oo.Mesh.Knots)-1-oo.Deg

    def test02__get_Func_Supps(self):
        for oo in self.LObj:
            Supps = oo._get_Func_Supps()
            assert type(Supps) is np.ndarray and Supps.shape==(2,oo.NFunc) and np.all(Supps[1,:]-Supps[0,:]>0.)

    def test03__get_Func_InterFunc(self):
        for oo in self.LObj:
            indF = oo._get_Func_InterFunc()
            assert type(indF) is np.ndarray and indF.shape==(2*oo.Deg,oo.NFunc)

    def test05_get_TotFunc(self):
        for oo in self.LObj:
            TF = []
            TF.append(oo.get_TotFunc(Deriv=0, Coefs=1., thr=1.e-8, thrmode='rel', Test=True))
            TF.append(oo.get_TotFunc(Deriv=oo.Deg, Coefs=10.*np.ones((oo.NFunc,)), thr=1.e-8, thrmode='rel', Test=True))
            assert hasattr(TF[0],'__call__')
            assert hasattr(TF[1],'__call__')
            if oo.Deg>1:
                TF.append(oo.get_TotFunc(Deriv='D1FI', Coefs=-5.*np.ones((10,oo.NFunc)), thr=1.e-8, thrmode='rel', Test=True))
                assert type(TF[2]) is list and len(TF[2])==10 and all([hasattr(tf,'__call__') for tf in TF[2]])

    def test06_get_TotVal(self):
        Pts = np.linspace(-5.,25.,100.)
        indout = (Pts<0.) | (Pts>20.)
        for oo in self.LObj:
            TV = oo.get_TotVal(Pts, Deriv='D0ME', Coefs=np.ones((10,oo.NFunc)), thr=1.e-8, thrmode='rel', Test=True)
            assert type(TV) is np.ndarray and TV.shape==(10,100) and np.all(np.isnan(TV[:,indout]))
            if oo.Deg>1:
                TV = oo.get_TotVal(Pts, Deriv='D1FI', Coefs=1., thr=1.e-8, thrmode='rel', Test=True)
                assert type(TV) is np.ndarray and TV.shape==(100.,) and np.all(np.isnan(TV[indout]))

    def test07_get_Coefs(self):  # To be finished

        for oo in self.LObj:
            cc, rr = oo.get_Coefs(xx=None, yy=None, ff=None, Sub=tfd.BF1Sub, SubMode=tfd.BF1SubMode, Test=True)






#######################################################
#
#     LBF2D
#
#######################################################









