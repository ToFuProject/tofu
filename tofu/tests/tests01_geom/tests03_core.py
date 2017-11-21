"""
This module contains tests for tofu.geom in its structured version
"""

# External modules
import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


# Nose-specific
from nose import with_setup # optional


# Importing package tofu.geom
from tofu import __version__
import tofu.defaults as tfd
import tofu.pathfile as tfpf
import tofu.geom as tfg


here = os.path.abspath(os.path.dirname(__file__))
VerbHead = 'tofu.geom.tests03_core'


#######################################################
#
#     Setup and Teardown
#
#######################################################

def setup_module(module):
    print("") # this is to get a newline after the dots
    LF = os.listdir(here)
    LF = [lf for lf in LF if all([ss in lf for ss in ['TFG_','Test','_Vv','.npz']])]
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
    LF = [lf for lf in LF if all([ss in lf for ss in ['TFG_','Test','_Vv','.npz']])]
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


#R, r = 1.5, 0.6
#PDiv = [R+r*np.array([-0.4,-1./8.,1./8.,0.4]),r*np.array([-1.3,-1.1,-1.1,-1.3])]
#PVes = np.linspace(-2.*np.pi/5.,7.*np.pi/5., 100)
#PVes = [R + r*np.cos(PVes), r*np.sin(PVes)]
#PVes = np.concatenate((PVes,PDiv),axis=1)

PVes = np.loadtxt(os.path.join(here,'test_Ves.txt'), dtype='float', skiprows=1, ndmin=2, comments='#')
Lim = 2.*np.pi*1.7*np.array([-0.5,0.5])



class Test01_Ves:

    @classmethod
    def setup_class(cls, PVes=PVes, Lim=Lim):
        #print("")
        #print("---- "+cls.__name__)
        cls.LObj = [tfg.Ves('Test', PVes, Type='Tor', shot=0, Exp='Test', SavePath=here)]
        cls.LObj.append(tfg.Ves('Test', PVes, Type='Lin', Lim=Lim, shot=0, Exp='Test', SavePath=here))

    @classmethod
    def teardown_class(cls):
        #print("teardown_class() after any methods in this class")
        cls.VesTor = tfg.Ves('Test00', PVes, Type='Tor', shot=0, Exp='Test', SavePath=here)
        cls.VesLin = tfg.Ves('Test00', PVes, Type='Lin', Lim=Lim, shot=0, Exp='Test', SavePath=here)
        cls.VesTor.save()
        cls.VesLin.save()

    def setup(self):
        #print("TestUM:setup() before each test method")
        pass

    def teardown(self):
        #print("TestUM:teardown() after each test method")
        pass

    def test01_isInside(self, NR=20, NZ=20, NThet=10):
        for ii in range(0,len(self.LObj)):
            PtsR = np.linspace(self.LObj[ii].geom['P1Min'][0],self.LObj[ii].geom['P1Max'][0],NR)
            PtsZ = np.linspace(self.LObj[ii].geom['P2Min'][0],self.LObj[ii].geom['P2Max'][0],NZ)
            PtsRZ = np.array([np.tile(PtsR,(NZ,1)).flatten(), np.tile(PtsZ,(NR,1)).T.flatten()])
            if self.LObj[ii].Type=='Tor' and self.LObj[ii].Lim is None:
                In = '(R,Z)'
            if self.LObj[ii].Type=='Lin':
                PtsRZ = np.concatenate((np.zeros((1,NR*NZ)),PtsRZ),axis=0)
                In = '(X,Y,Z)'
            elif self.LObj[ii].Type=='Tor' and self.LObj[ii].Lim is not None:
                PtsRZ = np.concatenate((PtsRZ,np.zeros((1,NR*NZ))),axis=0)
                In = '(R,Z,Phi)'
            indRZ = self.LObj[ii].isInside(PtsRZ, In=In)
            assert type(indRZ) is np.ndarray
            if self.LObj[ii].Lim is None or self.LObj[ii]._Multi is False:
                assert indRZ.shape==(PtsRZ.shape[1],)
            else:
                assert indRZ.shape==(len(self.LObj[ii].Lim),PtsRZ.shape[1])

    def test02_InsideConvexPoly(self):
        self.LObj[0].get_InsideConvexPoly(Plot=False, Test=True)

    def test03_get_meshEdge(self):
        Pts, dLr, ind = self.LObj[0].get_meshEdge(dL=0.05, DS=None, dLMode='abs', DIn=0.001)
        Pts, dLr, ind = self.LObj[0].get_meshEdge(dL=0.1, DS=None, dLMode='rel', DIn=-0.001)
        Pts, dLr, ind = self.LObj[0].get_meshEdge(dL=0.05, DS=[None,[-2.,0.]], dLMode='abs', DIn=0.)

    def test04_get_meshCross(self):
        Pts, dS, ind, dSr = self.LObj[0].get_meshCross(0.02, DS=None, dSMode='abs', ind=None)
        Pts, dS, ind, dSr = self.LObj[0].get_meshCross(0.02, DS=None, dSMode='abs', ind=ind)
        Pts, dS, ind, dSr = self.LObj[0].get_meshCross(0.1, DS=[[0.,2.5],None], dSMode='rel', ind=None)

    def test05_get_meshS(self):
        for ii in range(0,len(self.LObj)):
            Pts0, dS, ind, dSr = self.LObj[ii].get_meshS(0.02, DS=[[2.,3.],[0.,5.],[0.,np.pi/2.]], dSMode='abs', ind=None, DIn=0.001, Out='(X,Y,Z)')
            Pts1, dS, ind, dSr = self.LObj[ii].get_meshS(0.02, DS=None, dSMode='abs', ind=ind, DIn=0.001, Out='(X,Y,Z)')
            if type(Pts0) is list:
                assert all([np.allclose(Pts0[ii],Pts1[ii]) for ii in range(0,len(Pts0))])
            else:
                assert np.allclose(Pts0,Pts1)

    def test06_get_meshV(self):
        if self.LObj[0].Id.Cls=='Ves':
            LDV = [[[1.,2.],[0.,2.],[3.*np.pi/4.,5.*np.pi/4.]], [[-1.,1.],[1.,2.],[0.,2.]]]
            for ii in range(0,len(self.LObj)):
                Pts, dV, ind, dVr = self.LObj[ii].get_meshV(0.05, DV=LDV[ii], dVMode='abs', ind=None, Out='(R,Z,Phi)')
                Pts, dV, ind, dVr = self.LObj[ii].get_meshV(0.05, DV=None, dVMode='abs', ind=ind, Out='(R,Z,Phi)')

    def test07_plot(self):
        for ii in range(0,len(self.LObj)):
            if self.LObj[ii].Id.Cls=='Ves':
                Pdict = {'c':'k'}
            else:
                Pdict = {'ec':'None','fc':(0.8,0.8,0.8,0.5)}
            Lax1 = self.LObj[ii].plot(Proj='All', Elt='PIBsBvV', Pdict=Pdict, draw=False, a4=False, Test=True)
            Lax2 = self.LObj[ii].plot(Proj='Cross', Elt='PIBsBvV', Pdict=Pdict, draw=False, a4=False, Test=True)
            Lax3 = self.LObj[ii].plot(Proj='Hor', Elt='PIBsBvV', Pdict=Pdict, draw=False, a4=False, Test=True)
            plt.close('all')

    def test08_plot_sino(self):
        Lax1 = self.LObj[0].plot_sino(Proj='Cross', Ang='xi', AngUnit='deg', Sketch=True, draw=False, a4=False, Test=True)
        Lax2 = self.LObj[0].plot_sino(Proj='Cross', Ang='theta', AngUnit='rad', Sketch=True, draw=False, a4=False, Test=True)
        plt.close('all')

    def test09_saveload(self):
        for ii in range(0,len(self.LObj)):
            self.LObj[ii].save(Print=False)
            PathFileExt = os.path.join(self.LObj[ii].Id.SavePath, self.LObj[ii].Id.SaveName+'.npz')
            obj = tfpf.Open(PathFileExt, Print=False)
            os.remove(PathFileExt)


#######################################################
#
#  Creating Struct objects and testing methods
#
#######################################################

class Test02_Struct(Test01_Ves):

    @classmethod
    def setup_class(cls, PVes=PVes, Lim=Lim):
        #print("")
        #print("--------- "+VerbHead+cls.__name__)
        cls.LObj = [tfg.Struct('Test02', PVes, Type='Tor', shot=0, Exp='Test', SavePath=here)]
        cls.LObj.append(tfg.Struct('Test', PVes, Type='Tor', Lim=[-np.pi/2.,np.pi/4.], shot=0, Exp='Test', SavePath=here))
        cls.LObj.append(tfg.Struct('Test', PVes, Type='Lin', Lim=Lim, shot=0, Exp='Test', SavePath=here))
        cls.LObj.append(tfg.Struct('Test', PVes, Type='Tor', Lim=np.pi*np.array([[0.,1/4.],[3./4.,5./4.],[-1./2,0.]]), shot=0, Exp='Test', SavePath=here))
        cls.LObj.append(tfg.Struct('Test', PVes, Type='Lin', Lim=np.array([[0.,1.],[0.5,1.5],[-2.,-1.]]), shot=0, Exp='Test', SavePath=here))

    @classmethod
    def teardown_class(cls):
        #print("teardown_class() after any methods in this class")
        cls.SL0 = tfg.Struct('Test02', PVes, Type='Lin', Lim=np.array([0.,1.]), shot=0, Exp='Test', SavePath=here)
        cls.SL1 = tfg.Struct('Test03', PVes, Type='Lin', Lim=np.array([[0.,1/4.],[3./4.,5./4.],[-1./2,0.]]), shot=0, Exp='Test', SavePath=here)
        cls.ST0 = tfg.Struct('Test02', PVes, Type='Tor', Lim=None, shot=0, Exp='Test', SavePath=here)
        cls.ST1 = tfg.Struct('Test03', PVes, Type='Tor', Lim=np.pi*np.array([[0.,1/4.],[3./4.,5./4.],[-1./2,0.]]), shot=0, Exp='Test', SavePath=here)
        cls.SL0.save(), cls.SL1.save()
        cls.ST0.save(), cls.ST1.save()



#######################################################
#
#     Creating LOS objects and testing methods
#
#######################################################


class Test03_LOS:

    @classmethod
    def setup_class(cls): #LVes=[Test01_Ves.VesLin]*3+[Test01_Ves.VesTor]*3, LS=[None, SL0, [SL0,SL1], None, ST0, [ST0,ST1]]):
        #print ("")
        #print "--------- "+VerbHead+cls.__name__
        LVes = [Test01_Ves.VesLin]*3+[Test01_Ves.VesTor]*3
        LS = [None, Test02_Struct.SL0, [Test02_Struct.SL0,Test02_Struct.SL1], None, Test02_Struct. ST0, [Test02_Struct.ST0,Test02_Struct.ST1]]
        cls.LObj = [None for vv in LVes]
        for ii in range(0,len(LVes)):
            D = (0,0.95*LVes[ii].geom['P1Max'][0], 0)
            u = (0,1,0)
            cls.LObj[ii] = tfg.LOS('Test'+str(ii), (D,u), Ves=LVes[ii], LStruct=LS[ii], Exp=None, Diag='Test', SavePath=here)

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

    def test01_get_mesh(self):
        for ii in range(0,len(self.LObj)):
            out = self.LObj[ii].get_mesh(0.01, dLMode='abs')
            assert np.all((out[1]>=self.LObj[ii].geom['kPIn']) & (out[1]<=self.LObj[ii].geom['kPOut']))
            assert np.abs(out[2]-0.01)<0.001
            out = self.LObj[ii].get_mesh(0.1, dLMode='rel')
            assert out[0].shape[1]==out[1].size
            assert out[1].size==10

    def test02_calc_signal(self):
        for ii in range(0,len(self.LObj)):
            if self.LObj[ii].Ves.Type=='Tor':
                ff1 = lambda Pts: np.exp(-(np.hypot(Pts[0,:],Pts[1,:]))**2/0.1-(Pts[2,:])**2/0.1)
                ff2 = lambda Pts, Vect: np.cos(np.arctan2(Vect[2,:],Vect[1,:]))*np.exp(-(np.hypot(Pts[0,:],Pts[1,:]))**2/0.1-(Pts[2,:])**2/0.1)
            else:
                ff1 = lambda Pts: np.exp(-(Pts[1,:])**2/0.1-(Pts[2,:])**2/0.1)
                ff2 = lambda Pts, Vect: np.cos(np.arctan2(Vect[2,:],Vect[1,:]))*np.exp(-(Pts[1,:])**2/0.1-(Pts[2,:])**2/0.1)
            out1 = self.LObj[ii].calc_signal(ff1)
            out2 = self.LObj[ii].calc_signal(ff2)
            assert all([not hasattr(oo,'__iter__') for oo in [out1,out2]])
            assert not out1==out2

    def test03_plot(self):
        for ii in range(0,len(self.LObj)):
            Lax1 = self.LObj[ii].plot(Proj='All', Elt='LDIORP', EltVes='P', Leg='', draw=False, a4=False, Test=True)
            Lax2 = self.LObj[ii].plot(Proj='Cross', Elt='LDIORP', EltVes='P', Leg='Test', draw=False, a4=False, Test=True)
            Lax3 = self.LObj[ii].plot(Proj='Hor', Elt='LDIORP', EltVes='PBv', Leg='', draw=False, a4=False, Test=True)
            #Lax3 = self.LObj[ii].plot(Proj='3d', Elt='LDIORP', EltVes='PBv', Leg='', draw=False, a4=False, Test=True)
            plt.close('all')

    def test04_plot_sino(self):
        for ii in range(0,len(self.LObj)):
            self.LObj[ii].plot_sino(Proj='Cross', Elt='LV', Sketch=True, Ang='xi', AngUnit='rad', draw=False, a4=False, Test=True)
            self.LObj[ii].plot_sino(Proj='Cross', Elt='L', Sketch=False, Ang='theta', AngUnit='deg', draw=False, a4=False, Test=True)
            plt.close('all')

    def test05_saveload(self):
        print(os.listdir(here))
        for ii in range(0,len(self.LObj)):
            self.LObj[ii].save(Print=False)
            PathFileExt = os.path.join(self.LObj[ii].Id.SavePath, self.LObj[ii].Id.SaveName+'.npz')
            obj = tfpf.Open(PathFileExt, Print=False)
            os.remove(PathFileExt)



"""
#######################################################
#
#  Creating LOS and GLOS objects and testing methods
#
#######################################################



class Test05_LOSTor:

    @classmethod
    def setup_class(cls, Ves=VesTor):
        print ("")
        print "--------- "+VerbHead+cls.__name__
        DR, DZ = Ves._P1Max[0],Ves._P2Max[1]
        uR, uz = [-1.,-1.]
        Dthet, uthet = 0., 0.5
        er, ethet = np.array([np.cos(Dthet), np.sin(Dthet)]), np.array([-np.sin(Dthet), np.cos(Dthet)])
        D, u = [DR*er[0], DR*er[1], DZ], [uR*er[0]+uthet*ethet[0],   uR*er[1]+uthet*ethet[1], uz]
        cls.Obj = tfg.LOS('Test', (D,u), Ves=Ves, shot=0, Diag='Test', Exp='AUG', SavePath=here)

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

    def test01_plot(self):
        Lax1 = self.Obj.plot(Proj='All', Elt='LDIORP', EltVes='P', Leg='', draw=False, a4=False, Test=True)
        Lax2 = self.Obj.plot(Proj='Cross', Elt='LDIORP', EltVes='P', Leg='Test', draw=False, a4=False, Test=True)
        Lax3 = self.Obj.plot(Proj='Hor', Elt='LDIORP', EltVes='PBv', Leg='', draw=False, a4=False, Test=True)
        plt.close('all')

    def test02_plot_Sinogram(self):
        Lax1 = self.Obj.plot_Sinogram(Proj='Cross', Elt='LV', Sketch=True, Ang='xi', AngUnit='rad', draw=False, a4=False, Test=True)
        Lax2 = self.Obj.plot_Sinogram(Proj='Cross', Elt='L', Sketch=False, Ang='theta', AngUnit='deg', draw=False, a4=False, Test=True)
        plt.close('all')

    def test03_saveload(self):
        self.Obj.save()
        Los = tfpf.Open(self.Obj.Id.SavePath + self.Obj.Id.SaveName + '.npz')
        os.remove(self.Obj.Id.SavePath + self.Obj.Id.SaveName + '.npz')


class Test06_LOSLin:

    @classmethod
    def setup_class(cls, Ves=VesLin):
        print ("")
        print "--------- "+VerbHead+cls.__name__
        DY, DZ = Ves._P1Max[0],Ves._P2Max[1]
        D, u = [0.5, DY, DZ], [0.1, -1., -1.]
        cls.Obj = tfg.LOS('Test', (D,u), Ves=Ves, shot=0, Diag='Test', Exp='AUG', SavePath=Root+Addpath)

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

    def test01_plot(self):
        Lax1 = self.Obj.plot(Proj='All', Elt='LDIORP', EltVes='P', Leg='', draw=False, a4=False, Test=True)
        Lax2 = self.Obj.plot(Proj='Cross', Elt='LDIORP', EltVes='P', Leg='Test', draw=False, a4=False, Test=True)
        Lax3 = self.Obj.plot(Proj='Hor', Elt='LDIORP', EltVes='PBv', Leg='', draw=False, a4=False, Test=True)
        plt.close('all')

    def test02_plot_Sinogram(self):
        Lax1 = self.Obj.plot_Sinogram(Proj='Cross', Elt='LV', Sketch=True, Ang='xi', AngUnit='rad', draw=False, a4=False, Test=True)
        Lax2 = self.Obj.plot_Sinogram(Proj='Cross', Elt='L', Sketch=False, Ang='theta', AngUnit='deg', draw=False, a4=False, Test=True)
        plt.close('all')

    def test03_saveload(self):
        self.Obj.save()
        Los = tfpf.Open(self.Obj.Id.SavePath + self.Obj.Id.SaveName + '.npz')
        os.remove(self.Obj.Id.SavePath + self.Obj.Id.SaveName + '.npz')




class Test07_GLOSTor:

    @classmethod
    def setup_class(cls, Ves=VesTor):
        print ("")
        print "--------- "+VerbHead+cls.__name__
        DR, DZ = Ves._P1Max[0],Ves._P2Max[1]
        uR, uz1, uz2, uz3 = -1., -0.5,-1.,-1.5
        Dthet, uthet = 0., 0.5
        er, ethet = np.array([np.cos(Dthet), np.sin(Dthet)]), np.array([-np.sin(Dthet), np.cos(Dthet)])
        D = [DR*er[0], DR*er[1], DZ]
        u1, u2, u3 = [uR*er[0]+uthet*ethet[0],uR*er[1]+uthet*ethet[1],uz1], [uR*er[0]+uthet*ethet[0],uR*er[1]+uthet*ethet[1],uz2], [uR*er[0]+uthet*ethet[0],uR*er[1]+uthet*ethet[1],uz3]
        L1 = tfg.LOS('Test1', (D,u1), Ves=Ves, shot=0, Diag='Test', Exp='AUG', SavePath=Root+Addpath)
        L2 = tfg.LOS('Test2', (D,u2), Ves=Ves, shot=0, Diag='Test', Exp='AUG', SavePath=Root+Addpath)
        L3 = tfg.LOS('Test3', (D,u3), Ves=Ves, shot=0, Diag='Test', Exp='AUG', SavePath=Root+Addpath)
        cls.Obj = tfg.GLOS('Test', [L1,L2,L3], shot=0, Diag='Test', Exp='AUG', SavePath=Root+Addpath)

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

    def test01_select(self):
        ind = self.Obj.select(Val='Test1', Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='In', Out=bool)
        ind = self.Obj.select(Val='Test2', Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='In', Out=int)
        ind = self.Obj.select(Val='Test3', Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='Out', Out=int)

    def test02_plot(self):
        Lax1 = self.Obj.plot(Proj='All', Elt='LDIORP', EltVes='P', Leg='', draw=False, a4=False, Test=True)
        Lax2 = self.Obj.plot(Proj='Cross', Elt='LDIORP', EltVes='P', Leg='Test', draw=False, a4=False, Test=True)
        Lax3 = self.Obj.plot(Proj='Hor', Elt='LDIORP', EltVes='PBv', Leg='', draw=False, a4=False, Test=True)
        plt.close('all')

    def test03_plot_Sinogram(self):
        Lax1 = self.Obj.plot_Sinogram(Proj='Cross', Elt='LV', Sketch=True, Ang='xi', AngUnit='rad', draw=False, a4=False, Test=True)
        Lax2 = self.Obj.plot_Sinogram(Proj='Cross', Elt='L', Sketch=False, Ang='theta', AngUnit='deg', draw=False, a4=False, Test=True)
        plt.close('all')

    def test04_saveload(self):
        self.Obj.save()
        Los = tfpf.Open(self.Obj.Id.SavePath + self.Obj.Id.SaveName + '.npz')
        os.remove(self.Obj.Id.SavePath + self.Obj.Id.SaveName + '.npz')



class Test08_GLOSLin:

    @classmethod
    def setup_class(cls, Ves=VesLin):
        print ("")
        print "--------- "+VerbHead+cls.__name__
        DY, DZ = Ves._P1Max[0],Ves._P2Max[1]
        D = [0.5, DY, DZ]
        u1, u2, u3 = [0.1,-1.,-1.], [-0.1,-1.,-1.], [0.,-0.5,-1.]
        L1 = tfg.LOS('Test1', (D,u1), Ves=Ves, shot=0, Diag='Test', Exp='AUG', SavePath=Root+Addpath)
        L2 = tfg.LOS('Test2', (D,u2), Ves=Ves, shot=0, Diag='Test', Exp='AUG', SavePath=Root+Addpath)
        L3 = tfg.LOS('Test3', (D,u3), Ves=Ves, shot=0, Diag='Test', Exp='AUG', SavePath=Root+Addpath)
        cls.Obj = tfg.GLOS('Test', [L1,L2,L3], shot=0, Diag='Test', Exp='AUG', SavePath=Root+Addpath)
        pass

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

    def test01_select(self):
        ind = self.Obj.select(Val='Test1', Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='In', Out=bool)
        ind = self.Obj.select(Val='Test2', Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='In', Out=int)
        ind = self.Obj.select(Val='Test3', Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='Out', Out=int)

    def test02_plot(self):
        Lax1 = self.Obj.plot(Proj='All', Elt='LDIORP', EltVes='P', Leg='', draw=False, a4=False, Test=True)
        Lax2 = self.Obj.plot(Proj='Cross', Elt='LDIORP', EltVes='P', Leg='Test', draw=False, a4=False, Test=True)
        Lax3 = self.Obj.plot(Proj='Hor', Elt='LDIORP', EltVes='PBv', Leg='', draw=False, a4=False, Test=True)
        plt.close('all')

    def test03_plot_Sinogram(self):
        Lax1 = self.Obj.plot_Sinogram(Proj='Cross', Elt='LV', Sketch=True, Ang='xi', AngUnit='rad', draw=False, a4=False, Test=True)
        Lax2 = self.Obj.plot_Sinogram(Proj='Cross', Elt='L', Sketch=False, Ang='theta', AngUnit='deg', draw=False, a4=False, Test=True)
        plt.close('all')

    def test04_saveload(self):
        self.Obj.save()
        Los = tfpf.Open(self.Obj.Id.SavePath + self.Obj.Id.SaveName + '.npz')
        os.remove(self.Obj.Id.SavePath + self.Obj.Id.SaveName + '.npz')







#######################################################
#
#  Creating Lens and Apert objects and testing methods
#
#######################################################



class Test09_LensTor:

    @classmethod
    def setup_class(cls, Ves=VesTor):
        print ("")
        print "--------- "+VerbHead+cls.__name__
        O = np.array([0.,Ves._P1Max[0],Ves._P2Max[1]])
        nIn = np.array([0.,-1.,-1.])
        Rad, F1 = 0.05, 0.05
        cls.Obj = tfg.Lens('Test', O, nIn, Rad, F1, Ves=Ves, Type='Sph', Exp='AUG', Diag='Test', shot=0, SavePath=Root+Addpath)

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

    def test01_plot_alone(self):
        Lax = self.Obj.plot_alone(V='red', nin=1.5, nout=1., Lmax='F', V_NP=50, src=None, draw=False, a4=False, Test=True)
        Lax = self.Obj.plot_alone(V='red', nin=1.5, nout=1., Lmax='F', V_NP=50, src={'Pt':[0.1,0.005], 'Type':'Pt', 'nn':[-1.,-0.005], 'NP':10}, draw=False, a4=False, Test=True)
        Lax = self.Obj.plot_alone(V='red', nin=1.5, nout=1., Lmax='F', V_NP=50, src={'Pt':[0.1,0.005], 'Type':'Lin', 'nn':[-1.,-0.005], 'NP':10}, draw=False, a4=False, Test=True)
        plt.close('all')

    def test02_plot(self):
        Lax = self.Obj.plot(Proj='All', Elt='PV', EltVes='P', draw=False, a4=False, Test=True)
        Lax = self.Obj.plot(Proj='Cross', Elt='P', EltVes='', draw=False, a4=False, Test=True)
        Lax = self.Obj.plot(Proj='Hor', Elt='V', EltVes='P', draw=False, a4=False, Test=True)
        plt.close('all')

    def test03_saveload(self):
        self.Obj.save()
        obj = tfpf.Open(self.Obj.Id.SavePath + self.Obj.Id.SaveName + '.npz')
        os.remove(self.Obj.Id.SavePath + self.Obj.Id.SaveName + '.npz')


class Test10_LensLin:

    @classmethod
    def setup_class(cls, Ves=VesLin):
        print ("")
        print "--------- "+VerbHead+cls.__name__
        O = np.array([0.5,Ves._P1Max[0],Ves._P2Max[1]])
        nIn = np.array([0.,-1.,-1.])
        Rad, F1 = 0.05, 0.05
        cls.Obj = tfg.Lens('Test', O, nIn, Rad, F1, Ves=Ves, Type='Sph', Exp='AUG', Diag='Test', shot=0, SavePath=Root+Addpath)

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

    def test01_plot_alone(self):
        Lax = self.Obj.plot_alone(V='red', nin=1.5, nout=1., Lmax='F', V_NP=50, src=None, draw=False, a4=False, Test=True)
        Lax = self.Obj.plot_alone(V='red', nin=1.5, nout=1., Lmax='F', V_NP=50, src={'Pt':[0.1,0.005], 'Type':'Pt', 'nn':[-1.,-0.005], 'NP':10}, draw=False, a4=False, Test=True)
        Lax = self.Obj.plot_alone(V='red', nin=1.5, nout=1., Lmax='F', V_NP=50, src={'Pt':[0.1,0.005], 'Type':'Lin', 'nn':[-1.,-0.005], 'NP':10}, draw=False, a4=False, Test=True)
        plt.close('all')

    def test02_plot(self):
        Lax = self.Obj.plot(Proj='All', Elt='PV', EltVes='P', draw=False, a4=False, Test=True)
        Lax = self.Obj.plot(Proj='Cross', Elt='P', EltVes='', draw=False, a4=False, Test=True)
        Lax = self.Obj.plot(Proj='Hor', Elt='V', EltVes='P', draw=False, a4=False, Test=True)
        plt.close('all')

    def test03_saveload(self):
        self.Obj.save()
        obj = tfpf.Open(self.Obj.Id.SavePath + self.Obj.Id.SaveName + '.npz')
        os.remove(self.Obj.Id.SavePath + self.Obj.Id.SaveName + '.npz')




class Test11_ApertTor:

    @classmethod
    def setup_class(cls, Ves=VesLin):
        print ("")
        print "--------- "+VerbHead+cls.__name__
        O = np.array([0.,Ves._P1Max[0],0.5*Ves._P2Max[1]])
        Poly = np.array([O[0] + 0.01*np.array([-1,1,1,-1]), O[1] + np.zeros((4,)), O[2] + 0.005*np.array([-1,-1,1,1])])
        cls.Obj = tfg.Apert('Test', Poly, Ves=Ves, Exp='AUG', Diag='Test', shot=0, SavePath=Root+Addpath)

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

    def test01_plot(self):
        Lax = self.Obj.plot(Proj='All', Elt='PV', EltVes='P', draw=False, a4=False, Test=True)
        Lax = self.Obj.plot(Proj='Cross', Elt='P', EltVes='', draw=False, a4=False, Test=True)
        Lax = self.Obj.plot(Proj='Hor', Elt='V', EltVes='P', draw=False, a4=False, Test=True)
        plt.close('all')

    def test02_saveload(self):
        self.Obj.save()
        obj = tfpf.Open(self.Obj.Id.SavePath + self.Obj.Id.SaveName + '.npz')
        os.remove(self.Obj.Id.SavePath + self.Obj.Id.SaveName + '.npz')


class Test12_ApertLin:

    @classmethod
    def setup_class(cls, Ves=VesLin):
        print ("")
        print "--------- "+VerbHead+cls.__name__
        O = np.array([0.5,Ves._P1Max[0],0.5*Ves._P2Max[1]])
        Poly = np.array([O[0] + 0.01*np.array([-1,1,1,-1]), O[1] + np.zeros((4,)), O[2] + 0.005*np.array([-1,-1,1,1])])
        cls.Obj = tfg.Apert('Test', Poly, Ves=Ves, Exp='AUG', Diag='Test', shot=0, SavePath=Root+Addpath)


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

    def test01_plot(self):
        Lax = self.Obj.plot(Proj='All', Elt='PV', EltVes='P', draw=False, a4=False, Test=True)
        Lax = self.Obj.plot(Proj='Cross', Elt='P', EltVes='', draw=False, a4=False, Test=True)
        Lax = self.Obj.plot(Proj='Hor', Elt='V', EltVes='P', draw=False, a4=False, Test=True)
        plt.close('all')

    def test02_saveload(self):
        self.Obj.save()
        obj = tfpf.Open(self.Obj.Id.SavePath + self.Obj.Id.SaveName + '.npz')
        os.remove(self.Obj.Id.SavePath + self.Obj.Id.SaveName + '.npz')



###########################################################
#
#  Creating Detect objects and testing methods (Apert and Lens, Tor and Lin)
#
###########################################################



class Test13_DetectApertTor:

    @classmethod
    def setup_class(cls, Ves=VesTor):
        print ("")
        print "--------- "+VerbHead+cls.__name__
        Out = np.load(Root+Addpath+'L_009.npz')
        Poly, PAp0, PAp1 = Out['Poly'], Out['PolyAp0'], Out['PolyAp1']
        Ap0 = tfg.Apert('Test0', PAp0, Ves=Ves, Exp='AUG', Diag='Test', shot=0, SavePath=Root+Addpath)
        Ap1 = tfg.Apert('Test1', PAp1, Ves=Ves, Exp='AUG', Diag='Test', shot=0, SavePath=Root+Addpath)
        cls.Obj = tfg.Detect('Test', Poly, Optics=[Ap0,Ap1], Ves=Ves, Exp='AUG', Diag='Test', shot=0, SavePath=Root+Addpath,
                             Etend_Method='quad', Etend_RelErr=1.e-2,
                             Cone_DRY=0.005, Cone_DXTheta=np.pi/512., Cone_DZ=0.005)

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


    def test01_refine_ConePoly(self):
        self.Obj.refine_ConePoly( Proj='Cross', indPoly=0, Verb=False)

    def test02_isInside(self, NR=10,NZ=10):
        R = np.linspace(self.Obj.Ves._P1Min[0],self.Obj.Ves._P1Max[0],NR)
        Z = np.linspace(self.Obj.Ves._P2Min[0],self.Obj.Ves._P2Max[0],NZ)
        Pts = np.array([np.tile(R,(NZ,1)).flatten(), np.tile(Z,(NR,1)).T.flatten()])
        indRZ = self.Obj.isInside(Pts, In='(R,Z)', Test=True)

        Thet = np.pi/4.
        Pts = np.array([Pts[0,:]*np.cos(Thet), Pts[0,:]*np.sin(Thet), Pts[1,:]])
        ind = self.Obj.isInside(Pts, In='(X,Y,Z)', Test=True)
        assert np.all(indRZ==ind)


    def test03_calc_SAngVect(self, NR=10,NZ=10):
        R = np.linspace(self.Obj.Ves._P1Min[0],self.Obj.Ves._P1Max[0],NR)
        Z = np.linspace(self.Obj.Ves._P2Min[0],self.Obj.Ves._P2Max[0],NZ)
        Pts = np.array([np.tile(R,(NZ,1)).flatten(), np.tile(Z,(NR,1)).T.flatten()])
        SAngRZ1, VectRZ1 = self.Obj.calc_SAngVect(Pts, In='(R,Z)', Colis=False, Test=True)
        SAngRZ, VectRZ = self.Obj.calc_SAngVect(Pts, In='(R,Z)', Colis=True, Test=True)

        Thet = np.arctan2(self.Obj.LOS[self.Obj._LOSRef]['PRef'][1],self.Obj.LOS[self.Obj._LOSRef]['PRef'][0])
        Pts = np.array([Pts[0,:]*np.cos(Thet), Pts[0,:]*np.sin(Thet), Pts[1,:]])
        SAng1, Vect1 = self.Obj.calc_SAngVect(Pts, In='(X,Y,Z)', Colis=False, Test=True)
        SAng, Vect = self.Obj.calc_SAngVect(Pts, In='(X,Y,Z)', Colis=True, Test=True)
        ind1RZ0, ind10 = SAngRZ1>0, SAng1>0.
        indRZ0, ind0 = SAngRZ>0., SAng>0.
        assert np.all(ind1RZ0==ind10)
        assert np.all(indRZ0==ind0)
        assert np.all(SAngRZ1==SAng1) and np.all(VectRZ1[:,ind10]==Vect1[:,ind10])
        assert np.all(SAngRZ==SAng) and np.all(VectRZ[:,ind0]==Vect[:,ind0])

    def test04_calc_Sig(self):
        func = lambda Pts, A=1.: A*np.exp(-(((np.hypot(Pts[0,:],Pts[1,:])-1.7)/0.3)**2 + ((Pts[2,:]-0.)/0.5)**2))
        Sig1 = self.Obj.calc_Sig(func, extargs={'A':1.}, Method='Vol', Mode='simps', PreComp=True, Colis=True, dX12=[0.005, 0.005], dX12Mode='abs', ds=0.005, dsMode='abs', MarginS=0.001)
        Sig2 = self.Obj.calc_Sig(func, extargs={'A':1.}, Method='Vol', Mode='sum', PreComp=False, Colis=True, dX12=[0.005, 0.005], dX12Mode='abs', ds=0.005, dsMode='abs', MarginS=0.001)
        Sig3 = self.Obj.calc_Sig(func, extargs={'A':1.}, Method='LOS', Mode='trapz', PreComp=True, Colis=True, ds=0.005, dsMode='abs', MarginS=0.001)
        Sig4 = self.Obj.calc_Sig(func, extargs={'A':1.}, Method='LOS', Mode='quad', PreComp=False, Colis=True, epsrel=1.e-4, MarginS=0.001)
        assert np.abs(Sig1-Sig2)<0.001*min(Sig1,Sig2), str(Sig1)+" vs "+str(Sig2)
        assert np.abs(Sig3-Sig4)<0.001*min(Sig3,Sig4), str(Sig3)+" vs "+str(Sig4)


    def test05_debug_Etendue_BenchmarkRatioMode(self):
        Etends, Ratio, RelErr, dX12, dX12Mode, Colis = self.Obj._debug_Etendue_BenchmarkRatioMode(RelErr=1.e-3, Ratio=[0.01,0.1], Modes=['simps','quad'], dX12=[0.002,0.002], dX12Mode='abs', Colis=True)
        for kk in Etends.keys():
            assert all([np.abs(ee-Etends[kk][0,0])<0.01*Etends[kk][0,0] for ee in Etends[kk][0,:]]), str(Etends[kk][0,:])

    #def test06_calc_Etendue_AlongLOS(self):
    #    Etends, Ps, k, LOSRef = self.Obj.calc_Etendue_AlongLOS(Length='', NP=5, kMode='rel', Modes=['trapz','quad'], RelErr=1.e-3, dX12=[0.005, 0.005], dX12Mode='abs', Ratio=0.02, Colis=True, Test=True)


    #def test07_calc_SAngNb(self):
    #    SA, ind, Pts = self.Obj.calc_SAngNb(Pts=None, Proj='Cross', Slice='Int', DRY=None, DXTheta=None, DZ=None, Colis=True)
    #    SA, ind, Pts = self.Obj.calc_SAngNb(Pts=None, Proj='Hor', Slice='Int', DRY=None, DXTheta=None, DZ=None, Colis=True)
    #    SA, ind, Pts = self.Obj.calc_SAngNb(Pts=None, Proj='Cross', Slice='Int', DRY=None, DXTheta=None, DZ=None, Colis=False)
    #    SA, ind, Pts = self.Obj.calc_SAngNb(Pts=None, Proj='Hor', Slice='Int', DRY=None, DXTheta=None, DZ=None, Colis=False)


    def test08_set_Res(self):
        self.Obj._set_Res(CrossMesh=[0.1,0.1], CrossMeshMode='abs', Mode='Iso', Amp=1., Deg=0, steps=0.01, Thres=0.05, ThresMode='rel', ThresMin=0.01,
                          IntResCross=[0.1,0.1], IntResCrossMode='rel', IntResLong=0.1, IntResLongMode='rel', Eq=None, Ntt=50, EqName=None, save=False, Test=True)


    def test09_plot(self):
        Lax = self.Obj.plot(Proj='All', Elt='PV', EltVes='P', draw=False, a4=False, Test=True)
        Lax = self.Obj.plot(Proj='Cross', Elt='PC', EltVes='', draw=False, a4=False, Test=True)
        Lax = self.Obj.plot(Proj='Hor', Elt='PVC', EltVes='P', draw=False, a4=False, Test=True)
        plt.close('all')


    def test10_plot_SAngNb(self):
        Lax = self.Obj.plot_SAngNb(Lax=None, Proj='Cross', Slice='Int', Pts=None, plotfunc='scatter', DRY=None, DXTheta=None, DZ=None, Elt='P', EltVes='P', EltLOS='L', EltOptics='P', Colis=True, a4=False, draw=False, Test=True)
        Lax = self.Obj.plot_SAngNb(Lax=None, Proj='Hor', Slice='Max', Pts=None, plotfunc='scatter', DRY=None, DXTheta=None, DZ=None, Elt='P', EltVes='P', EltLOS='', EltOptics='P', Colis=True, a4=False, draw=False, Test=True)
        plt.close('all')

    def test11_debug_plot_SAng_OnPlanePerp(self):
        ax = self.Obj._debug_plot_SAng_OnPlanePerp(ax=None, Pos=tfd.DetSAngPlRa, dX12=tfd.DetSAngPldX12, dX12Mode=tfd.DetSAngPldX12Mode, Ratio=tfd.DetSAngPlRatio, Colis=True, draw=False, a4=False, Test=True)
        plt.close('all')

    def test12_plot_Etend_AlongLOS(self):
        ax = self.Obj.plot_Etend_AlongLOS(ax=None, NP=5, kMode='rel', Modes=['trapz','simps'], dX12=[0.005, 0.005], dX12Mode='abs', Ratio=0.02, LOSPts=True, Colis=True, draw=False, a4=True, Test=True)
        ax = self.Obj.plot_Etend_AlongLOS(ax=None, NP=5, kMode='rel', Modes=['quad'], RelErr=1.e-3, Ratio=0.01, LOSPts=True, Colis=False, draw=False, a4=False, Test=True)
        plt.close('all')

    def test13_plot_Sinogram(self):
        ax = self.Obj.plot_Sinogram(ax=None, Proj='Cross', Elt='DLV', Ang='theta', AngUnit='rad', Sketch=True, LOSRef=None, draw=False, a4=False, Test=True)
        ax = self.Obj.plot_Sinogram(ax=None, Proj='Cross', Elt='DLV', Ang='xi', AngUnit='deg', Sketch=False, LOSRef=None, draw=False, a4=True, Test=True)
        plt.close('all')

    def test14_plot_Res(self):
        ax = self.Obj._plot_Res(ax=None, plotfunc='scatter', NC=20, draw=False, a4=False, Test=True)
        ax = self.Obj._plot_Res(ax=None, plotfunc='contourf', NC=20, draw=False, a4=True, Test=True)
        plt.close('all')

    def test15_saveload(self):
        self.Obj.save()
        obj = tfpf.Open(self.Obj.Id.SavePath + self.Obj.Id.SaveName + '.npz')
        os.remove(self.Obj.Id.SavePath + self.Obj.Id.SaveName + '.npz')






def shiftthet(poly, thetobj=0.):
    thet, R = np.arctan2(poly[1,:],poly[0,:]), np.hypot(poly[1,:],poly[0,:])
    extthet = thetobj-np.mean(thet)
    return np.array([R*np.cos(thet+extthet), R*np.sin(thet+extthet), poly[2,:]])



class Test14_DetectApertLin:

    @classmethod
    def setup_class(cls, Ves=VesLin):
        print ("")
        print "--------- "+VerbHead+cls.__name__
        Out = np.load(Root+Addpath+'G_015.npz')
        Poly, PAp0, PAp1 = Out['Poly'], Out['PolyAp0'], Out['PolyAp1']

        Poly = shiftthet(Poly, thetobj=np.pi/2.)
        PAp0, PAp1 = shiftthet(PAp0, thetobj=np.pi/2.), shiftthet(PAp1, thetobj=np.pi/2.)

        Ap0 = tfg.Apert('Test0', PAp0, Ves=Ves, Exp='AUG', Diag='Test', shot=0, SavePath=Root+Addpath)
        Ap1 = tfg.Apert('Test1', PAp1, Ves=Ves, Exp='AUG', Diag='Test', shot=0, SavePath=Root+Addpath)
        cls.Obj = tfg.Detect('Test', Poly, Optics=[Ap0,Ap1], Ves=Ves, Exp='AUG', Diag='Test', shot=0, SavePath=Root+Addpath,
                             Etend_Method='quad', Etend_RelErr=1.e-2,
                             Cone_DRY=0.005, Cone_DXTheta=0.005, Cone_DZ=0.005)

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


    def test01_refine_ConePoly(self):
        self.Obj.refine_ConePoly( Proj='Cross', indPoly=0, Verb=False)

    def test02_isInside(self, NR=10,NZ=10):
        R = np.linspace(self.Obj.Ves._P1Min[0],self.Obj.Ves._P1Max[0],NR)
        Z = np.linspace(self.Obj.Ves._P2Min[0],self.Obj.Ves._P2Max[0],NZ)
        Pts = np.array([np.tile(R,(NZ,1)).flatten(), np.tile(Z,(NR,1)).T.flatten()])
        indRZ = self.Obj.isInside(Pts, In='(Y,Z)', Test=True)

        Thet = 0.
        Pts = np.array([Thet*np.ones((Pts.shape[1],)), Pts[0,:], Pts[1,:]])
        ind = self.Obj.isInside(Pts, In='(X,Y,Z)', Test=True)
        assert np.all(indRZ==ind)


    def test03_calc_SAngVect(self, NR=10,NZ=10):
        R = np.linspace(self.Obj.Ves._P1Min[0],self.Obj.Ves._P1Max[0],NR)
        Z = np.linspace(self.Obj.Ves._P2Min[0],self.Obj.Ves._P2Max[0],NZ)
        Pts = np.array([np.tile(R,(NZ,1)).flatten(), np.tile(Z,(NR,1)).T.flatten()])
        SAngRZ1, VectRZ1 = self.Obj.calc_SAngVect(Pts, In='(Y,Z)', Colis=False, Test=True)
        SAngRZ, VectRZ = self.Obj.calc_SAngVect(Pts, In='(Y,Z)', Colis=True, Test=True)

        Thet = self.Obj.LOS[self.Obj._LOSRef]['PRef'][0]
        Pts = np.array([Thet*np.ones((Pts.shape[1],)), Pts[0,:], Pts[1,:]])
        SAng1, Vect1 = self.Obj.calc_SAngVect(Pts, In='(X,Y,Z)', Colis=False, Test=True)
        SAng, Vect = self.Obj.calc_SAngVect(Pts, In='(X,Y,Z)', Colis=True, Test=True)
        ind1RZ0, ind10 = SAngRZ1>0, SAng1>0.
        indRZ0, ind0 = SAngRZ>0., SAng>0.
        assert np.all(ind1RZ0==ind10)
        assert np.all(indRZ0==ind0)
        assert np.all(SAngRZ1==SAng1) and np.all(VectRZ1[:,ind10]==Vect1[:,ind10])
        assert np.all(SAngRZ==SAng) and np.all(VectRZ[:,ind0]==Vect[:,ind0])

    def test04_calc_Sig(self):
        func = lambda Pts, A=1.: A*np.exp(-(((Pts[1,:]-1.7)/0.3)**2 + ((Pts[2,:]-0.)/0.5)**2))
        Sig1 = self.Obj.calc_Sig(func, extargs={'A':1.}, Method='Vol', Mode='simps', PreComp=True, Colis=True, dX12=[0.005, 0.005], dX12Mode='abs', ds=0.005, dsMode='abs', MarginS=0.001)
        Sig2 = self.Obj.calc_Sig(func, extargs={'A':1.}, Method='Vol', Mode='sum', PreComp=False, Colis=True, dX12=[0.005, 0.005], dX12Mode='abs', ds=0.005, dsMode='abs', MarginS=0.001)
        Sig3 = self.Obj.calc_Sig(func, extargs={'A':1.}, Method='LOS', Mode='trapz', PreComp=True, Colis=True, ds=0.005, dsMode='abs', MarginS=0.001)
        Sig4 = self.Obj.calc_Sig(func, extargs={'A':1.}, Method='LOS', Mode='quad', PreComp=False, Colis=True, epsrel=1.e-4, MarginS=0.001)
        assert np.abs(Sig1-Sig2)<0.001*min(Sig1,Sig2), str(Sig1)+" vs "+str(Sig2)
        assert np.abs(Sig3-Sig4)<0.001*min(Sig3,Sig4), str(Sig3)+" vs "+str(Sig4)


    def test05_debug_Etendue_BenchmarkRatioMode(self):
        Etends, Ratio, RelErr, dX12, dX12Mode, Colis = self.Obj._debug_Etendue_BenchmarkRatioMode(RelErr=1.e-3, Ratio=[0.01,0.1], Modes=['simps','quad'], dX12=[0.002,0.002], dX12Mode='abs', Colis=True)
        for kk in Etends.keys():
            assert all([np.abs(ee-Etends[kk][0,0])<0.01*Etends[kk][0,0] for ee in Etends[kk][0,:]]), str(Etends[kk][0,:])

    #def test06_calc_Etendue_AlongLOS(self):
    #    Etends, Ps, k, LOSRef = self.Obj.calc_Etendue_AlongLOS(Length='', NP=5, kMode='rel', Modes=['trapz','quad'], RelErr=1.e-3, dX12=[0.005, 0.005], dX12Mode='abs', Ratio=0.02, Colis=True, Test=True)


    #def test07_calc_SAngNb(self):
    #    SA, ind, Pts = self.Obj.calc_SAngNb(Pts=None, Proj='Cross', Slice='Int', DRY=None, DXTheta=None, DZ=None, Colis=True)
    #    SA, ind, Pts = self.Obj.calc_SAngNb(Pts=None, Proj='Hor', Slice='Int', DRY=None, DXTheta=None, DZ=None, Colis=True)
    #    SA, ind, Pts = self.Obj.calc_SAngNb(Pts=None, Proj='Cross', Slice='Int', DRY=None, DXTheta=None, DZ=None, Colis=False)
    #    SA, ind, Pts = self.Obj.calc_SAngNb(Pts=None, Proj='Hor', Slice='Int', DRY=None, DXTheta=None, DZ=None, Colis=False)


    def test08_set_Res(self):
        self.Obj._set_Res(CrossMesh=[0.1,0.1], CrossMeshMode='abs', Mode='Iso', Amp=1., Deg=0, steps=0.01, Thres=0.05, ThresMode='rel', ThresMin=0.01,
                          IntResCross=[0.1,0.1], IntResCrossMode='rel', IntResLong=0.1, IntResLongMode='rel', Eq=None, Ntt=50, EqName=None, save=False, Test=True)


    def test09_plot(self):
        Lax = self.Obj.plot(Proj='All', Elt='PV', EltVes='P', draw=False, a4=False, Test=True)
        Lax = self.Obj.plot(Proj='Cross', Elt='PC', EltVes='', draw=False, a4=False, Test=True)
        Lax = self.Obj.plot(Proj='Hor', Elt='PVC', EltVes='P', draw=False, a4=False, Test=True)
        plt.close('all')


    def test10_plot_SAngNb(self):
        Lax = self.Obj.plot_SAngNb(Lax=None, Proj='Cross', Slice='Int', Pts=None, plotfunc='scatter', DRY=None, DXTheta=None, DZ=None, Elt='P', EltVes='P', EltLOS='L', EltOptics='P', Colis=True, a4=False, draw=False, Test=True)
        Lax = self.Obj.plot_SAngNb(Lax=None, Proj='Hor', Slice='Max', Pts=None, plotfunc='scatter', DRY=None, DXTheta=None, DZ=None, Elt='P', EltVes='P', EltLOS='', EltOptics='P', Colis=True, a4=False, draw=False, Test=True)
        plt.close('all')

    def test11_debug_plot_SAng_OnPlanePerp(self):
        ax = self.Obj._debug_plot_SAng_OnPlanePerp(ax=None, Pos=tfd.DetSAngPlRa, dX12=tfd.DetSAngPldX12, dX12Mode=tfd.DetSAngPldX12Mode, Ratio=tfd.DetSAngPlRatio, Colis=True, draw=False, a4=False, Test=True)
        plt.close('all')

    def test12_plot_Etend_AlongLOS(self):
        ax = self.Obj.plot_Etend_AlongLOS(ax=None, NP=5, kMode='rel', Modes=['trapz','simps'], dX12=[0.005, 0.005], dX12Mode='abs', Ratio=0.02, LOSPts=True, Colis=True, draw=False, a4=True, Test=True)
        ax = self.Obj.plot_Etend_AlongLOS(ax=None, NP=5, kMode='rel', Modes=['quad'], RelErr=1.e-3, Ratio=0.01, LOSPts=True, Colis=False, draw=False, a4=False, Test=True)
        plt.close('all')

    def test13_plot_Sinogram(self):
        ax = self.Obj.plot_Sinogram(ax=None, Proj='Cross', Elt='DLV', Ang='theta', AngUnit='rad', Sketch=True, LOSRef=None, draw=False, a4=False, Test=True)
        ax = self.Obj.plot_Sinogram(ax=None, Proj='Cross', Elt='DLV', Ang='xi', AngUnit='deg', Sketch=False, LOSRef=None, draw=False, a4=True, Test=True)
        plt.close('all')

    def test14_plot_Res(self):
        ax = self.Obj._plot_Res(ax=None, plotfunc='scatter', NC=20, draw=False, a4=False, Test=True)
        ax = self.Obj._plot_Res(ax=None, plotfunc='contourf', NC=20, draw=False, a4=True, Test=True)
        plt.close('all')

    def test15_saveload(self):
        self.Obj.save()
        obj = tfpf.Open(self.Obj.Id.SavePath + self.Obj.Id.SaveName + '.npz')
        os.remove(self.Obj.Id.SavePath + self.Obj.Id.SaveName + '.npz')








class Test15_DetectLensTor:

    @classmethod
    def setup_class(cls, Ves=VesTor):
        print ("")
        print "--------- "+VerbHead+cls.__name__
        Out = np.load(Root+Addpath+'G_015.npz')

        O = np.array([0.,Ves._P1Max[0],Ves._P2Max[1]])
        nIn = np.array([0.,-1.,-1.])
        Rad, F1 = 0.005, 0.010
        Lens = tfg.Lens('Test', O, nIn, Rad, F1, Ves=Ves, Type='Sph', Exp='AUG', Diag='Test', shot=0, SavePath=Root+Addpath)
        cls.Obj = tfg.Detect('Test', {'Rad':0.001}, Optics=Lens, Ves=Ves, Exp='AUG', Diag='Test', shot=0, SavePath=Root+Addpath,
                             Etend_Method='quad', Etend_RelErr=1.e-2,
                             Cone_DRY=0.005, Cone_DXTheta=np.pi/512., Cone_DZ=0.005)

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


    def test01_refine_ConePoly(self):
        self.Obj.refine_ConePoly( Proj='Cross', indPoly=0, Verb=False)


    def test02_isInside(self, NR=10,NZ=10):
        R = np.linspace(self.Obj.Ves._P1Min[0],self.Obj.Ves._P1Max[0],NR)
        Z = np.linspace(self.Obj.Ves._P2Min[0],self.Obj.Ves._P2Max[0],NZ)
        Pts = np.array([np.tile(R,(NZ,1)).flatten(), np.tile(Z,(NR,1)).T.flatten()])
        indRZ = self.Obj.isInside(Pts, In='(R,Z)', Test=True)

        Thet = np.pi/4.
        Pts = np.array([Pts[0,:]*np.cos(Thet), Pts[0,:]*np.sin(Thet), Pts[1,:]])
        ind = self.Obj.isInside(Pts, In='(X,Y,Z)', Test=True)
        assert np.all(indRZ==ind)


    def test03_calc_SAngVect(self, NR=10,NZ=10):
        R = np.linspace(self.Obj.Ves._P1Min[0],self.Obj.Ves._P1Max[0],NR)
        Z = np.linspace(self.Obj.Ves._P2Min[0],self.Obj.Ves._P2Max[0],NZ)
        Pts = np.array([np.tile(R,(NZ,1)).flatten(), np.tile(Z,(NR,1)).T.flatten()])
        SAngRZ1, VectRZ1 = self.Obj.calc_SAngVect(Pts, In='(R,Z)', Colis=False, Test=True)
        SAngRZ, VectRZ = self.Obj.calc_SAngVect(Pts, In='(R,Z)', Colis=True, Test=True)

        Thet = np.arctan2(self.Obj.LOS[self.Obj._LOSRef]['PRef'][1],self.Obj.LOS[self.Obj._LOSRef]['PRef'][0])
        Pts = np.array([Pts[0,:]*np.cos(Thet), Pts[0,:]*np.sin(Thet), Pts[1,:]])
        SAng1, Vect1 = self.Obj.calc_SAngVect(Pts, In='(X,Y,Z)', Colis=False, Test=True)
        SAng, Vect = self.Obj.calc_SAngVect(Pts, In='(X,Y,Z)', Colis=True, Test=True)
        ind1RZ0, ind10 = SAngRZ1>0, SAng1>0.
        indRZ0, ind0 = SAngRZ>0., SAng>0.
        assert np.all(ind1RZ0==ind10)
        assert np.all(indRZ0==ind0)
        assert np.all(SAngRZ1==SAng1) and np.all(VectRZ1[:,ind10]==Vect1[:,ind10])
        assert np.all(SAngRZ==SAng) and np.all(VectRZ[:,ind0]==Vect[:,ind0])

    def test04_calc_Sig(self):
        func = lambda Pts, A=1.: A*np.exp(-(((np.hypot(Pts[0,:],Pts[1,:])-1.7)/0.3)**2 + ((Pts[2,:]-0.)/0.5)**2))
        Sig1 = self.Obj.calc_Sig(func, extargs={'A':1.}, Method='Vol', Mode='simps', PreComp=True, Colis=True, dX12=[0.005, 0.005], dX12Mode='abs', ds=0.005, dsMode='abs', MarginS=0.001)
        Sig2 = self.Obj.calc_Sig(func, extargs={'A':1.}, Method='Vol', Mode='sum', PreComp=False, Colis=True, dX12=[0.005, 0.005], dX12Mode='abs', ds=0.005, dsMode='abs', MarginS=0.001)
        Sig3 = self.Obj.calc_Sig(func, extargs={'A':1.}, Method='LOS', Mode='trapz', PreComp=True, Colis=True, ds=0.005, dsMode='abs', MarginS=0.001)
        Sig4 = self.Obj.calc_Sig(func, extargs={'A':1.}, Method='LOS', Mode='quad', PreComp=False, Colis=True, epsrel=1.e-4, MarginS=0.001)
        assert np.abs(Sig1-Sig2)<0.001*min(Sig1,Sig2), str(Sig1)+" vs "+str(Sig2)
        assert np.abs(Sig3-Sig4)<0.001*min(Sig3,Sig4), str(Sig3)+" vs "+str(Sig4)


    def test05_debug_Etendue_BenchmarkRatioMode(self):
        Etends, Ratio, RelErr, dX12, dX12Mode, Colis = self.Obj._debug_Etendue_BenchmarkRatioMode(RelErr=1.e-3, Ratio=[0.01,0.1], Modes=['simps','quad'], dX12=[0.002,0.002], dX12Mode='abs', Colis=True)
        for kk in Etends.keys():
            assert all([np.abs(ee-Etends[kk][0,0])<0.01*Etends[kk][0,0] for ee in Etends[kk][0,:]]), str(Etends[kk][0,:])

    #def test06_calc_Etendue_AlongLOS(self):
    #    Etends, Ps, k, LOSRef = self.Obj.calc_Etendue_AlongLOS(Length='', NP=5, kMode='rel', Modes=['trapz','quad'], RelErr=1.e-3, dX12=[0.005, 0.005], dX12Mode='abs', Ratio=0.02, Colis=True, Test=True)


    #def test07_calc_SAngNb(self):
    #    SA, ind, Pts = self.Obj.calc_SAngNb(Pts=None, Proj='Cross', Slice='Int', DRY=None, DXTheta=None, DZ=None, Colis=True)
    #    SA, ind, Pts = self.Obj.calc_SAngNb(Pts=None, Proj='Hor', Slice='Int', DRY=None, DXTheta=None, DZ=None, Colis=True)
    #    SA, ind, Pts = self.Obj.calc_SAngNb(Pts=None, Proj='Cross', Slice='Int', DRY=None, DXTheta=None, DZ=None, Colis=False)
    #    SA, ind, Pts = self.Obj.calc_SAngNb(Pts=None, Proj='Hor', Slice='Int', DRY=None, DXTheta=None, DZ=None, Colis=False)


    def test08_set_Res(self):
        self.Obj._set_Res(CrossMesh=[0.1,0.1], CrossMeshMode='abs', Mode='Iso', Amp=1., Deg=0, steps=0.01, Thres=0.05, ThresMode='rel', ThresMin=0.01,
                         IntResCross=[0.1,0.1], IntResCrossMode='rel', IntResLong=0.1, IntResLongMode='rel', Eq=None, Ntt=50, EqName=None, save=False, Test=True)


    def test09_plot(self):
        Lax = self.Obj.plot(Proj='All', Elt='PV', EltVes='P', draw=False, a4=False, Test=True)
        Lax = self.Obj.plot(Proj='Cross', Elt='PC', EltVes='', draw=False, a4=False, Test=True)
        Lax = self.Obj.plot(Proj='Hor', Elt='PVC', EltVes='P', draw=False, a4=False, Test=True)
        plt.close('all')


    def test10_plot_SAngNb(self):
        Lax = self.Obj.plot_SAngNb(Lax=None, Proj='Cross', Slice='Int', Pts=None, plotfunc='scatter', DRY=None, DXTheta=None, DZ=None, Elt='P', EltVes='P', EltLOS='L', EltOptics='P', Colis=True, a4=False, draw=False, Test=True)
        Lax = self.Obj.plot_SAngNb(Lax=None, Proj='Hor', Slice='Max', Pts=None, plotfunc='scatter', DRY=None, DXTheta=None, DZ=None, Elt='P', EltVes='P', EltLOS='', EltOptics='P', Colis=True, a4=False, draw=False, Test=True)
        plt.close('all')

    def test11_debug_plot_SAng_OnPlanePerp(self):
        ax = self.Obj._debug_plot_SAng_OnPlanePerp(ax=None, Pos=tfd.DetSAngPlRa, dX12=tfd.DetSAngPldX12, dX12Mode=tfd.DetSAngPldX12Mode, Ratio=tfd.DetSAngPlRatio, Colis=True, draw=False, a4=False, Test=True)
        plt.close('all')

    def test12_plot_Etend_AlongLOS(self):
        ax = self.Obj.plot_Etend_AlongLOS(ax=None, NP=5, kMode='rel', Modes=['trapz','simps'], dX12=[0.005, 0.005], dX12Mode='abs', Ratio=0.02, LOSPts=True, Colis=True, draw=False, a4=True, Test=True)
        ax = self.Obj.plot_Etend_AlongLOS(ax=None, NP=5, kMode='rel', Modes=['quad'], RelErr=1.e-3, Ratio=0.01, LOSPts=True, Colis=False, draw=False, a4=False, Test=True)
        plt.close('all')

    def test13_plot_Sinogram(self):
        ax = self.Obj.plot_Sinogram(ax=None, Proj='Cross', Elt='DLV', Ang='theta', AngUnit='rad', Sketch=True, LOSRef=None, draw=False, a4=False, Test=True)
        ax = self.Obj.plot_Sinogram(ax=None, Proj='Cross', Elt='DLV', Ang='xi', AngUnit='deg', Sketch=False, LOSRef=None, draw=False, a4=True, Test=True)
        plt.close('all')

    def test14_plot_Res(self):
        ax = self.Obj._plot_Res(ax=None, plotfunc='scatter', NC=20, draw=False, a4=False, Test=True)
        ax = self.Obj._plot_Res(ax=None, plotfunc='contourf', NC=20, draw=False, a4=True, Test=True)
        plt.close('all')

    def test15_saveload(self):
        self.Obj.save()
        obj = tfpf.Open(self.Obj.Id.SavePath + self.Obj.Id.SaveName + '.npz')
        os.remove(self.Obj.Id.SavePath + self.Obj.Id.SaveName + '.npz')








class Test16_DetectLensLin:

    @classmethod
    def setup_class(cls, Ves=VesLin):
        print ("")
        print "--------- "+VerbHead+cls.__name__
        Out = np.load(Root+Addpath+'G_015.npz')

        O = np.array([0.,Ves._P1Max[0],Ves._P2Max[1]])
        nIn = np.array([0.,-1.,-1.])
        Rad, F1 = 0.005, 0.010
        Lens = tfg.Lens('Test', O, nIn, Rad, F1, Ves=Ves, Type='Sph', Exp='AUG', Diag='Test', shot=0, SavePath=Root+Addpath)
        cls.Obj = tfg.Detect('Test', {'Rad':0.001}, Optics=Lens, Ves=Ves, Exp='AUG', Diag='Test', shot=0, SavePath=Root+Addpath,
                             Etend_Method='quad', Etend_RelErr=1.e-2,
                             Cone_DRY=0.005, Cone_DXTheta=0.005, Cone_DZ=0.005)

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


    def test01_refine_ConePoly(self):
        self.Obj.refine_ConePoly( Proj='Cross', indPoly=0, Verb=False)


    def test02_isInside(self, NR=10,NZ=10):
        R = np.linspace(self.Obj.Ves._P1Min[0],self.Obj.Ves._P1Max[0],NR)
        Z = np.linspace(self.Obj.Ves._P2Min[0],self.Obj.Ves._P2Max[0],NZ)
        Pts = np.array([np.tile(R,(NZ,1)).flatten(), np.tile(Z,(NR,1)).T.flatten()])
        indRZ = self.Obj.isInside(Pts, In='(Y,Z)', Test=True)

        Thet = self.Obj.LOS[self.Obj._LOSRef]['PRef'][0]
        Pts = np.array([Thet*np.ones((Pts.shape[1],)), Pts[0,:], Pts[1,:]])
        ind = self.Obj.isInside(Pts, In='(X,Y,Z)', Test=True)
        assert np.all(indRZ==ind)


    def test03_calc_SAngVect(self, NR=10,NZ=10):
        R = np.linspace(self.Obj.Ves._P1Min[0],self.Obj.Ves._P1Max[0],NR)
        Z = np.linspace(self.Obj.Ves._P2Min[0],self.Obj.Ves._P2Max[0],NZ)
        Pts = np.array([np.tile(R,(NZ,1)).flatten(), np.tile(Z,(NR,1)).T.flatten()])
        SAngRZ1, VectRZ1 = self.Obj.calc_SAngVect(Pts, In='(Y,Z)', Colis=False, Test=True)
        SAngRZ, VectRZ = self.Obj.calc_SAngVect(Pts, In='(Y,Z)', Colis=True, Test=True)

        Thet = self.Obj.LOS[self.Obj._LOSRef]['PRef'][0]
        Pts = np.array([Thet*np.ones((Pts.shape[1],)), Pts[0,:], Pts[1,:]])
        SAng1, Vect1 = self.Obj.calc_SAngVect(Pts, In='(X,Y,Z)', Colis=False, Test=True)
        SAng, Vect = self.Obj.calc_SAngVect(Pts, In='(X,Y,Z)', Colis=True, Test=True)
        ind1RZ0, ind10 = SAngRZ1>0, SAng1>0.
        indRZ0, ind0 = SAngRZ>0., SAng>0.
        assert np.all(ind1RZ0==ind10)
        assert np.all(indRZ0==ind0)
        assert np.all(SAngRZ1==SAng1) and np.all(VectRZ1[:,ind10]==Vect1[:,ind10])
        assert np.all(SAngRZ==SAng) and np.all(VectRZ[:,ind0]==Vect[:,ind0])

    def test04_calc_Sig(self):
        func = lambda Pts, A=1.: A*np.exp(-(((Pts[1,:]-1.7)/0.3)**2 + ((Pts[2,:]-0.)/0.5)**2))
        Sig1 = self.Obj.calc_Sig(func, extargs={'A':1.}, Method='Vol', Mode='simps', PreComp=True, Colis=True, dX12=[0.005, 0.005], dX12Mode='abs', ds=0.005, dsMode='abs', MarginS=0.001)
        Sig2 = self.Obj.calc_Sig(func, extargs={'A':1.}, Method='Vol', Mode='sum', PreComp=False, Colis=True, dX12=[0.005, 0.005], dX12Mode='abs', ds=0.005, dsMode='abs', MarginS=0.001)
        Sig3 = self.Obj.calc_Sig(func, extargs={'A':1.}, Method='LOS', Mode='trapz', PreComp=True, Colis=True, ds=0.005, dsMode='abs', MarginS=0.001)
        Sig4 = self.Obj.calc_Sig(func, extargs={'A':1.}, Method='LOS', Mode='quad', PreComp=False, Colis=True, epsrel=1.e-4, MarginS=0.001)
        assert np.abs(Sig1-Sig2)<0.001*min(Sig1,Sig2), str(Sig1)+" vs "+str(Sig2)
        assert np.abs(Sig3-Sig4)<0.001*min(Sig3,Sig4), str(Sig3)+" vs "+str(Sig4)


    def test05_debug_Etendue_BenchmarkRatioMode(self):
        Etends, Ratio, RelErr, dX12, dX12Mode, Colis = self.Obj._debug_Etendue_BenchmarkRatioMode(RelErr=1.e-3, Ratio=[0.01,0.1], Modes=['simps','quad'], dX12=[0.002,0.002], dX12Mode='abs', Colis=True)
        for kk in Etends.keys():
            assert all([np.abs(ee-Etends[kk][0,0])<0.01*Etends[kk][0,0] for ee in Etends[kk][0,:]]), str(Etends[kk][0,:])

    #def test06_calc_Etendue_AlongLOS(self):
    #    Etends, Ps, k, LOSRef = self.Obj.calc_Etendue_AlongLOS(Length='', NP=5, kMode='rel', Modes=['trapz','quad'], RelErr=1.e-3, dX12=[0.005, 0.005], dX12Mode='abs', Ratio=0.02, Colis=True, Test=True)


    #def test07_calc_SAngNb(self):
    #    SA, ind, Pts = self.Obj.calc_SAngNb(Pts=None, Proj='Cross', Slice='Int', DRY=None, DXTheta=None, DZ=None, Colis=True)
    #    SA, ind, Pts = self.Obj.calc_SAngNb(Pts=None, Proj='Hor', Slice='Int', DRY=None, DXTheta=None, DZ=None, Colis=True)
    #    SA, ind, Pts = self.Obj.calc_SAngNb(Pts=None, Proj='Cross', Slice='Int', DRY=None, DXTheta=None, DZ=None, Colis=False)
    #    SA, ind, Pts = self.Obj.calc_SAngNb(Pts=None, Proj='Hor', Slice='Int', DRY=None, DXTheta=None, DZ=None, Colis=False)


    def test08_set_Res(self):
        self.Obj._set_Res(CrossMesh=[0.1,0.1], CrossMeshMode='abs', Mode='Iso', Amp=1., Deg=0, steps=0.01, Thres=0.05, ThresMode='rel', ThresMin=0.01,
                         IntResCross=[0.1,0.1], IntResCrossMode='rel', IntResLong=0.1, IntResLongMode='rel', Eq=None, Ntt=50, EqName=None, save=False, Test=True)


    def test09_plot(self):
        Lax = self.Obj.plot(Proj='All', Elt='PV', EltVes='P', draw=False, a4=False, Test=True)
        Lax = self.Obj.plot(Proj='Cross', Elt='PC', EltVes='', draw=False, a4=False, Test=True)
        Lax = self.Obj.plot(Proj='Hor', Elt='PVC', EltVes='P', draw=False, a4=False, Test=True)
        plt.close('all')


    def test10_plot_SAngNb(self):
        Lax = self.Obj.plot_SAngNb(Lax=None, Proj='Cross', Slice='Int', Pts=None, plotfunc='scatter', DRY=None, DXTheta=None, DZ=None, Elt='P', EltVes='P', EltLOS='L', EltOptics='P', Colis=True, a4=False, draw=False, Test=True)
        Lax = self.Obj.plot_SAngNb(Lax=None, Proj='Hor', Slice='Max', Pts=None, plotfunc='scatter', DRY=None, DXTheta=None, DZ=None, Elt='P', EltVes='P', EltLOS='', EltOptics='P', Colis=True, a4=False, draw=False, Test=True)
        plt.close('all')

    def test11_debug_plot_SAng_OnPlanePerp(self):
        ax = self.Obj._debug_plot_SAng_OnPlanePerp(ax=None, Pos=tfd.DetSAngPlRa, dX12=tfd.DetSAngPldX12, dX12Mode=tfd.DetSAngPldX12Mode, Ratio=tfd.DetSAngPlRatio, Colis=True, draw=False, a4=False, Test=True)
        plt.close('all')

    def test12_plot_Etend_AlongLOS(self):
        ax = self.Obj.plot_Etend_AlongLOS(ax=None, NP=5, kMode='rel', Modes=['trapz','simps'], dX12=[0.005, 0.005], dX12Mode='abs', Ratio=0.02, LOSPts=True, Colis=True, draw=False, a4=True, Test=True)
        ax = self.Obj.plot_Etend_AlongLOS(ax=None, NP=5, kMode='rel', Modes=['quad'], RelErr=1.e-3, Ratio=0.01, LOSPts=True, Colis=False, draw=False, a4=False, Test=True)
        plt.close('all')

    def test13_plot_Sinogram(self):
        ax = self.Obj.plot_Sinogram(ax=None, Proj='Cross', Elt='DLV', Ang='theta', AngUnit='rad', Sketch=True, LOSRef=None, draw=False, a4=False, Test=True)
        ax = self.Obj.plot_Sinogram(ax=None, Proj='Cross', Elt='DLV', Ang='xi', AngUnit='deg', Sketch=False, LOSRef=None, draw=False, a4=True, Test=True)
        plt.close('all')

    def test14_plot_Res(self):
        ax = self.Obj._plot_Res(ax=None, plotfunc='scatter', NC=20, draw=False, a4=False, Test=True)
        ax = self.Obj._plot_Res(ax=None, plotfunc='contourf', NC=20, draw=False, a4=True, Test=True)
        plt.close('all')

    def test15_saveload(self):
        self.Obj.save()
        obj = tfpf.Open(self.Obj.Id.SavePath + self.Obj.Id.SaveName + '.npz')
        os.remove(self.Obj.Id.SavePath + self.Obj.Id.SaveName + '.npz')











###########################################################
#
#  Creating GDetect objects and testing methods (Apert and Lens, Tor and Lin)
#
###########################################################



class Test17_GDetectApertTor:

    @classmethod
    def setup_class(cls, Ves=VesTor):
        print ("")
        print "--------- "+VerbHead+cls.__name__

        ld = ['L_009','L_012']#'L_010','L_011'
        LD = []
        Out = np.load(Root+Addpath+'L_009.npz')
        PAp0, PAp1 = Out['PolyAp0'], Out['PolyAp1']
        Ap0L = tfg.Apert('Test0L', PAp0, Ves=Ves, Exp='AUG', Diag='Test', shot=0, SavePath=Root+Addpath)
        Ap1L = tfg.Apert('Test1L', PAp1, Ves=Ves, Exp='AUG', Diag='Test', shot=0, SavePath=Root+Addpath)
        for dd in ld:
            Poly = np.load(Root+Addpath+dd+'.npz')['Poly']
            LD.append(tfg.Detect(dd, Poly, Optics=[Ap0L,Ap1L], Ves=Ves, Exp='AUG', Diag='Test', shot=0, SavePath=Root+Addpath,
                                 CalcEtend=True, CalcSpanImp=True, CalcCone=True, CalcPreComp=False, Calc=True, Verb=False,
                                 Etend_Method='simps', Etend_dX12=[0.05,0.05], Etend_dX12Mode='rel',
                                 Cone_DRY=0.005, Cone_DXTheta=0.005, Cone_DZ=0.005))

        ld = ['G_015','G_020']
        Out = np.load(Root+Addpath+'G_015.npz')
        PAp0, PAp1 = Out['PolyAp0'], Out['PolyAp1']
        Ap0G = tfg.Apert('Test0G', PAp0, Ves=Ves, Exp='AUG', Diag='Test', shot=0, SavePath=Root+Addpath)
        Ap1G = tfg.Apert('Test1G', PAp1, Ves=Ves, Exp='AUG', Diag='Test', shot=0, SavePath=Root+Addpath)
        for dd in ld:
            Poly = np.load(Root+Addpath+dd+'.npz')['Poly']
            LD.append(tfg.Detect(dd, Poly, Optics=[Ap0G,Ap1G], Ves=Ves, Exp='AUG', Diag='Test', shot=0, SavePath=Root+Addpath,
                                 CalcEtend=True, CalcSpanImp=True, CalcCone=True, CalcPreComp=False, Calc=True, Verb=False,
                                 Etend_Method='simps', Etend_dX12=[0.05,0.05], Etend_dX12Mode='rel',
                                 Cone_DRY=0.005, Cone_DXTheta=np.pi/512., Cone_DZ=0.005))

        cls.Obj = tfg.GDetect('LG', LD, Exp='AUG', Diag='Test', shot=0, SavePath=Root+Addpath)

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

    def test01_select(self):
        ind = self.Obj.select(Val='L_010', Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='In', Out=bool)
        ind = self.Obj.select(Val='G_015', Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='In', Out=int)
        ind = self.Obj.select(Val=['L_009','G_020'], Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='Out', Out=int)

    def test02_isInside(self, NR=10,NZ=10):
        R = np.linspace(self.Obj.Ves._P1Min[0],self.Obj.Ves._P1Max[0],NR)
        Z = np.linspace(self.Obj.Ves._P2Min[0],self.Obj.Ves._P2Max[0],NZ)
        Pts = np.array([np.tile(R,(NZ,1)).flatten(), np.tile(Z,(NR,1)).T.flatten()])
        indRZ = self.Obj.isInside(Pts, In='(R,Z)', Test=True)

        Thet = np.pi/4.
        Pts = np.array([Pts[0,:]*np.cos(Thet), Pts[0,:]*np.sin(Thet), Pts[1,:]])
        ind = self.Obj.isInside(Pts, In='(X,Y,Z)', Test=True)
        assert np.all(indRZ==ind)

    def test03_get_GLOS(self):
        GLOS = self.Obj.get_GLOS(Name='GLOSExtract')

    def test04_calc_SAngVect(self, NR=10,NZ=10):
        R = np.linspace(self.Obj.Ves._P1Min[0],self.Obj.Ves._P1Max[0],NR)
        Z = np.linspace(self.Obj.Ves._P2Min[0],self.Obj.Ves._P2Max[0],NZ)
        Pts = np.array([np.tile(R,(NZ,1)).flatten(), np.tile(Z,(NR,1)).T.flatten()])
        SAngRZ1, VectRZ1 = self.Obj.calc_SAngVect(Pts, In='(R,Z)', Colis=False, Test=True)
        SAngRZ, VectRZ = self.Obj.calc_SAngVect(Pts, In='(R,Z)', Colis=True, Test=True)

        Thet = np.arctan2(self.Obj.LDetect[0].LOS[self.Obj._LOSRef]['PRef'][1],self.Obj.LDetect[0].LOS[self.Obj._LOSRef]['PRef'][0])
        Pts = np.array([Pts[0,:]*np.cos(Thet), Pts[0,:]*np.sin(Thet), Pts[1,:]])
        SAng1, Vect1 = self.Obj.calc_SAngVect(Pts, In='(X,Y,Z)', Colis=False, Test=True)
        SAng, Vect = self.Obj.calc_SAngVect(Pts, In='(X,Y,Z)', Colis=True, Test=True)

    def test05_calc_Sig(self):
        func = lambda Pts, A=1.: A*np.exp(-(((np.hypot(Pts[0,:],Pts[1,:])-1.7)/0.3)**2 + ((Pts[2,:]-0.)/0.5)**2))
        Sig1, GD1 = self.Obj.calc_Sig(func, extargs={'A':1.}, Method='Vol', Mode='simps', PreComp=True, Colis=True, dX12=[0.005, 0.005], dX12Mode='abs', ds=0.005, dsMode='abs', MarginS=0.001, Val=['L_009','L_011'])
        Sig2, GD2 = self.Obj.calc_Sig(func, extargs={'A':1.}, Method='Vol', Mode='sum', PreComp=False, Colis=True, dX12=[0.005, 0.005], dX12Mode='abs', ds=0.005, dsMode='abs', MarginS=0.001, Val=['L_009','L_011'])
        Sig3, GD3 = self.Obj.calc_Sig(func, extargs={'A':1.}, Method='LOS', Mode='trapz', PreComp=True, Colis=True, ds=0.005, dsMode='abs', MarginS=0.001, Val=['L_009','L_011'])
        Sig4, GD4 = self.Obj.calc_Sig(func, extargs={'A':1.}, Method='LOS', Mode='quad', PreComp=False, Colis=True, epsrel=1.e-4, MarginS=0.001, Val=['L_009','L_011'])
        assert np.all(np.abs(Sig1-Sig2)<0.001*Sig1), str(Sig1)+" vs "+str(Sig2)
        assert np.all(np.abs(Sig3-Sig4)<0.001*Sig3), str(Sig3)+" vs "+str(Sig4)


    #def test06_calc_SAngNb(self):
    #    SA, ind, Pts = self.Obj.calc_SAngNb(Pts=None, Proj='Cross', Slice='Int', DRY=None, DXTheta=None, DZ=None, Colis=True)
    #    SA, ind, Pts = self.Obj.calc_SAngNb(Pts=None, Proj='Hor', Slice='Int', DRY=None, DXTheta=None, DZ=None, Colis=True)
    #    SA, ind, Pts = self.Obj.calc_SAngNb(Pts=None, Proj='Cross', Slice='Int', DRY=None, DXTheta=None, DZ=None, Colis=False)
    #    SA, ind, Pts = self.Obj.calc_SAngNb(Pts=None, Proj='Hor', Slice='Int', DRY=None, DXTheta=None, DZ=None, Colis=False)


    def test07_set_Res(self):
        self.Obj._set_Res(CrossMesh=[0.1,0.1], CrossMeshMode='abs', Mode='Iso', Amp=1., Deg=0, steps=0.01, Thres=0.05, ThresMode='rel', ThresMin=0.01,
                         IntResCross=[0.1,0.1], IntResCrossMode='rel', IntResLong=0.1, IntResLongMode='rel', Eq=None, Ntt=50, EqName=None, save=False, Test=True)


    def test08_plot(self):
        Lax = self.Obj.plot(Proj='All', Elt='PV', EltLOS='LIO', EltOptics='P', EltVes='P', draw=False, a4=False, Test=True)
        Lax = self.Obj.plot(Proj='Cross', Elt='PC', EltLOS='DIORP', EltOptics='PV', EltVes='', draw=False, a4=False, Test=True)
        Lax = self.Obj.plot(Proj='Hor', Elt='PVC', EltLOS='L', EltOptics='', EltVes='P', draw=False, a4=False, Test=True)
        plt.close('all')


    def test09_plot_SAngNb(self):
        Lax = self.Obj.plot_SAngNb(Lax=None, Proj='Cross', Slice='Int', Pts=None, plotfunc='scatter', DRY=None, DXTheta=None, DZ=None, Elt='P', EltVes='P', EltLOS='L', EltOptics='P', Colis=True, a4=False, draw=False, Test=True)
        Lax = self.Obj.plot_SAngNb(Lax=None, Proj='Hor', Slice='Max', Pts=None, plotfunc='scatter', DRY=None, DXTheta=None, DZ=None, Elt='P', EltVes='P', EltLOS='', EltOptics='P', Colis=True, a4=False, draw=False, Test=True)
        plt.close('all')

    def test10_plot_Etend_AlongLOS(self):
        ax = self.Obj.plot_Etend_AlongLOS(ax=None, NP=2, kMode='rel', Modes=['trapz','simps'], dX12=[0.005, 0.005], dX12Mode='abs', Ratio=0.02, LOSPts=True, Colis=True, draw=False, a4=True, Test=True, Val=['L_009','L_011'])
        ax = self.Obj.plot_Etend_AlongLOS(ax=None, NP=2, kMode='rel', Modes=['quad'], RelErr=1.e-3, Ratio=0.01, LOSPts=True, Colis=False, draw=False, a4=False, Test=True, Val=['L_009','L_011'])
        plt.close('all')

    def test11_plot_Sinogram(self):
        ax = self.Obj.plot_Sinogram(ax=None, Proj='Cross', Elt='DLV', Ang='theta', AngUnit='rad', Sketch=True, LOSRef=None, draw=False, a4=False, Test=True)
        ax = self.Obj.plot_Sinogram(ax=None, Proj='Cross', Elt='DLV', Ang='xi', AngUnit='deg', Sketch=False, LOSRef=None, draw=False, a4=True, Test=True)
        plt.close('all')

    def test12_plot_Etendues(self):
        ax = self.Obj.plot_Etendues(Mode='Etend', Elt='', ax=None, draw=False, a4=False, Test=True, Val='L_011', InOut='Out')
        plt.close('all')

    def test13_plot_Sig(self):
        func = lambda Pts, A=1.: A*np.exp(-(((np.hypot(Pts[0,:],Pts[1,:])-1.7)/0.3)**2 + ((Pts[2,:]-0.)/0.5)**2))
        ax = self.Obj.plot_Sig(func, extargs={'A':1.}, Method='Vol', Mode='simps', PreComp=True, Colis=True, dX12=[0.005, 0.005], dX12Mode='abs', ds=0.005, dsMode='abs', draw=False, a4=True, MarginS=0.001)
        plt.close('all')

    def test14_plot_Res(self):
        ax = self.Obj._plot_Res(ax=None, plotfunc='scatter', NC=20, draw=False, a4=False, Test=True)
        ax = self.Obj._plot_Res(ax=None, plotfunc='contourf', NC=20, draw=False, a4=True, Test=True)
        plt.close('all')

    def test15_saveload(self):
        self.Obj.save()
        obj = tfpf.Open(self.Obj.Id.SavePath + self.Obj.Id.SaveName + '.npz')
        os.remove(self.Obj.Id.SavePath + self.Obj.Id.SaveName + '.npz')









class Test18_GDetectApertLin:

    @classmethod
    def setup_class(cls, Ves=VesLin):
        print ("")
        print "--------- "+VerbHead+cls.__name__

        ld = ['L_009','L_012']#'L_010','L_011'
        LD = []
        Out = np.load(Root+Addpath+'L_009.npz')
        PAp0, PAp1 = shiftthet(Out['PolyAp0'], thetobj=np.pi/2.), shiftthet(Out['PolyAp1'], thetobj=np.pi/2.)

        Ap0L = tfg.Apert('Test0L', PAp0, Ves=Ves, Exp='AUG', Diag='Test', shot=0, SavePath=Root+Addpath)
        Ap1L = tfg.Apert('Test1L', PAp1, Ves=Ves, Exp='AUG', Diag='Test', shot=0, SavePath=Root+Addpath)
        for dd in ld:
            Poly = shiftthet(np.load(Root+Addpath+dd+'.npz')['Poly'], thetobj=np.pi/2.)
            LD.append(tfg.Detect(dd, Poly, Optics=[Ap0L,Ap1L], Ves=Ves, Exp='AUG', Diag='Test', shot=0, SavePath=Root+Addpath,
                                 CalcEtend=True, CalcSpanImp=True, CalcCone=True, CalcPreComp=False, Calc=True, Verb=False,
                                 Etend_Method='simps', Etend_dX12=[0.05,0.05], Etend_dX12Mode='rel',
                                 Cone_DRY=0.005, Cone_DXTheta=0.005, Cone_DZ=0.005))

        ld = ['G_015','G_020']
        Out = np.load(Root+Addpath+'G_015.npz')
        PAp0, PAp1 = shiftthet(Out['PolyAp0'], thetobj=np.pi/2.), shiftthet(Out['PolyAp1'], thetobj=np.pi/2.)
        Ap0G = tfg.Apert('Test0G', PAp0, Ves=Ves, Exp='AUG', Diag='Test', shot=0, SavePath=Root+Addpath)
        Ap1G = tfg.Apert('Test1G', PAp1, Ves=Ves, Exp='AUG', Diag='Test', shot=0, SavePath=Root+Addpath)
        for dd in ld:
            Poly = shiftthet(np.load(Root+Addpath+dd+'.npz')['Poly'], thetobj=np.pi/2.)
            LD.append(tfg.Detect(dd, Poly, Optics=[Ap0G,Ap1G], Ves=Ves, Exp='AUG', Diag='Test', shot=0, SavePath=Root+Addpath,
                                 CalcEtend=True, CalcSpanImp=True, CalcCone=True, CalcPreComp=False, Calc=True, Verb=False,
                                 Etend_Method='simps', Etend_dX12=[0.05,0.05], Etend_dX12Mode='rel',
                                 Cone_DRY=0.005, Cone_DXTheta=0.005, Cone_DZ=0.005))

        cls.Obj = tfg.GDetect('LG', LD, Exp='AUG', Diag='Test', shot=0, SavePath=Root+Addpath)

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

    def test01_select(self):
        ind = self.Obj.select(Val='L_010', Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='In', Out=bool)
        ind = self.Obj.select(Val='G_015', Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='In', Out=int)
        ind = self.Obj.select(Val=['L_009','G_020'], Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='Out', Out=int)

    def test02_isInside(self, NR=10,NZ=10):
        R = np.linspace(self.Obj.Ves._P1Min[0],self.Obj.Ves._P1Max[0],NR)
        Z = np.linspace(self.Obj.Ves._P2Min[0],self.Obj.Ves._P2Max[0],NZ)
        Pts = np.array([np.tile(R,(NZ,1)).flatten(), np.tile(Z,(NR,1)).T.flatten()])
        indRZ = self.Obj.isInside(Pts, In='(R,Z)', Test=True)

        Thet = np.pi/4.
        Pts = np.array([Pts[0,:]*np.cos(Thet), Pts[0,:]*np.sin(Thet), Pts[1,:]])
        ind = self.Obj.isInside(Pts, In='(X,Y,Z)', Test=True)
        assert np.all(indRZ==ind)

    def test03_get_GLOS(self):
        GLOS = self.Obj.get_GLOS(Name='GLOSExtract')

    def test04_calc_SAngVect(self, NR=10,NZ=10):
        R = np.linspace(self.Obj.Ves._P1Min[0],self.Obj.Ves._P1Max[0],NR)
        Z = np.linspace(self.Obj.Ves._P2Min[0],self.Obj.Ves._P2Max[0],NZ)
        Pts = np.array([np.tile(R,(NZ,1)).flatten(), np.tile(Z,(NR,1)).T.flatten()])
        SAngRZ1, VectRZ1 = self.Obj.calc_SAngVect(Pts, In='(R,Z)', Colis=False, Test=True)
        SAngRZ, VectRZ = self.Obj.calc_SAngVect(Pts, In='(R,Z)', Colis=True, Test=True)

        Thet = np.arctan2(self.Obj.LDetect[0].LOS[self.Obj._LOSRef]['PRef'][1],self.Obj.LDetect[0].LOS[self.Obj._LOSRef]['PRef'][0])
        Pts = np.array([Pts[0,:]*np.cos(Thet), Pts[0,:]*np.sin(Thet), Pts[1,:]])
        SAng1, Vect1 = self.Obj.calc_SAngVect(Pts, In='(X,Y,Z)', Colis=False, Test=True)
        SAng, Vect = self.Obj.calc_SAngVect(Pts, In='(X,Y,Z)', Colis=True, Test=True)

    def test05_calc_Sig(self):
        func = lambda Pts, A=1.: A*np.exp(-(((np.hypot(Pts[0,:],Pts[1,:])-1.7)/0.3)**2 + ((Pts[2,:]-0.)/0.5)**2))
        Sig1, GD1 = self.Obj.calc_Sig(func, extargs={'A':1.}, Method='Vol', Mode='simps', PreComp=True, Colis=True, dX12=[0.005, 0.005], dX12Mode='abs', ds=0.005, dsMode='abs', MarginS=0.001, Val=['L_009','L_011'])
        Sig2, GD2 = self.Obj.calc_Sig(func, extargs={'A':1.}, Method='Vol', Mode='sum', PreComp=False, Colis=True, dX12=[0.005, 0.005], dX12Mode='abs', ds=0.005, dsMode='abs', MarginS=0.001, Val=['L_009','L_011'])
        Sig3, GD3 = self.Obj.calc_Sig(func, extargs={'A':1.}, Method='LOS', Mode='trapz', PreComp=True, Colis=True, ds=0.005, dsMode='abs', MarginS=0.001, Val=['L_009','L_011'])
        Sig4, GD4 = self.Obj.calc_Sig(func, extargs={'A':1.}, Method='LOS', Mode='quad', PreComp=False, Colis=True, epsrel=1.e-4, MarginS=0.001, Val=['L_009','L_011'])
        assert np.all(np.abs(Sig1-Sig2)<0.001*Sig1), str(Sig1)+" vs "+str(Sig2)
        assert np.all(np.abs(Sig3-Sig4)<0.001*Sig3), str(Sig3)+" vs "+str(Sig4)


    #def test06_calc_SAngNb(self):
    #    SA, ind, Pts = self.Obj.calc_SAngNb(Pts=None, Proj='Cross', Slice='Int', DRY=None, DXTheta=None, DZ=None, Colis=True)
    #    SA, ind, Pts = self.Obj.calc_SAngNb(Pts=None, Proj='Hor', Slice='Int', DRY=None, DXTheta=None, DZ=None, Colis=True)
    #    SA, ind, Pts = self.Obj.calc_SAngNb(Pts=None, Proj='Cross', Slice='Int', DRY=None, DXTheta=None, DZ=None, Colis=False)
    #    SA, ind, Pts = self.Obj.calc_SAngNb(Pts=None, Proj='Hor', Slice='Int', DRY=None, DXTheta=None, DZ=None, Colis=False)


    def test07_set_Res(self):
        self.Obj._set_Res(CrossMesh=[0.1,0.1], CrossMeshMode='abs', Mode='Iso', Amp=1., Deg=0, steps=0.01, Thres=0.05, ThresMode='rel', ThresMin=0.01,
                         IntResCross=[0.1,0.1], IntResCrossMode='rel', IntResLong=0.1, IntResLongMode='rel', Eq=None, Ntt=50, EqName=None, save=False, Test=True)


    def test08_plot(self):
        Lax = self.Obj.plot(Proj='All', Elt='PV', EltLOS='LIO', EltOptics='P', EltVes='P', draw=False, a4=False, Test=True)
        Lax = self.Obj.plot(Proj='Cross', Elt='PC', EltLOS='DIORP', EltOptics='PV', EltVes='', draw=False, a4=False, Test=True)
        Lax = self.Obj.plot(Proj='Hor', Elt='PVC', EltLOS='L', EltOptics='', EltVes='P', draw=False, a4=False, Test=True)
        plt.close('all')


    def test09_plot_SAngNb(self):
        Lax = self.Obj.plot_SAngNb(Lax=None, Proj='Cross', Slice='Int', Pts=None, plotfunc='scatter', DRY=None, DXTheta=None, DZ=None, Elt='P', EltVes='P', EltLOS='L', EltOptics='P', Colis=True, a4=False, draw=False, Test=True)
        Lax = self.Obj.plot_SAngNb(Lax=None, Proj='Hor', Slice='Max', Pts=None, plotfunc='scatter', DRY=None, DXTheta=None, DZ=None, Elt='P', EltVes='P', EltLOS='', EltOptics='P', Colis=True, a4=False, draw=False, Test=True)
        plt.close('all')

    def test10_plot_Etend_AlongLOS(self):
        ax = self.Obj.plot_Etend_AlongLOS(ax=None, NP=2, kMode='rel', Modes=['trapz','simps'], dX12=[0.005, 0.005], dX12Mode='abs', Ratio=0.02, LOSPts=True, Colis=True, draw=False, a4=True, Test=True, Val=['L_009','L_011'])
        ax = self.Obj.plot_Etend_AlongLOS(ax=None, NP=2, kMode='rel', Modes=['quad'], RelErr=1.e-3, Ratio=0.01, LOSPts=True, Colis=False, draw=False, a4=False, Test=True, Val=['L_009','L_011'])
        plt.close('all')

    def test11_plot_Sinogram(self):
        ax = self.Obj.plot_Sinogram(ax=None, Proj='Cross', Elt='DLV', Ang='theta', AngUnit='rad', Sketch=True, LOSRef=None, draw=False, a4=False, Test=True)
        ax = self.Obj.plot_Sinogram(ax=None, Proj='Cross', Elt='DLV', Ang='xi', AngUnit='deg', Sketch=False, LOSRef=None, draw=False, a4=True, Test=True)
        plt.close('all')

    def test12_plot_Etendues(self):
        ax = self.Obj.plot_Etendues(Mode='Etend', Elt='', ax=None, draw=False, a4=False, Test=True, Val='L_011', InOut='Out')
        plt.close('all')

    def test13_plot_Sig(self):
        func = lambda Pts, A=1.: A*np.exp(-(((np.hypot(Pts[0,:],Pts[1,:])-1.7)/0.3)**2 + ((Pts[2,:]-0.)/0.5)**2))
        ax = self.Obj.plot_Sig(func, extargs={'A':1.}, Method='Vol', Mode='simps', PreComp=True, Colis=True, dX12=[0.005, 0.005], dX12Mode='abs', ds=0.005, dsMode='abs', draw=False, a4=True, MarginS=0.001)
        plt.close('all')

    def test14_plot_Res(self):
        ax = self.Obj._plot_Res(ax=None, plotfunc='scatter', NC=20, draw=False, a4=False, Test=True)
        ax = self.Obj._plot_Res(ax=None, plotfunc='contourf', NC=20, draw=False, a4=True, Test=True)
        plt.close('all')

    def test15_saveload(self):
        self.Obj.save()
        obj = tfpf.Open(self.Obj.Id.SavePath + self.Obj.Id.SaveName + '.npz')
        os.remove(self.Obj.Id.SavePath + self.Obj.Id.SaveName + '.npz')







class Test19_GDetectLensTor:

    @classmethod
    def setup_class(cls, Ves=VesTor):
        print ("")
        print "--------- "+VerbHead+cls.__name__

        LD = []
        O = np.array([0.,Ves._P1Min[0],Ves._P2Max[1]])
        nIn = np.array([0.,1.,-1.])
        Rad, F1 = 0.005, 0.010
        Lens = tfg.Lens('Test', O, nIn, Rad, F1, Ves=Ves, Type='Sph', Exp='AUG', Diag='Test', shot=0, SavePath=Root+Addpath)
        ld = [('d01',0.),('d02',-0.01),('d03',-0.02)]
        for (dd,dz) in ld:
            O = np.array([0.,Ves._P1Min[0],Ves._P2Max[1]-dz])
            Lens = tfg.Lens('Test', O, nIn, Rad, F1, Ves=Ves, Type='Sph', Exp='AUG', Diag='Test', shot=0, SavePath=Root+Addpath)
            LD.append(tfg.Detect(dd, {'Rad':0.001}, Optics=Lens, Ves=Ves, Exp='AUG', Diag='Test', shot=0, SavePath=Root+Addpath,
                                 CalcEtend=True, CalcSpanImp=True, CalcCone=True, CalcPreComp=False, Calc=True, Verb=False,
                                 Etend_Method='simps', Etend_dX12=[0.05,0.05], Etend_dX12Mode='rel',
                                 Cone_DRY=0.005, Cone_DXTheta=np.pi/512., Cone_DZ=0.005))

        cls.Obj = tfg.GDetect('LG', LD, Exp='AUG', Diag='Test', shot=0, SavePath=Root+Addpath)

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

    def test01_select(self):
        ind = self.Obj.select(Val='d01', Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='In', Out=bool)
        ind = self.Obj.select(Val='d02', Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='In', Out=int)
        ind = self.Obj.select(Val=['d01','d03'], Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='Out', Out=int)

    def test02_isInside(self, NR=10,NZ=10):
        R = np.linspace(self.Obj.Ves._P1Min[0],self.Obj.Ves._P1Max[0],NR)
        Z = np.linspace(self.Obj.Ves._P2Min[0],self.Obj.Ves._P2Max[0],NZ)
        Pts = np.array([np.tile(R,(NZ,1)).flatten(), np.tile(Z,(NR,1)).T.flatten()])
        indRZ = self.Obj.isInside(Pts, In='(R,Z)', Test=True)

        Thet = np.pi/4.
        Pts = np.array([Pts[0,:]*np.cos(Thet), Pts[0,:]*np.sin(Thet), Pts[1,:]])
        ind = self.Obj.isInside(Pts, In='(X,Y,Z)', Test=True)
        assert np.all(indRZ==ind)

    def test03_get_GLOS(self):
        GLOS = self.Obj.get_GLOS(Name='GLOSExtract')

    def test04_calc_SAngVect(self, NR=10,NZ=10):
        R = np.linspace(self.Obj.Ves._P1Min[0],self.Obj.Ves._P1Max[0],NR)
        Z = np.linspace(self.Obj.Ves._P2Min[0],self.Obj.Ves._P2Max[0],NZ)
        Pts = np.array([np.tile(R,(NZ,1)).flatten(), np.tile(Z,(NR,1)).T.flatten()])
        SAngRZ1, VectRZ1 = self.Obj.calc_SAngVect(Pts, In='(R,Z)', Colis=False, Test=True)
        SAngRZ, VectRZ = self.Obj.calc_SAngVect(Pts, In='(R,Z)', Colis=True, Test=True)

        Thet = np.arctan2(self.Obj.LDetect[0].LOS[self.Obj._LOSRef]['PRef'][1],self.Obj.LDetect[0].LOS[self.Obj._LOSRef]['PRef'][0])
        Pts = np.array([Pts[0,:]*np.cos(Thet), Pts[0,:]*np.sin(Thet), Pts[1,:]])
        SAng1, Vect1 = self.Obj.calc_SAngVect(Pts, In='(X,Y,Z)', Colis=False, Test=True)
        SAng, Vect = self.Obj.calc_SAngVect(Pts, In='(X,Y,Z)', Colis=True, Test=True)

    def test05_calc_Sig(self):
        func = lambda Pts, A=1.: A*np.exp(-(((np.hypot(Pts[0,:],Pts[1,:])-1.7)/0.3)**2 + ((Pts[2,:]-0.)/0.5)**2))
        Sig1, GD1 = self.Obj.calc_Sig(func, extargs={'A':1.}, Method='Vol', Mode='simps', PreComp=True, Colis=True, dX12=[0.005, 0.005], dX12Mode='abs', ds=0.005, dsMode='abs', MarginS=0.001, Val=['d01','d03'])
        Sig2, GD2 = self.Obj.calc_Sig(func, extargs={'A':1.}, Method='Vol', Mode='sum', PreComp=False, Colis=True, dX12=[0.005, 0.005], dX12Mode='abs', ds=0.005, dsMode='abs', MarginS=0.001, Val=['d01','d03'])
        Sig3, GD3 = self.Obj.calc_Sig(func, extargs={'A':1.}, Method='LOS', Mode='trapz', PreComp=True, Colis=True, ds=0.005, dsMode='abs', MarginS=0.001, Val=['d01','d03'])
        Sig4, GD4 = self.Obj.calc_Sig(func, extargs={'A':1.}, Method='LOS', Mode='quad', PreComp=False, Colis=True, epsrel=1.e-4, MarginS=0.001, Val=['d01','d03'])
        assert np.all(np.abs(Sig1-Sig2)<0.001*Sig1), str(Sig1)+" vs "+str(Sig2)
        assert np.all(np.abs(Sig3-Sig4)<0.001*Sig3), str(Sig3)+" vs "+str(Sig4)


    #def test06_calc_SAngNb(self):
    #    SA, ind, Pts = self.Obj.calc_SAngNb(Pts=None, Proj='Cross', Slice='Int', DRY=None, DXTheta=None, DZ=None, Colis=True)
    #    SA, ind, Pts = self.Obj.calc_SAngNb(Pts=None, Proj='Hor', Slice='Int', DRY=None, DXTheta=None, DZ=None, Colis=True)
    #    SA, ind, Pts = self.Obj.calc_SAngNb(Pts=None, Proj='Cross', Slice='Int', DRY=None, DXTheta=None, DZ=None, Colis=False)
    #    SA, ind, Pts = self.Obj.calc_SAngNb(Pts=None, Proj='Hor', Slice='Int', DRY=None, DXTheta=None, DZ=None, Colis=False)


    def test07_set_Res(self):
        self.Obj._set_Res(CrossMesh=[0.1,0.1], CrossMeshMode='abs', Mode='Iso', Amp=1., Deg=0, steps=0.01, Thres=0.05, ThresMode='rel', ThresMin=0.01,
                          IntResCross=[0.1,0.1], IntResCrossMode='rel', IntResLong=0.1, IntResLongMode='rel', Eq=None, Ntt=50, EqName=None, save=False, Test=True)


    def test08_plot(self):
        Lax = self.Obj.plot(Proj='All', Elt='PV', EltLOS='LIO', EltOptics='P', EltVes='P', draw=False, a4=False, Test=True)
        Lax = self.Obj.plot(Proj='Cross', Elt='PC', EltLOS='DIORP', EltOptics='PV', EltVes='', draw=False, a4=False, Test=True)
        Lax = self.Obj.plot(Proj='Hor', Elt='PVC', EltLOS='L', EltOptics='', EltVes='P', draw=False, a4=False, Test=True)
        plt.close('all')


    def test09_plot_SAngNb(self):
        Lax = self.Obj.plot_SAngNb(Lax=None, Proj='Cross', Slice='Int', Pts=None, plotfunc='scatter', DRY=None, DXTheta=None, DZ=None, Elt='P', EltVes='P', EltLOS='L', EltOptics='P', Colis=True, a4=False, draw=False, Test=True)
        Lax = self.Obj.plot_SAngNb(Lax=None, Proj='Hor', Slice='Max', Pts=None, plotfunc='scatter', DRY=None, DXTheta=None, DZ=None, Elt='P', EltVes='P', EltLOS='', EltOptics='P', Colis=True, a4=False, draw=False, Test=True)
        plt.close('all')

    def test10_plot_Etend_AlongLOS(self):
        ax = self.Obj.plot_Etend_AlongLOS(ax=None, NP=2, kMode='rel', Modes=['trapz','simps'], dX12=[0.005, 0.005], dX12Mode='abs', Ratio=0.02, LOSPts=True, Colis=True, draw=False, a4=True, Test=True, Val=['d01','d03'])
        ax = self.Obj.plot_Etend_AlongLOS(ax=None, NP=2, kMode='rel', Modes=['quad'], RelErr=1.e-3, Ratio=0.01, LOSPts=True, Colis=False, draw=False, a4=False, Test=True, Val=['d01','d03'])
        plt.close('all')

    def test11_plot_Sinogram(self):
        ax = self.Obj.plot_Sinogram(ax=None, Proj='Cross', Elt='DLV', Ang='theta', AngUnit='rad', Sketch=True, LOSRef=None, draw=False, a4=False, Test=True)
        ax = self.Obj.plot_Sinogram(ax=None, Proj='Cross', Elt='DLV', Ang='xi', AngUnit='deg', Sketch=False, LOSRef=None, draw=False, a4=True, Test=True)
        plt.close('all')

    def test12_plot_Etendues(self):
        ax = self.Obj.plot_Etendues(Mode='Etend', Elt='', ax=None, draw=False, a4=False, Test=True, Val='d02', InOut='Out')
        plt.close('all')

    def test13_plot_Sig(self):
        func = lambda Pts, A=1.: A*np.exp(-(((np.hypot(Pts[0,:],Pts[1,:])-1.7)/0.3)**2 + ((Pts[2,:]-0.)/0.5)**2))
        ax = self.Obj.plot_Sig(func, extargs={'A':1.}, Method='Vol', Mode='simps', PreComp=True, Colis=True, dX12=[0.005, 0.005], dX12Mode='abs', ds=0.005, dsMode='abs', draw=False, a4=True, MarginS=0.001)
        plt.close('all')

    def test14_plot_Res(self):
        ax = self.Obj._plot_Res(ax=None, plotfunc='scatter', NC=20, draw=False, a4=False, Test=True)
        ax = self.Obj._plot_Res(ax=None, plotfunc='contourf', NC=20, draw=False, a4=True, Test=True)
        plt.close('all')

    def test15_saveload(self):
        self.Obj.save()
        obj = tfpf.Open(self.Obj.Id.SavePath + self.Obj.Id.SaveName + '.npz')
        os.remove(self.Obj.Id.SavePath + self.Obj.Id.SaveName + '.npz')






class Test20_GDetectLensLin:

    @classmethod
    def setup_class(cls, Ves=VesTor):
        print ("")
        print "--------- "+VerbHead+cls.__name__

        LD = []
        O = np.array([0.,Ves._P1Min[0],Ves._P2Max[1]])
        nIn = np.array([0.,1.,-1.])
        Rad, F1 = 0.005, 0.010
        Lens = tfg.Lens('Test', O, nIn, Rad, F1, Ves=Ves, Type='Sph', Exp='AUG', Diag='Test', shot=0, SavePath=Root+Addpath)
        ld = [('d01',0.),('d02',-0.01),('d03',-0.02)]
        for (dd,dz) in ld:
            O = np.array([0.,Ves._P1Min[0],Ves._P2Max[1]-dz])
            Lens = tfg.Lens('Test', O, nIn, Rad, F1, Ves=Ves, Type='Sph', Exp='AUG', Diag='Test', shot=0, SavePath=Root+Addpath)
            LD.append(tfg.Detect(dd, {'Rad':0.001}, Optics=Lens, Ves=Ves, Exp='AUG', Diag='Test', shot=0, SavePath=Root+Addpath,
                                 CalcEtend=True, CalcSpanImp=True, CalcCone=True, CalcPreComp=False, Calc=True, Verb=False,
                                 Etend_Method='simps', Etend_dX12=[0.05,0.05], Etend_dX12Mode='rel',
                                 Cone_DRY=0.005, Cone_DXTheta=0.005, Cone_DZ=0.005))

        cls.Obj = tfg.GDetect('LG', LD, Exp='AUG', Diag='Test', shot=0, SavePath=Root+Addpath)

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

    def test01_select(self):
        ind = self.Obj.select(Val='d01', Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='In', Out=bool)
        ind = self.Obj.select(Val='d02', Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='In', Out=int)
        ind = self.Obj.select(Val=['d01','d03'], Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='Out', Out=int)

    def test02_isInside(self, NR=10,NZ=10):
        R = np.linspace(self.Obj.Ves._P1Min[0],self.Obj.Ves._P1Max[0],NR)
        Z = np.linspace(self.Obj.Ves._P2Min[0],self.Obj.Ves._P2Max[0],NZ)
        Pts = np.array([np.tile(R,(NZ,1)).flatten(), np.tile(Z,(NR,1)).T.flatten()])
        indRZ = self.Obj.isInside(Pts, In='(R,Z)', Test=True)

        Thet = np.pi/4.
        Pts = np.array([Pts[0,:]*np.cos(Thet), Pts[0,:]*np.sin(Thet), Pts[1,:]])
        ind = self.Obj.isInside(Pts, In='(X,Y,Z)', Test=True)
        assert np.all(indRZ==ind)

    def test03_get_GLOS(self):
        GLOS = self.Obj.get_GLOS(Name='GLOSExtract')

    def test04_calc_SAngVect(self, NR=10,NZ=10):
        R = np.linspace(self.Obj.Ves._P1Min[0],self.Obj.Ves._P1Max[0],NR)
        Z = np.linspace(self.Obj.Ves._P2Min[0],self.Obj.Ves._P2Max[0],NZ)
        Pts = np.array([np.tile(R,(NZ,1)).flatten(), np.tile(Z,(NR,1)).T.flatten()])
        SAngRZ1, VectRZ1 = self.Obj.calc_SAngVect(Pts, In='(R,Z)', Colis=False, Test=True)
        SAngRZ, VectRZ = self.Obj.calc_SAngVect(Pts, In='(R,Z)', Colis=True, Test=True)

        Thet = np.arctan2(self.Obj.LDetect[0].LOS[self.Obj._LOSRef]['PRef'][1],self.Obj.LDetect[0].LOS[self.Obj._LOSRef]['PRef'][0])
        Pts = np.array([Pts[0,:]*np.cos(Thet), Pts[0,:]*np.sin(Thet), Pts[1,:]])
        SAng1, Vect1 = self.Obj.calc_SAngVect(Pts, In='(X,Y,Z)', Colis=False, Test=True)
        SAng, Vect = self.Obj.calc_SAngVect(Pts, In='(X,Y,Z)', Colis=True, Test=True)

    def test05_calc_Sig(self):
        func = lambda Pts, A=1.: A*np.exp(-(((np.hypot(Pts[0,:],Pts[1,:])-1.7)/0.3)**2 + ((Pts[2,:]-0.)/0.5)**2))
        Sig1, GD1 = self.Obj.calc_Sig(func, extargs={'A':1.}, Method='Vol', Mode='simps', PreComp=True, Colis=True, dX12=[0.005, 0.005], dX12Mode='abs', ds=0.005, dsMode='abs', MarginS=0.001, Val=['d01','d03'])
        Sig2, GD2 = self.Obj.calc_Sig(func, extargs={'A':1.}, Method='Vol', Mode='sum', PreComp=False, Colis=True, dX12=[0.005, 0.005], dX12Mode='abs', ds=0.005, dsMode='abs', MarginS=0.001, Val=['d01','d03'])
        Sig3, GD3 = self.Obj.calc_Sig(func, extargs={'A':1.}, Method='LOS', Mode='trapz', PreComp=True, Colis=True, ds=0.005, dsMode='abs', MarginS=0.001, Val=['d01','d03'])
        Sig4, GD4 = self.Obj.calc_Sig(func, extargs={'A':1.}, Method='LOS', Mode='quad', PreComp=False, Colis=True, epsrel=1.e-4, MarginS=0.001, Val=['d01','d03'])
        assert np.all(np.abs(Sig1-Sig2)<0.001*Sig1), str(Sig1)+" vs "+str(Sig2)
        assert np.all(np.abs(Sig3-Sig4)<0.001*Sig3), str(Sig3)+" vs "+str(Sig4)


    #def test06_calc_SAngNb(self):
    #    SA, ind, Pts = self.Obj.calc_SAngNb(Pts=None, Proj='Cross', Slice='Int', DRY=None, DXTheta=None, DZ=None, Colis=True)
    #    SA, ind, Pts = self.Obj.calc_SAngNb(Pts=None, Proj='Hor', Slice='Int', DRY=None, DXTheta=None, DZ=None, Colis=True)
    #    SA, ind, Pts = self.Obj.calc_SAngNb(Pts=None, Proj='Cross', Slice='Int', DRY=None, DXTheta=None, DZ=None, Colis=False)
    #    SA, ind, Pts = self.Obj.calc_SAngNb(Pts=None, Proj='Hor', Slice='Int', DRY=None, DXTheta=None, DZ=None, Colis=False)


    def test07_set_Res(self):
        self.Obj._set_Res(CrossMesh=[0.1,0.1], CrossMeshMode='abs', Mode='Iso', Amp=1., Deg=0, steps=0.01, Thres=0.05, ThresMode='rel', ThresMin=0.01,
                         IntResCross=[0.1,0.1], IntResCrossMode='rel', IntResLong=0.1, IntResLongMode='rel', Eq=None, Ntt=50, EqName=None, save=False, Test=True)


    def test08_plot(self):
        Lax = self.Obj.plot(Proj='All', Elt='PV', EltLOS='LIO', EltOptics='P', EltVes='P', draw=False, a4=False, Test=True)
        Lax = self.Obj.plot(Proj='Cross', Elt='PC', EltLOS='DIORP', EltOptics='PV', EltVes='', draw=False, a4=False, Test=True)
        Lax = self.Obj.plot(Proj='Hor', Elt='PVC', EltLOS='L', EltOptics='', EltVes='P', draw=False, a4=False, Test=True)
        plt.close('all')


    def test09_plot_SAngNb(self):
        Lax = self.Obj.plot_SAngNb(Lax=None, Proj='Cross', Slice='Int', Pts=None, plotfunc='scatter', DRY=None, DXTheta=None, DZ=None, Elt='P', EltVes='P', EltLOS='L', EltOptics='P', Colis=True, a4=False, draw=False, Test=True)
        Lax = self.Obj.plot_SAngNb(Lax=None, Proj='Hor', Slice='Max', Pts=None, plotfunc='scatter', DRY=None, DXTheta=None, DZ=None, Elt='P', EltVes='P', EltLOS='', EltOptics='P', Colis=True, a4=False, draw=False, Test=True)
        plt.close('all')

    def test10_plot_Etend_AlongLOS(self):
        ax = self.Obj.plot_Etend_AlongLOS(ax=None, NP=2, kMode='rel', Modes=['trapz','simps'], dX12=[0.005, 0.005], dX12Mode='abs', Ratio=0.02, LOSPts=True, Colis=True, draw=False, a4=True, Test=True, Val=['d01','d03'])
        ax = self.Obj.plot_Etend_AlongLOS(ax=None, NP=2, kMode='rel', Modes=['quad'], RelErr=1.e-3, Ratio=0.01, LOSPts=True, Colis=False, draw=False, a4=False, Test=True, Val=['d01','d03'])
        plt.close('all')

    def test11_plot_Sinogram(self):
        ax = self.Obj.plot_Sinogram(ax=None, Proj='Cross', Elt='DLV', Ang='theta', AngUnit='rad', Sketch=True, LOSRef=None, draw=False, a4=False, Test=True)
        ax = self.Obj.plot_Sinogram(ax=None, Proj='Cross', Elt='DLV', Ang='xi', AngUnit='deg', Sketch=False, LOSRef=None, draw=False, a4=True, Test=True)
        plt.close('all')

    def test12_plot_Etendues(self):
        ax = self.Obj.plot_Etendues(Mode='Etend', Elt='', ax=None, draw=False, a4=False, Test=True, Val='d02', InOut='Out')
        plt.close('all')

    def test13_plot_Sig(self):
        func = lambda Pts, A=1.: A*np.exp(-(((np.hypot(Pts[0,:],Pts[1,:])-1.7)/0.3)**2 + ((Pts[2,:]-0.)/0.5)**2))
        ax = self.Obj.plot_Sig(func, extargs={'A':1.}, Method='Vol', Mode='simps', PreComp=True, Colis=True, dX12=[0.005, 0.005], dX12Mode='abs', ds=0.005, dsMode='abs', draw=False, a4=True, MarginS=0.001)
        plt.close('all')

    def test14_plot_Res(self):
        ax = self.Obj._plot_Res(ax=None, plotfunc='scatter', NC=20, draw=False, a4=False, Test=True)
        ax = self.Obj._plot_Res(ax=None, plotfunc='contourf', NC=20, draw=False, a4=True, Test=True)
        plt.close('all')

    def test15_saveload(self):
        self.Obj.save()
        obj = tfpf.Open(self.Obj.Id.SavePath + self.Obj.Id.SaveName + '.npz')
        os.remove(self.Obj.Id.SavePath + self.Obj.Id.SaveName + '.npz')

"""
