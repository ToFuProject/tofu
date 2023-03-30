"""
This module contains tests for tofu.geom in its structured version
"""



# External modules
import os
import itertools as itt
import numpy as np
import matplotlib.pyplot as plt
import warnings as warn

# Importing package tofu.gem
import tofu as tf
from tofu import __version__
import tofu.defaults as tfd
import tofu.pathfile as tfpf
import tofu.utils as tfu
import tofu.geom as tfg


_here = os.path.abspath(os.path.dirname(__file__))
VerbHead = 'tofu.geom.test_03_core'
keyVers = 'Vers'
_Exp = 'WEST'


#######################################################
#
#     Setup and Teardown
#
#######################################################

def setup_module(module):
    print("") # this is to get a newline after the dots
    lf = os.listdir(_here)
    lf = [f for f in lf
         if all([s in f for s in ['TFG_',_Exp,'.npz']])]
    lF = []
    for f in lf:
        ff = f.split('_')
        v = [fff[len(keyVers):] for fff in ff
             if fff[:len(keyVers)]==keyVers]
        msg = f + "\n    "+str(ff) + "\n    " + str(v)
        assert len(v)==1, msg
        v = v[0]
        if '.npz' in v:
            v = v[:v.index('.npz')]
        # print(v, __version__)
        if v!=__version__:
            lF.append(f)
    if len(lF)>0:
        print("Removing the following previous test files:")
        for f in lF:
            os.remove(os.path.join(_here,f))
        #print("setup_module before anything in this file")

def teardown_module(module):
    #os.remove(VesTor.Id.SavePath + VesTor.Id.SaveName + '.npz')
    #os.remove(VesLin.Id.SavePath + VesLin.Id.SaveName + '.npz')
    #print("teardown_module after everything in this file")
    #print("") # this is to get a newline
    lf = os.listdir(_here)
    lf = [f for f in lf
         if all([s in f for s in ['TFG_',_Exp,'.npz']])]
    lF = []
    for f in lf:
        ff = f.split('_')
        v = [fff[len(keyVers):] for fff in ff
             if fff[:len(keyVers)]==keyVers]
        msg = f + "\n    "+str(ff) + "\n    " + str(v)
        assert len(v)==1, msg
        v = v[0]
        if '.npz' in v:
            v = v[:v.index('.npz')]
        # print(v, __version__)
        if v==__version__:
            lF.append(f)
    if len(lF)>0:
        print("Removing the following test files:")
        for f in lF:
            os.remove(os.path.join(_here,f))


#######################################################
#
#   Struct subclasses
#
#######################################################
path = os.path.join(_here, 'test_data')
lf = os.listdir(path)
lf = [f for f in lf if all([s in f for s in [_Exp,'.txt']])]
lCls = sorted(set([f.split('_')[1] for f in lf]))


# Define a dict of objects to be tested, in linear and toroidal geometry
dobj = {'Tor':{}, 'Lin':{}}
for tt in dobj.keys():
    for cc in lCls:
        lfc = [f for f in lf if f.split('_')[1] == cc and 'V0' in f]
        ln = []
        for f in lfc:
            if 'CoilCS' in f:
                ln.append(f.split('_')[2].split('.')[0])
            else:
                ln.append(f.split('_')[2].split('.')[0])
        lnu = sorted(set(ln))
        if not len(lnu) == len(ln):
            msg = ("Non-unique name list for {0}:".format(cc)
                   + "\n\tln = [{0}]".format(', '.join(ln))
                   + "\n\tlnu = [{0}]".format(', '.join(lnu)))
            raise Exception(msg)
        dobj[tt][cc] = {}
        for ii in range(0, len(ln)):
            if 'BumperOuter' in ln[ii]:
                Lim = np.r_[10.,20.]*np.pi/180.
            elif 'BumperInner' in ln[ii]:
                t0 = np.arange(0,360,60)*np.pi/180.
                Dt = 5.*np.pi/180.
                Lim = t0[np.newaxis,:] + Dt*np.r_[-1.,1.][:,np.newaxis]
            elif 'Ripple' in ln[ii]:
                t0 = np.arange(0,360,30)*np.pi/180.
                Dt = 2.5*np.pi/180.
                Lim = t0[np.newaxis,:] + Dt*np.r_[-1.,1.][:,np.newaxis]
            elif 'IC' in ln[ii]:
                t0 = np.arange(0,360,120)*np.pi/180.
                Dt = 10.*np.pi/180.
                Lim = t0[np.newaxis,:] + Dt*np.r_[-1.,1.][:,np.newaxis]
            elif 'LH' in ln[ii]:
                t0 = np.arange(-180,180,120)*np.pi/180.
                Dt = 10.*np.pi/180.
                Lim = t0[np.newaxis,:] + Dt*np.r_[-1.,1.][:,np.newaxis]
            elif tt=='Lin':
                Lim = np.r_[0.,10.]
            else:
                Lim = None

            Poly = np.loadtxt(os.path.join(path, lfc[ii]))
            assert Poly.ndim == 2
            assert Poly.size >= 2*3
            kwd = dict(Name=ln[ii]+tt, Exp=_Exp, SavePath=_here,
                       Poly=Poly, Lim=Lim, Type=tt)
            dobj[tt][cc][ln[ii]] = eval('tfg.{}(**kwd)'.format(cc))


class Test01_Struct(object):
    """ Class for testing the Struct clas and its methods

    In tofu, a Struct is a 3D object defined by a 2D contour in a cross-section
    It has a - toroidal or linear - extension (None if axisymmetric)
    It has methods for plotting, computing key parameters...

    """

    @classmethod
    def setup_class(cls, dobj=dobj):
        #print("")
        #print("---- "+cls.__name__)
        cls.dobj = dobj

    @classmethod
    def teardown_class(cls):
        #print("teardown_class() after any methods in this class")
        pass

    def setup_method(self):
        #print("TestUM:setup_method() before each test method")
        pass

    def teardown_method(self):
        #print("TestUM:teardown_method() after each test method")
        pass

    def test00_set_move(self):
        for typ in self.dobj.keys():
            if typ == 'Tor':
                move = 'rotate_around_torusaxis'
                kwd = {}
            else:
                move = 'translate_in_cross_section'
                kwd = {'direction_rz': [1., 0., 0.]}
            for c in self.dobj[typ].keys():
                for ii, n in enumerate(self.dobj[typ][c].keys()):
                    if ii % 2 == 0:
                        self.dobj[typ][c][n].set_move(move=move,
                                                      **kwd)

    def test01_todict(self):
        for typ in self.dobj.keys():
            for c in self.dobj[typ].keys():
                for n in self.dobj[typ][c].keys():
                    d = self.dobj[typ][c][n].to_dict()
                    assert type(d) is dict
                    assert all([any([s in k for k in d.keys()]
                                    for s in ['Id','geom','sino','strip'])])

    def test02_fromdict(self):
        for typ in self.dobj.keys():
            for c in self.dobj[typ].keys():
                d = list(self.dobj[typ][c].values())[0].to_dict()
                obj = eval('tfg.%s(fromdict=d)'%c)
                assert isinstance(obj,eval('tfg.%s'%c))

    def test03_copy_equal(self):
        for typ in self.dobj.keys():
            for c in self.dobj[typ].keys():
                for n in self.dobj[typ][c].keys():
                    obj = self.dobj[typ][c][n].copy()
                    assert obj == self.dobj[typ][c][n]
                    assert not  obj != self.dobj[typ][c][n]

    def test04_get_nbytes(self):
        for typ in self.dobj.keys():
            for c in self.dobj[typ].keys():
                for n in self.dobj[typ][c].keys():
                    nb, dnb = self.dobj[typ][c][n].get_nbytes()

    def test05_strip_nbytes(self):
        for typ in self.dobj.keys():
            for c in self.dobj[typ].keys():
                lok = eval('tfg.%s'%c)._dstrip['allowed']
                nb = np.full((len(lok),), np.nan)
                for n in self.dobj[typ][c].keys():
                    obj = self.dobj[typ][c][n]
                    for ii in lok:
                        obj.strip(ii)
                        nb[ii] = obj.get_nbytes()[0]
                    assert np.all(np.diff(nb)<0.)
                    for ii in lok[::-1]:
                        obj.strip(ii)

    def test06_set_move_None(self):
        for typ in self.dobj.keys():
            for c in self.dobj[typ].keys():
                for n in self.dobj[typ][c].keys():
                    self.dobj[typ][c][n].set_move()

    def test07_rotate_copy(self):
        for typ in self.dobj.keys():
            if typ == 'Lin':
                continue
            dkwd0 = dict(axis_rz=[2.4, 0], angle=np.pi/4,
                         return_copy=True)
            dkwd1 = dict(direction_rz=[1, 0], distance=0.1,
                         return_copy=True)
            for c in self.dobj[typ].keys():
                for ii, n in enumerate(self.dobj[typ][c].keys()):
                    if ii % 2 == 0:
                        obj = getattr(self.dobj[typ][c][n],
                                      'rotate_in_cross_section')(**dkwd0)
                    else:
                        obj = getattr(self.dobj[typ][c][n],
                                      'translate_in_cross_section')(**dkwd1)

    def test08_set_dsino(self):
        for typ in self.dobj.keys():
            for c in self.dobj[typ].keys():
                for n in self.dobj[typ][c].keys():
                    self.dobj[typ][c][n].set_dsino([2.4,0.])

    def test09_setget_color(self):
        for typ in self.dobj.keys():
            for c in self.dobj[typ].keys():
                for n in self.dobj[typ][c].keys():
                    col = self.dobj[typ][c][n].get_color()
                    self.dobj[typ][c][n].set_color(col)

    def test10_isInside(self, NR=20, NZ=20, NThet=10):
        for typ in self.dobj.keys():
            if tt=='Tor':
                R = np.linspace(1,3,100)
                Z = np.linspace(-1,1,100)
                phi = np.pi/4.
                pts = np.array([np.tile(R,Z.size),
                                np.repeat(Z,R.size),
                                np.full((R.size*Z.size),phi)])
                In = '(R,Z,Phi)'
            else:
                X = 4.
                Y = np.linspace(1,3,100)
                Z = np.linspace(-1,1,100)
                pts = np.array([np.full((Y.size*Z.size),X),
                                np.tile(Y,Z.size),
                                np.repeat(Z,Y.size)])
                In = '(X,Y,Z)'

            for c in self.dobj[typ].keys():
                for n in self.dobj[typ][c].keys():
                    obj = self.dobj[typ][c][n]
                    ind = obj.isInside(pts, In=In)
                    if obj.noccur<=1:
                        assert ind.shape==(pts.shape[1],)
                    elif not ind.shape == (obj.noccur,pts.shape[1]):
                        msg = "ind.shape = {0}".format(str(ind.shape))
                        msg += "\n  But noccur = {0}".format(obj.noccur)
                        msg += "\n  and npts = {0}".format(pts.shape[1])
                        raise Exception(msg)

    def test11_InsideConvexPoly(self):
        for typ in self.dobj.keys():
            for c in self.dobj[typ].keys():
                for n in self.dobj[typ][c].keys():
                    self.dobj[typ][c][n].get_InsideConvexPoly(Plot=False,
                                                              Test=True)

    def test12_get_sampleEdge(self):
        for typ in self.dobj.keys():
            for c in self.dobj[typ].keys():
                for n in self.dobj[typ][c].keys():
                    obj = self.dobj[typ][c][n]
                    out = obj.get_sampleEdge(0.05, resMode='abs',
                                             offsetIn=0.001)
                    out = obj.get_sampleEdge(0.1, resMode='rel',
                                             offsetIn=-0.001)
                    out = obj.get_sampleEdge(0.05, domain=[None, [-2., 0.]],
                                             resMode='abs', offsetIn=0.)

    def test13_get_sampleCross(self):
        for typ in self.dobj.keys():
            for c in self.dobj[typ].keys():
                for n in self.dobj[typ][c].keys():
                    try:
                        obj = self.dobj[typ][c][n]
                        ii = 0
                        out = obj.get_sampleCross(0.02, resMode='abs')
                        ind = out[2]
                        ii = 1
                        out = obj.get_sampleCross(0.02, resMode='abs', ind=ind)
                        PMinMax = (obj.dgeom['P1Min'][0],
                                   obj.dgeom['P1Max'][0])
                        DS1 = PMinMax[0] + (PMinMax[1]-PMinMax[0])/2.
                        ii = 2
                        out = obj.get_sampleCross(0.1, domain=[[None, DS1],
                                                               None],
                                                  resMode='rel')
                    except Exception as err:
                        msg = str(err)
                        msg += "\nFailed for {0}_{1}_{2}".format(typ, c, n)
                        msg += " and ii={0}".format(ii)
                        raise Exception(msg)

    def test14_get_sampleS(self):
        for typ in self.dobj.keys():
            # Todo : introduce possibility of choosing In coordinates !
            for c in self.dobj[typ].keys():
                for n in self.dobj[typ][c].keys():
                    obj = self.dobj[typ][c][n]
                    P1Mm = (obj.dgeom['P1Min'][0], obj.dgeom['P1Max'][0])
                    P2Mm = (obj.dgeom['P2Min'][1], obj.dgeom['P2Max'][1])
                    DS = None#[[2., 3.], [0., 5.], [0., np.pi/2.]]
                    try:
                        ii = 0
                        out = obj.get_sampleS(0.05, resMode='abs', domain=DS,
                                              offsetIn=0.02,
                                              returnas='(X,Y,Z)')
                        pts0, ind = out[0], out[2]
                        ii = 1
                        out = obj.get_sampleS(0.05, resMode='abs', ind=ind,
                                              offsetIn=0.02,
                                              returnas='(X,Y,Z)')
                        pts1 = out[0]
                    except Exception as err:
                        msg = str(err)
                        msg += "\nFailed for {0}_{1}_{2}".format(typ,c,n)
                        msg += "\n    ii={0}".format(ii)
                        msg += "\n    Lim={0}".format(str(obj.Lim))
                        msg += "\n    DS={0}".format(str(DS))
                        raise Exception(msg)

                    if type(pts0) is list:
                        assert all([np.allclose(pts0[ii],pts1[ii])
                                    for ii in range(0,len(pts0))])
                    else:
                        assert np.allclose(pts0,pts1)

    def test15_get_sampleV(self):
        ldomain = [None,
                   [[2., 3.], [0., None], [0., np.pi/2.]]]
        for typ in self.dobj.keys():
            # Todo : introduce possibility of choosing In coordinates !
            for c in self.dobj[typ].keys():
                if issubclass(eval('tfg.%s'%c), tfg._core.StructOut):
                    continue
                for n in self.dobj[typ][c].keys():
                    obj = self.dobj[typ][c][n]
                    for ii in range(len(ldomain)):
                        try:
                            print("Computing pts 0", ldomain[ii])
                            out = obj.get_sampleV(0.1, resMode='abs',
                                                  domain=ldomain[ii],
                                                  returnas='(X,Y,Z)',
                                                  algo='old')
                            pts0, ind0 = out[0], out[2]
                        except Exception as err:
                            msg = (str(err) +
                                   "\nFailed for {0}_{1}_{2}\n".format(typ,
                                                                       c, n)
                                   + "\t- ii = {0}\n".format(ii)
                                   + "\t- Lim = {0}\n".format(obj.Lim)
                                   + "\t- domain = {0}\n".format(ldomain[ii])
                                   + "\t- algo = 'old'"
                                   )
                            raise Exception(msg)
                        try:
                            print("Computing pts 1")
                            out = obj.get_sampleV(0.1, resMode='abs',
                                                  ind=ind0,
                                                  returnas='(X,Y,Z)',
                                                  algo='old')
                            pts1, ind1 = out[0], out[2]
                        except Exception as err:
                            msg = (str(err) +
                                   "\nFailed for {0}_{1}_{2}\n".format(typ,
                                                                       c, n)
                                   + "\t- ii = {0}\n".format(ii)
                                   + "\t- Lim = {0}\n".format(obj.Lim)
                                   + "\t- ind = {0}\n".format(ind0)
                                   + "\t- algo = 'old'"
                                   )
                            raise Exception(msg)
                        try:
                            print("ii, Computing pts 2")
                            print(ldomain[ii])
                            out = obj.get_sampleV(0.1, resMode='abs',
                                                  domain=ldomain[ii],
                                                  returnas='(X,Y,Z)',
                                                  algo='new')
                            pts2, ind2 = out[0], out[2]
                        except Exception as err:
                            msg = (str(err) +
                                   "\nFailed for {0}_{1}_{2}\n".format(typ, c, n)
                                   + "\t- ii = {0}\n".format(ii)
                                   + "\t- Lim = {0}\n".format(obj.Lim)
                                   + "\t- domain = {0}\n".format(ldomain[ii])
                                   + "\t- algo = 'new'"
                                   )
                            raise Exception(msg)
                        try:
                            print("Computing pts 3")
                            out = obj.get_sampleV(0.1, resMode='abs',
                                                  ind=ind0,
                                                  returnas='(X,Y,Z)',
                                                  algo='new')
                            pts3, ind3 = out[0], out[2]
                        except Exception as err:
                            msg = (str(err) +
                                   "\nFailed for {0}_{1}_{2}\n".format(typ,
                                                                       c, n)
                                   + "\t- ii = {0}\n".format(ii)
                                   + "\t- Lim = {0}\n".format(obj.Lim)
                                   + "\t- domain = {0}\n".format(ldomain[ii])
                                   + "\t- ind = {0}\n".format(ind0)
                                   + "\t- algo = 'new'"
                                   )
                            raise Exception(msg)

                        if type(pts0) is list:
                            assert all([np.allclose(pts0[ii], pts1[ii])
                                        for ii in range(0, len(pts0))])
                            assert all([np.allclose(pts0[ii], pts2[ii])
                                        for ii in range(0, len(pts0))])
                            assert all([np.allclose(pts0[ii], pts3[ii])
                                        for ii in range(0, len(pts0))])
                        else:

                            c0 = pts0.shape == pts1.shape
                            c1 = c0 and np.allclose(pts0, pts1)
                            if not c1:
                                msg = ("Volume sampling:\n"
                                       + "\t- bad reconstruction from ind\n"
                                       + "\t- (old algo)\n"
                                       + "\t- same shape: {}\n".format(c0)
                                       + "\t- np.allclose() {}\n".format(c1)
                                       + "\t- domain= {}\n".format(ldomain[ii])
                                       + "\t- ind = {}".format(ind0))
                                raise Exception(msg)

                            c0 = pts0.shape == pts2.shape
                            c1 = c0 and np.allclose(pts0, pts2)

                            if not c0:
                                lax = obj.plot()
                                lax[1].plot(pts0[0, :],
                                            pts0[1, :], '.b')  # x, y
                                lax[0].plot(np.sqrt(pts0[0, :]**2
                                                    + pts0[1, :]**2),
                                            pts0[2, :], '.r')  # r, z
                                plt.title("pts0 : OLD")
                                plt.show(block=True)
                                lax = obj.plot()
                                lax[1].plot(pts2[0, :],
                                            pts2[1, :], '.b')  # x, y
                                lax[0].plot(np.sqrt(pts2[0, :]**2
                                                    + pts2[1, :]**2),
                                            pts2[2, :], '.r')  # r, z
                                plt.title("pts2 : NEW")
                                plt.show(block=True)

                                n1 = pts0.shape
                                n2 = pts2.shape
                                msg = ("Volume sampling:\n"
                                       + "\t- no match old vs new algo\n"
                                       + "\t- same shape: {} ".format(c0)
                                       + "(old : {}, new: {})\n".format(n1,
                                                                        n2)
                                       + "\t- np.allclose() {}\n".format(c1)
                                       + "\t- domain = {}".format(ldomain[ii]))
                                raise Exception(msg)

                            c0 = pts0.shape == pts3.shape
                            c1 = c0 and np.allclose(pts0, pts3)
                            if not c0:
                                msg = ("Volume sampling:\n"
                                       + "\t- bad reconstruction from ind\n"
                                       + "\t- (new algo)\n"
                                       + "\t- same shape: {}\n".format(c0)
                                       + "\t- np.allclose() {}\n".format(c1)
                                       + "\t- domain = {}".format(ldomain[ii])
                                       + "\t- ind = {}".format(ind0))
                                raise Exception(msg)

    def test16_plot(self):
        for typ in self.dobj.keys():
            for c in self.dobj[typ].keys():
                for n in self.dobj[typ][c].keys():
                    obj = self.dobj[typ][c][n]
                    lax = obj.plot(element='P', proj='all', draw=False)
                    lax = obj.plot(element='PIBsBvV', proj='all', draw=False)
                    lax = obj.plot(element='P', indices=True, draw=False)
                    plt.close('all')

    def test17_plot_sino(self):
        for typ in self.dobj.keys():
            for c in self.dobj[typ].keys():
                if issubclass(eval('tfg.%s'%c), tfg._core.StructOut):
                    continue
                for n in self.dobj[typ][c].keys():
                    obj = self.dobj[typ][c][n]
                    lax = obj.plot_sino(Ang='xi', AngUnit='deg',
                                        Sketch=True, draw=False)
                    lax = obj.plot_sino(Ang='theta', AngUnit='rad',
                                        Sketch=False, draw=False, fs='a4')
                    plt.close('all')

    def test18_saveload(self, verb=False):
        for typ in self.dobj.keys():
            for c in self.dobj[typ].keys():
                for n in self.dobj[typ][c].keys():
                    obj = self.dobj[typ][c][n]
                    pfe = obj.save(return_pfe=True, verb=verb)
                    obj2 = tf.load(pfe, verb=verb)
                    assert obj==obj2
                    os.remove(pfe)

    def test19_save_to_txt(self, verb=False):
        for typ in self.dobj.keys():
            for c in self.dobj[typ].keys():
                for n in self.dobj[typ][c].keys():
                    obj = self.dobj[typ][c][n]
                    pfe = obj.save_to_txt(return_pfe=True, verb=verb)
                    os.remove(pfe)


#######################################################
#
#  Creating Config objects and testing methods
#
#######################################################


# Define a dict dconfig holding all the typical Config we want to test


dconf = dict.fromkeys(dobj.keys())
for typ in dobj.keys():

    # Get list of structures (lS) composing the config
    lS = list(itt.chain.from_iterable(
        [list(dobj[typ][c].values()) for c in dobj[typ].keys()]
    ))

    # Set the limits (none in toroidal geometry, [0., 10.] in linear geometry)
    Lim = None if typ == 'Tor' else [0., 10.]

    # Create config and store in dict
    dconf[typ] = tfg.Config(Name='Test%s'%typ, Exp=_Exp,
                            lStruct=lS, Lim=Lim,
                            Type=typ, SavePath=_here)


class Test02_Config(object):
    """ Class for testing the Config class and its methods

    A Config class holds the geometrical configuration of a tokamak
    It holds all the structural elements constituting it
    It provides methods to plot, move them

    """

    @classmethod
    def setup_class(cls, dobj=dconf, verb=False):
        #print("")
        #print("--------- "+VerbHead+cls.__name__)
        cls.dlpfe = {}
        for typ in dobj.keys():
            lS = dobj[typ].lStruct
            lpfe = [os.path.join(ss.Id.SavePath, ss.Id.SaveName+'.npz')
                    for ss in lS]
            for ss in lS:
                ss.save(verb=verb)
            cls.dlpfe[typ] = lpfe
        cls.dobj = dobj

    @classmethod
    def teardown_class(cls):
        #print("teardown_class() after any methods in this class")
        for typ in cls.dlpfe.keys():
            for f in cls.dlpfe[typ]:
                os.remove(f)

    def test01_todict(self):
        for typ in self.dobj.keys():
            d = self.dobj[typ].to_dict()
            assert type(d) is dict
            assert all([any([s in k for k in d.keys()]
                            for s in ['Id','geom','struct','sino','strip'])])

    def test02_fromdict(self):
        for typ in self.dobj.keys():
            d = self.dobj[typ].to_dict()
            obj = tfg.Config(fromdict=d)
            assert isinstance(obj, tfg.Config)

    def test03_copy_equal(self):
        for typ in self.dobj.keys():
            obj = self.dobj[typ].copy()
            assert obj == self.dobj[typ]
            assert not  obj != self.dobj[typ]

    def test04_get_nbytes(self):
        for typ in self.dobj.keys():
            nb, dnb = self.dobj[typ].get_nbytes()

    def test05_strip_nbytes(self, verb=False):
        lok = tfg.Config._dstrip['allowed']
        nb = np.full((len(lok),), np.nan)
        for typ in self.dobj.keys():
            obj = self.dobj[typ]
            for ii in lok:
                obj.strip(ii, verb=verb)
                nb[ii] = obj.get_nbytes()[0]
            assert np.all(np.diff(nb)<0.)
            for ii in lok[::-1]:
                obj.strip(ii, verb=verb)

    def test06_set_dsino(self):
        for typ in self.dobj.keys():
            self.dobj[typ].set_dsino([2.4,0.])

    def test07_addremove_struct(self):
        for typ in self.dobj.keys():
            obj = self.dobj[typ]
            n = [ss.Id.Name for ss in obj.lStruct if 'Baffle' in ss.Id.Name]
            n = n[0]
            B = obj.dStruct['dObj']['PFC'][n].copy()
            assert n in obj.dStruct['dObj']['PFC'].keys()
            assert n in obj.dextraprop['dvisible']['PFC'].keys()
            assert hasattr(obj.PFC,n)
            obj.remove_Struct('PFC', n)
            assert n not in obj.dStruct['dObj']['PFC'].keys()
            assert n not in obj.dextraprop['dvisible']['PFC'].keys()
            try:
                hasattr(obj.PFC, n)
            except Exception as err:
                assert err.__class__.__name__=='KeyError'
            self.dobj[typ].add_Struct(
                struct=B,
                dextraprop={'visible': True},
            )
            assert n in obj.dStruct['dObj']['PFC'].keys()
            assert n in obj.dextraprop['dvisible']['PFC'].keys()
            assert hasattr(obj.PFC,n)

    def test08_setget_color(self):
        for typ in self.dobj.keys():
            col = self.dobj[typ].get_color()
            for c in self.dobj[typ].dStruct['dObj'].keys():
                eval("self.dobj[typ].%s.set_color('r')"%c)

    def test09_get_summary(self):
        for typ in self.dobj.keys():
            self.dobj[typ].get_summary()

    def test10_isInside(self, NR=20, NZ=20, NThet=10):
        for typ in self.dobj.keys():
            if typ=='Tor':
                R = np.linspace(1,3,100)
                Z = np.linspace(-1,1,100)
                phi = np.pi/4.
                pts = np.array([np.tile(R,Z.size),
                                np.repeat(Z,R.size),
                                np.full((R.size*Z.size),phi)])
                In = '(R,Z,Phi)'
            else:
                X = 4.
                Y = np.linspace(1,3,100)
                Z = np.linspace(-1,1,100)
                pts = np.array([np.full((Y.size*Z.size),X),
                                np.tile(Y,Z.size),
                                np.repeat(Z,Y.size)])
                In = '(X,Y,Z)'

                obj = self.dobj[typ]
                ind = obj.isInside(pts, In=In)
                if not ind.shape == (obj.nStruct, pts.shape[1]):
                    msg = "ind.shape = {0}".format(str(ind.shape))
                    msg += "\n  But nStruct = {0}".format(obj.nStruct)
                    msg += "\n  and npts = {0}".format(pts.shape[1])
                    raise Exception(msg)

    def test11_setget_visible(self):
        for typ in self.dobj.keys():
            vis = self.dobj[typ].get_visible()
            self.dobj[typ].CoilPF.set_visible(False)

    def test12_plot(self):
        for typ in self.dobj.keys():
            n = [ss.Id.Name for ss in self.dobj[typ].lStruct
                 if 'Baffle' in ss.Id.Name][0]
            eval("self.dobj[typ].PFC.%s.set_color('g')"%n)
            lax = self.dobj[typ].plot()
        plt.close('all')

    def test13_plot_sino(self):
        for typ in self.dobj.keys():
            lax = self.dobj[typ].plot_sino()
        plt.close('all')

    def test14_from_svg(self):
        pfe = os.path.join(_here, 'test_data', 'Inkscape.svg')
        # to be solved when optional dependence svg.path is handled
        # (or integrated)
        conf = tfg.Config.from_svg(pfe, Name='Test', Exp='Test', res=10)
        conf = tfg.Config.from_svg(
            pfe, Name='Test', Exp='Test',
            res=10, r0=-100, z0=-150, scale=0.01,
        )
        conf = tfg.Config.from_svg(
            pfe, Name='Test', Exp='Test',
            res=10, point_ref1=(0.7, -2), point_ref2=(2.8, 2),
        )
        conf = tfg.Config.from_svg(
            pfe, Name='Test', Exp='Test',
            res=10, point_ref1=(0.7, -2), length_ref=4.5,
        )

    def test15_load_config(self):
        lc = sorted(tfg.utils._get_listconfig(returnas=dict).keys())
        for cc in lc:
            conf = tf.load_config(cc, strict=True)

    def test16_calc_solidangle_particle(self):
        conf = tf.load_config('AUG', strict=True)
        pts = np.array([[2.5, 0., 0.], [2.5, 0., 0.5]])
        theta = np.linspace(-1, 1, 4)*np.pi/4.
        part_traj = np.array([
            2.4*np.cos(theta),
            2.4*np.sin(theta),
            0*theta,
        ])
        part_radius = np.array([1e-6, 10e-6, 100e-6, 1e-3])
        out = conf.calc_solidangle_particle(
            pts=pts,
            part_traj=part_traj,
            part_radius=part_radius,
        )

    def test17_calc_solidangle_particle_integrated(self):
        conf = tf.load_config('WEST', strict=True)
        theta = np.linspace(-1, 1, 4)*np.pi/4.
        part_traj = np.array([
            2.4*np.cos(theta),
            2.4*np.sin(theta),
            0*theta,
        ])
        part_radius = np.array([1e-6, 10e-6, 100e-6, 1e-3])
        out = conf.calc_solidangle_particle_integrated(
            part_traj=part_traj,
            part_radius=part_radius,
            resolution=0.2,
        )
        plt.close('all')

    def test18_saveload(self, verb=False):
        for typ in self.dobj.keys():
            self.dobj[typ].strip(-1)
            pfe = self.dobj[typ].save(verb=verb, return_pfe=True)
            obj = tf.load(pfe, verb=verb)
            msg = "Unequal saved / loaded objects !"
            assert obj==self.dobj[typ], msg
            # Just to check the loaded version works fine
            obj.strip(0, verb=verb)
            os.remove(pfe)


#######################################################
#
#     Creating Rays objects and testing methods
#
#######################################################


# Define a dict of cams to be tested
def get_dCams(dconf=dconf):
    dCams = {}
    foc = 0.08
    DX = 0.05
    for typ in dconf.keys():
        dCams[typ] = {}
        if typ == 'Tor':
            phi = np.pi/4.
            eR = np.r_[np.cos(phi), np.sin(phi), 0.]
            ephi = np.r_[np.sin(phi), -np.cos(phi), 0.]
            R = 3.5
            ph = np.r_[R*np.cos(phi), R*np.sin(phi), 0.2]
        else:
            ph = np.r_[3., 4., 0.]
        ez = np.r_[0., 0., 1.]
        for c in ['CamLOS2D', 'CamLOS1D']:
            if '1D' in c:
                nP = 100
                X = np.linspace(-DX, DX, nP)
                if typ == 'Tor':
                    D = (ph[:, np.newaxis] + foc*eR[:, np.newaxis]
                         + X[np.newaxis, :]*ephi[:, np.newaxis])
                else:
                    D = np.array([3. + X,
                                  np.full((nP,), 4. + foc),
                                  np.full((nP,), 0.02)])
            else:
                if typ == 'Tor':
                    nP = 100
                    X = np.linspace(-DX, DX, nP)
                    D = (
                        ph[:, None] + foc*eR[:, None]
                        + np.repeat(X[::-1], nP)[None, :]*ephi[:, None]
                        + np.tile(X, nP)[None, :]*ez[:, None]
                    )
                else:
                    nP = 100
                    X = np.linspace(-DX, DX, nP)
                    D = np.array([np.repeat(3. + X[::-1], nP),
                                  np.full((nP*nP,), 4. + foc),
                                  np.tile(0.01 + X, nP)])
            cls = eval(f"tfg.{c}")
            assert len(dconf[typ].lStruct) > 0
            dCams[typ][c] = cls(
                Name='V1000', config=dconf[typ],
                dgeom={'pinhole': ph, 'D': D}, method="optimized",
                Exp=_Exp, Diag='Test', SavePath=_here,
            )
    return dCams


class Test03_Rays(object):

    @classmethod
    def setup_class(cls, dobj=get_dCams(), verb=False):
        #print ("")
        #print "--------- "+VerbHead+cls.__name__
        dlpfe = {}
        for typ in dobj.keys():
            dlpfe[typ] = {}
            for c in dobj[typ].keys():
                dlpfe[typ][c] = []
                for s in dobj[typ][c].config.lStruct:
                    pfe = os.path.join(s.Id.SavePath,s.Id.SaveName+'.npz')
                    s.save(verb=verb)
                    dlpfe[typ][c].append(pfe)
                dobj[typ][c].config.strip(-1)
                dobj[typ][c].config.save(verb=verb)
                dobj[typ][c].config.strip(0, verb=verb)
                pfe = os.path.join(dobj[typ][c].config.Id.SavePath,
                                   dobj[typ][c].config.Id.SaveName+'.npz')
                dlpfe[typ][c].append(pfe)
        cls.dobj = dobj
        cls.dlpfe = dlpfe

    @classmethod
    def teardown_class(cls):
        #print ("teardown_class() after any methods in this class")
        pass
        # for typ in cls.dobj.keys():
            # for c in cls.dobj[typ].keys():
                # for f in cls.dlpfe[typ][c]:
                    # try:
                        # os.remove(f)
                    # except Exception as err:
                        # msg = str(err)
                        # msg += '\n\n'+str(f)
                        # raise Exception(msg)

    def setup_method(self):
        #print ("TestUM:setup_method() before each test method")
        pass

    def teardown_method(self):
        #print ("TestUM:teardown_method() after each test method")
        pass

    def test00_set_move(self):
        for typ in self.dobj.keys():
            if typ == 'Tor':
                move = 'rotate_around_torusaxis'
                kwd = {}
            else:
                move = 'translate_in_cross_section'
                kwd = {'direction_rz': [1., 0., 0.]}
            for c in self.dobj[typ].keys():
                self.dobj[typ][c].set_move(move=move, **kwd)

    def test01_todict(self):
        for typ in self.dobj.keys():
            for c in self.dobj[typ].keys():
                d = self.dobj[typ][c].to_dict()
                assert type(d) is dict
                assert all([any([s in k for k in d.keys()]
                                for s in ['Id','geom','config',
                                          'sino','chan','strip'])])

    def test02_fromdict(self):
        for typ in self.dobj.keys():
            for c in self.dobj[typ].keys():
                d = self.dobj[typ][c].to_dict()
                obj = eval('tfg.%s(fromdict=d)'%c)
                assert isinstance(obj, self.dobj[typ][c].__class__)

    def test03_copy_equal(self):
        for typ in self.dobj.keys():
            for c in self.dobj[typ].keys():
                obj = self.dobj[typ][c].copy()
                assert obj == self.dobj[typ][c]
                assert not  obj != self.dobj[typ][c]

    def test04_get_nbytes(self):
        for typ in self.dobj.keys():
            for c in self.dobj[typ].keys():
                nb, dnb = self.dobj[typ][c].get_nbytes()

    def test05_strip_nbytes(self, verb=False):
        lok = tfg.Rays._dstrip['allowed']
        nb = np.full((len(lok),), np.nan)
        for typ in self.dobj.keys():
            for c in self.dobj[typ].keys():
                obj = self.dobj[typ][c]
                for ii in lok:
                    obj.strip(ii, verb=verb)
                    nb[ii] = obj.get_nbytes()[0]
                assert np.all(np.diff(nb)<0.)
                for ii in lok[::-1]:
                    obj.strip(ii, verb=verb)

    def test06_set_move_None(self):
        for typ in self.dobj.keys():
            for c in self.dobj[typ].keys():
                self.dobj[typ][c].set_move()

    def test07_rotate_copy(self):
        for typ in self.dobj.keys():
            if typ == 'Lin':
                continue
            dkwd0 = dict(axis_rz=[2.4, 0], angle=np.pi/4,
                         return_copy=True)
            for c in self.dobj[typ].keys():
                obj = getattr(self.dobj[typ][c],
                              'rotate_in_cross_section')(**dkwd0)

    def test07_set_dsino(self):
        for typ in self.dobj.keys():
            for c in self.dobj[typ].keys():
                self.dobj[typ][c].set_dsino([2.4,0.])

    def test08_select(self):
        for typ in self.dobj.keys():
            for c in self.dobj[typ].keys():
                n = [ss.Id.Name for ss in self.dobj[typ][c].config.lStruct
                     if 'Baffle' in ss.Id.Name][0]
                ind = self.dobj[typ][c].select(touch='PFC_%s'%n)
                ind = self.dobj[typ][c].select(touch=['PFC_%s'%n,[],[7,8,9]])

    def test09_get_sample(self):
        for typ in self.dobj.keys():
            for c in self.dobj[typ].keys():
                obj = self.dobj[typ][c]
                Ds, us = obj.D[:], obj.u[:]
                out = obj.get_sample(0.02, resMode='abs',
                                     method='sum',DL=None)
                k, res, lind = out
                # nbrepet = np.r_[lind[0], np.diff(lind), k.size - lind[-1]]
                # kus = k * np.repeat(us, nbrepet, axis=1)
                # Pts = np.repeat(Ds, nbrepet, axis=1) + kus
                k = np.split(k, lind)
                assert len(res) == len(k) == obj.nRays
                for ii in range(0, len(k)):
                    if not (np.isnan(obj.kIn[ii]) or np.isnan(obj.kOut[ii])):
                        ind = ~np.isnan(k[ii])
                        assert np.all((k[ii][ind]>=obj.kIn[ii]-res[ii])
                                      & (k[ii][ind]<=obj.kOut[ii]+res[ii]))
                assert np.all(res[~np.isnan(res)]<0.02)

                out = obj.get_sample(0.1, resMode='rel',
                                     method='simps',DL=[0,1])
                k, res, lind = out
                k = np.split(k, lind)
                assert len(res)==len(k)==obj.nRays
                for ii in range(0,len(k)):
                    if not (np.isnan(obj.kIn[ii]) or np.isnan(obj.kOut[ii])):
                        ind = ~np.isnan(k[ii])
                        if not np.all((k[ii][ind]>=obj.kIn[ii]-res[ii])
                                      & (k[ii][ind]<=obj.kOut[ii]+res[ii])):
                            msg = typ+' '+c+' '+str(ii)
                            msg += "\n {0} {1}".format(obj.kIn[ii],obj.kOut[ii])
                            msg += "\n {0}".format(str(k[ii][ind]))
                            print(msg)
                            raise Exception(msg)

                out = obj.get_sample(0.1, resMode='rel',
                                     method='romb',DL=[0,1])
                k, res, lind = out
                k = np.split(k, lind)
                assert len(res)==len(k)==obj.nRays
                for ii in range(0,len(k)):
                    if not (np.isnan(obj.kIn[ii]) or np.isnan(obj.kOut[ii])):
                        ind = ~np.isnan(k[ii])
                        assert np.all((k[ii][ind]>=obj.kIn[ii]-res[ii])
                                      & (k[ii][ind]<=obj.kOut[ii]+res[ii]))

    def test10_calc_kInkOut_Isoflux(self):
        nP = 10
        r = np.linspace(0.1,0.4,nP)
        theta = np.linspace(0.,2*np.pi,100)
        lp2D = [np.array([2.4+r[ii]*np.cos(theta),
                          0.+r[ii]*np.sin(theta)]) for ii in range(0,nP)]
        for typ in self.dobj.keys():
            for c in self.dobj[typ].keys():
                obj = self.dobj[typ][c]
                kIn, kOut = obj.calc_kInkOut_Isoflux(lp2D)
                assert kIn.shape == (nP, obj.nRays)
                assert kOut.shape == (nP, obj.nRays)
                for ii in range(0, nP):
                    ind = ~np.isnan(kIn[ii, :])
                    if not np.all((kIn[ii, ind] >= obj.kIn[ind])
                                  & (kIn[ii, ind] <= obj.kOut[ind])):
                        msg = typ+' '+c+' '+str(ii)
                        msg += "\n {0} {1}".format(obj.kIn[ind], obj.kOut[ind])
                        msg += "\n {0}".format(str(kIn[ii, ind]))
                        raise Exception(msg)

                    ind = ~np.isnan(kOut[ii, :])
                    if not np.all((kOut[ii, ind] >= obj.kIn[ind])
                                  & (kOut[ii, ind] <= obj.kOut[ind])):
                        msg = typ+' '+c+' '+str(ii)
                        msg += "\n {0} {1}".format(obj.kIn[ind], obj.kOut[ind])
                        msg += "\n {0}".format(str(kOut[ii, ind]))
                        raise Exception(msg)
                    ind = (~np.isnan(kIn[ii, :])) & (~np.isnan(kOut[ii, :]))
                    if not np.all(kIn[ii, ind] <= kOut[ii, ind]):
                        msg = typ+' '+c+' '+str(ii)
                        msg += "\n {0}".format(str(kIn[ii, ind]))
                        msg += "\n {0}".format(str(kOut[ii, ind]))
                        raise Exception(msg)

    def test11_calc_signal(self):
        def ffL(Pts, t=None, vect=None):
            E = np.exp(-(Pts[1,:]-2.4)**2/0.1 - Pts[2,:]**2/0.1)
            if vect is not None:
                if np.asarray(vect).ndim==2:
                    E = E*vect[0,:]
                else:
                    E = E*vect[0]
            if t is not None:
                E = E[np.newaxis,:]*t
            return E
        def ffT(Pts, t=None, vect=None):
            E = np.exp(-(np.hypot(Pts[0,:],Pts[1,:])-2.4)**2/0.1
                       - Pts[2,:]**2/0.1)
            if vect is not None:
                if np.asarray(vect).ndim==2:
                    E = E*vect[0,:]
                else:
                    E = E*vect[0]
            if t is not None:
                E = E[np.newaxis,:]*t
            return E

        ind = None#[0,10,20,30,40]
        minimize = ["memory", "calls", "hybrid"]
        for typ in self.dobj.keys():
            c = 'CamLOS1D'
            obj = self.dobj[typ][c]
            for aa in [True, False]:
                rm = 'rel'
                # for rm in ["abs", "rel"]:
                sigref, ii = None, 0
                for dm in ["simps", "romb", "sum"]:
                    for mmz in minimize:
                        ff = ffT if obj.config.Id.Type == 'Tor' else ffL
                        t = np.arange(0, 10, 10)
                        connect = (hasattr(plt.get_current_fig_manager(),
                                           'toolbar')
                                   and getattr(plt.get_current_fig_manager(),
                                               'toolbar')
                                   is not None)
                        out = obj.calc_signal(ff, t=t, ani=aa,
                                              fkwdargs={},
                                              res=0.01, DL=None,
                                              resMode=rm,
                                              method=dm, minimize=mmz,
                                              ind=ind,
                                              plot=False, returnas=np.ndarray,
                                              fs=(12, 6), connect=connect)
                        sig, units = out
                        assert not np.all(np.isnan(sig)), str(ii)
                        if sigref is not None:
                            assert np.allclose(sig, sigref)
                        if obj.nRays <= 100 and ii == 0:
                            sigref = sig
                            ii += 1
        plt.close('all')

    def test12_plot(self):
        for typ in self.dobj.keys():
            for c in self.dobj[typ].keys():
                obj = self.dobj[typ][c]
                if '2D' in c:
                    ind = np.arange(0,obj.nRays,100)
                else:
                    ind = None
                try:
                    lax = obj.plot(proj='all', element='LDIORP',
                                   Leg='', draw=False)
                    lax = obj.plot(proj='cross', element='L',
                                   Leg=None, draw=False)
                    lax = obj.plot(proj='hor', element='LDIO',
                                   Leg='KD', draw=False)
                except Exception as err:
                    pass
                    # msg = str(err)
                    # msg += typ+' '+c
                    # print(msg)
                plt.close('all')

    def test13_plot_sino(self):
        for typ in self.dobj.keys():
            for c in self.dobj[typ].keys():
                obj = self.dobj[typ][c]
                ax = obj.plot_sino()
            plt.close('all')

    def test14_plot_touch(self):
        connect = (hasattr(plt.get_current_fig_manager(),'toolbar')
                   and getattr(plt.get_current_fig_manager(),'toolbar')
                   is not None)
        for typ in self.dobj.keys():
            for c in self.dobj[typ].keys():
                obj = self.dobj[typ][c]
                ind = np.arange(0,obj.nRays,100)
                lax = obj.plot_touch(ind=ind, connect=connect)
            plt.close('all')

    def test15_saveload(self, verb=False):
        for typ in self.dobj.keys():
            for c in self.dobj[typ].keys():
                obj = self.dobj[typ][c]
                obj.strip(-1, verb=verb)
                pfe = obj.save(verb=verb, return_pfe=True)
                obj2 = tf.load(pfe, verb=verb)
                msg = "Unequal saved / loaded objects !"
                assert obj2==obj, msg
                # Just to check the loaded version works fine
                obj2.strip(0, verb=verb)
                os.remove(pfe)

    def test16_get_sample_same_res_unit(self):
        dmeths = ['rel', 'abs']
        qmeths = ['simps', 'romb', 'sum']
        list_res = [0.25, np.r_[0.2, 0.5]]
        DL = np.array([[1.,10.],[2.,20.]])

        for dL in list_res:
            for dm in dmeths:
                for qm in qmeths:
                    out = tfg._GG.LOS_get_sample(2, dL, DL, dmethod=dm, method=qm)
                    k = out[0]
                    lind = out[2]
                    assert np.all(k[:lind[0]] >= DL[0][0])
                    assert np.all(k[:lind[0]] <= DL[1][0])
                    assert np.all(k[lind[0]:] >= DL[0][1])
                    assert np.all(k[lind[0]:] <= DL[1][1])


"""
class Test04_LOSCams(Test03_Rays):

    @classmethod
    def setup_class(cls):
        #print ("")
        #print "--------- "+VerbHead+cls.__name__
        LVes = [Test01_Ves.VesLin]*3+[Test01_Ves.VesTor]*3
        LS = [None, Test02_Struct.SL0, [Test02_Struct.SL0,Test02_Struct.SL1],
              None, Test02_Struct.ST0, [Test02_Struct.ST0,Test02_Struct.ST1]]
        cls.LObj = [None for vv in LVes]
        N = 50
        cls.N = N
        for ii in range(0,len(LVes)):
            P1M = LVes[ii].geom['P1Max'][0]
            dchans = {'Name':['{0:02.0f}'.format(jj) for jj in range(0,N)]}
            if ii%2==0:
                Ds = np.array([np.linspace(-0.5,0.5,N),
                               np.full((N,),(0.95+0.3*ii/len(LVes))*P1M),
                               np.zeros((N,))])
                us = np.array([np.linspace(-0.5,0.5,N),
                               -np.ones((N,)),
                               np.linspace(-0.5,0.5,N)])
                cls.LObj[ii] = tfg.LOSCam1D('Test'+str(ii), (Ds,us), Ves=LVes[ii],
                                           LStruct=LS[ii], Exp=None, Diag='Test',
                                           SavePath=here, dchans=dchans)
            else:
                Ds = np.array([np.repeat(np.linspace(-0.5,0.5,int(N/10)),10),
                               np.full((N,),(0.95+0.3*ii/len(LVes))*P1M),
                               np.tile(np.linspace(-0.1,0.1,10),int(N/10))])
                us = np.array([np.linspace(-0.5,0.5,N),
                               -np.ones((N,)),
                               np.linspace(-0.5,0.5,N)])
                cls.LObj[ii] = tfg.LOSCam2D('Test'+str(ii), (Ds,us), Ves=LVes[ii],
                                            LStruct=LS[ii], Exp=None, Diag='Test',
                                            SavePath=here, dchans=dchans)

    def test01_select(self):
        for ii in range(0,len(self.LObj)):
            if self.LObj[ii].LStruct is None:
                el = 'Ves'
            else:
                el = self.LObj[ii].LStruct[-1].Id.Name
            ind = self.LObj[ii].select(touch=el)
            ind = self.LObj[ii].select(key='Name', val='15', out=bool)
            ind = self.LObj[ii].select(key='Name', val=['02','35'], out=int)
"""


"""
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

    def setup_method(self):
        #print ("TestUM:setup_method() before each test method")
        pass

    def teardown_method(self):
        #print ("TestUM:teardown_method() after each test method")
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
        self.Obj.save(verb=False)
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

    def setup_method(self):
        #print ("TestUM:setup_method() before each test method")
        pass

    def teardown_method(self):
        #print ("TestUM:teardown_method() after each test method")
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
        self.Obj.save(verb=False)
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

    def setup_method(self):
        #print ("TestUM:setup_method() before each test method")
        pass

    def teardown_method(self):
        #print ("TestUM:teardown_method() after each test method")
        pass

    def test01_plot(self):
        Lax = self.Obj.plot(Proj='All', Elt='PV', EltVes='P', draw=False, a4=False, Test=True)
        Lax = self.Obj.plot(Proj='Cross', Elt='P', EltVes='', draw=False, a4=False, Test=True)
        Lax = self.Obj.plot(Proj='Hor', Elt='V', EltVes='P', draw=False, a4=False, Test=True)
        plt.close('all')

    def test02_saveload(self):
        self.Obj.save(verb=False)
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

    def setup_method(self):
        #print ("TestUM:setup_method() before each test method")
        pass

    def teardown_method(self):
        #print ("TestUM:teardown_method() after each test method")
        pass

    def test01_plot(self):
        Lax = self.Obj.plot(Proj='All', Elt='PV', EltVes='P', draw=False, a4=False, Test=True)
        Lax = self.Obj.plot(Proj='Cross', Elt='P', EltVes='', draw=False, a4=False, Test=True)
        Lax = self.Obj.plot(Proj='Hor', Elt='V', EltVes='P', draw=False, a4=False, Test=True)
        plt.close('all')

    def test02_saveload(self):
        self.Obj.save(verb=False)
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

    def setup_method(self):
        #print ("TestUM:setup_method() before each test method")
        pass

    def teardown_method(self):
        #print ("TestUM:teardown_method() after each test method")
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
        self.Obj.save(verb=False)
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

    def setup_method(self):
        #print ("TestUM:setup_method() before each test method")
        pass

    def teardown_method(self):
        #print ("TestUM:teardown_method() after each test method")
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
        self.Obj.save(verb=False)
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

    def setup_method(self):
        #print ("TestUM:setup_method() before each test method")
        pass

    def teardown_method(self):
        #print ("TestUM:teardown_method() after each test method")
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
        self.Obj.save(verb=False)
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

    def setup_method(self):
        #print ("TestUM:setup_method() before each test method")
        pass

    def teardown_method(self):
        #print ("TestUM:teardown_method() after each test method")
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
        self.Obj.save(verb=False)
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

    def setup_method(self):
        #print ("TestUM:setup_method() before each test method")
        pass

    def teardown_method(self):
        #print ("TestUM:teardown_method() after each test method")
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
        self.Obj.save(verb=False)
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

    def setup_method(self):
        #print ("TestUM:setup_method() before each test method")
        pass

    def teardown_method(self):
        #print ("TestUM:teardown_method() after each test method")
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
        self.Obj.save(verb=False)
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

    def setup_method(self):
        #print ("TestUM:setup_method() before each test method")
        pass

    def teardown_method(self):
        #print ("TestUM:teardown_method() after each test method")
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
        self.Obj.save(verb=False)
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

    def setup_method(self):
        #print ("TestUM:setup_method() before each test method")
        pass

    def teardown_method(self):
        #print ("TestUM:teardown_method() after each test method")
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
        self.Obj.save(verb=False)
        obj = tfpf.Open(self.Obj.Id.SavePath + self.Obj.Id.SaveName + '.npz')
        os.remove(self.Obj.Id.SavePath + self.Obj.Id.SaveName + '.npz')

"""
