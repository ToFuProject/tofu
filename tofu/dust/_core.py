



import numpy as np
import matplotlib.pyplot as plt

# ToFu-specific
import tofu.geom as tfg
import tofu.dust._comp as _comp







class Dust(object):
    """ A generic Dust class to handle dust trajectories, radiative power
    balance...

    """

    def __init__(self):
        # Initialize fields
        self._dmat = None
        self._geom = None
        self._res = None
        self._emiss = None
        self._direct = None

    @property
    def dmat(self):
        return self._dmat
    @property
    def Ves(self):
        return self._geom['Ves']
    @property
    def LStruct(self):
        return self._geom['LStruct']
    @property
    def res(self):
        return self._res
    @property
    def emiss(self):
        return self._emiss
    @property
    def direct(self):
        return self._direct

    def set_material_properties(self, dmat={}):
        assert type(dmat) is dict
        self._dmat = dmat

    def set_geom(self, Ves=None, LStruct=None, res=None):
        msg = "Arg Ves must be a tf.geom.Ves"
        assert Ves is None or isinstance(Ves,tfg.Ves), msg
        C0 = LStruct is None
        C1 = isinstance(LStruct,tfg.Struct)
        C2 = (type(LStruct) is list
              and all([isinstance(ss,tfg.Struct) for ss in LStruct]))
        msg = ""
        assert C0 or C1 or C2, msg
        assert Ves is not None or LStruct is None, "Ves must be provided !"

        LStruct = [Lstruct] if isinstance(LStruct,tfg.Struct) else LStruct
        self._geom = {'Ves':Ves, 'LStruct':LStruct}
        self._sampleV = None
        if res is not None:
            self.set_sampleV(res)

    def set_sampleV(self, res=None):
        #assert 
        self._res = res

    def set_emiss(self, emiss=None, t=None, Ani=None, axisym=True):
        ani = tfg._GG.check_ff(emiss, t=t, Ani=Ani, Vuniq=False)
        self._emiss = {'ff':emiss, 't':t, 'ani':ani, 'axisym':axisym}

    def set_directproblem(self, pts=None, r=None, t=None):
        C0 = type(pts) is np.ndarray
        C1 = C0 and pts.ndim==2 and pts.shape[0]==3
        C2 = C0 and pts.shape==(3,)
        msg = "pts must be a (3,) or (3,N) np.ndarray (X,Y,Z coordinates of N points)"
        assert C1 or C2, msg
        if C2:
            pts = pts.reshape((3,1))
        nt = pts.shape[1]

        C0 = type(r) in [int, float, np.int64, np.float64]
        C1 = type(r) is np.ndarray and r.ndim==1
        C2 = C1 and r.size>1
        msg = "r must be a float or a 1D np.ndarray"
        assert C0 or C1, msg
        if C0:
            r = np.array([r],dtype=float)
        if r.size==1:
            r = np.tile(r,nt)
        assert r.size==nt, "r must be a (nt,) np.ndarray !"

        C0 = t is None or type(t) is np.ndarray and t.shape==(nt,)
        msg = "t must be None or a (N,) np.ndarray"
        assert C0, msg
        if t is None:
            t = np.arange(0,nt)

        self._direct = {'pts':pts, 'r':r, 't':t}



    def calc_directproblem(self):
        msg = "set the direct problem before !"
        assert self.direct is not None, msg
        if self.emiss is None:
            msg = "emiss not set => only the solid angle will be computed !"
            warnings.warn(msg)
            out = _comp.
            lpolyCross, lpolyHor, gridCross, gridHor = out[:4]
            saCross, saHor, volIn, saIn = out[4:]
        else:
            out = 
            lpolyCross, lpolyHor, gridCross, gridHor = out[:4]
            saCross, saHor, volIn, saIn = out[4:8]
            contribCross, contribHor, powIn = out[8:]

        self._direct_sol = {'lpolyCross':lpolyCross, 'lpolyHor':lpolyHor,
                            'gridCross'}

        pts, dV, ind, dVr = get_sampleV(dV, DV=None, dVMode='abs', ind=None, Out='(X,Y,Z)'):







