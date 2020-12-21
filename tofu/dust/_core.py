



import numpy as np
import matplotlib.pyplot as plt

# ToFu-specific
import tofu.geom as tfg
try:
    import tofu.dust._comp as _comp
    import tofu.dust._plot as _plot
except Exception:
    from . import _comp as _comp
    from . import _plot as _plot






class Dust(object):
    """ A generic Dust class to handle dust trajectories, radiative power
    balance...

    """

    def __init__(self, Id=None, dmat=None, traj=None, VType='Tor',
                 Ves=None, LStruct=None, res=None):
        # Initialize fields
        self._Id = None
        self._dmat = None
        self._traj = None
        self._geom = None
        self._res = None
        self._emiss = None
        self._direct = None
        if Id is not None:
            self._set_Id(Id)
        if dmat is not None:
            self.set_material_properties(dmat)
        if traj is not None:
            self.set_traj(traj)
        if Ves is not None:
            self.set_geom(Ves=Ves, LStruct=LStruct, res=res)

    @property
    def dmat(self):
        return self._dmat
    @property
    def traj(self):
        return self._traj
    @property
    def Ves(self):
        if self._geom is None:
            out = None
        else:
            out = self._geom['Ves']
        return out
    @property
    def LStruct(self):
        if self._geom is None:
            out = None
        else:
            out = self._geom['LStruct']
        return out
    @property
    def res(self):
        return self._res
    @property
    def emiss(self):
        return self._emiss
    @property
    def direct(self):
        return self._direct

    def set_material_properties(self, dmat=None):
        assert dmat is None or type(dmat) is dict
        self._dmat = dmat

    def set_traj(self, traj=None):
        C0 = traj is None
        C1 = type(traj) is dict
        C2 = hasattr(traj,'__iter__')
        assert C0 or C1 or C2
        if C1:
            assert 'pts' in traj.keys()
            traj['pts'] = _check_trajpts(traj['pts'])
            if 't' in traj.keys():
                traj['t'] = _check_trajt(traj['t'],traj['pts'].shape[0])
        if C2:
            pts = _check_trajpts(traj)
            traj = {'pts':pts}
        traj['npts'] = traj['pts'].shape[0]
        self._traj = traj

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

        LStruct = [LStruct] if isinstance(LStruct, tfg.Struct) else LStruct
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


    """
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
            #out =
            lpolyCross, lpolyHor, gridCross, gridHor = out[:4]
            saCross, saHor, volIn, saIn = out[4:8]
            contribCross, contribHor, powIn = out[8:]

        self._direct_sol = {'lpolyCross':lpolyCross, 'lpolyHor':lpolyHor,
                            'gridCross'}
    """

    def plot(self):
        dax, KH = _plot.plot(self)
        return dax, KH



def _check_trajpts(pts):
    pts = np.asarray(pts).astype(float)
    assert pts.ndim in [1,2]
    if pts.ndim==1:
        assert pts.size==3
        pts = pts.reshape((1,3))
    assert 3 in pts.shape
    if not pts.shape[1]==3:
        pts = pts.T
    return pts

def _check_trajt(t, npts):
    t = np.asarray(t).astype(float)
    assert t.ndim==1
    assert t.size==npts
    return t
