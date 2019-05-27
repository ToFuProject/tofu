# External modules
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings as warn

# Nose-specific
from nose import with_setup # optional


# Importing package tofu.geom
import tofu as tf
from tofu import __version__
import tofu.defaults as tfd
import tofu.pathfile as tfpf
import tofu.utils as tfu
import tofu.geom as tfg

_here = os.path.abspath(os.path.dirname(__file__))
VerbHead = 'tofu.geom.tests03_core'
keyVers = 'Vers'
_Exp = 'WEST'

path = os.path.join(_here,'tests03_core_data')
lf = os.listdir(path)
lf = [f for f in lf if all([s in f for s in [_Exp,'.txt']])]
lCls = sorted(set([f.split('_')[1] for f in lf]))

dobj = {'Tor':{}, 'Lin':{}}
for tt in dobj.keys():
    for cc in lCls:
        lfc = [f for f in lf if f.split('_')[1]==cc and 'V0' in f]
        ln = []
        for f in lfc:
            if 'CoilCS' in f:
                ln.append(f.split('_')[2].split('.')[0])
            else:
                ln.append(f.split('_')[2].split('.')[0])
        lnu = sorted(set(ln))
        if not len(lnu)==len(ln):
            msg = "Non-unique name list for {0}:".format(cc)
            msg += "\n    ln = [{0}]".format(', '.join(ln))
            msg += "\n    lnu = [{0}]".format(', '.join(lnu))
            raise Exception(msg)
        dobj[tt][cc] = {}
        for ii in range(0,len(ln)):
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

            Poly = np.loadtxt(os.path.join(path,lfc[ii]))
            assert Poly.ndim==2
            assert Poly.size>=2*3
            kwd = dict(Name=ln[ii]+tt, Exp=_Exp, SavePath=_here,
                       Poly=Poly, Lim=Lim, Type=tt)
            dobj[tt][cc][ln[ii]] = eval('tfg.%s(**kwd)'%cc)


for typ in dobj.keys():
    # Todo : introduce possibility of choosing In coordinates !
    for c in dobj[typ].keys():
        if issubclass(eval('tfg.%s'%c), tfg._core.StructOut):
            continue
        for n in dobj[typ][c].keys():
            obj = dobj[typ][c][n]
            P1Mm = (obj.dgeom['P1Min'][0], obj.dgeom['P1Max'][0])
            P2Mm = (obj.dgeom['P2Min'][1], obj.dgeom['P2Max'][1])
            box = None#[[2.,3.], [0.,5.], [0.,np.pi/2.]]
            try:
                ii = 0
                out = obj.get_sampleV(0.1, resMode='abs', DV=box,
                                      Out='(X,Y,Z)')
                pts0, ind = out[0], out[2]
                ii = 1
                out = obj.get_sampleV(0.1, resMode='abs', ind=ind,
                                      Out='(X,Y,Z)')
                pts1 = out[0]
            except Exception as err:
                msg = str(err)
                msg += "\nFailed for {0}_{1}_{2}".format(typ,c,n)
                msg += "\n    ii={0}".format(ii)
                msg += "\n    Lim={0}".format(str(obj.Lim))
                msg += "\n    DS={0}".format(str(box))
                raise Exception(msg)

            if type(pts0) is list:
                assert all([np.allclose(pts0[ii],pts1[ii])
                            for ii in range(0,len(pts0))])
            else:
                assert np.allclose(pts0,pts1)

print("dooooone")
