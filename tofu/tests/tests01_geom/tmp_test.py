# External modules
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings as warn

# Nose-specific
#from nose import with_setup # optional


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

dconf = {}
for typ in dobj.keys():
    lS = []
    for c in dobj[typ].keys():
        lS += list(dobj[typ][c].values())
    Lim = None if typ=='Tor' else [0.,10.]
    dconf[typ] = tfg.Config(Name='Test%s'%typ, Exp=_Exp,
                            lStruct=lS, Lim=Lim,
                            Type=typ, SavePath=_here)

dCams = {}
foc = 0.08
DX = 0.05
for typ in dconf.keys():
    dCams[typ] = {}
    if typ=='Tor':
        phi = np.pi/4.
        eR = np.r_[np.cos(phi),np.sin(phi),0.]
        ephi = np.r_[np.sin(phi),-np.cos(phi),0.]
        R = 3.5
        ph = np.r_[R*np.cos(phi),R*np.sin(phi),0.2]
    else:
        ph = np.r_[3.,4.,0.]
    ez = np.r_[0.,0.,1.]
    for c in ['CamLOS2D','CamLOS1D']:
        if '1D' in c:
            nP = 100
            X = np.linspace(-DX,DX,nP)
            if typ=='Tor':
                D = (ph[:,np.newaxis] + foc*eR[:,np.newaxis]
                     + X[np.newaxis,:]*ephi[:,np.newaxis])
            else:
                D = np.array([3.+X,
                              np.full((nP,),4.+foc),
                              np.full((nP,),0.02)])
        else:
            if typ=='Tor':
                nP = 600
                X = np.linspace(-DX,DX,nP)
                D = (ph[:,np.newaxis] + foc*eR[:,np.newaxis]
                     + np.repeat(X[::-1],nP)[np.newaxis,:]*ephi[:,np.newaxis]
                     + np.tile(X,nP)[np.newaxis,:]*ez[:,np.newaxis])
            else:
                nP = 100
                X = np.linspace(-DX,DX,nP)
                D = np.array([np.repeat(3.+X[::-1],nP),
                              np.full((nP*nP,),4.+foc),
                              np.tile(0.01+X,nP)])
        cls = eval("tfg.%s"%c)
        dCams[typ][c] = cls(Name='V1000', config=dconf[typ],
                            dgeom={'pinhole':ph, 'D':D}, method="optimized",
                            Exp=_Exp, Diag='Test', SavePath=_here)

dobj=dCams
verb=False
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



#@profile
def run_and_rest():
    for typ in dobj.keys():
        for c in dobj[typ].keys():
            obj = dobj[typ][c]
            Ds, us = obj.D[:], obj.u[:]
            out = obj.get_sample(0.02, resMode='abs',
                                 method='sum',DL=None)
            k, res, lind = out
            #nbrepet = np.r_[lind[0], np.diff(lind), k.size - lind[-1]]
            nbrepet = lind
            Pts = np.repeat(Ds, nbrepet, axis=1) + k* np.repeat(us, nbrepet, axis=1)
            lind = np.cumsum(lind)[:-1]
            k = np.split(k, lind)
            assert len(res)==len(k)==obj.nRays
            for ii in range(0,len(k)):
                if not (np.isnan(obj.kIn[ii]) or np.isnan(obj.kOut[ii])):
                    ind = ~np.isnan(k[ii])
                    assert np.all((k[ii][ind]>=obj.kIn[ii]-res[ii]))
                    assert np.all(k[ii][ind]<=obj.kOut[ii]+res[ii])
            assert np.all(res[~np.isnan(res)]<0.02)
            out = obj.get_sample(0.1, resMode='rel',
                                 method='simps',DL=[0,1])
            k, res, lind = out
            k = np.split(k, lind[:-1])
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
            k = np.split(k, lind[:-1])
            assert len(res)==len(k)==obj.nRays
            for ii in range(0,len(k)):
                if not (np.isnan(obj.kIn[ii]) or np.isnan(obj.kOut[ii])):
                    ind = ~np.isnan(k[ii])
                    assert np.all((k[ii][ind]>=obj.kIn[ii]-res[ii])
                              & (k[ii][ind]<=obj.kOut[ii]+res[ii]))


import timeit
start = timeit.default_timer()
run_and_rest()
stop = timeit.default_timer()
print('Time: ', stop - start)
print("dooooone")
