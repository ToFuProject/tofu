
# Built-in
import os
import sys

# Common
import datetime as dtm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# tofu
import tofu as tf


np.set_printoptions(linewidth=200)


###################
# Emissivity
###################

def emiss(pts, t=None, vect=None):
    r, z = np.hypot(pts[0,:],pts[1,:]), pts[2,:]
    e = np.exp(-(r-2.4)**2/0.2**2 - z**2/0.2**2)
    if t is not None:
        e = np.cos(np.atleast_1d(t))[:,None] * e[None,:]
    return e

###################
# Defining defaults
###################

_LRES = [-1,-2,0]
_LLOS = [1,2,0]
_LT = [1,2,0]
_NREP = 3
_DRES = abs(_LRES[1] - _LRES[0])
_DLOS = abs(_LLOS[1] - _LLOS[0])
_DT = abs(_LT[1] - _LT[0])
_RES = np.logspace(_LRES[0], _LRES[1], _LRES[2]*_DRES + _DRES + 1 ,
                   base=10)
_NLOS = np.round(np.logspace(_LLOS[0], _LLOS[1], _LLOS[2]*_DLOS + _DLOS + 1,
                             base=10))
_NT = np.round(np.logspace(_LT[0], _LT[1], _LT[2]*_DT + _DT + 1, base=10))
_SHOT = 54178
_IDS = ['core_profiles', 'equilibrium', 'core_sources']
_QUANT = 'core_profiles.1dne'
_REF1D = 'core_profiles.1drhotn'
_REF2D = 'equilibrium.2drhotn'
_DALGO = {'ref-sum':{'newcalc':False},
          'calls-sum':    {'newcalc':True,  'minimize':'calls', 'method':'sum'},
          # 'calls-simps':  {'newcalc':True,  'minimize':'calls', 'method':'simps'},
          # 'calls-romb':   {'newcalc':True,  'minimize':'calls', 'method':'romb'},
          'hybrid-sum':   {'newcalc':True,  'minimize':'hybrid', 'method':'sum'},
          # 'hybrid-simps': {'newcalc':True,  'minimize':'hybrid', 'method':'simps'},
          # 'hybrid-romb':  {'newcalc':True,  'minimize':'hybrid', 'method':'romb'},
          'memory-sum':   {'newcalc':True,  'minimize':'memory', 'method':'sum'},
          # 'memory-simps': {'newcalc':True,  'minimize':'memory', 'method':'simps'},
          # 'memory-romb':  {'newcalc':True,  'minimize':'memory', 'method':'romb'},
         }
_DCAM = {'P':[3.4,0.,0.], 'F':0.1, 'D12':0.1,
         'angs':[1.05*np.pi, np.pi/4, np.pi/4]}

_PATH = os.path.abspath(os.path.dirname(__file__))


_FS = (14,10)
_DMARGIN = {'left':0.05, 'right':0.95,
            'bottom':0.05, 'top':0.95,
            'wspace':0.1, 'hspace':0.1}



###################
# Main function
###################


def benchmark(config=None, func=None, plasma=None, shot=None, ids=None,
              quant=None, ref1d=None, ref2d=None,
              res=None, nlos=None, nt=None, t=None,
              dalgo=None, nrep=None, txtfile=None,
              path=None, name=None, nameappend=None,
              plot=False, save=True):

    # --------------
    # Prepare inputs

    # config
    if config is None:
        config = 'B2'
    if type(config) is str:
        config = tf.geom.utils.create_config(config)

    # func vs plasma
    if func == True:
        func = emiss
    elif func is None:
        if plasma is None:
            if ids is None:
                ids = _IDS
            if shot is None:
                shot = _SHOT
            didd = tf.imas2tofu.MultiIDSLoader(shot=shot, ids=ids)
            plasma = didd.to_Plasma2D()

        # quant, ref1d, ref2d
        if quant is None:
            quant = _QUANT
            ref1d = _REF1D
            ref2d = _REF2D

    # res and los
    if res is None:
        res = _RES
    nres = len(res)
    if nlos is None:
        nlos = _NLOS
    nlos = np.array(nlos, dtype=int)
    nnlos = len(nlos)

    # dalgo
    if dalgo is None:
        dalgo = _DALGO
    lalgo = list(dalgo.keys())
    nalgo = len(dalgo)

    # nrep
    if nrep is None:
        nrep = _NREP

    if t is None:
        if nt is None:
            nt = _NT
        lt = [np.linspace(0,100,ntt) for ntt in nt]
    else:
        lt = [np.atleast_1d(t).ravel()]
    nnt = len(lt)

    if path is None:
        path = _PATH
    if name is None:
        lvar = [('nalgo',nalgo), ('nnlos',nnlos),
                ('nres',nres), ('nnt',nnt)]
        name = 'benchmark_LOScalcsignal_'
        name += '_'.join(['%s%s'%(nn,vv)
                          for nn,vv in lvar])
    if nameappend is not None:
        name += '_'+nameappend

    #---------------
    # Prepare output

    # data
    t_av = np.full((nalgo, nnlos, nres, nnt), np.nan)
    t_std = np.full((nalgo, nnlos, nres, nnt), np.nan)
    memerr = np.zeros((nalgo, nnlos, nres, nnt), dtype=bool)
    win = np.zeros((nnlos, nres, nnt), dtype=int)

    # printing file
    if txtfile is None:
        txtfile = sys.stdout
    elif type(txtfile) is str:
        txtfile = open(os.path.join(path,txtfile), 'w')
    elif txtfile is True:
        txtfile = open(os.path.join(path,name+'.txt'), 'w')

    #---------------
    # Prepare saving params
    if save:
        pfe = os.path.join(path,name+'.npz')
        lk = ['t_std', 'nnt', 'lt', 'nalgo', 'nres',
              'name', 'path', 'save', 'plot', 'nrep', 'dalgo', 't', 'nt',
              'res', 'ref2d', 'ref1d', 'quant', 'ids', 'shot',
              'win', 't_av', 'nnlos', 'nlos', 'ncase', 'lalgo']

    #------------
    # Start loop

    names = np.array([['%s  los = %s'%(lalgo[ii],int(nlos[jj]))
                       for jj in range(nnlos)]
                      for ii in range(nalgo)])
    lennames = np.max(np.char.str_len(names))
    msg = "\n###################################"*2
    msg += "\nBenchmark about to be run with:"
    msg += "\n-------------------------------\n\n"
    msg += "lalgo = %s\n"%str(lalgo)
    msg += "nlos = %s\n"%str(nlos)
    msg += "res  = %s\n"%str(res)
    msg += "nt   = %s\n"%str(nt)
    msg += "rep  = %s\n\n"%str(nrep)
    msg += "    algo:".ljust(lennames) + '  times:'
    print(msg, file=txtfile)

    err0 = None
    for ii in range(nalgo):
        print('', file=txtfile)
        for jj in range(nnlos):
            cam = tf.geom.utils.create_CamLOS1D(N12=nlos[jj],
                                                config=config,
                                                Name=str(names[ii,jj]), Exp='dummy',
                                                Diag='Dummy',
                                                **_DCAM)
            msg = "    %s"%(names[ii,jj].ljust(lennames))
            print(msg, file=txtfile)

            for ll in range(nres):
                msg = "\r        res %s/%s"%(ll+1, nres)
                for tt in range(nnt):
                    dt = np.zeros((nrep,))
                    for rr in range(nrep):
                        msgi = msg + "   nt %s/%s    rep %s/%s"%(tt+1,nnt,rr+1,nrep)
                        print(msg + msgi, end='', file=txtfile, flush=True)

                        try:
                            if func is None:
                                t0 = dtm.datetime.now()
                                _ = cam.calc_signal_from_Plasma2D(plasma,
                                                                  quant=quant,
                                                                  ref1d=ref1d,
                                                                  ref2d=ref2d,
                                                                  res=res[ll],
                                                                  resMode='abs',
                                                                  plot=False,
                                                                  t = lt[tt],
                                                                  **dalgo[lalgo[ii]])
                                dt[rr] = (dtm.datetime.now()-t0).total_seconds()
                            else:
                                t0 = dtm.datetime.now()
                                _ = cam.calc_signal(func, res=res[ll],
                                                    resMode='abs', plot=False,
                                                    t = lt[tt],
                                                    **dalgo[lalgo[ii]])
                                dt[rr] = (dtm.datetime.now()-t0).total_seconds()
                        except MemoryError as err:
                            dt[rr] = -1
                            memerr[ii,jj,ll,tt] = True
                        except Exception as err:
                            if err0 is None:
                                err0 = err

                    t_av[ii,jj,ll,tt] = np.mean(dt)
                    t_std[ii,jj,ll,tt] = np.std(dt)

                msgi = ': %s\n'%str(t_av[ii,jj,ll,:])
                print(msg + msgi, end='', file=txtfile, flush=True)
            if save:
                out = {kk:vv for kk,vv in locals().items() if kk in lk}
                np.savez(pfe, **out)
                print("        (saved)", file=txtfile)



    t_av[memerr] = np.nan
    win = np.nanargmin(t_av, axis=0)
    ncase = win.size

    # Print synthesis
    ln = np.max([len(aa) for aa in lalgo])
    msg = "\n  --------------------\n  --- Synthesis ---"
    msg += "\n\n  Speed score:\n    "
    msg += "\n    ".join(["%s : %s"%(lalgo[ii].ljust(ln),
                                     100.*np.sum(win==ii)/ncase) +' %'
                          for ii in range(nalgo)])

    winname = np.char.rjust(np.asarray(lalgo)[win], ln)
    lsblocks = ['nlos = %s'%str(nlos[jj]) + "\n        "
                 + "\n        ".join([('res %s/%s    '%(ll,nres)
                                       + str(winname[jj,ll,:]))
                                      for ll in range(nres)])
                for jj in range(nnlos)]
    msg += "\n" +  "\n    " + "\n    ".join(lsblocks)


    msg += "\n\n  Memory score:\n    "
    msg += "\n    ".join(["%s : %s"%(lalgo[ii].ljust(ln),
                                     100.*np.sum(t_av[ii,...]>=0)/ncase) + ' %'
                          for ii in range(nalgo)])
    print(msg, file=txtfile)


    if err0 is not None:
        raise err0

    #-------------
    # Plot / save


    if save:
        out = {kk:vv for kk,vv in locals().items() if kk in lk}
        np.savez(pfe, **out)
        print('Saved in:\n    %s'%pfe, file=txtfile)

    if plot:
        try:
            plot_benchmark(**out)
        except Exception:
            pass
    if txtfile != sys.stdout:
        txtfile.close()
    return out



###################
#   Plotting
###################


def plot_benchmark(fname=None, fs=None, dmargin=None, **kwdargs):

    if fname is not None:
        assert type(fname) is str
        out = dict(np.load(fname))
    else:
        out = kwdargs

    # Prepare inputs
    # --------------
    if fs is None:
        fs = _FS
    if dmargin is None:
        dmargin = _DMARGIN

    indok = out['t_av'] >= 0.
    vmin, vmax = np.nanmin(out['t_av'][indok]), np.nanmax(out['t_av'][indok])


    # Plotting
    # --------------
    fig = plt.figure(figsize=fs)
    # axarr = GridSpec(1,4, **dmargin)
    # ax0 = fig.add_subplot(axarr[0,0])
    # ax1 =
    # ax2 =
    # ax3 =
    ax0 = fig.add_axes([0.1,0.1,0.8,0.8])

    ax0.scatter(out['nlos'], out['res'], out['nt'],
                c=out['t_av'][0,...], s=8, marker='o')

    return
