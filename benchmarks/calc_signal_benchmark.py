
# Built-in
import os

# Common
import datetime as dtm
import numpy as np
import matplotlib.pyplot as plt

# tofu
import tofu as tf


###################
# Defining defaults
###################

_LRES = [-1,-2,0]
_LLOS = [1,3,0]
_LT = [1,2,0]
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
_NREP = 1
_DCAM = {'P':[3.4,0.,0.], 'F':0.1, 'D12':0.1,
         'angs':[1.05*np.pi, np.pi/4, np.pi/4]}

_PATH = os.path.abspath(os.path.dirname(__file__))


###################
# Main function
###################


def benchmark(config=None, func=None, plasma=None, shot=None, ids=None,
              quant=None, ref1d=None, ref2d=None,
              res=None, nlos=None, nt=None, t=None,
              dalgo=None, nrep=None,
              plot=True, save=True, path=None, name=None):

    # --------------
    # Prepare inputs

    # config
    if config is None:
        config = 'B2'
    if type(config) is str:
        config = tf.geom.utils.create_config(config)

    # plasma
    if func is None:
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

    #---------------
    # Prepare output
    t_av = np.full((nalgo, nnlos, nres, nnt), np.nan)
    t_std = np.full((nalgo, nnlos, nres, nnt), np.nan)
    win = np.zeros((nnlos, nres, nnt), dtype=int)

    #------------
    # Start loop

    names = [[len('%s-%s'%(lalgo[ii],int(nlos[jj]))) for jj in range(nnlos)]
             for ii in range(nalgo)]
    lennames = np.max(names)
    msg = "\n-------------------------------\n"
    msg += "Benchmark about to be run with:\n"
    msg += "lalgo = %s\n"%str(lalgo)
    msg += "nlos = %s\n"%str(nlos)
    msg += "res  = %s\n"%str(res)
    msg += "nt   = %s\n"%str(nt)
    msg += "rep  = %s\n\n"%str(nrep)
    msg += "    algo:".ljust(lennames) + '  times:'
    print(msg)

    err0 = None

    for ii in range(nalgo):
        print('')
        for jj in range(nnlos):
            namei = '%s-%s'%(lalgo[ii],int(nlos[jj]))
            cam = tf.geom.utils.create_CamLOS1D(N12=nlos[jj],
                                                config=config,
                                                Name=namei, Exp='dummy',
                                                Diag='Dummy',
                                                **_DCAM)
            msg = "    %s"%(namei.ljust(lennames))
            print(msg)

            for ll in range(nres):
                msg = "\r        res %s/%s"%(ll+1, nres)
                for tt in range(nnt):
                    dt = np.zeros((nrep,))
                    for rr in range(nrep):
                        msgi = msg + "   nt %s/%s    rep %s/%s"%(tt+1,nnt,rr+1,nrep)
                        print(msg + msgi, end='', flush=True)

                        try:
                            if func is None:
                                t0 = dtm.datetime.now()
                                out = cam.calc_signal_from_Plasma2D(plasma, quant=quant,
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
                                out = cam.calc_signal(func, res=res[ll], resMode='abs',
                                                      plot=False, t = lt[tt], **dalgo[lalgo[ii]])
                                dt[rr] = (dtm.datetime.now()-t0).total_seconds()
                        except MemoryError as err:
                            dt[rr] = -1
                        except Exception as err:
                            if err0 is None:
                                err0 = err

                    t_av[ii,jj,ll,tt] = np.mean(dt)
                    t_std[ii,jj,ll,tt] = np.std(dt)

                msgi = ': %s\n'%str(t_av[ii,jj,ll,:])
                print(msg + msgi, end='', flush=True)
    win = np.argmin(t_av, axis=0)
    ncase = win.size

    # Print synthesis
    msg = "\n  --- Synthesis ---"
    msg += "\n  Speed: (algo ... is fastest in...):\n    "
    msg += "\n    ".join(["%s : %s"%(lalgo[ii],100.*np.sum(win==ii)/ncase) +' %'
                          for ii in range(nalgo)])
    msg += "\n  Memory: (algo ... has no memory error in...):\n    "
    msg += "\n    ".join(["%s : %s"%(lalgo[ii],
                                     100.*np.sum(t_av[ii,...]>=0)/ncase) + ' %'
                          for ii in range(nalgo)])
    print(msg)

    if err0 is not None:
        msg = str(err0)
        lind = np.where(np.isnan(t_av))
        lind = [ii[0] for ii in lind]
        msg += "\n\n The above error occured for:\n"
        msg += "algo %s nlos %s/%s, res %s/%s, nt %s/%s"%(lalgo[lind[0]],
                                                          lind[1], nnlos,
                                                          lind[2], nres,
                                                          lind[3], nnt)
        warnings.warn(msg)

    #-------------
    # Plot / save

    lk = ['t_std', 'nnt', 'lt', 'nalgo',
          'nres', 'name', 'path', 'save', 'plot', 'nrep', 'dalgo', 't', 'nt',
          'res', 'ref2d', 'ref1d', 'quant', 'ids', 'shot',
          'win', 't_av', 'nnlos', 'nlos', 'ncase', 'lalgo']
    out = {kk:vv for kk,vv in locals().items() if kk in lk}

    if plot:
        plot_benchmark(**out)

    if save:
        if name is None:
            lvar = [('nalgo',nalgo), ('nnlos',nnlos),
                    ('nres',nres), ('nnt',nnt)]
            name = 'benchmark_LOScalcsignal_'
            name += '_'.join(['%s%s'%(nn,vv)
                              for nn,vv in lvar])
        if path is None:
            path = _PATH
        pfe = os.path.join(path,name+'.npz')
        np.savez(pfe, **out)
        print('Saved in:\n    %s'%pfe)
    return out



###################
#   Plotting
###################


def plot_benchmark(fname=None, **kwdargs):

    if fname is not None:
        assert type(fname) is str
        out = np.load(fname)
    else:
        out = kwdargs

    return
