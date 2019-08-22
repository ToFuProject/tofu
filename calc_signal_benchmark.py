
# Common
import datetime as dtm
import numpy as np
import matplotlib.pyplot as plt

# tofu
import tofu as tf

_LRES = [-2,-3,1]
_LLOS = [1,3,1]
_LT = [1,4,1]
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
          # 'memory-sum':   {'newcalc':True,  'minimize':'memory', 'method':'sum'},
          # 'memory-simps': {'newcalc':True,  'minimize':'memory', 'method':'simps'},
          # 'memory-romb':  {'newcalc':True,  'minimize':'memory', 'method':'romb'},
          'hybrid-sum':   {'newcalc':True,  'minimize':'hybrid', 'method':'sum'},
          # 'hybrid-simps': {'newcalc':True,  'minimize':'hybrid', 'method':'simps'},
          # 'hybrid-romb':  {'newcalc':True,  'minimize':'hybrid', 'method':'romb'},
         }
_NREP = 3
_DCAM = {'P':[3.4,0.,0.], 'F':0.1, 'D12':0.1,
         'angs':[1.05*np.pi, np.pi/4, np.pi/4]}


def benchmark(config=None, func=None, plasma=None, shot=None, ids=None,
              quant=None, ref1d=None, ref2d=None,
              res=None, nlos=None, t=None,
              dalgo=None, nrep=None):

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

    #---------------
    # Prepare output
    t_av = np.full((nalgo, nnlos, nres), np.nan)
    t_std = np.full((nalgo, nnlos, nres), np.nan)
    win = np.zeros((nnlos, nres), dtype=int)

    #------------
    # Start loop

    names = [[len('%s-%s'%(lalgo[ii],int(nlos[jj]))) for jj in range(nnlos)]
             for ii in range(nalgo)]
    lennames = np.max(names)
    msg = "\n-------------------------------"
    msg += "Benchmark about to be run with:\n"
    msg += "lalgo = %s\n"%str(lalgo)
    msg += "nlos = %s\n"%str(nlos)
    msg += "res  = %s\n"%str(res)
    msg += "rep  = %s\n\n"%str(nrep)
    msg += "    algo:".ljust(lennames) + '  times:'
    print(msg)




    for ii in range(nalgo):
        print('')
        for jj in range(nnlos):
            name = '%s-%s'%(lalgo[ii],int(nlos[jj]))
            cam = tf.geom.utils.create_CamLOS1D(N12=nlos[jj],
                                                config=config,
                                                Name=name, Exp='dummy',
                                                Diag='Dummy',
                                                **_DCAM)
            msg = "\r    %s"%(name.ljust(lennames))

            for ll in range(nres):
                dt = np.zeros((nrep,))
                for rr in range(nrep):
                    msgi = msg + "   res %s/%s  rep %s/%s"%(ll+1,nres,rr+1,nrep)
                    print(msgi, end='', flush=True)

                    if func is None:
                        t0 = dtm.datetime.now()
                        out = cam.calc_signal_from_Plasma2D(plasma, quant=quant,
                                                            ref1d=ref1d,
                                                            ref2d=ref2d,
                                                            res=res[ll],
                                                            resMode='abs',
                                                            plot=False,
                                                            t = t,
                                                            **dalgo[lalgo[ii]])
                        dt[rr] = (dtm.datetime.now()-t0).total_seconds()
                    else:
                        t0 = dtm.datetime.now()
                        out = cam.calc_signal(func, res=res[ll], resMode='abs',
                                              plot=False, t = t, **dalgo[lalgo[ii]])
                        dt[rr] = (dtm.datetime.now()-t0).total_seconds()

                t_av[ii,jj,ll] = np.mean(dt)
                t_std[ii,jj,ll] = np.std(dt)

            msgi = msg + ': %s\n'%str(t_av[ii,jj,:])
            print(msgi, end='', flush=True)
    win = np.argmin(t_av, axis=0)

    #-------------
    # Plot results



    return locals()
