#!/usr/bin/env python

# Built-in
import os
import sys
import argparse

# Common
import datetime as dtm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import socket
import getpass

# tofu
# test if in a tofu git repo
_HERE = os.path.abspath(os.path.dirname(__file__))
istofugit = False
heresplit = _HERE.split(os.path.sep)
if 'benchmarks' in heresplit:
    ind = heresplit[::-1].index('benchmarks')
    pp = os.path.sep + os.path.join(*heresplit[:-ind-1])
    lf = os.listdir(pp)
    if '.git' in lf and 'tofu' in lf:
        istofugit = True

if istofugit:
    # Make sure we load the corresponding tofu
    sys.path.insert(1,pp)
    import tofu as tf
    _ = sys.path.pop(1)
else:
    import tofu as tf
tforigin = tf.__file__
tfversion = tf.__version__

np.set_printoptions(linewidth=200)


###################
# Defining defaults
###################

_LRES = [-2, -1, 0]
_LLOS = [2, 3, 0]
_LT = [2, 2, 0]
_NREP = 1

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

_PATH = _HERE
_FUNC = True
_TXTFILE = None
_SAVE = True
_PLOT = False

_FS = (14,10)
_DMARGIN = {'left':0.05, 'right':0.95,
            'bottom':0.05, 'top':0.95,
            'wspace':0.1, 'hspace':0.1}



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
# Main function
###################


def benchmark(config=None, func=_FUNC, plasma=None, shot=None, ids=None,
              quant=None, ref1d=None, ref2d=None,
              res=None, nlos=None, nt=None, t=None,
              dalgo=None, nrep=None, txtfile=None,
              path=None, name=None, nameappend=None,
              plot=_PLOT, save=_SAVE):

    # --------------
    # Prepare inputs

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
                ('nres',nres), ('nnt',nnt),
                ('Host',socket.gethostname()), ('USR',getpass.getuser())]
        name = 'benchmark_LOScalcsignal_'
        name += '_'.join(['{}{}'.format(nn, vv)
                          for nn,vv in lvar])
    if nameappend is not None:
        name += '_'+nameappend

    # printing file
    stdout = False
    msg_loc = "\ntofu {} loaded from:\n    {}\n".format(tfversion, tforigin)
    if txtfile is None:
        txtfile = sys.stdout
        stdout = True
        print(msg_loc)
    elif type(txtfile) is str:
        txtfile = os.path.join(path, txtfile)
        with open(txtfile, 'w') as f:
            f.write(msg_loc)
    elif txtfile is True:
        txtfile = os.path.join(path, name+'.txt')
        with open(txtfile, 'w') as f:
            f.write(msg_loc)

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

    #---------------
    # Prepare output

    # data
    t_av = np.full((nalgo, nnlos, nres, nnt), np.nan)
    t_std = np.full((nalgo, nnlos, nres, nnt), np.nan)
    memerr = np.zeros((nalgo, nnlos, nres, nnt), dtype=bool)
    win = np.zeros((nnlos, nres, nnt), dtype=int)

    #---------------
    # Prepare saving params
    if save:
        pfe = os.path.join(path,name+'.npz')
        lk = ['tforigin', 'tfversion', 't_std', 'nnt', 'lt', 'nalgo', 'nres',
              'name', 'path', 'save', 'plot', 'nrep', 'dalgo', 't', 'nt',
              'res', 'ref2d', 'ref1d', 'quant', 'ids', 'shot',
              'win', 't_av', 'nnlos', 'nlos', 'ncase', 'lalgo']

    #------------
    # Start loop

    names = np.array([['{}  los = {}'.format(lalgo[ii], int(nlos[jj]))
                       for jj in range(nnlos)]
                      for ii in range(nalgo)])
    lennames = np.max(np.char.str_len(names))
    msg = "\n###################################"*2
    msg += "\nBenchmark about to be run with:"
    msg += "\n-------------------------------\n\n"
    msg += "lalgo = {}\n".format(lalgo)
    msg += "nlos = {}\n".format(nlos)
    msg += "res  = {}\n".format(res)
    msg += "nt   = {}\n".format(nt)
    msg += "rep  = {}\n\n".format(nrep)
    msg += "    algo:"
    msg = msg.ljust(lennames)
    msg += '  times:'
    if stdout:
        print(msg)
    else:
        with open(txtfile, 'w') as f:
            f.write(msg)

    err0 = None
    for ii in range(nalgo):
        if stdout:
            print('')
            sys.stdout.flush()
        else:
            with open(txtfile, 'w') as f:
                f.write(msg)
        for jj in range(nnlos):
            cam = tf.geom.utils.create_CamLOS1D(N12=nlos[jj],
                                                config=config,
                                                Name=str(names[ii, jj]),
                                                Exp='dummy',
                                                Diag='Dummy',
                                                **_DCAM)
            msg = "    {}".format(names[ii, jj].ljust(lennames))
            if stdout:
                print(msg)
                sys.stdout.flush()
            else:
                with open(txtfile, 'w') as f:
                    f.write(msg)
            for ll in range(nres):
                msg = "        res {}/{}".format(ll + 1, nres)
                for tt in range(nnt):
                    dt = np.zeros((nrep,))
                    for rr in range(nrep):
                        if stdout:
                            msgi = "\r" + msg \
                              + "   nt {}/{}    rep {}/{}".format(tt + 1,
                                                                  nnt,
                                                                  rr + 1,
                                                                  nrep)
                            print(msgi, end='')
                            sys.stdout.flush()
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

                msgi = msg + ': {}'.format(t_av[ii, jj, ll, :])
                if stdout:
                    msgi = '\r'+msgi
                    print(msgi)
                    sys.stdout.flush()
                else:
                    with open(txtfile, 'w') as f:
                        f.write(msgi)

            if save:
                out = {kk:vv for kk,vv in locals().items() if kk in lk}
                np.savez(pfe, **out)
                if stdout:
                    print("        (saved)")
                else:
                    with open(txtfile, 'w') as f:
                        f.write("        (saved)")



    t_av[memerr] = np.nan
    win = np.nanargmin(t_av, axis=0)
    ncase = win.size

    # Print synthesis
    ln = np.max([len(aa) for aa in lalgo])
    msg = "\n  --------------------\n  --- Synthesis ---"
    msg += "\n\n  Speed score:\n    "
    msg += "\n    ".join(["{} : {}".format(lalgo[ii].ljust(ln),
                                     100.*np.sum(win==ii)/ncase) +' %'
                          for ii in range(nalgo)])

    winname = np.char.rjust(np.asarray(lalgo)[win], ln)
    lsblocks = ['nlos = {}'.format(nlos[jj]) + "\n        "
                + "\n        ".join([('res {}/{}    '.format(ll, nres)
                                      + str(winname[jj, ll, :]))
                                     for ll in range(nres)])
                for jj in range(nnlos)]
    msg += "\n" +  "\n    " + "\n    ".join(lsblocks)


    msg += "\n\n  Memory score:\n    "
    msg += "\n    ".join(["{} : {}".format(lalgo[ii].ljust(ln),
                                     100.*np.sum(t_av[ii,...]>=0)/ncase) + ' %'
                          for ii in range(nalgo)])
    if stdout:
        print(msg)
    else:
        with open(txtfile, 'w') as f:
            f.write(msg)
    if err0 is not None:
        raise err0

    #-------------
    # Plot / save

    if save:
        out = {kk:vv for kk,vv in locals().items() if kk in lk}
        np.savez(pfe, **out)
        if stdout:
            print('Saved in:\n    {}'.format(pfe))
        else:
            with open(txtfile, 'w') as f:
                f.write('Saved in:\n    {}'.format(pfe))

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



###################
#   Bash interface
###################

if __name__ == '__main__':

    # Parse input arguments
    msg = \
    """ Launch benchmark for _GG.LOS_calc_signal

    This is a bash wrapper around the function benchmark()
    """
    parser = argparse.ArgumentParser(description=msg)

    parser.add_argument('-f', '--func', type=bool,
                        help='emissivity function', required=False, default=_FUNC)
    parser.add_argument('-tf', '--txtfile', type=bool,
                        help='write to txt file ?', required=False,
                        default=_TXTFILE)
    parser.add_argument('-na', '--nameappend', type=str,
                        help='str to be appended to the name', required=False,
                        default=None)
    parser.add_argument('-s', '--save', type=bool,
                        help='save results ?', required=False,
                        default=_SAVE)
    parser.add_argument('-p', '--plot', type=bool,
                        help='plot results ?', required=False,
                        default=_PLOT)
    parser.add_argument('-pa', '--path', type=str,
                        help='path where to save results', required=False,
                        default=_PATH)

    args = parser.parse_args()

    # Call wrapper function
    benchmark(**dict(args._get_kwargs()))
