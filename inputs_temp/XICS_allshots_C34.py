
import os
import shutil
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines

_HERE = os.path.dirname(__file__)
cwd = os.getcwd()
os.chdir(os.path.abspath(os.path.join(_HERE, os.pardir)))
import tofu as tf
os.chdir(cwd)



# #############################################################################
#                   Prepare PATH
# #############################################################################


_HERE = os.path.abspath(os.path.dirname(__file__))
_PATHC3 = os.path.abspath(os.path.join(
    _HERE,
    'XICS_allshots_C3_sh53700-54178.npz'))
_PATHC4 = os.path.abspath(os.path.join(
    _HERE,
    'XICS_allshots_C4_sh54179-55987.npz'))
_PATH = _HERE


# #############################################################################
#                   Hand-picked database
# #############################################################################


_SHOTS = np.r_[
    54041,
    np.arange(54043, 54055),
    54058, 54059,
    np.arange(54061, 54068),
    np.arange(54069, 54076),
    np.arange(54077, 54080),
    54081,
    54083, 54084,
    np.arange(54088, 54109),
    54123,
    np.arange(54126, 54146),
    np.arange(54150, 54156),
    np.arange(54158, 54169),
    np.arange(54170,54176),
    54177, 54178,
]

_NSHOT = _SHOTS.size
_CAMP = np.full((_NSHOT,), 3)
_CAMP[_SHOTS>54178] = 4

_TLIM = np.tile([-np.inf, np.inf], (_NSHOT, 1))

_CRYST = np.full((_NSHOT,), 'ArXVII', dtype='<U7')
iArXVIII = (_SHOTS>=54062) & (_SHOTS<=54107)
iFe = (_SHOTS>=54123) & (_SHOTS<=54178)
_CRYST[iArXVIII] = 'ArXVIII'
_CRYST[iFe] = 'FeXXV'
_ANG = np.full((_NSHOT,), np.nan)

_DSHOTS = {
    'ArXVII': {
        54041: {'ang': 1.1498, 'tlim': [32,36]},
        54043: {'ang': 1.1498, 'tlim': [35,39]},
        54044: {'ang': 1.1498, 'tlim': [33,47]},
        54045: {'ang': 1.28075, 'tlim': [32,46]},
        54046: {'ang': 1.3124, 'tlim': [32,46]},
        54047: {'ang': 1.3995, 'tlim': [32,46]},
        54048: {'ang': 1.51995, 'tlim': [32,46]},
        54049: {'ang': 1.51995, 'tlim': [32,34]},
        54050: {'ang': 1.51995, 'tlim': [32,46]},
        54051: {'ang': 1.51995, 'tlim': [32,40]},
        54052: {'ang': 1.51995, 'tlim': [32,37]},
        54053: {'ang': 1.51995, 'tlim': [32,34]},
        54054: {'ang': 1.51995, 'tlim': [32,37]},
        54061: {'ang': 1.6240, 'tlim': [32,43]},
    },

    'ArXVIII':{
        54062: {'ang': -101.0, 'tlim': [32,37]},
        54063: {'ang': -101.0, 'tlim': [32,43]},
        54064: {'ang': -101.0, 'tlim': [32,43]},
        54065: {'ang': -101.099, 'tlim': [32,44]},
        54066: {'ang': -101.099, 'tlim': [32,41]},
        54067: {'ang': -101.099, 'tlim': [32,43]},
        54069: {'ang': -101.099, 'tlim': [32,40]},
        54070: {'ang': -101.099, 'tlim': [32,38]},
        54071: {'ang': -101.099, 'tlim': [32,40]},
        54072: {'ang': -101.099, 'tlim': [32,37]},
        54073: {'ang': -101.2218, 'tlim': [32,38]},
        54074: {'ang': -101.2218, 'tlim': [32,37]},
        54075: {'ang': -101.2218, 'tlim': [32,37]},
        54077: {'ang': -101.3507, 'tlim': [32,34]},
        54088: {'ang': -101.3507, 'tlim': [32,38]},
        54089: {'ang': -101.3507, 'tlim': [32,45]},
        54090: {'ang': -101.4831, 'tlim': [32,40]},
        54091: {'ang': -101.5800, 'tlim': [32,40]},
        54092: {'ang': -101.5800, 'tlim': [32,40]},
        54093: {'ang': -100.924, 'tlim': [32,37]},
        54094: {'ang': -100.924, 'tlim': [32,40]},
        54095: {'ang': -100.799, 'tlim': [32,48]},
        54096: {'ang': -100.799, 'tlim': [32,39]},
        54097: {'ang': -100.799, 'tlim': [32,37]},
        54098: {'ang': -100.706, 'tlim': [32,39]},
        54099: {'ang': -100.706, 'tlim': [32,39]},
        54100: {'ang': -100.580, 'tlim': [32,44]},
        54101: {'ang': -100.483, 'tlim': [32,40]},
        54102: {'ang': -100.386, 'tlim': [32,45]},
        54103: {'ang': -100.386, 'tlim': [32,38]},
        54104: {'ang': -100.2644, 'tlim': [32,38]},
        54105: {'ang': -100.132, 'tlim': [32,40]},
        54107: {'ang': -100.038, 'tlim': [32,38]},
    },

    'FeXXV':{
        54123: {'ang': -181.547, 'tlim': [32,59]},
        54126: {'ang': -181.547, 'tlim': [32,38]},
        54127: {'ang': -181.547, 'tlim': [32,49]},
        54128: {'ang': -181.547, 'tlim': [32,61]},
        54129: {'ang': -181.547, 'tlim': [32,46]},
        54130: {'ang': -181.547, 'tlim': [32,59]},
        54131: {'ang': -181.647, 'tlim': [32,64]},
        54133: {'ang': -181.746, 'tlim': [32,67]},
        54134: {'ang': -181.846, 'tlim': [32,63]},
        54135: {'ang': -181.946, 'tlim': [32,60]},
        54136: {'ang': -181.428, 'tlim': [32,63]},
        54137: {'ang': -181.3222, 'tlim': [32,44]},
        54138: {'ang': -181.1954, 'tlim': [32,42]},
        54139: {'ang': -181.1954, 'tlim': [32,65]},
        54141: {'ang': -181.1954, 'tlim': [32,59]},
        54142: {'ang': -181.1954, 'tlim': [32,54]},
        54143: {'ang': -181.1954, 'tlim': [32,66]},
        54144: {'ang': -181.1954, 'tlim': [32,65]},
        54145: {'ang': -181.1954, 'tlim': [32,40]},
        54150: {'ang': -181.0942, 'tlim': [32,57]},
        54151: {'ang': -181.0942, 'tlim': [32,40]},
        54152: {'ang': -180.9625, 'tlim': [32,61]},
        54153: {'ang': -180.9625, 'tlim': [32,49]},
        54154: {'ang': -180.8651, 'tlim': [32,49]},
        54155: {'ang': -180.8651, 'tlim': [32,47]},
        54158: {'ang': -180.8651, 'tlim': [32,67]},
        54159: {'ang': -180.7667, 'tlim': [32,63]},
        54160: {'ang': -180.7667, 'tlim': [32,66]},
        54161: {'ang': -180.6687, 'tlim': [32,40]},
        54162: {'ang': -180.6687, 'tlim': [32,37]},
        54163: {'ang': -180.6687, 'tlim': [32,66]},
        54164: {'ang': -180.5434, 'tlim': [32,65]},
        54165: {'ang': -180.5803, 'tlim': [32,39]},
        54166: {'ang': -180.5803, 'tlim': [32,65]},
        54167: {'ang': -181.6169, 'tlim': [32,37]}
    }
}


for cryst, v0 in _DSHOTS.items():
    for shot, v1 in v0.items():
        ishot = _SHOTS == shot
        if not np.any(ishot):
            msg = "shot in dict missing in array: {}".format(shot)
            warnings.warn(msg)
            continue
        if ishot.sum() > 1:
            msg = "{} shots in array for shot in dict: {}".format(ishot.sum(),
                                                                  shot)
            raise Exception(msg)
        if _CRYST[ishot][0] != cryst:
            msg = ("Inconsistent crystal!\n"
                   + "\t- shot: {}\n".format(shot)
                   + "\t- cryst:      {}\n".format(cryst)
                   + "\t- _CRYST[{}]: {}".format(ishot.nonzero()[0][0],
                                                 _CRYST[ishot][0]))
            raise Exception(msg)
        _ANG[ishot] = v1['ang']
        _TLIM[ishot] = v1['tlim']


_DCRYST = {
    'ArXVII': os.path.abspath(os.path.join(
        _HERE,
        'TFG_CrystalBragg_ExpWEST_DgXICS_ArXVII_sh00000_Vers1.4.1-174-g453d6a3.npz')),
    'ArXVIII': os.path.abspath(os.path.join(
        _HERE,
        'TFG_CrystalBragg_ExpWEST_DgXICS_ArXVIII_sh00000_Vers.npz')),
    'FeXXV': os.path.abspath(os.path.join(
        _HERE,
        'TFG_CrystalBragg_ExpWEST_DgXICS_FeXXV_sh00000_Vers1.4.2-a5-117-g6496abee.npz')),
}

_DDET = {'ArXVII':
         dict(ddist=0.04, di=-0.004, dj=0.,
              dtheta=0., dpsi=0., tilt=0.011, tangent_to_rowland=True)}


# #############################################################################
#                   Function to unify databases
# #############################################################################


_NT = 10
_SPECT1D = [(0., 0.02), (0.8, 0.02)]
_MASKPATH = os.path.abspath(os.path.join(
    _HERE,
    'XICS_mask.npz'
))
_DETPATH = os.path.abspath(os.path.join(
    _HERE,
    'det37.npz'
))
_DLINES = None
_XJ = np.r_[-0.08, -0.05, 0., 0.05, 0.1]
_DXJ = 0.002

def main(shots=_SHOTS,
         path=None,
         nt=None,
         dcryst=None,
         lfiles=None,
         maskpath=None,
         xj=None,
         dxj=None):

    # ---------
    # Check input
    if path is None:
        path = _PATH
    if dcryst is None:
        dcryst = _DCRYST
    if lfiles is None:
        lfiles = [_PATHC3, _PATHC4]
    if isinstance(lfiles, str):
        lfiles = [lfiles]
    if nt is None:
        nt = _NT
    if xj is None:
        xj = _XJ
    if dxj is None:
        dxj = _DXJ
    if maskpath is None:
        maskpath = _MASKPATH
    if maskpath is not False:
        mask = ~np.any(np.load(maskpath)['ind'], axis=0)

    # ---------
    # Prepare
    ni, nj = 487, 1467
    xiref = (np.arange(0, ni)-(ni-1)/2.)*172e-6
    xjref = (np.arange(0, nj)-(nj-1)/2.)*172e-6
    indxj = [(np.abs(xjref-xjj) <= dxj).nonzero()[0] for xjj in xj]

    # ---------
    # Loop on cryst
    for cc in _DCRYST.keys():

        ind = (_CRYST == cc).nonzero()[0]
        ns = ind.size
        if ns == 0:
            continue
        ang = _ANG[ind]
        angu, ianginv = np.unique(ang,
                                  return_index=False,
                                  return_inverse=True)
        pfe = os.path.join(path, 'XICS_data_{}.npz'.format(cc))

        # prepare output
        shotc = shots[ind]
        tlim = _TLIM[ind, :]
        tc = np.full((ns, nt), np.nan)
        thr = np.full((ns,), np.nan)
        texp = np.full((ns,), np.nan)
        tdelay = np.full((ns,), np.nan)
        tc = np.full((ns, nt), np.nan)
        spect = np.full((ns, nt, xj.size, ni), np.nan)
        success = ['OK' for ii in ind]

        msg = "\nLoading data for crystal {}:".format(cc)
        print(msg)

        n0 = 0
        ind0 = np.arange(0, ns)
        for ii in range(angu.size):
            msg = "\t for angle = {} deg:".format(angu[ii])
            print(msg)
            iii = ianginv == ii
            for ij in ind0[iii]:
                try:
                    msg = ("\t\tshot {}".format(shotc[ij])
                           + " ({}/{})...".format(n0+1, ind.size))
                    print(msg, end='', flush=True)
                    data, t, dbonus = _load_data(int(shotc[ij]),
                                                 tlim=tlim[ij, :],
                                                 tmode='mean',
                                                 path=None, Brightness=None,
                                                 mask=True, Verb=False)
                    thr[ij] = dbonus['THR']
                    texp[ij] = dbonus['TExpP']
                    tdelay[ij] = dbonus['TDelay']
                    tbis = np.linspace(t[0]+0.001, t[-1]-0.001, nt)
                    indt = np.digitize(tbis, t)
                    tc[ij, :] = t[indt]
                    for it in range(indt.size):
                        for ll in range(xj.size):
                            spect[ij, it, ll, :] = np.nanmean(
                                data[indt[it], indxj[ll], :],  axis=0)
                    print('\tok')
                except Exception as err:
                    success[ij] = str(err)
                    print('\tfailed: '+str(err))
                finally:
                    # save
                    np.savez(pfe,
                             shots=shotc, t=tc,
                             xi=xiref, xj=xj, indxj=indxj, spect=spect,
                             ang=ang, thr=thr, texp=texp, tdelay=tdelay,
                             success=success)
                    n0 += 1
        msg = ("Saved in:\n\t" + pfe)
        print(msg)


# #############################################################################
#                   Function to load data
# #############################################################################


_GEOM = {'pix':{'sizeH':172.e-6, 'sizeV':172.e-6,
                'nbH':487,'nbV':1467,'nbVGap':17,'nbVMod':195,
         'mod':{'nbV':7,'nbH':1,
                'sizeH':83.764e-3, 'sizeV':33.54e-3}}}


def _get_THR(shot):
    if shot>=53700 and shot<=53723:
        THR = 4024
    else:
        THR = np.nan
    return THR


def _get_Ang(shot):
    if shot>=53700 and shot<=53723:
        angle = -181.546
    elif shot>=54038 and shot<=54040:
        angle = 1.3115
    elif shot>=54041 and shot<=54044:
        angle = 1.1498
    elif shot==54045:
        angle = 1.28075
    elif shot==54046:
        angle = 1.3124
    elif shot==54047:
        angle = 1.3995
    elif shot>=54048:
        angle = 1.51995
    else:
        angle = np.nan
    return angle


def _utils_get_Pix2D(D1=0., D2=0., center=False, geom=_GEOM):

    gridH = geom['pix']['sizeH']*np.arange(0,geom['pix']['nbH'])
    gridV = geom['pix']['sizeV']*np.arange(0,geom['pix']['nbV'])
    GH = np.tile(gridH,geom['pix']['nbV'])
    GV = np.repeat(gridV,geom['pix']['nbH'])
    mH = np.mean(gridH) if center else 0.
    mV = np.mean(gridV) if center else 0.
    pts2D = np.array([D1+GH-mH, D2+GV-mV])
    return pts2D


def _get_indtlim(t, tlim=None, shot=None, out=bool):
    C0 = tlim is None
    C1 = type(tlim) in [list,tuple,np.ndarray]
    assert C0 or C1
    assert type(t) is np.ndarray

    if C0:
        tlim = [-np.inf,np.inf]
    else:
        assert len(tlim)==2
        ls = [str,int,float,np.int64,np.float64]
        assert all([tt is None or type(tt) in ls for tt in tlim])
        tlim = list(tlim)
        for (ii,sgn) in [(0,-1.),(1,1.)]:
            if tlim[ii] is None:
                tlim[ii] = sgn*np.inf
            elif type(tlim[ii]) is str and 'ign' in tlim[ii].lower():
                tlim[ii] = get_t0(shot)

    assert tlim[0]<tlim[1]
    indt = (t>=tlim[0]) & (t<=tlim[1])
    if out is int:
        indt = indt.nonzero()[0]
    return indt


def _load_data(shot, tlim=None, tmode='mean',
               path=None, geom=_GEOM,
               Brightness=None, mask=True,
               tempdir=_HERE, Verb=True):
    import pywed as pw
    from PIL import Image
    import zipfile

    assert tmode in ['mean','start','end']

    # Pre-format input
    if path is None:
        path = os.path.abspath(tempdir)
    rootstr = 'XICS {0:05.0f}:'.format(shot)

    # Load and unzip temporary file
    if Verb:
        msg = '(1/4) ' + rootstr + ' loading and unziping files...'
        print(msg)
    targetf = os.path.join(path,'xics_{0:05.0f}.zip'.format(shot))
    targetd = os.path.join(path,'xics_{0:05.0f}/'.format(shot))
    out = pw.TSRfile(shot, 'FXICS_MIDDLE', targetf)

    if not out==0:
        msg = ("Could not run:"
               + "\n    out = "
               + "pw.TSRfile({0}, 'FXICS_MIDDLE', {1})".format(shot, targetf)
               + "\n    => returned out = {0}".format(out)
               + "\n    => Maybe no data ?")
        raise Exception(msg)

    zip_ref = zipfile.ZipFile(targetf, 'r')
    zip_ref.extractall(targetd)
    zip_ref.close()

    # Load parameters to rebuild time vector
    if Verb:
        msg = '(2/4) ' + rootstr + ' loading parameters...'
        print(msg)
    t0 = 0. # Because startAcq on topOrigin (tIGNOTRON - 32 s)
    NExp = pw.TSRqParm(shot,'DXICS', 'PIL_N', 'PIL_NMax', 1)[0][0][0]
    TExpT = pw.TSRqParm(shot,'DXICS', 'PIL_Times', 'PIL_TExpT', 1)[0][0][0]
    TExpP = pw.TSRqParm(shot,'DXICS', 'PIL_Times', 'PIL_TExpP', 1)[0][0][0]
    TDelay = pw.TSRqParm(shot,'DXICS', 'PIL_Times', 'PIL_TDelay', 1)[0][0][0]
    # Delay not taken into account in this acquisition mode
    if TDelay >= 50:
        # TDelay now in ms
        TDelay *= 1.e-3
    try:
        THR = pw.TSRqParm(shot,'DXICS','PIL_THR','THR',1)[0][0][0]
    except Exception as err:
        THR = _get_THR(shot)
    try:
        Ang = pw.TSRqParm(shot,'DXICS','CRYST','Ang',1)[0][0][0]
    except Exception as err:
        Ang = _get_Ang(shot)
    if TExpP <= TExpT:
        msg = "{0:05.0f}: PIL_TExpP < PIL_TExpT in Top !".format(shot)
        raise Exception(msg)

    # Rebuild time vector
    if Verb:
        msg = '(3/4) ' + rootstr + ' Building t and data arrays...'
        print(msg)

    # Load data to numpy array and info into dict
    lf = os.listdir(targetd)
    lf = sorted([ff for ff in lf if '.tif' in ff])
    nIm = len(lf)

    # Check consistency of number of images (in case of early kill)
    if nIm>NExp:
        msg = "The zip file contains more images than parameter NExp !"
        raise Exception(msg)

    # Build time vector (parameter Delay is only for external trigger !!!)
    Dt = t0 + TExpP*np.arange(0,nIm) + np.array([[0.], [TExpT]])
    if shot >= 54132:
        # Previously, TDelay had no effect
        # From 54132, TDelay is fed to a home-made QtTimer in:
        #    controller_acquisitions.cpp:168
        #    controller_pilotage.cpp:126
        Dt += TDelay
    if tmode=='mean':
        t = np.mean(Dt, axis=0)
    elif tmode=='start':
        t = Dt[0,:]
    else:
        t = Dt[1,:]
    indt = _get_indtlim(t, tlim=tlim, out=int)
    if indt.size==0:
        msg = ("No time steps in the selected time interval:\n"
               + "\ttlim = [{0}, {1}]\n".format(tlim[0], tlim[1])
               + "\tt    = {0}".format(str(t)))
        raise Exception(msg)
    Dt, t = Dt[:, indt], t[indt]
    nt = t.size

    # Select relevant images
    lf = [lf[ii] for ii in indt]
    data = np.zeros((nt, geom['pix']['nbV'], geom['pix']['nbH']))
    ls = []
    try:
        for ii in range(0,nt):
            im = Image.open(os.path.join(targetd, lf[ii]))
            s = str(im.tag.tagdata[270]).split('#')[1:]
            s = [ss[:ss.index('\\r')] for ss in s if '\\r' in ss]
            ls.append(s)
            data[ii, :, :] = np.asarray(im, dtype=np.int32)
    finally:
        # Delete temporary files
        if Verb:
            msg = '(4/4) ' + rootstr + ' Deleting temporary files...'
            print(msg)
        os.remove(targetf)
        shutil.rmtree(targetd)

    dunits = r'photons'
    dbonus = {'Dt': Dt, 'dt': TExpT, 'THR': THR, 'mask': mask,
              'NExp': NExp, 'nIm': nIm,
              'TExpT': TExpT, 'TExpP': TExpP, 'TDelay':TDelay,
              'nH': geom['pix']['nbH'], 'nV': geom['pix']['nbV']}

    return data, t, dbonus


# #############################################################################
#                   Function to plot results
# #############################################################################

_MASKXI = np.ones((487,), dtype=bool)
_MASKXI[436:] = False


def _get_crystanddet(cryst=None, det=None):
    # Cryst part
    if cryst is None:
        cryst = False
    elif isinstance(cryst, str) and cryst in _DCRYST.keys():
        crystobj = tf.load(_DCRYST[cryst])
        if det is None:
            if cryst in _DDET.keys():
                det = crystobj.get_detector_approx(**_DDET[cryst])
                cryst = crystobj
            else:
                msg = "Det must be provided if cryst is provided!"
                raise Exception(msg)
            det = {'det_cent': det[0], 'det_nout': det[1],
                   'det_ei': det[2], 'det_ej': det[3]}
        cryst = crystobj

    if isinstance(cryst, str) and os.path.isfile(cryst):
        cryst = tf.load(cryst)

    if cryst is not False:
        c0 = (isinstance(det, dict)
              and all([kk in det.keys() for kk in ['det_cent', 'det_nout',
                                                   'det_ei', 'det_ej']]))
        if not c0:
            msg = ("det must be a dict with keys:\n"
                   + "\t- det_cent: [x,y,z] of the detector center\n"
                   + "\t- det_nout: [x,y,z] of unit vector normal to plane\n"
                   + "\t- det_ei: [x,y,z] of unit vector ei\n"
                   + "\t- det_ej: [x,y,z] of unit vector ej = nout x ei\n"
                   + "\n\t- provided: {}".format(det))
            raise Exception(msg)
    return cryst, det


def _extract_data(pfe, allow_pickle=None,
                  maskxi=None, shot=None, indt=None, indxj=None):
    # Prepare data
    out = np.load(pfe, allow_pickle=allow_pickle)
    t, xi, xj, ang = [out[kk] for kk in ['t', 'xi', 'xj', 'ang']]
    spect, shots, thr = [out[kk] for kk in ['spect', 'shots', 'thr']]

    if maskxi is not False:
        xi = xi[maskxi]
        spect = spect[:, :, :, maskxi]

    # Remove unknown angles
    indok = ~np.isnan(ang)
    if not np.any(indok):
        msg = "All nan angles!"
        raise Exception(msg)
    shots = shots[indok]
    t = t[indok, :]
    ang = ang[indok]
    spect = spect[indok, :, :, :]

    if shot is not None:
        indok = np.array([ss in shot for ss in shots])
        if not np.any(indok):
            msg = ("Desired shot not in shots!\n"
                   + "\t- provided: {}\n".format(shot)
                   + "\t- shots: {}\n".format(shots)
                   + "\t (ang):  {}".format(ang))
            raise Exception(msg)
        shots = shots[indok]
        t = t[indok, :]
        ang = ang[indok]
        spect = spect[indok, :, :, :]

    if indt is not None:
        indt = np.r_[indt].astype(int)
        t = t[:, indt]
        spect = spect[:, indt, :, :]

    if indxj is not None:
        indxj = np.r_[indxj].astype(int)
        xj = xj[indxj]
        spect = spect[:, :, indxj, :]

    spectn = spect - np.nanmin(spect, axis=-1)[..., None]
    spectn /= np.nanmax(spectn, axis=-1)[..., None]
    return spect, spectn, shots, t, ang, xi, xj, thr


def plot(pfe=None, allow_pickle=True,
         shot=None, maskxi=None,
         cryst=None, det=None,
         fs=None, dmargin=None, cmap=None):

    # Check input
    if not os.path.isfile(pfe):
        msg = ("Provided file does not exist!"
               + "\t- provided: {}".format(pfe))
        raise Exception(msg)

    if shot is not None:
        if not hasattr(shot, '__iter__'):
            shot = np.array([shot], dtype=int)
        else:
            shot = np.r_[shot].astype(int)

    if maskxi is None:
        maskxi = _MASKXI

    # Cryst part
    cryst, det = _get_crystanddet(cryst=cryst, det=det)

    # extract data
    spect, spectn, shots, t, ang, xi, xj, thr = _extract_data(pfe,
                                                              allow_pickle,
                                                              maskxi, shot)
    nshot, nt, nxj, nxi = spect.shape

    # smean = np.nanmean(spect, axis=-1)
    # smax = np.nanmax(spect, axis=-1)
    iout = np.any(np.nanmean(spectn**2, axis=-1) > 0.1, axis=-1)

    # Group by angle
    angu, ind_ang = np.unique(ang, return_inverse=True)
    nang = angu.size
    lcol = ['r', 'g', 'b', 'k', 'm', 'c', 'y']
    ncol = len(lcol)

    # Cryst data
    if cryst is not False:
        lamb = np.full((nshot, nxj, nxi), np.nan)
        phi = np.full((nshot, nxj, nxi), np.nan)
        xif = np.tile(xi, (nxj, 1))
        xjf = np.repeat(xj[:, None], nxi, axis=1)
        for jj in range(nang):
            ind = (ang == angu[jj]).nonzero()[0]
            # Beware to provide angles in rad !
            cryst.move(angle=angu[jj]*np.pi/180.)

            bragg, phii = cryst.calc_phibragg_from_xixj(
                xif, xjf, n=1,
                dtheta=None, psi=None, plot=False, **det)
            phi[ind, ...] = phii[None, ...]
            lamb[ind, ...] = cryst.get_lamb_from_bragg(bragg, n=1)[None, ...]

    # -------------
    # Plot 1

    if fs is None:
        fs = (18, 9)
    if cmap is None:
        cmap = plt.cm.viridis
    if dmargin is None:
        dmargin = {'left':0.05, 'right':0.99,
                   'bottom':0.06, 'top':0.93,
                   'wspace':0.3, 'hspace':0.2}

    fig = plt.figure(figsize=fs)
    if shot is not None:
        fig.suptitle('shot = {}'.format(shot))
    gs = gridspec.GridSpec(nxj, 2, **dmargin)

    dax = {'spect': [None for ii in range(nxj)],
           'spectn': [None for ii in range(nxj)]}

    shx = None
    for ii in range(nxj):
        dax['spect'][ii] = fig.add_subplot(gs[ii, 0], sharex=shx)
        if ii == 0:
            shx = dax['spect'][0]
        dax['spectn'][ii] = fig.add_subplot(gs[ii, 1], sharex=shx)
        dax['spect'][ii].set_ylabel('xj = {}\ndata (a.u.)'.format(xj[ii]))

        for jj in range(nang):
            col = lcol[jj%ncol]
            lab0 = 'ang {}'.format(angu[jj])
            ind = (ang == angu[jj]).nonzero()[0]
            xibis = xi #+ angu[jj]*0.05
            for ss in range(ind.size):
                for tt in range(nt):
                    ls = '--' if iout[ind[ss], tt] else '-'
                    lab = lab0 + ', {}, t = {} s'.format(shots[ind[ss]],
                                                         t[ind[ss], tt])
                    dax['spect'][ii].plot(xibis,
                                          spect[ind[ss], tt, ii, :],
                                          c=col, ls=ls, label=lab)
                    dax['spectn'][ii].plot(xibis,
                                           spectn[ind[ss], tt, ii, :],
                                           c=col, ls=ls, label=lab)

    # Polish
    dax['spect'][0].set_title('raw spectra')
    dax['spectn'][0].set_title('normalized spectra')
    dax['spect'][-1].set_xlabel('xi (m)')
    dax['spectn'][-1].set_xlabel('xi (m)')
    hand = [mlines.Line2D([], [], c=lcol[jj%ncol], ls='-')
            for jj in range(nang)]
    lab = ['{}'.format(aa) for aa in angu]
    dax['spect'][0].legend(hand, lab,
                           title='Table angle (deg.)',
                           loc='upper left',
                           bbox_to_anchor=(1.01, 1.))

    # -------------
    # Plot 2
    if cryst is False:
        return dax

    fig = plt.figure(figsize=fs)
    if shot is not None:
        fig.suptitle('shot = {}'.format(shot))
    gs = gridspec.GridSpec(nxj, 2, **dmargin)

    dax2 = {'spect': [None for ii in range(nxj)],
            'spectn': [None for ii in range(nxj)]}

    shx = None
    for ii in range(nxj):
        dax2['spect'][ii] = fig.add_subplot(gs[ii, 0], sharex=shx)
        if ii == 0:
            shx = dax2['spect'][0]
        dax2['spectn'][ii] = fig.add_subplot(gs[ii, 1], sharex=shx)
        dax2['spect'][ii].set_ylabel('data (a.u.)'.format(xj[ii]))

        for jj in range(nang):
            col = lcol[jj%ncol]
            lab0 = 'ang {}'.format(angu[jj])
            ind = (ang == angu[jj]).nonzero()[0]
            xibis = xi #+ angu[jj]*0.05
            for ss in range(ind.size):
                for tt in range(nt):
                    ls = '--' if iout[ind[ss], tt] else '-'
                    lab = lab0 + ', {}, t = {} s'.format(shots[ind[ss]],
                                                         t[ind[ss], tt])
                    dax2['spect'][ii].plot(lamb[ind[ss], ii, :],
                                           spect[ind[ss], tt, ii, :],
                                           c=col, ls=ls, label=lab)
                    dax2['spectn'][ii].plot(lamb[ind[ss], ii, :],
                                            spectn[ind[ss], tt, ii, :],
                                            c=col, ls=ls, label=lab)

    # Polish
    dax2['spect'][0].set_title('raw spectra')
    dax2['spectn'][0].set_title('normalized spectra')
    dax2['spect'][-1].set_xlabel(r'$\lambda$' + ' (m)')
    dax2['spectn'][-1].set_xlabel(r'$\lambda$' + ' (m)')
    hand = [mlines.Line2D([], [], c=lcol[jj%ncol], ls='-')
            for jj in range(nang)]
    lab = ['{}'.format(aa) for aa in angu]
    dax2['spect'][0].legend(hand, lab,
                            title='Table angle (deg.)',
                            loc='upper left',
                            bbox_to_anchor=(1.01, 1.))
    return dax, dax2


# #############################################################################
#                   Scan detector
# #############################################################################


def fit(pfe=None, allow_pickle=True,
        shot=None, indt=None, indxj=None, maskxi=None,
        cryst=None, det=None,
        dlines=None, dconstraints=None, dx0=None,
        key0=None, key1=None,
        lambmin=None, lambmax=None,
        method=None, max_nfev=None,
        scales=None, x0_scale=None, bounds_scale=None,
        xtol=None, ftol=None, gtol=None,
        loss=None, verbose=None, showonly=None,
        fs=None, dmargin=None, cmap=None):

    # Check input
    if not os.path.isfile(pfe):
        msg = ("Provided file does not exist!\n"
               + "\t- provided: {}".format(pfe))
        raise Exception(msg)

    if shot is not None:
        if not hasattr(shot, '__iter__'):
            shot = np.array([shot], dtype=int)
        else:
            shot = np.r_[shot].astype(int)

    if maskxi is None:
        maskxi = _MASKXI

    # Cryst part
    cryst, det = _get_crystanddet(cryst=cryst, det=det)
    assert cryst is not False

    # extract data
    spect, spectn, shots, t, ang, xi, xj, thr = _extract_data(pfe,
                                                              allow_pickle,
                                                              maskxi, shot,
                                                              indt, indxj)
    nshot, nt, nxj, nxi = spect.shape

    # smean = np.nanmean(spect, axis=-1)
    # smax = np.nanmax(spect, axis=-1)
    iout = np.any(np.nanmean(spectn**2, axis=-1) > 0.1, axis=-1)

    # Group by angle
    angu, ind_ang = np.unique(ang, return_inverse=True)
    nang = angu.size
    lcol = ['r', 'g', 'b', 'k', 'm', 'c', 'y']
    ncol = len(lcol)

    # Get dinput for 1d fitting
    dinput = tf.data._spectrafit2d.multigausfit1d_from_dlines_dinput(
        dlines=dlines,
        dconstraints=dconstraints,
        lambmin=lambmin, lambmax=lambmax)
    if key0 is not None:
        indl0 = (dinput['keys'] == key0).nonzero()[0]
        if indl0.size != 1:
            msg = ("key0 not valid:\n"
                   + "\t- provided:  {}\n".format(key0)
                   + "\t- available: {}".format(dinput['keys']))
            raise Exception(msg)
        indl0 = indl0[0]
    if key1 is not None:
        indl1 = (dinput['keys'] == key1).nonzero()[0]
        if indl1.size != 1:
            msg = ("key1 not valid:\n"
                   + "\t- provided:  {}\n".format(key1)
                   + "\t- available: {}".format(dinput['keys']))
            raise Exception(msg)
        indl1 = indl1[0]

    # Cryst data
    lamb = np.full((nang, nxj, nxi), np.nan)
    phi = np.full((nang, nxj, nxi), np.nan)
    xif = np.tile(xi, (nxj, 1))
    xjf = np.repeat(xj[:, None], nxi, axis=1)
    for jj in range(nang):
        # Beware to provide angles in rad !
        cryst.move(angle=angu[jj]*np.pi/180.)

        bragg, phii = cryst.calc_phibragg_from_xixj(
            xif, xjf, n=1,
            dtheta=None, psi=None, plot=False, **det)
        phi[jj, ...] = phii[None, ...]
        lamb[jj, ...] = cryst.get_lamb_from_bragg(bragg, n=1)[None, ...]

    # Reorder to sort lamb
    assert np.all(np.argsort(lamb, axis=-1)
                  == np.arange(nxi-1, -1, -1)[None, None, :])
    xi = xi[::-1]
    lamb = lamb[:, :, ::-1]
    phi = phi[:, :, ::-1]
    spect = spect[:, :, :, ::-1]
    spectn = spectn[:, :, :, ::-1]

    # Fit
    spectfit = np.full(spect.shape, np.nan)
    time = np.full(spect.shape[:-1], np.nan)
    cost = np.full(spect.shape[:-1], np.nan)
    if key0 is not None:
        shift0 = np.full(spect.shape[:-1], np.nan)
    if key1 is not None:
        shift1 = np.full(spect.shape[:-1], np.nan)
    for jj in range(nang):
        msg = ("\nOptimizing for ang = {}  ({}/{})\n".format(angu[jj],
                                                             jj+1, nang)
               + "--------------------------------")
        print(msg)
        ind = (ang == angu[jj]).nonzero()[0]
        for ll in range(ind.size):
            msgsh = "---------- shot {} ({}/{})".format(shots[ind[ll]],
                                                        ll+1, ind.size)
            for ii in range(nxj):
                msg = ("  xj = {}  ({}/{}):".format(xj[ii], ii+1, nxj)
                       + "\t{} spectra".format(spectn[jj][ii].shape[0]))
                print(msgsh + msg)
                dfit1d = tf.data._spectrafit2d.multigausfit1d_from_dlines(
                    spectn[ind[ll], :, ii, :],
                    lamb[jj, ii, :],
                    dinput=dinput, dx0=dx0,
                    lambmin=lambmin, lambmax=lambmax,
                    scales=scales, x0_scale=x0_scale, bounds_scale=bounds_scale,
                    method=method, max_nfev=max_nfev,
                    chain=True, verbose=verbose,
                    xtol=xtol, ftol=ftol, gtol=gtol, loss=loss,
                    ratio=None, jac='call')
                spectfit[ind[ll], :, ii, :] = dfit1d['sol']
                time[ind[ll], :, ii] = dfit1d['time']
                cost[ind[ll], :, ii] = dfit1d['cost']
                if key0 is not None:
                    shift0[ind[ll], :, ii] = dfit1d['shift'][:, indl0]
                if key1 is not None:
                    shift1[ind[ll], :, ii] = dfit1d['shift'][:, indl1]
    shiftabs = 0.
    if key0 is not None:
        shiftabs = max(np.max(np.abs(shift0)), shiftabs)
    if key1 is not None:
        shiftabs = max(np.max(np.abs(shift1)), shiftabs)

    # -------------
    # Plot
    if plot is False:
        return {'shots': shots, 't': t, 'ang': ang,
                'lamb': lamb, 'phi': phi,
                'spectn': spectn, 'spectfit': spectfit,
                'time': time, 'cost': cost,
                'shitf0': shift0, 'shift1': shitf1}

    if fs is None:
        fs = (18, 9)
    if cmap is None:
        cmap = plt.cm.viridis
    if dmargin is None:
        dmargin = {'left':0.05, 'right':0.99,
                   'bottom':0.06, 'top':0.93,
                   'wspace':0.3, 'hspace':0.2}
    extent = (0.5, nshot+0.5, -0.5, nt-0.5)

    fig = plt.figure(figsize=fs)
    if shot is not None:
        fig.suptitle('shot = {}'.format(shot))
    gs = gridspec.GridSpec(nxj*2, 7, **dmargin)

    dax = {'spectn': [None for ii in range(nxj)],
           'time': [None for ii in range(nxj)],
           'cost': [None for ii in range(nxj)],
           'shift0': [None for ii in range(nxj)],
           'shift1': [None for ii in range(nxj)],
           'shift0_z': [None for ii in range(nxj)],
           'shift1_z': [None for ii in range(nxj)],
          }

    xones = np.zeros((nt,))
    isortxj = np.argsort(xj)[::-1]
    shx0, shx1, shy1, shx20, shx21 = None, None, None, None, None
    for ii in range(nxj):
        iax = isortxj[ii]
        dax['spectn'][iax] = fig.add_subplot(gs[iax*2:iax*2+2, :3], sharex=shx0)
        if ii == 0:
            shx0 = dax['spectn'][isortxj[0]]
        dax['time'][iax] = fig.add_subplot(gs[iax*2, 3],
                                           sharex=shx1, sharey=shy1)
        if ii == 0:
            shx1 = dax['time'][iax]
            shy1 = dax['time'][iax]
        dax['cost'][iax] = fig.add_subplot(gs[iax*2, 4],
                                           sharex=shx1, sharey=shy1)
        dax['shift0'][iax] = fig.add_subplot(gs[iax*2, 5],
                                             sharex=shx1, sharey=shy1)
        dax['shift1'][iax] = fig.add_subplot(gs[iax*2, 6],
                                             sharex=shx1, sharey=shy1)
        dax['shift0_z'][iax] = fig.add_subplot(gs[iax*2+1, 5],
                                               sharex=shx20)
        dax['shift1_z'][iax] = fig.add_subplot(gs[iax*2+1, 6],
                                               sharex=shx21)
        if ii == 0:
            shx20 = dax['shift0_z'][iax]
            shx21 = dax['shift1_z'][iax]
        dax['spectn'][iax].set_ylabel('data (a.u.)'.format(xj[ii]))
        if iax != nxj-1:
            plt.setp(dax['time'][iax].get_xticklabels(), visible=False)
            plt.setp(dax['cost'][iax].get_xticklabels(), visible=False)

        for jj in range(nang):
            col = lcol[jj%ncol]
            ind = (ang == angu[jj]).nonzero()[0]
            for ll in range(ind.size):
                dax['spectn'][iax].plot(lamb[jj, ii, :],
                                        spectn[ind[ll], :, ii, :].T,
                                        ls='None', marker='.', ms=4., c=col)
                dax['spectn'][iax].plot(lamb[jj, ii, :],
                                        spectfit[ind[ll], :, ii, :].T,
                                        ls='-', lw=1, c=col)
                if key0 is not None:
                    dax['shift0_z'][iax].plot((dinput['lines'][indl0]
                                               + shift0[ind[ll], :, ii]),
                                              xones,
                                              marker='.', ls='None', c=col)
                if key1 is not None:
                    dax['shift1_z'][iax].plot((dinput['lines'][indl1]
                                               + shift1[ind[ll], :, ii]),
                                              xones,
                                              marker='.', ls='None', c=col)
        dax['time'][iax].imshow(time[:, :, ii].T,
                                extent=extent, cmap=cmap,
                                interpolation='nearest', origin='lower')
        dax['cost'][iax].imshow(cost[:, :, ii].T,
                                extent=extent, cmap=cmap,
                                interpolation='nearest', origin='lower')
        if key0 is not None:
            dax['shift0'][iax].imshow(shift0[:, :, ii].T,
                                      extent=extent, cmap=plt.cm.seismic,
                                      interpolation='nearest', origin='lower',
                                      vmin=-shiftabs, vmax=shiftabs)
            dax['spectn'][iax].axvline(dinput['lines'][indl0],
                                       c='k', ls='-', lw=1.)
            dax['shift0_z'][iax].axvline(dinput['lines'][indl0],
                                         c='k', ls='-', lw=1.)
        if key1 is not None:
            dax['shift1'][iax].imshow(shift1[:, :, ii].T,
                                      extent=extent, cmap=plt.cm.seismic,
                                      interpolation='nearest', origin='lower',
                                      vmin=-shiftabs, vmax=shiftabs)
            dax['spectn'][iax].axvline(dinput['lines'][indl1],
                                       c='k', ls='-', lw=1.)
            dax['shift1_z'][iax].axvline(dinput['lines'][indl1],
                                         c='k', ls='-', lw=1.)
    # Polish
    xlab = ['{}'.format(ss) for ss in shots]
    dax['time'][0].set_title(r'time')
    dax['cost'][0].set_title(r'$\chi^2_{norm}$')
    dax['shift0'][0].set_title('shift {}'.format(key0))
    dax['shift1'][0].set_title('shift {}'.format(key1))
    dax['time'][-1].set_xticks(range(1, nshot+1))
    dax['time'][-1].set_xticklabels(xlab, rotation=75)
    dax['cost'][-1].set_xticklabels(xlab, rotation=75)
    dax['time'][-1].set_yticks(range(0, nt))
    return dax


def scan_det(pfe=None, allow_pickle=True,
             ndx=None, dx=None, ndrot=None, drot=None,
             shot=None, indt=None, indxj=None, maskxi=None,
             cryst=None, det=None,
             dlines=None, dconstraints=None, dx0=None,
             key0=None, key1=None,
             lambmin=None, lambmax=None,
             method=None, max_nfev=None,
             scales=None, x0_scale=None, bounds_scale=None,
             xtol=None, ftol=None, gtol=None,
             loss=None, verbose=None, showonly=None,
             fs=None, dmargin=None, cmap=None):

    refdet = None
    dxv = np.linspace(-ndx*dx, ndx*dx, 2*nx)
    drotv = np.linspace(-ndrot*drot, ndrot*drot, 2*ndrot)
