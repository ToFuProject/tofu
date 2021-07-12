# -*- coding: utf-8 -*-


import os
import sys
import shutil
import warnings
import datetime as dtm


import numpy as np
import scipy.optimize as scpopt
import scipy.stats as scpstats
import scipy.interpolate as scpinterp
import scipy.optimize as scpopt
import scipy.sparse as scpsparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.colors as mcolors


_HERE = os.path.dirname(__file__)
_TOFUPATH = os.path.abspath(os.path.join(_HERE, os.pardir))


sys.path.insert(1, _TOFUPATH)
import tofu as tf
from inputs_temp.dlines import dlines
import inputs_temp.XICS_allshots_C34 as xics
_ = sys.path.pop(1)


print(
    (
        'tofu in {}: \n\t'.format(__file__)
        + tf.__version__
        + '\n\t'
        + tf.__file__
    ),
    file=sys.stdout,
)


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
#                   Detector from CAD
# #############################################################################


_DET_CAD_CORNERS_XYZ = np.array([
    [-2332.061, -126.662, -7606.628],
    [-2363.382, -126.662, -7685.393],
    [-2363.382,  126.662, -7685.393],
    [-2332.061,  126.662, -7606.628],
]).T*1.e-3
# (-x)zy -> xyz
_DET_CAD_CORNERS_XYZ = np.array([-_DET_CAD_CORNERS_XYZ[0, :],
                                 _DET_CAD_CORNERS_XYZ[2, :],
                                 _DET_CAD_CORNERS_XYZ[1, :]])
_DET_CAD_CENT = np.mean(_DET_CAD_CORNERS_XYZ, axis=1)
_DET_CAD_EI = _DET_CAD_CORNERS_XYZ[:, 1] - _DET_CAD_CORNERS_XYZ[:, 0]
_DET_CAD_EI = _DET_CAD_EI / np.linalg.norm(_DET_CAD_EI)
_DET_CAD_EJ = _DET_CAD_CORNERS_XYZ[:, -1] - _DET_CAD_CORNERS_XYZ[:, 0]
_DET_CAD_EJ = _DET_CAD_EJ - np.sum(_DET_CAD_EJ*_DET_CAD_EI)*_DET_CAD_EI
_DET_CAD_EJ = _DET_CAD_EJ / np.linalg.norm(_DET_CAD_EJ)
_DET_CAD_NOUT = np.cross(_DET_CAD_EI, _DET_CAD_EJ)
_DET_CAD_NOUT = _DET_CAD_NOUT / np.linalg.norm(_DET_CAD_NOUT)


# #############################################################################
#                   Spectral lines dict
# #############################################################################


_DLINES_ARXVII = {
    k0: v0 for k0, v0 in dlines.items()
    if (
        (
            v0['source'] == 'Vainshtein 85'
            and v0['ION'] == 'ArXVII'
            and v0['symbol'] not in ['y2', 'z2']
        )
        or (
            v0['source'] == 'Goryaev 17'
            and v0['ION'] == 'ArXVI'
            and v0['symbol'] not in [
                'l', 'n3-h1', 'n3-h2', 'd',
                'n3-e1', 'n3-f4', 'n3-f2', 'n3-e2',
                'n3-f1', 'n3-g1', 'n3-g2', 'n3-g3',
                'n3-f3', 'n3-a1', 'n3-a2', 'n3-c1',
                'n3-c2', 'g', 'i', 'e', 'f', 'u',
                'v', 'h', 'c', 'b', 'n3-b1',
                'n3-b2', 'n3-b4', 'n3-d1', 'n3-d2',
            ]
        )
    )
}


# #############################################################################
#                   Hand-picked database
# #############################################################################


_SHOTS = np.r_[
    # C3
    54041,
    np.arange(54043, 54055),
    54058, 54059,
    np.arange(54061, 54068),
    np.arange(54069, 54076),
    np.arange(54077, 54080),
    54081,
    54083, 54084,
    np.arange(54088, 54108),
    54123,
    np.arange(54126, 54146),
    np.arange(54150, 54156),
    np.arange(54158, 54169),
    np.arange(54170, 54176),
    54177, 54178,
    # C4
    54762, 54765, 54766, 55045, 55049,
    55076, 55077, 55092, 55095, 55080,
    55147, 55160, 55161, 55164, 55165, 55166, 55167,
    55292, 55297,
    55562, 55572, 55573, 55607,
]

_NSHOT = _SHOTS.size
_CAMP = np.full((_NSHOT,), 3)
_CAMP[_SHOTS > 54178] = 4

_TLIM = np.tile([-np.inf, np.inf], (_NSHOT, 1))

_CRYST = np.full((_NSHOT,), 'ArXVII', dtype='<U7')
iArXVIII = (_SHOTS >= 54062) & (_SHOTS <= 54107)
iFe = (_SHOTS >= 54123) & (_SHOTS <= 54178)
_CRYST[iArXVIII] = 'ArXVIII'
_CRYST[iFe] = 'FeXXV'
_ANG = np.full((_NSHOT,), np.nan)

_DSHOTS = {
    'ArXVII': {
        # C3
        # 54041: {'ang': 1.1498, 'tlim': [32, 36]}, # Almost no signal
        54043: {'ang': 1.1498, 'tlim': [35, 39]},
        54044: {'ang': 1.1498, 'tlim': [33, 47]},
        54045: {'ang': 1.28075, 'tlim': [32, 46]},
        54046: {'ang': 1.3124, 'tlim': [32, 46]},
        54047: {'ang': 1.3995, 'tlim': [32, 46]},
        54048: {'ang': 1.51995, 'tlim': [32, 46]},
        54049: {'ang': 1.51995, 'tlim': [32, 34]},
        54050: {'ang': 1.51995, 'tlim': [32, 46]},
        54051: {'ang': 1.51995, 'tlim': [32, 40]},
        54052: {'ang': 1.51995, 'tlim': [32, 37]},
        54053: {'ang': 1.51995, 'tlim': [32, 34]},
        54054: {'ang': 1.51995, 'tlim': [32, 37]},
        54061: {'ang': 1.6240, 'tlim': [32, 43]},
        # C4   1.3115 ?
        54762: {'ang': 1.3405, 'tlim': [34.0, 44.5]},   # ok
        54765: {'ang': 1.3405, 'tlim': [33.0, 44.5]},   # ok
        54766: {'ang': 1.3405, 'tlim': [33.0, 41.0]},   # ok
        55045: {'ang': 1.3405, 'tlim': [32.5, 38.5]},   # ok
        55049: {'ang': 1.3405, 'tlim': [32.5, 44.5]},   # ok
        55076: {'ang': 1.3405, 'tlim': [32.0, 56.0]},   # ok
        55077: {'ang': 1.3405, 'tlim': [32.5, 54.0]},   # ok
        55080: {'ang': 1.3405, 'tlim': [32.5, 49.0]},   # ok
        55092: {'ang': 1.3405, 'tlim': [32.5, 57.5]},   # ok
        55095: {'ang': 1.3405, 'tlim': [32.5, 47.5]},   # ok
        55147: {'ang': 1.3405, 'tlim': [32.6, 45.6]},   # ICRH, ok, good
        55160: {'ang': 1.3405, 'tlim': [32.4, 44.6]},   # ICRH, ok
        55161: {'ang': 1.3405, 'tlim': [32.6, 42.4]},   # ICRH, ok
        55164: {'ang': 1.3405, 'tlim': [32.5, 44.7]},   # ICRH, ok
        55165: {'ang': 1.3405, 'tlim': [32.7, 44.5]},   # ICRH, ok
        55166: {'ang': 1.3405, 'tlim': [32.5, 42.2]},   # ok
        55167: {'ang': 1.3405, 'tlim': [32.5, 42.4]},   # ICRH, ok
        55292: {'ang': 1.3405, 'tlim': [32.5, 46.2]},   # ok
        55297: {'ang': 1.3405, 'tlim': [32.5, 46.0]},   # ICRH, ok
        55562: {'ang': 1.3405, 'tlim': [32.5, 47.5]},   # ICRH, ok, good
        55572: {'ang': 1.3405, 'tlim': [33.0, 45.2]},   # ICRH, ok, good
        55573: {'ang': 1.3405, 'tlim': [30.6, 32.6]},   # ok, good startup
        55607: {'ang': 1.3405, 'tlim': [32.5, 40.4]},   # ICRH
    },

    'ArXVIII': {
        54062: {'ang': -101.0, 'tlim': [32, 37]},
        54063: {'ang': -101.0, 'tlim': [32, 43]},
        54064: {'ang': -101.0, 'tlim': [32, 43]},
        54065: {'ang': -101.099, 'tlim': [32, 44]},
        54066: {'ang': -101.099, 'tlim': [32, 41]},
        54067: {'ang': -101.099, 'tlim': [32, 43]},
        54069: {'ang': -101.099, 'tlim': [32, 40]},
        54070: {'ang': -101.099, 'tlim': [32, 38]},
        54071: {'ang': -101.099, 'tlim': [32, 40]},
        54072: {'ang': -101.099, 'tlim': [32, 37]},
        54073: {'ang': -101.2218, 'tlim': [32, 38]},
        54074: {'ang': -101.2218, 'tlim': [32, 37]},
        54075: {'ang': -101.2218, 'tlim': [32, 37]},
        54077: {'ang': -101.3507, 'tlim': [32, 34]},
        54088: {'ang': -101.3507, 'tlim': [32, 38]},
        54089: {'ang': -101.3507, 'tlim': [32, 45]},
        54090: {'ang': -101.4831, 'tlim': [32, 40]},
        54091: {'ang': -101.5800, 'tlim': [32, 40]},
        54092: {'ang': -101.5800, 'tlim': [32, 40]},
        54093: {'ang': -100.924, 'tlim': [32, 37]},
        54094: {'ang': -100.924, 'tlim': [32, 40]},
        54095: {'ang': -100.799, 'tlim': [32, 48]},
        54096: {'ang': -100.799, 'tlim': [32, 39]},
        54097: {'ang': -100.799, 'tlim': [32, 37]},
        54098: {'ang': -100.706, 'tlim': [32, 39]},
        54099: {'ang': -100.706, 'tlim': [32, 39]},
        54100: {'ang': -100.580, 'tlim': [32, 44]},
        54101: {'ang': -100.483, 'tlim': [32, 40]},
        54102: {'ang': -100.386, 'tlim': [32, 45]},
        54103: {'ang': -100.386, 'tlim': [32, 38]},
        54104: {'ang': -100.2644, 'tlim': [32, 38]},
        54105: {'ang': -100.132, 'tlim': [32, 40]},
        54107: {'ang': -100.038, 'tlim': [32, 38]},
    },

    'FeXXV': {
        54123: {'ang': -181.547, 'tlim': [32, 59]},
        54126: {'ang': -181.547, 'tlim': [32, 38]},
        54127: {'ang': -181.547, 'tlim': [32, 49]},
        54128: {'ang': -181.547, 'tlim': [32, 61]},
        54129: {'ang': -181.547, 'tlim': [32, 46]},
        54130: {'ang': -181.547, 'tlim': [32, 59]},
        54131: {'ang': -181.647, 'tlim': [32, 64]},
        54133: {'ang': -181.746, 'tlim': [32, 67]},
        54134: {'ang': -181.846, 'tlim': [32, 63]},
        54135: {'ang': -181.946, 'tlim': [32, 60]},
        54136: {'ang': -181.428, 'tlim': [32, 63]},
        54137: {'ang': -181.3222, 'tlim': [32, 44]},
        54138: {'ang': -181.1954, 'tlim': [32, 42]},
        54139: {'ang': -181.1954, 'tlim': [32, 65]},
        54141: {'ang': -181.1954, 'tlim': [32, 59]},
        54142: {'ang': -181.1954, 'tlim': [32, 54]},
        54143: {'ang': -181.1954, 'tlim': [32, 66]},
        54144: {'ang': -181.1954, 'tlim': [32, 65]},
        54145: {'ang': -181.1954, 'tlim': [32, 40]},
        54150: {'ang': -181.0942, 'tlim': [32, 57]},
        54151: {'ang': -181.0942, 'tlim': [32, 40]},
        54152: {'ang': -180.9625, 'tlim': [32, 61]},
        54153: {'ang': -180.9625, 'tlim': [32, 49]},
        54154: {'ang': -180.8651, 'tlim': [32, 49]},
        54155: {'ang': -180.8651, 'tlim': [32, 47]},
        54158: {'ang': -180.8651, 'tlim': [32, 67]},
        54159: {'ang': -180.7667, 'tlim': [32, 63]},
        54160: {'ang': -180.7667, 'tlim': [32, 66]},
        54161: {'ang': -180.6687, 'tlim': [32, 40]},
        54162: {'ang': -180.6687, 'tlim': [32, 37]},
        54163: {'ang': -180.6687, 'tlim': [32, 66]},
        54164: {'ang': -180.5434, 'tlim': [32, 65]},
        54165: {'ang': -180.5803, 'tlim': [32, 39]},
        54166: {'ang': -180.5803, 'tlim': [32, 65]},
        54167: {'ang': -181.6169, 'tlim': [32, 37]},
        54173: {'ang': -181.6169, 'tlim': [32, 69.5]},
        54178: {'ang': -181.6169, 'tlim': [32, 69.5]},
    }
}

_DSHOTS_DEF = {
    'ArXVII': [54044, 54045, 54046, 54047, 54049, 54061],
    'FeXXV': [],
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


_CRYSTBASE = 'TFG_CrystalBragg_ExpWEST_DgXICS_'
_DCRYST = {
    'ArXVII': os.path.abspath(os.path.join(
        _HERE,
        _CRYSTBASE + 'ArXVII_sh00000_Vers1.4.7-208-gb3dcce6e.npz',
    )),
    'ArXVIII': os.path.abspath(os.path.join(
        _HERE,
        _CRYSTBASE + 'ArXVIII_sh00000_Vers1.4.7-221-g65718177.npz',
    )),
    'FeXXV': os.path.abspath(os.path.join(
        _HERE,
        _CRYSTBASE + 'FeXXV_sh00000_Vers1.4.7-221-g65718177.npz',
    )),
}

_DDET = {
    'ArXVII': dict(
        ddist=0., di=-0.005, dj=0., dtheta=0., dpsi=-0.01,
        tilt=0.008, tangent_to_rowland=True,
    ),
    'FeXXV': dict(
        ddist=0., di=0., dj=0.,
        dtheta=0., dpsi=0., tilt=0., tangent_to_rowland=True,
    ),
}


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
    'det37_CTVD_incC4_New.npz'
))
_MASK = ~np.any(np.load(_MASKPATH)['ind'], axis=0)
_DLINES = None
_XJJ = np.r_[-0.08, -0.05, 0., 0.05, 0.1]
_DXJ = 0.002


def main(shots=_SHOTS,
         path=None,
         nt=None,
         cryst=None,
         dcryst=None,
         lfiles=None,
         maskpath=None,
         xj=None,
         dxj=None):
    """ Create file XICS_data_{cryst}.npz  """

    # ---------
    # Check input
    if path is None:
        path = _PATH
    if dcryst is None:
        dcryst = _DCRYST
    if cryst is None:
        cryst = sorted(dcryst.keys())
    if isinstance(cryst, str):
        cryst = [cryst]
    assert all([cc in dcryst.keys() for cc in cryst])
    cryst = set(dcryst.keys()).intersection(cryst)
    if lfiles is None:
        lfiles = [_PATHC3, _PATHC4]
    if isinstance(lfiles, str):
        lfiles = [lfiles]
    if nt is None:
        nt = _NT
    if xj is None:
        xj = _XJJ
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
    for cc in cryst:

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


_GEOM = {
    'pix': {
        'sizeH': 172.e-6, 'sizeV': 172.e-6,
        'nbH': 487, 'nbV': 1467, 'nbVGap': 17, 'nbVMod': 195,
        'mod': {'nbV': 7, 'nbH': 1, 'sizeH': 83.764e-3, 'sizeV': 33.54e-3}
    }
}


def _get_THR(shot):
    if shot >= 53700 and shot <= 53723:
        THR = 4024
    else:
        THR = np.nan
    return THR


def _get_Ang(shot):
    if shot >= 53700 and shot <= 53723:
        angle = -181.546
    elif shot >= 54038 and shot <= 54040:
        angle = 1.3115
    elif shot >= 54041 and shot <= 54044:
        angle = 1.1498
    elif shot == 54045:
        angle = 1.28075
    elif shot == 54046:
        angle = 1.3124
    elif shot == 54047:
        angle = 1.3995
    elif shot >= 54048:
        angle = 1.51995
    else:
        angle = np.nan
    return angle


def _utils_get_Pix2D(D1=0., D2=0., center=False, geom=_GEOM):

    gridH = geom['pix']['sizeH']*np.arange(0, geom['pix']['nbH'])
    gridV = geom['pix']['sizeV']*np.arange(0, geom['pix']['nbV'])
    GH = np.tile(gridH, geom['pix']['nbV'])
    GV = np.repeat(gridV, geom['pix']['nbH'])
    mH = np.mean(gridH) if center else 0.
    mV = np.mean(gridV) if center else 0.
    pts2D = np.array([D1+GH-mH, D2+GV-mV])
    return pts2D


def _get_indtlim(t, tlim=None, shot=None, out=bool):
    C0 = tlim is None
    C1 = type(tlim) in [list, tuple, np.ndarray]
    assert C0 or C1
    assert type(t) is np.ndarray

    if C0:
        tlim = [-np.inf, np.inf]
    else:
        assert len(tlim) == 2
        ls = [str, int, float, np.int64, np.float64]
        assert all([tt is None or type(tt) in ls for tt in tlim])
        tlim = list(tlim)
        for (ii, sgn) in [(0, -1.), (1, 1.)]:
            if tlim[ii] is None:
                tlim[ii] = sgn*np.inf
            elif type(tlim[ii]) is str and 'ign' in tlim[ii].lower():
                tlim[ii] = get_t0(shot)

    assert tlim[0] < tlim[1]
    indt = (t >= tlim[0]) & (t <= tlim[1])
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

    assert tmode in ['mean', 'start', 'end']

    # Pre-format input
    if path is None:
        path = os.path.abspath(tempdir)
    rootstr = 'XICS {0:05.0f}:'.format(shot)

    # Load and unzip temporary file
    if Verb:
        msg = '(1/4) ' + rootstr + ' loading and unziping files...'
        print(msg)
    targetf = os.path.join(path, 'xics_{0:05.0f}.zip'.format(shot))
    targetd = os.path.join(path, 'xics_{0:05.0f}/'.format(shot))
    out = pw.TSRfile(shot, 'FXICS_MIDDLE', targetf)

    if not out == 0:
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
    t0 = 0.     # Because startAcq on topOrigin (tIGNOTRON - 32 s)
    NExp = pw.TSRqParm(shot, 'DXICS', 'PIL_N', 'PIL_NMax', 1)[0][0][0]
    TExpT = pw.TSRqParm(shot, 'DXICS', 'PIL_Times', 'PIL_TExpT', 1)[0][0][0]
    TExpP = pw.TSRqParm(shot, 'DXICS', 'PIL_Times', 'PIL_TExpP', 1)[0][0][0]
    TDelay = pw.TSRqParm(shot, 'DXICS', 'PIL_Times', 'PIL_TDelay', 1)[0][0][0]
    # Delay not taken into account in this acquisition mode
    if TDelay >= 50:
        # TDelay now in ms
        TDelay *= 1.e-3
    try:
        THR = pw.TSRqParm(shot, 'DXICS', 'PIL_THR', 'THR', 1)[0][0][0]
    except Exception as err:
        THR = _get_THR(shot)
    try:
        Ang = pw.TSRqParm(shot, 'DXICS', 'CRYST', 'Ang', 1)[0][0][0]
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
    if nIm > NExp:
        msg = "The zip file contains more images than parameter NExp !"
        raise Exception(msg)

    # Build time vector (parameter Delay is only for external trigger !!!)
    Dt = t0 + TExpP*np.arange(0, nIm) + np.array([[0.], [TExpT]])
    if shot >= 54132:
        # Previously, TDelay had no effect
        # From 54132, TDelay is fed to a home-made QtTimer in:
        #    controller_acquisitions.cpp:168
        #    controller_pilotage.cpp:126
        Dt += TDelay
    if tmode == 'mean':
        t = np.mean(Dt, axis=0)
    elif tmode == 'start':
        t = Dt[0, :]
    else:
        t = Dt[1, :]
    indt = _get_indtlim(t, tlim=tlim, out=int)
    if indt.size == 0:
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
        for ii in range(0, nt):
            im = Image.open(os.path.join(targetd, lf[ii]))
            s = str(im.tag.tagdata[270]).split('#')[1:]
            s = [ss[:ss.index('\\r')] for ss in s if '\\r' in ss]
            ls.append(s)
            data[ii, :, :] = np.flipud(np.asarray(im, dtype=np.int32))
    finally:
        # Delete temporary files
        if Verb:
            msg = '(4/4) ' + rootstr + ' Deleting temporary files...'
            print(msg)
        os.remove(targetf)
        shutil.rmtree(targetd)

    dunits = r'photons'
    dbonus = {
        'Dt': Dt, 'dt': TExpT, 'THR': THR, 'mask': mask,
        'NExp': NExp, 'nIm': nIm,
        'TExpT': TExpT, 'TExpP': TExpP, 'TDelay': TDelay,
        'nH': geom['pix']['nbH'], 'nV': geom['pix']['nbV'],
    }
    return data, t, dbonus


# #############################################################################
#                   Function to plot results
# #############################################################################


_NI, _NJ = 487, 1467
_XI = (np.arange(0, _NI)-(_NI-1)/2.)*172e-6
_XJ = (np.arange(0, _NJ)-(_NJ-1)/2.)*172e-6
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
        cryst = crystobj

    if isinstance(cryst, str) and os.path.isfile(cryst):
        cryst = tf.load(cryst)
    elif cryst is not False:
        assert cryst.__class__.__name__ == 'CrystalBragg'

    if cryst is not False:
        if det is False:
            det = cryst.get_detector_approx()
        c0 = (isinstance(det, dict)
              and all([kk in det.keys() for kk in ['cent', 'nout',
                                                   'ei', 'ej']]))
        if not c0:
            msg = ("det must be a dict with keys:\n"
                   + "\t- cent: [x,y,z] of the detector center\n"
                   + "\t- nout: [x,y,z] of unit vector normal to plane\n"
                   + "\t- ei: [x,y,z] of unit vector ei\n"
                   + "\t- ej: [x,y,z] of unit vector ej = nout x ei\n"
                   + "\n\t- provided: {}".format(det))
            raise Exception(msg)
    return cryst, det


def _extract_data(pfe, allow_pickle=None,
                  maskxi=None, shot=None, indt=None, indxj=None):
    # Prepare data
    out = dict(np.load(pfe, allow_pickle=allow_pickle))
    t, ang = [out[kk] for kk in ['t', 'ang']]
    spect, shots, thr = [out[kk] for kk in ['spect', 'shots', 'thr']]
    xi = out.get('xi', _XI)
    xj = out.get('xj', _XJ)

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
         dlines=None, indt=None, indxj=None,
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
                                                              maskxi, shot,
                                                              indt=indt,
                                                              indxj=indxj)
    nshot, nt, nxj, nxi = spect.shape
    iout = np.any(np.nanmean(spectn**2, axis=-1) > 0.1, axis=-1)

    # Group by angle
    angu = np.unique(ang)
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
            cryst.move(param=angu[jj]*np.pi/180.)

            bragg, phii = cryst.calc_phibragg_from_xixj(
                xif, xjf, n=1,
                dtheta=None, psi=None, plot=False, det=det)
            phi[ind, ...] = phii[None, ...]
            lamb[ind, ...] = cryst.get_lamb_from_bragg(bragg, n=1)[None, ...]

    isortxj = nxj - 1 - np.argsort(xj)

    # -------------
    # Plot 1

    if fs is None:
        fs = (18, 9)
    if cmap is None:
        cmap = plt.cm.viridis
    if dmargin is None:
        dmargin = {'left': 0.05, 'right': 0.99,
                   'bottom': 0.06, 'top': 0.93,
                   'wspace': 0.3, 'hspace': 0.2}

    fig = plt.figure(figsize=fs)
    if shot is not None:
        fig.suptitle('shot = {}'.format(shot))
    gs = gridspec.GridSpec(nxj, 2, **dmargin)

    dax = {'spect': [None for ii in range(nxj)],
           'spectn': [None for ii in range(nxj)]}

    shx = None
    for ii in range(nxj):
        iax = isortxj[ii]
        dax['spect'][ii] = fig.add_subplot(gs[iax, 0], sharex=shx)
        if ii == 0:
            shx = dax['spect'][0]
        dax['spectn'][ii] = fig.add_subplot(gs[iax, 1], sharex=shx)
        dax['spect'][ii].set_ylabel('xj = {}\ndata (a.u.)'.format(xj[ii]))

        for jj in range(nang):
            col = lcol[jj % ncol]
            lab0 = 'ang {}'.format(angu[jj])
            ind = (ang == angu[jj]).nonzero()[0]
            xibis = xi  # + angu[jj]*0.05
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
    hand = [
        mlines.Line2D([], [], c=lcol[jj % ncol], ls='-')
        for jj in range(nang)
    ]
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
        iax = isortxj[ii]
        dax2['spect'][ii] = fig.add_subplot(gs[iax, 0],
                                            sharex=shx)
        if ii == 0:
            shx, shy = dax2['spect'][0], dax2['spect'][0]
        dax2['spectn'][ii] = fig.add_subplot(gs[iax, 1],
                                             sharex=shx)
        dax2['spect'][ii].set_ylabel('data (a.u.)'.format(xj[ii]))

        for jj in range(nang):
            col = lcol[jj % ncol]
            lab0 = 'ang {}'.format(angu[jj])
            ind = (ang == angu[jj]).nonzero()[0]
            xibis = xi  # + angu[jj]*0.05
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
        if dlines is not None:
            for kk in dlines.keys():
                dax2['spect'][ii].axvline(dlines[kk]['lambda'],
                                          c='k', ls='-', lw=1.)
                dax2['spectn'][ii].axvline(dlines[kk]['lambda'],
                                           c='k', ls='-', lw=1.)
    if dlines is not None:
        for kk in dlines.keys():
            dax2['spect'][0].annotate(kk,
                                      xy=(dlines[kk]['lambda'], 1.),
                                      xycoords=('data', 'axes fraction'),
                                      horizontalalignment='left',
                                      verticalalignment='bottom',
                                      rotation=45,
                                      arrowprops=None)

    # Polish
    dax2['spect'][0].set_title('raw spectra')
    dax2['spectn'][0].set_title('normalized spectra')
    dax2['spect'][-1].set_xlabel(r'$\lambda$' + ' (m)')
    dax2['spectn'][-1].set_xlabel(r'$\lambda$' + ' (m)')
    hand = [
        mlines.Line2D([], [], c=lcol[jj % ncol], ls='-')
        for jj in range(nang)
    ]
    lab = ['{}'.format(aa) for aa in angu]
    dax2['spect'][0].legend(hand, lab,
                            title='Table angle (deg.)',
                            loc='upper left',
                            bbox_to_anchor=(1.01, 1.))
    return dax, dax2


# #############################################################################
#                   Fit several data for one det
# #############################################################################


def _get_dinput_key01(dinput=None,
                      key0=None, key1=None, indl0=None, indl1=None,
                      dlines=None, dconstraints=None,
                      lambmin=None, lambmax=None,
                      same_spectrum=None, nspect=None, dlamb=None):
    lc = [all([aa is not None for aa in [dinput, key0, key1, indl0, indl1]]),
          all([aa is not None for aa in [dlines, dconstraints]])]
    if np.sum(lc) != 1:
        msg = ("Please provide either (xor):\n"
               + "\t- dinput, key0, key1, indl0, indl1\n"
               + "\t- dlines, dconstraints")
        raise Exception(msg)
    if lc[1]:
        dinput = tf.data._spectrafit2d.multigausfit1d_from_dlines_dinput(
            dlines=dlines,
            dconstraints=dconstraints,
            lambmin=lambmin, lambmax=lambmax,
            same_spectrum=same_spectrum, nspect=nspect, dlamb=dlamb)
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
    return dinput, key0, key1, indl0, indl1


def fit(pfe=None, allow_pickle=True,
        spectn=None, shots=None, t=None, ang=None, xi=None, xj=None, thr=None,
        shot=None, indt=None, indxj=None, maskxi=None,
        cryst=None, det=None,
        dlines=None, dconstraints=None, dx0=None,
        key0=None, key1=None,
        dinput=None, indl0=None, indl1=None,
        lambmin=None, lambmax=None,
        same_spectrum=None, dlamb=None,
        method=None, max_nfev=None,
        dscales=None, x0_scale=None, bounds_scale=None,
        xtol=None, ftol=None, gtol=None,
        loss=None, verbose=None, plot=None,
        fs=None, dmargin=None, cmap=None, warn=True):

    # -----------
    # Check input

    if verbose is None or verbose is True:
        verbose = 1
    if verbose is False:
        verbose = 0

    # input data file
    lc = [pfe is not None,
          all([aa is not None for aa in [spectn, shots, t, ang, xi, xj]]),
          (shot is not None
           and all([aa is None for aa in [pfe, spectn, shots, t, ang]]))]
    if np.sum(lc) != 1:
        msg = ("Please provide eithe (xor):\n"
               + "\t- pfe\n"
               + "\t- spectn, shots, t, ang, xi, xj\n"
               + "\t- shot, xi, xj (loaded from ARCADE)")
        raise Exception(msg)
    if lc[0]:
        if not os.path.isfile(pfe):
            msg = ("Provided file does not exist!\n"
                   + "\t- provided: {}".format(pfe))
            raise Exception(msg)

        # subset of shots
        if shot is not None:
            if not hasattr(shot, '__iter__'):
                shot = np.array([shot], dtype=int)
            else:
                shot = np.r_[shot].astype(int)

        if maskxi is None:
            maskxi = _MASKXI

        # extract data
        spectn, shots, t, ang, xi, xj, thr = _extract_data(pfe,
                                                           allow_pickle,
                                                           maskxi, shot,
                                                           indt, indxj)[1:]
    elif lc[2]:
        data, t, dbonus = _load_data(int(shot),
                                     tlim=_DSHOT[shot]['tlim'],
                                     tmode='mean',
                                     path=None, Brightness=None,
                                     mask=True, Verb=False)
        pass

    # Cryst part
    cryst, det = _get_crystanddet(cryst=cryst, det=det)
    assert cryst is not False

    nshot, nt, nxj, nxi = spectn.shape
    iout = np.any(np.nanmean(spectn**2, axis=-1) > 0.1, axis=-1)

    # Group by angle
    angu, ind_ang = np.unique(ang, return_inverse=True)
    nang = angu.size
    lcol = ['r', 'g', 'b', 'k', 'm', 'c', 'y']
    ncol = len(lcol)

    # -----------
    # Convert xi, xj to lamb, phi

    # Cryst data
    lamb = np.full((nang, nxj, nxi), np.nan)
    phi = np.full((nang, nxj, nxi), np.nan)
    xif = np.tile(xi, (nxj, 1))
    xjf = np.repeat(xj[:, None], nxi, axis=1)
    for jj in range(nang):
        # Beware to provide angles in rad !
        cryst.move(param=angu[jj]*np.pi/180.)

        bragg, phii = cryst.calc_phibragg_from_xixj(
            xif, xjf, n=1,
            dtheta=None, psi=None, plot=False, det=det)
        phi[jj, ...] = phii
        lamb[jj, ...] = cryst.get_lamb_from_bragg(bragg, n=1)

    # Reorder to sort lamb
    assert np.all(np.argsort(lamb, axis=-1)
                  == np.arange(nxi-1, -1, -1)[None, None, :])
    xi = xi[::-1]
    lamb = lamb[:, :, ::-1]
    phi = phi[:, :, ::-1]
    spectn = spectn[:, :, :, ::-1]
    lambminpershot = np.min(np.nanmin(lamb, axis=-1), axis=-1)
    lambmaxpershot = np.max(np.nanmax(lamb, axis=-1), axis=-1)
    dshiftmin = 0.02*(lambmaxpershot - lambminpershot) / lambmaxpershot

    # -----------
    # Get dinput for 1d fitting
    if dlamb is None:
        dlamb = 2.*(np.nanmax(lamb) - np.nanmin(lamb))
    dinput, key0, key1, indl0, indl1 = _get_dinput_key01(
        dinput=dinput, key0=key0, key1=key1, indl0=indl0, indl1=indl1,
        dlines=dlines, dconstraints=dconstraints,
        lambmin=lambmin, lambmax=lambmax,
        same_spectrum=same_spectrum, nspect=spectn.shape[1], dlamb=dlamb)

    # -----------
    # Optimize

    # Fit
    spectfit = np.full(spectn.shape, np.nan)
    time = np.full(spectn.shape[:-1], np.nan)
    chinorm = np.full(spectn.shape[:-1], np.nan)
    if key0 is not None:
        shift0 = np.full(spectn.shape[:-1], np.nan)
    if key1 is not None:
        shift1 = np.full(spectn.shape[:-1], np.nan)
    for jj in range(nang):
        if verbose > 0:
            msg = ("\nOptimizing for ang = {}  ({}/{})\n".format(angu[jj],
                                                                 jj+1, nang)
                   + "--------------------------------")
            print(msg)
        ind = (ang == angu[jj]).nonzero()[0]
        for ll in range(ind.size):
            msgsh = "---------- shot {} ({}/{})".format(shots[ind[ll]],
                                                        ll+1, ind.size)
            for ii in range(nxj):
                if verbose > 0:
                    msg = ("  xj = {}  ({}/{}):".format(xj[ii], ii+1, nxj)
                           + "\t{} spectra".format(spectn.shape[1]))
                    print(msgsh + msg)
                dfit1d = tf.data._spectrafit2d.multigausfit1d_from_dlines(
                    spectn[ind[ll], :, ii, :],
                    lamb[jj, ii, :],
                    dinput=dinput, dx0=dx0,
                    lambmin=lambmin, lambmax=lambmax,
                    dscales=dscales,
                    x0_scale=x0_scale,
                    bounds_scale=bounds_scale,
                    method=method, max_nfev=max_nfev,
                    chain=True, verbose=verbose,
                    xtol=xtol, ftol=ftol, gtol=gtol, loss=loss,
                    ratio=None, jac='call',
                )
                spectfit[ind[ll], :, ii, :] = dfit1d['sol']
                time[ind[ll], :, ii] = dfit1d['time']
                chinorm[ind[ll], :, ii] = np.sqrt(dfit1d['cost']) / nxi
                indsig = np.abs(dfit1d['dshift']) >= dshiftmin[jj]
                indpos = dfit1d['dshift'] > 0.
                ind098 = indsig & indpos & (dfit1d['dratio'] > 0.99)
                ind102 = indsig & (~indpos) & (dfit1d['dratio'] < 1.01)
                if np.any(ind098) and warn is True:
                    msg = ("Some to high (> 0.98) dratio with dshift > 0:\n"
                           + "\t- shot: {}\n".format(shots[ind[ll]])
                           + "\t- xj[{}] = {}\n".format(ii, xj[ii])
                           + "\t- shitmin = {}\n".format(dshiftmin[jj])
                           + "\t- dshift[{}]".format(ind098.nonzero()[0])
                           + " = {}\n".format(dfit1d['dshift'][ind098])
                           + "\t- dratio[{}]".format(ind098.nonzero()[0])
                           + " = {}".format(dfit1d['dratio'][ind098]))
                    warnings.warn(msg)
                if np.any(ind102) and warn is True:
                    msg = ("Some to high dratio with dshift > 0:\n"
                           + "\t- shitmin = {}\n".format(dshiftmin[jj])
                           + "\t- dshift[{}]".format(ind102.nonzero()[0])
                           + " = {}\n".format(dfit1d['dshift'][ind102])
                           + "\t- dratio[{}]".format(ind102.nonzero()[0])
                           + " = {}".format(dfit1d['dratio'][ind102]))
                    warnings.warn(msg)
                if key0 is not None or key1 is not None:
                    ineg = dfit1d['dshift'] < 0.
                if key0 is not None:
                    shift0[ind[ll], :, ii] = dfit1d['shift'][:, indl0]
                    shift0[ind[ll], ineg, ii] += (
                        dfit1d['dshift'][ineg]*dinput['lines'][indl0]
                    )
                    shift0[ind[ll], ind098 | ind102, ii] = np.nan
                if key1 is not None:
                    shift1[ind[ll], :, ii] = dfit1d['shift'][:, indl1]
                    shift1[ind[ll], ineg, ii] += (
                        dfit1d['dshift'][ineg]*dinput['lines'][indl1]
                    )
                    shift1[ind[ll], ind098 | ind102, ii] = np.nan

    dcost = {}
    shift0m, shift1m = None, None
    if key0 is not None:
        dcost[key0] = {
            'shift': shift0,
            'shiftm': np.array([[np.nanmean(shift0[ang == angu[jj], :, ii])
                                 for jj in range(nang)] for ii in range(nxj)])}
    if key1 is not None:
        dcost[key1] = {
            'shift': shift1,
            'shiftm': np.array([[np.nanmean(shift1[ang == angu[jj], :, ii])
                                 for jj in range(nang)] for ii in range(nxj)])}

    shiftabs = 0.
    if key0 is not None:
        shiftabs = max(np.nanmax(np.abs(shift0)), shiftabs)
    if key1 is not None:
        shiftabs = max(np.nanmax(np.abs(shift1)), shiftabs)

    # -------------
    # Plot
    if plot is False:
        return {'shots': shots, 't': t, 'ang': ang,
                'lamb': lamb, 'phi': phi,
                'spectn': spectn, 'spectfit': spectfit,
                'time': time, 'chinorm': chinorm,
                'dcost': dcost,
                }

    if fs is None:
        fs = (18, 9)
    if cmap is None:
        cmap = plt.cm.viridis
    if dmargin is None:
        dmargin = {'left': 0.06, 'right': 0.99,
                   'bottom': 0.06, 'top': 0.93,
                   'wspace': 0.3, 'hspace': 0.2}
    extent = (0.5, nshot+0.5, -0.5, nt-0.5)
    tmin, tmax = np.nanmin(time), np.nanmax(time)
    chimin, chimax = np.nanmin(chinorm), np.nanmax(chinorm)

    fig = plt.figure(figsize=fs)
    if shot is not None:
        fig.suptitle('shot = {}'.format(shot))
    gs = gridspec.GridSpec(nxj*2, 7, **dmargin)

    dax = {
        'spectn': [None for ii in range(nxj)],
        'time': [None for ii in range(nxj)],
        'chinorm': [None for ii in range(nxj)],
        'shift0': [None for ii in range(nxj)],
        'shift1': [None for ii in range(nxj)],
        'shift0_z': [None for ii in range(nxj)],
        'shift1_z': [None for ii in range(nxj)],
    }

    xones = np.zeros((nt,))
    isortxj = nxj - 1 - np.argsort(xj)
    shx0, shx1, shy1, shx20, shx21 = None, None, None, None, None
    for ii in range(nxj):
        iax = isortxj[ii]
        dax['spectn'][ii] = fig.add_subplot(gs[iax*2:iax*2+2, :3], sharex=shx0)
        if ii == 0:
            shx0 = dax['spectn'][ii]
        dax['time'][ii] = fig.add_subplot(
            gs[iax*2:iax*2+2, 3],
            sharex=shx1, sharey=shy1,
        )
        if ii == 0:
            shx1 = dax['time'][ii]
            shy1 = dax['time'][ii]
        dax['chinorm'][ii] = fig.add_subplot(
            gs[iax*2:iax*2+2, 4], sharex=shx1, sharey=shy1,
        )
        dax['shift0'][ii] = fig.add_subplot(
            gs[iax*2, 5], sharex=shx1, sharey=shy1,
        )
        dax['shift1'][ii] = fig.add_subplot(
            gs[iax*2, 6], sharex=shx1, sharey=shy1,
        )
        dax['shift0_z'][ii] = fig.add_subplot(
            gs[iax*2+1, 5], sharex=shx20,
        )
        dax['shift1_z'][ii] = fig.add_subplot(
            gs[iax*2+1, 6], sharex=shx21,
        )
        if ii == 0:
            shx20 = dax['shift0_z'][ii]
            shx21 = dax['shift1_z'][ii]
        dax['spectn'][ii].set_ylabel('xj = {}\ndata (a.u.)'.format(xj[ii]))
        if iax != nxj-1:
            plt.setp(dax['time'][ii].get_xticklabels(), visible=False)
            plt.setp(dax['chinorm'][ii].get_xticklabels(), visible=False)

        for jj in range(nang):
            col = lcol[jj % ncol]
            ind = (ang == angu[jj]).nonzero()[0]
            for ll in range(ind.size):
                dax['spectn'][ii].plot(
                    lamb[jj, ii, :],
                    spectn[ind[ll], :, ii, :].T,
                    ls='None', marker='.', ms=4., c=col,
                )
                dax['spectn'][ii].plot(
                    lamb[jj, ii, :],
                    spectfit[ind[ll], :, ii, :].T,
                    ls='-', lw=1, c=col,
                )
                if key0 is not None:
                    dax['shift0_z'][ii].plot((dinput['lines'][indl0]
                                              + shift0[ind[ll], :, ii]),
                                             xones,
                                             marker='.', ls='None', c=col)
                if key1 is not None:
                    dax['shift1_z'][ii].plot((dinput['lines'][indl1]
                                              + shift1[ind[ll], :, ii]),
                                             xones,
                                             marker='.', ls='None', c=col)
        dax['time'][ii].imshow(time[:, :, ii].T,
                               extent=extent, cmap=cmap,
                               vmin=tmin, vmax=tmax,
                               interpolation='nearest', origin='lower')
        dax['chinorm'][ii].imshow(chinorm[:, :, ii].T,
                                  extent=extent, cmap=cmap,
                                  vmin=chimin, vmax=chimax,
                                  interpolation='nearest', origin='lower')
        if key0 is not None:
            dax['shift0'][ii].imshow(shift0[:, :, ii].T,
                                     extent=extent, cmap=plt.cm.seismic,
                                     interpolation='nearest', origin='lower',
                                     vmin=-shiftabs, vmax=shiftabs)
            dax['spectn'][ii].axvline(dinput['lines'][indl0],
                                      c='k', ls='-', lw=1.)
            dax['shift0_z'][ii].axvline(dinput['lines'][indl0],
                                        c='k', ls='-', lw=1.)
        if key1 is not None:
            dax['shift1'][ii].imshow(shift1[:, :, ii].T,
                                     extent=extent, cmap=plt.cm.seismic,
                                     interpolation='nearest', origin='lower',
                                     vmin=-shiftabs, vmax=shiftabs)
            dax['spectn'][ii].axvline(dinput['lines'][indl1],
                                      c='k', ls='-', lw=1.)
            dax['shift1_z'][ii].axvline(dinput['lines'][indl1],
                                        c='k', ls='-', lw=1.)
    # Polish
    i0 = (isortxj == 0).nonzero()[0][0]
    i1 = (isortxj == nxj-1).nonzero()[0][0]
    xlab = ['{}'.format(ss) for ss in shots]
    dax['time'][i0].set_title(r'time')
    dax['chinorm'][i0].set_title(r'$\chi_{norm}$')
    dax['shift0'][i0].set_title('shift {}'.format(key0))
    dax['shift1'][i0].set_title('shift {}'.format(key1))
    dax['time'][i1].set_xticks(range(1, nshot+1))
    dax['time'][i1].set_xticklabels(xlab, rotation=75)
    dax['chinorm'][i1].set_xticklabels(xlab, rotation=75)
    dax['time'][i1].set_yticks(range(0, nt))
    if indt is not None:
        dax['time'][i1].set_yticklabels(indt)

    # -------- Extra plot ----
    if nang == 1 or (key0 is None and key1 is None):
        return dax

    lkey = [kk for kk in [key0, key1] if kk is not None]
    nl = len(lkey)

    dmargin = {'left': 0.06, 'right': 0.95,
               'bottom': 0.08, 'top': 0.90,
               'wspace': 0.4, 'hspace': 0.2}

    fig = plt.figure(figsize=fs)
    if shot is not None:
        fig.suptitle('shot = {}'.format(shot))
    gs = gridspec.GridSpec(2, nl, **dmargin)

    shx0, shx1, shy = None, None, None
    ax0 = [None for ii in range(nl)]
    ax1 = [None for ii in range(nl)]
    for ii in range(nl):
        ax0[ii] = fig.add_subplot(gs[0, ii], sharex=shx0, sharey=shy)
        if ii == 0:
            shx0, shy = ax0[ii], ax0[ii]
        ax1[ii] = fig.add_subplot(gs[1, ii], sharex=shx1, sharey=shy)
        if ii == 0:
            shx1 = ax1[ii]
        ax0[ii].set_title(lkey[ii])
        ax0[ii].set_xlabel('table angle')
        ax0[ii].set_ylabel(r'$\Delta \lambda$ (m)')
        ax1[ii].set_xlabel('xi (m)')
        for jj in range(nxj):
            ax0[ii].plot(angu, dcost[lkey[ii]]['shiftm'][jj, :],
                         ls='-', lw=1., marker='.', ms=8,
                         label='xj[{}] = {} m'.format(jj, xj[jj]))
        ax0[ii].axhline(0, ls='--', lw=1., c='k')
        for jj in range(nang):
            ax1[ii].plot(xj, dcost[lkey[ii]]['shiftm'][:, jj],
                         ls='-', lw=1., marker='.', ms=8,
                         label='ang[{}] = {}'.format(jj, angu[jj]))
        ax1[ii].axhline(0, ls='--', lw=1., c='k')

    ax0[0].legend(loc='upper left', bbox_to_anchor=(1.01, 1.))
    ax1[0].legend(loc='upper left', bbox_to_anchor=(1.01, 1.))
    return dax


# #############################################################################
#                   Scan det
# #############################################################################


_DX = [0.01, 0.002, 0.002]      # 0.0004
_DROT = [0.01, 0.01, 0.01]      # 0.0004
_NDX = 2
_NDROT = 2


def _check_orthonormal(cent, nout, ei, ej, ndx, ndrot, msg=''):
    shape = (3, 2*ndrot+1, 2*ndrot+1, 2*ndrot+1)
    lc = [
        cent.shape == (3, 2*ndx+1, 2*ndx+1, 2*ndx+1),
        nout.shape == ei.shape == ej.shape == shape,
        np.allclose(np.sum(nout**2, axis=0), 1.),
        np.allclose(np.sum(ei**2, axis=0), 1.),
        np.allclose(np.sum(ej**2, axis=0), 1.),
        np.allclose(np.sum(nout*ei, axis=0), 0.),
        np.allclose(np.sum(nout*ej, axis=0), 0.),
    ]
    if not all(lc):
        msg = ("Non-conform set of detector parameters! " + msg)
        raise Exception(msg)


def scan_det(pfe=None, allow_pickle=True,
             ndx=None, dx=None, ndrot=None, drot=None,
             spectn=None, shots=None, t=None, ang=None,
             xi=None, xj=None, thr=None,
             dinput=None, indl0=None, indl1=None,
             shot=None, indt=None, indxj=None, maskxi=None,
             cryst=None, det=None,
             dlines=None, dconstraints=None, dx0=None,
             key0=None, key1=None,
             lambmin=None, lambmax=None,
             same_spectrum=None, dlamb=None,
             method=None, max_nfev=None,
             dscales=None, x0_scale=None, bounds_scale=None,
             xtol=None, ftol=None, gtol=None,
             loss=None, verbose=None, plot=None,
             fs=None, dmargin=None, cmap=None,
             save=None, pfe_out=None):

    # Check input
    if verbose is None:
        verbose = 1
    if dx is None:
        dx = _DX
    if drot is None:
        drot = _DROT
    if ndx is None:
        ndx = _NDX
    if ndrot is None:
        ndrot = _NDROT
    if plot is None:
        plot = True
    if save is None:
        save = isinstance(pfe_out, str)

    if not (hasattr(dx, '__iter__') and len(dx) == 3):
        dx = [dx, dx, dx]
    if not (hasattr(drot, '__iter__') and len(drot) == 3):
        drot = [drot, drot, drot]

    if dconstraints is None:
        dconstraints = {
            'double': True,
            'symmetry': False,
            'width': {
                'wxyzkj': [
                    'ArXVII_w_Bruhns', 'ArXVII_z_Amaro',
                    'ArXVII_x_Adhoc200408', 'ArXVII_y_Adhoc200408',
                    'ArXVI_k_Adhoc200408', 'ArXVI_j_Adhoc200408',
                    'ArXVI_q_Adhoc200408', 'ArXVI_r_Adhoc200408',
                    'ArXVI_a_Adhoc200408',
                ],
            },
            'amp': {
                'ArXVI_k_Adhoc200408': {'key': 'kj'},
                'ArXVI_j_Adhoc200408': {'key': 'kj', 'coef': 1.3576},
            },
            'shift': {'wz': ['ArXVII_w_Bruhns', 'ArXVII_z_Amaro']},
        }

    # input data file
    lc = [pfe is not None,
          all([aa is not None for aa in [spectn, shots, t, ang, xi, xj]])]
    if np.sum(lc) != 1:
        msg = ("Please provide eithe (xor):\n"
               + "\t- pfe\n"
               + "\t- spectn, shots, t, ang, xi, xj")
        raise Exception(msg)
    if lc[0]:
        if not os.path.isfile(pfe):
            msg = ("Provided file does not exist!\n"
                   + "\t- provided: {}".format(pfe))
            raise Exception(msg)

        # subset of shots
        if shot is not None:
            if not hasattr(shot, '__iter__'):
                shot = np.array([shot], dtype=int)
            else:
                shot = np.r_[shot].astype(int)

        if maskxi is None:
            maskxi = _MASKXI

        # extract data
        spectn, shots, t, ang, xi, xj, thr = _extract_data(pfe,
                                                           allow_pickle,
                                                           maskxi, shot,
                                                           indt, indxj)[1:]
    # Group by angle
    angu = np.unique(ang)
    nang = angu.size
    lcol = ['r', 'g', 'b', 'k', 'm', 'c', 'y']
    ncol = len(lcol)

    nshot, nt, nxj, nxi = spectn.shape

    # Cryst part
    cryst, det = _get_crystanddet(cryst=cryst, det=det)
    assert cryst is not False

    # Get dinput for 1d fitting
    dinput, key0, key1, indl0, indl1 = _get_dinput_key01(
        dinput=dinput, key0=key0, key1=key1, indl0=indl0, indl1=indl1,
        dlines=dlines, dconstraints=dconstraints,
        lambmin=lambmin, lambmax=lambmax,
        same_spectrum=same_spectrum, nspect=spectn.shape[0], dlamb=dlamb)

    # --------
    # Prepare
    dxv = np.linspace(-ndx, ndx, 2*ndx+1)
    drotv = np.linspace(-ndrot, ndrot, 2*ndrot+1)

    cent = det['cent'][:, None, None, None]
    eout = det['nout'][:, None, None, None]
    e1 = det['ei'][:, None, None, None]
    e2 = det['ej'][:, None, None, None]

    # x and rot
    x0 = dxv[None, :, None, None]*dx[0]
    x1 = dxv[None, None, :, None]*dx[1]
    x2 = dxv[None, None, None, :]*dx[2]
    rot0 = drotv[None, :, None, None]*drot[0]
    rot1 = drotv[None, None, :, None]*drot[1]
    rot2 = drotv[None, None, None, :]*drot[2]

    # Cent and vect
    cent = cent + x0*eout + x1*e1 + x2*e2
    nout = (np.sin(rot0)*e2
            + np.cos(rot0)*(eout*np.cos(rot1) + e1*np.sin(rot1)))
    nout = np.repeat(nout, 2*ndrot+1, axis=-1)
    ei = np.repeat(np.repeat(
        np.cos(rot1)*e1 - np.sin(rot1)*eout, 2*ndrot+1, axis=1),
        2*ndrot+1, axis=-1)
    ej = np.array([nout[1, ...]*ei[2, ...] - nout[2, ...]*ei[1, ...],
                   nout[2, ...]*ei[0, ...] - nout[0, ...]*ei[2, ...],
                   nout[0, ...]*ei[1, ...] - nout[1, ...]*ei[0, ...]])
    _check_orthonormal(cent, nout, ei, ej, ndx, ndrot, '1')
    ei = np.cos(rot2)*ei + np.sin(rot2)*ej
    ej = np.array([nout[1, ...]*ei[2, ...] - nout[2, ...]*ei[1, ...],
                   nout[2, ...]*ei[0, ...] - nout[0, ...]*ei[2, ...],
                   nout[0, ...]*ei[1, ...] - nout[1, ...]*ei[0, ...]])
    _check_orthonormal(cent, nout, ei, ej, ndx, ndrot, '2')

    def func_msg(ndx, ndrot, i0, i1, i2, j0, j1, j2):
        nx = 2*ndx + 1
        nrot = 2*ndrot + 1
        msg = ("-"*10 + "\n"
               + "ii = {1}/{0} {2}/{0} {3}/{0}\t".format(nx, i0+1, i1+1, i2+1)
               + "jj = {1}/{0} {2}/{0} {3}/{0}".format(nrot, j0+1, j1+1, j2+1)
               + "...\t")
        return msg

    # --------------
    # Iterate around reference
    x0_scale = None
    func = tf.data._spectrafit2d.multigausfit1d_from_dlines
    shape = tuple(np.r_[[2*ndx+1]*3 + [2*ndrot+1]*3 + [nxj, nang]])
    time = np.full(shape, np.nan)
    done = np.array([np.zeros((6,)),
                     [2*ndx+1, 2*ndx+1, 2*ndx+1,
                      2*ndrot+1, 2*ndrot+1, 2*ndrot+1]])
    dcost = {kk: {'detail': np.full(shape, np.nan),
                  'chin': np.full(shape[:-2], np.nan)}
             for kk in [key0, key1] if kk is not None}

    # --------------
    # Iterate around reference
    dout = {'dcost': dcost, 'time': time, 'angu': angu, 'xj': xj,
            'dx': dx, 'ndx': ndx, 'drot': drot, 'ndrot': ndrot,
            'x0': x0, 'x1': x1, 'x2': x2,
            'rot0': rot0, 'rot1': rot1, 'rot2': rot2,
            'cent': cent, 'nout': nout, 'ei': ei, 'ej': ej,
            'pfe': pfe, 'shots': shots, 'done': done}

    # --------------
    # Loop
    for i0 in range(x0.size):
        for i1 in range(x1.size):
            for i2 in range(x2.size):
                for j0 in range(rot0.size):
                    for j1 in range(rot1.size):
                        for j2 in range(rot2.size):
                            ind = (i0, i1, i2, j0, j1, j2)
                            if verbose > 0:
                                print(func_msg(ndx, ndrot, *ind),
                                      end='', flush=True, file=sys.stdout)
                            det = {'cent': cent[:, i0, i1, i2],
                                   'nout': nout[:, j0, j1, j2],
                                   'ei': ei[:, j0, j1, j2],
                                   'ej': ej[:, j0, j1, j2]}
                            try:
                                dfit1d = fit(
                                    spectn=spectn, shots=shots, t=t, ang=ang,
                                    xi=xi, xj=xj, thr=thr,
                                    shot=None, indt=None, indxj=None,
                                    maskxi=None, cryst=cryst, det=det,
                                    dlines=None, dconstraints=None, dx0=None,
                                    key0=key0, key1=key1,
                                    dinput=dinput, indl0=indl0, indl1=indl1,
                                    lambmin=lambmin, lambmax=lambmax,
                                    same_spectrum=same_spectrum, dlamb=dlamb,
                                    method=method, max_nfev=max_nfev,
                                    dscales=dscales, x0_scale=x0_scale,
                                    bounds_scale=bounds_scale,
                                    xtol=xtol, ftol=ftol, gtol=gtol,
                                    loss=None, verbose=0,
                                    warn=False, plot=False)
                                for ii in range(nang):
                                    indi = ang == angu[ii]
                                    dout['time'][ind][:, ii] = (
                                        np.nanmean(np.nanmean(
                                            dfit1d['time'][indi, :, :],
                                            axis=1,
                                        ), axis=0)
                                    )
                                for kk in dfit1d['dcost'].keys():
                                    aa = dfit1d['dcost'][kk]['shiftm']
                                    dout['dcost'][kk]['detail'][ind] = aa
                                    nbok = np.sum(np.sum(
                                        ~np.isnan(aa),
                                        axis=-1,
                                    ), axis=-1)
                                    dout['dcost'][kk]['chin'][ind] = (
                                        np.sqrt(np.nansum(
                                            np.nansum(aa**2, axis=-1),
                                            axis=-1,
                                        )) / nbok
                                    )
                                    dout['done'][0, :] = [i0+1, i1+1, i2+1,
                                                          j0+1, j1+1, j2+1]
                                    print('ok', flush=True, file=sys.stdout)
                            except Exception as err:
                                print('failed: ' + str(err),
                                      flush=True, file=sys.stdout)
                                pass
                        # Save regulary (for long jobs)
                        if save is True:
                            np.savez(pfe_out, **dout)
                            msg = "Saved in :\n\t" + pfe_out
                            print(msg)

    if save is True:
        np.savez(pfe_out, **dout)
        msg = "Saved in:\n\t" + pfe_out
        print(msg)
    return dout


def scan_det_plot(din,
                  nsol=None, yscale='log',
                  fs=None, dmargin=None, cmap=None):

    if isinstance(din, str):
        din = dict(np.load(din, allow_pickle=True))
        din['dcost'] = din['dcost'].tolist()
    if nsol is None:
        nsol = 3

    print('done:\n\t{}\n\t{}'.format(din['done'][0, :], din['done'][1, :]))

    # Prepare
    ndx = din.get('ndx', int((din['x0'].size-1)/2))
    ndrot = din.get('ndrot', int((din['rot0'].size-1)/2))
    dx = din.get('dx', np.nanmean(np.diff(din['x0'], axis=1)))
    drot = din.get('drot', np.nanmean(np.diff(din['rot0'], axis=1)))
    if not (hasattr(dx, '__iter__') and len(dx) == 3):
        dx = np.r_[dx, dx, dx]
    if not (hasattr(drot, '__iter__') and len(drot) == 3):
        drot = np.r_[drot, drot, drot]
    if np.any(np.isnan(dx)):
        ylx = (-0.001, 0.001)
        dxv = np.r_[0]
    else:
        ylx = (ndx+0.5)*np.max(dx)*np.r_[-1, 1]
        dxv = np.linspace(-ndx, ndx, 2*ndx+1)
    if np.any(np.isnan(drot)):
        ylr = (-0.001, 0.001)
        drotv = np.r_[0]
    else:
        ylr = (ndx+0.5)*np.max(drot)*np.r_[-1, 1]
        drotv = np.linspace(-ndrot, ndrot, 2*ndrot+1)

    dind = dict.fromkeys(din['dcost'].keys(), {'chi2d': None,
                                               'ind': None, 'lab': None})
    for kk in dind.keys():
        if 'chin' not in din['dcost'][kk].keys():
            nbok = np.sum(np.sum(~np.isnan(din['dcost'][kk]['detail']),
                                 axis=-1), axis=-1)
            dind[kk]['chinf'] = np.ravel(np.sqrt(np.nansum(np.nansum(
                din['dcost'][kk]['detail']**2, axis=-1), axis=-1)) / nbok)
        else:
            dind[kk]['chinf'] = din['dcost'][kk]['chin'].ravel()
        dind[kk]['ind'] = np.argsort(dind[kk]['chinf'], axis=None)
        [ix0, ix1, ix2,
         irot0, irot1, irot2] = np.unravel_index(
             dind[kk]['ind'], din['dcost'][kk]['chin'].shape)
        dind[kk]['indxrot'] = np.array([ix0, ix1, ix2,
                                        irot0, irot1, irot2])
        dind[kk]['valxrot'] = np.array([
            dxv[ix0]*dx[0], dxv[ix1]*dx[1], dxv[ix2]*dx[2],
            drotv[irot0]*drot[0], drotv[irot1]*drot[1], drotv[irot2]*drot[2]])
    i0 = int((dind[kk]['chinf'].size-1)/2)
    i0bis = (dind[kk]['ind'] == i0).nonzero()[0] + 1

    ldet = [{'cent': din['cent'][:, ix0[ll], ix1[ll], ix2[ll]],
             'nout': din['nout'][:, irot0[ll], irot1[ll], irot2[ll]],
             'ei': din['ei'][:, irot0[ll], irot1[ll], irot2[ll]],
             'ej': din['ej'][:, irot0[ll], irot1[ll], irot2[ll]]}
            for ll in range(nsol)]

    xlbck = np.r_[1, 2, 3][:, None] + 0.3*np.r_[-1, 1, np.nan][None, :]
    xlbckx = np.tile(xlbck.ravel(), 2*ndx+1)
    xlbckrot = np.tile(xlbck.ravel(), 2*ndrot+1)
    ylbckx = dxv[:, None] * dx[None, :]
    ylbckx = np.repeat(ylbckx.ravel(), 3)
    ylbckrot = drotv[:, None] * drot[None, :]
    ylbckrot = np.repeat(ylbckrot.ravel(), 3)

    # --------------
    # Plot

    if fs is None:
        fs = (18, 9)
    if cmap is None:
        cmap = plt.cm.viridis
    if dmargin is None:
        dmargin = {'left': 0.06, 'right': 0.99,
                   'bottom': 0.06, 'top': 0.93,
                   'wspace': 0.3, 'hspace': 0.2}

    fig = plt.figure(figsize=fs)
    if din.get('shots') is not None:
        fig.suptitle('shots = {}'.format(din['shots']))
    gs = gridspec.GridSpec(2, len(dind)*2, **dmargin)

    shx0, shy0, shx1, shy1, shx2, shy2 = None, None, None, None, None, None
    dax = {'best': [None for kk in dind.keys()],
           'map_x': [None for kk in dind.keys()],
           'map_rot': [None for kk in dind.keys()]}
    for ii, kk in enumerate(dind.keys()):
        dax['best'][ii] = fig.add_subplot(gs[0, ii*2:(ii+1)*2],
                                          yscale=yscale,
                                          sharex=shx0, sharey=shy0)
        dax['map_x'][ii] = fig.add_subplot(gs[1, ii*2],
                                           sharex=shx1, sharey=shy1)
        dax['map_rot'][ii] = fig.add_subplot(gs[1, ii*2+1],
                                             sharex=shx2, sharey=shy2)
        if ii == 0:
            shx0, shy0 = dax['best'][ii], dax['best'][ii]
            shx1, shy1 = dax['map_x'][ii], dax['map_x'][ii]
            shx2, shy2 = dax['map_rot'][ii], dax['map_rot'][ii]
        dax['best'][ii].set_title(kk)

        dax['best'][ii].plot(range(nsol+1, dind[kk]['ind'].size + 1),
                             dind[kk]['chinf'][dind[kk]['ind'][nsol:]],
                             c='k', ls='-', lw=1., marker='.', ms=3)
        dax['best'][ii].plot(i0bis,
                             dind[kk]['chinf'][i0],
                             c='k', ls='None', lw=2., marker='x')
        for jj in range(nsol):
            l, = dax['best'][ii].plot(
                jj+1,
                dind[kk]['chinf'][dind[kk]['ind'][jj]],
                ls='None', marker='x', ms=6)
            dax['map_x'][ii].plot(range(1, 4), dind[kk]['valxrot'][:3, jj],
                                  ls='-', marker='o', lw=1., c=l.get_color())
            dax['map_rot'][ii].plot(range(1, 4), dind[kk]['valxrot'][3:, jj],
                                    ls='-', marker='o', lw=1., c=l.get_color())

            dax['map_x'][ii].plot(xlbckx, ylbckx,
                                  ls='-', lw=1., c='k')
            dax['map_rot'][ii].plot(xlbckrot, ylbckrot,
                                    ls='-', lw=1., c='k')
            dax['map_x'][ii].axhline(0., ls='--', lw=1., c='k')
            dax['map_rot'][ii].axhline(0., ls='--', lw=1., c='k')

    dax['best'][0].set_xlim(0, max(50, i0bis))
    dax['map_x'][0].set_xlim(0, 4)
    dax['map_rot'][0].set_xlim(0, 4)
    dax['map_x'][0].set_ylim(*ylx)
    dax['best'][0].set_ylabel(r'$\chi_{norm}$')
    dax['map_x'][0].set_xticks([1, 2, 3])
    # dax['map_x'][0].set_yticks(dxv)
    dax['map_rot'][0].set_xticks([1, 2, 3])
    # dax['map_rot'][0].set_yticks(drotv)
    dax['map_x'][0].set_xticklabels([r'$x_0$', r'$x_1$', r'$x_2$'])
    dax['map_x'][0].set_ylabel(r'$\delta x$')
    dax['map_rot'][0].set_xticklabels([r'$rot_0$', r'$rot_1$', r'$rot_2$'])
    dax['map_rot'][0].set_ylabel(r'$\delta rot$')
    dax['map_rot'][0].set_ylim(*ylr)
    return dax, ldet


# #############################################################################
#                   least_square det
# #############################################################################

def _get_det_from_det0_xscale(x, det0=None, scales=None):
    xs = x*scales
    cent = (det0['cent']
            + xs[0]*det0['nout']
            + xs[1]*det0['ei'] + xs[2]*det0['ei'])
    nout = (np.sin(xs[3])*det0['ej']
            + np.cos(xs[3])*(det0['nout']*np.cos(xs[4])
                             + det0['ei']*np.sin(xs[4])))
    ei = np.cos(xs[4])*det0['ei'] - np.sin(xs[4])*det0['nout']
    ej = np.cross(nout, ei)
    ei = np.cos(xs[5])*ei + np.sin(xs[5])*ej
    ej = np.cross(nout, ei)
    return {'cent': cent, 'nout': nout,
            'ei': ei, 'ej': ej}


def get_func_cost(spectn=None, shots=None, t=None, ang=None,
                  xi=None, xj=None,
                  cryst=None, det0=None,
                  key0=None, key1=None, dinput=None, indl0=None, indl1=None,
                  lambmin=None, lambmax=None, same_spectrum=None, dlamb=None):

    def func_cost(x, scales=None):
        dfit1d = fit(spectn=spectn, shots=shots, t=t, ang=ang,
                     xi=xi, xj=xj,
                     shot=None, indt=None, indxj=None, maskxi=None,
                     det=_get_det_from_det0_xscale(x, det0, scales=scales),
                     cryst=cryst,
                     dlines=None, dconstraints=None, dx0=None,
                     key0=key0, key1=key1,
                     dinput=dinput, indl0=indl0, indl1=indl1,
                     lambmin=lambmin, lambmax=lambmax,
                     same_spectrum=same_spectrum, dlamb=dlamb,
                     method=None, max_nfev=None,
                     dscales=None, x0_scale=None,
                     bounds_scale=None,
                     xtol=None, ftol=None, gtol=None,
                     loss=None, verbose=0, plot=False, warn=False)
        shiftm = np.concatenate([dfit1d['dcost'][kk]['shiftm'].ravel()
                                 for kk in dfit1d['dcost'].keys()])
        return shiftm*1.e13
    return func_cost


def scan_det_least_square(pfe=None, allow_pickle=True,
                          ndx=None, dx=None, ndrot=None, drot=None,
                          spectn=None, shots=None, t=None, ang=None,
                          xi=None, xj=None,
                          dinput=None, indl0=None, indl1=None,
                          shot=None, indt=None, indxj=None, maskxi=None,
                          cryst=None, det=None,
                          dlines=None, dconstraints=None, dx0=None,
                          key0=None, key1=None,
                          lambmin=None, lambmax=None,
                          same_spectrum=None, dlamb=None,
                          method=None, max_nfev=None,
                          dscales=None, x0_scale=None, bounds_scale=None,
                          xtol=None, ftol=None, gtol=None, jac=None,
                          loss=None, verbose=None, plot=None,
                          fs=None, dmargin=None, cmap=None,
                          save=None, pfe_out=None):

    # Check input
    if verbose is None:
        verbose = 2
    if dx is None:
        dx = _DX
    if drot is None:
        drot = _DROT
    if ndx is None:
        ndx = _NDX
    if ndrot is None:
        ndrot = _NDROT
    if method is None:
        method = 'trf'
    if xtol is None:
        xtol = 1.e-6
    if ftol is None:
        ftol = 1.e-6
    if gtol is None:
        gtol = 1.e-6
    if jac is None:
        jac = '3-point'
    if loss is None:
        loss = 'linear'
    if plot is None:
        plot = True
    if save is None:
        save = isinstance(pfe_out, str)

    # input data file
    lc = [pfe is not None,
          all([aa is not None for aa in [spectn, shots, t, ang, xi, xj]])]
    if np.sum(lc) != 1:
        msg = ("Please provide eithe (xor):\n"
               + "\t- pfe\n"
               + "\t- spectn, shots, t, ang, xi, xj")
        raise Exception(msg)
    if lc[0]:
        if not os.path.isfile(pfe):
            msg = ("Provided file does not exist!\n"
                   + "\t- provided: {}".format(pfe))
            raise Exception(msg)

        # subset of shots
        if shot is not None:
            if not hasattr(shot, '__iter__'):
                shot = np.array([shot], dtype=int)
            else:
                shot = np.r_[shot].astype(int)

        if maskxi is None:
            maskxi = _MASKXI

        # extract data
        spectn, shots, t, ang, xi, xj, thr = _extract_data(pfe,
                                                           allow_pickle,
                                                           maskxi, shot,
                                                           indt, indxj)[1:]
    # Group by angle
    angu = np.unique(ang)
    nang = angu.size
    nshot, nt, nxj, nxi = spectn.shape

    # Cryst part
    cryst, det0 = _get_crystanddet(cryst=cryst, det=det)
    assert cryst is not False

    # Get dinput for 1d fitting
    dinput, key0, key1, indl0, indl1 = _get_dinput_key01(
        dinput=dinput, key0=key0, key1=key1, indl0=indl0, indl1=indl1,
        dlines=dlines, dconstraints=dconstraints,
        lambmin=lambmin, lambmax=lambmax,
        same_spectrum=same_spectrum, nspect=spectn.shape[0], dlamb=dlamb)

    # --------
    # Prepare
    scales = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    x0_scale = np.zeros((6,), dtype=float)
    if method == 'lm':
        jac = '2-point'
        bounds_scale = (-np.inf, np.inf)
    else:
        bounds_scale = np.r_[0.10, 0.10, 0.10, 0.10, 0.10, 0.10]/scales
        bounds_scale = (-bounds_scale, bounds_scale)
    diff_step = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]

    func_cost = get_func_cost(spectn=spectn, shots=shots, t=t, ang=ang,
                              xi=xi, xj=xj,
                              cryst=cryst, det0=det0,
                              key0=key0, key1=key1,
                              dinput=dinput, indl0=indl0, indl1=indl1,
                              lambmin=lambmin, lambmax=lambmax,
                              same_spectrum=same_spectrum, dlamb=dlamb)

    # --------
    # Optimize
    t0 = dtm.datetime.now()
    res = scpopt.least_squares(func_cost, x0_scale,
                               jac=jac, bounds=bounds_scale,
                               method=method, ftol=ftol, xtol=xtol,
                               gtol=gtol, x_scale='jac', f_scale=1.0,
                               loss=loss, diff_step=diff_step,
                               tr_solver=None, tr_options={},
                               jac_sparsity=None, max_nfev=max_nfev,
                               verbose=verbose,
                               kwargs={'scales': scales})

    dt = (dtm.datetime.now()-t0).total_seconds()
    msg = ("Ellapsed time: {} min".format(dt/60.))
    print(msg)

    # --------
    # Extract solution
    det = _get_det_from_det0_xscale(res.x, det0, scales=scales)
    return {
        'chin': np.sqrt(res.cost)/(nang*nxj),
        'time': dt,
        'x': res.x, 'det0': det0, 'det': det,
        'nfev': res.nfev, 'xtol': xtol, 'ftol': ftol, 'gtol': gtol,
        'method': method, 'scales': scales,
    }


# #############################################################################
#                   Treat all shots and save
# #############################################################################


def treat(cryst, shot=None,
          dlines=None,
          dconst=None,
          nbsplines=None,
          ratio=None,
          binning=None,
          tol=None,
          plasma=True,
          nameextra=None,
          domain=None,
          xi=_XI, xj=_XJ,
          path=_HERE):

    if nbsplines is None:
        nbsplines = 15
    if ratio is None:
        ratio = {'up': ['ArXVII_w_Bruhns', 'ArXVII_y_Adhoc200408'],
                 'low': ['ArXVII_z_Amaro', 'ArXVII_x_Adhoc200408']}
    if binning is None:
        binning = {'lamb': 487, 'phi': 200}
    if tol is None:
        tol = 1.e-5
    if nameextra is None:
        nameextra = ''
    if nameextra != '' and nameextra[0] != '_':
        nameextra = '_' + nameextra
    if dlines is None:
        from inputs_temp.dlines import dlines
        dlines = {k0: v0 for k0, v0 in dlines.items()
                  if (k0 in ['ArXVII_w_Bruhns', 'ArXVII_z_Amaro']
                      or ('Adhoc200408' in k0))}
    if plasma is True:
        dsig = {
            'ece': {'t': 't', 'data': 'Te0'},
            'interferometer': {'t': 't', 'data': 'ne_integ'},
            'ic_antennas': {'t': 't', 'data': 'power'},
            'lh_antennas': {'t': 't', 'data': 'power'}}

    # Leave double ratio / dshift and x/y free, then plot them to get robust
    # values
    if dconst is None:
        dconst = {
            'double': True,
            'symmetry': False,
            'width': {'wxyzkj':
                      ['ArXVII_w_Bruhns', 'ArXVII_z_Amaro',
                       'ArXVII_x_Adhoc200408', 'ArXVII_y_Adhoc200408',
                       'ArXVI_k_Adhoc200408', 'ArXVI_j_Adhoc200408',
                       'ArXVI_q_Adhoc200408', 'ArXVI_r_Adhoc200408',
                       'ArXVI_a_Adhoc200408']},
            'amp': {'ArXVI_k_Adhoc200408': {'key': 'kj'},
                    'ArXVI_j_Adhoc200408': {'key': 'kj', 'coef': 1.3576}},
            'shift': {'wz': ['ArXVII_w_Bruhns', 'ArXVII_z_Amaro'],
                      'qra': ['ArXVI_q_Adhoc200408', 'ArXVI_r_Adhoc200408',
                              'ArXVI_a_Adhoc200408'],
                      'xy': ['ArXVII_x_Adhoc200408', 'ArXVII_y_Adhoc200408']}}

    # Shots
    dshots = _DSHOTS[cryst]
    shots = np.unique([kk for kk in dshots.keys()
                       if len(dshots[kk]['tlim']) == 2])
    if shot is not None:
        if not hasattr(shot, '__iter__'):
            shot = [shot]
        ind = np.array([shots == ss for ss in shot])
        shots = shots[ind]

    # Cryst part
    det = dict(np.load(os.path.join(_HERE, 'det37_CTVD_incC4.npz')))
    cryst, det = _get_crystanddet(cryst=cryst, det=det)
    assert cryst is not False

    mask = ~np.any(np.load(_MASKPATH)['ind'], axis=0)

    for ii in range(shots.size):
        if len(dshots[int(shots[ii])]['tlim']) != 2:
            continue
        print('\n\nshot {} ({} / {})'.format(shots[ii], ii+1, shots.size))
        try:
            cryst.move(dshots[int(shots[ii])]['ang']*np.pi/180.)

            data, t, dbonus = _load_data(int(shots[ii]),
                                         tlim=dshots[int(shots[ii])]['tlim'])
            dout = cryst.plot_data_fit2d_dlines(
                dlines=dlines, dconstraints=dconst, data=data,
                xi=xi, xj=xj, det=det,
                deg=2, verbose=2, subset=None, binning=binning,
                nbsplines=nbsplines, mask=mask, ratio=ratio,
                domain=domain,
                Ti=True, vi=True,
                chain=True, plot=False,
                xtol=tol, ftol=tol, gtol=tol)
            dout['t'] = t
            dout.update(dbonus)

            # Include info on shot ?
            if plasma is True and shots[ii] > 54178:
                dt = np.nanmean(np.diff(t))
                tbins = np.r_[t[0]-dt/2, 0.5*(t[1:] + t[:-1]), t[-1]+dt/2]
                try:
                    multi = tf.imas2tofu.MultiIDSLoader(
                        ids=list(dsig.keys()),
                        shot=int(shots[ii]), ids_base=False)
                    for k0, v0 in dsig.items():
                        try:
                            if k0 == 'interferometer':
                                indch = [2, 3, 4, 5, 6, 7, 8, 9]
                                out = multi.get_data(k0, list(v0.values()),
                                                     indch=indch)
                                out[v0['data']] = out[v0['data']][7, :]
                            else:
                                out = multi.get_data(k0, list(v0.values()))
                            if k0 == 'lh_power':
                                out['power'] = np.nansum(out['power'], axis=0)
                            key = '{}_{}'.format(k0, v0['data'])
                            indout = (np.isnan(out[v0['data']])
                                      | (out[v0['data']] < 0.))
                            out[v0['data']] = out[v0['data']][~indout]
                            out[v0['t']] = out[v0['t']][~indout]
                            dout[key] = scpstats.binned_statistic(
                                out[v0['t']], out[v0['data']],
                                statistic='mean', bins=tbins,
                                range=None)[0]
                        except Exception as err:
                            pass

                except Exception as err:
                    pass

            name = 'XICS_fit2d_{}_nbs{}_tol{}_bin{}{}{}.npz'.format(
                shots[ii], nbsplines, int(-np.log10(tol)),
                binning['phi']['nbins'],
                '_Plasma' if plasma is True else '',
                nameextra)
            pfe = os.path.join(path, name)
            np.savez(pfe, **dout)
            msg = ('shot {}: saved in {}'.format(shots[ii], pfe))
            print(msg)

        except Exception as err:
            if 'All nan in region scanned for scale' in str(err):
                plt.close('all')
            msg = ("shot {}: {}".format(shots[ii], str(err)))
            warnings.warn(msg)


def _get_files(path, nameextra, nameexclude):
    if nameextra is None:
        nameextra = ''
    if isinstance(nameextra, str):
        nameextra = [nameextra]
    if nameexclude is None:
        nameexclude = '----------'
    if isinstance(nameexclude, str):
        nameexclude = [nameexclude]
    lf = [ff for ff in os.listdir(path)
          if (all([ss in ff
                   for ss in ['XICS', 'fit2d', 'nbs', '.npz'] + nameextra])
              and all([ss not in ff for ss in nameexclude]))]
    return lf


def _get_dall_from_lf(lf, ls, lsextra, path, ratio=None):
    dall = {kk: [] for kk in ls + lsextra}
    if 'vims' in ls:
        dall['vims_keys'] = []
    indshot = len('XICS_fit2d_')
    for ff in lf:
        din = {}
        try:
            cryst = 'ArXVII'
            shot = int(ff[indshot:indshot+5])
            out = np.load(os.path.join(path, ff), allow_pickle=True)
            nt = out['t'].size
            for kk in ls:
                if kk == 'angle':
                    din[kk] = np.full((nt,), _DSHOTS[cryst][shot]['ang'])
                if kk not in out.keys():
                    continue
                if kk == 'ratio':
                    ind = out['ratio'].tolist()['str'].index(ratio)
                    din[kk] = out['ratio'].tolist()['value'][:, ind, :]
                elif kk == 'vims':
                    din['vims_keys'] = out['dinput'].tolist()['shift']['keys']
                    din[kk] = out.get(kk, np.full((nt,), np.nan))
                else:
                    din[kk] = out.get(kk, np.full((nt,), np.nan))
            lambmean = np.mean(out['dinput'].tolist()['lambminmax'])
            din['shot'] = np.full((nt,), shot)
            din['sumsig'] = np.nansum(out['data'], axis=1)
            din['lambmean'] = np.full((nt,), lambmean)
            din['ff'] = np.full((nt,), ff, dtype='U')

        except Exception as err:
            continue
        for kk in din.keys():
            if kk == 'pts_phi':
                dall[kk] = din[kk]
            elif kk == 'vims_keys':
                dall[kk] = din[kk]
            elif din[kk].ndim in [2, 3]:
                if dall[kk] == []:
                    dall[kk] = din[kk]
                else:
                    dall[kk] = np.concatenate((dall[kk], din[kk]), axis=0)
            else:
                dall[kk] = np.append(dall[kk], din[kk])
    return dall


def treat_plot_double(path=None, nameextra=None, nameexclude=None,
                      cmap=None, color=None, alpha=None,
                      vmin=None, vmax=None, size=None,
                      fs=None, dmargin=None):

    # ---------
    # Prepare
    if path is None:
        path = _HERE
    if cmap is None:
        cmap = plt.cm.viridis
    if color is None:
        color = 'shot'
    assert isinstance(color, str)
    if alpha is None:
        alpha = 'sumsig'
    if size is None:
        size = 30

    lf = _get_files(path, nameextra, nameexclude)
    nf = len(lf)
    ls = ['dratio', 'dshift', 't', 'cost', 'angle',
          'ic_power', 'lh_power', 'ece_Te0']
    lsextra = ['shot', 'sumsig', 'lambmean', 'ff']
    dall = _get_dall_from_lf(lf, ls, lsextra, path)
    if len(dall.keys()) == 0:
        warnings.warn("No data in dall!")

    # Prepare color, alpha and size
    if isinstance(size, str):
        size = dall[size]
    color = cmap(mcolors.Normalize(vmin=vmin, vmax=vmax)(dall[color]))
    if isinstance(alpha, str):
        alpha = mcolors.Normalize()(dall[alpha])
    else:
        alpha = 1.
    color[:, -1] = alpha

    # ---------
    # plot
    if fs is None:
        fs = (12, 6)
    if cmap is None:
        cmap = plt.cm.viridis
    if dmargin is None:
        dmargin = {'left': 0.08, 'right': 0.96,
                   'bottom': 0.08, 'top': 0.93,
                   'wspace': 0.4, 'hspace': 0.2}

    fig = plt.figure(figsize=fs)
    gs = gridspec.GridSpec(2, 16, **dmargin)

    shx0, shy0, shx1, shy1, shx2, shy2 = None, None, None, None, None, None
    dax = {'dratio': None,
           'dshift': None}
    dax['dratio'] = fig.add_subplot(gs[0, :-1])
    dax['dshift'] = fig.add_subplot(gs[1, :-1], sharex=dax['dratio'])
    dax['dratio_c'] = fig.add_subplot(gs[0, -1])
    dax['dshift_c'] = fig.add_subplot(gs[1, -1])
    dax['dratio'].set_ylabel('double ratio (a.u.)')
    dax['dshift'].set_ylabel('double shift (a.u.)')
    dax['dshift'].set_xlabel(r'$\lambda$' + ' (m)')

    dr = dax['dratio'].scatter(dall['lambmean'], dall['dratio'],
                               c=color, s=size, marker='o', edgecolors='None')
    dax['dshift'].scatter(dall['lambmean'], dall['dshift'],
                          c=color, s=size, marker='o', edgecolors='None')

    # dax['dratio'].set_ylim(0, 2)
    dax['dratio'].set_xlim(3.94e-10, 4e-10)
    dax['dratio'].set_ylim(0, 1.)
    dax['dshift'].set_ylim(0, 6e-4)

    plt.colorbar(dr, cax=dax['dratio_c'])
    # plt.colorbar(dr, cax=dax['dshift_c'])

    # Optional figure vs angle
    if np.unique(dall['angle']).size == 1:
        return dall, dax

    fig1 = plt.figure(figsize=fs)
    gs = gridspec.GridSpec(2, 16, **dmargin)

    shx0, shy0, shx1, shy1, shx2, shy2 = None, None, None, None, None, None
    dax1 = {'dratio': None,
            'dshift': None}
    dax1['dratio'] = fig1.add_subplot(gs[0, :-1])
    dax1['dshift'] = fig1.add_subplot(gs[1, :-1], sharex=dax1['dratio'])
    dax1['dratio_c'] = fig1.add_subplot(gs[0, -1])
    dax1['dshift_c'] = fig1.add_subplot(gs[1, -1])
    dax1['dratio'].set_ylabel('double ratio (a.u.)')
    dax1['dshift'].set_ylabel('double shift (a.u.)')
    dax1['dshift'].set_xlabel('rotation angle (rad)')

    llamb, langle, ldratio, ldshift = [], [], [], []
    done = np.zeros((dall['lambmean'].size,), dtype=bool)
    for ll in np.unique(dall['lambmean']):
        indl = (~done) & (np.abs(dall['lambmean'] - ll) < 0.005e-10)
        if not np.any(indl):
            continue
        for aa in np.unique(dall['angle']):
            inda = indl & (dall['angle'] == aa)
            if not np.any(inda):
                continue
            llamb.append(ll)
            langle.append(aa)
            ldratio.append(np.nansum(dall['dratio'][inda]
                                     * dall['sumsig'][inda])
                           / np.nansum(dall['sumsig'][inda]))
            ldshift.append(np.nansum(dall['dshift'][inda]
                                     * dall['sumsig'][inda])
                           / np.nansum(dall['sumsig'][inda]))
        done[indl] = True
    llamb, langle, ldratio, ldshift = map(np.asarray,
                                          [llamb, langle, ldratio, ldshift])

    for ll in np.unique(llamb):
        ind = llamb == ll
        l,  = dax1['dratio'].plot(langle[ind], ldratio[ind],
                                  ms=8, marker='o',
                                  label=(r'$\lambda\approx$'
                                         + '{:4.2e} m'.format(ll)))
        dax1['dshift'].plot(langle[ind], ldshift[ind],
                            ms=8, marker='o', c=l.get_color(),
                            label=(r'$\lambda\approx$'
                                   + '{:4.2e} m'.format(ll)))

    # dax1['dratio'].set_xlim(3.94e-10, 4e-10)
    dax1['dratio'].set_ylim(0, 1.)
    dax1['dshift'].set_ylim(0, 6e-4)

    dax1['dratio'].legend()
    return dall, (dax, dax1)


def treat_plot_lineratio(ratio=None, path=None,
                         nameextra=None, nameexclude=None,
                         cmap=None, color=None, alpha=None,
                         vmin=None, vmax=None, size=None,
                         fs=None, dmargin=None):

    # ---------
    # Prepare
    if path is None:
        path = _HERE
    if cmap is None:
        cmap = plt.cm.viridis
    if color is None:
        color = 'shot'
    assert isinstance(color, str)
    if alpha is None:
        alpha = 'sumsig'
    if size is None:
        size = 30

    lf = _get_files(path, nameextra, nameexclude)
    nf = len(lf)
    ls = ['ratio', 'pts_phi', 't', 'cost',
          'ic_t', 'ic_power', 'lh_t', 'lh_power', 'ece_Te0', 'ece_t']
    lsextra = ['shot', 'sumsig', 'lambmean', 'ff']
    dall = _get_dall_from_lf(lf, ls, lsextra, path, ratio=ratio)
    if len(dall.keys()) == 0:
        warnings.warn("No data in dall!")

    shotsu = np.unique(dall['shot'])

    # Prepare color, alpha and size
    if isinstance(size, str):
        size = dall[size]
    if isinstance(color, str):
        if color == 'shot_index':
            shotu = np.unique(dall['shot'])
            color = dall['shot'].astype(int)
            for ii in range(0, shotu.size):
                color[dall['shot'] == shotu[ii]] = ii
        else:
            color = dall[color]
        color = cmap(mcolors.Normalize(vmin=vmin, vmax=vmax)(color))
    if isinstance(alpha, str):
        alpha = mcolors.Normalize()(dall[alpha])
    else:
        alpha = 1.
    color[:, -1] = alpha

    # ---------
    # plot
    if fs is None:
        fs = (12, 6)
    if cmap is None:
        cmap = plt.cm.viridis
    if dmargin is None:
        dmargin = {'left': 0.08, 'right': 0.96,
                   'bottom': 0.08, 'top': 0.93,
                   'wspace': 0.4, 'hspace': 0.2}

    fig = plt.figure(figsize=fs)
    gs = gridspec.GridSpec(2, 16, **dmargin)

    shx0, shy0, shx1, shy1, shx2, shy2 = None, None, None, None, None, None
    dax = {'ratio': None,
           'dshift': None}
    dax['ratio'] = fig.add_subplot(gs[0, :-1])
    dax['dshift'] = fig.add_subplot(gs[1, :-1], sharex=dax['ratio'])
    dax['ratio_c'] = fig.add_subplot(gs[0, -1])
    dax['dshift_c'] = fig.add_subplot(gs[1, -1])
    dax['ratio'].set_ylabel('line ratio (a.u.)')
    dax['dshift'].set_ylabel('double shift (a.u.)')
    dax['dshift'].set_xlabel('channel (rad)')

    for ii in range(dall['ratio'].shape[0]):
        dax['ratio'].plot(dall['pts_phi'], dall['ratio'][ii, :],
                          color=color[ii, :], ls='-', lw=1.)

    # dax['dratio'].set_ylim(0, 2)
    # dax['ratio'].set_xlim(3.94e-10, 4e-10)
    dax['ratio'].set_ylim(0, 2.)
    dax['dshift'].set_ylim(0, 6e-4)

    # plt.colorbar(dr, cax=dax['dratio_c'])
    # plt.colorbar(dr, cax=dax['dshift_c'])
    return dall, dax


def treat_plot_lineshift(path=None, diff=None,
                         nameextra=None, nameexclude=None,
                         cmap=None, color=None, alpha=None,
                         vmin=None, vmax=None, size=None,
                         fs=None, dmargin=None):

    # ---------
    # Prepare
    if path is None:
        path = _HERE
    if cmap is None:
        cmap = plt.cm.viridis
    if color is None:
        color = 'line'
    assert isinstance(color, str)
    if alpha is None:
        alpha = 'sumsig'
    if size is None:
        size = 30

    lf = _get_files(path, nameextra, nameexclude)
    nf = len(lf)
    ls = ['vims', 'pts_phi', 't', 'cost',
          'ic_power', 'lh_power', 'ece_Te0']
    lsextra = ['shot', 'sumsig', 'lambmean', 'ff']
    dall = _get_dall_from_lf(lf, ls, lsextra, path)
    if len(dall.keys()) == 0:
        warnings.warn("No data in dall!")

    shotsu = np.unique(dall['shot'])

    # Prepare color, alpha and size
    if isinstance(size, str):
        size = dall[size]
    if isinstance(color, str):
        if color == 'line':
            color = ['r', 'b', 'g']
    if isinstance(alpha, str):
        alpha = np.array(mcolors.Normalize()(dall[alpha]))
    else:
        alpha = 1.

    dall['vims'] = dall['vims']*1.e-3
    if diff is None:
        diff = dall['vims'].shape[1] in [2, 3]
    if diff is True:
        if dall['vims'].shape[1] == 2:
            dvims = np.diff(dall['vims'], axis=1)
            labd = [dall['vims_keys'][1] + '-' + dall['vims_keys'][0]]
        elif dall['vims'].shape[1] == 3:
            dvims = np.concatenate(
                (np.diff(dall['vims'], axis=1),
                 dall['vims'][:, 0:1, :] - dall['vims'][:, 2:3, :]), axis=1)
            labd = [dall['vims_keys'][1] + ' - ' + dall['vims_keys'][0],
                    dall['vims_keys'][2] + ' - ' + dall['vims_keys'][1],
                    dall['vims_keys'][0] + ' - ' + dall['vims_keys'][2]]
        colord = ['k', 'y', 'c']

    # ---------
    # plot
    if fs is None:
        fs = (12, 6)
    if cmap is None:
        cmap = plt.cm.viridis
    if dmargin is None:
        dmargin = {'left': 0.08, 'right': 0.96,
                   'bottom': 0.08, 'top': 0.93,
                   'wspace': 0.4, 'hspace': 0.2}

    fig = plt.figure(figsize=fs)
    gs = gridspec.GridSpec(2, 16, **dmargin)

    shx0, shy0, shx1, shy1, shx2, shy2 = None, None, None, None, None, None
    dax = {'ratio': None,
           'dshift': None}
    dax['vims'] = fig.add_subplot(gs[0, :-1])
    dax['dvims'] = fig.add_subplot(gs[1, :-1], sharex=dax['ratio'])
    # dax['vims_c'] = fig.add_subplot(gs[0, -1])
    dax['vims'].set_ylabel(r'$v_i$' + '  (km.s^-1)')
    dax['dvims'].set_ylabel(r'$\Delta v_i$' + '  (km.s^-1)')
    dax['dvims'].set_xlabel('channel (rad)')

    for ii in range(dall['vims'].shape[0]):
        for jj in range(dall['vims'].shape[1]):
            dax['vims'].plot(dall['pts_phi'], dall['vims'][ii, jj, :],
                             color=color[jj], alpha=alpha[ii], ls='-', lw=1.)

    if diff is True:
        for jj in range(dvims.shape[1]):
            for ii in range(dall['vims'].shape[0]):
                dax['dvims'].plot(dall['pts_phi'], dvims[ii, jj, :],
                                  color=colord[jj], alpha=alpha[ii])
            dvmean = (np.nansum(dvims[:, jj, :]*dall['sumsig'][:, None])
                      / (dvims.shape[2]*np.nansum(dall['sumsig'])))
            dax['dvims'].axhline(dvmean, c=colord[jj], ls='--', lw=1.)
            dax['dvims'].annotate(
                '{:5.3e}'.format(dvmean),
                xy=(1., dvmean),
                xycoords=('axes fraction', 'data'),
                color=colord[jj], size=10,
            )

    dax['vims'].axhline(0., c='k', ls='--', lw=1.)
    dax['dvims'].axhline(0., c='k', ls='--', lw=1.)
    hand = [mlines.Line2D([], [], c=color[jj], lw=1., ls='-')
            for jj in range(dall['vims'].shape[1])]
    lab = dall['vims_keys'].tolist()
    dax['vims'].legend(hand, lab)
    if diff is True:
        hand = [mlines.Line2D([], [], c=colord[jj], lw=1., ls='-')
                for jj in range(dvims.shape[1])]
        dax['dvims'].legend(hand, labd)

    return dall, dax


# #############################################################################
#                   Noise estimates
# #############################################################################


def noise(cryst, shot=None,
          dlines=None,
          dconst=None,
          nbsplines=None,
          ratio=None,
          binning=None,
          tol=None,
          plasma=True,
          nameextra=None,
          domain=None,
          xi=_XI, xj=_XJ,
          path=_HERE):

    if nbsplines is None:
        nbsplines = 15
    if binning is None:
        binning = {'lamb': 487, 'phi': 100}
    if nameextra is None:
        nameextra = ''
    if nameextra != '' and nameextra[0] != '_':
        nameextra = '_' + nameextra

    # Shots
    dshots = _DSHOTS[cryst]
    shots = np.unique([kk for kk in dshots.keys()
                       if len(dshots[kk]['tlim']) == 2])
    if shot is not None:
        if not hasattr(shot, '__iter__'):
            shot = [shot]
        ind = np.array([shots == ss for ss in shot])
        shots = shots[ind]

    # Cryst part
    det = dict(np.load(os.path.join(_HERE, 'det37_CTVD_incC4.npz')))
    cryst, det = _get_crystanddet(cryst=cryst, det=det)
    assert cryst is not False

    mask = ~np.any(np.load(_MASKPATH)['ind'], axis=0)

    for ii in range(shots.size):
        if len(dshots[int(shots[ii])]['tlim']) != 2:
            continue
        print('\n\nshot {} ({} / {})'.format(shots[ii], ii+1, shots.size))
        try:
            cryst.move(dshots[int(shots[ii])]['ang']*np.pi/180.)
            tlim = [None, dshots[int(shots[ii])]['tlim'][1]]
            data, t, dbonus = _load_data(int(shots[ii]), tlim=tlim)
            dout = cryst.fit2d_prepare(
                data=data, xi=xi, xj=xj, det=det,
                subset=None, binning=binning,
                nbsplines=nbsplines, mask=mask, domain=domain)
            dout['t'] = t
            dout['shot'] = shots[ii]
            dout.update(dbonus)

            name = 'XICS_fit2d_prepare_{}_nbs{}_bin{}{}.npz'.format(
                shots[ii], nbsplines, dout['binning']['phi']['nbins'],
                nameextra)
            pfe = os.path.join(path, name)
            np.savez(pfe, **dout)
            msg = ('shot {}: saved in {}'.format(shots[ii], pfe))
            print(msg)

        except Exception as err:
            if 'All nan in region scanned for scale' in str(err):
                plt.close('all')
            msg = ("shot {}: {}".format(shots[ii], str(err)))
            warnings.warn(msg)


def get_noise_costjac(deg=None, nbsplines=None, phi=None,
                      phiminmax=None, symmetryaxis=None, sparse=None):

    if sparse is None:
        sparse = False

    dbsplines = tf.data._spectrafit2d.multigausfit2d_from_dlines_dbsplines(
        knots=None, deg=deg, nbsplines=nbsplines,
        phimin=phiminmax[0], phimax=phiminmax[1],
        symmetryaxis=symmetryaxis)

    def cost(x, km=dbsplines['knots_mult'], data=None, phi=phi):
        return scpinterp.BSpline(km, x, deg,
                                 extrapolate=False, axis=0)(phi) - data

    jac = np.zeros((phi.size, dbsplines['nbs']), dtype=float)
    km = dbsplines['knots_mult']
    kpb = dbsplines['nknotsperbs']
    lind = [(phi >= km[ii]) & (phi < km[ii+kpb-1])
            for ii in range(dbsplines['nbs'])]
    if sparse is True:
        def jac_func(x, jac=jac, km=km, data=None,
                     phi=phi, kpb=kpb, lind=lind):
            for ii in range(x.size):
                jac[lind[ii], ii] = scpinterp.BSpline.basis_element(
                    km[ii:ii+kpb], extrapolate=False)(phi[lind[ii]])
            return scpsparse.csr_matrix(jac)
    else:
        def jac_func(x, jac=jac, km=km, data=None,
                     phi=phi, kpb=kpb, lind=lind):
            for ii in range(x.size):
                jac[lind[ii], ii] = scpinterp.BSpline.basis_element(
                    km[ii:ii+kpb], extrapolate=False)(phi[lind[ii]])
            return jac
    return cost, jac_func


def plot_noise(filekeys=None, lf=None, path=None, deg=None, tnoise=None,
               nbsplines=None, symmetryaxis=None, lnbsplines=None, nbins=None,
               sparse=None, method=None, xtol=None, ftol=None, gtol=None,
               loss=None, max_nfev=None, tr_solver=None,
               alpha=None, verb=None, timeit=None,
               plot=None, fs=None, cmap=None, dmargin=None):

    # ---------------
    # Check inputs
    if deg is None:
        deg = 2
    if nbsplines is None:
        nbsplines = 13
    if lnbsplines is None:
        lnbsplines = np.arange(5, 20)
    lnbsplines = np.array(lnbsplines)
    if symmetryaxis is None:
        symmetryaxis = False
    if path is None:
        path = os.path.dirname(__file__)
    if filekeys is None:
        filekeys = []
    filekeys += ['XICS', 'fit2d', 'prepare', '.npz']
    if lf is None:
        lf = [
            ff for ff in os.listdir(path)
            if all([ss in ff for ss in filekeys])
        ]
    if len(lf) == 0:
        return
    if tnoise is None:
        tnoise = 30.
    if method is None:
        method = 'trf'
    assert method in ['trf', 'dogbox', 'lm'], method
    if tr_solver is None:
        tr_solver = 'exact'
    _TOL = 1.e-14
    if xtol is None:
        xtol = _TOL
    if ftol is None:
        ftol = _TOL
    if gtol is None:
        gtol = _TOL
    if loss is None:
        loss = 'linear'
    if max_nfev is None:
        max_nfev = None
    if plot is None:
        plot = True
    if verb is None:
        verb = True
    if timeit is None:
        timeit = not verb
    if nbins is None:
        nbins = 15

    # ---------------
    # Get data
    dout = {ff: dict(np.load(os.path.join(path, ff), allow_pickle=True))
            for ff in lf}
    dnt = {ff: dout[ff]['t'].size for ff in lf}

    dall = {
        'shot': np.concatenate([np.full((dnt[ff],), dout[ff]['shot'])
                                for ff in lf]),
        't': np.concatenate([dout[ff]['t'] for ff in lf]),
        'phi1d': dout[lf[0]]['phi1d'],
        'dataphi1d': np.concatenate([dout[ff]['dataphi1d'] for ff in lf],
                                    axis=0),
        'data': np.concatenate([dout[ff]['data'] for ff in lf],
                               axis=0)}

    nlamb = dout[lf[0]]['binning'].tolist()['lamb']['nbins']
    coeflamb = nlamb / (nlamb - 1)
    dataphidmean = np.nanmean(dall['dataphi1d'], axis=1)
    dall['indnosignal'] = (dall['t'] <= tnoise) | (dataphidmean < 0.4)
    nbnosignal = dall['indnosignal'].sum()
    coefstd = nbnosignal / (nbnosignal - 1)
    dall['nosignal_mean'] = np.nanmean(dall['data'][dall['indnosignal'], :, :],
                                       axis=0)
    dall['nosignal_var'] = np.nanstd(dall['data'][dall['indnosignal'], :, :],
                                     axis=0)**2 * coefstd
    dall['nosignal_1dmean'] = np.nanmean(
        dall['dataphi1d'][dall['indnosignal'], :], axis=0)
    dall['dataphi1d_var'] = np.nanstd(dall['data'], axis=1) * coeflamb
    dall['nosignal_1dvar'] = np.nanmean(
        dall['dataphi1d'][dall['indnosignal'], :], axis=0) * coefstd

    lambminmax = dout[lf[0]]['domain'].tolist()['lamb']['minmax']
    phiminmax = dout[lf[0]]['domain'].tolist()['phi']['minmax']
    extent = (lambminmax[0], lambminmax[1], phiminmax[0], phiminmax[1])
    shotu = np.unique(dall['shot'])
    if alpha is None:
        alpha = mcolors.Normalize()(np.nanmean(dall['dataphi1d'], axis=1))
        alpha = np.array(alpha)
        alpha[alpha < 0.005] = 0.005
    else:
        alpha = np.full((dall['dataphi1d'].shape[0],), alpha)
    knots = None
    datamax = np.nanmax(dall['dataphi1d'], axis=1)
    dataphi1dnorm = dall['dataphi1d'] / datamax[:, None]
    indj = (~dall['indnosignal']).nonzero()[0]
    dataphi1dok = dall['dataphi1d'][indj, :]
    lchi2 = np.full((dall['dataphi1d'].shape[0], lnbsplines.size), np.nan)
    shape = tuple(np.r_[dall['dataphi1d'].shape, lnbsplines.size])
    err = np.full(shape, np.nan)
    sol_x = np.full((dall['dataphi1d'].shape[0], nbsplines), np.nan)
    if timeit is True:
        t0 = dtm.datetime.now()
    for ii in range(lnbsplines.size):
        x0 = 1. - (2.*np.arange(lnbsplines[ii])/lnbsplines[ii] - 1.)**2
        cost, jac = get_noise_costjac(deg=deg, phi=dall['phi1d'],
                                      nbsplines=int(lnbsplines[ii]),
                                      phiminmax=phiminmax, sparse=sparse,
                                      symmetryaxis=symmetryaxis)
        for jj in range(indj.size):
            if verb is True:
                msg = ("\tnbsplines = {} ({}/{}),".format(lnbsplines[ii], ii+1,
                                                          lnbsplines.size)
                       + "\tprofile {} ({}/{})".format(indj[jj], jj+1,
                                                       indj.size))
                print(msg.ljust(60), flush=True, end='\r')
            res = scpopt.least_squares(
                cost, x0, jac=jac,
                method=method, ftol=ftol, xtol=xtol, gtol=gtol,
                x_scale='jac', f_scale=1.0, loss=loss, diff_step=None,
                tr_solver=tr_solver, tr_options={}, jac_sparsity=None,
                max_nfev=max_nfev, verbose=0, args=(),
                kwargs={'data': dataphi1dnorm[indj[jj], :]})

            lchi2[indj[jj], ii] = np.nansum(
                cost(x=res.x, data=dataphi1dnorm[indj[jj], :])**2)
            err[indj[jj], :, ii] = (cost(res.x, data=0.) * datamax[indj[jj]]
                                    - dall['dataphi1d'][indj[jj], :])
            if lnbsplines[ii] == nbsplines:
                sol_x[indj[jj], :] = res.x
    dall['err'] = err

    # Mean and var of err
    errok = err[indj, :]
    dataphi1dbin = np.linspace(0., np.nanmax(dataphi1dok), nbins)
    indbin = np.searchsorted(dataphi1dbin, dataphi1dok)
    errbin_mean = np.full((dataphi1dbin.size, lnbsplines.size), np.nan)
    errbin_var = np.full((dataphi1dbin.size, lnbsplines.size), np.nan)
    alpha_err = np.full((lnbsplines.size,), np.nan)
    for ii in range(dataphi1dbin.size):
        nok = (~np.isnan(errok[indbin == ii])).sum()
        errbin_mean[ii, :] = np.nanmean(errok[indbin == ii], axis=0)
        errbin_var[ii, :] = (
            np.nanstd(errok[indbin == ii], axis=0)**2 * (nok-1)/nok
        )

    if timeit is True:
        dall['timeit'] = (dtm.datetime.now()-t0).total_seconds()
    lchi2 = lchi2 / np.nanmax(lchi2, axis=1)[:, None]

    dbsplines = tf.data._spectrafit2d.multigausfit2d_from_dlines_dbsplines(
        knots=None, deg=deg, nbsplines=nbsplines,
        phimin=phiminmax[0], phimax=phiminmax[1],
        symmetryaxis=symmetryaxis)
    dall['fitphi1d'] = np.full(dall['dataphi1d'].shape, np.nan)
    for ii in indj:
        dall['fitphi1d'][ii, :] = scpinterp.BSpline(
            dbsplines['knots_mult'],
            sol_x[ii, :], dbsplines['deg'],
            extrapolate=False, axis=0)(dall['phi1d'])

    dall['dataphi1dbin'] = dataphi1dbin
    dall['errbin_mean'] = errbin_mean
    dall['errbin_var'] = errbin_var
    dall['indbin'] = indbin

    # ---------
    # Debug
    # ijk = np.any(dall['dataphi1d'] > 1100, axis=1)
    # ijk = np.nonzero(ijk)[0]
    # plt.figure(figsize=(18, 6))
    # for ii in range(len(ijk)):
    # plt.subplot(1, len(ijk), ii+1)
    # tit = "{} - t = {} s".format(dall['shot'][ijk[ii]],
    # dall['t'][ijk[ii]])
    # plt.gca().set_title(tit)
    # plt.plot(dall['phi1d'], dall['dataphi1d'][ijk[ii], :],
    # c='k', marker='.', ls='None')
    # plt.plot(dall['phi1d'], dall['fitphi1d'][ijk[ii], :]*datamax[ijk[ii]],
    # c='r', marker='None', ls='-')
    # dbsplines = tf.data._spectrafit2d.multigausfit2d_from_dlines_dbsplines(
    # knots=None, deg=deg, nbsplines=nbsplines-1,
    # phimin=phiminmax[0], phimax=phiminmax[1],
    # symmetryaxis=symmetryaxis)
    # fitphi1dtemp = scpinterp.LSQUnivariateSpline(
    # dall['phi1d'], dall['dataphi1d'][ijk[ii], :],
    # dbsplines['knots'][1:-1], w=None,
    # bbox=[dbsplines['knots'][0], dbsplines['knots'][-1]],
    # k=deg, ext=0, check_finite=False)(dall['phi1d'])
    # plt.plot(dall['phi1d'], fitphi1dtemp,
    # c='g', marker='None', ls='-')
    # dbsplines = tf.data._spectrafit2d.multigausfit2d_from_dlines_dbsplines(
    # knots=None, deg=deg, nbsplines=nbsplines+1,
    # phimin=phiminmax[0], phimax=phiminmax[1],
    # symmetryaxis=symmetryaxis)
    # fitphi1dtemp = scpinterp.LSQUnivariateSpline(
    # dall['phi1d'], dall['dataphi1d'][ijk[ii], :],
    # dbsplines['knots'][1:-1], w=None,
    # bbox=[dbsplines['knots'][0], dbsplines['knots'][-1]],
    # k=deg, ext=0, check_finite=False)(dall['phi1d'])
    # plt.plot(dall['phi1d'], fitphi1dtemp,
    # c='b', marker='None', ls='-')

    # import pdb; pdb.set_trace()     # DB
    # End debug
    # --------------

    if plot is False:
        return dall

    # ---------------
    # Plot
    if fs is None:
        fs = (16, 8)
    if cmap is None:
        cmap = plt.cm.viridis
    if dmargin is None:
        dmargin = {'left': 0.08, 'right': 0.96,
                   'bottom': 0.08, 'top': 0.93,
                   'wspace': 0.4, 'hspace': 0.2}
    tstr0 = '(t <= {} s)'.format(tnoise)
    tstr1 = '(t > {} s)'.format(tnoise)

    fig = plt.figure(figsize=fs)
    gs = gridspec.GridSpec(3, 4, **dmargin)

    shx0, shy0, shx1, shy1, shx2, shy2 = None, None, None, None, None, None
    dax = {}
    dax['nosignal_mean2d'] = fig.add_subplot(gs[0, 0])
    dax['nosignal_var2d'] = fig.add_subplot(gs[1, 0],
                                            sharex=dax['nosignal_mean2d'],
                                            sharey=dax['nosignal_mean2d'])
    dax['nosignal_mean'] = fig.add_subplot(gs[0, 1],
                                           sharey=dax['nosignal_mean2d'])
    dax['nosignal_var'] = fig.add_subplot(gs[1, 1],
                                          sharex=dax['nosignal_mean'],
                                          sharey=dax['nosignal_mean2d'])
    dax['signal_fit'] = fig.add_subplot(gs[0, 2],
                                        sharey=dax['nosignal_mean2d'])
    dax['signal_chi2'] = fig.add_subplot(gs[1, 2],
                                         sharey=dax['nosignal_mean2d'])
    dax['signal_conv'] = fig.add_subplot(gs[0, 3])
    dax['signal_err'] = fig.add_subplot(gs[1, 3])
    dax['signal_err_hist'] = fig.add_subplot(gs[2, 0])
    dax['signal_err_mean'] = fig.add_subplot(gs[2, 1])
    dax['signal_err_var'] = fig.add_subplot(gs[2, 2])
    dax['nosignal_mean2d'].set_title('mean of noise\nno signal ' + tstr0)
    dax['nosignal_var2d'].set_title('variance of noise\nno signal ' + tstr0)
    dax['nosignal_mean'].set_title('mean of noise\nno signal ' + tstr0)
    dax['nosignal_var'].set_title('variance of noise\nno signal ' + tstr0)
    dax['signal_fit'].set_title('fit of mean signal' + tstr1)
    dax['signal_chi2'].set_title('fit chi2' + tstr1)
    dax['signal_conv'].set_title('Convergence')
    dax['signal_err'].set_title('Error')
    dax['signal_err_hist'].set_title('Error histogram')
    dax['signal_err_mean'].set_title('Error mean')
    dax['signal_err_var'].set_title('Error var')

    dax['nosignal_mean2d'].set_ylabel('phi (rad)')
    dax['nosignal_var2d'].set_ylabel('phi (rad)')
    dax['nosignal_var2d'].set_xlabel('lamb (m)')
    dax['signal_conv'].set_xlabel('nbsplines')
    dax['signal_conv'].set_ylabel(r'$\chi^2$')
    dax['signal_err'].set_xlabel('data')
    dax['signal_err'].set_ylabel('error')
    dax['signal_err_hist'].set_xlabel('error')
    dax['signal_err_hist'].set_ylabel('occurences')
    dax['signal_err_mean'].set_xlabel('data')
    dax['signal_err_mean'].set_ylabel('error mean')
    dax['signal_err_var'].set_xlabel('data')
    dax['signal_err_var'].set_ylabel('error var')

    # Plot data
    dax['nosignal_mean2d'].imshow(dall['nosignal_mean'].T,
                                  extent=extent, aspect='auto',
                                  origin='lower', interpolation='nearest',
                                  vmin=0, vmax=5)
    dax['nosignal_var2d'].imshow(dall['nosignal_var'].T,
                                 extent=extent, aspect='auto',
                                 origin='lower', interpolation='nearest',
                                 vmin=0, vmax=5)

    col = None
    dataph1dflat = dall['dataphi1d'].ravel()
    for ii in range(shotu.size):
        # No signal
        ind = (dall['indnosignal'] & (dall['shot'] == shotu[ii])).nonzero()[0]
        for jj in range(ind.size):
            if jj == 0:
                l, = dax['nosignal_mean'].plot(dall['dataphi1d'][ind[jj], :],
                                               dall['phi1d'],
                                               ls='-', marker='None', lw=1.,
                                               alpha=alpha[ind[jj]])
                col = l.get_color()
            else:
                dax['nosignal_mean'].plot(dall['dataphi1d'][ind[jj], :],
                                          dall['phi1d'],
                                          ls='-', marker='None', lw=1., c=col,
                                          alpha=alpha[ind[jj]])

            dax['nosignal_var'].plot(dall['dataphi1d_var'][ind[jj], :],
                                     dall['phi1d'],
                                     ls='-', marker='None', lw=1., c=col,
                                     alpha=alpha[ind[jj]])
        # Signal
        ind = ((~dall['indnosignal'])
               & (dall['shot'] == shotu[ii])).nonzero()[0]
        for jj in range(ind.size):
            dax['signal_fit'].plot(dataphi1dnorm[ind[jj], :],
                                   dall['phi1d'],
                                   ls='None', marker='.', lw=1., c=col,
                                   alpha=alpha[ind[jj]])
            dax['signal_fit'].plot(dall['fitphi1d'][ind[jj], :],
                                   dall['phi1d'],
                                   ls='-', marker='None', lw=1., c=col,
                                   alpha=alpha[ind[jj]])
            dax['signal_chi2'].plot(
                (dall['fitphi1d'][ind[jj], :] - dataphi1dnorm[ind[jj], :]),
                dall['phi1d'],
                ls='-', marker='None', lw=1., c=col,
                alpha=alpha[ind[jj]],
            )

            # Convergence
            dax['signal_conv'].plot(lnbsplines, lchi2[ind[jj], :],
                                    ls='-', marker='.', lw=1., c=col,
                                    alpha=alpha[ind[jj]])

    # Error
    lnbsplinesok = np.r_[10, 11, 12, 13, 14, 15, 16]
    dista = np.abs(lnbsplinesok - nbsplines)
    alpha_err = (1. - dista/np.max(dista))
    for jj in range(lnbsplinesok.size):
        indjj = (lnbsplines == lnbsplinesok[jj]).nonzero()[0]
        if indjj.size == 0:
            continue
        indjj = indjj[0]
        lab = '{} bsplines'.format(lnbsplinesok[jj])
        l, = dax['signal_err'].plot(dataph1dflat,
                                    dall['err'][:, :, indjj].ravel(),
                                    ls='None', marker='.', alpha=alpha_err[jj])
        col = l.get_color()
        if lnbsplinesok[jj] == nbsplines:
            for ii in range(nbins):
                if np.any(indbin == ii):
                    dax['signal_err_hist'].hist(
                        errok[indbin == ii, jj],
                        bins=10, density=True,
                    )
        dax['signal_err_mean'].plot(dataphi1dbin,
                                    errbin_mean[:, indjj],
                                    ls='None', marker='.', c=col,
                                    alpha=alpha_err[jj])
        dax['signal_err_var'].plot(dataphi1dbin,
                                   errbin_var[:, indjj],
                                   ls='None', marker='.', c=col,
                                   alpha=alpha_err[jj],
                                   label=lab)

    if nbsplines in lnbsplines:
        indjbs = (lnbsplines == nbsplines).nonzero()[0][0]
        dax['signal_err_mean'].axhline(0., c='k', ls='--')
        indok = ~np.isnan(errbin_var[:, indjbs])
        pf = np.polyfit(dataphi1dbin[indok],
                        np.sqrt(errbin_var[indok, indjbs]), 1)
        dax['signal_err_var'].plot(dataphi1dbin,
                                   np.polyval(pf, dataphi1dbin)**2,
                                   c='k', ls='--')

        txt = '({:5.3e}x + {:5.3e})^2'.format(pf[0], pf[1])
        dax['signal_err_var'].annotate(txt,
                                       xy=(500, 200),
                                       xycoords='data',
                                       horizontalalignment='center',
                                       verticalalignment='center',
                                       rotation=np.arctan(np.sqrt(pf[0])),
                                       size=8)

    dax['nosignal_mean'].plot(dall['nosignal_1dmean'],  dall['phi1d'],
                              ls='-', marker='None', lw=2., c='k')
    dax['nosignal_var'].plot(dall['nosignal_1dvar'], dall['phi1d'],
                             ls='-', marker='None', lw=2., c='k')

    dax['nosignal_mean2d'].set_xlim(lambminmax)
    dax['nosignal_mean2d'].set_ylim(phiminmax)
    # dax['nosignal_mean'].set_ylim(phiminmax)
    dax['signal_err_var'].legend(loc='center left',
                                 bbox_to_anchor=(1., 0.5))
    return dall, dax
