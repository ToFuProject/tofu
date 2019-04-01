#

# Built-in
import os

# Common
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend("Qt5Agg")

# tofu-specific ( >= 1.3.23-58 )
import tofu as tf


# ------------------
# Default parameters

_path = os.path.abspath(os.path.dirname(__file__))
_path_Inputs = os.path.join(_path, './Objects')
_path_Objects = os.path.join(_path,'./Objects/')
_path_laura_former = './Objects'



#########################################
#########################################
#       Define config dictionary
#########################################

_dconfig = {'A1': {'Exp':'WEST',
                   'Ves': ['V1']},
            'A2': {'Exp':'ITER',
                   'Ves': ['V0']},
            'A3': {'Exp':'WEST',
                   'PlasmaDomain': ['Sep']},
            'B1': {'Exp':'WEST',
                   'Ves': ['V2'],
                   'PFC': ['BaffleV0', 'DivUpV1', 'DivLowITERV1']},
            'B2': {'Exp':'WEST',
                   'Ves': ['V2'],
                   'PFC': ['BaffleV1', 'DivUpV2', 'DivLowITERV2',
                           'BumperInnerV1', 'BumperOuterV1',
                           'IC1V1', 'IC2V1', 'IC3V1']},
            'B3': {'Exp':'WEST',
                   'Ves': ['V2'],
                   'PFC': ['BaffleV2', 'DivUpV3', 'DivLowITERV3',
                           'BumperInnerV3', 'BumperOuterV3',
                           'IC1V1', 'IC2V1', 'IC3V1',
                           'LH1V1', 'LH2V1',
                           'RippleV1', 'VDEV0']}}


_P = [1.5,3.2,0.]
_F = 0.1
_testF = 0.4
_D12 = [0.3,0.1]
_nIn = [-0.5,-1.,0.]

_PA3 = [1.2, 3,0.]

_P1 = [1.5,-3.2,0.]
_nIn1 = [-0.5,1.,0.]

_PA = [4.9,-6.9,0.]
_nInA = [-0.75, 1.,0.]
_D12A = [0.4,0.3]
_dcam = {'V1':       {'P':_P1, 'F':_F, 'D12':_D12, 'nIn':_nIn1, 'N12':[1,1]},
         'V10':      {'P':_P, 'F':_F, 'D12':_D12, 'nIn':_nIn, 'N12':[5,2]},
         'V100':     {'P':_P, 'F':_F, 'D12':_D12, 'nIn':_nIn, 'N12':[20,5]},
         'V1000':    {'P':_P, 'F':_F, 'D12':_D12, 'nIn':_nIn, 'N12':[50,20]},
         'V10000':   {'P':_P, 'F':_F, 'D12':_D12, 'nIn':_nIn, 'N12':[125,80]},
         'V100000':  {'P':_P, 'F':_F, 'D12':_D12, 'nIn':_nIn, 'N12':[500,200]},
         'V1000000': {'P':_P, 'F':_F, 'D12':_D12, 'nIn':_nIn, 'N12':[1600,625]},
         'VA1':       {'P':_PA, 'F':_F, 'D12':_D12A, 'nIn':_nInA, 'N12':[1,1]},
         'VA10':      {'P':_PA, 'F':_F, 'D12':_D12A, 'nIn':_nInA, 'N12':[5,2]},
         'VA100':     {'P':_PA, 'F':_F, 'D12':_D12A, 'nIn':_nInA, 'N12':[20,5]},
         'VA1000':    {'P':_PA, 'F':_F, 'D12':_D12A, 'nIn':_nInA, 'N12':[50,20]},
         'VA10000':   {'P':_PA, 'F':_F, 'D12':_D12A, 'nIn':_nInA, 'N12':[125,80]},
         'VA100000':  {'P':_PA, 'F':_F, 'D12':_D12A, 'nIn':_nInA, 'N12':[500,200]},
         'VA1000000': {'P':_PA, 'F':_F, 'D12':_D12A, 'nIn':_nInA, 'N12':[1600,625]},
         'V31':       {'P':_PA3, 'F':_F, 'D12':_D12, 'nIn':_nIn, 'N12':[1,1]},
         'V310':      {'P':_PA3, 'F':_F, 'D12':_D12, 'nIn':_nIn, 'N12':[5,2]},
         'V3100':     {'P':_PA3, 'F':_F, 'D12':_D12, 'nIn':_nIn, 'N12':[20,5]},
         'V31000':    {'P':_PA3, 'F':_F, 'D12':_D12, 'nIn':_nIn, 'N12':[50,20]},
         'V310000':   {'P':_PA3, 'F':_F, 'D12':_D12, 'nIn':_nIn, 'N12':[125,80]},
         'V3100000':  {'P':_PA3, 'F':_F, 'D12':_D12, 'nIn':_nIn, 'N12':[500,200]},
         'V31000000': {'P':_PA3, 'F':_F, 'D12':_D12, 'nIn':_nIn, 'N12':[1600,625]},
         'testV': {'P':_P, 'F':_testF, 'D12':_D12, 'nIn':_nIn, 'N12':[1600,625]}
}

#########################################
#########################################
#       Reset objects
#########################################


def recreate_compatible_objects(path=_path_laura_former, save=True):

    lf = os.listdir(path)
    lf = [f for f in lf if all([s in f for s in ['TFG_','.npz']])]
    lS = []
    for f in lf:
        if 'Config' in f:
            continue
        print("File: ", f)

        cls, Exp, nn = f.split('_')[1:4]
        if 'Tor' in cls:
            cls = cls[:cls.index('Tor')]
        Exp = Exp[3:]
        if 'Sep' in Exp:
            Exp = Exp[:Exp.index('Sep')]
        if cls=='Ves' and 'Sep' in f:
            nn, v = 'Sep', ''
        elif cls=='Ves':
            nn, v = nn, ''
        else:
            nn = nn.split('-')
            if len(nn) == 1:
                nn = nn[0]
                if "V" in nn :
                    nn, v = nn[:nn.index("V")], nn[nn.index("V"):]
                else:
                    v = ""
            elif len(nn)==2:
                nn, v = nn
            elif len(nn)==3:
                nn, v = nn[0]+nn[1], nn[2]
        if 'Sep' in nn:
            cls = 'PlasmaDomain'
        if 'UpDiv' in nn:
            nn = 'DivUp'
        if 'LowDivITER' in nn:
            nn = 'DivLowITER'
        if 'OuterBumper' in nn:
            nn = 'BumperOuter'
        if 'InnerBumper' in nn:
            nn = 'BumperInner'
        if 'Test' in nn:
            nn, v = '', 'V0'
        if cls=='Struct':
            cls = 'PFC'
        out = np.load(os.path.join(path,f))
        print("keys: ", list(out.keys()))
        if 'Poly' in out.keys() and "Lim" in out.keys():
            Poly = out['Poly']
            Lim = out['Lim']
        elif "dgeom_Lim" in out.keys() :
            Poly = out['dgeom_Poly']
            Lim = out['dgeom_Lim']
        else :
            Poly = out['dgeom_Poly']
            pos = out['dgeom_pos']
            extent = out["dgeom_extent"]
            if pos.size > 0 :
                Lim = pos[np.newaxis, :] + np.array([[-0.5], [0.5]]) * extent
            else:
                Lim = pos
            print(Exp, cls, nn+v, Poly.shape, Lim.shape)
        if Lim.shape==():
            Lim = Lim.tolist()
        ss = eval("tf.geom.%s"%cls
                  +"(Name=nn+v, Exp=Exp, Poly=Poly, Lim=Lim, SavePath='./')")
        if save:
            ss.save()
        else:
            print("    Would be ",ss)
        lS.append(ss)
    return lS

def recreate_config(dconfig=_dconfig,
                    path=_path_Objects, save=True):

    dout = _get_filenames(dconfig, path)
    dout2 = {}
    for conf in dout:
        Exp = dout[conf]['Exp']
        lS = []
        for cls in dout[conf].keys():
            if cls=='Exp':
                continue
            lS += [tf.load(f) for f in dout[conf][cls]]
        config = tf.geom.Config(Exp=Exp, Name=conf, Type='Tor',
                                lStruct=lS, SavePath=path)
        dout2[conf] = config
        if save:
            config.strip(-1)
            config.save()
        else:
            print("Would be :", config)
    return dout2


#########################################
#########################################
#       Loading routine
#########################################

def _get_filenames(dconfig=_dconfig, path=_path_Objects):
    """ Preliminary hidden routine for getting file names of desired objects
    """

    # Get all files avbailable in path
    lf = os.listdir(path)

    print(path)
    print(lf)
    # Keep only those that are tofu.geom objects
    lf = [ff for ff in lf if all([ss in ff for ss in ['TFG_','.npz']])]

    dout = {}
    for conf in dconfig.keys():
        exp = dconfig[conf]['Exp']
        dout[conf] = {'Exp':exp}
        for cls in dconfig[conf].keys():
            if cls=='Exp':
                continue
            dout[conf][cls] = []
            for n in dconfig[conf][cls]:
                f = [f for f in lf
                     if all([s in f for s in [exp,cls,n]])]
                if not len(f)==1:
                    msg = "None / several matches"
                    msg += " for {0}_{1}_{2}".format(exp,cls,n)
                    msg += " in {0}:\n".format(path)
                    msg += "\n    ".join([ff for ff in f])
                    raise Exception(msg)
                dout[conf][cls].append(f[0])
    return dout


def load_config(config, path=_path_Objects, dconfig=_dconfig, plot=True,
                reset=False):
    """ Load the desired configuration
    """
    assert all([type(ss) is str for ss in [config,path]])
    assert type(dconfig) is dict

    # Get file names from config
    lf = os.listdir(path)
    lf = [f for f in lf if all([s in f for s in ['TFG_Config',config,'.npz']])]
    assert len(lf)==1

    config = tf.load(os.path.join(path,lf[0]))
    config.strip(0)

    # ------------------
    # Optionnal plotting
    if plot:
        lax = config.plot(element='P')

    return config


def get_Du(cam, dcam=_dcam, make_cam=False, plot=False,
           config=None, path=_path_Objects, is_new_ver=True):
    """ Get the (D,u) tuple for the desired camera

    Optionally create the camera with the chosen config
    Optionally plot
    """

    # Extract pinhole, focal length, width, nb. of pix., unit vector
    P, F = dcam[cam]['P'], dcam[cam]['F']
    D12, N12, nIn = dcam[cam]['D12'], dcam[cam]['N12'], dcam[cam]['nIn']
    nIn = nIn / np.linalg.norm(nIn)

    # Compute the LOS starting points and unit vectors
    (D,u) = tf.geom.utils._compute_CamLOS2D_pinhole(P, F, D12, N12,
                                                    nIn=nIn, angs=None,
                                                    return_Du=True)

    if make_cam or plot:
        assert config is not None, "You must specify a config !"

        # Load the config
        conf = load_config(config, path=path, plot=False)

        # Create the LOSCam2D object
        # Note : this is where the computation goes on...
        if is_new_ver :
            method = "optimized"
        else:
            method ="ref"
        cam = tf.geom.CamLOS2D(Exp=conf.Id.Exp, Name=cam, dgeom=(D,u),
                               config=conf, Diag='Test', method=method, plotdebug=False)

    else:
        cam = None

    if plot:
        # Plot
        cam.plot_touch()

    return D, u, cam
