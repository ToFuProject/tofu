#

# Built-in
import os

# Common
import numpy as np
import matplotlib.pyplot as plt
# plt.swtich_backend("Qt5Agg")

# tofu-specific ( >= 1.3.23-58 )
import tofu as tf


# ------------------
# Default parameters

_path = os.path.abspath(os.path.dirname(__file__))
_path_Inputs = os.path.join(_path, '../tests03_core_laura')
_path_Objects = os.path.join(_path,'../tests03_core_laura')
_path_laura_former = '/Home/DV226270/ForOthers/Laura_MENDOZA/tofu_1323'




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
_D12 = [0.3,0.1]
_nIn = [-0.5,-1.,0.]


_dcam = {'V1':       {'P':_P, 'F':_F, 'D12':_D12, 'nIn':_nIn, 'N12':[1,1]},
         'V10':      {'P':_P, 'F':_F, 'D12':_D12, 'nIn':_nIn, 'N12':[5,2]},
         'V100':     {'P':_P, 'F':_F, 'D12':_D12, 'nIn':_nIn, 'N12':[20,5]},
         'V1000':    {'P':_P, 'F':_F, 'D12':_D12, 'nIn':_nIn, 'N12':[50,20]},
         'V10000':   {'P':_P, 'F':_F, 'D12':_D12, 'nIn':_nIn, 'N12':[125,80]},
         'V100000':  {'P':_P, 'F':_F, 'D12':_D12, 'nIn':_nIn, 'N12':[500,200]},
         'V1000000': {'P':_P, 'F':_F, 'D12':_D12, 'nIn':_nIn, 'N12':[1600,625]}}


#########################################
#########################################
#       Reset objects
#########################################


def _recreate_compatible_objects(path=_path_laura_former, save=True):

    lf = os.listdir(path)
    lf = [f for f in lf if all([s in f for s in ['TFG_','.npz']])]
    for f in lf:
        cls, Exp, nn = f.split('_')[1:4]
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
            if len(nn)==2:
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
        Poly = out['Poly']
        Lim = out['Lim']
        print(Exp, cls, nn+v, Poly.shape, Lim.shape)
        if Lim.shape==():
            Lim = Lim.tolist()
        ss = eval("tf.geom.%s"%cls
                  +"(Name=nn+v, Exp=Exp, Poly=Poly, Lim=Lim, SavePath='./')")
        if save:
            ss.save()
        else:
            print("    Would be ",ss)

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
           config=None, path=_path_Objects):
    """ Get the (D,u) tuple for the desired camera

    Optionally create the camera with the chosen config
    Optionally plot
    """

    # Extract pinhole, focal length, width, nb. of pix., unit vector
    P, F = dcam[cam]['P'], dcam[cam]['F']
    D12, N12, nIn = dcam[cam]['D12'], dcam[cam]['N12'], dcam[cam]['nIn']

    # Compute the LOS starting points and unit vectors
    (D,u) = tf.utils.create_CamLOS2D(P, F, D12, N12, nIn=nIn)

    if make_cam or plot:
        assert config is not None, "You must specify a config !"

        # Load the config
        conf = load_config(config, path=path, plot=False)

        # Create the LOSCam2D object
        # Note : thsis is where the computation goes on...
        cam = tf.geom.LOSCam2D(Exp=conf.Id.Exp, Name=cam, dgeom=(D,u),
                               config=conf, Diag='Test', method="optimized")

    else:
        cam = None

    if plot:
        # Plot
        cam.plot_touch()

    return D, u, cam
