#

# Built-in
import os

# Common
import numpy as np
import matplotlib.pyplot as plt

# tofu-specific
import tofu as tf



# ------------------
# Default parameters

_path = os.path.abspath(os.path.dirname(__file__))


#########################################
#########################################
#       Define config dictionary
#########################################

_dconfig = {'A1': {'Ves': ['WEST','V1']},
            'A2': {'Ves': ['ITER','Test']},
            'A3': {'Ves': ['WEST','Sep']},
            'B1': {'Ves': ['WEST','V2'],
                   'Struct': {'Baffle': ['Baffle','V0'],
                              'UpDiv':  ['UpDiv','V1'],
                              'LowDiv': ['LowDiv','V1']}},
            'B2': {'Ves': ['WEST','V2'],
                   'Struct': {'Baffle': ['Baffle','V1'],
                              'UpDiv':  ['UpDiv','V2'],
                              'LowDiv': ['LowDiv','V2'],
                              'InBump': ['InnerBumper','V1'],
                              'OutBump':['OuterBumper','V1'],
                              'IC1':    ['IC1','V1'],
                              'IC2':    ['IC2','V1'],
                              'IC3':    ['IC3','V1']}},
            'B3': {'Ves': ['WEST','V2'],
                   'Struct': {'Baffle': ['Baffle','V2'],
                              'UpDiv':  ['UpDiv','V3'],
                              'LowDiv': ['LowDiv','V3'],
                              'InBump': ['InnerBumper','V3'],
                              'OutBump':['OuterBumper','V3'],
                              'IC1':    ['IC1','V1'],
                              'IC2':    ['IC2','V1'],
                              'IC3':    ['IC3','V1'],
                              'LH1':    ['LH1','V1'],
                              'LH2':    ['LH2','V1'],
                              'Ripple': ['Ripple','V1'],
                              'VDE':    ['VDE','V0']}}}

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
#       Loading routine
#########################################

def _get_filenames(dconf, path=_path):
    """ Preliminary hidden routine for getting file names of desired objects
    """

    # Get all files avbailable in path
    lf = os.listdir(path)

    # Keep only those that are tofu.geom objects
    lf = [ff for ff in lf if all([ss in ff for ss in ['TFG_','.npz']])]

    # Get the vessel object (mandatory)
    # Check there is only one matching file name
    f = [ff for ff in lf if all([ss in ff for ss in ['_Ves']+dconf['Ves']])]
    msg = "None / several matches for {0} in {1}:".format(dconf['Ves'], path)
    msg += "\n    ".join([ff for ff in f])
    assert len(f)==1, msg

    dout = {'Ves':f[0]}

    if 'Struct' in dconf.keys():
        dout['Struct'] = {}
        for kk in dconf['Struct'].keys():
            f = [ff for ff in lf
                 if all([ss in ff for ss in ['_Struct']+dconf['Struct'][kk]])]
            msg = "None / several matches for {0} in {1}:".format(kk, path)
            msg += "\n    ".join([ff for ff in f])
            assert len(f)==1, msg
            dout['Struct'][kk] = f[0]
    return dout


def load_config(config, path=_path, dconfig=_dconfig, plot=True):
    """ Load all objects in the desired configuration

    Return them as a dictionary
    """
    assert type(config) is str
    assert type(path) is str
    assert type(dconfig) is dict
    assert type(plot) is bool

    # Get file names from config
    dout = _get_filenames(dconfig[config], path=_path)

    # Load Ves object
    dout['Ves'] = tf.pathfile.Open(os.path.join(path,dout['Ves']))

    # Load Struct objects
    if 'Struct' in dout.keys():
        for kk in dout['Struct'].keys():
            pathfile = os.path.join(path,dout['Struct'][kk])
            dout['Struct'][kk] = tf.pathfile.Open(pathfile)

    # ------------------
    # Optionnal plotting
    if plot:
        axC, axH = dout['Ves'].plot(Elt='P')
        if 'Struct' in dout.keys():
            for kk in dout['Struct'].keys():
                axC, axH = dout['Struct'][kk].plot(Lax=[axC,axH], Elt='P')

    return dout


def get_Du(cam, dcam=_dcam, plot=False, config=None, path=_path):
    """ Get the (D,u) tuple for the desired camera

    Optionally plot the camera with the chosen config
    """

    P, F = dcam[cam]['P'], dcam[cam]['F']
    D12, N12, nIn = dcam[cam]['D12'], dcam[cam]['N12'], dcam[cam]['nIn']

    (D,u) = tf.utils.create_CamLOS2D(P, F, D12, N12, nIn=nIn)

    if plot:
        assert config is not None, "You must specify a config !"

        # Load the config
        dconf = load_config(config, path=path, plot=False)
        if 'Struct' in dconf.keys():
            LStruct = list(dconf['Struct'].values())
        else:
            LStruct = None

        # Create the LOSCam2D object
        Cam = tf.geom.LOSCam2D(Id=cam, Du=(D,u), Ves=dconf['Ves'], LStruct=LStruct)

        # Plot
        Cam.plot(Elt='L', EltVes='P', EltStruct='P')

    return D, u



