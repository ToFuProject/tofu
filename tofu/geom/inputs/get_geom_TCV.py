
import sys
import os
import scipy.io as scpio


import numpy as np


_HERE = os.path.abspath(os.path.dirname(__file__))
_TOFU_PATH = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))


# import parser dict
sys.path.insert(1, _TOFU_PATH)
import tofu.geom as tfg
_ = sys.path.pop(1)


def get_from_mat_file_from_tcv_modified(pfe=None, Exp='TCV'):

    import tofu.geom as tfg

    # Check inputs
    c0 = (os.path.isfile(pfe)
          and pfe[-4:] == '.mat')
    if c0 is not True:
        msg = ("File does not exist or not a .mat file!\n"
               + "\t- pfe: {}\n".format(pfe))
        raise Exception(msg)

    # Extract data
    out = scpio.loadmat(pfe)

    # Check file conformity
    lka = ['h2']
    lkb = [k0 for k0 in out.keys() if '__' not in k0]
    if lka != lkb:
        msg = ("Content of file not as expected!\n"
               + "\t- pfe: {}\n".format(pfe)
               + "\t- expected keys: {}\n".format(lka)
               + "\t- observed keys: {}".format(lkb))
        raise Exception(msg)

    # Create objects
    ls = []
    out = out[lka[0]]
    for ii in range(out.shape[1]):
        name = out[0][ii][0][0].replace('_', '').replace('-', '')
        if name == 'Bcoil':
            continue
        if name[0] in ['A', 'B', 'C', 'D', 'E', 'F', 'T']:
            poly = out[0][ii][1].T
            obj = tfg.CoilPF(Name=name, Poly=poly, Exp=Exp)
        elif name == 'v':
            if 512 not in out[0][ii][1].shape:
                msg = ("Size of {} not as expected!\n".format(name)
                       + "\t- expected: 512 pts\n"
                       + "\t- observed: {} pts".format(out[0][ii][1].size/2))
                raise Exception(msg)
            iin = np.r_[0, np.arange(512)]
            iout = np.arange(1, 257)
            poly = out[0][ii][1][iin, :].T
            obj = tfg.Ves(Name=name, Poly=poly, Exp=Exp)
        elif name == 't':
            if 314 not in out[0][ii][1].shape:
                msg = ("Size of {} not as expected!\n".format(name)
                       + "\t- expected: 314 pts\n"
                       + "\t- observed: {} pts".format(out[0][ii][1].size/2))
                raise Exception(msg)
            ind = np.arange(57)
            poly = out[0][ii][1][ind, :].T
            obj = tfg.Ves(Name=name, Poly=poly, Exp=Exp)
        else:
            msg = "Unidentified element {}".format(name)
            raise Exception(msg)
        ls.append(obj)
    return tfg.Config(lStruct=ls, Name='V1', Exp=Exp)
