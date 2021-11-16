#!/usr/bin/env python
# encoding: utf-8


# Built-in
import os


# Common
import numpy as np
import matplotlib.pyplot as plt


# tofu 
import tofu as tf



_PATH_TOFU = os.path.dirname(os.path.dirname(tf.__file__))
_PATH_INPUTS = os.path.join(_PATH_TOFU, 'inputs_temp')
_PFE_CRYST = os.path.join(
    _PATH_INPUTS,
    'TFG_CrystalBragg_ExpWEST_DgXICS_ArXVII_sh00000_Vers1.4.15-112-gddc0126a.npz',
)
_PFE_DET = os.path.join(
    _PATH_INPUTS,
    'det37_CTVD_incC4_New.npz',
)



# #############################################################################
# 
# #############################################################################


def fig00_plasma(
    conf='WEST',
    pfe_cryst=_PFE_CRYST,
    pfe_det=_PFE_DET,
):

    # ------------
    # load objects

    conf0 = tf.load_config(f'{conf}-V0')
    conf = tf.load_config(conf)

    cryst = tf.load(pfe_cryst)
    det = dict(np.load(pfe_det, allow_pickle=True))

    # ------------
    # sample volume


    return dax
