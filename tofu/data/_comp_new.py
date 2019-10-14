
# -*- coding: utf-8 -*-

# Builtin
import warnings

# Common
import numpy as np
# import scipy.signal as scpsig
# import scipy.interpolate as scpinterp
# import scipy.linalg as scplin
# import scipy.stats as scpstats




#############################################
#############################################
#############################################
#       1d fits
#############################################
#############################################

def fit_1d(data, x=None, axis=None, Type=None, func=None,
           dTypes=None, **kwdargs):

    lc = [Type is not None, func is not None]
    assert np.sum(lc) == 1


    if lc[0]:


        # Pre-defined models dict
        # ------------------------

        _DTYPES = {'staircase': _fit1d_staircase}

        # Use a pre-defined model
        # ------------------------

        if dTypes is None:
            dTypes = _DTYPES

        if Type not in dTypes.keys():
            msg = "Chosen Type not available:\n"
            msg += "    - provided: %s\n"%str(Type)
            msg += "    - Available: %s"%str(list(dTypes.keys()))
            raise Exception(msg)

        dout = dTypes[Type](data, x=x, axis=axis, **kwdargs)

    else:

        # Use a user-provided model
        # -------------------------

        dout = func(data, x=x, axis=axis, **kwdargs)

    return dout



# -------------------------------------------
#       1d fit models
# -------------------------------------------

def _fit1d_staircase(data, x=None, axis=None):
    """ Model data as a staircase (ramps + plateaux)

    Return a the fitted parameters as a dict:
        {'plateaux': {'plateaux': {'Dt': (2,N) np.ndarray
                                   '':
                                    }}
        - to be discussed.... ?

    """

    pass
