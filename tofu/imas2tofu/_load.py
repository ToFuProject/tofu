# -*- coding: utf-8 -*-
#! /usr/bin/python
"""
Thie imas-compatibility module of tofu

loading routines

"""

# Standard
import numpy as np

# tofu-specific
try:
    import tofu.data as tfdata
    import tofu.imas2tofu._utils as _utils
    import tofu.imas2tofu._Plasma2DLoader as  _Plasma2DLoader
except Exception:
    from .. import data as tfdata
    from . import _utils
    from . import _Plasma2DLoader

#  imas-specific
import imas



__all__ = ['core_profile_1d']



def _get_nodes_core_profiles_dict():
    dprof = {'ne': {'nodes':['electrons','density'], 'units':r'$m^{-3}$'},
             'Te': {'nodes':['electrons','temperature'], 'units':r'$eV$'},
             'zeff':{'nodes':['zeff'], 'units':r'$a.u.$'}}
    return dprof

def _checkformat_lprof(lprof, dprof=None):

    if dprof is None:
        dprof = _get_nodes_core_profiles_dict()
    lkprof = sorted(dprof.keys())

    # check either list or str
    lc = [isinstance(lprof,list), isinstance(lprof,str)]
    if not any(lc):
        msg = "lprof must be either a str of list of str !"
        raise Exception(msg)

    # make list and get length
    if lc[1]:
        lprof = [lprof]
    nprof = len(lprof)

    # Check all desired profiles are available
    lc = [pp in lkprof for pp in lprof]
    if not all(lc):
        ls = [lprof[ii] for ii in range(0,nprof) if not lc[ii]]
        ls = "\n    - ".join(ls)
        lp = "\n    - ".join(lkprof)
        msg = "The following profiles are not available from imas2tofu:\n"
        msg += "    - " + ls
        msg += "Available profiles are:\n"
        msg += "    - " + lp
        raise Exception(msg)

    return lprof



#############################################################
#############################################################
#               Loading
#############################################################

def core_profile_1d(lprof, tlim=None,
                    user=None, shot=None, run=None, occ=None,
                    tokamak=None, version=None,
                    dids=None, profile=True, x='rho_tor_norm',
                    dids_eq=None, mapping=True,
                    Struct=None, res=0.01, resmode='abs',
                    dprof=None, out=object, plot=True, verb=True):
    """ Load the desired core_profile from an ids

    The ids is identified by a dict of identifiers (ids)
    The profile can be loaded only in the desired time interval
    The profile is returned as a tofu.DataCam1D instance or a dict

    Parameters
    ----------
    prof    :

    tlim    :

    dids    :

    dids_Eq :


    """

    # Checks
    if profile is False:
        msg = "The out can only be a dict if only the 2D mapping is returned!"
        assert out is dict, msg

    if mapping:
        msg = "A bounding Structure instance must be provided if mapping is True!"
        assert Struct is None, msg

    if verb:
        nsteps = 3 + np.sum([mapping,plot])

    #-----------------
    # Preformat inputs
    #-----------------

    if verb:
        print("(1/%s) Checking inputs..."%str(nsteps))

    # ids dict
    dids = _utils._get_defaults(user=user, shot=shot, run=run, occ=occ,
                                tokamak=tokamak, version=version, dids=dids)

    # Equilibrium ids dict: by default use the same ids
    if dids_eq is None:
        dids_eq = dict(dids)
    else:
        dids_eq = _utils._get_defaults(dids=dids_eq)

    # get dict of available profiles
    if dprof is None:
        dprof = _get_nodes_core_profiles_dict()

    # check lprof
    lprof = _checkformat_lprof(lprof, dprof=dprof)
    nprof = len(lprof)

    #-----------------
    # load ids
    #-----------------
    if verb:
        print("(2/%s) Opening ids and getting core_profiles..."%str(nsteps))

    ids = imas.ids(s=dids['shot'], r=dids['run'],
                   rs=dids['shotr'], rr=dids['runr'])
    ids.open_env(dids['user'], dids['tokamak'], dids['version'])
    ids.core_profiles.get()


    #-----------------
    # get each profile
    #-----------------
    if verb:
        print("(3/%s) Recovering time, grid and data..."%str(nsteps))


    # Prepare output dict
    dout = {'t':None, 'indt':None, 'x':None}

    # Get time
    t = np.asarray(ids.core_profiles.time).ravel()
    indt = np.arange(0,t.size)
    if tlim is not None:
        indt = indt[ tlim[0] <= t <= tlim[1] ]
        t = t[indt]
    nt = t.size

    # Get x
    nx = len(eval('ids.core_profiles.profiles_1d[0].grid.%s'%x))
    x_values = np.full((nt, nx), np.nan)

    # Prepare arrays
    ds = {}
    for pp in lprof:
        dout[pp] = {'data': np.full((nt, nx), np.nan), 'units':dprof[pp]['units']}
        ds[pp] = '.'.join(dprof[pp]['nodes'])

    # Get slices
    for ii in range(0,nt):
        p1d = ids.core_profiles.profiles_1d[indt[ii]]
        x_values[ii,:] = eval('p1d.grid.%s'%x)
        for pp in lprof:
            dout[pp]['data'][ii,:] = eval('p1d.'+ds[pp])

    dout.update( {'t':t, 'indt':indt, x:x_values} )


    #-----------------
    # get 2D mappings
    #-----------------

    if mapping:
        if verb:
            print("(4/%s) Recovering 2D mapping..."%str(nsteps))

        # sampling of the cross-section
        x1, x2 = Struct.get_sampleCross(mode='img')
        ptsRZ = np.array([np.tile(x1, x2.size),
                          np.repeat(x2, x1.size)])
        indok = Struct.isInside(ptsRZ, In='(R,Z)')

        # Computing of data
        data2D = np.full((nt, x1,x2), np.nan)

        # Get Equilibrium2D for position interpolation
        eq = _equilibrium.Equilibrium2D(didsEq, dquant={'ggd':['phi']})

        for pp in lprof:
            pass


            dout[pp]['dextra'] = {'map': {'t':t, 'data2D':data2D, 'extent':extent}}
    else:
        for pp in lprof:
            dout[pp]['dextra'] = {}

    ids.close()

    #-----------------
    # format output
    #-----------------
    # Making tofu.data objects if necessary
    if plot or out is object:
        lout = [None for pp in lprof]
        for ii in range(0,nprof):
            pp = lprof[ii]
            dlabels = {'data':{'units':dout[pp]['units'],
                               'name':pp},
                       'X':{'units':'a.u.',
                            'name':x}}
            lout[ii] = tfdata.DataCam1D(Exp=dids['tokamak'],
                                        Diag='core_profile',
                                        Name=pp,
                                        shot=dids['shot'],
                                        data=dout[pp]['data'],
                                        t=dout['t'],
                                        X=dout[x],
                                        dlabels=dlabels,
                                        dextra=dout[pp]['dextra'])

    # Optional plotting
    if plot:
        if verb:
            print("(%s/%s) Plotting..."%(str(nsteps),str(nsteps)))

        for oo in lout:
            kh = oo.plot()

    # Returning
    if out is object:
        if len(lout) == 1:
            lout = lout[0]
        return lout
    else:
        if len(dout.keys()) == 1:
            dout = list(dout.values())[0]
        return dout
