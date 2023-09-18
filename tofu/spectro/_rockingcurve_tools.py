# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 16:34:30 2023

@author: dvezinet
"""

import itertools as itt


import numpy as np
from scipy.interpolate import interp1d
import datastock as ds
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


from . import _rockingcurve as rc


__all__ = [
    'scope_reflected_intensities',
]


_DMARKER = {
    'Germanium': {'marker': 'o', 'ms': 8, 'mfc': 'None', 'lw': 2},
    'Quartz': {'marker': 'x', 'ms': 8, 'lw': 2},
}


###########################################################
#
#           Scope over integrated reflectivity
#
###########################################################


def scope_reflected_intensities(
    miller_max=None,
    target_lamb=None,
    materials=None,  
    # plotting
    dmarker=None,
    tit=None,
):
    
    """
    
    Provide target_lamb in m
    
    """

    # -------------------
    # check input
    # -------------------
    
    # miller_max
    miller_max = int(ds._generic_check._check_var(
        miller_max, 'miller_max',
        types=(int, float),
        default=10,
        sign='>0',
    ))
    
    # materials
    lok = ['Quartz', 'Germanium']
    if isinstance(materials, str):
        materials = [materials]
    materials = list(ds._generic_check._check_var_iter(
        materials, 'materials',
        types=(list, tuple),
        types_iter=str,
        default=lok,
        allowed=lok,
    ))
    
    # target_bragg
    target_lamb = ds._generic_check._check_flat1darray(
        target_lamb, 'target_lamb',
        sign='>0',
    )
    
    # dmarker
    if dmarker is None:
        dmarker = _DMARKER
        
    # tit
    tit = ds._generic_check._check_var(
        tit, 'tit',
        types=str,
        default="Scoping integrated reflectivity",
    )
    
    # -------------------
    # Prepare
    # -------------------
    
    ddd = miller_max + 1
    
    dout = {
        'materials': materials,
        'target_lamb': target_lamb,
        'miller_max': miller_max,
        'reflectivity': {
            k0: {
                k1: {
                    'Bragg': np.full((ddd, ddd, ddd), np.nan),
                    'Rint':  np.full((ddd, ddd, ddd), np.nan),
                    'miller': np.full((ddd, ddd, ddd), 'hkl', dtype='S3')
                }
                for k1 in target_lamb
            }
            for k0 in materials
        },
    }    

    # miller indices ranges
    rangeh = np.arange(0, miller_max + 1)
    rangek = np.arange(0, miller_max + 1)
    rangel = np.arange(0, miller_max + 1)

    # -------------------
    # computing
    # -------------------
    
    # Loop over materials
    for lamb0, mat in itt.product(target_lamb, materials):
        results = dout['reflectivity'][mat][lamb0]
    
        # Loop over Miller indices
        for hh, kk, ll in itt.product(rangeh, rangek, rangel):
        
            # skip bulk reflection
            if hh == 0 and kk == 0 and ll == 0:
                continue
 
            # Builds material matrix
            dcry4 = {
                'material': mat,
                'name': 'SCOPING',
                'symbol': f"{mat[:2]}{hh}{kk}{ll}",
                'miller': np.r_[hh, kk, ll],
                'target': {
                    # 'ion': 'Kr34+',
                    'lamb': lamb0 * 1e10,
                    'units': 'A',
                    },
                'd_hkl': None,
            }
 
            results['miller'][hh, kk, ll] = f"{hh}{kk}{ll}"
 
            # Calculates rocking curve
            try:
                
                dout4 = rc.compute_rockingcurve(
                    crystal=dcry4['name'],
                    din=dcry4,
                    lamb=dcry4['target']['lamb'], 
                    plot_power_ratio=False,
                )
    
                results['Bragg'][hh,kk,ll] = dout4['Bragg angle of reference (rad)']*180/np.pi
    
                results['Rint'][hh,kk,ll] = np.trapz(
                    np.nanmean(
                        [
                            dout4['Power ratio'][0,0,0,:],
                            interp1d(
                                dout4['Glancing angles'][1,0,0,:],
                                dout4['Power ratio'][1,0,0,:],
                                bounds_error=False,
                                fill_value = (
                                    dout4['Power ratio'][1,0,0,0],
                                    dout4['Power ratio'][1,0,0,-1],
                                )
                            )(dout4['Glancing angles'][0,0,0,:]),
                        ],
                        axis=0,
                    ),
                    dout4['Glancing angles'][0,0,0,:],
                )

            except:
                pass
 
    # -------------------
    # plotting
    # -------------------
        
    # figure
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(tit, size=14, fontweight='bold')
    
    # axes
    ax = fig.add_axes([0.1, 0.1, 0.7, 0.8], yscale='log')
    ax.set_xlabel(r'$\theta_B$ [deg]', size=12)
    ax.set_ylabel('Integrated reflectivity [rad]', size=12)
    ax.set_title(f"max miller index = {miller_max}", size=12, fontweight='bold')
    ax.grid('on')
    
    # plotting
    dcolor = {lamb0: None for lamb0 in target_lamb}
    for lamb0 in target_lamb:
        
        for ii, mat in enumerate(materials):
            
            li, = ax.plot(
                dout['reflectivity'][mat][lamb0]['Bragg'].flatten(),
                dout['reflectivity'][mat][lamb0]['Rint'].flatten(),
                ls='None',
                label=mat,
                color=dcolor[lamb0],
                **dmarker[mat],
            )
            if ii == 0:
                dcolor[lamb0] = li.get_color()
    
    # -----------------------
    # legend proxys
    
    # legend proxys for lamb
    lh = [
        mlines.Line2D([], [], color=dcolor[lamb0], ls='-')
        for lamb0 in target_lamb
    ]
    llab = [f"{lamb0*1e10: 5.3} AA" for lamb0 in target_lamb]
    legend0 = ax.legend(
        lh,
        llab,
        bbox_to_anchor=(1.05, 1),
        loc='upper left', 
        borderaxespad=0.,
    )
    ax.add_artist(legend0)
    
    # legend proxys for material
    lh = [
        mlines.Line2D([], [], color='k', ls='None', **dmarker[mat])
        for mat in materials
    ]
    ax.legend(
        lh,
        materials,
        bbox_to_anchor=(1.05, 0),
        loc='lower left', 
        borderaxespad=0.,
    )
    
    # -------------------
    # return
    # -------------------

    return dout, ax
