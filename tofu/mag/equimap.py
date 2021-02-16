# -*- coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
# Also if needed: retab
'''
    EQUIMAP tools module, functions ...
'''
# Standard python modules
from __future__ import (unicode_literals, absolute_import,  \
                        print_function, division)
import numpy as np
import os
import re
import scipy.interpolate as interpolate
import warnings
#import sys

# Local modules
import imas
try:
    import imas_west
except ImportError as err:
    pass
    print(err)
try:
    import pywed as pw
except ImportError as err:
    pass
    print(err)

# Project modules
try:
    import mag_ripple as mr
except ImportError as err:
    pass
    print(err)

__all__ = ['get']

# Parameters
min_Ip = 100000

def get(shot, time, R, Phi, Z, quantity, no_ripple=False, \
        run=0, occ=0, user='imas_public', machine='west'):
    '''

    Interpolation of the requested quantity for the list time at
    input coodinates R, Phi, Z

    WARNING: Ripple is only taken into account for 'b_field_norm', 'b_field_r',
             'b_field_z' and b_field_tor' quantities for the moment

    Coordinate convention used (COCOS 11, see Sauter, Medvedev, Comp.Phys.Com. 184, 2013)
    -------------------------------------------------------------------------------------
     Cylindrical |     Poloidal      | Phi from top  | theta (pol ang) from front |   psi
    ------------------------------------------------------------------------------------------
     (R, Phi, Z) | (rho, theta, Phi) | cnt-clockwise |         clockwise          | increasing

    Parameters
    ----------
    shot : int
        shot number
    time : list of floats [s]
        times where to perform interpolation (ABSOLUTE TIME, without t_ignitron)
    R : list of floats [m]
        big radius where to perform interpolation
    Phi : list of floats, same length as R [rad]
        toroidal angle where to perform interpolation
    Z : list of floats, same length as R [m]
        vertical coordinate where to perform interpolation
    quantity : string
        for which quantity perform interpolation. One of

          ``rho_pol_norm``
            poloidal flux coordinate (normalized)

          ``rho_tor_norm``
            toroidal flux coordinate (normalized)

          ``rho_tor``
            toroidal flux coordinate [m]

          ``psi``
            poloidal flux [Wb]

          ``phi``
            toroidal flux [Wb]

          ``theta``
            poloidal angle [rad] in the range [0, 2*pi[

          ``j_tor``
            toroidal current density [A.m^-2]

          ``j_parallel``
            parallel current density [A.m^-2]

          ``b_field_r``
            big radius (R) component of the poloidal magnetic field [T]

          ``b_field_z``
            vertical (Z) component of the poloidal magnetic field [T]

          ``b_field_tor``
            toroidal component of the magnetic field [T]

          ``b_field_norm``
            total magnetic field norm [T]

    no_ripple : boolean, optional (default=False)
        do not calculate magnetic ripple
    run : run number, optional (default=0)
    occ : occurrence number, optional (default=0)
    user : user name, optional (default=imas_public)
    machine : machine name, optional (default=west)

    Returns
    -------
    out : ndarray, shape (time, points)
        interpolated quantity for list time at points with coordinates R, Phi, Z

    '''

    #print('time       =', time)
    #print('R          =', R)
    #print('Phi        =', Phi)
    #print('Z          =', Z)
    print('quantity   =', quantity)

    # Check if shot exists
    run_number = '{:04d}'.format(run)
    shot_file  = os.path.expanduser('~' + user + '/public/imasdb/' + machine + \
                                    '/3/0/' + 'ids_' + str(shot) + run_number + \
                                    '.datafile')
    if (not os.path.isfile(shot_file)):
        raise FileNotFoundError('IMAS file does not exist')

    # Get equilibrium fast
    idd = imas.ids(shot, run)
    idd.open_env(user, machine, '3')
    idd.equilibrium.get()
    equi = idd.equilibrium

    # Check code.output_flag for data validity
    if (np.any(np.isnan(equi.code.output_flag))):
        mask = np.full(len(equi.time), True, dtype=bool)
    else:
        mask = np.asarray(equi.code.output_flag) >= 0

    bool_b_field = False
    if isinstance(quantity, list):
        ar_bool = np.full(len(quantity), False)
        for ii in range(len(quantity)):
            if (re.match('b_field_*', quantity[ii])):
                ar_bool[ii] = True
        if (np.any(ar_bool)):
            bool_b_field = True
    else:
        if (re.match('b_field_*', quantity)):
            bool_b_field = True

    if (bool_b_field):
        # Get Itor (current of toroidal coils, coils that produce the toroidal field)
        itor, t_itor = pw.tsbase(shot, 'gmag_itor', nargout=2)
        t_ignitron   = pw.tsmat(shot, 'IGNITRON|1')
        t_itor += t_ignitron[0]

        ismag, tsmag  = pw.tsbase(shot, 'SMAG_IP', nargout=2)
        t_smag_filter = tsmag[(abs(ismag[:, 0]*1000) > min_Ip), 0]

        t_mid = 0.5*(t_smag_filter[-1] - t_smag_filter[0]) \
              + t_ignitron[0]

        ind_mid = np.abs(equi.time[mask] - t_mid).argmin()
    else:
        itor       = None
        t_itor     = None
        t_ignitron = None
        ind_mid    = None

    ar_time = np.atleast_1d(np.squeeze(np.asarray([time])))
    ar_R    = np.atleast_1d(np.squeeze(np.asarray([R])))
    ar_Phi  = np.atleast_1d(np.squeeze(np.asarray([Phi])))
    ar_Z    = np.atleast_1d(np.squeeze(np.asarray([Z])))

    if (ar_time.size > 1):
        mask_time_tmp = (equi.time[mask] >= ar_time.min()) \
                      & (equi.time[mask] <= ar_time.max())
        indMin = np.abs(equi.time[mask] \
               - equi.time[mask][mask_time_tmp][0]).argmin()
        indMax = np.abs(equi.time[mask] \
               - equi.time[mask][mask_time_tmp][-1]).argmin()
        if (indMin == 0):
            indMinApply = indMin
        else:
            indMinApply = indMin - 1
        if (indMax == (equi.time[mask].size-1)):
            indMaxApply = indMax
        else:
            indMaxApply = indMax + 1
        mask_time     = (equi.time[mask] >= equi.time[mask][indMinApply]) \
                      & (equi.time[mask] <= equi.time[mask][indMaxApply])
        time_points   = equi.time[mask][mask_time]
        #firstSpaceInterp = True # For test
        if (ar_time.size > time_points.size):
            print('__________> First perform space interpolation')
            firstSpaceInterp = True # In this case is fast to interpolate first spatially
        else:
            firstSpaceInterp = False
    else:
        firstSpaceInterp = False
        mask_time        = None
        time_points      = None

    equiDict = {}

    # Declaration of arrays 2d plots
    equi_grid = idd.equilibrium.grids_ggd[0].grid[0]
    NbrPoints = len(equi_grid.space[0].objects_per_dimension[0].object)
    equiDict['r'] = np.full(NbrPoints, np.nan)
    equiDict['z'] = np.full(NbrPoints, np.nan)
    for ii in range(NbrPoints):
        equiDict['r'][ii] = equi_grid.space[0].objects_per_dimension[0]. \
                            object[ii].geometry[0]
        equiDict['z'][ii] = equi_grid.space[0].objects_per_dimension[0]. \
                            object[ii].geometry[1]

    ind_valid = np.argmax(mask)
    NbrProf    = len(equi.time_slice[ind_valid].profiles_1d.psi)
    equiDict['psi']         = np.full((len(equi.time), NbrPoints), np.nan)
    equiDict['phi']         = np.full((len(equi.time), NbrPoints), np.nan)
    equiDict['theta']       = np.full((len(equi.time), NbrPoints), np.nan)
    equiDict['j_tor']       = np.full((len(equi.time), NbrPoints), np.nan)
    equiDict['j_parallel']  = np.full((len(equi.time), NbrPoints), np.nan)
    equiDict['b_field_r']   = np.full((len(equi.time), NbrPoints), np.nan)
    equiDict['b_field_z']   = np.full((len(equi.time), NbrPoints), np.nan)
    equiDict['b_field_tor'] = np.full((len(equi.time), NbrPoints), np.nan)
    equiDict['prof_1d_psi'] = np.full((len(equi.time), NbrProf), np.nan)
    equiDict['prof_1d_rho_tor'] = np.full((len(equi.time), NbrProf), np.nan)
    equiDict['mag_axis_r']      = np.full(len(equi.time), np.nan)
    equiDict['mag_axis_z']      = np.full(len(equi.time), np.nan)
    for ii in range(len(equi.time)):
        equi_slice  = equi.time_slice[ii]
        equi_space  = equi.time_slice[ii].ggd[0]
        if (equi_space.psi):
            equiDict['psi'][ii]         = equi_space.psi[0].values
        if (equi_space.phi):
            equiDict['phi'][ii]         = equi_space.phi[0].values
        if (equi_space.theta):
            equiDict['theta'][ii]       = equi_space.theta[0].values
        if (equi_space.j_tor):
            equiDict['j_tor'][ii]       = equi_space.j_tor[0].values
        if (equi_space.j_parallel):
            equiDict['j_parallel'][ii]  = equi_space.j_parallel[0].values
        if (equi_space.b_field_r):
            equiDict['b_field_r'][ii]   = equi_space.b_field_r[0].values
        if (equi_space.b_field_z):
            equiDict['b_field_z'][ii]   = equi_space.b_field_z[0].values
        if (equi_space.b_field_tor):
            equiDict['b_field_tor'][ii] = equi_space.b_field_tor[0].values
        equiDict['prof_1d_psi'][ii]     = equi_slice.profiles_1d.psi
        equiDict['prof_1d_rho_tor'][ii] = equi_slice.profiles_1d.rho_tor
        equiDict['mag_axis_r'][ii]      = equi_slice.global_quantities.magnetic_axis.r
        equiDict['mag_axis_z'][ii]      = equi_slice.global_quantities.magnetic_axis.z

    points        = np.vstack((equiDict['r'], equiDict['z'])).transpose()
    interp_points = np.vstack((ar_R, ar_Z)).transpose()

    if isinstance(quantity, list):
        out = {}
        for iquant in quantity:
            out[iquant] = \
              interp_quantity(iquant, interp_points, points, time_points, equi, \
                              ar_time, ar_R, ar_Phi, ar_Z, mask, mask_time, \
                              firstSpaceInterp, itor, t_itor, t_ignitron, \
                              no_ripple, ind_mid, equiDict)
    else:
        out = interp_quantity(quantity, interp_points, points, time_points, equi, \
                              ar_time, ar_R, ar_Phi, ar_Z, mask, mask_time, \
                              firstSpaceInterp, itor, t_itor, t_ignitron, \
                              no_ripple, ind_mid, equiDict)
    return out


def interp_quantity(quantity, interp_points, points, time_points, equi, \
                    ar_time, ar_R, ar_Phi, ar_Z, mask, mask_time, \
                    firstSpaceInterp, itor, t_itor, t_ignitron, \
                    no_ripple, ind_mid, equiDict):

    value_interpolated = np.full((ar_time.size, ar_R.size), np.nan)
    if (firstSpaceInterp):
        value_interpSpace  = np.full((time_points.size, ar_R.size), np.nan)

    # Computation of requested quantities
    if (quantity == 'b_field_norm'):
        if (no_ripple):
            b_field_norm = np.sqrt(equiDict['b_field_r']**2. \
                                 + equiDict['b_field_z']**2. \
                                 + equiDict['b_field_tor']**2.)
            if (firstSpaceInterp):
                # Space interpolation
                for ii in range(time_points.size):
                    lin_intp = interpolate.LinearNDInterpolator(points, \
                                 b_field_norm[mask, :][mask_time][ii, :])
                    value_interpSpace[ii, :] = lin_intp.__call__(interp_points)
            else:
                # Time interpolation
                f_intp = interpolate.interp1d(equi.time[mask], \
                                              b_field_norm[mask, :], axis=0, \
                                              bounds_error=False)
        else: # B_norm with ripple calculation
            # Declaration arrays
            br_intp = np.full((ar_time.size, ar_R.size), np.nan)
            bt_intp = np.full((ar_time.size, ar_R.size), np.nan)
            bz_intp = np.full((ar_time.size, ar_R.size), np.nan)

            if (firstSpaceInterp):
                br_Sintp = np.full((time_points.size, ar_R.size), np.nan)
                bt_Sintp = np.full((time_points.size, ar_R.size), np.nan)
                bz_Sintp = np.full((time_points.size, ar_R.size), np.nan)
                # Space interpolation
                for ii in range(time_points.size):
                    lin_intp = interpolate.LinearNDInterpolator(points, \
                               equiDict['b_field_r'][mask, :][mask_time][ii, :])
                    br_Sintp[ii, :] = lin_intp.__call__(interp_points)
                    lin_intp = interpolate.LinearNDInterpolator(points, \
                               equiDict['b_field_tor'][mask, :][mask_time][ii, :])
                    bt_Sintp[ii, :] = lin_intp.__call__(interp_points)
                    lin_intp = interpolate.LinearNDInterpolator(points, \
                               equiDict['b_field_z'][mask, :][mask_time][ii, :])
                    bz_Sintp[ii, :] = lin_intp.__call__(interp_points)
                # Time interpolation
                f_intp = interpolate.interp1d(time_points, \
                                              br_Sintp, axis=0, \
                                              bounds_error=False)
                br_intp = np.atleast_2d(np.squeeze(f_intp(ar_time)))

                f_intp = interpolate.interp1d(time_points, \
                                              bt_Sintp, axis=0, \
                                              bounds_error=False)
                bt_intp = np.atleast_2d(np.squeeze(f_intp(ar_time)))

                f_intp = interpolate.interp1d(time_points, \
                                              bz_Sintp, axis=0, \
                                              bounds_error=False)
                bz_intp = np.atleast_2d(np.squeeze(f_intp(ar_time)))
            else:
                # Time interpolation
                f_intp_br = interpolate.interp1d(equi.time[mask], \
                              equiDict['b_field_r'][mask, :], axis=0, \
                              bounds_error=False)
                f_intp_bt = interpolate.interp1d(equi.time[mask], \
                              equiDict['b_field_tor'][mask, :], axis=0, \
                              bounds_error=False)
                f_intp_bz = interpolate.interp1d(equi.time[mask], \
                              equiDict['b_field_z'][mask, :], axis=0, \
                              bounds_error=False)

                br_intp_t = np.atleast_2d(np.squeeze(f_intp_br(ar_time)))
                bt_intp_t = np.atleast_2d(np.squeeze(f_intp_bt(ar_time)))
                bz_intp_t = np.atleast_2d(np.squeeze(f_intp_bz(ar_time)))
                # Space interpolation
                for ii in range(ar_time.size):
                    lin_intp = interpolate.LinearNDInterpolator(points, br_intp_t[ii])
                    br_intp[ii, :] = lin_intp.__call__(interp_points)

                    lin_intp = interpolate.LinearNDInterpolator(points, bt_intp_t[ii])
                    bt_intp[ii, :] = lin_intp.__call__(interp_points)

                    lin_intp = interpolate.LinearNDInterpolator(points, bz_intp_t[ii])
                    bz_intp[ii, :] = lin_intp.__call__(interp_points)

            # Interpolate current
            itor_intp_t = np.interp(ar_time, t_itor[:, 0], itor[:, 0])
            b0_intp_t   = np.interp(ar_time, equi.time[mask], \
                                    equi.vacuum_toroidal_field.b0[mask])

            # Compute reference vaccuum magnetic field
            bt_vac = equi.vacuum_toroidal_field.r0*b0_intp_t[:, np.newaxis] \
                   / ar_R[np.newaxis, :]

            # Compute magnetic field for given Phi
            br_ripple, bt_ripple, bz_ripple = mr.mag_ripple(ar_R, ar_Phi, \
                                                 ar_Z, itor_intp_t)

            # Check and correct Br if needed
            r_ax = equi.time_slice[ind_mid].global_quantities.magnetic_axis.r

            z_mid_ax = \
              equi.time_slice[ind_mid].global_quantities.magnetic_axis.z \
                     + 0.5*equi.time_slice[ind_mid].boundary.minor_radius

            ind_p = np.abs((equiDict['r'] - r_ax)**2. \
                         + (equiDict['z'] - z_mid_ax)**2).argmin()

            if (equiDict['b_field_r'][ind_mid, ind_p] > 0):
                br_intp *= -1
                warnings.warn('Correcting b_field_r in b_field_norm, for negative toroidal current COCOS 11')

            # Check and correct Bz if needed
            r_mid_ax = \
              equi.time_slice[ind_mid].global_quantities.magnetic_axis.r \
                     + 0.5*equi.time_slice[ind_mid].boundary.minor_radius
            z_ax = equi.time_slice[ind_mid].global_quantities.magnetic_axis.z

            ind_p = np.abs((equiDict['r'] - r_mid_ax)**2. \
                         + (equiDict['z'] - z_ax)**2).argmin()

            if (equiDict['b_field_z'][ind_mid, ind_p] < 0):
                bz_intp *= -1
                warnings.warn('Correcting b_field_z in b_field_norm, for negative toroidal current COCOS 11')

            # Check and correct Btor if needed
            ind_p = np.abs((equiDict['r'] - r_ax)**2. \
                         + (equiDict['z'] - z_ax)**2).argmin()

            if (equiDict['b_field_tor'][ind_mid, ind_p] > 0):
                bt_intp *= -1
                warnings.warn('Correcting b_field_tor in b_field_norm, for negative toroidal field COCOS 11')

            # Value interpolated
            value_interpolated = \
              np.sqrt((br_intp - br_ripple)**2. \
                    + (np.abs(bt_intp - bt_ripple) - np.abs(bt_vac))**2. \
                    + (bz_intp - bz_ripple)**2.)

            return np.squeeze(value_interpolated)

    elif (not no_ripple and re.match('b_field_*', quantity)):
            # Declaration arrays
            br_ripple = np.full((ar_time.size, ar_R.size), np.nan)
            bt_ripple = np.full((ar_time.size, ar_R.size), np.nan)
            bz_ripple = np.full((ar_time.size, ar_R.size), np.nan)

            if (firstSpaceInterp):
                # Space interpolation
                quant_mask = eval('equiDict["'+quantity+'"][mask, :][mask_time]')
                for ii in range(time_points.size):
                    lin_intp = interpolate.LinearNDInterpolator(points, \
                                                      quant_mask[ii, :])
                    value_interpSpace[ii, :] = lin_intp.__call__(interp_points)

                # Time interpolation
                f_intp = interpolate.interp1d(time_points, \
                                              value_interpSpace, axis=0, \
                                              bounds_error=False)
                value_interpolated = np.atleast_2d(np.squeeze(f_intp(ar_time)))
            else:
                # Time interpolation
                f_intp = interpolate.interp1d(equi.time[mask], \
                              eval('equiDict["'+quantity+'"][mask, :]'), axis=0, \
                              bounds_error=False)
                b_intp_t = np.atleast_2d(np.squeeze(f_intp(ar_time)))

                # Space interpolation
                for ii in range(ar_time.size):
                    lin_intp = interpolate.LinearNDInterpolator(points, b_intp_t[ii])
                    value_interpolated[ii, :] = lin_intp.__call__(interp_points)

            # Interpolate current
            itor_intp_t = np.interp(ar_time, t_itor[:, 0], itor[:, 0])

            # Compute magnetic field for given Phi
            br_ripple, bt_ripple, bz_ripple = mr.mag_ripple(ar_R, ar_Phi, \
                                                 ar_Z, itor_intp_t)

            if (quantity == 'b_field_r'):
                r_ax = equi.time_slice[ind_mid].global_quantities.magnetic_axis.r

                z_mid_ax = \
                  equi.time_slice[ind_mid].global_quantities.magnetic_axis.z \
                         + 0.5*equi.time_slice[ind_mid].boundary.minor_radius

                ind_p = np.abs((equiDict['r'] - r_ax)**2. \
                             + (equiDict['z'] - z_mid_ax)**2).argmin()

                if (equiDict['b_field_r'][ind_mid, ind_p] > 0):
                    value_interpolated *= -1
                    value_interpolated -= br_ripple
                    warnings.warn('Correcting b_field_r, for negative toroidal current COCOS 11')
                else:
                    value_interpolated -= br_ripple

            elif (quantity == 'b_field_z'):
                r_mid_ax = \
                  equi.time_slice[ind_mid].global_quantities.magnetic_axis.r \
                         + 0.5*equi.time_slice[ind_mid].boundary.minor_radius
                z_ax = equi.time_slice[ind_mid].global_quantities.magnetic_axis.z

                ind_p = np.abs((equiDict['r'] - r_mid_ax)**2. \
                             + (equiDict['z'] - z_ax)**2).argmin()

                if (equiDict['b_field_z'][ind_mid, ind_p] < 0):
                    value_interpolated *= -1
                    value_interpolated -= bz_ripple
                    warnings.warn('Correcting b_field_z, for negative toroidal current COCOS 11')
                else:
                    value_interpolated -= bz_ripple

            elif (quantity == 'b_field_tor'):
                b0_intp_t = np.interp(ar_time, equi.time[mask], \
                                      equi.vacuum_toroidal_field.b0[mask])
                # Compute reference vaccuum magnetic field
                bt_vac = equi.vacuum_toroidal_field.r0*b0_intp_t[:, np.newaxis] \
                       / ar_R[np.newaxis, :]

                r_ax = equi.time_slice[ind_mid].global_quantities.magnetic_axis.r
                z_ax = equi.time_slice[ind_mid].global_quantities.magnetic_axis.z

                ind_p = np.abs((equiDict['r'] - r_ax)**2. \
                             + (equiDict['z'] - z_ax)**2).argmin()

                if (equiDict['b_field_tor'][ind_mid, ind_p] > 0):
                    value_interpolated *= -1
                    value_interpolated -= (bt_ripple - bt_vac)
                    warnings.warn('Correcting b_field_tor, for negative toroidal field COCOS 11')
                else:
                    value_interpolated -= (bt_ripple - bt_vac)
            else:
                print()
                print('ERROR: not valid quantity input:', quantity)
                print()
                raise SyntaxError

            return np.squeeze(value_interpolated)

    elif (quantity == 'rho_pol_norm'):
        rho_pol_norm = np.sqrt((equiDict['psi'] \
                     - equiDict['prof_1d_psi'][:, 0, np.newaxis]) \
                / (equiDict['prof_1d_psi'][:, -1, np.newaxis] \
                 - equiDict['prof_1d_psi'][:, 0, np.newaxis]))
        if (firstSpaceInterp):
            # Space interpolation
            for ii in range(time_points.size):
                lin_intp = interpolate.LinearNDInterpolator(points, \
                             rho_pol_norm[mask, :][mask_time][ii, :])
                value_interpSpace[ii, :] = lin_intp.__call__(interp_points)
        else:
            # Time interpolation
            f_intp = interpolate.interp1d(equi.time[mask], rho_pol_norm[mask, :], \
                                          axis=0, bounds_error=False)

    elif (quantity == 'rho_tor_norm' or quantity == 'rho_tor'):
        if (firstSpaceInterp):
            # Space interpolation
            for ii in range(time_points.size):
                lin_intp = interpolate.LinearNDInterpolator(points, \
                             equiDict['psi'][mask, :][mask_time][ii, :])
                value_interpSpace[ii, :] = lin_intp.__call__(interp_points)
        else:
            # Time interpolation
            f_intp = interpolate.interp1d(equi.time[mask], equiDict['psi'][mask, :], \
                                          axis=0, bounds_error=False)

    elif (quantity == 'theta'):
        mag_ax_r = np.atleast_1d(np.squeeze(np.interp(ar_time, \
                     equi.time[mask], equiDict['mag_axis_r'][mask])))
        mag_ax_z = np.atleast_1d(np.squeeze(np.interp(ar_time, \
                     equi.time[mask], equiDict['mag_axis_z'][mask])))

        for ii in range(ar_time.size):
            delta_R = ar_R - mag_ax_r[ii, np.newaxis]
            delta_Z = ar_Z - mag_ax_z[ii, np.newaxis]

            val_arctan = np.arctan(delta_Z/delta_R)

            mask_theta = (delta_R >= 0) & (delta_Z >= 0)
            value_interpolated[ii, mask_theta] = val_arctan[mask_theta]

            mask_theta = (delta_R >= 0) & (delta_Z < 0)
            value_interpolated[ii, mask_theta] = 2*np.pi + val_arctan[mask_theta]

            mask_theta = (delta_R < 0)
            value_interpolated[ii, mask_theta] = np.pi + val_arctan[mask_theta]

        return np.squeeze(2.*np.pi - value_interpolated)
    else:
        if (firstSpaceInterp):
            # Space interpolation
            for ii in range(time_points.size):
                lin_intp = interpolate.LinearNDInterpolator(points, \
                  eval('equiDict["'+quantity+'"][mask, :][mask_time][ii, :]'))
                value_interpSpace[ii, :] = lin_intp.__call__(interp_points)
        else:
            # Time interpolation
            f_intp = interpolate.interp1d(equi.time[mask], \
                       eval('equiDict["'+quantity+'"][mask, :]'), \
                       axis=0, bounds_error=False)

    if (firstSpaceInterp):
        # Time interpolation
        f_intp = interpolate.interp1d(time_points, \
                                      value_interpSpace, axis=0, \
                                      bounds_error=False)
        value_interpolated = np.atleast_2d(np.squeeze(f_intp(ar_time)))
    else:
        # Time interpolation
        out_time_interp = np.atleast_2d(np.squeeze(f_intp(ar_time)))
        # Space interpolation
        for ii in range(ar_time.size):
            lin_intp = interpolate.LinearNDInterpolator(points, out_time_interp[ii])
            value_interpolated[ii, :] = lin_intp.__call__(interp_points)

    # Extra calculations for rho_tor, rho_tor_norm, b_field_r and b_field_z
    if (quantity == 'rho_tor_norm' or quantity == 'rho_tor'):

        value_interp_rho_tor = np.full(value_interpolated.shape, np.nan)

        if (quantity == 'rho_tor_norm'):
            try:
                rho_tor_norm = equiDict['prof_1d_rho_tor'] \
                             / equiDict['prof_1d_rho_tor'][:, -1, np.newaxis]
            except ZeroDivisionError as err:
                print('Division by zero for rho_tor_norm interpolation:', err)
                raise
            f_intp_rho_tor = interpolate.interp1d(equi.time[mask], \
                               rho_tor_norm[mask, :], axis=0, bounds_error=False)
        elif (quantity == 'rho_tor'):
            f_intp_rho_tor = interpolate.interp1d(equi.time[mask], \
                                         equiDict['prof_1d_rho_tor'][mask, :], \
                                         axis=0, bounds_error=False)

        rho_tor1D = np.atleast_2d(np.squeeze(f_intp_rho_tor(ar_time)))

        f_intp_psi = interpolate.interp1d(equi.time[mask], \
                       equiDict['prof_1d_psi'][mask, :], axis=0, bounds_error=False)
        psi1D = np.atleast_2d(np.squeeze(f_intp_psi(ar_time)))

        for ii in range(ar_time.size):
            f_intp_psi_rho_tor = interpolate.interp1d(psi1D[ii], rho_tor1D[ii], \
                                                      bounds_error=False)
            value_interp_rho_tor[ii] = f_intp_psi_rho_tor(value_interpolated[ii])

        return np.squeeze(value_interp_rho_tor)

    elif (quantity == 'b_field_r'):
        r_ax = equi.time_slice[ind_mid].global_quantities.magnetic_axis.r

        z_mid_ax = equi.time_slice[ind_mid].global_quantities.magnetic_axis.z \
                 + 0.5*equi.time_slice[ind_mid].boundary.minor_radius

        ind_p = np.abs((equiDict['r'] - r_ax)**2. \
                     + (equiDict['z'] - z_mid_ax)**2).argmin()

        if (equiDict['b_field_r'][ind_mid, ind_p] > 0):
            value_interpolated *= -1
            warnings.warn('Correcting b_field_r, if Ip negative COCOS 11')

        return np.squeeze(value_interpolated)

    elif (quantity == 'b_field_z'):
        r_mid_ax = equi.time_slice[ind_mid].global_quantities.magnetic_axis.r \
                 + 0.5*equi.time_slice[ind_mid].boundary.minor_radius
        z_ax = equi.time_slice[ind_mid].global_quantities.magnetic_axis.z

        ind_p = np.abs((equiDict['r'] - r_mid_ax)**2. \
                     + (equiDict['z'] - z_ax)**2).argmin()

        if (equiDict['b_field_z'][ind_mid, ind_p] < 0):
            value_interpolated *= -1
            warnings.warn('Correcting b_field_z, if Ip negative COCOS 11')

        return np.squeeze(value_interpolated)

    elif (quantity == 'b_field_tor'):
        r_ax = equi.time_slice[ind_mid].global_quantities.magnetic_axis.r
        z_ax = equi.time_slice[ind_mid].global_quantities.magnetic_axis.z

        ind_p = np.abs((equiDict['r'] - r_ax)**2. \
                     + (equiDict['z'] - z_ax)**2).argmin()

        if (equiDict['b_field_tor'][ind_mid, ind_p] > 0):
            value_interpolated *= -1
            warnings.warn('Correcting b_field_tor, if b_field_tor negative COCOS 11')

        return np.squeeze(value_interpolated)
    else:
        return np.squeeze(value_interpolated)
