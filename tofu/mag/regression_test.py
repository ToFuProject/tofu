# -*- coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
# Also if needed: retab
'''
    Regression test
'''
from __future__ import (unicode_literals, absolute_import,  \
                        print_function, division)
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
import sys
import time

#print('path 1 =', sys.path)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#print('path 2 =', sys.path)

# Local modules
import equimap
import imas

# REFERENCE FILE !!!
# ==================
REF_FILE    = 'reference.npz'
REF_SHOT    = 54178
REF_RUN     = 9
REF_OCC     = 0
REF_USER    = 'imas_public'
REF_MACHINE = 'west'
# ==================

# Parameters
interp_points = 30
eps_time      = 1.23456789E-2
lquantities   = ('rho_pol_norm', 'rho_tor_norm', 'rho_tor', 'psi', 'phi', \
                 'theta', 'j_tor', 'b_field_r', 'b_field_z', 'b_field_tor', \
                 'b_field_norm')

def eval_diff(data, data_ref, name, rel_tolerance=1E-10):
    '''
        Function
        --------
        eval_diff(data, data_ref, name='data', rel_tolerance=1E-10)

        Output
        ------
        print the maximum and the maximum index of difference, displays
        an error if the maximum is above the given relative tolerance
    '''

    data     = np.asarray(data)
    data_ref = np.asarray(data_ref)

    if (data.shape != data_ref.shape):
        raise ValueError('Shape of input data is not equal')

    rel_diff         = np.abs( (data - data_ref) / data_ref )
    max_rel_diff     = np.nanmax(rel_diff)
    if (rel_diff.ndim != 0):
        ind_max_rel_diff = np.unravel_index(np.nanargmax(rel_diff), rel_diff.shape)
    else:
        ind_max_rel_diff = 0

    if (max_rel_diff > rel_tolerance):
        raise ValueError('ERROR test in: ' + name + ', max relative difference = '
              + '{0} at index = {1}'.format(max_rel_diff, ind_max_rel_diff))

    print('')
    print('In field name: ' + name + ', max relative difference = '
          + '{0} at index = {1}'.format(max_rel_diff, ind_max_rel_diff))
    print('')


if __name__ == '__main__':

    print(' ')
    # Parse input arguments
    parser = argparse.ArgumentParser(description= \
    '''Run regression EQUIMAP test using REF_FILE = {0}; REF_SHOT = {1};
       REF_RUN  = {2}; REF_OCC  = {3}; REF_USER = {4}; REF_MACHINE = {5}
    '''.format(REF_FILE, REF_SHOT, REF_RUN, REF_OCC, REF_USER, REF_MACHINE))
    # To exclude 2 conflict options use:
    #group = parser.add_mutually_exclusive_group()
    #parser.add_argument('shot', type=int, nargs='?', default=53259, help='shot, default=53259')

    parser.add_argument('--saveFile', action='store_true', \
           help='saves a Python .npz file')
    parser.add_argument('--figures', action='store_true', \
           help='plot figures')
    parser.add_argument('--no-git-check', action='store_true', \
           help='no check for changes that are not commited')

    args = parser.parse_args()

    print('REF FILE =', REF_FILE)
    print(' ')

    if (not args.no_git_check):
        try:
            subprocess.run(['git', 'diff', '--exit-code', '--quiet'], check=True)
            subprocess.run(['git', 'diff', '--cached', '--exit-code', '--quiet'], check=True)
        except subprocess.CalledProcessError as err:
            print(' ')
            print('ERROR: not commited changes, please commit the changes.', err)
            print(' ')
            raise

    # Initialize dictionary to store results
    results = {}

    idd = imas.ids(REF_SHOT, REF_RUN)
    idd.open_env(REF_USER, REF_MACHINE, '3')
    if (REF_OCC == 0):
        idd.equilibrium.get()
    else:
        idd.equilibrium.get(REF_OCC)
    equi = idd.equilibrium

    # Test one time and spatial 3D
    # ----------------------------
    time_in = eps_time + 0.5*(np.nanmax(equi.time) + np.nanmin(equi.time))

    equi_grid = idd.equilibrium.grids_ggd[0].grid[0]
    NbrPoints = len(equi_grid.space[0].objects_per_dimension[0].object)
    equiDict = {}
    equiDict['r'] = np.full(NbrPoints, np.nan)
    equiDict['z'] = np.full(NbrPoints, np.nan)
    for ii in range(NbrPoints):
        equiDict['r'][ii] = equi_grid.space[0].objects_per_dimension[0]. \
                            object[ii].geometry[0]
        equiDict['z'][ii] = equi_grid.space[0].objects_per_dimension[0]. \
                            object[ii].geometry[1]

    R_in   = np.linspace(np.min(equiDict['r']), \
                         np.max(equiDict['r']), interp_points)
    Z_in   = np.linspace(np.min(equiDict['z']), \
                         np.max(equiDict['z']), interp_points)
    Phi_in = np.linspace(0, 2*np.pi/18, interp_points)

    R_in_tot   = np.tile(R_in, int(interp_points**2))
    Z_in_tot   = np.tile(np.repeat(Z_in, interp_points), interp_points)
    Phi_in_tot = np.repeat(Phi_in, int(interp_points**2))

    Rr = R_in_tot.reshape((interp_points, interp_points, interp_points))
    Zr = Z_in_tot.reshape((interp_points, interp_points, interp_points))

    for iquant in lquantities:
        start = time.time()
        #sys.stdout = open(os.devnull, 'w')
        out = equimap.get(REF_SHOT, time=time_in, R=R_in_tot, Phi=Phi_in_tot, \
                          Z=Z_in_tot, quantity=iquant, no_ripple=False, \
                          run=REF_RUN, occ=REF_OCC, user=REF_USER, \
                          machine=REF_MACHINE)
        #sys.stdout = sys.__stdout__
        end = time.time()
        print()
        print('====================================')
        print('time for', iquant, ' =', end - start)
        print('====================================')
        print()
        if (args.figures):
            outr = out.reshape((interp_points, interp_points, interp_points))
            plt.figure()
            plt.contourf(Rr[int(0.2*interp_points), :, :], \
                         Zr[int(0.2*interp_points), :, :], \
                         outr[int(0.2*interp_points), :, :])
            arg_time = np.argmin(np.abs(equi.time - time_in))
            plt.plot(np.squeeze(equi.time_slice[arg_time].boundary.outline.r), \
                     np.squeeze(equi.time_slice[arg_time].boundary.outline.z), \
                     linewidth=2, color='red')
            plt.plot(equi.time_slice[arg_time].global_quantities.magnetic_axis.r, \
                     equi.time_slice[arg_time].global_quantities.magnetic_axis.z, \
                     marker='+', color='red', markersize=20)
            plt.xlabel('R [m]')
            plt.ylabel('Z [m]')
            plt.title('{0} t={1:.2f}'.format(iquant, time_in))
            plt.colorbar()

        # Save results in dict
        results[iquant] = out
    # End loop on lquantities

    # Test large time and spatial 2D (R, Phi)
    # --------------------------------------
    # Check code.output_flag for data validity
    if (np.any(np.isnan(equi.code.output_flag))):
        mask = np.full(len(equi.time), True, dtype=bool)
    else:
        mask = np.asarray(equi.code.output_flag) >= 0
    time1   = 0.495*(np.nanmax(equi.time[mask]) + np.nanmin(equi.time[mask]))
    time2   = 0.505*(np.nanmax(equi.time[mask]) + np.nanmin(equi.time[mask]))
    mask_time_tmp = (equi.time[mask] >= time1) \
                  & (equi.time[mask] <= time2)
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
    time_in = np.linspace(time1, time2, time_points.size + 1)
    time_in += eps_time

    R_in   = np.linspace(np.min(equiDict['r']), \
                         np.max(equiDict['r']), interp_points)
    Phi_in = np.linspace(0, 2*np.pi/18, interp_points)

    R_in_tot   = np.tile(R_in, interp_points)
    Z_in_tot   = np.zeros(R_in_tot.shape)
    Phi_in_tot = np.repeat(Phi_in, interp_points)

    Rr   = R_in_tot.reshape((interp_points, interp_points))
    Phir = Phi_in_tot.reshape((interp_points, interp_points))

    arg_time = np.argmin(np.abs(equi.time - time_in[int(0.5*time_in.size)]))
    if (args.figures):
        mask_LFS  = (equi.time_slice[arg_time].boundary.outline.r > equi.time_slice[arg_time].global_quantities.magnetic_axis.r)
        indZ0_LFS = np.argmin(np.abs(equi.time_slice[arg_time].boundary.outline.z[mask_LFS]))
        mask_HFS  = (equi.time_slice[arg_time].boundary.outline.r < equi.time_slice[arg_time].global_quantities.magnetic_axis.r)
        indZ0_HFS = np.argmin(np.abs(equi.time_slice[arg_time].boundary.outline.z[mask_HFS]))

    for iquant in lquantities:
        start = time.time()
        #sys.stdout = open(os.devnull, 'w')
        out = equimap.get(REF_SHOT, time=time_in, R=R_in_tot, Phi=Phi_in_tot, \
                          Z=Z_in_tot, quantity=iquant, no_ripple=False, \
                          run=REF_RUN, occ=REF_OCC, user=REF_USER, \
                          machine=REF_MACHINE)
        #sys.stdout = sys.__stdout__
        end = time.time()
        print()
        print('====================================')
        print('time (large time input) for', iquant, ' =', end - start)
        print('Z_axis =', equi.time_slice[arg_time].global_quantities.magnetic_axis.z)
        print('====================================')
        print()
        if (args.figures):
            outr = out[int(0.5*out.shape[0])].reshape((interp_points, interp_points))
            plt.figure()
            plt.contourf(Rr[:, :], Phir[:, :], outr[:, :])
            plt.axvline(np.squeeze(equi.time_slice[arg_time].boundary.outline.r[mask_LFS][indZ0_LFS]), \
                        linewidth=2, color='red')
            plt.axvline(np.squeeze(equi.time_slice[arg_time].boundary.outline.r[mask_HFS][indZ0_HFS]), \
                        linewidth=2, color='red')
            plt.axvline(equi.time_slice[arg_time].global_quantities.magnetic_axis.r, \
                        linewidth=2, color='red', linestyle='--')
            plt.xlabel('R [m]')
            plt.ylabel('Phi [rad]')
            plt.title('{0} t={1:.2f}'.format(iquant, time_in[int(0.5*out.shape[0])]))
            plt.colorbar()

        # Save results in dict
        results[iquant + '_LT'] = out
    # End loop on lquantities

    if (args.saveFile):
        filename = 'reg_test_{0}_Run{1}_Occ{2}_User_{3}_Machine_{4}.npz'.format( \
                    REF_SHOT, REF_RUN, REF_OCC, REF_USER, REF_MACHINE)
        np.savez(filename, **results)

    if (args.figures):
        plt.show()

    ref = np.load(REF_FILE)

    for iquant in lquantities:
        eval_diff(results[iquant], ref[iquant], iquant)
        eval_diff(results[iquant + '_LT'], ref[iquant + '_LT'], iquant + '_LT')

    print()
    print('End regression test')
    print()
