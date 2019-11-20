# -*- coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
# Also if needed: retab
'''
    TEST equimap
'''
from __future__ import (unicode_literals, absolute_import,  \
                        print_function, division)
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
#import warnings

if __name__ == '__main__':
    #print('path 1 =', sys.path)
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    #print('path 2 =', sys.path)

    # Local modules
    import imas
    import equimap
    #import imas_west
    #import pywed as pw

    shot    = 53221
    tol_val = 1E-10

    # For 2D plots
    interp_points = 60

    # FIRST POINT B_NORM
    # ------------------
    time_in = np.linspace(36, 37, 10)
    Phi_in  = np.linspace(0, 2*np.pi/18, 100)
    R_in    = np.full(Phi_in.shape, 3)
    Z_in    = np.zeros(R_in.shape)

    # Read equilibrium data
    idd = imas.ids(shot, 0)
    idd.open_env('imas_public', 'west', '3')
    idd.equilibrium.get()
    out = idd.equilibrium

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

    # For 2D plots
    R_all = np.linspace(np.min(equiDict['r']), np.max(equiDict['r']), interp_points)
    Z_all = np.linspace(np.min(equiDict['z']), np.max(equiDict['z']), interp_points)

    R_all_tot = np.repeat(R_all, interp_points)
    Z_all_tot = np.tile(Z_all, interp_points)

    Rr = R_all_tot.reshape((interp_points, interp_points))
    Zr = Z_all_tot.reshape((interp_points, interp_points))

    # CALL EQUIMAP
    start = time.time()
    oute = equimap.get(shot, time=time_in, \
            R=R_in, Phi=Phi_in, Z=Z_in, \
            quantity='b_field_norm')
    end = time.time()
    print()
    print('time in equimap.get b_norm =', end - start)
    print()
    print('oute.shape b_norm =', oute.shape)

    # CALL EQUIMAP
    start = time.time()
    oute_noR = equimap.get(shot, time=time_in, \
                 R=R_in, Phi=Phi_in, Z=Z_in, \
                 quantity='b_field_norm', no_ripple=True)
    end = time.time()
    print()
    print('time in equimap.get b_norm no Ripple =', end - start)
    print()
    print('oute.shape b_norm no ripple =', oute_noR.shape)

    print()
    print('Mean value B_norm ripple =', np.mean(oute[int(0.5*oute.shape[0]), :]))
    print('Mean value B_norm NO ripple =', \
                np.mean(oute_noR[int(0.5*oute_noR.shape[0]), :]))
    diff_mean_val = np.mean(oute[int(0.5*oute.shape[0]), :]) \
                  - np.mean(oute_noR[int(0.5*oute_noR.shape[0]), :])
    print('Diff mean values =', diff_mean_val)
    percent_diff = np.abs(100*diff_mean_val \
                 / np.mean(oute[int(0.5*oute.shape[0]), :]))
    print('Percent diff mean values =', percent_diff)

    # CHECK
    # -----
    if (np.abs(percent_diff - 0.011052598088) > tol_val):
        print()
        print('ERROR: Higher than tolerance percent difference ' \
                     + str(np.abs(percent_diff - 0.011052598088)))
        print()
        #raise RuntimeError
    # FOR:
    # shot    = 53221
    # time_in = np.linspace(36, 37, 10)
    # Phi_in  = np.linspace(0, 2*np.pi/18, 100)
    # R_in    = np.full(Phi_in.shape, 3)
    # Z_in    = np.zeros(R_in.shape)
    # RESULTS:
    # Mean value B_norm ripple = 3.05593472975
    # Mean value B_norm NO ripple = 3.05627248994
    # Diff mean values = -0.000337760183512
    # Percent diff mean values = 0.011052598088
    print()

    # PLOTS
    plt.figure()
    plt.plot(time_in, oute[:, -1], label='B_norm at R={0}, Phi=Z=0'.format(R_in[-1]))
    plt.plot(time_in, oute_noR[:, -1], label='B_norm no ripple at R={0}, Phi=Z=0'.format(R_in[-1]))
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('B_norm [T]')
    plt.figure()
    plt.plot(Phi_in, oute[int(0.5*oute.shape[0]), :], \
             label='B_norm at t={0:.2f}, R={1}, Z=0'.format( \
                   time_in[int(0.5*oute.shape[0])], R_in[-1]))
    plt.plot(Phi_in, oute_noR[int(0.5*oute.shape[0]), :], \
             label='B_norm no ripple at t={0:.2f}, R={1}, Z=0'.format( \
                   time_in[int(0.5*oute.shape[0])], R_in[-1]))
    plt.legend()
    plt.xlabel('Phi [rad]')
    plt.ylabel('B_norm [T]')

    # SECOND POSITION B_NORM
    # ----------------------
    t_ignitron = []
    t_ignitron.append(32)
    print()
    print('t_igni =', t_ignitron[0])
    print()
    time_in = np.linspace(t_ignitron[0], 38, 10)
    Phi_in = np.linspace(0, 2*np.pi/18, 100)
    R_in   = np.full(Phi_in.shape, 2.43)
    Z_in   = np.full(Phi_in.shape, 0.57)

    # CALL EQUIMAP
    start = time.time()
    oute = equimap.get(shot, time=time_in, \
            R=R_in, Phi=Phi_in, Z=Z_in, \
            quantity='b_field_norm')
    end = time.time()
    print()
    print('time in equimap.get 2 b_norm =', end - start)
    print()
    print('oute.shape 2 b_norm =', oute.shape)

    # CALL EQUIMAP
    start = time.time()
    oute_noR = equimap.get(shot, time=time_in, \
                 R=R_in, Phi=Phi_in, Z=Z_in, \
                 quantity='b_field_norm', no_ripple=True)
    end = time.time()
    print()
    print('time in equimap.get 2 b_norm no ripple =', end - start)
    print()
    print('oute.shape 2 b_norm no ripple =', oute_noR.shape)

    # PLOTS
    plt.figure()
    plt.plot(time_in, oute[:, -1], \
      label='B_norm at R={0}, Phi={1:.2f}, Z={2}'.format( \
            R_in[-1], Phi_in[-1], Z_in[-1]))
    plt.plot(time_in, oute_noR[:, -1], \
      label='B_norm no ripple at R={0}, Phi={1:.2f}, Z={2}'.format( \
            R_in[-1], Phi_in[-1], Z_in[-1]))
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('B_norm [T]')
    plt.figure()
    plt.plot(Phi_in, oute[int(0.5*oute.shape[0]), :], \
             label='B_norm at t={0:.2f}, R={1}, Z={2}'.format( \
                   time_in[int(0.5*oute.shape[0])], R_in[-1], Z_in[-1]))
    plt.plot(Phi_in, oute_noR[int(0.5*oute.shape[0]), :], \
             label='B_norm no ripple at t={0:.2f}, R={1}, Z={2}'.format( \
                   time_in[int(0.5*oute.shape[0])], R_in[-1], Z_in[-1]))
    plt.legend()
    plt.xlabel('Phi [rad]')
    plt.ylabel('B_norm [T]')

    # B_NORM 2D
    # ---------
    # CALL EQUIMAP
    start = time.time()
    outa = equimap.get(shot, time=time_in, \
            R=R_all_tot, Phi=np.zeros(R_all_tot.shape), Z=Z_all_tot, \
            quantity='b_field_norm')
    end = time.time()
    print()
    print('time in equimap.get b_norm 2D =', end - start)
    print()
    outar = outa[int(0.5*outa.shape[0])].reshape((interp_points, interp_points))
    plt.figure()
    plt.contourf(Rr, Zr, outar)
    plt.colorbar()
    arg_time = np.argmin(np.abs(out.time - time_in[int(0.5*outa.shape[0])]))
    plt.plot(np.squeeze(out.time_slice[arg_time].boundary.outline.r), \
             np.squeeze(out.time_slice[arg_time].boundary.outline.z), \
             linewidth=2, color='red')
    plt.plot(out.time_slice[arg_time].global_quantities.magnetic_axis.r, \
             out.time_slice[arg_time].global_quantities.magnetic_axis.z, \
             marker='+', color='red', markersize=20)
    plt.xlabel('R [m]')
    plt.ylabel('Z [m]')
    plt.title('B_norm t={0:.2f}'.format(time_in[int(0.5*outa.shape[0])]))

    # B_R TEST
    # --------
    Phi_in = np.linspace(0, 2*np.pi/18, 100)
    R_in   = np.full(Phi_in.shape, 3)
    Z_in   = np.full(Phi_in.shape, 0)

    # CALL EQUIMAP
    start = time.time()
    oute = equimap.get(shot, time=time_in, \
            R=R_in, Phi=Phi_in, Z=Z_in, \
            quantity='b_field_r')
    end = time.time()
    print()
    print('time in equimap.get br =', end - start)
    print()
    print('oute.shape br =', oute.shape)

    # CALL EQUIMAP
    start = time.time()
    oute_noR = equimap.get(shot, time=time_in, \
                 R=R_in, Phi=Phi_in, Z=Z_in, \
                 quantity='b_field_r', no_ripple=True)
    end = time.time()
    print()
    print('time in equimap.get br no ripple =', end - start)
    print()
    print('oute.shape br no ripple =', oute_noR.shape)

    # PLOTS
    plt.figure()
    plt.plot(time_in, oute[:, -1], \
      label='B_r at R={0}, Phi={1:.2f}, Z={2}'.format( \
            R_in[-1], Phi_in[-1], Z_in[-1]))
    plt.plot(time_in, oute_noR[:, -1], \
      label='B_r no ripple at R={0}, Phi={1:.2f}, Z={2}'.format( \
            R_in[-1], Phi_in[-1], Z_in[-1]))
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('B_r [T]')

    plt.figure()
    plt.plot(Phi_in, oute[int(0.5*oute.shape[0]), :], \
             label='B_r at t={0:.2f}, R={1}, Z={2}'.format( \
                   time_in[int(0.5*oute.shape[0])], R_in[-1], Z_in[-1]))
    plt.plot(Phi_in, oute_noR[int(0.5*oute.shape[0]), :], \
             label='B_r no ripple at t={0:.2f}, R={1}, Z={2}'.format( \
                   time_in[int(0.5*oute.shape[0])], R_in[-1], Z_in[-1]))
    plt.legend()
    plt.xlabel('Phi [rad]')
    plt.ylabel('B_r [T]')

    # CALL EQUIMAP
    start = time.time()
    outa = equimap.get(shot, time=time_in, \
            R=R_all_tot, Phi=np.zeros(R_all_tot.shape), Z=Z_all_tot, \
            quantity='b_field_r')
    end = time.time()
    print()
    print('time in equimap.get br 2D =', end - start)
    print()
    outar = outa[int(0.5*outa.shape[0])].reshape((interp_points, interp_points))
    plt.figure()
    plt.contourf(Rr, Zr, outar)
    plt.colorbar()
    arg_time = np.argmin(np.abs(out.time - time_in[int(0.5*outa.shape[0])]))
    plt.plot(np.squeeze(out.time_slice[arg_time].boundary.outline.r), \
             np.squeeze(out.time_slice[arg_time].boundary.outline.z), \
             linewidth=2, color='red')
    plt.plot(out.time_slice[arg_time].global_quantities.magnetic_axis.r, \
             out.time_slice[arg_time].global_quantities.magnetic_axis.z, \
             marker='+', color='red', markersize=20)
    plt.xlabel('R [m]')
    plt.ylabel('Z [m]')
    plt.title('B_r t={0:.2f}'.format(time_in[int(0.5*outa.shape[0])]))

    # B_Z TEST
    # --------
    Phi_in = np.linspace(0, 2*np.pi/18, 100)
    R_in   = np.full(Phi_in.shape, 3)
    Z_in   = np.full(Phi_in.shape, 0.2)

    # CALL EQUIMAP
    start = time.time()
    oute = equimap.get(shot, time=time_in, \
            R=R_in, Phi=Phi_in, Z=Z_in, \
            quantity='b_field_z')
    end = time.time()
    print()
    print('time in equimap.get bz =', end - start)
    print()
    print('oute.shape bz =', oute.shape)

    # CALL EQUIMAP
    start = time.time()
    oute_noR = equimap.get(shot, time=time_in, \
                 R=R_in, Phi=Phi_in, Z=Z_in, \
                 quantity='b_field_z', no_ripple=True)
    end = time.time()
    print()
    print('time in equimap.get bz no ripple =', end - start)
    print()
    print('oute.shape bz no ripple =', oute_noR.shape)

    # PLOTS
    plt.figure()
    plt.plot(time_in, oute[:, -1], \
      label='B_z at R={0}, Phi={1:.2f}, Z={2}'.format( \
            R_in[-1], Phi_in[-1], Z_in[-1]))
    plt.plot(time_in, oute_noR[:, -1], \
      label='B_z no ripple at R={0}, Phi={1:.2f}, Z={2}'.format( \
            R_in[-1], Phi_in[-1], Z_in[-1]))
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('B_z [T]')

    plt.figure()
    plt.plot(Phi_in, oute[int(0.5*oute.shape[0]), :], \
             label='B_z at t={0:.2f}, R={1}, Z={2}'.format( \
                   time_in[int(0.5*oute.shape[0])], R_in[-1], Z_in[-1]))
    plt.plot(Phi_in, oute_noR[int(0.5*oute.shape[0]), :], \
             label='B_z no ripple at t={0:.2f}, R={1}, Z={2}'.format( \
                   time_in[int(0.5*oute.shape[0])], R_in[-1], Z_in[-1]))
    plt.legend()
    plt.xlabel('Phi [rad]')
    plt.ylabel('B_z [T]')

    # CALL EQUIMAP
    start = time.time()
    outa = equimap.get(shot, time=time_in, \
            R=R_all_tot, Phi=np.zeros(R_all_tot.shape), Z=Z_all_tot, \
            quantity='b_field_z')
    end = time.time()
    print()
    print('time in equimap.get bz 2D =', end - start)
    print()
    outar = outa[int(0.5*outa.shape[0])].reshape((interp_points, interp_points))
    plt.figure()
    plt.contourf(Rr, Zr, outar)
    plt.colorbar()
    arg_time = np.argmin(np.abs(out.time - time_in[int(0.5*outa.shape[0])]))
    plt.plot(np.squeeze(out.time_slice[arg_time].boundary.outline.r), \
             np.squeeze(out.time_slice[arg_time].boundary.outline.z), \
             linewidth=2, color='red')
    plt.plot(out.time_slice[arg_time].global_quantities.magnetic_axis.r, \
             out.time_slice[arg_time].global_quantities.magnetic_axis.z, \
             marker='+', color='red', markersize=20)
    plt.xlabel('R [m]')
    plt.ylabel('Z [m]')
    plt.title('B_z t={0:.2f}'.format(time_in[int(0.5*outa.shape[0])]))

    # B_TOR TEST
    # ----------
    Phi_in = np.linspace(0, 2*np.pi/18, 100)
    R_in   = np.full(Phi_in.shape, 3)
    Z_in   = np.full(Phi_in.shape, 0)

    # CALL EQUIMAP
    start = time.time()
    oute = equimap.get(shot, time=time_in, \
            R=R_in, Phi=Phi_in, Z=Z_in, \
            quantity='b_field_tor')
    end = time.time()
    print()
    print('time in equimap.get btor =', end - start)
    print()
    print('oute.shape btor =', oute.shape)

    # CALL EQUIMAP
    start = time.time()
    oute_noR = equimap.get(shot, time=time_in, \
                 R=R_in, Phi=Phi_in, Z=Z_in, \
                 quantity='b_field_tor', no_ripple=True)
    end = time.time()
    print()
    print('time in equimap.get btor no ripple =', end - start)
    print()
    print('oute.shape btor no ripple =', oute_noR.shape)

    # PLOTS
    plt.figure()
    plt.plot(time_in, oute[:, -1], \
      label='B_tor at R={0}, Phi={1:.2f}, Z={2}'.format( \
            R_in[-1], Phi_in[-1], Z_in[-1]))
    plt.plot(time_in, oute_noR[:, -1], \
      label='B_tor no ripple at R={0}, Phi={1:.2f}, Z={2}'.format( \
            R_in[-1], Phi_in[-1], Z_in[-1]))
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('B_tor [T]')

    plt.figure()
    plt.plot(Phi_in, oute[int(0.5*oute.shape[0]), :], \
             label='B_tor at t={0:.2f}, R={1}, Z={2}'.format( \
                   time_in[int(0.5*oute.shape[0])], R_in[-1], Z_in[-1]))
    plt.plot(Phi_in, oute_noR[int(0.5*oute.shape[0]), :], \
             label='B_tor no ripple at t={0:.2f}, R={1}, Z={2}'.format( \
                   time_in[int(0.5*oute.shape[0])], R_in[-1], Z_in[-1]))
    plt.legend()
    plt.xlabel('Phi [rad]')
    plt.ylabel('B_tor [T]')

    # CALL EQUIMAP
    start = time.time()
    outa = equimap.get(shot, time=time_in, \
            R=R_all_tot, Phi=np.zeros(R_all_tot.shape), Z=Z_all_tot, \
            quantity='b_field_tor')
    end = time.time()
    print()
    print('time in equimap.get btor 2D =', end - start)
    print()
    outar = outa[int(0.5*outa.shape[0])].reshape((interp_points, interp_points))
    plt.figure()
    plt.contourf(Rr, Zr, outar)
    plt.colorbar()
    arg_time = np.argmin(np.abs(out.time - time_in[int(0.5*outa.shape[0])]))
    plt.plot(np.squeeze(out.time_slice[arg_time].boundary.outline.r), \
             np.squeeze(out.time_slice[arg_time].boundary.outline.z), \
             linewidth=2, color='red')
    plt.plot(out.time_slice[arg_time].global_quantities.magnetic_axis.r, \
             out.time_slice[arg_time].global_quantities.magnetic_axis.z, \
             marker='+', color='red', markersize=20)
    plt.xlabel('R [m]')
    plt.ylabel('Z [m]')
    plt.title('B_tor t={0:.2f}'.format(time_in[int(0.5*outa.shape[0])]))

    plt.show()
