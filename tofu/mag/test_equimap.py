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

if __name__ == '__main__':
    #print('path 1 =', sys.path)
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    #print('path 2 =', sys.path)

    # Local modules
    import equimap
    import imas

    interp_points = 60
    shot = 53221

    approx_time_in = 39.9881
    eps_time = 1E-3
    ctr_plot = [0]

    # CALL EQUIMAP
    oute = equimap.get(shot, time=[35.9249267578125, 39.98], \
            R=[2., 2.2, 2.4, 2.7, 3.], Phi=[0, 0, 0, 0, 0], Z=[0, 0, 0, 0, 0], \
            quantity='psi')

    print('')
    print('oute.shape =', oute.shape)
    # oute should be:
    # array([[-2.27165658, -3.01387112, -3.98753036, -3.92317787, -2.43752966],
    #        [ 1.09062263,  0.34659524, -0.63331812, -0.47944498,  1.01023176]])
    print('oute =', oute)
    print('')

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

    arg_time = np.argmin(np.abs(out.time - approx_time_in))

    print('\nDistance between requested and calculated time =', \
          np.abs(out.time[arg_time] - approx_time_in), '\n')

    time_in = out.time[arg_time] - eps_time

    print('\nNew distance between requested and calculated time =', \
          np.abs(out.time[arg_time] - time_in), '\n', \
          'with eps_time =', eps_time, '\n')

    R_all = np.linspace(np.min(equiDict['r']), np.max(equiDict['r']), interp_points)
    Z_all = np.linspace(np.min(equiDict['z']), np.max(equiDict['z']), interp_points)

    R_all_tot = np.repeat(R_all, interp_points)
    Z_all_tot = np.tile(Z_all, interp_points)

    # CALL EQUIMAP
    outa = equimap.get(shot, time=time_in, \
            R=R_all_tot, Phi=np.zeros(R_all_tot.shape), Z=Z_all_tot, \
            quantity='psi')

    outar = outa.reshape((interp_points, interp_points))
    Rr = R_all_tot.reshape((interp_points, interp_points))
    Zr = Z_all_tot.reshape((interp_points, interp_points))

    print('')
    print('outa.shape =', outa.shape)
    print('outar.shape =', outar.shape)
    #print('oute =', oute)
    print('')

    #plt.figure()
    #cs_ref = plt.contour(out.interp2D.r, out.interp2D.z, \
    #            out.interp2D.psi[np.argmin(np.abs(out.time - time_in))], \
    #            ctr_plot, colors='k', linestyle='solid')
    #cs_ref_tri = plt.tricontour(equiDict['r'], equiDict['z'], \
    #                 out.ggd[0].grid.space[0].objects_per_dimension[1].nodes, \
    #                 out.ggd[0].psi[np.argmin(np.abs(out.time - time_in))], \
    #                 ctr_plot, colors='red', linestyles='dashdot')
    #cs     = plt.contour(Rr, Zr, outar, ctr_plot, \
    #                     colors='orange', linestyles='dashed')
    #plt.plot(np.nan, np.nan, color='k', linestyle='solid', label='ref interp2D')
    #plt.plot(np.nan, np.nan, color='red', linestyle='dashdot', label='ref ggd')
    #plt.plot(np.nan, np.nan, color='orange', linestyle='dashed', label='test interp')
    #plt.legend()
    #plt.xlabel('R [m]')
    #plt.ylabel('Z [m]')
    #plt.title('Contour psi, level ' + str(ctr_plot) + ' at time ' + str(time_in) + 's')
    #plt.axes().set(aspect=1)
    #
    #p = cs.collections[0].get_paths()[0]
    #v = p.vertices
    #x_cs = v[:, 0]
    #y_cs = v[:, 1]

    #p = cs_ref.collections[0].get_paths()[0]
    #v = p.vertices
    #x_cs_ref = v[:, 0]
    #y_cs_ref = v[:, 1]

    plt.figure()
    plt.contourf(Rr, Zr, outar)
    plt.plot(np.squeeze(out.time_slice[arg_time].boundary.outline.r), \
             np.squeeze(out.time_slice[arg_time].boundary.outline.z), \
             linewidth=2, color='red')
    plt.plot(out.time_slice[arg_time].global_quantities.magnetic_axis.r, \
             out.time_slice[arg_time].global_quantities.magnetic_axis.z, \
             marker='+', color='red', markersize=20)
    plt.colorbar()
    plt.xlabel('R [m]')
    plt.ylabel('Z [m]')
    plt.title('psi interpolated at time ' + str(time_in) + 's')
    plt.axes().set(aspect=1)

    #plt.figure()
    #plt.plot(x_cs_ref, y_cs_ref, label='Ref')
    #plt.plot(x_cs, y_cs, '--', label='Interp')
    #plt.xlabel('R [m]')
    #plt.ylabel('Z [m]')
    #plt.legend()

    #pts_cs_ref = np.linspace(0, 1, x_cs_ref.shape[0])
    #pts_cs     = np.linspace(0, 1, x_cs.shape[0])

    #x_cs_intp = np.interp(pts_cs_ref, pts_cs, x_cs)
    #y_cs_intp = np.interp(pts_cs_ref, pts_cs, y_cs)

    #error_x = np.abs((x_cs_intp - x_cs_ref)/x_cs_ref)*100
    #error_y = np.abs((y_cs_intp - y_cs_ref)/y_cs_ref)*100

    #plt.figure()
    #plt.plot(error_x, label='err. x')
    #plt.plot(error_y, label='err. y')
    #plt.legend()
    #plt.ylabel('Error interp psi contour level ' + str(ctr_plot) + ' in [%]' \
    #          + '\nwith time difference, eps_time = ' + str(eps_time*100) + '%', \
    #          multialignment='center')

    # CALL EQUIMAP
    out_rho_pol_norm = equimap.get(shot, time=time_in, \
            R=R_all_tot, Phi=np.zeros(R_all_tot.shape), Z=Z_all_tot, \
            quantity='rho_pol_norm')

    print('')
    print('out_rho_pol_norm.shape =', out_rho_pol_norm.shape)
    #print('out_rho_pol_norm =', out_rho_pol_norm)
    print('')

    plt.figure()
    out_rho_pol_norm_r = out_rho_pol_norm.reshape((interp_points, interp_points))
    cs = plt.contourf(Rr, Zr, out_rho_pol_norm_r)
    plt.colorbar()
    plt.plot(np.squeeze(out.time_slice[arg_time].boundary.outline.r), \
             np.squeeze(out.time_slice[arg_time].boundary.outline.z), \
             linewidth=2, color='red')
    plt.plot(out.time_slice[arg_time].global_quantities.magnetic_axis.r, \
             out.time_slice[arg_time].global_quantities.magnetic_axis.z, \
             marker='+', color='red', markersize=20)
    plt.xlabel('R [m]')
    plt.ylabel('Z [m]')
    plt.title('rho_pol_norm')

    # CALL EQUIMAP
    out_rho_tor_norm = equimap.get(shot, time=time_in, \
            R=R_all_tot, Phi=np.zeros(R_all_tot.shape), Z=Z_all_tot, \
            quantity='rho_tor_norm')
    print('')
    print('out_rho_tor_norm.shape =', out_rho_tor_norm.shape)
    #print('out_rho_tor_norm =', out_rho_tor_norm)
    print('')
    plt.figure()
    out_rho_tor_norm_r = out_rho_tor_norm.reshape((interp_points, interp_points))
    cs = plt.contourf(Rr, Zr, out_rho_tor_norm_r)
    plt.colorbar()
    arg_time = np.argmin(np.abs(out.time - time_in))
    plt.plot(np.squeeze(out.time_slice[arg_time].boundary.outline.r), \
             np.squeeze(out.time_slice[arg_time].boundary.outline.z), \
             linewidth=2, color='red')
    plt.plot(out.time_slice[arg_time].global_quantities.magnetic_axis.r, \
             out.time_slice[arg_time].global_quantities.magnetic_axis.z, \
             marker='+', color='red', markersize=20)
    plt.xlabel('R [m]')
    plt.ylabel('Z [m]')
    plt.title('rho_tor_norm')

    # CALL EQUIMAP
    out_rho_tor = equimap.get(shot, time=time_in, \
            R=R_all_tot, Phi=np.zeros(R_all_tot.shape), Z=Z_all_tot, \
            quantity='rho_tor')

    print('')
    print('out_rho_tor.shape =', out_rho_tor.shape)
    #print('out_rho_tor =', out_rho_tor)
    print('')
    plt.figure()
    out_rho_tor_r = out_rho_tor.reshape((interp_points, interp_points))
    cs = plt.contourf(Rr, Zr, out_rho_tor_r)
    plt.colorbar()
    arg_time = np.argmin(np.abs(out.time - time_in))
    plt.plot(np.squeeze(out.time_slice[arg_time].boundary.outline.r), \
             np.squeeze(out.time_slice[arg_time].boundary.outline.z), \
             linewidth=2, color='red')
    plt.plot(out.time_slice[arg_time].global_quantities.magnetic_axis.r, \
             out.time_slice[arg_time].global_quantities.magnetic_axis.z, \
             marker='+', color='red', markersize=20)
    plt.xlabel('R [m]')
    plt.ylabel('Z [m]')
    plt.title('rho_tor')

    plt.figure()
    plt.plot(out.time_slice[arg_time].profiles_1d.psi, out.time_slice[arg_time].profiles_1d.rho_tor)
    plt.xlabel('Psi')
    plt.ylabel('rho_tor')

    #plt.figure()
    #plt.tricontourf(equiDict['r'], equiDict['z'], \
    #                out.ggd[0].grid.space[0].objects_per_dimension[1].nodes, \
    #                out.ggd[0].psi[arg_time])
    #plt.plot(np.squeeze(out.boundary.outline.r[arg_time]), \
    #         np.squeeze(out.boundary.outline.z[arg_time]), \
    #         linewidth=2, color='red')
    #plt.plot(out.global_quantities.magnetic_axis.r[arg_time], \
    #         out.global_quantities.magnetic_axis.z[arg_time], \
    #         marker='+', color='red', markersize=20)
    #plt.colorbar()
    #plt.xlabel('R [m]')
    #plt.ylabel('Z [m]')
    #plt.title('psi ggd')

    # CALL EQUIMAP
    out_b_norm = equimap.get(shot, time=time_in, \
            R=R_all_tot, Phi=np.zeros(R_all_tot.shape), Z=Z_all_tot, \
            quantity='b_field_norm')

    print('')
    print('out_b_norm.shape =', out_b_norm.shape)
    #print('out_b_norm =', out_b_norm)
    print('')
    plt.figure()
    out_b_norm_r = out_b_norm.reshape((interp_points, interp_points))
    cs = plt.contourf(Rr, Zr, out_b_norm_r)
    plt.colorbar()
    arg_time = np.argmin(np.abs(out.time - time_in))
    plt.plot(np.squeeze(out.time_slice[arg_time].boundary.outline.r), \
             np.squeeze(out.time_slice[arg_time].boundary.outline.z), \
             linewidth=2, color='red')
    plt.plot(out.time_slice[arg_time].global_quantities.magnetic_axis.r, \
             out.time_slice[arg_time].global_quantities.magnetic_axis.z, \
             marker='+', color='red', markersize=20)
    plt.xlabel('R [m]')
    plt.ylabel('Z [m]')
    plt.title('b_norm')

    # CALL EQUIMAP
    out_theta = equimap.get(shot, time=time_in, \
            R=R_all_tot, Phi=np.zeros(R_all_tot.shape), Z=Z_all_tot, \
            quantity='theta')

    print('')
    print('out_theta.shape =', out_theta.shape)
    print('out_theta =', out_theta)
    print('')
    plt.figure()
    out_theta_r = out_theta.reshape((interp_points, interp_points))
    cs = plt.contourf(Rr, Zr, out_theta_r, 100)
    plt.colorbar()
    arg_time = np.argmin(np.abs(out.time - time_in))
    plt.plot(np.squeeze(out.time_slice[arg_time].boundary.outline.r), \
             np.squeeze(out.time_slice[arg_time].boundary.outline.z), \
             linewidth=2, color='red')
    plt.plot(out.time_slice[arg_time].global_quantities.magnetic_axis.r, \
             out.time_slice[arg_time].global_quantities.magnetic_axis.z, \
             marker='+', color='red', markersize=20)
    plt.xlabel('R [m]')
    plt.ylabel('Z [m]')
    plt.title('theta')

    plt.show()
