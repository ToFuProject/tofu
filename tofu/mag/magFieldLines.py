# -*- coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
# Also if needed: retab
'''
    Magnetic field line tracing
'''
# Standard python modules
from __future__ import (unicode_literals, absolute_import,  \
                        print_function, division)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import numpy as np
import os
#import re
import scipy.integrate as spode
import scipy.interpolate as interpolate
import scipy.spatial
#import warnings
#import sys

# Local modules
import imas
try:
    import pywed as pw
except ImportError as err:
    print(err)

# Project modules
try:
    from tofu.mag.mag_ripple.mag_ripple import mag_ripple
except Exception:
    from mag.mag_ripple.mag_ripple import mag_ripple
#try:
#    from equimap import interp_quantity
#except ImportError as err:
#    print(err)

__all__ = ['MagFieldLines']

class MagFieldLines:
    '''
    Compute magnetic field lines

    Parameters
    ----------
    shot : int
        Shot number
    run : run number, optional (default=0)
    occ : occurrence number, optional (default=0)
    user : user name, optional (default=imas_public)
    machine : machine name, optional (default=west)

    Methods
    -------
    trace_mline(init_points_array, time_array, direction='FWD')

    In other words:
    trace_mline([[r_1, r_2, ...], [phi_1, phi_2, ...], [z_1, z_2, ...]], [t1, t2, ...], direction='FWD')
        returns list of list of traces with shape (times, init_points)
    plot_trace(trace[0][0])
        2D plots of traced magnetic field line
    plot_trace_3D(trace[0][0])
        3D plot of traced magnetic field line

    Examples
    --------
    For shot 54095
    Compute linear interpolation B_r, B_tor, B_z at r=2, phi=0, z=0
    and r, z, phi of magnetic line starting at r=2.7, phi=0, z=0
    >>> test = MagFieldLines(54095)

    Call method with specific time (t=34 seconds in this case)
    >>> test.trace_mline([2.7, 0., 0.], 34)[0][0]['r'][10]
    2.700041665466963
    >>> test.trace_mline([2.7, 0., 0.], 34)[0][0]['p'][10]
    -0.0037010956581712776
    >>> test.trace_mline([2.7, 0., 0.], 34)[0][0]['z'][10]
    0.000373472578674799

    Call method for list of points (2 items and t=34 seconds)
    >>> test.trace_mline([[2.7, 2.8], [0., 0.], [0., 0.1*np.pi]], 34)[0][0]['r'][10]
    2.700041665466963
    >>> test.trace_mline([[2.7, 2.8], [0., 0.], [0., 0.1*np.pi]], 34)[0][0]['p'][10]
    -0.0037010956581712776
    >>> test.trace_mline([[2.7, 2.8], [0., 0.], [0., 0.1*np.pi]], 34)[0][0]['z'][10]
    0.000373472578674799

    For interpolation of B magnetic field
    >>> test.b_field_interp(2, 0, 0)
    (0.006836488559673934, -4.547017391384852, -0.14305889263210517)
    '''
    def __init__(self, shot, run=0, occ=0, user='imas_public', machine='west'):

        # Check if shot exists
        run_number = '{:04d}'.format(run)
        shot_file  = os.path.expanduser('~' + user + '/public/imasdb/' + machine + \
                                        '/3/0/' + 'ids_' + str(shot) + run_number + \
                                        '.datafile')
        if (not os.path.isfile(shot_file)):
            raise FileNotFoundError('IMAS file does not exist')

        # Parameters
        self.wall_ck=False

        # Get equilibrium
        idd = imas.ids(shot, run)
        idd.open_env(user, machine, '3')
        idd.equilibrium.get()
        self.equi = idd.equilibrium

        # Check code.output_flag for data validity
        if (np.any(np.isnan(self.equi.code.output_flag))):
            self.mask = np.full(len(self.equi.time), True, dtype=bool)
        else:
            self.mask = np.asarray(self.equi.code.output_flag) >= 0

        # Get Itor (current of toroidal coils, coils that produce the toroidal field)
        self.itor, self.t_itor = pw.tsbase(shot, 'gmag_itor', nargout=2)
        self.t_ignitron        = pw.tsmat(shot, 'IGNITRON|1')
        self.t_itor           += self.t_ignitron[0]

        self.equiDict = {}

        # Declaration of arrays 2d from equilibrium IDS
        equi_grid = idd.equilibrium.grids_ggd[0].grid[0]
        NbrPoints = len(equi_grid.space[0].objects_per_dimension[0].object)
        self.equiDict['r'] = np.full(NbrPoints, np.nan)
        self.equiDict['z'] = np.full(NbrPoints, np.nan)
        for ii in range(NbrPoints):
            self.equiDict['r'][ii] = equi_grid.space[0].objects_per_dimension[0]. \
                                object[ii].geometry[0]
            self.equiDict['z'][ii] = equi_grid.space[0].objects_per_dimension[0]. \
                                object[ii].geometry[1]

        self.equiDict['b_field_r']   = np.full((len(self.equi.time), NbrPoints), np.nan)
        self.equiDict['b_field_z']   = np.full((len(self.equi.time), NbrPoints), np.nan)
        self.equiDict['b_field_tor'] = np.full((len(self.equi.time), NbrPoints), np.nan)
        self.equiDict['boundary_r']  = [None]*len(self.equi.time)
        self.equiDict['boundary_z']  = [None]*len(self.equi.time)
        for ii in range(len(self.equi.time)):
            equi_slice  = self.equi.time_slice[ii]
            self.equiDict['b_field_r'][ii]   = equi_slice.ggd[0].b_field_r[0].values
            self.equiDict['b_field_z'][ii]   = equi_slice.ggd[0].b_field_z[0].values
            self.equiDict['b_field_tor'][ii] = equi_slice.ggd[0].b_field_tor[0].values
            self.equiDict['boundary_r'][ii]  = equi_slice.boundary.outline.r
            self.equiDict['boundary_z'][ii]  = equi_slice.boundary.outline.z

        self.points   = np.vstack((self.equiDict['r'], self.equiDict['z'])).transpose()
        self.delaunay = scipy.spatial.Delaunay(self.points)

        # Time interpolation
        self.f_intp_br = interpolate.interp1d(self.equi.time[self.mask], \
                           self.equiDict['b_field_r'][self.mask, :], axis=0, \
                           bounds_error=False)
        self.f_intp_bt = interpolate.interp1d(self.equi.time[self.mask], \
                           self.equiDict['b_field_tor'][self.mask, :], axis=0, \
                           bounds_error=False)
        self.f_intp_bz = interpolate.interp1d(self.equi.time[self.mask], \
                           self.equiDict['b_field_z'][self.mask, :], axis=0, \
                           bounds_error=False)

    def trace_mline(self, init_state, time, direction='FWD', \
                    length_line=None, stp=None, ripple=True):
        '''
        Traces the field line given a starting point.
        Integration step defined by stp and maximum length of the field line
        defined by s.
        Collision with the wall stops integration.

        input:
            - init_state [[r_1, r_2, ...], [phi_1, phi_2, ...], [z_1, z_2, ...]] :
                  the coordinates of the starting points (list or numpy array)
                  OR list of lists OR numpy 2D array
            - time : array of times requested (list or numpy array)
            - direction : direction of integration 'FWD' forward or 'REV' reverse
              (string)
            - ripple : take into account magnetic ripple or not (boolean)

        output:
            returns a dictionary containing:
            - r  : radial coordinate (np.array)
            - p  : toridal coordinate (np.array)
            - z  : vertical coordinate (np.array)
            - x  : x coordinate (np.array)
            - y  : y coordinate (np.array)
            - cp : collision point with the wall (list)

        '''
        # Step for the integration
        if (stp is None):
            stp = 0.001

        # Length of the field line
        if (length_line is None):
            s = 100
        else:
            s = length_line

        ds  = np.linspace(0, s, int(s/stp))

        init_state = np.squeeze(np.asarray(init_state))
        if (init_state.ndim < 2):
            init_state = init_state[:, np.newaxis]

        self.ar_time = np.atleast_1d(np.squeeze(np.asarray([time])))

        br_intp_t = np.atleast_2d(np.squeeze(self.f_intp_br(self.ar_time)))
        bt_intp_t = np.atleast_2d(np.squeeze(self.f_intp_bt(self.ar_time)))
        bz_intp_t = np.atleast_2d(np.squeeze(self.f_intp_bz(self.ar_time)))

        # !!!!!!!!!!!!!!!!!!!!!
        # HARD CODED CORRECTION
        if (np.nanmean(bt_intp_t) > 0):
            bt_intp_t *= -1
        # !!!!!!!!!!!!!!!!!!!!!

        # Interpolate current
        itor_intp_t_vect = np.interp(self.ar_time, self.t_itor[:, 0], \
                                     self.itor[:, 0])
        b0_intp_t_vect   = np.interp(self.ar_time, self.equi.time[self.mask], \
                              self.equi.vacuum_toroidal_field.b0[self.mask])

        outMagLine = []

        for ii in range(self.ar_time.size):
            self.br_lin_intp = \
              interpolate.LinearNDInterpolator(self.delaunay, br_intp_t[ii])
            self.bt_lin_intp = \
              interpolate.LinearNDInterpolator(self.delaunay, bt_intp_t[ii])
            self.bz_lin_intp = \
              interpolate.LinearNDInterpolator(self.delaunay, bz_intp_t[ii])

            # Interpolated current and b0
            self.itor_intp_t = itor_intp_t_vect[ii]
            self.b0_intp_t   = b0_intp_t_vect[ii]

            out = []

            for jj in range(init_state.shape[1]):
                out_prime = self.integrate_solve_ivp(init_state[:, jj], \
                                                     direction, s, ds, ripple)
                out_prime['init_point'] = init_state[:, jj]
                out_prime['time']       = self.ar_time[ii]
                # Return list of dict
                out.append(out_prime)

            outMagLine.append(out)

        return outMagLine

    def integrate_solve_ivp(self, init_state, direction, s, ds, ripple=True):
        '''
        Integration step defined by stp and maximum length of the field line
        defined by s.
        Collision with the wall stops integration.

        input:
            - init_state [r, phi, z] : the coordinates of the starting point (list)
            - direction : direction of integration 'FWD' forward or 'REV' reverse
              (string)
            - s : field line maximum length
            - ds : vector of steps
            - ripple : boolean, take into account magnetic ripple or not

        output:
            returns a dictionary containing:
            - r  : radial coordinate (np.array)
            - p  : toridal coordinate (np.array)
            - z  : vertical coordinate (np.array)
            - x  : x coordinate (np.array)
            - y  : y coordinate (np.array)
            - cp : collision point with the wall (list)
        '''
        if (ripple):
            if direction == 'FWD':
                sol = spode.solve_ivp(self.mfld3dcylfwd, [0, s], init_state,
                                      method='RK23', t_eval=ds,
                                      events=self.hit_wall_circ)
            elif direction == 'REV':
                sol = spode.solve_ivp(self.mfld3dcylrev, [0, s], init_state,
                                      method='RK23', t_eval=ds,
                                      events=self.hit_wall_circ)
        else:
            if direction == 'FWD':
                sol = spode.solve_ivp(self.mfld3dcylfwd_no_ripple,
                                      [0, s], init_state,
                                      method='RK23', t_eval=ds,
                                      events=self.hit_wall_circ)
            elif direction == 'REV':
                sol = spode.solve_ivp(self.mfld3dcylrev_no_ripple,
                                      [0, s], init_state,
                                      method='RK23', t_eval=ds,
                                      events=self.hit_wall_circ)
        sgf = sol.t
        rgf = sol.y[0]
        pgf = sol.y[1]
        xgf = rgf*np.cos(pgf)
        ygf = rgf*np.sin(pgf)
        zgf = sol.y[2]

        if len(sgf)<len(ds):
            colpt = [rgf[-1], pgf[-1], zgf[-1]]
        else:
            colpt = []

        return {'s':  sgf,
                'r':  rgf,
                'p':  pgf,
                'z':  zgf,
                'x':  xgf,
                'y':  ygf,
                'cp': colpt}

    def mfld3dcylfwd(self, s, state):
        '''
        Returns the right end side of the field line system of equations in
        cyclindrical (R, Phi, Z) coord.
        This is the case for forward integration.
        '''
        R, P, Z = state
        Br, Bt, Bz = self.b_field_interp(R, P, Z)
        #Br=self.fBp_r(R,Z)[0]+self.fBt_r(P,self.ftheta(R,Z),R)
        #Bt=self.fBt_t(P,self.ftheta(R,Z),R)
        #Bz=self.fBp_z(R,Z)[0]
        B = np.sqrt(Br*Br + Bz*Bz + Bt*Bt)
        d_R = Br/B
        d_P = Bt/B*1/R
        d_Z = Bz/B
        return [d_R, d_P, d_Z]

    def mfld3dcylrev(self, s, state):
        '''
        Returns the right end side of the field line system of equations in
        cyclindrical (R, Phi, Z) coord.
        This is the case for backward integration.
        '''
        R, P, Z = state
        Br, Bt, Bz = self.b_field_interp(R, P, Z)
        # Br=self.fBp_r(R,Z)[0]+self.fBt_r(P,self.ftheta(R,Z),R)
        # Bt=self.fBt_t(P,self.ftheta(R,Z),R)
        # Bz=self.fBp_z(R,Z)[0]
        B = np.sqrt(Br*Br + Bz*Bz + Bt*Bt)
        d_R = -Br/B
        d_P = -Bt/B*1/R
        d_Z = -Bz/B
        return [d_R, d_P, d_Z]

    def mfld3dcylfwd_no_ripple(self, s, state):
        '''
        Returns the right end side of the field line system of equations in
        cyclindrical (R, Phi, Z) coord.
        This is the case for forward integration.
        '''
        R, P, Z = state
        Br, Bt, Bz = self.b_field_interp_no_ripple(R, P, Z)
        # Br=self.fBp_r(R,Z)[0]+self.fBt_r(P,self.ftheta(R,Z),R)
        # Bt=self.fBt_t(P,self.ftheta(R,Z),R)
        # Bz=self.fBp_z(R,Z)[0]
        B = np.sqrt(Br*Br + Bz*Bz + Bt*Bt)
        d_R = Br/B
        d_P = Bt/B*1/R
        d_Z = Bz/B
        return [d_R, d_P, d_Z]

    def mfld3dcylrev_no_ripple(self, s, state):
        '''
        Returns the right end side of the field line system of equations in
        cyclindrical (R, Phi, Z) coord.
        This is the case for backward integration.
        '''
        R, P, Z = state
        Br, Bt, Bz = self.b_field_interp_no_ripple(R, P, Z)
        # Br=self.fBp_r(R,Z)[0]+self.fBt_r(P,self.ftheta(R,Z),R)
        # Bt=self.fBt_t(P,self.ftheta(R,Z),R)
        # Bz=self.fBp_z(R,Z)[0]
        B = np.sqrt(Br*Br + Bz*Bz + Bt*Bt)
        d_R = -Br/B
        d_P = -Bt/B*1/R
        d_Z = -Bz/B
        return [d_R, d_P, d_Z]

    def hit_wall_circ(self, s, state):
        '''
        return 0 when hit wall.
        With wall_ck:
        - False => The wall is a simple circular torus with minor radius rw
            centered at (Rc,Zc), thus with major radius Rc.
        - True => check collision with wall boundary given in the Equilibrium
            mat file
        '''
        R, P, Z = state
        # if (np.abs(Z) <= 0.5 and R > 3.01
        #     and np.deg2rad(89.5) <= P <= np.deg2rad(90.5)):
        #     return 0
        # if (np.abs(Z) <= 0.5 and R > 3.01
        #     and np.deg2rad(179.5) <= P <= np.deg2rad(180.5)):
        #     print('Got 2')
        #     return 0
        # elif (np.abs(Z) <= 0.5 and R > 3.01
        #       and np.deg2rad(269.5) <= P <= np.deg2rad(270.5)):
        #     print('Got 3')
        #     return 0
        if self.wall_ck==False:
            Rc=2.460
            Zc=0.0
            rw=0.950
            return (R-Rc)**2+(Z-Zc)**2-rw**2
        elif self.wall_ck and Z>=0:
            zwall=self.fwall_up(R)
            return Z-zwall
        elif self.wall_ck and Z<0:
            zwall=self.fwall_dw(R)
            return zwall-Z

    hit_wall_circ.terminal=True

    def b_field_interp(self, R, Phi, Z):
        ''' Linear interpolation of B vector components at R, Phi, Z positions
        '''
        interp_points = np.vstack((R, Z)).transpose()

        br_intp = self.br_lin_intp.__call__(interp_points)
        bt_intp = self.bt_lin_intp.__call__(interp_points)
        bz_intp = self.bz_lin_intp.__call__(interp_points)

        # Compute magnetic field for given Phi
        br_ripple, bt_ripple, bz_ripple = mag_ripple(R, Phi, \
                                                     Z, self.itor_intp_t)
        # Compute reference vaccuum magnetic field
        self.bt_vac = self.equi.vacuum_toroidal_field.r0*self.b0_intp_t / R

        br_intp -= br_ripple[0]
        bt_intp -= (bt_ripple[0] - np.abs(self.bt_vac))
        bz_intp -= bz_ripple[0]

        return br_intp[0], bt_intp[0], bz_intp[0]

    def b_field_interp_no_ripple(self, R, Phi, Z):
        ''' Linear interpolation of B vector components at R, Phi, Z positions
        '''
        interp_points = np.vstack((R, Z)).transpose()

        br_intp = self.br_lin_intp.__call__(interp_points)
        bt_intp = self.bt_lin_intp.__call__(interp_points)
        bz_intp = self.bz_lin_intp.__call__(interp_points)

        return br_intp[0], bt_intp[0], bz_intp[0]

    def plot_trace(self, trace):
        '''
        Plots a summary of the magnetic field line trace.
        It contains a graph of the radial, vertical and toroidal corrdinates followed by
        the vertical projection (w/o and with the wall) and the toroidal projection.

        A black dot represents the starting point of the field line.
        The end of the trace is indicated by:
        - a red dot in case of collision with the wall
        - a black square otherwise

        input:
            - trace : the magneticl field line data (dictionary)
        '''
        sgf   = trace['s']
        rgf   = trace['r']
        zgf   = trace['z']
        pgf   = trace['p']
        xgf   = trace['x']
        ygf   = trace['y']
        colpt = trace['cp']

        ind_time = np.argmin(np.abs(self.equi.time - trace['time']))

        plt.figure(figsize=[12,8])

        plt.subplot(231)
        plt.plot(sgf, rgf)
        plt.grid()
        plt.ylabel('R [m]'); plt.xlabel('s [m]'); plt.title('Radial coord.')

        plt.subplot(232)
        plt.plot(sgf, zgf)
        plt.grid()
        plt.ylabel('Z [m]'); plt.xlabel('s [m]'); plt.title('Vertical coord.')

        plt.subplot(233)
        plt.plot(sgf, np.rad2deg(pgf))
        plt.grid()
        plt.ylabel('$\phi$ [deg]'); plt.xlabel('s [m]'); plt.title('Toroidal coord.')

        plt.subplot(234)
        plt.plot(rgf, zgf)
        plt.plot(rgf[0], zgf[0], marker='o', markersize=3, color="black")
        if not colpt:
            plt.plot(rgf[-1], zgf[-1], marker='s', markersize=5, color="black")
        else:
            plt.plot(rgf[-1], zgf[-1], marker='o', markersize=5, color="red")
        plt.axis('equal'); plt.xlabel('R [m]'); plt.ylabel('Z [m]')
        plt.plot(self.equiDict['boundary_r'][ind_time], \
                 self.equiDict['boundary_z'][ind_time])
        plt.title('Vertical projection')
        plt.grid()

        plt.subplot(235)
        plt.plot(rgf, zgf)
        #plt.plot(rwall,zwall)
        plt.plot(rgf[0], zgf[0], marker='o', markersize=3, color="black")
        if not colpt:
            plt.plot(rgf[-1], zgf[-1], marker='s', markersize=5, color="black")
        else:
            plt.plot(rgf[-1], zgf[-1], marker='o', markersize=5, color="red")
        plt.axis('equal'); plt.xlabel('R [m]'); plt.ylabel('Z [m]')
        plt.title('Vertical projection')

        plt.subplot(236)
        plt.plot(xgf, ygf)
        #plt.plot(ra*np.cos(pa),ra*np.sin(pa),'b:')
        plt.plot(xgf[0], ygf[0], marker='o', markersize=3, color="black")
        if not colpt:
            plt.plot(xgf[-1], ygf[-1], marker='s', markersize=5, color="black")
        else:
            plt.plot(xgf[-1], ygf[-1], marker='o', markersize=5, color="red")
        plt.axis('equal'); plt.xlabel('x [m]'); plt.ylabel('y [m]')
        plt.title('Toroidal projection')

        plt.tight_layout()
        plt.show()

    def plot_trace_3D(self, trace):
        '''
        Plots the magnetic field line trace in a 3D projection.

        The outline of the antenna is shown by black solid ines.
        A black dot represents the starting point of the field line.
        The end of the trace is indicated by:
        - a red dot in case of collision with the wall
        - a black square otherwise

        input:
            - trace : the magneticl field line data (dictionary)
        '''
        sgf   = trace['s']
        rgf   = trace['r']
        zgf   = trace['z']
        pgf   = trace['p']
        xgf   = trace['x']
        ygf   = trace['y']
        colpt = trace['cp']

        fig = plt.figure(figsize=[10, 10])
        ax = Axes3D(fig)
        #ax = fig.gca(projection='3d')
        #ax.set_aspect('equal')
        ax.plot(xgf, ygf, zgf)
        #ax.plot(ra[0]*np.cos(pa),ra[0]*np.sin(pa),za[0],'k')
        #ax.plot(ra*np.cos(pa[0]),ra*np.sin(pa[0]),za,'k')
        #ax.plot(ra[-1]*np.cos(pa),ra[-1]*np.sin(pa),za[-1],'k')
        #ax.plot(ra*np.cos(pa[-1]),ra*np.sin(pa[-1]),za,'k')
        ax.plot(3.5*np.cos(np.linspace(0, 2*np.pi, 36)), \
                3.5*np.sin(np.linspace(0, 2*np.pi, 36)), 0, 'k:')
        # Used to create the fake bounding box
        max_range = np.array([xgf.max()-xgf.min(), ygf.max()-ygf.min(), zgf.max()-zgf.min()]).max()
        Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(xgf.max()+xgf.min())
        Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(ygf.max()+ygf.min())
        Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(zgf.max()+zgf.min())
        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')
        ax.plot([xgf[0]], [ygf[0]], [zgf[0]], marker='o', markersize=3, color="black")
        if not colpt:
            ax.plot([xgf[-1]], [ygf[-1]], [zgf[-1]], marker='s', markersize=5, color="black")
        else:
            ax.plot([xgf[-1]], [ygf[-1]], [zgf[-1]], marker='o', markersize=5, color="red")
        ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_zlabel('z [m]')
        plt.title('Magnetic field line trace')
        plt.show()

    def plot_poincare(self, trace_plt, tor_angle=0):
        '''
        Plots Poincare plot at given toroidal angle

        input:
            - trace_plt : the magneticl field line data for several points
                          one single time (dictionary)
        '''

        fig, ax = plt.subplots()

        for ii in range(len(trace_plt)):
            loc_cross = trace_plt[ii]['p'] % (tor_angle + 2*np.pi)
            mask_cross = np.r_[False, loc_cross[1:]
                               < loc_cross[:-1]] & np.r_[loc_cross[:-1]
                                                         < loc_cross[1:],
                                                         False]
            ax.plot(trace_plt[ii]['r'][mask_cross],
                    trace_plt[ii]['z'][mask_cross], 'o', markersize=3)

        ax.set_aspect('equal')
        ax.set_xlabel('R [m]')
        ax.set_ylabel('Z [m]')
