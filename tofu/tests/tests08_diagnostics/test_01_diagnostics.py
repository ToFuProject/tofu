"""
This module contains tests for tofu.geom in its structured version
"""


# Built-in
import sys
import os
import warnings


# Standard
import numpy as np
import matplotlib.pyplot as plt


# tofu-specific
_PATH_HERE = os.path.abspath(os.path.dirname(__file__))
_PATH_TOFU = os.path.dirname(os.path.dirname(os.path.dirname(_PATH_HERE)))


sys.path.insert(0, _PATH_TOFU)
import tofu as tf
sys.path.pop(0)


#######################################################
#
#     Setup and Teardown
#
#######################################################


def setup_module():
    pass


def teardown_module():
    pass


#######################################################
#
#     Utilities routines
#
#######################################################


def _ref_line():

    start = np.r_[4, 0, 0]

    vect = np.r_[-1, 0, 0]
    vect = vect / np.linalg.norm(vect)

    v0 = np.r_[-vect[1], vect[0], 0.]
    v0 = v0 / np.linalg.norm(v0)

    v1 = np.cross(vect, v0)

    return start, vect, v0, v1


def _nine0e1_from_orientations(
    vect=None,
    v0=None,
    v1=None,
    theta=None,
    phi=None,
):

    nin = (
        vect * np.cos(theta)
        + np.sin(theta) * (np.cos(phi) * v0 + np.sin(phi) * v1)
    )

    e0 = (
        - vect * np.sin(theta)
        + np.cos(theta) * (np.cos(phi) * v0 + np.sin(phi) * v1)
    )

    e1 = np.cross(nin, e0)

    return nin, e0, e1


def _apertures():

    start, vect, v0, v1 = _ref_line()

    # ap0 : planar from outline
    out0 = 0.01 * np.r_[-1, 1, 1, -1]
    out1 = 0.005 * np.r_[-1, -1, 1, 1]
    cent = start + 0.15 * vect
    nin, e0, e1 = _nine0e1_from_orientations(
        vect=vect,
        v0=v0,
        v1=v1,
        theta=np.pi/10.,
        phi=0.,
    )
    ap0 = {
        'outline_x0': out0,
        'outline_x1': out1,
        'cent': cent,
        'nin': nin,
        'e0': e0,
        'e1': e1,
    }

    # ap1 : planar from 3d
    out0 = 0.01 * np.r_[-1, 1, 0]
    out1 = 0.01 * np.r_[-0.5, -0.5, 0.5]
    cent = start + 0.2 * vect
    nin, e0, e1 = _nine0e1_from_orientations(
        vect=vect,
        v0=v0,
        v1=v1,
        theta=-np.pi/10.,
        phi=np.pi/20.,
    )

    px = cent[0] + out0 * e0[0] + out1 * e1[0]
    py = cent[1] + out0 * e0[1] + out1 * e1[1]
    pz = cent[2] + out0 * e0[2] + out1 * e1[2]

    ap1 = {
        'poly_x': px,
        'poly_y': py,
        'poly_z': pz,
        'nin': nin,
    }

    # ap2 : non-planar
    out0 = 0.005 * np.r_[-1, 1, 2, 0, -2]
    out1 = 0.005 * np.r_[-1, -1, 0, 1, 0]
    cent = start + 0.3 * vect
    nin, e0, e1 = _nine0e1_from_orientations(
        vect=vect,
        v0=v0,
        v1=v1,
        theta=0.,
        phi=-np.pi/20.,
    )

    scatter = 0.001 * np.r_[-1, 1, 1, -1, -1]
    px = cent[0] + out0 * e0[0] + out1 * e1[0] + scatter * nin[0]
    py = cent[1] + out0 * e0[1] + out1 * e1[1] + scatter * nin[1]
    pz = cent[2] + out0 * e0[2] + out1 * e1[2] + scatter * nin[2]

    ap2 = {
        'poly_x': px,
        'poly_y': py,
        'poly_z': pz,
        'nin': nin,
    }

    return {'ap0': ap0, 'ap1': ap1, 'ap2': ap2}


def _cameras():

    start, vect, v0, v1 = _ref_line()

    # c0: 1d non-parallel
    out0 = 0.002 * np.r_[-1, 1, 1, -1]
    out1 = 0.005 * np.r_[-1, -1, 1, 1]
    cent = start + 0.01 * vect

    nin, e0, e1 = _nine0e1_from_orientations(
        vect=vect,
        v0=v0,
        v1=v1,
        theta=np.pi/20.,
        phi=0.,
    )

    npix = 10
    nins = np.full((npix, 3), np.nan)
    e0s = np.full((npix, 3), np.nan)
    e1s = np.full((npix, 3), np.nan)
    for ii, tt in enumerate(np.pi/20 * np.linspace(-1, 1, 10)):
        nins[ii, :], e0s[ii, :], e1s[ii, :] = _nine0e1_from_orientations(
            vect=vect,
            v0=v0,
            v1=v1,
            theta=tt,
            phi=0.,
        )

    kl0 = 0.02*np.linspace(-1, 1, 10)
    klin = 0.01 * np.linspace(-1, 1, 10)**2
    cents_x = cent[0] + kl0 * e0[0] + klin * nins[:, 0]
    cents_y = cent[1] + kl0 * e0[1] + klin * nins[:, 1]
    cents_z = cent[2] + kl0 * e0[2] + klin * nins[:, 2]

    c0 = {
        'outline_x0': out0,
        'outline_x1': out1,
        'cents_x': cents_x,
        'cents_y': cents_y,
        'cents_z': cents_z,
        'nin_x': nins[:, 0],
        'nin_y': nins[:, 1],
        'nin_z': nins[:, 2],
        'e0_x': e0s[:, 0],
        'e0_y': e0s[:, 1],
        'e0_z': e0s[:, 2],
        'e1_x': e1s[:, 0],
        'e1_y': e1s[:, 1],
        'e1_z': e1s[:, 2],
    }

    # c1: 1d parallel coplanar

    kl = 0.02*np.linspace(-1, 1, 10)
    cents_x = cent[0] + kl * e0[0]
    cents_y = cent[1] + kl * e0[1]
    cents_z = cent[2] + kl * e0[2]

    c1 = {
        'outline_x0': out0,
        'outline_x1': out1,
        'cents_x': cents_x,
        'cents_y': cents_y,
        'cents_z': cents_z,
        'nin_x': nins[0, 0],
        'nin_y': nins[0, 1],
        'nin_z': nins[0, 2],
        'e0_x': e0s[0, 0],
        'e0_y': e0s[0, 1],
        'e0_z': e0s[0, 2],
        'e1_x': e1s[0, 0],
        'e1_y': e1s[0, 1],
        'e1_z': e1s[0, 2],
        'lamb': np.linspace(3, 4, 100)*1e-10,
        'qeff': 0.99*np.ones((100,))
    }

    # c2: 2d
    out0 = 0.005 * np.r_[-1, 1, 1, -1]
    out1 = 0.005 * np.r_[-1, -1, 1, 1]
    cent = start + 0.01 * vect

    cent0 = 0.02*np.linspace(-1, 1, 5)
    cent1 = 0.02*np.linspace(-1, 1, 5)

    nin, e0, e1 = _nine0e1_from_orientations(
        vect=vect,
        v0=v0,
        v1=v1,
        theta=np.pi/20.,
        phi=0.,
    )

    c2 = {
        'outline_x0': out0,
        'outline_x1': out1,
        'cent': cent,
        'cents_x0': cent0,
        'cents_x1': cent1,
        'nin': nin,
        'e0': e0,
        'e1': e1,
    }

    return {'c0': c0, 'c1': c1, 'c2': c2}


def _diagnostics():

    # d0: single 1d camera
    d0 = {'optics': 'c0'}

    # d1: single 2d camera
    d1 = {'optics': 'c1', 'resolution': 0.2}

    # d2: single 2d camera
    d2 = {'optics': 'c2'}

    # d3: 1d + 1 aperture
    d3 = {'optics': ('c0', 'ap0')}

    # d4: 2d + 1 aperture
    d4 = {'optics': ('c1', 'ap0')}

    # d5: 2d + 1 aperture
    d5 = {'optics': ('c2', 'ap0')}

    # d6: 1d + multiple apertures
    d6 = {'optics': ('c0', 'ap0', 'ap1', 'ap2')}

    # d7: 1d parallel coplanar + multiple apertures
    d7 = {'optics': ('c1', 'ap0', 'ap1', 'ap2')}

    # d8: 2d + multiple apertures
    d8 = {'optics': ('c2', 'ap0', 'ap1', 'ap2')}

    return {
        'd0': d0,
        'd1': d1,
        'd2': d2,
        'd3': d3,
        'd4': d4,
        'd5': d5,
        'd6': d6,
        'd7': d7,
        'd8': d8,
    }


#######################################################
#
#     Tests
#
#######################################################


class Test01_Diagnostic():

    def setup(self):

        # get dict
        dapertures = _apertures()
        dcameras = _cameras()
        ddiag = _diagnostics()

        # instanciate
        self.obj = tf.data.Diagnostic()

        for k0, v0 in dapertures.items():
            self.obj.add_aperture(key=k0, **v0)

        for k0, v0 in dcameras.items():
            if k0 in ['c0', 'c1']:
                self.obj.add_camera_1d(key=k0, **v0)
            else:
                self.obj.add_camera_2d(key=k0, **v0)

        for k0, v0 in ddiag.items():
            self.obj.add_diagnostic(key=k0, **v0)

    # ----------
    # tests

    def test01_etendues(self, res=np.r_[0.005, 0.003, 0.001]):
        for k0, v0 in self.obj.dobj['diagnostic'].items():
            if k0 in ['d0', 'd1', 'd2']:
                continue
            self.obj.compute_diagnostic_etendue(
                key=k0,
                res=res,
                check=True,
            )
            plt.close('all')

    def test02_get_outline(self):

        # apertures
        for k0, v0 in self.obj.dobj['aperture'].items():
            dout = self.obj.get_optics_outline(k0)

        # camera
        for k0, v0 in self.obj.dobj['camera'].items():
            dout = self.obj.get_optics_outline(k0)


    def test03_plot(self):
        for k0, v0 in self.obj.dobj['diagnostic'].items():
            for pp in [None, 'cross', ['cross', 'hor']]:
                dax = self.obj.plot_diagnostic(k0, proj=pp)
            plt.close('all')
