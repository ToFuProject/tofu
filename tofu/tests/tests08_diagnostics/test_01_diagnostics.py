"""
This module contains tests for tofu.geom in its structured version
"""

# Built-in
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
        vect * np.sin(theta)
        + np.cos(theta) * (np.cos(phi) * v0 + np.sin(phi) * v1)
    )

    e1 = np.cross(nin, e0)

    return nin, e0, e1


def _apertures():

    start, vect, v0, v1 = _ref_line()

    # ap0 : planar from outline
    out0 = 0.01 * np.r_[-1, 1, 1, -1]
    out1 = 0.005 * np.r_[-1, -1, 1, 1]
    cent = start + 0.1 * vect
    nin, e0, e1 = _nine0e1_from_orientations(
        vect=vect,
        e0=v0,
        e1=v1,
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
        e0=v0,
        e1=v1,
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
        e0=v0,
        e1=v1,
        theta=0.,
        phi=-np.pi/20.,
    )

    scatter = 2.*(np.random.random(5) - 0.5)
    px = cent[0] + out0 * e0[0] + out1 * e1[0] + scatter * nin[0]
    py = cent[1] + out0 * e0[1] + out1 * e1[1] + scatter * nin[1]
    pz = cent[2] + out0 * e0[2] + out1 * e1[2] + scatter * nin[2]

    ap1 = {
        'poly_x': px,
        'poly_y': py,
        'poly_z': pz,
        'nin': nin,
    }

    return {'ap0': ap0, 'ap1': ap1, 'ap2': ap2}


def _cameras():

    start, vect = _ref_line()

    # c0: 1d
    out0 = 0.01 * np.r_[-1, 1, 1, -1]
    out1 = 0.005 * np.r_[-1, -1, 1, 1]
    cent = start + 0.01 * vect

    cents = None

    c0 = {
        'outline_x0': out0,
        'outline_x1': out1,
        'cents': cents,
        'nin': nin,
        'e0': e0,
        'e1': e1,
    }

    # c1: 2d
    out0 = 0.001 * np.r_[-1, 1, 1, -1]
    out1 = 0.001 * np.r_[-1, -1, 1, 1]
    cent = start + 0.01 * vect

    c0 = np.linspace(-1, 1, 200)
    c1 = np.linspace(-1, 1, 100)

    cents = (c0, c1)

    c1 = {
        'outline_x0': out0,
        'outline_x1': out1,
        'cent': cent,
        'cents': cents,
        'nin': nin,
        'e0': e0,
        'e1': e1,
    }

    return {'c0': c0, 'c1': c1}


def _diagnostics():

    # d0: single 1d camera


    # d1: single 2d camera


    # d2: 1d + 1 aperture


    # d3: 2d + 1 aperture


    # d4: 1d + multiple apertures


    # d5: 2d + multiple apertures

    return {}


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

        # for k0, v0 in dcameras.items():
            # self.obj.add_camera(key=k0, **v0)

        # for k0, v0 in ddiag.items():
            # self.obj.add_diagnostic(key=k0, **v0)

    # ----------
    # tests

    def test01_apertures(self):
        pass
