"""
This module contains tests for tofu.geom in its structured version
"""

# External modules
import sys
import os
import numpy as np
import matplotlib.pyplot as plt


_PATH_HERE = os.path.dirname(__file__)
_PATH_TOFU = os.path.dirname(os.path.dirname(os.path.dirname(_PATH_HERE)))


sys.path.insert(0, _PATH_TOFU)
import tofu as tf
sys.path.pop(0)


#######################################################
#
#     Setup and Teardown
#
#######################################################

def setup_module(module):
    print ("") # this is to get a newline after the dots
    #print ("setup_module before anything in this file")

def teardown_module(module):
    #os.remove(VesTor.Id.SavePath + VesTor.Id.SaveName + '.npz')
    #os.remove(VesLin.Id.SavePath + VesLin.Id.SaveName + '.npz')
    #print ("teardown_module after everything in this file")
    #print ("") # this is to get a newline
    pass


#######################################################
#
#     Utility setup routines
#
#######################################################


def _get_cases(ang0=np.pi/4., ang1=-np.pi/4.):

    # load and sample config
    conf = tf.load_config('WEST-V0')
    pts, dV, ind, reseff = conf.Ves.V1.get_sampleV(
        res=0.1,
        domain=[[2.2, 2.6], [-0.1, 0.1], [-np.pi/10, np.pi/10]],
    )

    # Main line
    P0 = np.r_[2.5, 0, 0]
    vect = np.array([
        np.cos(ang0)*np.cos(ang1),
        np.cos(ang0)*np.sin(ang1),
        np.sin(ang0),
    ])
    e0 = np.array([-vect[1], vect[0], 0])
    e0 /= np.linalg.norm(e0)
    e1 = np.cross(e0, vect)

    # centers positions in X, Y, Z
    nsides = np.r_[3, 4, 6, 4, 4]
    dx1 = np.r_[0, 0, 0, 0, 0.05]
    radius = 0.02
    lengths = 1. + np.linspace(0, 1, len(nsides))
    cents = (
        P0[:, None]
        + lengths[None, :]*vect[:, None]
        + dx1[None, :]*e1[:, None]
    )

    # outlines (triangle, rectangle, hexagon...)
    lout = []
    for ii, nn in enumerate(nsides):
        theta = np.linspace(-np.pi, np.pi, nn+1)
        lout.append(radius * np.array([np.cos(theta), np.sin(theta)]))

    # dict
    lcases = [f'case {ii}' for ii in range(2)]  # 4
    dcases = dict.fromkeys(lcases, {})
    for ii, cc in enumerate(lcases):
        dcases[cc]['pts_x'] = pts[0, :]
        dcases[cc]['pts_y'] = pts[1, :]
        dcases[cc]['pts_z'] = pts[2, :]
        dcases[cc]['visibility'] = ii % 2 == 1
        dcases[cc]['return_vector'] = ii % 4 < 2
        dcases[cc]['apertures'] = {
            'ap{jj}': {
                'poly_x': (
                    cents[0, ii]
                    + e0[0] * lout[jj][0, :]
                    + e1[0] * lout[jj][1, :]
                ),
                'poly_y': (
                    cents[1, ii]
                    + e0[1] * lout[jj][0, :]
                    + e1[1] * lout[jj][1, :]
                ),
                'poly_z': (
                    cents[2, ii]
                    + e0[2] * lout[jj][0, :]
                    + e1[2] * lout[jj][1, :]
                ),
                'nin': vect,
            }
            for jj in range(3)
        }
        dcases[cc]['detectors'] = {
            'outline_x0': lout[-1][0, :],
            'outline_x1': lout[-1][1, :],
            'cents_x': cents[0, -2:],
            'cents_y': cents[1, -2:],
            'cents_z': cents[2, -2:],
            'nin_x': -vect[0]*np.r_[1, 1],
            'nin_y': -vect[1]*np.r_[1, 1],
            'nin_z': -vect[2]*np.r_[1, 1],
            'e0_x': e0[0]*np.r_[1, 1],
            'e0_y': e0[1]*np.r_[1, 1],
            'e0_z': e0[2]*np.r_[1, 1],
            'e1_x': e1[0]*np.r_[1, 1],
            'e1_y': e1[1]*np.r_[1, 1],
            'e1_z': e1[2]*np.r_[1, 1],
        }
        dcases[cc]['config'] = conf if dcases[cc]['visibility'] else None

    return dcases


#######################################################
#
#     Class
#
#######################################################


class Test01_SolidAngles():

    # ------------------------
    #   setup
    # ------------------------

    def setup_method(self):

        self.dcases = _get_cases()

    # ------------------------
    #   Populating
    # ------------------------

    def test01_solid_angle_apertures(self):

        for k0, v0 in self.dcases.items():

            nd = v0['detectors']['cents_x'].size
            na = len(v0['apertures'])
            npts = v0['pts_x'].size

            # call function
            out = tf.geom._comp_solidangles.calc_solidangle_apertures(
                # observation points
                pts_x=v0['pts_x'],
                pts_y=v0['pts_y'],
                pts_z=v0['pts_z'],
                # polygons
                apertures=v0['apertures'],
                detectors=v0['detectors'],
                # possible obstacles
                config=v0.get('config'),
                # parameters
                visibility=v0['visibility'],
                return_vector=v0['return_vector'],
            )

            # prepare checks
            if not v0['return_vector']:
                out = (out,)
                nout = 1
            else:
                nout = 4

            # Check output format conformity
            assert isinstance(out, tuple) and len(out) == nout
            assert all([
                isinstance(oo, np.ndarray)
                and oo.shape == (nd, npts)
                for oo in out
            ]), f'{(nd, npts)} vs {[oo.shape for oo in out]}'
            assert np.all(out[0] >= 0.)

            # check unit vectors (normalized)
            if nout > 1:
                iok = (out[0] > 0)
                assert all([
                    np.all(np.isfinite(oo) == iok)
                    for oo in out[1:]
                ])
                norm2 = out[1]**2 + out[2]**2 + out[3]**2
                assert np.allclose(norm2[iok], 1)
