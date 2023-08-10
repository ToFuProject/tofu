# -*- coding: utf-8 -*-
"""
This module contains tests for tofu.geom in its structured version
"""


# Built-in
import sys
import os
import copy


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
        'cent': cent + np.r_[0., 0.02, 0.],
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


def _filters():

    dap = _apertures()

    # energy
    energ = np.linspace(1000, 5000, 100)
    trans = np.r_[
        np.zeros((10,)),
        np.linspace(0, 1, 80),
        np.ones((10,)),
    ]

    return {
        'filt0': {
            'dgeom': dict(dap['ap1']),
            'dmat': {
                'name': 'blabla',
                'symbol': 'bla',
                'thickness': 500e-6,
                'energy': energ,
                'transmission': trans,
            },
        },
    }


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
        'dgeom': {
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
        },
    }

    # c1: 1d parallel coplanar

    kl = 0.02*np.linspace(-1, 1, 10)
    cents_x = cent[0] + kl * e0[0]
    cents_y = cent[1] + kl * e0[1]
    cents_z = cent[2] + kl * e0[2]

    c1 = {
        'dgeom': {
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
        },
        'dmat': {
            'qeff_E': np.linspace(1, 10, 100)*1e3,
            'qeff': 0.99*np.ones((100,)),
        },
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
        'dgeom': {
            'outline_x0': out0,
            'outline_x1': out1,
            'cent': cent,
            'cents_x0': cent0,
            'cents_x1': cent1,
            'nin': nin,
            'e0': e0,
            'e1': e1,
        },
    }

    return {
        'cam0': c0,
        'cam1': c1,
        'cam2': c2,
        'cam00': copy.deepcopy(c0),
        'cam11': copy.deepcopy(c1),
        'cam22': copy.deepcopy(c2),
        'cam000': copy.deepcopy(c0),
        'cam111': copy.deepcopy(c1),
        'cam222': copy.deepcopy(c2),
    }


def _crystals():

    start, vect, v0, v1 = _ref_line()

    # cryst0: planar
    cent = start + 0.01 * vect

    nin, e0, e1 = _nine0e1_from_orientations(
        vect=vect,
        v0=v0,
        v1=v1,
        theta=-50*np.pi/180,
        phi=0.,
    )

    size = 2.e-2
    c0 = {
        'dgeom': {
            'cent': cent,
            'nin': nin,
            'e0': e0,
            'e1': e1,
            'extenthalf': size,
            'curve_r': np.inf,
        },
        'dmat': 'Quartz_110',
    }

    # c1: cylindrical
    size = 6.e-2
    rc = 2.
    c1 = {
        'dgeom': {
            'cent': cent,
            'nin': nin,
            'e0': e0,
            'e1': e1,
            'extenthalf': size * np.r_[1, 1/rc],
            'curve_r': [np.inf, rc],
        },
        'dmat': 'Quartz_110',
    }

    # c2: spherical
    rc = 2.
    c2 = {
        'dgeom': {
            'cent': cent,
            'nin': nin,
            'e0': e0,
            'e1': e1,
            'extenthalf': size * np.r_[1 / rc, 1 / rc],
            'curve_r': rc,
        },
        'dmat': 'Quartz_110',
    }

    # c3: toroidall
    rc = 2.
    c3 = {
        'dgeom': {
            'cent': cent,
            'nin': nin,
            'e0': e0,
            'e1': e1,
            'extenthalf': size * np.r_[1 / (3.*rc/2.), 1 / (rc/2.)],
            'curve_r': [rc, rc/2.],
        },
        'dmat': 'Quartz_110',
    }

    return {
        'cryst0': c0,
        'cryst1': c1,
        'cryst2': c2,
        'cryst3': c3,
    }


def _configurations():
    return {
        'cryst0': ['pinhole'],
        'cryst1': ['pinhole', 'von hamos'],
        'cryst2': ['pinhole', 'johann'],
        'cryst3': [],
    }


def _diagnostics():

    # d0: single 1d camera
    d0 = {'doptics': 'cam0'}

    # d1: single 2d camera
    d1 = {'doptics': 'cam1', 'resolution': 0.2}

    # d2: single 2d camera
    d2 = {'doptics': 'cam2'}

    # d3: 1d + 1 aperture
    d3 = {'doptics': ('cam00', 'ap0')}

    # d4: 2d + 1 aperture
    d4 = {'doptics': ('cam11', 'ap0')}

    # d5: 2d + 1 aperture
    d5 = {'doptics': ('cam22', 'ap0')}

    # d6: 1d + multiple apertures
    d6 = {'doptics': ('cam000', 'ap0', 'filt0', 'ap2')}

    # d7: 1d parallel coplanar + multiple apertures
    d7 = {'doptics': ('cam111', 'ap0', 'filt0', 'ap2')}

    # d8: 2d + multiple apertures
    d8 = {'doptics': ('cam222', 'ap0', 'filt0', 'ap2')}

    # # d9: 2d + spherical crystal
    # d9 = {'optics': ('c3','cryst0')}

    # # d10: 2d + cylindrical crystal + slit
    # d10 = {'optics': ('c3','cryst1', 'slit0')}

    # # d11: 2d + toroidal crystal + slit
    # d11 = {'optics': ('c3','cryst2', 'slit1')}

    return {
        'diag0': d0,
        'diag1': d1,
        'diag2': d2,
        'diag3': d3,
        'diag4': d4,
        'diag5': d5,
        'diag6': d6,
        'diag7': d7,
        'diag8': d8,
        # 'd9': d9,
        # 'd10': d10,
        # 'd11': d11,
    }


#######################################################
#
#     Tests
#
#######################################################


class Test01_Diagnostic():

    def setup_method(self):

        # get config
        conf = tf.load_config('SPARC')
        conf.remove_Struct(Cls='PFC', Name='ICRH0')
        self.conf = conf

        # get dict
        dapertures = _apertures()
        dfilters = _filters()
        dcameras = _cameras()
        dcrystals = _crystals()
        dconfig = _configurations()
        ddiag = _diagnostics()

        # instanciate
        self.obj = tf.data.Collection()

        # add apertures
        for k0, v0 in dapertures.items():
            self.obj.add_aperture(key=k0, **v0)

        # add filters
        for k0, v0 in dfilters.items():
            self.obj.add_filter(key=k0, **v0)

        # add cameras
        for k0, v0 in dcameras.items():
            if 'cam0' in k0 or 'cam1' in k0:
                self.obj.add_camera_1d(key=k0, **v0)
            else:
                self.obj.add_camera_2d(key=k0, **v0)

        # add diagnostics
        for k0, v0 in ddiag.items():
            self.obj.add_diagnostic(
                key=k0,
                config=conf,
                reflections_nb=2,
                reflections_type='specular',
                **v0,
            )

        # add crystals
        for k0, v0 in dcrystals.items():
            self.obj.add_crystal(key=k0, **v0)

        # add crystal optics
        self.doptics = {}
        for k0, v0 in dcrystals.items():

            for ii, cc in enumerate(dconfig[k0]):

                apdim = [100e-6, 8e-2] if cc != 'pinhole' else None
                pinrad = 500e-6 if cc == 'pinhole' else None

                loptics = self.obj.get_crystal_ideal_configuration(
                    key=k0,
                    configuration=cc,
                    # parameters
                    cam_on_e0=False,
                    cam_tangential=True,
                    cam_dimensions=[8e-2, 5e-2],
                    focal_distance=2.,
                    # store
                    store=True,
                    key_cam=f'{k0}-cam{ii}',
                    aperture_dimensions=apdim,
                    pinhole_radius=pinrad,
                    cam_pixels_nb=[5, 3],
                    # returnas
                    returnas=list,
                )

                if 'cryst1-slit' in loptics:
                    loptics.append('ap0')

                # add diag
                gtype = self.obj.dobj['crystal'][k0]['dgeom']['type']
                if gtype not in ['toroidal']:
                    self.doptics.update({
                        f'{k0}-{ii}': loptics,
                    })

        # add crystal optics
        for k0, v0 in self.doptics.items():
            self.obj.add_diagnostic(
                doptics=v0,
                config=self.conf,
            )
        # add toroidal
        # self.obj.add_diagnostic(optics=['cryst2-cam0', 'cryst3'])

    # ----------
    # tests

    """
    def test01_etendues(self, res=np.r_[0.005, 0.003, 0.001]):
        for k0, v0 in self.obj.dobj['diagnostic'].items():
            if len(v0['optics']) == 1 or v0['spectro'] is not False:
                continue
            self.obj.compute_diagnostic_etendue_los(
                key=k0,
                res=res,
                numerical=True,
                analytical=True,
                check=True,
                store=False,
            )
            plt.close('all')
    """

    def test02_get_outline(self):

        # apertures
        for k0, v0 in self.obj.dobj['aperture'].items():
            dout = self.obj.get_optics_outline(k0)

        # camera
        for k0, v0 in self.obj.dobj['camera'].items():
            dout = self.obj.get_optics_outline(k0)

        # crystals
        for k0, v0 in self.obj.dobj['crystal'].items():
            dout = self.obj.get_optics_outline(k0)

    def test03_plot(self):
        for ii, (k0, v0) in enumerate(self.obj.dobj['diagnostic'].items()):
            dax = self.obj.plot_diagnostic(
                k0,
                proj=(
                    None if ii % 3 == 0
                    else ('cross' if ii % 3 == 1 else ['cross', 'hor'])
                ),
            )
            plt.close('all')
            del dax

    def test04_sinogram(self):

        lrays = list(self.obj.dobj['rays'].keys())
        ldiag = [
            k0 for k0, v0 in self.obj.dobj['diagnostic'].items()
            if any([v1.get('los') is not None for v1 in v0['doptics'].values()])
        ]
        lk = [lrays] + ldiag
        for ii, k0 in enumerate(lk):
            dout, dax = self.obj.get_sinogram(
                key=k0,
                ang='theta' if ii % 2 == 0 else 'xi',
                ang_units='deg' if ii % 3 == 0 else 'radian',
                impact_pos=ii % 3 != 0,
                R0=2.4 if ii % 3 != 1 else None,
                config=None if ii % 3 != 1 else self.conf,
                pmax=None if ii % 3 == 0 else 5,
                plot=True,
            )
            plt.close('all')
            del dax

