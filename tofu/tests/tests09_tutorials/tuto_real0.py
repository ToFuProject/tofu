"""
Creating and using diagnostic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

"""

import numpy as np


import tofu as tf


__all__ = ['main']


# #####################################################
# #####################################################
#           Main
# #####################################################


def main():

    # ------------------------
    # create plasma

    conf, coll = _create_plasma()

    # ------------------------
    # add several diagnostics

    # add broadband
    _add_broadband(coll, conf)

    # add 2d camera
    # _add_2d(coll, conf)

    # add PHA
    # _add_PHA(coll, conf)

    # add spectrometer
    # _add_spectrometer(coll, conf) # , crystals=['c0'])

    # ------------------------
    # compute synthetic signal

    # _compute_synth_signal(coll) # , ldiag=['diag00'])

    # ------------------
    # geometry matrices

    # ----------
    # inversions

    return conf, coll


# #####################################################
# #####################################################
#           Routines - PLasma
# #####################################################


def _create_plasma():

    # -----------------
    # load a simple geometry

    conf = tf.load_config('WEST-V0')

    # -----------------
    # Instanciate a collection

    coll = tf.data.Collection()

    # -----------------
    # add a rect mesh with 2d bsplines

    coll.add_mesh_2d_rect(
        key='m0',
        res=0.1,
        crop_poly=conf,
        units='m',
        deg=1,
    )

    # --------
    # add time

    nt = 11
    t = np.linspace(0, 10, nt)

    coll.add_ref('nt', nt)
    coll.add_data('t', data=t, ref='nt', units='s', dime='time')

    # ------------------
    # add a spectra mesh

    nE = 100
    E = np.linspace(1000, 30000, nE)

    coll.add_mesh_1d(
        key='mE',
        knots=E,
        deg=1,
        units='eV',
    )

    # ---------------------
    # get shape of bs coefs to set a time-vaying 2d radius

    kapR, kapZ = coll.dobj['bsplines']['m0_bs1']['apex']
    apR = coll.ddata[kapR]['data']
    apZ = coll.ddata[kapZ]['data']

    # rho2d
    rho2d = (
        (1 + 0.1*np.cos(t[:, None, None]))
        * (1. - np.exp(
            -(apR[None, :, None]-2.5)**2/0.4**2
            -(apZ[None, None, :]-0)**2/0.6**2
        ))
    )

    coll.add_data(
        'rho2d',
        data=rho2d,
        ref=('nt', 'm0_bs1'),
        units='',
    )

    # add 1d radial mesh based on rho2d
    nrho = 20
    rho = np.linspace(0, 1, nrho)
    coll.add_mesh_1d(
        key='mr',
        knots=rho,
        units='',
        subkey='rho2d',
        deg=1,
    )

    # emiss1dE
    r0 = (0. + np.exp(-rho**2/0.4**2)[None, :, None])
    r1 = np.exp(-(rho-0.3)**2/0.15**2)[None, :, None]
    emiss1dE = (
        (1 + 0.1*np.cos(t[:, None, None]))
        * (
            r0 * (0.1 + np.exp(-E/10000))[None, None, :]
            + r1 * np.exp(-(E-15000)**2/1000**2)[None, None, :]
        )
    )

    coll.add_data(
        'emiss1d',
        data=np.sum(emiss1dE, axis=-1),
        ref=('nt', 'mr_bs1'),
        units='ph/(m3.s.sr)',
    )

    # emiss1dE
    coll.add_data(
        'emiss1dE',
        data=emiss1dE,
        ref=('nt', 'mr_bs1', 'mE_bs1'),
        units='ph/(m3.s.sr.eV)',
    )

    # emiss2dE
    r0 = (0. + np.exp(-rho2d**2/0.4**2)[:, :, :, None])
    r1 = np.exp(-(rho2d-0.3)**2/0.15**2)[:, :, :, None]
    emiss2dE = (
        (1 + 0.1*np.cos(t[:, None, None, None]))
        * (
            r0 * (1. + np.exp(-E/10000))[None, None, None, :]
            + r1 * np.exp(-(E-15000)**2/1000**2)[None, None, None, :]
        )
    )

    coll.add_data(
        'emiss2dE',
        data=emiss2dE,
        ref=('nt', 'm0_bs1', 'mE_bs1'),
        units='ph/(m3.s.sr.eV)',
    )

    return conf, coll


# #####################################################
# #####################################################
#           Routines - Diags
# #####################################################


def _add_broadband(
    coll=None,
    conf=None,
):

    # ---------------------
    # add 2 pinhole cameras

    coll.add_camera_pinhole(
        key='bb0',
        key_pinhole=None,
        key_diag='d0',
        cam_type='1d',
        R=3.2,
        z=-0.5,
        phi=0,
        theta=3.*np.pi/4,
        dphi=np.pi/10,
        tilt=np.pi/2,
        focal=0.1,
        pix_nb=10,
        pix_size=3e-3,
        pix_spacing=5e-3,
        pinhole_radius=None,
        pinhole_size=[1e-3, 1e-3],
        reflections_nb=0,
        reflections_type=None,
        compute=False,
        config=conf,
    )

    coll.add_camera_pinhole(
        key='bb1',
        key_pinhole=None,
        key_diag='d0',
        cam_type='1d',
        R=3.2,
        z=0.5,
        phi=0,
        theta=-3.*np.pi/4,
        dphi=np.pi/10,
        tilt=np.pi/2,
        focal=0.1,
        pix_nb=10,
        pix_size=3e-3,
        pix_spacing=5e-3,
        pinhole_radius=None,
        pinhole_size=[1e-3, 1e-3],
        reflections_nb=0,
        reflections_type=None,
        compute=True,
        config=conf,
    )

    return


def _add_2d(
    coll=None,
    conf=None,
):

    # ---------------------
    # add 2 pinhole cameras

    coll.add_camera_pinhole(
        key='c2d',
        key_diag='d1',
        key_pinhole=None,
        cam_type='2d',
        R=3.2,
        z=-0.5,
        phi=0,
        theta=3.*np.pi/4,
        dphi=np.pi/10,
        tilt=np.pi/2,
        focal=0.1,
        pix_nb=15,
        pix_size=3e-3,
        pix_spacing=5e-3,
        pinhole_radius=1e-2,
        pinhole_size=None,
        reflections_nb=0,
        reflections_type=None,
        compute=True,
        config=conf,
    )


def _add_PHA(
    coll=None,
    conf=None,
):

    nEb = 14
    Ebins = np.linspace(4000, 20000, nEb)

    # Single pinhole cam
    coll.add_camera_pinhole(
        key='pha0',
        key_pinhole=None,
        key_diag='d2',
        cam_type='1d',
        R=5,
        z=0,
        phi=0,
        theta=np.pi,
        dphi=0,
        tilt=0,
        focal=0.1,
        pix_nb=5,
        pix_size=1e-2,
        pix_spacing=0,
        pinhole_radius=1e-3,
        pinhole_size=None,
        reflections_nb=0,
        reflections_type=None,
        compute=True,
        config=conf,
        dmat={'bins': Ebins},
    )

    return


def _add_spectrometer(
    coll=None,
    conf=None,
    crystals=None,
):

    # ------------------
    # add crystal optics

    doptics = {}
    dcrystals = _crystals(coll, crystals=crystals)

    # ----------------------
    # add camera / apertures

    for k0, v0 in dcrystals.items():

        loptics = coll.get_crystal_ideal_configuration(
            key=k0,
            configuration=v0['configuration'],
            # parameters
            cam_on_e0=False,
            cam_tangential=True,
            cam_dimensions=[5e-2, 3e-2],
            pinhole_distance=2.,
            # store
            store=True,
            key_cam=f'{k0}_cam',
            aperture_dimensions=[100e-6, 1e-2],
            pinhole_radius=100e-6,
            cam_pixels_nb=[10, 5],
            # returnas
            returnas=list,
        )

        # add diag
        gtype = coll.dobj['crystal'][k0]['dgeom']['type']
        if gtype not in ['toroidal']:
            doptics.update({
                k0: loptics,
            })

    # ------------------
    # add diagnostic

    for k0, v0 in doptics.items():
        coll.add_diagnostic(
            doptics=v0,
            config=conf,
            compute=True,
            add_points=3,
            rocking_curve_fwhm=0.0001*np.pi/180,
            # rocking_curve_fwhm=None,
        )

    return


def _crystals(coll=None, crystals=None):

    # -------
    # check

    if crystals is None:
        crystals = ['c0', 'c1', 'c2']

    # -------
    # geom

    start, vect, v0, v1 = _ref_line(start=np.r_[7, 0., 0.001])

    # cryst0: planar
    cent = start + 0. * vect

    nin, e0, e1 = _nine0e1_from_orientations(
        vect=vect,
        v0=v0,
        v1=v1,
        theta=-np.pi/4,
        phi=0.,
    )

    dc = {}

    # c1: cylindrical (von hamos)
    if 'c0' in crystals:
        size = 1.e-2
        rc = 2.
        c0 = {
            'key': 'c0',
            'dgeom': {
                'cent': cent,
                'nin': nin,
                'e0': e0,
                'e1': e1,
                'extenthalf': size * np.r_[1, 1/rc],
                'curve_r': [np.inf, rc],
            },
            'dmat': 'Quartz_110',
            'configuration': 'von hamos',
        }
        dc['c0'] = c0
        coll.add_crystal(c0['key'], dgeom=c0['dgeom'], dmat=c0['dmat'])

    # c3: cylindrical (convex)
    if 'c1' in crystals:
        rc = 2.
        ang = np.linspace(-0.0001, 0.0005, 100)
        c1 = {
            'key': 'c1',
            'dgeom': {
                'cent': cent,
                'nin': nin,
                'e0': e0,
                'e1': e1,
                'extenthalf': size * np.r_[1/rc, 1],
                'curve_r': [-rc, np.inf],
            },
            'dmat': {
                'target': {'lamb': 3.94e-10},
                'd_hkl': 2.45652e-10,
                'drock': {
                    'angle_rel': ang,
                    'power_ratio': np.exp(-(ang-0.00025)**2/0.0001**2),
                },
            },
            'configuration': 'pinhole',
        }
        dc['c1'] = c1
        coll.add_crystal(c1['key'], dgeom=c1['dgeom'], dmat=c1['dmat'])

    # c2: spherical
    if 'c2' in crystals:
        rc = 2.
        c2 = {
            'key': 'c2',
            'dgeom': {
                'cent': cent,
                'nin': nin,
                'e0': e0,
                'e1': e1,
                'extenthalf': size * np.r_[1 / rc, 1 / rc],
                'curve_r': rc,
            },
            'dmat': 'Quartz_110',
            'configuration': 'johann',
        }
        dc['c2'] = c2
        coll.add_crystal(c2['key'], dgeom=c2['dgeom'], dmat=c2['dmat'])

    return dc


def _ref_line(start=np.r_[4, 0, 0], vect=np.r_[-1, 0, 0]):

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


# #####################################################
# #####################################################
#           Synthetic signal
# #####################################################


def _compute_synth_signal(coll=None, ldiag=None):

    # -------------
    # list of diags

    if ldiag is None:
        ldiag = [
            'd0',
            'd1',
            'diag00', 'diag01', 'diag02',
        ]

    # ------------
    # loop n diags

    for k0 in ldiag:

        if k0 not in coll.dobj['diagnostic'].keys():
            continue

        # params
        if k0 == 'd0':
            key_integrand = 'emiss1d'
            ref_com = 'nt'
        # elif k0 == 'diag00':
            # key_integrand = 'emiss2dE'
            # ref_com = None
        else:
            key_integrand = 'emiss1dE'
            ref_com = 'nt'

        # compute
        coll.compute_diagnostic_signal(
            key=None,
            key_diag=k0,
            key_cam=None,
            key_integrand=key_integrand,
            method='los',
            res=0.001,
            mode='abs',
            groupby=None,
            val_init=None,
            ref_com=ref_com,
            brightness=None,
            store=True,
            returnas=False,
        )


# #####################################################
# #####################################################
#           Script
# #####################################################


if __name__ == '__main__':
    main()
