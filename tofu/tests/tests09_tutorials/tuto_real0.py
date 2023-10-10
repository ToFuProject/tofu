"""
Creating and using diagnostic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

"""


import os
import copy
import numpy as np


import tofu as tf


_PATH_HERE = os.path.dirname(__file__)


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
    _add_broadband(coll, key_diag='d0', conf=conf, vos=True)

    # add collimator
    # _add_collimator(coll, conf, vos=True)

    # add 2d camera
    _add_2d(coll, key_diag='c2d', conf=conf, vos=True)

    # add PHA
    # _add_PHA(coll, conf)

    # add spectrometer
    # _add_spectrometer(coll, conf, vos=True)   # , crystals=['c0'])

    # add spectro-like without crystal
    # _add_spectrometer_like(coll, config=conf, key_diag='d02')

    # ------------------------
    # compute synthetic signal

    _compute_synth_signal(
        coll,
        ldiag=['d0', 'c2d'],
        method='vos',
        spectral_binning=True,
    )

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
        res=0.10,
        crop_poly=conf,
        units='m',
        deg=1,
    )

    # coll.add_mesh_2d_rect(
        # key='m1',
        # res=0.03,
        # crop_poly=conf,
        # units='m',
        # deg=0,
    # )

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
    key_diag=None,
    conf=None,
    vos=None,
):

    # ---------------------
    # add 2 pinhole cameras

    # coll.add_camera_pinhole(
        # key='bb0',
        # key_pinhole=None,
        # key_diag=key_diag,
        # cam_type='1d',
        # R=3.3,
        # z=-0.6,
        # phi=0,
        # theta=3.*np.pi/4,
        # dphi=np.pi/10,
        # tilt=np.pi/2,
        # focal=0.1,
        # pix_nb=10,
        # pix_size=3e-3,
        # pix_spacing=5e-3,
        # pinhole_radius=None,
        # pinhole_size=[2e-3, 1e-3],
        # reflections_nb=0,
        # reflections_type=None,
        # compute=False,
        # config=conf,
    # )

    coll.add_camera_pinhole(
        key='bb1',
        key_pinhole=None,
        key_diag=key_diag,
        cam_type='1d',
        R=3.3,
        z=0.6,
        phi=0,
        theta=-3.*np.pi/4,
        dphi=np.pi/10,
        tilt=np.pi/2,
        focal=0.1,
        pix_nb=10,
        pix_size=3e-3,
        pix_spacing=5e-3,
        pinhole_radius=None,
        pinhole_size=[3e-3, 2e-3],
        reflections_nb=0,
        reflections_type=None,
        compute=True,
        config=conf,
    )

    if vos is True:
        coll.compute_diagnostic_vos(
            key_diag=key_diag,
            key_mesh='m0',
            res_RZ=0.005,
            res_phi=0.01,
            visibility=False,
            store=True,
        )

    return


def _add_collimator(
    coll=None,
    conf=None,
    vos=None,
):

    # ---------------------
    # add 2 pinhole cameras

    # aperture 0
    dgeom = {
        'cent': np.r_[3.3, 0, 0],
        'nin': np.r_[-1, 0, 0],
        'e0': np.r_[0, -1, 0],
        'e1': np.r_[0, 0, 1],
        'outline_x0': 0.005 * np.r_[-1, 1, 1, -1],
        'outline_x1': 0.005 * np.r_[-1, -1, 1, 1],
    }

    coll.add_aperture(
        key='coll_ap00',
        **dgeom,
    )

    # add camera
    del dgeom['cent']
    dgeom['cents_x'] = 3.5 * np.r_[1, 1, 1]
    dgeom['cents_y'] = np.r_[0, 0, 0]
    dgeom['cents_z'] = np.r_[-0.05, 0, 0.05]

    coll.add_camera_1d(
        key='coll_cam',
        dgeom=dgeom,
    )

    # create diagnostic
    coll.add_diagnostic(
        key='coll',
        doptics={
            'coll_cam': [
                ('coll_ap00',),
                ('coll_ap00',),
                ('coll_ap00',),
            ],
        },
        compute=True,
    )

    if vos is True:
        coll.compute_diagnostic_vos(
            'dcoll',
            key_mesh='m0',
            res_RZ=0.01,
            res_phi=0.01,
            visibility=False,
            store=True,
        )

    return

def _add_2d(
    coll=None,
    key_diag=None,
    conf=None,
    vos=None,
):

    # ---------------------
    # add 2 pinhole cameras

    coll.add_camera_pinhole(
        key=key_diag,
        key_diag=key_diag,
        key_pinhole=None,
        cam_type='2d',
        R=3.3,
        z=-0.6,
        phi=0,
        theta=3.*np.pi/4,
        dphi=np.pi/6,
        tilt=np.pi/2,
        focal=0.1,
        pix_nb=[5, 3],
        pix_size=1e-3,
        pix_spacing=5e-3,
        pinhole_radius=5e-3,
        pinhole_size=None,
        reflections_nb=0,
        reflections_type=None,
        compute=True,
        config=conf,
    )

    if vos is True:
        coll.compute_diagnostic_vos(
            key_diag,
            key_mesh='m0',
            res_RZ=0.02,
            res_phi=0.02,
            visibility=False,
            store=True,
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


def _add_spectrometer_like(
    coll=None,
    config=None,
    key_diag=None,
):

    # --------
    # aperture

    doptics = coll.dobj['diagnostic'][key_diag]['doptics']
    key_cam = list(doptics.keys())[0]
    kcryst, kslit = doptics[key_cam]['optics']

    # ------------
    # flat crystal

    # dgeom
    dgeom = copy.deepcopy({
        k0: coll.dobj['crystal'][kcryst]['dgeom'][k0]
        for k0 in ['cent', 'nin', 'e0', 'e1']
    })
    dgeom['curve_r'] = [np.inf, np.inf]
    dgeom['extenthalf'] = 0.2 * np.r_[1, 1]

    # dmat
    dmat = copy.deepcopy({
        k0: coll.dobj['crystal'][kcryst]['dmat'][k0]
        for k0 in ['d_hkl', 'target', 'drock']
    })

    for k0 in ['angle_rel', 'power_ratio']:
        dmat['drock'][k0] = coll.ddata[dmat['drock'][k0]]['data']

    coll.add_crystal(
        key='c0_flat',
        dgeom=dgeom,
        dmat=dmat,
    )

    # ---------------
    # camera replica

    # dgeom
    dgeom = copy.deepcopy({
        k0: coll.dobj['camera'][key_cam]['dgeom'][k0]
        for k0 in ['cent', 'nin', 'e0', 'e1']
    })
    kout0, kout1 = coll.dobj['camera'][key_cam]['dgeom']['outline']
    dgeom['outline_x0'] = coll.ddata[kout0]['data']
    dgeom['outline_x1'] = coll.ddata[kout1]['data']
    cx0, cx1 = coll.dobj['camera'][key_cam]['dgeom']['cents']
    dgeom['cents_x0'] = coll.ddata[cx0]['data']
    dgeom['cents_x1'] = coll.ddata[cx1]['data']

    # add camera replica
    coll.add_camera_2d(
        key='c0_camf',
        dgeom=dgeom,
    )

    # ---------------
    # camera rotated

    dgeom = copy.deepcopy(dgeom)

    # bragg angle
    cent = coll.dobj['crystal'][kcryst]['dgeom']['cent']
    nin = coll.dobj['crystal'][kcryst]['dgeom']['nin']
    e0 = coll.dobj['crystal'][kcryst]['dgeom']['e0']
    e1 = coll.dobj['crystal'][kcryst]['dgeom']['e1']
    ang = 2*coll.get_crystal_bragglamb('c0')[0]

    # rotate cent

    dv = (dgeom['cent'] - cent)
    scain = np.sum(dv * nin)
    sca0 = np.sum(dv * e0)
    sca1 = np.sum(dv * e1)
    assert np.abs(sca1) < 1e-8

    dvbis = (
        np.cos(ang) * (scain * nin + sca0 * e0)
        + np.sin(ang) * (sca0 * nin - scain * e0)
        + sca1 * e1
    )
    dgeom['cent'] = cent + dvbis

    # unite vectors
    for k0 in ['nin', 'e0', 'e1']:
        scain = np.sum(dgeom[k0] * nin)
        sca0 = np.sum(dgeom[k0] * e0)
        sca1 = np.sum(dgeom[k0] * e1)
        dgeom[k0] = (
            np.cos(ang) * (scain * nin + sca0 * e0)
            + np.sin(ang) * (sca0 * nin - scain * e0)
            + sca1 * e1
        )

    # add rotated camera
    coll.add_camera_2d(
        key='c0_cam1',
        dgeom=dgeom,
    )

    # -----------
    # diagnostic

    # flat crystal
    coll.add_diagnostic(
        doptics={'c0_camf': ['c0_flat', kslit]},
        config=config,
        compute=True,
    )

    # no crystal
    coll.add_diagnostic(
        doptics={'c0_cam1': [kslit]},
        config=config,
        compute=True,
    )


def _add_spectrometer(
    coll=None,
    conf=None,
    crystals=None,
    vos=None,
):

    # ------------------
    # add crystal optics

    doptics = {}
    dcrystals = _crystals(coll, crystals=crystals)

    # ----------------------
    # add camera / apertures

    for k0, v0 in dcrystals.items():

        if k0 != 'c0':
            continue

        loptics = coll.get_crystal_ideal_configuration(
            key=k0,
            configuration=v0['configuration'],
            # parameters
            cam_on_e0=False,
            cam_tangential=True,
            cam_dimensions=np.r_[1028, 512]*75e-6,
            focal_distance=2.,
            defocus=0.,
            # defocus=-1.5,
            # store
            store=True,
            key_cam=f'{k0}_cam',
            aperture_dimensions=[100e-6, 1e-2],
            pinhole_radius=100e-6 if v0['configuration'] == 'pinhole' else None,
            cam_pixels_nb=[21, 11],
            # cam_pixels_nb=[41, 41],
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
            rocking_curve_fwhm=0.0001*np.pi/180 if k0 == 'c2' else None,
        )

        if vos is True:
            coll.compute_diagnostic_vos(
                list(coll.dobj['diagnostic'].keys())[-1],
                key_mesh='m0',
                res_RZ=[0.10, 0.01],
                res_phi=0.005,
                res_lamb=0.001e-10,
                n0=11,
                n1=21,
                store=True,
            )

    return


def _crystals(coll=None, crystals=None):

    # -------
    # check

    if crystals is None:
        crystals = ['c0', 'c1', 'c2']

    # -------
    # geom

    start, vect, v0, v1 = _ref_line(
        start=np.r_[17.918, -2.157, 0.043],
        vect=np.r_[-0.29770273, 0.95465862, 0.],
    )
    # start, vect, v0, v1 = _ref_line(start=np.r_[7., 0, 0.001])

    # cryst0: planar
    cent = start + 0. * vect

    nin, e0, e1 = _nine0e1_from_orientations(
        vect=vect,
        v0=v0,
        v1=v1,
        # theta=-np.pi/4,
        theta=0,
        phi=0.,
    )

    dc = {}

    # c1: cylindrical (von hamos)
    if 'c0' in crystals:

        # load rocking curve
        pfe = os.path.join(_PATH_HERE, 'Ge242.txt')
        out = np.loadtxt(pfe)
        drock = {
            'angle_rel': out[:, 0],
            'power_ratio': out[:, 1],
        }

        size = 1.e-2
        rc = 1.03

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
            # 'dmat': 'Quartz_110',
            'dmat': {
                'material': 'Germanium',
                'name': 'Ge224',
                'miller': np.r_[2,2,4],
                # 'd_hkl': 0.944e-10 / (2*np.sin(24.2*np.pi/180.)),
                'target': {'lamb': 0.944e-10},
            },
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


def _compute_synth_signal(
    coll=None,
    ldiag=None,
    method=None,
    spectral_binning=None,
):

    # -------------
    # list of diags

    if ldiag is None:
        ldiag = list(coll.dobj['diagnostic'])

    # ------------
    # loop n diags

    for k0 in ldiag:

        if k0 not in coll.dobj['diagnostic'].keys():
            continue

        # params
        if k0 in ['d0', 'c2d']:
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
            method=method,
            res=0.001,
            mode='abs',
            groupby=None,
            val_init=None,
            ref_com=ref_com,
            brightness=None,
            spectral_binning=spectral_binning,
            verb=True,
            timing=False,
            store=True,
            returnas=False,
        )


# #####################################################
# #####################################################
#           Script
# #####################################################


if __name__ == '__main__':
    main()
