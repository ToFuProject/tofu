

import os
import sys
import copy


import numpy as np
import matplotlib.pyplot as plt
import datastock as ds


# tofu-specific
_PATH_HERE = os.path.abspath(os.path.dirname(__file__))
_PATH_TOFU = os.path.dirname(os.path.dirname(os.path.dirname(_PATH_HERE)))


sys.path.insert(0, _PATH_TOFU)
import tofu as tf
sys.path.pop(0)


#######################################################
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


# #######################################
#              Apertures
# #######################################


def _apertures():

    start, vect, v0, v1 = _ref_line()

    # -------------------------
    # ap0 : planar from outline
    # -------------------------

    out0 = 0.01 * np.r_[-1, 1, 1, -1]
    out1 = 0.005 * np.r_[-1, -1, 1, 1]
    cent = start + 0.15 * vect
    nin, e0, e1 = _nine0e1_from_orientations(
        vect=vect,
        v0=v0,
        v1=v1,
        theta=-np.pi/20.,
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

    # ---------------------
    # ap1 : planar from 3d
    # ---------------------

    out0 = 0.05 * np.r_[-1, 1, 0]
    out1 = 0.05 * np.r_[-0.5, -0.5, 0.5]
    cent = start + 0.20 * vect
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

    # ---------------------
    # ap2 : non-planar
    # ---------------------

    out0 = 0.01 * np.r_[-1, 1, 2, 0, -2]
    out1 = 0.01 * np.r_[-1, -1, 0, 1, 0]
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

    # ----------------------------------
    # lap : non-planar for collimator
    # ----------------------------------

    out0 = 0.005 * np.r_[-1, 1, 1, -1]
    out1 = 0.003 * np.r_[-1, -1, 1, 1]

    nin, e0, e1 = _nine0e1_from_orientations(
        vect=vect,
        v0=v0,
        v1=v1,
        theta=0.,
        phi=0.,
    )

    npix = 10
    nins = np.full((npix, 3), np.nan)
    e0s = np.full((npix, 3), np.nan)
    e1s = np.full((npix, 3), np.nan)
    for ii, tt in enumerate(np.pi/20 * np.linspace(-1, 1, npix)):
        nins[ii, :], e0s[ii, :], e1s[ii, :] = _nine0e1_from_orientations(
            vect=nin,
            v0=e0,
            v1=e1,
            theta=tt,
            phi=0.,
        )

    # kl0 = 0.02 * np.linspace(-1, 1, npix)
    klin = 0.2
    delta = 0.05
    cents_x = cent[0] - (klin - delta) * nins[:, 0]    # + kl0 * e0[0]
    cents_y = cent[1] - (klin - delta) * nins[:, 1]    # + kl0 * e0[1]
    cents_z = cent[2] - (klin - delta) * nins[:, 2]    # + kl0 * e0[2]

    dap3 = {
        f'lap{ii}': {
            'outline_x0': out0,
            'outline_x1': out1,
            'cent': np.r_[cents_x[ii], cents_y[ii], cents_z[ii]],
            'nin': nins[ii, :],
            'e0': e0s[ii, :],
            'e1': e1s[ii, :],
        }
        for ii in range(npix)
    }

    # ------------
    # dout
    # -------------

    dout = {'ap0': ap0, 'ap1': ap1, 'ap2': ap2}
    dout.update(dap3)

    return dout


# #######################################
#              Filters
# #######################################


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


# #######################################
#              Cameras
# #######################################


def _cameras():

    start, vect, v0, v1 = _ref_line()

    # ---------------------
    # c0: 1d non-parallel
    # ---------------------

    out0 = 0.002 * np.r_[-1, 1, 1, -1]
    out1 = 0.005 * np.r_[-1, -1, 1, 1]
    cent = start + 0.01 * vect

    nin, e0, e1 = _nine0e1_from_orientations(
        vect=vect,
        v0=v0,
        v1=v1,
        theta=0.,
        phi=0.,
    )

    npix = 10
    nins = np.full((npix, 3), np.nan)
    e0s = np.full((npix, 3), np.nan)
    e1s = np.full((npix, 3), np.nan)
    for ii, tt in enumerate(np.pi/20 * np.linspace(-1, 1, 10)):
        nins[ii, :], e0s[ii, :], e1s[ii, :] = _nine0e1_from_orientations(
            vect=nin,
            v0=e0,
            v1=e1,
            theta=tt,
            phi=0.,
        )

    # kl0 = 0.02*np.linspace(-1, 1, 10)
    # klin = 0.01 * np.linspace(-1, 1, 10)**2
    klin = 0.2
    cents_x = cent[0] - klin * nins[:, 0]    # + kl0 * e0[0]
    cents_y = cent[1] - klin * nins[:, 1]    # + kl0 * e0[1]
    cents_z = cent[2] - klin * nins[:, 2]    # + kl0 * e0[2]

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

    # -------------------------
    # c1: 1d parallel coplanar
    # -------------------------

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

    # ---------------------
    # c2: 2d
    # ---------------------

    out0 = 0.005 * np.r_[-1, 1, 1, -1]
    out1 = 0.005 * np.r_[-1, -1, 1, 1]
    cent = start + 0.01 * vect

    cent0 = 0.02*np.linspace(-1, 1, 5)
    cent1 = 0.02*np.linspace(-1, 1, 5)

    nin, e0, e1 = _nine0e1_from_orientations(
        vect=vect,
        v0=v0,
        v1=v1,
        theta=-np.pi/20,
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
        'cam0000': copy.deepcopy(c0),
        'cam00000': copy.deepcopy(c0),
    }


# #######################################
#              Crystals
# #######################################


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


# #######################################
#              Diagnostics
# #######################################


def _diagnostics_broadband():

    # d0: single 1d camera
    d0 = {'doptics': 'cam0'}

    # d1: single 2d camera
    d1 = {'doptics': 'cam1', 'resolution': 0.2}

    # d2: single 2d camera
    d2 = {'doptics': 'cam2'}

    # d3: 1d + 1 aperture
    d3 = {'doptics': ['cam00', 'ap0']}

    # d4: 2d + 1 aperture
    d4 = {'doptics': ['cam11', 'ap0']}

    # d5: 2d + 1 aperture
    d5 = {'doptics': ['cam22', 'ap0']}

    # d6: 1d + multiple apertures
    d6 = {'doptics': ['cam000', 'ap0', 'filt0', 'ap2']}

    # d7: 1d parallel coplanar + multiple apertures
    d7 = {'doptics': ['cam111', 'ap0', 'filt0', 'ap2']}

    # d8: 2d + multiple apertures
    d8 = {'doptics': ['cam222', 'ap0', 'filt0', 'ap2']}

    # # d9: 2d + spherical crystal
    # d9 = {'optics': ('c3','cryst0')}

    # # d10: 2d + cylindrical crystal + slit
    # d10 = {'optics': ('c3','cryst1', 'slit0')}

    # # d11: 2d + toroidal crystal + slit
    # d11 = {'optics': ('c3','cryst2', 'slit1')}

    # d12: 1d collimator camera, one aperture per pixel
    d12 = {
        'doptics': tuple(
            ['cam0000'] + [k1 for k1 in [f'lap{ii}' for ii in range(10)]]
        ),
    }

    # d13: 1d collimator-hybrid camera
    d13 = {
        'doptics': {
            'cam00000': {
                'optics': [f'lap{ii}' for ii in range(10)] + ['ap0'],
                'paths': np.concatenate(
                    (np.eye(10), np.ones((10, 1))),
                    axis=1,
                ),
            },
        },
    }

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
        'd12': d12,
        'd13': d13,
    }


# #############################################
# #############################################
#       Get conf
# #############################################


def get_config():

    # get config
    conf = tf.load_config('SPARC')
    conf.remove_Struct(Cls='PFC', Name='ICRH0')

    # IRCH0 for get_touch, shifted
    conf_touch = tf.load_config('SPARC')
    poly = conf_touch.PFC.ICRH0.Poly
    poly[0, :] = poly[0, :] - 0.05
    conf_touch.remove_Struct(Cls='PFC', Name='ICRH0')
    conf_touch.add_Struct(
        Cls='PFC',
        Name='ICRH0',
        Poly=poly,
        Lim=np.r_[20, 340]*np.pi/180.,
        dextraprop={'visible': True},
    )

    return conf, conf_touch


# #############################################
# #############################################
#       Make Diags
# #############################################


def add_diags_broadband(
    coll=None,
    conf=None,
    conf_touch=None,
    compute=None,
):

    # ------------
    # instanciate
    # ------------

    if coll is None:
        coll = tf.data.Collection()

    # --------------
    # get inputs
    # --------------

    ddiag = _diagnostics_broadband()

    # get dict
    dapertures = _apertures()
    dfilters = _filters()
    dcameras = _cameras()
    # dconfig = _configurations()

    # ---------------
    # loop on diags
    # ---------------

    wdiag = coll._which_diagnostic
    wcam = coll._which_cam
    for kdiag, vdiag in ddiag.items():

        # already in => skip
        if kdiag in coll.dobj.get(wdiag, {}).keys():
            continue

        # -----------------------
        # add optics for each cam

        if isinstance(vdiag['doptics'], str):
            kcam = vdiag['doptics']
            dop = {kcam: {'optics': []}}
        elif isinstance(vdiag['doptics'], (list, tuple)):
            kcam = vdiag['doptics'][0]
            lop = vdiag['doptics'][1:]
            dop = {kcam: {'optics': lop}}
        else:
            dop = vdiag['doptics']

        # loop on kcam, lop
        for kcam, voptics in dop.items():

            # optics
            for kop in voptics['optics']:

                # optics class
                if 'ap' in kop:
                    opcls = 'aperture'
                    dop = dapertures
                else:
                    opcls = 'filter'
                    dop = dfilters
                if coll.dobj.get(opcls, {}).get(kop) is not None:
                    continue

                # add optic
                getattr(coll, f'add_{opcls}')(key=kop, **dop[kop])

            # camera
            if coll.dobj.get(wcam, {}).get(kcam) is None:
                if 'cam0' in kcam or 'cam1' in kcam:
                    coll.add_camera_1d(key=kcam, **dcameras[kcam])
                else:
                    coll.add_camera_2d(key=kcam, **dcameras[kcam])

        # --------------
        # add diag

        coll.add_diagnostic(
            key=kdiag,
            config=conf,
            reflections_nb=1,
            reflections_type='specular',
            compute=compute,
            **vdiag,
        )

    return coll


def add_diags_spectro(
    coll=None,
    conf=None,
    key_diag=None,
    compute=None,
):

    # ------------
    # instanciate
    # ------------

    if coll is None:
        coll = tf.data.Collection()

    # key_diag
    if key_diag is not None:
        if isinstance(key_diag, str):
            key_diag = [key_diag]

    # --------------
    # get inputs
    # --------------

    # get dict
    dapertures = _apertures()
    dcrystals = _crystals()
    dconfig = _configurations()

    # --------------
    # add aperture
    # --------------

    for k0, v0 in dapertures.items():
        if k0 not in ['ap0']:
            continue
        coll.add_aperture(key=k0, **v0)

    # --------------
    # add crystals
    # --------------

    for k0, v0 in dcrystals.items():
        coll.add_crystal(key=k0, **v0)

    # --------------
    # add crystals optics
    # --------------

    coll.doptics = {}
    for k0, v0 in dcrystals.items():

        for ii, cc in enumerate(dconfig[k0]):

            apdim = [100e-6, 8e-2] if cc != 'pinhole' else None
            pinrad = 1e-2 if cc == 'pinhole' else None

            loptics = coll.get_crystal_ideal_configuration(
                key=k0,
                configuration=cc,
                # parameters
                cam_on_e0=False,
                cam_tangential=True,
                cam_dimensions=[8e-2, 5e-2],
                focal_distance=2.,
                # store
                store=True,
                key_cam=f'{k0}_cam{ii}',
                aperture_dimensions=apdim,
                pinhole_radius=pinrad,
                cam_pixels_nb=[5, 3],
                # returnas
                returnas=list,
            )

            if 'cryst1_slit' in loptics:
                loptics.append('ap0')

            # add diag
            gtype = coll.dobj['crystal'][k0]['dgeom']['type']
            if gtype not in ['toroidal']:
                coll.doptics.update({
                    f'{k0}_{ii}': loptics,
                })

    # -----------------
    # add diagnostic
    # -----------------

    for ii, (k0, v0) in enumerate(coll.doptics.items()):
        kdiag = f'sd{ii}'
        if key_diag is None or kdiag in key_diag:
            coll.add_diagnostic(
                key=kdiag,
                doptics=v0,
                config=conf,
                compute_vos_from_los=True,
            )
        else:
            msg = f"not added: diag '{kdiag}'\n"
            print(msg)

    # add toroidal
    # coll.add_diagnostic(optics=['cryst2-cam0', 'cryst3'])
    return coll


# ####################################################
# ####################################################
#                 plot
# ####################################################


def _plot(
    coll=None,
    key_diag=None,
    conf=None,
    close=None,
    spectro=None,
):

    # --------------
    # key_diag
    # --------------

    key_diag = _get_key_diag(
        coll=coll,
        key_diag=key_diag,
        spectro=spectro,
    )

    # --------------
    # plot
    # --------------

    for ii, k0 in enumerate(key_diag):

        _ = coll.plot_diagnostic(
            k0,
            data='etendue',
            proj=(
                None if ii % 3 == 0
                else ('cross' if ii % 3 == 1 else ['cross', 'hor'])
            ),
            plot_config=conf,
        )

        # close
        if close is not False:
            plt.close('all')

    return


# ####################################################
# ####################################################
#                 Tests
#             Sinogram
# ####################################################


def _sinogram(
    coll=None,
    conf=None,
    key_diag=None,
    close=None,
):

    # --------------
    # key_diag
    # --------------

    key_diag = _get_key_diag(
        coll=coll,
        key_diag=key_diag,
        spectro=None,
    )

    lrays = list(coll.dobj['rays'].keys())
    ldiag = [
        k0 for k0 in key_diag
        if any([
            v1.get('los') is not None
            for v1 in coll.dobj['diagnostic'][k0]['doptics'].values()
        ])
    ]

    lk = [lrays] + ldiag

    # --------------
    # sinogram
    # --------------

    for ii, k0 in enumerate(lk):

        _, _ = coll.get_sinogram(
            key=k0,
            ang='theta' if ii % 2 == 0 else 'xi',
            ang_units='deg' if ii % 3 == 0 else 'radian',
            impact_pos=ii % 3 != 0,
            R0=2.4 if ii % 3 != 1 else None,
            config=None if ii % 3 != 1 else conf,
            pmax=None if ii % 3 == 0 else 5,
            plot=True,
            verb=2,
        )

        if close is not False:
            plt.close('all')

    return


# ####################################################
# ####################################################
#                 Tests
#           add_rays_from_diagnostic
# ####################################################


def _add_rays_from_diagnostic(
    coll=None,
    conf=None,
):

    wdiag = coll._which_diagnostic
    for ii, (k0, v0) in enumerate(coll.dobj[wdiag].items()):
        noptics = any([
            len(v1['optics']) == 0 for v1 in v0['doptics'].values()
        ])
        if v0['is2d'] or v0['spectro'] or noptics:
            continue
        dsamp = {'dedge': {'res': 'max'}, 'dsurface': {'nb': 3}}
        dout = coll.add_rays_from_diagnostic(
            key=k0,
            dsampling_pixel=dsamp,
            dsampling_optics=dsamp,
            optics=-1,
            config=conf,
            store=(ii % 2 == 0),
        )
        assert isinstance(dout, dict) or dout is None

    return


# ####################################################
# ####################################################
#                 Tests
#             add_single_point_camera2d
# ####################################################


def _add_single_point_camera2d(
    coll=None,
    kdiag=None,
    conf_touch=None,
):

    # ------------
    # choose diag
    # ------------

    # kdiag
    wdiag = coll._which_diagnostic
    lok = [
        k0 for k0, v0 in coll.dobj.get(wdiag, {}).items()
        if v0['spectro'] is False
    ]
    if len(lok) == 0:
        return
    kdiag = ds._generic_check._check_var(
        kdiag, 'kdiag',
        types=str,
        allowed=lok,
        default='diag5',
    )

    wrays = coll._which_rays
    klos = [
        k0 for k0 in coll.dobj.get(wrays, {}).keys()
        if k0.startswith(f'{kdiag}_')
        and k0.endswith('_los')
    ][0]
    krays = klos.replace('_los', '_rays')

    # ------------
    # add cam2d
    # ------------

    coll.add_single_point_camera2d(
        key='ptcam',
        key_rays=klos,
        angle0=55,
        angle1=55,
        config=conf_touch,
    )

    # -------------------
    # add rays from diag
    # -------------------

    # add rays
    dsamp = {'dedge': {'res': 'min'}, 'dsurface': {'nb': 3}}
    coll.add_rays_from_diagnostic(
        key=kdiag,
        dsampling_pixel=dsamp,
        dsampling_optics=dsamp,
        optics=-1,
        config=conf_touch,
        store=True,
        strict=None,
        key_rays=krays,
        overwrite=None,
    )

    # ---------------------
    # get angles from rays
    # ---------------------

    # simple
    dout = coll.get_rays_angles_from_single_point_camera2d(
        key_single_pt_cam='ptcam',
        key_rays=klos,
        return_indices=False,
    )

    # with indices and no convex hull
    dout = coll.get_rays_angles_from_single_point_camera2d(
        key_single_pt_cam='ptcam',
        key_rays=krays,
        return_indices=True,
        convex_axis=False,
    )

    # with indices and convesx axis
    dout = coll.get_rays_angles_from_single_point_camera2d(
        key_single_pt_cam='ptcam',
        key_rays=krays,
        return_indices=True,
        convex_axis=(-1, -2),
    )
    assert isinstance(dout, dict)

    return


# ####################################################
# ####################################################
#                 Tests
#             Reverse ray tracing
# ####################################################


def _reverse_ray_tracing(
    coll=None,
):

    wdiag = coll._which_diagnostic
    for ii, (k0, v0) in enumerate(coll.dobj[wdiag].items()):
        lcam = coll.dobj[wdiag][k0]['camera']
        doptics = coll.dobj[wdiag][k0]['doptics']
        if len(doptics[lcam[0]]['optics']) == 0:
            continue
        if not coll.dobj[wdiag][k0]['spectro']:
            continue

        # Get points
        ptsx, ptsy, ptsz = coll.get_rays_pts(k0)
        kcryst = list(doptics.values())[0]['optics'][0]
        lamb = coll.get_crystal_bragglamb(
            key=kcryst,
            lamb=None,
            bragg=None,
            norder=None,
            rocking_curve=None,
        )[1]

        _ = coll.get_raytracing_from_pts(
            key=k0,
            key_cam=None,
            key_mesh=None,
            res_RZ=None,
            res_phi=None,
            ptsx=ptsx[-1, ...],
            ptsy=ptsy[-1, ...],
            ptsz=ptsz[-1, ...],
            n0=3,
            n1=3,
            lamb0=lamb,
            res_lamb=None,
            rocking_curve=None,
            append=None,
            plot=True,
            dax=None,
            plot_pixels=None,
            plot_config=None,
            vmin=None,
            vmax=None,
            aspect3d=None,
            elements=None,
            colorbar=None,
        )

    return


# ####################################################
# ####################################################
#                 Tests
#           compute_diagnostic_vos
# ####################################################


def _compute_vos(
    coll=None,
    key_diag=None,
    conf=None,
    spectro=False,
    # options
    res_RZ=None,
    res_phi=None,
    n0=None,
    n1=None,
    res_lamb=None,
    lamb=None,
    compact_lamb=None,
):

    # ---------------
    # inputs
    # ---------------

    # key_diag
    key_diag = _get_key_diag(
        coll=coll,
        key_diag=key_diag,
        spectro=spectro,
    )

    # res_RZ
    res_RZ = float(ds._generic_check._check_var(
        res_RZ, 'res_RZ',
        types=(int, float),
        default=0.015 if spectro is True else 0.04,
        sign='>0',
    ))

    # res_phi
    res_phi = float(ds._generic_check._check_var(
        res_phi, 'res_phi',
        types=(int, float),
        default=0.015 if spectro is True else 0.04,
        sign='>0',
    ))

    # res_lamb
    res_lamb = float(ds._generic_check._check_var(
        res_lamb, 'res_lamb',
        types=(int, float),
        default=1e-13,
        sign='>0',
    ))

    # n0
    n0 = ds._generic_check._check_var(
        n0, 'n0',
        types=int,
        default=3,
        sign='>0',
    )

    # n1
    n1 = ds._generic_check._check_var(
        n1, 'n1',
        types=int,
        default=3,
        sign='>0',
    )

    # ---------------
    # add mesh
    # ---------------

    _add_mesh(
        coll=coll,
        conf=conf,
    )

    # -----------------
    # loop on key_diag
    # -----------------

    # compute vos
    for ii, k0 in enumerate(key_diag):

        keep_cross = True
        keep_hor = (ii % 2 == 0)
        keep_3d = ii == 0
        if compact_lamb is None:
            compact_lambi = (ii % 2 == 0)
        else:
            compact_lambi = compact_lamb

        coll.compute_diagnostic_vos(
            # keys
            key_diag=k0,
            key_mesh=None,
            # resolution
            res_RZ=res_RZ,
            res_phi=res_phi,
            # keep
            keep_cross=keep_cross,
            keep_hor=keep_hor,
            keep_3d=keep_3d,
            # compact
            compact_lamb=compact_lambi,
            # spectro
            n0=n0,
            n1=n1,
            res_lamb=res_lamb,
            lamb=lamb,
            visibility=False,
            overwrite=True,
            replace_poly=True,
            store=True,
        )

        # testing vos
        if keep_cross is True and keep_3d is True:
            if spectro is True:
                pattern = 'ph'
            else:
                pattern = 'sa'

            k_3d = [
                kk for kk in coll.ddata.keys()
                if kk.endswith(f'vos_{pattern}_3d')
                and k0 in kk
            ][0]
            k_cross = [
                kk for kk in coll.ddata.keys()
                if kk.endswith(f'vos_{pattern}_cross')
                and k0 in kk
            ][0]

            v_3d = coll.ddata[k_3d]['data']
            v_cross = coll.ddata[k_cross]['data']
            sum_3d = v_3d.sum(axis=(-2, -1) if spectro else -1)
            sum_cross = np.sum(v_cross, axis=(-2, -1) if spectro else -1)

            if (
                (sum_3d.shape != sum_cross.shape)
                or (not np.allclose(sum_3d, sum_cross, equal_nan=True))
            ):
                msg = (
                    "Mismatch between vos_3d and vos_cross (spectro)!\n"
                    f"\t- diag: '{k0}'\n"
                )
                raise Exception(msg)
    return


def _get_key_diag(
    coll=None,
    key_diag=None,
    spectro=None,
):

    wdiag = coll._which_diagnostic
    lok = [
        k0 for k0, v0 in coll.dobj.get(wdiag, {}).items()
        if (
            spectro is None
            or v0['spectro'] == spectro
        )
        and len(v0['doptics'][v0['camera'][0]]['optics']) > 0
    ]
    if isinstance(key_diag, str):
        key_diag = [key_diag]
    key_diag = ds._generic_check._check_var_iter(
        key_diag, 'key_diag',
        types=list,
        types_iter=str,
        allowed=lok,
    )

    return key_diag


# ####################################################
# ####################################################
#                 Tests
#             save to json
# ####################################################


def _save_to_json(
    coll=None,
    remove=True,
):

    wdiag = coll._which_diagnostic
    for ii, (k0, v0) in enumerate(coll.dobj[wdiag].items()):

        lcam = coll.dobj[wdiag][k0]['camera']
        doptics = coll.dobj[wdiag][k0]['doptics']
        if len(doptics[lcam[0]]['optics']) == 0 or k0 == 'diag6':
            continue

        # saving
        pfe = os.path.join(_PATH_HERE, f"{k0}.json")

        try:
            coll.save_diagnostic_to_file(k0, pfe_save=pfe)

            # reloading
            _ = tf.data.load_diagnostic_from_file(pfe)

        # remove file
        finally:
            if remove is True:
                try:
                    os.remove(pfe)
                except Exception:
                    pass

    return


# ####################################################
# ####################################################
#                 Tests
#             save to npz
# ####################################################


def _save_to_npz(
    coll=None,
    remove=True,
):

    wdiag = coll._which_diagnostic
    for ii, (k0, v0) in enumerate(coll.dobj[wdiag].items()):

        # pfe
        pfe = os.path.join(_PATH_HERE, f'test_diag_{k0}.npz')

        # save
        try:
            coll.save(pfe)

            # reloading
            _ = tf.data.load(pfe)

        finally:
            # remove file
            if remove is True:
                try:
                    os.remove(pfe)
                except Exception:
                    pass
    return


# ####################################################
# ####################################################
#                 VOS: plot_coverage
# ####################################################


def _plot_coverage(
    coll=None,
    key_diag=None,
    conf=None,
    spectro=False,
    close=None,
):

    # --------------
    # key_diag
    # --------------

    key_diag = _get_key_diag(
        coll=coll,
        key_diag=key_diag,
        spectro=spectro,
    )

    # --------------
    # plot
    # --------------

    for ii, k0 in enumerate(key_diag):

        try:
            _ = coll.plot_diagnostic_geometrical_coverage(k0, plot_config=conf)
        except NotImplementedError as err:
            if 'compact_lamb' in str(err):
                pass
            else:
                raise err

        if close is not False:
            plt.close('all')

    return


# ####################################################
# ####################################################
#           VOS: plot_coverage_slice
# ####################################################


def _plot_coverage_slice(
    coll=None,
    key_diag=None,
    conf=None,
    spectro=False,
    close=None,
    res=None,
    isZ=None,
):

    # --------------
    # inputs
    # --------------

    # key_diag
    key_diag = _get_key_diag(
        coll=coll,
        key_diag=key_diag,
        spectro=spectro,
    )

    # res
    res = ds._generic_check._check_var(
        res, 'res',
        default=0.05,
        types=(int, float),
        sign='>0',
    )

    # --------------
    # plot
    # --------------

    for ii, k0 in enumerate(key_diag):

        if isZ is None:
            c0 = ii % 2 == 0
        else:
            c0 = isZ

        if c0:
            Z = 0
            Dphi = np.pi/6 * np.r_[-1, 1]
            vect = None
            margin_par = None
            margin_perp = None
        else:
            Z = None
            Dphi = None
            vect = 'nin'
            margin_par = 0.5
            margin_perp = 0.1

        _ = coll.plot_diagnostic_geometrical_coverage_slice(
            key_diag=k0,
            key_cam=None,
            indch=None,
            indref=None,
            res=res,
            margin_par=margin_par,
            margin_perp=margin_perp,
            vect=vect,
            segment=0,
            key_mesh=None,
            phi=None,
            Z=Z,
            DR=None,
            DZ=None,
            Dphi=Dphi,
            adjust_phi=None,
            config=conf,
            visibility=True,
            n0=None,
            n1=None,
            res_lamb=None,
            verb=None,
            plot=True,
            indplot=None,
            dax=None,
            plot_config=None,
            fs=None,
            dmargin=None,
            dvminmax=None,
            markersize=None,
        )

        if close is not False:
            plt.close('all')

    return


# ####################################################
# ####################################################
#           Add mesh / bsplines
# ####################################################


def _add_mesh(
    coll=None,
    conf=None,
):

    wm = coll._which_mesh
    if len(coll.dobj.get(wm, {}).keys()) > 0:
        return

    key_mesh = 'm0'
    wmesh = coll._which_mesh
    if key_mesh not in coll.dobj.get(wmesh, {}).keys():
        coll.add_mesh_2d_rect(
            key=key_mesh,
            res=0.08,
            crop_poly=conf,
            deg=1,
        )

    return


# ####################################################
# ####################################################
#           Add emiss
# ####################################################


def _add_emiss(
    coll=None,
    conf=None,
    spectro=None,
):

    # ------------
    # add mesh
    # ------------

    _add_mesh(
        coll=coll,
        conf=conf,
    )

    # check emis
    if len([kk for kk in coll.ddata.keys() if 'emis' in kk]) > 0:
        return

    # -----------------
    # add bsplines
    # -----------------

    wbs = coll._which_bsplines
    key_bs = list(coll.dobj.get(wbs, {}).keys())[0]

    # R, Z
    kR, kZ = coll.dobj[wbs][key_bs]['apex']
    R = coll.ddata[kR]['data']
    Z = coll.ddata[kZ]['data']

    # -------------
    # emissivity profile with 1/1 mode
    # -------------

    # time
    t0 = 0
    t1 = 5
    t = np.linspace(t0, t1, 21)

    # fixed basis
    R0, Z0 = 1.8, 0
    DR, DZ = 0.2, 0.4
    e0 = np.exp(-(R[:, None] - R0)**2/DR**2 - (Z[None, :] - Z0)**2/DZ**2)

    # rotating mode
    theta = np.arctan2(np.sin(4*np.pi*t/(t1-t0)), np.cos(4*np.pi*t/(t1-t0)))
    r0R, r0Z = 0.2, 0.3
    dr, dz = 0.1, 0.15
    Rm = R0 + r0R*np.cos(theta)[:, None, None]
    Zm = Z0 + r0Z*np.sin(theta)[:, None, None]
    e1 = np.exp(
        - (R[None, :, None] - Rm)**2 / dr**2
        - (Z[None, None, :] - Zm)**2 / dz**2
    )

    emis = e0[None, :, :] + 0.3*e1

    # -------------
    # Store
    # -------------

    coll.add_data('t', data=t, ref='nt', units='s', dim='time')
    coll.add_data('theta', data=theta, ref='nt', units='rad', dim='angle')
    coll.add_data('Rm', data=Rm.ravel(), ref='nt', units='m', dim='distance')
    coll.add_data('Zm', data=Zm.ravel(), ref='nt', units='m', dim='distance')

    if spectro is False:
        coll.add_data(
            'emis',
            data=emis,
            ref=('nt', key_bs),
            units='ph/m3/sr/s',
            dim='emis',
        )

    else:
        # lamb
        lamb0 = 3.91e-10
        lamb1 = 4.01e-10
        lamb = np.linspace(lamb0, lamb1, 3000)
        coll.add_mesh_1d('mlamb', knots=lamb, deg=1, units='m')

        # spectral emis
        lambm = 0.5*(lamb0 + lamb1)
        dlamb = 0.1*(lamb1 - lamb0)
        elamb = 1 + 0.3*np.exp(-(lamb - lambm)**2/dlamb**2)

        # overall
        emis = emis[..., None] * elamb[None, None, None, :]

        coll.add_data(
            'emis',
            data=emis,
            ref=('nt', key_bs, 'mlamb_bs1'),
            units='ph/m3/sr/s/m',
            dim='emis',
        )

    return


# ####################################################
# ####################################################
#           Synthetic signal
# ####################################################


def _synthetic_signal(
    coll=None,
    key_diag=None,
    spectro=None,
    method=None,
    res=None,
    conf=None,
):

    # --------------
    # check emis
    # --------------

    _add_emiss(
        coll=coll,
        conf=conf,
        spectro=spectro,
    )

    # --------------
    # inputs
    # --------------

    # key_diag
    key_diag = _get_key_diag(
        coll=coll,
        key_diag=key_diag,
        spectro=spectro,
    )

    # --------------
    # compute
    # --------------

    for kdiag in key_diag:

        dout = coll.compute_diagnostic_signal(
            key=None,
            key_diag=kdiag,
            key_integrand='emis',
            method=method,
            key_ref_spectro='mlamb_bs1_nbs',   # None would work too
            res=res,
            mode=None,
            groupby=None,
            val_init=None,
            ref_com=None,
            brightness=False,
            spectral_binning=None,
            dvos=None,
            verb=None,
            timing=None,
            store=True,
            returnas=None,
        )
        assert dout is None

        if spectro and method in ['vos', 'vos_cross']:
            dproj = coll.check_diagnostic_vos_proj(kdiag)
            lcam = coll.dobj['diagnostic'][kdiag]['camera']
            if all([kcam in dproj['3d'] for kcam in lcam]):

                dout2 = coll.compute_diagnostic_signal(
                    key=None,
                    key_diag=kdiag,
                    key_integrand='emis',
                    method='vos_3d',
                    key_ref_spectro=None,
                    ref_com=None,
                    brightness=False,
                    store=False,
                    returnas=dict,
                )

                assert isinstance(dout2, dict)

    return
