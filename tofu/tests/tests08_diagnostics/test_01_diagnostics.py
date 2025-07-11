# -*- coding: utf-8 -*-
"""
This module contains tests for tofu.geom in its structured version
"""


# Built-in
import os


# Standard
import matplotlib.pyplot as plt


# local
from . import tests_inputs as _inputs


#######################################################
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
#######################################################
#
#     Tests DIag broadbadn without VOS (except computing)
#
#######################################################


class Test01_Diagnostic_Broadband():

    def setup_method(self):

        self.conf, self.conf_touch = _inputs.get_config()
        self.coll = _inputs.add_diags_broadband(
            conf=self.conf,
            conf_touch=self.conf_touch,
        )

    # ----------
    # tests

    """
    def test01_etendues(self, res=np.r_[0.005, 0.003, 0.001]):
        for k0, v0 in self.coll.dobj['diagnostic'].items():
            if len(v0['optics']) == 1 or v0['spectro'] is not False:
                continue
            self.coll.compute_diagnostic_etendue_los(
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
        for k0, v0 in self.coll.dobj['aperture'].items():
            _ = self.coll.get_optics_outline(k0)

        # camera
        for k0, v0 in self.coll.dobj['camera'].items():
            _ = self.coll.get_optics_outline(k0)

    def test03_plot(self):
        for ii, (k0, v0) in enumerate(self.coll.dobj['diagnostic'].items()):
            dax = self.coll.plot_diagnostic(
                k0,
                data='etendue',
                proj=(
                    None if ii % 3 == 0
                    else ('cross' if ii % 3 == 1 else ['cross', 'hor'])
                ),
            )
            plt.close('all')
            del dax

    def test04_sinogram(self):

        lrays = list(self.coll.dobj['rays'].keys())
        ldiag = [
            k0 for k0, v0 in self.coll.dobj['diagnostic'].items()
            if any([
                v1.get('los') is not None
                for v1 in v0['doptics'].values()
            ])
        ]
        lk = [lrays] + ldiag
        for ii, k0 in enumerate(lk):

            dout, dax = self.coll.get_sinogram(
                key=k0,
                ang='theta' if ii % 2 == 0 else 'xi',
                ang_units='deg' if ii % 3 == 0 else 'radian',
                impact_pos=ii % 3 != 0,
                R0=2.4 if ii % 3 != 1 else None,
                config=None if ii % 3 != 1 else self.conf,
                pmax=None if ii % 3 == 0 else 5,
                plot=True,
                verb=2,
            )
            plt.close('all')
            del dax

    def test07_add_single_point_camera2d(self):

        self.coll.add_single_point_camera2d(
            key='ptcam',
            key_rays='diag5_cam22_los',
            angle0=55,
            angle1=55,
            config=self.conf_touch,
        )

        # add rays
        dsamp = {'dedge': {'res': 'min'}, 'dsurface': {'nb': 3}}
        self.coll.add_rays_from_diagnostic(
            key='diag5',
            dsampling_pixel=dsamp,
            dsampling_optics=dsamp,
            optics=-1,
            config=self.conf_touch,
            store=True,
            strict=None,
            key_rays='diag5_cam22_rays',
            overwrite=None,
        )

        # get angles from rays
        dout = self.coll.get_rays_angles_from_single_point_camera2d(
            key_single_pt_cam='ptcam',
            key_rays='diag5_cam22_los',
            return_indices=False,
        )

        # get angles from rays
        dout = self.coll.get_rays_angles_from_single_point_camera2d(
            key_single_pt_cam='ptcam',
            key_rays='diag5_cam22_rays',
            return_indices=True,
            convex_axis=False,
        )

        # get angles from rays
        dout = self.coll.get_rays_angles_from_single_point_camera2d(
            key_single_pt_cam='ptcam',
            key_rays='diag5_cam22_rays',
            return_indices=True,
            convex_axis=(-1, -2),
        )
        assert isinstance(dout, dict)

    def test07_add_rays_from_diagnostic(self):
        for ii, (k0, v0) in enumerate(self.coll.dobj['diagnostic'].items()):
            noptics = any([
                len(v1['optics']) == 0 for v1 in v0['doptics'].values()
            ])
            if v0['is2d'] or v0['spectro'] or noptics:
                continue
            dsamp = {'dedge': {'res': 'max'}, 'dsurface': {'nb': 3}}
            dout = self.coll.add_rays_from_diagnostic(
                key=k0,
                dsampling_pixel=dsamp,
                dsampling_optics=dsamp,
                optics=-1,
                config=self.conf,
                store=(ii % 2 == 0),
            )
            assert isinstance(dout, dict) or dout is None

    def test09_get_rays_touch_dict(self):
        dout = self.coll.get_rays_touch_dict(
            key='diag5_cam22_los',
            config=self.conf_touch,
            segment=-1,
            allowed=['PFC_ICRH0', 'Ves_FirstWallV0'],
        )
        assert isinstance(dout, dict)

    def test05_compute_vos(self):

        # add mesh
        key_mesh = 'm0'

        if key_mesh not in self.coll.dobj.get('mesh', {}).keys():
            self.coll.add_mesh_2d_rect(
                key=key_mesh,
                res=0.1,
                crop_poly=self.conf,
            )

        # compute vos
        for ii, (k0, v0) in enumerate(self.coll.dobj['diagnostic'].items()):

            doptics = self.coll.dobj['diagnostic'][k0]['doptics']
            lcam = self.coll.dobj['diagnostic'][k0]['camera']
            if len(doptics[lcam[0]]['optics']) == 0:
                continue

            keep_cross = True
            keep_hor = (ii % 2 == 0)
            keep_3d = False

            self.coll.compute_diagnostic_vos(
                # keys
                key_diag=k0,
                key_mesh=key_mesh,
                # resolution
                res_RZ=0.04,
                res_phi=0.04,
                # keep
                keep_cross=keep_cross,
                keep_hor=keep_hor,
                keep_3d=keep_3d,
                # spectro
                n0=3,
                n1=3,
                res_lamb=2e-10,
                visibility=False,
                overwrite=True,
                replace_poly=True,
                store=True,
            )

    def test10_save_to_json(self):

        for ii, (k0, v0) in enumerate(self.coll.dobj['diagnostic'].items()):

            lcam = self.coll.dobj['diagnostic'][k0]['camera']
            doptics = self.coll.dobj['diagnostic'][k0]['doptics']
            if len(doptics[lcam[0]]['optics']) == 0 or k0 == 'diag6':
                continue

            # saving
            pfe = os.path.join(_PATH_HERE, f"{k0}.json")
            self.coll.save_diagnostic_to_file(k0, pfe_save=pfe)

            # reloading
            _ = tf.data.load_diagnostic_from_file(pfe)

            # remove file
            os.remove(pfe)


#######################################################
#######################################################
#
#     Tests broadband with VOS
#
#######################################################


class Test02_Diagnostic_VOS():

    def setup_method(self):

        # cerate diags
        setup_diag(self)

        # compute vos for all


    def test01_plot_coverage(self):

        # add mesh
        key_mesh = 'm0'

        if key_mesh not in self.coll.dobj.get('mesh', {}).keys():
            self.coll.add_mesh_2d_rect(
                key=key_mesh,
                res=0.1,
                crop_poly=self.conf,
            )

        for ii, (k0, v0) in enumerate(self.coll.dobj['diagnostic'].items()):
            lcam = self.coll.dobj['diagnostic'][k0]['camera']
            doptics = self.coll.dobj['diagnostic'][k0]['doptics']
            if len(doptics[lcam[0]]['optics']) == 0:
                continue
            if self.coll.dobj['diagnostic'][k0]['spectro']:
                continue

            if doptics[lcam[0]]['dvos'].get('keym') is None:
                self.coll.compute_diagnostic_vos(
                    # keys
                    key_diag=k0,
                    key_mesh=key_mesh,
                    # resolution
                    res_RZ=0.03,
                    res_phi=0.04,
                    keep_3d=(ii % 2 == 0),
                    # spectro
                    n0=5,
                    n1=5,
                    res_lamb=1e-10,
                    visibility=False,
                    overwrite=True,
                    replace_poly=True,
                    store=True,
                )

            _ = self.coll.plot_diagnostic_geometrical_coverage(k0)

        plt.close('all')

    def test02_plot_vos(self):
        pass


#######################################################
#######################################################
#
#     Tests spectro without VOS
#
#######################################################


class Test03_Diagnostic_Spectro():

    def setup_method(self):

        self.conf, self.conf_touch = _inputs.get_config()
        self.coll = _inputs.add_diags_broadband(
            conf=self.conf,
            conf_touch=self.conf_touch,
        )

    # ----------
    # tests

    """
    def test01_etendues(self, res=np.r_[0.005, 0.003, 0.001]):
        for k0, v0 in self.coll.dobj['diagnostic'].items():
            if len(v0['optics']) == 1 or v0['spectro'] is not False:
                continue
            self.coll.compute_diagnostic_etendue_los(
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
        for k0, v0 in self.coll.dobj['aperture'].items():
            _ = self.coll.get_optics_outline(k0)

        # camera
        for k0, v0 in self.coll.dobj['camera'].items():
            _ = self.coll.get_optics_outline(k0)

        # crystals
        for k0, v0 in self.coll.dobj['crystal'].items():
            _ = self.coll.get_optics_outline(k0)

    def test03_plot(self):
        for ii, (k0, v0) in enumerate(self.coll.dobj['diagnostic'].items()):
            dax = self.coll.plot_diagnostic(
                k0,
                data='etendue',
                proj=(
                    None if ii % 3 == 0
                    else ('cross' if ii % 3 == 1 else ['cross', 'hor'])
                ),
            )
            plt.close('all')
            del dax

    def test04_sinogram(self):

        lrays = list(self.coll.dobj['rays'].keys())
        ldiag = [
            k0 for k0, v0 in self.coll.dobj['diagnostic'].items()
            if any([
                v1.get('los') is not None
                for v1 in v0['doptics'].values()
            ])
        ]
        lk = [lrays] + ldiag
        for ii, k0 in enumerate(lk):

            dout, dax = self.coll.get_sinogram(
                key=k0,
                ang='theta' if ii % 2 == 0 else 'xi',
                ang_units='deg' if ii % 3 == 0 else 'radian',
                impact_pos=ii % 3 != 0,
                R0=2.4 if ii % 3 != 1 else None,
                config=None if ii % 3 != 1 else self.conf,
                pmax=None if ii % 3 == 0 else 5,
                plot=True,
                verb=2,
            )
            plt.close('all')
            del dax

    def test07_add_rays_from_diagnostic(self):
        for ii, (k0, v0) in enumerate(self.coll.dobj['diagnostic'].items()):
            noptics = any([
                len(v1['optics']) == 0 for v1 in v0['doptics'].values()
            ])
            if v0['is2d'] or v0['spectro'] or noptics:
                continue
            dsamp = {'dedge': {'res': 'max'}, 'dsurface': {'nb': 3}}
            dout = self.coll.add_rays_from_diagnostic(
                key=k0,
                dsampling_pixel=dsamp,
                dsampling_optics=dsamp,
                optics=-1,
                config=self.conf,
                store=(ii % 2 == 0),
            )
            assert isinstance(dout, dict) or dout is None

    def test08_reverse_ray_tracing(self):
        for ii, (k0, v0) in enumerate(self.coll.dobj['diagnostic'].items()):
            lcam = self.coll.dobj['diagnostic'][k0]['camera']
            doptics = self.coll.dobj['diagnostic'][k0]['doptics']
            if len(doptics[lcam[0]]['optics']) == 0:
                continue
            if not self.coll.dobj['diagnostic'][k0]['spectro']:
                continue

            # Get points
            ptsx, ptsy, ptsz = self.coll.get_rays_pts(k0)
            kcryst = list(doptics.values())[0]['optics'][0]
            lamb = self.coll.get_crystal_bragglamb(
                key=kcryst,
                lamb=None,
                bragg=None,
                norder=None,
                rocking_curve=None,
            )[1]

            _ = self.coll.get_raytracing_from_pts(
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
                res_rock_curve=None,
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

    def test09_get_rays_touch_dict(self):
        dout = self.coll.get_rays_touch_dict(
            key='diag5_cam22_los',
            config=self.conf_touch,
            segment=-1,
            allowed=['PFC_ICRH0', 'Ves_FirstWallV0'],
        )
        assert isinstance(dout, dict)

    def test05_compute_vos(self):

        # add mesh
        key_mesh = 'm0'

        if key_mesh not in self.coll.dobj.get('mesh', {}).keys():
            self.coll.add_mesh_2d_rect(
                key=key_mesh,
                res=0.1,
                crop_poly=self.conf,
            )

        # compute vos
        for ii, (k0, v0) in enumerate(self.coll.dobj['diagnostic'].items()):

            doptics = self.coll.dobj['diagnostic'][k0]['doptics']
            lcam = self.coll.dobj['diagnostic'][k0]['camera']
            if len(doptics[lcam[0]]['optics']) == 0:
                continue

            keep_cross = True
            keep_hor = (ii % 2 == 0)
            keep_3d = False

            self.coll.compute_diagnostic_vos(
                # keys
                key_diag=k0,
                key_mesh=key_mesh,
                # resolution
                res_RZ=0.04,
                res_phi=0.04,
                # keep
                keep_cross=keep_cross,
                keep_hor=keep_hor,
                keep_3d=keep_3d,
                # spectro
                n0=3,
                n1=3,
                res_lamb=2e-10,
                visibility=False,
                overwrite=True,
                replace_poly=True,
                store=True,
            )

    def test10_save_to_json(self):

        for ii, (k0, v0) in enumerate(self.coll.dobj['diagnostic'].items()):

            lcam = self.coll.dobj['diagnostic'][k0]['camera']
            doptics = self.coll.dobj['diagnostic'][k0]['doptics']
            if len(doptics[lcam[0]]['optics']) == 0 or k0 == 'diag6':
                continue

            # saving
            pfe = os.path.join(_PATH_HERE, f"{k0}.json")
            self.coll.save_diagnostic_to_file(k0, pfe_save=pfe)

            # reloading
            _ = tf.data.load_diagnostic_from_file(pfe)

            # remove file
            os.remove(pfe)

