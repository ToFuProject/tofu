# -*- coding: utf-8 -*-
"""
This module contains tests for tofu.geom in its structured version
"""


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
#     Tests Diag broadband without VOS (except computing)
#
#######################################################


class Diag_Los():

    def setup_class(self):
        self.conf, self.conf_touch = _inputs.get_config()

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
        for k0, v0 in self.coll.dobj.get('crystal', {}).items():
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
        _inputs._sinogram(
            coll=self.coll,
            conf=self.conf,
        )

    def test05_add_rays_from_diagnostic(self):
        _inputs._add_rays_from_diagnostic(
            coll=self.coll,
            conf=self.conf,
        )

    def test06_add_single_point_camera2d(self):
        _inputs._add_single_point_camera2d(
            coll=self.coll,
            kdiag=self._def_kdiag,
            conf_touch=self.conf_touch,
        )

    def test07_get_rays_touch_dict(self):
        dout = self.coll.get_rays_touch_dict(
            key=self._def_kdiag,
            config=self.conf_touch,
            segment=-1,
            allowed=['PFC_ICRH0', 'Ves_FirstWallV0'],
        )
        assert isinstance(dout, dict)

    def test08_reverse_ray_tracing(self):
        _inputs._reverse_ray_tracing(
            coll=self.coll,
        )

    def test09_compute_vos(self):
        _inputs._compute_vos(
            coll=self.coll,
            key_diag=None,
            conf=self.conf,
            spectro=self._spectro,
        )

    def test10_save_to_json(self):
        _inputs._save_to_json(
            self.coll,
            remove=True,
        )

    def test11_save_to_npz(self):
        _inputs._save_to_npz(
            self.coll,
            remove=True,
        )


class Test01_Diagnostic_Broadband_Los(Diag_Los):

    def setup_method(self):
        self.coll = _inputs.add_diags_broadband(
            conf=self.conf,
            conf_touch=self.conf_touch,
            compute=True,
        )
        self._def_kdiag = 'diag5'
        self._spectro = False


class Test02_Diagnostic_Spectro_Los(Diag_Los):

    def setup_method(self):
        self.coll = _inputs.add_diags_spectro(
            conf=self.conf,
            compute=True,
        )
        self._def_kdiag = 'd00_cryst0_cam0_los'
        self._spectro = True


#######################################################
#######################################################
#
#     Tests with VOS
#
#######################################################


class Diag_Vos():

    def setup_method(self):

        # cerate diags
        # compute vos for all
        pass

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
