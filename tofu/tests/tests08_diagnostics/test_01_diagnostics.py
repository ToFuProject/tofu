# -*- coding: utf-8 -*-
"""
This module contains tests for tofu.geom in its structured version
"""


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

    def test03_plot(self, key_diag=None, close=None):
        _inputs._plot(
            coll=self.coll,
            key_diag=key_diag,
            conf=self.conf,
            close=close,
        )

    def test04_sinogram(self, key_diag=None, close=None):
        _inputs._sinogram(
            coll=self.coll,
            conf=self.conf,
            key_diag=key_diag,
            close=close,
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
            key=self._def_krays,
            config=self.conf_touch,
            segment=-1,
            allowed=['PFC_ICRH0', 'Ves_FirstWallV0'],
        )
        assert isinstance(dout, dict)

    def test08_reverse_ray_tracing(self):
        _inputs._reverse_ray_tracing(
            coll=self.coll,
        )

    def test09_compute_vos(
        self,
        key_diag=None,
        # options
        res_RZ=None,
        res_phi=None,
        n0=None,
        n1=None,
        res_lamb=None,
    ):
        _inputs._compute_vos(
            coll=self.coll,
            key_diag=None,
            conf=self.conf,
            spectro=self._spectro,
            # options
            res_RZ=res_RZ,
            res_phi=res_phi,
            n0=n0,
            n1=n1,
            res_lamb=res_lamb,
        )

    def test10_compute_synthetic_signal(
        self,
        key_diag=None,
        res=None,
    ):
        _inputs._synthetic_signal(
            coll=self.coll,
            key_diag=key_diag,
            spectro=self._spectro,
            res=res,
            method='los',
            conf=self.conf,
        )

    def test11_save_to_json(self):
        _inputs._save_to_json(
            self.coll,
            remove=True,
        )

    def test12_save_to_npz(self):
        _inputs._save_to_npz(
            self.coll,
            remove=True,
        )


# ############
# Broadband
# ############


class Test01_Diagnostic_Broadband_Los(Diag_Los):

    def setup_method(self):
        self.coll = _inputs.add_diags_broadband(
            conf=self.conf,
            conf_touch=self.conf_touch,
            compute=True,
        )
        self._def_kdiag = 'diag5'
        self._def_krays = 'diag5_cam22_los'
        self._spectro = False


# ############
# Spectro
# ############


class Test02_Diagnostic_Spectro_Los(Diag_Los):

    def setup_method(self, var=None, key_diag=None):

        self.coll = _inputs.add_diags_spectro(
            conf=self.conf,
            compute=True,
            key_diag=key_diag,
        )
        self._def_kdiag = 'sd0_cryst0_cam0_los'
        self._def_krays = 'sd0_cryst0_cam0_los'
        self._spectro = True


#######################################################
#######################################################
#
#     Tests with VOS
#
#######################################################


class Diag_Vos():

    def setup_class(self):
        self.conf, self.conf_touch = _inputs.get_config()

    def test00_setup_done(self):
        pass

    def test01_plot_coverage(self, key_diag=None, close=None):
        _inputs._plot_coverage(
            coll=self.coll,
            key_diag=key_diag,
            conf=self.conf,
            spectro=self._spectro,
            close=close,
        )

    def test02_plot_coverage_slice(
        self,
        key_diag=None,
        res=None,
        close=None,
        isZ=None,
    ):
        _inputs._plot_coverage_slice(
            coll=self.coll,
            key_diag=key_diag,
            conf=self.conf,
            spectro=self._spectro,
            close=close,
            res=res,
            isZ=isZ,
        )

    def test03_compute_synthetic_signal(
        self,
        key_diag=None,
        res=None,
        method='vos',
    ):
        _inputs._synthetic_signal(
            coll=self.coll,
            key_diag=key_diag,
            spectro=self._spectro,
            res=res,
            method=method,
            conf=self.conf,
        )


# ############
# Broadband
# ############


class Test03_Diagnostic_Broadband_Vos(Diag_Vos):

    def setup_method(
        self,
        var=None,
        key_diag=None,
        # options
        res_RZ=None,
        res_phi=None,
        n0=None,
        n1=None,
        res_lamb=None,
    ):
        self.coll = _inputs.add_diags_broadband(
            conf=self.conf,
            conf_touch=self.conf_touch,
            compute=True,
        )
        self._def_kdiag = 'diag5'
        self._spectro = False

        # compute vos
        _inputs._compute_vos(
            coll=self.coll,
            key_diag=key_diag,
            conf=self.conf,
            spectro=self._spectro,
            # options
            res_RZ=res_RZ,
            res_phi=res_phi,
            n0=n0,
            n1=n1,
            res_lamb=res_lamb,
        )

        # add emissivity
        _inputs._add_emiss(
            coll=self.coll,
            spectro=False,
        )


# ############
# Spectro
# ############


class Test04_Diagnostic_Spectro_Vos(Diag_Vos):

    def setup_method(
        self,
        var=None,
        key_diag=None,
        # options
        res_RZ=None,
        res_phi=None,
        n0=None,
        n1=None,
        res_lamb=None,
        lamb=None,
    ):
        self.coll = _inputs.add_diags_spectro(
            conf=self.conf,
            compute=True,
            key_diag=key_diag,
        )
        self._def_kdiag = 'sd5'
        self._spectro = True

        # compute vos
        _inputs._compute_vos(
            coll=self.coll,
            key_diag=key_diag,
            conf=self.conf,
            spectro=self._spectro,
            # options
            res_RZ=res_RZ,
            res_phi=res_phi,
            n0=n0,
            n1=n1,
            res_lamb=res_lamb,
            lamb=lamb,
        )

        # add emissivity
        _inputs._add_emiss(
            coll=self.coll,
            spectro=True,
        )

    def test02_plot_coverage_slice(
        self,
        key_diag=None,
        res=None,
        close=None,
        isZ=None,
    ):
        return
