

# Standard
import numpy as np
import matplotlib.pyplot as plt


# tofu-specific
import tofu.physics_tools as tfpt


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
#     checking Distributions
#
#######################################################


class Test01_Distributions():

    @classmethod
    def setup_class(cls):
        pass

    def setup_method(self):
        pass

    def teardown_method(self):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def test01_maxwellian(self):

        E = np.linspace(0.1, 20, 100) * 1e3
        kTe = np.r_[0.1, 1, 10, 100] * 1e3

        # single
        dout = tfpt.electrons.distribution.get_distribution(
            Te_eV=kTe[0],
            E_eV=E,
            dist='maxwell',
        )
        assert isinstance(dout, dict)

        # arrays
        dout = tfpt.electrons.distribution.get_distribution(
            Te_eV=kTe[None, :],
            E_eV=E,
            dist='maxwell',
        )
        assert isinstance(dout, dict)


#######################################################
#
#     checking Runaways
#
#######################################################


class Test02_Runaways():

    @classmethod
    def setup_class(cls):
        pass

    def setup_method(self):
        pass

    def teardown_method(self):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def test01_convert(self):

        # from gamma
        beta = tfpt.electrons.convert_momentum_velocity_energy(
            gamma=[1, 2, 3],
        )['beta']['data']
        assert np.all((beta >= 0.) & (beta <= 1.))

        # from momentum normalized
        gamma = tfpt.electrons.convert_momentum_velocity_energy(
            momentum_normalized=10,
        )['gamma']['data']
        assert np.all(gamma >= 1.)

        # from kinetic energy
        dout = tfpt.electrons.convert_momentum_velocity_energy(
            energy_kinetic_eV=(1e3, 10e3),
        )
        assert isinstance(dout, dict)
        assert np.all(dout['velocity_ms']['data'] > 0.)
        assert np.all(dout['velocity_ms']['data'] < 3e8)

        # from velocity
        dout = tfpt.electrons.convert_momentum_velocity_energy(
            velocity_ms=1e6,
        )
        assert isinstance(dout, dict)

    def test02_electric_fields(self):
        dout = tfpt.electrons.distribution.get_RE_critical_dreicer_electric_fields(
            ne_m3=np.r_[1e19, 1e20][None, :],
            kTe_eV=np.r_[1, 2, 3][:, None]*1e3,
            lnG=20,
        )
        assert isinstance(dout, dict)

    def test03_growth_source_terms(self):
        dout = tfpt.electrons.distribution.get_RE_growth_source_terms(
            ne_m3=np.r_[1e19, 1e20][None, :],
            lnG=15,
            Epar_Vm=1,
            kTe_eV=np.r_[1, 2, 3][:, None]*1e3,
            Zeff=2,
        )
        assert isinstance(dout, dict)

    def test04_normalized_momentum_distribution(self):

        # case with both avalanche and dreicer
        pp = np.linspace(0.1, 10, 100)
        ne_m3 = np.r_[0.1, 1, 10, 100]*1e19
        Epar = 1
        Emax = 10e6

        # compute
        dout = tfpt.electrons.distribution.get_distribution(
            p_par_norm=pp,
            p_perp_norm=pp,
            ne_m3=ne_m3[None, :],
            Zeff=2.,
            Efield_par_Vm=Epar,
            Ekin_max_eV=Emax,
            lnG=None,
            sigmap=1.,
        )
        assert isinstance(dout, dict)

    def test05_emission_thick_anisotropy(self):
        E = np.r_[1, 10, 100] * 1e3
        gamma = tfpt.electrons.convert_momentum_velocity_energy(
            energy_kinetic_eV=E,
        )['gamma']['data']
        anis = tfpt.electrons.emission.get_xray_thick_anisotropy(
            gamma=gamma[None, :],
            costheta=np.linspace(-1, 1, 100)[:, None],
        )
        assert isinstance(anis, np.ndarray) and np.all(anis > 0)

    def test06_emission_get_xray_thick_dcross_ei(self):
        E_re = np.r_[1, 10, 100] * 1e3
        E_ph = np.linspace(1, 20, 50) * 1e3
        dout = tfpt.electrons.emission.get_xray_thick_dcross_ei(
            E_re_eV=E_re[None, :],
            E_ph_eV=E_ph[:, None],
            atomic_nb=13,
            adjust=True,
            costheta=None,
            return_intermediates=True,
        )
        assert isinstance(dout, dict)

    def test07_plot_xray_thick_dcross_ei_vs_Salvat(self):
        dax = tfpt.electrons.emission.plot_xray_thick_dcross_ei_vs_Salvat()
        plt.close('all')
        assert isinstance(dax, dict)
