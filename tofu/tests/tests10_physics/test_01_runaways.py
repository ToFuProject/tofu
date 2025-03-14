

# Standard
import numpy as np


# tofu-specific
import tofu as tf


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
        dout = tf.physics_tools.get_maxwellian(
            kTe_eV=kTe[0],
            energy_eV=E,
        )
        assert isinstance(dout, dict)

        # arrays
        dout = tf.physics_tools.get_maxwellian(
            kTe_eV=kTe[None, :],
            energy_eV=E[:, None],
        )
        assert isinstance(dout, dict)

        # wavelength
        dout = tf.physics_tools.get_maxwellian(
            kTe_eV=kTe[None, :],
            lambda_m=np.linspace(1, 5, 100)*1e-10,
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
        beta = tf.physics_tools.runaways.convert_momentum_velocity_energy(
            gamma=[1, 2, 3],
        )['beta']['data']
        assert np.all((beta >= 0.) & (beta <= 1.))

        # from momentum normalized
        gamma = tf.physics_tools.runaways.convert_momentum_velocity_energy(
            momentum_normalized=10,
        )['gamma']['data']
        assert np.all(gamma >= 1.)

        # from kinetic energy
        dout = tf.physics_tools.runaways.convert_momentum_velocity_energy(
            energy_kinetic_eV=(1e3, 10e3),
        )
        assert isinstance(dout, dict)

        # from velocity
        _ = tf.physics_tools.runaways.convert_momentum_velocity_energy(
            velocity_ms=1e6,
        )

    def test02_electric_fields(self):
        _ = tf.physics_tools.runaways.get_critical_dreicer_electric_fields(
            ne_m3=np.r_[1e19, 1e20][None, :],
            kTe_eV=np.r_[1, 2, 3][:, None]*1e3,
            lnG=20,
        )

    def test03_growth_source_terms(self):
        _ = tf.physics_tools.runaways.get_growth_source_terms(
            ne_m3=np.r_[1e19, 1e20][None, :],
            lnG=15,
            Epar_Vm=1,
            kTe_eV=np.r_[1, 2, 3][:, None]*1e3,
            Zeff=2,
        )

    def test04_normalized_momentum_distribution(self):

        # case with both avalanche and dreicer
        pp = np.linspace(0.1, 10, 100)
        ne_m3 = np.r_[0.1, 1, 10, 100]*1e19
        Epar = 1
        Emax = 10e6

        # compute
        _ = tf.physics_tools.runaways.get_normalized_momentum_distribution(
            momentum_normalized=pp[:, None],
            ne_m3=ne_m3[None, :],
            Zeff=2.,
            electric_field_par_Vm=Epar,
            energy_kinetic_max_eV=Emax,
            lnG=None,
            sigmap=1.,
        )
