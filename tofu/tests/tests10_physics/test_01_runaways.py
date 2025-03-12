

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

        # arrays
        dout = tf.physics_tools.get_maxwellian(
            kTe_eV=kTe[None, :],
            energy_eV=E[:, None],
        )

        # wavelength
        dout = tf.physics_tools.get_maxwellian(
            kTe_eV=kTe[None, :],
            lambda_m=np.linspace(1, 5, 100)*1e-10,
        )


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

    def test01_distribution(self):

        E = np.linspace(0.1, 20, 100) * 1e3
        kTe = np.r_[0.1, 1, 10, 100] * 1e3

        # single
        dout = tf.physics_tools.get_maxwellian(
            kTe_eV=kTe[0],
            energy_eV=E,
        )

        # arrays
        dout = tf.physics_tools.get_maxwellian(
            kTe_eV=kTe[None, :],
            energy_eV=E[:, None],
        )

        # wavelength
        dout = tf.physics_tools.get_maxwellian(
            kTe_eV=kTe[None, :],
            lambda_m=np.linspace(1, 5, 100)*1e-10,
        )
