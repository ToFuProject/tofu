

# Standard
import numpy as np


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


class Test00_Transmission():

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

    def test01_transmission(self):

        # input
        E = np.linspace(0.1, 20, 100) * 1e3
        dthick = {
            'filt1': {
                'mat': 'StainlessSteel',
                'thick': np.r_[5e-6, 10e-6],
            },
            'filt2': {
                'mat': 'Al',
                'thick': np.r_[10e-6, 100e-6],
            },
        }

        # format
        for k0, v0 in dthick.items():
            dthick[k0]['thick'] = v0['thick'][None, :]

        # single
        dout = tfpt.transmission.get_xray_transmission(
            E=E[:, None],
            dthick=dthick,
        )
        assert isinstance(dout, dict)
