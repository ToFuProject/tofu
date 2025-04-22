

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


class Test00_Geometry():

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

    def test01_pinhole_camera_estimator(self):
        for col, div in [(False, False), (True, False), (True, True)]:
            dout = tfpt.geometry.camera1d_estimator(
                collimator=col,
                diverging=div,
            )
            assert isinstance(dout, dict)
