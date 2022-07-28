# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.


import os


import numpy as np
import matplotlib.pyplot as plt


_PATH_HERE = os.path.dirname(__file__)
_PATH_TOFU = os.path.dirname(os.path.dirname(_PATH_HERE))

import tofu as tf
import tf.tests.tests01_geom.test_05_solid_angles as tftests


# #############################################################################
# #############################################################################
#               Benchmark of Mesh2D routines
#                   High level routines
# #############################################################################


class SolidAngle:
    """ Benchmark the solid angle computation routines

    In particular:

    """

    # -----------------------------
    # Attributes reckognized by asv

    # time before benchmark is killed
    timeout = 500
    repeat = (1, 10, 20.0)
    sample_time = 0.100

    # -------------------------------------------------------
    # Setup and teardown, run before / after benchmark methods

    def setup_cache(self):
        return 0

    def setup(self, out):
        """ run before each benchmark method, out from setup_cache  """
        self.light = tftests._create_light(npts=10000)
        self.visibility = tftests._create_visibility(npts=1000)

    def teardown(self, out):
        pass

    # -------------------------------------
    # benchmarks methods for solid angle

    # 
    def time_00_light(self, out):
        sa = tfg.calc_solidangle_apertures(
            pts_x=self.light['pts_x'],
            pts_y=self.light['pts_y'],
            pts_z=self.light['pts_z'],
            apertures=self.light['ap']['ap3'],
            detectors=self.light['det'],
            visibility=False,
            return_vector=False,
        )

    def time_01_unitvectors(self, out):
        sa, uv_x, uv_y, uv_z = tfg.calc_solidangle_apertures(
            pts_x=self.light['pts_x'],
            pts_y=self.light['pts_y'],
            pts_z=self.light['pts_z'],
            apertures=self.light['ap']['ap3'],
            detectors=self.light['det'],
            visibility=False,
            return_vector=True,
        )

    def time_02_visibility(self, out):
        sa = tfg.calc_solidangle_apertures(
            pts_x=self.visibility['pts_x'],
            pts_y=self.visibility['pts_y'],
            pts_z=self.visibility['pts_z'],
            apertures=self.visibility['ap'],
            detectors=self.visibility['det'],
            config=self.visibility['config'],
            visibility=True,
            return_vector=False,
        )
