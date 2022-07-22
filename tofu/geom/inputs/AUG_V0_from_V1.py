

import numpy as np


import tofu as tf


def make_v0_from_v1():

    conf = tf.load_config('AUG-V1')

    lorder = [
        ('SBi', np.r_[range(0, 23)]),
        ('TPLT5', np.r_[range(12, 22), range(0, 6)]),
        ('TPLT4', np.r_[range(14, 23), range(0, 9)]),
        ('TPLT3', np.r_[range(11, 15), range(0, 5)]),
        ('TPLT2', np.r_[range(8, 12), range(0, 4)]),
        ('TPLT1', np.r_[range(14, 17), range(0, 4)]),
        ('TPRT5', np.r_[range(14, 16), range(0, 4)]),
        ('TPRT4', np.r_[range(8, 12), range(0, 4)]),
        ('TPRT3', np.r_[range(9, 15), 0]),
        ('TPRT2', np.r_[range(23, 29), range(0, 4)]),
        ('D2dBu1', np.r_[range(11, 22), 0]),
        ('D2dBu2', np.r_[10, 0]),
        ('D2dBu3', np.r_[10, 0, 1]),
        ('ICRHa', np.r_[range(3, 61), 0]),
        ('D2dBl2', np.r_[range(10, 14), 0]),
        ('D2dBl3', np.r_[range(9, 13), 0, 1]),
        ('D2dBG2', np.r_[range(9, 28), 0, 1]),
        ('D3BG1', np.r_[range(18, 20), 0, 1]),
        ('D3BG10', np.r_[range(11, 18)]),
        ('D2cdomR', np.r_[range(7, 14), 0, 1, 2, 3]),
        ('D2cdome', np.r_[range(5, 14), 0, 1, 2]),
        ('D2cdomL', np.r_[range(5, 14), 0]),
        ('D2cTPic', np.r_[range(7, 12), 0]),
        ('D2cTPi', np.r_[range(12, 18), 0]),
        ('D2cTPib', np.r_[range(7, 23), 0, 1, 2]),
        ('D2ci2', np.r_[range(4, 10), 0, 1]),
        ('D2ci1', np.r_[range(4, 12), 0, 1]),
    ]

    poly = np.concatenate(
        tuple([
            conf.dStruct['dObj']['PFC'][ss].Poly_closed[:, ind]
            for (ss, ind) in lorder
        ]),
        axis=1,
    )

    return poly, lorder
