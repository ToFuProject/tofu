"""Functions that are used in different tests"""
import numpy as np


def compute_ves_norm(ves_poly):
    ves_norm = ves_poly[:, 1:] - ves_poly[:, :-1]
    ves_norm = np.array([-ves_norm[1, :], ves_norm[0, :]])
    ves_norm = ves_norm / np.sqrt(np.sum(ves_norm ** 2, axis=0))[np.newaxis, :]
    return ves_norm


def compute_min_max_r_and_z(ves_poly):
    r_min_max = np.array([np.min(ves_poly[0, :]), np.max(ves_poly[0, :])])
    z_min_max = np.array([np.min(ves_poly[1, :]), np.max(ves_poly[1, :])])
    return r_min_max, z_min_max
