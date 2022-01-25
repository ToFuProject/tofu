

# Built-in
import os
import itertools as itt

# Common
import numpy as np
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec
from matplotlib.axes._axes import Axes
from mpl_toolkits.mplot3d import Axes3D

# tofu
from tofu.version import __version__

# ##########################################################
# ##########################################################
#            compute rocking curve
# ##########################################################
# ##########################################################

def compute_rockingcurve(
    h=None,
    k=None,
    l=None,
    lamb=None,
):
    """ The code evaluates, for a given wavelength, the atomic plane distance d,
    the Bragg angle, the complex structure factor, the integrated reflectivity
    with the perfect and mosaic crystal models, the reflectivity curve with the
    full dynamical model for parallel and perpendicular photon polarizations.

    The alpha-Quartz, symmetry group D(4,3), is assumed left-handed.
    It makes a difference for certain planes, for example between
    (h,k,l)=(2,0,3) and (2,0,-3).
    The alpha-Quartz structure is hexagonal with 3 molecules of SiO2 in the
    unit cell.

    All crystal lattice constants and wavelengths are in Angstroms (1e-10 m).

    Parameters:
    -----------
    h, k, l:    int
        Miller indices of crystal used
    lamb:   float
        Wavelength of interest, in Angstroms (1e-10 m)
    """

    # Check inputs
    # ------------

    lc = [h is not None, k is not None, l is not None, lamb is not None]
    if any(lc) and not all(lc):
        msg = (
            "Args h, k, l and lamb must be provided together:\n"
            + "\t - h: first Miller index ({})\n".format(h)
            + "\t - k: second Miller index ({})\n".format(k)
            + "\t - l: third Miller index ({})\n".format(l)
            + "\t - lamb: wavelength of interest ({})\n".format(lamb)
        )
        raise Exception(msg)

    # Calculations of main crystal parameters
    # ---------------------------------------

    # Classical electronical radius
    re = 2.82084508e-5

    # From Ralph W.G. Wyckoff, "Crystal Structures" (1948)
    ## https://babel.hathitrust.org/cgi/pt?id=mdp.39015081138136&view=1up&seq=259&skin=2021
    ## Inter-atomic distances into hexagonal cell unit page 254 and
    ## calculation of the associated volume
    a0 = 4.9130
    c0 = 5.4045
    V = a**2.*c*np.sqrt(3.)/2.

    ## Position of Si atoms in unit cell page 242
    u = 0.4705
    xsi = np.r_[-u, u, 0.]
    ysi = np.r_[-u, 0., u]
    zsi = np.r_[1./3., 0., 2./3.]

    ## Position of O atoms in unit cell page 242
    x = 0.4152
    y = 0.2678
    z = 0.1184
    xo = np.r_[x, y-x, -y, x-y, y, -x]
    yo = np.r_[y, -x, x-y, -y, x, y-x]
    zo = np.r_[z, z+1./3., z+2./3., -z, 2./3.-z, 1./3.-z]

    # Bragg angle and atomic plane distance d for a given wavelength lamb and
    # Miller index (h,k,l)
    d_num = np.sqrt(3.)*a0*c0
    d_den = np.sqrt(4.*(h**2 + k**2 + h*k)*c**2 + 3.*l**2*a**2)
    if d_den == 0.:
        msg = (
            "Something went wrong in the calculation of d, equal to 0!\n"
            "Please verify the values for the following Miller indices:\n"
            + "\t - h: first Miller index ({})\n".format(h)
            + "\t - k: second Miller index ({})\n".format(k)
            + "\t - l: third Miller index ({})\n".format(l)
        )
        raise Exception(msg)
    else:
        d_atom = d_num/d_den
    if d_atom < lamb/2.:
        msg = (
            "According to Bragg law, Bragg scattering need d > lamb/2!\n"
            "Please check your wavelenght arg.\n"
        )
        raise Exception(msg)
    else:
        sol = 1./(2.*d_atom)
        sin_theta = lamb/(2.*d_atom)
        theta = np.arcsin(sin_theta)
        theta_deg = theta*180./np.pi
    lc = [theta_deg < 10., theta_deg > 89.]
    if any(lc):
        msg = (
            "The computed value of theta is behind the arbitrary limits.\n"
            "Limit condition: 10° < theta < 89° and\n"
            "theta = ({})°\n".format(theta_deg)
        )
        raise Exception(msg)

    # Atomic scattering factors ["asf"] for Si(2+) and O(1-) as a function of
    # sol = sin(theta)/lambda ["sol_values"], taking into account molecular
    # bounds
    #From Henry & Lonsdale, "International tables for Crystallography" (1969)

    sol_values_si = np.r_[
        0., 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6,
        0.7, 0.8, 0.9, 1., 1.1, 1.2, 1.3, 1.4, 1.5,
    ]
    asf_si = np.r_[
        12., 11., 9.5, 8.8, 8.3, 7.7, 7.27, 6.25, 5.3,
        4.45, 3.75, 3.15, 2.7, 2.35, 2.07, 1.87, 1.71, 1.6,
    ]
    sol_values_o = np.r_[
        0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1,
    ]
    asf_o = np.r_[
        9., 7.836, 5.756, 4.068, 2.968, 2.313, 1.934, 1.710, 1.566, 1.462,
        1.373, 1.294,
    ]

    # atomic absorption coefficient for Si and O as a function of the wavelength
    # focus on the photoelectric effect, proportionnal to (lamb*Z)**3 with
    # Z the atomic number
    mu_si = 1.38e-2*lamb**2.79*14.**2.73
    mu_si1 = 5.33e-4*lamb**2.74*14.**3.03
    if lamb > 6.74:
        mu_si = mu_si1
    mu_o = 5.4e-3*lamb**2.92*8.**3.07
    mu = 2.65e-8*(7.*mu_si + 8.*mu_o)/15.

    # Calculation of the structure factor
    # -----------------------------------

