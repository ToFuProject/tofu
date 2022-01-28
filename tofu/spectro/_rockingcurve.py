

# Built-in
import os
import itertools as itt

# Common
import numpy as np
import scipy.interpolate
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
    ih=None,
    ik=None,
    il=None,
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
    ih, ik, il:    int
        Miller indices of crystal used
    lamb:   float
        Wavelength of interest, in Angstroms (1e-10 m)
    """

    # Check inputs
    # ------------

    lc = [ih is not None, ik is not None, il is not None, lamb is not None]
    if any(lc) and not all(lc):
        msg = (
            "Args h, k, l and lamb must be provided together:\n"
            + "\t - h: first Miller index ({})\n".format(ih)
            + "\t - k: second Miller index ({})\n".format(ik)
            + "\t - l: third Miller index ({})\n".format(il)
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
    V = a0**2.*c0*np.sqrt(3.)/2.

    # Atomic number of Si and O atoms
    Zsi = 14.
    Zo = 8.

    ## Position of the three Si atoms in the unit cell (page 242 Wyckoff)

    u = 0.4705
    xsi = np.r_[-u, u, 0.]
    ysi = np.r_[-u, 0., u]
    zsi = np.r_[1./3., 0., 2./3.]

    ## Position of the six O atoms in the unit cell (page 242 Wyckoff)

    x = 0.4152
    y = 0.2678
    z = 0.1184
    xo = np.r_[x, y-x, -y, x-y, y, -x]
    yo = np.r_[y, -x, x-y, -y, x, y-x]
    zo = np.r_[z, z+1./3., z+2./3., -z, 2./3.-z, 1./3.-z]

    # Bragg angle and atomic plane distance d for a given wavelength lamb and
    # Miller index (h,k,l)

    d_num = np.sqrt(3.)*a0*c0
    d_den = np.sqrt(4.*(ih**2 + ik**2 + ih*ik)*c0**2 + 3.*il**2*a0**2)
    if d_den == 0.:
        msg = (
            "Something went wrong in the calculation of d, equal to 0!\n"
            "Please verify the values for the following Miller indices:\n"
            + "\t - h: first Miller index ({})\n".format(ih)
            + "\t - k: second Miller index ({})\n".format(ik)
            + "\t - l: third Miller index ({})\n".format(il)
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
    # From Henry & Lonsdale, "International tables for Crystallography" (1969)
    # Vol.IV page 73 for O(1-)

    sol_si = np.r_[
        0., 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6,
        0.7, 0.8, 0.9, 1., 1.1, 1.2, 1.3, 1.4, 1.5,
    ]
    asf_si = np.r_[
        12., 11., 9.5, 8.8, 8.3, 7.7, 7.27, 6.25, 5.3,
        4.45, 3.75, 3.15, 2.7, 2.35, 2.07, 1.87, 1.71, 1.6,
    ]
    sol_o = np.r_[
        0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1,
    ]
    asf_o = np.r_[
        9., 7.836, 5.756, 4.068, 2.968, 2.313, 1.934, 1.710, 1.566, 1.462,
        1.373, 1.294,
    ]

    plt.plot(sol_si, asf_si, label="Si")
    plt.xlabel("sin(theta)/lamb")
    plt.ylabel("atomic scattering factor")
    plt.plot(sol_o, asf_o, label="O")
    plt.legend()

    # atomic absorption coefficient for Si and O as a function of lamb
    # focus on the photoelectric effect, mu=cte*(lamb*Z)**3 with
    # Z the atomic number
    # TBD : I still didn't find where to pick up the cte values and the powers

    mu_si = 1.38e-2*lamb**2.79*Zsi**2.73
    mu_si1 = 5.33e-4*lamb**2.74*Zsi**3.03
    if lamb > 6.74:
        mu_si = mu_si1
    mu_o = 5.4e-3*lamb**2.92*Zo**3.07
    mu = 2.65e-8*(7.*mu_si + 8.*mu_o)/15.

    # Calculation of the structure factor for the alpha-quartz crystal
    # ----------------------------------------------------------------

    # interpolation of atomic scattering factor ("f") in function of sol
    # ("si") for Silicium and ("o") for Oxygen
    # ("_re") for Real part and ("_im") for Imaginary part

    fsi_re = scipy.interpolate.interp1d(sol_si, asf_si)    #fsire
    dfsi_re = 0.1335*lamb - 0.006    #dfsire
    print("interp de sin(theta)/lamb=", sol, "is:", fsi_re(sol))
    fsi_re = fsi_re(sol) + dfsi_re    #fsire
    fsi_im = 5.936e-4*Zsi*(mu_si/lamb)    #fsiim, =cte*(Z*mu)/lamb

    fo_re = scipy.interpolate.interp1d(sol_o, asf_o)    #fore
    dfo_re = 0.1335*lamb - 0.206    #dfore
    print("interp de sin(theta)/lamb=", sol, "is:", fo_re(sol))
    fo_re = fo_re(sol) + dfo_re    #fore
    fo_im = 5.936e-4*Zo*(mu_o/lamb)    #foim TBD: find where to find the cte

    # structure factor ("F") for (hkl) reflection
    # In a unit cell contains N atoms, the resultant wave scattered by all the
    # N atoms in the direction of the (hkl) reflection is proportionnal to the
    # atomic scattering factor ("fn") for each species and the phase difference
    # between all the waves scattered by the different atoms:
    # phasen = 2pi*(h*xn + k*yn + l*zn) for each atom
    # So F = f1*exp(i*phase1) + f2*exp(i*phase2) + ... + fN*exp(i*phaseN)
    # Then, F = sum(n=1 to N) fn*exp(2i*pi*(h*xn + k*yn + l*zn))
    # with (xn,yn,zn) the coordinates of each atom inside the unit cell
    # And finally |F|²={sum(n=1 to N) fn*cos(2pi.phasen)}² +
    # {sum(n=1 to N) fn*sin(2pi.phasen)}²

    phasesi = np.full((xsi.size), np.nan)
    phaseo = np.full((xo.size), np.nan)
    for i in range(xsi.size):
        phasesi[i] = ih*xsi[i] + ik*ysi[i] + il*zsi[i]    #arsi
    for j in range(xo.size):
        phaseo[j] = ih*xo[j] + ik*yo[j] + il*zo[j]    #aro

    Fsi_re1 = np.sum(fsi_re*np.cos(2*np.pi*phasesi))    #resip: real si p...
    Fsi_re2 = np.sum(fsi_re*np.sin(2*np.pi*phasesi))    #aimsip: imag si p...
    Fsi_im1 = np.sum(fsi_im*np.cos(2*np.pi*phasesi))    #resis
    Fsi_im2 = np.sum(fsi_im*np.sin(2*np.pi*phasesi))    #aimsis
    print("Fsi_re (cos & sin), Fsi_im (cos & sin)", Fsi_re1, Fsi_re2, Fsi_im1, Fsi_im2)

    Fo_re1 = np.sum(fo_re*np.cos(2*np.pi*phaseo))    #reop: real op...
    Fo_re2 = np.sum(fo_re*np.sin(2*np.pi*phaseo))    #aimop: imag. op...
    Fo_im1 = np.sum(fo_im*np.cos(2*np.pi*phaseo))    #reos: real os...
    Fo_im2 = np.sum(fo_im*np.sin(2*np.pi*phaseo))    #aimos: imag. os...
    print("Fo_re (cos & sin), Fo_im (cos & sin)", Fo_re1, Fo_re2, Fo_im1, Fo_im2)

    F_re_cos = Fsi_re1 + Fo_re1    #fpre
    F_re_sin = Fsi_re2 + Fo_re2    #fpim
    F_im_cos = Fsi_im1 + Fo_im1    #fsre
    F_im_sin = Fsi_im2 + Fo_im2    #fsim
    print("F_re_(cos, sin), F_im_(cos, sin)", F_re_cos, F_re_sin, F_im_cos, F_im_sin)

    F_re = np.sqrt(F_re_cos**2 + F_re_sin**2)    #fpmod
    F_im = np.sqrt(F_im_cos**2 + F_im_sin**2)    #fsmod

    Fmod = np.sqrt(
        F_re**2 + F_im**2 - 2.*(F_re_cos*F_re_sin - F_im_cos*F_im_sin)
    )    #fmod
    Fbmod = np.sqrt(
        F_re**2 + F_im**2 - 2.*(F_im_cos*F_im_sin - F_re_cos*F_re_sin)
    )    #fbmod
    if Fmod == 0.:
        Fmod == 1e-30
    if Fbmod == 0.:
        Fbmod == 1e-30
    print("Fmod=", Fmod, "Fbmod=", Fbmod)

    # ratio im. part and real part of structure factor
    kk = F_im/F_re
    # TBD : find what it is
    rek = (F_re_cos*F_im_cos + F_re_sin*F_im_sin)/(F_re)
    print("ratio real/im k=", kk, "rek=", rek)

    # psi = 4pi*alpha <=> dielec. cte epsilon = 1 + psi
    # Re(psi) = psi' = -(4pi*e**2*F'_H)/(m*w**2*V) => psi_re if 1/4piEps0 = 1
    # Im(psi) = psi'' = -(4pi*e**2*F''_H)/(m*w**2*V)

    psi_re = re*lamb**2*F_re/(np.pi*V)    # psihp
    psiop = -re*lamb**2/(np.pi*V)*(6.*(Zo + dfo_re) + 3.*(Zsi + dfsi_re))
    psios = -re*lamb**2/(np.pi*V)*(6.*fo_im + 3.*fsi_im)

    # Integrated reflectivity for crystals models: perfect (Darwin model) &
    # ideally mosaic thick crystal
    # -------------------------------------------------------------

    R_per = Zo*F_re*re*lamb**2*(1. + abs(np.cos(2.*theta)))/(
        6.*np.pi*V*np.sin(2.*theta)
    )
    R_mos = F_re**2*re**2*lamb**3*(1. + (np.cos(2.*theta))**2)/(
        4.*mu*V**2*np.sin(2.*theta)
    )

    # Rocking curve and integrated reflectivity with "dynamical" model vs angle
    # and for the 2 states of photon polarization
    # -------------------------------------------------------------------------

    # be careful about this part
    # From Zachariasen "The theory of X-ray diffraction in Crystals" (1945)
    # for crystals with inversion centers. The extension of the general case
    # is the Francesca's work.

    polar = np.r_[1., abs(np.cos(2.*theta))]
    g = psios/(polar*psi_re)


    return (
        "volume=", V,
        "d=", d_atom,
        "sin(theta)/lamb=", sol,
        "theta=", theta,
        "mu_si=", mu_si,
        "mu_o=", mu_o,
        "mu=", mu,
        "factor_si_real=", fsi_re,
        "factor_si_im=", fsi_im,
        "factor_o_real=", fo_re,
        "factor_o_im=", fo_im,
        "phase Si:", phasesi,
        "phase O:", phaseo,
        "structure factor real:", F_re,
        "structure factor imag.:", F_im,
        "real part of psi =", psi_re,
        "real dispersioned part of psi =", psiop,
        "imag. part of psi =", psios,
        "Integrated reflectivity, perfect model :", R_per,
        "Integrated reflectivity, mosaic model :", R_mos,
    )
