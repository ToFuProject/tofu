

# Built-in
import os
import itertools as itt

# Common
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
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

def compute_rockingcurve(self, ih=None, ik=None, il=None, lamb=None):
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

    # Classical electronical radius, in Angstroms
    re = 2.82084508e-5

    # From Ralph W.G. Wyckoff, "Crystal Structures" (1948)
    # https://babel.hathitrust.org/cgi/pt?id=mdp.39015081138136&view=1up&seq=259&skin=2021
    # Inter-atomic distances into hexagonal cell unit (page 239 & 242) and
    # calculation of the associated volume
    # TBC with Francesca
    a0 = 4.9130
    c0 = 5.4045
    V = a0**2.*c0*np.sqrt(3.)/2.

    # Atomic number of Si and O atoms
    Zsi = 14.
    Zo = 8.

    # Position of the three Si atoms in the unit cell (page 242 Wyckoff)
    u = 0.4705
    xsi = np.r_[-u, u, 0.]
    ysi = np.r_[-u, 0., u]
    zsi = np.r_[1./3., 0., 2./3.]

    # Position of the six O atoms in the unit cell (page 242 Wyckoff)
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
            "Please check your wavelength arg.\n"
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
    # Vol.III p202 or Vol.IV page 73 for O(1-), Vol.III p202 ? for Si(2+)
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

    fig = plt.figure(figsize=(8,6))
    gs = gridspec.GridSpec(1, 1)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlabel(r'sin($\theta$)/$\lambda$')
    ax.set_ylabel("atomic scattering factor")
    ax.plot(sol_si, asf_si, label="Si")
    ax.plot(sol_o, asf_o, label="O")
    ax.legend()

    # atomic absorption coefficient for Si and O as a function of lamb
    # focus on the photoelectric effect, mu=cte*(lamb*Z)**3 with
    # Z the atomic number
    # TBF : I still didn't find where to pick up the cte values and the powers
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
    fsi_re = fsi_re(sol) + dfsi_re    #fsire
    fsi_im = 5.936e-4*Zsi*(mu_si/lamb)    #fsiim, =cte*(Z*mu)/lamb

    fo_re = scipy.interpolate.interp1d(sol_o, asf_o)    #fore
    dfo_re = 0.1335*lamb - 0.206    #dfore
    fo_re = fo_re(sol) + dfo_re    #fore
    fo_im = 5.936e-4*Zo*(mu_o/lamb)    #foim TBF: find where to find the cte

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

    Fsi_re1 = np.sum(fsi_re*np.cos(2*np.pi*phasesi))    #resip
    Fsi_re2 = np.sum(fsi_re*np.sin(2*np.pi*phasesi))    #aimsip
    Fsi_im1 = np.sum(fsi_im*np.cos(2*np.pi*phasesi))    #resis
    Fsi_im2 = np.sum(fsi_im*np.sin(2*np.pi*phasesi))    #aimsis

    Fo_re1 = np.sum(fo_re*np.cos(2*np.pi*phaseo))    #reop
    Fo_re2 = np.sum(fo_re*np.sin(2*np.pi*phaseo))    #aimop
    Fo_im1 = np.sum(fo_im*np.cos(2*np.pi*phaseo))    #reos
    Fo_im2 = np.sum(fo_im*np.sin(2*np.pi*phaseo))    #aimos

    F_re_cos = Fsi_re1 + Fo_re1    #fpre
    F_re_sin = Fsi_re2 + Fo_re2    #fpim
    F_im_cos = Fsi_im1 + Fo_im1    #fsre
    F_im_sin = Fsi_im2 + Fo_im2    #fsim

    F_re = np.sqrt(F_re_cos**2 + F_re_sin**2)    #fpmod
    F_im = np.sqrt(F_im_cos**2 + F_im_sin**2)    #fsmod

    # Calculation of Fourier coefficients of polarization
    # ---------------------------------------------------

    # dielectric constant = 1 + psi = 1 + 4pi.alpha with
    # alpha the medium polarizability
    # psi : complex number = psi' + i.psi"

    # expression of the Fourier coef. psi_H
    Fmod = np.sqrt(
        F_re**2 + F_im**2 - 2.*(F_re_cos*F_re_sin - F_im_cos*F_im_sin)
    )    #fmod
    # psi_-H equivalent to (-ih, -ik, -il)
    Fbmod = np.sqrt(
        F_re**2 + F_im**2 - 2.*(F_im_cos*F_im_sin - F_re_cos*F_re_sin)
    )    #fbmod

    if Fmod == 0.:
        Fmod == 1e-30
    if Fbmod == 0.:
        Fbmod == 1e-30

    # ratio imaginary part and real part of the structure factor
    kk = F_im/F_re
    # rek = Real(kk)
    rek = (F_re_cos*F_im_cos + F_re_sin*F_im_sin)/(F_re**2.)

    # Re(psi) = psi' = -(4pi*e**2*F'_H)/(m*w**2*V) if 1/4piEps0 = 1
    # Im(psi) = psi'' = -(4pi*e**2*F''_H)/(m*w**2*V)
    # real part of psi_H
    psi_re = (re*(lamb**2)*F_re)/(np.pi*V)    # psihp
    # zero-order real part (averaged) TBF
    psi0_dre = -re*(lamb**2)*(
        6.*(Zo + dfo_re) + 3.*(Zsi + dfsi_re)
        )/(np.pi*V)   # psiop
    # zero-order imaginary part (averaged)
    psi0_im = -re*(lamb**2)*(6.*fo_im + 3.*fsi_im)/(np.pi*V)    #psios

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

    # incident wave polarization (normal & parallel components)
    polar = np.r_[1., abs(np.cos(2.*theta))]
    # variables of simplification y, dy, g, L
    g = psi0_im/(polar*psi_re)
    y = np.linspace(-10., 10., 501)    # ay
    dy = np.zeros(501) + 0.1    # day
    al = np.full((2, 501), 0.)

    power_ratio = np.full((al.shape), np.nan)    #phpo
    th = np.full((al.shape), np.nan)    #phpo
    rr = np.full((polar.shape), np.nan)
    for i in range(al[:, 0].size):
        al[i, ...] = (y**2 + g[i]**2 + np.sqrt(
            (y**2 - g[i]**2 + kk**2 - 1.)**2 + 4.*(g[i]*y - rek)**2
            ))/np.sqrt((kk**2 - 1.)**2 + 4.*rek**2)
        # reflecting power P_DIN
        power_ratio[i, ...] = (Fmod/Fbmod)*(
            al[i, :] - np.sqrt((al[i, :]**2 - 1.))
        )
        power_ratiob = power_ratio[i, ...]
        # intensity scale on the glancing angle (y=kk/g)
        th[i, ...] = (y*polar[i]*psi_re - psi0_dre)/np.sin(2.*theta)
        # integration of the power ratio over dy
        rhy = np.sum(dy*power_ratiob)
        # diffraction radiation
        # r(i=0): normal component & r(i=1): parallel component
        rr[i, ...] = (polar[i]*psi_re*rhy)/np.sin(2.*theta)
    R_dyn = np.sum(rr)/2.
    if R_dyn < 1e-7:
        msg = (
            "Please check the equations for integrated reflectivity:\n"
            "the value of R_din ({}) is less than 1e-7.\n".format(R_dyn)
        )
        raise Exception(msg)

    fmax = np.max(power_ratio[0] + power_ratio[1])
    det = (2.*R_dyn)/fmax

    # Plot power ratio
    # ----------------

    fig1 = plt.figure(figsize=(8,6))
    gs = gridspec.GridSpec(1, 1)
    ax = fig1.add_subplot(gs[0, 0])
    ax.set_title(
        'Qz, ' + f'({ih},{ik},{il})' + fr', $\lambda$={lamb} A' +
        fr', Bragg angle={np.round(theta, decimals=3)} rad'
    )
    ax.set_xlabel(r'$\theta$-$\theta_{B}$ (rad)')
    ax.set_ylabel('P$_H$/P$_0$')
    ax.plot(th[0, :], power_ratio[0, :], label='normal component')
    ax.plot(th[1, :], power_ratio[1, :], label='parallel component')
    ax.legend()

    return (
        'Wavelength (A):', lamb,
        'Miller indices:', (ih, ik, il),
        'Inter-reticular distance (A):', d_atom,
        'Volume of the unit cell (A^3)', V,
        'Bragg angle of reference (rad)', str(np.round(theta, decimals=3)),
        'Integrated reflectivity, perfect model', R_per,
        'Integrated reflectivity, mosaic model', R_mos,
        'Integrated reflectivity, thick crystal model', R_dyn,
        'Ratio imag. & real part of structure factor', kk,
        'R_perp/R_par', rr[1]/rr[0],
        'RC width', det,
    )
