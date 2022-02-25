

# Built-in
import sys
import os
import warnings
import copy

# Common
import numpy as np
import scipy.interpolate
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.axes._axes import Axes

# tofu
from tofu.version import __version__


# ##########################################################
# ##########################################################
#                  compute rocking curve
# ##########################################################
# ##########################################################


def compute_rockingcurve(
    ih=None, ik=None, il=None, lamb=None,
    use_non_parallelism=None, na=None,
    plot_asf=None, plot_power_ratio=None, plot_relation=None,
    verb=None, returnas=None,
):
    """The code evaluates, for a given wavelength and Miller indices set,
    the atomic plane distance d, the Bragg angle and the complex structure
    factor for alpha_Quartz crystals.
    Then, the power ratio and their integrated reflectivity, for 3 differents
    models (perfect, mosaic and dynamical) are computed to obtain
    rocking curves, for parallel and perpendicular photon polarizations.

    The possibility to add a non-parallelism between crystal's optical surface
    and inter-atomic planes is now available. Rocking curve plots are updated
    in order to show 3 cases: non-parallelism equal to zero and both limit
    cases close to +/- the reference Bragg angle.
    The relation between the value of this non-parallelism and 3 physical
    quantities can also be plotted: the integrated reflectivity for the normal
    component of polarization, the rocking curve widths and the symmetry
    parameter b.

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
    lamb:    float
        Wavelength of interest, in Angstroms (1e-10 m)
    use_non_parallelism:    str
        Introduce non-parallelism between dioptre and reflecting planes
    na:    int
        Number of non-parallelism angles steps, odd number preferred
    plot_asf:    str
        Plotting the atomic scattering factor thanks to data with respect to
        sin(theta)/lambda
    plot_power_ratio:    str
        Plot the power ratio with respect to the glancing angle
    plot_relation:    str
        Plot relations between the integrated reflectivity, the intrinsic
        width and the parameter b vs the glancing angle
    verb:    str
        True or False to print the content of the results dictionnary 'dout'
    returnas:    str
        Entry 'dict' to allow optionnal returning of 'dout' dictionnary
    """

    # Check inputs
    # ------------

    if use_non_parallelism is None:
        use_non_parallelism = False
    if na is None:
        na = 51
    if plot_asf is None:
        plot_asf = False
    if plot_power_ratio is None:
        plot_power_ratio = True
    if plot_relation is None and use_non_parallelism is not False:
        plot_relation = True
    if verb is None:
        verb = True
    if returnas is None:
        returnas = None

    ih, ik, il, lamb = CrystBragg_check_inputs_rockingcurve(
        ih=ih, ik=ik, il=il, lamb=lamb,
    )

    # Calculations of main crystal parameters
    # ---------------------------------------

    # Classical electronical radius, in Angstroms
    re = 2.8208e-5

    # Inter-atomic distances into hexagonal cell unit and associated volume
    # TBC with F.
    a0 = 4.9134
    c0 = 5.4051
    V = (a0**2.)*c0*np.sqrt(3.)/2.

    # Atomic number of Si and O atoms
    Zsi = 14.
    Zo = 8.

    # Position of the three Si atoms in the unit cell
    u = 0.4705
    xsi = np.r_[-u, u, 0.]
    ysi = np.r_[-u, 0., u]
    zsi = np.r_[1./3., 0., 2./3.]
    Nsi = np.size(xsi)

    # Position of the six O atoms in the unit cell
    x = 0.4152
    y = 0.2678
    z = 0.1184
    xo = np.r_[x, y - x, -y, x - y, y, -x]
    yo = np.r_[y, -x, x - y, -y, x, y - x]
    zo = np.r_[z, z + 1./3., z + 2./3., -z, 2./3. - z, 1./3. - z]
    No = np.size(xo)

    # Bragg angle and atomic plane distance d for a given wavelength lamb and
    # Miller indices (h,k,l)
    d_num = np.sqrt(3.)*a0*c0
    d_den = np.sqrt(4.*(ih**2 + ik**2 + ih*ik)*(c0**2) + 3.*(il**2)*(a0**2))
    if d_den == 0.:
        msg = (
            "Something went wrong in the calculation of d, equal to 0!\n"
            "Please verify the values for the following Miller indices:\n"
            + "\t - h: first Miller index ({})\n".format(ih)
            + "\t - k: second Miller index ({})\n".format(ik)
            + "\t - l: third Miller index ({})\n".format(il)
        )
        raise Exception(msg)
    d_atom = d_num/d_den
    if d_atom < lamb/2.:
        msg = (
            "According to Bragg law, Bragg scattering need d > lamb/2!\n"
            "Please check your wavelength argument.\n"
        )
        raise Exception(msg)
    sol = 1./(2.*d_atom)
    sin_theta = lamb/(2.*d_atom)
    theta = np.arcsin(sin_theta)
    theta_deg = theta*(180./np.pi)
    lc = [theta_deg < 10., theta_deg > 89.]
    if any(lc):
        msg = (
            "The computed value of theta is behind the arbitrary limits.\n"
            "Limit condition: 10° < theta < 89° and\n"
            "theta = ({})°\n".format(theta_deg)
        )
        raise Exception(msg)

    # Atomic scattering factors ("asf") for Si(2+) and O(1-) as a function of
    # sin(theta)/lambda ("sol"), taking into account molecular bounds
    # ("si") for Silicium and ("o") for Oxygen
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


    # atomic absorption coefficient for Si and O as a function of lamb
    mu_si = 1.38e-2*(lamb**2.79)*(Zsi**2.73)
    mu_si1 = 5.33e-4*(lamb**2.74)*(Zsi**3.03)
    if lamb > 6.74:
        mu_si = mu_si1
    mu_o = 5.4e-3*(lamb**2.92)*(Zo**3.07)
    mu = 2.65e-8*(7.*mu_si + 8.*mu_o)/15.

    # Calculation of the structure factor for the alpha-quartz crystal
    # ----------------------------------------------------------------

    # interpolation of atomic scattering factor ("f") in function of sol
    # ("_re") for Real part and ("_im") for Imaginary part
    fsi_re = scipy.interpolate.interp1d(sol_si, asf_si)
    dfsi_re = 0.1335*lamb - 0.006
    fsi_re = fsi_re(sol) + dfsi_re
    fsi_im = 5.936e-4*Zsi*(mu_si/lamb)

    fo_re = scipy.interpolate.interp1d(sol_o, asf_o)
    dfo_re = 0.1335*lamb - 0.206
    fo_re = fo_re(sol) + dfo_re
    fo_im = 5.936e-4*Zo*(mu_o/lamb)

    # structure factor ("F") for (hkl) reflection
    phasesi = np.full((xsi.size), np.nan)
    phaseo = np.full((xo.size), np.nan)
    for i in range(xsi.size):
        phasesi[i] = ih*xsi[i] + ik*ysi[i] + il*zsi[i]
    for j in range(xo.size):
        phaseo[j] = ih*xo[j] + ik*yo[j] + il*zo[j]

    Fsi_re1 = np.sum(fsi_re*np.cos(2*np.pi*phasesi))
    Fsi_re2 = np.sum(fsi_re*np.sin(2*np.pi*phasesi))
    Fsi_im1 = np.sum(fsi_im*np.cos(2*np.pi*phasesi))
    Fsi_im2 = np.sum(fsi_im*np.sin(2*np.pi*phasesi))

    Fo_re1 = np.sum(fo_re*np.cos(2*np.pi*phaseo))
    Fo_re2 = np.sum(fo_re*np.sin(2*np.pi*phaseo))
    Fo_im1 = np.sum(fo_im*np.cos(2*np.pi*phaseo))
    Fo_im2 = np.sum(fo_im*np.sin(2*np.pi*phaseo))

    F_re_cos = Fsi_re1 + Fo_re1
    F_re_sin = Fsi_re2 + Fo_re2
    F_im_cos = Fsi_im1 + Fo_im1
    F_im_sin = Fsi_im2 + Fo_im2

    F_re = np.sqrt(F_re_cos**2 + F_re_sin**2)
    F_im = np.sqrt(F_im_cos**2 + F_im_sin**2)

    # Calculation of Fourier coefficients of polarization
    # ---------------------------------------------------

    # expression of the Fourier coef. psi_H
    Fmod = np.sqrt(
        F_re**2 + F_im**2 - 2.*(F_re_cos*F_re_sin - F_im_cos*F_im_sin)
    )

    # psi_-H equivalent to (-ih, -ik, -il)
    Fbmod = np.sqrt(
        F_re**2 + F_im**2 - 2.*(F_im_cos*F_im_sin - F_re_cos*F_re_sin)
    )

    if Fmod == 0.:
        Fmod == 1e-30
    if Fbmod == 0.:
        Fbmod == 1e-30

    # ratio imaginary part and real part of the structure factor
    kk = F_im/F_re

    # rek : Real(kk)
    rek = (F_re_cos*F_im_cos + F_re_sin*F_im_sin)/(F_re**2.)

    # real part of psi_H
    psi_re = (re*(lamb**2)*F_re)/(np.pi*V)

    # zero-order real part (averaged)
    # TBC with F.
    psi0_dre = -re*(lamb**2)*(
        No*(Zo + dfo_re) + Nsi*(Zsi + dfsi_re)
    )/(np.pi*V)
    """psi0_dre = -re*(lamb**2)*(
        No*Zo*(1. + dfo_re) + Nsi*Zsi*(1. + dfsi_re)
    )/(np.pi*V)"""

    # zero-order imaginary part (averaged)
    psi0_im = -re*(lamb**2)*(No*fo_im + Nsi*fsi_im)/(np.pi*V)
    """psi0_im = -re*(lamb**2)*(No*Zo*fo_im + Nsi*Zsi*fsi_im)/(np.pi*V)"""

    # Power ratio and their integrated reflectivity for 3 crystals models:
    # perfect (Darwin model), ideally mosaic thick and dynamical
    # --------------------------------------------------------------------

    if use_non_parallelism:
        (
            alpha, bb, polar, g, y, y0, power_ratio, th,
            rhg, rhg_perp_norm, rhg_para_norm,
            P_per, P_mos, P_dyn, det, det_norm,
        ) = CrystBragg_comp_integrated_reflect(
            lamb=lamb, re=re, V=V, Zo=Zo, theta=theta, mu=mu,
            F_re=F_re, psi_re=psi_re, psi0_dre=psi0_dre, psi0_im=psi0_im,
            Fmod=Fmod, Fbmod=Fbmod, kk=kk, rek=rek,
            model=['perfect', 'mosaic', 'dynamical'],
            use_non_parallelism=use_non_parallelism, na=na,
        )
    else:
        (
            alpha, bb, polar, g, y, y0, power_ratio, th,
            rhg, P_per, P_mos, P_dyn, det,
        ) = CrystBragg_comp_integrated_reflect(
            lamb=lamb, re=re, V=V, Zo=Zo, theta=theta, mu=mu,
            F_re=F_re, psi_re=psi_re, psi0_dre=psi0_dre, psi0_im=psi0_im,
            Fmod=Fmod, Fbmod=Fbmod, kk=kk, rek=rek,
            model=['perfect', 'mosaic', 'dynamical'],
            use_non_parallelism=use_non_parallelism, na=na,
        )

    # Plot atomic scattering factor
    # -----------------------------

    if plot_asf:
        CrystalBragg_plot_atomic_scattering_factor(
            sol_si=sol_si, sol_o=sol_o, asf_si=asf_si, asf_o=asf_o,
        )

    # Plot power ratio
    # ----------------

    if plot_power_ratio:
        CrystalBragg_plot_power_ratio(
            ih=ih, ik=ik, il=il, lamb=lamb,
            theta=theta, theta_deg=theta_deg,
            th=th, power_ratio=power_ratio, y=y, y0=y0,
            bb=bb, polar=polar, alpha=alpha,
            use_non_parallelism=use_non_parallelism, na=na,
        )

    # Plot P_dyn vs glancing angle
    # ----------------------------

    if plot_relation:
        CrystalBragg_plot_reflect_glancing(
            ih=ih, ik=ik, il=il, lamb=lamb,
            theta=theta, theta_deg=theta_deg,
            alpha=alpha, bb=bb, th=th, rhg=rhg,
            rhg_perp_norm=rhg_perp_norm,
            rhg_para_norm=rhg_para_norm,
            det=det_norm,
        )

    # Print results
    # -------------

    dout = {
        'Wavelength (A)': lamb,
        'Miller indices': (ih, ik, il),
        'Inter-reticular distance (A)': d_atom,
        'Volume of the unit cell (A^3)': V,
        'Bragg angle of reference (rad, deg)': (theta, theta_deg),
        'Ratio imag & real part of structure factor': kk,
        'Integrated reflectivity': {
            'perfect model': P_per,
            'mosaic model': P_mos,
            'dynamical model': P_dyn,
        },
        'P_{dyn,para}/P_{dyn,norm} (integrated values)': rhg[1]/rhg[0],
        'RC width': det,
    }
    if use_non_parallelism:
        dout['Non-parallelism angles (deg)'] = alpha*(180/np.pi)

    if verb is True:
        dout['Inter-reticular distance (A)'] = np.round(d_atom, decimals=3)
        dout['Volume of the unit cell (A^3)'] = np.round(V, decimals=3)
        dout['Bragg angle of reference (rad, deg)'] = (
            np.round(theta, decimals=3), np.round(theta_deg, decimals=3),
        )
        dout['Ratio imag & real part of structure factor'] = (
            np.round(kk, decimals=3,)
        )
        dout['P_{dyn,para}/P_{dyn,norm} (integrated values)'] = np.round(
            rhg[1]/rhg[0], decimals=9,
        )
        dout['RC width'] = np.round(det, decimals=6)
        dout['Integrated reflectivity']['perfect model'] = (
            np.round(P_per, decimals=9),
        )
        dout['Integrated reflectivity']['mosaic model'] = (
            np.round(P_mos, decimals=9),
        )
        dout['Integrated reflectivity']['dynamical model'] = (
            np.round(P_dyn, decimals=9),
        )
        lstr = [f'\t -{k0}: {V0}' for k0, V0 in dout.items()]
        msg = (
            " The following data was calculated:\n"
            + "\n".join(lstr)
        )
        print(msg)

    if returnas is dict:
        return dout


def CrystBragg_check_inputs_rockingcurve(
    ih=None, ik=None, il=None, lamb=None,
):

    # All args are None
    dd = {'ih': ih, 'ik': ik, 'il': il, 'lamb': lamb}
    lc = [v0 is None for k0, v0 in dd.items()]
    if all(lc):
        ih = 1
        ik = 1
        il = 0
        lamb = 3.96
        msg = (
            "Args h, k, l and lamb were not explicitely specified\n"
            "and have been put to the following default values:\n"
            + "\t - h: first Miller index ({})\n".format(ih)
            + "\t - k: second Miller index ({})\n".format(ik)
            + "\t - l: third Miller index ({})\n".format(il)
            + "\t - lamb: wavelength of interest ({})\n".format(lamb)
        )
        warnings.warn(msg)

    # Some args are bot but not all
    dd2 = {'ih': ih, 'ik': ik, 'il': il, 'lamb': lamb}
    lc2 = [v0 is None for k0, v0 in dd2.items()]
    if any(lc2):
        msg = (
            "Args h, k, l and lamb must be provided together:\n"
            + "\t - h: first Miller index ({})\n".format(ih)
            + "\t - k: second Miller index ({})\n".format(ik)
            + "\t - l: third Miller index ({})\n".format(il)
            + "\t - lamb: wavelength of interest ({})\n".format(lamb)
        )
        raise Exception(msg)

    # Some args are string values
    cdt = [type(v0) == str for k0, v0 in dd.items()]
    if any(cdt) or all(cdt):
        msg = (
            "Args h, k, l and lamb must not be string inputs:\n"
            "and have been put to default values:\n"
            + "\t - h: first Miller index ({})\n".format(ih)
            + "\t - k: second Miller index ({})\n".format(ik)
            + "\t - l: third Miller index ({})\n".format(il)
            + "\t - lamb: wavelength of interest ({})\n".format(lamb)
        )
        raise Exception(msg)

    return ih, ik, il, lamb,


def CrystBragg_comp_integrated_reflect(
    lamb=None, re=None, V=None, Zo=None, theta=None, mu=None,
    F_re=None, psi_re=None, psi0_dre=None, psi0_im=None,
    Fmod=None, Fbmod=None, kk=None, rek=None,
    model=[None, None, None],
    use_non_parallelism=None, na=None,
):

    # Symmetry parameter b
    # ---------------------

    if not use_non_parallelism:
        alpha = np.r_[0.]
        bb = np.r_[-1.]
    else:
        alpha = np.linspace(-theta + 0.01, theta - 0.01, na)
        # alpha = np.linspace(-0.05, 0.05, 5)*(np.pi/180)
        bb = np.sin(alpha + theta)/np.sin(alpha - theta)

    # Perfect (darwin) model
    # ----------------------

    P_per = Zo*F_re*re*(lamb**2)*(1. + abs(np.cos(2.*theta)))/(
        6.*np.pi*V*np.sin(2.*theta)
    )

    # Ideally thick mosaic model
    # --------------------------

    P_mos = (F_re**2)*(re**2)*(lamb**3)*(1. + (np.cos(2.*theta))**2)/(
        4.*mu*(V**2)*np.sin(2.*theta)
    )

    # Dynamical model
    # ---------------

    # incident wave polarization (normal & parallel components)
    polar = np.r_[1., abs(np.cos(2.*theta))]

    # variables of simplification y, dy, g, L
    g = np.full((polar.size, bb.size), np.nan)
    for i in range(polar.size):
        for j in range(bb.size):
            g[i, j] = (((1. - bb[j])/2.)*psi0_im)/(
                np.sqrt(abs(bb[j]))*polar[i]*psi_re
            )
    y = np.linspace(-10., 10., 201)
    dy = np.zeros(201) + 0.1
    y0 = -psi0_dre/np.sin(2.*theta)

    al = np.full((polar.size, bb.size, y.size), 0.)
    power_ratio = np.full((al.shape), np.nan)
    power_ratiob = np.full((bb.size, y.size), np.nan)
    rhy = np.full((bb.size), np.nan)
    th = np.full((al.shape), np.nan)
    conv_ygscale = np.full((polar.size, bb.size), np.nan)
    rhg = np.full((polar.size, bb.size), np.nan)

    for i in range(polar.size):
        for j in range(bb.size):
            al[i, j, ...] = (y**2 + g[i, j]**2 + np.sqrt(
                (y**2 - g[i, j]**2 + abs(kk)**2 - 1.)**2 + 4.*(
                    g[i, j]*y - rek
                )**2
            ))/np.sqrt((abs(kk)**2 - 1.)**2 + 4.*(rek**2))
            # reflecting power or power ratio R_dyn
            power_ratio[i, j, ...] = (Fmod/Fbmod)*(
                al[i, j, :] - np.sqrt((al[i, j, :]**2) - 1.)
            )
            # integration of the power ratio over dy
            power_ratiob[j, :] = power_ratio[i, j, ...]
            rhy[j] = np.sum(dy*power_ratiob[j, :])
            # conversion formula from y-scale to glancing angle scale
            th[i, j, ...] = (
                -y*polar[i]*psi_re*np.sqrt(abs(bb[j])) + psi0_dre*(
                    (1. - bb[j])/2.
                )
            )/(bb[j]*np.sin(2.*theta))
            # integrated reflecting power in the glancing angle scale
            # r(i=0): normal component & r(i=1): parallel component
            conv_ygscale[i, ...] = (polar[i]*psi_re)/(
                np.sqrt(abs(bb))*np.sin(2*theta)
            )
            rhg[i, ...] = conv_ygscale[i, :]*rhy

    # Integrated reflectivity and rocking curve width
    def lin_interp(x, y, i, half):
        return x[i] + (x[i+1] - x[i])*((half - y[i])/(y[i+1] - y[i]))

    def half_max_x(x, y):
        half = np.max(y)/2.
        signs = np.sign(np.add(y, -half))
        zero_cross = (signs[0:-2] != signs[1:-1])
        zero_cross_ind = np.where(zero_cross)[0]
        return [
            lin_interp(x, y, zero_cross_ind[0], half),
            lin_interp(x, y, zero_cross_ind[1], half),
        ]

    P_dyn = np.full((bb.size), np.nan)
    rhg_perp = P_dyn.copy()
    rhg_para = P_dyn.copy()
    det = P_dyn.copy()

    for j in range(bb.size):
        rhg_perp[j] = rhg[0, j]
        rhg_para[j] = rhg[1, j]
        # each component accounts for half the intensity of the incident beam
        # if not polarized: reflecting power is an average over the 2 states
        P_dyn[j] = np.sum(rhg[:, j])/2.
        if P_dyn[j] < 1e-7:
            msg = (
                "Please check the equations for integrated reflectivity:\n"
                "the value of P_dyn ({}) is less than 1e-7.\n".format(P_dyn[j])
            )
            raise Exception(msg)
        hmx = half_max_x(th[0, j, :], power_ratio[0, j, :])
        det[j] = hmx[1] - hmx[0]

    # Normalization for alpha=0 case
    if use_non_parallelism:
        rhg_perp_norm = np.full((rhg_perp.size), np.nan)
        rhg_para_norm = np.full((rhg_para.size), np.nan)
        det_norm = np.full((det.size), np.nan)
        nn = (det.size/2.)
        if (nn % 2) == 0:
            nn = int(nn - 1)
        else:
            nn = int(nn - 0.5)
        det_norm = det/det[nn]
        rhg_perp_norm = rhg_perp/rhg_perp[nn]
        rhg_para_norm = rhg_para/rhg_para[nn]

    if use_non_parallelism:
        return (
            alpha, bb, polar, g, y, y0, power_ratio, th,
            rhg, rhg_perp_norm, rhg_para_norm,
            P_per, P_mos, P_dyn, det, det_norm,
        )
    else:
        return (
            alpha, bb, polar, g, y, y0, power_ratio, th,
            rhg, P_per, P_mos, P_dyn, det,
        )


def CrystalBragg_plot_atomic_scattering_factor(
    sol_si=None, sol_o=None, asf_si=None, asf_o=None,
):

    # Check inputs
    # ------------

    lc = [sol_si is None, sol_o is None, asf_si is None, asf_o is None]
    if any(lc):
        msg = (
            "Please make sure that all entry arguments are valid and not None!"
        )
        raise Exception(msg)

    # Plot
    # ----

    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(1, 1)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlabel(r'sin($\theta$)/$\lambda$')
    ax.set_ylabel("atomic scattering factor")
    ax.plot(sol_si, asf_si, label="Si")
    ax.plot(sol_o, asf_o, label="O")
    ax.legend()


def CrystalBragg_plot_power_ratio(
    ih=None, ik=None, il=None, lamb=None,
    theta=None, theta_deg=None,
    th=None, power_ratio=None, y=None, y0=None,
    bb=None, polar=None, alpha=None,
    use_non_parallelism=None, na=None,
):

    # Plot
    # ----

    fig1 = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(1, 1)
    ax = fig1.add_subplot(gs[0, 0])
    ax.set_title(
        'Qz, ' + f'({ih},{ik},{il})' + fr', $\lambda$={lamb} A' +
        fr', Bragg angle={np.round(theta_deg, decimals=3)}$\deg$'
    )
    ax.set_xlabel(r'$\theta$-$\theta_{B}$ (rad)')
    ax.set_ylabel('P$_H$/P$_0$')

    if use_non_parallelism:
        alpha_deg = alpha*(180/np.pi)
        nalpha = alpha_deg.size
        if (nalpha % 2) == 0:
            nalpha = int(nalpha - 1)
        else:
            nalpha = int(nalpha - 0.5)
        dd = np.r_[0, int(nalpha/2), nalpha]
        let = {'I': dd[0], 'II': dd[1], 'III': dd[2]}
    else:
        alpha_deg = np.r_[alpha]*(180/np.pi)
        dd = np.r_[0]
        let = {'I': dd[0]}

    for j in range(na):
        if any(j == dd):
            ind = np.where(
                power_ratio[0, j, :] == np.amax(power_ratio[0, j, :])
            )
            keylist = list(let.keys())
            valuelist = list(let.values())
            valuedd = valuelist.index(j)
            keydd = keylist[valuedd]
            ax.text(
                th[0, j, ind],
                np.max(power_ratio[0, j, :] + 0.005),
                '({})'.format(keydd),
            )
            ax.plot(
                th[0, j, :],
                power_ratio[0, j, :],
                'k-',
                label=r'normal, ({}): $\alpha$=({})deg'.format(
                    keydd, np.round(alpha_deg[j], 3)
                ),
            )
            ax.plot(
                th[1, j, :],
                power_ratio[1, j, :],
                'k:',
                label=r'parallel',
            )
    # ax.axvline(y0, linetsyle=":", label='pattern centeri in y-scale')
    ax.legend()


def CrystalBragg_plot_reflect_glancing(
    ih=None, ik=None, il=None, lamb=None, theta=None, theta_deg=None,
    alpha=None, bb=None, th=None,
    rhg=None, rhg_perp_norm=None, rhg_para_norm=None, det=None,
):

    # Plot
    # ----

    fig2 = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(1, 1)
    ax = fig2.add_subplot(gs[0, 0])
    ax.set_title(
        'Qz, ' + f'({ih},{ik},{il})' + fr', $\lambda$={lamb} A' +
        fr', Bragg angle={np.round(theta_deg, decimals=3)}deg'
    )
    ax.set_xlabel(r'$\alpha$ (deg)')
    ax.set_xlim(-theta_deg - 10., theta_deg + 10.)
    ax.set_ylim(0., 5.)

    alpha_deg = alpha*(180/np.pi)
    f = scipy.interpolate.interp1d(alpha_deg, abs(bb), kind='cubic')
    alpha_deg_bis = np.linspace(-alpha_deg, alpha_deg, 21)
    bb_bis = f(alpha_deg_bis)
    ax.plot(
        alpha_deg,
        det,
        'k:',
        label=r'RC width $\Delta\theta$ (normalized)',
    )
    ax.plot(
        alpha_deg,
        rhg_perp_norm,
        'k-',
        label='P$_{dyn}$ (normal comp.) (normalized)',
    )
    ax.plot(
        alpha_deg,
        rhg_para_norm,
        'k--',
        label='P$_{dyn}$ (parallel comp.) (normalized)',
    )
    ax.plot(
        alpha_deg_bis[:, 0],
        bb_bis[:, 0],
        'k-.',
        label='|b|',
    )
    ax.legend()

