

# Built-in
import sys
import os
import warnings
import copy

# Common
import numpy as np
import scipy.interpolate
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
    therm_exp=None, plot_therm_exp=None,
    plot_asf=None, plot_power_ratio=None,
    plot_asymmetry=None, plot_cmaps=None,
    verb=None, returnas=None,
):
    """ The code evaluates, for a given wavelength and Miller indices set,
    the inter-plane distance d, the Bragg angle of reference and the complex
    structure factor for alpha-Quartz crystals.
    Then, the power ratio and their integrated reflectivity, for 3 differents
    models (perfect, mosaic and dynamical) are computed to obtain
    rocking curves, for parallel and perpendicular photon polarizations.

    The alpha-Quartz, symmetry group D(4,3), is assumed left-handed.
    It makes a difference for certain planes, for example between
    (h,k,l)=(2,0,3) and (2,0,-3).
    The alpha-Quartz structure is hexagonal with 3 molecules of SiO2 in the
    unit cell.

    The possibility to add a non-parallelism between crystal's optical surface
    and inter-atomic planes is available. Rocking curve plots are updated
    in order to show 3 cases: non-parallelism equal to zero and both limit
    cases close to +/- the reference Bragg angle.
    The relation between the value of this non-parallelism and 3 physical
    quantities can also be plotted: the integrated reflectivity for both
    components of polarization, the rocking curve widths and the asymmetry
    parameter b.

    According to Delgado-Aparicio et al.(2013), temperatures changes can affect
    the crystal inter-planar distance and thus introduce new errors concerning
    the Bragg focalization.
    We are introducing temperature changes option around a mean temperature of
    25°C and studying the effect caused by a temperature change of +/- 25°C.
    Plots of the power ratio are taking into account both limit cases of the
    temperatures changes, in addition to the limit cases of the asymmetry angle.

    Finally, colormap plots can be used to study the dependency of main RC
    components such as the integrated reflectivity, its maximum, its width and
    the shift effect from the corresponding glancing angle of the maximum, with
    respect to the temperatures changes and the asymmetry angle.

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
    therm_exp:    str
        Compute relative changes of the crystal inter-planar distance by
        thermal expansion
    plot_therm_exp:    str
        Plot the variation of the crystal inter-planar distance with respect to
        the temperature variation
    plot_asf:    str
        Plotting the atomic scattering factor thanks to data with respect to
        sin(theta)/lambda
    plot_power_ratio:    str
        Plot the power ratio with respect to the glancing angle
    plot_asymmetry:    str
        Plot relations between the integrated reflectivity, the intrinsic
        width and the parameter b vs the glancing angle
    plot_cmaps:    str
        Build colormaps of the main properties of the rocking curves wanted
        (integrated and maxmimum values, FWMD and shift from reference curve)
        with respect to the asymmetry angle alpha and the temperature changes
    verb:    str
        True or False to print the content of the results dictionnary 'dout'
    returnas:    str
        Entry 'dict' to allow optionnal returning of 'dout' dictionnary
    """

    # Check inputs
    # ------------

    if therm_exp is None:
        therm_exp = False
    if plot_therm_exp is None and therm_exp is not False:
        plot_therm_exp = True
    if use_non_parallelism is None:
        use_non_parallelism = False
    if na is None:
        na = 51
    nn = (na/2.)
    if (nn % 2) == 0:
        nn = int(nn - 1)
    else:
        nn = int(nn - 0.5)
    if plot_asf is None:
        plot_asf = False
    if plot_power_ratio is None:
        plot_power_ratio = True
    if plot_asymmetry is None and use_non_parallelism is not False:
        plot_asymmetry = True
    if plot_cmaps is None and therm_exp is not False:
        plot_cmaps = True
    if verb is None:
        verb = True
    if returnas is None:
        returnas = None

    ih, ik, il, lamb = CrystBragg_check_inputs_rockingcurve(
        ih=ih, ik=ik, il=il, lamb=lamb,
    )

    # Main crystal parameters
    # -----------------------

    # Classical electronical radius, in Angstroms,
    # from the NIST Reference on Constants, Units and Uncertainty,
    # CODATA 2018 recommended values
    re = 2.817940e-5

    # Atomic number of Si and O atoms
    Zsi = 14.
    Zo = 8.

    # Position of the three Si atoms in the unit cell,
    # from Wyckoff "Crystal Structures"
    u = 0.465
    xsi = np.r_[-u, u, 0.]
    ysi = np.r_[-u, 0., u]
    zsi = np.r_[1./3., 0., 2./3.]
    Nsi = np.size(xsi)

    # Position of the six O atoms in the unit cell,
    # from Wyckoff "Crystal Structures"
    x = 0.415
    y = 0.272
    z = 0.120
    xo = np.r_[x, y - x, -y, x - y, y, -x]
    yo = np.r_[y, -x, x - y, -y, x, y - x]
    zo = np.r_[z, z + 1./3., z + 2./3., -z, 2./3. - z, 1./3. - z]
    No = np.size(xo)

    # Computation of the unit cell volume, inter-planar distance,
    # sin(theta)/lambda parameter and Bragg angle (rad, deg) associated to lamb
    (
        T0, TD, a1, c1, Volume, d_atom, sol, sin_theta, theta, theta_deg,
    ) = CrystBragg_comp_lattice_spacing(
        ih=ih, ik=ik, il=il, lamb=lamb, na=na, nn=nn,
        therm_exp=therm_exp, plot_therm_exp=plot_therm_exp,
    )

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

    # Calculation of the structure factor for the alpha-quartz crystal
    # ----------------------------------------------------------------

    # Atomic absorption coefficient for Si and O as a function of lamb
    mu_si = 1.38e-2*(lamb**2.79)*(Zsi**2.73)
    mu_si1 = 5.33e-4*(lamb**2.74)*(Zsi**3.03)
    if lamb > 6.74:
        mu_si = mu_si1
    mu_o = 5.4e-3*(lamb**2.92)*(Zo**3.07)
    mu = 2.65e-8*(7.*mu_si + 8.*mu_o)/15.

    # Interpolation of atomic scattering factor ("f") in function of sol
    # ("_re") for the real part and ("_im") for the imaginary part
    fsi_re = scipy.interpolate.interp1d(sol_si, asf_si)
    dfsi_re = 0.1335*lamb - 0.006
    fsi_re = fsi_re(sol) + dfsi_re
    fsi_im = 5.936e-4*Zsi*(mu_si/lamb)

    fo_re = scipy.interpolate.interp1d(sol_o, asf_o)
    dfo_re = 0.1335*lamb - 0.206
    fo_re = fo_re(sol) + dfo_re
    fo_im = 5.936e-4*Zo*(mu_o/lamb)

    # Structure factor ("F") for (hkl) reflection
    phasesi = np.full((xsi.size), np.nan)
    phaseo = np.full((xo.size), np.nan)
    for i in range(xsi.size):
        phasesi[i] = ih*xsi[i] + ik*ysi[i] + il*zsi[i]
    for j in range(xo.size):
        phaseo[j] = ih*xo[j] + ik*yo[j] + il*zo[j]

    Fsi_re1 = np.full((sol.size), np.nan)
    Fsi_re2 = Fsi_re1.copy()
    Fo_re1 = Fsi_re1.copy()
    Fo_re2 = Fsi_re1.copy()
    for i in range(sol.size):
        Fsi_re1[i] = np.sum(fsi_re[i]*np.cos(2*np.pi*phasesi))
        Fsi_re2[i] = np.sum(fsi_re[i]*np.sin(2*np.pi*phasesi))
        Fo_re1[i] = np.sum(fo_re[i]*np.cos(2*np.pi*phaseo))
        Fo_re2[i] = np.sum(fo_re[i]*np.sin(2*np.pi*phaseo))

    Fsi_im1 = np.sum(fsi_im*np.cos(2*np.pi*phasesi))
    Fsi_im2 = np.sum(fsi_im*np.sin(2*np.pi*phasesi))
    Fo_im1 = np.sum(fo_im*np.cos(2*np.pi*phaseo))
    Fo_im2 = np.sum(fo_im*np.sin(2*np.pi*phaseo))

    F_re_cos = np.full((sol.size), np.nan)
    F_re_sin = F_re_cos.copy()
    for i in range(sol.size):
        F_re_cos[i] = Fsi_re1[i] + Fo_re1[i]
        F_re_sin[i] = Fsi_re2[i] + Fo_re2[i]
    F_im_cos = Fsi_im1 + Fo_im1
    F_im_sin = Fsi_im2 + Fo_im2

    F_re = np.full((sol.size), np.nan)
    for i in range(sol.size):
        F_re[i] = np.sqrt(F_re_cos[i]**2 + F_re_sin[i]**2)
    F_im = np.sqrt(F_im_cos**2 + F_im_sin**2)

    # Calculation of Fourier coefficients of polarization
    # ---------------------------------------------------

    Fmod = np.full((sol.size), np.nan)
    Fbmod = Fmod.copy()
    kk = Fmod.copy()
    rek = Fmod.copy()
    psi_re = Fmod.copy()
    psi0_dre = Fmod.copy()
    psi0_im = Fmod.copy()
    for i in range(sol.size):
        # Expression of the Fourier coef. psi_H
        Fmod[i] = np.sqrt(
            F_re[i]**2 + F_im**2 - 2.*(
                F_re_cos[i]*F_im_sin - F_im_cos*F_re_sin[i]
            )
        )
        # psi_-H equivalent to (-ih, -ik, -il)
        Fbmod[i]= np.sqrt(
            F_re[i]**2 + F_im**2 - 2.*(
                F_re_sin[i]*F_im_cos - F_re_cos[i]*F_im_sin
            )
        )
        if Fmod[i] == 0.:
            Fmod[i] == 1e-30
        if Fbmod[i] == 0.:
            Fbmod[i] == 1e-30
        # Ratio imaginary part and real part of the structure factor
        kk[i] = F_im/F_re[i]
        # Real part of kk
        rek[i] = (F_re_cos[i]*F_im_cos + F_re_sin[i]*F_im_sin)/(F_re[i]**2.)
        # Real part of psi_H
        psi_re[i] = (re*(lamb**2)*F_re[i])/(np.pi*Volume[i])
        # Zero-order real part (averaged)
        psi0_dre[i] = -re*(lamb**2)*(
            No*(Zo + dfo_re) + Nsi*(Zsi + dfsi_re)
        )/(np.pi*Volume[i])
        # Zero-order imaginary part (averaged)
        psi0_im[i] = -re*(lamb**2)*(No*fo_im + Nsi*fsi_im)/(np.pi*Volume[i])

    # Power ratio and their integrated reflectivity for 3 crystals models:
    # perfect (Darwin model), ideally mosaic thick and dynamical
    # --------------------------------------------------------------------

    if use_non_parallelism:
        (
            alpha, bb, polar, g, y, power_ratio, max_pr, th,
            rhg, rhg_perp, rhg_para, rhg_perp_norm, rhg_para_norm,
            P_per, P_mos, P_dyn,
            det_perp, det_para, det_perp_norm, det_para_norm,
            shift_thmaxpr_perp, shift_thmaxpr_para,
        ) = CrystBragg_comp_integrated_reflect(
            lamb=lamb, re=re, Volume=Volume, Zo=Zo, theta=theta, mu=mu,
            F_re=F_re, psi_re=psi_re, psi0_dre=psi0_dre, psi0_im=psi0_im,
            Fmod=Fmod, Fbmod=Fbmod, kk=kk, rek=rek,
            model=['perfect', 'mosaic', 'dynamical'],
            use_non_parallelism=use_non_parallelism, na=na, nn=nn,
            therm_exp=therm_exp,
        )
    else:
        (
            alpha, bb, polar, g, y, power_ratio, max_pr, th,
            rhg, P_per, P_mos, P_dyn, det_perp, det_para,
        ) = CrystBragg_comp_integrated_reflect(
            lamb=lamb, re=re, Volume=Volume, Zo=Zo, theta=theta, mu=mu,
            F_re=F_re, psi_re=psi_re, psi0_dre=psi0_dre, psi0_im=psi0_im,
            Fmod=Fmod, Fbmod=Fbmod, kk=kk, rek=rek,
            model=['perfect', 'mosaic', 'dynamical'],
            use_non_parallelism=use_non_parallelism, na=na, nn=nn,
            therm_exp=therm_exp,
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
        CrystalBragg_plot_power_ratio_vs_glancing_angle(
            ih=ih, ik=ik, il=il, lamb=lamb,
            theta=theta, theta_deg=theta_deg,
            th=th, power_ratio=power_ratio, y=y,
            bb=bb, polar=polar, alpha=alpha,
            use_non_parallelism=use_non_parallelism, na=na, nn=nn,
            therm_exp=therm_exp, T0=T0, TD=TD,
        )

    # Plot integrated reflect., asymmetry angle & RC width vs glancing angle
    # ----------------------------------------------------------------------

    if plot_asymmetry:
        CrystalBragg_plot_rc_components_vs_asymmetry(
            ih=ih, ik=ik, il=il, lamb=lamb,
            theta=theta, theta_deg=theta_deg,
            alpha=alpha, bb=bb, th=th, rhg=rhg,
            rhg_perp_norm=rhg_perp_norm,
            rhg_para_norm=rhg_para_norm,
            det_perp_norm=det_perp_norm,
            det_para_norm=det_para_norm,
            therm_exp=therm_exp, nn=nn, T0=T0, TD=TD,
        )

    # Plot colormaps of main properties of RC vs temperature changes and
    # asymmetry angle
    # ------------------------------------------------------------------

    if plot_cmaps:
        CrystalBragg_plot_cmaps_rc_components_vs_asymmetry_temp(
            ih=ih, ik=ik, il=il, lamb=lamb,
            therm_exp=therm_exp, T0=T0, TD=TD,
            alpha=alpha, power_ratio=power_ratio, th=th,
            rhg_perp=rhg_perp, rhg_para=rhg_para,
            max_pr=max_pr, det_perp=det_perp, det_para=det_para,
            shift_thmaxpr_perp=shift_thmaxpr_perp,
            shift_thmaxpr_para=shift_thmaxpr_para,
        )

    # Print results
    # -------------
    rhg_perp = rhg[0]
    rhg_para = rhg[1]

    if not use_non_parallelism and not therm_exp:
        P_dyn = P_dyn[0, 0]
        rhg_perp = rhg[0, 0, 0]
        rhg_para = rhg[1, 0, 0]
        det_perp = det_perp[0, 0]

    dout = {
        'Wavelength (A)\n': lamb,
        'Miller indices\n': (ih, ik, il),
        'Inter-reticular distance (A)\n': d_atom,
        'Volume of the unit cell (A^3)\n': Volume,
        'Bragg angle of reference (rad, deg)\n': (theta, theta_deg),
        'Ratio imag & real part of structure factor\n': kk,
        'Integrated reflectivity\n': {
            'perfect model': P_per,
            'mosaic model': P_mos,
            'dynamical model': P_dyn,
        },
        'P_{dyn,para}/P_{dyn,perp} (integrated values)\n': rhg_para/rhg_perp,
        'RC width (perp. compo)\n': det_perp,
    }
    if use_non_parallelism:
        dout['Non-parallelism angles (deg)\n'] = alpha*(180/np.pi)

    if verb is True:
        dout['Inter-reticular distance (A)\n'] = np.round(d_atom, decimals=3)
        dout['Volume of the unit cell (A^3)\n'] = np.round(Volume, decimals=3)
        dout['Bragg angle of reference (rad, deg)\n'] = (
            np.round(theta, decimals=3), np.round(theta_deg, decimals=3),
        )
        dout['Ratio imag & real part of structure factor\n'] = (
            np.round(kk, decimals=3,)
        )
        dout['Integrated reflectivity\n']['perfect model'] = (
            np.round(P_per, decimals=9),
        )
        dout['Integrated reflectivity\n']['mosaic model'] = (
            np.round(P_mos, decimals=9),
        )
        dout['Integrated reflectivity\n']['dynamical model'] = (
            np.round(P_dyn, decimals=9),
        )
        dout['P_{dyn,para}/P_{dyn,perp} (integrated values)\n'] = np.round(
            rhg_para/rhg_perp, decimals=9,
        )
        dout['RC width (perp. compo)\n'] = np.round(det_perp, decimals=6)
        lstr = [f'\t -{k0}: {V0}' for k0, V0 in dout.items()]
        msg = (
            " The following data was calculated:\n"
            + "\n".join(lstr)
        )
        print(msg)

    if returnas is dict:
        return dout


# ##########################################################
# ##########################################################
#                       Check inputs
# ##########################################################
# ##########################################################


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


# ##########################################################
# ##########################################################
#             Computation of 2d lattice spacing
#                    and rocking curves
# ##########################################################
# ##########################################################


def CrystBragg_comp_lattice_spacing(
    ih=None, ik=None, il=None, lamb=None, na=None, nn=None,
    therm_exp=None, plot_therm_exp=None,
):

    # Lattice constants and thermal expansion coefficients for Qz
    # -----------------------------------------------------------

    # Inter-atomic distances into hexagonal cell unit and associated volume
    # at 25°C (=298K) in Angstroms, from Wyckoff "Crystal Structures"
    a0 = 4.91304
    c0 = 5.40463

    # Thermal expansion coefficients in the directions parallel to a0 & c0
    # Values in 1/°C <=> 1/K
    alpha_a = 13.37e-6
    alpha_c = 7.97e-6

    # Computation of the inter-planar spacing
    # ---------------------------------------

    T0 = 25  # Reference temperature in °C
    if therm_exp:
        TD = np.linspace(-T0, T0, na)
    else:
        TD = np.r_[0.]

    d_atom = np.full((TD.size), np.nan)
    a1, c1 = d_atom.copy(), d_atom.copy()
    d_num, d_den = d_atom.copy(), d_atom.copy()
    Volume, sol = d_atom.copy(), d_atom.copy()
    sin_theta, theta, theta_deg = d_atom.copy(), d_atom.copy(), d_atom.copy()

    for i in range(TD.size):
        a1[i] = a0*(1 + alpha_a*TD[i])
        c1[i] = c0*(1 + alpha_c*TD[i])
        Volume[i] = (a1[i]**2.)*c1[i]*np.sqrt(3.)/2.
        d_num[i] = np.sqrt(3.)*a1[i]*c1[i]
        d_den[i] = np.sqrt(
            4.*(ih**2 + ik**2 + ih*ik)*(c1[i]**2) + 3.*(il**2)*(a1[i]**2)
        )
        d_atom[i] = d_num[i]/d_den[i]
        if d_atom[i] < lamb/2.:
            msg = (
                "According to Bragg law, Bragg scattering need d > lamb/2!\n"
                "Please check your wavelength argument.\n"
            )
            raise Exception(msg)
        sol[i] = 1./(2.*d_atom[i])
        sin_theta[i] = lamb/(2.*d_atom[i])
        theta[i] = np.arcsin(sin_theta[i])
        theta_deg[i] = theta[i]*(180./np.pi)

        lc = [theta_deg[i] < 10., theta_deg[i] > 89.]
        if any(lc):
            msg = (
                "The computed value of theta is behind the arbitrary limits.\n"
                "Limit condition: 10° < theta < 89° and\n"
                "theta = ({})°\n".format(theta_deg)
            )
            raise Exception(msg)

    # Plot
    # ----

    if plot_therm_exp:
        CrystalBragg_plot_thermal_expansion_vs_d(
            ih=ih, ik=ik, il=il, lamb=lamb,
            T0=T0, TD=TD, d_atom=d_atom, nn=nn,
        )

    return T0, TD, a1, c1, Volume, d_atom, sol, sin_theta, theta, theta_deg


def CrystBragg_comp_integrated_reflect(
    lamb=None, re=None, Volume=None, Zo=None, theta=None, mu=None,
    F_re=None, psi_re=None, psi0_dre=None, psi0_im=None,
    Fmod=None, Fbmod=None, kk=None, rek=None,
    model=[None, None, None],
    use_non_parallelism=None, na=None, nn=None,
    therm_exp=None,
):
    """For simplification and line savings reasons, whether use_non_parallelism
    is True or False, alpha and bb arrays have the same shape
    For the same reasons, the theta-dimension, depending on therm_exp arg,
    is present in all the arrays, even if it means having a extra dimension
    equal to 1 and therfore useless.
    """

    # Asymmetry parameter b
    # ---------------------

    if not use_non_parallelism:
        alpha = np.full((na), 0.)
        bb = np.full((theta.size, alpha.size), -1.)
    else:
        alpha = np.full((na), np.nan)
        bb = np.full((theta.size, alpha.size), np.nan)
        for i in range(theta.size):
            if therm_exp:
                alpha = np.linspace(-theta[nn] + 0.01, theta[nn] - 0.01, na)
                bb[i, ...] = np.sin(alpha + theta[i])/np.sin(alpha - theta[i])
            else:
                alpha = np.linspace(-theta + 0.01, theta - 0.01, na).reshape(51)
                bb[i, ...] = np.sin(alpha + theta[i])/np.sin(alpha - theta[i])

    # Perfect (darwin) and ideally thick mosaic models
    # ------------------------------------------------

    P_per = np.full((theta.size), np.nan)
    P_mos = P_per.copy()
    for i in range(theta.size):
        P_per[i] = Zo*F_re[i]*re*(lamb**2)*(1. + abs(np.cos(2.*theta[i])))/(
            6.*np.pi*Volume[i]*np.sin(2.*theta[i])
        )

        P_mos[i] = (F_re[i]**2)*(re**2)*(lamb**3)*(
            1. + (np.cos(2.*theta[i]))**2
        )/(4.*mu*(Volume[i]**2)*np.sin(2.*theta[i]))

    # Dynamical model
    # ---------------

    # Incident wave polarization (normal & parallel components)
    ppval = np.full((theta.size), 1)
    npval = np.full((theta.size), np.nan)
    for i in range(theta.size):
        npval[i] = abs(np.cos(2.*theta[i]))
    polar = np.concatenate((ppval, npval), axis=0).reshape(2, theta.size)

    # Variables of simplification y, dy, g, L
    g = np.full((2, theta.size, alpha.size), np.nan)
    for h in range(2):
        for i in range(theta.size):
            for j in range(alpha.size):
                g[h, i, j] = (((1. - bb[i, j])/2.)*psi0_im[i])/(
                    np.sqrt(abs(bb[i, j]))*polar[h][i]*psi_re[i]
                )
    y = np.linspace(-10., 10., 201)
    dy = np.zeros(201) + 0.1

    al = np.full((2, theta.size, alpha.size, y.size), 0.)
    power_ratio = np.full((al.shape), np.nan)
    max_pr = np.full((2, theta.size, alpha.size), np.nan)
    ind_max_pr = max_pr.copy()
    power_ratiob = np.full((theta.size, alpha.size, y.size), np.nan)
    rhy = np.full((theta.size, alpha.size), np.nan)
    th = np.full((al.shape), np.nan)
    th_max_pr = max_pr.copy()
    conv_ygscale = np.full((2, theta.size, alpha.size), np.nan)
    rhg = np.full((2, theta.size, alpha.size), np.nan)

    for h in range(2):
        for i in range(theta.size):
            for j in range(alpha.size):
                al[h, i, j, ...] = (y**2 + g[h, i, j]**2 + np.sqrt(
                    (y**2 - g[h, i, j]**2 + abs(kk[i])**2 - 1.)**2 + 4.*(
                        g[h, i, j]*y - rek[i]
                    )**2
                ))/np.sqrt((abs(kk[i])**2 - 1.)**2 + 4.*(rek[i]**2))
                # Reflecting power or power ratio R_dyn
                power_ratio[h, i, j, ...] = (Fmod[i]/Fbmod[i])*(
                    al[h, i, j, :] - np.sqrt((al[h, i, j, :]**2) - 1.)
                )
                # Power ratio maximum and its index
                max_pr[h, i, j] = (power_ratio[h, i, j]).max()
                ind_max_pr[h, i, j] = np.where(
                    power_ratio[h, i, j] == max_pr[h, i, j]
                )[0][0]
                # Integration of the power ratio over dy
                power_ratiob[i, j, :] = power_ratio[h, i, j, ...]
                rhy[i, j] = np.sum(dy*power_ratiob[i, j, :])
                # Conversion formula from y-scale to glancing angle scale
                # and its corresponding value to max_pr
                th[h, i, j, ...] = (
                    -y*polar[h][i]*psi_re[i]*np.sqrt(abs(bb[i, j]))
                    + psi0_dre[i]*((1. - bb[i, j])/2.)
                )/(bb[i, j]*np.sin(2.*theta[i]))
                ind = int(ind_max_pr[h, i, j])
                th_max_pr[h, i, j] = th[h, i, j, int(ind_max_pr[h, i, j])]
                # Integrated reflecting power in the glancing angle scale
                # r(i=0): normal component & r(i=1): parallel component
                conv_ygscale[h, i, ...] = (polar[h][i]*psi_re[i])/(
                    np.sqrt(abs(bb[i]))*np.sin(2*theta[i])
                )
                rhg[h, i, ...] = conv_ygscale[h, i, :]*rhy[i, j]

    # Integrated reflectivity and rocking curve widths
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

    P_dyn = np.full((theta.size, alpha.size), np.nan)
    (
        rhg_perp, rhg_para, det_perp, det_para,
        shift_thmaxpr_perp, shift_thmaxpr_para,
    ) = (
        P_dyn.copy(), P_dyn.copy(), P_dyn.copy(), P_dyn.copy(),
        P_dyn.copy(), P_dyn.copy(),
    )

    for i in range(theta.size):
        for j in range(alpha.size):
            rhg_perp[i, j] = rhg[0, i, j]
            rhg_para[i, j] = rhg[1, i, j]
            # Each component accounts for half the intensity of the incident
            # beam; if not polarized, the reflecting power is an average over
            # the 2 polarization states
            P_dyn[i, j] = np.sum(rhg[:, i, j])/2.
            if P_dyn[i, j] < 1e-7:
                msg = (
                    "Please check the equations for integrated reflectivity:\n"
                    "the value of P_dyn ({}) is less than 1e-7.\n".format(
                        P_dyn[j],
                    )
                )
                raise Exception(msg)
            hmx_perp = half_max_x(th[0, i, j, :], power_ratio[0, i, j, :])
            hmx_para = half_max_x(th[1, i, j, :], power_ratio[1, i, j, :])
            det_perp[i, j] = hmx_perp[1] - hmx_perp[0]
            det_para[i, j] = hmx_para[1] - hmx_para[0]

    # Normalization for DeltaT=0 & alpha=0 and
    # Computation of the shift in glancing angle corresponding to
    # the maximum value of each power ratio computed (each rocking curve)
    if use_non_parallelism:
        rhg_perp_norm = np.full((rhg_perp.shape), np.nan)
        rhg_para_norm = np.full((rhg_para.shape), np.nan)
        det_perp_norm = np.full((det_perp.shape), np.nan)
        det_para_norm = np.full((det_para.shape), np.nan)
        for i in range(theta.size):
            for j in range(alpha.size):
                det_perp_norm = det_perp[i]/det_perp[i, nn]
                det_para_norm = det_para[i]/det_para[i, nn]
                rhg_perp_norm = rhg_perp[i]/rhg_perp[i, nn]
                rhg_para_norm = rhg_para[i]/rhg_para[i, nn]
                if therm_exp:
                    shift_thmaxpr_perp[i, j] = (
                        th_max_pr[0, nn, nn] - th_max_pr[0, i, j]
                    )
                    shift_thmaxpr_para[i, j] = (
                        th_max_pr[1, nn, nn] - th_max_pr[1, i, j]
                    )
                else:
                    shift_thmaxpr_perp[i, j] = (
                        th_max_pr[0, i, nn] - th_max_pr[0, i, j]
                    )
                    shift_thmaxpr_para[i, j] = (
                        th_max_pr[1, i, nn] - th_max_pr[1, i, j]
                    )

    if use_non_parallelism:
        return (
            alpha, bb, polar, g, y,
            power_ratio, max_pr, th,
            rhg, rhg_perp, rhg_para, rhg_perp_norm, rhg_para_norm,
            P_per, P_mos, P_dyn,
            det_perp, det_para, det_perp_norm, det_para_norm,
            shift_thmaxpr_perp, shift_thmaxpr_para,
        )
    else:
        return (
            alpha, bb, polar, g, y,
            power_ratio, max_pr, th,
            rhg, P_per, P_mos, P_dyn,
            det_perp, det_para,
        )


# ##########################################################
# ##########################################################
#                       Plot methods
# ##########################################################
# ##########################################################


def CrystalBragg_plot_thermal_expansion_vs_d(
    ih=None, ik=None, il=None, lamb=None,
    T0=None, TD=None, d_atom=None, nn=None,
):

    fig = plt.figure(figsize=(9, 6))
    gs = gridspec.GridSpec(1, 1)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title(
        'Hexagonal Qz, ' + f'({ih},{ik},{il})' + fr', $\lambda$={lamb} A'
    )
    ax.set_xlabel(r'$\Delta$T ($T_{0}$=25°C)')
    ax.set_ylabel("Inter-planar distance d (x1e-3) [Angstroms]")
    ax.scatter(
        TD, d_atom*(1e3),
        marker='o', c='k', alpha=0.5,
        label=r'd$_{(hkl)}$ computed points',
    )

    p = np.polyfit(TD[nn:], d_atom[nn:]*(1e3), 1)
    p2 = np.polyfit(TD[:nn], d_atom[:nn]*(1e3), 1)
    y_adj = p[0]*TD[nn:] + p[1]
    y2_adj = p2[0]*TD[:nn] + p2[1]
    ax.plot(
        TD[nn:], y_adj,
        'k-',
        label='Linear fit (0<T[°C]<25)' + '\n' +
            r'd = ' + str(np.round(p[1], 2)) +
            r'x(1 + $\alpha_{eff}$.$\Delta$T)' + '\n' +
            r'$\alpha_{eff}$ =' +
            str(np.round(p[0]/p[1], decimals=9)) +
            r'°C$^{-1}$',
        )
    ax.plot(
        TD[:nn], y2_adj,
        'k--',
        label='Linear fit (-25<T[°C]<0)' + '\n' +
            r'd = ' + str(np.round(p2[1], 2)) +
            r'x(1 + $\alpha_{eff}$.$\Delta$T)' + '\n' +
            r'$\alpha_{eff}$ =' +
            str(np.round(p2[0]/p2[1], decimals=9)) +
            r'°C$^{-1}$',
        )

    ax.legend(loc="best")


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


def CrystalBragg_plot_power_ratio_vs_glancing_angle(
    ih=None, ik=None, il=None, lamb=None,
    theta=None, theta_deg=None,
    th=None, power_ratio=None, y=None, y0=None,
    bb=None, polar=None, alpha=None,
    use_non_parallelism=None, na=None, nn=None,
    therm_exp=None, T0=None, TD=None,
):

    # Plot
    # ----

    if not therm_exp:
        fig1 = plt.figure(figsize=(8, 6))
        gs = gridspec.GridSpec(1, 1)
        ax = fig1.add_subplot(gs[0, 0])
        ax.set_title(
            'Hexagonal Qz, ' + f'({ih},{ik},{il})' + fr', $\lambda$={lamb} A' +
            fr', Bragg angle={np.round(theta_deg, decimals=3)}$\deg$'
        )
        ax.set_xlabel(r'$\theta$-$\theta_{B}$ (rad)')
        ax.set_ylabel('Power ratio P$_H$/P$_0$')
    else:
        fig1 = plt.figure(figsize=(20, 18))
        gs = gridspec.GridSpec(3, 3)
        ## 3 rows -> temperature changes -T0 < 0 < +T0
        ## 3 columns -> asymmetry angles -bragg < 0 < +bragg
        ax01 = fig1.add_subplot(gs[0, 1])
        ax00 = fig1.add_subplot(gs[0, 0])
        ax02 = fig1.add_subplot(gs[0, 2])
        ax11 = fig1.add_subplot(gs[1, 1])
        ax10 = fig1.add_subplot(gs[1, 0])
        ax12 = fig1.add_subplot(gs[1, 2])
        ax21 = fig1.add_subplot(gs[2, 1])
        ax20 = fig1.add_subplot(gs[2, 0])
        ax22 = fig1.add_subplot(gs[2, 2])
        ax01.set_title(
            'Hexagonal Qz, ' + f'({ih},{ik},{il})' + fr', $\lambda$={lamb} A,'+
            fr', Bragg angle={np.round(theta_deg[nn], decimals=3)} deg,'+
            r' $\Delta$T = (-25, 0, +25) °C,'+
            r' $\alpha$ = (-$\theta_{B}$, 0, +$\theta_{B}$) deg'
        )
        ax20.set_xlabel(r'$\theta$-$\theta_{B}$ (x1e4 rad)')
        ax21.set_xlabel(r'$\theta$-$\theta_{B}$ (x1e4 rad)')
        ax22.set_xlabel(r'$\theta$-$\theta_{B}$ (x1e4 rad)')
        ax00.set_ylabel('Power ratio P$_H$/P$_0$')
        ax10.set_ylabel('Power ratio P$_H$/P$_0$')
        ax20.set_ylabel('Power ratio P$_H$/P$_0$')

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

    if therm_exp:
        ntemp = theta.size
        if (ntemp % 2) == 0:
            ntemp = int(ntemp - 1)
        else:
            ntemp = int(ntemp - 0.5)
        dd2 = np.r_[0, int(ntemp/2), ntemp]
    else:
        dd2 = np.r_[0]

    if not therm_exp:
        for j in range(na):
            if any(j == dd):
                ind = np.where(
                    power_ratio[0, 0, j, :] == np.amax(power_ratio[0, 0, j, :])
                )
                keylist = list(let.keys())
                valuelist = list(let.values())
                valuedd = valuelist.index(j)
                keydd = keylist[valuedd]
                ax.text(
                    th[0, 0, j, ind],
                    np.max(power_ratio[0, 0, j, :] + 0.005),
                    '({})'.format(keydd),
                )
                ax.plot(
                    th[0, 0, j, :],
                    power_ratio[0, 0, j, :],
                    'k-',
                    label=r'normal, ({}): $\alpha$=({})deg'.format(
                        keydd, np.round(alpha_deg[j], 3)
                    ),
                )
                ax.plot(
                    th[1, 0, j, :],
                    power_ratio[1, 0, j, :],
                    'k:',
                    label=r'parallel',
                )

        ax.legend()
    else:
        ## DeltaT row = -T0 = -25°C
        ## ------------------------
        ax01.plot(
            th[0, dd2[0], dd[1], :]*(1e4),
            power_ratio[0, dd2[0], dd[1], :],
            'k-',
            label = 'normal compo,\n' +
                r' $\alpha$=({})deg,'.format(
                    np.round(alpha_deg[dd[1]], 3),
                ) + '\n' +
                r' $\Delta$T=({})°C'.format(
                    TD[dd2[0]],
                ),
        )
        ind = np.where(
            power_ratio[0, dd2[0], dd[1], :] == np.amax(
                power_ratio[0, dd2[0], dd[1], :],
            )
        )
        ax00.plot(
            th[0, dd2[0], dd[0], :]*(1e4),
            power_ratio[0, dd2[0], dd[0], :],
            'k-',
            label = 'normal compo,\n' +
                r' $\alpha$=({})deg,'.format(
                    np.round(alpha_deg[dd[0]], 3),
                ) + '\n' +
                r' $\Delta$T=({})°C'.format(
                    TD[dd2[0]],
                ),
        )
        ax02.plot(
            th[0, dd2[0], dd[2], :]*(1e4),
            power_ratio[0, dd2[0], dd[2], :],
            'k-',
            label = 'normal compo,\n' +
                r' $\alpha$=({})deg,'.format(
                    np.round(alpha_deg[dd[2]], 3),
                ) + '\n' +
                r' $\Delta$T=({})°C'.format(
                    TD[dd2[0]],
                ),
        )
        ax00.axvline(
            th[0, dd2[0], dd[1], ind]*(1e4), color='blue', linestyle='--',
        )
        ax01.axvline(
            th[0, dd2[0], dd[1], ind]*(1e4), color='blue', linestyle='--',
        )
        ax02.axvline(
            th[0, dd2[0], dd[1], ind]*(1e4), color='blue', linestyle='--',
        )
        ax00.axhline(
            power_ratio[0, dd2[0], dd[1], ind], color='blue', linestyle='--',
        )
        ax01.axhline(
            power_ratio[0, dd2[0], dd[1], ind], color='blue', linestyle='--',
        )
        ax02.axhline(
            power_ratio[0, dd2[0], dd[1], ind], color='blue', linestyle='--',
        )

        ## DeltaT row = 0
        ## --------------
        ax11.plot(
            th[0, dd2[1], dd[1], :]*(1e4),
            power_ratio[0, dd2[1], dd[1], :],
            'k-',
            label = 'normal compo,\n' +
                r' $\alpha$=({})deg,'.format(
                    np.round(alpha_deg[dd[1]], 3),
                ) + '\n' +
                r' $\Delta$T=({})°C'.format(
                    TD[dd2[1]],
                ),
        )
        ind = np.where(
            power_ratio[0, dd2[1], dd[1], :] == np.amax(
                power_ratio[0, dd2[1], dd[1], :],
            )
        )
        ax10.plot(
            th[0, dd2[1], dd[0], :]*(1e4),
            power_ratio[0, dd2[1], dd[0], :],
            'k-',
            label = 'normal compo,\n' +
                r' $\alpha$=({})deg,'.format(
                    np.round(alpha_deg[dd[0]], 3),
                ) + '\n' +
                r' $\Delta$T=({})°C'.format(
                    TD[dd2[1]],
                ),
        )
        ax12.plot(
            th[0, dd2[1], dd[2], :]*(1e4),
            power_ratio[0, dd2[1], dd[2], :],
            'k-',
            label = 'normal compo,\n' +
                r' $\alpha$=({})deg,'.format(
                    np.round(alpha_deg[dd[2]], 3),
                ) + '\n' +
                r' $\Delta$T=({})°C'.format(
                    TD[dd2[1]],
                ),
        )
        ax10.axvline(
            th[0, dd2[1], dd[1], ind]*(1e4), color='green', linestyle='--',
        )
        ax11.axvline(
            th[0, dd2[1], dd[1], ind]*(1e4), color='green', linestyle='--',
        )
        ax12.axvline(
            th[0, dd2[1], dd[1], ind]*(1e4), color='green', linestyle='--',
        )
        ax10.axhline(
            power_ratio[0, dd2[1], dd[1], ind], color='green', linestyle='--',
        )
        ax11.axhline(
            power_ratio[0, dd2[1], dd[1], ind], color='green', linestyle='--',
        )
        ax12.axhline(
            power_ratio[0, dd2[1], dd[1], ind], color='green', linestyle='--',
        )

        ## DeltaT row = +T0 = +25°C
        ## ------------------------
        ax21.plot(
            th[0, dd2[2], dd[1], :]*(1e4),
            power_ratio[0, dd2[2], dd[1], :],
            'k-',
            label = 'normal compo,\n' +
                r' $\alpha$=({})deg,'.format(
                    np.round(alpha_deg[dd[1]], 3),
                ) + '\n' +
                r' $\Delta$T=({})°C'.format(
                    TD[dd2[2]],
                ),
        )
        ind = np.where(
            power_ratio[0, dd2[2], dd[1], :] == np.amax(
                power_ratio[0, dd2[2], dd[1], :],
            )
        )
        ax20.plot(
            th[0, dd2[2], dd[0], :]*(1e4),
            power_ratio[0, dd2[2], dd[0], :],
            'k-',
            label = 'normal compo,\n' +
                r' $\alpha$=({})deg,'.format(
                    np.round(alpha_deg[dd[0]], 3),
                ) + '\n' +
                r' $\Delta$T=({})°C'.format(
                    TD[dd2[2]],
                ),
        )
        ax22.plot(
            th[0, dd2[2], dd[2], :]*(1e4),
            power_ratio[0, dd2[2], dd[2], :],
            'k-',
            label = 'normal compo,\n' +
                r' $\alpha$=({})deg,'.format(
                    np.round(alpha_deg[dd[2]], 3),
                ) + '\n' +
                r' $\Delta$T=({})°C'.format(
                    TD[dd2[2]],
                ),
        )
        ax20.axvline(
            th[0, dd2[2], dd[1], ind]*(1e4), color='red', linestyle='--',
        )
        ax21.axvline(
            th[0, dd2[2], dd[1], ind]*(1e4), color='red', linestyle='--',
        )
        ax22.axvline(
            th[0, dd2[2], dd[1], ind]*(1e4), color='red', linestyle='--',
        )
        ax20.axhline(
            power_ratio[0, dd2[2], dd[1], ind], color='red', linestyle='--',
        )
        ax21.axhline(
            power_ratio[0, dd2[2], dd[1], ind], color='red', linestyle='--',
        )
        ax22.axhline(
            power_ratio[0, dd2[2], dd[1], ind], color='red', linestyle='--',
        )

        ax00.legend(); ax01.legend(); ax02.legend();
        ax10.legend(); ax11.legend(); ax12.legend();
        ax20.legend(); ax21.legend(); ax22.legend();


def CrystalBragg_plot_rc_components_vs_asymmetry(
    ih=None, ik=None, il=None, lamb=None, theta=None, theta_deg=None,
    alpha=None, bb=None, th=None,
    rhg=None, rhg_perp_norm=None, rhg_para_norm=None,
    det_perp_norm=None, det_para_norm=None,
    therm_exp=None, nn=None, T0=None, TD=None,
):

    # Plot
    # ----

    fig2 = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(1, 1)
    ax = fig2.add_subplot(gs[0, 0])
    ax.set_title(
        'Hexagonal Qz, ' + f'({ih},{ik},{il})' + fr', $\lambda$={lamb} A'
    )
    ax.set_xlabel(r'$\alpha$ (deg)')
    ax.set_ylim(0., 5.)
    if therm_exp:
        ax.set_xlim(-theta_deg[nn] - 10., theta_deg[nn] + 10.)
    else:
        ax.set_xlim(-theta_deg - 10., theta_deg + 10.)

    alpha_deg = alpha*(180/np.pi)
    f = scipy.interpolate.interp1d(alpha_deg, abs(bb), kind='cubic')
    alpha_deg_bis = np.linspace(-alpha_deg, alpha_deg, 21)
    bb_bis = f(alpha_deg_bis)
    ax.plot(
        alpha_deg,
        det_perp_norm,
        'r--',
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
        bb_bis[0, :, 0],
        'b-.',
        label='|b|',
    )
    ax.legend()


def CrystalBragg_plot_cmaps_rc_components_vs_asymmetry_temp(
    ih=None, ik=None, il=None, lamb=None,
    therm_exp=None, T0=None, TD=None,
    alpha=None, power_ratio=None, th=None,
    rhg_perp=None, rhg_para=None,
    max_pr=None, det_perp=None, det_para=None,
    shift_thmaxpr_perp=None,
    shift_thmaxpr_para=None,
):

    cmap = plt.cm.viridis
    fs = (24,16)
    dmargin = {'left': 0.05, 'right': 0.97,
               'bottom': 0.06, 'top': 0.92,
               'wspace': None, 'hspace': 0.4}

    fig = plt.figure(figsize=fs)
    gs = gridspec.GridSpec(3, 4, **dmargin)

    ax00 = fig.add_subplot(gs[0,0])
    ax00.set_title('Integrated reflectivity')
    ax01 = fig.add_subplot(gs[0,1])
    ax01.set_title('Maximum reflectivity')
    ax02 = fig.add_subplot(gs[0,2])
    ax02.set_title('Rocking curve width [deg]')
    ax03 = fig.add_subplot(gs[0,3])
    ax03.set_title('Maximum shift from reference RC (abs) [deg]')
    ax10 = fig.add_subplot(gs[1,0])
    ax11 = fig.add_subplot(gs[1,1])
    ax12 = fig.add_subplot(gs[1,2])
    ax13 = fig.add_subplot(gs[1,3])
    ax2 = fig.add_subplot(gs[2,:])

    ax00.set_ylabel(r'$\Delta$T ($T_{0}$=25°C)')
    ax10.set_ylabel(r'$\Delta$T ($T_{0}$=25°C)')
    ax10.set_xlabel(r'$\alpha$ (deg)')
    ax11.set_xlabel(r'$\alpha$ (deg)')
    ax12.set_xlabel(r'$\alpha$ (deg)')
    ax13.set_xlabel(r'$\alpha$ (deg)')

    alpha_deg = alpha*(180/np.pi)
    extent = (alpha_deg.min(), alpha_deg.max(), TD.min(), TD.max())

    ## Integrated reflectivities
    ## -------------------------
    rghmap_perp = ax00.imshow(
        np.log10(rhg_perp),
        cmap=cmap,
        #vmin=-7,
        #vmax=-4,
        origin='lower',
        extent=extent,
        aspect='auto',
    )
    cbar = plt.colorbar(
        rghmap_perp,
        orientation='vertical',
        label='Log scale',
        ax=ax00,
    )
    rghmap_para = ax10.imshow(
        np.log10(rhg_para),
        cmap=cmap,
        origin='lower',
        extent=extent,
        aspect='auto',
    )
    cbar = plt.colorbar(
        rghmap_para,
        orientation='vertical',
        label='Log scale',
        ax=ax10,
    )

    ## Maximum values of reflectivities
    ## --------------------------------
    maxpowerratio_perp = ax01.imshow(
        max_pr[0],
        cmap=cmap,
        origin='lower',
        extent=extent,
        aspect='auto',
    )
    cbar = plt.colorbar(
        maxpowerratio_perp,
        orientation='vertical',
        ax=ax01,
    )
    maxpowerratio_para = ax11.imshow(
        max_pr[1],
        cmap=cmap,
        origin='lower',
        extent=extent,
        aspect='auto',
    )
    cbar = plt.colorbar(
        maxpowerratio_para,
        orientation='vertical',
        ax=ax11,
    )

    ## Rocking curve widths
    ## --------------------
    width_perp = ax02.imshow(
        np.log10(det_perp*(180/np.pi)),
        cmap=cmap,
        origin='lower',
        extent=extent,
        aspect='auto',
    )
    cbar = plt.colorbar(
        width_perp,
        orientation='vertical',
        label='Log scale',
        ax=ax02,
    )
    width_para = ax12.imshow(
        np.log10(det_para*(180/np.pi)),
        cmap=cmap,
        origin='lower',
        extent=extent,
        aspect='auto',
    )
    cbar = plt.colorbar(
        width_para,
        orientation='vertical',
        label='Log scale',
        ax=ax12,
    )

    ## Shift on max. reflect. values from reference RC (TD = 0. & alpha=0.)
    ## --------------------------------------------------------------------
    shift_perp = ax03.imshow(
        np.log10(abs(shift_thmaxpr_perp*(180/np.pi))),
        cmap=cmap,
        origin='lower',
        extent=extent,
        aspect='auto',
    )
    cbar = plt.colorbar(
        shift_perp,
        label='Log scale \n (normal component)',
        orientation='vertical',
        ax=ax03,
    )
    shift_para = ax13.imshow(
        np.log10(abs(shift_thmaxpr_para*(180/np.pi))),
        cmap=cmap,
        origin='lower',
        extent=extent,
        aspect='auto',
    )
    cbar = plt.colorbar(
        shift_para,
        label='Log scale \n (parallel component)',
        orientation='vertical',
        ax=ax13,
    )
