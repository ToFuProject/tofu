

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
    alpha_limits=None,
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
    temperatures changes, in addition to the limit cases of asymmetry angle.

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
    alpha_limits:    array
        Asymmetry angle range. Provide only both boundary limits
        Ex: np.r_[-3, 3] in radians
    na:    int
        Number of non-parallelism angles and thermical changes steps,
        odd number preferred in order to have a median value at 0
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
    if alpha_limits is None:
        alpha_limits = np.r_[-(5/60)*np.pi/180, (5/60)*np.pi/180]
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
    lc = [therm_exp, use_non_parallelism]
    if plot_cmaps is None and all(lc) is True:
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

    # Check validity of asymmetry angle alpha limits in arguments
    alpha, bb = CrystBragg_check_alpha_angle(
        theta=theta, alpha_limits=alpha_limits, na=na, nn=nn,
        use_non_parallelism=use_non_parallelism, therm_exp=therm_exp,
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
        Fbmod[i] = np.sqrt(
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

    if use_non_parallelism is False and therm_exp is False:
        (
            alpha, bb, polar, g, y, power_ratio, max_pr, th, dth,
            rhg, P_per, P_mos, P_dyn, det_perp, det_para,
        ) = CrystBragg_comp_integrated_reflect(
            lamb=lamb, re=re, Volume=Volume, Zo=Zo, theta=theta, mu=mu,
            F_re=F_re, psi_re=psi_re, psi0_dre=psi0_dre, psi0_im=psi0_im,
            Fmod=Fmod, Fbmod=Fbmod, kk=kk, rek=rek,
            model=['perfect', 'mosaic', 'dynamical'],
            use_non_parallelism=use_non_parallelism, alpha=alpha, bb=bb,
            na=na, nn=nn,
            therm_exp=therm_exp,
        )
    else:
        (
            alpha, bb, polar, g, y, power_ratio, max_pr, th, dth,
            rhg, rhg_perp, rhg_para, rhg_perp_norm, rhg_para_norm,
            P_per, P_mos, P_dyn,
            det_perp, det_para, det_perp_norm, det_para_norm,
            shift_perp, shift_para,
        ) = CrystBragg_comp_integrated_reflect(
            lamb=lamb, re=re, Volume=Volume, Zo=Zo, theta=theta, mu=mu,
            F_re=F_re, psi_re=psi_re, psi0_dre=psi0_dre, psi0_im=psi0_im,
            Fmod=Fmod, Fbmod=Fbmod, kk=kk, rek=rek,
            model=['perfect', 'mosaic', 'dynamical'],
            use_non_parallelism=use_non_parallelism, alpha=alpha, bb=bb,
            na=na, nn=nn,
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
            alpha_limits=alpha_limits,
            theta=theta, theta_deg=theta_deg,
            th=th, dth=dth, power_ratio=power_ratio,
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
            ih=ih, ik=ik, il=il, lamb=lamb, theta=theta,
            therm_exp=therm_exp, T0=T0, TD=TD, na=na, nn=nn,
            alpha=alpha, power_ratio=power_ratio,
            th=th, dth=dth,
            rhg_perp=rhg_perp, rhg_para=rhg_para,
            max_pr=max_pr, det_perp=det_perp, det_para=det_para,
            shift_perp=shift_perp, shift_para=shift_para,
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
        'Bragg angle of reference (rad)\n': theta,
        'Glancing angles': dth,
        'Power ratio': power_ratio,
        'Integrated reflectivity\n': {
            'perfect model': P_per,
            'mosaic model': P_mos,
            'dynamical model': P_dyn,
        },
        'P_{dyn,para}/P_{dyn,perp} (integrated values)\n': rhg_para/rhg_perp,
        'Maximum reflectivity (perp. compo)\n': max_pr[0],
        'Maximum reflectivity (para. compo)\n': max_pr[1],
        'RC width (perp. compo)\n': det_perp,
        'RC width (para. compo)\n': det_para,
    }
    if use_non_parallelism:
        dout['Non-parallelism angles (deg)\n'] = alpha*(180/np.pi)
        dout['Shift from RC of reference (perp. compo)\n'] = shift_perp
        dout['Shift from RC of reference (para. compo)\n'] = shift_para
    if therm_exp:
        dout['Temperature changes (°C)\n'] = TD

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
        dout['Maximum reflectivity (perp. compo)\n'] = np.round(
            max_pr[0], decimals=3,
        )
        dout['Maximum reflectivity (para. compo)\n'] = np.round(
            max_pr[1], decimals=3,
        )
        dout['RC width (perp. compo)\n'] = np.round(det_perp, decimals=8)
        dout['RC width (para. compo)\n'] = np.round(det_para, decimals=8)
        if use_non_parallelism:
            dout['Shift from RC of reference (perp. compo)\n'] = np.round(
                shift_perp, decimals=8,
            )
            dout['Shift from RC of reference (para. compo)\n'] = np.round(
                shift_para, decimals=8,
            )
        lstr = [f'\t -{k0}: {V0}' for k0, V0 in dout.items()]
        msg = (
            " The following data was calculated:\n"
            + "\n".join(lstr)
        )
        print(msg)

    if returnas is dict:
        return dout


# #############################################################################
# #############################################################################
#          Plot variations of RC components vs temperature & asymetry
#                        for multiple wavelengths
# #############################################################################
# #############################################################################


def plot_var_temp_changes_wavelengths(
    ih=None, ik=None, il=None, lambdas=None,
    use_non_parallelism=None, na=None,
    alpha_limits=None,
    therm_exp=None, plot_therm_exp=None,
    plot_asf=None, plot_power_ratio=None,
    plot_asymmetry=None, plot_cmaps=None,
    quantity=None,
    curv_radius=None, pixel_size=None,
):
    """ Using results from compute_rockingcurve() method, the aim is to study
    the variations of few physical quantities accoridng to the temperature
    changes for multiple wavelengths.
    It is a complementary method to get a global view, for a specific
    crystal, of the spectral shift impinging very specific wavelengths, due to
    the temperature changes and the asymetry angle.

    All args of compute_rockingcurve() method are needed to compute this,
    for each wavelength choosen in the input array 'lambdas'.
    By default, two plots are expected: variations with respect to the
    temperature changes of the inter-planar spacing and another quantity to be
    observed, by the mean of 'quantity'.
    The quantities 'integrated reflectivity', 'maximum reflectivity' and
    'rocking curve width' are available.

    By default, the 'lambdas' input array is setted to correspond to the
    characteristic spectral lines for ArXVII (He- and Li-like) crystal:
        - the line w at 3.949066 A (He-like Ar),
        - the line x at 3.965858 A (He-like Ar),
        - the line y at 3.969356 A (He-like Ar),
        - the line z at 3.994145 A (He-like Ar),
        - the line k at 3.989810 A (Li-like Ar),
    Source: NIST https://physics.nist.gov/PhysRefData/ASD/lines_form.html

    The programm will also need thr crystal's curvature radius and the pixel
    size of the detector used in order to compute, thanks to RC shifts
    computations, the inferred pixel shift.
    The curvature radius and the pixels size should both be given in mm.
    Default values are corresponding to the experimental set-up on WEST with
    the ArXVII crystal and the installed PILATUS detector.
    """

    # Check inputs
    # ------------

    if use_non_parallelism is None:
        use_non_parallelism = True
    if therm_exp is None:
        therm_exp = True
    if na is None:
        na = 51
    nn = (na/2.)
    if (nn % 2) == 0:
        nn = int(nn - 1)
    else:
        nn = int(nn - 0.5)
    if lambdas is None:
        lambdas = np.r_[3.949066, 3.965858, 3.969356, 3.994145, 3.989810]
    nlamb = lambdas.size
    if quantity is None:
        quantity = 'integrated reflectivity'
    if pixel_size is None:
        pixel_size = 0.172
    if curv_radius is None:
        curv_radius = 2745

    # Creating new dict with needed data
    # ----------------------------------

    din = {}
    for aa in range(nlamb):
        din[lambdas[aa]] = {}
        dout = compute_rockingcurve(
            ih=ih, ik=ik, il=il, lamb=lambdas[aa],
            use_non_parallelism=use_non_parallelism,
            alpha_limits=alpha_limits, na=na,
            therm_exp=therm_exp, plot_therm_exp=False,
            plot_asf=False, plot_power_ratio=False,
            plot_asymmetry=False, plot_cmaps=False,
            verb=False, returnas=dict,
        )
        din[lambdas[aa]]['Wavelength (A)'] = dout['Wavelength (A)\n']
        din[lambdas[aa]]['Inter-reticular distance (A)'] = (
            dout['Inter-reticular distance (A)\n']
        )
        din[lambdas[aa]]['Bragg angle of reference (rad)'] = (
            dout['Bragg angle of reference (rad)\n']
        )
        din[lambdas[aa]]['Non-parallelism angles (deg)'] = (
            dout['Non-parallelism angles (deg)\n']
        )
        din[lambdas[aa]]['Temperature changes (°C)'] = (
            dout['Temperature changes (°C)\n']
        )
        din[lambdas[aa]]['Integrated reflectivity'] = (
            dout['Integrated reflectivity\n']['perfect model']
        )
        din[lambdas[aa]]['Maximum reflectivity (perp. compo)'] = (
            dout['Maximum reflectivity (perp. compo)\n']
        )
        din[lambdas[aa]]['Maximum reflectivity (para. compo)'] = (
            dout['Maximum reflectivity (para. compo)\n']
        )
        din[lambdas[aa]]['RC width (perp. compo)'] = (
            dout['RC width (perp. compo)\n']
        )
        din[lambdas[aa]]['RC width (para. compo)'] = (
            dout['RC width (para. compo)\n']
        )
        din[lambdas[aa]]['Shift from RC of reference (perp. compo)'] = (
            dout['Shift from RC of reference (perp. compo)\n']
        )
        din[lambdas[aa]]['Shift from RC of reference (para. compo)'] = (
            dout['Shift from RC of reference (para. compo)\n']
        )
    for aa in range(nlamb):
        din[lambdas[aa]]['Inter-planar spacing variations (perp. compo)'] = (
            din[lambdas[aa]]['Shift from RC of reference (perp. compo)'] /
            np.tan(din[lambdas[aa]]['Bragg angle of reference (rad)'][nn])
        )
        din[lambdas[aa]]['Inter-planar spacing variations (para. compo)'] = (
            din[lambdas[aa]]['Shift from RC of reference (para. compo)'] /
            np.tan(din[lambdas[aa]]['Bragg angle of reference (rad)'][nn])
        )

    for aa in range(nlamb):
        din[lambdas[aa]]['shift in pixel'] = (
            din[lambdas[aa]]['Shift from RC of reference (perp. compo)']
            )*curv_radius*np.sin(
                din[lambdas[aa]]['Bragg angle of reference (rad)'][nn]
            )/pixel_size

    # Plots
    # -----

    # 1st: comparisons between spacing variations from theoretical (d_hkl)
    # and those induced from RC shifts computations; study of the impact of
    # asymetry and thermal expansion on few wavelengths of interest
    fig = plt.figure(figsize=(15, 9))
    gs = gridspec.GridSpec(1, 2)
    ax = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    fig.suptitle('Hexagonal Qz, ' + f'({ih},{ik},{il})', fontsize=15)
    ax.set_title('Computed spacing variations', fontsize=15)
    ax.set_xlabel(r'$\Delta$T ($T_{0}$=25°C)', fontsize=15)
    ax.set_ylabel(r'$\Delta$d$_{(hkl)}$', fontsize=15)
    ax1.set_xlabel(r'$\Delta$T ($T_{0}$=25°C)', fontsize=15)

    markers = ['o', '^', 'D', 's', 'X']

    for aa in range(nlamb):
        diff = (
            din[lambdas[aa]][
                'Inter-planar spacing variations (perp. compo)'
            ][:, nn] - (
                din[lambdas[aa]]['Inter-reticular distance (A)']
                - din[lambdas[aa]]['Inter-reticular distance (A)'][nn]
            )/din[lambdas[aa]]['Inter-reticular distance (A)'][nn]
        )
        ax.scatter(
            din[lambdas[aa]]['Temperature changes (°C)'],
            din[lambdas[aa]]['Inter-reticular distance (A)'],
            marker=markers[aa], c='k', alpha=0.5,
            label=r'$\lambda$ = ({})$\AA$'.format(lambdas[aa]),
        )
        if quantity == 'integrated reflectivity':
            ax1.set_ylabel(r'Integrated reflectivity', fontsize=15)
            ax1.scatter(
                din[lambdas[aa]]['Temperature changes (°C)'],
                din[lambdas[aa]]['Integrated reflectivity'],
                marker=markers[aa], c='k', alpha=0.5,
                label=r'$\lambda$ = ({})$\AA$'.format(lambdas[aa]),
            )
        if quantity == 'maximum reflectivity':
            ax1.set_ylabel(r'Maximum reflectivity', fontsize=15)
            ax1.scatter(
                din[lambdas[aa]]['Temperature changes (°C)'],
                din[lambdas[aa]]['Maximum reflectivity (perp. compo)'][:, nn],
                marker=markers[aa], c='k', alpha=0.5,
                label=r'$\lambda$ = ({})$\AA$'.format(lambdas[aa]),
            )
        if quantity == 'rocking curve width':
            ax1.set_ylabel(r'Rocking curve width', fontsize=15)
            ax1.scatter(
                din[lambdas[aa]]['Temperature changes (°C)'],
                din[lambdas[aa]]['RC width (perp. compo)'][:, nn],
                marker=markers[aa], c='k', alpha=0.5,
                label=r'$\lambda$ = ({})$\AA$'.format(lambdas[aa]),
            )
    ax.legend()
    ax1.legend()

    # 2nd: colormpas of inferred pixel shift computed from angular shifts of
    # rocking curves, with respect ot the temperature changes and asymetry
    # angles, for the choosen wavelengths of interest
    cmap = plt.cm.seismic
    fs = (22, 10)
    dmargin = {'left': 0.05, 'right': 0.94,
               'bottom': 0.06, 'top': 0.92,
               'wspace': 0.4, 'hspace': 0.4}

    fig2 = plt.figure(figsize=fs)
    nrows = 1
    ncols = nlamb
    gs2 = gridspec.GridSpec(nrows, ncols, **dmargin)
    fig2.suptitle(
        r'Hexagonal $\alpha$-Qz, ' + f'({ih},{ik},{il})' +
        r', inferred pixel shift $\Delta$p',
        fontsize=15,
    )
    alpha = din[lambdas[aa]]['Non-parallelism angles (deg)']*(np.pi/180)
    TD = din[lambdas[aa]]['Temperature changes (°C)']
    extent = (alpha.min(), alpha.max(), TD.min(), TD.max())

    for aa in range(ncols):
        ax2 = fig2.add_subplot(gs2[0, aa])
        if aa == 0:
            ax2.set_ylabel(r'$\Delta$T ($T_{0}$=25°C)', fontsize=15)
        ax2.set_xlabel(r'$\alpha$ (rad)', fontsize=15)
        ax2.set_title(r'$\lambda$=({}) $\AA$'.format(lambdas[aa]), fontsize=15)

        shiftmin = (din[lambdas[aa]]['shift in pixel']).min()
        shiftmax = (din[lambdas[aa]]['shift in pixel']).max()
        if abs(shiftmin) < abs(shiftmax):
            vmax = shiftmax
            vmin = -shiftmax
        if abs(shiftmin) > abs(shiftmax):
            vmax = -shiftmin
            vmin = shiftmin
        if abs(shiftmin) == abs(shiftmax):
            vmax = shiftmax
            vmin = shiftmin
        cmaps = ax2.imshow(
            din[lambdas[aa]]['shift in pixel'],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin='lower',
            extent=extent,
            aspect='auto',
        )
        cbar = plt.colorbar(
            cmaps,
            orientation='vertical',
            ax=ax2,
        )
        if aa == 4:
            cbar.set_label(
                r'Pixel shift $\Delta$p' + '\n' +
                'Perpendicular pola.',
                fontsize=15,
            )


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


def CrystBragg_check_alpha_angle(
    theta=None, alpha_limits=None, na=None, nn=None,
    use_non_parallelism=None, therm_exp=None,
):

    if alpha_limits is None:
        if not use_non_parallelism:
            alpha = np.full((na), 0.)
            bb = np.full((theta.size, alpha.size), -1.)
        else:
            alpha = np.full((na), np.nan)
            bb = np.full((theta.size, alpha.size), np.nan)
            for i in range(theta.size):
                if therm_exp:
                    alpha = np.linspace(
                        -theta[nn] + 0.01, theta[nn] - 0.01, na
                    )
                    bb[i, ...] = np.sin(alpha + theta[i])/np.sin(
                        alpha - theta[i]
                    )
                else:
                    alpha = np.linspace(
                        -theta + 0.01, theta - 0.01, na
                    ).reshape(51)
                    bb[i, ...] = np.sin(alpha + theta[i])/np.sin(
                        alpha - theta[i]
                    )
    else:
        if not use_non_parallelism:
            alpha = np.full((na), 0.)
            bb = np.full((theta.size, alpha.size), -1.)
        else:
            if therm_exp:
                aa = theta[nn]
                lc = [alpha_limits[0] < -aa, alpha_limits[1] > aa]
            else:
                lc = [alpha_limits[0] < -theta, alpha_limits[1] > theta]
            if any(lc):
                msg = (
                    "Alpha angle limits are not valid!\n"
                    "Please check the values:\n"
                    "Limit condition: -({}) < |alpha| < ({}) and\n".format(
                        theta, theta,
                    )
                    + "alpha = [({})] rad\n".format(alpha_limits)
                )
                raise Exception(msg)
            alpha = np.full((na), np.nan)
            bb = np.full((theta.size, alpha.size), np.nan)
            for i in range(theta.size):
                if therm_exp:
                    alpha = np.linspace(alpha_limits[0], alpha_limits[1], na)
                    bb[i, ...] = np.sin(alpha + theta[i])/np.sin(
                        alpha - theta[i]
                    )
                else:
                    alpha = np.linspace(
                        alpha_limits[0], alpha_limits[1], na
                    ).reshape(na)
                    bb[i, ...] = np.sin(alpha + theta[i])/np.sin(
                        alpha - theta[i]
                    )

    return alpha, bb,


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
    """
    Compute the inter-atomic spacing d_hkl for a given crystal of Miller
    indices (ih, ik ,il).
    Eiher heating or colling it have been proved experimentally affecting
    this quantity and this code will provide this case, for temperatures
    changes of +/- 25°C around an ambiant estimated at 25°C.
    This results to the variations of the inter-atomic spacing with respect
    to the temperature changes.
    An array containing these new values of d_hkl is provided through 'd_atom'
    containing the 'na' values between -25°C and +25°C.
    Indeed, the 'na'/2 value of 'd_atom' will correspond to the crystal
    inter-atomic spacing without any temperature changes.

    The values of the lattice parameters in the directions a and c for an
    alpha-quartz crystal have been picked from the book "Crystal Structures"
    of Wyckoff, as well as the thermal expansion coefficients in the directions
    """
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

    # Plot calling
    # ------------

    if plot_therm_exp:
        CrystalBragg_plot_thermal_expansion_vs_d(
            ih=ih, ik=ik, il=il, lamb=lamb, theta=theta, theta_deg=theta_deg,
            T0=T0, TD=TD, d_atom=d_atom, nn=nn,
        )

    return T0, TD, a1, c1, Volume, d_atom, sol, sin_theta, theta, theta_deg


def CrystBragg_comp_integrated_reflect(
    lamb=None, re=None, Volume=None, Zo=None, theta=None, mu=None,
    F_re=None, psi_re=None, psi0_dre=None, psi0_im=None,
    Fmod=None, Fbmod=None, kk=None, rek=None,
    model=[None, None, None],
    use_non_parallelism=None, alpha=None, bb=None, na=None, nn=None,
    therm_exp=None,
):
    """
    This method provide a method to compute the rocking curve of the specified
    crystal (ih, ik, il, lambda).
    Three crystal model are then done: perfect, mosaic and dynamical.
    The 'power_ratio' element correspond to the the reflecting power at a
    specific mathematic data, which have been converted into glancing angle
    data named 'th' for the quantity theta-theta_Bragg.
    Be careful: theta_Bragg isn't the same in all cases if the arg
    'therm_exp' have been picked!
    The conversion made of the quantity theta-thetaBragg computed is at TD=!0,
    as called 'th' has to be made adding to each the Bragg angle of reference
    at no thermal expansion AND no asymetry angle:
    (theta - thetaBragg) + thetaBragg(at TD=alpha=0) => theta[na/2] !!
    This is done because we have to get a fixed Bragg angle of reference
    through all computations of rocking curves, whether the application of
    temperature changes or asymetry angle is set.

    For simplification and line savings reasons, whether use_non_parallelism
    is True or False, alpha and bb arrays have the same shape.
    For the same reasons, the theta-dimension, depending on therm_exp arg,
    is present in all the arrays, even if it means having a extra dimension
    equal to 1 and therefore useless.
    """

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
    g = np.full((polar.ndim, theta.size, alpha.size), np.nan)
    for h in range(polar.ndim):
        for i in range(theta.size):
            for j in range(alpha.size):
                g[h, i, j] = (((1. - bb[i, j])/2.)*psi0_im[i])/(
                    np.sqrt(abs(bb[i, j]))*polar[h][i]*psi_re[i]
                )
    y = np.linspace(-10., 10., 201)
    dy = np.zeros(201) + 0.1

    al = np.full((polar.ndim, theta.size, alpha.size, y.size), 0.)
    power_ratio = np.full((al.shape), np.nan)
    power_ratiob = np.full((theta.size, alpha.size, y.size), np.nan)
    max_pr = np.full((polar.ndim, theta.size, alpha.size), np.nan)
    ind_max_pr = max_pr.copy()
    th_max_pr = max_pr.copy()
    rhy = np.full((theta.size, alpha.size), np.nan)
    th = np.full((al.shape), np.nan)
    dth = th.copy()
    conv_ygscale = np.full((polar.ndim, theta.size, alpha.size), np.nan)
    rhg = np.full((polar.ndim, theta.size, alpha.size), np.nan)

    for h in range(polar.ndim):
        for i in range(theta.size):
            for j in range(alpha.size):
                al[h, i, j, ...] = (y**2 + g[h, i, j]**2 + np.sqrt(
                    (y**2 - g[h, i, j]**2 + (kk[i])**2 - 1.)**2 + 4.*(
                        g[h, i, j]*y - rek[i]
                    )**2
                ))/np.sqrt(((kk[i])**2 - 1.)**2 + 4.*(rek[i]**2))
                # Reflecting power
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
                th[h, i, j, ...] = (
                    -y*polar[h][i]*psi_re[i]*np.sqrt(abs(bb[i, j]))
                    + psi0_dre[i]*((1. - bb[i, j])/2.)
                )/(bb[i, j]*np.sin(2.*theta[i]))
                # Find the glancing angle coordinate of the power ratio maximum
                ind = int(ind_max_pr[h, i, j])
                th_max_pr[h, i, j] = th[h, i, j, int(ind_max_pr[h, i, j])]
                # conversion of (theta-thetaBragg(TD=!0)) scale to
                # theta scale by adding the value of Theta_B at
                # alpha=TD=0 in both case
                dth[h, i, j, ...] = th[h, i, j, :] + theta[i]
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
        rhg_perp, rhg_para, pat_cent_perp, pat_cent_para,
        det_perp, det_para, shift_perp, shift_para,
    ) = (
        P_dyn.copy(), P_dyn.copy(), P_dyn.copy(), P_dyn.copy(), P_dyn.copy(),
        P_dyn.copy(), P_dyn.copy(), P_dyn.copy(),
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
            # Coordinates of full width at mid high sides of FWHM
            hmx_perp = half_max_x(dth[0, i, j, :], power_ratio[0, i, j, :])
            hmx_para = half_max_x(dth[1, i, j, :], power_ratio[1, i, j, :])
            # Center of FWMH
            pat_cent_perp[i, j] = (hmx_perp[1] + hmx_perp[0])/2.
            pat_cent_para[i, j] = (hmx_para[1] + hmx_para[0])/2.
            # Width of FWMH
            det_perp[i, j] = hmx_perp[1] - hmx_perp[0]
            det_para[i, j] = hmx_para[1] - hmx_para[0]

    # Normalization for DeltaT=0 & alpha=0 and
    # computation of the shift in glancing angle corresponding to
    # the maximum value of each power ratio computed (each rocking curve)
    lc = [use_non_parallelism is True, therm_exp is True]
    if any(lc) or all(lc):
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
                # Shift between each RC's max value or pattern center
                if therm_exp:
                    shift_perp[i, j] = (
                        pat_cent_perp[nn, nn] - pat_cent_perp[i, j]
                    )
                    shift_para[i, j] = (
                        pat_cent_para[nn, nn] - pat_cent_para[i, j]
                    )
                else:
                    shift_perp[i, j] = (
                        pat_cent_perp[i, nn] - pat_cent_perp[i, j]
                    )
                    shift_para[i, j] = (
                        pat_cent_para[i, nn] - pat_cent_para[i, j]
                    )

    if use_non_parallelism is False and therm_exp is False:
        return (
            alpha, bb, polar, g, y,
            power_ratio, max_pr, th, dth,
            rhg, P_per, P_mos, P_dyn,
            det_perp, det_para,
        )
    else:
        return (
            alpha, bb, polar, g, y,
            power_ratio, max_pr, th, dth,
            rhg, rhg_perp, rhg_para, rhg_perp_norm, rhg_para_norm,
            P_per, P_mos, P_dyn,
            det_perp, det_para, det_perp_norm, det_para_norm,
            shift_perp, shift_para,
        )


# ##########################################################
# ##########################################################
#                       Plot methods
# ##########################################################
# ##########################################################


def CrystalBragg_plot_thermal_expansion_vs_d(
    ih=None, ik=None, il=None, lamb=None, theta=None, theta_deg=None,
    T0=None, TD=None, d_atom=None, nn=None,
):

    fig = plt.figure(figsize=(9, 6))
    gs = gridspec.GridSpec(1, 1)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title(
        'Hexagonal Qz, ' + f'({ih},{ik},{il})' + fr', $\lambda$={lamb} $\AA$' +
        r', $\theta_{B}$=' + fr'{np.round(theta[nn], 5)} rad',
        fontsize=15,
    )
    ax.set_xlabel(r'$\Delta$T ($T_{0}$=25°C)', fontsize=15)
    ax.set_ylabel(r'Inter-planar distance $d_{hkl}$ [m$\AA$]', fontsize=15)
    ax.scatter(
        TD, d_atom*(1e3),
        marker='o', c='k', alpha=0.5,
        label=r'd$_{(hkl)}$ computed points',
    )
    p = np.polyfit(TD, d_atom*(1e3), 1)
    y_adj = p[0]*TD + p[1]
    ax.plot(
        TD, y_adj,
        'k-',
        label=(
            r'$d_{hkl}$ = ' + str(np.round(p[1], 3)) +
            r' x (1 + $\gamma_{eff}$.$\Delta$T)' + '\n' +
            r'$\gamma_{eff}$ = ' +
            str(np.round(p[0]/p[1], decimals=9)) +
            r'°C$^{-1}$',
        ),
    )
    ax.legend(loc="upper left", fontsize=12)


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
    alpha_limits=None,
    theta=None, theta_deg=None,
    th=None, dth=None, power_ratio=None,
    bb=None, polar=None, alpha=None,
    use_non_parallelism=None, na=None, nn=None,
    therm_exp=None, T0=None, TD=None,
):
    """ All plots of rocking curve is done, not with respect to the glancing
    angle (theta - thetaBragg) where thetaBragg may vary if the temperature
    is also varying, but according to the angle theta.
    The conversion is made by adding to the glancing angle range the value of
    the Bragg angle of reference when no asymmetry or thermal expansion is
    applied.
    """

    # Prepare
    # -------

    if use_non_parallelism:
        alpha_deg = alpha*(180/np.pi)
        nalpha = alpha_deg.size
        if (nalpha % 2) == 0:
            nalpha = int(nalpha - 1)
        else:
            nalpha = int(nalpha - 0.5)
        dd = np.r_[0, int(nalpha/2), nalpha]
        let = {'I': dd[0], 'II': dd[1], 'III': dd[2]}
        col = {'blue': dd[0], 'black': dd[1], 'red': dd[2]}
    else:
        alpha_deg = np.r_[alpha]*(180/np.pi)
        dd = np.r_[0]
        let = {'I': dd[0]}
        col = {'black': dd[0]}

    if therm_exp:
        ntemp = theta.size
        if (ntemp % 2) == 0:
            ntemp = int(ntemp - 1)
        else:
            ntemp = int(ntemp - 0.5)
        dd2 = np.r_[0, int(ntemp/2), ntemp]
        let2 = {'I': dd2[0], 'II': dd2[1], 'III': dd2[2]}
        col2 = {'blue': dd2[0], 'black': dd2[1], 'red': dd2[2]}
    else:
        dd2 = np.r_[0]
        let2 = {'I': dd2[0]}
        col2 = {'black': dd2[0]}

    lc = [
        use_non_parallelism is False and therm_exp is False,
        use_non_parallelism is False and therm_exp is True,
        use_non_parallelism is True and therm_exp is False,
    ]
    if any(lc):
        fig1 = plt.figure(figsize=(8, 6))
        gs = gridspec.GridSpec(1, 1)
        ax = fig1.add_subplot(gs[0, 0])
        ax.set_title(
            'Hexagonal Qz, ' + f'({ih},{ik},{il})' +
            fr', $\lambda$={lamb} $\AA$', fontsize=15,
        )
        ax.set_xlabel(r'$\theta$ (rad)', fontsize=15)
        ax.set_ylabel('Power ratio P$_H$/P$_0$', fontsize=15)
    if use_non_parallelism and therm_exp:
        gs = gridspec.GridSpec(3, 3)
        fig1 = plt.figure(figsize=(22, 20))
        # 3 rows -> temperature changes -T0 < 0 < +T0
        # 3 columns -> asymmetry angles -bragg < 0 < +bragg
        ax01 = fig1.add_subplot(gs[0, 1])
        ax00 = fig1.add_subplot(gs[0, 0])
        ax02 = fig1.add_subplot(gs[0, 2])
        ax11 = fig1.add_subplot(gs[1, 1])
        ax10 = fig1.add_subplot(gs[1, 0])
        ax12 = fig1.add_subplot(gs[1, 2])
        ax21 = fig1.add_subplot(gs[2, 1])
        ax20 = fig1.add_subplot(gs[2, 0])
        ax22 = fig1.add_subplot(gs[2, 2])
        fig1.suptitle(
            'Hexagonal Qz, ' + f'({ih},{ik},{il})' +
            fr', $\lambda$={lamb} $\AA$' +
            r', $\theta_{B}$=' + fr'{np.round(theta[nn], 5)} rad',
            fontsize=15,
        )
        ax00.set_title(
            r'$\alpha$=({}) arcmin'.format(np.round(alpha_deg[0]*60, 3)),
            fontsize=15
        )
        ax01.set_title(
            r'$\alpha$=({}) arcmin'.format(np.round(alpha_deg[nn]*60, 3)),
            fontsize=15
        )
        ax02.set_title(
            r'$\alpha$=({}) arcmin'.format(np.round(alpha_deg[na-1]*60, 3)),
            fontsize=15
        )
        ax022 = ax02.twinx()
        ax022.set_ylabel(
            r'$\Delta T$=({})°C'.format(TD[0]), fontsize=15
        )
        ax122 = ax12.twinx()
        ax122.set_ylabel(
            r'$\Delta T$=({})°C'.format(TD[nn]), fontsize=15
        )
        ax222 = ax22.twinx()
        ax222.set_ylabel(
            r'$\Delta T$=({})°C'.format(TD[na-1]), fontsize=15
        )
        ax20.set_xlabel(r'$\theta$ (rad)', fontsize=15)
        ax21.set_xlabel(r'$\theta$ (rad)', fontsize=15)
        ax22.set_xlabel(r'$\theta$ (rad)', fontsize=15)
        ax00.set_ylabel('Power ratio P$_H$/P$_0$', fontsize=15)
        ax10.set_ylabel('Power ratio P$_H$/P$_0$', fontsize=15)
        ax20.set_ylabel('Power ratio P$_H$/P$_0$', fontsize=15)

    # Plot
    # ----

    lc = [use_non_parallelism is True, use_non_parallelism is False]
    if not therm_exp and any(lc):
        for j in range(na):
            if any(j == dd):
                # power_ratio.shape: (polar, TD, alpha, y)
                ind = np.where(
                    power_ratio[0, 0, j, :] == np.amax(power_ratio[0, 0, j, :])
                )
                let_keylist = list(let.keys())
                let_valuelist = list(let.values())
                let_valuedd = let_valuelist.index(j)
                let_keydd = let_keylist[let_valuedd]
                c_keylist = list(col.keys())
                c_valuelist = list(col.values())
                c_valuedd = c_valuelist.index(j)
                c_keydd = c_keylist[c_valuedd]
                ax.text(
                    dth[0, 0, j, ind],
                    np.max(power_ratio[0, 0, j, :] + 0.005),
                    '({})'.format(let_keydd),
                    c=c_keydd,
                )
                ax.plot(
                    dth[0, 0, j, :],
                    power_ratio[0, 0, j, :],
                    '-',
                    c=c_keydd,
                    label=(
                        r'normal pola.,' + '\n' +
                        r' ({}): $\alpha$=({}) arcmin'.format(
                            let_keydd, np.round(alpha_deg[j]*60, 3)
                        ),
                    ),
                )
                ax.plot(
                    dth[1, 0, j, :],
                    power_ratio[1, 0, j, :],
                    '--',
                    c=c_keydd,
                    label=r'parallel pola.',
                )
        ax.axvline(
            theta, color='black', linestyle='-.',
            label=r'$\theta_B$= {} rad'.format(
                np.round(theta, 6)
            ),
        )
        ax.legend(fontsize=12)

    if not use_non_parallelism and therm_exp is True:
        colors = ['blue', 'black', 'red']
        for i in range(na):
            if any(i == dd2):
                ind = np.where(
                    power_ratio[0, i, 0, :] == np.amax(power_ratio[0, i, 0, :])
                )
                let_keylist = list(let2.keys())
                let_valuelist = list(let2.values())
                let_valuedd = let_valuelist.index(i)
                let_keydd = let_keylist[let_valuedd]
                c_keylist = list(col2.keys())
                c_valuelist = list(col2.values())
                c_valuedd = c_valuelist.index(i)
                c_keydd = c_keylist[c_valuedd]
                ax.text(
                    dth[0, i, 0, ind],
                    np.max(power_ratio[0, i, 0, :] + 0.005),
                    '({})'.format(let_keydd),
                    c=c_keydd,
                )
                ax.plot(
                    dth[0, i, 0, :],
                    power_ratio[0, i, 0, :],
                    '-',
                    c=c_keydd,
                    label=r'normal pola., ({}): $\Delta T$=({})°C'.format(
                        let_keydd, TD[i]
                    ),
                )
                ax.plot(
                    dth[1, i, 0, :],
                    power_ratio[1, i, 0, :],
                    '--',
                    c=c_keydd,
                    label=r'parallel pola.',
                )
        ax.axvline(
            theta[nn], color='black', linestyle='--',
            label=r'Bragg angle of ref. : {} rad'.format(
                np.round(theta[nn], 6)
            ),
        )
        ax.legend(fontsize=12)

    if use_non_parallelism and therm_exp:
        # DeltaT row = 0
        # --------------
        ax11.plot(
            dth[0, dd2[1], dd[1], :],
            power_ratio[0, dd2[1], dd[1], :],
            'g-',
            label='normal pola.',
        )
        ax11.axvline(
            theta[nn], color='black', linestyle='-.',
            label=r'$\theta_B$= {} rad'.format(
                np.round(theta[nn], 6)
            ),
        )
        ax11.plot(
            dth[1, dd2[1], dd[1], :],
            power_ratio[1, dd2[1], dd[1], :],
            'g--',
            label='parallel pola.',
        )
        ax10.plot(
            dth[0, dd2[1], dd[0], :],
            power_ratio[0, dd2[1], dd[0], :],
            'k-',
        )
        ax10.axvline(
            theta[nn], color='black', linestyle='-.',
        )
        ax10.plot(
            dth[1, dd2[1], dd[0], :],
            power_ratio[1, dd2[1], dd[0], :],
            'k--',
        )
        ax12.plot(
            dth[0, dd2[1], dd[2], :],
            power_ratio[0, dd2[1], dd[2], :],
            'k-',
        )
        ax12.axvline(
            theta[nn], color='black', linestyle='-.',
        )
        ax12.plot(
            dth[1, dd2[1], dd[2], :],
            power_ratio[1, dd2[1], dd[2], :],
            'k--',
        )
        # DeltaT row = -T0 = -25°C
        # ------------------------
        ax01.plot(
            dth[0, dd2[0], dd[1], :],
            power_ratio[0, dd2[0], dd[1], :],
            'k-',
        )
        ax01.axvline(
            theta[nn], color='black', linestyle='-.',
        )
        ax01.plot(
            dth[1, dd2[0], dd[1], :],
            power_ratio[1, dd2[0], dd[1], :],
            'k--',
        )
        ax00.plot(
            dth[0, dd2[0], dd[0], :],
            power_ratio[0, dd2[0], dd[0], :],
            'k-',
        )
        ax00.axvline(
            theta[nn], color='black', linestyle='-.',
        )
        ax00.plot(
            dth[1, dd2[0], dd[0], :],
            power_ratio[1, dd2[0], dd[0], :],
            'k--',
        )
        ax02.plot(
            dth[0, dd2[0], dd[2], :],
            power_ratio[0, dd2[0], dd[2], :],
            'k-',
        )
        ax02.axvline(
            theta[nn], color='black', linestyle='-.',
        )
        ax02.plot(
            dth[1, dd2[0], dd[2], :],
            power_ratio[1, dd2[0], dd[2], :],
            'k--',
        )
        # DeltaT row = +T0 = +25°C
        # ------------------------
        ax21.plot(
            dth[0, dd2[2], dd[1], :],
            power_ratio[0, dd2[2], dd[1], :],
            'k-',
        )
        ax21.axvline(
            theta[nn], color='black', linestyle='-.',
        )
        ax21.plot(
            dth[1, dd2[2], dd[1], :],
            power_ratio[1, dd2[2], dd[1], :],
            'k--',
        )
        ax20.plot(
            dth[0, dd2[2], dd[0], :],
            power_ratio[0, dd2[2], dd[0], :],
            'k-',
        )
        ax20.axvline(
            theta[nn], color='black', linestyle='-.',
        )
        ax20.plot(
            dth[1, dd2[2], dd[0], :],
            power_ratio[1, dd2[2], dd[0], :],
            'k--',
        )
        ax22.plot(
            dth[0, dd2[2], dd[2], :],
            power_ratio[0, dd2[2], dd[2], :],
            'k-',
        )
        ax22.axvline(
            theta[nn], color='black', linestyle='-.',
        )
        ax22.plot(
            dth[1, dd2[2], dd[2], :],
            power_ratio[1, dd2[2], dd[2], :],
            'k--',
        )
        # Replot everywhere in transprent the case alpha=deltaT=0
        dax = {
            'ax00': ax00, 'ax01': ax01, 'ax02': ax02,
            'ax10': ax10, 'ax12': ax12,
            'ax20': ax20, 'ax21': ax21, 'ax22': ax22,
        }
        for key in dax:
            dax[key].plot(
                dth[0, dd2[1], dd[1], :],
                power_ratio[0, dd2[1], dd[1], :],
                'g-', alpha=0.5,
            )
            dax[key].plot(
                dth[1, dd2[1], dd[1], :],
                power_ratio[1, dd2[1], dd[1], :],
                'g--', alpha=0.65,
            )

        ax11.legend()


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
        'Hexagonal Qz, ' + f'({ih},{ik},{il})' +
        fr', $\lambda$={lamb} $\AA$'
    )
    ax.set_xlabel(r'$\alpha$ (deg)', fontsize=15)
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
    ih=None, ik=None, il=None, lamb=None, theta=None,
    therm_exp=None, T0=None, TD=None, na=None, nn=None,
    alpha=None, power_ratio=None, th=None, dth=None,
    rhg_perp=None, rhg_para=None,
    max_pr=None, det_perp=None, det_para=None,
    shift_perp=None, shift_para=None,
):

    cmap = plt.cm.viridis
    fs = (24, 16)
    dmargin = {'left': 0.05, 'right': 0.95,
               'bottom': 0.06, 'top': 0.92,
               'wspace': None, 'hspace': 0.4}

    fig = plt.figure(figsize=fs)
    gs = gridspec.GridSpec(2, 4, **dmargin)

    ax00 = fig.add_subplot(gs[0, 0])
    ax00.set_title('Integrated reflectivity', fontsize=15)
    ax01 = fig.add_subplot(gs[0, 1])
    ax01.set_title('Maximum reflectivity', fontsize=15)
    ax02 = fig.add_subplot(gs[0, 2])
    ax02.set_title('Rocking curve width [rad]', fontsize=15)
    ax03 = fig.add_subplot(gs[0, 3])
    ax03.set_title('Shift from reference RC [rad]', fontsize=15)
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])
    ax12 = fig.add_subplot(gs[1, 2])
    ax13 = fig.add_subplot(gs[1, 3])
    # ax2 = fig.add_subplot(gs[2, :])

    ax00.set_ylabel(r'$\Delta$T ($T_{0}$=25°C)', fontsize=15)
    ax10.set_ylabel(r'$\Delta$T ($T_{0}$=25°C)', fontsize=15)
    ax10.set_xlabel(r'$\alpha$ (rad) [x1e3]', fontsize=15)
    ax11.set_xlabel(r'$\alpha$ (rad) [x1e3]', fontsize=15)
    ax12.set_xlabel(r'$\alpha$ (rad) [x1e3]', fontsize=15)
    ax13.set_xlabel(r'$\alpha$ (rad) [x1e3]', fontsize=15)

    fig.suptitle(
        'Hexagonal Qz, ' + f'({ih},{ik},{il})' +
        fr', $\lambda$={lamb} $\AA$' +
        r', $\theta_{B}$=' + fr'{np.round(theta[nn], 5)} rad',
        fontsize=15,
    )

    alpha_deg = alpha*(180/np.pi)
    extent = (alpha.min()*1e3, alpha.max()*1e3, TD.min(), TD.max())

    # Integrated reflectivities
    # -------------------------
    rghmap_perp = ax00.imshow(
        rhg_perp,
        cmap=cmap,
        origin='lower',
        extent=extent,
        aspect='auto',
    )
    cbar = plt.colorbar(
        rghmap_perp,
        orientation='vertical',
        ax=ax00,
    )
    rghmap_para = ax10.imshow(
        rhg_para,
        cmap=cmap,
        origin='lower',
        extent=extent,
        aspect='auto',
    )
    cbar = plt.colorbar(
        rghmap_para,
        orientation='vertical',
        ax=ax10,
    )
    # Maximum values of reflectivities
    # --------------------------------
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
    # Rocking curve widths
    # --------------------
    width_perp = ax02.imshow(
        det_perp,
        cmap=cmap,
        origin='lower',
        extent=extent,
        aspect='auto',
    )
    cbar = plt.colorbar(
        width_perp,
        orientation='vertical',
        ax=ax02,
    )
    width_para = ax12.imshow(
        det_para,
        cmap=cmap,
        origin='lower',
        extent=extent,
        aspect='auto',
    )
    cbar = plt.colorbar(
        width_para,
        orientation='vertical',
        ax=ax12,
    )
    # Shift on max. reflect. values from reference RC (TD = 0. & alpha=0.)
    # --------------------------------------------------------------------
    spemin = (shift_perp).min()
    spemax = (shift_perp).max()
    if abs(spemin) < abs(spemax):
        vmax = spemax
        vmin = -spemax
    if abs(spemin) > abs(spemax):
        vmax = -spemin
        vmin = spemin
    if abs(spemin) == abs(spemax):
        vmax = spemax
        vmin = spemin
    shift_perp_cmap = ax03.imshow(
        shift_perp,
        vmin=vmin,
        vmax=vmax,
        cmap=plt.cm.seismic,
        origin='lower',
        extent=extent,
        aspect='auto',
    )
    cbar = plt.colorbar(
        shift_perp_cmap,
        orientation='vertical',
        ax=ax03,
    )
    cbar.set_label('(perpendicular pola.)', fontsize=15)
    spamin = (shift_para).min()
    spamax = (shift_para).max()
    if abs(spamin) < abs(spamax):
        vmax = spamax
        vmin = -spamax
    if abs(spamin) > abs(spamax):
        vmax = -spamin
        vmin = spamin
    if abs(spamin) == abs(spamax):
        vmax = spamax
        vmin = spamin
    shift_para_cmap = ax13.imshow(
        shift_para,
        vmin=vmin,
        vmax=vmax,
        cmap=plt.cm.seismic,
        origin='lower',
        extent=extent,
        aspect='auto',
    )
    cbar = plt.colorbar(
        shift_para_cmap,
        orientation='vertical',
        ax=ax13,
    )
    cbar.set_label('(parallel pola.)', fontsize=15)