

# Built-in
import copy
import warnings


# Common
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# tofu
from . import _rockingcurve_def as _def


# ##########################################################
# ##########################################################
#                  compute rocking curve
# ##########################################################
# ##########################################################


def compute_rockingcurve(
    # Type of crystal
    crystal=None,
    din=None,
    # Wavelength
    lamb=None,
    # Lattice modifications
    miscut=None,
    nn=None,
    alpha_limits=None,
    therm_exp=None,
    temp_limits=None,
    # Plot
    plot_therm_exp=None,
    plot_asf=None,
    plot_power_ratio=None,
    plot_asymmetry=None,
    plot_cmaps=None,
    # Returning dictionnary
    returnas=None,
):
    """ The code evaluates, for a given wavelength and Miller indices set,
    the inter-plane distance d, the Bragg angle of reference and the complex
    structure factor for a given crystal.
    Then, the power ratio and their integrated reflectivity, for 3 differents
    models (perfect, mosaic and dynamical) are computed to obtain
    rocking curves, for parallel and perpendicular photon polarizations.

    Among the type of crystals available, there is the alpha-Quartz, symmetry
    group D(4,3), is assumed left-handed.
    It makes a difference for certain planes, for example between
    (h,k,l)=(2,0,3) and (2,0,-3).
    The alpha-Quartz structure is hexagonal with 3 molecules of SiO2 in the
    unit cell.
    Also included is Germanium crystal (diamond structure, 8 Ge atoms)

    The possibility to add a miscut between crystal's optical surface
    and inter-atomic planes is available. Rocking curve plots are updated
    in order to show 3 cases: miscut equal to zero and both limit
    cases close to +/- the reference Bragg angle.
    The relation between the value of this miscut and 3 physical
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
    crystal:    str
        Crystal definition to use, among 'Quartz_110', 'Quartz_102'
        or user-defined Quartz or Germanium crystals
    din:    str
        Crystal definition dictionary to use, among 'Quartz_110', 'Quartz_102'
        or user-defined Quartz or Germanium crystals
    lamb:    float
        Wavelength of interest, in Angstroms (1e-10 m)
        ex: lamb=np.r_[3.96]
    miscut:    str
        Introduce miscut between dioptre and reflecting planes
    alpha_limits:    array
        Asymmetry angle range. Provide only both boundary limits
        Ex: np.r_[-3, 3] in radians
    nn:    int
        Number of miscut angles and thermal changes steps,
        odd number preferred in order to have a median value at 0
    therm_exp:    str
        Compute relative changes of the crystal inter-planar distance by
        thermal expansion
    temp_limits:    array
        Limits of temperature variation around an average value
        Ex: np.r_[-10, 10, 25] for between 15 and 35°C
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
    returnas:    type
        Entry 'dict' to allow optionnal returning of 'dout' dictionnary
    """

    # Check inputs
    # ------------

    (
        crystal, din, lamb,
        # lattice expansion
        therm_exp,
        miscut,
        alpha_limits,
        temp_limits,
        nn,
        na,
        # plotting
        plot_asf,
        plot_therm_exp,
        plot_power_ratio,
        plot_asymmetry,
        plot_cmaps,
        # return
        returnas,
    ) =  _checks(**locals())

    # Classical electronical radius, in Angstroms, from the NIST Reference on
    # Constants, Units and Uncertainty, CODATA 2018 recommended values
    re = 2.817940e-5

    # Computation of the unit cell volume, inter-planar distance,
    # sin(theta)/lambda parameter and Bragg angle associated to the wavelength
    # exits: T0, TD, a1, c1, Volume, d_atom, sol, sin_theta, theta, theta_deg
    dout = CrystBragg_comp_lattice_spacing(
        crystal=crystal,
        din=din,
        lamb=lamb,
        na=na,
        nn=nn,
        therm_exp=therm_exp,
        temp_limits=temp_limits,
        plot_therm_exp=plot_therm_exp,
    )

    T0 = dout['Temperature of reference (°C)']
    TD = dout['Temperature variations (°C)']
    Volume = dout['Volume (1/m3)']
    d_atom = dout['Inter-reticular spacing (A)']
    sol = dout['sinus over lambda']
    theta = dout['theta_Bragg (rad)']
    theta_deg = dout['theta_Bragg (deg)']

    # Check validity of asymmetry angle alpha limits in arguments
    alpha, bb = CrystBragg_check_alpha_angle(
        theta=theta,
        alpha_limits=alpha_limits, na=na, nn=nn,
        miscut=miscut,
        therm_exp=therm_exp,
    )


    # Calculation of the structure factor
    # -----------------------------------

    # Atomic absorption coefficient
    mu = din['mu'](lamb)

    # Number of species in crystal
    nele = len(din['atoms'])

    # Intializes matrices
    f_re = np.full((nele, sol.size), np.nan) # dim(elements, temp), zeroth order scattering
    f_im = np.full((nele), np.nan) # dim(elements,), absorption corrrection
    df_re = np.full((nele), np.nan) # dim(elements,), scattering correction

    F_re1 = np.full((nele, sol.size), np.nan) # dim(elements, temp), structure factore component
    F_re2 = np.full((nele, sol.size), np.nan) # dim(elements, temp), structure factore component
    F_im1 = np.full((nele), np.nan) # dim(elements,), structure factore component
    F_im2 = np.full((nele), np.nan) # dim(elements,), structure factore component

    # Loop over elements
    for ee, el in enumerate(din['atoms']):
        # Atomic scattering factor ("f") in function of sol
        # ("_re") for the real part and ("_im") for the imaginary part

        # Interpolates scat. correction
        df_re[ee] = din['df'+el.lower() + '_re'](lamb)

        # Interpolates abs. corrction
        f_im[ee] = din['f'+el.lower()+'_im'](lamb)

        # Loop over temperature
        for ii in range(sol.size):
            # Interpolates zeroth order scat. factor
            f_re[ee,ii] = din['f'+el.lower()+'_re'](lamb,sol[ii])

        # Structure factor ("F") for (hkl) reflection
        # xsi and ih have already been defined with din

        # Atom lattice position, dot(s, r_atom)
        phase = din['phases'][el] # dim(atoms,)

        # Calculates \sum_atom f_atom * exp(1j * 2pi * dot(s, r_atom) )
        # Loop over temp.
        for ii in range(sol.size):
            F_re1[ee,ii] = np.sum(f_re[ee,ii]*np.cos(2*np.pi*phase))
            F_re2[ee,ii] = np.sum(f_re[ee,ii]*np.sin(2*np.pi*phase))

        F_im1[ee] = np.sum(f_im[ee]*np.cos(2*np.pi*phase))
        F_im2[ee] = np.sum(f_im[ee]*np.sin(2*np.pi*phase))

    # Sums structure factor compenents over species
    F_re_cos = np.sum(F_re1, axis=0) # dim(temp,)
    F_re_sin = np.sum(F_re2, axis=0) # dim(temp,)

    F_im_cos = np.sum(F_im1, axis=0) # dim()
    F_im_sin = np.sum(F_im2, axis=0) # dim()

    # Modulus
    F_re = np.sqrt(F_re_cos**2 + F_re_sin**2) # dim(temp,)
    F_im = np.sqrt(F_im_cos**2 + F_im_sin**2) # dim()


    # Calculation of Fourier coefficients of polarization
    # ---------------------------------------------------

    Fmod = np.full((sol.size), np.nan)
    Fbmod = Fmod.copy()
    kk = Fmod.copy()
    rek = Fmod.copy()
    psi_re = Fmod.copy()
    psi0_dre = np.zeros(sol.size)
    psi0_im = np.zeros(sol.size)

    for ii in range(sol.size):
        # Expression of the Fourier coef. psi_H
        Fmod[ii] = np.sqrt(
            F_re[ii]**2 + F_im**2 - 2.*(
                F_re_cos[ii]*F_im_sin - F_im_cos*F_re_sin[ii]
            )
        )
        # psi_-H equivalent to (-ih, -ik, -il)
        Fbmod[ii] = np.sqrt(
            F_re[ii]**2 + F_im**2 - 2.*(
                F_re_sin[ii]*F_im_cos - F_re_cos[ii]*F_im_sin
            )
        )
        if Fmod[ii] == 0.:
            Fmod[ii] == 1e-30
        if Fbmod[ii] == 0.:
            Fbmod[ii] == 1e-30

        # Ratio imaginary part and real part of the structure factor
        kk[ii] = F_im / F_re[ii]

        # Real part of kk
        rek[ii] = (
            (F_re_cos[ii]*F_im_cos + F_re_sin[ii]*F_im_sin)
            / (F_re[ii]**2.)
        )

        # Real part of psi_H
        psi_re[ii] = (re*(lamb**2)*F_re[ii])/(np.pi*Volume[ii])

        # Loop over elements
        for ee, el in enumerate(din['atoms']):
            # Zero-order real part (averaged)
            psi0_dre[ii] += (
                -re * (lamb**2)
                * din['mesh']['positions'][el]['N']
                * (din['atoms_Z'][ee] + df_re[ee])
                )/(np.pi*Volume[ii])

            # Zero-order imaginary part (averaged)
            psi0_im[ii] += (
                -re*(lamb**2)
                * din['mesh']['positions'][el]['N']
                * f_im[ee]
                )/(np.pi*Volume[ii])


    # Power ratio and their integrated reflectivity for 3 crystals models:
    # perfect (Darwin model), ideally mosaic thick and dynamical
    # --------------------------------------------------------------------

    if miscut is False and therm_exp is False:
        (
            alpha, bb, polar, g, y, power_ratio, max_pr, th, dth,
            rhg, P_per, P_mos, P_dyn, det_perp, det_para,
        ) = CrystBragg_comp_integrated_reflect(
            lamb=lamb, re=re, Volume=Volume, Zo=din['atoms_Z'][-1], theta=theta, mu=mu,
            F_re=F_re, psi_re=psi_re, psi0_dre=psi0_dre, psi0_im=psi0_im,
            Fmod=Fmod, Fbmod=Fbmod, kk=kk, rek=rek,
            model=['perfect', 'mosaic', 'dynamical'],
            miscut=miscut, alpha=alpha, bb=bb,
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
            lamb=lamb, re=re, Volume=Volume, Zo=din['atoms_Z'][-1], theta=theta, mu=mu,
            F_re=F_re, psi_re=psi_re, psi0_dre=psi0_dre, psi0_im=psi0_im,
            Fmod=Fmod, Fbmod=Fbmod, kk=kk, rek=rek,
            model=['perfect', 'mosaic', 'dynamical'],
            miscut=miscut, alpha=alpha, bb=bb,
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
            din=din, lamb=lamb,
            alpha_limits=alpha_limits,
            theta=theta, theta_deg=theta_deg,
            th=th, dth=dth, power_ratio=power_ratio,
            bb=bb, polar=polar, alpha=alpha,
            miscut=miscut, na=na, nn=nn,
            therm_exp=therm_exp, T0=T0, TD=TD,
        )

    # Plot integrated reflect., asymmetry angle & RC width vs glancing angle
    # ----------------------------------------------------------------------

    if plot_asymmetry:
        CrystalBragg_plot_rc_components_vs_asymmetry(
            din=din, lamb=lamb,
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
            din=din, lamb=lamb, theta=theta,
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

    if not miscut and not therm_exp:
        P_dyn = P_dyn[0, 0]
        rhg_perp = rhg[0, 0, 0]
        rhg_para = rhg[1, 0, 0]
        det_perp = det_perp[0, 0]

    # -------------
    # store results

    # reminder: dimensions of power_ratio
    # (polar.ndim, temperature.size, alpha.size, y.size)

    dreturn = copy.deepcopy(din)
    dreturn.update({
        'wavelength': lamb,
        'Inter-reticular distance (A)': d_atom,
        'Volume (A^3)': Volume,
        'Bragg angle of reference (rad)': theta,
        'Integrated reflectivity': {
            'perfect model': P_per,
            'mosaic model': P_mos,
            'dynamical model': P_dyn,
        },
        'P_{dyn,para}/P_{dyn,perp} (integrated values)': rhg_para/rhg_perp,
        'Maximum reflectivity (perp. compo)': max_pr[0],
        'Maximum reflectivity (para. compo)': max_pr[1],
        'RC width (perp. compo)': det_perp,
        'RC width (para. compo)': det_para,
        # polar
        'polar': polar,
        # y => angles
        'y': y,
        'Glancing angles': dth,
        'Glancing angles rel': th,
        'alpha': alpha,
        # power ratio
        'Power ratio': power_ratio,
        # temperatures
        'Temperature ref': T0,
        'Temperature changes (°C)': TD,
        # miscut
        'Miscut angles (deg)': alpha*(180/np.pi),

    })

    # add miscut
    if miscut:
        dreturn['Shift from RC of reference (perp. compo)'] = shift_perp
        dreturn['Shift from RC of reference (para. compo)'] = shift_para

    if returnas is dict:
        return dreturn


# ################################################################
# ################################################################
#               Checks
# ################################################################
# ################################################################


def _checks(
    # Type of crystal
    crystal=None,
    din=None,
    # Wavelength
    lamb=None,
    # Lattice modifications
    miscut=None,
    nn=None,
    alpha_limits=None,
    therm_exp=None,
    temp_limits=None,
    # Plot
    plot_therm_exp=None,
    plot_asf=None,
    plot_power_ratio=None,
    plot_asymmetry=None,
    plot_cmaps=None,
    # Returning dictionnary
    returnas=None,
):

    # ------------
    # crystal

    din = _def._build_cry(crystal=crystal, din=din)

    # lamb
    if lamb is None:
        lamb = din['target']['lamb']

    # --------------------
    # lattice modification

    if therm_exp is None:
        therm_exp = False

    if miscut is None:
        miscut = False

    if alpha_limits is None:
        alpha_limits = np.r_[-(3/60)*np.pi/180, (3/60)*np.pi/180]

    if temp_limits is None:
        temp_limits = np.r_[-10, 10, 25]

    if nn is None:
        nn = 20
    na = 2*nn + 1

    # --------------
    # plotting args

    if plot_asf is None:
        plot_asf = False

    if plot_therm_exp is None and therm_exp is not False:
        plot_therm_exp = True

    if plot_power_ratio is None:
        plot_power_ratio = True

    if plot_asymmetry is None and miscut is not False:
        plot_asymmetry = True

    lc = [therm_exp, miscut]
    if plot_cmaps is None and all(lc) is True:
        plot_cmaps = True

    # ---------
    # returnas

    if returnas is None:
        returnas = dict

    return (
        crystal, din, lamb,
        # lattice expansion
        therm_exp,
        miscut,
        alpha_limits,
        temp_limits,
        nn,
        na,
        # plotting
        plot_asf,
        plot_therm_exp,
        plot_power_ratio,
        plot_asymmetry,
        plot_cmaps,
        # return
        returnas,
    )


# ################################################################
# ################################################################
#          Plot variations of RC components vs temperature & asymetry
#                        for multiple wavelengths
# ################################################################
# ################################################################


def plot_var_temp_changes_wavelengths(
    # Lattice parameters
    ih=None, ik=None, il=None, lambdas=None,
    # lattice modifications
    miscut=None, nn=None,
    alpha_limits=None,
    therm_exp=None,
    # Plot
    plot_therm_exp=None,
    plot_asf=None, plot_power_ratio=None,
    plot_asymmetry=None, plot_cmaps=None,
    # Plot arguments
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

    if miscut is None:
        miscut = True
    if therm_exp is None:
        therm_exp = True
    if nn is None:
        nn = 25
    na = 2*nn + 1
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
            miscut=miscut,
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
        din[lambdas[aa]]['miscut angles (deg)'] = (
            dout['miscut angles (deg)\n']
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
    alpha = din[lambdas[aa]]['miscut angles (deg)']*(np.pi/180)
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
    ih=None,
    ik=None,
    il=None,
    lamb=None,
):

    dd = {'ih': ih, 'ik': ik, 'il': il, 'lamb': lamb}
    lc = [v0 is None for k0, v0 in dd.items()]

    # prepare msg
    msg = (
        "Args h, k, l and lamb were not explicitely specified\n"
        f"\t - h: first Miller index ({ih})\n"
        f"\t - k: second Miller index ({ik})\n"
        f"\t - l: third Miller index ({il})\n"
        f"\t - lamb: wavelength of interest ({lamb})\n"
    )

    # All args are None
    if all(lc):
        ih = 1
        ik = 1
        il = 0
        lamb = self.dbragg['lambref']
        msg = "The following default values are used because " + msg
        warnings.warn(msg)

    # Some args are bot but not all
    elif any(lc):
        raise Exception(msg)

    # Some args are string values
    cdt = [type(v0) == str for k0, v0 in dd.items()]
    if any(cdt) or all(cdt):
        msg = "No str allowed - " + msg
        raise Exception(msg)

    return ih, ik, il, lamb,


def CrystBragg_check_alpha_angle(
    theta=None,
    miscut=None,
    therm_exp=None,
    alpha_limits=None,
    na=None,
    nn=None,
):

    if alpha_limits is None:
        if not miscut:
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
        if not miscut:
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

    return alpha, bb


# ##########################################################
# ##########################################################
#             Computation of 2d lattice spacing
#                    and rocking curves
# ##########################################################
# ##########################################################


def CrystBragg_comp_lattice_spacing(
    # Type of crystal
    crystal=None,
    din=None,
    lamb=None,
    # Plot
    na=None,
    nn=None,
    therm_exp=None,
    temp_limits=None,
    plot_therm_exp=None,
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
    containing the 'na' values between -10°C and +10°C.
    Indeed, the 'na'/2 value of 'd_atom' will correspond to the crystal
    inter-atomic spacing without any temperature changes.

    The values of the lattice parameters in the directions a and c for an
    alpha-quartz crystal have been picked from the book "Crystal Structures"
    of Wyckoff, as well as the thermal expansion coefficients in the directions

    Parameters:
    -----------
    crystal:    str
        Crystal definition to use, among 'Quartz_110', 'Quartz_102'
        or user-defined Quartz or Germanium crystals
    din:    str
        Crystal definition dictionary to use, among 'Quartz_110', 'Quartz_102'
        or user-defined Quartz or Germanium crystals
    ih, ik, il:    int
        Miller indices of crystal used, by default to (1,1,0)
    lamb:    float
        Wavelength of interest, in Angstroms (1e-10 m), by default to 3.96A
    """

    # Check inputs
    # ------------

    ih, ik, il, lamb = CrystBragg_check_inputs_rockingcurve(
        ih=din['miller'][0],
        ik=din['miller'][1],
        il=din['miller'][2],
        lamb=lamb,
    )

    # Prepare
    # -------

    # Crystal structure
    struct = din['mesh']['type']

    # Inter-atomic distances and thermal expansion coefficients
    a0 = din['inter_atomic']['distances']['a0']
    alpha_a = din['thermal_expansion']['coefs']['alpha_a']
    if struct == 'hexagonal':
        c0 = din['inter_atomic']['distances']['c0']
        alpha_c = din['thermal_expansion']['coefs']['alpha_c']

    # Temperature changes
    T0 = temp_limits[2]  # Reference temperature in °C
    if therm_exp:
        TD = np.linspace(temp_limits[0], temp_limits[1], na)
    else:
        TD = np.r_[0.]

    # Prepare results arrays
    # -----------------------

    d_atom = np.full((TD.size), np.nan)
    a1, c1 = d_atom.copy(), d_atom.copy()
    Volume, sol = d_atom.copy(), d_atom.copy()
    sin_theta, theta, theta_deg = d_atom.copy(), d_atom.copy(), d_atom.copy()

    # Compute (loop on temperature)
    # -----------------------------

    for ii in range(TD.size):

        # Calculates thermal expansion
        a1[ii] = a0*(1 + alpha_a*TD[ii])
        if struct == 'hexagonal':
            c1[ii] = c0*(1 + alpha_c*TD[ii])
            Volume[ii] = _def.hexa_volume(a1[ii], c1[ii])
            d_atom[ii] = _def.hexa_spacing(
                ih, ik, il, a1[ii], c1[ii],
            )
        elif struct == 'diamond':
            Volume[ii] = _def.diam_volume(a1[ii])
            d_atom[ii] = _def.diam_spacing(
                ih, ik, il, a1[ii]
            )

        if d_atom[ii] < lamb/2.:
            msg = (
                "According to Bragg law, Bragg scattering need d > lamb/2!\n"
                "Please check your wavelength argument.\n"
                f"\t- d_atom[{ii}] = {d_atom[ii]}\n"
                f"\t- lamb/2 = {lamb} / 2.\n"
            )
            raise Exception(msg)

        sol[ii] = 1./(2.*d_atom[ii])
        sin_theta[ii] = lamb / (2.*d_atom[ii])
        theta[ii] = np.arcsin(sin_theta[ii])
        theta_deg[ii] = theta[ii]*(180./np.pi)

        lc = [theta_deg[ii] < 1., theta_deg[ii] > 89.]
        if any(lc):
            msg = (
                "The computed value of theta is behind the arbitrary limits.\n"
                "Limit condition: 1° < theta < 89° and\n"
                f"theta = {theta_deg} °\n"
            )
            raise Exception(msg)

    # Plot calling
    # ------------

    if plot_therm_exp:
        CrystalBragg_plot_thermal_expansion_vs_d(
            din=din, lamb=lamb, theta=theta, theta_deg=theta_deg,
            T0=T0, TD=TD, d_atom=d_atom, nn=nn,
        )

    dout = {
        'Temperature of reference (°C)': T0,
        'Temperature variations (°C)': TD,
        'Inter_atomic distance a1 (A)': a1,
        'Inter_atomic distance c1 (A)': c1,
        'Volume (1/m3)': Volume,                # [\AA ^3]
        'Inter-reticular spacing (A)': d_atom,
        'sinus over lambda': sol,
        'sinus theta_Bragg': sin_theta,
        'theta_Bragg (rad)': theta,
        'theta_Bragg (deg)': theta_deg,
    }

    return dout


def CrystBragg_comp_integrated_reflect(
    lamb=None, re=None, Volume=None, Zo=None, theta=None, mu=None,
    F_re=None, psi_re=None, psi0_dre=None, psi0_im=None,
    Fmod=None, Fbmod=None, kk=None, rek=None,
    model=[None, None, None],
    miscut=None, alpha=None, bb=None, na=None, nn=None,
    therm_exp=None,
):
    """
    This method provide a method to compute the rocking curve of the specified
    crystal (ih, ik, il, lambda) for an alpha-Quartz crystal !
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

    For simplification and line savings reasons, whether miscut
    is True or False, alpha and bb arrays have the same shape.
    For the same reasons, the theta-dimension, depending on therm_exp arg,
    is present in all the arrays, even if it means having a extra dimension
    equal to 1 and therefore useless.

    Made to be used within the compute_rockingcurve() method.
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
                    al[h, i, j, :] - np.sqrt(al[h, i, j, :]**2 - 1.)
                )
                # Power ratio maximum and its index
                max_pr[h,i,j] = np.nanmax(power_ratio[h,i,j])
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
            if P_dyn[i, j] < 1e-9:
                msg = (
                    "Please check the equations for integrated reflectivity, "
                    "some values lower than 1e-9:\n"
                    f"\t- P_dyn[{i}, {j}] = {P_dyn[i, j]}\n"
                    f"\t- rhg[:, {i}, {j}] = {rhg[:, i, j]}\n"
                    f"\t- conv_ygscale[:, {i}, {j}] = {conv_ygscale[:, i, j]}\n"
                    f"\t- rhy[{i}, {j}] = {rhy[i, j]}\n"
                    f"\t- dy = {np.mean(dy)}\n"
                    f"\t- Fmod[{i}]/Fbmod[{i}] = {Fmod[i]/Fbmod[i]}\n"
                    f"\t- g[:, {i}, {j}] = {g[:, i, j]}\n"
                    f"\t- kk[{i}] = {kk[i]}\n"
                    f"\t- rek[{i}] = {rek[i]}\n"
                    f"\t- bb[{i}, {j}] = {bb[i, j]}\n"
                    f"\t- psi0_im[i] = {psi0_im[i]}\n"
                    f"\t- psi_re[i] = {psi_re[i]}\n"
                    f"\t- polar[:][{i}] = {polar[:, i]}\n"
                    f"\t- power_ratiob[{i}, {j}, :] = {power_ratiob[i, j, :]}\n"
                    f"\t- al[:, {i}, {j}, :] = {al[:, i, j, :]}\n"
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
    lc = [miscut is True, therm_exp is True]
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

    if miscut is False and therm_exp is False:
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
    din=None, lamb=None, theta=None, theta_deg=None,
    T0=None, TD=None, d_atom=None, nn=None,
):

    fig = plt.figure(figsize=(9, 6))
    gs = gridspec.GridSpec(1, 1)
    ax = fig.add_subplot(gs[0, 0])
    name = din['name']
    miller = np.r_[
        int(din['miller'][0]),
        int(din['miller'][1]),
        int(din['miller'][2]),
    ]
    ax.set_title(
        f'{name}' + f', ({miller[0]},{miller[1]},{miller[2]})' +
        fr', $\lambda$={lamb} $\AA$' +
        r', $\theta_{B}$=' + fr'{np.round(theta[nn], 5)} rad',
        fontsize=15,
    )
    ax.set_xlabel(r'$\Delta$T ($T_{0}$'+fr'={T0}°C)', fontsize=15)
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
            r'$\gamma_{eff}$ = ' + str(np.round(p[0]/p[1], decimals=9)) +
            r'°C$^{-1}$',
        ),
    )
    ax.legend(loc="best", fontsize=12)


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
    ax.set_ylabel("atomic_scattering factor")
    ax.plot(sol_si, asf_si, label="Si")
    ax.plot(sol_o, asf_o, label="O")
    ax.legend()


def CrystalBragg_plot_power_ratio_vs_glancing_angle(
    # Lattice parameters
    din=None, lamb=None,
    # Lattice modifications
    alpha_limits=None,
    theta=None, theta_deg=None,
    miscut=None, na=None, nn=None,
    therm_exp=None, T0=None, TD=None,
    # Diffraction pattern main components
    th=None, dth=None, power_ratio=None,
    bb=None, polar=None, alpha=None,
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

    if miscut:
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

    name = din['name']
    miller = np.r_[
        int(din['miller'][0]),
        int(din['miller'][1]),
        int(din['miller'][2]),
    ]

    lc = [
        miscut is False and therm_exp is False,
        miscut is False and therm_exp is True,
        miscut is True and therm_exp is False,
    ]
    if any(lc):
        fig1 = plt.figure(figsize=(8, 6))
        gs = gridspec.GridSpec(1, 1)
        ax = fig1.add_subplot(gs[0, 0])
        ax.set_title(
            f'{name}, ' + f'({miller[0]},{miller[1]},{miller[2]})' +
            fr', $\lambda$={lamb} $\AA$', fontsize=15,
        )
        ax.set_xlabel(r'Diffracting angle $\theta$ (rad)', fontsize=15)
        ax.set_ylabel('Power ratio P$_H$/P$_0$', fontsize=15)
    if miscut and therm_exp:
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
            f'{name}, ' + f'({miller[0]},{miller[1]},{miller[2]})' +
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
    lc = [miscut is True, miscut is False]
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
    """
    # Plot the sum of both polarizations
    lc = [miscut is True, miscut is False]
    if not therm_exp and any(lc):
        ax.plot(
            dth[0, 0, 0, :],
            power_ratio[0, 0, 0] + power_ratio[1, 0, 0],
            '-',
            c='black',
        )
        ax.axvline(
            theta, color='black', linestyle='-.',
            label=r'$\theta_B$= {} rad'.format(
                np.round(theta, 6)
            ),
        )
    """

    if not miscut and therm_exp is True:
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

    if miscut and therm_exp:
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
    din=None, lamb=None,
    theta=None, theta_deg=None,
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
    name = din['name']
    miller = np.r_[
        int(din['miller'][0]),
        int(din['miller'][1]),
        int(din['miller'][2]),
    ]
    ax.set_title(
        f'{name}' + f', ({miller[0]},{miller[1]},{miller[2]})' +
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
    din=None, lamb=None, theta=None,
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
    ax03.set_title(r'Angular shift / ideal [$\mu$rad]', fontsize=15)
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])
    ax12 = fig.add_subplot(gs[1, 2])
    ax13 = fig.add_subplot(gs[1, 3])
    # ax2 = fig.add_subplot(gs[2, :])

    ax00.set_ylabel(r'$\Delta$T ($T_{0}$' + f'={T0}°C)', fontsize=15)
    ax10.set_ylabel(r'$\Delta$T ($T_{0}$' + f'={T0}°C)', fontsize=15)
    ax10.set_xlabel(r'$\alpha$ (mrad)', fontsize=15)
    ax11.set_xlabel(r'$\alpha$ (mrad)', fontsize=15)
    ax12.set_xlabel(r'$\alpha$ (mrad)', fontsize=15)
    ax13.set_xlabel(r'$\alpha$ (mrad)', fontsize=15)

    name = din['name']
    miller = np.r_[
        int(din['miller'][0]),
        int(din['miller'][1]),
        int(din['miller'][2]),
    ]
    fig.suptitle(
        f'{name}' + f', ({miller[0]},{miller[1]},{miller[2]})' +
        fr', $\lambda$={lamb} $\AA$' +
        r', $\theta_{B}$=' + fr'{np.round(theta[nn], 5)} rad',
        fontsize=15,
    )

    alpha_deg = alpha*(180/np.pi)
    extent = (alpha.min()*1e3, alpha.max()*1e3, TD.min(), TD.max())

    # Integrated reflectivities
    # -------------------------
    # Perpendicular component
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
    # Parallel component
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
    # Perpendicular component
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
    # Parallel component
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
    # Perpendicular component
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
    # Parallel component
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
    # Perpendicular component
    spemin = (shift_perp*1e3).min()
    spemax = (shift_perp*1e3).max()
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
        shift_perp*1e3,
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
    # Parallel component
    spamin = (shift_para*1e3).min()
    spamax = (shift_para*1e3).max()
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
        shift_para*1e3,
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
