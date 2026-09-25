""" Analytical formulation of a 2d shielding cylinder


"""


import os
import copy


import numpy as np
import matplotlib.pyplot as plt
import datastock as ds


# ################################################################
# ################################################################
#                DEFAULTS
# ################################################################


_PATH_HERE = os.path.dirname(__file__)


# MAG FIELD
_B0_T = 0.010
_BIN_LIM_T = 0.001


# MAG PERMEABILITY
_MUR0 = 'SS1008'
_DMUR = {
    'SS1008': 2000,
    'mumetal': 1,   # 470000,
}
_MUR = np.linspace(200, 5000, 51)


# DIAMETERS
_DGEOM = {
    # 'a': 66e-3,
    'a': 69e-3,
    'a_tol': 0.5e-3,
    # 'b': 139.725e-3,
    'b': 140e-3,
    'b_tol': 0.1e-3,
    'thick': 1.5e-3,
    # length
    # 'len0': 250e-3,           # S1008
    'len0': 230e-3,
    'len1': (240-24.5)*1e-3,  # mumetal
}


# BIBLIO REF
_REF = "[1] A. Mager, IEEE Transactions on Magnetics, vol. 6, 1970"


# ################################################################
# ################################################################
#       Scan inner diameter of single shield
# ################################################################


def main(
    # inner diameter
    diam_inner=None,
    diam_inner_ref=None,
    diam_inner_ref_tol=None,
    # outer diameter
    diam_outer=None,
    diam_outer_ref=None,
    diam_outer_ref_tol=None,
    # mag fields
    B0_T=None,
    Bin_lim_T=None,
    Bin_levels=None,
    # mag permeability
    dmur0='SS1008',
    # bool
    plot=None,
    save=None,
    pfe_save=None,
):
    """ Compute the magnetic shielding of an infitine hollow cylinder

    Scans the magnetic permeability and either the inner or outer diameter
    The other diameter is set

    Additionally, any number of reference magnetic permeabiliies
    and diameter can be indicated

    Optionally plots and saves a figure

    All diameters provided in m
    All B fields provided in T

    """

    # ---------------
    # check
    # ---------------

    kwd = _check(**locals())

    # ---------------
    # Compute
    # ---------------

    # prepare
    if kwd['kd_vary'] == 'inner':
        diam_inner = kwd['diam_inner'][:, None]
        diam_outer = kwd['diam_outer_ref']
    else:
        diam_inner = kwd['diam_inner_ref']
        diam_outer = kwd['diam_outer'][:, None]

    # compute
    shield = shielding_trans(
        diam_inner=diam_inner,
        diam_outer=diam_outer,
        mur=kwd['mur'][None, :],
    )

    # derive Bin
    Bin = kwd['B0_T'] * shield

    # prepare output
    dout = copy.deepcopy(kwd)
    dout.update({'Bin_T': Bin, 'shield': shield})

    # ---------------
    # plot & save
    # ---------------

    dax = None
    if kwd['plot'] is True:
        dax = _plot(
            dout=dout,
        )

        if kwd['save'] is True:
            fig = dax['main'].figure
            fig.savefig(kwd['pfe_save'], dpi=300)
            msg = "Saved figure in:\n{kwd['pfe_save']}"
            print(msg)

    return dout, dax


# ################################################################
# ################################################################
#           Check
# ################################################################


def _check(**kwd):

    # ----------------
    # mag field
    # ----------------

    # Background
    kwd['B0_T'] = ds._generic_check._check_var(
        kwd.get('B0_T'), 'B0_T',
        types=(float, int),
        default=_B0_T,
        sign='>0',
    )

    # target
    kwd['Bin_lim_T'] = ds._generic_check._check_var(
        kwd.get('Bin_lim_T'), 'Bin_lim_T',
        types=(float, int),
        default=_BIN_LIM_T,
        sign='>0',
    )

    # levels
    if kwd.get('Bin_levels') is None:
        pow10_base = np.floor(np.log10(kwd['Bin_lim_T']))
        pow10_min = pow10_base - 1
        pow10_max = pow10_base + 1
        kwd['Bin_levels'] = np.r_[
            10**(pow10_min-1)*np.r_[1, 2, 5],
            10**pow10_min*np.r_[1, 2, 5],
            10**pow10_base*np.r_[1, 2, 5],
            10**pow10_max,
        ]
    kwd['Bin_levels'] = ds._generic_check._check_flat1darray(
        kwd.get('Bin_levels'), 'Bin_levels',
        dtype=float,
        unique=True,
        sign='>0',
    )

    # ----------------
    # mag permeability
    # ----------------

    # -----------
    # dmur0

    dmur0 = kwd.get('dmur0')

    if dmur0 is None:
        dmur0 = _MUR0

    if isinstance(dmur0, str):
        assert dmur0 in _DMUR.keys(), dmur0
        dmur0 = {dmur0: _DMUR[dmur0]}
    elif isinstance(dmur0, (int, float)):
        dmur0 = {'custom': dmur0}

    c0 = (
        isinstance(dmur0, dict)
        and all([
            isinstance(k0, str) and isinstance(v0, (float, int))
            for k0, v0 in dmur0.items()
        ])
    )
    if not c0:
        msg = (
            "Arg 'dmur0' must be a dict of the form:\n"
            "{\n"
            "\t- 'key0': float,\n"
            "\t-  ...  : float,\n"
            "\t- 'keyn': float,\n"
            "}\n"
        )
        raise Exception(msg)
    kwd['dmur0'] = dmur0

    # -----------
    # mur

    mur = kwd.get('mur')
    if mur is None:
        mur = _MUR

    kwd['mur'] = ds._generic_check._check_flat1darray(
        mur, 'mur',
        dtype=float,
        unique=True,
        sign='>0',
    )

    # ----------------
    # inner vs outer diameter scan
    # ----------------

    lc = [
        kwd.get('diam_inner') is not None
        and kwd.get('diam_outer_ref') is not None,
        kwd.get('diam_outer') is not None
        and kwd.get('diam_inner_ref') is not None,
    ]
    if np.sum(lc) != 1:
        lk = ['diam_inner', 'diam_outer_ref', 'diam_outer', 'diam_inner_ref']
        lstr = [f"\t- {k0}: {kwd[k0]}" for k0 in lk]
        msg = (
            "Provide either (xor):\n"
            "\t- diam_inner (array) and diam_outer_ref (float)\n"
            "\t- diam_outer (array) and diam_inner_ref (float)\n"
            "Provided:\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)

    # ----------------
    # Check each diam
    # ----------------

    ldiam = ['inner', 'outer']
    idiam_vary = lc.index(True)
    kd_vary = ldiam[idiam_vary]
    kd_fix = ldiam[1 - idiam_vary]

    # set to None unused
    kwd[f'diam_{kd_fix}'] = None
    kwd[f'diam_{kd_fix}_ref_tol'] = None

    # --------------
    # diam_vary

    kd = f"diam_{kd_vary}"
    kwd[kd] = ds._generic_check._check_flat1darray(
        kwd[kd], kd,
        dtype=float,
        unique=True,
        sign='>0',
    )

    kd = f"diam_{kd_vary}_ref"
    kwd[kd] = float(ds._generic_check._check_var(
        kwd[kd], kd,
        types=(int, float),
        default=np.mean(kwd[f"diam_{kd_vary}"]),
        sign='>0',
    ))

    kd = f"diam_{kd_vary}_ref_tol"
    kwd[kd] = float(ds._generic_check._check_var(
        kwd[kd], kd,
        types=(int, float),
        default=0.001,
        sign='>0',
    ))

    # --------------
    # diam_fix

    kd = f"diam_{kd_fix}_ref"
    kwd[kd] = float(ds._generic_check._check_var(
        kwd[kd], kd,
        types=(int, float),
        sign='>0',
    ))

    # --------------
    # store keys

    kwd['kd_vary'] = kd_vary
    kwd['kd_fix'] = kd_fix

    # ----------------
    # bool
    # ----------------

    # plot
    kwd['plot'] = ds._generic_check._check_var(
        kwd.get('plot'), 'plot',
        types=bool,
        default=True,
    )

    # save
    kwd['save'] = ds._generic_check._check_var(
        kwd.get('save'), 'save',
        types=bool,
        default=False,
    )

    # pfe_save
    if kwd['save'] is True:
        path = os.path.abs('.')
        name = 'MagneticShield_Transverse.png'
        kwd['pfe_save'] = ds._generic_check_check_var(
            kwd.get('pfe_save'), 'pfe_save',
            types=str,
            default=os.path.join(path, name),
        )

        if not os.path.isdir(kwd['pfe_save'].split()[0]):
            msg = (
                "Arg 'pfe_save' seems to point to a non-existing dir:\n"
                f"\t{kwd['pfe_save']}\n"
            )
            raise Exception(msg)

    else:
        kwd['pfe_save'] = None

    return kwd


# ################################################################
# ################################################################
#       shielding formula
# ################################################################


def shielding_trans(diam_inner=None, diam_outer=None, mur=None):

    num = 4 * mur
    denom0 = (1 + mur)**2
    denom1 = (diam_inner / diam_outer)**2 * (1 - mur)**2

    return num / (denom0 - denom1)


def shielding_axial(p=None):

    p2m1 = p**2 - 1
    sqp2m1 = np.sqrt(p2m1)

    return (1./p2m1) * ((p / sqp2m1) * np.log(p + sqp2m1) - 1)


# ################################################################
# ################################################################
#           Plot
# ################################################################


def _plot(
    dout=None,
    # figure
    figsize=(12, 6),
    # unused
    **kwdargs,
):

    # ---------------
    # prepare data
    # ---------------

    kvary = dout['kd_vary']
    kfix = dout['kd_fix']
    diam_vary = dout[f"diam_{kvary}"]
    diam_vary_ref = dout[f"diam_{kvary}_ref"]
    diam_vary_ref_tol = dout[f"diam_{kvary}_ref_tol"]
    diam_fix_ref = dout[f"diam_{kfix}_ref"]

    # ---------------
    # prepare figure
    # ---------------

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([.06, 0.08, 0.6, 0.80])

    # title
    ax.set_title(
        "Transverse B field in a 2d infinite shielding cylinder\n"
        f"{_REF}\n"
        f"{kfix.capitalize()} diameter = {diam_fix_ref} m\n"
        f"External field B0 = {dout['B0_T']} T  ({int(dout['B0_T']*1e3)} mT)",
        size=14,
        fontweight='bold',
    )

    # xlabel
    ax.set_xlabel(
        f"{kvary.capitalize()} diameter (m)",
        size=12,
        fontweight='bold',
    )

    # ylabel
    ax.set_ylabel(
        "Relative permeability  " + r"$\mu_r$",
        size=12,
        fontweight='bold',
    )

    # ---------------
    # contours
    # ---------------

    # contourf plot
    ax.contourf(
        diam_vary,
        dout['mur'],
        dout['Bin_T'].T*1e3,
        [0, dout['Bin_lim_T']*1e3, np.max(dout['Bin_T']*1e3)],
        cmap='RdYlGn_r',
    )

    # contour plot
    cs = ax.contour(
        diam_vary,
        dout['mur'],
        dout['Bin_T'].T*1e3,
        dout['Bin_levels']*1e3,
        colors='k',
    )

    # labelled contours
    ax.clabel(
        cs,
        cs.levels,
        inline=True,
        fontsize=14,
        fmt=lambda x: f'{x} mT',
    )

    # ---------------
    # add mur0
    # ---------------

    for k0, v0 in dout['dmur0'].items():
        ax.axhline(
            v0,
            ls='--',
            c='k',
            lw=1,
        )

        ax.text(
            diam_vary[-1], v0, k0,
            horizontalalignment='right',
            verticalalignment='bottom',
            transform=ax.transData,
            size=12,
            fontweight='bold',
        )

    # ------------------
    # add inner diameter
    # ------------------

    ax.axvspan(
        diam_vary_ref - diam_vary_ref_tol,
        diam_vary_ref + diam_vary_ref_tol,
        fc=(0.8, 0.8, 0.8, 0.5),
    )

    # ---------------
    # add expression
    # ---------------

    exp = (
        r"$\frac{B_{in}}{B_0} = $"
        + r"$\frac{4\mu_r}{(1+\mu_r)^2 - \left(\frac{a}{b}\right)^2(1-\mu_r)^2}$"
    )

    ax.text(
        1.28, 0.3, exp,
        horizontalalignment='center',
        transform=ax.transAxes,
        size=24,
    )

    # shield value
    kmur0 = list(dout['dmur0'].keys())[0]
    imur = np.argmin(np.abs(dout['mur'] - dout['dmur0'][kmur0]))
    ia = np.argmin(np.abs(diam_vary - diam_vary_ref))

    ax.plot(
        [diam_vary[ia]],
        [dout['mur'][imur]],
        marker='s',
        ms=12,
        c='r'
    )

    exp = (
        r"$\frac{B_{in}}{B_0} = $"
        + f"{dout['shield'][ia, imur]:3.2e}"
    )

    ax.text(
        1.28, 0.05, exp,
        horizontalalignment='center',
        transform=ax.transAxes,
        size=24,
    )

    # -----------------
    # add image
    # -----------------

    pfe = os.path.join(_PATH_HERE, '2dproblem.png')
    image = plt.imread(pfe)
    axim = fig.add_axes([0.72, 0.4, 0.25, 0.55])
    axim.imshow(image)
    axim.axis('off')

    return {'main': ax}


# ################################################################
# ################################################################
#           __main__
# ################################################################


if __name__ == '__main__':
    main()
