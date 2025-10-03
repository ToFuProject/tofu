

import numpy as np
import astropy.units as asunits


# #####################################################
# #####################################################
#           Dict of functions
# #####################################################


def f2d_ppar_pperp_avalanche(
    p_par_norm=None,
    p_perp_norm=None,
    p_max_norm=None,
    Cz=None,
    lnG=None,
    Ehat=None,
    sigmap=None,
    # unused
    **kwdargs,
):
    """ See [1], eq. (6)

    [1] S. P. Pandya et al., Phys. Scr., 93, p. 115601, 2018
        doi: 10.1088/1402-4896/aaded0.
    """

    # fermi decay factor, adim
    fermi = 1. / (np.exp((p_par_norm - p_max_norm) / sigmap) + 1.)

    # ratio2
    pperp2par = p_perp_norm**2 / p_par_norm

    # distribution, adim
    dist = (
        (Ehat / (2.*np.pi*Cz*lnG))
        * (1./p_par_norm)
        * np.exp(- p_par_norm / (Cz * lnG) - 0.5*Ehat*pperp2par)
        * fermi
    )

    units = asunits.Unit('')

    return dist, units
