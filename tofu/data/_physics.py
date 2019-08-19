


import numpy as np
import scipy.constants as scpct



def compute_bremzeff(Te=None, ne=None, zeff=None, lamb=None):
    """ Return the bremsstrahlun spectral radiance at lamb

    The plasma conditions are set by:
        - Te   (eV)
        - ne   (/m3)
        - zeff (adim.)

    The wavelength is set by the diagnostics
        - lamb (m)

    The vol. spectral emis. is returned in ph / (s.m3.sr.m)

    The computation requires an intermediate : gff(Te, zeff)
    """
    ktkeV = Te * 1.e-3
    ktJ = Te * scpct.e
    gff = 5.54 - (3.11-np.log(ktkeV))*(0.69-0.13/zeff)
    Const = ((scpct.e**6/(scpct.h*scpct.c**3*(np.pi*scpct.epsilon_0)**3))
             * np.sqrt(np.pi/(864.*scpct.m_e**3)))
    hc = scpct.h*scpct.c
    emis = Const/lamb * ne**2*zeff * np.exp(-hc/(lamb*ktJ)) * gff/np.sqrt(ktJ)
    units = r'ph / (s.m3.sr.m)'
    return emis, units


def compute_fangle(BR=None, BPhi=None, BZ=None, ne=None, lamb=None):
    """ The the vector quantity to be integrated on LOS to get faraday angle

    fangle = int_LOS ( abs(sca(quant, u_LOS)) )
    Where:
        quant = C * lamb**2 * ne * Bv

    With:
        - C = 2.615e-13  (1/T)
        - ne    (/m3)
        - lamb  (m)
        - Bv    (T)

    The resulting faraday angle (after integration) will be in radians
    """
    const = scpct.e**3 / (8.*scpct.pi**2
                          * scpct.epsilon_0 * scpct.m_e**2 * scpct.c**3)
    quant = const * lamb**2 * ne * np.array([BR, BPhi, BZ])
    units = r'rad / m'
    return quant, units
