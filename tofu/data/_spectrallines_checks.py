

import numpy as np
import datastock as ds


# #############################################################################
# #############################################################################
#               ddata - special case: roman to int (SpectralLines)
# #############################################################################


def roman2int(ss):
    """
    :type s: str
    :rtype: int

    source: https://www.tutorialspoint.com/roman-to-integer-in-python
    """
    roman = {
        'I': 1,
        'V': 5,
        'X': 10,
        'L': 50,
        'C': 100,
        'D': 500,
        'M': 1000,
        'IV': 4,
        'IX': 9,
        'XL': 40,
        'XC': 90,
        'CD': 400,
        'CM': 900,
    }
    i = 0
    num = 0
    while i < len(ss):
        if i+1 < len(ss) and ss[i:i+2] in roman:
            num += roman[ss[i:i+2]]
            i += 2
        else:
            num += roman[ss[i]]
            i += 1
    return num


def int2roman(num):
    roman = {
        1000: "M",
        900: "CM",
        500: "D",
        400: "CD",
        100: "C",
        90: "XC",
        50: "L",
        40: "XL",
        10: "X",
        9: "IX",
        5: "V",
        4: "IV",
        1: "I",
    }

    def roman_num(num):
        for r in roman.keys():
            x, y = divmod(num, r)
            yield roman[r] * x
            num -= (r * x)
            if num <= 0:
                break

    return "".join([a for a in roman_num(num)])


# #############################################################################
# #############################################################################
#               element - ion - charge equivalence
# #############################################################################


def _check_elementioncharge(
    ION=None, ion=None,
    element=None, charge=None,
    warn=None,
):
    """ Specific to SpectralLines """

    if warn is None:
        warn = True

    # Assess if relevant
    lc = [
        ION is not None,
        ion is not None,
        element is not None and charge is not None,
    ]
    if not any(lc):
        if warn is True:
            msg = (
                """
                To determine ION, ion, element and charge, provide either:
                - ION:  {}
                - ion:  {}
                - element and charge: {}, {}
                """.format(ION, ion, element, charge)
            )
            warnings.warn(msg)
        return None, None, None, None

    # Get element and charge from ION if any
    if lc[0] or lc[1]:
        indc = 1
        if (lc[0] and ION[1].islower()) or (lc[1] and ion[1].islower()):
            indc = 2

        # Infer element
        elementi = ION[:indc] if lc[0] else ion[:indc]
        if element is not None and element != elementi:
            msg = (
                """
                Inconsistent ION ({}) vs element ({})
                """.format(element, elementi)
            )
            raise Exception(msg)

        # Infer charge
        if lc[0]:
            chargei = roman2int(ION[indc:]) - 1
        else:
            chargei = int(ion[indc:].replace('+', ''))
        if charge is not None and charge != chargei:
            msg = (
                """
                Inconsistent ION ({}) vs charge ({})
                """.format(charge, chargei)
            )
            raise Exception(msg)
        element = elementi
        charge = chargei
        if lc[0]:
            ioni = '{}{}+'.format(element, charge)
            if lc[1] and ioni != ion:
                msg = (
                    """
                    Inconsistent ION ({}) vs ion ({})
                    """.format(ION, ion)
                )
                raise Exception(msg)
            ion = ioni

        elif lc[1]:
            IONi = '{}{}'.format(element, int2roman(charge+1))
            if lc[0] and IONi != ION:
                msg = (
                    """
                    Inconsistent ion ({}) vs ION ({})
                    """.format(ion, ION)
                )
                raise Exception(msg)
            ION = IONi

    # ion provided -> element and charge
    elif lc[2]:
        ioni = '{}{}+'.format(element, charge)
        IONi = '{}{}'.format(element, int2roman(charge+1))
        if ion is not None and ion != ioni:
            msg = (
                """
                Inconsistent (element, charge) ({}, {}) vs ion ({})
                """.format(element, charge, ion)
            )
            raise Exception(msg)
        if ION is not None and ION != IONi:
            msg = (
                """
                Inconsistent (element, charge) ({}, {}) vs ION ({})
                """.format(element, charge, ION)
            )
            raise Exception(msg)
        ion = ioni
        ION = IONi

    return ION, ion, element, charge


def _check_elementioncharge_dict(dstatic):
    """ Specific to SpectralLines """

    # Assess if relevant
    lk = [kk for kk in ['ion', 'ION'] if kk in dstatic.keys()]
    if len(lk) == 0:
        return
    kion = lk[0]
    kION = 'ION' if kion == 'ion' else 'ion'
    if kion == 'ION':
        dstatic['ion'] = {}

    lerr = []
    for k0, v0 in dstatic[kion].items():
        try:
            if kion == 'ION':
                ION, ion, element, charge = _check_elementioncharge(
                    ION=k0,
                    ion=v0.get('ion'),
                    element=v0.get('element'),
                    charge=v0.get('charge'),
                )
            else:
                ION, ion, element, charge = _check_elementioncharge(
                    ION=v0.get('ION'),
                    ion=k0,
                    element=v0.get('element'),
                    charge=v0.get('charge'),
                )

            if ION is None:
                continue
            if kion == 'ION':
                dstatic['ion'][ion] = {
                    'ION': ION,
                    'element': element,
                    'charge': charge,
                }
            else:
                dstatic['ion'][k0]['ION'] = ION
                dstatic['ion'][k0]['element'] = element
                dstatic['ion'][k0]['charge'] = charge

        except Exception as err:
            lerr.append((k0, str(err)))

    if kion == 'ION':
        del dstatic['ION']

    if len(lerr) > 0:
        lerr = ['\t- {}: {}'.format(pp[0], pp[1]) for pp in lerr]
        msg = (
            """
            The following entries have non-conform ion / ION / element / charge
            {}
            """.format('\n'.join(lerr))
        )
        raise Exception(msg)


# #############################################################################
# #############################################################################
#               Check for computing pec / intensity
# #############################################################################


def _check_compute_pec(
    # check keys
    key=None,
    dlines=None,
    ddata=None,
    _quant_ne=None,
    _quant_Te=None,
    # check ne, te
    ne=None,
    Te=None,
    return_params=None,
):

    # Check data conformity
    lc = [
        k0 for k0 in key
        if (
            dlines[k0].get('pec') is None
            or [
                ddata[pp]['quant']
                for pp in ddata[dlines[k0]['pec']]['ref']
            ] != [_quant_ne, _quant_Te]
        )
    ]
    if len(lc) > 0:
        msg = (
            "The following lines have non-conform pec data:\n"
            + "\t- {}\n\n".format(lc)
            + "  => pec data should be tabulated vs (ne, Te)"
        )
        warnings.warn(msg)
        key = [kk for kk in key if kk not in lc]

    # Check ne, Te
    ltype = (int, float, np.integer)
    dnTe = {'ne': ne, 'Te': Te}
    for k0, v0 in dnTe.items():
        if isinstance(v0, ltype):
            dnTe[k0] = np.r_[v0]
        if isinstance(dnTe[k0], list) or isinstance(dnTe[k0], tuple):
            dnTe[k0] = np.array([dnTe[k0]])
        if not (isinstance(dnTe[k0], np.ndarray) and dnTe[k0].ndim == 1):
            msg = (
                "Arg {} should be a 1d np.ndarray!".format(k0)
            )
            raise Exception(msg)

    # -----------
    # return_neTe

    return_params = ds._generic_check._check_var(
        return_params, 'return_params',
        types=bool,
        default=False,
    )

    return key, dnTe, return_params


def _check_compute_intensity(
    # check keys
    key=None,
    concentration=None,
    shape=None,
):

    if np.isscalar(concentration):
        concentration = {k0: np.full(shape, concentration) for k0 in key}

    if isinstance(concentration, np.ndarray):
        concentration = {k0: concentration for k0 in key}

    c0 = (
        isinstance(concentration, dict)
        and all([k0 in concentration.keys() for k0 in key])
        and all([
            k0 in key
            and isinstance(cc, np.ndarray)
            and cc.shape == shape
            and np.all((cc > 0.) & (cc <= 1.))
            for k0, cc in concentration.items()
        ])
    )
    if not c0:
        msg = (
            "Arg concentration is non-conform:\n"
            f"\t- Expected: dict of shape {shape} arrays in [0, 1]\n"
            f"\t- Provided: {concentration}"
        )
        raise Exception(msg)

    return concentration
