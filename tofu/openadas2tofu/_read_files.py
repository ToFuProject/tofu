
# Built-in
import os
import re
import itertools as itt
import warnings

# Common
import numpy as np
from scipy.interpolate import RectBivariateSpline as scpRectSpl


__all__ = ['step03_read', 'step03_read_all']


_DTYPES = {'adf11': ['acd', 'ccd', 'scd', 'plt', 'prb'],
           'adf15': None}
_DEG = 1
_PECASFUNC = True


# #############################################################################
# #############################################################################
#                       Utility functions
# #############################################################################


def _get_PATH_LOCAL():
    pfe = os.path.join(os.path.expanduser('~'), '.tofu', 'openadas2tofu')
    if os.path.isdir(pfe):
        return pfe
    else:
        return None


def _get_subdir_from_pattern(path, pattern, mult=None):
    """ Get list of files matching patterns in path

    If no match => Exception
    If multiple matches
        => mult = True: pass
        => mult = 'warn': warning
        => mult = 'err': Exception
    """
    # Check inputs
    if mult is None:
        mult = 'err'
    if mult not in [True, 'warn', 'err']:
        msg = (
            "Arg mult must be in [True, 'warn', 'err']!\n"
            + "\t- provided: {}".format(mult)
        )
        raise Exception(msg)

    if isinstance(pattern, str):
        pattern = [pattern]

    # Get matches
    ld = [
        dd for dd in os.listdir(path)
        if os.path.isdir(os.path.join(path, dd))
        and all([pp in dd for pp in pattern])
    ]
    if len(ld) != 1:
        msg = ("You have no / many directories in your local "
               + "~/.tofu/openadas2tofu/ matching the desired file type:\n"
               + "\t- path: {}\n".format(path)
               + "\t- provided (all): {}\n".format(pattern)
               + "\t- available: {}\n".format(ld)
               + "  => download the data with "
               + "tf.openadas2tofu.step02_download()")
        if len(ld) == 0:
            raise Exception(msg)
        else:
            if mult == 'err':
                raise Exception(msg)
            elif mult == 'warn':
                warnings.warn(msg)
    return [os.path.join(path, dd) for dd in ld]


def _get_available_elements_from_path(path=None, typ1=None):
    # Check inputs
    if not os.path.isdir(path):
        msg = (
            "Provided path is not an existing directory!\n"
            + "\t- provided: {}".format(path)
        )
        raise Exception(msg)

    ltyp = ['adf15']
    if typ1 not in ltyp:
        msg = (
            "Only the following types of files are handled up to now:\n"
            + "\t- handled: {}\n".format(ltyp)
            + "\t- provided: {}".format(typ1)
        )
        raise Exception(msg)

    if typ1 == 'adf15':
        lf = [
            ff for ff in os.listdir(path)
            if all([ss in ff for ss in ['pec', '][']])
        ]
        element = [ff[ff.index('][')+2:] for ff in lf]
    return element


def _format_for_DataStock_adf15(
    dout,
    dsource0=None,
    dref0=None,
    ddata0=None,
    dlines0=None,
):
    """
    Format dout from step03_read_all() for SPectralLines object
    (separated te, ne, ions, sources, lines)
    """

    # ----------------------------------------------
    # Remove already known lines of dlines0 provided

    if dlines0 is not None:
        # Check for mistakes
        dk0 = {
            k0: [
                k1 for k1, v1 in dlines0.items()
                if np.sum([(
                    v1['ion'] == v0['ion']
                    and v1['transition'] == v0['transition'],
                    v1['ion'] == v0['ion'] and v1['symbol'] == v0['symbol']
                )]) == 1
            ]
            for k0, v0 in dout.items()
        }
        dk0 = {k0: v0 for k0, v0 in dk0.items() if len(v0) > 0}
        if len(dk0) > 0:
            msg = (
                "\nPossible error in openadas detected,\n"
                + "the following lines have same ion and transition but "
                + "different symbol (typ0typ1-isoel):\n"
                + "\n".join([
                    "\t- {}: {}".format(k0, v0) for k0, v0 in dk0.items()
                ])
                + "\n\n  => There might be redundancy / errors in openadas"
            )
            warnings.warn(msg)

        dout = {
            k0: v0 for k0, v0 in dout.items()
            if not any([
                v1['ion'] == v0['ion'] and v1['transition'] == v0['transition']
                for v1 in dlines0.values()
            ])
        }

    # --------------------------
    # Get dict of unique sources

    lsource = sorted(set([v0['source'] for v0 in dout.values()]))
    dsource = {}
    if dsource0 is None:
        dsource = {
            'oa-adf15-{:02}'.format(ii): {'long': ss}
            for ii, ss in enumerate(lsource)
        }

    else:
        # Check against existing sources
        nmax = int(np.max([
            int(k0.split('-')[-1])
            for k0 in dsource0.keys() if 'oa-adf15' in k0
        ])) + 1
        for ii, ss in enumerate(lsource):
            lk0 = [k0 for k0, v0 in dsource0.items() if v0['long'] == ss]
            if len(lk0) == 0:
                k0 = 'oa-adf15-{:02}'.format(nmax)
                nmax += 1
            elif len(lk0) == 1:
                k0 = lk0[0]
            else:
                msg = (
                    "\nMultiple possible matches for source {}".format(ss)
                )
                raise Exception(msg)
            dsource[k0] = {'long': ss}

    # ----------------------------
    # Get dict of unique Te and ne

    # initialize dict and numbers
    dte, dne = {}, {}
    if dref0 is None:
        ite, ine = 0, 0
    else:
        ite = int(np.max([
            int(k0.split('-')[-1]) for k0 in dref0.keys()
            if 'Te-' in k0 and 'oa-adf15' in ddata0[k0]['source']
        ])) + 1
        ine = int(np.max([
            int(k0.split('-')[-1]) for k0 in dref0.keys()
            if 'ne-' in k0 and 'oa-adf15' in ddata0[k0]['source']
        ])) + 1

    # scan
    for k0, v0 in dout.items():

        # Get source
        sour = [
            k1 for k1, v1 in dsource.items() if v1['long'] == v0['source']
        ][0]

        # fill dte
        kte = [
            kk for kk, vv in dte.items()
            if v0['te'].shape == vv['data'].shape
            and np.allclose(v0['te'], vv['data'])
            # and sour == vv['source']
        ]
        normal = dref0 is None
        if normal is False:
            # Check vs existing Te
            lk0 = [
                k1 for k1, v1 in dref0.items()
                if v0['te'].shape == ddata0[k1]['data'].shape
                and np.allclose(v0['te'], ddata0[k1]['data'])
                # and ddata0[k1]['source'] == sour
            ]
            if len(lk0) == 0:
                normal = True
            elif len(lk0) == 1:
                keyte = lk0[0]
                # dte[keyte] = {
                    # 'data': ddata0[lk0[0]]['data'],
                    # 'units': v0['te_units'],
                    # 'source': sour,
                    # 'dim': 'temperature',
                    # 'quant': 'Te',
                    # 'name': 'Te',
                # }
            elif len(lk0) > 1:
                msg = (
                    "Multiple matches for dout[{}] in dref0:\n".format(k0)
                    + "\t- {}".format(lk0)
                )
                raise Exception(msg)

        if normal is True:
            if len(kte) == 0:
                keyte = 'Te-{:02}'.format(ite)
                dte[keyte] = {
                    'data': v0['te'],
                    'units': v0['te_units'],
                    'source': sour,
                    'dim': 'temperature',
                    'quant': 'Te',
                    'name': 'Te',
                }
                ite += 1
            elif len(kte) == 1:
                pass
            else:
                msg = (
                    "len(kte) != 1:\n"
                    + "\t- kte = {}\n".format(kte)
                )
                raise Exception(msg)

        dout[k0]['keyte'] = keyte

        # fill dne
        kne = [
            kk for kk, vv in dne.items()
            if v0['ne'].shape == vv['data'].shape
            and np.allclose(v0['ne'], vv['data'])
            # and sour == vv['source']
        ]
        normal = dref0 is None
        if normal is False:
            # Check vs existing ne
            lk0 = [
                k1 for k1, v1 in dref0.items()
                if v0['ne'].shape == ddata0[k1]['data'].shape
                and np.allclose(v0['ne'], ddata0[k1]['data'])
                # and ddata0[k1]['source'] == sour
            ]
            if len(lk0) == 0:
                normal = True
            elif len(lk0) == 1:
                keyne = lk0[0]
                # dne[keyne] = {
                    # 'data': ddata0[lk0[0]]['data'],
                    # 'units': v0['ne_units'],
                    # 'source': sour,
                    # 'dim': 'density',
                    # 'quant': 'ne',
                    # 'name': 'ne',
                # }
            elif len(lk0) > 1:
                msg = (
                    "Multiple matches for dout[{}] in dref0:\n".format(k0)
                    + "\t- {}".format(lk0)
                )
                raise Exception(msg)

        if normal is True:
            if len(kne) == 0:
                keyne = 'ne-{:02}'.format(ine)
                dne[keyne] = {
                    'data': v0['ne'],
                    'units': v0['ne_units'],
                    'source': sour,
                    'dim': 'density',
                    'quant': 'ne',
                    'name': 'ne',
                }
                ine += 1
            elif len(kne) == 1:
                pass
            else:
                msg = (
                    "len(kne) != 1:\n"
                    + "\t- kne = {}\n".format(kne)
                )
                raise Exception(msg)
        dout[k0]['keyne'] = keyne

    # ---------------
    # Get dict of pec

    dpec = {
        '{}-pec'.format(k0): {
            'data': v0['pec'], 'units': v0['pec_units'],
            'ref': (v0['keyne'], v0['keyte']),
            'source': [
                k1 for k1, v1 in dsource.items()
                if v1['long'] == v0['source']
            ][0],
            'dim': '<sigma v>',
            'quant': 'pec',
        }
        for k0, v0 in dout.items()
    }

    # --------
    # dlines

    inds = np.argsort([v0['lambda0'] for v0 in dout.values()])
    lk0 = np.array(list(dout.keys()), dtype=str)[inds]
    dlines = {
        k0: {
            'ion': dout[k0]['ion'],
            'source': [
                k1 for k1, v1 in dsource.items()
                if v1['long'] == dout[k0]['source']
            ][0],
            'lambda0': dout[k0]['lambda0'],
            'pec': '{}-pec'.format(k0),
            'symbol': dout[k0]['symbol'],
            'type': dout[k0]['type'],
            'transition': dout[k0]['transition'],
        }
        for k0 in lk0
    }

    # Get dict of unique ions
    lion = sorted(set([v0['ion'] for v0 in dout.values()]))

    return dne, dte, dpec, lion, dsource, dlines


# #############################################################################
# #############################################################################
#                       Main functions
# #############################################################################


def step03_read(
    adas_path,
    pec_as_func=None,
    **kwdargs,
):
    """ Read openadas-formatted files and return a dict with the data

    Povide the full adas file name
    The result is returned as a dict

    example
    -------
        >>> import tofu as tf
        >>> fn = '/adf11/scd74/scd74_ar.dat'
        >>> out = tf.openadas2tofu.step03_read(fn)
        >>> fn = '/adf15/pec40][ar/pec40][ar_ca][ar16.dat'
        >>> out = tf.openadas2tofu.step03_read(fn)
    """

    path_local = _get_PATH_LOCAL()

    # Check whether the local .tofu repo exists, if not recommend tofu-custom
    if path_local is None:
        path = os.path.join(os.path.expanduser('~'), '.tofu', 'openadas2tofu')
        msg = ("You do not seem to have a local ./tofu repository\n"
               + "tofu uses that local repository to store all user-specific "
               + "data and downloads\n"
               + "In particular, openadas files are downloaded and saved in:\n"
               + "\t{}\n".format(path)
               + "  => to set-up your local .tofu repo, run in a terminal:\n"
               + "\ttofu-custom")
        raise Exception(msg)

    # Determine whether adas_path is an absolute path or an adas full name
    if os.path.isfile(adas_path):
        pfe = adas_path
    else:
        # make sure adas_path is not understood as absolute local path
        if adas_path[0] == '/':
            adas_path = adas_path[1:]

        # Check file was downloaded locally
        pfe = os.path.join(path_local, adas_path)
        if not os.path.isfile(pfe):
            msg = ("Provided file does not seem to exist:\n"
                   + "\t{}\n".format(pfe)
                   + "  => Search it online with "
                   + "tofu.openadas2tofu.step01_search_online()\n"
                   + "  => Download it with "
                   + "tofu.openadas2tofu.step02_download()")
            raise FileNotFoundError(msg)

    lc = [ss for ss in _DTYPES.keys() if ss in pfe]
    if not len(lc) == 1:
        msg = ("File type could not be derived from absolute path:\n"
               + "\t- provided:  {}\n".format(pfe)
               + "\t- supported: {}".format(sorted(_DTYPES.keys())))
        raise Exception(msg)

    func = eval('_read_{}'.format(lc[0]))
    return func(pfe, **kwdargs)


def step03_read_all(
    element=None, charge=None, typ1=None, typ2=None,
    pec_as_func=None,
    format_for_DataStock=None,
    dsource0=None,
    dref0=None,
    ddata0=None,
    dlines0=None,
    verb=None, **kwdargs,
):
    """ Read all relevant openadas files for chosen typ1

    Please specify:
        - typ1:
            - 'adf11': ionisation / recombination data
            - 'adf15': pec data
        - element: the symbol of the element

    If typ1 = 'adf11', you can also provide typ2 to specify the coefficients:
        - 'scd': effective ionisation coefficients
        - 'acd': effective electron-impact recombination coefficients
        - 'ccd': effective hydrogen-impact recombination coefficients
        - 'plt': line power due to electron-impact excitation
        - 'prc': line power due to hydrogen-impact excitation
        - 'prb': rad. recombination and bremmstrahlung due to electron-impact

    If typ1 = 'adf15', you can optioanlly provide a min/max wavelength

    The result is returned as a dict

    examples
    --------
        >>> import tofu as tf
        >>> dout = tf.openadas2tofu.step03_read_all(element='ar', typ1='adf11')
        >>> dout = tf.openadas2tofu.step03_read_all(element='ar', typ1='adf15',
                                                    charge=16,
                                                    lambmin=3.94e-10,
                                                    lambmax=4.e-10)
    """

    # --------------------
    # Check whether the local .tofu repo exists, if not recommend tofu-custom
    path_local = _get_PATH_LOCAL()
    if path_local is None:
        path = os.path.join(os.path.expanduser('~'), '.tofu', 'openadas2tofu')
        msg = ("You do not seem to have a local ./tofu repository\n"
               + "tofu uses that local repository to store all user-specific "
               + "data and downloads\n"
               + "In particular, openadas files are downloaded and saved in:\n"
               + "\t{}\n".format(path)
               + "  => to set-up your local .tofu repo, run in a terminal:\n"
               + "\ttofu-custom")
        raise Exception(msg)

    # --------------------
    # Check / format input
    if typ1 is None:
        typ1 = 'adf15'
    if not isinstance(typ1, str) or typ1.lower() not in _DTYPES.keys():
        msg = ("Please choose a valid adas file type:\n"
               + "\t- allowed:  {}\n".format(_DTYPES.keys())
               + "\t- provided: {}".format(typ1))
        raise Exception(msg)
    typ1 = typ1.lower()
    if typ1 == 'adf11' and typ2 is None:
        typ2 = _DTYPES[typ1]
        fd = os.listdir(os.path.join(path_local, typ1))
        typ2 = [sorted([ss for ss in fd if tt in ss])[-1] for tt in typ2]
    if isinstance(typ2, str):
        typ2 = [typ2]
    if (_DTYPES[typ1] is not None
        and (not isinstance(typ2, list)
             or not all([any([s1 in ss for s1 in _DTYPES[typ1]])
                         for ss in typ2]))):
        msg = ("typ2 must be a list of valid openadas file types for typ1:\n"
               + "\t- provided:         {}\n".format(typ2)
               + "\t- available for {}: {}".format(typ1, _DTYPES[typ1]))
        raise Exception(msg)

    # --------------------
    # Get elevant directory
    # Level 1: Type
    path = _get_subdir_from_pattern(path_local, typ1, mult='err')[0]

    # --------------------
    # element
    c0 = (
        element is None
        or isinstance(element, str)
        or (
            isinstance(element, list)
            and all([isinstance(ee, str) for ee in element])
        )
        or (
            isinstance(element, tuple)
            and all([isinstance(ee, str) for ee in element])
        )
    )
    if not c0:
        msg = "Please choose an element!"
        raise Exception(msg)

    if element is None or isinstance(element, tuple):
        el = _get_available_elements_from_path(path=path, typ1=typ1)
        if element is None:
            element = el
        else:
            element = tuple([ee.lower() for ee in element])
            element = [ee for ee in el if ee not in element]
    if isinstance(element, str):
        element = [element]
    element = [ee.lower() for ee in element]

    # --------------------
    # charge
    if charge is not None:
        c0 = (
            isinstance(charge, int)
            or (
                isinstance(charge, list)
                and all([isinstance(cc, int) for cc in charge])
            )
            or (
                isinstance(charge, tuple)
                and all([isinstance(cc, int) for cc in charge])
            )
        )
        if not c0:
            msg = ("Arg charge must be a int or list (e.g.: 16 or [0])\n"
                   + "\t- provided: {}".format(charge))
            raise Exception(msg)
        if isinstance(charge, int):
            charge = [charge]
        if isinstance(charge, list):
            charge = ['{}.dat'.format(cc) for cc in charge]
        elif isinstance(charge, tuple):
            charge = tuple(['{}.dat'.format(cc) for cc in charge])

    if format_for_DataStock is None:
        format_for_DataStock = False

    if verb is None:
        verb = True

    # --------------------
    # Get list of relevant directories per element

    # Level 2: element or typ2
    if typ1 == 'adf11':
        lpath = [
            _get_subdir_from_pattern(path, tt, mult='err')[0]
            for tt in typ2
        ]
    elif typ1 == 'adf15':
        lpath = np.concatenate([
            _get_subdir_from_pattern(
                path, ['pec', '][{}'.format(ee)], mult=True,
            )
            for ee in element
        ]).tolist()

    # --------------------
    # Get list of relevant files pfe
    lpfe = list(itt.chain.from_iterable([[
        os.path.join(path, ff) for ff in os.listdir(path)
        if (
            os.path.isfile(os.path.join(path, ff))
            and ff[-4:] == '.dat'
            and any(['][{}'.format(ee) in ff for ee in element])
        )]
        for path in lpath
    ]))

    if typ1 == 'adf15':
        kwdargs['pec_as_func'] = pec_as_func
        if isinstance(charge, list):
            lpfe = [
                ff for ff in lpfe
                if any([
                    ''.join([ss for ss in ff.split('][')[-1] if ss.isdigit()])
                    + '.dat'
                    == cc
                    for cc in charge
                ])
            ]
        elif isinstance(charge, tuple):
            lpfe = [
                ff for ff in lpfe
                if not any([
                    ''.join([ss for ss in ff.split('][')[-1] if ss.isdigit()])
                    + '.dat'
                    == cc
                    for cc in charge
                ])
            ]

    # --------------------
    # Extract data from each file
    func = eval('_read_{}'.format(typ1))
    dout = {}
    for pfe in lpfe:
        if verb is True:
            msg = "\tLoading data from {}".format(pfe)
            print(msg)
        dout = func(pfe, dout=dout, **kwdargs)

    if typ1 == 'adf15' and format_for_DataStock is True:
        return _format_for_DataStock_adf15(
            dout,
            dsource0=dsource0,
            dref0=dref0,
            ddata0=ddata0,
            dlines0=dlines0,
        )
    else:
        return dout


# #############################################################################
#                      Specialized functions for ADF 11
# #############################################################################


def _read_adf11(pfe, deg=None, dout=None):
    if deg is None:
        deg = _DEG
    if dout is None:
        dout = {}

    # Get second order file type
    typ1 = [vv for vv in _DTYPES['adf11'] if vv in pfe]
    if len(typ1) != 1:
        msg = ("Second order file type could not be inferred from file name!\n"
               + "\t- available: {}\n".format(_DTYPES['adf11'])
               + "\t- provided: {}".format(pfe))
        raise Exception(msg)
    typ1 = typ1[0]

    # Get element
    elem = pfe[:-4].split('_')[1]
    comline = '-'*60
    comline2 = 'C'+comline

    if typ1 in ['acd', 'ccd', 'scd', 'plt', 'prb']:

        # read blocks
        with open(pfe) as search:
            for ii, line in enumerate(search):

                if comline2 in line:
                    break

                # Get atomic number (transitions) stored in this file
                if ii == 0:
                    lstr = line.split('/')
                    lin = [ss for ss in lstr[0].strip().split(' ')
                           if ss.strip() != '']
                    lc = [
                        len(lin) == 5 and all([ss.isdigit() for ss in lin]),
                        elem.upper() in lstr[1],
                        # 'ADF11' in lstr[-1],
                    ]
                    if not all(lc):
                        msg = ("File header format seems to have changed!\n"
                               + "\t- pfe: {}\n".format(pfe)
                               + "\t- lc = {}\n".format(lc)
                               + "\t- lstr = {}".format(lstr))
                        raise Exception(msg)
                    Z, nne, nte, q0, qend = map(int, lin)
                    nelog10 = np.array([])
                    telog10 = np.array([])
                    in_ne = True
                    continue

                if comline in line:
                    continue

                # Get nelog10
                if in_ne:
                    li = [ss for ss in line.strip().split(' ')
                          if ss.strip() != '']
                    nelog10 = np.append(nelog10, np.array(li, dtype=float))
                    if nelog10.size == nne:
                        in_ne = False
                        in_te = True

                # Get telog10
                elif in_te is True:
                    li = [ss for ss in line.strip().split(' ')
                          if ss.strip() != '']
                    telog10 = np.append(telog10, np.array(li, dtype=float))
                    if telog10.size == nte:
                        in_te = False
                        in_ion = True

                # Get ion block
                elif (in_ion is True and 'Z1=' in line
                      and ('------/' in line and 'DATE=' in line)):
                    nion = int(
                        line[line.index('Z1=')+len('Z1='):].split('/')[0])
                    if typ1 in ['scd', 'plt']:
                        charge = nion - 1
                    else:
                        charge = nion
                    coefslog10 = np.array([])
                elif in_ion is True and charge is not None:
                    li = [ss for ss in line.strip().split(' ')
                          if ss.strip() != '']
                    coefslog10 = np.append(coefslog10,
                                           np.array(li, dtype=float))
                    if coefslog10.size == nne*nte:
                        key = '{}{}'.format(elem, charge)
                        tkv = [('element', elem), ('Z', Z), ('charge', charge)]
                        if key in dout.keys():
                            assert all([dout[key][ss] == vv for ss, vv in tkv])
                        else:
                            dout[key] = {ss: vv for ss, vv in tkv}
                        if typ1 == 'scd':
                            # nelog10+6 to convert /cm3 -> /m3
                            # coefslog10-6 to convert cm3/s -> m3/s
                            func = scpRectSpl(nelog10+6, telog10,
                                              coefslog10.reshape((nne, nte))-6,
                                              kx=deg, ky=deg)
                            dout[key]['ionis'] = {'func': func,
                                                  'type': 'log10_nete',
                                                  'units': 'log10(m3/s)',
                                                  'source': pfe}
                        elif typ1 == 'acd':
                            # nelog10+6 to convert /cm3 -> /m3
                            # coefslog10-6 to convert cm3/s -> m3/s
                            func = scpRectSpl(nelog10+6, telog10,
                                              coefslog10.reshape((nne, nte))-6,
                                              kx=deg, ky=deg)
                            dout[key]['recomb'] = {'func': func,
                                                   'type': 'log10_nete',
                                                   'units': 'log10(m3/s)',
                                                   'source': pfe}
                        elif typ1 == 'ccd':
                            # nelog10+6 to convert /cm3 -> /m3
                            # coefslog10-6 to convert cm3/s -> m3/s
                            func = scpRectSpl(nelog10+6, telog10,
                                              coefslog10.reshape((nne, nte))-6,
                                              kx=deg, ky=deg)
                            dout[key]['recomb_ce'] = {'func': func,
                                                      'type': 'log10_nete',
                                                      'units': 'log10(m3/s)',
                                                      'source': pfe}
                        elif typ1 == 'plt':
                            # nelog10+6 to convert /cm3 -> /m3
                            # coefslog10+6 to convert W.cm3 -> W.m3
                            func = scpRectSpl(nelog10+6, telog10,
                                              coefslog10.reshape((nne, nte))+6,
                                              kx=deg, ky=deg)
                            dout[key]['rad_bb'] = {'func': func,
                                                   'type': 'log10_nete',
                                                   'units': 'log10(W.m3)',
                                                   'source': pfe}
                        elif typ1 == 'prb':
                            # nelog10+6 to convert /cm3 -> /m3
                            # coefslog10+6 to convert W.cm3 -> W.m3
                            func = scpRectSpl(nelog10+6, telog10,
                                              coefslog10.reshape((nne, nte))+6,
                                              kx=deg, ky=deg)
                            dout[key]['rad_fffb'] = {'func': func,
                                                     'type': 'log10_nete',
                                                     'units': 'log10(W.m3)',
                                                     'source': pfe}
                        if nion == Z:
                            break

    return dout


# #############################################################################
#                      Specialized functions for ADF 15
# #############################################################################


def _get_adf15_key(elem, charge, isoel, typ0, typ1):
    return '{}{}_{}_oa_{}_{}'.format(elem, charge, isoel, typ0, typ1)


def _read_adf15(
    pfe,
    dout=None,
    lambmin=None,
    lambmax=None,
    pec_as_func=None,
    deg=None,
):
    """
    Here lambmin and lambmax are provided in m
    """

    if deg is None:
        deg = _DEG
    if pec_as_func is None:
        pec_as_func = _PECASFUNC
    if dout is None:
        dout = {}

    # Get summary of transitions
    flagblock = '/isel ='
    flag0 = 'superstage partition information'

    # Get file markers from name (elem, charge, typ0, typ1)
    typ0, typ1, elemq = pfe.split('][')[1:]
    ind = re.search(r'\d', elemq).start()
    elem = elemq[:ind].title()
    charge = int(elemq[ind:-4])
    assert elem.lower() in typ0[:2]
    assert elem.lower() == typ1.split('_')[0]
    typ0 = typ0[len(elem)+1:]
    typ1 = typ1.split('_')[1]

    # Extract data from file
    nlines, nblock = None, 0
    in_ne, in_te, in_pec, in_tab, itab = False, False, False, False, np.inf
    skip = False
    with open(pfe) as search:
        for ii, line in enumerate(search):

            # Get number of lines (transitions) stored in this file
            if ii == 0:
                lstr = line.split('/')
                nlines = int(lstr[0].replace(' ', ''))
                continue

            # Get info about the transition being scanned (block)
            if flagblock in line and 'C' not in line and nblock < nlines:
                lstr = [kk for kk in line.rstrip().split(' ') if len(kk) > 0]
                lamb = float(lstr[0])*1.e-10
                isoel = nblock + 1
                nblock += 1
                c0 = ((lambmin is not None and lamb < lambmin)
                      or (lambmax is not None and lamb > lambmax))
                if c0:
                    skip = True
                    continue
                skip = False
                nne, nte = int(lstr[1]), int(lstr[2])
                typ = [ss[ss.index('type=')+len('type='):ss.index('/ispb')]
                       for ss in lstr[3:] if 'type=' in ss]
                assert len(typ) == 1
                # To be updated : proper reading from line
                in_ne = True
                ne = np.array([])
                te = np.array([])
                pec = np.full((nne*nte,), np.nan)
                ind = 0
                # il = 0
                continue

            if 'root partition information' in line and skip is True:
                skip = False

            # Check lamb is ok
            if skip is True:
                continue

            # Get ne for the transition being scanned (block)
            if in_ne is True:
                ne = np.append(ne,
                               np.array(line.rstrip().strip().split(' '),
                                        dtype=float))
                if ne.size == nne:
                    in_ne = False
                    in_te = True

            # Get te for the transition being scanned (block)
            elif in_te is True:
                te = np.append(te,
                               np.array(line.rstrip().strip().split(' '),
                                        dtype=float))
                if te.size == nte:
                    in_te = False
                    in_pec = True

            # Get pec for the transition being scanned (block)
            elif in_pec is True:
                data = np.array(line.rstrip().strip().split(' '),
                                dtype=float)
                pec[ind:ind+data.size] = data
                # pec[il, :] = data
                ind += data.size
                # il += 1
                if ind == pec.size:
                    in_pec = False
                    key = _get_adf15_key(elem, charge, isoel, typ0, typ1)
                    # PEC reshaping and conversion to cm3/s -> m3/s
                    pec = pec.reshape((nne, nte)) * 1e-6
                    # log(ne)+6 to convert /cm3 -> /m3
                    ne = ne*1e6

                    if pec_as_func is True:
                        pec_rec = scpRectSpl(
                            np.log(ne),
                            np.log(te),
                            np.log(pec),
                            kx=deg,
                            ky=deg,
                        )

                        def pec(Te=None, ne=None, pec_rec=pec_rec):
                            return np.exp(pec_rec(np.log(ne), np.log(Te)))

                    dout[key] = {
                        'lambda0': lamb,
                        'ion': '{}{}+'.format(elem, charge),
                        'charge': charge,
                        'element': elem,
                        'symbol': '{}{}-{}'.format(typ0, typ1, isoel),
                        'source': pfe,
                        'type': typ[0],
                        'ne': ne,
                        'ne_units': '/m3',
                        'te': te,
                        'te_units': 'eV',
                        'pec': pec,
                        'pec_type': 'f(ne, Te)',
                        'pec_units': 'm3/s',
                    }

            # Get transitions from table at the end
            if 'photon emissivity atomic transitions' in line:
                itab = ii + 6
            if ii == itab:
                in_tab = True
            if in_tab is True:
                lstr = [kk for kk in line.rstrip().split(' ') if len(kk) > 0]
                isoel = int(lstr[1])
                lamb = float(lstr[2])*1.e-10
                key = _get_adf15_key(elem, charge, isoel, typ0, typ1)
                c0 = ((lambmin is None or lambmin < lamb)
                      and (lambmax is None or lambmax > lamb))
                if c0 and key not in dout.keys():
                    msg = ("Inconsistency in file {}:\n".format(pfe)
                           + "\t- line should be present".format(key))
                    raise Exception(msg)
                if key in dout.keys():
                    if dout[key]['lambda0'] != lamb:
                        msg = "Inconsistency in file {}".format(pfe)
                        raise Exception(msg)
                    c0 = (dout[key]['type'] not in lstr
                          or lstr.index(dout[key]['type']) < 4)
                    if c0:
                        msg = ("Inconsistency in table, type not found:\n"
                               + "\t- expected: {}\n".format(dout[key]['type'])
                               + "\t- line: {}".format(line))
                        raise Exception(msg)
                    trans = lstr[3:lstr.index(dout[key]['type'])]
                    dout[key]['transition'] = ''.join(trans)
                if isoel == nlines:
                    in_tab = False
    assert all(['transition' in vv.keys() for vv in dout.values()])
    return dout
