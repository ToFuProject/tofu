

import numpy as np


_OPENADAS_ONLINE = True


# #############################################################################
# #############################################################################
#                       from openadas
# #############################################################################


def from_openadas(
    lambmin=None,
    lambmax=None,
    element=None,
    charge=None,
    online=None,
    update=None,
    create_custom=None,
    dsource0=None,
    dref0=None,
    ddata0=None,
    dlines0=None,
    group_lines=None,
):
    """
    Load lines and pec from openadas, either:
        - online = True:  directly from the website
        - online = False: from pre-downloaded files in ~/.tofu/openadas/

    Provide wavelengths in m

    Example:
    --------
            >>> import tofu as tf
            >>> lines_mo = tf.data.SpectralLines.from_openadas(
                element='Mo',
                lambmin=3.94e-10,
                lambmax=4e-10,
            )

    """

    # Preliminary import and checks
    from ..openadas2tofu import _requests
    from ..openadas2tofu import _read_files

    if online is None:
        online = _OPENADAS_ONLINE

    # Load from online if relevant
    if online is True:
        try:
            out = _requests.step01_search_online_by_wavelengthA(
                lambmin=lambmin*1e10,
                lambmax=lambmax*1e10,
                element=element,
                charge=charge,
                verb=False,
                returnas=np.ndarray,
                resolveby='file',
            )
            lf = sorted(set([oo[0] for oo in out]))
            out = _requests.step02_download_all(
                files=lf,
                update=update,
                create_custom=create_custom,
                verb=False,
            )
        except Exception as err:
            msg = (
                """
                {}

                For some reason data could not be downloaded from openadas
                    => see error message above
                    => maybe check your internet connection?
                """.format(err)
            )
            raise Exception(msg)

    # Load for local files
    dne, dte, dpec, lion, dsource, dlines = _read_files.step03_read_all(
        lambmin=lambmin,
        lambmax=lambmax,
        element=element,
        charge=charge,
        pec_as_func=False,
        format_for_DataCollection=True,
        dsource0=dsource0,
        dref0=dref0,
        ddata0=ddata0,
        dlines0=dlines0,
        verb=False,
    )

    # # dgroup
    # dgroup = ['Te', 'ne']

    # dref - Te + ne
    dref = dte
    dref.update(dne)

    # ddata - pec
    ddata = dpec

    # dobj (lines, ion, source)
    dobj = {
        group_lines: dlines,
        'ion': {k0: {} for k0 in lion},
        'source': dsource,
    }
    return ddata, dref, dobj


# #############################################################################
# #############################################################################
#                       from nist
# #############################################################################


def _from_nist(
    lambmin=None,
    lambmax=None,
    element=None,
    charge=None,
    ion=None,
    wav_observed=None,
    wav_calculated=None,
    transitions_allowed=None,
    transitions_forbidden=None,
    cache_from=None,
    cache_info=None,
    verb=None,
    create_custom=None,
    dsource0=None,
    dlines0=None,
    group_lines=None,
):
    """
    Load lines from nist, either:
        - cache_from = False:  directly from the website
        - cache_from = True: from pre-downloaded files in ~/.tofu/nist/

    Provide wavelengths in m

    Example:
    --------
            >>> import tofu as tf
            >>> lines_mo = tf.data.SpectralLines.from_nist(
                element='Mo',
                lambmin=3.94e-10,
                lambmax=4e-10,
            )

    """

    # Preliminary import and checks
    from ..nist2tofu import _requests

    if verb is None:
        verb = False
    if cache_info is None:
        cache_info = False

    # Load from online if relevant
    dlines, dsources = _requests.step01_search_online_by_wavelengthA(
        element=element,
        charge=charge,
        ion=ion,
        lambmin=lambmin*1e10,
        lambmax=lambmax*1e10,
        wav_observed=wav_observed,
        wav_calculated=wav_calculated,
        transitions_allowed=transitions_allowed,
        transitions_forbidden=transitions_forbidden,
        info_ref=True,
        info_conf=True,
        info_term=True,
        info_J=True,
        info_g=True,
        cache_from=cache_from,
        cache_info=cache_info,
        return_dout=True,
        return_dsources=True,
        verb=verb,
        create_custom=create_custom,
        format_for_DataCollection=True,
        dsource0=dsource0,
        dlines0=dlines0,
    )

    # ions
    lion = sorted(set([dlines[k0]['ion'] for k0 in dlines.keys()]))

    # dobj (lines)
    dobj = {
        group_lines: dlines,
        'ion': {k0: {} for k0 in lion},
        'source': dsources,
    }
    return dobj


# #############################################################################
# #############################################################################
#                       from module
# #############################################################################


def _check_extract_dict_from_mod(mod, k0):
    lk1 = [
        k0, k0.upper(),
        '_'+k0, '_'+k0.upper(),
        '_d'+k0, '_D'+k0.upper(),
        'd'+k0, 'D'+k0.upper(),
        k0+'s', k0.upper()+'S'
        '_d'+k0+'s', '_D'+k0.upper()+'S',
        'd'+k0+'s', 'D'+k0.upper()+'S',
    ]
    lk1 = [k1 for k1 in lk1 if hasattr(mod, k1)]
    if len(lk1) > 1:
        msg = "Ambiguous attributes: {}".format(lk1)
        raise Exception(msg)
    elif len(lk1) == 0:
        return

    if hasattr(mod, lk1[0]):
        return getattr(mod, lk1[0])
    else:
        return

def from_module(pfe=None):

    # Check input
    c0 = (
        os.path.isfile(pfe)
        and pfe[-3:] == '.py'
    )
    if not c0:
        msg = (
            "\nProvided Path-File-Extension (pfe) not valid!\n"
            + "\t- expected: absolute path to python module\n"
            + "\t- provided: {}".format(pfe)
        )
        raise Exception(msg)
    pfe = os.path.abspath(pfe)

    # Load module
    path, fid = os.path.split(pfe)
    import importlib.util
    spec = importlib.util.spec_from_file_location(fid[:-3], pfe)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # extract source, transition, ion, element
    dobj = {}
    for k0 in ['source', 'transition', 'ion', 'element']:
        dd = _check_extract_dict_from_mod(mod, k0)
        if dd is not None:
            dobj[k0] = dd

    # add ion
    if 'ion' not in dobj.keys():
        lions = np.array([
                v0['ion'] for k0, v0 in mod.dlines.items()
                if 'ion' in v0.keys()
        ]).ravel()
        if len(lions) > 0:
            dobj['ion'] = {
                k0: {'ion': k0} for k0 in lions
            }
        else:
            lIONS = np.array([
                    v0['ION'] for k0, v0 in mod.dlines.items()
                    if 'ION' in v0.keys()
            ]).ravel()
            if len(lIONS) > 0:
                dobj['ION'] = {
                    k0: {'ION': k0} for k0 in lIONS
                }

    # extract lines
    dobj['lines'] = mod.dlines
    return dobj
