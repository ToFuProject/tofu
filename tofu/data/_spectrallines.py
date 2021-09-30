

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings


from ._DataCollection_class import DataCollection
from . import _comp_spectrallines
from . import _DataCollection_comp
from . import _DataCollection_plot


__all__ = ['SpectralLines', 'TimeTraces']


_OPENADAS_ONLINE = True

_GROUP_LINES = 'lines'
_GROUP_NE = 'ne'
_GROUP_TE = 'Te'
_UNITS_LAMBDA0 = 'm'


#############################################
#############################################
#       Spectral Lines
#############################################


class SpectralLines(DataCollection):

    _ddef = {
        'Id': {'include': ['Mod', 'Cls', 'Name', 'version']},
        'params': {
            'lambda0': (float, 0.),
            'source': (str, 'unknown'),
            'transition':    (str, 'unknown'),
            'element':  (str, 'unknown'),
            'charge':  (int, 0),
            'ion':  (str, 'unknown'),
            'symbol':   (str, 'unknown'),
        },
    }
    _forced_group = [_GROUP_NE, _GROUP_TE]
    _data_none = True

    _show_in_summary_core = ['shape', 'ref', 'group']
    _show_in_summary = 'all'

    _grouplines = _GROUP_LINES
    _groupne = _GROUP_NE
    _groupte = _GROUP_TE

    _units_lambda0 = _UNITS_LAMBDA0

    def add_line(
        self,
        key=None,
        lambda0=None,
        pec=None,
        source=None,
        transition=None,
        ion=None,
        symbol=None,
        **kwdargs,
    ):
        """ Add a spectral line by key and rest wavelength, optionally with

        """
        self.add_obj(
            which='lines',
            key=key,
            lambda0=lambda0,
            pec=pec,
            source=source,
            transition=transition,
            ion=ion,
            symbol=symbol,
            **kwdargs,
        )

    def add_pec(self, key=None, pec=None, ref=None):
        pass

    def remove_line(
        self,
        key=None,
        ion=None,
        source=None,
        lambda0=None,
    ):
        # Check inputs
        lc = [
            key is not None,
            ion is not None or source is not None or lambda0 is not None,
        ]
        raise NotImplementedError

    # -----------------
    # from openadas
    # ------------------

    @classmethod
    def _from_openadas(
        cls,
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
        grouplines=None,
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
        if grouplines is None:
            grouplines = cls._grouplines
        else:
            cls._grouplines = grouplines

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

        # dref_static
        dref_static = {
            'ion': {k0: {} for k0 in lion},
            'source': dsource,
        }

        # dobj (lines)
        dobj = {
            grouplines: dlines,
        }
        return ddata, dref, dref_static, dobj

    @classmethod
    def from_openadas(
        cls,
        lambmin=None,
        lambmax=None,
        element=None,
        charge=None,
        online=None,
        update=None,
        create_custom=None,
        grouplines=None,
    ):
        """
        Load lines and pec from openadas, either:
            - online = True:  directly from the website
            - online = False: from pre-downloaded files in ~/.tofu/openadas/
        """
        ddata, dref, dref_static, dobj = cls._from_openadas(
            lambmin=lambmin,
            lambmax=lambmax,
            element=element,
            charge=charge,
            online=online,
            update=update,
            create_custom=create_custom,
            grouplines=grouplines,
        )
        return cls(ddata=ddata, dref=dref, dref_static=dref_static, dobj=dobj)

    def add_from_openadas(
        self,
        lambmin=None,
        lambmax=None,
        element=None,
        charge=None,
        online=None,
        update=None,
        create_custom=None,
    ):
        """
        Load and add lines and pec from openadas, either:
            - online = True:  directly from the website
            - online = False: from pre-downloaded files in ~/.tofu/openadas/
        """
        ddata, dref, dref_static, dobj = self._from_openadas(
            lambmin=lambmin,
            lambmax=lambmax,
            element=element,
            charge=charge,
            online=online,
            update=update,
            create_custom=create_custom,
            dsource0=self._dref_static.get('source'),
            dref0=self._dref,
            ddata0=self._ddata,
            dlines0=self._dobj.get('lines'),
        )
        self.update(ddata=ddata, dref=dref, dref_static=dref_static, dobj=dobj)

    # -----------------
    # from nist
    # ------------------

    @classmethod
    def _from_nist(
        cls,
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
        grouplines=None,
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

        if grouplines is None:
            grouplines = cls._grouplines
        else:
            cls._grouplines = grouplines
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

        # dref_static
        lion = sorted(set([dlines[k0]['ion'] for k0 in dlines.keys()]))

        dref_static = {
            'ion': {k0: {} for k0 in lion},
            'source': dsources,
        }

        # dobj (lines)
        dobj = {
            grouplines: dlines,
        }
        return dref_static, dobj

    @classmethod
    def from_nist(
        cls,
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
        grouplines=None,
    ):
        """
        Load lines and pec from openadas, either:
            - online = True:  directly from the website
            - online = False: from pre-downloaded files in ~/.tofu/openadas/
        """
        dref_static, dobj = cls._from_nist(
            lambmin=lambmin,
            lambmax=lambmax,
            element=element,
            charge=charge,
            ion=ion,
            wav_observed=wav_observed,
            wav_calculated=wav_calculated,
            transitions_allowed=transitions_allowed,
            transitions_forbidden=transitions_forbidden,
            cache_from=cache_from,
            cache_info=cache_info,
            verb=verb,
            create_custom=create_custom,
            grouplines=grouplines,
        )
        return cls(dref_static=dref_static, dobj=dobj)

    def add_from_nist(
        self,
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
        grouplines=None,
    ):
        """
        Load and add lines and pec from openadas, either:
            - online = True:  directly from the website
            - online = False: from pre-downloaded files in ~/.tofu/openadas/
        """
        dref_static, dobj = self._from_nist(
            lambmin=lambmin,
            lambmax=lambmax,
            element=element,
            charge=charge,
            ion=ion,
            wav_observed=wav_observed,
            wav_calculated=wav_calculated,
            transitions_allowed=transitions_allowed,
            transitions_forbidden=transitions_forbidden,
            cache_from=cache_from,
            cache_info=cache_info,
            verb=verb,
            create_custom=create_custom,
            dsource0=self._dref_static.get('source'),
            dlines0=self._dobj.get('lines'),
            grouplines=grouplines,
        )
        self.update(dref_static=dref_static, dobj=dobj)

    # -----------------
    # from file (.py)
    # ------------------

    @staticmethod
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

    @classmethod
    def from_module(cls, pfe=None):

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

        # extract dref_static
        dref_static = {}
        for k0 in ['source', 'transition', 'ion', 'element']:
            dd = cls._check_extract_dict_from_mod(mod, k0)
            if dd is not None:
                dref_static[k0] = dd

        # add ion
        if 'ion' not in dref_static.keys():
            lions = np.array([
                    v0['ion'] for k0, v0 in mod.dlines.items()
                    if 'ion' in v0.keys()
            ]).ravel()
            if len(lions) > 0:
                dref_static['ion'] = {
                    k0: {'ion': k0} for k0 in lions
                }
            else:
                lIONS = np.array([
                        v0['ION'] for k0, v0 in mod.dlines.items()
                        if 'ION' in v0.keys()
                ]).ravel()
                if len(lIONS) > 0:
                    dref_static['ION'] = {
                        k0: {'ION': k0} for k0 in lIONS
                    }

        # extract lines
        dobj = {
            'lines': mod.dlines
        }

        # Create collection
        out = cls(dref_static=dref_static, dobj=dobj)

        # Replace ION by ion if relevant
        c0 = (
            'ion' in out.dref_static.keys()
            and 'ion' not in out.get_lparam(which='lines')
            and 'ION' in out.get_lparam(which='lines')
        )
        if c0:
            for k0, v0 in out._dobj['lines'].items():
                ion = [
                    k1 for k1, v1 in out._dref_static['ion'].items()
                    if out._dobj['lines'][k0]['ION'] == v1['ION']
                ][0]
                out._dobj['lines'][k0]['ion'] = ion
                del out._dobj['lines'][k0]['ION']
        return out

    # -----------------
    # summary
    # ------------------

    # -----------------
    # conversion wavelength - energy - frequency
    # ------------------

    def convert_lines(self, units=None, key=None, ind=None, returnas=None):
        """ Convert wavelength (m) to other units or other quantities

        Avalaible units:
            wavelength: km, m, mm, um, nm, pm, A
            energy:     J, eV, keV, MeV, GeV
            frequency:  Hz, kHz, MHz, GHz, THz

        Return the result as a np.ndarray (returnas = 'data')

        Can also just return the conversion coef if returnas='coef'
        In that case, a bool is also returned indicating whether the result is
        the proportional to the inverse of lambda0::
            - False: data = coef * lambda0
            - True: data = coef / lambda0
        """
        if units is None:
            units = self._units_lambda0
        if returnas is None:
            returnas = dict

        lok = [dict, np.ndarray, 'data', 'coef']
        if returnas not in lok:
            msg = (
                "Arg returnas must be in:\n"
                + "\t- {}\n".format(lok)
                + "\t- provided: {}".format(returnas)
            )
            raise Exception(msg)
        if returnas in [dict, np.ndarray, 'data']:
            returnas2 = 'data'
        else:
            returnas2 = 'coef'

        key = self._ind_tofrom_key(
            which=self._grouplines, key=key, ind=ind, returnas=str,
        )
        lamb_in = self.get_param(
            'lambda0', key=key, returnas=np.ndarray,
        )['lambda0']
        out = self.convert_spectral(
            data=lamb_in, units_in='m', units_out=units, returnas=returnas2,
        )
        if returnas is dict:
            out = {k0: out[ii] for ii, k0 in enumerate(key)}
        return out

    # -----------------
    # PEC interpolation
    # ------------------

    def calc_pec(
        self,
        key=None,
        ind=None,
        ne=None,
        Te=None,
        deg=None,
        grid=None,
    ):
        """ Compute the pec (<sigma v>) by interpolation for chosen lines

        Assumes Maxwellian electron distribution

        Provide ne and Te and 1d np.ndarrays

        if grid=False:
            - ne is a (n,) 1d array
            - Te is a (n,) 1d array
          => the result is a dict of (n,) 1d array

        if grid=True:
            - ne is a (n1,) 1d array
            - Te is a (n2,) 1d array
          => the result is a dict of (n1, n2) 2d array
        """

        # Check keys
        key = self._ind_tofrom_key(
            which=self._grouplines, key=key, ind=ind, returnas=str,
        )
        dlines = self._dobj[self._grouplines]

        if deg is None:
            deg = 2

        # Check data conformity
        lg = (self._groupne, self._groupte)
        lc = [
            k0 for k0 in key
            if (
                dlines[k0].get('pec') is None
                or self._ddata[dlines[k0]['pec']]['group'] != lg
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
        ltype = [int, float, np.integer, np.floating]
        dnTe = {'ne': ne, 'Te': Te}
        for k0, v0 in dnTe.items():
            if type(v0) in ltype:
                dnTe[k0] = np.r_[v0]
            if isinstance(dnTe[k0], list) or isinstance(dnTe[k0], tuple):
                dnTe[k0] = np.array([dnTe[k0]])
            if not (isinstance(dnTe[k0], np.ndarray) and dnTe[k0].ndim == 1):
                msg = (
                    "Arg {} should be a 1d np.ndarray!".format(k0)
                )
                raise Exception(msg)

        # Interpolate
        dout = {}
        derr = {}
        for k0 in key:
            try:
                ne0 = [
                    kk for kk in self._ddata[dlines[k0]['pec']]['ref']
                    if self._ddata[kk]['group'] == (self._groupne,)
                ][0]
                ne0 = self._ddata[ne0]['data']
                Te0 = [
                    kk for kk in self._ddata[dlines[k0]['pec']]['ref']
                    if self._ddata[kk]['group'] == (self._groupte,)
                ][0]
                Te0 = self._ddata[Te0]['data']

                dout[k0] = _comp_spectrallines._interp_pec(
                    ne0=ne0,
                    Te0=Te0,
                    pec0=self._ddata[dlines[k0]['pec']]['data'],
                    ne=dnTe['ne'],
                    Te=dnTe['Te'],
                    deg=deg,
                    grid=grid,
                )
            except Exception as err:
                derr[k0] = str(err)

        if len(derr) > 0:
            msg = (
                "The pec could not be interpolated for the following lines:\n"
                + "\n".join([
                    '\t- {} : {}'.format(k0, v0) for k0, v0 in derr.items()
                ])
            )
            raise Exception(msg)

        return dout

    def calc_intensity(
        self,
        key=None,
        ind=None,
        ne=None,
        Te=None,
        concentration=None,
        deg=None,
        grid=None,
    ):
        """ Compute the lines intensities by pec interpolation for chosen lines

        Assumes Maxwellian electron distribution

        Provide ne and Te and 1d np.ndarrays

        Provide concentration as:
            - a np.ndarray (same concentration assumed for all lines)
            - a dict of {key: np.ndarray}

        if grid=False:
            - ne is a (n,) 1d array
            - Te is a (n,) 1d array
            - concentration is a (dict of) (n,) 1d array(s)
          => the result is a dict of (n1, n2) 2d array

        if grid=True:
            - ne is a (n1,) 1d array
            - Te is a (n2,) 1d array
            - concentration is a (dict of) (n1, n2) 2d array(s)
          => the result is a dict of (n1, n2) 2d array


        """

        # check inputs
        if grid is None:
            grid = ne.size != Te.size

        # Check keys
        key = self._ind_tofrom_key(
            which=self._grouplines, key=key, ind=ind, returnas=str,
        )

        if isinstance(concentration, np.ndarray):
            concentration = {k0: concentration for k0 in key}
        c0 = (
            isinstance(concentration, dict)
            and all([
                k0 in key
                and isinstance(cc, np.ndarray)
                and (
                    (grid is False and cc.shape == ne.shape == Te.shape)
                    or
                    (grid is True and cc.shape == (ne.size, Te.size))
                )
                and np.all((cc > 0.) & (cc <= 1.))
                for k0, cc in concentration.items()
            ])
        )
        if not c0:
            shape = ne.shape if grid is False else (ne.size, Te.size)
            msg = (
                "Arg concentration is non-conform:\n"
                + "\t- Expected: dict of {} arrays in [0, 1]\n".format(shape)
                + "\t- Provided: {}".format(concentration)
            )
            raise Exception(msg)

        # interpolate pec
        dpec = self.calc_pec(
            key=key,
            ind=ind,
            ne=ne,
            Te=Te,
            grid=grid,
            deg=deg,
        )

        # ne for broadcasting
        if grid is True:
            neb = ne[:, None]
        else:
            neb = ne

        # Derive intensity
        dint = {
            k0: v0*neb**2*concentration[k0] for k0, v0 in dpec.items()
        }
        return dint

    # -----------------
    # plotting
    # ------------------

    def plot(
        self,
        key=None,
        ind=None,
        ax=None,
        sortby=None,
        param_txt=None,
        ymin=None,
        ymax=None,
        ls=None,
        lw=None,
        fontsize=None,
        side=None,
        dcolor=None,
        fraction=None,
        figsize=None,
        dmargin=None,
        wintit=None,
        tit=None,
    ):
        """ plot rest wavelengths as vertical lines """
        if param_txt is None:
            param_txt = 'symbol'

        return super()._plot_axvlines(
            which='lines',
            key=key,
            param_x='lambda0',
            param_txt=param_txt,
            sortby=sortby,
            sortby_def='ion',
            sortby_lok=['ion', 'ION', 'source'],
            ax=ax, ymin=ymin, ymax=ymax,
            ls=ls, lw=lw, fontsize=fontsize,
            side=side, dcolor=dcolor, fraction=fraction,
            figsize=figsize, dmargin=dmargin,
            wintit=wintit, tit=tit,
        )

    def plot_pec_single(
        self,
        key=None,
        ind=None,
        ne=None,
        Te=None,
        concentration=None,
        deg=None,
        grid=None,
        ax=None,
        sortby=None,
        param_txt=None,
        ymin=None,
        ymax=None,
        ls=None,
        lw=None,
        fontsize=None,
        side=None,
        dcolor=None,
        fraction=None,
        figsize=None,
        dmargin=None,
        wintit=None,
        tit=None,
    ):

        # Check input
        if param_txt is None:
            param_txt = 'symbol'

        # Check ne, Te
        ltypes = [int, float, np.integer, np.floating]
        dnTe = {'ne': ne, 'Te': Te}
        single = all([
            type(v0) in ltypes or len(v0) == 1 for v0 in dnTe.values()
        ])
        if not single:
            msg = ("Arg ne and Te must be floats!")
            raise Exception(msg)

        # Get dpec
        dpec = self.calc_pec(
            key=key,
            ind=ind,
            ne=ne,
            Te=Te,
            deg=deg,
            grid=grid,
        )
        key = list(dpec.keys())

        ne = float(ne)
        Te = float(Te)
        tit = (
            r'$n_e$' + '= {} '.format(ne) + r'$/m^3$'
            + r' -  $T_e$ = ' + '{} keV'.format(Te/1000.)
        )

        pmax = np.max([np.log10(v0) for v0 in dpec.values()])
        pmin = np.min([np.log10(v0) for v0 in dpec.values()])
        dsize = {
            k0: (np.log10(v0)-pmin)/(pmax-pmin)*19 + 1
            for k0, v0 in dpec.items()
        }

        sortby_lok = ['ion', 'ION', 'source']
        lk0 = [k0 for k0 in sortby_lok if k0 in self._dref_static.keys()]
        if len(lk0) > 0:
            sortby_def = lk0[0]
        else:
            sortby_def = None

        return super()._plot_axvlines(
            which='lines',
            key=key,
            param_x='lambda0',
            param_txt=param_txt,
            sortby=sortby,
            sortby_def=sortby_def,
            sortby_lok=sortby_lok,
            dsize=dsize,
            ax=ax, ymin=ymin, ymax=ymax,
            ls=ls, lw=lw, fontsize=fontsize,
            side=side, dcolor=dcolor, fraction=fraction,
            figsize=figsize, dmargin=dmargin,
            wintit=wintit, tit=tit,
        )

    def plot_pec(
        self,
        key=None,
        ind=None,
        ne=None,
        Te=None,
        norder=None,
        ne_scale=None,
        Te_scale=None,
        param_txt=None,
        param_color=None,
        deg=None,
        dax=None,
        proj=None,
        ymin=None,
        ymax=None,
        ls=None,
        lw=None,
        fontsize=None,
        side=None,
        dcolor=None,
        fraction=None,
        figsize=None,
        dmargin=None,
        dtit=None,
        tit=None,
        wintit=None,
    ):

        # Check input
        if param_txt is None:
            param_txt = 'symbol'
        if param_color is None:
            param_color = 'ion'
        if norder is None:
            norder = 0

        if ne_scale is None:
            ne_scale = 'log'
        if Te_scale is None:
            Te_scale = 'linear'

        # Check ne, Te
        ltypes = [int, float, np.integer, np.floating]
        dnTe = {
            'ne': type(ne) in ltypes or len(ne) == 1,
            'Te': type(Te) in ltypes or len(Te) == 1,
        }
        if all([v0 for v0 in dnTe.values()]):
            msg = (
                "For a single point in (ne, Te) space, use plot_pec_singe()"
            )
            raise Exception(msg)
        elif dnTe['ne']:
            ne = np.r_[ne].ravel()
            ne = np.full((Te.size), ne[0])
        elif dnTe['Te']:
            Te = np.r_[Te].ravel()
            Te = np.full((ne.size), Te[0])

        if len(ne) != len(Te):
            msg = (
                "Please provide ne and Te as vectors of same size!"
            )
            raise Exception(msg)

        # Get dpec
        dpec = self.calc_pec(
            key=key,
            ind=ind,
            ne=ne,
            Te=Te,
            deg=deg,
            grid=False,
        )
        damp = {k0: {'data': v0} for k0, v0 in dpec.items()}

        # Create grid
        ne_grid = _DataCollection_comp._get_grid1d(
            ne, scale=ne_scale, npts=ne.size*2, nptsmin=3,
        )
        Te_grid = _DataCollection_comp._get_grid1d(
            Te, scale=Te_scale, npts=Te.size*2, nptsmin=3,
        )

        dpec_grid = self.calc_pec(
            key=key,
            ind=ind,
            ne=ne_grid,
            Te=Te_grid,
            deg=deg,
            grid=True,
        )

        # Get dcolor
        lcol = plt.rcParams['axes.prop_cycle'].by_key()['color']
        dcolor = {}
        if param_color != 'key':
            lion = [self._dobj['lines'][k0][param_color] for k0 in dpec.keys()]
            for ii, k0 in enumerate(set(lion)):
                dcolor[k0] = mcolors.to_rgb(lcol[ii % len(lcol)])
                lk1 = [
                    k2 for k2 in dpec.keys()
                    if self._dobj['lines'][k2][param_color] == k0
                ]
                for k1 in lk1:
                    damp[k1]['color'] = k0
        else:
            for ii, k0 in enumerate(dpec.keys()):
                dcolor[k0] = mcolors.to_rgb(lcol[ii % len(lcoil)])
                damp[k0]['color'] = k0

        # Create image
        im_data = np.full((ne_grid.size, Te_grid.size), np.nan)
        im = np.full((ne_grid.size, Te_grid.size, 4), np.nan)
        dom_val = np.concatenate(
            [v0[None, :, :] for v0 in dpec_grid.values()],
            axis=0,
        )

        if norder == 0:
            im_ind = np.nanargmax(dom_val, axis=0)
        else:
            im_ind = np.argsort(dom_val, axis=0)[-norder, :, :]

        for ii in np.unique(im_ind):
            ind = im_ind == ii
            im_data[ind] = dom_val[ii, ind]

        pmin = np.nanmin(np.log10(im_data))
        pmax = np.nanmax(np.log10(im_data))

        for ii, k0 in enumerate(dpec_grid.keys()):
            if ii in np.unique(im_ind):
                ind = im_ind == ii
                im[ind, :-1] = dcolor[damp[k0]['color']]
                im[ind, -1] = (
                    (np.log10(im_data[ind])-pmin)/(pmax-pmin)*0.9 + 0.1
                )
        extent = (ne_grid.min(), ne_grid.max(), Te_grid.min(), Te_grid.max())

        if tit is None:
            tit = 'spectral lines PEC interpolations'
        if dtit is None:
            dtit = {'map': 'norder = {}'.format(norder)}

        return _DataCollection_plot.plot_dominance_map(
            din=self._dobj['lines'], im=im, extent=extent,
            xval=ne, yval=Te, damp=damp,
            x_scale=ne_scale, y_scale=Te_scale, amp_scale='log',
            param_txt='symbol',
            dcolor=dcolor,
            dax=dax, proj=proj,
            figsize=figsize, dmargin=dmargin,
            wintit=wintit, tit=tit, dtit=dtit,
        )


#############################################
#############################################
#       Time traces
#############################################


# TBC
class TimeTraces(DataCollection):
    """ A generic class for handling multiple time traces """

    _forced_group = 'time'
    _dallowed_params = {
        'time': {
            'origin': (str, 'unknown'),
            'dim':    (str, 'time'),
            'quant':  (str, 't'),
            'name':   (str, 't'),
            'units':  (str, 's')},
    }
    _plot_vignettes = False

    def fit(self, ind=None, key=None,
            Type='staircase', func=None,
            plot=True, fs=None, ax=None, draw=True, **kwdargs):
        """  Fit the times traces with a model

        Typically try to fit plateaux and ramps i.e.: Type = 'staircase')
        Return a dictionary of the fitted parameters, ordered by data key

        """

        dout = self._fit_one_dim(ind=ind, key=key, group=self._forced_group,
                                 Type=Type, func=func, **kwdargs)
        if plot:
            kh = _DataCollection_plot.plot_fit_1d(self, dout)
        return dout

    def add_plateaux(self, verb=False):
        dout = self.fit(
            ind=ind, key=key, group=group, Type='staircase',
        )

        # Make Pandas Dataframe attribute
        self.plateaux = None
        if verb:
            msg = ""

    def plot(self, **kwdargs):
        return self._plot_timetraces(**kwdargs)

    def plot_incremental(self, key=None, ind=None,
                         plateaux=True, connect=True):
        return

    def plot_plateau_validate(self, key=None, ind=None):
        return
