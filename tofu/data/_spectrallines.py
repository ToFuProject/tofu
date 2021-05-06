

import numpy as np


from ._core_new import DataCollection
from . import _comp_spectrallines


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

    _ddef = {'Id': {'include': ['Mod', 'Cls', 'Name', 'version']},
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

    # def update(self, **kwdargs):
        # super().update(**kwdargs)

        # # check data 
        # lc = [
            # k0 for k0, v0 in self._ddata.items()
            # if v0.get('data') is not None
            # and not (
                # isinstance(v0['data'], np.ndarray)
                # and v0['data'].ndim <= 2
            # )
        # ]
        # if len(lc) > 0:
            # msg = (
                # """
                # The data provided for a line must be a tabulation of its pec

                # The following lines have non-conform data:
                # {}
                # """.format(lc)
            # )
            # raise Exception(msg)

    def add_line(
        self,
        key=None,
        lambda0=None,
        pec=None,
        ref=None,
        source=None,
        transition=None,
        ion=None,
        element=None,
        charge=None,
        symbol=None,
        **kwdargs,
    ):
        """ Add a spectral line by key and rest wavelength, optionally with 

        """
        self.add_obj(
            key=key,
            lambda0=lambda0,
            pec=pec,
            ref=ref,
            source=source,
            transition=transition,
            ion=ion,
            element=element,
            charge=charge,
            symbol=symbol,
            **kwdargs,
        )

    def add_pec(self, key=None, pec=None, ref=None):
        pass

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
    # summary
    # ------------------


    # -----------------
    # conversion wavelength - energy - frequency
    # ------------------

    def convert_lines(self, units=None, key=None, ind=None, returnas=None):
        """ Convert wavelength (m) to other units or other quantities (energy)

        Avalaible units:
            wavelength: km, m, mm, um, nm, pm, A
            energy:     J, eV, keV, MeV, GeV
            frequency:  Hz, kHz, MHz, GHz, THz

        Can also just return the conversion coef if returnas='coef'
        """
        if units is None:
            units = self._units_lambda0

        key = self._ind_tofrom_key(
            which=self._grouplines, key=key, ind=ind, returnas=str,
        )
        lamb_in = self.get_param(
            'lambda0', key=key, returnas=np.ndarray,
        )['lambda0']
        return self.convert_spectral(
            data=lamb_in, units_in='m', units_out=units, returnas=returnas,
        )

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
                    ne=ne,
                    Te=Te,
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
            sortby_lok=['ion', 'source'],
            ax=ax, ymin=ymin, ymax=ymax,
            ls=ls, lw=lw, fontsize=fontsize,
            side=side, dcolor=dcolor, fraction=fraction,
            figsize=figsize, dmargin=dmargin,
            wintit=wintit, tit=tit,
        )


    def plot_pec(self):
        raise NotImplementedError




#############################################
#############################################
#       Time traces
#############################################


# TBC
class TimeTraces(DataCollection):
    """ A generic class for handling multiple time traces """

    _forced_group = 'time'
    _dallowed_params = {'time':{'origin': (str, 'unknown'),
                                'dim':    (str, 'time'),
                                'quant':  (str, 't'),
                                'name':   (str, 't'),
                                'units':  (str, 's')}}
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
            kh = _plot_new.plot_fit_1d(self, dout)
        return dout

    def add_plateaux(self, verb=False):
        dout = self.fit(ind=ind, key=key, group=group,
                       Type='staircase')

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
