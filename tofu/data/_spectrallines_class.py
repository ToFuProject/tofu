

import os
import sys
import warnings


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import datastock as ds


from . import _spectrallines_checks
from . import _spectrallines_compute
from . import _spectrallines_plot
# from ._DataCollection_class import DataCollection
# from . import _comp_spectrallines
# from . import _DataCollection_comp
# from . import _DataCollection_plot_as_array


__all__ = ['SpectralLines']


_WHICH_LINES = 'lines'
_GROUP_NE = 'ne'
_GROUP_TE = 'Te'
_UNITS_LAMBDA0 = 'm'


#############################################
#############################################
#       Spectral Lines
#############################################


class SpectralLines(ds.DataStock):

    _ddef = {
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

    _show_in_summary_core = ['shape', 'ref']
    _show_in_summary = 'all'

    _which_lines = _WHICH_LINES
    _quant_ne = _GROUP_NE
    _quant_Te = _GROUP_TE

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

    # -----------------
    # from openadas
    # ------------------

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
    ):
        """
        Load lines and pec from openadas, either:
            - online = True:  directly from the website
            - online = False: from pre-downloaded files in ~/.tofu/openadas/
        """
        ddata, dref, dobj = _spectrallines_compute.from_openadas(
            lambmin=lambmin,
            lambmax=lambmax,
            element=element,
            charge=charge,
            online=online,
            update=update,
            create_custom=create_custom,
            group_lines=cls._which_lines,
        )
        return cls(ddata=ddata, dref=dref, dobj=dobj)

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
        ddata, dref, dobj = _spectrallines_compute.from_openadas(
            lambmin=lambmin,
            lambmax=lambmax,
            element=element,
            charge=charge,
            online=online,
            update=update,
            create_custom=create_custom,
            dsource0=self._dobj.get('source'),
            dref0=self._dref,
            ddata0=self._ddata,
            dlines0=self._dobj.get('lines'),
            group_lines=cls._which_lines,
        )
        self.update(ddata=ddata, dref=dref, dobj=dobj)

    # -----------------
    # from nist
    # ------------------

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
    ):
        """
        Load lines and pec from openadas, either:
            - online = True:  directly from the website
            - online = False: from pre-downloaded files in ~/.tofu/openadas/
        """
        dobj = _spectrallines_compute._from_nist(
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
            group_lines=cls._which_lines,
        )
        return cls(dobj=dobj)

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
    ):
        """
        Load and add lines and pec from openadas, either:
            - online = True:  directly from the website
            - online = False: from pre-downloaded files in ~/.tofu/openadas/
        """
        dobj = _spectrallines_compute._from_nist(
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
            dsource0=self._dobj.get('source'),
            dlines0=self._dobj.get('lines'),
            group_lines=self._which_lines,
        )
        self.update(dobj=dobj)

    # -----------------
    # from file (.py)
    # ------------------

    @classmethod
    def from_module(cls, pfe=None):

        dobj = _spectrallines_compute.from_module(pfe=pfe)

        # Create collection
        out = cls(dobj=dobj)

        # Replace ION by ion if relevant
        c0 = (
            'ion' in out._dobj.keys()
            and 'ion' not in out.get_lparam(which='lines')
            and 'ION' in out.get_lparam(which='lines')
        )
        if c0:
            for k0, v0 in out._dobj['lines'].items():
                ion = [
                    k1 for k1, v1 in out._dobj['ion'].items()
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

        # get keys of desired lines
        key = self._ind_tofrom_key(
            which=self._which_lines, key=key, ind=ind, returnas=str,
        )

        # get wavelength in m
        lamb_in = self.get_param(
            which=self._which_lines, param='lambda0',
            key=key, returnas=np.ndarray,
        )['lambda0']

        # conversion
        out = _spectrallines_compute.convert_spectral(
            data_in=lamb_in,
            units_in='m',
            units_out=units,
            returnas=returnas2,
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
        return_params=None,
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
            which=self._which_lines, key=key, ind=ind, returnas=str,
        )
        dlines = self._dobj[self._which_lines]

        key, dnTe, return_params = _spectrallines_checks._check_compute_pec(
            # check keys
            key=key,
            dlines=dlines,
            ddata=self._ddata,
            _quant_ne=self._quant_ne,
            _quant_Te=self._quant_Te,
            # check ne, Te
            ne=ne,
            Te=Te,
            return_params=return_params,
        )
        keypec = [f'{k0}-pec' for k0 in key]

        # group lines per common ref
        lref = set([self._ddata[k0]['ref'] for k0 in keypec])
        dref = {
            k0: [k1 for k1 in keypec if self._ddata[k1]['ref'] == k0]
            for k0 in lref
        }

        # Interpolate
        for ii, (k0, v0) in enumerate(dref.items()):
            douti, dparami = self.interpolate(
                # interpolation base
                keys=v0,
                ref_keys=None,
                ref_quant=[self._quant_ne, self._quant_Te],
                # interpolation pts
                pts_axis0=dnTe['ne'],
                pts_axis1=dnTe['Te'],
                # parameters
                deg=deg,
                deriv=0,
                grid=grid,
                log_log=True,
                return_params=True,
            )

            # update dict
            if ii == 0:
                dout = douti
                dparam = dparami
            else:
                dout.update(**douti)
                dparam['keys'] += dparami['keys']
                dparam['ref_keys'] += dparami['ref_keys']

        # -------
        # return

        if return_params is True:
            dparam['key'] = dparam['keys']
            del dparam['keys']
            dparam['ne'] = dparam['pts_axis0']
            dparam['Te'] = dparam['pts_axis1']
            del dparam['pts_axis0'], dparam['pts_axis1'], dparam['pts_axis2']
            return dout, dparam
        else:
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
        Assumes concentration = nz / ne

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

        # Check keys
        key = self._ind_tofrom_key(
            which=self._which_lines, key=key, ind=ind, returnas=str,
        )


        # interpolate pec
        dout, dparam = self.calc_pec(
            key=key,
            ind=ind,
            ne=ne,
            Te=Te,
            grid=grid,
            deg=deg,
            return_params=True,
        )

        # check concentrations
        concentration = _spectrallines_checks._check_compute_intensity(
            key=[k0[:-4] for k0 in dparam['key']],
            concentration=concentration,
            shape=dparam['ne'].shape,
        )

        # Derive intensity
        for k0, v0 in dout.items():
            dout[k0] = v0*dparam['ne']**2*concentration[k0[:-4]]

        return dout

    # -----------------
    # plotting
    # ------------------

    def plot_spectral_lines(
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
        dsize=None,
        dcolor=None,
        fraction=None,
        figsize=None,
        dmargin=None,
        wintit=None,
        tit=None,
    ):
        """ plot rest wavelengths as vertical lines """

        # Check inputs
        key = self._ind_tofrom_key(
            which=self._which_lines, key=key, ind=ind, returnas=str,
        )

        return _spectrallines_plot.plot_axvlines(
            din=self._dobj[self._which_lines],
            key=key,
            param_x=param_x,
            param_txt=param_txt,
            sortby=sortby,
            dsize=dsize,
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
        param_x=None,
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
        """ Same as plot_spectral_lines() with extra scatter plot with circles

        The circles' diameters depend on the pec value for each line

        Requires:
            - Te = scalar (eV)
            - ne = scalar (/m3)

        """

        # ------------
        # Check ne, Te

        ltypes = [int, float, np.integer, np.floating]
        dnTe = {'ne': ne, 'Te': Te}
        single = all([
            type(v0) in ltypes or len(v0) == 1 for v0 in dnTe.values()
        ])
        if not single:
            msg = ("Arg ne and Te must be floats!")
            raise Exception(msg)

        # --------
        # Get dpec
        dpec = self.calc_pec(
            key=key,
            ind=ind,
            ne=ne,
            Te=Te,
            deg=deg,
            grid=False,
            return_params=False,
        )
        key = [k0[:-4] for k0 in dpec.keys()]

        ne = float(ne)
        Te = float(Te)
        tit = (
            r'$n_e$' + f'= {ne} ' + r'$/m^3$'
            + r' -  $T_e$ = ' + f'{Te/1000.} keV'
        )

        pmax = np.max([np.log10(v0) for v0 in dpec.values()])
        pmin = np.min([np.log10(v0) for v0 in dpec.values()])
        dsize = {
            k0[:-4]: (np.log10(v0) - pmin) / (pmax - pmin)*19 + 1
            for k0, v0 in dpec.items()
        }

        return _spectrallines_plot.plot_axvlines(
            din=self._dobj[self._which_lines],
            key=key,
            param_x=param_x,
            param_txt=param_txt,
            sortby=sortby,
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

        # -----------
        # Check input

        # Check ne, Te
        lc = [np.isscalar(ne) or len(ne) == 1, np.isscalar(Te) or len(Te) == 1]
        if all(lc):
            msg = "For a single (ne, Te) space, use plot_pec_singe()"
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

        # get dpec for grid
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

        return _spectrallines_plot.plot_dominance_map(
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
# class TimeTraces(DataCollection):
    # """ A generic class for handling multiple time traces """

    # _forced_group = 'time'
    # _dallowed_params = {
        # 'time': {
            # 'origin': (str, 'unknown'),
            # 'dim':    (str, 'time'),
            # 'quant':  (str, 't'),
            # 'name':   (str, 't'),
            # 'units':  (str, 's')},
    # }
    # _plot_vignettes = False

    # def fit(self, ind=None, key=None,
            # Type='staircase', func=None,
            # plot=True, fs=None, ax=None, draw=True, **kwdargs):
        # """  Fit the times traces with a model

        # Typically try to fit plateaux and ramps i.e.: Type = 'staircase')
        # Return a dictionary of the fitted parameters, ordered by data key

        # """

        # dout = self._fit_one_dim(ind=ind, key=key, group=self._forced_group,
                                 # Type=Type, func=func, **kwdargs)
        # if plot:
            # kh = _DataCollection_plot.plot_fit_1d(self, dout)
        # return dout

    # def add_plateaux(self, verb=False):
        # dout = self.fit(
            # ind=ind, key=key, group=group, Type='staircase',
        # )

        # # Make Pandas Dataframe attribute
        # self.plateaux = None
        # if verb:
            # msg = ""

    # def plot(self, **kwdargs):
        # return self._plot_timetraces(**kwdargs)

    # def plot_incremental(self, key=None, ind=None,
                         # plateaux=True, connect=True):
        # return

    # def plot_plateau_validate(self, key=None, ind=None):
        # return
