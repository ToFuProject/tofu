

import numpy as np


from ._core_new import DataCollection
from . import _plot_spectrallines


__all__ = ['SpectralLines', 'TimeTraces']


#############################################
#############################################
#       Spectral Lines
#############################################


class SpectralLines(DataCollection):

    _ddef = {'Id': {'include': ['Mod', 'Cls', 'Name', 'version']},
             'params': {
                 'lambda0': (float, 0.),
                 'origin': (str, 'unknown'),
                 'transition':    (str, 'unknown'),
                 'element':  (str, 'unknown'),
                 'charge':  (int, 0),
                 'ion':  (str, 'unknown'),
                 'symbol':   (str, 'unknown'),
                 },
            }
    _forced_group = ['Te', 'ne']
    _data_none = True

    _show_in_summary_core = ['shape', 'ref', 'group']
    _show_in_summary = [
        'data', 'lambda0', 'symbol', 'ion', 'transition', 'origin'
    ]

    def update(self, ddata=None, dref=None, dgroup=None):
        super().update(ddata=ddata, dref=dref, dgroup=dgroup)

        # check data 
        lc = [
            k0 for k0, v0 in self._ddata.items()
            if v0.get('data') is not None
            and not (
                isinstance(v0['data'], np.ndarray)
                and v0['data'].ndim == 2
            )
        ]
        if len(lc) > 0:
            msg = (
                """
                The data provided for a line must be a tabulation of its pec

                The following lines have non-conform data:
                {}
                """.format(lc)
            )
            raise Exception(msg)

    def add_line(
        self,
        key=None,
        lambda0=None,
        pec=None,
        ref=None,
        origin=None,
        transition=None,
        ion=None,
        element=None,
        charge=None,
        symbol=None,
        **kwdargs,
    ):
        """ Add a spectral line by key and rest wavelength, optionally with 

        """
        self.add_data(
            key=key,
            lambda0=lambda0,
            data=pec,
            ref=ref,
            origin=origin,
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
    # summary
    # ------------------

    def get_summary(
        self,
        show=None, show_core=None,
        sep='  ', line='-', just='l',
        table_sep=None, verb=True, return_=False,
    ):

        # -----------------------
        # Build for groups
        col0 = ['group name', 'nb. ref', 'nb. data']
        ar0 = [(k0,
                len(self._dgroup[k0]['lref']),
                len(self._dgroup[k0]['ldata']))
               for k0 in self._dgroup.keys()]

        # -----------------------
        # Build for refs
        col1 = ['ref key', 'group', 'size', 'nb. data']
        ar1 = [(k0,
                self._dref[k0]['group'],
                str(self._dref[k0]['size']),
                len(self._dref[k0]['ldata']))
               for k0 in self._dref.keys()]

        # -----------------------
        # Build for ions
        ions = sorted(set([
            v0['ion'] for v0 in self._ddata.values()
            if v0.get('ion') is not None
        ]))
        col2 = ['ions', 'nb. lines', 'lambda0 min', 'lambda0 max']
        ar2 = [(
            ion,
            np.sum([v0.get('ion') == ion for v0 in self._ddata.values()]),
            np.min([
                v0['lambda0'] for v0 in self._ddata.values()
                if v0.get('ion') == ion
            ]),
            np.max([
                v0['lambda0'] for v0 in self._ddata.values()
                if v0.get('ion') == ion
            ]),
        ) for ion in ions]

        # -----------------------
        # Build for ddata
        col3 = ['key']
        if show_core is None:
            show_core = self._show_in_summary_core
        if isinstance(show_core, str):
            show_core = [show_core]
        lp = self.lparam
        # lkcore = ['shape', 'group', 'ref']
        col3 += [ss for ss in show_core if ss in lp]

        if show is None:
            show = self._show_in_summary
        if show == 'all':
            col3 += [pp for pp in lp if pp not in col3]
        else:
            if isinstance(show, str):
                show = [show]
            show = [ss for ss in show if ss in lp]
            col3 += [pp for pp in show if pp not in col3]

        ar3 = []
        for k0 in self._ddata.keys():
            lu = [k0] + [str(self._ddata[k0][cc]) for cc in col3[1:]]
            ar3.append(lu)

        return self._get_summary(
            [ar0, ar1, ar2, ar3], [col0, col1, col2, col3],
            sep=sep, line=line, table_sep=table_sep,
            verb=verb, return_=return_)

    # -----------------
    # plotting
    # ------------------

    def plot(
        self,
        ax=None,
        sortby=None,
        ymin=None,
        ymax=None,
        ls=None,
        lw=None,
        fontsize=None,
        side=None,
        dcolor=None,
        fraction=None,
        **kwdargs,
    ):
        """ plot rest wavelengths as vertical lines """

        key = self.select(**kwdargs, returnas=str)
        if sortby is None:
            sortby = 'ion'
        lok = ['ion', 'element']
        if sortby not in lok:
            msg = (
                """
                For plotting, sorting can be done only by:
                {}

                You provided:
                {}
                """.format(lok, param)
            )
            raise Exception(msg)
        return _plot_spectrallines.plot_axvline(
            ddata=self._ddata, key=key, sortby=sortby,
            ax=ax, ymin=ymin, ymax=ymax,
            ls=ls, lw=lw, fontsize=fontsize,
            side=side, dcolor=dcolor, fraction=fraction,
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
