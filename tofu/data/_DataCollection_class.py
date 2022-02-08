

import matplotlib.pyplot as plt


from ._DataCollection_Base_class import DataCollectionBase
from . import _DataCollection_plot


__all__ = ['DataCollection']    # , 'TimeTraceCollection']


class DataCollection(DataCollectionBase):
    """ Handles matplotlib interactivity """

    _LPAXES = ['ax', 'type']

    def add_axes(self, key=None, ax=None, type=None, **kwdargs):
        super().add_obj(which='axes', key=key, ax=ax, type=type, **kwdargs)
        if 'canvas' not in self._dobj.keys():
            self.add_canvas(handle=ax.figure.canvas)
        else:
            lisin = [
                k0 for k0, v0 in self._dobj['canvas'].items()
                if v0['handle'] == ax.figure.canvas
            ]
            if len(lisin) == 0:
                self.add_canvas(handle=ax.figure.canvas)

    def add_canvas(self, key=None, handle=None):
        interactive = (
            hasattr(handle, 'toolbar')
            and handle.toolbar is not None
        )
        self.add_obj(
            which='canvas',
            key=key,
            handle=handle,
            interactive=interactive,
        )

    def add_mobile(
        self,
        key=None,
        handle=None,
        ref=None,
        data=None,
        **kwdargs,
    ):
        super().add_obj(
            which='mobile',
            key=key,
            handle=handle,
            ref=ref,
            data=data,
            **kwdargs,
        )

    @property
    def dax(self):
        return self.dobj.get('axes', {})

    # ----------------------------
    # Ensure connectivity possible
    # ----------------------------

    def _warn_ifnotInteractive(self):
        warn = False

        c0 = (
            len(self._dobj.get('axes', {})) > 0
            and len(self._dobj.get('canvas', {})) > 0
        )
        if c0:
            dcout = {
                k0: v0['handle'].__class__.__name__
                for k0, v0 in self._dobj['canvas'].items()
                if not v0['interactive']
            }
            if len(lcout) > 0:
                lstr = [f'\t- {k0}: {v0}' for k0, v0 in dcout.items()]
                msg = (
                    "Non-interactive backends identified (prefer Qt5Agg):\n"
                    "\t- backend : {plt.get_backend()}\n"
                    "\t- canvas  :\n{'\n'.join(lstr)}"
                )
                warn = True
        else:
            msg = ("No available axes / canvas for interactivity")
            warn = True

        # raise warning
        if warn:
            warnings.warn(msg)

        return warn

    # ----------------------
    # Connect / disconnect
    # ----------------------

    def connect(self):
        if self._warn_ifnotInteractive():
            return
        for k0, v0 in sef._dobj['canvas'].items():
            keyp = v0['handle'].mpl_connect('key_press_event', self.onkeypress)
            keyr = v0['handle'].mpl_connect('key_release_event', self.onkeypress)
            butp = v0['handle'].mpl_connect('button_press_event', self.mouseclic)
            res = v0['handle'].mpl_connect('resize_event', self.resize)
            #butr = self.can.mpl_connect('button_release_event', self.mouserelease)
            #if not plt.get_backend() == "agg":
            v0['handle'].manager.toolbar.release = self.mouserelease

            self._dobj['canvas'][k0]['cid'] = {
                'keyp': keyp,
                'keyr': keyr,
                'butp': butp,
                'res': res,
                # 'butr': butr,
            }

    def disconnect(self):
        if self._warn_ifnotInteractive():
            return
        for k0, v0 in sef._dobj['canvas'].items():
            for k1, v1 in v0['cid'].items():
                v0['handle'].mpl_disconnect(v1)
            v0['handle'].manager.toolbar.release = lambda event: None

    # ----------------------
    # Interactivity handling
    # ----------------------

    # -------------------
    # Generic plotting
    # -------------------

    def plot_as_array(
        self,
        key=None,
        ind=None,
        vmin=None,
        vmax=None,
        cmap=None,
        aspect=None,
        dax=None,
        dmargin=None,
        fs=None,
        dcolorbar=None,
        dleg=None,
        connect=None,
    ):
        """ Plot the desired 2d data array as a matrix """
        return _DataCollection_plot.plot_as_array(
            coll=self,
            key=key,
            ind=ind,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            aspect=aspect,
            dax=dax,
            dmargin=dmargin,
            fs=fs,
            dcolorbar=dcolorbar,
            dleg=dleg,
            connect=connect,
        )

    def _plot_timetraces(self, ntmax=1,
                         key=None, ind=None, Name=None,
                         color=None, ls=None, marker=None, ax=None,
                         axgrid=None, fs=None, dmargin=None,
                         legend=None, draw=None, connect=None, lib=None):
        plotcoll = self.to_PlotCollection(ind=ind, key=key,
                                          Name=Name, dnmax={})
        return _DataCollection_plot.plot_DataColl(
            plotcoll,
            color=color, ls=ls, marker=marker, ax=ax,
            axgrid=axgrid, fs=fs, dmargin=dmargin,
            draw=draw, legend=legend,
            connect=connect, lib=lib,
        )

    def _plot_axvlines(
        self,
        which=None,
        key=None,
        ind=None,
        param_x=None,
        param_txt=None,
        sortby=None,
        sortby_def=None,
        sortby_lok=None,
        ax=None,
        ymin=None,
        ymax=None,
        ls=None,
        lw=None,
        fontsize=None,
        side=None,
        dcolor=None,
        dsize=None,
        fraction=None,
        figsize=None,
        dmargin=None,
        wintit=None,
        tit=None,
    ):
        """ plot rest wavelengths as vertical lines """

        # Check inputs
        which, dd = self.__check_which(
            which=which, return_dict=True,
        )
        key = self._ind_tofrom_key(which=which, key=key, ind=ind, returnas=str)

        if sortby is None:
            sortby = sortby_def
        if sortby not in sortby_lok:
            msg = (
                """
                For plotting, sorting can be done only by:
                {}

                You provided:
                {}
                """.format(sortby_lok, sortby)
            )
            raise Exception(msg)

        return _DataCollection_plot.plot_axvline(
            din=dd,
            key=key,
            param_x='lambda0',
            param_txt='symbol',
            sortby=sortby, dsize=dsize,
            ax=ax, ymin=ymin, ymax=ymax,
            ls=ls, lw=lw, fontsize=fontsize,
            side=side, dcolor=dcolor,
            fraction=fraction,
            figsize=figsize, dmargin=dmargin,
            wintit=wintit, tit=tit,
        )

