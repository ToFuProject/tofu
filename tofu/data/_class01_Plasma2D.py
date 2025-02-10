# -*- coding: utf-8 -*-


# Built-in
import copy


# tofu
# from tofu import __version__ as __version__
from ._class00_Config import Config as Previous
from . import _class01_compute as _compute
from . import _class01_load_equilibrium as _load_equilibrium


__all__ = ['Plasma2D']


_WHICH_MESH = 'mesh'
_QUANT_R = 'R'
_QUANT_Z = 'Z'


# #############################################################################
# #############################################################################
#                           Plasma2D
# #############################################################################


class Plasma2D(Previous):

    _ddef = copy.deepcopy(Previous._ddef)

    # _show_in_summary_core = ['shape', 'ref', 'group']
    _dshow = dict(Previous._dshow)

    # _quant_R = _QUANT_R
    # _quant_Z = _QUANT_Z

    # -------------------
    # units conversione
    # -------------------

    def convert_units_spectral(
        self,
        data=None,
        units=None,
        units_in=None,
    ):

        return _compute.convert_spectral_units(
            coll=self,
            data=data,
            units=units,
            units_in=units_in,
        )

    # -------------------
    # load specific file formats
    # -------------------

    def load_equilibrium_from_files(
        self,
        dpfe=None,
        returnas=None,
        # keys
        kmesh=None,
        # user-defined dunits
        dunits=None,
        # group naming
        func_key_groups=None,
        # sorting
        sort_vs=None,
        # options
        verb=None,
        strict=None,
        explore=None,
    ):
        """ Load an equilibria maps from several files

        Arg dpfe is fed to ds.get_files()
        Can be a dict of path / list of patterns
        Handles either:
            - .eqdsk
            - .mat (mep files)

        If all files do not have the same mesh size, they are grouped by shape
            - func_key_groups: callable to name those groups

        Loaded files can be  sorted vs any scalar quantity using
            - sort_vs

        Loads the R, Z mesh and corresponding 2d maps
        Also loads all 1d or scalar quantities
        """
        return _load_equilibrium.main(
            dpfe=dpfe,
            returnas=returnas,
            # keys
            kmesh=kmesh,
            # user-defined dunits
            dunits=dunits,
            # group naming
            func_key_groups=func_key_groups,
            # sorting
            sort_vs=sort_vs,
            # options
            verb=verb,
            strict=strict,
            explore=explore,
        )

    # -------------------
    # get data time
    # -------------------

    def get_time(
        self,
        key=None,
        t=None,
        indt=None,
        ind_strict=None,
        dim=None,
    ):
        """ Return the time vector or time macthing indices

        hastime, keyt, reft, keyt, val, dind = self.get_time(key='prof0')

        Return
        ------
        hastime:    bool
            flag, True if key has a time dimension
        keyt:       None /  str
            if hastime and a time vector exists, the key to that time vector
        t:          None / np.ndarray
            if hastime
        dind:       dict, with:
            - indt:  None / np.ndarray
                if indt or t was provided, and keyt exists
                int indices of nearest matching times
            - indtu: None / np.ndarray
                if indt is returned, np.unique(indt)
            - indtr: None / np.ndarray
                if indt is returned, a bool (ntu, nt) array
            - indok: None / np.ndarray
                if indt is returned, a bool (nt,) array

        """

        if dim is None:
            dim = 'time'

        return self.get_ref_vector(
            key0=key,
            values=t,
            indices=indt,
            ind_strict=ind_strict,
            dim=dim,
        )

    def get_time_common(
        self,
        keys=None,
        t=None,
        indt=None,
        ind_strict=None,
        dim=None,
    ):
        """ Return the time vector or time macthing indices

        hastime, hasvect, t, dind = self.get_time_common(
            keys=['prof0', 'prof1'],
            t=np.linspace(0, 5, 10),
        )

        Return
        ------
        hastime:        bool
            flag, True if key has a time dimension
        keyt:           None /  str
            if hastime and a time vector exists, the key to that time vector
        t:              None / np.ndarray
            if hastime
        indt:           None / np.ndarray
            if indt or t was provided, and keyt exists
            int indices of nearest matching times
        indtu:          None / np.ndarray
            if indt is returned, np.unique(indt)
        indt_reverse:   None / np.ndarray
            if indt is returned, a bool (ntu, nt) array

        """

        if dim is None:
            dim = 'time'

        return self.get_ref_vector_common(
            keys=keys,
            values=t,
            indices=indt,
            ind_strict=ind_strict,
            dim=dim,
        )

    def plot_as_profile2d(
        self,
        key=None,
        # parameters
        dres=None,
        dunique_mesh_2d=None,
        # levels
        dlevels=None,
        ref_com=None,
        # options
        plot_details=None,
        plot_config=None,
        # ref vectors
        dref_vectorZ=None,
        dref_vectorU=None,
        ref_vector_strategy=None,
        uniform=None,
        # interpolation
        val_out=None,
        nan0=None,
        # plot options
        dvminmax=None,
        cmap=None,
        dax=None,
        dmargin=None,
        fs=None,
        dcolorbar=None,
        dleg=None,
        # interactivity
        dinc=None,
        connect=True,
    ):

        # Plot profile 2d
        dax, dgroup = super().plot_as_profile2d(
            key=key,
            # parameters
            dres=dres,
            dunique_mesh_2d=dunique_mesh_2d,
            # levels
            dlevels=dlevels,
            ref_com=ref_com,
            # details
            plot_details=plot_details,
            # ref vectors
            dref_vectorZ=dref_vectorZ,
            dref_vectorU=dref_vectorU,
            ref_vector_strategy=ref_vector_strategy,
            uniform=uniform,
            # interpolation
            val_out=val_out,
            nan0=nan0,
            # plot options
            dvminmax=dvminmax,
            cmap=cmap,
            dax=dax,
            dmargin=dmargin,
            fs=fs,
            dcolorbar=dcolorbar,
            dleg=dleg,
            connect=False,
        )

        # plot config if relevant
        if plot_config.__class__.__name__ == 'Config':
            if 'matrix' in dax.dax.keys():
                ax = dax.dax['matrix']['handle']
                ax = plot_config.plot(lax=ax, proj='cross')

        # -----------
        # connect

        if connect is True:
            dax.setup_interactivity(kinter='inter0', dgroup=dgroup, dinc=dinc)
            dax.disconnect_old()
            dax.connect()

            dax.show_commands()
            return dax
        else:
            return dax, dgroup

    def plot_as_profile2d_compare(
        self,
        keys=None,
        # parameters
        dres=None,
        # levels
        dlevels=None,
        ref_com=None,
        # options
        plot_config=None,
        plot_details=None,
        # ref vectors
        dref_vectorZ=None,
        dref_vectorU=None,
        ref_vector_strategy=None,
        # interpolation
        val_out=None,
        nan0=None,
        # plot options
        vmin=None,
        vmax=None,
        cmap=None,
        dax=None,
        dmargin=None,
        fs=None,
        dcolorbar=None,
        dleg=None,
        # interactivity
        dinc=None,
        connect=True,
    ):

        # Plot profile 2d
        dax, dgroup = super().plot_as_profile2d_compare(
            keys=keys,
            # parameters
            dres=dres,
            # levels
            dlevels=dlevels,
            ref_com=ref_com,
            # details
            plot_details=plot_details,
            # ref vectors
            dref_vectorZ=dref_vectorZ,
            dref_vectorU=dref_vectorU,
            ref_vector_strategy=ref_vector_strategy,
            # interpolation
            val_out=val_out,
            nan0=nan0,
            # plot options
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            dax=dax,
            dmargin=dmargin,
            fs=fs,
            dcolorbar=dcolorbar,
            dleg=dleg,
            # interactivity
            connect=False,
        )

        # plot config if relevant
        if plot_config.__class__.__name__ == 'Config':
            for kax in ['prof0', 'prof1']:
                if kax in dax.dax.keys():
                    ax = dax.dax[kax]['handle']
                    ax = plot_config.plot(lax=ax, proj='cross')

        # -----------
        # connect

        if connect is True:
            dax.setup_interactivity(kinter='inter0', dgroup=dgroup, dinc=dinc)
            dax.disconnect_old()
            dax.connect()

            dax.show_commands()
            return dax
        else:
            return dax, dgroup