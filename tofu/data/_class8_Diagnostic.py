# -*- coding: utf-8 -*-


# Built-in
# import copy


# Common
import numpy as np
import datastock as ds


# tofu
from . import _class7_Camera
from . import _class8_check as _check
from . import _class8_compute as _compute
from . import _class8_equivalent_apertures as _equivalent_apertures
from . import _class8_etendue_los as _etendue_los
from . import _class8_los_angles as _los_angles
from . import _class8_plot as _plot


__all__ = ['Diagnostic']


# #############################################################################
# #############################################################################
#                           Diagnostic
# #############################################################################


class Diagnostic(_class7_Camera.Camera):

    _show_in_summary = 'all'

    _dshow = dict(_class7_Camera.Camera._dshow)
    _dshow.update({
        'diagnostic': [
            'type',
            'optics',
            'spectro',
            'ref',
            'etendue',
            'etend_type',
            'los',
            'amin',
            'amax',
            'spectrum',
            'time res.',
        ],
    })

    def add_diagnostic(
        self,
        key=None,
        optics=None,
        # etendue
        etendue=None,
        # config for los
        config=None,
        length=None,
        reflections_nb=None,
        reflections_type=None,
        # compute
        compute=True,
        # others
        verb=None,
        **kwdargs,
    ):

        # -----------
        # adding diag

        # check / format input
        dref, ddata, dobj = _check._diagnostics(
            coll=self,
            key=key,
            optics=optics,
            **kwdargs,
        )
        # update dicts
        self.update(dref=dref, ddata=ddata, dobj=dobj)

        # --------------
        # adding etendue

        key = list(dobj['diagnostic'].keys())[0]
        optics = self.dobj['diagnostic'][key]['optics']

        if len(optics) > 1 and compute is True:
            self.compute_diagnostic_etendue_los(
                key=key,
                analytical=True,
                numerical=False,
                res=None,
                check=False,
                # los
                config=config,
                length=length,
                reflections_nb=reflections_nb,
                reflections_type=reflections_type,
                # bool
                verb=verb,
                plot=False,
                store='analytical',
            )

    # -----------------
    # utilities
    # -----------------

    def get_diagnostic_ref(self, key=None):
        return _check.get_ref(coll=self, key=key)

    # -----------------
    # etendue computing
    # -----------------

    def compute_diagnostic_etendue_los(
        self,
        key=None,
        # parameters
        analytical=None,
        numerical=None,
        res=None,
        check=None,
        # for storing los
        config=None,
        length=None,
        reflections_nb=None,
        reflections_type=None,
        # bool
        verb=None,
        plot=None,
        store=None,
    ):
        """ Compute the etendue of the diagnostic (per pixel)

        Etendue (m2.sr) can be computed analytically or numerically
        If plot, plot the comparison between all computations
        If store = 'analytical' or 'numerical', overwrites the diag etendue

        """
        dlos, store = _etendue_los.compute_etendue_los(
            coll=self,
            key=key,
            analytical=analytical,
            numerical=numerical,
            res=res,
            check=check,
            # bool
            verb=verb,
            plot=plot,
            store=store,
        )

        if store is not False and np.any(np.isfinite(dlos['los_x'])):
            _los_angles.compute_los_angles(
                coll=self,
                key=key,
                # los
                config=config,
                length=length,
                reflections_nb=reflections_nb,
                reflections_type=reflections_type,
                **dlos,
            )

    # ---------------
    # utilities
    # ---------------

    def get_diagnostic_equivalent_aperture(
        self,
        key=None,
        pixel=None,
        # inital contour
        add_points=None,
        # options
        convex=None,
        harmonize=None,
        reshape=None,
        return_for_etendue=None,
        # plot
        plot=None,
        verb=None,
        store=None,
    ):
        """"""
        return _equivalent_apertures.equivalent_apertures(
            coll=self,
            key=key,
            pixel=pixel,
            # inital contour
            add_points=add_points,
            # options
            convex=convex,
            harmonize=harmonize,
            reshape=reshape,
            return_for_etendue=return_for_etendue,
            # plot
            plot=plot,
            verb=verb,
            store=store,
        )

    # ---------------
    # wavelneght from angle
    # ---------------

    def get_diagnostic_lamb(
        self,
        key=None,
        lamb=None,
        rocking_curve=None,
    ):
        """ Return the wavelength associated to
        - 'lamb'
        - 'lambmin'
        - 'lambmax'
        - 'res' = lamb / (lambmax - lambmin)

        """
        return _compute.get_lamb_from_angle(
            coll=self,
            key=key,
            lamb=lamb,
            rocking_curve=rocking_curve,
        )

    # ---------------
    # utilities
    # ---------------

    def get_diagnostic_optics(self, key=None, optics=None):
        """ Get list of optics and list of corresponding classes """
        return _check._get_optics(coll=self, key=key, optics=optics)

    def get_optics_outline(
        self,
        key=None,
        add_points=None,
        mode=None,
        closed=None,
        ravel=None,
        total=None,
    ):
        """ Return the optics outline """
        return _compute.get_optics_outline(
            coll=self,
            key=key,
            add_points=add_points,
            mode=mode,
            closed=closed,
            ravel=ravel,
            total=total,
        )

    def get_optics_poly(
        self,
        key=None,
        add_points=None,
        mode=None,
        closed=None,
        ravel=None,
        total=None,
        return_outline=None,
    ):
        """ Return the optics outline """
        return _compute.get_optics_poly(
            coll=self,
            key=key,
            add_points=add_points,
            mode=mode,
            closed=closed,
            ravel=ravel,
            total=total,
            return_outline=return_outline,
        )

    def set_optics_color(self, key=None, color=None):
        return _check._set_optics_color(
            coll=self,
            key=key,
            color=color,
        )

    # -----------------
    # plotting
    # -----------------

    def get_diagnostic_dplot(
        self,
        key=None,
        optics=None,
        elements=None,
        vect_length=None,
    ):
        """ Return a dict with all that's necessary for plotting

        If no optics is provided, all are returned

        elements indicate, for each optics, what should be represented:
            - 'o': outline
            - 'v': unit vectors
            - 's': summit ( = center for non-curved)
            - 'c': center (of curvature)
            - 'r': rowland circle / axis of cylinder

        returned as a dict:

        dplot = {
            'optics0': {
                'o': {
                    'x0': ...,
                    'x1': ...,
                    'x': ...,
                    'y': ...,
                    'z': ...,
                    'r': ...,
                },
                'v': {
                    'x': ...,
                    'y': ...,
                    'z': ...,
                    'r': ...,
                },
            },
        }

        """

        return _compute._dplot(
            coll=self,
            key=key,
            optics=optics,
            elements=elements,
            vect_length=vect_length,
        )

    def plot_diagnostic(
        self,
        key=None,
        optics=None,
        elements=None,
        proj=None,
        los_res=None,
        # data plot
        data=None,
        cmap=None,
        vmin=None,
        vmax=None,
        # config
        plot_config=None,
        # figure
        dax=None,
        dmargin=None,
        fs=None,
        wintit=None,
        # interactivity
        color_dict=None,
        nlos=None,
        dinc=None,
        connect=None,
    ):

        return _plot._plot_diagnostic(
            coll=self,
            key=key,
            optics=optics,
            elements=elements,
            proj=proj,
            los_res=los_res,
            # data plot
            data=data,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            # config
            plot_config=plot_config,
            # figure
            dax=dax,
            dmargin=dmargin,
            fs=fs,
            wintit=wintit,
            # interactivity
            color_dict=color_dict,
            nlos=nlos,
            dinc=dinc,
            connect=connect,
        )
