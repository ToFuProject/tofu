# -*- coding: utf-8 -*-


# Built-in
# import copy


# Common
import numpy as np
import datastock as ds


# tofu
from ._class07_Camera import Camera as Previous
from . import _class8_check as _check
from . import _class8_compute as _compute
from . import _class8_move as _move
from . import _class8_los_data as _los_data
from . import _class8_equivalent_apertures as _equivalent_apertures
from . import _class8_etendue_los as _etendue_los
from . import _class8_los_angles as _los_angles
from . import _class8_compute_signal as _compute_signal
from . import _class8_plot as _plot


__all__ = ['Diagnostic']


# #############################################################################
# #############################################################################
#                           Diagnostic
# #############################################################################


class Diagnostic(Previous):

    _show_in_summary = 'all'

    _dshow = dict(Previous._dshow)
    _dshow.update({
        'diagnostic': [
            'is2d',
            'spectro',
            'camera',
        ],
    })

    def add_diagnostic(
        self,
        key=None,
        doptics=None,
        # etendue
        etendue=None,
        # config for los
        config=None,
        length=None,
        # reflections
        reflections_nb=None,
        reflections_type=None,
        key_nseg=None,
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
            doptics=doptics,
            **kwdargs,
        )
        # update dicts
        self.update(dref=dref, ddata=ddata, dobj=dobj)

        # ---------------------
        # adding etendue / los

        key = list(dobj['diagnostic'].keys())[0]
        dopt = dobj['diagnostic'][key]['doptics']
        computable = any([len(v0['optics']) > 0 for v0 in dopt.values()])
        if compute is True and computable:
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
                key_nseg=key_nseg,
                # bool
                verb=verb,
                plot=False,
                store='analytical',
            )

    # -----------------
    # remove
    # -----------------

    def remove_diagnostic(self, key=None, key_cam=None):
        return _check._remove(
            coll=self,
            key=key,
            key_cam=key_cam,
        )

    # -----------------
    # utilities
    # -----------------

    def get_diagnostic_ref(self, key=None, key_cam=None):
        return _check.get_ref(coll=self, key=key, key_cam=key_cam)

    def get_diagnostic_cam(self, key=None, key_cam=None):
        return _check._get_default_cam(coll=self, key=key, key_cam=key_cam)

    def get_diagnostic_data(
        self,
        key=None,
        key_cam=None,
        data=None,
        rocking_curve=None,
        **kwdargs,
        ):
        """ Return dict of data for chosen cameras

        data can be:
            'etendue'
            'amin'
            'amax'
            'tangency radius'
            'lamb'
            'lambmin'
            'lambmax'
            'res'

        """
        return _compute._get_data(
            coll=self,
            key=key,
            key_cam=key_cam,
            data=data,
            rocking_curve=rocking_curve,
            **kwdargs,
        )

    def get_diagnostic_data_concatenated(
        self,
        key=None,
        key_data=None,
        flat=None,
        ):
        """ Return concatenated data for chosen cameras


        """
        return _compute._concatenate_data(
            coll=self,
            key=key,
            key_data=key_data,
            flat=flat,
        )

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
        margin_par=None,
        margin_perp=None,
        # equivalent aperture
        add_points=None,
        convex=None,
        # for storing los
        config=None,
        length=None,
        reflections_nb=None,
        reflections_type=None,
        key_nseg=None,
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

        dcompute, store = _etendue_los.compute_etendue_los(
            coll=self,
            key=key,
            # etendue
            analytical=analytical,
            numerical=numerical,
            res=res,
            check=check,
            margin_par=margin_par,
            margin_perp=margin_perp,
            # equivalent aperture
            add_points=add_points,
            convex=convex,
            # bool
            verb=verb,
            plot=plot,
            store=store,
        )

        # compute los angles
        c0 = (
            any([np.any(np.isfinite(v0['los_x'])) for v0 in dcompute.values()])
            and store
            )
        if c0:
            _los_angles.compute_los_angles(
                coll=self,
                key=key,
                # los
                config=config,
                length=length,
                reflections_nb=reflections_nb,
                reflections_type=reflections_type,
                key_nseg=key_nseg,
                dcompute=dcompute,
            )

    # ---------------
    # utilities
    # ---------------

    def get_diagnostic_equivalent_aperture(
        self,
        key=None,
        key_cam=None,
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
            key_cam=key_cam,
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
        key_cam=None,
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
            key_cam=key_cam,
            lamb=lamb,
            rocking_curve=rocking_curve,
        )

    # ---------------
    # utilities
    # ---------------

    def get_optics_cls(self, optics=None):
        """ Return list of optics and list of their classes

        """
        return _check._get_optics_cls(coll=self, optics=optics)

    # def get_diagnostic_doptics(self, key=None):
    #     """ 
    #     Get dict of optics and corresponding classes 

    #     """
    #     return _check._get_diagnostic_doptics(coll=self, key=key)

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
    # Moving
    # -----------------

    def move_diagnostic_to(
        self,
        key=None,
        key_cam=None,
        x=None,
        y=None,
        R=None,
        z=None,
        phi=None,
        theta=None,
        dphi=None,
        # computing
        compute=None,
        # los
        config=None,
        length=None,
        reflections_nb=None,
        reflections_type=None,
        key_nseg=None,
        # equivalent aperture
        add_points=None,
        convex=None,
        # etendue
        margin_par=None,
        margin_perp=None,
        verb=None,
    ):

        if compute is None:
            compute = True

        _move.move_to(
            self,
            key=key,
            key_cam=key_cam,
            x=x,
            y=y,
            R=R,
            z=z,
            phi=phi,
            theta=theta,
            dphi=dphi,
        )

        if compute:
            self.compute_diagnostic_etendue_los(
                key=key,
                #e etendue
                analytical=True,
                numerical=False,
                res=None,
                check=False,
                margin_par=margin_par,
                margin_perp=margin_perp,
                # equivalent aperture
                add_points=add_points,
                convex=convex,
                # los
                config=config,
                length=length,
                reflections_nb=reflections_nb,
                reflections_type=reflections_type,
                key_nseg=key_nseg,
                # bool
                verb=verb,
                plot=False,
                store='analytical',
            )

    # -----------------
    # computing
    # -----------------

    def compute_diagnostic_solid_angle(
        self,
        key=None,
        key_cam=None,
        # pts
        ptsx=None,
        ptsy=None,
        ptsz=None,
        # options
        config=None,
        visibility=None,
        # return
        return_vect=None,
        return_alpha=None,
    ):
        return _los_data.compute_solid_angles(
            coll=self,
            key=key,
            key_cam=key_cam,
            # pts
            ptsx=ptsx,
            ptsy=ptsy,
            ptsz=ptsz,
            # options
            config=config,
            visibility=visibility,
            # return
            return_vect=return_vect,
            return_alpha=return_alpha,
        )


    def compute_diagnostic_signal(
        self,
        key=None,
        key_diag=None,
        key_cam=None,
        # integrand
        key_integrand=None,
        # sampling
        method=None,
        res=None,
        mode=None,
        groupby=None,
        val_init=None,
        # signal
        brightness=None,
        # store
        store=None,
        # return
        returnas=None,
    ):
        """ Compute synthetic signal for a diagnostic and an emissivity field

        """

        return _compute_signal.compute_signal(
            coll=self,
            key=key,
            key_diag=key_diag,
            key_cam=key_cam,
            # integrand
            key_integrand=key_integrand,
            # sampling
            method=method,
            res=res,
            mode=mode,
            groupby=groupby,
            val_init=val_init,
            # signal
            brightness=brightness,
            # store
            store=store,
            # return
            returnas=returnas,
        )


    # -----------------
    # plotting
    # -----------------

    def get_diagnostic_dplot(
        self,
        key=None,
        key_cam=None,
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
            key_cam=key_cam,
            optics=optics,
            elements=elements,
            vect_length=vect_length,
        )

    def plot_diagnostic(
        self,
        key=None,
        key_cam=None,
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
            key_cam=key_cam,
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

    def plot_diagnostic_interpolated_along_los(
        self,
        key=None,
        key_cam=None,
        key_data_x=None,
        key_data_y=None,
        # sampling
        res=None,
        mode=None,
        segment=None,
        radius_max=None,
        # plotting
        vmin=None,
        vmax=None,
        plot=None,
        dcolor=None,
        dax=None,
    ):
        """ Compute and plot interpolated data along the los of the diagnostic

        """
        return _los_data._interpolated_along_los(
            coll=self,
            key=key,
            key_cam=key_cam,
            key_data_x=key_data_x,
            key_data_y=key_data_y,
            # sampling
            res=res,
            mode=mode,
            segment=segment,
            radius_max=radius_max,
            # plotting
            vmin=vmin,
            vmax=vmax,
            plot=plot,
            dcolor=dcolor,
            dax=dax,
            )
