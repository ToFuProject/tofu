# -*- coding: utf-8 -*-


# Built-in
# import copy


# tofu
from ._class01_Plasma2D import Plasma2D as Previous
from . import _class2_check as _check
from . import _class02_touch as _touch
from . import _class02_sample as _sample
from . import _class02_tangency_radius as _tangency_radius
from . import _class02_single_point_camera as _single_point_cam
from . import _class2_plot as _plot
from . import _class2_sinogram as _sinogram
from . import _class02_save2stp as _save2stp
from . import _class8_plot


__all__ = ['Rays']


# ##############################################################
# ##############################################################
#                       Rays
# ##############################################################


class Rays(Previous):

    _which_rays = 'rays'
    _show_in_summary = 'all'

    _dshow = dict(Previous._dshow)
    _dshow.update({
        _which_rays: [
            'shape',
            'ref',
            'pts',
            'lamb',
            'Rmin',
            'alpha',
            'dalpha',
            'dbeta',
        ],
    })

    def add_rays(
        self,
        key=None,
        # start points
        start_x=None,
        start_y=None,
        start_z=None,
        # wavelength
        lamb=None,
        # ref
        ref=None,
        # from pts
        pts_x=None,
        pts_y=None,
        pts_z=None,
        # from ray-tracing (vect + length or config or diag)
        vect_x=None,
        vect_y=None,
        vect_z=None,
        length=None,
        config=None,
        strict=None,
        # angles
        alpha=None,
        dalpha=None,
        dbeta=None,
        # reflections
        reflections_nb=None,
        reflections_type=None,
        key_nseg=None,
        diag=None,
        key_cam=None,
    ):
        """ Add a set of rays

        Can be defined from:
            - pts: explicit
            - starting pts + vectors

        """

        # check / format input
        dref, ddata, dobj = _check._rays(
            coll=self,
            key=key,
            # start points
            start_x=start_x,
            start_y=start_y,
            start_z=start_z,
            # wavelength
            lamb=lamb,
            # ref
            ref=ref,
            # from pts
            pts_x=pts_x,
            pts_y=pts_y,
            pts_z=pts_z,
            # from ray-tracing (vect + length or config or diag)
            vect_x=vect_x,
            vect_y=vect_y,
            vect_z=vect_z,
            length=length,
            config=config,
            strict=strict,
            # angles
            alpha=alpha,
            dalpha=dalpha,
            dbeta=dbeta,
            # reflections
            reflections_nb=reflections_nb,
            reflections_type=reflections_type,
            key_nseg=key_nseg,
            diag=diag,
            key_cam=key_cam,
        )

        # update dicts
        self.update(dref=dref, ddata=ddata, dobj=dobj)

    # -----------------
    # remove
    # -----------------

    def remove_rays(self, key=None):
        return _check._remove(
            coll=self,
            key=key,
        )

    # --------------
    #  utilities
    # --------------

    def get_rays_start(
        self,
        key=None,
        key_cam=None,
    ):
        return _check._get_start(coll=self, key=key, key_cam=key_cam)

    def get_rays_pts(
        self,
        key=None,
        key_cam=None,
    ):
        return _check._get_pts(coll=self, key=key, key_cam=key_cam)

    def get_rays_vect(
        self,
        key=None,
        key_cam=None,
        norm=None,
    ):
        return _check._get_vect(
            coll=self,
            key=key,
            key_cam=key_cam,
            norm=norm,
        )

    # --------------
    # touch
    # --------------

    def get_rays_touch_dict(
        self,
        key=None,
        config=None,
        segment=None,
        allowed=None,
        excluded=None,
    ):
        """ Compute which ray touches which solid in a config


        Parameters
        ----------
        key:    str
            key to an existing ray
        config: Config
            an existing Config instance

        Returns
        -------
        dout: dict
            output dict

        """
        return _touch.main(
            coll=self,
            key=key,
            config=config,
            segment=segment,
            allowed=allowed,
            excluded=excluded,
        )

    # --------------
    # discretizing
    # --------------

    def sample_rays(
        self,
        key=None,
        key_cam=None,
        res=None,
        mode=None,
        segment=None,
        ind_ch=None,
        radius_max=None,
        concatenate=None,
        return_coords=None,
    ):
        """ Return the sampled rays

        Parameters
        ----------
        key:        str
            key of the rays / diag los to sample
        res:        float
            sampling resolution
        mode:       str
            sampling mode
                - 'rel': relative, res is in [0, 1], 0.1 = 10 samples / segment
                - 'abs': absolute, res is a distance in m
        ind_ch:    tuple of indices (as from np.nonzero())
            indices of the channels that need to be sampled
        segment:    None / int / iterable of ints
            indices of te segments to be sampled
                - None: all
                - int: a single segment
                - iterable of ints: several segments
            Typical usage: None or -1 (last segment)
        radius_max:     None / float
            If provided, only sample the portion of segments that are inside
            the provided ;ajor radius
        concatenate:     bool
            flag indicating whether to concatenate the sampled points per ray
        """

        return _sample.main(
            coll=self,
            key=key,
            key_cam=key_cam,
            res=res,
            mode=mode,
            segment=segment,
            ind_ch=ind_ch,
            radius_max=radius_max,
            concatenate=concatenate,
            return_coords=return_coords,
        )

    # --------------
    # tangency radius
    # --------------

    def get_rays_quantity(
        self,
        key=None,
        key_cam=None,
        quantity=None,
        # limits
        segment=None,
        lim_to_segments=None,
        # for tangency radius
        axis_pt=None,
        axis_vect=None,
    ):
        """ Return a ray-specific quantity of each ray segment

        parameters
        ----------
        quantity:   str
            - 'length': length of rays
            - 'tangency radius': the tangency radius to an axis
        axis_pt:    len=3 iterable
            (x, y, z) coordinates of a pt on the axis, default to [0, 0, 0]
        axis_vect:  len=3 iterable
            (x, y, z) coordinates of the axis vector, default to [0, 0, 1]
        lim_to_segments: bool
            flag indicating whether to limit solutions to the segments

        Return
        -------
        radius:     np.ndarray of floats
            the tangency radii
        kk:         np.ndarray of floats
            the normalized longitudinal coordinate of the tangency points
        ref:        tuple
            The ref tuple on which the data depends
        """

        return _tangency_radius._tangency_radius(
            coll=self,
            key=key,
            key_cam=key_cam,
            quantity=quantity,
            # limits
            segment=segment,
            lim_to_segments=lim_to_segments,
            # for tangency radius
            axis_pt=axis_pt,
            axis_vect=axis_vect,
        )

    def get_rays_intersect_radius(
        self,
        key=None,
        key_cam=None,
        axis_pt=None,
        axis_vect=None,
        axis_radius=None,
        segment=None,
        lim_to_segments=None,
        return_pts=None,
        return_itot=None,
    ):
        """ Return the tangency radius to an axis of each ray segment

        parameters
        ----------
        axis_pt:    len=3 iterable
            (x, y, z) coordinates of a pt on the axis, default to [0, 0, 0]
        axis_vect:  len=3 iterable
            (x, y, z) coordinates of the axis vector, default to [0, 0, 1]
        axis_radius:    float
            The radius around the axis defining the cylinder to intersect
        lim_to_segments: bool
            flag indicating whether to limit solutions to the segments
        return_pts:
            flag indicating whether to return the pts (x, y, z) coordinates

        Return
        -------
        k0:         np.ndarray of floats
            First solution, per segment
        k1:         np.ndarray of floats
            Second solution, per segment
        iok:        np.ndarray of bool
            Flag indicating which segments have at least an intersection
        pts_x:      np.ndarray of floats
            The x coordinates of the points inside the cylinder
        pts_y:      np.ndarray of floats
            The y coordinates of the points inside the cylinder
        pts_z:      np.ndarray of floats
            The z coordinates of the points inside the cylinder

        """

        return _tangency_radius.intersect_radius(
            coll=self,
            key=key,
            key_cam=key_cam,
            axis_pt=axis_pt,
            axis_vect=axis_vect,
            axis_radius=axis_radius,
            segment=segment,
            lim_to_segments=lim_to_segments,
            return_pts=return_pts,
            return_itot=return_itot,
            )

    # --------------
    # Single point camera
    # --------------

    def add_single_point_camera2d(
        self,
        key=None,
        cent=None,
        nin=None,
        e0=None,
        e1=None,
        angle0=None,
        angle1=None,
        config=None,
        strict=None,
        # optional naming
        key_angle0=None,
        key_angle1=None,
        ref_angle0=None,
        ref_angle1=None,
        units_angles=None,
    ):
        """ Add a set of 2d rays from a single point

        Rays are sampling a portion of sphere around cent
        Portion is defined by 2 angles:
            - alpha
            - beta

        """

        return _single_point_cam.main(
            coll=self,
            key=key,
            cent=cent,
            nin=nin,
            e0=e0,
            e1=e1,
            angle0=angle0,
            angle1=angle1,
            config=config,
            strict=strict,
            # optional naming
            key_angle0=key_angle0,
            key_angle1=key_angle1,
            ref_angle0=ref_angle0,
            ref_angle1=ref_angle1,
            units_angles=units_angles,
        )

    # ------------------
    # plotting - static
    # ------------------

    def plot_rays_static(
        self,
        key=None,
        proj=None,
        concatenate=None,
        mode=None,
        res=None,
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
        return _plot._plot_rays(
            coll=self,
            key=key,
            proj=proj,
            concatenate=concatenate,
            mode=mode,
            res=res,
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

    # ------------------
    # plotting - interactive
    # ------------------

    def plot_rays(
        self,
        # keys
        key=None,
        keyZ=None,
        # options
        proj=None,
        res=None,
        # data plot
        data=None,
        units=None,
        cmap=None,
        vmin=None,
        vmax=None,
        alpha=None,
        dx0=None,
        dx1=None,
        # config
        plot_config=None,
        plot_colorbar=None,
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

        return _class8_plot._plot_diagnostic(
            coll=self,
            isray=True,
            # keys
            key=key,
            keyZ=keyZ,
            # options
            proj=proj,
            los_res=res,
            # data plot
            data=data,
            units=units,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=alpha,
            dx0=dx0,
            dx1=dx1,
            # config
            plot_config=plot_config,
            plot_colorbar=plot_colorbar,
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

    # --------------
    # sinogram
    # --------------

    def get_sinogram(
        self,
        key=None,
        segment=None,
        # config
        config=None,
        kVes=None,
        # sinogram ref
        R0=None,
        Z0=None,
        # sinogram options
        ang=None,
        ang_units=None,
        impact_pos=None,
        pmax=None,
        Dphimax=None,
        # plotting options
        plot=None,
        sketch=None,
        color=None,
        marker=None,
        label=None,
        # other options
        verb=None,
        # figure
        dax=None,
        dmargin=None,
        fs=None,
        wintit=None,
    ):
        """ plot the sinogram of a set of rays / diagnostic LOS

        Optionally plot also the configuration

        The sinogram can be plotted with:
            * ang = 'ksi' or 'theta'
            * ang_units = 'rad' or 'deg'
            * impact_pos = True or False
            * sketch = True or False

        """

        return _sinogram.sinogram(
            coll=self,
            key=key,
            segment=segment,
            # config
            config=config,
            kVes=kVes,
            # sinogram ref
            R0=R0,
            Z0=Z0,
            # sinogram options
            ang=ang,
            ang_units=ang_units,
            impact_pos=impact_pos,
            pmax=pmax,
            Dphimax=Dphimax,
            # plotting options
            plot=plot,
            sketch=sketch,
            color=color,
            marker=marker,
            label=label,
            # other options
            verb=verb,
            # figure
            dax=dax,
            dmargin=dmargin,
            fs=fs,
            wintit=wintit,
        )

    # --------------
    # save to step file (CAD)
    # --------------

    def save_rays_to_stp(
        # ---------------
        # input from tofu
        self,
        key=None,
        key_cam=None,
        # input from arrays
        ptsx=None,
        ptsy=None,
        ptsz=None,
        # input from file
        pfe_in=None,
        # ---------------
        # options
        outline_only=None,
        include_centroid=None,
        factor=None,
        color=None,
        color_by_pixel=None,
        chain=None,
        # ---------------
        # saving
        pfe_save=None,
        overwrite=None,
    ):
        """ Save a set of 'rays' to a stp file (CAD-readable)

        Parameters
        ----------
        key : str, optional
            key to the existing set of rays
        pfe : str, optional
            valid path-file-extension (file str) where to save the stp file

        Returns
        -------
        None.

        """

        return _save2stp.main(
            # ---------------
            # input from tofu
            coll=self,
            key=key,
            key_cam=key_cam,
            # input from arrays
            ptsx=ptsx,
            ptsy=ptsy,
            ptsz=ptsz,
            # input from file
            pfe_in=pfe_in,
            # ---------------
            # options
            outline_only=outline_only,
            include_centroid=include_centroid,
            factor=factor,
            color=color,
            color_by_pixel=color_by_pixel,
            chain=chain,
            # ---------------
            # saving
            pfe_save=pfe_save,
            overwrite=overwrite,
        )
