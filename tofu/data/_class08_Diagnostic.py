# -*- coding: utf-8 -*-


# Built-in
# import copy


# Common
import numpy as np


# tofu
from ._class07_Camera import Camera as Previous
from . import _class8_check as _check
from . import _class8_compute as _compute
from . import _class08_get_data as _get_data
from . import _class08_concatenate_data as _concatenate
from . import _class8_move as _move
from . import _class8_los_data as _los_data
from . import _class08_interpolate_along_los as _interpolate_along_los
from . import _class8_equivalent_apertures as _equivalent_apertures
from . import _class8_etendue_los as _etendue_los
from . import _class8_vos as _vos
from . import _class8_vos_spectro_nobin_at_lamb as _vos_nobin_at_lamb
from . import _class8_los_angles as _los_angles
from . import _class08_generate_rays as _generate_rays
from . import _class8_plane_perp_to_los as _planeperp
from . import _class8_compute_signal as _compute_signal
from . import _class8_compute_signal_moments as _signal_moments
from . import _class8_reverse_ray_tracing as _reverse_rt
from . import _class8_plot as _plot
from . import _class8_plot_vos as _plot_vos
from . import _class8_plot_coverage as _plot_coverage
# from . import _class08_save2stp as _save2stp
from . import _class08_saveload_from_file as _saveload_from_file


__all__ = ['Diagnostic']


# ###############################################################
# ###############################################################
#                           Diagnostic
# ###############################################################


class Diagnostic(Previous):

    _show_in_summary = 'all'

    _dshow = dict(Previous._dshow)
    _dshow.update({
        'diagnostic': [
            'is2d',
            'spectro',
            'PHA',
            'camera',
            'signal',
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
        strict=None,
        length=None,
        # reflections
        reflections_nb=None,
        reflections_type=None,
        key_nseg=None,
        # compute
        compute=True,
        add_points=None,
        convex=None,
        # spectro-only
        rocking_curve_fw=None,
        # others
        compute_vos_from_los=None,
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
                strict=strict,
                length=length,
                reflections_nb=reflections_nb,
                reflections_type=reflections_type,
                key_nseg=key_nseg,
                # equivalent aperture
                add_points=add_points,
                convex=convex,
                # spectro-only
                rocking_curve_fw=rocking_curve_fw,
                # bool
                compute_vos_from_los=compute_vos_from_los,
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

    def get_diagnostic_cam(self, key=None, key_cam=None, default=None):
        return _check._get_default_cam(
            coll=self,
            key=key,
            key_cam=key_cam,
            default=default,
        )

    def get_diagnostic_data(
        self,
        key=None,
        key_cam=None,
        data=None,
        # relevant for LOS data
        segment=None,
        # relevant for spectro data
        rocking_curve=None,
        units=None,
        default=None,
        print_full_doc=None,
        **kwdargs,
    ):
        """ Return dict of built-in data for chosen cameras

        data can be:
            'etendue'
            'amin'
            'amax'
            'tangency_radius'
            'length'
            'alpha'
            'alpha_pixel'
            'lamb'
            'lambmin'
            'lambmax'
            'res'

        """
        return _get_data.main(
            coll=self,
            key=key,
            key_cam=key_cam,
            data=data,
            # relevant for LOS data
            segment=segment,
            # relevant for spectro data
            rocking_curve=rocking_curve,
            units=units,
            default=default,
            print_full_doc=print_full_doc,
            **kwdargs,
        )

    def get_diagnostic_data_concatenated(
        self,
        key=None,
        key_data=None,
        key_cam=None,
        flat=None,
    ):
        """ Return concatenated data for chosen cameras


        """
        return _concatenate.main(
            coll=self,
            key=key,
            key_data=key_data,
            key_cam=key_cam,
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
        # spectro-only
        ind_ap_lim_spectral=None,
        rocking_curve_fw=None,
        rocking_curve_max=None,
        # equivalent aperture
        add_points=None,
        convex=None,
        # for storing los
        config=None,
        strict=None,
        length=None,
        reflections_nb=None,
        reflections_type=None,
        key_nseg=None,
        # bool
        compute_vos_from_los=None,
        return_dcompute=None,
        verb=None,
        plot=None,
        store=None,
        overwrite=None,
        debug=None,
    ):
        """ Compute the etendue of the diagnostic (per pixel)

        Etendue (m2.sr) can be computed analytically or numerically
        If plot, plot the comparison between all computations
        If store = 'analytical' or 'numerical', overwrites the diag etendue

        """

        # prepare computation
        dcompute, store, overwrite = _etendue_los.compute_etendue_los(
            coll=self,
            key=key,
            # etendue
            analytical=analytical,
            numerical=numerical,
            res=res,
            check=check,
            margin_par=margin_par,
            margin_perp=margin_perp,
            # spectro-only
            ind_ap_lim_spectral=ind_ap_lim_spectral,
            rocking_curve_fw=rocking_curve_fw,
            rocking_curve_max=rocking_curve_max,
            # equivalent aperture
            add_points=add_points,
            convex=convex,
            # bool
            verb=verb,
            plot=plot,
            store=store,
            overwrite=overwrite,
            debug=debug,
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
                strict=strict,
                length=length,
                reflections_nb=reflections_nb,
                reflections_type=reflections_type,
                key_nseg=key_nseg,
                dcompute=dcompute,
                compute_vos_from_los=compute_vos_from_los,
                overwrite=overwrite,
            )

        if return_dcompute is True:
            return dcompute

    def compute_diagnostic_solidangle_from_plane(
        self,
        key_diag=None,
        key_cam=None,
        indch=None,
        indref=None,
        # parameters
        res=None,
        margin_par=None,
        margin_perp=None,
        config=None,
        # solid angle
        n0=None,
        n1=None,
        # lamb
        res_lamb=None,
        # bool
        verb=None,
        plot=None,
        # plotting
        indplot=None,
        dax=None,
        plot_config=None,
        fs=None,
        dmargin=None,
        vmin_cam0=None,
        vmax_cam0=None,
        vmin_cam=None,
        vmax_cam=None,
        vmin_cam_lamb=None,
        vmax_cam_lamb=None,
        vmin_plane=None,
        vmax_plane=None,
    ):
        """ Creates a plane perpendicular to los
        compute contribution of each point to the signal

        return dout
        """

        return _planeperp.main(
            coll=self,
            key_diag=key_diag,
            key_cam=key_cam,
            indch=indch,
            indref=indref,
            # parameters
            res=res,
            margin_par=margin_par,
            margin_perp=margin_perp,
            config=config,
            # solid angle
            n0=n0,
            n1=n1,
            # lamb
            res_lamb=res_lamb,
            # bool
            verb=verb,
            plot=plot,
            # plotting
            indplot=indplot,
            dax=dax,
            plot_config=plot_config,
            fs=fs,
            dmargin=dmargin,
            vmin_cam0=vmin_cam0,
            vmax_cam0=vmax_cam0,
            vmin_cam=vmin_cam,
            vmax_cam=vmax_cam,
            vmin_cam_lamb=vmin_cam_lamb,
            vmax_cam_lamb=vmax_cam_lamb,
            vmin_plane=vmin_plane,
            vmax_plane=vmax_plane,
        )

    # -----------------
    # add rays from diag
    # -----------------

    def add_rays_from_diagnostic(
        self,
        key=None,
        # sampling
        dsampling_pixel=None,
        dsampling_optics=None,
        # optics (to restrain to certain optics only for faster)
        optics=None,
        # computing
        config=None,
        # storing
        store=None,
        strict=None,
        key_rays=None,
        overwrite=None,
    ):
        return _generate_rays.main(
            coll=self,
            key=key,
            # sampling
            dsampling_pixel=dsampling_pixel,
            dsampling_optics=dsampling_optics,
            # optics (to restrain to certain optics only for faster)
            optics=optics,
            # computing
            config=config,
            # storing
            store=store,
            strict=strict,
            key_rays=key_rays,
            overwrite=overwrite,
        )

    # -----------------
    # solid angle from plane
    # -----------------

    def plot_diagnostic_solidangle_from_plane(
        self,
        dout=None,
        # plotting
        indplot=None,
        dax=None,
        plot_config=None,
        fs=None,
        dmargin=None,
        vmin_cam0=None,
        vmax_cam0=None,
        vmin_cam=None,
        vmax_cam=None,
        vmin_cam_lamb=None,
        vmax_cam_lamb=None,
        vmin_plane=None,
        vmax_plane=None,
    ):
        """ Creates a plane perpendicular to los
        compute contribution of each point to the signal
        """

        return _planeperp._plot(
            coll=self,
            # extra
            indplot=indplot,
            dax=dax,
            plot_config=plot_config,
            fs=fs,
            dmargin=dmargin,
            vmin_cam0=vmin_cam0,
            vmax_cam0=vmax_cam0,
            vmin_cam=vmin_cam,
            vmax_cam=vmax_cam,
            vmin_cam_lamb=vmin_cam_lamb,
            vmax_cam_lamb=vmax_cam_lamb,
            vmin_plane=vmin_plane,
            vmax_plane=vmax_plane,
            # dout
            **dout,
        )

    def compute_diagnostic_vos(
        self,
        key_diag=None,
        key_cam=None,
        key_mesh=None,
        config=None,
        # sampling
        res_RZ=None,
        res_phi=None,
        lamb=None,
        res_lamb=None,
        res_rock_curve=None,
        n0=None,
        n1=None,
        convexHull=None,
        # user-defined limits
        user_limits=None,
        # keep3d
        keep3d=None,
        return_vector=None,
        # margins
        margin_poly=None,
        # raytracing
        visibility=None,
        # spectro-only
        rocking_curve_fw=None,
        rocking_curve_max=None,
        # bool
        check=None,
        verb=None,
        debug=None,
        store=None,
        overwrite=None,
        replace_poly=None,
        timing=None,
    ):
        """ Compute the vos of the diagnostic (per pixel)

        - poly_margin (0.3) fraction by which the los-estimated vos is widened
        -store:
            - if replace_poly, will replace the vos polygon approximation
            - will store the toroidally-integrated solid angles

        Return dvos, dref

        """

        return _vos.compute_vos(
            coll=self,
            key_diag=key_diag,
            key_cam=key_cam,
            key_mesh=key_mesh,
            config=config,
            # etendue
            res_RZ=res_RZ,
            res_phi=res_phi,
            lamb=lamb,
            res_lamb=res_lamb,
            res_rock_curve=res_rock_curve,
            n0=n0,
            n1=n1,
            convexHull=convexHull,
            # user-defined limits
            user_limits=user_limits,
            # keep3d
            keep3d=keep3d,
            return_vector=return_vector,
            # margins
            margin_poly=margin_poly,
            # spectro-only
            rocking_curve_fw=rocking_curve_fw,
            rocking_curve_max=rocking_curve_max,
            # bool
            visibility=visibility,
            check=check,
            verb=verb,
            debug=debug,
            store=store,
            overwrite=overwrite,
            replace_poly=replace_poly,
            timing=timing,
        )

    def check_diagnostic_dvos(
        self,
        key=None,
        key_cam=None,
        dvos=None,
    ):
        """ Check dvos and return it if stored """
        return _vos._check_get_dvos(
            coll=self,
            key=key,
            key_cam=key_cam,
            dvos=dvos,
        )

    def get_dvos_xyz(
        self,
        key=None,
        key_cam=None,
        dvos=None,
    ):
        """ Return ptsx, ptsy, ptsz from 3d vos """
        return _vos.get_dvos_xyz(
            coll=self,
            key_diag=key,
            key_cam=key_cam,
            dvos=dvos,
        )

    def store_diagnostic_vos(
        self,
        key_diag=None,
        dvos=None,
        dref=None,
        spectro=None,
        overwrite=None,
        replace_poly=None,
    ):
        """ Store a pre-computed dvos """
        _vos._store(
            coll=self,
            key_diag=key_diag,
            dvos=dvos,
            dref=dref,
            spectro=spectro,
            overwrite=overwrite,
            replace_poly=replace_poly,
        )

    def compute_diagnostic_vos_nobin_at_lamb(
        self,
        key_diag=None,
        key_cam=None,
        key_mesh=None,
        lamb=None,
        config=None,
        # parameters
        res_RZ=None,
        res_phi=None,
        res_rock_curve=None,
        n0=None,
        n1=None,
        convexHull=None,
        # margins
        margin_poly=None,
        nmax_rays=None,
        # spectro-only
        rocking_curve_fw=None,
        rocking_curve_max=None,
        # optional binning
        dobin=None,
        bin0=None,
        bin1=None,
        remove_raw=None,
        # bool
        visibility=None,
        verb=None,
        debug=None,
        # plot
        plot=None,
        pix0=None,
        pix1=None,
        tit=None,
    ):
        """ Compute the image of a mono-wavelength plasma volume

        Does ray-tracing from the whole plasma volume
        For a set of discrete user-defined wavelengths
        Optionally (dobin=True) also provides the binned (per-pixel) image

        - lamb: float or sequence of floats, wavelength to be traced

        Parameters for plasma volume sampling:
            - res_RZ: float (m)
            - res_phi: float (m)
            - res_rock_curve: float (rad)

        Parameters for sampling the solid angle for each point source
            - n0: int, nb of rays in horizontal direction
            - n1: int, nb of rays in vertical direction

        Parameters for binning (by default uses the camera pixels)
            - dobin: bool, do the binning or not
            - bin0: int, nb of pixels in the horizontal direction
            - bin1: int, nb of pixels in the vertical direction

        Parameters for plotting
            - plot: bool
            - pix0: index 0 of pixels outlines to be plotted
            - pix1: index 1 of pixels outlines to be plotted

        """

        return _vos_nobin_at_lamb.compute_vos_nobin_at_lamb(
            coll=self,
            key_diag=key_diag,
            key_cam=key_cam,
            key_mesh=key_mesh,
            lamb=lamb,
            config=config,
            # etendue
            res_RZ=res_RZ,
            res_phi=res_phi,
            res_rock_curve=res_rock_curve,
            n0=n0,
            n1=n1,
            convexHull=convexHull,
            # margins
            margin_poly=margin_poly,
            nmax_rays=nmax_rays,
            # spectro-only
            rocking_curve_fw=rocking_curve_fw,
            rocking_curve_max=rocking_curve_max,
            # optional binning
            dobin=dobin,
            bin0=bin0,
            bin1=bin1,
            remove_raw=remove_raw,
            # bool
            visibility=visibility,
            verb=verb,
            debug=debug,
            # plot
            plot=plot,
            pix0=pix0,
            pix1=pix1,
            tit=tit,
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
        ind_ap_lim_spectral=None,
        convex=None,
        harmonize=None,
        reshape=None,
        return_for_etendue=None,
        # plot
        plot=None,
        verb=None,
        store=None,
        debug=None,
    ):
        """ Get the equivalent projected aperture for each pixel
        """

        return _equivalent_apertures.equivalent_apertures(
            coll=self,
            key=key,
            key_cam=key_cam,
            pixel=pixel,
            # inital contour
            add_points=add_points,
            # options
            ind_ap_lim_spectral=ind_ap_lim_spectral,
            convex=convex,
            harmonize=harmonize,
            reshape=reshape,
            return_for_etendue=return_for_etendue,
            # plot
            plot=plot,
            verb=verb,
            store=store,
            debug=debug,
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
        units=None,
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
            units=units,
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
        min_threshold=None,
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
            min_threshold=min_threshold,
            mode=mode,
            closed=closed,
            ravel=ravel,
            total=total,
        )

    def get_optics_poly(
        self,
        key=None,
        add_points=None,
        min_threshold=None,
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
            min_threshold=min_threshold,
            mode=mode,
            closed=closed,
            ravel=ravel,
            total=total,
            return_outline=return_outline,
        )

    def get_optics_as_input_solid_angle(
        self,
        keys=None,
    ):
        """ Return the optics outline """
        return _compute.get_optics_as_input_solid_angle(
            coll=self,
            keys=keys,
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
                # etendue
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
        # spectral ref
        key_ref_spectro=None,
        # sampling
        method=None,
        res=None,
        mode=None,
        groupby=None,
        val_init=None,
        ref_com=None,
        # signal
        brightness=None,
        spectral_binning=None,
        # vos
        dvos=None,
        # verb
        verb=None,
        # timing
        timing=None,
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
            # spectral ref
            key_ref_spectro=key_ref_spectro,
            # sampling
            method=method,
            res=res,
            mode=mode,
            groupby=groupby,
            val_init=val_init,
            ref_com=ref_com,
            # signal
            brightness=brightness,
            spectral_binning=spectral_binning,
            # vos
            dvos=dvos,
            # verb
            verb=verb,
            # timing
            timing=timing,
            # store
            store=store,
            # return
            returnas=returnas,
        )

    # -----------------------
    # ray-tracing from plasma
    # -----------------------

    def get_raytracing_from_pts(
        self,
        # diag
        key=None,
        key_cam=None,
        # mesh sampling
        key_mesh=None,
        res_RZ=None,
        res_phi=None,
        # pts coordinates
        ptsx=None,
        ptsy=None,
        ptsz=None,
        # res rays
        n0=None,
        n1=None,
        # optional lamb
        lamb0=None,
        res_lamb=None,
        rocking_curve=None,
        res_rock_curve=None,
        # options
        append=None,
        plot=None,
        dax=None,
        plot_pixels=None,
        plot_config=None,
        vmin=None,
        vmax=None,
        aspect3d=None,
        elements=None,
        colorbar=None,
    ):
        """ Get rays from plasma points to camera for a spectrometer diag """

        return _reverse_rt._from_pts(
            coll=self,
            # diag
            key=key,
            key_cam=key_cam,
            # mesh sampling
            key_mesh=key_mesh,
            res_RZ=res_RZ,
            res_phi=res_phi,
            # pts coordinates
            ptsx=ptsx,
            ptsy=ptsy,
            ptsz=ptsz,
            # res rays
            n0=n0,
            n1=n1,
            # optional lamb
            lamb0=lamb0,
            res_lamb=res_lamb,
            rocking_curve=rocking_curve,
            res_rock_curve=res_rock_curve,
            # options
            append=append,
            plot=plot,
            dax=dax,
            plot_pixels=plot_pixels,
            plot_config=plot_config,
            vmin=vmin,
            vmax=vmax,
            aspect3d=aspect3d,
            elements=elements,
            colorbar=colorbar,
        )

    # ---------------------
    # interpolate along los
    # ---------------------

    def interpolate_along_los(
        self,
        key_diag=None,
        key_cam=None,
        key_integrand=None,
        key_coords=None,
        # sampling
        res=None,
        mode=None,
        segment=None,
        radius_max=None,
        concatenate=None,
        # interpolating
        domain=None,
        val_out=None,
        # plotting
        vmin=None,
        vmax=None,
        plot=None,
        dcolor=None,
        dax=None,
    ):
        """ Compute and plot interpolated data along the los of the diagnostic

        """
        return _interpolate_along_los.main(
            coll=self,
            key_diag=key_diag,
            key_cam=key_cam,
            key_integrand=key_integrand,
            key_coords=key_coords,
            # sampling
            res=res,
            mode=mode,
            segment=segment,
            radius_max=radius_max,
            concatenate=concatenate,
            # interpolating
            domain=domain,
            val_out=val_out,
            # plotting
            vmin=vmin,
            vmax=vmax,
            plot=plot,
            dcolor=dcolor,
            dax=dax,
        )

    # -----------------
    # data moments
    # -----------------

    def compute_diagnostic_binned_data(
        self,
        key_diag=None,
        key_cam=None,
        # data to be binned
        data=None,
        # binning dimension
        bins0=None,
        bins1=None,
        bin_data0=None,
        bin_data1=None,
        # store
        store=None,
        # plotting
        plot=None,
    ):

        return _signal_moments.binned(
            coll=self,
            key_diag=key_diag,
            key_cam=key_cam,
            # data to be binned
            data=data,
            # binning dimension
            bins0=bins0,
            bins1=bins1,
            bin_data0=bin_data0,
            bin_data1=bin_data1,
            # store
            store=store,
            # plotting
            plot=plot,
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
        dx0=None,
        dx1=None,
        default=None,
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
            dx0=dx0,
            dx1=dx1,
            default=default,
        )

    def plot_diagnostic(
        self,
        # keys
        key=None,
        key_cam=None,
        keyZ=None,
        # options
        optics=None,
        elements=None,
        proj=None,
        los_res=None,
        # data plot
        data=None,
        units=None,
        cmap=None,
        vmin=None,
        vmax=None,
        alpha=None,
        dx0=None,
        dx1=None,
        # plot vos polygons
        plot_pcross=None,
        plot_phor=None,
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

        return _plot._plot_diagnostic(
            coll=self,
            # keys
            key=key,
            key_cam=key_cam,
            keyZ=keyZ,
            # options
            optics=optics,
            elements=elements,
            proj=proj,
            los_res=los_res,
            # data plot
            data=data,
            units=units,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=alpha,
            dx0=dx0,
            dx1=dx1,
            # plot vos polygons
            plot_pcross=plot_pcross,
            plot_phor=plot_phor,
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

    def plot_diagnostic_vos(
        self,
        key=None,
        key_cam=None,
        indch=None,
        indlamb=None,
        optics=None,
        elements=None,
        proj=None,
        los_res=None,
        # data plot
        dvos=None,
        units=None,
        cmap=None,
        vmin=None,
        vmax=None,
        vmin_tot=None,
        vmax_tot=None,
        vmin_cam=None,
        vmax_cam=None,
        dvminmax=None,
        alpha=None,
        # plot vos polygons
        plot_pcross=None,
        plot_phor=None,
        plot_colorbar=None,
        # config
        plot_config=None,
        # figure
        dax=None,
        dmargin=None,
        fs=None,
        wintit=None,
        # interactivity
        color_dict=None,
    ):

        return _plot_vos._plot_diagnostic_vos(
            coll=self,
            key=key,
            key_cam=key_cam,
            indch=indch,
            indlamb=indlamb,
            optics=optics,
            elements=elements,
            proj=proj,
            los_res=los_res,
            # data plot
            dvos=dvos,
            units=units,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            vmin_tot=vmin_tot,
            vmax_tot=vmax_tot,
            vmin_cam=vmin_cam,
            vmax_cam=vmax_cam,
            dvminmax=dvminmax,
            alpha=alpha,
            # plot vos polygons
            plot_pcross=plot_pcross,
            plot_phor=plot_phor,
            plot_colorbar=plot_colorbar,
            # config
            plot_config=plot_config,
            # figure
            dax=dax,
            dmargin=dmargin,
            fs=fs,
            wintit=wintit,
            # interactivity
            color_dict=color_dict,
        )

    # --------------------------
    # plot geometrical converage
    # --------------------------

    def plot_diagnostic_geometrical_coverage(
        self,
        key=None,
        # mesh sampling
        key_mesh=None,
        res_RZ=None,
        nan0=None,
        # plotting options
        plot_config=None,
        dcolor=None,
        dax=None,
        fs=None,
        dmargin=None,
        tit=None,
        cmap=None,
        vmin=None,
        vmax=None,
    ):
        """ Plot the geometrical coverage of a diagnostic, in a cross-section


        Parameters
        ----------
        key : str, optional
            key to the diagnostic
        key_mesh : str, optional
            key to the mesh used for sampling the cross-section
        res_RZ : float / list, optional
            resolution for sampling the cross-section. The default is None.
        plot_config : Config, optional
            DESCRIPTION. The default is None.
        dcolor : dict, optional
            dict of color, per camera

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        return _plot_coverage.main(
            coll=self,
            key=key,
            # mesh sampling
            key_mesh=key_mesh,
            res_RZ=res_RZ,
            nan0=nan0,
            # plotting options
            config=plot_config,
            dcolor=dcolor,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            dax=dax,
            fs=fs,
            dmargin=dmargin,
            tit=tit,
        )

    # --------------------------
    # save to file
    # --------------------------

    def save_diagnostic_to_file(
        # ---------------
        # input from tofu
        self,
        key=None,
        key_cam=None,
        key_optics=None,
        # ---------------
        # options
        factor=None,
        color=None,
        # ---------------
        # saving
        pfe_save=None,
        overwrite=None,
    ):
        """ Save desired diagnostic to a json or stp file

        Parameters
        ----------
        key : str, optional
            key to the existing set of rays
        pfe : str, optional
            valid path-file-extension (file str) where to save the file

        Returns
        -------
        None.

        """

        return _saveload_from_file.save(
            # ---------------
            # input from tofu
            coll=self,
            key=key,
            key_cam=key_cam,
            key_optics=key_optics,
            # ---------------
            # options
            factor=factor,
            color=color,
            # ---------------
            # saving
            pfe_save=pfe_save,
            overwrite=overwrite,
        )

    def add_diagnostic_from_file(
        self,
        pfe=None,
        returnas=False,
    ):
        """ Adds a diagnostic instance (and necessary optics) from json file

        Parameters
        ----------
        pfe :     str
            path/file.ext to desired file
        returnas: bool
            whether to return the Collection instance

        Returns
        -------
        coll:    Collection instance
            optional

        """

        return _saveload_from_file.load(
            coll=self,
            pfe=pfe,
            returnas=returnas,
        )
