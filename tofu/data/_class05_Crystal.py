# -*- coding: utf-8 -*-


# Built-in
# import copy


# tofu
from ._class04_Filter import Filter as Previous
from . import _class3_check
from . import _class5_check as _check
from . import _class5_compute as _compute
from . import _class5_coordinates as _coordinates
from . import _class5_reflections_pts2pt as _reflections_pts2pt
from . import _class5_reflections_ptsvect as _reflections_ptsvect
from . import _class5_projections as _projections
from . import _class5_plot as _plot


__all__ = ['Crystal']


# #############################################################################
# #############################################################################
#                           Diagnostic
# #############################################################################


class Crystal(Previous):

    _dshow = dict(Previous._dshow)
    _dshow.update({
        'crystal': [
            'dgeom.type',
            'dgeom.curve_r',
            'dgeom.area',
            'dmat.name',
            'dmat.miller',
            'dgeom.outline',
            'dgeom.poly',
            'dgeom.cent',
            'dmisc.color',
        ],
    })

    def add_crystal(
        self,
        key=None,
        # geometry
        dgeom=None,
        # material
        dmat=None,
        alpha=None,
        beta=None,
        # dmisc
        color=None,
    ):
        """ Add a crystal

        Can be defined from:
            - 2d outline + 3d center + unit vectors (nin, e0, e1)
            - 3d polygon + nin

        Unit vectors will be checked and normalized
        If planar, area will be computed
        Outline will be made counter-clockwise

        """

        # check / format input
        dref, ddata, dobj = _class3_check._add_surface3d(
            coll=self,
            key=key,
            which='crystal',
            which_short='cryst',
            # 2d outline
            **dgeom,
        )

        key = list(dobj['crystal'].keys())[0]

        # material
        dobj['crystal'][key]['dmat'] = _check._dmat(
            coll=self,
            key=key,
            dgeom=dobj['crystal'][key]['dgeom'],
            dmat=dmat,
            alpha=alpha,
            beta=beta,
        )

        # dmisc
        dobj['crystal'][key]['dmisc'] = _class3_check._dmisc(
            key=key,
            color=color,
        )

        # update dicts
        self.update(dref=dref, ddata=ddata, dobj=dobj)

        # compute rocking curve
        # if dmat['ready_to_compute'] is True:
        # self.set_crystal_rocking_curve()

    def set_crystal_rocking_curve(
        self,
        key=None,
        dbragg=None,
        intensity=None,
    ):
        return _compute.rocking_curve(
            coll=self,
            key=key,
        )

    # --------------------
    # Utilities
    # ---------------------

    def get_crystal_bragglamb(
        self,
        key=None,
        lamb=None,
        bragg=None,
        norder=None,
        rocking_curve=None,
    ):
        """ Return bragg angle

        If bragg provided, simply return bragg as np.ndarray

        If lamb is provided return corresponding:
            - bragg angle (simple bragg's law)
            - bragg angle + reflectivity, interpolated on rocking_curve

        """

        return _compute._bragglamb(
            coll=self,
            key=key,
            lamb=lamb,
            bragg=bragg,
            rocking_curve=rocking_curve,
        )

    # --------------------
    # Local vs xyz coordinates
    # ---------------------

    def get_optics_x01toxyz(
        self,
        key=None,
        asplane=None,
    ):
        """ Return a dict of formatted """
        return _coordinates._get_x01toxyz(
            coll=self,
            key=key,
            asplane=asplane,
        )

    # --------------------
    # Reflections
    # ---------------------

    def get_optics_reflect_pts2pt(self, key=None):
        """ Return a dict of formatted """
        return _reflections_pts2pt._get_pts2pt(
            coll=self,
            key=key,
        )

    def get_optics_reflect_ptsvect(
        self,
        key=None,
        asplane=None,
        isnorm=None,
        fast=None,
    ):
        """ Return a dict of formatted """
        return _reflections_ptsvect._get_ptsvect(
            coll=self,
            key=key,
            asplane=asplane,
            isnorm=isnorm,
            fast=fast,
        )

    # -------------------------------------------
    # Projections from point in local coordinates
    # -------------------------------------------

    def get_optics_project_poly_from_pt(self, key=None):
        """ Return a dict of formatted """
        return _projections._get_pts_from_pt(coll=self, key=key)

    # --------------------
    # ideal configurations
    # ---------------------

    def get_crystal_ideal_configuration(
        self,
        key=None,
        configuration=None,
        lamb=None,
        bragg=None,
        defocus=None,
        strict_aperture=None,
        # parameters
        cam_on_e0=None,
        # johann-specific
        cam_tangential=None,
        # pinhole-specific
        cam_dimensions=None,
        cam_distance=None,
        focal_distance=None,
        # store
        store=None,
        key_cam=None,
        key_aperture=None,
        aperture_dimensions=None,
        pinhole_radius=None,
        cam_pixels_nb=None,
        # returnas
        returnas=None,
    ):
        """ Return the ideal positions of other elements for a configuration

        'ideal' means:
            - maximizing focalization
            - for the chosen wavelength / bragg angle

        'johann': for spherical crystals only
            build the rowland circle and return the ideal detector position

        'von hamos': for cylindrical only
            build the axis and return
             - slit position
             - detector position given its height

        'pinhole': for cylindrical only
            buids the axis and return
             - pinhole position
             - detector position given its height
        """

        return _compute._ideal_configuration(
            coll=self,
            key=key,
            configuration=configuration,
            lamb=lamb,
            bragg=bragg,
            defocus=defocus,
            strict_aperture=strict_aperture,
            # parameters
            cam_on_e0=cam_on_e0,
            # johann-specific
            cam_tangential=cam_tangential,
            # pinhole-specific
            cam_dimensions=cam_dimensions,
            cam_distance=cam_distance,
            focal_distance=focal_distance,
            # store
            store=store,
            key_cam=key_cam,
            key_aperture=key_aperture,
            aperture_dimensions=aperture_dimensions,
            pinhole_radius=pinhole_radius,
            cam_pixels_nb=cam_pixels_nb,
            # returnas
            returnas=returnas,
        )

    # --------------------
    # plotting
    # ---------------------

    def plot_crystal_rocking_curve(
        self,
        key=None,
        # option
        T=None,
        # plotting
        dax=None,
        color=None,
        plot_FW=None,
    ):

        return _plot.plot_rocking_curve(
            coll=self,
            key=key,
            # option
            T=T,
            # plotting
            dax=dax,
            color=color,
            plot_FW=plot_FW,
        )
