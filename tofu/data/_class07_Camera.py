# -*- coding: utf-8 -*-


# Built-in
import copy


# Common
import numpy as np
import datastock as ds


# tofu
from ._class06_Grating import Grating as Previous
from . import _class3_check
from . import _class7_check as _check
from . import _class7_compute as _compute
from . import _class07_legacy as _legacy


__all__ = ['Camera']


# ################################################################
# ################################################################
#                           Diagnostic
# ################################################################


class Camera(Previous):

    _ddef = copy.deepcopy(Previous._ddef)
    _ddef['params']['ddata'].update({
          'camera': {'cls': str, 'def': ''},
    })

    _dshow = dict(Previous._dshow)
    _dshow.update({
        'camera': [
            'dgeom.type',
            'dgeom.nd',
            'dmat.mode',
            'dgeom.parallel',
            'dgeom.shape',
            'dgeom.ref',
            'dgeom.pix_area',
            'dgeom.pix_nb',
            'dgeom.outline',
            'dgeom.cent',
            'dgeom.cents',
            'dmat.qeff_energy',
            'dmat.qeff',
            # 'dmisc.color',
        ],
    })

    def _add_camera(
        self,
        dref=None,
        ddata=None,
        dobj=None,
        dmat=None,
        color=None,
    ):
        key = list(dobj['camera'].keys())[0]

        # material
        dref2, ddata2, dmat = _check._dmat(
            coll=self,
            key=key,
            dmat=dmat,
        )

        if dmat is not None:
            if dref2 is not None:
                dref.update(dref2)
                ddata.update(ddata2)
            dobj['camera'][key]['dmat'] = dmat

        # dmisc
        dobj['camera'][key]['dmisc'] = _class3_check._dmisc(
            key=key,
            color=color,
        )

        # update dicts
        self.update(dref=dref, ddata=ddata, dobj=dobj)

    def add_camera_1d(
        self,
        key=None,
        # geometry
        dgeom=None,
        # quantum efficiency
        dmat=None,
        # dmisc
        color=None,
    ):
        """ add a 1d camera

        A 1d camera is an unordered set of pixels of indentical outline
        Its geometry os defined by dgeom
        Its material properties (i.e: quantum efficiency) in dmat

        The geometry in dgeom must contain:
            - 'outline_x0': 1st coordinate of planar outline of a single pixel
            - 'outline_x1': 1st coordinate of planar outline of a single pixel
            - 'cents_x': x coordinate of the centers of ll pixels
            - 'cents_y': y coordinate of the centers of ll pixels
            - 'cents_z': z coordinate of the centers of ll pixels
            - 'nin_x': x coordinate of inward normal unit vector of all pixels
            - 'nin_y': y coordinate of inward normal unit vector of all pixels
            - 'nin_z': z coordinate of inward normal unit vector of all pixels
            - 'e0_x': x coordinate of e0 unit vector of all pixels
            - 'e0_y': y coordinate of e0 unit vector of all pixels
            - 'e0_z': z coordinate of e0 unit vector of all pixels
            - 'e1_x': x coordinate of e1 unit vector of all pixels
            - 'e1_y': y coordinate of e1 unit vector of all pixels
            - 'e1_z': z coordinate of e1 unit vector of all pixels

        The material dict, dmat can contain:
            - 'energy': a 1d energy vector , in eV
            - 'qeff': a 1d vector, same size as energy, with values in [0; 1]

        """
        # check / format input
        dref, ddata, dobj = _check._camera_1d(
            coll=self,
            key=key,
            **dgeom,
        )

        # add generic parts
        self._add_camera(
            dref=dref,
            ddata=ddata,
            dobj=dobj,
            dmat=dmat,
            color=color,
        )

    def add_camera_2d(
        self,
        key=None,
        # geometry
        dgeom=None,
        # material
        dmat=None,
        # dmisc
        color=None,
    ):
        """ add a 2d camera

        A 2d camera is an ordered 2d grid of pixels of indentical outline
        Its geometry os defined by dgeom
        Its material properties (i.e: quantum efficiency) in dmat

        The geometry in dgeom must contain:
            - 'outline_x0': 1st coordinate of planar outline of a single pixel
            - 'outline_x1': 1st coordinate of planar outline of a single pixel
            - 'cent': (x, y, z) coordinate of the center of the camera
            - 'cents_x0': x0 coordinate of the centers of all pixels
            - 'cents_x1': x1 coordinate of the centers of all pixels
            - 'nin': x coordinate of inward normal unit vector of all pixels
            - 'e0': x coordinate of e0 unit vector of all pixels
            - 'e1': x coordinate of e1 unit vector of all pixels

        The material dict, dmat can contain:
            - 'energy': a 1d energy vector , in eV
            - 'qeff': a 1d vector, same size as energy, with values in [0; 1]

        """
        # check / format input
        dref, ddata, dobj = _check._camera_2d(
            coll=self,
            key=key,
            **dgeom,
        )

        # add generic parts
        self._add_camera(
            dref=dref,
            ddata=ddata,
            dobj=dobj,
            dmat=dmat,
            color=color,
        )

    # ---------------
    # add pinhole cameras
    # ---------------

    def add_camera_pinhole(
        self,
        key=None,
        key_pinhole=None,
        key_diag=None,
        cam_type=None,
        # position
        x=None,
        y=None,
        R=None,
        z=None,
        phi=None,
        # orientation
        theta=None,
        dphi=None,
        tilt=None,
        # camera
        focal=None,
        pix_nb=None,
        pix_size=None,
        pix_spacing=None,
        # pinhole
        pinhole_radius=None,
        pinhole_size=None,
        # reflections
        reflections_nb=None,
        reflections_type=None,
        # diagnostic
        compute=None,
        config=None,
        length=None,
        # dmat
        dmat=None,
    ):

        return _compute.add_camera_pinhole(
            coll=self,
            key=key,
            key_pinhole=key_pinhole,
            key_diag=key_diag,
            cam_type=cam_type,
            # position
            x=x,
            y=y,
            R=R,
            z=z,
            phi=phi,
            # orientation
            theta=theta,
            dphi=dphi,
            tilt=tilt,
            # camera
            focal=focal,
            pix_nb=pix_nb,
            pix_size=pix_size,
            pix_spacing=pix_spacing,
            # pinhole
            pinhole_radius=pinhole_radius,
            pinhole_size=pinhole_size,
            # reflections
            reflections_nb=reflections_nb,
            reflections_type=reflections_type,
            # diagnostic
            compute=compute,
            config=config,
            length=length,
            # dmat
            dmat=dmat,
        )

    # -----------------
    # add_data
    # ------------------

    def update(
        self,
        dobj=None,
        ddata=None,
        dref=None,
        harmonize=None,
    ):
        """ Overload datastock update() method """

        # update
        super().update(
            dobj=dobj,
            ddata=ddata,
            dref=dref,
            harmonize=harmonize,
        )

        # assign diagnostic
        if self._dobj.get('camera') is not None:
            for k0, v0 in self._ddata.items():
                lcam = [
                    k1 for k1, v1 in self._dobj['camera'].items()
                    if v1['dgeom']['ref'] == tuple([
                        rr for rr in v0['ref']
                        if rr in v1['dgeom']['ref']
                    ])
                ]

                if len(lcam) == 0:
                    pass
                elif len(lcam) == 1:
                    self._ddata[k0]['camera'] = lcam[0]
                else:
                    msg = f"Multiple cameras:\n{lcam}"
                    raise Exception(msg)

    # --------------------
    # Legacy utility
    # ---------------------

    def add_camera_from_legacy(
        self,
        cam=None,
        key=None,
    ):
        """ Add a camera from a dict (or pfe) """
        return _legacy.add_camera(
            self,
            cam=cam,
            key=key,
        )

    # ---------------
    # utilities
    # ---------------

    def get_camera_unit_vectors(self, key=None):
        """ Return unit vectors components as dict """
        return _check.get_camera_unitvectors(
            coll=self,
            key=key,
        )

    def get_camera_dxyz(self, key=None, include_center=None):
        """ Return dx, dy, dz to get the outline from any pixel center
        Only works on 2d or parallel cameras

        """
        return _check.get_camera_dxyz(
            coll=self,
            key=key,
            include_center=include_center,
        )

    def get_camera_cents_xyz(self, key=None):
        """ Return cents_x, cents_y, cents_z """
        return _check.get_camera_cents_xyz(
            coll=self,
            key=key,
        )

    def get_as_dict(self, key=None):
        """ Return the desired object as a dict (input to some routines) """

        return _class3_check._return_as_dict(
            coll=self,
            key=key,
        )