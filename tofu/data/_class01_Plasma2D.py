# -*- coding: utf-8 -*-


# Built-in
import copy


# Common
import numpy as np
import datastock as ds


# tofu
# from tofu import __version__ as __version__
from ._class00_Config import Config as Previous
from . import _class1_checks as _checks
from . import _class1_compute as _compute
from . import _class1_plot as _plot


__all__ = ['Plasma2D']


_WHICH_MESH = 'mesh'
_QUANT_R = 'R'
_QUANT_Z = 'Z'


# #############################################################################
# #############################################################################
#                           Plasma2D
# #############################################################################


class Plasma2D(Previous):

    _ddef = copy.deepcopy(ds.DataStock._ddef)
    _ddef['params']['ddata'].update({
        'bsplines': (str, ''),
    })
    _ddef['params']['dobj'] = None
    _ddef['params']['dref'] = None

    # _show_in_summary_core = ['shape', 'ref', 'group']
    _show_in_summary = 'all'
    _dshow = dict(Previous._dshow)

    _which_mesh = _WHICH_MESH
    _quant_R = _QUANT_R
    _quant_Z = _QUANT_Z

    def add_mesh(
        self,
        # rectangular mesh
        key=None,
        domain=None,
        res=None,
        R=None,
        Z=None,
        # triangular mesh
        knots=None,
        cents=None,
        # cropping
        crop_poly=None,
        thresh_in=None,
        remove_isolated=None,
        # direct addition of bsplines
        deg=None,
        **kwdargs,
    ):
        """ Add a mesh by key and domain / resolution

        Can create a rectangular or triangular mesh:
            - rectangular: provide (domain, res) or (R, Z)
                - domain:
                - res:
                - R:
                - Z:
            - triangular:
                - knots: (nknots, 2) array of (R, Z) coordinates
                - cents: (ncents, 3 or 4) array of int indices

        Can optionally be cropped by a closed polygon crop_poly, that can be:
            - a (2, N) np.narray of (R, Z) coordinates
            - a tuple (Config, key_struct) to designate a struct poly

        Args thresh_in and remove_isolated control the level of cropping:
            - thresh_in:
            - remove_isolated:

        If deg is provided, immediately adds a bsplines

        Example:
        --------
                >>> import tofu as tf
                >>> conf = tf.load_config('ITER')
                >>> mesh = tf.data.Plasma2D()
                >>> mesh.add_mesh(config=conf, res=0.1, deg=1)

        """

        # get domain, poly from crop_poly
        if crop_poly is not None:
            domain, poly = _checks._mesh2DRect_from_croppoly(
                crop_poly=crop_poly,
                domain=domain,
            )
        else:
            poly = None

        # check input data and get input dicts
        dref, ddata, dmesh = _checks._mesh2D_check(
            coll=self,
            # rectangular
            domain=domain,
            res=res,
            R=R,
            Z=Z,
            # triangular
            knots=knots,
            cents=cents,
            trifind=None,
            # key
            key=key,
        )

        # add kwdargs
        key = list(dmesh.keys())[0]
        dmesh[key].update(**kwdargs)

        # define dobj['mesh']
        dobj = {
            self._which_mesh: dmesh,
        }

        # update data source
        for k0, v0 in ddata.items():
            ddata[k0]['source'] = kwdargs.get('source')

        # update dicts
        self.update(dref=dref, ddata=ddata, dobj=dobj)

        # optional bspline
        if deg is not None:
            self.add_bsplines(key=key, deg=deg)

        # optional cropping
        c0 = (
            self.dobj[self._which_mesh][key]['type'] == 'rect'
            and poly is not None
        )
        if c0:
            self.crop(
                key=key,
                crop=poly,
                thresh_in=thresh_in,
                remove_isolated=remove_isolated,
            )

    def add_mesh_polar(
        self,
        # polar mesh
        key=None,
        radius=None,
        angle=None,
        # Defined on
        radius2d=None,
        angle2d=None,
        # optional special points coordinates vs time
        O_pts=None,         # computed if not provided
        X_pts=None,         # unused
        strike_pts=None,    # unused
        # res for contour discontinuity of angle2d
        res=None,
        # parameters
        radius_dim=None,
        radius_quant=None,
        radius_name=None,
        radius_units=None,
        angle_dim=None,
        angle_quant=None,
        angle_name=None,
        # direct addition of bsplines
        deg=None,
        **kwdargs,
    ):
        """ Add a 2d polar mesh

        For now only includes radial mesh
        radius has to be backed-up by:
            - a radius quantity from a pre-existing rect or tri mesh
            - a function

        """

        # check input data and get input dicts
        dref, ddata, dmesh = _checks._mesh2D_polar_check(
            coll=self,
            # polar
            radius=radius,
            angle=angle,
            radius2d=radius2d,
            angle2d=angle2d,
            # parameters
            radius_dim=radius_dim,
            radius_quant=radius_quant,
            radius_name=radius_name,
            radius_units=radius_units,
            angle_dim=angle_dim,
            angle_quant=angle_quant,
            angle_name=angle_name,
            # key
            key=key,
        )

        # add kwdargs
        key = list(dmesh.keys())[0]
        dmesh[key].update(**kwdargs)

        # update data source
        for k0, v0 in ddata.items():
            ddata[k0]['source'] = kwdargs.get('source')

        # special treatment of radius2d
        assert O_pts is None
        drefO, ddataO, kR, kZ = _compute.radius2d_special_points(
            coll=self,
            key=dmesh[key]['radius2d'],
            keym0=key,
            res=res,
        )
        dref.update(drefO)
        ddata.update(ddataO)
        dmesh[key]['pts_O'] = (kR, kZ)

        # define dobj['mesh']
        dobj = {
            self._which_mesh: dmesh,
        }

        # update dicts
        self.update(dref=dref, ddata=ddata, dobj=dobj)

        # special treatment of angle2d
        if dmesh[key]['angle2d'] is not None:
            drefa, ddataa, kR, kZ = _compute.angle2d_zone(
                coll=self,
                key=dmesh[key]['angle2d'],
                keyrad2d=dmesh[key]['radius2d'],
                key_ptsO=dmesh[key]['pts_O'],
                res=res,
                keym0=key,
            )

            # update dicts
            self.update(dref=drefa, ddata=ddataa)
            if 'azone' in self.get_lparam(self._which_mesh):
                self.set_param(
                    key=key,
                    param='azone',
                    value=(kR, kZ),
                    which=self._which_mesh,
                )
            else:
                self.add_param(
                    'azone', value={key: (kR, kZ)}, which=self._which_mesh,
                )

        # optional bspline
        if deg is not None:
            self.add_bsplines(key=key, deg=deg)

    # -----------------
    # bsplines
    # ------------------

    def add_bsplines(self, key=None, deg=None, angle=None):
        """ Add bspline basis functions on the chosen mesh """

        # --------------
        # check inputs

        keym, keybs, deg = _checks._mesh2D_bsplines(
            key=key,
            lkeys=list(self.dobj[self._which_mesh].keys()),
            deg=deg,
        )

        # --------------
        # get bsplines

        if self.dobj[self._which_mesh][keym]['type'] == 'rect':
            dref, ddata, dobj = _compute._mesh2DRect_bsplines(
                coll=self, keym=keym, keybs=keybs, deg=deg,
            )
        elif self.dobj[self._which_mesh][keym]['type'] == 'tri':
            dref, ddata, dobj = _compute._mesh2DTri_bsplines(
                coll=self, keym=keym, keybs=keybs, deg=deg,
            )
        else:
            dref, ddata, dobj = _compute._mesh2Dpolar_bsplines(
                coll=self, keym=keym, keybs=keybs, deg=deg, angle=angle,
            )

        # --------------
        # update dict and crop if relevant

        self.update(dobj=dobj, ddata=ddata, dref=dref)
        if self.dobj[self._which_mesh][keym]['type'] == 'rect':
            _compute.add_cropbs_from_crop(
                coll=self,
                keybs=keybs,
                keym=keym,
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

        # if ddata => check ref for bsplines
        if ddata is not None:
            for k0, v0 in ddata.items():
                (
                    ddata[k0]['ref'], ddata[k0]['data'],
                ) = _checks.add_data_meshbsplines_ref(
                    ref=v0['ref'],
                    data=v0['data'],
                    dmesh=self._dobj.get(self._which_mesh),
                    dbsplines=self._dobj.get('bsplines'),
                )

        # update
        super().update(
            dobj=dobj,
            ddata=ddata,
            dref=dref,
            harmonize=harmonize,
        )

        # assign bsplines
        if self._dobj.get('bsplines') is not None:
            for k0, v0 in self._ddata.items():
                lbs = [
                    k1 for k1, v1 in self._dobj['bsplines'].items()
                    if v1['ref'] == tuple([
                        rr for rr in v0['ref']
                        if rr in v1['ref']
                    ])
                ]
                if len(lbs) == 0:
                    pass
                elif len(lbs) == 1:
                    self._ddata[k0]['bsplines'] = lbs[0]
                else:
                    msg = f"Multiple nsplines:\n{lbs}"
                    raise Exception(msg)

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

    # -----------------
    # crop
    # ------------------

    def crop(self, key=None, crop=None, thresh_in=None, remove_isolated=None):
        """ Crop a mesh / bspline

        Uses:
            - a mask of bool for each mesh elements
            - a 2d (R, Z) closed polygon

        If applied on a bspline, cropping is double-checked to make sure
        all remaining bsplines have full support domain
        """
        crop, key, thresh_in = _compute.crop(
            coll=self,
            key=key,
            crop=crop,
            thresh_in=thresh_in,
            remove_isolated=remove_isolated,
        )

        # add crop data
        keycrop = f'{key}-crop'
        ref = tuple([
            self._ddata[k0]['ref'][0]
            for k0 in self._dobj[self._which_mesh][key]['cents']
        ])
        self.add_data(
            key=keycrop,
            data=crop,
            ref=ref,
            dim='bool',
            quant='bool',
        )

        # update obj
        self._dobj[self._which_mesh][key]['crop'] = keycrop
        self._dobj[self._which_mesh][key]['crop-thresh'] = thresh_in

        # also crop bsplines
        for k0 in self.dobj.get('bsplines', {}).keys():
            if self.dobj['bsplines'][k0][self._which_mesh] == key:
                _compute.add_cropbs_from_crop(coll=self, keybs=k0, keym=key)

    # -----------------
    # get data subset
    # ------------------

    def get_profiles2d(self):
        """ Return dict of profiles2d with associated bsplines as values """

        # dict of profiles2d
        dk = {
            k0: v0['bsplines']
            for k0, v0 in self._ddata.items()
            if v0['bsplines'] != ''
        }
        dk.update({k0: k0 for k0 in self._dobj['bsplines'].keys()})

        return dk

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
            key=key,
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

    # -----------------
    # indices
    # ------------------

    def select_ind(
        self,
        key=None,
        ind=None,
        elements=None,
        returnas=None,
        crop=None,
    ):
        """ Return ind for selected key (mesh or bspline) as:
                - tuple (default)
                - 'flat'

        Can covert one into the other
        """
        return _compute._select_ind(
            coll=self,
            key=key,
            ind=ind,
            elements=elements,
            returnas=returnas,
            crop=crop,
        )

    def select_mesh_elements(
        self,
        key=None,
        ind=None,
        elements=None,
        returnas=None,
        return_neighbours=None,
        crop=None,
    ):
        """ Return indices or values of selected knots / cent

        Can be used to convert tuple (R, Z) indices to flat (RZ,) indices
        Can return values instead of indices
        Can return indices / values of neighbourgs

        """
        # check key
        key = ds._generic_check._check_var(
            key, 'key',
            allowed=list(self.dobj.get('mesh', {}).keys()),
            types=str,
        )

        # get ind
        if self.dobj['mesh'][key]['type'] == 'rect':
            returnas_ind = tuple
        else:
            returnas_ind = bool

        ind = self.select_ind(
            key=key,
            ind=ind,
            elements=elements,
            returnas=returnas_ind,
            crop=crop,
        )

        return _compute._select_mesh(
            coll=self,
            key=key,
            ind=ind,
            elements=elements,
            returnas=returnas,
            return_neighbours=return_neighbours,
        )

    def select_bsplines(
        self,
        key=None,
        ind=None,
        returnas=None,
        return_cents=None,
        return_knots=None,
        crop=None,
    ):
        """ Return indices or values of selected knots / cent

        Can be used to convert tuple (R, Z) indices to flat (RZ,) indices
        Can return values instead of indices
        Can return indices / values of neighbourgs

        """
        return _compute._select_bsplines(
            coll=self,
            key=key,
            ind=ind,
            returnas=returnas,
            return_cents=return_cents,
            return_knots=return_knots,
            crop=crop,
        )

    # -----------------
    # Integration operators
    # ------------------

    def add_bsplines_operator(
        self,
        key=None,
        operator=None,
        geometry=None,
        crop=None,
        store=None,
        returnas=None,
        # specific to deg = 0
        centered=None,
        # to return gradR, gradZ, for D1N2 deg 0, for tomotok
        returnas_element=None,
    ):
        """ Get a matrix operator to compute an integral

        operator specifies the integrand:
            - 'D0N1': integral of the value
            - 'D0N2': integral of the squared value
            - 'D1N2': integral of the squared gradient
            - 'D2N2': integral of the squared laplacian

        geometry specifies in which geometry:
            - 'linear': linear geometry (cross-section = surface)
            - 'toroidal': toroildal geometry (cross-section = volumic slice)

        """

        (
            opmat, operator, geometry, dim, ref, crop,
            store, returnas, key,
        ) = _compute.get_bsplines_operator(
            self,
            key=key,
            operator=operator,
            geometry=geometry,
            crop=crop,
            store=store,
            returnas=returnas,
            # specific to deg = 0
            centered=centered,
            # to return gradR, gradZ, for D1N2 deg 0, for tomotok
            returnas_element=returnas_element,
        )

        # store
        if store is True:
            if operator == 'D1':
                name = f'{key}-{operator}-dR'
                if crop is True:
                    name = f'{name}-cropped'
                self.add_data(
                    key=name,
                    data=opmat[0],
                    ref=ref,
                    units='',
                    name=operator,
                    dim=dim,
                )
                name = f'{key}-{operator}-dZ'
                if crop is True:
                    name = f'{name}-cropped'
                self.add_data(
                    key=name,
                    data=opmat[1],
                    ref=ref,
                    units='',
                    name=operator,
                    dim=dim,
                )

            elif operator in ['D0N1', 'D0N2']:
                name = f'{key}-{operator}-{geometry}'
                if crop is True:
                    name = f'{name}-cropped'
                self.add_data(
                    key=name,
                    data=opmat,
                    ref=ref,
                    units='',
                    name=operator,
                    dim=dim,
                )
            elif operator == 'D1N2':
                name = f'{key}-{operator}-dR-{geometry}'
                if crop is True:
                    name = f'{name}-cropped'
                self.add_data(
                    key=name,
                    data=opmat[0],
                    ref=ref,
                    units='',
                    name=operator,
                    dim=dim,
                )
                name = f'{key}-{operator}-dZ-{geometry}'
                if crop is True:
                    name = f'{name}-cropped'
                self.add_data(
                    key=name,
                    data=opmat[1],
                    ref=ref,
                    units='',
                    name=operator,
                    dim=dim,
                )
            elif operator == 'D2N2':
                name = f'{key}-{operator}-d2R-{geometry}'
                if crop is True:
                    name = f'{name}-cropped'
                self.add_data(
                    key=name,
                    data=opmat[0],
                    ref=ref,
                    units='',
                    name=operator,
                    dim=dim,
                )
                name = f'{key}-{operator}-d2Z-{geometry}'
                if crop is True:
                    name = f'{name}-cropped'
                self.add_data(
                    key=name,
                    data=opmat[1],
                    ref=ref,
                    units='',
                    name=operator,
                    dim=dim,
                )
                name = f'{key}-{operator}-dRZ-{geometry}'
                if crop is True:
                    name = f'{name}-cropped'
                self.add_data(
                    key=name,
                    data=opmat[2],
                    ref=ref,
                    units='',
                    name=operator,
                    dim=dim,
                )
            else:
                msg = "Unknown opmat type!"
                raise Exception(msg)

        # return
        if returnas is True:
            return opmat, operator, geometry, dim, ref, crop

    # -----------------
    # interp tools
    # ------------------

    def get_sample_mesh(
        self,
        key=None,
        res=None,
        grid=None,
        mode=None,
        R=None,
        Z=None,
        DR=None,
        DZ=None,
        imshow=None,
    ):
        """ Return a sampled version of the chosen mesh """
        return _compute.sample_mesh(
            coll=self,
            key=key,
            res=res,
            grid=grid,
            mode=mode,
            R=R,
            Z=Z,
            DR=DR,
            DZ=DZ,
            imshow=imshow,
        )


    """
    def get_sample_bspline(self, key=None, res=None, grid=None, mode=None):
        return _compute.sample_bsplines(
            coll=self,
            key=key,
            res=res,
            grid=grid,
            mode=mode,
        )
    """

    def _check_qr12RPZ(
        self,
        quant=None,
        ref1d=None,
        ref2d=None,
        q2dR=None,
        q2dPhi=None,
        q2dZ=None,
        group1d=None,
        group2d=None,
    ):

        if group1d is None:
            group1d = self._group1d
        if group2d is None:
            group2d = self._group2d

        lc0 = [quant is None, ref1d is None, ref2d is None]
        lc1 = [q2dR is None, q2dPhi is None, q2dZ is None]
        if np.sum([all(lc0), all(lc1)]) != 1:
            msg = (
                "Please provide either (xor):\n"
                + "\t- a scalar field (isotropic emissivity):\n"
                + "\t\tquant : scalar quantity to interpolate\n"
                + "\t\t\tif quant is 1d, intermediate reference\n"
                + "\t\t\tfields are necessary for 2d interpolation\n"
                + "\t\tref1d : 1d reference field on which to interpolate\n"
                + "\t\tref2d : 2d reference field on which to interpolate\n"
                + "\t- a vector (R,Phi,Z) field (anisotropic emissivity):\n"
                + "\t\tq2dR :  R component of the vector field\n"
                + "\t\tq2dPhi: R component of the vector field\n"
                + "\t\tq2dZ :  Z component of the vector field\n"
                + "\t\t=> all components have the same time and mesh!\n"
            )
            raise Exception(msg)

        # Check requested quant is available in 2d or 1d
        if all(lc1):
            (
                idquant, idref1d, idref2d,
            ) = _compute._get_possible_ref12d(
                dd=self._ddata,
                key=quant, ref1d=ref1d, ref2d=ref2d,
                group1d=group1d,
                group2d=group2d,
            )
            idq2dR, idq2dPhi, idq2dZ = None, None, None
            ani = False
        else:
            idq2dR, msg = _compute._get_keyingroup_ddata(
                dd=self._ddata,
                key=q2dR, group=group2d, msgstr='quant', raise_=True,
            )
            idq2dPhi, msg = _compute._get_keyingroup_ddata(
                dd=self._ddata,
                key=q2dPhi, group=group2d, msgstr='quant', raise_=True,
            )
            idq2dZ, msg = _compute._get_keyingroup_ddata(
                dd=self._ddata,
                key=q2dZ, group=group2d, msgstr='quant', raise_=True,
            )
            idquant, idref1d, idref2d = None, None, None
            ani = True
        return idquant, idref1d, idref2d, idq2dR, idq2dPhi, idq2dZ, ani

    def _interp_pts2d_to_quant1d(
        self,
        pts=None,
        vect=None,
        t=None,
        quant=None,
        ref1d=None,
        ref2d=None,
        q2dR=None,
        q2dPhi=None,
        q2dZ=None,
        interp_t=None,
        interp_space=None,
        fill_value=None,
        Type=None,
        group0d=None,
        group1d=None,
        group2d=None,
        return_all=None,
    ):
        """ Return the value of the desired 1d quantity at 2d points

        For the desired inputs points (pts):
            - pts are in (X, Y, Z) coordinates
            - space interpolation is linear on the 1d profiles
        At the desired input times (t):
            - using a nearest-neighbourg approach for time

        """
        # Check inputs
        if group0d is None:
            group0d = self._group0d
        if group1d is None:
            group1d = self._group1d
        if group2d is None:
            group2d = self._group2d
        # msg = "Only 'nearest' available so far for interp_t!"
        # assert interp_t == 'nearest', msg

        # Check requested quant is available in 2d or 1d
        idquant, idref1d, idref2d, idq2dR, idq2dPhi, idq2dZ, ani = \
                self._check_qr12RPZ(
                    quant=quant, ref1d=ref1d, ref2d=ref2d,
                    q2dR=q2dR, q2dPhi=q2dPhi, q2dZ=q2dZ,
                    group1d=group1d, group2d=group2d,
                )

        # Check the pts is (3,...) array of floats
        idmesh = None
        if pts is None:
            # Identify mesh to get default points
            if ani:
                idmesh = [id_ for id_ in self._ddata[idq2dR]['ref']
                          if self._dref[id_]['group'] == group2d][0]
            else:
                if idref1d is None:
                    idmesh = [id_ for id_ in self._ddata[idquant]['ref']
                              if self._dref[id_]['group'] == group2d][0]
                else:
                    idmesh = [id_ for id_ in self._ddata[idref2d]['ref']
                              if self._dref[id_]['group'] == group2d][0]

            # Derive pts
            pts = self._get_pts_from_mesh(key=idmesh)

        pts = np.atleast_2d(pts)
        if pts.shape[0] != 3:
            msg = (
                "pts must be np.ndarray of (X,Y,Z) points coordinates\n"
                + "Can be multi-dimensional, but 1st dimension is (X,Y,Z)\n"
                + "    - Expected shape : (3,...)\n"
                + "    - Provided shape : {}".format(pts.shape)
            )
            raise Exception(msg)

        # Check t
        lc = [t is None, type(t) is str, type(t) is np.ndarray]
        assert any(lc)
        if lc[1]:
            assert t in self._ddata.keys()
            t = self._ddata[t]['data']

        # Interpolation (including time broadcasting)
        # this is the second slowest step (~0.08 s)
        func = self._get_finterp(
            idquant=idquant, idref1d=idref1d, idref2d=idref2d,
            idq2dR=idq2dR, idq2dPhi=idq2dPhi, idq2dZ=idq2dZ,
            idmesh=idmesh,
            interp_t=interp_t, interp_space=interp_space,
            fill_value=fill_value, ani=ani, Type=Type,
            group0d=group0d, group2d=group2d,
        )

        # Check vect of ani
        c0 = (
            ani is True
            and (
                vect is None
                or not (
                    isinstance(vect, np.ndarray)
                    and vect.shape == pts.shape
                )
            )
        )
        if c0:
            msg = (
                "Anisotropic field interpolation needs a field of local vect\n"
                + "  => Please provide vect as (3, npts) np.ndarray!"
            )
            raise Exception(msg)

        # This is the slowest step (~1.8 s)
        val, t = func(pts, vect=vect, t=t)

        # return
        if return_all is None:
            return_all = True
        if return_all is True:
            dout = {
                't': t,
                'pts': pts,
                'ref1d': idref1d,
                'ref2d': idref2d,
                'q2dR': idq2dR,
                'q2dPhi': idq2dPhi,
                'q2dZ': idq2dZ,
                'interp_t': interp_t,
                'interp_space': interp_space,
            }
            return val, dout
        else:
            return val

    def _interp_pts2d_to_quant1d(
        self,
        pts=None,
        vect=None,
        t=None,
        quant=None,
        ref1d=None,
        ref2d=None,
        q2dR=None,
        q2dPhi=None,
        q2dZ=None,
        interp_t=None,
        interp_space=None,
        fill_value=None,
        Type=None,
        group0d=None,
        group1d=None,
        group2d=None,
        return_all=None,
    ):
        """ Return the value of the desired 1d quantity at 2d points

        For the desired inputs points (pts):
            - pts are in (X, Y, Z) coordinates
            - space interpolation is linear on the 1d profiles
        At the desired input times (t):
            - using a nearest-neighbourg approach for time

        """
        # Check inputs
        if group0d is None:
            group0d = self._group0d
        if group1d is None:
            group1d = self._group1d
        if group2d is None:
            group2d = self._group2d
        # msg = "Only 'nearest' available so far for interp_t!"
        # assert interp_t == 'nearest', msg

        # Check requested quant is available in 2d or 1d
        idquant, idref1d, idref2d, idq2dR, idq2dPhi, idq2dZ, ani = \
                self._check_qr12RPZ(
                    quant=quant, ref1d=ref1d, ref2d=ref2d,
                    q2dR=q2dR, q2dPhi=q2dPhi, q2dZ=q2dZ,
                    group1d=group1d, group2d=group2d,
                )

        # Check the pts is (3,...) array of floats
        idmesh = None
        if pts is None:
            # Identify mesh to get default points
            if ani:
                idmesh = [id_ for id_ in self._ddata[idq2dR]['ref']
                          if self._dref[id_]['group'] == group2d][0]
            else:
                if idref1d is None:
                    idmesh = [id_ for id_ in self._ddata[idquant]['ref']
                              if self._dref[id_]['group'] == group2d][0]
                else:
                    idmesh = [id_ for id_ in self._ddata[idref2d]['ref']
                              if self._dref[id_]['group'] == group2d][0]

            # Derive pts
            pts = self._get_pts_from_mesh(key=idmesh)

        pts = np.atleast_2d(pts)
        if pts.shape[0] != 3:
            msg = (
                "pts must be np.ndarray of (X,Y,Z) points coordinates\n"
                + "Can be multi-dimensional, but 1st dimension is (X,Y,Z)\n"
                + "    - Expected shape : (3,...)\n"
                + "    - Provided shape : {}".format(pts.shape)
            )
            raise Exception(msg)

        # Check t
        lc = [t is None, type(t) is str, type(t) is np.ndarray]
        assert any(lc)
        if lc[1]:
            assert t in self._ddata.keys()
            t = self._ddata[t]['data']

        # Interpolation (including time broadcasting)
        # this is the second slowest step (~0.08 s)
        func = self._get_finterp(
            idquant=idquant, idref1d=idref1d, idref2d=idref2d,
            idq2dR=idq2dR, idq2dPhi=idq2dPhi, idq2dZ=idq2dZ,
            idmesh=idmesh,
            interp_t=interp_t, interp_space=interp_space,
            fill_value=fill_value, ani=ani, Type=Type,
            group0d=group0d, group2d=group2d,
        )

        # Check vect of ani
        c0 = (
            ani is True
            and (
                vect is None
                or not (
                    isinstance(vect, np.ndarray)
                    and vect.shape == pts.shape
                )
            )
        )
        if c0:
            msg = (
                "Anisotropic field interpolation needs a field of local vect\n"
                + "  => Please provide vect as (3, npts) np.ndarray!"
            )
            raise Exception(msg)

        # This is the slowest step (~1.8 s)
        val, t = func(pts, vect=vect, t=t)

        # return
        if return_all is None:
            return_all = True
        if return_all is True:
            dout = {
                't': t,
                'pts': pts,
                'ref1d': idref1d,
                'ref2d': idref2d,
                'q2dR': idq2dR,
                'q2dPhi': idq2dPhi,
                'q2dZ': idq2dZ,
                'interp_t': interp_t,
                'interp_space': interp_space,
            }
            return val, dout
        else:
            return val

    def interpolate_profile2d(
        # ressources
        self,
        # interpolation base, 1d or 2d
        key=None,
        # external coefs (optional)
        coefs=None,
        # interpolation points
        R=None,
        Z=None,
        radius=None,
        angle=None,
        grid=None,
        radius_vs_time=None,
        azone=None,
        # time: t or indt
        t=None,
        indt=None,
        indt_strict=None,
        # bsplines
        indbs=None,
        # parameters
        details=None,
        reshape=None,
        res=None,
        crop=None,
        nan0=None,
        val_out=None,
        imshow=None,
        return_params=None,
        # storing
        store=None,
        inplace=None,
    ):
        """ Interpolate desired profile2d (i.e.: data on bsplines)

        Interpolate:
            - key: a data on bsplines
            - coefs: external-provided set of coefs

        coefs can only be provided if:
            - details = False
            - key = keybs
            - coefs is a scalar or has shape = shapebs

        At points:
            - R:  R coordinates (np.ndarray or scalar)
            - Z:  Z coordinates (np.ndarray, same shape as R, or scalar)
            - grid: bool, if True mesh R x Z
            - indt: if provided, only interpolate at desired time indices

        With options:
            - details: bool, if True returns value for each bspline
            - indbs:   optional, select bsplines for which to interpolate
            - reshape: bool,
            - res:  optional, resolution to generate R and Z if they are None
            - crop: bool, whether to use the cropped mesh
            - nan0: value for out-of-mesh points
            - imshow: bool, whether to return as imshow (transpose)
            - return_params: bool, whether to return dict of input params

        """

        return _compute.interp2d(
            # ressources
            coll=self,
            # interpolation base, 1d or 2d
            key=key,
            # external coefs (optional)
            coefs=coefs,
            # interpolation points
            R=R,
            Z=Z,
            radius=radius,
            angle=angle,
            grid=grid,
            radius_vs_time=radius_vs_time,
            azone=azone,
            # time: t or indt
            t=t,
            indt=indt,
            indt_strict=indt_strict,
            # bsplines
            indbs=indbs,
            # parameters
            details=details,
            reshape=reshape,
            res=res,
            crop=crop,
            nan0=nan0,
            val_out=val_out,
            imshow=imshow,
            return_params=return_params,
            # storing
            store=store,
            inplace=inplace,
        )

    # TBF after polar meshes
    def interpolate_2dto1d(
        # resources
        self,
        # interpolation base
        key1d=None,
        key2d=None,
        # interpolation pts
        R=None,
        Z=None,
        grid=None,
        # parameters
        interp_t=None,
        fill_value=None,
        ani=False,
    ):

        return _compute.interp2dto1d(
            coll=self,
            key1d=key1d,
            key2d=key2d,
            R=R,
            Z=Z,
            grid=grid,
            crop=crop,
            nan0=nan0,
            return_params=return_params,
        )

    # -----------------
    # plotting
    # ------------------

    def plot_mesh(
        self,
        key=None,
        ind_knot=None,
        ind_cent=None,
        crop=None,
        color=None,
        dax=None,
        dmargin=None,
        fs=None,
        dleg=None,
        connect=None,
    ):

        return _plot.plot_mesh(
            coll=self,
            key=key,
            ind_knot=ind_knot,
            ind_cent=ind_cent,
            crop=crop,
            color=color,
            dax=dax,
            dmargin=dmargin,
            fs=fs,
            dleg=dleg,
            connect=connect,
        )

    def plot_bsplines(
        self,
        key=None,
        indbs=None,
        indt=None,
        knots=None,
        cents=None,
        res=None,
        plot_mesh=None,
        val_out=None,
        nan0=None,
        cmap=None,
        dax=None,
        dmargin=None,
        fs=None,
        dleg=None,
    ):

        return _plot.plot_bspline(
            coll=self,
            key=key,
            indbs=indbs,
            indt=indt,
            knots=knots,
            cents=cents,
            res=res,
            plot_mesh=plot_mesh,
            val_out=val_out,
            nan0=nan0,
            cmap=cmap,
            dax=dax,
            dmargin=dmargin,
            fs=fs,
            dleg=dleg,
        )

    def plot_profile2d(
        self,
        # inputs
        key=None,
        coefs=None,
        indt=None,
        res=None,
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
        connect=None,
    ):
        return _plot.plot_profile2d(
            coll=self,
            # inputs
            key=key,
            coefs=coefs,
            indt=indt,
            res=res,
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
            dinc=dinc,
            connect=connect,
        )
