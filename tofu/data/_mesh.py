# -*- coding: utf-8 -*-


# Built-in


# Common
import numpy as np


# tofu
# from tofu import __version__ as __version__
from . import _generic_check
from ._DataCollection_class import DataCollection
from . import _mesh_checks
from . import _mesh_comp
from . import _mesh_plot
from . import _matrix_comp
from . import _matrix_plot
from . import _inversions_comp
from . import _inversions_plot


_GROUP_MESH = 'mesh'
_GROUP_R = 'R'
_GROUP_Z = 'Z'


# #############################################################################
# #############################################################################
#                           Mesh2D
# #############################################################################


class Mesh2D(DataCollection):

    _ddef = {
        'Id': {'include': ['Mod', 'Cls', 'Name', 'version']},
        'params': {
            'lambda0': (float, 0.),
            'source': (str, 'unknown'),
            'transition':    (str, 'unknown'),
            'element':  (str, 'unknown'),
            'charge':  (int, 0),
            'ion':  (str, 'unknown'),
            'symbol':   (str, 'unknown'),
        },
    }
    _forced_group = [_GROUP_R, _GROUP_Z]
    _data_none = True

    _show_in_summary_core = ['shape', 'ref', 'group']
    _show_in_summary = 'all'

    _groupmesh = _GROUP_MESH
    _groupR = _GROUP_R
    _groupZ = _GROUP_Z

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
                >>> mesh = tf.data.Mesh2D()
                >>> mesh.add_mesh(config=conf, res=0.1, deg=1)

        """

        # get domain, poly from crop_poly
        if crop_poly is not None:
            domain, poly = _mesh_checks._mesh2DRect_from_croppoly(crop_poly)
        else:
            poly = None

        # check input data and get input dicts
        dref, ddata, dmesh = _mesh_checks._mesh2D_check(
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

        dobj = {
            self._groupmesh: dmesh,
        }

        # update dicts
        self.update(dref=dref, ddata=ddata, dobj=dobj)

        # optional bspline
        if deg is not None:
            self.add_bsplines(deg=deg, key=key)

        # optional cropping
        if self.dobj['mesh'][key]['type'] == 'rect' and poly is not None:
            self.crop(
                key=key,
                crop=poly,
                thresh_in=thresh_in,
                remove_isolated=remove_isolated,
            )

    # -----------------
    # bsplines
    # ------------------

    def add_bsplines(self, key=None, deg=None):
        """ Add bspline basis functions on the chosen mesh """

        # --------------
        # check inputs

        keym, keybs, deg = _mesh_checks._mesh2D_bsplines(
            key=key,
            lkeys=list(self.dobj['mesh'].keys()),
            deg=deg,
        )

        # --------------
        # get bsplines

        if self.dobj['mesh'][keym]['type'] == 'rect':
            dref, dobj = _mesh_comp._mesh2DRect_bsplines(
                coll=self, keym=keym, keybs=keybs, deg=deg,
            )
        else:
            dref, dobj = _mesh_comp._mesh2DTri_bsplines(
                coll=self, keym=keym, keybs=keybs, deg=deg,
            )

        # --------------
        # update dict and crop if relevant

        self.update(dobj=dobj, dref=dref)
        if self.dobj['mesh'][keym]['type'] == 'rect':
            _mesh_comp.add_cropbs_from_crop(coll=self, keybs=keybs, keym=keym)

    # -----------------
    # crop
    # ------------------

    def crop(self, key=None, crop=None, thresh_in=None, remove_isolated=None):
        """ Crop a mesh using

            - a mask of bool for each mesh elements
            - a 2d (R, Z) closed polygon

        If applied on a bspline, cropping is double-checked to make sure
        all remaining bsplines have full support domain
        """
        crop, key, thresh_in = _mesh_comp.crop(
            coll=self,
            key=key,
            crop=crop,
            thresh_in=thresh_in,
            remove_isolated=remove_isolated,
        )

        # add crop data
        keycrop = f'{key}-crop'
        self.add_data(
            key=keycrop,
            data=crop,
            ref=self.dobj['mesh'][key]['ref'],
            dim='bool',
            quant='bool',
        )

        # update obj
        self._dobj['mesh'][key]['crop'] = keycrop
        self._dobj['mesh'][key]['crop-thresh'] = thresh_in

        # also crop bsplines
        for k0 in self.dobj.get('bsplines', {}).keys():
            if self.dobj['bsplines'][k0]['mesh'] == key:
                _mesh_comp.add_cropbs_from_crop(coll=self, keybs=k0, keym=key)

    # -----------------
    # get data subset
    # ------------------

    def get_profiles2d(self):
        """ Return dict of profiles2d with associated bsplines as values """

        # dict of bsplines shapes
        dbs = {
            k0: v0['ref']
            for k0, v0 in self.dobj['bsplines'].items()
        }

        # dict of profiles2d
        dk = {
            k0: [k1 for k1, v1 in dbs.items() if v0['ref'][-2:] == v1][0]
            for k0, v0 in self.ddata.items()
            if len([k1 for k1, v1 in dbs.items() if v0['ref'][-2:] == v1]) == 1
        }
        dk.update({k0: k0 for k0 in dbs.keys()})

        return dk

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
        return _mesh_comp._select_ind(
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
        key = _generic_check._check_var(
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

        if self.dobj['mesh'][key]['type'] == 'rect':
            return _mesh_comp._select_mesh_rect(
                coll=self,
                key=key,
                ind=ind,
                elements=elements,
                returnas=returnas,
                return_neighbours=return_neighbours,
            )
        else:
            return _mesh_comp._select_mesh_tri(
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
        return _mesh_comp._select_bsplines(
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
        ) = _mesh_comp.get_bsplines_operator(
            self,
            key=key,
            operator=operator,
            geometry=geometry,
            crop=crop,
            store=store,
            returnas=returnas,
            # specific to deg = 0
            centered=centered,
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
        return _mesh_comp.sample_mesh(
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
        return _mesh_comp.sample_bsplines(
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
            ) = _DataCollection_check_inputs._get_possible_ref12d(
                dd=self._ddata,
                key=quant, ref1d=ref1d, ref2d=ref2d,
                group1d=group1d,
                group2d=group2d,
            )
            idq2dR, idq2dPhi, idq2dZ = None, None, None
            ani = False
        else:
            idq2dR, msg = _DataCollection_check_inputs._get_keyingroup_ddata(
                dd=self._ddata,
                key=q2dR, group=group2d, msgstr='quant', raise_=True,
            )
            idq2dPhi, msg = _DataCollection_check_inputs._get_keyingroup_ddata(
                dd=self._ddata,
                key=q2dPhi, group=group2d, msgstr='quant', raise_=True,
            )
            idq2dZ, msg = _DataCollection_check_inputs._get_keyingroup_ddata(
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

    def interp2d(
        self,
        key=None,
        R=None,
        Z=None,
        grid=None,
        indbs=None,
        indt=None,
        details=None,
        reshape=None,
        res=None,
        coefs=None,
        crop=None,
        nan0=None,
        imshow=None,
    ):
        """ Interp desired data on pts """
        return _mesh_comp.interp2d(
            coll=self,
            key=key,
            R=R,
            Z=Z,
            grid=grid,
            indbs=indbs,
            indt=indt,
            details=details,
            reshape=reshape,
            res=res,
            coefs=coefs,
            crop=crop,
            nan0=nan0,
            imshow=imshow,
        )

    # -----------------
    # geometry matrix
    # ------------------

    def add_geometry_matrix(
        self,
        key=None,
        key_chan=None,
        cam=None,
        res=None,
        resMode=None,
        method=None,
        crop=None,
        name=None,
        verb=None,
        store=None,
    ):

        return _matrix_comp.compute(
            coll=self,
            key=key,
            key_chan=key_chan,
            cam=cam,
            res=res,
            resMode=resMode,
            method=method,
            crop=crop,
            name=name,
            verb=verb,
            store=store,
        )

    # -----------------
    # inversions
    # ------------------

    def add_inversion(
        self,
        # input data
        key_matrix=None,
        key_data=None,
        key_sigma=None,
        data=None,
        sigma=None,
        # choice of algo
        isotropic=None,
        sparse=None,
        positive=None,
        cholesky=None,
        regparam_algo=None,
        algo=None,
        # regularity operator
        operator=None,
        geometry=None,
        # misc
        solver=None,
        conv_crit=None,
        chain=None,
        verb=None,
        store=None,
        # algo and solver-specific options
        kwdargs=None,
        method=None,
        options=None,
    ):
        """ Compute tomographic inversion

        """

        return _inversions_comp.compute_inversions(
            # input data
            coll=self,
            key_matrix=key_matrix,
            key_data=key_data,
            key_sigma=key_sigma,
            data=data,
            sigma=sigma,
            # choice of algo
            isotropic=isotropic,
            sparse=sparse,
            positive=positive,
            cholesky=cholesky,
            regparam_algo=regparam_algo,
            algo=algo,
            # regularity operator
            operator=operator,
            geometry=geometry,
            # misc
            conv_crit=conv_crit,
            chain=chain,
            verb=verb,
            store=store,
            # algo and solver-specific options
            kwdargs=kwdargs,
            method=method,
            options=options,
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
    ):

        return _mesh_plot.plot_mesh(
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
        )

    def plot_bsplines(
        self,
        key=None,
        ind=None,
        knots=None,
        cents=None,
        res=None,
        plot_mesh=None,
        cmap=None,
        dax=None,
        dmargin=None,
        fs=None,
        dleg=None,
    ):

        return _mesh_plot.plot_bspline(
            coll=self,
            key=key,
            ind=ind,
            knots=knots,
            cents=cents,
            res=res,
            plot_mesh=plot_mesh,
            cmap=cmap,
            dax=dax,
            dmargin=dmargin,
            fs=fs,
            dleg=dleg,
        )

    def plot_profile2d(
        self,
        key=None,
        coefs=None,
        indt=None,
        res=None,
        vmin=None,
        vmax=None,
        cmap=None,
        dax=None,
        dmargin=None,
        fs=None,
        dcolorbar=None,
        dleg=None,
    ):
        return _mesh_plot.plot_profile2d(
            coll=self,
            key=key,
            coefs=coefs,
            indt=indt,
            res=res,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            dax=dax,
            dmargin=dmargin,
            fs=fs,
            dcolorbar=dcolorbar,
            dleg=dleg,
        )

    def plot_geometry_matrix(
        self,
        cam=None,
        key=None,
        indbf=None,
        indchan=None,
        vmin=None,
        vmax=None,
        res=None,
        cmap=None,
        dax=None,
        dmargin=None,
        fs=None,
        dcolorbar=None,
        dleg=None,
    ):
        return _matrix_plot.plot_geometry_matrix(
            cam=cam,
            coll=self,
            key=key,
            indbf=indbf,
            indchan=indchan,
            vmin=vmin,
            vmax=vmax,
            res=res,
            cmap=cmap,
            dax=dax,
            dmargin=dmargin,
            fs=fs,
            dcolorbar=dcolorbar,
            dleg=dleg,
        )

    def plot_inversion(
        self,
        key=None,
        indt=None,
        vmin=None,
        vmax=None,
        res=None,
        cmap=None,
        dax=None,
        dmargin=None,
        fs=None,
        dcolorbar=None,
        dleg=None,
    ):

        return _inversions_plot.plot_inversion(
            coll=self,
            key=key,
            indt=indt,
            vmin=vmin,
            vmax=vmax,
            res=res,
            cmap=cmap,
            dax=dax,
            dmargin=dmargin,
            fs=fs,
            dcolorbar=dcolorbar,
            dleg=dleg,
        )
