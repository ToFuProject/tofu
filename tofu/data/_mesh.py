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


_GROUP_MESH = 'mesh'
_GROUP_R = 'R'
_GROUP_Z = 'Z'


# #############################################################################
# #############################################################################
#                           Mesh2DRect
# #############################################################################


class Mesh2DRect(DataCollection):

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
        key=None,
        domain=None,
        res=None,
    ):
        """ Add a mesh by key

        """

        dref, dmesh = _mesh_checks._mesh2DRect_to_dict(
            domain=domain,
            res=res,
            key=key,
        )
        dobj = {
            self._groupmesh: dmesh,
        }
        self.update(dref=dref, dobj=dobj)

    # -----------------
    # from config
    # ------------------

    @classmethod
    def from_Config(
        cls,
        config=None,
        key_struct=None,
        res=None,
        deg=None,
        key=None,
        thresh_in=None,
    ):
        """

        Example:
        --------
                >>> import tofu as tf
                >>> conf = tf.load_config('ITER')
                >>> mesh = tf.data.Mesh2DRect.from_Config(
                    config=conf,
                    res=[],
                )

        """

        domain, poly = _mesh_checks._mesh2DRect_from_Config(
            config=config, key_struct=key_struct,
        )

        obj = cls()
        obj.add_mesh(domain=domain, res=res, key=key)
        if deg is not None:
            obj.add_bsplines(deg=deg)
        obj.crop(key=key, crop=poly, thresh_in=thresh_in)
        return obj

    # -----------------
    # bsplines
    # ------------------

    def add_bsplines(self, key=None, deg=None):
        """ Add bspline basis functions on the chosen mesh """

        # --------------
        # check inputs

        keym, keybs, deg = _mesh_checks._mesh2DRect_bsplines(
            key=key,
            lkeys=list(self.dobj[self._groupmesh].keys()),
            deg=deg,
        )

        # --------------
        # get bsplines

        dref, dobj = _mesh_comp._mesh2DRect_bsplines(
            mesh=self, keym=keym, keybs=keybs, deg=deg,
        )

        self.update(dobj=dobj, dref=dref)
        _mesh_comp.add_cropbs_from_crop(mesh=self, keybs=keybs, keym=keym)

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
            mesh=self,
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
        lk = list(self.dobj[self._groupmesh].keys())
        if key is None and len(lk) == 1:
            key = lk[0]
        ind = self.select_ind(
            key=key, ind=ind, elements=elements, returnas=tuple, crop=crop,
        )
        return _mesh_comp._select_mesh(
            mesh=self,
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
        lk = list(self.dobj['bsplines'].keys())
        if key is None and len(lk) == 1:
            key = lk[0]
        ind = self.select_ind(key=key, ind=ind, returnas=tuple, crop=crop)
        return _mesh_comp._select_bsplines(
            mesh=self,
            key=key,
            ind=ind,
            returnas=returnas,
            return_cents=return_cents,
            return_knots=return_knots,
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
    ):
        """ Get a matrix operator to compute an integral

        operator specifies the integrand:
            - 'D0': integral of the value
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
        )

        # store
        if store is True:
            if operator in ['D0', 'D0N2']:
                name = f'{key}-{operator}-{geometry}'
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
                self.add_data(
                    key=name,
                    data=opmat[0],
                    ref=ref,
                    units='',
                    name=operator,
                    dim=dim,
                )
                name = f'{key}-{operator}-dZ-{geometry}'
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
                self.add_data(
                    key=name,
                    data=opmat[0],
                    ref=ref,
                    units='',
                    name=operator,
                    dim=dim,
                )
                name = f'{key}-{operator}-d2Z-{geometry}'
                self.add_data(
                    key=name,
                    data=opmat[1],
                    ref=ref,
                    units='',
                    name=operator,
                    dim=dim,
                )
                name = f'{key}-{operator}-dRZ-{geometry}'
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
            mesh=self,
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
            mesh=self,
            key=key,
            res=res,
            grid=grid,
            mode=mode,
        )
    """

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
            mesh=self,
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
    # crop
    # ------------------

    def crop(self, key=None, crop=None, thresh_in=None):
        """ Crop a mesh using

            - a mask of bool for each mesh elements
            - a 2d (R, Z) closed polygon

        If applied on a bspline, cropping is double-checked to make sure
        all remaining bsplines have full support domain
        """
        crop, key, thresh_in = _mesh_comp.crop(
            mesh=self,
            key=key,
            crop=crop,
            thresh_in=thresh_in,
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
                _mesh_comp.add_cropbs_from_crop(mesh=self, keybs=k0, keym=key)

    # -----------------
    # geometry matrix
    # ------------------

    def compute_geometry_matrix(
        self,
        key=None,
        cam=None,
        res=None,
        resMode=None,
        method=None,
        crop=None,
        name=None,
        verb=None,
    ):

        dref, ddata, dobj = _matrix_comp.compute(
            mesh=self,
            key=key,
            cam=cam,
            res=res,
            resMode=resMode,
            method=method,
            crop=crop,
            name=name,
            verb=verb,
        )

        return Matrix(dref=dref, ddata=ddata, dobj=dobj)

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
            mesh=self,
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
            mesh=self,
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
            mesh=self,
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


class Matrix(Mesh2DRect):

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
            matrix=self,
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
