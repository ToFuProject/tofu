# -*- coding: utf-8 -*-


# Built-in


# Common
import numpy as np


# tofu
# from tofu import __version__ as __version__
import tofu.utils as utils
from . import _core_new
from . import _mesh_checks
from . import _mesh_comp
from . import _mesh_plot


_GROUP_MESH = 'mesh'
_GROUP_R = 'R'
_GROUP_Z = 'Z'


# #############################################################################
# #############################################################################
#                           Mesh2DRect
# #############################################################################


class Mesh2DRect(_core_new.DataCollection):

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
        key=None,
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

        domain = _mesh_checks._mesh2DRect_from_Config(
            config=config, key_struct=key_struct,
        )

        obj = cls()
        obj.add_mesh(domain=domain, res=res, key=key)
        return obj

    # -----------------
    # bsplines
    # ------------------

    def add_bsplines(self, key=None, deg=None):
        """ Add bspline basis functions on the chosen mesh """

        # --------------
        # check inputs

        key, deg = _mesh_checks._mesh2DRect_bsplines(
            key=key,
            lkeys=list(self.dobj[self._groupmesh].keys()),
            deg=deg,
        )

        # --------------
        # get bsplines

        dref, dobj = _mesh_comp._mesh2DRect_bsplines(
            mesh=self, key=key, deg=deg,
        )

        self.update(dobj=dobj, dref=dref)

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
            returnas=returnas,
        )

    def select_mesh_elements(
        self,
        key=None,
        ind=None,
        elements=None,
        returnas=None,
        return_neighbours=None,
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
            key=key, ind=ind, elements=elements, returnas=tuple,
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
    ):
        """ Return indices or values of selected knots / cent

        Can be used to convert tuple (R, Z) indices to flat (RZ,) indices
        Can return values instead of indices
        Can return indices / values of neighbourgs

        """
        lk = list(self.dobj['bsplines'].keys())
        if key is None and len(lk) == 1:
            key = lk[0]
        ind = self.select_ind(key=key, ind=ind, returnas=tuple)
        return _mesh_comp._select_bsplines(
            mesh=self,
            key=key,
            ind=ind,
            returnas=returnas,
            return_cents=return_cents,
            return_knots=return_knots,
        )

    # -----------------
    # interp tools
    # ------------------

    def get_sample_mesh(self, key=None, res=None, grid=None, mode=None):
        """ Return a sampled version of the chosen mesh """
        return _mesh_comp.sample_mesh(
            mesh=self,
            key=key,
            res=res,
            grid=grid,
            mode=mode,
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

    def interp(self, key=None, R=None, Z=None, grid=None, details=None):
        """ Interp desired data on pts """
        return _mesh_comp.interp(
            mesh=self,
            key=key,
            R=R,
            Z=Z,
            grid=grid,
            details=details,
        )

    # -----------------
    # plotting
    # ------------------

    def plot_mesh(
        self,
        key=None,
        ind_knot=None,
        ind_cent=None,
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
            cmap=cmap,
            dax=dax,
            dmargin=dmargin,
            fs=fs,
            dleg=dleg,
        )

    def plot_profile2d(
        self,
        key=None,
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
