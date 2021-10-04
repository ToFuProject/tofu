
# -*- coding: utf-8 -*-

# Built-in
import sys
import os
# import itertools as itt
import copy
import warnings
if sys.version[0] == '3':
    import inspect
else:
    # Python 2 back-porting
    import funcsigs as inspect

# Common
import numpy as np

# tofu
from tofu import __version__ as __version__
import tofu.pathfile as tfpf
import tofu.utils as utils
from ._DataCollection_class import DataCollection


__all__ = ['DataCollectPlot']
_SAVEPATH = os.path.abspath('./')


#############################################
#############################################
#       Matplotlib
#############################################
#############################################


class DataCollectionPlot_mpl(DataCollection):

    _DTYPES = {'vline': 'set_xdata',
               'hline': 'set_ydata',
               'xline': 'set_xdata',
               'xline': 'set_xdata',
               'yline': 'set_ydata',
               'line': 'set_data',
               'imshow': 'set_data'}

    _dax = {}
    _dobj = {}

    # ---------------------
    # Complement plot-specific dict
    # ---------------------

    def set_dnmax(self, dnmax={}):
        c0 = isinstance(dnmax, dict)
        c0 = c0 and all([kk in self._dgroup['lkey'] and isinstance(vv, int)
                         for kk, vv in dnmax.items()])
        c0 = c0 and all([kk in dnmax.keys() for kk in self._dgroup['lkey']])
        if not c0:
            msg = "dnmax must be a dict associating an int to each group\n"
            msg += "\t- self.dgroup.keys(): {}\n".format(self._dgroup.keys())
            msg += "\t- dnmax: {}".format(dnmax)
            raise Exception(msg)

        for kk, vv in dnmax.items():
            self._dgroup['dict'][kk]['nmax'] = vv
            for kr in self._dgroup['dict'][kk]['lref']:
                self._dref['dict'][kr]['ind'] = np.zeros((vv,), dtype=int)
                self._dref['dict'][kr]['value'] = np.zeros((vv,), dtype=int)

    # ---------------------
    # Get back to collection
    # ---------------------

    def extract_collection(self, cls=None):
        return cls(dref=self.dref, ddata=self.ddata)

    # ---------------------
    # Set dax, dobj
    # ---------------------

    def set_daxobj(self, dax=None, dobj=None):

        # ----------------------
        # Check basic formatting on dax

        c0 = isinstance(dax, dict)
        c0 = c0 and all([isinstance(kk, str) and isinstance(vv, dict)
                         for kk, vv in dax.items()])
        ls = ['ax', 'lobj', 'xref', 'yref']
        c0 = c0 and all([all([ss in vv.keys() for ss in ls]) for vv in
                         dax.values()])
        if not c0:
            msg = (
                "dax must be a dict with unique keys\n"
                + "Values must be dict with keys:\n"
                + "\t- {}".format(ls)
            )
            raise Exception(msg)

        # ----------------------
        # Check basic formatting on dobj

        c0 = isinstance(dobj, dict)
        c0 = c0 and all([isinstance(kk, str) and isinstance(vv, dict)
                         for kk, vv in dobj.items()])
        ls = ['axe', 'Type', 'refs', 'inds']
        c0 = c0 and all([all([ss in vv.keys() for ss in ls]) for vv in
                         dobj.values()])
        if not c0:
            msg = (
                "dax must be a dict with unique keys\n"
                + "Values must be dict with keys:\n"
                + "\t- {}".format(ls)
            )
            raise Exception(msg)

        # ----------------------
        # Check further formatting on dax

        dax['dict'] = {}
        dax['lkey'] = []
        for k0, v0 in dax.items():
            assert all([k1 in self._ddata['lkey'] for k1 in v0['xref']])
            assert all([k1 in self._ddata['lkey'] for k1 in v0['yref']])
            assert all([k1 in doj.keys() for k1 in v0['lobj']])
            # mpl-specific
            assert isinstance(vv['ax'], )
            dax['dict'][k0] = v0
            del dax[k0]
            dax['lkey'].append(k0)

        # ----------------------
        # Check further formatting on dobj

        dobj['dict'] = {}
        dobj['lkey'] = []
        for k0, v0 in dobj.items():
            assert all([k1 in dax['lkey'] for k1 in v0['axe']])
            assert all([k1 in self._DTYPES.keys() for k1 in v0['Type']])
            assert all([k1 in self._ddata['lkey'] for k1 in v0['data']])
            assert all([k1 in doj.keys() for k1 in v0['inds']])
            # mpl-specific
            assert isinstance(vv['ax'], )
            dobj['dict'][k0] = v0
            del dobj[k0]
            dobj['lkey'].append(k0)

        self._dax.update(dax)
        self._dobj.update(dobj)

    # ---------------------
    # Methods for extracting axes
    # ---------------------

    def get_lax(self, inkey):
        return [kk for kk in self._dax['lkey'] if inkey in kk]

    # ---------------------
    # Methods for showing data
    # ---------------------

    def get_summary(self, data=False, show=None, show_core=None,
                    sep='  ', line='-', just='l',
                    table_sep=None, verb=True, return_=False):
        """ Summary description of the object content """
        # # Make sure the data is accessible
        # msg = "The data is not accessible because self.strip(2) was used !"
        # assert self._dstrip['strip']<2, msg

        # -----------------------
        # Build for groups
        col0 = ['group name', 'nb. ref', 'nb. data', 'nmax']
        ar0 = [(k0,
                len(self._dgroup['dict'][k0]['lref']),
                len(self._dgroup['dict'][k0]['ldata']),
                self._dgroup['dict'][k0]['nmax'])
               for k0 in self._dgroup['lkey']]

        # -----------------------
        # Build for refs
        col1 = ['ref key', 'group', 'size', 'nb. data', 'ind', 'value']
        ar1 = [(k0,
                self._dref['dict'][k0]['group'],
                self._dref['dict'][k0]['size'],
                len(self._dref['dict'][k0]['ldata']),
                str(self._dref['dict'][k0]['ind']),
                str(self._dref['dict'][k0]['value']))
               for k0 in self._dref['lkey']]

        # -----------------------
        # Build for ddata
        if data:
            col2 = ['data key']
            if show_core is None:
                show_core = self._show_in_summary_core
            if isinstance(show_core, str):
                show_core = [show_core]
            lkcore = ['shape', 'groups', 'refs']
            assert all([
                ss in self._ddata['lparam'] + lkcore for ss in show_core
            ])
            col2 += show_core

            if show is None:
                show = self._show_in_summary
            if show == 'all':
                col2 += self._ddata['lparam']
            else:
                if isinstance(show, str):
                    show = [show]
                assert all([ss in self._ddata['lparam'] for ss in show])
                col2 += show

            ar2 = []
            for k0 in self._ddata['lkey']:
                v0 = self._ddata['dict'][k0]
                lu = [k0] + [str(v0[cc]) for cc in col2[1:]]
                ar2.append(lu)

        # -----------------------
        # Build for dax
        if len(self._dax) > 0:
            col3 = ['axes key', 'pos',
                    'xlabel', 'ylabel', 'xlim', 'ylim',
                    'nobj', 'xref / nxref', 'yref / nyref']
            ar3 = [(k0,
                    self._dax['dict'][k0]['ax'].get_position(),
                    self._dax['dict'][k0]['ax'].get_xlabel(),
                    self._dax['dict'][k0]['ax'].get_ylabel(),
                    self._dax['dict'][k0]['ax'].get_xlim(),
                    self._dax['dict'][k0]['ax'].get_ylim(),
                    len(self._dax['dict'][k0]['lobj']),
                    (self._dax['dict'][k0]['xref'][0]
                     + ' / ' + len(self._dax['dict'][k0]['xref']))
                    (self._dax['dict'][k0]['yref'][0]
                     + ' / ' + len(self._dax['dict'][k0]['yref'])))
                   for k0 in self._dax['lkey']]
        else:
            col3 = None
            ar3 = None

        # -----------------------
        # Build for dobj
        if len(self._dobj) > 0:
            col4 = ['obj key', 'axe',
                    'Type', 'refs', 'inds']
            ar4 = [(k0,
                    self._dobj['dict'][k0]['ax'],
                    self._dobj['dict'][k0]['Type'],
                    self._dobj['dict'][k0]['refs'],
                    self._dobj['dict'][k0]['inds'])]
        else:
            col4 = None
            ar4 = None

        # -----------------------
        # concatenate
        lar = [ar0, ar1]
        lcol = [col0, col1]
        if data is True:
            lar.append(ar2)
            lcol.append(col2)
        if ar3 is not None:
            lar.append(ar3)
            lcol.append(col3)
        if ar4 is not None:
            lar.append(ar4)
            lcol.append(col4)

        return self._get_summary(
            lar, lcol,
            sep=sep, line=line, table_sep=table_sep,
            verb=verb, return_=return_,
        )

    # ---------------------
    # Method for getting corresponding nearest index in != ref
    # ---------------------

    def _get_current_ind_for_data(self, key, obj=None):
        if obj is None:
            ind = [self._dref['dict'][kr]['ind']
                   for kr in self._ddata['dict'][key]['refs']]
        else:
            ind = [
                self._dref['dict'][kr]['ind'][
                    self._dobj['dict'][obj]['indi'][kr]
                ]
                for kr in self._ddata['dict'][key]['refs']
            ]
        return ind

    def _get_current_data_along_axis(self, key, axis=None, obj=None):
        dim = self._ddata['dict'][key]['dim']
        if dim == 1:
            return self._ddata['dict'][key]['data']
        else:
            ind = self._get_current_ind_for_data(key)
            if dim == 2:
                if axis == 0:
                    return self._ddata['dict'][key]['data'][:, ind[1]]
                else:
                    return self._ddata['dict'][key]['data'][ind[0]]
            elif dim == 3:
                if axis == 0:
                    return self._ddata['dict'][key]['data'][:, ind[1], ind[2]]
                elif axis == 1:
                    return self._ddata['dict'][key]['data'][ind[0]][:, ind[2]]
                else:
                    return self._ddata['dict'][key]['data'][ind[0]][ind[1]]

    def _get_data_ind_from_value(self, value=None, key=None,
                                 ref=None, axis=None):
        if axis is None:
            axis = self._ddata['dict'][key]['refs'].index(ref)
            size = self._dref['dict'][key]['size']
        else:
            pass

        if self._ddata['dict'][key]['bins'] is not None:
            # bins only exist for 1d sorted data
            ind = np.searchsorted([value], self._ddata['dict'][key]['bins'])[0]
        else:
            x = self._get_current_data_along_axis(key, axis)
            ind = np.nanargmin(np.abs(x-value))
        return ind % size

    def _get_current_value(self, key):
        ind = self._get_current_ind_for_data(key)
        dim = self._ddata['dict'][key]['dim']
        if dim == 1:
            return self._ddata['dict'][key]['data'][ind[0]]
        elif dim == 2:
            return self._ddata['dict'][key]['data'][ind[0]][ind[1]]
        elif dim == 3:
            return self._ddata['dict'][key]['data'][ind[0]][ind[1]][ind[2]]
