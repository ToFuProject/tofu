# -*- coding: utf-8 -*-
#! /usr/bin/python
"""
Thie imas-compatibility module of tofu

Default parameters and input checking

"""

# Built-ins
import sys
import os
import itertools as itt
import copy
import functools as ftools
import getpass
import inspect
import warnings
import traceback

# Standard
import numpy as np
import matplotlib as mpl
import datetime as dtm

# tofu
pfe = os.path.join(os.path.expanduser('~'), '.tofu', '_imas2tofu_def.py')
if os.path.isfile(pfe):
    # Make sure we load the user-specific file
    # sys.path method
    # sys.path.insert(1, os.path.join(os.path.expanduser('~'), '.tofu'))
    # import _scripts_def as _defscripts
    # _ = sys.path.pop(1)
    # importlib method
    import importlib.util
    spec = importlib.util.spec_from_file_location("_defimas2tofu", pfe)
    _defimas2tofu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_defimas2tofu)
else:
    try:
        import tofu.imas2tofu._def as _defimas2tofu
    except Exception as err:
        from . import _def as _defimas2tofu
try:
    import tofu.imas2tofu._comp as _comp
    import tofu.imas2tofu._comp_toobjects as _comp_toobjects
    import tofu.imas2tofu._comp_mesh as _comp_mesh
except Exception as err:
    from . import _comp as _comp
    from . import _comp_toobjects as _comp_toobjects
    from . import _comp_mesh as _comp_mesh

# imas
try:
    import imas
    from imas import imasdef
except Exception as err:
    raise Exception('imas not available')

__all__ = [
    'check_units_IMASvsDSHORT',
    'MultiIDSLoader',
    'load_Config', 'load_Plasma2D', 'load_Cam', 'load_Data',
    '_save_to_imas',
]


# Root tofu path (for saving repo in IDS)
_ROOT = os.path.abspath(os.path.dirname(__file__))
_ROOT = _ROOT[:_ROOT.index('tofu')+len('tofu')]


#############################################################
#           Preliminary units check
#############################################################


def check_units_IMASvsDSHORT(dshort=None, dcomp=None,
                             verb=True, returnas=False):

    # Check input
    if dshort is None:
        dshort = _defimas2tofu._dshort
    if dcomp is None:
        dcomp = _defimas2tofu._dcomp

    # loop on keys
    ddiff = {}
    for k0, v0 in dshort.items():
        for k1, v1 in v0.items():
            u0 = _comp.get_units(k0, k1,
                                 dshort=dshort, dcomp=dcomp,
                                 force=False)
            u1 = v1.get('units', None)
            longstr = dshort[k0][k1]['str']
            if u0 != u1:
                key = '{}.{}'.format(k0, k1)
                ddiff[key] = (u0, u1, longstr)

    if verb is True:
        msg = np.array(([('key', 'imas (dd_units)', 'tofu (dshort)',
                          'long version')]
                        + [(kk, vv[0], vv[1], vv[2])
                           for kk, vv in ddiff.items()]),
                       dtype='U')
        length = np.max(np.char.str_len(msg), axis=0)
        msg = np.array([np.char.ljust(msg[:, ii], length[ii])
                        for ii in range(length.size)]).T
        msg = (' '.join([aa for aa in msg[0, :]]) + '\n'
               + ' '.join(['-'*ll for ll in length]) + '\n'
               + '\n'.join([' '.join(aa) for aa in msg[1:, :]]))
        print(msg)
    if returnas is dict:
        return ddiff


#############################################################
#############################################################
#           IdsMultiLoader
#############################################################


class MultiIDSLoader(object):
    """ Class for handling multi-ids possibly from different idd

    For each desired ids (e.g.: core_profile, ece, equilibrium...), you can
    specify a different idd (i.e.: (shot, run, user, database, version))

    The instance will remember from which idd each ids comes from.
    It provides a structure to keep track of these dependencies
    Also provides methods for opening idd and getting ids and closing idd

    """



    ###################################
    #       Default class attributes
    ###################################

    _def = {
        'isget': False,
        'ids': None,
        'occ': 0,
        'needidd': True,
    }
    _defidd = dict(_defimas2tofu._IMAS_DIDD)

    _lidsnames = [k for k in dir(imas) if k[0] != '_']
    _lidsk = [
        'database', 'user', 'version', 'shot', 'run', 'refshot', 'refrun',
    ]

    # Known short version of signal str
    _dshort = _defimas2tofu._dshort
    _didsdiag = _defimas2tofu._didsdiag
    _lidsconfig = _defimas2tofu._lidsconfig
    _lidsdiag = _defimas2tofu._lidsdiag
    _lidslos = _defimas2tofu._lidslos
    _lidssynth = _defimas2tofu._lidssynth
    _lidsplasma = [
        'equilibrium', 'core_profiles', 'core_sources',
        'edge_profiles', 'edge_sources',
    ]

    # Computed signals
    _dcomp = _defimas2tofu._dcomp
    _lids = _defimas2tofu._lids

    # All except (for when sig not specified in get_data())
    _dall_except = _defimas2tofu._dall_except

    # Preset
    _dpreset = _defimas2tofu._dpreset

    # basis ids
    _IDS_BASE = _defimas2tofu._IDS_BASE


    ###################################
    ###################################
    #       Methods
    ###################################


    def __init__(self, preset=None, dids=None, ids=None, occ=None, idd=None,
                 shot=None, run=None, refshot=None, refrun=None,
                 user=None, database=None, version=None, backend=None,
                 ids_base=None, synthdiag=None, get=None, ref=True):
        """ A class for handling multiple ids loading from IMAS

        IMAS provides access to a database via a standardized structure (idd
        and ids). This class is a convenient tool to interact with IMAS.

        idd: An IMAS data dictionnary (idd) is like a 'shotfile', it contains
            all data for a given shot, in a given version.
            An idd is identified by:
                - user:     the name of the user to whom the idd belongs.
                            Indeed, each idd can e stored in an official
                            centralized database identified by a generic user
                            name (e.g.: 'public') or locally on a personnal
                            database identified by your own user name.
                            An idd stored locally on the database of user A can
                            be read by other users if they provide the
                            user name 'A'.
                - database:  the name of the experiment (e.g.: 'ITER')
                - shot:     the shot number
                - run:      It's the 'version' of the shotfile.
                            Indeed, IMAS allows to store both experimental and
                            simulation data. A given experimental data can
                            exist in several versions (more or less filtered or
                            treated) and the same goes for simulation data (the
                            same simulation can be run with different sets of
                            parameters, or with a different code).
                            For a given shot, several runs can exist

        ids: Once the idd has been chosen, it contains all the available data
            in the form of IMAS data Structures (ids).
            Each ids contains a 'family' or 'group' of data. It has an explicit
            name to indicate what that group is.
            There are typically diagnostic ids (e.g.: 'barometry',
            'intereferometer', 'soft_x_rays', ...) that contain all data
            produced by these diagnostics (with their time bases, units...),
            advanced data treatment ids (e.g.: 'equilibrium'...) and simulation
            ids (e.g.: 'core_profiles', 'edge_sources'...)

        In a typical use case, you would want to load all data from several ids
        from the same idd (e.g.: from the official centralized idd of a shot
        that contains official, validated data).
        But for some analysis, you may want to load different ids from
        different idd (e.g.: to compare official experimental data of a
        diagnostics to synthetic data computed from the core profiles produced
        by a simulation and interpolated via an equilibrium produced by a
        third-party code).

        This class provides an easy interface to access several ids from
        several idd (or a unique idd of course).


        Example
        -------

        # Tis will load 2 ids from the same public idd
        # But we know we will need to add another ids from a different idd
        # So we instanciate the class and secpify get=False to postpone the
        # data loading itself until we have added all we need
        import tofu as tf
        user = 'imas_public'
        ids = ['interferometer', 'polarimeter']
        multi = tf.imas2tofu.MultiIDSLoader(
            shot=55583, user=user,
            database='west', ids=ids, get=False,
        )

        # This will ad an ids from a different idd and automatically load
        # ('get') everything
        multi.add_ids('bolometer', shot=55583, user='myusername',
                      database='west')

        # To have an overview of what your multi instance contains, type
        multi

        """
        super(MultiIDSLoader, self).__init__()

        # Initialize dicts
        self._init_dict()

        # Check and format inputs
        if dids is None:
            self.add_idd(
                idd=idd,
                shot=shot, run=run, refshot=refshot, refrun=refrun,
                user=user, database=database, version=version, ref=ref,
                backend=backend,
            )
            lidd = list(self._didd.keys())
            assert len(lidd) <= 1
            idd = lidd[0] if len(lidd) > 0 else None
            self.add_ids(
                preset=preset,
                ids=ids,
                occ=occ,
                idd=idd,
                get=False,
                backend=backend,
            )
            if ids_base is None:
                if not all([iids in self._IDS_BASE
                            for iids in self._dids.keys()]):
                    ids_base = True
                else:
                    ids_base = False
            if not isinstance(ids_base, bool):
                msg = ("Arg ids_base must be bool:\n"
                       + "\t- False: adds no ids\n"
                       + "\t- True: adds ids in self._IDS_BASE\n"
                       + "  You provided:\n{}".format(ids_base))
                raise Exception(msg)
            if ids_base is True:
                self.add_ids_base(get=False)
            if synthdiag is None:
                synthdiag = False
            if synthdiag is True:
                self.add_ids_synthdiag(get=False)
            if get is None and (len(self._dids) > 0 or preset is not None):
                get = True
        else:
            self.set_dids(dids)
            if get is None:
                get = True
        self._set_fsig()
        if get is True:
            self.open_get_close()

    def _init_dict(self):
        self._didd = {}
        self._dids = {}
        self._refidd = None
        self._preset = None

    def set_dids(self, dids=None):
        didd, dids, refidd = self._get_diddids(dids)
        self._dids = dids
        self._didd = didd
        self._refidd = refidd


    @classmethod
    def _get_diddids(cls, dids, defidd=None):

        # Check input
        assert type(dids) is dict
        assert all([type(kk) is str for kk in dids.keys()])
        assert all([kk in cls._lidsnames for kk in dids.keys()])
        if defidd is None:
            defidd = cls._defidd
        didd = {}

        # Check ids
        for k, v in dids.items():

            lc0 = [
                v is None or v == {},
                type(v) is dict,
                hasattr(v, 'ids_properties'),
            ]
            assert any(lc0)

            if lc0[0]:
                dids[k] = {'ids': None}
            elif lc0[-1]:
                dids[k] = {'ids': v}
            dids[k]['ids'] = dids[k].get('ids', None)
            v = dids[k]

            # Implement possibly missing default values
            for kk, vv in cls._def.items():
                dids[k][kk] = v.get(kk, vv)
            v = dids[k]

            # Check / format occ and deduce nocc
            assert type(dids[k]['occ']) in [int, list]
            dids[k]['occ'] = np.r_[dids[k]['occ']].astype(int)
            dids[k]['nocc'] = dids[k]['occ'].size
            v = dids[k]

            # Format isget / get
            for kk in ['isget']:    #, 'get']:
                assert type(v[kk]) in [bool, list]
                v[kk] = np.r_[v[kk]].astype(bool)
                assert v[kk].size in set([1,v['nocc']])
                if v[kk].size < v['nocc']:
                    dids[k][kk] = np.repeat(v[kk], v['nocc'])
            v = dids[k]

            # Check / format ids
            lc = [
                v['ids'] is None,
                hasattr(v['ids'], 'ids_properties'),
                type(v['ids']) is list,
            ]
            assert any(lc)

            if lc[0]:
                dids[k]['needidd'] = True
            elif lc[1]:
                assert v['nocc'] == 1
                dids[k]['ids'] = [v['ids']]
                dids[k]['needidd'] = False
            elif lc[2]:
                assert all([hasattr(ids, 'ids_properties') for ids in v['ids']])
                assert len(v['ids']) == v['nocc']
                dids[k]['needidd'] = False
            v = dids[k]

            # ----------------
            # check and open idd, populate didd
            # ----------------
            idd = v.get('idd', None)
            if idd is None:
                dids[k]['idd'] = None
                continue
            kwargs = {}
            if type(idd) is dict:
                idd, kwargs = None, idd
            diddi = cls._checkformat_idd(idd=idd, **kwargs)

            name = list(diddi.keys())[0]
            didd[name] = diddi[name]
            dids[k]['idd'] = name

        # --------------
        #   Now use idd for ids without idd needing one
        # --------------

        # set reference idd, if any
        lc = [(k,v['needidd']) for k,v in dids.items()]
        lc0, lc1 = zip(*lc)
        refidd = None
        if any(lc1):
            if len(didd) == 0:
                msg = "The following ids need an idd to be accessible:\n"
                msg += "    - "
                msg += "    - ".join([lc0[ii] for ii in range(0,len(lc0))
                                      if lc1[ii]])
                raise Exception(msg)
            refidd = list(didd.keys())[0]

        # set missing idd to ref
        need = False
        for k, v in dids.items():
            lc = [v['needidd'], v['idd'] is None]
            if all(lc):
                dids[k]['idd'] = refidd

        return didd, dids, refidd


    #############
    # properties

    @property
    def dids(self):
        return self._dids
    @property
    def didd(self):
        return self._didd
    @property
    def refidd(self):
        return self._refidd

    #############
    #############
    # methods
    #############


    #############
    # shortcuts

    @staticmethod
    def _getcharray(ar, col=None, sep='  ', line='-', just='l', msg=True):
        c0 = ar is None or len(ar) == 0
        if c0:
            return ''
        ar = np.array(ar, dtype='U')

        if ar.ndim == 1:
            ar = ar.reshape((1, ar.size))

        # Get just len
        nn = np.char.str_len(ar).max(axis=0)
        if col is not None:
            if len(col) not in ar.shape:
                msg = ("len(col) should be in np.array(ar, dtype='U').shape:\n"
                       + "\t- len(col) = {}\n".format(len(col))
                       + "\t- ar.shape = {}".format(ar.shape))
                raise Exception(msg)
            if len(col) != ar.shape[1]:
                ar = ar.T
                nn = np.char.str_len(ar).max(axis=0)
            nn = np.fmax(nn, [len(cc) for cc in col])

        # Apply to array
        fjust = np.char.ljust if just == 'l' else np.char.rjust
        out = np.array([sep.join(v) for v in fjust(ar,nn)])

        # Apply to col
        if col is not None:
            arcol = np.array([col, [line*n for n in nn]], dtype='U')
            arcol = np.array([sep.join(v) for v in fjust(arcol,nn)])
            out = np.append(arcol,out)

        if msg:
            out = '\n'.join(out)
        return out

    @staticmethod
    def _shortcuts(obj, ids=None, return_=False, force=None,
                   verb=True, sep='  ', line='-', just='l'):
        if ids is None:
            if hasattr(obj, '_dids'):
                lids = list(obj._dids.keys())
            else:
                lids = list(obj._dshort.keys())
        elif ids == 'all':
            lids = list(obj._dshort.keys())
        else:
            lc = [type(ids) is str, type(ids) is list and all([type(ss) is str
                                                               for ss in ids])]
            assert any(lc), "ids must be an ids name or a list of such !"
            if lc[0]:
                lids = [ids]
            else:
                lids = ids
        lids = sorted(set(lids).intersection(obj._dshort.keys()))

        short = []
        for ids in lids:
            lks = obj._dshort[ids].keys()
            if ids in obj._dcomp.keys():
                lkc = obj._dcomp[ids].keys()
                lk = sorted(set(lks).union(lkc))
            else:
                lk = sorted(lks)
            for kk in lk:
                if kk in lks:
                    ss = obj._dshort[ids][kk]['str']
                else:
                    ss = 'f( %s )'%(', '.join(obj._dcomp[ids][kk]['lstr']))
                uu = obj.get_units(ids, kk, force=force)
                short.append((ids, kk, uu, ss))

        if verb:
            col = ['ids', 'shortcut', 'units', 'long version']
            msg = obj._getcharray(short, col, sep=sep, line=line, just=just)
            print(msg)
        if return_:
            return short

    @classmethod
    def get_shortcutsc(cls, ids=None, return_=False, force=None,
                       verb=True, sep='  ', line='-', just='l'):
        """ Display and/or return the builtin shortcuts for imas signal names

        By default (ids=None), only display shortcuts for stored ids
        To display all possible shortcuts, use ids='all'
        To display shortcuts for a specific ids, use ids=<idsname>

        These shortcuts can be customized (with self.set_shortcuts())
        They are useful for use with self.get_data()

        """
        return cls._shortcuts(cls, ids=ids, return_=return_, verb=verb,
                              sep=sep, line=line, just=just, force=force)

    def get_shortcuts(self, ids=None, return_=False, force=None,
                      verb=True, sep='  ', line='-', just='l'):
        """ Display and/or return the builtin shortcuts for imas signal names

        By default (ids=None), only display shortcuts for stored ids
        To display all possible shortcuts, use ids='all'
        To display shortcuts for a specific ids, use ids=<idsname>

        These shortcuts can be customized (with self.set_shortcuts())
        They are useful for use with self.get_data()

        """
        return self._shortcuts(self, ids=ids, return_=return_, verb=verb,
                               sep=sep, line=line, just=just, force=force)

    def set_shortcuts(self, dshort=None):
        """ Set the dictionary of shortcuts

        If None, resets to the class's default dict
        """
        dsh = copy.deepcopy(self.__class__._dshort)
        if dshort is not None:
            c0 = type(dshort) is dict
            c1 = c0 and all([k0 in self._lidsnames and type(v0) is dict
                             for k0,v0 in dshort.items()])
            c2 = c1 and all([all([type(k1) is str and type(v1) in {str,dict}
                                  for k1,v1 in v0.items()])
                             for v0 in dshort.values()])
            if not c2:
                msg = "Arg dshort should be a dict with valid ids as keys:\n"
                msg += "    {'ids0': {'shortcut0':'long_version0',\n"
                msg += "              'shortcut1':'long_version1'},\n"
                msg += "     'ids1': {'shortcut2':'long_version2',...}}"
                raise Exception(msg)

            for k0,v0 in dshort.items():
                for k1,v1 in v0.items():
                    if type(v1) is str:
                        dshort[k0][k1] = {'str':v1}
                    else:
                        assert 'str' in v1.keys() and type(v1['str']) is str

            for k0, v0 in dshort.items():
                dsh[k0].update(dshort[k0])
        self._dshort = dsh
        self._set_fsig()

    def update_shortcuts(self, ids, short, longstr):
        assert ids in self._dids.keys()
        assert type(short) is str
        assert type(longstr) is str
        self._dshort[ids][short] = {'str':longstr}
        self._set_fsig()

    #############
    # preset

    @classmethod
    def _getpreset(cls, obj=None, key=None, return_=False,
                   verb=True, sep='  ', line='-', just='l'):
        if obj is None:
            obj = cls
        if key is None:
            lkeys = list(obj._dpreset.keys())
        else:
            lc = [type(key) is str, type(key) is list and all([type(ss) is str
                                                               for ss in key])]
            assert any(lc), "key must be an vali key of self.dpreset or a list of such !"
            if lc[0]:
                lkeys = [key]
            else:
                lkeys = key

        preset = []
        for key in lkeys:
            lids = sorted(obj._dpreset[key].keys())
            for ii in range(0,len(lids)):
                s0 = key if ii == 0 else ''
                if obj._dpreset[key][lids[ii]] is None:
                    s1 = sorted(obj._dshort[lids[ii]].keys())
                else:
                    s1 = obj._dpreset[key][lids[ii]]
                    assert type(s1) in [str,list]
                    if type(s1) is str:
                        s1 = [s1]
                s1 = ', '.join(s1)
                preset.append((s0,lids[ii],s1))

        if verb:
            col = ['preset', 'ids', 'shortcuts']
            msg = obj._getcharray(preset, col, sep=sep, line=line, just=just)
            print(msg)
        if return_:
            return preset

    def get_preset(self, key=None, return_=False,
                   verb=True, sep='  ', line='-', just='l'):
        """ Display and/or return the builtin shortcuts for imas signal names

        By default (ids=None), only display shortcuts for stored ids
        To display all possible shortcuts, use ids='all'
        To display shortcuts for a specific ids, use ids=<idsname>

        These shortcuts can be customized (with self.set_shortcuts())
        They are useful for use with self.get_data()

        """
        return self._getpreset(obj=self, key=key, return_=return_, verb=verb,
                               sep=sep, line=line, just=just)

    def set_preset(self, dpreset=None):
        """ Set the dictionary of preselections

        If None, resets to the class's default dict
        """
        dpr = copy.deepcopy(self.__class__._dpreset)
        if dpreset is not None:
            c0 = type(dpreset) is dict
            c1 = c0 and all([type(k0) is str and type(v0) is dict
                             for k0,v0 in dpreset.items()])
            c2 = c1 and all([[(k1 in self._lidsnames
                               and (type(v1) in [str,list] or v1 is None))
                              for k1,v1 in v0.items()]
                             for v0 in dpreset.values()])
            c3 = True and c2
            for k0,v0 in dpreset.items():
                for k1, v1 in v0.items():
                    if type(v1) is str:
                        dpreset[k0][k1] = [v1]
                    c3 = c3 and all([ss in self._dshort[k1].keys()
                                     for ss in dpreset[k0][k1]])
            if not c3:
                msg = "Arg dpreset should be a dict of shape:\n"
                msg += "    {'key0': {'ids0': ['shortcut0','shortcut1',...],\n"
                msg += "              'ids1':  'shortcut2'},\n"
                msg += "     'key1': {'ids2':  'shortcut3',\n"
                msg += "              'ids3':  None}}\n\n"
                msg += "  i.e.: each preset (key) is a dict of (ids,value)"
                msg += "        where value is either:\n"
                msg += "            - None: all shortuc of ids are taken\n"
                msg += "            - str : a valid shortut\n"
                msg += "            - list: a list of valid shortcuts"
                raise Exception(msg)

            for k0, v0 in dpreset.keys():
                dpr[k0].update(dpreset[k0])
        self._dpreset = dpr

    def update_preset(self, key, ids, lshort):
        assert ids in self._dshort.keys()
        assert lshort is None or type(lshort) in [str,list]
        if type(lshort) is str:
            lshort = [lshort]
        if lshort is not None:
            assert all([ss in self._dshort[ids].keys() for ss in lshort])
        self._dpreset[key][ids] = lshort

    #############
    # ids getting

    def _checkformat_get_idd(self, idd=None):
        lk = self._didd.keys()
        lc = [
            idd is None, type(idd) is str and idd in lk,
            hasattr(idd, '__iter__') and all([ii in lk for ii in idd])
        ]
        assert any(lc)

        if lc[0]:
            lidd = lk
        elif lc[1]:
            lidd = [idd]
        else:
            lidd = idd
        return lidd

    def _checkformat_get_ids(self, ids=None):
        """ Return a list of tuple (idd, lids) """
        lk = self._dids.keys()
        lc = [ids is None, type(ids) is str and ids in lk,
              hasattr(ids,'__iter__') and all([ii in lk for ii in ids])]
        assert any(lc)

        if lc[0]:
            lids = lk
        elif lc[1]:
            lids = [ids]
        else:
            lids = ids
        lidd = sorted(set([self._dids[ids]['idd'] for ids in lids]))
        llids = [
            (
                idd,
                sorted([ids for ids in lids if self._dids[ids]['idd'] == idd]),
            )
            for idd in lidd
        ]
        return llids

    def _open(self, idd=None):
        lidd = self._checkformat_get_idd(idd)
        for k in lidd:
            if self._didd[k]['isopen'] == False:
                self._didd[k]['idd'].open()
                self._didd[k]['isopen'] = True

    def _get(
        self,
        idsname=None,
        occ=None,
        llids=None,
        verb=True,
        sep='  ',
        line='-',
        just='l',
    ):
        """ get ids """

        # ------------
        # check input

        lerr = []
        if llids is None:
            llids = self._checkformat_get_ids(idsname)
        if len(llids) == 0:
            return lerr

        if verb:
            msg0 = ['Getting ids', '[occ]'] + self._lidsk
            lmsg = []

        # ------------
        # check input

        docc = {}
        for ii in range(0,len(llids)):

            # initialise dict of occurrences
            docc[ii] = {
                jj: {'oc': None, 'indok': None}
                for jj in range(len(llids[ii][1]))
            }

            # fill dict of occurrences
            for jj in range(len(llids[ii][1])):
                ids = llids[ii][1][jj]
                occref = self._dids[ids]['occ']
                if occ is None:
                    oc = occref
                else:
                    oc = np.unique(np.r_[occ].astype(int))
                    oc = np.intersect1(oc, occref)

                docc[ii][jj]['oc'] = oc
                docc[ii][jj]['indoc'] = np.array([
                    (occref == oc[ll]).nonzero()[0][0]
                    for ll in range(len(oc))
                ])

                # verb
                if verb:
                    msg = [ids, str(oc)]
                    if jj == 0:
                        msg += [
                            str(self._didd[llids[ii][0]]['params'][kk])
                            for kk in self._lidsk
                        ]
                    else:
                        msg += ['"' for _ in self._lidsk]
                    lmsg.append(msg)

        # verb
        if verb:
            msgar = self._getcharray(
                lmsg, col=msg0,
                sep=sep, line=line, just=just, msg=False,
            )
            print('\n'.join(msgar[:2]))

        # ------------
        # check input

        nline = 0
        for ii in range(len(llids)):
            for jj in range(len(llids[ii][1])):
                ids = llids[ii][1][jj]
                oc = docc[ii][jj]['oc']
                indoc = docc[ii][jj]['indoc']

                # if ids not provided
                if self._dids[ids]['ids'] is None:
                    self._dids[ids]['ids'] = [None for ii in range(len(oc))]
                    self._dids[ids]['needidd'] = True

                if verb:
                    print(msgar[2+nline])

                # get ids
                idd = self._didd[self._dids[ids]['idd']]['idd']
                try:
                    for ll in range(len(oc)):
                        c0 = (
                            self._dids[ids]['ids'][ll] is None
                            or self._dids[ids]['isget'][indoc[ll]] == False
                        )
                        if c0:
                            self._dids[ids]['ids'][ll] = idd.get(
                                ids,
                                occurrence=oc[ll],
                            )
                            self._dids[ids]['isget'][indoc[ll]] = True
                    done = 'ok'
                except Exception as err:
                    done = 'error !'
                    lerr.append(err)
                    if verb:
                        print('        => failed !')

                nline += 1

        return lerr

    def _close(self, idd=None):
        lidd = self._checkformat_get_idd(idd)
        for k in lidd:
            self._didd[k]['idd'].close()
            self._didd[k]['isopen'] = False

    def get_list_notget_ids(self):
        lids = [k for k,v in self._dids.items() if np.any(v['isget'] == False)]
        return lids

    def open_get_close(self, ids=None, occ=None, verb=True):
        """ Force data loading

        If at instanciation or when using method add_ids() you specified option
        get = False, then the latest added ids may not have been loaded.

        This method forces a refresh and loads all ids contained in the instance

        The name comes from:
            - open (all the idd)
            - get (all the ids)
            - close (all the idd)
        """
        llids = self._checkformat_get_ids(ids)
        lidd = [lids[0] for lids in llids]
        self._open(idd=lidd)
        lerr = self._get(occ=occ, llids=llids)
        self._close(idd=lidd)
        if verb and len(lerr) > 0:
            for err in lerr:
                warnings.warn(str(err))

    #---------------------
    # Methods for adding / removing idd / ids
    #---------------------

    @classmethod
    def _checkformat_idd(
        cls, idd=None,
        shot=None, run=None, refshot=None, refrun=None,
        user=None, database=None, version=None,
        isopen=None, ref=None, defidd=None, backend=None,
    ):
        """ Check imas entry parameters
        """

        lc = [idd is None, shot is None]
        if not any(lc):
            msg = "You cannot provide both idd and shot !"
            raise Exception(msg)

        if all(lc):
            didd = {}
            return didd
        if defidd is None:
            defidd = cls._defidd

        if lc[0]:
            assert type(shot) in [int,np.int_]
            params = dict(
                shot=int(shot), run=run, refshot=refshot, refrun=refrun,
                user=user, database=database, version=version,
                backend=backend,
            )
            for kk,vv in defidd.items():
                if params[kk] is None:
                    params[kk] = vv

            # convert backend str => pointer
            params['backend'] = getattr(
                imasdef,
                f"{params['backend']}_BACKEND".upper(),
            )

            # create entry
            idd = imas.DBEntry(
                params['backend'], params['database'], params['shot'],
                params['run'],
                user_name=params['user'],
                data_version=params['version'],
            )
            isopen = False

        elif lc[1]:
            if not hasattr(idd, 'close'):
                msg = "idd does not seem to be data entry!"
                raise Exception(msg)

            params = {
                'shot': idd.shot,
                'run': idd.run,
                'refshot': idd.getRefShot(),
                'refrun': idd.getRefRun(),
            }

            expIdx = idd.expIdx
            if not (expIdx == -1 or expIdx > 0):
                msg = (
                    "Status of the provided idd could not be determined:\n"
                    f"\t- idd.expIdx : {expIdx}   (should be -1 or >0)\n"
                    f"\t- (shot, run): {idd.shot}, {idd.run}\n"
                )
                raise Exception(msg)

            if isopen is not None:
                if isopen != (expIdx > 0):
                    msg = (
                        "Provided isopen does not match observed value:\n"
                        f"\t- isopen: {isopen}\n"
                        f"\t- expIdx: {expIdx}"
                    )
                    raise Exception(msg)

            isopen = expIdx > 0

        if 'user' in params.keys():
            name = [params['user'], params['database'], params['version']]
        else:
            name = [str(id(idd))]
        name += ['{:06.0f}'.format(params['shot']),
                 '{:05.0f}'.format(params['run'])]
        name = '_'.join(name)
        didd = {name: {'idd': idd, 'params': params, 'isopen': isopen}}
        return didd

    def set_refidd(self, idd=None):
        if len(self._didd.keys()) == 0:
            assert idd is None
        else:
            assert idd in self._didd.keys()
        self._refidd = idd

    def add_idd(self, idd=None,
                shot=None, run=None, refshot=None, refrun=None,
                user=None, database=None, version=None, backend=None,
                ref=None, return_name=False):
        assert ref in [None, True]
        # didd
        didd = self._checkformat_idd(
            idd=idd,
            shot=shot, run=run,
            refshot=refshot, refrun=refrun,
            user=user, database=database,
            version=version,
            backend=backend,
        )
        self._didd.update(didd)
        name = list(didd.keys())[0]

        # ref
        if ref is None:
            ref = self._refidd  is None
        if ref == True and len(didd.keys())>0:
            self.set_refidd(name)
        if return_name:
            return name

    def remove_idd(self, idd=None):
        """Remove an idd and all its dependencies (ids) """
        if idd is not None:
            if not idd in self._didd.keys():
                msg = "Please provide the name (str) of a valid idd\n"
                msg += "Currently available idd are:\n"
                msg += "    - %s"%str(sorted(self._didd.keys()))
                raise Exception(msg)
            lids = [k for k,v in self._dids.items() if v['idd']==idd]
            del self._didd[idd]
            for ids in lids:
                del self._dids[ids]

    def get_idd(self, idd=None):
        if idd is None and len(self._didd.keys()) == 1:
            idd = list(self._didd.keys())[0]
        assert idd in self._didd.keys()
        return self._didd[idd]['idd']

    def _checkformat_ids_synthdiag(self, ids=None):
        lc = [ids is None, isinstance(ids, str), isinstance(ids, list),
              hasattr(ids, 'ids_properties')]
        if not any(lc):
            msg = ("Provided ids not understood!\n"
                   + "\t- provided: {}".format(str(ids)))
            raise Exception(msg)

        lidssynth = [kk for kk, vv in self._didsdiag.items()
                     if 'synth' in vv.keys()]
        if lc[0]:
            ids = sorted(set(self._dids.keys()).intersection(lidssynth))
        elif lc[1]:
            ids = [ids]
        elif lc[3]:
            ids = [ids.__class__.__name__]

        ids = sorted(
            set(ids).intersection(lidssynth).intersection(self._dids.keys()))
        if len(ids) == 0:
            msg = ("The provided ids must be:\n"
                   + "\t- an ids name (str)\n"
                   + "\t- a list of ids names\n"
                   + "\t- an ids instance\n"
                   + "\t- None\n"
                   + "And it must:\n"
                   + "\t- Already be added (cf. self.dids.keys())\n"
                   + "\t- Be a diagnostic ids with tabulated 'synth'")
            # Turn to warning? => see user feedback
            raise Exception(msg)
        return ids

    def get_inputs_for_synthsignal(self, ids=None, verb=True, returnas=False):
        """ Return and / or print a dict of the default inputs for desired ids

        Synthetic signal for a given diagnostic ids is computed from
        signal that comes from other ids (e.g. core_profiles, equilibrium...)
        For some diagnostics, the inputs required are already tabulated in
        self._didsdiag[<ids>]['synth']

        This method simply shows this already tabulated information
        Advanced users may edit this hidden dictionnary to their needs

        """
        assert returnas in [False, True, dict, list]
        ids = self._checkformat_ids_synthdiag(ids)

        # Deal with real case
        if len(ids) == 1:
            out = self._didsdiag[ids[0]]['synth']
            lids = sorted(out.get('dsig', {}).keys())
            if verb:
                dmsg = ("\n\t-" +
                        "\n\t-".join([
                            kk+':\n\t\t'+'\n\t\t'.join(vv)
                            for kk, vv in out.get('dsig', {}).items()]))
                extra = {kk: vv for kk, vv in out.items()
                         if kk not in ['dsynth', 'dsig']}
                msg = ("For computing synthetic signal for ids {}".format(ids)
                       + dmsg + '\n'
                       + "\t- Extra parameters (if any):\n"
                       + "\t\t{}\n".format(extra))
                print(msg)
            if returnas is True:
                returnas = dict
        else:
            out = None
            lids = sorted(set(itt.chain.from_iterable([
                self._didsdiag[idsi]['synth'].get('dsig', {}).keys()
                for idsi in ids])))
            if verb:
                print(lids)
            if returnas is True:
                returnas = list
        if returnas is dict:
            return out
        elif returnas is list:
            return lids

    def _checkformat_ids(
        self,
        ids,
        occ=None,
        idd=None,
        isget=None,
        synthdiag=False,
    ):

        # Check value and make dict if necessary
        lc = [
            isinstance(ids, str),
            isinstance(ids, list),
            hasattr(ids, 'ids_properties'),
            ids is None and synthdiag is True
        ]
        if not any(lc):
            msg = (
                "Arg ids must be either:\n"
                "\t- str : valid ids name\n"
                "\t- a list of such\n"
                "\t- an ids itself\n"
                f"  Provided: {ids}"
                f"  Conditions: {lc}"
            )
            raise Exception(msg)

        # Synthdiag-specific
        if synthdiag is True:
            ids = self.get_inputs_for_synthsignal(
                ids=ids, verb=False, returnas=list,
            )
            lc[1] = True

        # Prepare dids[name] = {'ids':None/ids, 'needidd':bool}
        dids = {}
        if lc[0] or lc[1]:
            if lc[0]:
                ids = [ids]

            # check ids is allowed
            for ids_ in ids:
                if not ids_ in self._lidsnames:
                    msg = (
                        "ids {ids_} matched no known imas ids !"
                        f"  => Available ids are:\n{repr(self._lidsnames)}"
                    )
                    raise Exception(msg)

            # initialise dict
            for k in ids:
                dids[k] = {'ids':None, 'needidd':True, 'idd':idd}
            lids = ids

        elif lc[2]:
            dids[ids.__class__.__name__] = {
                'ids': ids,
                'needidd': False,
                'idd': idd,
            }
            lids = [ids.__class__.__name__]

        nids = len(lids)

        # ----------------------------------
        # Check / format occ and deduce nocc

        if occ is None:
            occ = 0
        lc = [type(occ) in [int, np.int], hasattr(occ, '__iter__')]
        assert any(lc)

        if lc[0]:
            occ = [np.r_[occ].astype(int) for _ in range(nids)]
        else:
            if len(occ) == nids:
                occ = [np.r_[oc].astype(int) for oc in occ]
            else:
                occ = [np.r_[occ].astype(int) for _ in range(0,nids)]

        for ii in range(nids):
            nocc = occ[ii].size
            dids[lids[ii]]['occ'] = occ[ii]
            dids[lids[ii]]['nocc'] = nocc
            if dids[lids[ii]]['ids'] is not None:
                dids[lids[ii]]['ids'] = [dids[lids[ii]]['ids']]*nocc

        # ----------------------------------
        # Format isget / get

        for ii in range(0,nids):
            nocc = dids[lids[ii]]['nocc']

            # initialize get
            if dids[lids[ii]]['ids'] is None:
                isgeti = np.zeros((nocc,), dtype=bool)

            # set get
            if dids[lids[ii]]['ids'] is not None:
                if isget is None:
                    isgeti = np.r_[False]
                elif type(isget) is bool:
                    isgeti = np.r_[bool(isget)]
                elif hasattr(isget,'__iter__'):
                    if len(isget) == nids:
                        isgeti = np.r_[isget[ii]]
                    else:
                        isgeti = np.r_[isget]

            assert isgeti.size in [1, nocc]
            if isgeti.size < nocc:
                isgeti = np.repeat(isgeti,nocc)
            dids[lids[ii]]['isget'] = isgeti

        return dids

    def add_ids(self, ids=None, occ=None, idd=None, preset=None,
                shot=None, run=None, refshot=None, refrun=None,
                user=None, database=None, version=None, backend=None,
                ref=None, isget=None, get=None):
        """ Add an ids (or a list of ids)

        Optionally specify also a specific idd to which the ids will be linked
        The ids can be provided as such, or by name (str)

        """

        if get is None:
            get = False if preset is None else True

        # preset
        if preset is not None:

            if preset not in self._dpreset.keys():
                msg = (
                    "Available preset values are:\n"
                    f"\t- {sorted(self._dpreset.keys())}\n"
                    f"\t- Provided: {preset}"
                )
                raise Exception(msg)

            ids = sorted(self._dpreset[preset].keys())
        self._preset = preset

        # Add idd if relevant
        if hasattr(idd, 'close') or shot is not None:
            name = self.add_idd(
                idd=idd,
                shot=shot, run=run,
                refshot=refshot, refrun=refrun,
                user=user, database=database,
                version=version,
                backend=backend,
                ref=ref,
                return_name=True,
            )
            idd = name

        if idd is None and ids is not None:
            if self._refidd is None:
                lstr = [(k, v.get('ref', None)) for k, v in self._didd.items()]
                msg = (
                    "No idd was provided (and ref idd is not clear)!\n"
                    "Please provide an idd either directly or via \n"
                    "args (shot, user, database...)!\n"
                    "\t- {lstr}"
                )
                raise Exception(msg)
            idd = self._refidd
        elif idd is not None:
            assert idd in self._didd.keys()

        # Add ids
        if ids is not None:
            dids = self._checkformat_ids(ids, occ=occ, idd=idd, isget=isget)
            self._dids.update(dids)
            if get:
                self.open_get_close()

    def add_ids_base(
        self, occ=None, idd=None,
        shot=None, run=None, refshot=None, refrun=None,
        user=None, database=None, version=None, backend=None,
        ref=None, isget=None, get=None,
    ):
        """ Add th list of ids stored in self._IDS_BASE

        Typically used to add a list of common ids without having to re-type
        them every time
        """
        self.add_ids(
            ids=self._IDS_BASE, occ=occ, idd=idd,
            shot=shot, run=run, refshot=refshot, refrun=refrun,
            user=user, database=database, version=version, backend=backend,
            ref=ref, isget=isget, get=get,
        )

    def add_ids_synthdiag(
        self, ids=None, occ=None, idd=None,
        shot=None, run=None, refshot=None, refrun=None,
        user=None, database=None, version=None, backend=None,
        ref=None, isget=None, get=None,
    ):
        """ Add pre-tabulated input ids necessary for calculating synth. signal

        The necessary input ids are given by self.get_inputs_for_synthsignal()

        """
        if get is None:
            get = True
        ids = self.get_inputs_for_synthsignal(ids=ids, verb=False,
                                              returnas=list)
        self.add_ids(
            ids=ids, occ=occ, idd=idd, preset=None,
            shot=shot, run=run, refshot=refshot, refrun=refrun,
            user=user, database=database,
            version=version, backend=backend,
            ref=ref, isget=isget, get=get,
        )

    def remove_ids(self, ids=None, occ=None):
        """ Remove an ids (optionally remove only an occurence)

        If all the ids linked to an idd are removed, said idd is removed too

        """
        if ids is not None:
            if not ids in self._dids.keys():
                msg = "Please provide the name (str) of a valid ids\n"
                msg += "Currently available ids are:\n"
                msg += "    - %s"%str(sorted(self._dids.keys()))
                raise Exception(msg)
            occref = self._dids[ids]['occ']
            if occ is None:
                occ = occref
            else:
                occ = np.unique(np.r_[occ].astype(int))
                occ = np.intersect1d(occ, occref, assume_unique=True)
            idd = self._dids[ids]['idd']
            lids = [k for k,v in self._dids.items() if v['idd']==idd]
            if lids == [ids]:
                del self._didd[idd]
            if np.all(occ == occref):
                del self._dids[ids]
            else:
                isgetref = self._dids[ids]['isget']
                indok = np.array([ii for ii in range(0,occref.size)
                                  if occref[ii] not in occ])
                self._dids[ids]['ids'] = [self._dids[ids]['ids'][ii]
                                              for ii in indok]
                self._dids[ids]['occ'] = occref[indok]
                self._dids[ids]['isget'] = isgetref[indok]
                self._dids[ids]['nocc'] = self._dids[ids]['occ'].size

    def get_ids(self, ids=None, occ=None):
        if ids is None and len(self._dids.keys()) == 1:
            ids = list(self._dids.keys())[0]
        assert ids in self._dids.keys()
        if occ is None:
            occ = self._dids[ids]['occ'][0]
        else:
            assert occ in self._dids[ids]['occ']
        indoc = np.where(self._dids[ids]['occ'] == occ)[0][0]
        return self._dids[ids]['ids'][indoc]


    #---------------------
    # Methods for showing content
    #---------------------

    def get_summary(self, sep='  ', line='-', just='l',
                    table_sep=None, verb=True, return_=False):
        """ Summary description of the object content as a np.array of str """
        if table_sep is None:
            table_sep = '\n\n'

        # -----------------------
        # idd
        a0 = []
        if len(self._didd) > 0:
            c0 = ['idd', 'user', 'database', 'version',
                  'shot', 'run', 'refshot', 'refrun', 'isopen', '']
            for k0,v0 in self._didd.items():
                lu = ([k0] + [str(v0['params'][k]) for k in c0[1:-2]]
                      + [str(v0['isopen'])])
                ref = '(ref)' if k0==self._refidd else ''
                lu += [ref]
                a0.append(lu)
            a0 = np.array(a0, dtype='U')

        # -----------------------
        # ids
        if len(self._dids) > 0:
            c1 = ['ids', 'idd', 'occ', 'isget']
            llids = self._checkformat_get_ids()
            a1 = []
            for (k0, lk1) in llids:
                for ii in range(0,len(lk1)):
                    idd = k0 if ii == 0 else '"'
                    a1.append([lk1[ii], idd,
                               str(self._dids[lk1[ii]]['occ']),
                               str(self._dids[lk1[ii]]['isget'])])
            a1 = np.array(a1, dtype='U')
        else:
            a1 = []


        # Out
        if verb or return_ in [True,'msg']:
            if len(self._didd) > 0:
                msg0 = self._getcharray(a0, c0,
                                        sep=sep, line=line, just=just)
            else:
                msg0 = ''
            if len(self._dids) > 0:
                msg1 = self._getcharray(a1, c1,
                                        sep=sep, line=line, just=just)
            else:
                msg1 = ''
            if verb:
                msg = table_sep.join([msg0,msg1])
                print(msg)
        if return_ != False:
            if return_ == True:
                out = (a0, a1, msg0, msg1)
            elif return_ == 'array':
                out = (a0, a1)
            elif return_ == 'msg':
                out = table_sep.join([msg0,msg1])
            else:
                lok = [False, True, 'msg', 'array']
                raise Exception("Valid return_ values are: %s"%str(lok))
            return out

    def __repr__(self):
        if hasattr(self, 'get_summary'):
            return self.get_summary(return_='msg', verb=False)
        else:
            return object.__repr__(self)

    #---------------------
    # Methods for returning data
    #---------------------

    # DEPRECATED ?
    def _checkformat_getdata_indt(self, indt):
        msg = "Arg indt must be a either:\n"
        msg += "    - None: all channels used\n"
        msg += "    - int: times to use (index)\n"
        msg += "    - array of int: times to use (indices)"
        lc = [type(indt) is None, type(indt) is int, hasattr(indt,'__iter__')]
        if not any(lc):
            raise Exception(msg)
        if lc[1] or lc[2]:
            indt = np.r_[indt].rave()
            lc = [indt.dtype == np.int]
            if not any(lc):
                raise Exception(msg)
            assert np.all(indt>=0)
        return indt

    def _set_fsig(self):
        for ids in self._dshort.keys():
            for k, v in self._dshort[ids].items():
                self._dshort[ids][k]['fsig'] = _comp.get_fsig(v['str'])

    @classmethod
    def get_units(cls, ids, sig, force=None):
        return _comp.get_units(ids, sig,
                               dshort=cls._dshort, dcomp=cls._dcomp,
                               force=force)

    def get_data(self, dsig=None, occ=None,
                 data=None, units=None,
                 indch=None, indt=None, stack=None,
                 isclose=None, flatocc=True,
                 nan=None, pos=None, empty=None, strict=None,
                 return_all=None, warn=None):
        """ Return a dict of the desired signals extracted from specified ids

        For each signal, loads the data and / or units
        If the ids has a field 'channel', indch is used to specify from which
        channel data shall be loaded (all by default)

        Parameters
        ----------
        ids:        None / str
            ids from which the data should be loaded
            ids should be available (check self.get_summary())
            ids should be loaded if not available, using:
                - self.add_ids() to add the ids
                - self.open_get_close() to force loading if necessary
        sig:        None / str / list
            shortcuts of signals to be loaded from the ids
            Check available shortcuts using self.get_shortcuts(ids)
            You can add custom shortcuts if needed (cf. self.add_shortcuts())
            sig can be a single str (shortcut) or a list of such
        occ:        None / int
            occurence from which to load the data
        data:       None / bool
            Flag indicating whether to load the data
        units:      None / bool
            Flag indicating whether to load the units
        indch:      None / list / np.ndarray
            If the data has channels, this lists / array of int indices can be
            used to specify which channels to load from (all if None)
        indt:       None / list / np.ndarray
            If data is time-dependent, the list / array of int indices can be
            used to specify which time steps to load
        stack:      bool
            Flag indicating whether common data (e.g.: data from different
            channels) should be agregated / stacked into a single array
        isclose:    None / bool
            Flag indicating whether the agregated data is a collection of
            identical vectors, if which case it will be checked (np.isclose())
            and only a single vector will be kept
        flatocc:    bool
            By default, the data is returned as a list for each occurence
            If there is only one occ and flatocc = True, only the first element
            of the list is returned
        nan:        bool
            Flag indicating whether to check for abs(data) > 1.30
            All data is this case will be set to nan
            Due to the fact IMAS default value is 1.e49
        pos:        None / bool
            Flag indicating whether the data should be positive (negative
            values will be set to nan)
        empty:      None / bool
            Check whether the loaded data array ie empty (or full of nans)
                If so, a flag isempty is set to True
        return_all: bool
            Flag indicating whether to return only dout or also dfail and dsig
        warn:       bool
            Flag indicating whether to print warning messages for data could
            not be retrieved

        Return
        ------
        dout:   dict
            Dictionnary containing the loaded data
        dfail:  dict, only returned in return_all = True
            Dictionnary of failed data loading, with error messages
        dsig:  dict, only returned in return_all = True
            Dictionnary of requested signals, occ, indt, indch

        """
        return _comp.get_data_units(
            dsig=dsig,
            occ=occ,
            data=data,
            units=units,
            indch=indch,
            indt=indt,
            stack=stack,
            isclose=isclose,
            flatocc=flatocc,
            nan=nan,
            pos=pos,
            empty=empty,
            strict=strict,
            warn=warn,
            dids=self._dids,
            dshort=self._dshort,
            dcomp=self._dcomp,
            dall_except=self._dall_except,
            return_all=return_all)

    def get_events(self, occ=None, verb=True, returnas=False):
        """ Return chronoligical events stored in pulse_schedule

        If verb = True              => print (default)
                  False             => don't print
        If returnas = list          => return as list of tuples (name, time)
                      tuple         => return as tuple of (names, times)
                      False         => don't return (default)
        """

        # Check / format inputs
        if verb is None:
            verb = True
        if returnas is None:
            returnas = False
        assert isinstance(verb, bool)
        assert returnas in [False, list, tuple, str]

        # Get events and sort
        dout = self.get_data(
            dsig={'pulse_schedule': ['events_names', 'events_times']},
            occ=occ, nan=False, pos=False, stack=True,
            empty=True, strict=True, return_all=False
        )['pulse_schedule']

        names, times = dout['events_names']['data'], dout['events_times']
        tunits = times['units']
        times = times['data']
        c0 = len(names) == len(times)
        if not c0:
            msg = ("events names and times seem incompatible!\n"
                   + "\t- len(events_names['data']) = {}\n".format(len(names))
                   + "\t- len(events_times['data']) = {}".format(len(times)))
            raise Exception(msg)
        if np.size(names) == 0:
            msg = ("ids pulse_schedule has no events!\n"
                   + "\t- len(events_names['data']) = {}\n".format(len(names))
                   + "\t- len(events_times['data']) = {}".format(len(times)))
            raise Exception(msg)
        ind = np.argsort(times)
        names, times = names[ind], times[ind]

        # print and / or return as list / tuple
        if verb is True or returnas is str:
            msg = np.array([range(times.size), names, times], dtype='U').T
            length = np.nanmax(np.char.str_len(msg))
            msg = np.char.ljust(msg, length)
            msg = ('index'.ljust(length) + ' name'.ljust(length)
                   + '  time ({})'.format(tunits).ljust(length)
                   + '\n' + ' '.join(['-'*length for ii in [0, 1, 2]]) + '\n'
                   + '\n'.join([' '.join(aa) for aa in msg]))
        if verb is True:
            print(msg)
        if returnas is list:
            return list(zip(names, times))
        elif returnas is tuple:
            return names, times
        elif returnas is str:
            return msg

    #---------------------
    # Methods for exporting to tofu objects
    #---------------------

    def get_lidsidd_shotExp(self, lidsok,
                            errshot=None, errExp=None, upper=True):
        return _comp_toobjects.get_lidsidd_shotExp(
            lidsok,
            errshot=errshot, errExp=errExp, upper=upper,
            dids=self._dids, didd=self._didd)

    def _get_t0(self, t0=None, ind=None):
        if ind is None:
            ind = False
        assert ind is False or isinstance(ind, int)
        if t0 is None:
            t0 = _defimas2tofu._T0
        elif t0 != False:
            if type(t0) in [int, float, np.int_, np.float_]:
                t0 = float(t0)
            elif type(t0) is str:
                t0 = t0.strip()
                c0 = (len(t0.split('.')) <= 2
                      and all([ss.isdecimal() for ss in t0.split('.')]))
                if 'pulse_schedule' in self._dids.keys():
                    events = self.get_data(
                        dsig={'pulse_schedule': ['events_names',
                                                 'events_times']},
                        return_all=False,
                    )['pulse_schedule']
                    names = np.char.strip(events['events_names']['data'])
                    if t0 in names:
                        indt = np.nonzero(names == t0)[0]
                        if ind is not False:
                            indt = indt[ind]
                        t0 = events['events_times']['data'][indt]
                    elif c0:
                        t0 = float(t0)
                    else:
                        msg = ("Desired event ({}) unavailable!\n".format(t0)
                               + "  Please choose from:\n"
                               + self.get_events(verb=False, returnas=str))
                        raise Exception(msg)
                elif c0:
                    t0 = float(t0)
                else:
                    msg = ("Desired t0 ({}) not loaded".format(t0)
                           + " because ids 'pulse_schedule' not loaded\n"
                           + "  => setting t0 = False")
                    warnings.warn(msg)
                    t0 = False
            else:
                t0 = False
            if t0 is False:
                msg = "t0 set to False because could not be interpreted !"
                warnings.warn(msg)
        return t0

    def to_Config(self, Name=None, occ=None,
                  description_2d=None, mobile=None, plot=True):
        """ Export the content of wall ids as a tofu Config object

        Choose the occurence (occ), and index (description_2d, cf. dd_doc) to
        be exported.
        Specify whether to pick from limiter or mobile
        If not specified, will be decided automatically from the content
        Optionally plot the result

        This requires that the wall ids was previously loaded.
        If not run:
            self.add_ids('wall')
        """
        lidsok = ['wall']

        # ---------------------------
        # Preliminary checks on data source consistency
        lids, lidd, shot, Exp = _comp_toobjects.get_lidsidd_shotExp(
            lidsok,
            errshot=False, errExp=False, upper=True,
            dids=self._dids, didd=self._didd)

        # ----------------
        #   Trivial case
        if 'wall' not in lids:
            if plot:
                msg = "ids 'wall' has not been loaded => Config not available!"
                raise Exception(msg)
            return None

        # ----------------
        #   Get relevant occurence and description_2d

        ids = 'wall'
        # occ = np.ndarray of valid int
        occ = _comp._checkformat_getdata_occ(occ, ids, dids=self._dids)
        assert occ.size == 1, "Please choose one occ only !"
        occ = occ[0]
        indoc = np.nonzero(self._dids[ids]['occ'] == occ)[0][0]

        if description_2d is None:
            if len(self._dids[ids]['ids'][indoc].description_2d) >= 2:
                description_2d = 1
            else:
                description_2d = 0

        ndescript = len(self._dids[ids]['ids'][indoc].description_2d)
        if ndescript < description_2d+1:
            msg = ("Requested description_2d not available!\n"
                   + "\t- len(wall[].description_2d) = {}\n".format(indoc,
                                                                    ndescript)
                   + "\t- required description_2d: {}".format(description_2d))
            raise Exception(msg)

        # ----------------
        # Extract all relevant structures
        import tofu.geom as mod
        wall = self._dids[ids]['ids'][indoc].description_2d[description_2d]
        kwargs = dict(Exp=Exp, Type='Tor')
        lS = _comp_toobjects.config_extract_lS(ids, occ, wall, description_2d,
                                               mod, kwargs=kwargs,
                                               mobile=mobile)

        # ----------------
        # Define Config from structures and Name
        if Name is None:
            Name = wall.type.name
            if Name == '':
                Name = 'imas wall'
        if '_' in Name:
            Name = Name.strip('_')
            ln = Name.split('_')
            if len(ln) > 1:
                for ii, nn in enumerate(ln[1:]):
                    if nn[0].islower():
                        ln[ii+1] = nn.capitalize()
                Name = ''.join(ln)
        if ' ' in Name:
            Name = Name.strip(' ')
            ln = Name.split(' ')
            if len(ln) > 1:
                for ii, nn in enumerate(ln[1:]):
                    if nn[0].islower():
                        ln[ii+1] = nn.capitalize()
                Name = ''.join(ln)
        config = mod.Config(lStruct=lS, Name=Name, **kwargs)

        # Output
        if plot is True:
            lax = config.plot()
        return config

    # TBF
    def get_mesh_from_ggd(path_to_ggd, ggdindex=0):
        pass

    def _get_dextra(self, dextra=None, fordata=False,
                    nan=True, pos=None, stack=None):

        if stack is None:
            stack = True

        # -------------
        # Check / format dextra
        dextra = _comp_toobjects.extra_checkformat(
            dextra, fordata=fordata,
            dids=self._dids, didd=self._didd, dshort=self._dshort)

        # -------------
        # Trival case
        if dextra in [None, (None, None)] or len(dextra) == 0:
            if fordata:
                return None
            else:
                return None, None

        # -------------
        # Loading data
        if fordata is True:
            dout = {}
            for ids, vv in dextra.items():
                vs = [vvv if type(vvv) is str else vvv[0] for vvv in vv]
                vc = ['k' if type(vvv) is str else vvv[1] for vvv in vv]
                out = self.get_data(
                    dsig={ids: vs}, nan=nan,
                    pos=pos, stack=stack,
                    return_all=False,
                )[ids]

                inds = [ii for ii in range(0, len(vs)) if vs[ii] in out.keys()]
                _comp_toobjects.extra_get_fordataTrue(
                    inds, vs, vc, out, dout,
                    ids=ids, dshort=self._dshort, dcomp=self._dcomp)
            return dout

        else:
            d0d, dt0 = {}, {}
            for ids, vv in dextra.items():
                vs = [vvv if type(vvv) is str else vvv[0] for vvv in vv]
                vc = ['k' if type(vvv) is str else vvv[1] for vvv in vv]
                out = self.get_data(
                    dsig={ids: vs}, nan=nan,
                    pos=pos, stack=stack,
                    return_all=False,
                )[ids]

                _comp_toobjects.extra_get_fordataFalse(
                    out, d0d, dt0,
                    ids=ids, dshort=self._dshort, dcomp=self._dcomp)
            return d0d, dt0

    def to_Plasma2D(
        self,
        # dict of signals to be extracted
        dsig=None,
        # time parameters
        tlim=None,
        t0=None,
        indt0=None,
        indevent=None,
        # other parameters
        occ=None,
        stack=None,
        dextra=None,
        isclose=None,
        nan=True,
        pos=None,
        empty=None,
        strict=None,
        flatocc=None,
        # plotting parameters
        plot=None,
        plot_sig=None,
        plot_X=None,
        bck=True,
    ):
        """ Export the content of some ids as a tofu Plasma2D object

        Some ids typically contain plasma 1d (radial) or 2d (mesh) profiles
        They include for example ids:
            - core_profiles
            - core_sources
            - edge_profiles
            - edge_sources
            - equilibrium

        tofu offers a class for handling multiple profiles characterizing a
        plasma, it's called Plasma2D
        This method automatically identifies the ids that may contain profiles,
        extract all profiles (i.e.: all profiles identified by a shortcut, see
        self.get_shortcuts()) and export everything to a fresh Plasma2D
        instance.

        Parameters
        ----------
        tlim:   None / list
            Restrict the loaded data to a time interval with tlim
            if None, loads all time steps
        dsig:   None / dict
            Specify exactly which data (shortcut) should be loaded by ids
            If None, loads all available data
        t0:     None / float / str
            Specify a time to be used as source:
                - None: absolute time vectors are untouched
                - float : the roigin of all time vectors is set to t0
                - str : the source is taken from an event in ids pulse_schedule
        Name:   None / str
            Name to be given to the instance
            If None, a default Name is built
        occ:    None / int
            occurence to be used for loading the data
        config: None / Config
            Configuration (i.e.: database geometry) to be used for the instance
            If None, created from the wall ids with self.to_Config().
        out:    type
            class with which the output shall be returned
                - object :  as a Plasma2D instance
                - dict:     as a dict
        description_2d: None / int
            description_2d index to be used if the Config is to be built from
            wall ids. See self.to_Config()
        plot:       None / bool
            Flag whether to plot the result
        plot_sig:   None / str
            shortcut of the signal to be plotted, if any
        plot_X:     None / str
            shortcut of the abscissa against which to plot the signal, if any
        bck:        bool
            Flag indicating whether to plot the grey envelop of the signal as a
            background, if plot is True
        dextra:     None / dict
            dict of extra signals (time traces) to be plotted, for context
        shapeRZ:    None / tuple
            If provided, tuple indicating the order of 2d data arrays
            associated to rectangular meshes
            Only necessary when shape cannot be infered from data shape
                - ('R', 'Z'): first dimension is R, second Z
                - ('Z', 'R'): the other way around

        Args nan and pos are fed to self.get_data()

        Return
        ------
        plasma:     dict / Plasma2D
            dict or Plasma2D instance depending on out

        """

        # ---------------------------
        # Preliminary checks

        # check and format dsig
        dsig = _comp_toobjects.plasma_checkformat_dsig(
            dsig,
            lidsplasma=self._lidsplasma, dids=self._dids,
            dshort=self._dshort, dcomp=self._dcomp)

        # check and format plot arguments
        # plot, plot_X, plot_sig = _comp_toobjects.plasma_plot_args(
            # plot, plot_X, plot_sig,
            # dsig=dsig)

        # lids
        lids = sorted(dsig.keys())

        # data source consistency
        _, _, shot, Exp = _comp_toobjects.get_lidsidd_shotExp(
            lids, upper=True, errshot=False, errExp=False,
            dids=self._dids, didd=self._didd,
        )

        # -------------
        #   get all relevant data

        out0 = self.get_data(
            dsig=dsig,
            nan=nan,
            pos=pos,
            empty=empty,
            isclose=isclose,
            strict=True,
            return_all=False,
        )

        # -------------
        #   Input dicts

        # config
        # if config is None:
            # try:
                # config = self.to_Config(
                    # Name=Name,
                    # occ=occ,
                    # description_2d=description_2d,
                    # plot=False)
            # except Exception as err:
                # msg = (str(err)
                       # + "\nCould not load wall from wall ids\n"
                       # + "  => No config provided to Plasma2D instance!")
                # warnings.warn(msg)

        # dextra
        d0d, dtime0 = self._get_dextra(dextra)

        # get data
        return _comp_toobjects.get_plasma(
            # ressources
            multi=self,
            dtime0=dtime0,
            d0d=d0d,
            out0=out0,
            lids=lids,
            # time parameters
            tlim=tlim,
            indt0=indt0,
            t0=t0,
            indevent=indevent,
            # other parameters
            nan=nan,
            pos=pos,
            stack=stack,
            isclose=isclose,
            empty=empty,
            strict=strict,
            # plotting
            plot=plot,
            plot_sig=plot_sig,
        )

    def inspect_channels(self, ids=None, occ=None, indch=None, geom=None,
                         dsig=None, data=None, X=None, stack=None,
                         datacls=None, geomcls=None,
                         return_dict=None, return_ind=None,
                         return_msg=None, verb=None):

        # ------------------
        # Preliminary checks
        if return_dict is None:
            return_dict = False
        if return_ind is None:
            return_ind = False
        if return_msg is None:
            return_msg = False
        if verb is None:
            verb = True
        if occ is None:
            occ = 0
        if geom is None:
            geom = True
        compute_ind = return_ind or return_msg or verb

        # Check ids is relevant
        idsok = set(self._lidsdiag).intersection(self._dids.keys())
        if ids is None and len(idsok) == 1:
            ids = next(iter(idsok))

        # Check ids has channels (channel, gauge, ...)
        lch = ['channel', 'gauge', 'group', 'antenna',
               'pipe', 'reciprocating', 'bpol_probe']
        ind = [ii for ii in range(len(lch))
               if hasattr(self._dids[ids]['ids'][occ], lch[ii])]
        if len(ind) == 0:
            msg = "ids {} has no attribute with '[chan]' index!".format(ids)
            raise Exception(msg)
        nch = len(getattr(self._dids[ids]['ids'][0], lch[ind[0]]))
        if nch == 0:
            msg = ('ids {} has 0 channels:\n'.format(ids)
                   + '\t- len({}.{}) = 0\n'.format(ids, lch[ind[0]])
                   + '\t- idd: {}'.format(self._dids[ids]['idd']))
            raise Exception(msg)

        datacls, geomcls, dsig = _comp_toobjects.data_checkformat_dsig(
            ids=ids, dsig=dsig, data=data, X=X,
            datacls=datacls, geomcls=geomcls,
            lidsdiag=self._lidsdiag, dids=self._dids, didsdiag=self._didsdiag,
            dshort=self._dshort, dcomp=self._dcomp)

        if geomcls is False:
            geom = False

        # ------------------
        # Extract sig and shapes / values
        if geom == 'only':
            lsig = []
        else:
            lsig = sorted(dsig.values())
        lsigshape = list(lsig)
        if geom in ['only', True] and 'LOS' in geomcls:
            lkok = set(self._dshort[ids].keys()).union(self._dcomp[ids].keys())
            lsig += [ss for ss in ['los_ptsRZPhi', 'etendue',
                                   'surface', 'names']
                     if ss in lkok]

        # STcak has to be False for inspect_channels...
        # if stack is None:
            # stack = self._didsdiag[ids].get('stack', False)
        out = self.get_data(dsig={ids: lsig},
                            isclose=False, stack=False,
                            nan=True, pos=False, return_all=False)[ids]

        # --------------
        # dout, indchout
        dout, indchout = _comp_toobjects.inspect_channels_dout(
            ids=ids, indch=indch, geom=geom,
            out=out, nch=nch, dshort=self._dshort,
            lsig=lsig, lsigshape=lsigshape,
            compute_ind=compute_ind)

        # msg
        msg = None
        if return_msg is True or verb is True:
            col = ['index'] + [k0 for k0 in dout.keys()]
            ar = ([np.arange(nch)]
                  + [['{} {}'.format(tuple(v0['shapes'][ii]), 'nan')
                      if v0['isnan'][ii] else str(tuple(v0['shapes'][ii]))
                      for ii in range(nch)]
                     if 'shapes' in v0.keys()
                     else v0['value'].astype(str) for v0 in dout.values()])
            msg = self._getcharray(ar, col, msg=True)
            if verb is True:
                indstr = ', '.join(map(str, indchout))
                msg += "\n\n => recommended indch = [{}]".format(indstr)
                print(msg)

        # ------
        # Return
        lv = [(dout, return_dict), (indchout, return_ind), (msg, return_msg)]
        lout = [vv[0] for vv in lv if vv[1] is True]
        if len(lout) == 1:
            return lout[0]
        elif len(lout) > 1:
            return lout

    def _to_Cam_Du(self, ids, lk, indch):
        out = self.get_data(dsig={ids: list(lk)}, indch=indch,
                            nan=True, pos=False, stack=True,
                            empty=True, strict=True,
                            return_all=False)[ids]
        return _comp_toobjects.cam_to_Cam_Du(out, ids=ids)

    def to_Cam(self, ids=None, indch=None, indch_auto=False,
               description_2d=None, stack=None,
               Name=None, occ=None, config=None,
               plot=True, nan=True, pos=None):
        """ Export the content of a diagnostic ids as a tofu CamLos1D instance

        Some ids contain the geometry of a diagnostics
        They typically have a 'channels' field
        Generally in the form of a set of Lines of Sights (LOS)
        They include for example ids:
            - interferometer
            - polarimeter
            - bolometer
            - soft_x_rays
            - bremsstrahlung_visible
            - spectrometer_visible

        tofu offers a class for handling sets fo LOS as a camera: CamLOS1D
        This method extracts the geometry of the desired diagnostic (ids) and
        exports it as a CamLOS1D instance.

        Parameters
        ----------
        ids:   None / str
            Specify the ids (will be checked against known diagnostics ids)
            Should have a 'channels' field
            If None and a unique diagnostic ids has been added, set to this one
        Name:   None / str
            Name to be given to the instance
            If None, a default Name is built
        occ:    None / int
            occurence to be used for loading the data
        indch:  None / list / array
            If provided, array of int indices specifying which channels shall
            be loaded (fed to self.get_data())
        indch_auto: bool
            If True and indch is not provided, will try to guess which channels
            can be loaded. If possible all channels are loaded by default, but
            only if they have uniform data (same shape, i.e.: same time
            vectors). In case of channels with non-uniform data, will try to
            identify a sub-group of channels with uniform data
        config: None / Config
            Configuration (i.e.: database geometry) to be used for the instance
            If None, created from the wall ids with self.to_Config().
        description_2d: None / int
            description_2d index to be used if the Config is to be built from
            wall ids. See self.to_Config()
        plot:       None / bool
            Flag whether to plot the result

        Args nan and pos are fed to self.get_data()

        Return
        ------
        cam:     CamLOS1D
            CamLOS1D instance

        """
        # Check ids
        idsok = set(self._lidslos).intersection(self._dids.keys())
        if ids is None and len(idsok) == 1:
            ids = next(iter(idsok))

        # dsig
        geom = _comp_toobjects.cam_checkformat_geom(
            ids, geomcls=None, indch=indch,
            lidsdiag=self._lidsdiag, dids=self._dids, didsdiag=self._didsdiag)

        if Name is None:
            Name = 'custom'
        if stack is None:
            stack = self._didsdiag[ids].get('stack', True)

        # ---------------------------
        # Preliminary checks on data source consistency
        _, _, shot, Exp = _comp_toobjects.get_lidsidd_shotExp(
            [ids],
            errshot=False, errExp=False, upper=True,
            dids=self._dids, didd=self._didd)

        # -------------
        #   Input dicts

        # config
        if config is None:
            config = self.to_Config(Name=Name, occ=occ,
                                    description_2d=description_2d, plot=False)

        # dchans
        dchans = {}
        if indch is not None:
            dchans['ind'] = indch

        # cam
        cam = None
        nchMax = len(self._dids[ids]['ids'][0].channel)
        Etendues, Surfaces = None, None
        if config is None:
            msg = "A config must be provided to compute the geometry !"
            raise Exception(msg)

        if 'LOS' in geom:
            # Check channel indices
            indchr = self.inspect_channels(ids, indch=indch,
                                           geom='only', return_ind=True,
                                           verb=False)
            indch = _comp_toobjects.cam_compare_indch_indchr(
                indch, indchr, nchMax,
                indch_auto=indch_auto)

            # Load geometrical data
            lk = ['los_ptsRZPhi', 'etendue', 'surface', 'names']
            lkok = set(self._dshort[ids].keys())
            lkok = lkok.union(self._dcomp[ids].keys())
            lk = list(set(lk).intersection(lkok))
            dgeom, Etendues, Surfaces, names = self._to_Cam_Du(ids, lk, indch)

            if names is not None:
                dchans['names'] = names

        import tofu.geom as tfg
        cam = getattr(tfg, geom)(dgeom=dgeom, config=config,
                                 Etendues=Etendues, Surfaces=Surfaces,
                                 Name=Name, Diag=ids, Exp=Exp,
                                 dchans=dchans)
        cam.Id.set_dUSR( {'imas-nchMax': nchMax} )

        if plot is True:
            cam.plot_touch(draw=True)
        return cam

    def get_tlim(
        self,
        t=None,
        tlim=None,
        indevent=None,
        returnas=None,
    ):
        """ Retrun the time indices corresponding to the desired time limts

        Return a dict with:
            'tlim': the requested time interval as a list of len = 2
            't':    the resulting time vector
            'nt':   the rersulting number of time steps
            'indt': the resulting time index

        The indices 'indt' can be returned as a bool or int array

        tlim can be:
            - False: no time limit
            - None: set to default (False)
            - a list [t0, t1] of len = 2, where t0 and t1 can be:
                None : no lower / upper limit
                float: a time value
                str:    a valid event name from ids pulse_schedule

        """
        names, times = None, None
        c0 = (isinstance(tlim, list)
              and all([type(tt) in [float, int, np.float_, np.int_]
                       for tt in tlim]))
        if not c0 and 'pulse_schedule' in self._dids.keys():
            try:
                names, times = self.get_events(verb=False, returnas=tuple)
            except Exception as err:
                msg = (str(err)
                       + "\nEvents not loaded from ids pulse_schedule!")
                warnings.warn(msg)
        if 'pulse_schedule' in self._dids.keys():
            idd = self._dids['pulse_schedule']['idd']
            Exp = self._didd[idd]['params']['database']
        else:
            Exp = None
        return _comp_toobjects.data_checkformat_tlim(t, tlim=tlim,
                                                     names=names, times=times,
                                                     indevent=indevent,
                                                     returnas=returnas,
                                                     Exp=Exp)

    def to_Data(self, ids=None, dsig=None, data=None, X=None, tlim=None,
                indch=None, indch_auto=False, Name=None, occ=None,
                config=None, description_2d=None, stack=None,
                dextra=None, t0=None, indt0=None, datacls=None, geomcls=None,
                plot=True, bck=True, fallback_X=None,
                nan=True, pos=None, empty=None, strict=None,
                return_indch=False, indevent=None):
        """ Export the content of a diagnostic ids as a tofu DataCam1D instance

        Some ids contain the geometry and data of a diagnostics
        They typically have a 'channels' field
        They include for example ids:
            - interferometer
            - polarimeter
            - bolometer
            - soft_x_rays
            - bremsstrahlung_visible
            - spectrometer_visible
            - reflectometer_profile
            - ece
            - magnetics
            - barometry
            - neutron_diagnostics

        tofu offers a class for handling data: DataCam1D
        If available, this method also loads the geometry using self.to_Cam()
        on the same ids.
        But it will load the data even if no geometry (LOS) is available.
        This method extracts the data of the desired diagnostic (ids) and
        exports it as a DataCam1D instance.

        Parameters
        ----------
        ids:   None / str
            Specify the ids (will be checked against known diagnostics ids)
            Should have a 'channels' field
            If None and a unique diagnostic ids has been added, set to this one
        Name:   None / str
            Name to be given to the instance
            If None, a default Name is built
        occ:    None / int
            occurence to be used for loading the data
        indch:  None / list / array
            If provided, array of int indices specifying which channels shall
            be loaded (fed to self.get_data())
        indch_auto: bool
            If True and indch is not provided, will try to guess which channels
            can be loaded. If possible all channels are loaded by default, but
            only if they have uniform data (same shape, i.e.: same time
            vectors). In case of channels with non-uniform data, will try to
            identify a sub-group of channels with uniform data
        dsig:   None / dict
            Specify exactly which data (shortcut) should be loaded by ids
            If None, loads all available data
        data:   None / str
            If dsig is not provided, specify the shortcut of the data to be
            loaded (from channels)
        X:      None / str
            If dsig is not provided, specify the shortcut of the data to be
            used as abscissa
        tlim:   None / list
            Restrict the loaded data to a time interval with tlim
            if None, loads all time steps
        config: None / Config
            Configuration (i.e.: database geometry) to be used for the instance
            If None, created from the wall ids with self.to_Config().
        description_2d: None / int
            description_2d index to be used if the Config is to be built from
            wall ids. See self.to_Config()
        dextra:     None / dict
            dict of extra signals (time traces) to be plotted, for context
        t0:     None / float / str
            Specify a time to be used as source:
                - None: absolute time vectors are untouched
                - float : the roigin of all time vectors is set to t0
                - str : the source is taken from an event in ids pulse_schedule
        datacls:    None / str
            tofu calss to be used for the data
                - None : determined from tabulated info (self._didsdiag[ids])
                - str  : should be a valid data class name from tofu.data
        geomcls:    None / False / str
            tofu class to be used for the geometry
                - False: geometry not loaded
                - None : determined from tabulated info (self._didsdiag[ids])
                - str  : should be a valid camera class name from tofu.geom
        fallback_X: None / float
            fallback value for X when X is nan
                X[np.isnan(X)] = fallback_X
            If None, set to 1.1*np.nanmax(X)

        return_indch:   bool
            Flag indicating whether to return also the indch
            Useful if indch was determined automatically by indch_auto
        plot:       None / bool
            Flag whether to plot the result
        bck:        bool
            Flag indicating whether to plot the grey envelop of the signal as a
            background, if plot is True

        Args nan and pos are fed to self.get_data()

        Return
        ------
        data:   DataCam1D
            DataCam1D instance
        indch:  np.ndarray
            int array of indices of the loaded channels, returned only if
            return_indch = True
        """

        # Check ids
        idsok = set(self._lidsdiag).intersection(self._dids.keys())
        if ids is None and len(idsok) == 1:
            ids = next(iter(idsok))

        # dsig
        datacls, geomcls, dsig = _comp_toobjects.data_checkformat_dsig(
            ids, dsig, data=data, X=X,
            datacls=datacls, geomcls=geomcls,
            lidsdiag=self._lidsdiag, dids=self._dids, didsdiag=self._didsdiag,
            dshort=self._dshort, dcomp=self._dcomp)

        if Name is None:
            Name = 'custom'
        if stack is None:
            stack = self._didsdiag[ids].get('stack', True)

        # ---------------------------
        # Preliminary checks on data source consistency
        _, _, shot, Exp = _comp_toobjects.get_lidsidd_shotExp(
            [ids],
            errshot=False, errExp=False, upper=True,
            dids=self._dids, didd=self._didd)

        # -------------
        #   Input dicts

        # config
        if config is None:
            config = self.to_Config(Name=Name, occ=occ,
                                    description_2d=description_2d, plot=False)

        # dchans
        if indch is not None:
            dchans = {'ind':indch}
        else:
            dchans = None

        # -----------
        # Get geom
        cam = None
        indchanstr = self._dshort[ids][dsig['data']]['str'].index('[chan]')
        chanstr = self._dshort[ids][dsig['data']]['str'][:indchanstr]
        nchMax = len(getattr(self._dids[ids]['ids'][0], chanstr))

        # Check channel indices
        indchr = self.inspect_channels(ids, indch=indch,
                                       geom=(geomcls is not False),
                                       return_ind=True,
                                       verb=False)
        indch = _comp_toobjects.cam_compare_indch_indchr(
            indch, indchr, nchMax,
            indch_auto=indch_auto)

        dgeom, names = None, None
        if geomcls is not False:
            Etendues, Surfaces = None, None
            if config is None:
                msg = "A config must be provided to compute the geometry !"
                raise Exception(msg)

            if 'LOS' in geomcls:
                lk_geom = ['los_ptsRZPhi', 'etendue', 'surface']
                lkok = set(self._dshort[ids].keys())
                lkok = lkok.union(self._dcomp[ids].keys())
                lk_geom = list(set(lk_geom).intersection(lkok))
                dgeom, Etendues, Surfaces, names = self._to_Cam_Du(
                    ids, lk_geom, indch)

        # ----------
        # Get time
        lk = sorted(dsig.keys())
        dins = dict.fromkeys(lk)
        t = self.get_data(dsig={ids: dsig.get('t', 't')},
                          indch=indch, stack=stack)[ids]['t']['data']
        if len(t) == 0:
            msg = "The time vector is not available for %s:\n"%ids
            msg += "    - 't' <=> %s.%s\n"%(ids,self._dshort[ids]['t']['str'])
            msg += "    - 't' = %s"%str(t)
            raise Exception(msg)

        # ----------
        # Get data
        out = self.get_data(dsig={ids: dsig['data']},
                            indch=indch, nan=nan, pos=pos,
                            empty=empty, strict=strict, stack=stack)[ids]
        if len(out[dsig['data']]['data']) == 0:
            msgstr = self._dshort[ids]['data']['str']
            msg = ("The data array is not available for {}:\n".format(ids)
                   + "    - 'data' <=> {}.{}\n".format(ids, msgstr)
                   + "    - 'data' = {}".format(out[dsig['data']['data']]))
            raise Exception(msg)

        if names is not None:
            dchans['names'] = names

        if t.ndim == 2:
            if not np.all(np.isclose(t, t[0:1, :])):
                msg = ("Non-identical time vectors!\n"
                       "  => double-check indch\n"
                       + str(t))
                raise Exception(msg)
            t = t[0, :]
        dins['t'] = t

        indt = self.get_tlim(t, tlim=tlim,
                             indevent=indevent, returnas=int)['indt']

        # -----------
        # Get data
        out = self.get_data(dsig={ids: [dsig[k] for k in lk]},
                            indt=indt, indch=indch, nan=nan, pos=pos,
                            stack=stack)[ids]
        for kk in set(lk).difference('t'):
            # Arrange depending on shape and field
            if type(out[dsig[kk]]['data']) is not np.ndarray:
                msg = "BEWARE : non-conform data !"
                raise Exception(msg)

            c0 = (out[dsig[kk]]['data'].size == 0
                  or out[dsig[kk]]['data'].ndim not in [1, 2, 3])
            if c0 is True:
                msg = ("\nSome data seem to have inconsistent shape:\n"
                       + "\t- out[{}].shape = {}".format(
                           dsig[kk], out[dsig[kk]]['data'].shape))
                raise Exception(msg)

            if out[dsig[kk]]['data'].ndim == 1:
                out[dsig[kk]]['data'] = np.atleast_2d(out[dsig[kk]]['data'])

            if out[dsig[kk]]['data'].ndim == 2:
                if dsig[kk] in ['X','lamb']:
                    if np.allclose(out[dsig[kk]]['data'],
                                   out[dsig[kk]]['data'][:, 0:1]):
                        dins[kk] = out[dsig[kk]]['data'][:, 0]
                    else:
                        dins[kk] = out[dsig[kk]]['data']
                else:
                    dins[kk] = out[dsig[kk]]['data'].T

            elif out[dsig[kk]]['data'].ndim == 3:
                if kk != 'data':
                    msg = ("field {} has dimension 3!\n".format(kk)
                           + "  => Only data should have dimension 3!")
                    raise Exception(msg)
                # Temporary fix until clean-uo and upgrading of _set_fsig()
                if kk == 'data' and indt is not None:
                    out[dsig[kk]]['data'] = out[dsig[kk]]['data'][:, :, indt]
                dins[kk] = np.swapaxes(out[dsig[kk]]['data'].T, 1, 2)

        # --------------------------
        # Format special ids cases
        if ids == 'reflectometer_profile':
            dins['X'] = np.fliplr(dins['X'])
            dins['data'] = np.fliplr(dins['data'])

        if 'validity_timed' in self._dshort[ids].keys():
            inan = self.get_data(dsig={ids: 'validity_timed'},
                                 indt=indt, indch=indch,
                                 nan=nan, stack=stack,
                                 pos=pos)[ids]['validity_timed']['data'].T < 0.
            dins['data'][inan] = np.nan
        if 'X' in dins.keys() and np.any(np.isnan(dins['X'])):
            if fallback_X is None:
                fallback_X = 1.1*np.nanmax(dins['X'])
            dins['X'][np.isnan(dins['X'])] = fallback_X

        # Apply indt if was not done in get_data
        for kk,vv in dins.items():
            c0 = (((vv.ndim == 2 and kk != 'lamb') or kk == 't')
                  and vv.shape[0] > indt.size)
            if c0:
                dins[kk] = vv[indt, ...]

        # dlabels
        dins['dlabels'] = dict.fromkeys(lk)
        for kk in lk:
            dins['dlabels'][kk] = {'name': dsig[kk],
                                   'units': out[dsig[kk]]['units']}

        # dextra
        dextra = self._get_dextra(dextra, fordata=True)

        # t0
        if indt0 is None:
            indt0 = 0
        t0 = self._get_t0(t0, ind=indt0)
        if t0 != False:
            if 't' in dins.keys():
                dins['t'] = dins['t'] - t0
            if dextra is not None:
                for tt in dextra.keys():
                    dextra[tt]['t'] = dextra[tt]['t'] - t0

        # --------------
        # Create objects
        if geomcls is not False and dgeom is not None:
            import tofu.geom as tfg
            cam = getattr(tfg, geomcls)(dgeom=dgeom, config=config,
                                        Etendues=Etendues, Surfaces=Surfaces,
                                        Name=Name, Diag=ids, Exp=Exp,
                                        dchans=dchans)
            cam.Id.set_dUSR({'imas-nchMax': nchMax})

        import tofu.data as tfd
        conf = None if cam is not None else config
        Data = getattr(tfd, datacls)(Name=Name, Diag=ids, Exp=Exp, shot=shot,
                                     lCam=cam, config=conf, dextra=dextra,
                                     dchans=dchans, **dins)
        Data.Id.set_dUSR( {'imas-nchMax': nchMax} )

        if plot:
            Data.plot(draw=True, bck=bck)
        if return_indch is True:
            return Data, indch
        else:
            return Data

    def calc_signal(self, ids=None, dsig=None, tlim=None, t=None, res=None,
                    quant=None, ref1d=None, ref2d=None,
                    q2dR=None, q2dPhi=None, q2dZ=None,
                    Brightness=None, interp_t=None, newcalc=True,
                    indch=None, indch_auto=False, Name=None, coefs=None,
                    occ_cam=None, occ_plasma=None, check_units=None,
                    config=None, description_2d=None, indevent=None,
                    dextra=None, t0=None, datacls=None, geomcls=None,
                    bck=True, fallback_X=None, nan=True, pos=None,
                    plot=True, plot_compare=None, plot_plasma=None):
        """ Compute synthetic data for a diagnostics and export as DataCam1D

        Some ids typically contain plasma 1d (radial) or 2d (mesh) profiles
        They include for example ids:
            - core_profiles
            - core_sources
            - edge_profiles
            - edge_sources
            - equilibrium

        From these profiles, tofu can computed syntheic data for a diagnostic
        ids which provides a geometry (channels.line_of_sight).
        tofu extracts the geometry, and integrates the desired profile along
        the lines of sight (LOS), using 2D interpolation when necessary

        It requires:
            - a diagnostic ids with geometry (LOS)
            - an ids containing the 1d or 2d profile to be integrated
            - if necessary, an intermediate ids to interpolate the 1d profile
            to 2d (e.g.: equilibrium)

        For each ids, you need to specify:
            - profile ids:
                profile (signal) to be integrated
                quantity to be used for 1d interpolation
            - equilibrium / intermediate ids:
                quantity to be used for 2d interpolation
                    (shall be the same dimension as quantity for 1d interp.)

        This method is a combination of self.to_Plasma2D() (used for extracting
        profiles and equilibrium and for interpolation) and self.to_Cam() (used
        for extracting diagnostic geometry) and to_Data() (used for exportig
        computed result as a tofu DataCam1D instance.

        Args ids, dsig, tlim, occ_plasma (occ), nan, pos, plot_plasma (plot)
        are fed to to_Plasma2D()
        Args indch, indch_auto, occ_cam (occ), config, description_2d, are fed
        to to_Cam()
        Args Name, bck, fallback_X, plot, t0, dextra are fed to to_Data()

        Parameters
        ----------
        t:      None / float / np.ndarray
            time at which the synthetic signal shall be computed
            If None, computed for all available time steps
        res:    None / float
            absolute spatial resolution (sampling steps) used for Line-of-Sight
            intergation (in meters)
        quant:  None / str
            Shortcut of the quantity to be integrated
        ref1d:  None / str
            Shortcut of the quantity to be used as reference for 1d
            interpolation
        ref2d:  None / str
            Shortcut of the quantity to be used as reference for 2d
            interpolation
        q2dR:   None / str
            If integrating an anisotropic vector field (e.g. magnetic field)
                q2dR if the shortcut of the R-component of the quantity
        q2dPhi:   None / str
            If integrating an anisotropic vector field (e.g. magnetic field)
                q2dPhi if the shortcut of the Phi-component of the quantity
        q2dR:   None / str
            If integrating an anisotropic vector field (e.g. magnetic field)
                q2dZ if the shortcut of the Z-component of the quantity
        Brightness:     bool
            Flag indicating whether the result shall be returned as a
            Brightness (i.e.: line integral) or an incident flux (Brightness x
            Etendue), which requires the Etendue
        plot_compare:   bool
            Flag indicating whether to plot the experimental data against the
            computed synthetic data
        Return
        ------
        sig:     DataCam1D
            DataCam1D instance

        """

        # Check / format inputs
        if check_units is None:
            check_units = True
        if plot is None:
            plot = True

        if plot:
            if plot_compare is None:
                plot_compare = True
            if plot_plasma is None:
                plot_plasma = True

        # Get experimental data first if relevant
        # to get correct indch for comparison
        if plot and plot_compare:
            data, indch = self.to_Data(ids, indch=indch,
                                       indch_auto=indch_auto, t0=t0,
                                       config=config, tlim=tlim,
                                       indevent=indevent,
                                       description_2d=description_2d,
                                       return_indch=True, plot=False)

        # Get camera
        cam = self.to_Cam(ids=ids, indch=indch, indch_auto=indch_auto,
                          Name=None, occ=occ_cam,
                          config=config, description_2d=description_2d,
                          plot=False, nan=True, pos=None)

        # Get relevant parameters
        dsig, dq, lq = _comp_toobjects.signal_get_synth(
            ids, dsig,
            quant, ref1d, ref2d, q2dR, q2dPhi, q2dZ,
            didsdiag=self._didsdiag, lidsplasma=self._lidsplasma,
            dshort=self._dshort, dcomp=self._dcomp)

        # Get relevant plasma
        plasma = self.to_Plasma2D(tlim=tlim, indevent=indevent,
                                  dsig=dsig, t0=t0,
                                  Name=None, occ=occ_plasma,
                                  config=cam.config, out=object,
                                  plot=False, dextra=dextra,
                                  nan=True, pos=None)

        # Intermediate computation if necessary
        ani = False
        if ids == 'bremsstrahlung_visible':
            try:
                lamb = self.get_data(dsig={ids: 'lamb'},
                                     stack=True)[ids]['lamb']['data']
            except Exception as err:
                lamb = 5238.e-10
                msg = "bremsstrahlung_visible.lamb could not be retrived!\n"
                msg += "  => fallback to lamb = 5338.e-10 m (WEST case)"
                warnings.warn(msg)
            out = plasma.compute_bremzeff(Te='core_profiles.1dTe',
                                          ne='core_profiles.1dne',
                                          zeff='core_profiles.1dzeff',
                                          lamb=lamb)
            quant, _, units = out
            source = 'f(core_profiles, bremsstrahlung_visible)'
            depend = ('core_profiles.t','core_profiles.1dTe')
            plasma.add_quantity(key='core_profiles.1dbrem', data=quant,
                                depend=depend, source=source, units=units,
                                dim=None, quant=None, name=None)
            dq['quant'] = ['core_profiles.1dbrem']

        elif ids == 'polarimeter':
            lamb = self.get_data(dsig={ids: 'lamb'},
                                 stack=True)[ids]['lamb']['data'][0]

            # Get time reference
            doutt, dtut, tref = plasma.get_time_common(lq)
            if t is None:
                t = tref

            # Add necessary 2dne (and time reference)
            ne2d, tne2d = plasma.interp_pts2profile(quant='core_profiles.1dne',
                                                    ref1d='core_profiles.1drhotn',
                                                    ref2d='equilibrium.2drhotn',
                                                    t=t, interp_t='nearest')
            # Add fanglev
            out = plasma.compute_fanglev(BR='equilibrium.2dBR',
                                         BPhi='equilibrium.2dBT',
                                         BZ='equilibrium.2dBZ',
                                         ne=ne2d, tne=tne2d, lamb=lamb)
            fangleRPZ, tfang, units = out

            plasma.add_ref(key='tfangleRPZ', data=tfang, group='time')

            source = 'f(equilibrium, core_profiles, polarimeter)'
            depend = ('tfangleRPZ','equilibrium.mesh')

            plasma.add_quantity(key='2dfangleR', data=fangleRPZ[0,...],
                                depend=depend, source=source, units=units,
                                dim=None, quant=None, name=None)
            plasma.add_quantity(key='2dfanglePhi', data=fangleRPZ[1,...],
                                depend=depend, source=source, units=units,
                                dim=None, quant=None, name=None)
            plasma.add_quantity(key='2dfangleZ', data=fangleRPZ[2,...],
                                depend=depend, source=source, units=units,
                                dim=None, quant=None, name=None)

            dq['q2dR'] = ['2dfangleR']
            dq['q2dPhi'] = ['2dfanglePhi']
            dq['q2dZ'] = ['2dfangleZ']
            dq['Type'] = ['sca']
            ani = True

        for kk,vv in dq.items():
            c0 = [vv is None,
                  type(vv) is list and len(vv) == 1 and type(vv[0]) is str]
            if not any(c0):
                msg = "All in dq must be None or list of 1 string !\n"
                msg += "    - Provided: dq[%s] = %s"%(kk,str(vv))
                raise Exception(msg)
            if vv is not None:
                dq[kk] = vv[0]

        # Check units of integrated field
        if check_units is True:
            if 'quant' in dq.keys():
                units_input = plasma._ddata[dq['quant']]['units']
            else:
                units_input = plasma._ddata[dq['q2dR']]['units']
            if any([ss in units_input for ss in ['W', 'ph', 'photons']]):
                if 'sr^-1' not in units_input:
                    dq['coefs'] = 1./(4.*np.pi)
        if ids == 'interferometer':
            # For intereferometers, the data corresponds to 2 laser passages
            dq['coefs'] = 2.
        if ids == 'polarimeter':
            # For polarimeter, the vect is along the LOS
            # it is not the direction of
            dq['coefs'] = -2.
        if coefs is not None:
            dq['coefs'] = dq.get('coefs', 1.)*coefs

        # Calculate synthetic signal
        if Brightness is None:
            Brightness = self._didsdiag[ids]['synth'].get('Brightness', None)
        dq['fill_value'] = 0.
        sig, units = cam.calc_signal_from_Plasma2D(plasma, res=res, t=t,
                                                   Brightness=Brightness,
                                                   newcalc=newcalc,
                                                   plot=False, **dq)

        sig._dextra = plasma.get_dextra(dextra)

        # Safety check regarding Brightness
        _, _, dsig_exp = _comp_toobjects.data_checkformat_dsig(
            ids, dsig=None, data=None, X=None,
            datacls=None, geomcls=None,
            lidsdiag=self._lidsdiag, dids=self._dids, didsdiag=self._didsdiag,
            dshort=self._dshort, dcomp=self._dcomp)

        kdata = dsig_exp['data']
        B_exp = self._dshort[ids][kdata].get('Brightness', None)
        err_comp = False
        if Brightness != B_exp:
            u_exp = self._dshort[ids][kdata].get('units')
            msg = ("\nCalculated synthetic and chosen experimental data "
                   + "do not seem directly comparable !\n"
                   + "\t- chosen experimental data: "
                   + "{}, ({}), Brightness = {}\n".format(kdata,
                                                          u_exp, B_exp)
                   + "\t- calculated synthetic data: "
                   + "int({}), ({}), Brightness = {}\n".format(dq['quant'],
                                                               units,
                                                               Brightness)
                   + "\n  => Consider changing data or Brigthness value")
            err_comp = True
            warnings.warn(msg)

        # plot
        if plot:
            if plot_compare:
                if err_comp:
                    raise Exception(msg)
                sig._dlabels = data.dlabels
                data.plot_compare(sig, bck=bck)
            else:
                sig.plot(bck=bck)
            c0 = (plot_plasma
                  and dq.get('quant') is not None and '1d' in dq['quant'])
            if c0 is True:
                plasma.plot(dq['quant'], X=dq['ref1d'], bck=bck)
        return sig


#############################################################
#############################################################
#           Function-oriented interfaces to IdsMultiLoader
#############################################################


def load_Config(
    shot=None, run=None, user=None, database=None, version=None, backend=None,
    Name=None, occ=0, description_2d=None, plot=True,
):

    didd = MultiIDSLoader()
    didd.add_idd(
        shot=shot, run=run,
        user=user, database=database,
        version=version, backend=backend,
    )
    didd.add_ids('wall', get=True)

    return didd.to_Config(Name=Name, occ=occ,
                          description_2d=description_2d, plot=plot)


# occ ?
def load_Plasma2D(shot=None, run=None, user=None, database=None,
                  version=None, backend=None,
                  tlim=None, occ=None, dsig=None, ids=None,
                  config=None, description_2d=None,
                  Name=None, t0=None, out=object, dextra=None,
                  plot=None, plot_sig=None, plot_X=None, bck=True):

    didd = MultiIDSLoader()
    didd.add_idd(
        shot=shot, run=run,
        user=user, database=database,
        version=version, backend=backend,
    )

    if dsig is dict:
        lids = sorted(dsig.keys())
    else:
        if type(ids) not in [str,list]:
            msg = "If dsig not provided => provide an ids to load Plasma2D!\n"
            msg += "  => Available ids for Plasma2D include:\n"
            msg += "     ['equilibrium',\n"
            msg += "      'core_profiles', 'core_sources'\n,"
            msg += "      'edge_profiles', edge_sources]"
            raise Exception(msg)
        lids = [ids] if type(ids) is str else ids
    lids.append('wall')
    if t0 != False and t0 != None:
        lids.append('pulse_schedule')

    didd.add_ids(ids=lids, get=True)

    return didd.to_Plasma2D(Name=Name, tlim=tlim, dsig=dsig, t0=t0,
                            occ=occ, config=config,
                            description_2d=description_2d, out=out,
                            plot=plot, plot_sig=plot_sig, plot_X=plot_X,
                            bck=bck, dextra=dextra)


def load_Cam(shot=None, run=None, user=None, database=None,
             version=None, backend=None,
             ids=None, indch=None, config=None, description_2d=None,
             occ=None, Name=None, plot=True):

    didd = MultiIDSLoader()
    didd.add_idd(
        shot=shot, run=run,
        user=user, database=database,
        version=version, backend=backend,
    )

    if type(ids) is not str:
        msg = "Please provide ids to load Cam !\n"
        msg += "  => Which diagnostic do you wish to load ?"
        raise Exception(msg)

    lids = ['wall',ids]
    didd.add_ids(ids=lids, get=True)

    return didd.to_Cam(ids=ids, Name=Name, indch=indch,
                       config=config, description_2d=description_2d,
                       occ=occ, plot=plot)


def load_Data(shot=None, run=None, user=None, database=None,
              version=None, backend=None,
              ids=None, datacls=None, geomcls=None, indch_auto=True,
              tlim=None, dsig=None, data=None, X=None, indch=None,
              config=None, description_2d=None,
              occ=None, Name=None, dextra=None,
              t0=None, plot=True, bck=True):

    didd = MultiIDSLoader()
    didd.add_idd(
        shot=shot, run=run,
        user=user, database=database,
        version=version, backend=backend,
    )

    if type(ids) is not str:
        msg = "Please provide ids to load Data !\n"
        msg += "  => Which diagnostic do you wish to load ?"
        raise Exception(msg)

    lids = ['wall',ids]
    if dextra is None and plot:
        lids.append('equilibrium')
    if t0 != False and t0 != None:
        lids.append('pulse_schedule')

    didd.add_ids(ids=lids, get=True)

    return didd.to_Data(ids=ids, Name=Name, tlim=tlim, t0=t0,
                        datacls=datacls, geomcls=geomcls,
                        dsig=dsig, data=data, X=X, indch=indch,
                        config=config, description_2d=description_2d,
                        occ=occ, dextra=dextra,
                        plot=plot, bck=bck, indch_auto=indch_auto)


#############################################################
#############################################################
#           save_to_imas object-specific functions
#############################################################


#--------------------------------
#   Generic functions
#--------------------------------

def _open_create_idd(
    shot=None, run=None,
    refshot=None, refrun=None,
    user=None, database=None,
    version=None, verb=True,
):

    # Check idd inputs and get default values
    didd = dict(
        shot=shot, run=run,
        refshot=refshot, refrun=refrun,
        user=user, database=database,
        version=version,
    )
    for k, v in didd.items():
        if v is None:
            didd[k] = _defimas2tofu._IMAS_DIDD[k]
    didd['shot'] = int(didd['shot'])
    didd['run'] = int(didd['run'])
    assert all(
        [type(didd[ss]) is str for ss in ['user', 'database', 'version']]
    )

    # Check existence of database
    path = os.path.join(
        os.path.expanduser('~{}'.format(didd['user'])),
        'public',
        'imasdb',
        didd['database'],
        '3',
        '0',
    )

    if not os.path.exists(path):
        msg = "IMAS: The required imas ddatabase does not seem to exist:\n"
        msg += "         - looking for: {}\n".format(path)
        if user == getpass.getuser():
            msg += "       => Maybe run imasdb {} (in shell)?".format(database)
        raise Exception(msg)

    # Check existence of file
    filen = 'ids_{0}{1:04d}.datafile'.format(didd['shot'], didd['run'])
    shot_file = os.path.join(path, filen)

    idd = imas.ids(didd['shot'], didd['run'])
    if os.path.isfile(shot_file):
        if verb:
            msg = "IMAS: opening shotfile %s"%shot_file
            print(msg)
        idd.open_env(didd['user'], didd['database'], didd['version'])
    else:
        if user == _defimas2tofu._IMAS_USER_PUBLIC:
            msg = "IMAS: required shotfile does not exist\n"
            msg += "      Shotfiles with user=%s are public\n"%didd['user']
            msg += "      They have to be created centrally\n"
            msg += "       - required shotfile: %s"%shot_file
            raise Exception(msg)
        else:
            if verb:
                msg = "IMAS: creating shotfile %s"%shot_file
                print(msg)
            idd.create_env(didd['user'], didd['database'], didd['version'])

    return idd, shot_file

def _except_ids(ids, nt=None):
    traceback.print_exc(file=sys.stdout)
    if len(ids.time) > 0:
        if nt is None:
            ids.code.output_flag = -1
        else:
            ids.code.output_flag = -np.ones((nt,))
    else:
        ids.code.output_flag.resize(1)
        ids.code.output_flag[0] = -1


def _fill_idsproperties(ids, com, tfversion, nt=None):
    ids.ids_properties.comment = com
    ids.ids_properties.homogeneous_time = 1
    ids.ids_properties.provider = getpass.getuser()
    ids.ids_properties.creation_date = \
                      dtm.datetime.today().strftime('%Y%m%d%H%M%S')

    # Code
    # --------
    ids.code.name = "tofu"
    ids.code.repository = _ROOT
    ids.code.version = tfversion
    if nt is None:
        nt = 1
    ids.code.output_flag = np.zeros((nt,),dtype=int)
    ids.code.parameters = ""

def _put_ids(idd, ids, shotfile, occ=0, cls_name=None,
             err=None, dryrun=False, close=True, verb=True):
    idsn = ids.__class__.__name__
    if not dryrun and err is None:
        try:
            ids.put( occ )
        except Exception as err:
            msg = str(err)
            msg += "\n  There was a pb. when putting the ids:\n"
            msg += "    - shotfile: %s\n"%shotfile
            msg += "    - ids: %s\n"%idsn
            msg += "    - occ: %s\n"%str(occ)
            raise Exception(msg)
        finally:
            # Close idd
            if close:
                idd.close()

    # print info
    if verb:
        if err is not None:
            raise err
        else:
            if cls_name is None:
                cls_name = ''
            if dryrun:
                msg = "  => %s (not put) in %s in %s"%(cls_name,idsn,shotfile)
            else:
                msg = "  => %s put in %s in %s"%(cls_name,idsn,shotfile)
        print(msg)



def _save_to_imas(obj, shot=None, run=None, refshot=None, refrun=None,
                  occ=None, user=None, database=None, version=None,
                  dryrun=False, tfversion=None, verb=True, **kwdargs):

    dfunc = {'Struct': _save_to_imas_Struct,
             'Config': _save_to_imas_Config,
             'CamLOS1D': _save_to_imas_CamLOS1D,
             'DataCam1D': _save_to_imas_DataCam1D}


    # Preliminary check on object class
    cls = obj.__class__
    if cls not in dfunc.keys():
        parents = [cc.__name__ for cc in inspect.getmro(cls)]
        lc = [k for k,v in dfunc.items() if k in parents]
        if len(lc) != 1:
            msg = "save_to_imas() not implemented for class %s !\n"%cls.__name__
            msg += "Only available for classes and subclasses of:\n"
            msg += "    - " + "\n    - ".join(dfunc.keys())
            msg += "\n  => None / too many were found in parent classes:\n"
            msg += "    %s"%str(parents)
            raise Exception(msg)
        cls = lc[0]

    # Try getting imas info from tofu object
    if shot is None:
        try:
            shot = obj.Id.shot
        except Exception:
            msg = "Arg shot must be provided !\n"
            msg += "  (could not be retrieved from self.Id.shot)"
            raise Exception(msg)
    if database is None:
        try:
            database = obj.Id.Exp.lower()
        except Exception:
            msg = "Arg database must be provided !\n"
            msg += "  (could not be retrieved from self.Id.Exp.lower())"
            raise Exception(msg)
    if cls in ['CamLOS1D', 'DataCam1D'] and kwdargs.get('ids',None) is None:
        try:
            kwdargs['ids'] = obj.Id.Diag.lower()
        except Exception:
            msg = "Arg ids must be provided !\n"
            msg += "  (could not be retrieved from self.Id.Diag.lower())"
            raise Exception(msg)

    # Call relevant function
    out = dfunc[cls](
        obj, shot=shot, run=run, refshot=refshot,
        refrun=refrun, occ=occ, user=user, database=database,
        version=version, dryrun=dryrun, tfversion=tfversion,
        verb=verb, **kwdargs,
    )
    return out


#--------------------------------
#   Class-specific functions
#--------------------------------

def _save_to_imas_Struct(obj,
                         shot=None, run=None, refshot=None, refrun=None,
                         occ=None, user=None, database=None, version=None,
                         dryrun=False, tfversion=None, verb=True,
                         description_2d=None, description_typeindex=None,
                         unit=None, mobile=None):

    if occ is None:
        occ = 0
    if description_2d is None:
        description_2d = 0
    if description_typeindex is None:
        description_typeindex = 2
    description_typeindex = int(description_typeindex)
    if unit is None:
        unit = 0
    if mobile is None:
        mobile = False

    # Create or open IDS
    # ------------------
    idd, shotfile = _open_create_idd(
        shot=shot, run=run,
        refshot=refshot, refrun=refrun,
        user=user, database=database, version=version,
        verb=verb,
    )

    # Fill in data
    # ------------------
    try:
        # data
        # --------
        idd.wall.description_2d.resize( description_2d + 1 )
        idd.wall.description_2d[description_2d].type.index = (
            description_typeindex)
        idd.wall.description_2d[description_2d].type.name = (
            '{}_{}'.format(obj.__class__.__name__, obj.Id.Name))
        idd.wall.description_2d[description_2d].type.description = (
            "tofu-generated wall. Each PFC is represented independently as a"
            + " closed polygon in tofu, which saves them as disjoint PFCs")
        if mobile is True:
            idd.wall.description_2d[description_2d].mobile.unit.resize(unit+1)
            node = idd.wall.description_2d[description_2d].mobile.unit[unit]
        else:
            idd.wall.description_2d[description_2d].limiter.unit.resize(unit+1)
            node = idd.wall.description_2d[description_2d].limiter.unit[unit]
        node.outline.r = obj._dgeom['Poly'][0,:]
        node.outline.z = obj._dgeom['Poly'][1,:]
        if obj.noccur > 0:
            node.phi_extensions = np.array([obj.pos, obj.extent]).T
        node.closed = True
        node.name = '%s_%s'%(obj.__class__.__name__, obj.Id.Name)


        # IDS properties
        # --------------
        com = "PFC contour generated:\n"
        com += "    - from %s"%obj.Id.SaveName
        com += "    - by tofu %s"%tfversion
        _fill_idsproperties(idd.wall, com, tfversion)
        err0 = None

    except Exception as err:
        _except_ids(idd.wall, nt=None)
        err0 = err

    finally:

        # Put IDS
        # ------------------
        _put_ids(idd, idd.wall, shotfile, 'wall', occ=occ,
                 cls_name='%s_%s'%(obj.Id.Cls,obj.Id.Name),
                 err=err0, dryrun=dryrun, verb=verb)


def _save_to_imas_Config(obj, idd=None, shotfile=None,
                         shot=None, run=None, refshot=None, refrun=None,
                         occ=None, user=None, database=None, version=None,
                         dryrun=False, tfversion=None, close=True, verb=True,
                         description_2d=None, description_typeindex=None,
                         mobile=None):

    if occ is None:
        occ = 0
    if description_2d is None:
        description_2d = 0
    if mobile is None:
        mobile = False

    # Create or open IDS
    # ------------------
    if idd is None:
        idd, shotfile = _open_create_idd(
            shot=shot, run=run,
            refshot=refshot, refrun=refrun,
            user=user, database=database, version=version,
            verb=verb,
        )
    assert type(shotfile) is str


    # Choose description_2d from config
    lS = obj.lStruct
    lcls = [ss.__class__.__name__ for ss in lS]
    lclsIn = [cc for cc in lcls if cc in ['Ves','PlasmaDomain']]
    nS = len(lS)

    if len(lclsIn) != 1:
        msg = "One (and only one) StructIn subclass is allowed / necessary !"
        raise Exception(msg)

    if description_typeindex is None:
        if nS == 1 and lcls[0] in ['Ves', 'PlasmaDomain']:
            description_typeindex = 0
        else:
            description_typeindex = 1
    assert description_typeindex in [0, 1]

    # Isolate StructIn and take out from lS
    ves = lS.pop(lcls.index(lclsIn[0]))
    nS = len(lS)

    # Fill in data
    # ------------------
    try:
        # data
        # --------
        idd.wall.description_2d.resize( description_2d + 1 )
        wall = idd.wall.description_2d[description_2d]
        wall.type.name = obj.Id.Name
        wall.type.index = description_typeindex
        wall.type.description = (
            "tofu-generated wall. Each PFC is represented independently as a"
            + " closed polygon in tofu, which saves them as disjoint PFCs")

        # Fill limiter / mobile
        if mobile is True:
            # resize nS + 1 for vessel
            wall.mobile.unit.resize(nS + 1)
            units = wall.mobile.unit
            for ii in range(0, nS):
                units[ii].outline.resize(1)
                units[ii].outline[0].r = lS[ii].Poly[0, :]
                units[ii].outline[0].z = lS[ii].Poly[1, :]
                if lS[ii].noccur > 0:
                    units[ii].phi_extensions = np.array([lS[ii].pos,
                                                         lS[ii].extent]).T
                units[ii].closed = True
                name = '{}_{}'.format(lS[ii].__class__.__name__,
                                      lS[ii].Id.Name)
                if lS[ii]._dgeom['move'] is not None:
                    name = name + '_mobile'
                units[ii].name = name

        else:
            # resize nS + 1 for vessel
            wall.limiter.unit.resize(nS + 1)
            units = wall.limiter.unit
            for ii in range(0, nS):
                units[ii].outline.r = lS[ii].Poly[0, :]
                units[ii].outline.z = lS[ii].Poly[1, :]
                if lS[ii].noccur > 0:
                    units[ii].phi_extensions = np.array([lS[ii].pos,
                                                         lS[ii].extent]).T
                units[ii].closed = True
                name = '{}_{}'.format(lS[ii].__class__.__name__,
                                      lS[ii].Id.Name)
                if lS[ii]._dgeom['move'] is not None:
                    name = name + '_mobile'
                units[ii].name = name

        # Add Vessel at the end
        ii = nS
        if mobile:
            units[ii].outline.resize(1)
            units[ii].outline[0].r = ves.Poly[0, :]
            units[ii].outline[0].z = ves.Poly[1, :]
        else:
            units[ii].outline.r = ves.Poly[0, :]
            units[ii].outline.z = ves.Poly[1, :]
        units[ii].closed = True
        units[ii].name = '{}_{}'.format(ves.__class__.__name__,
                                        ves.Id.Name)

        # ----------------------------------
        # Fill vessel if needed
        # vesname = '{}_{}'.format(ves.__class__.__name__, ves.Id.Name)
        # wall.vessel.name = vesname
        # wall.vessel.index = 1
        # wall.vessel.description = (
        #     "tofu-generated vessel outline, with a unique unit / element")

        # wall.vessel.unit.resize(1)
        # wall.vessel.unit[0].element.resize(1)
        # element = wall.vessel.unit[0].element[0]
        # element.name = vesname
        # element.outline.r = ves.Poly[0, :]
        # element.outline.z = ves.Poly[1, :]
        # ----------------------------------

        # IDS properties
        # --------------
        com = "PFC contour generated:\n"
        com += "    - from {}".format(obj.Id.SaveName)
        com += "    - by tofu {}".format(tfversion)
        _fill_idsproperties(idd.wall, com, tfversion)
        err0 = None

    except Exception as err:
        _except_ids(idd.wall, nt=None)
        err0 = err

    finally:

        # Put IDS
        # ------------------
        _put_ids(
            idd, idd.wall, shotfile, occ=occ, err=err0, dryrun=dryrun,
            cls_name='{}_{}'.format(obj.Id.Cls, obj.Id.Name),
            close=close, verb=verb,
        )


def _save_to_imas_CamLOS1D(
    obj, idd=None, shotfile=None,
    shot=None, run=None, refshot=None, refrun=None,
    occ=None, user=None, database=None, version=None,
    dryrun=False, tfversion=None, close=True, verb=True,
    ids=None, deep=None, restore_size=False,
    config_occ=None, config_description_2d=None,
):

    if occ is None:
        occ = 0
    if deep is None:
        deep = False

    # Create or open IDS
    # ------------------
    if idd is None:
        idd, shotfile = _open_create_idd(
            shot=shot, run=run,
            refshot=refshot, refrun=refrun,
            user=user, database=database, version=version,
            verb=verb,
        )
    assert type(shotfile) is str

    # Check choice of ids
    c0 = ids in dir(idd)
    c0 = c0 and hasattr(getattr(idd,ids), 'channel')
    if not c0:
        msg = "Please provide a valid value for arg ids:\n"
        msg += "  => ids should be a valid ids name\n"
        msg += "  => it should refer to an ids with tha attribute channeli\n"
        msg += "    - provided: %s"%ids
        raise Exception(msg)

    # First save dependencies
    if deep:
        _save_to_imas_Config(obj.config, idd=idd, shotfile=shotfile,
                             dryrun=dryrun, verb=verb, close=False,
                             occ=config_occ,
                             description_2d=config_description_2d)

    # Choose description_2d from config
    nch = obj.nRays
    assert nch > 0

    # Get first / second points
    D0 = obj.D
    RZP1 = np.array([np.hypot(D0[0,:],D0[1,:]),
                     D0[2,:],
                     np.arctan2(D0[1,:],D0[0,:])])
    D1 = D0 + obj._dgeom['kOut'][None,:]*obj.u
    RZP2 = np.array([np.hypot(D1[0,:],D1[1,:]),
                     D1[2,:],
                     np.arctan2(D1[1,:],D1[0,:])])

    # Get names
    lk = obj._dchans.keys()
    ln = [k for k in lk if k.lower() == 'name']
    if len(ln) == 1:
        ln = obj.dchans(ln[0])
    else:
        ln = ['ch%s'%str(ii) for ii in range(0,nch)]

    # Get indices
    lk = obj._dchans.keys()
    lind = [k for k in lk if k.lower() in ['ind', 'indch','index','indices']]
    if restore_size and len(lind) == 1:
        lind = obj.dchans[lind[0]]
    else:
        lind = np.arange(0,nch)

    # Check if info on nMax stored
    if restore_size and obj.Id.dUSR is not None:
        nchMax = obj.Id.dUSR.get('imas-nchMax', lind.max()+1)
    else:
        nchMax = lind.max()+1

    # Fill in data
    # ------------------
    try:
        # data
        # --------
        ids = getattr(idd,ids)
        ids.channel.resize( nchMax )
        for ii in range(0,lind.size):
            ids.channel[lind[ii]].line_of_sight.first_point.r = RZP1[0,ii]
            ids.channel[lind[ii]].line_of_sight.first_point.z = RZP1[1,ii]
            ids.channel[lind[ii]].line_of_sight.first_point.phi = RZP1[2,ii]
            ids.channel[lind[ii]].line_of_sight.second_point.r = RZP2[0,ii]
            ids.channel[lind[ii]].line_of_sight.second_point.z = RZP2[1,ii]
            ids.channel[lind[ii]].line_of_sight.second_point.phi = RZP2[2,ii]
            if obj.Etendues is not None:
                ids.channel[lind[ii]].etendue = obj.Etendues[ii]
            if obj.Surfaces is not None:
                ids.channel[lind[ii]].detector.surface = obj.Surfaces[ii]
            ids.channel[lind[ii]].name = ln[ii]


        # IDS properties
        # --------------
        com = "LOS-approximated camera generated:\n"
        com += "    - from %s"%obj.Id.SaveName
        com += "    - by tofu %s"%tfversion
        _fill_idsproperties(ids, com, tfversion)
        err0 = None

    except Exception as err:
        _except_ids(ids, nt=None)
        err0 = err

    finally:
        # Put IDS
        # ------------------
        _put_ids(
            idd, ids, shotfile, occ=occ,
            cls_name='{}_{}'.format(obj.Id.Cls, obj.Id.Name),
            err=err0, dryrun=dryrun, close=close, verb=verb,
        )


def _save_to_imas_DataCam1D(
    obj,
    shot=None, run=None, refshot=None, refrun=None,
    occ=None, user=None, database=None, version=None,
    dryrun=False, tfversion=None, verb=True,
    ids=None, deep=None, restore_size=True, forceupdate=False,
    path_data=None, path_X=None,
    config_occ=None, config_description_2d=None,
):

    if occ is None:
        occ = 0
    if deep is None:
        deep = False

    # Create or open IDS
    # ------------------
    idd, shotfile = _open_create_idd(
        shot=shot, run=run,
        refshot=refshot, refrun=refrun,
        user=user, database=database, version=version,
        verb=verb,
    )

    # Check choice of ids
    c0 = ids in dir(idd)
    c0 = c0 and hasattr(getattr(idd,ids), 'channel')
    if not c0:
        msg = "Please provide a valid value for arg ids:\n"
        msg += "  => ids should be a valid ids name\n"
        msg += "  => it should refer to an ids with tha attribute channel"
        raise Exception(msg)

    # Check path_data and path_X
    if not type(path_data) is str:
        msg = "path_data is not valid !\n"
        msg += "path_data must be a (str) valid path to a field in idd.%s"%ids
        raise Exception(msg)
    if not (path_X is None or type(path_X) is str):
        msg = "path_X is not valid !\n"
        msg += "path_X must be a (str) valid path to a field in idd.%s"%ids
        raise Exception(msg)

    # First save dependencies
    doneresize = False
    if deep:
        if obj.config is not None:
            _save_to_imas_Config(obj.config, idd=idd, shotfile=shotfile,
                                 dryrun=dryrun, verb=verb, close=False,
                                 occ=config_occ,
                                 description_2d=config_description_2d)
        if obj.lCam is not None:
            if not len(obj.lCam) == 1:
                msg = "Geometry can only be saved to imas if unique CamLOS1D !"
                raise Exception(msg)
            _save_to_imas_CamLOS1D(obj.lCam[0], idd=idd, shotfile=shotfile,
                                   ids=ids, restore_size=restore_size,
                                   dryrun=True, verb=verb, close=False,
                                   occ=occ, deep=False)
            doneresize = True

    # Make sure data is up-to-date
    if forceupdate:
        obj._ddata['uptodate'] = False
        obj._set_ddata()

    # Choose description_2d from config
    nch = obj.nch
    assert nch > 0

    # Get names
    lk = obj._dchans.keys()
    ln = [k for k in lk if k.lower() == 'name']
    if len(ln) == 1:
        ln = obj.dchans(ln[0])
    else:
        ln = ['ch%s'%str(ii) for ii in range(0,nch)]

    # Get indices
    lk = obj._dchans.keys()
    lind = [k for k in lk if k.lower() in ['ind','index','indices']]
    if restore_size and len(lind) == 1:
        lind = obj.dchans(lind[0])
    else:
        lind = np.arange(0,nch)

    # Check if info on nMax stored
    if restore_size and obj.Id.dUSR is not None:
        nchMax = obj.Id.dUSR.get('imas-nchMax', lind.max()+1)
    else:
        nchMax = lind.max()+1

    # Fill in data
    # ------------------
    try:
        ids = getattr(idd,ids)

        # time
        ids.time = obj.t

        # data
        # --------
        if not doneresize:
            ids.channel.resize(nchMax)
        data, X = obj.data, obj.X

        lpdata = path_data.split('.')
        if path_X is not None:
            lpX = path_X.split('.')
        if not hasattr(ids.channel[lind[0]], lpdata[0]):
            msg = "Non-valid path_data:\n"
            msg += "    - path_data: %s\n"%path_data
            msg += "    - dir(ids.channel[%s]) = %s"%(str(lind[0]),
                                                      str(dir(ids.channel[lind[0]])))
            raise Exception(msg)
        if path_X is not None and not hasattr(ids.channel[lind[0]], lpX[0]):
            msg = "Non-valid path_X:\n"
            msg += "    - path_X: %s\n"%path_X
            msg += "    - dir(ids.channel[%s]) = %s"%(str(lind[0]),
                                                      str(dir(ids.channel[lind[0]])))
            raise Exception(msg)

        for ii in range(0,lind.size):
            setattr(ftools.reduce(getattr, [ids.channel[lind[ii]]] +
                                  lpdata[:-1]), lpdata[-1], data[:,ii])
            if path_X is not None:
                setattr(ftools.reduce(getattr, [ids.channel[lind[ii]]] +
                                      lpX[:-1]), lpX[-1], X[:,ii])
            ids.channel[ii].name = ln[ii]

        # IDS properties
        # --------------
        com = "LOS-approximated tofu generated signal:\n"
        com += "    - from %s\n"%obj.Id.SaveName
        com += "    - by tofu %s"%tfversion
        _fill_idsproperties(ids, com, tfversion)
        err0 = None

    except Exception as err:
        _except_ids(ids, nt=None)
        err0 = err

    finally:

        # Put IDS
        # ------------------
        _put_ids(idd, ids, shotfile, occ=occ,
                 cls_name='%s_%s'%(obj.Id.Cls,obj.Id.Name),
                 err=err0, dryrun=dryrun, verb=verb)
