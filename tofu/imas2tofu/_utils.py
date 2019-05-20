# -*- coding: utf-8 -*-
#! /usr/bin/python
"""
Thie imas-compatibility module of tofu

Default parameters and input checking

"""

# Built-ins
import itertools as itt

# Standard
import numpy as np

# imas
import imas


_IMAS_USER = 'imas_public'
_IMAS_SHOT = 0
_IMAS_RUN = 0
_IMAS_OCC = 0
_IMAS_TOKAMAK = 'west'
_IMAS_VERSION = '3'
_IMAS_SHOTR = -1
_IMAS_RUNR = -1
_IMAS_DIDD = {'shot':_IMAS_SHOT, 'run':_IMAS_RUN,
              'refshot':_IMAS_SHOTR, 'refrun':_IMAS_RUNR,
              'user':_IMAS_USER, 'tokamak':_IMAS_TOKAMAK, 'version':_IMAS_VERSION}


# Deprecated ?
_LK = list(itt.chain.from_iterable([list(v.keys())
                                    for v in _IMAS_DIDS_old.values()]))


#############################################################
#############################################################
#           IdsMultiLoader
#############################################################


class MultiIDSLoader(object):
    """ Class for handling multi-ids possibly from different idd

    For each desired ids (e.g.: core_profile, ece, equilibrium...), you can
    specify a different idd (i.e.: (shot, run, user, tokamak, version))

    The instance will remember from which idd each ids comes from.
    It provides a structure to keep track of these dependencies
    Also provides methods for opening idd and getting ids and closing idd

    """



    _def = {'isget':False,
            'ids':None, 'occ':0, 'needidd':True}
    _defidd = _IMAS_DIDD

    _lidsnames = [k for k in dir(imas) if k[0] != '_']


    def __init__(self, dids=None,
                 shot=None, run=None, refshot=None, refrun=None,
                 user=None, tokamak=None, version=None, lids=None, get=True):
        super(MultiIDSLoader, self).__init__()
        self.set_dids(dids=dids, shot=shot, run=run, refshot=refshot,
                      refrun=refrun, user=user, tokamak=tokamak,
                      version=version, lids=lids)
        if get:
            self.open_get_close()


    def set_dids(self, dids=None,
                 shot=None, run=None, refshot=None, refrun=None,
                 user=None, tokamak=None, version=None, lids=None):
        dids = self._prepare_dids()
        if dids is None:
            return
        didd, dids, refidd = self._get_diddids(dids)
        self._dids = dids
        self._didd = didd
        self._refidd = refidd

    @classmethod
    def _prepare_dids(cls, dids=None,
                      shot=None, run=None, refshot=None, refrun=None,
                      user=None, tokamak=None, version=None, lids=None):
        if dids is None:
            lc = [shot != None, run != None, refshot != None, refrun != None,
                  user != None, tokamak != None, version != None, lids != None]
            if any(lc):
                dids = cls._get_didsfromkwdargs(shot=shot, run=run,
                                                refshot=refshot,
                                                refrun=refrun,
                                                user=user, tokamak=tokamak,
                                                version=version, lids=lids)
        return dids

    @classmethod
    def _get_didsfromkwdargs(cls, shot=None, run=None, refshot=None, refrun=None,
                             user=None, tokamak=None, version=None, lids=None):
        dids = None
        lc = [shot != None, run != None, refshot != None, refrun != None,
              user != None, tokamak != None, version != None]
        if any(lc):
            lc = [type(lids) is str, type(lids) is list]
            assert any(lc)
            if lc[0]:
                lc = [lids]
            assert all([ids in cls._lidsnames for ids in lids])

            kwdargs = dict(shot=shot, run=run, refshot=refshot, refrun=refrun,
                           user=user, tokamak=tokamak, version=version)
            dids = dict([(ids,{'idd':kwdargs}) for ids in lids])
        return dids

    @classmethod
    def _get_diddids(cls, dids):

        # Check input
        assert type(dids) is dict
        assert all([type(kk) is str for kk in dids.keys()])
        assert all([kk in cls._lidsnames for kk in dids.keys()])
        didd = {}
        for k, v in dids.items():

            # Check value and make dict if necessary
            lc0 = [v is None or v == {}, type(v) is dict, hasattr(v, 'ids_properties')]
            assert any(lc0)
            if lc0[0]:
                dids[k] = {'ids':None}
            elif lc0[-1]:
                dids[k] = {'ids':v}
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
            lc = [v['ids'] is None, hasattr(v['ids'], 'ids_properties'),
                  type(v['ids']) is list]
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
            lc = [idd is None, type(idd) is dict, hasattr(idd, 'close')]
            assert any(lc)

            if lc[0]:
                dids[k]['idd'] = None
                continue

            id_, params = None, {}
            if lc[1]:
                isopen = v.get('isopen', False)
                for kk,vv in _IMAS_DIDD.items():
                    params[kk] = idd.get(kk,vv)
                id_ = imas.ids(params['shot'], params['run'],
                               params['refshot'], params['refrun'])
            elif lc[2]:
                params = {'shot':idd.shot, 'run':idd.run,
                          'refshot':idd.getRefShot(), 'refrun':idd.getRefRun()}
                expIdx = idd.expIdx
                if not (expIdx == -1 or expIdx > 0):
                    msg = "Status of the provided idd could not be determined:\n"
                    msg += "    - idd.expIdx : %s   (should be -1 or >0)\n"%str(expIdx)
                    msg += "    - (shot, run): %s\n"%str((dd['idd'].shot, dd['idd'].run))
                    raise Exception(msg)
                isopen = dd.get('isopen', expIdx > 0)
                if isopen != (expIdx > 0):
                    msg = "Provided isopen does not match observed value:\n"
                    msg += "    - isopen: %s\n"%str(isopen)
                    msg += "    - expIdx: %s"%str(expIdx)
                    raise Exception(msg)
                id_ = idd
            if 'user' in params.keys():
                name = [params['user'], params['tokamak'], params['version']]
            else:
                name = [str(id(id_))]
            name += ['{:06.0f}'.format(params['shot']),
                     '{:05.0f}'.format(params['run'])]
            name = '_'.join(name)
            didd[name] = {'idd':id_, 'params':params, 'isopen':isopen}
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
                msg += "    - ".join([lcO[ii] for ii in range(0,len(lc0))
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
    # methods


    def _checkformat_idd(self, idd=None):
        lk = self._didd.keys()
        lc = [idd is None, type(idd) is str and idd in lk,
              hasattr(idd,'__iter__') and all([ii in lk for ii in idd])]
        assert any(lc)
        if lc[0]:
            lidd = lk
        elif lc[1]:
            lidd = [idd]
        else:
            lidd = idd
        return lidd

    def _checkformat_ids(self, ids=None):
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
        return lids

    def _open(self, idd=None):
        lidd = self._checkformat_idd(idd)
        for k in lidd:
            if self._didd[k]['isopen'] == False:
                args = (self._didd[k]['params']['user'],
                        self._didd[k]['params']['tokamak'],
                        self._didd[k]['params']['version'])
                self._didd[k]['idd'].open_env( *args )
                self._didd[k]['isopen'] = True

    def _get(self, idsname=None, occ=None):
        lids = self._checkformat_ids(idsname)
        for k in lids:
            if occ is None:
                oc = self._dids[k]['occ']
            else:
                oc = np.r_[occ].astype(int)
            if self._dids[k]['ids'] is None:
                idd = self._didd[self._dids[k]['idd']]['idd']
                self._dids[k]['ids'] = [getattr(idd, k) for ii in oc]
                self._dids[k]['needidd'] = False
            for ii in range(0,self._dids[k]['nocc']):
                self._dids[k]['ids'][ii].get( oc[ii]  )
                self._dids[k]['isget'][ii] = True

    def _close(self, idd=None):
        lidd = self._checkformat_idd(idd)
        for k in lidd:
            self._didd[k]['idd'].close()
            self._didd[k]['isopen'] = False


    def _get_liddfromids(self, idsname=None):
        lids = self._checkformat_ids(idsname)
        lidd = sorted(set([self._dids[ids]['idd'] for ids in lids]))
        return lidd

    def get_list_notget_ids(self):
        lids = [k for k,v in self._dids.items() if np.any(v['isget'] == False)]
        return lids

    def open_get_close(self, idsname=None, occ=None):
        lidd = self._get_liddfromids(idsname)
        self._open(idd=lidd)
        self._get(idsname, occ)
        self._close(idd=lidd)


    #---------------------
    # Methods for adding ids
    #---------------------

    def add_ids(self, idsname=None, shot=None, idd=None):
        pass


    def remove_ids(self, idsname=None):
        if idsname is not None:
            assert idsname in self._dids.keys()

    #---------------------
    # Methods for showing data
    #---------------------

    def get_summary(self, max_columns=100, width=1000,
                    verb=True, Return=False):
        """ Summary description of the object content as a pandas DataFrame """
        # # Make sure the data is accessible
        # msg = "The data is not accessible because self.strip(2) was used !"
        # assert self._dstrip['strip']<2, msg

        # -----------------------
        # Build the list
        data = []
        lk = ['user', 'tokamak', 'version',
              'shot', 'run', 'refshot', 'refrun', 'isopen']
        for k0,v0 in self._didd.items():
            lu = [k0] + [v0[k] for k in lk]
            data.append(lu)

        # Build the pandas DataFrame for ddata
        col = ['id', 'user', 'tokamak', 'version',
               'shot', 'run', 'refshot', 'refrun', 'isopen']
        df0 = pd.DataFrame(data, columns=col)

        # -----------------------
        # Build the list
        data = []
        for k0,v0 in self._dindref.items():
            lu = [k0, v0['group'], v0['size']]
            data.append(lu)

        # Build the pandas DataFrame for ddata
        col = ['id', 'group', 'size']
        df1 = pd.DataFrame(data, columns=col)

        # -----------------------
        # Build the list
        data = []
        for k0,v0 in self._ddata.items():
            if type(v0['data']) is np.ndarray:
                shape = v0['data'].shape
            else:
                shape = v0['data'].__class__.__name__
            lu = [k0, v0['quant'], v0['name'], v0['units'], shape,
                  v0['indref'], v0['lgroup']]
            data.append(lu)

        # Build the pandas DataFrame for ddata
        col = ['id', 'quant', 'name', 'units', 'shape', 'indref', 'lgroup']
        df2 = pd.DataFrame(data, columns=col)
        pd.set_option('display.max_columns',max_columns)
        pd.set_option('display.width',width)

        if verb:
            sep = "\n------------\n"
            print("didd", sep, df0, "\n")
            print("dindref", sep, df1, "\n")
            print("ddata", sep, df2, "\n")
        if Return:
            return df0, df1, df2
