# -*- coding: utf-8 -*-
#! /usr/bin/python
"""
Thie imas-compatibility module of tofu

Default parameters and input checking

"""

# Built-ins
import itertools as itt
import copy

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
# _LK = list(itt.chain.from_iterable([list(v.keys())
                                    # for v in _IMAS_DIDS_old.values()]))


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
    _didsk = {'tokamak':15, 'user':15, 'version':7,
              'shot':6, 'run':3, 'refshot':6, 'refrun':3}

    # Known short version of signal str
    _dshort = {'core_profiles':
               {'1dTe':{'str':'profiles_1d[time].electrons.temperature'},
                '1dne':{'str':'profiles_1d[time].electrons.density'},
                '1dzeff':{'str':'profiles_1d[time].zeff'},
                '1dphi':{'str':'profiles_1d[time].grid.phi'},
                '1dpsi':{'str':'profiles_1d[time].grid.psi'},
                '1drhotn':{'str':'profiles_1d[time].grid.rho_tor_norm'},
                '1drhopn':{'str':'profiles_1d[time].grid.rho_pol_norm'}},
               'equilibrium':
               {},
               'core_sources':
               {'brem':{'str':"sources['bremsstrahlung'].profiles_1d[time].electrons.energy"},
                'line':{'str':"sources['lineradiation'].profiles_1d[time].electrons.energy"}},
               'wall':
               {},
               'ece':
               {'Te': {'str':'channel[chan].t_e.data'},
                'R': {'str':'channel[chan].position.r.data'},
                'rhotn':{'str':'channel[chan].position.rho_tor_norm.data'},
                'tau':{'str':'channel[chan].tau1kev'}}
              }



    def __init__(self, dids=None,
                 shot=None, run=None, refshot=None, refrun=None,
                 user=None, tokamak=None, version=None, lids=None, get=True):
        super(MultiIDSLoader, self).__init__()
        self.set_dids(dids=dids, shot=shot, run=run, refshot=refshot,
                      refrun=refrun, user=user, tokamak=tokamak,
                      version=version, lids=lids)
        self._set_fsig()
        if get:
            self.open_get_close()


    def set_dids(self, dids=None,
                 shot=None, run=None, refshot=None, refrun=None,
                 user=None, tokamak=None, version=None, lids=None):
        dids = self._prepare_dids(dids=dids, shot=shot, run=run,
                                  refshot=refshot, refrun=refrun,
                                  user=user, tokamak=tokamak, version=version)
        if dids is None:
            didd, dids, refidd = None, None, None
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
        # lc = [shot != None, run != None, refshot != None, refrun != None,
              # user != None, tokamak != None, version != None]
        if lids is not None:
            lc = [type(lids) is str, type(lids) is list]
            assert any(lc)
            if lc[0]:
                lids = [lids]
            if not all([ids in cls._lidsnames for ids in lids]):
                msg = "All provided ids names must be valid ids identifiers:\n"
                msg += "    - Provided: %s\n"%str(lids)
                msg += "    - Expected: %s"%str(cls._lidsnames)
                raise Exception(msg)

            kwdargs = dict(shot=shot, run=run, refshot=refshot, refrun=refrun,
                           user=user, tokamak=tokamak, version=version)
            dids = dict([(ids,{'idd':kwdargs}) for ids in lids])
        return dids

    @classmethod
    def _get_diddids(cls, dids, defidd=None):

        # Check input
        assert type(dids) is dict
        assert all([type(kk) is str for kk in dids.keys()])
        assert all([kk in cls._lidsnames for kk in dids.keys()])
        if defidd is None:
            defidd = cls._defidd
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
                idd = dict([(kk,vv) for kk,vv in idd.items() if vv is not None])
                isopen = v.get('isopen', False)
                for kk,vv in defidd.items():
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
    #############
    # methods
    #############


    #############
    # shortcuts

    @staticmethod
    def _getcharray(ar, col, sep='  ', line='-', just='l'):
        assert len(col) == ar.shape[1]
        nn = np.char.str_len(ar).max(axis=0)
        nn = np.fmax(nn, [len(cc) for cc in col])
        line = [line*n for n in nn]
        if just == 'l':
            col = [col[ii].ljust(nn[ii]) for ii in range(0,len(nn))]
            block = '\n'.join([sep.join(v) for v in np.char.ljust(ar,nn)])
        else:
            col = [col[ii].rjust(nn[ii]) for ii in range(0,len(nn))]
            block = '\n'.join([sep.join(v) for v in np.char.rjust(ar,nn)])
        col = sep.join(col)
        line = sep.join(line)
        return '\n'.join([col,line,block])

    @classmethod
    def _shortcuts(cls, obj=None, ids=None, return_=False, verb=True, sep='  ', line='-', just='l'):
        if obj is None:
            obj = cls
        if ids is None:
            lids = list(obj._dids.keys())
        elif ids == 'all':
            lids = list(obj._dshort.keys())
        else:
            lc = [type(ids) is str, type(ids) is list and all([type(ss) is str
                                                               for ss in ids])]
            assert any(lc), "ids must be an ids name or a list of such !"
            if lc[0]:
                lids = [ids]
            else:
                lids = lids

        lids = sorted(set(lids).intersection(obj._dshort.keys()))

        short = [[(ids, kk, vv['str']) for kk,vv in obj._dshort[ids].items()]
                 for ids in lids]
        short = np.array(list(itt.chain.from_iterable(short)), dtype='U')
        if verb:
            col = ['ids', 'shortcut', 'long version']
            msg = obj._getcharray(short, col, sep=sep, line=line, just=just)
            print(msg)
        if return_:
            return short

    def get_shortcuts(self, ids=None, return_=False, verb=True, sep='  ', line='-', just='l'):
        """ Display and/or return the builtin shortcuts for imas signal names

        By default (ids=None), only display shortcuts for stored ids
        To display all possible shortcuts, use ids='all'
        To display shortcuts for a specific ids, use ids=<idsname>

        These shortcuts can be customized (with self.set_shortcuts())
        They are useful for use with self.get_data()

        """
        return self._shortcuts(obj=self, ids=ids, return_=return_, verb=verb,
                               sep=sep, line=line, just=just)

    def set_shortcuts(self, dshort=None):
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


    #############
    # data access

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
        lidd = sorted(set([self._dids[ids]['idd'] for ids in lids]))
        llids = [(idd, [ids for ids in lids if self._dids[ids]['idd'] == idd])
                for idd in lidd]
        return llids

    def _open(self, idd=None):
        lidd = self._checkformat_idd(idd)
        for k in lidd:
            if self._didd[k]['isopen'] == False:
                args = (self._didd[k]['params']['user'],
                        self._didd[k]['params']['tokamak'],
                        self._didd[k]['params']['version'])
                self._didd[k]['idd'].open_env( *args )
                self._didd[k]['isopen'] = True

    def _get(self, idsname=None, occ=None, llids=None, verb=True):
        if llids is None:
            llids = self._checkformat_ids(idsname)
        if verb and len(llids)>0:
            msgroot = "Getting    ..."
            ls = [len(ids) + len(str(self._dids[ids]['occ']))
                  for ids in self._dids.keys()]
            rjust = len(msgroot) + max(ls)
            msg = ''.rjust(rjust)
            msg += '  '.join([kk.rjust(vv) for kk,vv in self._didsk.items()])
            print(msg)
        for ii in range(0,len(llids)):
            for jj in range(0,len(llids[ii][1])):
                k = llids[ii][1][jj]
                occref = self._dids[k]['occ']
                if occ is None:
                    oc = occref
                else:
                    oc = np.unique(np.r_[occ].astype(int))
                    oc = np.intersect1(oc, occref)
                indoc = np.array([np.nonzero(occref==oc[ll])[0][0]
                                  for ll in range(0,len(oc))]).ravel()

                # if ids not provided
                if self._dids[k]['ids'] is None:
                    idd = self._didd[self._dids[k]['idd']]['idd']
                    self._dids[k]['ids'] = [getattr(idd, k) for ii in oc]
                    self._dids[k]['needidd'] = False

                if verb:
                    msg = ("Getting %s %s..."%(k,str(oc))).rjust(rjust)
                    if jj == 0:
                        params = self._didd[llids[ii][0]]['params']
                        msg += '  '.join([str(params[kk]).rjust(vv)
                                            for kk,vv in self._didsk.items()])
                    else:
                        msg += '  '.join(['"'.rjust(vv)
                                          for vv in self._didsk.values()])
                    print(msg)

                for ll in range(0,len(oc)):
                    if self._dids[k]['isget'][indoc[ll]] == False:
                        self._dids[k]['ids'][indoc[ll]].get( oc[ll] )
                        self._dids[k]['isget'][indoc[ll]] = True

    def _close(self, idd=None):
        lidd = self._checkformat_idd(idd)
        for k in lidd:
            self._didd[k]['idd'].close()
            self._didd[k]['isopen'] = False

    def get_list_notget_ids(self):
        lids = [k for k,v in self._dids.items() if np.any(v['isget'] == False)]
        return lids

    def open_get_close(self, idsname=None, occ=None, verb=True):
        llids = self._checkformat_ids(idsname)
        lidd = [lids[0] for lids in llids]
        self._open(idd=lidd)
        self._get(occ=occ, llids=llids)
        self._close(idd=lidd)


    #---------------------
    # Methods for adding ids
    #---------------------

    def add_ids(self, idsname=None, occ=None,
                shot=None, run=None, refshot=None, refrun=None,
                user=None, tokamak=None, version=None,
                dids=None, get=False):
        lc = [dids is not None, idsname is not None]
        if not np.sum(lc) == 1:
            msg = "Provide either a dids (dict) or an idsname !"
            raise Exception(msg)
        if lc[1]:
            dids = self._get_didsfromkwdargs(shot=shot, run=run,
                                             refshot=refshot, refrun=refrun,
                                             user=user, tokamak=tokamak,
                                             version=version, lids=idsname)
        defidd = self._didd[self._refidd]['params']
        didd, dids, refidd = self._get_diddids(dids, defidd=defidd)
        self._dids.update(dids)
        self._didd.update(didd)
        if get:
            self.open_get_close(idsname=idsname)


    def remove_ids(self, idsname=None, occ=None):
        if idsname is not None:
            assert idsname in self._dids.keys()
            occref = self._dids[idsname]['occ']
            if occ is None:
                occ = occref
            else:
                occ = np.unique(np.r_[occ].astype(int))
                occ = np.intersect1d(occ, occref, assume_unique=True)
            idd = self._dids[idsname]['idd']
            lids = [k for k,v in self._dids.items() if v['idd']==idd]
            if lids == [idsname]:
                del self._didd[idd]
            if np.all(occ == occref):
                del self._dids[idsname]
            else:
                isgetref = self._dids[idsname]['isget']
                indok = np.array([ii for ii in range(0,occref.size)
                                  if occref[ii] not in occ])
                self._dids[idsname]['ids'] = [self._dids[idsname]['ids'][ii]
                                              for ii in indok]
                self._dids[idsname]['occ'] = occref[indok]
                self._dids[idsname]['isget'] = isgetref[indok]
                self._dids[idsname]['nocc'] = self._dids[idsname]['occ'].size

    #---------------------
    # Methods for showing content
    #---------------------

    def get_summary(self, sep='  ', line='-', just='l',
                    verb=True, return_=False):
        """ Summary description of the object content as a np.array of str """

        # -----------------------
        # idd
        a0 = []
        c0 = ['idd', 'user', 'tokamak', 'version',
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
        c1 = ['ids', 'idd', 'occ', 'isget']
        a1 = [[k0, v0['idd'], str(v0['occ']), str(v0['isget'])]
              for k0,v0 in self._dids.items()]
        a1 = np.array(a1, dtype='U')

        if verb or return_ in [True,'msg']:
            msg0 = self._getcharray(a0, c0,
                                    sep=sep, line=line, just=just)
            msg1 = self._getcharray(a1, c1,
                                    sep=sep, line=line, just=just)
            if verb:
                msg = '\n\n'.join([msg0,msg1])
                print(msg)
        if return_ != False:
            if return_ == True:
                out = (a0, a1, msg0, msg1)
            elif return_ == 'array':
                out = (a0, a1)
            elif return_ == 'msg':
                out = (msg0, msg1)
            else:
                lok = [False, True, 'msg', 'array']
                raise Exception("Valid return_ values are: %s"%str(lok))
            return out


    #---------------------
    # Methods for returning data
    #---------------------

    def _checkformat_getdata_ids(self, ids):
        msg = "Arg ids must be either:\n"
        msg += "    - None: if self.dids only has one key\n"
        msg += "    - str: a valid key of self.dids"

        lc = [ids is None, type(ids) is str]
        if not any(lc):
            raise Exception(msg)

        if lc[0]:
            if len(self._dids.keys()) != 1:
                raise Exception(msg)
            ids = list(self._dids.keys())[0]
        elif lc[1]:
            if ids not in self._dids.keys():
                raise Exception(msg)
        return ids

    def _checkformat_getdata_sig(self, sig, ids):
        msg = "Arg sig must be a str or a list of str !\n"
        msg += "  More specifically, a list of valid ids nodes paths"
        lc = [type(sig) is str, type(sig) is list]
        if not any(lc):
            raise Exception(msg)
        if lc[0]:
            sig = [sig]
        elif lc[1]:
            if any([type(ss) is not str for ss in sig]):
                raise Exception(msg)

        # Check each sig is either a key / value[str] to self._dshort
        lk = list(self._dshort[ids].keys())
        for ii in range(0,len(sig)):
            c0 = sig[ii] in lk
            lc1 = [sig[ii] == self._dshort[ids][kk]['str'] for kk in lk]
            if not c0 or any(lc1):
                msg = "Each provided sig must be either:\n"
                msg += "    - a valid shortcut (cf. self.shortcuts()\n"
                msg += "    - a valid long version (cf. self.shortcuts)\n"
                msg += "\n  Provided sig: %s"%str(sig)
                raise Exception(msg)
            if not c0:
                sig[ii] = lk[lc1.index(True)]
        return sig

    def _checkformat_getdata_occ(self, occ, ids):
        msg = "Arg occ must be a either:\n"
        msg += "    - None: all occurences are used\n"
        msg += "    - int: occurence to use (in self.dids[ids]['occ'])\n"
        msg += "    - array of int: occurences to use (in self.dids[ids]['occ'])"
        lc = [occ is None, type(occ) is int, hasattr(occ,'__iter__')]
        if not any(lc):
            raise Exception(msg)
        if lc[0]:
            occ = self._dids[ids]['occ']
        else:
            occ = np.r_[occ].astype(int).ravel()
            if any([oc not in self._dids[ids]['occ'] for oc in occ]):
                raise Exception(msg)
        return occ

    def _checkformat_getdata_indch(self, indch, nch):
        msg = "Arg indch must be a either:\n"
        msg += "    - None: all channels used\n"
        msg += "    - int: channel to use (index)\n"
        msg += "    - array of int: channels to use (indices)\n"
        msg += "    - array of bool: channels to use (indices)"
        lc = [type(indch) is None, type(indch) is int, hasattr(indch,'__iter__')]
        if not any(lc):
            raise Exception(msg)
        if lc[0]:
            indch = np.arange(0,nch)
        else:
            indch = np.r_[indch].rave()
            lc = [indch.dtype == np.int, indch.dtype == np.bool]
            if not any(lc):
                raise Exception(msg)
            if lc[1]:
                indch = np.nonzero(indch)[0]
            assert np.all((indch>=0) & (indch<nch))
        return indch

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

    @staticmethod
    def _get_fsig(sig):
        ls0 = sig.split('.')
        lct = ['[time]' in ss for ss in ls0]
        lch = ['channel[chan]' in ss for ss in ls0]
        lc = [any(lct), any(lch)]

        if not any(lc):
            def fsig(obj, ls0):
                sig = getattr(obj, ls0[0])
                for ss in ls0[1:]:
                    sig = getattr(sig, ss)
                return sig

        elif all(lc):
            msg = "Not implemented yet for sig with both [time] and [chan]"
            raise Exception(msg)

        elif lc[0]:
            sig = sig.replace('[time]','[ii]')
            it = lct.index(True)
            ls1, st, ls2 = ls0[:it], ls0[it].replace('[time]',''), ls0[it+1:]
            ls1, ls2 = '.''.'.join(ls2)
            if len(ls2) > 0:
                ls2 = '.'+ls2
            def fsig(obj, indt=None, ls1=ls1, st=st, ls2=ls2):
                sig0 = obj
                for ss in ls1:
                    sig0 = getattr(sig0, ss)
                sig0 = getattr(sig0,st)
                nt = len(sig0)
                if indt is None:
                    indt = range(0,nt)
                out = [eval( 'sig0[ii]'+ls2 ) for ii in indt]
                if all([oo.shape == out[0].shape for oo in out[1:]]):
                    out = np.vstack(out)
                return out

        elif lc[1]:
            fsig = None

        return fsig

    def _set_fsig(self):
        for ids in self._dshort.keys():
            for k,v in self._dshort[ids].items():
                self._dshort[ids][k]['fsig'] = self._get_fsig(v['str'])

    def _get_data(self, ids, sig, occ):

        # get list of results for occ
        occref = self._dids[ids]['occ']
        indoc = np.array([np.nonzero(occref==oc)[0][0] for oc in occ])
        out = [self._dshort[ids][sig]['fsig']( self._dids[ids]['ids'][ii] )
               for ii in indoc]
        return out

    def get_data(self, ids=None, sig=None, occ=None, indch=None, indt=None):
        """ Return a dict of the desired signals extracted from specified ids

        If the ids has a field 'channel', indch is used to specify from which
        channel data shall be loaded (all by default)

        """

        # ------------------
        # Check format input

        # ids = valid self.dids.keys()
        ids = self._checkformat_getdata_ids(ids)

        # sig = list of str
        sig = self._checkformat_getdata_sig(sig, ids)

        # occ = np.ndarray of valid int
        occ = self._checkformat_getdata_occ(occ, ids)

        # Check all occ have isget = True
        indok = self._dids[ids]['isget'][occ]
        if not np.all(indok):
            msg = "All desired occurences shall have been gotten !\n"
            msg += "    - occ:   %s\n"%str(occ)
            msg += "    - isget: %s"%str(self._dids[ids]['isget'])
            raise Exception(msg)

        # check indch if ids has channels
        if hasattr(self._dids[ids]['ids'][occ[0]], 'channel'):
            nch = len(getattr(self._dids[ids]['ids'][occ[0]], 'channel'))
            indch = self._checkformat_getdata_indch(indch, nch)

        # ------------------
        # get data

        dout = dict.fromkeys(sig)
        for ss in sig:
            dout[ss] = self._get_data(ids, ss, occ)
        return out
