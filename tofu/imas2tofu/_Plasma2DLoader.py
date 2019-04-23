

# Built-in
import os
import warnings
import itertools as itt
import operator

# Common
import numpy as np
import scipy.interpolate as scpinterp
import matplotlib as mpl

# tofu-specific
try:
    import tofu.imas2tofu._utils as _utils
except Exception:
    from . import _utils

# imas-specific
import imas



__all__ = ['Plasma2DLoader']


class Plasma2DLoader(object):
    """ A generic class for handling 2D (and 1D) plasma profiles

    Provides:
        - equilibrium-related quantities
        - any 1d profile (can be remapped on 2D equilibrium)
        - spatial interpolation methods

    """

    #----------------
    # Class attributes
    #----------------

    # Quantities can be either stored in the ids ('ids')
    # Or have to be computed ('comp')

    _dquantall ={'eq':{'2d':['psi','phi',
                             'b_field_r', 'b_field_z', 'b_field_tor',
                             'b_field_norm'],
                       '1d':['phi','psi','rho_tor_norm','rho_pol_norm']},
                 'cprof':{'1d':['grid.phi','grid.psi',
                                'grid.rho_tor_norm','grid.rho_pol_norm', 'zeff',
                                'electrons.density','electrons.temperature']},
                 'csource':{'1d':['bremsstrahlung/electrons.energy',
                                  'lineradiation/electrons.energy', 'prad']}}

    _dcomp = {'b_field_norm':['b_field_r', 'b_field_z', 'b_field_tor'],
              'rho_pol_norm':['psi'],
              'prad':['bremsstrahlung/electrons.energy','lineradiation/electrons.energy']}

    _dquantmap1d = ['psi','phi','rho_tor_norm','rho_pol_norm']

    # Pre-set lists of quantities to be loaded from the ids, useful in typical use cases
    _dpreset = {'pos':{'eq':{'2d':['psi','phi'],
                             '1d':['phi','psi','rho_tor_norm','rho_pol_norm']}},
                'base':{'eq':{'2d':['psi','phi'],
                              '1d':['phi','psi','rho_tor_norm','rho_pol_norm']},
                        'cprof':{'1d':['grid.phi','grid.psi',
                                       'grid.rho_tor_norm','grid.rho_pol_norm', 'zeff',
                                       'electrons.density','electrons.temperature']}}}
                        #'csource':{'1d':['prad']}}}

    _lpriorityref = ['eq','cprof','csource']


    #----------------
    # Class creation and instanciation
    #----------------

    def __init__(self, dquant='base', tlim=None,
                 dids_cprof=None, dids_csource=None, dids_eq=None,
                 lpriorityref=None, verb=True):

        # Preformat
        if lpriorityref is not None:
            assert type(lpriorityref) is list and all([type(ss) is str
                                                       for ss in lpriorityref])
            self._lpriorityref = lpriorityref

        # Check inputs
        out = self._checkformat_dins(dquant=dquant,
                                     dids_eq=dids_eq,
                                     dids_cprof=dids_cprof,
                                     dids_csource=dids_csource,
                                     tlim=tlim)
        dquant, dids, tlim = out
        self._dquant = dict([(k0,{}) for k0 in dquant.keys()])
        for k0 in dquant.keys():
            for k1 in dquant[k0].keys():
                self._dquant[k0][k1] = {'dq':dict([(qq,
                                                    {'val':None,'origin':None})
                                                   for qq in dquant[k0][k1]]),
                                        'npts':None}
        self._dtime = dict([(k0,{}) for k0 in self._dquant.keys()])


        # Get quantities
        idsref = None
        lk0 = [k0 for k0 in self._lpriorityref if k0 in self._dquant.keys()]
        for k0 in lk0:
            # Fall back to idsref if None
            if idsref is None:
                dids[k0]['ids'] = self._openids(dids[k0]['dict'])
                idsref = dids[k0]['ids']
            elif dids[k0]['dict'] is None:
                dids[k0]['ids'] = idsref
            else:
                dids[k0]['ids'] = self._openids(dids[k0]['dict'])
            if verb:
                print("Getting %s..."%k0)
            idsnode = self._get_idsnode(dids['eq']['ids'], k0)

            # Check availability and get quantities
            self._checkformat_quantav(idsnode, k0, tlim)

            # Get computables
            lcomp = self._dcomp.keys()
            for k1 in self._dquant[k0].keys():
                nt = self._dtime[k0]['nt']
                npts = self._dquant[k0][k1]['npts']
                lq = set(self._dquant[k0][k1]['dq'].keys()).intersection(lcomp)
                for qq in lq:
                    lqcomp = [self._dquant[k0][k1]['dq'][qcomp]['val']
                              for qcomp in self._dcomp[qq]]
                    val = self._comp(qq, lqcomp, nt, npts)
                    self._dquant[k0][k1]['dq'][qq]['val'] = val

            if k0 == 'eq':
                # get dmesh if 2d != None
                self._dmesh = self._get_dmesh(idsnode, self._dquant[k0]['2d']['npts'])


        # Close ids
        for k0 in lk0:
            dids[k0]['ids'].close()




    #---------------------
    # Methods for checking and formatting inputs
    #---------------------

    @classmethod
    def _checkformat_dins(cls, dquant=None,
                          dids_cprof=None, dids_csource=None,
                          dids_eq=None, tlim=None):

        # Check dquant
        lk0, lk1 = cls._lpriorityref, ['1d','2d']
        c0 = type(dquant) is str and dquant in cls._dpreset.keys()
        if c0:
            dquant = cls._dpreset[dquant]

        c1 = type(dquant) is dict and all([k0 in lk0 for k0 in dquant.keys()])
        c1 = c1 and all([type(v0) is dict and all([k1 in lk1 for k1 in v0.keys()])
                         for v0 in dquant.values()])
        c1 = c1 and all([all([all([v2 in cls._dquantall[k0][k1] for v2 in v1])
                              for k1,v1 in v0.items()])
                         for k0,v0 in dquant.items()])
        if not c1:
            msg = "Arg dquant must be either:\n"
            msg += "    - str : a valid key to self._dpreset\n"
            msg += "    - dict: with structure:\n"
            msg += "        {'eq':     {'1d':lq0, '2d':lq1},\n"
            msg += "         'cprof':  {'1d':lq2},\n"
            msg += "         'csource':{'1d':lq3},\n"
            msg += "        where (lq0, lq1, lq2,lq3) are lists of str\n"
            msg += "            (valid ids fields, optionally with path)"
            raise Exception(msg)

        for k0 in dquant.keys():
            for k1 in dquant[k0].keys():
                if k1 in dquant[k0].keys() and len(dquant[k0][k1]) == 0:
                    del dquant[k0][k1]
            if all([k1 not in dquant[k0].keys() for k1 in dquant[k0].keys()]):
                del dquant[k0]

        # Add quantities necessary for computing
        for k0 in dquant.keys():
            for k1 in dquant[k0].keys():
                for qq in dquant[k0][k1]:
                    if qq in cls._dcomp.keys():
                        dquant[k0][k1] = sorted(set(dquant[k0][k1]).union(cls._dcomp[qq]))


        # Check dids
        dids = {'eq':     {'dict':dids_eq, 'ids':None},
                'cprof':  {'dict':dids_cprof, 'ids':None},
                'csource':{'dict':dids_csource, 'ids':None}}
        lc = [vv['dict'] is None for vv in dids.values()]
        if all(lc):
            ls = str(list(lk0))
            msg = "Please provide at least one of ids of %s!"%ls
            raise Exception(msg)

        idsref = [k for k in lk0 if dids[k]['dict'] is not None][0]
        dids[idsref]['dict'] = _utils._get_defaults( dids=dids[idsref]['dict'] )
        for k in lk0:
            if dids[k]['dict'] is not None:
                dids[k]['dict'] = _utils._get_defaults( dids=dids[k]['dict'] )

        # Check tlim
        if tlim is not None:
            try:
                tlim = np.asarray(tlim).ravel().astype(float)
                assert tlim.size == 2 and np.diff(tlim) > 0.
            except Exception as err:
                msg = str(err) + "\n\n"
                msg += "tlim must be an increasing sequence of 2 time values!\n"
                msg += "    - Expected : tlim = [t0,t1] with t1 > t0\n"
                msg += "    - Received : %s"%str(tlim)
                raise Exception(msg)

        return dquant, dids, tlim

    @staticmethod
    def _get_idsnode(ids, idskey='eq'):
        if idskey == 'eq':
            idsnode = ids.equilibrium
        elif idskey == 'cprof':
            idsnode = ids.core_profiles
        else:
            idsnode  = ids.core_sources
        idsnode.get()
        return idsnode


    @staticmethod
    def _checkformat_tlim(tlim, idseq):
        # Extract time indices and vector
        t = np.asarray(idseq.time).ravel()
        indt = np.ones((t.size,), dtype=bool)
        if tlim is not None:
            indt[(t<tlim[0]) | (t>tlim[1])] = False
        t = t[indt]
        indt = np.nonzero(indt)[0]
        nt = t.size
        return {'tlim':tlim, 'nt':nt, 't':t, 'indt':indt}

    @staticmethod
    def _openids(dids):
        try:
            ids = imas.ids(s=dids['shot'], r=dids['run'])
        except Exception:
            msg = "Error running:\n"
            msg += "    ids = imas.ids(s=%s, r=%s)"%(str(dids['shot']),
                                                     str(dids['run']))
            raise Exception(msg)
        try:
            ids.open_env(dids['user'], dids['tokamak'], dids['version'])
        except Exception:
            ids.close()
            msg = "Error running:\n"
            msg += "    ids.open_env(%s, %s, %s)"%(dids['user'],
                                                   dids['tokamak'],
                                                   dids['version'])
            raise Exception(msg)
        return ids


    def _checkformat_quantav(self, idsnode,
                              k0='eq', tlim=None):
        assert k0 in self._lpriorityref

        for k1 in self._dquant[k0].keys():
            lq = self._dquant[k0][k1]['dq'].keys()

            if k0 == 'eq':
                self._dtime[k0] = self._checkformat_tlim(tlim, idsnode)
                indt = self._dtime[k0]['indt']
                nt = self._dtime[k0]['nt']
                valid = np.ones((nt,),dtype=bool)
                if k1 == '1d':
                    for qq in lq:
                        obj = idsnode.time_slice[0].profiles_1d
                        if not hasattr(obj,qq):
                            continue
                        npts = len(getattr(obj,qq))
                        if npts == 0:
                            continue
                        val = np.full((nt, npts), np.nan)
                        for ii in range(0,nt):
                            if idsnode.code.output_flag[indt[ii]] <= -1:
                               valid[ii] = False
                               continue
                            obj = idsnode.time_slice[indt[ii]].profiles_1d
                            val[ii,:] = getattr(obj,qq)
                        self._dquant[k0][k1]['dq'][qq]['val'] = val
                        self._dquant[k0][k1]['dq'][qq]['origin'] = 'ids'
                        self._dquant[k0][k1]['npts'] = npts
                        self._dtime[k0]['valid'] = valid
                else:
                    for qq in lq:
                        obj = idsnode.time_slice[0].ggd[0]
                        if not hasattr(obj,qq):
                            continue
                        npts = len(getattr(obj,qq)[0].values)
                        if npts == 0:
                            continue
                        val = np.full((nt, npts), np.nan)
                        for ii in range(0,nt):
                            obj = idsnode.time_slice[indt[ii]].ggd[0]
                            val[ii,:] = getattr(obj,qq)[0].values
                        self._dquant[k0][k1]['dq'][qq]['val'] = val
                        self._dquant[k0][k1]['dq'][qq]['origin'] = 'ids'
                        self._dquant[k0][k1]['npts'] = npts

            elif k0 == 'cprof':
                self._dtime[k0] = self._checkformat_tlim(tlim, idsnode)
                indt = self._dtime[k0]['indt']
                nt = self._dtime[k0]['nt']
                assert k1 == '1d'
                for qq in lq:
                    obj = idsnode.profiles_1d[0]
                    if '.' in qq:
                        q0, q1 = qq.split('.')
                        if not hasattr(obj,q0):
                            continue
                        obj = getattr(obj,q0)
                    else:
                        q0 = None
                        q1 = qq
                    if not hasattr(obj,qq):
                        continue
                    npts = len(getattr(obj,qq))
                    if npts == 0:
                        continue
                    val = np.full((nt, npts), np.nan)
                    for ii in range(0,nt):
                        obj = idsnode.profiles_1d[indt[ii]]
                        if q0 is None:
                            val[ii,:] = getattr(obj,q1)
                        else:
                            val[ii,:] = getattr(getattr(obj,q0),q1)
                    self._dquant[k0][k1]['dq'][qq]['val'] = val
                    self._dquant[k0][k1]['dq'][qq]['origin'] = 'ids'
                    self._dquant[k0][k1]['npts'] = npts


            else:
                self._dtime[k0] = self._checkformat_tlim(tlim, idsnode)
                indt = self._dtime[k0]['indt']
                nt = self._dtime[k0]['nt']
                assert k1 == '1d'
                for qq in lq:
                    if not '/' in qq:
                        continue
                    q0, qq = qq.split('/')
                    obj = idsnode.sources[q0].profiles_1d[0]
                    if '.' in qq:
                        q1, q2 = qq.split('.')
                        if not hasattr(obj,q1):
                            continue
                        obj = getattr(obj,q1)
                    else:
                        q1 = None
                        q2 = qq
                    if not hasattr(obj,q2):
                        continue
                    npts = len(getattr(obj,q2))
                    if npts == 0:
                        continue
                    val = np.full((nt, npts), np.nan)
                    for ii in range(0,nt):
                        obj = idsnode.sources[q0].profiles_1d[indt[ii]]
                        if q1 is None:
                            val[ii,:] = getattr(obj,q2)
                        else:
                            val[ii,:] = getattr(getattr(obj,q1),q2)
                    self._dquant[k0][k1]['dq'][qq]['val'] = val
                    self._dquant[k0][k1]['dq'][qq]['origin'] = 'ids'
                    self._dquant[k0][k1]['npts'] = npts


    @classmethod
    def _get_dmesh(cls, idseq, npts):

        # Check the grid exists
        c0 = len(idseq.grids_ggd) == 1
        c0 &= len(idseq.grids_ggd[0].grid) == 1
        c0 &= len(idseq.grids_ggd[0].grid[0].space) == 1
        if not c0:
            msg = "No grid seems to exist at idseq.grids_ggd[0].grid[0].space"
            raise Exception(msg)
        space0 = idseq.grids_ggd[0].grid[0].space[0]

        # Check it is triangular with more than 1 node and triangle
        nnod = len(space0.objects_per_dimension[0].object)
        ntri = len(space0.objects_per_dimension[2].object)
        if nnod <=1 or ntri <= 1:
            msg = "There seem to be an unsufficient number of nodes / triangles"
            raise Exception(msg)
        nodes = np.vstack([nod.geometry
                           for nod in space0.objects_per_dimension[0].object])
        indtri = np.vstack([tri.nodes
                            for tri in space0.objects_per_dimension[2].object])
        indtri = indtri.astype(int)
        c0 = indtri.shape[1] == 3
        c0 &= np.max(indtri) < nnod
        if not c0:
            msg = "The mesh does not seem to be trianguler:\n"
            msg += "    - each face has %s nodes\n"%str(indtri.shape[1])
            raise Exception(msg)

        # Determine whether it is a piece-wise constant or linear interpolation
        if not npts in [nnod, ntri]:
            msg = "The number of values in 2d grid quantities is not conform\n"
            msg += "For a triangular grid, it should be either equal to:\n"
            msg += "    - the nb. of triangles (nearest neighbourg)\n"
            msg += "    - the nb. of nodes (linear interpolation)"
            raise Exception(msg)
        ftype = 'linear' if npts == nnod else 'nearest'
        indtri = cls._checkformat_tri(nodes, indtri)
        mpltri = mpl.tri.Triangulation(nodes[:,0], nodes[:,1], indtri)
        dmesh = {'nodes':nodes, 'triangles':indtri, 'ftype':ftype,
                 'nnodes':nnod,'ntri':ntri,'mpltri':mpltri}
        return dmesh

    def _comp(self, qq, lq, nt, npts):
        val = np.full((nt,npts), np.nan)
        if qq == 'b_field_norm':
            val[:] = np.sqrt(np.sum([qi**2
                                     for qi in lq], axis=0))

        elif qq == 'rho_pol_norm':
            psi = lq[0]
            psiM0 = psi - psi[:,0:1]
            psi10 = psi[:,-1] - psi[:,0]
            ind = psi10 != 0.
            val[ind,:] = np.sqrt( psiM0[ind,:] / psi10[ind,None] )
        return val

    @staticmethod
    def _checkformat_tri(nodes, indtri):
        x = nodes[indtri,0]
        y = nodes[indtri,1]
        orient = ((y[:,1]-y[:,0])*(x[:,2]-x[:,1])
                  - (y[:,2]-y[:,1])*(x[:,1]-x[:,0]))

        indclock = orient > 0.
        if np.any(indclock):
            msg = "Some triangles in are not counter-clockwise\n"
            msg += "  (necessary for matplotlib.tri.Triangulation)\n"
            msg += "    => %s / %s triangles are re-defined"
            warnings.warn(msg)
            indtri[indclock,1], indtri[indclock,2] = (indtri[indclock,2],
                                                      indtri[indclock,1])
        return indtri


    #---------------------
    # Properties (read-only attributes)
    #---------------------

    @property
    def d2d(self):
        return self._d2d
    @property
    def d1d(self):
        return self._d1d
    @property
    def dmesh(self):
        return self._dmesh
    @property
    def dtime(self):
        return self._dtime
    @property
    def t(self):
        return self._dtime['t']

















    #####################
    #####################
    # Hidden extra methods for backup (not used yet)
    #####################


    def _get_ptsRZindpts(self, ptsRZ, indpts, nt):
        # Format ptsRZ
        if ptsRZ is not None:
            assert type(ptsRZ) in [list,tuple,np.ndarray]
            if type(ptsRZ) in [list,tuple]:
                ptsRZ = np.array(ptsRZ,dtype=float)
            C0 = ptsRZ.shape==(2,)
            C1 = ptsRZ.ndim==2 and 2 in ptsRZ.shape
            C2 = ptsRZ.ndim==3 and ptsRZ.shape[:2]==(nt,2)
            # Input format consistent with tofu
            assert C0 or C1 or C2, "ptsRZ must be a (2,), (2,N), (nt,2,N) np.ndarray !"

            # But output must be consistent with scipy.interpolate (N,2)
            if C0:
                ptsRZ = ptsRZ.reshape((1,2))
                npts = 1
            if C1:
                if not ptsRZ.shape[1]==2:
                    ptsRZ = ptsRZ.T
                npts = ptsRZ.shape[0]
            if C2:
                ptsRZ = ptsRZ.swapaxes(1,2)
                npts = ptsRZ.shape[1]

        # Format indpts
        if indpts is not None:
            if type(indpts) in [int,float,np.int64,np.float64]:
                indpts = np.array([indpts],dtype=int)
            else:
                indpts = np.asarray(indpts).astype(int).ravel()
            npts = indpts.size

        pts_all = ptsRZ is None and indpts is None
        if pts_all:
            npts = self._ids.ggd[0].r.size
            indpts = np.arange(0,npts)
        return ptsRZ, indpts, npts, pts_all



    # General interpolation method

    def _interp2d(self, lquant=[], ptsRZ=None, t=None,
                  indt=None, indpts=None, method_t='nearest', ravel=True):

        #######################
        # Basic check of inputs
        msg = "Arg lquant must be a single quantity (str) or a list of such"
        assert type(lquant) in [str,list], msg
        msg = "You cannot provide both t and indt !"
        assert np.sum([t is not None,indt is not None])<=1, msg
        msg =  "You cannot provide both ptsRZ and indpts !"
        assert np.sum([ptsRZ is not None,indpts is not None])<=1, msg


        ###################
        # Pre-format inputs

        # Make sure the required quantity is a list of allowed quantities
        lquant, nquant = self._format_lquant(lquant)

        # Time
        t, indt, nt, t_all = self._get_tindt(t, indt)

        # Space
        ptsRZ, indpts, npts, pts_all = self._get_ptsRZindpts(ptsRZ, indpts, nt)

        #########################
        # Prepare list of outputs
        lout = [np.full((nt,npts),np.nan) for ii in range(0,nquant)]

        #########################
        # Interp

        if t_all:
            if pts_all:
                lout = [self._get_quant(qq) for qq in lquant]
            elif ptsRZ is None:
                lout = [self._get_quant(qq)[:,indpts] for qq in lquant]
            else:
                gridpts = np.array([self._ids.ggd[0].r, self._ids.ggd[0].z]).T
                if ptsRZ.ndim==3:
                    for ii in range(0,nquant):
                        qq = self._get_quant(lquant[ii])
                        for jj in range(0,nt):
                            ff = scpinterp.LinearNDInterpolator(gridpts, qq[jj,:])
                            lout[ii][jj,:] = ff(ptsRZ[jj,:,:])
                else:
                    for ii in range(0,nquant):
                        qq = self._get_quant(lquant[ii]).T
                        ff = scpinterp.LinearNDInterpolator(gridpts, qq)
                        lout[ii] = ff(ptsRZ).T

        elif t is None:
            if pts_all:
                lout = [self._get_quant(qq)[indt,:] for qq in lquant]
            elif ptsRZ is None:
                for ii in range(0,nquant):
                    qq = self._get_quant(lquant[ii])[indt,:]
                    lout[ii] = qq[:,indpts]
            else:
                gridpts = np.array([self._ids.ggd[0].r, self._ids.ggd[0].z]).T
                if ptsRZ.ndim==3:
                    for ii in range(0,nquant):
                        qq = self._get_quant(lquant[ii])
                        for jj in range(0,nt):
                            ff = scpinterp.LinearNDInterpolator(gridpts, qq[indt[jj],:])
                            lout[ii][jj,:] = ff(ptsRZ[jj,:,:])
                else:
                    for ii in range(0,nquant):
                        qq = self._get_quant(lquant[ii]).T
                        ff = scpinterp.LinearNDInterpolator(gridpts, qq[:,indt])
                        lout[ii] = ff(ptsRZ).T
        else:

            if pts_all:
                for ii in range(0,nquant):
                    ff = scpinterp.interp1d(self._ids.time,
                                            self._get_quant(lquant[ii]),
                                            kind=method_t, axis=0,
                                            bounds_error=False,
                                            fill_value=np.nan)
                    lout[ii] = ff(t)
            elif ptsRZ is None:
                for ii in range(0,nquant):
                    ff = scpinterp.interp1d(self._ids.time,
                                            self._get_quant(lquant[ii])[:,indpts],
                                            kind=method_t, axis=0,
                                            bounds_error=False,
                                            fill_value=np.nan)
                    lout[ii] = ff(t)
            else:
                if method_t=='nearest':
                    indtt = np.array([np.nanargmin(np.abs(self._ids.time-tt))
                                      for tt in t])
                    indttu = np.unique(indtt)
                    ntu = indttu.size

                gridpts = np.array([self._ids.ggd[0].r, self._ids.ggd[0].z]).T
                if ptsRZ.ndim==3:
                    if method_t=='nearest':
                        #import datetime as dtm  # DB
                        for ii in range(0,nquant):
                            #tt0 = dtm.datetime.now()    # DB
                            qq = self._get_quant(lquant[ii])
                            #t0, t1, t2 = 0., 0., 0.    # DB
                            for jj in range(0,ntu):
                                #tt0 = dtm.datetime.now()    # DB
                                # time-consuming : 1
                                f = scpinterp.LinearNDInterpolator(gridpts,
                                                                   qq[indttu[jj],:])
                                #tt1 = dtm.datetime.now()    # DB
                                #t0 += (tt1-tt0).total_seconds() # DB
                                #tt2 = dtm.datetime.now()    # DB
                                # time-consuming : 3 (by far)
                                ind = (indtt==indttu[jj]).nonzero()[0]
                                #tt3 = dtm.datetime.now()    # DB
                                #t1 += (tt3-tt2).total_seconds() # DB
                                #tt4 = dtm.datetime.now()    # DB
                                # time-consuming : 2
                                lout[ii][ind,:] = f(ptsRZ[ind[0],:,:])[np.newaxis,:]
                                #tt5 = dtm.datetime.now()    # DB
                                #t2 += (tt5-tt4).total_seconds() # DB
                            #print("    ",lquant[ii]," - total:",t0+t1+t2)
                            #print("        LinearND creation:",t0)
                            #print("        Indices finding:",t1)
                            #print("        Interp computing:",t2)
                    else:
                        for ii in range(0,nquant):
                            ff = scpinterp.interp1d(self._ids.time,
                                                    self._get_quant(lquant[ii]),
                                                    kind=method_t, axis=0,
                                                    bounds_error=False,
                                                    fill_value=np.nan)
                            for jj in range(0,nt):
                                f = scpinterp.LinearNDInterpolator(gridpts,
                                                                   ff(t[jj]))
                                lout[ii][jj,:] = f(ptsRZ[jj,:,:])
                else:
                    if method_t=='nearest':
                        for ii in range(0,nquant):
                            qq = self._get_quant(lquant[ii]).T
                            f = scpinterp.LinearNDInterpolator(gridpts,
                                                               qq[:,indttu])
                            q = f(ptsRZ)
                            for jj in range(0,ntu):
                                ind = (indtt==indttu[jj]).nonzero()[0]
                                lout[ii][ind,:] = q[np.newaxis,:,jj]
                    else:
                        for ii in range(0,nquant):
                            ff = scpinterp.interp1d(self._ids.time,
                                                    self._get_quant(lquant[ii]),
                                                    kind=method_t, axis=0,
                                                    bounds_error=False,
                                                    fill_value=np.nan)
                            for jj in range(0,nt):
                                f = scpinterp.LinearNDInterpolator(gridpts, ff(t[jj]))
                                lout[ii][jj,:] = f(ptsRZ)
        if nquant==1 and ravel:
            lout = lout[0]
        return lout



    def _format_lval(self, lval, nquant, nt):
        msg = "lval must be a np.ndarray or a list of such !"
        assert type(lval) in [list, np.ndarray], msg
        if type(lval) is np.ndarray:
            lval = [lval]
        C0 = [type(vv) is np.ndarray and vv.ndim in [1,2] for vv in lval]
        msg = "lval must be a list of np.ndarray of ndim in [1,2] !"
        assert all(C0), msg
        assert len(lval)==nquant, "lval must have len()=={0}".format(nquant)
        C0 = [vv.ndim==1 or vv.shape[0]==nt for vv in lval]
        assert all(C0), ""
        for ii in range(0,nquant):
            if lval[ii].ndim==1:
                lval[ii] = np.tile(lval[ii],(nt,1))
            assert lval[ii].shape[0]==nt, "All lval must have shape[0]==nt !"
        return lval




    def _interp2d_inv(self, lquant=[], lval=[],
                      t=None, indt=None, method_t='nearest',
                      line=None, res=None, ravel=True):

        #######################
        # Basic check of inputs
        msg = "Arg lquant must be a single quantity (str) or a list of such"
        assert type(lquant) in [str,list], msg
        msg = "You cannot provide both t and indt !"
        assert np.sum([t is not None,indt is not None])<=1, msg
        line = np.asarray(line).astype(float)
        C0 = line.shape==(2,2) and type(res) in [float,int,np.float64,np.int64]
        C1 = line.ndim==2 and 2 in line.shape and max(line.shape)>2
        msg = "Provide either:"
        msg += "\n    - line = (2,2) np.ndarray and res is a float (m)"
        msg += "\n    - line = (N,2) np.ndarray and res=None (implicit in line)"
        assert C0 or C1, msg

        ###################
        # Pre-format inputs

        # Make sure the required quantity is a list of allowed quantities
        lquant, nquant = self._format_lquant(lquant)

        # Time
        t, indt, nt, t_all = self._get_tindt(t, indt)

        # Desired values
        lval = self._format_lval(lval, nquant, nt)

        # Line
        if C0:
            vect = line[:,1]-line[:,0]
            d = np.linalg.norm(vect)
            u = vect/d
            k = np.linspace(0,d,np.ceil(d/res))
            line = line[:,0:1] + k[np.newaxis,:]*u[:,np.newaxis]
        else:
            k = np.r_[0., np.sqrt(np.sum(np.diff(line,axis=0)**2,axis=1))]
            d = np.sum(k)
            # Valid only for straight lines !
            u = (line[:,-1]-line[:,0])/d
        npts = k.size

        ######################
        # Start interp
        lq = self.interp2d(lquant, ptsRZ=line.T, t=t, indt=indt,
                           method_t=method_t, ravel=False)
        lout = [None for ii in range(0,nquant)]
        for ii in range(0,nquant):
            nval = lval[ii].shape[1]
            lout[ii] = np.full((nt,2,nval),np.nan)
            assert lq[ii].shape==(nt,npts)
            for jj in range(0,lq[ii].shape[0]):
                inds = np.argsort(lq[ii][jj,:])
                kk = np.interp(lval[ii][jj,:], lq[ii][jj,inds], k[inds],
                               left=np.nan, right=np.nan)
                pts = line[:,0:1] + kk[np.newaxis,:]*u[:,np.newaxis]
                lout[ii][jj,:,:] = pts

        if nquant==1 and ravel:
            lout = lout[0]
        return lout
