

# Built-in
import sys
import os
import warnings
import itertools as itt
import operator

if sys.version[0] == '2':
    import funcsigs as inspect
else:
    import inspect


# Common
import numpy as np
import scipy.interpolate as scpinterp
import matplotlib as mpl

# tofu-specific
try:
    import tofu.imas2tofu._utils as _utils
    import tofu.data as tfd
except Exception:
    from . import _utils
    from .. import data as tfd

# imas-specific
import imas



__all__ = ['Plasma2DLoader', 'load_Plasma2D']



#######################################
#######################################
#       class
######################################



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


    _dqmap = {'Te':{'lq':['electrons.temperature','t_e','Te'],
                    'units':'eV'},
              'ne':{'lq':['electrons.density','n_e','ne'],
                    'units':'m-3'},
              'psi':{'lq':['psi','grid.psi'],
                     'units':'a.u.'},
              'phi':{'lq':['phi','grid.phi'],
                     'units':'a.u.'},
              'rhotn':{'lq':['rho_tor_norm','grid.rho_tor_norm'],
                       'units':'adim.'},
              'rhopn':{'lq':['rho_pol_norm','grid.rho_pol_norm'],
                       'units':'adim.'},
              'zeff':{'lq':['zeff'], 'units':'adim.'},
              'prad':{'lq':['prad'], 'units':'W/m3'},
              'time':{'lq':['time'], 'units':'s'}}

    _didsk = {'tokamak':15, 'user':15, 'version':7,
              'shot':6, 'run':3, 'occ':3, 'shotr':6, 'runr':3}


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
                if k0 == lk0[0]:
                    msg = ''.rjust(20)
                    msg += '  '.join([kk.rjust(vv)
                                      for kk,vv in self._didsk.items()])
                    print(msg)
                msg = ("Getting %s..."%k0).rjust(20)
                if dids[k0]['dict'] is not None:
                    msg += '  '.join([str(dids[k0]['dict'][kk]).rjust(vv)
                                        for kk,vv in self._didsk.items()])
                print(msg)
            idsnode = self._get_idsnode(dids[k0]['ids'], k0)

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
        self._dids = dids




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
                    if not hasattr(obj,q1):
                        continue
                    npts = len(getattr(obj,q1))
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
        dmesh = {'nodes':nodes, 'faces':indtri,
                 'type':'tri', 'ftype':ftype,
                 'nnodes':nnod,'nfaces':ntri,'mpltri':mpltri}
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
    # Export to Plasma2D
    #---------------------

    def _get_quantunitsfromname(self,qq):
        lk = [kk for kk,vv in self._dqmap.items()
              if qq in vv['lq']]
        if len(lk) == 1:
            quant = lk[0]
            units = self._dqmap[quant]['units']
        else:
            quant = qq
            units = 'a.u.'
        return quant, units


    def to_object(self, Name=None, config=None, out=object):

        # ---------------------------
        # Preliminary checks on data source consistency
        lc = np.array([vv['ids'].shot for vv in self._dids.values()
                       if vv['ids'] is not None])
        if not np.all(lc==lc[0]):
            msg = "All data sources do not refer to the same shot:\n"
            msg += "    - %s\n"%str(lc)
            msg += "  => shot %s is used for labelling"%str(lc[0])
            warnings.warn(msg)
        shot = lc[0]

        lc = [vv['dict']['tokamak'] for vv in self._dids.values()
              if vv['dict'] is not None]
        if not all([lc[0] == cc for cc in lc]):
            msg = "All data sources do not refer to the same tokamak:\n"
            msg += "    - %s"%str(lc)
            msg += "  => tokamak %s is used for labelling"%lc[0]
            raise Exception(msg)
        Exp = lc[0]

        if Name is None:
            Name = 'CustomName'

        # ---------------------------
        # dtime
        dtime = self._dtime

        # dmesh
        dmesh = {'eq':self._dmesh}

        # d1d
        d1d, d2d, dradius = {}, {}, {}
        for k0, v0 in self._dquant.items():
            for k1, v1 in v0.items():
                for qq, v2 in v1['dq'].items():
                    if v2['val'] is None:
                        continue
                    quant, units = self._get_quantunitsfromname(qq)
                    if k1 == '1d':
                        d1d[k0+'.'+qq] = {'data':v2['val'],
                                          'quant':quant,
                                          'units':units,
                                          'radius':k0,
                                          'time':k0}
                        if k0 not in dradius.keys():
                            dradius[k0] = {'size':v2['val'].shape[-1]}
                    elif k1 == '2d':
                        d2d[k0+'.'+qq] = {'data':v2['val'],
                                          'quant':quant,
                                          'units':units,
                                          'mesh':k0,
                                          'time':k0}

        plasma = dict(dtime=dtime, dradius=dradius, dmesh=dmesh,
                      d1d=d1d, d2d=d2d,
                      Exp=Exp, shot=shot, Name=Name, config=config)
        if out == object:
            plasma = tfd.Plasma2D( **plasma )
        return plasma


#######################################
#######################################
#       function
######################################



def load_Plasma2D(Name=None, config=None, out=object, **kwdargs):
    plasma = Plasma2DLoader(**kwdargs)
    return plasma.to_object(Name=Name, config=config, out=out)

sig = inspect.signature(Plasma2DLoader)
kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
params = list(sig.parameters.values())
params = params + [params[0].replace(name='Name', default=None, kind=kind),
                   params[0].replace(name='config', default=None, kind=kind)]
load_Plasma2D.__signature__ = sig.replace(parameters=params)

del sig, params, kind
