

# Built-in
import os
import warnings
import itertools as itt

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




class Plasma2D(object):
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

    _lquantall = ['phi','psi','rho_tor_norm', 'rho_pol_norm',
                  'b_field_r', 'b_field_z', 'b_field_tor', 'b_field_norm']

    _dcomp = {'b_field_norm':['b_field_r', 'b_field_z', 'b_field_tor'],
              'rho_pol_norm':['psi']}

    _dquantmap1d = ['psi','phi','rho_tor_norm','rho_pol_norm']

    # Pre-set lists of quantities to be loaded from the ids, useful in typical use cases
    _dpreset = {'pos':{'2d':['psi','phi'],
                       '1d':['phi','psi','rho_tor_norm','rho_pol_norm']}}


    #----------------
    # Class creation and instanciation
    #----------------

    def __init__(self, preset=None, tlim=None,
                 dids_2d=None, lquant_2d=None, d2d=None,
                 dids_1d=None, lquant_1d=None, d1d=None,
                 dmesh=None, dtime=None,
                 verb=True):

        if verb:
            print("(1/2) Get equilibrium ids from imas, 2d quantities...")

        # Check inputs
        out = self._checkformat_dins(preset=preset,
                                     dids_2d=dids_2d, lquant_2d=lquant_2d,
                                     d2d=d2d, dids_1d=dids_1d,
                                     lquant_1d=lquant_1d, d1d=d1d,
                                     dtime=dtime, dmesh=dmesh)
        dids_2d, lquant_2d, d2d, dids_1d, lquant_1d, d1d, dtime, dmesh = out

        # Get d2d from ids
        idseq2 = None
        if d2d is None:
            ids2 = self._openids(dids_2d)
            idseq2 = ids2.equilibrium
            # dtime and dmesh
            dtime = self._checkformat_tlim(tlim, idseq2)
            lqav, npts = self._checkformat_lquantav(idseq2, nd=2)
            dmesh = self._get_dmesh(idseq2, npts)
            # quantities stored in ids
            self._dquantav = {2:{'lqav':lqav, 'npts':npts}}
            d2d, valid2 = self._get_dnd(idseq2, lquant_2d,
                                        dtime['nt'], dtime['indt'], nd=2)
            # compute missing quantities
            d2d = self._comp_quant(d2d, lquant_2d, dtime['nt'], npts, nd=2)


        if verb:
            print("(2/2) Get equilibrium ids from imas, 1d quantities...")

        # Get d1d from ids
        idseq1 = None
        if d1d is None:
            if dids_1d == dids_2d:
                ids1 = ids2
                t = None
            else:
                ids1 = self._openids(dids_1d)
                t = idseq1.time
            idseq1 = ids1.equilibrium
            # quantities stored in ids
            lqav, npts = self._checkformat_lquantav(idseq1, nd=1)
            self._dquantav[1] = {'lqav':lqav, 'npts':npts}
            d1d, valid1 = self._get_dnd(idseq1, lquant_1d,
                                        dtime['nt'], dtime['indt'], nd=1,
                                        t=t, tref=dtime['t'])
            dtime['valid'] = np.vstack((valid2,valid1)).T
            # compute missing quantities
            d1d = self._comp_quant(d1d, lquant_1d, dtime['nt'], npts, nd=1)

        # close idss
        if idseq2 is not None:
            ids2.close()
        if idseq1 is not None and dids_1d != dids_2d:
            ids1.close()

        # Check if remapping can be done
        self._lquantboth = list(set(d2d.keys()).intersection(d1d.keys()))
        if len(self._lquantboth) == 0:
            msg = "No quantity seems available both in 2d and 1d!\n"
            msg += "  => impossible to perform pts2profiles1d interpolations!"
            warnings.warn(msg)

        # -------------------------------------
        # Update dict
        self._dtime = dtime
        self._dmesh = dmesh
        self._d2d = d2d
        self._d1d = d1d





    #---------------------
    # Methods for checking and formatting inputs
    #---------------------

    @staticmethod
    def _checkformat_didslquantdnd(dids=None, lquant=None, dnd=None, nd=1):

        # Conflicts
        lc = [dids is not None and lquant is not None, dnd is not None]

        if np.sum(lc) != 1:
            msg = "Provide (dids_{0}d,lquant_{0}d) xor d{0}d !".format(nd)
            raise Exception(msg)

        # dids and dnd
        if lc[0]:
            dids = _utils._get_defaults( dids=dids )
        else:
            c0 = type(dnd) is dict
            c0 &= all([type(kk) is str for kk in dnd.keys()])
            c0 &= all([type(vv) is np.ndarray for vv in dnd.values()])
            lv = list(dnd.values())
            c0 &= all([vv.shape == lv[0].shape for vv in dnd.values()])
            if not c0:
                msg = "All values in dids_{}d must be:\n".format(nd)
                msg += "    - np.ndarray of same shape!"
                raise Exception(msg)

        # lquant
        c0 = type(lquant) is str
        c1 = type(lquant) is list and all([type(ss) is str
                                           for ss in lquant])
        if not (c0 or c1):
            msg = "Arg lquant_{0}d must be a str or a list of str !".format(nd)
            raise Exception(msg)
        if c0:
            lquant = [lquant]
        return dids, lquant, dnd

    @staticmethod
    def _checkformat_dtimemesh(dtime, dmesh, d2d):
        assert type(dtime) is dict and type(dmesh) is dict

        # dtime
        lk = ['indt','nt','t','tlim']
        assert sorted(dtime.keys()) == sorted(lk)
        dtime['t'] = np.asarray(dtime['t']).ravel()
        dtime['indt'] = np.asarray(dtime['indt']).ravel().astype(int)
        dtime['tlim'] = np.asarray(dtime['tlim']).ravel()
        assert dtime['nt'] == dtime['t'].size == dtime['indt'].size
        assert np.allclose(dtime['t'], np.unique(dtime['t']))
        assert np.allclose(dtime['indt'], np.unique(dtime['indt']))
        assert dtime['tlim'].size == 2

        # dmesh
        lk = ['nodes','triangles','nnodes','ntri','mpltri','fkind']
        assert sorted(dmesh.keys()) == sorted(lk)
        dmesh['nodes'] = np.atleast_2d(dmesh['nodes'], dtype=float)
        dmesh['triangles'] = np.atleast_2d(dmesh['triangles'], dtype=int)
        assert dmesh['nodes'].shape == (dmesh['nnodes'],2)
        assert dmesh['triangles'].shape == (dmesh['ntri'],3)
        assert np.max(dmesh['triangles']) < dmesh['nnodes']
        if 'mpltri' not in dmesh.keys() or dmesh['mpltri'] is None:
            dmesh['mpltri'] = mpl.tri.Triangulation(dmesh['nodes'][:,0],
                                                    dmesh['nodes'][:,1],
                                                    dmesh['triangles'])
        assert isinstance(dmesh['mpltri'], mpl.tri.Triangulation)
        assert dmesh['fkind'] in ['nearest','linear']
        return dtime, dmesh


    @classmethod
    def _checkformat_dins(cls, preset=None,
                          dids_2d=None, lquant_2d=None, d2d=None,
                          dids_1d=None, lquant_1d=None, d1d=None,
                          dtime=None, dmesh=None):

        if preset is not None:
            if not d2d is None and d1d is None:
                msg = "Cannot use preset and d2d/d1d simultaneously !"
                raise Exception(msg)
            lquant_2d = cls._dpreset[preset]['2d']
            lquant_1d = cls._dpreset[preset]['1d']

        # Check 2d
        dids_2d, lquant_2d, d2d = cls._checkformat_didslquantdnd(dids_2d,
                                                                 lquant_2d,
                                                                 d2d, nd=2)
        # Check 1d
        if d1d is None and lquant_1d is not None and dids_1d is None:
            dids_1d = dict(dids_2d)
        dids_1d, lquant_1d, d1d = cls._checkformat_didslquantdnd(dids_1d,
                                                                 lquant_1d,
                                                                 d1d, nd=1)
        # Check dmesh and dtime
        lc = [dmesh is None, dtime is None]
        c0 = all(lc) and d2d is None
        c1 = not any(lc) and d2d is not None
        if (np.sum(lc) not in [0,2]) or not (c0 or c1):
            msg = ""
            raise Exception(msg)

        if c1:
            dtime, dmesh = cls._checkformat_dtimemesh(dtime, dmesh, d2d)
            assert dtime['nt'] == list(d2d.values())[0].shape[0]
            if d1d is not None:
                assert dtime['nt'] == list(d1d.values())[0].shape[0]

        return dids_2d, lquant_2d, d2d, dids_1d, lquant_1d, d1d, dtime, dmesh

    @staticmethod
    def _checkformat_tlim(tlim, idseq):
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
        ids = imas.ids(s=dids['shot'], r=dids['run'])
        ids.open_env(dids['user'], dids['tokamak'], dids['version'])
        ids.equilibrium.get()
        return ids

    @staticmethod
    def _checkformat_lquantav(idseq, nd=1):
        lout = ['get','getNodeType','grid','setExpIdx','error']

        # Check
        if nd == 2:
            obj = idseq.time_slice[0].ggd[0]
        else:
            obj = idseq.time_slice[0].profiles_1d

        lq = [ss for ss in dir(obj)
              if ss[0] != '_' and all([ou not in ss for ou in lout])]

        if nd == 2:
            lqav = [qq for qq in lq
                    if (hasattr(getattr(obj,qq),'__len__')
                        and len(getattr(obj,qq)) > 0
                        and len(getattr(obj,qq)[0].values) > 0)]
            llen = np.array([len(getattr(obj,qq)[0].values) for qq in lqav])
            assert np.all(llen == llen[0])
            if len(lqav) > 0:
                  npts = llen[0]
        else:
            lqav = [qq for qq in lq
                    if (hasattr(getattr(obj,qq),'__len__')
                        and len(getattr(obj,qq)) > 0)]
            llen = np.array([len(getattr(obj,qq)) for qq in lqav])
            assert np.all(llen == llen[0])
            if len(lqav) > 0:
                npts = llen[0]

        if len(lqav) == 0:
            npts = 0
        return lqav, npts

    def _get_dnd(self, idseq, lquant, nt, indt, nd=1, t=None, tref=None):

        # Check availability or computability
        lout = [qq for qq in lquant if qq not in self._dquantav[nd]['lqav']]
        lnot = [qq for qq in lout if qq not in self._dcomp.keys()
                or any([qi not in self._dquantav[nd]['lqav'] for qi in self._dcomp[qq]])]
        if len(lnot) > 0:
            msg = "Some quantities not available / not computable from ids:\n"
            msg += "    - " + "\n    - ".join(lnot)
            import ipdb
            ipdb.set_trace()
            raise Exception(msg)

        # get size
        npts = self._dquantav[nd]['npts']

        # Get data
        valid = np.ones((nt,),dtype=bool)
        dnd = dict([(qq, np.full((nt, npts), np.nan)) for qq in lquant])
        if nd == 1:
            if t is None:
                for ii in range(0,nt):
                    if idseq.code.output_flag[indt[ii]] <= -1:
                       valid[ii] = False
                       continue
                    eqii = idseq.time_slice[indt[ii]].profiles_1d
                    for qq in lquant:
                        dnd[qq][ii,:] = getattr(eqii, qq)
            else:
                indtref = np.digitize(tref, 0.5*(t[1:]+t[:-1]))
                for ii in range(0,nt):
                    indtii = indtref[indt[ii]]
                    if idseq.code.output_flag[indtii] <= -1:
                       valid[ii] = False
                       continue
                    eqii = idseq.time_slice[indtii].profiles_1d
                    for qq in lquant:
                        dnd[qq][ii,:] = getattr(eqii, qq)

        else:
            for ii in range(0,nt):
                if idseq.code.output_flag[indt[ii]] <= -1:
                   valid[ii] = False
                   continue
                eqii = idseq.time_slice[indt[ii]].ggd[0]
                for qq in lquant:
                    dnd[qq][ii,:] = getattr(eqii, qq)[0].values

        return dnd, valid

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

    @classmethod
    def _comp_quant(cls, dnd, lquant, nt, npts, nd=1):
        for qq in lquant:
            c0 = qq not in dnd.keys() or dnd[qq] is None
            c0 = c0 and qq in cls._dcomp.keys()
            c0 = c0 and all([ss in dnd.keys() for ss in cls._dcomp[qq]])
            if not c0:
                continue
            val = np.full((nt,npts), np.nan)
            if qq == 'b_field_norm':
                val[:] = np.sqrt(np.sum([dnd[qi]**2
                                         for qi in cls._dcomp[qq]], axis=0))

            elif qq == 'rho_pol_norm' and nd == 1:
                psi = dnd[self._dcomp[qq][0]]
                psiM0 = psi - psi[:,0:1]
                psi10 = psi[:,-1] - psi[:,0]
                ind = psi10 != 0.
                val[ind,:] = np.sqrt( psiM0[ind,:] / psi10[ind,None] )
            dnd[qq] = val
        return dnd

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


    #---------------------
    # Methods for interpolation
    #---------------------

    def _interp_pts2profile1d(self, ptsRZ, quant, indt, indtu, ref='psi',
                              kind='linear', fill_value=np.nan):

        nteq = indtu.size
        if indt is None:
            nt = nteq
        else:
            nt = indt.size
        npts = ptsRZ.shape[1]

        # aliases
        mpltri = self._dmesh['mpltri']
        trifind = mpltri.get_trifinder()
        vref = self._d2d[ref]
        r, z = ptsRZ[0,:], ptsRZ[1,:]

        # loop on time
        val = np.full((nt, npts), np.nan)

        if quant == ref:
            for ii in range(0,nteq):
                if not self.dtime['valid'][indtu[ii],0]:
                    continue

                ind = ii if indt is None else indt == indtu[ii]
                # get ref values for mapping
                vrefii = mpl.tri.LinearTriInterpolator(mpltri,
                                                       vref[indtu[ii],:],
                                                       trifinder=trifind)(r,z)
                # Broadcast in time
                val[ind,:] = vrefii[None,:]

        else:
            refprof = self._d1d[ref]
            prof = self._d1d[quant]

            for ii in range(0,nteq):
                if not np.all(self._dtime['valid'][indtu[ii],:]):
                    continue

                ind = ii if indt is None else indt == indtu[ii]
                # get ref values for mapping
                vrefii = mpl.tri.LinearTriInterpolator(mpltri,
                                                       vref[indtu[ii],:],
                                                       trifinder=trifind)(r,z)

                # interpolate 1d
                vii = scpinterp.interp1d(refprof[indtu[ii],:],
                                         prof[indtu[ii],:],
                                         kind=kind,
                                         bounds_error=False,
                                         fill_value=fill_value)(np.asarray(vrefii))

                # Broadcast in time
                val[ind,:] = vii[None,:]

        return val


    def interp_pts2profile1d(self, ptsRZ, quant, t=None, ref=None,
                             kind='linear', fill_value=np.nan):
        """ Return the value of the desired profiles_1d quantity

        For the desired inputs points (pts):
            - pts are in (R,Z) coordinates
            - space interpolation is linear on the 1d profiles
        At the desired input times (t):
            - using a nearest-neighbourg approach for time

        """
        # Check requested quant is available
        if not quant in self.d1d.keys():
            msg = "Requested quant (%s) not available as 1d field:\n"
            msg += "Available quant are (set at instanciation):\n"
            msg += "    - %s"%str(self.d1d.keys())
            raise Exception(msg)

        # Check the ptsRZ is (2,npts) array of floats
        ptsRZ = np.atleast_2d(ptsRZ)
        if not 2 in ptsRZ.shape:
            msg = "ptsRZ must ba np.ndarray of (R,Z) points coordinates"
            raise Exception(msg)

        if ptsRZ.shape[0] != 2:
            ptsRZ = ptsRZ.T

        # Get time indices
        if t is None:
            indt = None
            indtu = np.arange(0, self.t.size)
        else:
            tbins = 0.5*(self.t[1:] + self.t[:-1])
            indt = np.digitize(t, tbins)
            indtu = np.unique(indt)

        # Check choosen ref is available
        lref = sorted(set(self._lquantboth).intersection(self._dquantmap1d))
        if ref is None:
            if quant in lref:
                ref = quant
            else:
                ref = 'phi'
        if ref not in lref:
            msg = "The chosen ref for remapping (%s) is not available:\n"
            msg += "    - Available refs: %s"%str(lref)
            raise Exception(msg)

        # Interpolation (including time broadcasting)
        val = self._interp_pts2profile1d(ptsRZ, quant,
                                         indt, indtu, ref=ref,
                                         kind='linear',
                                         fill_value=fill_value)

        return val






    #####################
    # Hidden extra methods for backup (not used yet)


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
