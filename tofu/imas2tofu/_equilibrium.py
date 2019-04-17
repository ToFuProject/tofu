

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




class Equilibrium2D(object):
    """ A generic class for handling 2D magnetic equilibria

    Provides:
        - equilibrium-related quantities
        - spatial interpolation methods

    """

    #----------------
    # Class attributes
    #----------------

    # Quantities can be either stored in the ids ('ids')
    # Or have to be computed ('comp')

    _lquantall = ['phi','psi','rho_tor_norm', 'rho_pol_norm',
                  'b_field_r', 'b_field_z', 'b_field_tor', 'b_field_norm']
    _dquantall = {'ggd':None, 'profiles_1d':None, 'profiles_2d':None}

    _dquantcomp = {'b_field_norm':['b_field_r', 'b_field_z', 'b_field_tor'],
                   'rho_pol_norm':['psi']}

    _dquantmap1d = ['psi','phi','rho_tor_norm','rho_pol_norm']

    # Pre-set lists of quantities to be loaded from the ids, useful in typical use cases
    _dpreset = {'pos':{'ggd':{'ids':['psi','phi'],
                              'comp':[]},
                       'profiles_1d':{'ids':['phi','psi',
                                             'rho_tor_norm', 'rho_pol_norm'],
                                      'comp':[]}}}


    #----------------
    # Class creation and instanciation
    #----------------

    def __init__(self, dquant='pos', tlim=None, user=None, shot=None, run=None, occ=None,
                 tokamak=None, version=None, dids=None, verb=True):

        if verb:
            print("Equilibrium")
            print("\t(1/2) Get equilibrium ids from imas...")

        # ids input dict
        dids = self._checkformat_dids(user=user, shot=shot, run=run, occ=occ,
                                      tokamak=tokamak, version=version, dids=dids)

        # get the ids
        ids = imas.ids(s=dids['shot'], r=dids['run'])
        ids.open_env(dids['user'], dids['tokamak'], dids['version'])
        ids.equilibrium.get()
        idseq = ids.equilibrium

        if verb:
            print("\t(2/2) Get time, 2d triangular mesh and profiles...")

        # tlim
        tlim, t, indt = self._checkformat_tlim(tlim, idseq)
        nt = t.size

        # check available quantities
        nggd, np1 = self._checkformat_avquant(idseq)

        # Extract 2D mesh as matplotlib triangulation
        nodes, indtri, ftype = self._checkformat_2dmesh(idseq, nggd)
        indtri = self._checkformat_tri(nodes, indtri)
        mpltri = mpl.tri.Triangulation(nodes[:,0], nodes[:,1],
                                       triangles=indtri)
        nnod, ntri = nodes.shape[0], indtri.shape[0]

        # Extract (or compute) required quantities
        dquant = self._checkformat_dquant(dquant)
        self._dquant = dict([(k0, {}) for k0 in dquant.keys()])
        indok = self._extract_quantids(dquant, idseq, nt, indt, nggd, np1)
        self._comp_dquant(dquant, nt, nggd, np1)

        # close ids
        ids.close()

        # Update dict
        self._dids = dids
        self._dmesh2d = {'mpltri':mpltri, 'ftype':ftype,
                         'nnod':nnod, 'ntri':ntri,
                         'nodes':None, 'faces':None}
        self._dtime = {'t':t, 'indt':indt, 'nt':nt, 'tlim':tlim, 'indok':indok}



    #---------------------
    # Methods for checking and formatting inputs
    #---------------------

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
        return tlim, t, indt

    @staticmethod
    def _checkformat_dids(user=None, shot=None, run=None, occ=None,
                          tokamak=None, version=None, dids=None):
        return _utils._get_defaults(user=user, shot=shot, run=run, occ=occ,
                                    tokamak=tokamak, version=version,
                                    dids=dids)

    def _checkformat_avquant(self, idseq):
        s0 = idseq.time_slice[0]
        lout = ['get','getNodeType','grid','setExpIdx']

        # Check 2D ggd
        lq = [ss for ss in dir(s0.ggd[0])
              if ss[0] != '_' and ss not in lout]

        lggd = [qq for qq in self._lquantall
                if qq in lq and len(getattr(s0.ggd[0],qq)[0].values) != 0]
        if len(lggd) > 0:
            nggd = len(getattr(s0.ggd[0],lggd[0])[0].values)
        else:
            nggd = 0

        # Check if in profiles_1d
        lq = [ss for ss in dir(s0.profiles_1d)
              if ss[0] != '_' and ss not in lout]
        lp1 = [qq for qq in self._lquantall
               if qq in lq and len(getattr(s0.profiles_1d,qq)) != 0]
        if len(lp1) > 0:
            np1 = len(getattr(s0.profiles_1d,lp1[0]))
        else:
            np1 = 0

        # Update instance attributes with actually available quantities
        self._lquantall = list(set(lggd).union(lp1))
        self._lquantboth = list(set(lggd).intersection(lp1))
        self._dquantall['ggd'] = lggd
        self._dquantall['profiles_1d'] = lp1

        if len(self._lquantboth) == 0:
            msg = "No quantity seems available both in ggd and profiles_1d!\n"
            msg += "  => impossible to perform pts2profiles1d interpolations!"
            warnings.warn(msg)
        return nggd , np1


    def _checkformat_dquant(self, dquant):
        if type(dquant) is str:
            if not dquant in self._dpreset.keys():
                msg = "If dquant is a str, must be a valid key of self._dpreset"
                raise Exception(msg)
            dquant = self._dpreset[dquant]

        c0 = (type(dquant) is dict
              and all([kk in ['ggd','profiles_1d'] for kk in dquant.keys()]))
        c01 = c0 and all([type(v) is list and all([type(ss) is str for ss in v])
                          for v in dquant.values()])
        c02 = c0 and all([type(v) is dict and all([vv in ['ids','comp']
                                               for vv in v.keys()])
                          for v in dquant.values()])
        if not (c01 or c02):
            msg = "Arg dquant must be dict of keys in ['ggd','profiles_1d']\n"
            msg += "Values must be either:\n"
            msg += "    - a list of quant str (see cls._lquantall for ex.)\n"
            msg += "    - a dict {'ids':l0, 'comp':l1}\n"
            msg += "        (where l0 and l1 are lists of quant str)"
            raise Exception(msg)

        if c01:
            dquant = {'ggd':{'ids':dquant['ggd'], 'comp':[]},
                      'profiles_1d':{'ids':dquant['profiles_1d'], 'comp':[]}}

        # Make sure no double and all quant in self._lquantall
        for k0 in dquant.keys():
            dquant[k0]['ids'] = sorted(set(dquant[k0]['ids']))
            dquant[k0]['comp'] = sorted(set(dquant[k0]['comp']))
            for qq in dquant[k0]['ids']:
                c0 = qq in self._dquantall[k0]
                if not c0:
                    dquant[k0]['ids'].remove(qq)
                    dquant[k0]['comp'].append(qq)

            dquant[k0]['ids'] = sorted(set(dquant[k0]['ids']))
            dquant[k0]['comp'] = sorted(set(dquant[k0]['comp']))
            for qq in dquant[k0]['comp']:
                c0 = qq in self._dquantcomp.keys()
                c0 &= all([qi in self._dquantall[k0]
                           for qi in self._dquantcomp[qq]])
                if not c0:
                    msg = "The following quant cannot be computed:\n"
                    msg += "    - %s"%qq
                    raise Exception(msg)
        return dquant


    @staticmethod
    def _checkformat_2dmesh(idseq, nval):

        # Check the grid exists
        c0 = len(idseq.grids_ggd) == 1
        c0 &= len(idseq.grids_ggd[0].grid) == 1
        c0 &= len(idseq.grids_ggd[0].grid[0].space) == 1
        if not c0:
            msg = ""
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
        if not nval in [nnod, ntri]:
            msg = "The number of values in 2d grid quantities is not conform\n"
            msg += "For a triangular grid, it should be either equal to:\n"
            msg += "    - the nb. of triangles (nearest neighbourg)\n"
            msg += "    - the nb. of nodes (linear interpolation)"
            raise Exception(msg)
        ftype = 'linear' if nval == nnod else 'nearest'
        return nodes, indtri, ftype


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
    # Methods for getting quant
    #---------------------


    def _extract_quantids(self, dquant, idseq, nt, indt, nggd, np1):

        # Prepare data arrays
        for qq in dquant['ggd']['ids']:
            self._dquant['ggd'][qq] = np.full((nt,nggd), np.nan)
        for qq in dquant['profiles_1d']['ids']:
            self._dquant['profiles_1d'][qq] = np.full((nt,np1), np.nan)

        indok = np.ones((nt,),dtype=bool)

        # Extract 2d quantities of interest
        for ii in range(0,nt):

            # Check output convergence flag
            if idseq.code.output_flag[indt[ii]] <= -1:
                indok[ii] = False
                continue

            # ggd
            idseqii = idseq.time_slice[indt[ii]]
            for qq in self._dquant['ggd'].keys():
                self._dquant['ggd'][qq][ii,:] = getattr(idseqii.ggd[0],qq)[0].values

            # profiles_1d
            for qq in self._dquant['profiles_1d'].keys():
                self._dquant['profiles_1d'][qq][ii,:] = getattr(idseqii.profiles_1d,qq)

        # Set indok in dtime
        return indok

    def _comp_dquant(self, dquant, nt, nggd, np1):
        for k0, v0 in dquant.items():
            nn = nggd if k0 == 'ggd' else np1
            for qq in v0['comp']:
                val = np.full((nt,nn), np.nan)

                if qq == 'b_field_norm':
                    val[:] = np.sqrt(np.sum([self._dquant[k0][qi]**2
                                             for qi in self._dquantcomp[qq]],
                                            axis=0))

                elif qq == 'rho_pol_norm' and k0 == 'profiles_1d':
                    psi = self._dquant[k0][self._dquantcomp[qq][0]]
                    psiM0 = psi - psi[:,0:1]
                    psi10 = psi[:,-1] - psi[:,0]
                    ind = psi10 != 0.
                    val[ind,:] = np.sqrt( psiM0[ind,:] / psi10[ind,None] )

                self._dquant[k0][qq] = val



    #---------------------
    # Properties (read-only attributes)
    #---------------------

    @property
    def dids(self):
        return self._dids
    @property
    def dmesh2d(self):
        return self._dmesh2d
    @property
    def dquant(self):
        return self._dquant
    @property
    def t(self):
        return self._dtime['t']


    #---------------------
    # Compute extra quantities
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
        mpltri = self._dmesh2d['mpltri']
        trifind = mpltri.get_trifinder()
        vref = self._dquant['ggd'][ref]
        r, z = ptsRZ[0,:], ptsRZ[1,:]

        # loop on time
        val = np.full((nt, npts), np.nan)

        if quant == ref:
            for ii in range(0,nteq):
                if not self._dtime['indok'][indtu[ii]]:
                    continue

                ind = ii if indt is None else indt == indtu[ii]
                # get ref values for mapping
                vrefii = mpl.tri.LinearTriInterpolator(mpltri,
                                                       vref[indtu[ii],:],
                                                       trifinder=trifind)(x,y)
                # Broadcast in time
                val[ind,:] = vrefii[None,:]

        else:
            refprof = self.dquant['profiles_1d'][ref]
            prof = self.dquant['profiles_1d'][quant]

            for ii in range(0,nteq):
                if not self._dtime['indok'][indtu[ii]]:
                    continue

                ind = ii if indt is None else indt == indtu[ii]
                # get ref values for mapping
                vrefii = mpl.tri.LinearTriInterpolator(mpltri,
                                                       vref[indtu[ii],:],
                                                       trifinder=trifind)(x,y)

                # interpolate 1d
                import ipdb
                ipdb.set_trace()

                vii = scpinterp.interp1d(refprof[indtu[ii],:], prof[indtu[ii],:],
                                         kind=kind, fill_value=fill_value)(vrefii)

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
        if not quant in self.dquant['profiles_1d'].keys():
            msg = "Requested quant (%s) not available as profiles_1d:\n"
            msg += "Available quant are (set at instanciation):\n"
            msg += "    - %s"%str(self.dquant['profiles_1d'].keys())
            raise Exception(msg)

        # Check the ptsRZ is (2,npts) array of floats
        ptsRZ = np.atleast_2d(ptsRZ)
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

        # Check chosen ref is available
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
    # Hidden methods for getting equilibrium quantities

    def _get_quant(self, quant):

        # Check
        if self.idseq is None:
            msg = "self.idseq is None!\n"
            msg += "    => you need to get the idseq first!\n"
            msg += "    => use self.get_idseq()"
            raise Exception(msg)

        if quant in self.dquant['ids'].keys():
            out = eval('self.idseq.ggd[0].{0}'.format(quant))
        else:
            if self._dtemp[quant] is None:
                if quant == 'b_field_norm':
                   q  = np.sqrt(self._ids.ggd[0].b_field_r**2
                                + self._ids.ggd[0].b_field_z**2
                                + self._ids.ggd[0].b_field_tor**2)
                elif quant == 'rho_pol_norm':
                    q = np.full(self._ids.ggd[0].psi.shape,np.nan)
                    psiM0 = self._ids.ggd[0].psi - self._ids.profiles_1d.psi[:,0:1]
                    psi10 = self._ids.profiles_1d.psi[:,-1] - self._ids.profiles_1d.psi[:,0]
                    ind = psi10!=0
                    q[ind,:] = np.sqrt(psiM0[ind,:]/psi10[ind,np.newaxis])
                elif 'rho_tor' in quant:
                    qq = self._ids.profiles_1d.rho_tor
                    if 'norm' in quant:
                        ind = (qq[:,-1]!=0.) & (~np.isnan(qq[:,-1]))
                        qq[ind,:] = qq[ind,:]/qq[ind,-1:]
                        qq[~ind,:] = np.nan
                    q = np.full(self._ids.ggd[0].psi.shape,np.nan)
                    for ii in range(0,self._ids.time.size):
                        q[ii,:] = np.interp(self._ids.ggd[0].psi[ii,:],
                                            self._ids.profiles_1d.psi[ii,:],
                                            qq[ii,:], right=np.nan)
                self._dtemp[quant] = q
            out = self._dtemp[quant]
        return out

    #####################
    # Hidden methods for interpolation

    def _format_lquant(self, lquant):
        if type(lquant) is str:
            lquant = [lquant]

        ok = np.array([ss in self._lquant_total for ss in lquant])
        indnotok = (~ok).nonzero()[0]
        if indnotok.size>0:
            msg = ["The following quantities cannot be interpolated:"]
            msg += [lquant[ii] for ii in indnotok]
            msg = "\n    ".join(msg)
            warnings.warn(msg)
        lquant = [lquant[ii] for ii in ok.nonzero()[0]]
        nquant = len(lquant)
        return lquant, nquant


    def _get_tindt(self, t, indt):
        # Format time vector
        if t is not None:
            if type(t) in [int,float,np.int64,np.float64]:
                t = np.array([t],dtype=float)
            else:
                t = np.asarray(t).astype(float).ravel()
            nt = t.size

        # Format indt vector
        if indt is not None:
            if type(indt) in [int,float,np.int64,np.float64]:
                indt = np.array([t],dtype=int)
            else:
                indt = np.asarray(indt).astype(int).ravel()
            nt = indt.size

        # If no tim provided, take all
        t_all = t is None and indt is None
        if t_all:
            nt = self._ids.time.size
            indt = np.arange(0,nt)

        return t, indt, nt, t_all


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



    #####################
    # General interpolation method

    def interp2d(self, lquant=[], ptsRZ=None, t=None,
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




    def interp2d_inv(self, lquant=[], lval=[],
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
