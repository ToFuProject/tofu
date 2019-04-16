

# Built-in
import os
import warnings
import itertools as itt

# Common
import numpy as np
import scipy.interpolate as scpinterp

# tofu-specific
import tofu.tofu2imas._utils as _utils

#import tools as tools




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
    _dquantdef = {'mesh2d':['psi','phi','j_tor','j_parallel',
                            'b_field_r', 'b_field_z', 'b_field_tor'],
                  'profiles1d':['rho_tor_norm'],
                  'comp':['b_field_norm', 'rho_pol_norm',
                          'rho_tor_norm','theta']}

    # The path (in the ids) of each quantity stored in it
    _dquantpath = {'rho_tor_norm':''}

    # Pre-set lists of quantitiesi to be loaded from the ids, useful in typical use cases
    _dpreset = {'pos':['psi', 'phi', 'theta', 'rho_pol_norm', 'rho_tor_norm'],
                'pos_tor':['phi', 'theta', 'rho_tor_norm'],
                'pos_pol':['psi', 'theta', 'rho_pol_norm'],
                'ece':['b_field_r', 'b_field_z', 'b_field_tor', 'b_field_norm']}


    #----------------
    # Class creation and instanciation
    #----------------

    def __init_subclass__(cls):
        # At class creation, deduce list of available quantities from class
        # attributes
        super(Equilibrium2D, cls).__init_subclass__()
        cls._lquant_av = sorted(set(itt.chain.from_iterable(cls._dquantdef.values())))


    def __init__(self, lquant=None, tlim=None, user=None, shot=None, run=None, occ=None,
                 tokamak=None, version=None, dids=None, get=True, verb=True):
        # Set the equilibrium dict
        self.set_dequilibrium(lquant=lquant, tlim=tlim, user=user, shot=shot, run=run, occ=occ,
                              tokamak=tokamak, version=version, dids=dids,
                              get=get, verb=verb)


    #---------------------
    # Methods for checking and formatting inputs
    #---------------------

    @staticmethod
    def _checkfomat_tlim(tlim):
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
        return tlim

    @staticmethod
    def _checkformat_dids(user=None, shot=None, run=None, occ=None,
                          tokamak=None, version=None, dids=None):
        return _utils._get_defaults(user=user, shot=shot, run=run, occ=occ,
                                    tokamak=tokamak, version=version,
                                    dids=dids)

    @classmethod
    def _checkformat_lquant(cls, lquant):
        lc = [type(lquant) is str, type(lquant) is list]
        if lc[0] and lquant in cls._dpreset.keys():
            lquant = cls._dpreset[lquant]
        elif lc[0] and lquant in cls._lquant_av:
            lquant = [lquant]
        elif lc[1] and all([ss in cls._lquant_av for ss in lquant]):
            lquant = sorted(set(lquant))
        else:
            lpreset = "[%s]"%(", ".join(cls._dpreset.keys()))
            lq = "[%s]"%(", ".join(cls._lquant_av))
            msg =  "Provided lquant does not match any known option\n"
            msg += "lquant should be either:\n"
            msg += "    - a key to a preset list of quantities : %s\n"%lpreset
            msg += "    - a valid quantity : %s\n"%lq
            msg += "    - a list of valid quantities"
            raise Exception(msg)
        return lquant

    @staticmethod
    def _checkformat_2dmesh(idseq):

        # Check the grid exists
        c0 = len(idseq.grids_ggd) == 1
        c0 &= len(idseq.grids_ggd[0].grid) == 1
        c0 &= len(idseq.grids_ggd[0].grid[0].space) == 1
        if not c0:
            msg = ""
            raise Exception(msg)
        space0 = idseq.grids_ggd[0].grid[0].space[0]

        # Check it is triangular with more than 1 node and triangle
        nnod = len(space0.objects_per_dimension[0])
        ntri = len(space0.objects_per_dimension[2])
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
        return nodes, indtri


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
    # Methods for getting / setting the equilibrium dict
    #---------------------

    def set_dEquilibrium(self, lquant=None, tlim=None, user=None, shot=None,
                         run=None, occ=None, tokamak=None, version=None,
                         dids=None, get=True, verb=True):
        # ids input dict
        dids = self._checkformat_dids(user=user, shot=shot, run=run, occ=occ,
                                      tokamak=tokamak, version=version, dids=dids)

        # Quantities
        lquant = self._checkformat_lquant(lquant)
        dquant = dict.fromkeys(lquant)

        # tlim
        tlim = self._checkformat_tlim(tlim)

        # equilibrium dict
        dEq = {'tlim':tlim,
               'dids':dids,
               'dquant':dquant}
        self._dequilibrium = dEq

        # get data
        if get:
            self.get_idseq_2d()
            self.comp_quant()

    def get_idseq(self, tlim=None):

        # Preformat
        if tlim is None:
            tlim = self._dequilibrium['tlim']
        else:
            tlim = self._checkformat_tlim(tlim)
            self._dequilibrium['tlim'] = tlim

        # get the ids
        dids = self.dids
        ids = imas.ids(s=dids['shot'], r=dids['run'])
        ids.open_env(dids['user'], dids['tokamak'], dids['version'])
        ids.equilibrium.get()
        idseq = ids.equilibrium

        # Extract time indices and vector
        t = np.asarray(idseq.time).ravel()
        indt = np.ones((t.size,), dtype=bool)
        if tlim is not None:
            indt[(t<tlim[0]) | (t>tlim[1])] = False
        t = t[indt]
        indt = np.nonzero()[0]
        nt = t.size

        # Determine which quantities are stored directly on the grid
        for qq in self._dquantdef['mesh2d']:
            if len(eval('idseq.ggd[0].%s'%qq)) == 0:
                self._dquantdef['mesh2d'].remove(qq)

        # Check all required quantities are available
        lq = set(self.lquant).intersection(self.__class__._dquantdef['mesh2d'])
        for qq in lq:
            if not qq in self._dquantdef['mesh2d']:
                msg = "A required field (%s) is missing from the ids:\n"%qq
                msg += "  len(ids.equilibrium.ggd[0].%s) = 0"%qq
                raise Exception(msg)

        # Extract 2D mesh as matplotlib triangulation
        nodes, indtri = self._checkformat_2dmesh(idseq)
        indtri = self._checkformat_tri(nodes, indtri)
        mpltri = mpl.tri.Triangulation(nodes[:,0], nodes[:,1],
                                       triangles=indtri)
        nnod, ntri = nodes.shape[0], indtri.shape[0]

        # Determine whether it is a piece-wise constant or linear interpolation
        nval = idseq.ggd[0].psi.values.size
        if not nval in [nnod, ntri]:
            msg = "The number of values in 2d grid quantities is not conform\n"
            msg += "For a triangular grid, it should be either equal to:\n"
            msg += "    - the nb. of triangles (nearest neighbourg)\n"
            msg += "    - the nb. of nodes (linear interpolation)"
            raise Exception(msg)
        ftype = 'linear' if nval == nnod else 'nearest'

        # Extract 2d quantities of interest
        l2d = set(self.lquant).intersection(self._dquantdef['ids2d'])
        dq2d = dict([(qq, np.full((nt,nval), np.nan)) for qq in l2d])

        for ii in range(0,nt):
            # Check output convergence flag
            if idseq.code.output_flag[indt[ii]] <= -1:
                continue
            idseqii = idseq.ggd[indt[ii]]
            for qq in l2d:
                dq2d[qq][ii,:] = eval('idseqii.%s.values'%qq)

        # Close ids
        ids.close()

       # Update dict
       self._dequilibrium['dmesh2d'] = {'mpltri':mpltri, 'ftype':ftype,
                                        'nodes':None, 'faces':None}
       self._dequilibrium['dquant2d'] = dq2d
       self._dequilibrium['indt'] = indt
       self._dequilibrium['t'] = t

    #---------------------
    # Properties (read-only attributes)
    #---------------------

    @property
    def dequilibrium(self):
        return self._dequilibrium
    @property
    def dids(self):
        return self._dequilibrium['dids']
    @property
    def dmesh2d(self):
        return self._dequilibrium['dmesh2d']
    @property
    def dquant2d(self):
        return self._dequilibrium['dquant2d']
    @property
    def lquant(self):
        return sorted(self._dequilibrium['dquant'].keys())


    #---------------------
    # Compute extra quantities
    #---------------------

    def _interp1d_phsi2profile1d(self, quant, phsi, mode='psi',
                                 kind='linear', fill_value=np.nan):

        idseq = self._dequilibrium['idseq']
        nt, indt = self._dequilibrium['nt'], self._dequilibrium['indt']
        npts = phsi.shape[1]

        # aliases
        mpltri = self._dequilibrium['dmesh2d']['mpltri']
        trifind = mpltri.get_trifinder()
        phsi = self._dequilibrium['dmesh2d']['dquant2d'][mode]

        # loop on time
        val = np.full((nt, npts), np.nan)
        for ii in range(0,self._dequilibrium['nt']):
            if idseq.code.output_flag[indt[ii]] <= -1:
                continue
            # get phi / psi values
            phsii = mpl.tri.LinearTriInterpolator(mpltri,
                                                  phsi[indt[ii],:],
                                                  trifinder=trifind)

            idseqii = self._dequilibrium['idseq'].profiles_1d[indt[ii]]
            val[ii,:] = scipy.interp1d(eval('idseqii.%s'%mode),
                                       eval('idseqii.%s'%quant)
                                       kind=kind, fill_value=fill_value)(phsii)
        return val


    def interp2d_pts2profile1d(self, ptsRZ, quant, t=None, mode='phi'):
        """ Return the value of the desired profiles_1d quantity

        For the desired inputs points (pts):
            - pts are in (R,Z) coordinates
            - space interpolation is linear on the 1d profiles
        At the desired input times (t):
            - using a nearest-neighbourg approach for time

        """

        # Get time indices
        if t is None:
            indt = np.arange(0,self._dequilibrium['nt'])
        else:
            tbins = 0.5*(self._dequilibrium['t'][1:]
                         + self._dequilibrium['t'][:-1])
            indt = np.digitize(t, tbins)
        nt = indt.size

        # Get phi / psi value for each pts
        phsi, mode = self._interp2d_pts2phsi(ptsRZ, indt=indt, nt=nt, mode=mode)

        # Interpolate in 1d over phi / psi
        if quant in self._dquantdef['profile_1d']:
            val = self._interp1d_phsi2profile1d(quant, phsi, mode=mode)

        elif quant in self._dquantdef['profile_2d']:
            raise Exception("Not coded yet !")

        elif quant in ['psi','phi']:
            val = phsi

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
