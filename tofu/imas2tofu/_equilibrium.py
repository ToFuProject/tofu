

# Built-in
import os
import warnings

# Common
import numpy as np
import scipy.interpolate as scpinterp

# tofu-specific
import tofu.tofu2imas._utils as _utils

#import tools as tools






class Eq2D(object):
    """ A generic class for handling 2D magnetic equilibria and getting basic
    interpolations and quantities from an EQUINOX ids

    """


    def __init__(self, **kwdargs):
        assert kwdargs
        lkeys = list(kwdargs.keys())

        largs = ['shot','usr','run','machine','occ']
        lin = [ss for ss in largs if ss in lkeys]
        C0 = 'shot' in lin and kwdargs['shot'] is not None
        C1 = 'ids' in lkeys and kwdargs['ids'] is not None
        assert np.sum([C0,C1])<=1, "Provide either ids or shot !"

        ids = None
        if C0:
            din = dict([(kk,kwdargs[kk]) for kk in lin])
            ids = self._get_ids(**din)
        elif C1:
            ids = kwdargs['ids']
        self._set_Eq_from_ids(ids)


    ######################
    # Read-only attributes

    @property
    def ids(self):
        return self._ids


    #####################
    # Hidden methods for setting the reference equilibrium quantities

    def _get_ids(self, shot=_shot, usr=_usr, machine=_machine, run=_run, occ=_occ):
        # IRFM-specific
        import imas_west

        # Check if shot exists
        run_number = '{:04d}'.format(run)
        shot_file  = os.path.expanduser('~' + usr + '/public/imasdb/' + machine + \
                                        '/3/0/' + 'ids_' + str(shot) + run_number + \
                                        '.datafile')
        if (not os.path.isfile(shot_file)):
            raise FileNotFoundError('IMAS file does not exist')

        # Get ids
        ids = imas_west.get(shot=shot, ids_name='equilibrium',
                            imas_run=run, imas_user=usr,
                            imas_machine=machine, imas_occurrence=occ)
        return ids

    def _set_Eq_from_ids(self, ids):
        self._ids = ids

        self._lquant_ids = ['psi', 'phi', 'theta',
                            'j_tor', 'j_parallel',
                            'b_field_r', 'b_field_z', 'b_field_tor']
        self._lquant_temp = ['b_field_norm', 'rho_pol_norm',
                             'rho_tor', 'rho_tor_norm']
        self._lquant_total = self._lquant_ids + self._lquant_temp
        self._dtemp = dict([(ss,None) for ss in self._lquant_temp])


    #####################
    # Hidden methods for getting equilibrium quantities

    def _get_quant(self, quant):
        assert self.ids is not None, "ids was not set !"
        if quant in self._lquant_ids:
            out = eval('self._ids.ggd[0].{0}'.format(quant))
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
