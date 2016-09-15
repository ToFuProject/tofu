# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 10:03:58 2014

@author: didiervezinet
"""

import numpy as np
import datetime as dtm
import warnings

# ToFu-specific
import tofu.defaults as tfd
import tofu.pathfile as tfpf
from . import _compute as _tfEq_c
from . import _plot as _tfEq_p


__all__ = ['Eq2D']



"""
###############################################################################
###############################################################################
                        Eq 2D class (for GBF)
###############################################################################
"""



class Eq2D(object):
    """ Create a 2D equilibrium object, which stores all relevant data (rho, q, theta mappings) for one time point """

    def __init__(self, Id, PtsCross, t=None, MagAx=None, Sep=None, rho_p=None, rho_t=None, surf=None, vol=None, q=None, jp=None, pf=None, tf=None, theta=None, thetastar=None, BTX=None, BRY=None, BZ=None, Ref=None,
                 Type='Tor', Exp=None, shot=None, Diag=None, dtime=None, dtimeIn=False, SavePath=None):

        self._Done = False
        self._set_Id(Id, Type=Type, Exp=Exp, shot=shot, Diag=Diag, dtime=dtime, dtimeIn=dtimeIn, SavePath=SavePath)

        self._Tabs_vRho = ['q','jp','rho_p','rho_t','surf','vol','pf','tf']
        self._Tabs_vPts_add = ['theta','thetastar','BTX','BRY','BZ']
        self._Tabs_vPts = self._Tabs_vPts_add + self._Tabs_vRho
        self._Tabs_Quants = ['MagAx','Sep'] + self._Tabs_vPts

        btx = r"$B_T$" if Type=='Tor' else r"$B_X$"
        bry = r"$B_R$" if Type=='Tor' else r"$B_Y$"
        self._Tabs_LTXUnits = {'q':{'LTX':r"$q$",'units':r""}, 'rho_p':{'LTX':r"$\rho_P$",'units':r""}, 'rho_t':{'LTX':r"$\rho_T$",'units':r""}, 'vol':{'LTX':r"$V$",'units':r"$m^3$"}, 'surf':{'LTX':r"$S$",'units':r"$m^2$"},
                               'pf':{'LTX':r"$\phi$",'units':r"$V.s$"}, 'tf':{'LTX':r"$\psi$",'units':r"$V.s$"}, 'jp':{'LTX':r"$j_P$",'units':r"$A$"},
                               'BTX':{'LTX':btx,'units':r"$T$"}, 'BRY':{'LTX':bry,'units':r"$T$"}, 'BZ':{'LTX':r"$B_Z$",'units':r"$T$"},
                               'theta':{'LTX':r"$\theta$",'units':r"rad."}, 'thetastar':{'LTX':r"$\theta^*$",'units':r"rad."}}

        self._preset_Tab()
        self._add_Eq(PtsCross=PtsCross, t=t, MagAx=MagAx, Sep=Sep, rho_p=rho_p, rho_t=rho_t, surf=surf, vol=vol, q=q, jp=jp, theta=theta, thetastar=thetastar, pf=pf, tf=tf, BTX=BTX, BRY=BRY, BZ=BZ, Ref=Ref)
        self._Done = True

    @property
    def Id(self):
        return self._Id
    @property
    def Type(self):
        return self.Id.Type
    @property
    def shot(self):
        return self.Id.shot
    @property
    def Diag(self):
        return self.Id.Diag

    @property
    def PtsCross(self):
        return self._Tab['PtsCross']
    @property
    def NP(self):
        return self._NP

    @property
    def t(self):
        return self._Tab['t']
    @property
    def Nt(self):
        return self._Nt

    @property
    def MagAx(self):
        return self._Tab['MagAx']
    @property
    def Sep(self):
        return self._Tab['Sep']
    @property
    def pf(self):
        return self._get_vPtsFromvRef('pf')
    @property
    def tf(self):
        return self._get_vPtsFromvRef('tf')
    @property
    def rho_p(self):
        return self._get_vPtsFromvRef('rho_p')
    @property
    def rho_t(self):
        return self._get_vPtsFromvRef('rho_t')
    @property
    def surf(self):
        return self._get_vPtsFromvRef('surf')
    @property
    def vol(self):
        return self._get_vPtsFromvRef('vol')
    @property
    def q(self):
        return self._get_vPtsFromvRef('q')
    @property
    def jp(self):
        return self._get_vPtsFromvRef('jp')
    @property
    def theta(self):
        return self._get_vPtsFromvRef('theta')
    @property
    def thetastar(self):
        return self._get_vPtsFromvRef('thetastar')
    @property
    def BTX(self):
        return self._get_vPtsFromvRef('BTX')
    @property
    def BRY(self):
        return self._get_vPtsFromvRef('BRY')
    @property
    def BZ(self):
        return self._get_vPtsFromvRef('BZ')
    @property
    def Tabs_vPts(self):
        return self._Tabs_vPts
    @property
    def Ref(self):
        return self._Tab['Ref']
    @property
    def NRef(self):
        return self._NRef



    def _check_inputs(self, Id=None, PtsCross=None, t=None, MagAx=None, Sep=None, rho_p=None, rho_t=None, surf=None, vol=None, q=None, jp=None, pf=None, tf=None, theta=None, thetastar=None, BTX=None, BRY=None, BZ=None, Ref=None,
                      Type=None, Exp=None, shot=None, Diag=None, dtime=None, dtimeIn=False, SavePath=None):
        _Eq2D_check_inputs(Id=Id, PtsCross=PtsCross, t=t, MagAx=MagAx, Sep=Sep, rho_p=rho_p, rho_t=rho_t, surf=surf, vol=vol, q=q, jp=jp, pf=pf, tf=tf, theta=theta, thetastar=thetastar, BTX=BTX, BRY=BRY, BZ=BZ, Ref=Ref,
                           Type=Type, Exp=Exp, shot=shot, Diag=Diag, dtime=dtime, dtimeIn=dtimeIn, SavePath=SavePath)

    def _set_Id(self, Id, Type=None, Exp=None, shot=None, Diag=None, dtime=None, dtimeIn=False, SavePath=None):
        if self._Done:
            Out = tfpf._get_FromItself(self.Id,{'Type':Type, 'Exp':Exp, 'shot':shot, 'Diag':Diag, 'dtime':dtime, '_dtimeIn':dtimeIn, 'SavePath':SavePath})
            Type, Exp, shot, Diag, dtime, dtimeIn, SavePath = Out['Type'], Out['Exp'], Out['shot'], Out['Diag'], Out['dtime'], Out['dtimeIn'], Out['SavePath']
        tfpf._check_NotNone({'Id':Id})
        self._check_inputs(Id=Id)
        if type(Id) is str:
            Exp = 'Test' if Exp is None else Exp
            tfpf._check_NotNone({'Exp':Exp, 'dtimeIn':dtimeIn})
            self._check_inputs(Type=Type, Exp=Exp, shot=shot, Diag=Diag, SavePath=SavePath, dtime=dtime, dtimeIn=dtimeIn)
            Id = tfpf.ID('Eq2D', Id, Type=Type, Exp=Exp, shot=shot, Diag=Diag, SavePath=SavePath, dtime=dtime, dtimeIn=dtimeIn)
        self._Id = Id

    def _preset_Tab(self, SepNPmax=100):
        self._Tab = {}

    def _add_Eq(self, PtsCross=None, t=None, MagAx=None, Sep=None, rho_p=None, rho_t=None, surf=None, vol=None, q=None, jp=None, theta=None, thetastar=None, pf=None, tf=None, BTX=None, BRY=None, BZ=None, Ref=None):
        """
        Stores all provided quantities of the equilibrium as a dict called Tab, only one radial quantity is stored as a 2D map (vPts), the others are mapped against this one (vRef)
        """
        self._check_inputs(PtsCross=PtsCross, t=t, MagAx=MagAx, Sep=Sep, rho_p=rho_p, rho_t=rho_t, surf=surf, vol=vol, q=q, jp=jp, theta=theta, thetastar=thetastar, pf=pf, tf=tf, BTX=BTX, BRY=BRY, BZ=BZ, Ref=Ref)
        for ss in self._Tabs_Quants:
            qq = eval(ss)
            self._Tab[ss] = qq
        self._Tab['Ref'] = Ref
        self._Tab['PtsCross'] = PtsCross
        self._Tab['t'] = t
        self._Nt = self._Tab[self.Ref]['vRef'].shape[0]
        self._NP = PtsCross.shape[1]
        self._NRef = self._Tab[self.Ref]['vRef'].shape[1]
        self._correct_Ref()

    def _correct_Ref(self):
        RefV = _tfEq_c._correctRef(self.Ref, self._Tab[self.Ref]['vPts'], self.Nt, self.PtsCross, self.MagAx)
        self._Tab[self.Ref]['vPts'] = RefV

    def _get_vPtsFromvRef(self, ss):
        assert ss in self._Tabs_Quants, "Provided quantity is not in self._Tabs_Quants !"
        if ss==self.Ref:
            vPts = self._Tab[ss]['vPts']
        elif not self._Tab[ss] is None:
            Extra = False
            vPts = np.nan*np.ones((self.Nt,self.NP))
            for ii in range(0,self.Nt):
                indsort = np.argsort(self._Tab[self.Ref]['vRef'][ii,:])
                vPts[ii,:] = np.interp(self._Tab[self.Ref]['vPts'][ii,:], self._Tab[self.Ref]['vRef'][ii,indsort], self._Tab[ss]['vRef'][ii,indsort], left=None, right=None)
                if ss in ['surf','vol','rho_p','rho_t','tf']:    # Extrapolate to 0 if necessary
                    r = np.hypot(self.PtsCross[0,:]-self.MagAx[ii,0], self.PtsCross[1,:]-self.MagAx[ii,1])
                    indnan = np.isnan(vPts[ii,:])
                    rmin = np.nanmin(r[~indnan])
                    if np.any(indnan & (r<=rmin)):
                        rminarg = ((r==rmin) & (~indnan)).nonzero()[0]
                        refmin = vPts[ii,rminarg]
                        N1 = np.isnan(vPts[ii,:]).sum()
                        vPts[ii,indnan] = refmin*r[indnan]/rmin
                        Extra = True
            if Extra:
                warnings.warn("{0} / {1} points close to the Magnetic Axis could be extrapolated to 0 because quantity = ".format(N1-np.isnan(vPts[ii,:]).sum(),np.isnan(vPts[ii,:]).sum())+ss)
        else:
            vPts = None
        return vPts

    def interp(self, Pts, Quant='rho_p', indt=None, t=None, deg=3, Test=True):
        """
        Interpolate the chosen equilibrium quantities at the provided points (from the stored/tabulated grid)

        Inputs:
        -------
            Pts         iterable            Iterable containing the 2D cartesian coordinates of points in a cross-section where the quantity of interest should be interpolated
            Quant       str / iterable      Key or iterable of keys designing the quantities to be interpolated (from self.Tabs_vPts)
            indt        None / int          Index (in time) at which the quantities should be plotted
            t           None / float        Time at which the quantities should be plotted (used if indt not provided)
            deg         int                 Degree to be used forthe 2D spline interpolation
            Test

        Outputs:
        --------
            dQuant      np.ndarray / dict   Array of b-spline interpolated quantities at desired times (first dimension) and points (second dimension), dictionary of uch if several quantities were asked

        """
        assert (Quant in self.Tabs_vPts) or (hasattr(Quant,'__iter__') and all([ss in self.Tabs_vPts for ss in Quant])), "Arg Quant must be a valid str or iterable of such !"
        Quant = [Quant] if type(Quant) is str else Quant
        Tab_vPts = [eval('self.'+ss) for ss in Quant]
        dQuant = _tfEq_c._interp_Quant(self.t, self.PtsCross, Tab_vPts, Pts, LQuant=Quant, indt=indt, t=t, deg=deg, Test=Test)
        return dQuant

    def get_RadDir(self, Pts, indt=None, t=None, Test=True):
        """
        Return an array of normalised vectors showing the direction of negative (outward) radial gradient at each input point

        Inputs:
        -------
            Pts         iterable            Iterable containing the 2D cartesian coordinates of points in a cross-section where the quantity of interest should be interpolated

        Outputs:
        --------
            rgrad

        """
        refrad = self._Tab[self.Ref]['vPts']
        rgrad = _tfEq_c._get_rgrad(Pts, self.PtsCross, refrad, self.t, indt=None, t=None, Test=Test)
        return rgrad


    def plot(self, V='inter', ax=None, Quant=['q','MagAx','Sep','pol'], plotfunc='contour', lvls=[1.,1.5,2.], indt=None, t=None, Cdict=None, RadDict=None, PolDict=None, MagAxDict=None, SepDict=None, LegDict=tfd.TorLegd,
             ZRef='MagAx', NP=100, Ratio=-0.1, ylim=None,
             NaNout=True, Abs=True, clab=True, cbar=False, draw=True, a4=False, Test=True):

        """
        Plot a 2D map of the chosen quantity, plus optionally the magnetic axis ('MagAx') and the separatrix ('Sep')

        Inputs:
        -------
            V           str                     Flag indicating which version of the plot is expected:
                                                    'static':   a simple 2D plot of the desired quantity
                                                    'inter':    an interactive figure, with a 2D plot, a 1D plot of a cut through the equatorial plane and time traces of the cut points, time is changed with left/right arrows
            ax          None / plt.Axes         Axes to be used for plotting, if None a new figure / axes is created
            Quant       str / iterable          Flag(s) indicating which 2D quantity to plot (from self._Tabs_vPts), if iterable, can contain also 'MagAx' and 'Sep', and 'rad' and 'pol' for direction arrows
            plotfunc    str                     Flag indicating which function to use for plotting in ['scatter','contour','contourf','imshow']
            lvls        None / iterable         Iterable of the levels to be used for plotfunc='contour'
            indt        None / int              Index (in time) at which the quantities should be plotted
            t           None / float            Time at which the quantities should be plotted (used if indt not provided)
            Cdict       None / dict             Dictionary of properties used for plotting the 2D quantity (fed to the chosen plotfunc), default used from tofu.defaults if None
            RadDict     None / dict             Dictionary of properties used for plotting the arrows of the radial direction (fed to plt.quiver()), default used from tofu.defaults if None
            PolDict     None / dict             Dictionary of properties used for plotting the arrows of the poloidal direction (fed to plt.quiver()), default used from tofu.defaults if None
            MagAxDict   None / dict             Dictionary of properties used for plotting the Magnetic Axis (fed to plt.plot()), default used from tofu.defaults if None
            SepDict     None / dict             Dictionary of properties used for plotting the Separatrix (fed to plt.plot()), default used from tofu.defaults if None
            LegDict     None / dict             Dictionary of properties for plotting the legend, legend not plotted if None
            ZRef        float / str             The height at which the equatorial cut should be made, if 'MagAx', the cut is made at the height of the magnetic axis, averaged over the time interval
            NP          int                     Number of points to be used for computing the quantity along the equatorial cut
            Ratio       float                   Ratio to be used for adjusting the width of the equatorial cut with respect to the width spanned by self.PtsCross (typically -0.1 for a small span or 0.1 for a larger one)
            ylim        None / iterable         If not None, a len()==2 iterable of floats to be used to set the ylim of the plot of the equatorial cut
            NaNout      bool                    Flag indicating whether all points outside of the separatrix shall be set to NaN
            Abs         bool                    Flag indicating whether the absolute value of the quantity should be plotted (e.g.: avoid negative q...)
            clab        bool                    Flag indicating whether the contours should be labelled with ther corresponding values (for plotfunc='contour' only)
            cbar        bool                    Flag indicating whether the colorbar axes should be plotted
            draw        bool                    Flag indicating whether the fig.canvas.draw() shall be called automatically
            a4          bool                    Flag indicating whether the figure should be plotted in a4 dimensions for printing
            Test        bool                    Flag indicating whether the inputs should be tested for conformity

        Outputs:
        --------
            ax          plt.Axes / dict         Axes used for plotting, returned as a single axes (V='static') or a dictionary of axes (V='inter')

        """
        ax = _tfEq_p.Eq2D_plot(self, Quant, V=V, ax=ax, plotfunc=plotfunc, lvls=lvls, indt=indt, t=t, clab=clab, Cdict=Cdict, RadDict=RadDict, PolDict=PolDict, MagAxDict=MagAxDict, SepDict=SepDict, LegDict=LegDict,
              ZRef=ZRef, NP=NP, Ratio=Ratio, ylim=ylim,
              VType=self.Id.Type, NaNout=NaNout, Abs=Abs, cbar=cbar, draw=draw, a4=a4, Test=Test)
        return ax


    def plot_vs(self, ax=None, Qy='q', Qx='rho_p', indt=None, t=None, Dict=None, xlim=None, ylim=None, Abs=True, LegDict=None, draw=True, a4=False, Test=True):
        """
        Plot a quantity vs another

        Inputs:
        -------
            ax          None / plt.Axes         Axes to be used for plotting, if None a new figure / axes is created
            Qy          str                     Flag indicating which quantity should be plotted on the y-axis
            Qx          str                     Flag indicating which quantity should be plotted on the x-axis
            indt        None / int              Index (in time) at which the quantities should be plotted
            t           None / float            Time at which the quantities should be plotted (used if indt not provided)
            Dict        dict                    Dictionary of properties used for plotting the line (fed to plt.plot()), default used from tofu.defaults if None
            Abs         bool                    Flag indicating whether the absolute value of the quantity should be plotted (e.g.: avoid negative q...)
            LegDict     None / dict             Dictionary of properties for plotting the legend, legend not plotted if None
            draw        bool                    Flag indicating whether the fig.canvas.draw() shall be called automatically
            a4          bool                    Flag indicating whether the figure should be plotted in a4 dimensions for printing
            Test        bool                    Flag indicating whether the inputs should be tested for conformity

        Outputs:
        --------
            ax          plt.Axes                Axes used for plotting

        """
        ax = _tfEq_p.Eq2D_plot_vs(self, ax=ax, Qy=Qy, Qx=Qx, indt=indt, t=t, Dict=Dict, Abs=Abs, LegDict=LegDict, xlim=xlim, ylim=ylim, draw=draw, a4=a4, Test=Test)
        return ax


    def save(self, SaveName=None, Path=None, Mode='npz'):
        """
        Save the object in folder Name, under file name SaveName, using specified mode

        Inputs:
        ------
            SaveName    str     The name to be used to for the saved file, if None (recommended) uses Ves.Id.SaveName (default: None)
            Path        str     Path specifying where to save the file, if None (recommended) uses Ves.Id.SavePath (default: None)
            Mode        str     Flag specifying whether to save the object as a numpy array file ('.npz', recommended) or an object using cPickle (not recommended, may cause retro-compatibility issues with later versions)
        """
        tfpf.Save_Generic(self, SaveName=SaveName, Path=Path, Mode=Mode)








def _Eq2D_check_inputs(Id=None, PtsCross=None, t=None, MagAx=None, Sep=None, rho_p=None, rho_t=None, surf=None, vol=None, q=None, jp=None, pf=None, tf=None, theta=None, thetastar=None, BTX=None, BRY=None, BZ=None, Ref=None,
                       Type=None, Exp=None, shot=None, Diag=None, dtime=None, dtimeIn=False, SavePath=None):

    if Id is not None:
        assert type(Id) in [str,tfpf.ID], "Arg Id must be a str or a tfpf.ID object !"

    # Here all inputs must be provided simultaneously !
    vRho = ['q','jp','rho_p','rho_t','surf','vol','pf','tf']
    vPts_add = ['theta','thetastar','BTX','BRY','BZ']
    vPts = vPts_add + vPts_add
    Quants = ['MagAx','Sep'] + vPts

    if PtsCross is not None:
        pts = np.asarray(PtsCross)
        assert np.asarray(pts).ndim==2 and 2 in np.asarray(pts).shape and max(np.asarray(pts).shape)>2 and pts.dtype.name=='float64', "Arg PtsCross must be a 2-dim np.ndarray of floats !"
        NP = max(pts.shape)

    assert Ref is None or (type(Ref) is str and Ref in vRho), "Arg Ref must be a str in vRho !"
    if any([eval(ss+' is not None') for ss in vRho]):
        assert Ref is not None, "Arg Ref must be provided if any vRho is provided too !"
        assert eval(Ref+' is not None'), "The Ref quantity must be provided if any vRho is provided too !"
        Sh = eval(Ref+"['vRef'].shape")
        for ss in vRho:
            val = eval(ss)
            if val is not None:
                assert type(val) is dict and all([kk in val.keys() for kk in ['vPts','vRef']]), "Arg "+ss+" must be a dict with keys ['vPts','vRef'] !"
                assert val['vRef'] is None or (type(val['vRef']) is np.ndarray and val['vRef'].ndim==2 and val['vRef'].dtype.name=='float64'), "Arg "+ss+"['vRef'] must be a 2-dim np.ndarray of floats !"
                assert val['vRef'].shape==Sh, "Arg "+ss+"['vRef'] must be the same shape as Ref !"

    if Ref is not None:
        assert PtsCross is not None, "PtsCross must be provided if 'Ref' is provided !"
        val = eval(Ref)
        assert type(val) is dict and all([dd in val.keys() for dd in ['vRef','vPts']]), Ref+" must be a dict with keys ['vPts','vRef'] !"
        assert type(val['vPts']) is np.ndarray and val['vPts'].ndim==2 and val['vPts'].shape[1]==NP, "Arg "+Ref+"['vPts'] must be a np.ndarray of float with shape (Nt,NP) !"
        assert type(val['vRef']) is np.ndarray and val['vRef'].ndim==2 and val['vRef'].dtype.name=='float64', "Arg "+Ref+"['vRef'] must be a 2-dim np.ndarray of floats !"

    if any([eval(ss+' is not None') for ss in vPts_add]):
        assert PtsCross is not None, "PtsCross must be provided if any vPts is provided !"
        for ss in vPts_add:
            val = eval(ss)
            if val is not None:
                assert type(val) is dict and all([dd in val.keys() for dd in ['vRef','vPts']]), "Ref must be provided if 'Ref' is provided !"
                assert type(val['vPts']) is np.ndarray and val['vPts'].ndim==2 and val['vPts'].shape[1]==NP, "Arg "+ss+"['vPts'] must be a np.ndarray of float with shape (Nt,NP) !"

    if MagAx is not None:
        assert type(MagAx) is np.ndarray and MagAx.ndim==2 and MagAx.shape[1]==2 and MagAx.dtype.name=='float64', "MagAx must be a 2-dim (Nt,2) np.ndarray of floats !"

    if Sep is not None:
        assert type(Sep) is list and all([type(sp) is np.ndarray and sp.ndim==2 and sp.shape[0]==2 and sp.dtype.name=='float64' for sp in Sep]), "sp must be a list of 2-dim (2,N) np.ndarray of floats !"

    if not t is None:
        assert type(t) is np.ndarray and t.ndim==1 and t.dtype.name=='float64', "Arg t must be a 1-dim np.ndarray of floats !"

    # Inputs below can be provided separately
    if not Type is None:
        assert Type in ['Tor','Lin'], "Arg Type must be in ['Tor','Lin'] !"
    if not Exp is None:
        assert Exp in tfd.AllowedExp, "Ar Exp must be in "+str(tfd.AllowedExp)+" !"
    Ints = [shot]
    if any([not aa is None for aa in Ints]):
        assert all([aa is None or type(aa) is int for aa in Ints]), "Args [shot] must be int !"
    strs = [SavePath]
    if any([not aa is None for aa in strs]):
        assert all([aa is None or type(aa) is str for aa in strs]), "Args [SavePath] must all be str !"
    if not dtime is None:
        assert type(dtime) is dtm.datetime, "Arg dtime must be a dtm.datetime !"
    bools = [dtimeIn]
    if any([not aa is None for aa in bools]):
        assert all([aa is None or type(aa) is bool for aa in bools]), " Args [dtimeIn] must all be bool !"














