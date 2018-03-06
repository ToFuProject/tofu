"""
This module is the geometrical part of the ToFu general package
It includes all functions and object classes necessary for tomography on Tokamaks
"""

import os
import warnings
import numpy as np
import datetime as dtm

# ToFu-specific
import tofu.pathfile as tfpf
import tofu.utils as utils
try:
    import tofu.geom._def as _def
    import tofu.geom._GG as _GG 
    import tofu.geom._comp as _comp
    import tofu.geom._plot as _plot
except Exception:
    from . import _def as _def
    from . import _GG as _GG
    from . import _comp as _comp
    from . import _plot as _plot

__all__ = ['Ves', 'Struct',
           'Rays','LOSCam1D','LOSCam2D']



"""
###############################################################################
###############################################################################
                        Ves class and functions
###############################################################################
"""



class Ves(object):
    """ A class defining a Linear or Toroidal vaccum vessel (i.e. a 2D polygon representing a cross-section and assumed to be linearly or toroidally invariant)

    A Ves object is mostly defined by a close 2D polygon, which can be understood as a poloidal cross-section in (R,Z) cylindrical coordinates if Type='Tor' (toroidal shape) or as a straight cross-section through a cylinder in (Y,Z) cartesian coordinates if Type='Lin' (linear shape).
    Attributes such as the surface, the angular volume (if Type='Tor') or the center of mass are automatically computed.
    The instance is identified thanks to an attribute Id (which is itself a tofu.ID class object) which contains informations on the specific instance (name, Type...).

    Parameters
    ----------
    Id :            str / tfpf.ID
        A name string or a pre-built tfpf.ID class to be used to identify this particular instance, if a string is provided, it is fed to tfpf.ID()
    Poly :          np.ndarray
        An array (2,N) or (N,2) defining the contour of the vacuum vessel in a cross-section, if not closed, will be closed automatically
    Type :          str
        Flag indicating whether the vessel will be a torus ('Tor') or a linear device ('Lin')
    Lim :         list / np.ndarray
        Array or list of len=2 indicating the limits of the linear device volume on the x axis
    Sino_RefPt :    None / np.ndarray
        Array specifying a reference point for computing the sinogram (i.e. impact parameter), if None automatically set to the (surfacic) center of mass of the cross-section
    Sino_NP :       int
        Number of points in [0,2*pi] to be used to plot the vessel sinogram envelop
    Clock :         bool
        Flag indicating whether the input polygon should be made clockwise (True) or counter-clockwise (False)
    arrayorder:     str
        Flag indicating whether the attributes of type=np.ndarray (e.g.: Poly) should be made C-contiguous ('C') or Fortran-contiguous ('F')
    Exp :           None / str
        Flag indicating which experiment the object corresponds to, allowed values are in [None,'AUG','MISTRAL','JET','ITER','TCV','TS','Misc']
    shot :          None / int
        Shot number from which this Ves is usable (in case of change of geometry)
    SavePath :      None / str
        If provided, forces the default saving path of the object to the provided value

    Returns
    -------
    Ves :        Ves object
        The created Ves object, with all necessary computed attributes and methods

    """

    def __init__(self, Id=None, Poly=None, Type='Tor', Lim=None, Exp=None, shot=0,
                 Sino_RefPt=None, Sino_NP=_def.TorNP,
                 Clock=False, arrayorder='C', fromdict=None,
                 SavePath=os.path.abspath('./'),
                 SavePath_Include=tfpf.defInclude):
        self._Done = False
        if fromdict is None:
            tfpf._check_NotNone({'Clock':Clock,'arrayorder':arrayorder})
            _Ves_check_inputs(Clock=Clock, arrayorder=arrayorder)
            self._arrayorder = arrayorder
            self._Clock = Clock
            self._set_Id(Id, Type=Type, Exp=Exp, shot=shot, SavePath=SavePath,
                         SavePath_Include=SavePath_Include)
            self._set_geom(Poly, Lim=Lim, Clock=Clock, Sino_RefPt=Sino_RefPt, Sino_NP=Sino_NP)
            self._set_arrayorder(arrayorder)
        else:
            self._fromdict(fromdict)
        self._Done = True


    def _todict(self):
        out = {'Id':self.Id._todict(),
               'Multi':self._Multi,
               'geom':self.geom, 'sino':self.sino,
               'arrayorder':self._arrayorder}
        if self._Id.Cls=='Struct':
            out['mobile'] = self._mobile
        return out

    def _fromdict(self, fd):
        _Ves_check_fromdict(fd)
        self._Id = tfpf.ID(fromdict=fd['Id'])
        self._geom = fd['geom']
        self._Multi = fd['Multi']
        self._sino = fd['sino']
        self._set_arrayorder(fd['arrayorder'])
        if self._Id.Cls=='Struct':
            self._mobile = fd['mobile']

    @property
    def Id(self):
        """Return the tfpf.ID object of the vessel"""
        return self._Id
    @property
    def Type(self):
        """Return the type of vessel"""
        return self.Id.Type
    @property
    def geom(self):
        return self._geom
    @property
    def Poly(self):
        """Return the polygon defining the vessel cross-section"""
        return self.geom['Poly']
    @property
    def Lim(self):
        return self.geom['Lim']
    @property
    def sino(self):
        return self._sino


    def _check_inputs(self, Id=None, Poly=None, Type=None, Lim=None,
                      Sino_RefPt=None, Sino_NP=None, Clock=None,
                      arrayorder=None, Exp=None, shot=None, SavePath=None):
        _Ves_check_inputs(Id=Id, Poly=Poly, Type=Type, Lim=Lim,
                          Sino_RefPt=Sino_RefPt, Sino_NP=Sino_NP, Clock=Clock,
                          arrayorder=arrayorder, Exp=Exp, shot=shot,
                          SavePath=SavePath, Cls=self.Id.Cls)

    def _set_Id(self, Val, Type=None, Exp=None, shot=None,
                SavePath=os.path.abspath('./'),
                SavePath_Include=None):
        if self._Done:
            Out = tfpf._get_FromItself(self.Id,{'Type':Type, 'Exp':Exp, 'shot':shot, 'SavePath':SavePath})
            Type, Exp, shot, SavePath = Out['Type'], Out['Exp'], Out['shot'], Out['SavePath']
        tfpf._check_NotNone({'Id':Val})
        _Ves_check_inputs(Id=Val)
        if type(Val) is str:
            tfpf._check_NotNone({'Type':Type, 'Exp':Exp, 'shot':shot})
            _Ves_check_inputs(Type=Type, Exp=Exp, shot=shot, SavePath=SavePath)
            Val = tfpf.ID(self.__class__, Val, Type=Type, Exp=Exp, shot=shot,
                          SavePath=SavePath, Include=SavePath_Include)
        self._Id = Val

    def _set_arrayorder(self, arrayorder):
        tfpf._set_arrayorder(self, arrayorder)

    def _set_geom(self, Poly, Lim=None, Clock=False, Sino_RefPt=None, Sino_NP=_def.TorNP):
        if self._Done:
            Out = tfpf._get_FromItself(self, {'Lim':Lim, '_Clock':Clock})
            Lim, Clock = Out['Lim'], Out['_Clock']
        tfpf._check_NotNone({'Poly':Poly, 'Clock':Clock})
        self._check_inputs(Poly=Poly)
        out = _comp._Ves_set_Poly(np.array(Poly), self._arrayorder, self.Type, Lim=Lim, Clock=Clock)
        SS = ['Poly','NP','P1Max','P1Min','P2Max','P2Min','BaryP','BaryL',
              'Surf','BaryS','Lim','VolLin','BaryV','Vect','VIn']
        self._geom = dict([(SS[ii],out[ii]) for ii in range(0,len(SS))])
        self._Multi = out[-1]
        self.set_sino(Sino_RefPt, NP=Sino_NP)

    def set_sino(self, RefPt=None, NP=_def.TorNP):
        if self._Done:
            RefPt, NP = self.sino['RefPt'], self.sino['NP']
            tfpf._check_NotNone({'NP':NP})
        if RefPt is None:
            RefPt = self.geom['BaryS']
        RefPt = np.asarray(RefPt).flatten()
        EnvTheta, EnvMinMax = _GG.Sino_ImpactEnv(RefPt, self.Poly, NP=NP, Test=False)
        self._sino = {'RefPt':RefPt, 'NP':NP, 'EnvTheta':EnvTheta, 'EnvMinMax':EnvMinMax}

    def isInside(self, Pts, In='(X,Y,Z)'):
        """ Return an array of booleans indicating whether each point lies inside the Ves volume

        Tests for each point whether it lies inside the Ves object.
        The points coordinates can be provided in 2D or 3D, just specify which coordinate system is provided using the 'In' parameter.
        An array of boolean flags is returned.

        Parameters
        ----------
        Pts :   np.ndarray
            (2,N) or (3,N) array with the coordinates of the points to be tested
        In :    str
            Flag indicating the coordinate system in which the points are provided, e.g '(X,Y,Z)' or '(R,Z)'

        Returns
        -------
        ind :   np.ndarray
            Array of booleans of shape (N,), True if a point is inside the Ves volume

        """
        ind = _GG._Ves_isInside(Pts, self.Poly, Lim=self.geom['Lim'], VType=self.Type, In=In, Test=True)
        return ind


    def get_InsideConvexPoly(self, RelOff=_def.TorRelOff, ZLim='Def', Spline=True, Splprms=_def.TorSplprms, NP=_def.TorInsideNP, Plot=False, Test=True):
        """ Return a polygon that is a smaller and smoothed approximation of Ves.Poly, useful for excluding the divertor region in a Tokamak

        For some uses, it can be practical to approximate the polygon defining the Ves object (which can be non-convex, like with a divertor), by a simpler, sligthly smaller and convex polygon.
        This method provides a fast solution for computing such a proxy.

        Parameters
        ----------
        RelOff :    float
            Fraction by which an homothetic polygon should be reduced (1.-RelOff)*(Poly-BaryS)
        ZLim :      None / str / tuple
            Flag indicating what limits shall be put to the height of the polygon (used for excluding divertor)
        Spline :    bool
            Flag indiating whether the reduced and truncated polygon shall be smoothed by 2D b-spline curves
        Splprms :   list
            List of 3 parameters to be used for the smoothing [weights,smoothness,b-spline order], fed to scipy.interpolate.splprep()
        NP :        int
            Number of points to be used to define the smoothed polygon
        Plot :      bool
            Flag indicating whether the result shall be plotted for visual inspection
        Test :      bool
            Flag indicating whether the inputs should be tested for conformity

        Returns
        -------
        Poly :      np.ndarray
            (2,N) polygon resulting from homothetic transform, truncating and optional smoothing

        """
        return _comp._Ves_get_InsideConvexPoly(self.Poly, self.geom['P2Min'], self.geom['P2Max'], self.geom['BaryS'], RelOff=RelOff, ZLim=ZLim, Spline=Spline, Splprms=Splprms, NP=NP, Plot=Plot, Test=Test)

    def get_sampleEdge(self, dl, DS=None, dlMode='abs', DIn=0.):
        """ Sample the polygon edges

        Sample each segment of the 2D polygon
        Sampling can be limited to a subdomain defined by DS
        It is done with resolution dl
        """
        Pts, dlr, ind = _comp._Ves_get_sampleEdge(self.Poly, dl, DS=DS,
                                                  dLMode=dlMode, DIn=DIn,
                                                  VIn=self.geom['VIn'],
                                                  margin=1.e-9)
        return Pts, dlr, ind

    def get_sampleCross(self, dS, DS=None, dSMode='abs', ind=None):
        """ Mesh the 2D cross-section fraction defined by DS or ind, with resolution dS """
        Pts, dS, ind, dSr = _comp._Ves_get_sampleCross(self.Poly, self.geom['P1Min'][0], self.geom['P1Max'][0], self.geom['P2Min'][1], self.geom['P2Max'][1], dS, DS=DS, dSMode=dSMode, ind=ind, margin=1.e-9)
        return Pts, dS, ind, dSr

    def get_sampleS(self, dS, DS=None, dSMode='abs', ind=None, DIn=0., Out='(X,Y,Z)'):
        """ Mesh the surface fraction defined by DS or ind, with resolution dS and optional offset DIn

        Parameters
        ----------
        dS      :   float / list of 2 floats
            Desired resolution of the surfacic sample
                float   : same resolution for all directions of the sample
                list    : [dl,dXPhi] where:
                    dl      : resolution along the polygon contour in the cross-section
                    dXPhi   : resolution along the axis (toroidal direction if self.Id.Type=='Tor' or linear direction if self.Id.Type=='Lin')
        DS      :   None / list of 3 lists of 2 floats
            Limits of the domain in which the surfacic sample should be computed
                None : whole surface of the object
                list : [D1,D2,D3] where each Di is a len()=2 list of increasing floats marking the boundaries of the domain along coordinate i, with
                    [DR,DZ,DPhi]: if toroidal geometry (self.Id.Type=='Tor')
                    [DX,DY,DZ]  : if linear geometry (self.Id.Type=='Lin')
        dSMode  :   str
            Flag specifying whether the resoltion dS shall be understood as an absolute distance or as a fraction of the distance of each element
                'abs'   :   dS is an absolute distance
                'rel'   :   if dS=0.1, each segment of the polygon will be divided in 10, and the toroidal/linear length will also be divided in 10
        ind     :   None / np.ndarray of int
            If provided, then DS is ignored and the method computes the points of the sample corresponding to the provided indices
            Example (assuming S is a Ves or Struct object)
                > # We create a 5x5 cm2 sample of the whole surface
                > Pts, dS, ind, dSr = S.get_sample(0.05)
                > # Performing operations, saving only the indices of the points and not the points themselves (to save space)
                > ...
                > # Retrieving the points from their indices (requires the same resolution), here Ptsbis = Pts
                > Ptsbis, dSbis, indbis, dSrbis = S.get_sample(0.05, ind=ind)
        DIn     :   float
            Offset distance from the actual surface of the object, can be positive (towards the inside) or negative (towards the outside), useful to avoid numerical errors
        Out     :   str
            Flag indicating which coordinate systems the points should be returned, e.g. : '(X,Y,Z)' or '(R,Z,Phi)'

        Returns
        -------
        Pts :   np.ndarray / list of np.ndarrays
            The points coordinates as a (3,N) array. A list is returned if the Struct object has multiple entities in the toroidal / linear direction
        dS  :   np.ndarray / list of np.ndarrays
            The surface (in m^2) associated to each point
        ind :   np.ndarray / list of np.ndarrays
            The index of each points
        dSr :   np.ndarray / list of np.ndarrays
            The effective resolution in both directions after computation of the sample
        """
        Pts, dS, ind, dSr = _comp._Ves_get_sampleS(self.Poly, self.geom['P1Min'][0], self.geom['P1Max'][0], self.geom['P2Min'][1], self.geom['P2Max'][1], dS, DS=DS, dSMode=dSMode, ind=ind, DIn=DIn, VIn=self.geom['VIn'], VType=self.Type, VLim=self.Lim, Out=Out, margin=1.e-9)
        return Pts, dS, ind, dSr

    def get_sampleV(self, dV, DV=None, dVMode='abs', ind=None, Out='(X,Y,Z)'):
        """ Sample the volume defined by DV or ind, with resolution dV """
        Pts, dV, ind, dVr = _comp._Ves_get_sampleV(self.Poly, self.geom['P1Min'][0], self.geom['P1Max'][0], self.geom['P2Min'][1], self.geom['P2Max'][1], dV, DV=DV, dVMode=dVMode, ind=ind, VType=self.Type, VLim=self.Lim, Out=Out, margin=1.e-9)
        return Pts, dV, ind, dVr


    def plot(self, Lax=None, Proj='All', Elt='PIBsBvV',
             dP=None, dI=_def.TorId, dBs=_def.TorBsd, dBv=_def.TorBvd,
             dVect=_def.TorVind, dIHor=_def.TorITord, dBsHor=_def.TorBsTord,
             dBvHor=_def.TorBvTord, Lim=_def.Tor3DThetalim,Nstep=_def.TorNTheta,
             dLeg=_def.TorLegd, draw=True, a4=False, Test=True):
        """ Plot the polygon defining the vessel, in chosen projection

        Generic method for plotting the Ves object
        The projections to be plotted, the elements to plot can be specified
        Dictionaries of properties for each elements can also be specified
        If an ax is not provided a default one is created.

        Parameters
        ----------
        Lax :       list or plt.Axes
            The axes to be used for plotting
            Provide a list of 2 axes if Proj='All'
            If None a new figure with axes is created
        Proj :      str
            Flag specifying the kind of projection
                - 'Cross' : cross-section projection
                - 'Hor' : horizontal projection
                - 'All' : both
                - '3d' : a 3d matplotlib plot
        Elt  :      str
            Flag specifying which elements to plot
            Each capital letter corresponds to an element:
                * 'P': polygon
                * 'I': point used as a reference for impact parameters
                * 'Bs': (surfacic) center of mass
                * 'Bv': (volumic) center of mass for Tor type
                * 'V': vector pointing inward perpendicular to each segment
        dP :        dict / None
            Dict of properties for plotting the polygon
            Fed to plt.Axes.plot() or plt.plot_surface() if Proj='3d'
        dI :        dict / None
            Dict of properties for plotting point 'I' in Cross-section projection
        dIHor :     dict / None
            Dict of properties for plotting point 'I' in horizontal projection
        dBs :       dict / None
            Dict of properties for plotting point 'Bs' in Cross-section projection
        dBsHor :    dict / None
            Dict of properties for plotting point 'Bs' in horizontal projection
        dBv :       dict / None
            Dict of properties for plotting point 'Bv' in Cross-section projection
        dBvHor :    dict / None
            Dict of properties for plotting point 'Bv' in horizontal projection
        dVect :     dict / None
            Dict of properties for plotting point 'V' in cross-section projection
        dLeg :      dict / None
            Dict of properties for plotting the legend, fed to plt.legend()
            The legend is not plotted if None
        Lim :       list or tuple
            Array of a lower and upper limit of angle (rad.) or length for plotting the '3d' Proj
        Nstep :     int
            Number of points for sampling in ignorable coordinate (toroidal angle or length)
        draw :      bool
            Flag indicating whether the fig.canvas.draw() shall be called automatically
        a4 :        bool
            Flag indicating whether the figure should be plotted in a4 dimensions for printing
        Test :      bool
            Flag indicating whether the inputs should be tested for conformity

        Returns
        -------
        La          list / plt.Axes
            Handles of the axes used for plotting (list if several axes where used)

        """
        return _plot.Ves_plot(self, Lax=Lax, Proj=Proj, Elt=Elt,
                              Pdict=dP, Idict=dI, Bsdict=dBs, Bvdict=dBv,
                              Vdict=dVect, IdictHor=dIHor, BsdictHor=dBsHor,
                              BvdictHor=dBvHor, Lim=Lim, Nstep=Nstep,
                              LegDict=dLeg, draw=draw, a4=a4, Test=Test)


    def plot_sino(self, Proj='Cross', ax=None, Ang=_def.LOSImpAng, AngUnit=_def.LOSImpAngUnit, Sketch=True, Pdict=None, LegDict=_def.TorLegd, draw=True, a4=False, Test=True):
        """ Plot the sinogram of the vessel polygon, by computing its envelopp in a cross-section, can also plot a 3D version of it

        The envelop of the polygon is computed using self.Sino_RefPt as a reference point in projection space, and plotted using the provided dictionary of properties.
        Optionaly a smal sketch can be included illustrating how the angle and the impact parameters are defined (if the axes is not provided).

        Parameters
        ----------
        Proj :      str
            Flag indicating whether to plot a classic sinogram ('Cross') from the vessel cross-section (assuming 2D), or an extended 3D version '3d' of it with additional angle
        ax   :      None or plt.Axes
            The axes on which the plot should be done, if None a new figure and axes is created
        Ang  :      str
            Flag indicating which angle to use for the impact parameter, the angle of the line itself (xi) or of its impact parameter (theta)
        AngUnit :   str
            Flag for the angle units to be displayed, 'rad' for radians or 'deg' for degrees
        Sketch :    bool
            Flag indicating whether a small skecth showing the definitions of angles 'theta' and 'xi' should be included or not
        Pdict :     dict
            Dictionary of properties used for plotting the polygon envelopp, fed to plt.plot() if Proj='Cross' and to plt.plot_surface() if Proj='3d'
        LegDict :   None or dict
            Dictionary of properties used for plotting the legend, fed to plt.legend(), the legend is not plotted if None
        draw :      bool
            Flag indicating whether the fig.canvas.draw() shall be called automatically
        a4 :        bool
            Flag indicating whether the figure should be plotted in a4 dimensions for printing
        Test :      bool
            Flag indicating whether the inputs shall be tested for conformity

        Returns
        -------
        ax :        plt.Axes
            The axes used to plot

        """
        if Test:
            assert not self.sino['RefPt'] is None, 'The impact parameters must be computed first !'
            assert Proj in ['Cross','3d'], "Arg Proj must be in ['Cross','3d'] !"
        if Proj=='Cross':
            Pdict = _def.TorPFilld if Pdict is None else Pdict
            ax = _plot.Plot_Impact_PolProjPoly(self, ax=ax, Ang=Ang,
                                               AngUnit=AngUnit, Sketch=Sketch,
                                               Leg=self.Id.NameLTX, Pdict=Pdict,
                                               dLeg=LegDict,
                                               draw=False, a4=a4, Test=Test)
        else:
            Pdict = _def.TorP3DFilld if Pdict is None else Pdict
            ax = _plot.Plot_Impact_3DPoly(self, ax=ax, Ang=Ang, AngUnit=AngUnit,
                                          Pdict=Pdict, dLeg=LegDict,
                                          draw=False, a4=a4, Test=Test)
        if draw:
            ax.figure.canvas.draw()
        return ax

    def save(self, SaveName=None, Path=None,
             Mode='npz', compressed=False, Print=True):
        """ Save the object in folder Name, under SaveName

        Parameters
        ----------
        SaveName :  None / str
            The name to be used for the saved file
            If None (recommended) uses self.Id.SaveName
        Path :      None / str
            Path specifying where to save the file
            If None (recommended) uses self.Id.SavePath
        Mode :      str
            Flag specifying how to save the object:
                'npz': as a numpy array file (recommended)
        compressed :    bool
            Flag, used when Mode='npz', indicates whether to use:
                - False : np.savez
                - True :  np.savez_compressed (slower but smaller files)

        """
        tfpf.Save_Generic(self, SaveName=SaveName, Path=Path,
                          Mode=Mode, compressed=compressed, Print=Print)




def _Ves_check_inputs(Id=None, Poly=None, Type=None, Lim=None, Sino_RefPt=None,
                      Sino_NP=None, Clock=None, arrayorder=None, Exp=None,
                      shot=None, SavePath=None, Cls=None, fromdict=None):
    if not Id is None:
        assert type(Id) in [str,tfpf.ID], "Arg Id must be a str or a tfpf.ID object !"
    if not Poly is None:
        assert hasattr(Poly,'__iter__') and np.asarray(Poly).ndim==2 and 2 in np.asarray(Poly).shape, "Arg Poly must be a dict or an iterable with 2D coordinates of cross section poly !"
    bools = [Clock]
    if any([not aa is None for aa in bools]):
        assert all([aa is None or type(aa) is bool for aa in bools]), " Args [Clock] must all be bool !"
    if not arrayorder is None:
        assert arrayorder in ['C','F'], "Arg arrayorder must be in ['C','F'] !"
    if not Type is None:
        assert Type in ['Tor','Lin'], "Arg Type must be in ['Tor','Lin'] !"
    strs = [Exp,SavePath]
    if any([not aa is None for aa in strs]):
        assert all([aa is None or type(aa) is str for aa in strs]), "Args [Exp,SavePath] must all be str !"
    Iter2 = [Sino_RefPt]
    if any([not aa is None for aa in Iter2]):
        assert all([aa is None or (hasattr(aa,'__iter__') and np.asarray(aa).ndim==1 and np.asarray(aa).size==2) for aa in Iter2]), "Args [Lim,Sino_RefPt] must be an iterable with len()=2 !"
    assert Cls is None or (type(Cls) is str and Cls in ['Ves','Struct']), "Arg Cls must be a Ves or Struct !"
    if Cls is not None:
        if Cls=='Ves':
            assert Lim is None or (hasattr(Lim,'__iter__') and len(Lim)==2 and all([not hasattr(ll,'__iter__') for ll in Lim])), "Arg Lim must be an iterable of 2 scalars !"
        else:
            assert Lim is None or hasattr(Lim,'__iter__'), "Arg Lim must be an iterable !"
            if Lim is not None:
                assert (len(Lim)==2 and all([not hasattr(ll,'__iter__') for ll in Lim])) or all([hasattr(ll,'__iter__') and len(ll)==2 and all([not hasattr(lll,'__iter__') for lll in ll]) for ll in Lim]), "Arg Lim must be an iterable of 2 scalars or of iterables of 2 scalars !"
    Ints = [Sino_NP,shot]
    if any([not aa is None for aa in Ints]):
        assert all([aa is None or type(aa) is int for aa in Ints]), "Args [Sino_NP,shot] must be int !"


def _Ves_check_fromdict(fd):
    assert type(fd) is dict, "Arg from dict must be a dict !"
    k0 = {'Id':dict,'geom':dict,'sino':dict,'arrayorder':str, 'Multi':bool}
    keys = list(fd.keys())
    for kk in k0:
        assert kk in keys, "%s must be a key of fromdict"%kk
        typ = type(fd[kk])
        C = typ is k0[kk] or typ in k0[kk] or fd[kk] in k0[kk]
        assert C, "Wrong type of fromdict[%s]: %s"%(kk,str(typ))
    # Maybe more details ?
    #k0 = {'Poly':{'type':np.ndarray,'dim':2},
    #      'NP':{'type':int,'val':fd['geom']['Poly'].shape[1]-1},
    #      'P1Max':{'type':np.ndarray,'shape':(3,)},
    #      'P1Min':{'type':np.ndarray,'shape':(3,)},
    #      'P2Max':{'type':np.ndarray,'shape':(3,)},
    #      'P2Min':{'type':np.ndarray,'shape':(3,)},
    #      'BaryP':{'type':np.ndarray,'shape':(3,)},
    #      'BaryL':{'type':np.ndarray,'shape':(3,)},
    #      'BaryS':{'type':np.ndarray,'shape':(3,)},
    #      'BaryV':{'type':np.ndarray,'shape':(3,)}} # To be finsihed ?





"""
###############################################################################
###############################################################################
                        Struct class and functions
###############################################################################
"""

class Struct(Ves):

    def __init__(self, Id=None, Poly=None, Type='Tor', Lim=None,
                 Sino_RefPt=None, Sino_NP=_def.TorNP,
                 Clock=False, arrayorder='C', fromdict=None,
                 Exp=None, shot=0,
                 SavePath=os.path.abspath('./'),
                 SavePath_Include=tfpf.defInclude,
                 mobile=False):
        assert type(mobile) is bool
        self._mobile = mobile
        Ves.__init__(self, Id, Poly, Type=Type, Lim=Lim,
                     Sino_RefPt=Sino_RefPt, Sino_NP=Sino_NP,
                     Clock=Clock, arrayorder=arrayorder, fromdict=fromdict,
                     Exp=Exp, shot=shot, SavePath=SavePath,
                     SavePath_Include=SavePath_Include)

    def move(self):
        """ To be overriden at object-level after instance creation

        To do so:
            1/ create the instance:
                >> S = tfg.Struct('test', poly, Exp='Test')
            2/ Define a moving function f taking the instance as first argument
                >> def f(self, Delta=1.):
                       Polynew = self.Poly
                       Polynew[0,:] = Polynew[0,:] + Delta
                       self._set_geom(Polynew, Lim=self.Lim)
            3/ Bound your custom function to the self.move() method
               using types.MethodType() found in the types module
                >> import types
                >> S.move = types.MethodType(f, S)

            See the following page for info and details on method-patching:
            https://tryolabs.com/blog/2013/07/05/run-time-method-patching-python/
        """
        print(self.move.__doc__)

    def get_sampleS(self, dS, DS=None, dSMode='abs', ind=None, DIn=0., Out='(X,Y,Z)', Ind=None):
        """ Mesh the surface fraction defined by DS or ind, with resolution dS and optional offset DIn

        Parameters
        ----------
        dS      :   float / list of 2 floats
            Desired resolution of the surfacic sample
                float   : same resolution for all directions of the sample
                list    : [dl,dXPhi] where:
                    dl      : resolution along the polygon contour in the cross-section
                    dXPhi   : resolution along the axis (toroidal direction if self.Id.Type=='Tor' or linear direction if self.Id.Type=='Lin')
        DS      :   None / list of 3 lists of 2 floats
            Limits of the domain in which the surfacic sample should be computed
                None : whole surface of the object
                list : [D1,D2,D3] where each Di is a len()=2 list of increasing floats marking the boundaries of the domain along coordinate i, with
                    [DR,DZ,DPhi]: if toroidal geometry (self.Id.Type=='Tor')
                    [DX,DY,DZ]  : if linear geometry (self.Id.Type=='Lin')
        dSMode  :   str
            Flag specifying whether the resoltion dS shall be understood as an absolute distance or as a fraction of the distance of each element
                'abs'   :   dS is an absolute distance
                'rel'   :   if dS=0.1, each segment of the polygon will be divided in 10, and the toroidal/linear length will also be divided in 10
        ind     :   None / np.ndarray of int
            If provided, then DS is ignored and the method computes the points of the sample corresponding to the provided indices
            Example (assuming S is a Ves or Struct object)
                > # We create a 5x5 cm2 sample of the whole surface
                > Pts, dS, ind, dSr = S.get_sample(0.05)
                > # Performing operations, saving only the indices of the points and not the points themselves (to save space)
                > ...
                > # Retrieving the points from their indices (requires the same resolution), here Ptsbis = Pts
                > Ptsbis, dSbis, indbis, dSrbis = S.get_sample(0.05, ind=ind)
        DIn     :   float
            Offset distance from the actual surface of the object, can be positive (towards the inside) or negative (towards the outside), useful to avoid numerical errors
        Out     :   str
            Flag indicating which coordinate systems the points should be returned, e.g. : '(X,Y,Z)' or '(R,Z,Phi)'
        Ind     :   None / iterable of ints
            Array of indices of the entities to be considered (in the case of Struct object with multiple entities in the toroidal / linear direction)

        Returns
        -------
        Pts :   np.ndarray / list of np.ndarrays
            The points coordinates as a (3,N) array. A list is returned if the Struct object has multiple entities in the toroidal / linear direction
        dS  :   np.ndarray / list of np.ndarrays
            The surface (in m^2) associated to each point
        ind :   np.ndarray / list of np.ndarrays
            The index of each points
        dSr :   np.ndarray / list of np.ndarrays
            The effective resolution in both directions after computation of the sample
        """
        Pts, dS, ind, dSr = _comp._Ves_get_sampleS(self.Poly, self.geom['P1Min'][0], self.geom['P1Max'][0], self.geom['P2Min'][1], self.geom['P2Max'][1], dS, DS=DS, dSMode=dSMode, ind=ind, DIn=DIn, VIn=self.geom['VIn'], VType=self.Type, VLim=self.Lim, Out=Out, margin=1.e-9, Multi=self._Multi, Ind=Ind)
        return Pts, dS, ind, dSr

    def get_sampleV(self, dV, DV=None, dVMode='abs', ind=None, Out='(X,Y,Z)'):
        raise AttributeError("Struct class cannot use the get_sampleV() method (only surface sampleing) !")



"""
###############################################################################
###############################################################################
                        Rays-derived classes and functions
###############################################################################
"""


class Rays(object):
    """ Parent class of rays (ray-tracing), LOS, LOSCam1D and LOSCam2D

    Focused on optimizing the computation time for many rays.

    Each ray is defined by a starting point (D) and a unit vector(u).
    If a vessel (Ves) and structural elements (LStruct) are provided,
    the intersection points are automatically computed.

    Methods for plootting, computing synthetic signal are provided.

    Parameters
    ----------
    Id :            str  / :class:`~tofu.pathfile.ID`
        A name string or a :class:`~tofu.pathfile.ID` to identify this instance,
        if a string is provided, it is fed to :class:`~tofu.pathfile.ID`
    Du :            iterable
        Iterable of len=2, containing 2 np.ndarrays represnting, for N rays:
            - Ds: a (3,N) array of the (X,Y,Z) coordinates of starting points
            - us: a (3,N) array of the (X,Y,Z) coordinates of the unit vectors
    Ves :           None / :class:`~tofu.geom.Ves`
        A :class:`~tofu.geom.Ves` instance to be associated to the rays
    LStruct:        None / :class:`~tofu.geom.Struct` / list
        A :class:`~tofu.geom.Struct` instance or list of such, for obstructions
    Sino_RefPt :    None / np.ndarray
        Iterable of len=2 with the coordinates of the sinogram reference point
            - (R,Z) coordinates if the vessel is of Type 'Tor'
            - (Y,Z) coordinates if the vessel is of Type 'Lin'
    Type :          None
        (not used in the current version)
    Exp        :    None / str
        Experiment to which the LOS belongs:
            - if both Exp and Ves are provided: Exp==Ves.Id.Exp
            - if Ves is provided but not Exp: Ves.Id.Exp is used
    Diag       :    None / str
        Diagnostic to which the LOS belongs
    shot       :    None / int
        Shot number from which this LOS is valid
    SavePath :      None / str
        If provided, default saving path of the object

    """

    def __init__(self, Id=None, Du=None, Ves=None, LStruct=None,
                 Sino_RefPt=None, fromdict=None,
                 Exp=None, Diag=None, shot=0, dchans=None,
                 SavePath=os.path.abspath('./'),
                 plotdebug=True):
        self._Done = False
        if fromdict is None:
            self._check_inputs(Id=Id, Du=Du, Ves=Ves, LStruct=LStruct,
                               Sino_RefPt=Sino_RefPt, Exp=Exp, Diag=Diag,
                               shot=shot, dchans=dchans, SavePath=SavePath)
            if Ves is not None:
                Exp = Ves.Id.Exp if Exp is None else Exp
            self._set_Id(Id, Exp=Exp, Diag=Diag, shot=shot, SavePath=SavePath)
            self._set_Ves(Ves, LStruct=LStruct, Du=Du, dchans=dchans,
                          plotdebug=plotdebug)
            self.set_sino(RefPt=Sino_RefPt)
        else:
            self._fromdict(fromdict)
        self._Done = True

    def _fromdict(self, fd):
        _Rays_check_fromdict(fd)
        self._Id = tfpf.ID(fromdict=fd['Id'])
        self._dchans = fd['dchans']
        if fd['Ves'] is None:
            self._Ves = None
        else:
            self._Ves = Ves(fromdict=fd['Ves'])
        if fd['LStruct'] is None:
            self._LStruct = None
        else:
            self._LStruct = [Struct(fromdict=ds) for ds in fd['LStruct']]
        self._geom = fd['geom']
        self._sino = fd['sino']

    def _todict(self):
        out = {'Id':self.Id._todict(),
               'dchans':self.dchans,
               'geom':self.geom, 'sino':self.sino}
        out['Ves'] = None if self.Ves is None else self.Ves._todict()
        if self.LStruct is None:
            out['LStruct'] = None
        else:
            out['LStruct'] = [ss._todict() for ss in self.LStruct]
        return out

    @property
    def Id(self):
        return self._Id
    @property
    def geom(self):
        return self._geom
    @property
    def nRays(self):
        return self.geom['nRays']
    @property
    def D(self):
        return self.geom['D']
    @property
    def u(self):
        return self.geom['u']
    @property
    def PIn(self):
        return self.geom['PIn']
    @property
    def POut(self):
        return self.geom['POut']
    @property
    def dchans(self):
        return self._dchans
    @property
    def Ves(self):
        return self._Ves
    @property
    def LStruct(self):
        return self._LStruct
    @property
    def sino(self):
        return self._sino

    def _check_inputs(self, Id=None, Du=None, Ves=None, LStruct=None,
                      Sino_RefPt=None, Exp=None, shot=None, Diag=None,
                      SavePath=None, ind=None,
                      dchans=None, fromdict=None):
        _Rays_check_inputs(Id=Id, Du=Du, Vess=Ves, LStruct=LStruct,
                          Sino_RefPt=Sino_RefPt, Exp=Exp, shot=shot, ind=ind,
                          Diag=Diag, SavePath=SavePath,
                          dchans=dchans, fromdict=fromdict)

    def _set_Id(self, Val,
                Exp=None, Diag=None, shot=None,
                SavePath=os.path.abspath('./')):
        dd = {'Exp':Exp, 'shot':shot, 'Diag':Diag, 'SavePath':SavePath}
        if self._Done:
            tfpf._get_FromItself(self.Id, dd)
        tfpf._check_NotNone({'Id':Val})
        self._check_inputs(Id=Val)
        if type(Val) is str:
            Val = tfpf.ID(self.__class__, Val, **dd)
        self._Id = Val

    def _set_Ves(self, Ves=None, LStruct=None, Du=None, dchans=None,
                 plotdebug=True):
        self._check_inputs(Ves=Ves, Exp=self.Id.Exp, LStruct=LStruct, Du=Du)
        LObj = []
        if not Ves is None:
            LObj.append(Ves.Id)
        if not LStruct is None:
            LStruct = [LStruct] if type(LStruct) is Struct else LStruct
            LObj += [ss.Id for ss in LStruct]
        if len(LObj)>0:
            self.Id.set_LObj(LObj)
        self._Ves = Ves
        self._LStruct = LStruct
        Du = Du if Du is not None else (self.D,self.u)
        self._set_geom(Du, dchans=dchans, plotdebug=plotdebug)

    def _set_geom(self, Du, dchans=None, plotdebug=True):
        tfpf._check_NotNone({'Du':Du})
        self._check_inputs(Du=Du, dchans=dchans)
        D, u = np.asarray(Du[0]), np.asarray(Du[1])
        if D.ndim==2:
            if D.shape[1]==3 and not D.shape[0]==3:
                D, u = D.T, u.T
        if D.ndim==1:
            D, u = D.reshape((3,1)), u.reshape((3,1))
        u = u/np.sqrt(np.sum(u**2,axis=0))
        D = np.ascontiguousarray(D)
        u = np.ascontiguousarray(u)
        nRays = D.shape[1]

        kPIn, kPOut = np.full((nRays,),np.nan), np.full((nRays,),np.nan)
        PIn, POut = np.full((3,nRays),np.nan), np.full((3,nRays),np.nan)
        VPerpIn, VPerpOut = np.full((3,nRays),np.nan), np.full((3,nRays),np.nan)
        IndIn, IndOut = np.full((nRays,),np.nan), np.full((nRays,),np.nan)
        if self.Ves is not None:
            if self.LStruct is not None:
                lSPoly = [ss.Poly for ss in self.LStruct]
                lSLim = [ss.Lim for ss in self.LStruct]
                lSVIn = [ss.geom['VIn'] for ss in self.LStruct]
            else:
                lSPoly, lSLim, lSVIn = None, None, None

            kargs = dict(RMin=None, Forbid=True, EpsUz=1.e-6, EpsVz=1.e-9,
                         EpsA=1.e-9, EpsB=1.e-9, EpsPlane=1.e-9, Test=True)
            out = _GG.LOS_Calc_PInOut_VesStruct(D, u, self.Ves.Poly,
                                                self.Ves.geom['VIn'],
                                                Lim=self.Ves.Lim, LSPoly=lSPoly,
                                                LSLim=lSLim, LSVIn=lSVIn,
                                                VType=self.Ves.Type, **kargs)
            PIn, POut, kPIn, kPOut, VPerpIn, VPerpOut, IndIn, IndOut = out
            ind = (np.isnan(kPOut) | np.isinf(kPOut)
                   | np.any(np.isnan(POut),axis=0))
            kPOut[ind] = np.nan
            if np.any(ind):
                warnings.warn("Some LOS have no visibility inside the vessel !")
                if plotdebug:
                    _plot._LOS_calc_InOutPolProj_Debug(self.Ves, D[:,ind], u[:,ind],
                                                       PIn[:,ind], POut[:,ind])
            ind = np.isnan(kPIn)
            PIn[:,ind], kPIn[ind] = D[:,ind], 0.

        PRMin, kRMin, RMin = _comp.LOS_PRMin(D, u, kPOut=kPOut, Eps=1.e-12)
        self._geom = {'D':D, 'u':u, 'nRays':nRays,
                      'PIn':PIn, 'POut':POut, 'kPIn':kPIn, 'kPOut':kPOut,
                      'VPerpIn':VPerpIn, 'VPerpOut':VPerpOut,
                      'IndIn':IndIn, 'IndOut':IndOut,
                      'PRMin':PRMin, 'kRMin':kRMin, 'RMin':RMin}

        # Get basics of 2D geometry
        if self.Id.Cls=='LOSCam2D':
            C = np.nanmean(D,axis=1)
            CD0 = D[:,:-1] - C[:,np.newaxis]
            CD1 = D[:,1:] - C[:,np.newaxis]
            cross = np.array([CD1[1,1:]*CD0[2,:-1]-CD1[2,1:]*CD0[1,:-1],
                              CD1[2,1:]*CD0[0,:-1]-CD1[0,1:]*CD0[2,:-1],
                              CD1[0,1:]*CD0[1,:-1]-CD1[1,1:]*CD0[0,:-1]])
            crossn2 = np.sum(cross**2,axis=0)
            if np.all(np.abs(crossn2)<1.e-12):
                msg = "Is %s really a 2D camera ? (LOS aligned?)"%self.Id.Name
                warning.warn(msg)
            cross = cross[:,np.nanargmax(crossn2)]
            cross = cross / np.linalg.norm(cross)
            nIn = cross if np.sum(cross*np.nanmean(u,axis=1))>0. else -cross
            nIn, e1, e2 = utils.get_nIne1e2(C, nIn=nIn, e1=D[:,1]-D[:,0])
            if np.abs(np.abs(nIn[2])-1.)>1.e-12:
                if np.abs(e1[2])>np.abs(e2[2]):
                    e1, e2 = e2, e1
            e2 = e2 if e2[2]>0. else -e2
            self._geom.update({'C':C, 'nIn':nIn, 'e1':e1, 'e2':e2})

        if dchans is None:
            self._dchans = dchans
        else:
            lK = list(dchans.keys())
            self._dchans = dict([(kk,np.asarray(dchans[kk]).ravel()) for kk in lK])

    def set_sino(self, RefPt=None):
        self._check_inputs(Sino_RefPt=RefPt)
        if RefPt is None and self.Ves is None:
            self._sino = None
        else:
            if RefPt is None:
                RefPt = self.Ves.sino['RefPt']
            if RefPt is None:
                if self.Ves.Type=='Tor':
                    RefPt = self.Ves.geom['BaryV']
                else:
                    RefPt = self.Ves.geom['BaryS']
            RefPt = np.asarray(RefPt).ravel()
            if self.Ves is not None:
                self._Ves.set_sino(RefPt)
                VType = self.Ves.Type
            else:
                VType = 'Lin'
            kMax = np.copy(self.geom['kPOut'])
            kMax[np.isnan(kMax)] = np.inf
            out = _GG.LOS_sino(self.D, self.u, RefPt, kMax,
                               Mode='LOS', VType=VType)
            Pt, kPt, r, Theta, p, theta, Phi = out
            self._sino = {'RefPt':RefPt, 'Pt':Pt, 'kPt':kPt, 'r':r,
                          'Theta':Theta, 'p':p, 'theta':theta, 'Phi':Phi}

    def select(self, key=None, val=None, touch=None, log='any', out=int):
        assert out in [int,bool]
        assert log in ['any','all','not']
        C = [key is None,touch is None]
        assert np.sum(C)>=1
        if np.sum(C)==2:
            ind = np.ones((self.nRays,),dtype=bool)
        else:
            if key is not None:
                assert type(key) is str and key in self._dchans.keys()
                ltypes = [str,int,float,np.int64,np.float64]
                C0 = type(val) in ltypes
                C1 = type(val) in [list,tuple,np.ndarray]
                assert C0 or C1
                if C0:
                    val = [val]
                else:
                    assert all([type(vv) in ltypes for vv in val])
                ind = np.vstack([self._dchans[key]==ii for ii in val])
                if log=='any':
                    ind = np.any(ind,axis=0)
                elif log=='all':
                    ind = np.all(ind,axis=0)
                else:
                    ind = ~np.any(ind,axis=0)
            elif touch is not None:
                VesOk, StructOk = self.Ves is not None, self.LStruct is not None
                StructNames = [ss.Id.Name for ss in self.LStruct] if StructOk else None
                ind = _comp.Rays_touch(VesOk, StructOk, self.geom['IndOut'],
                                       StructNames,touch=touch)
                ind = ~ind if log=='not' else ind
        if out is int:
            ind = ind.nonzero()[0]
        return ind

    def _get_plotL(self, Lplot='Tot', Proj='All', ind=None, multi=False):
        self._check_inputs(ind=ind)
        if ind is not None:
            ind = np.asarray(ind)
            if ind.dtype in [bool,np.bool_]:
                ind = ind.nonzero()[0]
        else:
            ind = np.arange(0,self.nRays)
        if len(ind)>0:
            Ds, us = self.D[:,ind], self.u[:,ind]
            if len(ind)==1:
                Ds, us = Ds.reshape((3,1)), us.reshape((3,1))
            kPIn, kPOut = self.geom['kPIn'][ind], self.geom['kPOut'][ind]
            kRMin = self.geom['kRMin'][ind]
            pts = _comp.LOS_CrossProj(self.Ves.Type, Ds, us, kPIn, kPOut,
                                      kRMin, Proj=Proj,Lplot=Lplot,multi=multi)
        else:
            pts = None
        return pts

    def get_sample(self, dl, dlMode='abs', DL=None, method='sum', ind=None):
        """ Return a linear sampling of the LOS

        The LOS is sampled into a series a points and segments lengths
        The resolution (segments length) is <= dl
        The sampling can be done according to different methods
        It is possible to sample only a subset of the LOS

        Parameters
        ----------
        dl:     float
            Desired resolution
        dlMode: str
            Flag indicating dl should be understood as:
                - 'abs':    an absolute distance in meters
                - 'rel':    a relative distance (fraction of the LOS length)
        DL:     None / iterable
            The fraction [L1;L2] of the LOS that should be sampled, where
            L1 and L2 are distances from the starting point of the LOS (LOS.D)
        method: str
            Flag indicating which to use for sampling:
                - 'sum':    the LOS is sampled into N segments of equal length,
                            where N is the smallest int such that:
                                * segment length <= resolution(dl,dlMode)
                            The points returned are the center of each segment
                - 'simps':  the LOS is sampled into N segments of equal length,
                            where N is the smallest int such that:
                                * segment length <= resolution(dl,dlMode)
                                * N is even
                            The points returned are the egdes of each segment
                - 'romb':   the LOS is sampled into N segments of equal length,
                            where N is the smallest int such that:
                                * segment length <= resolution(dl,dlMode)
                                * N = 2^k + 1
                            The points returned are the egdes of each segment

        Returns
        -------
        Pts:    np.ndarray
            A (3,NP) array of NP points along the LOS in (X,Y,Z) coordinates
        kPts:   np.ndarray
            A (NP,) array of the points distances from the LOS starting point
        dl:     float
            The effective resolution (<= dl input), as an absolute distance

        """
        self._check_inputs(ind=ind)
        # Preformat ind
        if ind is None:
            ind = np.arange(0,self.nRays)
        if np.asarray(ind).dtype is bool:
            ind = ind.nonzero()[0]
        # Preformat DL
        if DL is None:
            DL = np.array([self.geom['kPIn'][ind],self.geom['kPOut'][ind]])
        elif np.asarray(DL).size==2:
            DL = np.tile(np.asarray(DL).ravel(),(len(ind),1)).T
        DL = np.ascontiguousarray(DL).astype(float)
        assert type(DL) is np.ndarray and DL.ndim==2
        assert DL.shape==(2,len(ind)), "Arg DL has wrong shape !"
        ii = DL[0,:]<self.geom['kPIn'][ind]
        DL[0,ii] = self.geom['kPIn'][ind][ii]
        ii = DL[0,:]>=self.geom['kPOut'][ind]
        DL[0,ii] = self.geom['kPOut'][ind][ii]
        ii = DL[1,:]>self.geom['kPOut'][ind]
        DL[1,ii] = self.geom['kPOut'][ind][ii]
        ii = DL[1,:]<=self.geom['kPIn'][ind]
        DL[1,ii] = self.geom['kPIn'][ind][ii]
        # Preformat Ds, us
        Ds, us = self.D[:,ind], self.u[:,ind]
        if len(ind)==1:
            Ds, us = Ds.reshape((3,1)), us.reshape((3,1))
        Ds, us = np.ascontiguousarray(Ds), np.ascontiguousarray(us)
        # Launch    # NB : find a way to exclude cases with DL[0,:]>=DL[1,:] !!
        Pts, kPts, dlr = _GG.LOS_get_sample(Ds, us, dl, DL,
                                            dLMode=dlMode, method=method)
        return Pts, kPts, dlr


    def calc_signal(self, ff, t=None, Ani=None, fkwdargs={},
                    dl=0.005, DL=None, dlMode='abs', method='sum',
                    ind=None, out=object, plot=True, plotmethod='imshow',
                    fs=None, Warn=True):
        """ Return the line-integrated emissivity

        Beware that it is only a line-integral !
        There is no multiplication by an Etendue
        (which cannot be computed for a LOS object, because it depends on the
        surfaces and respective positions of the detector and its apertures,
        which are not provided for a LOS object).

        Hence, if the emissivity is provided in W/m3 (resp. W/m3/sr),
        the method returns W/m2 (resp. W/m2/sr)
        The line is sampled using :meth:`~tofu.geom.LOS.get_sample`,

        The integral can be computed using three different methods:
            - 'sum':    A numpy.sum() on the local values (x segments lengths)
            - 'simps':  using :meth:`scipy.integrate.simps`
            - 'romb':   using :meth:`scipy.integrate.romb`

        Except ff, arguments common to :meth:`~tofu.geom.LOS.get_sample`

        Parameters
        ----------
        ff :    callable
            The user-provided

        """
        if Warn:
            warnings.warn("! CAUTION : returns W/m^2 (no Etendue, see help) !")
        self._check_inputs(ind=ind)
        # Preformat ind
        if ind is None:
            ind = np.arange(0,self.nRays)
        if np.asarray(ind).dtype is bool:
            ind = ind.nonzero()[0]
        # Preformat DL
        if DL is None:
            DL = np.array([self.geom['kPIn'][ind],self.geom['kPOut'][ind]])
        elif np.asarray(DL).size==2:
            DL = np.tile(np.asarray(DL).ravel(),(len(ind),1)).T
        DL = np.ascontiguousarray(DL).astype(float)
        assert type(DL) is np.ndarray and DL.ndim==2
        assert DL.shape==(2,len(ind)), "Arg DL has wrong shape !"
        ii = DL[0,:]<self.geom['kPIn'][ind]
        DL[0,ii] = self.geom['kPIn'][ind][ii]
        ii = DL[0,:]>=self.geom['kPOut'][ind]
        DL[0,ii] = self.geom['kPOut'][ind][ii]
        ii = DL[1,:]>self.geom['kPOut'][ind]
        DL[1,ii] = self.geom['kPOut'][ind][ii]
        ii = DL[1,:]<=self.geom['kPIn'][ind]
        DL[1,ii] = self.geom['kPIn'][ind][ii]
        # Preformat Ds, us
        Ds, us = self.D[:,ind], self.u[:,ind]
        if len(ind)==1:
            Ds, us = Ds.reshape((3,1)), us.reshape((3,1))
        if t is None or len(t)==1:
            sig = np.full((Ds.shape[1],),np.nan)
        else:
            sig = np.full((len(t),Ds.shape[1]),np.nan)
        indok = ~(np.any(np.isnan(DL),axis=0) | np.any(np.isinf(DL),axis=0)
                  | ((DL[1,:]-DL[0,:])<=0.))
        if np.any(indok):
            Ds, us, DL = Ds[:,indok], us[:,indok], DL[:,indok]
            if indok.sum()==1:
                Ds, us = Ds.reshape((3,1)), us.reshape((3,1))
                DL = DL.reshape((2,1))
            Ds, us = np.ascontiguousarray(Ds), np.ascontiguousarray(us)
            DL = np.ascontiguousarray(DL)
            # Launch    # NB : find a way to exclude cases with DL[0,:]>=DL[1,:] !!
            # Exclude Rays not seeing the plasma
            s = _GG.LOS_calc_signal(ff, Ds, us, dl, DL,
                                    dLMode=dlMode, method=method,
                                    t=t, Ani=Ani, fkwdargs=fkwdargs, Test=True)
            if t is None or len(t)==1:
                sig[indok] = s
            else:
                sig[:,indok] = s

        if plot or out is object:
            assert '1D' in self.Id.Cls or '2D' in self.Id.Cls, "Set Cam type!!"
            import tofu.data as tfd
            if '1D' in self.Id.Cls:
                osig = tfd.Data1D(data=sig, t=t, LCam=self, Id=self.Id.Name,
                                  Exp=self.Id.Exp, Diag=self.Id.Diag)
            else:
                osig = tfd.Data2D(data=sig, t=t, LCam=self, Id=self.Id.Name,
                                  Exp=self.Id.Exp, Diag=self.Id.Diag)
            if plot:
                dax, KH = osig.plot(fs=fs, plotmethod=plotmethod)
            if out is object:
                sig = osig
        return sig

    def plot(self, Lax=None, Proj='All', Lplot=_def.LOSLplot, Elt='LDIORP',
             EltVes='', EltStruct='', Leg='', dL=None, dPtD=_def.LOSMd,
             dPtI=_def.LOSMd, dPtO=_def.LOSMd, dPtR=_def.LOSMd,
             dPtP=_def.LOSMd, dLeg=_def.TorLegd, dVes=_def.Vesdict,
             dStruct=_def.Structdict,
             multi=False, ind=None, draw=True, a4=False, Test=True):
        """ Plot the Rays / LOS, in the chosen projection(s)

        Optionnally also plot associated :class:`~tofu.geom.Ves` and Struct
        The plot can also include:
            - special points
            - the unit directing vector

        Parameters
        ----------
        Lax :       list / plt.Axes
            The axes for plotting (list of 2 axes if Proj='All')
            If None a new figure with new axes is created
        Proj :      str
            Flag specifying the kind of projection:
                - 'Cross' : cross-section
                - 'Hor' : horizontal
                - 'All' : both cross-section and horizontal (on 2 axes)
                - '3d' : a (matplotlib) 3d plot
        Elt :       str
            Flag specifying which elements to plot
            Each capital letter corresponds to an element:
                * 'L': LOS
                * 'D': Starting point of the LOS
                * 'I': Input point (i.e.: where the LOS enters the Vessel)
                * 'O': Output point (i.e.: where the LOS exits the Vessel)
                * 'R': Point of minimal major radius R (only if Ves.Type='Tor')
                * 'P': Point of used for impact parameter (i.e.: with minimal
                        distance to reference point Sino_RefPt)
        Lplot :     str
            Flag specifying the length to plot:
                - 'Tot': total length, from starting point (D) to output point
                - 'In' : only the in-vessel fraction (from input to output)
        EltVes :    str
            Flag for Ves elements to plot (:meth:`~tofu.geom.Ves.plot`)
        EltStruct : str
            Flag for Struct elements to plot (:meth:`~tofu.geom.Struct.plot`)
        Leg :       str
            Legend, if Leg='' the LOS name is used
        dL :     dict / None
            Dictionary of properties for plotting the lines
            Fed to plt.Axes.plot(), set to default if None
        dPtD :      dict
            Dictionary of properties for plotting point 'D'
        dPtI :      dict
            Dictionary of properties for plotting point 'I'
        dPtO :      dict
            Dictionary of properties for plotting point 'O'
        dPtR :      dict
            Dictionary of properties for plotting point 'R'
        dPtP :      dict
            Dictionary of properties for plotting point 'P'
        dLeg :      dict or None
            Dictionary of properties for plotting the legend
            Fed to plt.legend(), the legend is not plotted if None
        dVes :      dict
            Dictionary of kwdargs to fed to :meth:`~tofu.geom.Ves.plot`
            And 'EltVes' is used instead of 'Elt'
        dStruct:    dict
            Dictionary of kwdargs to fed to :meth:`~tofu.geom.Struct.plot`
            And 'EltStruct' is used instead of 'Elt'
        draw :      bool
            Flag indicating whether fig.canvas.draw() shall be called
        a4 :        bool
            Flag indicating whether to plot the figure in a4 dimensions
        Test :      bool
        a4 :        bool
            Flag indicating whether to plot the figure in a4 dimensions
        Test :      bool
        a4 :        bool
            Flag indicating whether to plot the figure in a4 dimensions
        Test :      bool
        a4 :        bool
            Flag indicating whether to plot the figure in a4 dimensions
        Test :      bool
        Test :      bool
            Flag indicating whether the inputs should be tested for conformity

        Returns
        -------
        La :        list / plt.Axes
            Handles of the axes used for plotting (list if Proj='All')

        """

        return _plot.Rays_plot(self, Lax=Lax, Proj=Proj, Lplot=Lplot, Elt=Elt,
                               EltVes=EltVes, EltStruct=EltStruct, Leg=Leg,
                               dL=dL, dPtD=dPtD, dPtI=dPtI, dPtO=dPtO, dPtR=dPtR,
                               dPtP=dPtP, dLeg=dLeg, dVes=dVes, dStruct=dStruct,
                               multi=multi, ind=ind, draw=draw, a4=a4, Test=Test)


    def plot_sino(self, Proj='Cross', ax=None, Elt=_def.LOSImpElt, Sketch=True,
                  Ang=_def.LOSImpAng, AngUnit=_def.LOSImpAngUnit, Leg=None,
                  dL=_def.LOSMImpd, dVes=_def.TorPFilld, dLeg=_def.TorLegd,
                  ind=None, multi=False, draw=True, a4=False, Test=True):
        """ Plot the LOS in projection space (sinogram)

        Plot the Rays in projection space (cf. sinograms) as points.
        Can also optionnally plot the associated :class:`~tofu.geom.Ves`

        Can plot the conventional projection-space (in 2D in a cross-section),
        or a 3D extrapolation of it, where the third coordinate is provided by
        the angle that the LOS makes with the cross-section plane
        (useful in case of multiple LOS with a partially tangential view)

        Parameters
        ----------
        Proj :      str
            Flag indicating whether to plot:
                - 'Cross':  a classic sinogram (vessel cross-section)
                - '3d': an extended 3D version ('3d'), with an additional angle
        ax :        None / plt.Axes
            The axes on which to plot, if None a new figure is created
        Elt :       str
            Flag indicating which elements to plot (one per capital letter):
                * 'L': LOS
                * 'V': Vessel
        Ang  :      str
            Flag indicating which angle to use for the impact parameter:
                - 'xi': the angle of the line itself
                - 'theta': its impact parameter (theta)
        AngUnit :   str
            Flag for the angle units to be displayed:
                - 'rad': for radians
                - 'deg': for degrees
        Sketch :    bool
            Flag indicating whether to plot a skecth with angles definitions
        dL :        dict
            Dictionary of properties for plotting the Rays points
        dV :        dict
            Dictionary of properties for plotting the vessel envelopp
        dLeg :      None / dict
            Dictionary of properties for plotting the legend
            The legend is not plotted if None
        draw :      bool
            Flag indicating whether to draw the figure
        a4 :        bool
            Flag indicating whether the figure should be a4
        Test :      bool
            Flag indicating whether the inputs shall be tested for conformity

        Returns
        -------
        ax :        plt.Axes
            The axes used to plot

        """
        assert self.sino is not None, "The sinogram ref. point is not set !"
        return _plot.GLOS_plot_Sino(self, Proj=Proj, ax=ax, Elt=Elt, Leg=Leg,
                                    Sketch=Sketch, Ang=Ang, AngUnit=AngUnit,
                                    dL=dL, dVes=dVes, dLeg=dLeg,
                                    ind=ind, draw=draw, a4=a4, Test=Test)

    def plot_touch(self, key=None, invert=None,
                   lcol=['k','r','b','g','y','m','c']):
        assert self.Id.Cls in ['LOSCam1D','LOSCam2D'], "Specify camera type !"
        assert self.Ves is not None, "self.Ves should not be None !"
        out = _plot.Rays_plot_touch(self, key=key, invert=invert, lcol=lcol)
        return out

    def save(self, SaveName=None, Path=None,
             Mode='npz', compressed=False, Print=True):
        """ Save the object in folder Name, under SaveName

        Parameters
        ----------
        SaveName :  None / str
            The name to be used for the saved file
            If None (recommended) uses self.Id.SaveName
        Path :      None / str
            Path specifying where to save the file
            If None (recommended) uses self.Id.SavePath
        Mode :      str
            Flag specifying how to save the object:
                'npz': as a numpy array file (recommended)
        compressed :    bool
            Flag, used when Mode='npz', indicates whether to use:
                - False : np.savez
                - True :  np.savez_compressed (slower but smaller files)

        """
        tfpf.Save_Generic(self, SaveName=SaveName, Path=Path,
                          Mode=Mode, compressed=compressed, Print=Print)







def _Rays_check_inputs(Id=None, Du=None, Vess=None, LStruct=None,
                       Sino_RefPt=None, Exp=None, shot=None, Diag=None,
                       SavePath=None, Calc=None, ind=None,
                       dchans=None, fromdict=None):
    if not Id is None:
        assert type(Id) in [str,tfpf.ID], "Arg Id must be a str or a tfpf.ID !"
    if not Du is None:
        C0 = hasattr(Du,'__iter__') and len(Du)==2
        C1 = 3 in np.asarray(Du[0]).shape and np.asarray(Du[0]).ndim in [1,2]
        C2 = 3 in np.asarray(Du[1]).shape and np.asarray(Du[1]).ndim in [1,2]
        C3 = np.asarray(Du[0]).shape==np.asarray(Du[1]).shape
        assert C0, "Arg Du must be an iterable of len()=2 !"
        assert C1, "Du[0] must contain 3D coordinates of all starting points !"
        assert C2, "Du[1] must contain 3D coordinates of all unit vectors !"
        assert C3, "Du[0] and Du[1] must be of same shape !"
    if not Vess is None:
        assert type(Vess) is Ves, "Arg Ves must be a Ves instance !"
        if Exp is not None and Vess.Id.Exp is not None:
            assert Exp==Vess.Id.Exp, "Arg Exp must be the same as Ves.Id.Exp !"
    if LStruct is not None:
        assert type(LStruct) in [list,Struct], "LStruct = %s !"%(type(LStruct))
        if type(LStruct) is list:
            for ss in LStruct:
                assert type(ss) is Struct, "LStruct = list of Struct !"
                if Exp is not None and ss.Id.Exp is not None:
                    assert Exp==ss.Id.Exp, "Struct elements for a different Exp"
        else:
            if Exp is not None and LStruct.Id.Exp is not None:
                assert Exp==LStruct.Id.Exp, "Struct element for a different Exp"
    bools = [Calc]
    if any([not aa is None for aa in bools]):
        C = all([aa is None or type(aa) is bool for aa in bools])
        assert C, " Args [Calc] must all be bool !"
    strs = [Exp,Diag,SavePath]
    if any([not aa is None for aa in strs]):
        C = all([aa is None or type(aa) is str for aa in strs])
        assert C, "Args [Exp,Diag,SavePath] must all be str !"
    Iter2 = [Sino_RefPt]
    for aa in Iter2:
        if aa is not None:
            C0 = np.asarray(aa).shape==(2,)
            assert C0, "Args [Sino_RefPt] must be an iterable with len()=2 !"
    Ints = [shot]
    for aa in Ints:
        if aa is not None:
            assert type(aa) is int, "Args [shot] must be int !"
    if ind is not None:
        assert np.asarray(ind).ndim==1
        assert np.asarray(ind).dtype in [bool,np.int64]
    if dchans is not None:
        assert type(dchans) is dict
        if Du is not None:
            nch = Du[0].shape[1]
            assert all([len(dchans[kk])==nch for kk in dchans.keys()])



def _Rays_check_fromdict(fd):
    assert type(fd) is dict, "Arg from dict must be a dict !"
    k0 = {'Id':dict,'geom':dict,'sino':dict,
          'dchans':[None,dict],
          'Ves':[None,dict], 'LStruct':[None,list]}
    keys = list(fd.keys())
    for kk in k0:
        assert kk in keys, "%s must be a key of fromdict"%kk
        typ = type(fd[kk])
        C = typ is k0[kk] or typ in k0[kk] or fd[kk] in k0[kk]
        assert C, "Wrong type of fromdict[%s]: %s"%(kk,str(typ))







class LOSCam1D(Rays):
    def __init__(self, Id=None, Du=None, Ves=None, LStruct=None,
                 Sino_RefPt=None, fromdict=None,
                 Exp=None, Diag=None, shot=0,
                 dchans=None, SavePath=os.path.abspath('./'),
                 plotdebug=True):
        Rays.__init__(self, Id=Id, Du=Du, Ves=Ves, LStruct=LStruct,
                 Sino_RefPt=Sino_RefPt, fromdict=fromdict,
                 Exp=Exp, Diag=Diag, shot=shot, plotdebug=plotdebug,
                 dchans=dchans, SavePath=SavePath)

class LOSCam2D(Rays):
    def __init__(self, Id=None, Du=None, Ves=None, LStruct=None,
                 Sino_RefPt=None, fromdict=None,
                 Exp=None, Diag=None, shot=0, X12=None,
                 dchans=None, SavePath=os.path.abspath('./'),
                 plotdebug=True):
        Rays.__init__(self, Id=Id, Du=Du, Ves=Ves, LStruct=LStruct,
                 Sino_RefPt=Sino_RefPt, fromdict=fromdict,
                 Exp=Exp, Diag=Diag, shot=shot, plotdebug=plotdebug,
                 dchans=dchans, SavePath=SavePath)
        self.set_X12(X12)

    def set_e12(self, e1=None, e2=None):
        assert e1 is None or (hasattr(e1,'__iter__') and len(e1)==3)
        assert e2 is None or (hasattr(e2,'__iter__') and len(e2)==3)
        if e1 is None:
            e1 = self._geom['e1']
        else:
            e1 = np.asarray(e1).astype(float).ravel()
        e1 = e1 / np.linalg.norm(e1)
        if e2 is None:
            e2 = self._geom['e2']
        else:
            e2 = np.asarray(e1).astype(float).ravel()
        e2 = e2 / np.linalg.norm(e2)
        assert np.abs(np.sum(e1*self._geom['nIn']))<1.e-12
        assert np.abs(np.sum(e2*self._geom['nIn']))<1.e-12
        assert np.abs(np.sum(e1*e2))<1.e-12
        self._geom['e1'] = e1
        self._geom['e2'] = e2

    def set_X12(self, X12=None):
        if X12 is not None:
            X12 = np.asarray(X12)
            assert X12.shape==(2,self.Ref['nch'])
        self._X12 = X12

    def get_X12(self, out='1d'):
        if self._X12 is None:
            Ds = self.D
            C = np.mean(Ds,axis=1)
            X12 = Ds-C[:,np.newaxis]
            X12 = np.array([np.sum(X12*self.geom['e1'][:,np.newaxis],axis=0),
                            np.sum(X12*self.geom['e2'][:,np.newaxis],axis=0)])
        else:
            X12 = self._X12
        if X12 is None or out.lower()=='1d':
            DX12 = None
        else:
            x1u, x2u, ind, DX12 = utils.get_X12fromflat(X12)
            if out.lower()=='2d':
                X12 = [x1u, x2u, ind]
        return X12, DX12







    """ Return the indices or instances of all LOS matching criteria

    The selection can be done according to 2 different mechanisms

    Mechanism (1): provide the value (Val) a criterion (Crit) should match
    The criteria are typically attributes of :class:`~tofu.pathfile.ID`
    (i.e.: name, or user-defined attributes like the camera head...)

    Mechanism (2): (used if Val=None)
    Provide a str expression (or a list of such) to be fed to eval()
    Used to check on quantitative criteria.
        - PreExp: placed before the criterion value (e.g.: 'not ' or '<=')
        - PostExp: placed after the criterion value
        - you can use both

    Other parameters are used to specify logical operators for the selection
    (match any or all the criterion...) and the type of output.

    Parameters
    ----------
    Crit :      str
        Flag indicating which criterion to use for discrimination
        Can be set to:
            - any attribute of :class:`~tofu.pathfile.ID`
        A str (or list of such) expression to be fed to eval()
        Placed after the criterion value
        Used for selection mechanism (2)
    Log :       str
        Flag indicating whether the criterion shall match:
            - 'all': all provided values
            - 'any': at least one of them
    InOut :     str
        Flag indicating whether the returned indices are:
            - 'In': the ones matching the criterion
            - 'Out': the ones not matching it
    Out :       type / str
        Flag indicating in which form to return the result:
            - int: as an array of integer indices
            - bool: as an array of boolean indices
            - 'Name': as a list of names
            - 'LOS': as a list of :class:`~tofu.geom.LOS` instances

    Returns
    -------
    ind :       list / np.ndarray
        The computed output, of nature defined by parameter Out

    Examples
    --------
    >>> import tofu.geom as tfg
    >>> VPoly, VLim = [[0.,1.,1.,0.],[0.,0.,1.,1.]], [-1.,1.]
    >>> V = tfg.Ves('ves', VPoly, Lim=VLim, Type='Lin', Exp='Misc', shot=0)
    >>> Du1 = ([0.,-0.1,-0.1],[0.,1.,1.])
    >>> Du2 = ([0.,-0.1,-0.1],[0.,0.5,1.])
    >>> Du3 = ([0.,-0.1,-0.1],[0.,1.,0.5])
    >>> l1 = tfg.LOS('l1', Du1, Ves=V, Exp='Misc', Diag='A', shot=0)
    >>> l2 = tfg.LOS('l2', Du2, Ves=V, Exp='Misc', Diag='A', shot=1)
    >>> l3 = tfg.LOS('l3', Du3, Ves=V, Exp='Misc', Diag='B', shot=1)
    >>> gl = tfg.GLOS('gl', [l1,l2,l3])
    >>> Arg1 = dict(Val=['l1','l3'],Log='any',Out='LOS')
    >>> Arg2 = dict(Val=['l1','l3'],Log='any',InOut='Out',Out=int)
    >>> Arg3 = dict(Crit='Diag', Val='A', Out='Name')
    >>> Arg4 = dict(Crit='shot', PostExp='>=1')
    >>> gl.select(**Arg1)
    [l1,l3]
    >>> gl.select(**Arg2)
    array([1])
    >>> gl.select(**Arg3)
    ['l1','l2']
    >>> gl.select(**Arg4)
    array([False, True, True], dtype=bool)

    """
