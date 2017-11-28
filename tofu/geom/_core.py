"""
This module is the geometrical part of the ToFu general package
It includes all functions and object classes necessary for tomography on Tokamaks
"""

import warnings
import numpy as np
import datetime as dtm

# ToFu-specific
import tofu.pathfile as tfpf
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

__all__ = ['Ves', 'Struct', 'LOS']



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

    def __init__(self, Id, Poly, Type='Tor', Lim=None, Sino_RefPt=None, Sino_NP=_def.TorNP, Clock=False, arrayorder='C', Exp=None, shot=0, SavePath=None, SavePath_Include=_def.SavePath_Include, Cls='Ves'):
        self._Done = False
        tfpf._check_NotNone({'Clock':Clock,'arrayorder':arrayorder})
        _Ves_check_inputs(Clock=Clock, arrayorder=arrayorder)
        self._arrayorder = arrayorder
        self._Clock = Clock
        self._set_Id(Id, Type=Type, Exp=Exp, shot=shot, SavePath=SavePath, SavePath_Include=SavePath_Include, Cls=Cls)
        self._set_geom(Poly, Lim=Lim, Clock=Clock, Sino_RefPt=Sino_RefPt, Sino_NP=Sino_NP)
        self._set_arrayorder(arrayorder)
        self._Done = True


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


    def _check_inputs(self, Id=None, Poly=None, Type=None, Lim=None, Sino_RefPt=None, Sino_NP=None, Clock=None, arrayorder=None, Exp=None, shot=None, SavePath=None):
        _Ves_check_inputs(Id=Id, Poly=Poly, Type=Type, Lim=Lim, Sino_RefPt=Sino_RefPt, Sino_NP=Sino_NP, Clock=Clock, arrayorder=arrayorder, Exp=Exp, shot=shot, SavePath=SavePath, Cls=self.Id.Cls)

    def _set_Id(self, Val, Type=None, Exp=None, shot=None, SavePath=None, SavePath_Include=None, Cls='Ves'):
        if self._Done:
            Out = tfpf._get_FromItself(self.Id,{'Type':Type, 'Exp':Exp, 'shot':shot, 'SavePath':SavePath})
            Type, Exp, shot, SavePath = Out['Type'], Out['Exp'], Out['shot'], Out['SavePath']
        tfpf._check_NotNone({'Id':Val,'Cls':Cls})
        _Ves_check_inputs(Id=Val, Cls=Cls)
        if type(Val) is str:
            tfpf._check_NotNone({'Type':Type, 'Exp':Exp, 'shot':shot})
            _Ves_check_inputs(Type=Type, Exp=Exp, shot=shot, SavePath=SavePath)
            Val = tfpf.ID(Cls, Val, Type=Type, Exp=Exp, shot=shot, SavePath=SavePath, Include=SavePath_Include)
        self._Id = Val

    def _set_arrayorder(self, arrayorder):
        tfpf._set_arrayorder(self, arrayorder)

    def _set_geom(self, Poly, Lim=None, Clock=False, Sino_RefPt=None, Sino_NP=_def.TorNP):
        if self._Done:
            Out = tfpf._get_FromItself(self, {'Lim':Lim, '_Clock':Clock})
            Lim, Clock = Out['Lim'], Out['Clock']
        tfpf._check_NotNone({'Poly':Poly, 'Clock':Clock})
        out = _comp._Ves_set_Poly(Poly, self._arrayorder, self.Type, Lim=Lim, Clock=Clock)
        SS = ['Poly','NP','P1Max','P1Min','P2Max','P2Min','BaryP','BaryL','Surf','BaryS','Lim','VolLin','BaryV','Vect','VIn']
        self._geom = dict([(SS[ii],out[ii]) for ii in range(0,len(SS))])
        self._Multi = out[-1]
        self._set_sino(Sino_RefPt, NP=Sino_NP)

    def _set_sino(self, RefPt=None, NP=_def.TorNP):
        if self._Done:
            Out = tfpf._get_FromItself(self, {'_sino':{'RefPt':RefPt, 'NP':NP}})
            RefPt, NP = Out['_sino']['RefPt'], Out['_sino']['NP']
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

    def get_meshEdge(self, dL, DS=None, dLMode='abs', DIn=0.):
        """ Mesh the 2D polygon edges (each segment is meshed), in the subdomain defined by DS, with resolution dL """
        Pts, dLr, ind = _comp._Ves_get_meshEdge(self.Poly, dL, DS=DS, dLMode=dLMode, DIn=DIn, VIn=self.geom['VIn'], margin=1.e-9)
        return Pts, dLr, ind

    def get_meshCross(self, dS, DS=None, dSMode='abs', ind=None):
        """ Mesh the 2D cross-section fraction defined by DS or ind, with resolution dS """
        Pts, dS, ind, dSr = _comp._Ves_get_meshCross(self.Poly, self.geom['P1Min'][0], self.geom['P1Max'][0], self.geom['P2Min'][1], self.geom['P2Max'][1], dS, DS=DS, dSMode=dSMode, ind=ind, margin=1.e-9)
        return Pts, dS, ind, dSr

    def get_meshS(self, dS, DS=None, dSMode='abs', ind=None, DIn=0., Out='(X,Y,Z)'):
        """ Mesh the surface fraction defined by DS or ind, with resolution dS and optional offset DIn

        Parameters
        ----------
        dS      :   float / list of 2 floats
            Desired resolution of the surfacic mesh
                float   : same resolution for all directions of the mesh
                list    : [dl,dXPhi] where:
                    dl      : resolution along the polygon contour in the cross-section
                    dXPhi   : resolution along the axis (toroidal direction if self.Id.Type=='Tor' or linear direction if self.Id.Type=='Lin')
        DS      :   None / list of 3 lists of 2 floats
            Limits of the domain in which the surfacic mesh should be computed
                None : whole surface of the object
                list : [D1,D2,D3] where each Di is a len()=2 list of increasing floats marking the boundaries of the domain along coordinate i, with
                    [DR,DZ,DPhi]: if toroidal geometry (self.Id.Type=='Tor')
                    [DX,DY,DZ]  : if linear geometry (self.Id.Type=='Lin')
        dSMode  :   str
            Flag specifying whether the resoltion dS shall be understood as an absolute distance or as a fraction of the distance of each element
                'abs'   :   dS is an absolute distance
                'rel'   :   if dS=0.1, each segment of the polygon will be divided in 10, and the toroidal/linear length will also be divided in 10
        ind     :   None / np.ndarray of int
            If provided, then DS is ignored and the method computes the points of the mesh corresponding to the provided indices
            Example (assuming S is a Ves or Struct object)
                > # We create a 5x5 cm2 mesh of the whole surface
                > Pts, dS, ind, dSr = S.get_mesh(0.05)
                > # Performing operations, saving only the indices of the points and not the points themselves (to save space)
                > ...
                > # Retrieving the points from their indices (requires the same resolution), here Ptsbis = Pts
                > Ptsbis, dSbis, indbis, dSrbis = S.get_mesh(0.05, ind=ind)
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
            The effective resolution in both directions after computation of the mesh
        """
        Pts, dS, ind, dSr = _comp._Ves_get_meshS(self.Poly, self.geom['P1Min'][0], self.geom['P1Max'][0], self.geom['P2Min'][1], self.geom['P2Max'][1], dS, DS=DS, dSMode=dSMode, ind=ind, DIn=DIn, VIn=self.geom['VIn'], VType=self.Type, VLim=self.Lim, Out=Out, margin=1.e-9)
        return Pts, dS, ind, dSr

    def get_meshV(self, dV, DV=None, dVMode='abs', ind=None, Out='(X,Y,Z)'):
        """ Mesh the volume fraction defined by DV or ind, with resolution dV """
        Pts, dV, ind, dVr = _comp._Ves_get_meshV(self.Poly, self.geom['P1Min'][0], self.geom['P1Max'][0], self.geom['P2Min'][1], self.geom['P2Max'][1], dV, DV=DV, dVMode=dVMode, ind=ind, VType=self.Type, VLim=self.Lim, Out=Out, margin=1.e-9)    
        return Pts, dV, ind, dVr


    def plot(self, Lax=None, Proj='All', Elt='PIBsBvV', Pdict=None, Idict=_def.TorId, Bsdict=_def.TorBsd, Bvdict=_def.TorBvd, Vdict=_def.TorVind,
            IdictHor=_def.TorITord, BsdictHor=_def.TorBsTord, BvdictHor=_def.TorBvTord, Lim=_def.Tor3DThetalim, Nstep=_def.TorNTheta, LegDict=_def.TorLegd, draw=True, a4=False, Test=True):
        """ Plot the polygon defining the vessel, with a cross-section view, a longitudinal view or both, and optionally its reference point for plotting it in projection space

        Generic method for plotting the Ves object, the projections to be plotted, the elements to plot, and the dictionaries or properties to be used for plotting each elements can all be specified using keyword arguments.
        If an ax is not provided a default one is created.

        Parameters
        ----------
        Lax :       list or plt.Axes
            The axes to be used for plotting (provide a list of 2 axes if Proj='All'), if None a new figure with axes is created
        Proj :      str
            Flag specifying the kind of projection used for the plot ('Cross' for a cross-section, 'Hor' for a horizontal plane, or 'All' for the two plots)
        Elt  :      str
            Flag specifying which elements to plot, each capital letter corresponds to an element
                * 'P': polygon
                * 'I': point used as a reference for computing impact parameters
                * 'Bs': (surfacic) center of mass
                * 'Bv': (volumic) center of mass for Tor type
                * 'V': vector pointing inward perpendicular to each segment defining the polygon
        Pdict :     dict or None
            Dictionary of properties used for plotting the polygon, fed to plt.Axes.plot() or plt.plot_surface() if Proj='3d', set to ToFu_Defauts.py if None
        Idict :     dict
            Dictionary of properties used for plotting point 'I' in Cross-section projection, fed to plt.Axes.plot()
        IdictHor :  dict
            Dictionary of properties used for plotting point 'I' in horizontal projection, fed to plt.Axes.plot()
        Bsdict :    dict
            Dictionary of properties used for plotting point 'Bs' in Cross-section projection, fed to plt.Axes.plot()
        BsdictHor : dict
            Dictionry of properties used for plotting point 'Bs' in horizontal projection, fed to plt.Axes.plot()
        Bvdict :    dict
            Dictionary of properties used for plotting point 'Bv' in Cross-section projection, fed to plt.Axes.plot()
        BvdictHor : dict
            Dictionary of properties used for plotting point 'Bv' in horizontal projection, fed to plt.Axes.plot()
        Vdict :     dict
            Dictionary of properties used for plotting point 'V' in cross-section projection, fed to plt.Axes.quiver()
        LegDict :   dict or None
            Dictionary of properties used for plotting the legend, fed to plt.legend(), the legend is not plotted if None
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
        La          list or plt.Axes    Handles of the axes used for plotting (list if several axes where used)

        """
        return _plot.Ves_plot(self, Lax=Lax, Proj=Proj, Elt=Elt, Pdict=Pdict, Idict=Idict, Bsdict=Bsdict, Bvdict=Bvdict, Vdict=Vdict,
                IdictHor=IdictHor, BsdictHor=BsdictHor, BvdictHor=BvdictHor, Lim=Lim, Nstep=Nstep, LegDict=LegDict, draw=draw, a4=a4, Test=Test)


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
            ax = _plot.Plot_Impact_PolProjPoly(self, ax=ax, Ang=Ang, AngUnit=AngUnit, Sketch=Sketch, Leg=self.Id.NameLTX, Pdict=Pdict, LegDict=LegDict, draw=False, a4=a4, Test=Test)
        else:
            Pdict = _def.TorP3DFilld if Pdict is None else Pdict
            ax = _plot.Plot_Impact_3DPoly(self, ax=ax, Ang=Ang, AngUnit=AngUnit, Pdict=Pdict, LegDict=LegDict, draw=False, a4=a4, Test=Test)
        if draw:
            ax.figure.canvas.draw()
        return ax

    def save(self, SaveName=None, Path=None, Mode='npz', compressed=False, Print=True):
        """ Save the object in folder Name, under file name SaveName, using specified mode

        Most tofu objects can be saved automatically as numpy arrays (.npz, recommended) at the default location (recommended) by simply calling self.save()

        Parameters
        ----------
        SaveName :  None / str
            The name to be used for the saved file, if None (recommended) uses self.Id.SaveName
        Path :      None / str
            Path specifying where to save the file, if None (recommended) uses self.Id.SavePath
        Mode :      str
            Flag specifying whether to save the object as a numpy array file ('.npz', recommended) or an object using cPickle (not recommended, heavier and may cause retro-compatibility issues)
        compressed :    bool
            Flag, used when Mode='npz', indicating whether to use np.savez or np.savez_compressed (slower saving and loading but smaller files)

        """
        tfpf.Save_Generic(self, SaveName=SaveName, Path=Path, Mode=Mode, compressed=compressed, Print=Print)




def _Ves_check_inputs(Id=None, Poly=None, Type=None, Lim=None, Sino_RefPt=None, Sino_NP=None, Clock=None, arrayorder=None, Exp=None, shot=None, SavePath=None, Cls=None):
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






"""
###############################################################################
###############################################################################
                        Struct class and functions
###############################################################################
"""


class Struct(Ves):

    def __init__(self, Id, Poly, Type='Tor', Lim=None, Sino_RefPt=None, Sino_NP=_def.TorNP, Clock=False, arrayorder='C', Exp=None, shot=0, SavePath=None, SavePath_Include=_def.SavePath_Include):
        Ves.__init__(self, Id, Poly, Type=Type, Lim=Lim, Sino_RefPt=Sino_RefPt, Sino_NP=Sino_NP, Clock=Clock, arrayorder=arrayorder, Exp=Exp, shot=shot, SavePath=SavePath, SavePath_Include=SavePath_Include, Cls="Struct")

    def get_meshS(self, dS, DS=None, dSMode='abs', ind=None, DIn=0., Out='(X,Y,Z)', Ind=None):
        """ Mesh the surface fraction defined by DS or ind, with resolution dS and optional offset DIn

        Parameters
        ----------
        dS      :   float / list of 2 floats
            Desired resolution of the surfacic mesh
                float   : same resolution for all directions of the mesh
                list    : [dl,dXPhi] where:
                    dl      : resolution along the polygon contour in the cross-section
                    dXPhi   : resolution along the axis (toroidal direction if self.Id.Type=='Tor' or linear direction if self.Id.Type=='Lin')
        DS      :   None / list of 3 lists of 2 floats
            Limits of the domain in which the surfacic mesh should be computed
                None : whole surface of the object
                list : [D1,D2,D3] where each Di is a len()=2 list of increasing floats marking the boundaries of the domain along coordinate i, with
                    [DR,DZ,DPhi]: if toroidal geometry (self.Id.Type=='Tor')
                    [DX,DY,DZ]  : if linear geometry (self.Id.Type=='Lin')
        dSMode  :   str
            Flag specifying whether the resoltion dS shall be understood as an absolute distance or as a fraction of the distance of each element
                'abs'   :   dS is an absolute distance
                'rel'   :   if dS=0.1, each segment of the polygon will be divided in 10, and the toroidal/linear length will also be divided in 10
        ind     :   None / np.ndarray of int
            If provided, then DS is ignored and the method computes the points of the mesh corresponding to the provided indices
            Example (assuming S is a Ves or Struct object)
                > # We create a 5x5 cm2 mesh of the whole surface
                > Pts, dS, ind, dSr = S.get_mesh(0.05)
                > # Performing operations, saving only the indices of the points and not the points themselves (to save space)
                > ...
                > # Retrieving the points from their indices (requires the same resolution), here Ptsbis = Pts
                > Ptsbis, dSbis, indbis, dSrbis = S.get_mesh(0.05, ind=ind)
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
            The effective resolution in both directions after computation of the mesh
        """
        Pts, dS, ind, dSr = _comp._Ves_get_meshS(self.Poly, self.geom['P1Min'][0], self.geom['P1Max'][0], self.geom['P2Min'][1], self.geom['P2Max'][1], dS, DS=DS, dSMode=dSMode, ind=ind, DIn=DIn, VIn=self.geom['VIn'], VType=self.Type, VLim=self.Lim, Out=Out, margin=1.e-9, Multi=self._Multi, Ind=Ind)
        return Pts, dS, ind, dSr

    def get_meshV(self, dV, DV=None, dVMode='abs', ind=None, Out='(X,Y,Z)'):
        raise AttributeError("Struct class cannot use the get_meshV() method (only surface meshing) !")



"""
###############################################################################
###############################################################################
                        LOS class and functions
###############################################################################
"""


class LOS(object):
    """ A Line-Of-Sight object (semi-line with signed direction) with all useful geometrical parameters, associated :class:`~tofu.geom.Ves` object and built-in methods for plotting, defined in (X,Y,Z) cartesian coordinates

    A Line of Sight (LOS) is a semi-line. It is a useful approximate representation of a (more accurate) Volume of Sight (VOS) when the latter is narrow and elongated.
    It is usually associated to a detector placed behind apertures.
    When associated to a :class:`~tofu.geom.Ves` object, special points are automatically computed (entry point, exit point, closest point to the center of the :class:`~tofu.geom.Ves` object...) as well as a projection in a cross-section.
    While tofu provides the possibility of creating LOS objects for academic and simplification pueposes, it is generally not recommended to use them for doing physics, consider using a Detect object instead (which will provide you with a proper and automatically-computed VOS as well as with a LOS if you want).

    Parameters
    ----------
        Id :            str / tfpf.ID
            A name string or a pre-built tfpf.ID class to be used to identify this particular instance, if a string is provided, it is fed to tfpf.ID()
        Du :            list / tuple
            List of 2 arrays of len=3, the (X,Y,Z) coordinates of respectively the starting point D of the LOS and its directing vector u (will be automatically normalized)
        Ves :           :class:`~tofu.geom.Ves`
            A :class:`~tofu.geom.Ves` instance to be associated to the created LOS
        Sino_RefPt :    None or np.ndarray
            If provided, array of size=2 containing the (R,Z) (for 'Tor' Type) or (Y,Z) (for 'Lin' Type) coordinates of the reference point for the sinogram
        arrayorder :    str
            Flag indicating whether the attributes of type=np.ndarray (e.g.: Poly) should be made C-contiguous ('C') or Fortran-contiguous ('F')
        Type       :    None
            (not used in the current version)
        Exp        :    None / str
            Experiment to which the Lens belongs, should be identical to Ves.Id.Exp if Ves is provided, if None and Ves is provided, Ves.Id.Exp is used
        Diag       :    None / str
            Diagnostic to which the Lens belongs
        shot       :    None / int
            Shot number from which this Lens is usable (in case its position was changed from a previous configuration)
        SavePath :      None / str
            If provided, forces the default saving path of the object to the provided value

    """


    def __init__(self, Id, Du, Ves=None, LStruct=None, Sino_RefPt=None, Type=None, Exp=None, Diag=None, shot=0, SavePath=None):
        self._Done = False
        if not Ves is None:
            Exp = Exp if not Exp is None else Ves.Id.Exp
            assert Exp==Ves.Id.Exp, "Arg Exp must be identical to the Ves.Exp !"
        self._set_Id(Id, Type=Type, Exp=Exp, Diag=Diag, shot=shot, SavePath=SavePath)
        self._set_Ves(Ves, LStruct=LStruct, Du=Du)
        self._set_sino(RefPt=Sino_RefPt)
        self._Done = True

    @property
    def Id(self):
        return self._Id
    @property
    def geom(self):
        return self._geom
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
    def Ves(self):
        return self._Ves
    @property
    def LStruct(self):
        return self._LStruct
    @property
    def sino(self):
        return self._sino


    def _check_inputs(self, Id=None, Du=None, Ves=None, Type=None, Sino_RefPt=None, Clock=None, arrayorder=None, Exp=None, shot=None, Diag=None, SavePath=None, Calc=None):
        _LOS_check_inputs(Id=Id, Du=Du, Vess=Ves, Type=Type, Sino_RefPt=Sino_RefPt, Clock=Clock, arrayorder=arrayorder, Exp=Exp, shot=shot, Diag=Diag, SavePath=SavePath, Calc=Calc)


    def _set_Id(self, Val, Type=None, Exp=None, Diag=None, shot=None, SavePath=None):
        if self._Done:
            Out = tfpf._get_FromItself(self.Id, {'Type':Type, 'Exp':Exp, 'shot':shot, 'Diag':Diag, 'SavePath':SavePath})
            Type, Exp, shot, Diag, SavePath = Out['Type'], Out['Exp'], Out['shot'], Out['Diag'], Out['SavePath']
        tfpf._check_NotNone({'Id':Val})
        self._check_inputs(Id=Val)
        if type(Val) is str:
            tfpf._check_NotNone({'Exp':Exp, 'shot':shot, 'Diag':Diag})
            self._check_inputs(Type=Type, Exp=Exp, shot=shot, Diag=Diag, SavePath=SavePath)
            Val = tfpf.ID('LOS', Val, Type=Type, Exp=Exp, Diag=Diag, shot=shot, SavePath=SavePath)
        self._Id = Val

    def _set_Ves(self, Ves=None, LStruct=None, Du=None):
        self._check_inputs(Ves=Ves, Exp=self.Id.Exp)
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
        self._set_geom(Du)

    def _set_geom(self, Du):
        tfpf._check_NotNone({'Du':Du})
        self._check_inputs(Du=Du)
        D, u = np.asarray(Du[0]).flatten(), np.asarray(Du[1]).flatten()
        u = u/np.linalg.norm(u,2)

        PIn, POut, kPIn, kPOut, VPerpIn, VPerpOut, IndIn, IndOut = np.NaN*np.ones((3,)), np.NaN*np.ones((3,)), np.nan, np.nan, np.NaN*np.ones((3,)), np.NaN*np.ones((3,)), np.nan, np.nan
        if not self.Ves is None:
            (LSPoly, LSLim, LSVIn) = zip(*[(ss.Poly,ss.Lim,ss.geom['VIn']) for ss in self.LStruct]) if not self.LStruct is None else (None,None,None)
            PIn, POut, kPIn, kPOut, VPerpIn, VPerpOut, IndIn, IndOut = _GG.LOS_Calc_PInOut_VesStruct(D, u, self.Ves.Poly, self.Ves.geom['VIn'], Lim=self.Ves.Lim, LSPoly=LSPoly, LSLim=LSLim, LSVIn=LSVIn,
                                                                                                     RMin=None, Forbid=True, EpsUz=1.e-6, EpsVz=1.e-9, EpsA=1.e-9, EpsB=1.e-9, EpsPlane=1.e-9,
                                                                                                     VType=self.Ves.Type, Test=True)
            if np.isnan(kPOut):
                Warnings.warn()
                La = _plot._LOS_calc_InOutPolProj_Debug(self, PIn, POut)
            if np.isnan(kPIn):
                PIn, kPIn = D, 0.

        PRMin, kRMin, RMin = _comp.LOS_PRMin(D, u, kPOut=kPOut, Eps=1.e-12, Test=True)
        self._geom = {'D':D, 'u':u,
                      'PIn':PIn, 'POut':POut, 'kPIn':kPIn, 'kPOut':kPOut,
                      'VPerpIn':VPerpIn, 'VPerpOut':VPerpOut, 'IndIn':IndIn, 'IndOut':IndOut,
                      'PRMin':PRMin, 'kRMin':kRMin, 'RMin':RMin}
        self._set_CrossProj()

    def _set_CrossProj(self):
        if not np.isnan(self.geom['kPOut']):
            CrossProjAng, kplotTot, kplotIn = _comp.LOS_CrossProj(self.Ves.Type, self.D, self.u, self.geom['kPIn'], self.geom['kPOut'], self.geom['kRMin'])
            self._geom['CrossProjAng'] = CrossProjAng
            self._geom['kplotTot'] = kplotTot
            self._geom['kplotIn'] = kplotIn

    def _set_sino(self, RefPt=None):
        self._check_inputs(Sino_RefPt=RefPt)
        if RefPt is None and self.Ves is None:
            self._sino = None
        else:
            RefPt = self.Ves.sino['RefPt'] if RefPt is None else np.asarray(RefPt).flatten()
            if self.Ves is not None:
                self._Ves._set_sino(RefPt)
                VType = self.Ves.Type
            else:
                VType = 'Lin'
            kMax = np.inf if np.isnan(self.geom['kPOut']) else self.geom['kPOut']
            Pt, kPt, r, Theta, p, theta, Phi = _GG.LOS_sino(self.D, self.u, RefPt, Mode='LOS', kOut=kMax, VType=VType)
            self._sino = {'RefPt':RefPt, 'Pt':Pt, 'kPt':kPt, 'r':r, 'Theta':Theta, 'p':p, 'theta':theta, 'Phi':Phi}

    def get_mesh(self, dL, DL=None, dLMode='abs', method='sum'):
        """ Return a linear mesh of the LOS, with desired resolution dL, in the DL interval (distance from D)  """
        DL = DL if (hasattr(DL,'__iter__') and len(DL)==2) else [self.geom['kPIn'],self.geom['kPOut']]
        Pts, kPts, dL = _comp.LOS_get_mesh(self.D, self.u, dL, DL=DL, dLMode=dLMode, method=method)
        return Pts, kPts, dL

    def calc_signal(self, ff, dL=0.001, DL=None, dLMode='abs', method='romb'):
        """ Return the line-integrated emissivity

        Beware that it is only a line-integral, there is no multiplication by an Etendue (which cannot be computed for a LOS object, because the etendue depends on the surfaces and respective positions of the detector and its apertures, which are not provided for a LOS object.
        Hence, if the emissivity is provided in W/m3, the method return a signal in W/m2
        The line is sampled using LOS.get_mesh(), and the integral can be computed using three different methods: 'sum', 'simps' or 'romb'

        Parameters
        ----------
        ff :    callable
            The user-provided



        """
        DL = DL if (hasattr(DL,'__iter__') and len(DL)==2) else [self.geom['kPIn'],self.geom['kPOut']]
        Sig = _comp.LOS_calc_signal(ff, self.D, self.u, dL=dL, DL=DL, dLMode=dLMode, method=method)
        return Sig

    def plot(self, Lax=None, Proj='All', Lplot=_def.LOSLplot, Elt='LDIORP', EltVes='', Leg='',
             Ldict=_def.LOSLd, MdictD=_def.LOSMd, MdictI=_def.LOSMd, MdictO=_def.LOSMd, MdictR=_def.LOSMd, MdictP=_def.LOSMd, LegDict=_def.TorLegd,
             Vesdict=_def.Vesdict, draw=True, a4=False, Test=True):
        """ Plot the LOS, in a cross-section projection, a horizontal projection or both, and optionally the :class:`~tofu.geom.Ves` object associated to it.

        Plot the desired projections of the LOS object.
        The plot can include the special points, the directing vector, and the properties of the plotted objects are specified by dictionaries.

        Parameters
        ----------
        Lax :       list / plt.Axes
            The axes to be used for plotting (provide a list of 2 axes if Proj='All'), if None a new figure with axes is created
        Proj :      str
            Flag specifying the kind of projection used for the plot ('Cross' for a cross-section, 'Hor' for a horizontal plane, 'All' both and '3d' for 3d)
        Elt :       str
            Flag specifying which elements to plot, each capital letter corresponds to an element
                * 'L': LOS
                * 'D': Starting point of the LOS
                * 'I': Input point (i.e.: where the LOS enters the Vessel)
                * 'O': Output point (i.e.: where the LOS exits the Vessel)
                * 'R': Point of minimal major radius R (only for Vessel of Type='Tor')
                * 'P': Point of used for impact parameter (i.e.: minimal distance to reference point Sino_RefPt)
        Lplot :     str
            Flag specifying whether to plot the full LOS ('Tot': from starting point output point) or only the fraction inside the vessel ('In': from input to output point)
        EltVes :    str
            Flag specifying the elements of the Vessel to be plotted, fed to :meth:`~tofu.geom.Ves.plot`
        Leg :       str
            Legend to be used to identify this LOS, if Leg='' the LOS name is used
        Ldict :     dict / None
            Dictionary of properties used for plotting the polygon, fed to plt.Axes.plot() or plt.plot_surface() if Proj='3d', set to ToFu_Defauts.py if None
        MdictD :    dict
            Dictionary of properties used for plotting point 'D', fed to plt.Axes.plot()
        MdictI :    dict
            Dictionary of properties used for plotting point 'I', fed to plt.Axes.plot()
        MdictO :    dict
            Dictionary of properties used for plotting point 'O', fed to plt.Axes.plot()
        MdictR :    dict
            Dictionary of properties used for plotting point 'R', fed to plt.Axes.plot()
        MdictP :    dict
            Dictionary of properties used for plotting point 'P', fed to plt.Axes.plot()
        LegDict :   dict or None
            Dictionary of properties used for plotting the legend, fed to plt.legend(), the legend is not plotted if None
        Vesdict :   dict
            Dictionary of kwdargs to fed to :meth:`~tofu.geom.Ves.plot`, and 'EltVes' is used instead of 'Elt'
        draw :      bool
            Flag indicating whether the fig.canvas.draw() shall be called automatically
        a4 :        bool
            Flag indicating whether the figure should be plotted in a4 dimensions for printing
        Test :      bool
            Flag indicating whether the inputs should be tested for conformity

        Returns
        -------
        La :        list / plt.Axes
            Handles of the axes used for plotting (list if several axes where used)

        """
        return _plot.GLLOS_plot(self, Lax=Lax, Proj=Proj, Lplot=Lplot, Elt=Elt, EltVes=EltVes, Leg=Leg,
                                Ldict=Ldict, MdictD=MdictD, MdictI=MdictI, MdictO=MdictO, MdictR=MdictR, MdictP=MdictP, LegDict=LegDict,
                                Vesdict=Vesdict, draw=draw, a4=a4, Test=Test)

#    def plot_3D_mlab(self,Lplot='Tot',PDIOR='DIOR',axP='None',axT='None', Ldict=Ldict_Def,Mdict=Mdict_Def,LegDict=LegDict_Def):
#        fig = Plot_3D_mlab_GLOS()
#        return fig


    def plot_sino(self, Proj='Cross', ax=None, Elt=_def.LOSImpElt, Sketch=True, Ang=_def.LOSImpAng, AngUnit=_def.LOSImpAngUnit,
                      Ldict=_def.LOSMImpd, Vdict=_def.TorPFilld, LegDict=_def.TorLegd, draw=True, a4=False, Test=True):
        """ Plot the sinogram of the vessel polygon, by computing its envelopp in a cross-section, can also plot a 3D version of it

        Plot the LOS in projection space (where sinograms are plotted) as a point.
        You can plot the conventional projection-space (in 2D in a cross-section), or a 3D extrapolation of it, where the third coordinate is provided by the angle that the LOS makes with the cross-section plane (useful in case of multiple LOS with a partially tangential view).

        Parameters
        ----------
        Proj :      str
            Flag indicating whether to plot a classic sinogram ('Cross') from the vessel cross-section (assuming 2D), or an extended 3D version ('3d') of it with additional angle
        ax :        None or plt.Axes
            The axes on which the plot should be done, if None a new figure and axes is created
        Elt :       str
            Flag indicating which elements to plot, each capital letter stands for one element
                * 'L': LOS
                * 'V': Vessel
        Ang  :      str
            Flag indicating which angle to use for the impact parameter, the angle of the line itself (xi) or of its impact parameter (theta)
        AngUnit :   str
            Flag for the angle units to be displayed, 'rad' for radians or 'deg' for degrees
        Sketch :    bool
            Flag indicating whether a small skecth showing the definitions of angles 'theta' and 'xi' should be included or not
        Ldict :     dict
            Dictionary of properties used for plotting the LOS point, fed to plt.plot() if Proj='Cross' and to plt.plot_surface() if Proj='3d'
        Vdict :     dict
            Dictionary of properties used for plotting the polygon envelopp, fed to plt.plot() if Proj='Cross' and to plt.plot_surface() if Proj='3d'
        LegDict :   None or dict
            Dictionary of properties used for plotting the legend, fed to plt.legend(), the legend is not plotted if None
        draw :      bool
            Flag indicating whether to draw the figure
        a4 :        bool
            Flag indicating whether the figure should be plotted in a4 dimensions for printing
        Test :      bool
            Flag indicating whether the inputs shall be tested for conformity

        Returns
        -------
        ax :        plt.Axes
            The axes used to plot

        """
        return _plot.GLOS_plot_Sinogram(self, Proj=Proj, ax=ax, Elt=Elt, Sketch=Sketch, Ang=Ang, AngUnit=AngUnit,
                                        Ldict=Ldict, Vdict=Vdict, LegDict=LegDict, draw=draw, a4=a4, Test=Test)


    def save(self, SaveName=None, Path=None, Mode='npz', compressed=False, Print=True):
        """ Save the object in folder Name, under file name SaveName, using specified mode

        Most tofu objects can be saved automatically as numpy arrays (.npz, recommended) at the default location (recommended) by simply calling self.save()

        Parameters
        ----------
        SaveName :  None / str
            The name to be used for the saved file, if None (recommended) uses self.Id.SaveName
        Path :      None / str
            Path specifying where to save the file, if None (recommended) uses self.Id.SavePath
        Mode :      str
            Flag specifying whether to save the object as a numpy array file ('.npz', recommended) or an object using cPickle (not recommended, heavier and may cause retro-compatibility issues)
        compressed :    bool
            Flag, used when Mode='npz', indicating whether to use np.savez or np.savez_compressed (slower saving and loading but smaller files)

        """
        tfpf.Save_Generic(self, SaveName=SaveName, Path=Path, Mode=Mode, compressed=compressed, Print=Print)



def _LOS_check_inputs(Id=None, Du=None, Vess=None, Type=None, Sino_RefPt=None, Clock=None, arrayorder=None, Exp=None, shot=None, Diag=None, SavePath=None, Calc=None):
    if not Id is None:
        assert type(Id) in [str,tfpf.ID], "Arg Id must be a str or a tfpf.ID object !"
    if not Du is None:
        assert hasattr(Du,'__iter__') and len(Du)==2 and all([hasattr(du,'__iter__') and len(du)==3 for du in Du]), "Arg Du must be an iterable containing of two iterables of len()=3 (cartesian coordinates) !"
    if not Vess is None:
        assert type(Vess) is Ves, "Arg Ves must be a Ves instance !"
        if not Exp is None:
            assert Exp==Vess.Id.Exp, "Arg Exp must be the same as Ves.Id.Exp !"
    bools = [Clock,Calc]
    if any([not aa is None for aa in bools]):
        assert all([aa is None or type(aa) is bool for aa in bools]), " Args [Clock,Calc] must all be bool !"
    if not arrayorder is None:
        assert arrayorder in ['C','F'], "Arg arrayorder must be in ['C','F'] !"
    assert Type is None, "Arg Type must be None for a LOS object !"
    strs = [Exp,Diag,SavePath]
    if any([not aa is None for aa in strs]):
        assert all([aa is None or type(aa) is str for aa in strs]), "Args [Exp,Diag,SavePath] must all be str !"
    Iter2 = [Sino_RefPt]
    if any([not aa is None for aa in Iter2]):
        assert all([aa is None or (hasattr(aa,'__iter__') and np.asarray(aa).ndim==1 and np.asarray(aa).size==2) for aa in Iter2]), "Args [DLong,Sino_RefPt] must be an iterable with len()=2 !"
    Ints = [shot]
    if any([not aa is None for aa in Ints]):
        assert all([aa is None or type(aa) is int for aa in Ints]), "Args [Sino_NP,shot] must be int !"
