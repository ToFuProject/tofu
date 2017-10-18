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
    import tofu.geom._defaults as _tfd
    import tofu.geom._GG as _GG 
    import tofu.geom._comp as _comp
    import tofu.geom._plot02 as _plot
except Exception:
    from . import _defaults as _tfd
    import _GG as _GG
    from . import _comp as _comp
    from . import _plot02 as _plot

__all__ = ['Ves', 'Struct',
           '_GG','_comp','_plot','_tfd']



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
    dtime :         None / dtm.datetime
        A time reference to be used to identify this particular instance (used for debugging mostly)

    Returns
    -------
    Ves :        Ves object
        The created Ves object, with all necessary computed attributes and methods

    """

    def __init__(self, Id, Poly, Type='Tor', Lim=None, Sino_RefPt=None, Sino_NP=_tfd.TorNP, Clock=False, arrayorder='C', Exp=None, shot=None, dtime=None, SavePath=None, SavePath_Include=_tfd.SavePath_Include, Cls='Ves'):

        self._Done = False
        tfpf._check_NotNone({'Clock':Clock,'arrayorder':arrayorder})
        self._check_inputs(Clock=Clock, arrayorder=arrayorder)
        self._arrayorder = arrayorder
        self._Clock = Clock
        self._set_Id(Id, Type=Type, Exp=Exp, shot=shot, SavePath=SavePath, dtime=dtime, SavePath_Include=SavePath_Include, Cls=Cls)
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


    def _check_inputs(self, Id=None, Poly=None, Type=None, Lim=None, Sino_RefPt=None, Sino_NP=None, Clock=None, arrayorder=None, Exp=None, shot=None, dtime=None, SavePath=None):
        _Ves_check_inputs(Id=Id, Poly=Poly, Type=Type, Lim=Lim, Sino_RefPt=Sino_RefPt, Sino_NP=Sino_NP, Clock=Clock, arrayorder=arrayorder, Exp=Exp, shot=shot, dtime=dtime, SavePath=SavePath)

    def _set_Id(self, Val, Type=None, Exp=None, shot=None, dtime=None, SavePath=None, SavePath_Include=None, Cls='Ves'):
        if self._Done:
            Out = tfpf._get_FromItself(self.Id,{'Type':Type, 'Exp':Exp, 'shot':shot, 'dtime':dtime, 'SavePath':SavePath})
            Type, Exp, shot, dtime, SavePath = Out['Type'], Out['Exp'], Out['shot'], Out['dtime'], Out['SavePath']
        tfpf._check_NotNone({'Id':Val})
        self._check_inputs(Id=Val)
        if type(Val) is str:
            tfpf._check_NotNone({'Type':Type, 'Exp':Exp, 'shot':shot})
            self._check_inputs(Type=Type, Exp=Exp, shot=shot, SavePath=SavePath, dtime=dtime)
            Val = tfpf.ID(Cls, Val, Type=Type, Exp=Exp, shot=shot, SavePath=SavePath, Include=SavePath_Include, dtime=dtime)
        self._Id = Val

    def _set_arrayorder(self, arrayorder):
        tfpf._set_arrayorder(self, arrayorder)

    def _set_geom(self, Poly, Lim=None, Clock=False, Sino_RefPt=None, Sino_NP=_tfd.TorNP):
        if self._Done:
            Out = tfpf._get_FromItself(self, {'Lim':Lim, '_Clock':Clock})
            Lim, Clock = Out['Lim'], Out['Clock']
        tfpf._check_NotNone({'Poly':Poly, 'Clock':Clock})
        out = _comp._Ves_set_Poly(Poly, self._arrayorder, self.Type, Lim=Lim, Clock=Clock)
        SS = ['Poly','NP','P1Max','P1Min','P2Max','P2Min','BaryP','BaryL','Surf','BaryS','Lim','VolLin','BaryV','Vect','VIn']
        self._geom = dict([(SS[ii],out[ii]) for ii in range(0,len(out))])
        self._set_Sino(Sino_RefPt, NP=Sino_NP)

    def _set_Sino(self, RefPt=None, NP=_tfd.TorNP):
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
            Flag indicating the coordinate system in which the points are provided, in ['(X,Y,Z)','(R,Z)','']

        Returns
        -------
        ind :   np.ndarray
            Array of booleans of shape (N,), True if a point is inside the Ves volume

        """
        ind = _GG._Ves_isInside(Pts, self.Poly, VLong=self.geom['Lim'], VType=self.Type, In=In, Test=True)
        return ind


    def get_InsideConvexPoly(self, RelOff=_tfd.TorRelOff, ZLim='Def', Spline=True, Splprms=_tfd.TorSplprms, NP=_tfd.TorInsideNP, Plot=False, Test=True):
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
        """ Mesh the surface fraction defined by DS or ind, with resolution dS and optional offset DIn """
        Pts, dS, ind, dSr = _comp._Ves_get_meshS(self.Poly, self.geom['P1Min'][0], self.geom['P1Max'][0], self.geom['P2Min'][1], self.geom['P2Max'][1], dS, DS=DS, dSMode=dSMode, ind=ind, DIn=DIn, VIn=self.geom['VIn'], VType=self.Type, VLim=self.Lim, Out=Out, margin=1.e-9)
        return Pts, dS, ind, dSr

    def get_meshV(self, dV, DV=None, dVMode='abs', ind=None, Out='(X,Y,Z)'):
        """ Mesh the volume fraction defined by DV or ind, with resolution dV """
        Pts, dV, ind, dVr = _comp._Ves_get_meshV(self.Poly, self.geom['P1Min'][0], self.geom['P1Max'][0], self.geom['P2Min'][1], self.geom['P2Max'][1], dV, DV=DV, dVMode=dVMode, ind=ind, VType=self.Type, VLim=self.Lim, Out=Out, margin=1.e-9)    
        return Pts, dV, ind, dVr


    def plot(self, Lax=None, Proj='All', Elt='PIBsBvV', Pdict=None, Idict=_tfd.TorId, Bsdict=_tfd.TorBsd, Bvdict=_tfd.TorBvd, Vdict=_tfd.TorVind,
            IdictHor=_tfd.TorITord, BsdictHor=_tfd.TorBsTord, BvdictHor=_tfd.TorBvTord, Lim=_tfd.Tor3DThetalim, Nstep=_tfd.TorNTheta, LegDict=_tfd.TorLegd, draw=True, a4=False, Test=True):
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


    def plot_sino(self, Proj='Cross', ax=None, Ang=_tfd.LOSImpAng, AngUnit=_tfd.LOSImpAngUnit, Sketch=True, Pdict=None, LegDict=_tfd.TorLegd, draw=True, a4=False, Test=True):
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
            Pdict = _tfd.TorPFilld if Pdict is None else Pdict
            ax = _plot.Plot_Impact_PolProjPoly(self, ax=ax, Ang=Ang, AngUnit=AngUnit, Sketch=Sketch, Leg=self.Id.NameLTX, Pdict=Pdict, LegDict=LegDict, draw=False, a4=a4, Test=Test)
        else:
            Pdict = _tfd.TorP3DFilld if Pdict is None else Pdict
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




def _Ves_check_inputs(Id=None, Poly=None, Type=None, Lim=None, Sino_RefPt=None, Sino_NP=None, Clock=None, arrayorder=None, Exp=None, shot=None, dtime=None, SavePath=None):
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
    if not Exp is None:
        assert type(Exp) is str, "Ar Exp must be a str !"
    strs = [SavePath]
    if any([not aa is None for aa in strs]):
        assert all([aa is None or type(aa) is str for aa in strs]), "Args [Type,Exp,SavePath] must all be str !"
    Iter2 = [Lim,Sino_RefPt]
    if any([not aa is None for aa in Iter2]):
        assert all([aa is None or (hasattr(aa,'__iter__') and np.asarray(aa).ndim==1 and np.asarray(aa).size==2) for aa in Iter2]), "Args [Lim,Sino_RefPt] must be an iterable with len()=2 !"
    Ints = [Sino_NP,shot]
    if any([not aa is None for aa in Ints]):
        assert all([aa is None or type(aa) is int for aa in Ints]), "Args [Sino_NP,shot] must be int !"
    if not dtime is None:
        assert type(dtime) is dtm.datetime, "Arg dtime must be a dtm.datetime !"










"""
###############################################################################
###############################################################################
                        Struct class and functions
###############################################################################
"""


class Struct(Ves):

    def __init__(self, Id, Poly, Type='Tor', Lim=None, Sino_RefPt=None, Sino_NP=_tfd.TorNP, Clock=False, arrayorder='C', Exp=None, shot=None, dtime=None, SavePath=None, SavePath_Include=_tfd.SavePath_Include):
        Ves.__init__(self, Id, Poly, Type=Type, Lim=Lim, Sino_RefPt=Sino_RefPt, Sino_NP=Sino_NP, Clock=Clock, arrayorder=arrayorder, Exp=Exp, shot=shot, dtime=dtime, SavePath=SavePath, SavePath_Include=SavePath_Include, Cls="Struct")

    def get_meshS(self, dS, DS=None, dSMode='abs', ind=None, DIn=0., Out='(X,Y,Z)'):
        """ Mesh the surface fraction defined by DS or ind, with resolution dS and optional offset DIn """
        Pts, dS, ind, dSr = _comp._Ves_get_meshS(self.Poly, self.geom['P1Min'][0], self.geom['P1Max'][0], self.geom['P2Min'][1], self.geom['P2Max'][1], dS, DS=DS, dSMode=dSMode, ind=ind, DIn=DIn, VIn=self.geom['VIn'], VType=self.Type, VLim=self.Lim, Out=Out, margin=1.e-9)
        return Pts, dS, ind, dSr

    def get_meshV(self, dV, DV=None, dVMode='abs', ind=None, Out='(X,Y,Z)'):
        raise AttributeError("Struct class cannot use the get_meshV() method (only surface meshing) !")


