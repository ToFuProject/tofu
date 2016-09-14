"""
This module is the geometrical part of the ToFu general package
It includes all functions and object classes necessary for tomography on Tokamaks
"""

import warnings
import numpy as np
import datetime as dtm

# ToFu-specific
import tofu.defaults as tfd
import tofu.pathfile as tfpf
from . import General_Geom_cy as _tfg_gg
from . import _compute as _tfg_c
from . import _plot as _tfg_p

__author__ =    "D. Vezinet"
__all__ = ['Ves','Struct','LOS','GLOS','Lens','Apert','Detect','GDetect']




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
    DLong :         list / np.ndarray
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
    dtimeIn :       bool
        Flag indicating whether dtime should be included in the SaveName (used for debugging mostly)

    Returns
    -------
    Ves :        Ves object
        The created Ves object, with all necessary computed attributes and methods

    """

    def __init__(self, Id, Poly, Type='Tor', DLong=None, Sino_RefPt=None, Sino_NP=tfd.TorNP, Clock=False, arrayorder='C', Exp=None, shot=None, dtime=None, dtimeIn=False, SavePath=None):

        self._Done = False
        tfpf._check_NotNone({'Clock':Clock,'arrayorder':arrayorder})
        self._check_inputs(Clock=Clock, arrayorder=arrayorder)
        self._arrayorder = arrayorder
        self._Clock = Clock
        self._set_Id(Id, Type=Type, Exp=Exp, shot=shot, SavePath=SavePath, dtime=dtime, dtimeIn=dtimeIn)
        self._set_Poly(Poly, DLong=DLong, Clock=Clock, Sino_RefPt=Sino_RefPt, Sino_NP=Sino_NP)
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
    def Poly(self):
        """Return the polygon defining the vessel cross-section"""
        return self._Poly
    @property
    def Vect(self):
        """Return the polygon elementary vectors"""
        return self._Vect
    @property
    def Vin(self):
        """Return the normalized vectors pointing inwards for each segment of the polygon"""
        return self._Vin
    @property
    def DLong(self):
        return self._DLong
    @property
    def Surf(self):
        """Return the area of the polygon defining the vessel cross-section"""
        return self._Surf
    @property
    def VolLin(self):
        """Return the angular volume of the polygon defining the vessel cross-section of Tor type"""
        return self._VolLin
    @property
    def BaryS(self):
        """Return the (surfacic) center of mass of the polygon defining the vessel cross-section"""
        return self._BaryS
    @property
    def BaryV(self):
        """Return the (volumic) center of mass of the polygon defining the vessel cross-section"""
        return self._BaryV
    @property
    def Sino_RefPt(self):
        """Return the 2D coordinates of the points used as a reference for computing the Ves polygon in projection space (where sinograms are plotted)"""
        return self._Sino_RefPt
    @property
    def Sino_NP(self):
        """Return the number of points used used for plotting the Ves polygon in projection space"""
        return self._Sino_NP
    @property
    def arrayorder(self):
        """Return the flag indicating which order is used for multi-dimensional array attributes"""
        return self._arrayorder


    def _check_inputs(self, Id=None, Poly=None, Type=None, DLong=None, Sino_RefPt=None, Sino_NP=None, Clock=None, arrayorder=None, Exp=None, shot=None, dtime=None, dtimeIn=None, SavePath=None):
        _Ves_check_inputs(Id=Id, Poly=Poly, Type=Type, DLong=DLong, Sino_RefPt=Sino_RefPt, Sino_NP=Sino_NP, Clock=Clock, arrayorder=arrayorder, Exp=Exp, shot=shot, dtime=dtime, dtimeIn=dtimeIn, SavePath=SavePath)

    def _set_Id(self, Val, Type=None, Exp=None, shot=None, dtime=None, dtimeIn=False, SavePath=None):
        if self._Done:
            Out = tfpf._get_FromItself(self.Id,{'Type':Type, 'Exp':Exp, 'shot':shot, 'dtime':dtime, '_dtimeIn':dtimeIn, 'SavePath':SavePath})
            Type, Exp, shot, dtime, dtimeIn, SavePath = Out['Type'], Out['Exp'], Out['shot'], Out['dtime'], Out['dtimeIn'], Out['SavePath']
        tfpf._check_NotNone({'Id':Val})
        self._check_inputs(Id=Val)
        if type(Val) is str:
            tfpf._check_NotNone({'Type':Type, 'Exp':Exp, 'shot':shot,'dtimeIn':dtimeIn})
            self._check_inputs(Type=Type, Exp=Exp, shot=shot, SavePath=SavePath, dtime=dtime, dtimeIn=dtimeIn)
            Val = tfpf.ID('Ves', Val, Type=Type, Exp=Exp, shot=shot, SavePath=SavePath, dtime=dtime, dtimeIn=dtimeIn)
        self._Id = Val

    def _set_arrayorder(self, arrayorder):
        tfpf._set_arrayorder(self, arrayorder)

    def _set_Poly(self, Poly, DLong=None, Clock=False, Sino_RefPt=None, Sino_NP=tfd.TorNP):
        if self._Done:
            Out = tfpf._get_FromItself(self, {'DLong':DLong, '_Clock':Clock})
            DLong, Clock = Out['DLong'], Out['Clock']
        tfpf._check_NotNone({'Poly':Poly, 'Clock':Clock})
        self._Poly, self._NP, self._P1Max, self._P1Min, self._P2Max, self._P2Min, self._BaryP, self._BaryL, self._Surf, self._BaryS, self._DLong, self._VolLin, self._BaryV, self._Vect, self._Vin = _tfg_c._Ves_set_Poly(Poly, self.arrayorder, self.Type, DLong=DLong, Clock=Clock)
        self._set_Sino(Sino_RefPt, NP=Sino_NP)

    def _set_Sino(self, RefPt=None, NP=tfd.TorNP):
        if self._Done:
            Out = tfpf._get_FromItself(self, {'Sino_RefPt':RefPt, 'Sino_NP':NP})
            RefPt, NP = Out['Sino_RefPt'], Out['Sino_NP']
            tfpf._check_NotNone({'Sino_NP':NP})
        if RefPt is None:
            RefPt = self.BaryS
        RefPt = np.asarray(RefPt).flatten()
        self._Sino_EnvTheta, self._Sino_EnvMinMax = _tfg_gg.Calc_ImpactEnv(RefPt, self.Poly, NP=NP, Test=False)
        self._Sino_RefPt, self._Sino_NP = RefPt, NP

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
        return _tfg_c._Ves_isInside(self.Poly, self.Type, self.DLong, Pts, In=In)

    def get_InsideConvexPoly(self, RelOff=tfd.TorRelOff, ZLim='Def', Spline=True, Splprms=tfd.TorSplprms, NP=tfd.TorInsideNP, Plot=False, Test=True):
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
        return _tfg_c._Ves_get_InsideConvexPoly(self.Poly, self._P2Min, self._P2Max, self.BaryS, RelOff=RelOff, ZLim=ZLim, Spline=Spline, Splprms=Splprms, NP=NP, Plot=Plot, Test=Test)

    def get_MeshCrossSection(self, CrossMesh=[0.01,0.01], CrossMeshMode='abs', Test=True):
        """ Return a (2,N) array of 2D points coordinates meshing the Ves cross-section using the spacing specified by CrossMesh for each direction (taken as absolute distance or relative to the total size)

        Method used for fast automatic meshing of the cross-section using a rectangular mesh uniform in each direction.
        Returns the flattened points coordinates array, as well as the two increasing vectors and number of points.

        Parameters
        ----------
        CrossMesh :     iterable
            Iterable of len()==2 specifying the distance to be used between points in each direction (R or Y and Z), in absolute value or relative to the total size of the Ves in each direction
        CrossMeshMode : str
            Flag specifying whether the distances provided in CrossMesh are absolute ('abs') or relative ('rel')
        Test :          bool
            Flag indicating whether the inputs should be tested for conformity

        Returns
        -------
        Pts :           np.ndarray
            Array of shape (2,N), comtaining the 2D coordinates of the N points consituting the mesh, only points lying inside the cross-section are returned
        X1 :            np.ndarray
            Flat array of the unique first coordinates of the mesh points (R or Y)
        X2 :            np.ndarray
            Flat array of the unique second coordinates of the mesh points (Z)
        NumX1 :         int
            Number of unique values in X1 (=X1.size)
        NumX2 :         int
            Number of unique values in X2 (=X2.size)

        """
        Pts, X1, X2, NumX1, NumX2 = _tfg_c._Ves_get_MeshCrossSection(self._P1Min, self._P1Max, self._P2Min, self._P2Max, self.Poly, self.Type, DLong=self.DLong, CrossMesh=CrossMesh, CrossMeshMode=CrossMeshMode, Test=Test)
        return Pts, X1, X2, NumX1, NumX2



    def plot(self, Lax=None, Proj='All', Elt='PIBsBvV', Pdict=None, Idict=tfd.TorId, Bsdict=tfd.TorBsd, Bvdict=tfd.TorBvd, Vdict=tfd.TorVind,
            IdictHor=tfd.TorITord, BsdictHor=tfd.TorBsTord, BvdictHor=tfd.TorBvTord, Lim=tfd.Tor3DThetalim, Nstep=tfd.TorNTheta, LegDict=tfd.TorLegd, draw=True, a4=False, Test=True):
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
        return _tfg_p.Ves_plot(self, Lax=Lax, Proj=Proj, Elt=Elt, Pdict=Pdict, Idict=Idict, Bsdict=Bsdict, Bvdict=Bvdict, Vdict=Vdict,
                IdictHor=IdictHor, BsdictHor=BsdictHor, BvdictHor=BvdictHor, Lim=Lim, Nstep=Nstep, LegDict=LegDict, draw=draw, a4=a4, Test=Test)

    """
    def plot_3D_mlab(self,f=None,Tdict=Dict_3D_mlab_Tor_Def,LegDict=LegDict_Def,Test=True):
        f = Plot_3D_mlab_Tor(self,fig=f,Tdict=Tdict,LegDict=LegDict,Test=Test)
        return f
    """

    def plot_Sinogram(self, Proj='Cross', ax=None, Ang=tfd.LOSImpAng, AngUnit=tfd.LOSImpAngUnit, Sketch=True, Pdict=None, LegDict=tfd.TorLegd, draw=True, a4=False, Test=True):
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
            assert not self.Sino_RefPt is None, 'The impact parameters must be computed first !'
            assert Proj in ['Cross','3d'], "Arg Proj must be in ['Cross','3d'] !"
        if Proj=='Cross':
            Pdict = tfd.TorPFilld if Pdict is None else Pdict
            ax = _tfg_p.Plot_Impact_PolProjPoly(self, ax=ax, Ang=Ang, AngUnit=AngUnit, Sketch=Sketch, Leg=self.Id.NameLTX, Pdict=Pdict, LegDict=LegDict, draw=False, a4=a4, Test=Test)
        else:
            Pdict = tfd.TorP3DFilld if Pdict is None else Pdict
            ax = _tfg_p.Plot_Impact_3DPoly(self, ax=ax, Ang=Ang, AngUnit=AngUnit, Pdict=Pdict, LegDict=LegDict, draw=False, a4=a4, Test=Test)
        if draw:
            ax.figure.canvas.draw()
        return ax

    def save(self, SaveName=None, Path=None, Mode='npz', compressed=False):
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
        tfpf.Save_Generic(self, SaveName=SaveName, Path=Path, Mode=Mode, compressed=compressed)




def _Ves_check_inputs(Id=None, Poly=None, Type=None, DLong=None, Sino_RefPt=None, Sino_NP=None, Clock=None, arrayorder=None, Exp=None, shot=None, dtime=None, dtimeIn=None, SavePath=None):
    if not Id is None:
        assert type(Id) in [str,tfpf.ID], "Arg Id must be a str or a tfpf.ID object !"
    if not Poly is None:
        assert hasattr(Poly,'__iter__') and np.asarray(Poly).ndim==2 and 2 in np.asarray(Poly).shape, "Arg Poly must be a dict or an iterable with 2D coordinates of cross section poly !"
    bools = [Clock,dtimeIn]
    if any([not aa is None for aa in bools]):
        assert all([aa is None or type(aa) is bool for aa in bools]), " Args [Clock,dtimeIn] must all be bool !"
    if not arrayorder is None:
        assert arrayorder in ['C','F'], "Arg arrayorder must be in ['C','F'] !"
    if not Type is None:
        assert Type in ['Tor','Lin'], "Arg Type must be in ['Tor','Lin'] !"
    if not Exp is None:
        assert Exp in tfd.AllowedExp, "Ar Exp must be in "+str(tfd.AllowedExp)+" !"
    strs = [SavePath]
    if any([not aa is None for aa in strs]):
        assert all([aa is None or type(aa) is str for aa in strs]), "Args [Type,Exp,SavePath] must all be str !"
    Iter2 = [DLong,Sino_RefPt]
    if any([not aa is None for aa in Iter2]):
        assert all([aa is None or (hasattr(aa,'__iter__') and np.asarray(aa).ndim==1 and np.asarray(aa).size==2) for aa in Iter2]), "Args [DLong,Sino_RefPt] must be an iterable with len()=2 !"
    Ints = [Sino_NP,shot]
    if any([not aa is None for aa in Ints]):
        assert all([aa is None or type(aa) is int for aa in Ints]), "Args [Sino_NP,shot] must be int !"
    if not dtime is None:
        assert type(dtime) is dtm.datetime, "Arg dtime must be a dtm.datetime !"










"""
###############################################################################
###############################################################################
                        Sruct class and functions
###############################################################################
"""

# A class defining a Linear or Toroidal structural element (i.e. a 2D polygon representing a cross-section and assumed to be linearly or toroidally invariant), has no physical role, just used for illustrative purposes in plots


class Struct(object):
    """ A class defining a Linear or Toroidal structural element (i.e. a 2D polygon representing a cross-section and assumed to be linearly or toroidally invariant), like a :class:`~tofu.geom.Ves` but with less properties.

    A Struct object is mostly defined by a close 2D polygon, which can be understood as a poloidal cross-section in (R,Z) cylindrical coordinates if Type='Tor' (toroidal shape) or as a straight cross-section through a cylinder in (Y,Z) cartesian coordinates if Type='Lin' (linear shape).
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
    DLong :         list / np.ndarray
        Array or list of len=2 indicating the limits of the linear device volume on the x axis
    Ves :           None or :class:`~tofu.geom.Ves`
        An optional associated vessel
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
    dtimeIn :       bool
        Flag indicating whether dtime should be included in the SaveName (used for debugging mostly)

    Returns
    -------
    struct :        Struct object
        The created Struct object, with all necessary computed attributes and methods

    """

    def __init__(self, Id, Poly, Type='Tor', DLong=None, Ves=None, Clock=False, arrayorder='C', Exp=None, shot=None, dtime=None, dtimeIn=False, SavePath=None):

        self._Done = False
        tfpf._check_NotNone({'Id':Id, 'Poly':Poly, 'Type':Type, 'Clock':Clock,'arrayorder':arrayorder})
        self._check_inputs(Clock=Clock, arrayorder=arrayorder)
        self._arrayorder = arrayorder
        self._Clock = Clock
        Type = Type if Ves is None else Ves.Id.Type
        self._set_Id(Id, Type=Type, Exp=Exp, shot=shot, SavePath=SavePath, dtime=dtime, dtimeIn=dtimeIn)
        self._set_Ves(Ves)
        self._set_Poly(Poly, DLong=DLong, Clock=Clock)
        self._set_arrayorder(arrayorder)
        self._Done = True

    @property
    def Id(self):
        """Return the tfpf.ID object of the structure """
        return self._Id
    @property
    def Type(self):
        """Return the type of structure """
        return self.Id.Type
    @property
    def Poly(self):
        """Return the polygon defining the vessel cross-section"""
        return self._Poly
    @property
    def Vect(self):
        """Return the polygon elementary vectors"""
        return self._Vect
    @property
    def Vin(self):
        """Return the normalized vectors pointing inwards for each segment of the polygon"""
        return self._Vin
    @property
    def DLong(self):
        """ Return the length spanned by the object in the ignorable coordinate """
        return self._DLong
    @property
    def Surf(self):
        """Return the area of the polygon defining the vessel cross-section"""
        return self._Surf
    @property
    def VolLin(self):
        """Return the angular volume of the polygon defining the vessel cross-section of Tor type"""
        return self._VolLin
    @property
    def BaryS(self):
        """Return the (surfacic) center of mass of the polygon defining the vessel cross-section"""
        return self._BaryS
    @property
    def BaryV(self):
        """Return the (volumic) center of mass of the polygon defining the vessel cross-section"""
        return self._BaryV
    @property
    def Ves(self):
        """ Return the associated :class:`~tofu.goem.Ves` object, if any """
        return self._Ves
    @property
    def arrayorder(self):
        """Return the flag indicating which order is used for multi-dimensional array attributes"""
        return self._arrayorder


    def _check_inputs(self, Id=None, Poly=None, Type=None, DLong=None, Ves=None, Clock=None, arrayorder=None, Exp=None, shot=None, dtime=None, dtimeIn=None, SavePath=None):
        _Struct_check_inputs(Id=Id, Poly=Poly, Type=Type, DLong=DLong, Vess=Ves, Clock=Clock, arrayorder=arrayorder, Exp=Exp, shot=shot, dtime=dtime, dtimeIn=dtimeIn, SavePath=SavePath)

    def _set_Id(self, Val, Type=None, Exp=None, shot=None, dtime=None, dtimeIn=False, SavePath=None):
        if self._Done:
            Out = tfpf._get_FromItself(self.Id,{'Type':Type, 'Exp':Exp, 'shot':shot, 'dtime':dtime, '_dtimeIn':dtimeIn, 'SavePath':SavePath})
            Type, Exp, shot, dtime, dtimeIn, SavePath = Out['Type'], Out['Exp'], Out['shot'], Out['dtime'], Out['dtimeIn'], Out['SavePath']
        tfpf._check_NotNone({'Id':Val})
        self._check_inputs(Id=Val)
        if type(Val) is str:
            tfpf._check_NotNone({'Type':Type, 'Exp':Exp, 'shot':shot, 'dtimeIn':dtimeIn})
            self._check_inputs(Type=Type, Exp=Exp, shot=shot, SavePath=SavePath, dtime=dtime, dtimeIn=dtimeIn)
            Val = tfpf.ID('Struct', Val, Type=Type, Exp=Exp, shot=shot, SavePath=SavePath, dtime=dtime, dtimeIn=dtimeIn)
        self._Id = Val

    def _set_arrayorder(self, arrayorder):
        tfpf._set_arrayorder(self, arrayorder)

    def _set_Ves(self, Ves):
        self._check_inputs(Ves=Ves, Type=self.Id.Type)
        if not Ves is None:
            self.Id.set_LObj([Ves.Id])
        self._Ves = Ves

    def _set_Poly(self, Poly, DLong=None, Clock=False, Sino_RefPt=None, Sino_NP=tfd.TorNP):
        if self._Done:
            Out = tfpf._get_FromItself(self, {'DLong':DLong, '_Clock':Clock})
            DLong, Clock = Out['DLong'], Out['Clock']
        tfpf._check_NotNone({'Poly':Poly, 'Clock':Clock})
        if self.Ves is not None and self.Ves.Type=='Lin' and DLong is None:
            DLong = self.Ves.DLong
        Out = _tfg_c._Ves_set_Poly(Poly, self.arrayorder, self.Type, DLong=DLong, Clock=Clock)
        self._Poly, self._NP, self._P1Max, self._P1Min, self._P2Max, self._P2Min, self._BaryP, self._BaryL, self._Surf, self._BaryS, self._DLong, self._VolLin, self._BaryV, self._Vect, self._Vin = Out

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
        return _tfg_c._Ves_isInside(self.Poly, self.Type, self.DLong, Pts, In=In)

    def plot(self, Lax=None, Proj='All', Elt='P', Pdict=None, Bsdict=tfd.TorBsd, Bvdict=tfd.TorBvd, Vdict=tfd.TorVind,
             BsdictHor=tfd.TorBsTord, BvdictHor=tfd.TorBvTord, Lim=tfd.Tor3DThetalim, Nstep=tfd.TorNTheta, LegDict=tfd.TorLegd, draw=True, a4=False, Test=True):
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
                * 'Bs': (surfacic) center of mass
                * 'Bv': (volumic) center of mass for Tor type
                * 'V': vector pointing inward perpendicular to each segment defining the polygon
        Pdict :     dict or None
            Dictionary of properties used for plotting the polygon, fed to plt.Axes.plot() or plt.plot_surface() if Proj='3d', set to ToFu_Defauts.py if None
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
        return _tfg_p.Struct_plot(self, Lax=Lax, Proj=Proj, Elt=Elt, Pdict=Pdict, Bsdict=Bsdict, Bvdict=Bvdict, Vdict=Vdict,
                                  BsdictHor=BsdictHor, BvdictHor=BvdictHor, Lim=Lim, Nstep=Nstep, LegDict=LegDict, draw=draw, a4=a4, Test=Test)

    def save(self, SaveName=None, Path=None, Mode='npz', compressed=False):
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
        tfpf.Save_Generic(self, SaveName=SaveName, Path=Path, Mode=Mode, compressed=compressed)




def _Struct_check_inputs(Id=None, Poly=None, Type=None, DLong=None, Vess=None, Clock=None, arrayorder=None, Exp=None, shot=None, dtime=None, dtimeIn=None, SavePath=None):
    if not Id is None:
        assert type(Id) in [str,tfpf.ID], "Arg Id must be a str or a tfpf.ID object !"
    if not Poly is None:
        assert hasattr(Poly,'__iter__') and np.asarray(Poly).ndim==2 and 2 in np.asarray(Poly).shape, "Arg Poly must be a dict or an iterable with 2D coordinates of cross section poly !"
    bools = [Clock,dtimeIn]
    if any([not aa is None for aa in bools]):
        assert all([aa is None or type(aa) is bool for aa in bools]), " Args [Clock,dtimeIn] must all be bool !"
    if not arrayorder is None:
        assert arrayorder in ['C','F'], "Arg arrayorder must be in ['C','F'] !"
    if not Type is None:
        assert Type in ['Tor','Lin'], "Arg Type must be in ['Tor','Lin'] !"
    if not Exp is None:
        assert Exp in tfd.AllowedExp, "Ar Exp must be in "+str(tfd.AllowedExp)+" !"
    strs = [SavePath]
    if any([not aa is None for aa in strs]):
        assert all([aa is None or type(aa) is str for aa in strs]), "Args [Type,Exp,SavePath] must all be str !"
    Iter2 = [DLong]
    if any([not aa is None for aa in Iter2]):
        assert all([aa is None or (hasattr(aa,'__iter__') and np.asarray(aa).ndim==1 and np.asarray(aa).size==2) for aa in Iter2]), "Args [DLong,Sino_RefPt] must be an iterable with len()=2 !"
    if not Vess is None:
        assert type(Vess) is Ves, "Arg Ves must be a tofu.geom.Ves instance !"
        if not Type is None:
            assert Type==Vess.Type, "Arg Ves must have same Type as the Struct instance !"
    if not dtime is None:
        assert type(dtime) is dtm.datetime, "Arg dtime must be a dtm.datetime !"

























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
        dtime      :    None / dtm.datetime
            A time reference to be used to identify this particular instance (used for debugging mostly)
        dtimeIn    :    bool
            Flag indicating whether dtime should be included in the SaveName (used for debugging mostly)

    """

    def __init__(self, Id, Du, Ves=None, Sino_RefPt=None, arrayorder='C', Clock=False, Type=None, Exp=None, Diag=None, shot=None, dtime=None, dtimeIn=False, SavePath=None):
        self._Done = False
        tfpf._check_NotNone({'Clock':Clock,'arrayorder':arrayorder})
        self._check_inputs(Clock=Clock, arrayorder=arrayorder)
        self._arrayorder = arrayorder
        self._Clock = Clock
        if not Ves is None:
            Exp = Exp if not Exp is None else Ves.Id.Exp
            assert Exp==Ves.Id.Exp, "Arg Exp must be identical to the Ves.Exp !"
        self._set_Id(Id, Type=Type, Exp=Exp, Diag=Diag, shot=shot, SavePath=SavePath, dtime=dtime, dtimeIn=dtimeIn)
        self._set_Du(Du, Calc=False)
        self._set_Ves(Ves)
        self._set_Sino(RefPt=Sino_RefPt)
        self._Done = True

    @property
    def Id(self):
        return self._Id
    @property
    def D(self):
        return self._Du[0]
    @property
    def u(self):
        return self._Du[1]
    @property
    def Du(self):
        return self._Du
    @property
    def Ves(self):
        return self._Ves
    @property
    def PIn(self):
        return self._PIn
    @property
    def POut(self):
        return self._POut
    @property
    def kPIn(self):
        return self._kPIn
    @property
    def kPOut(self):
        return self._kPOut
    @property
    def PRMin(self):
        return self._PRMin
    @property
    def Sino_RefPt(self):
        return self._Sino_RefPt
    @property
    def Sino_P(self):
        return self._Sino_P
    @property
    def Sino_Pk(self):
        return self._Sino_Pk
    @property
    def Sino_p(self):
        return self._Sino_p
    @property
    def Sino_theta(self):
        return self._Sino_theta

    def _check_inputs(self, Id=None, Du=None, Ves=None, Type=None, Sino_RefPt=None, Clock=None, arrayorder=None, Exp=None, shot=None, Diag=None, dtime=None, dtimeIn=None, SavePath=None, Calc=None):
        _LOS_check_inputs(Id=Id, Du=Du, Vess=Ves, Type=Type, Sino_RefPt=Sino_RefPt, Clock=Clock, arrayorder=arrayorder, Exp=Exp, shot=shot, Diag=Diag, dtime=dtime, dtimeIn=dtimeIn, SavePath=SavePath, Calc=Calc)


    def _set_Id(self, Val, Type=None, Exp=None, Diag=None, shot=None, SavePath=None, dtime=None, dtimeIn=False):
        if self._Done:
            Out = tfpf._get_FromItself(self.Id, {'Type':Type, 'Exp':Exp, 'shot':shot, 'Diag':Diag, 'dtime':dtime, '_dtimeIn':dtimeIn, 'SavePath':SavePath})
            Type, Exp, shot, Diag, dtime, dtimeIn, SavePath = Out['Type'], Out['Exp'], Out['shot'], Out['Diag'], Out['dtime'], Out['dtimeIn'], Out['SavePath']
        tfpf._check_NotNone({'Id':Val})
        self._check_inputs(Id=Val)
        if type(Val) is str:
            tfpf._check_NotNone({'Exp':Exp, 'shot':shot, 'Diag':Diag, 'dtimeIn':dtimeIn})
            self._check_inputs(Type=Type, Exp=Exp, shot=shot, Diag=Diag, SavePath=SavePath, dtime=dtime, dtimeIn=dtimeIn)
            Val = tfpf.ID('LOS', Val, Type=Type, Exp=Exp, Diag=Diag, shot=shot, SavePath=SavePath, dtime=dtime, dtimeIn=dtimeIn)
        self._Id = Val

    def _set_Du(self, Du, Calc=True):
        tfpf._check_NotNone({'Du':Du,'Calc':Calc})
        self._check_inputs(Du=Du, Calc=Calc)
        DD, uu = np.asarray(Du[0]).flatten(), np.asarray(Du[1]).flatten()
        uu = uu/np.linalg.norm(uu,2)
        self._Du = (DD,uu)
        if Calc:
            self._calc_InOutPolProj()

    def _set_Ves(self, Ves=None):
        tfpf._check_NotNone({'Ves':Ves, 'Exp':self.Id.Exp})
        self._check_inputs(Ves=Ves, Exp=self.Id.Exp)
        if not Ves is None:
            self.Id.set_LObj([Ves.Id])
        self._Ves = Ves
        self._calc_InOutPolProj()

    def _calc_InOutPolProj(self):
        PIn, POut, kPOut, kPIn = np.NaN*np.ones((3,)), np.NaN*np.ones((3,)), np.nan, np.nan
        if not self.Ves is None:
            PIn, POut, kPIn, kPOut, Err = _tfg_c._LOS_calc_InOutPolProj(self.Ves.Type, self.Ves.Poly, self.Ves.Vin, self.Ves.DLong, self.D, self.u, self.Id.Name)
            if Err:
                La = _tfg_p._LOS_calc_InOutPolProj_Debug(self,PIn, POut)
        self._PIn, self._POut, self._kPIn, self._kPOut = PIn, POut, kPIn, kPOut
        self._set_CrossProj()

    def _set_CrossProj(self):
        if np.isnan(self.kPIn) or np.isnan(self.kPOut):
            print('LOS '+self.Id.Name+' has no PIn or POut for computing the PolProj !')
            return
        self._PRMin, self._RMin, self._kRMin, self._PolProjAng, self._PplotOut, self._PplotIn = _tfg_c._LOS_set_CrossProj(self.Ves.Type, self.D, self.u, self.kPIn, self.kPOut)

    def _set_Sino(self, RefPt=None):
        self._check_inputs(Sino_RefPt=RefPt)
        RefPt = self.Ves.Sino_RefPt if RefPt is None else np.asarray(RefPt).flatten()
        self._Sino_RefPt = RefPt
        self._Ves._set_Sino(RefPt)
        kMax = self.kPOut
        if np.isnan(kMax):
            kMax = np.inf
        if self.Ves.Type=='Tor':
            self._Sino_P, self._Sino_Pk, self._Sino_Pr, self._Sino_PTheta, self._Sino_p, self._Sino_theta, self._Sino_Phi = _tfg_gg.Calc_Impact_Line(self.D, self.u, RefPt, kOut=kMax)
        elif self.Ves.Type=='Lin':
            self._Sino_P, self._Sino_Pk, self._Sino_Pr, self._Sino_PTheta, self._Sino_p, self._Sino_theta, self._Sino_Phi = _tfg_gg.Calc_Impact_Line_Lin(self.D, self.u, RefPt, kOut=kMax)

    def plot(self, Lax=None, Proj='All', Lplot=tfd.LOSLplot, Elt='LDIORP', EltVes='', Leg='',
            Ldict=tfd.LOSLd, MdictD=tfd.LOSMd, MdictI=tfd.LOSMd, MdictO=tfd.LOSMd, MdictR=tfd.LOSMd, MdictP=tfd.LOSMd, LegDict=tfd.TorLegd,
            Vesdict=tfd.Vesdict, draw=True, a4=False, Test=True):
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
        return _tfg_p.GLLOS_plot(self, Lax=Lax, Proj=Proj, Lplot=Lplot, Elt=Elt, EltVes=EltVes, Leg=Leg,
            Ldict=Ldict, MdictD=MdictD, MdictI=MdictI, MdictO=MdictO, MdictR=MdictR, MdictP=MdictP, LegDict=LegDict,
            Vesdict=Vesdict, draw=draw, a4=a4, Test=Test)


#    def plot_3D_mlab(self,Lplot='Tot',PDIOR='DIOR',axP='None',axT='None', Ldict=Ldict_Def,Mdict=Mdict_Def,LegDict=LegDict_Def):
#        fig = Plot_3D_mlab_GLOS()
#        return fig

    def plot_Sinogram(self, Proj='Cross', ax=None, Elt=tfd.LOSImpElt, Sketch=True, Ang=tfd.LOSImpAng, AngUnit=tfd.LOSImpAngUnit,
            Ldict=tfd.LOSMImpd, Vdict=tfd.TorPFilld, LegDict=tfd.TorLegd, draw=True, a4=False, Test=True):
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
        return _tfg_p.GLOS_plot_Sinogram(self, Proj=Proj, ax=ax, Elt=Elt, Sketch=Sketch, Ang=Ang, AngUnit=AngUnit,
            Ldict=Ldict, Vdict=Vdict, LegDict=LegDict, draw=draw, a4=a4, Test=Test)


    def save(self, SaveName=None, Path=None, Mode='npz', compressed=False):
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
        tfpf.Save_Generic(self, SaveName=SaveName, Path=Path, Mode=Mode, compressed=compressed)






def _LOS_check_inputs(Id=None, Du=None, Vess=None, Type=None, Sino_RefPt=None, Clock=None, arrayorder=None, Exp=None, shot=None, Diag=None, dtime=None, dtimeIn=None, SavePath=None, Calc=None):
    if not Id is None:
        assert type(Id) in [str,tfpf.ID], "Arg Id must be a str or a tfpf.ID object !"
    if not Du is None:
        assert hasattr(Du,'__iter__') and len(Du)==2 and all([hasattr(du,'__iter__') and len(du)==3 for du in Du]), "Arg Du must be an iterable containing of two iterables of len()=3 (cartesian coordinates) !"
    if not Vess is None:
        assert type(Vess) is Ves, "Arg Ves must be a Ves instance !"
        if not Exp is None:
            assert Exp==Vess.Id.Exp, "Arg Exp must be the same as Ves.Id.Exp !"
    bools = [Clock,dtimeIn,Calc]
    if any([not aa is None for aa in bools]):
        assert all([aa is None or type(aa) is bool for aa in bools]), " Args [Clock,dtimeIn,Calc] must all be bool !"
    if not arrayorder is None:
        assert arrayorder in ['C','F'], "Arg arrayorder must be in ['C','F'] !"
    assert Type is None, "Arg Type must be None for a LOS object !"
    if not Exp is None:
        assert Exp in tfd.AllowedExp, "Arg Exp must be in "+str(tfd.AllowedExp)+" !"
    strs = [Diag,SavePath]
    if any([not aa is None for aa in strs]):
        assert all([aa is None or type(aa) is str for aa in strs]), "Args [Diag,SavePath] must all be str !"
    Iter2 = [Sino_RefPt]
    if any([not aa is None for aa in Iter2]):
        assert all([aa is None or (hasattr(aa,'__iter__') and np.asarray(aa).ndim==1 and np.asarray(aa).size==2) for aa in Iter2]), "Args [DLong,Sino_RefPt] must be an iterable with len()=2 !"
    Ints = [shot]
    if any([not aa is None for aa in Ints]):
        assert all([aa is None or type(aa) is int for aa in Ints]), "Args [Sino_NP,shot] must be int !"
    if not dtime is None:
        assert type(dtime) is dtm.datetime, "Arg dtime must be a dtm.datetime !"





class GLOS(object):
    """ An object regrouping a group of LOS objects with some common features (e.g.: all belong to the same camera) and the same :class:`~tofu.geom.Ves` object, provides methods for common computing and plotting

    Usually :class:`LOS` correspond to detectors which are naturally grouped in 'cameras' (sets of detectors located in the same place or sharing an aperture or a data acquisition system).
    The GLOS object provided by tofu provides the object-oriented equivalent.
    The GLOS objects provides the same methods as the :class:`LOS` objects, plus extra methods for fast handling or selecting of the whole set.
    Note that you must first create each :class:`LOS` independently and then provide them as a list argument to a GLOS object.

    Parameters
    ----------
    Id :            str / tfpf.ID
        A name string or a pre-built tfpf.ID class to be used to identify this particular instance, if a string is provided, it is fed to tfpf.ID()
    LLOS :          list / :class:'LOS'
        List of LOS instances with the same :class:`~tofu.geom.Ves` instance
    Type :          None
        (not used in the current version)
    Exp :           None / str
        Experiment to which the Lens belongs, should be identical to Ves.Id.Exp if Ves is provided, if None and Ves is provided, Ves.Id.Exp is used
    Diag :          None / str
        Diagnostic to which the Lens belongs
    shot :          None / int
        Shot number from which this Lens is usable (in case its position was changed from a previous configuration)
    Sino_RefPt :    None / iterable
        If provided, array of size=2 containing the (R,Z) (for 'Tor' Type) or (Y,Z) (for 'Lin' Type) coordinates of the reference point for the sinogram
    arrayorder :    str
        Flag indicating whether the attributes of type=np.ndarray (e.g.: Poly) should be made C-contiguous ('C') or Fortran-contiguous ('F')
    SavePath :      None / str
        If provided, forces the default saving path of the object to the provided value
    dtime       None / dtm.datetime
        A time reference to be used to identify this particular instance (used for debugging mostly)
    dtimeIn     bool
        Flag indicating whether dtime should be included in the SaveName (used for debugging mostly)

    """
    def __init__(self, Id, LLOS, Ves=None, Sino_RefPt=None, Type=None, Exp=None, Diag=None, shot=None, arrayorder='C', Clock=False, dtime=None, dtimeIn=False, SavePath=None):
        self._Done = False
        tfpf._check_NotNone({'Clock':Clock,'arrayorder':arrayorder})
        self._check_inputs(Clock=Clock, arrayorder=arrayorder)
        self._arrayorder = arrayorder
        self._Clock = Clock

        self._check_inputs(Exp=Exp, Diag=Diag, shot=shot, Ves=Ves, LLOS=LLOS)
        Exp = Exp if not Exp is None else LLOS[0].Id.Exp
        assert Exp==LLOS[0].Id.Exp, "Arg Exp must be identical to the LLOS !"
        Diag = Diag if not Diag is None else LLOS[0].Id.Diag
        assert Diag==LLOS[0].Id.Diag, "Arg Diag must be identical to the LLOS !"
        shot = shot if not shot is None else LLOS[0].Id.shot
        assert shot==LLOS[0].Id.shot, "Arg shot must be identical to the LLOS !"

        self._set_Id(Id, Exp=Exp, Diag=Diag, shot=shot, Type=Type, SavePath=SavePath, dtime=dtime, dtimeIn=dtimeIn)
        self._set_LLOS(LLOS)
        if not Ves is None and not tfpf.CheckSameObj(LLOS[0].Ves, Ves, ['Poly','Type','Name','Exp']):
            self._set_Ves(Ves)

        self._set_Sino(RefPt=Sino_RefPt)
        self._Done = True


    @property
    def Id(self):
        return self._Id
    @property
    def LLOS(self):
        return self._LLOS
    @property
    def Ves(self):
        return self._LLOS[0].Ves
    @property
    def nLOS(self):
        return self._nLOS
    @property
    def Sino_RefPt(self):
        return self._LLOS[0].Sino_RefPt

    def _check_inputs(self, Id=None, LLOS=None, Ves=None, Sino_RefPt=None, Type=None, Exp=None, Diag=None, shot=None, arrayorder=None, Clock=None, dtime=None, dtimeIn=False, SavePath=None):
        _GLOS_check_inputs(Id=Id, LLOS=LLOS, Vess=Ves, Sino_RefPt=Sino_RefPt, Type=Type, Exp=Exp, Diag=Diag, shot=shot, arrayorder=arrayorder, Clock=Clock, SavePath=SavePath, dtime=dtime, dtimeIn=dtimeIn)

    def _set_Id(self, Val, Type=None, Exp=None, Diag=None, shot=None, SavePath=None, dtime=None, dtimeIn=False):
        if self._Done:
            Out = tfpf._get_FromItself(self.Id, {'Type':Type, 'Exp':Exp, 'shot':shot, 'Diag':Diag, 'dtime':dtime, '_dtimeIn':dtimeIn, 'SavePath':SavePath})
            Type, Exp, shot, Diag, dtime, dtimeIn, SavePath = Out['Type'], Out['Exp'], Out['shot'], Out['Diag'], Out['dtime'], Out['dtimeIn'], Out['SavePath']
        tfpf._check_NotNone({'Id':Val})
        self._check_inputs(Id=Val)
        if type(Val) is str:
            tfpf._check_NotNone({'Exp':Exp, 'shot':shot, 'Diag':Diag, 'dtimeIn':dtimeIn})
            self._check_inputs(Type=Type, Exp=Exp, shot=shot, Diag=Diag, SavePath=SavePath, dtime=dtime, dtimeIn=dtimeIn)
            Val = tfpf.ID('GLOS', Val, Type=Type, Exp=Exp, Diag=Diag, shot=shot, SavePath=SavePath, dtime=dtime, dtimeIn=dtimeIn)
        self._Id = Val


    def _set_LLOS(self, LLOS):
        self._check_inputs(LLOS=LLOS)
        if isinstance(LLOS,LOS):
            LLOS = [LLOS]
        self._nLOS = len(LLOS)
        self._LLOS = LLOS
        LObj = [ll.Id for ll in LLOS]
        if not LLOS[0].Ves is None:
            LObj.append(LLOS[0].Ves.Id)
        self.Id.set_LObj(LObj)

    def _set_Ves(self, Ves=None):
        self._check_inputs(Ves=V)
        for ii in range(0,self.nLOS):
            self._LLOS[ii]._set_Ves(Ves)
        if not Ves is None:
            self.Id.set_LObj([Ves.Id])

    def _set_Sino(self, RefPt=None):
        self._check_inputs(Sino_RefPt=RefPt)
        for ii in range(self.nLOS):
            self._LLOS[ii]._set_Sino(RefPt=RefPt)

    def select(self, Val=None, Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='In', Out=bool):
        """ Return the indices or instances of all instances matching the specified criterion.

        The selection can be done according to 2 different mechanism (1) and (2).

        For mechanism (1): the user provides the value (Val) that the specified criterion (Crit) should take for a :class:`LOS` to be selected.
        The criteria are typically attributes of the self.Id attribute (i.e.: name of the instance, or user-defined attributes like the camera head...)

        For mechanism (2), used if Val=None: the user provides a str expression (or a list of such) to be fed to eval(), used to check on quantitative criteria, placed before the criterion value (e.g.: 'not ' or '<=').
        Another str or list of str expressions can be provided that will be placed after the criterion value.

        Other parameters are used to specify logical operators for the selection (match any or all the criterion...) and the type of output.

        Parameters
        ----------
        Crit :      str
            Flag indicating which criterion to use for discrimination
            Can be set to any attribute of the tofu.pathfile.ID class (e.g.: 'Name','SaveName','SavePath'...) or any key of ID.USRdict (e.g.: 'Exp'...)
        Val :       list, str or None
            The value to match for the chosen criterion, can be a list of different values
            Used for selection mechanism (1)
        PreExp :    list, str or None
            A str of list of str expressions to be fed to eval(), used to check on quantitative criteria, placed before the criterion value (e.g.: 'not ')
            Used for selection mechanism (2)
        PostExp :   list, str or None
            A str of list of str expressions to be fed to eval(), used to check on quantitative criteria, placed after the criterion value (e.g.: '>=5.')
            Used for selection mechanism (2)
        Log :       str
            Flag indicating whether the criterion shall match all provided values or one of them ('any' or 'all')
        InOut :     str
            Flag indicating whether the returned indices are the ones matching the criterion ('In') or the ones not matching it ('Out')
        Out :       type / str
            Flag indicating in which form shall the result be returned, as an array of integer indices (int), an array of booleans (bool), a list of names ('Name') or a list of instances ('LOS')

        Returns
        -------
        ind :       list / np.ndarray
            The computed output (array of index, list of names or instances depending on parameter 'Out')

        Examples
        --------

        >>> import tofu.geom as tfg
        >>> ves = tfg.Ves('ves', [[0.,1.,1.,0.],[0.,0.,1.,1.]], DLong=[-1.,1.], Type='Lin', Exp='Misc', shot=0)
        >>> los1 = tfg.LOS('los1', ([0.,-0.1,-0.1],[0.,1.,1.]), Ves=ves, Exp='Misc', Diag='D', shot=0)
        >>> los2 = tfg.LOS('los2', ([0.,-0.1,-0.1],[0.,0.5,1.]), Ves=ves, Exp='Misc', Diag='D', shot=1)
        >>> los3 = tfg.LOS('los3', ([0.,-0.1,-0.1],[0.,1.,0.5]), Ves=ves, Exp='Misc', Diag='D', shot=1)
        >>> glos = tfg.GLOS('glos', [los1,los2,los3])
        >>> ind = glos.select(Val=['los1','los3'], Log='any', Out='LOS')
        >>> print [ii.Id.Name for ii in ind]
        ['los1', 'los3']
        >>> ind = glos.select(Val=['los1','los3'], Log='any', InOut='Out', Out=int)
        array([1])

        """
        if not Out=='LOS':
            ind = tfpf.SelectFromListId([ll.Id for ll in self.LLOS], Val=Val, Crit=Crit, PreExp=PreExp, PostExp=PostExp, Log=Log, InOut=InOut, Out=Out)
        else:
            ind = tfpf.SelectFromListId([ll.Id for ll in self.LLOS], Val=Val, Crit=Crit, PreExp=PreExp, PostExp=PostExp, Log=Log, InOut=InOut, Out=int)
            ind = [self.LLOS[ii] for ii in ind]
        return ind


    def plot(self, Lax=None, Proj='All', Lplot=tfd.LOSLplot, Elt='LDIORP', EltVes='', Leg='',
            Ldict=tfd.LOSLd, MdictD=tfd.LOSMd, MdictI=tfd.LOSMd, MdictO=tfd.LOSMd, MdictR=tfd.LOSMd, MdictP=tfd.LOSMd, LegDict=tfd.TorLegd,
            ind=None, Val=None, Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='In',
            Vesdict=tfd.Vesdict, draw=True, a4=False, Test=True):
        """ Plot the GLOS, with a cross-section view, a horizontal view or both, and optionally the :class:`~tofu.geom.Ves` object associated to it.

        Plot all the :class:`LOS` of the GLOS, or only a selection of them (using the same parameters as self.select()).

        Parameters
        ----------
        Lax :       list or plt.Axes
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
                * 'P': Point of used for impact parameter (i.e.: minimal distance to reference point ImpRZ)
        Lplot :     str
            Flag specifying whether to plot the full LOS ('Tot': from starting point output point) or only the fraction inside the vessel ('In': from input to output point)
        EltVes :    str
            Flag specifying the elements of the Vessel to be plotted, fed to :meth:`~tofu.geom.Ves.plot`
        Leg :       str
            Legend to be used to identify this LOS, if Leg='' the LOS name is used
        Ldict :     dict or None
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
        Lim :       list or tuple
            Array of a lower and upper limit of angle (rad.) or length for plotting the '3d' Proj
        draw :      bool
            Flag indicating whether the fig.canvas.draw() shall be called automatically
        Test :      bool
            Flag indicating whether the inputs should be tested for conformity

        ind         None or np.ndarray
            Array of indices (int or bool) of the LOS to be plotted if only some of them are to be plotted
        kwdargs
            kwdargs to be fed to GLOS.select() if ind=None and only a fraction of the LOS are to be plotted

        Returns
        -------
            La :    list or plt.Axes
                Handles of the axes used for plotting (list if several axes where used)

        """
        return _tfg_p.GLLOS_plot(self, Lax=Lax, Proj=Proj, Lplot=Lplot, Elt=Elt, EltVes=EltVes, Leg=Leg,
            Ldict=Ldict, MdictD=MdictD, MdictI=MdictI, MdictO=MdictO, MdictR=MdictR, MdictP=MdictP, LegDict=LegDict,
            Vesdict=Vesdict, draw=draw, a4=a4, Test=Test,
            ind=ind, Val=Val, Crit=Crit, PreExp=PreExp, PostExp=PostExp, Log=Log, InOut=InOut)

#    def plot_3D_mlab(self,Lplot='Tot',PDIOR='DIOR',ax='None', Ldict=Ldict_Def,Mdict=Mdict_Def,LegDict=LegDict_Def):
#        fig = Plot_3D_mlab_GLOS()
#        return fig

    def plot_Sinogram(self, Proj='Cross', ax=None, Elt=tfd.LOSImpElt, Sketch=True, Ang=tfd.LOSImpAng, AngUnit=tfd.LOSImpAngUnit,
            Ldict=tfd.LOSMImpd, Vdict=tfd.TorPFilld, LegDict=tfd.TorLegd, draw=True, a4=False, Test=True,
            ind=None, Val=None, Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='In'):
        """ Plot the sinogram of the vessel polygon, by computing its envelopp in a cross-section, can also plot a 3D version of it

        Plot all the :class:`LOS` of the GLOS, or only a selection of them in projection space

        Parameters
        ----------
        Proj :      str
            Flag indicating whether to plot a classic sinogram ('Cross') from the vessel cross-section (assuming 2D), or an extended 3D version '3d' of it with additional angle, default: 'Cross'
        ax :        None or plt.Axes
            The axes on which the plot should be done, if None a new figure and axes is created, default: None
        Elt :       str
            Flag indicating which elements to plot, each capital letter stands for one element, default: 'LV'
                * 'L': LOS
                * 'V': Vessel
        Ang :       str
            Flag indicating which angle to use for the impact parameter, the angle of the line itself (xi) or of its impact parameter (theta), default: 'theta'
        AngUnit :   str
            Flag for the angle units to be displayed, 'rad' for radians or 'deg' for degrees, default: 'rad'
        Sketch :    bool
            Flag indicating whether a small skecth showing the definitions of angles 'theta' and 'xi' should be included or not
        Ldict :     dict
            Dictionary of properties used for plotting the LOS point, fed to plt.plot() if Proj='Cross' and to plt.plot_surface() if Proj='3d', default: see ToFu_Defaults.py
        Vdict :     dict
            Dictionary of properties used for plotting the polygon envelopp, fed to plt.plot() if Proj='Cross' and to plt.plot_surface() if Proj='3d', default: see ToFu_Defaults.py
        LegDict :   None or dict
            Dictionary of properties used for plotting the legend, fed to plt.legend(), the legend is not plotted if None, default: see ToFu_Defaults.py
        draw :      bool
            Flag indicating whether to draw the figure, default: True
        Test :      bool
            Flag indicating whether the inputs shall be tested for conformity, default: True

        Returns
        -------
        ax :        plt.Axes
            The axes used to plot

        """
        return _tfg_p.GLOS_plot_Sinogram(self, Proj=Proj, ax=ax, Elt=Elt, Sketch=Sketch, Ang=Ang, AngUnit=AngUnit,
            Ldict=Ldict, Vdict=Vdict, LegDict=LegDict, draw=draw, a4=a4, Test=Test,
            ind=ind, Val=Val, Crit=Crit, PreExp=PreExp, PostExp=PostExp, Log=Log, InOut=InOut)



    def save(self,SaveName=None,Path=None,Mode='npz', compressed=False):
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
        tfpf.Save_Generic(self, SaveName=SaveName, Path=Path, Mode=Mode, compressed=compressed)




def _GLOS_check_inputs(Id=None, LLOS=None, Vess=None, Type=None, Sino_RefPt=None, Clock=None, arrayorder=None, Exp=None, shot=None, Diag=None, dtime=None, dtimeIn=None, SavePath=None):
    if not Id is None:
        assert type(Id) in [str,tfpf.ID], "Arg Id must be a str or a tfpf.ID object !"
    if not Vess is None:
        assert type(Vess) is Ves, "Arg Ves must be a Ves instance !"
        if not Exp is None:
            assert Exp==Vess.Id.Exp, "Arg Exp must be identical to Ves.Id.Exp !"
    if not arrayorder is None:
        assert arrayorder in ['C','F'], "Arg arrayorder must be in ['C','F'] !"
    if not LLOS is None:
        assert all([type(ll) is LOS for ll in LLOS]), "Arg LLOS must be a list of LOS objects !"
        assert all([ll.Id.Exp==LLOS[0].Id.Exp for ll in LLOS]), "All LOS must have the same Exp !"
        assert all([ll.Id.Type==LLOS[0].Id.Type for ll in LLOS]), "All LOS must have the same Type !"
        assert all([tfpf.CheckSameObj(LLOS[0].Ves,ll.Ves, ['Poly','Type','Name','Exp','SaveName','shot']) for ll in LLOS]), "All LOS in LLOS must have the same Ves instance !"
        assert all([ll._arrayorder==LLOS[0].Ves._arrayorder for ll in LLOS]), "All LOS should have the same arrayorder !"
        assert all([ll.Id.Diag==LLOS[0].Id.Diag for ll in LLOS]), "All LOS should have the same Diag !"
        if not arrayorder is None:
            assert LLOS[0]._arrayorder==arrayorder, "All LOS should have the same arrayorder as provided !"
    assert Type is None, "Arg Type must be None for a GLOS object !"
    bools = [Clock,dtimeIn]
    if any([not aa is None for aa in bools]):
        assert all([aa is None or type(aa) is bool for aa in bools]), " Args [Clock,dtimeIn] must all be bool !"
    if not Exp is None:
        assert Exp in tfd.AllowedExp, "Arg Exp must be in "+str(tfd.AllowedExp)+" !"
    strs = [Diag,SavePath]
    if any([not aa is None for aa in strs]):
        assert all([aa is None or type(aa) is str for aa in strs]), "Args [Diag,SavePath] must all be str !"
    Ints = [shot]
    if any([not aa is None for aa in Ints]):
        assert all([aa is None or type(aa) is int for aa in Ints]), "Args [Sino_NP,shot] must be int !"
    Iter2 = [Sino_RefPt]
    if any([not aa is None for aa in Iter2]):
        assert all([aa is None or (hasattr(aa,'__iter__') and np.asarray(aa).ndim==1 and np.asarray(aa).size==2) for aa in Iter2]), "Args [DLong,Sino_RefPt] must be an iterable with len()=2 !"
    if not dtime is None:
        assert type(dtime) is dtm.datetime, "Arg dtime must be a dtm.datetime !"







"""
###############################################################################
###############################################################################
                   Lens class and functions
###############################################################################
"""



class Lens(object):
    """ A Lens class with all geometrical data and built-in methods, defined as a planar polygon in 3D cartesian coordinates, with optional :class:`~tofu.geom.Ves` object

    A Lens object is useful for implementing one of the two possible optical arrangements available in tofu.
    A Lens (implicitly convergent) is used for focusing incoming light on a detector of reduced size (i.e.g: like the end of an optic fiber cable).
    In this case, anmd in its current version, tofu only handles spherical lenses and assumes that the detector has a circular active surface, centered on the same axis as the lens and located in its focal plane.

    Parameters
    ----------
    Id :            str or tfpf.ID
        A name string or a pre-built tfpf.ID class to be used to identify this particular instance, if a string is provided, it is fed to tfpf.ID()
    O :             iterable
        Array of 3D cartesian coordinates of the center of the Lens
    nIn :           iterable
        Array of 3D cartesian coordiantes of the vector defining the axis of the Lens
    Rad :           float
        Radius of the Lens
    F1 :            float
        Focal length of the Lens, on the detector side
    F2 :            float
        Focal length of the Lens, on the plasma side (only np.inf supported so far)
    Type :          str
        Flag indicating the type of Lens (only 'Sph' - for spherical lens - supported so far)
    R1 :            None or float
        Radius of the first face of the Lens, for full description only
    R2 :            None or float
        Radius of the second face of the Lens, for full description only
    dd :            None or float
        Width of the Lens along its axis, for full description only
    Ves :           :class:`~tofu.geom.Ves`
        :class:`~tofu.geom.Ves` object to which the aperture is assigned
    Exp :           None or str
        Experiment to which the Lens belongs, should be identical to Ves.Id.Exp if Ves is provided, if None and Ves is provided, Ves.Id.Exp is used
    Diag :          None or str
        Diagnostic to which the Lens belongs
    shot :          None or int
        Shot number from which this Lens is usable (in case its position was changed from a previous configuration)
    SavePath :      None / str
        If provided, forces the default saving path of the object to the provided value
    Clock :         bool
        Flag indicating whether the input polygon should be made clockwise (True) or counter-clockwise (False), default: False
    arrayorder :    str
        Flag indicating whether the attributes of type=np.ndarray (e.g.: Poly) should be made C-contiguous ('C') or Fortran-contiguous ('F'), default: 'C'
    dtime :         None or dtm.datetime
        A time reference to be used to identify this particular instance (used for debugging mostly), default: None
    dtimeIn :       bool
        Flag indicating whether dtime should be included in the SaveName (used for debugging mostly), default: False

    """

    def __init__(self, Id, O, nIn, Rad, F1, F2=np.inf, R1=None, R2=None, dd=None, Ves=None, Type='Sph', Exp=None, Diag=None, shot=None, arrayorder='C', Clock=False, SavePath=None, dtime=None, dtimeIn=False):

        self._Done = False
        tfpf._check_NotNone({'Clock':Clock,'arrayorder':arrayorder})
        self._check_inputs(Clock=Clock, arrayorder=arrayorder)
        self._arrayorder = arrayorder
        self._Clock = Clock

        if not Ves is None:
            Exp = Exp if not Exp is None else Ves.Id.Exp
            assert Exp==Ves.Id.Exp, "Arg Exp must be identical to the Ves.Id.Exp !"

        self._set_Id(Id, Type=Type, Exp=Exp, Diag=Diag, shot=shot, SavePath=SavePath, dtime=dtime, dtimeIn=dtimeIn)
        self._set_geom(O, nIn, Rad, F1, F2=F2, R1=R1, R2=R2, dd=dd)
        self._set_Ves(Ves)
        self._Done = True

    @property
    def Type(self):
        return self.Id.Type
    @property
    def Id(self):
        return self._Id
    @property
    def O(self):
        return self._O
    @property
    def BaryS(self):
        return self._O
    @property
    def Rad(self):
        return self._Rad
    @property
    def F1(self):
        return self._F1
    @property
    def F2(self):
        return self._F2
    @property
    def nIn(self):
        return self._nIn
    @property
    def Poly(self,NP=100):
        """ Return a simple representation of the Lens as a 3D circle (if Lens.Type='Sph') """
        assert self.Type=='Sph', "Coded only for Lens.Type='Sph' !"
        thet = np.linspace(0.,2.*np.pi,NP)
        e1 = np.array([-self.nIn[1],self.nIn[0],0.])
        e1 = e1/np.linalg.norm(e1)
        e2 = np.cross(self.nIn,e1)
        Poly = np.tile(self.O,(NP,1)).T + self.Rad*np.array([np.cos(thet)*e1[0]+np.sin(thet)*e2[0], np.cos(thet)*e1[1]+np.sin(thet)*e2[1], np.cos(thet)*e1[2]+np.sin(thet)*e2[2]])
        Poly = np.ascontiguousarray(Poly) if self._arrayorder=='C' else np.asfortranarray(Poly)
        return Poly
    @property
    def Surf(self):
        assert self.Type=='Sph', "Coded only for Lens.Type='Sph' !"
        return np.pi*self.Rad**2

    @property
    def Full(self):
        return self._Full
    @property
    def R1(self):
        return self._R1
    @property
    def R2(self):
        return self._R2
    @property
    def dd(self):
        return self._dd

    @property
    def Ves(self):
        return self._Ves

    def _check_inputs(self, Id=None, O=None, nIn=None, Rad=None, F1=None, F2=None, R1=None, R2=None, dd=None, Ves=None, Type=None, Exp=None, Diag=None, shot=None, arrayorder=None, Clock=None, SavePath=None, dtime=None, dtimeIn=None):
        _Lens_check_inputs(Id=Id, O=O, nIn=nIn, Rad=Rad, F1=F1, F2=F2, R1=R1, R2=R2, dd=dd, Vess=Ves, Type=Type, Exp=Exp, Diag=Diag, shot=shot, arrayorder=arrayorder, Clock=Clock, SavePath=SavePath, dtime=dtime, dtimeIn=dtimeIn)


    def _set_Id(self, Val, Type=None, Exp=None, Diag=None, shot=None, SavePath=None, dtime=None, dtimeIn=False):
        if self._Done:
            Out = tfpf._get_FromItself(self.Id, {'Type':Type, 'Exp':Exp, 'shot':shot, 'Diag':Diag, 'dtime':dtime, '_dtimeIn':dtimeIn, 'SavePath':SavePath})
            Type, Exp, shot, Diag, dtime, dtimeIn, SavePath = Out['Type'], Out['Exp'], Out['shot'], Out['Diag'], Out['dtime'], Out['dtimeIn'], Out['SavePath']
        tfpf._check_NotNone({'Id':Val})
        self._check_inputs(Id=Val)
        if type(Val) is str:
            tfpf._check_NotNone({'Type':Type, 'Exp':Exp, 'shot':shot, 'Diag':Diag, 'dtimeIn':dtimeIn})
            self._check_inputs(Type=Type, Exp=Exp, shot=shot, Diag=Diag, SavePath=SavePath, dtime=dtime, dtimeIn=dtimeIn)
            Val = tfpf.ID('Lens', Val, Type=Type, Exp=Exp, Diag=Diag, shot=shot, SavePath=SavePath, dtime=dtime, dtimeIn=dtimeIn)
        self._Id = Val

    def _set_geom(self, O, nIn, Rad, F1, F2=np.inf, R1=None, R2=None, dd=None):
        tfpf._check_NotNone({'O':O,'nIn':nIn,'Rad':Rad,'F1':F1,'F2':F2})
        self._check_inputs(O=O, nIn=nIn, Rad=Rad, F1=F1, F2=F2, R1=R1, R2=R2, dd=dd)
        self._O, self._nIn, self._Rad, self._F1, self._F2 = _tfg_c._Lens_set_geom_reduced(O, nIn, Rad, F1, F2=F2, Type=self.Id.Type)
        self._Full, self._R1, self._R2, self._dd, self._C1, self._C2, self._Angmax1, self._Angmax2 = _tfg_c._Lens_set_geom_full(R1=R1, R2=R2, dd=dd, O=self.O, Rad=self.Rad, nIn=self.nIn, Type=self.Id.Type)

    def _set_Ves(self,Ves):
        tfpf._check_NotNone({'Ves':Ves, 'Exp':self.Id.Exp})
        self._check_inputs(Ves=Ves, Exp=self.Id.Exp)
        if not Ves is None:
            self.Id.set_LObj([Ves.Id])
            if Ves.Type=='Tor':
                self._nIn = _tfg_c.Calc_nInFromTor_Poly(self.BaryS, self.nIn, Ves.BaryS)
            elif Ves.Type=='Lin':
                self._nIn = _tfg_c.Calc_nInFromLin_Poly(self.BaryS, self.nIn, Ves.BaryS)
            self._set_geom(self.O, self.nIn, self.Rad, self.F1, F2=self.F2, R1=self.R1, R2=self.R2, dd=self.dd)
        self._Ves = Ves


    def _get_CircleInFocPlaneFromPts(self, Pts, Test=True):
        """ Compute the image of the lens projected on its focal plane as seen from arbitrary points in the plasma, treated with 3D coordinates and reduced lens model as input

        Parameters
        ----------
        Pts :   np.ndarray
            (3,N) or (3,) array of points 3D cartesian coordinates
        Test :  bool
            Flag indicating whether the inputs should be tested for conformity, default: True

        Returns
        -------
        Cents : np.ndarray
            (3,N) array of the 3D cartesian coordinates of the centers of the image circles of the lens on the focal plane from points Pts
        Rads :  np.ndarray
            (N,) array of the radius of the image circles of the lens on the focal plane from points Pts
        d :     np.ndarray
            (N,) array of the algebraic distance along the lens axis between the Lens O-point and Pts
        r :     np.ndarray
            (N,) array of the absolute distance between Pts and the Lens axis
        rIm :   np.ndarray
            (N,) array of the absolute distance between Pts images (center of the image circles on the focal plane) and the Lens axis

        """
        Pts = np.asarray(Pts)
        if Test:
            assert self.Type=='Sph', "Can only be computed for spherical lenses, cylindrical lenses not coded yet !"
            assert type(Pts) is np.ndarray and Pts.ndim in [1,2] and 3 in Pts.shape, "Arg Pts must be a (3,N), (N,3) or (3,) np.ndarray !"
            assert out.lower() in ['2d','3d'], "Arg out must be '2D' or '3D' !"
            assert self.Id.Type=='Sph', "Only coded for spherical lens !"
        if Pts.ndim==1:
            Pts = Pts.reshape((3,1))
        if not Pts.shape[0]==3 and Pts.shape[1]==3:
            Pts = Pts.T
        Cs0,Cs1,Cs2, RadIm, din, r, rIm, nperp0,nperp1,nperp2 = _tfg_gg._Lens_get_CircleInFocPlaneFromPts(self.O[0],self.O[1],self.O[2], self.nIn[0],self.nIn[1],self.nIn[2], self.Rad, self.F1, Pts[0,:],Pts[1,:],Pts[2,:], F2=self.F2)
        return np.array([Cs0,Cs1,Cs2]), RadIm, din, r, rIm, np.array([nperp0,nperp1,nperp2])


    def plot_alone(self, ax=None, V='red', nin=1.5, nout=1., Lmax='F', V_NP=50, src=None, draw=True, a4=False, Test=True):
        """ Plot a 2D representation of the Lens object, optionally with 2D viewing cone and rays of several sources in the plane, either with reduced of full representation

        Plot a sketch of the Lens, optionally with ray-traced incoming light beams.
        This plotting routine does not consider any syurrounding and plots everything assuming the origine of the coordinate system is on the Lens

        Parameters
        ----------
        ax :    None or plt.Axes
            Axes to be used for plotting, if None a new figure with axes is created (default: None)
        V :     str
            Flag indicating whether the Lens should be considered in its reduced geometry model ('red') or its full version ('full'), default: 'red'
        nin :   float
            Value of the optical index to be used inside the Lens (useful when V='full' only)
        nout :  float
            Value of the optical index to be used outside the Lens (useful when V='full' only)
        Lmax :  float
            Maximum length on which the source beams should be plotted after going through the Lens, if 'F' all beams are plotted up to the focal plane
        V_NP :  int
            Number of points to be used to plot each circle fraction of the full version of the Lens geometry (useful when V='full' only)
        src :   None or dict
            Dictionary of parameters for the source of ray beams:
                * 'Pt':   iterable of len()=2 with the 2D cartesian coordinates of the point where the source should be located with reference to the Lens center (0,0) and axis (1,0)
                * 'Type': Flag indicating whether the source should a point ('Pt') or an array of parallel beams perpendicular to a plane passing through Pt
                * 'nn':   iterable of len()=2 with the 2D cartesian coordinates of a vector directing the array of parallel beams
                * 'NP':   int, number of beams to be plotted from the source
        draw :  bool
            Flag indicating whether the fig.canvas.draw() shall be called automatically, default: True
        a4 :    bool
            Flag indicating whether the figure should be a4 size (for printing or saving as pdf for example)
        Test :  bool
            Flag indicating whether the inputs should be tested for conformity, default: True

        Returns
        --------
        ax :    plt.Axes
            Handle of the axes used for plotting

        """
        return _tfg_p.Lens_plot_alone(self, ax=ax, V=V, nin=nin, nout=nout, Lmax=Lmax, V_NP=V_NP, src=src, draw=draw, a4=a4, Test=Test)


    def plot(self, Lax=None, Proj='All', Elt='PV', EltVes='', Leg=None, LVIn=tfd.ApLVin, Pdict=tfd.ApPd, Vdict=tfd.ApVd, Vesdict=tfd.Vesdict, LegDict=tfd.TorLegd, draw=True, a4=False, Test=True):
        """ Plot the Lens object, optionally with the associated :class:`~tofu.geom.Ves` object

        Plot the chosen projections of the Lens polygon.

        Parameters
        ----------
        Lax :       list or plt.Axes
            The axes to be used for plotting (provide a list of 2 axes if Proj='All'), if None a new figure with axes is created
        Proj :      str
            Flag specifying the kind of projection used for the plot ('Cross' for a cross-section, 'Hor' for a horizontal plane, 'All' both and '3d' for 3d)
        Elt :       str
            Flag specifying which elements to plot, each capital letter corresponds to an element
                * 'P': polygon
                * 'V': vector perpendicular to the polygon, oriented towards the interior of the Vessel
        EltVes :    str
            Flag specifying the elements of the Vessel to be plotted, fed to :meth:`~tofu.geom.Ves.plot`
        Leg :       str
            Legend to be used to identify this LOS, if Leg='' the LOS name is used
        LVIn :      float
            Length (in data coordinates, meters) of the vector 'V'
        Pdict :     dict
            Dictionary of properties used for plotting the polygon, fed to plt.Axes.plot() or plt.plot_surface() if Proj='3d', set to ToFu_Defauts.py if None (default: None)
        Vdict :     dict
            Dictionary of properties used for plotting vector 'V', fed to plt.Axes.plot()
        Vesdict :   dict
            Dictionary of kwdargs to fed to :meth:`~tofu.geom.Ves.plot`, and 'EltVes' is used instead of 'Elt'
        LegDict :   dict or None
            Dictionary of properties used for plotting the legend, fed to plt.legend(), the legend is not plotted if None
        draw :      bool
            Flag indicating whether the fig.canvas.draw() shall be called automatically
        a4 :        bool
            Flag indicating whether the figure should be a4 size (for printing or saving as pdf for example)
        Test :      bool
            Flag indicating whether the inputs should be tested for conformity

        Returns
        -------
        Lax :       list or plt.Axes
            Handles of the axes used for plotting (list if several axes where used)

        """
        return _tfg_p.LLens_plot(self, Lax=Lax, Proj=Proj, Elt=Elt, EltVes=EltVes, Leg=Leg, LVIn=LVIn, Pdict=Pdict, Vdict=Vdict, Vesdict=Vesdict, LegDict=LegDict, draw=draw, a4=a4, Test=Test)


    def save(self,SaveName=None,Path=None,Mode='npz', compressed=False):
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
        tfpf.Save_Generic(self, SaveName=SaveName, Path=Path, Mode=Mode, compressed=compressed)



def _Lens_check_inputs(Id=None, O=None, nIn=None, Rad=None, F1=None, F2=None, R1=None, R2=None, dd=None, Vess=None, Type=None, Exp=None, Diag=None, shot=None, arrayorder=None, Clock=None, SavePath=None, dtime=None, dtimeIn=None):
    if not Id is None:
        assert type(Id) in [str,tfpf.ID], "Arg Id must be a str or a tfpf.ID object !"
    floats = [Rad,F1,F2,R1,R2,dd]
    if any([not oo is None for oo in floats]):
        assert all([oo is None or type(oo) in [float,np.float64] for oo in floats]), "Args Rad, F1 and F2 must be floats !"
    if not Vess is None:
        assert type(Vess) is Ves, "Arg Ves must be a Ves instance !"
        if not Exp is None:
            assert Exp==Vess.Id.Exp, "Arg Exp must be identical to Ves.Id.Exp !"
    if not arrayorder is None:
        assert arrayorder in ['C','F'], "Arg arrayorder must be in ['C','F'] !"
    if not Type is None:
        assert Type in ['Sph'], "Arg Type must be in ['Sph'] !"
    bools = [Clock,dtimeIn]
    if any([not aa is None for aa in bools]):
        assert all([aa is None or type(aa) is bool for aa in bools]), " Args [Clock,dtimeIn] must all be bool !"
    if not Exp is None:
        assert Exp in tfd.AllowedExp, "Arg Exp must be in "+str(tfd.AllowedExp)+" !"
    strs = [Diag,SavePath]
    if any([not aa is None for aa in strs]):
        assert all([aa is None or type(aa) is str for aa in strs]), "Args [Diag,SavePath] must all be str !"
    Ints = [shot]
    if any([not aa is None for aa in Ints]):
        assert all([aa is None or type(aa) is int for aa in Ints]), "Args [Sino_NP,shot] must be int !"
    Iter3 = [O,nIn]
    if any([not aa is None for aa in Iter3]):
        assert all([aa is None or (hasattr(aa,'__iter__') and np.asarray(aa).shape==(3,)) for aa in Iter3]), "Args [O,nIn] must be an iterable with len()=3 (3D cartesian coordinates) !"
    if not dtime is None:
        assert type(dtime) is dtm.datetime, "Arg dtime must be a dtm.datetime !"









"""
###############################################################################
###############################################################################
                   Aperture class and functions
###############################################################################
"""


class Apert(object):
    """ An Aperture class with all geometrical data and built-in methods, defined as a planar polygon in 3D cartesian coordinates, with optional :class:`~tofu.geom.Ves` object

    An Apert object is useful for implementing one of the two possible optical arrangements available in tofu.
    An aperture is modelled as a planar polygon (of any non self-intersecting shape) through which light can pass (fully transparent) and around which light cannot pass (fully non-transparent).
    One of the added-values of tofu is that it allows to create several non-coplanar aperture and assign them to a single detector. It then computes automatically the volume of sight by assuming that a detectable photon should go through all apertures.

    Parameters
    ----------
    Id :        str or tfpf.ID
        A name string or a pre-built tfpf.ID class to be used to identify this particular instance, if a string is provided, it is fed to tfpf.ID()
    Poly :      np.ndarray
        An array (2,N) or (N,2) defining the contour of the aperture in 3D (X,Y,Z) cartesian coordinates, if not closed, will be closed automatically
    Ves :       :class:`~tofu.geom.Ves`
        :class:`~tofu.geom.Ves` object to which the aperture is assigned
    Type :      None or str
        Flag specifying the type of Apert
    Exp :       None or str
        Experiment to which the Lens belongs, should be identical to Ves.Id.Exp if Ves is provided, if None and Ves is provided, Ves.Id.Exp is used
    Diag :      None or str
        Diagnostic to which the Lens belongs
    shot :      None or int
        Shot number from which this Lens is usable (in case its position was changed from a previous configuration)
    SavePath :      None / str
        If provided, forces the default saving path of the object to the provided value
    Clock :     bool
        Flag indicating whether the input polygon should be made clockwise (True) or counter-clockwise (False)
    dtime :     None or dtm.datetime
        A time reference to be used to identify this particular instance (mostly used for debugging)
    dtimeIn :   bool
        Flag indicating whether dtime should be included in the SaveName (mostly used for debugging)

    """

    def __init__(self, Id, Poly, Type=None, Ves=None, Exp=None, Diag=None, shot=None, arrayorder='C', Clock=False, SavePath=None, dtime=None, dtimeIn=False):

        self._Done = False
        tfpf._check_NotNone({'Clock':Clock,'arrayorder':arrayorder})
        self._check_inputs(Clock=Clock, arrayorder=arrayorder)
        self._arrayorder = arrayorder
        self._Clock = Clock

        if not Ves is None:
            Exp = Exp if not Exp is None else Ves.Id.Exp
            assert Exp==Ves.Id.Exp, "Arg Exp must be identical to the Ves.Id.Exp !"

        self._set_Id(Id, Type=Type, Exp=Exp, Diag=Diag, shot=shot, SavePath=SavePath, dtime=dtime, dtimeIn=dtimeIn)
        self._set_Poly(Poly)
        self._set_Ves(Ves)
        self._set_arrayorder(arrayorder)
        self._Done = True

    @property
    def Id(self):
        """ Return the associated tfpf.ID object """
        return self._Id
    @property
    def Poly(self):
        """ Return the planar polygon defining the aperture (in 3D cartesian coordinates) """
        return self._Poly
    @property
    def NP(self):
        """ Return the number of points defining the polygon """
        return self._NP
    @property
    def nIn(self):
        """ Return the normalized vector perpendicular to the polygon surface and oriented towards the interior of the associated vessel (in 3D cartesian coordinates) """
        return self._nIn
    @property
    def BaryS(self):
        """ Return the (surfacic) center of mass of the polygon (in 3D cartesian coordinates) """
        return self._BaryS
    @property
    def Surf(self):
        """ Return the area of the polygon """
        return self._Surf
    @property
    def Rad(self):
        return self._Rad
    @property
    def F1(self):
        return None
    @property
    def Ves(self):
        """ Return the associated :class:`~tofu.geom.Ves` object """
        return self._Ves

    def _check_inputs(self, Id=None, Poly=None, Type=None, Ves=None, Exp=None, Diag=None, shot=None, arrayorder=None, Clock=None, SavePath=None, dtime=None, dtimeIn=None):
        _Apert_check_inputs(Id=Id, Poly=Poly, Type=Type, Vess=Ves, Exp=Exp, Diag=Diag, shot=shot, arrayorder=arrayorder, Clock=Clock, SavePath=SavePath, dtime=dtime, dtimeIn=dtimeIn)


    def _set_Id(self, Val, Type=None, Exp=None, Diag=None, shot=None, dtime=None, dtimeIn=False, SavePath=None):
        if self._Done:
            Out = tfpf._get_FromItself(self.Id, {'Type':Type, 'Exp':Exp, 'shot':shot, 'Diag':Diag, 'dtime':dtime, '_dtimeIn':dtimeIn, 'SavePath':SavePath})
            Type, Exp, shot, Diag, dtime, dtimeIn, SavePath = Out['Type'], Out['Exp'], Out['shot'], Out['Diag'], Out['dtime'], Out['dtimeIn'], Out['SavePath']
        tfpf._check_NotNone({'Id':Val})
        self._check_inputs(Id=Val)
        if type(Val) is str:
            tfpf._check_NotNone({'Exp':Exp, 'shot':shot, 'Diag':Diag, 'dtimeIn':dtimeIn})
            self._check_inputs(Type=Type, Exp=Exp, shot=shot, Diag=Diag, SavePath=SavePath, dtime=dtime, dtimeIn=dtimeIn)
            Val = tfpf.ID('Apert', Val, Type=Type, Exp=Exp, Diag=Diag, shot=shot, SavePath=SavePath, dtime=dtime, dtimeIn=dtimeIn)
        self._Id = Val

    def _set_arrayorder(self, arrayorder):
        tfpf._set_arrayorder(self, arrayorder)

    def _set_Poly(self, Poly):
        tfpf._check_NotNone({'Poly':Poly})
        self._check_inputs(Poly=Poly)
        self._Poly, self._NP, self._nIn, self._BaryP, self._Surf, self._BaryS, self._Rad =  _tfg_c._ApDetect_set_Poly(Poly, self._arrayorder, Clock=self._Clock)
        assert self._Surf>0., "Input Poly has 0 area !"

    def _set_Ves(self, Ves=None):
        tfpf._check_NotNone({'Ves':Ves, 'Exp':self.Id.Exp})
        self._check_inputs(Ves=Ves, Exp=self.Id.Exp)
        if not Ves is None:
            self.Id.set_LObj([Ves.Id])
            if Ves.Type=='Tor':
                self._nIn = _tfg_c.Calc_nInFromTor_Poly(self.BaryS, self.nIn, Ves.BaryS)
            elif Ves.Type=='Lin':
                self._nIn = _tfg_c.Calc_nInFromLin_Poly(self.BaryS, self.nIn, Ves.BaryS)
        self._Ves = Ves


    def plot(self, Lax=None, Proj='All', Elt='PV', EltVes='', Leg=None, LVIn=tfd.ApLVin, Pdict=tfd.ApPd, Vdict=tfd.ApVd,
            Vesdict=tfd.Vesdict, LegDict=tfd.TorLegd, draw=True, a4=False, Test=True):
        """ Plot the Apert, with a cross-section view, a horizontal view or both, or a 3d view, and optionally the :class:`~tofu.geom.Ves` object associated to it.

        Plot the desired projections of the polygon defining the aperture.

        Parameters
        ----------
        Lax         list or plt.Axes
            The axes to be used for plotting (provide a list of 2 axes if Proj='All'), if None a new figure with axes is created
        Proj        str
            Flag specifying the kind of projection used for the plot ('Cross' for a cross-section, 'Hor' for a horizontal plane, 'All' both and '3d' for 3d)
        Elt         str
            Flag specifying which elements to plot, each capital letter corresponds to an element
                * 'P': polygon
                * 'V': vector perpendicular to the polygon, oriented towards the interior of the Vessel
        EltVes      str
            Flag specifying the elements of the Vessel to be plotted, fed to :meth:`~tofu.geom.Ves.plot`
        Leg         str
            Legend to be used to identify this LOS, if Leg='' the LOS name is used
        LVIn        float
            Length (in data coordinates, meters) of the vector 'V'
        Pdict       dict
            Dictionary of properties used for plotting the polygon, fed to plt.Axes.plot() or plt.plot_surface() if Proj='3d', set to ToFu_Defauts.py if None
        Vdict       dict
            Dictionary of properties used for plotting vector 'V', fed to plt.Axes.plot()
        Vesdict     dict
            Dictionary of kwdargs to fed to :meth:`~tofu.geom.Ves.plot`, and 'EltVes' is used instead of 'Elt'
        LegDict     dict or None
            Dictionary of properties used for plotting the legend, fed to plt.legend(), the legend is not plotted if None
        draw        bool
            Flag indicating whether the fig.canvas.draw() shall be called automatically
        a4          bool
            Flag indicating whether the figure should be a4 size (for printing or saving as pdf for example)
        Test        bool
            Flag indicating whether the inputs should be tested for conformity

        Returns
        -------
        La          list or plt.Axes
            Handles of the axes used for plotting (list if several axes where used)

        """
        return _tfg_p.LApert_plot(self, Lax=Lax, Proj=Proj, Elt=Elt, EltVes=EltVes, Leg=Leg, LVIn=LVIn, Pdict=Pdict, Vdict=Vdict, Vesdict=Vesdict, LegDict=LegDict, draw=draw, a4=a4, Test=Test)


    def save(self,SaveName=None,Path=None,Mode='npz', compressed=False):
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
        tfpf.Save_Generic(self, SaveName=SaveName, Path=Path, Mode=Mode, compressed=compressed)



def _Apert_check_inputs(Id=None, Poly=None, Type=None, Vess=None, Exp=None, Diag=None, shot=None, arrayorder=None, Clock=None, SavePath=None, dtime=None, dtimeIn=None):
    if not Id is None:
        assert type(Id) in [str,tfpf.ID], "Arg Id must be a str or a tfpf.ID object !"
    if not Poly is None:
        assert hasattr(Poly,'__getitem__') and np.asarray(Poly).ndim==2 and 3 in np.asarray(Poly).shape, "Arg Poly must be a dict or an iterable with 3D cartesian coordinates of points !"
    if not Vess is None:
        assert type(Vess) is Ves, "Arg Ves must be a Ves instance !"
        if not Exp is None:
            assert Exp==Vess.Id.Exp, "Arg Exp must be the same as the Ves.Id.Exp !"
    if not arrayorder is None:
        assert arrayorder in ['C','F'], "Arg arrayorder must be in ['C','F'] !"
    bools = [Clock,dtimeIn]
    if any([not aa is None for aa in bools]):
        assert all([aa is None or type(aa) is bool for aa in bools]), " Args [CalcEtend,CalcSpanImp,CalcCone,CalcPreComp,Calc,Verb,Clock,dtimeIn] must all be bool !"
    if not Exp is None:
        assert Exp in tfd.AllowedExp, "Arg Exp must be in "+str(tfd.AllowedExp)+" !"
    assert Type is None, "Arg Type must be None for Apert objects !"
    strs = [Diag,SavePath]
    if any([not aa is None for aa in strs]):
        assert all([aa is None or type(aa) is str for aa in strs]), "Args [Type,Exp,Diag,SavePath] must all be str !"
    if not shot is None:
        assert type(shot) is int, "Arg shot must be a int !"
    if not dtime is None:
        assert type(dtime) is dtm.datetime, "Arg dtime must be a dtm.datetime !"




"""
###############################################################################
###############################################################################
                   Detector and GDetect classes and functions
###############################################################################
"""




class Detect(object):
    """ A Detector class with all geometrical data and built-in methods, defined as a planar polygon in 3D cartesian coordinates, with optional aperture objects

    A Detect object is at the core of tofu's added value and is mostly defined by a 3D planar polygon of any non self-intersecting shape representing the active surface of a detector.
    It can then be associated to optics (a :class:`Lens` or a list of :class:`Apert` objects) and to a :class:`~tofu.geom.Ves` to automatically compute a natural :class:'LOS' (with its etendue) and, most importantly, a proper VOS (that can be discretized for 3D numerical integration).
    It can be 2 different types: either 'Circ' if it is associated to a :class:`Lens` (in which case it is simply defined by radius and is assumed to be circular and placed at the focal plane of the :class:`Lens` object), or None in the more general case in which it is associated to a set of apertures.
    Most of the commonly used quantities are automatically calculated (etendue of the LOS, VOS...) and it comes with built-in methods for plotting and computing synthetic data.

    To compute the VOS, tofu tests all points inside a 3D grid to see if each point is visible from the detector through the apertures or not.
    The meshed space is determined by the volume spanned by a LOS sampling of the VOS.
    Then, a contour function is used to find the polygons limiting the cross-section and horizontal projections of the VOS.
    Once computed, the viewing cones are assigned to attributes of the Detect instance.

    In the particular case (1) when the LOS of the detector lies entirely inside one cross-section (e.g.: tomography diagnostics), tofu also computes the integral in the direction of the ignorable coordinate of the solid angle on a regular mesh (for faster computation of the geometry assuming toroidaly invariant basis functions).
    This regular mesh is defined in 2D, by the distance between a mesh point and the detector (k) and by the poloidal angle between the LOS and the line going from the detector to the mesh point (psi)


    Parameters
    ----------
    Id :                str or tfpf.ID
        A name string or a pre-built tfpf.ID class to be used to identify this particular instance, if a string is provided, it is fed to tfpf.ID()
    Poly :              dict or np.ndarray
        Contains the information regarding the geometry of the Detect object
            * np.ndarray: (2,N) or (N,2) defining the contour of the detector active surface in 3D (X,Y,Z) cartesian coordinates, if not closed, will be closed automatically, if Type=None
            * dict: dictionary of properties for a circular detector placed in the focal plane of a Lens on its axis, contains field 'Rad'=float (radius), if Optics is Lens and Type='Circ'
    Optics :            list or Lens
        The optics to be associated to the detector, either a spherical :class:`~tofu.geom.Lens` or a list of apertures :class:`~tofu.geom.Apert`
    Ves :               :class:`~tofu.geom.Ves` or None
        :class:`~tofu.geom.Ves` object to which the detector is assigned
    Sino_RefPt :        np.ndarray or None
        Array of size=2 containing the (R,Z) (for 'Tor' Type) or (Y,Z) (for 'Lin' Type) coordinates of the reference point for the sinogram
    CalcEtend :         bool
        Flag indicating whether to compute the etendue
    CalcSpanImp :       bool
        Flag indicating whether to compute the maximal span of the viewing volume
    CalcCone :          bool
        Flag indicating whether to compute the viewing volume or viewing cone and its two projections
    CalcPreComp :       bool
        Flag indicating whether to pre-compute a set of pre-defined points inside the viewing volume for faster computation of signal from 3D emissivity
    Calc :              bool
        Flag indicating whether to compute all the above
    Verb :              bool
        Flag indicating whether the creation of the object should be verbose (comments for each step)

    Etend_Method :      str
        Flag indicating which numerical integration to use for the computation of the etendue (picked from scipy.integrate : 'quad', 'simps', 'trapz')
    Etend_RelErr :      float
        If Etend_Method='quad', specifies the maximum relative error to be tolerated on the value of the integral (i.e.: etendue)
    Etend_dX12 :        list
        If Etend_Method in ['simps','trapz'], which implies a discretization of the plane perpendicular to the LOS, specifies the resolution of the discretization
    Etend_dX12Mode :    str
        If Etend_Method in ['simps','trapz'], specifies whether Etend_dX12 should be iunderstood as an absolute distance ('abs') or a fraction of the maximum width ('rel')
    Etend_Ratio :       float
        The numerical integration is performed on an automatically-deterimned interval, this ratio (fraction of unity) is a safety margin to increase a bit the interval and make sure all non-zero values are included
    Colis :             bool
        Flag indicating whether the collision detection mechanism should be considered when computing the VOS
    LOSRef :            str
        Key indicating which of the :class:`~tofu.geom.LOS` in the LOS dictionary should be considered as the reference LOS
    Cone_DRY :          float
        Resolution of the grid in the R (for 'Tor' vessel types) or Y (for 'Lin' vessel types) direction, in meters
    Cone_DXTheta :      float
        Resolution of the grid in the toroidal (for 'Tor' vessel types, in radians) or X (for 'Lin' vessel types, in meters) direction
    Cone_DZ :           float
        Resolution of the grid in the Z direction, in meters
    Cone_NPsi :         int
        Number of points of the regular mesh in psi direction (angle), in case (1)
    Cone_Nk :           bool
        Flag indicating whether the inputs should be tested for conformity

    Type :              None / str
        If the detector is associated to a :class:`~tofu.geom.Lens`, it should be of type 'Circ' (only circular shaped detectors are handled by tofu behind spherical lenses)
    Exp :               None or str
        Experiment to which the Lens belongs, should be identical to Ves.Id.Exp if Ves is provided, if None and Ves is provided, Ves.Id.Exp is used
    Diag :              None or str
        Diagnostic to which the Lens belongs
    shot :              None or int
        Shot number from which this Lens is usable (in case its position was changed from a previous configuration)
    SavePath :          None / str
        If provided, forces the default saving path of the object to the provided value
    Clock :             bool
        Flag indicating whether the input polygon should be made clockwise (True) or counter-clockwise (False), default: False
    arrayorder :        str
        Flag indicating whether the attributes of type=np.ndarray (e.g.: Poly) should be made C-contiguous ('C') or Fortran-contiguous ('F'), default: 'C'
    dtime :         None or dtm.datetime
        A time reference to be used to identify this particular instance (used for debugging mostly)
    dtimeIn :       bool
        Flag indicating whether dtime should be included in the SaveName (used for debugging mostly)

    """

    def __init__(self, Id, Poly, Optics=None, Ves=None, Sino_RefPt=None, CalcEtend=True, CalcSpanImp=True, CalcCone=True, CalcPreComp=True, Calc=True, Verb=True,
                 Etend_Method=tfd.DetEtendMethod, Etend_RelErr=tfd.DetEtendepsrel, Etend_dX12=tfd.DetEtenddX12, Etend_dX12Mode=tfd.DetEtenddX12Mode, Etend_Ratio=tfd.DetEtendRatio, Colis=True, LOSRef='Cart',
                 Cone_DRY=tfd.DetConeDRY, Cone_DXTheta=None, Cone_DZ=tfd.DetConeDZ, Cone_NPsi=20, Cone_Nk=60,
                 arrayorder='C', Clock=False, Type=None, Exp=None, Diag=None, shot=None, dtime=None, dtimeIn=False, SavePath=None):

        self._Done = False
        tfpf._check_NotNone({'Clock':Clock,'arrayorder':arrayorder})
        self._check_inputs(Clock=Clock, arrayorder=arrayorder)
        self._arrayorder = arrayorder
        self._Clock = Clock

        # Check consistency of Type, Exp, Diag, shot
        self._check_inputs(Poly=Poly, Type=Type, Exp=Exp, Diag=Diag, shot=shot, Ves=Ves, Optics=Optics)
        Poly, Type, Exp, Diag, shot, Ves = _Detect_set_Defaults(Poly, Type=Type, Exp=Exp, Diag=Diag, shot=shot, Ves=Ves, Optics=Optics)

        # Run all computation routines
        self._set_Id(Id, Type=Type, Exp=Exp, Diag=Diag, shot=shot, SavePath=SavePath, dtime=dtime, dtimeIn=dtimeIn)
        if Verb:
            print "TFG.Detect object "+self.Id.Name+" : Creating..."
        self._set_Poly(Poly, Calc=False)
        self._initAll()
        self._set_Optics(Optics, Calc=False)
        self._set_Ves(Ves, Calc=False)
        self._set_arrayorder(arrayorder)
        if Calc:
            self._calc_All(Sino_RefPt=Sino_RefPt, CalcEtend=CalcEtend, CalcSpanImp=CalcSpanImp, CalcCone=CalcCone, CalcPreComp=CalcPreComp, Verb=Verb,
                    Etend_Method=Etend_Method, Etend_RelErr=Etend_RelErr, Etend_dX12=Etend_dX12, Etend_dX12Mode=Etend_dX12Mode, Etend_Ratio=Etend_Ratio, LOSRef=LOSRef,
                    Cone_DRY=Cone_DRY, Cone_DXTheta=Cone_DXTheta, Cone_DZ=Cone_DZ,
                    Cone_NPsi=Cone_NPsi, Cone_Nk=Cone_Nk, Colis=Colis)

        if Verb:
            print "TFG.Detect object "+self.Id.Name+" : Created !"
        self._Done = True



    @property
    def Id(self):
        """ Return the associated tfpf.ID object """
        return self._Id
    @property
    def Poly(self):
        """ Return the planar polygon defining the aperture (in 3D cartesian coordinates) """
        if self.Id.Type=='Circ':
            NP = self.NP
            thet = np.linspace(0.,2.*np.pi,NP)
            e1 = np.array([-self.nIn[1],self.nIn[0],0.])
            e1 = e1/np.linalg.norm(e1)
            e2 = np.cross(self.nIn,e1)
            Poly = np.tile(self.BaryS,(NP,1)).T + self.Rad*np.array([np.cos(thet)*e1[0]+np.sin(thet)*e2[0], np.cos(thet)*e1[1]+np.sin(thet)*e2[1], np.cos(thet)*e1[2]+np.sin(thet)*e2[2]])
            Poly = np.ascontiguousarray(Poly) if self._arrayorder=='C' else np.asfortranarray(Poly)
            return Poly
        else:
            return self._Poly
    @property
    def Rad(self):
        """ Return the radius of the polygon (if Type='Circ', else None) """
        return self._Rad
    @property
    def NP(self):
        """ Return the number of points defining the polygon """
        return self._NP
    @property
    def nIn(self):
        """ Return the normalized vector perpendicular to the polygon surface and oriented towards the interior of the associated vessel (in 3D cartesian coordinates) """
        return self._nIn
    @property
    def BaryS(self):
        """ Return the (surfacic) center of mass of the polygon (in 3D cartesian coordinates) """
        return self._BaryS
    @property
    def Surf(self):
        """ Return the area of the polygon """
        return self._Surf
    @property
    def Ves(self):
        """ Return the associated :class:`~tofu.geom.Ves` object """
        return self._Ves
    @property
    def Optics(self):
        """ Return the list of associated Optics objects (:class:`Lens` or list of :class:`Apert`) """
        return self._Optics
    @property
    def OpticsNb(self):
        """ Return the number of associated Optics """
        return self._OpticsNb
    @property
    def OpticsType(self):
        """ Return the type of associated Optics objects """
        return self._OpticsType
    @property
    def LOS(self):
        """ Return the dictionary of associated :class:`LOS` objects """
        return self._LOS
    @property
    def Sino_RefPt(self):
        """ Return the coordinates (R,Z) or (Y,Z) for Ves of Type 'Tor' or (Y,Z) for Ves of Type 'Lin' of the reference point used to compute the sinogram """
        return self._Sino_RefPt
    @property
    def Cone_PolyCross(self):
        """ Return the polygon that is the projection in a cross-section of the viewing cone """
        return self._Cone_PolyCrossbis
    @property
    def Cone_PolyHor(self):
        """ Return the polygon that is the projection in a horizontal plane of the viewing cone """
        return self._Cone_PolyHorbis
    @property
    def SAngCross_Points(self):
        """ Return the pre-computed points of the VOS in a cross-section projection """
        return self._SAngCross_Points
    @property
    def SAngCross_Int(self):
        """ Return the integral of the solid angle at pre-computed points of the VOS in a cross-section projection """
        return self._SAngCross_Int
    @property
    def SAngHor_Points(self):
        """ Return the pre-computed points of the VOS in a horizontal projection """
        return self._SAngHor_Points
    @property
    def SAngHor_Int(self):
        """ Return the integral of the solid angle at pre-computed points of the VOS in a horizontal projection """
        return self._SAngHor_Int


    def _check_inputs(self, Id=None, Poly=None, Type=None, Optics=None, Ves=None, Sino_RefPt=None, Exp=None, Diag=None, shot=None, CalcEtend=None, CalcSpanImp=None, CalcCone=None, CalcPreComp=None, Calc=None, Verb=None,
        Etend_RelErr=None, Etend_dX12=None, Etend_dX12Mode=None, Etend_Ratio=None, Colis=None, LOSRef=None, Etend_Method=None,
        MarginRMin=None, NEdge=None, NRad=None, Nk=None,
        Cone_DRY=None, Cone_DXTheta=None, Cone_DZ=None, Cone_NPsi=None, Cone_Nk=None,
        arrayorder=None, Clock=None, SavePath=None, dtime=None, dtimeIn=None):
        _Detect_check_inputs(Id=Id, Poly=Poly, Type=Type, Optics=Optics, Vess=Ves, Sino_RefPt=Sino_RefPt, CalcEtend=CalcEtend, CalcSpanImp=CalcSpanImp, CalcCone=CalcCone, CalcPreComp=CalcPreComp, Calc=Calc, Verb=Verb,
                Etend_RelErr=Etend_RelErr, Etend_dX12=Etend_dX12, Etend_dX12Mode=Etend_dX12Mode, Etend_Ratio=Etend_Ratio, Colis=Colis, LOSRef=LOSRef, Etend_Method=Etend_Method,
                MarginRMin=MarginRMin, NEdge=NEdge, NRad=NRad, Nk=Nk,
                Cone_DRY=Cone_DRY, Cone_DXTheta=Cone_DXTheta, Cone_DZ=Cone_DZ, Cone_NPsi=Cone_NPsi, Cone_Nk=Cone_Nk,
                arrayorder=arrayorder, Clock=Clock, Exp=Exp, Diag=Diag, shot=shot, SavePath=SavePath, dtime=dtime, dtimeIn=dtimeIn)

    def _set_Id(self, Val, Type=None, Exp=None, Diag=None, shot=None, dtime=None, dtimeIn=False, SavePath=None):
        if self._Done:
            Out = tfpf._get_FromItself(self.Id, {'Type':Type, 'Exp':Exp, 'shot':shot, 'Diag':Diag, 'dtime':dtime, '_dtimeIn':dtimeIn, 'SavePath':SavePath})
            Type, Exp, shot, Diag, dtime, dtimeIn, SavePath = Out['Type'], Out['Exp'], Out['shot'], Out['Diag'], Out['dtime'], Out['dtimeIn'], Out['SavePath']
        tfpf._check_NotNone({'Id':Val})
        self._check_inputs(Id=Val)
        if type(Val) is str:
            tfpf._check_NotNone({'Exp':Exp, 'shot':shot, 'Diag':Diag, 'dtimeIn':dtimeIn})
            self._check_inputs(Type=Type, Exp=Exp, shot=shot, Diag=Diag, SavePath=SavePath, dtime=dtime, dtimeIn=dtimeIn)
            Val = tfpf.ID('Detect', Val, Type=Type, Exp=Exp, Diag=Diag, shot=shot, SavePath=SavePath, dtime=dtime, dtimeIn=dtimeIn)
        self._Id = Val

    def _set_Poly(self, Poly, Calc=True, CalcEtend=True, CalcSpanImp=True, CalcCone=True, CalcPreComp=True, NPDef=100):
        tfpf._check_NotNone({'Poly':Poly})
        if self._Done and self.OpticsType=='Lens':
            self._check_inputs(Poly=Poly, Optics=self.Optics[0], Calc=Calc, CalcEtend=CalcEtend, CalcSpanImp=CalcSpanImp, CalcCone=CalcCone, CalcPreComp=CalcPreComp)
        else:
            self._check_inputs(Poly=Poly, Calc=Calc, CalcEtend=CalcEtend, CalcSpanImp=CalcSpanImp, CalcCone=CalcCone, CalcPreComp=CalcPreComp)
        self._Poly, self._NP, self._nIn, self._BaryP, self._Surf, self._BaryS, self._Rad =  _tfg_c._ApDetect_set_Poly(Poly, Type=self.Id.Type, arrayorder=self._arrayorder, Clock=self._Clock, NP=NPDef)
        assert self._Surf>0., "Input Poly has 0 area !"
        if Calc:
            self._calc_All(CalcEtend=CalcEtend, CalcSpanImp=CalcSpanImp, CalcCone=CalcCone, CalcPreComp=CalcPreComp)

    def _initAll(self):
        self._Ves = None
        self._Optics, self._nOptics = None, 0
        self._SAngPlane = None
        self._LOS_ApertPolyInt, self._LOS_ApertPolyInt_S, self._LOS_ApertPolyInt_BaryS, self._LOS, self._TorAngRef, self._LOS_NP = None, None, None, None, None, None
        self._Sino_RefPt, self._Sino_CrossProj, self._LOSRef =  None, None, None
        self._Span_R, self._Span_Theta, self._Span_X, self._Span_Y, self._Span_Z, self._Span_k, self._Span_NEdge, self._Span_NRad = None, None, None, None, None, None, None, None
        self._Cone_PolyCross, self._Cone_PolyHor, self._Cone_PolyCrossbis, self._Cone_PolyHorbis = None, None, None, None
        self._Cone_Poly_DR, self._Cone_Poly_DZ, self._Cone_Poly_DTheta, self._Cone_Poly_NEdge, self._Cone_Poly_NRad = None, None, None, None, None
        self._Cone_PolyCross_RefLCorners, self._Cone_PolyCross_RefLBary, self._Cone_PolyCross_RefdMax = None, None, None
        self._Cone_PolyHor_RefLCorners, self._Cone_PolyHor_RefLBary, self._Cone_PolyHor_RefdMax = None, None, None
        self._SAngCross_Points, self._SAngHor_Points, self._SAngCross_Max, self._SAngHor_Max, self._SAngCross_Int, self._SAngHor_Int = None, None, None, None, None, None
        self._SAngCross_Reg, self._SAngCross_Reg_K, self._SAngCross_Reg_Psi, self._SAngCross_Reg_Int = False, None, None, None
        # Parameters of Synthetic diagnostics
        self._SynthDiag_Done = False
        self._SynthDiag_ds, self._SynthDiag_dsMode, self._SynthDiag_MarginS, self._SynthDiag_dX12, self._SynthDiag_dX12Mode, self._SynthDiag_Colis = None, None, None, None, None, None
        self._reset_SynthDiag()
        # Parameters of Resolution computing
        self._reset_Res()

    def _set_Optics(self, Optics=None, Calc=True, CalcEtend=True, CalcSpanImp=True, CalcCone=True, CalcPreComp=True):
        Polytemp = {'Rad':self.Rad} if self.Id.Type=='Circ' else self.Poly
        self._check_inputs(Poly=Polytemp, Optics=Optics, Calc=Calc, CalcEtend=CalcEtend, CalcSpanImp=CalcSpanImp, CalcCone=CalcCone, CalcPreComp=CalcPreComp)
        if not Optics is None:
            Optics = Optics if type(Optics) is list else [Optics]
            self._Optics = Optics
            self._OpticsNb = len(Optics)
            self._OpticsType = "Lens" if type(Optics[0]) is Lens else "Apert"
            self.Id.set_LObj([aa.Id for aa in Optics])
            self._set_Optics_Lens_Cone()
            if Calc:
                self._calc_All(CalcEtend=CalcEtend, CalcSpanImp=CalcSpanImp, CalcCone=CalcCone, CalcPreComp=CalcPreComp)

    def _set_Optics_Lens_Cone(self):
        if self.OpticsType == "Lens":
            self._Optics_Lens_ConeHalfAng = np.arctan2(self.Rad,self.Optics[0].F1)
            self._Optics_Lens_ConeTip = self.Optics[0].O - self.Optics[0].nIn * self.Optics[0].Rad * self.Optics[0].F1 / self.Rad
        else:
            self._Optics_Lens_ConeTip = None
            self._Optics_Lens_ConeHalfAng = None


    def _set_Ves(self, Ves, Calc=True, CalcEtend=True, CalcSpanImp=True, CalcCone=True, CalcPreComp=True):
        self._check_inputs(Ves=Ves, Calc=Calc, CalcEtend=CalcEtend, CalcSpanImp=CalcSpanImp, CalcCone=CalcCone, CalcPreComp=CalcPreComp)
        if not Ves is None:
            self._Ves = Ves
            self.Id.set_LObj([Ves.Id])
            if Ves.Type=='Tor':
                self._nIn = _tfg_c.Calc_nInFromTor_Poly(self.BaryS, self.nIn, Ves.BaryS)
            elif Ves.Type=='Lin':
                self._nIn = _tfg_c.Calc_nInFromLin_Poly(self.BaryS, self.nIn, Ves.BaryS)
            if Calc:
                self._calc_All(CalcEtend=CalcEtend, CalcSpanImp=CalcSpanImp, CalcCone=CalcCone, CalcPreComp=CalcPreComp)

    def _set_arrayorder(self, arrayorder):
        tfpf._set_arrayorder(self, arrayorder)

    def _calc_All(self, Sino_RefPt=None, CalcEtend=True, CalcSpanImp=True, CalcCone=True, CalcPreComp=True,
            Etend_Method=tfd.DetEtendMethod, Etend_RelErr=tfd.DetEtendepsrel, Etend_dX12=tfd.DetEtenddX12, Etend_dX12Mode=tfd.DetEtenddX12Mode, Etend_Ratio=tfd.DetEtendRatio, Colis=tfd.DetCalcEtendColis, LOSRef='Cart',
            Cone_DRY=tfd.DetConeDRY, Cone_DXTheta=None, Cone_DZ=tfd.DetConeDZ, Cone_NPsi=20, Cone_Nk=60, Verb=True):

        self._check_inputs(Sino_RefPt=Sino_RefPt, CalcEtend=CalcEtend, CalcSpanImp=CalcSpanImp, CalcCone=CalcCone, CalcPreComp=CalcPreComp,
                Etend_Method=Etend_Method, Etend_RelErr=Etend_RelErr, Etend_dX12=Etend_dX12, Etend_dX12Mode=Etend_dX12Mode, Etend_Ratio=Etend_Ratio, LOSRef=LOSRef,
                Cone_DRY=Cone_DRY, Cone_DXTheta=Cone_DXTheta, Cone_DZ=Cone_DZ, Cone_NPsi=Cone_NPsi, Cone_Nk=Cone_Nk, Colis=Colis)

        assert self.OpticsNb>0 and not self.Ves is None, "Calculation of [LOS, Etendue, Span and Cone] not possible without Optics and Ves !"
        self._set_SAngPnPe1e2()
        self._set_LOS(CalcEtend=CalcEtend, Method=Etend_Method, RelErr=Etend_RelErr, dX12=Etend_dX12, dX12Mode=Etend_dX12Mode, Ratio=Etend_Ratio, Colis=Colis, LOSRef=LOSRef, Verb=Verb)
        self._set_SinoSpan(CalcSpanImp=CalcSpanImp, Sino_RefPt=Sino_RefPt)
        self._set_ConeWidthAlongLOS()
        self._set_ConePoly(CalcCone=CalcCone, DRY=Cone_DRY, DXTheta=Cone_DXTheta, DZ=Cone_DZ, NPsi=Cone_NPsi, Nk=Cone_Nk)
        self.set_SigPrecomp(CalcPreComp=CalcPreComp)

    def _set_SAngPnPe1e2(self):
        if not self.Optics is None:
            #sca = np.array([np.sum((aa.BaryS-self.BaryS)*self.nIn) for aa in self.LApert])
            sca = np.array([aa.Surf for aa in self.Optics])
            ind = np.argmax(sca)
            e1, e2 = _tfg_gg.Calc_DefaultCheck_e1e2_PLane_1D(self.Optics[ind].BaryS, self.Optics[ind].nIn)
            self._SAngPlane = (self.Optics[ind].BaryS, self.Optics[ind].nIn, e1, e2)

    def _set_LOS(self, CalcEtend=True, Method=tfd.DetEtendMethod, RelErr=tfd.DetEtendepsrel, dX12=tfd.DetEtenddX12, dX12Mode=tfd.DetEtenddX12Mode, Ratio=tfd.DetEtendRatio, Colis=tfd.DetCalcEtendColis, LOSRef='Cart', Verb=True):
        self._check_inputs(CalcEtend=CalcEtend, Etend_Method=Method, Etend_RelErr=RelErr, Etend_dX12=dX12, Etend_dX12Mode=dX12Mode, Etend_Ratio=Ratio, Colis=Colis, LOSRef=LOSRef, Verb=Verb)
        if not (self.Ves is None or self.Optics is None):
            #try:
            self._LOS_ApertPolyInt, self._LOS_ApertPolyInt_S, self._LOS_ApertPolyInt_BaryS, du = _tfg_c._Detect_set_LOS(self.Id.Name, [oo.Surf for oo in self.Optics], [oo.BaryS for oo in self.Optics],
                    [oo.nIn for oo in self.Optics], [oo.Poly for oo in self.Optics], self.BaryS, self.Poly, OpType=self.OpticsType, Verb=Verb, Test=True)
            LOSCart = LOS(self.Id.Name+"_Cart", (self.BaryS,du), Ves=self.Ves, Exp=self.Id.Exp, Diag=self.Id.Diag, shot=self.Id.shot,
                    dtime=self.Id.dtime, dtimeIn=self.Id._dtimeIn, SavePath=self.Id.SavePath)
            PRef = (LOSCart.POut+LOSCart.PIn)/2.
            self._LOS = {'Cart':{'LOS':LOSCart,'PRef':PRef}}
            self._LOSRef = LOSRef
            if CalcEtend:
                self._set_Etendue(Method=Method, RelErr=RelErr, dX12=dX12, dX12Mode=dX12Mode, Ratio=Ratio, Colis=Colis)
            #except:
            #    self._LOS = "Impossible !"
        else:
            self._LOS = "Impossible !"

    def _set_Etendue(self, Method=tfd.DetEtendMethod, RelErr=tfd.DetEtendepsrel, dX12=tfd.DetEtenddX12, dX12Mode=tfd.DetEtenddX12Mode, Ratio=tfd.DetEtendRatio, Colis=tfd.DetCalcEtendColis):    # Pb with Lens quad vs trapz !
        self._check_inputs(Etend_Method=Method, Etend_RelErr=RelErr, Etend_dX12=dX12, Etend_dX12Mode=dX12Mode, Etend_Ratio=Ratio, Colis=Colis)
        if not self.LOS in ["Impossible !",None]:
            print "    "+self.Id.Name+" : Computing Entendue..."
            LOPolys = [oo.Poly for oo in self.Optics]
            LOnIns = [oo.nIn for oo in self.Optics]
            LSurfs = [oo.Surf for oo in self.Optics]
            LOBaryS = [oo.BaryS for oo in self.Optics]

            for kk in self.LOS.keys():
                self.LOS[kk]['Etend_0Dir'] = self.Surf * _tfg_gg.Calc_SAngVect_LPolys1Point_Flex([self._LOS_ApertPolyInt], self.BaryS, self._SAngPlane[0], self._SAngPlane[1], self._SAngPlane[2], self._SAngPlane[3])[0]
                self.LOS[kk]['Etend_0Inv'] = self._LOS_ApertPolyInt_S * _tfg_gg.Calc_SAngVect_LPolys1Point_Flex([self.Poly], self._LOS_ApertPolyInt_BaryS, self.BaryS, self._SAngPlane[1], self._SAngPlane[2], self._SAngPlane[3])[0]
                PRef, LOSu = self.LOS[kk]['PRef'], self.LOS[kk]['LOS'].u
                e1, e2 = _tfg_gg.Calc_DefaultCheck_e1e2_PLane_1D(PRef, LOSu)

                self.LOS[kk]['Etend'] = _tfg_c.Calc_Etendue_PlaneLOS(PRef.reshape((3,1)), LOSu.reshape((3,1)),
                        self.Poly, self.BaryS, self.nIn, LOPolys, LOnIns, LSurfs, LOBaryS, self._SAngPlane,
                        self.Ves.Poly, self.Ves._Vin, DLong=self.Ves.DLong,
                        Lens_ConeTip = self._Optics_Lens_ConeTip, Lens_ConeHalfAng=self._Optics_Lens_ConeHalfAng, RadL=self.Optics[0].Rad, RadD=self.Rad, F1=self.Optics[0].F1,
                        OpType=self.OpticsType, VType=self.Ves.Type, Mode=Method, e1=e1.reshape((3,1)), e2=e2.reshape((3,1)), epsrel=RelErr, Ratio=Ratio, dX12=dX12, dX12Mode=dX12Mode, Colis=Colis, Test=True)[0][0]

                self.LOS[kk]['Etend_Method'], self.LOS[kk]['Etend_Ratio'], self.LOS[kk]['Etend_Colis'] = Method, Ratio, Colis
                self.LOS[kk]['Etend_RelErr'] = RelErr if Method=='quad' else None
                self.LOS[kk]['Etend_dX12'] = None if Method=='quad' else dX12
                self.LOS[kk]['Etend_dX12Mode'] = None if Method=='quad' else dX12Mode
        else:
            warnings.warn("Detect "+ self.Id.Name +" : calculation of Etendue not possible because LOS impossible !")


    def _set_SinoSpan(self, Sino_RefPt=None, CalcSpanImp=True, MarginRMin=tfd.DetSpanRMinMargin, NEdge=tfd.DetSpanNEdge, NRad=tfd.DetSpanNRad):
        self._check_inputs(Sino_RefPt=Sino_RefPt, CalcSpanImp=CalcSpanImp, MarginRMin=MarginRMin, NEdge=NEdge, NRad=NRad)
        if CalcSpanImp and not (self.LOS=='Impossible !' or self.LOS is None):
            print "    "+self.Id.Name+" : Computing Span and Sinogram..."
            if Sino_RefPt is None:
                Sino_RefPt = self.Ves.BaryS
            Sino_RefPt = np.asarray(Sino_RefPt).flatten()
            for kk in self.LOS.keys():
                self.LOS[kk]['LOS']._set_Sino(RefPt=Sino_RefPt)
            P, nP, e1, e2 = self._SAngPlane
            LOPolys = [oo.Poly for oo in self.Optics]
            LOBaryS = [oo.BaryS for oo in self.Optics]
            LOSD, LOSu = self.LOS[self._LOSRef]['LOS'].D, self.LOS[self._LOSRef]['LOS'].u
            if self.Ves.Type=='Tor':
                RMinMax, ThetaMinMax, ZMinMax, kMinMax, Sino_CrossProj, Span_NEdge, Span_NRad = _tfg_c.Calc_SpanImpBoth_2Steps(self.Poly, self.NP, self.BaryS, LOPolys, LOBaryS, LOSD, LOSu, Sino_RefPt, P, nP,
                        self.Ves.Poly, self.Ves.Vin, DLong=self.Ves.DLong, VType=self.Ves.Type, e1=e1, e2=e2, OpType=self.OpticsType, Lens_ConeTip=self._Optics_Lens_ConeTip, NEdge=NEdge, NRad=NRad, Test=True)
                RMinMax[0] = np.max(np.array([MarginRMin*RMinMax[0],self.Ves._P1Min[0]]))
                self._Sino_RefPt, self._Span_R, self._Span_Theta, self._Span_Z, self._Span_k = Sino_RefPt, RMinMax, ThetaMinMax, ZMinMax, kMinMax
                self._Span_X, self._Span_Y = None, None
            elif self.Ves.Type=='Lin':
                XMinMax, YMinMax, ZMinMax, kMinMax, Sino_CrossProj, Span_NEdge, Span_NRad = _tfg_c.Calc_SpanImpBoth_2Steps(self.Poly, self.NP, self.BaryS, LOPolys, LOBaryS, LOSD, LOSu, Sino_RefPt, P, nP,
                        self.Ves.Poly, self.Ves.Vin, DLong=self.Ves.DLong, VType=self.Ves.Type, e1=e1, e2=e2, OpType=self.OpticsType, Lens_ConeTip=self._Optics_Lens_ConeTip, NEdge=NEdge, NRad=NRad, Test=True)
                self._Sino_RefPt, self._Span_X, self._Span_Y, self._Span_Z, self._Span_k = Sino_RefPt, XMinMax, YMinMax, ZMinMax, kMinMax
                self._Span_R, self._Span_Theta = None, None
            self._Sino_CrossProj, self._Span_NEdge, self._Span_NRad = Sino_CrossProj, Span_NEdge, Span_NRad
            # Sino_CrossProj = Imp_PolProj

    def _set_ConeWidthAlongLOS(self,Nk=10):
        self._check_inputs(Nk=Nk)
        if not (self.LOS=='Impossible !' or self.LOS is None or self._Span_k is None):
            k = np.linspace(self._Span_k[0],self._Span_k[1],Nk)
            P, u = self.LOS[self._LOSRef]['LOS'].D, self.LOS[self._LOSRef]['LOS'].u
            e1, e2 = _tfg_gg.Calc_DefaultCheck_e1e2_PLane_1D(P, u)
            Ps = np.array([P[0]+k*u[0], P[1]+k*u[1], P[2]+k*u[2]])
            nPs = np.tile(u,(Nk,1)).T
            e1s, e2s = np.tile(e1,(Nk,1)).T, np.tile(e2,(Nk,1)).T
            LOPolys = [oo.Poly for oo in self.Optics]
            if self.OpticsType=='Apert':
                LOSurfs = [oo.Surf for oo in self.Optics]
                LOnIns = [oo.nIn for oo in self.Optics]
                LnPtemp = np.asarray(LOnIns)*np.tile(LOSurfs,(3,1)).T
                MinX1, MinX2, MaxX1, MaxX2, e1, e2 = _tfg_c.Calc_ViewConePointsMinMax_PlanesDetectApert_2Steps(self.Poly, LOPolys, LnPtemp, LOSurfs, self.Optics[0].BaryS, Ps, nPs, e1=e1s, e2=e2s, Test=True)
            elif self.OpticsType=='Lens':
                MinX1, MinX2, MaxX1, MaxX2, e1, e2 = _tfg_c.Calc_ViewConePointsMinMax_PlanesDetectLens(LOPolys[0], self._Optics_Lens_ConeTip, Ps, nPs, e1=e1s, e2=e2s, Test=True)
            self._ConeWidth_k = k
            self._ConeWidth_X1 = np.array([MinX1,MaxX1])
            self._ConeWidth_X2 = np.array([MinX2,MaxX2])
            self._ConeWidth = np.min(np.array([np.diff(self._ConeWidth_X1,axis=0),np.diff(self._ConeWidth_X2,axis=0)]),axis=0).flatten()

    def _set_Sino(self,RefPt=None):
        self._check_inputs(Sino_RefPt=Sino_RefPt)
        self._Ves._set_Sino(RefPt)
        self._set_SinoSpan(RefPt)

    def _isOnGoodSide(self, Pts, NbPoly=None, Log='all'):
        """ Check whether each point is on the inside or the outside of each Detect and Apert (with respect to nIn) """
        return _tfg_c._Detect_isOnGoodSide(Pts, self.BaryS, self.nIn, [oo.BaryS for oo in self.Optics], [oo.nIn for oo in self.Optics], NbPoly=NbPoly, Log=Log)

    def _isInsideConeWidthLim(self, Pts):
        """ Check whether each point lies inside the enveloppe of the viewing cone """
        return _tfg_c._Detect_isInsideConeWidthLim(Pts, self.LOS[self._LOSRef]['LOS'].D, self.LOS[self._LOSRef]['LOS'].u, self._ConeWidth_k, self._ConeWidth_X1, self._ConeWidth_X2)

    def _set_ConePoly(self, CalcCone=True, DRY=tfd.DetConeDRY, DXTheta=None, DZ=tfd.DetConeDZ, NPsi=20, Nk=60, Test=True):
        """ If CalcCone is True, compute the projections of the VOS, also called viewing cones elsewhere in the documentation

        To compute the VOS, tofu tests all points inside a 3D grid to see if each point is visible from the detector through the apertures or not.
        The meshed space is determined by the volume spanned by a LOS sampling of the VOS.
        Then, a contour function is used to find the polygons limiting the cross-section and horizontal projections of the VOS.
        Once computed, the viewing cones are assigned to attributes of the Detect instance.

        In the particular case (1) when the LOS of the detector lies entirely inside one cross-section (e.g.: tomography diagnostics), tofu also computes the integral in the direction of the ignorable coordinate of the solid angle on a regular mesh (for faster computation of the geometry assuming toroidaly invariant basis functions).
        This regular mesh is defined in 2D, by the distance between a mesh point and the detector (k) and by the poloidal angle between the LOS and the line going from the detector to the mesh point (psi)

        Parameters
        ----------
        DRY :       float
            Resolution of the grid in the R (for 'Tor' vessel types) or Y (for 'Lin' vessel types) direction, in meters
        DXTheta :    float
            Resolution of the grid in the toroidal (for 'Tor' vessel types, in radians) or X (for 'Lin' vessel types, in meters) direction
        DZ :        float
            Resolution of the grid in the Z direction, in meters
        NPsi :      int
            Number of points of the regular mesh in psi direction (angle), in case (1)
        Nk :        int
            Number of points of the regular mesh in k direction (distance), in case (1)
        Test :      bool
            Flag indicating whether the inputs should be tested for conformity

        """

        if CalcCone and not (self.LOS=='Impossible' or self.LOS is None):
            print "    "+self.Id.Name+" : Computing ConePoly..."
            DPoly, DBaryS, DnIn = self.Poly, self.BaryS, self.nIn
            LOPolys = [oo.Poly for oo in self.Optics]
            LOnIns = [oo.nIn for oo in self.Optics]
            LSurfs = [oo.Surf for oo in self.Optics]
            LOBaryS = [oo.BaryS for oo in self.Optics]
            LOSD, LOSu = self.LOS[self._LOSRef]['LOS'].D, self.LOS[self._LOSRef]['LOS'].u
            LOSPIn, LOSPOut = self.LOS[self._LOSRef]['LOS'].PIn, self.LOS[self._LOSRef]['LOS'].POut

            self._SAngCross_Reg, self._SAngCross_Reg_Int, self._SAngCross_Reg_K, self._SAngCross_Reg_Psi, self._SAngCross_Points, self._SAngCross_Max, self._SAngCross_Int, self._SAngHor_Points, self._SAngHor_Max, self._SAngHor_Int, self._Cone_PolyCross, self._Cone_PolyHor, self._Cone_PolyCrossbis, self._Cone_PolyHorbis, self._Cone_Poly_DX, self._Cone_Poly_DY, self._Cone_Poly_DR, self._Cone_Poly_DTheta, self._Cone_Poly_DZ \
                    = _tfg_c._Detect_set_ConePoly(DPoly, DBaryS, DnIn, LOPolys, LOnIns, LSurfs, LOBaryS, self._SAngPlane, LOSD, LOSu, LOSPIn, LOSPOut, self._Span_k,
                            Span_R=self._Span_R, Span_Theta=self._Span_Theta, Span_X=self._Span_X, Span_Y=self._Span_Y, Span_Z=self._Span_Z,
                            ConeWidth_k=self._ConeWidth_k, ConeWidth_X1=self._ConeWidth_X1, ConeWidth_X2=self._ConeWidth_X2, Lens_ConeTip=self._Optics_Lens_ConeTip, Lens_ConeHalfAng=self._Optics_Lens_ConeHalfAng,
                            RadD=self.Rad, RadL=self.Optics[0].Rad, F1=self.Optics[0].F1, VPoly=self.Ves.Poly, VVin=self.Ves.Vin, DLong=self.Ves.DLong,
                            VType=self.Ves.Type, OpType=self.OpticsType, NPsi=NPsi, Nk=Nk, thet=np.linspace(0.,2.*np.pi,DPoly.shape[1]),
                            DXTheta=DXTheta, DRY=DRY, DZ=DZ, Test=True)


    def _get_KPsiCrossInt(self,PtsRZ):
        """ Computes k and psi for a set of points in cross-section (R,Z) or (Y,Z) coordinates """
        return _tfg_c._Detect_get_KPsiCrossInt(PtsRZ, SAngCross_Reg=self._SAngCross_Reg, LOSPOut=self.LOS['Cart']['LOS'].POut, DBaryS=self.BaryS, VType=self.Ves.Type)



    def refine_ConePoly(self, dMax=tfd.DetConeRefdMax, Proj='Cross', indPoly=0, Verb=True, Test=True):
        """ Reduce the number of points of the selected Cone_Poly projection using the provided maximum distance and checking for convexity

        Provide a built-in method to simplify the 2 projections of the viewing cone (VOS).
        In its raw form, the projection of the VOS is a polygon with potentially a high number of points (computed using matplotlib._cntr() function).
        A re-sampled version of this polygon is computed by taking its convex hull and checking, for each edge, how far it is from the original edge.
        Each edge (2 points) of the convex hull is then compared to the set of original edges it encloses.
        If the maximum distance between this convex hull-derived edge and the original set of edges is smaller than dMax, then the convex hull-derived egde is used, otherwise the original edges are preserved.
        The method does not return a value, instead it assigns the new polygon to a dedicated attribute of the object, thus ensuring that both the original and the re-sampled projections of the VOS are available.

        Parameters
        ----------
        dMax :      float
            Threshold absolute distance that limits the acceptable discrepancy between the original polygon and its convex hull (checked for each edge of the convex hull)
        Proj :      str
            Flag indicating to which projection of the VOS the method should be applied
        indPoly :   int
            Index of the polygon to be treated (i.e.: in case one projection of the VOS results in a list of several polygons instead of just one polygon as is usually the case)
        Verb :      bool
            Flag indicating whether a one-line comment should be printed at the end of the calculation giving the number of points of the new polygon vs the number of points of the original polygon
        Test :      bool
            Flag indicating whether the inputs should be tested for conformity

        """
        assert Proj in ['Cross','Hor'], "Arg Proj must be in ['Cross','Hor'] !"
        assert type(indPoly) is int and indPoly<=max(len(self._Cone_PolyCrossbis),len(self._Cone_PolyHorbis)), "Arg indPoly must be a valid int index !"
        Poly = np.copy(self._Cone_PolyCross[indPoly]) if Proj=='Cross' else np.copy(self._Cone_PolyHor[indPoly])
        PP = _tfg_c.Refine_ConePoly_All(Poly, dMax=dMax)
        if Proj=='Cross':
            self._Cone_PolyCrossbis[indPoly], self._Cone_PolyCross_dMax = PP, dMax
        else:
            self._Cone_PolyHorbis[indPoly], self._Cone_PolyHor_dMax = PP, dMax
        if Verb:
            print "        "+self.Id.Name+".refine_ConePoly('"+Proj+"') : from ", Poly.shape[1], "to", PP.shape[1], "points"

    def isInside(self, Points, In='(X,Y,Z)', Test=True):
        """ Return an array of indices indicating whether each point lies both in the cross-section and horizontal porojections of the viewing cone

        Like for the :class:`~tofu.geom.Ves` object, points can be provided in 2D or 3D coordinates (specified by 'In'), and an array of booleans is returned.

        Parameters
        ----------
        Points :    np.ndarray
            (2,N) or (3,N) array of coordinates of the N points to be tested
        In :        str
            Flag indicating in which coordinate system the Points are provided, must be in ['(R,Z)','(Y,Z)','(X,Y)','(X,Y,Z)','(R,phi,Z)']
                * '(R,Z)': All points are assumed to lie in the horizontal projection, for 'Tor' vessel type only
                * '(Y,Z)': All points are assumed to lie in the horizontal projection, for 'Lin' vessel type only
                * '(X,Y)': All points are assumed to lie in the cross-section projection
        Test :      bool
            Flag indicating whether the inputs should be tested for conformity

        Returns
        -------
        ind :       np.ndarray
            (N,) array of booleans with True if a point lies inside both projections of the viewing cone

        """
        assert not self.LOS=='Impossible !', "The detected volume is zero !"
        TorAngRef = np.arctan2(self.LOS[self._LOSRef]['PRef'][1],self.LOS[self._LOSRef]['PRef'][0]) if self.Ves.Type=='Tor' else None
        return _tfg_c._Detect_isInside(self._Cone_PolyCrossbis, self._Cone_PolyHorbis, Points, In=In, VType=self.Ves.Type, TorAngRef=TorAngRef, Test=Test)


    def calc_SAngVect(self, Pts, In='(X,Y,Z)', Colis=tfd.DetCalcSAngVectColis, Test=True):
        """ Return the Solid Angle of the Detect-Apert system as seen from the specified points, including collisions detection or not

        Compute the solid angle and the directing vector subtended by the Detect-Optics system as seen from the desired points (provided in the specified coordinates).
        This can be useful for visualizing the solid angle distribution or for computing synthetic signal from simulated emissivity in a 3D numerical integration manner.
        The automtic detection of collisions with the edges of the :class:`~tofu.geom.Ves` object can be switched off (not recommended).

        Parameters
        ----------
        Pts :   np.ndarray
            (2,N) or (3,N) array of coordinates of the provided N points
        In :    str
            Flag indicating in which coordinate system the Pts are provided, must be in ['(R,Z)','(X,Y,Z)','(R,phi,Z)']
        Colis : bool
            Flag indicating whether collision detection should be activated
        Test :  bool
            Flag indicating whether the inputs should be tested for conformity

        Returns
        -------
        SAng :  np.ndarray
            (N,) array of floats, the computed solid angles

        """
        if Test:
            assert isinstance(Pts,np.ndarray) and Pts.ndim==2 and Pts.shape[0] in [2,3], "Arg Pts must be a 2D np.ndarray !"
        CrossRef = np.arctan2(self.LOS[self._LOSRef]['PRef'][1],self.LOS[self._LOSRef]['PRef'][0]) if self.Ves.Type=='Tor' else self.LOS[self._LOSRef]['PRef'][0]
        Pts = _tfg_gg.CoordShift(Pts, In=In, Out='(X,Y,Z)', CrossRef=CrossRef)
        LOPolys = [oo.Poly for oo in self.Optics]
        LOnIns = [oo.nIn for oo in self.Optics]
        LOBaryS = [oo.BaryS for oo in self.Optics]
        return _tfg_c._Detect_SAngVect_Points(Pts, DPoly=self.Poly, DBaryS=self.BaryS, DnIn=self.nIn, LOBaryS=LOBaryS, LOnIns=LOnIns, LOPolys=LOPolys, SAngPlane=self._SAngPlane, Lens_ConeTip=self._Optics_Lens_ConeTip, Lens_ConeHalfAng=self._Optics_Lens_ConeHalfAng, RadL=self.Optics[0].Rad, RadD=self.Rad, F1=self.Optics[0].F1, thet=np.linspace(0.,2.*np.pi,self.NP), OpType=self.OpticsType, VPoly=self.Ves.Poly, VVin=self.Ves.Vin, DLong=self.Ves.DLong, VType=self.Ves.Type, Cone_PolyCrossbis=self._Cone_PolyCrossbis, Cone_PolyHorbis=self._Cone_PolyHorbis, TorAngRef=CrossRef, Colis=Colis, Test=Test)


    def _get_SAngIntMax(self, Proj='Cross', SAng='Int'):
        """ Get the Int or Max of the SAng in a cross-section or horizontal projection """
        CrossRef = np.arctan2(self.LOS[self._LOSRef]['PRef'][1],self.LOS[self._LOSRef]['PRef'][0]) if self.Ves.Type=='Tor' else self.LOS[self._LOSRef]['PRef'][0]
        return _tfg_c._Detect_get_SAngIntMax(SAngCross_Reg=self._SAngCross_Reg, SAngCross_Points=self._SAngCross_Points, SAngCross_Reg_K=self._SAngCross_Reg_K, SAngCross_Reg_Psi=self._SAngCross_Reg_Psi, SAngCross_Reg_Int=self._SAngCross_Reg_Int, SAngCross_Int=self._SAngCross_Int, SAngCross_Max=self._SAngCross_Max, SAngHor_Points=self._SAngHor_Points, SAngHor_Int=self._SAngHor_Int, SAngHor_Max=self._SAngHor_Max, Cone_PolyCrossbis=self._Cone_PolyCrossbis, Cone_PolyHorbis=self._Cone_PolyHorbis, TorAngRef=CrossRef, DBaryS=self.BaryS, LOSPOut=self.LOS[self._LOSRef]['LOS'].POut, Proj=Proj, SAng=SAng, VType=self.Ves.Type)



    def set_SigPrecomp(self, CalcPreComp=True, dX12=None, dX12Mode=None, ds=None, dsMode=None, MarginS=None, Colis=None):
        """ Precompute a 3D grid for fast integration of a 3D emissivity for a synthetic diagnostic approach

        In order to accelerate the computation of synthetic signal from simulated emissivity, it is possible to pre-compute a discretisation of the VOS (mesh points + solid angle) and store it as an attribute of the Detect object.
        While such pre-computation does speed-up significantly the numerical integration, it also burdens the object with heavy attributes that can make it too big to save.
        Hence, the saving method has a special argument that allows to specify that these pre-computed attributes should not be saved but should instead be re-computed automatically when loading the file.
        The parameters dX12, dX12Mode, ds and dsMode give the user control over how fine the discretization of the VOS should be, which affects both the accuracy of the numerical integration and the size of the resulting mesh.

        Parameters
        ----------
        CalcPreComp :   bool
            Flag indicating whether the pre-computation should be run
        dX12 :          list
            Array of the 2 resolutions to be used to define the grid in a plane perpendicular to the LOS
        dX12Mode :      str
            Flag specifying whether the values in dX12 are absolute distances or relative values (i.e. fraction of the total width [0;1])
        ds :            float
            Float indicating the resolution in the longitudinal direction
        dsMode :        str
            Flag specifying whether ds is an absolute distance or relative (i.e. fraction of the total length [0;1])
        MarginS :       float
            Float specifying
        Colis :         bool
            Flag indicating whether collision detection should be used

        """
        if CalcPreComp and not (self.LOS=='Impossible !' or self.LOS is None):
            print "    "+self.Id.Name+" : Pre-computing 3D matrix for synthetic diag..."

            LOPolys = [oo.Poly for oo in self.Optics]
            LOBaryS = [oo.BaryS for oo in self.Optics]
            LOnIns = [oo.nIn for oo in self.Optics]
            LOSD = self.LOS[self._LOSRef]['LOS'].D
            LOSu = self.LOS[self._LOSRef]['LOS'].u
            thet = np.linspace(0.,2.*np.pi,self.Poly.shape[1])
            CrossRef = np.arctan2(self.LOS[self._LOSRef]['PRef'][1],self.LOS[self._LOSRef]['PRef'][0]) if self.Ves.Type=='Tor' else self.LOS[self._LOSRef]['PRef'][0]

            if self._SynthDiag_Done:
                dX12 = tfd.DetSynthdX12 if dX12 is None else self._SynthDiag_dX12
                dX12Mode = tfd.DetSynthdX12Mode if dX12Mode is None else self._SynthDiag_dX12Mode
                ds = tfd.DetSynthds if ds is None else self._SynthDiag_ds
                dsMode = tfd.DetSynthdsMode if dsMode is None else self._SynthDiag_dsMode
                MarginS = tfd.DetSynthMarginS if MarginS is None else self._SynthDiag_MarginS
                Colis = tfd.DetCalcSAngVectColis if Colis is None else self._SynthDiag_Colis
            else:
                dX12 = tfd.DetSynthdX12 if dX12 is None else dX12
                dX12Mode = tfd.DetSynthdX12Mode if dX12Mode is None else dX12Mode
                ds = tfd.DetSynthds if ds is None else ds
                dsMode = tfd.DetSynthdsMode if dsMode is None else dsMode
                MarginS = tfd.DetSynthMarginS if MarginS is None else MarginS
                Colis = tfd.DetCalcSAngVectColis if Colis is None else Colis

            Out = _tfg_c._Detect_set_SigPrecomp(self.Poly, self.BaryS, self.nIn, LOPolys, LOBaryS, LOnIns, self._SAngPlane, LOSD=LOSD, LOSu=LOSu, Span_k=self._Span_k, ConeWidth_k=self._ConeWidth_k, ConeWidth_X1=self._ConeWidth_X1,
                    ConeWidth_X2=self._ConeWidth_X2, Cone_PolyCrossbis=self._Cone_PolyCrossbis, Cone_PolyHorbis=self._Cone_PolyHorbis,
                    Lens_ConeTip=self._Optics_Lens_ConeTip, Lens_ConeHalfAng=self._Optics_Lens_ConeHalfAng, RadL=self.Optics[0].Rad, RadD=self.Rad, F1=self.Optics[0].F1, thet=thet,
                    VPoly=self.Ves.Poly, VVin=self.Ves.Vin, DLong=self.Ves.DLong, CrossRef=CrossRef, dX12=dX12, dX12Mode=dX12Mode, ds=ds, dsMode=dsMode, MarginS=MarginS, VType=self.Ves.Type, OpType=self.OpticsType, Colis=Colis)
            self._SynthDiag_Points, self._SynthDiag_SAng, self._SynthDiag_Vect, self._SynthDiag_dV = Out[0], Out[1], Out[2], Out[3]
            self._SynthDiag_ds, self._SynthDiag_dsMode, self._SynthDiag_MarginS, self._SynthDiag_dX12, self._SynthDiag_dX12Mode, self._SynthDiag_Colis = Out[4], Out[5], Out[6], Out[7], Out[8], Out[9]
            self._SynthDiag_Done = True

    def _reset_SynthDiag(self):
        self._SynthDiag_Points, self._SynthDiag_SAng, self._SynthDiag_Vect, self._SynthDiag_dV = None, None, None, None

    def calc_Sig(self, ff, extargs={}, Method='Vol', Mode='simps', PreComp=True,
            epsrel=tfd.DetSynthEpsrel, dX12=tfd.DetSynthdX12, dX12Mode=tfd.DetSynthdX12Mode, ds=tfd.DetSynthds, dsMode=tfd.DetSynthdsMode, MarginS=tfd.DetSynthMarginS, Colis=tfd.DetCalcSAngVectColis,  Test=True):
        """ Return the signal computed from an input emissivity function, using a 3D or LOS method

        The synthetic signal resulting from a simulated emissivity can be computed automatically in several ways.
        The user can choose between a VOS and a LOS approach (volume integration or line integration with etendue).
        In each case the user can choose between the numerical integration method (from scipy.integrate + np.sum()).
        It is possible to specify that, for a VOS approach, you want to use the pre-conputed mesh for faster computation (see :meth:`~tofu.geom.Detect.set_SigPrecomp`).
        For a VOS approach, the user can specify how fine the discretization should be.
        The collision detection with the edges of the :class:`~tofu.geom.Ves` object can be switched off (not recommended).

        Parameters
        ----------
        ff :        function
            Input emissiviy function, should take one input as follows:
                * ff(Pts), where Points is a np.ndarray of shape=(3,N), with the (X,Y,Z) coordinates of any N number of points
        Method :    str
            Flag indicating whether the spatial integration should be done with a volume ('Vol') or a LOS ('LOS') approach
        Mode :      str
            Flag indicating the numerical integration method in ['quad','simps','trapz','nptrapz','sum']
        PreComp :   bool
            Flag indicating whether the pre-computed grid should be used
        epsrel :    float
            Float specifying the tolerated relative error on the numerical integration, used for 'quad'
        dX12 :      list
            Array of the 2 resolutions to be used to define the grid in a plane perpendicular to the LOS
        dX12Mode :  str
            Flag specifying whether the values in dX12 are absolute distances or relative values (i.e. fraction of the total width [0;1])
        ds :        float
            Float indicating the resolution in the longitudinal direction
        dsMode :    str
            Flag specifying whether ds is an absolute distance or relative (i.e. fraction of the total length [0;1])
        Colis :     bool
            Flag indicating whether collision detection should be used
        Test :      bool
            Flag indicating whether the inputs should be tested for conformity

        Returns
        --------
        Sig :       float
            The computed signal

        """
        # * ff(Pts, Vect), where Vect is a np.ndarray of shape=(3,N) with the (X,Y,Z) coordinates of a vector indicating the direction in which photons are emitted
        if PreComp and not Method=='LOS':
            assert not self._SynthDiag_ds is None, "The precomputed matrix shall be computed before using it..... "

        LOPolys = [oo.Poly for oo in self.Optics]
        LOBaryS = [oo.BaryS for oo in self.Optics]
        LOnIn = [oo.nIn for oo in self.Optics]
        LOSD = self.LOS[self._LOSRef]['LOS'].D
        LOSu = self.LOS[self._LOSRef]['LOS'].u
        LOSkPIn = self.LOS[self._LOSRef]['LOS'].kPIn
        LOSkPOut = self.LOS[self._LOSRef]['LOS'].kPOut
        LOSEtend = self.LOS[self._LOSRef]['Etend']
        thet = np.linspace(0.,2.*np.pi,self.Poly.shape[1])
        CrossRef = np.arctan2(self.LOS[self._LOSRef]['PRef'][1],self.LOS[self._LOSRef]['PRef'][0]) if self.Ves.Type=='Tor' else self.LOS[self._LOSRef]['PRef'][0]

        Sig = _tfg_c._Detect_SigSynthDiag(ff, extargs=extargs, Method=Method, Mode=Mode, PreComp=PreComp,
            DPoly=self.Poly, DBaryS=self.BaryS, DnIn=self.nIn, LOPolys=LOPolys, LOBaryS=LOBaryS, LOnIn=LOnIn, Lens_ConeTip=self._Optics_Lens_ConeTip, Lens_ConeHalfAng=self._Optics_Lens_ConeHalfAng,
            RadL=self.Optics[0].Rad, RadD=self.Rad, F1=self.Optics[0].F1, thet=thet, OpType=self.OpticsType,
            LOSD=LOSD, LOSu=LOSu, LOSkPIn=LOSkPIn, LOSkPOut=LOSkPOut, LOSEtend=LOSEtend, Span_k=self._Span_k, ConeWidth_X1=self._ConeWidth_X1, ConeWidth_X2=self._ConeWidth_X2, SAngPlane=self._SAngPlane, CrossRef=CrossRef,
            Cone_PolyCrossbis=self._Cone_PolyCrossbis, Cone_PolyHorbis=self._Cone_PolyHorbis, VPoly=self.Ves.Poly,  VVin=self.Ves.Vin, VType=self.Ves.Type,
            SynthDiag_Points=self._SynthDiag_Points, SynthDiag_SAng=self._SynthDiag_SAng, SynthDiag_Vect=self._SynthDiag_Vect, SynthDiag_dV=self._SynthDiag_dV,
            SynthDiag_dX12=self._SynthDiag_dX12, SynthDiag_dX12Mode=self._SynthDiag_dX12Mode, SynthDiag_ds=self._SynthDiag_ds,
            SynthDiag_dsMode=self._SynthDiag_dsMode, SynthDiag_MarginS=self._SynthDiag_MarginS, SynthDiag_Colis=self._SynthDiag_Colis,
            epsrel=epsrel, dX12=dX12, dX12Mode=dX12Mode, ds=ds, dsMode=dsMode, MarginS=MarginS, Colis=Colis, Test=Test)
        return Sig

    def _debug_Etendue_BenchmarkRatioMode(self, RelErr=tfd.DetEtendepsrel, Ratio=[0.01,0.05,0.2,0.5], Modes=['simps','trapz','quad'], dX12=[0.002,0.002], dX12Mode='abs', Colis=tfd.DetCalcEtendColis):
        """ Return the etendue computed 3 different numerical integration methods, with or without collisions, with more or less margin for the perpendicular plane size (Ratio)

        USed for debugging, compute the etendue with various values of the Ratio (extra margin for the integration intervals), to check it does not affect the computed value

        Parameters
        ----------
        RelErr :        float
            Float specifying the tolerated relative error on the numerical integration, used for 'quad'
        Ratio :         list
            Array of values in [0,1] specifying margin to be used to define the edges of the perpendicular plane
        Colis :         bool
            Flag indicating whether collision detection should be used

        Returns
        -------
        EtendSimps :    np.ndarray
            (NLos,NRatio) array of the computed etendues with numerical integration 'simps', where NLos is the number of LOS of Detect and NRatio=len(Ratio)
        EtendTrapz :    np.ndarray
            (NLos,NRatio) array of the computed etendues with numerical integration 'trapz', where NLos is the number of LOS of Detect and NRatio=len(Ratio)
        EtendQuad :     np.ndarray
            (NLos,NRatio) array of the computed etendues with numerical integration 'quad', where NLos is the number of LOS of Detect and NRatio=len(Ratio)
        Keys :          list
            List of the available LOS

        """
        if not self.LOS=='Impossible !':
            Keys = self.LOS.keys()
            NLOS, NR = len(Keys), len(Ratio)
            Etends = {}
            for kk in Modes:
                Etends[kk] = np.nan*np.ones((NLOS,NR))
                for jj in range(0,NLOS):
                    PRef = self.LOS[Keys[jj]]['PRef']
                    LOSu = self.LOS[Keys[jj]]['LOS'].u
                    e1, e2 = _tfg_gg.Calc_DefaultCheck_e1e2_PLane_1D(PRef, LOSu)
                    PRef = PRef.reshape((3,1))
                    LOSu = LOSu.reshape((3,1))
                    e1, e2 = e1.reshape((3,1)), e2.reshape((3,1))
                    LOPolys = [oo.Poly for oo in self.Optics]
                    LOBaryS = [oo.BaryS for oo in self.Optics]
                    LOnIn = [oo.nIn for oo in self.Optics]
                    LOSurfs = [oo.Surf for oo in self.Optics]
                    for ii in range(0,len(Ratio)):
                        print "    ...Computing Etendue with integration method", kk, " for LOS ", Keys[jj], " and Ratio=", Ratio[ii]
                        Etends[kk][jj,ii] = _tfg_c.Calc_Etendue_PlaneLOS(PRef, LOSu, self.Poly, self.BaryS, self.nIn, LOPolys, LOnIn, LOSurfs, LOBaryS, self._SAngPlane, self.Ves.Poly, self.Ves.Vin, DLong=self.Ves.DLong,
                                Lens_ConeTip = self._Optics_Lens_ConeTip, Lens_ConeHalfAng=self._Optics_Lens_ConeHalfAng, RadL=self.Optics[0].Rad, RadD=self.Rad, F1=self.Optics[0].F1,
                                OpType=self.OpticsType, VType=self.Ves.Type, Mode=kk, e1=e1, e2=e2, epsrel=RelErr, Ratio=Ratio[ii], dX12=dX12, dX12Mode=dX12Mode, Colis=Colis, Test=True)[0][0]
            return Etends, Ratio, RelErr, dX12, dX12Mode, Colis


    def calc_Etendue_AlongLOS(self, Length='', NP=tfd.DetEtendOnLOSNP, Modes=['trapz','quad'], RelErr=tfd.DetEtendepsrel, dX12=tfd.DetSynthdX12, dX12Mode=tfd.DetSynthdX12Mode, Ratio=tfd.DetEtendRatio,
            Colis=tfd.DetSAngColis, LOSRef=None, Test=True):
        """ Return the etendue computed at different points along the LOS, with various numerical methods, with or without collision detection

        Computing the etendue along the LOS of a Detect object can be useful for checking whether the etendue is constant (as it should be if the LOS approximation is to be used).
        Cases with non-constant etendue include in particular partially obstructed VOS in the divertor region of Tokamaks.
        Also useful for debugging: if the etendue is not constant but the VOS is not obstructed, something might be wrong with the computation of the etendue or with the model (e.g.: for Lens optics).
        Indeed, the model implemented for a Lens is ideal, but a close look at the etendue shows that the model is not perfect (but sufficiently accurate for most uses though).

        Parameters
        ----------
        Length :    str
            Flag indicating whether to use the full length of the VOS (including partially obstructed parts: ''), or just the length of the LOS unil its exit point ('LOS').
        NP :        int
            Number of points (uniformly distributed along the LOS) where the etendue should be computed
        Modes :     list or str
            Flag or list of flags indicating which numerical integration methods shoud be used in ['quad','simps','trapz']
        RelErr :    float
            For 'quad', a positive float defining the relative tolerance allowed
        dX12 :      list
            For 'simps' or 'trapz', a list of 2 floats defining the resolution of the sampling in X1 and X2
        dX12Mode :  str
            For 'simps' or'trapz', 'rel' or 'abs', if 'rel' the resolution dX12 is in dimensionless units in [0;1] (hence a value of 0.1 means 10 discretisation points between the extremes), if 'abs' dX12 is in meters
        Ratio :     float
            A float specifying the relative margin to be taken for integration boundaries
        Colis :     bool
            Flag indicating whether collision detection should be used
        LOSRef :    None or str
            Flag indicating which LOS should be used
        Test :      bool
            Flag indicating whether the inputs should be tested for conformity

        Returns
        -------
        Etend :     np.ndarray
            Computed etendues
        Pts :       np.ndarray
            (3,NP) array specifying the 3D (X,Y,Z) coordinates of the points along the LOS where the etendue was computed
        kPts :      np.ndarray
            (NP,) array of the distance-coordinate k along the LOS
        LOSRef :    str
            The LOS that was used

        """
        if Test:
            assert Modes in ['quad','simps','trapz'] or (type(Modes) is list and all([mode in ['quad','simps','trapz'] for mode in Modes])), "Arg Modes must be a list of Modes to be used ('quad', 'simps' or 'trapz') !"
        if type(Modes) is str:
            Modes = [Modes]
        NMod = len(Modes)

        if not self.LOS=='Impossible !':
            LOSRef = self._LOSRef if LOSRef is None else LOSRef
            RelErr = self.LOS[LOSRef]['Etend_RelErr'] if 'quad' in Modes and RelErr is None else RelErr
            dX12 = self.LOS[LOSRef]['Etend_dX12'] if (not (Modes=='quad' or Modes==['quad'])) and dX12 is None else dX12
            dX12Mode = self.LOS[LOSRef]['Etend_dX12Mode'] if (not (Modes=='quad' or Modes==['quad'])) and dX12Mode is None else dX12Mode
            Ratio = self.LOS[LOSRef]['Etend_Ratio'] if Ratio is None else Ratio
            Etends = {}

            PRef = self.LOS[LOSRef]['PRef']
            LOSu = self.LOS[LOSRef]['LOS'].u
            e1, e2 = _tfg_gg.Calc_DefaultCheck_e1e2_PLane_1D(PRef, LOSu)
            LOPolys = [oo.Poly for oo in self.Optics]
            LOBaryS = [oo.BaryS for oo in self.Optics]
            LOnIn = [oo.nIn for oo in self.Optics]
            LOSurfs = [oo.Surf for oo in self.Optics]

            k1 = self.LOS[LOSRef]['LOS'].kPIn if Length=='LOS' else self._Span_k[0]
            k2 = self.LOS[LOSRef]['LOS'].kPOut if Length=='LOS' else self._Span_k[1]
            k = np.linspace(k1,k2,NP)
            P1 = self.LOS[LOSRef]['LOS'].D
            Ps = np.array([P1[0] + k*LOSu[0], P1[1] + k*LOSu[1], P1[2] + k*LOSu[2]])
            nPs = np.tile(LOSu,(NP,1)).T
            e1, e2 = np.tile(e1,(NP,1)).T, np.tile(e2,(NP,1)).T
            for ii in range(0,NMod):
                print "    ...Computing Etendues of "+ self.Id.Name +" for ",NP," planes with integration method ",Modes[ii]
                Etends[Modes[ii]] = _tfg_c.Calc_Etendue_PlaneLOS(Ps, nPs, self.Poly, self.BaryS, self.nIn, LOPolys, LOnIn, LOSurfs, LOBaryS, self._SAngPlane,
                        self.Ves.Poly, self.Ves.Vin, DLong=self.Ves.DLong, Lens_ConeTip=self._Optics_Lens_ConeTip, Lens_ConeHalfAng=self._Optics_Lens_ConeHalfAng, RadL=self.Optics[0].Rad, RadD=self.Rad, F1=self.Optics[0].F1,
                        OpType=self.OpticsType, VType=self.Ves.Type, Mode=Modes[ii], e1=e1, e2=e2, epsrel=RelErr, Ratio=Ratio, dX12=dX12, dX12Mode=dX12Mode, Colis=Colis, Test=True)[0]
            return Etends, Ps, k, LOSRef


    def calc_SAngNb(self, Pts=None, Proj='Cross', Slice='Int', DRY=None, DXTheta=None, DZ=None, Colis=tfd.DetSAngColis):
        """ Compute the solid angle subtended by the Detect+Optics system as seen for desired points, in a slice or integrated manner

        Mostly useful in the :class:`GDetect` object when there are several detectors.
        Computes, for each point in the desired projection, the total solid angle subtended by all the detectors (or its integral) and the number of detectors that 'see' each point.

        Parameters
        ----------
        Pts :       None / np.ndarray
            (3,N) array of cartesian (X,Y,Z) coordinates of the provided N points, if None a default set of points is computed according to DRY, DXTheta and DZ
        Proj :      str
            Flag indicating to which projection of the VOS the method should be applied
        Slice :     str
            Flag indicating whether to compute the solid angle ('Slice'), the maximum solid angle along the ignorable coordinate ('Max'), or the integral over the ignorable coordinate ('Int')
        DRY :       None / float
            Resolution (in horizontal direction of the cross-section) of the mesh to be constructed if the points are not provided
        DXTheta :   None / float
            Resolution (in ignorable coordinate direction) of the mesh to be constructed if the points are not provided
        DZ :        None / float
            Resolution (in vertical direction) of the mesh to be constructed if the points are not provided
        Colis :     bool
            Flag indicating whether collision detection should be used

        Returns
        -------
        SA :        np.ndarray
            Array of (ND,NP) solid angle values, where ND is the number of detectors and NP the number of points
        Nb :        np.ndarray
            Array of (ND,NP) booleans, True if a point is seen by a detector
        Pts :       np.ndarray
            The computed points (in case they were not provided)

        """
        # Return pre-computed data if matches
        if all([ss is None for ss in [Pts,DRY,DXTheta,DZ]]) and Slice in ['Int','Max']:
            if Proj=='Cross':
                SA = self._SAngCross_Int if Slice=='Int' else self._SAngCross_Max
                Pts = self._SAngCross_Points
            elif Proj=='Hor':
                SA = self._SAngHor_Int if Slice=='Int' else self._SAngHor_Max
                Pts = self._SAngHor_Points
        else:
            # Get the mesh if Pts not provided
            if Pts is None:
                LOS = self.LOS[self._LOSRef]['LOS']
                SingPts = np.vstack((LOS.PIn, LOS.PIn+0.002*LOS.u, 0.5*(LOS.POut+LOS.PIn), LOS.POut-0.002*LOS.u , LOS.POut)).T
                if self.Ves.Type=='Tor':
                    X1, XIgn, Z, NX1, NIgn, NZ, Pts, out = _tfg_c._get_CrossHorMesh(SingPoints=SingPts, LSpan_R=[self._Span_R], LSpan_Theta=[self._Span_Theta], LSpan_Z=[self._Span_Z], DR=DRY, DTheta=DXTheta, DZ=DZ,
                            VType=self.Ves.Type, Proj=Proj, ReturnPts=True)
                elif self.Ves.Type=='Lin':
                    XIgn, X1, Z, NIgn, NX1, NZ, Pts, out = _tfg_c._get_CrossHorMesh(SingPoints=SingPts, LSpan_X=[self._Span_X], LSpan_Y=[self._Span_Y], LSpan_Z=[self._Span_Z], DX=DXTheta, DY=DRY, DZ=DZ,
                            VType=self.Ves.Type, Proj=Proj, ReturnPts=True)
            # Get the Solid angle (itself, or Int or Max)
            if Slice in ['Int','Max']:
                FF = self._get_SAngIntMax(Proj=Proj, SAng=Slice)
                SA = FF(Pts, In=out)
            else:
                if Proj=='Hor':
                    Span = self._Span_Z
                else:
                    Span = self._Span_Theta if self.Ves.Type=='Tor' else self._Span_X
                assert Slice>=Span[0] and Slice<=Span[1], "Arg Slice is outside of the interval were non-zeros values can be found !"
                Ptsint = _tfg_gg.CoordShift(Pts, In=out, Out='(X,Y,Z)', CrossRef=Slice)
                SA = self.calc_SAngVect(Ptsint, In='(X,Y,Z)', Colis=Colis, Test=True)[0]
        return SA, SA>0., Pts


    def _calc_Res(self, Pts=None, CrossMesh=[0.01,0.01], CrossMeshMode='abs', Mode='Iso', Amp=1., Deg=0, steps=0.001, Thres=0.05, ThresMode='rel', ThresMin=0.01,
                 IntResCross=[0.1,0.1], IntResCrossMode='rel', IntResLong=0.05, IntResLongMode='rel',
                 Eq=None, PlotDetail=False, Cdict=dict(tfd.DetConed), Ntt=100, SaveName=None, SavePath='./', save=False, Test=True):
        """ Compute the resolution and given input points or grid knots, with specified method and accuracy, the result can be automatically saved (useful for long computations)

        The definition that tofu proposes for the spatial resolution of a tomography diagnostic is as follows:
        (describe after publication)

        Parameters
        ----------
        Pts :               None or iterable
        CrossMesh :         None or iterable
        CrossMeshMode :     str
        Mode :              str
        Amp :               float
        Deg :               int
        steps :             float
        Thres :             float
        ThresMode :         str
        ThresMin :          float
        IntResCross :       iterable
        IntResCrossMode :   str
        IntResLong :        float
        IntResLongMode :    str
        Eq :                None or callable
        PlotDetail :        bool
        Cdict :             dict
        Ntt :               int
        SaveName :          None or str
        SavePath :          str
        save :              bool
        Test :              bool

        Returns
        -------
        Res :               np.ndarray
        Pts :               np.ndarray

        """

        Res, Pts, LDetLim, Mode, steps, Thres, ThresMode, ThresMin, IntResCross, IntResCrossMode, IntResLong, IntResLongMode \
                = _Calc_Resolution(self, Pts=Pts, CrossMesh=CrossMesh, CrossMeshMode=CrossMeshMode, Mode=Mode, Amp=Amp, Deg=Deg, steps=steps, Thres=Thres, ThresMode=ThresMode, ThresMin=ThresMin,
                                  IntResCross=IntResCross, IntResCrossMode=IntResCrossMode, IntResLong=IntResLong, IntResLongMode=IntResLongMode,
                                  Eq=Eq, PlotDetail=PlotDetail, Cdict=Cdict, Ntt=Ntt, SaveName=SaveName, SavePath=SavePath, save=save, Test=Test)
        return Res, Pts

    def _set_Res(self, CrossMesh=[0.05,0.02], CrossMeshMode='rel', Mode='Iso', Amp=1., Deg=0, steps=0.001, Thres=0.05, ThresMode='rel', ThresMin=0.01,
                 IntResCross=[0.1,0.1], IntResCrossMode='rel', IntResLong=0.05, IntResLongMode='rel', Eq=None, EqName=None, Ntt=100, save=False, Test=True):
        """ Compute the resolution of the Detect instance on a mesh grid of the Cross section, with specified parameters (see :meth:`~self.calc_Res` for details)

        Parameters
        ----------
        EqName :    str
            Flag name used to identify the equilibrium reconstruction that was used, if any
        save :      bool
            if True the Detect instance saves itself automatically once the computation is finished (useful for long computations)
        """

        Res, Pts, LDetLim, Mode, steps, Thres, ThresMode, ThresMin, IntResCross, IntResCrossMode, IntResLong, IntResLongMode \
                = _Calc_Resolution(self, Pts=None, CrossMesh=CrossMesh, CrossMeshMode=CrossMeshMode, Mode=Mode, Amp=Amp, Deg=Deg, steps=steps, Thres=Thres, ThresMode=ThresMode, ThresMin=ThresMin,
                                  IntResCross=IntResCross, IntResCrossMode=IntResCrossMode, IntResLong=IntResLong, IntResLongMode=IntResLongMode, Eq=Eq, Ntt=Ntt, PlotDetail=False, save=False, Test=Test)

        self._Res_Mode, self._Res_Amp, self._Res_Deg = Mode, Amp, Deg
        self._Res_Pts, self._Res_Res, self._Res_CrossMesh, self._Res_CrossMeshMode = Pts, Res, CrossMesh, CrossMeshMode
        self._Res_steps, self._Res_Thres, self._Res_ThresMode, self._Res_ThresMin = steps, Thres, ThresMode, ThresMin
        self._Res_IntResCross, self._Res_IntResCrossMode, self._Res_IntResLong, self._Res_IntResLongMode, self._Res_IntNtt = IntResCross, IntResCrossMode, IntResLong, IntResLongMode, Ntt
        self._Res_EqName = EqName
        self._Res_Done = True
        if save:
            self.save()

    def _reset_Res(self):
        self._Res_Mode, self._Res_Amp, self._Res_Deg = None, None, None
        self._Res_Pts, self._Res_Res, self._Res_CrossMesh, self._Res_CrossMeshMode = None, None, None, None
        self._Res_steps, self._Res_Thres, self._Res_ThresMode, self._Res_ThresMin = None, None, None, None
        self._Res_IntResCross, self._Res_IntResCrossMode, self._Res_IntResLong, self._Res_IntResLongMode, self._Res_IntNtt = None, None, None, None, None
        self._Res_EqName = None
        self._Res_Done = False


    def plot(self, Lax=None, Proj='All', Elt='PVC', EltLOS='LDIORP', EltOptics='P', EltVes='',  Leg=None, LOSRef=None,
            Pdict=tfd.ApPd, Vdict=tfd.ApVd, Cdict=tfd.DetConed, LVIn=tfd.ApLVin,
            LOSdict=tfd.LOSdict, Opticsdict=tfd.Apertdict, Vesdict=tfd.Vesdict,
            LegDict=tfd.TorLegd, draw=True, a4=False, Test=True):
        """ Plot the Detect instance in a projection or in 3D, its polygon, perpendicular vector, projected viewing cones and optionally its :class:`~tofu.geom.LOS`, :class:`~tofu.geom.Apert`, and :class:`~tofu.geom.Ves` objects

        The Detect instance can be plotted in a cross-section or horizontal projection, or in 3D.
        Several of its attributes can be plotted too using the usual 'Elt' keyword argument.
        Dedicated 'Elt' keyword arguments are also usable to specify the elements to be plotted for sub-classes like :class:`~tofu.geom.LOS`, :class:`~tofu.geom.Apert`, and :class:`~tofu.geom.Ves`.
        Dedicated dictionary help specify how each element sshould be plotted.

        Parameters
        ----------
        Lax :       None, plt.Axes or list
            Axes or list of axes to be used for plotting, if None a new figure and appropriate axes are created
        Proj :      str
            Flag indicating whether to plot the cross-section ('Cross'), the horizontal projection ('Hor'), both ('All') or a 3D representation ('3D')
        Elt :       str
            Flag indicating which elements of the Detect instance to plot, each capital letter stands for an element
                * 'P': polygon
                * 'V': perpendicular vector
                * 'C': viewing cone
        EltLOS :    None or str
            Flag indicating which elements of the LOS to plot, will be fed to LOS.plot(), if None uses the 'Elt' arg of LOSdict instead
        EltOptics : None or str
            Flag indicating which elements of the Aperts to plot, will be fed to Apert.plot(), if None uses the 'Elt' arg of Apertdict instead
        EltVes :    None or str
            Flag indicating which elements of the Ves to plot, will be fed to :meth:`~tofu.geom.Ves.plot`, if None uses the 'Elt' arg of Vesdict instead
        Leg :       str
            Legend to be used for the detector, if '' the Detect.iD.Name is used
        LOSRef :    None or str
            Flag indicating which LOS should be represented, if None Detect._LOSRef is used
        Pdict :     dict
            Dictionary of properties for the Polygon
        Vdict :     dict
            Dictionary of properties for the Vector
        Cdict :     dict
            Dictionary of properties for the Cone
        LVIn :      float
            Length of the Vector
        LOSdict :   dict
            Dictionary of properties for the LOS if EltLOS is not '', fed to LOS.plot()
        Apertdict : dict
            Dictionary of properties for the Apert if EltOptics is not '', fed to Apert.plot()
        Vesdict :   dict
            Dictionary of properties for the Ves if EltVes is not '', fed to :meth:`~tofu.geom.Ves.plot`
        LegDict :   dict
            Dictionary of properties for the legend, fed to plt.legend()
        draw :      bool
            Flag indicating whether to draw the figure
        a4 :        bool
            Flag indicating whether the default figure should be of size a4 paper
        Test :      bool
            Flag indicating whether the inputs should be tested for conformity

        Returns
        -------
        Lax     plt.Axes or list
            Axes or list of axes used for plotting

        """
        return _tfg_p.GLDetect_plot(self, Lax=Lax, Proj=Proj, Elt=Elt, EltOptics=EltOptics, EltLOS=EltLOS, EltVes=EltVes, Leg=Leg, LOSRef=LOSRef,
            Pdict=Pdict, Vdict=Vdict, Cdict=Cdict, LVIn=LVIn,
            LOSdict=LOSdict, Opticsdict=Opticsdict, Vesdict=Vesdict, LegDict=LegDict, draw=draw, Test=Test)

    def plot_SAngNb(self, Lax=None, Proj='Cross', Slice='Int', Pts=None, plotfunc='scatter', DRY=None, DXTheta=None, DZ=None,
            Elt='P', EltVes='P', EltLOS='', EltOptics='P',
            Pdict=tfd.ApPd, Vdict=tfd.ApVd, Cdict=tfd.DetConed, LVIn=tfd.ApLVin,
            LOSdict=tfd.LOSdict, Opticsdict=tfd.Apertdict, Vesdict=tfd.Vesdict,
            CDictSA=None, CDictNb=None, Colis=tfd.DetSAngColis, a4=False, draw=True, Test=True):
        """ Plot the solid angle projections (integrated 'Int' or maximum 'Max') as well as the number of detectors visible from each point in the plasma

        Mostly useful with the :class:`~tofu.geom.GDetect` object, used to visualize the goemetrical coverage in terms of total solid angle and number of detectors 'seeing' each point for a set of detectors (see :meth:`~tofu.geom.Detect.calc_SAngNb` method for details).

        Parameters
        ----------
        Lax :       None or list or plt.Axes
            Axes or list of Axes to be used for plotting, if None a new figure and appropriate axes are created
        Proj :      str
            Flag indicating whether to plot the cross-section ('Cross') or the horizontal projection ('Hor')
        Mode :      str, None or float
            Flag indicating whether to plot:
                * 'Int': the integrated value along the projected coordinates
                * 'Max': the maximum value along the projected coordinates
                * float: the projected coordinate at which to plot the slice (Theta or X if Proj='Cross', Z if Proj='Hor')
                * None: the slice is done in the middle of the viewing volume
        plotfunc :  str
            Flag indicating which plotting method to use ('scatter', 'contour', 'contourf' or 'imshow')
        DCross :    float
            Resolution along the 1st cross-section coordinate (R for Type='Tor', Y for Type='Lin')
        DXTheta :   float
            Resolution along the ignorable coordinate (Theta for Type='Tor', X for Type='Lin')
        DZ :        float
            Vertical resolution (for both Types)
        CDictSA :   dict
            Properties of the solid angle plot, to be fed to the function chosen by plotfunc
        CDictNb :   dict
            Properties of the Nb plot, to be fed to the chsoen plotting routine
        Colis :     bool
            Flag indicating whether collision detection should be used
        a4 :        bool
            Flag indicating whether to use a4 dimensions to create a new figure if Lax=None
        draw :      bool
            Flag indicating whether to draw the figure
        Test :      bool
            Flag indicating whether the inputs should be tested for conformity

        Returns
        -------
        Lax :       plt.Axes or list
            List of the axes used for plotting

        """
        SA, Nb, Pts = self.calc_SAngNb(Pts=Pts, Proj=Proj, Slice=Slice, DRY=DRY, DXTheta=DXTheta, DZ=DZ, Colis=Colis)
        Lax = _tfg_p._GLDetect_plot_SAngNb(Leg=self.Id.Name, SA=SA, Nb=Nb, Pts=Pts, Lax=Lax, Proj=Proj, Slice=Slice, plotfunc=plotfunc, CDictSA=CDictSA, CDictNb=CDictNb, Colis=Colis,
                DRY=DRY, DXTheta=DXTheta, VType=self.Ves.Type, a4=a4, draw=False, Test=Test)
        if any([not ss=='' for ss in [Elt,EltVes, EltLOS, EltOptics]]):
            Lax[0] = self.plot(Proj=Proj, Lax=Lax[0], Elt=Elt, EltVes=EltVes, EltLOS=EltLOS, EltOptics=EltOptics, Pdict=Pdict, Vdict=Vdict, Cdict=Cdict, LVIn=LVIn,
                    LOSdict=LOSdict, Opticsdict=Opticsdict, Vesdict=Vesdict, LegDict=None, a4=a4, draw=False, Test=Test)
            Lax[1] = self.plot(Proj=Proj, Lax=Lax[1], Elt=Elt, EltVes=EltVes, EltLOS=EltLOS, EltOptics=EltOptics, Pdict=Pdict, Vdict=Vdict, Cdict=Cdict, LVIn=LVIn,
                    LOSdict=LOSdict, Opticsdict=Opticsdict, Vesdict=Vesdict, LegDict=None, a4=a4, draw=False, Test=Test)
        if draw:
            Lax[0].figure.canvas.draw()
        return Lax

    def _debug_plot_SAng_OnPlanePerp(self, ax=None, Pos=tfd.DetSAngPlRa, dX12=tfd.DetSAngPldX12, dX12Mode=tfd.DetSAngPldX12Mode, Ratio=tfd.DetSAngPlRatio, SurfDict=tfd.DetSAngPld, LegDict=tfd.TorLegd,
            Colis=tfd.DetSAngColis, LOSRef=None, draw=True, a4=False, Test=True):
        """ Plot the solid angle subtended by the Detect-Apert system as seen from points on a plane perpendicular to the LOS

        Used for debugging or illustrative purposes.
        Plot a surface plot of the solid angle subtended by the Detect+Optics system as seen for all points standing on a plane perpendicular to the LOS (the integral of which is the etendue).

        Parameters
        ----------
        ax :        None or plt.Axes
            Axes to be used for plotting, if None a new figure and appropriate axes are created
        Pos :       float
            Relative position between LOS.PIn and LOS.POut where the plane os to be placed, in ]0;1[
        dX12 :      list
            List of 2 floats defining the resolution of the sampling in X1 and X2
        dX12Mode :  str
            Flag indicating whether the resolution dX12 is in dimensionless units (in [0;1], hence a value of 0.1 means 10 discretisation points between the extremes), if 'abs' dX12 is in meters
        Ratio :     float
            A float specifying the relative margin to be taken for integration boundaries
        SurfDict :  dict
            Dictionary of properties to be used for plotting the surface plot (fed to :meth:`~matplotlib.pyplot.plot_surface`)
        LegDict :   None or dict
            If None, no legend is plotted, else LegDict is fed to :meth:'~matplotlib.pyplot.Axes.legend'
        Colis :     bool
            Flag indicating whether collision detection should be used
        draw :      bool
            Flag indicating whether to draw the figure
        a4 :        bool
            Flag indicating whether the created figure should have a4 dimensions (useful for printing)
        Test :      bool
            Flag indicating whether the inputs should be tested for conformity

        Returns
        -------
        ax :        plt.Axes
            The axes used for plotting

        """
        if Test:
            assert type(Pos) is float, "Arg Pos must be a float specifying the relative distance on the LOS at which the plane is to be placed !"
        if LOSRef is None:
            LOSRef = self._LOSRef
        Ps = (self.LOS[LOSRef]['LOS'].PIn + Pos*(self.LOS[LOSRef]['LOS'].POut - self.LOS[LOSRef]['LOS'].PIn)).reshape((3,1))
        nPs = self.LOS[LOSRef]['LOS'].u.reshape((3,1))
        LOPolys = [oo.Poly for oo in self.Optics]
        LOBaryS = [oo.BaryS for oo in self.Optics]
        LOnIns = [oo.nIn for oo in self.Optics]
        LOSurfs = [oo.Surf for oo in self.Optics]
        Etend, e1, e2, err, SA, X1, X2, NumX1, NumX2 = _tfg_c.Calc_Etendue_PlaneLOS(Ps, nPs, self.Poly, self.BaryS, self.nIn, LOPolys, LOnIns, LOSurfs, LOBaryS, self._SAngPlane, self.Ves.Poly, self.Ves.Vin, DLong=self.Ves.DLong,
                Lens_ConeTip=self._Optics_Lens_ConeTip, Lens_ConeHalfAng=self._Optics_Lens_ConeHalfAng, RadL=self.Optics[0].Rad, RadD=self.Rad, F1=self.Optics[0].F1,
                OpType=self.OpticsType, VType=self.Ves.Type, Mode='trapz', dX12=dX12, dX12Mode=dX12Mode, Ratio=Ratio, Colis=Colis, Details=True, Test=True)

        #SA, X1, X2, numX1, numX2 = _tfg_c.Calc_SAngOnPlane(self, P, self.LOS[LOSRef]['LOS'].u, dX12=dX12, dX12Mode=dX12Mode, e1=None,e2=None, Ratio=Ratio, Colis=Colis, Test=True)
        SA = SA.reshape((NumX1[0],NumX2[0]))
        X1, X2 = np.tile(X1.flatten(),(NumX2[0],1)).T, np.tile(X2.flatten(),(NumX1[0],1))
        Name = self.Id.NameLTX + " Pos={0} (Ratio={1})".format(Pos,Ratio)
        ax = _tfg_p.Plot_SAng_Plane(SA, X1, X2, Name=Name, ax=ax, SurfDict=SurfDict, LegDict=LegDict, draw=False, a4=a4)
        if draw:
            ax.figure.canvas.draw()
        return ax


    def plot_Etend_AlongLOS(self, ax=None, NP=tfd.DetEtendOnLOSNP, kMode='rel', Modes=['trapz'], Length='', RelErr=tfd.DetEtendepsrel, dX12=tfd.DetSynthdX12, dX12Mode=tfd.DetSynthdX12Mode, Ratio=tfd.DetEtendRatio, LOSRef=None,
            LOSPts=True, Ldict=dict(tfd.DetEtendOnLOSLd), LegDict=tfd.TorLegd, Colis=tfd.DetSAngColis, draw=True, a4=True, Test=True):
        """ Plot the etendue of the selected LOS along it, with or without collision detection

        The number of points along the LOS where the etendue is computed can be specified via arguments, as well as the numerical integration method.
        Arguments Length, NP, Modes, RelErr, dX12, dX12Mode, Ratio, Colis, LOSRef are fed to :meth:`~tofu.geom.Detect.calc_Etendue_AlongLOS`

        Parameters
        ----------
        ax :        None or plt.Axes
            Axes to be used for plotting, if None a new figure and appropriate axes are created
        kMode :     str
            Flag indicating whether the distance on the line should be plotted as abolute distance ('abs') or relative to the total length ('rel')
        Ldict :     dict
            Dictionary of properties for plotting the result
        LegDict :   None / dict
            If None, no legend is plotted, else LegDict is fed to :meth:'~matplotlib.pyplot.Axes.legend'
        draw :      bool
            Flag indicating whether to draw the figure
        a4 :        bool
            Flag indicating whether the created figure should have a4 dimensions (useful for printing)
        Test :      bool
            Flag indicating whether the inputs should be tested for conformity

        Returns
        -------
        ax :        plt.Axes
            The axes used for plotting

        """
        Etends, Pts, kPts, LOSRef = self.calc_Etendue_AlongLOS(Length=Length, NP=NP, Modes=Modes, RelErr=RelErr, dX12=dX12, dX12Mode=dX12Mode, Ratio=Ratio, Colis=Colis, LOSRef=LOSRef, Test=True)
        ax = _tfg_p.Plot_Etendue_AlongLOS(kPts, Etends, kMode, self.LOS[LOSRef]['LOS'].Id.NameLTX, ax=ax, Colis=Colis,
                Etend=self.LOS[LOSRef]['Etend'], kPIn=self.LOS[LOSRef]['LOS'].kPIn, kPOut=self.LOS[LOSRef]['LOS'].kPOut, y0=0.,
                RelErr=RelErr, dX12=dX12, dX12Mode=dX12Mode, Ratio=Ratio,
                Ldict=Ldict, LegDict=LegDict, draw=draw, a4=a4, Test=Test)
        return ax

    def plot_Sinogram(self, ax=None, Proj='Cross', Elt='DLV', Ang='theta', AngUnit='rad', Sketch=True, Ddict=tfd.DetImpd, Ldict=tfd.LOSMImpd, Vdict=tfd.TorPFilld, LegDict=tfd.TorLegd, LOSRef=None, draw=True, a4=False, Test=True):
        """ Plot the the Detect VOS in projection space, optionally also the associated :class:`~tofu.geom.Ves` object and reference LOS

        In projection space, a VOS is a patch (as opposed to a LOS which is a point).
        The patch is estimated by plotting a large number of LOS sampling the VOS and taking the convex hull of the resulting points on projection space.
        Notice that this method results in irrelevant patches for VOS lying at the edges of the projection space.
        See :meth:`~tofu.geom.LOS.plot_Sinogram` for details.

        Parameters
        ----------
        ax :        None / plt.Axes
            Axes on which to plot the Etendue, if None a default axes is created
        Proj :      str
            Flag indicating whether to plot the traditional sinogram in a cross-section ('Cross') or a 3D sinogram ('3d'), cannot be '3d' if 'D' in Elt.
        Elt :       str
            Flags indicating whether to plot the VOS of the Detect ('D' in Elt => only Proj='Cross'), the LOS ('L' in Elt) and the :class:`~tofu.geom.Ves` ('V' in Elt)
        Ang :       str
            Flag indicating which angle to use for the plot, with respect to the considered line () or to the impact parameter line ()
        AngUnit :   str
            Flag indicating whether the angle should be measured in 'rad' or 'deg'
        Sketch :    bool
            Flag indicating whether a small sketch illustrating the definitions of angles and impact parameter should be included
        Ddict :     dict
            Plotting properties of the VOS of the Detect, fed to plt.plot()
        Ldict :     dict
            Plotting properties of the LOS, fed to plt.plot()
        Vdict :     dict
            Plotting properties of the Ves, fed to plt.plot()
        LegDict :   None / dict
            Plotting properties of the legend, if None no legend is plotted
        LOSRef :    None / str
            Flag indicating which LOS to plot, if None self._LOSRef is used
        draw :      bool
            Flag indicating whether to draw the figure
        a4 :        bool
            Flag indicating whether the created figure should have a4 dimensions (useful for printing)
        Test :      bool
            Flag indicating whether the inputs should be tested for conformity

        Returns
        -------
        ax :        plt.Axes
            The axes used for plotting

        """
        ax = _tfg_p.GLDetect_plot_Sinogram(self, Proj=Proj, ax=ax, Elt=Elt, Sketch=Sketch, Ang=Ang, AngUnit=AngUnit, Ddict=Ddict, Ldict=Ldict, Vdict=Vdict, LegDict=LegDict, LOSRef=LOSRef, draw=draw, a4=a4, Test=Test)
        return ax



    def _plot_Res(self, ax=None, plotfunc='scatter', NC=20, CDict=tfd.DetConed, draw=True, a4=False, Test=True):
        """ Plot the resolution as defined by tofu (see :meth:`~tofu.geom.Detect._calc_Res` for details)

        Parameters
        ----------
        ax :        None / plt.Axes
            Axes on which to plot the Etendue, if None a default axes is created
        plotfunc :  str
            Flag indicating which plotting method to use in ['scatter','contour','contourf','imshow']
        NC :        int
            Number of contours to be plotted if plotfunc in ['contour','contourf']
        CDict :     dict
            Dictionary of properties for plotting, fed to the plotting routine
        draw :      bool
            Flag indicating whether to draw the figure
        a4 :        bool
            Flag indicating whether the created figure should have a4 dimensions (useful for printing)
        Test :      bool
            Flag indicating whether the inputs should be tested for conformity

        Returns
        -------
        ax :        plt.Axes

        """
        assert self._Res_Done, "Cannot plot the resolution before it has been computed on a mesh grid with self.set_Res() !"
        ax = _tfg_p._Resolution_Plot(self._Res_Pts, self._Res_Res, self, [self.Id.Name], ax=ax, plotfunc=plotfunc, NC=NC, CDict=dict(CDict),
                                     ind=None, Val=None, draw=draw, a4=a4, Test=Test)
        return ax



    def save(self,SaveName=None,Path=None,Mode='npz', compressed=False, SynthDiag=False):
        """ Save the object in folder Name, under file name SaveName, using specified mode

        Most tofu objects can be saved automatically as numpy arrays (.npz, recommended) at the default location (recommended) by simply calling self.save()
        In the case of Detect and GDetect instances, there is an additional keyword argument 'SynthDiag' which allows to **not** save the pre-computed 3D mesh of the VOS for synthetic diagnostic.
        Indeed, this pre-computed data is often large and results in big files. Not saving it results in significantly smaller files, and it can be re-computed when loading the instance.

        Parameters
        ----------
        SaveName :      None / str
            The name to be used for the saved file, if None (recommended) uses self.Id.SaveName
        Path :          None / str
            Path specifying where to save the file, if None (recommended) uses self.Id.SavePath
        Mode :          str
            Flag specifying whether to save the object as a numpy array file ('.npz', recommended) or an object using cPickle (not recommended, heavier and may cause retro-compatibility issues)
        compressed :    bool
            Flag, used when Mode='npz', indicating whether to use np.savez or np.savez_compressed (slower saving and loading but smaller files)
        SynthDiag :     bool
            Flag indicating whether the pre-computed mesh for synthetic diagnostics calculations shall be saved too (can be heavy, if False, it will be re-computed when opening the saved object)

        """
        if not SynthDiag:
            self._reset_SynthDiag()
        tfpf.Save_Generic(self, SaveName=SaveName, Path=Path, Mode=Mode, compressed=compressed)






def _Detect_set_Defaults(Poly=None, Type=None, Exp=None, Diag=None, shot=None, Ves=None, Optics=None):
    if not Optics is None:
        if type(Optics) is list:
            Diag = Diag if not Diag is None else Optics[0].Id.Diag
            Exp = Exp if not Exp is None else Optics[0].Id.Exp
            Ves = Optics[0].Ves if Ves is None else Ves
        else:
            Diag = Diag if not Diag is None else Optics.Id.Diag
            Exp = Exp if not Exp is None else Optics.Id.Exp
            Ves = Optics.Ves if Ves is None else Ves
        if type(Optics) is Lens:
            Type = Type if not Type is None else 'Circ'
    elif not Ves is None:
        Exp = Exp if not Exp is None else Ves.Id.Exp
    if type(Poly) is dict and not Optics is None:
        if type(Optics) is Lens:
            Poly['O'] = Optics.O-Optics.F1*Optics.nIn
            Poly['nIn'] = Optics.nIn
    return Poly, Type, Exp, Diag, shot, Ves






def _Detect_check_inputs(Id=None, Poly=None, Type=None, Optics=None, Vess=None, Sino_RefPt=None, Exp=None, Diag=None, shot=None, CalcEtend=None, CalcSpanImp=None, CalcCone=None, CalcPreComp=None, Calc=None, Verb=None,
        Etend_RelErr=None, Etend_dX12=None, Etend_dX12Mode=None, Etend_Ratio=None, Colis=None, LOSRef=None, Etend_Method=None,
        MarginRMin=None, NEdge=None, NRad=None, Nk=None,
        Cone_DRY=None, Cone_DXTheta=None, Cone_DZ=None, Cone_NPsi=None, Cone_Nk=None,
        arrayorder=None, Clock=None, SavePath=None, dtime=None, dtimeIn=None):

    if not Id is None:
        assert type(Id) in [str,tfpf.ID], "Arg Id must be a str or a tfpf.ID object !"
    if not Poly is None:
        assert  type(Poly) is dict or (hasattr(Poly,'__getitem__') and np.asarray(Poly).ndim==2 and 3 in np.asarray(Poly).shape), "Arg Poly must be a dict or an iterable with 3D cartesian coordinates of points !"
        if type(Poly) is dict:
            assert all([aa in Poly.keys() for aa in ['Rad']]), "Arg Poly must be a dict with keys ['Rad'] !"
            assert type(Poly['Rad']) in [float,np.float64], "Arg Poly['Rad'] must be a float !"
    if not Optics is None:
        assert type(Optics) in [list,Apert,Lens], "Arg Optics must be a list, Apert or Lens"
        if type(Optics) is list:
            assert all([type(oo) is Apert for oo in Optics]), "Arg Optics must be a list of Apert !"
        if type(Optics) is Lens:
            assert type(Poly) is dict and 'Rad' in Poly.keys(), "When Optics is a Lens, Poly must be a dict with field 'Rad' !"
        if not Exp is None:
            if type(Optics) is list:
                assert Exp==Optics[0].Id.Exp, "Arg Exp must be the same as the Optics[0].Id.Exp !"
            else:
                assert Exp==Optics.Id.Exp, "Arg Exp must be the same as the Optics.Id.Exp !"
        if not Diag is None:
            if type(Optics) is list:
                assert Diag==Optics[0].Id.Diag, "Arg Exp must be the same as the Optics[0].Id.Diag !"
            else:
                assert Diag==Optics.Id.Diag, "Arg Diag must be the same as the Optics.Id.Diag !"
    if not Vess is None:
        assert type(Vess) is Ves, "Arg Ves must be a Ves instance !"
        if not Exp is None:
            assert Exp==Vess.Id.Exp, "Arg Exp must be the same as the Ves.Id.Exp !"
    if not arrayorder is None:
        assert arrayorder in ['C','F'], "Arg arrayorder must be in ['C','F'] !"
    bools = [CalcEtend,CalcSpanImp,CalcCone,CalcPreComp,Calc,Verb,Colis,Clock,dtimeIn]
    if any([not aa is None for aa in bools]):
        assert all([aa is None or type(aa) is bool for aa in bools]), " Args [CalcEtend,CalcSpanImp,CalcCone,CalcPreComp,Calc,Verb,Colis,Clock,dtimeIn] must all be bool !"
    if not Exp is None:
        assert Exp in tfd.AllowedExp, "Arg Exp must be in "+str(tfd.AllowedExp)+" !"
    assert Type is None or Type=='Circ', "Arg Type must be Circ or None for Detect objects !"
    Iter2 = [Sino_RefPt,Etend_dX12]
    if any([not aa is None for aa in Iter2]):
        assert all([aa is None or (hasattr(aa,'__iter__') and np.asarray(aa).shape==(2,)) for aa in Iter2]), "Args [Sino_RefPt,Etend_dX12] must be an iterable with len()=2 !"
    strs = [Etend_dX12Mode,LOSRef,Etend_Method,Diag,SavePath]
    if any([not aa is None for aa in strs]):
        assert all([aa is None or type(aa) is str for aa in strs]), "Args [dX12Mode,LOSRef,Method,Diag,SavePath] must all be str !"
    floats = [Etend_RelErr,Etend_Ratio,MarginRMin,Cone_DRY,Cone_DXTheta,Cone_DZ]
    if any([not aa is None for aa in floats]):
        assert all([aa is None or type(aa) in [float,np.float64] for aa in floats]), "Args [RelErr,dX12,Ratio,MarginRMin] must all be floats !"
    ints = [shot,NEdge,NRad,Nk,Cone_NPsi,Cone_Nk]
    if any([not aa is None for aa in ints]):
        assert all([aa is None or type(aa) is int for aa in ints]), "Args [shot,NEdge,NRad] must be int !"
    if not dtime is None:
        assert type(dtime) is dtm.datetime, "Arg dtime must be a dtm.datetime !"














class GDetect(object):
    """ An object grouping a list of :class:`~tofu.geom.Detect` objects with some common features (e.g.: all belong to the same camera) and the same :class:`~tofu.geom.Ves` object, provides methods for common computing and plotting

    A GDetect object is a convenient tool for managing groups of detectors, applying common treatment, plotting...
    It is typically suited for a camera (e.g.: a group of detectors sharing a common aperture)

    Parameters
    ----------
    Id :            str or tfpf.ID
        A name string or a pre-built tfpf.ID class to be used to identify this particular instance, if a string is provided, it is fed to :class:`~tofu.pathfile.ID`
    LDetect :       list or Detect
        List of Detect instances with the same :class:`~tofu.geom.Ves` instance
    Type :          None
        Not used in the current verion of tofu
    Exp :           None or str
        Experiment to which the Lens belongs, should be identical to Ves.Id.Exp if Ves is provided, if None and Ves is provided, Ves.Id.Exp is used
    Diag :          None or str
        Diagnostic to which the Lens belongs
    shot :          None or int
        Shot number from which this Lens is usable (in case its position was changed from a previous configuration)
    SavePath :      None / str
        If provided, forces the default saving path of the object to the provided value
    Sino_RefPt :    None or iterable
        If provided, forces the common :attr:`~tofu.geom.Detect.Sino_RefPt` to the provided value for all :class:`~tofu.geom.Detect` instances
    arrayorder :    str
        Flag indicating whether the attributes of type=np.ndarray (e.g.: Poly) should be made C-contiguous ('C') or Fortran-contiguous ('F')
    dtime :         None or dtm.datetime
        A time reference to be used to identify this particular instance (used for debugging mostly)
    dtimeIn :       bool
        Flag indicating whether dtime should be included in the SaveName (used for debugging mostly)

    """
    def __init__(self, Id, LDetect, Type=None, Exp=None, Diag=None, shot=None, Sino_RefPt=None, LOSRef=None, arrayorder='C', Clock=False, dtime=None, dtimeIn=False, SavePath=None):

        self._Done = False
        tfpf._check_NotNone({'Clock':Clock,'arrayorder':arrayorder})
        self._check_inputs(Clock=Clock, arrayorder=arrayorder)
        self._arrayorder = arrayorder
        self._Clock = Clock

        self._check_inputs(LDetect=LDetect, Exp=Exp, Diag=Diag)
        LDetect, Exp, Diag, Sino_RefPt, LOSRef = _GDetect_set_Defaults(LDetect=LDetect, Exp=Exp, Diag=Diag, Sino_RefPt=Sino_RefPt, LOSRef=LOSRef)
        self._LOSRef = LOSRef

        self._set_Id(Id, Type=Type, Exp=Exp, Diag=Diag, shot=shot, dtime=dtime, dtimeIn=dtimeIn, SavePath=SavePath)
        self._set_LDetect(LDetect)
        self._reset_Res()
        self._Done = True

    @property
    def Id(self):
        """ the associated tfpf.ID object """
        return self._Id
    @property
    def LDetect(self):
        """ Return the list of :class:`~tofu.geom.Detect` instances the GDetect object comprises """
        return self._LDetect
    @property
    def nDetect(self):
        """ Return the number of :class:`~tofu.geom.Detect` instances the GDetect object comprises """
        return self._nDetect
    @property
    def Optics(self):
        """ Return the list of optics the GDetect object comprises (either :class:`~tofu.geom.Lens` or :class:`~tofu.geom.Apert`) """
        return self._Optics
    @property
    def Ves(self):
        """ Return the :class:`~tofu.geom.Ves` instance associated to the GDetect object """
        return self._Ves
    @property
    def Sino_RefPt(self):
        """ Return the coordinates (R,Z) or (Y,Z) for Ves of Type 'Tor' or (Y,Z) for Ves of Type 'Lin' of the reference point used to compute the sinogram """
        return self._Sino_RefPt



    def _check_inputs(self, Id=None, LDetect=None, Type=None, Sino_RefPt=None, LOSRef=None, Exp=None, Diag=None, shot=None,
                      arrayorder=None, Clock=None, SavePath=None, dtime=None, dtimeIn=None):
        _GDetect_check_inputs(Id=Id, LDetect=LDetect, Type=Type, Sino_RefPt=Sino_RefPt, LOSRef=LOSRef, Exp=Exp, Diag=Diag, shot=shot,
                             arrayorder=arrayorder, Clock=Clock, SavePath=SavePath, dtime=dtime, dtimeIn=dtimeIn)

    def _set_Id(self, Val, Type=None, Exp=None, Diag=None, shot=None, dtime=None, dtimeIn=False, SavePath=None):
        if self._Done:
            Out = tfpf._get_FromItself(self.Id, {'Type':Type, 'Exp':Exp, 'shot':shot, 'Diag':Diag, 'dtime':dtime, '_dtimeIn':dtimeIn, 'SavePath':SavePath})
            Type, Exp, shot, Diag, dtime, dtimeIn, SavePath = Out['Type'], Out['Exp'], Out['shot'], Out['Diag'], Out['dtime'], Out['dtimeIn'], Out['SavePath']
        tfpf._check_NotNone({'Id':Val})
        self._check_inputs(Id=Val)
        if type(Val) is str:
            tfpf._check_NotNone({'Exp':Exp, 'shot':shot, 'Diag':Diag, 'dtimeIn':dtimeIn})
            self._check_inputs(Type=Type, Exp=Exp, shot=shot, Diag=Diag, SavePath=SavePath, dtime=dtime, dtimeIn=dtimeIn)
            Val = tfpf.ID('GDetect', Val, Type=Type, Exp=Exp, Diag=Diag, shot=shot, SavePath=SavePath, dtime=dtime, dtimeIn=dtimeIn)
        self._Id = Val

    def _set_LDetect(self, LDetect):
        self._check_inputs(LDetect=LDetect)
        self._LDetect = LDetect
        self._nDetect = len(LDetect)
        self._Optics = _get_OpticsFromLDetect(LDetect)

        LObj = [dd.Id for dd in LDetect] + [aa.Id for aa in self._Optics]
        if not LDetect[0].Ves is None:
            LObj.append(LDetect[0].Ves.Id)
        self.Id.set_LObj(LObj)
        self._Ves = LDetect[0].Ves
        self._Sino_RefPt = LDetect[0].Sino_RefPt

    def _calc_All(self, Sino_RefPt=None, CalcEtend=True, CalcSpanImp=True, CalcCone=True, CalcPreComp=True,
                  Etend_Method=tfd.DetEtendMethod, Etend_RelErr=tfd.DetEtendepsrel, Etend_dX12=tfd.DetEtenddX12, Etend_dX12Mode=tfd.DetEtenddX12Mode, Etend_Ratio=tfd.DetEtendRatio, Colis=tfd.DetCalcEtendColis, LOSRef=None,
                  Cone_DRY=tfd.DetConeDRY, Cone_DXTheta=None, Cone_DZ=tfd.DetConeDZ, Cone_NPsi=20, Cone_Nk=60, Verb=True):
        LOSRef = self._LOSRef if LOSRef is None else LOSRef
        for ii in range(0,self.nDetect):
            self._LDetect[ii]._calc_All(Sino_RefPt=Sino_RefPt, CalcEtend=CalcEtend, CalcSpanImp=CalcSpanImp, CalcCone=CalcCone, CalcPreComp=CalcPreComp,
                                        Etend_Method=Etend_Method, Etend_RelErr=Etend_RelErr, Etend_dX12=Etend_dX12, Etend_dX12Mode=Etend_dX12Mode, Etend_Ratio=Etend_Ratio, Colis=Colis, LOSRef=LOSRef,
                                        Cone_DRY=Cone_DRY, Cone_DXTheta=Cone_DXTheta, Cone_DZ=Cone_DZ, Cone_NPsi=Cone_NPsi, Cone_Nk=Cone_Nk, Verb=Verb)

    def _set_Sino(self, RefPt=None):
        self._check_inputs(RefPt=RefPt)
        self._Ves._set_Sino(RefPt)
        for ii in range(0,self.nDetect):
            self._LDetect[ii]._set_Sino(RefPt)
        self._Sino_RefPt = RefPt

    def select(self, Val=None, Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='In', Out=bool):
        """ Return the indices or instances of all instances matching the specified criterion.

        The selection can be done according to 2 different mechanism (1) and (2).

        For mechanism (1): the user provides the value (Val) that the specified criterion (Crit) should take for a :class:`tofu.geom.Detect` to be selected.
        The criteria are typically attributes of the self.Id attribute (i.e.: name of the instance, or user-defined attributes like the camera head...)

        For mechanism (2), used if Val=None: the user provides a str expression (or a list of such) to be fed to eval(), used to check on quantitative criteria, placed before the criterion value (e.g.: 'not ' or '<=').
        Another str or list of str expressions can be provided that will be placed after the criterion value.

        Other parameters are used to specify logical operators for the selection (match any or all the criterion...) and the type of output.
        See :meth:`~tofu.geom.GLOS.select` for examples

        Parameters
        ----------
        Crit :      str
            Flag indicating which criterion to use for discrimination
            Can be set to any attribute of the tofu.pathfile.ID class (e.g.: 'Name','SaveName','SavePath'...) or any key of ID.USRdict (e.g.: 'Exp'...)
        Val :       list, str or None
            The value to match for the chosen criterion, can be a list of different values
            Used for selection mechanism (1)
        PreExp :    list, str or None
            A str of list of str expressions to be fed to eval(), used to check on quantitative criteria, placed before the criterion value (e.g.: 'not ')
            Used for selection mechanism (2)
        PostExp :   list, str or None
            A str of list of str expressions to be fed to eval(), used to check on quantitative criteria, placed after the criterion value (e.g.: '>=5.')
            Used for selection mechanism (2)
        Log :       str
            Flag indicating whether the criterion shall match all provided values or one of them ('any' or 'all')
        InOut :     str
            Flag indicating whether the returned indices are the ones matching the criterion ('In') or the ones not matching it ('Out')
        Out :       type / str
            Flag indicating in which form shall the result be returned, as an array of integer indices (int), an array of booleans (bool), a list of names ('Name') or a list of instances ('Detect')

        Returns
        -------
        ind :       list / np.ndarray
            The computed output (array of index, list of names or instances depending on parameter 'Out')

        """
        if not Out=='Detect':
            return tfpf.SelectFromListId([ll.Id for ll in self.LDetect], Val=Val, Crit=Crit, PreExp=PreExp, PostExp=PostExp, Log=Log, InOut=InOut, Out=Out)
        else:
            ind = tfpf.SelectFromListId([ll.Id for ll in self.LDetect], Val=Val, Crit=Crit, PreExp=PreExp, PostExp=PostExp, Log=Log, InOut=InOut, Out=int)
            return [self.LDetect[ii] for ii in ind]


    def isInside(self, Points, In='(X,Y,Z)', Test=True):
        """ Return an array of indices indicating whether each point lies both in the cross-section and horizontal porojections of the viewing cone of each :class:`~tofu.geom.Detect`

        see :meth:`~tofu.geom.Detect.isInside` for details

        Parameters
        ----------
        Points :    np.ndarray
            (2,N) or (3,N) array of coordinates of the N points to be tested
        In :        str
            Flag indicating in which coordinate system the Points are provided, must be in ['(R,Z)','(Y,Z)','(X,Y)','(X,Y,Z)','(R,phi,Z)']
                * '(R,Z)': All points are assumed to lie in the horizontal projection, for 'Tor' vessel type only
                * '(Y,Z)': All points are assumed to lie in the horizontal projection, for 'Lin' vessel type only
                * '(X,Y)': All points are assumed to lie in the cross-section projection
        Test :      bool
            Flag indicating whether the inputs should be tested for conformity

        Returns
        -------
        ind :       np.ndarray
            (ND,N) array of booleans with True if a point lies inside both projections of the viewing cone, where ND is the number of Detect instances

        """
        assert isinstance(Points,np.ndarray), "Arg Points must be a np.ndarray !"
        return np.vstack([dd.isInside(Points, In=In, Test=Test) for dd in self.LDetect])

    def get_GLOS(self, Name=None, LOSRef=None):
        """ Return the :class:`~tofu.geom.GLOS` instance that can be built by grouping the :class:`~tofu.geom.LOS` of each :class:`~tofu.geom.Detect` instance

        Can be useful for handling a GLOS instead of a GDetect (heavier) instance

        Parameters
        ----------
        Name :      None / str
            Name to be given to the GLOS instance, if None a name is built from the name of the GDetect object by appending '_GLOS'
        LOSRef :    None / str
            Key indicating which LOS to be used, if None the default LOSRef is used

        Returns
        -------
        glos :      :class:`~tofu.geom.GLOS`
            The constructed :class:`~tofu.geom.GLOS` instance

        """
        LOSRef = self._LOSRef if LOSRef is None else LOSRef
        LLOS = [dd.LOS[LOSRef]['LOS'] for dd in self.LDetect if not dd.LOS in ['Impossible !',None]]
        if Name is None:
            Name = self.Id.Name+'_GLOS'
        return GLOS(Name,LLOS)

    def set_SigPrecomp(self, CalcPreComp=True, dX12=tfd.DetSynthdX12, dX12Mode=tfd.DetSynthdX12Mode, ds=tfd.DetSynthds, dsMode=tfd.DetSynthdsMode, MarginS=tfd.DetSynthMarginS, Colis=tfd.DetCalcSAngVectColis):
        """ Applies :meth:`~tofu.geom.Detect.set_SigPrecomp` to all :class:`~tofu.geom.Detect` instances """
        for ii in range(0,self.nDetect):
            self._LDetect[ii].set_SigPrecomp(CalcPreComp=CalcPreComp, dX12=dX12, dX12Mode=dX12Mode, ds=ds, dsMode=dsMode, MarginS=MarginS, Colis=Colis)

    def calc_SAngVect(self, Pts, In='(X,Y,Z)', Colis=tfd.DetCalcSAngVectColis, Test=True):
        """ Applies :meth:`~tofu.geom.Detect.calc_SAngVect` to all :class:`~tofu.geom.Detect` instances

        Return the result as two 2D arrays where the first dimension is the number of :class:`~tofu.geom.Detect` instances
        see :meth:`~tofu.geom.Detect.calc_SAngVect` for details

        """
        SAng, Vect = np.zeros((self.nDetect,Pts.shape[1])), [0 for ii in range(0,self.nDetect)]
        for ii in range(0,self.nDetect):
            SAng[ii,:], Vect[ii] = self.LDetect[ii].calc_SAngVect(Pts, In=In, Colis=Colis, Test=Test)
        return SAng, Vect

    def calc_SAngNb(self, Pts=None, Proj='Cross', Slice='Int', DRY=None, DXTheta=None, DZ=None, Colis=tfd.DetSAngColis,
                    ind=None, Val=None, Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='In'):
        SA, Nb, Pts = _GDetect_Calc_SAngNb(self, Pts=Pts, Proj=Proj, Slice=Slice, DRY=DRY, DXTheta=DXTheta, DZ=DZ, Colis=Colis,
                                           ind=ind, Val=Val, Crit=Crit, PreExp=PreExp, PostExp=PostExp, Log=Log, InOut=InOut)
        """ Applies :meth:`~tofu.geom.Detect.calc_SAngNb` to all :class:`~tofu.geom.Detect` instances

        See :meth:`~tofu.geom.Detect.calc_SAngNb` for details
        Arguments ind, Val, Crit, PreExp, PostExp, Log and InOut are fed to :meth:`~tofu.geom.GDetect.select`

        """
        return SA, Nb, Pts

    def calc_Sig(self, ff, extargs={}, Method='Vol', Mode='simps', PreComp=True,
                 epsrel=tfd.DetSynthEpsrel, dX12=tfd.DetSynthdX12, dX12Mode=tfd.DetSynthdX12Mode, ds=tfd.DetSynthds, dsMode=tfd.DetSynthdsMode, MarginS=tfd.DetSynthMarginS, Colis=tfd.DetCalcSAngVectColis,  Test=True,
                 ind=None, Val=None, Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='In'):
        """ Applies :meth:`~tofu.geom.Detect.calc_Sig` to all :class:`~tofu.geom.Detect` instances

        See :meth:`~tofu.geom.Detect.calc_Sig` for details
        Arguments ind, Val, Crit, PreExp, PostExp, Log and InOut are fed to :meth:`~tofu.geom.GDetect.select`

        """
        GD, Leg, LOSRef = _tfg_p._get_LD_Leg_LOSRef(self, LOSRef=self._LOSRef, ind=ind, Val=Val, Crit=Crit, PreExp=PreExp, PostExp=PostExp, Log=Log, InOut=InOut)
        Sig = [dd.calc_Sig(ff, extargs=extargs, Method=Method, Mode=Mode, PreComp=PreComp, epsrel=epsrel, dX12=dX12, dX12Mode=dX12Mode, ds=ds, dsMode=dsMode, MarginS=MarginS, Colis=Colis,Test=Test) for dd in GD]
        return np.vstack(Sig).T, GD


    def _calc_Res(self, Pts=None, CrossMesh=[0.01,0.01], CrossMeshMode='abs', Mode='Iso', Amp=1., Deg=0, steps=0.001, Thres=0.05, ThresMode='rel', ThresMin=0.01,
                 IntResCross=[0.1,0.1], IntResCrossMode='rel', IntResLong=0.05, IntResLongMode='rel',
                 Eq=None, PlotDetail=False, Cdict=dict(tfd.DetConed), Ntt=100, SaveName=None, SavePath='./', save=False, Test=True):
        """ Applies :meth:`~tofu.geom.Detect._calc_Res` to all :class:`~tofu.geom.Detect` instances

        See :meth:`~tofu.geom.Detect._calc_Res` for details

        """

        Res, Pts, LDetLim, Mode, steps, Thres, ThresMode, ThresMin, IntResCross, IntResCrossMode, IntResLong, IntResLongMode \
                = _Calc_Resolution(self, Pts=Pts, CrossMesh=CrossMesh, CrossMeshMode=CrossMeshMode, Mode=Mode, Amp=Amp, Deg=Deg, steps=steps, Thres=Thres, ThresMode=ThresMode, ThresMin=ThresMin,
                                  IntResCross=IntResCross, IntResCrossMode=IntResCrossMode, IntResLong=IntResLong, IntResLongMode=IntResLongMode,
                                  Eq=Eq, PlotDetail=PlotDetail, Cdict=Cdict, Ntt=Ntt, SaveName=SaveName, SavePath=SavePath, save=save, Test=Test)
        return Res, Pts, LDetLim, Mode, steps, Thres, ThresMode, ThresMin, IntResCross, IntResCrossMode, IntResLong, IntResLongMode



    def _set_Res(self, CrossMesh=[0.05,0.02], CrossMeshMode='rel', Mode='Iso', Amp=1., Deg=0, steps=0.001, Thres=0.05, ThresMode='rel', ThresMin=0.01,
                 IntResCross=[0.1,0.1], IntResCrossMode='rel', IntResLong=0.05, IntResLongMode='rel', Eq=None, Ntt=100, EqName=None, save=False, Test=True):
        """ Compute the resolution of the Detect instance on a mesh grid of the Cross section, with specified parameters

        See :meth:`~tofu.geom.Detect._set_Res` for details

        """
        Res, Pts, LDetLim, Mode, steps, Thres, ThresMode, ThresMin, IntResCross, IntResCrossMode, IntResLong, IntResLongMode \
                = _Calc_Resolution(self, Pts=None, CrossMesh=CrossMesh, CrossMeshMode=CrossMeshMode, Mode=Mode, Amp=Amp, Deg=Deg, steps=steps, Thres=Thres, ThresMode=ThresMode, ThresMin=ThresMin,
                                  IntResCross=IntResCross, IntResCrossMode=IntResCrossMode, IntResLong=IntResLong, IntResLongMode=IntResLongMode, Eq=Eq, Ntt=Ntt, PlotDetail=False, save=False, Test=Test)

        self._Res_Mode, self._Res_Amp, self._Res_Deg = Mode, Amp, Deg
        self._Res_Pts, self._Res_Res, self._Res_DetLim, self._Res_CrossMesh, self._Res_CrossMeshMode = Pts, Res, LDetLim, CrossMesh, CrossMeshMode
        self._Res_steps, self._Res_Thres, self._Res_ThresMode, self._Res_ThresMin = steps, Thres, ThresMode, ThresMin
        self._Res_IntResCross, self._Res_IntResCrossMode, self._Res_IntResLong, self._Res_IntResLongMode = IntResCross, IntResCrossMode, IntResLong, IntResLongMode
        self._Res_EqName = EqName
        self._Res_Done = True
        if save:
            self.save()

    def _reset_Res(self):
        self._Res_Mode, self._Res_Amp, self._Res_Deg = None, None, None
        self._Res_Pts, self._Res_Res, self._Res_DetLim, self._Res_CrossMesh, self._Res_CrossMeshMode = None, None, None, None, None
        self._Res_steps, self._Res_Thres, self._Res_ThresMode, self._Res_ThresMin = None, None, None, None
        self._Res_IntResCross, self._Res_IntResCrossMode, self._Res_IntResLong, self._Res_IntResLongMode = None, None, None, None
        self._Res_EqName = None
        self._Res_Done = False


    def plot(self, Lax=None, Proj='All', Elt='PVC', EltLOS='LDIORP', EltOptics='P', EltVes='',  Leg=None, LOSRef=None,
             Pdict=tfd.ApPd, Vdict=tfd.ApVd, Cdict=tfd.DetConed, LVIn=tfd.ApLVin,
             LOSdict=tfd.LOSdict, Opticsdict=tfd.Apertdict, Vesdict=tfd.Vesdict,
             LegDict=tfd.TorLegd, draw=True, a4=False, Test=True,
             ind=None, Val=None, Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='In'):
        """ Plot all or a subset of the Detect instances in a projection or in 3D

        See :meth:`~tofu.geom.Detect.plot` for details
        Arguments ind, Val, Crit, PreExp, PostExp, Log and InOut are fed to :meth:`~tofu.geom.GDetect.select`

        Parameters
        ----------
        Lax :       None, plt.Axes or list
            Axes or list of axes to be used for plotting, if None a new figure and appropriate axes are created
        Proj :      str
            Flag indicating whether to plot the cross-section ('Cross'), the horizontal projection ('Hor'), both ('All') or a 3D representation ('3D')
        Elt :       str
            Flag indicating which elements of the Detect instance to plot, each capital letter stands for an element
                * 'P': polygon
                * 'V': perpendicular vector
                * 'C': viewing cone
        EltLOS :    None or str
            Flag indicating which elements of the LOS to plot, will be fed to LOS.plot(), if None uses the 'Elt' arg of LOSdict instead
        EltOptics : None or str
            Flag indicating which elements of the Aperts to plot, will be fed to Apert.plot(), if None uses the 'Elt' arg of Apertdict instead
        EltVes :    None or str
            Flag indicating which elements of the :class:`~tofu.geom.Ves` to plot, will be fed to :meth:`~tofu.geom.Ves.plot`, if None uses the 'Elt' arg of Vesdict instead
        Leg :       str
            Legend to be used for the detector, if '' the Detect.iD.Name is used
        LOSRef :    None or str
            Flag indicating which LOS should be represented, if None Detect._LOSRef is used
        Pdict :     dict
            Dictionary of properties for the Polygon
        Vdict :     dict
            Dictionary of properties for the Vector
        Cdict :     dict
            Dictionary of properties for the Cone
        LVIn :      float
            Length of the Vector
        LOSdict :   dict
            Dictionary of properties for the LOS if EltLOS is not '', fed to LOS.plot()
        Apertdict : dict
            Dictionary of properties for the Apert if EltOptics is not '', fed to Apert.plot()
        Vesdict :   dict
            Dictionary of properties for the :class:`~tofu.geom.Ves` if EltVes is not '', fed to :meth:`~tofu.geom.Ves.plot`
        LegDict :   dict
            Dictionary of properties for the legend, fed to plt.legend()
        draw :      bool
            Flag indicating whether to draw the figure
        a4 :        bool
            Flag indicating whether the default figure should be of size a4 paper
        Test :      bool
            Flag indicating whether the inputs should be tested for conformity

        Returns
        -------
        Lax     plt.Axes or list
            Axes or list of axes used for plotting

        """
        return _tfg_p.GLDetect_plot(self, Lax=Lax, Proj=Proj, Elt=Elt, EltOptics=EltOptics, EltLOS=EltLOS, EltVes=EltVes, Leg=Leg, LOSRef=LOSRef,
                                   Pdict=Pdict, Vdict=Vdict, Cdict=Cdict, LVIn=LVIn,
                                   LOSdict=LOSdict, Opticsdict=Opticsdict, Vesdict=Vesdict, LegDict=LegDict, draw=draw, Test=Test,
                                   ind=ind, Val=Val, Crit=Crit, PreExp=PreExp, PostExp=PostExp, Log=Log, InOut=InOut)


    def plot_SAngNb(self, Lax=None, Proj='Cross', Slice='Int', Pts=None, plotfunc='scatter', DRY=None, DXTheta=None, DZ=None,
            Elt='P', EltVes='P', EltLOS='', EltOptics='P',
            Pdict=tfd.ApPd, Vdict=tfd.ApVd, Cdict=tfd.DetConed, LVIn=tfd.ApLVin,
            LOSdict=tfd.LOSdict, Opticsdict=tfd.Apertdict, Vesdict=tfd.Vesdict,
            CDictSA=None, CDictNb=None, Colis=tfd.DetSAngColis, a4=False, draw=True, Test=True,
            ind=None, Val=None, Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='In'):
        """ Plot the solid angle projections (integrated 'Int' or maximum 'Max') as well as the number of detectors visible from each point in the plasma

        See :meth:`~tofu.geom.Detect.plot_SAngNb` for details
        Arguments ind, Val, Crit, PreExp, PostExp, Log and InOut are fed to :meth:`~tofu.geom.GDetect.select`

        Parameters
        ----------
        Lax :       None or list or plt.Axes
            Axes or list of Axes to be used for plotting, if None a new figure and appropriate axes are created
        Proj :      str
            Flag indicating whether to plot the cross-section ('Cross') or the horizontal projection ('Hor')
        Mode :      str, None or float
            Flag indicating whether to plot:
                * 'Int': the integrated value along the projected coordinates
                * 'Max': the maximum value along the projected coordinates
                * float: the projected coordinate at which to plot the slice (Theta or X if Proj='Cross', Z if Proj='Hor')
                * None: the slice is done in the middle of the viewing volume
        plotfunc :  str
            Flag indicating which plotting method to use ('scatter', 'contour', 'contourf' or 'imshow')
        DCross :    float
            Resolution along the 1st cross-section coordinate (R for Type='Tor', Y for Type='Lin')
        DXTheta :   float
            Resolution along the ignorable coordinate (Theta for Type='Tor', X for Type='Lin')
        DZ :        float
            Vertical resolution (for both Types)
        CDictSA :   dict
            Properties of the solid angle plot, to be fed to the function chosen by plotfunc
        CDictNb :   dict
            Properties of the Nb plot, to be fed to ...
        Colis :     bool
            Flag indicating whether collision detection should be used
        a4 :        bool
            Flag indicating whether to use a4 dimensions to create a new figure if Lax=None
        draw :      bool
            Flag indicating whether to draw the figure
        Test :      bool
            Flag indicating whether the inputs should be tested for conformity

        Returns
        -------
        Lax         plt.Axes or list            List of the axes used for plotting

        """
        SA, Nb, Pts = self.calc_SAngNb(Pts=Pts, Proj=Proj, Slice=Slice, DRY=DRY, DXTheta=DXTheta, DZ=DZ, Colis=Colis, ind=ind, Val=Val, Crit=Crit, PreExp=PreExp, PostExp=PostExp, Log=Log, InOut=InOut)
        Lax = _tfg_p._GLDetect_plot_SAngNb(Leg=self.Id.Name, SA=SA, Nb=Nb, Pts=Pts, Lax=Lax, Proj=Proj, Slice=Slice, plotfunc=plotfunc, CDictSA=CDictSA, CDictNb=CDictNb, Colis=Colis,
                                        DRY=DRY, DXTheta=DXTheta, VType=self.Ves.Type, a4=a4, draw=False, Test=Test)
        if any([not ss=='' for ss in [Elt,EltVes, EltLOS, EltOptics]]):
            Lax[0] = self.plot(Proj=Proj, Lax=Lax[0], Elt=Elt, EltVes=EltVes, EltLOS=EltLOS, EltOptics=EltOptics, Pdict=Pdict, Vdict=Vdict, Cdict=Cdict, LVIn=LVIn,
                    LOSdict=LOSdict, Opticsdict=Opticsdict, Vesdict=Vesdict, LegDict=None, a4=a4, draw=False, Test=Test,
                    ind=ind, Val=Val, Crit=Crit, PreExp=PreExp, PostExp=PostExp, Log=Log, InOut=InOut)
            Lax[1] = self.plot(Proj=Proj, Lax=Lax[1], Elt=Elt, EltVes=EltVes, EltLOS=EltLOS, EltOptics=EltOptics, Pdict=Pdict, Vdict=Vdict, Cdict=Cdict, LVIn=LVIn,
                    LOSdict=LOSdict, Opticsdict=Opticsdict, Vesdict=Vesdict, LegDict=None, a4=a4, draw=False, Test=Test,
                    ind=ind, Val=Val, Crit=Crit, PreExp=PreExp, PostExp=PostExp, Log=Log, InOut=InOut)
        if draw:
            Lax[0].figure.canvas.draw()
        return Lax


    def plot_Etend_AlongLOS(self, ax=None, NP=tfd.DetEtendOnLOSNP, kMode='rel', Modes=['trapz'], RelErr=None, dX12=None, dX12Mode=None, Ratio=None, LOSRef=None,
                            LOSPts=True, Ldict=tfd.DetEtendOnLOSLd, LegDict=tfd.TorLegd, Colis=tfd.DetSAngColis, draw=True, a4=True, Test=True,
                            ind=None, Val=None, Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='In'):
        """ Plot the etendue of the selected LOS along it, with or without collision detection

        The number of points along the LOS where the etendue is computed can be specified via arguments, as well as the numerical integration method.
        See :meth:`~tofu.geom.Detect.plot_Etendue_AlongLOS` for details
        Arguments Length, NP, Modes, RelErr, dX12, dX12Mode, Ratio, Colis, LOSRef are fed to :meth:`~tofu.geom.Detect.calc_Etendue_AlongLOS`
        Arguments ind, Val, Crit, PreExp, PostExp, Log and InOut are fed to :meth:`~tofu.geom.GDetect.select`

        Parameters
        ----------
        ax :        None or plt.Axes
            Axes to be used for plotting, if None a new figure and appropriate axes are created
        NP :        int
            Number of points along the LOS at which the Etendue should be computed
        kMode :     str
            Flag indicating whether the distance on the line should be plotted as abolute distance ('abs') or relative to the total length ('rel')
        Modes :     str or list
            Flag or list of flags indicating which integration method should be used
        Colis :     bool
            Flag indicating whether collision detection should be used
        LOSRef :    None or str
            Flag indicating which LOS should be used
        Ldict :     dict
            Dictionary of properties for plotting the result
        LegDict :   None / dict
            If None, no legend is plotted, else LegDict is fed to :meth:'~matplotlib.pyplot.Axes.legend'
        draw :      bool
            Flag indicating whether to draw the figure
        a4 :        bool
            Flag indicating whether the created figure should have a4 dimensions (useful for printing)
        Test :      bool
            Flag indicating whether the inputs should be tested for conformity

        Returns
        -------
        ax :        plt.Axes
            The axes used for plotting

        """
        if ind is None:
            ind = self.select(Val=Val,Crit=Crit,InOut=InOut,Out=int)
        elif ind.dtype.name=='bool':
            ind = ind.nonzero()[0]
        LD = [self.LDetect[ii] for ii in ind]
        nD = len(LD)
        LOSRef = self._LOSRef if LOSRef is None else LOSRef
        for ii in range(0,nD):
            ax = LD[ii].plot_Etend_AlongLOS(ax=ax, NP=NP, kMode=kMode, Modes=Modes, RelErr=RelErr, dX12=dX12, dX12Mode=dX12Mode, Ratio=Ratio, LOSRef=LOSRef,
                                     LOSPts=LOSPts, Ldict=Ldict, LegDict=None, Colis=Colis, draw=False, a4=a4, Test=Test)
        if LegDict is not None:
            ax.legend(**LegDict)
        if draw:
            ax.figure.canvas.draw()
        return ax

    def plot_Sinogram(self, ax=None, Proj='Cross', Elt='DLV', Ang='theta', AngUnit='rad', Sketch=True, Ddict=tfd.DetImpd, Ldict=tfd.LOSMImpd, Vdict=tfd.TorPFilld, LegDict=tfd.TorLegd, LOSRef=None, draw=True, a4=False, Test=True,
            ind=None, Val=None, Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='In'):
        """ Plot the VOS of all or of a subset of the :class:`~tofu.geom.Detect` instances in projection space, optionally also the associated :class:`~tofu.geom.Ves` object and reference :class:`~tofu.geom.LOS`

        See :meth:`~tofu.geom.Detect.plot_Sinogram` for details
        Arguments ind, Val, Crit, PreExp, PostExp, Log and InOut are fed to :meth:`~tofu.geom.GDetect.select`

        """
        ax = _tfg_p.GLDetect_plot_Sinogram(self, Proj=Proj, ax=ax, Elt=Elt, Sketch=Sketch, Ang=Ang, AngUnit=AngUnit, Ddict=Ddict, Ldict=Ldict, Vdict=Vdict, LegDict=LegDict, LOSRef=LOSRef, draw=draw, a4=a4, Test=Test,
                                          ind=ind, Val=Val, Crit=Crit, PreExp=PreExp, PostExp=PostExp, Log=Log, InOut=InOut)
        return ax

    def plot_Etendues(self, Mode='Etend', Elt='', ax=None, Adict=tfd.GDetEtendMdA, Rdict=tfd.GDetEtendMdR, Edict=tfd.GDetEtendMdS, LegDict=tfd.TorLegd, draw=True, a4=False, Test=True,
                      ind=None, Val=None, Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='In'):
        """ Plot the etendues of all or a subset of the :class:`~tofu.geom.Detect` instances for the chosen :class:`~tofu.geom.LOS`

        A given Detect+Optics system has a VOS, under proper conditions, this VOS can be approximated by a LOS, but the choice of the LOS is not unique, there is an infinite number of possible LOS in a single VOS.
        The LOS automatically computed by tofu os the 'natural' option : goes from the midlle of the Detect area throught the middle of the optics.
        Then tofu automatically computes the associated etendue.
        This methods plots all the etendues of all the chosen :class:`~tofu.geom.Detect` instances for the chosen :class:`~tofu.geom.LOS`, which is by default the 'natural' LOS computed by tofu

        Parameters
        ----------
        Mode :      str
            Flasg indicating whether to plot the etendue ('Etend') or a geometrical calibration factor ('Calib') computed as the 4pi/etendue
        Elt :       str
            Flag indicating whether to plot, in addition to the etendue, also the direct ('A') and reverse ('R') 0-order approximation of the etendue
        ax :        None or plt.Axes
            Axes to be used for plotting, if None a new figure and appropriate axes are created
        Adict :     dict
            Dictionary of properties for plotting the direct 0-order approximation of the etendue (if 'A' in Elt), fed to :meth:`~matplotlib.pyplot.Axes.plot`
        Rdict :     dict
            Dictionary of properties for plotting the reverse 0-order approximation of the etendue (if 'R' in Elt), fed to :meth:`~matplotlib.pyplot.Axes.plot`
        Edict :     dict
            Dictionary of properties for plotting the etendue, fed to :meth:`~matplotlib.pyplot.Axes.plot`
        LegDict :   dict
            If None, no legend is plotted, else LegDict is fed to :meth:'~matplotlib.pyplot.Axes.legend'
        draw :      bool
            Flag indicating whether to draw the figure
        a4 :        bool
            Flag indicating whether the created figure should have a4 dimensions (useful for printing)
        Test :      bool
            Flag indicating whether the inputs should be tested for conformity

        Returns
        -------
        ax :        plt.Axes
            The axes used for plotting

        """
        ax = _tfg_p.Plot_Etendues_GDetect(self, ax=ax, Mode=Mode, Elt=Elt, Adict=Adict, Rdict=Rdict, Edict=Edict, LegDict=LegDict, draw=draw, a4=a4, Test=Test,
                                         ind=ind, Val=Val, Crit=Crit, PreExp=PreExp, PostExp=PostExp, Log=Log, InOut=InOut)
        return ax


    def plot_Sig(self, ffSig, extargs={}, Method='Vol', Mode='simps', ax=None, Leg='', Sdict=tfd.GDetSigd, LegDict=tfd.TorLegd, draw=True, a4=False, Test=True,
                 PreComp=True, epsrel=tfd.DetSynthEpsrel, dX12=tfd.DetSynthdX12, dX12Mode=tfd.DetSynthdX12Mode, ds=tfd.DetSynthds, dsMode=tfd.DetSynthdsMode, MarginS=tfd.DetSynthMarginS, Colis=tfd.DetCalcSAngVectColis,
                 ind=None, Val=None, Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='In'):
        """ Plot the ignal computed for each or a subset of the :class:`~tofu.geom.Detect` instances

        If the signal is not directly provided as an array, it is computed from a function.
        If ffSig is a callable function, arguments ffSig, extargs, Method, Mode, PreComp, epsrel, dX12, dX12Mode, ds, dsMode, MarginS, Colis and Test are fed to :meth:`~tofu.geom.GDetect.calc_Sig`
        Arguments ind, Val, Crit, PreExp, PostExp, Log and InOut are fed to :meth:`~tofu.geom.GDetect.select`

        Parameters
        ----------
        ffSig       np.ndarray or callable
            Either a np.ndarray containing the signal to be plotted (of shape (ND,) or (N,ND) where ND is the number of detectors to be plotted) or a callable to be fed to for computing the signal
        ax :        None or plt.Axes
            Axes to be used for plotting, if None a new figure and appropriate axes are created
        Sdict :     dict
            Dictionary of properties for plotting the signal, fed to :meth:`~matplotlib.pyplot.Axes.plot`
        Leg :       str
            Label to be used for the plot
        LegDict :   dict
            If None, no legend is plotted, else LegDict is fed to :meth:'~matplotlib.pyplot.Axes.legend'
        draw :      bool
            Flag indicating whether to draw the figure
        a4 :        bool
            Flag indicating whether the created figure should have a4 dimensions (useful for printing)
        Test :      bool
            Flag indicating whether the inputs should be tested for conformity

        Returns
        -------
        ax :        plt.Axes
            The axes used for plotting

        """
        assert type(ffSig) is np.ndarray or hasattr(ffSig,'__call__'), "Arg ffSig must be either pre-computed np.ndarray of signals or a callable function for computing it (fed to GDetect.calc_Sig()) !"
        if type(ffSig) is not np.ndarray:
            ffSig, GD = self.calc_Sig(ffSig, extargs=extargs, Method=Method, Mode=Mode, PreComp=PreComp, epsrel=epsrel, dX12=dX12, dX12Mode=dX12Mode, ds=ds, dsMode=dsMode, MarginS=MarginS, Colis=Colis,  Test=Test)
        else:
            GD, Leg, LOSRef = _tfg_p._get_LD_Leg_LOSRef(self, Leg=Leg, LOSRef=LOSRef, ind=ind, Val=Val, Crit=Crit, PreExp=PreExp, PostExp=PostExp, Log=Log, InOut=InOut)
            assert (ffSig.ndim==1 and ffSig.size==len(GD)) or (ffSig.ndim==2 and ffSig.shape[1]==len(GD)), "Arg ffSig does not have the good shape !"
        ax = _tfg_p.Plot_Sig_GDetect(GD, ffSig, ax=ax, Leg=Leg, Sdict=Sdict, LegDict=LegDict, draw=draw, a4=a4, Test=Test)
        return ax


    def _plot_Res(self, ax=None, plotfunc='scatter', NC=20, CDict=None, draw=True, a4=False, Test=True):
        """ see :meth:`~tofu.geom.Detect._plot_Res` for details

        """
        assert self._Res_Done, "Cannot plot the resolution before it has been computed on a mesh grid with self.set_Res() !"
        ax = _tfg_p._Resolution_Plot(self._Res_Pts, self._Res_Res, self, self._Res_DetLim, ax=ax, plotfunc=plotfunc, NC=NC, CDict=CDict, draw=draw, a4=a4, Test=Test)
        return ax


    def save(self,SaveName=None,Path=None,Mode='npz', compressed=False, SynthDiag=False):
        """ Save the object in folder Name, under file name SaveName, using specified mode

        Most tofu objects can be saved automatically as numpy arrays (.npz, recommended) at the default location (recommended) by simply calling self.save()
        In the case of Detect and GDetect instances, there is an additional keyword argument 'SynthDiag' which allows to **not** save the pre-computed 3D mesh of the VOS for synthetic diagnostic.
        Indeed, this pre-computed data is often large and results in big files. Not saving it results in significantly smaller files, and it can be re-computed when loading the instance.

        Parameters
        ----------
        SaveName :      None / str
            The name to be used for the saved file, if None (recommended) uses self.Id.SaveName
        Path :          None / str
            Path specifying where to save the file, if None (recommended) uses self.Id.SavePath
        Mode :          str
            Flag specifying whether to save the object as a numpy array file ('.npz', recommended) or an object using cPickle (not recommended, heavier and may cause retro-compatibility issues)
        compressed :    bool
            Flag, used when Mode='npz', indicating whether to use np.savez or np.savez_compressed (slower saving and loading but smaller files)
        SynthDiag :     bool
            Flag indicating whether the pre-computed mesh for synthetic diagnostics calculations shall be saved too (can be heavy, if False, it will be re-computed when opening the saved object)

        """
        if not SynthDiag:
            for ii in range(0,self.nDetect):
                self.LDetect[ii]._reset_SynthDiag()
        tfpf.Save_Generic(self, SaveName=SaveName, Path=Path, Mode=Mode, compressed=compressed)





def _get_OpticsFromLDetect(LD):
        LO = []
        for ii in range(0,len(LD)):
            if ii==0:
                LO += LD[ii].Optics
            else:
                for jj in range(0,len(LD[ii].Optics)):
                    if not any([tfpf.CheckSameObj(aa, LD[ii].Optics[jj], ['Poly','Name','SaveName','Type']) for aa in LO]):
                        LO.append(LD[ii].Optics[jj])
        return LO


def _GDetect_set_Defaults(LDetect=None, Exp=None, Diag=None, Sino_RefPt=None, LOSRef=None):
    if not LDetect is None:
        if type(LDetect) is list:
            Diag = Diag if not Diag is None else LDetect[0].Id.Diag
            Exp = Exp if not Exp is None else LDetect[0].Id.Exp
            Sino_RefPt = Sino_RefPt if not Sino_RefPt is None else LDetect[0].Sino_RefPt
            LOSRef = LOSRef if not LOSRef is None else LDetect[0]._LOSRef
        else:
            Diag = Diag if not Diag is None else LDetect.Id.Diag
            Exp = Exp if not Exp is None else LDetect.Id.Exp
            Sino_RefPt = Sino_RefPt if not Sino_RefPt is None else LDetect.Sino_RefPt
            LOSRef = LOSRef if not LOSRef is None else LDetect._LOSRef
            LDetect = [LDetect]
    return LDetect, Exp, Diag, Sino_RefPt, LOSRef





def _GDetect_check_inputs(Id=None, LDetect=None, Type=None, Optics=None, Vess=None, Sino_RefPt=None, Exp=None, Diag=None, shot=None, CalcEtend=None, CalcSpanImp=None, CalcCone=None, CalcPreComp=None, Calc=None, Verb=None,
        Etend_RelErr=None, Etend_dX12=None, Etend_dX12Mode=None, Etend_Ratio=None, Colis=None, LOSRef=None, Etend_Method=None,
        MarginRMin=None, NEdge=None, NRad=None, Nk=None,
        Cone_DRY=None, Cone_DXTheta=None, Cone_DZ=None, Cone_NPsi=None, Cone_Nk=None,
        arrayorder=None, Clock=None, SavePath=None, dtime=None, dtimeIn=None):

    if not Id is None:
        assert type(Id) in [str,tfpf.ID], "Arg Id must be a str or a tfpf.ID object !"
    if not LDetect is None:
        assert type(LDetect) is list and all([type(dd) is Detect for dd in LDetect]), "Arg LDetect must be a list of Detect instances !"
        assert all([tfpf.CheckSameObj(LDetect[0].Ves,dd.Ves, ['Poly','Name','SaveName']) for dd in LDetect]), "All Detect objects must have the same :class:`~tofu.geom.Ves` object !"
        assert all([np.all(dd.Sino_RefPt==LDetect[0].Sino_RefPt) for dd in LDetect])
        assert all([dd.Id.Exp==LDetect[0].Id.Exp for dd in LDetect]), "All Detect instances in LDetect must belong to the same Exp !"
        assert all([dd.Id.Diag==LDetect[0].Id.Diag for dd in LDetect]), "All Detect instances in LDetect must belong to the same Diag !"
        if not Exp is None:
            assert Exp==LDetect[0].Id.Exp, "Arg Exp must be identical to the LDetect Exp !"
        if not Diag is None:
            assert Diag==LDetect[0].Id.Diag, "Arg Diag must be identical to the LDetect Diag !"
    if not arrayorder is None:
        assert arrayorder in ['C','F'], "Arg arrayorder must be in ['C','F'] !"
    bools = [Clock,dtimeIn]
    if any([not aa is None for aa in bools]):
        assert all([aa is None or type(aa) is bool for aa in bools]), " Args [CalcEtend,CalcSpanImp,CalcCone,CalcPreComp,Calc,Verb,Colis,Clock,dtimeIn] must all be bool !"
    if not Exp is None:
        assert Exp in tfd.AllowedExp, "Arg Exp must be in "+str(tfd.AllowedExp)+" !"
    assert Type is None, "Arg Type must be None for GDetect objects !"
    Iter2 = [Sino_RefPt]
    if any([not aa is None for aa in Iter2]):
        assert all([aa is None or (hasattr(aa,'__iter__') and np.asarray(aa).shape==(2,)) for aa in Iter2]), "Args [Sino_RefPt] must be an iterable with len()=2 !"
    strs = [Diag,SavePath]
    if any([not aa is None for aa in strs]):
        assert all([aa is None or type(aa) is str for aa in strs]), "Args [dX12Mode,LOSRef,Method,Diag,SavePath] must all be str !"
    ints = [shot]
    if any([not aa is None for aa in ints]):
        assert all([aa is None or type(aa) is int for aa in ints]), "Args [shot,NEdge,NRad] must be int !"
    if not dtime is None:
        assert type(dtime) is dtm.datetime, "Arg dtime must be a dtm.datetime !"




def _GDetect_Calc_SAngNb(GD, Pts=None, Proj='Cross', Slice='Int', DRY=None, DXTheta=None, DZ=None, Colis=tfd.DetSAngColis,
                         ind=None, Val=None, Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='In'):
        if ind is None:
            ind = GD.select(Val=Val,Crit=Crit,InOut=InOut,Out=int)
        elif ind.dtype.name=='bool':
            ind = ind.nonzero()[0]
        LD = [GD.LDetect[ii] for ii in ind]
        nD = len(LD)
        if DRY is None:
            DRY = min([dd._Cone_Poly_DR for dd in LD]) if GD.Ves.Type=='Tor' else min([dd._Cone_Poly_DY for dd in LD])
        if DXTheta is None:
            DXTheta = min([dd._Cone_Poly_DTheta for dd in LD]) if GD.Ves.Type=='Tor' else min([dd._Cone_Poly_DX for dd in LD])
        if DZ is None:
            DZ = min([dd._Cone_Poly_DZ for dd in LD])
        # Get the mesh if Pts not provided
        if Pts is None:
            LLOS = GD.get_GLOS().LLOS
            LLOS = [LLOS[ii] for ii in ind]
            SingPts = np.vstack(tuple([np.vstack((ll.PIn, ll.PIn+0.002*ll.u, 0.5*(ll.POut+ll.PIn), ll.POut-0.002*ll.u , ll.POut)) for ll in LLOS])).T
            LSpan_Z = [dd._Span_Z for dd in LD]
            if GD.Ves.Type=='Tor':
                LSpan_R = [dd._Span_R for dd in LD]
                LSpan_Theta = [dd._Span_Theta for dd in LD]
                LSpan_Z = [dd._Span_Z for dd in LD]
                X1, XIgn, Z, NX1, NIgn, NZ, Pts, out = _tfg_c._get_CrossHorMesh(SingPoints=SingPts, LSpan_R=LSpan_R, LSpan_Theta=LSpan_Theta, LSpan_Z=LSpan_Z, DR=DRY, DTheta=DXTheta, DZ=DZ,
                        VType=GD.Ves.Type, Proj=Proj, ReturnPts=True)
            elif GD.Ves.Type=='Lin':
                LSpan_X = [dd._Span_X for dd in LD]
                LSpan_Y = [dd._Span_Y for dd in LD]
                XIgn, X1, Z, NIgn, NX1, NZ, Pts, out = _tfg_c._get_CrossHorMesh(SingPoints=SingPts, LSpan_X=LSpan_X, LSpan_Y=LSpan_Y, LSpan_Z=LSpan_Z, DX=DXTheta, DY=DRY, DZ=DZ, VType=GD.Ves.Type, Proj=Proj, ReturnPts=True)

        # Get the Solid angle (itself, or Int or Max)
        SA = np.zeros((nD,Pts.shape[1]))
        if Slice in ['Int','Max']:
            for ii in range(0,nD):
                FF = LD[ii]._get_SAngIntMax(Proj=Proj, SAng=Slice)
                SA[ii,:] = FF(Pts, In=out)
                if np.any(SA[ii,:]<0.):
                    print "    SAngNb : ", LD[ii].Id.Name, " has negative SAng values !"
        else:
            if Proj=='Hor':
                Span = [min([oo[0] for oo in LSpan_Z]), max([oo[1] for oo in LSpan_Z])]
            else:
                Span = [min([oo[0] for oo in LSpan_Theta]), max([oo[1] for oo in LSpan_Theta])] if GD.Ves.Type=='Tor' else [min([oo[0] for oo in LSpan_X]), max([oo[1] for oo in LSpan_X])]
            assert Slice>=Span[0] and Slice<=Span[1], "Arg Slice is outside of the interval were non-zeros values can be found !"
            Ptsint = _tfg_gg.CoordShift(Pts, In=out, Out='(X,Y,Z)', CrossRef=Slice)
            for ii in range(0,nD):
                SA[ii,:] = LD[ii].calc_SAngVect(Ptsint, In='(X,Y,Z)', Colis=Colis, Test=True)[0]
                if np.any(SA[ii,:]<0.):
                    print "    SAngNb : ", LD[ii].Id.Name, " has negative SAng values !"
        Nb = np.sum(SA>0.,axis=0)
        SA = np.sum(SA,axis=0)
        return SA, Nb, Pts




























"""
###############################################################################
###############################################################################
                   Special high-level functions
###############################################################################
"""



def _Calc_Resolution(GLD, Pts=None, CrossMesh=[0.01,0.01], CrossMeshMode='abs', Mode='Iso', Amp=1., Deg=0, steps=0.001, Thres=0.05, ThresMode='rel', ThresMin=0.01,
                    IntResCross=[0.1,0.1], IntResCrossMode='rel', IntResLong=0.05, IntResLongMode='rel',
                    Eq=None, PlotDetail=False, Cdict=dict(tfd.DetConed), Ntt=100,
                    ind=None, Val=None, Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='In', SaveName=None, SavePath='./', save=False, Test=True):
    if Test:
        assert type(GLD) in [list,Detect,GDetect], "Arg GLD must be a Detect or list of such or a GDetect instance !"
        assert Pts is None or (hasattr(Pts,'__iter__') and np.asarray(Pts).ndim in [1,2]), "Arg Pts must be an iterable with Points coordinates !"
        assert len(CrossMesh)==2, "Arg CrossMesh must be a len()==2 iterable with the desired mesh resolution in the two directions !"
        assert CrossMeshMode in ['abs','rel'], "Arg CrossMeshMode must be in ['abs','rel'] !"
        assert Mode in ['Iso','HorVert','Equi'], "Arg Mode must be in ['Iso','HorVert','Equi'] !"
        assert type(Amp) is float, "Arg Amp must be a float (amplitude of the emissivity) !"
        assert type(Deg) is int, "Arg Deg must be a int (degree of the bivariate b-splines) !"
        assert type(steps) is float or (hasattr(steps,'__iter__') and len(steps)==2), "Arg steps must be a float or an iterable of 2 floats (incremental increase of emissivity size, in absolute value, meters or rad) !"
        assert type(Thres) is float or hasattr(Thres,'__iter__'), "Arg Thres must be a float or an iterable (fraction of the initial signal above which a change can be considered visible) !"
        assert len(IntResCross)==2, "Arg IntResCross must be an iterable of len()==2 with absolute resolution to be used for signal integration ([DRY,DZ]) !"
        assert type(IntResLong) is float, "Arg IntResLong must be a float with the absolute resolution to be used for signal integration (DXTheta) !"
        assert Eq is None or hasattr(Eq,'__call__'), "Arg Eq must be None or a callable function (delivering etheta tangent to flux surface for each point in cross-section) !"

    # Convert to list of Detect, with common legend if GDetect
    GLD, Leg, LOSRef = _tfg_p._get_LD_Leg_LOSRef(GLD, Leg=None, LOSRef=None, ind=ind, Val=Val, Crit=Crit, PreExp=PreExp, PostExp=PostExp, Log=Log, InOut=InOut)
    ND = len(GLD)
    Ves = GLD[0].Ves
    if hasattr(Thres,'__iter__'):
        Thres = np.asarray(Thres)
        assert Thres.ndim==1 and Thres.size==len(GLD), "If an iterable, arg Thres must be the same len() as the input list of Detect !"

    DXTheta = np.array([dd._Span_Theta for dd in GLD]) if Ves.Type=='Tor' else np.array([dd._Span_X for dd in GLD])
    DXTheta = [np.nanmin(DXTheta[:,0]), np.nanmax(DXTheta[:,1])]

    # Build name if save
    if save:
        if SaveName is None:
            Thresstr = "Thres{0:02.0f}-{1:02.0f}".format(100.*ThresMin,100.*Thres)
            Thresstr = Thresstr+"Rel" if ThresMode.lower()=='rel' else Thresstr+"Abs"
            Stepstr = "Step{0:03.1f}mm".format(1000.*steps)
            Intstr = "IntCross{0:02.0f}-{1:02.0f}mm".format(1000.*IntResCross[0],1000.*IntResCross[1]) if IntResCrossMode.lower()=='abs' else "IntCross{0:4.2f}-{1:4.2f}".format(IntResCross[0],IntResCross[1])
            Intstr = Intstr+"_IntLong{0:02.0f}mm".format(1000.*IntResLong) if IntResLongMode.lower()=='abs' else Intstr+"_IntLong{0:4.2f}".format(IntResLong)
            SaveName = 'Res_'+GLD[0].Id.Exp+'_Diag'+GLD[0].Id.Diag+'_'+Mode+'_'+Thresstr+'_'+Stepstr+'_'+Intstr
            print SaveName

    # Prepare mesh
    if Pts is None:
        Pts, X1, X2, NumX1, NumX2 = Ves.get_MeshCrossSection(CrossMesh=CrossMesh, CrossMeshMode=CrossMeshMode, Test=True)
    else:
        Pts = np.asarray(Pts)
        Pts = Pts if Pts.ndim==2 else Pts.reshape((Pts.size,1))
        assert Pts.shape[0]==2, "Arg Pts must be provided in (R,Z) or (Y,Z) coordinates (for Ves.Type=Tor or Ves.Type=Lin) !"
    In = '(R,Z)' if Ves.Type=='Tor' else '(Y,Z)'

    ind = np.vstack([dd.isInside(Pts, In=In) for dd in GLD])

    # Prepare THR fraction
    Thres = Thres*np.ones((ND,),dtype=float) if type(Thres) is float else Thres
    if ThresMode=='abs':
        THR = np.copy(Thres)
    else:
        # Restrict to points initially detected by at least one detector
        Pts = Pts[:,np.any(ind,axis=0)]
        ind = ind[:,np.any(ind,axis=0)]
    NP = Pts.shape[1]

    LDetLim = []
    if Mode=='Iso':
        tt = np.linspace(0.,2.*np.pi,Ntt)
        Res = np.nan*np.ones((NP,))
        for ii in range(0,NP):
            print "    Resolution : Point", ii+1, "/", NP
            size = 0.
            InitSigs = np.zeros((ND,),dtype=float)
            if np.any(ind[:,ii]):
                Ind = ind[:,ii].nonzero()[0]
                pp, Emiss, dV = _Resolution_PpsEmissdV_Iso(Pts[:,ii], DXTheta, size=size, Deg=Deg, Amp=Amp, IntResCross=IntResCross, IntResCrossMode=IntResCrossMode,
                                                           IntResLong=IntResLong, IntResLongMode=IntResLongMode, VType=Ves.Type)
                InitSigs[ind[:,ii]] = np.asarray([np.sum(Emiss * dV * GLD[jj].calc_SAngVect(pp, In='(X,Y,Z)', Colis=True)[0]) for jj in Ind])
                if not np.any(InitSigs[ind[:,ii]]>0.):
                    continue
                if ThresMode=='rel':
                    THR = np.min(Thres)*np.nanmin(InitSigs[ind[:,ii]])*np.ones((ND,))
                    THR[ind[:,ii]] = Thres[ind[:,ii]]*InitSigs[ind[:,ii]]
                    if not ThresMin is None:
                        THRmin = ThresMin*np.nanmax(InitSigs[ind[:,ii]])
                        THR[THR<THRmin] = THRmin

            Lsigs = [np.copy(InitSigs)]
            Lsize = [size]
            while not np.any(np.abs(Lsigs[-1]-InitSigs)>THR):
                Lsize.append(Lsize[-1]+steps)
                pp, Emiss, dV = _Resolution_PpsEmissdV_Iso(Pts[:,ii], DXTheta, size=Lsize[-1], Deg=Deg, Amp=Amp, IntResCross=IntResCross, IntResCrossMode=IntResCrossMode,
                                                           IntResLong=IntResLong, IntResLongMode=IntResLongMode, VType=Ves.Type)
                Lsigs.append(np.asarray([np.sum(Emiss * dV * dd.calc_SAngVect(pp, In='(X,Y,Z)', Colis=True)[0]) for dd in GLD]))

            if len(Lsize) == 2:
                #print "    ...Refining..."
                Lsigs = [np.copy(InitSigs)]
                Lsize = [0.]
                stepsbis = steps/5.
                while not np.any(np.abs(Lsigs[-1]-InitSigs)>THR):
                    Lsize.append(Lsize[-1]+stepsbis)
                    pp, Emiss, dV = _Resolution_PpsEmissdV_Iso(Pts[:,ii], DXTheta, size=Lsize[-1], Deg=Deg, Amp=Amp, IntResCross=IntResCross, IntResCrossMode=IntResCrossMode,
                                                               IntResLong=IntResLong, IntResLongMode=IntResLongMode, VType=Ves.Type)
                    Lsigs.append(np.asarray([np.sum(Emiss * dV * dd.calc_SAngVect(pp, In='(X,Y,Z)', Colis=True)[0]) for dd in GLD]))

            # Identify the Detect that passed the threshold and interpolate an accurate value of Res from Lsize
            indDet = (np.abs(Lsigs[-1]-InitSigs)>THR).nonzero()[0][0]
            ss = [Lsigs[-2][indDet],Lsigs[-1][indDet]]
            crit = InitSigs[indDet]+THR[indDet] if Lsigs[-1][indDet] > InitSigs[indDet]+THR[indDet] else InitSigs[indDet]-THR[indDet]
            Res[ii] = (np.diff(Lsize[-2:]))/np.diff(ss) * (crit-Lsigs[-2][indDet]) + Lsize[-2]
            LDetLim.append(GLD[indDet].Id.Name)

            if PlotDetail:
                print 'InitSigs[ind[:,ii]]', InitSigs[ind[:,ii]]
                print 'THR[ind[:,ii]]', THR[ind[:,ii]]
                ax1, ax2, ax3 = _tfg_p._Resolution_PlotDetails(GLD, ND, Pts[:,ii], np.array(Lsize), np.vstack(Lsigs), InitSigs, len(Lsigs), indDet, Ind, Res[ii], Ves, THR, tt=tt, Cdict=dict(Cdict), draw=True)
            #print "    ", GLD[indDet].Id.Name, Res[ii]

        if save:
            np.savez(SavePath+SaveName+'.npz', Res=Res, LDetLim=LDetLim, Pts=Pts, Mode=Mode, steps=steps, Thres=Thres, ThresMode=ThresMode, ThresMin=ThresMin,
                     IntResCross=IntResCross, IntResCrossMode=IntResCrossMode, IntResLong=IntResLong, IntResLongMode=IntResLongMode)
        return Res, Pts, LDetLim, Mode, steps, Thres, ThresMode, ThresMin, IntResCross, IntResCrossMode, IntResLong, IntResLongMode
    else:
        Res0, Res1 = np.nan*np.ones((NP,)), np.nan*np.ones((NP,))
        for ii in range(0,NP):
            ind = [dd.isInside(Pts[:,ii:ii+1], In=In) for dd in GLD]
        if save:
            np.savez(SavePath+SaveName+'.npz', Res0=Res0, Res1=Res1, LDetLim=LDetLim, Pts=Pts, Mode=Mode, steps=steps, Thres=Thres, ThresMode=ThresMode, ThresMin=ThresMin,
                     IntResCross=IntResCross, IntResCrossMode=IntResCrossMode, IntResLong=IntResLong, IntResLongMode=IntResLongMode)
        return [Res0,Res1], Pts, LDetLim, Mode, steps, Thres, ThresMode, ThresMin, IntResCross, IntResCrossMode, IntResLong, IntResLongMode



def _Resolution_PpsEmissdV_Iso(Pt, DXTheta, size=0., Deg=0, Amp=1., IntResCross=[0.1,0.1], IntResCrossMode='rel', IntResLong=0.05, IntResLongMode='rel', VType='Tor'):
    Nxtheta = int(np.diff(DXTheta)/IntResLong) if IntResLongMode=='abs' else int(1./IntResLong)
    xtheta = np.linspace(DXTheta[0],DXTheta[1],Nxtheta)
    if size==0.:
        pp = np.array([Pt[0]*np.cos(xtheta), Pt[0]*np.sin(xtheta), Pt[1]*np.ones((Nxtheta,))]) if VType=='Tor' else np.array([xtheta, Pt[0]*np.ones((Nxtheta,)), Pt[1]*np.ones((Nxtheta,))])
        Emiss = Amp * np.ones((pp.shape[1],))
        dV = IntResLong
    else:
        # Get Number of points, make sure it is odd (to get the center)
        NRY = max(3,int(np.ceil(size/IntResCross[0]))) if IntResCrossMode=='abs' else int(1./IntResCross[0])
        NZ = max(3,int(np.ceil(size/IntResCross[1]))) if IntResCrossMode=='abs' else int(1./IntResCross[1])
        NRY = NRY+1 if NRY%2==0 else NRY
        NZ = NZ+1 if NZ%2==0 else NZ

        RY = np.linspace(Pt[0]-0.5*size,Pt[0]+0.5*size,NRY)
        Z = np.linspace(Pt[1]-0.5*size,Pt[1]+0.5*size,NZ)
        RRYY = np.tile(RY,(NZ,1)).flatten()
        ZZ = np.tile(Z,(NRY,1)).T.flatten()

        r = np.hypot(RRYY-Pt[0], ZZ-Pt[1])
        ind = r<=size/2.

        ds = IntResCross[1]*((RRYY+IntResCross[0]/2.)**2 - (RRYY-IntResCross[0]/2.)**2)/2. if VType=='Tor' else IntResCross[0]*IntResCross[1]*np.ones((RRYY.size,))
        dS = np.sum(ds[ind])

        if VType=='Tor':
            RRRYYY = np.tile(RRYY[ind],(Nxtheta,1)).flatten()
            ZZZ = np.tile(ZZ[ind],(Nxtheta,1)).flatten()
            TTT = np.tile(xtheta,(ind.sum(),1)).T.flatten()
            pp = np.array([RRRYYY*np.cos(TTT), RRRYYY*np.sin(TTT), ZZZ])
        else:
            pp = np.array([np.tile(xtheta,(ind.sum(),1)).T.flatten(), np.tile(RRYY[ind],(Nxtheta,1)).flatten(), np.tile(ZZ[ind],(Nxtheta,1)).flatten()])
        dV = IntResLong*np.tile(ds[ind],(Nxtheta,1)).flatten()
        if Deg==0:
            Emiss = Amp/dS * np.ones((ind.sum()*Nxtheta,))

    return pp, Emiss, dV




def _Plot_Resolution(GLD, ax=None, Pts=None, Res=[0.01,0.01], ResMode='abs', Mode='Iso', Amp=1., Deg=0, steps=0.001, Thres=0.05, ThresMode='rel', ThresMin=None,
                    IntResCross=[0.1,0.1], IntResCrossMode='rel', IntResLong=0.05, IntResLongMode='rel',
                    Eq=None, Cdict=dict(tfd.DetConed), tt=np.linspace(0.,2.*np.pi,100),
                    plotfunc='scatter', NC=20, CDictRes=None,
                    ind=None, Val=None, Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='In', Test=True):

    Res, LDetLim, Pts = _Calc_Resolution(GLD, Pts=Pts, Res=Res, ResMode=ResMode, Mode=Mode, Amp=Amp, Deg=Deg, steps=steps, Thres=Thres, ThresMode=ThresMode, ThresMin=ThresMin,
                                        IntResCross=IntResCross, IntResCrossMode=IntResCrossMode, IntResLong=IntResLong, IntResLongMode=IntResLongMode,
                                        Eq=Eq, PlotDetail=False, Cdict=Cdict, tt=tt,
                                        ind=ind, Val=Val, Crit=Crit, PreExp=PreExp, PostExp=PostExp, Log=Log, InOut=InOut, Test=Test)
    ax = _tfg_p._Resolution_Plot(Pts, Res, GLD, LDetLim, ax=ax, plotfunc=plotfunc, NC=NC, CDictRes=CDictRes,
                                ind=ind, Val=Val, Crit=Crit, PreExp=PreExp, PostExp=PostExp, Log=Log, InOut=InOut, Test=Test)
    return ax




