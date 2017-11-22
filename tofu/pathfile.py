# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 14:37:31 2014

@author: didiervezinet
"""
# Built-in
import os                   # For accessing cuurent working direcVesy
import subprocess
import getpass
import inspect
import warnings

# Common
import numpy as np
import datetime as dtm

try:
    import cPickle as pck       # For saving / loading objects
except Exception:
    pck = None
    warnings.warn("cPickle not available => all file-handling done with numpy !")
    pass


# ToFu specific
from tofu import __version__
from tofu.defaults import dtmFormat as TFDdtmFormat


__author__ = "Didier Vezinet"
__all__ = ["Find_Rootpath","get_DefaultPaths","get_Default_dtimeFmt","_get_PathFileExt_FromName","convert_units","get_PolyFromPolyFileObj",
           "ID",
           "CheckSameObj","Save_Generic","Open"]


"""
###############################################################################
###############################################################################
###############################################################################
                Path Handling
###############################################################################
###############################################################################
"""


def Find_Rootpath(Path=os.getcwd(),substr='/tofu/'):
    """
    Return the absolute path of the root directory of ToFu, searching for a pattern in the provided input path

    Parameters
    ----------
    Path :      str
        An absolute path in which a pattern is to be looked for
    substr :    str
        The pattern to look for

    Returns
    -------
    root :  str
        The absolute path containing the pattern

    """
    Path = (Path+'/').replace('//','/')
    indstr = Path.find(substr)
    if indstr==-1:
        substr = substr.lower()
        indstr = Path.find(substr)
    root = Path[:indstr] + substr
    root = root[:-1] if root[-1]=='/' else root
    return root



def get_DefaultPaths(RootPath=None, Plugin=None, lSubs=[], PathInp=None, PathObj=None, PathOut=None):
    """
    Return the default paths for input loading, object and output saving depending on the root path used
    """
    assert RootPath is None or type(RootPath) is str, "Arg RootPath must be a str !"
    assert Plugin is None or type(Plugin) is str, "Arg Plugin must be a str !"
    assert all([ss is None or type(ss) is str for ss in [PathInp, PathObj, PathOut]]), "Args PathInp, PathObj, PathOut must be None or str !"
    assert type(lSubs) is list and all([type(ss) is str for ss in lSubs])

    if RootPath is None:
        RootPath = Find_Rootpath()
    for pp, ss in [(PathInp,'Inputs'), (PathObj,'Objects'), (PathOut,'Outputs')]:
        if pp is None:
            pp = str(RootPath)
            if not Plugin is None:
                pp = pp + '/' + Plugin
            for sub in lSubs:
                pp = pp + '/' + sub
            pp = pp + '/' + ss

    return PathInp+'/', PathObj+'/', PathOut+'/'


def get_Default_dtimeFmt(dtime=None, dtFormat=TFDdtmFormat):
    """
    Return the default datetime value and format
    """
    if dtime is None:
        dtime = dtm.datetime.now()
    elif type(dtime) is str:
        dtime = dtm.strptime(dtime,dtFormat)
    return dtime, dtFormat


def _get_PathFileExt_FromName(Name=None, Path=os.getcwd(), Lstr=[]):    # Used in tofu.plugin.ITER.Ves.create()
    """ Retrieve PathFileExt from a Name, in Path if absolute path not specified in Name, with possible TFMod and ext options """
    if not Name is None and '/' in Name:
        PathFileExt = Name
    else:
        LF = os.listdir(Path)
        if type(Name) is str:
            Lstr.append(Name)
        elif type(Name) is list:
            Lstr = Lstr+Name
        if not Lstr==[]:
            LF = [ff for ff in LF if all([ss in ff for ss in Lstr])]
        assert len(LF)==1, "Several or no possibility for "+str(Name)+" in "+Path+" with "+str(Lstr)+"    LF = "+str(LF)
        PathFileExt = Path + LF[0]
    return PathFileExt



def _set_arrayorder(obj, arrayorder):
    assert arrayorder in ['C','F'], "Arg arrayorder must be in ['C','F']"
    Lattr = dir(obj)
    for aa in Lattr:
        bb = getattr(obj,aa)
        if type(bb) is np.array and bb.ndim>1:
            try:
                if arrayorder=='C':
                    setattr(obj,aa,np.ascontiguousarray(bb))
                else:
                    setattr(obj,aa,np.asfortranarray(bb))
            except Exception:
                pass
    obj._arrayorder = arrayorder


def convert_units(P, In='cm', Out='m'):
    """
    Quickly convert distance units between meters, centimeters and millimeters
    """
    c = {'m':{'mm':1000.,'cm':100.,'m':1.},
            'cm':{'mm':10.,'cm':1.,'m':0.01},
            'mm':{'mm':1.,'cm':0.1,'m':0.001}}
    return c[In][Out]*P



def get_PolyFromPolyFileObj(PolyFileObj, SavePathInp=None, units='m', comments='#', skiprows=0, shape0=2):
    """ Return a polygon as a np.ndarray, extracted from a txt file or from a ToFu object, with appropriate units

    Useful for :meth:`tofu.plugins.AUG.Ves._create()`

    Parameters
    ----------
    PolyFileObj :   str / :mod:`tofu.geom` object / np.ndarray
        The source where the polygon is to be found, either:
            - str: the name of a file containing the coorindates of a polygon to be loaded with :meth:`numpy.loadtxt()`
            - A :mod:`tofu.geom` object: with attribute 'Poly'
            - np.ndarray: an 2-dimensional array containing the 2D cartesian coordinates of a polygon
    SavePathInp :   str / None
        The absolute path where the input file is stored
    units :         str
        Flag indicating in which units the polygon coordinates is expressed in the input file / object / array (will be converted to meters)
    comments :      str
        Parameter to be fed to :meth:`numpy.loadtxt()` if PolyFileObj is a file name
    skiprows :      int
        Parameter to be fed to :meth:`numpy.loadtxt()` if PolyFileObj is a file name
    shape0 :          int
        Specifies whether the loaded array is a (2,N) or (3,N) array (transposed it if necessary)

    Returns
    -------
    Poly :          np.ndarray
        (2,N) np.ndarray containing the 2D cartesian coordinates of the polygon, where N is the number of points
    addInfo :       dict
        Dictionaryb containing information on the origin of the polygon, for the record (e.g.: the name and absolute path of the file from which it was extracted)

    """
    assert type(PolyFileObj) in [list,str] or hasattr(PolyFileObj,"Poly") or np.asarray(PolyFileObj).ndim==2, "Arg PolyFileObj must be str (PathFileExt), a ToFu object with attribute Poly or an iterable convertible to 2d np.ndarray !"

    # Load PolyFileObj if file and check shape
    addInfo = {}
    if type(PolyFileObj) in [list,str]:
        PathFileExt = _get_PathFileExt_FromName(PolyFileObj, Path=SavePathInp, Lstr=[])
        # Include PathFileExt in ID for tracability
        addInfo = {'Input':PathFileExt}
        PolyFileObj = np.loadtxt(PathFileExt, dtype=float, comments=comments, delimiter=None, converters=None, skiprows=skiprows, usecols=None, unpack=False, ndmin=2)
    elif hasattr(PolyFileObj,"Poly"):
        addInfo = {'Input':PolyFileObj.Id.SaveName}
        PolyFileObj = PolyFileObj.Poly

    Poly = np.asarray(PolyFileObj)
    assert Poly.ndim==2 and shape0 in Poly.shape and max(Poly.shape)>=3 and not np.any(np.isnan(Poly)), "Arg np.asarray(PolyFileObj) must be a (2,N) or (N,2) np.ndarray with non NaNs !"
    Poly = Poly if Poly.shape[0]==shape0 else Poly.T
    Poly = convert_units(Poly, In=units, Out='m')
    return Poly, addInfo






"""
###############################################################################
###############################################################################
###############################################################################
                Generic input checking and retrieving
###############################################################################
###############################################################################
"""


def _check_NotNone(Dict):
    for aa in Dict.keys():
        assert not Dict[aa] is None, "Arg "+aa+" must not be None !"


def _get_FromItself(obj, Dict):
    for aa in Dict.keys():
        if Dict[aa] is None:
            try:
                Dict[aa] = getattr(obj,aa)
            except:
                pass
    return Dict




"""
###############################################################################
###############################################################################
###############################################################################
                ID Class and naming
###############################################################################
###############################################################################
"""


class ID(object):
    """ A class used by all ToFu objects as an attribute, storing all relevant data for the identification of created instances and providing default path and names for saving objects

    Each detector created in ToFu should be identifiable thanks to user-friendly criteria, like its name, the diagnostic and experiment it is attached to, the shot number from from it was physically installed on the experiement...
    Users can also freely add some information they consider relevant, such as some characteristics of the detector (size, manufacturer, date of last calibration...)

    Parameters
    ----------
    Cls :       str
        Flag indicating which class is the object fro which the ID instance is being created, available class flags are:
            - From :mod:`tofu.geom`: 'Ves', 'Struct', 'LOS', 'GLOS', 'Apert', 'Lens', 'Detect' and 'GDetect'
            - From :mod:`tofu.mesh`: 'Mesh1D', 'Mesh2D', 'LBF1D' and 'LBF2D' (to be finished)
            - From :mod:`tofu.`: (to be completed)
    Name :      str
        Name to be assigned to the created instance, should be a str without ' ' or '_' (spaces and underscores will be automatically removed if present)
    Type :      None / str
        If provided (necessary for some objects, but not for all), specifies the Type of object (i.e.: 'Tor' or 'Lin' for a Ves instance)
    Deg :       None / int
        If provided (necessary only for objects of class 'LBF1D' and 'LBF2D'), specifies the degree of the b-splines constituting the :mod:`tofu.mesh` object
    Exp :       None / str
        A short 3-4 letters max flag specifying the experiment to which the created instance belongs (e.g.: 'AUG', 'ITER', 'TCV', 'JET'...)
    Diag :      None / str
        A short flag indicating which diagnostic the the created instance belongs to (e.g.: 'SXR', 'HXR', 'Bolo'...)
    shot :      None / int
        A shot number from which the created instance can be considered valid (useful for tracking geometry changes in the case of Ves and Detect objects)
    SaveName :  None / str
        If provided, overrides the automatically generated name for saving the created instance (not recommended)
    SavePath :  None / str
        If provided, overrides the automatically generated path for saving the created instance
    USRdict :   None / dict
        If provided, a user-defined dictionary containing information about the instance considered relevant (e.g.: thickness of the diode, date of installation...)
    LObj :      None / dict / list
        If provided, either
            - list: list of other ID instances of objects on which the created object depends (this list will then be sorted by class and formatted into a dictionary storign key attributes)
            - dict: a ready-made such dictionary
    dtime :     None / dtm.datetime
        If provided, a time reference to be used to identify this particular instance (used for debugging mostly)
    dtFormat :  None / str
        If provided, the format in which dtime should be written in the automatically generated saving name for the created instance

    Returns
    -------
    Id          ID instance
        The created ID instance, with all necessary computed attributes and methods

    """

    def __init__(self, Cls, Name, Type=None, Deg=None, Exp=None, Diag=None, shot=None, SaveName=None, SavePath=None, USRdict=None, LObj=None, dtime=None, dtFormat=TFDdtmFormat, Include=['Mod','Cls','Typ','Exp','Deg','Diag','Name','shot','Vers','Usr'], version=None):
        assert type(Exp) is str, "Arg Exp must be a str !"
        assert type(Cls) is str, "Arg Cls must be a str !"
        assert shot is None or type(shot) is int, "Arg shot must be a int !"
        assert Diag is None or type(Diag) is str, "Arg Diag must be a str !"
        assert Type is None or type(Type) is str, "Arg Type must be a str !"

        if version is None:
            self._version = __version__
        else:
            self._version = version
            if not version==__version__:
                warnings.warn(Id.Name +" was created from a different version of ToFu !!!    %s vs %s" % (version, __version__))
        self._Exp = Exp
        self._Cls = Cls
        self._Type = Type
        self._Deg = Deg
        self._Diag = Diag
        self._shot = shot
        try:
            self._usr = getpass.getuser()
        except:
            self._usr = None 
        self._Include = None
        self._set_dtime(dtime=dtime, dtFormat=dtFormat)
        self.set_Name(Name, SaveName=SaveName, Include=Include)
        self.set_SavePath(SavePath=SavePath)
        self._LObj = {}
        self.set_LObj(LObj)
        self.set_USRdict(USRdict)

    def _set_dtime(self,dtime=None, dtFormat=TFDdtmFormat):
        assert dtime is None or isinstance(dtime,dtm.datetime), "Arg dtime must be a dtm.datetime instance !"
        if dtime is None:
            dtime = dtm.datetime.now()
        self._dtime = dtime
        self._dtFormat = dtFormat

    def set_Name(self, Name, SaveName=None, Include=['Mod','Cls','Typ','Exp','Deg','Diag','Name','shot','Vers','Usr']):
        """ Set the Name of the created instance, automatically updating the SaveName

        When the name is changed (reminding it should not include space (' ') or underscore ('_') characters), the name used for saving the object is automatically changed

        Parameters
        ----------
        Name :      str
            Name to be assigned to the created instance, should be a str without ' ' or '_' (spaces and underscores will be automatically removed if present)
        SaveName :  None / str
            If provided, overrides the automatically generated name for saving the created instance (not recommended)

        """
        assert type(Name) is str, "ID.Name must be assigned to a str !"
        self._Name = Name
        self.set_SaveName(SaveName=SaveName, Include=Include)

    def set_SaveName(self,SaveName=None, Include=['Mod','Cls','Typ','Exp','Deg','Diag','Name','shot','Vers','Usr']):
        """ Enables either to automatically compute a ToFu-consistent saving name for the created instance, or to override that default saving name with the user-provided SaveName

        When creating an object of any class, an ID object is assigned to it that automatically computes a saving name in case the user wants to save the object.
        This automatically generated saving name includes not only the name given to the instance but also useful classifying info such as the class of the object, its Type, experiment, diagnostic...
        It is not recommended to override this automatically generated saving name (because some routines rely on it for fast identification of saved files), but it is made possible for flexibility.

        Parameters
        ----------
        SaveName :  None / str
            If provided, a str that overrides the automatically generated saving name

        """
        assert SaveName is None or type(SaveName) is str, "ID.SaveName must be assigned to a str !"
        if SaveName is None:
            self._SaveName = SaveName_Conv(self._Exp, self.Cls, self._Type, self._Deg, self._Diag, self._Name, shot=self._shot, version=self._version, usr=self._usr, dtime=self._dtime, Format=self._dtFormat, Include=Include)
            self._Include = Include
        else:
            self._SaveName = SaveName
            self._Include = None

    def set_SavePath(self,SavePath=None):
        """ Enables to automatically generate a saving path for the created object, or to override that default path with the user-provided SavePath

        Similarily to SaveName, ToFu automatically generates a saving path for any created object.
        This saving path can be overriden if desired.
        This is less crucial than SaveName, changing the SavePath has little consequences as long as you remember what you are doing.

        Parameters
        ----------
        SavePath :  None / str
            If provided, a str that overrides the automatically generated saving path

        """
        assert SavePath is None or type(SavePath) is str, "ID.SavePath must be assigned to a str !"
        if SavePath is None:
            SavePath = Find_Rootpath()+'/Objects_'+self.Exp+'/' if not self.Cls in ['PreData','Sol2D'] else Find_Rootpath()+'/Outputs_'+self.Exp+'/'
        self._SavePath = SavePath.replace('//','/')

    def set_LObj(self,LObj=None):
        """ Set the LObj attribute to store the list of ID of other objects the created instance depends on

        A Detect object depends on a vessel and some optics (e.g.: a list of apertures).
        It is necessary that the link between the created Detect object and the already-existing Ves and Apert objects be stored somewhere, so that even after saving and closing the session, this correspondence can be retrieved and the Detect object can be re-loaded with links to the proper Ves and Apert objects, themselves beoing possibly saved elsewhere (so their respective SavePath must also be stored).
        The LObj parameter does this: it stores all information necessary about each of the other objects the created instance depends on, mostly by storing their ID attributes as dictionaries.

        Parameters
        ----------
        LObj :  None / dict / list
            If provided, either
                - list: list of other ID instances of objects on which the created object depends (this list will then be sorted by class and formatted into a dictionary storign key attributes)
                - dict: a ready-made such dictionary

        """
        assert LObj is None or type(LObj) is dict or (type(LObj) is list and all([isinstance(oo,ID) for oo in LObj])), "Arg LObj must be a list of ID instances !"
        if type(LObj) is list:
            LCls = [oo.Cls for oo in LObj]
            ClsU = list(set(LCls))
            for Cls in ClsU:
                self._LObj[Cls] = {'Exp':[oo.Exp for oo in LObj if oo.Cls==Cls], 'Name':[oo.Name for oo in LObj if oo.Cls==Cls], 'SaveName':[oo.SaveName for oo in LObj if oo.Cls==Cls],
                        'SavePath':[oo.SavePath for oo in LObj if oo.Cls==Cls], 'Type':[oo.Type for oo in LObj if oo.Cls==Cls], 'Diag':[oo.Diag for oo in LObj if oo.Cls==Cls],
                        'shot':[oo.shot for oo in LObj if oo.Cls==Cls],
                        'dtFormat':[oo._dtFormat for oo in LObj if oo.Cls==Cls], 'dtime':[oo.dtime.strftime(oo._dtFormat) for oo in LObj if oo.Cls==Cls],
                        'USRdict':[oo._USRdict for oo in LObj if oo.Cls==Cls], 'LObj':[oo._LObj for oo in LObj if oo.Cls==Cls]}
        elif type(LObj) is dict:
            self._LObj = LObj

    def get_LObjasLId(self,Cls=None):
        if Cls is None:
            Cls = sorted([self._LObj.keys()])
        Cls = [Cls] if type(Cls) is str else Cls
        DId = {}
        for ii in range(0,len(Cls)):
            lid = []
            keys = self._LObj[Cls[ii]].keys()
            for jj in range(0,len(self._LObj[Cls[ii]]['Exp'])):
                dd = {}
                for kk in keys:
                    dd[kk] = self._LObj[Cls[ii]][kk][jj]
                name = dd['Name']
                del dd['Name'], dd['dtime']
                lid.append(ID(Cls[ii], name, **dd))
            DId[Cls[ii]] = lid
        return DId


    def set_USRdict(self,USRdict=None):
        """ Set the USRdict to enable the user to store arbitrary information about the instance created

        It may sometimes be useful to store unforeseen arbitrary info about some objects created, like the manufacturing date of a diode, the material used for a filter...
        The USRdict attribute is a user-defined dictionary for this purpose.

        Parameters
        ----------
        USRdict :   None / dict
            If provided, a user-defined dictionary containing information about the instance considered relevant (e.g.: thickness of the diode, date of installation...)

        """
        assert USRdict is None or type(USRdict) is dict, "Arg USRdict must be a dictionnary !"
        USRdict = {} if USRdict is None else USRdict
        self._USRdict = USRdict

    @property
    def Cls(self):
        return self._Cls
    @property
    def Name(self):
        return self._Name
    @Name.setter
    def Name(self,Val):
        self.set_Name(Val)
    @property
    def NameLTX(self):
        return r"$"+self.Name.replace('_','\_')+r"$"
    @property
    def Diag(self):
        return self._Diag
    @property
    def shot(self):
        return self._shot
    @property
    def Type(self):
        return self._Type
    @property
    def SaveName(self):
        return self._SaveName
    @SaveName.setter
    def SaveName(self,Val):
        self.set_SaveName(Val)
    @property
    def SavePath(self):
        return self._SavePath
    @SavePath.setter
    def SavePath(self,Val):
        self.set_SavePath(Val)
    @property
    def dtime(self):
        return self._dtime
    @property
    def Exp(self):
        return self._Exp
    @property
    def LObj(self):
        return self._LObj
    @property
    def USRdict(self):
        return self._USRdict
    @USRdict.setter
    def USRdict(self, Val):
        self.set_USRdict(Val)

    def todict(self):
        return _Id_todict(self)



def _get_Str4SVN(kstr, Exp=None, Cls=None, Type=None, Deg=None, Diag=None, Name=None, shot=None, version=None, usr=None, dtime=None, Format="D%Y%m%d_T%H%M%S"):
    if kstr=='Mod' and all([Cls is not None]):
        modes = {'TFG':['Ves','Struct','LOS','GLOS','Lens','Apert','Detect','GDetect'],
                 'TFM':['Mesh1D','Mesh2D','Mesh3D','LBF1D','LBF2D','LBF3D'],
                 'TFEq':['Eq2D'],
                 'TFMC':['GMat2D','GMat3D'],
                 'TFD':['Corresp','PreData'],
                 'TFI':['Sol2D']}
        Str = '_'+[kk for kk in modes.keys() if Cls in modes[kk]][0]
    elif kstr=='Cls' and Cls is not None:
        Str = '_'+Cls
    elif kstr=='Typ' and Type is not None:
        Str = Type
    elif kstr=='Exp' and Exp is not None:
        Str = '_'+Exp
    elif kstr=='Deg' and Deg is not None:
        Str = 'D{0:01.0f}'.format(Deg)
    elif kstr=='Diag' and Diag is not None:
        Str = '_Dg'+Diag
    elif kstr=='Name' and Name is not None:
        Str = '_'+Name
    elif kstr=='shot' and shot is not None:
        Str = '_sh{0:05.0f}'.format(shot)
    elif kstr=='Vers' and version is not None:
        Str = '_V'+version
    elif kstr=='Usr' and usr is not None:
        Str = '_U'+usr
    elif kstr=='DTM' and dtime is not None:
        Str = '_'+dtime.strftime(Format)
    else:
        Str = ''
    return Str



def SaveName_Conv(Exp, Cls, Type, Deg, Diag, Name, shot=None, version=None, usr=None, dtime=None, Format="D%Y%m%d_T%H%M%S", Include=['Mod','Cls','Typ','Exp','Deg','Diag','Name','shot','Vers','Usr']):
    """ Create a default name for saving the object, including key info for fast identification of the object class, type, experiement...

    When create a ToFu object, this function called by the ID class to generate a default name for saving the object.
    This automatically generated name includes all key info about the object, in addition to its user-defined name.
    It is recommended to use this default name.

    Parameters
    ----------
    Exp :       None / str
        A short 3-4 letters max flag specifying the experiment to which the created instance belongs (e.g.: 'AUG', 'ITER', 'TCV', 'JET'...)
    Cls :       None / str
        Flag indicating which class is the object fro which the ID instance is being created, available class flags are:
            - From :mod:`tofu.geom`: 'Ves', 'Struct', 'LOS', 'GLOS', 'Apert', 'Lens', 'Detect' and 'GDetect'
            - From :mod:`tofu.mesh`: 'Mesh1D', 'Mesh2D', 'LBF1D' and 'LBF2D' (to be finished)
            - From :mod:`tofu.`: (to be completed)
    Type :      None / str
        If provided (necessary for some objects, but not for all), specifies the Type of object (i.e.: 'Tor' or 'Lin' for a Ves instance)
    Deg :       None / int
        If provided (necessary only for objects of class 'LBF1D' and 'LBF2D'), specifies the degree of the b-splines constituting the :mod:`tofu.mesh` object
    Diag :      None / str
        A short flag indicating which diagnostic the the created instance belongs to (e.g.: 'SXR', 'HXR', 'Bolo'...)
    Name :      str
        Name to be assigned to the created instance, should be a str without ' ' or '_' (spaces and underscores will be automatically removed if present)
    shot :      None / int
        A shot number from which the created instance can be considered valid (useful for tracking geometry changes in the case of Ves and Detect objects)
    dtime :     None / dtm.datetime
        If provided, a time reference to be used to identify this particular instance (used for debugging mostly)
    Format :    None / str
        If provided, the format in which dtime should be written in the automatically generated saving name for the created instance

    Returns
    -------
    SVN :       str
        The automatically generated saving name

    """
    if Cls=='PreData':
        Order = ['Mod','Cls','Typ','Exp','Deg','Diag','shot','Name','Vers','Usr','DTM']
    else:
        Order = ['Mod','Cls','Typ','Exp','Deg','Diag','Name','shot','Vers','Usr','DTM']

    SVN = ""
    for ii in range(0,len(Order)):
        if Order[ii] in Include:
            Str = _get_Str4SVN(Order[ii], Exp=Exp, Cls=Cls, Type=Type, Deg=Deg, Diag=Diag, Name=Name, shot=shot, version=version, usr=usr, dtime=dtime, Format=Format)
            SVN += Str
    SVN = SVN.replace('__','_')
    if SVN[0]=='_':
        SVN = SVN[1:]
    return SVN

"""
    if Cls in ['Ves','Struct','LOS','GLOS','Lens','Apert','Detect','GDetect']:
        Mod = 'TFG'
    elif Cls in ['Mesh1D','Mesh2D','Mesh3D','LBF1D','LBF2D','LBF3D']:
        Mod = 'TFM'
    elif Cls in ['Eq2D']:
        Mod = 'TFEq'
    elif Cls in ['GMat2D','GMat3D']:
        Mod = 'TFMC'
    elif Cls in ['Corresp','PreData']:
        Mod = 'TFT'
    elif Cls=='Sol2D':
        Mod = 'TFI'
    if not isinstance(dtime,dtm.datetime):
        dtime = dtm.datetime.now()
    ClsType = Cls if Type is None else Cls+Type
    if Cls=='PreData':
        Elts = [(ClsType,''),(Exp,''),(Deg,'D','01.0f'),(Diag,'Dg'),(shot,'sh','05.0f'),(Name,'')]
    else:
        Elts = [(ClsType,''),(Exp,''),(Deg,'D','01.0f'),(Diag,'Dg'),(Name,''),(shot,'sh','05.0f')]
    SVN = Mod
    for ii in range(0,len(Elts)):
        if len(Elts[ii])==3 and Elts[ii][0] is not None:
            temp = "{0:"+Elts[ii][2]+"}"
            SVN = SVN + '_' + Elts[ii][1] + temp.format(Elts[ii][0])
        elif Elts[ii][0] is not None:
            SVN = SVN + '_' + Elts[ii][1] + str(Elts[ii][0]).replace('_','')
    if dtimeIn:
        SVN = SVN+'_'+dtime.strftime(Format)
    SVN = SVN.replace('__','_')
    if version is not None:
        SVN += '_V' + version
    if usr is not None:
        SVN = SVN + '_U' + usr
    return SVN
"""

# Checking several instances are the same object
def CheckSameObj(obj0, obj1, LFields=None):
    """ Check two variable refer to the same instance of a ToFu class by checking some key attributes

    Sometimes two different variables can refer to the same instance (for example if an object was created and assigned to obj0, then later saved and loaded and assigned to obj1).
    Occasionally it may be useful to check whether two variables really represent the same instance, according to important criteria from the point of view of ToFu.

    Parameters
    ----------
    obj0 :      tofu object
        A variable refering to a ToFu object of any class
    obj1 :      tofu object
        A variable refering to a ToFu object of any class (but the same class as obj0)
    LFields :   None / str / list
        The criteria against which the two objects are evaluated, if not None, must be str matching an attribute of the ID class or an attribute of the object class itself (or a list of such)
            - None: True is returned
            - str or list: tests whether the attributes have the same value or not and only returns True if all do

    Returns
    -------
    A :     bool

    """
    assert LFields is None or type(LFields) is str or (type(LFields) is list and all([type(ss) is str for ss in LFields])), "Arg LFields must be a str or a list of str !"
    if LFields is None:
        return True
    if not type(LFields) is list:
        LFields = [LFields]
    ind = [False for ii in range(0,len(LFields))]
    for ii in range(0,len(LFields)):
        assert LFields[ii] in dir(obj0.Id)+dir(obj0), LFields[ii]+" not in "+obj0.Id.Name+" !"
        assert LFields[ii] in dir(obj1.Id)+dir(obj1), LFields[ii]+" not in "+obj1.Id.Name+" !"
        if LFields[ii] in dir(obj0):
            ind[ii] = np.all(getattr(obj0,LFields[ii])==getattr(obj1,LFields[ii]))
        else:
            ind[ii] = getattr(obj0.Id,LFields[ii])==getattr(obj1.Id,LFields[ii])
    A = all(ind)
    return A


def SelectFromIdLObj(IdLObjCls, Val=None, Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='In', Out=bool):
    """ To do (deprecated ?)
    """
    assert type(Crit) is str or (type(Crit) is list and all([type(cc) is str for cc in Crit])), "Arg Crit must be a str or list of str !"
    assert all([rr is None or type(rr) is str or (type(rr) is list and all([type(ee) is str for ee in rr])) for rr in [PreExp,PostExp]]), "Args PreExp and PostExp must be a str or list of str !"
    assert Log in ['any','all'], "Arg Log must be in ['and','or'] !"
    assert InOut in ['In','Out'], "Arg InOut must be in ['In','Out'] !"
    NObj = len(IdLObjCls['Name'])
    if Val is None and PreExp is None and PostExp is None:
        ind = np.ones((1,NObj),dtype=bool)
    elif not Val is None:
        if type(Val) is str:
            Val=[Val]
        N = len(Val)
        ind = np.zeros((N,NObj),dtype=bool)
        if Crit in dir(ID):
            for ii in range(0,N):
                ind[ii,:] = np.asarray([idd==Val[ii] for idd in IdLObjCls[Crit]],dtype=bool)
        else:
            for ii in range(0,N):
                ind[ii,:] = np.asarray([idd[Crit]==Val[ii] for idd in IdLObjCls['USRdict']],dtype=bool)
    else:
        if type(PreExp) is str:
            PreExp = [PreExp]
        if type(PostExp) is str:
            PostExp = [PostExp]
        if PreExp is None:
            PreExp = ["" for ss in PostExp]
        if PostExp is None:
            PostExp = ["" for ss in PreExp]
        assert len(PreExp)==len(PostExp), "Arg Exp must be a list of same length as Crit !"
        N = len(PreExp)
        ind = np.zeros((N,NObj),dtype=bool)
        if Crit in dir(ID):
            for ii in range(0,N):
                ind[ii,:] = np.asarray([eval(PreExp[ii]+" idd "+PostExp[ii]) for idd in IdLObjCls[Crit]],dtype=bool)
        else:
            for ii in range(0,N):
                ind[ii,:] = np.asarray([eval(PreExp[ii]+" idd[Crit] "+PostExp[ii]) for idd in IdLObjCls['USRdict']],dtype=bool)
    ind = np.any(ind,axis=0) if Log=='any' else np.all(ind,axis=0)
    if InOut=='Out':
        ind = ~ind
    if Out==bool:
        return ind
    elif Out==int:
        return ind.nonzero()[0]
    else:
        if Out in dir(ID):
            return [IdLObjCls[Out][ii] for ii in ind.nonzero()[0]]
        else:
            return [IdLObjCls['USRdict'][ii][Out] for ii in ind.nonzero()[0]]



def SelectFromListId(LId, Val=None, Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='In', Out=bool):
    """ Select the ID instances in the provided list that match the criteria and return their index or other attributes

    Return

    Parameters
    ----------
    LId :       list

    Val :       None / str / list

    Crit :      str

    PreExp :    None / str

    PostExp :   None / str

    Log :       str

    InOut :     str

    Out :       type / str


    Returns
    -------
    ind :       np.ndarray / list


    """
    assert type(Crit) is str or (type(Crit) is list and all([type(cc) is str for cc in Crit])), "Arg Crit must be a str or list of str !"
    assert all([rr is None or type(rr) is str or (type(rr) is list and all([type(ee) is str for ee in rr])) for rr in [PreExp,PostExp]]), "Args PreExp and PostExp must be a str or list of str !"
    assert Log in ['any','all'], "Arg Log must be in ['any','all'] !"
    assert InOut in ['In','Out'], "Arg InOut must be in ['In','Out'] !"
    if Val is None and PreExp is None and PostExp is None:
        ind = np.ones((1,len(LId)),dtype=bool)
    elif not Val is None:
        if type(Val) is str:
            Val=[Val]
        N = len(Val)
        ind = np.zeros((N,len(LId)),dtype=bool)
        if Crit in dir(ID):
            for ii in range(0,N):
                ind[ii,:] = np.asarray([getattr(iid,Crit)==Val[ii] for iid in LId],dtype=bool)
        else:
            for ii in range(0,N):
                ind[ii,:] = np.asarray([iid.USRdict[Crit]==Val[ii] for iid in LId],dtype=bool)
    else:
        if type(PreExp) is str:
            PreExp = [PreExp]
        if type(PostExp) is str:
            PostExp = [PostExp]
        if PreExp is None:
            PreExp = ["" for ss in PostExp]
        if PostExp is None:
            PostExp = ["" for ss in PreExp]
        assert len(PreExp)==len(PostExp), "Arg Exp must be a list of same length as Crit !"
        N = len(PreExp)
        ind = np.zeros((N,len(LId)),dtype=bool)
        if Crit in dir(ID):
            for ii in range(0,N):
                ind[ii,:] = np.asarray([eval(PreExp[ii]+" getattr(iid,Crit) "+PostExp[ii]) for iid in LId],dtype=bool)
        else:
            for ii in range(0,N):
                ind[ii,:] = np.asarray([eval(PreExp[ii]+" iid.USRdict[Crit] "+PostExp[ii]) for iid in LId],dtype=bool)
    ind = np.any(ind,axis=0) if Log=='any' else np.all(ind,axis=0)
    if InOut=='Out':
        ind = ~ind
    if Out==int:
        ind = ind.nonzero()[0]
    elif Out is not bool and hasattr(ID,Out):
        ind = [getattr(LId[ii],Out) for ii in ind.nonzero()[0]]
    elif Out is not bool and Out in LId[0].USRdict.keys():
        ind = [LId[ii].USRdict[Out] for ii in ind.nonzero()[0]]
    return ind



def _Id_todict(Id):
    IdTxt = {'version':Id._version, 'Cls':Id.Cls, 'Name':Id.Name, 'SaveName':Id.SaveName, 'SavePath':Id.SavePath, 'Diag':Id.Diag, 'Type':Id.Type, 'shot':Id.shot, 'Exp':Id.Exp}
    Iddtime = {'dtime':Id.dtime, 'dtFormat':Id._dtFormat}
    IdLobjUsr = {'LObj':Id.LObj, 'USRdict':Id.USRdict}
    return [IdTxt,Iddtime,IdLobjUsr]




def _Id_recreateFromdict(IdS):
    Id = ID(Cls=IdS[0]['Cls'], Type=IdS[0]['Type'], Exp=IdS[0]['Exp'], Diag=IdS[0]['Diag'], shot=IdS[0]['shot'], Name=IdS[0]['Name'], SaveName=IdS[0]['SaveName'], SavePath=IdS[0]['SavePath'],
            dtime=IdS[1]['dtime'], dtFormat=IdS[1]['dtFormat'],
            LObj=IdS[2]['LObj'], USRdict=IdS[2]['USRdict'], version=IdS[0]['version'])
    return Id





"""
###############################################################################
###############################################################################
###############################################################################
        Saving and loading ToFu objects with numpy and cPickle
###############################################################################
###############################################################################
"""


###########################
#   IDentify a Sol2D file
###########################

def FindSolFile(shot=0, t=0, Dt=None, Mesh='Rough1', Deg=2, Deriv='D2N2', Sep=True, Pos=True, OutPath='/afs/ipp-garching.mpg.de/home/d/didiv/Python/tofu/src/Outputs_AUG/'):
    """ Identify the good Sol2D saved file in a given folder (OutPath), based on key ToFu criteria

    When trying to load a Sol2D object (i.e.: solution of a tomographic inversion), it may be handy to provide the key parameters (shot, time, mesh name, degree of basis functions, regularisation functional) instead of copy-pasting the full file name.
    This function identifies, within the relevant repository (OutPath), the files matching the provided criteria.
    This function only works of the automatically generated default SaveName was preserved for the Sol2D objects.

    Parameters
    ----------
    shot :      int
        A shot number
    t :         None / int / float
        A time value that must be contained in the time interval of the Sol2D file, must be provided if Dt is None
    Dt :        None / iterable
        A time interval that the Sol2D file has to match, must be provided if t is None
    Mesh :      str
        The name of the mesh that was used to compute the inversion
    Deg :       int
        The of the b-splines (LBF2D object) that were used to discretize the solution
    Deriv :     str
        The flag indicating the regularization functional that was used for the inversion
    Sep :       bool
        The flag value that was used for indicating whether the boundary constraint at the separatrix should be considered
    Pos :       bool
        The flag value that was used for indicating whether the positivity constraint was considered
    Outpath :   str
        The absolute path of the repository where to look

    Returns
    -------
    out :       None / str
        The matching file name, if any

    """
    assert None in [t,Dt] and not (t is None and Dt is None), "Arg t or Dt must be None, but not both !"
    LF = [ff for ff in os.listdir(OutPath) if 'TFI_Sol2D_AUG_SXR' in ff]
    LF = [ff for ff in LF if all([ss in ff for ss in ['_'+str(shot)+'_', '_'+Mesh+'_D'+str(Deg), '_Deriv'+Deriv+'_Sep'+str(Sep)+'_Pos'+str(Pos)]])]
    if len(LF)==0:
        print("No matching Sol2D file in ", OutPath)
        out = None
    LDTstr = [ff[ff.index('_Dt')+3:ff.index('s_')] for ff in LF]
    LDTstr = [(ss[:7],ss[8:]) for ss in LDTstr]
    if t is None:
        LF = [LF[ii] for ii in range(0,len(LF)) if LDTstr[ii][0]+'-'+LDTstr[ii][1]=='{0:07.4f}-{1:07.4f}'.format(Dt[0],Dt[1])]
    elif Dt is None:
        LF = [LF[ii] for ii in range(0,len(LF)) if t>=float(LDTstr[ii][0]) and t<=float(LDTstr[ii][1])]
    if len(LF)==0:
        print("No matching Sol2D file in ", OutPath)
        out = None
    elif len(LF)>1:
        print("Several matching Sol2D files in ", OutPath)
        print(LF)
        out = None
    else:
        out = LF[0]
    return out



def _get_ClsFromName(PathFileExt):
    assert type(PathFileExt) is str, "Arg PathFileExt must be a str !"
    LCls = ['Ves','LOS','GLOS','Apert','Lens','Detect','GDetect','Mesh1D','Mesh2D','BF1D','BF2D','PreData','Sol2D']
    Found = False
    for nn in LCls:
        if '_'+nn+'_' in PathFileExt:
            Found = True
            break
    assert Found, "Class could not be identified in "+PathFileExt
    return nn



###########################
#   Saving
###########################


def Save_Generic(obj, SaveName=None, Path=None, Mode='npz', compressed=False, Print=True):
    """ Save a ToFu object under file name SaveName, in folder Path, using specified mode

    ToFu provides built-in saving and loading functions for ToFu objects.
    They can be saved at their default SavePath under their default SaveName or user-defined values can be forced if necessary.
    Saving can be done in two ways :
        - by direct object saving using cPickle (straightforward but heavy)
        - by mapping the key object attributes to a dictionary and using :meth:`numpy.savez_compressed()` (faster and lighter, recommended)

    ToFu now automatically saves information on smaller objects on which the object of interest depends (like apertures for detectors), so that all info is stored in a single file.
    In particular, provided the Ves object is saved separately, a whole camera can be saved in a single file (i.e.: all detectors and apertures).

    Parameters
    ----------
    SaveName :      str
        The name to be used to for the saved file, if None (recommended) uses obj.Id.SaveName
    Path :          str
        Path specifying where to save the file, if None (recommended) uses obj.Id.SavePath
    Mode :          str
        Flag specifying whether to save the object as a numpy array file ('.npz', recommended) or an object using cPickle (not recommended, may cause retro-compatibility issues with later versions)
    compressed :    bool
        Flag, used when Mode='npz', indicating whether to use np.savez or np.savez_compressed (slower saving and loading but smaller files)

    """
    if Path is None:
        Path = obj.Id.SavePath
    else:
        assert type(Path) is str, "Arg Path must be a str !"
        obj._Id.SavePath = Path
    if SaveName is None:
        SaveName = obj.Id.SaveName
    else:
        assert type(SaveName) is str, "Arg SaveName must be a str !"
        obj.Id.SaveName = SaveName
    Ext = '.npz' if 'npz' in Mode.lower() or pck is None else '.pck'
    pathfileext = os.path.join(Path,SaveName+Ext)
    if '.npz' in Ext:
        _save_np(obj, pathfileext, compressed=compressed)
    else:
        _save_object(obj, pathfileext)
    if Print:
        print("Saved in :  "+pathfileext)


def _save_object(obj,pathfileext):
    if 'TFM' in pathfileext:
        obj.LFunc = None
    with open(pathfileext, 'wb') as output:
        pck.dump(obj, output, -1)



def _convert_Detect2Ldict(obj):
    # Store LOS data
    llos = obj.LOS.keys()
    LOSprops = {'Keys':llos, 'Id':[obj.LOS[kk]['LOS'].Id.todict() for kk in llos], 'Du':[(obj.LOS[kk]['LOS'].D,obj.LOS[kk]['LOS'].u) for kk in llos]}
    lprops = obj.LOS[kk].keys()
    for pp in lprops:
        if not pp=='LOS':
            LOSprops[pp] = [obj.LOS[kk][pp] for kk in llos]

    # Get all attributes
    lAttr = dir(obj)
    Sino, Span, Cone, SAng, SynthDiag, Res = {}, {}, {}, {}, {}, {}

    # Store Sino data
    for pp in lAttr:
        #print( inspect.ismethod(getattr(obj,pp)), type(getattr(obj,pp)), pp
        if not inspect.ismethod(getattr(obj,pp)):
            if '_Sino' in pp:
                Sino[pp] = getattr(obj,pp)
            elif '_Span' in pp:
                Span[pp] = getattr(obj,pp)
            elif '_Cone' in pp:
                Cone[pp] = getattr(obj,pp)
            elif '_SAng' in pp:
                SAng[pp] = getattr(obj,pp)
            elif '_SynthDiag' in pp:
                SynthDiag[pp] = getattr(obj,pp)
            elif '_Res' in pp:
                Res[pp] = getattr(obj,pp)


    # Store Optics key parameters (for re-creating if not saved independantly)
    Optics = []
    if len(obj.Optics)>0:
        if obj.OpticsType=='Apert':
            for aa in obj.Optics:
                Optics.append({'Id':aa.Id.todict(), 'Poly':aa.Poly, 'arrayorder':aa._arrayorder, 'Clock':aa._Clock})
        elif obj.OpticsType=='Lens':
            ln = obj.Optics[0]
            Optics.append({'Id':ln.Id.todict(), 'O':ln.O, 'nIn':ln.nIn, 'Rad':ln.Rad, 'F1':ln.F1, 'F2':ln.F2, 'R1':ln.R1, 'R2':ln.R2, 'dd':ln.dd, 'Type':ln.Type, 'arrayorder':ln._arrayorder, 'Clock':ln._Clock})

    return LOSprops, Sino, Span, Cone, SAng, SynthDiag, Res, Optics



def _convert_PreData2Ldict(obj):
    Init = {'data':obj._dataRef, 't':obj._tRef, 'Chans':obj._ChansRef, 'DtRef':obj._DtRef}
    PhysNoiseParam = None if obj._PhysNoise is None else obj._PhysNoise['Param'].update(obj._NoiseModel['Par'])
    Update = {'Dt':obj.Dt, 'Resamp_t':obj._Resamp_t, 'Resamp_f':obj._Resamp_f, 'Resamp_Method':obj._Resamp_Method, 'Resamp_interpkind':obj._Resamp_interpkind,
            'indOut':obj._indOut, 'indCorr':obj._indCorr, 'interp_lt':obj._interp_lt, 'interp_lNames':obj._interp_lNames, 'Subtract_tsub':obj._Subtract_tsub,
            'FFTPar':obj._FFTPar, 'PhysNoiseParam':PhysNoiseParam}
    return Init, Update



def _save_np(obj, pathfileext, compressed=False):

    func = np.savez_compressed if compressed else np.savez
    Idsave = obj.Id.todict()

    # tofu.geom
    if obj.Id.Cls=='Ves':
        func(pathfileext, Idsave=Idsave, arrayorder=obj._arrayorder, Clock=obj._Clock, Poly=obj.Poly, Lim=obj.Lim, Sino_RefPt=obj.sino['RefPt'], Sino_NP=obj.sino['NP'])

    if obj.Id.Cls=='Struct':
        func(pathfileext, Idsave=Idsave, arrayorder=obj._arrayorder, Clock=obj._Clock, Poly=obj.Poly, Lim=obj.Lim)

    elif obj.Id.Cls=='LOS':
        func(pathfileext, Idsave=Idsave, Du=(obj.D,obj.u), Sino_RefPt=obj._sino['RefPt'])

    """
    elif obj.Id.Cls=='GLOS':
        LIdLOS = [ll.Id.todict() for ll in obj.LLOS]
        LDs, Lus = np.array([ll.D for ll in obj.LLOS]).T, np.array([ll.u for ll in obj.LLOS]).T
        func(pathfileext, Idsave=Idsave, LIdLOS=LIdLOS, LDs=LDs, Lus=Lus, Sino_RefPt=obj.Sino_RefPt, arrayorder=obj._arrayorder, Clock=obj._Clock)

    elif obj.Id.Cls=='Lens':
        func(pathfileext, Idsave=Idsave, arrayorder=obj._arrayorder, Clock=obj._Clock, O=obj.O, nIn=obj.nIn, Rad=[obj.Rad], F1=[obj.F1], F2=[obj.F2], R1=[obj.R1], R2=[obj.R2], dd=[obj.dd])

    elif obj.Id.Cls=='Apert':
        func(pathfileext, Idsave=Idsave, arrayorder=obj._arrayorder, Clock=obj._Clock, Poly=obj.Poly)

    elif obj.Id.Cls=='Detect':
        LOSprops, Sino, Span, Cone, SAng, SynthDiag, Res, Optics = _convert_Detect2Ldict(obj)
        VesCalc = {'SavePath':None} if (not hasattr(obj,'_VesCalc') or obj._VesCalc is None) else {'SavePath':obj._VesCalc.Id.SavePath, 'SaveName':obj._VesCalc.Id.SaveName}
        func(pathfileext, Idsave=Idsave, Poly=obj.Poly, Rad=obj.Rad, BaryS=obj.BaryS, nIn=obj.nIn, arrayorder=obj._arrayorder, Clock=obj._Clock, Sino_RefPt=obj.Sino_RefPt, LOSNP=[obj._LOS_NP],
                LOSprops=[LOSprops], Sino=[Sino], Span=[Span], Cone=[Cone], SAng=[SAng], SynthDiag=[SynthDiag], Res=[Res], Optics=[Optics], VesCalc=[VesCalc])

    elif obj.Id.Cls=='GDetect':
        LDetsave, LDetSynthRes = [], []
        for ii in range(0,obj.nDetect):
            ddIdsave = obj.LDetect[ii].Id.todict()
            LOSprops, Sino, Span, Cone, SAng, SynthDiag, Res, Optics = _convert_Detect2Ldict(obj.LDetect[ii])
            VesCalc = {'SavePath':None} if (not hasattr(obj.LDetect[ii],'_VesCalc') or obj.LDetect[ii]._VesCalc is None) else {'SavePath':obj.LDetect[ii]._VesCalc.Id.SavePath, 'SaveName':obj.LDetect[ii]._VesCalc.Id.SaveName}
            dd = dict(Idsave=ddIdsave, Poly=obj.LDetect[ii].Poly, Rad=obj.LDetect[ii].Rad, BaryS=obj.LDetect[ii].BaryS, nIn=obj.LDetect[ii].nIn, arrayorder=obj._arrayorder, Clock=obj._Clock, Sino_RefPt=obj.Sino_RefPt,
                      LOSNP=[obj.LDetect[ii]._LOS_NP], LOSprops=[LOSprops], Sino=[Sino], Span=[Span], Cone=[Cone], SAng=[SAng], Optics=[Optics], VesCalc=[VesCalc])
            LDetsave.append(dd)
            LDetSynthRes.append({'SynthDiag':[SynthDiag],'Res':[Res]})
        Res, lAttr = {}, dir(obj)
        for pp in lAttr:
            if not inspect.ismethod(getattr(obj,pp)) and '_Res' in pp:
                Res[pp] = getattr(obj,pp)
        func(pathfileext, Idsave=Idsave, arrayorder=obj._arrayorder, Clock=obj._Clock, Sino_RefPt=obj.Sino_RefPt, LOSRef=obj._LOSRef, Res=[Res], LDetsave=LDetsave, LDetSynthRes=LDetSynthRes)

    # tofu.Eq
    elif obj.Id.Cls=='Eq2D':
        np.savez(pathfileext, Idsave=Idsave, **obj._Tab)

    # tofu.mesh
    elif obj.Id.Cls=='Mesh1D':
        func(pathfileext, Idsave=Idsave, Knots=obj.Knots)

    elif obj.Id.Cls=='Mesh2D':
        SubMinds = [{'Name':kk, 'ind':obj._SubMesh[kk]['ind']} for kk in obj._SubMesh.keys()]
        func(pathfileext, Idsave=Idsave, Knots=[obj.MeshX1.Knots,obj.MeshX2.Knots], SubMinds=SubMinds, IndBg=obj._get_CentBckg()[1])

    elif obj.Id.Cls=='BF2D':
        Id = np.array(['BF2D',obj.Id.Name,obj.Id.SaveName,obj.Id.SavePath,obj.Id._dtFormat,obj.Id._Diag,str(obj.Id._shot), [obj.Id.Type], obj.Id.Exp],dtype=str)
        IdMesh = np.array(['Mesh2D',obj.Mesh.Id.Name,obj.Mesh.Id.SaveName,obj.Mesh.Id.SavePath,obj.Mesh.Id._dtFormat],dtype=str)
        dtime, dtimeMesh = np.array([obj.Id._dtime],dtype=object), np.array([obj.Mesh.Id._dtime],dtype=object)
        USR = np.asarray(obj.Id.USRdict)
        func(pathfileext, Id=Id, IdMesh=IdMesh, dtime=dtime, IdUSR=USR, dtimeMesh=dtimeMesh, KnotsR=obj.Mesh.MeshR.Knots, KnotsZ=obj.Mesh.MeshZ.Knots, Deg=np.array([obj.Deg],dtype=int), Ind=obj.Mesh._get_CentBckg()[1])

    # tofu.matcomp
    elif obj.Id.Cls=='GMat2D':
        Id = np.array(['GMat2D',obj.Id.Name,obj.Id.SaveName,obj.Id.SavePath,obj.Id._dtFormat,obj.Id._Diag,str(obj.Id._shot), [obj.Id.Type], obj.Id.Exp],dtype=str)
        dtime = np.array([obj.Id._dtime],dtype=object)
        USR = np.asarray(obj.Id.USRdict)
        IdObj, IdObjUSR = save_np_IdObj(obj.Id)
        CompParamVal = np.array([obj._Mat_epsrel, obj._Mat_SubP, obj._Mat_SubTheta, obj._indMat_SubP, obj._MatLOS_epsrel, obj._MatLOS_SubP, int(obj._Mat_Fast)])
        CompParamStr = np.array([obj._Mat_Mode, obj._Mat_SubMode, obj._Mat_SubThetaMode, obj._MatLOS_Mode, obj._MatLOS_SubMode])
        func(pathfileext, Id=Id, dtime=dtime, IdUSR=USR, Ves=IdObj[2], VesUSR=IdObjUSR[2], LDetect=IdObj[1], BF2=IdObj[0], BF2USR=IdObjUSR[0], LDetectUSR=IdObjUSR[1], CompParamVal=CompParamVal,
                CompParamStr=CompParamStr, indMat=obj._indMat, Matdata=obj._Mat_csr.data, Matind=obj._Mat_csr.indices, Matindpr=obj._Mat_csr.indptr, Matshape=obj._Mat_csr.shape,
                MatLOSdata=obj._MatLOS_csr.data, MatLOSind=obj._MatLOS_csr.indices, MatLOSindpr=obj._MatLOS_csr.indptr, MatLOSshape=obj._MatLOS_csr.shape,
                BF2Par=np.array([obj._BF2_Deg,obj._BF2_NFunc,obj._BF2_NCents]), LD_nD=obj._LD_nDetect)

    # tofu.treat
    elif obj.Id.Cls=='PreData':
        Init, Update = _convert_PreData2Ldict(obj)
        func(pathfileext, Idsave=Idsave, Init=[Init], Update=[Update])

        #Id = np.array(['PreData',obj.Id.Name,obj.Id.SaveName,obj.Id.SavePath,obj.Id._dtFormat,obj.Id._Diag,str(obj.Id._shot), [obj.Id.Type], obj.Id.Exp],dtype=str)
        #dtime = np.array([obj.Id._dtime],dtype=object)
        #USR = np.asarray(obj.Id.USRdict)
        #IdObj, IdObjUSR = save_np_IdObj(obj.Id)
        #StrPar = np.asarray([obj._Exp, obj._interpkind])
        #func(pathfileext, Id=Id, dtime=dtime, IdUSR=USR, LDetect=IdObj[0], LDetectUSR=IdObjUSR[0],
        #        DLPar=obj._DLPar, shot=obj._shot, StrPar=StrPar, Dt=obj._Dt, DtMarg=obj._DtMargin, MovMeanfreq=obj._MovMeanfreq, Resamp=obj._Resamp,
        #        indOut=obj._indOut, indCorr=obj._indCorr, PhysNoise=obj._PhysNoise, NoiseMod=obj._NoiseModel, interp_lt=obj._interp_lt, interp_lN=obj._interp_lNames)

    # tofu.inv
    elif obj.Id.Cls=='Sol2D':
        Id = np.array(['Sol2D',obj.Id.Name,obj.Id.SaveName,obj.Id.SavePath,obj.Id._dtFormat,obj.Id._Diag,str(obj.Id._shot), [obj.Id.Type], obj.Id.Exp],dtype=str)
        dtime = np.array([obj.Id._dtime],dtype=object)
        USR = np.asarray(obj.Id.USRdict)
        IdObj, IdObjUSR = save_np_IdObj(obj.Id)
        try:
            timing = obj._timing
        except Exception:
            timing = obj._t2
        func(pathfileext, Id=Id, dtime=dtime, IdUSR=USR, PreData=IdObj[2], PreDataUSR=IdObjUSR[2], GMat2D=IdObj[1], GMatUSR=IdObjUSR[1], BF2D=IdObj[0], BF2DUSR=IdObjUSR[0],
                InvParam=obj.InvParam, shot=obj.shot, LNames=obj._LNames, Run=obj._run,
                LOS=obj._LOS, data=obj._data, t=obj._t, Coefs=obj._Coefs, sigma=obj._sigma, Mu=obj._Mu, Chi2N=obj._Chi2N, R = obj._R, Nit=obj._Nit, Spec=obj._Spec, t2=timing, PostTreat=obj._PostTreat)
    """

def save_np_IdObj(Id):
    """ (to do) """
    LObj, LObjUSR = [], []
    Keys = sorted(Id.LObj.keys())
    for ii in range(0,len(Keys)):
        kk = sorted(Id.LObj[Keys[ii]].keys())
        Larr, LarrUSR = [], []
        for jj in range(0,len(kk)):
            if kk[jj]=='USRdict':
                LarrUSR.append(np.asarray([Id.LObj[Keys[ii]][kk[jj]]],dtype=object))
            else:
                Larr.append(np.asarray([Id.LObj[Keys[ii]][kk[jj]]],dtype=str))
        LObj.append( np.concatenate(tuple(Larr),axis=0) )
        LObjUSR.append( np.concatenate(tuple(LarrUSR),axis=0) )
    return LObj, LObjUSR



###########################
#   Opening
###########################


def Open(pathfileext=None, shot=None, t=None, Dt=None, Mesh=None, Deg=None, Deriv=None, Sep=True, Pos=True, OutPath=None, ReplacePath=None, Ves=None, out='full', Verb=False, Print=True):
    """ Open a ToFu object saved file

    This generic open function identifies the required loading routine by detecting how the object was saved from the file name extension.
    Also, it uses :meth:`~tofu.pathfile.FindSolFile()` to identify the relevant file in case key criteria such as shot, Deg... are provided instead of the file name itself.
    Finally, once all the relevant data is loaded from the file, a ToFu object is re-created, if necessary by implicitly loading all other objects it may depend on (i.e.: vessel, apertures...)

    If pathfileext is not provided (None), then the following keyword arguments are fed to :meth:`~tofu.pathfile.FindSolFile()`: shot, t, Dt, Mesh, Deg, Deriv, Sep, Pos

    Parameters
    ----------
    pathfileext :   None / str
        If provided, the name of the file to load
    OutPath :       None / str
        If provided, the absolute path where the file is to be found
    ReplacePath :   str
        If provided, ? (to finish)
    Ves :           None /
        If provided, the :class:`tofu.geom.Ves` object that shall be used to reconstruct the object (if not provided, the appropriate vessel will be loaded).
    out :           str
        Flag indicating whether the object should be loaded completely ('full'), in a light dismissing the heaviest attributes ('light') or whether only the Id or a list of Id should be returned ('Id'), valid only for '.npz'
    Verb :          bool
        Flag indicating whether to pring intermediate comments on the loading procedure

    Returns
    -------
    obj             ToFu object
        The loaded and re-created ToFu object

    """
    assert None in [pathfileext,shot] and not (pathfileext is None and shot is None), "Arg pathfileext or shot must be None, but not both !"
    if pathfileext is None:
        File = FindSolFile(shot=shot, t=t, Dt=Dt, Mesh=Mesh, Deg=Deg, Deriv=Deriv, Sep=Sep, Pos=Pos, OutPath=OutPath)
        if File is None:
            return File
        pathfileext = OutPath+File
    assert '.npz' in pathfileext or '.pck' in pathfileext, "Arg pathfileext must contain '.npz' or .pck !"

    if '.npz' in pathfileext:
        obj = _open_np(pathfileext, Ves=Ves, ReplacePath=ReplacePath, out=out, Verb=Verb, Print=Print)
    else:
        obj = _open_object(pathfileext)
    if Print:
        print("Loaded :  "+pathfileext)
    return obj



def _open_object(pathfileext):
    with Open(pathfileext, 'rb') as input:
        obj = pck.load(input)
    if 'TFM' in pathfileext:
        obj.set_BasisFunc(obj.Deg)
    return obj


def open_np_IdObj(LCls=None,LIdArr=None,LIdUSR=None):
    LIdObj = []
    if not LIdArr is None:
        assert type(LIdArr) is list and type(LCls) is list, "Args LCls and LIdArr must be lists !"
        NObj = len(LIdArr)
        for ii in range(0,NObj):
            no = LIdArr[ii].shape[1]
            for jj in range(0,no):
                if not LIdUSR is None and not LIdUSR[ii][0][jj] is None:
                    LIdObj.append(ID(LCls[ii],str(LIdArr[ii][1,jj]),SaveName=str(LIdArr[ii][2,jj]), SavePath=str(LIdArr[ii][3,jj]), Exp=str(LIdArr[ii][0,jj]), dtime=dtm.datetime.strptime(str(LIdArr[ii][5,jj]),str(LIdArr[ii][4,jj])), dtFormat=str(LIdArr[ii][4,jj]), USRdict=LIdUSR[ii][0][jj]))
                else:
                    LIdObj.append(ID(LCls[ii],str(LIdArr[ii][1,jj]),SaveName=str(LIdArr[ii][2,jj]), SavePath=str(LIdArr[ii][3,jj]), Exp=str(LIdArr[ii][0,jj]), dtime=dtm.datetime.strptime(str(LIdArr[ii][5,jj]),str(LIdArr[ii][4,jj])), dtFormat=str(LIdArr[ii][4,jj])))
    return LIdObj


def _tryloadVesStruct(Id, VesStruct=None, Print=True):
    if VesStruct is not None and hasattr(VesStruct,'__iter__') and VesStruct[0].Id.Cls=='Ves':
        return VesStruct[0], VesStruct[1]
    else:
        Ves, LStruct = None, None
        if 'Ves' in Id.LObj.keys():
            PathFileExt = os.path.join(Id.LObj['Ves']['SavePath'][0],Id.LObj['Ves']['SaveName'][0]+'.npz')
            try:
                Ves = Open(PathFileExt, Print=Print)
            except:
                warnings.warn(Id.Name +" : associated Ves/Struct object(s) could not be loaded from "+PathFileExt)
        if 'Struct' in Id.LObj.keys():
            LStruct = []
            for ii in range(0,len(Id.LObj['Struct']['Name'])):
                PathFileExt = os.path.join(Id.LObj['Struct']['SavePath'][ii],Id.LObj['Struct']['SaveName'][ii]+'.npz')
                try:
                    LStruct.append(Open(PathFileExt, Print=Print))
                except:
                    warnings.warn(Id.Name +" : associated Ves/Struct object(s) could not be loaded from "+PathFileExt)
        return Ves, LStruct

def _tryLoadOpticsElseCreate(Id, Opt=None, Ves=None, Verb=False):
    import tofu.geom as TFG
    if 'Apert' in Id.LObj.keys():
        Optics = []
        for ii in range(0,len(Id.LObj['Apert']['SaveName'])):
            try:
                PathFileExt = Id.LObj['Apert']['SavePath'][ii]+Id.LObj['Apert']['SaveName'][ii]+'.npz'
                aa = Open(PathFileExt, Ves=Ves)
                Optics.append(aa)
            except Exception:
                if not Opt is None:
                    assert type(Ves) is TFG.Ves, "Arg Ves must be a TFG.Ves instance !"
                    if Verb:
                        print(Id.Name +" : no saved Apert => creating the associated Apert object !")
                    ind = [jj for jj in range(0,len(Opt)) if Opt[jj]['Id'][0]['SaveName']==Id.LObj['Apert']['SaveName'][ii] and Opt[jj]['Id'][0]['SavePath']==Id.LObj['Apert']['SavePath'][ii]]
                    assert len(ind)==1, "Several possible solutions !"
                    ind = ind[0]
                    iid = _Id_recreateFromdict(Opt[ind]['Id'])
                    aa = TFG.Apert(iid, Opt[ind]['Poly'], Ves=Ves, arrayorder=Opt[ind]['arrayorder'], Clock=Opt[ind]['Clock'])
                    Optics.append(aa)
                else:
                    warnings.warn(Id.Name +" : associated Apert object could not be loaded from "+PathFileExt)
    elif 'Lens' in Id.LObj.keys():
        try:
            PathFileExt = Id.LObj['Lens']['SavePath'][0]+Id.LObj['Lens']['SaveName'][0]+'.npz'
            Optics = Open(PathFileExt, Ves=Ves)
        except Exception:
            if not Opt is None:
                assert type(Ves) is TFG.Ves, "Arg Ves must be a TFG.Ves instance !"
                if Verb:
                    print(Id.Name +" : no saved Lens => creating the associated Lens object !")
                iid = _Id_recreateFromdict(Opt[0]['Id'])
                aa = TFG.Lens(iid, Opt[0]['O'], Opt[0]['nIn'], Opt[0]['Rad'], Opt[0]['F1'], F2=Opt[0]['F2'], R1=Opt[0]['R1'], R2=Opt[0]['R2'], dd=Opt[0]['dd'], Type=Opt[0]['Type'], Ves=Ves,
                              arrayorder=Opt[0]['arrayorder'], Clock=Opt[0]['Clock'])
                Optics = aa
            else:
                warnings.warn(Id.Name +" : associated Lens object could not be loaded from "+PathFileExt)
    return Optics



def _resetDetectAttr(obj, Out):
    import tofu.geom as TFG
    # Re-creating LOS
    LOS = {}
    kkeys = Out['LOSprops'].keys()
    for ii in range(0,len(Out['LOSprops']['Keys'])):
        idlos = _Id_recreateFromdict(Out['LOSprops']['Id'][ii])
        los = TFG.LOS(idlos, Out['LOSprops']['Du'][ii], Ves=obj.Ves, Sino_RefPt=obj.Sino_RefPt)
        LOS[Out['LOSprops']['Keys'][ii]] = {'LOS':los}
        for jj in range(0,len(kkeys)):
            if not kkeys[jj] in ['Keys','Id','Du']:
                LOS[Out['LOSprops']['Keys'][ii]][kkeys[jj]] = Out['LOSprops'][kkeys[jj]][ii]
    obj._LOS = LOS

    # Re-assigning tabulated data
    fields = ['Sino', 'Span', 'Cone', 'SAng', 'SynthDiag', 'Res']
    for ff in fields:
        for kk in Out[ff].keys():
            setattr(obj,kk,Out[ff][kk])
    return obj




def _get_light_SynthDiag_Res():
    SynthDiag = {'_SynthDiag_Done':False, '_SynthDiag_ds':None, '_SynthDiag_dsMode':None, '_SynthDiag_MarginS':None, '_SynthDiag_dX12':None, '_SynthDiag_dX12Mode':None, '_SynthDiag_Colis':None,
                 '_SynthDiag_Points':None, '_SynthDiag_SAng':None, '_SynthDiag_Vect':None, '_SynthDiag_dV':None}
    Res = {'_Res_Mode':None, '_Res_Amp':None, '_Res_Deg':None,
           '_Res_Pts':None, '_Res_Res':None, '_Res_CrossMesh':None, '_Res_CrossMeshMode':None,
           '_Res_steps':None, '_Res_Thres':None, '_Res_ThresMode':None, '_Res_ThresMin':None,
           '_Res_IntResCross':None, '_Res_IntResCrossMode':None, '_Res_IntResLong':None, '_Res_IntResLongMode':None, '_Res_IntNtt':None,
           '_Res_EqName': None,
           '_Res_Done': False}

    return SynthDiag, Res




def _open_np(pathfileext, Ves=None, ReplacePath=None, out='full', Verb=False, Print=True):

    if 'TFG' in pathfileext:
        import tofu.geom as tfg
    #elif 'TFEq' in pathfileext:
    #    import tofu.Eq as tfEq
    #elif 'TFM' in pathfileext:
    #    import tofu.mesh as TFM
    #elif 'TFMC' in pathfileext:
    #    import tofu.matcomp as TFMC
    #elif 'TFT' in pathfileext:
    #    import tofu.treat as tft
    #elif 'TFI' in pathfileext:
    #    import tofu.inv as TFI

    try:
        Out = np.load(pathfileext,mmap_mode=None)
    except UnicodeError:
        Out = np.load(pathfileext,mmap_mode=None, encoding='latin1')
    Id = _Id_recreateFromdict(Out['Idsave'])
    if out=='Id':
        return Id

    if Id.Cls == 'Ves':
        Lim = None if Out['Lim'].tolist() is None else Out['Lim']
        obj = tfg.Ves(Id, Out['Poly'], Type=Id.Type, Exp=Id.Exp, Lim=Lim, Clock=bool(Out['Clock']), arrayorder=str(Out['arrayorder']), Sino_RefPt=Out['Sino_RefPt'], Sino_NP=int(Out['Sino_NP']), SavePath=Id.SavePath)

    elif Id.Cls == 'Struct':
        Lim = None if Out['Lim'].tolist() is None else Out['Lim']
        obj = tfg.Struct(Id, Out['Poly'], Type=Id.Type, Exp=Id.Exp, Lim=Lim, Clock=bool(Out['Clock']), arrayorder=str(Out['arrayorder']), SavePath=Id.SavePath)

    elif Id.Cls == 'LOS':
        Ves, LStruct = _tryloadVesStruct(Id, Print=Print)
        assert not (Ves is None and 'Ves' in Id.LObj.keys() and len(Id.LObj['Ves']['Name'])>0), "Ves could not be loaded !"
        assert not (LStruct is None and 'Struct' in Id.LObj.keys() and len(Id.LObj['Struct']['Name'])>0), "Struct could not be loaded !"
        obj = tfg.LOS(Id, tuple(Out['Du']), Ves=Ves, LStruct=LStruct, Sino_RefPt=Out['Sino_RefPt'], Exp=Id.Exp, Diag=Id.Diag, shot=Id.shot, SavePath=Id.SavePath)

    """
    elif Id.Cls == 'GLOS':
        Ves = _tryloadVes(Id)
        LLOS, IdLOS = [], Id.LObj['LOS']
        for ii in range(0,len(IdLOS['Name'])):
            Idl = _Id_recreateFromdict(Out['LIdLOS'][ii])
            ll = TFG.LOS(Idl, Du=(Out['LDs'][:,ii],Out['Lus'][:,ii]), Ves=Ves, Sino_RefPt=Out['Sino_RefPt'], arrayorder=str(Out['arrayorder']))
            LLOS.append(ll)
        obj = TFG.GLOS(Id, LLOS, Ves=Ves, Type=Id.Type, Exp=Id.Exp, Diag=Id.Diag, shot=Id.shot, Sino_RefPt=Out['Sino_RefPt'], SavePath=Id.SavePath, arrayorder=str(Out['arrayorder']), Clock=bool(Out['Clock']),
                       dtime=Id.dtime)

    elif Id.Cls == 'Lens':
        Ves = _tryloadVes(Id, Ves=Ves)
        obj = TFG.Lens(Id, Out['O'], Out['nIn'], Out['Rad'][0], Out['F1'][0], F2=Out['F2'][0], Type=Id.Type, R1=Out['R1'][0], R2=Out['R2'][0], dd=Out['dd'][0], Ves=Ves,
                Exp=Id.Exp, Clock=bool(Out['Clock']), Diag=Id.Diag, shot=Id.shot, arrayorder=str(Out['arrayorder']), SavePath=Id.SavePath, dtime=Id.dtime)

    elif Id.Cls == 'Apert':
        Ves = _tryloadVes(Id, Ves=Ves)
        obj = TFG.Apert(Id, Out['Poly'], Clock=bool(Out['Clock']), arrayorder=str(Out['arrayorder']), Ves=Ves, Exp=Id.Exp, Diag=Id.Diag, shot=Id.shot, dtime=Id.dtime)

    elif Id.Cls == 'Detect':
        Ves = _tryloadVes(Id, Ves=Ves)
        if 'VesCalc'in Out.keys() and Out['VesCalc'][0]['SavePath'] is not None:
            VesCalc = Open(Out['VesCalc'][0]['SavePath']+Out['VesCalc'][0]['SaveName']+'.npz')
        else:
            VesCalc = None
        LOSprops, Sino, Span, Cone, SAng, Opt = Out['LOSprops'][0], Out['Sino'][0], Out['Span'][0], Out['Cone'][0], Out['SAng'][0], Out['Optics'][0]
        (SynthDiag,Res) = (Out['SynthDiag'][0],Out['Res'][0]) if out=='full' else _get_light_SynthDiag_Res()
        Optics = _tryLoadOpticsElseCreate(Id, Opt=Opt, Ves=Ves, Verb=Verb)

        Poly = Out['Poly'] if type(Optics) is list else dict(Rad=float(Out['Rad']),O=Out['BaryS'],nIn=Out['nIn'])
        obj = TFG.Detect(Id, Poly, Optics=Optics, Ves=Ves, VesCalc=VesCalc, Sino_RefPt=Sino['_Sino_RefPt'], CalcEtend=False, CalcSpanImp=False, CalcCone=False, CalcPreComp=False, Calc=True, Verb=Verb,
                         arrayorder=str(Out['arrayorder']), Clock=bool(Out['Clock']))
        obj = _resetDetectAttr(obj, {'LOSprops':LOSprops, 'Sino':Sino, 'Span':Span, 'Cone':Cone, 'SAng':SAng, 'SynthDiag':SynthDiag, 'Res':Res, 'Optics':Opt})
        obj._LOS_NP = Out['LOSNP']
        if obj._SynthDiag_Done and obj._SynthDiag_Points is None:
            obj.set_SigPrecomp()

    elif Id.Cls == 'GDetect':
        LDetsave = list(Out['LDetsave'])
        LDet = []
        Ves = _tryloadVes(Id, Ves=Ves)
        if out=='light':
            SynthDiag, Res = _get_light_SynthDiag_Res()
        else:
            LDetSynthRes = Out['LDetSynthRes']
        for ii in range(0,len(LDetsave)):
            ddIdsave = _Id_recreateFromdict(LDetsave[ii]['Idsave'])
            if 'VesCalc'in LDetsave[ii].keys() and LDetsave[ii]['VesCalc'][0]['SavePath'] is not None:
                VesCalc = Open(LDetsave[ii]['VesCalc'][0]['SavePath']+LDetsave[ii]['VesCalc'][0]['SaveName']+'.npz')
            else:
                VesCalc = None
            LOSprops, Sino, Span, Cone, SAng, Opt = LDetsave[ii]['LOSprops'][0], LDetsave[ii]['Sino'][0], LDetsave[ii]['Span'][0], LDetsave[ii]['Cone'][0], LDetsave[ii]['SAng'][0], LDetsave[ii]['Optics'][0]
            if out=='full':
                SynthDiag, Res = LDetSynthRes[ii]['SynthDiag'][0], LDetSynthRes[ii]['Res'][0]
            Optics = _tryLoadOpticsElseCreate(ddIdsave, Opt=Opt, Ves=Ves, Verb=Verb)
            Poly = LDetsave[ii]['Poly'] if type(Optics) is list else dict(Rad=float(LDetsave[ii]['Rad']),O=LDetsave[ii]['BaryS'],nIn=LDetsave[ii]['nIn'])
            Sino_RefPt = None if Out['Sino_RefPt'].shape==() else Out['Sino_RefPt']
            dd = TFG.Detect(ddIdsave, Poly, Optics=Optics, Ves=Ves, VesCalc=VesCalc, Sino_RefPt=Sino_RefPt, CalcEtend=False, CalcSpanImp=False, CalcCone=False, CalcPreComp=False, Calc=True, Verb=Verb,
                            arrayorder=str(Out['arrayorder']), Clock=bool(Out['Clock']))
            dd = _resetDetectAttr(dd, {'LOSprops':LOSprops, 'Sino':Sino, 'Span':Span, 'Cone':Cone, 'SAng':SAng, 'SynthDiag':SynthDiag, 'Res':Res, 'Optics':Opt})
            dd._LOS_NP = LDetsave[ii]['LOSNP']
            if dd._SynthDiag_Done and dd._SynthDiag_Points is None:
                dd.set_SigPrecomp()
            LDet.append(dd)
        obj = TFG.GDetect(Id, LDet, Type=Id.Type, Exp=Id.Exp, Diag=Id.Diag, shot=Id.shot, dtime=Id.dtime, Sino_RefPt=Out['Sino_RefPt'], LOSRef=str(Out['LOSRef']),
                          arrayorder=str(Out['arrayorder']), Clock=bool(Out['Clock']), SavePath=Id.SavePath)
        Res = Out['Res'][0] if out=='full' else Res
        for kk in Res.keys():
            setattr(obj,kk,Res[kk])

    elif Id.Cls=='Eq2D':
        Sep = [np.array(ss) for ss in Out['Sep'].tolist()]
        obj = tfEq.Eq2D(Id, Out['PtsCross'], t=Out['t'], MagAx=Out['MagAx'], Sep=Sep, rho_p=Out['rho_p'].tolist(), rho_t=Out['rho_t'].tolist(), surf=Out['surf'].tolist(), vol=Out['vol'].tolist(),
                        q=Out['q'].tolist(), jp=Out['jp'].tolist(), pf=Out['pf'].tolist(), tf=Out['tf'].tolist(), theta=Out['theta'].tolist(), thetastar=Out['thetastar'].tolist(),
                        BTX=Out['BTX'].tolist(), BRY=Out['BRY'].tolist(), BZ=Out['BZ'].tolist(), Ref=str(Out['Ref']))

    elif Id.Cls=='Mesh1D':
        obj = TFM.Mesh1D(Id, Out['Knots'])

    elif Id.Cls=='Mesh2D':
        obj = TFM.Mesh2D(Id, [Out['Knots'][0],Out['Knots'][1]])
        obj = TFM.Mesh2D(Id, Knots=obj, ind=Out['IndBg'])
        for ii in range(0,len(Out['SubMinds'])):
            obj.add_SubMesh(Name=Out['SubMinds'][ii]['Name'], ind=Out['SubMinds'][ii]['ind'])

    elif Id.Cls=='Metric1D':
        obj = TFM.Metric1D(Id)

    elif Id.Cls=='Metric2D':
        obj = TFM.Metric2D(Id)


    elif Id.Cls in 'BF2D':
        IdMesh = ID(str(Out['IdMesh'][0]), str(Out['IdMesh'][1]), SaveName=str(Out['IdMesh'][2]), SavePath=str(Out['IdMesh'][3]), dtime=Out['dtimeMesh'][0], dtFormat=str(Out['IdMesh'][4]))
        M2 = TFM.Mesh2D(IdMesh, Knots=[Out['KnotsR'],Out['KnotsZ']])
        M2bis = TFM.Mesh2D(IdMesh,Knots=M2,Ind=Out['Ind'])
        obj = TFM.BF2D(Id, M2bis, int(Out['Deg'][0]))
    elif Id.Cls=='GMat2D':
        import ToFu_MatComp as TFMC
        import scipy.sparse as scpsp
        Id.set_LObj(open_np_IdObj(['Ves','BF2D','Detect'], [Out['Ves'],Out['BF2'],Out['LDetect']], [Out['VesUSR'],Out['BF2USR'],Out['LDetectUSR']]))
        Mat = scpsp.csr_matrix((Out['Matdata'], Out['Matind'], Out['Matindpr']), shape=Out['Matshape'])
        MatLOS = scpsp.csr_matrix((Out['MatLOSdata'], Out['MatLOSind'], Out['MatLOSindpr']), shape=Out['MatLOSshape'])
        obj = TFMC.GMat2D(Id, None, None, Mat=None, indMat=None, MatLOS=None, Calcind=False, Calc=False, CalcLOS=False)
        obj._init_CompParam(Mode=str(Out['CompParamStr'][0]), epsrel=Out['CompParamVal'][0], SubP=Out['CompParamVal'][1], SubMode=str(Out['CompParamStr'][1]), SubTheta=Out['CompParamVal'][2], SubThetaMode=str(Out['CompParamStr'][2]), Fast=bool(Out['CompParamVal'][-1]), SubPind=Out['CompParamVal'][3], ModeLOS=str(Out['CompParamStr'][3]), epsrelLOS=Out['CompParamVal'][4], SubPLOS=Out['CompParamVal'][5], SubModeLOS=str(Out['CompParamStr'][4]))
        obj._BF2 = None
        obj._BF2_Deg = int(Out['BF2Par'][0])
        obj._BF2_NCents = int(Out['BF2Par'][2])
        obj._BF2_NFunc = int(Out['BF2Par'][1])
        obj._Ves = None
        obj._LD = None
        obj._LD_nDetect = int(Out['LD_nD'])
        obj._set_indMat(indMat=Out['indMat'], Verb=False)
        obj._set_MatLOS(MatLOS=MatLOS, Verb=False)
        obj._set_Mat(Mat=Mat, Verb=False)



    elif Id.Cls=='PreData':
        LIdDet = Id.get_LObjasLId('Detect') if 'Detect' in Id.LObj.keys() else None
        Init, Update = Out['Init'][0], Out['Update'][0]
        obj = tft.PreData(Init['data'], Id=Id, t=Init['t'], Chans=Init['Chans'], DtRef=Init['DtRef'], LIdDet=LIdDet)
        obj.set_Dt(Update['Dt'], Calc=False)
        obj.set_Resamp(t=Update['Resamp_t'], f=Update['Resamp_f'], Method=Update['Resamp_Method'], interpkind=Update['Resamp_interpkind'], Calc=False)
        obj.Out_add(indOut=Update['indOut'], Calc=False)
        obj.Corr_add(indCorr=Update['indCorr'], Calc=False)
        obj.interp(lt=Update['interp_lt'], lNames=Update['interp_lNames'], Calc=False)
        obj.substract_Dt(tsub=Update['Subtract_tsub'], Calc=False)
        obj.set_fft(Calc=True, **Update['FFTPar'])
        if not Update['PhysNoiseParam'] is None:
            Method = 'svd' if 'Modes' in Update['PhysNoiseParam'].keys() else 'fft'
            obj.set_PhysNoise(**Update['PhysNoiseParam'].update({'Method':Method}))


        #Id.set_LObj(open_np_IdObj(['Detect'],[Out['LDetect']], [Out['LDetectUSR']]))
        #obj = TFT.PreData(Id=Id, shot=int(Out['shot']), DLPar=Out['DLPar'].item(), Exp=str(Out['StrPar'][0]), Dt=list(Out['Dt']), DtMargin=float(Out['DtMarg']), MovMeanfreq=float(Out['MovMeanfreq']), Resamp=bool(Out['Resamp']),
        #        interpkind=str(Out['StrPar'][1]), indOut=Out['indOut'], indCorr=Out['indCorr'], lt=Out['interp_lt'], lNames=Out['interp_lN'].tolist(), Test=True)
        #if not Out['PhysNoise'].item() is None:
        #    obj.set_PhysNoise(Deg=int(Out['NoiseMod'].item()['Deg']), Nbin=int(Out['NoiseMod'].item()['Nbin']), LimRatio=float(Out['NoiseMod'].item()['LimRatio']), **Out['PhysNoise'].item()['Param'])


    elif Id.Cls=='Sol2D':
        Id.set_LObj(open_np_IdObj(['PreData','GMat2D','BF2D'],[Out['PreData'], Out['GMat2D'], Out['BF2D']], [Out['PreDataUSR'],Out['GMatUSR'],Out['BF2DUSR']]))
        GMSaveName = Id.LObj['GMat2D']['SaveName'][0]
        try:
            GMat = Open(Id.LObj['GMat2D']['SavePath'][0]+GMSaveName+'.npz')
        except Exception:
            GMSaveName = GMSaveName[:GMSaveName.index('All_')+4]+'sh'+GMSaveName[GMSaveName.index('All_')+4:]
            GMat = Open(Id.LObj['GMat2D']['SavePath'][0]+GMSaveName+'.npz')
        obj = TFI.Sol2D(Id, PreData=None, GMat=GMat, InvParam=Out['InvParam'].item(), SVesePreData=False, SVeseGMat=True, SVeseBF=True)
        obj._PreData = None
        obj._GMat = obj.GMat.get_SubGMat2D(Val=list(Out['LNames']), Crit='Name',InOut='In')
        obj._shot = int(Out['shot'])
        try:
            obj._LNames = Out['LNames'].tolist()
        except Exception:
            obj._LNames = obj.PreData.In_list()
        obj._run = bool(Out['Run'])
        if bool(Out['Run']):
            obj._LOS = bool(Out['LOS'])
            obj._t, obj._data = Out['t'], Out['data']
            obj._Coefs, obj._sigma = Out['Coefs'], Out['sigma']
            obj._Mu, obj._Chi2N, obj._R, obj._Nit = Out['Mu'], Out['Chi2N'], Out['R'], Out['Nit']
            obj._Spec = list(Out['Spec'])
            obj._timing = Out['t2']
            obj._PostTreat = list(Out['PostTreat'])
    """
    return obj




