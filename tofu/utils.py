
# Built-in
import os
import sys
import collections
from abc import ABCMeta, abstractmethod
import getpass
import subprocess
import itertools as itt
import warnings

# Common
import numpy as np
import matplotlib as mpl
from matplotlib.tri import Triangulation as mplTri
import matplotlib.pyplot as plt


# tofu-specific
from tofu import __version__
import tofu.pathfile as tfpf

_sep = '_'
_dict_lexcept_key = []
_pyv = int(sys.version[0])

###############################################
#           File searching
###############################################

def FileNotFoundMsg(pattern,path,lF, nocc=1, ntab=0):
    assert type(pattern) in [str,list]
    assert type(path) is str
    assert type(lF) is list
    pat = pattern if type(pattern) is str else str(pattern)
    tab = "    "*ntab
    msg = ["Wrong number of matches (%i) !"%nocc]
    msg += ["    for : %s"%pat]
    msg += ["    in  : %s"%path]
    msg += ["    =>    %s"%str(lF)]
    msg = "\n".join([tab+ss for ss in msg])
    return msg


def FindFilePattern(pattern, path, nocc=1, ntab=0):
    assert type(pattern) in [str,list]
    assert type(path) is str
    pat = [pattern] if type(pattern) is str else pattern
    assert all([type(ss) is str for ss in pat])
    lF = os.listdir(path)
    lF = [ff for ff in lF if all([ss in ff for ss in pat])]
    assert len(lF)==nocc, FileNotFoundMsg(pat,path,lF, nocc, ntab=ntab)
    return lF


def get_pathfileext(path=None, name=None,
                    path_def='./', name_def='dummy', mode='npz'):
    modeok = ['npz','mat']
    modeokstr = "["+", ".join(modeok)+"]"

    if name is not None:
        C = type(name) is str and not '.' in name
        assert C, "name should not include the extension !"
    assert path is None or type(path) is str, "Arg path must be None or a str !"
    assert mode in modeok, "Arg mode must be in {0}".format(modeokstr)

    if path is None:
        path = path_def
    path = os.path.abspath(path)
    if name is None:
        name = name_def
    return path, name, mode



#############################################
#       figure size
#############################################

def get_figuresize(fs, fsdef=(12,6),
                   orient='landscape', method='xrandr'):
    """ Generic function to return figure size in inches

        Useful for str-based flags such as:
            - 'a4'  : use orient='portrait' or 'landscape'
            - 'full': to get screen size
                use method='xrandr' (recommended),
                as 'xdpyinfo' tends to be wrong
    """

    assert fs is None or type(fs) in [str,tuple]
    if fs is None:
        fs = fsdef
    elif type(fs) is str:
        if fs=='a4':
            fs = (8.27,11.69)
            if orient=='landscape':
                fs = (fs[1],fs[0])
        elif fs=='full':
            assert method in ['xrandr','xdpyinfo']
            if method=='xrandr':
                cmd0 = "xrandr"
                #cmd1 = "grep '*'"
                out = subprocess.check_output(cmd0.split())
                s = [o for o in out.decode('utf-8').split('\n') if 'mm x ' in o]
                assert len(s)==1
                s = [ss for ss in s[0].split(' ') if 'mm' in ss]
                assert len(s)==2
                fsmm = [int(ss.replace('mm','')) for ss in s]
            else:
                cmd0 = 'xdpyinfo'
                out = subprocess.check_output(cmd0.split())
                s = [o for o in out.decode('utf-8').split('\n')
                     if 'dimensions' in o]
                assert len(s)==1
                s = s[0][s[0].index('(')+1:s[0].index(' millimeters')]
                fsmm = [int(ss) for ss in s.split('x')]
            fs = (fsmm[0]/(10*2.54), fsmm[1]/(10*2.54))
    assert type(fs) is tuple and len(fs)==2
    return fs






#############################################
#       todict formatting
#############################################


def flatten_dict(d, parent_key='', sep=_sep, deep='ref',
                 lexcept_key=_dict_lexcept_key):

    items = []
    lexcept_key = [] if lexcept_key is None else lexcept_key
    for k, v in d.items():
        if k not in lexcept_key:
            if issubclass(v.__class__, ToFuObjectBase):
                if deep=='dict':
                    v = v.to_dict(deep='dict')
                elif deep=='copy':
                    v = v.copy(deep='copy')
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, collections.MutableMapping):
                items.extend(flatten_dict(v, new_key,
                                          deep=deep, sep=sep).items())
            else:
                items.append((new_key, v))
    return dict(items)

def _reshape_dict(ss, vv, dinit={}, sep=_sep):
    ls = ss.split(sep)
    k = ss if len(ls)==1 else ls[0]
    if len(ls) == 2:
        dk = {ls[1]:vv}
        if k not in dinit.keys():
            dinit[k] = {}
        assert isinstance(dinit[k],dict)
        dinit[k].update({ls[1]:vv})
    elif len(ls) > 2:
        if k not in dinit.keys():
            dinit[k] = {}
        _reshape_dict(sep.join(ls[1:]), vv, dinit=dinit[k], sep=sep)
    else:
        assert k not in dinit.keys()
        dinit[k] = vv

def reshape_dict(d, sep=_sep, lcls=[]):
    # Get all individual keys
    out = {}
    for ss, vv in d.items():
        _reshape_dict(ss, vv, dinit=out, sep=sep)
    return out


# Check if deprecated ???
def get_todictfields(ld, ls):
    C0 = type(ld) is list and type(ls) is list and len(ld)==len(ls)
    C1 = type(ld) is dict and type(ls) is dict
    assert C0 or C1, "Provide two list of dict or two dict !"
    if C1:
        ld, ls = [ld], [ls]
    nd = len(ld)
    out = {}
    for ii in range(0,nd):
        for ss in ld[ii].keys():
            ks = '{0}_{1}'.format(ls[ii], ss)
            out[ks] = ld[ii][ss]
    return out


#############################################
#       Special dict subclass for dynamic attributes creation
#############################################

class Dictattr(dict):
    __getattr__ = dict.__getitem__

    def __init__(self, extra, *args, **kwdargs):
        #super()
        super(Dictattr, self).__init__(*args, **kwdargs)
        self._extra = extra

    def __dir__(self):
        return [str(k) for k in self.keys()]+self._extra




#############################################
#       Miscellaneous
#############################################

def _set_arrayorder(obj, arrayorder='C'):
    """ Set the memory order of all np.ndarrays in a tofu object """
    msg = "Arg arrayorder must be in ['C','F']"
    assert arrayorder in ['C','F'], msg

    d = obj.to_dict(strip=-1)
    account = {'Success':[], 'Failed':[]}
    for k, v in d.items():
        if type(v) is np.array and v.ndim>1:
            try:
                if arrayorder=='C':
                    d[k] = np.ascontiguousarray(v)
                else:
                    d[k] = np.asfortranarray(v)
                account['Success'].append(k)
            except Exception as err:
                warnings.warn(str(err))
                account['Failed'].append(k)

    return d, account



#############################################
#       save / load
#############################################

def save(obj, path=None, name=None, sep=_sep, deep=False, mode='npz',
         strip=None, compressed=False, verb=True, return_pfe=False):
    """ Save the ToFu object

    ToFu provides built-in saving and loading functions for ToFu objects.
    Specifying saving path ad name is optional (defaults recommended)
    The mode refers to the file format

    Good practices are:
        - save all struct objects

    Parameters
    ----------
    obj  :      ToFuObject subclass instance
        The object to be saved
    path :      None / str
        The folder where to save, if None (recommended), uses obj.Id.SavePath
    name :      None / str
        The file name, if None (recommended), uses obj.Id.SaveName
    mode :      str
        Flag specifying the saving mode
            - 'npz': numpy file
            - 'mat': matlab file
    strip:      int
        Flag indicating how stripped the saved object should be
        See docstring of self.strip()
    deep:       bool
        Flag, used when the object has other tofu objects as attributes
        Indicated whether these attribute object should be:
            - True: converted to dict themselves in order to be saved inside
                the same file as attributes
                (-> uses self.to_dict(deep='dict'))
            - False: not converted, in that the strategy would be to save them
                separately and store only the reference to the saved files
                instead of the objects themselves.
                To do this, you must:
                    1/ Save all object attributes independently
                    2/ Store only the reference by doing self.strip(-1)
                       The strip() method will check they have been saved
                       before removing them, and throw an Exception otherwise
                    3/ self.save(deep=False)
    compressed :    bool
        Flag indicating whether to compress the file (slower, not recommended)
    verb :          bool
        Flag indicating whether to print a summary (recommended)

    """
    msg = "Arg obj must be a tofu subclass instance !"
    assert issubclass(obj.__class__, ToFuObject), msg
    msg = "Arg path must be None or a str (folder) !"
    assert path is None or isinstance(path,str), msg
    msg = "Arg name must be None or a str (file name) !"
    assert name is None or isinstance(name,str), msg
    msg = "Arg mode must be in ['npz','mat'] !"
    assert mode in ['npz','mat'], msg
    msg = "Arg compressed must be a bool !"
    assert type(compressed) is bool, msg
    msg = "Arg verb must be a bool !"
    assert type(verb) is bool, msg

    # Check path, name, mode
    path, name, mode = get_pathfileext(path=path, name=name,
                                       path_def=obj.Id.SavePath,
                                       name_def=obj.Id.SaveName, mode=mode)

    # Update self._Id fields
    obj._Id._SavePath = path
    if name!=obj.Id.SaveName:
        obj._Id.set_SaveName(name)

    # Get stripped dictionnary
    deep = 'dict' if deep else 'ref'
    dd = obj.to_dict(strip=strip, sep=sep, deep=deep)

    pathfileext = os.path.join(path,name+'.'+mode)

    if mode=='npz':
        _save_npz(dd, pathfileext, compressed=compressed)
    elif mode=='mat':
        _save_mat(dd, pathfileext, compressed=compressed)

    # print
    if verb:
        msg = "Saved in :\n"
        msg += "    "+pathfileext
        print(msg)
    if return_pfe:
        return pathfileext

def _save_npz(dd, pathfileext, compressed=False):

    func = np.savez_compressed if compressed else np.savez
    msg = "How to deal with:"
    msg += "\n SaveName : {0}".format(dd['dId_dall_SaveName'])
    msg += "\n Attributes:"
    err = False
    dt = {}
    for k in dd.keys():
        kt = k+'_type'
        dt[kt] = np.asarray([type(dd[k]).__name__])
        if dd[k] is None:
            dd[k] = np.asarray([None])
        elif (type(dd[k]) in [int,float,bool,str]
              or issubclass(dd[k].__class__,np.int)
              or issubclass(dd[k].__class__,np.float)):
            dd[k] = np.asarray([dd[k]])
        elif type(dd[k]) in [list,tuple]:
            dd[k] = np.asarray(dd[k])
        elif not isinstance(dd[k], np.ndarray):
            msg += "\n    {0} : {1}".format(k,str(type(dd[k])))
            err = True
    if err:
        raise Exception(msg)
    dd.update(**dt)
    func(pathfileext, **dd)

def _save_mat(dd, pathfileext, compressed=False):
    # Create intermediate dict to make sure to get rid of None values
    dmat = {}
    msg = "How to deal with:"
    msg += "\n SaveName : {0}".format(dd['dId_dall_SaveName'])
    msg += "\n Attributes:"
    err = False
    dt = {}
    for k in dd.keys():
        kt = k+'_type'
        dt[kt] = np.asarray([type(d[k]).__name__])
        if type(dd[k]) in [int,float,np.int64,np.float64,bool]:
            dmat[k] = np.asarray([dd[k]])
        elif type(dd[k]) in [tuple,list]:
            dmat[k] = np.asarray(dd[k])
        elif isinstance(dd[k],str):
            dmat[k] = np.asarray([dd[k]])
        elif type(dd[k]) is np.ndarray:
            dmat[k] = dd[k]
        else:
            msg += "\n    {0} : {1}".format(k,str(type(dd[k])))
            err = True
    if err:
        raise Exception(msg)
    dmat.update(**dt)
    scpio.savemat(pathfileext, dmat, do_compression=compressed, format='5')

###################################
#       loading routines
###################################

def _filefind(name, path=None, lmodes=['.npz','.mat']):
    c0 = isinstance(name,str)
    c1 = isinstance(name,list) and all([isinstance(ss,str) for ss in name])
    if not (c0 or c1):
        msg = "Arg name must be a str (file name or full path+file)"
        msg += " or a list of str patterns to be found at pathi\n"
        msg += "    name : %s"%name
        raise Exception(msg)
    if path is not None and not isinstance(path,str):
        msg = "Arg path must be a str !"
        raise Exception(msg)

    # Extract folder and file name
    if isinstance(name,str):
        p, f = os.path.split(name)
        name = [f]
        if p!='':
            path = p
        elif path is None:
            path = './'
    path = os.path.normpath(os.path.abspath(path))
    if not os.path.isdir(path):
        msg = "Specified folder does not exist :"
        msg += "\n    {0}".format(path)
        raise Exception(msg)

    # Check unicity of matching file
    lf = os.listdir(path)
    lf = [ff for ff in lf if all([ss in ff for ss in name])]
    if len(lf) != 1:
        msg = "No / several matching files found:"
        msg += "\n  folder: {0}".format(path)
        msg += "\n  for   : {0}".format('['+', '.join(name)+']')
        msg += "\n    " + "\n    ".join(lf)
        raise Exception(msg)
    nameext = lf[0]

    # Check file extension
    indend = [ss==nameext[-4:] for ss in lmodes]
    indin = [ss in nameext for ss in lmodes]
    if np.sum(indend) != 1 or np.sum(indin) != 1:
        msg = "None / too many of the available file extensions !"
        msg += "\n  file: {0}".format(nameext)
        msg += "\n  ext.: {0}:".format('['+', '.format(lmodes)+']')
        raise Exception(msg)

    # load and format dict
    name = nameext[:-4]
    mode = lmodes[np.argmax(indend)].replace('.','')
    pfe = os.path.join(path,nameext)
    return name, mode, pfe



def load(name, path=None, strip=None, verb=True):
    """     Load a tofu object file

    Can load from .npz or .txt files
        In future versions, will also load from .mat

    The file must have been saved with tofu (i.e.: must be tofu-formatted)
    The associated tofu object will be created and returned

    Parameters
    ----------
    name:   str
        Name of the file to load from, can include the path
    path:   None / str
        Path where the file is located (if not provided in name), defaults './'
    strip:  None / int
        FLag indicating whether to strip the object of some attributes
            => see the docstring of the class strip() method for details
    verb:   bool
        Flag indocating whether to print a summary of the loaded file
    """

    lmodes = ['.npz','.mat','.txt']
    name, mode, pfe = _filefind(name=name, path=path, lmodes=lmodes)

    if mode == 'txt':
        obj = _load_from_txt(name, pfe)
    else:
        if mode == 'npz':
            dd = _load_npz(pfe)
        elif mode == 'mat':
            dd = _load_mat(pfe)

        # Recreate from dict
        exec("import tofu.{0} as mod".format(dd['dId_dall_Mod']))
        obj = eval("mod.{0}(fromdict=dd)".format(dd['dId_dall_Cls']))

    if strip is not None:
        obj.strip(strip=strip)

    # print
    if verb:
        msg = "Loaded from:\n"
        msg += "    "+pfe
        print(msg)
    return obj

def _load_npz(pathfileext):

    try:
        out = np.load(pathfileext, mmap_mode=None)
    except UnicodeError:
        out = np.load(pathfileext, mmap_mode=None, encoding='latin1')
    except Exception as err:
        raise err

    C = ['dId' in kk for kk in out.keys()]
    if np.sum(C)<1:
        msg = "There does not seem to be a dId in {0}".format(pathfileext)
        msg += "\n    => Is it really a tofu-generated file ?"
        raise Exception(msg)

    lk = [k for k in out.keys() if not k[-5:]=='_type']
    dout = dict.fromkeys(lk)

    msg = "How to deal with:"
    msg += "\n SaveName : {0}".format(out['dId_dall_SaveName'])
    msg += "\n Attributes:"
    err = False
    for k in lk:
        kt = k+'_type'
        typ = out[kt]
        if typ=='NoneType':
            dout[k] = None
        elif typ in ['int','int64']:
            dout[k] = int(out[k][0]) if typ=='int' else out[k][0]
        elif typ in ['float','float64']:
            dout[k] = float(out[k][0]) if typ=='float' else out[k][0]
        elif typ in ['bool','bool_']:
            dout[k] = bool(out[k][0]) if typ=='bool' else out[k][0]
        elif typ in ['str','str_']:
            dout[k] = str(out[k][0]) if typ=='str' else out[k][0]
        elif typ in ['list','tuple']:
            dout[k] = out[k].tolist()
            if typ=='tuple':
                dout[k] = tuple(out[k])
        elif typ=='ndarray':
            dout[k] = np.array(out[k])
        else:
            msg += "\n    {0} : {1}".format(k,typ)
            err = True
    if err:
        raise Exception(msg)
    return dout


#######
#   tf.geom.Struct - specific
#######

def _load_from_txt(name, pfe):

    # Extract class
    lk = name.split('_')
    lCls = ['PFC','CoilPF','CoilCS','Ves','PlasmaDomain']
    lcc = [np.sum([k == cls for k in lk]) == 1 for cls in lCls]
    if not np.sum(lcc) == 1:
        msg = "Provided file name does not include any known Struct subclass:\n"
        msg += "    - Provided file name: %s\n"%name
        msg += "     - Valid classes: [%s]"%', '.join(lCls)
        raise Exception(msg)
    cls = lCls[np.nonzero(lcc)[0][0]]

    # Recreate object
    import tofu.geom as mod
    obj = eval("mod.%s.from_txt(pfe, Name=Name, Exp=Exp)"%cls)
    return obj



#############################################
#       Generic tofu object
#############################################

def _check_notNone(dd, lk):
    for k, v in dd.items():
        if k in lk:
            assert v is not None, "{0} should not be None !".format(k)

def _check_InputsGeneric(ld, tab=0):

    # Prepare
    bstr0 = "\n"+"    "*tab + "Error on arg %s:"
    bstr1 = "\n"+"    "*(tab+1) + "Expected: "
    bstr2 = "\n"+"    "*(tab+1) + "Provided: "

    ltypes_f2i = [int,float,np.integer,np.floating]
    ltypes_i2f = [int,float,np.integer,np.floating]

    # Check
    err, msg = False, ''
    for k in ld.keys():
        errk, msgk = False, bstr0%k
        if 'cls' in ld[k].keys():
            if not isinstance(ld[k]['var'],ld[k]['cls']):
                errk = True
                msgk += bstr1 + "class {0}".format(ld[k]['cls'].__name__)
                msgk += bstr2 + "class %s"%ld[k]['var'].__class__.__name__
        if 'NoneOrCls' in ld[k].keys():
            c = ld[k]['var'] is None or isinstance(ld[k]['var'],ld[k]['cls'])
            if not c:
                errk = True
                msgk += bstr1 + "None or class {0}".format(ld[k]['cls'].__name__)
                msgk += bstr2 + "class %s"%ld[k]['var'].__class__.__name__
        if 'in' in ld[k].keys():
            if not ld[k]['var'] in ld[k]['in']:
                errk = True
                msgk += bstr1 + "in {0}".format(ld[k]['in'])
                msgk += bstr2 + "{0}".format(ld[k]['var'])
        if 'lisfof' in ld[k].keys():
            c0 = isinstance(ld[k]['var'], list)
            c1 = c0 and all([isinstance(s,ld[k]['listof']) for s in ld[k]])
            if not c1:
                errk = True
                msgk += bstr1 + "list of {0}".format(ld[k]['listof'].__name__)
                msgk += bstr2 + "{0}".format(ld[k]['var'])
        if 'iter2array' in ld[k].keys():
            c0 = ld[k]['var'] is not None and hasattr(ld[k]['var'],'__iter__')
            if not c0:
                errk = True
                msgk += bstr1 + "iterable of %s"%ld[k]['iter2array'].__name__
                msgk += bstr2 + "{0}".format(ld[k]['var'])
            ld[k]['var'] = np.asarray(ld[k]['var'], dtype=ld[k]['iter2array'])
        if 'ndim' in ld[k].keys():
            c0 = isinstance(ld[k]['var'], np.ndarray)
            c1 = c0 and ld[k]['var'].ndim == ld[k]['ndim']
            if not c1:
                errk = True
                msgk += bstr1 + "array of {0} dimensions".format(ld[k]['ndim'])
                msgk += bstr2 + "shape {0}".format(ld[k]['ndim'].shape)
        if 'inshape' in ld[k].keys():
            c0 = isinstance(ld[k]['var'], np.ndarray)
            c1 = c0 and ld[k]['inshape'] in ld[k]['var'].shape
            if not c1:
                errk = True
                msgk += bstr1 + "shape including {0}".format(ld[k]['inshape'])
                msgk += bstr2 + "shape {0}".format(ld[k]['var'].shape)
        if 'float2int' in ld[k].keys():
            lc = [(issubclass(ld[k]['var'].__class__, cc)
                   and int(ld[k]['var'])==ld[k]['var'])
                  for cc in ltypes_f2i]
            if not any(lc):
                errk = True
                msgk += bstr1 + "convertible to int from %s"%str(ltypes_f2i)
                msgk += bstr2+"{0} ({1})".format(ld[k]['var'],
                                                 ld[k]['var'].__class__.__name__)
            ld[k]['var'] = int(ld[k]['var'])
        if 'int2float' in ld[k].keys():
            lc = [issubclass(ld[k]['var'].__class__, cc)
                  for cc in ltypes_i2f]
            if not any(lc):
                errk = True
                msgk += bstr1 + "convertible to float from %s"%str(ltypes_i2f)
                msgk += bstr2 + "class %s"%ld[k]['var'].__class__.__name__
            ld[k]['var'] = float(ld[k]['var'])
        if 'NoneOrIntPos' in ld[k].keys():
            c0 = ld[k]['var'] is None
            lc = [(issubclass(ld[k]['var'].__class__, cc)
                   and int(ld[k]['var'])==ld[k]['var']
                   and ld[k]['var']>0)
                  for cc in ltypes_f2i]
            if not (c0 or any(lc)):
                errk = True
                msgk += bstr1 + "convertible to >0 int from %s"%str(ltypes_f2i)
                msgk += bstr2 + "{0}".format(ld[k]['var'])
            ld[k]['var'] = None if c0 else int(ld[k]['var'])
        if '>' in ld[k].keys():
            if not np.all(np.greater(ld[k]['var'], ld[k]['>'])):
                errk = True
                msgk += bstr1 + "> {0}".format(ld[k]['>'])
                msgk += bstr2 + "{0}".format(ld[k]['var'])
        if 'vectnd' in ld[k].keys():
            c0 = any([isinstance(ld[k]['var'],tt)
                      for tt in [list,tuple,np.ndarray]])
            if ld[k]['vectnd'] is not None:
                c0 &= np.asarray(ld[k]['var']).size==ld[k]['vectnd']
            if not c0:
                errk = True
                msgk += bstr1 + "array of size {0}".format(ld[k]['vectnd'])
                msgk += bstr2 + "{0}".format(ld[k]['var'])
            ld[k]['var'] = np.asarray(ld[k]['var'],dtype=float).ravel()
        if 'unitvectnd' in ld[k].keys():
            c0 = any([isinstance(ld[k]['var'],tt)
                      for tt in [list,tuple,np.ndarray]])
            c1 = c0 and np.asarray(ld[k]['var']).size==ld[k]['unitvectnd']
            if not c1:
                errk = True
                msgk += bstr1 + "array of size {0}".format(ld[k]['unitvectnd'])
                msgk += bstr2 + "{0}".format(ld[k]['var'])
            temp = np.asarray(ld[k]['var'],dtype=float).ravel()
            ld[k]['var'] = temp/np.linalg.norm(temp)

        if errk:
            err = True
            msg += msgk
    return ld, err, msg

def _get_attrdictfromobj(obj, dd):
    for k in dd.keys():
        if dd[k] is None:
            dd[k] = getattr(obj,k)
    return dd


class ToFuObjectBase(object):

    __metaclass__ = ABCMeta
    _dstrip = {'strip':None, 'allowed':None}

    # Does not exist before Python 3.6 !!!
    def __init_subclass__(cls, *args, **kwdargs):
        # super()
        super(ToFuObjectBase,cls).__init_subclass__(*args, **kwdargs)
        cls._dstrip = ToFuObjectBase._dstrip.copy()
        cls._strip_init()

    def __init__(self, fromdict=None,
                 **kwdargs):

        self._Done = False
        self._dstrip = self.__class__._dstrip.copy()
        if fromdict is not None:
            self.from_dict(fromdict)
        else:
            self._reset()
            self._set_Id(**kwdargs)
            self._init(**kwdargs)
        self._Done = True


    @abstractmethod
    def _reset(self):
        """ To be overloaded """
        pass

    def _set_Id(self, *args, **kwdargs):
        """ To be overloaded """
        pass

    @abstractmethod
    def _init(self, **kwdargs):
        """ To be overloaded """
        pass

    @classmethod
    def _strip_init(cls):
        """ To be overloaded """
        pass

    @staticmethod
    def _get_largs_Id():
        largs = ['Id','Name','Type','Deg','Exp','Diag','shot',
                 'SaveName','SavePath','usr','dUSR','lObj','include']
        return largs

    @staticmethod
    def _extract_kwdargs(din, largs):
        dout = {}
        for k in largs:
            if k in din.keys():
                dout[k] = din[k]
        return dout

    def _set_arrayorder(self, arrayorder='C', verb=True):
        d, account = _set_arrayorder(self, arrayorder=arrayorder)
        if len(account['Failed'])>0:
            msg = "All np.ndarrays were not set to {0} :\n".format(arrayorder)
            msg += "Success : [{0}]".format(', '.join(account['Success']))
            msg += "Failed :  [{0}]".format(', '.join(account['Failed']))
            raise Exception(msg)
        else:
            self.from_dict(d)
            self._dextra['arrayorder'] = arrayorder

    @staticmethod
    def _strip_dict(dd, lkeep=[]):
        for k in dd.keys():
            if not k in lkeep:
                dd[k] = None

    @staticmethod
    def _test_Rebuild(dd, lkeep=[]):
        reset = False
        for k in dd.keys():
            if dd[k] is None and k not in lkeep:
                reset = True
                break
        return reset

    @staticmethod
    def _check_Fields4Rebuild(dd, lkeep=[], dname=''):
        for kk in lkeep:
            if kk not in dd.keys() or dd[kk] is None:
                msg = "Rebuilding {0}:\n".format(dname)
                msg += "Field '{0}' is missing !".format(kk)
                raise Exception(msg)

    @staticmethod
    def _check_InputsGeneric(ld, tab=0):
        return _check_InputsGeneric(ld, tab=tab)

    def strip(self, strip=0, **kwdargs):
        """ Remove non-essential attributes to save memory / disk usage

        Useful to save a very compressed version of the object
        The higher strip => the more stripped the object
        Use strip=0 to recover all atributes

        See the difference by doing:
            > self.get_nbytes()
            > self.strip(-1)
            > self.get_nbytes()

        Parameters
        ----------
        strip:      int
            Flag indicating how much to strip from the object
                 0: Make complete (retrieve all attributes){0}
                -1: Equivalent to strip={1}
        """
        msg = "Only allowed strip values are:\n"
        msg += "    "+ ", ".join(["{0}".format(ii)
                                  for ii in self._dstrip['allowed']])
        assert strip in [-1]+self._dstrip['allowed'], msg
        strip = self._dstrip['allowed'][strip]

        # --------------------------------
        # Call class-specific strip method
        self._strip(strip, **kwdargs)
        # --------------------------------

        self._dstrip['strip'] = strip


    def to_dict(self, strip=None, sep=_sep, deep='ref'):
        """ Return a flat dict view of the object's attributes

        Useful for:
            * displaying all attributes
            * saving to file
            * exchaning data with non-tofu libraries

        Parameters
        ----------
        strip :     int
            Flag indicating how stripped the object should be
            Fed to self.strip()
        sep :       str
            Separator char used for flattening the dict
            The output dict is flat (i.e.: no nested dict)
            Keys are created from the keys of nested dict, separated by sep
        deep:       str
            Flag indicating how to behave when an attribute is itself a tofu
            object. The associated field in the exported dict can be:
                - 'ref' : a simple reference to the object
                - 'copy': a tofu object itself (i.e.: a copy of the original)
                - 'dict': the tofu object is itself exported as a dict
                    (using also self.to_dict())

        Return
        ------
        dout :      dict
            Flat dict containing all the objects attributes

        """
        if deep not in ['ref','copy','dict']:
            msg = "Arg deep must be a flag in ['ref','copy','dict'] !"
            raise Exception(msg)
        if strip is None:
            strip = self._dstrip['strip']
        if self._dstrip['strip'] != strip:
            self.strip(strip)

        # ---------------------
        # Call class-specific
        dd = self._to_dict()
        # ---------------------
        dd['dId'] = self._get_dId()
        dd['dstrip'] = {'dict':self._dstrip, 'lexcept':None}

        dout = {}
        for k, v in dd.items():
            lexcept_key = v.get('lexcept_key', None)
            try:
                d = flatten_dict(v['dict'],
                                 parent_key='', sep=sep, deep=deep,
                                 lexcept_key=lexcept_key)
            except Exception as err:
                msg = str(err)
                msg += "\nIssue flattening dict %s"%k
                msg += "\n\n\n" + str(v['dict'])
                raise Exception(msg)
            dout[k] = d
        dout = flatten_dict(dout, parent_key='', sep=sep, deep=deep)
        return dout

    def _get_dId(self):
        """ To be overloaded """
        return {'dict':{}}

    def from_dict(self, fd, sep=_sep, strip=None):
        """ Populate the instances attributes using an input dict

        The input dict must be properly formatted
        In practice it should be the return output of a similar class to_dict()

        Parameters
        ----------
        fd :    dict
            The properly formatted ditionnary from which to read the attributes
        sep :   str
            The separator that was used to format fd keys (cf. self.to_dict())
        strip : int
            Flag indicating how stripped the resulting object shouyld be
            (cf. self.strip())
        """

        self._reset()
        dd = reshape_dict(fd)

        # ---------------------
        # Call class-specific
        self._from_dict(dd)
        # ---------------------
        self._dstrip.update(**dd['dstrip'])
        if 'dId' in dd.keys():
            self._set_Id(Id=ID(fromdict=dd['dId']))

        if strip is None:
            strip = self._dstrip['strip']
        if self._dstrip['strip'] != strip:
            self.strip(strip, verb=verb)

    def copy(self, strip=None, deep='ref'):
        """ Return another instance of the object, with the same attributes

        If deep=True, all attributes themselves are also copies
        """
        dd = self.to_dict(strip=strip, deep=deep)
        return self.__class__(fromdict=dd)

    def get_nbytes(self):
        """ Compute and return the object size in bytes (i.e.: octets)

        A flat dict containing all the objects attributes is first created
        The size of each attribute is then estimated with np.asarray().nbytes

        Note :
            if the attribute is a tofu object, get_nbytes() is recursive

        Returns
        -------
        total :     int
            The total object estimated size, in bytes
        dsize :     dict
            A dictionnary giving the size of each attribute
        """
        dd = self.to_dict()
        dsize = dd.fromkeys(dd.keys(),0)
        total = 0
        for k, v in dd.items():
            if issubclass(v.__class__, ToFuObjectBase):
                dsize[k] = v.get_nbytes()[0]
            else:
                dsize[k] = np.asarray(v).nbytes
            total += dsize[k]
        return total, dsize


    def __eq__(self, obj, lexcept=[], detail=True, verb=True):
        msg = "The 2 objects have different "
        # Check class
        eq = self.__class__==obj.__class__
        if not eq:
            msg += "classes :\n"
            msg += str(self.__class__)+"\n"
            msg += str(obj.__class__)

        # Check keys
        if eq:
            d0 = self.to_dict(strip=None)
            d1 = obj.to_dict(strip=None)
            lk0 = sorted(list(d0.keys()))
            lk1 = sorted(list(d1.keys()))
            eq = lk0==lk1
            if not eq:
                msg += "dict keys :\n"
                msg += '    ['+', '.join([k for k in lk0 if k not in lk1])+']\n'
                msg += '    ['+', '.join([k for k in lk1 if k not in lk0])+']'

        # Check values
        if eq:
            msg += "dict values :\n"
            lsimple = [str,bool,np.str_,np.bool_,
                       tuple, list]
            for k in lk0:
                if any([ss in k for ss in lexcept]):
                    continue
                eqk = type(d0[k]) == type(d1[k])
                if not eqk:
                    eq = False
                    msg += k+" types :\n"
                    msg += "    "+str(type(d0[k]))+"\n"
                    msg += "    "+str(type(d1[k]))+"\n"
                    if not detail:
                        break
                if eqk:
                    if d0[k] is None or type(d0[k]) in lsimple:
                        eqk = d0[k]==d1[k]
                        if not eqk:
                            m0 = str(d0[k])
                            m1 = str(d1[k])
                    elif type(d0[k]) in [int,float,np.int64,np.float64]:
                        eqk = np.allclose([d0[k]],[d1[k]], equal_nan=True)
                        if not eqk:
                            m0 = str(d0[k])
                            m1 = str(d1[k])
                    elif type(d0[k]) is np.ndarray:
                        eqk = d0[k].shape==d1[k].shape
                        if eqk:
                            eqk = d0[k].dtype == d1[k].dtype
                            if eqk:
                                if (issubclass(d0[k].dtype.type, np.int)
                                    or issubclass(d0[k].dtype.type, np.float)):
                                    eqk = np.allclose(d0[k],d1[k], equal_nan=True)
                                else:
                                    eqk = np.all(d0[k]==d1[k])
                                if not eqk:
                                    m0 = str(d0[k])
                                    m1 = str(d1[k])
                            else:
                                m0 = str(d0[k].dtype)
                                m1 = str(d1[k].dtype)
                        else:
                            m0 = "shape {0}".format(d0[k].shape)
                            m1 = "shape {0}".format(d1[k].shape)
                    elif issubclass(d0[k].__class__, ToFuObjectBase):
                        eqk = d0[k]==d1[k]
                        if not eqk:
                            m0 = str(d0[k])
                            m1 = str(d1[k])
                    elif isinstance(d0[k], mplTri):
                        eqk = np.allclose(d0[k].x, d1[k].x)
                        if not eqk:
                            m0 = 'x ' + str(d0[k].x)
                            m1 = 'x ' + str(d1[k].x)
                        else:
                            eqk = np.allclose(d0[k].y, d1[k].y)
                            if not eqk:
                                m0 = 'y ' + str(d0[k].y)
                                m1 = 'y ' + str(d1[k].y)
                            else:
                                eqk = np.all(d0[k].triangles == d1[k].triangles)
                                if not eqk:
                                    m0 = 'tri ' + str(d0[k].triangles)
                                    m1 = 'tri ' + str(d1[k].triangles)
                    else:
                        msg = "How to handle :\n"
                        msg += "    {0} is a {1}".format(k,str(type(d0[k])))
                        raise Exception(msg)
                    if not eqk:
                        eq = False
                        msg += k+" :\n"
                        msg += "    "+m0+"\n"
                        msg += "    "+m1+"\n"
                        if not detail:
                            break

        if not eq and verb:
            print(msg)
        return eq

    # Python 3
    def __neq__(self, obj, detail=True, verb=True):
        return not self.__eq__(obj, detail=detail, verb=verb)

    # Python 2
    def __ne__(self, obj, detail=True, verb=True):
        return not self.__eq__(obj, detail=detail, verb=verb)



class ToFuObject(ToFuObjectBase):

    # Does not exist before Python 3.6 !!!
    def __init_subclass__(cls, *args, **kwdargs):
        # super()
        super(ToFuObject,cls).__init_subclass__(*args, **kwdargs)

    def _set_Id(self, Id=None, Name=None, SaveName=None, SavePath=None,
                Type=None, Deg=None, Exp=None, Diag=None, shot=None, usr=None,
                dUSR=None, lObj=None, include=None, **kwdargs):
        largs = self._get_largs_Id()
        dId = self._extract_kwdargs(locals(), largs)
        dId = self._checkformat_inputs_Id(**dId)
        if Id is None:
            Id = ID(Cls=self.__class__, **dId)
        self._Id = Id


    @property
    def Id(self):
        """ The ID instance of the object

        The ID class in tofu objects contains all information used for
        identifying the instance and automatically managing saving / loading
        In particular, it yields information such as:
                - Name of the object
                - Name that would be used for the file in case of saving
                - Path under which it would be saved
                - Experiment the object is associated to
                - Diagnostic the object is associated to
                - Type of the object
                - shot the object is associated to
                - version of tofu used for creating the object
                - ...

        """
        return self._Id

    def _get_dId(self):
        return {'dict':self.Id.to_dict()}

    def _reset(self):
        if hasattr(self,'_Id'):
            self._Id._reset()


    def _set_dlObj(self, lObj, din={}):

        # Make sure to kill the link to the mutable being provided
        nObj = len(lObj)
        lCls, lorder, lerr = [], [], []
        for obj in lObj:
            if not issubclass(obj.__class__, ToFuObject):
                msg = "The following obj is not a ToFuObject subclass:\n"
                msg += str(obj)
                raise Exception(msg)
            cls = obj.__class__.__name__
            name = obj.Id.Name
            clsname = obj.Id.SaveName_Conv(Cls=cls, Name=name,
                                           include=['Cls','Name'])
            if cls not in lCls:
                lCls.append(cls)
                if clsname in lorder:
                    lerr.append(clsname)
            lorder.append(clsname)

        if len(lerr)>0:
            msg = "There is an ambiguity in the names :"
            msg += "\n    - " + "\n    - ".join(lerr)
            msg += "\n => Please clarify (choose unique Cls/Names)"
            raise Exception(msg)

        # Initialize dObj is not existing
        if 'dObj' not in din.keys() or din['dObj'] is None:
            din['dObj'] = dict([(k,{}) for k in lCls])

        # Initisalize
        for k in lCls:
            if not k in din['dObj'].keys():
                din['dObj'][k] = {}
            lk = din['dObj'][k].keys()
            ls = [ss for ss in lObj if ss.Id.Cls == k]
            for ss in ls:
                name = ss.Id.Name
                if not name in lk:
                    din['dObj'][k][name] = ss.copy()
                if din['dObj'][k][name]._dstrip['strip'] != 0:
                    din['dObj'][k][name].strip(0)
        din.update({'nObj':nObj, 'lorder':lorder, 'lCls':lCls})

    @staticmethod
    def _get_ind12r_n12(ind1=None, ind2=None, n1=None, n2=None):
        c0 = ind1 is None and ind2 is None
        c1 = n1 is not None and n2 is not None
        assert c1, "Provide n1 and n2 !"
        if c0:
            ind1 = np.tile(np.arange(0,n1), n2)
            ind2 = np.repeat(np.arange(0,n2), n1)
        else:
            ind1 = np.asarray(ind1).ravel().astype(int)
            ind2 = np.asarray(ind2).ravel().astype(int)
            assert ind1.size == ind2.size
            assert np.all(ind1>=0) and np.all(ind1<n1)
            assert np.all(ind2>=0) and np.all(ind2<n2)
        indr = np.zeros((n2,n1),dtype=int)
        for ii in range(0,ind1.size):
            indr[ind2[ii],ind1[ii]] = ii
        return ind1, ind2, indr

    def save(self, path=None, name=None,
             strip=None, sep=_sep, deep=True, mode='npz',
             compressed=False, verb=True, return_pfe=False):
        return save(self, path=path, name=name,
                    sep=sep, deep=deep, mode=mode,
                    strip=strip, compressed=compressed,
                    return_pfe=return_pfe, verb=verb)

if sys.version[0]=='2':
    ToFuObject.save.__func__.__doc__ = save.__doc__
else:
    ToFuObject.save.__doc__ = save.__doc__


#############################################
#       ID class
#############################################


class ID(ToFuObjectBase):
    """ A class used by all ToFu objects as an attribute

    It stores all relevant data for the identification of instances
    Stored info can be the name of the instance, the experiment and diagnostics
    it belongs to, or other user-defined info
    Also provides default names for saving the instances

    Parameters
    ----------
    Cls :       str
        Class of the object on which info should be stored:
    Name :      str
        Name of the instance (user-defined)
        Should be a str without space ' ' or underscore '_'
        (automatically removed if present)
    Type :      None / str
        Type of object (i.e.: 'Tor' or 'Lin' for a :class:`~tofu.geom.Ves`)
    Deg :       None / int
        Degree of the b-splines constituting the :mod:`tofu.mesh` object
    Exp :       None / str
        Flag specifying the experiment (e.g.: 'WEST', 'AUG', 'ITER', 'JET'...)
    Diag :      None / str
        Flag indicating the diagnostic (e.g.: 'SXR', 'HXR', 'Bolo'...)
    shot :      None / int
        A shot number from which the instance is valid (for tracking changes)
    SaveName :  None / str
        Overrides the default file name for saving (not recommended)
    SavePath :  None / str
        Absolute path where the instance should be saved
    dUSR :   None / dict
        A user-defined dictionary containing information about the instance
        All info considered relevant can be passed here
        (e.g.: thickness of the diode, date of installation...)
    lObj :      None / dict / list
        Either:
            - list: list of other ID instances of objects on which the created object depends
              (this list will then be sorted by class and formatted into a dictionary storign key attributes)
            - dict: a ready-made such dictionary

    """

    _dModes = {'geom':'TFG', 'data':'TFD'}
    _defInclude = ['Mod','Cls','Type','Exp','Deg','Diag','Name','shot']
    _dPref = {'Exp':'Exp','Diag':'Dg','shot':'sh','Deg':'Deg',
              'version':'Vers','usr':'U'}

    def __init__(self, Cls=None, Name=None, Type=None, Deg=None,
                 Exp=None, Diag=None, shot=None, SaveName=None,
                 SavePath=None, usr=None, dUSR=None, lObj=None,
                 fromdict=None, include=None):

        # To replace __init_subclass__ for Python 2
        if sys.version[0]=='2':
            self._dstrip = ToFuObjectBase._dstrip.copy()
            self.__class__._strip_init()

        kwdargs = locals()
        del kwdargs['self']
        #super()
        super(ID, self).__init__(**kwdargs)

    def _reset(self):
        self._dall = dict.fromkeys(self._get_keys_dall())

    ###########
    # Get largs
    ###########

    @staticmethod
    def _get_largs_dall():
        largs = ['Cls', 'Name', 'Type', 'Deg',
                 'Exp', 'Diag', 'shot', 'SaveName',
                 'SavePath', 'usr', 'dUSR', 'lObj', 'include']
        return largs

    ###########
    # Get check and format inputs
    ###########

    @staticmethod
    def _checkformat_inputs_dall(usr=None, Cls=None, Type=None,
                                 SavePath=None, Exp=None, Diag=None,
                                 shot=None, Deg=None, Name=None,
                                 SaveName=None, include=None,
                                 lObj=None, dUSR=None):
        # Str args
        ls = [usr,Type,SavePath,Exp,Diag,SaveName]
        assert all(ss is None or type(ss) is str for ss in ls)
        if usr is None:
            try:
                usr = getpass.getuser()
            except:
                pass
        assert shot is None or type(shot) is int and shot>=0
        assert Deg is None or type(Deg) is int and Deg>=0
        assert Cls is not None
        assert issubclass(Cls, ToFuObject)
        assert include is None or type(include) is list
        dout = locals()
        del dout['ls']
        return dout

    ###########
    # Get keys of dictionnaries
    ###########

    @staticmethod
    def _get_keys_dall():
        lk = ['Mod', 'Cls', 'Type', 'Name', 'SaveName',
              'SavePath', 'Exp', 'Diag', 'shot', 'Deg',
              'version', 'usr', 'dUSR', 'lObj', 'SaveName-usr']
        return lk

    ###########
    # _init
    ###########

    def _init(self, usr=None, Cls=None, Type=None, SavePath=None,
              Exp=None, Diag=None, shot=None, Deg=None,
              Name=None, SaveName=None, include=None,
              lObj=None, dUSR=None, **kwdargs):
        largs = self._get_largs_dall()
        kwd = self._extract_kwdargs(locals(), largs)
        largs = self._set_dall(**kwd)

    ###########
    # set dictionaries
    ###########

    def _set_dall(self, usr=None, Cls=None, Type=None, SavePath=None,
                  Exp=None, Diag=None, shot=None, Deg=None,
                  Name=None, SaveName=None, include=None,
                  lObj=None, dUSR=None):

        dargs = locals()
        del dargs['self']
        dargs = ID._checkformat_inputs_dall(**dargs)

        self._dall['version'] = __version__
        lasis = ['usr','Type','SavePath','Exp','Diag','shot','Deg']
        dasis = dict([(k,dargs[k]) for k in lasis])
        self._dall.update(dasis)

        # Set fixed attributes
        Mod, Cls = ID._extract_ModClsFrom_class(dargs['Cls'])
        self._dall['Mod'] = Mod
        self._dall['Cls'] = Cls

        # Set variable attributes
        self.set_Name(Name, SaveName=SaveName, include=include)

        self.set_lObj(lObj)
        self.set_dUSR(dUSR)

    ###########
    # strip dictionaries
    ###########

    def _strip_dall(self, lkeep=[]):
        pass

    ###########
    # rebuild dictionaries
    ###########

    def _rebuild_dall(self, lkeep=[]):
        pass


    ###########
    # _strip and get/from dict
    ###########

    @classmethod
    def _strip_init(cls):
        cls._dstrip['allowed'] = [0]
        nMax = max(cls._dstrip['allowed'])
        doc = ""
        doc = ToFuObjectBase.strip.__doc__.format(doc,nMax)
        if sys.version[0]=='2':
            cls.strip.__func__.__doc__ = doc
        else:
            cls.strip.__doc__ = doc

    def strip(self, strip=0):
        #super()
        super(ID,self).strip(strip=strip)

    def _strip(self, strip=0):
        pass

    def _to_dict(self):
        dout = {'dall':{'dict':self.dall, 'lexcept':None}}
        return dout

    def _from_dict(self, fd):
        self._dall.update(**fd['dall'])
        if 'version' in fd.keys() and fd['version']!=__version__:
            msg = "The object was created with a different tofu version !"
            msg += "\n  object: {0}".format(fd['SaveName'])
            msg += "\n    original : tofu {0}".fd['version']
            msg += "\n    current  : tofu {0}".__version__
            warnings.warn(msg)

    ###########
    # Properties
    ###########

    @property
    def dall(self):
        return self._dall
    @property
    def Mod(self):
        return self._dall['Mod']
    @property
    def Cls(self):
        return self._dall['Cls']
    @property
    def Name(self):
        return self._dall['Name']
    @property
    def NameLTX(self):
        return r"$"+self.Name.replace('_','\_')+r"$"
    @property
    def Exp(self):
        return self._dall['Exp']
    @property
    def Diag(self):
        return self._dall['Diag']
    @property
    def shot(self):
        return self._dall['shot']
    @property
    def usr(self):
        return self._dall['usr']
    @property
    def Type(self):
        return self._dall['Type']
    @property
    def Deg(self):
        return self._dall['Deg']
    @property
    def SaveName(self):
        return self._dall['SaveName']
    @property
    def SavePath(self):
        return self._dall['SavePath']
    @property
    def lObj(self):
        return self._dall['lObj']
    @property
    def dUSR(self):
        return self._dall['dUSR']
    @property
    def version(self):
        return self._dall['version']

    ###########
    # semi-public methods
    ###########

    @staticmethod
    def _extract_ModClsFrom_class(Cls):
        strc = str(Cls)
        ind0 = strc.index('tofu.')+5
        indeol = strc.index("'>")
        strc = strc[ind0:indeol]
        indp = strc.index('.')
        Mod = strc[:indp]
        strc = strc[indp+1:][::-1]
        #cls = strc[:strc.index('.')][::-1]
        return Mod, Cls.__name__

    @staticmethod
    def SaveName_Conv(Mod=None, Cls=None, Type=None, Name=None, Deg=None,
                      Exp=None, Diag=None, shot=None, version=None, usr=None,
                      include=None):
        """ Return a default name for saving the object

        Includes key info for fast identification of the object from file name
        Used on object creation by :class:`~tofu.pathfile.ID`
        It is recommended to use this default name.

        """
        Modstr = ID._dModes[Mod] if Mod is not None else None
        include = ID._defInclude if include is None else include
        if Cls is not None and Type is not None and 'Type' in include:
            Clsstr = Cls+Type
        else:
            Clsstr = Cls
        Dict = {'Mod':Modstr, 'Cls':Clsstr, 'Name':Name}
        for ii in include:
            if not ii in ['Mod','Cls','Type','Name']:
                Dict[ii] = None
            if ii=='Deg' and Deg is not None:
                Dict[ii] = ID._dPref[ii]+'{0:02.0f}'.format(Deg)
            elif ii=='shot' and shot is not None:
                Dict[ii] = ID._dPref[ii]+'{0:05.0f}'.format(shot)
            elif not (ii in ['Mod','Cls','Type','Name'] or eval(ii+' is None')):
                Dict[ii] = ID._dPref[ii]+eval(ii)
        if 'Data' in Cls:
            Order = ['Mod','Cls','Exp','Deg','Diag','shot',
                     'Name','version','usr']
        else:
            Order = ['Mod','Cls','Exp','Deg','Diag','Name',
                     'shot','version','usr']

        SVN = ""
        for ii in range(0,len(Order)):
            if Order[ii] in include and Dict[Order[ii]] is not None:
                SVN += '_' + Dict[Order[ii]]
        SVN = SVN.replace('__','_')
        if SVN[0]=='_':
            SVN = SVN[1:]
        return SVN

    ###########
    # public methods
    ###########


    def set_Name(self, Name, SaveName=None,
                 include=None,
                 ForceUpdate=False):
        """ Set the Name of the instance, automatically updating the SaveName

        The name should be a str without spaces or underscores (removed)
        When the name is changed, if SaveName (i.e. the name used for saving)
        was not user-defined, it is automatically updated

        Parameters
        ----------
        Name :      str
            Name of the instance, without ' ' or '_' (automatically removed)
        SaveName :  None / str
            If provided, overrides the default name for saving (not recommended)
        include:    list
            Controls how te default SaveName is generated
            Each element of the list is a key str indicating whether an element
            should be present in the SaveName

        """
        self._dall['Name'] = Name
        self.set_SaveName(SaveName=SaveName, include=include,
                          ForceUpdate=ForceUpdate)

    def set_SaveName(self,SaveName=None,
                     include=None,
                     ForceUpdate=False):
        """ Set the name for saving the instance (SaveName)

        SaveName can be either:
            - provided by the user (no constraint) - not recommended
            - automatically generated from Name and key attributes (cf. include)

        Parameters
        ----------
        SaveName :      None / str
            If provided, overrides the default name for saving (not recommended)
        include :       list
            Controls how te default SaveName is generated
            Each element of the list is a key str indicating whether an element
            should be present in the SaveName
        ForceUpdate :   bool
            Flag indicating the behaviour when SaveName=None:
                - True : A new SaveName is generated, overriding the old one
                - False : The former SaveName is preserved (default)
        """
        if not 'SaveName-usr' in self.dall.keys():
            self._dall['SaveName-usr'] = (SaveName is not None)
        # If SaveName provided by user, override
        if SaveName is not None:
            self._dall['SaveName'] = SaveName
            self._dall['SaveName-usr'] = True
        else:
            # Don't update if former is user-defined and ForceUpdate is False
            # Override if previous was:
            # automatic or (user-defined but ForceUpdate is True)
            C0 = self._dall['SaveName-usr']
            C1 = self._dall['SaveName-usr'] and ForceUpdate
            if (not C0) or C1:
                SN = ID.SaveName_Conv(Mod=self.Mod, Cls=self.Cls,
                                      Type=self.Type, Name=self.Name,
                                      Deg=self.Deg, Exp=self.Exp,
                                      Diag=self.Diag, shot=self.shot,
                                      version=self.version, usr=self.usr,
                                      include=include)
                self._dall['SaveName'] = SN
                self._dall['SaveName-usr'] = False

    def generate_SaveName(self, include=None):
        SN = self.SaveName_Conv(Mod=self.Mod, Cls=self.Cls,
                              Type=self.Type, Name=self.Name,
                              Deg=self.Deg, Exp=self.Exp,
                              Diag=self.Diag, shot=self.shot,
                              version=self.version, usr=self.usr,
                              include=include)
        return SN

    def set_lObj(self, lObj=None):
        """ Set the lObj attribute, storing objects the instance depends on

        For example:
        A Detect object depends on a vessel and some apertures
        That link between should be stored somewhere (for saving/loading).
        lObj does this: it stores the ID (as dict) of all objects depended on.

        Parameters
        ----------
        lObj :  None / dict / :class:`~tofu.pathfile.ID` / list of such
            Provide either:
                - A dict (derived from :meth:`~tofu.pathfile.ID._todict`)
                - A :class:`~tofu.pathfile.ID` instance
                - A list of dict or :class:`~tofu.pathfile.ID` instances

        """
        if self.lObj is None and lObj is not None:
            self._dall['lObj'] = {}
        if lObj is not None:
            if type(lObj) is not list:
                lObj = [lObj]
            for ii in range(0,len(lObj)):
                if type(lObj[ii]) is ID:
                    lObj[ii] = lObj[ii].to_dict()
            ClsU = list(set([oo['Cls'] for oo in lObj]))
            for c in ClsU:
                self._dall['lObj'][c] = [oo for oo in lObj if oo['Cls']==c]

    def set_dUSR(self, dUSR={}):
        """ Set the dUSR, containing user-defined info about the instance

        Useful for arbitrary info (e.g.: manufacturing date, material...)

        Parameters
        ----------
        dUSR :   dict
            A user-defined dictionary containing info about the instance

        """
        self._dall['dUSR'] = dUSR






#############################################
#       Geometry
#############################################

def dict_cmp(d1,d2):
    msg = "Different types: %s, %s"%(str(type(d1)),str(type(d2)))
    assert type(d1)==type(d2), msg
    assert type(d1) in [dict,list,tuple]
    if type(d1) is dict:
        l1, l2 = sorted(list(d1.keys())), sorted(list(d2.keys()))
        out = (l1==l2)
    else:
        out = (len(d1)==len(d2))
        l1 = range(0,len(d1))
    if out:
        for k in l1:
            if type(d1[k]) is np.ndarray:
                out = np.all(d1[k]==d2[k])
            elif type(d1[k]) in [dict,list,tuple]:
                out = dict_cmp(d1[k],d2[k])
            else:
                try:
                    out = (d1[k]==d2[k])
                except Exception as err:
                    print(type(d1[k]),type(d2[k]))
                    raise err
            if out is False:
                break
    return out



###############################################
#           DChans
###############################################


class DChans(object):
    """ Base class for handling event on tofu interactive figures """

    def __init__(self, dchans, fromdict=None):

        if fromdict is None:
            dchans, nch = self._check_inputs(dchans)
            self._dchans = dchans
            self._nch = nch
        else:
            self._fromdict(fromdict)

    def _check_inputs(self, fd):
        assert isinstance(fd, dict)
        size = []
        for kk in fd.keys():
            fd[kk] = np.asarray(fd[kk])
            if fd[kk].ndim == 1:
                ss = fd[kk].size
            elif fd[kk].ndim == 2:
                ss = fd[kk].shape[1]
            size.append(ss)
        nch = int(size[0])
        assert np.all([ss == nch for ss in size])
        return fd, ch

    def _todict(self):
        return self._dchans


    def _fromdict(self, fd):
        fd, nch = self._check_inputs(fd)
        self._dchans = fd
        self._nch = nch

    @property
    def dchans(self):
        """ Return the dchans dict """
        return self._dchans

    @property
    def nch(self):
        """ Return the dchans dict """
        return self._nch

    def select(self, key=None, val=None, log='any', out=bool):
        """ The the indices of all channels matching the (key,val) pairs """
        assert out in [bool, int], "Arg out is not valid (int or bool) !"
        C0 = key is None or val is None
        if C0:
            if out is bool:
                ind = np.ones((self._nch,), dtype=bool)
            else:
                ind = np.arange(0, self._nch)
            return ind

        lt0 = [list, tuple, np.ndarray]
        lt1 = [str, int, float, np.int64, np.float64, bool]
        C0 = log in ['any', 'all']
        C1 = type(log) in lt0 and all([ll in ['any', 'all'] for ll in log])
        assert C0 or C1, "Arg out is not valid ('any','all' or an iterable) !"
        C2 = isinstance(key, str) and key in self._dchans.keys()
        assert C2, "Arg key not valid: provide key of self.dchans"
        C4 = type(val) in lt1
        C5 = type(val) in lt0 and all([type(vv) in lt1 for vv in val])
        assert C4 or C5, "Arg val not valid, should be in %s !"%str(lt1)
        if C4:
            val = [val]
        nv = len(val)
        ind = np.vstack([self._dchans[key] == vv for vv in val])
        if log == 'any':
            ind = np.any(ind,axis=0)
        else:
            ind = np.all(ind,axis=0)

        # To be finsihed: add operators and str operations + not

        return ind


###############################################
#           Plot KeyHandler 2
###############################################


def get_indncurind(dind, linds):
    ind = [dind['lrefid'].index(rid) for rid in linds]
    return np.asarray(ind,dtype=int)


def get_indrefind(dind, linds, drefid):
    ninds = len(linds)
    ind = np.zeros((ninds,),dtype=int)
    for ii in range(0,ninds):
        i0 = dind['lrefid'].index(linds[ii])
        ind[ii] = dind['cumsum0'][i0] + drefid[linds[ii]]
    return ind


def get_valf(val, lrids, linds):
    # Python 2 vs 3:
    # The order of arguments is reversed for lambda functions !
    #   => use py2 convention, compatible with both, WRONG !!!
    # Replace *li by li (which is always an array of max 3 elements
    ninds = len(linds)
    if type(val) is list:
        assert ninds == 1 and lrids == linds
        func = lambda li, val=val: val[li[0]]

    elif type(val) is tuple:
        assert ninds == 1 and lrids == linds
        n1, n2 = val[0].size, val[1].size
        # Python 2 and 3 syntax
        def func(li, val=val, n1=n1, n2=n2):
            i1 = li[0] % n1
            i2 = li[0] // n1
            return (val[0][i1], val[1][i2])

    else:
        assert type(val) is np.ndarray
        val = val.squeeze()
        ndim = val.ndim
        assert ndim >= len(lrids)
        assert len(lrids) >= ninds
        assert ndim >= ninds

        if ndim == ninds:
            if ndim == 1:
                func = lambda li, val=val: val[li[0]]

            elif ndim == 2:
                func = lambda li,  val=val: val[li[0],li[1]]

            elif ndim == 3:
                func = lambda li, val=val: val[li[0],li[1],li[2]]

        else:
            lord = np.r_[[lrids.index(ii) for ii in linds]].astype(int)
            if ninds == 1:
                if ndim == 2:
                    if lord[0] == 0:
                        func = lambda li, val=val: val[li[0],:]

                    elif lord[0] == 1:
                        func = lambda li, val=val: val[:,li[0]]

                elif ndim == 3:
                    if lord[0] == 0:
                        func = lambda li, val=val: val[li[0],:,:]

                    elif lord[0] == 1:
                        func = lambda li, val=val: val[:,li[0],:]

                    elif lord[0] == 2:
                        func = lambda li, val=val: val[:,:,li[0]]

            elif ninds == 2:
                assert ndim == 3
                args = np.argsort(lord)
                if np.all(lord[args] == [0,1]):
                    func = lambda  li, val=val: val[li[args[0]], li[args[1]],:]

                elif np.all(lord[args] == [0,2]):
                    func = lambda  li, val=val: val[li[args[0]], :, li[args[1]]]

                if np.all(lord[args] == [1,2]):
                    func = lambda li, val=val: val[:, li[args[0]], li[args[1]]]

    return func

def get_fupdate(obj, typ, n12=None, norm=None, bstr=None):
    if typ == 'xdata':
        f = lambda val, obj=obj: obj.set_xdata(val)
    elif typ == 'ydata':
        f = lambda val, obj=obj: obj.set_ydata(val)
    elif typ in ['data']:   # Also works for imshow
        f = lambda val, obj=obj: obj.set_data(val)
    elif typ in ['data-reshape']:   # Also works for imshow
        f = lambda val, obj=obj, n12=n12: obj.set_data(val.reshape(n12[1],n12[0]))
    elif typ in ['alpha']:   # Also works for imshow
        f = lambda val, obj=obj, norm=norm: obj.set_alpha(norm(val))
    elif typ == 'txt':
        f = lambda val, obj=obj, bstr=bstr: obj.set_text(bstr.format(val))
    return f


def get_ind_frompos(Type='x', ref=None, ref2=None, otherid=None, indother=None):
    assert Type in ['x','y','2d']

    if Type in ['x','y']:
        if otherid is None:
            assert ref.size == np.max(ref.shape)
            ref = ref.ravel()
            refb = 0.5*(ref[1:]+ref[:-1])
            if Type == 'x':
                def func(val, ind0=None, refb=refb):
                    return np.digitize([val[0]], refb)[0]
            else:
                def func(val, ind0=None, refb=refb):
                    return np.digitize([val[1]], refb)[0]
        elif indother is None:
            assert ref.ndim == 2
            if Type == 'x':
                def func(val, ind0=None, ref=ref):
                    refb = 0.5*(ref[ind0,1:]+ref[ind0,:-1])
                    return np.digitize([val[0]], refb)[0]
            else:
                def func(val, ind0=None, ref=ref):
                    refb = 0.5*(ref[ind0,1:]+ref[ind0,:-1])
                    return np.digitize([val[1]], refb)[0]
        else:
            assert ref.ndim == 2
            if Type == 'x':
                def func(val, ind0=None, ref=ref, indother=indother):
                    refb = 0.5*(ref[indother[ind0],1:]+ref[indother[ind0],:-1])
                    return np.digitize([val[0]], refb)[0]
            else:
                def func(val, ind0=None, ref=ref, indother=indother):
                    refb = 0.5*(ref[indother[ind0],1:]+ref[indother[ind0],:-1])
                    return np.digitize([val[1]], refb)[0]
    else:
        assert type(ref2) is tuple and len(ref2) == 2
        n1, n2 = ref2[0].size, ref2[1].size
        refb1 = 0.5*(ref2[0][1:]+ref2[0][:-1])
        refb2 = 0.5*(ref2[1][1:]+ref2[1][:-1])
        def func(val, ind0=None, refb1=refb1, refb2=refb2, n1=n1, n2=n2):
            i1 = np.digitize([val[0]], refb1)[0]
            i2 = np.digitize([val[1]], refb2)[0]
            return i2*n1 + i1
    return func

def get_pos_fromind(ref=None, ref2=None, otherid=None, indother=None):
    if ref2 is not None:
        assert type(ref2) is tuple and len(ref2) == 2
        n1, n2 = ref2[0].size, ref2[1].size
        def func(ind, ind0=None, ref2=ref2, n1=n1, n2=n2):
            i1 = ind % n1
            i2 = ind // n1
            return (ref2[0][i1], ref2[1][i2])

    else:
        if otherid is None:
            assert ref.size == np.max(ref.shape)
            ref = ref.ravel()
            def func(ind, ind0=None, ref=ref):
                val = ref[ind]
                return (val, val)
        elif indother is None:
            assert ref.ndim == 2
            def func(ind, ind0, ref=ref):
                val = ref[ind0,ind]
                return (val, val)
        else:
            assert ref.ndim == 2
            def func(ind, ind0, ref=ref, indother=indother):
                val = ref[indother[ind0],ind]
                return (val, val)
    return func

def get_ind_fromkey(dmovkeys={}, nn=[], is2d=False):
    if is2d:
        n1, n2 = nn
        def func(movk, ind, doinc=False, dmovkeys=dmovkeys, n1=n1, n2=n2):
            i1 = ind % n1
            i2 = ind // n1
            if movk in ['left','right']:
                i1 += dmovkeys[movk][doinc]
                i1 = i1 % n1
            else:
                i2 += dmovkeys[movk][doinc]
                i2 = i2 % n2
            return i2 * n1 + i1
    else:
        nx = nn[0] if len(nn)==1 else nn[1]
        def func(movk, ind, doinc=False, dmovkeys=dmovkeys, nx=nx):
            ind += dmovkeys[movk][doinc]
            ind = ind % nx
            return ind
    return func




class KeyHandler_mpl(object):

    _msgdobj = """ Arg dobj must be a dict
    The keys are handles to matplotlib Artist objects (e.g.: lines, imshow...)

    For each object oo, dobj[oo] is itself a dict with (key,value) pairs:
        - 'ref'   : a 1D flat np.ndarray, used as reference
        - 'lgroup': list of int, indicating to which groups of indices oo belongs
        - 'lind'  : list of int, indicating the index of oo in each group
        - 'update': a callable (function), to be called when updating
    """

    _ltypesref = ['x','y','2d']


    def __init__(self, can=None, dgroup=None, dref=None, ddata=None,
                 dobj=None, dax=None, lax_fix=[],
                 groupinit='time', follow=True):
        assert issubclass(can.__class__, mpl.backend_bases.FigureCanvasBase)
        self.can = can
        out = self._checkformat_dgrouprefaxobj(dgroup, dref, dobj, dax, ddata)
        dgroup, dref, dax, dobj, dind, ddata = out
        self.dgroup = dgroup
        self.dref = dref
        self.ddata = ddata
        self.dax = dax
        self.dind = dind
        self.dax.update(dict([(ax,{'fix':None, 'lobj':[]}) for ax in lax_fix
                              if ax not in dax.keys()]))
        self.dobj = dobj
        self.dcur = self._get_dcur_init(dgroup, dax, groupinit)
        dkeys = self._get_dkeys()
        lact = set([v['action'] for v in dkeys.values()])
        dkeys_r = dict([(aa, [k for k in dkeys.keys() if dkeys[k]['action']==aa])
                             for aa in lact])
        self.dkeys = dkeys
        self.dkeys_r = dkeys_r
        #self.init(dgroup=dgroup, dobj=dobj)
        self.isinteractive = hasattr(can,'toolbar') and can.toolbar is not None
        self._follow = follow

        self._set_dBck(self.dax.keys())

    @staticmethod
    def _get_dcur_init(dgroup, dax, groupinit='time'):
        if not groupinit in dgroup.keys():
            msg = "Arg groupinit must be a valid key of dgroup !\n"
            msg += "    - Valid keys: %s\n"%str(dgroup.keys())
            msg += "    - Provided  : %s"%groupinit
            raise Exception(msg)
        refid = dgroup[groupinit]['defid']
        ax = dgroup[groupinit]['defax']
        dcur = {'refid':refid, 'group':groupinit, 'ax':ax}
        return dcur

    def _warn_ifnotInteractive(self):
        warn = False
        if not self.isinteractive:
            msg = "Not interactive backend!:\n"
            msg += "    - backend : %s   (prefer Qt5Agg)\n"%plt.get_backend()
            msg += "    - canvas  : %s"%self.can.__class__.__name__
            warnings.warn(msg)
            warn = True
        return warn

    @classmethod
    def _get_dmovkeys(cls, Type, inc, invert=False):
        assert Type in cls._ltypesref
        if Type[0] == 'x':
            dmovkeys = {'left':{False:-inc[0], True:-inc[1]},
                        'right':{False:inc[0], True:inc[1]}}
        elif Type[0] == 'y':
            dmovkeys = {'down':{False:-inc[0], True:-inc[1]},
                        'up':{False:inc[0], True:inc[1]}}
        elif Type == '2d':
            sig = -1 if invert else 1
            dmovkeys = {'left':{False:-sig*inc[0], True:-sig*inc[1]},
                        'right':{False:sig*inc[0], True:sig*inc[1]},
                        'down':{False:-sig*inc[0], True:-sig*inc[1]},
                        'up':{False:sig*inc[0], True:sig*inc[1]}}
        return dmovkeys

    @classmethod
    def _checkformat_dgrouprefaxobj(cls, dgroup, dref, dobj, dax, ddata):
        assert all([type(dd) is dict for dd in [dgroup, dref, dobj, dax]])

        #---------------
        # Preliminary checks

        ls = ['nMax','key','defax']
        for k,v in dgroup.items():
            c0 = type(k) is str
            c1 = type(v) is dict
            c2 = all([s in v.keys() for s in ls])
            if not (c0 and c1 and c2):
                raise Exception(cls._msgdobj)
            assert type(v['nMax']) in [int,np.int64]
            assert type(v['key']) is str
            assert v['defax'] in dax.keys()
        lg = sorted(list(dgroup.keys()))
        assert len(set(lg))==len(lg)

        ls = ['val','refids']
        for k,v in ddata.items():
            c0 = type(k) is int
            c1 = type(v) is dict
            c2 = all([s in v.keys() for s in ls])
            if not (c0 and c1 and c2):
                raise Exception(cls._msgdobj)
            assert all([vv in dref.keys() for vv in v['refids']])
        ldid = sorted(list(ddata.keys()))
        assert len(set(ldid))==len(ldid)

        ls = ['group','val','inc']
        for k,v in dref.items():
            c0 = type(k) is int
            c1 = type(v) is dict
            c2 = all([s in v.keys() for s in ls])
            if not (c0 and c1 and c2):
                raise Exception(cls._msgdobj)
            assert v['group'] in lg
            assert type(v['val']) in [np.ndarray,tuple]
            assert len(v['inc']) == 2
            v['inc'] = np.asarray(v['inc'],dtype=int).ravel()
        lrid = sorted(list(dref.keys()))
        lr = [dref[rid]['val'] for rid in lrid]
        assert len(set(lrid))==len(lrid)

        ls = cls._ltypesref
        for k,v in dax.items():
            c0 = issubclass(k.__class__, mpl.axes.Axes)
            c1 = type(v) is dict
            c2 = 'ref' in v.keys() and type(v['ref']) is dict
            c3 = all([vv in ls for vv in v['ref'].values()])
            c4 = all([kk in lrid for kk in v['ref'].keys()])
            lc = [c0,c1,c2,c3,c4]
            assert all(lc), str(lc)
            if 'defrefid' in v.keys():
                assert v['defrefid'] in lrid
        la = list(dax.keys())
        assert len(set(la))==len(la)

        ls = ['dupdate','drefid']
        for k,v in dobj.items():
            c0 = issubclass(k.__class__, mpl.artist.Artist)
            c1 = type(v) is dict
            c2 = any([s in v.keys() for s in ls])
            assert (c0 and c1 and c2)
            assert all([vv in lrid for vv in v['drefid'].keys()])
            dobj[k]['ax'] = k.axes
            if dobj[k]['ax'] not in dax.keys():
                dax[dobj[k]['ax']] = {'fix':None, 'ref':{}}
        lo = list(dobj.keys())
        assert len(set(lo))==len(lo)

        #---------------
        # Complement

        # dax
        for ax in dax.keys():
            lobj = [obj for obj in dobj.keys() if dobj[obj]['ax'] is ax]
            dax[ax]['lobj'] = lobj
            if 'invert' in dax[ax].keys():
                assert type(dax[ax]['invert']) is bool
                assert '2d' in dax[ax]['ref'].values()
            else:
                dax[ax]['invert'] = False

            # Handle cases were several ref are associated to the same typ
            lrefid = list(dax[ax]['ref'].keys())
            ltypu = sorted(set(dax[ax]['ref'].values()))
            dtypu = dict([(typ,[rid for rid in lrefid
                                if dax[ax]['ref'][rid] == typ])
                         for typ in ltypu])
            # Check unicity of graphical ref
            if 'graph' not in dax[ax].keys():
                assert all([len(vv) == 1 for vv in dtypu.values()])
                dax[ax]['graph'] = dax[ax]['ref']
            else:
                for typ in dtypu.keys():
                    if len(dtypu[typ]) > 1:
                        assert np.sum([rid in dax[ax]['graph'].keys()
                                       for rid in dtypu[typ]]) == 1
                    else:
                        dax[ax]['graph'][dtypu[typ][0]] = typ

            # func dict
            dmovkeys = {}
            for rid in lrefid:
                dmovkeys[rid] = cls._get_dmovkeys(dax[ax]['ref'][rid],
                                                  dref[rid]['inc'],
                                                  invert=dax[ax]['invert'])
            dax[ax]['dmovkeys'] = dmovkeys

            # Find default refid for ax (in case of ref from several groups)
            lrefid = list(dax[ax]['graph'].keys())
            if len(lrefid) > 1:
                assert 'defrefid' in dax[ax].keys()
                assert dax[ax]['defrefid'] in lrefid


        # dgroup
        for g in lg:
            dgroup[g]['indcur'] = 0
            dgroup[g]['ncur'] = 0
            dgroup[g]['valind'] = np.full((dgroup[g]['nMax'],2), np.nan)
            lridg = [rid for rid in lrid if dref[rid]['group']==g]
            dgroup[g]['lrefid'] = lridg
            # To catch cross-dependencies:
            # All axes of all objects with a group dependency
            lla = []
            for o in lo:
                if any([dref[rid]['group']==g
                        for rid in dobj[o]['drefid'].keys()]):
                    if dobj[o]['ax'] not in lla:
                        lla.append(dobj[o]['ax'])
            dgroup[g]['lax'] = lla

            # Set defid
            ldefid = dax[dgroup[g]['defax']]['graph'].keys()
            ldefid = [defid for defid in ldefid if dref[defid]['group'] == g]
            assert len(ldefid) == 1
            dgroup[g]['defid'] = ldefid[0]

            # Get list of obj with their indices, for fast updates
            lobj = [obj for obj in dobj.keys()
                    if any([dref[rid]['group'] == g
                            for rid in dobj[obj]['drefid'].keys()])]
            nobj = len(lobj)
            lind = []
            for obj in lobj:
                ii = [ii for rid, ii in dobj[obj]['drefid'].items()
                      if dref[rid]['group'] == g]
                assert len(ii) >= 1
                if len(ii)>1:
                    # Case when same group refered to via several refs
                    # Only works if consistent indices !
                    assert all([iii==ii[0] for iii in ii])
                    ii = [ii[0]]
                lind += ii
            lindu = np.unique(lind)
            assert np.all(lindu >= 0) and np.all(lindu < dgroup[g]['nMax'])
            d2obj = dict([(ind, [lobj[ii] for ii in range(0,nobj)
                                if lind[ii] == ind]) for ind in lindu])
            dgroup[g]['d2obj'] = d2obj
            dgroup[g]['lobj'] = lobj


        # dref
        for rid in lrid:
            dref[rid]['ind'] = np.zeros((dgroup[dref[rid]['group']]['nMax'],),
                                        dtype=int)

            # Check consistency of otherid and indother
            c0 = 'indother' not in dref[rid].keys()
            c1 = not c0 and dref[rid]['indother'] is None
            if 'otherid' not in dref[rid].keys():
                if isinstance(dref[rid]['val'], np.ndarray):
                    assert dref[rid]['val'].size == max(dref[rid]['val'].shape)
                elif isinstance(dref[rid]['val'], tuple):
                    assert len(dref[rid]['val'])==2
                assert c0 or c1
                dref[rid]['otherid'] = None
                if c0:
                    dref[rid]['indother'] = None
            else:
                otherid = dref[rid]['otherid']
                assert otherid is None or otherid in dref.keys()
                if otherid is None:
                    assert c0 or c1
                    if c0:
                        dref[rid]['indother'] = None
                else:
                    assert dref[rid]['val'].ndim == 2
                    if c0 or c1:
                        assert dref[rid]['val'].shape[0]==dref[otherid]['val'].size
                        if c0:
                            dref[rid]['indother'] = None
                    else:
                        assert dref[rid]['indother'].ndim == 1

            # Get nn
            val = dref[rid]['val']

            # Check if is2d
            ltypes = []
            for ax in dax.keys():
                if rid in dax[ax]['ref'].keys():
                    ltypes.append(dax[ax]['ref'][rid])
            ltypes = sorted(set(ltypes))
            assert ltypes == ['2d'] or not '2d' in ltypes
            is2d = '2d' in ltypes
            dref[rid]['is2d'] = is2d

            # Get functions
            otherid = dref[rid]['otherid']
            indother = dref[rid]['indother']
            df_ind_pos, df_ind_key, df_pos_ind = {}, {}, {}
            for ax in dax.keys():
                if rid in dax[ax]['ref'].keys():
                    typ = dax[ax]['ref'][rid]
                    if typ == '2d':
                        assert '2d' in dref[rid].keys()
                        val2 = dref[rid]['2d']
                        is2d = True
                        nn = (val2[0].size, val2[1].size)
                    else:
                        val2 = None
                        is2d = False
                        if isinstance(val, np.ndarray):
                            nn = val.shape
                        elif isinstance(val, list):
                            nn = (len(val),)
                        else:
                            raise Exception("Unknown val type !")
                    df_ind_pos[ax] = get_ind_frompos(typ, val, val2,
                                                     otherid = otherid,
                                                     indother = indother)
                    dmovkeys = dax[ax]['dmovkeys'][rid]
                    df_ind_key[ax] = get_ind_fromkey(dmovkeys,
                                                     is2d = is2d,
                                                     nn = nn)
                    df_pos_ind[ax] = get_pos_fromind(val, val2,
                                                     otherid = otherid,
                                                     indother = indother)

            dref[rid]['df_ind_pos'] = df_ind_pos
            dref[rid]['df_ind_key'] = df_ind_key
            dref[rid]['df_pos_ind'] = df_pos_ind

            # lobj
            lobj = [oo for oo in dobj.keys()
                    if rid in dobj[oo]['drefid'].keys()]
            dref[rid]['lobj'] = lobj

        # dind
        lref = list(dref.keys())
        nref = len(lref)
        ancur = np.zeros((2,nref),dtype=int)
        ancur[1,:] = [dgroup[dref[rid]['group']]['nMax'] for rid in lref]
        cumsum0 = np.r_[0, np.cumsum(ancur[1,:])]
        arefind = np.zeros((np.sum(ancur[1,:]),),dtype=int)
        dind = {'lrefid':lref, 'anMaxcur':ancur, 'arefind':arefind,
                'cumsum0':cumsum0}

        # dobj
        for oo in dobj.keys():
            dobj[oo]['vis'] = False

            # get functions to update
            lrefid = list(dobj[oo]['drefid'].keys())
            for k, v in dobj[oo]['dupdate'].items():
                # Check consistency with ddata
                if v['id'] not in ddata.keys():
                    if not v['id'] in dref.keys():
                        msg = "Missing id in ddata or dref "
                        msg += "(vs dobj[idobj]['dupdate'][k]['id']) !\n"
                        msg += "    idobj: %s\n"%oo
                        msg += "    k    : %s\n"%k
                        msg += "    id   : %s"%v['id']
                        raise Exception(msg)

                    ddata[v['id']] = {'val':dref[v['id']]['val']}
                    if dref[v['id']]['otherid'] is None:
                        ddata[v['id']]['refids'] = [v['id']]
                    else:
                        ddata[v['id']]['refids'] = [dref[v['id']]['otherid'],v['id']]

                # get update tools
                val = ddata[v['id']]['val']
                lrefs = ddata[v['id']]['refids']
                linds = v['lrid']
                fgetval = get_valf(val, lrefs, linds)

                lkv = [(kk,vv) for kk,vv in dobj[oo]['dupdate'][k].items()
                       if kk not in ['id','lrid']]
                fupdate = get_fupdate(oo, k, **dict(lkv))
                dobj[oo]['dupdate'][k]['fgetval'] = fgetval
                dobj[oo]['dupdate'][k]['fupdate'] = fupdate
                indrefind = get_indrefind(dind, linds, dobj[oo]['drefid'])
                dobj[oo]['dupdate'][k]['indrefind'] = indrefind

            # linds not necessarily identical !
            dobj[oo]['aindvis'] = np.array([dobj[oo]['drefid'][rid]
                                            for rid in lrefid], dtype=int)
            indncurind = get_indncurind(dind, lrefid)
            dobj[oo]['indncurind'] = indncurind

        return dgroup, dref, dax, dobj, dind, ddata


    def _get_dkeys(self):
        dkeys = {'shift':{'val':False, 'action':'generic'},
                 'control':{'val':False, 'action':'generic'},
                 'ctrl':{'val':False, 'action':'generic'},
                 'alt':{'val':False, 'action':'generic'},
                 'left':{'val':False, 'action':'move'},
                 'right':{'val':False, 'action':'move'},
                 'up':{'val':False, 'action':'move'},
                 'down':{'val':False, 'action':'move'}}
                 # 'pageup':{'val':False, 'action':'move'},
                 # 'pagedown':{'val':False, 'action':'move'}}
        dkeys.update(dict([(v['key'],{'group':k, 'val':False, 'action':'group'})
                           for k,v in self.dgroup.items()]))
        nMax = np.max([v['nMax'] for v in self.dgroup.values()])
        dkeys.update(dict([(str(ii),{'ind':ii, 'val':False, 'action':'indices'})
                          for ii in range(0,nMax)]))
        return dkeys

    def disconnect_old(self, force=False):
        if self._warn_ifnotInteractive():
            return
        if force:
            self.can.mpl_disconnect(self.can.manager.key_press_handler_id)
        else:
            lk = [kk for kk in list(plt.rcParams.keys()) if 'keymap' in kk]
            self.store_rcParams = {}
            for kd in self.dkeys.keys():
                self.store_rcParams[kd] = []
                for kk in lk:
                    if kd in plt.rcParams[kk]:
                        self.store_rcParams[kd].append(kk)
                        plt.rcParams[kk].remove(kd)
        self.can.mpl_disconnect(self.can.button_pick_id)

    def reconnect_old(self):
        if self._warn_ifnotInteractive():
            return
        if self.store_rcParams is not None:
            for kd in self.store_rcParams.keys():
                for kk in self.store_rcParams[kk]:
                    if kd not in plt.rcParams[kk]:
                        plt.rcParams[kk].append(kd)

    def connect(self):
        if self._warn_ifnotInteractive():
            return
        keyp = self.can.mpl_connect('key_press_event', self.onkeypress)
        keyr = self.can.mpl_connect('key_release_event', self.onkeypress)
        butp = self.can.mpl_connect('button_press_event', self.mouseclic)
        res = self.can.mpl_connect('resize_event', self.resize)
        #butr = self.can.mpl_connect('button_release_event', self.mouserelease)
        #if not plt.get_backend() == "agg":
        self.can.manager.toolbar.release = self.mouserelease

        self._cid = {'keyp':keyp, 'keyr':keyr, 'butp':butp, 'res':res}#, 'butr':butr}

    def disconnect(self):
        if self._warn_ifnotInteractive():
            return
        for kk in self._cid.keys():
            self.can.mpl_disconnect(self._cid[kk])
        self.can.manager.toolbar.release = lambda event: None


    def resize(self, event):
        self._set_dBck(self.dax.keys())

    def _set_dBck(self, lax):
        # Make all invisible
        for ax in lax:
            for obj in self.dax[ax]['lobj']:
                obj.set_visible(False)

        # Draw and reset Bck
        self.can.draw()
        for ax in lax:
            #ax.draw(self.can.renderer)
            self.dax[ax]['Bck'] = self.can.copy_from_bbox(ax.bbox)

        # Redraw
        for ax in lax:
            for obj in self.dax[ax]['lobj']:
                obj.set_visible(self.dobj[obj]['vis'])
                #ax.draw(self.can.renderer)
        self.can.draw()

    def init(self, dgroup=None, ngroup=None, dobj=None):
        pass

    def update(self, excluderef=True):
        self._update_dcur() # 0.4 ms
        self._update_dref(excluderef=excluderef)    # 0.1 ms
        self._update_dobj() # 0.2 s

    def _update_dcur(self):
        group = self.dcur['group']
        refid = self.dcur['refid']
        assert self.dref[refid]['group'] == group
        assert refid in self.dax[self.dcur['ax']]['graph'].keys()

        # Update also dind !
        an = [self.dgroup[self.dref[rid]['group']]['ncur']
              for rid in self.dind['lrefid']]
        self.dind['anMaxcur'][0,:] = an

        # Update array ncur
        for obj in self.dgroup[group]['lobj']:
            a0 = self.dind['anMaxcur'][0,self.dobj[obj]['indncurind']]
            a1 = self.dobj[obj]['aindvis']
            self.dobj[obj]['vis'] = np.all( a0 >= a1 )


    def _update_dref(self, excluderef=True):
        group = self.dcur['group']
        ind = self.dgroup[group]['indcur']
        val = self.dgroup[group]['valind'][ind,:]

        if excluderef and len(self.dgroup[group]['lrefid'])>1:
            for rid in self.dgroup[group]['lrefid']:
                if rid == self.dcur['refid']:
                    continue
                if self.dref[rid]['otherid'] is None:
                    indother = None
                else:
                    group2 = self.dref[self.dref[rid]['otherid']]['group']
                    ind2 = self.dgroup[group2]['indcur']
                    indother = self.dref[self.dref[rid]['otherid']]['ind'][ind2]
                lax = list(self.dref[rid]['df_ind_pos'].keys())
                if len(lax) == 0:
                    msg = "A ref has no associated ax !\n"
                    msg += "    - group: %s\n"%group
                    msg += "    - rid  : %s"%rid
                    raise Exception(msg)

                ii = self.dref[rid]['df_ind_pos'][lax[0]](val, indother)
                if self._follow:
                    self.dref[rid]['ind'][ind:] = ii
                else:
                    self.dref[rid]['ind'][ind] = ii
        else:
            for rid in self.dgroup[group]['lrefid']:
                if self.dref[rid]['otherid'] is None:
                    indother = None
                else:
                    group2 = self.dref[self.dref[rid]['otherid']]['group']
                    ind2 = self.dgroup[group2]['indcur']
                    indother = self.dref[self.dref[rid]['otherid']]['ind'][ind2]
                lax = list(self.dref[rid]['df_ind_pos'].keys())
                if len(lax) == 0:
                    msg = "A ref has no associated ax !\n"
                    msg += "    - group: %s\n"%group
                    msg += "    - rid  : %s"%rid
                    raise Exception(msg)

                ii = self.dref[rid]['df_ind_pos'][lax[0]](val, indother)
                if self._follow:
                    self.dref[rid]['ind'][ind:] = ii
                else:
                    self.dref[rid]['ind'][ind] = ii

        # Update dind['arefind']
        for ii in range(0,len(self.dind['lrefid'])):
            rid = self.dind['lrefid'][ii]
            i0 = self.dind['cumsum0'][ii]
            i1 = i0 + self.dgroup[self.dref[rid]['group']]['nMax']
            self.dind['arefind'][i0:i1] = self.dref[rid]['ind']


    def _update_dobj(self):

        # --- Prepare ----- 2 us
        group = self.dcur['group']
        refid = self.dcur['refid']
        indcur = self.dgroup[group]['indcur']
        lax = self.dgroup[group]['lax']

        # ---- Restore backgrounds ---- 1 ms
        for aa in lax:
            self.can.restore_region(self.dax[aa]['Bck'])

        # ---- update data of group objects ----  0.15 s
        for obj in self.dgroup[group]['d2obj'][indcur]:
            for k in self.dobj[obj]['dupdate'].keys():
                ii = self.dobj[obj]['dupdate'][k]['indrefind']  # 20 us
                li = self.dind['arefind'][ii]   # 50 us
                val = self.dobj[obj]['dupdate'][k]['fgetval']( li )    # 0.0001 s
                self.dobj[obj]['dupdate'][k]['fupdate']( val )  # 2 ms

        # --- Redraw all objects (due to background restore) --- 25 ms
        for obj in self.dobj.keys():
            obj.set_visible(self.dobj[obj]['vis'])
            self.dobj[obj]['ax'].draw_artist(obj)

        # ---- blit axes ------ 5 ms
        for aa in lax:
            self.can.blit(aa.bbox)

    def mouseclic(self,event):

        # Check click is relevant
        C0 = event.inaxes is not None and event.button == 1
        if not C0:
            return

        self.curax_panzoom = event.inaxes   # DB ?

        # Check axes is relevant and toolbar not active
        c_activeax = 'fix' not in self.dax[event.inaxes].keys()
        c_toolbar = self.can.manager.toolbar._active in [None,False]
        if not all([c_activeax,c_toolbar]):
            return

        # Set self.dcur
        self.dcur['ax'] = event.inaxes
        lrid = list(self.dax[event.inaxes]['graph'].keys())
        if len(lrid)>1:
            lg = [self.dref[rid]['group'] for rid in lrid]
            if self.dcur['group'] in lg:
                rid = lrid[lg.index(self.dcur['group'])]
            else:
                rid = self.dax[event.inaxes]['defrefid']
        else:
            rid= lrid[0]
        self.dcur['refid'] = rid
        self.dcur['group'] = self.dref[self.dcur['refid']]['group']

        group = self.dcur['group']
        ax = self.dcur['ax']
        refid = self.dcur['refid']

        # Check max number of occurences not reached if shift
        c0 = (self.dkeys['shift']['val']
              and self.dgroup[group]['indcur'] == self.dgroup[group]['nMax']-1)
        if c0:
            msg = "Max nb. of plots reached ({0}) for group {1}"
            msg  = msg.format(self.dgroup[group]['nMax'], group)
            print(msg)
            return

        # Update indcur
        ctrl = self.dkeys['control']['val'] or self.dkeys['ctrl']['val']
        if ctrl:
            nn = 0
            ii = 0
        elif self.dkeys['shift']['val']:
            nn = int(self.dgroup[group]['ncur'])+1
            ii = nn
        else:
            nn = int(self.dgroup[group]['ncur'])
            ii = int(self.dgroup[group]['indcur'])

        # Update dcur
        self.dgroup[group]['ncur'] = nn
        self.dgroup[group]['indcur'] = ii

        # Update group val
        val = (event.xdata, event.ydata)
        if self._follow:
            self.dgroup[group]['valind'][ii:,:] = val
        else:
            self.dgroup[group]['valind'][ii,:] = val

        self.update(excluderef=False)


    def mouserelease(self, event):
        msg = "Make sure you release the mouse button on an axes !"
        msg += "\n Otherwise the background plot cannot be properly updated !"
        c0 = self.can.manager.toolbar._active == 'PAN'
        c1 = self.can.manager.toolbar._active == 'ZOOM'

        if c0 or c1:
            ax = self.curax_panzoom
            assert ax is not None, msg
            lax = ax.get_shared_x_axes().get_siblings(ax)
            lax += ax.get_shared_y_axes().get_siblings(ax)
            lax = list(set(lax))
            self._set_dBck(lax)


    def onkeypress(self, event):

        lkey = event.key.split('+')

        c0 = self.can.manager.toolbar._active is not None
        c1 = len(lkey) not in [1,2]
        c2 = [ss not in self.dkeys.keys() for ss in lkey]
        if c0 or c1 or any(c2):
            return

        lgen = [k for k in self.dkeys_r['generic'] if k in lkey]
        lmov = [k for k in self.dkeys_r['move'] if k in lkey]
        lgrp = [k for k in self.dkeys_r['group'] if k in lkey]
        lind = [k for k in self.dkeys_r['indices'] if k in lkey]

        ngen, nmov, ngrp, nind = len(lgen), len(lmov), len(lgrp), len(lind)
        ln = np.r_[ngen, nmov, ngrp, nind]
        if np.any(ln>1) or np.sum(ln)>2:
            return
        if np.sum(ln)==2 and (ngrp==1 or nind==1):
            return

        genk = None if ngen == 0 else lgen[0]
        movk = None if nmov == 0 else lmov[0]
        grpk = None if ngrp == 0 else lgrp[0]
        indk = None if nind == 0 else lind[0]


        if event.name == 'key_release_event':
            if event.key == genk:
                self.dkeys[genk]['val'] = False
            return

        if genk is not None and event.key == genk:
            self.dkeys[genk]['val'] = True
            return

        if grpk is not None:
            group = self.dkeys[event.key]['group']
            self.dcur['group'] = group
            self.dcur['refid'] = self.dgroup[group]['defid']
            self.dcur['ax'] = self.dgroup[group]['defax']
            return

        if indk is not None:
            group = self.dcur['group']
            indmax = self.dgroup[group]['ncur']
            ii = int(event.key)
            if ii > indmax:
                msg = "Maximum allowed index for group {0} is {1}, set to it"
                msg = msg.format(group, indmax)
                print(msg)
            ii = min(ii,indmax)
            self.dgroup[group]['indcur'] = ii
            return

        if movk is not None:
            group = self.dcur['group']
            refid = self.dcur['refid']
            ax = self.dcur['ax']

            # # Debug
            # if refid not in self.dax[ax]['dmovkeys'].keys():    # DB
                # print(refid, self.dref[refid]['group'])  # DB
                # print(ax, self.dax[ax]['dmovkeys']) # DB

            if movk not in self.dax[ax]['dmovkeys'][refid].keys():
                return

            # Check max number of occurences not reached if shift
            c0 = (self.dkeys['shift']['val']
                  and self.dgroup[group]['indcur'] == self.dgroup[group]['nMax']-1)
            if c0:
                msg = "Max nb. of plots reached ({0}) for group {1}"
                msg  = msg.format(self.dgroup[group]['nMax'], group)
                print(msg)
                return

            ctrl = self.dkeys['control']['val'] or self.dkeys['ctrl']['val']
            if ctrl:
                nn = 0
                ii = 0
            elif self.dkeys['shift']['val']:
                nn = int(self.dgroup[group]['ncur'])+1
                ii = nn
            else:
                nn = int(self.dgroup[group]['ncur'])
                ii = int(self.dgroup[group]['indcur'])

            # Update dcur
            self.dgroup[group]['ncur'] = nn
            self.dgroup[group]['indcur'] = ii

            # Update refid ind
            ind = self.dref[refid]['df_ind_key'][ax](movk,
                                                     self.dref[refid]['ind'][ii],
                                                     self.dkeys['alt']['val'])
            if self._follow:
                self.dref[refid]['ind'][ii:] = ind
            else:
                self.dref[refid]['ind'][ii] = ind

            # Update group val
            if self.dref[refid]['otherid'] is None:
                indother = None
            else:
                group2 = self.dref[self.dref[refid]['otherid']]['group']
                ind2 = self.dgroup[group2]['indcur']
                indother = self.dref[self.dref[refid]['otherid']]['ind'][ind2]
            val = self.dref[refid]['df_pos_ind'][ax]( ind, indother )
            if self._follow:
                self.dgroup[group]['valind'][ii:,:] = val
            else:
                self.dgroup[group]['valind'][ii,:] = val

            # Upadte all
            self.update(excluderef=True)





###################################################################
###################################################################
#       Start new abstract Data2DBase with indices for next version
###################################################################

class Data2DBase(ToFuObjectBase):
    """ Provide a a dict of sequences depending on 2 indices
    """
    pass
