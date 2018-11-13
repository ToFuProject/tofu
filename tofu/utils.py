
# Built-in
import os
import collections
from abc import ABCMeta, abstractmethod
import getpass

# Common
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# tofu-specific
from tofu import __version__
import tofu.pathfile as tfpf

_sep = '_'
_dict_lexcept_key = []

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
        C = type(name) is str and not (name[-4]=='.')
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
#       todict formatting
#############################################


def flatten_dict(d, parent_key='', sep=_sep, rec=False,
                 lexcept_key=_dict_lexcept_key):

    items = []
    if rec:
        if lexcept_key is None:
            for k, v in d.items():
                new_key = parent_key + sep + k if parent_key else k
                if issubclass(v.__class__, ToFuObjectBase):
                    v = v.to_dict()
                if isinstance(v, collections.MutableMapping):
                    items.extend(flatten_dict(v, new_key,
                                              rec=rec, sep=sep).items())
                else:
                    items.append((new_key, v))
        else:
            for k, v in d.items():
                if k not in lexcept_key:
                    if issubclass(v.__class__, ToFuObjectBase):
                        v = v.to_dict()
                    new_key = parent_key + sep + k if parent_key else k
                    if isinstance(v, collections.MutableMapping):
                        items.extend(flatten_dict(v, new_key,
                                                  rec=rec, sep=sep).items())
                    else:
                        items.append((new_key, v))
    else:
        if lexcept_key is None:
            for k, v in d.items():
                new_key = parent_key + sep + k if parent_key else k
                if isinstance(v, collections.MutableMapping):
                    items.extend(flatten_dict(v, new_key,
                                              rec=rec, sep=sep).items())
                else:
                    items.append((new_key, v))
        else:
            for k, v in d.items():
                if k not in lexcept_key:
                    new_key = parent_key + sep + k if parent_key else k
                    if isinstance(v, collections.MutableMapping):
                        items.extend(flatten_dict(v, new_key,
                                                  rec=rec, sep=sep).items())
                    else:
                        items.append((new_key, v))
    return dict(items)

def _reshape_dict(ss, vv, dinit={}, sep=_sep):
    ls = ss.split(sep)
    k = ss if len(ls)==1 else ls[0]
    if len(ls)==2:
        dk = {ls[1]:vv}
        if k not in dinit.keys():
            dinit[k] = {}
        assert isinstance(dinit[k],dict)
        dinit[k].update({ls[1]:vv})
    elif len(ls)>2:
        if k not in dinit.keys():
            dinit[k] = {}
        _reshape_dict(sep.join(ls[1:]), vv, dinit=dinit[k], sep=sep)
    else:
        assert k not in dinit.keys()
        dinit[k] = vv

def reshape_dict(d, sep=_sep):
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

class dictattr(dict):
    __getattr__ = dict.__getitem__

    def __init__(self, extra, *args, **kwdargs):
        super().__init__(*args, **kwdargs)
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

def save(obj, path=None, name=None, mode='npz',
         strip=None, compressed=False, verb=True):
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
    dd = obj.to_dict(strip=strip)

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
        elif type(dd[k]) in [int,float,np.int64,np.float64,bool,str]:
            dd[k] = np.asarray([dd[k]])
        elif type(dd[k]) in [list,tuple]:
            dd[k] = np.asarray(dd[k])
        elif not isinstance(dd[k],np.ndarray):
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




def load(name, path=None, strip=None, verb=True):
    """

    """
    msg = "Arg name must be a str (file name or full path+file)"
    msg += " or a list of str patterns to be found at path"
    C0 = isinstance(name,str)
    C1 = isinstance(name,list) and all([isinstance(ss,str) for ss in name])
    assert C0 or C1, msg
    msg = "Arg path must be a str !"
    assert path is None or isinstance(path,str), msg

    # Extract folder and file name
    if isinstance(name,str):
        p, f = os.path.split(name)
        name = [f]
        if p!='':
            path = p
        elif path is None:
            path = './'
    path = os.path.normpath(os.path.abspath(path))
    msg = "Specified folder does not exist :"
    msg += "\n    {0}".format(path)
    assert os.path.isdir(path), msg

    # Check unicity of matching file
    lf = os.listdir(path)
    lf = [ff for ff in lf if all([ss in ff for ss in name])]
    if len(lf)!=1:
        msg = "No / several matching files found:"
        msg += "\n  folder: {0}".format(path)
        msg += "\n  for   : {0}".format('['+', '.join(name)+']')
        msg += "\n    " + "\n    ".join(lf)
        raise Exception(msg)
    name = lf[0]

    # Check file extension
    lmodes = ['.npz','.mat']
    msg = "None / too many of the available file extensions !"
    msg += "\n  file: {0}".format(name)
    msg += "\n  ext.: {0}:".format('['+', '.format(lmodes)+']')
    indend = [ss==name[-4:] for ss in lmodes]
    indin = [ss in name for ss in lmodes]
    assert np.sum(indend)==1 and np.sum(indin)==1, msg

    # load and format dict
    mode = lmodes[np.argmax(indend)].replace('.','')
    pathfileext = os.path.join(path,name)
    if mode=='npz':
        dd = _load_npz(pathfileext)
    elif mode=='npz':
        dd = _load_mat(pathfileext)

    # Recreate from dict
    exec("import tofu.{0} as mod".format(dd['dId_dall_Mod']))
    obj = eval("mod.{0}(fromdict=dd)".format(dd['dId_dall_Cls']))

    if strip is not None:
        obj.strip(strip=strip)

    # print
    if verb:
        msg = "Loaded from:\n"
        msg += "    "+pathfileext
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





#############################################
#       Generic tofu object
#############################################

def _check_notNone(dd, lk):
    for k, v in dd.items():
        if k in lk:
            assert v is not None, "{0} should not be None !".format(k)


def _get_attrdictfromobj(obj, dd):
    for k in dd.keys():
        if dd[k] is None:
            dd[k] = getattr(obj,k)
    return dd


class ToFuObjectBase(object):

    __metaclass__ = ABCMeta
    _dstrip = {'strip':None, 'allowed':None}


    def __init_subclass__(cls, *args, **kwdargs):
        super().__init_subclass__(*args, **kwdargs)
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

    @abstractmethod
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


    def to_dict(self, strip=None, sep=_sep, rec=False, deepcopy=False):
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
        rec :       bool
            Flag indicating how to deal with attributes which are themselves
            tofu objects:
                - True : recursively turn them to flat dict
                - False : keep them as tofu objects
        deepcopy:   bool
            Flag indicating whether to populate the dict with:
                - True : copies the object attributes
                - False: references to the object attributes

        Return
        ------
        dout :      dict
            Flat dict containing all the objects attributes

        """
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
                                 parent_key='', sep=sep, rec=rec,
                                 lexcept_key=lexcept_key)
            except Exception as err:
                msg = str(err)
                msg += "\nIssue flattening dict %s"%k
                msg += "\n\n\n" + str(v['dict'])
                raise Exception(msg)
            dout[k] = d
        dout = flatten_dict(dout, parent_key='', sep=sep, rec=rec)
        if deepcopy:
            lkobj = [k for k in dout.keys()
                     if issubclass(dout[k].__class__,ToFuObjectBase)]
            lk = [k for k in dout.keys() if k not in lkobj]
            dout.update(**dict([(k,dout[k]) for k in lk]))
            dout.update(**dict([(k,dout[k].copy(deepcopy=True))
                                for k in lkobj]))
        return dout

    @abstractmethod
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

    def copy(self, strip=None, deepcopy=True):
        """ Return another instance of the object, with the same attributes

        If deep=True, all attributes themselves are also copies
        """
        dd = self.to_dict(strip=strip, deepcopy=deepcopy)
        obj = self.__class__(fromdict=dd)
        return obj

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


    def __eq__(self, obj, detail=True, verb=True):
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
                            eqk = np.allclose(d0[k],d1[k], equal_nan=True)
                            if not eqk:
                                m0 = str(d0[k])
                                m1 = str(d1[k])
                        else:
                            m0 = "shape {0}".format(d0[k].shape)
                            m1 = "shape {0}".format(d1[k].shape)
                    elif issubclass(d0[k].__class__, ToFuObjectBase):
                        eqk = d0[k]==d1[k]
                        if not eqk:
                            m0 = str(d0[k])
                            m1 = str(d1[k])
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

    def __neq__(self, obj, detail=True, verb=True):
        return not self.__eq__(obj, detail=detail, verb=verb)




class ToFuObject(ToFuObjectBase):

    def __init_subclass__(cls, *args, **kwdargs):
        super().__init_subclass__(*args, **kwdargs)

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

    def save(self, path=None, name=None,
             strip=None, sep=_sep, mode='npz',
             compressed=False, verb=True):
        save(self, path=path, name=name,
             strip=strip, compressed=compressed)

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

        kwdargs = locals()
        del kwdargs['self']
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
        cls.strip.__doc__ = doc

    def strip(self, strip=0):
        super().strip(strip=strip)

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

def get_nIne1e2(P, nIn=None, e1=None, e2=None):
    assert np.hypot(P[0],P[1])>1.e-12
    phi = np.arctan2(P[1],P[0])
    ephi = np.array([-np.sin(phi), np.cos(phi), 0.])
    ez = np.array([0.,0.,1.])

    if nIn is None:
        nIn = -P
    nIn = nIn / np.linalg.norm(nIn)
    if e1 is None:
        if np.abs(np.abs(nIn[2])-1.)<1.e-12:
            e1 = ephi
        else:
            e1 = np.cross(nIn,ez)
        e1 = e1 if np.sum(e1*ephi)>0. else -e1
    e1 = e1 / np.linalg.norm(e1)
    msg = "nIn = %s\n"%str(nIn)
    msg += "e1 = %s\n"%str(e1)
    msg += "np.sum(nIn*e1) = {0}".format(np.sum(nIn*e1))
    assert np.abs(np.sum(nIn*e1))<1.e-12, msg
    if e2 is None:
        e2 = np.cross(nIn,e1)
    e2 = e2 / np.linalg.norm(e2)
    return nIn, e1, e2


def get_X12fromflat(X12):
    X1u, X2u = np.unique(X12[0,:]), np.unique(X12[1,:])
    dx1 = np.nanmax(X1u)-np.nanmin(X1u)
    dx2 = np.nanmax(X2u)-np.nanmin(X2u)
    ds = dx1*dx2 / X12.shape[1]
    tol = np.sqrt(ds)/100.
    x1u, x2u = [X1u[0]], [X2u[0]]
    for ii in X1u[1:]:
        if np.abs(ii-x1u[-1])>tol:
            x1u.append(ii)
    for ii in X2u[1:]:
        if np.abs(ii-x2u[-1])>tol:
            x2u.append(ii)
    Dx12 = (np.nanmean(np.diff(x1u)), np.nanmean(np.diff(x2u)))
    x1u, x2u = np.unique(x1u), np.unique(x2u)
    ind = np.full((x1u.size,x2u.size),np.nan)
    for ii in range(0,X12.shape[1]):
        i1 = (np.abs(x1u-X12[0,ii])<tol).nonzero()[0]
        i2 = (np.abs(x2u-X12[1,ii])<tol).nonzero()[0]
        ind[i1,i2] = ii
    return x1u, x2u, ind, Dx12


def create_RaysCones(Ds, us, angs=np.pi/90., nP=40):
    # Check inputs
    Ddim, udim = Ds.ndim, us.ndim
    assert Ddim in [1,2]
    assert Ds.shape[0]==3 and Ds.size%3==0
    assert udim in [1,2]
    assert us.shape[0]==3 and us.size%3==0
    assert type(angs) in [int,float,np.int64,np.float64]
    if udim==2:
        assert Ds.shape==us.shape
    if Ddim==1:
        Ds = Ds.reshape((3,1))
    nD = Ds.shape[1]

    # Compute
    phi = np.linspace(0.,2.*np.pi, nP)
    phi = np.tile(phi,nD)[np.newaxis,:]
    if udim==1:
        us = us[:,np.newaxis]/np.linalg.norm(us)
        us = us.repeat(nD,axis=1)
    else:
        us = us/np.sqrt(np.sum(us**2,axis=0))[np.newaxis,:]
    us = us.repeat(nP, axis=1)
    e1 = np.array([us[1,:],-us[0,:],np.zeros((us.shape[1],))])
    e2 = np.array([-us[2,:]*e1[1,:], us[2,:]*e1[0,:],
                   us[0,:]*e1[1,:]-us[1,:]*e1[0,:]])
    ub = (us*np.cos(angs)
          + (np.cos(phi)*e1+np.sin(phi)*e2)*np.sin(angs))
    Db = Ds.repeat(nP,axis=1)
    return Db, ub



def create_CamLOS2D(P, F, D12, N12,
                    nIn=None, e1=None, e2=None, VType='Tor'):

    # Check/ format inputs
    P = np.asarray(P)
    assert P.shape==(3,)
    assert type(F) in [int, float, np.int64, np.float64]
    F = float(F)
    if type(D12) in [int, float, np.int64, np.float64]:
        D12 = np.array([D12,D12],dtype=float)
    else:
        assert hasattr(D12,'__iter__') and len(D12)==2
        D12 = np.asarray(D12).astype(float)
    if type(N12) in [int, float, np.int64, np.float64]:
        N12 = np.array([N12,N12],dtype=int)
    else:
        assert hasattr(N12,'__iter__') and len(N12)==2
        N12 = np.asarray(N12).astype(int)
    assert type(VType) is str and VType.lower() in ['tor','lin']
    VType = VType.lower()

    # Get vectors
    for vv in [nIn,e1,e2]:
        if not vv is None:
            assert hasattr(vv,'__iter__') and len(vv)==3
            vv = np.asarray(vv).astype(float)
    if nIn is None:
        if VType=='tor':
            nIn = -P
        else:
            nIn = np.r_[0.,-P[1],-P[2]]
    nIn = np.asarray(nIn)
    nIn = nIn/np.linalg.norm(nIn)
    if e1 is None:
       if VType=='tor':
            phi = np.arctan2(P[1],P[0])
            ephi = np.r_[-np.sin(phi),np.cos(phi),0.]
            if np.abs(np.abs(nIn[2])-1.)<1.e-12:
                e1 = ephi
            else:
                e1 = np.cross(nIn,np.r_[0.,0.,1.])
                e1 = e1 if np.sum(e1*ephi)>0. else -e1
       else:
            if np.abs(np.abs(nIn[0])-1.)<1.e-12:
                e1 = np.r_[0.,1.,0.]
            else:
                e1 = np.cross(nIn,np.r_[0.,0.,1.])
                e1 = e1 if e1[0]>0. else -e1
    e1 = np.asarray(e1)
    e1 = e1/np.linalg.norm(e1)
    assert np.abs(np.sum(nIn*e1))<1.e-12
    if e2 is None:
        e2 = np.cross(e1,nIn)
    e2 = np.asarray(e2)
    e2 = e2/np.linalg.norm(e2)
    assert np.abs(np.sum(nIn*e2))<1.e-12
    assert np.abs(np.sum(e1*e2))<1.e-12

    # Get starting points
    d1 = D12[0]*np.linspace(-0.5,0.5,N12[0],endpoint=True)
    d2 = D12[1]*np.linspace(-0.5,0.5,N12[1],endpoint=True)
    d1 = np.repeat(d1,N12[1])
    d2 = np.tile(d2,N12[0])
    d1 = d1[np.newaxis,:]*e1[:,np.newaxis]
    d2 = d2[np.newaxis,:]*e2[:,np.newaxis]

    Ds = P[:,np.newaxis] - F*nIn[:,np.newaxis] + d1 + d2
    us = P[:,np.newaxis] - Ds
    return Ds, us



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
#           Plot KeyHandler
################ ##############################

class KeyHandler(object):
    """ Base class for handling event on tofu interactive figures """

    def __init__(self, can=None, daxT=None, ntMax=3, nchMax=3, nlambMax=3,
                 combine=False):
        lk = ['t','chan','chan2D','lamb','cross','hor','colorbar','txtt','txtch','txtlamb','other']
        assert all([kk in lk for kk in daxT.keys()]), str(daxT.keys())
        assert all([type(dd) is list for dd in daxT.values()]), str(daxT.values())
        self.lk = sorted(list(daxT.keys()))

        # Remove None axes from daxT
        for kk in daxT.keys():
            daxT[kk] = [dd for dd in daxT[kk] if dd['ax'] is not None]

        self.can = can
        self.combine = combine
        daxr, dh = self._make_daxr_dh(daxT)

        self.daxT = daxT
        self.daxr, self.dh = daxr, dh
        self.store_rcParams = None
        self.lkeys = ['right','left','control','shift','alt']
        if 'chan2D' in self.daxT.keys() and len(self.daxT['chan2D'])>0:
            self.lkeys += ['up','down']
        self.curax = None
        self.ctrl = False
        self.shift = False
        self.alt = False
        self.ref, dnMax = {}, {'chan':nchMax, 'chan2D':nchMax,'t':ntMax,'lamb':nlambMax}
        for kk in self.lk:
            if not kk in ['cross','hor','colorbar','txtt','txtch','txtlamb','other']:
                if kk=='t':
                    nn = ntMax
                elif 'chan' in kk:
                    nn = nchMax
                else:
                    nn = nlambMax
                self.ref[kk] = {'ind':np.zeros((nn,),dtype=int),
                                'val':[None for ii in range(0,dnMax[kk])],
                                'ncur':1, 'nMax':dnMax[kk]}

        self._set_dBck(list(self.daxr.keys()))

    def _make_daxr_dh(self,daxT):
        daxr, lh, dh = {}, [], {}
        for kk in self.lk:
            for ii in range(0,len(daxT[kk])):
                dax = daxT[kk][ii]
                if 'invert' in dax.keys():
                    invert = dax['invert']
                else:
                    invert = None
                if 'xref' in dax.keys():
                    xref = dax['xref']
                else:
                    xref = None
                if 'dh' in dax.keys() and dax['dh'] is not None:
                    for tt in dax['dh'].keys():
                        for jj in range(0,len(dax['dh'][tt])):
                            if 'trig' not in dax['dh'][tt][jj].keys():
                                dax['dh'][tt][jj]['trig'] = None
                            if 'xref' not in dax['dh'][tt][jj].keys():
                                dax['dh'][tt][jj]['xref'] = xref
                    dhh = dax['dh']
                else:
                    dhh = None
                daxr[dax['ax']] = {'Type':kk, 'invert':invert,
                                   'xref':xref, 'Bck':None, 'dh':dhh}
                if 'incx' in dax.keys():
                    daxr[dax['ax']]['incx'] = dax['incx']

                if dhh is not None:
                    for kh in dhh.keys():
                        for jj in range(0,len(dhh[kh])):
                            for ii in range(0,len(dhh[kh][jj]['h'])):
                                hh = dhh[kh][jj]['h'][ii]
                                if hh not in lh:
                                    lh.append(hh)
                                    dh[hh] = {'ax':dax['ax'],
                                              'Type':kh, 'vis':False,
                                              'xref':dhh[kh][jj]['xref']}
                                    if ii==0:
                                        dh[hh]['trig'] = dhh[kh][jj]['trig']

        return daxr, dh


    def disconnect_old(self, force=False):
        if force:
            self.can.mpl_disconnect(self.can.manager.key_press_handler_id)
        else:
            lk = [kk for kk in list(plt.rcParams.keys()) if 'keymap' in kk]
            self.store_rcParams = {}
            for kd in self.lkeys:
                self.store_rcParams[kd] = []
                for kk in lk:
                    if kd in plt.rcParams[kk]:
                        self.store_rcParams[kd].append(kk)
                        plt.rcParams[kk].remove(kd)
        self.can.mpl_disconnect(self.can.button_pick_id)

    def reconnect_old(self):
        if self.store_rcParams is not None:
            for kd in self.store_rcParams.keys():
                for kk in self.store_rcParams[kk]:
                    if kd not in plt.rcParams[kk]:
                        plt.rcParams[kk].append(kd)

    def connect(self):
        keyp = self.can.mpl_connect('key_press_event', self.onkeypress)
        keyr = self.can.mpl_connect('key_release_event', self.onkeypress)
        butp = self.can.mpl_connect('button_press_event', self.mouseclic)
        res = self.can.mpl_connect('resize_event', self.resize)
        #butr = self.can.mpl_connect('button_release_event', self.mouserelease)
        self.can.manager.toolbar.release = self.mouserelease

        self._cid = {'keyp':keyp, 'keyr':keyr,
                     'butp':butp, 'res':res}#, 'butr':butr}

    def disconnect(self):
        for kk in self._cid.keys():
            self.can.mpl_disconnect(self._cid[kk])
        self.can.manager.toolbar.release = lambda event: None

    def home(self):
        """ To be filled when matplotlib issue completed """

    def mouserelease(self, event):
        msg = "Make sure you release the mouse button on an axes !"
        msg += "\n Otherwise the background plot cannot be properly updated !"
        C0 = self.can.manager.toolbar._active == 'PAN'
        C1 = self.can.manager.toolbar._active == 'ZOOM'

        if C0 or C1:
            ax = self.curax_panzoom
            assert ax is not None, msg
            lax = ax.get_shared_x_axes().get_siblings(ax)
            lax += ax.get_shared_y_axes().get_siblings(ax)
            lax = list(set(lax))
            self._set_dBck(lax)

    def resize(self, event):
        self._set_dBck(list(self.daxr.keys()))

    def _set_dBck(self, lax):
        # Make all invisible
        for ax in lax:
            if self.daxr[ax]['dh'] is not None:
                for typ in self.daxr[ax]['dh']:
                    for ii in range(0,len(self.daxr[ax]['dh'][typ])):
                        for hh in self.daxr[ax]['dh'][typ][ii]['h']:
                            hh.set_visible(False)

        # Draw and reset Bck
        self.can.draw()
        for ax in lax:
            #ax.draw(self.can.renderer)
            self.daxr[ax]['Bck'] = self.can.copy_from_bbox(ax.bbox)

        # Redraw
        for ax in lax:
            if self.daxr[ax]['dh'] is not None:
                for typ in self.daxr[ax]['dh']:
                    for ii in range(0,len(self.daxr[ax]['dh'][typ])):
                        for hh in self.daxr[ax]['dh'][typ][ii]['h']:
                            hh.set_visible(self.dh[hh]['vis'])
                #ax.draw(self.can.renderer)
        self.can.draw()

    def _update_restore_Bck(self, lax):
        for ax in lax:
            self.can.restore_region(self.daxr[ax]['Bck'])

    def _update_vlines_ax(self, ax, axT):
        for jj in range(0,len(self.daxr[ax]['dh']['vline'])):
            if (self.daxr[ax]['dh']['vline'][jj]['xref'] is
                self.daxr[self.curax]['xref']):
                for ii in range(0,self.ref[axT]['ncur']):
                    hh = self.daxr[ax]['dh']['vline'][jj]['h'][ii]
                    hh.set_xdata(self.ref[axT]['val'][ii])
                    self.dh[hh]['vis'] = True
                    hh.set_visible(self.dh[hh]['vis'])
                    ax.draw_artist(hh)
            else:
                xref = self.daxr[ax]['dh']['vline'][jj]['xref']
                for ii in range(0,self.ref[axT]['ncur']):
                    hh = self.daxr[ax]['dh']['vline'][jj]['h'][ii]
                    val = self.ref[axT]['val'][ii]
                    ind = np.argmin(np.abs(xref-val))
                    hh.set_xdata(xref[ind])
                    self.dh[hh]['vis'] = True
                    hh.set_visible(self.dh[hh]['vis'])
                    ax.draw_artist(hh)
            for ii in range(self.ref[axT]['ncur'],self.ref[axT]['nMax']):
                hh = self.daxr[ax]['dh']['vline'][jj]['h'][ii]
                self.dh[hh]['vis'] = False
                hh.set_visible(self.dh[hh]['vis'])
                ax.draw_artist(hh)

    def _update_vlines(self):
        lax = []
        axT = self.daxr[self.curax]['Type']
        for dax in self.daxT[axT]:
            self._update_vlines_ax(dax['ax'], axT)
            lax.append(dax['ax'])
        return lax

    def _update_vlines_and_Eq(self):
        axT = self.daxr[self.curax]['Type']
        if not axT in ['t','chan','chan2D','lamb']:
            lax = self._update_vlines()
            return lax

        lax = []
        if self.combine and axT in ['chan','chan2D']:
            ldAX = [dd for dd in self.daxT[axT] if self.curax==dd['ax']]
        else:
            ldAX = self.daxT[axT]
        #xref = self.ref[axT]['val']
        for dax in ldAX:
            ax = dax['ax']
            if self.daxr[ax]['dh'] is None:
                continue
            lax.append(ax)
            dtg = self.daxr[ax]['dh']['vline'][0]['trig']
            if dtg is None:
                self._update_vlines_ax(ax, axT)
                continue

            for jj in range(0,len(self.daxr[ax]['dh']['vline'])):
                dtg = self.daxr[ax]['dh']['vline'][jj]['trig']
                xref = self.daxr[ax]['dh']['vline'][jj]['xref']
                if xref.ndim==1:
                    xvfunc = np.abs
                    xvset = lambda h, v: h.set_xdata(v)
                else:
                    xvfunc = lambda xv: np.sum(xv**2,axis=1)
                    xvset = lambda h, v: h.set_data(v)
                for ii in range(0,self.ref[axT]['ncur']):
                    hh = self.daxr[ax]['dh']['vline'][jj]['h'][ii]
                    ind = self.ref[axT]['ind'][ii]
                    val = self.ref[axT]['val'][ii]
                    if xref is not self.daxr[self.curax]['xref']:
                        ind = np.argmin(xvfunc(xref-val))
                        val = xref[ind]
                    xvset(hh,val)
                    #hh.set_xdata(val)
                    self.dh[hh]['vis'] = True
                    for kk in dtg.keys():
                        for ll in range(0,len(dtg[kk])):
                            h = dtg[kk][ll]['h'][ii]
                            if dtg[kk][ll]['xref'] is xref:
                                indh = ind
                            else:
                                indh = np.argmin(np.abs(dtg[kk][ll]['xref']-val))

                            if 'txt' in dtg[kk][ll].keys():
                                if 'format' in dtg[kk][ll].keys():
                                    sss = '{0:%s}'%dtg[kk][ll]['format']
                                    h.set_text(sss.format(dtg[kk][ll]['txt'][indh]))
                                else:
                                    h.set_text(dtg[kk][ll]['txt'][indh])
                            elif 'xy' in dtg[kk][ll].keys():
                                h.set_data(dtg[kk][ll]['xy'][indh])
                            elif 'imshow' in dtg[kk][ll].keys():
                                h.set_data(dtg[kk][ll]['imshow']['data'][indh,:][dtg[kk][ll]['imshow']['ind']])
                            elif 'pcolormesh' in dtg[kk][ll].keys():
                                ncol = dtg[kk][ll]['pcolormesh']['cm'](dtg[kk][ll]['pcolormesh']['norm'](dtg[kk][ll]['pcolormesh']['data'][indh,:]))
                                h.set_facecolor(ncol)
                            else:
                                if 'x' in dtg[kk][ll].keys():
                                    h.set_xdata(dtg[kk][ll]['x'][indh,:])
                                if 'y' in dtg[kk][ll].keys():
                                    h.set_ydata(dtg[kk][ll]['y'][indh,:])
                            self.dh[h]['vis'] = True
                            h.set_visible(self.dh[h]['vis'])
                            #self.dh[h]['ax'].draw_artist(h)
                            if not self.dh[h]['ax'] in lax:
                                lax.append(self.dh[h]['ax'])
                    hh.set_visible(self.dh[hh]['vis'])
                    #ax.draw_artist(hh)
                for ii in range(self.ref[axT]['ncur'],self.ref[axT]['nMax']):
                    hh = self.daxr[ax]['dh']['vline'][jj]['h'][ii]
                    self.dh[hh]['vis'] = False
                    for kk in dtg.keys():
                        for ll in range(0,len(dtg[kk])):
                            h = dtg[kk][ll]['h'][ii]
                            self.dh[h]['vis'] = False
                            h.set_visible(self.dh[h]['vis'])
                            #self.dh[h]['ax'].draw_artist(h)
                            if not self.dh[h]['ax'] in lax:
                                lax.append(self.dh[h]['ax'])
                    hh.set_visible(self.dh[hh]['vis'])
                    #ax.draw_artist(hh)

        for ax in lax:
            # Sort alphabetically to make sure vline (pix) is plotted after
            # imshow / pcolormesh (otherwise pixel not visible)
            for kk in sorted(list(self.daxr[ax]['dh'].keys())):
                for ii in range(0,len(self.daxr[ax]['dh'][kk])):
                    for h in self.daxr[ax]['dh'][kk][ii]['h']:
                        ax.draw_artist(h)
        return lax


    def _update_blit(self, lax):
        for ax in lax:
            self.can.blit(ax.bbox)

    def mouseclic(self,event):
        C0 = event.inaxes is not None and event.button == 1
        if not C0:
            return
        self.curax_panzoom = event.inaxes
        C1 = self.daxr[event.inaxes]['Type'] in ['t','chan','chan2D']
        C2 = self.can.manager.toolbar._active is None
        C3 = self.daxr[event.inaxes]['Type']=='chan2D'
        if not (C1 and C2):
            return
        self.curax = event.inaxes

        Type = self.daxr[self.curax]['Type']
        #Type = 'chan' if 'chan' in Type else Type
        if self.shift and self.ref[Type]['ncur']>=self.ref[Type]['nMax']:
            print("     Max. nb. of %s plots reached !!!"%Type)
            return

        val = self.daxr[event.inaxes]['xref']
        if C3:
            evxy = np.r_[event.xdata,event.ydata]
            d2 = np.sum((val-evxy[np.newaxis,:])**2,axis=1)
        else:
            d2 = np.abs(event.xdata-val)
        ind = np.nanargmin(d2)
        val = val[ind]
        if self.ctrl:
            ii = 0
        elif self.shift:
            ii = int(self.ref[Type]['ncur'])
        else:
            ii = int(self.ref[Type]['ncur'])-1
        self.ref[Type]['ind'][ii] = ind
        self.ref[Type]['val'][ii] = val
        self.ref[Type]['ncur'] = ii+1
        self.update()

    def onkeypress(self,event):
        C0 = self.can.manager.toolbar._active is None
        C1 = [kk in event.key for kk in self.lkeys]
        C2 = event.name is 'key_release_event' and event.key=='shift'
        C3 = event.name is 'key_press_event'
        C4 = event.name is 'key_release_event' and event.key=='alt'
        C5 = event.name is 'key_release_event' and event.key=='control'

        if not (C0 and any(C1) and (C2 or C3 or C4 or C5)):
            return

        if event.key=='control':
            self.ctrl = False if C5 else True
            return
        if event.key=='shift':
            self.shift = False if C2 else True
            return
        if event.key=='alt':
            self.alt = False if C4 else True
            return

        Type = self.daxr[self.curax]['Type']
        #Type = 'chan' if 'chan' in Type else Type
        if self.shift and self.ref[Type]['ncur']>=self.ref[Type]['nMax']:
                print("     Max. nb. of %s plots reached !!!"%Type)
                return

        val = self.daxr[self.curax]['xref']
        if self.alt:
            inc = 50 if Type=='t' else 10
        else:
            inc = 1
        if self.daxr[self.curax]['Type']=='chan2D':
            kdir = [kk for kk in self.lkeys
                    if (kk in event.key and not kk in ['shift','control','alt'])]
            c = -inc if self.daxr[self.curax]['invert'] else inc
            x12 = val[self.ref[Type]['ind'][self.ref[Type]['ncur']-1],:]
            x12 = x12 + c*self.daxr[self.curax]['incx'][kdir[0]]
            d2 = np.sum((val-x12[np.newaxis,:])**2,axis=1)
            ind = np.nanargmin(d2)
        else:
            c = -inc if 'left' in event.key else inc
            ind = (self.ref[Type]['ind'][self.ref[Type]['ncur']-1]+c)
            ind = ind%val.size
        val = val[ind]
        if self.ctrl:
            ii = 0
        elif self.shift:
            ii = self.ref[Type]['ncur']
        else:
            ii = self.ref[Type]['ncur']-1
        self.ref[Type]['ind'][ii] = ind
        self.ref[Type]['val'][ii] = val
        self.ref[Type]['ncur'] = ii+1
        self.update()

    ##### To be implemented for each case ####
    def set_dBack(self):
        """ Choose which axes need redrawing and call self._set_dBck() """
    def update(self):
        """ Implement basic behaviour, and call self._restore_Bck() """
