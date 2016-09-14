# -*- coding: utf-8 -*-
"""
Provide data handling class and methods (storing, processing, plotting...)
"""

import numpy as np
import matplotlib.pyplot as plt


# ToFu-specific
import tofu.defaults as tfd
import tofu.pathfile as tfpf
from . import _compute as _tft_c
from . import _plot as _tft_p


__author__ = "Didier Vezinet"
__all__ = ["PreData"]



class PreData(object):
    """ A class defining a data-handling object, data is stored as read-only attribute, copies of it can be modified, methods for plotting, saving...

    The name of the class refers to Pre-treatment Data (i.e.: in the context of tomography, data that is pre-treated before being fed to an inversion algorithm).
    ToFu provide a generic data-handling class, which comes a robust data storing policy: the input data is stored in a read-only attribute and the data-processing methods are used on a copy (e.g.: for computing the SVD, Fourier transform, shorten the time interval of interest, eliminate some channels...).
    Furthermore, methods for interactive plotting are provided as well as a saving method

    Parameters:
    -----------



    Returns:
    --------
    obj :       PreData
        The created instance

    """




    def __init__(self, data, t=None, Chans=None, Id=None, Exp='AUG', shot=None, Diag='SXR', dtime=None, dtimeIn=False, SavePath=None,
                 LIdDet=None, DtRef=None):

        self._Done = False
        self._set_Id(Id, Exp=Exp, shot=shot, Diag=Diag, SavePath=SavePath, dtime=dtime, dtimeIn=dtimeIn, Dt=DtRef, t=t)
        self._set_data(data, t=t, Chans=Chans, DtRef=DtRef, LIdDet=LIdDet)
        self._init()
        self._doAll()
        self._Done = True


    @property
    def Id(self):
        return self._Id
    @property
    def Exp(self):
        return self.Id.Exp
    @property
    def shot(self):
        return self.Id.shot
    @property
    def data(self):
        return self._data
    @property
    def t(self):
        return self._t
    @property
    def Dt(self):
        return self._Dt
    @property
    def Chans(self):
        return self._Chans

    @property
    def svd(self):
        return self._svd
    @property
    def FFT(self):
        return self._FFT


    def _check_inputs(self, Id=None, data=None, t=None, Chans=None, LIdDet=None, DtRef=None, Exp=None, shot=None, Diag=None, dtime=None, dtimeIn=None, SavePath=None,
                      Dt=None, Resamp_t=None, Resamp_f=None, Resamp_Method=None, Resamp_interpkind=None, Calc=None):
        _PreData_check_inputs(Id=Id, data=data, t=t, Chans=Chans, LIdDet=LIdDet, DtRef=DtRef, Exp=Exp, shot=shot, Diag=Diag, dtime=dtime, dtimeIn=dtimeIn, SavePath=SavePath,
                              Dt=Dt, Resamp_t=Resamp_t, Resamp_f=Resamp_f, Resamp_Method=Resamp_Method, Resamp_interpkind=Resamp_interpkind, Calc=Calc)

    def _set_Id(self, Val, Exp=None, Diag=None, shot=None, dtime=None, dtimeIn=False, SavePath=None, t=None, Dt=None):
        if self._Done:
            Out = tfpf._get_FromItself(self.Id,{'Exp':Exp, 'Diag':Diag, 'shot':shot, 'dtime':dtime, '_dtimeIn':dtimeIn, 'SavePath':SavePath})
            Exp, Diag, shot, dtime, dtimeIn, SavePath = Out['Exp'], Out['Diag'], Out['shot'], Out['dtime'], Out['dtimeIn'], Out['SavePath']
        if Val is None:
            Val = _tft_c.get_DefName(t=t, Dt=Dt)
        self._check_inputs(Id=Val)
        if type(Val) is str:
            tfpf._check_NotNone({'Exp':Exp, 'Diag':Diag, 'shot':shot, 'dtimeIn':dtimeIn})
            self._check_inputs(Exp=Exp, Diag=Diag, shot=shot, SavePath=SavePath, dtime=dtime, dtimeIn=dtimeIn)
            Val = tfpf.ID('PreData', Val, Exp=Exp, Diag=Diag, shot=shot, SavePath=SavePath, dtime=dtime, dtimeIn=dtimeIn)
        self._Id = Val

    def _set_data(self, data, t=None, Chans=None, DtRef=None, LIdDet=None):
        self._check_inputs(data=data, t=t, Chans=Chans, DtRef=DtRef, LIdDet=LIdDet)

        OutRef, Out, Outind = _tft_c._PreData_set_data(data, t, Chans, DtRef=DtRef, LIdDet=LIdDet)
        self._dataRef, self._tRef, self._ChansRef, self._NChansRef, self._LIdDetRef, self._DtRef = OutRef
        self._data, self._t, self._Chans, self._NChans, self._LIdDet, self._Dt = Out
        self._indOut, self._indCorr = Outind

        if not LIdDet is None:    # On purpose, because if they are created they should not be stored as objects
            self.Id.set_LObj(self._LIdDetRef)

    def _init(self):
        self._Resamp_t, self._Resamp_f, self._Resamp_Method, self._Resamp_interpkind = None, None, None, None
        self._interp_lt, self._interp_lNames, self._interp_UNames = None, None, None
        self._Subtract_tsub = None
        self._FFTPar = None
        self._PhysNoise = None
        self._NoiseModel = None

    def set_Dt(self, Dt=None, Calc=True):
        """ Set the time interval to which the data should be limited (does not affect the reference data)

        While the original data set and time base are always preserved in the background, you can change your mind and focus on a smaller interval included in the original one.
        This can be convenient for applying data treatment (SVD, fft...) to parts of the signal lifetime only.

        Parameters
        ----------
        Dt :    None / list
            The time interval of interest, as a list of len()=2 in increasing values
        Calc :  bool
            Flag indicating whether the calculation should be triggered immediately

        """
        self._check_inputs(Dt=Dt)
        # assert Dt is None or (hasattr(Dt,'__getitem__') and len(Dt)==2), "Arg Dt must be a len==2 list, tuple or np.ndarray !"
        self._Dt = Dt
        if Calc:
            self._doAll()



    def set_Resamp(self, t=None, f=None, Method='movavrg', interpkind='linear', Calc=True):
        """ Re-sample the data and time vector

        Use a new time vector that can either be:
            - provided directly (if t is not None)
            - computed from an input sampling frequency (if f is not None)
        If but t and f are provided, t is used as the time vector and f is only used for the moving average

        Then, the data is re-computed on this new time vector using either interpolation ('interp') or moving average ('movavrg')

        Parameters
        ----------
        t :             None / np.ndarray

        f :             None / int / float

        Method :        str

        Resamp :        bool

        interpkind :    str

        Calc :          bool
            Flag indicating whether the calculation should be triggered immediately

        """
        self._check_inputs(Resamp_t=t, Resamp_f=f, Resamp_Method=Method, Resamp_interpkind=interpkind, Calc=Calc)

        self._Resamp_t = t
        self._Resamp_f = f
        self._Resamp_Method = Method
        self._Resamp_interpkind = interpkind
        if Calc:
            self._doAll()




    def select(self, Val=None, Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='In', Out=bool, ToIn=False):
        """ Return a sub-set of the data (channels-wise selection)

        Return an array of indices of channels selected according to the chosen criteria with chosen values
        Use either Val or (PreExp and PostExp)

        Parameters
        ----------
        Val :       list or str
            List of values that the chosen criteria must match (converted to one-item list if str)
        Crit :      str
            Criterion used to select some channels, must be among their tfpf.ID class attributes (e.g.: 'Name', 'SaveName'...) or IFTF.ID.USRdict ('Cam',...)
        PreExp :    list or str
            List of str expressions to be fed to eval(PreExp[ii]+" Detect.Crit "+PostExp[ii]) or eval(PreExp[ii]+" Detect.USRdict.Crit "+PostExp[ii])
        PostExp :   list or str
            List of str expressions to be fed to eval(PreExp[ii]+" Detect.Crit "+PostExp[ii]) or eval(PreExp[ii]+" Detect.USRdict.Crit "+PostExp[ii])
        Log :       str
            Flag ('or' or 'and') indicating whether to select the channels matching all criteria or any
        InOut :     str
            Flag ('In' or 'Out') indicating whether to select all channels matching the criterion, or all except those
        Out :       type or str
            Flag (bool, int or an attribute of tfpf.ID or tfpf.ID.USRdict) indicating whether to return an array of boolean indices or int indices, or a list of the chosen attributes (e.g.: 'Name')
        ToIn :      bool
            Flag indicating whether indices should be returned with respect to the channels that are considered as included only (see obj.In_list() to see these channels)

        Returns
        -------
        ind :       np.ndarray
            Indices of the selected channels, as a bool or int array


        Examples
        --------
        >> ind = TFT.PreData.select(Val=['H','J'], Crit='Cam', Log='any', InOut='In', Out=bool)
           Will return a bool array of the indices of all channels for which 'Cam' is 'H' or 'J'
        >> ind = PreData.select(Crit='Name', PreExp=["'F' in ", "'6' in "], Log='and', InOut='In', Out=int)
           Will return an int array of indices of all channels for which 'F' and '6' are both included in the name
        >> ind = PreData.select(Crit='CamHead', PreExp=["'F' in ", "'2' in "], Log='any', InOut='Out', Out='Name')
           Will return the names (as a list) of all channels except those that have a camera head name that includes a 'F' or a '2' (i.e.: except camera heads 'F' and 'H2', 'I2', 'J2', 'K2')

        """

        if ToIn:
            return tfpf.SelectFromListId(self._LIdDet, Val=Val, Crit=Crit, PreExp=PreExp, PostExp=PostExp, Log=Log, InOut=InOut, Out=Out)
        else:
            return tfpf.SelectFromListId(self._LIdDetRef, Val=Val, Crit=Crit, PreExp=PreExp, PostExp=PostExp, Log=Log, InOut=InOut, Out=Out)




    def Out_add(self, Val=[], LCrit=['Name','Cam','CamHead'], indOut=None, Calc=True):
        """ Add desired channels to the list of channels to be excluded

        It is possible to store a list a list of channels that are thought to be corrupted or more generally that, after closer inspection, are considered not fit.
        This list is then automatically passed on to further ToFu objects (e.g.: for inversions), so that the corresponding data is excluded from all further processes.
        PreData provides methods to append channel names to this list (in fact you can even exclude whole cameras).

        Parameters
        ----------
        Val :      list
            Fed to self.select(), list of values for criteria in LCrit that should be used to exclude channels (e.g.: list of channel names of camera names)
        LCrit :     list
            Fed to self.select(), list of criteria against which to select the channels matching the values in Val (should be attributes of :class:`tofu.pathfile.ID` or of its USRdict attribute)
        indOut :    None / np.ndarray
            Alternatively, you can directly pass a (N,) bool array whereN matches the number of channels and True means that a channel should be excluded, thus setting self._indOut
        Calc :      bool
            Flag indicating whether the calculation should be triggered immediately

        """
        if not indOut is None:
            assert isinstance(indOut,np.ndarray) and indOut.dtype.name=='bool' and indOut.ndim==1 and indOut.size==self._NChansRef, "Arg indOut must be a (N,) np.ndarray of bool !"
            self._indOut = indOut
        elif not Val==[]:
            ind = np.zeros((len(LCrit),self._NChansRef),dtype=bool)
            for ii in range(0,len(LCrit)):
                ind[ii,:] = self.select(Val=Val, Crit=LCrit[ii], InOut='In', Out=bool)
            self._indOut = self._indOut | np.any(ind,axis=0)
        if Calc & (not Val==[] or not indOut is None):
            self._doAll()

    def In_add(self, Val=[], LCrit=['Name','Cam','CamHead'], Calc=True):
        """ Add channels to the list of channels to be re-included as valid channels

        Provides a mechanism opposite to :meth:`~tofu.treat.PreData.Out_add()`.
        We you change your mind about a series of channel and think they should be re-included as valid, pass them to this method using the same arguments as self.Out_add()

        Parameters
        ----------
        Val :      list
            Fed to self.select(), list of values for criteria in LCrit that should be used to exclude channels (e.g.: list of channel names of camera names)
        LCrit :     list
            Fed to self.select(), list of criteria against which to select the channels matching the values in Val (should be attributes of :class:`tofu.pathfile.ID` or of its USRdict attribute)
        indOut :    None / np.ndarray
            Alternatively, you can directly pass a (N,) bool array whereN matches the number of channels and True means that a channel should be excluded, thus setting self._indOut
        Calc :      bool
            Flag indicating whether the calculation should be triggered immediately

        """
        if not Val==[]:
            ind = np.ones((len(LCrit),self._NChansRef),dtype=bool)
            for ii in range(0,len(LCrit)):
                ind[ii,:] = self.select(Val=Val, Crit=LCrit[ii], InOut='Out', Out=bool)
            self._indOut = self._indOut & np.all(ind,axis=0)
            if Calc:
                self._doAll()

    def Out_list(self, Out='Name'):
        """ Return the list of excluded channel names (considered corrupted)

        This lists the channels indicated by self._indOut, populated using self.Out_add() and de-populated using self.In_add().
        The output can be returned as a list of channel Names

        Parameters
        ----------
        Out :       str
            Flag indicating in which form to return the output (fed to :meth:`~tofu.treat.PreData.select()`)

        Returns
        -------
        L :         list
            List of excluded channels in the required form

        """
        L = self.select(Out=Out)
        L = [L[ii] for ii in range(0,len(L)) if self._indOut[ii]]
        return L

    def In_list(self, Out='Name'):
        """ Return the list of included channel names (considered valid)

        The equivalent of :meth:`~tofu.treat.PreData.Out_list()`, but this time returning the complementary list

        Parameters
        ----------
        Out :       str
            Flag indicating in which form to return the output (fed to :meth:`~tofu.treat.PreData.select()`)

        Returns
        -------
        L :         list
            List of excluded channels in the required form

        """
        L = self.select(Out=Out)
        L = [L[ii] for ii in range(0,len(L)) if not self._indOut[ii]]
        return L

    def interp(self, lt=[], lNames=[], Calc=True):
        """ Perform linear interpolation of data at chosen times for chosen channels

        As opposed to self.set_t(), this method shall be used to interpolate data of a small number of channels at a small sumber of time points.
        Use this to correct a small number of time points that are clearly corrupted when you think the rest shall be preserved.

        !!! This is done with respect to the reference time vector and dataset, to avoid propagating errors through later data treatment (use self.plot(V='Ref') to plot the reference data set) !!!

        Parameters
        ----------
        lt :        list
            Times at which linear interpolation should be performed
        lNames :    list
            Channels for which interpolation should be performed, one element per corresponding time point, elements can be:
                - list of str: list of channel names that should be interpolated for the corresponding time point
                - str: single channel name that should be interpolated for the corresponding time point
                - 'All': all channels should be interpolated for the corresponding time point
        Calc :      bool
            Flag indicating whether data should be updated immediately

        Examples
        --------
        >> obj.interp(lt=[2.55, 5.10, 6.84], lNames=[['H_021','J_014'], 'F_10', 'All'], Calc=True)
           Will perform interpolation for 2 channels for the first time point, for one channel for the second, and for all channels for the last time point

        """
        self._interp_lt, self._interp_lNames, self._interp_UNames = _tft_c._PreData_interp(lt=lt, lNames=lNames)
        if Calc:
            self._doAll()

    def substract_Dt(self, tsub=None, Calc=True):
        """ Allows subtraction of data at one time step from all data

        Can be convenient for plotting background-subtracted signal (background meaning signal before a reference time step).

        Parameters
        ----------
        tsub :     int / float / iterable
            A time value, or a time interval indicating which part of the signal is to be considered as reference and subtracted from the rest
                - int / float :
        Calc :      bool
            Flag indicating whether data should be updated immediately

        """
        assert tsub is None or type(tsub) in [int,float,np.float64,list,tuple,np.ndarray], "Arg tsub must be a time value (int,float) or a time interval (list,tuple,np.ndarray) !"
        if hasattr(tsub,'__iter__'):
            assert len(tsub)==2 and tsub[1]>tsub[0] and np.diff(tsub)>np.mean(np.diff(self.t)), "Arg tsub must be an increasing time interval larger than the signal time reolsution !"
        self._Subtract_tsub = tsub
        if Calc:
            self._doAll()


    def set_fft(self, DF=None, Harm=True, DFEx=None, HarmEx=True, Calc=True):
        """ Return the FFT-filtered signal (and the rest) in the chosen frequency window (in Hz) and in all the higher harmonics (optional)

        Can also exclude a given interval and its higher harmonics from the filtering (optional)

        Parameters
        ----------
        DF :        iterable
            Iterable of len()=2, containing the lower and upper bounds of the frequency interval (Hz) to be used for filtering
        Harm :      bool
            If True all the higher harmonics of the interval DF will also be included
        DFEx :      list
            List or tuple of len()=2, containing the lower and upper bounds of the frequency interval to be excluded from filtering (in case it overlaps with some high harmonics of DF)
        HarmEx :    bool
            If True all the higher harmonics of the interval DFEx will also be excluded

        """
        self._FFTPar = {'DF':DF, 'Harm':Harm, 'DFEx':DFEx, 'HarmEx':HarmEx}
        if Calc:
            self._doAll()


    def _doAll(self):
        """ Centralizes all the computations, run everytime something is updated (time interval, valid channels, resampling, subtraction, fft...) """

        # Get the list of channels considered valid
        self._Chans = self.In_list()
        self._NChans = np.sum(~self._indOut)
        self._LIdDet = [self._LIdDetRef[ii] for ii in range(0,self._NChansRef) if not self._indOut[ii]]

        data = np.copy(self._dataRef[:,~self._indOut])
        t = np.copy(self._tRef)

        # Interp of individual corrupted time points (must be done on reference data and time vector to be robust)
        if not self._interp_lt is None:
            unames = [nn for nn in self._interp_UNames if not nn=='All']
            inds = [self.select(Val=nn, Crit='Name', Out=bool)[~self._indOut].nonzero()[0][0] for nn in unames]
            data = _tft_c._PreData_doAll_interp(data, t, self._interp_lt, self._interp_lNames, unames, inds)

        # Time re-sampling
        if not (self._Resamp_t is None and self._Resamp_f is None):
            data, t = _tft_c._PreData_doAll_Resamp(self._DtRef, t, data, self._Resamp_t, self._Resamp_f, self._Resamp_Method, self._Resamp_interpkind)

        # Subtracting reference time
        if not self._Subtract_tsub is None:
            data = _tft_c._PreData_doAll_Subtract(data, t, self._Subtract_tsub)

        # Performing fft
        if not self._FFTPar is None:
            data = _tft_c._PreData_doAll_FFT(data, t, self._FFTPar)

        # Focus on time interval (only for visualization)
        if not self._Dt is None:
            indt = (t>=self._Dt[0]) & (t<=self._Dt[1]) if not self._Dt is None else np.ones((t.size,),dtype=bool)
            data, t = data[indt,:], t[indt]

        self._data, self._t = data, t


    def Corr_add(self, Val=[], LCrit=['Name','Cam','CamHead'], indCorr=None, Calc=True):
        """ Add channels to the list of channels that are thought to need correction

        When a channel is suspected to need correction (mismatching retrofit due for example to wrong calibration), it can be included in a dedicated correction list.
        Channels in this list can then be discarded for the inversion, a correction coefficient can be computed from the retrofit, and the inversion can be re-done using this correction coefficient.
        This list works like the list of excluded / corrupted channels self.Out_list()

        Parameters
        ----------
        Val :      list
            Fed to self.select(), list of values for criteria in LCrit that should be used to exclude channels (e.g.: list of channel names of camera names)
        LCrit :     list
            Fed to self.select(), list of criteria against which to select the channels matching the values in Val (should be attributes of :class:`tofu.pathfile.ID` or of its USRdict attribute)
        indCorr :    None / np.ndarray
            Alternatively, you can directly pass a (N,) bool array whereN matches the number of channels and True means that a channel should be excluded, thus setting self._indCorr
        Calc :      bool
            Flag indicating whether the calculation should be triggered immediately

        """
        if not indCorr is None:
            assert isinstance(indCorr,np.ndarray) and indCorr.dtype.name=='bool' and indCorr.ndim==1 and indCorr.size==self._NChansRef, "Arg indCorr must be a (N,) np.ndarray of bool !"
            self._indCorr = indCorr
        elif not Val==[]:
            ind = np.zeros((len(LCrit),self._NChansRef),dtype=bool)
            for ii in range(0,len(LCrit)):
                ind[ii,:] = self.select(Val=Val, Crit=LCrit[ii], InOut='In', Out=bool)
            self._indCorr = self._indCorr | np.any(ind,axis=0)


    def Corr_remove(self, Val=[], LCrit=['Name','Cam','CamHead'], Calc=True):
        """ Add channels to the list of channels to be re-inserted as valid channels

        Works like self.In_add() (i.e.: opposite of self.Corr_add())

        Parameters
        ----------
        Val :      list
            Fed to self.select(), list of values for criteria in LCrit that should be used to exclude channels (e.g.: list of channel names of camera names)
        LCrit :     list
            Fed to self.select(), list of criteria against which to select the channels matching the values in Val (should be attributes of :class:`tofu.pathfile.ID` or of its USRdict attribute)
        indCorr :    None / np.ndarray
            Alternatively, you can directly pass a (N,) bool array whereN matches the number of channels and True means that a channel should be excluded, thus setting self._indCorr
        Calc :      bool
            Flag indicating whether the calculation should be triggered immediately

        """
        if not Val==[]:
            ind = np.ones((len(LCrit),self._NChansRef),dtype=bool)
            for ii in range(0,len(LCrit)):
                ind[ii,:] = self.select(Val=Val, Crit=LCrit[ii], InOut='Out', Out=bool)
            self._indCorr = self._indCorr & np.all(ind,axis=0)

    def Corr_list(self, Out='Name'):
        """ Return the list of channel names needing correction

        This lists the channels indicated by self._indOut, populated using self.Out_add() and de-populated using self.In_add().
        The output can be returned as a list of channel Names

        Parameters
        ----------
        Out :       str
            Flag indicating in which form to return the output (fed to :meth:`~tofu.treat.PreData.select()`)

        Returns
        -------
        L :         list
            List of excluded channels in the required form

        """
        L = self.select(Out=Out)
        L = [L[ii] for ii in range(0,len(L)) if self._indCorr[ii]]
        return L

    def set_PhysNoise(self, Method='svd', Modes=range(0,8), DF=None, DFEx=None, Harm=True, HarmEx=True, Deg=0, Nbin=3, LimRatio=0.05, Plot=False):
        """ Use a svd or a fft to estimate the physical part of the signal and the part which can be assimilated to noise, then uses specified degree for polynomial noise model

        This method provides an easy way to compute the noise level on each channel.
        It can be done in 2 different ways:
            - 'svd': you have to provide the mode numbers that you think can be considered as physical, the signal will be re-constructed from these and the rest discarded as noise
            - 'fft': you have to provide the frequency window that you think is physical (optionaly the higher harmonics can be included), the signal is re-constructed via inverse fourier and the rest discarded as noise

        To help you decide which mode numbers of frequency interval to use, you can preliminarily use self.plot_svd() and self.plot_fft() to visualize the decompositions.

        Note : this is only used to compute a noise estimate, stored separately, the total original signal is preserved

        Parameters
        ----------
        Method :      str
            Flag indicating with which method should the noise be estimated ('svd' or 'fft')
        Phys        list
            Modes to be extracted from the svd (default: first 8 modes), use method .plot_svd() to choose the modes
        DF          list
            2 values delimiting a frequency interval (in Hz) from which to extract signal using a fft and rfft
        Harm        bool
            Flag, if True all the available higher harmonics of FreqIn will also be included in the physical signal
        DFEx        list
            2 values delimiting a frequency interval (in Hz) that shall be avoided in the physical signal (relevant if some high harmonics of DF intersect DFEx)
        HarmEx      bool
            Flag, if True all the available higher harmonics of Freqout will also be avoided in the physical signal
        Deg         int
            Degree to be used for the polynomial noise model
        Nbin        int
            Number of bins to be used for evaluating the noise (std) at various signal values
        LimRatio    float
            Ratio ... to be finished...
        Plot        bool
            Flag, if True the histogram of the estimated noise is plotted

        Examples
        --------
        >> obj.set_PhysNoise(Mode='svd', Phys=[0,1,2,3,4,5], Deg=0)
            Will take the first 6 modes of the signal svd and consider as physical, the rest is used to compute a constant (Deg=0) noise estimate on each channel

        """
        if Method=='svd':
            Phys, Noise = _tft_c.SVDExtractPhysNoise(self.data, Modes=Modes)
            Param = {'Modes':Modes}
        elif Method=='fft':
            Phys, Noise = _tft_c.FourierExtract(self.t, self.data, DF=DF, DFEx=DFEx, Harm=Harm, HarmEx=HarmEx, Test=True)
            Param = {'DF':DF, 'DFEx':DFEx, 'Harm':Harm, 'HarmEx':HarmEx}
        self._PhysNoise = {'Method':Method, 'Param':Param, 'Phys':Phys, 'Noise':Noise}
        self._set_NoiseModel(Deg=Deg, Nbin=Nbin, LimRatio=LimRatio, Plot=Plot)


    def _set_NoiseModel(self, Deg=0, Nbin=3, LimRatio=0.05, Plot=False):
        """ Fit the noise as a function of the physical part of the signal by a polynomial, using np.polyfit and the noise level estimated from self.set_PhysNoise()

        After the physical part of the data has been extracted with self.set_PhysNoise(), this function provides tools for estimating how the noise level varies with the signal value (i.e. fixed noise vs signal-dependent noise).
        It fits the noise vs data plot to give a least-square noise model.
        If you want a constant noise model, just use Deg=0.

        Parameters
        ----------
        Deg :       int
            Degree to be used for the polynomial noise model
        Nbin :      int
            Number of bins to be used for evaluating the noise (std) at various signal values
        LimRatio :  float
            Ratio ...
        Plot :      bool
            Flag, if True the histogram of the estimated noise is plotted

        """
        NoiseModel = {}
        NoiseModel['Par'], NoiseModel['Noise'], NoiseModel['Coefs'] = _tft_c._PreData_set_NoiseModel(self._PhysNoise, self.Chans, len(self.Chans), self.t.size, Deg, Nbin, LimRatio, Plot=Plot)
        self._NoiseModel = NoiseModel

    def plot(self, a4=False):
        """ Plot the signal in an interactive window, no arguments needed

        Plot an interactive matplotlib window to explore the data

        Parameters
        ----------
        a4 :        bool
            Flag indicating whether the figure should be the size of a a4 sheet of paper (to facilitate printing)

        Returns
        -------
        Lax :   list
            List of plt.Axes on which the plots are made

        """
        Lax = _tft_p.Plot_Signal(self.data, self.t, self.Chans, nMax=4, shot=self.shot, a4=a4)
        return Lax


    def plot_svd(self, Modes=10, NRef=None, a4=False, Test=True):
        """ Plot the chosen modes (topos and chronos) of the svd of the data, and the associated spectrum on a separate figure

        Performs a svd of the data and plots the singular values, the temporal and spacial modes

        Paramaters
        ----------
        Modes :     int / iterable
            Index of the modes to be plotted, the modes and sorted in decreasing order of singular value
                - int : plots all modes in range(0,Modes)
                - iterable : plots all modes whose index is contained in Modes
        NRef :      None
            Number of columns in the plot, if None set to len(Modes)/2 (i.e.: 2 modes plotted per axes)
        a4 :        bool
            Flag indicating whether the figure should be the size of a a4 sheet of paper (to facilitate printing)
        Test :      bool
            Flag indicating whether the inputs should be tested for conformity

        Returns
        -------
        Lax :   list
            List of plt.Axes on which the plots were made

        """
        Lax = _tft_p.SVDNoisePlot(self.data, t=self.t, Modes=Modes, NRef=NRef, a4=a4, Test=Test)
        return Lax


    def plot_fft(self, Val=None, Crit='Name', V='simple', tselect=None, Fselect=None, PreExp=None, PostExp=None, Log='any', InOut='In', SpectNorm=True, DTF=None, RatDef=100., Inst=True, MainF=True,
                 ylim=(None,None), cmap=plt.cm.gray_r, a4=False):
        """ Plot the power spectrum (fft) of the chosen signals

        Computes the fft of the data and plots the power spectrum, normalized or not, for the chosen channels

        Parameters Val, Crit, PreExp, PostExp, Log and InOut are for channel selection and are fed to :meth:`~tofu.treat.PreData.select()`

        Parameters
        ----------
        V :         str
            Flag indicating whether the plot should be interactive, values in ['simple','inter']
        tselect :   None /

        Fselect :   None /

        SpectNorm : bool
            Flag, if True the power spectrum is normalised to its maximum at each time step (default: True)
        DTF :       float
            Size (in seconds) of the running time window to be used for the windowed fft
        RatDef :    float
            Used if DTF not provided, the number by which the total signal duration is divided to get a time window
        Inst :      bool
            Flag, if true, the average of the signal is substracted at each time step to emphasize high frequencies (higher than the one associated to the running time window, default: True)
        MainF :     bool
            Flag
        ylim :      tuple
            Each limit which is not None is fed to plt.Axes.set_ylim()
        a4 :        bool
            Flag, if true the figure is sized so as to fill a a4 paper sheet

        Returns
        -------
        Lax :       list
            List of plt.Axes on which the plots were made

        """
        ind = self.select(Val=Val, Crit=Crit, PreExp=PreExp, PostExp=PostExp, Log=Log, InOut=InOut, Out=int, ToIn=True)
        Pow, MainFreq, Freq = [None for ii in range(0,ind.size)], [None for ii in range(0,ind.size)], [None for ii in range(0,ind.size)]
        v = 2 if Inst else 1
        for ii in range(0,ind.size):
            Pow[ii], MainFreq[ii], Freq[ii] = _tft_c.Fourier_MainFreqPowSpect(self.data[:,ind[ii]], self.t, DTF=DTF, RatDef=RatDef, Method='Max', Trunc=0.60, V=v, Test=True)
        Lax = _tft_p._PreData_plot_fft(self.Chans, V, ind, Pow, MainFreq, Freq, self.t, SpectNorm, cmap, ylim, tselect, Fselect, MainF=MainF, a4=False)
        return Lax

    def _plot_NoiseVSPhys(self, Val=None, Crit='Name', PreExp=None, PostExp=None, Log='any', a4=False):
        ind = self.select(Val=Val, Crit=Crit, PreExp=PreExp, PostExp=PostExp, Log=Log, Out=int, ToIn=True)
        LNames = self.select(Val=Val, Crit=Crit, PreExp=PreExp, PostExp=PostExp, Log=Log, Out='Name', ToIn=True)
        Coefs = self._NoiseCoefs[:,ind] if self._NoiseCoefs.ndim==2 else self._NoiseCoefs[ind]
        return Plot_Noise(self._PhysNoise['Phys'][:,ind], self._PhysNoise['Noise'][:,ind], self._NoiseCoefs[:,ind], LNames, self._NoiseModel['Deg'], a4=a4)

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













def _PreData_check_inputs(Id=None, data=None, t=None, Chans=None, LIdDet=None, DtRef=None, Exp=None, shot=None, Diag=None, dtime=None, dtimeIn=None, SavePath=None,
                          Dt=None, Resamp_t=None, Resamp_f=None, Resamp_Method=None, Resamp_interpkind=None, Calc=None):
    if not Id is None:
        assert type(Id) in [str,tfpf.ID], "Arg Id must be a str or a tfpf.ID object !"
    if not data is None:
        assert type(data) is np.ndarray and data.ndim==2, "Arg data must be a 2D np.ndarray !"
    if not t is None:
        assert type(t) is np.ndarray and t.ndim==1 and np.all(t==np.unique(t)), "Arg t must be a 1D np.ndarray of increasing values !"
        if not data is None:
            assert t.size==data.shape[0], "Shapes of data and t are not matching !"
    if not Chans is None:
        assert hasattr(Chans,'__iter__') and all([type(ss) is str for ss in Chans]), "Arg Chans must be an iterable of str !"
        if not data is None:
            assert len(Chans)==data.shape[1], "Shapes of data and Chans are not matching !"
    if not LIdDet is None:
        assert hasattr(LIdDet,'__iter__') and all([type(ss) is tfpf.ID for ss in LIdDet]), "Arg LIdDet must be an iterable of :class:`~tofu.pathfile.ID` !"
        if not data is None:
            assert len(LIdDet)==data.shape[1], "Shapes of data and LIdDet are not matching !"
    bools = [dtimeIn,Calc]
    if any([not aa is None for aa in bools]):
        assert all([aa is None or type(aa) is bool for aa in bools]), " Args [dtimeIn] must all be bool !"
    if not Exp is None:
        assert Exp in tfd.AllowedExp, "Ar Exp must be in "+str(tfd.AllowedExp)+" !"
    strs = [SavePath]
    if any([not aa is None for aa in strs]):
        assert all([aa is None or type(aa) is str for aa in strs]), "Args [Type,Exp,SavePath] must all be str !"
    Iter2 = [DtRef,Dt]
    if any([not aa is None for aa in Iter2]):
        assert all([aa is None or (hasattr(aa,'__iter__') and np.asarray(aa).ndim==1 and len(aa)==2) for aa in Iter2]), "Args [DtRef] must be an iterable with len()=2 !"
    Ints = [shot]
    if any([not aa is None for aa in Ints]):
        assert all([aa is None or type(aa) is int for aa in Ints]), "Args [shot] must be int !"
    if not dtime is None:
        assert type(dtime) is dtm.datetime, "Arg dtime must be a dtm.datetime !"

    if not Resamp_t is None:
        assert type(Resamp_t) is np.ndarray and Resamp_t.ndim==1 and np.unique(Resamp_t==np.unique(Resamp_t)), "Arg Resamp_t must be a 1D np.ndarray of increasing values !"
    if not Resamp_f is None:
        assert type(Resamp_f) in [int,float,np.float64] and Resamp_f>0, "Arg Resamp_f must be a strictly positive value !"
    if not Resamp_Method is None:
        assert type(Resamp_Method) is str and Resamp_Method in ['interp','movavrg'], "Arg Resamp_Method must be in ['interp','movavrg'] !"




