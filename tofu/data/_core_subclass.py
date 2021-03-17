


from ._core_new import DataCollection



#############################################
#############################################
#       Spectral Lines
#############################################


class SpectralLines(DataCollection):

    _forced_group = ['Te', 'ne']
    _data_none = True









#############################################
#############################################
#       Time traces
#############################################


# TBC
class TimeTraces(DataCollection):
    """ A generic class for handling multiple time traces """

    _forced_group = 'time'
    _dallowed_params = {'time':{'origin': (str, 'unknown'),
                                'dim':    (str, 'time'),
                                'quant':  (str, 't'),
                                'name':   (str, 't'),
                                'units':  (str, 's')}}
    _plot_vignettes = False


    def fit(self, ind=None, key=None,
            Type='staircase', func=None,
            plot=True, fs=None, ax=None, draw=True, **kwdargs):
        """  Fit the times traces with a model

        Typically try to fit plateaux and ramps i.e.: Type = 'staircase')
        Return a dictionary of the fitted parameters, ordered by data key

        """

        dout = self._fit_one_dim(ind=ind, key=key, group=self._forced_group,
                                 Type=Type, func=func, **kwdargs)
        if plot:
            kh = _plot_new.plot_fit_1d(self, dout)
        return dout

    def add_plateaux(self, verb=False):
        dout = self.fit(ind=ind, key=key, group=group,
                       Type='staircase')

        # Make Pandas Dataframe attribute
        self.plateaux = None
        if verb:
            msg = ""




    def plot(self, **kwdargs):
        return self._plot_timetraces(**kwdargs)

    def plot_incremental(self, key=None, ind=None,
                         plateaux=True, connect=True):
        return

    def plot_plateau_validate(self, key=None, ind=None):
        return
