

import matplotlib.pyplot as plt


from ._DataCollection_class0_Base import DataCollection0
from . import _DataCollection_interactivity
from . import _DataCollection_plot


__all__ = ['DataCollection1']    # , 'TimeTraceCollection']


class DataCollection1(DataCollection0):
    """ Handles matplotlib interactivity """

    _LPAXES = ['ax', 'type']
    _dinteractivity = dict.fromkeys(['curax_panzoom'])

    # ----------------------
    #   Add objects
    # ----------------------

    def add_axes(
        self,
        key=None,
        handle=None,
        type=None,
        refx=None,
        refy=None,
        **kwdargs,
    ):
        super().add_obj(
            which='axes',
            key=key,
            handle=handle,
            type=type,
            refx=refx,
            refy=refy,
            **kwdargs,
        )

        # add canvas if not already stored
        if 'canvas' not in self._dobj.keys():
            self.add_canvas(handle=ax.figure.canvas)
        else:
            lisin = [
                k0 for k0, v0 in self._dobj['canvas'].items()
                if v0['handle'] == ax.figure.canvas
            ]
            if len(lisin) == 0:
                self.add_canvas(handle=ax.figure.canvas)

    def add_canvas(self, key=None, handle=None):
        """ Add canvas and interactivity obj """
        interactive = (
            hasattr(handle, 'toolbar')
            and handle.toolbar is not None
        )
        self.add_obj(
            which='canvas',
            key=key,
            handle=handle,
            interactive=interactive,
        )

    def add_mobile(
        self,
        key=None,
        handle=None,
        ref=None,
        data=None,
        visible=None,
        **kwdargs,
    ):
        super().add_obj(
            which='mobile',
            key=key,
            handle=handle,
            ref=ref,
            data=data,
            visible=visible,
            **kwdargs,
        )

    # ------------------
    # Properties
    # ------------------

    @property
    def dax(self):
        return self.dobj.get('axes', {})

    @property
    def dinteractivity(self):
        return self._dinteractivity

    # ------------------
    # Setup interactivity
    # ------------------

    def setup_interactivity(
        self,
        kinter=None,
        dgroup=None,
    ):
        """

        dgroup = {
            'group0': {
                'ref': ['ref0', 'ref1', ...],
                'nmax': 3,
                'colors': ['r', 'g', 'b'],
            }
        }

        """

        # Check dgroup

        c0 = (
            isinstance(dgroup, dict)
            and all([
                isinstance(k0, str)
                and isinstance(v0, dict)
                and isinstance(v0.get('ref'), list)
                and all([ss in self._dref.keys() for ss in v0['ref']])
            ])
        )
        if not c0:
            msg = "Arg dgroup must be a dict of the form:\n"
            raise Exception(msg)

        for k0, v0 in dgroup.items():
            if v0.get('nmax') is None:
                dgroup[k0]['nmax'] = 1

        # group, ref, nmax
        lgroup = sorted(dgroup.keys())
        ngroup = len(lgroup)
        nmax = np.array([dgroup[k0]['nmax'] for k0 in lgroup])
        nmaxcur = np.zeros((ngroup,), dtype=int)
        indcur = np.zeros((ngroup,), dtype=int)

        # cumsum0 = np.r_[0, np.cumsum(nmax[1, :])]
        # arefind = np.zeros((np.sum(nmax[1, :]),), dtype=int)

        # dind = {
            # 'lrefid': lref,
            # 'nmax': nmax,
            # 'arefind': arefind,
            # 'cumsum0': cumsum0,
        # }

        # ---------
        # dkeys
        dkeys = {
            'control': {'val': False},
            'ctrl': {'val': False},
            'shift': {'val': False},
        }

        # ---------
        # dinter
        dinter = {
            'dkeys': dkeys,
            'dgroup': dgroup,
            'lgroup': lgroup,
            'nmax': nmax,
            'nmaxcur': nmaxcur,
            'indcur': indcur,
            'current_ax': None,
            'current_group': None,
            'current_ref': None,
            'current_ind': None,
            'follow': False,
        }

        self.add_obj(
            which='interactivity',
            key='inter0',
            **dinter,
        )

    # ----------------------------
    # Ensure connectivity possible
    # ----------------------------

    def _warn_ifnotInteractive(self):
        warn = False

        c0 = (
            len(self._dobj.get('axes', {})) > 0
            and len(self._dobj.get('canvas', {})) > 0
        )
        if c0:
            dcout = {
                k0: v0['handle'].__class__.__name__
                for k0, v0 in self._dobj['canvas'].items()
                if not v0['interactive']
            }
            if len(dcout) > 0:
                lstr = '\n'.join(
                    [f'\t- {k0}: {v0}' for k0, v0 in dcout.items()]
                )
                msg = (
                    "Non-interactive backends identified (prefer Qt5Agg):\n"
                    f"\t- backend : {plt.get_backend()}\n"
                    f"\t- canvas  :\n{lstr}"
                )
                warn = True
        else:
            msg = ("No available axes / canvas for interactivity")
            warn = True

        # raise warning
        if warn:
            warnings.warn(msg)

        return warn

    # ----------------------
    # Connect / disconnect
    # ----------------------

    def connect(self):
        if self._warn_ifnotInteractive():
            return
        for k0, v0 in self._dobj['canvas'].items():
            # keyp = v0['handle'].mpl_connect('key_press_event', self.onkeypress)
            # keyr = v0['handle'].mpl_connect('key_release_event', self.onkeypress)
            butp = v0['handle'].mpl_connect('button_press_event', self.mouseclic)
            # res = v0['handle'].mpl_connect('resize_event', self.resize)
            #butr = self.can.mpl_connect('button_release_event', self.mouserelease)
            #if not plt.get_backend() == "agg":
            # v0['handle'].manager.toolbar.release = self.mouserelease

            self._dobj['canvas'][k0]['cid'] = {
                # 'keyp': keyp,
                # 'keyr': keyr,
                'butp': butp,
                # 'res': res,
                # 'butr': butr,
            }

    def disconnect(self):
        if self._warn_ifnotInteractive():
            return
        for k0, v0 in self._dobj['canvas'].items():
            for k1, v1 in v0['cid'].items():
                v0['handle'].mpl_disconnect(v1)
            v0['handle'].manager.toolbar.release = lambda event: None

    # ------------------------------------
    # Interactivity handling - preliminary
    # ------------------------------------

    def _getset_current_axref(self, event):
        # Check click is relevant
        c0 = event.inaxes is not None and event.button == 1
        if not c0:
            raise Exception("clic not in axes")

        # get current ax key
        lkax = [
            k0 for k0, v0 in self._dobj['axes'].items()
            if v0['handle'] == event.inaxes
        ]
        kax = _generic_check._check_var(
            None, 'kax',
            types=str,
            allowed=lkax,
        )
        ax = self._dobj['axes'][kax]['handle']

        # Get current group and ref
        groupx = self._dobj['axes'][kax]['groupx']
        groupy = self._dobj['axes'][kax]['groupy']
        refx = self._dobj['axes'][kax]['refx']
        refy = self._dobj['axes'][kax]['refy']

        current_group = self._dobj['interactivity'][kinter]['current_group']
        if current_group in groupx + groupy:
            pass
        else:
            pass

        # Check axes is relevant and toolbar not active
        c_activeax = 'fix' not in self.dobj['axes'][kax].keys() # TBC
        c_toolbar = not self.can.manager.toolbar.mode
        if not all([c_activeax, c_toolbar]):
            raise Exception("Not usable axes!")

        # get interactivity key
        kinter = list(self._dobj['interactivity'].keys())[0]

        # get current group
        dgroup = self._dobj['interactivity'][kinter]['dgroup']
        # group = self._dobj['interactivity'][kinter]['group']
        lgroup = [
            k0 for k0, v0['ref'] in dgroup.items()
            if any([rr in v0['ref'] for rr in self._dobj['axes'][kax]['ref']])
        ]

        # current group not ok?
        if group not in lgroup:
            if len(lgroup) == 1:
                group = lgroup[0]
            else:
                import pdb; pdb.set_trace()     # DB
                pass

        # get current ref
        lref = [
            rr for rr in self._dobj['axes'][kax]['ref']
            if rr in dgroup[group]['ref']
        ]
        # lref = list(self.dax[event.inaxes]['graph'].keys())
        if len(lref) == 1:
            ref = lref[0]
        else:
            import pdb; pdb.set_trace()     # DB
            pass

        # Update interactivity dict
        self._dobj['interactivity'][kinter]['ax_panzoom'] = kax
        self._dobj['interactivity'][kinter]['ax'] = kax
        self._dobj['interactivity'][kinter]['group'] = group
        self._dobj['interactivity'][kinter]['ref'] = ref
        # self._dobj['interactivity'][kinter]['nind'] = ref
        # self._dobj['interactivity'][kinter]['ind'] = ref
        # self._dobj['interactivity'][kinter]['value'] = ref

        self.kinter = kinter
        self._dobj['interactivity'][kinter].update({
            'current_ax': kax,
            'current_group': group,
            'current_ref': ref,
        })

    # -----------------------------
    # Interactivity: generic update
    # -----------------------------

    def update_interactivity(self, ref_update=None, excluderef=True):
        """ Called at each event """

        # Propagate indices through refs
        for rr in ref_update:
            lref = None
            ldata = None
            self.propagate_indices_per_ref(ref=rr, lref=lref, ldata=ldata)

        # Set visibility of mobile objects
        for k0, v0 in self._dobj['mobile'].items():
            lref = v0['ref']
            vis = None
            self._dobj['mobile'][k0]['visible'] = vis

        # Set list of mobile objects to be updated

        # self._update_dcur() # 0.4 ms
        # self._update_dref(excluderef=excluderef)    # 0.1 ms
        self._update_mobiles(lmobiles=lmobiles) # 0.2 s

    def _update_dcur(self):
        """ Called at each event """

        # Update also dind !
        an = [
            self.dgroup[self.dref[rid]['group']]['ncur']
            for rid in self.dind['lrefid']
        ]
        self.dind['anMaxcur'][0, :] = an

        # Update array ncur
        for obj in self.dgroup[group]['lobj']:
            a0 = self.dind['anMaxcur'][0, self.dobj[obj]['indncurind']]
            a1 = self.dobj[obj]['aindvis']
            self.dobj[obj]['vis'] = np.all( a0 >= a1 )

    def _update_dref(self, excluderef=True):
        """   """
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


    def _update_mobiles(self):

        # --- Prepare ----- 2 us
        group = self.dcur['group']
        refid = self.dcur['refid']
        indcur = self.dgroup[group]['indcur']
        lax = self.dgroup[group]['lax']

        # ----- Get list of canvas and axes to be updated -----
        lax = None
        lcan = None

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
        for k0, v0 in self._dobj['mobile'].items():
            v0['handle'].set_visible(v0['visible'])
            self._dobj['axes'][v0['ax']]['handle'].draw_artist(v0['handle'])

        # ---- blit axes ------ 5 ms
        for aa in lax:
            self.can.blit(aa.bbox)

    # ----------------------
    # Interactivity: mouse
    # ----------------------

    def mouseclic(self, event):

        # Check click is relevant
        c0 = event.button == 1
        if not c0:
            return

        # get / set cuurent interactive usabel axes and ref
        try:
            self._getset_current_axref(event)
        except Exception as err:
            # warnings.warn(str(err))
            return

        kinter = self.kinter
        kax = self._dobj['interactivity'][kinter]['current_axe']
        refx = self._dobj['axes'][kax]['refx']
        refy = self._dobj['axes'][kax]['refy']
        if refx is None and refy is None:
            return

        dkeys = self._dobj['interactivity'][kinter]['dkeys']
        shift = dkeys['shift']['val']
        ctrl = dkeys['control']['val'] or dkeys['ctrl']['val']

        # Update number of indices (for visibility)
        for ii, (rr, gg) in [(refx, 'groupx'), (refy, 'groupy')]:
            if rr is not None:
                _DataCollection_interactivity._update_indices_nb(
                    group=self._dobj['axes'][kax][gg],
                    dinter=self._dobj['interactivity'][kinter],
                    ctrl=ctrl,
                    shift=shift,
                )

        # Update ref indices

        if refx is not None and refy is not None:
            kx = self._dobj['axes'][kax]['datax']
            ky = self._dobj['axes'][kax]['datay']
            datax = self._ddata[kx]['data']
            datay = self._ddata[ky]['data']
            deltax = None
            deltay = None
            dist = np.sqrt()

            ind = np.nanargmin(dist)

        elif refx is not None:
            bins =
            ind =
            self._dref[refx]['indices'][iix] = ind

        elif refy is not None:
            pass

        # Update ref indices
        if self._dobj['interactivity'][kinter]['follow']:
            self._dref[refx]['indices'][iix:] = indx
        else:
            self._dref[refx]['indices'][iix] = indx

        # Update group val
        # val = (event.xdata, event.ydata)
        # if self._dobj['interactivity'][kinter]['follow']:
            # dgroup[group]['valind'][ii:, :] = val
        # else:
            # dgroup[group]['valind'][ii, :] = val

        self.update_interactivity(excluderef=False)

    # -------------------
    # Generic plotting
    # -------------------

    def plot_as_array(
        self,
        key=None,
        ind=None,
        vmin=None,
        vmax=None,
        cmap=None,
        aspect=None,
        dax=None,
        dmargin=None,
        fs=None,
        dcolorbar=None,
        dleg=None,
        connect=None,
    ):
        """ Plot the desired 2d data array as a matrix """
        return _DataCollection_plot.plot_as_array(
            coll=self,
            key=key,
            ind=ind,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            aspect=aspect,
            dax=dax,
            dmargin=dmargin,
            fs=fs,
            dcolorbar=dcolorbar,
            dleg=dleg,
            connect=connect,
        )

    def _plot_timetraces(self, ntmax=1,
                         key=None, ind=None, Name=None,
                         color=None, ls=None, marker=None, ax=None,
                         axgrid=None, fs=None, dmargin=None,
                         legend=None, draw=None, connect=None, lib=None):
        plotcoll = self.to_PlotCollection(ind=ind, key=key,
                                          Name=Name, dnmax={})
        return _DataCollection_plot.plot_DataColl(
            plotcoll,
            color=color, ls=ls, marker=marker, ax=ax,
            axgrid=axgrid, fs=fs, dmargin=dmargin,
            draw=draw, legend=legend,
            connect=connect, lib=lib,
        )

    def _plot_axvlines(
        self,
        which=None,
        key=None,
        ind=None,
        param_x=None,
        param_txt=None,
        sortby=None,
        sortby_def=None,
        sortby_lok=None,
        ax=None,
        ymin=None,
        ymax=None,
        ls=None,
        lw=None,
        fontsize=None,
        side=None,
        dcolor=None,
        dsize=None,
        fraction=None,
        figsize=None,
        dmargin=None,
        wintit=None,
        tit=None,
    ):
        """ plot rest wavelengths as vertical lines """

        # Check inputs
        which, dd = self.__check_which(
            which=which, return_dict=True,
        )
        key = self._ind_tofrom_key(which=which, key=key, ind=ind, returnas=str)

        if sortby is None:
            sortby = sortby_def
        if sortby not in sortby_lok:
            msg = (
                """
                For plotting, sorting can be done only by:
                {}

                You provided:
                {}
                """.format(sortby_lok, sortby)
            )
            raise Exception(msg)

        return _DataCollection_plot.plot_axvline(
            din=dd,
            key=key,
            param_x='lambda0',
            param_txt='symbol',
            sortby=sortby, dsize=dsize,
            ax=ax, ymin=ymin, ymax=ymax,
            ls=ls, lw=lw, fontsize=fontsize,
            side=side, dcolor=dcolor,
            fraction=fraction,
            figsize=figsize, dmargin=dmargin,
            wintit=wintit, tit=tit,
        )
