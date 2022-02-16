

import numpy as np
import matplotlib.pyplot as plt


from . import _generic_check
from ._DataCollection_class0_Base import DataCollection0
from . import _DataCollection_interactivity
from . import _DataCollection_comp
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
        datax=None,
        datay=None,
        bck=None,
        lmobile=None,
        **kwdargs,
    ):

        # ----------------
        # check refx, refy

        if refx is None and refy is None:
            msg = "Please provide at leats refx or refy"
            raise Exception(msg)

        if isinstance(refx, str):
            refx = [refx]
        if isinstance(refy, str):
            refy = [refy]

        c0 =(
            isinstance(refx, list)
            and all([rr in self._dref.keys() for rr in refx])
        )
        if refx is not None and not c0:
            msg = "Arg refx must be a list of valid ref keys!"
            raise Exception(msg)

        c0 =(
            isinstance(refy, list)
            and all([rr in self._dref.keys() for rr in refy])
        )
        if refy is not None and not c0:
            msg = "Arg refy must be a list of valid ref keys!"
            raise Exception(msg)

        # data
        if datax is None and refx is not None:
            datax = ['index' for rr in refx]
        if datay is None and refy is not None:
            datay = ['index' for rr in refy]

        super().add_obj(
            which='axes',
            key=key,
            handle=handle,
            type=type,
            refx=refx,
            refy=refy,
            datax=datax,
            datay=datay,
            bck=bck,
            lmobile=lmobile,
            **kwdargs,
        )

        # add canvas if not already stored
        if 'canvas' not in self._dobj.keys():
            self.add_canvas(handle=handle.figure.canvas)
        else:
            lisin = [
                k0 for k0, v0 in self._dobj['canvas'].items()
                if v0['handle'] == handle.figure.canvas
            ]
            if len(lisin) == 0:
                self.add_canvas(handle=handle.figure.canvas)

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
        ax=None,
        **kwdargs,
    ):
        super().add_obj(
            which='mobile',
            key=key,
            handle=handle,
            ref=ref,
            data=data,
            visible=visible,
            ax=ax,
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

        # ----------
        # Check dgroup

        c0 = (
            isinstance(dgroup, dict)
            and all([
                isinstance(k0, str)
                and isinstance(v0, dict)
                and isinstance(v0.get('ref'), list)
                and isinstance(v0.get('data'), list)
                and len(v0['ref']) == len(v0['data'])
                and all([ss in self._dref.keys() for ss in v0['ref']])
                for k0, v0 in dgroup.items()
            ])
        )
        if not c0:
            msg = "Arg dgroup must be a dict of the form:\n"
            raise Exception(msg)

        for k0, v0 in dgroup.items():
            if v0.get('nmax') is None:
                dgroup[k0]['nmax'] = 1

        # ----------------------------
        # make sure all refs are known

        drefgroup = dict.fromkeys(self._dref.keys())
        for k0, v0 in self._dref.items():
            lg = [k1 for k1, v1 in dgroup.items() if k0 in v1['ref']]
            if len(lg) > 1:
                msg = f"Ref {k0} has no/several groups!\n\t- found: {lg}"
                raise Exception(msg)
            elif len(lg) == 0:
                lg = [None]
            drefgroup[k0] = lg[0]

            #  add indices
            if lg[0] is not None:
                self.add_indices_per_ref(
                    indices=np.zeros((dgroup[lg[0]]['nmax'],), dtype=int),
                    ref=k0,
                    distribute=False,
                )

        self.add_param(which='ref', param='group')
        self.set_param(which='ref', param='group', value=drefgroup)

        # ----------------------------
        # add indices to ref



        # ------------------------------
        # update dax with groupx, groupy

        daxgroupx = dict.fromkeys(self._dobj['axes'].keys())
        daxgroupy = dict.fromkeys(self._dobj['axes'].keys())
        for k0, v0 in self._dobj['axes'].items():
            if v0['refx'] is None:
                daxgroupx[k0] = None
            else:
                daxgroupx[k0] = [
                    self._dref[k1]['group'] for k1 in v0['refx']
                ]
            if v0['refy'] is None:
                daxgroupy[k0] = None
            else:
                daxgroupy[k0] = [
                    self._dref[k1]['group'] for k1 in v0['refy']
                ]

        self.add_param(which='axes', param='groupx')
        self.add_param(which='axes', param='groupy')
        self.set_param(which='axes', param='groupx', value=daxgroupx)
        self.set_param(which='axes', param='groupy', value=daxgroupy)

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

        # -----
        # axes

        for k0, v0 in self._dobj['axes'].items():
            self._dobj['axes'][k0]['lmob'] = [
                k1 for k1, v1 in self._dobj['mobile'].items()
                if v1['ax'] == k0
            ]
            if v0['refx'] is not None:
                for ii, rr in enumerate(v0['refx']):
                    if v0['datax'][ii] is None:
                        self._dobj['axes'][k0]['datax'][ii] = 'index'

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
            'cur_ax': None,
            'cur_groupx': None,
            'cur_groupy': None,
            'cur_refx': None,
            'cur_refy': None,
            'cur_ix': None,
            'cur_iy': None,
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

        # Get kinter
        kinter = list(self._dobj['interactivity'].keys())[0]
        dgroup = self._dobj['interactivity'][kinter]['dgroup']

        # Get current groups
        cur_groupx = self._dobj['interactivity'][kinter]['cur_groupx']
        cur_groupy = self._dobj['interactivity'][kinter]['cur_groupy']
        if cur_groupx in groupx + groupy:
            pass
        else:
            cur_groupx = groupx[0]
        if cur_groupy in groupx + groupy:
            pass
        else:
            cur_groupy = groupy[0]

        # get current refs
        cur_refx = self._dobj['interactivity'][kinter]['cur_refx']
        cur_refy = self._dobj['interactivity'][kinter]['cur_refy']
        if cur_refx in dgroup[cur_groupx]['ref']:
            pass
        else:
            cur_refx = dgroup[cur_groupx]['ref'][0]
        if cur_refy in dgroup[cur_groupy]['ref']:
            pass
        else:
            cur_refy = dgroup[cur_groupy]['ref'][0]

        # data
        ix = self._dobj['axes'][kax]['refx'].index(cur_refx)
        cur_datax = self._dobj['axes'][kax]['datax'][ix]
        iy = self._dobj['axes'][kax]['refy'].index(cur_refy)
        cur_datay = self._dobj['axes'][kax]['datay'][iy]

        # cur_ind
        ix = self._dobj['interactivity'][kinter]['lgroup'].index(cur_groupx)
        cur_ix = self._dobj['interactivity'][kinter]['indcur'][ix]
        iy = self._dobj['interactivity'][kinter]['lgroup'].index(cur_groupy)
        cur_iy = self._dobj['interactivity'][kinter]['indcur'][iy]

        # Check axes is relevant and toolbar not active
        lc = [
            all([
                'fix' not in v0.keys()
                for v0 in self._dobj['axes'].values()
            ]),
            all([
                not v0['handle'].manager.toolbar.mode
                for v0 in self._dobj['canvas'].values()
            ]),
        ]
        if not all(lc):
            raise Exception("Not usable axes!")

        # Update interactivity dict
        self.kinter = kinter
        self._dobj['interactivity'][kinter].update({
            'cur_ax': kax,
            'ax_panzoom': kax,
            'cur_groupx': cur_groupx,
            'cur_groupy': cur_groupy,
            'cur_refx': cur_refx,
            'cur_refy': cur_refy,
            'cur_datax': cur_datax,
            'cur_datay': cur_datay,
            'cur_ix': cur_ix,
            'cur_iy': cur_iy,
        })

    # -----------------------------
    # Interactivity: generic update
    # -----------------------------

    def update_interactivity(
        self,
        cur_groupx=None,
        cur_groupy=None,
        cur_refx=None,
        cur_refy=None,
        cur_datax=None,
        cur_datay=None,
        excluderef=True,
    ):
        """ Called at each event """

        dgroup = self._dobj['interactivity'][self.kinter]['dgroup']

        # Propagate indices through refs
        if cur_refx is not None:
            lref = dgroup[cur_groupx]['ref']
            ldata = dgroup[cur_groupx]['data']
            self.propagate_indices_per_ref(
                ref=cur_refx,
                lref=[rr for rr in lref if rr != cur_refx],
                ldata=[cur_datax] + [dd for dd in ldata if dd != cur_datax],
                param=None,
            )

        if cur_refy is not None:
            lref = dgroup[cur_groupy]['ref']
            ldata = dgroup[cur_groupy]['data']
            self.propagate_indices_per_ref(
                ref=cur_refy,
                lref=[rr for rr in lref if rr != cur_refy],
                ldata=[cur_datay] + [dd for dd in ldata if dd != cur_datay],
                param=None,
            )

        # Set visibility of mobile objects - TBF/TBC
        for k0, v0 in self._dobj['mobile'].items():
            lref = v0['ref']
            vis = None
            self._dobj['mobile'][k0]['visible'] = vis

        # Set list of mobile objects to be updated

        # self._update_dcur() # 0.4 ms
        # self._update_dref(excluderef=excluderef)    # 0.1 ms
        self._update_mobiles(lmobiles=lmobiles) # 0.2 s

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

    def resize(self, event):
        self._set_dbck(self.dax.keys())

    def _set_dbck(self, lax):
        # Make all invisible
        for ax in lax:
            for obj in self.dax[ax]['lobj']:
                obj.set_visible(False)

        # Draw and reset Bck
        self.can.draw()
        for ax in lax:
            #ax.draw(self.can.renderer)
            self.dax[ax]['bck'] = self.can.copy_from_bbox(ax.bbox)

        # Redraw
        for kax in lax:
            for obj in self._dobj['axes'][kax]['lobj']:
                obj.set_visible(self._obj['mobile'][obj]['vis'])
                #ax.draw(self.can.renderer)
        self.can.draw()

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
            raise err
            # warnings.warn(str(err))
            return

        kinter = self.kinter
        kax = self._dobj['interactivity'][kinter]['cur_ax']

        refx = self._dobj['axes'][kax]['refx']
        refy = self._dobj['axes'][kax]['refy']
        if refx is None and refy is None:
            return

        cur_groupx = self._dobj['interactivity'][kinter]['cur_groupx']
        cur_groupy = self._dobj['interactivity'][kinter]['cur_groupy']
        cur_refx = self._dobj['interactivity'][kinter]['cur_refx']
        cur_refy = self._dobj['interactivity'][kinter]['cur_refy']

        dkeys = self._dobj['interactivity'][kinter]['dkeys']
        shift = dkeys['shift']['val']
        ctrl = dkeys['control']['val'] or dkeys['ctrl']['val']

        # Update number of indices (for visibility)
        for gg in [cur_groupx, cur_groupy]:
            _DataCollection_interactivity._update_indices_nb(
                group=gg,
                dinter=self._dobj['interactivity'][kinter],
                ctrl=ctrl,
                shift=shift,
            )

        # Update ref indices
        if cur_refx is not None:
            cur_datax = self._dobj['interactivity'][kinter]['cur_datax']
            if cur_datax != 'index':
                cur_datax = self._ddata[cur_datax]['data']

        if cur_refy is not None:
            cur_datay = self._dobj['interactivity'][kinter]['cur_datay']
            if cur_datay != 'index':
                cur_datay = self._ddata[cur_datay]['data']

        if cur_refx is not None and cur_refy is not None:
            c0 = (
                'index' in [cur_datax, cur_datay]
                or ((cur_refx == cur_refy) == (cur_datax == cur_datay))
            )
            if not c0:
                msg = (
                    "Invalid ref / data pairs:\n"
                    f"\t- cur_refx, cur_refy: {cur_refx}, {cur_refy}\n"
                    f"\t- cur_datax, cur_datay: {cur_datax}, {cur_datay}"
                )
                raise Exception(msg)

        if None not in [cur_refx, cur_refy] and cur_refx == cur_refy:

            dist = (cur_datax - event.xdata)**2 + (cur_datay - event.ydata)**2
            lind = [
                np.nanargmin(dist, axis=ii) for ii in range(datax.ndim)
            ]

        else:

            if cur_refx is not None:
                monot = None
                if cur_datax != 'index':
                    monot = self._ddata[cur_datax]['monot'] == (True,)
                    cur_datax = self._ddata[cur_datax]['data']
                ix = _DataCollection_comp._get_index_from_data(
                    data=cur_datax,
                    data_pick=np.r_[event.xdata],
                    monot=monot,
                )[0]

            if cur_refy is not None:
                monot = None
                if cur_datay != 'index':
                    monot = self._ddata[cur_datay]['monot'] == (True,)
                    cur_datay = self._ddata[cur_datay]['data']
                iy = _DataCollection_comp._get_index_from_data(
                    data=cur_datay,
                    data_pick=np.r_[event.ydata],
                    monot=monot,
                )[0]

        # Update ref indices
        if cur_refx is not None:
            cur_ix = self._dobj['interactivity'][kinter]['cur_ix']
            if self._dobj['interactivity'][kinter]['follow']:
                self._dref[cur_refx]['indices'][cur_ix:] = ix
            else:
                self._dref[cur_refx]['indices'][cur_ix] = ix

        # Update ref indices
        if cur_refy is not None:
            cur_iy = self._dobj['interactivity'][kinter]['cur_iy']
            if self._dobj['interactivity'][kinter]['follow']:
                self._dref[cur_refy]['indices'][cur_iy:] = iy
            else:
                self._dref[cur_refy]['indices'][cur_iy] = iy

        self.update_interactivity(
            cur_groupx=cur_groupx,
            cur_groupy=cur_groupy,
            cur_refx=cur_refx,
            cur_refy=cur_refy,
            cur_datax=cur_datax,
            cur_datay=cur_datay,
            # excluderef=True,
        )

    def mouserelease(self, event):
        c0 = 'pan' in self.can.manager.toolbar.mode.lower()
        c1 = 'zoom' in self.can.manager.toolbar.mode.lower()

        if c0 or c1:
            ax = self.curax_panzoom
            if ax is None:
                msg = (
                    "Make sure you release the mouse button on an axes !"
                    "\n Otherwise the background plot cannot be properly updated !"
                )
                raise Exception(msg)
            lax = ax.get_shared_x_axes().get_siblings(ax)
            lax += ax.get_shared_y_axes().get_siblings(ax)
            lax = list(set(lax))
            self._set_dbck(lax)

    # -------------------
    # Generic plotting
    # -------------------

    def plot_as_array(
        self,
        # parameters
        key=None,
        ind=None,
        vmin=None,
        vmax=None,
        cmap=None,
        aspect=None,
        nmax=None,
        # figure-specific
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
            nmax=nmax,
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
