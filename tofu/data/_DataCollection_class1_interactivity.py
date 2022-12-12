

import warnings


import numpy as np
import matplotlib.pyplot as plt
import datastock as ds


from ._DataCollection_class0_Base import DataCollection0
from . import _DataCollection_interactivity as _interactivity
from . import _DataCollection_comp


__all__ = ['DataCollection1']    # , 'TimeTraceCollection']


_DKEYS = {
    'control': {'val': False, 'action': 'generic'},
    'ctrl': {'val': False, 'action': 'generic'},
    'shift': {'val': False, 'action': 'generic'},
    'alt': {'val': False, 'action': 'generic'},
    'left': {'val': False, 'action': 'move'},
    'right': {'val': False, 'action': 'move'},
    'up': {'val': False, 'action': 'move'},
    'down': {'val': False, 'action': 'move'},
}
_INCREMENTS = [1, 10]


# #################################################################
# #################################################################
#               Main class
# #################################################################


class DataCollection1(DataCollection0):
    """ Handles matplotlib interactivity """

    _LPAXES = ['ax', 'type']
    _dinteractivity = dict.fromkeys(['curax_panzoom'])

    # ----------------------
    #   Add objects
    # ----------------------

    def add_mobile(
        self,
        key=None,
        handle=None,
        ref=None,
        data=None,
        dtype=None,
        bstr=None,
        visible=None,
        ax=None,
        **kwdargs,
    ):

        # ----------
        # check ref

        if isinstance(ref, str):
            ref = (ref,)
        if isinstance(ref, list):
            ref = tuple(ref)

        if ref is None or not all([rr in self._dref.keys() for rr in ref]):
            msg = (
                "Arg ref must be a tuple of existing ref keys!\n"
                f"\t- Provided: {ref}"
            )
            raise Exception(msg)

        # ----------
        # check dtype

        dtype = ds._generic_check._check_var(
            dtype,
            'dtype',
            types=str,
            allowed=['xdata', 'ydata', 'data', 'alpha', 'txt']
        )

        # ----------
        # check data

        if isinstance(data, str):
            data = (data,)
        if isinstance(data, list):
            data = tuple(data)
        if data is None:
            data = ['index' for rr in ref]

        c0 = (
            len(ref) == len(data)
            and all([rr == 'index' or rr in self._ddata.keys() for rr in data])
        )
        if not c0:
            msg = (
                "Arg data must be a tuple of existing data keys!\n"
                "It should hqve the same length as ref!\n"
                f"\t- Provided ref: {ref}\n"
                f"\t- Provided data: {data}"
            )
            raise Exception(msg)

        super().add_obj(
            which='mobile',
            key=key,
            handle=handle,
            group=None,
            ref=ref,
            data=data,
            dtype=dtype,
            visible=visible,
            bstr=bstr,
            ax=ax,
            func=None,
            **kwdargs,
        )

    def add_axes(
        self,
        key=None,
        handle=None,
        type=None,
        refx=None,
        refy=None,
        datax=None,
        datay=None,
        invertx=None,
        inverty=None,
        **kwdargs,
    ):

        # ----------------
        # check refx, refy

        if isinstance(refx, str):
            refx = [refx]
        if isinstance(refy, str):
            refy = [refy]

        c0 = (
            isinstance(refx, list)
            and all([rr in self._dref.keys() for rr in refx])
        )
        if refx is not None and not c0:
            msg = "Arg refx must be a list of valid ref keys!"
            raise Exception(msg)

        c0 = (
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
            groupx=None,
            groupy=None,
            refx=refx,
            refy=refy,
            datax=datax,
            datay=datay,
            invertx=invertx,
            inverty=inverty,
            bck=None,
            mobile=None,
            canvas=None,
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
    # Debug mode
    # ------------------

    def set_debug(self, debug=None):
        """ Set debug mode to True / False """
        debug = ds._generic_check._check_var(
            debug,
            'debug',
            default=False,
            types=bool,
        )
        self.debug = debug

    def show_debug(self):
        """ Display information relevant for live debugging """
        print('\n\n')
        return self.get_summary(
            show_which=['ref', 'group', 'interactivity'],
        )

    # ------------------
    # Setup interactivity
    # ------------------

    def setup_interactivity(
        self,
        kinter=None,
        dgroup=None,
        dkeys=None,
        dinc=None,
        cur_ax=None,
        debug=None,
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

        ic = 0
        for k0, v0 in dgroup.items():
            if v0.get('nmax') is None:
                dgroup[k0]['nmax'] = 0
            dgroup[k0]['nmaxcur'] = 0
            dgroup[k0]['indcur'] = 0

        # ----------
        # Check increment dict

        if dinc is None:
            dinc = {k0: _INCREMENTS for k0 in self._dref.keys()}
        elif isinstance(dinc, list) and len(dinc) == 2:
            dinc = {k0: dinc for k0 in self._dref.keys()}
        elif isinstance(dinc, dict):
            c0 = all([
                ss in self._dref.keys()
                and isinstance(vv, list)
                and len(vv) == 2
                for ss, vv in dinc.items()
            ])
            if not c0:
                msg = (
                    "Arg dinc must be a dict of type {ref0: [inc0, inc1]}\n"
                    f"\t- Provided: {dinc}"
                )
                raise Exception(msg)
            for k0 in self._dref.keys():
                if k0 not in dinc.keys():
                    dinc[k0] = _INCREMENTS
        else:
            msg = (
                "Arg dinc must be a dict of type {ref0: [inc0, inc1]}\n"
                f"\t- Provided: {dinc}"
            )
            raise Exception(msg)

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

        self.add_param(which='ref', param='group', value=drefgroup)
        self.add_param(which='ref', param='inc', value=dinc)

        # --------------------------------------
        # update dax with groupx, groupy and inc

        daxgroupx = dict.fromkeys(self._dobj['axes'].keys())
        daxgroupy = dict.fromkeys(self._dobj['axes'].keys())
        dinc = dict.fromkeys(self._dobj['axes'].keys())
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

            # increment
            dinc[k0] = {
                'left': -1, 'right': 1,
                'down': -1, 'up': 1,
            }

        self.set_param(which='axes', param='groupx', value=daxgroupx)
        self.set_param(which='axes', param='groupy', value=daxgroupy)
        self.add_param(which='axes', param='inc', value=dinc)

        # group, ref, nmax

        # cumsum0 = np.r_[0, np.cumsum(nmax[1, :])]
        # arefind = np.zeros((np.sum(nmax[1, :]),), dtype=int)

        # --------------------------
        # update mobile with groups

        for k0, v0 in self._dobj['mobile'].items():
            self._dobj['mobile'][k0]['group'] = tuple([
                self._dref[rr]['group'] for rr in v0['ref']
            ])
            self._dobj['mobile'][k0]['func'] = _interactivity.get_fupdate(
                handle=v0['handle'],
                dtype=v0['dtype'],
                norm=None,
                bstr=v0.get('bstr'),
            )

        # --------------------
        # axes mobile, refs and canvas

        daxcan = dict.fromkeys(self._dobj['axes'].keys())
        for k0, v0 in self._dobj['axes'].items():

            # Update mobile
            self._dobj['axes'][k0]['mobile'] = [
                k1 for k1, v1 in self._dobj['mobile'].items()
                if v1['ax'] == k0
            ]

            # ref
            if v0['refx'] is not None:
                for ii, rr in enumerate(v0['refx']):
                    if v0['datax'][ii] is None:
                        self._dobj['axes'][k0]['datax'][ii] = 'index'

            # canvas
            lcan = [
                k1 for k1, v1 in self._dobj['canvas'].items()
                if v1['handle'] == v0['handle'].figure.canvas
            ]
            assert len(lcan) == 1
            self._dobj['axes'][k0]['canvas'] = lcan[0]

        # -------
        # dgroup

        # update with axes
        for k0, v0 in dgroup.items():
            lkax = [
                k1 for k1, v1 in self._dobj['axes'].items()
                if (v1['groupx'] is not None and k0 in v1['groupx'])
                or (v1['groupy'] is not None and k0 in v1['groupy'])
            ]
            dgroup[k0]['axes'] = lkax

        for k0, v0 in dgroup.items():
            self.add_obj(
                which='group',
                key=k0,
                **v0,
            )

        # ---------
        # dkeys

        if dkeys is None:
            dkeys = _DKEYS

        # add key for switching groups
        dkeys.update({
            v0.get('key', f'f{ii+1}'): {
                'group': k0,
                'val': False,
                'action': 'group',
            }
            for ii, (k0, v0) in enumerate(self._dobj['group'].items())
        })

        # add keys for switching indices within groups
        nMax = np.max([v0['nmax'] for v0 in dgroup.values()])
        dkeys.update({
            str(ii): {'ind': ii, 'val': False, 'action': 'indices'}
            for ii in range(0, nMax)
        })

        # implement dict
        for k0, v0 in dkeys.items():
            self.add_obj(
                which='key',
                key=k0,
                **v0,
            )

        lact = set([v0['action'] for v0 in dkeys.values()])
        self.__dkeys_r = {
            k0: [k1 for k1 in dkeys.keys() if dkeys[k1]['action'] == k0]
            for k0 in lact
        }

        # ---------
        # dinter

        dinter = {
            'cur_ax': cur_ax,
            'cur_groupx': None,
            'cur_groupy': None,
            'cur_refx': None,
            'cur_refy': None,
            'cur_datax': None,
            'cur_datay': None,
            'follow': True,
        }

        self.add_obj(
            which='interactivity',
            key='inter0',
            **dinter,
        )
        self.kinter = 'inter0'

        _interactivity._set_dbck(
            lax=self._dobj['axes'].keys(),
            daxes=self._dobj['axes'],
            dcanvas=self._dobj['canvas'],
            dmobile=self._dobj['mobile'],
        )

        # -----------------------------------
        # set current axe / group / ref / data...

        if cur_ax is None:
            cur_ax = [
                k0 for k0, v0 in self._dobj['axes'].items()
                if v0['groupx'] is not None and v0['groupy'] is not None
            ]
            if len(cur_ax) == 0:
                cur_ax = list(self._dobj['axes'].keys())[0]
            else:
                cur_ax = cur_ax[0]

        self._get_current_grouprefdata_from_kax(kax=cur_ax)
        self.set_debug(debug)

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
            keyp = v0['handle'].mpl_connect('key_press_event', self.onkeypress)
            keyr = v0['handle'].mpl_connect('key_release_event', self.onkeypress)
            butp = v0['handle'].mpl_connect('button_press_event', self.mouseclic)
            # res = v0['handle'].mpl_connect('resize_event', self.resize)
            butr = v0['handle'].mpl_connect('button_release_event', self.mouserelease)
            close = v0['handle'].mpl_connect('close_event', self.on_close)
            # if not plt.get_backend() == "agg":
            # v0['handle'].manager.toolbar.release = self.mouserelease

            self._dobj['canvas'][k0]['cid'] = {
                'keyp': keyp,
                'keyr': keyr,
                'butp': butp,
                # 'res': res,
                'butr': butr,
                'close': close,
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

    def _get_current_grouprefdata_from_kax(self, kax=None):

        # Get current group and ref
        groupx = self._dobj['axes'][kax]['groupx']
        groupy = self._dobj['axes'][kax]['groupy']
        refx = self._dobj['axes'][kax]['refx']
        refy = self._dobj['axes'][kax]['refy']

        # Get kinter
        kinter = list(self._dobj['interactivity'].keys())[0]

        # Get current groups
        cur_groupx = self._dobj['interactivity'][kinter]['cur_groupx']
        cur_groupy = self._dobj['interactivity'][kinter]['cur_groupy']

        # determine whether cur_groupx shall be updated
        if groupx is not None:
            if cur_groupx in groupx:
                pass
            elif groupx is not None:
                cur_groupx = groupx[0]
        if groupy is not None:
            if cur_groupy in groupy:
                pass
            elif groupy is not None:
                cur_groupy = groupy[0]

        # # get current refs
        cur_refx = self._dobj['interactivity'][kinter]['cur_refx']
        cur_refy = self._dobj['interactivity'][kinter]['cur_refy']
        if groupx is not None:
            if cur_refx in self._dobj['group'][cur_groupx]['ref']:
                pass
            elif cur_groupx is not None:
                cur_refx = self._dobj['group'][cur_groupx]['ref'][0]
        if groupy is not None:
            if cur_refy in self._dobj['group'][cur_groupy]['ref']:
                pass
            elif cur_groupy is not None:
                cur_refy = self._dobj['group'][cur_groupy]['ref'][0]

        # data
        cur_datax = self._dobj['interactivity'][kinter]['cur_datax']
        if self._dobj['axes'][kax]['refx'] is not None:
            ix = self._dobj['axes'][kax]['refx'].index(cur_refx)
            cur_datax = self._dobj['axes'][kax]['datax'][ix]

        cur_datay = self._dobj['interactivity'][kinter]['cur_datay']
        if self._dobj['axes'][kax]['refy'] is not None:
            iy = self._dobj['axes'][kax]['refy'].index(cur_refy)
            cur_datay = self._dobj['axes'][kax]['datay'][iy]

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
        })

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
        kax = ds._generic_check._check_var(
            None, 'kax',
            types=str,
            allowed=lkax,
        )
        ax = self._dobj['axes'][kax]['handle']

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

        self._get_current_grouprefdata_from_kax(kax=kax)

    # -----------------------------
    # Interactivity: generic update
    # -----------------------------

    def update_interactivity(self):
        """ Called at each event """

        cur_groupx = self._dobj['interactivity'][self.kinter]['cur_groupx']
        cur_groupy = self._dobj['interactivity'][self.kinter]['cur_groupy']
        cur_refx = self._dobj['interactivity'][self.kinter]['cur_refx']
        cur_refy = self._dobj['interactivity'][self.kinter]['cur_refy']
        cur_datax = self._dobj['interactivity'][self.kinter]['cur_datax']
        cur_datay = self._dobj['interactivity'][self.kinter]['cur_datay']

        # Propagate indices through refs
        if cur_refx is not None:
            lref = self._dobj['group'][cur_groupx]['ref']
            ldata = self._dobj['group'][cur_groupx]['data']
            self.propagate_indices_per_ref(
                ref=cur_refx,
                lref=[rr for rr in lref if rr != cur_refx],
                ldata=[cur_datax] + [dd for dd in ldata if dd != cur_datax],
                param=None,
            )

        if cur_refy is not None:
            lref = self._dobj['group'][cur_groupy]['ref']
            ldata = self._dobj['group'][cur_groupy]['data']
            self.propagate_indices_per_ref(
                ref=cur_refy,
                lref=[rr for rr in lref if rr != cur_refy],
                ldata=[cur_datay] + [dd for dd in ldata if dd != cur_datay],
                param=None,
            )

        # get list of mobiles to update and set visible
        lmobiles = []
        if cur_groupx is not None:
            lmobiles += [
                k0 for k0, v0 in self._dobj['mobile'].items()
                if any([
                    rr in v0['ref']
                    for rr in self._dobj['group'][cur_groupx]['ref']
                ])
            ]

        if cur_groupy is not None:
            lmobiles += [
                k0 for k0, v0 in self._dobj['mobile'].items()
                if any([
                    rr in v0['ref']
                    for rr in self._dobj['group'][cur_groupy]['ref']
                ])
            ]

        self._update_mobiles(lmobiles=lmobiles)     # 0.2 s

        if self.debug:
            self.show_debug()

    def _update_mobiles(self, lmobiles=None):

        # Set visibility of mobile objects - TBF/TBC
        for k0 in lmobiles:
            vis = all([
                self._dobj['mobile'][k0]['ind']
                < self._dobj['group'][gg]['nmaxcur']
                for gg in self._dobj['mobile'][k0]['group']
            ])
            self._dobj['mobile'][k0]['visible'] = vis

        # get list of axes to update
        lax = [
            k0 for k0, v0 in self._dobj['axes'].items()
            if any([self._dobj['mobile'][k1]['ax'] == k0 for k1 in lmobiles])
        ]

        # ---- Restore backgrounds ---- 1 ms
        for aa in lax:
            self._dobj['canvas'][
                self._dobj['axes'][aa]['canvas']
            ]['handle'].restore_region(
                self._dobj['axes'][aa]['bck'],
            )

        # ---- update data of group objects ----  0.15 s
        for k0 in lmobiles:
            _interactivity._update_mobile(
                dmobile=self._dobj['mobile'],
                dref=self._dref,
                ddata=self._ddata,
                k0=k0,
            )

        # --- Redraw all objects (due to background restore) --- 25 ms
        for k0, v0 in self._dobj['mobile'].items():
            v0['handle'].set_visible(v0['visible'])
            self._dobj['axes'][v0['ax']]['handle'].draw_artist(v0['handle'])

        # ---- blit axes ------ 5 ms
        for aa in lax:
            self._dobj['canvas'][
                self._dobj['axes'][aa]['canvas']
            ]['handle'].blit(self._dobj['axes'][aa]['handle'].bbox)

    def resize(self, event):
        _interactivity._set_dbck(
            lax=self._dobj['axes'].keys(),
            daxes=self._dobj['axes'],
            dcanvas=self._dobj['canvas'],
            dmobile=self._dobj['mobile'],
        )

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
            if str(err) == 'clic not in axes':
                return
            raise err
            # warnings.warn(str(err))
            # return

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
        cur_datax = self._dobj['interactivity'][kinter]['cur_datax']
        cur_datay = self._dobj['interactivity'][kinter]['cur_datay']

        shift = self._dobj['key']['shift']['val']
        ctrl = any([
            self._dobj['key'][ss]['val'] for ss in ['control', 'ctrl']
        ])

        # Update number of indices (for visibility)
        for gg in [cur_groupx, cur_groupy]:
            if gg is not None:
                out = _interactivity._update_indices_nb(
                    group=gg,
                    dgroup=self._dobj['group'],
                    ctrl=ctrl,
                    shift=shift,
                )

        # update indcur x/y vs shift / ctrl ?  TBF
        pass

        # Check refx/refy vs datax/datay
        if cur_refx is not None and cur_refy is not None:
            c0 = (
                'index' in [cur_datax, cur_datay]
                or ((cur_refx == cur_refy) == (cur_datax == cur_datay))
            )
            if not c0:
                msg = (
                    "Invalid ref / data pairs:\n"
                    f"\t- cur_refx, cur_refy: {cur_refx}, {cur_refy}\n"
                    f"\t- cdx, cdy: {cdx}, {cdy}"
                )
                raise Exception(msg)

        # update ref indices
        if None not in [cur_refx, cur_refy] and cur_refx == cur_refy:

            raise NotImplementedError()
            dist = (cdx - event.xdata)**2 + (cdy - event.ydata)**2
            lind = [
                np.nanargmin(dist, axis=ii) for ii in range(datax.ndim)
            ]

        else:

            c0x = (
                cur_refx is not None
                and self._dobj['axes'][kax]['refx'] is not None
                and cur_refx in self._dobj['axes'][kax]['refx']
            )
            if c0x:
                monot = None
                if cur_datax == 'index':
                    cdx = 'index'
                else:
                    monot = self._ddata[cur_datax]['monot'] == (True,)
                    cdx = self._ddata[cur_datax]['data']
                ix = _DataCollection_comp._get_index_from_data(
                    data=cdx,
                    data_pick=np.r_[event.xdata],
                    monot=monot,
                )[0]

            c0y = (
                cur_refy is not None
                and self._dobj['axes'][kax]['refy'] is not None
                and cur_refy in self._dobj['axes'][kax]['refy']
            )
            if c0y:
                monot = None
                if cur_datay == 'index':
                    cdy = 'index'
                else:
                    monot = self._ddata[cur_datay]['monot'] == (True,)
                    cdy = self._ddata[cur_datay]['data']
                iy = _DataCollection_comp._get_index_from_data(
                    data=cdy,
                    data_pick=np.r_[event.ydata],
                    monot=monot,
                )[0]

        # Update ref indices
        if c0x:
            cur_ix = self._dobj['group'][cur_groupx]['indcur']
            follow = (
                cur_ix == self._dobj['group'][cur_groupx]['nmaxcur'] - 1
                and self._dobj['interactivity'][kinter]['follow']
            )
            if follow:
                self._dref[cur_refx]['indices'][cur_ix:] = ix
            else:
                self._dref[cur_refx]['indices'][cur_ix] = ix

        # Update ref indices
        if c0y:
            cur_iy = self._dobj['group'][cur_groupy]['indcur']
            follow = (
                cur_iy == self._dobj['group'][cur_groupy]['nmaxcur'] - 1
                and self._dobj['interactivity'][kinter]['follow']
            )
            if follow:
                self._dref[cur_refy]['indices'][cur_iy:] = iy
            else:
                self._dref[cur_refy]['indices'][cur_iy] = iy

        self.update_interactivity()

    def mouserelease(self, event):
        """ Mouse release: nothing except if resize ongoing (redraw bck) """

        c0 = event.inaxes is not None and event.button == 1
        if not c0:
            return

        can = [
            k0 for k0, v0 in self._dobj['canvas'].items()
            if v0['handle'] == event.inaxes.figure.canvas
        ][0]
        mode = self._dobj['canvas'][can]['handle'].manager.toolbar.mode.lower()
        c0 = 'pan' in mode
        c1 = 'zoom' in mode

        if c0 or c1:
            kax = self._dobj['interactivity']['curax_panzoom']
            if kax is None:
                msg = (
                    "Make sure you release the mouse button on an axes!"
                    "\n Otherwise background plot cannot be properly updated!"
                )
                raise Exception(msg)
            ax = self._dobj['axes'][kax]['handle']
            lax = ax.get_shared_x_axes().get_siblings(ax)
            lax += ax.get_shared_y_axes().get_siblings(ax)
            lax = list(set(lax))
            _interactivity._set_dbck(
                lax=lax,
                daxes=self._dobj['axes'],
                dcanvas=self._dobj['canvas'],
                dmobile=self._dobj['mobile'],
            )

    # ----------------------
    # Interactivity: keys
    # ----------------------

    def onkeypress(self, event):
        """ Event handler in case of key press / release """

        # -----------------------
        # Check event is relevant 1

        # decompose key combinations
        lkey = event.key.split('+')

        # get current inter, axes, canvas
        kinter = self.kinter
        kax = self._dobj['interactivity'][kinter]['cur_ax']
        kcan = [
            k0 for k0, v0 in self._dobj['canvas'].items()
            if k0 == self._dobj['axes'][kax]['canvas']
        ][0]
        can = self._dobj['canvas'][kcan]['handle']

        # check relevance
        c0 = can.manager.toolbar.mode != ''
        c1 = len(lkey) not in [1, 2]
        c2 = [ss not in self._dobj['key'].keys() for ss in lkey]

        if c0 or c1 or any(c2):
            return

        # -----------------------
        # Check event is relevant 2

        # get list of current keys for each action type
        lgen = [kk for kk in self.__dkeys_r['generic'] if kk in lkey]
        lmov = [kk for kk in self.__dkeys_r['move'] if kk in lkey]
        lgrp = [kk for kk in self.__dkeys_r['group'] if kk in lkey]
        lind = [kk for kk in self.__dkeys_r['indices'] if kk in lkey]

        # if no relevant key pressed => return
        ngen, nmov, ngrp, nind = len(lgen), len(lmov), len(lgrp), len(lind)
        ln = np.r_[ngen, nmov, ngrp, nind]
        if np.any(ln > 1) or np.sum(ln) > 2:
            return
        if np.sum(ln) == 2 and (ngrp == 1 or nind == 1):
            return

        # only keep relevant keys
        genk = None if ngen == 0 else lgen[0]
        movk = None if nmov == 0 else lmov[0]
        grpk = None if ngrp == 0 else lgrp[0]
        indk = None if nind == 0 else lind[0]

        # ------------------------
        # Event = change key value

        # change key values if relevant
        if event.name == 'key_release_event':
            if event.key == genk:
                self._dobj['key'][genk]['val'] = False
            return

        if genk is not None and event.key == genk:
            self._dobj['key'][genk]['val'] = True
            return

        # ----------------------------
        # Event = change current group

        if grpk is not None:
            # group
            group = self._dobj['key'][event.key]['group']
            cx = any([
                v0['groupx'] is not None and group in v0['groupx']
                for v0 in self._dobj['axes'].values()
            ])
            if cx:
                self._dobj['interactivity'][self.kinter]['cur_groupx'] = group
            cy = any([
                v0['groupy'] is not None and group in v0['groupy']
                for v0 in self._dobj['axes'].values()
            ])
            if cy:
                self._dobj['interactivity'][self.kinter]['cur_groupy'] = group

            # axes
            cur_ax = self._dobj['interactivity'][self.kinter]['cur_ax']
            if cur_ax not in self._dobj['group'][group]['axes']:
                self._dobj['interactivity'][self.kinter]['cur_ax'] = (
                    self._dobj['group'][group]['axes'][0]
                )

            # ref
            if cx:
                cur_refx = self._dobj['interactivity'][self.kinter]['cur_refx']
                if self._dref[cur_refx]['group'] != group:
                    cur_refx = self._dobj['group'][group]['ref'][0]
                self._dobj['interactivity'][self.kinter]['cur_refx'] = cur_refx

            if cy:
                cur_refy = self._dobj['interactivity'][self.kinter]['cur_refy']
                if self._dref[cur_refy]['group'] != group:
                    cur_refy = self._dobj['group'][group]['ref'][0]
                self._dobj['interactivity'][self.kinter]['cur_refy'] = cur_refy

            # data
            if c0:
                self._dobj['interactivity'][self.kinter]['cur_datax'] = 'index'
            if c0:
                self._dobj['interactivity'][self.kinter]['cur_datay'] = 'index'

            msg = f"Current group set to {group}"
            print(msg)
            return

        # ----------------------------
        # Event = change current index

        if indk is not None:
            groupx = self._dobj['interactivity'][self.kinter]['cur_groupx']
            groupy = self._dobj['interactivity'][self.kinter]['cur_groupy']

            # groupx
            if groupx is not None:
                imax = self._dobj['group'][groupx]['nmaxcur']
                ii = int(event.key)
                if ii > imax:
                    msg = "Set to current max index for '{groupx}': {imax}"
                    print(msg)
                ii = min(ii, imax)
                self._dobj['group'][groupx]['indcur'] = ii

            # groupy
            if groupy is not None:
                imax = self._dobj['group'][groupy]['nmaxcur']
                ii = int(event.key)
                if ii > imax:
                    msg = "Set to current max index for '{groupy}': {imax}"
                    print(msg)
                ii = min(ii, imax)
                self._dobj['group'][groupy]['indcur'] = ii

            msg = f"Current indices set to {ii}"
            print(msg)
            return

        # ----------------------------
        # Event = move current index

        if movk is not None:

            if movk in ['left', 'right']:
                group = self._dobj['interactivity'][self.kinter]['cur_groupx']
                ref = self._dobj['interactivity'][self.kinter]['cur_refx']
                incsign = 1.
                if self._dobj['axes'][kax].get('invertx', False):
                    incsign = -1
            elif movk in ['up', 'down']:
                group = self._dobj['interactivity'][self.kinter]['cur_groupy']
                ref = self._dobj['interactivity'][self.kinter]['cur_refy']
                incsign = 1.
                if self._dobj['axes'][kax].get('inverty', False):
                    incsign = -1

            if group is None:
                return

            # dmovkeys for inversions and steps ?

            shift = self._dobj['key']['shift']['val']
            ctrl = any([
                self._dobj['key'][ss]['val'] for ss in ['control', 'ctrl']
            ])
            alt = self._dobj['key']['alt']['val']

            # Check max number of occurences not reached if shift
            c0 = (
                shift
                and (
                    self._dobj['group'][group]['indcur']
                    == self._dobj['group'][group]['nmax'] - 1
                )
            )
            if c0:
                msg = "Max nb. of plots reached ({0}) for group {1}"
                msg = msg.format(self._dobj['group'][group]['nmax'], group)
                print(msg)
                return

            # update nb of visible indices
            out = _interactivity._update_indices_nb(
                group=group,
                dgroup=self._dobj['group'],
                ctrl=ctrl,
                shift=shift,
            )
            if out is False:
                return

            # get increment from key
            cax = self._dobj['interactivity'][self.kinter]['cur_ax']
            inc = incsign * (
                self._dref[ref]['inc'][int(alt)]
                * self._dobj['axes'][cax]['inc'][movk]
            )

            # update ref indices
            icur = self._dobj['group'][group]['indcur']
            ix = (
                (self._dref[ref]['indices'][icur] + inc)
                % self._dref[ref]['size']
            )

            # Update ref indices
            follow = (
                icur == self._dobj['group'][group]['nmaxcur'] - 1
                and self._dobj['interactivity'][kinter]['follow']
            )
            if follow:
                self._dref[ref]['indices'][icur:] = ix
            else:
                self._dref[ref]['indices'][icur] = ix

            # global update of interactivity
            self.update_interactivity()

    # -------------------
    # Close all
    # -------------------

    def on_close(self, event):
        kcan = [
            k0 for k0, v0 in self._dobj['canvas'].items()
            if v0['handle'] == event.canvas
        ]
        if len(kcan) > 1:
            raise Exception('Several matching canvas')
        elif len(kcan) == 1:

            if len(self._dobj['canvas']) == 1:
                self.close_all()

            else:
                lax = [
                    k1 for k1, v1 in self._dobj['axes'].items()
                    if v1['canvas'] == kcan[0]
                ]
                lmob = [
                    k1 for k1, v1 in self._dobj['mobile'].items()
                    if v1['ax'] in lax
                ]
                for k1 in lax:
                    del self._dobj['axes'][k1]
                for k1 in lmob:
                    del self._dobj['mobile'][k1]
                del self._dobj['canvas'][kcan[0]]

    def close_all(self):

        # close figures
        if 'axes' in self._dobj.keys():
            lfig = set([
                v0['handle'].figure for v0 in self._dobj['axes'].values()
            ])
            for ff in lfig:
                plt.close(ff)

        # delete obj dict
        lk = ['interactivity', 'mobile', 'key', 'canvas', 'group', 'axes']
        for kk in lk:
            if kk in self._dobj.keys():
                del self._dobj[kk]

        # remove interactivity-specific param in dref
        lp = list(set(self.get_lparam(which='ref')).intersection(
            ['indices', 'group', 'inc']
        ))
        self.remove_param(which='ref', param=lp)
