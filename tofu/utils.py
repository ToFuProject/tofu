
# Built-in
import os

# Common
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


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
        e2 = np.cross(nIn,e1)
    e2 = np.asarray(e2)
    e2 = e2/np.linalg.norm(e2)
    assert np.abs(np.sum(nIn*e2))<1.e-12
    assert np.abs(np.sum(e1*e2))<1.e-12

    # Get starting points
    d1 = D12[0]*np.linspace(-0.5,0.5,N12[0],endpoint=True)
    d2 = D12[1]*np.linspace(-0.5,0.5,N12[1],endpoint=True)
    d1 = np.tile(d1,N12[1])
    d2 = np.repeat(d2,N12[0])
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
###############################################


class KeyHandler(object):
    """ Base class for handling event on tofu interactive figures """

    def __init__(self, can=None, daxT=None, ntMax=3, nchMax=3, nlambMax=3):
        lk = ['t','chan','chan2D','lamb','cross','hor','txtt','txtch','txtlamb','other']
        assert all([kk in lk for kk in daxT.keys()]), str(daxT.keys())
        assert all([type(dd) is list for dd in daxT.values()]), str(daxT.values())
        self.lk = sorted(list(daxT.keys()))

        self.can = can
        daxr, dh = self._make_daxr_dh(daxT)

        self.daxT = daxT
        self.daxr, self.dh = daxr, dh
        self.store_rcParams = None
        self.lkeys = ['right','left','shift']
        if 'chan2D' in self.daxT.keys() and len(self.daxT['chan2D'])>0:
            self.lkeys += ['up','down']
        self.curax = None
        self.shift = False
        self.ref, dnMax = {}, {'chan':nchMax,'t':ntMax,'lamb':nlambMax}
        for kk in self.lk:
            if not kk in ['chan2D','cross','hor','txtt','txtch','txtlamb','other']:
                self.ref[kk] = {'ind':np.zeros((ntMax,),dtype=int),
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
        xref = self.ref[axT]['val']
        for dax in self.daxT[axT]:
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
                for ii in range(0,self.ref[axT]['ncur']):
                    hh = self.daxr[ax]['dh']['vline'][jj]['h'][ii]
                    ind = self.ref[axT]['ind'][ii]
                    val = self.ref[axT]['val'][ii]
                    if xref is not self.daxr[self.curax]['xref']:
                        ind = np.argmin(np.abs(xref-val))
                        val = xref[ind]
                    hh.set_xdata(val)
                    self.dh[hh]['vis'] = True
                    for kk in dtg.keys():
                        for ll in range(0,len(dtg[kk])):
                            if dtg[kk][ll]['xref'] is xref:
                                indh = ind
                            else:
                                indh = np.argmin(np.abs(dtg[kk][ll]['xref']-val))
                            h = dtg[kk][ll]['h'][ii]
                            if 'txt' in dtg[kk][ll].keys():
                                if 'format' in dtg[kk][ll].keys():
                                    sss = '{0:%s}'%dtg[kk][ll]['format']
                                    h.set_text(sss.format(dtg[kk][ll]['txt'][indh]))
                                else:
                                    h.set_text(dtg[kk][ll]['txt'][indh])
                            elif 'xy' in dtg[kk][ll].keys():
                                h.set_data(dtg[kk][ll]['xy'][indh])
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
            for kk in self.daxr[ax]['dh'].keys():
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
        Type = 'chan' if 'chan' in Type else Type
        if self.shift and self.ref[Type]['ncur']>=self.ref[Type]['nMax']:
            print("     Max. nb. of %s plots reached !!!"%Type)
            return

        val = self.daxr[event.inaxes]['xref']
        if C3:
            evxy = np.r_[event.xdata,event.ydata]
            d2 = np.sum((val-evxy[:,np.newaxis])**2,axis=0)
        else:
            d2 = np.abs(event.xdata-val)
        ind = np.nanargmin(d2)
        val = val[ind]
        ii = int(self.ref[Type]['ncur']) if self.shift else 0
        self.ref[Type]['ind'][ii] = ind
        self.ref[Type]['val'][ii] = val
        self.ref[Type]['ncur'] = ii+1
        self.update()

    def onkeypress(self,event):
        C0 = self.can.manager.toolbar._active is None
        C1 = [kk in event.key for kk in self.lkeys]
        C2 = event.name is 'key_release_event' and event.key=='shift'
        C3 = event.name is 'key_press_event'

        if not (C0 and any(C1) and (C2 or C3)):
            return

        if event.key=='shift':
            self.shift = False if C2 else True
            return

        Type = self.daxr[self.curax]['Type']
        Type = 'chan' if 'chan' in Type else Type
        if self.shift and self.ref[Type]['ncur']>=self.ref[Type]['nMax']:
                print("     Max. nb. of %s plots reached !!!"%Type)
                return

        kdir = [kk for kk in self.lkeys if kk in event.key and not kk=='shift']
        val = self.daxr[self.curax]['xref']
        if self.daxr[self.curax]['Type']=='chan2D':
            c = -1. if self.daxr[self.curax]['invert'] else 1.
            x12 = val[:,self.ref[Type]['ind'][self.ref[Type]['ncur']-1]]
            x12 = x12 + c*self.daxr[self.curax]['incx'][kdir]
            d2 = np.sum((val-x12[:,np.newaxis])**2,axis=0)
            ind = np.nanargmin(d2)
        else:
            inc = -1 if 'left' in event.key else 1
            ind = (self.ref[Type]['ind'][self.ref[Type]['ncur']-1]+inc)
            ind = ind%val.size
        val = val[ind]
        ii = self.ref[Type]['ncur'] if self.shift else self.ref[Type]['ncur']-1
        self.ref[Type]['ind'][ii] = ind
        self.ref[Type]['val'][ii] = val
        if self.shift:
            self.ref[Type]['ncur'] = ii+1
        self.update()

    ##### To be implemented for each case ####
    def set_dBack(self):
        """ Choose which axes need redrawing and call self._set_dBck() """
    def update(self):
        """ Implement basic behaviour, and call self._restore_Bck() """
