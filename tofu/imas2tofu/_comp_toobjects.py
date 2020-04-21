
import os
import warnings

import numpy as np


_ERRSHOT = False
_ERREXP = False


# #############################################################################
#                       Generic
# #############################################################################


def _check_shotExp_consistency(didd, lidd, tofustr='shot', imasstr='shot',
                               err=True, fallback=0):
    crit = None
    for idd in lidd:
        v0 = didd[idd]
        if imasstr in v0['params']:
            if crit is None:
                crit = v0['params'][imasstr]
            elif crit != v0['params'][imasstr]:
                ss = '{} : {}'.format(idd, str(v0['params'][imasstr]))
                msg = ("All idd refer to different {}!\n".format(imasstr)
                       + "\t- {}".format(ss))
                if err:
                    raise Exception(msg)
                else:
                    warnings.warn(msg)
    if crit is None:
        crit  = fallback
    return crit


def get_lidsidd_shotExp(lidsok,
                        errshot=None, errExp=None, upper=True,
                        dids=None, didd=None):
    """ Check whether all shot / Exp are consistent accross the ids """

    if errshot is None:
        errshot = _ERRSHOT
    if errExp is None:
        errExp = _ERREXP

    lids = set(lidsok).intersection(dids.keys())
    lidd = set([dids[ids]['idd'] for ids in lids])

    # shot (non-identical => error if errshot is True, warning otherwise)
    shot = _check_shotExp_consistency(didd, lidd,
                                      tofustr='shot', imasstr='shot',
                                      err=errshot, fallback=0)

    # Exp (non-identical => error if errExp is True, warning otherwise)
    Exp = _check_shotExp_consistency(didd, lidd,
                                     tofustr='Exp', imasstr='tokamak',
                                     err=errExp, fallback='Dummy')
    if upper is True:
        Exp = Exp.upper()
    return lids, lidd, shot, Exp


# #############################################################################
#                       Extra
# #############################################################################


def extra_checkformat(dextra, fordata=None,
                      dids=None, didd=None, dshort=None):

    lc = [dextra is False, dextra is None,
          isinstance(dextra, str),
          isinstance(dextra, list),
          isinstance(dextra, dict)]
    if not any(lc):
        msg = ("Arg dextra must be either:\n"
               + "\t- None:     set to default\n"
               + "\t- False:    no extra signal\n"
               + "\t- str:      a single extra signal (shortcut)\n"
               + "\t- list:     a list of extra signals\n"
               + "\t- dict:     a dict of extra signals {ids: list of short}\n"
               + "\n  You provided: {}".format(dextra))
        raise Exception(msg)

    if dextra is False:
        if fordata is True:
            return None
        else:
            return None, None

    elif dextra is None:
        dextra = {}
        if 'equilibrium' in dids.keys():
            dextra.update({'equilibrium': [('ip', 'k'), ('BT0', 'm'),
                                           ('axR', (0., 0.8, 0.)),
                                           ('axZ', (0., 1., 0.)),
                                           'ax', 'sep', 't']})
        if 'core_profiles' in dids.keys():
            dextra.update({'core_profiles': ['ip', 'vloop', 't']})
        if 'lh_antennas' in dids.keys():
            dextra.update({'lh_antennas': [('power0', (0.8, 0., 0.)),
                                           ('power1', (1., 0., 0.)), 't']})
        if 'ic_antennas' in dids.keys():
            dextra.update({'ic_antennas': [
                ('power0', (0. ,0. ,0.8)),
                ('power1', (0. ,0. ,1.)),
                ('power2', (0. ,0. ,0.9)), 't']})
    if type(dextra) is str:
        dextra = [dextra]
    if type(dextra) is list:
        dex = {}
        for ee in dextra:
            lids = [ids for ids in dids.keys()
                    if ee in dshort[ids].keys()]
            if len(lids) != 1:
                msg = "No / multiple matches:\n"
                msg = "extra %s not available from self._dshort"%ee
                raise Exception(msg)
            if lids[0] not in dex.keys():
                dex = {lids[0]:[ee]}
            else:
                dex[lids[0]].append(ee)
        dextra = dex
    return dextra


def extra_get_fordataTrue(inds, vs, vc, out, dout,
                          ids=None, dshort=None, dcomp=None):
    for ii in inds:
        ss = vs[ii]
        if ss == 't':
            continue
        if out[ss]['isemtpy'] is True:
            continue
        if ss in self._dshort[ids].keys():
            dd = self._dshort[ids][ss]
        else:
            dd = self._dcomp[ids][ss]
        label = dd.get('quant', 'unknown')
        units = out[ss]['units']
        key = '%s.%s'%(ids, ss)

        if 'sep' == ss.split('.')[-1].lower():
            out[ss] = np.swapaxes(out[ss]['data'], 1, 2)

        datastr = 'data'
        if any([ss.split('.')[-1].lower() == s0 for s0 in
                ['sep','ax','x']]):
            datastr = 'data2D'

        dout[key] = {'t': out['t']['data'],
                     datastr: out[ss]['data'],
                     'label': label, 'units': units, 'c': vc[ii]}


def extra_get_fordataFalse(out, d0d, dt0,
                           ids=None, dshort=None, dcomp=None):
    any_ = False
    keyt = '{}.t'.format(ids)
    for ss in out.keys():
        if ss == 't':
            continue
        if out[ss]['isempty'] is True:
            continue
        if ss in dshort[ids].keys():
            dd = dshort[ids][ss]
        else:
            dd = dcomp[ids][ss]
        dim = dd.get('dim', 'unknown')
        quant = dd.get('quant', 'unknown')
        units = out[ss]['units']
        key = '%s.%s'%(ids, ss)

        if 'sep' == ss.split('.')[-1].lower():
            out[ss]['data'] = np.swapaxes(out[ss]['data'], 1, 2)

        d0d[key] = {'data': out[ss]['data'], 'name': ss,
                    'origin': ids, 'dim': dim, 'quant': quant,
                    'units': units, 'depend': (keyt,)}
        any_ = True
    if any_ is True:
        dt0[keyt] = {'data': out['t']['data'], 'name': 't',
                     'origin': ids, 'depend': (keyt,)}



# #############################################################################
#                       Config
# #############################################################################


def config_extract_lS(ids, occ, wall, description_2d, mod,
                      kwargs=None, mobile=None):
    """ Extract all relevant structures """

    nlim = len(wall.limiter.unit)
    nmob = len(wall.mobile.unit)
    # onelimonly = False

    # ----------------------------------
    # Relevant only if vessel is filled
    # try:
    #    if len(wall.vessel.unit) != 1:
    #        msg = "There is no / several vessel.unit!"
    #        raise Exception(msg)
    #    if len(wall.vessel.unit[0].element) != 1:
    #        msg = "There is no / several vessel.unit[0].element!"
    #        raise Exception(msg)
    #    if len(wall.vessel.unit[0].element[0].outline.r) < 3:
    #        msg = "wall.vessel polygon has less than 3 points!"
    #        raise Exception(msg)
    #    name = wall.vessel.unit[0].element[0].name
    #    poly = np.array([wall.vessel.unit[0].element[0].outline.r,
    #                     wall.vessel.unit[0].element[0].outline.z])
    # except Exception as err:
    #    # If vessel not in vessel, sometimes stored a a single limiter
    #    if nlim == 1:
    #        name = wall.limiter.unit[0].name
    #        poly = np.array([wall.limiter.unit[0].outline.r,
    #                         wall.limiter.unit[0].outline.z])
    #        onelimonly = True
    #    else:
    #        msg = ("There does not seem to be any vessel, "
    #               + "not in wall.vessel nor in wall.limiter!")
    #        raise Exception(msg)
    # cls = None
    # if name == '':
    #     name = 'ImasVessel'
    # if '_' in name:
    #     ln = name.split('_')
    #     if len(ln) == 2:
    #         cls, name = ln
    #     else:
    #         name = name.replace('_', '')
    # if cls is None:
    #     cls = 'Ves'
    # assert cls in ['Ves', 'PlasmaDomain']
    # ves = getattr(mod, cls)(Poly=poly, Name=name, **kwargs)

    # Determine if mobile or not
    # if onelimonly is False:
    if mobile is None:
        if nlim == 0 and nmob > 0:
            mobile = True
        elif nmob == 0 and nlim > 0:
            mobile = False
        elif nmob > nlim:
            msgw = 'wall.description_2[{}]'.format(description_2d)
            msg = ("\nids wall has less limiter than mobile units\n"
                   + "\t- len({}.limiter.unit) = {}\n".format(msgw, nlim)
                   + "\t- len({}.mobile.unit) = {}\n".format(msgw, nmob)
                   + "  => Choosing mobile by default")
            warnings.warn(msg)
            mobile = True
        elif nmob <= nlim:
            msgw = 'wall.description_2[{}]'.format(description_2d)
            msg = ("\nids wall has more limiter than mobile units\n"
                   + "\t- len({}.limiter.unit) = {}\n".format(msgw, nlim)
                   + "\t- len({}.mobile.unit) = {}\n".format(msgw, nmob)
                   + "  => Choosing limiter by default")
            warnings.warn(msg)
            mobile = False
    assert isinstance(mobile, bool)

    # Get PFC
    if mobile is True:
        units = wall.mobile.unit
    else:
        units = wall.limiter.unit
    nunits = len(units)

    if nunits == 0:
        msg = ("There is no unit stored !\n"
               + "The required 2d description is empty:\n")
        ms = "len(idd.{}[occ={}].description_2d".format(ids, occ)
        msg += "{}[{}].limiter.unit) = 0".format(ms,
                                                 description_2d)
        raise Exception(msg)

    lS = [None for _ in units]
    for ii in range(0, nunits):
        try:
            if mobile is True:
                outline = units[ii].outline[0]
            else:
                outline = units[ii].outline
            poly = np.array([outline.r, outline.z])

            if units[ii].phi_extensions.size > 0:
                pos, extent = units[ii].phi_extensions.T
            else:
                pos, extent = None, None
            name = units[ii].name
            cls, mobi = None, None
            if name == '':
                name = 'unit{:02.0f}'.format(ii)
            if '_' in name:
                ln = name.split('_')
                if len(ln) == 2:
                    cls, name = ln
                elif len(ln) == 3:
                    cls, name, mobi = ln
                else:
                    name = name.replace('_', '')
            if cls is None:
                if ii == nunits - 1:
                    cls = 'Ves'
                else:
                    cls = 'PFC'
            # mobi = mobi == 'mobile'
            lS[ii] = getattr(mod, cls)(Poly=poly, pos=pos,
                                       extent=extent,
                                       Name=name,
                                       **kwargs)
        except Exception as err:
            msg = ("PFC unit[{}] named {} ".format(ii, name)
                   + "could not be loaded!\n"
                   + str(err))
            raise Exception(msg)
    return lS


# #############################################################################
#                       Plasma
# #############################################################################


def plasma_checkformat_dsig(dsig=None,
                            lidsplasma=None, dids=None,
                            dshort=None, dcomp=None):
    lidsok = set(lidsplasma).intersection(dids.keys())

    lscom = ['t']
    lsmesh = ['2dmeshNodes', '2dmeshFaces',
              '2dmeshR', '2dmeshZ']

    lc = [dsig is None,
          type(dsig) is str,
          type(dsig) is list,
          type(dsig) is dict]
    assert any(lc)

    # Convert to dict
    if lc[0]:
        dsig = {}
        dsig = {ids: sorted(set(list(dshort[ids].keys())
                                + list(dcomp[ids].keys())))
                for ids in lidsok}
    elif lc[1] or lc[2]:
        if lc[1]:
            dsig = [dsig]
        dsig = {ids: dsig for ids in lidsok}

    # Check content
    dout = {}
    for k0, v0 in dsig.items():
        lkeysok = sorted(set(list(dshort[k0].keys())
                             + list(dcomp[k0].keys())))
        if k0 not in lidsok:
            msg = "Only the following ids are relevant to Plasma2D:\n"
            msg += "    - %s"%str(lidsok)
            msg += "  => ids %s from dsig is ignored"%str(k0)
            warnings.warn(msg)
            continue
        lc = [v0 is None, type(v0) is str, type(v0) is list]
        if not any(lc):
            msg = "Each value in dsig must be either:\n"
            msg += "    - None\n"
            msg += "    - str : a valid shortcut\n"
            msg += "    - list of str: list of valid shortcuts\n"
            msg += "You provided:\n"
            msg += str(dsig)
            raise Exception(msg)
        if lc[0]:
            dsig[k0] = lkeysok
        if lc[1]:
            dsig[k0] = [dsig[k0]]
        if not all([ss in lkeysok for ss in dsig[k0]]):
            msg = "All requested signals must be valid shortcuts !\n"
            msg += "    - dsig[%s] = %s"%(k0, str(dsig[k0]))
            raise Exception(msg)

        # Check presence of minimum
        lc = [ss for ss in lscom if ss not in dsig[k0]]
        if len(lc) > 0:
            msg = "dsig[%s] does not have %s\n"%(k0,str(lc))
            msg += "    - dsig[%s] = %s"%(k0,str(dsig[k0]))
            raise Exception(msg)
        if any(['2d' in ss for ss in dsig[k0]]):
            for ss in lsmesh:
                if ss not in dsig[k0]:
                    dsig[k0].append(ss)
        dout[k0] = dsig[k0]
    return dout


def plasma_plot_args(plot, plot_X, plot_sig,
                     dsig=None):
    # Set plot
    if plot is None:
        plot = not (plot_sig is None and plot_X is None)

    if plot is True:
        # set plot_sig
        if plot_sig is None:
            lsplot = [ss for ss in list(dsig.values())[0]
                      if ('1d' in ss and ss != 't'
                          and all([sub not in ss
                                   for sub in ['rho','psi','phi']]))]
            if not (len(dsig) == 1 and len(lsplot) == 1):
                msg = ("Direct plotting only possible if\n"
                       + "sig_plot is provided, or can be derived from:\n"
                       + "\t- unique ids: {}\n\t".format(dsig.keys())
                       + "- unique non-(t, radius) 1d sig: {}".format(lsplot))
                raise Exception(msg)
            plot_sig = lsplot
        if type(plot_sig) is str:
            plot_sig = [plot_sig]

        # set plot_X
        if plot_X is None:
            lsplot = [ss for ss in list(dsig.values())[0]
                      if ('1d' in ss and ss != 't'
                          and any([sub in ss
                                   for sub in ['rho','psi','phi']]))]
            if not (len(dsig) == 1 and len(lsplot) == 1):
                msg = ("Direct plotting only possible if\n"
                       + "X_plot is provided, or can be derived from:\n"
                       + "\t- unique ids: {}\n".format(dsig.keys())
                       + "\t- unique non-t, 1d radius: {}".format(lsplot))
                raise Exception(msg)
            plot_X = lsplot
        if type(plot_X) is str:
            plot_X = [plot_X]
    return plot, plot_X, plot_sig
