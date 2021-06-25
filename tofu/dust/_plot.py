# Common
import numpy as np
import matplotlib.pyplot as plt

# ToFu-specific



def plot(dust, lax=None, Proj='All'):

    if dust.Ves is not None:
        lax = dust.Ves.plot(Lax=lax, Elt='P', Proj=Proj)
    if dust.LStruct is not None:
        for ss in dust.LStruct:
            lax = ss.plot(Lax=lax, Elt='P', Proj=Proj)

    if dust.traj is not None:
        lax = _plot_traj(dust, lax=lax, Proj=Proj)





def _plot_traj(dust, lax=None, Proj='All'):

    if dust.Type=='Tor':
        ptsCross = np.r_[np.hypot(dust.traj['pts'][0,:],dust.traj['pts'][1,:]),
                         dust.traj['pts'][2,:]]
    else:
        ptsCross = dust.traj['pts'][1:,:]
    ptsHor = dust.traj['pts'][:2,:]

    if lax[0] is not None:
        lax[0].plot(ptsCross[0,:],ptsCross[1,:], c='k', lw=1., ls='-')
    if lax[1] is not None:
        lax[1].plot(ptsHor[0,:],ptsHor[1,:], c='k', lw=1., ls='-')
    return lax
