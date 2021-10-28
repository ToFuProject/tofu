

import os


import numpy as np
import matplotlib.pyplot as plt


_PATH_HERE = os.path.dirname(__file__)
_PATH_TOFU = os.path.dirname(_PATH_HERE)


cwd = os.getcwd()
os.chdir(_PATH_TOFU)
import tofu as tf
os.chdir(cwd)


def run(
    plot_emis=False,
    plot_cam=False,
    plot_sino=False,
    plot_mesh=False,
    plot_data=False,
    plot_inv=True,
    vmin=0,
    vmax=0.8,
    lbs=['try1-bs1', 'try1-bs2'],
):

    # ------------
    # prepare data

    # load config
    conf0 = tf.load_config('ITER-V0')
    conf = tf.load_config('ITER-V1')

    # create mesh and bsplines
    mesh = tf.data.Mesh2DRect.from_Config(
        config=conf0, res=0.15, key='try1', deg=0,
    )
    mesh.add_bsplines(deg=1)
    mesh.add_bsplines(deg=2)

    # sample mesh
    R, Z = mesh.get_sample_mesh(res=0.05, grid=True)

    # define emis func
    def emis(pts, t=None):
        R = np.hypot(pts[0, :], pts[1, :])
        Z = pts[2, :]
        return (
            np.exp(-(R-6.3 + (Z-0.5)**2/3**2)**2/1.5**2 - (Z-0.5)**2/2.5**2)
            - 0.7*np.exp(-(R-6)**2/0.6**2 - (Z-0.5)**2/1.**2)
        )

    # create values on sampled mesh
    pts = np.array([R.ravel(), np.zeros((R.size,)), Z.ravel()])
    val = emis(pts).reshape(R.shape)

    # compute extent for imshow
    extent = (R.min(), R.max(), Z.min(), Z.max())

    # set pts out of vessel to nan
    isin = conf0.Ves.V0.isInside(pts).reshape(R.shape)
    val[~isin] = np.nan

    # ------------------
    # compute cameras

    # Create cameras dict
    focal = 0.08
    sensors_nb = 20
    sensors_size = 0.1

    dcam = {
        'c0': {
            'pinhole': [5.5, 0, 5],
            'orientation': [-np.pi/2., 0, 0],
        },
        'c1': {
            'pinhole': [6.5, 0, 4.5],
            'orientation': [-5*np.pi/8, 0, 0],
        },
        'c2': {
            'pinhole': [7.5, 0, 3.8],
            'orientation': [-6*np.pi/8, 0, 0],
        },
        'c3': {
            'pinhole': [8.5, 0, 2.5],
            'orientation': [-7*np.pi/8, 0, 0],
        },
        'c4': {
            'pinhole': [8.8, 0, 1.3],
            'orientation': [np.pi, 0, 0],
        },
        'c5': {
            'pinhole': [9., 0, 0.],
            'orientation': [np.pi, 0, 0],
        },
        'c6': {
            'pinhole': [8.5, 0, -1.],
            'orientation': [7*np.pi/8, 0, 0],
        },
        'c7': {
            'pinhole': [7.5, 0, -2.5],
            'orientation': [6*np.pi/8, 0, 0],
        },
        'c8': {
            'pinhole': [6.5, 0, -3.6],
            'orientation': [5*np.pi/8, 0, 0],
        },
        'c9': {
            'pinhole': [5., 0, -4.],
            'orientation': [3.*np.pi/8, 0, 0],
        },
        'c10': {
            'pinhole': [4., 0, -3.],
            'orientation': [2*np.pi/8, 0, 0],
        },
    }


    # create camera objects
    for ii, (k0, v0) in enumerate(dcam.items()):
        try:
            dcam[k0]['obj'] = tf.geom.utils.create_CamLOS1D(
                focal=focal,
                sensor_nb=sensors_nb,
                sensor_size=sensors_size,
                Diag='Bolo',
                Exp='ITER',
                Name=k0,
                config=conf0,
                **v0
            )
        except Exception as err:
            import pdb; pdb.set_trace()     # DB
            pass

    # macro-cams
    lcam = [
        dcam['c0']['obj'] + dcam['c2']['obj'] + dcam['c4']['obj']
        + dcam['c6']['obj'] + dcam['c8']['obj'] + dcam['c10']['obj'],
        dcam['c0']['obj'] + dcam['c1']['obj'] + dcam['c2']['obj']
        + dcam['c3']['obj'] + dcam['c4']['obj'] + dcam['c5']['obj'],
        dcam['c0']['obj'] + dcam['c1']['obj'] + dcam['c2']['obj']
        + dcam['c3']['obj'] + dcam['c4']['obj'] + dcam['c5']['obj']
        + dcam['c6']['obj'] + dcam['c7']['obj'] + dcam['c8']['obj']
        + dcam['c9']['obj'] + dcam['c10']['obj'],
    ]

    # -------------------------
    # compute geometry matrices

    if plot_inv:
        for ii, kbs in enumerate(lbs):
            for jj, cc in enumerate(lcam):
                kch = None if ii == 0 else f'chan{jj}'
                mesh.add_geometry_matrix(
                    key=kbs, cam=cc, key_chan=kch, res=0.005,
                )

    # ----------------------
    # compute synthetic data 

    if plot_data or plot_inv:
        for ii, cc in enumerate(lcam):
            data, units = cc.calc_signal(
                emis,
                res=0.01,
                returnas=np.ndarray,
                plot=plot_data,
            )
            mesh.add_data(key=f'data{ii}', data=data[0], ref=(f'chan{ii}',))

    # ------------------
    # compute inversions

    if plot_inv:
        for ii, kbs in enumerate(lbs):
            for jj, cc in enumerate(lcam):
                kmat = f'matrix{ii*3+jj}'
                kdat = f'data{jj}'
                op = 'D1N2' if int(kbs[-1]) < 2 else 'D2N2'
                mesh.add_inversion(
                    key_matrix=kmat,
                    key_data=kdat,
                    sigma=0.01,
                    operator=op,
                    geometry='toroidal',
                    verb=1,
                )

    # -------
    # plot

    # plot emissivity
    if plot_emis:
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0., 0.8, 0.8])
        im = ax.imshow(
            val.T,
            extent=extent,
            origin='lower',
            cmap='viridis',
            vmin=vmin,
            vmax=vmax,
        )
        dax = conf.plot(proj='cross', dLeg=False, lax=ax)
        plt.colorbar(mappable=im, ax=ax)

    # plot mesh
    if plot_mesh:
        dax = mesh.plot_mesh()
        dax = conf.plot(proj='cross', dLeg=False, lax=dax['cross']['ax'])

    # plot sinograms
    if plot_sino:
        for cc in lcam:
            cc.set_dsino([6., 0.5])
            dax = cc.plot_sino()

    # plot cam
    if plot_cam:
        for cc in lcam:
            dax = cc.plot(proj='cross')

    # inversions
    if plot_inv:
        for ii, kbs in enumerate(lbs):
            for jj, cc in enumerate(lcam):
                kinv = f'inv{ii*3+jj}'
                dax = mesh.plot_inversion(
                    key=kinv,
                    vmin=vmin,
                    vmax=vmax,
                )

                # err
                kinv = f'inv{ii*3+jj}'
                vij = mesh.interp2d(
                    key=kinv,
                    R=R,
                    Z=Z,
                    grid=False,
                )[0]
                err = val - vij

                fig = plt.figure()
                ax = fig.add_axes([0.1, 0., 0.8, 0.8])
                tit = (
                    f"{kbs} - cam{jj}\n"
                    f"err = {100*np.nansum(err)/np.nansum(val):3.2f} %"
                )
                ax.set_title(tit)
                im = ax.imshow(
                    err.T,
                    extent=extent,
                    origin='lower',
                    cmap='seismic',
                    vmin=-0.3,
                    vmax=0.3,
                )
                dax = conf.plot(proj='cross', dLeg=False, lax=ax)
                plt.colorbar(mappable=im, ax=ax)

