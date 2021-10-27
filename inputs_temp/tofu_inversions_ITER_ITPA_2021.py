

import os


cwd = os.getcwd()
os.chdir('~/ToFu_All/tofu_git/tofu')
import tofu as tf
os.chdir(cwd)



conf0 = tf.load_config('ITER-V0')
conf = tf.load_config('ITER')


mesh = tf.data.Mesh2DRect.from_Config(config=conf0, res=0.1, key='try1', deg=0)
mesh.add_bsplines(deg=1)
mesh.add_bsplines(deg=2)


R, Z = mesh.get_sample_mesh(res=0.01, grid=True)

val = (
    np.exp(-(R-6.3)**2/1.5**2 - (Z-0.5)**2/2.5**2)
    - 0.7*np.exp(-(R-6)**2/0.6**2 - (Z-0.5)**2/1.**2)
)


extent = (R.min(), R.max(), Z.min(), Z.max())

plt.figure(); plt.imshow(val.T, extent=extent, origin='lower')
dax = conf.plot(proj='cross', dLeg=False, lax=plt.gca())


dax = mesh.plot_mesh()
dax = conf.plot(proj='cross', dLeg=False, lax=plt.gca())

# Create cameras
focal = 0.08
sensors_nb = 20
sensors_size = 0.1

dcam = {
    'c1': {
        'pinhole': [5, 0, 5],
        'orientation': [-np.pi/2., 0, 0],
    },
    'c2': {
        'pinhole': [6, 0, 4.3],
        'orientation': [-5*np.pi/8, 0, 0],
    },
    'c3': {
        'pinhole': [7, 0, 3.8],
        'orientation': [-6*np.pi/8, 0, 0],
    },
    'c4': {
        'pinhole': [8, 0, 2.7],
        'orientation': [-7*np.pi/8, 0, 0],
    },
    'c5': {
        'pinhole': [8.6, 0, 0.],
        'orientation': [np.pi, 0, 0],
    },
}


for ii, (k0, v0) in enumerate(dcam.items()):
    dcam['obj'] = tf.geom.utils.create_CamLOS1D(
        focal=focal,
        sensors_nb=sensors_nb,
        sensors_size=sensors_size,
        Diag='Bolo',
        Exp='ITER',
        Name=k0,
        config=conf0,
        **v0
    )

# macro-cams
lcam = [
    dcam['c0']['obj'] + dcam['c2']['obj'] + dcam['c4']['obj']
    + dcam['c6']['obj'] + dcam['c8']['obj'] + dcam['c10']['obj'],
    dcam['c0']['obj'] + dcam['c1']['obj'] + dcam['c2']['obj']
    + dcam['c3']['obj'] + dcam['c4']['obj'] + dcam['c5']['obj'],
    dcam['c0']['obj'] + dcam['c1']['obj'] + dcam['c2']['obj']
    + dcam['c3']['obj'] + dcam['c4']['obj'] + dcam['c5']['obj']
    + dcam['c6']['obj'] + dcam['c7']['obj'] + dcam['c8']['obj'],
    + dcam['c9']['obj'] + dcam['c10']['obj'],
]

# geometry matrices
for kbs in ['try1-bs0', 'try1-bs1', 'try1-bs2']:
    for cc in lcam:
        mesh.add_geometry_matrix(key=kbs, cam=cc)



# synthetic data


# inversions
