# External modules
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.gridspec as mplgrid
from mpl_toolkits.mplot3d import Axes3D
import tofu as tf
# Nose-specific
from nose import with_setup # optional

# ToFu-specific
import tofu.geom._GG as GG

# .. First configuration ....................................................
ves_poly1 = np.zeros((3,9))
x1 = np.r_[2,4,6,6,4,3,4,3,2.0]
y1 = np.r_[2,0,2,5,2,2,3,4,3.0]
ves_poly1[0] = x1
ves_poly1[1] = y1
# .. Second configuration ..................................................
x2 = np.r_[0,3.5,5.5,7,8,7, 6,5,3,4]
y2 = np.r_[2.5,0,1.5,1,5,4.5, 6,3,4,8]
z2 = np.array([0 if xi < 5. else 1. for xi in x2])
npts = np.size(x2)
ves_poly2 = np.zeros((3,npts))
ves_poly2[0] = x2
ves_poly2[1] = y2
ves_poly2[2] = z2
#  === Creating configurations tabs ===
vignetts = [ves_poly1, ves_poly2]
lnvert = np.r_[9, npts]
# === Ray tabs ====
num_ray = 5
rays_origin = np.zeros((3, num_ray))
rays_direct = np.zeros((3, num_ray))
# -- First ray
orig = np.r_[3.75, 2.5, -2]
dire = np.r_[0, 0,  1]
rays_origin[:,0] = orig
rays_direct[:,0] = dire
# -- Second ray
orig = np.r_[5, 3.1, -2]
dire = np.r_[0, 0,  1]
rays_origin[:,1] = orig
rays_direct[:,1] = dire
# -- Third ray
orig = np.r_[0, 0, 5]
dire = np.r_[4, 1,  -5]/2.
rays_origin[:,2] = orig
rays_direct[:,2] = dire
# ==== 3D TESTS ====
orig = np.r_[0, 2.5, 1]
fina = np.r_[6.1, 2., 0]
dire = fina - orig
rays_origin[:,3] = orig
rays_direct[:,3] = dire
# Another ray
orig2 = np.r_[0, 2.5, 1]
fina2 = np.r_[6., 6., 0]
dire2 = fina2 - orig2
rays_origin[:,4] = orig2
rays_direct[:,4] = dire2

print(type(rays_origin), type(rays_direct),
      type(vignetts), type(lnvert))
out = GG.vignetting(rays_origin, rays_direct,
                    vignetts, lnvert)

print("out = ", out)
