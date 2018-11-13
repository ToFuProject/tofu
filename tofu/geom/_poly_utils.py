# This function contain functions that probably should be defined by functions
# directly in ToFu (by D. Vezinet). In the mean time we get this messy functions...
from tofu.geom._GG import CoordShift
import matplotlib.pyplot as plt
import numpy as np


def get_bbox_poly_extruded(lpoly):
    dim = lpoly.shape[0]
    assert dim == 2, "Poly should be of dim 2 (expected shape=(2,Npoints), got "+str(lpoly.shape)+")"
    # # Scatter plot on top of lines
    # plt.subplot(311)
    # plt.plot(lpoly[0,:], lpoly[1,:], 'C3', zorder=1, lw=3)
    # plt.plot(-lpoly[0,:], lpoly[1,:], 'C3', zorder=1, lw=3)
    # plt.scatter(lpoly[0,:], lpoly[1,:], s=120, zorder=2)
    # plt.scatter(-lpoly[0,:], lpoly[1,:], s=120, zorder=2)
    # plt.title('Poly in R,Z')
    # plt.tight_layout()
    npts = lpoly.shape[1]
    ones = np.ones_like(lpoly[0,:])*np.pi/2.
    rzphi_poly = np.zeros((3, npts))
    rzphi_poly[0, :] = lpoly[0,:]
    rzphi_poly[1, :] = lpoly[1,:]
    rzphi_poly[2, :] = ones
    xyz_poly = CoordShift(rzphi_poly, In='(R,Z,Phi)', Out="(X,Y,Z)")
    ymax = xyz_poly[1,:].max()
    ymin = -ymax + 0.
    xmin = -ymax + 0.
    xmax = ymax + 0.
    zmin = xyz_poly[2,:].min()
    zmax = xyz_poly[2,:].max()
    # plt.clf()
    # plt.subplot(311)
    # plt.plot(    xyz_poly[0,:],  xyz_poly[1,:], 'C3', zorder=1, lw=3)
    # plt.plot(    xyz_poly[0,:], -xyz_poly[1,:], 'C3', zorder=1, lw=3)
    # plt.scatter( xyz_poly[0,:],  xyz_poly[1,:], s=120, zorder=2)
    # plt.scatter( xyz_poly[0,:], -xyz_poly[1,:], s=120, zorder=2)
    # plt.plot([xmin, xmin, xmax, xmax, xmin], [ymin, ymax, ymax, ymin, ymin], 'C3', zorder=1, lw=3)
    # plt.title('Poly in X,Y')
    # plt.subplot(312)
    # plt.plot(    xyz_poly[0,:], xyz_poly[2,:], 'C3', zorder=1, lw=3)
    # plt.plot(   -xyz_poly[0,:], xyz_poly[2,:], 'C3', zorder=1, lw=3)
    # plt.scatter( xyz_poly[0,:], xyz_poly[2,:], s=120, zorder=2)
    # plt.scatter(-xyz_poly[0,:], xyz_poly[2,:], s=120, zorder=2)
    # plt.plot([xmin, xmin, xmax, xmax, xmin], [zmin, zmax, zmax, zmin, zmin], 'C3', zorder=1, lw=3)
    # plt.title('Poly in X,Z')
    # plt.tight_layout()    
    # plt.subplot(313)
    # plt.plot(    xyz_poly[1,:], xyz_poly[2,:], 'C3', zorder=1, lw=3)
    # plt.plot(   -xyz_poly[1,:], xyz_poly[2,:], 'C3', zorder=1, lw=3)
    # plt.scatter( xyz_poly[1,:], xyz_poly[2,:], s=120, zorder=2)
    # plt.scatter(-xyz_poly[1,:], xyz_poly[2,:], s=120, zorder=2)
    # plt.plot([ymin, ymin, ymax, ymax, ymin], [zmin, zmax, zmax, zmin, zmin], 'C3', zorder=1, lw=3)
    # plt.title('Poly in Y,Z')
    # plt.tight_layout()    
    # plt.show(block=True)
    return [xmin, ymin, zmin, xmax, ymax, zmax]

