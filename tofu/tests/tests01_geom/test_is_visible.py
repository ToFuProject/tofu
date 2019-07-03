import tofu.geom._GG as GG
import numpy as np


VP = np.array([[6.,8.,8.,6.,6.],[6.,6.,8.,8.,6.]])
VIn = np.array([[0.,-1.,0.,1.],[1.,0.,-1.,0.]])
VL = np.array([0.,1.])*2.*np.pi
SP0x = [6.,6.5,6.5,6.,6.]
SP0y = [6.,6.,6.5,6.5,6.]
SP1x = [7.5,8.,8.,7.5,7.5]
SP2x = [6.,6.5,6.5,6.,6.]
SP1y = [7.5,7.5,8.,8.,7.5]
SP2y = [7.5,7.5,8.,8.,7.5]

# import matplotlib
# print(matplotlib.matplotlib_fname())
# print(matplotlib.get_backend())
# matplotlib.use("qt5agg")
# print(matplotlib.get_backend())
# import matplotlib.pyplot as plt

# plt.plot(VP[0], VP[1])
# plt.plot(SP0x, SP0y)
# plt.plot(SP1x, SP1y)
# plt.plot(SP2x, SP2y)
# plt.show(block=True)

nstruct_lim = 3
nstruct_tot =1+2+1
lstruct_nlim=np.asarray([1, 2, 1])
SL0 = np.asarray([np.array([0.,1.])*2.*np.pi])
SL1 = np.asarray([np.array(ss)*2.*np.pi for ss in [[0.,1./3.],[2./3.,1.]]])
SL2 = np.asarray([np.array([2./3.,1.])*2.*np.pi])
lspolyx = np.asarray(SP0x + SP1x + SP2x)
lspolyy = np.asarray(SP0y + SP1y + SP2y)
lnvert = np.cumsum(np.ones(nstruct_tot, dtype=np.int64)*5)
lsvinx = np.asarray([VIn[0], VIn[0], VIn[0]]).flatten()
lsviny = np.asarray([VIn[1], VIn[1], VIn[1]]).flatten()
# ...
# Toroidal, with Struct
SL0 = np.asarray([None])
SL1 = np.asarray([np.array(ss)*np.pi for ss in [[0.,0.5],[1.,3./2.]]])
SL2 = np.asarray([np.array([0.5,3./2.])*np.pi])
lstruct_nlim = np.array([0, 2, 1])
nstruct_lim = 3
nstruct_tot =1+2+1
lstruct_nlim=np.asarray([0, 2, 1])

def test():
    # First point (in the center of poloidal plane
    pt0 = 7.
    pt1 = 0.
    pt2 = 7.
    # Other points (to check if visible or not)
    # first test point: same point (should be visible), in torus, out torus
    other_x = np.r_[7, 7.,  8.]
    other_y = np.r_[0, 0.1, 8.]
    other_z = np.r_[7, 7.5, 8.]
    npts = len(other_x)
    others = np.zeros((3,3))
    others[0,:] = other_x
    others[1,:] = other_y
    others[2,:] = other_z

    is_vis = GG.LOS_isVis_PtFromPts_VesStruct(pt0, pt1, pt2,
                                              others,
                                              ves_poly=VP,
                                              ves_norm=VIn,
                                              ves_lims=None,
                                              nstruct_tot=nstruct_tot,
                                              nstruct_lim=nstruct_lim,
                                              lnvert=lnvert,
                                              lstruct_polyx=lspolyx,
                                              lstruct_polyy=lspolyy,
                                              lstruct_nlim=lstruct_nlim,
                                              lstruct_lims=[SL0,SL1,SL2],
                                              lstruct_normx=lsvinx,
                                              lstruct_normy=lsviny,
                                              ves_type='Tor', test=True)
    # assert np.allclose(is_vis, [True, True, False])
    distance = np.sqrt(np.sum((others - np.tile(np.r_[pt0,pt1,pt2], (npts,1)).T)**2, axis=0))
    is_vis = GG.LOS_isVis_PtFromPts_VesStruct(pt0, pt1, pt2,
                                              others,
                                              k=distance,
                                              ves_poly=VP,
                                              ves_norm=VIn,
                                              ves_lims=None,
                                              nstruct_tot=nstruct_tot,
                                              nstruct_lim=nstruct_lim,
                                              lnvert=lnvert,
                                              lstruct_polyx=lspolyx,
                                              lstruct_polyy=lspolyy,
                                              lstruct_nlim=lstruct_nlim,
                                              lstruct_lims=[SL0,SL1,SL2],
                                              lstruct_normx=lsvinx,
                                              lstruct_normy=lsviny,
                                              ves_type='Tor', test=True)
    # assert np.allclose(is_vis, [True, True, False])
    pt_x = np.r_[7, 7.0, 0.0]
    pt_y = np.r_[0, 0.1, 0.0]
    pt_z = np.r_[7, 7.5, 0.0]
    npts2 = len(pt_x)
    pts2 = np.zeros((3,npts2))
    pts2[0,:] = pt_x
    pts2[1,:] = pt_y
    pts2[2,:] = pt_z
    are_vis = GG.LOS_areVis_PtsFromPts_VesStruct(pts2,
                                              others,
                                              ves_poly=VP,
                                              ves_norm=VIn,
                                              ves_lims=None,
                                              nstruct_tot=nstruct_tot,
                                              nstruct_lim=nstruct_lim,
                                              lnvert=lnvert,
                                              lstruct_polyx=lspolyx,
                                              lstruct_polyy=lspolyy,
                                              lstruct_nlim=lstruct_nlim,
                                              lstruct_lims=[SL0,SL1,SL2],
                                              lstruct_normx=lsvinx,
                                              lstruct_normy=lsviny,
                                              ves_type='Tor', test=True)
    # assert np.allclose(are_vis.flatten(), [True, True, False,
    #                                        True, True, False,
    #                                        False, False, True])

    pt_x = np.r_[7, 7.0, 0.0,7, 7.0, 0.0,7, 7.0, 0.0,7, 7.0, 0.0]
    pt_y = np.r_[0, 0.1, 0.0,0, 0.1, 0.0,0, 0.1, 0.0,0, 0.1, 0.0]
    pt_z = np.r_[7, 7.5, 0.0,7, 7.5, 0.0,7, 7.5, 0.0,7, 7.5, 0.0]
    npts2 = len(pt_x)
    pts2 = np.zeros((3,npts2))
    pts2[0,:] = pt_x
    pts2[1,:] = pt_y
    pts2[2,:] = pt_z
    are_vis = GG.LOS_areVis_PtsFromPts_VesStruct(pts2,
                                              others,
                                              ves_poly=VP,
                                              ves_norm=VIn,
                                              ves_lims=None,
                                              nstruct_tot=nstruct_tot,
                                              nstruct_lim=nstruct_lim,
                                              lnvert=lnvert,
                                              lstruct_polyx=lspolyx,
                                              lstruct_polyy=lspolyy,
                                              lstruct_nlim=lstruct_nlim,
                                              lstruct_lims=[SL0,SL1,SL2],
                                              lstruct_normx=lsvinx,
                                              lstruct_normy=lsviny,
                                              ves_type='Tor', test=True)

def test2():
    other_x = np.r_[7, 7.,  8.]
    other_y = np.r_[0, 0.1, 8.]
    other_z = np.r_[7, 7.5, 8.]
    npts = len(other_x)
    others = np.zeros((3,npts))
    others[0,:] = other_x
    others[1,:] = other_y
    others[2,:] = other_z
    r = np.ones(npts) * 0.1
    res = GG.Dust_calc_SolidAngle(others, r, others+0.1,
                                  VPoly=VP, VIn=VIn)
    print(res)

if __name__=='__main__':
    # from timeit import Timer
    # t = Timer("test()", "from __main__ import test")
    # print(t.timeit(number=10000))

    test2()
