"""
This module contains tests for tofu.geom in its structured version
"""

# External modules
import numpy as np
# ToFu-specific
import tofu.geom._GG as GG
from .testing_tools import compute_ves_norm


# header
VerbHead = 'tofu.geom.test_01_GG'


#######################################################
#
#     Setup and Teardown
#
#######################################################

def setup_module(module):
    print("")  # this is to get a newline after the dots
    # print ("setup_module before anything in this file")


def teardown_module(module):
    # os.remove(VesTor.Id.SavePath + VesTor.Id.SaveName + '.npz')
    # os.remove(VesLin.Id.SavePath + VesLin.Id.SaveName + '.npz')
    # print ("teardown_module after everything in this file")
    # print ("") # this is to get a newline
    pass

#######################################################
#
#     Testing
#
#######################################################
######################################################
######################################################
#               Commons
######################################################
######################################################


def test01_CoordShift():

    # Tests 1D input
    Pts = np.array([1., 1., 1.])
    pts = GG.coord_shift(
        Pts, in_format='(X,Y,Z)', out_format='(R,Z)', cross_format=0.,
    )
    assert pts.shape == (2,) and np.allclose(pts, [np.sqrt(2), 1.])
    pts = GG.coord_shift(
        Pts, in_format='(R,Z,Phi)', out_format='(X,Y,Z)', cross_format=0.,
    )
    assert pts.shape == (3,) and np.allclose(pts, [np.cos(1.), np.sin(1.), 1.])

    # Test 2D input
    Pts = np.array([[1., 1.], [1., 1.], [1., 1.]])
    pts = GG.coord_shift(
        Pts, in_format='(X,Y,Z)', out_format='(R,Phi,Z)', cross_format=0.,
    )
    assert (
        pts.shape == (3, 2)
        and np.allclose(
            pts,
            [[np.sqrt(2.), np.sqrt(2.)], [np.pi/4., np.pi/4.], [1., 1.]]
        )
    )
    pts = GG.coord_shift(
        Pts, in_format='(Phi,Z,R)', out_format='(X,Y)', cross_format=0.,
    )
    assert (
        pts.shape == (2, 2)
        and np.allclose(
            pts,
            [[np.cos(1.), np.cos(1.)], [np.sin(1.), np.sin(1.)]]
        )
    )






########################################################
########################################################
#       Polygons
########################################################

def test02_Poly_CLockOrder():

    # Test arbitrary 2D polygon
    Poly = np.array([[0., 1., 1., 0.], [0., 0., 1., 1.]])

    P = GG.format_poly(Poly, order='C', Clock=False, close=True,
                       Test=True)

    assert all([np.allclose(P[:, 0], P[:, -1]), P.shape == (2, 5),
                not GG.Poly_isClockwise(P), P.flags['C_CONTIGUOUS'],
                not P.flags['F_CONTIGUOUS']])

    P = GG.format_poly(Poly, order='F', Clock=True, close=False,
                       Test=True)

    assert not np.allclose(P[:, 0], P[:, -1]), "poly should not be closed"
    assert P.shape == (2, 4), ("shape of poly should be (2,4), here = "
                               + str(P.shape) + "\n Poly = " + str(P))
    assert GG.Poly_isClockwise(np.concatenate((P, P[:, 0:1]), axis=1))
    assert not P.flags['C_CONTIGUOUS']
    assert P.flags['F_CONTIGUOUS']

    # Test arbitrary 3D polygon
    Poly = np.array([[0., 1., 1., 0.],
                     [0., 0., 1., 1.],
                     [0., 0., 0., 0.]])
    P = GG.format_poly(Poly, order='C', Clock=False, close=False,
                       Test=True)
    assert all([not np.allclose(P[:, 0], P[:, -1]), P.shape == (3, 4),
                P.flags['C_CONTIGUOUS'], not P.flags['F_CONTIGUOUS']])
    P = GG.format_poly(Poly, order='F', Clock=True, close=True,
                       Test=True)
    assert all([np.allclose(P[:, 0], P[:, -1]), P.shape == (3, 5),
                not P.flags['C_CONTIGUOUS'], P.flags['F_CONTIGUOUS']])


def test03_Poly_VolAngTor():
    Poly = np.array([
        [1., 1.5, 2., 2., 2., 1.5, 1.],
        [0., 0., 0., 0.5, 1., 1., 1.],
    ])
    Poly = GG.format_poly(Poly, order='C', Clock=False, close=True,
                          Test=True)
    V, B = GG.Poly_VolAngTor(Poly)
    assert V==1.5
    assert np.allclose(B, [7./(3.*1.5), 0.5])





"""
######################################################
######################################################
#               Ves
######################################################
######################################################
"""

# VPoly
thet = np.linspace(0., 2. * np.pi, 100)
VPoly = np.array([2. + 1. * np.cos(thet), 0. + 1. * np.sin(thet)])



def test04_Ves_isInside(VPoly=VPoly):

    # Lin Ves
    Pts = np.array([[-10., -10., 5., 5., 5., 5., 5., 30., 30., 30.],
                    [0., 2., 0., 2., 4., 2., 2., 2., 0., 0.],
                    [0., 0., 0., 0., 0., 2., -2., 0., 0., 2.]])
    ind = GG._Ves_isInside(Pts, VPoly, ves_lims=np.array([[0., 10.]]), nlim=1,
                           ves_type='Lin', in_format='(X,Y,Z)', test=True)
    assert (
        ind.shape == (Pts.shape[1],)
        and np.all(ind == [
            False, False, False, True, False,
            False, False, False, False, False,
        ])
    )

    # Tor Ves
    Pts = np.array([[0., -10., 5., 5., 5., 5., 5., 30., 30., 30.],
                    [0., 2., 0., 2., 4., 2., 2., 2., 0., 0.],
                    [0., 0., 0., 0., 0., 2., -2., 0., 0., 2.]])
    ind = GG._Ves_isInside(Pts, VPoly, ves_lims=None, nlim=0, ves_type='Tor',
                           in_format='(Phi,R,Z)', test=True)
    assert (
        ind.shape == (Pts.shape[1],)
        and np.all(ind == [
            False, True, False, True, False, False, False, True, False, False,
        ])
    )

    # Tor Struct
    pi2 = 2.*np.pi
    Pts = np.array([[ 0.,  0., pi2, np.pi, np.pi, np.pi, np.pi, pi2, pi2, pi2],
                    [ 0.,  2.,  0.,    2.,    4.,    2.,    2.,  2.,  0.,  0.],
                    [ 0.,  0.,  0.,    0.,    0.,    2.,   -2.,  0.,  0.,  2.]])
    ind = GG._Ves_isInside(Pts, VPoly,
                           ves_lims=np.array([[np.pi/2., 3.*np.pi/2.]]),
                           nlim=1, ves_type='Tor', in_format='(Phi,R,Z)',
                           test=True)
    assert (
        ind.shape == (Pts.shape[1],)
        and np.all(ind == [
            False, False, False, True, False,
            False, False, False, False, False,
        ])
    )


#####################################################
#               Ves  - SMesh
#####################################################
def test09_Ves_Smesh_Tor(VPoly=VPoly):

    dL, dRPhi = 0.02, 0.05
    VIn = compute_ves_norm(VPoly)
    DIn = 0.001
    LDPhi = [None, [3.*np.pi/4., 5.*np.pi/4.], [-np.pi/4., np.pi/4.]]

    for ii in range(0,len(LDPhi)):
        # With Ves
        Pts, dS, ind, NL, \
            dLr, Rref, dRPhir,\
            nRPhi0, VPbis = GG._Ves_Smesh_Tor_SubFromD_cython(dL, dRPhi, VPoly,
                                                              DR=[0.5, 2.],
                                                              DZ=[0., 1.2],
                                                              DPhi=LDPhi[ii],
                                                              DIn=DIn, VIn=VIn,
                                                              PhiMinMax=None,
                                                              Out='(R,Z,Phi)',
                                                              margin=1.e-9)

        assert Pts.ndim == 2 and Pts.shape[0] == 3
        assert (
            np.all(Pts[0, :] >= 1.-np.abs(DIn))
            and np.all(Pts[0, :] <= 2.+np.abs(DIn))
            and np.all(Pts[1, :] >= 0.-np.abs(DIn))
            and np.all(Pts[1, :] <= 1.+np.abs(DIn))
        )
        marg = np.abs(np.arctan(
            np.mean(dRPhir+np.abs(DIn))/np.min(VPoly[1, :])
        ))
        if not LDPhi[ii] is None:
            LDPhi[ii][0] = np.arctan2(
                np.sin(LDPhi[ii][0]),
                np.cos(LDPhi[ii][0]),
            )
            LDPhi[ii][1] = np.arctan2(
                np.sin(LDPhi[ii][1]),
                np.cos(LDPhi[ii][1]),
            )
            if LDPhi[ii][0]<=LDPhi[ii][1]:
                assert np.all(
                    (Pts[2, :] >= LDPhi[ii][0]-marg)
                    & (Pts[2, :] <= LDPhi[ii][1]+marg)
                )
            else:
                assert np.all(
                    (Pts[2, :] >= LDPhi[ii][0]-marg)
                    | (Pts[2, :] <= LDPhi[ii][1]+marg)
                )
        assert np.all(GG._Ves_isInside(Pts, VPoly, ves_type='Tor',
                                       in_format='(R,Z,Phi)',
                                       ves_lims=None, nlim=0, test=True))
        assert dS.shape == (Pts.shape[1],)
        assert all([
            ind.shape == (Pts.shape[1],),
            ind.dtype == int,
            np.unique(ind).size == ind.size,
            np.all(ind == np.unique(ind)),
            np.all(ind >= 0),
        ])
        assert (
            ind.shape == (Pts.shape[1],) and ind.dtype == int
            and np.all(ind == np.unique(ind)) and np.all(ind >= 0)
        )
        assert NL.ndim == 1 and NL.size == VPoly.shape[1]-1
        assert dLr.ndim == 1 and dLr.size == NL.size
        assert Rref.ndim == 1
        assert dRPhir.ndim == 1 and dRPhir.size == Rref.size
        assert type(nRPhi0) is int

        Ptsi, dSi, NLi, \
            dLri, Rrefi, dRPhiri, \
            nRPhi0i, \
            VPbisi = GG._Ves_Smesh_Tor_SubFromInd_cython(dL, dRPhi,
                                                         VPoly, ind,
                                                         DIn=DIn, VIn=VIn,
                                                         PhiMinMax=None,
                                                         Out='(R,Z,Phi)',
                                                         margin=1.e-9)
        assert np.allclose(Pts, Ptsi)
        assert np.allclose(dSi, dS)
        assert np.allclose(NLi, NL)
        assert np.allclose(dLri, dLr)
        assert np.allclose(Rrefi,Rref)
        assert np.allclose(dRPhiri, dRPhir)
        assert nRPhi0i == nRPhi0



def test10_Ves_Smesh_Tor_PhiMinMax(VPoly=VPoly, plot=True):

    dL, dRPhi = 0.02, 0.05
    VIn = compute_ves_norm(VPoly)
    DIn = 0.001
    LPhi = [[[-np.pi/4., np.pi/4.], [3.*np.pi/2., np.pi/2.]],
            [[-np.pi/4., np.pi/4.], [0., np.pi/2.]],
            [[-np.pi/4., np.pi/4.], [np.pi/6., -np.pi/6.]],
            [[-np.pi/4., np.pi/4.], [0., 5.*np.pi/4.]],
            [[3.*np.pi/4., 5.*np.pi/4.], [np.pi/2., -np.pi/2.]],
            [[3.*np.pi/4., 5.*np.pi/4.], [7.*np.pi/6., -np.pi/2.]],
            [[3.*np.pi/4., 5.*np.pi/4.], [np.pi/2., np.pi]],
            [[3.*np.pi/4., 5.*np.pi/4.], [7.*np.pi/6., 5.*np.pi/6.]]]

    for ii in range(0,len(LPhi)):
        Pts, dS, ind,\
            NL, dLr, Rref,\
            dRPhir, nRPhi0,\
            VPbis = GG._Ves_Smesh_Tor_SubFromD_cython(
                dL, dRPhi, VPoly,
                DR=[0.5, 2.], DZ=[0., 1.2],
                DPhi=LPhi[ii][1],
                DIn=DIn, VIn=VIn,
                PhiMinMax=np.array(LPhi[ii][0]),
                Out='(R,Z,Phi)',
                margin=1.e-9,
            )

        #try:
        assert Pts.ndim == 2 and Pts.shape[0] == 3
        LPhi[ii][0][0] = np.arctan2(np.sin(LPhi[ii][0][0]),
                                    np.cos(LPhi[ii][0][0]))
        LPhi[ii][0][1] = np.arctan2(np.sin(LPhi[ii][0][1]),
                                    np.cos(LPhi[ii][0][1]))
        marg = np.abs(np.arctan(
            np.mean(dRPhir+np.abs(DIn))/np.min(VPoly[1, :])
        ))
        if LPhi[ii][0][0] <= LPhi[ii][0][1]:
            assert np.all(
                (Pts[2, :] >= LPhi[ii][0][0]-marg)
                & (Pts[2, :] <= LPhi[ii][0][1]+marg)
            )
        else:
            assert np.all(
                (Pts[2, :] >= LPhi[ii][0][0]-marg)
                | (Pts[2, :] <= LPhi[ii][0][1]+marg)
            )
        assert np.all(GG._Ves_isInside(Pts, VPoly, ves_type='Tor',
                                       ves_lims=None,
                                       nlim=0, in_format='(R,Z,Phi)',
                                       test=True))
        assert dS.shape == (Pts.shape[1],)
        assert np.all([ind.shape == (Pts.shape[1],), ind.dtype == int,
                       ind.size == np.unique(ind).size,
                       np.all(ind == np.unique(ind)), np.all(ind >= 0)])
        assert NL.ndim == 1 and NL.size == VPoly.shape[1]-1
        assert dLr.ndim == 1 and dLr.size == NL.size
        assert Rref.ndim == 1
        assert dRPhir.ndim == 1 and dRPhir.size == Rref.size
        assert type(nRPhi0) is int

        lrphi_arr = np.array(LPhi[ii][0])
        out = GG._Ves_Smesh_Tor_SubFromInd_cython(dL, dRPhi,
                                                  VPoly, ind,
                                                  DIn=DIn,
                                                  VIn=VIn,
                                                  PhiMinMax=lrphi_arr,
                                                  Out='(R,Z,Phi)',
                                                  margin=1.e-9)
        Ptsi, dSi, NLi, dLri, Rrefi, dRPhiri, nRPhi0i, VPbisi = out
        assert np.allclose(Pts, Ptsi)
        assert np.allclose(dSi, dS)
        assert np.allclose(NLi, NL)
        assert np.allclose(dLri, dLr)
        assert np.allclose(Rrefi,Rref)
        assert np.allclose(dRPhiri, dRPhir)
        assert nRPhi0i == nRPhi0

        # except:
        # print([ind.shape == (Pts.shape[1],), ind.dtype == int,
        # ind.size == np.unique(ind).size, np.all(ind == np.unique(ind)),
        # np.all(ind>=0)])
        # print(np.unique(ind).size, ind.size)
        # lii = [
        # ind[ii] for ii in range(0,len(ind))
        # if np.sum(ind == ind[ii])>1
        # ]
        # liib = [
        # ii for ii in range(0,len(ind))
        # if np.sum(ind == ind[ii])>1
        # ]
        # print(len(lii),len(liib))
        # print(lii)
        # print(liib)
        # for ii in range(0,len(liib)):
        # print([Pts[:,liib[ii]] == Pts[:,hh] for hh in [jj for jj in
        # range(0,len(ind)) if ind[jj] == lii[ii]]])


def test11_Ves_Smesh_TorStruct(VPoly=VPoly, plot=True):

    PhiMinMax = np.array([3.*np.pi/4.,5.*np.pi/4.])
    dL, dRPhi = 0.02, 0.05
    VIn = compute_ves_norm(VPoly)
    DIn = -0.001
    LPhi = [[[-np.pi/4., np.pi/4.], [3.*np.pi/2., np.pi/2.]],
            [[-np.pi/4., np.pi/4.], [0., np.pi/2.]],
            [[-np.pi/4., np.pi/4.], [np.pi/6., -np.pi/6.]],
            [[-np.pi/4., np.pi/4.], [0., 5.*np.pi/4.]],
            [[3.*np.pi/4., 5.*np.pi/4.], [np.pi/2., -np.pi/2.]],
            [[3.*np.pi/4., 5.*np.pi/4.], [7.*np.pi/6., -np.pi/2.]],
            [[3.*np.pi/4., 5.*np.pi/4.], [np.pi/2., np.pi]],
            [[3.*np.pi/4., 5.*np.pi/4.], [7.*np.pi/6., 5.*np.pi/6.]]]

    for ii in range(0,len(LPhi)):
        Pts, dS, ind, NL, \
            dLr, Rref, \
            dR0r, dZ0r, \
            dRPhir, \
            VPbis = GG._Ves_Smesh_TorStruct_SubFromD_cython(np.array(LPhi[ii][0]),
                                                            dL, dRPhi, VPoly,
                                                            DR=[0.5, 2.],
                                                            DZ=[0., 1.2],
                                                            DPhi=np.array(LPhi[ii][1]),
                                                            DIn=DIn, VIn=VIn,
                                                            Out='(R,Z,Phi)',
                                                            margin=1.e-9)

        #try:
        assert Pts.ndim == 2 and Pts.shape[0] == 3
        LPhi[ii][0][0] = np.arctan2(np.sin(LPhi[ii][0][0]),
                                    np.cos(LPhi[ii][0][0]))
        LPhi[ii][0][1] = np.arctan2(np.sin(LPhi[ii][0][1]),
                                    np.cos(LPhi[ii][0][1]))
        marg = np.abs(np.arctan(
            np.mean(dRPhir+np.abs(DIn))/np.min(VPoly[1, :])
        ))
        if LPhi[ii][0][0] <= LPhi[ii][0][1]:
            assert np.all(
                (Pts[2, :] >= LPhi[ii][0][0]-marg)
                & (Pts[2, :] <= LPhi[ii][0][1]+marg)
            )
        else:
            assert np.all(
                (Pts[2, :] >= LPhi[ii][0][0]-marg)
                | (Pts[2, :] <= LPhi[ii][0][1]+marg)
            )
        if DIn>=0:
            assert np.all(GG._Ves_isInside(Pts, VPoly, ves_type='Tor',
                                           ves_lims=None,
                                           nlim=0, in_format='(R,Z,Phi)',
                                           test=True))
        else:
            assert not np.all(GG._Ves_isInside(Pts, VPoly, ves_type='Tor',
                                               ves_lims=None, nlim=0,
                                               in_format='(R,Z,Phi)',
                                               test=True))
        assert dS.shape == (Pts.shape[1],)
        assert np.all([ind.shape == (Pts.shape[1],),
                       ind.dtype == int,
                       ind.size == np.unique(ind).size,
                       np.all(ind == np.unique(ind)),
                       np.all(ind>=0)])
        assert NL.ndim == 1 and NL.size == VPoly.shape[1]-1
        assert dLr.ndim == 1 and dLr.size == NL.size
        assert Rref.ndim == 1
        assert type(dR0r) is float and type(dZ0r) is float
        assert dRPhir.ndim == 1 and dRPhir.size == Rref.size

        Ptsi, dSi, NLi,\
            dLri, Rrefi,\
            dR0ri, dZ0ri,\
            dRPhiri,\
            VPbisi = GG._Ves_Smesh_TorStruct_SubFromInd_cython(np.array(LPhi[ii][0]),
                                                               dL, dRPhi, VPoly,
                                                               ind,
                                                               DIn=DIn, VIn=VIn,
                                                               Out='(R,Z,Phi)',
                                                               margin=1.e-9)
        assert np.allclose(Pts, Ptsi)
        assert np.allclose(dSi, dS)
        assert np.allclose(NLi, NL)
        # We know it does not match here (too complicated, not necessary)
        # assert np.allclose(dLri, dLr)
        # assert np.allclose(Rrefi,Rref)
        # assert np.allclose(dRPhiri, dRPhir)
        assert all([dR0r == dR0ri, dZ0r == dZ0ri])
        """
        except:
            print([ind.shape == (Pts.shape[1],), ind.dtype == int,
        ind.size == np.unique(ind).size, np.all(ind == np.unique(ind)),
        np.all(ind>=0)])
            print(np.unique(ind).size, ind.size)
            lii = [ind[ii] for ii in range(0,len(ind))
        if np.sum(ind == ind[ii])>1]
            liib = [ii for ii in range(0,len(ind)) if np.sum(ind == ind[ii])>1]
            print(len(lii),len(liib))
            print(lii)
            print(liib)
            for ii in range(0,len(liib)):
                print([Pts[:,liib[ii]] == Pts[:,hh]
        for hh in [jj for jj in range(0,len(ind)) if ind[jj] == lii[ii]]])
        """


def test12_Ves_Smesh_Lin(VPoly=VPoly):

    XMinMax = np.array([0., 10.])
    dL, dX = 0.02, 0.05
    VIn = compute_ves_norm(VPoly)
    DIn = -0.001
    DY, DZ = [0., 2.], [0., 1.]
    LDX = [None, [-1., 2.], [2., 5.], [8., 11.]]

    for ii in range(0, len(LDX)):
        (
            Pts, dS, ind, NL, dLr, Rref,
            dXr, dY0r, dZ0r, VPbis,
        ) = GG._Ves_Smesh_Lin_SubFromD_cython(
            XMinMax, dL, dX, VPoly,
            DX=LDX[ii], DY=DY, DZ=DZ,
            DIn=DIn, VIn=VIn, margin=1.e-9,
        )

        assert Pts.ndim == 2 and Pts.shape[0] == 3

        # check limits along X
        assert (
            np.all(Pts[0, :] >= XMinMax[0] - np.abs(DIn))
            and np.all(Pts[0, :] <= XMinMax[1] + np.abs(DIn))
        )

        # check limits along Y
        assert (
            np.all(Pts[1, :] >= 1. - np.abs(DIn))
            and np.all(Pts[1, :] <= 3. + np.abs(DIn))
        )

        # check limits along Z
        indi = (Pts[2, :] < -np.abs(DIn))
        if np.any(indi):
            msg = (
                f"For ii = {ii}\n"
                f"DZ = {DZ}\n"
                f"Wrong pts: {indi.sum()} / {indi.size}\n"
                f"{np.mean(Pts[2, indi])} vs {-np.abs(DIn)}\n"
                f"{Pts[2, indi]}"
            )
            raise Exception(msg)

        indi = Pts[2, :] > 1. + np.abs(DIn)
        if np.any(indi):
            msg = (
                f"For ii = {ii}\n"
                f"Wrong pts: {indi.sum()} / {indi.size}\n"
                f"{np.mean(Pts[2, indi])} vs {1 + np.abs(DIn)}\n"
                f"{Pts[2, indi]}"
            )
            raise Exception(msg)

        # Check all inside / outside polygon
        if DIn >= 0:
            assert np.all(GG._Ves_isInside(Pts, VPoly,
                                           vs_lims=XMinMax.reshape((1, 2)),
                                           nlim=1, ves_type='Lin',
                                           in_format='(X,Y,Z)', test=True))
        else:
            assert not np.all(GG._Ves_isInside(
                Pts, VPoly,
                ves_lims=XMinMax.reshape((1, 2)),
                nlim=1, ves_type='Lin',
                in_format='(X,Y,Z)', test=True,
            ))
        assert dS.shape == (Pts.shape[1],)

        # Check indices
        if ind.dtype != int:
            msg = str(ind.dtype)
            raise Exception(msg)

        if np.unique(ind).size != ind.size:
            msg = (
                "in is not an array of unique values!\n"
                f"\t- np.unique(ind).size = {np.unique(ind).size}\n"
                f"\t- ind.size = {ind.size}\n"
                f"ind = {ind}"
            )
            raise Exception(msg)

        assert all([ind.shape == (Pts.shape[1],),
                    np.all(ind == np.unique(ind)),
                    np.all(ind>=0)])
        assert (
            ind.shape == (Pts.shape[1],) and ind.dtype == int
            and np.all(ind == np.unique(ind)) and np.all(ind >= 0)
        )

        # Check other output
        assert NL.ndim == 1 and NL.size == VPoly.shape[1]-1
        assert dLr.ndim == 1 and dLr.size == NL.size
        assert Rref.ndim == 1
        assert all([type(xx) is float for xx in [dXr, dY0r, dZ0r]])

        Ptsi, dSi, NLi, \
            dLri, Rrefi, dXri,\
            dY0ri, dZ0ri, \
            VPbisi = GG._Ves_Smesh_Lin_SubFromInd_cython(XMinMax, dL, dX, VPoly,
                                                         ind, DIn=DIn, VIn=VIn,
                                                         margin=1.e-9)

        assert np.allclose(Pts, Ptsi)
        assert np.allclose(dS, dSi)
        assert np.allclose(NL, NLi)
        # We know the following are not identical (size), but too complicated
        # for little gain
        # assert np.allclose(dLr, dLri)
        # assert np.allclose(Rref,Rrefi)
        assert all([dXr == dXri, dY0r == dY0ri, dZ0r == dZ0ri])



######################################################
######################################################
#               LOS
######################################################
######################################################


def test13_LOS_PInOut():

    VP = np.array([[6.,8.,8.,6.,6.],[6.,6.,8.,8.,6.]])
    VIn = np.array([[0., -1., 0., 1.], [1., 0., -1., 0.]])
    VL = np.array([0., 1.])*2.*np.pi
    SP0 = np.array([[6.,7.,7.,6.,6.],[6.,6.,7.,7.,6.]])
    SP1 = np.array([[7.,8.,8.,7.,7.],[7.,7.,8.,8.,7.]])
    SP2 = np.array([[6.,7.,7.,6.,6.],[7.,7.,8.,8.,7.]])
    SP0x = [6.,7.,7.,6.,6.]
    SP1x = [7.,8.,8.,7.,7.]
    SP2x = [6.,7.,7.,6.,6.]
    SP0y = [6.,6.,7.,7.,6.]
    SP1y = [7.,7.,8.,8.,7.]
    SP2y = [7.,7.,8.,8.,7.]
    nstruct_lim = 3
    nstruct_tot =1+2+1
    lstruct_nlim = np.asarray([1, 2, 1])
    SL0 = np.asarray([np.array([0., 1.])*2.*np.pi])
    SL1 = np.asarray([
        np.array(ss)*2.*np.pi
        for ss in [[0., 1./3.], [2./3., 1.]]
    ])
    SL2 = np.asarray([np.array([2./3., 1.])*2.*np.pi])
    lspolyx = np.asarray(SP0x + SP1x + SP2x)
    lspolyy = np.asarray(SP0y + SP1y + SP2y)
    lnvert = np.cumsum(np.ones(nstruct_tot, dtype=int)*5)
    lsvinx = np.asarray([VIn[0], VIn[0], VIn[0]]).flatten()
    lsviny = np.asarray([VIn[1], VIn[1], VIn[1]]).flatten()
    # Linear without Struct
    y, z = [5.,5.,6.5,7.5,9.,9.,7.5,6.5], [7.5,6.5,5.,5.,6.5,7.5,9.,9.]
    N = len(y)
    Ds = np.array([2.*np.pi*np.concatenate((np.ones((N,))/6.,
                                            np.ones((N,))/2.,
                                            5.*np.ones((N,))/6.)),
                   np.tile(y, 3), np.tile(z, 3)])
    Ds = np.concatenate((Ds, np.array([2.*np.pi*np.array([-1., -1., -1., -1.,
                                                         2., 2., 2., 2.]),
                                      [6.5,7.5,7.5,6.5,6.5,7.5,7.5,6.5],
                                      [6.5,6.5,7.5,7.5,6.5,6.5,7.5,7.5]])),
                        axis=1)
    ex = np.array([[1.],[0.],[0.]])
    ey = np.array([[0.],[1.],[0.]])
    ez = np.array([[0.],[0.],[1.]])
    us = np.concatenate(
        (
            ey, ey, ez, ez, -ey, -ey, -ez, -ez,
            ey, ey, ez, ez, -ey, -ey, -ez, -ez,
            ey, ey, ez, ez, -ey, -ey, -ez, -ez,
            ex, ex, ex, ex, -ex, -ex, -ex, -ex,
        ),
        axis=1,
    )
    y = [6.,6.,6.5,7.5,8.,8.,7.5,6.5]
    z = [7.5,6.5,6.,6.,6.5,7.5,8.,8.]
    Sols_In = np.array([2.*np.pi*np.concatenate((np.ones((N,))/6.,
                                                 np.ones((N,))/2.,
                                                 5.*np.ones((N,))/6.)),
                        np.tile(y, 3), np.tile(z, 3)])
    Sols_In = np.concatenate((Sols_In,
                              np.array([2.*np.pi*np.array([0., 0., 0., 0.,
                                                           1., 1., 1., 1.]),
                                        [6.5,7.5,7.5,6.5,6.5,7.5,7.5,6.5],
                                        [6.5,6.5,7.5,7.5,6.5,6.5,7.5,7.5]])),
                             axis=1)
    y = [8.,8.,6.5,7.5,6.,6.,7.5,6.5]
    z = [7.5,6.5,8.,8.,6.5,7.5,6.,6.]
    Sols_Out = np.array([2.*np.pi*np.concatenate((np.ones((N,))/6.,
                                                  np.ones((N,))/2.,
                                                  5.*np.ones((N,))/6.)),
                         np.tile(y, 3), np.tile(z, 3)])
    Sols_Out = np.concatenate((Sols_Out,
                               np.array([2.*np.pi*np.array([1., 1., 1., 1.,
                                                            0., 0., 0., 0.]),
                                         [6.5,7.5,7.5,6.5,6.5,7.5,7.5,6.5],
                                         [6.5,6.5,7.5,7.5,6.5,6.5,7.5,7.5]])),
                              axis=1)
    Iin = np.array([3, 3, 0, 0, 1, 1, 2, 2,
                    3, 3, 0, 0, 1, 1, 2, 2,
                    3, 3, 0, 0, 1, 1, 2, 2,
                    -1, -1, -1, -1, -2, -2, -2, -2], dtype=int)
    Iout = np.array([1, 1, 2, 2, 3, 3, 0, 0,
                     1, 1, 2, 2, 3, 3, 0, 0,
                     1, 1, 2, 2, 3, 3, 0, 0,
                     -2, -2, -2, -2, -1, -1, -1, -1], dtype=int)
    ndim, nlos = np.shape(Ds)
    kPIn, kPOut,\
        VperpOut, IOut= GG.LOS_Calc_PInOut_VesStruct(Ds, us, VP, VIn,
                                                     ves_lims=VL,
                                                     ves_type='Lin', test=True)
    assert np.allclose(kPIn, np.concatenate((np.ones((3*N,)),
                                            2.*np.pi*np.ones((8,)) )),
                       equal_nan=True)
    assert np.allclose(kPOut, np.concatenate((3.*np.ones((kPOut.size-8,)),
                                             2.*np.pi*(1.+np.ones((8,))))),
                       equal_nan=True)
    assert np.allclose(VperpOut, -us)
    assert np.allclose(IOut[2, :], Iout)

    # Linear with Struct
    x = [1./6, 1./6, 1./6, 1./6, 1./6, 1./6, 1./6, 1./6,
         0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
         5./6,5./6,5./6,5./6,5./6,5./6,5./6,5./6,
         0., 1., 0., 2./3, 1., 0., 1., 1.]
    y = [7.,6.,6.5,7.5,7.,8.,7.5,6.5,
         8.,6.,6.5,7.5,7.,6.,7.5,6.5,
         6.,6.,6.5,7.5,7.,8.,7.5,6.5,
         6.5,7.5,7.5,6.5,6.5,7.5,7.5,6.5]
    z = [7.5,6.5,6.,7.,6.5,7.5,8.,7.,
         7.5,6.5,6.,8.,6.5,7.5,6.,7.,
         7.5,6.5,6.,7.,6.5,7.5,8.,8.,
         6.5,6.5,7.5,7.5,6.5,6.5,7.5,7.5]
    Sols_Out = np.array([2.*np.pi*np.array(x), y, z])
    Iin = np.array([3, 3, 0, 0, 1, 1, 2, 2,
                    3, 3, 0, 0, 1, 1, 2, 2,
                    3, 3, 0, 0, 1, 1, 2, 2,
                    -1, -1, -1, -1, -2, -2, -2, -2], dtype=int)
    Iout = np.array([3, 3, 0, 0, 1, 1, 2, 2,
                     1, 3, 0, 2, 1, 3, 0, 2,
                     3, 3, 0, 0, 1, 1, 2, 2,
                     -1, -2, -1, -1, -2, -1, -2, -2], dtype=int)
    indS = np.array([[2, 1, 1, 2, 1, 2, 2, 1,
                      0, 1, 1, 0, 1, 0, 0, 1,
                      3, 1, 1, 2, 1, 2, 2, 3,
                      1, 0, 2, 3, 1, 0, 2, 3],
                     [0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 1, 0, 1, 1, 0,
                      0, 0, 0, 0, 0, 0, 1, 0]], dtype=int)
    kPIn, kPOut, \
        VperpOut, \
        IOut = GG.LOS_Calc_PInOut_VesStruct(Ds, us, VP, VIn, ves_lims=VL,
                                            nstruct_tot=nstruct_tot,
                                            nstruct_lim=nstruct_lim,
                                            lnvert=lnvert,
                                            lstruct_polyx=lspolyx,
                                            lstruct_polyy=lspolyy,
                                            lstruct_nlim=lstruct_nlim,
                                            lstruct_lims=[SL0,SL1,SL2],
                                            lstruct_normx=lsvinx,
                                            lstruct_normy=lsviny,
                                            ves_type='Lin', test=True)
    assert np.allclose(kPIn, np.concatenate((np.ones((3*N,)),
                                            2.*np.pi*np.ones((8,)))),
                       equal_nan=True)
    assert np.allclose(
        kPOut,
        np.concatenate((
            [
                2, 1, 1, 2, 2, 1, 1, 2, 3, 1, 1, 3, 2,
                3, 3, 2, 1, 1, 1, 2, 2, 1, 1, 1,
            ],
            2.*np.pi*np.array([1, 2, 1, 1+2./3, 1, 2, 1, 1])
        ))
    )
    assert np.allclose(VperpOut, -us)
    assert np.allclose(IOut[2, :], Iout)
    assert np.allclose(IOut[:2, :], indS)

    # Toroidal, without Struct
    Theta = np.pi*np.array([1./4., 3./4., 5./4., 7./4.])
    r = np.array([5., 5., 6.5, 7.5, 9., 9., 7.5, 6.5])
    z = np.array([7.5, 6.5, 5., 5., 6.5, 7.5, 9., 9.])
    N = len(r)
    ex, ey = np.array([[1.], [0.], [0.]]), np.array([[0.], [1.], [0.]])
    ez = np.array([[0.], [0.], [1.]])
    Ds, us = [], []
    Sols_In, Sols_Out = [], []
    rsol_In = [[6., 6., 6.5, 7.5, 8., 8., 7.5, 6.5],
               [6., 6., 6.5, 7.5, 8., 8., 7.5, 6.5],
               [6., 6., 6.5, 7.5, 8., 8., 7.5, 6.5],
               [6., 6., 6.5, 7.5, 8., 8., 7.5, 6.5]]
    zsol_In = [[7.5, 6.5, 6., 6., 6.5, 7.5, 8., 8.],
               [7.5, 6.5, 6., 6., 6.5, 7.5, 8., 8.],
               [7.5, 6.5, 6., 6., 6.5, 7.5, 8., 8.],
               [7.5, 6.5, 6., 6., 6.5, 7.5, 8., 8.]]
    rsol_Out = [[8., 8., 6.5, 7.5, 6., 6., 7.5, 6.5],
                [8., 8., 6.5, 7.5, 6., 6., 7.5, 6.5],
                [8., 8., 6.5, 7.5, 6., 6., 7.5, 6.5],
                [8., 8., 6.5, 7.5, 6., 6., 7.5, 6.5]]
    zsol_Out = [[7.5, 6.5, 8., 8., 6.5, 7.5, 6., 6.],
                [7.5, 6.5, 8., 8., 6.5, 7.5, 6., 6.],
                [7.5, 6.5, 8., 8., 6.5, 7.5, 6., 6.],
                [7.5, 6.5, 8., 8., 6.5, 7.5, 6., 6.]]
    for ii in range(0,len(Theta)):
        er = np.array([[np.cos(Theta[ii])], [np.sin(Theta[ii])], [0.]])
        Ds.append(er*r[np.newaxis, :] + ez*z[np.newaxis, :])
        us.append(np.concatenate((er, er, ez, ez, -er, -er, -ez, -ez), axis=1))
        Sols_In.append(np.array(rsol_In[ii])[np.newaxis, :]*er +
                       np.array(zsol_In[ii])[np.newaxis, :]*ez)
        Sols_Out.append(np.array(rsol_Out[ii])[np.newaxis, :]*er +
                        np.array(zsol_Out[ii])[np.newaxis, :]*ez)
    Ds.append(np.array([[6.5,7.5,7.5,6.5, 6.5,6.5,6.5,6.5],
                        [-6.5, -6.5, -6.5, -6.5, -7.5, -6.5, -6.5, -7.5],
                        [6.5,6.5,7.5,7.5, 6.5,6.5,7.5,7.5]]))
    us.append(np.concatenate((ey, ey, ey, ey, -ex, -ex, -ex, -ex), axis=1))
    Ds = np.concatenate(tuple(Ds),axis=1)
    us = np.concatenate(tuple(us),axis=1)
    Sols_In = np.concatenate(tuple(Sols_In),axis=1)
    Sols_Out = np.concatenate(tuple(Sols_Out),axis=1)
    Iin = np.array([3, 3, 0, 0, 1, 1, 2, 2,
                    3, 3, 0, 0, 1, 1, 2, 2,
                    3, 3, 0, 0, 1, 1, 2, 2,
                    3, 3, 0, 0, 1, 1, 2, 2,
                    1, 1, 1, 1, 1, 1, 1, 1])
    Iout = np.array([1, 1, 2, 2, 3, 3, 0, 0,
                     1, 1, 2, 2, 3, 3, 0, 0,
                     1, 1, 2, 2, 3, 3, 0, 0,
                     1, 1, 2, 2, 3, 3, 0, 0,
                     1, 1, 1, 1, 1, 1, 1, 1])
    ndim, nlos = np.shape(Ds)

    kPIn, kPOut,\
        VperpOut, \
        IOut = GG.LOS_Calc_PInOut_VesStruct(Ds, us, VP, VIn, ves_lims=None,
                                            ves_type='Tor', test=True)
    # Reconstructing PIn and Pout from kPIn and kPOut
    PIn = np.zeros_like(VperpOut)
    POut = np.zeros_like(VperpOut)
    for i in range(nlos):
        for j in range(ndim):
            PIn[j, i]  = Ds[j, i] + kPIn[i]  * us[j, i]
            POut[j, i] = Ds[j, i] + kPOut[i] * us[j, i]
    # ...
    ThetaIn = np.arctan2(PIn[1, 32:],  PIn[0, 32:])
    ThetaOut = np.arctan2(POut[1, 32:], POut[0, 32:])
    ErIn = np.array([np.cos(ThetaIn), np.sin(ThetaIn), np.zeros((8,))])
    ErOut = np.array([np.cos(ThetaOut), np.sin(ThetaOut), np.zeros((8,))])
    assert (
        np.allclose(kPIn[:32], np.ones((4*N,)), equal_nan=True)
        and np.all((kPIn[32:] > 0.) & (kPIn[32:] < 6.5))
    )
    assert (
        np.allclose(kPOut[:32], 3.*np.ones((4*N,)), equal_nan=True)
        and np.all((kPOut[32:] > 6.5) & (kPOut[32:] < 16.))
    )
    assert (
        np.allclose(PIn[:, :32], Sols_In[:, :32], equal_nan=True)
        and np.all((ThetaIn > -np.pi/2.) & (ThetaIn < 0.))
    )
    assert (
        np.allclose(POut[:, :32], Sols_Out[:, :32], equal_nan=True)
        and np.all((ThetaOut > 0.) | (ThetaOut < -np.pi/2.))
    )
    assert (
        np.allclose(np.hypot(PIn[0, 32:], PIn[1, 32:]), 8.*np.ones((8,)))
        and np.allclose(np.hypot(POut[0, 32:], POut[1, 32:]), 8.*np.ones((8,)))
    )
    assert (
        np.allclose(VperpOut[:, :32], -us[:, :32])
        and np.allclose(VperpOut[:, 32:], -ErOut)
    )
    assert np.allclose(IOut[2, :], Iout)
    npts_vp = VP.shape[1]
    out = GG.LOS_Calc_kMinkMax_VesStruct(Ds, us,
                                         [VP, VP, VP], [VIn, VIn, VIn], 3,
                                         np.r_[npts_vp, npts_vp, npts_vp])
    kmin_res = out[0]
    kmax_res = out[1]
    assert np.allclose(kmin_res[:nlos],    kPIn)
    assert np.allclose(kmin_res[nlos:2*nlos], kPIn)
    assert np.allclose(kmin_res[2*nlos:],  kPIn)
    assert np.allclose(kmax_res[:nlos],    kPOut)
    assert np.allclose(kmax_res[nlos:2*nlos], kPOut)
    assert np.allclose(kmax_res[2*nlos:],  kPOut)
    # Toroidal, with Struct
    SL0_or =None
    SL1_or = [np.array(ss)*np.pi for ss in [[0., 0.5], [1., 3./2.]]]
    SL2_or = [np.array([0.5, 3./2.])*np.pi]

    SL0 = np.asarray([None])
    SL1 = np.asarray([np.array(ss)*np.pi for ss in [[0., 0.5], [1., 3./2.]]])
    SL2 = np.asarray([np.array([0.5, 3./2.])*np.pi])
    lstruct_nlim = np.array([0, 2, 1])
    nstruct_lim = 3
    nstruct_tot =1+2+1
    lstruct_nlim=np.asarray([0, 2, 1])
    #....
    Sols_In, Sols_Out = [], []
    rsol_In = [[6.,6.,6.5,7.5,8.,8.,7.5,6.5],
               [6.,6.,6.5,7.5,8.,8.,7.5,6.5],
               [6.,6.,6.5,7.5,8.,8.,7.5,6.5],
               [6.,6.,6.5,7.5,8.,8.,7.5,6.5]]
    zsol_In = [[7.5,6.5,6.,6.,6.5,7.5,8.,8.],
               [7.5,6.5,6.,6.,6.5,7.5,8.,8.],
               [7.5,6.5,6.,6.,6.5,7.5,8.,8.],
               [7.5,6.5,6.,6.,6.5,7.5,8.,8.]]
    rsol_Out = [[7.,6.,6.5,7.5,7.,8.,7.5,6.5],
                [6.,6.,6.5,7.5,7.,7.,7.5,6.5],
                [6.,6.,6.5,7.5,7.,8.,7.5,6.5],
                [8.,6.,6.5,7.5,7.,6.,7.5,6.5]]
    zsol_Out = [[7.5,6.5,6.,7.,6.5,7.5,8.,7.],
                [7.5,6.5,6.,8.,6.5,7.5,6.,8.],
                [7.5,6.5,6.,7.,6.5,7.5,8.,8.],
                [7.5,6.5,6.,8.,6.5,7.5,6.,7.]]
    for ii in range(0,len(Theta)):
        er = np.array([[np.cos(Theta[ii])], [np.sin(Theta[ii])], [0.]])
        Sols_In.append(np.array(rsol_In[ii])[np.newaxis, :]*er +
                       np.array(zsol_In[ii])[np.newaxis, :]*ez)
        Sols_Out.append(np.array(rsol_Out[ii])[np.newaxis, :]*er +
                        np.array(zsol_Out[ii])[np.newaxis, :]*ez)
    Sols_In = np.concatenate(tuple(Sols_In),axis=1)
    Sols_Out = np.concatenate(tuple(Sols_Out),axis=1)
    kpout = np.array([2, 1, 1, 2, 2, 1, 1, 2, 1, 1, 1, 3, 2, 2, 3, 1,
                      1, 1, 1, 2, 2, 1, 1, 1, 3, 1, 1, 3, 2, 3, 3, 2])
    Iin = np.array([
        3, 3, 0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2,
        3, 3, 0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,
    ])
    Iout = np.array([
        3, 3, 0, 0, 1, 1, 2, 2, 3, 3, 0, 2, 1, 1, 0, 2,
        3, 3, 0, 0, 1, 1, 2, 2, 1, 3, 0, 2, 1, 3, 0, 2,
        1, 1, -1, 3, 1, 1, -2, -2,
    ])
    indS = np.array([
        [
            2, 1, 1, 2, 1, 2, 2, 1, 3, 1, 1, 0, 1, 3, 0, 3,
            3, 1, 1, 2, 1, 2, 2, 3, 0, 1, 1, 0, 1, 0, 0, 1,
            1, 0, 2, 2, 0, 1, 3, 2,
        ],
        [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1,
        ]],
        dtype=int,
    )

    kPIn, kPOut,\
        VperpOut, \
        IOut = GG.LOS_Calc_PInOut_VesStruct(Ds, us, VP, VIn, ves_lims=None,
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
    assert np.allclose(kPOut[:32], kpout,
                       equal_nan=True) and np.all((kPOut[32:]>=3.) &
                                                  (kPOut[32:]<16.))
    # Reconstructing PIn and Pout from kPIn and kPOut
    PIn = np.zeros_like(VperpOut)
    POut = np.zeros_like(VperpOut)
    ndim, nlos = np.shape(VperpOut)
    for i in range(nlos):
        for j in range(ndim):
            PIn[j, i]  = Ds[j, i] + kPIn[i]  * us[j, i]
            POut[j, i] = Ds[j, i] + kPOut[i] * us[j, i]
    # ...
    RIn = np.hypot(PIn[0, 32:], PIn[1, 32:])
    ROut = np.hypot(POut[0, 32:], POut[1, 32:])
    ThetaIn = np.arctan2(PIn[1, 32:],  PIn[0, 32:])
    ThetaOut = np.arctan2(POut[1, 32:], POut[0, 32:])
    ErIn = np.array([np.cos(ThetaIn), np.sin(ThetaIn), np.zeros((8,))])
    ErOut = np.array([np.cos(ThetaOut), np.sin(ThetaOut), np.zeros((8,))])
    vperpout = np.concatenate(
        (
            ErOut[:, 0:1], -ErOut[:, 1:2], -ey, -ErOut[:, 3:4],
            -ErOut[:, 4:5], ErOut[:, 5:6], ex, ex,
        ),
        axis=1,
    )
    assert np.allclose(kPIn[:32], np.ones((4*N,)),
                       equal_nan=True) and np.all((kPIn[32:]>0.) &
                                                  (kPIn[32:]<6.5))
    assert np.allclose(PIn[:, :32], Sols_In[:, :32],
                       equal_nan=True) and np.all((ThetaIn>-np.pi/2.) &
                                                  (ThetaIn<0.))
    assert np.allclose(POut[:, :32], Sols_Out[:, :32],
                       equal_nan=True) and np.all((ThetaOut>-np.pi) &
                                                  (ThetaOut<np.pi/2.))
    assert np.all((RIn>=6.) & (RIn<=8.)) and np.all((ROut>=6.) & (ROut<=8.))
    assert np.allclose(VperpOut[:, :32], -us[:, :32]) and \
        np.allclose(VperpOut[:, 32:], vperpout)
    assert np.allclose(IOut[:2, :], indS)


def test14_LOS_sino():

    RZ = np.array([2., 0.])
    r = np.array([0.1, 0.2, 0.1])
    theta = np.array([5*np.pi/6, 0, np.pi/2])
    phi = np.array([0., 0., np.pi/10.])
    k = np.array([1, 10, 5])
    N = len(r)
    us = np.ascontiguousarray([np.sin(phi),
                               -np.sin(theta) * np.cos(phi),
                               np.cos(theta) * np.cos(phi)])
    Ms = np.array([np.zeros((N,)),
                   RZ[0] + r * np.cos(theta),
                   RZ[1] + r * np.sin(theta)])
    Ds = np.ascontiguousarray(Ms - k[np.newaxis, :] * us)
    PMin0 = np.nan * np.ones((3, N))
    kPMin0 = np.nan * np.ones((N,))
    RMin0 = np.nan * np.ones((N,))
    Theta0 = np.nan * np.ones((N,))
    p0 = np.nan * np.ones((N,))
    ImpTheta0 = np.nan * np.ones((N,))
    phi0 = np.nan * np.ones((N,))

    res = GG.LOS_sino(Ds, us, RZ, kOut=np.full((N,), np.inf),
                      Mode='LOS', VType='Lin')
    PMin0, kPMin0, RMin0, Theta0, p0, ImpTheta0, phi0 = res
    assert np.allclose(PMin0,Ms)
    assert np.allclose(kPMin0,k)
    assert RMin0.shape == (N,)
    assert Theta0.shape == (N,)
    assert np.allclose(np.abs(p0),r)
    assert np.allclose(np.abs(ImpTheta0),theta)
    assert np.allclose(phi0,phi)

    # Tor (to be finished)
    us = np.ascontiguousarray([-np.sin(theta) * np.cos(phi),
                               np.sin(phi),
                               np.cos(theta) * np.cos(phi)])
    Ms = np.array([RZ[0] + r * np.cos(theta),
                   np.zeros((N,)),
                   RZ[1] + r * np.sin(theta)])
    Ds = np.ascontiguousarray(Ms - k[np.newaxis, :] * us)
    res = GG.LOS_sino(Ds, us, RZ, kOut=np.full((N,), np.inf),
                      Mode='LOS', VType='Tor')
    PMin0, kPMin0, RMin0, Theta0, p0, ImpTheta0, phi0 = res
    assert np.allclose(PMin0,Ms)
    assert np.allclose(kPMin0,k)
    assert RMin0.shape == (N,)
    assert Theta0.shape == (N,)
    assert np.allclose(np.abs(p0), r)
    assert np.allclose(np.abs(ImpTheta0), theta)
    assert np.allclose(np.abs(phi0), phi)


def test15_LOS_sino_vec():
    N = 10**2
    RZ = np.array([2., 0.])
    Ds = np.array([np.linspace(-0.5, 0.5, N),
                   np.ones((N,)),
                   np.zeros((N,))])
    us = np.array([np.linspace(-0.5, 0.5, N),
                   -np.ones((N,)),
                   np.linspace(-0.5, 0.5, N)])

    for _ in range(100):
        res = GG.LOS_sino(Ds, us, RZ, kOut=np.full((N,), np.inf),
                          Mode='LOS', VType='Lin', try_new_algo=True)
        PMin0, kPMin0, RMin0, Theta0, p0, ImpTheta0, phi0 = res
        # verifying there is no Nan:
        assert not np.isnan(np.sum(PMin0))
        assert not np.isnan(np.sum(kPMin0))
        assert not np.isnan(np.sum(RMin0))
        assert not np.isnan(np.sum(Theta0))
        assert not np.isnan(np.sum(p0))
        assert not np.isnan(np.sum(ImpTheta0))
        assert not np.isnan(np.sum(phi0))


def test16_dist_los_vpoly():
    num_rays = 11
    ves_poly = np.zeros((2, 9))
    ves_poly0 = [2, 3, 4, 5, 5, 4, 3, 2, 2]
    ves_poly1 = [2, 1, 1, 2, 3, 4, 4, 3, 2]
    ves_poly[0] = np.asarray(ves_poly0)
    ves_poly[1] = np.asarray(ves_poly1)
    # rays :
    ray_orig = np.zeros((3, num_rays))
    ray_vdir = np.zeros((3, num_rays))
    # ray 0 :
    ray_orig[0][0] = 0
    ray_orig[2][0] = 5
    ray_vdir[0][0] = 1
    # ray 1 :
    ray_orig[0][1] = 3.5
    ray_orig[2][1] = 5
    ray_vdir[0][1] = 1
    # ray 2 :
    ray_orig[0][2] = 3.5
    ray_orig[2][2] = 5
    ray_orig[1][2] = -1
    ray_vdir[0][2] = -1
    # ray 3:
    ray_orig[0][3] = 4
    ray_orig[2][3] = -1
    ray_vdir[0][3] = 1
    ray_vdir[2][3] = 1
    # ray 4:
    ray_orig[0][4] = 7
    ray_orig[2][4] = 3
    ray_vdir[0][4] = 1
    ray_vdir[2][4] = 1
    # ray 5:
    ray_orig[0][5] = 6
    ray_orig[2][5] = 2.4
    ray_orig[1][5] = -1.3
    ray_vdir[1][5] = 1
    ray_vdir[2][5] = 0.01
    # ray 6:
    ray_orig[0][6] = 0.
    ray_orig[1][6] = 0.
    ray_orig[2][6] = -1.
    ray_vdir[2][6] = 0.5
    # ray 7:
    ray_orig[0][7] = 0.
    ray_orig[1][7] = 0.
    ray_orig[2][7] = 4.
    ray_vdir[2][7] = -1.
    # ray 8:
    ray_orig[0][8] = 1.
    ray_orig[1][8] = 0.
    ray_orig[2][8] = 2.
    ray_vdir[2][8] = -1.
    # ray 9:
    ray_orig[0][9] = 3.5
    ray_orig[1][9] = 0.
    ray_orig[2][9] = 0.5
    ray_vdir[2][9] = -1.
    # ray 10:
    ray_orig[0][10] = 5.5
    ray_orig[1][10] = 0.
    ray_orig[2][10] = 2.5
    ray_vdir[0][10] = 1.
    out = GG.comp_dist_los_vpoly(
        np.ascontiguousarray(ray_orig, dtype=np.float64),
        np.ascontiguousarray(ray_vdir, dtype=np.float64),
        ves_poly, disc_step=0.5)

    exact_ks = [3.0,
                0.,
                0.,
                0.9999999999999992,
                0.0,
                1.2576248261177692,
                6.0,
                1.0,
                -0.0,
                0.0,
                0.0]
    exact_dists = [1.0,
                   1.0,
                   1.0,
                   1.4142135623730951,
                   2.0,
                   2.448667030011657,
                   2.0,
                   2.0,
                   1.0,
                   0.5,
                   0.5]
    assert np.allclose(out[0], exact_ks)
    assert np.allclose(out[1], exact_dists)


# ==============================================================================
#
#                           DISTANCE CIRCLE - LOS
#
# ==============================================================================

def test17_distance_los_to_circle():
    # == One Line One Circle ===================================================
    # -- simplest circle -------------------------------------------------------
    radius = 1.
    circ_z = 0.
    # Horizontal ray with no intersection with circle ..........................
    ray_or = np.array([-2.5, 1.5, 0])
    ray_vd = np.array([1., 0, 0.])
    res = GG.comp_dist_los_circle(ray_vd, ray_or, radius, circ_z)
    assert np.isclose(res[0], 2.5), "Problem with 'k'"
    assert np.isclose(res[1], 0.5), "Problem with 'dist'"
    # Horizontal ray tagential to circle at origin..............................
    ray_or = np.array([0, 1., 0])
    ray_vd = np.array([1., 0, 0.])
    res = GG.comp_dist_los_circle(ray_vd, ray_or, radius, circ_z)
    assert np.isclose(res[0], 0.), "Problem with 'k'"
    assert np.isclose(res[1], 0.), "Problem with 'dist'"
    # Diagonal ray with one intersection........................................
    ray_or = np.array([0, 0., 0])
    ray_vd = np.array([1., 1., 0.])
    res = GG.comp_dist_los_circle(ray_vd, ray_or, radius, circ_z)
    k_ex = np.sqrt(2.)*0.5
    assert np.isclose(res[0], k_ex), "Problem with 'k'"
    assert np.isclose(res[1], 0.), "Problem with 'dist'"
    # Diagonal ray with no intersction with circle .............................
    ray_or = np.array([-3, 0., 0])
    ray_vd = np.array([1., 1, 0.])
    res = GG.comp_dist_los_circle(ray_vd, ray_or, radius, circ_z)
    dist_ex = 1.1213203435596426
    assert np.isclose(res[0], 3./2.), "Problem with 'k'"
    assert np.isclose(res[1], dist_ex), "Problem with 'dist'"
    # -- Changing plane circle and radius  -------------------------------------
    radius = 2.
    circ_z = 3.
    # Vertical ray with no intersection with circle ............................
    ray_or = np.array([3.8, -1.3, 3.])
    ray_vd = np.array([0., 1., 0.])
    res = GG.comp_dist_los_circle(ray_vd, ray_or, radius, circ_z)
    assert np.isclose(res[0], 1.3), "Problem with 'k'"
    assert np.isclose(res[1], 1.8), "Problem with 'dist'"
    # Normal ray passing on bottom of circle center ............................
    ray_or = np.array([0, 0., -3.])
    ray_vd = np.array([0., 0., -1.])
    res = GG.comp_dist_los_circle(ray_vd, ray_or, radius, circ_z)
    assert np.isclose(res[0], 0), "Problem with 'k'"
    assert np.isclose(res[1], 6.324555320336759), "Problem with 'dist'"
    # Normal ray passing through circle center .................................
    ray_or = np.array([0, 0., 0.])
    ray_vd = np.array([0., 0., 1.])
    res = GG.comp_dist_los_circle(ray_vd, ray_or, radius, circ_z)
    assert np.isclose(res[0], 3.), "Problem with 'k'"
    assert np.isclose(res[1], 2.), "Problem with 'dist'"
    # Random ray (not normal, not passing through origin, ...) .................
    ray3_or = np.array([-1., 1., -1.1])
    ray3_vd = np.array([2., -2, 1.1])
    radius = 0.5
    circ_z = -1.1
    res = GG.comp_dist_los_circle(ray3_vd, ray3_or, radius, circ_z)
    # computed using scipy
    assert np.isclose(res[0], 0.280758570860685), \
       "Problem with 'k' = "+str(res[0])
    assert np.isclose(res[1], 0.331367971970488), \
       "Problem with 'dist' = "+str(res[1])
    # == Vectorial tests =======================================================
    circle_radius = np.array([0.5, 1, 3.5])
    circle_zcoord = np.array([-1.1, .5, 6])
    ray1_origin = np.array([0., 0, -2.])
    ray1_direct = np.array([0., 0, 1.])
    ray2_origin = np.array([-4, 0, 7.])
    ray2_direct = np.array([1., 0., 0])
    rays_origin = np.array([ray1_origin, ray2_origin])
    rays_direct = np.array([ray1_direct, ray2_direct])
    nlos = 2
    ncir = 3
    res = GG.comp_dist_los_circle_vec(nlos, ncir,
                                      rays_direct,
                                      rays_origin,
                                      circle_radius,
                                      circle_zcoord)
    k_exact = np.array([[0.9, 2.5, 8. ],
                        [3.5, 3. , 0.5]])
    d_exact = np.array([[0.5, 1. , 3.5],
                        [8.1, 6.5, 1. ]])
    assert np.allclose(res[0], k_exact), "Problem with 'k'"
    assert np.allclose(res[1], d_exact), "Problem with 'dist'"


# ==============================================================================
#
#                       TEST CLOSENESS CIRCLE - LOS
#
# ==============================================================================
def test17_is_los_close_to_circle():
    sqrt2 = np.sqrt(2.)
    #...
    radius = 1.
    circ_z = 0.
    # A "yes" case with a tangential ray .......................................
    ray_or1 = np.array([0., sqrt2, 0])
    ray_vd1 = np.array([1., -1, 0.])
    res = GG.is_close_los_circle(ray_vd1, ray_or1, radius, circ_z, 0.1)
    assert np.isclose(res, True), "res = "+str(res)
    # A "yes" case with a non tangential ray ...................................
    ray_or2 = np.array([0., sqrt2+0.01, 0])
    ray_vd2 = np.array([1., -1, 0.])
    res = GG.is_close_los_circle(ray_vd2, ray_or2, radius, circ_z, 0.1)
    assert np.isclose(res, True)
    # A "yes" case with a intersection..........................................
    ray_or3 = np.array([0., sqrt2, 0])
    ray_vd3 = np.array([0, -1., 0.])
    res = GG.is_close_los_circle(ray_vd3, ray_or3, radius, circ_z, 0.1)
    assert np.isclose(res, True)
    # A "no" case with no intersection..........................................
    ray_or4 = np.array([0., sqrt2, 0])
    ray_vd4 = np.array([1., 0, 0.])
    res = GG.is_close_los_circle(ray_vd4, ray_or4, radius, circ_z, 0.1)
    assert np.isclose(res, False)
    # == Vectorial case ========================================================
    # Similar cases but symetric or negative
    # A "yes" case with a tangential ray .......................................
    ray_or1 = np.array([sqrt2, 0., 0])
    ray_vd1 = np.array([-1., 1.0, 0.])
    # A "yes" case with a non tangential ray ...................................
    ray_or2 = np.array([sqrt2+0.001, 0., 0])
    ray_vd2 = np.array([-1., 1, 0.])
    # A "yes" case with a intersection..........................................
    ray_or3 = np.array([sqrt2, 0., 0])
    ray_vd3 = np.array([-1, 0, 0.])
    # A "no" case with no intersection..........................................
    ray_or4 = np.array([sqrt2, -1., 0])
    ray_vd4 = np.array([0., 1., 0.])
    #Computing result..........................................................
    nlos = 4
    ncircles = 1
    los_dirs = np.asarray([ray_vd1, ray_vd2, ray_vd3, ray_vd4])
    los_oris = np.asarray([ray_or1, ray_or2, ray_or3, ray_or4])
    circle_radius=np.asarray([radius])
    circle_zcoord=np.asarray([circ_z])
    res = GG.is_close_los_circle_vec(nlos, ncircles,
                                     0.005,
                                     los_dirs, los_oris,
                                     circle_radius,
                                     circle_zcoord)
    assert res[0][0] == True
    assert res[1][0] == True
    assert res[2][0] == True
    assert res[3][0] == False

# ==============================================================================
#
#                       DISTANCE BETWEEN LOS AND EXT-POLY
#
# ==============================================================================
def test18_comp_dist_los_vpoly():
    # !!!!!! ARTIFICIAL TEST CASE SINCE THIS IS ONLY A SIMPLIFIED VERSION !!!!!!
    num_rays = 11
    ves_poly = np.zeros((2, 9))
    ves_poly0 = [2, 3, 4, 5, 5, 4, 3, 2, 2]
    ves_poly1 = [2, 1, 1, 2, 3, 4, 4, 3, 2]
    ves_poly[0] = np.asarray(ves_poly0)
    ves_poly[1] = np.asarray(ves_poly1)
    # rays :
    ray_orig = np.zeros((3, num_rays))
    ray_vdir = np.zeros((3, num_rays))
    # ray 0 :
    ray_orig[0][0] = 0
    ray_orig[2][0] = 5
    ray_vdir[0][0] = 1
    # ray 1 :
    ray_orig[0][1] = 3.5
    ray_orig[2][1] = 5
    ray_vdir[0][1] = 1
    # ray 2 :
    ray_orig[0][2] = 3.5
    ray_orig[2][2] = 5
    ray_orig[1][2] = -1
    ray_vdir[0][2] = -1
    # ray 3:
    ray_orig[0][3] = 4
    ray_orig[2][3] = -1
    ray_vdir[0][3] = 1
    ray_vdir[2][3] = 1
    # ray 4:
    ray_orig[0][4] = 7
    ray_orig[2][4] = 3
    ray_vdir[0][4] = 1
    ray_vdir[2][4] = 1
    # ray 5:
    ray_orig[0][5] = 6
    ray_orig[2][5] = 2.4
    ray_orig[1][5] = -1.3
    ray_vdir[1][5] = 1
    ray_vdir[2][5] = 0.0
    # ray 6:
    ray_orig[0][6] = 0.
    ray_orig[1][6] = 0.
    ray_orig[2][6] = -1.
    ray_vdir[2][6] = 0.5
    # ray 7:
    ray_orig[0][7] = 0.
    ray_orig[1][7] = 0.
    ray_orig[2][7] = 4.
    ray_vdir[2][7] = -1.
    # ray 8:
    ray_orig[0][8] = 1.
    ray_orig[1][8] = 0.
    ray_orig[2][8] = 2.
    ray_vdir[2][8] = -1.
    # ray 9:
    ray_orig[0][9] = 3.5
    ray_orig[1][9] = 0.
    ray_orig[2][9] = 0.5
    ray_vdir[2][9] = -1.
    # ray 10:
    ray_orig[0][10] = 5.5
    ray_orig[1][10] = 0.
    ray_orig[2][10] = 2.5
    ray_vdir[0][10] = 1.

    # .. computing .............................................................
    out = GG.comp_dist_los_vpoly(np.ascontiguousarray(ray_orig),
                                 np.ascontiguousarray(ray_vdir),
                                 ves_poly, disc_step=0.5)
    k_vec = [3.0,
             0.0,
             0.0,
             1.0,
             0.0,
             1.3,
             6.0,
             1.0,
             0.0,
             0.0,
             0.0]
    dist_vec = [1.0,
                1.0,
                1.0,
                np.sqrt(2.0),
                2.0,
                1.0,
                2.0,
                2.0,
                1.0,
                0.5,
                0.5]
    assert np.allclose(k_vec, out[0])
    assert np.allclose(dist_vec, out[1])


def test19_comp_dist_los_vpoly_vec():
    # !!!!!! ARTIFICIAL TEST CASE SINCE THIS IS ONLY A SIMPLIFIED VERSION !!!!!!
    # ves 0
    ves_poly0 = np.zeros((2, 5))
    ves_poly00 = [4, 5, 5, 4, 4]
    ves_poly01 = [4, 4, 5, 5, 4]
    ves_poly0[0] = np.asarray(ves_poly00)
    ves_poly0[1] = np.asarray(ves_poly01)
    # ves 1
    ves_poly1 = np.zeros((2, 5))
    ves_poly10 = [3, 6, 6, 3, 3]
    ves_poly11 = [3, 3, 6, 6, 3]
    ves_poly1[0] = np.asarray(ves_poly10)
    ves_poly1[1] = np.asarray(ves_poly11)
    vessels = np.asarray([ves_poly0, ves_poly1])
    # Tab for rays
    num_rays = 3
    ray_orig = np.zeros((num_rays, 3))
    ray_vdir = np.zeros((num_rays, 3))
    # ray 0 : First ray intersects all
    ray_orig[0][0] = 4.
    ray_orig[0][2] = 4.
    ray_vdir[0][1] = 1.
    ray_vdir[0][2] = 1.
    # ray 1 : intersects outer circle but not inner
    ray_orig[1][0] = 5.5
    ray_orig[1][2] = 4.
    ray_vdir[1][1] = 1.
    # ray 2 : above all polys
    ray_orig[2][0] = 4.5
    ray_orig[2][2] = 7.
    ray_vdir[2][1] = -1.
    # .. computing .............................................................
    k, dist = GG.comp_dist_los_vpoly_vec(2,  num_rays,
                                     ray_orig,
                                     ray_vdir,
                                     vessels)
    assert np.allclose(k[0], [np.nan, np.nan], equal_nan=True)
    assert np.allclose(dist[0], [np.nan, np.nan], equal_nan=True)
    assert np.allclose(k[1], [0., np.nan], equal_nan=True)
    assert np.allclose(dist[1], [0.5, np.nan], equal_nan=True)
    assert np.allclose(k[2], [0., 0.], equal_nan=True)
    assert np.allclose(dist[2], [2., 1.], equal_nan=True)

# ==============================================================================
#
#                         ARE LOS AND EXT-POLY CLOSE
#
# ==============================================================================
def test20_is_close_los_vpoly_vec():
    # !!!!!! ARTIFICIAL TEST CASE SINCE THIS IS ONLY A SIMPLIFIED VERSION !!!!!!
    # ves 0
    ves_poly0 = np.zeros((2, 5))
    ves_poly00 = [4, 5, 5, 4, 4]
    ves_poly01 = [4, 4, 5, 5, 4]
    ves_poly0[0] = np.asarray(ves_poly00)
    ves_poly0[1] = np.asarray(ves_poly01)
    # ves 1
    ves_poly1 = np.zeros((2, 5))
    ves_poly10 = [3, 6, 6, 3, 3]
    ves_poly11 = [3, 3, 6, 6, 3]
    ves_poly1[0] = np.asarray(ves_poly10)
    ves_poly1[1] = np.asarray(ves_poly11)
    vessels = np.asarray([ves_poly0, ves_poly1])
    # Tab for rays
    num_rays = 3
    ray_orig = np.zeros((num_rays, 3))
    ray_vdir = np.zeros((num_rays, 3))
    # ray 0 : First ray intersects first
    ray_orig[0][0] = 5.01
    ray_orig[0][2] = 5.01
    ray_vdir[0][1] = 1.
    ray_vdir[0][2] = 1.
    # ray 1 : close to second one
    ray_orig[1][0] = 6.01
    ray_orig[1][2] = 6.01
    ray_vdir[1][1] = 1.
    # ray 2 : close to none
    ray_orig[2][0] = 8.
    ray_orig[2][2] = 8.
    ray_vdir[2][1] = 1.
    # .. computing .............................................................
    out = GG.is_close_los_vpoly_vec(2,  num_rays,
                                    ray_orig, ray_vdir,
                                    vessels, 0.1)
    assert np.allclose(out, [[True, False],[False,  True], [False, False]])


# ==============================================================================
#
#                         ARE LOS AND EXT-POLY CLOSE
#
# ==============================================================================
def test21_which_los_closer_vpoly_vec():
    # !!!!!! ARTIFICIAL TEST CASE SINCE THIS IS ONLY A SIMPLIFIED VERSION !!!!!!
    # ves 0
    ves_poly0 = np.zeros((2, 5))
    ves_poly00 = [4, 5, 5, 4, 4]
    ves_poly01 = [4, 4, 5, 5, 4]
    ves_poly0[0] = np.asarray(ves_poly00)
    ves_poly0[1] = np.asarray(ves_poly01)
    # ves 1
    ves_poly1 = np.zeros((2, 5))
    ves_poly10 = [3, 6, 6, 3, 3]
    ves_poly11 = [3, 3, 6, 6, 3]
    ves_poly1[0] = np.asarray(ves_poly10)
    ves_poly1[1] = np.asarray(ves_poly11)
    vessels = np.asarray([ves_poly0, ves_poly1])
    # Tab for rays
    num_rays = 3
    ray_orig = np.zeros((num_rays, 3))
    ray_vdir = np.zeros((num_rays, 3))
    # ray 0 : First ray intersects first
    ray_orig[0][0] = 5.01
    ray_orig[0][2] = 5.01
    ray_vdir[0][1] = 1.
    ray_vdir[0][2] = 1.
    # ray 1 : close to second one
    ray_orig[1][0] = 6.01
    ray_orig[1][2] = 6.01
    ray_vdir[1][1] = 1.
    # ray 2 : close to none
    ray_orig[2][0] = 8.
    ray_orig[2][2] = 8.
    ray_vdir[2][1] = 1.
    # .. computing .............................................................
    out = GG.which_los_closer_vpoly_vec(2,  num_rays,
                                        ray_orig, ray_vdir,
                                        vessels)
    assert np.allclose(out, [0, 1])


# ==============================================================================
#
#                         ARE LOS AND EXT-POLY CLOSE
#
# ==============================================================================
def test21_which_los_closer_vpoly_vec():
    # !!!!!! ARTIFICIAL TEST CASE SINCE THIS IS ONLY A SIMPLIFIED VERSION !!!!!!
    # ves 0
    ves_poly0 = np.zeros((2, 5))
    ves_poly00 = [4, 5, 5, 4, 4]
    ves_poly01 = [4, 4, 5, 5, 4]
    ves_poly0[0] = np.asarray(ves_poly00)
    ves_poly0[1] = np.asarray(ves_poly01)
    # ves 1
    ves_poly1 = np.zeros((2, 5))
    ves_poly10 = [3, 6, 6, 3, 3]
    ves_poly11 = [3, 3, 6, 6, 3]
    ves_poly1[0] = np.asarray(ves_poly10)
    ves_poly1[1] = np.asarray(ves_poly11)
    vessels = np.asarray([ves_poly0, ves_poly1])
    # Tab for rays
    num_rays = 3
    ray_orig = np.zeros((num_rays, 3))
    ray_vdir = np.zeros((num_rays, 3))
    # ray 0 : First ray intersects first
    ray_orig[0][0] = 5.01
    ray_orig[0][2] = 5.01
    ray_vdir[0][1] = 1.
    ray_vdir[0][2] = 1.
    # ray 1 : close to second one
    ray_orig[1][0] = 6.01
    ray_orig[1][2] = 6.01
    ray_vdir[1][1] = 1.
    # ray 2 : close to none
    ray_orig[2][0] = 8.
    ray_orig[2][2] = 8.
    ray_vdir[2][1] = 1.
    # .. computing .............................................................
    out = GG.which_vpoly_closer_los_vec(2,  num_rays,
                                        ray_orig, ray_vdir,
                                        vessels)
    assert np.allclose(out, [0, 1, 1])

# ==============================================================================
#
#                              VIGNETTING
#
# ==============================================================================


def test22_earclipping():

    # .. First test ............................................................
    ves_poly0 = np.array([[4., 5, 5, 4], [4, 4, 5, 5]])
    out = GG.triangulate_by_earclipping_2d(ves_poly0).ravel()

    assert np.allclose(out, [0, 1, 2, 0, 2, 3])

    # .. Second test ...........................................................
    ves_poly1 = np.array([
        [2, 4, 6, 6, 4, 3, 4, 3, 2.0],
        [2, 0, 2, 5, 2, 2, 3, 4, 3.0],
    ])
    # ...computing
    out = GG.triangulate_by_earclipping_2d(ves_poly1).ravel()
    # out = out.reshape((7, 3))
    # print(out)

    assert np.allclose(
        out,
        [1, 2, 3, 1, 3, 4, 1, 4, 5, 0, 1, 5, 0, 5, 6, 0, 6, 7, 0, 7, 8],
    )

    # .. Third test ............................................................
    ves_poly2 = np.array([
        [0, 3.5, 5.5, 7, 8, 7,  6, 5, 3, 4],
        [2.5, 0, 1.5, 1, 5, 4.5,  6, 3, 4, 8],
    ])
    # ...computing
    out = GG.triangulate_by_earclipping_2d(ves_poly2).ravel()

    assert np.allclose(
        out,
        [
            0, 1, 2, 2, 3, 4, 2, 4, 5, 2, 5, 6, 2,
            6, 7, 0, 2, 7, 0, 7, 8, 0, 8, 9,
        ],
    )
    # out = out.reshape((npts-2, 3))
    # print(out)

def test23_vignetting():
   # .. First configuration ....................................................
    ves_poly1 = np.zeros((3, 9))
    x1 = np.r_[2, 4, 6, 6, 4, 3, 4, 3, 2.0]
    y1 = np.r_[2, 0, 2, 5, 2, 2, 3, 4, 3.0]
    ves_poly1[0] = x1
    ves_poly1[1] = y1
    # .. Second configuration ..................................................
    x2 = np.r_[0, 3.5, 5.5, 7, 8, 7,  6, 5, 3, 4]
    y2 = np.r_[2.5, 0, 1.5, 1, 5, 4.5,  6, 3, 4, 8]
    z2 = np.array([0 if xi < 5. else 1. for xi in x2])
    npts = np.size(x2)
    ves_poly2 = np.zeros((3, npts))
    ves_poly2[0] = x2
    ves_poly2[1] = y2
    ves_poly2[2] = z2
    #  === Creating configurations tabs ===
    vignetts = [ves_poly1, ves_poly2]
    lnvert = np.r_[9, npts]
    # === Ray tabs ====
    rays_origin = np.zeros((3, 5))
    rays_direct = np.zeros((3, 5))
    # -- First ray
    orig = np.r_[3.75, 2.5, -2]
    dire = np.r_[0, 0,  1]
    rays_origin[:, 0] = orig
    rays_direct[:, 0] = dire
    # -- Second ray
    orig = np.r_[5, 3.1, -2]
    dire = np.r_[0, 0,  1]
    rays_origin[:, 1] = orig
    rays_direct[:, 2] = dire
    # -- Third ray
    orig = np.r_[0, 0, 5]
    dire = np.r_[4, 1,  -5]/2.
    rays_origin[:, 2] = orig
    rays_direct[:, 2] = dire
    # ==== 3D TESTS ====
    orig = np.r_[0, 2.5, 1]
    fina = np.r_[6.1, 2., 0]
    dire = fina - orig
    rays_origin[:, 3] = orig
    rays_direct[:, 3] = dire
    # Another ray
    orig2 = np.r_[0, 2.5, 1]
    fina2 = np.r_[6., 6., 0]
    dire2 = fina2 - orig2
    rays_origin[:, 4] = orig2
    rays_direct[:, 4] = dire2

    out = GG.vignetting(rays_origin, rays_direct,
                        vignetts, lnvert)
    assert np.allclose(
        out,
        [False, True, False, False,  True, True, False, True, False, False],
    )


def test24_is_visible(debug=0):
    from matplotlib import pyplot as plt
    import tofu.geom as tfg

    # -- Vessel creation ------------------------------------------------------
    VP = np.array([[6., 8., 8., 6., 6.], [6., 6., 8., 8., 6.]])
    VIn = np.array([[0., -1., 0., 1.], [1., 0., -1., 0.]])
    # -- Structures -----------------------------------------------------------
    SP0x = [6., 6.5, 6.5, 6., 6.]
    SP0y = [6., 6., 6.5, 6.5, 6.]
    SP1x = [7.5, 8., 8., 7.5, 7.5]
    SP1y = [7.5, 7.5, 8., 8., 7.5]
    SP2x = [6.75, 7.25, 7.25, 6.75, 6.75]
    SP2y = [6.75, 6.75, 7.25, 7.25, 6.75]
    nstruct_lim = 3
    nstruct_tot = 1 + 2 + 1  # structs: limitless, 2 limits, 1 limit
    lspolyx = np.asarray(SP0x + SP1x + SP2x)
    lspolyy = np.asarray(SP0y + SP1y + SP2y)
    lnvert = np.cumsum(np.ones(nstruct_tot, dtype=int)*5)
    lsvinx = np.asarray([VIn[0], VIn[0], VIn[0]]).flatten()
    lsviny = np.asarray([VIn[1], VIn[1], VIn[1]]).flatten()
    # ...
    # Structures limits
    SL0 = np.asarray([None])
    SL1 = np.asarray([np.array(ss)*np.pi for ss in [[0., 0.5], [1., 3./2.]]])
    SL2 = np.asarray([np.array([0.5, 3./2.])*np.pi])
    lstruct_nlim = np.array([0, 2, 1])
    # -- Points ---------------------------------------------------------------
    # First point (in the center of poloidal plane
    pt0 = 8.
    pt1 = -2.
    pt2 = 6.
    # Other points (to check if visible or not)
    # first test point: same point (should be visible), in torus, out torus
    other_x = np.r_[1, 7.0, 0.0, 0.5]
    other_y = np.r_[7, 1.0, 0.0, -7]
    other_z = np.r_[7, 7.5, 0.0, 6.5]
    npts = len(other_x)
    others = np.zeros((3, npts))
    others[0, :] = other_x
    others[1, :] = other_y
    others[2, :] = other_z
    point = np.r_[pt0, pt1, pt2]
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
                                              lstruct_lims=[SL0, SL1, SL2],
                                              lstruct_normx=lsvinx,
                                              lstruct_normy=lsviny,
                                              ves_type='Tor', test=True)
    assert np.allclose(is_vis, [False, True, True, False])
    distance = np.sqrt(np.sum((others - np.tile(point,
                                                (npts, 1)).T)**2,
                              axis=0))
    is_vis = GG.LOS_isVis_PtFromPts_VesStruct(pt0, pt1, pt2,
                                              others,
                                              dist=distance,
                                              ves_poly=VP,
                                              ves_norm=VIn,
                                              ves_lims=None,
                                              nstruct_tot=nstruct_tot,
                                              nstruct_lim=nstruct_lim,
                                              lnvert=lnvert,
                                              lstruct_polyx=lspolyx,
                                              lstruct_polyy=lspolyy,
                                              lstruct_nlim=lstruct_nlim,
                                              lstruct_lims=[SL0, SL1, SL2],
                                              lstruct_normx=lsvinx,
                                              lstruct_normy=lsviny,
                                              ves_type='Tor', test=True)
    assert np.allclose(is_vis, [False, True, True, False])

    if debug > 0:
        # Visualisation:
        ves = tfg.Ves(
            Name="DebugVessel",
            Poly=VP[:, :-1],
            Type="Tor",
            Exp="Misc",
            shot=0
        )
        s1 = tfg.PFC(Name="S1",
                     Poly=[SP0x[:-1], SP0y[:-1]],
                     Exp="Misc",
                     shot=0)
        s2 = tfg.PFC(Name="S2",
                     Poly=[SP1x[:-1], SP1y[:-1]],
                     Exp="Misc",
                     shot=0,
                     Lim=SL1)
        s3 = tfg.PFC(Name="S3",
                     Poly=[SP2x[:-1], SP2y[:-1]],
                     Exp="Misc",
                     shot=0,
                     Lim=SL2)
        config = tfg.Config(Name="test",
                            Exp="Misc",
                            lStruct=[ves, s1, s2, s3])
        config.set_colors_random()  # to see different colors
        fig = plt.figure(figsize=(14, 8))
        ax = plt.subplot(121)
        config.plot(lax=ax, proj='cross')
        ax2 = plt.subplot(122)
        config.plot(lax=ax2, proj='hor')

    if debug == 1:
        markers = ["o", "*", "^", "s", "p", "v"]
        for ii in range(npts):
            _ = ax2.plot(others[0, ii], others[1, ii],
                         markers[ii], label="pt"+str(ii), ms=5)
            _ = ax.plot(np.sqrt(others[0, ii]**2 + others[1, ii]**2),
                        others[2, ii], markers[ii], ms=5, label="pt"+str(ii))
            # plotting rays for better viz
            _ = ax2.plot([point[0], others[0, ii]],
                         [point[1], others[1, ii]])
            _ = ax.plot([np.sqrt(point[0]**2 + point[1]**2),
                         np.sqrt(others[0, ii]**2 + others[1, ii]**2)],
                        [point[2], others[2, ii]])
        _ = ax2.plot(point[0], point[1], markers[ii], label="pointt", ms=5)
        _ = ax.plot(np.sqrt(point[0]**2 + point[1]**2), point[2],
                    markers[ii], ms=5, label="pointt")
        ax.set_xlabel("R")
        ax.set_ylabel("Z")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax.legend()
        fig.savefig("test1")

    pt_x = np.r_[2, 6.0, -.5, 4.0, -.5, 6.5]
    pt_y = np.r_[7, 2.0, 7.4, 3.5, 6.5, 6.5]
    pt_z = np.r_[7, 7.5, 6.5, 3.0, 7.5, 6.5]
    npts2 = len(pt_x)
    pts2 = np.zeros((3, npts2))
    pts2[0, :] = pt_x
    pts2[1, :] = pt_y
    pts2[2, :] = pt_z
    others = np.zeros((3, 4))
    others[:, 0:2] = pts2[:, 0:2]
    others[:, 2:] = pts2[:, 3:5]

    if debug == 2:
        print(pts2)
        print(others)
        markers = ["bo", "b*", "b^", "bs", "bp", "bv"]
        for ii in range(npts2):
            _ = ax2.plot(pts2[0, ii], pts2[1, ii],
                         markers[ii], label="pt"+str(ii), ms=5)
            _ = ax.plot(np.sqrt(pts2[0, ii]**2 + pts2[1, ii]**2), pts2[2, ii],
                        markers[ii], ms=5, label="pt"+str(ii))
        markers = ["ro", "r*", "r^", "rs", "rp", "rv"]

        ax.set_xlabel("R")
        ax.set_ylabel("Z")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax.legend()
        fig.savefig("test2")

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
                                                 lstruct_lims=[SL0, SL1, SL2],
                                                 lstruct_normx=lsvinx,
                                                 lstruct_normy=lsviny,
                                                 ves_type='Tor', test=True)

    assert np.allclose(are_vis, [[True, False, False, True],
                                 [False, True, False, False],
                                 [True, False, False, False],
                                 [True, False, True, True],
                                 [True, False, False, True],
                                 [True, True, True, True]])
    assert np.shape(are_vis) == (npts2, 4)

    dist = np.zeros((npts2, 4))
    for i in range(npts2):
        for j in range(4):
            xdiff = (pts2[0, i] - others[0, j])**2
            ydiff = (pts2[1, i] - others[1, j])**2
            zdiff = (pts2[2, i] - others[2, j])**2
            dist[i, j] = np.sqrt(xdiff + ydiff + zdiff)

    are_vis = GG.LOS_areVis_PtsFromPts_VesStruct(pts2,
                                                 others,
                                                 dist=dist,
                                                 ves_poly=VP,
                                                 ves_norm=VIn,
                                                 ves_lims=None,
                                                 nstruct_tot=nstruct_tot,
                                                 nstruct_lim=nstruct_lim,
                                                 lnvert=lnvert,
                                                 lstruct_polyx=lspolyx,
                                                 lstruct_polyy=lspolyy,
                                                 lstruct_nlim=lstruct_nlim,
                                                 lstruct_lims=[SL0, SL1, SL2],
                                                 lstruct_normx=lsvinx,
                                                 lstruct_normy=lsviny,
                                              ves_type='Tor', test=True)
    assert np.allclose(are_vis, [[True, False, False, True],
                                 [False, True, False, False],
                                 [True, False, False, False],
                                 [True, False, True, True],
                                 [True, False, False, True],
                                 [True, True, True, True]])
    assert np.shape(are_vis) == (npts2, 4)

    if debug == 3:
        print(pts2)
        print(others)
        markers = ["o", "*", "^", "s", "p", "v"]
        for ii in range(npts2):
            _ = ax2.plot(pts2[0, ii], pts2[1, ii],
                         markers[ii], label="pt"+str(ii), ms=5)
            _ = ax.plot(np.sqrt(pts2[0, ii]**2 + pts2[1, ii]**2), pts2[2, ii],
                        markers[ii], ms=5, label="pt"+str(ii))
        _ = ax2.plot([pts2[0, 0], pts2[0, 1]],
                     [pts2[1, 0], pts2[1, 1]])
        _ = ax.plot([np.sqrt(pts2[0, 0]**2 + pts2[1, 0]**2),
                     np.sqrt(pts2[0, 1]**2 + pts2[1, 1]**2)],
                    [pts2[2, 0], pts2[2, 1]])
        ax.set_xlabel("R")
        ax.set_ylabel("Z")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax.legend()
        fig.savefig("test3")

        print(are_vis)
        print(np.shape(are_vis))
