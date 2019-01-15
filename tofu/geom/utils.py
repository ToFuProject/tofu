
# Built-in
import os

# Common
import numpy as np

_sep = '_'
_dict_lexcept_key = []



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


###########################################################
#       Fast creation of basic objects
###########################################################


def create_VesPoly(R=2.4, r=1., elong=0., Dshape=0.,
                   divlow=False, divup=False, nP=200):
    """ Utility to create a 2D (R,Z) polygon to be used a Ves

    The polygon is centered on (R,0.)
    It has a minor radius of r
    It can have a vertical (>0) or horizontal(<0) elongation
    It can be D-shaped (Dshape in [0.,1.])
    It can be non-convex, with:
        * a lower divertor-like shape
        * a upper divertor-like shape
        * an outer bumper-like shape
    """

    # Basics (center, theta, unit vectors)
    cent = np.r_[R,0.]
    theta = np.linspace(-np.pi,np.pi,nP)
    poly = np.array([np.cos(theta), np.sin(theta)])

    # Divertors
    pdivR = np.r_[-0.1,0.,0.1]
    pdivZ = np.r_[-0.1,0.,-0.1]
    if divlow:
        ind = (np.sin(theta)<-0.8).nonzero()[0]
        pinsert = np.array([pdivR, -1.+pdivZ])
        poly = np.concatenate((poly[:,:ind[0]], pinsert, poly[:,ind[-1]+1:]),
                              axis=1)

    if divup:
        theta = np.arctan2(poly[1,:], poly[0,:])
        ind = (np.sin(theta)>0.8).nonzero()[0]
        pinsert = np.array([pdivR[::-1], 1.-pdivZ])
        poly = np.concatenate((poly[:,:ind[0]], pinsert, poly[:,ind[-1]+1:]),
                              axis=1)

    # Modified radius (by elongation and Dshape)
    rbis = r*np.hypot(poly[0,:],poly[1,:])
    theta = np.arctan2(poly[1,:],poly[0,:])
    rbis = rbis*(1+elong*0.15*np.sin(2.*theta-np.pi/2.))
    if Dshape>0.:
        ind = np.cos(theta)<0.
        coef = np.abs(np.sin(theta[ind]))
        coef = coef + Dshape*(1-coefs)
        rbis[ind] = rbis[ind]*coefs

    er = np.array([np.cos(theta), np.sin(theta)])
    poly = cent[:,np.newaxis] + rbis[np.newaxis,:]*er

    # Outer bumper
    Dbeta = 2.*np.pi/6.
    beta = np.linspace(-Dbeta/2.,Dbeta/2., 20)
    pbRin = 0.85*np.array([np.cos(beta), np.sin(beta)])
    pbRout = 0.95*np.array([np.cos(beta), np.sin(beta)])[:,::-1]
    pinsert = np.array([[0.95,1.05,1.05,0.95],
                        [0.05,0.05,-0.05,-0.05]])

    ind = (np.abs(pbRout[1,:])<0.05).nonzero()[0]
    pbump = (pbRin, pbRout[:,:ind[0]], pinsert, pbRout[:,ind[-1]+1:])
    pbump = np.concatenate(pbump, axis=1)
    theta = np.arctan2(pbump[1,:],pbump[0,:])
    er = np.array([np.cos(theta), np.sin(theta)])
    rbis = r*(np.hypot(pbump[0,:],pbump[1,:])
              *(1.+elong*0.15*np.sin(2.*theta-np.pi/2.)))
    pbump = cent[:,np.newaxis] + rbis[np.newaxis,:]*er

    # Baffle
    offR, offZ = 0.1, -0.85
    wR, wZ = 0.2, 0.05
    pbaffle = np.array([offR + wR*np.r_[-1,1,1,-1],
                        offZ + wZ*np.r_[1,1,-1,-1]])
    theta = np.arctan2(pbaffle[1,:],pbaffle[0,:])
    er = np.array([np.cos(theta), np.sin(theta)])
    rbis = r*(np.hypot(pbaffle[0,:],pbaffle[1,:])
              *(1.+elong*0.15*np.sin(2.*theta-np.pi/2.)))
    pbaffle = cent[:,np.newaxis] + rbis[np.newaxis,:]*er

    return poly, pbump, pbaffle




# Create basics for pinhole camera
def _create_PinHoleCam_Basics(P, F, D12, N12,
                              nIn=None, e1=None, e2=None,
                              VType='tor'):

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
        e2 = np.cross(e1,nIn)
    e2 = np.asarray(e2)
    e2 = e2/np.linalg.norm(e2)
    assert np.abs(np.sum(nIn*e2))<1.e-12
    assert np.abs(np.sum(e1*e2))<1.e-12

    return P, F, nIn, e1, e2




#def create_CamLOS1D_pinholeDu(R=, Z=, Phi=, nch=100)






def create_CamLOS2D_pinholeDu(P, F, D12, N12,
                              nIn=None, e1=None, e2=None,
                              VType='Tor'):

    # Check/ format inputs
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

    P, F, nIn, e1, e2 = _create_PinHoleCam_Basics(P, F, nIn=nIn, e1=e1, e2=e2)

    # Get starting points
    d1 = D12[0]*np.linspace(-0.5,0.5,N12[0],endpoint=True)
    d2 = D12[1]*np.linspace(-0.5,0.5,N12[1],endpoint=True)
    d1 = np.repeat(d1,N12[1])
    d2 = np.tile(d2,N12[0])
    d1 = d1[np.newaxis,:]*e1[:,np.newaxis]
    d2 = d2[np.newaxis,:]*e2[:,np.newaxis]

    Ds = P[:,np.newaxis] - F*nIn[:,np.newaxis] + d1 + d2
    us = P[:,np.newaxis] - Ds
    return Ds, us
