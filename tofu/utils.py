
import numpy as np

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
        N12 = np.array([N12,N12],dtype=float)
    else:
        assert hasattr(N12,'__iter__') and len(N12)==2
        N12 = np.asarray(N12).astype(float)
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
    if e1 is None:
       if VType=='tor':
            phi = np.arctan2(P[1],P[0])
            ephi = np.r_[-np.sin(phi),np.cos(phi),0.]
            if np.abs(np.abs(nIn[2])-1.)<1.e-12:
                e1 = ephi
            else:
                e1 = np.cross(nIn,np.r_[0.,0.,1.])
                e1 = e1 if np.sum(e1,ephi)>0. else -e1
       else:
            if np.abs(np.abs(nIn[0])-1.)<1.e-12:
                e1 = np.r_[0.,1.,0.]
            else:
                e1 = np.cross(nIn,np.r_[0.,0.,1.])
                e1 = e1 if e1[0]>0. else -e1
    assert np.abs(np.sum(nIn*e1))<1.e-12
    if e2 is None:
        e2 = np.cross(nIn,e1)
    assert np.abs(np.sum(nIn*e2))<1.e-12
    assert np.abs(np.sum(e1*e2))<1.e-12

    for vv in [nIn,e1,e2]:
        vv = vv/np.linalg.norm(vv)


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
