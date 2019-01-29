
# Built-in
import os

# Common
import numpy as np

# tofu
try:
    import tofu.geom._core as _core
except Exception:
    from . import _core




__all__ = ['coords_transform',
           'get_nIne1e2', 'get_X12fromflat', 'compute_RaysCones',
           'compute_VesPoly',
           'compute_CamLOS1D_pinhole', 'compute_CamLOS1D_pinhole']

_sep = '_'
_dict_lexcept_key = []

_lok = np.arange(0,9)
_lok = np.array([_lok, _lok+10])

_path_testcases = './'


###########################################################
#       COCOS
###########################################################

class CoordinateInputError(Exception):

    _cocosref = "O. Sauter, S. Yu. Medvedev, "
    _cocosref += "Computer Physics Communications 184 (2103) 293-302"

    msg = "The provided coords flag should be a str\n"
    msg += "It should match a known flag:\n"
    msg += "    - 'cart' / 'xyz' : cartesian coordinates\n"
    msg += "    - cocos flag indicating the cocos number (1-8, 11-18)\n"
    msg += "        Valid cocos flags include:\n"
    msg += "            '11', '02', '5', '14', ..."
    msg += "\n"
    msg += "The cocos (COordinates COnvetionS) are descibed in:\n"
    msg += "    [1] %s"%_cocosref

    def __init__(self, msg, errors):

        # Call the base class constructor with the parameters it
        # needs
        super(CoordinateInputError, self).__init__(msg + '\n\n' + self.msg)

        # Now for your custom code...
        self.errors = errors



def _coords_checkformatcoords(coords='11'):
    if not type(coords) is str:
        msg = "Arg coords must be a str !"
        raise CoordinateInputError(msg)
    coords = coords.lower()

    iint = np.array([ss.isdigit() for ss in coords]).nonzero()[0]
    if coords in ['cart','xyz']:
        coords = 'xyz'
    elif iint.size in [1,2]:
        coords = int(''.join([coords[jj] for jj in iint]))
        if not coords in _lok.ravel():
            msg = 'Not allowed number ({0) !'.format(coords)
            raise CoordinateInputError(msg)
    else:
        msg = "Not allowed coords ({0}) !".format(coords)
        raise CoordinateInputError(msg)
    return coords


def _coords_cocos2cart(pts, coords=11):
    R = pts[0,:]
    if (coords%0)%2==1:
        indphi, indZi, sig = 1, 2, 1.
    else:
        indphi, indZ , sig= 2, 1, -1.
    phi = sig*pts[indphi,:]

    X = R*np.cos(phi)
    Y = R*np.sin(phi)
    Z = pts[indZ,:]
    return np.array([X,Y,Z])

def _coords_cart2cocos(pts, coords=11):
    R = np.hypot(pts[0,:],pts[1,:])
    phi = np.arctan2(pts[1,:],pts[0,:])
    Z = pts[2,:]
    if (coords%0)%2==1:
        indphi, indZ, sig = 1, 2, 1.
    else:
        indphi, indZ , sig= 2, 1, -1.
    pts_out = np.empty((3,pts.shape[1]),dtype=float)
    pts_out[0,:] = R
    pts_out[indphi,:] = sig*phi
    pts_out[indZ,:] = Z
    return pts_out

def coords_transform(pts, coords_in='11', coords_out='11'):

    coords_in = _coords_checkformatcoords(coords=coords_in)
    coords_out = _coords_checkformatcoords(coords=coords_out)

    if coords_in==coords_out:
        pass
    elif coords_in=='xyz':
        pts = _coords_cart2cocos(pts, coords_out)
    elif coords_out=='xyz':
        pts = _coords_cocos2cart(pts, coords_out)
    else:
        pts = _coords_cocos2cart(pts, coords_in)
        pts = _coords_cocos2cart(pts, coords_out)
    return pts



###########################################################
###########################################################
#       Useful functions
###########################################################

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


def compute_RaysCones(Ds, us, angs=np.pi/90., nP=40):
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
###########################################################
#       Fast computation of basic geometry (poly and LOS)
###########################################################


def compute_VesPoly(R=2.4, r=1., elong=0., Dshape=0.,
                    divlow=True, divup=True, nP=200):
    """ Utility to compute three 2D (R,Z) polygons

    One represents a vacuum vessel, one an outer bumper, one a baffle

    The vessel polygon is centered on (R,0.), with minor radius r
    It can have a vertical (>0) or horizontal(<0) elongation in [-1;1]
    It can be D-shaped (Dshape in [0.,1.], typically 0.2)
    It can be non-convex, with:
        * a lower divertor-like shape
        * a upper divertor-like shape
    The elongation also affects the outer bumper and baffle

    Parameters
    ----------
    R:          int / float
        Major radius used as a center of the vessel
    r :         int / float
        Minor radius of the vessel
    elong:      int / float
        Dimensionless elongation parameter in [-1;1]
    Dshape:     int / float
        Dimensionless parameter for the D-shape (in-out asymmetry) in [0;1]
    divlow:     bool
        Flag indicating whether to incude a lower divertor-like shape
    divup:      bool
        Flag indicating whether to incude an upper divertor-like shape
    nP :        int
        Parameter specifying approximately the number of points of the vessel

    Return
    ------
    poly:       np.ndarray
        Closed (2,nP) polygon of the vacuum vessel, optionnally with divertors
    pbump:      np.ndarray
        Closed (2,N) polygon defining the outer bumper
    pbaffle:    np.ndarray
        Closed (2,N) polygon defining the lower baffle
    """

    # Basics (center, theta, unit vectors)
    cent = np.r_[R,0.]
    theta = np.linspace(-np.pi,np.pi,nP)
    poly = np.array([np.cos(theta), np.sin(theta)])

    # Divertors
    pdivR = np.r_[-0.1,0.,0.1]
    pdivZ = np.r_[-0.1,0.,-0.1]
    if divlow:
        ind = (np.sin(theta)<-0.85).nonzero()[0]
        pinsert = np.array([pdivR, -1.+pdivZ])
        poly = np.concatenate((poly[:,:ind[0]], pinsert, poly[:,ind[-1]+1:]),
                              axis=1)

    if divup:
        theta = np.arctan2(poly[1,:], poly[0,:])
        ind = (np.sin(theta)>0.85).nonzero()[0]
        pinsert = np.array([pdivR[::-1], 1.-pdivZ])
        poly = np.concatenate((poly[:,:ind[0]], pinsert, poly[:,ind[-1]+1:]),
                              axis=1)

    # Modified radius (by elongation and Dshape)
    rbis = r*np.hypot(poly[0,:],poly[1,:])
    theta = np.arctan2(poly[1,:],poly[0,:])
    rbis = rbis*(1+elong*0.15*np.sin(2.*theta-np.pi/2.))
    if Dshape>0.:
        ind = np.cos(theta)<0.
        coef = 1 + Dshape*(np.sin(theta[ind])**2-1.)
        rbis[ind] = rbis[ind]*coef

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
    pbump = (pbRin, pbRout[:,:ind[0]], pinsert,
             pbRout[:,ind[-1]+1:], pbRin[:,0:1])
    pbump = np.concatenate(pbump, axis=1)
    theta = np.arctan2(pbump[1,:],pbump[0,:])
    er = np.array([np.cos(theta), np.sin(theta)])
    rbis = r*(np.hypot(pbump[0,:],pbump[1,:])
              *(1.+elong*0.15*np.sin(2.*theta-np.pi/2.)))
    pbump = cent[:,np.newaxis] + rbis[np.newaxis,:]*er

    # Baffle
    offR, offZ = 0.1, -0.85
    wR, wZ = 0.2, 0.05
    pbaffle = np.array([offR + wR*np.r_[-1,1,1,-1,-1],
                        offZ + wZ*np.r_[1,1,-1,-1,1]])
    theta = np.arctan2(pbaffle[1,:],pbaffle[0,:])
    er = np.array([np.cos(theta), np.sin(theta)])
    rbis = r*(np.hypot(pbaffle[0,:],pbaffle[1,:])
              *(1.+elong*0.15*np.sin(2.*theta-np.pi/2.)))
    pbaffle = cent[:,np.newaxis] + rbis[np.newaxis,:]*er

    return poly, pbump, pbaffle



def _compute_PinholeCam_checkformatinputs(P=None, F=0.1, D12=None, N12=100,
                                          angs=0, nIn=None, VType='Tor', defRY=None, Lim=None):
    assert type(VType) is str
    VType = VType.lower()
    assert VType in ['tor','lin']
    if np.sum([angs is None, nIn is None])!=1:
        msg = "Either angs xor nIn should be provided !"
        raise Exception(msg)

    # Pinhole
    if P is None:
        if defRY is None:
            msg = "If P is not provided, a value msut be set for defRY!"
            raise Exception(msg)
        if VType=='tor':
            P = np.array([defRY,0.,0.])
        else:
            if Lim is None:
                msg = "If P is not provided, Lim must be set!"
                raise Exception(msg)
            Lim = np.array(Lim).ravel()
            assert Lim.size==2 and Lim[0]<Lim[1]
            P = np.array([np.sum(Lim)/2., defRY, 0.])

    # Camera inner parameters
    assert type(F) in [int, float, np.int64, np.float64]
    F = float(F)

    if D12 is None:
        D12 = F
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

    # Angles
    if angs is None:
        assert hasattr(nIn,'__iter__')
        nIn = np.asarray(nIn, dtype=float).ravel()
        assert nIn.size==3

    else:
        if type(angs) in [int, float, np.int64, np.float64]:
            angs = np.array([angs,angs,angs],dtype=float)
        angs = np.asarray(angs).astype(float).ravel()
        assert angs.size==3
        angs = np.arctan2(np.sin(angs),np.cos(angs))

    if VType=='tor':
        R = np.hypot(P[0],P[1])
        phi = np.arctan2(P[1],P[0])
        eR = np.array([np.cos(phi), np.sin(phi), 0.])
        ePhi = np.array([-np.sin(phi), np.cos(phi), 0.])
        eZ = np.array([0.,0.,1.])

        if nIn is None:
            nIncross = eR*np.cos(angs[0]) + eZ*np.sin(angs[0])
            nIn = nIncross*np.cos(angs[1]) + ePhi*np.sin(angs[1])
            nIn = nIn/np.linalg.norm(nIn)

        if np.abs(np.abs(nIn[2])-1.)<1.e-12:
            e10 = ePhi
        else:
            e10 = np.cross(nIn,eZ)
            e10 = e10/np.linalg.norm(e10)

    else:
        X = P[0]
        eX, eY, eZ = np.r_[1.,0.,0.], np.r_[0.,1.,0.], np.r_[0.,0.,1.]
        if nIn is None:
            nIncross = eY*np.cos(angs[0]) + eY*np.sin(angs[0])
            nIn = nIncross*np.cos(angs[1]) + eZ*np.sin(angs[1])
            nIn = nIn/np.linalg.norm(nIn)

        if np.abs(np.abs(nIn[2])-1.)<1.e-12:
            e10 = eX
        else:
            e10 = np.cross(nIn,eZ)
            e10 = e10/np.linalg.norm(e10)

    e20 = np.cross(e10,nIn)
    e20 = e20/np.linalg.norm(e20)
    if e20[2]<0.:
        e10, e20 = -e10, -e20

    if ansg is None:
        e1 = e10
        e2 = e20
    else:
        e1 = np.cos(angs[2])*e10 + np.sin(angs[2])*e20
        e2 = -np.sin(angs[2])*e10 + np.cos(angs[2])*e20

    # Check consistency of vector base
    assert all([np.abs(np.linalg.norm(ee)-1.)<1.e-12 for ee in [e1,nIn,e2]])
    assert np.abs(np.sum(nIn*e1))<1.e-12
    assert np.abs(np.sum(nIn*e2))<1.e-12
    assert np.abs(np.sum(e1*e2))<1.e-12
    assert np.linalg.norm(np.cross(e1,nIn)-e2)<1.e-12

    return P, F, D12, N12, angs, nIn, e1, e2, VType



_comdoc = \
        """ Generate LOS for a {0}D camera

        Generate the tofu inputs to instanciate a {0}D LOS pinhole camera

        Internally, the camera is defined with:
            - P : (X,Y,Z) position of the pinhole
            - F : focal length (distance pinhole-detector plane)
            - (e1,nIn,e2): a right-handed normalized vector base
                nIn: the vector pointing inwards, from the detector plane to
                    the pinhole and plasma
                (e1,e2):  the 2 vector defining the detector plane coordinates
                    By default, e1 is horizontal and e2 points upwards
            - D12: The size of the detector plane in each direction (e1 and e2)
            - N12: The number of detectors (LOS) in each direction (e1 and e2)

        To simplify parameterization, the vector base (e1,nIn,e2) is
        automatically computed from a set of 3 angles contained in angs

        Parameters
        ----------
        P:      np.ndarray
            (3,) array containing the pinhole (X,Y,Z) cartesian coordinates
        F:      float
            The focal length
        D12:    float {1}
            The absolute size of the detector plane
            {2}
        N12:    int {3}
            The number of detectors (LOS) on the detector plane
            {4}
        angs:   list
            The three angles defining the orientation of the camera vector base
                - angs[0] : 'vertical' angle, the angle between the projection of
                    nIn in a cross-section and the equatorial plane
                - angs[1] : 'longitudinal' angle, the angle between nIn and a
                    cross-section plane
                - angs[2] : 'twist' angle, the angle between e1 and the equatorial
                    plane, this is the angle the camera is rotated around its own
                    axis
        VType:  str
            Flag indicating whether the geometry type is:
                - 'Lin': linear
                - 'Tor': toroidal
        defRY:  None / float
            Only used if P not provided
            The default R (if 'Tor') or 'Y' (of 'Lin') position at which to
            place P, in the equatorial plane.
        Lim:    None / list / np.ndarray
            Only used if P is None and VTYpe is 'Lin'
            The vessel limits, by default P will be place in the middle

        Return
        ------
        Ds:     np.ndarray
            (3,N) array of the LOS starting points cartesian (X,Y,Z) coordinates
            Can be fed to tofu.geom.CamLOSCam{0}D
        P:      np.ndarray
            (3,) array of pinhole (X,Y,Z) coordinates
            Can be fed to tofu.geom.CamLOS{0}D
        {5}
        d2:     np.ndarray
            (N2,) coordinates array of the LOS starting point along local
            vector e2 (0 being the perpendicular to the pinhole on the detector plane)

        """





def compute_CamLOS1D_pinhole(P=None, F=0.1, D12=0.1, N12=100,
                             angs=[-np.pi,0.,0.], nIn=None,
                             VType='Tor', defRY=None, Lim=None):

    # Check/ format inputs
    P, F, D12, N12, angs, nIn, e1, e2, VType\
            = _compute_PinholeCam_checkformatinputs(P=P, F=F, D12=D12, N12=N12,
                                                    angs=angs, nIn=nIn,
                                                    VType=VType, defRY=defRY,
                                                    Lim=Lim)

    # Get starting points
    d2 = 0.5*D12[1]*np.linspace(-1.,1.,N12[1],endpoint=True)
    d2e = d2[np.newaxis,:]*e2[:,np.newaxis]

    Ds = P[:,np.newaxis] - F*nIn[:,np.newaxis] + d2e
    return Ds, P, d2


_comdoc1 = _comdoc.format('1','',
                          'Extension of the detector alignment along e2',
                          '',
                          'Number of detectors along e2',
                          '')

compute_CamLOS1D_pinhole.__doc__ = _comdoc1


def compute_CamLOS2D_pinhole(P=None, F=0.1, D12=0.1, N12=100,
                             angs=[-np.pi,0.,0.], nIn=None,
                             VType='Tor', defRY=None, Lim=None):

    # Check/ format inputs
    P, F, D12, N12, angs, nIn, e1, e2, VType\
            = _compute_PinholeCam_checkformatinputs(P=P, F=F, D12=D12, N12=N12,
                                                    angs=angs, nIn=nIn,
                                                    VType=VType, defRY=defRY,
                                                    Lim=Lim)

    # Get starting points
    d1 = 0.5*D12[0]*np.linspace(-1.,1.,N12[0],endpoint=True)
    d2 = 0.5*D12[1]*np.linspace(-1.,1.,N12[1],endpoint=True)
    d1f = np.repeat(d1,N12[1])
    d2f = np.tile(d2,N12[0])
    d1e = d1f[np.newaxis,:]*e1[:,np.newaxis]
    d2e = d2f[np.newaxis,:]*e2[:,np.newaxis]


    Ds = P[:,np.newaxis] - F*nIn[:,np.newaxis] + d1e + d2e
    return Ds, P, d1, d2



_extracom2 = '(N1,) coordinates array of the LOS starting point along local\n\
\t    vector e1 (0 being the perpendicular to the pinhole on the detector plane)'
_comdoc2 = _comdoc.format('2','/ list',
                          'Extended to [D12,D12] if a float is provided',
                          '/ list',
                          'Extended to [D12,D12] if a float is provided',
                          'd1:    np.ndarray\n\t    '+_extracom2)

compute_CamLOS2D_pinhole.__doc__ = _comdoc2


###########################################################
#       Fast creation of basic objects
###########################################################

def create_config(Exp='Dummy', Type='Tor', Lim=None, Lim_Bump=[0.,np.pi/8.],
                  R=2.4, r=1., elong=0., Dshape=0.,
                  divlow=True, divup=True, nP=200):

    poly, pbump, pbaffle = compute_VesPoly(R=R, r=r, elong=elong, Dshape=Dshape,
                                           divlow=divlow, divup=divup, nP=nP)

    ves = _core.Ves(Poly=poly, Type=Type, Lim=Lim, Exp=Exp, Name='Ves')
    baf = _core.PFC(Poly=pbaffle, Type=Type, Lim=Lim,
                    Exp=Exp, Name='Baffle', color='b')
    bump = _core.PFC(Poly=pbump, Type=Type, Lim=Lim_Bump,
                     Exp=Exp, Name='Bumper', color='g')

    conf = _core.Config(Name='Conf', lStruct=[ves,baf,bump])
    return conf


_dconfig = {'A1': {'Exp':'WEST',
                   'Ves': ['V1']},
            'A2': {'Exp':'ITER',
                   'Ves': ['V0']},
            'A3': {'Exp':'WEST',
                   'PlasmaDomain': ['Sep']},
            'B1': {'Exp':'WEST',
                   'Ves': ['V2'],
                   'PFC': ['BaffleV0', 'DivUpV1', 'DivLowITERV1']},
            'B2': {'Exp':'WEST',
                   'Ves': ['V2'],
                   'PFC': ['BaffleV1', 'DivUpV2', 'DivLowITERV2',
                           'BumperInnerV1', 'BumperOuterV1',
                           'IC1V1', 'IC2V1', 'IC3V1']},
            'B3': {'Exp':'WEST',
                   'Ves': ['V2'],
                   'PFC': ['BaffleV2', 'DivUpV3', 'DivLowITERV3',
                           'BumperInnerV3', 'BumperOuterV3',
                           'IC1V1', 'IC2V1', 'IC3V1',
                           'LH1V1', 'LH2V1',
                           'RippleV1', 'VDEV0']}}

def create_config_testcase(config='A1',
                           path=_path_testcases, dconfig=_dconfig):
    """ Load the desired test case configuration

    Choose from one of the reference preset configurations:
        {0}

    """.format('['+', '.join(dconfig.keys())+']')
    assert all([type(ss) is str for ss in [config,path]])
    assert type(dconfig) is dict
    if not config in dconfig.keys():
        msg = "Please a valid config, from one of the following:\n"
        msg += "["+", ".join(dconfig.keys())+"+]"
        raise Exception(msg)
    path = os.path.abspath(path)

    # Get file names for config
    lf = os.listdir(path)
    lS = []
    for cc in dconfig[config].keys():
        if cc=='Exp':
            continue
        for ss in dconfig[config][cc].keys():
            for vv in dconfig[config][cc][ss]:
                ff = [f for f in lf
                      if all([s in f for s in ['TFG_',cc,ss,vv,'.txt']])]
                if not len(ff)==1:
                    msg = "No / several matching files in %s:\n"%path
                    msg += "Criteria: [%s, %s, %s]\n"%(cc,ss,vv)
                    msg += "Matching: "+"\n          ".join(ff)
                    raise Exception(msg)
                out = np.loadtxt(os.path.join(path,ff[0]))
                npts, nunits = out[0,:]
                poly = out[1:1+npts,:].T
                if n>=1:
                    pass
                    #lim = out[0,:].T
                oo = _core.cc(Name=ss+vv, Poly=poly, Lim=lim, Limtype='pos',
                              Exp=dconfig[config]['Exp'])
                lS.append(oo)

    conf = _core.Config(Name=config, lStruct=lS)

    # ------------------
    # Optionnal plotting
    if plot:
        lax = conf.plot(element='P')

    return conf
