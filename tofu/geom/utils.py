
# Standard
import sys
import os
import warnings
import inspect

# Third-party
import numpy as np

# Local
from . import _core
from . import _def_config


__all__ = ['coords_transform',
           'get_nIne1e2', 'get_X12fromflat',
           'compute_RaysCones',
           'get_available_config', 'create_config',
           'create_CamLOS1D', 'create_CamLOS2D']


_sep = '_'
_dict_lexcept_key = []

_lok = np.arange(0,9)
_lok = np.array([_lok, _lok+10])

_path_testcases = os.path.join(os.path.dirname(__file__), 'inputs')

_URL_TUTO = ('https://tofuproject.github.io/tofu/auto_examples/tutorials/'
             + 'tuto_plot_create_geometry.html')


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
        if np.abs(np.abs(nIn[2])-1.) < 1.e-12:
            e1 = ephi
        else:
            e1 = np.cross(nIn,ez)
        e1 = e1 if np.sum(e1*ephi) > 0. else -e1
    e1 = e1 / np.linalg.norm(e1)

    if not np.abs(np.sum(nIn*e1))<1.e-12:
        msg = "Identified local base does not seem valid!\n"
        msg += "nIn = %s\n"%str(nIn)
        msg += "e1 =  %s\n"%str(e1)
        msg += "np.sum(nIn*e1) = sum(%s) = %s"%(nIn*e1, np.sum(nIn*e1))
        raise Exception(msg)

    if e2 is None:
        e2 = np.cross(nIn,e1)
    e2 = e2 / np.linalg.norm(e2)
    return nIn, e1, e2


def get_X12fromflat(X12, x12u=None, nx12=None):
    if x12u is None:
        x1u, x2u = np.unique(X12[0,:]), np.unique(X12[1,:])
        if x1u.size*x2u.size != X12.shape[1]:
            tol = np.linalg.norm(np.diff(X12[:,:2],axis=1))/100.
            tolmag = int(np.log10(tol))-1
            x1u = np.unique(np.round(X12[0,:], -tolmag))
            x2u = np.unique(np.round(X12[1,:], -tolmag))
            indx1 = np.digitize(X12[0,:], 0.5*(x1u[1:]+x1u[:-1]))
            indx2 = np.digitize(X12[1,:], 0.5*(x2u[1:]+x2u[:-1]))
            indx1u, indx2u = np.unique(indx1), np.unique(indx2)
            x1u = np.unique([np.mean(X12[0,indx1==ii]) for ii in indx1u])
            x2u = np.unique([np.mean(X12[1,indx2==ii]) for ii in indx2u])
    else:
        x1u, x2u = x12u
    if nx12 is None:
        nx1, nx2 = x1u.size, x2u.size
    else:
        nx1, nx2 = nx12

    Dx12 = (x1u[1]-x1u[0], x2u[1]-x2u[0])
    ind = np.zeros((nx1,nx2),dtype=np.int64)

    indr = np.array([np.digitize(X12[0,:], 0.5*(x1u[1:]+x1u[:-1])),
                     np.digitize(X12[1,:], 0.5*(x2u[1:]+x2u[:-1]))])
    ind[indr[0,:],indr[1,:]] = np.arange(0,X12.shape[1])
    return x1u, x2u, ind, Dx12

###########################################################
###########################################################
#       Fast computation of cones with rays
###########################################################


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
#       Fast computation of poly
###########################################################


def _compute_VesPoly(R=None, r=None, elong=None, Dshape=None,
                     divlow=None, divup=None, nP=None):
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
    R:          None / int / float
        Major radius used as a center of the vessel
        If None, defaults to 2.4
    r :         None / int / float
        Minor radius of the vessel
        If None, defaults to 1.
    elong:      None / int / float
        Dimensionless elongation parameter in [-1;1]
        If None, defaults to 0.
    Dshape:     None / int / float
        Dimensionless parameter for the D-shape (in-out asymmetry) in [0;1]
        If None, defaults to 0.
    divlow:     None / bool
        Flag indicating whether to incude a lower divertor-like shape
        If None, defaults to True
    divup:      None / bool
        Flag indicating whether to incude an upper divertor-like shape
        If None, defaults to True
    nP :        None / int
        Parameter specifying approximately the number of points of the vessel
        If None, defaults to 200

    Return
    ------
    poly:       np.ndarray
        Closed (2,nP) polygon of the vacuum vessel, optionnally with divertors
    pbump:      np.ndarray
        Closed (2,N) polygon defining the outer bumper
    pbaffle:    np.ndarray
        Closed (2,N) polygon defining the lower baffle
    """

    # Check inputs
    if R is None:
        R = 2.4
    if r is None:
        r = 1.
    if elong is None:
        elong = 0.
    if Dshape is None:
        Dshape = 0.
    if divlow is None:
        divlow = True
    if divup is None:
        divup = True
    if nP is None:
        nP = 200

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


###########################################################
###########################################################
#       Fast computation of camera parameters
###########################################################


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
    else:
        P = np.asarray(P, dtype=float).ravel()
        assert P.size==3

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
        N12 = np.array([N12,N12],dtype=np.int64)
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

    if angs is None:
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





def _compute_CamLOS1D_pinhole(P=None, F=0.1, D12=0.1, N12=100,
                              angs=[-np.pi,0.,0.], nIn=None,
                              VType='Tor', defRY=None, Lim=None,
                              return_Du=False):

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
    if return_Du:
        us = P[:,np.newaxis]-Ds
        us = us / np.sqrt(np.sum(us**2,axis=0))[np.newaxis,:]
        return Ds, us
    else:
        return Ds, P, d2


_comdoc1 = _comdoc.format('1','',
                          'Extension of the detector alignment along e2',
                          '',
                          'Number of detectors along e2',
                          '')

_compute_CamLOS1D_pinhole.__doc__ = _comdoc1


def _compute_CamLOS2D_pinhole(P=None, F=0.1, D12=0.1, N12=100,
                              angs=[-np.pi,0.,0.], nIn=None,
                              VType='Tor', defRY=None, Lim=None,
                              return_Du=False):

    # Check/ format inputs
    P, F, D12, N12, angs, nIn, e1, e2, VType\
            = _compute_PinholeCam_checkformatinputs(P=P, F=F, D12=D12, N12=N12,
                                                    angs=angs, nIn=nIn,
                                                    VType=VType, defRY=defRY,
                                                    Lim=Lim)

    # Get starting points
    d1 = 0.5*D12[0]*np.linspace(-1.,1.,N12[0],endpoint=True)
    d2 = 0.5*D12[1]*np.linspace(-1.,1.,N12[1],endpoint=True)
    d1f = np.tile(d1,N12[1])
    d2f = np.repeat(d2,N12[0])
    d1e = d1f[np.newaxis,:]*e1[:,np.newaxis]
    d2e = d2f[np.newaxis,:]*e2[:,np.newaxis]

    # Here compute ind12
    indflat2img = None
    indimg2flat = None

    Ds = P[:,np.newaxis] - F*nIn[:,np.newaxis] + d1e + d2e
    if return_Du:
        us = P[:,np.newaxis]-Ds
        us = us / np.sqrt(np.sum(us**2,axis=0))[np.newaxis,:]
        return Ds, us
    else:
        return Ds, P, d1, d2, indflat2img, indimg2flat



_extracom2 = '(N1,) coordinates array of the LOS starting point along local\n\
\t    vector e1 (0 being the perpendicular to the pinhole on the detector plane)'
_comdoc2 = _comdoc.format('2','/ list',
                          'Extended to [D12,D12] if a float is provided',
                          '/ list',
                          'Extended to [D12,D12] if a float is provided',
                          'd1:    np.ndarray\n\t    '+_extracom2)

_compute_CamLOS2D_pinhole.__doc__ = _comdoc2


# #############################################################################
# #############################################################################
#       Fast creation of config
# #############################################################################


def _get_listconfig(
    dconfig=None,
    dconfig_shortcuts=None,
    returnas=str
):
    """ Hidden function generating the config names table as a str or dict """

    # Check inputs
    if dconfig is None:
        dconfig = _def_config._DCONFIG
    if dconfig_shortcuts is None:
        dconfig_shortcuts = _def_config._DCONFIG_SHORTCUTS
    assert returnas in [dict, str]

    # Get dict of configs
    dc = {k0: [k0] + sorted([k1 for k1, v1 in dconfig_shortcuts.items()
                             if v1 == k0])
          for k0 in sorted(dconfig.keys())}

    # Return
    if returnas is dict:
        return dc
    else:
        l0 = np.max([len(k0) for k0 in dc.keys()] + [len('unique names')])
        l1 = np.max([len(str(v0)) for v0 in dc.values()] + [len('shortcuts')])
        msg = ("\n\t" + "unique names".ljust(l0) + "\tshortcuts"
               + "\n\t" + "-"*l0 + " \t" + "-"*l1
               + "\n\t- "
               + "\n\t- ".join(["{}\t{}".format(k0.ljust(l0), v0)
                                for k0, v0 in dc.items()]))
        return msg


def get_available_config(
    dconfig=None,
    dconfig_shortcuts=None,
    verb=True,
    returnas=False,
):
    """ Print a table showing all pre-defined config

    Each pre-defined config in tofu can be called by its unique name or
    by a series of shortcuts / alterantive names refereing to the same unique
    name. this feature is useful for retro-compatibility and for making sure a
    standard name always refers to the latest (most detailed) available version
    of the geometry.

    Can also return the table as str

    No input arg needed:
        >>> import tofu as tf
        >>> tf.geom.utils.get_available_config()

    """
    msg = (
        "A config is the geometry of a tokamak\n"
        + "You can define your own"
        + ", see online tutorial at:\n\t{}\n".format(_URL_TUTO)
        + "tofu also also provides some pre-defined config ready to load\n"
        + "They are available via their name or via shortcuts\n"
        + _get_listconfig(dconfig=dconfig,
                          dconfig_shortcuts=dconfig_shortcuts)
        + "\n\n  => to get a pre-defined config, call for example:\n"
        + "\tconfig = tf.geom.utils.create_config('ITER')\n"
        + "\n => to also load coils add '-coils' to the config name, e.g.:\n"
        + "\tconfig = tf.geom.utils.create_config('ITER-coils')"
    )
    if verb is True:
        print(msg)
    if returnas in [None, True, str]:
        return msg


def _create_config_testcase_check_inputs(
    config=None,
    path=None,
    dconfig=None,
    dconfig_shortcuts=None,
    returnas=None,
    strict=None,
):
    # config
    if config is None:
        config = _def_config._DEFCONFIG
    if not isinstance(config, str):
        msg = ("Arg config must be a str!\n"
               + "\tProvided: {}".format(config))
        raise Exception(msg)

    # path
    if path is None:
        path = _path_testcases
    if not isinstance(path, str) or not os.path.isdir(path):
        msg = ("Arg path must be a valid path!\n"
               + "\tProvided: {}".format(path))
        raise Exception(msg)
    path = os.path.abspath(path)

    # dconfig
    if dconfig is None:
        dconfig = _def_config._DCONFIG
    if not isinstance(dconfig, dict):
        msg = ("Arg dconfig must be a dict!\n"
               + "\tProvided: {}".format(dconfig))
        raise Exception(msg)

    # dconfig_shortcuts
    if dconfig_shortcuts is None:
        dconfig_shortcuts = _def_config._DCONFIG_SHORTCUTS
    if not isinstance(dconfig_shortcuts, dict):
        msg = ("Arg dconfig_shortcuts must be a dict!\n"
               + "\tProvided: {}".format(dconfig_shortcuts))
        raise Exception(msg)

    # return
    if returnas is None:
        returnas = object
    if returnas not in [object, dict]:
        msg = ("Arg returnas must be either object or dict!\n"
               + "\t-Provided: {}".format(returnas))
        raise Exception(msg)

    # strict
    if strict is None:
        strict = False
    if not isinstance(strict, bool):
        msg = ("Arg strict must be a bool!\n"
               + "\t-Provided: {}".format(strict))
        raise Exception(msg)
    return config, path, dconfig, dconfig_shortcuts, returnas, strict


def _create_config_testcase(
    config=None,
    path=None,
    dconfig=None,
    dconfig_shortcuts=None,
    returnas=None,
    strict=None,
):
    """ Load the desired test case configuration

    Choose from one of the reference preset configurations (from dconfig)

    """

    # -----------
    # Check input
    (
        config, path, dconfig,
        dconfig_shortcuts,
        returnas, strict,
    ) = _create_config_testcase_check_inputs(
        config=config,
        path=path,
        dconfig=dconfig,
        dconfig_shortcuts=dconfig_shortcuts,
        returnas=returnas,
        strict=strict,
     )

    # --------------------------
    # Check config is available
    coils = 'coils' in config
    if coils is True:
        config = config.replace('-coils', '')
    if config in dconfig.keys():
        pass

    elif config in dconfig_shortcuts.keys():
        # Get corresponding config
        config = dconfig_shortcuts[config]

    else:
        msg = ("\nThe provided config name is not valid.\n"
               + get_available_config(verb=False, returnas=str)
               + "\n\n  => you provided: {}\n".format(config))
        raise Exception(msg)

    # -------------------------
    # Get file and build config

    lf = [f for f in os.listdir(path) if f[-4:]=='.txt']
    lS = []
    # include / exclude coils
    if coils is True:
        lcls = sorted([kk for kk in dconfig[config].keys() if kk != 'Exp'])
    else:
        lcls = sorted([
            kk for kk in dconfig[config].keys()
            if kk != 'Exp' and 'Coil' not in kk
        ])

    Exp = dconfig[config]['Exp']
    dfail = {}
    for cc in lcls:
        for ss in dconfig[config][cc]:
            k0 = '{}_{}'.format(cc, ss)
            ff = [
                f for f in lf
                if all([s in f for s in [cc, Exp, ss]])
            ]
            if len(ff) == 0:
                msg = (
                    "No matching files in {} ".format(path)
                    + "for ['{}', '{}']".format(cc, ss)
                )
                dfail[k0] = msg
                continue
            elif len(ff) > 1:
                # More demanding criterion
                ssbis, Expbis = '_'+ss+'.txt', '_Exp'+Exp+'_'
                ff = [fff for fff in ff if ssbis in fff and Expbis in fff]
                if len(ff) == 0:
                    msg = (
                        "No matching files in {} ".format(path)
                        + "for ['{}', '{}', '{}']".format(cc, ssbis, Expbis)
                    )
                    dfail[k0] = msg
                    continue
                elif len(ff) > 1:
                    msg = (
                        "Many ({}) matching files in {} ".format(len(ff), path)
                        + "for ['{}', '{}', '{}']".format(cc, ssbis, Expbis)
                    )
                    dfail[k0] = msg
                    continue

            pfe = os.path.join(path, ff[0])
            try:
                obj = eval('_core.'+cc).from_txt(
                    pfe, Name=ss, Type='Tor',
                    Exp=dconfig[config]['Exp'],
                    returnas=returnas,
                )
                if returnas not in ['object', object]:
                    obj = ((
                        ss, {'Poly': obj[0], 'pos': obj[1], 'extent': obj[2]}
                    ),)
                lS.append(obj)
            except Exception as err:
                msg = "Error loading: {}".format(str(err))
                dfail[k0] = msg
                continue

    if len(dfail) > 0:
        lstr = ['\t- {}: {}'.format(k0, v0) for k0, v0 in dfail.items()]
        msg = (
            "The following structures could not be loaded:\n"
            + "\n".join(lstr)
        )
        if strict is True or len(lS) == 0:
            raise Exception(msg)
        else:
            warnings.warn(msg)

    if returnas is dict:
        conf = dict([tt for tt in lS])
    else:
        conf = _core.Config(Name=config, Exp=dconfig[config]['Exp'], lStruct=lS)
    return conf


def create_config(
    case=None, Exp='Dummy', Type='Tor',
    Lim=None, Bump_posextent=[np.pi/4., np.pi/4],
    R=None, r=None, elong=None, Dshape=None,
    divlow=None, divup=None, nP=None,
    returnas=None, strict=None,
    SavePath='./', path=_path_testcases,
):
    """ Create easily a tofu.geom.Config object

    In tofu, a Config (short for geometrical configuration) refers to the 3D
    geometry of a fusion device.
    It includes, at least, a simple 2D polygon describing the first wall of the
    fusion chamber, and can also include other structural elements (tiles,
    limiters...) that can be non-axisymmetric.

    To create a simple Config, provide either the name of a reference test
    case, of a set of geometrical parameters (major radius, elongation...).

    This is just a tool for fast testing, if you want to create a custom
    config, use directly tofu.geom.Config and provide the parameters you want.

    Parameters
    ----------
    case :      str
        The name of a reference test case, if provided, this arguments is
        sufficient, the others are ignored
    Exp  :      str
        The name of the experiment
    Type :      str
        The type of configuration (toroidal 'Tor' or linear 'Lin')
    Lim_Bump:   list
        The angular (poloidal) limits, in the cross-section of the extension of
        the outer bumper
    R   :       float
        The major radius of the center of the cross-section
    r   :       float
        The minor radius of the cross-section
    elong:      float
        An elongation parameter (in [-1;1])
    Dshape:     float
        A parameter specifying the D-shape of the cross-section (in [-1;1])
    divlow:     bool
        A flag specifying whether to include a lower divertor-like shape
    divup:     bool
        A flag specifying whether to include an upper divertor-like shape
    nP:         int
        Number of points used to describe the cross-section polygon
    out:        str
        FLag indicating whether to return:
            - 'dict'  : the polygons as a dictionary of np.ndarrays
            - 'object': the configuration as a tofu.geom.Config instance
    returnas:   object / dict
        Flag indicating whether to return the config as:
            - object: a Config instance
            - dict: a dict of Struct instances
    strict:     bool
        Flag indicating whether to raise an error if a Struct cannot be loaded
            Otherwise only raises a warning
    path:       str
        Absolute path where to find the test case data
    SavePath:   str
        The default path used for saving Struct and Config objects returned by
        the routine.

    Return
    ------
    conf:   tofu.geom.Config / dict
        Depending on the value of parameter out, either:
            - the tofu.geom.Config object created
            - a dictionary of the polygons and their pos/extent (if any)
    """

    lp = [R, r, elong, Dshape, divlow, divup, nP]
    lpstr = '[R, r, elong, Dshape, divlow, divup, nP]'
    lc = [case is not None,
          any([pp is not None for pp in lp])]
    if np.sum(lc) > 1:
        msg = ("Please provide either:\n"
               + "\t- case: the name of a pre-defined config\n"
               + "\t- geometrical parameters {}\n\n".format(lpstr))
        raise Exception(msg)
    elif not any(lc):
        msg = get_available_config(verb=False, returnas=str)
        raise Exception(msg)

    # Get config, either from known case or geometrical parameterization
    if case is not None:
        conf = _create_config_testcase(
            config=case,
            path=path,
            returnas=returnas,
            strict=strict,
        )
    else:
        poly, pbump, pbaffle = _compute_VesPoly(R=R, r=r,
                                                elong=elong, Dshape=Dshape,
                                                divlow=divlow, divup=divup,
                                                nP=nP)

        if returnas == 'dict':
            conf = {'Ves':{'Poly':poly},
                    'Baffle':{'Poly':pbaffle},
                    'Bumper':{'Poly':pbump,
                              'pos':Bump_posextent[0],
                              'extent':Bump_posextent[1]}}
        else:
            ves = _core.Ves(Poly=poly, Type=Type, Lim=Lim, Exp=Exp, Name='Ves',
                            SavePath=SavePath)
            baf = _core.PFC(Poly=pbaffle, Type=Type, Lim=Lim,
                            Exp=Exp, Name='Baffle', color='b', SavePath=SavePath)
            bump = _core.PFC(Poly=pbump, Type=Type,
                             pos=Bump_posextent[0], extent=Bump_posextent[1],
                             Exp=Exp, Name='Bumper', color='g', SavePath=SavePath)

            conf = _core.Config(Name='Dummy', Exp=Exp, lStruct=[ves,baf,bump],
                                SavePath=SavePath)
    return conf

###########################################################
#       Fast creation of cam
###########################################################

_P = [1.5,3.2,0.]
_F = 0.1
_testF = 0.4
_D12 = [0.3,0.1]
_nIn = [-0.5,-1.,0.]

_P1 = [1.5,-3.2,0.]
_nIn1 = [-0.5,1.,0.]

_PA = [4.9,-6.9,0.]
_nInA = [-0.75, 1.,0.]
_D12A = [0.4,0.3]
_dcam = {'V1':       {'P':_P1, 'F':_F, 'D12':_D12, 'nIn':_nIn1, 'N12':[1,1]},
         'V10':      {'P':_P, 'F':_F, 'D12':_D12, 'nIn':_nIn, 'N12':[5,2]},
         'V100':     {'P':_P, 'F':_F, 'D12':_D12, 'nIn':_nIn, 'N12':[20,5]},
         'V1000':    {'P':_P, 'F':_F, 'D12':_D12, 'nIn':_nIn, 'N12':[50,20]},
         'V10000':   {'P':_P, 'F':_F, 'D12':_D12, 'nIn':_nIn, 'N12':[125,80]},
         'V100000':  {'P':_P, 'F':_F, 'D12':_D12, 'nIn':_nIn, 'N12':[500,200]},
         'V1000000': {'P':_P, 'F':_F, 'D12':_D12, 'nIn':_nIn, 'N12':[1600,625]},
         'VA1':      {'P':_PA, 'F':_F, 'D12':_D12A, 'nIn':_nInA, 'N12':[1,1]},
         'VA10':     {'P':_PA, 'F':_F, 'D12':_D12A, 'nIn':_nInA, 'N12':[5,2]},
         'VA100':    {'P':_PA, 'F':_F, 'D12':_D12A, 'nIn':_nInA, 'N12':[20,5]},
         'VA1000':   {'P':_PA, 'F':_F, 'D12':_D12A, 'nIn':_nInA, 'N12':[50,20]},
         'VA10000':  {'P':_PA, 'F':_F, 'D12':_D12A, 'nIn':_nInA,'N12':[125,80]},
         'VA100000': {'P':_PA, 'F':_F, 'D12':_D12A,'nIn':_nInA,'N12':[500,200]},
         'VA1000000':{'P':_PA, 'F':_F,'D12':_D12A,'nIn':_nInA,'N12':[1600,625]},
         'testV': {'P':_P, 'F':_testF, 'D12':_D12, 'nIn':_nIn,'N12':[1600,625]}}


_createCamstr = """
    Create a pinhole CamLOS{0}D

    Typical use case:
        >>> import tofu as tf
        >>> conf = tf.load_config('WEST')
        >>> cam1d = tf.geom.utils.create_CamLOS1D(
            pinhole=[3., 0., 0.],       # pinhole position [X, Y, Z] [m]
            focal=0.1,                  # pinhole-sensors distance [m]
            sensor_nb=100,              # number of sensors on sensor plane
            sensor_size=0.1,            # size of sensor plane [m]
            orientation=[np.pi, 0., 0.],# 3 angles orienting whole camera [rad]
            config=conf,                # configuration (for computing LOS)
            Etendues=None,              # Etendues (optional)
            Surfaces=None,              # Surfaces (optional) [m^2]
            Exp='WEST',                 # Experiment
            Diag='Bolo',                # Diagnostic
            Name='Cam1',                # Name of this particular camera
        )
        >>> cam2d = tf.geom.utils.create_CamLOS2D(
            pinhole=[3., 0., 0.],       # pinhole position [X, Y, Z] [m]
            focal=0.1,                  # pinhole-sensors distance [m]
            sensor_nb=[100, 100],       # number of sensors on sensor plane
            sensor_size=[0.1, 0.1],     # size of sensor plane [m]
            orientation=[np.pi, 0., 0.],# 3 angles orienting whole camera [rad]
            config=conf,                # configuration (for computing LOS)
            Etendues=None,              # Etendues (optional)
            Surfaces=None,              # Surfaces (optional) [m^2]
            Exp='WEST',                 # Experiment
            Diag='Bolo',                # Diagnostic
            Name='Cam1',                # Name of this particular camera
        )

    In tofu, a CamLOS is a camera described as a set of Lines of Sight (LOS),
    as opposed to a Cam, where the Volume of Sight (VOS) of all pixels are
    computed in 3D. The CamLOS is then a simplified approximation of the Cam.
    It can be:
        - 1D : like when all LOS are included in a common plane
        - 2D : like a regular everyday camera, producing a 2D image
    For a pinhole camera, all LOS pass through a common point (pinhole)

    This function provides an easy way to create a pinhole CamLOS{0}D
    In tofu, LOS are described as semi-lines using:
        - a starting point (D)
        - a unit vector (u)
    All coordinates are 3D cartesian (X,Y,Z)

    Here, you simply need to provide, either:
        - the name of a standard test case
        - a set of geometrical parameters:
            - pinhole: throught which the camera axis passes
            - focal: distance between pinhole and sensor plane
            - size: dimensions of the sensor planendicular to the camera axis
            - N12 : number of pixels (LOS)
            - angs: 3 angles defining the orientation of the camera

    The computed set of LOS, optionnaly associated to a Config, can then be
    returned as:
        - a tofu object (i.e.: a CamLOS{0}D)
        - a set of starting points (D) and unit vectors (u)
        - a set of starting points (D), a pinhole, and and the coordinates of D
        in the camera frame

    Parameters
    ----------

    Return
    ------
    """

_createCamerr = """ Arg out, specifying the output, must be either:
        - object   : return a tofu object
        - 'Du'     : return the LOS starting points (D) and unit vectors (u)
        - 'pinhole': return the starting points (D), pinhole (P) and the coordinates of D in the camera frame
        """


def _create_CamLOS_check_inputs(
    case=None,
    pinhole=None,
    P=None,         # Deprecated
    focal=None,
    F=None,         # Deprecated
    sensor_nb=None,
    N12=None,       # Deprecated
    sensor_size=None,
    D12=None,       # Deprecated
    orientation=None,
    angs=None,      # Deprecated
    etendues=None,
    Etendues=None,  # Deprecated
    surfaces=None,
    Surfaces=None,  # Deprecated
    config=None,
    Name=None,
    nD=None,
    nIn=None,
    VType='Tor',
    defRY=None,
    Lim=None,
    returnas=object,
):

    # D
    if nD is None:
        nD = 1
    if nD not in [1, 2]:
        msg = (
            """
            Arg nD must be 1 or 2
              You provided:
            {}
            """.format(nD)
        )
        raise Exception(msg)

    # config
    if config is not None:
        Lim = config.Lim
        VType = config.Id.Type
        lS = config.lStructIn
        if len(lS) == 0:
            lS = config.lStruct
        defRY = np.max([ss.dgeom['P1Max'][0] for ss in lS])

    # Get parameters for test case if any
    if case is not None and nD == 2:
        if case not in dcam.keys():
            msg = (
                """
                {} is not a known test case!\n"
                Available test cases include:\n"
                {}
                """.format(
                    case,
                    '\t- ' + '\n\t- '.join(sorted(dcam.keys()))
                )
            )
            raise Exception(msg)

        # Extract pinhole, focal length, width, nb. of pix., unit vector
        pinhole, focal = dcam[case]['P'], dcam[case]['F']
        size, npixels = dcam[case]['D12'], dcam[case]['N12']
        nIn = dcam[case]['nIn']
        nIn = nIn / np.linalg.norm(nIn)

        # Compute the LOS starting points and unit vectors
        nD, VType, Lim, angles = 2, 'Tor', None, None
        Name = case

    # Deprecated args
    dprecate = {
        'pinhole': [pinhole, P, 'P'],
        'focal': [focal, F, 'F'],
        'sensor_size': [sensor_size, D12, 'D12'],
        'sensor_nb': [sensor_nb, N12, 'N12'],
        'orientation': [orientation, angs, 'angs'],
        'etendues': [etendues, Etendues, 'Etendues'],
        'surfaces': [surfaces, Surfaces, 'Surfaces'],
    }
    ldprec = sorted([
        (ss, '{}  ->  {}'.format(vv[2], ss)) for ss, vv in dprecate.items()
        if vv[1] is not None
    ])
    if len(ldprec) > 0:
        msg = (
            """
    The following arguments (left column) are deprecated:
    {}
    Please use the new forms (right column)
    They are equivalent, but more explicit
    Old forms will not work in future versions
            """.format(
                '\t- ' + '\n\t- '.join([ll[1] for ll in ldprec]),
            )
        )
        warnings.warn(msg, DeprecationWarning)

        for ss, _ in ldprec:
            if dprecate[ss][0] is None:
                dprecate[ss][0] = dprecate[ss][1]
        pinhole = dprecate['pinhole'][0]
        focal = dprecate['focal'][0]
        sensor_size = dprecate['sensor_size'][0]
        sensor_nb = dprecate['sensor_nb'][0]
        orientation = dprecate['orientation'][0]

    # returnas
    c0 = returnas in ['Du', dict, object]
    if not c0:
        msg = (
            """
            Arg returnas must be:
                - 'Du': return tuple (D, us)
                - dict: return a dict
                - object: return a tofu instance
            You provided:
                {}
            """.format(returnas)
        )
        raise Exception(msg)

    # Return
    out = [
        nD, Lim, VType, defRY,
        pinhole, focal, sensor_size, sensor_nb, orientation, nIn, Name,
        etendues, surfaces,
    ]

    return out


def _create_CamLOS(
    case=None,
    pinhole=None,
    P=None,         # Deprecated
    focal=None,
    F=None,         # Deprecated
    sensor_nb=None,
    N12=None,       # Deprecated
    sensor_size=None,
    D12=None,       # Deprecated
    orientation=None,
    angs=None,      # Deprecated
    config=None,
    Etendues=None,  # Deprecated
    etendues=None,
    Surfaces=None,  # Deprecated
    surfaces=None,
    Exp=None,
    Diag=None,
    Name=None,
    nD=None,
    dchans=None,
    color=None,
    method=None,
    nIn=None,
    VType='Tor',
    dcam=_dcam,
    defRY=None,
    Lim=None,
    returnas=object,
    SavePath='./',
):
    # -------------
    # Check inputs
    (
        nD, Lim, VType, defRY,
        pinhole, focal, sensor_size, sensor_nb, orientation, nIn, Name,
        etendues, surfaces,
    ) = _create_CamLOS_check_inputs(
        case=case,
        pinhole=pinhole, P=P,
        focal=focal, F=F,
        sensor_nb=sensor_nb, N12=N12,
        sensor_size=sensor_size, D12=D12,
        orientation=orientation, angs=angs,
        Etendues=Etendues, etendues=etendues,
        Surfaces=Surfaces, surfaces=surfaces,
        config=config,
        Name=Name,
        nD=nD,
        nIn=nIn,
        VType=VType,
        defRY=defRY,
        Lim=Lim,
        returnas=returnas,
    )

    # -------------
    # Create
    kwdargs = dict(
        P=pinhole, F=focal, D12=sensor_size, N12=sensor_nb, angs=orientation,
        nIn=nIn, VType=VType, defRY=defRY, Lim=Lim,
    )
    if nD == 1:
        Ds, pinhole, d2 = _compute_CamLOS1D_pinhole(**kwdargs)
    else:
        (Ds, pinhole, d1, d2,
         indflat2img, indimg2flat) = _compute_CamLOS2D_pinhole(**kwdargs)

    # -------------
    # Return
    if returnas is dict:
        dout = {'D': Ds, 'pinhole': pinhole}
        if nD == 2:
            dout.update({
                'x1': d1, 'x2': d2,
                'indflat2img': indflat2img, 'indimg2flat': indimg2flat,
            })
        return dout

    elif returnas == 'Du':
        us = P[:, None] - Ds
        us = us / np.sqrt(np.sum(us**2, axis=0))[None, :]
        return Ds, us

    else:
        cls = eval('_core.CamLOS{0:01.0f}D'.format(nD))
        cam = cls(
            Name=Name, Exp=Exp, Diag=Diag,
            dgeom={'pinhole': pinhole, 'D': Ds}, method=method,
            Etendues=etendues, Surfaces=surfaces, dchans=dchans,
            color=color, config=config, SavePath=SavePath
        )
        return cam


def create_CamLOS1D(**kwdargs):
    return _create_CamLOS(nD=1, **kwdargs)

create_CamLOS1D.__signature__ = inspect.signature(_create_CamLOS)
create_CamLOS1D.__doc__ = _createCamstr.format('1')

def create_CamLOS2D(**kwdargs):
    return _create_CamLOS(nD=2, **kwdargs)

create_CamLOS2D.__signature__ = inspect.signature(_create_CamLOS)
create_CamLOS2D.__doc__ = _createCamstr.format('2')
