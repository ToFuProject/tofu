from tests01_GG import *

def test01_CoordShift2():

    # Tests 1D input
    Pts = np.array([1.,1.,1.])
    pts = GG.CoordShift(Pts, In='(X,Y,Z)', Out='(R,Z)', CrossRef=0.)
    assert pts.shape==(2,) and np.allclose(pts,[np.sqrt(2),1.])
    pts = GG.CoordShift(Pts, In='(X,Y,Z)', Out='(R,Phi,Z)', CrossRef=0.)
    print(pts)
    pts = GG.CoordShift(Pts, In='(R,Z,Phi)', Out='(X,Y,Z)', CrossRef=0.)
    assert pts.shape==(3,) and np.allclose(pts,[np.cos(1.),np.sin(1.),1.])

    # Test 2D input
    Pts = np.array([[1.,1.],[1.,1.],[1.,1.]])
    pts = GG.CoordShift(Pts, In='(X,Y,Z)', Out='(R,Phi,Z)', CrossRef=0.)
    assert pts.shape==(3,2) and np.allclose(pts,[[np.sqrt(2.),np.sqrt(2.)],
                                                 [np.pi/4.,np.pi/4.],[1.,1.]])
    pts = GG.CoordShift(Pts, In='(Phi,Z,R)', Out='(X,Y)', CrossRef=0.)
    assert pts.shape==(2,2) and np.allclose(pts,[[np.cos(1.),np.cos(1.)],
                                                 [np.sin(1.),np.sin(1.)]])
    print(".......all passed......")


test01_CoordShift()
test01_CoordShift2()
