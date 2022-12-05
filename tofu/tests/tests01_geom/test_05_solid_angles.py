"""
This module contains tests for tofu.geom solid angle routines
"""


# Built-in
import os
import warnings


# Standard
import numpy as np
import matplotlib.pyplot as plt

# tofu-specific
from ... import geom as tfg


_PATH_HERE = os.path.dirname(__file__)
_PATH_OUTPUT = os.path.join(_PATH_HERE, 'output')


#######################################################
#
#     Setup and Teardown
#
#######################################################


def clean(path=_PATH_OUTPUT):
    pass


def setup_module(module):
    clean()


def teardown_module(module):
    clean()


#######################################################
#
#     Utilities
#
#######################################################


def _create_poly_2d_ccw():
    """ Reference case, example taken from:

    https://www.geometrictools.com/Documentation/TriangulationByEarClipping.pdf
    """
    # add references (i.e.: store size of each dimension under a unique key)
    outline_x0 = np.r_[0, 10, 20, 30, 40, 28, 22, 15, 4, 6]
    outline_x1 = np.r_[0, -20, 5, -10, 20, 15, 40, 0, 10, 30]

    reflex0 = np.r_[2, 5, 7, 8]
    ears0 = np.r_[3, 4, 6, 9]

    tri = np.array([
        [2, 3, 4],
        [2, 4, 5],
        [2, 5, 6],
        [2, 6, 7],
        [1, 2, 7],
        [0, 1, 7],
        [0, 7, 8],
        [0, 8, 9],
    ])

    cents = np.r_[10, 0, 0]

    nin = np.r_[-1, 0, 0]
    e0 = np.r_[0, -1, 0]
    e1 = np.r_[0, 0, 1]

    return {
        'outline_x0': outline_x0,
        'outline_x1': outline_x1,
        'reflex0': reflex0,
        'ears0': ears0,
        'tri': tri,
        'cents_x': cents[0],
        'cents_y': cents[1],
        'cents_z': cents[2],
        'nin_x': nin[0],
        'nin_y': nin[1],
        'nin_z': nin[2],
        'e0_x': e0[0],
        'e0_y': e0[1],
        'e0_z': e0[2],
        'e1_x': e1[0],
        'e1_y': e1[1],
        'e1_z': e1[2],
    }


def _create_single_triangle():

    poly_x = np.r_[1, 0, 0]
    poly_y = np.r_[0, 1, 0]
    poly_z = np.r_[0, 0, 1]

    cents_x = 1/3.
    cents_y = 1/3.
    cents_z = 1/3.

    u = np.r_[1, 1, 1]
    nin = -u / np.sqrt(3)
    e0 = np.r_[nin[1], -nin[0], 0.]
    e0 = e0 / np.linalg.norm(e0)
    e1 = np.cross(nin, e0)

    outx0 = (
        (poly_x - cents_x)*e0[0]
        + (poly_y - cents_y)*e0[1]
        + (poly_z - cents_z)*e0[2]
    )
    outx1 = (
        (poly_x - cents_x)*e1[0]
        + (poly_y - cents_y)*e1[1]
        + (poly_z - cents_z)*e1[2]
    )

    pts_x = np.r_[0, 1] * u[0]
    pts_y = np.r_[0, 1] * u[1]
    pts_z = np.r_[0, 1] * u[2]

    sa = (4*np.pi / 8.) * np.r_[1, 0]

    return {
        'poly_x': poly_x,
        'poly_y': poly_y,
        'poly_z': poly_z,
        'outline_x0': outx0,
        'outline_x1': outx1,
        'cents_x': cents_x,
        'cents_y': cents_y,
        'cents_z': cents_z,
        'nin_x': nin[0],
        'nin_y': nin[1],
        'nin_z': nin[2],
        'e0_x': e0[0],
        'e0_y': e0[1],
        'e0_z': e0[2],
        'e1_x': e1[0],
        'e1_y': e1[1],
        'e1_z': e1[2],
        'pts_x': pts_x,
        'pts_y': pts_y,
        'pts_z': pts_z,
        'sa': sa,
    }


def _create_single_rectangle():

    a, b = 2, 1
    outline_x0 = a*np.r_[-1, 1, 1, -1]
    outline_x1 = b*np.r_[-1, -1, 1, 1]

    cents = np.r_[0, 0, 0]

    nin = np.r_[0, 0, 1]
    e0 = np.r_[1, 0, 0]
    e1 = np.r_[0, 1, 0]

    pts_z = np.r_[-1, 0.001, 0.1, 0.5, 1, 10, 50]
    pts_x = np.zeros((pts_z.size,))
    pts_y = np.zeros((pts_z.size,))

    # For a rectangle of dimensions (a, b) seen from height h
    # https://www.planetmath.org/solidangleofrectangularpyramid
    # sa = 4 arcsin( ab / sqrt( (a^2 + h^2)*(b^2 + h^2) ) )
    h = pts_z[1:]
    sa = 4*np.arcsin(a*b / np.sqrt((a**2 + h**2) * (b**2 + h**2)))
    sa = np.r_[0, sa]

    return {
        'outline_x0': outline_x0,
        'outline_x1': outline_x1,
        'cents_x': cents[0],
        'cents_y': cents[1],
        'cents_z': cents[2],
        'nin_x': nin[0],
        'nin_y': nin[1],
        'nin_z': nin[2],
        'e0_x': e0[0],
        'e0_y': e0[1],
        'e0_z': e0[2],
        'e1_x': e1[0],
        'e1_y': e1[1],
        'e1_z': e1[2],
        'pts_x': pts_x,
        'pts_y': pts_y,
        'pts_z': pts_z,
        'sa': sa,
    }


def _create_light(npts=5):

    a, b = 2, 1
    outline_x0 = a*np.r_[-1, 1, 1, -1]
    outline_x1 = b*np.r_[-1, -1, 1, 1]

    cents = np.r_[0, 0, 0]

    nin = np.r_[0, 0, 1]
    e0 = np.r_[1, 0, 0]
    e1 = np.r_[0, 1, 0]

    det = {
        'outline_x0': outline_x0,
        'outline_x1': outline_x1,
        'cents_x': cents[0],
        'cents_y': cents[1],
        'cents_z': cents[2],
        'nin_x': nin[0],
        'nin_y': nin[1],
        'nin_z': nin[2],
        'e0_x': e0[0],
        'e0_y': e0[1],
        'e0_z': e0[2],
        'e1_x': e1[0],
        'e1_y': e1[1],
        'e1_z': e1[2],
    }

    ap = {
        'ap0': {
            'poly_x': 10*a*np.r_[-1, 1, 1, -1],
            'poly_y': 10*b*np.r_[-1, -1, 1, 1],
            'poly_z': 0.1*np.ones((4,)),
            'nin': nin,
        },
        'ap1': {
            'poly_x': 0.1*a*np.r_[-1, 1, 1, -1],
            'poly_y': 0.1*b*np.r_[-1, -1, 1, 1],
            'poly_z': 0.2*np.ones((4,)),
            'nin': nin,
        },
        'ap3': {
            'poly_x': 10*a*np.r_[-1, 0, 0, -1],
            'poly_y': 10*b*np.r_[-1, -1, 1, 1],
            'poly_z': 0.25*np.ones((4,)),
            'nin': nin,
        },
    }
    ap['ap2'] = {k0: dict(ap[k0]) for k0 in ['ap0', 'ap1']}
    ap['ap3'] = {k0: dict(ap[k0]) for k0 in ['ap0', 'ap1', 'ap3']}

    pts_z = np.r_[-1, -0.5, np.linspace(1, 50, npts)]
    pts_x = np.zeros((pts_z.size,))
    pts_y = np.zeros((pts_z.size,))

    # For a rectangle of dimensions (a, b) seen from height h
    # https://www.planetmath.org/solidangleofrectangularpyramid
    # sa = 4 arcsin( ab / sqrt( (a^2 + h^2)*(b^2 + h^2) ) )
    h = pts_z[2:]
    sa0 = 4*np.arcsin(a*b / np.sqrt((a**2 + h**2) * (b**2 + h**2)))

    h = pts_z[2:] - 0.2
    sa1 = 4*np.arcsin(
        0.01*a*b / np.sqrt((0.01*a**2 + h**2) * (0.01*b**2 + h**2))
    )

    h = pts_z[2:] - 0.2
    sa2 = 4*np.arcsin(
        0.01*a*b / np.sqrt((0.01*a**2 + h**2) * (0.01*b**2 + h**2))
    )
    sa0 = np.r_[0, 0, sa0]
    sa1 = np.r_[0, 0, sa1]
    sa2 = np.r_[0, 0, sa2]
    sa3 = sa2/2.

    uvx = np.r_[np.nan, np.nan, np.zeros((sa0.size-2,))]
    uvy = np.r_[np.nan, np.nan, np.zeros((sa0.size-2,))]
    uvz = np.r_[np.nan, np.nan, -np.ones((sa0.size-2,))]

    return {
        'pts_x': pts_x,
        'pts_y': pts_y,
        'pts_z': pts_z,
        'det': det,
        'ap': ap,
        'sa0': sa0,
        'sa1': sa1,
        'sa2': sa2,
        'sa3': sa3,
        'uvx': uvx,
        'uvy': uvy,
        'uvz': uvz,
    }


def _create_visibility(npts=4):

    conf = tfg.utils.create_config('ITER-V0')

    a, b = 2, 1
    outline_x0 = a*np.r_[-1, 1, 1, -1]
    outline_x1 = b*np.r_[-1, -1, 1, 1]

    cents = np.r_[5.6, 0, -3.8]

    nin = np.r_[-1, 0, 0]
    e0 = np.r_[0, 1, 0]
    e1 = np.cross(nin, e0)

    det = {
        'outline_x0': outline_x0,
        'outline_x1': outline_x1,
        'cents_x': cents[0],
        'cents_y': cents[1],
        'cents_z': cents[2],
        'nin_x': nin[0],
        'nin_y': nin[1],
        'nin_z': nin[2],
        'e0_x': e0[0],
        'e0_y': e0[1],
        'e0_z': e0[2],
        'e1_x': e1[0],
        'e1_y': e1[1],
        'e1_z': e1[2],
    }

    ap = {
        'ap0': {
            'poly_x': (cents[0] - 0.05)*np.ones((4,)),
            'poly_y': cents[1] + 10*a*np.r_[-1, 1, 1, -1],
            'poly_z': cents[2] + 10*b*np.r_[-1, -1, 1, 1],
            'nin': nin,
        },
        'ap1': {
            'poly_x': (cents[0] - 0.1)*np.ones((4,)),
            'poly_y': cents[1] + 0.1*a*np.r_[-1, 1, 1, -1],
            'poly_z': cents[2] + 0.1*b*np.r_[-1, -1, 1, 1],
            'nin': nin,
        },
        'ap3': {
            'poly_x': (cents[0] - 0.15)*np.ones((4,)),
            'poly_y': cents[1] + 10*a*np.r_[-1, 0, 0, -1],
            'poly_z': cents[2] + 10*b*np.r_[-1, -1, 1, 1],
            'nin': nin,
        },
    }

    # pts
    if npts == 4:
        pts_x = np.r_[5.4, 5.2, 4.8, 4.4]
    else:
        pts_x = np.linspace(5.4, 4.4, npts)
    pts_y = np.zeros((pts_x.size,))
    pts_z = cents[2] * np.ones((pts_x.size,))

    # sa
    h = cents[0] - pts_x - 0.1
    sa = 4*np.arcsin(
        0.01*a*b / np.sqrt((0.01*a**2 + h**2) * (0.01*b**2 + h**2))
    )
    sa = np.r_[sa[:2]*0.5, 0., 0.]

    return {
        'pts_x': pts_x,
        'pts_y': pts_y,
        'pts_z': pts_z,
        'det': det,
        'ap': ap,
        'sa': sa,
        'config': conf,
    }


def _create_etendue(res=None):

    # unit vectors
    nout, ei, ej = np.r_[-1, 0, 0], np.r_[0, -1, 0], np.r_[0, 0, 1]
    nout2 = nout*np.cos(np.pi/6) + ei*np.sin(np.pi/6.)
    ei2 = nout*np.sin(np.pi/6) - ei*np.cos(np.pi/6.)
    nout3 = nout*np.cos(-2.*np.pi/6) + ei*np.sin(-2.*np.pi/6.)
    ei3 = nout*np.sin(-2.*np.pi/6) - ei*np.cos(-2.*np.pi/6.)

    # outlines
    out0, out1 = np.r_[-1, 1, 1, -1, -1], np.r_[-1, -1, 1, 1, -1]
    outa0, outa1 = 2e-2*out0, 5e-3*out1
    outd0, outd1 = 0.04*out0, 0.04*out1

    # centers
    ca, cd = np.r_[10, 0, 0], np.r_[20, 0, 0]

    # aperture polygon
    poly_x = ca[0] + outa0*ei2[0] + outa1*ej[0]
    poly_y = ca[1] + outa0*ei2[1] + outa1*ej[1]
    poly_z = ca[2] + outa0*ei2[2] + outa1*ej[2]

    # aperture dict
    aperture = {
        'a0': {
            'poly_x': poly_x,
            'poly_y': poly_y,
            'poly_z': poly_z,
            'cent': ca,
            'nin': nout2,
            'e0': ei2,
            'e1': ej,
        },
    }

    # detector dict
    det = {
        'outline_x0': outd0,
        'outline_x1': outd1,
        'cents_x': cd[0],
        'cents_y': cd[1],
        'cents_z': cd[2],
        'nin_x': nout3[0],
        'nin_y': nout3[1],
        'nin_z': nout3[2],
        'e0_x': ei3[0],
        'e0_y': ei3[1],
        'e0_z': ei3[2],
        'e1_x': ej[0],
        'e1_y': ej[1],
        'e1_z': ej[2],
    }

    return {
        'aperture': aperture,
        'det': det,
        'etendue': np.r_[1.10852804e-08],
        'res': res,
    }


#######################################################
#
#     Instanciate
#
#######################################################


class Test01_SolidAngles():

    def setup_method(self):
        self.poly_2d_ccw = _create_poly_2d_ccw()
        self.single_triangle = _create_single_triangle()
        self.single_rectangle = _create_single_rectangle()
        self.light = _create_light()
        self.visibility = _create_visibility()
        self.etendue = _create_etendue()

    # ------------------------
    #   Populating
    # ------------------------

    def test01_triangulation_2d_rectangle(self):
        tri = tfg._comp_solidangles.triangulate_polygon_2d(
            2*np.r_[-1., 1, 1, -1],
            np.r_[-1., -1, 1, 1],
        )

        triref = np.array([[0, 1, 2], [1, 2, 3]])
        if np.allclose(tri, triref):
            msg = (
                "Wrong rectangle triangulation:\n"
                f"\t- expected: {triref}\n"
                f"\t- obtained: {tri}\n"
            )

    def test02_triangulation_2d_ccw(self):
        tri = tfg._comp_solidangles.triangulate_polygon_2d(
            self.poly_2d_ccw['outline_x0'],
            self.poly_2d_ccw['outline_x1'],
        )

        if not tri.shape == self.poly_2d_ccw['tri'].shape:
            msg = (
                "Wrong shape of triangulation:\n"
                f"\t- Expected: {self.poly_2d_ccw['tri'].shape}\n"
                f"\t- obtained: {tri.shape}\n"
            )
            raise Exception(msg)

        if not np.allclose(tri, self.poly_2d_ccw['tri']):
            msg = (
                "Wrong tringulation of reference test case!\n"
                f"\t- expected: {self.poly_2d_ccw['tri']}\n"
                f"\t- obtained: {tri}\n"
            )
            raise Exception(msg)

    def test03_solid_angle_triangle(self):
        # single triangle
        sa = tfg.calc_solidangle_apertures(
            pts_x=self.single_triangle['pts_x'],
            pts_y=self.single_triangle['pts_y'],
            pts_z=self.single_triangle['pts_z'],
            apertures=None,
            detectors=self.single_triangle,
            visibility=False,
            return_vector=False,
        ).ravel()

        # check solid angle value
        if np.any(sa < 0.):
            msg = (
                "Solid angle of triangle is negative!\n"
                f"\t- obtained: {sa}"
            )
            raise Exception(msg)

        saref = self.single_triangle['sa']
        if np.any(np.abs(sa - saref) > 1.e-10 * saref):
            msg = (
                "Solid angle of triangle is wrong!\n"
                f"\t- Expected: {saref}\n"
                f"\t- Obtained: {sa}\n"
            )
            raise Exception(msg)

    def test04_solid_angle_rectangle(self):
        # rectangle
        sa = tfg.calc_solidangle_apertures(
            pts_x=self.single_rectangle['pts_x'],
            pts_y=self.single_rectangle['pts_y'],
            pts_z=self.single_rectangle['pts_z'],
            apertures=None,
            detectors=self.single_rectangle,
            visibility=False,
            return_vector=False,
        ).ravel()

        # check solid angle value
        if np.any(sa < 0.):
            msg = (
                "Solid angle of triangle is negative!\n"
                f"\t- obtained: {sa}"
            )
            raise Exception(msg)

        saref = self.single_rectangle['sa']
        if np.any(np.abs(sa - saref) > 1.e-10 * saref):
            msg = (
                "Solid angle of triangle is wrong!\n"
                f"\t- Expected: {saref}\n"
                f"\t- Obtained: {sa}\n"
            )
            raise Exception(msg)

    def test05_solid_angle_light(self):

        for ii in range(4):
            sa = tfg.calc_solidangle_apertures(
                pts_x=self.light['pts_x'],
                pts_y=self.light['pts_y'],
                pts_z=self.light['pts_z'],
                apertures=self.light['ap'][f'ap{ii}'],
                detectors=self.light['det'],
                visibility=False,
                return_vector=False,
            ).ravel()

            # check solid angle value
            saref = self.light[f'sa{ii}']
            if np.any(np.abs(sa - saref) > 1.e-6 * saref):
                msg = (
                    f"Solid angle of light (case{ii}) is wrong!\n"
                    f"\t- Expected: {saref}\n"
                    f"\t- Obtained: {sa}\n"
                )
                raise Exception(msg)

    def test06_solid_angle_vector(self):
        for ii in range(3):
            sa, uvx, uvy, uvz = tfg.calc_solidangle_apertures(
                pts_x=self.light['pts_x'],
                pts_y=self.light['pts_y'],
                pts_z=self.light['pts_z'],
                apertures=self.light['ap'][f'ap{ii}'],
                detectors=self.light['det'],
                visibility=False,
                return_vector=True,
            )
            sa = sa.ravel()
            uvx = uvx.ravel()
            uvy = uvy.ravel()
            uvz = uvz.ravel()

            # check solid angle value
            saref = self.light[f'sa{ii}']
            if np.any(np.abs(sa - saref) > 1.e-6 * saref):
                msg = (
                    f"Solid angle of light (case{ii}) is wrong!\n"
                    f"\t- Expected: {saref}\n"
                    f"\t- Obtained: {sa}\n"
                )
                raise Exception(msg)

            # check unit vectors
            uvxr = self.light['uvx']
            uvyr = self.light['uvy']
            uvzr = self.light['uvz']
            c0 = (
                np.allclose(uvx, uvxr, equal_nan=True)
                and np.allclose(uvx, uvxr, equal_nan=True)
                and np.allclose(uvx, uvxr, equal_nan=True)
            )
            if not c0:
                msg = (
                    f"Unit vectors of light (case{ii}) are wrong!\n"
                    "\t- uvx:\n"
                    f"\t\t- expected {uvxr}\n"
                    f"\t\t- obtained {uvx}\n"
                    "\t- uvy:\n"
                    f"\t\t- expected {uvyr}\n"
                    f"\t\t- obtained {uvy}\n"
                    "\t- uvz:\n"
                    f"\t\t- expected {uvzr}\n"
                    f"\t\t- obtained {uvz}\n"
                )
                raise Exception(msg)

    def test07_solid_angle_visible(self):
        sa = tfg.calc_solidangle_apertures(
            pts_x=self.visibility['pts_x'],
            pts_y=self.visibility['pts_y'],
            pts_z=self.visibility['pts_z'],
            apertures=self.visibility['ap'],
            detectors=self.visibility['det'],
            config=self.visibility['config'],
            visibility=True,
            return_vector=False,
        ).ravel()

        # check solid angle value
        saref = self.visibility['sa']
        if np.any(np.abs(sa - saref) > 1.e-6 * saref):
            msg = (
                f"Solid angle of visibility is wrong!\n"
                f"\t- Expected: {saref}\n"
                f"\t- Obtained: {sa}\n"
            )
            raise Exception(msg)

    def test08_etendue(self):

        detend = tfg.compute_etendue(
            det=self.etendue['det'],
            aperture=self.etendue['aperture'],
            res=self.etendue['res'],
            check=True,
            verb=True,
        )

        # non-regression
        c0 = np.allclose(
            detend['numerical'][-1, 0],
            self.etendue['etendue'],
            atol=1e-11,
            rtol=1e-2,
        )
        if not c0:
            msg = "Regression detected!"
            raise Exception(msg)

        # check match
        c0 = np.allclose(
            detend['analytical'][-1, :],
            detend['numerical'][-1, :],
            atol=1e-11,
            rtol=1e-2,
        )

        if not c0:
            msg = (
                "Mismatching analytical vs numerical etendue!\n"
                f"\t- analytical: {detend['analytical'][-1, :]}\n"
                f"\t- numerical: {detend['numerical'][-1, :]}\n"
            )
            raise Exception(msg)

        # without check
        detend1 = tfg.compute_etendue(
            det=self.etendue['det'],
            aperture=self.etendue['aperture'],
            res=self.etendue['res'],
            check=False,
            verb=False,
            analytical=False,
            plot=False,
        )

        # check match
        c0 = np.allclose(
            detend['numerical'],
            detend1['numerical'],
        )
        if not c0:
            msg = (
                "Mismatching numerical check vs no_check etendue!\n"
                f"\t- with check: {detend['numerical']}\n"
                f"\t- w/o check:  {detend1['numerical']}\n"
            )
            raise Exception(msg)

        plt.close('all')
