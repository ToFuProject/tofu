# -*- coding: utf-8 -*-


import warnings
import datetime as dtm


import numpy as np
import matplotlib.pyplot as plt    # DB


import Polygon as plg
import datastock as ds


from . import _class8_compute


# ##############################################################
# ##############################################################
#           Local to global coordinates
# ##############################################################


def _get_reflection(
    # inital contour on crystal
    x0=None,
    x1=None,
    # poly to be observed
    poly_x0=None,
    poly_x1=None,
    # observation point
    pt=None,
    add_points=None,
    # functions
    coord_x01toxyz=None,
    coord_x01toxyz_poly=None,
    pts2pt=None,
    ptsvect=None,
    ptsvect_poly=None,
    # timing
    dt=None,
):

    # -------------------
    #     compute
    # -------------------

    # outline to 3d
    px, py, pz = coord_x01toxyz(x0=x0, x1=x1)

    # Compute poly reflections
    vx, vy, vz, _, iok = ptsvect(
        pts_x=pt[0],
        pts_y=pt[1],
        pts_z=pt[2],
        vect_x=px - pt[0],
        vect_y=py - pt[1],
        vect_z=pz - pt[2],
        strict=False,
        return_x01=False,
    )[3:]

    if not np.any(iok):
        # print('iok')
        return None, None

    # project on target plane
    p0, p1 = ptsvect_poly(
        pts_x=px[iok],
        pts_y=py[iok],
        pts_z=pz[iok],
        vect_x=vx[iok],
        vect_y=vy[iok],
        vect_z=vz[iok],
        strict=False,
        return_x01=True,
    )[-2:]

    pa = plg.Polygon(np.array([poly_x0, poly_x1]).T)

    # isinside
    if np.all([pa.isInside(xx, yy) for xx, yy in zip(p0, p1)]):
        return x0, x1

    # intersection
    p_a = pa & plg.Polygon(np.array([p0, p1]).T)
    if p_a.nPoints() < 3:
        # print('pts < 3')
        return None, None

    # plt.figure()
    # plt.plot(
    #     poly_x0, poly_x1, '.-k',
    #     p0, p1, '.-r',
    #     )

    p0, p1 = np.array(p_a.contour(0)).T

    # interpolate to add points
    if p0.size < 50:
        p0, p1 = _class8_compute._interp_poly(
            lp=[p0, p1],
            add_points=add_points,
            mode='min',
            isclosed=False,
            closed=False,
            ravel=True,
        )

    # back to 3d
    px, py, pz = coord_x01toxyz_poly(x0=p0, x1=p1)

    # back projection on crystal (slowest part)
    p0, p1 = pts2pt(
        pt_x=pt[0],
        pt_y=pt[1],
        pt_z=pt[2],
        # poly
        pts_x=px,
        pts_y=py,
        pts_z=pz,
        # surface
        strict=False,
        return_xyz=False,
        return_x01=True,
        debug=False,
        # timing
        dt=dt,
    )
    
    return p0, p1


# ##############################################################
# ##############################################################
#           Global to local coordinates
# ##############################################################

# DEPRECATED ????
# def _get_project_plane(
#     plane_pt=None,
#     plane_nin=None,
#     plane_e0=None,
#     plane_e1=None,
# ):

#     def _project_poly_on_plane_from_pt(
#         pt_x=None,
#         pt_y=None,
#         pt_z=None,
#         poly_x=None,
#         poly_y=None,
#         poly_z=None,
#         vx=None,
#         vy=None,
#         vz=None,
#         plane_pt=plane_pt,
#         plane_nin=plane_nin,
#         plane_e0=plane_e0,
#         plane_e1=plane_e1,
#     ):

#         sca0 = (
#             (plane_pt[0] - pt_x)*plane_nin[0]
#             + (plane_pt[1] - pt_y)*plane_nin[1]
#             + (plane_pt[2] - pt_z)*plane_nin[2]
#         )

#         if vx is None:
#             vx = poly_x - pt_x
#             vy = poly_y - pt_y
#             vz = poly_z - pt_z

#         sca1 = vx*plane_nin[0] + vy*plane_nin[1] + vz*plane_nin[2]

#         k = sca0 / sca1

#         px = pt_x + k * vx
#         py = pt_y + k * vy
#         pz = pt_z + k * vz

#         p0 = (
#             (px - plane_pt[0])*plane_e0[0]
#             + (py - plane_pt[1])*plane_e0[1]
#             + (pz - plane_pt[2])*plane_e0[2]
#         )
#         p1 = (
#             (px - plane_pt[0])*plane_e1[0]
#             + (py - plane_pt[1])*plane_e1[1]
#             + (pz - plane_pt[2])*plane_e1[2]
#         )

#         return p0, p1

#     def _back_to_3d(
#         x0=None,
#         x1=None,
#         plane_pt=plane_pt,
#         plane_e0=plane_e0,
#         plane_e1=plane_e1,
#     ):

#         return (
#             plane_pt[0] + x0*plane_e0[0] + x1*plane_e1[0],
#             plane_pt[1] + x0*plane_e0[1] + x1*plane_e1[1],
#             plane_pt[2] + x0*plane_e0[2] + x1*plane_e1[2],
#         )

#     return _project_poly_on_plane_from_pt, _back_to_3d
