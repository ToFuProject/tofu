# Built-in
import os
import warnings

# Common
import numpy as np


# #############################################################################
#                   Triangular meshes
# #############################################################################

# DEPRECATED ?
# def tri_checkformat_NodesFaces(nodes, indfaces, ids=None):

    # # Check mesh type
    # if indfaces.shape[1] == 3:
        # mtype = 'tri'
    # elif indfaces.shape[1] == 4:
        # mtype = 'quad'
    # else:
        # msg = ("Mesh seems to be neither triangular nor quadrilateral\n"
               # + "  => unrecognized mesh type, not implemented yet")
        # raise Exception(msg)

    # # Check indexing !!!
    # indmax = int(np.nanmax(indfaces))
    # if indmax == nodes.shape[0]:
        # indfaces = indfaces - 1
    # elif indmax > nodes.shape[0]:
        # msg = ("There seems to be an indexing error\n"
               # + "\t- np.max(indfaces) = {}".format(indmax)
               # + "\t- nodes.shape[0] = {}".format(nodes.shape[0]))
        # raise Exception(msg)

    # # Check for duplicates
    # nnodes = nodes.shape[0]
    # nfaces = indfaces.shape[0]
    # nodesu, indnodesu = np.unique(nodes, axis=0, return_index=True)
    # facesu, indfacesu = np.unique(indfaces, axis=0, return_index=True)
    # facesuu = np.unique(facesu)
    # lc = [nodesu.shape[0] != nnodes,
          # facesu.shape[0] != nfaces,
          # facesuu.size != nnodes or np.any(facesuu != np.arange(0, nnodes))]
    # if any(lc):
        # msg = "Non-valid mesh in {}:\n".format(ids)
        # if lc[0]:
            # noddup = [ii for ii in range(0, nnodes) if ii not in indnodesu]
            # msg += ("  Duplicate nodes: {}\n".format(nnodes - nodesu.shape[0])
                    # + "\t- nodes.shape: {}\n".format(nodes.shape)
                    # + "\t- unique nodes.shape: {}\n".format(nodesu.shape)
                    # + "\t- duplicate nodes indices: {}\n".format(noddup))
        # if lc[1]:
            # dupf = [ii for ii in range(0, nfaces) if ii not in indfacesu]
            # msg += ("  Duplicate faces: {}\n".format(nfaces - facesu.shape[0])
                    # + "\t- faces.shape: {}\n".format(indfaces.shape)
                    # + "\t- unique faces.shape: {}".format(facesu.shape)
                    # + "\t- duplicate facess indices: {}\n".format(dupf))
        # if lc[2]:
            # nfu = facesuu.size
            # nodnotf = [ii for ii in range(0, nnodes) if ii not in facesuu]
            # fnotn = [ii for ii in facesuu if ii < 0 or ii >= nnodes]
            # msg += ("  Non-bijective nodes indices vs faces:\n"
                    # + "\t- nb. nodes: {}\n".format(nnodes)
                    # + "\t- nb. unique nodes index in faces: {}\n".format(nfu)
                    # + "\t- nodes not in faces: {}\n".format(nodnotf)
                    # + "\t- faces ind not in nodes: {}\n".format(fnotn))
        # raise Exception(msg)

    # # Test for unused nodes
    # facesu = np.unique(indfaces)
    # c0 = np.all(facesu >= 0) and facesu.size == nnodes
    # if not c0:
        # indnot = [ii for ii in range(0, nnodes) if ii not in facesu]
        # msg = ("Some nodes not used in mesh of ids {}:\n".format(ids)
               # + "    - unused nodes indices: {}".format(indnot))
        # warnings.warn(msg)

    # # Convert to triangular mesh if necessary
    # if mtype == 'quad':
        # # Convert to tri mesh (solution for unstructured meshes)
        # indface = np.empty((indfaces.shape[0]*2, 3), dtype=int)

        # indface[::2, :] = indfaces[:, :3]
        # indface[1::2, :-1] = indfaces[:, 2:]
        # indface[1::2, -1] = indfaces[:, 0]
        # indfaces = indface
        # mtype = 'quadtri'
        # ntri = 2
    # else:
        # ntri = 1

    # # Check orientation
    # x, y = nodes[indfaces, 0], nodes[indfaces, 1]
    # orient = ((y[:, 1] - y[:, 0])*(x[:, 2] - x[:, 1])
              # - (y[:, 2] - y[:, 1])*(x[:, 1] - x[:, 0]))

    # indclock = orient > 0.
    # if np.any(indclock):
        # nclock, ntot = indclock.sum(), indfaces.shape[0]
        # msg = ("Some triangles not counter-clockwise\n"
               # + "  (necessary for matplotlib.tri.Triangulation)\n"
               # + "    => {}/{} triangles reshaped".format(nclock, ntot))
        # warnings.warn(msg)
        # (indfaces[indclock, 1],
         # indfaces[indclock, 2]) = indfaces[indclock, 2], indfaces[indclock, 1]
    # return indfaces, mtype, ntri


# #############################################################################
#                   Rectangular meshes
# #############################################################################

# DEPRECATED ?
def _rect_checkRZ(aa, name='R', shapeRZ=None):
    if aa.ndim == 1 and np.any(np.diff(aa) < 0.):
        msg = "{} must be increasing!".format(name)
        raise Exception(msg)
    elif aa.ndim == 2:
        lc = [np.all(np.diff(aa[0, :])) > 0.,
              np.all(np.diff(aa[:, 0])) > 0.]
        if np.sum(lc) != 1:
            msg = "{} must have exactly one dim increasing".format(name)
            raise Exception(msg)
        if lc[0]:
            aa = aa[0, :]
            if shapeRZ[1] is None:
                shapeRZ[1] = name
            if shapeRZ[1] != name:
                msg = ("Inconsistent shapeRZ[1]\n"
                       + "\t- expected: [{}, ...]\n".format(name)
                       + "\t- provided: {}".format(shapeRZ))
                raise Exception(msg)
        else:
            aa = aa[:, 0]
            if shapeRZ[0] is None:
                shapeRZ[0] = name
            assert shapeRZ[0] == name
    return aa, shapeRZ


def rect_checkformat(R, Z, datashape=None,
                     shapeRZ=None, ids=None):
    if R.ndim not in [1, 2] or Z.ndim not in [1, 2]:
        msg = ""
        raise Exception(msg)

    shapeu = np.unique(np.r_[R.shape, Z.shape])
    if shapeRZ is None:
        shapeRZ = [None, None]

    # Check R, Z
    R, shapeRZ = _rect_checkRZ(R, name='R', shapeRZ=shapeRZ)
    Z, shapeRZ = _rect_checkRZ(Z, name='Z', shapeRZ=shapeRZ)

    if datashape is not None:
        if None in shapeRZ:
            pass
        shapeRZ = tuple(shapeRZ)

        if shapeRZ == ('R', 'Z'):
            datashape_exp = (R.size, Z.size)
        elif shapeRZ == ('Z', 'R'):
            datashape_exp = (Z.size, R.size)
        else:
            msg = "Inconsistent data shape !"
            raise Exception(msg)
        if datashape != datashape_exp:
            msg = ("Inconsistent data shape\n"
                   + "\t- shapeRZ = {}\n".format(shapeRZ)
                   + "\t- datashape expected: {}\n".format(datashape_exp)
                   + "\t- datashape provided: {}\n".format(datashape))
            raise Exception(msg)

    if None not in shapeRZ:
        shapeRZ = tuple(shapeRZ)
        if shapeRZ not in [('R', 'Z'), ('Z', 'R')]:
            msg = ("Wrong value for shapeRZ:\n"
                   + "\t- expected: ('R', 'Z') or ('Z', 'R')\n"
                   + "\t- provided: {}".format(shapeRZ))
            raise Exception(msg)
    return R, Z, shapeRZ, 0
