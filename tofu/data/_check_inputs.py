

# Standard
import itertools as itt

# Common
import numpy as np
from matplotlib.tri import Triangulation as mplTri



_DRESERVED_KEYS = {
    'dgroup': ['lref', 'ldata'],
    'dref': ['ldata', 'group', 'size', 'ind'],
    'ddata': ['ref', 'group', 'shape', 'data'],
}


_DDEF_PARAMS = {
    'origin': (str, 'unknown'),
    'dim':    (str, 'unknown'),
    'quant':  (str, 'unknown'),
    'name':   (str, 'unknown'),
    'units':  (str, 'a.u.'),
}

_DATA_NONE = False

# #############################################################################
# #############################################################################
#                           Generic
# #############################################################################


def _check_remove(key=None, dkey=None, name=None):
    c0 = isinstance(key, str) and key in dkey.keys()
    c1 = (
        isinstance(key, list)
        and all([isinstance(kk, str) and kk in dkey.keys()])
    )
    if not c0:
        msg = (
            """
            Removed {} must be a str already in self.d{}
            It can also be a list of such
            \t- provided: {}
            \t- already available: {}
            """.format(name, key, sorted(dkey.keys()))
        )
        raise Exception(msg)
    if c0:
        key = [key]
    return key


# #############################################################################
# #############################################################################
#                           Removing routines
# #############################################################################

def _remove_group(
    group=None, dgroup0=None, dref0=None, ddata0=None,
    allowed_groups=None,
    reserved_keys=None,
    ddefparams=None,
    data_none=None,
    max_ndim=None,
):
    """ Remove a group (or list of groups) and all associated ref, data """
    if group is None:
        return dgroup0, dref0, ddata0
    group = _check_remove(key=group, dkey=dgroup0, name='group')

    # Remove groups and orphan ref and data
    for k0 in groups:
        for k1 in dgroup0['lref']:
            del dref0[k1]
        lkdata = [k1 for k1, v1 in ddata0.items() if v1['group'] == (k0,)]
        for kk in lkdata:
            del ddata0[kk]
        del dgroup0[k0]

    # Double-check consistency
    return _consistency(
        ddata=None, ddata0=ddata0,
        dref=None, dref0=dref0,
        dgroup=None, dgroup0=dgroup0,
        allowed_groups=allowed_groups,
        reserved_keys=reserved_keys,
        ddefparams=ddefparams,
        data_none=data_none,
        max_ndim=max_ndim,
    )


def _remove_ref(
    key=None, dgroup0=None, dref0=None, ddata0=None, propagate=None,
    allowed_groups=None,
    reserved_keys=None,
    ddefparams=None,
    data_none=None,
    max_ndim=None,
):
    """ Remove a ref (or list of refs) and all associated data """
    if key is None:
        return group0, dref0, ddata0
    key = _check_remove(
        key=key, dkey=dref0, name='ref',
    )

    for k0 in key:
        # Remove orphan ddata
        for k1 in dref0[k0]['ldata']:
            del ddata0[k1]

        # Remove ref from dgroup['lref']
        for k1 in dgroup0.keys():
            if k0 in dgroup0[k1]['lref']:
                dgroup0[k1]['lref'].remove(k0)
        del dref0[k0]

    # Propagate upward
    if propagate is True:
        lg = [k0 for k0 in dgroup0.keys() if len(dgroup0['lref']) == 0]
        for gg in lg:
            del dgroup0[gg]

    # Double-check consistency
    return _consistency(
        ddata=None, ddata0=ddata0,
        dref=None, dref0=dref0,
        dgroup=None, dgroup0=dgroup0,
        allowed_groups=allowed_groups,
        reserved_keys=reserved_keys,
        ddefparams=ddefparams,
        data_none=data_none,
        max_ndim=max_ndim,
    )


def _remove_data(
    key=None, dgroup0=None, dref0=None, ddata0=None, propagate=None,
    allowed_groups=None,
    reserved_keys=None,
    ddefparams=None,
    data_none=None,
    max_ndim=None,
):
    """ Remove a ref (or list of refs) and all associated data """
    if key is None:
        return group0, dref0, ddata0
    key = _check_remove(
        key=key, dkey=ddata0, name='data',
    )

    for k0 in key:
        # Remove key from dgroup['ldata'] and dref['ldata']
        for k1 in dgroup0.keys():
            if k0 in dgroup0[k1]['ldata']:
                dgroup0[k1]['ldata'].remove(k0)
            if k0 in dref0[k1]['ldata']:
                dref0[k1]['ldata'].remove(k0)
        del ddata0[k0]

    # Propagate upward
    if propagate is True:
        lk = [k0 for k0 in dgroup0.keys() if len(dgroup0['ldata']) == 0]
        for kk in lk:
            del dgroup0[kk]
        lk = [k0 for k0 in dref0.keys() if len(dref0['ldata']) == 0]
        for kk in lk:
            del dref0[kk]

    # Double-check consistency
    return _consistency(
        ddata=None, ddata0=ddata0,
        dref=None, dref0=dref0,
        dgroup=None, dgroup0=dgroup0,
        allowed_groups=allowed_groups,
        reserved_keys=reserved_keys,
        ddefparams=ddefparams,
        data_none=data_none,
        max_ndim=max_ndim,
    )

# #############################################################################
# #############################################################################
#                           dgroup
# #############################################################################


def _check_dgroup(dgroup=None, dgroup0=None, allowed_groups=None):
    """ dgroup must be
    - str: turned to list
    - list of str
    - dict of dict
    """

    # ----------------
    # Trivial case
    if dgroup in [None, {}]:
        return {}

    # ----------------
    # Check conformity
    c0 = isinstance(dgroup, str) and dgroup not in dgroup0.keys()
    c1 = (
        isinstance(dgroup, list)
        and all([
            isinstance(gg, str) and gg not in dgroup0.keys() for gg in dgroup
        ])
    )
    c2 = (
        isinstance(dgroup, dict)
        and all([
            isinstance(k0, str)
            and k0 not in dgroup0.keys()
            and isinstance(v0, dict)
            and all([
                isinstance(k1, str)
                and k1 in _DRESERVED_KEYS['dgroup']
                and isinstance(v1, list)
                and all([isinstance(v2, str) for v2 in v1])
                for k1, v1 in v0.items()
            ])
            for k0, v0 in dgroup.items()
        ])
    )
    if not (c0 or c1 or c2):
        msg = (
            """
            Added group must be either:
            \t- str: not already in self.dgroup
            \t- list of str: each not already in self.dgroup
            \t- dict: each key not already in self.dgroup, each value a dict
            \t- allowed keys in values are:
            \t\t- {}
            You provided:
            \t- {}
            Already available in self.dgroup:
            {}
            """.format(
                sorted(_DRESERVED_KEYS['dgroup']),
                dgroup,
                '\t- ' + '\n\t- '.join(sorted(dgroup0.keys())),
            )
        )
        raise Exception(msg)

    # Check compliance with allowed groups, if any
    if allowed_groups is not None:
        if c0:
            lg = [dgroup] if dgroup not in allowed_groups else []
        elif c1:
            lg = [k0 for k0 in dgroup if k0 not in allowed_groups]
        else:
            lg = [k0 for k0 in dgroup>keys() if k0 not in allowed_groups]
        if len(lg) > 0:
            msg = (
                """
                The following group names are not allowed:
                {}

                Only the following group names are allowed:
                {}
                """.format(
                    '\t- ' + '\n\t- '.join(lg),
                    '\t- ' + '\n\t- '.join(allowed_groups),
                )
            )
            raise Exception(msg)

    # Convert if necessary
    if c0:
        dgroup = {dgroup: {'lref': [], 'ldata': []}}
    elif c1:
        dgroup = {k0: {'lref': [], 'ldata': []} for k0 in dgroup}
    else:
        dgroup = {k0: {'lref': [], 'ldata': []} for k0 in dgroup.keys()}

    return dgroup


# #############################################################################
# #############################################################################
#                           dref
# #############################################################################


class DataRefException(Exception):

    def __init__(ref=None, data=None):
        msg = (
            """
            To be a valid reference for {}, provided data must be either:
            \t- np.ndarray:  of dimension 1 with increasing values
            \t- list, tuple: convertible to the above
            \t- dict / other class: used for meshes

            You provided:
            \t- {}

            """.format(ref, data)
        )
        self.message = msg


def _check_dataref(data=None, key=None):
    """ Check the conformity of data to be a valid reference """

    # if not array
    # => try converting or get class (dict, mesh...)
    group = None
    if not isinstance(data, np.ndarray):
        if isinstance(data, list) or isinstance(data, tuple):
            try:
                data = np.array(data)
                size = data.size
            except Exception as err:
                raise DataRefException(ref=key, data=data)
        else:
            try:
                data, size = _check_mesh_temp(data=data, key=key)
                if len(size) == 1:
                    size = size[0]
                group = 'mesh2d'
            except Exception as err:
                import pdb; pdb.set_trace()     # DB
                size = data.__class__.__name__

    # if array => check unique (unique + sorted)
    if isinstance(data, np.ndarray):
        if data.ndim != 1:
            raise DataRefException(ref=key, data=data)

        datau = np.unique(data)
        if not (datau.size == data.size and np.allclose(datau, data)):
            raise DataRefException(ref=key, data=data)
        size = data.size

    return data, size, group


def _check_dref(
    dref=None, dref0=None, dgroup0=None, ddata0=None, allowed_groups=None,
):
    """ Check and format dref

    dref can be:
        - dict

    If some groups are not already on dgroup0
        => completes dgroups0

    If some data is provided
        => returns ddata to be added

    Also think about meshes !!!
    """

    # ----------------
    # Trivial case
    if dref in [None, {}]:
        return {}, None, None

    # ----------------
    # Check conformity
    ngroup = len(dgroup0)
    if ngroup == 1:
        groupref = list(dgroup0.keys())[0]

    # Basis
    # lk_opt = ['ldata', 'size', 'group', 'data']
    c0 = isinstance(dref, dict)
    lc = [
        k0 for k0, v0 in dref.items()
        if not (
            isinstance(k0, str)
            and k0 not in dref0.keys()
            and (
                (
                    ngroup == 1
                    and (
                        type(v0) in [np.ndarray, list, tuple]
                        or (
                            isinstance(v0, dict)
                            and all([isinstance(ss, str) for ss in v0.keys()])
                            and ('size' in v0.keys() or 'data' in v0.keys())
                        )
                    )
                )
                or (
                    (ngroup == 0 or ngroup > 1)
                    and isinstance(v0, dict)
                    and all([isinstance(ss, str) for ss in v0.keys()])
                    and ('size' in v0.keys() or 'data' in v0.keys())
                    and (
                        'group' in v0.keys()
                        or (
                            'data' in v0.keys()
                            and isinstance(v0['data'], dict)
                        )
                    )
                )
            )
        )
    ]

    # Raise exception if non-conformity
    if not (c0 and len(lc)==0):
        msg = (
            """
            Arg dref must be a dict of the form:
            {
                'ref0': {'group': str, 'size': int, ...},       (A)
                'ref1': {'group': str, 'data': np.array, ...},  (B)
                'ref2': {'data': np.array, ...},                (C)
                ...
                'refn': np.array,                               (D)
            }

            Where:
                - each 'refi' is a unique str identifier
                - (A) & (B): 'group' is provided as well as 'size' of 'data'
                - (C): 'group' is not provided if len(self.dgroup) == 1
                - (D): only the data array is provided if len(self.dgroup) == 1

            Non-conform refs
            """
            + '\t- ' + '\n\t- '.join(lc)
        )
        raise Exception(msg)

    # -----------------------
    # Make sure all are dict
    for k0, v0 in dref.items():
        if not isinstance(v0, dict):
            dref[k0] = {'data': v0}

    # ----------------
    # Add size / data if relevant
    ddata_add = {
        k0: {'data': None}
        for k0, v0 in dref.items()
        if 'data' in v0.keys() and k0 not in ddata0.keys()
    }
    for k0, v0 in dref.items():
        if 'data' in v0.keys():
            data, dref[k0]['size'], group = _check_dataref(
                data=v0['data'], key=k0,
            )
            if k0 in ddata_add.keys():
                ddata_add[k0]['data'] = data
                ddata_add[k0]['ref'] = (k0,)
                ddata_add[k0].update({
                    k1: v1 for k1, v1 in v0.items()
                    if k1 not in ['group', 'size', 'ldata']
                })
            if group is not None and dref.get('group') is None:
                dref[k0]['group'] = group

    # Make sure, if ngroup != 1, that NOW all refs have a group
    if ngroup != 1:
        lerr = [k0 for k0, v0 in dref.items() if v0.get('group') is None]
        if len(lerr) > 0:
            msg = "Some groups remain ambiguous!:\n{}".format(lerr)
            raise Exception(msg)

    # ----------------
    # Convert and/or add group if necessary
    for k0, v0 in dref.items():
        if v0.get('group') is None:
            dref[k0]['group'] = groupref

    # Add missing groups
    lgroups = sorted(set([
        v0['group'] for v0 in dref.values()
        if 'group' in v0.keys() and v0['group'] not in dgroup0.keys()
    ]))

    dgroup_add = None
    if len(lgroups) > 0:
        dgroup_add = _check_dgroup(
            lgroups, dgroup0=dgroup0, allowed_groups=allowed_groups,
        )

    # get rid of extra keys
    dref = {
        k0: {k1: v1 for k1, v1 in v0.items() if k1 in _DRESERVED_KEYS['dref']}
        for k0, v0 in dref.items()
    }
    return dref, dgroup_add, ddata_add


# #############################################################################
# #############################################################################
#               ddata - special case: meshes
# #############################################################################


def _get_RZ(arr, name=None, shapeRZ=None):
    if arr.ndim == 1:
        if np.any(np.diff(arr) <= 0.):
            msg = "Non-increasing {}".format(name)
            raise Exception(msg)
    else:
        lc = [np.all(np.diff(arr[0, :])) > 0.,
              np.all(np.diff(arr[:, 0])) > 0.]
        if np.sum(lc) != 1:
            msg = "Impossible to know {} dimension!".format(name)
            raise Exception(msg)
        if lc[0]:
            arr = arr[0, :]
            if shapeRZ[1] is None:
                shapeRZ[1] = name
            if shapeRZ[1] != name:
                msg = "Inconsistent shapeRZ"
                raise Exception(msg)
        else:
            arr = arr[:, 0]
            if shapeRZ[0] is None:
                shapeRZ[0] = name
            if shapeRZ[0] != name:
                msg = "Inconsistent shapeRZ"
                raise Exception(msg)
    return arr, shapeRZ


def _duplicates(arr, arru, nn, name=None, msg=None):
    msg += (
        "  Duplicate {}: {}\n".format(name, nn - arru.shape[0])
        + "\t- {}.shape: {}\n".format(name, arr.shape)
        + "\t- unique shape: {}".format(arru.shape)
    )
    return msg


def _check_trimesh_conformity(nodes, faces, key=None):
    nnodes = nodes.shape[0]
    nfaces = faces.shape[0]

    # Test for duplicates
    nodesu = np.unique(nodes, axis=0)
    facesu = np.unique(faces, axis=0)
    lc = [nodesu.shape[0] != nnodes,
          facesu.shape[0] != nfaces]
    if any(lc):
        msg = "Non-valid mesh ddata[{0}]: \n".format(key)
        if lc[0]:
            msg = _duplicates(nodes, nodesu, nnodes, name='nodes', msg=msg)
        if lc[1]:
            msg = _duplicates(faces, facesu, nfaces, name='faces', msg=msg)
        raise Exception(msg)

    # Test for unused nodes
    facesu = np.unique(facesu)
    c0 = np.all(facesu >= 0) and facesu.size == nnodes
    if not c0:
        ino = str([ii for ii in range(0, nnodes) if ii not in facesu])
        msg = "Unused nodes in ddata[{0}]:\n".format(key)
        msg += "    - unused nodes indices: {}".format(ino)
        warnings.warn(msg)

    # Check counter-clockwise orientation
    x, y = nodes[faces, 0], nodes[faces, 1]
    orient = ((y[:, 1] - y[:, 0])*(x[:, 2] - x[:, 1])
              - (y[:, 2] - y[:, 1])*(x[:, 1] - x[:, 0]))

    clock = orient > 0.
    if np.any(clock):
        msg = ("Some triangles not counter-clockwise\n"
               + "  (necessary for matplotlib.tri.Triangulation)\n"
               + "    => {}/{} triangles reshaped".format(clock.sum(), nfaces))
        warnings.warn(msg)
        faces[clock, 1], faces[clock, 2] = faces[clock, 2], faces[clock, 1]
    return faces


def _check_mesh_temp(data=None, key=None):
    # Check if provided data is mesh (as a dict)

    # ------------
    # Check basics
    lmok = ['rect', 'tri', 'quadtri']
    c0 = (
        isinstance(data, dict)
        and all([ss in data.keys() for ss in ['type']])
        and data['type'] in lmok
        and (
            (
                data['type'] == 'rect'
                and all([ss in data.keys() for ss in ['R', 'Z']])
                and isinstance(data['R'], np.ndarray)
                and isinstance(data['Z'], np.ndarray)
                and data['R'].ndim in [1, 2]
                and data['Z'].ndim in [1, 2]
            )
            or (
                data['type'] in ['tri', 'quadtri', 'quad']
                and all([ss in data.keys() for ss in ['nodes', 'faces']])
                and isinstance(data['nodes'], np.ndarray)
                and isinstance(data['faces'], np.ndarray)
                and data['nodes'].ndim == 2
                and data['faces'].ndim == 2
                and data['faces'].dtype == np.int
                and data['nodes'].shape[1] == 2
                and (
                    (
                        data['type'] in ['tri', 'quadtri']
                        and data['faces'].shape[1] == 3
                    )
                    or (
                        data['type'] == 'quad'
                        and data['faces'].shape[1] == 4
                    )
                )
                and np.max(data['faces']) <= data['nodes'].shape[0]
            )
        )
    )
    if not c0:
        msg = (
            """
            A mesh should be a dict of one of the following form:

                dict(
                 'type': 'rect',
                 'R': np.ndarray (with ndim in [1, 2]),
                 'Z': np.ndarray (with ndim in [1, 2]),
                 'shapeRZ': ('R', 'Z') or ('Z', 'R')
                )

                 dict(
                 'type': 'tri' or 'quadtri',
                 'nodes': np.ndarray of shape (N, 2),
                 'faces': np.ndarray of int of shape (N, 3)
                )

                dict(
                 'type': 'quad',
                 'nodes': np.ndarray of shape (N, 2),
                 'faces': np.ndarray of int of shape (N, 4)
                )

            Provided:
            {}
            """.format(data)
        )
        raise Exception(msg)

    # ------------
    # Check per type
    if data['type'] == 'rect':

        shapeRZ = data.get('shapeRZ', [None, None])
        if shapeRZ is None:
            shapeRZ = [None, None]
        else:
            shapeRZ = list(shapeRZ)

        R, shapeRZ = _get_RZ(data['R'], name='R', shapeRZ=shapeRZ)
        Z, shapeRZ = _get_RZ(data['Z'], name='Z', shapeRZ=shapeRZ)
        shapeRZ = tuple(shapeRZ)

        if shapeRZ not in [('R', 'Z'), ('Z', 'R')]:
            msg = "Inconsistent shapeRZ"
            raise Exception(msg)

        def trifind(
            r, z,
            Rbin=0.5*(R[1:] + R[:-1]),
            Zbin=0.5*(Z[1:] + Z[:-1]),
            nR=R.size, nZ=Z.size,
            shapeRZ=shapeRZ
        ):
            indR = np.searchsorted(Rbin, r)
            indZ = np.searchsorted(Zbin, z)
            if shapeRZ == ('R', 'Z'):
                indpts = indR*nZ + indZ
            else:
                indpts = indZ*nR + indR
            indout = ((r < R[0]) | (r > R[-1])
                      | (z < Z[0]) | (z > Z[-1]))
            indpts[indout] = -1
            return indpts

        data['R'] = R
        data['Z'] = Z
        data['shapeRZ'] = shapeRZ
        data['nR'] = R.size
        data['nZ'] = Z.size
        data['shape'] = (R.size, Z.size)
        data['trifind'] = trifind
        data['ftype'] = data.get('ftype', 0)

        if data['ftype'] != 0:
            msg = "Linear interpolation not handled yet !"
            raise Exception(msg)

    else:
        # Check mesh conformity for triangulation
        data['faces'] = _check_trimesh_conformity(
            nodes=data['nodes'], faces=data['faces'], key=key
        )

        data['nnodes'] = data['nodes'].shape[0]
        data['nfaces'] = data['faces'].shape[0]
        data['ftype'] = data.get('ftype', 0)

        # Convert 'quad' to 'quadtri' if relevant
        if data['type'] == 'quad':
            # Convert to tri mesh (solution for unstructured meshes)
            faces = np.empty((data['nfaces']*2, 3), dtype=int)
            faces[::2, :] = data['faces'][:, :3]
            faces[1::2, :-1] = data['faces'][:, 2:]
            faces[1::2, -1] = data['faces'][:, 0]
            data['faces'] = faces
            data['type'] = 'quadtri'
            data['ntri'] = 2

            # Re-check mesh conformity
            data['faces'] = _check_trimesh_conformity(
                nodes=data['nodes'], faces=data['faces'], key=key
            )

        # Check ntri
        if data['type'] == 'tri':
            data['ntri'] = 1
        elif 'ntri' not in data.keys():
            msg = (
                """
                For ddata[{}] of type 'quadtri', 'ntri' must be provided
                """.format(key)
            )
            raise Exception(msg)

        # Only triangular meshes so far
        if 'tri' in data['type']:
            if data.get('mpltri', None) is None:
                data['mpltri'] = mplTri(
                    data['nodes'][:, 0],
                    data['nodes'][:, 1],
                    data['faces']
                )
            if not isinstance(data['mpltri'], mplTri):
                msg = (
                    """
                    ddata[{}]['mpltri'] must be a matplotlib Triangulation
                    Provided:
                    {}
                    """.format(key, data['mpltri'])
                )
            assert data['ftype'] in [0, 1]
            if data['ftype'] == 1:
                data['shape'] = (data['nnodes'],)
            else:
                data['shape'] = (int(data['nfaces'] / data['ntri']),)

    return data, data['shape']


# #############################################################################
# #############################################################################
#               ddata - special case: roman to int (SpectralLines)
# #############################################################################


def romanToInt(ss):
      """
      :type s: str
      :rtype: int

      source: https://www.tutorialspoint.com/roman-to-integer-in-python
      """
      roman = {
          'I': 1,
          'V': 5,
          'X': 10,
          'L': 50,
          'C': 100,
          'D': 500,
          'M': 1000,
          'IV': 4,
          'IX': 9,
          'XL': 40,
          'XC': 90,
          'CD': 400,
          'CM': 900,
      }
      i = 0
      num = 0
      while i < len(ss):
         if i+1 < len(ss) and ss[i:i+2] in roman:
            num += roman[ss[i:i+2]]
            i += 2
         else:
            num += roman[ss[i]]
            i += 1
      return num


# #############################################################################
# #############################################################################
#                           ddata
# #############################################################################


def _check_data(data=None, key=None, max_ndim=None):
    """ Check the conformity of data to be a valid reference """

    # if not array
    # => try converting or get class (dict, mesh...)
    shape = None
    group = None
    if not isinstance(data, np.ndarray):
        if isinstance(data, list) or isinstance(data, tuple):
            c0 = (
                all([hasattr(oo, '__iter__') for oo in data])
                and len(set([len(oo) for oo in data])) != 1
            )
            c1 = (
                all([hasattr(oo, '__iter__') for oo in data])
                and len(set([len(oo) for oo in data])) == 1
            )
            if c0:
                data = np.array(data, dtype=object)
                shape = (data.shape[0],)
            elif c1:
                data = np.array(data)
            else:
                try:
                    data = np.array(data)
                    shape = data.shape
                except Exception as err:
                    raise DataRefException(ref=key, data=data)
        else:
            try:
                data, shape = _check_mesh_temp(data=data, key=key)
                group = 'mesh2d'
            except Exception as err:
                shape = data.__class__.__name__

    # if array => check unique (unique + sorted)
    if isinstance(data, np.ndarray) and shape is None:
        shape = data.shape

    # Check max_dim if any
    if isinstance(data, np.ndarray) and max_ndim is not None:
        if data.ndim > max_ndim:
            msg = (
                """
                Provided data for ddata[{}] has too many dimensions!
                - ndim:     {}
                - max_ndim: {}
                """.format(key, data.ndim, max_ndim)
            )
            raise Exception(msg)

    # Check if valid ref candidate
    refcandidate = bool(
        isinstance(data, np.ndarray)
        and data.ndim == 1
        and (
            np.all(np.diff(data) > 0.)
            or np.all(np.diff(data) < 0.)
        )
    )

    return data, shape, group, refcandidate


def _check_ddata(
    ddata=None,
    ddata0=None, dref0=None, dgroup0=None,
    reserved_keys=None,
    allowed_groups=None,
    data_none=None,
    max_ndim=None,
):

    # ----------------
    # Trivial case
    if ddata in [None, {}]:
        return {}, None, None
    if data_none is None:
        data_none = _DATA_NONE

    # ----------------
    # Check conformity
    nref = len(dref0)
    if nref == 1:
        refref = list(dref0.keys())[0]

    # Basis
    # lk_opt = ['ldata', 'size', 'group', 'data']
    c0 = isinstance(ddata, dict)
    lc = [
        k0 for k0, v0 in ddata.items()
        if not (
            isinstance(k0, str)
            and k0 not in ddata0.keys()
            and (
                (
                    nref == 1
                    and (
                        type(v0) in [np.ndarray, list, tuple]
                        or (
                            isinstance(v0, dict)
                            and all([isinstance(ss, str) for ss in v0.keys()])
                            and (
                                (
                                    'data' in v0.keys()
                                    and (
                                        v0.get('ref') is None
                                        or isinstance(v0.get('ref'), str)
                                        or isinstance(v0.get('ref'), tuple)
                                        or v0.get('ref') is True
                                    )
                                )
                                or (
                                    data_none is True
                                    and v0.get('data') is None
                                )
                            )
                        )
                    )
                )
                or (
                    (nref == 0 or nref > 1)
                    and isinstance(v0, dict)
                    and all([isinstance(ss, str) for ss in v0.keys()])
                    and (
                        (
                            'data' in v0.keys()
                            and (
                                (
                                'ref' in v0.keys()
                                 and (
                                    isinstance(v0.get('ref'), str)
                                    or isinstance(v0.get('ref'), tuple)
                                    or v0.get('ref') is True
                                 )
                                )
                                or (
                                    isinstance(v0['data'], dict)
                                    or isinstance(v0.get('ref'), str)
                                    or isinstance(v0.get('ref'), tuple)
                                    or v0.get('ref') in [None, True]
                                )
                            )
                        )
                        or (
                            data_none is True
                            and v0.get('data') is None
                        )
                    )
                )
            )
        )
    ]

    # Raise exception if non-conformity
    if not (c0 and len(lc)==0):
        msg = (
            """
            Arg ddata must be a dict of the form:
            dict(
                'data0': {'ref': 'ref0', 'data': list, ...},       (A)
                'data1': {'ref': ('ref0', 'ref1'), 'data': np.array, ...},  (B)
                'data2': {'data': np.array, ...},                (C)
                ...
                'datan': np.array,                               (D)
            )

            Where:
                - each 'datai' is a unique str identifier
                - (A) & (B): 'data' is provided as well as 'ref'
                - (C): 'ref' is not provided if len(self.dref) == 1
                - (D): only the data array is provided if len(self.dgroup) == 1

            If ref = True, the data is itself considered a ref

            The following keys do not match the criteria:
            """
            + '\t- '+'\n\t- '.join(lc)
        )
        raise Exception(msg)

    # ----------------
    # Convert and/or add ref if necessary
    lref_add = None
    for k0, v0 in ddata.items():
            if not isinstance(v0, dict):
                ddata[k0] = {'ref': (refref,), 'data': v0}
            else:
                if v0.get('data') is None:
                    continue
                if v0.get('ref') is None:
                    if not isinstance(v0['data'], dict):
                        ddata[k0]['ref'] = (refref,)
                elif isinstance(v0['ref'], str):
                    ddata[k0]['ref'] = (v0['ref'],)
                elif v0['ref'] is True:
                    if k0 not in dref0.keys():
                        if lref_add is None:
                            lref_add = [k0]
                        else:
                            lref_add.append(k0)
                    ddata[k0]['ref'] = (k0,)


    # Check data and ref vs shape - and optionnally add to ref if mesh2d
    for k0, v0 in ddata.items():
        if v0.get('data') is not None:
            (
                ddata[k0]['data'], ddata[k0]['shape'],
                group, ddata[k0]['refcandidate']
            ) = _check_data(
                data=v0['data'], key=k0, max_ndim=max_ndim,
            )

            # Check if group / mesh2d
            if group is not None:
                c0 = ddata[k0].get('ref') in [None, (k0,)]
                if not c0:
                    msg = (
                        """
                        ddata[{}]['ref'] is a {}
                          => it should have ref = ({},)
                        """.format(k0, group, k0)
                    )
                    raise Exception(msg)
                ddata[k0]['ref'] = (k0,)
                c0 = (
                    (lref_add is None or k0 not in lref_add)
                    and k0 not in dref0.keys()
                )
                if c0:
                    if lref_add is None:
                        lref_add = [k0]
                    else:
                        lref_add.append(k0)

    # Add missing refs (only in ddata)
    dgroup_add = None
    dref_add = None
    lref = list(itt.chain.from_iterable([
        [
            rr for rr in v0['ref']
            if rr not in dref0.keys() and rr in ddata.keys()
        ]
        for v0 in ddata.values() if (
            'ref' in v0.keys() and v0.get('data') is not None
        )
    ]
    ))
    if lref_add is not None:
        lref += lref_add

    if len(lref) > 0:
        lref = set(lref)
        dref_add = {rr: {'data': ddata[rr]['data']} for rr in lref}
        dref_add, dgroup_add, ddata_dadd = _check_dref(
            dref=dref_add, dref0=dref0, ddata0=ddata0, dgroup0=dgroup0,
            allowed_groups=allowed_groups,
        )

    # Check shape vs ref
    for k0, v0 in ddata.items():
        if v0.get('data') is None:
            continue
        c0 = (
            isinstance(v0['ref'], tuple)
            and all([
                ss in dref0.keys() or ss in dref_add.keys() for ss in v0['ref']
            ])
        )
        if isinstance(v0['shape'], tuple):
            shaperef = [
                dref0[rr]['size'] if rr in dref0.keys()
                else dref_add[rr]['size']
                for rr in v0['ref']
            ]
            c1 = (
                len(shaperef) > 1
                or any([isinstance(ss, tuple) for ss in shaperef])
            )
            if c1:
                shaperef = np.r_[shaperef].ravel()
            c0 = c0 and tuple(shaperef) == v0['shape']
        else:
            c0 = v0['ref'] == (k0,)
        if not c0:
            msg = (
                """
                Inconsistent shape vs ref for ddata[{0}]:
                    - ddata['{0}']['ref'] = {1}  ({2})
                    - ddata['{0}']['shape'] = {3}

                If dict / object it should be its own ref!
                """.format(k0, v0['ref'], tuple(shaperef), v0['shape'])
            )
            raise Exception(msg)

    return ddata, dref_add, dgroup_add


# #############################################################################
# #############################################################################
#                           Params
# #############################################################################


def _check_elementioncharge(ddata, lparams=None):
    """ Specific to SpectralLines """

    # Assess if relevant
    c0 = any([ss in lparams for ss in ['ION', 'ion', 'element', 'charge']])
    if not c0:
        return

    for k0, v0 in ddata.items():

        # Get element and charge from ION if any
        if v0.get('ION') is not None:
            indc = 1
            if v0['ION'][1].islower():
                indc = 2
            element = v0['ION'][:indc]
            charge = romanToInt(v0['ION'][indc:]) - 1
            if v0.get('element') is not None and v0['element'] != element:
                msg = (
                    """
                    Inconsistent ION  vs element for key {}:
                    """.format(k0, v0.get('element'), element)
                )
                raise Exception(msg)
            if v0.get('charge') is not None and v0['charge'] != charge:
                msg = (
                    """
                    Inconsistent ION vs charge for key {}:
                    """.format(k0, v0.get('charge'), charge)
                )
                raise Exception(msg)
            ddata[k0]['element'] = element
            ddata[k0]['charge'] = charge

        # Check ion / element / charge consistency
        lc = [
            v0.get('ion') is not None,
            v0.get('element') is not None,
            v0.get('charge') is not None,
        ]
        if not any(lc):
            continue

        # ion provided -> element and charge
        if lc[0]:
            element = ''.join([
                ss for ss in v0['ion'].strip('+') if not ss.isdigit()
            ])
            charge = int(''.join([
                ss for ss in v0['ion'].strip('+') if ss.isdigit()
            ]))
            if lc[1] and v0['element'] != element:
                msg = (
                    'Non-matching element for key {}:\n\t{}\n\t{}'.format(
                        v0['element'], element
                    )
                )
                raise Exception(msg)
            ddata[k0]['element'] = element
            if lc[2] and v0['charge'] != charge:
                msg = (
                    'Non-matching charge for key {}:\n\t{}\n\t{}'.format(
                        v0['charge'], charge
                    )
                )
                raise Exception(msg)
            ddata[k0]['charge'] = charge

        # element and charge provided -> ion
        elif lc[1] and lc[2]:
            ddata[k0]['ion'] = '{}{}+'.format(v0['element'], v0['charge'])

        # Lack of info
        else:
            msg = (
                """
                element / charge / ion cannot be infered for {}
                - element:{}
                - charge: {}
                - ion:    {}
                """.format(
                    v0.get('element'), v0.get('charge'), v0.get('ion'),
                )
            )
            raise Exception(msg)



def _harmonize_params(
    ddata=None, lkeys=None, reserved_keys=None, ddefparams=None,
):

    # Check inputs
    if reserved_keys is None:
        reserved_keys = _DRESERVED_KEYS['ddata']
    if ddefparams is None:
        ddefparams = _DDEF_PARAMS

    # ------------------
    # list of param keys

    # Get list of known param keys
    lparams = set(itt.chain.from_iterable([
        [k1 for k1 in v0.keys() if k1 not in reserved_keys]
        for k0, v0 in ddata.items()
    ]))

    # Add arbitrary params
    if lkeys is not None:
        if isinstance(lkeys, str):
            lkeys = [lkeys]
        c0 = (
            isinstance(lkeys, list)
            and all([isinstance(ss, str) for ss in lkeys])
        )
        if not c0:
            msg = "lkeys must be a list of str!"
            raise Exception(msg)
        lparams = set(lparams).intersection(lkeys)

    # ------------------
    # dparam
    for k0, v0 in ddefparams.items():
        for k1, v1 in ddata.items():
            if k0 not in v1.keys():
                ddata[k1][k0] = v0[1]
            else:
                # Check type if already included
                if not isinstance(ddata[k1][k0], v0[0]):
                    msg = (
                        """
                        Wrong type for parameter:
                            - type(ddata[{}][{}]) = {}
                        - Expected: {}
                        """.format(
                            k1, k0, type(ddata[k1][k0]), v0[0],
                        )
                    )
                    raise Exception(msg)

    for k0 in lparams:
        for k1, v1 in ddata.items():
            ddata[k1][k0] = ddata[k1].get(k0)

    # ------------------
    # Check element / ion / charge
    _check_elementioncharge(ddata, lparams=lparams)
    return ddata


# #############################################################################
# #############################################################################
#                           Consistency
# #############################################################################


def _consistency(
    ddata=None, ddata0=None,
    dref=None, dref0=None,
    dgroup=None, dgroup0=None,
    allowed_groups=None,
    reserved_keys=None,
    ddefparams=None,
    data_none=None,
    max_ndim=None,
):

    # --------------
    # dgroup
    dgroup = _check_dgroup(
        dgroup=dgroup, dgroup0=dgroup0, allowed_groups=allowed_groups,
    )
    dgroup0.update(dgroup)

    # --------------
    # dref
    dref, dgroup_add, ddata_add = _check_dref(
        dref=dref, dref0=dref0, dgroup0=dgroup0, ddata0=ddata0,
        allowed_groups=allowed_groups,
    )
    if dgroup_add is not None:
        dgroup0.update(dgroup_add)
    if ddata_add is not None:
        if ddata is None:
            ddata = ddata_add
        else:
            ddata.update(ddata_add)
    dref0.update(dref)

    # --------------
    # ddata
    ddata, dref_add, dgroup_add = _check_ddata(
        ddata=ddata, ddata0=ddata0, dref0=dref0, dgroup0=dgroup0,
        reserved_keys=reserved_keys, allowed_groups=allowed_groups,
        data_none=data_none, max_ndim=max_ndim,
    )
    if dgroup_add is not None:
        dgroup0.update(dgroup_add)
    if dref_add is not None:
        dref0.update(dref_add)
    ddata0.update(ddata)

    # --------------
    # params harmonization
    ddata0 = _harmonize_params(
        ddata=ddata0,
        ddefparams=ddefparams, reserved_keys=reserved_keys,
    )

    # --------------
    # Complement

    # ddata0
    for k0, v0 in ddata0.items():
        if v0.get('data') is None:
            continue
        ddata0[k0]['group'] = tuple([dref0[rr]['group'] for rr in v0['ref']])

    # dref0
    for k0, v0 in dref0.items():
        dref0[k0]['ldata'] = sorted(set(
            k1 for k1 in ddata0.keys()
            if ddata0[k1].get('data') is not None and k0 in ddata0[k1]['ref']
        ))

    # dgroup0
    for k0, v0 in dgroup0.items():
        dgroup0[k0]['lref'] = sorted(set(
            k1 for k1, v1 in dref0.items() if v1['group'] == k0
        ))
        dgroup0[k0]['ldata'] = sorted(set(
            k1 for k1 in ddata0.keys()
            if ddata0[k1].get('data') is not None and k0 in ddata0[k1]['group']
        ))

    return dgroup0, dref0, ddata0

"""
    # --------------
    # params
    lparam = self._ddata['lparam']
    for kk in self._ddata['lkey']:
        for pp in self._ddata['dict'][kk].keys():
            if pp not in self._reserved_all and pp not in lparam:
                lparam.append(pp)
"""


# #############################################################################
# #############################################################################
#               Switch ref
# #############################################################################


def switch_ref(
    new_ref=None,
    ddata=None,
    dref=None,
    dgroup=None,
    allowed_groups=None,
    reserved_keys=None,
    ddefparams=None,
    data_none=None,
    max_ndim=None,
):
    """Use the provided key as ref (if valid) """

    # Check input
    c0 = (
        new_ref in ddata.keys()
        and ddata[new_ref].get('refcandidate') is True
    )
    if not c0:
        strgroup = [
            '{}: {}'.format(
                k0,
                [
                    k1 for k1 in v0['ldata']
                    if ddata[k1].get('refcandidate') is True
                ]
            )
            for k0, v0 in dgroup.items()
        ]
        msg = (
            """
            Arg new_ref must be a key to a valid refcandidate!
            - Provided: {}

            Available valid ref candidates:
            - {}
            """.format(new_ref, '\n\t- '.join(strgroup))
        )
        raise Exception(msg)

    # Substitute in dref
    old_ref = ddata[new_ref]['ref'][0]
    dref[new_ref] = dict(dref[old_ref])
    del dref[old_ref]

    # substitute in ddata['ref']
    for k0, v0 in ddata.items():
        if v0.get('ref') is not None and old_ref in v0['ref']:
            new = tuple([rr for rr in v0['ref']])
            ddata[k0]['ref'] = tuple([
                new_ref if rr == old_ref else rr
                for rr in v0['ref']
            ])

    return _consistency(
        ddata=ddata, ddata0={},
        dref=dref, dref0={},
        dgroup=dgroup, dgroup0={},
        allowed_groups=None,
        reserved_keys=None,
        ddefparams=None,
        data_none=None,
        max_ndim=None,
    )


# #############################################################################
# #############################################################################
#               Get / set / add / remove param
# #############################################################################


def _get_param(ddata=None, param=None, key=None, ind=None, returnas=np.ndarray):
    """ Return the array of the chosen parameter (or list of parameters)

    Can be returned as:
        - dict: {param0: {key0: values0, key1: value1...}, ...}
        - np[.ndarray: {param0: np.r_[values0, value1...], ...}

    """
    # Trivial case
    lp = [kk for kk in list(ddata.values())[0].keys() if kk != 'data']
    if param is None:
        param = lp

    # Get key (which data to return param for)
    key = _ind_tofrom_key(ddata=ddata, key=key, ind=ind, returnas=str)

    # Check inputs
    lc = [
        isinstance(param, str) and param in lp and param != 'data',
        isinstance(param, list)
        and all([isinstance(pp, str) and pp in lp for pp in param])
    ]
    if not any(lc):
        msg = (
            """
            Arg param must a valid param key of a list of such (except 'data')

            Valid params:
            {}

            Provided:
            {}
            """.format('\t- ' + '\n\t- '.join(lp), param)
        )
        raise Exception(msg)

    if lc[0]:
        param = [param]

    c0 = returnas in [np.ndarray, dict]
    if not c0:
        msg = (
            """
            Arg returnas must be in [np.ndarray, dict]
            Provided: {}
            """.format(returnas)
        )
        raise Exception(msg)

    # Get output
    if returnas == dict:
        out = {
            k0: {k1: ddata[k1][k0] for k1 in key}
            for k0 in param
        }
    else:
        out = {
            k0: [ddata[k1][k0] for k1 in key]
            for k0 in param
        }
    return out


def _set_param(ddata=None, param=None, value=None, ind=None, key=None):
    """ Set the value of a parameter

    values can be:
        - None
        - a unique value (int, float, bool, str, tuple) => common to all keys
        - an iterable of values (array, list) => one for each key
        - a dict of values (per key)

    A subset of keys can be chosen (ind, key, fed to self.select()) to set
    only the values of some key

    """

    # Check param
    lp = [kk for kk in list(ddata.values())[0].keys() if kk != 'data']
    if param is None:
        return
    c0 = isinstance(param, str) and param in lp
    if not c0:
        msg = (
            """
            Provided param in not valid
            Valid param:
            {}

            Provided:
            {}
            """.format('\t- ' + '\n\t- '.join(lp), param)
        )
        raise Exception(msg)

    # Check ind / key
    key = _ind_tofrom_key(ddata=ddata, ind=ind, key=key, returnas='key')

    # Check value
    ltypes = [str, int, np.int, float, np.float, tuple]
    lc = [
        type(value) in ltypes,
        isinstance(value, list) and all([type(tt) in ltypes for tt in value])
        and len(value) == len(key),
        isinstance(value, np.ndarray) and value.shape[0] == len(key),
        isinstance(value, dict)
        and all([
            kk in ddata.keys() and type(vv) in ltypes
            for kk, vv in value.items()
        ])
    ]
    if not (value is None or any(lc)):
        msg = (
            """
            Accepted types for value include:
                - None
                - {}: common to all
                - list, np.ndarray: key by key
                - dict of {key: scalar / str}

            The length of value must match the selected keys ({})
            """.format(ltypes, len(key))
        )
        raise Exception(msg)

    # Update data
    if value is None or lc[0]:
        for kk in key:
            ddata[kk][param] = value
    elif lc[1] or lc[2]:
        for ii, kk in enumerate(key):
            ddata[kk][param] = value[ii]
    else:
        for kk, vv in value.items():
            ddata[kk][param] = vv


def _add_param(ddata=None, param=None, value=None):
    """ Add a parameter, optionnally also set its value """
    lp = [kk for kk in list(ddata.values())[0].keys() if kk != 'data']
    c0 = isinstance(param, str) and param not in lp
    if not c0:
        msg = (
            """
            param must be a str not matching any existing param

            Available param:
            {}

            Provided:
            {}
            """.format(lp, param)
        )
        raise Exception(msg)

    # Initialize and set
    for kk in ddata.keys():
        ddata[kk][param] = None
    _set_param(ddata=ddata, param=param, value=value)


def _remove_param(ddata=None, param=None):
    """ Remove a parameter, none by default, all if param = 'all' """

    # Check inputs
    lp = [kk for kk in list(ddata.values())[0].keys() if kk != 'data']
    if param is None:
        return
    if param == 'all':
        param = lp

    c0 = isinstance(param, str) and param in lp
    if not c0:
        msg = ()
        raise Exception(msg)

    # Remove
    for k0 in ddata.keys():
        del ddata[k0][param]



# #############################################################################
# #############################################################################
#               Selection
# #############################################################################


def _ind_tofrom_key(
    ddata=None, dgroup=None,
    ind=None, key=None, group=None, returnas=int,
):

    # --------------------
    # Check / format input
    lc = [ind is not None, key is not None]
    if not np.sum(lc) <= 1:
        msg = ("Args ind and key cannot be prescribed simulatneously!")
        raise Exception(msg)

    if group is not None:
        if not (isinstance(group, str) and group in group.keys()):
            msg = (
                """
                Provided group must be valid key of dgroup:
                {}

                Provided:
                {}
                """.format(sorted(dgroup.keys()), group)
            )
            raise Exception(msg)

    lret = [int, bool, str, 'key']
    if returnas not in lret:
        msg = (
            """
            Possible values for returnas are:
            {}

            Provided:
            {}
            """.format(lret, returnas)
        )
        raise Exception(msg)

    # -----------------
    # Compute

    # Intialize output
    out = np.zeros((len(ddata),), dtype=bool)

    if not any(lc) and group is not None:
        key = dgroup[group]['ldata']
        lc[1] = True

    # Get output
    lk = list(ddata.keys())
    if lc[0]:

        # Check ind
        if not isinstance(ind, np.ndarray):
            ind = np.atleast_1d(ind).ravel()
        c0 = (
            ind.ndim == 1
            and (
                (ind.dtype == np.bool and ind.size == len(ddata))
                or (ind.dtype == np.int and ind.size <= len(ddata))
        ))
        if not c0:
            msg = "Arg ind must be an iterable of bool or int indices!"
            raise Exception(msg)

        # return
        out[ind] = True
        if returnas in [int, str, 'key']:
            out = out.nonzero()[0]
            if returnas in [str, 'key']:
                out = np.array(
                    [kk for ii, kk in enumerate(lk) if ii in out],
                    dtype=str
                )

    elif lc[1]:

        # Check key
        if isinstance(key, str):
            key = [key]
        c0 = (
            isinstance(key, list)
            and all([isinstance(kk, str) and kk in lk for kk in key])
        )
        if not c0:
            msg = (
                """
                key must be valid key to ddata (or list of such)
                Provided: {}
                """.format(key)
            )
            raise Exception(msg)

        if returnas in ['key', str]:
            out = key
        else:
            for kk in key:
                out[lk.index(kk)] = True
            if returnas == int:
                out = out.nonzero()[0]
    else:
        if returnas == bool:
            out[:] = True
        elif returnas == int:
            out = np.arange(0, len(lk))
        else:
            out = lk
    return out


def _select(ddata=None, log=None, returnas=None, **kwdargs):
    """ Return the indices / keys of data matching criteria

    The selection is done comparing the value of all provided parameters
    The result is a boolean indices array, optionally with the keys list
    It can include:
        - log = 'all': only the data matching all criteria
        - log = 'any': the data matching any criterion

    If log = 'raw', a dict of indices arrays is returned, showing the
    details for each criterion

    """

    # Format and check input
    if returnas is None:
        returnas = int
    if log is None:
        log = 'all'
    assert returnas in [int, bool, str, 'key']
    assert log in ['all', 'any', 'raw']
    if log == 'raw':
        assert returnas == bool

    # Get list of relevant criteria
    lp = [kk for kk in list(ddata.values())[0].keys() if kk != 'data']
    lcritout = [ss for ss in kwdargs.keys() if ss not in lp]
    if len(lcritout) > 0:
        msg = (
            """
            The following criteria correspond to no parameters:
                - {}
              => only use known parameters (self.dparam.keys()):
                - {}
            """.format(lcritout, '\n\t- '.join(lp))
        )
        raise Exception(msg)

    # Prepare array of bool indices and populate
    ind = np.zeros((len(kwdargs), len(ddata)), dtype=bool)
    for ii, kk in enumerate(kwdargs.keys()):
        try:
            par = self.get_param(ddata=ddata, param=kk, returnas=np.ndarray)
            ind[ii, :] = par == kwdargs[kk]
        except Exception as err:
            try:
                ind[ii, :] = [
                    ddata[k0][kk] == kwdargs[kk] for k0 in ddata.keys()
                ]
            except Exception as err:
                pass

    # Format output ind
    if log == 'raw':
        if returnas in [str, 'key']:
            ind = {
                kk: [k0 for jj, k0 in enumerate(ddata.keys()) if ind[ii, jj]]
                for ii, kk in enumerate(kwdargs.keys())
            }
        if returnas == int:
            ind = {
                kk: ind[ii, :].nonzero()[0]
                for ii, kk in enumerate(kwdargs.keys())
            }
        else:
            ind = {kk: ind[ii, :] for ii, kk in enumerate(kwdargs.keys())}
    else:
        if log == 'all':
            ind = np.all(ind, axis=0)
        else:
            ind = np.any(ind, axis=0)
        if returnas == int:
            ind = ind.nonzero()[0]
        elif returnas in [str, 'key']:
            ind = np.array(
                [k0 for jj, k0 in enumerate(ddata.keys()) if ind[jj]],
                dtype=str,
            )
    return ind
