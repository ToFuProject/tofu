"""
This module contains tests for tofu.geom in its structured version
"""

# Built-in
import os
import shutil
import itertools as itt
import warnings


# Standard
import numpy as np
import matplotlib.pyplot as plt


# tofu-specific
from tofu import __version__
import tofu as tf
import tofu.data as tfd


_HERE = os.path.abspath(os.path.dirname(__file__))
_PATH_DATA = os.path.join(_HERE, 'test_data')
_TOFU_USER = os.path.join(os.path.expanduser("~"), '.tofu')
_CUSTOM = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
_CUSTOM = os.path.join(_CUSTOM, 'scripts', 'tofucustom.py')
_FNAME = 'ITER_JINTRAC_sh134000_run30_public_edgesources_quadtrimesh.npz'
VerbHead = 'tofu.mesh.test_01_checks'


_PFE = os.path.join(_PATH_DATA, _FNAME)
_DTRI = {
    k0: v0.tolist()
    for k0, v0 in dict(np.load(_PFE, allow_pickle=True)).items()
}


#######################################################
#
#     Setup and Teardown
#
#######################################################


def setup_module():
    print("Removing user ~/.tofu/ if any")
    if os.path.isdir(_TOFU_USER):
        shutil.rmtree(_TOFU_USER)
    # Recreating clean .tofu
    # out = subprocess.run(_CUSTOM, stdout=PIPE, stderr=PIPE)
    os.system('python '+_CUSTOM)


def teardown_module():
    print("Removing user ~/.tofu/ if any")
    if os.path.isdir(_TOFU_USER):
        shutil.rmtree(_TOFU_USER)


#######################################################
#
#     Basic instanciation
#
#######################################################


def _add_rect_uniform(plasma):
    # add uniform rect mesh
    plasma.add_mesh_2d_rect(key='m0', domain=[[2, 3], [-1, 1]], res=0.1)


def _add_rect_variable(plasma):
    # add variable rect mesh
    plasma.add_mesh_2d_rect(
        key='m1',
        domain=[[2, 2.3, 2.6, 3], [-1, 0., 1]],
        res=[[0.2, 0.1, 0.1, 0.2], [0.2, 0.1, 0.2]],
    )


def _add_rect_variable_crop(plasma):
    # add variable rect mesh
    plasma.add_mesh_2d_rect(
        key='m2',
        domain=[[2, 2.3, 2.6, 3], [-1, 0., 1]],
        res=[[0.2, 0.1, 0.1, 0.2], [0.2, 0.1, 0.2]],
        crop_poly=tf.load_config('WEST'),
    )


def _add_tri_ntri1(plasma):
    cents = _DTRI['cents']['data']
    cents2 = np.zeros((cents.shape[0]*2, 3))
    cents2[::2, :] = cents[:, :3]
    cents2[1::2, :2] = cents[:, 2:]
    cents2[1::2, -1] = cents[:, 0]

    plasma.add_mesh_2d_tri(
        key='m3',
        knots=_DTRI['nodes']['data'],
        indices=cents2,
    )


def _add_tri_ntri2(plasma):
    plasma.add_mesh_2d_tri(
        key='m4',
        knots=_DTRI['nodes']['data'],
        indices=_DTRI['cents']['data'],
    )


def _add_polar1(plasma, key='m5'):
    """ Time-independent """

    kR, kZ = plasma.dobj['bsplines']['m2_bs1']['apex']
    R = plasma.ddata[kR]['data']
    Z = plasma.ddata[kZ]['data']
    RR = np.repeat(R[:, None], Z.size, axis=1)
    ZZ = np.repeat(Z[None, :], R.size, axis=0)
    rho = (RR - 2.5)**2/0.08 + (ZZ - 0)**2/0.35

    plasma.add_data(
        key='rho1',
        data=rho,
        ref='m2_bs1',
        unit='',
        dim='',
        quant='rho',
        name='rho',
    )

    plasma.add_mesh_1d(
        key=key,
        knots=np.linspace(0, 1.2, 7),
        subkey='rho1',
    )


def _add_polar2(plasma, key='m6'):
    """ Time-dependent """

    kR, kZ = plasma.dobj['bsplines']['m2_bs1']['apex']
    R = plasma.ddata[kR]['data']
    Z = plasma.ddata[kZ]['data']
    RR = np.repeat(R[:, None], Z.size, axis=1)
    ZZ = np.repeat(Z[None, :], R.size, axis=0)

    rho = (RR - 2.5)**2/0.08 + (ZZ - 0)**2/0.35
    angle = np.arctan2(ZZ/2., (RR - 2.5))

    nt = 11
    t = np.linspace(30, 40, nt)
    rho = rho[None, ...] + 0.1*np.cos(t)[:, None, None]**2
    angle = angle[None, ...] + 0.01*np.sin(t)[:, None, None]**2


    if 'nt' not in plasma.dref.keys():
        plasma.add_ref(
            key='nt',
            size=nt,
        )

    if 't' not in plasma.ddata.keys():
        plasma.add_data(
            key='t',
            data=t,
            ref=('nt',),
            dim='time',
        )

    if 'rho2' not in plasma.ddata.keys():
        plasma.add_data(
            key='rho2',
            data=rho,
            ref=('nt', 'm2_bs1'),
            unit='',
            dim='',
            quant='rho',
            name='rho',
        )

    if 'angle2' not in plasma.ddata.keys():
        plasma.add_data(
            key='angle2',
            data=angle,
            ref=('nt', 'm2_bs1'),
            unit='rad',
            dim='',
            quant='angle',
            name='theta',
        )

    # ang
    if key == 'm6':
        ang = np.pi*np.r_[-3./4., -1/4, 0, 1/4, 3/4]
    else:
        ang = None

    # mesh
    plasma.add_mesh_polar(
        key=key,
        radius=np.linspace(0, 1.2, 7),
        angle=ang,
        radius2d='rho2',
        angle2d='angle2',
    )


def _add_bsplines(plasma, key=None, kind=None, angle=None):

    if kind is None:
        kind = ['rect', 'tri', 'polar']
    if key is None:
        key = list(plasma.dobj['mesh'].keys())

    for k0, v0 in plasma.dobj['mesh'].items():
        if v0['type'] not in kind:
            continue
        if k0 not in key:
            continue
        if v0['type'] == 'tri':
            plasma.add_bsplines(key=k0, deg=0)
            plasma.add_bsplines(key=k0, deg=1)
        elif v0['type'] == 'rect':
            plasma.add_bsplines(key=k0, deg=0)
            plasma.add_bsplines(key=k0, deg=1)
            plasma.add_bsplines(key=k0, deg=2)
            plasma.add_bsplines(key=k0, deg=3)
        elif v0['type'] == 'polar':
            if angle is None:
                plasma.add_bsplines(key=k0, deg=0)
                plasma.add_bsplines(key=k0, deg=1)
                plasma.add_bsplines(key=k0, deg=2)
                plasma.add_bsplines(key=k0, deg=3)
            else:
                plasma.add_bsplines(key=k0, deg=0, angle=[None]*5 + [angle])
                plasma.add_bsplines(key=k0, deg=1, angle=[None]*6 + [angle])
                plasma.add_bsplines(key=k0, deg=2, angle=[None]*7 + [angle])
                plasma.add_bsplines(key=k0, deg=3, angle=[None]*8 + [angle])


def _add_data_fix(plasma, key):

    kdata = f'{key}_data_fix'
    shape = plasma.dobj['bsplines'][key]['shape']
    data = np.random.random(shape)

    if kdata not in plasma.ddata.keys():
        plasma.add_data(
            key=kdata,
            data=data,
            ref=key,
        )
    return kdata


def _add_data_var(plasma, key):

    if 't' not in plasma.ddata.keys():
        nt = 11
        t = np.linspace(30, 40, nt)
        plasma.add_ref('nt', nt)
        plasma.add_data(
            key='t',
            data=t,
            ref=('nt',),
            dim='time',
        )

    kdata = f'{key}-data-var'
    shape = plasma.dobj['bsplines'][key]['shape']
    t = plasma.ddata['t']['data']
    tsh = tuple([t.size] + [1 for ii in shape])
    data = np.cos(t.reshape(tsh)) * np.random.random(shape)[None, ...]

    if kdata not in plasma.ddata.keys():
        plasma.add_data(
            key=kdata,
            data=data,
            ref=('nt', key),
        )
    return kdata


#######################################################
#
#     checking routines
#
#######################################################


class Test01_checks_Instanciate():

    @classmethod
    def setup_class(cls):
        pass

    @classmethod
    def setup_method(self):
        pass

    def teardown_method(self):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def test01_mesh2DRect_X_check(self):

        lx = [[1, 2], [1, 2, 3, 4]]
        lres = [None, 10, 0.1, [0.1, 0.2], [0.1, 0.2, 0.3, 0.1]]

        for comb in itt.product(lx, lres):
            if hasattr(lres, '__iter__') and len(lres) != len(lx):
                continue
            x, res, ind = tfd._mesh_checks._mesh2DRect_X_check(
                x=[1, 2, 3, 4],
                res=10,
            )
            if hasattr(lres, '__iter__'):
                assert x_new.size == np.unique(x_new).size == res.size + 1

    def test02_add_mesh_rect_uniform(self):
        plasma = tfd.Collection()
        _add_rect_uniform(plasma)
        _add_bsplines(plasma)

    def test03_add_mesh_rect_variable(self):
        plasma = tfd.Collection()
        _add_rect_variable(plasma)
        _add_bsplines(plasma)

    def test04_add_mesh_rect_variable_crop(self):
        plasma = tfd.Collection()
        _add_rect_variable_crop(plasma)
        _add_bsplines(plasma)

    def test05_add_mesh_tri_ntri1(self):
        plasma = tfd.Collection()
        _add_tri_ntri1(plasma)
        _add_bsplines(plasma)

    def test06_add_mesh_tri_ntri2(self):
        plasma = tfd.Collection()
        _add_tri_ntri2(plasma)
        _add_bsplines(plasma)

    def test07_add_mesh_polar_radial(self):
        plasma = tfd.Collection()
        _add_rect_variable_crop(plasma)
        _add_bsplines(plasma)
        _add_polar1(plasma)
        _add_bsplines(plasma, kind=['polar'])

    # def test08_add_mesh_polar_angle_regular(self):
        # plasma = tfd.Collection()
        # _add_rect_variable_crop(plasma)
        # _add_bsplines(plasma)
        # _add_polar2(plasma)
        # _add_bsplines(plasma, kind=['polar'])

    # def test09_add_mesh_polar_angle_variable(self):
        # plasma = tfd.Collection()
        # _add_rect_variable_crop(plasma)
        # _add_bsplines(plasma)
        # _add_polar2(plasma, key='m7')
        # _add_bsplines(
            # plasma,
            # kind=['polar'],
            # angle=np.pi*np.r_[-3./4., -1/4, 0, 1/4, 3/4],
        # )


#######################################################
#
#     object mesh2D
#
#######################################################


class Test02_Collection():

    @classmethod
    def setup_class(cls):
        pass

    def setup_method(self):
        plasma = tfd.Collection()

        # add rect mesh
        _add_rect_uniform(plasma)
        _add_rect_variable(plasma)
        _add_rect_variable_crop(plasma)

        # add tri mesh
        _add_tri_ntri1(plasma)
        _add_tri_ntri2(plasma)

        # add bsplines
        _add_bsplines(plasma)

        # add polar mesh
        _add_polar1(plasma)
        # _add_polar2(plasma)

        # add bsplines for polar meshes
        _add_bsplines(plasma, kind=['polar'])

        # Add polar with variable poloidal discretization
        # _add_polar2(plasma, key='m7')
        # _add_bsplines(
            # plasma,
            # key=['m7'],
            # angle=np.pi*np.r_[-3./4., -1/4, 0, 1/4, 3/4],
        # )

        # add data
        lbsdata = []
        for k0 in list(plasma.dobj['bsplines'].keys()):
            k1 = f'{k0}_data'
            plasma.add_data(
                key=k1,
                data=np.ones(plasma.dobj['bsplines'][k0]['shape']),
                ref=k0,
                units='W',
            )
            lbsdata.append(k1)

        # store
        self.obj = plasma
        self.lm = list(plasma.dobj['mesh'].keys())
        self.lbs = list(plasma.dobj['bsplines'].keys())
        self.lbsdata = lbsdata

    def teardown_method(self):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def test01_show(self):
        self.obj.show()

    def test02_select_ind(self):

        # Rect mesh
        nn = 4
        lelements = ['knots', None, 'cents', None]
        lind = [None, ([0, 2], [0, 3]), [0, 5, 8], ([0, 5, 6], [0, 2, 3])]
        lcrop = [True, False, True, False]

        # select fom mesh
        for ii, k0 in enumerate(self.lm):

            ind = ii % nn

            if len(self.obj.dobj['mesh'][k0]['shape-c']) == 2:
                indt = self.obj.select_ind(
                    key=k0,
                    ind=lind[ind],
                    elements=lelements[ind],
                    returnas=tuple,
                    crop=lcrop[ind],
                )
                indf = self.obj.select_ind(
                    key=k0,
                    ind=indt,
                    elements=lelements[ind],
                    returnas=np.ndarray,
                    crop=lcrop[ind],
                )
                indt2 = self.obj.select_ind(
                    key=k0,
                    ind=indf,
                    elements=lelements[ind],
                    returnas=tuple,
                    crop=lcrop[ind],
                )
                assert all([np.allclose(indt[jj], indt2[jj]) for jj in [0, 1]])

            elif ind not in [1, 3]:
                indt = self.obj.select_ind(
                    key=k0,
                    ind=lind[ind],
                    elements=lelements[ind],
                    returnas=int,
                )

    def test03_select_mesh(self):

        lind0 = [None, ([0, 2], [0, 4]), [0, 2, 4], ([0, 2, 4], [0, 2, 3])]
        lind1 = [None, [1], 1, [0, 1]]
        lelem = [None, 'cents', 'knots']
        for ii, k0 in enumerate(self.lm):

            if len(self.obj.dobj['mesh'][k0]['shape-c']) == 2:
                lind = lind0
            else:
                lind = lind1

            if self.obj.dobj['mesh'][k0]['type'] == 'polar':
                return_neighbours = False
            else:
                return_neighbours = None if ii == 0 else bool(ii%2)

            out = self.obj.select_mesh_elements(
                key=k0,
                ind=lind[ii%len(lind)],
                elements=lelem[ii%3],
                returnas='ind' if ii%2 == 0 else 'data',
                return_neighbours=return_neighbours,
                crop=ii%3 == 1,
            )

    # def test04_select_bsplines(self):

        # lind0 = [None, ([0, 2], [0, 4]), [0, 2, 4], ([0, 2, 4], [0, 2, 3])]
        # lind1 = [None, [1], 1, [0, 1]]
        # for ii, k0 in enumerate(self.lbs):

            # km = self.obj.dobj['bsplines'][k0]['mesh']
            # if len(self.obj.dobj['bsplines'][k0]['shape']) == 2:
                # lind = lind0
            # else:
                # lind = lind1

            # if self.obj.dobj['mesh'][km]['type'] == 'polar':
                # return_cents = False
                # return_knots = False
            # else:
                # return_cents = None if ii == 1 else bool(ii%3)
                # return_knots = None if ii == 2 else bool(ii%2)

            # out = self.obj.select_bsplines(
                # key=k0,
                # ind=lind[ii%len(lind)],
                # returnas='ind' if ii%3 == 0 else 'data',
                # return_cents=return_cents,
                # return_knots=return_knots,
            # )

    def test05_sample_mesh(self):

        lres = [None, 0.1, [0.1, 0.05]]
        lmode = [None, 'rel', 'abs']
        lgrid = [None, True, False]
        for ii, k0 in enumerate(self.lm):

            res = lres[ii%len(lres)]
            mode = lmode[ii%len(lmode)]
            if self.obj.dobj['mesh'][k0]['type'] == 'tri':
                if mode == 'rel':
                    if res == 0.1:
                        res = 0.5
                    elif res == [0.1, 0.05]:
                        res = [0.5, 0.4]
            elif self.obj.dobj['mesh'][k0]['nd'] == '1d':
                if isinstance(res, list):
                    res = res[0]

            out = self.obj.get_sample_mesh(
                key=k0,
                res=res,
                mode=mode,
                grid=lgrid[ii%len(lgrid)],
            )

    """
    def test06_sample_bspline(self):
        lres = [None, 0.1, 0.01, [0.1, 0.05]]
        lmode = [None, 'rel', 'abs', 'abs']
        lgrid = [None, True, False, False]
        for ii, (k0, v0) in enumerate(self.dobj.items()):
            out = v0.get_sample_bspline(
                res=lres[ii], grid=lgrid[ii], mode=lmode[ii],
            )
    """

    def test07_interpolate_sum(self):
        x = np.linspace(2.2, 2.8, 5)
        y = np.linspace(-0.5, 0.5, 5)
        x = np.tile(x, (y.size, 1))
        y = np.tile(y, (x.shape[1], 1)).T

        dfail = {}
        for ii, k0 in enumerate(self.lbsdata):

            kbs = self.obj.ddata[k0]['bsplines'][0]
            # try:
            val = self.obj.interpolate(
                keys=k0,
                x0=x,
                x1=y,
                grid=False,
                details=False,
                res=None,
                crop=None,
                nan0=ii % 2 == 0,
            )

            # add fix data
            kdata = _add_data_fix(self.obj, kbs)
            val = self.obj.interpolate(
                keys=kdata,
                x0=x,
                x1=y,
                grid=False,
                details=False,
                res=None,
                crop=None,
                nan0=ii % 2 == 0,
            )

            # add time-dependent data
            kdata = _add_data_var(self.obj, kbs)
            val = self.obj.interpolate(
                keys=kdata,
                x0=x,
                x1=y,
                grid=False,
                details=False,
                res=None,
                crop=None,
                nan0=ii % 2 == 0,
            )
            # except Exception as err:
                # dfail[k0] = str(err)

        # raise error if any fail
        if len(dfail) > 0:
            lstr = [f"\t- {k0}: {v0}" for k0, v0 in dfail.items()]
            msg = (
                "The following bsplines could not be interpolated:\n"
                + "\n".join(lstr)
            )
            raise Exception(msg)

    def test08_interpolate_details_vs_sum(self):

        x = np.linspace(2.2, 2.8, 5)
        y = np.linspace(-0.5, 0.5, 5)
        x = np.tile(x, (y.size, 1))
        y = np.tile(y, (x.shape[1], 1)).T

        for ii, k0 in enumerate(self.lbsdata):

            kbs = self.obj.ddata[k0]['bsplines'][0]
            keym = self.obj.dobj['bsplines'][kbs]['mesh']
            mtype = self.obj.dobj['mesh'][keym]['type']

            dout0 = self.obj.interpolate(
                keys=k0,
                ref_key=kbs,
                x0=x,
                x1=y,
                grid=False,
                details=True,
                res=None,
                crop=None,
                nan0=ii % 2 == 0,
                return_params=False,
            )[f'{kbs}_details']

            crop = self.obj.dobj['bsplines'][kbs].get('crop', False)
            nbs = np.prod(self.obj.dobj['bsplines'][kbs]['shape'])
            if isinstance(crop, str):
                nbs = self.obj.ddata[crop]['data'].sum()

            val = dout0['data']
            vshap0 = tuple(np.r_[x.shape, nbs])
            if mtype == 'polar':
                # radius2d can be time-dependent => additional dimension
                vshap = val.shape[-len(vshap0):]
            else:
                vshap = val.shape
            assert vshap == vshap0, val.shape

            dout1 = self.obj.interpolate(
                keys=k0,
                ref_key=kbs,
                x0=x,
                x1=y,
                grid=False,
                details=False,
                res=None,
                crop=None,
                nan0=False,
                val_out=0.,
                return_params=False,
            )[k0]

            val_sum = dout1['data']
            if mtype == 'polar':
                # radius2d can be time-dependent => additional dimension
                vshap_sum = val_sum.shape[-len(x.shape):]
            else:
                vshap_sum = val_sum.shape
            assert vshap_sum == x.shape, val_sum.shape
            assert (val.ndim == x.ndim + 2) == (val_sum.ndim == x.ndim + 1), [val.shape, val_sum.shape]

            indok = np.isfinite(val_sum)
            indok[indok] = val_sum[indok] != 0

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # Does not work for rect mesh
            # because of knots padding used in func_details
            # Due to scpinterp._bspl.evaluate_spline()...
            if mtype in ['tri', 'polar']:   # To be debugged
                assert np.allclose(
                    val_sum[indok],
                    np.nansum(val, axis=-1)[indok],
                    equal_nan=True,
                )
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    def test09_plot_mesh(self):
        lik0 = [None, ([0, 2], [0, 3]), [2, 3], None]
        lic0 = [None, ([0, 2], [0, 3]), None, [2, 3]]
        lik1 = [None, [0, 2], [2, 3], None]
        lic1 = [None, [0, 2], None, [2, 3]]
        for ii, k0 in enumerate(self.lm):

            if self.obj.dobj['mesh'][k0]['type'] == 'rect':
                lik = lik0
                lic = lic0
            elif self.obj.dobj['mesh'][k0]['type'] == 'tri':
                lik = lik1
                lic = lic1
            else:
                lik = None
                lic = None

            dax = self.obj.plot_mesh(
                key=k0,
                ind_knot=lik[ii%len(lik)] if lik is not None else None,
                ind_cent=lic[ii%len(lic)] if lic is not None else None,
            )
        plt.close('all')

    # def test10_plot_bsplines(self):

        # li0 = [None, ([1, 2], [2, 1]), (1, 1), [1, 2, 4]]
        # li1 = [None, [1, 2], (1, 1), [1, 2, 4]]
        # for ii, k0 in enumerate(self.lbs):

            # km = self.obj.dobj['bsplines'][k0]['mesh']
            # if len(self.obj.dobj['mesh'][km]['shape-c']) == 2:
                # li = li0
            # else:
                # li = li1

            # if self.obj.dobj['mesh'][km]['type'] == 'polar':
                # plot_mesh = False
            # else:
                # plot_mesh = True

            # dax = self.obj.plot_bsplines(
                # key=k0,
                # indbs=li[ii%len(li)],
                # knots=bool(ii%3),
                # cents=bool(ii%2),
                # res=0.05,
                # plot_mesh=plot_mesh,
            # )
            # plt.close('all')

    # def test11_plot_as_profile2d(self):

        # # plotting
        # for k0 in self.lbs:

            # # fix
            # k1 = _add_data_fix(self.obj, k0)
            # dax = self.obj.plot_as_profile2d(key=k1, dres=0.05)

            # # time-variable
            # k1 = _add_data_var(self.obj, k0)
            # dax = self.obj.plot_as_profile2d(key=k1, dres=0.05)

            # plt.close('all')

    # def test12_add_bsplines_operator(self):
        # lkey = ['m0-bs0', 'm1-bs1', 'm2-bs2']
        # lop = ['D0N1', 'D0N2', 'D1N2', 'D2N2']
        # lgeom = ['linear', 'toroidal']
        # lcrop = [False, True]

        # dfail = {}
        # for ii, k0 in enumerate(self.lbs):

            # km = self.obj.dobj['bsplines'][k0]['mesh']
            # if self.obj.dobj['mesh'][km]['type'] == 'tri':
                # continue
            # elif self.obj.dobj['mesh'][km]['type'] == 'polar':
                # continue

            # for comb in itt.product(lop, lgeom, lcrop):
                # deg = self.obj.dobj['bsplines'][k0]['deg']

                # if deg == 3 and comb[0] in ['D0N1', 'D0N2', 'D1N2', 'D2N2']:
                    # continue

                # # only test exact operators
                # if int(comb[0][1]) > deg:
                    # # except deg = 0 D1N2
                    # if deg == 0 and comb[0] == 'D1N2':
                        # pass
                    # else:
                        # continue
                # try:
                    # self.obj.add_bsplines_operator(
                        # key=k0,
                        # operator=comb[0],
                        # geometry=comb[1],
                        # crop=comb[2],
                    # )
                # except Exception as err:
                    # dfail[k0] = (
                        # f"key {k0}, op '{comb[0]}', geom '{comb[1]}': "
                        # + str(err)
                    # )

        # # Raise error if any fail
        # if len(dfail) > 0:
            # lstr = [f'\t- {k0}: {v0}' for k0, v0 in dfail.items()]
            # msg = (
                # "The following operators failed:\n"
                # + "\n".join(lstr)
            # )
            # raise Exception(msg)

    # TBF for triangular
    # def test13_compute_plot_geometry_matrix(self, kind=None):

        # # get config and cam
        # conf = tf.load_config('WEST-V0')
        # cam = tf.geom.utils.create_CamLOS1D(
            # pinhole=[3., 1., 0.],
            # orientation=[np.pi, 0., 0],
            # focal=0.1,
            # sensor_nb=50,
            # sensor_size=0.15,
            # config=conf,
            # Diag='SXR',
            # Exp='WEST',
            # Name='cam1',
        # )

        # lbs = list(self.lbs)
        # if kind is not None:
            # lbs = [
                # kbs for kbs in lbs
                # if self.obj.dobj['mesh'][
                    # self.obj.dobj['bsplines'][kbs]['mesh']
                # ]['type'] == kind
            # ]

        # # compute geometry matrices
        # for ii, k0 in enumerate(lbs):
            # self.obj.add_geometry_matrix(
                # key=k0,
                # cam=cam,
                # res=0.01,
                # crop=None,
                # store=True,
            # )

        # # plot geometry matrices
        # imax = 3
        # for ii, k0 in enumerate(self.obj.dobj['matrix']):

            # if '-' in k0 and int(k0[k0.index('-')+1:]) > 0:
                # continue

            # dax = self.obj.plot_geometry_matrix(
                # key=k0,
                # cam=cam,
                # indchan=40,
                # indbf=5,
                # res=0.05,
            # )
            # if ii % imax == 0:
                # plt.close('all')
        # plt.close('all')
