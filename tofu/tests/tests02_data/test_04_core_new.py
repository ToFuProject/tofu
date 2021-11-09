
"""
This module contains tests for tofu.geom in its structured version
"""


# Built-in
import os
import warnings


# Standard
import numpy as np
import scipy.constants as scpct
import matplotlib.pyplot as plt


# tofu-specific
from tofu import __version__
import tofu.data as tfd
import tofu.utils as tfu

_here = os.path.abspath(os.path.dirname(__file__))
VerbHead = 'tofu.data.DataCollection'


#######################################################
#
#     Setup and Teardown
#
#######################################################


def setup_module(module):
    print("")   # this is to get a newline after the dots
    LF = os.listdir(_here)
    lss = ['TFD_', 'Test', '.npz']
    LF = [lf for lf in LF if all([ss in lf for ss in lss])]
    LF = [
        lf for lf in LF
        if not lf[lf.index('_Vv')+2:lf.index('_U')] == __version__
    ]
    print("Removing the following previous test files:")
    print(LF)
    for lf in LF:
        os.remove(os.path.join(_here, lf))
    # print("setup_module before anything in this file")


def teardown_module(module):
    # os.remove(VesTor.Id.SavePath + VesTor.Id.SaveName + '.npz')
    # os.remove(VesLin.Id.SavePath + VesLin.Id.SaveName + '.npz')
    # print("teardown_module after everything in this file")
    # print("") # this is to get a newline
    LF = os.listdir(_here)
    lss = ['TFD_', 'Test', '.npz']
    LF = [lf for lf in LF if all([ss in lf for ss in lss])]
    LF = [
        lf for lf in LF
        if lf[lf.index('_Vv')+2:lf.index('_U')] == __version__
    ]
    print("Removing the following test files:")
    print(LF)
    for lf in LF:
        os.remove(os.path.join(_here, lf))
    pass


# def my_setup_function():
#    print ("my_setup_function")

# def my_teardown_function():
#    print ("my_teardown_function")

# @with_setup(my_setup_function, my_teardown_function)
# def test_numbers_3_4():
#    print 'test_numbers_3_4  <============================ actual test code'
#    assert multiply(3,4) == 12

# @with_setup(my_setup_function, my_teardown_function)
# def test_strings_a_3():
#    print 'test_strings_a_3  <============================ actual test code'
#    assert multiply('a',3) == 'aaa'


#######################################################
#
#     Creating Ves objects and testing methods
#
#######################################################


class Test01_DataCollection(object):

    @classmethod
    def setup_class(cls, Name='data1',  SavePath='./', verb=False):

        # time vectors
        t0 = np.linspace(0, 10, 10)
        t1 = np.linspace(0, 10, 50)
        t2 = np.linspace(0, 10, 100)
        cls.lt = [t0, t1, t2]

        # radii vectors
        r0 = np.linspace(0, 1, 10)
        r1 = np.linspace(0, 1, 50)
        r2 = np.linspace(0, 1, 200)
        cls.lr = [r0, r1, r2]

        # chan
        ch0 = np.arange(0, 2)
        ch1 = np.arange(0, 5)
        cls.lch = [ch0, ch1]

        # meshes
        mesh0 = {
            'type': 'rect',
            'R': np.r_[0, 1, 2, 3],
            'Z': np.r_[0, 1, 2],
            'shapeRZ': ('R', 'Z'),
        }
        mesh1 = {
            'type': 'tri',
            'nodes': np.array([[0, 1, 1, 0], [0, 0, 1, 1]]).T,
            'faces': np.array([[0, 1, 2], [2, 3, 0]]),
        }
        cls.lmesh = [mesh0, mesh1]

        # traces
        trace00, trace01 = 2.*t0, np.sin(t0)
        trace10, trace11 = np.cos(t1), t1[:, None]*t0
        trace20, trace21 = np.sin(r0), r0[:, None]*r1
        trace30, trace31 = np.cos(r2), t0[:, None]*np.sin(r2)
        trace40 = t2[:, None, None]*r1[None, :, None]*ch0[None, None, :]
        trace41 = t2[None, None, :]*r2[:, None, None]*ch1[None, :, None]
        trace50 = np.cos(t0)[:, None, None]*(
            mesh0['R'][None, :, None]*mesh0['Z'][None, None, :]
        )
        trace51 = t1[:, None]*(mesh1['faces'][:, 0:1]).T
        cls.ltrace = [trace00, trace01, trace10, trace11,
                      trace20, trace21, trace30, trace31,
                      trace40, trace41, trace50, trace51]

        # Anisotropic rect + tri
        BRr = np.cos(t1)[:, None, None]*(
            mesh0['R'][None, :, None]*mesh0['Z'][None, None, :]
        )
        BPhir = np.cos(t1)[:, None, None]*(
            mesh0['R'][None, :, None]*mesh0['Z'][None, None, :]
        )
        BZr = np.cos(t1)[:, None, None]*(
            mesh0['R'][None, :, None]*mesh0['Z'][None, None, :]
        )
        BRt = t0[:, None]*(mesh1['faces'][:, 0:1]).T
        BPhit = t0[:, None]*(mesh1['faces'][:, 0:1]).T
        BZt = t0[:, None]*(mesh1['faces'][:, 0:1]).T
        cls.lB = [BRr, BPhir, BZr, BRt, BPhit, BZt]

        # polygons
        lpoly0 = [np.ones((2, 5)), np.ones((2, 8))]
        lpoly1 = [np.ones((2, 5)), np.ones((2, 8)), np.ones((2, 5))]
        lpoly2 = [np.ones((2, 5)), np.ones((2, 5))]
        cls.lpoly = [lpoly0, lpoly1, lpoly2]

        # spectral lines
        l0 = {
            'key': 'l0', 'lambda0': 5e-10,
            'origin': '[1]', 'transition': 'A->B',
        }
        l1 = {
            'key': 'l1', 'lambda0': 5e-10,
            'origin': '[2]', 'transition': 'B->C',
        }
        l2 = {
            'key': 'l2',
            'data': t0[:, None]*t1[None, :], 'ref': ('t0', 't1'),
            'lambda0': 5e-10, 'origin': '[2]', 'transition': 'B->C'
        }
        cls.llines = [l0, l1, l2]

        # Configs
        # conf0 = tfg.utils.create_config(case='B2')
        # conf1 = tfg.utils.create_config(case='B3')

        dref = {
            't0': {
                'data': cls.lt[0], 'group': 'time', 'units': 's',
                'quant': 'time',
            },
            't1': {
                'data': cls.lt[1], 'group': 'time', 'units': 'min',
                'quant': 'time',
            },
            'r2': {
                'data': cls.lr[2],
                'group': 'radius', 'units': 'm', 'quant': 'rho',
            },
            'mesh0': {
                'data': mesh0,
            },
            'mesh1': {
                'data': mesh1,
            },
        }
        ddata = {
            'trace00': {'data': cls.ltrace[0], 'ref': ('t0',)},
            'trace10': {'data': cls.ltrace[2], 'ref': ('t1',), 'units': 's'},
            'trace11': {'data': cls.ltrace[3], 'ref': ('t1', 't0')},
            'trace30': {'data': cls.ltrace[6], 'ref': ('r2',)},
            'trace31': {'data': cls.ltrace[7], 'ref': ('t0', 'r2')},
            'trace50': {'data': trace50, 'ref': ('t0', 'mesh0')},
            'BRr': {'data': BRr, 'ref': ('t1', 'mesh0'), 'quant': 'BR'},
            'BPhir': {'data': BPhir, 'ref': ('t1', 'mesh0'), 'quant': 'BPhi'},
            'BZr': {'data': BZr, 'ref': ('t1', 'mesh0'), 'quant': 'BZ'},
            'BRt': {'data': BRt, 'ref': ('t0', 'mesh1'), 'quant': 'BR'},
            'BPhit': {'data': BPhit, 'ref': ('t0', 'mesh1'), 'quant': 'BPhi'},
            'BZt': {'data': BZt, 'ref': ('t0', 'mesh1'), 'quant': 'BZ'},
        }
        data = tfd.DataCollection(dref=dref, ddata=ddata, Name=Name)

        # Spectrallines
        dref = {
            't0': {'data': cls.lt[0], 'group': 'time', 'units': 's'},
            't1': {'data': cls.lt[1], 'group': 'time', 'units': 'min'},
        }
        dref_static = {
            'source': {
                '[1]': {'long': 'blabla'},
                '[2]': {'long': 'blibli'},
            },
            'ion': {
                'O3+': {'element': 'O'},
                'Ca6+': {'element': 'Ca'},
            },
        }
        dobj = {
            'lines': {
                'l0': {
                    'lambda0': 3e-10, 'source': '[1]', 'transition': 'A->B',
                },
                'l1': {
                    'lambda0': 4e-10, 'source': '[2]', 'transition': 'B->C',
                },
                'l2': {
                    'data': t0[:, None]*t1[None, :], 'ref': ('t0', 't1'),
                    'lambda0': 5e-10, 'source': '[2]', 'transition': 'B->C',
                },
            }
        }
        sl = tfd.DataCollection()
        sl._data_none = True
        sl.update(dref=dref, dref_static=dref_static, dobj=dobj)

        cls.lobj = [data, sl]

    @classmethod
    def setup(self):
        pass

    def teardown(self):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def test01_init_from_combinations(self):

        # Try with minimalist input (implicit with n = 1)
        dgroup = 'time'
        dref = {'t0': self.lt[0]}
        ddata = {
            'trace00': self.ltrace[0],
            'trace01': {'data': self.ltrace[1], 'units': 'a.u.'},
        }
        data = tfd.DataCollection(
            dgroup=dgroup, dref=dref, ddata=ddata,
            Name='data',
        )

        # Try with minimalist input
        dref = {'t0': {'data': self.lt[0], 'group': 'time'},
                't1': {'data': self.lt[1], 'group': 'time', 'units': 's'},
                'r2': {'data': self.lr[2], 'group': 'radius', 'foo': 'bar'}}
        ddata = {
            'trace00': {'data': self.ltrace[0], 'ref': 't0'},
            'trace10': {'data': self.ltrace[2], 'ref': 't1', 'units': 'a'},
            'trace11': {'data': self.ltrace[3], 'ref': ('t1', 't0')},
            'trace30': {'data': self.ltrace[6], 'ref': ('r2',), 'foo': 'bar'},
            'trace31': {'data': self.ltrace[7], 'ref': ('t0', 'r2')}
        }
        data = tfd.DataCollection(
            dgroup=None, dref=dref, ddata=ddata,
            Name='data',
        )

        # Try with meshes
        dref = {
            't0': {'data': self.lt[0], 'group': 'time', 'units': 's'},
            't1': {'data': self.lt[1], 'group': 'time', 'units': 's'},
            'r2': {'data': self.lr[2], 'group': 'radius', 'foo': 'bar'},
            'mesh1': {'data': self.lmesh[1], 'foo': 'bar', 'quant': 'rho'},
        }
        ddata = {
            'trace10': {'data': self.ltrace[2], 'ref': 't1', 'units': 'a'},
            'trace50': {'data': self.ltrace[-2], 'ref': ('t0', 'mesh0')},
            'trace51': {'data': self.ltrace[-1], 'ref': ('t1', 'mesh1')},
            'mesh0': {'data': self.lmesh[0], 'foo': 'bar', 'group': 'mesh2d'},
        }
        data = tfd.DataCollection(
            dref=dref, ddata=ddata,
            Name='data',
        )

        # Try with lines
        data = tfd.DataCollection()
        data.add_data(**self.llines[0])
        data.add_data(**self.llines[1])
        data.add_ref(key='t0', data=self.lt[0], group='ne')
        data.add_ref(key='t1', data=self.lt[1], group='Te')
        data.add_data(**self.llines[2])

    def test02_wrong_init(self):
        # Try with minimalist input
        dref = {
            't0': {'data': self.lt[0], 'group': 'time'},
            't1': {'data': self.lt[1], 'group': 'time'},
        }
        ddata = {
            'trace00': self.ltrace[0], 'ref': 't0',
            'trace11': self.ltrace[3], 'ref': ('t0', 't1'),
        }
        err = False
        try:
            data = tfd.DataCollection(
                dgroup=None, dref=dref, ddata=ddata,
                Name='data',
            )
        except Exception as er:
            err = True
        assert err, "Exception was not detected properly!"

    def test03_add_remove_refdataobj(self):
        data = self.lobj[0]

        data.add_ref(key='r0', data=self.lr[0], group='radius', foo='bar')
        assert 'r0' in data.dref.keys()

        data.remove_ref(key='t0')
        assert 't0' not in data.dref.keys()
        assert 't0' not in data.ddata.keys()
        assert all([tt not in data.ddata.keys()
                    for tt in ['trace00', 'trace11', 'trace31']])

        data.add_ref('t0', data=self.lt[0], group='time')
        assert 't0' in data.dref.keys()

        # Check ambiguous throws error
        err = False
        try:
            data.add_data(key='trace00', data=self.ltrace[0])
        except Exception:
            err = True
        assert err
        data.add_data('trace00', data=self.ltrace[0], ref=('t0',))
        data.add_data('trace11', data=self.ltrace[3], ref=('t1', 't0'))
        data.add_data('trace31', data=self.ltrace[7], ref=('t0', 'r2'),
                      foo='bar')
        assert all([tt in data.ddata.keys()
                    for tt in ['trace00', 'trace11', 'trace31']])

        # Add/remove mesh
        data.add_ref(key='mesh0', data=self.lmesh[0], group='mesh2d')
        data.add_data(
            key='trace51', data=self.ltrace[-1],
            ref=('t1', 'mesh1'), quant='rho',
        )
        data.add_data(
            key='BRt', data=self.lB[3], ref=('t0', 'mesh1'),
            quant='BR', units='T',
        )
        data.add_data(
            key='BPhit', data=self.lB[4], ref=('t0', 'mesh1'),
            quant='BPhi', units='T',
        )
        data.add_data(
            key='BZt', data=self.lB[5], ref=('t0', 'mesh1'),
            quant='BZ', units='T',
        )

        # Add / remove obj and ref_static
        self.lobj[1].add_ref_static(key='[3]', which='source', long='bloblo')
        self.lobj[1].add_obj(
            which='lines', key='l3',
            lambda0=5e-10, source='[3]', transition='C->D',
        )
        self.lobj[1].remove_obj(key='l3')
        self.lobj[1].remove_ref_static(key='[3]')
        self.lobj[1].remove_ref_static(which='ion')

    def test04_select(self):
        data = self.lobj[0]

        key = data.select(which='data', units='s', returnas=str)
        assert key == ['trace10']

        out = data.select(units='a.u.', returnas=int)
        assert len(out) == 12, out

        # test quantitative param selection
        out = self.lobj[1].select(which='lines', lambda0=[3.5e-10, 6e-10])
        assert len(out) == 2

        out = self.lobj[1].select(which='lines', lambda0=(3.5e-10, 6e-10))
        assert len(out) == 1

    def test05_sortby(self):
        for oo in self.lobj:
            oo.sortby(which='data', param='units')

    def test06_get_summary(self):
        for oo in self.lobj:
            oo.get_summary()

    def test07_getsetaddremove_param(self):
        data = self.lobj[0]

        out = data.get_param('units')
        data.set_param('units', value='T', key='trace00')
        data.add_param('shot', value=np.arange(0, len(data.ddata)))
        assert np.all(
            data.get_param('shot')['shot'] == np.arange(0, len(data.ddata))
        )
        data.remove_param('shot')
        assert 'shot' not in data.get_lparam(which='data')

    def test08_switch_ref(self):

        data = self.lobj[0]
        data.switch_ref('trace00')

        # Check t0 removed
        assert 'trace00' in data.dref.keys()
        assert 'trace00' in data.dgroup['time']['lref']
        assert all(['trace00' in v0['ref'] for k0, v0 in data.ddata.items()
                    if k0 in data.dref['trace00']['ldata']])
        # Check t0 removed
        assert 't0' not in data.dref.keys()
        assert 't0' not in data.dgroup['time']['lref']
        assert all(['t0' not in v0['ref'] for k0, v0 in data.ddata.items()
                    if k0 in data.dref['trace00']['ldata']])
        # .. but still in data
        assert 't0' in data.ddata.keys()

    def test09_convert_spectral(self):
        coef, inv = self.lobj[0].convert_spectral(
            units_in='eV', units_out='J', returnas='coef',
        )
        assert coef == scpct.e and inv is False

        coef, inv = self.lobj[0].convert_spectral(
            units_in='nm', units_out='keV', returnas='coef',
        )
        assert coef == (0.001*(1/scpct.e)*scpct.h*scpct.c / 1.e-9)
        assert inv is True

        data = [[0], [1], [2], [3]]
        out = self.lobj[0].convert_spectral(
            data=data,
            units_in='A', units_out='MHz',
        )
        assert out.shape == (4, 1)

    # ------------------------
    #   Interpolation tools
    # ------------------------

    def test10_check_qr12RPZ(self):
        data = self.lobj[0]
        # Directly get 2d quant
        out = data._check_qr12RPZ(
             quant='mesh0', ref1d=None, ref2d=None,
             q2dR=None, q2dPhi=None, q2dZ=None,
         )
        assert (
            out[0] == 'mesh0'
            and all([out[ii] is None for ii in [1, 2, 3, 4, 5]])
        )
        out = data._check_qr12RPZ(
             quant='trace51', ref1d=None, ref2d=None,
             q2dR=None, q2dPhi=None, q2dZ=None,
         )
        assert (
            out[0] == 'trace51'
            and all([out[ii] is None for ii in [1, 2, 3, 4, 5]])
        )

        # Get 1d quant
        out = data._check_qr12RPZ(
             quant='r2', ref1d='r2', ref2d='trace51',
             q2dR=None, q2dPhi=None, q2dZ=None,
         )
        assert (
            out[0] == 'r2' and out[1] == 'r2' and out[2] == 'trace51'
            and all([out[ii] is None for ii in [3, 4, 5]])
        )

        # Get 1d quant from 2d
        out = data._check_qr12RPZ(
             quant='trace30', ref1d='r2', ref2d='trace51',
             q2dR=None, q2dPhi=None, q2dZ=None,
         )
        assert (
            out[0] == 'trace30' and out[1] == 'r2' and out[2] == 'trace51'
            and all([out[ii] is None for ii in [3, 4, 5]])
        )

        # Get 1d quant from 2d
        out = data._check_qr12RPZ(
             quant='trace30', ref1d=None, ref2d=None,
             q2dR=None, q2dPhi=None, q2dZ=None,
         )
        assert (
            out[0] == 'trace30' and out[1] == 'r2' and out[2] == 'trace51'
            and all([out[ii] is None for ii in [3, 4, 5]])
        )

    def test11_interp_pts2d_to_quant1d(self):
        data = self.lobj[0]

        # Not specifying points
        val, dout = data._interp_pts2d_to_quant1d(quant='trace30')

        # Specifying wrong ref2d
        error = False
        try:
            val, dout = data._interp_pts2d_to_quant1d(
                quant='trace30', ref2d='mesh1',
            )
        except Exception as err:
            error = err
        assert isinstance(error, Exception) and 'Non-valid' in str(error)

        # Specifying points and time
        pts = np.copy(dout['pts'])*0.5
        val, dout = data._interp_pts2d_to_quant1d(
            quant='trace30', pts=pts, t=np.linspace(1, 5, 10),
        )

        # Not specifying points
        val, t = data._interp_pts2d_to_quant1d(quant='trace30')

        # anisotropic rect with time
        pts = data._get_pts_from_mesh(key='mesh0')
        vect = pts
        val, t = data._interp_pts2d_to_quant1d(
            q2dR='BRr', q2dZ='BZr', q2dPhi='BPhir',
            pts=pts, vect=vect, t=np.linspace(2, 5, 10),
        )

        # anisotropic tri
        pts = data._get_pts_from_mesh(key='mesh1')
        vect = pts
        val, t = data._interp_pts2d_to_quant1d(
            q2dR='BRt', q2dZ='BZt', q2dPhi='BPhit',
            pts=pts, vect=vect,
        )

    # ------------------------
    #   Generic TofuObject methods
    # ------------------------

    def test20_copy_equal(self):
        for oo in self.lobj:
            obj = oo.copy()
            assert obj == oo

    def test21_get_nbytes(self):
        for oo in self.lobj:
            nb, dnb = oo.get_nbytes()

    def test22_strip_nbytes(self, verb=False):
        lok = self.lobj[0].__class__._dstrip['allowed']
        nb = np.full((len(lok),), np.nan)
        for oo in self.lobj:
            for ii in lok:
                oo.strip(ii, verb=verb)
                nb[ii] = oo.get_nbytes()[0]
            assert np.all(np.diff(nb) <= 0.), nb
            for ii in lok[::-1]:
                oo.strip(ii, verb=verb)

    def test23_saveload(self, verb=False):
        for oo in self.lobj:
            if oo.Id.Name is None:
                try:
                    pfe = oo.save(deep=False, verb=verb, return_pfe=True)
                except Exception as err:
                    pass
            else:
                pfe = oo.save(deep=False, verb=verb, return_pfe=True)
                obj = tfu.load(pfe, verb=verb)
                # Just to check the loaded version works fine
                assert oo == obj
                os.remove(pfe)


# #############################################################################
# #############################################################################
#           Specific to SpectralLines
# #############################################################################


class Test02_SpectralLines(object):

    @classmethod
    def setup_class(cls, Name='data1',  SavePath='./', verb=False):
        cls.sl = tfd.SpectralLines.from_openadas(
            lambmin=3.94e-10,
            lambmax=4e-10,
            element=['Ar', 'W'],
        )

    @classmethod
    def setup(self):
        pass

    def teardown(self):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def test01_add_from_openadas(self):
        lines = self.sl.dobj['lines']
        self.sl.add_from_openadas(
            lambmin=3.90e-10,
            lambmax=3.96e-10,
            element='W',
        )
        assert all([k0 in self.sl.dobj['lines'].keys() for k0 in lines.keys()])

    def test02_sortby(self):
        self.sl.sortby(param='lambda0', which='lines')
        self.sl.sortby(param='ion', which='lines')

    def test03_convert_lines(self):
        self.sl.convert_lines(units='Hz')

    def test04_calc_pec(self):
        ne = np.r_[1e15, 1e18, 1e21]
        Te = np.r_[1e3, 2e3, 3e3, 4e3, 5e3]
        dpec = self.sl.calc_pec(ne=ne, Te=Te[:ne.size], grid=False)
        dpec = self.sl.calc_pec(
            key='Ar16_9_oa_pec40_cl', ne=ne, Te=Te[:ne.size], grid=False,
        )
        dpec = self.sl.calc_pec(ne=ne, Te=Te, grid=True)
        dpec = self.sl.calc_pec(
            key='Ar16_9_oa_pec40_cl', ne=ne, Te=Te[:ne.size], grid=False,
        )

    def test05_calc_intensity(self):
        ne = np.r_[1e15, 1e18, 1e21]
        Te = np.r_[1e3, 2e3, 3e3, 4e3, 5e3]

        concentration = np.r_[0.1, 0.2, 0.3]
        dint = self.sl.calc_intensity(
            ne=ne, Te=Te[:ne.size], concentration=concentration, grid=False,
        )

        key = ['Ar16_9_oa_pec40_cl']
        concentration = {k0: np.r_[0.1, 0.2, 0.3] for k0 in key}
        dint = self.sl.calc_intensity(
            key=key,
            ne=ne, Te=Te[:ne.size], concentration=concentration, grid=False,
        )

        concentration = np.random.random((ne.size, Te.size))
        dint = self.sl.calc_intensity(
            ne=ne, Te=Te, concentration=concentration, grid=True,
        )

        key = ['Ar16_9_oa_pec40_cl']
        concentration = {k0: concentration for k0 in key}
        dint = self.sl.calc_intensity(
            key=key,
            ne=ne, Te=Te, concentration=concentration, grid=True,
        )

    def test06_plot(self):
        ax = self.sl.plot()
        plt.close('all')

    def test07_plot_pec_single(self):
        Te = 1.e3
        ne = 1.e20
        ax = self.sl.plot_pec_single(Te=Te, ne=ne)

    def test08_plot_pec(self):
        Te = np.linspace(1, 7, 7)*1e3
        ne = np.logspace(15, 21, 7)
        ax = self.sl.plot_pec(Te=1e3, ne=ne)
        ax = self.sl.plot_pec(Te=Te, ne=1e19)
        ax = self.sl.plot_pec(Te=Te, ne=ne)
        plt.close('all')
