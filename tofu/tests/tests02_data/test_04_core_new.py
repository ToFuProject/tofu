
"""
This module contains tests for tofu.geom in its structured version
"""

# Built-in
import os
import warnings

# Standard
import numpy as np
import matplotlib.pyplot as plt

# Nose-specific
from nose import with_setup # optional

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
    print("") # this is to get a newline after the dots
    LF = os.listdir(_here)
    LF = [lf for lf in LF if all([ss in lf for ss in ['TFD_','Test','.npz']])]
    LF = [lf for lf in LF if not lf[lf.index('_Vv')+2:lf.index('_U')]==__version__]
    print("Removing the following previous test files:")
    print (LF)
    for lf in LF:
        os.remove(os.path.join(_here,lf))
    #print("setup_module before anything in this file")

def teardown_module(module):
    #os.remove(VesTor.Id.SavePath + VesTor.Id.SaveName + '.npz')
    #os.remove(VesLin.Id.SavePath + VesLin.Id.SaveName + '.npz')
    #print("teardown_module after everything in this file")
    #print("") # this is to get a newline
    LF = os.listdir(_here)
    LF = [lf for lf in LF if all([ss in lf for ss in ['TFD_','Test','.npz']])]
    LF = [lf for lf in LF if lf[lf.index('_Vv')+2:lf.index('_U')]==__version__]
    print("Removing the following test files:")
    print (LF)
    for lf in LF:
        os.remove(os.path.join(_here,lf))
    pass


#def my_setup_function():
#    print ("my_setup_function")

#def my_teardown_function():
#    print ("my_teardown_function")

#@with_setup(my_setup_function, my_teardown_function)
#def test_numbers_3_4():
#    print 'test_numbers_3_4  <============================ actual test code'
#    assert multiply(3,4) == 12

#@with_setup(my_setup_function, my_teardown_function)
#def test_strings_a_3():
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
        r0 = np.linspace(0,1,10)
        r1 = np.linspace(0,1,50)
        r2 = np.linspace(0,1,200)
        cls.lr = [r0, r1, r2]

        # traces
        trace00, trace01 = np.cos(t0), np.sin(t2)
        trace10, trace11 = np.cos(t1), t1[:,None]*t0
        trace20, trace21 = np.sin(r0), r0[:,None]*r1
        trace30, trace31 = np.cos(r2), t0[:,None]*np.sin(r2)
        trace40, trace41 = t2[:,None]*r1, t2[:,None]*r2
        cls.ltrace = [trace00, trace01, trace10, trace11,
                      trace20, trace21, trace30, trace31,
                      trace40, trace41]

        # Configs
        # conf0 = tfg.utils.create_config(case='B2')
        # conf1 = tfg.utils.create_config(case='B3')

        dref = {'t0': {'data': cls.lt[0], 'group': 'time', 'units': 's'},
                't1': {'data': cls.lt[1], 'group': 'time', 'units': 'min'},
                'r2': {'data': cls.lr[2], 'group': 'time', 'units': 'm'}}
        ddata = {'trace00': {'data': cls.ltrace[0], 'refs': ('t0',)},
                 'trace10': {'data': cls.ltrace[2], 'refs': ('t1',)},
                 'trace11': {'data': cls.ltrace[3], 'refs': ('t1','t0')},
                 'trace30': {'data': cls.ltrace[6], 'refs': ('r2',)},
                 'trace31': {'data': cls.ltrace[7], 'refs': ('t0','r2')}}
        data = tfd.DataCollection(dref=dref, ddata=ddata, Name=Name)
        cls.lobj = [data]


    @classmethod
    def setup(self):
        pass
    def teardown(self):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def test01_init_from_minimalist(self):

        # Try with minimalist input
        dref = {'time':{'t0': self.lt[0], 't1': self.lt[1]},
                'radius': {'r2': self.lr[2]}}
        ddata = {'trace00': self.ltrace[0], 'trace10': self.ltrace[2],
                 'trace11': self.ltrace[3],
                 'trace30': self.ltrace[6], 'trace31': self.ltrace[7]}
        data2 = tfd.DataCollection(dref=dref, ddata=ddata, Name='data2')

        print(self.lobj[0] == data2)

    def tests02_add_remove_refdata(self):
        data = self.lobj[0]

        data.add_ref('r0', self.lr[0], group='radius')
        assert 'r0' in data.dref.keys()

        data.remove_ref('t0')
        assert 't0' not in data.dref.keys()
        assert 't0' not in data.ddata.keys()
        assert all([tt not in data.ddata.keys()
                    for tt in ['trace00', 'trace11', 'trace31']])

        data.add_ref('t0', self.lt[0], group='time')
        assert 't0' in data.dref.keys()

        # Check ambiguous throws error
        err = False
        try:
            data.add_data('trace00', self.ltrace[0])
        except Exception:
            err = True
        assert err
        data.add_data('trace00', self.ltrace[0], refs=('t0',))
        data.add_data('trace11', self.ltrace[2])
        data.add_data('trace31', self.ltrace[7], refs=('t0','r2'))
        assert all([tt in data.ddata.keys()
                    for tt in ['trace00', 'trace11', 'trace31']])

    def tests03_select(self):
        data = self.lobj[0]

        key = data.select(units='min', returnas=str)
        assert len(key) == 1 and key[0] == 't1'

        out = data.select(units='a.u.', returnas=int)
        assert len(out) == 7, out

    def tests04_get_summary(self):
        data = self.lobj[0]
        data.get_summary()

    def tests05_getsetaddremove_param(self):
        data = self.lobj[0]

        out = data.get_param('units')
        data.set_param('units', 'T', key='trace00')
        data.add_param('shot', np.arange(0,len(data.ldata)))
        assert np.all(data.get_param('shot') == np.arange(0,len(data.ldata)))
        data.remove_param('shot')
        assert 'shot' not in data.lparam



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
            assert np.all(np.diff(nb)<=0.), nb
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






class Test02_TimeTraceCollection(Test01_DataCollection):

    @classmethod
    def setup_class(cls, Name=None,  SavePath='./', verb=False):
        super(Test02_TimeTraceCollection, cls).setup_class(Name=Name,
                                                           SavePath=SavePath,
                                                           verb=verb)
