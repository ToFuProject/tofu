"""
This module contains tests for tofu.geom in its structured version
"""



# External modules
import os
import numpy as np
import matplotlib.pyplot as plt


_here = os.path.abspath(os.path.dirname(__file__))
VerbHead = 'tofu.test_03_plot'
keyVers = 'Vers'
_Exp = 'WEST'


#######################################################
#
#     Setup and Teardown
#
#######################################################

def setup_module(module):
    from tofu import __version__
    print("") # this is to get a newline after the dots
    lf = os.listdir(_here)
    lf = [f for f in lf
         if all([s in f for s in ['TFG_',_Exp,'.npz']])]
    lF = []
    for f in lf:
        ff = f.split('_')
        v = [fff[len(keyVers):] for fff in ff
             if fff[:len(keyVers)]==keyVers]
        msg = f + "\n    "+str(ff) + "\n    " + str(v)
        assert len(v)==1, msg
        v = v[0]
        if '.npz' in v:
            v = v[:v.index('.npz')]
        # print(v, __version__)
        if v!=__version__:
            lF.append(f)
    if len(lF)>0:
        print("Removing the following previous test files:")
        for f in lF:
            os.remove(os.path.join(_here,f))
        #print("setup_module before anything in this file")

def teardown_module(module):
    from tofu import __version__
    #os.remove(VesTor.Id.SavePath + VesTor.Id.SaveName + '.npz')
    #os.remove(VesLin.Id.SavePath + VesLin.Id.SaveName + '.npz')
    #print("teardown_module after everything in this file")
    #print("") # this is to get a newline
    lf = os.listdir(_here)
    lf = [f for f in lf
         if all([s in f for s in ['TFG_',_Exp,'.npz']])]
    lF = []
    for f in lf:
        ff = f.split('_')
        v = [fff[len(keyVers):] for fff in ff
             if fff[:len(keyVers)]==keyVers]
        msg = f + "\n    "+str(ff) + "\n    " + str(v)
        assert len(v)==1, msg
        v = v[0]
        if '.npz' in v:
            v = v[:v.index('.npz')]
        # print(v, __version__)
        if v==__version__:
            lF.append(f)
    if len(lF)>0:
        print("Removing the following test files:")
        for f in lF:
            os.remove(os.path.join(_here,f))



#######################################################
#
#   Struct subclasses
#
#######################################################


class Test01_plot_shotovervew(object):

    @classmethod
    def setup_class(cls):
        #print("")
        #print("---- "+cls.__name__)

        # conf
        import tofu as tf
        cls.conf = tf.geom.utils.create_config('B3')

        # time vectors
        t0 = np.linspace(0,10,100)
        teq0 = t0 + 0.1
        t1 = np.linspace(t0[0],t0[-1]+1, t0.size//2)
        t2 = np.linspace(t0[0]-1.,t0[-1]-1., 2*t0.size)
        teq2 = t2 - 0.1

        Ax0 = np.array([2.4+0.1*np.cos(teq0), 0.1*np.sin(teq0)]).T
        Ax2 = np.array([2.4+0.1*np.sin(teq2), 0.05*np.cos(teq2)]).T
        Sep0 =  (Ax0[:,:,np.newaxis]
                 + 0.4*np.array([[-1,1,1,-1],[-1,-1,1,1]])[None,:,:])
        Sep2 =  (Ax2[:,:,np.newaxis]
                 + 0.4*np.array([[-1,1,1,-1],[-1,-1,1,1]])[None,:,:])


        # dextra
        dextra0 = {'pouet':{'t':t0, 'c':'k', 'data':np.sin(t0),
                            'units':'a.u.', 'label':'pouet0'},
                   'Ax':{'t':teq0, 'data2D':Ax0},
                   'Sep':{'t':teq0,'data2D':Sep0}}
        dextra1 = {'pouet':{'t':t1, 'c':'k', 'data':np.cos(t1),
                            'units':'a.u.', 'label':'pouet1'}}
        dextra2 = {'Ax':{'t':teq2, 'data2D':Ax2},
                   'Sep':{'t':teq2,'data2D':Sep2}}

        cls.dobj = {0:dextra0, 1:dextra1, 2:dextra2}

    @classmethod
    def teardown_class(cls):
        #print("teardown_class() after any methods in this class")
        pass

    def setup_method(self):
        #print("TestUM:setup() before each test method")
        pass

    def teardown_method(self):
        #print("TestUM:teardown_method() after each test method")
        pass

    def test01_plot_shotoverview(self):
        # One by one, without conf
        import tofu as tf
        for shot, dextra in self.dobj.items():
            _ = tf._plot.plot_shotoverview({shot:dextra})

        # All together, without conf
        _ = tf._plot.plot_shotoverview(self.dobj)
        plt.close('all')

        # One by one, with conf
        for shot, dextra in self.dobj.items():
            _ = tf._plot.plot_shotoverview({shot:dextra}, config=self.conf)

        # All together, with conf
        _ = tf._plot.plot_shotoverview(self.dobj, config=self.conf)
        plt.close('all')