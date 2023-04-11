import os
import shutil
import subprocess
import matplotlib.pyplot as plt
from tofu import __version__
from pathlib import Path

_HERE = os.path.abspath(os.path.dirname(__file__))
_TFROOT = Path(_HERE).parent.parent.parent

keyVers = 'Vers'


#######################################################
#
#     Setup and Teardown
#
#######################################################


def setup_module(module, here=_HERE):
    print("")   # this is to get a newline after the dots
    lf = os.listdir(here)
    lf = [f for f in lf
          if all([s in f for s in ['.npz']])]
    lF = []
    for f in lf:
        ff = f.split('_')
        v = [fff[len(keyVers):] for fff in ff
             if fff[:len(keyVers)] == keyVers]
        msg = f + "\n    " + str(ff) + "\n    " + str(v)
        assert len(v) == 1, msg
        v = v[0]
        if '.npz' in v:
            v = v[:v.index('.npz')]
        # print(v, __version__)
        if v != __version__:
            lF.append(f)
    if len(lF) > 0:
        print("Removing the following previous test files:")
        for f in lF:
            os.remove(os.path.join(here, f))
        # print("setup_module before anything in this file")


def teardown_module(module, here=_HERE):
    # os.remove(VesTor.Id.SavePath + VesTor.Id.SaveName + '.npz')
    # os.remove(VesLin.Id.SavePath + VesLin.Id.SaveName + '.npz')
    # print("teardown_module after everything in this file")
    # print("") # this is to get a newline
    lf = os.listdir(here)
    lf = [f for f in lf
          if all([s in f for s in ['.npz']])]
    lF = []
    for f in lf:
        ff = f.split('_')
        v = [fff[len(keyVers):] for fff in ff
             if fff[:len(keyVers)] == keyVers]
        msg = f + "\n    "+str(ff) + "\n    " + str(v)
        assert len(v) == 1, msg
        v = v[0]
        if '.npz' in v:
            v = v[:v.index('.npz')]
        # print(v, __version__)
        if v == __version__:
            lF.append(f)
    if len(lF) > 0:
        print("Removing the following test files:")
        for f in lF:
            os.remove(os.path.join(here, f))


#######################################################
#
#   Struct subclasses
#
#######################################################

def get_list_toturials(path=_HERE):
    ltut = os.listdir(path)
    ltut = [tt[:-3] for tt in ltut
            if (all([ss in tt for ss in ['tuto', '.py']])
                and not any([ss in tt for ss in ['__init__', '.swp']]))]
    return ltut


class Test00_tuto(object):

    @classmethod
    def setup_class(cls, here=_HERE, pathtuto=_HERE, root=_TFROOT):
        # print("")
        # print("---- "+cls.__name__)
        # ii = 0
        # for tuto in get_list_toturials(path=pathtuto):
        # # Create local temporary copy to make sure the
        # local version of tofu is imported
        # method = 'test{0:02.0f}_{1}'.format(ii, tuto)
        # fmethod = lambda tuto=tuto: cls.Test_tuto._test_tuto(tuto)
        # setattr(cls, method, types.MethodType(fmethod, cls))
        # # setattr(cls, method, lambda cls: cls._test_tuto(tuto))
        # ii += 1
        pass

    @classmethod
    def teardown_class(cls):
        # print("teardown_class() after any methods in this class")
        pass

    def setup_method(self):
        # print("TestUM:setup_method() before each test method")
        # self.setup_class()
        pass

    def teardown_method(self):
        # print("TestUM:teardown_method() after each test method")
        pass

    @classmethod
    def _test_tuto(cls, tuto=None, pathtuto=_HERE, root=_TFROOT):
        src = os.path.join(pathtuto, tuto + '.py')
        target = os.path.join(root, tuto + '.py')
        shutil.copyfile(src, target)
        error = None
        try:
            cmd = 'python ' + target
            out = subprocess.run(cmd, shell=True, check=False,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
            ls = out.stderr.decode().split('\n')
            if any(['Error' in ss and not 'ModuleNotFoundError' in ss
                    for ss in ls]):
                error = '\n'.join(ls)
        except Exception as err:
            error = err

        if error is not None:
            msg = str(error)
            raise Exception(msg)
        plt.close('all')

        # Remove temporary files and saved files
        os.remove(target)
        stdout = out.stdout.decode().split('\n')
        lf = [
            stdout[ii].strip() for ii in range(len(stdout))
            if 'saved' in stdout[ii-1].lower()
        ]

        for ii in range(len(lf)):
            os.remove(lf[ii])

    def test01_plot_basic_tutorial(self):
        self._test_tuto('tuto_plot_basic')

    def test02_plot_create_geometry(self):
        self._test_tuto('tuto_plot_create_geometry')

    def test03_plot_custom_emissivity(self):
        self._test_tuto('tuto_plot_custom_emissivity')
