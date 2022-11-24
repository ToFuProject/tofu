"""
This module contains tests for tofu.geom in its structured version
"""

# External modules
import os
import itertools as itt
import numpy as np
import matplotlib.pyplot as plt
import warnings as warn

# Importing package tofu.gem
import tofu as tf
from tofu import __version__
import tofu.defaults as tfd
import tofu.utils as tfu
import tofu.geom as tfg

import tofu.spectro._rockingcurve_def as _rockingcurve_def

_here = os.path.abspath(os.path.dirname(__file__))
_PATH_DATA = os.path.join(_here, 'test_data')
_PFE_DET = os.path.join(_PATH_DATA, 'det37_CTVD_incC4_New.npz')

VerbHead = 'tofu.geom.test_04_core_optics'
keyVers = 'Vers'
_Exp = 'WEST'




#######################################################
#
#     Setup and Teardown
#
#######################################################

def setup_module(module):
    print("")   # this is to get a newline after the dots
    lf = os.listdir(_here)
    lf = [
        ff for ff in lf
        if all([s in ff for s in ['TFG_', _Exp, '.npz']])
    ]
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
        if v != __version__:
            lF.append(f)
    if len(lF) > 0:
        print("Removing the following previous test files:")
        for f in lF:
            os.remove(os.path.join(_here, f))
        # print("setup_module before anything in this file")


def teardown_module(module):
    # os.remove(VesTor.Id.SavePath + VesTor.Id.SaveName + '.npz')
    # os.remove(VesLin.Id.SavePath + VesLin.Id.SaveName + '.npz')
    # print("teardown_module after everything in this file")
    # print("") # this is to get a newline
    lf = os.listdir(_here)
    lf = [
        ff for ff in lf
        if all([s in ff for s in ['TFG_', _Exp, '.npz']])
    ]
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
            os.remove(os.path.join(_here, f))


#######################################################
#
#   Crystal class
#
#######################################################

class Test01_Crystal(object):

    @classmethod
    def setup_class(cls, verb=False):
        # print ("")
        # print "--------- "+VerbHead+cls.__name__

        # Prepare input
        dgeom = {
            'Type': 'sph',
            'Typeoutline': 'rect',
            'summit': np.array([4.6497750e-01, -8.8277925e+00, 3.5125000e-03]),
            'center': np.array([1.560921, -6.31106476, 0.00729429]),
            'extenthalf': np.array([0.01457195, 0.01821494]),
            'rcurve': 2.745,
            'move': 'rotate_around_3daxis',
            'move_param': 0.022889993139905633,
            'move_kwdargs': {
                'axis': np.array([
                    [4.95e-01, -8.95e+00, -8.63e-02],
                    [-1.37e-04, -2.18e-03,  9.99e-01],
                ])
            }
        }
        dmat = {
            'formula': 'Quartz',
            'density': 2.6576,
            'symmetry': 'hexagonal',
            'lengths': np.array([4.9079e-10, 4.9079e-10, 5.3991e-10]),
            'angles': np.array([1.57079633, 1.57079633, 2.0943951]),
            'cut': np.array([1,  1, -2,  0]),
            'd': 2.4539499999999996e-10,
        }
        dbragg = {
            'lambref': 3.96e-10,
        }

        dmat1 = dict(dmat)
        dmat2 = dict(dmat)
        dmat3 = dict(dmat)

        # cryst1
        cryst1 = tfg.CrystalBragg(
            dgeom=dgeom,
            dmat=dmat1,
            dbragg=dbragg,
            Name='Cryst1',
            Diag='SpectrX2D',
            Exp='WEST',
        )

        # cryst2
        dmat2['alpha'] = 0.
        dmat2['beta'] = 0.
        cryst2 = tfg.CrystalBragg(
            dgeom=dgeom,
            dmat=dmat2,
            dbragg=dbragg,
            Name='Cryst2',
            Diag='SpectrX2D',
            Exp='WEST',
        )

        # cryst3
        dmat3['alpha'] = (3/60)*np.pi/180
        dmat3['beta'] = 0.
        cryst3 = tfg.CrystalBragg(
            dgeom=dgeom,
            dmat=dmat3,
            dbragg=dbragg,
            Name='Cryst3',
            Diag='SpectrX2D',
            Exp='WEST',
        )

        # cryst4
        dmat['alpha'] = (3/60)*np.pi/180
        dmat['beta'] = np.pi/1000.
        cryst4 = tfg.CrystalBragg(
            dgeom=dgeom,
            dmat=dmat,
            dbragg=dbragg,
            Name='Cryst4',
            Diag='SpectrX2D',
            Exp='WEST',
        )

        cls.dobj = {
            'cryst1': cryst1,
            'cryst2': cryst2,
            'cryst3': cryst3,
            'cryst4': cryst4,
        }

        cls.xi = 0.05*np.linspace(-1, 1, 100)
        cls.xj = 0.10*np.linspace(-1, 1, 200)

    @classmethod
    def teardown_class(cls):
        pass

    def setup_method(self):
        # print ("TestUM:setup_method() before each test method")
        pass

    def teardown_method(self):
        # print ("TestUM:teardown_method() after each test method")
        pass

    # def test00_todo(self):
        # pass

    def test01_todict(self):
        for k0 in self.dobj.keys():
            dd = self.dobj[k0].to_dict()
            assert type(dd) is dict

    def test02_fromdict(self):
        for k0 in self.dobj.keys():
            dd = self.dobj[k0].to_dict()
            obj = tfg.CrystalBragg(fromdict=dd)
            assert isinstance(obj, self.dobj[k0].__class__)

    def test03_copy_equal(self):
        for k0 in self.dobj.keys():
            obj = self.dobj[k0].copy()
            assert obj == self.dobj[k0]
            assert not obj != self.dobj[k0]

    def test04_get_nbytes(self):
        for k0 in self.dobj.keys():
            nb, dnb = self.dobj[k0].get_nbytes()

    def test05_strip_nbytes(self, verb=False):
        lok = tfg.CrystalBragg._dstrip['allowed']
        nb = np.full((len(lok),), np.nan)
        for k0, obj in self.dobj.items():
            for ii in lok:
                obj.strip(ii)
                nb[ii] = obj.get_nbytes()[0]
            assert np.all(np.diff(nb) <= 0.)
            for ii in lok[::-1]:
                obj.strip(ii)

    def test06_set_move_None(self):
        pass

    def test07_rotate_copy(self):
        pass

    def test08_get_detector_ideal(self):
        for k0, obj in self.dobj.items():
            det0 = obj.get_detector_ideal(miscut=False)
            det1 = obj.get_detector_ideal(miscut=True)
            assert isinstance(det0, dict) and isinstance(det0, dict)
            lk = ['nout', 'ei']
            assert all([ss in det0.keys() for ss in lk])
            assert all([ss in det1.keys() for ss in lk])
            if k0 in ['cryst1', 'cryst2']:
                assert all([
                    np.allclose(det0[kk], det1[kk])
                    for kk in lk
                ])
            elif k0 in ['cryst3', 'cryst4']:
                assert not any([
                    np.allclose(det0[kk], det1[kk])
                    for kk in lk
                ])
                for k1, v1 in det0.items():
                    assert np.linalg.norm(v1 - det1[k1]) <= 0.01

    def test09_plot(self):
        ii = 0
        for k0, obj in self.dobj.items():
            det = obj.get_detector_ideal()
            det['outline'] = np.array([
                0.1*np.r_[-1, 1, 1, -1, -1],
                0.1*np.r_[-1, -1, 1, 1, -1],
            ])
            pts, vect = obj.get_rays_from_cryst(
                phi=np.pi, returnas='(pts, vect)',
            )
            dist = obj.get_rowland_dist_from_lambbragg()
            pts = pts + dist*np.r_[0.5, 1., 2][None, :]*vect[:, 0:1, 0]
            lamb = obj.dbragg['lambref'] + np.r_[-1, 0, 1, 2]*1-12
            dax = obj.plot(
                pts=pts,
                lamb=lamb,
                det=det,
                rays_color='pts' if ii % 2 == 0 else 'lamb',
            )
            ii += 1
        plt.close('all')

    def test10_get_lamb_avail_from_pts(self):
        for k0, obj in self.dobj.items():
            det = obj.get_detector_ideal()
            pts, vect = obj.get_rays_from_cryst(
                phi=-9*np.pi/10., returnas='(pts, vect)',
            )
            dist = obj.get_rowland_dist_from_lambbragg()
            pts = pts + dist*np.r_[0.5, 1., 2][None, :]*vect[:, :, 0]
            lamb, phi, dtheta, psi, xi, xj = obj.get_lamb_avail_from_pts(
                pts=pts, det=det,
            )
            pts = pts + np.r_[7.5][None, :]*vect[:, :, 0]
            lamb, phi, dtheta, psi, xi, xj = obj.get_lamb_avail_from_pts(
                pts=pts, det=det,
            )
            conf = tf.load_config('WEST-V0')
            pts, dv, ind, res_eff = conf.Ves.V1.get_sampleV(
                res=0.3,
                domain=[None, None, [-np.pi, -np.pi/2.]],
            )
            lamb, phi, dtheta, psi, xi, xj = obj.get_lamb_avail_from_pts(
                pts=pts, det=det, strict=True,
            )

    def test11_calc_johann_error(self):
        for k0, obj in self.dobj.items():
            det = obj.get_detector_ideal()
            err_lamb, err_phi, _, _, _ = obj.calc_johannerror(
                xi=self.xi,
                xj=self.xj,
                det=det,
            )

    def test12_plot_line_on_det_tracing(self):
        for k0, obj in self.dobj.items():
            det = obj.get_detector_ideal()
            det['outline'] = np.array([
                0.1*np.r_[-1, 1, 1, -1, -1],
                0.1*np.r_[-1, -1, 1, 1, -1],
            ])
            dax = obj.plot_line_on_det_tracing(
                det=det,
                crystal='Quartz_110',
                merge_rc_data=True,
                miscut=False,
                therm_exp=False,
                plot=False,
            )

    def test13_calc_meridional_sagittal_focus(self):
        derr = {}
        for k0, obj in self.dobj.items():
            out = obj.calc_meridional_sagittal_focus(
                miscut=False,
                verb=False,
            )
            c0 = round(out[0], ndigits=12) == round(out[2], ndigits=12)
            c1 = round(out[1], ndigits=12) == round(out[3], ndigits=12)
            if not c0:
                derr[k0] = f'Meridional ({out[0]} vs {out[2]})'
                if not c1:
                    derr[k0] += f' + Sagittal ({out[1]} vs {out[3]})'
                derr[k0] += ' focus wrong'
            elif not c1:
                derr[k0] = f'Sagittal ({out[1]} vs {out[3]}) focus wrong'

            if obj.dmat['alpha'] != 0.0:
                out = obj.calc_meridional_sagittal_focus(
                    miscut=True,
                    verb=False,
                )
                c0 = round(out[0], ndigits=12) != round(out[2], ndigits=12)
                c1 = round(out[1], ndigits=12) != round(out[3], ndigits=12)
                if not c0:
                    derr[k0] = f'Meridional ({out[0]} vs {out[2]})'
                    if not c1:
                        derr[k0] += f' + Sagittal ({out[1]} vs {out[3]})'
                    derr[k0] += ' focus wrong'
                elif not c1:
                    derr[k0] = f'Sagittal ({out[1]} vs {out[3]}) focus wrong'
        if len(derr) > 0:
            lstr = [f'\t- {k0}: {v0}' for k0, v0 in derr.items()]
            msg = (
                "The following crystals have wrong focus:\n"
                + "\n".join(lstr)
            )
            raise Exception(msg)

    def test14_plot_focal_error_summed(self):
        det = dict(np.load(
            os.path.join(_PATH_DATA, 'det37_CTVD_incC4_New.npz'),
            allow_pickle=True,
            ))
        for k0, obj in self.dobj.items():
            out = obj.plot_focal_error_summed(
                dist_min=-0.02, dist_max=0.02, ndist=5,
                di_min=-0.02, di_max=0.02, ndi=5,
                xi=self.xi[::20], xj=self.xj[::20],
                miscut=False,
                det_ref=det,
                plot_dets=True,
            )
        plt.close('all')

    def test15_split(self):
        for ii, (k0, obj) in enumerate(self.dobj.items()):
            direction = None if ii == 0 else ('e1' if ii % 2 == 0 else 'e2')
            nb = None if ii == 0 else (2 if ii % 2 == 0 else 3)
            lcryst = obj.split(direction=direction, nb=nb)

    def test16_get_plasmadomain_at_lamb(self):

        # load useful objects
        conf0 = tf.load_config('WEST-V0')
        det = dict(np.load(_PFE_DET, allow_pickle=True))

        # test all crystals
        for ii, (k0, obj) in enumerate(self.dobj.items()):
            if ii % 2 == 0:
                plot_as = 'poly'
                xixj_lim = None
                domain = [None, None, [-4*np.pi/5., -np.pi/2.]]
                res = 0.05
            else:
                plot_as = 'pts'
                xixj_lim = [[-0.04, 0.04], 172.e-4*np.r_[-0.5, 0.5]]
                domain = [None, [-0.1, 0.1], [-4*np.pi/5., -np.pi/2.]]
                res = 0.02

            pts, lambok, dax = obj.get_plasmadomain_at_lamb(
                det=det,
                lamb=[3.94e-10, 4.e-10],
                res=res,
                config=conf0,
                domain=domain,
                xixj_lim=xixj_lim,
                plot_as=plot_as,
            )
        plt.close('all')

    def test17_saveload(self, verb=False):
        for k0, obj in self.dobj.items():
            obj.strip(-1)
            pfe = obj.save(verb=verb, return_pfe=True)
            obj2 = tf.load(pfe, verb=verb)
            msg = "Unequal saved / loaded objects !"
            assert obj2 == obj, msg
            # Just to check the loaded version works fine
            obj2.strip(0)
            os.remove(pfe)
