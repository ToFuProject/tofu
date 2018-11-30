# coding: utf-8
from tofu_LauraBenchmarck_load_config import *
import tofu.geom._GG_LM as _GG
import time
#import line_profiler
import pstats, cProfile


def prepare_inputs(vcam, config, method='ref'):

    D, u, loscam = get_Du(vcam)
    u = u/np.sqrt(np.sum(u**2,axis=0))[np.newaxis,:]
    D = np.ascontiguousarray(D)
    u = np.ascontiguousarray(u)

    # Get reference
    lS = config.lStruct

    lSIn = [ss for ss in lS if ss._InOut=='in']
    if len(lSIn)==0:
        msg = "self.config must have at least a StructIn subclass !"
        assert len(lSIn)>0, msg
    elif len(lSIn)>1:
        S = lSIn[np.argmin([ss.dgeom['Surf'] for ss in lSIn])]
    else:
        S = lSIn[0]

    VPoly = S.Poly_closed
    VVIn =  S.dgeom['VIn']
    Lim = S.Lim
    nLim = S.nLim
    VType = config.Id.Type

    lS = [ss for ss in lS if ss._InOut=='out']
    lSPoly, lSVIn, lSLim, lSnLim = [], [], [], []
    for ss in lS:
        lSPoly.append(ss.Poly_closed)
        lSVIn.append(ss.dgeom['VIn'])
        lSLim.append(ss.Lim)
        lSnLim.append(ss.nLim)

    largs = [D, u, VPoly, VVIn]
    dkwd = dict(Lim=Lim, nLim=nLim,
                LSPoly=lSPoly, LSLim=lSLim,
                lSnLim=lSnLim, LSVIn=lSVIn, VType=VType,
                RMin=None, Forbid=True, EpsUz=1.e-6, EpsVz=1.e-9,
                EpsA=1.e-9, EpsB=1.e-9, EpsPlane=1.e-9, Test=True)

    return largs, dkwd


def test_LOS_west_configs(config="B2", cams=["V1000"], plot=False, save=False, saveCam=[]):
    dconf = load_config(config, plot=plot)
    if plot:
        plt.show(block=True)
    times = []
    for vcam in cams:
        largs, dkwd = prepare_inputs(vcam, dconf)
        start = time.time()
        out = _GG.LOS_Calc_PInOut_VesStruct(*largs, **dkwd)
        elapsed = time.time() - start
        if save and vcam in saveCam :
            np.savez("out_kin_"     +config+"_"+vcam+".npz", out[0])
            np.savez("out_kout_"    +config+"_"+vcam+".npz", out[1])
            np.savez("out_vperpout_"+config+"_"+vcam+".npz", out[2])
            np.savez("out_iout_"    +config+"_"+vcam+".npz", out[3])
        times.append(elapsed)
    return times




def test_LOS_compact(save=False, saveCam=[]):
    Cams = ["V1", "V10", "V100", "V1000", "V10000",
            "V100000"]#, "V1000000"]
    configs = ["A1", "A2", "A3", "B1", "B2", "B3"]
    for icon in configs :
        print("*..................................*")
        print("*      Testing the "+icon+" config       *")
        print("*..................................*")
        times = test_LOS_west_configs(icon, Cams, save=save, saveCam=saveCam)
        for indt,ttt in enumerate(times):
            print(indt, ttt)


def test_LOS_all(save=False, saveCam=[]):
    Cams = ["V1", "V10", "V100", "V1000", "V10000",
            "V100000", "V1000000"]
    configs = ["A1", "A2", "A3", "B1", "B2", "B3"]
    for icon in configs :
        print("*..................................*")
        print("*      Testing the "+icon+" config       *")
        print("*..................................*")
        times = test_LOS_west_configs(icon, Cams, save=save, saveCam=saveCam)
        for ttt in times:
            print(ttt)

def test_line_profile(config="B2", cam="V1000"):
    dconf = load_config(config, plot=False)
    largs, dkwd = prepare_inputs(cam, dconf)
    profile = line_profiler.LineProfiler(_GG.LOS_Calc_PInOut_VesStruct)
    profile.runcall(_GG.LOS_Calc_PInOut_VesStruct, *largs, **dkwd)
    profile.print_stats()

def test_LOS_profiling():
    Cams = ["V100000"]
    Bconfigs = ["B2"]
    for icon in Bconfigs :
        print("*..................................*")
        print("*      Testing the "+icon+" config       *")
        print("*..................................*")
        times = test_LOS_west_configs(icon, Cams, plot=False, save=False)
        for ttt in times:
            print(ttt)

def test_LOS_cprofiling():
    import pyximport
    pyximport.install()
    cProfile.runctx("test_LOS_profiling()", globals(), locals(), "Profile.prof")
    s = pstats.Stats("Profile.prof")
    s.strip_dirs().sort_stats("cumtime").print_stats()


def plot_all_configs():
    ABconfigs = ["A1", "A2", "A3", "B1", "B2", "B3"]
    for config in ABconfigs:
        out = load_config(config, plot=True)
        plt.savefig("config"+config)

def touch_plot_all_configs():
    ABconfigs = ["A1", "A2", "A3", "B1", "B2", "B3"]
    Cams = ["V1", "V10", "V100", "V1000", "V10000",
            "V100000", "V1000000"]

    for indx, config in enumerate(ABconfigs):
        indcam = -2
        cam = Cams[indcam]
        D, u, loscam = get_Du(cam, config=config, make_cam=True)
        loscam.plot_touch()
        plt.savefig("plottouch_dconfig"+config+"_"+cam)

def touch_plot_config_cam(config, cam):
    D, u, loscam = get_Du(cam, config=config, make_cam=True)
    # loscam.plot(Elt='L', EltVes='P', EltStruct='P')
    # plt.savefig("erasemeplz")
    start = time.time()
    loscam.plot_touch()
    end = time.time()
    print("Time for calling plot_touch = ", end-start)
    plt.savefig("plottouch_dconfig"+config+"_"+cam)


if __name__ == "__main__":
    # test_LOS_compact()
    # test_LOS_all()
    # test_LOS_all(save=True,saveCam=["V1000"])
    # test_LOS_cprofiling()
    # plot_all_configs()
    # touch_plot_all_configs()
    # touch_plot_config_cam("B1", "V100000")
    touch_plot_config_cam("B2", "V100000")
    # touch_plot_config_cam("B3", "V100000")
    # line profiling.....
    # test_line_profile(cam="V100000")
    # print(test_LOS_west_configs("B2", ["V10000"]))
