# coding: utf-8
from tofu_LauraBenchmarck_load_config import *
plt.ion()
import tofu.geom._GG_LM as _GG
import time
import pstats, cProfile



def test_LOS_west_Aconfig(config, cams, plot=False, save=False, saveCam=[]):
    dconf = load_config(config, plot=plot)
    if plot:
        plt.show(block=True)
    ves = dconf["Ves"]
    times = []
    for vcam in cams:
        D, u = get_Du(vcam)
        u = u/np.sqrt(np.sum(u**2, axis=0))
        start = time.time()
        out = _GG.LOS_Calc_PInOut_VesStruct(D, u,
                                            ves.Poly,
                                            ves.geom['VIn'],
                                            Lim=ves.Lim,
                                            VType=ves.Type)
        elapsed = time.time() - start
        if save and vcam in saveCam :
            np.savez("out_sin_"+config+"_"+vcam+".npz", out[0])
            np.savez("out_sout_"+config+"_"+vcam+".npz", out[1])
            np.savez("out_vperpin_"+config+"_"+vcam+".npz", out[2])
            np.savez("out_vperpout_"+config+"_"+vcam+".npz", out[3])
            np.savez("out_indin_"+config+"_"+vcam+".npz", out[4])
            np.savez("out_indout_"+config+"_"+vcam+".npz", out[5])
        times.append(elapsed)
    return times


def test_LOS_west_Bconfig(config, cams, plot=False, save=False, saveCam=[],
                          plot_cam=False):
    dconf = load_config(config, plot=plot)
    if plot:
        plt.show(block=True)
    ves = dconf["Ves"]
    struct = list(dconf['Struct'].values())
    times = []
    for vcam in cams:
        D, u = get_Du(vcam)
        if plot_cam :
            Cam = tf.geom.LOSCam1D(Id=vcam, Du=(D,u), Ves=ves, LStruct=struct)
            Cam.plot(Elt='L', EltVes='P', EltStruct='P')
            plt.show(block=True)
        u = u/np.sqrt(np.sum(u**2, axis=0))
        lSPoly = [ss.Poly for ss in struct]
        lSLim = [ss.Lim for ss in struct]
        lSVIn = [ss.geom['VIn'] for ss in struct]
        start = time.time()
        # profile = line_profiler.LineProfiler(_GG.LOS_Calc_PInOut_VesStruct)
        # profile.runcall(_GG.LOS_Calc_PInOut_VesStruct, D, u,
        #                 ves.Poly,
        #                 ves.geom['VIn'],
        #                 Lim=ves.Lim,
        #                 LSPoly = lSPoly,
        #                 LSLim=lSLim,
        #                 LSVIn=lSVIn,
        #                 VType=ves.Type)
        # profile.print_stats()
        out = _GG.LOS_Calc_PInOut_VesStruct(D, u,
                                            ves.Poly,
                                            ves.geom['VIn'],
                                            Lim=ves.Lim,
                                            LSPoly = lSPoly,
                                            LSLim=lSLim,
                                            LSVIn=lSVIn,
                                            VType=ves.Type)
        elapsed = time.time() - start
        if save and vcam in saveCam :
            np.savez("out_sin_"+config+"_"+vcam+".npz", out[0])
            np.savez("out_sout_"+config+"_"+vcam+".npz", out[1])
            np.savez("out_vperpin_"+config+"_"+vcam+".npz", out[2])
            np.savez("out_vperpout_"+config+"_"+vcam+".npz", out[3])
            np.savez("out_indin_"+config+"_"+vcam+".npz", out[4])
            np.savez("out_indout_"+config+"_"+vcam+".npz", out[5])
        times.append(elapsed)
    return times

def test_LOS_compact(save=False, saveCam=[]):
    Cams = ["V1", "V10", "V100", "V1000", "V10000"]#,
    #"V100000"]#, "V1000000"]
    CamsA = ["VA1", "VA10", "VA100", "VA1000", "VA10000"]#,
    #"VA100000", "VA1000000"]
    Aconfigs = ["A1", "A2", "A3"]
    Bconfigs = ["B1", "B2", "B3"]
    for icon in Aconfigs :
        print("*..................................*")
        print("*      Testing the "+icon+" config       *")
        print("*..................................*")
        if icon == "A2":
            times = test_LOS_west_Aconfig(icon, CamsA, save=save, saveCam=saveCam)
        else:
            times = test_LOS_west_Aconfig(icon, Cams, save=save, saveCam=saveCam)
        for ttt in times:
            print(ttt)

    for icon in Bconfigs :
        print("*..................................*")
        print("*      Testing the "+icon+" config       *")
        print("*..................................*")
        times = test_LOS_west_Bconfig(icon, Cams, save=save, saveCam=saveCam)
        for ttt in times:
            print(ttt)

def test_LOS_all(save=False, saveCam=[]):
    Cams = ["V1", "V10", "V100", "V1000", "V10000",
            "V100000", "V1000000"]
    CamsA = ["VA1", "VA10", "VA100", "VA1000", "VA10000",
             "VA100000", "VA1000000"]
    Aconfigs = ["A1", "A2", "A3"]
    Bconfigs = ["B1", "B2", "B3"]
    for icon in Aconfigs :
        print("*..................................*")
        print("*      Testing the "+icon+" config       *")
        print("*..................................*")
        if icon == 2:
            times = test_LOS_west_Aconfig(icon, CamsA, save=save, saveCam=saveCam)
        else:
            times = test_LOS_west_Aconfig(icon, Cams, save=save, saveCam=saveCam)
        for ttt in times:
            print(ttt)

    for icon in Bconfigs :
        print("*..................................*")
        print("*      Testing the "+icon+" config       *")
        print("*..................................*")
        times = test_LOS_west_Bconfig(icon, Cams, save=save, saveCam=saveCam)
        print(times)
        for ttt in times:
            print(ttt)

def test_LOS_profiling():
    Cams = ["V10"]
    Bconfigs = ["B2"]
    for icon in Bconfigs :
        print("*..................................*")
        print("*      Testing the "+icon+" config       *")
        print("*..................................*")
        times = test_LOS_west_Bconfig(icon, Cams, plot=True, save=True, plot_cam=True)
        for ttt in times:
            print(ttt)

def test_LOS_profilingA():
    CamsA = ["VA1000000"]
    Aconfigs = ["A2"]
    for icon in Aconfigs :
        print("*..................................*")
        print("*      Testing the "+icon+" config       *")
        print("*..................................*")
        times = test_LOS_west_Aconfig(icon, CamsA)
        for ttt in times:
            print(ttt)

def test_LOS_cprofiling(num=0):
    import pyximport
    pyximport.install()
    if num == 0:
        cProfile.runctx("test_LOS_profiling()", globals(), locals(), "Profile.prof")
    else:
        cProfile.runctx("test_LOS_profilingA()", globals(), locals(), "Profile.prof")
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
    CamsA = ["VA1", "VA10", "VA100", "VA1000", "VA10000",
            "VA100000", "VA1000000"]

    for indx, config in enumerate(ABconfigs):
        dconfig = load_config(config, plot=False)
        indcam = -1
        if indx < 3:
            indcam = indcam-1
        else:
            indcam = indcam-2
        if indx == 1:
            cam = CamsA[indcam]
        else:
            cam = Cams[indcam]
        (D,u) = get_Du(cam)
        if 'Struct' in dconfig.keys():
            LStruct = list(dconfig['Struct'].values())
        else:
            LStruct = None

        # Create the LOSCam2D object
        Cam = tf.geom.LOSCam2D(Id=cam, Du=(D,u), Ves=dconfig['Ves'], LStruct=LStruct)
        Cam.plot_touch()
        plt.savefig("plottouch_dconfig"+config+"_"+cam)

def touch_plot_config_cam(config, cam):
    dconfig = load_config(config, plot=True)
    (D,u) = get_Du(cam)
    if 'Struct' in dconfig.keys():
        LStruct = list(dconfig['Struct'].values())
    else:
        LStruct = None

    # Create the LOSCam2D object
    Cam = tf.geom.LOSCam2D(Id=cam, Du=(D,u), Ves=dconfig['Ves'], LStruct=LStruct)
    # Cam.plot(Elt='L', EltVes='P', EltStruct='P')
    # plt.savefig("erasemeplz")
    Cam.plot_touch()
    plt.savefig("plottouch_dconfig"+config+"_"+cam)


    
if __name__ == "__main__":
    # test_LOS_compact()
    test_LOS_all(save=False,saveCam=["V1000"])
    # test_LOS_profiling()
    # test_LOS_cprofiling()
    # plot_all_configs()
    # touch_plot_all_configs()
    # touch_plot_config_cam("A2", "VA100")
    # line profiling.....
    # profile = line_profiler.LineProfiler(test_LOS_profilingA)
    # profile.runcall(test_LOS_profilingA)
    # profile.print_stats()
    #test_LOS_profiling()
