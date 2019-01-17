# coding: utf-8
from tofu_LauraBenchmarck_load_config import *
import tofu.geom._GG_LM as _GG
import time
#import line_profiler
# import pstats, cProfile
# from pathlib import Path
# from resource import getpagesize
# import os
# import psutil
# from memory_profiler import profile

def get_resident_set_size():
    # Columns are: size resident shared text lib data dt
    statm = Path('/proc/self/statm').read_text()
    fields = statm.split()
    return int(fields[1]) * getpagesize()


_is_new_version = True
_all_cams = ["V1", "V10", "V100", "V1000", "V10000",
            "V100000", "V1000000"]
def mem():
	print(str(round(psutil.Process().memory_info().rss/1024./1024., 2)) + ' MB')


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
    num_tot_structs = 0
    for ss in lS:
        lSPoly.append(ss.Poly_closed)
        lSVIn.append(ss.dgeom['VIn'])
        lSLim.append(ss.Lim)
        lSnLim.append(ss.nLim)
        if ss.Lim is None or len(ss.Lim) == 0:
            num_tot_structs += 1
        else:
            num_tot_structs += len(ss.Lim)

    largs = [D, u, VPoly, VVIn]
    if _is_new_version:
        loc_rmin = -1
        dkwd = dict(Lim=Lim, nLim=nLim, nstruct=num_tot_structs,
                    LSPoly=lSPoly, LSLim=lSLim,
                    lSnLim=np.asarray(lSnLim, dtype=np.int64), LSVIn=lSVIn, VType=VType,
                    RMin=loc_rmin, Forbid=True, EpsUz=1.e-6, EpsVz=1.e-9,
                    EpsA=1.e-9, EpsB=1.e-9, EpsPlane=1.e-9, Test=True)

    else:
        loc_rmin = None
        dkwd = dict(Lim=Lim, nLim=nLim,
                    LSPoly=lSPoly, LSLim=lSLim,
                    lSnLim=lSnLim, LSVIn=lSVIn, VType=VType,
                    RMin=loc_rmin, Forbid=True, EpsUz=1.e-6, EpsVz=1.e-9,
                    EpsA=1.e-9, EpsB=1.e-9, EpsPlane=1.e-9, Test=True)

    return largs, dkwd

def test_LOS_west_configs(config="B2", cams=["V1000"], plot=False, save=False, saveCam=[]):
    dconf = load_config(config, plot=plot)
    if plot:
        plt.show(block=True)
    times = []
    if not _is_new_version :
        if config=="A2":
            cams = ["VA1000"]
        else:
            cams = ["V1000"]
        print("WARNING : old version so only computing for V1000")
    for vcam in cams:
        largs, dkwd = prepare_inputs(vcam, dconf)
        start = time.time()
        out = _GG.LOS_Calc_PInOut_VesStruct(*largs, **dkwd)
        elapsed = time.time() - start
        if config == "A2":
            num = int(vcam[2:])
        else:
            num = int(vcam[1:])
        if save and vcam in saveCam and _is_new_version:
            np.savez("out_kin_"     +config+"_V"+str(num)+".npz", out[0])
            np.savez("out_kout_"    +config+"_V"+str(num)+".npz", out[1])
            np.savez("out_vperpout_"+config+"_V"+str(num)+".npz",
                     np.transpose(out[2].reshape(num,3)))
            np.savez("out_indout_"    +config+"_V"+str(num)+".npz",
                     np.transpose(out[3].reshape(num,3)))
        if save and vcam in saveCam and not  _is_new_version:
            np.savez("out_kin_"     +config+"_V"+str(num)+".npz", out[2])
            np.savez("out_kout_"    +config+"_V"+str(num)+".npz", out[3])
            np.savez("out_vperpout_"+config+"_V"+str(num)+".npz", out[5])
            np.savez("out_indout_"    +config+"_V"+str(num)+".npz", out[7])

        times.append(elapsed)
    return times




def test_LOS_compact(save=False, saveCam=[]):
    Cams = ["V1", "V10", "V100", "V1000", "V10000",
            "V100000", "V1000000"]
    CamsA = ["VA1", "VA10", "VA100", "VA1000", "VA10000",
             "VA100000"]
    # Cams = ["V1000"]
    # CamsA = ["VA1000"]
    configs = ["A1", "A2", "A3", "B1", "B2", "B3"]
    configs = ["B1", "B2", "B3"]
    for icon in configs :
        print("*..................................*")
        print("*      Testing the "+icon+" config       *")
        print("*..................................*")
        if icon == "A2":
            times = test_LOS_west_configs(icon, CamsA, save=save, saveCam=saveCam)
        else:
            times = test_LOS_west_configs(icon, Cams, save=save, saveCam=saveCam)
        for indt,ttt in enumerate(times):
            print(indt, ttt)


def test_LOS_all(save=False, saveCam=[]):
    Cams  = ["V1", "V10", "V100", "V1000", "V10000",
             "V100000", "V1000000"]
    CamsA = ["VA1", "VA10", "VA100", "VA1000", "VA10000",
             "VA100000", "VA1000000"]
    configs = ["A1", "A2", "A3", "B1", "B2", "B3"]
    for icon in configs :
        print("*..................................*")
        print("*      Testing the "+icon+" config       *")
        print("*..................................*")
        if icon == "A2":
            times = test_LOS_west_configs(icon, CamsA,
                                          save=save, saveCam=saveCam)
        else:
            times = test_LOS_west_configs(icon, Cams,
                                          save=save, saveCam=saveCam)
        for ttt in times:
            print(ttt)

def test_line_profile(config="B2", cam="V1000"):
    dconf = load_config(config, plot=False)
    largs, dkwd = prepare_inputs(cam, dconf)
    profiler = line_profiler.LineProfiler(_GG.LOS_Calc_PInOut_VesStruct)
    profiler.runcall(_GG.LOS_Calc_PInOut_VesStruct, *largs, **dkwd)
    profiler.print_stats()

def test_LOS_profiling():
    Cams = ["V1000000"]
    Bconfigs = ["B3"]
    for icon in Bconfigs :
        print("*............................................*")
        print("*      Testing the "+icon+" config with "+Cams[0]+"  *")
        print("*............................................*")
        times = test_LOS_west_configs(icon, Cams, plot=False, save=False)
        for ttt in times:
            print(ttt)

def test_LOS_cprofiling():
    # import pyximport
    # pyximport.install()
    # cProfile.runctx("test_LOS_profiling()", globals(), locals(), "Profile.prof")
    # s = pstats.Stats("Profile.prof")
    # s.strip_dirs().sort_stats("cumtime").print_stats()
    return

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
        indcam = -3
        if config=="A2":
            cam = CamsA[indcam]
        else:
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


def are_results_the_same():
    from os import listdir
    from os.path import isfile, join
    mypath = "new_res/"
    os.system("mv out_*.npz "+mypath)
    print(listdir(mypath))
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    for f in onlyfiles:
        res_old = np.load("old_res/"+f)
        res_new = np.load("new_res/"+f)
        arr_old = res_old['arr_0']
        arr_new = res_new['arr_0']
        arr_err = np.sum(np.abs(arr_old - arr_new))
        arr_eql = np.allclose(arr_old, arr_new, equal_nan=True)
        if arr_err > 0.0001 and not arr_eql:
            print(f)
            wh_diff = np.where(arr_old != arr_new)
            if np.size(arr_new.shape) == 2:
                print(arr_old.shape, arr_new.shape)
                print(wh_diff)
                for ii in range(4):
                    print("where = ", wh_diff[0][ii], wh_diff[1][ii],
                          "old =",arr_old[wh_diff[0][ii]:, wh_diff[1][ii]],
                          "new =",arr_new[wh_diff[0][ii]:, wh_diff[1][ii]])
                print( "old : ", arr_old[0][0], type(arr_old),
                       type(arr_old[0][0]), arr_old.shape)
                print( "new : ", arr_new[0][0], type(arr_new),
                      type(arr_new[0][0]), arr_new.shape)
                print("is elem 0 equal and error :", arr_new[0][0]==arr_old[0][0],
                      arr_err)

            else:
                print(arr_new[0], arr_old[0], arr_new[0]==arr_old[0],
                      type(arr_new), type(arr_old), type(arr_new[0]),
                      type(arr_old[0]), arr_err)
        else :
            print(f, ": TRUE ", arr_err, arr_eql)
        if not arr_eql:
            if np.size(arr_new.shape) == 2:
                print("old")
                print(arr_old[:,:3])
                print("new")
                print(arr_new[:,:3])
            else:
                print("old")
                print(arr_old[:3])
                print("new")
                print(arr_new[:3])
        print("--------------------------\n\n")


def check_memory_usage(cam="V1000000", config="B2"):
    start_memory = get_resident_set_size()
    test_LOS_west_configs(config, [cam])
    print(get_resident_set_size() - start_memory)

def check_memory_usage2(cam="V1000000", config="B2"):
    process = psutil.Process(os.getpid())
    test_LOS_west_configs(config, [cam])
    print(process.memory_info()[0])


if __name__ == "__main__":
    test_LOS_compact()
    # test_LOS_all()
    # test_LOS_all(save=True, saveCam=["V1000", "VA1000"])
    # test_LOS_cprofiling()
    # plot_all_configs()
    # touch_plot_all_configs()
    #touch_plot_config_cam("A2", "VA10000")
    # touch_plot_config_cam("A1", "V10000")
    # touch_plot_config_cam("B3", "V100000")
    # line profiling.....
    # test_line_profile(cam="V100000")
    # print(test_LOS_west_configs("B3", ["V1000000"]))
    # print(test_LOS_west_configs("B2", "V1000", save=True, saveCam=["V1000"]))
    # test_LOS_all(save=True,saveCam=["V1000", "VA1000"])
    # are_results_the_same()
    # check_memory_usage()
    # mem()
    # check_memory_usage2()
    # mem()
