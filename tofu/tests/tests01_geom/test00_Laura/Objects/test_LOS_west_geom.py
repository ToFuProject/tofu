# coding: utf-8
from tofu_LauraBenchmarck_load_config import *
plt.ion()
import tofu.geom._GG as _GG
import time


def test_LOS_west_Aconfig(config) :
    out = load_config(config, plot=False)
    ves = out["Ves"]

    # 1 LOS .............................
    Du = get_Du("V1")
    # cam = tf.geom.LOSCam1D(Id="Test", Du = Du, Ves = ves)
    # # for plot Calc_Los_PInOut_Tor is being called....
    # cam.plot(Elt='L', EltVes="P")
    # plt.show(block=True)
    out = _GG.LOS_Calc_PInOut_VesStruct(Du[0],
                                        Du[1]/np.sqrt(np.sum(Du[1]**2, axis=0)),
                                        ves.Poly,
                                        ves.geom['VIn'],
                                        Lim=ves.Lim,
                                        VType=ves.Type)

    # 10 LOS .............................
    Du = get_Du("V10")
    # cam = tf.geom.LOSCam1D(Id="Test", Du = Du, Ves = ves)
    # # for plot Calc_Los_PInOut_Tor is being called....
    # cam.plot(Elt='L', EltVes="P")
    # plt.show(block=True)
    out = _GG.LOS_Calc_PInOut_VesStruct(Du[0],
                                        Du[1]/np.sqrt(np.sum(Du[1]**2, axis=0)),
                                        ves.Poly, 
                                        ves.geom['VIn'],
                                        Lim=ves.Lim,
                                        VType=ves.Type)


    # 100 LOS .............................
    Du = get_Du("V100")
    # cam = tf.geom.LOSCam1D(Id="Test", Du = Du, Ves = ves)
    # # for plot Calc_Los_PInOut_Tor is being called....
    # cam.plot(Elt='L', EltVes="P")
    # plt.show(block=True)
    out = _GG.LOS_Calc_PInOut_VesStruct(Du[0],
                                        Du[1]/np.sqrt(np.sum(Du[1]**2, axis=0)),
                                        ves.Poly,
                                        ves.geom['VIn'],
                                        Lim=ves.Lim,
                                        VType=ves.Type)


    # 1000 LOS .............................
    Du = get_Du("V1000")
    # cam = tf.geom.LOSCam1D(Id="Test", Du = Du, Ves = ves)
    # # for plot Calc_Los_PInOut_Tor is being called....
    # cam.plot(Elt='L', EltVes="P")
    # plt.show(block=True)
    out = _GG.LOS_Calc_PInOut_VesStruct(Du[0], Du[1]/np.sqrt(np.sum(Du[1]**2, axis=0)), ves.Poly,
                                        ves.geom['VIn'],
                                        Lim=ves.Lim,
                                        VType=ves.Type)

    # 10000 LOS .............................
    Du = get_Du("V10000")
    # cam = tf.geom.LOSCam1D(Id="Test", Du = Du, Ves = ves)
    # # for plot Calc_Los_PInOut_Tor is being called....
    # cam.plot(Elt='L', EltVes="P")
    # plt.show(block=True)
    out = _GG.LOS_Calc_PInOut_VesStruct(Du[0], Du[1]/np.sqrt(np.sum(Du[1]**2, axis=0)), ves.Poly,
                                        ves.geom['VIn'],
                                        Lim=ves.Lim,
                                        VType=ves.Type)

    # 100000 LOS .............................
    Du = get_Du("V100000")
    # cam = tf.geom.LOSCam1D(Id="Test", Du = Du, Ves = ves)
    out = _GG.LOS_Calc_PInOut_VesStruct(Du[0],
                                        Du[1]/np.sqrt(np.sum(Du[1]**2, axis=0)),
                                        ves.Poly,
                                        ves.geom['VIn'],
                                        Lim=ves.Lim,
                                        VType=ves.Type)
 
    # 1000000 LOS .............................
    Du = get_Du("V1000000")
    # cam = tf.geom.LOSCam1D(Id="Test", Du = Du, Ves = ves)
    out = _GG.LOS_Calc_PInOut_VesStruct(Du[0],
                                        Du[1]/np.sqrt(np.sum(Du[1]**2, axis=0)),
                                        ves.Poly,
                                        ves.geom['VIn'],
                                        Lim=ves.Lim,
                                        VType=ves.Type)

def test_LOS_west_Aconfig_short(config, cams, plot=False):
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
        times.append(elapsed)
    return times


def test_LOS_west_Bconfig(config, cams, plot=False):
    dconf = load_config(config, plot=plot)
    if plot:
        plt.show(block=True)
    ves = dconf["Ves"]
    struct = list(dconf['Struct'].values())

    for vcam in cams:
        D, u = get_Du(vcam)
        u = u/np.sqrt(np.sum(u**2, axis=0))
        lSPoly = [ss.Poly for ss in struct]
        lSLim = [ss.Lim for ss in struct]
        lSVIn = [ss.geom['VIn'] for ss in struct]
        start = time.time()
        out = _GG.LOS_Calc_PInOut_VesStruct(D, u,
                                            ves.Poly,
                                            ves.geom['VIn'],
                                            Lim=ves.Lim,
                                            LSPoly = lSPoly,
                                            LSLim=lSLim,
                                            LSVIn=lSVIn,
                                            VType=ves.Type)
        elapsed = time.time() - start
        times.append(elapsed)
    return times

if __name__ == "__main__":
    Cams = ["V1", "V10", "V100", "V1000", "V10000",
            "V100000", "V1000000"]
    Aconfigs = ["A1", "A2", "A3"]
    Bconfigs = ["B1", "B2", "B3"]
    for icon in Aconfigs :
        print("*..................................*")
        print("*      Testing the "+icon+" config       *")
        print("*..................................*")
        times = test_LOS_west_Aconfig_short(icon, Cams)
        print(times)

    for icon in Bconfigs :
        print("*..................................*")
        print("*      Testing the "+icon+" config       *")
        print("*..................................*")
        times = test_LOS_west_Bconfig(icon, Cams)
        print(times)


