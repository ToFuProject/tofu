# coding: utf-8
import tofu.geom._GG as _GG
from tofu_LauraBenchmarck_load_config import *
plt.ion()



def test_LOS_west_A1() :
    out = load_config('A1', plot=False)
    ves = out["Ves"]

    # 1 LOS .............................
    Du = get_Du("V1")
    cam = tf.geom.LOSCam1D(Id="Test", Du = Du, Ves = ves)
    # for plot Calc_Los_PInOut_Tor is being called....
    cam.plot(Elt='L', EltVes="P")
    plt.show(block=True)
    out = _GG.LOS_Calc_PInOut_VesStruct(Du[0],
                                        Du[1]/np.sqrt(np.sum(Du[1]**2, axis=0)),
                                        ves.Poly,
                                        ves.geom['VIn'],
                                        Lim=ves.Lim,
                                        VType=ves.Type)

    # 10 LOS .............................
    Du = get_Du("V10")
    cam = tf.geom.LOSCam1D(Id="Test", Du = Du, Ves = ves)
    # for plot Calc_Los_PInOut_Tor is being called....
    cam.plot(Elt='L', EltVes="P")
    plt.show(block=True)
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
