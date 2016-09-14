# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 10:42:24 2014

@author: didiervezinet
"""
import sys
sys.path.append('/afs/ipp/home/g/git/python/repository')
import dd
import kk
import itertools
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib.colors as colors
import scipy.interpolate as scpinterp
import datetime as dtm
import ToFu_Defaults as TFD
import ToFu_PathFile as TFPF
import ToFu_Helper as TFH
import ToFu_Geom as TFG
import ToFu_Mesh as TFM
import ToFu_MatComp as TFMC
import ToFu_Treat as TFT
import ToFu_Inv as TFI
import warnings
cwd = os.getcwd()
os.chdir('./Inputs_TCV/')
import tcv_geometry_New as TCV
os.chdir(cwd)



Exp = 'TCV'


DiagsL = ['SXA','SXB','SXC','SXD','SXF','SXG','SXH','SXI','SXJ','SXK','SXL','SXM','SXN']
DiagsN = [{'F':[16,17,18],'I':[43],'J':[15,22,23,24,25,46,47,48,55,83,84,85]}, {'F':[19],'J':[26,27,49,50,51,52,53,54,56,57,58,59,60,79,82]}, {'H':range(18,24)+range(51,61)},
        {'H':[46,47,48,49,50]+range(85,96)}, {'H':[81,82,83,84],'I':range(53,65)}, {'F':[13,14,15],'I':range(44,53),'J':[16,17,18,80]},
        {'F':[10,11,12,20],'I':[89,91],'J':[12,13,14,19,20,21,86,87,88,89]}, {'F':range(21,28),'H':[17,24,25],'I':[15,16,17,90,92,93]}, {'G':range(9,25)},
        {'G':[25,26,27,28],'K':range(14,26)}, {'K':range(48,59),'M':range(25,30)}, {'K':[46,47],'L':range(9,23)},
        {'L':[23,24,25],'M':range(12,25)}]
CamsL = ['F', 'G', 'H1','H2','H3', 'I1','I2','I3', 'J1','J2','J3', 'K1','K2', 'L','M']
NumsJ3 = range(79,89+1)
NumsJ3.remove(81)   # J_081 not working
CamsN = [range(10,27+1), range(9,28+1), range(17,25+1),range(46,60+1),range(81,95+1), range(15,17+1),range(43,64+1),range(89,93+1), range(12,27+1),range(46,60+1),NumsJ3, range(14,25+1),range(46,58+1), range(9,25+1), range(12,29+1)]
NamesL = []
for ii in range(0,len(CamsL)):
    NamesL += [CamsL[ii][0]+'_'+ "{0:03.0f}".format(nb) for nb in CamsN[ii]]









############################################################################
############################################################################
############################################################################
# --------- Objects creation -------
############################################################################



def _get_defaultsSavePathsdtime(SavePathObj=None, SavePathInp=None, dtime=None, dtFormat=TFD.dtmFormat, Type='Object'):
    assert SavePathObj is None or type(SavePathObj) is str, "Arg SavePathObj must be a str !"
    assert SavePathInp is None or type(SavePathInp) is str, "Arg SavePathInp must be a str !"
    assert dtime is None or type(dtime) is str or isinstance(dtime,dtm.datetime), "Arg dtime must be a str or a dtm.datetime instance !"
    assert type(dtFormat) is str, "Arg dtFormat must be a str !"
    RP = TFPF.Find_Rootpath()
    if SavePathInp is None:
        SavePathInp = RP+'/Inputs_'+Exp+'/'
    if SavePathObj is None:
        SavePathObj = RP+'/Objects_'+Exp+'/' if Type=='Object' else RP+'/Outputs_'+Exp+'/'
    if dtime is None:
        dtime = dtm.datetime.now()
    elif type(dtime) is str:
        dtime = dtm.strptime(dtime,dtFormat)
    return SavePathObj, SavePathInp, dtime


def _Create_All(NameTor='V1', NameMesh='Rough1', Deg=2, SavePathObj=None, SavePathInp=None, dtime=None, dtFormat=TFD.dtmFormat):
    SavePathObj, SavePathInp, dtime = _get_defaultsSavePathsdtime(SavePathObj=SavePathObj, SavePathInp=SavePathInp, dtime=dtime)
    _Create_Tor(Name=NameTor, SavePathObj=SavePathObj, SavePathInp=SavePathInp, dtime=dtime)
    _Create_BF2D(Name=NameMesh, Deg=Deg)


def _Create_Tor(Name='V1', SavePathObj=None, SavePathInp=None, dtime=None, dtFormat=TFD.dtmFormat):
    SavePathObj, SavePathInp, dtime = _get_defaultsSavePathsdtime(SavePathObj=SavePathObj, SavePathInp=SavePathInp, dtime=dtime)
    if Name=='V1':
        PolyRef = np.loadtxt(SavePathInp+Exp+'_Tor.txt', dtype='float', comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=2)
    Id = TFPF.ID('Tor', 'V1', SaveName=None, SavePath=SavePathObj, dtime=dtime, Exp=Exp, LObj=None, dtFormat=dtFormat)
    Tor = TFG.Tor(Id, PolyRef)
    Tor.save()


def _Create_BF2D(Name='Rough1', Degs=[0,1,2], TorName='V1', SavePathObj=None, SavePathInp=None, dtime=None, dtFormat=TFD.dtmFormat):
    assert Name in ['Rough1','Medium1','Fine1'], "Arg Name must be in ['Rough1','Medium1','Fine1'] !"
    SavePathObj, SavePathInp, dtime = _get_defaultsSavePathsdtime(SavePathObj=SavePathObj, SavePathInp=SavePathInp, dtime=dtime)
    LFiles = [f for f in os.listdir(SavePathObj) if all([ss in f for ss in ['TFG_Tor_'+Exp+'_'+TorName, dtime.strftime(dtFormat)]])]
    assert len(LFiles)==1, "None or several possible Tor instances for loading !   "+'TFG_Tor_'+Exp+'_'+TorName+"   and   "+dtime.strftime(dtFormat)
    Tor = TFPF.Open(SavePathObj+LFiles[0])

    if Name=='Rough1':
        KnotsR, ResR = TFM.LinMesh_List([(Tor._PRMin[0],0.75),(0.75,1.0),(1.0,Tor._PRMax[0])], [(0.04,0.02),(0.02,0.02),(0.02,0.04)])
        KnotsZ, ResZ = TFM.LinMesh_List([(Tor._PZMin[1],-0.4),(-0.4,0.4),(0.4,Tor._PZMax[1])], [(0.08,0.03),(0.03,0.03),(0.03,0.08)])
    elif Name=='Medium1':
        KnotsR, ResR = TFM.LinMesh_List([(Tor._PRMin[0],0.75),(0.75,1.05),(1.05,Tor._PRMax[0])], [(0.04,0.015),(0.015,0.015),(0.015,0.04)])
        KnotsZ, ResZ = TFM.LinMesh_List([(Tor._PZMin[1],-0.5),(-0.5,0.50),(0.50,Tor._PZMax[1])], [(0.06,0.02),(0.02,0.02),(0.02,0.06)])
    elif Name=='Fine1':
        KnotsR, ResR = TFM.LinMesh_List([(Tor._PRMin[0],0.70),(0.70,1.1),(1.1,Tor._PRMax[0])], [(0.03,0.01),(0.01,0.01),(0.01,0.03)])
        KnotsZ, ResZ = TFM.LinMesh_List([(Tor._PZMin[1],-0.6),(-0.6,0.6),(0.6,Tor._PZMax[1])], [(0.04,0.015),(0.015,0.015),(0.015,0.04)])

    Id = TFPF.ID('Mesh2D', Name, Diag='', SaveName=None, SavePath=SavePathObj, dtime=dtime, Exp=Exp, LObj=None, dtFormat=TFD.dtmFormat)
    M2 = TFM.Mesh2D(Id, [KnotsR,KnotsZ])
    Poly = Tor.get_InsideConvexPoly(Spline=True)
    M2bis = M2.get_SubMeshPolygon(Poly, NLim=2)
    for ii in range(0,len(Degs)):
        Id = TFPF.ID('BF2D', Name+'_D'+str(Degs[ii]), Diag='', SaveName=None, SavePath=SavePathObj, dtime=dtime, Exp=Exp, LObj=None, dtFormat=TFD.dtmFormat)
        M2 = TFM.BF2D(Id, M2bis, Degs[ii])
        M2.save()


def _Create_LDetectApert_SXR(shot, TorName='V1', SavePathObj=None, SavePathInp=None, dtime=None, dtFormat=TFD.dtmFormat, LMNoTile=False, Plot=True):
    SavePathObj, SavePathInp, dtime = _get_defaultsSavePathsdtime(SavePathObj=SavePathObj, SavePathInp=SavePathInp, dtime=dtime)
    #LFiles = [f for f in os.listdir(SavePathObj) if all([ss in f for ss in ['TFG_Tor_'+Exp+'_'+TorName, dtime.strftime(dtFormat)]])]
    LFiles = [f for f in os.listdir(SavePathObj) if all([ss in f for ss in ['TFG_Tor_'+Exp+'_'+TorName]])]
    assert len(LFiles)==1, "None or several possible Tor instances for loading !   "+'TFG_Tor_'+Exp+'_'+TorName+"   and   "+dtime.strftime(dtFormat)
    Tor = TFPF.Open(SavePathObj+LFiles[0])
    if Plot:
        Lax = Tor.plot(Elt='PI')

    Cams = range(0,10)
    TorAng = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    ff = os.listdir(SavePathObj)
    ffApert = [ss for ss in ff if all([sss in ss for sss in ['TFG_Apert',Exp]])]
    ffDet = [ss for ss in ff if all([sss in ss for sss in ['TFG_Detect',Exp]])]
    for ii in range(0,len(Cams)):
        Cam = TCV.camera(Cams[ii])
        Ndet = len(Cam.chip.centers_of_mass()[0])
        Nslit = len(Cam.slit.coordinates3D())
        assert Cam.chip.coordinates3D().shape[2]==Ndet and Cam.chip.chip.shape[2]==Ndet, "Inconsistent number of chip for Cam "+str(Cams[ii])

        epol = np.array([[np.cos(TorAng[ii])], [np.sin(TorAng[ii])], [0.]])
        eTor = np.array([[-np.sin(TorAng[ii])], [np.cos(TorAng[ii])], [0.]])

        # Create the Apert
        LApert, LIdApert = [], []
        for jj in range(0,Nslit):
            Name = 'Cam{0:02.0f}Ap{1:02.0f}'.format(Cams[ii],jj+1)
            if not any([Name in ss for ss in ffApert]):
                (width, height, thickness, distance) = (Cam.slit.width1, Cam.slit.height1, Cam.slit.thickness1, None) if jj in [0,1] else (Cam.slit.width2, Cam.slit.height2, Cam.slit.thickness2, Cam.slit.distance)
                IdApert = TFPF.ID('Apert', Name, shot=shot, Diag='SXR', SaveName=None, SavePath=SavePathObj, dtime=dtime, Exp=Exp, LObj=None, dtFormat=dtFormat,
                        USRdict={'Cam':Cams[ii],'angle':Cam.slit.angle,'pos':Cam.slit.pos,'width':width,'height':height,'thickness':thickness,'distance':distance,'FiltMat':'Be'})
                PolyApert = Cam.slit.coordinates3D()[jj].T
                PolyApert = epol*PolyApert[0:1,:] + eTor*PolyApert[2:3,:] + np.array([[0.],[0.],[1.]])*PolyApert[1:2,:]
                Apert = TFG.Apert(IdApert, PolyApert, Tor=Tor)
                Apert.save()
            else:
                ffApertjj = [ss for ss in ffApert if Name in ss]
                assert len(ffApertjj)==1, "Several TFG.Apert possible for "+Name
                print "Loading ", ffApertjj[0]
                Apert = TFPF.Open(SavePathObj+ffApertjj[0])
                IdApert = Apert.Id
            LApert.append(Apert)
            LIdApert.append(IdApert)
            if Plot:
                Lax = Apert.plot(Lax=Lax, Elt='PV',EltTor='', LVIn=0.05)

        # Create the Detect
        for jj in range(0,Ndet):
            Name = 'Cam{0:02.0f}Det{1:02.0f}'.format(Cams[ii],jj+1)
            if not any([Name in ss for ss in ffDet]):
                Id = TFPF.ID('Detect', Name, shot=shot, Diag='SXR', SaveName=None, SavePath=SavePathObj, dtime=dtime, Exp=Exp, LObj=None, dtFormat=dtFormat,
                        USRdict={'xc':Cam.chip.xc, 'yc':Cam.chip.yc, 'angle':Cam.chip.angle})
                PolyDet = Cam.chip.coordinates3D()[:,:,jj].T
                PolyDet = epol*PolyDet[0:1,:] + eTor*PolyDet[2:3,:] + np.array([[0.],[0.],[1.]])*PolyDet[1:2,:]
                Det = TFG.Detect(Id, PolyDet, LApert=LApert, Tor=Tor, Calc=True, CalcEtend=True, CalcSpanImp=True, CalcCone=True, CalcPreComp=True)
                Det.save()
            else:
                ffDetjj = [ss for ss in ffDet if Name in ss]
                assert len(ffDetjj)==1, "Several TFG.Detect possible for "+Name
                print "Loading ", ffDetjj[0]
                Det = TFPF.Open(SavePathObj+ffDetjj[0])
                Id = Det.Id
            if Plot:
                Lax = Det.plot(Lax=Lax, Elt='PVC',EltLOS='L',EltApert='',EltTor='', LVIn=0.02)
            del Det



def _RefineConePolys(Diag='SXR', GD=None, Cam=None, SavePathObj=None, SavePathInp=None, dtime=None, dtFormat=TFD.dtmFormat, dMaxPol=0.005, dMaxHor=0.01):
    if GD is None:
        GD = _Load_Geom(Diag=Diag, Cam=Cam, SavePathObj=SavePathObj, dtime=dtime, dtFormat=dtFormat)
    Lax = GD.Tor.plot(Elt='P')
    for ii in range(0,GD.nDetect):
        Lax[0].plot(GD.LDetect[ii]._Cone_PolyPol[0][0,:],GD.LDetect[ii]._Cone_PolyPol[0][1,:],'-k')
        Lax[1].plot(GD.LDetect[ii]._Cone_PolyHor[0][0,:],GD.LDetect[ii]._Cone_PolyHor[0][1,:],'-k')
        GD._LDetect[ii].refine_ConePoly(dMax=dMaxPol, Proj='Pol', indPoly=0)
        GD._LDetect[ii].refine_ConePoly(dMax=dMaxHor, Proj='Hor', indPoly=0)
        Lax = GD.LDetect[ii].plot(Lax=Lax, Elt='PC',EltTor='',EltLOS='',EltApert='P', Conedict={'edgecolors':plt.cm.jet(1.*ii/GD.nDetect), 'facecolors':(0.8,0.8,0.8,0.1), 'linewidths':1.})
        GD._LDetect[ii].save()

def _Create_GMat2D_SXR(Cam=None, Name=None, NameBF2='Rough1', Deg=0, shot=np.inf, SavePathObj=None, SavePathInp=None, dtime=None, LMNoTile=False, dtFormat=TFD.dtmFormat):
    SavePathObj, SavePathInp, dtime = _get_defaultsSavePathsdtime(SavePathObj=SavePathObj, SavePathInp=SavePathInp, dtime=dtime)
    BF2 = get_BF(Name=NameBF2, Deg=Deg, SavePathObj=SavePathObj, SavePathInp=SavePathInp, dtime=dtime, dtFormat=dtFormat)
    GD = _Load_Geom(Diag='SXR', Cam=Cam, shot=shot, SavePathObj=SavePathObj, dtime=dtime, dtFormat=dtFormat, LMNoTile=LMNoTile, Verb=False)
    if Name is None:
        Name = BF2.Id.Name+'_All' if Cam is None else BF2.Id.Name+'_'+Cam
    Id = TFPF.ID('GMat2D', Name, shot=shot, Diag='SXR', SaveName=None, SavePath=SavePathObj, dtime=dtime, Exp=Exp, LObj=None, dtFormat=dtFormat)
    GMat = TFMC.GMat2D(Id, BF2, GD, Mat=None, indMat=None, MatLOS=None, Calcind=True, Calc=True, CalcLOS=True, Iso='Iso', StoreBF2=False, StoreTor=True, StoreLD=False, dtime=dtime, Fast=True, Verb=True)
    GMat.save()
    return GMat

def _Load_Geom(Diag='SXR', Cam=None, shot=np.inf, Exp=Exp, SavePathObj=None, SavePathInp=None, dtime=None, DTime=True, dtFormat=TFD.dtmFormat, LMNoTile=False, Verb=False):
    assert Cam is None or (type(Cam) is str and len(Cam)==1), "Arg Cam must be a str of len()==1 (e.g.: in ['F','G','H','I','J','K','L','M'] for SXR) !"
    assert shot==np.inf or type(shot) is int, "Arg shot must be a int !"
    SavePathObj, SavePathInp, dtime = _get_defaultsSavePathsdtime(SavePathObj=SavePathObj, SavePathInp=SavePathInp, dtime=dtime)
    # Get all Detect
    Keystr = 'TFG_Detect_'+Exp+'_'+Diag if Cam is None else 'TFG_Detect_AUG_'+Diag+'_'+Cam
    LDF = sorted([f for f in os.listdir(SavePathObj) if Keystr in f])
    LDD = [(f[19:29],int(f[32:37])) for f in LDF]
    LDDu = sorted(list(set([f[19:29] for f in LDF])))
    LD = []
    for ii in range(0,len(LDDu)):
        ff = [jj for jj in range(0,len(LDF)) if LDDu[ii]==LDD[jj][0] and LDD[jj][1]<=shot]
        LD.append(LDF[ff[np.argmax([LDD[jj][1] for jj in ff])]])
    for ii in range(0,len(LD)):
        print "    Loading ", LD[ii]
        LD[ii] = TFPF.Open(SavePathObj+LD[ii],Verb=Verb)
    strcam = 'All' if Cam is None else Cam
    Id = TFPF.ID('GDetect',strcam,Diag=Diag,Exp=Exp,LObj=[dd.Id for dd in LD], dtime=LD[0].Id._dtime)
    GD = TFG.GDetect(Id,LD)
    return GD



############################################################################
############################################################################
# --------- Objects Loading -------
############################################################################


def get_Geom(Diag='SXR', Cam=None, shot=np.inf, SavePathObj=None, SavePathInp=None, dtime=None, DTime=True, dtFormat=TFD.dtmFormat, Verb=False):
    return _Load_Geom(Diag=Diag, Cam=Cam, shot=shot, SavePathObj=SavePathObj, SavePathInp=SavePathInp, dtime=dtime, DTime=DTime, dtFormat=dtFormat, Verb=Verb)

def get_BF(Name='Rough1', Deg=0, SavePathObj=None, SavePathInp=None, dtime=None, DTime=True, dtFormat=TFD.dtmFormat):
    SavePathObj, SavePathInp, dtime = _get_defaultsSavePathsdtime(SavePathObj=SavePathObj, SavePathInp=SavePathInp, dtime=dtime)
    if DTime:
        LFiles = [f for f in os.listdir(SavePathObj) if all([ss in f for ss in ['TFM_BF2D_'+Exp+'_'+Name+'_D'+str(Deg), dtime.strftime(dtFormat)]])]
    else:
        LFiles = [f for f in os.listdir(SavePathObj) if 'TFM_BF2D_'+Exp+'_'+Name+'_D'+str(Deg) in f]
    assert len(LFiles)==1, "None or several possible Tor instances for loading !   "+'TFM_BF2D_'+Exp+'_'+Name+'_D'+str(Deg)+"   and   "+dtime.strftime(dtFormat)
    BF2 = TFPF.Open(SavePathObj+LFiles[0])
    return BF2

def get_GMat(NameBF='Rough1', Deg=0, shot=np.inf, SavePathObj=None, SavePathInp=None, dtime=None, DTime=True, dtFormat=TFD.dtmFormat):
    SavePathObj, SavePathInp, dtime = _get_defaultsSavePathsdtime(SavePathObj=SavePathObj, SavePathInp=SavePathInp, dtime=dtime)

    LFiles = [f for f in os.listdir(SavePathObj) if 'TFMC_GMat2D_'+Exp+'_'+NameBF+'_D'+str(Deg) in f and int(f[f.index('sh')+2:f.index('sh')+7])<=shot]
    Lshots = np.array([int(f[f.index('sh')+2:f.index('sh')+7]) for f in LFile])
    LFiles = LFiles[np.argmax(Lshots)]
    assert len(LFiles)==1, "None or several possible Tor instances for loading !   "+'TFMC_GMat2D_'+Exp+'_'+NameBF+'_D'+str(Deg)+"   and   "+dtime.strftime(dtFormat)
    GM2 = TFPF.Open(SavePathObj+LFiles[0])
    return BM2


