

from numpy import *
from matplotlib.pylab import *
from scipy.io import loadmat

#import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#fig = figure()
#ax = fig.add_subplot(111, projection='3d')



#he forget 1.5mm shift in camera 9!!!
#dZ   = {1:6.1, 2:0 ,3:-6 ,4:5 ,5:9 ,6:-9 ,7:-5 ,8:7 ,9:1.5 ,10:-5.5}  #values from technical drawing
#dZ   = {1:6.09, 2:0 ,3:-6.01 ,4:5 ,5:9 ,6:-9 ,7:-5 ,8:6.85 ,9:1.5 ,10:-5.6}  #valus used by previos student (cam 9 is corrected)
#dR   = {1:17.93, 2:15.84 ,3:16.06 ,4:9.9 ,5:13.4 ,6:13.4 ,7: 9.9,8:22.41 ,9:21.78 ,10:25.44}#values from technical drawing
#tilt = {1:18., 2:0 ,3:-21 ,4: 0,5: 0,6:0 ,7:0 ,8:17 ,9:0 ,10:-12}#values from technical drawing

#chamber boundary
coord= [0.624, 1.136, -0.75, 0.75]
vessel = c_[
 [1.137 ,  0.547],
 [0.9685,  0.748],
 [0.678,    0.749],
 [0.624,   0.703],
 [0.624, -0.704],
 [0.667,  -0.745],
 [0.9704, -0.747],
 [1.134,   -0.551],
 [1.137,  0.547]]

#inner boundary
vessel_in = [ [1.137, 0.9685,0.678,0.624,0.624 ,0.667 ,0.9704 ,1.134,1.137 ],
                [0.547, 0.748,0.749,0.703,-0.704,-0.745,-0.747,-0.551,0.547]]

#outer boundary
vessel_out = [[1.154, 0.997,.6385,.6035,0.6,0.654, 1.012,1.154,1.154  ],
            [0.631,0.77, 0.769, 0.7241,-0.709,-0.77,-0.766,-0.631,0.631]]


#detector
d1=0.90;                     # detector width      in mm
d2=4.0;                      # detector length     in mm
sep=0.05;                    # detector width separation  in mm
Ndiods = 20;


#poloidal slit
b1x=0.8;                      # aperture width      in mm
b1y=6;                        # aperture length     in mm
b1z=0.02;                     # aperture thickness  in mm
#toroidal slit
b2x=200;                 # aperture width      in mm
b2y=[8,8,8,15,15,15,15,8,8,8];# aperture length     in mm
b2z=0;                        # aperture thickness  in mm

tilt = [18.,0,-21,0,0,0,0,17,0, -12]#values from technical drawing
# angle of detector surface normal
#vangle=[108,90,69,0,0,0,0,-73,-90,-102];

xpos_slit=r_[71.5,87.7,103.9,123.1,123.1,123.1,123.1,104.04,87.84,71.64]
# x position of the slits in [cm]

ypos_slit=r_[80.35,80.35,80.35,48.1,1.95,-2.45,-48.6,-80.35,-80.35,-80.35];
# y position of the slits in [cm]

ae=r_[6.09,0,-6.01,5,9,-9,-5,6.85,1.5,-5.6] #cam 9 is corrected
#excentricity array/slit [mm]

da=   r_[17.93,15.84,16.06,9.9,13.4,13.4, 9.9,22.41,21.78,25.44];
# poloidal slit-array distance in [mm]

da2= r_[37,34.4,37,55.9,59.4,59.4,55.9,37,34.4,37];
#toriodal slit array [mm];

### da2 are not used, not clear what they are (heat shield apertures?), HW 15.11.05

#small correction in distance between slit and chip
#where is it caming from????
deltada=r_[   -0.0311 ,0,  -0.0458  , -0.1179   ,-0.0615  , -0.1105,
        -0.0510  , -0.0515 , 0 , -0.3223] #[mm]
deltada[3]=0;
deltada[5]=0;
da+= deltada;
da2+=deltada;





class slit:
    def __init__(self, num, angle):

        self.width1 = b1x*1e-3
        self.height1 = b1y*1e-3
        self.thickness1 = b1z*1e-3

        self.width2 = b2x*1e-3
        self.height2 =  b2y[num]*1e-3
        self.thickness2 = b2z*1e-3
        self.distance = (da2[num]-da[num])*1e-3

        self.pos = xpos_slit[num]/100,ypos_slit[num]/100
        self.angle = angle


    def show(self):

        slit1,slit2,slit3 = self.coordinates3D()
        plot(slit1[:,0],slit1[:,1])
        plot(slit2[:,0],slit2[:,1])
        plot(slit3[:,0],slit3[:,1])

    def show3D(self,ax):

        slit1,slit2,slit3 = self.coordinates3D()
        slit1 = r_[slit1[(-1,),:],slit1]
        slit2 = r_[slit2[(-1,),:],slit2]
        slit3 = r_[slit3[(-1,),:],slit3]

        ax.plot(slit1[:,0],slit1[:,1],slit1[:,2])
        ax.plot(slit2[:,0],slit2[:,1],slit2[:,2])
        ax.plot(slit3[:,0],slit3[:,1],slit3[:,2])



    def coordinates3D(self):

        dx = cos(self.angle)*self.width1/2
        dy = sin(self.angle)*self.width1/2
        dx_ = sin(self.angle)*self.thickness1/2
        dy_ = cos(self.angle)*self.thickness1/2

        A1 = [self.pos[0]-dx, self.pos[1]-dy, self.height1/2]
        A2 = [self.pos[0]+dx, self.pos[1]+dy, self.height1/2]
        A3 = [self.pos[0]+dx, self.pos[1]+dy, -self.height1/2]
        A4 = [self.pos[0]-dx, self.pos[1]-dy, -self.height1/2]

        slit1 = vstack((A1,A2,A3,A4))+r_[dx_,dy_,0]
        slit2 = vstack((A1,A2,A3,A4))-r_[dx_,dy_,0]

        dx = cos(self.angle)*self.width2/2
        dy = sin(self.angle)*self.width2/2
        #dx_ = sin(self.angle)*self.thickness2/2
        #dy_ = cos(self.angle)*self.thickness2/2

        dax_ = -abs(sin(self.angle))*self.distance
        day_ = -cos(self.angle)*self.distance

        A1 = [self.pos[0]-dx, self.pos[1]-dy, self.height2/2]
        A2 = [self.pos[0]+dx, self.pos[1]+dy, self.height2/2]
        A3 = [self.pos[0]+dx, self.pos[1]+dy, -self.height2/2]
        A4 = [self.pos[0]-dx, self.pos[1]-dy, -self.height2/2]

        slit3 = vstack((A1,A2,A3,A4))+r_[dax_,day_,0]

        return slit1, slit2, slit3



class diod_chip:
    def __init__(self, angle,xc,yc):

        one_diod = array(([0,-d2/2],[d1,d2/2]))
        chip = dstack([one_diod+[i*(d1+sep),0] for i in range(Ndiods)])
        w_center = (chip[0,0,0]+chip[1,0,-1])/2
        chip[:,0,:] -= w_center

        self.chip = chip*1e-3
        self.angle = angle
        self.xc=xc
        self.yc=yc


    def show(self):
        diods = self.coordinates3D()
        for d in diods.T:
            plot(d[0],d[1])

    def show3D(self,ax):
        diods = self.coordinates3D()
        diods = r_[diods[(-1,),:],diods]
        for d in diods.T:
            ax.plot(d[0],d[1],d[2],'g')


    def centers_of_mass(self):
        X, Y = [],[]
        for i in range(self.chip.shape[2]):
            dx = self.chip[:,0,i]*cos(self.angle)
            dy = self.chip[:,0,i]*sin(self.angle)
            X.append(mean(dx)+self.xc)
            Y.append(mean(dy)+self.yc)
        return X,Y

    def coordinates3D(self):
        diods_coord = []
        for i in range(self.chip.shape[2]):
            x = self.chip[:,0,i]*cos(self.angle)+self.xc
            y = self.chip[:,0,i]*sin(self.angle)+self.yc
            z = self.chip[:,1,i]

            A = [x[0],y[0],z[0]]
            B = [x[0],y[0],z[1]]
            C = [x[1],y[1],z[1]]
            D = [x[1],y[1],z[0]]

            diods_coord.append(vstack((A,B,C,D)))

        return dstack(diods_coord)





class camera:
    def __init__(self,num):
        self.num = num+1
        if self.num > 3 and self.num < 8 :
            self.camera_angle = pi/2*3
        elif self.num <= 3:
            self.camera_angle = 0
        elif self.num >= 8:
            self.camera_angle = pi

        #print shape(da), num
        self.dR = da[num]*1e-3 #distance between slit and center of the diod
        self.dZ = ae[num]*1e-3 #distance between slit and center of the diod
        self.tilt = tilt[num]/180.*pi

        self.slit = slit(num, self.camera_angle)

        shift_x = cos(self.camera_angle+pi/2)*self.dR -sin(self.camera_angle+pi/2)*self.dZ
        shift_y = sin(self.camera_angle+pi/2)*self.dR +cos(self.camera_angle+pi/2)*self.dZ

        self.chip = diod_chip( self.camera_angle+self.tilt,
                              self.slit.pos[0]+shift_x , self.slit.pos[1]+shift_y  )


    def plot_los(self):
        Xc,Yc = self.chip.centers_of_mass()

        for x,y in zip(Xc,Yc):
            t = linspace(0,100, 10000)
            X = -(x-self.slit.pos[0])*t+x
            Y = -(y-self.slit.pos[1])*t+y
            ind = (t<3)|((X>coord[0])&(X<coord[1])&(Y>coord[2])&(Y<coord[3]))
            X = X[ind]
            Y = Y[ind]

            plot(r_[X[0],X[-1]],r_[Y[0],Y[-1]],'b')

    def show3D(self,ax):
        Xc,Yc = self.chip.centers_of_mass()

        for x,y in zip(Xc,Yc):
            t = linspace(0,100, 10000)
            X = -(x-self.slit.pos[0])*t+x
            Y = -(y-self.slit.pos[1])*t+y
            ind = (t<3)|((X>coord[0])&(X<coord[1])&(Y>coord[2])&(Y<coord[3]))
            X = X[ind]
            Y = Y[ind]

            ax.plot(r_[X[0],X[-1]],r_[Y[0],Y[-1]],[0,0],'b')


#geom=loadmat('./geometry/TCV/XTOMO/cat_defaults2008.mat')
geom=loadmat('./cat_defaults2008.mat')


def PlotAll():

    f1 = figure()
    old_etendue  = 1/geom['angfact']/1e4
    for i in range(10):
        plot(1+arange(Ndiods)+i*Ndiods,old_etendue[:,i])
        #plot(arange(Ndiods)+i*Ndiods+.5,old_etendue[:,i])
    #[axvline(x=i*Ndiods) for i in range(10)]
    title('old etendue')
    ylabel('etendue')
    xlabel('detetor')


    f2 = figure()
    title('geometry')
    plot(xpos_slit/100,ypos_slit/100,'ro')

    for i in range(10):
        ychords = geom['ychord'][:,i*Ndiods:(i+1)*Ndiods]/100
        xchords = geom['xchord'][:,i*Ndiods:(i+1)*Ndiods]/100
        plot(xchords,ychords,'k',lw=.2)

        C = camera(i)
        C.plot_los()
        C.chip.show()
        C.slit.show()

    plot(vessel_in[0], vessel_in[1],'k')
    plot(vessel_out[0], vessel_out[1],'k')

    axis('equal')
    coord= [0.624, 1.136, -0.75, 0.75]
    xlim(coord[0],coord[1])
    return f1, f2



#################################  calculate 3D coordinates of the slits + detectors ########################

diod_arrays = []
slits = []
for i in range(10):
    C = camera(i)
    diod_arrays.append(C.chip.coordinates3D())
    slits.append(C.slit.coordinates3D())



#close()

def Plot3D():
    fig = figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_title('geometry 3D')
    ax.plot(xpos_slit/100,ypos_slit/100, 0,'ro')
    for i in range(10):
        ychords = geom['ychord'][:,i*Ndiods:(i+1)*Ndiods]/100
        xchords = geom['xchord'][:,i*Ndiods:(i+1)*Ndiods]/100
        #ax.plot(xchords,ychords,0*xchords,'k')

        C = camera(i)
        C.show3D(ax)

        C.chip.show3D(ax)
        C.slit.show3D(ax)


    #axis('equal')
    #coord= [0.624, 1.136, -0.75, 0.75]
    ax.set_xlim(coord[0],coord[1])
    ax.set_ylim(coord[2],coord[3])
    ax.set_zlim(-.1,.1)

    ##################################  calculate 3D coordinates of the slits+ detectors ########################

    ax.plot(vessel_in[0], vessel_in[1],zeros_like(vessel_in[1]),'k')
    ax.plot(vessel_out[0], vessel_out[1],zeros_like(vessel_out[1]),'k')

    show()
    return fig



diod_arrays = []
slits = []
for i in range(10):
    C = camera(i)
    diod_arrays.append(C.chip.coordinates3D())
    slits.append(C.slit.coordinates3D())



