

from numpy import *
from matplotlib.pylab import *
from scipy.io import loadmat

#he forget 1.5mm shift in camera 9!!!
#dZ   = {1:6.1, 2:0 ,3:-6 ,4:5 ,5:9 ,6:-9 ,7:-5 ,8:7 ,9:1.5 ,10:-5.5}  #values from technical drawing
dZ   = {1:6.09, 2:0 ,3:-6.01 ,4:5 ,5:9 ,6:-9 ,7:-5 ,8:6.85 ,9:1.5 ,10:-5.6}  #valus used by previos student (cam 9 is corrected)
dR   = {1:17.93, 2:15.84 ,3:16.06 ,4:9.9 ,5:13.4 ,6:13.4 ,7: 9.9,8:22.41 ,9:21.78 ,10:25.44}#values from technical drawing
tilt = {1:18., 2:0 ,3:-21 ,4: 0,5: 0,6:0 ,7:0 ,8:17 ,9:0 ,10:-12}#values from technical drawing

#chamber boundary
coord= [0.624, 1.136, -0.75, 0.75]

#slit
d1=0.90;                     # detector width      in mm
d2=4.0;                      # detector length     in mm
sep=0.05;                    # detector width separation  in mm
Ndiods = 20;
b1=0.800;                    # aperture width      in mm (pol.)
b2=[8,8,8,15,15,15,15,8,8,8];# aperture length     in mm (tor.)
b3x=0.020;                   # aperture thickness  in mm (poloidal)
b3y=0;                       # aperture thickness  in mm (toroidal)


vangle=[108,90,69,0,0,0,0,-73,-90,-102];
# angle of detector surface normal

xpos_slit=r_[71.5,87.7,103.9,123.1,123.1,123.1,123.1,104.04,87.84,71.64]
# x position of the slits in [cm]

ypos_slit=r_[80.35,80.35,80.35,48.1,1.95,-2.45,-48.6,-80.35,-80.35,-80.35];
# y position of the slits in [cm]

ae=r_[-6.09,0,6.01,5,9,-9,-5,6.85,1.5,-5.6]/10 #cam 9 is corrected
# excentricity array/slit [cm]

da=   r_[17.93,15.84,16.06,9.9,13.4,13.4, 9.9,-22.41,-21.78,-25.44]/10;
# slit-array distance in [cm] (poloidal)

da2= r_[37,34.4,37,55.9,59.4,59.4,55.9,37,34.4,37]/10;
# dist to slit in toroidal direction [cm];?????


#What it is?? wery small except of the last detector
deltada=r_[   -0.0311 ,0,  -0.0458  , -0.1179   ,-0.0615  , -0.1105,
        -0.0510  , -0.0515 , 0 , -0.3223]/10
deltada[3]=0;
deltada[5]=0;
da+= deltada;
da2+=deltada;








class slit:
    def __init__(self, num, angle):

        self.width = b1*1e-3
        self.height = b2[num-1]*1e-3
        self.thickness =  b3x*1e-3

        self.pos = xpos_slit[num-1]/100,ypos_slit[num-1]/100
        self.angle = angle


    def show(self):

        slit1,slit2 = self.coordinates3D()
        plot(slit1[:,0],slit1[:,1])
        plot(slit2[:,0],slit2[:,1])


    def coordinates3D(self):

        dx = cos(self.angle)*self.width/2
        dy = sin(self.angle)*self.width/2
        dx_ = sin(self.angle)*self.thickness/2
        dy_ = cos(self.angle)*self.thickness/2

        A1 = [self.pos[0]-dx, self.pos[1]-dy, self.thickness/2]
        A2 = [self.pos[0]+dx, self.pos[1]+dy, self.thickness/2]
        A3 = [self.pos[0]+dx, self.pos[1]+dy, -self.thickness/2]
        A4 = [self.pos[0]-dx, self.pos[1]-dy, -self.thickness/2]

        slit1 = vstack((A1,A2,A3,A4))+r_[dx_,dy_,0]
        slit2 = vstack((A1,A2,A3,A4))-r_[dx_,dy_,0]

        return slit1, slit2



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
        for d in diods:
            plot(d[:,0],d[:,1])


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

        return diods_coord





class camera:
    def __init__(self,num):
        self.num = num
        if num > 3 and num < 8 :
            self.camera_angle = pi/2*3
        elif num <= 3:
            self.camera_angle = 0
        elif num >= 8:
            self.camera_angle = pi


        self.dR = dR[num]*1e-3
        self.dZ = dZ[num]*1e-3
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


#geom=loadmat('./geometry/TCV/XTOMO/cat_defaults2008.mat')
geom=loadmat('./cat_defaults2008.mat')


def PlotAll():

    f1 = figure()
    old_etendue  = 1/geom['angfact']/1e4
    for i in range(10):
        plot(arange(Ndiods)+i*Ndiods+.5,old_etendue[:,i])
    [axvline(x=i*Ndiods) for i in range(10)]
    title('old etendue')
    ylabel('etendue')
    xlabel('detetor')


    f2 = figure()
    title('geometry')
    plot(xpos_slit/100,ypos_slit/100,'ro')

    for i in range(1,11):
        ychords = geom['ychord'][:,(i-1)*Ndiods:i*Ndiods]/100
        xchords = geom['xchord'][:,(i-1)*Ndiods:i*Ndiods]/100
        plot(xchords,ychords,'k',lw=.2)

        C = camera(i)
        C.plot_los()
        C.chip.show()
        C.slit.show()

    axis('equal')
    show()
    return f1, f2

#################################  calculate 3D coordinates of the slits+ detectors ########################

diod_arrays = []
slits = []
for i in range(1,11):
    C = camera(i)
    diod_arrays.append(C.chip.coordinates3D())
    slits.append(C.slit.coordinates3D())





