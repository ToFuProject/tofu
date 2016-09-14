function [fans,vangle,xchord,ychord,aomega,angfact]=xtomo_geometry(i_detec,fans,flag)

%
%   [fans,vangle,xchord,ychord,aomega,angfact]=xtomo_geometry(i_detec,fans,flag);
% 
%	INPUTS:: 
%        i_detec:	=2: Xtomo prototype cameras (shot# < 6768)
%                       =1: Xtomo 9-cameras	    (shot# > 682x)
%			=3: Xtomo 10 cameras upgrade 2008 (shot > 34600)
%	 fans           = array containing 1 for active fans, 0 for non active
%	 flag	  	= empty or 0 for not saving the results,
%			= 1 to save cat_defaults
%
%	OUTPUTS:
%        fans:    camera switch, 1=on,0=off (1x10)
%        vangle:  angle between detect. surface normal and pos. x-axis (1x10)
%        xchord:  two x-coordinates.
%        ychord:  two y-coordinates. 
%                 For each line (2xnl), they specify start + end points
%        aomega:  etendue in mm^2 x steradians
%        angfact: angular factors, inverse of relative etendue (throughput) (20x10)
%
%	Subroutines:
%		          etendue_n2.m;
%		          angular_fact_*.mat , '*'=i_detec
%  
%
% This routine works on both Matlab4, Matlab5 and Matlab6
% Original routine for Matlab4 by Anton Mathias.  
%
% Last update: 2008
%
%-------------MAC:[FURNO.MATLAB5.XTOMO]----------------------------------

%addpath /mac/furno/MATLAB5/XTOMO/

disp('*----------------------------*')
disp('|   this is xtomo_geometry   |')
disp('*----------------------------*')

global xap yap xdet ydet
global ae da

% ======== tokamak parameters =======================

machine=computer; % processor type

try
    if strcmp(machine(1:2),'AX')
        load mac:[furno.matlab5.xtomo]tcv_vesc1
    elseif strcmp(machine(1:2),'SO')
        load /tcv/furno/matlab5/xtomo/tcv_vesc1
    elseif strcmp(machine(1:2),'IB')
        load /mac/furno/MATLAB5/XTOMO/tcv_vesc1
    end
catch
    load tcv_vesc1
    disp('load the local space file')
end 
if ~exist('rzvin')
  load tcv_vesc1
end


xcont=rzvin(:,1);
ycont=rzvin(:,2);
xmin=min(xcont);
ymin=min(ycont);
xmax=max(xcont);
ymax=max(ycont);
xedge=100;
yedge=60;




% =========  detector parameters =============================================

if i_detec==2
    
    cw=1;                         % detector numbers cw=1:clockwise cw=0:ccw
    
    if nargin<2
        fans=[0 0 0 0 0 0 1 0 1 0];   % camera switch
    end
    
    vangle=[90 90 90 0 0 0 0 -90 -90 -90];
    % angle of detector surface normal 
    xpos=[0 0 0 0 0 0  118.05 0 87.84 0]; 
    % x position of the diaphragmas in [cm]
    ypos=[0 0 0 0 0 0 -46 0 -80.45 0];
    % y position of the diaphragmas in [cm]
    ae=[0 0 0 0 0 0 -2.5 0 -0.1 0]/10; 
    %excentricity array/diaphragm in [cm]
    da=[0 0 0 0 0 0 10.1  0 -11.7 0]/10;   
    % diaphragma-array distance in [cm]
    da2=da;
    
    d1=0.950/10;                     % detector width      in mm
    d2=4.050/10;                     % detector length     in mm
    b1=1.000/10;                     % aperture width      in mm
    b2=4.000*ones(1,10)/10;          % aperture length     in mm
    b3x=0;                        % aperture thickness  in mm
    b3y=0;
    
elseif i_detec==1
    
    cw=1;                         % detector numbers cw=1:clockwise cw=0:ccw
    
    if nargin<2	
        fans=[1 1 1 1 1 1 1 1 1 1];   % camera switch
    end
    
    vangle=[90 90 90 0 0 0 0 -90 -90 -90];
    % angle of detector surface normal 
    xpos=[71.5 87.7 103.9 123.1  123.1 123.1 123.1 104.04 87.84 71.64]; 
    % x position of the diaphragmas in [cm]
    ypos=[80.35 80.35 80.35 48.1 1.95 -2.45 -48.6 -80.35 -80.35 -80.35];
    % y position of the diaphragmas in [cm]
    ae=[-8 0 8 5 9 -9 -5 8 0 -8]/10; %excentricity array/diaphragm [cm]
    
    
    ae= ae + [-0.0915 0   0.1361    0.2123    0.0923   -0.0994   ...
            0.0872   -0.1520  0  0.9410 ]/10;
    ae(1)=ae(1)+0.1/10;
    ae(3)=8/10+0.14/10;
    ae(4)=4.9/10;
    ae(5)=9/10+0.2/10;
    ae(6)=ae(6)-0.2/10;
    ae(7)=-4.9/10;
    ae(10)=-7.1/10;
    
    da=   [12.4 9.9 12.4 9.9 13.4 13.4  9.9   -12.4  -9.9 -12.4]/10;
    % diaphragma-array distance in [cm] (poloidal)
    da2=[37 34.4 37 55.9 59.4 59.4 55.9 37 34.4 37]/10;
    % dist to diaphragm in toroidal direction [cm];
    deltada=[   -0.0311 0  -0.0458   -0.1179   -0.0615   -0.1105 ...
            -0.0510   -0.0515  0  -0.3223]/10;
    deltada(4)=0;
    deltada(6)=0;
    
    
    da=da+ deltada;
    
    da2=da2+deltada;
    
    
    d1=0.90/10;			 % detector width      in cm
    d2=4.0/10;			 % detector length     in cm
    b1=0.800/10;			 % aperture width      in cm (pol.)
    b2=[8 8 8 15 15 15 15 8 8 8]/10;			 
    % aperture length     in mm (tor.)
    b3x=0.020/10;			 % aperture thickness  in mm (poloidal)
    b3y=0;			 % aperture thickness  in mm (toroidal)
    
elseif i_detec==3
    % changes input 30th April 2008 by GPT
    cw=1;                         % detector numbers cw=1:clockwise cw=0:ccw
    
    if nargin<2	
        fans=[1 1 1 1 1 1 1 1 1 1];   % camera switch
    end
    
    vangle=[108 90 69 0 0 0 0 -73 -90 -102];
    % angle of detector surface normal 
    
    xpos=[71.5 87.7 103.9 123.1  123.1 123.1 123.1 104.04 87.84 71.64]; 
    % x position of the diaphragmas in [cm]
    
    ypos=[80.35 80.35 80.35 48.1 1.95 -2.45 -48.6 -80.35 -80.35 -80.35];
    % y position of the diaphragmas in [cm]
    
    ae=[-6.09 0 6.01 5 9 -9 -5 6.85 0 -5.6]/10; 
    %excentricity array/diaphragm [cm]   
    
    da=   [17.93 15.84 16.06 9.9 13.4 13.4  9.9  -22.41  -21.78 -25.44]/10;
    % diaphragma-array distance in [cm] (poloidal)
    
    da2=[37 34.4 37 55.9 59.4 59.4 55.9 37 34.4 37]/10;
    % dist to diaphragm in toroidal direction [cm];
    
    deltada=[   -0.0311 0  -0.0458   -0.1179   -0.0615   -0.1105 ...
            -0.0510   -0.0515  0  -0.3223]/10;
    deltada(4)=0;
    deltada(6)=0;   
    
    da=da+ deltada;
    
    da2=da2+deltada;   
    
    d1=0.90/10;			 % detector width      in cm
    d2=4.0/10;			 % detector length     in cm
    b1=0.800/10;			 % aperture width      in cm (pol.)
    b2=[8 8 8 15 15 15 15 8 8 8]/10;			 
    % aperture length     in cm (tor.)
    b3x=0.020/10;			 % aperture thickness  in cm (poloidal)
    b3y=0/10;			 % aperture thickness  in cm (toroidal)
end


%======== calculation of the chords of view ===================================

nact=sum(fans);
iact=find(fans);
ndet=20;
ncam=10;



% ---- apertures: ------------------

xap=ones(ndet,1)*xpos(iact);    % x-coord of the aperture for every diode
xap=xap(:)';
yap=ones(ndet,1)*ypos(iact);    % y-coord forthe aperture for every diode
yap=yap(:)';

% ---- detectors: ------------------

vorz(find(vangle>=60))=(-1)^(cw+1)*ones(size(find(vangle>=60)));
vorz(find(vangle==0))=(-1)^cw*ones(size(find(vangle==0)));
vorz(find(vangle<=-60))=(-1)^cw*ones(size(find(vangle<=-60)));

for I=1:nact
  if iact(I)<=3 | iact(I)>=8
    dete(:,I)=cos((vangle(iact(I))-90)*pi/180)*(-9.025:0.950:9.025)'/10*vorz(iact(I))+ones(ndet,1)*ae(iact(I)); % X-position of diodes from center of array with eccentricity 
    detey(:,I)=sin((vangle(iact(I))-90)*pi/180)*(-9.025:0.950:9.025)'/10*vorz(iact(I));
  else
    dete(:,I)=(-9.025:0.950:9.025)'/10*vorz(iact(I))+ones(ndet,1)*ae(iact(I)); % X-position of diodes from center of array with eccentricity 
    detey(:,I)=zeros(20,1);
  end
end
% 0.95 distance between two adjacent centres od diodes. -9.025:9.025 way to
% create array of 20 diodes centred in the middle.
dum_ae=dete(:)';
dum_aey=detey(:)';

dum_vangle=ones(ndet,1)*vangle(iact);           % angle for diodes with respect to axis
dum_vangle=dum_vangle(:)';



ivert=find(dum_vangle>=60 | dum_vangle<=-60);
ihori=find(dum_vangle==0);

dum_da=ones(ndet,1)*da(iact);
dum_da=dum_da(:)';                              % distance diode-diaphgram 

dxd=zeros(1,ndet*nact);
dyd=zeros(1,ndet*nact);

dxd(ivert)=dum_ae(ivert);                       % x-coord of diodes cam 1,2,3,8,9,10 with rispect to aperture
dxd(ihori)=dum_da(ihori);                       % x-coord of diodes cam 4,5,6,7 with rispect to center aperture

dyd(ivert)=dum_da(ivert)+dum_aey(ivert);                       % y-coord of diodes cam 1,2,3,8,9,10 with rispect to center aperture
dyd(ihori)=dum_ae(ihori);                       % y-coord of diodes cam  4,5,6,7 with rispect to center aperture

xdet=xap+dxd;
ydet=yap+dyd;


%plot_vessel(rzvin,rzvout)
%hold on
% plot(xap,yap,'.g',xdet,ydet,'.m')


% ---- calculate the equations of lines of sight

m=(ydet-yap)./(xdet-xap);                       % slope of all the lines of sight
b=ydet-m.*xdet;                                 % coeff of eq y= mx + b

nl=length(xdet);
xchord=zeros(2,nl);                             % prepare chords
ychord=zeros(2,nl);


xchord(1,:)=xdet;ychord(1,:)=ydet;              % initial point of chords

iup=find(dum_vangle>=60);
isi=find(dum_vangle==0);
ido=find(dum_vangle<=-60);

if ~isempty(iup)   %here it calculates the real lines of sight
    ychord(2,iup)=ymin*ones(size(iup));         % for top cameras, yend it's the vessel
    xchord(2,iup)=(ychord(2,iup)-b(iup))./m(iup); % xend from line equation
end
if ~isempty(ido)
    ychord(2,ido)=ymax*ones(size(ido));
    xchord(2,ido)=(ychord(2,ido)-b(ido))./m(ido);
end
if ~isempty(isi)
    xchord(2,isi)=xmin*ones(size(isi));
    ychord(2,isi)=m(isi).*xchord(2,isi)+b(isi);
end

ileft=find(xchord(2,:)<xmin);

if ~isempty(ileft)
    xchord(2,ileft)=xmin*ones(size(ileft));
    ychord(2,ileft)=m(ileft).*xchord(2,ileft)+b(ileft);
end
irig=find(xchord(2,:)>xmax);
if ~isempty(irig)
    xchord(2,irig)=xmax*ones(size(irig));
    ychord(2,irig)=m(irig).*xchord(2,irig)+b(irig);
end
PrevCam=sum(fans(1:7)); %index of cameras loaded before 8,9,10
if isempty(PrevCam)
  PrevCam=0;
end
PrevCam=PrevCam*20;

if fans(8)
  xchord(:,1+PrevCam:20+PrevCam)=xchord(:,20+PrevCam:-1:1+PrevCam); 
  ychord(:,1+PrevCam:20+PrevCam)=ychord(:,20+PrevCam:-1:1+PrevCam);   %for historical reasons
  PrevCam=PrevCam+20;
end  
if fans(9)
  xchord(:,1+PrevCam:20+PrevCam)=xchord(:,20+PrevCam:-1:1+PrevCam); 
  ychord(:,1+PrevCam:20+PrevCam)=ychord(:,20+PrevCam:-1:1+PrevCam);   %for historical reasons
  PrevCam=PrevCam+20;
end  
if fans(10)
  xchord(:,1+PrevCam:20+PrevCam)=xchord(:,20+PrevCam:-1:1+PrevCam); 
  ychord(:,1+PrevCam:20+PrevCam)=ychord(:,20+PrevCam:-1:1+PrevCam);   %for historical reasons
end    

%======== prepare output ======================================================

vangle=vangle(iact);

%======== calculation of angular correction factors, if necessary =============

if i_detec==2 & exist('angular_fact_2.mat')==2
    
    disp('loading angular_fact_2')
    load angular_fact_2
    
    
%elseif i_detec==1 & exist('angular_fact_1.mat')==2
    
%    disp('loading angular_fact_1')
%    load angular_fact_1
    
else
    
    aomega=zeros(ndet,ncam);
    angfact=ones(ndet,ncam);
    
    
    for l=1:sum(fans)
        
        %		Z0X=abs(da(iact(l))*10)
        %	 	Z0Y=abs(da2(iact(l))*10)
        %		X0=ae(iact(l))*10 % back to mm, sorry about that...
        %		X0=X0*vorz(iact(l))
        %		B2=b2(iact(l))
        %		AOMEGA=etendue_n(b1,B2,b3x,b3y,Z0X,Z0Y,X0,cw);
        
        b1x=0.8;
        b1y=6;
        b1z=0.02;
        b2x=10000000;
        b2y=b2(iact(l));
        b2z=0;
        z01=abs(da(iact(l))*10);
        z02=abs(da2(iact(l))*10);
        X0=ae(iact(l))*10*vorz(iact(l));
        
        AOMEGA=etendue_n2(b1x,b1y,b1z,b2x,b2y,b2z,z01,z02,X0,cw);
        
        aomega(:,iact(l))=AOMEGA(:,1);
        
        
    end
    
    
    aomega=aomega/100; 	% conversion mm2 -> cm2
    indm=min(find(aomega==max(aomega(:)))); %index of the maximum
    aomegan=aomega/aomega(indm); % normalize the omega factors
    nonz=find(aomega);
    angfact(nonz)=ones(size(nonz))./aomega(nonz);
    
    angfact=round(1000*angfact)/1000;
    
    
    unitstring='units aomega: cm^2 * sterad';
 
    if nargin>2 
      if flag==1
        disp('saving the results ...')
        if i_detec==1
          save angular_fact_2001 angfact aomega unitstring
        elseif i_detec==2
          save angular_fact_2 angfact aomega unitstring
        elseif i_detec==3
          save angular_fact_2008 angfact aomega unitstring
        end
      else
        disp('did not save the results, use flag=1 otherwise')
      end
    end  
end
return

th=atan(diff(ychord)./diff(xchord));
thp=th;
neg=find(thp<0);
thp(neg)=180+thp(neg);
thdet=ones(1,20*sum(fans));
for k=1:sum(fans)
    thdet((k-1)*20+1:k*20)=vangle(k)*ones(1,20);
end
angles=thdet-thp;
mist=find(angles<0 & abs(angles)>90);
angles(mist)=angles(mist)+180;
th_inc=angles*pi/180;


% ---- correct for the edges of tcv ( some chords may be too long )





down=find(xcont>xedge & ycont<-yedge);
up=find(xcont>xedge & ycont>yedge);
cd=polyfit(xcont(down),ycont(down),1);
cu=polyfit(xcont(up),ycont(up),1);


iu1=find(xchord(1,:)>xedge & ychord(1,:)>0 & dum_vangle==-90 );
if ~isempty(iu1)
    xchord(1,iu1)=-(b(iu1)-cu(2))./(m(iu1)-cu(1)+eps);
    ychord(1,iu1)=m(iu1).*xchord(1,iu1)+b(iu1);
end

iu2=find(xchord(2,:)>xedge & ychord(2,:)>0 & ychord(1,:) & ....
    dum_vangle==-90);
if ~isempty(iu2)
    xchord(2,iu2)=-(b(iu2)-cu(2))./(m(iu2)-cu(1)+eps);
    ychord(2,iu2)=m(iu2).*xchord(2,iu2)+b(iu2);
end

id1=find(xchord(1,:)>xedge & ychord(1,:)<0 & dum_vangle==90);
if ~isempty(id1)
    xchord(1,id1)=-(b(id1)-cd(2))./(m(id1)-cd(1)+eps);
    ychord(1,id1)=m(id1).*xchord(1,id1)+b(id1);
end

id2=find(xchord(2,:)>xedge & ychord(2,:)<0 & dum_vangle==90);
if ~isempty(id2)
    xchord(2,id2)=-(b(id2)-cd(2))./(m(id2)-cd(1)+eps);
    ychord(2,id2)=m(id2).*xchord(2,id2)+b(id2);
end

ilow=find(ychord(1,:)<ymin);
ihig=find(ychord(1,:)>ymax);
ilef=find(xchord(1,:)<xmin);
irig=find(xchord(1,:)>xmax);
if ~isempty(ilow)
    ychord(1,ilow)=ymin*ones(size(ilow));
    xchord(1,ilow)=ymin./m(ilow)-b(ilow)./m(ilow);
end
if ~isempty(ihig)
    ychord(1,ihig)=ymax*ones(size(ihig));
    xchord(1,ihig)=ymax./m(ihig)-b(ihig)./m(ihig);
end
if ~isempty(ilef)
    xchord(1,ilef)=xmin*ones(size(ilef));
    ychord(1,ilef)=m(ilef)*xmin+b(ilef);
end
if ~isempty(irig)
    xchord(1,irig)=xmax*ones(size(irig));
    ychord(1,irig)=m(irig)*xmax+b(irig);
end


ilow=find(ychord(2,:)<ymin);
ihig=find(ychord(2,:)>ymax);
ilef=find(xchord(2,:)<xmin);
irig=find(xchord(2,:)>xmax);
if ~isempty(ilow)
    ychord(2,ilow)=ymin*ones(size(ilow));
    xchord(2,ilow)=ymin./m(ilow)-b(ilow)./m(ilow);
end
if ~isempty(ihig)
    ychord(2,ihig)=ymax*ones(size(ihig));
    xchord(2,ihig)=ymax./m(ihig)-b(ihig)./m(ihig);
end
if ~isempty(ilef)
    xchord(2,ilef)=xmin*ones(size(ilef));
    ychord(2,ilef)=m(ilef)*xmin+b(ilef);
end
if ~isempty(irig)
    xchord(2,irig)=xmax*ones(size(irig));
    ychord(2,irig)=m(irig)*xmax+b(irig);
end





