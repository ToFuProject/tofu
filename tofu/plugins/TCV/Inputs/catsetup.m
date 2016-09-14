function catsetup; 

% file which correspond to the MA's x_tomo_geometry.m ; 
% sets the SXR detectors' geometry and saves it into the file
% cat_defaults.mat

% GT 2008

% in xtomo_geometry.m:   "load tcvvesc1" and "load angular_fact_1" 

% ---------------- 

% ---------------- camera defaults
i_detec=menu('which system?','old (10 cameras)','Very Old','New'); % 1 for old setup

f_cam=[1 1 1 1 1 1 1 1 1 1];
fans=[1 1 1 1 1 1 1 1 1 1]; 

nl=200;

f_chan=[];
for k=1:length(f_cam)
    f_chan=[f_chan,f_cam(k)*ones(1,20)];
end
% remark: other channels have broken down later, sometimes intermittently. Selection in main program)

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
    %load tcv_vesc1
    disp('load the local space file')
end 
% ------- data, setup and calibration defaults

sigm=2.5e-2;

nx=15; 

corr=ones(200,1);
firstandlast=find(rem(1:200,20)==0 | rem(1:200,20)==1);
corr(firstandlast)=ones(size(firstandlast))*1.05;
% 5% larger surface in the edge cells of each SXR detector array

% ------- reconstruction parameters defaults

lambda_init=0.1;   % first smoothing fact lambda into iterations
ifish=3;           % number of min.fisher loops
iimax=4;           % iteration steps to find min chi2

% ----------------- start of xtomo_geometry


% according to [anton.public] function xtomo_geometry.m;
% 
%
%	outputs:
%
%  fans:    camera switch, 1=on,0=off (1x10)
%  vangle:  angle between detect. surface normal and pos. x-axis (1x10)
%  xchord:  two x-coordinates (2xnl) and
%  ychord:  two y-coord. for each line (2xnl), they specify start + end points
%  aomega:  etendue in mm^2 x steradians
%  angfact: angular factors, inverse of relative etendue (throughput) (20x10)
%
% 	uses tcv_vesc1.mat	angular_fact_1.mat  
%
%---------------- M.Anton 14/3/95 -------------------------------------------

% ======== tokamak parameters ================================================


xcont=rzvin(:,1);
ycont=rzvin(:,2);
xmin=min(xcont);
ymin=min(ycont);
xmax=max(xcont);
ymax=max(ycont);
xedge=100;
yedge=60;


% =========  detector parameters =============================================

cw=1;                         % detector numbers cw=1:clockwise cw=0:ccw
if i_detec==1
    vangle=[90 90 90 0 0 0 0 -90 -90 -90];
    % angle of detector surface normal 
    xpos=[71.5 87.7 103.9 123.1  123.1 123.1 123.1 104.04 87.84 71.64]; 
    % x position of the diaphragmas in [cm]
    ypos=[80.35 80.35 80.35 48.1 1.95 -2.45 -48.6 -80.35 -80.35 -80.35];
    % y position of the diaphragmas in [cm]
    % ae=[-8 0 8 5 9 -9 -5 8 0 -8]/10; %excentricity array/diaphragm [cm] (nominal HW 15.11.05)
    % ae= ae + [-0.0915 0   0.1361    0.2123    0.0923   -0.0994   ...
    %	 0.0872   -0.1520  0  0.9410 ]/10; % first attempt to adjust?
    % ae(1)=ae(1)+0.1/10;
    % ae(3)=8/10+0.14/10;
    % ae(4)=4.9/10;
    % ae(5)=9/10+0.2/10;
    % ae(6)=ae(6)-0.2/10;
    % ae(7)=-4.9/10;
    % ae(10)=-7.1/10; % history it seems, HW 15.11.05
    ae=[-0.79915 0 0.8140 0.4900 0.9200 -0.92994 -0.4900 0.7848 0 -0.7100];%excentricity array/diaphragm [cm] HW 15.11.2005 
    % I believe the changing values for ae reflect an empirical consistency iteration, HW 2005 
    da=   [12.4 9.9 12.4 9.9 13.4 13.4  9.9   -12.4  -9.9 -12.4]/10;
    % diaphragma-array distance in [cm] (poloidal)
    %%%  da2=[37 34.4 37 55.9 59.4 59.4 55.9 37 34.4 37]/10;
    % dist to diaphragm in toroidal direction [cm];
    %%%  deltada=[   -0.0311 0  -0.0458   -0.1179   -0.0615   -0.1105 ...
    %%%		  -0.0510   -0.0515  0  -0.3223]/10;
    %%% deltada(4)=0;
    %%% deltada(6)=0;
    %%%  da=da+ deltada;
    %%% da2=da2+deltada;
    %%%% da, da2 are not used, not clear what they are (heat shield apertures?), HW 15.11.05
    d1=0.90;			 % detector width      in mm
    d2=4.0;			 % detector length     in mm
    b1=0.800;			 % aperture width      in mm (pol.)
    b2=[8 8 8 15 15 15 15 8 8 8];			 
    % aperture length     in mm (tor.)
    b3x=0.020;			 % aperture thickness  in mm (poloidal)
    b3y=0;			 % aperture thickness  in mm (toroidal)
elseif i_detec==3
    % changes input 30th April 2008 by GPT
    cw=1;                         % detector numbers cw=1:clockwise cw=0:ccw
    
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
    
    d1=0.90;			 % detector width      in mm
    d2=4.0;			 % detector length     in mm
    b1=0.800;			 % aperture width      in mm (pol.)
    b2=[8 8 8 15 15 15 15 8 8 8];			 
    % aperture length     in mm (tor.)
    b3x=0.020;			 % aperture thickness  in mm (poloidal)
    b3y=0;			 % aperture thickness  in mm (toroidal)
end
% keyboard

%======== calculation of the chords of view ===================================

nact=sum(fans);
iact=find(fans);
ndet=20;
ncam=10;


% ---- apertures: ------------------

xap=ones(ndet,1)*xpos(iact);
xap=xap(:)';
yap=ones(ndet,1)*ypos(iact);
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
% dete are the distances of elements from normal to aperture
dum_ae=dete(:)';
dum_aey=detey(:)';


dum_vangle=ones(ndet,1)*vangle(iact);
dum_vangle=dum_vangle(:)';


ivert=find(dum_vangle>=60 | dum_vangle<=-60);
ihori=find(dum_vangle==0);

dum_da=ones(ndet,1)*da(iact);
dum_da=dum_da(:)';

dxd=zeros(1,ndet*nact);
dyd=zeros(1,ndet*nact);

dxd(ivert)=dum_ae(ivert);
dxd(ihori)=dum_da(ihori);

dyd(ivert)=dum_da(ivert)+dum_aey(ivert); 
dyd(ihori)=dum_ae(ihori);
% dum_da are the distances of elements from normal to aperture
xdet=xap+dxd;
ydet=yap+dyd;


% ---- calculate the equations of lines of sight

m=(ydet-yap)./(xdet-xap);
b=ydet-m.*xdet;

nl=length(xdet);
xchord=zeros(2,nl);
ychord=zeros(2,nl);


xchord(1,:)=xdet;ychord(1,:)=ydet;
keyboard
iup=find(dum_vangle>=60);
isi=find(dum_vangle==0);
ido=find(dum_vangle<=-60);


if ~isempty(iup)
    ychord(2,iup)=ymin*ones(size(iup));
    xchord(2,iup)=(ychord(2,iup)-b(iup))./m(iup);
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

xchord(:,141:160)=xchord(:,160:-1:141); ychord(:,141:160)=ychord(:,160:-1:141);   %for historical reasons
xchord(:,161:180)=xchord(:,180:-1:161); ychord(:,161:180)=ychord(:,180:-1:161);
xchord(:,181:200)=xchord(:,200:-1:181); ychord(:,181:200)=ychord(:,200:-1:181);

%======== prepare output ======================================================

vangle=vangle(iact);


% ----------------- end of xtomo_geometry



f_cam=[1 1 1 1 1 1 1 1 1 1];
% within the inherited algorithm, this is the simplest trick how to
% keep the loadable Tmatrix and PENTLAND data. The wrong channels
% are deleted by f_chan, anyway. But I keep this variable for
% possible upgrades.

if i_detec==1
    load cat_rzao        % rzvin rzvout aomega (what the heck are the former two? HW)
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
end

aomega=aomega/100; 	% conversion mm2 -> cm2
indm=min(find(aomega==max(aomega(:)))); %index of the maximum
aomegan=aomega/aomega(indm); % normalize the omega factors
nonz=find(aomega);
angfact(nonz)=ones(size(nonz))./aomega(nonz);
angfact=round(1000*angfact)/1000;
angfact(find(aomega))=1./aomega(find(aomega));
name=input('File name [cat_defaults]: ');
eval(['save ',name,' f_cam vangle xchord ychord angfact nl f_chan sigm nx corr lambda_init ifish iimax'])



