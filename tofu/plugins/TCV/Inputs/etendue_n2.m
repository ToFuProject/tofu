
 function AOMEGA=etendue_n2(b1x,b1y,b1z,b2x,b2y,b2z,z01,z02,X0,cw)

% -------[ANTON.ETENDUE]
%
% function 
%
%	AOMEGA=etendue_n2(b1x,b1y,b1z,b2x,b2y,b2z,z01,z02,X0,cw);
%
% calculates 'etendue geometrique' A*OMEGA for two rectangular apertures
%
% 		! ALL LENGTHS IN  MM !
%
% input:	b1x b1y b1z	: width, height and thickness of ap. 1
%		b2x b2y b2z	: 				 ap. 2
%		z01		: distance detector - ap.1
%		z02		: distance detector - ap.2
%		X0		: shift centers of detector arr. - ap1
%		cw		: det # clockwise if cw==1
%
% output:	AOMEGA		: etendue A x OMEGA in mm^2 x steradians (20x1)
%
%------M.Anton 24.8.93 & 29.9.1994 & 11.4.1995 ----------------------------

% number of steps for the integration 


%          b1x=0.8;
%          b1y=6;
%          b1z=0.02;
%          b2x=10000000;
%          b2y=[8 8 8 15 15 15 15 8 8 8]/10;
%          b2z=0;
%          z01=abs(da(iact(l))*10);
%          z02=abs(da2(iact(l))*10);
%          X0=ae(iact(l))*10*vorz(iact(l));

nx=45;
ny=100;

% detector parameters:

wx=0.9;			% width of active diode area
wy=4.0;			% height of active diode area

NDIODES=20;			% number of diodes
SPACE=0.95;			% spacing between diodes (LD20-5T) in mm



% positions of the single diodes

if cw==1
   x0=X0-(NDIODES-1)/2*SPACE:SPACE:(X0+(NDIODES-1)/2*SPACE);
else
   x0=X0+(NDIODES-1)/2*SPACE:-SPACE:(X0-(NDIODES-1)/2*SPACE);
end


% angles of incidence for the center of the diodes

alpha0=atan(x0/z01);

% coordinates for the integration

dx=wx/nx;
dy=wy/ny;
x=-wx/2+dx/2:dx:+wx/2-dx/2;
y=-wy/2+dy/2:dy:wy/2-dy/2;


%-----------------zeroeth order-----------------------------------------------


	BX=b1x;
	BY=min(b1y,b2y*z01/z02);        	
	r0square=z01^2+x0.^2;
	AOM0a=wx*wy*BX*BY*cos(alpha0).^2./r0square;
	AOM0b=wx*wy*BX*BY*cos(alpha0)....
                .*(cos(alpha0)-b1z/b1x*abs(sin(alpha0)))./r0square;



%------------------numerical integration--------------------------------------


%------------------loop over diodes-------------------------------------------

  % calculate  solid angles for nxny elements dxdy(x,y)

  for ix= 1:NDIODES

	BX=b1x;
	BY=min( (b2y/2-y')*z01/z02+y', ones(size(y'))*b1y/2 )-....
	   max( (-b2y/2-y')*z01/z02+y', -ones(size(y'))*b1y/2 );

	TANA=(x0(ix)-x)/z01;
	TANB=y/z02;
	alpha=atan(TANA);
	beta=atan(TANB);
        SINABS=abs(sin(alpha));
	SINBBS=abs(sin(beta));
	COSA=cos(alpha);
	COSB=cos(beta);

	Rsquare=y'.^2*ones(size(x))+ones(size(y'))*(x0(ix)-x).^2+z01^2;

        % without considering thickness b1z of aperture

	OMEGAXY2a=BX*((BY.*COSB')*COSA).*(COSB'*COSA)./Rsquare;
	COMEGAXY2a=OMEGAXY2a;
	AOM2a(ix)=sum(sum(dx*dy*COMEGAXY2a));

 	% considering thicknesses b3x,b3y of aperture

	OMEGAXY2b=BX*((BY.*COSB')*COSA).*...
	(COSB'*(COSA-b1z/b1x*SINABS))./Rsquare;
	COMEGAXY2b=OMEGAXY2b;
	AOM2b(ix)=sum(sum(dx*dy*COMEGAXY2b));

  end  % end of loop over diodes
%-----------------------------------------------------------------------------


  AOMEGA=[AOM2b',AOM2a',AOM0a',AOM0b'];

%---c'est fini----------------------------------------------------------------


