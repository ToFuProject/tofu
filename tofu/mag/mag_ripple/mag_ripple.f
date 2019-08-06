      subroutine mag_ripple(brr, bt, bzz, RP, PHIP, ZP, n_in,
     &                      WITOR, k_in)
c

      IMPLICIT REAL*8 (A-H,O-Z)
      IMPLICIT integer (I-N)

      integer ii
      integer kk

      integer n_in
      integer k_in
      double precision brr(k_in, n_in), bt(k_in, n_in), bzz(k_in, n_in)
      double precision RP(n_in), PHIP(n_in), ZP(n_in)
      double precision WITOR(k_in)

Cf2py intent(in) RP, PHIP, ZP, WITOR
Cf2py depend(RP) n_in
Cf2py depend(WITOR) k_in
Cf2py intent(out) brr, bt, bzz

C
C      write(*,*) "Magnetic ripple computation"
C

      do kk=1,k_in
          do ii=1,n_in
              call champs(bt(kk, ii), brr(kk, ii), bzz(kk, ii),
     &                    RP(ii), ZP(ii), PHIP(ii), WITOR(kk))
          enddo
      enddo

      return
      end

      subroutine champs(bt,brr,bzz,RP,ZP,PHIP,WITOR)


        double precision bt,RP,ZP
        double precision PHIP,WITOR
        double precision brr,bzz,BF,BR,BZ,BMOD,BTF
        common /bobine/ RB,WIBOB

	PI     = 3.14159265359
	DEGRAD = PI/180.
C
C	LE CHAMP EST CALCULE POUR UN COURANT DE 1000 A DANS LE SUPRA
C
	RB     = 2.443
	RB2    = RB*RB
C
C	WIBOB EST LE COURANT DANS UNE BOBINE TOROIDALE * MU0/4PI
C

	WIBOB  = WITOR*2028*1E-7
	EL     = 1.6022E-19
	WMP    = 1.6726E-27
	WME    = 9.1095E-31


        CALL CS(RP,PHIP,ZP,BR,BF,BZ,BMOD,BTF)
	bt     = BF
	brr    = BR
	bzz    = BZ

        return
	end

C
C---------------------------------------------
	SUBROUTINE CS(R,F,Z,BR,BF,BZ,BMOD,BTF)
C---------------------------------------------
C

C
C	CALCUL DES COMPOSANTES DU CHAMP MAGNETIQUE
C
        double precision R,F,Z,BF,BR,BZ,BMOD,BTF
        double precision BTR,BTZ,PI,DEGRAD
        common /bobine/ RB,WIBOB
C
	PI     = 3.14159265359
	DEGRAD = PI/180.
	BTR    = 0
	BTZ    = 0
	BTF    = 0
C
C	BOUCLE SUR LES SPIRES TOROIDALES (CALCUL COMPLET)
C
	DO 11 IS=1,18
C
	  FB=(IS*20-10)*DEGRAD
	  FD=FB+F
	  SFD=SIN(FD)
	  CFD=COS(FD)
	  H=R*SFD
	  D=R*CFD-RB
	  RHO=SQRT(Z*Z+D*D)
	  CALL BRHOBH(RHO,H,BRHO,BH)
	  BRHOHO=BRHO*D/RHO
	  BTR=BTR+(BH*SFD+BRHOHO*CFD)
	  BTZ=BTZ+BRHO*Z/RHO
	  BTF=BTF+(BH*CFD-BRHOHO*SFD)

 11	CONTINUE
C
	BTR=BTR*WIBOB
	BTF=BTF*WIBOB
	BTZ=BTZ*WIBOB

	BR=BTR
	BF=BTF
	BZ=BTZ
	BMOD=SQRT(BR*BR+BF*BF+BZ*BZ)
	END
C
C-----------------------------------------
	SUBROUTINE BRHOBH(RHO,H,BRHO,BH)
C-----------------------------------------
C
C	AB= PETIT RAYON DES BOBINES TOROIDALES
C
	DATA AB,AB2/1.2668,1.60478224/
C
	H2=H*H
	RHO2=RHO*RHO
	D=(AB-RHO)*(AB-RHO)+H2
	RHO2PH2=RHO2+H2
	WK=SQRT(4*AB*RHO/((AB+RHO)*(AB+RHO)+H2))
	C=WK/SQRT(AB*RHO)
	WI1=ELLIPK(WK)
	WI2=ELLIPE(WK)
	BRHO=(H/RHO)*C*(-WI1+WI2*(AB2+RHO2PH2)/D)
	BH=C*(WI1+WI2*(AB2-RHO2PH2)/D)

	END

C
C-------------------------
	FUNCTION ELLIPK(X)
C-------------------------
C
        DATA A0,A1,A2,A3,A4/
     +1.38629436112,0.09666344259,0.03590092383,
     +0.03742563713,0.01451196212/
        DATA B0,B1,B2,B3,B4/0.5,
     +0.12498593597,0.06880248576,
     +0.03328355346,0.00441787012/
        Y=1-X*X
        A=(((A4*Y+A3)*Y+A2)*Y+A1)*Y+A0
        B=(((B4*Y+B3)*Y+B2)*Y+B1)*Y+B0
	ELLIPK=A+B*LOG(1./Y)
	END
C
C-------------------------
	FUNCTION ELLIPE(X)
C-------------------------
C
        DATA A1,A2,A3,A4/
     +0.44325141463,0.06260601220,
     +0.04757383546,0.01736506451/
        DATA B1,B2,B3,B4/
     +0.24998368310,0.09200180037,
     +0.04069697526,0.00526449639/
        Y=1-X*X
        A=(((A4*Y+A3)*Y+A2)*Y+A1)*Y+1.
        B=(((B4*Y+B3)*Y+B2)*Y+B1)*Y
	ELLIPE=A+B*LOG(1./Y)
	END
