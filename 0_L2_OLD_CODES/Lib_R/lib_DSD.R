##################### refractive index for liquid water (Ray Model) ######################

refractive_index_water <- function(t,w) {

  ## Based on the work of P.Ray, Applied Optics Vol.8,p.1836-1844, 1972

  ## Inputs :
  ##    t = the temperature (in Celsius)
  ##    w = the wavelength (in cm)

  ## Outputs :
  ##   index = the complex refractive index m=m'+im'

  ## Source : Mark Hervig
  ## Modifications : A.Berne (May.2005), M.Schleiss (May.2008)
  
  dxr <- 0
  dxi <- 0
  
  if(t>50 || t< -20){stop("Temperature must be between -20 and 50 Celsius")}
  
  ## Parameters from eqn 7a-c : epsilon sub infinity, alpha,lambda
  ## Paramaters from eqn 4 : epsilon sub s
  cpf   <- 5.27137+0.021647*t-0.00131198*t^2
  alpha <- -16.8129/(t+273)+0.0609265
  wavs  <- 0.00033836*exp(2513.98/(t+273))
  cps   <- 78.54*(1.-4.579e-3 *(t-25.) + 1.19e-5 *(t-25.)^2 -2.8e-8 *(t-25.)^3)
  
  ## Complex permittivity cp=cpr-i*cpi from eqns 2&3
  term1 <- wavs/w
  cpr   <- cpf + (cps-cpf)/(1.+term1^2)
  cpi   <- (cps-cpf)*term1/(1.+term1^2)

  ## Complex Permitivity cp=cpr-i*cpi from eqns 5&6
  term1 <- (wavs/w)^(1-alpha)
  term2 <- (cps-cpf)*(1.+term1 * sin(alpha*pi/2.))
  term3 <- 1.+2.*term1*sin(alpha*pi/2.)+term1^2
  cpr   <- cpf + term2/term3

  term2 <- (cps-cpf)*term1 * cos(alpha*pi/2.)
  cpi   <- term2/term3+12.5664e8*w/18.8496e10

  ## Complex Refractive Indexes
  dxr <- +(((cpr^2+cpi^2)^0.5+cpr)/2)^0.5
  
  ## depending on the sign convention for m
  ## dxi <- -(((cpr^2+cpi^2)^0.5-cpr)/2)^0.5 # m'' < 0
  dxi <- +(((cpr^2+cpi^2)^0.5-cpr)/2)^0.5 # m'' > 0

  index <- complex(real=dxr,imaginary=dxi)
  return(index)
}


####################### refractive index for liquid water (Liebe Model) ##########################

ref_index_water_liebe <- function(t,f){

  ## Based on the work of H.J.Liebe, "A model for the complex permittivity of water at frequencies below 1 THz"
  ## More general than the Debye equation this formula is valid for frequencies up to 1THz

  ## Inputs : 
  ##   t = the temperature (in °C)
  ##   f = the frequency (in GHz)

  ## Outputs :
  ##   m = m' + im'' the complex refractive index

  ## Source : M.Schleiss (May.2008)
  ## Updated 06.12.2017 to replace 146.5 with 146.4 in Gamma_1, to align with Liebe article.

  Theta       <- 1 - 300/(273.15+t)
  Epsilon_0   <- 77.66 - 103.3*Theta
  Epsilon_1   <- 0.0671*Epsilon_0
  Epsilon_2   <- 3.52 + 7.52*Theta
  Gamma_1     <- 20.20 + 146.4*Theta + 316*Theta^2
  Gamma_2     <- 39.8*Gamma_1
  
  term1 <- Epsilon_0-Epsilon_1
  term2 <- 1+(f/Gamma_1)^2
  term3 <- 1+(f/Gamma_2)^2
  term4 <- Epsilon_1-Epsilon_2
  term5 <- Epsilon_2

  Epsilon_real <- term1/term2 + term4/term3 + term5
  Epsilon_imag <- (term1/term2)*(f/Gamma_1) + (term4/term3)*(f/Gamma_2)

  Epsilon <- complex(real=Epsilon_real,imaginary=Epsilon_imag)

  m <- sqrt(Epsilon)
  return(m)
}


############################# Terminal Velocity of a rain drops ##########################

drop_velocity <- function(D) {

  ## Based on the work of Beard, JAS Vol.34 1977, pp.1293-1298
  ## This is an approximation valid at Sea level : P=1atm ; T=20°C ; p=1.194 km/m3
  ## For more precise computations use the complete Beard Model in /USERS/lte/commun1/Prog_com/lib_R/Beard_Model.R

  ## Inputs :
  ##  D = the diameter of the drop (in mm)

  ## Outputs : 
  ##  V = the final drop velocity (in cm/s)

  ## Source : A.Berne, 2007
  ## Modifications : M.Schleiss (May.2008)
  
  D <- D/10
  x  <- log(D)
  X  <- c(x,x^2,x^3,x^4,x^5,x^6,x^7,x^8,x^9)
  C  <- c(1.74951,4.86324,6.60631,4.84606,2.14922,0.58714,0.096348,0.00869209,0.00033089)
  c0 <- 7.06037
  V  <- exp(c0 + sum(X*C))
  return(V)
}

############################# Terminal Mean Velocity of a snow flake ############################

snow_velocity <- function(D) {

	## Not yet done !
	return(1)
}


########################### Andsager-Beard Model for axis ratio of rain drops ###########################

axis_ratio_AB <- function(D){
  
  ## This is a mixture between Beard model and Andsager model
  ## Valid for drops with diameter between 0.1 and 7.0 mm

  ## Beard & Chuang, "A new model for the equilibrium shape of raindrops", JAS Vol.44 1987
  ## Andsager & Beard "Laboratory measurements of axis-ratios for large raindrops", JAS Vol.56 1999

  ## Inputs : 
  ##    D = the diameter of the drop (in mm)

  ## Outputs :
  ##    alpha = the ratio of the vertical/horizontal axis

  ## Source : M.Schleiss (May.2008)

  D <- D/10
  if(D<0.1 || D>0.4){ alpha <- 1.0048 + 0.0057*D - 2.628*D^2 + 3.682*D^3 - 1.677*D^4 }
  else{ alpha <- 1.012 - 0.144*D - 1.03*D^2 }
  alpha <- min(1,alpha)
  alpha <- max(1/5,alpha)
  return(alpha)
}

axis_ratio <- function(D){
  
  ## Inputs : 
  ##    D = the diameter of the drop (in mm)

  ## Outputs :
  ##    alpha = the ratio of the vertical/horizontal axis

  D <- D/10
  alpha <- 1.012 - 0.144*D - 1.03*D^2
  alpha <- min(1,alpha)
  alpha <- max(1/5,alpha)
  return(alpha)
}

########################## Rain Intensity for measured DSD at ground #########################

Rain_int_ground <- function(tab_N,tab_D,S,dt){

   ## Computes the empirical rain intensity derived from ground DSD

   ## Inputs :
   ##   tab_N = the number of drops (on the ground) in each diameter class
   ##   tab_D = the mean diameter classes (32 values for Parsivel)  [in mm]
   ##   S     = the surface of the disdrometer [in m^2]
   ##   dt    = the time between two successive measurements [in seconds]

   ## Outputs :
   ##   R = the rain intensity [in mm/h]

   R <- (1/1e6)*(1/S)*(3600/dt)*(pi/6)*sum(tab_N*tab_D^3)
   return(R)
   
}

########################## Rain Intensity for volumic DSD #########################

Rain_int_volume <- function(tab_N,tab_D,tab_V){

   ## Computes the rain intensity derived from volumic DSD

   ## Inputs :
   ##   tab_N = the volumic drop density in each diameter class [in #/m^3]
   ##   tab_D = the mean diameter classes (32 values for Parsivel)  [in mm]
   ##   tab_V = the mean velocity of the diameter class (32 values for Parsivel) [in m/s]

   ## Outputs :
   ##   R = the rain intensity [in mm/h]

   R <- (1/1e6)*3600*(pi/6)*sum(tab_N*tab_V*tab_D^3,na.rm=TRUE)
   return(R)
   
}

########################## Rain Intensity for Exponential DSD #########################

Rain_int <- function(Nt,lambda,tab_D,tab_V){

  ## Computes the rain intensity for a Marshall Palmer DSD N(D)=Nt*exp(-lambda*D)
    
  ## Inputs :
  ##  Nt     = the number of drops per m^3
  ##  lambda = the inverse of the mean drop size (in 1/mm)
  ##  tab_D  = a discretized collection of possible drop sizes (in mm)
  ##  tab_V  = the corresponding terminal drop velocities (in cm/s)
  ##
  ## Outputs :
  ##  R  = the rain intensity (in mm/h)
  ##
  ## Source : A.Berne, 2007
  ## Modifications : M.Schleiss, April 2008

  Cr <- 6*pi/(1e6)
  w  <- sum(exp(-lambda*tab_D),na.rm=TRUE)
  R  <- Cr*sum(tab_D^3*tab_V*Nt*exp(-lambda*tab_D))/w
  return(R)
}

##################### Rain Intensity for Gamma DSD #####################

Rain_int_gamma <- function(Nt,alpha,beta,seqD){

	## Computes the rain intensity for a fitted volumic Gamma DSD
	## This function is more general than Rain_int() and includes the Marshall Palmer DSD.
	
	## Inputs :
	##   Nt    = the fitted number of drops 
	##   alpha = the fitted shape parameter
	##   beta  = the fitted rate parameter
	##   seqD  = the diameter discretization for the numerical integration
	
	## Outputs :
	##   R = the rain intensity [in mm/h]

	## Last Modification : M.Schleiss, January 2009

	R  <- NA
	lD <- length(seqD)

	if(!is.na(Nt) && !is.na(alpha) && !is.na(beta) && lD>0){
		if(Nt>0 && Nt<Inf && alpha>0 && alpha<Inf && beta>0 && beta<Inf){
			tabV  <- sapply(seqD,drop_velocity)/100
			term1 <- (1/1e6)*3600*(pi/6)
			term2 <- sum(seqD^(alpha-1)*exp(-beta*seqD),na.rm=TRUE)
			term3 <- sum(Nt*seqD^(alpha-1)*exp(-beta*seqD)*seqD^(3)*tabV,na.rm=TRUE)
			R     <- term1*term3/term2
			if(!is.na(R)){
				if(R<0){R <- 0}
				if(R==Inf){R <- NA}
			}
		}
	}
	return(R)
}

########################## Equivalent Radar Reflectivity for Exponential DSD ##########################

Radar_ref <- function(Nt,lambda,tab_D,tab_BS,wlength,K_w) {

  ## Computes the radar reflectivity for a Marshall Palmer DSD

  ## Inputs :
  ##  Nt      = the number of drops per m^3
  ##  lambda  = the inverse of the mean drop size (in 1/mm)
  ##  tab_D   = a discretized collection of possible drop sizes (in mm)
  ##  tab_BS  = the corresponding backscattering (in cm^2) 
  ##  wlength = the wavelength (in cm)
  ##  K_w     = the dielectric constant (derived from the refractive index)

  ## Outputs :
  ##  Z_eq  = the equivalent radar reflectivity (in mm6/m3)

  ## Source : A.Berne, 2007
  ## Modifications : M.Schleiss, April 2008 

  Cz <- ( (1e6)*wlength^4 ) / ( pi^5*abs(K_w)^2 )
  w  <- sum(exp(-lambda*tab_D))
  Z_eq <- Nt*sum(tab_BS*exp(-lambda*tab_D))
  Z_eq <- Cz*Z_eq/w
  return(Z_eq)
}

############################# Equivalent Radar Reflectivity for Gamma DSD #########################

Radar_ref_gamma <- function(Nt,alpha,beta,tab_D,tabBS,wlength,K_w){

  ## This is the same function than Radar_ref() but for a Gamma DSD
  
  ## Inputs :
  ## Nt      = the number of drops per m^3
  ## alpha   = shape parameter of the DSD
  ## beta    = rate parameter of the DSD 
  ## tab_D   = a discretized collection of possible drop sizes [in mm]
  ## tab_BS  = the corresponding backscattering cross sections [in cm^2]
  ## wlength = the wavelength [in cm]
  ## K_w     = the dielectric constant (derived from the refractive index)

  ## Outputs :
  ## Z_eq = the equivalent radar reflectivity [in mm^6/m^3]

#   Cz   <- 1e6*wlength^4*Nt/(pi^5*abs(K_w)^2)
#   fD   <- dgamma(tab_D,shape=alpha,rate=beta)
#   idx     <- which(fD==Inf)
#   if(length(idx)>0){
#     fD[idx] <- rep(1e3,length(idx))
#   }
#   Z_eq <- (Cz/sum(fD,na.rm=TRUE))*sum(tabBS*fD,na.rm=TRUE)
#   return(Z_eq)

	Cz   <- 1e6*wlength^4*Nt/(pi^5*abs(K_w)^2)
	num  <- sum(tabBS*tab_D^(alpha-1)*exp(-beta*tab_D),na.rm=TRUE)
	I    <- sum(tab_D^(alpha-1)*exp(-beta*tab_D),na.rm=TRUE)
	if(is.na(Cz)==FALSE && is.na(num)==FALSE && is.na(I)==FALSE){
		if(I<Inf && num<Inf && I!=0){Z_eq <- Cz*num/I}
		else{Z_eq <- NA}}
	else{Z_eq <- NA}
	return(Z_eq)
}


################### Rayleigh Approximation for Radar Reflectivity (Exponential DSD) ################

Radar_ref_ray <- function(Nt,lambda,tab_D) {
 
  ## Computes the radar reflectivity based on the approximation of Rayleigh
  ## Only valid for a Marshall Palmer distribution

  ## Inputs :
  ##  Nt     = the number of drops per m^3
  ##  lambda = the inverse of the mean drop size (in mm)
  ##  tab_D  = a discretized collection of possible drop sizes (in mm)

  ## Outputs : 
  ##  Z_ray  = the Rayleigh approximation of the radar reflectivity (in mm6/m3)
  ##
  ## Source : A.Berne, 2007
  ## Modifications : M.Schleiss, April 2008

  w <- sum(exp(-lambda*tab_D))
  Z_ray <- Nt*sum(tab_D^6*exp(-lambda*tab_D))/w
  return(Z_ray)
}


##################### Rayleigh Approximation for Radar Reflectivity (Gamma DSD) #####################

Radar_ref_ray_gamma <- function(Nt,alpha,beta,tab_D){

  ## This is the same function than Radar_ref_ray but for a Gamma DSD

  ## Inputs :
  ##  Nt     = the number of drops per m^3
  ##  alpha  = the shape parameter of the DSD
  ##  beta   = the rate parameter of the DSD
  ##  tab_D  = a discretized collection of possible drop sizes (in mm)

  ## Outputs : 
  ##  Z_ray  = the Rayleigh approximation of the radar reflectivity (in mm6/m3)

  ## Source : M.Schleiss, June 2008

  fD    <- dgamma(tab_D,shape=alpha,rate=beta)
  Z_ray <- (Nt/sum(fD))*sum(tab_D^6*fD)
  return(Z_ray)
}


############################# Specific one-way attenuation ##################

Spec_att <- function(Nt,lambda,tab_D,tab_E) {
  
  ## Inputs :
  ##  Nt     = the number of drops per m^3
  ##  lambda = the inverse of the mean drop size (in ??)
  ##  tab_D  = a discretized collection of possible drop sizes (in ??)
  ##  tab_E  = tabular with the mean extinction cross-sections for each D in tab_D

  ## Outputs :
  ##  k  =  specific one-way attenuation (in dB/km)

  ## Source : A.Berne, 2007
  ## Modifications : M.Schleiss, April 2008

  Ck <- 1/log(10.)
  w  <- sum(exp(-lambda*tab_D))
  k  <- Ck*sum(tab_E*Nt*exp(-lambda*tab_D))/w
  return(k)
}


###################### Spherical Bessel functions of 1st and 2nd kind ####################

sph_Bessel_1 <- function(x,n) {
  return((pi/(2*x))^0.5*besselJ(x,nu=n+0.5)) 
}

sph_Bessel_2 <- function(x,n) {
  return((pi/(2*x))^0.5*besselY(x,nu=n+0.5) )
}


################## Extinction scattering, total scattering and backscattering ##################

mie_scat <- function(x,m,nang) {

  ## Calculates total scattering, backscattering and extinction efficiencies according to Mie Theory
  ## Only valid in the framework of spherical rain drops.

  ## Input :
  ##  x     =  kr = 2*pi*r/wavelength (r = particle radius in the same unit than the wavelength)
  ##  m     =  complex relative refractive index m = m'+im' (where m'' is positive)
  ##  nang  =  number of angles for integration (20 or more is OK)

  ## Output :
  ##  qsca  =  scattering efficiency
  ##  qext  =  extinction efficiency
  ##  qback =  backscattering

  ## Source : C.Maetzler, Mie Scattering on Matlab (2002)
  ## R Implementation : A.Berne, Apr. 2008
  ## Modifications : M.Schleiss, (May.2008)

  ## TO DO  : Scattering amplitude matrix S integrated over the angles.

  ## The Series are terminated after the evaluation of nmax terms
  ## See Bohren & Huffman (1983)

  xstop <- x + 4*x^(1/3) + 2
  nstop <- round(xstop)
  nmx   <- floor(max(c(xstop,abs(m*x)))) + 15

  d <- array(as.complex(0),dim=nmx)
  d[nmx] <- complex(real=0,imaginary=0)

  for (i in seq(nmx-1,1,-1)) {
     d[i] <- (i+1)/(m*x) - 1/(d[i+1]+(i+1)/(m*x))
  }

  psi0 <- sin(x)
  psi1 <- sin(x)/x-cos(x)
  ksi0 <- complex(real=psi0,imaginary=-cos(x))
  ksi1 <- complex(real=psi1,imaginary=(-sin(x)-cos(x)/x))

  qsca <- 0
  qext <- 0
  qbak <- 0

  for (n in 1:nstop) {

    psi <- x*sph_Bessel_1(x,n)
    ksi <- complex(real=psi,imaginary=x*sph_Bessel_2(x,n))

    an <- ( x*d[n]*psi + m*n*psi - m*x*psi0 ) / ( x*d[n]*ksi + m*n*ksi - m*x*ksi0 )
    bn <- ( m*x*d[n]*psi + n*psi - x*psi0 ) / ( m*x*d[n]*ksi + n*ksi - x*ksi0 )

    qsca <- qsca + (2*n+1)*(abs(an)*abs(an) + abs(bn)*abs(bn))
    qext <- qext + (2*n+1)*Re(an+bn)
    qbak <- qbak + (2*n+1)*(-1)^n*(an - bn)

    psi0 <- psi
    ksi0 <- ksi

  }

  qsca <- qsca * 2/(x*x)
  qext <- qext * 2/(x*x)
  qbak <- abs(qbak)*abs(qbak) * 1/(x*x)

  Q <- c(qsca,qext,qbak)
  return(Q)
}

############################ Grid Resolution change ####################################

change_resolution <- function(M,size){

  ## Computes a new grid with smaller resolution than the original
  ## The values of the new nodes are averaged over a local neighbourhood

  ## Inputs :
  ##   M    : a matrix with nx^2 rows and 3 columns
  ##          1st row = the x coordinates
  ##          2nd row = the y coordinates
  ##          3rd row = the node value
  ##   size : the number of nodes that have to be merged

  ## Outputs :
  ##   New_M : a matrix with (nx/size)^2 rows and 3 columns
  ##           1st row = the x coordinates
  ##           2nd row = the y coordinates
  ##           3rd row = the node value

  ## Source : M.Schleiss, April 2008

  nxy  <- dim(M)[1]
  nx   <- sqrt(nxy)
  minx <- M[1,1]
  miny <- M[1,2]
  maxx <- M[nxy,1]
  maxy <- M[nxy,2]
  dx   <- M[2,1]-M[1,1]
  dy   <- M[nx+1,2]-M[1,2]

  new_minx <- minx+(size-1)*dx
  new_miny <- miny+(size-1)*dy
  new_maxx <- maxx
  new_maxy <- maxy
  Seqx <- seq(new_minx,new_maxx,size*dx)
  Seqy <- seq(new_miny,new_maxy,size*dy)
  nSeqx <- length(Seqx)
  nSeqy <- length(Seqy)
  xy    <- expand.grid(Seqx,Seqy)
  New_M <- as.matrix(xy)
  new_nxy <- dim(New_M)[1]

  new_values <- rep(0,nSeqx*nSeqy)
  itrx <- 1
  itry <- 1
  for(i in 1:new_nxy){
     Z <- 0
     for(itr1 in itrx:(itrx+size-1)){
        for(itr2 in itry:(itry+size-1)){
           itr <- (itr2-1)*nx+itr1
           Z <- Z + M[itr,3]
        }
     }
     Z <- Z/(size^2)
     new_values[i] <- Z
     itrx <- itrx+size
     if(itrx>nx-size+1){
        itrx <- 1
        itry <- itry+size
     }
  }
  New_M <- cbind(New_M,new_values)
  return(New_M)
}

########################## Isotropic variogram of a 2D Field ##########################

isotropic_variogram <- function(M,dr,dmax){

	## Computes the isotropic variogram of 2D fields
	
	## Inputs :
	##  M     = a matrix with N rows and 3 columns
	##          1st column : the x coordinate
	##          2nd column : the y coordinate
	##          3rd column : the value of the field variable
	##  dr    = the width of the classes in the variogram
	##  dmax  = the maximum distance do be considered
	
	## Output :
	##   V   = a variogram matrix with Nv=dmax/dr rows and 4 columns
	##       1st column : the distance
	##       2nd column : the number of points N(i) in the class
	##       3rd column : the Sum of Squared Differences SSD
	##       4th column : the semi-variance gamma = SSD/2N(i)
	
	## Complexity : O(N^2) where N is the number of observations
	## Source : M.Schleiss, March 2009, 2nd version

  	dmax    <- floor(dmax/dr)*dr
	nV      <- dmax/dr
	V       <- matrix(seq(dr,dmax,dr),ncol=1)
	V       <- cbind(V,matrix(rep(0,2*nV),ncol=2))
	V       <- cbind(V,matrix(rep(NA,nV),ncol=1))
	tab_nNA <- which(!is.na(M[,3]))
	ltna    <- length(tab_nNA)
	
	for(itr in 1:(ltna-1)){
		i      <- tab_nNA[itr]
		tabj   <- tab_nNA[(itr+1):ltna]
		tab_dx <- M[tabj,1]-M[i,1]
		tab_dy <- M[tabj,2]-M[i,2]
		tab_dz <- (M[tabj,3]-M[i,3])^2
		tabD   <- sqrt(tab_dx^2+tab_dy^2)
		tab_index <- which(tabD>dmax)
		if(length(tab_index)>0){tabD[tab_index] <- dmax}
		tab_index <- which(tabD<dr)
		if(length(tab_index)>0){tabD[tab_index] <- dr}
		tabI <- round(tabD/dr)
		lti  <- length(tabI)
		for(k in 1:lti){
			index <- tabI[k]
			V[index,2] <- V[index,2] + 1
			V[index,3] <- V[index,3] + tab_dz[k]
		}
	}
	tab_index <- which(V[,2]>0)
	if(length(tab_index)>0){
		V[tab_index,4] <- V[tab_index,3]/(2*V[tab_index,2])
	}
 	return(V)
}

########################### CROSS-VARIOGRAM OF 2D GRID ######################

cross_variogram <- function(X,Y,dr,dmax){

   ## Computes the cross-variogram of two collocated 2D fields  
	
   ## Input  :
   ##  X      = a matrix with N rows and 3 columns
   ##          1st column : the x coordinate
   ##          2nd column : the y coordinate
   ##          3rd column : the value of the field variable
   ##  Y      = a matrix with N rows and 3 columns
   ##           1st column : the x coordinate (this is the same than for X)
   ##           2nd column : the y coordinate (this is the same than for Y)
   ##           3rd column : the value of the field variable
   ##  dr    = the width of the classes in the variogram
   ##  dmax  = the maximum distance do be considered

   ## Output :
   ##   V   = a variogram matrix with dmax/dr+1 rows and 4 columns
   ##       1st column : the distance
   ##       2nd column : the number of points in the class
   ##       3rd column : the Sum of Centered Products SCP
   ##       4th column : the cross-variance gamma = SCP/2N(i)

   ## Complexity : O(N^2) where N is the number of observations
   ## Source : M.Schleiss, June 2008

   dmax <- floor(dmax/dr)*dr
   N   <- dim(X)[1]
   N_V <- dmax/dr
   V   <- matrix(seq(dr,dmax,dr),ncol=1)
   V   <- cbind(V,matrix(rep(0,3*N_V),ncol=3))
   for(i in 1:N){
      for(j in 1:N){
         dist  <- sqrt((X[i,1]-X[j,1])^2 + (X[i,2]-X[j,2])^2)
         index <- ceiling(dist/dr)
         index <- min(c(index,dmax/dr))
         if(is.na(X[i,3]) || is.na(X[j,3])){}
         else{
            V[index,2] <- V[index,2] + 1
            V[index,3] <- V[index,3] + (X[j,3]-X[i,3])*(Y[j,3]-Y[i,3])
         }
      }
   } 	
  for(i in 1:N_V){
    if(V[i,2]==0){
      V[i,4] <- NA
    }
    else{
      V[i,4] <- V[i,3]/(2*V[i,2])
    }
  }
  return(V)
}





########################### Autocorrelation #############################

autocorrelation <- function(tabX,tabT,dr,dmax){

	## Computes the autocorrelation of a time series
	
	## Inputs:
	## tabX = vector of observations
	## tabT = vector of time stamps (in numeric format)
	## dr   = discretization of time
	## dmax = maximum time lag to be considered
	
	## Remarks:
	## (1) tabX and tabT must have the same size
	## (2) NA values are allowed

	## Outputs
	## M = Autocorrelation matrix
	## 1st column : lag   = considered time lag
	## 2nd column : Npts  = number of points (X,Y) for the considered lag
	## 3rd column : SumX  = Sum of X observations
	## 4th column : SumY  = Sum of Y observations
	## 5th column : SumXY = Sum of X*Y observations
	## 6th column : SumXX = Sum of X*X observations
	## 7th column : SumYY = Sum of Y*Y observations
	## 8th column : Cor   = Autocorrelation at considered lag

	## Source : M.Schleiss, April 2009
	## Complexity : O(N²) where N is the size of tabX

	tab_nNA <- which(!is.na(tabX))
	ltna    <- length(tab_nNA)
	if(ltna<2){stop("invalid input")}

	tabX    <- tabX[tab_nNA]
	tabT    <- tabT[tab_nNA]
	seq_lag <- seq(dr,dmax,dr)
	nrow    <- length(seq_lag)
	M       <- matrix(seq_lag,nrow=nrow,ncol=1)
	M       <- cbind(M,matrix(rep(0,nrow*6),nrow=nrow,ncol=6))
	M       <- cbind(M,matrix(rep(NA,nrow),nrow=nrow,ncol=1))

	for(i in 1:(ltna-1)){
		tabY    <- tabX[(i+1):ltna]
		subT    <- tabT[(i+1):ltna]
		tab_lag <- abs(subT-tabT[i])
		tab_idx <- round(tab_lag/dr)
		id0     <- c(tab_idx>0)
		idmax   <- c(tab_idx<=nrow)
		tab_id  <- which(id0*idmax==1)
		lti     <- length(tab_id)
		if(lti>0){
			for(itr in 1:lti){
				j  <- tab_id[itr]
				id <- tab_idx[j]
				M[id,2] <- M[id,2]+1
				M[id,3] <- M[id,3]+tabX[i]
				M[id,4] <- M[id,4]+tabY[j]
				M[id,5] <- M[id,5]+tabX[i]*tabY[j]
				M[id,6] <- M[id,6]+tabX[i]^2
				M[id,7] <- M[id,7]+tabY[j]^2
			}
		}
	} 
	
	tab_id <- which(M[,2]>=3)
	lti    <- length(tab_id)
	if(lti==0){warning("no valid correlations found")}
	if(lti>0){
		for(itr in 1:lti){
			i   <- tab_id[itr]
			Mx  <- M[i,3]/M[i,2]
			My  <- M[i,4]/M[i,2]
			Cov <- M[i,5]/M[i,2]-Mx*My
			Sx2 <- M[i,6]/M[i,2]-Mx^2
			Sy2 <- M[i,7]/M[i,2]-My^2
			if(Sx2*Sy2>0){
				Cor <- Cov/(sqrt(Sx2*Sy2))
				M[i,8] <- Cor
			}
		}
	}

	return(M)
}

########################### Autocovariance #############################

autocovariance <- function(tabX,tabT,dr,dmax){

	## Computes the autocovariance of a time series
	
	## Inputs:
	## tabX = vector of observations
	## tabT = vector of time stamps (in numeric format)
	## dr   = discretization of time
	## dmax = maximum time lag to be considered
	
	## Remarks:
	## (1) tabX and tabT must have the same size
	## (2) NA values are allowed

	## Outputs
	## M = Autocovariance matrix
	## 1st column : lag   = considered time lag
	## 2nd column : Npts  = number of points (X,Y) for the considered lag
	## 3rd column : SumX  = Sum of X observations
	## 4th column : SumY  = Sum of Y observations
	## 5th column : SumXY = Sum of X*Y observations
	## 6th column : Cov   = Autocovariance at considered lag

	## Source : M.Schleiss, April 2009
	## Complexity : O(N²) where N is the size of tabX

	tab_nNA <- which(!is.na(tabX))
	ltna    <- length(tab_nNA)
	if(ltna<2){stop("invalid input")}

	tabX    <- tabX[tab_nNA]
	tabT    <- tabT[tab_nNA]
	seq_lag <- seq(dr,dmax,dr)
	nrow    <- length(seq_lag)
	M       <- matrix(seq_lag,nrow=nrow,ncol=1)
	M       <- cbind(M,matrix(rep(0,nrow*4),nrow=nrow,ncol=4))
	M       <- cbind(M,matrix(rep(NA,nrow),nrow=nrow,ncol=1))

	for(i in 1:(ltna-1)){
		tabY    <- tabX[(i+1):ltna]
		subT    <- tabT[(i+1):ltna]
		tab_lag <- abs(subT-tabT[i])
		tab_idx <- round(tab_lag/dr)
		id0     <- c(tab_idx>0)
		idmax   <- c(tab_idx<=nrow)
		tab_id  <- which(id0*idmax==1)
		lti     <- length(tab_id)
		if(lti>0){
			for(itr in 1:lti){
				j  <- tab_id[itr]
				id <- tab_idx[j]
				M[id,2] <- M[id,2]+1
				M[id,3] <- M[id,3]+tabX[i]
				M[id,4] <- M[id,4]+tabY[j]
				M[id,5] <- M[id,5]+tabX[i]*tabY[j]
			}
		}
	} 
	
	tab_id <- which(M[,2]>=3)
	lti    <- length(tab_id)
	if(lti==0){warning("no valid covariances found")}
	if(lti>0){
		for(itr in 1:lti){
			i   <- tab_id[itr]
			Mx  <- M[i,3]/M[i,2]
			My  <- M[i,4]/M[i,2]
			Cov <- M[i,5]/M[i,2]-Mx*My
			M[i,6] <- Cov
		}
	}

	return(M)
}

######################### Anisotropy Identification ###################

identify_anisotropy <- function(M,sill){

	## Identifies geometric anisotropy direction and ratio of a 2D field

	## Input:
	## M = a 2D variogram map with 3 columns (x/y/vario)
	## sill = the sill that must be reached by the anisotropy ellipse.

	## Output:
	## tab_anis = a vector containing (theta ; a ; b ; r):
	## 	theta : the anisotropy direction (in degrees from the x axis counter-clockwise)
	## 	a     : the semi-major axis of the ellipse (in the direction specified by theta)
	## 	b     : the semi-minor axis of the ellipse
	## 	r     : the anisotropy ratio b/a

	best      <- rep(NA,4)
	bestSS    <- 1e21

	tab_index <- which(M[,3]>=sill)
	if(length(tab_index)==0){stop("the specified sill is too large")}
	else{

		subM    <- M[tab_index,1:3]
		subTan  <- subM[,2]/subM[,1]
		Pmatrix <- c()

		tolA <- 5
		tolA <- tolA*pi/180
		dA   <- 5             # integration angle in degrees
		dA   <- dA*pi/180     # integration angle in radians
		seqA <- seq(dA,pi,dA) # integration sequence over the first and 2nd quadrants
		ltA  <- length(seqA)

		for(i in 1:ltA){

			tab_tan <- tan(seqA[i] + c(-tolA,tolA))
			tab_tan <- sort(tab_tan)
			if(prod(tab_tan)>=0){
				v1 <- subTan>=tab_tan[1]
				v2 <- subTan<=tab_tan[2]
			}
			else{
				v1 <- subTan>=tab_tan[2]
				v2 <- subTan<=tab_tan[1]
			}
			tab_index <- which(v1*v2==1)
			lti <- length(tab_index)
			if(lti>0){
				subsubM <- matrix(subM[tab_index,1:2],ncol=2,byrow=FALSE)
				tab_dist <- subsubM[,1]^2 + subsubM[,2]^2
				id_min   <- which.min(tab_dist)
				Pmatrix  <- rbind(Pmatrix,subsubM[id_min,1:2])
			}
		}
		Pmatrix <- rbind(Pmatrix,(-1)*Pmatrix)

		dimP <- length(Pmatrix)/2
		if(dimP>3){
			
			tab_phi   <- seq(0,2*pi,2*pi/100)
			tab_theta <- pi*seq(0,175,5)/180

			min_a   <- min(sqrt(Pmatrix[,1]^2 + Pmatrix[,2]^2),na.rm=TRUE)
			max_a   <- max(sqrt(Pmatrix[,1]^2 + Pmatrix[,2]^2),na.rm=TRUE)
			range_a <- max_a-min_a

			tab_a     <- seq(min_a,max_a,range_a/20)
			tab_r     <- seq(0.1,1,0.05)
			
			for(theta in tab_theta){
				for(a in tab_a){
					for(r in tab_r){
						b=r*a
						tabx <- a*cos(tab_phi)*cos(theta)-b*sin(tab_phi)*sin(theta)
						taby <- a*cos(tab_phi)*sin(theta)+b*sin(tab_phi)*cos(theta)
						tab_SS <- rep(1e21,dimP)
						for(k in 1:dimP){
							x <- Pmatrix[k,1]
							y <- Pmatrix[k,2]
							tab_dist <- (tabx-x)^2 + (taby-y)^2
							tab_SS[k] <- min(tab_dist,na.rm=TRUE)
						}
						SS <- sum(tab_SS,na.rm=TRUE)
						if(!is.na(SS)){
							if(SS<bestSS){
								best   <- c(theta,a,b,r)
								bestSS <- SS
							}
						}
					}
				}
			}
		}
	}
	tab_anis <- best
	return(tab_anis)
}


######################### Spherical Variogram function ###################

sph <- function(d,n,s,r){
  if(d>r){return(s+n)}
  else{return(n+s*(3*d/(2*r)-d^3/(2*r^3)))}
}

sph2 <- function(d,n,s,r){
  st <- sum(s)
  rt <- sum(r)
  nt <- sum(n)
  if(d<=rt){
    y1 <- sph(d,n[1],s[1],r[1])
    y2 <- sph(d,n[2],s[2],r[2])
    return(y1+y2)
  }
  if(d>rt){return(st+nt)}
}






