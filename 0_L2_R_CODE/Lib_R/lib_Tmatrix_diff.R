#################################### Tmatrix Code ###############################

Tmatrix <- function(wlength,ref_ind,D,ratio,THET0,PHI0,THET,PHI){

	## 	"Calculation of Light Scattering by Polydisperse Randomly Oriented Particles
	## 	of Identical Axially Symmetric Shape", by Dr.Michael Mishenko (08.06.2005)

	##  Input Arguments :
	##    wlength = the wavelenght of the considered electromagnetic wave (in mm)
	##    ref_ind = the complex refractive index of water
	##    D       = the equivolumetric diameter of the drop (in mm)
	##    ratio   = the ratio of vertical/horizontal axis of the drop
	##              for oblate spheroids ratio < 1
        ##              for prolate spheroids ratio > 1 
	##              See Pruppacher (1971), Beard (1987), Andsager (1999), Gorgucci (2006) or Thurai (2007)
	##              for different models and relations
	##    THET0   = zenith  of the incident wave (between 0 and 180°)(90° = horizontal)
	##    PHI0    = azimuth of the incident wave (between 0 and 360°)(90° = along the y-axis)
	##    THET    = zenith  of the scattered wave (between 0 and 180°)
	##    PHI     = azimuth of the scattered wave (between 0 and 360°)(for backscattering use PHI0+180 modulo 360)

	##  Fixed Arguments : The following arguments do not need modifications and are not relevant within the framework of our simulation

	##    B      = the variance of the size distribution (in mm²) as we deal with single particles this isn't necessary
	##    NDISTR = the distribution of the equivalent sphere radii. 
	##             This is not important as we work with single particles.
	##    NP     = the shape of the particles : NP = -1 for spheroids
	##    DDELT  = accuracy of the computations (by default 0.001)
	##    NPNA   = number of equidistant scattering angles (from 0 to 180) 
	##             for which the scattering matrix is calculated (by default 19).
	##             this isn't relevant because the incident and scattered angles are determined individually.
	##    NDGS   = number of division points in computing integrals over the particle surface
	##             for compact particles the recommended value is 2
	##             for highly aspherical particles larger values (3,4,...) may be necessary
	##    R1     = the minimum equivalent sphere radius (in mm)
	##             This isn't relevant for individual particles.
	##    R2     = the maximum equivalent sphere radius (in mm)
	##             This isn't relevant for individual particles.
	##    NPNAX  = the number of size distributions of the same type that are computed in the same run
	##             for a single particle use NPNAX = 1 
	##    RAT    = Configuration parameter for the equivalent sphere radius
	##             if  RAT = 1 the particle size is specified in terms of the equal volume-sphere radius
	##             else the particle size is specified in terms of the equal surface-sphere radius  
	##    AXMAX  = AXI if NPNAX=1
	##    GAM    = (only for Gamma distributions) is ignored in the current configuration of the program
	##    NKMAX  = 5 the number of Gaussian quadrature points used to integrate over the size distribution
        ##    NANGS   = Number of angles for the integration over the canting distribution
	##    name    = name of the computer (hostname)  : "ltepc3" ; "ltepc4" ; "ltepc5" ; "ltepc2"


	##  Output Arguments :
	##    S = the amplitude scattering matrix (2x2)


	MRR <- Re(ref_ind)
	MRI <- Im(ref_ind)
	AXI <- D/2	
	EPS <- 1/ratio
	B      <- 0.1	
	NDISTR <- 2 
	NP     <- -1
	DDELT  <- 0.001	
	NPNA   <- 19
	NDGS   <- 2
  	R1     <- 0.9999999*AXI
  	R2     <- 1.0000001*AXI
  	NPNAX  <- 1
	RAT    <- 1
	AXMAX  <- AXI
	GAM    <- 1
	NKMAX  <- -1
	NANGS  <- 1
	name <- system("hostname",intern=TRUE)

	## Loading the dynamic personal Shared Fortran Library
        ## This library must have been compiled on the machine it is beeing used.
	dyn.load(sprintf("/USERS/lte/commun1/Prog_com/Lib_fortran/Tmatrix/Tmatrix_%s.so",name))

	## All these parameters are written into the file "Tmatrix_parameters.txt"
	## The Fortran Code reads this file and writes the results into "Tmatrix_output.txt"
	## "R" reads the output and returns the values

	my_file <- file("Tmatrix_parameters.txt","w")
	write(wlength,my_file)
	write(MRR,my_file,append=TRUE)
	write(MRI,my_file,append=TRUE)
	write(AXI,my_file,append=TRUE)
	write(B,my_file,append=TRUE)
	write(EPS,my_file,append=TRUE)
	write(NDISTR,my_file,append=TRUE)
	write(NP,my_file,append=TRUE)
	write(DDELT,my_file,append=TRUE)
	write(NPNA,my_file,append=TRUE)
	write(NDGS,my_file,append=TRUE)
	write(R1,my_file,append=TRUE)
	write(R2,my_file,append=TRUE)
	write(NPNAX,my_file,append=TRUE)
	write(RAT,my_file,append=TRUE)
	write(AXMAX,my_file,append=TRUE)
	write(GAM,my_file,append=TRUE)
	write(NKMAX,my_file,append=TRUE)
	write(THET0,my_file,append=TRUE)
	write(THET,my_file,append=TRUE)
	write(PHI0,my_file,append=TRUE)
	write(PHI,my_file,append=TRUE)
	write(NANGS,my_file,append=TRUE)

	## generating NANGS random canting angles for the integration
        alpha <- rnorm(NANGS,mean=0,sd=0.01)
        beta  <- rnorm(NANGS,mean=0,sd=0.01)
        for(i in 1:NANGS){
           write(alpha[i],my_file,append=TRUE)
           write(beta[i],my_file,append=TRUE)
        }
        close(my_file)

	## Call of the external Tmatrix Fortran-77 Code of M.Mishchenko
  	returned_data <- .Fortran("tm")
  	Output <- read.table("Tmatrix_output.txt")
  	Output <- unlist(Output)
        S11 <- complex(real=Output[1],imaginary=Output[2])
        S12 <- complex(real=Output[3],imaginary=Output[4])
        S21 <- complex(real=Output[5],imaginary=Output[6])
        S22 <- complex(real=Output[7],imaginary=Output[8])
        S <- matrix(c(S11,S12,S21,S22),nrow=2,ncol=2,byrow=T)

	## The program ends
	dyn.unload(sprintf("/USERS/lte/commun1/Prog_com/Lib_fortran/Tmatrix/Tmatrix_%s.so",name))
  	return(S)
}

#################################### Backscattering cross sections ###############################

back_scat_cross <- function(D,alpha,wlength,m,theta,phi){

  ## Computes the mean backscattering cross section of a rain drop that oscillates according to a given canting distribution
  ## The raindrops are assumed to be spheroids with equivolumetric diameter D and shape parameter alpha = minor/major axis
  ## The canting distribution is given by a truncated gaussian law of mean 0° and standard deviation 10°
  ## The scattering matrix is calculated using the T-matrix code of M.Mishchenko.
  ## The integration over the canting distribution is performed directly in the Fortran Code.
  
  ## To use this code one must first compile the T-matrix code into a shared Library under Prog_com/Lib_Fortran 

  ## Inputs :
  ##   D        = the diameter of the drop (in mm)
  ##   alpha    = the ratio of the vertical/horizontal axis
  ##   wlength  = the wavelength (in mm)
  ##   m        = the complex refractive index of water
  ##   theta    = zenith of incident wave (90° = horizontal)
  ##   phi      = azimuth of incident wave (90° = along the y axis)

  ## Outputs :
  ##   BSH = Back-Scattering cross section for Horizontal polarization (in cm2)
  ##   BSV = Back-Scattering cross section for Vertical polarization (in cm2)

  ## Source : M.Schleiss (May.2008) 

  if(phi>180){phi <- phi-180}
  phi_inc <- phi
  phi_sca <- phi_inc+180
  theta_inc <- theta
  theta_sca <- theta

  S <- Tmatrix(wlength,m,D,alpha,theta_inc,phi_inc,theta_sca,phi_sca)
  BSH <- 4*pi*Mod(S[2,2])^2
  BSV <- 4*pi*Mod(S[1,1])^2
  
  ## Converting mm^2 to cm^2
  BSH <- BSH/100
  BSV <- BSV/100

  out <- c(BSH,BSV)
  return(out)
}


############################ Extinction Cross Section ###############################

ext_scat_cross <- function(D,alpha,wlength,m,theta,phi){

  ## Computes the mean extinction cross section of a rain drop that oscillates according to a given canting distribution
  ## See back_scat_cross() for more details.

  if(phi>180){phi <- phi-180}
  phi_inc <- phi
  phi_sca <- phi_inc
  theta_inc <- theta
  theta_sca <- theta

  S <- Tmatrix(wlength,m,D,alpha,theta_inc,phi_inc,theta_sca,phi_sca)
  k <- 2*pi/wlength
  ESH <- 4*pi*Im(S[2,2])/k
  ESV <- 4*pi*Im(S[1,1])/k
  
  ## Converting mm^2 to cm^2
  ESH <- ESH/100
  ESV <- ESV/100

  out <- c(ESH,ESV)
  return(out)
}

########################### Forward Scattering amplitude ###########################

forward_scat_ampl <- function(seqD,seqAR,wlength,m,theta=90,phi=90){

    ## Computes the (real part) of the forward scattering amplitude
    
    ## Inputs:
    ## seqD        = vector of drop diameters [mm]
    ## seqAR       = vector of vertical/horizontal axis-ratios [-] 
    ## wlength     = wavelength [mm]
    ## m           = complex refractive index of water
    ## theta       = zenith of incident wave (90° = horizontal)
    ## phi         = azimuth of incident wave (90° = along the y axis)

    ## Output:
    ## out = matrix with 2 columns Re(Shh)|Re(Svv), one row per diameter

    NseqD <- length(seqD)
    if(phi>180){phi <- phi-180}
    phi_inc <- phi
    phi_sca <- phi_inc
    theta_inc <- theta
    theta_sca <- theta

    out <- matrix(NA,nrow=NseqD,ncol=2)
    for(i in 1:NseqD){
	D  <- seqD[i]
	ar <- seqAR[i]
	S <- Tmatrix(wlength,m,D,ar,theta_inc,phi_inc,theta_sca,phi_sca)    
	out[i,1] <- Re(S[2,2])
	out[i,2] <- Re(S[1,1])
    }
    return(out)
}
