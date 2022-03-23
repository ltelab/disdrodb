################################################################################
############################ T-matrix library for R ############################
####################### by Marc Schleiss, EPFL-LTE, 2011 #######################
################################################################################

## New improved version: July 2011. See the documentation for further details

Tmatrix <- function(AXI,LAM,MRR,MRI,EPS,ALPHA,BETA,THET0,THET,PHI0,PHI,RAT=1,NP=-1,NDGS=2,DDELT=0.001){

    ## This is a wrapper between R and Mishchenko's FORTRAN code
    ## "Calculation of the amplitude matrix for a nonspherical particle in a fixed orientation"
    ## Michael Mishchenko, Applied Optics, Vol.39, No.6, 20th February 2000

    ## Inputs:
    ## AXI = equivolumetric drop radius [mm]
    ## LAM = wavelength of incident wave [mm]
    ## MRR = real part of complex refractive index [-]
    ## MRI = imaginary part of complex refractive index [-]
    ## EPS = ratio between vertical/horizontal particle axis [-]
    ## ALPHA = 1st Euler angle specifying the orientation of the scattering particle relative to the laboratory reference [°]
    ## BETA = 2nd Euler angle specifying the orientation of the scattering particle relative to the laboratory reference [°]
    ## THET0 = zenith angle of incident beam [°]
    ## THET  = zenith angle of scattered beam [°]
    ## PHI0  = azimuth angle of incident beam [°]
    ## PHI   = azimuth angle of scattered beam [°]

    ## Remarks: 
    ## (1) For liquid precipitation, EPS must be <= 1
    ## (1) THET0 and THET must be between 0 and 180° (90° = horizontal)
    ## (2) PHI0 and PHI must be between 0 and 360° (90° = along the y-axis)

    ## Fixed arguments:
    ## RAT   = Particle size is specified in terms of equivolumetric sphere radius 
    ## NP    = Particle shape: -1 corresponds to spheroids
    ## NDGS  = Number of division points in computing integrals over the particle surface. For compact particles, NDGS=2 is ok.
    ## DDELT = Accuracy of computations

    ## R creates an input file containing all necessary parameters to run the Fortran code
    file1 <- "/ltedata/Tmatrix/Tmatrix_input.txt" 
    if(file.exists(file1)){stop("Tmatrix_input.txt already exists on /ltedata/Tmatrix. Delete it or move it somewhere else!")}
    f1 <- file(file1,"w")
    X <- c(AXI,LAM,MRR,MRI,EPS,ALPHA,BETA,THET0,THET,PHI0,PHI,RAT,NP,NDGS,DDELT)
    write(X,f1,ncolumns=1)
    close(f1)

    ## The hostname is used to determine which dynamic library must be loaded
    name  <- system("hostname",intern=TRUE)
    file2 <- sprintf("/ltedata/Tmatrix/Tmatrix_%s.so",name)
    if(!file.exists(file2)){stop("dynamic library not found")}
    dyn.load(file2)

    ## Call the Fortran code to compute the amplitude matrix
    ## The elements of the matrix are written into a text file which can be read from R
    file3 <- "/ltedata/Tmatrix/Tmatrix_output.txt"
    if(file.exists(file3)){stop("Tmatrix_output.txt already exists on /ltedata/Tmatrix. Delete it or move it somewhere else!")}
    dummy <- .Fortran("tm")
    data  <- read.table(file3)
    data  <- unlist(data)
    S11 <- complex(real=data[1],imaginary=data[2])
    S12 <- complex(real=data[3],imaginary=data[4])
    S21 <- complex(real=data[5],imaginary=data[6])
    S22 <- complex(real=data[7],imaginary=data[8])
    S <- matrix(c(S11,S12,S21,S22),nrow=2,ncol=2,byrow=TRUE) 

    ## Clean up the workspace
    dyn.unload(file2)
    system(sprintf("rm %s %s",file1,file3))

    ## Return the amplitude scattering matrix S
    return(S)  
}

################################################################################

back_scat_cross <- function(tabD,tab_ratio,w,m,theta,phi){

    ## Returns the backscattering cross sections [cm2] for a sequence of drop diameters and axis-ratios.

    ## Inputs:
    ## tabD = vector of equivolumetric drop diameters [mm]
    ## tab_ratio = vector of drop axis ratios (vertical/horizontal) [-]
    ## w = wavelength [mm]
    ## m = complex refractivity index of water [-]
    ## theta = zenith of incident beam [°] (90° = horizontal)
    ## phi = azimuth of incident beam [°] (90° = along the y-axis)

    ## Output:
    ## B = matrix of backscattering cross sections [cm2]
    ## 1st column: backscattering cross sections for Hpol
    ## 2nd column: backscattering cross sections for Vpol

    ## Source: Marc Schleiss, EPFL-LTE, July 2011

    ## Some basic tests:
    if(any(is.na(tabD))){stop("NA values not allowed in tabD")}
    if(any(is.na(tab_ratio))){stop("NA values not allowed in tab_ratio")}
    if(is.na(w)){stop("NA values not allowed for w")}
    if(is.na(m)){stop("NA value not allowed for m")}
    if(is.na(theta)){stop("NA value not allowed for theta")}
    if(is.na(phi)){stop("NA value not allowed for phi")}
    if(w<=0){stop("wavelength must be strictly positive")}
    if(any(tabD<=0)){stop("drop diameters must be strictly positive")}
    if(any(tab_ratio<=0)){stop("axis ratios must be strictly positive")}

    NtabD <- length(tabD)
    if(length(tab_ratio)!=NtabD){stop("tabD and tab_ratio must have same length")}

    ## Determine incidence and scattering angle
    if(phi>180){phi <- phi-180}
    PHI0  <- phi
    PHI   <- PHI0+180
    THET0 <- theta
    THET  <- 180-theta

    ## Set wavelength, refractive index and reference angles
    LAM <- w
    MRR <- Re(m)
    MRI <- Im(m)
    ALPHA <- 0
    BETA  <- 0

    ## Go through each drop diameter
    B <- matrix(NA,nrow=NtabD,ncol=2)
    for(i in 1:NtabD){
	AXI <- tabD[i]/2
	EPS <- tab_ratio[i]
	S <- Tmatrix(AXI,LAM,MRR,MRI,EPS,ALPHA,BETA,THET0,THET,PHI0,PHI,RAT=1,NP=-1,NDGS=2,DDELT=0.001)
	B[i,1] <- (4*pi*Mod(S[1,1])^2)/100
	B[i,2] <- (4*pi*Mod(S[2,2])^2)/100
    }

    return(B)
}

################################################################################

ext_scat_cross <- function(tabD,tab_ratio,w,m,theta,phi){

    ## Returns the extinction cross sections [cm2] for a sequence of drop diameters and axis-ratios.

    ## Inputs:
    ## tabD = vector of equivolumetric drop diameters [mm]
    ## tab_ratio = vector of drop axis ratios (vertical/horizontal) [-]
    ## w = wavelength [mm]
    ## m = complex refractivity index of water [-]
    ## theta = zenith of incident beam [°] (90° = horizontal)
    ## phi = azimuth of incident beam [°] (90° = along the y-axis)

    ## Output:
    ## E = matrix of extinction cross sections [cm2]
    ## 1st column: extinction cross sections for Hpol
    ## 2nd column: extinction cross sections for Vpol

    ## Source: Marc Schleiss, EPFL-LTE, July 2011

    ## Some basic tests:
    if(any(is.na(tabD))){stop("NA values not allowed in tabD")}
    if(any(is.na(tab_ratio))){stop("NA values not allowed in tab_ratio")}
    if(is.na(w)){stop("NA values not allowed for w")}
    if(is.na(m)){stop("NA value not allowed for m")}
    if(is.na(theta)){stop("NA value not allowed for theta")}
    if(is.na(phi)){stop("NA value not allowed for phi")}
    if(w<=0){stop("wavelength must be strictly positive")}
    if(any(tabD<=0)){stop("drop diameters must be strictly positive")}
    if(any(tab_ratio<=0)){stop("axis ratios must be strictly positive")}

    NtabD <- length(tabD)
    if(length(tab_ratio)!=NtabD){stop("tabD and tab_ratio must have same length")}

    ## Determine incidence and scattering angle
    if(phi>180){phi <- phi-180}
    PHI0  <- phi
    PHI   <- phi
    THET0 <- theta
    THET  <- theta

    ## Set wavelength, refractive index and reference angles
    LAM <- w
    MRR <- Re(m)
    MRI <- Im(m)
    ALPHA <- 0
    BETA  <- 0

    ## Go through each drop diameter
    E <- matrix(NA,nrow=NtabD,ncol=2)
    for(i in 1:NtabD){
	AXI <- tabD[i]/2
	EPS <- tab_ratio[i]
	S <- Tmatrix(AXI,LAM,MRR,MRI,EPS,ALPHA,BETA,THET0,THET,PHI0,PHI,RAT=1,NP=-1,NDGS=2,DDELT=0.001)
	E[i,1] <- w*Im(S[1,1])/50
	E[i,2] <- w*Im(S[2,2])/50
    }

    return(E)
}

################################################################################

fwrd_scat_ampl <- function(tabD,tab_ratio,w,m,theta=90,phi=90){

    ## Computes the real part of the forward scattering amplitude

    ## Inputs:
    ## tabD = vector of equivolumetric drop diameters [mm]
    ## tab_ratio = vector of drop axis ratios (vertical/horizontal) [-]
    ## w = wavelength [mm]
    ## m = complex refractive index of water [-]
    ## theta = zenith of incident beam [°] (90° = horizontal)
    ## phi = azimuth of incident beam [°] (90° = along the y-axis)

    ## Output:
    ## A = matrix of forward scattering amplitudes [-]
    ## 1st columns: amplitudes for Hpol
    ## 2nd column: amplitude for Vpol

    ## Source: Marc Schleiss, EPFL-LTE, July 2011

    ## Some basic tests:
    if(any(is.na(tabD))){stop("NA values not allowed in tabD")}
    if(any(is.na(tab_ratio))){stop("NA values not allowed in tab_ratio")}
    if(is.na(w)){stop("NA values not allowed for w")}
    if(is.na(m)){stop("NA value not allowed for m")}
    if(is.na(theta)){stop("NA value not allowed for theta")}
    if(is.na(phi)){stop("NA value not allowed for phi")}
    if(w<=0){stop("wavelength must be strictly positive")}
    if(any(tabD<=0)){stop("drop diameters must be strictly positive")}
    if(any(tab_ratio<=0)){stop("axis ratios must be strictly positive")}

    NtabD <- length(tabD)
    if(length(tab_ratio)!=NtabD){stop("tabD and tab_ratio must have same length")}

    ## Determine incidence and scattering angle
    if(phi>180){phi <- phi-180}
    PHI0  <- phi
    PHI   <- phi
    THET0 <- theta
    THET  <- theta

    ALPHA <- 0
    BETA  <- 0
    MRR   <- Re(m)
    MRI   <- Im(m)
    LAM   <- w

    A <- matrix(NA,nrow=NtabD,ncol=2)
    for(i in 1:NtabD){
	AXI <- tabD[i]/2
	EPS <- tab_ratio[i]
	S <- Tmatrix(AXI,LAM,MRR,MRI,EPS,ALPHA,BETA,THET0,THET,PHI0,PHI,RAT=1,NP=-1,NDGS=2,DDELT=0.001)
	A[i,1] <- Re(S[1,1])
	A[i,2] <- Re(S[2,2])
    }

    return(A)
}


