################################################################################
########################## Library for DSD Processing ########################## 
######################### by M.Schleiss, EPFL-LTE 2011 #########################
################################################################################

  get.classD <- function(){
  
      ## Returns a 32x2 matrix containing the lower/upper diameter limits (in mm)
      ## of the Parsivel disdrometer.
  
      classD <- matrix(NA,nrow=32,ncol=2)
      classD[1,] <- c(0,0.1245)
      classD[2,] <- c(0.1245,0.2495)
      classD[3,] <- c(0.2495,0.3745)
      classD[4,] <- c(0.3745,0.4995)
      classD[5,] <- c(0.4995,0.6245)
      classD[6,] <- c(0.6245,0.7495)
      classD[7,] <- c(0.7495,0.8745)
      classD[8,] <- c(0.8745,0.9995)
      classD[9,] <- c(0.9995,1.1245)
      classD[10,] <- c(1.1245,1.25)
      classD[11,] <- c(1.25,1.50)
      classD[12,] <- c(1.50,1.75)
      classD[13,] <- c(1.75,2.00)
      classD[14,] <- c(2.00,2.25)
      classD[15,] <- c(2.25,2.50)
      classD[16,] <- c(2.50,3.00)
      classD[17,] <- c(3.00,3.50)
      classD[18,] <- c(3.50,4.00)
      classD[19,] <- c(4.00,4.50)
      classD[20,] <- c(4.50,5.00)
      classD[21,] <- c(5.00,6.00)
      classD[22,] <- c(6.00,7.00)
      classD[23,] <- c(7.00,8.00)
      classD[24,] <- c(8.00,9.00)
      classD[25,] <- c(9.00,10.0)
      classD[26,] <- c(10.0,12.0)
      classD[27,] <- c(12.0,14.0)
      classD[28,] <- c(14.0,16.0)
      classD[29,] <- c(16.0,18.0)
      classD[30,] <- c(18.0,20.0)
      classD[31,] <- c(20.0,23.0)
      classD[32,] <- c(23.0,26.0)
      return(classD)
  }
  
  get.classV <- function(){
  
      ## Returns a 32x2 matrix containing the lower/upper velocity limits (in m/s)
      ## of the Parsivel disdrometer.
  
      classV <- matrix(NA,nrow=32,ncol=2)
      classV[1,] <- c(0,0.1)
      classV[2,] <- c(0.1,0.2)
      classV[3,] <- c(0.2,0.3)
      classV[4,] <- c(0.3,0.4)
      classV[5,] <- c(0.4,0.5)
      classV[6,] <- c(0.5,0.6)
      classV[7,] <- c(0.6,0.7)
      classV[8,] <- c(0.7,0.8)
      classV[9,] <- c(0.8,0.9)
      classV[10,] <- c(0.9,1.0)
      classV[11,] <- c(1.0,1.2)
      classV[12,] <- c(1.2,1.4)
      classV[13,] <- c(1.4,1.6)
      classV[14,] <- c(1.6,1.8)
      classV[15,] <- c(1.8,2.0)
      classV[16,] <- c(2.0,2.4)
      classV[17,] <- c(2.4,2.8)
      classV[18,] <- c(2.8,3.2)
      classV[19,] <- c(3.2,3.6)
      classV[20,] <- c(3.6,4.0)
      classV[21,] <- c(4.0,4.8)
      classV[22,] <- c(4.8,5.6)
      classV[23,] <- c(5.6,6.4)
      classV[24,] <- c(6.4,7.2)
      classV[25,] <- c(7.2,8.0)
      classV[26,] <- c(8.0,9.6)
      classV[27,] <- c(9.6,11.2)
      classV[28,] <- c(11.2,12.8)
      classV[29,] <- c(12.8,14.4)
      classV[30,] <- c(14.4,16.0)
      classV[31,] <- c(16.0,19.2)
      classV[32,] <- c(19.2,22.4)
      return(classV)
  }

ref_index_water <- function(t,f){

    ## Computes the complex refractive index of liquid water
    ## Based on the article of H.Liebe, 1991, Int. J. Infrared. Milli., vol.12, no.7, pp.659-675

    ## Inputs:
    ## t = temperature [°C]
    ## f = frequency [GHz]

    ## Outputs:
    ## m = m'+im'' the complex refractive index

    if(is.na(t)){stop("NA values not allowed for t")}
    if(is.na(f)){stop("NA values not allowed for f")}
    if(t<0 || t>30){stop("temperature must be between 0 and 30°C")}
    if(f<=0 || f>1000){stop("frequency must be between 0 and 1000 GHz")}

    Theta <- 1 - 300/(273.15+t)
    Epsilon_0 <- 77.66 - 103.3*Theta
    Epsilon_1 <- 0.0671*Epsilon_0
    Epsilon_2 <- 3.52 + 7.52*Theta
    Gamma_1 <- 20.20 + 146.5*Theta + 316*Theta^2
    Gamma_2 <- 39.8*Gamma_1
  
    expr1 <- Epsilon_0-Epsilon_1
    expr2 <- 1+(f/Gamma_1)^2
    expr3 <- 1+(f/Gamma_2)^2
    expr4 <- Epsilon_1-Epsilon_2
    expr5 <- Epsilon_2

    Epsilon_real <- expr1/expr2 + expr4/expr3 + expr5
    Epsilon_imag <- (expr1/expr2)*(f/Gamma_1) + (expr4/expr3)*(f/Gamma_2)
    Epsilon <- complex(real=Epsilon_real,imaginary=Epsilon_imag)
    m <- sqrt(Epsilon)
    return(m)
}

raindrop_axis_ratio <- function(tabD,model="PK"){

    ## Computes the raindrop axis ratio for a given drop diameter

    ## Inputs:
    ## tabD  = vector of equivolume drop diameters [mm]
    ## model = character string specifying the axis-ratio model (default is "PK")

    ## Currently implemented axis-ratio models are:
    ## "PK" = Pruppacher and Klett, 1997 (from the book "Microphysics of Clouds and Precipitation") 
    ## "Andsager" = Andsager et al., 1999, JAS, vol.56, pp.2673-2683
    ## "Brandes" = Brandes et al., JAM, 2002, vol.41, pp.674-685

    ## Output:
    ## tabr <- vector of drop axis ratios (vertical/horizontal) [-]

    tabm  <- c("PK","Andsager","Brandes")
    NtabD <- length(tabD)

    ## Basic tests:
    if(NtabD==0){return(c())}
    if(any(is.na(tabD))){stop("NA values are not allowed in tabD")}
    if(all(tabm!=model)){stop("invalid axis-ratio model")}
    if(any(tabD<=0)){stop("drop diameters must be strictly positive")}

    tabr <- rep(NA,NtabD)

    ## Pruppacher and Klett, 1997
    ## Valid for drop diameters between 0 and 6 mm
    if(model=="PK"){
	if(any(tabD>6)){warning("The Pruppacher and Klett model is only valid for drop diameters between 0 and 6 mm")}
	D <- tabD/10
	tabr <- 1.0048 + 0.0057*D - 2.628*D^2 + 3.682*D^3 - 1.677*D^4
    }

    ## Andsager et al., 1998
    ## Valid for drop diameters between 1.1 and 4.4 mm
    ## For diameters between 0 - 1.1 mm and 4.4 - 6 mm, the "PK" model is used.
    if(model=="Andsager"){
	if(any(tabD>6)){warning("The Andsager model is only valid for drop diameters between 0 and 6 mm")}
	D <- tabD/10
	tabr <- 1.0048 + 0.0057*D - 2.628*D^2 + 3.682*D^3 - 1.677*D^4
	id <- intersect(which(tabD>=1.1),which(tabD<=4.4))
	Nid <- length(id)
	if(Nid>0){tabr[id] <- 1.012 - 0.144*D[id] - 1.03*D[id]^2}	
    }

    ## Brandes et al., 2002
    ## Valid for drop diameters between 0.1 to 8.1 mm
    if(model=="Brandes"){
	if(any(tabD<0.1) || any(tabD>8.1)){warning("The Brandes model is only valid for drop diameters between 0.1 and 8.1 mm")}
	tabr <- 0.9951 + 0.0251*tabD - 0.03644*tabD^2 + 0.00503*tabD^3 - 0.0002492*tabD^4		
    }

    ## Return the drop axis ratios
    tabr[tabr>1] <- 1
    return(tabr)

}

raindrop_velocity <- function(tabD){

    ## Returns the terminal fall speed of a drop of equivolume diameter D
    ## From the article of Beard, 1977, JAS, vol.34, pp.1293-1298
    ## Note that this is an approximation valid at sea level, P=1atm ; T=20°C ; rho=1.194 kg/m3

    ## Inputs:
    ## tabD = vector of equivolume drop diameters [mm]

    ## Output:
    ## tabV = vector with terminal fall speeds [m/s]

    ## Basic tests:
    if(any(is.na(tabD))){stop("NA values not allowed in tabD")}
    if(any(tabD<=0)){stop("drop diameters must be strictly positive")}
    if(any(tabD>7)){warning("some drop diameters were larger than 7 mm")}
  
    N  <- length(tabD)
    X  <- log(tabD/10)
    C0 <- 7.06037
    C1 <- 1.74951
    C2 <- 4.86324
    C3 <- 6.60631
    C4 <- 4.84606
    C5 <- 2.14922
    C6 <- 0.58714
    C7 <- 0.096348
    C8 <- 0.00869209
    C9 <- 0.00033089
    S <- rep(C0,N) + C1*X + C2*X^2 + C3*X^3 + C4*X^4 + C5*X^5 + C6*X^6 + C7*X^7 + C8*X^8 + C9*X^9
    tabV <- exp(S)/100
    tabV[tabD>7]  <- NA
    return(tabV)
}

compute_volR <- function(DSD,tabD){

    ## Computes the rain rate associated to volumic DSD spectra

    ## Inputs:
    ## DSD = matrix containing the drop counts per cubic meter per diameter class.
    ##       each column represents the center of a given diameter class.
    ## tabD = vector of diameter classes [mm] (i.e., 32 values for Parsivel)  

    ## Output:
    ## tabR = vector of rain rates [mm]

    ## Basic tests
    nrow  <- dim(DSD)[1]
    ncol  <- dim(DSD)[2]
    NtabD <- length(tabD)
    if(ncol!=NtabD){stop("the number of columns in DSD must be equal to the length of tabD")}

    ## Compute the terminal fall speeds for each diameter class
    tabV <- raindrop_velocity(tabD)

    ## Compute the rain rate for each DSD spectrum
    tabR   <- rep(NA,nrow)
    rM     <- rowMeans(DSD)
    id.dry <- which(rM==0)
    id.wet <- which(rM>0)
    Ndry   <- length(id.dry)
    Nwet   <- length(id.wet)
    if(Ndry>0){tabR[id.dry] <- 0}
    for(i in id.wet){tabR[i] <- (6*pi/10000)*sum(DSD[i,]*tabV*tabD^3,na.rm=TRUE)}
    return(tabR)
}

compute_volmD <- function(DSD,tabD){

    ## Returns the mean drop diameter associated to volumic DSD spectra

    ## Inputs:
    ## DSD = matrix containing the drop counts per cubic meter per diameter class.
    ##       each column represents the center a given diameter class
    ## tabD = vector of diameter classes [mm] (i.e., 32 values for Parsivel)

    ## Output:
    ## tabmeanD = vector of mean drop diameters [mm]

    nrow <- dim(DSD)[1]
    ncol <- dim(DSD)[2]
    NtabD <- length(tabD)
    if(ncol!=NtabD){stop("the number of columns in DSD must be equal to the length of tabD")}

    tabmD  <- rep(NA,nrow)
    D      <- matrix(rep(tabD,nrow),nrow=nrow,ncol=ncol,byrow=TRUE)
    rS     <- .rowSums(DSD,nrow,ncol)
    id.wet <- which(rS>0)
    Nwet   <- length(id.wet)
    if(Nwet>0){
	sub.DSD <- matrix(DSD[id.wet,],nrow=Nwet,ncol=ncol)
	sub.rS  <- rS[id.wet]
	sub.D   <- D[id.wet,]
	tabmD[id.wet] <- .rowSums(sub.DSD*sub.D,Nwet,ncol)/sub.rS
    }
    return(tabmD)
}

compute_volDm <- function(DSD,tabD){

    ## Returns the mass-weighted mean drop diameter associated to volumic DSD spectra

    ## Inputs:
    ## DSD = matrix containing the drop counts per cubic meter per diameter class.
    ##       each column represents the center a given diameter class
    ## tabD = vector of diameter classes [mm] (i.e., 32 values for Parsivel)

    ## Output:
    ## tabDm = vector of mass-weighted mean drop diameters [mm]

    nrow <- dim(DSD)[1]
    ncol <- dim(DSD)[2]
    NtabD <- length(tabD)
    if(ncol!=NtabD){stop("the number of columns in DSD must match the length of tabD")}

    tabDm  <- rep(NA,nrow)
    rS     <- .rowSums(DSD,nrow,ncol)
    D3     <- matrix(rep(tabD^3,nrow),nrow=nrow,ncol=ncol,byrow=TRUE)
    D4     <- matrix(rep(tabD^4,nrow),nrow=nrow,ncol=ncol,byrow=TRUE)
    id.wet <- which(rS>0)
    Nwet   <- length(id.wet)
    if(Nwet>0){
	sub.D3  <- D3[id.wet,]
	sub.D4  <- D4[id.wet,]
	sub.DSD <- DSD[id.wet,]
	if(Nwet==1){
	    sub.D3  <- matrix(sub.D3,nrow=1,ncol=ncol)
	    sub.D4  <- matrix(sub.D4,nrow=1,ncol=ncol)
	    sub.DSD <- matrix(sub.DSD,nrow=1,ncol=ncol)    
	}
	tabDm[id.wet] <- .rowSums(sub.DSD*sub.D4,Nwet,ncol)/(.rowSums(sub.DSD*sub.D3,Nwet,ncol))
    }
    return(tabDm)
}

compute_volD0 <- function(DSD,tabD){

    ## Computes the median volume drop diameter associated to volumic DSD spectra

    ## Inputs:
    ## DSD = matrix containing the drop counts per cubic meter per diameter class.
    ##       each column represents the center of a given diameter class
    ## tabD = vector of diameter classes [mm] (i.e., 32 values for Parsivel)

    ## Output:
    ## tabD0 = vector of median volume drop diameters [mm]

    nrow  <- dim(DSD)[1]
    ncol  <- dim(DSD)[2]
    NtabD <- length(tabD)
    if(ncol!=NtabD){stop("the number of columns in DSD must be equal to the length of tabD")}
    if(any(is.na(tabD))){stop("NA values are not allowed in tabD")}
    if(any(tabD<=0)){stop("the values in tabD must be strictly positive")}

    tabD0  <- rep(NA,nrow)
    id.wet <- which(rowMeans(DSD)>0)
    for(i in id.wet){
	tabN <- DSD[i,]
	cs   <- cumsum(tabN*tabD^3)
	s    <- cs[NtabD]
	low  <- which(cs<=s/2)
	up   <- which(cs>=s/2)
	if(length(low)*length(up)==0){next}
	low <- max(low)
	up  <- min(up)
	if(low==up){
	    tabD0[i] <- tabD[low]
	    next
	}
	s1 <- cs[up]
	s2 <- cs[low]
	tabD0[i] <- tabD[low]+(tabD[up]-tabD[low])*(s/2-s1)/(s2-s1)
    }
    return(tabD0)
}

compute_volA <- function(DSD,tabExt){

    ## Computes the specific attenuation on propagation associated to volumic DSD spectra

    ## Inputs:
    ## DSD = matrix containing the drop counts per cubic meter per diameter class.
    ##       each column represents the center of a given diameter class
    ## tabExt = vector of extinction cross sections [cm2] (at a given frequency) for each diameter class

    ## Output:
    ## tabA = vector of specific attenuations on propagation [dB/km]

    nrow <- dim(DSD)[1]
    ncol <- dim(DSD)[2]
    NExt <- length(tabExt)
    if(ncol!=NExt){stop("the number of columns in DSD must be equal to the length of tabExt")}
    if(any(is.na(tabExt))){stop("NA values are not allowed in tabExt")}
    if(any(tabExt<=0)){stop("the values in tabExt must be strictly positive")}

    tabA   <- rep(NA,nrow)
    rM     <- rowMeans(DSD)
    id.dry <- which(rM==0)
    id.wet <- which(rM>0)
    Ndry   <- length(id.dry)
    if(Ndry>0){tabA[id.dry] <- 0}
    for(i in id.wet){tabA[i] <- sum(DSD[i,]*tabExt)/log(10)}
    return(tabA)
}

compute_volZ <- function(DSD,tabB,w,m){

    ## Computes the radar reflectivity associated to volumic DSD spectra

    ## Inputs: 
    ## DSD = matrix containing the drop counts per cubic meter per diameter class
    ##       each column represents the center of a given diameter class
    ## tabB = vector of back-scattering cross sections [cm2] (at a given frequency and temperature) for each diameter class
    ## w = radar wavelength [cm]
    ## m = complex refractive index of water (at a given frequency and temperature)

    ## Output:
    ## tabZ = vector of radar reflectivities [dBZ]

    nrow  <- dim(DSD)[1]
    ncol  <- dim(DSD)[2]
    NtabB <- length(tabB)
    if(nrow==0){stop("DSD must contain at least 1 row")}
    if(NtabB==0){stop("tabB must contain at least 1 element")}
    if(ncol!=NtabB){stop("the number of columns in DSD must be equal to the length of tabB")}
    if(any(is.na(tabB))){stop("NA values are not allowed in tabB")}
    if(any(tabB<0)){stop("the values in tabB must be positive")}
    if(any(as.vector(DSD)<0,na.rm=TRUE)){stop("negative values are not allowed in DSD")}

    tabZ   <- rep(NA,nrow)
    tabNt  <- rowSums(DSD)
    id.wet <- which(tabNt>0)
    Nwet   <- length(id.wet)
    if(Nwet==0){return(tabZ)}

    Kw <- (m^2-1)/(m^2+2)
    Cz <- 1e6*w^4/(pi^5*abs(Kw)^2) 

    for(i in id.wet){
	tabN <- DSD[i,]
	tabZ[i] <- Cz*sum(tabB*tabN)
    }
    tabZ[id.wet] <- 10*log10(tabZ[id.wet])
    tabZ[tabZ==Inf] <- NA
    tabZ[tabZ==(-Inf)] <- NA
    return(tabZ)
}

compute_volW <- function(DSD,tabD,deltaD){

    ## Returns the liquid water content associated to volumic DSD spectra
  
    ## Inputs:
    ## DSD = matrix containing the drop counts per cubic meter per diameter class.
    ##       each column represents the center of a given diameter class
    ## tabD = vector of diameter classes [mm] (i.e., 32 values for Parsivel)
    ## deltaD = the width of each diameter class [mm]

    ## Output:
    ## tabW = vector with liquid water content [g/m^3]

    ## Remark:
    ## The water density is assumed to be 1 g/m^3

    nrow <- dim(DSD)[1]
    ncol <- dim(DSD)[2]
    if(ncol!=length(tabD)){stop("the number of columns in DSD must match length of tabD")}

    tabW   <- rep(NA,nrow)
    D3     <- matrix(rep(tabD^3,nrow),nrow=nrow,ncol=ncol,byrow=TRUE)
    DD     <- matrix(rep(deltaD,nrow),nrow=nrow,ncol=ncol,byrow=TRUE)
    rS     <- .rowSums(DSD,nrow,ncol)
    id.wet <- which(rS>0)
    id.dry <- which(rS==0)
    Nwet   <- length(id.wet)
    Ndry   <- length(id.dry)
    if(Ndry>0){tabW[id.dry] <- 0}
    if(Nwet>0){
	sub.DSD <- DSD[id.wet,]
	sub.D3  <- D3[id.wet,]
	sub.DD  <- DD[id.wet,]
	if(Nwet==1){
	    sub.DSD <- matrix(DSD[id.wet,],nrow=1,ncol=ncol)
	    sub.D3  <- matrix(D3[id.wet,],nrow=1,ncol=ncol)
	    sub.DD  <- matrix(DD[id.wet,],nrow=1,ncol=ncol)
	}
	tabW[id.wet] <- .rowSums(sub.D3*sub.DSD*sub.DD,Nwet,ncol)*pi/6000
    }
    return(tabW)    
}

compute_volNw <- function(DSD,tabD,deltaD){

    ## Returns the normalized drop concentration associated to volumic DSD spectra

    ## Inputs:
    ## DSD = matrix containing the drop counts per cubic meter per diameter class.
    ##       each column represents the center of a given diameter class
    ## tabD = vector of diameter classes [mm] (i.e., 32 values for Parsivel)
    ## deltaD = the width of each diameter class [mm]

    ## Output:
    ## tabNw = vector with normalized drop concentrations [mm^-1 m^-3]

    tabDm <- compute_volDm(DSD,tabD)
    tabW  <- compute_volW(DSD,tabD,deltaD)
    tabNw <- 1000*256*tabW/(pi*tabDm^4)
    return(tabNw)
}

compute_volSk <- function(DSD,tabD){

    ## Returns the sample skewness associated to volumic DSD spectra

    ## Inputs:
    ## DSD = matrix containing the drop counts per cubic meter per diameter class.
    ##       each column represents the center of a given diameter class
    ## tabD = vector of diameter classes [mm] (i.e., 32 values for Parsivel)

    ## Output:
    ## tabSk = vector with sample skewness

    nrow  <- dim(DSD)[1]
    NtabD <- length(tabD)
    D     <- matrix(rep(tabD,nrow),nrow=nrow,ncol=NtabD,byrow=TRUE)
    rS    <- rowSums(DSD)
    idw   <- which(rS>0)
    Nw    <- length(idw)
    tabm  <- rep(NA,nrow)
    if(Nw>0){tabm[idw] <- rowSums(DSD[idw,]*D[idw,])/rS[idw]}
    M <- matrix(rep(tabm,NtabD),nrow=nrow,ncol=NtabD)
    tabm2 <- rep(NA,nrow)
    tabm3 <- rep(NA,nrow)
    tabSk <- rep(NA,nrow)
    if(Nw>0){
	tabm2[idw] <- rowSums((DSD[idw,]*(D[idw,]-M[idw,])^2))/rS[idw]
	tabm3[idw] <- rowSums((DSD[idw,]*(D[idw,]-M[idw,])^3))/rS[idw]
	tabSk[idw] <- tabm3[idw]/(tabm2[idw]^(3/2))
    }
    return(tabSk)
}

compute_gamma.R <- function(tab_mu,tab_lam,tab_Nt){

    ## Computes the rain rates associated to Gamma DSD parameters
    ## The Gamma DSD model is given by N(D) = alpha*Nt*D^mu*exp(-lam*D)

    ## Inputs:
    ## tab_mu  = vector of mu values [-]
    ## tab_lam = vector of lambda values [1/mm]
    ## tab_Nt  = vector of Nt values [1/m^3]

    ## Output:
    ## tabR = vector of rain rate values [mm/h]

    Nmu  <- length(tab_mu)
    Nlam <- length(tab_lam)
    NNt  <- length(tab_Nt)
    if(Nmu!=Nlam || Nmu!=NNt){stop("tab_mu, tab_lam and tab_Nt must have the same length")}
    if(Nmu==0){return(c())}

    if(any(tab_mu<=(-1),na.rm=TRUE)){stop("mu values must be larger than -1")}
    if(any(tab_lam<=0,na.rm=TRUE)){stop("negative values for lambda are not allowed")}
    if(any(tab_Nt<0,na.rm=TRUE)){stop("Nt values must be strictly positive")}

    tabR   <- rep(NA,Nmu)
    tabR[tab_Nt==0] <- 0

    id.wet <- which(tab_Nt>0)
    Nwet   <- length(id.wet)
    if(Nwet==0){return(tabR)}
    seqD <- seq(0.1,7.0,0.01)
    seqV <- raindrop_velocity(seqD)
    C1   <- (1e-6)*3600*(pi/6)
    for(i in id.wet){
	mu  <- tab_mu[i]
	lam <- tab_lam[i]
	Nt  <- tab_Nt[i]
	C2  <- sum(seqD^mu*exp(-lam*seqD))
	if(is.na(C2)){next}
	if(C2==0){next}
	C3 <- Nt*sum(seqD^mu*exp(-lam*seqD)*seqV*seqD^3)
	tabR[i] <- C1*C3/C2
    }
    tabR[tabR<0] <- 0
    tabR[tabR==Inf] <- NA
    return(tabR)
} 

compute_gamma.A <- function(tab_mu,tab_lam,tab_Nt,seqD,seqExt){

    ## Computes the specific attenuation associated to Gamma DSD parameters
    ## The Gamma DSD model is given by N(D) = alpha*Nt*D^mu*exp(-lam*D)

    ## Inputs:
    ## tab_mu  = vector of mu values [-]
    ## tab_lam = vector of lambda values [1/mm]
    ## tab_Nt  = vector of Nt values [1/m^3]
    ## seqD    = vector of drop diameters [mm] for numerical integration
    ## seqExt  = vector of extinction cross sections [cm2] associated to seqD (at a given frequency and polarization) 
    
    ## Output:
    ## tabA = vector of specific attenuations [dB/km]

    Nmu   <- length(tab_mu)
    Nlam  <- length(tab_lam)
    NNt   <- length(tab_Nt)
    NseqD <- length(seqD)
    NExt  <- length(seqExt)

    if(Nmu!=Nlam || Nmu!=NNt){stop("tab_mu, tab_lam and tab_Nt must have the same length")}
    if(NseqD!=NExt){stop("seqD and seqExt must have the same length")}
    if(any(is.na(seqD))){stop("NA values are not allowed in seqD")}
    if(any(is.na(seqExt))){stop("NA values are not allowed in seqExt")}
    if(any(seqD<=0)){stop("the values in seqD must be strictly positive")}
    if(any(seqExt<=0)){stop("the values in seqExt must be strictly positive")}
    if(any(tab_mu<=(-1))){stop("mu values must be larger than -1")}
    if(any(tab_lam<=0)){stop("negative values for lambda are not allowed")}
    if(any(tab_Nt<0)){stop("Nt values must be strictly positive")}
    if(Nmu==0){return(c())}

    tabA <- rep(NA,Nmu)
    tabA[tab_Nt==0] <- 0

    id.wet <- which(tab_Nt>0)
    Nwet   <- length(id.wet)
    if(Nwet==0){return(tabA)}
    
    for(i in id.wet){
	mu  <- tab_mu[i]
	lam <- tab_lam[i]
	Nt  <- tab_Nt[i]
	num <- Nt*sum(seqExt*seqD^mu*exp(-lam*seqD))
	denom <- log(10)*sum(seqD^mu*exp(-lam*seqD))
	if(is.na(num*denom)){next}
	if(denom==0){next}
	tabA[i] <- num/denom
    }
    tabA[tabA<0] <- 0
    tabA[tabA==Inf] <- NA
    return(tabA)
}

compute_gamma.Z <- function(tab_mu,tab_lam,tab_Nt,w,m,seqD,seqBack){

    ## Computes the radar reflectivity associated to Gamma DSD parameters
    ## The Gamma DSD model is given by N(D) = alpha*Nt*D^mu*exp(-lam*D)

    ## Inputs:
    ## tab_mu  = vector of mu values [-]
    ## tab_lam = vector of lambda values [1/mm]
    ## tab_Nt  = vector of Nt values [1/m^3]
    ## w       = radar wavelength [cm]
    ## m       = complex refractive index of water (at the radar frequency and for a given temperature)
    ## seqD    = vector of drop diameters [mm] for numerical integration
    ## seqBack = vector of backscattering cross sections [cm^2] associated to seqD (at the radar frequency)

    ## Output:
    ## tabZ = vector of reflectivities [dBZ]

    Nmu   <- length(tab_mu)
    Nlam  <- length(tab_lam)
    NNt   <- length(tab_Nt)
    NseqD <- length(seqD)
    NBack <- length(seqBack)

    if(Nmu!=Nlam || Nmu!=NNt){stop("tab_mu, tab_lam and tab_Nt must have the same length")}
    if(NseqD!=NBack){stop("seqD and seqBack must have the same length")}
    if(is.na(w)){stop("NA value not allowed for w")}
    if(any(is.na(seqD))){stop("NA values are not allowed in seqD")}
    if(any(is.na(seqBack))){stop("NA values are not allowed in seqBack")}
    if(w<=0){stop("w must be strictly positive")}
    if(any(seqD<=0)){stop("the values in seqD must be strictly positive")}
    if(any(seqBack<=0)){stop("the values in seqBack must be strictly positive")}
    if(any(tab_mu<=(-1))){stop("mu values must be larger than -1")}
    if(any(tab_lam<=0)){stop("negative values for lambda are not allowed")}
    if(any(tab_Nt<0)){stop("Nt values must be strictly positive")}
    if(Nmu==0){return(c())}
    
    Kw     <- (m^2-1)/(m^2+2)
    tab_Cz <- 1e6*w^4*tab_Nt/(pi^5*abs(Kw)^2)

    tabZ   <- rep(NA,Nmu)
    id.wet <- which(tab_Nt>0)
    Nwet   <- length(id.wet)
    if(Nwet==0){return(tabZ)}

    for(i in id.wet){
	mu  <- tab_mu[i]
	lam <- tab_lam[i]
	Nt  <- tab_Nt[i]
	num <- sum(seqBack*seqD^mu*exp(-lam*seqD))
	I   <- sum(seqD^mu*exp(-lam*seqD))
	tabZ[i] <- 10*log10(tab_Cz[i]*num/I)
    }
    tabZ[tabZ==(-Inf)] <- NA
    tabZ[tabZ==Inf] <- NA
    return(tabZ)
}

compute_gamma.Kdp <- function(tab_mu,tab_lam,tab_Nt,seqD,tab_Shh,tab_Svv,freq){

    ## Computes the one-way specific differential phase for a Gamma DSD model
    
    ## Inputs:
    ## tab_mu     = vector of mu values [-]
    ## tab_lambda = vector of lambda values [1/mm]
    ## tab_Nt     = vector of concentration values [1/m^3]
    ## seqD       = diameter discretization table [mm]
    ## tab_Shh    = vector of forward scattering amplitudes (pol=H, only real part)
    ## tab_Svv    = vector of forward scattering amplitudes (pol=V, only real part)
    ## freq       = the frequency [GHz]

    ## Output:
    ## tab_Kdp = vector of specific differential phase shifts [°/km]

    ## Performing some tests
    if(length(tab_mu)!=length(tab_lam)){stop("dimensions do not match")}
    if(length(tab_lam)!=length(tab_Nt)){stop("dimensions do not match")}
    if(length(tab_Shh)!=length(tab_Svv)){stop("dimensions do not match")}
    if(length(tab_Shh)!=length(seqD)){stop("dimensions do not match")}

    ## Defining some variables
    c       <- 299792458	## speed of light [m/s]
    wlength <- 1e3*c/(freq*1e9)	## wavelength [mm]
    nDSD    <- length(tab_mu)
    Ck      <- (180/pi)*wlength*1e-3    ## [m °C]

    ## Computing Kdp for each DSD
    tab_Kdp <- rep(NA,nDSD)
    not_NA  <- which(!is.na(tab_mu))
    for(i in not_NA){
	mu    <- tab_mu[i]
	lam   <- tab_lam[i]
	Nt    <- tab_Nt[i]
	num   <- sum((tab_Shh-tab_Svv)*seqD^mu*exp(-lam*seqD),na.rm=TRUE)
	denom <- sum(seqD^mu*exp(-lam*seqD),na.rm=TRUE)
	if(denom==0){next}
	tab_Kdp[i] <- Ck*Nt*num/denom
    }
    return(tab_Kdp)
}

compute_gamma.D0 <- function(tab_mu,tab_lam,tab_Nt,seqD){
    
    ## Computes the median volume drop diameter D0 for a Gamma DSD model
    
    ## Inputs:
    ## tab_mu  = vector of shape parameters [-]
    ## tab_lam = vector of rate parameters [1/mm]
    ## tab_Nt  = vector of concentration parameters [1/m^3]
    ## seqD    = drop diameter discretization table [mm]

    ## Output:
    ## tabD0 = vector of median water content diameters [mm]

    N1 <- length(tab_mu)
    N2 <- length(tab_lam)
    N3 <- length(tab_Nt)
    if(N1!=N2 || N1!=N3){stop("dimensions do not match")}
    N <- N1
    if(N==0){stop("input of size zero")}
    NseqD <- length(seqD)
    if(NseqD==0){stop("diameter table of size zero")}
    if(sum(is.na(seqD))>0){stop("NA values are not allowed in drop diameters")}
    if(sum(seqD<=0)>0){stop("drop diameters must be positive")}
    if(sum(seqD>10)>0){stop("drop diameters larger than 10mm are not realistic")}
    id <- which(tab_mu>(-1))
    id <- intersect(id,which(tab_lam>0))
    id <- intersect(id,which(tab_Nt>=0))
    Nid <- length(id)
    tabD0 <- rep(NA,N)
    for(i in id){
	mu   <- tab_mu[i]
	lam  <- tab_lam[i]
	Nt   <- tab_Nt[i]
	cumS <- cumsum(seqD^(mu+3)*exp(-lam*seqD))
	S    <- cumS[NseqD]
	D0   <- seqD[which.min(abs(cumS-S/2))]
	tabD0[i] <- D0
    }
    return(tabD0)
}

compute_gamma.Dm <- function(tab_mu,tab_lam,seqD){
    
    ## Computes the average drop diameter for a Gamma DSD model

    ## Inputs:
    ## tab_mu  = vector of shape parameters [-]
    ## tab_lam = vector of rate parameters [1/mm]
    ## seqD    = drop diameter discretization table [mm]

    ## Outputs:
    ## tab_Dm = vector with average drop diameters [mm]

    N1 <- length(tab_mu)
    N2 <- length(tab_lam)
    if(N1!=N2){stop("dimensions of tab_mu and tab_lam do not match")}
    if(sum(is.na(seqD))>0){stop("NA values not allowed in seqD")}
    dD <- seqD[2]-seqD[1]
    
    tab_Dm <- rep(NA,N1)
    id <- intersect(which(tab_mu>(-1)),which(tab_lam>0))
    for(i in id){
	mu  <- tab_mu[i]
	lam <- tab_lam[i]
	num <- sum(seqD^(mu+1)*exp(-lam*seqD))
	denom <- sum(seqD^mu*exp(-lam*seqD))
	if(is.na(num*denom)){next}
	if(denom==0){next}
	tab_Dm[i] <- num/denom
    }
    return(tab_Dm)
}
