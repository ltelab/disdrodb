############################## New DSD Library ##############################
############################### by M.Schleiss ############################### 

## This is the new "R" shared DSD library for EPFL-LTE
## It contains all useful functions and routines related to DSD
## Old and/or obsolete functions were removed from the library  

######################## Parsivel Diameter Classes ########################

low_D <- rep(0,32)
low_D[1] <- 0.0000
low_D[2] <- 0.1245
low_D[3] <- 0.2495
low_D[4] <- 0.3745
low_D[5] <- 0.4995
low_D[6] <- 0.6245
low_D[7] <- 0.7495
low_D[8] <- 0.8745
low_D[9] <- 0.9995
low_D[10] <- 1.1245
low_D[11] <- 1.25
low_D[12] <- 1.50
low_D[13] <- 1.75
low_D[14] <- 2.00
low_D[15] <- 2.25
low_D[16] <- 2.50
low_D[17] <- 3.00
low_D[18] <- 3.50
low_D[19] <- 4.00
low_D[20] <- 4.50
low_D[21] <- 5.00
low_D[22] <- 6.00
low_D[23] <- 7.00
low_D[24] <- 8.00
low_D[25] <- 9.00
low_D[26] <- 10.0
low_D[27] <- 12.0
low_D[28] <- 14.0
low_D[29] <- 16.0
low_D[30] <- 18.0
low_D[31] <- 20.0
low_D[32] <- 23.0

up_D <- rep(0,32)
for(i in 1:31){
   up_D[i] <- low_D[i+1]
}
up_D[32] <- 26
mean_D <- (low_D+up_D)/2

######################### Parsivel Velocity Classes #########################

Velocity <- rep(0,32)
Velocity[1] <- 0.050
Velocity[2] <- 0.150
Velocity[3] <- 0.250
Velocity[4] <- 0.350
Velocity[5] <- 0.450
Velocity[6] <- 0.550
Velocity[7] <- 0.650
Velocity[8] <- 0.750
Velocity[9] <- 0.850
Velocity[10] <- 0.950
Velocity[11] <- 1.100
Velocity[12] <- 1.300
Velocity[13] <- 1.500
Velocity[14] <- 1.700
Velocity[15] <- 1.900
Velocity[16] <- 2.200
Velocity[17] <- 2.600
Velocity[18] <- 3.000
Velocity[19] <- 3.400
Velocity[20] <- 3.800
Velocity[21] <- 4.400
Velocity[22] <- 5.200
Velocity[23] <- 6.000
Velocity[24] <- 6.800
Velocity[25] <- 7.600
Velocity[26] <- 8.800
Velocity[27] <- 10.400
Velocity[28] <- 12.000
Velocity[29] <- 13.600
Velocity[30] <- 15.200
Velocity[31] <- 17.600
Velocity[32] <- 20.800

######################## Network Station Coordinates ########################

get.network_coordinates <- function(id_station,campaign="Network_EPFL_2009",all=FALSE){

    ## Returns the coordinates (lat,long,alt,estY,nordX) of LTE-network stations
    
    ## Input: 
    ## id_station = a list with the desired stations (caution: id = 10,11,12,13,20,21,...)
    ## all = logical, if TRUE the coordinates of all stations available for this campaign are returned
    
    ## Output:
    ## CoordM = a matrix of station coordinates (id;lat;long;alt;Est;Nord), 1 row per station
    
    ## Source: M.Schleiss, October 2009
    ## Modified by J. Jaffrain, Oct 12th 2010.
    
    ## Remarks:
    ## All coordinates were measured with Garmin GPS Dakota 20. 
    ## EstY and NordX coordinates were obtained using the online NAVREF
    ## projection tool provided by the swiss topographic institute 
    ## http://www.swisstopo.admin.ch/internet/swisstopo/fr/home/apps/calc/navref.html
    ## All altitudes are assumed constant at 400m.

    ## Campaign names:
    ## Network EPFL 2009	campaign <- paste("Network_EPFL_2009",sep="")
    ## Davos 2009-2010		campaign <- paste("Davos_2009-2010",sep="")
    ## Roof 2010		campaign <- paste("Roof_2010",sep="")
    ## Hpiconet_2010		campaign <- paste("Hpiconet_2010",sep="")
    ## COMMON_2011		campaign <- paste("COMMON_2011",sep="")

    if(campaign=="Roof_2008"){
	TotalCoordM <- data.frame(matrix(NA,nrow=6,ncol=6))
# 	TotalCoordM[,1] <- c("01","02","03","41","42","43")
	TotalCoordM[1,] <- c("01",46.521400,6.567867,400,533182,152605)
	TotalCoordM[2,] <- c("02",46.521400,6.567867,400,533182,152605)
	TotalCoordM[3,] <- c("03",46.521400,6.567867,400,533182,152605)
	TotalCoordM[4,] <- c("41",46.521400,6.567867,400,533182,152605)
	TotalCoordM[5,] <- c("42",46.521400,6.567867,400,533182,152605)
	TotalCoordM[6,] <- c("43",46.521400,6.567867,400,533182,152605)
    }

    if(campaign=="Network_EPFL_2009"){
	TotalCoordM <- matrix(NA,nrow=16,ncol=6)
	TotalCoordM[1,]  <- c(10,46.520500,6.565200,400,532977,152507)
	TotalCoordM[2,]  <- c(11,46.520433,6.562833,400,532795,152502)
	TotalCoordM[3,]  <- c(12,46.521900,6.565183,400,532977,152663)
	TotalCoordM[4,]  <- c(13,46.521267,6.566767,400,533098,152591)
    
	TotalCoordM[5,]  <- c(20,46.519800,6.570500,400,533383,152425)
	TotalCoordM[6,]  <- c(21,46.519583,6.572317,400,533522,152399)
	TotalCoordM[7,]  <- c(22,46.521200,6.572583,400,533544,152579)
	TotalCoordM[8,]  <- c(23,46.520533,6.571100,400,533429,152506)
    
	TotalCoordM[9,]  <- c(30,46.518333,6.563933,400,532877,152267)
	TotalCoordM[10,] <- c(31,46.519650,6.563900,400,532876,152414)
	TotalCoordM[11,] <- c(32,46.518700,6.562733,400,532785,152309)
	TotalCoordM[12,] <- c(33,46.517633,6.564583,400,532926,152189)
    
	TotalCoordM[13,] <- c(40,46.521017,6.569733,400,533325,152561)
	TotalCoordM[14,] <- c(41,46.519500,6.567883,400,533181,152394)
	TotalCoordM[15,] <- c(42,46.520600,6.567850,400,533180,152516)
	TotalCoordM[16,] <- c(43,46.521400,6.567867,400,533182,152605)
    }

    if(campaign=="Davos_2009-2010"){

	TotalCoordM <- matrix(NA,nrow=3,ncol=6)
	TotalCoordM[1,] <- c(50,46.829683,9.809417,2543,780859,189236)
	TotalCoordM[2,] <- c(60,46.821067,9.820250,2276,781714,188304)
# 	TotalCoordM[3,] <- c(70,46.808983,9.863817,1520,785078,187063)		.
	TotalCoordM[3,] <- c(70,NA,NA,1520,NA,NA)		## Need to be redefined, Joël 2011-09-06.
    }

    if(campaign=="Roof_2010"){

	TotalCoordM <- matrix(NA,nrow=4,ncol=6)
	TotalCoordM[1,] <- c(23,46.521400,6.567867,400,533182,152605)
	TotalCoordM[2,] <- c(41,46.521400,6.567867,400,533182,152605)
	TotalCoordM[3,] <- c(42,46.521400,6.567867,400,533182,152605)
	TotalCoordM[4,] <- c(43,46.521400,6.567867,400,533182,152605)
    }

    if(campaign=="Hpiconet_2010-2011"){

	TotalCoordM <- matrix(NA,nrow=8,ncol=6)
	TotalCoordM[,1] <- c(10,11,12,13,30,31,32,33)
# 	TotalCoordM[1,] <- c(10,)
# 	TotalCoordM[2,] <- c(41,46.521400,6.567867,400,533182,152605)
# 	TotalCoordM[3,] <- c(42,46.521400,6.567867,400,533182,152605)
# 	TotalCoordM[4,] <- c(43,46.521400,6.567867,400,533182,152605)
    }

    if(campaign=="COMMON_2011"){

	TotalCoordM <- matrix(NA,nrow=5,ncol=6)
	TotalCoordM[1,] <- c(20,47.4166,8.6380,433,690509,252446)
	TotalCoordM[2,] <- c(21,47.4138,8.6357,435,690340,252132)
	TotalCoordM[3,] <- c(22,47.4068,8.6327,455,690125,251350)
	TotalCoordM[4,] <- c(40,47.4049,8.6299,446,689917,251136)
	TotalCoordM[5,] <- c(41,47.4049,8.6299,446,689917,251136)
    }

    if(campaign=="HYMEX/SOP_2012"){

	TotalCoordM <- matrix(NA,nrow=7,ncol=6)
	TotalCoordM[,1] <- c(10,11,13,30,31,32,33)
    }

    if(all==TRUE){CoordM <- TotalCoordM}
    if(all==FALSE){
        Nstations <- length(id_station)
	if(Nstations==0){stop("no station has been specified")}
	CoordM <- data.frame(matrix(NA,nrow=Nstations,ncol=6))		# modified 2012-01-16, Joël (add 'data.frame()')
	for(i in 1:Nstations){
	    id <- id_station[i]
	    j  <- which(TotalCoordM[,1]==id)
	    Nj <- length(j)
	    if(Nj!=1){print(sprintf("warning: station %i not found",id))}
	    if(Nj==1){CoordM[i,] <- TotalCoordM[j,]}
	}
    }
    for(j in 2:6){CoordM[,j] <- as.numeric(CoordM[,j])}
    return(CoordM)
}

##################### Distance Between Network Stations #####################

get.network_station_distance <- function(id_station1,id_station2){

    ## Returns the distance (in meters) between two LTE network stations
    ## If the two stations are identical, the returned distance is zero
    ## If one of the stations does not exist, the function returns NA
    
    ## Input:
    ## id_station1 = number of the first station
    ## id_station2 = number of the second station
    
    ## Output:
    ## dist = the distance (in meters) between the two stations

    rt <- 6378100

    if(is.na(id_station1) || is.na(id_station2)){
	print("Error: NA in station number")
	stop()
    }
    if(id_station1==id_station2){dist <- 0}
    else{
	coords <- get.network_coordinates(c(id_station1,id_station2))
	lat1   <- coords[1,2]*pi/180
	long1  <- coords[1,3]*pi/180
	lat2   <- coords[2,2]*pi/180
	long2  <- coords[2,3]*pi/180
	dist   <- rt*acos(sin(lat1)*sin(lat2)+cos(lat1)*cos(lat2)*cos(long2-long1))
    }
    return(dist)  
}

######################### Refractive index of water #########################

ref_index_water <- function(t,f){

  ## From the article of H.Liebe: "A model for the complex permittivity of
  ## water at frequencies below 1 THz"

  ## More general than the Debye equation this formula is valid for 
  ## frequencies up to 1THz.
  ## Replaces the work of P.Ray, Applied Optics Vol.8,p.1836-1844, 1972

  ## Inputs : 
  ##   t = the temperature (in °C)
  ##   f = the frequency (in GHz)

  ## Outputs :
  ##   m = m' + im'' the complex refractive index

    ## Source : M.Schleiss (May.2008)
    ## Updated 06.12.2017 by T. Raupach to replace 146.5 with 146.4 as per original Liebe article.

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

######################### Refractive index of ice #########################

ref_index_ice <- function(t,f){

  ## From the article of Hufford 1991: "A model for the complex permittivity of
  ## water at frequencies below 1 THz" 

  ## And also, "Ice and water permettivities for millimeter and sub-millimeter 
  ## remote sensing applications" by Jiang, 2004.

  ## shOULD BE VALID FOR TEMPERATURES IN THE RANGE -40 TO 0 °C
 
  ## Inputs : 
  ##   t = the temperature (in °C)
  ##   f = the frequency (in GHz)

  ## Outputs :
  ##   m = m' + im'' the complex refractive index

  ## Source : J. Grazioli (Jan. 2015)

  Theta       <- -1 + 300/(273.15+t)
  alpha       <- (50.4+62*Theta)*10^(-4.)*exp(-22.1*Theta)
  beta        <- 10.^(-4.)*(0.502-0.131*Theta)/(1+Theta)
  beta        <- beta+0.542*10^(-6)*((1+Theta)/(Theta+0.0073))^2

  epsilonprime    <- 3.15
  epsilonsecond   <- alpha/f+beta*f

  Epsilon <- complex(real=epsilonprime,imaginary=epsilonsecond)

  m <- sqrt(Epsilon)
  return(m)

}


#################################################################################
ref_index_water_sal <- function(t,f,S){

    ## Computes the complex dielectric constant of water
    ## Taking into account the salinity of rain

    ## Inputs:
    ## t = temperature [°C]
    ## f = frequency [GHz]
    ## S =salinity of water in per thousand

    ## Outputs:
    ## m = m'+im'' the complex refractive index
    ## Ellison (2005)

    ## Grazioli 2012

T<-t
fGHz <- f
	a1<- 0.46606917*10^(-2)
	a2<--0.26087876*10^(-4)
	a3<--0.63926782*10^(-5)
	a4<- 0.63000075*10
	a5<- 0.26242021*10^(-2)
	a6<--0.42984155*10^(-2)
	a7<- 0.34414691*10^(-4)
	a8<- 0.17667420
	a9<--0.20491560*10^(-3)
	a10<-0.58366888*10^(3)
	a11<-0.12634992*10^(3)
	a12<-0.69227972*10^(-1)
	a13<-0.38957681*10^(-3)
	a14<-0.30742330*10^(3)
	a15<-0.12634992*10^(3)
	a16<-0.37245044*10
	a17<-0.92609781*10^(-2)
	a18<--0.26093754*10^(-1)

T2 <- T*T
T3 <- T2*T
T4 <- T3*T
S2 <- S*S
alfa0 <- (6.9431+3.2841*S-0.099486*S2)/(84.850+69.024*S+S2)
alfa1 <- 49.843-0.2276*S+0.00198*S2
RTQ <- 1+alfa0*(T-15)/(alfa1+T)
R15 <- S*(37.5109+5.45216*S+1.4409e-02*S2)/(1004.75+182.283*S+S2)
sigma35 <- 2.903602+8.607*0.01*T+4.738817*0.0001*T2-2.991*10^(-6)*T3+4.3041*10^(-9)*T4
sigma <- sigma35*RTQ*R15
es0<- 87.85306
es<- es0*exp(-0.00456992*T-a1*S-a2*S2-a3*S*T)
e1<-  a4*exp(-a5*T-a6*S-a7*S*T)
einf<-a16+a17*T+a18*S
tau1<-(a8 + a9*S)*exp(a10/(T+a11))
tau2<-(a12+a13*S)*exp(a14/(T+a15))
tp<-2*pi/1000
delta1<-es-e1 
delta2<-e1-einf
i <- complex(real=0,imaginary=1)
eps<-delta1/(1-i*tp*fGHz*tau1)+delta2/(1-i*tp*fGHz*tau2)+einf+i*17.9751*sigma/fGHz#dielectric constant
eps <- eps^0.5 #refractive index
return(eps) 

}



################### Raindrop Terminal Fall Speed ################### 

raindrop_velocity <- function(tabD){

  ## From the article of Beard, JAS Vol.34 1977 pp.1293-1298
  ## This is an approximation valid at sea level P=1atm ; T=20°C ; p=1.194 kg/m3
  ## The complete model is available under /USERS/lte/commun1/Prog_com/lib_R/Beard_Model.R

  ## Remark: This function replaces "drop_velocity()" from previous lib_DSD library.
  ##         The new version has been vectorized and now returns the speed in m/s.

  ## Inputs :
  ##   tabD = a vector of equivolumetric drop diameters (in mm)
  
  ## Outputs :
  ##   tabV = a vector of terminal fall speeds (in m/s)

  ## Source : A.Berne, 2007
  ## Modifications : M.Schleiss (May 2008) ; M.Schleiss (April 2009)

  if(sum(is.na(tabD))>0){stop("NA values are not allowed")}
  if(sum(tabD<0)>0){stop("All diameters must be positive")}

  N <- length(tabD)
  tabD <- tabD/10
  tabx <- log(tabD)

  C0   <- 7.06037
  C1   <- 1.74951
  C2   <- 4.86324
  C3   <- 6.60631
  C4   <- 4.84606
  C5   <- 2.14922
  C6   <- 0.58714
  C7   <- 0.096348
  C8   <- 0.00869209
  C9   <- 0.00033089

  C  <- c(C1,C2,C3,C4,C5,C6,C7,C8,C9)

  S  <- rep(C0,N)
  for(i in 1:9){
     S <- S+C[i]*tabx^i
  }
  tabV <- exp(S)/100
  return(tabV) 
}

###################################################################

raindrop_axis_ratio <- function(tabD,Aydin=FALSE){

  ## From the article of Andsager & Beard (1999) "Laboratory measurements
  ## of axis-ratios for large raindrops", JAS Vol.56

  ## Only valid for drops with equivolumetric diameter between 0.1 and 7.0mm

  ## If D is between 1.1 and 4.4 mm the axis ratio is computed according
  ## to the average axis-ratio relationship given by Kubesh & Beard (1993):
  ## "Laboratory measurements of spontaneous oscillations for moderate-size 
  ## raindrops", J. Atmos. Sci. vol.50, 1089–1098.

  ## If D is between 0.1 and 1.1 mm or between 4.4 and 7.0 mm the axis-ratio
  ## is computed according to the equilibrium shape equation given by 
  ## Beard & Chuang (1987):"A new model for the equilibrium shape of 
  ## raindrops", JAS Vol.44

  ## Remark: This function replaces "axis_ratio_AB()" from previous lib_DSD library.
  ##         The new version has been vectorized.

  ## Inputs : 
  ##   tabD  = a vector of equivolumetric drop diameters (in mm)
  ##   Aydin = logical, if TRUE the Aydin relationship is used.

  ## Outputs :
  ##   tab_ratio = vector of axis ratios (vertical/horizontal)

  ## Source : M.Schleiss, May.2008
  ## Modifications: M.Schleiss, April 2009

  if(sum(is.na(tabD))>0){stop("NA values are not allowed")}
  if(sum(tabD<=0)>0){stop("All diameters must be positive")}
  if(sum(tabD>7.0)>0){
      print("Drop diameters higher than 7.0 mm")
      ind <- which(tabD > 7.0)
      tabD[ind] <- 7
  }
  if(sum(tabD<0.1)>0){
      print("Drop diameters smaller than 0.1 mm")
      ind <- which(tabD < 0.1)
      tabD[ind] <- 0.1
  }

  N    <- length(tabD)
  tabD <- tabD/10
  v1   <- c(tabD<=0.44)
  v2   <- c(tabD>=0.11)
  id1  <- which(v1*v2==1)
  id2  <- which(v1*v2==0)

  if(Aydin==TRUE){tab_ratio <- 0.993+0.082*tabD-1.874*tabD^2+1.469*tabD^3}
  else{
      tab_ratio <- rep(NA,N)
      if(length(id1)>0){
	subD <- tabD[id1]
	tabR <- 1.012 - 0.144*subD-1.03*subD^2 
	tab_ratio[id1] <- tabR
      }
      if(length(id2)>0){
	subD <- tabD[id2]
	tabR <- 1.0048 + 0.0057*subD - 2.628*subD^2 + 3.682*subD^3 - 1.677*subD^4 
	tab_ratio[id2] <- tabR
      }
  }
  return(tab_ratio)
}

###################################################################
raindrop_axis_ratio_seq <- function(tabD){

    ## Returns the raindrop axis ratio (vertical/horizontal) [-]
    ## From the article of Andsager et al., JAS Vol.56 1998, pp.2673-2683
    ## and Thurai 2005. Different calculations for different diameters

    ## Grazioli 2012

    ## Inputs:
    ## tabD = vector of equivolumetric drop diameters [mm]

    ## Output:
    ## tab_ratio = vector of axis ratios [-]

    ## Basic tests:
    if(any(is.na(tabD))){stop("NA values not allowed in tabD")}
    if(any(tabD<=0)){stop("drop diameters must be strictly positive")}
    if(any(tabD>7.5)){warning("some drop diameters were larger than 7.5 mm")}

    NtabD <- length(tabD)   
    tab_ratio <- rep(NA,NtabD)
for(i in 1:NtabD){
	Di <- tabD[i]
	if (Di > 8){Di <- 8}
	
	if(Di < 0.7){
	tab_ratio[i] <- 1
        }
        if((Di >= 0.7)&(Di <= 1.5)){
	tab_ratio[i] <- (1.173-0.5165*Di+0.4698*Di^2-0.1317*Di^3-8.5*0.001*Di^4)
        }
        if(Di >1.5){
	tab_ratio[i] <- (1.065-6.25*0.01*Di-3.99*0.001*Di^2+7.66*0.0001*Di^3-4.095*0.00001*Di^4)
        }
}
    tab_ratio[tab_ratio>1] <- 1
    return(tab_ratio)
}

####################################################################3


surface_DSD.R <- function(tabN,tabD,S,dt){

  ## Computes the rain intensity derived from ground DSD measurements

  ## Inputs :
  ##   tabN = the number of drops (on the ground) in each diameter class
  ##   tabD = the mean diameter classes (32 values for Parsivel)  (in mm)
  ##   S    = the sampling surface in square meters
  ##   dt   = the time between two successive measurements (in seconds)

  ## Outputs :
  ##   R = the rain intensity in mm/h

  R <- (1/1e6)*(1/S)*(3600/dt)*(pi/6)*sum(tabN*tabD^3,na.rm=TRUE)
  return(R)
}

###################################################################

volumic_DSD.R <- function(DSDmatrix,tabD){

    ## Computes the rain intensity derived from volumic DSD measurements

    ## Inputs :
    ##   DSDmatrix = matrix containing the number of drops (per cubic meter)
    ##               for each diameter class (1 row = 1 DSD)
    ##   tabD = the diameter classes [mm] (32 values for Parsivel)

    ## Outputs :
    ##   tabR = vector of rain intensities [mm/h]

    ## Basic tests
    nrow  <- dim(DSDmatrix)[1]
    ncol  <- dim(DSDmatrix)[2]
    NtabD <- length(tabD)
    if(ncol!=NtabD){stop("invalid dimensions")}

    ## Compute terminal fall speed
    tabV <- raindrop_velocity(tabD)
    
    ## Computing the rain intensity for each DSD
    tabR   <- rep(NA,nrow)
    rM     <- rowMeans(DSDmatrix,na.rm=TRUE)
    id.wet <- which(rM>0)
    for(i in id.wet){
	tabN <- DSDmatrix[i,]
	tabR[i] <- (6*pi/10^4)*sum(tabN*tabV*tabD^3,na.rm=TRUE)
    }
    return(tabR)
}

########################Differential phase upon backscattering############################
gamma_DSD.delta <- function(res,tab_mu,tab_lam,tab_Nt,w,seqD,tab_backscatt_ampl){

    ## Computes the radar delta [°] for a Gamma DSD model
    
    ## Inputs:
    ## res      = resolution of the discretization vector of D
    ## tab_mu   = vector of "shape" parameters (mu = alpha-1)
    ## tab_lam  = vector of rate parameters [1/mm]
    ## tab_Nt   = vector of concentration parameters [1/m3]
    ## w        = wavelength [mm]
    ## seqD     = diameter discretization vector [mm]
    ## tab_backscatt_ampl   = backscattering amplitudes for [1]-[H], [2]-[V]

    ##Grazioli, Jun 2012

    ## Output:
    ## tabKDP = vector of KDP [°/km]
    tab_bH <- tab_backscatt_ampl[,1] 
    tab_bV <- tab_backscatt_ampl[,2]

    ## Some basic tests:
    Ntab_mu  <- length(tab_mu)
    Ntab_lam <- length(tab_lam)
    Ntab_Nt  <- length(tab_Nt)
    NseqD    <- length(seqD)
    Ntab_bH  <- length(tab_bH)

    if(Ntab_mu!=Ntab_lam){stop("tab_mu and tab_lam must have same length")}
    if(Ntab_mu!=Ntab_Nt){stop("tab_mu and tab_Nt must have same length")}
    if(is.na(w)){stop("NA value not allowed for w")}
    if(is.na(m)){stop("NA value not allowed for m")}
    if(w<=0){stop("w must be strictly positive")}
    if(NseqD!=Ntab_bH){stop("seqD and tab_fH must have same length")}
    if(any(is.na(seqD))){stop("NA values not allowed in seqD")}
    if(any(is.na(tab_bH))){stop("NA values not allowed in tab_fH")}
    if(any(tab_mu<=(-1),na.rm=TRUE)){stop("values in tab_mu must be greater than -1")}
    if(any(tab_lam<0,na.rm=TRUE)){stop("values in tab_lam must be positive")}
    if(any(tab_Nt<0,na.rm=TRUE)){stop("negative values not allowed in tab_Nt")}
    if(any(seqD<=0)){stop("negative diameters are not allowed")}

    ## Compute delta for valid measurements
    tab_delta <- rep(NA,Ntab_mu)
    id     <- which(tab_mu>(-1))
    id     <- intersect(id,which(tab_lam>0))
    id     <- intersect(id,which(tab_Nt>0))
    for(i in id){
	mu  <- tab_mu[i]
	lam <- tab_lam[i]
	Nt  <- tab_Nt[i]
	#num   <- sum((tab_bH*Conj(tab_bV))*seqD^mu*exp(-lam*seqD),na.rm=TRUE)
	#denom <- sum(seqD^mu*exp(-lam*seqD),na.rm=TRUE)
	#tab_delta[i] <- abs(Arg(num/denom)*180/pi+180)

	dummy <- sum((tab_bH*Conj(tab_bV))*Nt*seqD^mu*exp(-lam*seqD),na.rm=TRUE)
	denom <- sum(seqD^mu*exp(-lam*seqD),na.rm=TRUE)
	if(denom==0){next}
	dummy <- dummy/denom
	tab_delta[i] <- (Arg(dummy)*180/pi+180)
    }

    ## Return deltas
    return(tab_delta)
}

#########DSD moments from Gamma##########################################3
gamma_DSD.moments <- function(tab_mu,tab_lam,tab_Nt,seqD,res,ii){

    ## Computes the radar delta [°] for a Gamma DSD model
    
    ## Inputs:
    ## res      = resolution of the discretization vector of D
    ## tab_mu   = vector of mu" parameters (mu = alpha-1)
    ## tab_lam  = vector of rate parameters [1/mm]
    ## tab_Nt   = vector of concentration parameters [1/m3]
    ## seqD     = diameter discretization vector [mm]
    ## ii       =order of the moment

    ##OUTPUT
    ##Tab_moments  =table of the ii-th moments

    ## Grazioli 2012

    ## Some basic tests:
    Ntab_mu  <- length(tab_mu)
    Ntab_lam <- length(tab_lam)
    Ntab_Nt  <- length(tab_Nt)
    NseqD    <- length(seqD)

    if(Ntab_mu!=Ntab_lam){stop("tab_mu and tab_lam must have same length")}
    if(Ntab_mu!=Ntab_Nt){stop("tab_mu and tab_Nt must have same length")}
    if(is.na(ii)){stop("NA value not allowed for ii")}
    if(ii<=0){stop("ii must be strictly positive")}
    if(any(is.na(seqD))){stop("NA values not allowed in seqD")}
    if(any(tab_mu<=(-1),na.rm=TRUE)){stop("values in tab_mu must be greater than -1")}
    if(any(tab_lam<0,na.rm=TRUE)){stop("values in tab_lam must be positive")}
    if(any(tab_Nt<0,na.rm=TRUE)){stop("negative values not allowed in tab_Nt")}
    if(any(seqD<=0)){stop("negative diameters are not allowed")}

    ## Compute delta for valid measurements
    tab_moments <- rep(NA,Ntab_mu)
    id     <- which(tab_mu>(-1))
    id     <- intersect(id,which(tab_lam>0))
    id     <- intersect(id,which(tab_Nt>0))
    for(i in id){
	mu  <- tab_mu[i]
	lam <- tab_lam[i]
	Nt  <- tab_Nt[i]
	

	tab_moments[i] <- sum((seqD)^(ii)*Nt*seqD^mu*exp(-lam*seqD),na.rm=TRUE)*res
    }

    ## Return deltas
    return(tab_moments)
}


####################################################################

gamma_DSD.R <- function(tab_mu,tab_lam,tab_Nt,dD=0.001){

    ## Compute the rain rate using a gamma DSD model

    ## Inputs:
    ## tab_mu  = vector of shape parameters
    ## tab_lam = vector of rate parameters
    ## tab_Nt  = vector of concentration parameters
    ## dD      = diameter discretization

    ## Output
    ## tabR = vector of rain-rates [mm/h]

    ## Remarks:
    ## (1) NA values are ignored
    ## (2) A waerning is produced for impossible mu,lam or Nt values
    ## (3) A warning is returned for rain-rate values larger than 300 mm/h

    N1 <- length(tab_mu)
    N2 <- length(tab_lam)
    N3 <- length(tab_Nt)
    if(length(unique(c(N1,N2,N3)))>1){stop("dimensions do not match")}
    N   <- N1
    id1 <- which(tab_mu>(-1))
    id2 <- which(tab_lam>0)
    id3 <- which(tab_Nt>=0)
    N1  <- length(id1)
    N2  <- length(id2)
    N3  <- length(id3)
    if(N1<N){warning(sprintf("there were %i invalid mu values",N-N1))}
    if(N2<N){warning(sprintf("there were %i invalid lam values",N-N2))}
    if(N3<N){warning(sprintf("there were %i invalid Nt values",N-N3))}
    id  <- intersect(intersect(id1,id2),id3)
    Nid <- length(id)
    tabR <- rep(NA,N)
    if(Nid>0){
	seqD <- seq(0.1,7.0,dD)
	seqV <- raindrop_velocity(seqD)
	C1   <- (1e-6)*3600*(pi/6) 
	for(i in id){
	    mu  <- tab_mu[i]
	    lam <- tab_lam[i]
	    Nt  <- tab_Nt[i]
	    C2  <- sum(seqD^mu*exp(-lam*seqD),na.rm=TRUE)
	    C3  <- Nt*sum(seqD^mu*exp(-lam*seqD)*seqV*seqD^3,na.rm=TRUE)
	    tabR[i] <- C1*C3/C2
	}
	
    }
    id <- which(tabR==Inf)
    if(length(id)>0){tabR[id] <- NA}
    id <- which(tabR<0)
    if(length(id)>0){
	tabR[id] <- NA
	warning(sprintf("there were %i negative rain rates",length(id)))
    }
#     m <- sum(tabR>300,na.rm=TRUE)
#     if(m>0){warning(sprintf("there were %i rain rates over 300 mm/h",m))}
    return(tabR)
}


###################################################################

volumic_DSD.Z <- function(DSDmatrix,tabD,w,m,tabBS){

    ## Computes the radar reflectivity derived from volumic DSD measurements
    
    ## Inputs:
    ##  DSDmatrix = Matrix containing the volumic DSD (1 row = 1 DSD spectra)
    ##    columns : diameter classes [mm] (32 for Parsivel)
    ##    rows    : number of drops per cubic meter for each diameter class
    ##  tabD      = diameter classes [mm]
    ##  w         = wavelength [cm]
    ##  m         = the complex refractive index of liquid water
    ##  tabBS     = backscattering cross-sections for each diameter class [cm^2] 
    
    ## Outputs:
    ##  tabZ = vector of radar reflectivities [dBZ]
    
    ## Some basic tests
    nrow   <- dim(DSDmatrix)[1]
    ncol   <- dim(DSDmatrix)[2]
    NtabD  <- length(tabD)
    NtabBS <- length(tabBS)
    
    if(nrow==0 || ncol==0){stop("invalid DSD matrix")}
    if(ncol!=NtabD || ncol!=NtabBS || NtabD!=NtabBS){stop("invalid dimensions")}
        
    ## Computing reflectivity for each spectra
    Kw   <- (m^2-1)/(m^2+2)
    Cz   <- 1e6*w^4/(pi^5*abs(Kw)^2)
    tabZ <- rep(NA,nrow)
    for(i in 1:nrow){
	tabN <- DSDmatrix[i,]
	if(sum(is.na(tabN))>0){next}
	if(sum(tabN)==0){tabZ[i] <- -Inf}
	if(sum(tabN)>0){tabZ[i] <- 10*log10(Cz*sum(tabN*tabBS))}
    }  
    return(tabZ) 
}


###################################################################

gamma_DSD.Z <- function(tab_shape,tab_rate,tab_Nt,wlength,m,seqD,tabBS){

    ## Computes the radar reflectivity derived from a gamma DSD model
    ## This function replaces Radar_ref_gamma() from previous lib_DSD
    ## The new version has been vectorized.
    ## The reflectivity values are now returned in dBZ
    
    ## Inputs :
    ##   tab_shape = vector of shape parameters (alpha=mu+1)
    ##   tab_rate  = vector of rate parameters (beta)
    ##   tab_Nt    = vector of concentration parameters (Nt)
    ##   wlength   = wavelength [cm]
    ##   m         = the complex refractive index of liquid water
    ##   seqD      = drop diameter discretization table [mm]
    ##   tabBS     = backscattering cross-sections [cm^2] (same size than seqD)
    
    ## Outputs:
    ##   tabZ = vector of radar reflectivities [dBZ]
    
    ## Remarks:
    ##   NA values are allowed for shape, rate and concentration parameter.
  
    l1 <- length(tab_shape)
    l2 <- length(tab_rate)
    l3 <- length(tab_Nt)
    lD <- length(seqD)
    lB <- length(tabBS)
    
    if(l1!=l2 || l1!=l3 || l2!=l3){stop("DSD parameters must have same length")}
    if(lD!=lB){stop("backscatterinbg table must have same length than seqD")}
    
    Kw   <- (m^2-1)/(m^2+2)
    Cz   <- 1e6*wlength^4*tab_Nt/(pi^5*abs(Kw)^2)
    tabZ <- rep(NA,l1)
    for(i in 1:l1){
	alpha <- tab_shape[i]
	beta  <- tab_rate[i]
	Nt    <- tab_Nt[i]
	if(!is.na(alpha) && !is.na(beta) && !is.na(Nt)){
	    num <- sum(tabBS*seqD^(alpha-1)*exp(-beta*seqD),na.rm=TRUE)
	    I   <- sum(seqD^(alpha-1)*exp(-beta*seqD),na.rm=TRUE)
	    if(!is.na(Cz[i]) && !is.na(num) && !is.na(I)){
		if(I<Inf && num<Inf && I!=0){
		    Z <- Cz[i]*num/I
		    tabZ[i] <- 10*log10(Z)
		}
	    }
	}
    }
    return(tabZ)
}

###################################################################

gamma_DSD.ZRay <- function(tab_shape,tab_rate,tab_Nt,seqD){

    ## Computes the radar Rayleigh derived from a gamma DSD model
    ## This function replaces Radar_ref_gamma() from previous lib_DSD
    ## The new version has been vectorized.
    ## The reflectivity values are now returned in dBZ
    
    ## Inputs :
    ##   tab_shape = vector of shape parameters (alpha=mu+1)
    ##   tab_rate  = vector of rate parameters (beta)
    ##   tab_Nt    = vector of concentration parameters (Nt)
    ##   seqD      = drop diameter discretization table [mm]
   
    
    ## Outputs:
    ##   tabZ = vector of radar reflectivities [dBZ]
    
    ## Remarks:
    ##   NA values are allowed for shape, rate and concentration parameter.

    ##  Grazioli, 2014
  
    l1 <- length(tab_shape)
    l2 <- length(tab_rate)
    l3 <- length(tab_Nt)
    lD <- length(seqD)
    
    
    if(l1!=l2 || l1!=l3 || l2!=l3){stop("DSD parameters must have same length")}
    
    

    Cz   <- tab_Nt
    tabZ <- rep(NA,l1)
    for(i in 1:l1){
	alpha <- tab_shape[i]
	beta  <- tab_rate[i]
	Nt    <- tab_Nt[i]
	if(!is.na(alpha) && !is.na(beta) && !is.na(Nt)){
	    num <- sum(seqD^(alpha-1+6)*exp(-beta*seqD),na.rm=TRUE)
	    I   <- sum(seqD^(alpha-1)*exp(-beta*seqD),na.rm=TRUE)
	    if(!is.na(Cz[i]) && !is.na(num) && !is.na(I)){
		if(I<Inf && num<Inf && I!=0){
		    Z <- Cz[i]*num/I
		    tabZ[i] <- 10*log10(Z)
		}
	    }
	}
    }
    return(tabZ)
}


###################################################################

gamma_DSD.Zdr <- function(tab_mu,tab_lam,wlength,m,seqD,tabBSH,tabBSV){

    ## Computes the radar differential reflectivity derived from a gamma DSD model
    
    ## Inputs :
    ## tab_mu  = vector of shape parameters (alpha=mu+1)
    ## tab_lam = vector of rate parameters (lam=beta)
    ## wlength = wavelength [cm]
    ## m       = the complex refractive index of liquid water
    ## seqD    = drop diameter discretization table [mm]
    ## tabBSH  = backscattering cross-sections for polH [cm^2] (same size than seqD)
    ## tabBSV  = backscattering cross-sections for polV [cm^2] (same size than seqD)
    
    ## Outputs:
    ## tabZdr = vector of differnetial radar reflectivities [dB]
      
    Ntab_mu  <- length(tab_mu)
    Ntab_lam <- length(tab_lam)
    NseqD    <- length(seqD)
    NtabBSH  <- length(tabBSH)
    NtabBSV  <- length(tabBSV)

    ## Some basic tests
    if(Ntab_mu!=Ntab_lam){stop()}
    if(NseqD!=NtabBSH || NseqD!=NtabBSV || NtabBSH!=NtabBSV){stop()}
    
    tabZdr <- rep(NA,Ntab_mu)
    warn <- FALSE
    for(i in 1:Ntab_mu){
	mu  <- tab_mu[i]
	lam <- tab_lam[i]
	if(is.na(mu*lam)){next}
	if(mu<=(-1) || lam<=0){warn <- TRUE}
	var1 <- sum(tabBSH*seqD^(mu)*exp(-lam*seqD),na.rm=TRUE)
	var2 <- sum(tabBSV*seqD^(mu)*exp(-lam*seqD),na.rm=TRUE)
	if(is.na(var1*var2)){next}
	if(var1==Inf || var2==Inf || var2==0){next}
	Zdr <- var1/var2
	if(is.na(Zdr)){next}
	tabZdr[i] <- 10*log10(Zdr)
    }
    if(warn==TRUE){warning("negative shape or rate parameter in DSD")}
    return(tabZdr)
}

###################################################################

volumic_DSD.A <- function(DSDmatrix,ExtCross){
    
    ## Computes the specific attenuation [dB/km] for a volumic DSD spectra
    
    ## Inputs:
    ## DSDmatrix  = matrix containing the volumic DSD (per m^3) 
    ##              1 row = 1 DSD spectra
    ##              1 column = 1 diameter class
    ## ExtCross   = extinction cross-sections [cm^2] at a given frequency 
    ##              and for each diameter class
    
    ## Outputs:
    ## tab_Att = vector of specific attenuations [dB/km] 
    
    ## Basic tests
    nrow <- dim(DSDmatrix)[1]
    ncol <- dim(DSDmatrix)[2]
    nExt <- length(ExtCross)
    if(ncol!=nExt){stop("incorrect dimensions")}
    
    ## Computing the attenuation for each DSD
    tab_Att <- rep(NA,nrow)
    for(i in 1:nrow){	    
	tab_Nt <- DSDmatrix[i,] 
	if(sum(!is.na(tab_Nt))>0){
	    if(sum(tab_Nt)>0){
		tab_Att[i] <- sum(tab_Nt*ExtCross,na.rm=TRUE)
	    }
	}
    }
    return(tab_Att)
}

###################################################################

gamma_DSD.A <- function(tab_mu,tab_lambda,tab_Nt,seqD,ExtCross){
    
    ## Computes the specific attenuation [dB/km] for a Gamma DSD model
    
    ## Inputs:
    ## tab_mu     = vector of "shape" parameters (mu = alpha-1)
    ## tab_lambda = vector of rate parameters
    ## tab_Nt     = vector of concentration parameters
    ## seqD       = diameter discretization table [mm]
    ## ExtCross   = extinction cross-sections [cm^2] at fixed frequency.
    
    ## Outputs:
    ## tab_Att = vector of specific attenuations [dB/km] 
    
    ## Basic tests
    Nmu   <- length(tab_mu)
    Nlam  <- length(tab_lambda)
    NNt   <- length(tab_Nt)
    NseqD <- length(seqD)
    NExt  <- length(ExtCross)
    if(Nmu!=Nlam){stop("incorrect dimensions")}
    if(Nmu!=NNt){stop("incorrect dimensions")}
    if(Nlam!=NNt){stop("incorrect dimensions")}
    if(NseqD!=NExt){stop("incorrect dimensions")}
    
    ## Computing the attenuation for each DSD
    tab_Att <- rep(NA,Nmu)
    for(i in 1:Nmu){
	mu  <- tab_mu[i]
	lam <- tab_lambda[i]
	Nt  <- tab_Nt[i]
	if(!is.na(mu*lam*Nt)){
	    num   <- Nt*sum(ExtCross*seqD^mu*exp(-lam*seqD),na.rm=TRUE)
	    denom <- log(10)*sum(seqD^mu*exp(-lam*seqD),na.rm=TRUE)
	    if(!is.na(num*denom)){
		if(denom!=0){
		    tab_Att[i] <- num/denom
		}
	    }
	}
    }
    return(tab_Att)
}

###################################################################

gamma_DSD.Kdp <- function(tab_mu,tab_lam,tab_Nt,seqD,tab_Shh,tab_Svv,freq){

    ## Computes the one-way specific differential phase using Gamma DSD model
    
    ## Inputs:
    ## tab_mu     = vector of "shape" parameters (mu = alpha-1)
    ## tab_lambda = vector of rate parameters
    ## tab_Nt     = vector of concentration parameters
    ## seqD       = diameter discretization table [mm]
    ## tab_Shh    = vector of forward scattering amplitudes (pol=H, only real part)
    ## tab_Svv    = vector of forward scattering amplitudes (pol=V, only real part)
    ## freq       = the frequency [GHz]

    ## Output:
    ## tab_Kdp = vector of specific differential phase shifts [°/km]

    ## Remarks:
    ## (1) This function needs the Tmatrix.R library to be loaded
    ## (2) 

    ## Author: Marc Schleiss, EPFL-LTE
    ## Last Modification: August 2010

    ## Performing some tests
    if(length(tab_mu)!=length(tab_lam)){stop("dimensions do not match")}
    if(length(tab_lam)!=length(tab_Nt)){stop("dimensions do not match")}
    if(length(tab_Shh)!=length(tab_Svv)){stop("dimensions do not match")}
    if(length(tab_Shh)!=length(seqD)){stop("dimensions do not match")}

    ## Defining some variables
    c       <- 299792458	## speed of light [m/s]
    wlength <- 1e3*c/(freq*1e9)	## wavelength [mm]
    nDSD    <- length(tab_mu)
    Ck      <- (180/pi)*wlength*1e-3    ## [m °]

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

###################################################################

gamma_DSD.Dm <- function(tab_mu,tab_lam,seqD){

    ## Computes the mean drop diameter for a gamma DSD

    ## Inputs:
    ## tab_mu  = vector of shape parameters [-]
    ## tab_lam = vector of rate parameters [mm^-1]
    ## seqD    = sequence of drop diameters [mm]

    ## Output:
    ## tab_Dm = vector of mean drop diameters [mm]

    NseqD  <- length(seqD)
    if(NseqD==0){stop("drop diameter table is empty")}
    N      <- length(tab_mu)
    if(N==0){stop("no input")}
    if(length(tab_lam)!=N){stop("dimensions of mu and lam do not match")}

    tab_Dm <- rep(NA,N)
    id     <- which(!is.na(tab_mu))
    id     <- intersect(id,which(!is.na(tab_lam)))
    for(i in id){
	mu <- tab_mu[i]
	lam <- tab_lam[i]
	a <- sum(seqD^(mu+1)*exp(-lam*seqD))
	b <- sum(seqD^mu*exp(-lam*seqD))
	if(is.na(a*b)){next}
	tab_Dm[i] <- a/b
    }
    return(tab_Dm)
}

###################################################################

gamma_DSD.D0 <- function(tab_mu,tab_lam,tab_Nt,seqD){
    
    ## Compute D0 = median water content drop diameter
    
    ## Inputs:
    ## tab_mu  = vector of shape parameters [-]
    ## tab_lam = vector of rate parameters [mm^-1]
    ## tab_Nt  = vector of concentration parameters [m^-3]
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
