# KIMBO: CAN YOU CHECK THEY CORRESPOND TO THE ONE WE USE? 
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

 
 
 