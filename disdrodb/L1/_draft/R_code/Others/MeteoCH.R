###############################################################################
############################# Meteo Swiss Library #############################
############################### by Marc Schleiss ##############################
################################# 04 Oct 2010 #################################
###############################################################################

## See the reference document: 
## "Operational Use of Radar for Precipitation Measurements in Switzerland"
## by Jürg Joss, Bruno Schädler, Gianmario Galli, Remo Cavalli, and others.
## Locarno, 23.Sept 1997

######################
## Global variables ##
######################

NclassR <- 16			## number of rain rate classes [-]
NclassZ <- 16			## number of reflectivity classes [-]
Nelev   <- 20			## number of elevations [-]
low.R <- rep(NA,NclassR)	## lower rain rate limits [mm/h]
low.Z <- rep(NA,NclassZ)	## lower reflectivity limits [dBZ]
up.R  <- rep(NA,NclassR)	## upper rain rate limits [mm/h]
up.Z  <- rep(NA,NclassZ)	## upper reflectivity limits [dBZ]
meanR <- rep(NA,NclassR)	## mean rain rate classes [mm/h]
meanZ <- rep(NA,NclassR)	## mean reflectivity classes [dBZ]
tab_elev  <- rep(NA,Nelev)	## elevation angles [°]
tab_range <- rep(NA,Nelev)	## max. range per elevation [km]
Scan_Time <- rep(NA,Nelev)	## scan time per elevation [s]
Step_Time <- rep(NA,Nelev)	## step time per elevation [s]

#######################
## Rain rate classes ##
#######################

## Lower & Upper limits
## All values are in mm/h
## For convenience, I put up.R[1]=0

low.R[1] <- 0
low.R[2] <- 0.16
low.R[3] <- 0.25
low.R[4] <- 0.40
low.R[5] <- 0.63
low.R[6] <- 1.0
low.R[7] <- 1.6
low.R[8] <- 2.5
low.R[9] <- 4.0
low.R[10] <- 6.3
low.R[11] <- 10
low.R[12] <- 16
low.R[13] <- 25
low.R[14] <- 40
low.R[15] <- 63
low.R[16] <- 100

up.R <- low.R[2:16]
up.R <- c(up.R,100)
up.R[1] <- 0

meanR <- (up.R+low.R)/2

###########################
## Reflectivity classes  ##
###########################

## Lower and Upper limits
## All values are in dBZ
## For convenience, I put up.Z[1]=0

low.Z[1] <- 0
low.Z[2] <- 13
low.Z[3] <- 16
low.Z[4] <- 19
low.Z[5] <- 22
low.Z[6] <- 25
low.Z[7] <- 28
low.Z[8] <- 31
low.Z[9] <- 34
low.Z[10] <- 37
low.Z[11] <- 40
low.Z[12] <- 43
low.Z[13] <- 46
low.Z[14] <- 49
low.Z[15] <- 52
low.Z[16] <- 55

up.Z <- low.Z[2:16]
up.Z <- c(up.Z,Inf)
up.Z[1] <- 0

meanZ <- (low.Z+up.Z)/2

############################
## Radar Elevation Angles ##
############################

tab_elev[1] <- -0.3
tab_elev[2] <- 1.5
tab_elev[3] <- 3.5
tab_elev[4] <- 5.5
tab_elev[5] <- 7.5
tab_elev[6] <- 9.5
tab_elev[7] <- 13.0
tab_elev[8] <- 18.3
tab_elev[9] <- 25.3
tab_elev[10] <- 34.5
tab_elev[11] <- 0.5
tab_elev[12] <- 2.5
tab_elev[13] <- 4.5
tab_elev[14] <- 6.5
tab_elev[15] <- 8.5
tab_elev[16] <- 11.0
tab_elev[17] <- 15.5
tab_elev[18] <- 21.6
tab_elev[19] <- 29.6
tab_elev[20] <- 40.0

##################################################
## Radar max. range resolution w.r.t. elevation ##
##################################################

tab_range[1] <- 230
tab_range[2] <- 230
tab_range[3] <- 162
tab_range[4] <- 112
tab_range[5] <- 85
tab_range[6] <- 68
tab_range[7] <- 51
tab_range[8] <- 37
tab_range[9] <- 27
tab_range[10] <- 20
tab_range[11] <- 230
tab_range[12] <- 205
tab_range[13] <- 133
tab_range[14] <- 97
tab_range[15] <- 76
tab_range[16] <- 59
tab_range[17] <- 43
tab_range[18] <- 31
tab_range[19] <- 23
tab_range[20] <- 18

#############################
## Scan Time per elevation ##
#############################

Scan_Time[1] <- 20
Scan_Time[2] <- 20
Scan_Time[3] <- 15
Scan_Time[4] <- 15
Scan_Time[5] <- 10
Scan_Time[6] <- 10
Scan_Time[7] <- 10
Scan_Time[8] <- 10
Scan_Time[9] <- 10
Scan_Time[10] <- 10
Scan_Time[11] <- 20
Scan_Time[12] <- 20
Scan_Time[13] <- 15
Scan_Time[14] <- 15
Scan_Time[15] <- 10
Scan_Time[16] <- 10
Scan_Time[17] <- 10
Scan_Time[18] <- 10
Scan_Time[19] <- 10
Scan_Time[20] <- 10

#############################
## Step time per elevation ##
#############################

Step_Time[1] <- 1.4
Step_Time[2] <- 1.4
Step_Time[3] <- 1.4
Step_Time[4] <- 1.4
Step_Time[5] <- 1.4
Step_Time[6] <- 1.7
Step_Time[7] <- 2.0
Step_Time[8] <- 2.2
Step_Time[9] <- 2.4
Step_Time[10] <- 4.2
Step_Time[11] <- 1.4
Step_Time[12] <- 1.4
Step_Time[13] <- 1.4
Step_Time[14] <- 1.4
Step_Time[15] <- 1.5
Step_Time[16] <- 1.9
Step_Time[17] <- 2.1
Step_Time[18] <- 2.3
Step_Time[19] <- 2.6
Step_Time[20] <- 4.5

#########################
## Locations of Radars ##
#########################

## see page 71

################################################################################
######################## Useful functions and routines #########################
################################################################################ 

read_MeteoCH_Rain <- function(file){

    ## Reads a Meteo Swiss radar rain-rate composite map

    ## Input:
    ## file: filename to read (full path)
    
    ## Output
    ## radarR = matrix with 3 columns: (x|y|R)
    ## (x,y) are given in Swiss coordinate system [km], R is given in mm/h.
    ## Pixel size is 1km. NA and zero rain-rate values are kept.

    ## Comments: 
    ## (1) This function needs the rgdal library to be loaded

    ## Read GIF image file using the rgdal library
    gif <- readGDAL(file,silent=TRUE)
    gif <- as.data.frame(gif)
    names(gif) <- c("code","x","y")
    coded_values <- matrix(gif[["code"]],nrow=578,ncol=650,byrow=TRUE)
    matX <- matrix(gif[["x"]],nrow=578,ncol=650,byrow=TRUE)
    matY <- matrix(gif[["y"]],nrow=578,ncol=650,byrow=TRUE)

    ## Extract the 538x610 pixels at the bottom left
    ## These represent the coded rain-rate values
    ## Remaining values (side projections) are not needed here
    coded_values <- coded_values[1:538,1:610]
    matX  <- matX[1:538,1:610]
    matY  <- matY[1:538,1:610]
    tabX  <- as.vector(matX)
    tabY  <- as.vector(matY)
    Ngrid <- length(coded_values)

    ## Convert the coded values into rain-rate values. The rain-rate values 
    ## are coded on 16 levels from 0-15. The equivalence table is given on 
    ## p.51 of the Meteo Swiss report.
    tabR <- rep(NA,Ngrid)
    for(i in 1:16){tabR[coded_values==(i-1)] <- meanR[i]}

    ## Convert coordinates into Swiss national system
    tabX <- tabX+297.5
    tabY <- tabY-100.5

    ## return rain rate matrix
    radarR <- matrix(c(tabX,tabY,tabR),nrow=Ngrid,ncol=3)
    return(radarR)
}

################################################################################

read_MeteoCH_Ref <- function(date,n.elev){

    ## Reads the Meteo Swiss radar reflectivity data.
    ## For now, only data from La Dôle is available.

    ## Input:
    ## date = GMT time and date ending by 3 or 8, e.g. "201004280018" or "201004280023"
    ## n.elev = elevation (1-20)

    ## Output:
    ## A list with 2 elements:
    ## 1st element: the time (in seconds since 1970, GMT time zone)
    ## 2nd element: a matrix with 3 columns: azimuth|range|reflectivity

    ## Comments: 
    ## (1) This function needs the rgdal library to be loaded
    ## (2) This function needs the /net/ltesrv1 to be mounted
    ## (3) This function needs ark to be installed.
    ## (4) The data is copied in the current directory, unzipped
    ##     and extracted. At the end, the unzipped files are destroyed.

    path  <- "/net/ltesrv1/data/RadarMeteoSwiss/Raw_data"
    year  <- substr(date,1,4)
    month <- substr(date,5,6)
    day   <- substr(date,7,8)
    hour  <- substr(date,9,10)
    min   <- substr(date,11,12)
    file  <- sprintf("VIBZ02.%s",date)
    name  <- sprintf("%s/%s-%s/%s",path,year,month,file)
    pwd   <- system("pwd",intern=TRUE)

    if(nchar(pwd)==nchar(sprintf("%s/%s-%s",path,year,month))){
	if(pwd==sprintf("%s/%s-%s",path,year,month)){stop("please run the script from another path")}
    }

    ## Copy the archive to the current directory
    system(sprintf("cp %s %s",name,pwd))

    ## Extract the archive to current directory
    system(sprintf("ark -b %s/%s",pwd,file))
    
    tab_elev.id <- c(1:9,c("A","B","C","D","E","F","G","H","I","J","K"))
    elev    <- tab_elev[n.elev]
    elev.id <- tab_elev.id[n.elev]

    data <- readGDAL(system(sprintf("ls gZ%sD*",elev.id),intern=TRUE),silent=TRUE)
    data <- as.data.frame(data)
    names(data) <- c("code","range","azimuth")

    ## Transform the coded values into reflectivities:
    ## L=0: below minimum detectable signal (Z=-Inf)
    ## L=1: -35.5 dBZ
    ## 1<L<250: Z=0.5*(L-0.5)-35.75
    ## L=250: 89 dBZ or more
    ## L=251-255: NA

    tabL  <- data[["code"]]
    NtabL <- length(tabL)
    tabZ <- rep(-Inf,NtabL)
    tabZ[tabL==1] <- -35.5
    tabZ[tabL==250] <- 89.0
    tabZ[tabL>250] <- NA
    id <- intersect(which(tabL>1),which(tabL<250))
    if(length(id)>0){tabZ[id] <- 0.5*(tabL[id]-0.5)-35.75}

    date <- sprintf("%s-%s-%s %s:%s:00",year,month,day,hour,min)

    time <- as.POSIXct(date,origin="1970-01-01",tz="GMT")
    time <- as.integer(as.numeric(time))

    ## Determine the exact scan time
    time1 <- sum(Scan_Time[1:n.elev])
    time2 <- 0
    if(n.elev>1){time2 <- sum(Step_Time[1:(n.elev-1)])}
    time3 <- time1+time2
    time <- time-300+time3

    ## Delete all temporary files
    system(sprintf("rm %s/gZ*.prd",pwd))
    system(sprintf("rm %s/%s",pwd,file))

    ## Return the extracted data
    radarZ <- c(data[["azimuth"]],data[["range"]],tabZ)
    radarZ <- matrix(radarZ,nrow=NtabL,ncol=3)
    list_output <- vector("list",2)
    list_output[[1]] <- time
    list_output[[2]] <- radarZ
    return(list_output)
}
 
