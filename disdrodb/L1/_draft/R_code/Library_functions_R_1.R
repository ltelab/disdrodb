################### Usefull additional Library #####################
path_lib <- "/USERS/lte/commun1/Prog_com/Lib_R/"
# path_lib <- "/home/jaffrain/Lib_R/"
source(paste(path_lib,"lib_DSD.R",sep=""))
source(paste(path_lib,"new_lib_DSD.R",sep=""))
source(paste(path_lib,"Tmatrix.R",sep=""))
# source("/USERS/jaffrain/R/LibParsivel.R")

########################## Object_size ##############################
object_size <- function(list_objects){
	## Aim: Give an estimation of objects size in R environment and return it.
	## Inputs: a vector with the list of objects names (can be generated in the script using 'ls()' function)
	## Outputs: a matrix with the names of the objects (1st col) and their corresponding size in bytes (2nd col).
	# Written by Joel Jaffrain, April 2009
	# Last update: April,06th 2009

	memory_obj <- matrix(NA,ncol=2,nrow=length(list_objects))
	memory_obj[,1] <- list_objects
	for (p in 1:length(list_objects)){
		memory_obj[p,2] <- object.size(get(list_objects[p]))
	}
	return(memory_obj)
}

########################## Calculation of rain amounts from R ##############################
rain_amounts <- function(vect_R,dt_datafile){
	## Inputs:
	##   'vect_R': 	    vector with the rain rate values for each measurement [mm/h]
	## Optional arguments:
	##   'dt_datafile': sampling resolution in SEC (by default=20s) [SEC]
	## Outputs:
	##   'amount': a vector with the cumulative rain [mm]
	# Written by Joel Jaffrain, April 2009
	# Last update: April,06th 2009

	dim <- length(vect_R)
	amount_inc <- vector(mode = "numeric", length = dim)
	pos <- is.na(vect_R) == FALSE

	amount_inc[pos] <- vect_R[pos] * (dt_datafile / 3600)	# Amount(t) = R(t) * d_t
	amount <- round(cumsum(amount_inc),digits=3)
	return(amount)
}


########################## Parsivel_classes ##############################
Parsivel_classes <- function(test=0){
	## Equivolumetric drop diameter classes in mm ###
	class_size <- vector(mode = "numeric", length = 32)
	class_size[1] = 0.062
	class_size[2] = 0.187
	class_size[3] = 0.312
	class_size[4] = 0.437
	class_size[5] = 0.562
	class_size[6] = 0.687
	class_size[7] = 0.812
	class_size[8] = 0.937
	class_size[9] = 1.062
	class_size[10] = 1.187
	class_size[11] = 1.375
	class_size[12] = 1.625
	class_size[13] = 1.875
	class_size[14] = 2.125
	class_size[15] = 2.375
	class_size[16] = 2.750
	class_size[17] = 3.250
	class_size[18] = 3.750
	class_size[19] = 4.250
	class_size[20] = 4.750
	class_size[21] = 5.500
	class_size[22] = 6.500
	class_size[23] = 7.500
	class_size[24] = 8.500
	class_size[25] = 9.500
	class_size[26] = 11.000
	class_size[27] = 13.000
	class_size[28] = 15.000
	class_size[29] = 17.000
	class_size[30] = 19.000
	class_size[31] = 21.500
	class_size[32] = 24.500
	
	## Classes Spread in mm
	class_size_spread <- c(rep(0.125,10),rep(0.250,5),rep(0.500,5),rep(1.000,5),rep(2.000,5),rep(3.000,2))

	## Fall velocity classes in m.s^-1 ###
	class_speed <- vector(mode = "numeric", length = 32)
	class_speed[1] = 0.050
	class_speed[2] = 0.150
	class_speed[3] = 0.250
	class_speed[4] = 0.350
	class_speed[5] = 0.450
	class_speed[6] = 0.550
	class_speed[7] = 0.650
	class_speed[8] = 0.750
	class_speed[9] = 0.850
	class_speed[10] = 0.950
	class_speed[11] = 1.100
	class_speed[12] = 1.300
	class_speed[13] = 1.500
	class_speed[14] = 1.700
	class_speed[15] = 1.900
	class_speed[16] = 2.200
	class_speed[17] = 2.600
	class_speed[18] = 3.000
	class_speed[19] = 3.400
	class_speed[20] = 3.800
	class_speed[21] = 4.400
	class_speed[22] = 5.200
	class_speed[23] = 6.000
	class_speed[24] = 6.800
	class_speed[25] = 7.600
	class_speed[26] = 8.800
	class_speed[27] = 10.400
	class_speed[28] = 12.000
	class_speed[29] = 13.600
	class_speed[30] = 15.200
	class_speed[31] = 17.600
	class_speed[32] = 20.800
    
    list <- list(class_size,class_speed,class_size_spread)
    names(list) <- c("class_size","class_speed","class_size_spread")
    return(list)
}
######################################################

########################## Extraction of DSD spectrum from Parsivel ##############################
extract_DSD <- function(frame_N,frame_V,frame_DSD,measurements=seq(1,length(frame_N),1)){
	## Inputs:
	##   'frame_N': A vector of Field_N in 'characters' format
	##   'frame_V': A vector of Field_V in 'characters' format
	##   'frame_DSD': A vector of DSD in 'characters' format
	## Optional arguments:
	##   'measurements': Measurements of interest to extract (position of interesting measurements), by default= all.
	## Outputs:
	##   A list containing:
	##   	-> 'FieldN': Field_N [m^-3.mm^-1]
	##   	-> 'FieldV': Field_V [m.s^-1]
	##   	-> 'DSD': Surfacic DSD Spectrum 
	
	# Written by Joel Jaffrain, June 2009
	# Last update: July,02nd 2009
	
    ## Check that input vectors have the same lengths.
    dim_1 <- length(frame_N)
    dim_2 <- length(frame_V)
    dim_3 <- length(frame_DSD)
    cond <- c(dim_1==dim_2,dim_1==dim_3,dim_2==dim_3)
    if(prod(cond)!=1){
      stop(paste("Warning: frame_N, frame_V and frame_DSD do not have the same size! (",paste(dim_1,dim_2,dim_3,sep=",")," respectively)",sep=""))
    }else{
      dim_com <- dim_1
    }
    ## Check that input vectors have the correct format (i.e. 'character').
    cond <- is.character(frame_N)*is.character(frame_V)*is.character(frame_DSD)
    if(sum(cond)!=1){
      stop(paste("Warning: frame_N, frame_V and frame_DSD do not have 'character' format!",sep=""))
    }

#     ## Detect NA values
#      <- is.na(R)
    
    DSD <-  data.frame(matrix(0,nrow=dim_com,ncol=32))
    FieldV  <-  matrix(NA,nrow=dim_com,ncol=32)		# Create or reset 'Field_V' matrix
    FieldN  <-  matrix(NA,nrow=dim_com,ncol=32)		# Create or reset 'Field_N' matrix

    for (lrow in measurements) {		# row is the line number of the interesting measurement in zoom_data
# 	print(paste(lrow,"/",length(measurements)))
	DSD_tmp      <- as.character(frame_DSD[lrow])	# Define the string extract from data file as a character class
	Fv   <- as.character(frame_V[lrow])
	Fn   <- as.character(frame_N[lrow])
	DSD_tmp     <- unlist(strsplit(DSD_tmp,","))		# transform the latter string 'toto' in a vector (separator is ",")
	Fv     <- unlist(strsplit(Fv,","))
	Fn      <- unlist(strsplit(Fn,","))
	DSD_tmp <- as.numeric(DSD_tmp)			# Convert the vector of string 'temp_vect' in a numeric vector containing DSD data
	FieldN[lrow,]  <- round(10^(as.numeric(Fn)),digits=4)	## Field_N_filtered = 10^Field_N_file !!!	Field_N Conversion !!!
	FieldV[lrow,]  <- as.numeric(Fv)
	
    ## Create a frame "DSD[]" containing N(D) (= sum of each of the 32 class size) with the appropriate date   => DON'T USE FALL SPEED INFORMATIONS
    ##    -> each line of "DSD[]" correspond to a measurement
	DSD_mat   <-  matrix(data=DSD_tmp,nrow=32,ncol=32,byrow=TRUE)	# Write DSD data from vector 'DSD_tmp' into a 32x32 matrix called 'DSD_mat'
	DSD[lrow,] <- colSums(DSD_mat)
    }
    
    list <- list(FieldN,FieldV,DSD)
    names(list) <- c("FieldN","FieldV","DSD")
    return(list)
}

########################## Extraction of raw DSD spectrum from Parsivel ##############################
extract_rawDSD <- function(frame_DSD,measurements=seq(1,length(frame_DSD),1)){
	## Inputs:
	##   'frame_DSD': A vector of DSD in 'characters' format
	## Optional arguments:
	##   'measurements': Measurements of interest to extract (position of interesting measurements), by default= all.
	## Outputs:
	##   'DSDraw': a matrix with the 1024 values (32 x 32 classes) for each measurement.
	##   rawdata in the form: (v1,D1), (v1,D2),...,(v1,D32),(v2,D1),(v2,D2),...,(v2,D32),[...],(v32,D1),(v32,D2),...,(v32,D32).  <= 1024 values !
	
	# Written by Joel Jaffrain, June 2010
	# Last update: August,7th 2010
	
#     dim_com <- length(frame_DSD)
    dim_com <- length(measurements)
    ## Check that input vectors have the correct format (i.e. 'character').
    cond <- is.character(frame_DSD)
    if(sum(cond)!=1){
      stop(paste("Warning (fct 'extract_rawDSD'): frame_DSD is not a 'character'!",sep=""))
    }


    DSD_tmp  <- as.character(frame_DSD[measurements])
    DSD_tmp2  <- unlist(strsplit(DSD_tmp,","))
    num <- as.numeric(DSD_tmp2)
    DSDraw <- matrix(num,nrow=dim_com,ncol=1024,byrow=TRUE)

# # print(paste("Method 2",Sys.time()))	##
# # measurements <- 1:length(frame_DSD)
# #     DSDraw <-  data.frame(matrix(0,nrow=dim_com,ncol=1024))
# #     for (lrow in measurements) {		# row is the line number of the interesting measurement in zoom_data
# # 	DSD_tmp      <- as.character(frame_DSD[lrow])	# Define the string extract from data file as a character class
# # 	DSD_tmp     <- unlist(strsplit(DSD_tmp,","))		# transform the latter string 'toto' in a vector (separator is ",")
# # 	DSDraw[lrow,] <- as.numeric(DSD_tmp)			# Convert the vector of string 'temp_vect' in a numeric vector containing DSD data
# # 	
# # #     ## Create a frame "DSD[]" containing N(D) (= sum of each of the 32 class size) with the appropriate date   => DON'T USE FALL SPEED INFORMATIONS
# # #     ##    -> each line of "DSD[]" correspond to a measurement
# # # 	DSD_mat   <-  matrix(data=DSD_tmp,nrow=32,ncol=32,byrow=TRUE)	# Write DSD data from vector 'DSD_tmp' into a 32x32 matrix called 'DSD_mat'
# # # 	DSD[lrow,] <- colSums(DSD_mat)
# #     }
# # print(paste("End Method 2",Sys.time()))	##
   
    return(DSDraw)
}

########################## Filtering absurd measurements ##############################
# The main objective of this function is to filter the measurements for which a significant difference is observed between 'Field_N' and guaranted parameters (R and Z).
filter_absurd_meas <- function(R,FieldN,tol.err=0.3){
	## Inputs:
	##   'R': A vector with rain rate values.
	##   'FieldN': A matrix with the concentration of drop according to their size [m^-3 mm^-1]. Class sizes are provided in column.
	## Optional arguments:
	##   'tol.err': The tolerated error in % (value from 0 to 1, default=0.3) comparing DSD parameters calculated from DSD measurements (FieldN) to moments of DSD provided by Parsivel.
	## Outputs: a 'list' with the following components:
	##   'seqdel': A numeric vector giving the position (rows) of measurements to be removed (not in agreement with the filter).
	##   'n_seq': A numeric vector giving the number of measurements to be removed.
	##   'per': A numeric vector giving the corresponding percentage of meas that will be removed.

	# Written by Joel Jaffrain, December 2009
	# Last update: December,17th 2009

  ###### Consistency of inputs #######
    ## Check that 'R' is vector.
    if(is.vector(R)==FALSE){
      stop("'R' is not a vector.")
    }
    ## Check dimension of 'FieldN' parameter
    if (length(dim(FieldN))!=2){
      stop("'FieldN' do not have 2 dimensions !")
    }else{
      if (dim(FieldN)[1]!=length(R)){
	stop("'FieldN' and 'R' do not have the same number of measurements.")
      }else{
	n_meas <- length(R)
      }
    }

  ## Constants
    out_Parsivel <- Parsivel_classes()
    class_size <- out_Parsivel$class_size			# define class-size of volume-equivalent diameter in mm
#     class_size <- Parsivel_classes$class_size			# define class-size of volume-equivalent diameter in mm

  ###### Filter body #######
    R <- as.numeric(R)
    ## Calculation of moments of DSD from 'FieldN'
    drop_vel <- NA
    for (diam in 1:32){drop_vel[diam] <- raindrop_velocity(class_size[diam])}
    Rdsd  <- rain_fromDSD(FieldN,drop_vel,input.matrix=TRUE)	        ## Rain rate R from DSD [mm.h^-1]

    diff <- Rdsd-R
    norm <- (Rdsd-R)/R

    t1 <- norm >= tol.err	
    t2 <- norm <= -tol.err		
    t <- as.logical(t1 + t2)
    t[is.na(t)] <- FALSE
    seq <- seq(1,n_meas,1)
    seqdel <- seq[t]
    n_seq <- length(seqdel)  	# number of measuremnets to remove
    n_tot <- length(R[is.na(R)==FALSE])
    per <- (n_seq*100)/n_tot

    ret <- list(seqdel,n_seq,per)
    names(ret) <- c("seqdel","n_seq","per")
    return(ret)
}


########################## Filtering rain data ##############################
filter_data <- function(data_R_allstation,data_precip_type,status_station,thresholds,code_precip=65,filter="prod",miss_meas=0.2,defrain=2){
	## Inputs:
	##   'data_R_allstation': A data.frame with rain rate values from each station (from column 2 to n_station). 1st col= Time Stamp.
	##   'data_precip_type':  A data.frame with rain type values from each station (from column 2 to n_station). 1st col= Time Stamp.
	##   'status':  A data.frame with status values from each station (from column 2 to n_station). 1st col= Time Stamp.
	##   'thresholds': A numeric vector with 2 values (min and max) of threshold in [mm/h].
	## Optional arguments:
	##   'code_precip': The code wich define the type of precipitation to keep, by default= 65 (rain).
	##   'filter': The type of filter to be used: 'sum'(at least one station must respect conditions) or 'prod'(all stations must respect conditions), by default= "prod".
	##   'miss_meas': maximum percentage of missing measurements allowed, by default= 0.2 (20%); otherwise stop program.
	##   'defrain' : the number of station that have to record rain to keep the measurement.
	## Outputs:
	##   'seq_indic': A numeric vector giving the position (rows) of measurements to be kept.

	# Written by Joel Jaffrain, June 2009
	# Last update: October,21st 2009

  ###### Consistency of inputs #######
    ## Check that 'data_R_allstation' and 'data_precip_type' have the same dimensions.
    if( (dim(data_R_allstation)[1]!=dim(data_precip_type)[1]) || (dim(data_R_allstation)[2]!=dim(data_precip_type)[2])  || (dim(data_R_allstation)[1]!=dim(status_station)[1]) ){
      stop(paste("'data_R_allstation'(",dim(data_R_allstation)[1],") and/or 'data_precip_type'(",dim(data_precip_type)[1],") and/or 'station status'(",dim(status_station)[1],") do not have the same dimensions"))
    }else{
      dim_com <- dim(data_R_allstation)[1]
      n_station <- dim(data_R_allstation)[2]-1
    }
    ## Check dimension of 'thresholds' parameter
    if (length(thresholds)!=2){
      stop("You must provide min AND max values of thresholds")
    }else{
      threshold_min_filter <- thresholds[1]
      threshold_max_filter <- thresholds[2]
    }

    ## Check if a station has to much missing measurements
    if (n_station == 1){toto <- data_R_allstation[,2:(n_station+1)]}else{toto <- apply(data_R_allstation[,2:(n_station+1)],1,sum,na.rm=TRUE)}
    rain <- which(toto > 0)
    n_rain <- length(rain)
    for (s in 1:n_station){
      search_NA <- length(which(is.na(data_R_allstation[rain,(s+1)]))) >= miss_meas*n_rain	# If more than 20% of the rainy measurements is missing (filled with NA), advertise user in order to remove this station from analyses.
      if (search_NA == TRUE){
	      print(c(paste("!!! WARNINGS (filter fct): Station ",station[s]," has too many missing measurements (NA) for reliable statistical analyses !!!",sep=""),paste("Miss meas =",length(which(is.na(data_R_allstation[rain,s+1]))),"/",n_rain,"rainy measurements !!")))
	      stop()
      }
    }

  ###### Filter body #######
    ## Filtering Part I: Rain rate values.
    # Condition 1.1: NA values
    cond_1_1 <- is.na.data.frame(data_R_allstation[,2:(n_station+1)]) == FALSE	# Check that all stations have stored data (no NA value).
    if (n_station == 1){log_cond_1_1 <- cond_1_1}else{log_cond_1_1 <- apply(cond_1_1,1,prod,na.rm=FALSE)}	## Fixed: ALL STATIONS have to respect the condition
    final_cond_1_1 <- log_cond_1_1 > 0		# 'TRUE' when the corresponding measurements have to be kept !
    print(paste("Cond 1.1 (NA):",round(sum(final_cond_1_1,na.rm=TRUE)*100/dim_com,digits=1),"% kept."))
    ## [1.1]
    data_R_allstation[which(final_cond_1_1==FALSE),2:(n_station+1)] <- NA		## Remove measurements that are not respecting the filter (missing meas & non rainy period)
    data_precip_type[which(final_cond_1_1==FALSE),2:(n_station+1)]  <- NA		## Remove measurements that are not respecting the filter (missing meas & non rainy period)
    # Condition 1.2: 0 values
    cond_1_2 <- (data_R_allstation[,2:(n_station+1)]) > 0	# Remove non rainy period, i.e. when less than 'defrain' STATIONS have recorded R>0.
    if (n_station == 1){log_cond_1_2 <- cond_1_2}else{log_cond_1_2 <- apply(cond_1_2,1,sum,na.rm=FALSE)}	## Fixed: AT LEAST 'defrain' STATIONS have to respect the condition
    final_cond_1_2 <- log_cond_1_2 >= defrain		# 'TRUE' when the corresponding measurements have to be kept !  --> required positive rain rate from at least 'defrain' stations.
    print(paste("Cond 1.2 (Rainy [at least",defrain," station(s)]):",round(sum(final_cond_1_2,na.rm=TRUE)*100/dim_com,digits=1),"% kept."))
    ## [1.2]
    data_R_allstation[which(final_cond_1_2==FALSE),2:(n_station+1)] <- NA		## Remove measurements that are not respecting the filter (missing meas & non rainy period)
    data_precip_type[which(final_cond_1_2==FALSE),2:(n_station+1)]  <- NA		## Remove measurements that are not respecting the filter (missing meas & non rainy period)
    # # Total Condition 1: remove all measurements that are not respecting Cond 1
    dim_stat <-sum(final_cond_1_1 * final_cond_1_2,na.rm=TRUE)
    print(paste("=> Total Cond 1:",round(dim_stat*100/dim_com,digits=1),"% of measurements kept;","corresponding to",dim_stat,"over",dim_com,"measurements."))
    final_1 <- which(as.logical(final_cond_1_1 * final_cond_1_2))

    # Condition 2: threshold min
    if (is.na(threshold_min_filter) == FALSE){
	if (threshold_min_filter == 0){
		cond_2 <- (data_R_allstation[,2:(n_station+1)] > threshold_min_filter)		# Give positions where the measurements have to be kept !
	}else{
		cond_2 <- (data_R_allstation[,2:(n_station+1)] >= threshold_min_filter)		# Give positions where the measurements have to be kept !
	}
	if (n_station == 1){log_cond_2 <- cond_2}else{log_cond_2 <- apply(cond_2,1,ans_filt,na.rm=FALSE)}	## Depending of the answer ('ans_filt') specified at the beginning of the script
	final_cond_2 <- log_cond_2 > 0		# 'TRUE' when the corresponding measurements have to be kept !
	print(paste("Cond 2:",round(sum(final_cond_2,na.rm=TRUE)*100/dim_stat,digits=1),"% kept."))
	}else{		# # No filter if 'threshold_min_filter == NA'
	final_cond_2 <- rep(TRUE,dim_com)		# 'TRUE' when the corresponding measurements have to be kept !
    }

    # Condition 3:threshold max
    if (is.na(threshold_max_filter) == FALSE){
	cond_3 <- (data_R_allstation[,2:(n_station+1)] <= threshold_max_filter)		# Give positions where the measurements have to be kept !
	if (n_station == 1){log_cond_3 <- cond_3}else{log_cond_3 <- apply(cond_3,1,prod,na.rm=FALSE)}	## Fixed: ALL STATIONS have to respect the condition
	final_cond_3 <- log_cond_3 > 0		# 'TRUE' when the corresponding measurements have to be kept !
	print(paste("Cond 3:",round(sum(final_cond_3,na.rm=TRUE)*100/dim_stat,digits=1),"% kept."))
    }else{
	final_cond_3 <- rep(TRUE,dim_com)		# 'TRUE' when the corresponding measurements have to be kept !
    }

    ## Filtering Part II: filtering data according to precipitation type 
    if (is.na(code_precip) == FALSE){
	# cond_4 <- ((data_precip_type[,2:(n_station+1)] >= code_precip[1]) && (data_precip_type[,2:(n_station+1)] <= code_precip[2]))
	cond_4 <- (data_precip_type[,2:(n_station+1)] <= code_precip)
	if (n_station == 1){log_cond_4 <- cond_4}else{log_cond_4 <- apply(cond_4,1,prod,na.rm=FALSE)}	## Fixed: ALL STATIONS have to respect the condition
	final_cond_4 <- log_cond_4 > 0		# 'TRUE' when the corresponding measurements have to be kept !
	print(paste("Cond 4:",round(sum(final_cond_4,na.rm=TRUE)*100/dim_stat,digits=1),"% kept."))
    }else{
	final_cond_4 <- rep(TRUE,dim_com)		# 'TRUE' when the corresponding measurements have to be kept !
    }

    ## Filtering Part III: filtering data according to station status
    cond_5 <- (status_station[,2:(n_station+1)] <= 1)
    if (n_station == 1){log_cond_5 <- cond_5}else{log_cond_5 <- apply(cond_5,1,prod,na.rm=FALSE)}	## Fixed: ALL STATIONS have to respect the condition
    final_cond_5 <- log_cond_5 > 0		# 'TRUE' when the corresponding measurements have to be kept !
    print(paste("Cond 5 (all meas):",round(sum(final_cond_5,na.rm=TRUE)*100/dim_com,digits=1),"% kept."))
    print(paste("Cond 5 (rainy meas):",round(sum(final_cond_5[final_1],na.rm=TRUE)*100/dim_stat,digits=1),"% kept."))

    ## All Conditions:
    all_cond <- as.logical(final_cond_1_1 * final_cond_1_2 * final_cond_2 * final_cond_3 * final_cond_4 * final_cond_5)
    all_cond[is.na(all_cond)] <- FALSE
# print("after all_cond")
    seq_indic <- seq(1,dim_com)[all_cond]

    ## Impact of filter: advertize if too many measurements were removed
    if(length(seq_indic) <= 0.8*dim_stat){	# if less than 80 % of rainy measurements are kept.
      print(paste("!!! WARNINGS (filter fct): Only",round(length(seq_indic)*100/dim_stat,digits=1),"% of rainy measurements kept."))
#       stop("!!! WARNINGS (filter fct): less than 80% of measurements were keep after filtering process...")
    }else{
      print(paste("Total filter fct:",round(length(seq_indic)*100/dim_stat,digits=1),"% of rainy measurements kept."))
    }
    return(seq_indic)
}


########################## Calculation of drop diameter quantiles ##############################
quant_diameters <- function(DSD,probs,input.matrix=FALSE){
	## Inputs:
	##   'DSD': A matrix with Volumic DSD data. Each row corresponds to a measurements, column are the diameter classes. DSD has to be given in [m-3.mm-1]
	## Optional arguments:
	##   'input.matrix': logical, is input DSD a matrix ? by default= FALSE (i.e. vector of length 32).
	## Outputs:
	##   'Q': A vector of quantiles of diameter values in [mm] according to probabilities in input.

	# Written by Joel Jaffrain, March 2010
	# Last update: March, 16th 2010

  ## Check dimension of DSD matrix:
  if (input.matrix == TRUE){
      if (dim(DSD)[2] != 32){stop("Provided DSD matrix doesn't have 32 classes of diameter.")}
      n_meas <- dim(DSD)[1]
  }else{
      if (length(DSD) != 32){stop("Provided DSD vector doesn't have 32 classes of diameter.")}
      n_meas <- 1
  }

  ## Check consistency of probabilities
  if((sum(probs < 0)!=0) || (sum(probs > 1)!=0)){stop("You must provide probilities in the interval [0,1] !")}
  if(is.numeric(probs) == FALSE){stop("Probabilities must be numeric values !")}

  out_Parsivel <- Parsivel_classes()
  class_spread <- out_Parsivel$class_size_spread
  class_size <- out_Parsivel$class_size	## Center of diameter classes in [mm]
#   class_spread <- Parsivel_classes$class_size_spread
#   class_size <- Parsivel_classes$class_size	## Center of diameter classes in [mm]
  n_probs <- length(probs)
  Q <- matrix(NA,nrow=n_meas,ncol=n_probs)
  meas_data <- which(is.na(DSD[,3]) == FALSE)

  for (meas in meas_data){
      if(input.matrix == TRUE){dsd <- DSD[meas,]}else{dsd <- DSD}
      sum <- sum(dsd*class_size,na.rm=FALSE)
      cumsum <- cumsum(as.numeric(dsd*class_size))
      if(sum != 0){
	  for (np in 1:n_probs){
	    p <- probs[np]
	    c1 <- cumsum <= sum*p 
	    c2 <- cumsum >= sum*p 
	    mc1 <- max(which(c1))
	    mc2 <- min(which(c2))
	    lambda1 <- 1/((sum*p) - cumsum[mc1])
	    lambda2 <- 1/(cumsum[mc2] - (sum*p))
	    Q[meas,np] <- round((lambda1*class_size[mc1] + lambda2*class_size[mc2])/(lambda1+lambda2),digits=3)
	  }
      }else{
	    Q[meas,] <- 0	# if no drops (dry period)
      }
  }

#   names(Q) <- probs
  return(Q)
}


########################## Calculation of Total concentration of drop Nt from DSD ##############################
Nt_fromDSD <- function(DSD,input.matrix=FALSE){
	## Inputs:
	##   'DSD': A matrix with Volumic DSD data. Each row corresponds to a measurements, column are the diameter classes. DSD has to be given in [m-3.mm-1]
	##   '': 
	## Optional arguments:
	##   'input.matrix': logical, is input DSD a matrix ? by default= FALSE (i.e. vector of length 32).
	## Outputs:
	##   'Nt': A vector of R values in [m-3]

	# Written by Joel Jaffrain, March 2010
	# Last update: March, 16th 2010

  ## Check dimension of DSD matrix:
  if (input.matrix == TRUE){
      if (dim(DSD)[2] != 32){stop("Given DSD matrix doesn't have 32 classes of diameter")}
      n_meas <- dim(DSD)[1]
  }else{
      if (length(DSD) != 32){stop("Given DSD vector doesn't have 32 classes of diameter")}
      n_meas <- 1
  }
  
  out_Parsivel <- Parsivel_classes()
  class_spread <- out_Parsivel$class_size_spread
#   class_spread <- Parsivel_classes$class_size_spread
  Nt <- c()

  for (meas in 1:n_meas){
    if(input.matrix == TRUE){dsd <- DSD[meas,]}else{dsd <- DSD}
    Nt[meas] <- sum(dsd*class_spread,na.rm=FALSE)
  }

return(Nt)
}

########################## Calculation of median drop diameter Dmed from DSD ##############################
D0_fromDSD <- function(DSD,input.matrix=FALSE){
	## Inputs:
	##   'DSD': A matrix with Volumic DSD data. Each row corresponds to a measurements, column are the diameter classes. DSD has to be given in [m-3.mm-1]
	## Optional arguments:
	##   'input.matrix': logical, is input DSD a matrix ? by default= FALSE (i.e. vector of length 32).
	## Outputs:
	##   'D0': A vector of median drop values in [mm]

	# Written by Joel Jaffrain, March 2010
	# Last update: March, 16th 2010

  ## Check dimension of DSD matrix:
  if (input.matrix == TRUE){
      if (dim(DSD)[2] != 32){stop("Given DSD matrix doesn't have 32 classes of diameter")}
      n_meas <- dim(DSD)[1]
  }else{
      if (length(DSD) != 32){stop("Given DSD vector doesn't have 32 classes of diameter")}
      n_meas <- 1
  }

  out_Parsivel <- Parsivel_classes()
  class_spread <- out_Parsivel$class_size_spread
  class_size <- out_Parsivel$class_size	## Center of diameter classes in [mm]
#   class_spread <- Parsivel_classes$class_size_spread
#   class_size <- Parsivel_classes$class_size	## Center of diameter classes in [mm]
  D0 <- c()

  for (meas in 1:n_meas){
    if(input.matrix == TRUE){dsd <- DSD[meas,]}else{dsd <- DSD}
    sum <- sum(dsd*class_size^3*class_spread,na.rm=FALSE)
    cumsum <- cumsum(as.numeric(dsd*class_size^3*class_spread))
    c1 <- cumsum <= sum/2 
    c2 <- cumsum >= sum/2 
    mc1 <- max(which(c1))
    mc2 <- min(which(c2))
    lambda1 <- 1/((sum/2) - cumsum[mc1])
    lambda2 <- 1/(cumsum[mc2] - (sum/2))
    D0[meas] <- round((lambda1*class_size[mc1] + lambda2*class_size[mc2])/(lambda1+lambda2),digits=3)
  }

return(D0)
}


########################## Calculation of Rain rates R from DSD ##############################
rain_fromDSD <- function(DSD,FieldV,input.matrix=FALSE){
	## Inputs:
	##   'DSD': A matrix with Volumic DSD data. Each row corresponds to a measurements, column are the diameter classes. DSD has to be given in [m-3.mm-1]
	##   'tab_V':  A matrix (or vector)
	##   '': 
	## Optional arguments:
	##   '': 
	##   'input.matrix': logical, is input DSD a matrix ? by default= FALSE (i.e. vector of length 32).
	## Outputs:
	##   'R': A vector of R values in [mm.h^-1]

	# Written by Joel Jaffrain, July 2009
	# Last update: July, 17th 2009

  ## Check dimension of DSD matrix:
  if (input.matrix == TRUE){
      if (dim(DSD)[2] != 32){stop("Given DSD matrix doesn't have 32 classes of diameter")}
      n_meas <- dim(DSD)[1]
  }else{
      if (length(DSD) != 32){stop("Given DSD vector doesn't have 32 classes of diameter")}
      n_meas <- 1
  }

  out_Parsivel <- Parsivel_classes()
  class_size <- out_Parsivel$class_size	## Center of diameter classes in [mm]
  class_spread <- out_Parsivel$class_size_spread
#   class_spread <- Parsivel_classes$class_size_spread
#   class_size <- Parsivel_classes$class_size	## Center of diameter classes in [mm]
  tab_V <- raindrop_velocity(class_size)
  R <- c()

  for (meas in 1:n_meas){
#     tab_V <- FieldV[meas,]
    if(input.matrix == TRUE){dsd <- DSD[meas,]}else{dsd <- DSD}
    Cr <- (1/1e4)*(6*pi)
    R[meas] <- Cr*sum(dsd*tab_V*class_size^3*class_spread,na.rm=FALSE)
  #   R <- Cr*apply(DSD*tab_V*class_size^3*class_spread,sum,na.rm=TRUE)
  }

return(R)
}

########################## Calculation of Radar Reflectivity Z from DSD ##############################
radar_reflectivity <- function(DSD,freq,T=20,input.matrix=FALSE){
	## Inputs:
	##   'DSD': A matrix with Volumic DSD data. Each row corresponds to a measurements, column are the diameter classes. DSD has to be given in [m-3.mm-1]
	##   'freq':  the frequency in [GHz]
	## Optional arguments:
	##   'T': Temperature in [°C], by default 20°C.
	##   'input.matrix': logical, is input DSD a matrix ? by default= FALSE (i.e. vector of length 32).
	## Outputs:
	##   'Z': A matrix of Z values in [mm^6.m^-3] for both polarization (1st col=H, 2nd col=V)

	# Written by Joel Jaffrain, July 2009
	# Last update: July, 23rd 2009
# source("/USERS/jaffrain/R/Library_functions_R.R")

  ## Check dimension of DSD matrix:
  if (input.matrix == TRUE){
      if (dim(DSD)[2] != 32){stop("Given DSD matrix doesn't have 32 classes of diameter")}
      n_meas <- dim(DSD)[1]
  }else{
      if (length(DSD) != 32){stop("Given DSD vector doesn't have 32 classes of diameter")}
      n_meas <- 1
  }


  ## Constants
  out_Parsivel <- Parsivel_classes()
  class_size <- out_Parsivel$class_size	## Center of diameter classes in [mm]
  class_spread <- out_Parsivel$class_size_spread
#   class_spread <- Parsivel_classes$class_size_spread
#   class_size <- Parsivel_classes$class_size	## Center of diameter classes in [mm]
  c <- 299792.458*1e3			# [m.s^-1]

#   axis_ratio <- raindrop_axis_ratio(class_size)
  index_water <- ref_index_water(T,freq)
  wavelength <- (c/freq*1e-9)*1e3	# [mm]
  ## Scattering Matrix
  file_Tmat <- paste("/USERS/jaffrain/R/TmatrixOutputs/","ScatteringMatrix_back_",freq,"GHz.out",sep="")	# Read outputs from Tmatrix code.
  Tmat_out <- read.table(file_Tmat,header=TRUE)
  Shh <- complex(real=Tmat_out$Re_Shh,imaginary=Tmat_out$Im_Shh)
  Svv <- complex(real=Tmat_out$Re_Svv,imaginary=Tmat_out$Im_Svv)
  ## Backscattering cross section
  BSH <- 4*pi*Mod(Shh)^2
  BSV <- 4*pi*Mod(Svv)^2
  ## Converting mm^2 to cm^2
  BSH <- BSH/100	# [cm^2]
  BSV <- BSV/100	# [cm^2]

  Kw   <- (index_water^2-1)/(index_water^2+2)
  Cz <- (1e6*(wavelength*1e-1)^4)/(pi^5*abs(Kw)^2)	# Cz in [cm^4.10^6]	10^6 is used to convert cm^6 into mm^6 !!!
  Zh <- c()
  Zv <- c()
  if (input.matrix == TRUE){
    for (meas in 1:n_meas){
      Zh[meas] <- Cz*sum(BSH*DSD[meas,]*class_spread,na.rm=FALSE)
      Zv[meas] <- Cz*sum(BSV*DSD[meas,]*class_spread,na.rm=FALSE)
    }
  }else{
      Zh <- Cz*sum(BSH*DSD*class_spread,na.rm=FALSE)
      Zv <- Cz*sum(BSV*DSD*class_spread,na.rm=FALSE)
  }

out <- cbind(as.numeric(Zh),as.numeric(Zv))
names(out) <- c("Zh","Zv")
return(out)
}

########################## Calculation of Radar Reflectivity Z from DSD calling the T-matrix explixitly ##############################
radar_reflectivity_specific <- function(DSD,freq,T,incidence){
	## Inputs:
	##   'DSD': A matrix with Volumic DSD data. Each row corresponds to a measurements, column are the diameter classes. DSD has to be given in [m-3.mm-1]
	##   'freq':  the frequency in [GHz]
	##   'T': Temperature in [°C]
	##   'incidence':incidence angle in degrees, with respect to the radar (e.g, 90 ° = vertical scan)
	
	## Outputs:
	##   'Z': A matrix of Z values in [mm^6.m^-3] for both polarization (1st col=H, 2nd col=V)

	# Written by Jacopo Grazioli May 16th 2010

        #JACOPO GRAZIOLI

	##NOTES: CANTING ANGLE NOT TAKEN INTO ACCOUNT
	##Modification:

	


  ## Check dimension of DSD matrix:
  
      if (dim(DSD)[2] != 32){stop("Given DSD matrix doesn't have 32 classes of diameter")}
      n_meas <- dim(DSD)[1]
  
     

  ## Constants
  out_Parsivel <- Parsivel_classes()
  class_size <- out_Parsivel$class_size	## Center of diameter classes in [mm]
  class_spread <- out_Parsivel$class_size_spread
  c <- 299792.458*1e3			# [m.s^-1]

  #Derivable quantities
  axis_ratios <- raindrop_axis_ratio(class_size)
  index_water <- ref_index_water(T,freq)
  wavelength <- (c/freq*1e-9)*1e3	# [cm]
  ## Backscattering cross section

  dummy <- back_scat_cross(class_size,axis_ratios,wavelength,index_water,(90-incidence),90)
  BSH <-dummy[,1]
  BSV <-dummy[,2]

  BSH[which(class_size > 8)] <- 0
  BSV[which(class_size > 8)] <- 0
  
  
  Kw   <- (index_water^2-1)/(index_water^2+2)
  Cz <- (1e6*(wavelength*1e-1)^4)/(pi^5*abs(Kw)^2)	# Cz in [cm^4.10^6]	10^6 is used to convert cm^6 into mm^6 !!!
  Zh <- c()
  Zv <- c()
  
 
      for (meas in 1:n_meas){
      Zh[meas] <- Cz*sum(BSH*DSD[meas,]*class_spread,na.rm=FALSE)
      Zv[meas] <- Cz*sum(BSV*DSD[meas,]*class_spread,na.rm=FALSE)
    }
  
  

out <- cbind(as.numeric(Zh),as.numeric(Zv))
names(out) <- c("Zh","Zv")
return(out)
}


########################## Calculation of water content W from DSD ##############################
water_content <- function(DSD,T=20,input.matrix=FALSE){
	## Inputs:
	##   'DSD': A matrix with Volumic DSD data. Each row corresponds to a measurements, column are the diameter classes. DSD has to be given in [m-3.mm-1]
	## Optional arguments:
	##   'T': Temperature in [°C], by default 20°C.
	##   'input.matrix': logical, is input DSD a matrix ? by default= FALSE (i.e. vector of length 32).
	## Outputs:
	##   'W': A vector of W values in [g.m^-3]

	# Written by Joel Jaffrain, July 2009
	# Last update: July, 20th 2009
# source("/USERS/jaffrain/R/Library_functions_R.R")

  ## Check dimension of DSD matrix:
  if (input.matrix == TRUE){
      if (dim(DSD)[2] != 32){stop("Given DSD matrix doesn't have 32 classes of diameter")}
      n_meas <- dim(DSD)[1]
  }else{
      if (length(DSD) != 32){stop("Given DSD vector doesn't have 32 classes of diameter")}
      n_meas <- 1
  }


  ## Constants
  out_Parsivel <- Parsivel_classes()
  class_size <- out_Parsivel$class_size	## Center of diameter classes in [mm]
  class_spread <- out_Parsivel$class_size_spread
#   class_spread <- Parsivel_classes$class_size_spread
#   class_size <- Parsivel_classes$class_size	## Center of diameter classes in [mm]

  if (T == 10){density_water <- 999.7}else if(T == 20){density_water <- 998.2}else{density_water <- 995.7}	# Density of water [kg.m^-3]
  Cw <- (pi/6)*density_water*1e-6	# 1e-6 to convert into g.m^-3
  W <- c()
  if (input.matrix == TRUE){
    for (meas in 1:n_meas){
      W[meas] <- Cw*sum(class_size^3*DSD[meas,]*class_spread,na.rm=FALSE)
    }
  }else{
      W <- Cw*sum(class_size^3*DSD*class_spread,na.rm=FALSE)
  }
return(W)
}

########################## Calculation of the one-way attenuation K from DSD ##############################
oneway_attenuation <- function(DSD,freq,T=20,input.matrix=FALSE){
	## Inputs:
	##   'DSD': A matrix with Volumic DSD data. Each row corresponds to a measurements, column are the diameter classes. DSD has to be given in [m-3.mm-1]
	##   'freq':  the frequency in [GHz]
	## Optional arguments:
	##   'T': Temperature in [°C], by default 20°C.
	##   'input.matrix': logical, is input DSD a matrix ? by default= FALSE (i.e. vector of length 32).
	## Outputs:
	##   'k': A vector of k values in [dB.km^-1] for both polarization (1st col=H, 2nd col=V)

	# Written by Joel Jaffrain, July 2009
	# Last update: July, 20th 2009
# source("/USERS/jaffrain/R/Library_functions_R.R")

  ## Check dimension of DSD matrix:
  if (input.matrix == TRUE){
      if (dim(DSD)[2] != 32){stop("Given DSD matrix doesn't have 32 classes of diameter")}
      n_meas <- dim(DSD)[1]
  }else{
      if (length(DSD) != 32){stop("Given DSD vector doesn't have 32 classes of diameter")}
      n_meas <- 1
  }


  ## Constants
  out_Parsivel <- Parsivel_classes()
  class_size <- out_Parsivel$class_size	## Center of diameter classes in [mm]
  class_spread <- out_Parsivel$class_size_spread
#   class_spread <- Parsivel_classes$class_size_spread
#   class_size <- Parsivel_classes$class_size	## Center of diameter classes in [mm]
  c <- 299792.458*1e3			# [m.s^-1]

  axis_ratio <- raindrop_axis_ratio(class_size)
  index_water <- ref_index_water(T,freq)
  wavelength <- (c/freq*1e-9)*1e3	# [mm]

  ## Scattering Matrix
  file_Tmat <- paste("/USERS/jaffrain/R/TmatrixOutputs/","ScatteringMatrix_ext_",freq,"GHz.out",sep="")	# Read outputs from Tmatrix code.
  Tmat_out <- read.table(file_Tmat,header=TRUE)
  Shh <- complex(real=Tmat_out$Re_Shh,imaginary=Tmat_out$Im_Shh)
  Svv <- complex(real=Tmat_out$Re_Svv,imaginary=Tmat_out$Im_Svv)

  ## Extinction cross section
  k <- 2*pi/wavelength
  ESH <- 4*pi*Im(Shh)/k
  ESV <- 4*pi*Im(Svv)/k
  ## Converting mm^2 to cm^2
  ESH <- ESH/100
  ESV <- ESV/100

  Ck <- 1/log(10)
  kh <- c()
  kv <- c()
  if (input.matrix == TRUE){
    for (meas in 1:n_meas){
      kh[meas] <- Ck*sum(ESH*DSD[meas,]*class_spread,na.rm=FALSE)
      kv[meas] <- Ck*sum(ESV*DSD[meas,]*class_spread,na.rm=FALSE)
    }
  }else{
      kh <- Ck*sum(ESH*DSD*class_spread,na.rm=FALSE)
      kv <- Ck*sum(ESV*DSD*class_spread,na.rm=FALSE)
  }
out <- cbind(kh,kv)
names(out) <- c("kh","kv")
return(out)
}

########################## Calculation of the one-way specific differential phase Kdp from DSD, calling TMATRIX  ##############################
oneway_attenuation_specific <- function(DSD,freq,T,incidence){
	## Inputs:
	##   'DSD': A matrix with Volumic DSD data. Each row corresponds to a measurements, column are the diameter classes. DSD has to be given in [m-3.mm-1]
	##   'freq':  the frequency in [GHz]
	## Optional arguments:
	##   'T': Temperature in [°C]
	##   incidence=incidence angle of the radar wave.
	## Outputs:
	##   'k': A vector of k values in [dB.km^-1] for both polarization (1st col=H, 2nd col=V)

	# Written by Joel Jaffrain, July 2009, modified JAcopo GRazioli, Jan 2013
	# Last update: Jan, 12 2013

	##NOTES: CANTING ANGLE NOT TAKEN INTO ACCOUNT

  ## Check dimension of DSD matrix:
  
      if (dim(DSD)[2] != 32){stop("Given DSD matrix doesn't have 32 classes of diameter")}
      n_meas <- dim(DSD)[1]


  ## Constants
  out_Parsivel <- Parsivel_classes()
  class_size <- out_Parsivel$class_size	## Center of diameter classes in [mm]
  class_spread <- out_Parsivel$class_size_spread
  c <- 299792.458*1e3			# [m.s^-1]

  axis_ratio <- raindrop_axis_ratio(class_size)
  index_water <- ref_index_water(T,freq)
  wavelength <- (c/freq*1e-9)*1e3	# [mm]

  ## Scattering Matrix
  
  tab_forward_ampl <- fwrd_scat_ampl_comp(class_size,axis_ratio,wavelength,index_water,(90-incidence),0)

  Shh <- tab_forward_ampl[,1]
  Svv <- tab_forward_ampl[,2]

  ## Extinction cross section
  k <- 2*pi/wavelength
  ESH <- 4*pi*Im(Shh)/k
  ESV <- 4*pi*Im(Svv)/k
  ## Converting mm^2 to cm^2
  ESH <- ESH/100
  ESV <- ESV/100

  Ck <- 1/log(10)
  kh <- c()
  kv <- c()
 
    for (meas in 1:n_meas){
      kh[meas] <- Ck*sum(ESH*DSD[meas,]*class_spread,na.rm=FALSE)
      kv[meas] <- Ck*sum(ESV*DSD[meas,]*class_spread,na.rm=FALSE)
  	}
out <- cbind(kh,kv)
names(out) <- c("kh","kv")
return(out)
}

########################## Calculation of the one-way specific differential phase Kdp from DSD ##############################
specific_dif_phase <- function(DSD,freq,T=20,input.matrix=FALSE){
	## Inputs:
	##   'DSD': A matrix with Volumic DSD data. Each row corresponds to a measurements, column are the diameter classes. DSD has to be given in [m-3.mm-1]
	##   'freq':  the frequency in [GHz]
	## Optional arguments:
	##   'T': Temperature in [°C], by default 20°C.
	##   'input.matrix': logical, is input DSD a matrix ? by default= FALSE (i.e. vector of length 32).
	##   'Theta'   = zenith  of the incident wave (between 0 and 180°)(90° = horizontal)
	##   'Phi'    = azimuth of the incident wave (between 0 and 360°)(90° = along the y-axis)
	## Outputs:
	##   'kdp': A vector of specific differential phase (Kdp) values in [°.km^-1].

	# Written by Joel Jaffrain, July 2009
	# Last update: July, 28th 2009
#   source("/USERS/jaffrain/R/Library_functions_R.R")

  ## Check dimension of DSD matrix:
  if (input.matrix == TRUE){
      if (dim(DSD)[2] != 32){stop("Given DSD matrix doesn't have 32 classes of diameter")}
      n_meas <- dim(DSD)[1]
  }else{
      if (length(DSD) != 32){stop("Given DSD vector doesn't have 32 classes of diameter")}
      n_meas <- 1
  }

  ## Constants
  out_Parsivel <- Parsivel_classes()
  class_size <- out_Parsivel$class_size	## Center of diameter classes in [mm]
  class_spread <- out_Parsivel$class_size_spread
#   class_spread <- Parsivel_classes$class_size_spread
#   class_size <- Parsivel_classes$class_size	## Center of diameter classes in [mm]
  c <- 299792.458*1e3			# [m.s^-1]
  wavelength <- (c/freq*1e-9)*1e3	# [mm]

  ## Scattering Matrix
  file_Tmat <- paste("/USERS/jaffrain/R/TmatrixOutputs/","ScatteringMatrix_ext_",freq,"GHz.out",sep="")	# Read outputs from Tmatrix code.
  Tmat_out <- read.table(file_Tmat,header=TRUE)
  Shh <- complex(real=Tmat_out$Re_Shh,imaginary=Tmat_out$Im_Shh)		# S(h,h) [mm]
  Svv <- complex(real=Tmat_out$Re_Svv,imaginary=Tmat_out$Im_Svv)		# S(v,v) [mm]

  Ck <- (180/pi)*wavelength*1e-3	# [°.m]
  kdp <- c()
  if (input.matrix == TRUE){
    for (meas in 1:n_meas){
      kdp[meas] <- Ck*sum(Re(Shh-Svv)*DSD[meas,]*class_spread,na.rm=FALSE)
    }
  }else{
      kdp <- Ck*sum(Re(Shh-Svv)*DSD*class_spread,na.rm=FALSE)
  }
return(kdp)
}

##################################################################################################################################
#CALCULATE SPECIFIC DIFFERENTIAL PHASE USING A T-MATRIX CALL
specific_dif_phase_specific <- function(DSD,freq,T,incidence){
	## Inputs:
	##   'DSD': A matrix with Volumic DSD data. Each row corresponds to a measurements, column are the diameter classes. DSD has to be given in [m-3.mm-1]
	##   'freq':  the frequency in [GHz]
	##   'T': Temperature in [°C]
	##   incidence=incidence angle of the radar wave.
	## Outputs:
	##   'kdp': A vector of specific differential phase (Kdp) values in [°.km^-1].

	# Written by Joel Jaffrain, July 2009 MODIFIED Jacopo Grazioli Jan 2013


	##NOTES: CANTING ANGLE NOT TAKEN INTO ACCOUNT


  ## Check dimension of DSD matrix:
  
      if (dim(DSD)[2] != 32){stop("Given DSD matrix doesn't have 32 classes of diameter")}
      n_meas <- dim(DSD)[1]

  ## Constants
  out_Parsivel <- Parsivel_classes()
  class_size <- out_Parsivel$class_size	## Center of diameter classes in [mm]
  class_spread <- out_Parsivel$class_size_spread

  axis_ratio <- raindrop_axis_ratio(class_size)
  index_water <- ref_index_water(T,freq)
  c <- 299792.458*1e3			# [m.s^-1]
  wavelength <- (c/freq*1e-9)*1e3	# [mm]

  ## Scattering Matrix
  
  tab_forward_ampl <- fwrd_scat_ampl_comp(class_size,axis_ratio,wavelength,index_water,(90-incidence),90)

  Shh <- tab_forward_ampl[,1]
  Svv <- tab_forward_ampl[,2]

  Ck <- (180/pi)*wavelength*1e-3	# [°.m]
  kdp <- c()
  
    for (meas in 1:n_meas){
      kdp[meas] <- Ck*sum(Re(Shh-Svv)*DSD[meas,]*class_spread,na.rm=FALSE)
    }
 
return(kdp)
}


########################## Automatic construction of classes ##############################
delimit_classes <- function(lim_min,lim_max,percent=0.1){
	## Inputs:
	##   'data': A vector with the corresponding data for which classes have to be defined.
	##   'lim_min': The lower limit of the first class.  
	##   'lim_max': The upper limit of the last class.  
	## Optional arguments:
	##   'percent':  Center - percent < Center of class < Center + percent.
	## Outputs:
	##   'classes': A vector of the lower limits of classes.

	# Written by Joel Jaffrain, September 2009
	# Last update: September, 15th 2009


  low <- lim_min
  classes <- low
  up <- 0
  while(up < lim_max){
    c <- low/(1-percent)			# center of the considered class
    up <- round(c*(1+percent),digits=2)		# upper limit of the considered class
    classes <- c(classes,up)
    low <- up
  }
return(classes)
}

########################## Moving windows for Z-R fit ##############################
moving_ZR <- function(R,Z,tabT,width,tol.na=0.3){
	## Inputs:
	##   'R': vector of rain rate values [mm/h].
	##   'Z': vector of reflectivity values [mm6/m3].
	##   'tabT': time series (POSIXct).
	##   'width': width of the moving window [min].
	## Optional arguments:
	##   'tol.na': tolerance for NA values (%) in the moving window
	## Outputs:
	##   'a': prefactor ('a') of the Z-R power law according to each moving window
	##   'b': exponent ('b') of the Z-R power law according to each moving window
	##   'new_time': associated time serie.

	# Written by Joel Jaffrain, February 2010
	# Last update: February, 5th 2010

    nR   <- length(R)
    nZ   <- length(Z)
    nT   <- length(tabT)
    dt	 <- round(as.numeric(difftime(tabT[nT],tabT[1],units="sec")/nT),digits=0)	# temporal resolution in 'tabT'
    nrow <- (width*60)/dt
    seq_a <- seq(200,800,10)
    seq_b <- seq(0.4,2.6,0.01)
    a    <- NA
    b    <- NA

    if (is.null(dim(R))==FALSE || is.null(dim(Z))==FALSE  || is.null(dim(tabT))==FALSE){stop("R, Z and Time must be vectors")}
    if (nR != nZ || nR != nT || nZ != nT){stop("R, Z and Time must have the same length")}

    for(i in (nrow/2+1):(nT-nrow/2)){
	start <- i - nrow/2
	end   <- i + nrow/2
	if( (sum(is.na(R[start:end])) > tol.na*nrow) || (sum(is.na(Z[start:end])) > tol.na*nrow) ){
	    a[start] <- NA
	    b[start] <- NA
	}else{
	    model <- fit(R[start:end],Z[start:end],seq_a,seq_b,cutoff=0.01)		## Powerlaw model
	    a[start] <- model[[1]][1]
	    b[start] <- model[[1]][2]
	    plot(R[start:end],Z[start:end],pch="+",main=paste(i,";",tabT[start],"to",tabT[end]))
	    points(R[start:end],a[start]*R[start:end]^b[start],col="red")
	    legend("bottomright",legend=c(paste("a=",a[start]),paste("b=",b[start])),text.col="red")
	}
    }
    new_time <- tabT[(nrow/2+1):(nT-nrow/2)]
    out <- list(a,b,new_time)
    names(out) <- c("a","b","time")
    return(out)
}

########################## Read Sensorscope data ##############################
read_sensorscope <- function(filename) {
	## Inputs: filename = path of SensorScope data file [string]
	##
	## Output: data.frame$time = measurement times [POSIXct]
	##         data.frame$rain = rain accumulation [mm]
	##         data.frame$speed = wind speed [m/s]
	##         data.frame$direction = wind direction [°]
	##
	## description: reads SensorScope data file and extracts some meteorological
	##              parameters, with time step usually 128s
	##
	# creation: Florian Pantillon, December 2008


    data <- read.table(filename)

    tmp <- data[,8] # time [s]
    time <- as.POSIXct(tmp,tz="GMT",origin="1970-01-01") # to timestamp

    rain <- data[,15] # rain accumulation [mm]
    speed <- data[,16] # wind speed [m/s]
    direction <- data[,17] # wind direction [°]

    n <- length(time)

    for (i in 1:(n-1)) {
      d_rain <- rain[i+1]-rain[i]
      if (!is.nan(d_rain) && d_rain<0) {
	rain[(i+1):n] <- rain[(i+1):n]-d_rain # remove gaps due to resetting
      }
    }

    read_sensorscope <- data.frame(time,rain,speed,direction)

}

########################## Polar to XY ##############################
polar_to_xy <- function(speed,direction) {
	## Inputs: speed (vector) = wind speed [m/s]
	##            direction (vector) = wind direction [°]
	##
	## Output: data.frame$x = wind speed in x direction [m/s]
	##         data.frame$y = wind speed in y direction [m/s]
	##
	## description: converts wind speed and direction to cartesian coordinates
	##
	# creation: Florian Pantillon, December 2008

    dir_rad <- (90-direction)*pi/180

    x <- speed * cos(dir_rad)
    y <- speed * sin(dir_rad)

    polar_to_xy <- data.frame(x,y)
}

########################## XY to Polar ##############################
xy_to_polar <- function(x,y) {
	## Inputs: x (vector) = wind speed in x direction [m/s]
	##            y (vector) = wind speed in y direction [m/s]
	##
	## Output: data.frame$speed = wind speed [m/s]
	##         data.frame$direction = wind direction [°]
	##
	## description: converts wind speed in cartesian coordinates
	##              to speed and direction
	##
	# creation: Florian Pantillon, December 2008


    speed <- sqrt(x^2+y^2)

    if (x>=0) {
      dir_rad <- atan(y/x)
    } else {
      dir_rad <- atan(y/x)+pi
    }
    direction <- (90-dir_rad*180/pi)%%360

    xy_to_polar <- data.frame(speed,direction)
}


######################### Uncertainty ################################
uncertainty_Parsiv <- function(vect,var_type,dt,freq=9.4){
	## Inputs:
	##    'vect': the vector of value to calculate the uncertainty (ex: rain rates, reflectivities,...)
	##    'var_type': a character vector with the name of the variable of interest (choice between: 'Nt', 'D0', 'R', 'Z' or 'Zdr')
	##    'dt': temporal resolution [in sec]
	## Optional arguments:
	##    'freq': numeric value with the radar frequency to consider (9.4, 5.6 or 2.8).
	## Outputs:
	##    'uncer': a matrix with 2 columns: 1st= uncertainty values (sd), 2nd=sd*vect


	# Written by Joel Jaffrain, May 2010
	# Last update: May, 19th 2010

  if(is.null(dim(vect)) == FALSE){stop("The input should be a vector !")}
  if(sum(round(freq,digits=1) == c(9.4,5.6,2.8))==0){stop("Provide a useful frequency band: 9.4 GHz(X), 5.6 GHz(C) or 2.8 GHz(S)")}
  
  list_tres <- c(20,60,120,180,240,300,600,900,1800,2700,3600)
  n_tres <- length(list_tres)
  pos_dt <- which(dt == list_tres)
  uncer <- matrix(NA,ncol=2,nrow=length(vect))
  n_meas <- length(vect)

  if(var_type == "Nt"){
    class_inf <- c(0,50,100,200,400,600,800,1000,1500,2000)
    mat <- matrix(NA,nrow=9,ncol=n_tres)
    mat[1,] <- c(0.24,0.15,0.13,0.13,0.12,0.09,0.08,0.07,0.06,0.06,0.08)
    mat[2,] <- c(0.20,0.13,0.10,0.09,0.08,0.07,0.06,0.07,0.08,0.08,0.05)
    mat[3,] <- c(0.16,0.11,0.09,0.07,0.07,0.07,0.06,0.06,0.04,0.06,0.04)
    mat[4,] <- c(0.13,0.09,0.07,0.07,0.06,0.06,0.05,0.05,0.05,0.08,0.04)
    mat[5,] <- c(0.10,0.07,0.06,0.05,0.05,0.05,0.04,0.04,0.03,NA,NA)
    mat[6,] <- c(0.09,0.07,0.06,0.05,0.05,0.05,0.03,NA,NA,NA,NA)
    mat[7,] <- c(0.09,0.06,0.05,0.05,0.05,NA,NA,NA,NA,NA,NA)
    mat[8,] <- c(0.08,0.06,0.06,0.06,NA,NA,NA,NA,NA,NA,NA)
    mat[9,] <- c(0.07,0.05,NA,NA,NA,NA,NA,NA,NA,NA,NA)
  }else if(var_type == "D0"){
    class_inf <- c(0.6,0.7,0.8,0.9,1,1.25,1.5,1.75,2,2.25,2.5)
    mat <- matrix(NA,nrow=10,ncol=n_tres)
    mat[1,] <- c(0.07,0.05,0.03,0.03,0.03,0.03,0.03,0.02,0.02,0.02,NA)
    mat[2,] <- c(0.09,0.06,0.04,0.06,0.06,0.03,0.03,0.02,0.02,0.02,0.02)
    mat[3,] <- c(0.10,0.07,0.05,0.04,0.04,0.03,0.03,0.03,0.03,0.02,0.02)
    mat[4,] <- c(0.12,0.06,0.05,0.05,0.04,0.04,0.04,0.03,0.03,0.03,0.03)
    mat[5,] <- c(0.13,0.08,0.06,0.05,0.05,0.05,0.04,0.04,0.03,0.04,0.03)
    mat[6,] <- c(0.15,0.10,0.07,0.08,0.09,0.06,0.04,0.05,0.04,0.04,0.04)
    mat[7,] <- c(0.19,0.11,0.09,0.08,0.07,0.07,0.07,0.07,0.07,0.05,NA)
    mat[8,] <- c(0.23,0.16,0.11,0.11,0.10,0.09,0.12,NA,NA,NA,NA)
    mat[9,] <- c(0.29,0.22,0.16,0.12,0.13,NA,NA,NA,NA,NA,NA)
    mat[10,] <- c(0.38,0.24,0.20,NA,NA,NA,NA,NA,NA,NA,NA)
  }else if(var_type == "R"){
    class_inf <- c(0.1,2,4,6,8,10,15,20,25)
    mat <- matrix(NA,nrow=8,ncol=n_tres)
    mat[1,] <- c(0.25,0.18,0.15,0.14,0.13,0.12,0.11,0.11,0.11,0.11,0.10)
    mat[2,] <- c(0.22,0.16,0.13,0.12,0.12,0.11,0.10,0.10,0.09,0.09,0.09)
    mat[3,] <- c(0.21,0.15,0.13,0.11,0.10,0.10,0.09,0.08,0.08,NA,NA)
    mat[4,] <- c(0.19,0.14,0.12,0.10,0.10,0.10,0.08,NA,NA,NA,NA)
    mat[5,] <- c(0.17,0.13,0.12,0.13,0.09,0.10,NA,NA,NA,NA,NA)
    mat[6,] <- c(0.17,0.17,0.15,0.12,0.10,NA,NA,NA,NA,NA,NA)
    mat[7,] <- c(0.18,0.16,NA,NA,NA,NA,NA,NA,NA,NA,NA)
    mat[8,] <- c(0.14,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA)
#     mat[9,] <- c(NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA)
  }else if(var_type == "Z"){
    class_inf <- c(seq(10,50,5))
    mat <- matrix(NA,nrow=8,ncol=n_tres)
    if(freq == 9.4){
      mat[1,] <- c(0.14,0.10,0.09,0.08,0.08,0.08,0.07,0.06,0.06,0.07,0.06)
      mat[2,] <- c(0.13,0.10,0.08,0.07,0.07,0.07,0.06,0.06,0.06,0.06,0.06)
      mat[3,] <- c(0.11,0.08,0.07,0.06,0.06,0.06,0.06,0.05,0.06,0.07,0.06)
      mat[4,] <- c(0.10,0.07,0.06,0.06,0.06,0.06,0.07,0.07,0.07,0.08,0.06)
      mat[5,] <- c(0.10,0.08,0.07,0.08,0.08,0.08,0.07,0.07,0.07,0.06,0.07)
      mat[6,] <- c(0.14,0.11,0.10,0.09,0.07,0.07,0.07,0.06,0.07,0.07,0.11)
      mat[7,] <- c(0.17,0.12,0.10,0.09,0.11,0.09,0.07,0.08,NA,NA,NA)
      mat[8,] <- c(0.16,0.15,0.16,0.13,NA,NA,NA,NA,NA,NA,NA)
    }else if(freq == 5.6){
      mat[1,] <- c(0.14,0.10,0.09,0.08,0.08,0.08,0.07,0.06,0.06,0.07,0.06)
      mat[2,] <- c(0.13,0.10,0.08,0.08,0.07,0.07,0.06,0.06,0.06,0.06,0.06)
      mat[3,] <- c(0.11,0.08,0.07,0.06,0.06,0.06,0.06,0.05,0.05,0.06,0.05)
      mat[4,] <- c(0.11,0.07,0.06,0.05,0.05,0.05,0.06,0.05,0.05,0.05,0.04)
      mat[5,] <- c(0.10,0.06,0.05,0.05,0.05,0.06,0.04,0.04,0.04,0.04,0.07)
      mat[6,] <- c(0.09,0.06,0.05,0.05,0.06,0.05,0.05,0.04,0.09,0.09,NA)
      mat[7,] <- c(0.10,0.10,0.09,0.10,0.08,0.11,NA,NA,NA,NA,NA)
      mat[8,] <- c(0.15,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA)
    }else if(freq == 2.8){
      mat[1,] <- c(0.14,0.11,0.09,0.08,0.08,0.08,0.07,0.06,0.06,0.07,0.06)
      mat[2,] <- c(0.13,0.10,0.08,0.08,0.07,0.07,0.06,0.06,0.06,0.06,0.06)
      mat[3,] <- c(0.11,0.08,0.07,0.07,0.06,0.06,0.06,0.05,0.05,0.07,0.05)
      mat[4,] <- c(0.11,0.08,0.06,0.06,0.05,0.05,0.06,0.06,0.05,0.05,0.05)
      mat[5,] <- c(0.11,0.07,0.06,0.06,0.05,0.06,0.04,0.04,0.04,0.05,0.07)
      mat[6,] <- c(0.10,0.08,0.06,0.05,0.05,0.05,0.05,0.05,0.06,0.04,NA)
      mat[7,] <- c(0.11,0.09,0.10,0.08,0.08,0.07,NA,NA,NA,NA,NA)
      mat[8,] <- c(0.15,0.16,NA,NA,NA,NA,NA,NA,NA,NA,NA)
    }else{}
  }else if(var_type == "Zdr"){
    class_inf <- c(seq(0.1,0.5,0.1),0.75,1,1.5,2,3)
    mat <- matrix(NA,nrow=9,ncol=n_tres)
    if(freq == 9.4){
      mat[1,] <- c(0.17,0.13,0.12,0.11,0.11,0.11,0.10,0.09,0.08,0.08,0.09)
      mat[2,] <- c(0.22,0.19,0.17,0.16,0.15,0.15,0.13,0.12,0.12,0.11,0.10)
      mat[3,] <- c(0.27,0.23,0.20,0.20,0.19,0.18,0.17,0.15,0.14,0.14,0.14)
      mat[4,] <- c(0.30,0.25,0.23,0.22,0.21,0.21,0.18,0.18,0.16,0.18,0.14)
      mat[5,] <- c(0.35,0.30,0.27,0.25,0.24,0.24,0.22,0.21,0.21,0.19,0.20)
      mat[6,] <- c(0.40,0.35,0.33,0.31,0.30,0.29,0.29,0.29,0.29,0.30,0.29)
      mat[7,] <- c(0.46,0.41,0.38,0.37,0.37,0.38,0.34,0.34,0.32,0.34,0.30)
      mat[8,] <- c(0.51,0.45,0.43,0.41,0.37,0.36,0.34,0.35,0.33,0.31,0.30)
      mat[9,] <- c(0.49,0.43,0.39,0.38,0.35,0.33,0.34,0.30,0.30,0.34,0.28)
    }else if(freq == 5.6){
      mat[1,] <- c(0.17,0.13,0.12,0.11,0.11,0.11,0.10,0.10,0.08,0.08,0.09)
      mat[2,] <- c(0.22,0.19,0.17,0.16,0.15,0.15,0.13,0.12,0.12,0.11,0.10)
      mat[3,] <- c(0.27,0.23,0.20,0.20,0.19,0.18,0.17,0.15,0.14,0.14,0.14)
      mat[4,] <- c(0.30,0.25,0.23,0.22,0.21,0.21,0.17,0.17,0.16,0.16,0.13)
      mat[5,] <- c(0.34,0.29,0.26,0.23,0.22,0.21,0.19,0.18,0.16,0.17,0.17)
      mat[6,] <- c(0.38,0.31,0.27,0.25,0.24,0.25,0.22,0.22,0.20,0.22,0.17)
      mat[7,] <- c(0.42,0.36,0.32,0.30,0.28,0.28,0.26,0.27,0.27,0.28,0.28)
      mat[8,] <- c(0.48,0.41,0.41,0.41,0.38,0.37,0.38,NA,NA,NA,NA)
      mat[9,] <- c(0.52,0.50,0.45,0.45,0.46,0.43,NA,NA,NA,NA,NA)
    }else if(freq == 2.8){
      mat[1,] <- c(0.17,0.13,0.12,0.11,0.11,0.11,0.10,0.10,0.08,0.09,0.09)
      mat[2,] <- c(0.22,0.19,0.17,0.16,0.15,0.15,0.13,0.13,0.11,0.11,0.10)
      mat[3,] <- c(0.27,0.23,0.20,0.20,0.19,0.18,0.17,0.15,0.14,0.13,0.14)
      mat[4,] <- c(0.30,0.25,0.23,0.22,0.21,0.21,0.17,0.18,0.17,0.17,0.14)
      mat[5,] <- c(0.34,0.29,0.26,0.23,0.22,0.22,0.19,0.18,0.17,0.17,0.17)
      mat[6,] <- c(0.38,0.32,0.27,0.25,0.24,0.25,0.23,0.23,0.22,0.24,0.18)
      mat[7,] <- c(0.42,0.36,0.33,0.32,0.29,0.29,0.27,0.28,0.27,0.29,0.28)
      mat[8,] <- c(0.48,0.41,0.40,0.39,0.39,0.38,0.37,0.36,0.34,NA,NA)
      mat[9,] <- c(0.50,0.46,0.42,0.40,0.38,0.36,0.33,0.29,NA,NA,NA)
    }else{}
  }else{
    stop("No uncertainty values for the specified variable !")
  }

  if(sum(dt == list_tres) != 0){
    vect_mat <- mat[,pos_dt]
  }else{	## interpolate uncertainty values using least squared if specific time res.
    pos_1 <- max(which(dt > list_tres)) 
    pos_2 <- min(which(dt < list_tres))
    lambda1 <- 1/(dt - list_tres[pos_1])
    lambda2 <- 1/(list_tres[pos_2] - dt)
    int <- ((lambda1*mat[,pos_1]) + (lambda2*mat[,pos_2]))/(lambda1 + lambda2)
    vect_mat <- round(int,digits=2)
  }
  class_limits <- matrix(data=sort(c(class_inf,class_inf[2:(length(class_inf)-1)])),nrow=length(class_inf)-1,ncol=2,byrow=T)
  class_spread <- (class_limits[,2]-class_limits[,1])
  class_avg <- c(((class_spread/2) + class_limits[,1]),NA)
  number_class <- length(class_avg)
#   legend_classes <- c(apply(round(class_limits,digits=3),1,paste,collapse=","),paste(">",class_limits[number_class-1,2],sep=""))
  logical_class <- matrix(NA,nrow=n_meas,ncol=number_class)

  for (class_lim in 1:number_class){
    if (class_lim != number_class){
      cond_inf <- vect >= class_limits[class_lim,1]
      cond_sup <- vect <  class_limits[class_lim,2]
      logical_class[,class_lim] <- as.logical(cond_inf * cond_sup)
    }else{logical_class[,class_lim] <- as.logical(vect >= class_limits[(class_lim-1),2])}

    temp <- which(logical_class[,class_lim] == TRUE)
    uncer[temp,1] <- vect_mat[class_lim]
  }
  n_meas_classes <- colSums (logical_class, na.rm = TRUE)		# number of measurements in each class of rain rate.

  uncer[,2] <- uncer[,1] * vect

  return(uncer)
}

######################### Atlas filtering (v-D relation) ################################
Atlas_filt <- function(rowDSD,tolpct=0.4,measurements=seq(1,dim(rowDSD)[1]),plot=TRUE,name_plot="~/AtlasFilter.eps"){
	## Function to filter unrealistic drops based on the velocity-diameter relation of Beard model. Such filter (+ or - 40%) was introduced by 
	## Kruger and Krajewski, JAOT 2002.
	## 
	## Inputs:
	##    'rowDSD': a character vector of 1024 values (32x32) as provided by Parsivel disdro.
	## Optional arguments:
	##    'tolpct': percentage of tolerance [0,1] accepted compared to the Atlas (1973) curve. (default=0.4, i.e. 40%)
	##    'measurements': measurements to consider (i.e., for instance the output of a previous filtering function). (default=all)
	##    'plot': If plot v(D)=f(D) should be drawn showing the drops kept after filtering process. (default=TRUE)
	##    'name_plot': Name (including path) for the plot. Only useful if 'plot=TRUE'. (default="~/AtlasFilter.eps")
	## Outputs:
	##    'FN_Atlas': Volumic DSD after Atlas correction.

	# Written by Joel Jaffrain, August 2010
	# Last update: August, 04th 2010

#     dim_meas <- length(measurements)
#     dim_com <- length(rowDSD)
    dim_com <- length(measurements)
    DSDraw <- as.matrix(extract_rawDSD(rowDSD,measurements=measurements))
#     test <- colSums(DSDraw,na.rm=T)
#     mat_test <- matrix(test,nrow=32,ncol=32,byrow=T)
#     nr_DSDraw <- dim(DSDraw)[1]
    nr_DSDraw <- dim_com
    nc_DSDraw <- dim(DSDraw)[2]
    vect_DSDraw <- as.vector(t(DSDraw))		# build a 1D vector with raw data.
    tmp_D <- rep(rep(class_size,32),nr_DSDraw)
    tmp_V <- rep(sort(rep(class_speed,32)),nr_DSDraw)
    detec <- which(vect_DSDraw != 0)
    D <- tmp_D[detec]
    V <- tmp_V[detec]
    vel_Beard <- raindrop_velocity(tmp_D)		# Theoretical velocity according to Beard model.
    drop_vel <- raindrop_velocity(class_size)
    Np_before <- sum(vect_DSDraw,na.rm=T)		# Total number of particles before filtering process

    # Filtering on fall velocity according to Atlas, 1973 (see paper of Kruger, 2002).
    cond1 <- V >= (1-tolpct)*vel_Beard[detec]
    cond2 <- V <= (1+tolpct)*vel_Beard[detec]
    logic_Atlas 	<- cond1*cond2
    ind_Atlas40 	<- which(logic_Atlas == 1)	# drops in the tolerated range
    rm_Atlas40 		<- which(logic_Atlas == 0)	# drops to remove
    x_Atlas <- D[ind_Atlas40]
    y_Atlas <- V[ind_Atlas40]

    ## Plot raw DSD spectrum: v(D)=f(D): Influence of Atlas filter.
    if(plot == TRUE){
	postscript(name_plot,height=(12/2.54), width=(12/2.54), pointsize=10,horizontal = FALSE, onefile = FALSE, paper = "special")
	plot(D,V,pch="+",cex=0.75,xaxt="n",yaxt="n",xlab="Equiv. diameter [mm]",ylab="Terminal fall speed [m/s]")
	points(x_Atlas,y_Atlas,col="red")	# Points kept after Atlas filtering.
	points(class_size,drop_vel,pch=20,cex=0.9,col="blue")
	axis(1,at=class_size)
	axis(2,at=class_speed)
	graphics.off()
    }

    ## Remove drops out of the tolerated range of variability.
    vect_DSDraw[detec[rm_Atlas40]] <- 0		# Remove drops out of the filter range
    Np_after <- sum(vect_DSDraw,na.rm=T)	# Total number of particles AFTER filtering process
    rm_drops <- Np_before - Np_after
    print(paste("'Atlas_filt' fct: About ",rm_drops," drops have been removed, i.e., ",round((rm_drops*100)/Np_before,digits=1)," % of the initial # of drops.",sep=""))

    DSDraw_Atlas <- matrix(vect_DSDraw,ncol=1024,nrow=nr_DSDraw,byrow=T)

    ## Go back to volumic DSD.
    S <- 0.0054	# sampling area in [m²]
    DSD_Atlas <- matrix(0,ncol=32,nrow=nr_DSDraw,byrow=T)
    FN_Atlas <- matrix(0,ncol=32,nrow=nr_DSDraw,byrow=T)
    for(meas in 1:dim_com){
	DSD_tmp <- DSDraw_Atlas[meas,]
	DSD_mat   <-  matrix(data=DSD_tmp,nrow=32,ncol=32,byrow=TRUE)	# Write DSD data from vector 'DSD_tmp' into a 32x32 matrix called 'DSD_mat'
	DSD_Atlas[meas,] <- colSums(DSD_mat)
	tmp_Ns <- colSums(DSD_mat)/(S*d_t*class_size_spread)		# Surfacic DSD
	FN_Atlas[meas,] <- tmp_Ns/drop_vel				# Volumic DSD
    }
  return(FN_Atlas)
}

######################### ?????????? ################################
