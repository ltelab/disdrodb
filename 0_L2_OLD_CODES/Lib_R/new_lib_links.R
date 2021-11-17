## This library contains all useful functions for the manipulation 
## of microwave-links and DSD estimation using dual-polarization
## Shared Library for EPFL-LTE, coded by M.Schleiss, 2008-2009

############################# get.link_name #############################

get.link_name <- function(link){

    ## Returns the link name as provided by Bouygues Telecom
    ## Returns "NA" if the link doesn't exist.
    
    name <- "NA"
    if(link==1){name  <- "FHS10394"}
    if(link==2){name  <- "FHS10329"}
    if(link==3){name  <- "FHS10202"}
    if(link==13){name <- "FHS11010"}
    if(link==14){name <- "FHS11160"}
    return(name)
}

############################## get.channels ##############################

get.channels <- function(link){
    
    ## Returns the number of channels that are available for a given link
    ## Returns NA if the link doesn't exist
    
    Nchannels <- NA
    if(link==1){Nchannels <- 2}
    if(link==2){Nchannels <- 2}
    if(link==3){Nchannels <- 2}
    if(link==13){Nchannels <- 3}
    if(link==14){Nchannels <- 3}
    return(Nchannels)
    
}

############################## get.link_pol ##############################

get.link_pol <- function(link,channel){

    ## Returns the polarization of a given link
    ## Returns NA if the link doesn't exist.
    
    pol <- "NA"
    if(link==1){pol <- "V"}
    if(link==2){pol <- "H"}
    if(link==3){pol <- "V"}
    if(link==13){
	    if(channel==1){pol <- "V"}
	    if(channel==2){pol <- "V"}
	    if(channel==3){pol <- "H"}
    }
    if(link==14){
	    if(channel==1){pol <- "V"}
	    if(channel==2){pol <- "V"}
	    if(channel==3){pol <- "H"}
    }
    return(pol)
}

##########################################################################

get.link_freq <- function(link,channel){
	
    ## Returns the frequencies for LxR and RxL (in MHz) of a given link
    ## Returns (NA,NA) if the link doesn't exist
    
    tabf <- rep(NA,2)
    if(link==1 && channel==1){tabf <- c(25067,26075)}
    if(link==1 && channel==2){tabf <- c(25123,26131)}
    if(link==2 && channel==1){tabf <- c(25095,26103)}
    if(link==2 && channel==2){tabf <- c(25151,26159)}
    if(link==3 && channel==1){tabf <- c(26075,25067)}
    if(link==3 && channel==2){tabf <- c(26131,25123)}
    if(link==13 && channel==1){tabf <- c(18099,19109)}
    if(link==13 && channel==2){tabf <- c(18044,19054)}
    if(link==13 && channel==3){tabf <- c(18071,19081)}
    if(link==14 && channel==1){tabf <- c(25207,26215)}
    if(link==14 && channel==2){tabf <- c(25151,26159)}
    if(link==14 && channel==3){tabf <- c(25179,26187)}
    return(tabf)
}

##########################################################################

get.link_length <- function(link){
  
    ## Returns the length (in meters) of a given link
    ## Returns NA if the link doesn't exist.
    
    length <- NA
    if(link==1){length <- 3700}
    if(link==2){length <- 3700}
    if(link==3){length <- 9045}
    if(link==13){length <- 7108}
    if(link==14){length <- 2440}
    return(length)
}

##########################################################################

get.link_coordinates <- function(link){

    ## Returns the coordinates of the master/slave antenna
    ## All parameters are given in Lambert2 etendue 
    ## Return order is c(master_x,master_y,slave_x,slave_y)
    ## Returns NA if the link doesn't exist.
        
    tab_coord <- c(NA,NA,NA,NA)
    if(link==1){tab_coord <- c(589.49,2433.85,593.0,2432.686)}
    if(link==2){tab_coord <- c(593.0,2432.686,589.49,2433.85)}
    if(link==3){tab_coord <- c(609.92,2414.15,601.812,2418.167)}
    if(link==13){tab_coord <- c(578.524,2441.649,584.15,2446.0)}
    if(link==14){tab_coord <- c(584.262,2446.0,586.262,2447.225)}
    return(tab_coord)
}

##########################################################################

get.rainsill <- function(idl,idc,way,tw,qdetect=NA){

    ## Returns the dry/rainy weather threshold for baseline fitting
    ## Threshold values depend on the link, the channel and the time window
    ## I already computed all rainsills for all links at 5,10,15,20,...,60min
    ## See "Rain_thresholds.R" for more details
    
    ## Inputs:
    ## idl = link id (1,2,3,13,14)
    ## idc = channel id (1,2,3)
    ## way = direction of signal ("LxR" or "RxL")
    ## tw  = time-window [s]
    ## qdetect = quantile for rain detection. If quantile==NA, the mean is taken
    
    ## Outputs:
    ## sill = rainsill for the given link, channel, way and time-window
    
    ## Basic tests
    if(is.na(idl*idc*tw)){stop("invalid input")}
    if(way!="LxR" && way!="RxL"){stop("invalid signal direction")}
    if(tw<=0){stop("invalid time window")}
    seqq <- seq(0.05,0.95,0.05)
    if(!is.na(qdetect)){
	idq  <- which(abs(seqq-qdetect)<1e-3)
	Nidq <- length(idq)
	if(Nidq==0){
	    print("warning, desired quantile not found")
	    qdetect <- NA
	}
    }
    
    ## Reading pre-processed threshold file
    ## Sorry but the following part is hard-coded. The file with the rain
    ## thresholds should be moved to the /USERS/lte so that everybody can access it.
    ## I didn't have time to do it yet so if you have some time left...
    ## Don't forget to modify the path in "Rain-thresholds.R"
    file <- "/USERS/mschleis/Microwave_Links/Data/Thresholds/thresholds.txt"
    Data <- read.table(file,header=FALSE)
    tab_idl  <- Data[[1]]
    tab_idc  <- Data[[2]]
    tab_tw   <- Data[[3]]
    mean_LxR <- Data[[4]]
    mean_RxL <- Data[[5]]

    v   <- c(tab_idl==idl)*c(tab_idc==idc)*c(tab_tw==tw)
    id  <- which(v==1)
    Nid <- length(id)
    
    if(Nid==1){
	if(is.na(qdetect)){
	    if(way=="LxR"){sill <- mean_LxR[id]}
	    if(way=="RxL"){sill <- mean_RxL[id]}
	}
	if(!is.na(qdetect)){
	    if(way=="LxR"){subData <- Data[[5+idq]]}
	    if(way=="RxL"){subData <- Data[[5+length(seqq)+idq]]}
	    sill <- subData[id]
	}
    }
    if(Nid==0){stop("error: no rain threshold found.")}
    return(sill)
}

##########################################################################

subtract_baseline <- function(tabA,tabB){
    
    ## Subtracts the attenuation baseline from the link signal

    ## Inputs:
    ## tabA = Attenuation measurements
    ## tabB = Attenuation baseline

    ## Ouput:
    ## newA = New attenuation values

    Na <- length(tabA)
    Nb <- length(tabB)
    if(Na!=Nb){stop("dimensions do not match")}
    newA <- tabA-tabB
    newA <- newA*c(newA>0)
    return(newA)
}

##########################################################################

subtract_wet_antenna <- function(tabA,link,dir){

    ## Subtracts the wet antenna from the link signal

    ## Inputs:
    ## tabA = Attenuation measurements (after baseline subtraction)
    ## link = Link number (13 or 14)
    ## dir  = direction ("LxR" or "RxL")

    ## Output:
    ## newA = New attenuation measurements

    wet <- c(NA,NA)
    if(link==14 && dir=="LxR"){wet <- c(5.0,0.45)}
    if(link==14 && dir=="RxL"){wet <- c(5.2,0.45)}
#     if(link==14){wet <- c(4.5,0.40)}
    if(link==13){wet <- c(3.5,1.0)}

    L <- get.link_length(link)
    specA <- 1000*tabA/L
    wetA  <- wet[1]*(1-exp(-specA*wet[2]))
    newA  <- tabA-wetA
    newA  <- newA*c(newA>0)
    return(newA)

}

##########################################################################

perf_eval <- function(link_sd,link_Att,link_B,radarR,sill){

    ## Computes the performance of rain detection algorithm using microwave links
    ## For comparison, radar data is considered as truth reference. 

    ## Input:
    ## link_sd  = vector of syncronized local standard deviation for the link.
    ## link_Att = vector of syncronized attenuation for the link
    ## link_B   = vector of syncronized baseline attenuation for the link
    ## radarR   = radar rain-rate [in mm/h], syncronized with link
    ## sill     = rain detection sill for standard deviation

    ## Output:
    ## perf = c(Tpositif,Fpositif,Fnegatif,Tnegatif,nRain,nDry,Rdetect,TotalR) where 
    ## Tpositif  = Number of true rain detections (captured rainy periods)
    ## Fpositif  = Number of false rain detections
    ## Tnegatif  = Number of true dry detections
    ## Fnegatif  = Number of false dry detections (missed rainy periods)
    ## nRain     = Number of Rainy periods
    ## nDry      = Number of Dry periods
    ## Rdetect   = Rain amount detected by link
    ## TotalR    = Total amount of rain (as seen by the radar)

    ## Last Modifications: M.Schleiss, September 2009
   
    ## Run some basic tests
    Nlink_sd  <- length(link_sd)
    Nlink_Att <- length(link_Att)
    Nlink_B   <- length(link_B)
    NradarT   <- length(radarT)
    NradarR   <- length(radarR)
    
    sizes <- unique(c(Nlink_sd,Nlink_Att,Nlink_B,NradarT,NradarR))
    if(length(sizes)>1){stop("vectors must have same length")}
    
    Tp <- 0
    Fp <- 0
    Tn <- 0
    Fn <- 0   
    Rdetect <- 0
    TotalR  <- sum(radarR,na.rm=TRUE)
    nRain   <- sum(radarR>0,na.rm=TRUE)
    nDry    <- sum(radarR==0,na.rm=TRUE) 
    
    for(i in 1:Nlink_sd){
	sd <- link_sd[i]
	R  <- radarR[i]
	A  <- link_Att[i]
	B  <- link_B[i]
	if(!is.na(sd*R*A*B)){
	    if(A>=B){
		if(sd>=sill && R>0){
		    Tp <- Tp+1
		    Rdetect <- Rdetect+R
		}
		if(sd>=sill && R==0){Fp <- Fp+1}
		if(sd<sill && R>0){Fn <- Fn+1}
		if(sd<sill && R==0){Tn <- Tn+1}
	    }
	    else{
		if(sd>=sill && R>0){
		    Tp <- Tp+1
		    Rdetect <- Rdetect+R
		}
		if(sd>=sill && R==0){Tn <- Tn+1}
		if(sd<sill && R>0){Tp <- Tp+1}
		if(sd<sill && R==0){Tn <- Tn +1}
	    }
	}
    }
        
    perf <- c(Tp,Fp,Fn,Tn,nRain,nDry,Rdetect,TotalR)
    return(perf)
}
    
##########################################################################

fit.baseline <- function(tabX,tabT,dt,level,back=TRUE){

    ## Fits an attenuation baseline on a microwave link signal
    ## based on moving standard deviation criterium.

    ## Needs the library "lib_stats.R" to be loaded !!
    
    ## Input:
    ## tabX  = vector of attenuation values
    ## tabT  = vector of time values (in numeric format)
    ## dt    = time window (use same units than tabT)
    ## level = detection level for standard deviation (See get.rainsill)
    ## back  = Logical, if TRUE the window is taken backwards

    ## Output:
    ## baseline = attenuation baseline (same length than tabX)

    ## Source Code: M.Schleiss, April 2009

    ## Run some basic tests
    NtabX <- length(tabX)
    NtabT <- length(tabT)
    nNA   <- sum(is.na(tabT))
    if(NtabX!=NtabT){stop("vectors must have same size")}
    if(nNA>0){stop("NA values are not allowed in tabT")}
    if(NtabX<3){stop("tabX must have at least 3 elements")}
    tabT <- as.numeric(tabT)
    Tinf <- tabT-dt
    Tsup <- tabT+dt*(!back)
    baseline <- rep(NA,NtabX)
    last_baseline <- mode(tabX)
    if(is.na(last_baseline)){last_baseline <- mode(tabX)}
    j <- 1
    k <- 1
    for(i in 2:NtabX){
	while(tabT[j]<Tinf[i] && j<NtabX){j <- j+1}
	while(tabT[k]<Tsup[i] && k<NtabX){k <- k+1}
	baseline[i] <- last_baseline
	if(k>j){
	    subX  <- tabX[j:k]
	    notNA <- sum(!is.na(subX))
	    if(notNA>1){
		sd <- sd(subX,na.rm=TRUE)
		if(sd<=level){
		    baseline[i]   <- median(subX,na.rm=TRUE)
		    last_baseline <- baseline[i]
		}
		if(!is.na(tabX[i])){
		    if(tabX[i]<=(last_baseline-2)){
			baseline[i]   <- median(tabX[i],na.rm=TRUE)
			last_baseline <- baseline[i]
		    }
		}
	    }
	}
    }  
    return(baseline)
}

##########################################################################

fit.baseline_using_radar <- function(tabT_link,tabA,tabT_radar,tabR){

    ## Fits an attenuation baseline using radar data for the
    ## identification of dry and rainy periods.

    ## Inputs:
    ## tabT_link  = link timetable [s]
    ## tabA       = link attenuation [dB]
    ## tabT_radar = radar timetable [s]
    ## tabR       = radar rain-rate [mm/h]

    ## Outputs:
    ## baseline = attenuation baseline [dB]

    ## Preliminary operations (1) Check adequacy of inputs
    tabT_link   <- as.numeric(tabT_link)
    tabT_radar  <- as.numeric(tabT_radar)
    NtabT_link  <- length(tabT_link)
    NtabA       <- length(tabA)
    NtabT_radar <- length(tabT_radar)
    NtabR       <- length(tabR)
    if(NtabT_link!=NtabA){stop("tabT_link and tabA must have same dimensions")}
    if(NtabT_radar!=NtabR){stop("tabT_radar and tabR must have same dimensions")}

    ## Preliminary operations (2) Delete NA values in radar
    tab_index   <- which(!is.na(tabT_radar))
    tab_index   <- intersect(tab_index,which(!is.na(tabR)))
    NtabR       <- length(tab_index)
    NtabT_radar <- NtabR
    tabR        <- tabR[tab_index]
    tabT_radar  <- tabT_radar[tab_index]

    ## Preliminary operations (3) Extract dry periods
    Idry <- which(tabR==0)
    Ndry <- length(Idry)

    ## Construct baseline for dry periods
    baseline <- rep(NA,NtabT_link)
    if(Ndry>0){
	for(i in Idry){
	    t <- tabT_radar[i]
	    tab_index1 <- which(tabT_link>(t-300))
	    tab_index2 <- which(tabT_link<=t)
	    tab_index3 <- intersect(tab_index1,tab_index2)
	    N <- length(tab_index3)
	    if(N==0){next}
	    subA <- tabA[tab_index3]
	    notNA <- sum(!is.na(subA))
	    if(notNA==0){next}
	    baseline[tab_index3[N]] <- mean(subA,na.rm=TRUE)
	}
    }

    ## Construct baseline for wet or missing periods
    last_baseline <- tabA[which(!is.na(tabA))[1]]
    for(i in 2:NtabT_link){
	b <- baseline[i]
	if(!is.na(b)){last_baseline <- b}
	else{
	    baseline[i] <- min(last_baseline,tabA[i])
	    last_baseline <- baseline[i]
	}
    }
    return(baseline)
} 

##########################################################################

compute_ratios <- function(tabAh,tabAv,minAh=0){

    ## Compute raw attenuation ratios 
  
    ## Inputs:
    ## tabAh = Attenuation measurements at polH
    ## tabAv = Attenuation measurements at polV
    ## minAh = Minimum attenuation at polH to be considered

    ## Output:
    ## tabr = Attenuation ratios (only for rainy periods)

    Nh <- length(tabAh)
    Nv <- length(tabAv)
    if(is.na(minAh)){stop("invalid expression for minAh")}
    if(Nh!=Nv){stop("dimensions of tabAh and tabAv do not match")}
    if(Nh==0){return(c())}
    if(Nh>0){
	tabr <- rep(NA,Nh)
	for(i in 1:Nh){
	    Ah <- tabAh[i]
	    Av <- tabAv[i]
	    if(is.na(Ah*Av)){next}
	    if(Ah<=0 || Av<=0){next}
	    if(Ah<minAh){next}
	    tabr[i] <- Av/Ah
	}
	return(tabr)
    }
}

##########################################################################

retrieve_DSD <- function(tab_Ah,tab_Av,idl,dir,temp,minAh=0){

    ## Retrieves the DSD parameters mu ; lam ; Nt

    ## Inputs:
    ## tab_Ah = Attenuation measurements on polH
    ## tab_Av = Attenuation measurements on polV
    ## idl    = Link number (13 or 14)
    ## dir    = Direction ("LxR" or "RxL")
    ## temp   = Temperature (rounded at 5°C)
    ## minAh  = minimum attenuation at polH to be considered

    ## Outputs:
    ## retrieved_DSD = retrieved DSD matrix mu|lam|Nt

    ## Some basic stuff
    Nh <- length(tab_Ah)
    Nv <- length(tab_Av)
    if(Nh!=Nv || Nh==0){stop()}
    temp <- 5*round(temp/5)
    if(dir!="LxR" && dir!="RxL"){stop()}
    if(idl!=13 && idl!=14){stop()}
    L <- get.link_length(idl)

    ## Drop diameter sequence
    Dmin   <- 0.1
    Dmax   <- 7.0
    dD     <- 0.01
    seqD   <- seq(Dmin,Dmax,dD)

    ## Path to extinction cross sections
    ext_path  <- "/home/mschleis/Extinction_Cross_Sections"

    ## Read extinction cross-sections
    txt0 <- "extscatcross_temperature"
    file1 <- sprintf("%s/link_%i_channel_1_%s_%s_%1.2i.txt",ext_path,idl,dir,txt0,temp)
    file2 <- sprintf("%s/link_%i_channel_3_%s_%s_%1.2i.txt",ext_path,idl,dir,txt0,temp)	   
    Data_Ext1 <- read.table(file1)
    Data_Ext2 <- read.table(file2) 
    names(Data_Ext1) <- c("H","V")
    names(Data_Ext2) <- c("H","V")
    Ext_polV <- Data_Ext1[["V"]]
    Ext_polH <- Data_Ext2[["H"]]

#     seqa <- seq(-3.0,1.0,0.1)
#     seqb <- seq(0.1,3.0,0.1)
#     maxmu <- 20
#     seqmu <- seq(1,maxmu,0.05)
#     Nseqmu <- length(seqmu)
#     cand <- c()
#     for(a in seqa){
# 	for(b in seqb){
# 	    if(b<0){next}
# 	    if(b<=a){next}
# 	    if(b<(a/maxmu)){next}
# 	    seqlam <- a+b*seqmu
# 	    seqr <- rep(0,Nseqmu)
# 	    for(i in 1:Nseqmu){
# 		mu  <- seqmu[i]
# 		lam <- seqlam[i]
# 		num <- sum(Ext_polV*seqD^mu*exp(-lam*seqD),na.rm=TRUE)
# 		denom <- sum(Ext_polH*seqD^mu*exp(-lam*seqD),na.rm=TRUE)
# 		seqr[i] <- num/denom
# 	    }   
# 	    minr <- min(seqr,na.rm=TRUE)
# 	    maxr <- max(seqr,na.rm=TRUE)
# 	    if(minr>0.85){next}
# 	    if(maxr<0.95){next}
# 	    cand <- rbind(cand,c(a,b,minr,maxr))
# 	}
#     }

    ## Defining the mu-lambda relationship: lam = a + b*mu
    a <- 0.7
    b <- 1.4
#     a <- -0.96
#     b <- 1.65
    dmu <- 0.05
    min_mu <- 1.0
    max_mu <- 30
    seq_mu <- seq(min_mu,max_mu,dmu)
    Nseq_mu <- length(seq_mu)
    seq_lam <- a + b*seq_mu
    seq_ratio <- rep(NA,Nseq_mu)

    ## Computing theoretical attenuation ratios Av/Ah
    for(i in 1:Nseq_mu){
	mu  <- seq_mu[i]
	lam <- seq_lam[i]
	num <- sum(Ext_polV*seqD^mu*exp(-lam*seqD),na.rm=TRUE)
	denom <- sum(Ext_polH*seqD^mu*exp(-lam*seqD),na.rm=TRUE)
	seq_ratio[i] <- num/denom
    }
    print(sprintf("ratio is between %1.3f and %1.3f",min(seq_ratio),max(seq_ratio)))

    ## Retrieving DSD
    retrieved_DSD <- matrix(NA,nrow=Nh,ncol=3)
    for(i in 1:Nh){
	Ah <- tab_Ah[i]
	Av <- tab_Av[i]
	if(is.na(Ah*Av)){next}
	if(Ah*Av==0){next}
	if(Av>Ah){next}
	if(Ah<=minAh){next}
	r <- Av/Ah
	index <- which.min(abs(seq_ratio-r))
	if(index==1 || index==Nseq_mu){next}
	mu <- seq_mu[index]
	lam <- seq_lam[index]
	denom <- sum(Ext_polH*seqD^mu*exp(-lam*seqD),na.rm=TRUE)
	num   <- sum(seqD^mu*exp(-lam*seqD),na.rm=TRUE)
	Nt    <- (num/denom)*Ah*0.5*(r+1)*log(10)*1000/L	
	if(is.na(Nt)){next}
	retrieved_DSD[i,] <- c(mu,lam,Nt)
    }
    return(retrieved_DSD)
}

##########################################################################

retrieve_bulks <- function(DSD,idl,dir,temp){

    ## Compute the radar bulk variables R ; Zh ; Zdr

    ## Inputs:
    ## DSD  = DSD matrix (mu|lam|Nt)
    ## idl  = Link identifier (13 or 14)
    ## dir  = Direction ("LxR" or "RxL")
    ## temp = Temperature (rounded at 5°C)

    ## Outputs:
    ## bulks = bulk matrix (R|Zh|Zdr)
   
    ## Some variables
    fRadar <- 5.6
    wRadar <- 5.353437
    Dmin   <- 0.1
    Dmax   <- 7.0
    dD     <- 0.01
    seqD   <- seq(Dmin,Dmax,dD)
    path   <- "/USERS/mschleis/2010_MWL_DSD_Retrieval/Saves"

    ## Read Backscattering cross-sections
    file <- sprintf("%s/radar_backscatcross_%1.2i.txt",path,temp)
    Data <- read.table(file)
    names(Data) <- c("H","V")
    Back_polH <- Data[["H"]]
    Back_polV <- Data[["V"]]

    ## Main program starts here
    temp  <- 5*round(temp/5)
    m     <- ref_index_water(temp,fRadar)
    tab_mu  <- DSD[,1]
    tab_lam <- DSD[,2]
    tab_Nt  <- DSD[,3]
    tab_R   <- gamma_DSD.R(tab_mu+1,tab_lam,tab_Nt,0.001)
    tab_Zh  <- gamma_DSD.Z(tab_mu+1,tab_lam,tab_Nt,wRadar,m,seqD,Back_polH)
    tab_Zv  <- gamma_DSD.Z(tab_mu+1,tab_lam,tab_Nt,wRadar,m,seqD,Back_polV)
    tab_Zdr <- tab_Zh-tab_Zv
    bulks <- matrix(c(tab_R,tab_Zh,tab_Zdr),nrow=dim(DSD)[1],ncol=3)
    return(bulks)
}

##########################################################################

resample_bulks <- function(bulks,tabT,newT){

    ## Inputs:
    ## bulks = bulk matrix (R|Zh|Zdr)
    ## tabT  = original timetable
    ## newT  = new timetable

    ## Outputs:
    ## new_bulks = resampled bulk matrix (R|Zh|Zdr)
    ##             Zh and Zdr are resampled in linear-space

    nrow <- length(newT)
    new_bulks <- matrix(NA,nrow=nrow,ncol=3)
    for(j in 1:3){
	if(j==1){Rsamp <- resample(bulks[,j],tabT,newT,300,"mean","backward")}
	if(j>1){Rsamp <- 10*log10(resample(10^(bulks[,j]/10),tabT,newT,300,"mean","backward"))}
	new_bulks[,j] <- Rsamp
    }
    return(new_bulks)

}

##########################################################################

identify_time_shift <- function(tabX,tabY,Nfirst){

    ## Identifies the most probable time shift between tabX and tabY
    ## Hypothesis: tabX[i] = f(tabY[i-s]) with f = monotonic continuous increasing function
    ## Identification is not based on a "rank collection matching algorithm"
    ## R(X[i]) = R(f(Y[i-s])) = R(Y[i-s]) because f does not change the ranks.
    ## Source code by Marc Schleiss, EPFL-LTE 2008-2009
    
    ## Inputs:
    ## tabX = vector of observations
    ## tabY = vector of observations (same size than tabX)
    ## Nfirst = restrict the collection match to the Nfirst values
    
    ## Output:
    ## MPS = Most probable shift
    
    ## Some basics
    Nx   <- length(tabX)
    Ny   <- length(tabY)
    Nfirst <- round(min(Nx,abs(Nfirst)))
    if(Nx==0){stop("vector of size zero")}
    if(Nx!=Ny){stop("tabX and tabY must have same size")}
    I1 <- which(!is.na(tabX))
    I2 <- which(!is.na(tabY))
    I  <- intersect(I1,I2)
    tabX <- tabX[I]
    tabY <- tabY[I]
    MPS <- NA
    
    Sx <- sort(tabX,decreasing=TRUE,index.return=TRUE)
    Sy <- sort(tabY,decreasing=TRUE,index.return=TRUE)
    
    I1 <- Sx$ix[1:Nfirst]
    I2 <- Sy$ix[1:Nfirst]
        
    seqS  <- seq(-36,36,1)
    NseqS <- length(seqS)
    match <- rep(0,NseqS)
    for(i in 1:NseqS){
	match[i] <- length(intersect(I1,I2-seqS[i]))
    }
    
    maxmatch <- max(match)
    Imax <- which(match==maxmatch)
    IMPS <- which.min(abs(seqS[Imax]))
    MPS  <- seqS[Imax[IMPS]]
    return(MPS)
}