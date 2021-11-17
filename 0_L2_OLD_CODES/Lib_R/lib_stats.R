################################################## Statistical Library ####################################################
##################################################  by Marc Schleiss ######################################################

## This is the 3rd version of the statistical library for R
## Please report any bug or modifications
## Feel free to add your own statistical functions

##########################################################################

mode <- function(tabX){

    ## Computes the mode of a series of values.
    
    ## Inputs:
    ## tabX = vector of real values
    
    ## Output:
    ## mode = most observed value in tabX. If there is no unique mode,
    ##        the mean mode value is returned.
	
    mode <- NA
    id   <- which(!is.na(tabX))
    Nid  <- length(id)
    if(Nid>0){
	tabX <- tabX[id]
	tabV <- unique(tabX)
	nV   <- length(tabV)
	tabN <- rep(0,nV)
	for(i in 1:nV){tabN[i] <- sum(tabX==tabV[i])}
	maxN <- max(tabN)
	idM  <- which(tabN==maxN)
	mode <- mean(tabV[idM])
    }
    return(mode)
}

##########################################################################

Spearman_correlation <- function(tabX,tabY){
    
    ## Computes the Spearman correlation coefficient
    ## This correlation coefficient is based on the ranks 
    ## of the observations. It is more robust than the standard 
    ## Pearson correlation coefficient
    
    ## Inputs:
    ## tabX = vector of real values (NA values allowed)
    ## tabY = vector of real values (NA values allowed, same size than tabX)
    
    nX <- length(tabX)
    nY <- length(tabY)
    
    if(nX!=nY){
	print("Error in Spearman: tabX and tabY must have same length")
	stop()
    }
    
    id1 <- which(!is.na(tabX))
    id2 <- which(!is.na(tabY))
    id  <- intersect(id1,id2)
    Nid <- length(id)
    
    if(Nid==0){spear <- NA}
    else{
	subX <- tabX[id]
	subY <- tabY[id]
	rX <- rank(subX,ties.method="average")
	rY <- rank(subY,ties.method="average")
	D2 <- sum((rY-rX)^2)
	spear <- 1-6*D2/(Nid*(Nid^2-1))
    }
    return(spear)
}

##########################################################################

autocorr <- function(tabX,min_lag,max_lag){

    ## Computes the autocorrelation function of tabX
    
    ## Inputs:
    ## tabX = vector of observations, equally spaced in time (NA values allowed)
    ## min_lag = minimum lag to consider (only integer value allowed)
    ## max_lag = maximum lag to consider (only integer value allowed)
    
    ## Output:
    ## AutoCorr = vector of autocorrelation for different lags

    ## Author: M.Schleiss EPFL-LTE 2009
    ## Remarks: If the observations in tabX are not equally spaced in time, 
    ## please start by resampling them using resample() or fast_resample()

    Nx       <- length(tabX)
    min_lag  <- max(1,round(min_lag))
    max_lag  <- max(round(max_lag),min_lag)
    max_lag  <- min(max_lag,Nx)    
    seq_lag  <- min_lag:max_lag
    Nseq_lag <- length(seq_lag)
    AutoCorr <- rep(NA,Nseq_lag)

    for(i in 1:Nseq_lag){
	lag <- seq_lag[i]
	X   <- tabX[(1+lag):Nx]
	Y   <- tabX[1:(Nx-lag)]
	if(sum(!is.na(X))>0 && sum(!is.na(Y))>0){
	    mX  <- mean(X,na.rm=TRUE)
	    mY  <- mean(Y,na.rm=TRUE)
	    sdX <- sqrt(mean((X-mX)^2,na.rm=TRUE))
	    sdY <- sqrt(mean((Y-mY)^2,na.rm=TRUE))
	    if(sdX>0 && sdY>0){
		mXY <- mean(X*Y,na.rm=TRUE)
		cor <- (mXY-mX*mY)/(sdX*sdY)
		AutoCorr[i] <- cor
	    }
	}

    }
    return(AutoCorr)
}

##########################################################################

moving_average <- function(tabX,tabT,dt,back=FALSE){

    ## Computes a moving average on the signal tabX
    
    ## Inputs:
    ##  tabX = vector of real values 
    ##  tabT = time indexes in increasing order (same size than tabX)
    ##  dt   = size of the time window for the moving average
    ##  back = logical, if TRUE, averaging is performed backwards only
    
    ## Outputs:
    ##  tab_mva = vector of moving average (same size than tabX)
    
    ## Remarks:
    ## To speed up calculations, a mobile time window [j,k] is used instead
    ## of successive calls to the which() command. In this way, the complexity 
    ## stays linear with the size of tabX.
    
    NtabX <- length(tabX)
    NtabT <- length(tabT)
    
    if(sum(is.na(tabT))>0){stop("NA values not allowed in tabT")}  
    if(NtabX!=NtabT){stop("tabX and tabT must be of same size")}
	
    tabT    <- as.numeric(tabT)          
    tab_mva <- rep(NA,NtabX)
    Tinf    <- tabT-dt
    Tsup    <- tabT+dt*1*(!back)
    j       <- 1
    k       <- 1
    
    for(i in 1:NtabX){
	while(tabT[j]<Tinf[i]){j <- j+1}
	if(k<NtabX){
	    while(tabT[k+1]<Tsup[i]){
		k <- k+1
		if(k==NtabX){break}
	    }
	}
	if(j<=k){
	    subX <- tabX[j:k]
	    nNA  <- sum(!is.na(subX))
	    if(nNA>0){tab_mva[i] <- mean(subX,na.rm=TRUE)}
	}
    }
    return(tab_mva) 
}

##########################################################################

moving_median <- function(tabX,tabT,dt,back=FALSE){

    ## Computes a moving median on the signal tabX
    
    ## Inputs:
    ##  tabX = vector of real values 
    ##  tabT = time indexes in increasing order (same size than tabX)
    ##  dt   = size of the time window for the moving median
    ##  back = logical, if TRUE, the moving median is performed backwards only
    
    ## Outputs:
    ##  tab_mvm = vector of moving median (same size than tabX)
    
    ## Remarks:
    ## To speed up calculations, a mobile time window [j,k] is used instead
    ## of successive calls to the which() command. In this way, the complexity 
    ## stays linear with the size of tabX.
    
    NtabX <- length(tabX)
    NtabT <- length(tabT)
    
    if(sum(is.na(tabT))>0){stop("NA values not allowed in tabT")}  
    if(NtabX!=NtabT){stop("tabX and tabT must be of same size")}
	
    tabT    <- as.numeric(tabT)          
    tab_mvm <- rep(NA,NtabX)
    Tinf    <- tabT-dt
    Tsup    <- tabT+dt*1*(!back)
    j       <- 1
    k       <- 1
    
    for(i in 1:NtabX){
	while(tabT[j]<Tinf[i]){j <- j+1}
	if(k<NtabX){
	    while(tabT[k+1]<Tsup[i]){
		k <- k+1
		if(k==NtabX){break}
	    }
	}
	if(j<=k){
	    subX <- tabX[j:k]
	    nNA  <- sum(!is.na(subX))
	    if(nNA>0){tab_mvm[i] <- median(subX,na.rm=TRUE)}
	}
    }
    return(tab_mvm) 
}
##########################################################################

moving_quantile <- function(tabX,tabT,dt,q, back=FALSE){

    ## Computes a moving quantile on the signal tabX
    
    ## Inputs:
    ##  tabX = vector of real values 
    ##  tabT = time indexes in increasing order (same size than tabX)
    ##  dt   = size of the time window for the moving standard deviation
    ##  back = logical, if TRUE, averaging is performed backwards only
    
    ## Outputs:
    ##  tab_quant = vector of standard deviations (same size than tabX)
    
    ## Remarks:
    ## To speed up calculations, a mobile time window [j,k] is used instead
    ## of successive calls to the which() command. In this way, the complexity 
    ## stays linear with the size of tabX.
    
    NtabX <- length(tabX)
    NtabT <- length(tabT)

   if(sum(is.na(tabT))>0){stop("NA values not allowed in tabT")}  
   if(NtabX!=NtabT){stop("tabX and tabT must be of same size")}
	
    tabT    <- as.numeric(tabT)          
    tab_quantile <- rep(NA,NtabX)
    Tinf    <- tabT-dt
    Tsup    <- tabT+dt*1*(!back)
    j       <- 1
    k       <- 1
    
    for(i in 1:NtabX){
	while(tabT[j]<Tinf[i]){j <- j+1}
	if(k<NtabX){
	    while(tabT[k+1]<Tsup[i]){
		k <- k+1
		if(k==NtabX){break}
	    }
	}
	if(j<=k){
	    subX <- tabX[j:k]
	    nNA  <- sum(!is.na(subX))
	    if(nNA>0){tab_quantile[i] <- quantile (subX,c(q),na.rm=TRUE)}
	}
    }
    
    return(tab_quantile) 
}

##########################################################################

moving_sd <- function(tabX,tabT,dt,back=FALSE){

    ## Computes a moving standard deviation on the signal tabX
    
    ## Inputs:
    ##  tabX = vector of real values 
    ##  tabT = time indexes in increasing order (same size than tabX)
    ##  dt   = size of the time window for the moving standard deviation
    ##  back = logical, if TRUE, averaging is performed backwards only
    
    ## Outputs:
    ##  tab_sd = vector of standard deviations (same size than tabX)
    
    ## Remarks:
    ## To speed up calculations, a mobile time window [j,k] is used instead
    ## of successive calls to the which() command. In this way, the complexity 
    ## stays linear with the size of tabX.
    
    NtabX <- length(tabX)
    NtabT <- length(tabT)
    
    if(sum(is.na(tabT))>0){stop("NA values not allowed in tabT")}  
    if(NtabX!=NtabT){stop("tabX and tabT must be of same size")}
	
    tabT    <- as.numeric(tabT)          
    tab_sd  <- rep(NA,NtabX)
    Tinf    <- tabT-dt
    Tsup    <- tabT+dt*1*(!back)
    j       <- 1
    k       <- 1
    
    for(i in 1:NtabX){
	while(tabT[j]<Tinf[i]){j <- j+1}
	if(k<NtabX){
	    while(tabT[k+1]<Tsup[i]){
		k <- k+1
		if(k==NtabX){break}
	    }
	}
	if(j==k && !is.na(tabX[i])){tab_sd[i] <- 0}
	if(j<k){
	    subX <- tabX[j:k]
	    nNA  <- sum(!is.na(subX))
	    if(nNA>=2){tab_sd[i] <- sd(subX,na.rm=TRUE)}
	}
    }
    return(tab_sd) 
}

##########################################################################

moving_cor <- function(tabX,tabY,tabT,w,back=TRUE,spearman=FALSE){

    ## Computes a moving correlation between tabX and tabT
    ## with window size given by w.
    
    ## Inputs:
    ## tabX = vector of real values (NA allowed)
    ## tabY = vector of real values (NA allowed, same size than tabX)
    ## tabT = timetable for tabX and tabY (in numeric format, no NA's allowed)
    ## w    = window size (same units than tabT)
    ## back = logical, if TRUE, the moving window is taken backwards only
    ## spearman = logical, if TRUE, the robust spearman correlation is used
    
    ## Outputs:
    ## tab_cor = vector of correlations (same size than tabX)
    
    tabT <- as.numeric(tabT)
    
    nX <- length(tabX)
    nY <- length(tabY)
    nT <- length(tabT)
    
    if(nX==0 || nY==0 || nT==0){
	print("Error in moving_cor: empty input")
	stop()
    }
    if(nX!=nY || nX!=nT || nY!=nT){
	print("Error in moving_cor: tabX, tabY and tabT must be of same size")
	stop()
    }
    if(sum(is.na(tabT))>0){
	print("Error in moving_cor: NA values are not allowed in tabT")
	stop()
    }
    
    nNAx <- which(!is.na(tabX))
    nNAy <- which(!is.na(tabY))
    nNA  <- intersect(nNAx,nNAy)
    
    nNA_tabX <- tabX[nNA]
    nNA_tabY <- tabY[nNA]
    nNA_tabT <- tabT[nNA] 
    tab_cor  <- rep(NA,nX)

    if(spearman==FALSE){
	for(i in 1:nX){
	    t    <- tabT[i]
	    tinf <- t-w
	    tsup <- t+w*(!back)
	    id1  <- which(nNA_tabT<=tsup)
	    id2  <- which(nNA_tabT>tinf)
	    id   <- intersect(id1,id2)
	    Nid  <- length(id)
	    if(Nid>0){
		subX  <- nNA_tabX[id]
		subY  <- nNA_tabY[id]
		mX    <- mean(subX)
		mY    <- mean(subY)
		mX2   <- mean(subX^2)
		mY2   <- mean(subY^2)
		mXY   <- mean(subX*subY)
		sdX   <- (mX2-mX^2)^(1/2)
		sdY   <- (mY2-mY^2)^(1/2)
		covXY <- mXY-mX*mY
		if(sdX>0 && sdY>0){
		    tab_cor[i] <- covXY/(sdX*sdY)
		}
		if(sdX==0 && sdY==0){
		    tab_cor[i] <- 0
		}
	    }
	}
    }
    if(spearman==TRUE){
	for(i in 1:nX){
	    t    <- tabT[i]
	    tinf <- t-w
	    tsup <- t+w*(!back)
	    id1  <- which(nNA_tabT<=tsup)
	    id2  <- which(nNA_tabT>tinf)
	    id   <- intersect(id1,id2)
	    Nid  <- length(id)
	    if(Nid>1){
		subX  <- nNA_tabX[id]
		subY  <- nNA_tabY[id]
		tab_cor[i] <- Spearman_correlation(subX,subY) 
	    }
	}    
    }
    return(tab_cor)
}

##########################################################################

fast_moving_cor <- function(tabX,tabY,tabT,w,back=TRUE,spearman=TRUE){

    ## Computes a moving correlation between tabX and tabY
    ## with window size given by w.
    ## This version does not use the which() function and runs in O(N*w).

    ## Inputs:
    ## tabX = vector of real values (NA allowed)
    ## tabY = vector of real values (NA allowed, same size than tabX)
    ## tabT = timetable for tabX and tabY (in numeric format, no NA's allowed)
    ## w    = window size (same units than tabT)
    ## back = logical, if TRUE, the moving window is taken backwards only
    ## spearman = logical, if TRUE, the robust spearman correlation is used
    
    ## Outputs:
    ## tab_cor = vector of correlations (same size than tabX)

    tabX <- as.numeric(tabX)
    tabY <- as.numeric(tabY)
    tabT <- as.numeric(tabT)

    Nx <- length(tabX)
    Ny <- length(tabY)
    Nt <- length(tabT)

    if(Nx!=Ny || Nx!=Nt){stop()}
    if(Nx==0){stop()}
    
    tab_cor <- rep(NA,Nt)
    Tleft   <- tabT-w
    if(back==TRUE){Tright <- tabT}
    else{Tright <- tabT+w}
    Tfake   <- c(tabT,tabT[Nt])

    l       <- 1
    r       <- 1

    if(spearman==FALSE){
	for(i in 1:Nt){
	    tl <- Tleft[i]
	    tr <- Tright[i]
	    while(tabT[l]<=tl){l <- l+1}
	    while(Tfake[r+1]<=tr){
		r <- r+1
		if(r>Nx){
		    r <- Nx
		    break
		}
	    }
	    if(tabT[l]>tl && tabT[r]<=tr && l<=r){
		subX   <- tabX[l:r]
		subY   <- tabY[l:r]
		nNA_x  <- which(!is.na(subX))
		nNA_y  <- which(!is.na(subY))
		notNA  <- intersect(nNA_x,nNA_y)
		NnotNA <- length(notNA)
		if(NnotNA>1){
		    subX  <- subX[notNA]
		    subY  <- subY[notNA]
		    mX    <- mean(subX)
		    mY    <- mean(subY)
		    mX2   <- mean(subX^2)
		    mY2   <- mean(subY^2)
		    mXY   <- mean(subX*subY)
		    sdX   <- (mX2-mX^2)^(1/2)
		    sdY   <- (mY2-mY^2)^(1/2)
		    covXY <- mXY-mX*mY
		    cor <- NA
		    if(sdX>0 && sdY>0){cor <- covXY/(sdX*sdY)}
		    if((sdX*sdY)==0 && (sdX+sdY)>0){cor <- 0}
                    if(sdX==0 && sdY==0){cor <- 1}
		    tab_cor[i] <- cor
		}
	    }
	}
    }

    if(spearman==TRUE){
	for(i in 1:Nt){
	    tl <- Tleft[i]
	    tr <- Tright[i]
	    while(tabT[l]<=tl){l <- l+1}
	    while(Tfake[r+1]<=tr){
		r <- r+1
		if(r>Nx){
		    r <- Nx
		    break
		}
	    }
	    if(tabT[l]>tl && tabT[r]<=tr && l<=r){
		subX   <- tabX[l:r]
		subY   <- tabY[l:r]
		nNA_x  <- which(!is.na(subX))
		nNA_y  <- which(!is.na(subY))
		notNA  <- intersect(nNA_x,nNA_y)
		NnotNA <- length(notNA)
		if(NnotNA>1){
		    subX  <- subX[notNA]
		    subY  <- subY[notNA]
		    #cor   <- Spearman_correlation(subX,subY)   # commentato il 18.11.2010
                    spearman_default <- cor(subX, subY, method = "spearman") 
		    tab_cor[i] <- spearman_default
		}
	    }
	}
    }
    return(tab_cor)
}

##########################################################################

moving_eff <- function(tabX,tabY,tabT,w,back=TRUE){

    ## Computes a moving efficience between tabX and tabY
    ## with window size given by w.
    
    ## Inputs:
    ## tabX = vector of real values (NA allowed)
    ## tabY = vector of real values (NA allowed, same size than tabX)
    ## tabT = timetable for tabX and tabY (in numeric format, no NA's allowed)
    ## w    = window size (same units than tabT)
    ## back = logical, if TRUE, the moving window is taken backwards only
 
    ## Outputs:
    ## tab_eff = vector of efficiences (same size than tabX)
    
    tabT <- as.numeric(tabT)
    
    nX <- length(tabX)
    nY <- length(tabY)
    nT <- length(tabT)
    
    if(nX==0 || nY==0 || nT==0){
	print("Error in moving_eff: empty input")
	stop()
    }
    if(nX!=nY || nX!=nT || nY!=nT){
	print("Error in moving_eff: tabX, tabY and tabT must be of same size")
	stop()
    }
    if(sum(is.na(tabT))>0){
	print("Error in moving_eff: NA values are not allowed in tabT")
	stop()
    }
    
    nNAx <- which(!is.na(tabX))
    nNAy <- which(!is.na(tabY))
    nNA  <- intersect(nNAx,nNAy)   
    nNA_tabX <- tabX[nNA]
    nNA_tabY <- tabY[nNA]
    nNA_tabT <- tabT[nNA] 
    tab_eff  <- rep(NA,nX)
       
    for(i in 1:nX){
      t    <- tabT[i]
      tinf <- t-w
      tsup <- t+w*(!back)
      id1  <- which(nNA_tabT<=tsup)
      id2  <- which(nNA_tabT>tinf)
      id   <- intersect(id1,id2)
      Nid  <- length(id)
      if(Nid>0){
        subX  <- nNA_tabX[id]
        subY  <- nNA_tabY[id]
        mX    <- mean(subX)      
        errXY   <- sum((subX-subY)^2)
        errXmX   <- sum((subX-mX)^2)
        
        if(errXmX!=0 && errXY!=0){
          tab_eff[i] <- 1 - (errXY/errXmX)
        }
        if(errXmX==0 && errXY==0){
          tab_eff[i] <- 1
        }
        if(errXmX==0 && errXY!=0){
          tab_eff[i] <- -Inf
        }       
      }
    }
    
    return(tab_eff)
  }


##########################################################################

fast_moving_diff <- function(tabX,tabT,w,back=TRUE){

    ## Computes Ndiff>0 - Ndiff<0 over a moving window of size w.
    ## This version does not use which() and runs in O(N*w).

    ## Inputs:
    ## tabX = vector of real values (NA allowed)
    ## tabT = timetable for tabX and tabY (in numeric format, no NA's allowed)
    ## w    = window size (same units than tabT)
    ## back = logical, if TRUE, the moving window is taken backwards only
    
    ## Outputs:
    ## tab_diff = vector of (Nup-Ndown)/w (same size than tabX)

    tabX <- as.numeric(tabX)
    tabT <- as.numeric(tabT)

    Nx <- length(tabX)
    Nt <- length(tabT)

    if(Nx!=Nt){stop()}
    if(Nx==0){stop()}
    
    tab_diff <- rep(NA,Nt)
    Tleft    <- tabT-w
    if(back==TRUE){Tright <- tabT}
    else{Tright <- tabT+w}
    Tfake   <- c(tabT,tabT[Nt])

    l       <- 1
    r       <- 1

    for(i in 1:Nt){
	tl <- Tleft[i]
	tr <- Tright[i]
	while(tabT[l]<=tl){l <- l+1}
	while(Tfake[r+1]<=tr){
	    r <- r+1
	    if(r>Nx){
		r <- Nx
		break
	    }
	}
	if(tabT[l]>tl && tabT[r]<=tr && l<=r){
	    subX   <- tabX[l:r]
	    nNA  <- which(!is.na(subX))
	    NnNA <- length(nNA)
	    if(NnNA>1){
		subX    <- subX[nNA]
		diffX   <- diff(subX)
		Nup     <- sum(diffX>0)
		Ndown   <- sum(diffX<0)
		tab_diff[i] <- (Nup-Ndown)/NnNA
	    }
	}
    }
    return(tab_diff)
}

##########################################################################

resample <- function(tabX,tabT1,tabT2,w,type="mean",window="centered"){

    ## Re-samples the values of tabX according to a new timetable
    ## For large timetables, you can use fast_resample() to speed up calculations

    ## Input:
    ## tabX   = vector of real values
    ## tabT1  = original timetable for tabX
    ## tabT2  = new timetable for the re-sampling
    ## w      = window size for the re-sampling (same units than tabT1)
    ## type   = the type of re-sampling. Must be one of the following:
    ##   "mean"   : arithmetic mean
    ##   "min"    : minimum value
    ##   "max"    : maximum value
    ##   "sd"     : standard deviation
    ##   "mode"   : most observed value
    ##   "sum"    : sum of values
    ##   "counts" : number of observations
    ##   window = type of averaging window. Must be one of the following:
    ##   "centered" : ]T-w/2,T+w/2[ 
    ##   "backward" : ]T-w,T]
    ##   "forward"  : [T,T+w[

    ## Output:
    ## Rsamp = re-sampled values of tabX (same size than tabT2)

    ## Author: M.Schleiss, May 11th 2009
    ## Modifications: March 29 2010

    Nx  <- length(tabX)
    Nt1 <- length(tabT1)
    Nt2 <- length(tabT2)
    if(Nx==0){stop("tabX is empty")}
    if(Nt1==0){stop("tabT1 is empty")}
    if(Nt2==0){stop("tabT2 is empty")}
    if(Nx!=Nt1){stop("dimensions of tabX and tabT1 do not match")}

    tabT1 <- as.numeric(tabT1)
    tabT2 <- as.numeric(tabT2)
    if(sum(is.na(tabT1))>0){stop("NA values not allowed in tabT1")}
    if(sum(is.na(tabT2))>0){stop("NA values not allowed in tabT2")}

    tab_index <- which(!is.na(tabX))
    Nx <- length(tab_index)
    if(Nx==0){return(rep(NA,Nt2))}
    tabX  <- tabX[tab_index]
    tabT1 <- tabT1[tab_index]

    if(is.na(w)){stop("NA value not allowed in w")}
    if(w<=0){stop("w must be strictly positive")}

    tab_type <- c("mean","min","max","sd","mode","sum","counts")
    index_type <- which(tab_type==type)
    if(length(index_type)!=1){stop("invalid type")}

    if(window=="centered"){
	Tleft  <- tabT2-w/2+1e-6
	Rright <- tabT2+w/2-1e-6
    }
    if(window=="backward"){
	Tleft  <- tabT2-w+1e-6
	Tright <- tabT2
    }
    if(window=="forward"){
	Tleft  <- tabT2
	Tright <- tabT2+w-1e-6
    }

    tab_window <- c("centered","backward","forward")
    index_window <- which(tab_window==window)
    if(length(index_window)!=1){stop("invalid window")}

    if(type=="sum"){opt <- function(X){return(sum(X))}}
    if(type=="mean"){opt <- function(X){return(mean(X))}}
    if(type=="min"){opt <- function(X){return(min(X))}}
    if(type=="max"){opt <- function(X){return(max(X))}}
    if(type=="mode"){opt <- function(X){return(mode(X))}}
    if(type=="counts"){opt <- function(X){return(length(X))}}
    if(type=="sd"){
	opt <- function(X){
	    if(length(unique(X))<=1){return(NA)}
	    else{return(sd(X))}
	}
    }

    newX <- rep(NA,Nt2)
    for(i in 1:Nt2){
	t <- tabT2[i]
	tab_index1 <- which(tabT1>=Tleft[i])
	tab_index2 <- which(tabT1<=Tright[i])
	tab_index3 <- intersect(tab_index1,tab_index2)
	if(length(tab_index3)==0){next}
	newX[i] <- opt(tabX[tab_index3])
    }
    return(newX)
}

##########################################################################

fast_resample <- function(tabX,tabT,newT,w,type="mean",tol_NA=TRUE){

    ## Re-samples the values of tabX according to a new timetable
    ## This algorithm does not use the which() command for resampling
    ## This is particularly useful for large timetables. 

    ## Input:
    ## tabX = vector of real values
    ## tabT = original timetable for tabX
    ## newT = new timetable for the re-sampling
    ## w    = size of the time window for resampling
    ## type = the type of re-sampling (mean,min,max,sum)
    ## tol_NA = NA tolerance. If true, NA's are tolerated

    ## Output:
    ## newX = re-sampled values of tabX (same size than newT)

    tabT <- as.numeric(tabT)
    newT <- as.numeric(newT)

    Nx <- length(tabX)
    Nt <- length(tabT)
    Nnew <- length(newT)

    if(Nx!=Nt){stop()}
    if(Nx==0){stop()}
    if(w<0){stop()}
    if(Nnew==0){stop()}

    if(type=="sum"){myfunc <- function(tabV){return(sum(tabV,na.rm=tol_NA))}}
    if(type=="mean"){myfunc <- function(tabV){return(mean(tabV,na.rm=tol_NA))}}
    if(type=="min"){myfunc <- function(tabV){return(min(tabV,na.rm=tol_NA))}}
    if(type=="max"){myfunc <- function(tabV){return(max(tabV,na.rm=tol_NA))}}
    if(type=="mode"){myfunc <- function(tabV){return(mode(tabV,na.rm=tol_NA))}}    

    Tleft  <- newT-w
    Tright <- newT
    fakeT  <- c(tabT,tabT[Nt])
    Rsamp  <- rep(NA,Nnew)
    ibegin <- 1

    while(Tright[ibegin]<tabT[1]){
	ibegin <- ibegin+1
	if(ibegin==(Nnew+1)){break}
    }

    if(ibegin<=Nnew){
	l <- 1
	r <- 1
	for(i in ibegin:Nnew){
	    while(tabT[l]<=Tleft[i]){
		l <- l+1
		if(l>Nx){
		    l <- Nx
		    break
		}
	    }
	    while(fakeT[r+1]<=Tright[i]){
		r <- r+1
		if(r==(Nt+1)){
		    r <- Nt
		    break
		}
	    }
	    if(l<=r && Tleft[i]<tabT[l] && tabT[r]<=Tright[i]){
		subX <- tabX[l:r]
		id_nNA <- which(!is.na(subX))
		if(length(id_nNA)>0){Rsamp[i] <- myfunc(subX)}
	    }
	}
    }
    return(Rsamp)
}

##########################################################################

resample_data <- function(X,tabT,newT,operator,w,type,na.action="ignore",p=1){

    ## Resamples the data in X according to a new timetable

    ## Inputs:
    ##  X = a data matrix (numeric format) with N rows and M columns
    ##  tabT = a timetable (numeric format) of size Nx1, in ascending order
    ##  newT = a new timetable (numeric format) of size newN (same units than tabT)
    ##  operator = a character string specifying the resampling operator
    ##    "mean"  = arithmetic mean
    ##    "min"   = minimum value
    ##    "max"   = maximum value
    ##    "sd"    = standard deviation
    ##    "count" = number of values
    ##  w = the size of the moving window (same units than tabT)
    ##  type = a character string specifying the type of resampling
    ##    "backward": the operator is applied to ]t-w,t]
    ##    "forward": the operator is applied to [t,t+w[
    ##    "symmetric": the operator is applied to ]t-w,t+w[
    ##  na.action = a character string that specifies what to do with NA values.
    ##    "ignore" = ignore all NA values
    ##    "discard" = discard all measurements with NA values
    ##    "ignore_if" = ignore the NA values if they represent less than p percent
    ##  p = tolerance level for NA values. Only necessary if na.action="ignore_if"

    ## Outputs:
    ## newX = a data matrix (numeric format) with newN rows and M columns
    ##        containing the resampled values.

    ## Remarks:
    ## (1) At least 2 different values are needed to compute the standard deviation
    ## (2) w must be strictly positive
    ## (3) tabT and newT must be sorted in ascending order
    ## (4) backward and forward windows are "semi-open". Symmetric windows are open.
    ## (5) In presence of M time series with identical timetable (of size N), it is 
    ##     faster to make one call to resample_data with a matrix input X (of size NxM) 
    ##     rather than M separate calls with input Nx1
    ## (6) The algorithm used for resampling is based on two moving indexes (I,J).
    ##     At each time step, the indexes are updated to match the new time interval.
    ##     In this way, the complexity is linear w.r.t. the size of X

    ## Author: Marc Schleiss, EPFL-LTE 2008-2012

    ########################################

    N <- dim(X)[1]
    M <- dim(X)[2]
    newN <- length(newT)

    if(length(tabT)!=N){stop("dimensions of X and tabT do not match")}
    if(newN==0){stop("newT is empty")}
    if(is.na(w)){stop("w cannot be NA")}
    if(w<=0){stop("w must be strictly positive")}
    if(is.na(p)){stop("p cannot be NA")}
    if(p<0){stop("p cannot be smaller than zero")}
    if(sum(sort(tabT)!=tabT)>0){stop("tabT must be sorted in ascending order")}
    if(sum(sort(newT)!=newT)>0){stop("tabT must be sorted in ascending order")}
    if(p>1){warning("p is larger than 1. Please use a value between 0 and 1")}
    if(p==0){warning("p is equal to zero. Please use na.action=discard")}


    tab_opt    <- c("mean","min","max","sd","count")
    tab_type   <- c("backward","forward","symmetric")
    tab_action <- c("ignore","discard","ignroe_if")
    
    id_opt    <- which(tab_opt==operator)
    id_type   <- which(tab_type==type)
    id_action <- which(tab_action==na.action)

    if(length(id_opt)==0){stop("invalid operator")}
    if(length(id_type)==0){stop("invalid type")}
    if(length(id_action)==0){stop("invalid na.action")}
    
    if(operator=="mean"){
	if(na.action=="ignore"){
	    opt <- function(tabX){
		if(sum(!is.na(tabX))==0){return(NA)}
		else{return(mean(tabX,na.rm=TRUE))}
	    }
	}
	if(na.action=="discard"){
	    opt <- function(tabX){
		n <- length(tabX)
		if(n==0){return(NA)}
		else{return(mean(tabX))}
	    }
	}
	if(na.action=="ignore_if"){
	    opt <- function(tabX){
		n <- length(tabX)
		if(n==0){return(NA)}
		q <- sum(is.na(tabX))/n
		if(q>=p){return(NA)}
		else{return(mean(tabX,na.rm=TRUE))}
	    }
	    
	}
    }
    if(operator=="min"){
	if(na.action=="ignore"){
	    opt <- function(tabX){
		if(sum(!is.na(tabX))==0){return(NA)}
		else{return(min(tabX,na.rm=TRUE))}
	    }
	}
	if(na.action=="discard"){
	    opt <- function(tabX){
		n <- length(tabX)
		if(n==0){return(NA)}
		else{return(min(tabX))}
	    }
	}
	if(na.action=="ignore_if"){
	    opt <- function(tabX){
		n <- length(tabX)
		if(n==0){return(NA)}
		q <- sum(is.na(tabX))/n
		if(q>=p){return(NA)}
		else{return(min(tabX,na.rm=TRUE))}
	    }
	}
    }
    if(operator=="max"){
	if(na.action=="ignore"){
	    opt <- function(tabX){
		if(sum(!is.na(tabX))==0){return(NA)}
		else{return(max(tabX,na.rm=TRUE))}
	    }
	}
	if(na.action=="discard"){
	    opt <- function(tabX){
		n <- length(tabX)
		if(n==0){return(NA)}
		else{return(max(tabX))}
	    }
	}
	if(na.action=="ignore_if"){
	    opt <- function(tabX){
		n <- length(tabX)
		if(n==0){return(NA)}
		q <- sum(is.na(tabX))/n
		if(q>=p){return(NA)}
		else{return(max(tabX,na.rm=TRUE))}
	    }
	}
    }
    if(operator=="sd"){
	if(na.action=="ignore"){
	    opt <- function(tabX){
		nX <- length(tabX)
		if(nX<2){return(NA)}
		id <- which(!is.na(tabX))
		n  <- length(id)
		if(n<2){return(NA)}
		tabX <- tabX[id]
		uX <- unique(tabX)
		if(length(uX)<2){return(NA)}
		return(sd(tabX))
	    }
	}
	if(na.action=="discard"){
	    opt <- function(tabX){
		nX <- length(tabX)
		if(nX<2){return(NA)}
		id <- which(!is.na(tabX))
		n  <- length(id)
		if(n<2){return(NA)}
		uX <- unique(tabX)
		if(length(uX)<2){return(NA)}
		return(sd(tabX))
	    }
	}
	if(na.action=="ignore_if"){
	    opt <- function(tabX){
		nX <- length(tabX)
		if(nX<2){return(NA)}
		id <- which(!is.na(tabX))
		n  <- length(id)
		if(n<2){return(NA)}
		q <- n/nX
		if(q>=p){return(NA)}
		tabX <- tabX[id]
		uX <- unique(tabX)
		if(length(uX)<2){return(NA)}
		return(sd(tabX))
	    }
	}
    }
    if(operator=="count"){
	if(na.action=="ignore"){
	    opt <- function(tabX){return(sum(!is.na(tabX)))}
	}
	if(na.action=="discard"){
	    opt <- function(tabX){
		S <- sum(is.na(tabX))
		if(S==0){return(sum(!is.na(tabX)))}
		if(S>0){return(NA)}
	    }
	}
	if(na.action=="ignore_if"){
	    opt <- function(tabX){
		n <- length(tabX)
		if(n==0){return(NA)}
		q <- sum(is.na(tabX))/n
		if(q>=p){return(NA)}
		else{return(sum(!is.na(tabX)))}
	    }  
	}
    }

    if(type=="backward"){
	leftT  <- newT-w + 1e-12
	rightT <- newT
    }
    if(type=="forward"){
	leftT  <- newT
	rightT <- newT+w - 1e-12
    }
    if(type=="symmetric"){
	leftT  <- newT-w + 1e-12
	rightT <- newT+w - 1e-12
    }

    newX <- matrix(NA,nrow=newN,ncol=M)
    I <- 1
    J <- 1
    for(i in 1:newN){
	tl <- leftT[i]
	tr <- rightT[i]
	while(tabT[I]<tl){
	    I <- I+1
	    if(I>N){
		I <- N
		break
	    }
	}
	if(J<N){
	    while(tabT[J+1]<=tr){
		J <- J+1
		if(J==N){break}
	    }
	}
	if(tabT[I]<=tr && tabT[J]>=tl){
	    for(j in 1:M){newX[i,j] <- opt(X[I:J,j])}
	}
    }
    return(newX)
}

####################################################################

syncronize <- function(tabX,tabY,TimeX,TimeY){

    ## Syncronizes tabX with tabY with respect to a common timetable
    
    ## Input:
    ##   tabX  = 1st vector of real values
    ##   tabY  = 2nd vector of real values
    ##   TimeX = timetable for tabX (in increasing order)
    ##   TimeY = timetable for tabY (in increasing order)
    
    ## Output:
    ##   M = syncronized matrix (syncT|syncX|syncY)
    ##       syncT = syncronized timetable
    ##       syncX = syncronized values from tabX
    ##       syncY = syncronized values from tabY
    
    ## Comments: 
    ## Simultaneous measurements are detected using two indexes [j,k]
    ## that are updated at each step of the algorithm. In this way, the
    ## complexity is linear with the size of the input. This is not the 
    ## case for vector comparison.
    
    ## Author: M.Schleiss, May 2009
    
    Nx   <- length(tabX)
    Ny   <- length(tabY)
    Ntx  <- length(TimeX)
    Nty  <- length(TimeY)
    
    if(Nx==0){stop("tabX is empty")}
    if(Ny==0){stop("tabY is empty")}
    if(Ntx==0){stop("TimeX is empty")}
    if(Nty==0){stop("TimeY is empty")}
    if(Nx!=Ntx){stop("tabX and TimeX must have same size")}
    if(Ny!=Nty){stop("tabY and TimeY must have same size")}
    
    NAtx <- sum(is.na(TimeX))
    NAty <- sum(is.na(TimeY))
    
    if(NAtx>0){stop("NA value in TimeX")}
    if(NAty>0){stop("NA value in TimeY")}
    
    TimeX <- as.numeric(TimeX)
    TimeY <- as.numeric(TimeY)
        
    syncT  <- intersect(TimeX,TimeY)
    NsyncT <- length(syncT)
    if(NsyncT==0){M <- matrix(NA,ncol=3)}
    if(NsyncT>0){
	M     <- matrix(NA,nrow=NsyncT,ncol=3)
	M[,1] <- syncT
	j     <- 1
	k     <- 1
	for(i in 1:NsyncT){
	    t  <- syncT[i]
	    while(TimeX[j]<t && j<Nx){j <- j+1}
	    while(TimeY[k]<t && k<Ny){k <- k+1}
	    M[i,2] <- tabX[j]
	    M[i,3] <- tabY[k]
	}
	v1 <- !is.na(M[,2])
	v2 <- !is.na(M[,3])
	v3 <- v1*v2
	selection <- which(v3==1)
	Nselect <- length(selection)
	if(Nselect==0){M <- matrix(NA,ncol=3)}
	if(Nselect>0){M <- M[selection,]}
    }
    return(M)
}

##########################################################################

power_law.fit <- function(tab_x,tab_y,n_steps=10000){

  ## Fits a power law by least squares: tab_y = a*tab_x^b

  ## Usefull functions
  gradient <- function(coef,tab_x,tab_y){
    a <- coef[1]
    b <- coef[2]
    dfa <- -2*sum(tab_x^b*(tab_y-a*tab_x^b),na.rm=TRUE)
    dfb <- -2*a*sum(log(tab_x)*tab_x^b*(tab_y-a*tab_x^b),na.rm=TRUE)
    df <- matrix(data=c(dfa,dfb),nrow=2,ncol=1)
    return(df)
  }
  
  jacobien <- function(coef,tab_x,tab_y){
    a <- coef[1]
    b <- coef[2]
    J11 <- 2*sum(tab_x^(2*b),na.rm=TRUE)
    J12 <- sum(4*a*tab_x^(2*b)*log(tab_x)-2*tab_y*tab_x^b*log(tab_x),na.rm=TRUE)
    J21 <- J12
    J22 <- -2*a*sum((log(tab_x))^2*tab_x^b*(tab_y-a*tab_x^b-a*tab_y*tab_x^b),na.rm=TRUE)
    J <- matrix(data=c(J11,J12,J21,J22),nrow=2,ncol=2,byrow=TRUE)
    return(J)
  }
  
  sum_of_squares <- function(coef,tab_x,tab_y){
    a <- coef[1]
    b <- coef[2]
    sum <- sum((tab_y-a*tab_x^b)^2,na.rm=TRUE)
    return(sum)
  }
  
  ## Input:
  ##	tab_x = 1st vector
  ##	tab_y = 2nd vector
  ## Output:

  ## Author: J.Jaffrain and M.Schleiss, June 30th 2009

  # Initial conditions from linear fit on logs
  log_tab_x <- log(tab_x)
  log_tab_y <- log(tab_y)
  LM <- lm(log_tab_y~1+log_tab_x)	# fit a linear model on logs.
  intercept <- LM$coefficients[1]
  slope <- LM$coefficients[2]
  a0 <- exp(intercept)
  b0 <- slope
  coef <- matrix(data=c(a0,b0),nrow=2,ncol=1)
#   n_steps <- 1000
  converge <- TRUE
  
  for (i in 1:n_steps){
#   print(coef)
    sum <- sum_of_squares(coef,tab_x,tab_y)
    df <- gradient(coef,tab_x,tab_y)
    J <- jacobien(coef,tab_x,tab_y)
    DetJ <- J[1,1]*J[2,2]-J[2,1]*J[1,2]
#     print(DetJ)
    if(is.na(DetJ)){
      stop("NA value in DetJ")
    }else{
      if(abs(DetJ) < 10^(-6)){
	if(DetJ == 0){
	  print("DetJ too close to zero")
	  converge <- FALSE
	  break
	}else{		## Switch to gradient method	
	  coef_new <- coef-DetJ*df
	  sum_new <- sum_of_squares(coef_new,tab_x,tab_y)
	  while(sum_new < sum){
	    coef <- coef_new
	    sum <- sum_of_squares(coef,tab_x,tab_y)
	    DetJ <- DetJ*2
	    coef_new <- coef-DetJ*df
	    sum_new <- sum_of_squares(coef_new,tab_x,tab_y)
	  }
	}
      }else{
# 	inv_J <- solve(J)
	inv_J <- 1/DetJ*matrix(data=c(J[2,2],-J[1,2],-J[2,1],J[1,1]),nrow=2,ncol=2,byrow=TRUE)
# 	print(inv_J)
# 	print(df)
	coef_new <- coef-(inv_J%*%df)
	sum_new <- sum_of_squares(coef_new,tab_x,tab_y)
	if(sum_new < sum){
	  coef <- coef_new
	}else{		## Switch to gradient method
# 	  print("Sum of squares has increased")
	  coef_new <- coef-DetJ*df
	  sum_new <- sum_of_squares(coef_new,tab_x,tab_y)
	  while(sum_new < sum){
	    coef <- coef_new
	    sum <- sum_of_squares(coef,tab_x,tab_y)
	    DetJ <- DetJ*2
	    coef_new <- coef-DetJ*df
	    sum_new <- sum_of_squares(coef_new,tab_x,tab_y)
	  }
	}
	if(sum(is.na(coef)) > 0){
	  stop("NA value in coef")
	}
      }
    }
  }
  ret <- list(coef,converge,i)
  names(ret) <- c("coef","converge","n_iterations")
  return(ret)
}

##########################################################################

robust_power_law.fit <- function(tabX,tabY,seqa,seqb,cutoff=0){

    ## Fits a robust power law given by Y = a*X^b
    
    ## Inputs:
    ## tabX   = vector of explanatory variables
    ## tabY   = vector of response variables
    ## seqa   = sequence of prefactors to test
    ## seqb   = sequence of exponents to test
    ## cutoff = the percentage of higher residual values that won't be taken into account (default=0%)

    ## Output:
    ## list_output = a list with different outputs:
    ## 1st element = a 1x2 vector with best estimated (a,b) values
    ## 2nd element = a vector with the estimated values of Y
    
    if(cutoff >= 0.5){
      stop("'cutoff' parameter might be lower than 0.5 (=50%) !")
    }

    range_a <- diff(range(seqa))
    if(range_a == 0){stop("Empty seqa")}
    da <- max(diff(seqa))

    Nres  <- round((1-cutoff)*length(tabY))
# print(paste("Nres=",Nres))
    if(Nres < 1){print("Warning: not enough values for a proper rounding, 'cutoff' was set up to remove 1 value !")}
    Nseqa <- length(seqa)
    Nseqb <- length(seqb)
    SS <- matrix(NA,nrow=Nseqa,ncol=Nseqb)	# Matrix with the residual values 
    for(i in 1:Nseqb){
	b       <- seqb[i]
	for(j in 1:Nseqa){
	    a       <- seqa[j]
	    Yhat    <- a*tabX^b
	    res2 	<- (tabY-Yhat)^2
	    nw  <- sort(res2)
	    new_res2  <- nw[1:Nres]
	    SS[j,i]   <- sum(new_res2,na.rm=TRUE)
	}
    }
    pos <- which.min(SS)

    ind_a <- pos %% Nseqa
    if(ind_a == 0){ind_a <- Nseqa}
    ind_b <- floor((pos-1)/Nseqa)+1
    best_a <- seqa[ind_a]
    best_b <- seqb[ind_b]
    min_a <- max(best_a-range_a/4,0)
    max_a <- (best_a+range_a/4)

    if(((da/2)>1) && (max_a > min_a)){		## rounding 'a' parameter to the integer (Joel)
	new_seqa <- seq(min_a,max_a,da/2)
	list_output <- robust_power_law.fit(tabX,tabY,new_seqa,seqb,cutoff=cutoff)
	best_a <- list_output[[1]][1]
	best_b <- list_output[[1]][2]
    }

    Yhat   <- best_a*tabX^best_b
    
    list_output <- vector("list",2)
    list_output[[1]] <- c(best_a,best_b)
    list_output[[2]] <- Yhat
    return(list_output)
}

#####################################################################################

fit <- function(tabX,tabY,seqa,seqb,cutoff=0){
    ## delete NA values
    i1 <- which(!is.na(tabX))
    i2 <- which(!is.na(tabY))
    i3 <- intersect(i1,i2)
    if(length(i3)==0){stop("only NA values")}
    tabX <- tabX[i3]
    tabY <- tabY[i3]
    ## delete zero values
    i1 <- which(tabX >= 1e-6)
    i2 <- which(tabY >= 1e-6)
    i3 <- intersect(i1,i2)
    if(length(i3)==0){stop("only zero values")}
    tabX <- tabX[i3]
    tabY <- tabY[i3]
    ## compute initial estimate in log space
    logY <- log10(tabY)
    logX <- log10(tabX)
    LM   <- lm(logY~logX)
    a    <- 10^(LM$coefficients[1])
    b    <- LM$coefficients[2]
    ## use a and b to identify outliers
    N <- length(tabX)
    if(N <= 2){stop("length of tabX <= 2")}
    residuals <- (tabY-a*tabX^b)^2
    sorted_residuals <- sort(residuals)
    i1 <- which(residuals==sorted_residuals[N])
    i2 <- which(residuals==sorted_residuals[N-1])
    ## remove outliers
    tabX <- tabX[setdiff(1:N,c(i1,i2))]
    tabY <- tabY[setdiff(1:N,c(i1,i2))]
    ## compute initial estimate in log space (without outliers)
    logY <- log10(tabY)
    logX <- log10(tabX)
    LM   <- lm(logY~logX)
    a    <- 10^(LM$coefficients[1])
    b    <- LM$coefficients[2]
#     seqa <- seq(max(a-200,200),min(a+200,1000),3.2)
#    seqa <- seq(max(a-200,0),a+200,2)		## Joel, 2010-02-09
    seqa <- seq(0.5*a,2*a,2) ## Alexis, 2010-04-22
    ## adjust a and b on new data
    list_return <- robust_power_law.fit(tabX,tabY,seqa,seqb,cutoff=cutoff)
    return(list_return) 
#     list_output <- vector("list",2)
#     list_output[[1]] <- c(a,b)
#     list_output[[2]] <- a*tabX^b
#     return(list_output)
} 


########################## Isotropic variogram of a 2D Field ##########################

isotropic_variogram <- function(M,dr,dmax){

    ## Computes the isotropic variogram of a 2D field
    
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
    ## For large data sets (more than 1000 observations) this can take 
    ## a significant amount of time !
    ## Source : M.Schleiss, March 2009, 2nd version

    tabd   <- seq(dr,dmax,dr)
    Nd     <- length(tabd)
    Npts   <- dim(M)[1]
    dmax   <- dmax+dr/2

    N   <- rep(0,Nd)
    SSD <- rep(0,Nd)

    for(i in 1:(Npts-1)){
	for(j in (i+1):Npts){
	    dx <- M[i,1]-M[j,1]
	    dy <- M[i,2]-M[j,2]
	    d  <- sqrt(dx^2+dy^2)
	    if(d<dmax){
		ssd <- (M[i,3]-M[j,3])^2
		if(!is.na(ssd)){
		    index <- which.min(abs(tabd-d))
		    N[index] <- N[index]+1
		    SSD[index] <- SSD[index]+ssd
		}
	    }
	}
    }

    V <- matrix(NA,nrow=Nd,ncol=4)
    for(i in 1:Nd){
	d <- tabd[i]
	n <- N[i]
	ssd <- SSD[i]
	gam <- NA
	if(n>0){gam <- ssd/(2*n)}
	V[i,] <- c(d,n,ssd,gam)
    }

    return(V)
}

########################## Isotropic variogram of a 3D Field ##########################

isotropic_variogram_3D <- function(M,dr,dmax){

    ## Computes the isotropic variogram of a 3D field
    
    ## Inputs :
    ##  M     = a matrix with N rows and 4 columns
    ##          1st column : the x coordinate
    ##          2nd column : the y coordinate
    ##          3rd column : the z coordinate
    ##          4th column : the value of the field variable
    ##  dr    = the width of the distance classes in the variogram
    ##  dmax  = the maximum distance do be considered
    
    ## Output :
    ##   V   = a variogram matrix with Nv=dmax/dr rows and 4 columns
    ##       1st column : the distance
    ##       2nd column : the number of points N(i) in the class
    ##       3rd column : the Sum of Squared Differences SSD
    ##       4th column : the semi-variance gamma = SSD/2N(i)
    
    ## Complexity : O(N^2) where N is the number of observations
    ## For large data sets (more than 1000 observations) this can take 
    ## a significant amount of time !
    ## Source : M.Schleiss, March 2009, 2nd version

    tab_index <- which(!is.na(rowSums(M)))
    Ntab_index <- length(tab_index)
    if(Ntab_index==0){stop("M contains only NA values")}
    M    <- M[tab_index,]
    Npts <- dim(M)[1]
    dmax <- dmax+dr/2

    tabd <- seq(dr,dmax,dr)
    Nd   <- length(tabd)
    N    <- rep(0,Nd)
    SSD  <- rep(0,Nd)

    for(i in 1:(Npts-1)){
	for(j in (i+1):Npts){
	    dx <- M[i,1]-M[j,1]
	    dy <- M[i,2]-M[j,2]
	    dz <- M[i,3]-M[j,3]
	    d  <- sqrt(dx^2+dy^2+dz^2)
	    if(d<dmax){
		ssd <- (M[i,4]-M[j,4])^2
		index <- which.min(abs(tabd-d))
		N[index] <- N[index]+1
		SSD[index] <- SSD[index]+ssd
	    }
	}
    }

    V <- matrix(NA,nrow=Nd,ncol=4)
    for(i in 1:Nd){
	V[i,1] <- tabd[i]
	V[i,2] <- N[i]
	V[i,3] <- SSD[i]
	if(N[i]>0){V[i,4] <- SSD[i]/(2*N[i])}
    }
    return(V)
}

################### Robust isotropic variogram of a 2D Field ################### 

robust_1D_variogram <- function(tabZ,tabT,minh,cutoff,dh){

    ## Computes a robust sample variogram on a 1D profile (time or space)
    ## Reference: "A moving window semivariance estimator",
    ##            Li & Lake, WRR, vol.30, no.5, pp.1479-1489, 1994
    
    ## Inputs:
    ## tabZ   = vector of observations (NA values are allowed)
    ## tabT   = vector of time stamps or distances
    ## minh   = minimum displacement distance
    ## cutoff = maximum displacement distance
    ## dh     = displacement step 
        
    ## Output:
    ## V = Variogram matrix with 3 columns (h|gamma1|gamma)
    
    ## Author: M.Schleiss, November 2009
    
    NtabZ <- length(tabZ)
    NtabT <- length(tabT)
    if(NtabZ!=NtabT){stop("tabZ and tabT must be of same size")}
    
    notNA <- which(!is.na(tabZ))
    if(length(notNA)==0){stop("only NA values in tabZ")}
    tabZ  <- tabZ[notNA]
    tabT  <- tabT[notNA]
    NtabZ <- length(tabZ)
    NtabT <- length(tabT)
    tabH  <- seq(minh,cutoff,dh)
    NtabH <- length(tabH)
    V     <- matrix(NA,nrow=NtabH,ncol=3)
    
    for(itr in 1:NtabH){
	h      <- tabH[itr]
	Ibegin <- 1
	Iend   <- 1
	Sum    <- 0
	for(i in 1:NtabZ){
	    t <- tabT[i]
	    while(tabT[Ibegin]+h<t && Ibegin<NtabZ){Ibegin <- Ibegin+1}	    
	    if(Iend<NtabZ){
		while(tabT[Iend+1]-h<t){
		    Iend <- Iend+1
		    if(Iend==NtabZ){break}
		}
	    }
	    if(Ibegin<Iend){
		id    <- setdiff(Ibegin:Iend,c(i))
		subZ  <- tabZ[id]
		diffZ <- subZ-tabZ[i]
		m     <- sum(!is.na(diffZ))
		if(m>0){
		    newSum <- sum(diffZ^2,na.rm=TRUE)/(2*m)
		    Sum    <- Sum+newSum
		}
	    }
	}
	V[itr,1] <- h
	V[itr,2] <- Sum/NtabZ
    }
    for(itr in 1:(NtabH-1)){
	h        <- V[itr,1]
	gam1     <- V[itr,2]
	gam2     <- V[itr+1,2]
	dgam     <- (gam2-gam1)/dh
	V[itr,3] <- V[itr,2]+h*dgam
    }
    V[NtabH,3] <- V[NtabH,2]
    return(V)
}

########################### 2D Variogram Map #############################

TwoD_variogram <- function(M,dx,dy,xmax,ymax){

    ## Computes the 2D variogram map of a random field

    ## Inputs:
    ## M = data matrix with 3 columns
    ##     1st column: the x coordinates
    ##     2nd column: the y coordinates
    ##     3rd column: the measured value 
    ## dx = class width on x-axis
    ## dy = class width on y-axis
    ## xmax = cutoff distance on x-axis
    ## ymax = cutoff distance on y-axis

    ## Output:
    ## V = variogram matrix with 5 columns
    ##     1st column: the displacement on the x-axis
    ##     2nd column: the displacement on the y-axis
    ##     3rd column: the number of points N(i)
    ##     4th column: the Sum of Squared Differences SSD
    ##     5th column: the semivariance gamma=SSD/2N(i)

    ## Author: M.Schleiss, EPFL-LTE 2009

    tabx <- seq(-xmax,xmax,dx)
    taby <- seq(-ymax,ymax,dy)
    Nx   <- length(tabx)
    Ny   <- length(taby)
    npts <- dim(M)[1]

    limx <- xmax+dx/2
    limy <- ymax+dy/2

    N   <- matrix(0,nrow=Nx,ncol=Ny)
    SSD <- matrix(0,nrow=Nx,ncol=Ny)

    for(i in 1:(npts-1)){
	for(j in (i+1):npts){
	    vx  <- M[i,1]-M[j,1]
	    if(vx<limx && vx>=(-limx)){
		vy  <- M[i,2]-M[j,2]
		if(vy<limy && vy>=(-limy)){
		    ssd <- (M[i,3]-M[j,3])^2
		    if(!is.na(ssd)){
			clx <- which.min(abs(tabx-vx))
			cly <- which.min(abs(taby-vy))
			N[clx,cly] <- N[clx,cly]+1
			SSD[clx,cly] <- SSD[clx,cly]+ssd
		    }
		}
	    }
	}
    }

    V <- matrix(NA,nrow=Nx*Ny,ncol=5)
    index <- 0
    for(i in 1:Nx){
	x <- tabx[i]
	for(j in 1:Ny){
	    index <- index+1
	    y <- taby[j]
	    V[index,1] <- x
	    V[index,2] <- y
	    V[index,3] <- N[i,j]
	    V[index,4] <- SSD[i,j]
	    if(!is.na(N[i,j])){
		if(N[i,j]>0){
		    V[index,5] <- SSD[i,j]/(2*N[i,j])
		}
	    }
	}
    }
    return(V)
}

########################### 3D Variogram Map #############################

ThreeD_variogram <- function(M,seqx,seqy,seqz){

    ## Computes a 3D Variogram

    ## Inputs:
    ## M  = data matrix (x|y|z|val)
    ## seqx = discretization table for x-axis
    ## seqy = discretization table for y-axis
    ## seqz = discretization table for z-axis

    ## Output:
    ## V = Variogram matrix (hx|hy|hz|npts|SSD|gamma)

    Nx  <- length(seqx)
    Ny  <- length(seqy)
    Nz  <- length(seqz)
    V   <- matrix(NA,nrow=Nx*Ny*Nz,ncol=6)
    itr <- 0
    for(i in 1:Nx){
	for(j in 1:Ny){
	    for(k in 1:Nz){
		itr <- itr+1
		V[itr,] <- c(seqx[i],seqy[j],seqz[k],0,0,NA)
	    }
	}
    }

    not_NA <- which(!is.na(M[,4]))
    NNA    <- length(not_NA)
    if(NNA>1){
	for(itr1 in 1:(NNA-1)){
	    i <- not_NA[itr1]
	    for(itr2 in (itr1+1):NNA){
		j  <- not_NA[itr2]
		hx <- abs(M[j,1]-M[i,1])
		hy <- abs(M[j,2]-M[i,2])
		hz <- abs(M[j,3]-M[i,3])
		ix <- which.min(abs(seqx-hx))
		iy <- which.min(abs(seqy-hy))
		iz <- which.min(abs(seqz-hz))
		id <- (ix-1)*Ny*Nz+(iy-1)*Nz+iz
		V[id,4] <- V[id,4]+1
		V[id,5] <- V[id,5]+(M[j,4]-M[i,4])^2
		V[id,6] <- V[id,5]/(2*V[id,4])
	    }
	}
    }
    return(V)
}

##################### TRANSITION PROBABILITY IN TIME #####################

transition_prob <- function(tabX,tabT,dt,dmax){

    ## Computes the transition probabilities for a time series 
    
    ## Inputs:
    ## tabX = vector of real values
    ## tabT = timetable for tabX 
    ## dt   = time lag resolution
    ## dmax = maximum time lag to be considered

    ## Outputs:
    ## M = matrix with 7 columns: lag|n00|n01|n10|n11|p00|p11
    ##   lag = time lag
    ##   n00 = number points such that x(t)=0 and x(t+lag)=0
    ##   n01 = number points such that x(t)=0 and x(t+lag)=1
    ##   n10 = number points such that x(t)=1 and x(t+lag)=0
    ##   n11 = number points such that x(t)=1 and x(t+lag)=1
    ##   p00 = P[X(t)=0|X(t-lag)=0]
    ##   p11 = P[X(t)>0|X(t-lag)>0]

    ## Preliminary operations (1) Check adequacy of input
    if(is.na(dt)){stop("NA value not allowed in dt")}
    if(is.na(dmax)){stop("NA value not allowed in dmax")}
    if(dt<=0){stop("dt must be positive")}
    if(dmax<dt){stop("dmax must be larger or equal to dt")}
    if(sum(is.na(tabT))>0){stop("NA values not allowed in tabT")}
    NtabX <- length(tabX)
    NtabT <- length(tabT)
    if(NtabX==0){stop("tabX is of length 0")}
    if(NtabX!=NtabT){stop("tabX and tabT must have same length")}

    ## Preliminary operations (2) Delete NA values in tabX
    index  <- which(!is.na(tabX))
    NtabX  <- length(index)
    if(NtabX==0){stop("tabX contains only NA values")}
    NtabT <- NtabX
    tabX  <- tabX[index]
    tabT  <- tabT[index]

    ## Main program starts here
    seqT  <- seq(dt,dmax,dt)
    NseqT <- length(seqT)
    tabX  <- tabX*c(tabX>0)
    M <- matrix(0,nrow=NseqT,ncol=7)
    M[,1] <- seqT
    for(i in 1:(NtabX-1)){
	t1 <- tabT[i]
	x1 <- tabX[i]
	for(j in (i+1):NtabX){
	    t2 <- tabT[j]
	    x2 <- tabX[j]
	    lag <- t2-t1
	    if(lag>(dmax+dt/2)){next}
	    k <- which.min(abs(seqT-lag))
	    if(x1==0){
		if(x2==0){M[k,2] <- M[k,2]+1}
		else{M[k,3] <- M[k,3]+1}
	    }
	    else{
		if(x2==0){M[k,4] <- M[k,4]+1}
		else{M[k,5] <- M[k,5]+1}
	    }
	}
    }
    for(i in 1:NseqT){
	n00 <- M[i,2]
	n01 <- M[i,3]
	n10 <- M[i,4]
	n11 <- M[i,5]
	n0  <- n00+n01
	n1  <- n10+n11
	if(n0>0){M[i,6] <- n00/n0}
	if(n0==0){M[i,6] <- NA}
	if(n1>0){M[i,7] <- n11/n1}
	if(n1==0){M[i,7] <- NA}
    }
    return(M)
}

#################### TRANSITION PROBABILITY MAPS (2D and 3D) ####################

transition_prob_map <- function(M,dlag,dmax){

    ## Computes the isotropic transition probability of a 3D field
    
    ## Inputs:
    ## M = data matrix with 4 columns: x|y|z|value
    ## dlag = time lag resolution
    ## dmax = maximum time lag to be considered

    ## Outputs:
    ## TransM = matrix with 7 columns: lag|n00|n01|n10|n11|p00|p11
    ##   lag = distance lag
    ##   n00 = number points such that x(h)=0 and x(h+lag)=0
    ##   n01 = number points such that x(h)=0 and x(h+lag)=1
    ##   n10 = number points such that x(h)=1 and x(h+lag)=0
    ##   n11 = number points such that x(h)=1 and x(h+lag)=1
    ##   p00 = P[X(h)=0|X(h-lag)=0]
    ##   p11 = P[X(h)>0|X(h-lag)>0]

    ## Preliminary operations (1) Check adequacy of input
    if(is.na(dlag)){stop("NA value not allowed in dlag")}
    if(is.na(dmax)){stop("NA value not allowed in dmax")}
    if(dlag<=0){stop("dlag must be positive")}
    if(dmax<dlag){stop("dmax must be larger or equal to dlag")}
    if(is.na(sum(colSums(M[,1:3])))){stop("NA values not allowed in coordinates")}

    ## Preliminary operations (2) Delete NA values in M
    index  <- which(!is.na(M[,4]))
    dimM   <- length(index)
    if(dimM==0){stop("M contains only NA values")}
    M <- M[index,]

    ## Main program starts here
    seq_lag <- seq(dlag,dmax,dlag)
    Nlag    <- length(seq_lag)
    M[,4]   <- 1*c(M[,4]>0)
    transM  <- matrix(0,nrow=Nlag,ncol=7)
    transM[,1] <- seq_lag
    for(i in 1:(dimM-1)){
	val1 <- M[i,4]
	for(j in (i+1):dimM){
	    val2 <- M[j,4]
	    lag <- sqrt(sum((M[i,1:3]-M[j,1:3])^2))
	    if(lag>(dmax+dlag/2)){next}
	    k <- which.min(abs(seq_lag-lag))
	    if(val1==0){
		if(val2==0){transM[k,2] <- transM[k,2]+1}
		else{transM[k,3] <- transM[k,3]+1}
	    }
	    else{
		if(val2==0){transM[k,4] <- transM[k,4]+1}
		else{transM[k,5] <- transM[k,5]+1}
	    }
	}
    }
    for(i in 1:Nlag){
	n00 <- transM[i,2]
	n01 <- transM[i,3]
	n10 <- transM[i,4]
	n11 <- transM[i,5]
	n0  <- n00+n01
	n1  <- n10+n11
	if(n0>0){transM[i,6] <- n00/n0}
	if(n0==0){transM[i,6] <- NA}
	if(n1>0){transM[i,7] <- n11/n1}
	if(n1==0){transM[i,7] <- NA}
    }
    return(transM)
}

#################### TRANSITION PROBABILITY MAPS (3D) #################### 

transition_prob3D <- function(M,seqx,seqy,seqz,npts){

    ## Computes the transition probabilities of a random binary 3D field
    ## p10 = P[ Z(x+h)=1 | Z(x)=0 ] 
    ## p11 = P[ Z(x+h)=1 | Z(x)=1 ]
    ## where h is a vector in space (hx|hy|hz)

    ## Inputs:
    ## M = data matrix with 4 columns (x|y|z|value)
    ## seqx = displacement on x-axis. Must be of the form -kx*dx,...,0,...,kx*dx
    ## seqy = displacement on y-axis. Must be of the form -ky*dy,...,0,...,ky*dy
    ## seqz = displacement on z-axis. Must be of the form -kz*dz,...,0,...,kz*dz
    ## npts = minimum number of points per class
    
    ## Output:
    ## Tmatrix = transition matrix with 4 columns (hx|hy|hz|p10|p11)

    ## Author: M.Schleiss, EPFL-LTE 2010
    ## Remarks: Complexity is O(N^2) N being the number of rows of M

    dimM <- dim(M)[1]
    Nx <- length(seqx)
    Ny <- length(seqy)
    Nz <- length(seqz)
    dx <- seqx[2]-seqx[1]
    dy <- seqy[2]-seqy[1]
    dz <- seqz[2]-seqz[1]
    kx <- (Nx-1)/2
    ky <- (Ny-1)/2
    kz <- (Nz-1)/2

    N <- matrix(0,nrow=Nx*Ny*Nz,ncol=4) ## (n0|n1|n10|n11|)
    indexI <- which(!is.na(M[,4]))
    if(length(indexI)>0){
	for(i in indexI){
	    zi <- M[i,4]
	    indexJ <- setdiff(indexI,i)
	    for(j in indexJ){
		zj <- M[j,4]
		ix <- round(1+kx+(M[j,1]-M[i,1])/dx)
		iy <- round(1+ky+(M[j,2]-M[i,2])/dy)
		iz <- round(1+kz+(M[j,3]-M[i,3])/dz)
		if(ix>0 && iy>0 && iz>0 && ix<=Nx && iy<=Ny && iz<=Nz){
		    id <- (ix-1)*Ny*Nz+(iy-1)*Nz+iz
		    if(zi==0){
			N[id,1] <- N[id,1]+1
			if(zj==1){N[id,3] <- N[id,3]+1}
		    }
		    else{
			N[id,2] <- N[id,2]+1
			if(zj==1){N[id,4] <- N[id,4]+1}
		    }
		}
	    }
	}
    }

    Tmatrix <- matrix(NA,nrow=Nx*Ny*Nz,ncol=5)
    colnames(Tmatrix) <- c("x","y","z","p10","p11")
    itr <- 0
    for(i in 1:Nx){
	x <- seqx[i]
	for(j in 1:Ny){
	    y <- seqy[j]
	    for(k in 1:Nz){
		itr <- itr+1
		z <- seqz[k]
		Tmatrix[itr,1:3] <- c(x,y,z)
		if(N[itr,1]>=npts){Tmatrix[itr,4] <- N[itr,3]/N[itr,1]}
		if(N[itr,2]>=npts){Tmatrix[itr,5] <- N[itr,4]/N[itr,2]}
	    }
	}
    }
    return(Tmatrix)
}

##########################################################################

interpolate <- function(data1,data2,k=2){

    ## Function for inverse distance interpolation
    ## By Marc Schleiss, 28.11.2008

    ## Inputs : 
    ## id    = a string specifying the name of the variable you want to interpolate
    ## data1 = matrix containing the measurements (x|y|val)
    ## data2 = matrix containing the grid points where you want to interpolate (x|y)
    ## k     = the type of norm (2 = usual euclidean distance)

    if(k<=0){stop("invalid norm parameter")}

    size_data1 <- dim(data1)[1]
    size_data2 <- dim(data2)[1]

    coord1 <- data1[,1:2]
    coord2 <- data2[,1:2]

    V <- data1[,3]
    tab_interpolation <- rep(NA,size_data2)

    for(i in 1:size_data2){
	    x <- coord2[i,1]
	    y <- coord2[i,2]
	    tab_distance <- sqrt((coord1[,1]-x)^2+(coord1[,2]-y)^2)
	    tab_interpolation[i] <- sum(V*tab_distance^(-k)/sum(tab_distance^(-k)))
	    
    }
    interpolation <- data.frame(coord2[,1],coord2[,2],tab_interpolation)
    names(interpolation)  <- c("x","y","val")
    return(interpolation)	
}

##########################################################################

bivariate_anamorphosis <- function(data){

    ## Performs a bivariate Gaussian anamorphosis
    ## Reference : "Stepwise Conditional Transformation for Simulation of Multiple Variables"
    ##             by Leuangthong and Deutsch, Mathematical Geology Vol.35 No.2, 2003

    ## Input:
    ## data = matrix with initial data (x|y)

    ## Output:
    ## new_data = matrix with transformed data (x|y)

    ## Remarks
    ## (1) Requires the "GenKern" package to be loaded
    
    tabX <- data[,1]
    tabY <- data[,2]
    Nx   <- length(tabX)
    i1   <- which(!is.na(tabX))
    i2   <- which(!is.na(tabY))
    id   <- intersect(i1,i2)
    N    <- length(id)

    ## Kernel estimate of bivariate density
    kern <- KernSur(tabX[id],tabY[id])
    seqx <- kern$xords
    seqy <- kern$yords
    density <- kern$zden

    tabX_new <- rep(NA,Nx)
    tabY_new <- rep(NA,Nx)

    ## Go iteratively through data
    ## transform x using normal score transform
    ## transform y conditionaly on x.
    for(i in 1:Nx){
	x <- tabX[i]
	y <- tabY[i]
	if(!is.na(x*y)){
	    xnew <- qnorm((sum(tabX<=x,na.rm=TRUE)-0.5)/N)
	    i1   <- which.min(abs(seqx-x))
	    i2   <- which(seqy<=y)
	    N2   <- length(i2)
	    if(N2>0){p1 <- sum(density[i1,i2],na.rm=TRUE)}
	    if(N2==0){p1 <- 1/(2*N)}
	    p2 <- sum(density[i1,],na.rm=TRUE)
	    if(p2==0){p <- (2*N-1)/2*N}
	    if(p2>0){p <- p1/p2}
	    ynew <- qnorm(p)
	    tabX_new[i] <- xnew
	    tabY_new[i] <- ynew
	}
    }
    tabX_new <- (tabX_new-mean(tabX_new,na.rm=TRUE))/sd(tabX_new,na.rm=TRUE)
    tabY_new <- (tabY_new-mean(tabY_new,na.rm=TRUE))/sd(tabY_new,na.rm=TRUE)
    new_data <- matrix(c(tabX_new,tabY_new),nrow=Nx,ncol=2,byrow=FALSE)
    return(new_data)
}

##########################################################################

independence_test <- function(R,n,confidence){

    ## Performs a test of independence for bivariate gaussian variables X1,...,XN and Y1,...YN

    ## Inputs: R = the sample correlation between X and Y
    ##         n = the size of the sample
    ##         confidence = the confidence level of the test

    ## Output: reject = logical: if TRUE the hypothesis is rejected

    reject          <- FALSE
    test_value      <- sqrt(n-2)*abs(R)/sqrt(1-R^2)
    rejection_value <- qt(confidence,df=n-2)
    if(test_value > rejection_value){reject<-TRUE}
    return(reject)
}

##########################################################################

find_next <- function(tabX,val,i){

    ## Returns the index of the next occurence of val in tabX, starting at i+1
    
    ## Inputs:
    ## tabX = a vector of values
    ## val  = a real value to find
    ## i    = the index from which to start in tabX
    
    ## Outputs:
    ## the index of the next occurence of val and NA otherwise.

    N <- length(tabX)
    if(i>=N){
	print("Error in find_next: index out of range")
	stop()
    }
    found <- FALSE
    while(i<N && found==FALSE){
	i <- i+1
	x <- tabX[i]
	if(!is.na(x)){
	    if(x==val){found <- TRUE}
	}
    }
    if(found==TRUE){return(i)}
    else{return(NA)}
}

##########################################################################

DFT <- function(tabX,maxlag=NA){

    ## Computes the Discrete Fourier Transform of tabX up to lag=maxlag
    ## NA values in tabX are allowed
    ## Values of tabX must be equally spaced
    
    ## Inputs:
    ## tabX   = vector of observations
    ## maxlag = maximum lag to be considered. If NA, all lags are considered
    
    ## Output:
    ## DFT = vector of Discrete Fourier Transforms (complex numbers) for lag=0,...,maxlag
    
    NtabX <- length(tabX)
    if(is.na(maxlag)){maxlag <- NtabX-1}
    DFT <- rep(NA,maxlag+1)
    j <- complex(real=0,imaginary=1)
    
    for(lag in 0:maxlag){
	Seq <- -2*pi*j*lag*seq(0:(NtabX-1))/NtabX
	DFT[lag+1] <- sum(tabX*exp(Seq),na.rm=TRUE)
    }
    
    return(DFT)
}

##########################################################################

sph <- function(d,n,s,r){
  if(d>r){return(s+n)}
  else{return(n+s*(3*d/(2*r)-d^3/(2*r^3)))}
}

