###################################### LIBRARY FOR THE FITTING AND MANIPULATION OF A GAMMA DSD #########################
################################################ Marc Schleiss, November 2008 ##########################################

## This is the shared "R" library for Gamma DSD manipulation
## Currently implemented functions are
## (1) low_D, up_D and mean_D for the Parsivel classes

###################### Gamma Log-Likelihood function ###################### 

logL <- function(b,tabN,tabD){

    ## Returns the log-likelihood (up to a constant) of a DSD sample with Gamma distribution

    ## Inputs :
    ## b = the parameters of the Gamma distribution (alpha,beta)
    ## tabN = the number of drops (per unit volume) in each diameter class
    ## tabD = the mean diameter classes (32 values for Parsivel)

    alpha <- b[1]
    beta  <- b[2]
        
    Sum <- sum(is.na(b))+sum(is.na(tabN))
    if(Sum>0){
	warning("warning: NA value in log-likelihood")
	logL <- -1e16
    }
    if(Sum==0){
    
	Nt <- sum(tabN)
	if(Nt==0){
	    warning("warning: Empty DSD in log-likelihood")
	    logL <- -1e16
	}
	if(Nt>0){
	    Sum <- c(alpha<0.01) + c(beta<0.01) + c(alpha>150) + c(beta>150)
	    if(Sum>0){
		warning("warning: Gamma parameters out of range in log-likelihood")
		logL <- -1e16
	    }
	    if(Sum==0){
		term1 <- alpha*sum(tabN*log(tabD))
		term2 <- beta*sum(tabN*tabD)
		term3 <- Nt*alpha*log(beta)
		term4 <- Nt*log(gamma(alpha))
		logL  <- term1-term2+term3-term4
		if(is.na(logL)){logL <- -1e16}
	    }
	}
    }
    return(logL)
}

######################## Gradient of Log-Likelihood ########################

DlogL <- function(b,tabN,tabD){

    ## Returns the derivatives of the Gamma log-likelihood function

    ## Inputs:
    ## b = vector of Gamma parameters (alpha,beta)
    ## tabN = the number of drops (per unit volume) in each diameter class
    ## tabD = the mean diameter classes (32 values for Parsivel)

    alpha <- b[1]
    beta  <- b[2]
    gradient <- c(0,0)

    Sum <- sum(is.na(b))+sum(is.na(tabN))
    if(Sum>0){warning("warning: NA value in Dlog-likelihood")}
    if(Sum==0){
	Nt <- sum(tabN)
	alpha <- max(alpha,0.1)
	beta  <- max(beta,0.1)
	alpha <- min(alpha,150)
	beta  <- min(beta,150)
	gradient1 <- sum(tabN*log(tabD))+Nt*log(beta)-Nt*digamma(alpha)
	gradient2 <- Nt*alpha/beta - sum(tabN*tabD)
	if(!is.na(gradient1*gradient2)){
	    gradient[1] <- gradient1
	    gradient[2] <- gradient2
	}
   }
   return(gradient)

}

############################## Inverse Hessian Matrix ##################################

invHlogL <- function(b,tab_N,tab_D){

   ## Inverse Hessian of the log-likelihood

   alpha <- b[1]
   beta  <- b[2]
   if(is.na(alpha) || is.na(beta)){
      Hinv <- matrix(c(1,0,0,1),nrow=2,ncol=2,byrow=TRUE)
   }
   else{
      alpha <- max(alpha,0.1)
      beta  <- max(beta,0.1)
      alpha <- min(alpha,150)
      beta  <- min(beta,150)
      Nt    <- sum(tab_N)
      Hinv  <- matrix(rep(0,4),nrow=2,ncol=2)
      Cste  <- -beta^2/(Nt*(alpha*trigamma(alpha)-1))
      Hinv[1,1] <- alpha/(beta^2)
      Hinv[1,2] <- 1/beta
      Hinv[2,1] <- 1/beta
      Hinv[2,2] <- trigamma(alpha)
      Hinv <- Cste*Hinv
   }
   return(Hinv)
}


############################## Estimation of Nr parameter ####################

Nr_fit <- function(tab_N,tab_D,x,Dmin=0.25,Dmax=7.0,dD=0.01){

    ## Fits the parameter of the gamma distribution : N(D) = Nr*f(D)*dD
    ## where f(D) is the density of a gamma distribution with shape=alpha and rate=beta
    ## The parameter Nr is fitted such that the theoretical and measured rain rates correspond.

    ## Inputs :
    ## tab_N = the number of drops (per unit volume) in each diameter class
    ## tab_D = the mean diameter classes (32 values for Parsivel)
    ## x     = the parameter vector for the Gamma distribution : 1st element = the shape (alpha) ; 2nd element = the rate (beta)

    ## Optional arguments 
    ## Dmax = maximum diameter size
    ## Dmin = minimum diameter size
    ## dD   = discretization of diameter

    ## Outputs :
    ## Nr = the Gamma DSD concentration

    ## Source : M.Schleiss, November 2008
    ## Complexity : O(N) where N is the size of the discretized diameter classes
    ## raindrop_velocity need the new_lib_DSD.R

    alpha <- x[1]
    beta  <- x[2]

    if(sum(is.na(tab_N))>0 || sum(is.na(tab_D)>0)){
	warning("NA values in DSD spectrum")
	Nr <- NA
    }
    else{
	if(length(tab_D)==0){
	    warning("diameter class vector is empty")
	    Nr <- NA
	}
	else{
	    tab_V <- raindrop_velocity(tab_D)
	    R_exp <- sum((tab_D^3)*tab_V*tab_N)
	    seqD  <- seq(Dmin,Dmax,dD)
	    seqV  <- raindrop_velocity(seqD)
	    fD    <- dgamma(seqD,shape=alpha,rate=beta)
	    R_theoretical <- dD*sum(seqD^3*seqV*fD)
	    Nr <- R_exp/R_theoretical	
	}
    }
    return(Nr)
}

####################################### Gamma Newton Estimates #######################

Newton_MLE_Gamma <- function(tab_N,tab_D,Steps=75){

    ## Computes the maximum likelihood estimates of a Gamma DSD given by N(D)=Nr*f(D)*dD
    ## The concentration Nr is derived in order to match the liquid water content of the recorded ground DSD.
    ## This version uses a Newton-Raphson algorithm to search for zeros in the log-likelihood.
    ## In the best case the convergence rate is quadratic
    ## The search is ended after a fixed amount of iterations, regardless of the convergence

    ## Inputs :
    ##   tab_N = the number of drops (per unit volume) in each diameter class
    ##   tab_D = the mean diameter classes (32 values for Parsivel)
    
    ## Optional Arguments :
    ##   Steps  = The number of iterations to be performed (by default 75)
    
    ## Outputs :
    ##   MLE_estimates = (alpha,beta,Nr) max.likelihood estimates 

    ## Remarks :
    ## Several cases may produce situations where the DSD parameters cannot be estimated
    ## (1) NA values in tab_N
    ## (2) No drops recorded on ground
    ## (3) Not enough different diameter classes for accurate estimation
    ## (4) Singular values in initial estimates
    ## (5) Singular values in Newton-Raphson inversion step
    ## In these cases, NA values are returned and a warning or error message is displayed
	    
    ## Source : M.Schleiss, November 2008
    ## Complexity : O(N) where N is the number of steps performed

    if(sum(is.na(tab_N))>0){
	MLE_estimates <- c(NA,NA,NA)
	warning("NA values in DSD spectrum")
    }
    else{
	Nt <- sum(tab_N)
	non_zero_values <- sum(tab_N>0)
	if(Nt==0 || non_zero_values<3){
	    MLE_estimates <- c(NA,NA,Nt)
	    warning("Intermittency detected in DSD spectra")
	}
	else{
	    ## Initial estimation with first and second order moments
	    M1 <- sum(tab_N*tab_D)/Nt
	    M2 <- sum(tab_N*tab_D^2)/Nt
	    if(abs(M2-M1^2)<1e-12){
		warning("Singular expression in initial estimation")
		MLE_estimates <- c(NA,NA,Nt)
	    }
	    else{
		beta0 <- M1/(M2-M1^2)
		alph0 <- beta0*M1
		x  <- c(alph0,beta0)
		## Applying the Newton-Raphson algorithm to maximize the log-likelihood
		for(i in 1:Steps){
		    fx <- logL(x,tab_N,tab_D)
		    gradient <- matrix(DlogL(x,tab_N,tab_D),ncol=1)
		    inverse_Hessian <- invHlogL(x,tab_N,tab_D)
		    x  <- x - inverse_Hessian%*%gradient 
		}
		alpha <- x[1]
		beta  <- x[2]
		Nr    <- Nr_fit(tab_N,tab_D,x)
		if(!is.na(alpha) && !is.na(beta)){
		    if(alpha<=0 || beta<=0){
			alpha <- NA
			beta  <- NA
		    }
		}
		MLE_estimates <- c(alpha,beta,Nr)
	    }	
	}
    }
    return(MLE_estimates)
}


############################### MLE estimation with bias correction due to censored sample ###############

MLE_bias_corrected <- function(tab_N,tab_D,alpha0,beta0,Npts=75,Censorized=0.25){

    ## Computes the max.likelihood estimation of a Gamma DSD that is Left Censored 
    ## This version includes a bias correction term in the likelihood equation 
    ## The additional term corrects for drops with diameter less than 0.25mm that cannot be recorded by Parsivel
    
    ## Inputs : 
    ##   tab_N = the number of drops (per unit volume) in each diameter class
    ##   tab_D = the mean diameter classes (32 classes for Parsivel)
    ##   alpha0 = initial estimate of alpha
    ##   beta0  = initial estimate of beta
    ##   Npts  = the number of interpolation points
    ##   Censorized = the minimum diameter value that can be recorded
    
    ## Outputs :
    ##  (alpha,beta,Nt) the max.likelihood estimates
    
    ## Source : M.Schleiss, October 2008
    ## Complexity : Exhaustive search in Parameter space : O(Npts^2)
	    
    best_alpha <- NA
    best_beta  <- NA
    best_Nt    <- NA
    best_score <- -1e12
    Npts       <- max(1,Npts)

    Nt <- sum(tab_N)
    if(is.na(Nt)==FALSE){
	if(Nt>0){
	    min_alpha <- max(0.1,alpha0/2)
	    min_beta  <- max(0.1,beta0/2)
	    max_alpha <- alpha0*2
	    max_beta  <- beta0*2
	    d_alpha   <- (max_alpha-min_alpha)/Npts
	    d_beta    <- (max_beta-min_beta)/Npts
	    tab_alpha <- seq(min_alpha,max_alpha,d_alpha)
	    tab_beta  <- seq(min_beta,max_beta,d_beta)
	    for(alpha in tab_alpha){
		for(beta in tab_beta){
		    fD <- dgamma(tab_D,shape=alpha,rate=beta)
		    g  <- fD/(1-pgamma(Censorized,shape=alpha,rate=beta))
		    L  <- sum(tab_N*log(g),na.rm=TRUE)
		    if(is.na(L)==FALSE){
			if(L>best_score){
			    best_score <- L
			    best_alpha <- alpha
			    best_beta  <- beta
			}
		    }
		}
	    }
	    best_Nt <- Nr_fit(tab_N,tab_D,c(best_alpha,best_beta))
	}		
    }
    return(c(best_alpha,best_beta,best_Nt))
}


####################################### Robust Gamma DSD Estimation ###################################

Robust_DSD_fit <- function(tabN,lowD,upD,Nt,tab_alpha,tab_ratio){
	
    ## Description:
    ##   Estimates the DSD parameters (alpha,beta,Nt) for Parsivel by taking 
    ##   into account the left-censoring and the diameter classification of the drops

    ## Inputs:
    ##   tabN = the number of drops (per unit volume) in each diameter class
    ##   lowD = lower limits of diameter classes (in mm)
    ##   upD  = upper limits of diameter classes (in mm)
    ##   Nt   = the concentration parameter of the DSD

    ## Output:
    ##   tabE = c(alpha,beta,Nt) the estimates of the Gamma parameters

    ## Remarks:
    ##   (1) : NA values or negative values are not allowed in tabN
    ##   (2) : The estimates are obtained by brute-forcing the score function (very slow !)
    ##   (3) : The estimate for alpha has been constraint to 0.2-60 with 0.02 resolution
    ##   (4) : The mean diameter has been constraint to 0.3-2mm with 0.02 resolution
    ## Author: M.Schleiss, April 4th 2009

    grid  <- expand.grid(tab_alpha,tab_ratio)
    Ngrid <- dim(grid)[1]
    tabS  <- rep(NA,Ngrid)
    for(i in 1:Ngrid){
	alpha <- grid[i,1]
	beta  <- alpha/grid[i,2]
	p1    <- pgamma(upD,shape=alpha,rate=beta)
	p2    <- pgamma(lowD,shape=alpha,rate=beta)
	tabP  <- p1-p2
	tabS[i] <- sum(abs(Nt*tabP[3:32]-tabN[3:32]),na.rm=TRUE)
    }
    iMin  <- which.min(tabS)
    alpha <- grid[iMin,1]
    beta  <- alpha/grid[iMin,2]
    tabE  <- c(alpha,beta,Nt)
    return(tabE)
}




####################################### Moment estimation using 3 different moments ###################

moment_estimation <- function(tab_N,tab_D,moments,tab_mu){

    ## Computes the Moment Estimates of a Gamma DSD N(D)=N0*D^mu*exp(-lambda*D)
    
    ## Inputs :
    ##   tab_N   = the number of drops (per unit volume) in each diameter class
    ##   tab_D   = the mean diameter classes (32 classes for Parsivel)
    ##   moments = (k,l,m) the 3 moments to use for the estimation.
    ##   tab_mu  = the values of mu to be computed 
    
    ## Output :
    ##   ME = The Moment Estimation of (alpha,beta,Nt)
    
    ## Remarks : 
    ## (1) Popular moments are : (0,1,2) ; (2,3,4) ; (2,4,6)
    ## (2) Significant bias can occur for higher moments in small samples with small drops.
    ## (3) One has the relations : mu=alpha-1 ; lambda = beta  

    best_mu     <- NA
    best_lambda <- NA
    best_delta  <- 1e12

    if(sum(tab_N)>0){
	k <- moments[1]
	l <- moments[2]
	m <- moments[3]

	ltD   <- length(tab_D)

	Mk <- sum(tab_N*tab_D^k)
	Ml <- sum(tab_N*tab_D^l)
	Mm <- sum(tab_N*tab_D^m)
	
	Moment_Product <- (Mm/Mk)^l * (Ml/Mm)^k * (Mk/Ml)^m
	
	Seq_lm <- (l+1):m
	Seq_kl <- (k+1):l

	## Estimating mu by finding root of function (exhaustive search)
	for(mu in tab_mu){
	    numerator   <- prod(Seq_lm+mu)^(l-k)
	    denominator <- prod(Seq_kl+mu)^(m-l)
	    delta <- abs(numerator/denominator - Moment_Product)
	    if(is.na(delta)==FALSE){
		if(delta<best_delta){
		    best_mu    <- mu
		    best_delta <- delta
		}   
	    }
	}

	## Estimating lambda
	best_lambda <- (Mk*gamma(best_mu+m+1)/(Mm*gamma(best_mu+k+1)))^(1/(m-k))
	
	## Estimating Nt
	Nt <- Nr_fit(tab_N,tab_D,c(best_mu+1,best_lambda))
    }
    
    ME <- c(best_mu+1,best_lambda,Nt)
    return(ME)

}

###########################################################

Moment_DSD <- function(tabN,tabD,M1,M2,seqmu,seqlam){

    Nmu  <- length(seqmu)
    Nlam <- length(seqlam)
    
    RefM1 <- sum(tabN*tabD^M1,na.rm=TRUE)/sum(tabN,na.rm=TRUE)
    RefM2 <- sum(tabN*tabD^M2,na.rm=TRUE)/sum(tabN,na.rm=TRUE)
    if(is.na(RefM1*RefM2)){return(c(NA,NA))}
    if(RefM1*RefM2==0){return(c(NA,NA))}

    Dmin <- 0.1
    Dmax <- 7.0
    dD   <- 0.01
    seqD  <- seq(0.1,7.0,0.01)
    NseqD <- length(seqD)

    best <- rep(NA,2)
    best_score <- 1e12

    for(mu in seqmu){
	for(lam in seqlam){
	    M1_hat <- dD*sum(seqD^(mu+M1)*exp(-lam*seqD),na.rm=TRUE)
	    M2_hat <- dD*sum(seqD^(mu+M2)*exp(-lam*seqD),na.rm=TRUE)
	    if(is.na(M1_hat*M2_hat)){next}
	    score <- abs(RefM1-M1_hat)/M1 + abs(RefM2-M2_hat)/M2
	    if(is.na(score)){next}
	    if(score<best_score){
		best_score <- score
		best <- c(mu,lam)
	    }
	}
    }
    return(best)

}

####################################### Rank estimation of DSD parameters for Parsivel #########################################

rank_estimation <- function(tab_Ng,Nt,dt,tab_alpha,tab_beta,nSim){

    ## Inputs :
    ## tab_Ng = Number of drops per diameter class on the GROUND !!!
    ## tab_D  = Diameter classes (32 for Parsivel)
    ## tab_V  = Mean velocity of diameter classes (32 for Parsivel)
    ## Nt     = Volume DSD concentration
    ## dt     = Time resolution
    ## S      = Surface of the disdrometer
    ## tab_alpha = values of alpha to be investigated
    ## tab_beta  = values of beta to be investigated
    ## nSim = number of simulations to be performed to compute the mean ranks
    ## Censorized = lower diameter censorization

    ## Outputs :
    ## tab_estimates <- c(alpha,beta,Nt) best estimation of parameters

    S    <- 54e-4
    Dmax <- 10.0
    Vmax <- drop_velocity(Dmax)/100
    Hmax <- Vmax*dt
    NtHS <- round(Nt*Hmax*S)

    velocity_table <- sapply(seq(0.01,Dmax,0.01),drop_velocity)/100
    
    best_alpha  <- NA
    best_beta   <- NA
    best_score  <- 1e15 

    for(alpha in tab_alpha){
	for(beta in tab_beta){
	    if(alpha/beta>6 || alpha/beta<0.15){next}
	    tab_N <- rep(0,32)
	    tab_random_D <- rgamma(NtHS*nSim,shape=alpha,rate=beta)
	    tab_random_H <- runif(NtHS*nSim,min=0,max=Hmax)
	    tab_maxH     <- dt*velocity_table[ceiling(tab_random_D/0.01)]
	    tab_recorded <- (tab_random_H<=tab_maxH)*tab_random_D
	    tab_recorded <- tab_recorded[which(tab_recorded>0)]
	    ltR <- length(tab_recorded)
	    for(i in 1:ltR){
		index <- sum(c(low_D<=tab_recorded[i])) 
		tab_N[index] <- tab_N[index]+1	
	    }
	    tab_N <- tab_N/nSim
	    tab_N[1:2] <- c(0,0)
	    score <- sum(abs(tab_Ng-tab_N))
	    if(!is.na(score)){
		if(score<best_score){
		    best_score <- score
		    best_alpha <- alpha
		    best_beta  <- beta
		}
	    }
	}
    }

    tab_estimates <- c(best_alpha,best_beta,Nt)
    return(tab_estimates)	
}


################################### KOLMOGOROV-SMIRNOV TEST ##############################

Kolmogorov <- function(X,alpha,beta,nmax=1000){
   
   ## Performs a Kolmogorov-Smirnov test on X
   ## Null hypothesis : the sample X comes from a Gamma distribution with given shape and rate parameter
   ## Alternative hypothesis : the sample X does not come from a Gamma distribution with these parameters

   ## Inputs :
   ##   X     = the sample vector
   ##   alpha = shape parameter of Gamma DSD
   ##   beta  = rate parameter of Gamma DSD

   ## Optional Arguments :
   ##   nmax = number of terms computed in infinite series. By default 1000

   ## Outputs :
   ##   pvalue = the p-value of the test (probability that the null hypothesis cannot be rejected)

   N   <- length(X)
   X   <- sort(X)
   Fx  <- pgamma(X,shape=alpha,rate=beta)
   Fnx <- (1:N)/N
   Dn  <- max(abs(Fnx-Fx))
   Kn  <- sqrt(N)*Dn 
   Seq <- 1:nmax   
   pvalue <- (sqrt(2*pi)/Kn)*sum(exp( (-1/8)*( pi*(2*Seq-1)/Kn )^2 ))
   return(pvalue)  
}


########################################## Power-Law Fit #########################################

power_law <- function(tabX,tabY,precision,intercept){

	## Fits a power-law of the type Y = b0 + b1*X^b2

	## Inputs :
	## tabX = the values of X
	## tabY = the values of Y
	## precision = the precision of the convergence (1e-4 or less is fine)
	## Intercept==TRUE fit the intercept b0 ; Intercept==FALSE the parameter b0 is equal to zero.

	## Outputs :
	## B = A vector containing the values of (b0,b1,b2) or (b1,b2) depending on Intercept==TRUE/FALSE

	## Source :
	## Marc Schleiss, 21.11.2008

	step <- 1
	increment <- precision+1

	if(intercept==TRUE){
		SS <- function(b0,b1,b2){return(sum((tabY-b0-b1*tabX^b2)^2))}
		dSS <- function(b0,b1,b2){
			dSS0 <- -2*sum(tabY-b0-b1*tabX^b2)
			dSS1 <- -2*sum((tabY-b0-b1*tabX^b2)*tabX^b2)
			dSS2 <- -2*sum((tabY-b0-b1*tabX^b2)*b1*log(tabX)*tabX^b2)
			return(c(dSS0,dSS1,dSS2))
		}
		## initial solution using log-linear regression without intercept
		b2 <- cov(log(tabY),log(tabX))/var(log(tabX))
		b1 <- exp(mean(log(tabY))-b2*mean(log(tabX)))
		b0 <- 0
		B   <- c(b0,b1,b2)
		SSB <- SS(b0,b1,b2)
		if(is.na(SSB)){SSB <- 1e16}
		## minimization of SS with respect to b0,b1,b2 (gradient method)
		while(increment>precision){	
			newB <- B - step*dSS(B[1],B[2],B[3])
			if(sum(is.na(newB))==0){
				newSSB <- SS(newB[1],newB[2],newB[3])
				if(!is.na(newSSB)){
					if(newSSB<SSB){
						increment <- sum(abs(B-newB))
						SSB <- newSSB
						B   <- newB
						step <- step*1.25
					}else{step <- step/2}	
				}else{step <- step/2} 
			}else{step <- step/2}
		}
		if(is.na(B) || is.na(SS(B[1],B[2],B[3]))){warning("Convergence problem in power law")}	
	}
	if(intercept==FALSE){
		SS <- function(b1,b2){return(sum((tabY-b1*tabX^b2)^2))}
		dSS <- function(b1,b2){
			dSS1 <- -2*sum((tabY-b1*tabX^b2)*tabX^b2)
			dSS2 <- -2*sum((tabY-b1*tabX^b2)*b1*log(tabX)*tabX^b2)
			return(c(dSS1,dSS2))
		}
		## initial solution using log-linear regression
		b2 <- cov(log(tabY),log(tabX))/var(log(tabX))
		b1 <- exp(mean(log(tabY))-b2*mean(log(tabX)))
		B   <- c(b1,b2)
		SSB <- SS(b1,b2)
		if(is.na(SSB)){SSB <- 1e16}
		## minimizing of SS with respect to b1,b2 (gradient method)
		while(increment>precision){	
			newB <- B - step*dSS(B[1],B[2])
			if(sum(is.na(newB))==0){
				newSSB <- SS(newB[1],newB[2])
				if(!is.na(newSSB)){
					if(newSSB<SSB){
						increment <- sum(abs(B-newB))
						SSB <- newSSB
						B   <- newB
						step <- step*1.25
					}else{step <- step/2}	
				}else{step <- step/2} 
			}else{step <- step/2}
		}
		if(is.na(B) || is.na(SS(B[1],B[2]))){warning("Convergence problem in power law")}	
	}
	return(c(B))
}
