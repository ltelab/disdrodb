## This is the statistical library for Weibull distributions

FWeibull <- function(x,alpha,beta){

    ## Cumulative distribution function of Weibull distribution
    
    ## Inputs:
    ## x     = value (x>0)
    ## alpha = shape parameter (alpha>0)
    ## beta  = scale parameter (beta>0)

    if(is.na(x)){stop("Error in FWeibull: NA value for x")}
    if(is.na(alpha)){stop("Error in FWeibull: NA value for alpha")}
    if(is.na(beta)){stop("Error in FWeibull: NA value for beta")}
    if(x<=0){stop("Error in FWeibull: x must be strictly positive")}
    if(alpha<=0){stop("Error in FWeibull: alpha must be strictly positive")}
    if(beta<=0){stop("Error in FWeibull: beta must be strictly positive")}
    return(1-exp(-(x/beta)^alpha))
}

fWeibull <- function(x,alpha,beta){

    ## Density function of Weibull distribution
    
    ## Inputs:
    ## x     = value (x>0)
    ## alpha = shape parameter (alpha>0)
    ## beta  = scale parameter (beta>0)
    
    if(is.na(x)){stop("Error in fWeibull: NA value for x")}
    if(is.na(alpha)){stop("Error in fWeibull: NA value for alpha")}
    if(is.na(beta)){stop("Error in fWeibull: NA value for beta")}
    if(x<=0){stop("Error in fWeibull: x must be strictly positive")}
    if(alpha<=0){stop("Error in fWeibull: alpha must be strictly positive")}
    if(beta<=0){stop("Error in fWeibull: beta must be strictly positive")}
    return(alpha*beta^(-alpha)*x^(alpha-1)*exp(-(x/beta)^alpha))
}

LogLWeibull <- function(tabX,alpha,beta){

    ## log-likelihood of i.i.d. Weibull distributed sample
    
    ## Inputs:
    ## tabX  = sample values
    ## alpha = shape parameter (alpha>0)
    ## beta  = scale parameter (beta>0)
    
    ## Output: logL = loglikelihood 
    
    if(is.na(alpha)){stop("Error in LogLWeibull: NA value for alpha")}
    if(is.na(beta)){stop("Error in LogLWeibull: NA value for beta")}
    if(alpha<=0){stop("Error in LogLWeibull: alpha must be strictly positive")}
    if(beta<=0){stop("Error in LogLWeibull: beta must be strictly positive")}   
    
    N <- length(tabX)
    if(N==0){stop("Error in LogLWeibull: tabX of size zero")}
    
    lna <- log(alpha)
    lnb <- log(beta)
    lX  <- log(tabX)
    Xa  <- tabX^(alpha)
    
    if(is.na(lna)){stop("Error in LogLWeibull: NA produced in ln(alpha)")}
    if(is.na(lnb)){stop("Error in LogLWeibull: NA produced in ln(beta)")}
    if(sum(is.na(lX))>0){stop("Error in LogLWeibull: NA produced in ln(tabX)")}
    if(sum(is.na(Xa))>0){stop("Error in LogLWeibull: NA produced in tabX^alpha")}
    
    logL <- N*lna+N*alpha*lnb+(alpha-1)*sum(lX)-beta^(-alpha)*sum(Xa)
    if(is.na(logL)){stop("Error in LogLWeibull: NA produced in final sum")}
    return(logL)
}

dLogLWeibull <- function(tabX,alpha,beta){

    ## Derivatives of log-likelihood
    
    ## Inputs:
    ## tabX  = sample values
    ## alpha = shape parameter (alpha>0)
    ## beta  = scale parameter (beta>0)
    
    ## Output: dlogL = c(dlogLa,dlogLb)  
    
    if(sum(is.na(tabX))>0){stop("Error in dLogLWeibull: NA value in tabX")}
    if(is.na(alpha)){stop("Error in dLogLWeibull: NA value for alpha")}
    if(is.na(beta)){stop("Error in dLogLWeibull: NA value for beta")}
    if(sum(tabX<=0)>0){stop("Error in dLogLWeibull: tabX values must be strictly positive")}
    if(alpha<=0){stop("Error in dLogLWeibull: alpha must be strictly positive")}
    if(beta<=0){stop("Error in dLogLWeibull: beta must be strictly positive")}   
    
    N <- length(tabX)
    if(N==0){stop("Error in dLogLWeibull: tabX of size zero")}   
    
    lnb <- log(beta)
    lX  <- log(tabX)
    Xa  <- tabX^(alpha)
  
    if(is.na(lnb)){stop("Error in dLogLWeibull: NA produced in ln(beta)")}
    if(sum(is.na(lX))>0){stop("Error in dLogLWeibull: NA produced in ln(tabX)")}
    if(sum(is.na(Xa))>0){stop("Error in dLogLWeibull: NA produced in tabX^alpha")}
    
    dLogLa <- N/alpha-N*lnb+sum(lX)+beta^(-alpha)*lnb*sum(Xa)-beta^(-alpha)*sum(Xa*lX)
    dLogLb <- alpha*beta^(-alpha-1)*sum(Xa)-N*alpha/beta
    
    if(is.na(dLogLa)){stop("Error in dLogLWeibull: NA produced in dLogLa")}
    if(is.na(dLogLb)){stop("Error in dLogLWeibull: NA produced in dLogLb")}
    
    dlogL <- c(dLogLa,dLogLb)
    return(dlogL)
}

HlogLWeibull <- function(tabX,alpha,beta){

    ## Hessian matrix of log-likelihood

    ## Inputs:
    ## tabX  = sample values
    ## alpha = shape parameter (alpha>0)
    ## beta  = scale parameter (beta>0)
    
    ## Output: H = Hessian matrix of log-likelihood (2 rows 2 columns)  
    
    if(sum(is.na(tabX))>0){stop("Error in dLogLWeibull: NA value in tabX")}
    if(is.na(alpha)){stop("Error in HlogLWeibull: NA value for alpha")}
    if(is.na(beta)){stop("Error in HlogLWeibull: NA value for beta")}
    if(sum(tabX<=0)>0){stop("Error in dLogLWeibull: tabX values must be strictly positive")}
    if(alpha<=0){stop("Error in HlogLWeibull: alpha must be strictly positive")}
    if(beta<=0){stop("Error in HlogLWeibull: beta must be strictly positive")}   
    
    N <- length(tabX)
    if(N==0){stop("Error in HlogLWeibull: tabX of size zero")}
    
    a      <- alpha
    b      <- beta
    aa     <- alpha^2
    bb     <- beta^2
    bma    <- beta^(-alpha)
    lnb    <- log(beta)
    lnbb   <- lnb^2
    Xa     <- tabX^(alpha)
    lX     <- log(tabX)
    lXX    <- lX^2
    SXa    <- sum(Xa)
    SXalX  <- sum(Xa*lX)
    SXalXX <- sum(Xa*lXX)
    
    H11 <- -N/aa-bma*lnbb*SXa+bma*lnb*SXalX+bma*lnb*SXalX-bma*SXalXX
    H12 <- -N/b-a*b^(-a-1)*lnb*SXa+b^(-a-1)*SXa+a*b^(-a-1)*SXalX
    H21 <- H12
    H22 <- N*a/bb-a*(a+1)*b^(-a-2)*SXa
    
    if(is.na(H11)){stop("Error in HlogLWeibull: NA value in H11")}
    if(is.na(H12)){stop("Error in HlogLWeibull: NA value in H12")}
    if(is.na(H22)){stop("Error in HlogLWeibull: NA value in H22")}
    
    H <- matrix(c(H11,H21,H12,H22),byrow=TRUE,nrow=2,ncol=2)
    return(H)
}

fit.Weibull <- function(tabX,Iterations=1000){

    ## Fits a Weibull distribution on tabX using the Newton algorithm
    
    ## Output: 
    ## param <- c(alpha,beta)
    
    ## Basic tests
    
    id  <- which(!is.na(tabX))
    Nid <- length(id)
    nNA <- sum(is.na(tabX))
    
    if(nNA>0){
	print("warning in fit.Weibull: NA values in tabX")
	if(Nid==0){stop("Error in fit.Weibull: only NA values in tabX")}
	tabX <- tabX[id]
    }
    N <- length(tabX)
    
    ## Initial Estimate through moments
    seqA <- seq(0.1,50.0,0.1)
    mX   <- mean(tabX)
    mXX  <- mean(tabX^2)
    G1   <- gamma(1+2/seqA)
    G2   <- gamma(1+1/seqA)^2
    id1  <- which(!is.na(G1))
    id2  <- which(!is.na(G2))
    id3  <- which(G1!=0)
    id4  <- intersect(id1,id2)
    id4  <- intersect(id4,id3)
    if(length(id4)==0){stop("Error in fit.Weibull: no initial value could be computed")}
    seqA <- seqA[id4]
    Imin <- which.min(abs(G2[id4]/G1[id4]-mX^2/mXX))
    a0   <- seqA[Imin]
    b0   <- mX/gamma(1+1/a0)
    
    print("fitting Weibull distribution")
    print(sprintf("initial value is (%f,%f)",a0,b0))
    
    bestx   <- c(a0,b0)
    oldx    <- matrix(bestx,nrow=2,ncol=1)
    oldLogL <- LogLWeibull(tabX,a0,b0)
   
    for(i in 1:Iterations){
    
	gradient <- dLogLWeibull(tabX,oldx[1],oldx[2])
	gradient <- matrix(gradient,nrow=2,ncol=1)
	H        <- HlogLWeibull(tabX,oldx[1],oldx[2])
	
	gradient <- (-1)*gradient
	H        <- (-1)*H	
	
	detH     <- H[1,1]*H[2,2]-H[1,2]^2
	if(is.na(detH)){
	    print("iterations aborted: NA value in detH")
	    bestx <- c(oldx[1],oldx[2])
	    break
	}
	if(abs(detH)<1e-6){
	    print("iterations aborted: detH close to 0")
	    bestx <- c(oldx[1],oldx[2])
	    break
	}
	
	invH  <- solve(H)
	newx  <- oldx-invH%*%gradient
	if(sum(is.na(newx))>0){
	    print("iterations aborted: NA value in newx")
	    bestx <- c(oldx[1],oldx[2])
	    break
	}
	if(newx[1]<=0 || newx[2]<=0){
	    print("iterations aborted: negative values in newx")
	    bestx <- c(oldx[1],oldx[2])
	    break
	}
	
	newLogL <- LogLWeibull(tabX,newx[1],newx[2])
	if(newLogL>oldLogL){
	    oldx    <- newx
	    oldlogL <- newLogL
	    bestx   <- c(newx[1],newx[2])
	}
	else{
	    print("iterations aborted: log-likelihood did not increase")
	    bestx <- c(oldx[1],oldx[2])
	    break
	}
    }
    
    return(bestx)
}
