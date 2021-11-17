## New Graphics library

DSD_time_spectra <- function(tabT,DSD_matrix,lowD,upD,min_class,max_class,minz,maxz,nxlab,hplot,wplot,name,log=FALSE,nlegend=5){

    ## Plots a nice color DSD time series

    ## Inputs:
    ## tabT = timetable [seconds since t0]
    ## DSD_matrix = DSD matrix (each row = 1DSD)
    ## lowD = lower diameter class limits
    ## upD  = upper diameter class limits
    ## min_class = minimum diameter class to be plotted
    ## max_class = maximum diameter class to be plotted
    ## minz = minimum concentration to be plotted
    ## maxz = maximum concentration to be plotted
    ## nxlab = number of thickmarks on x-axis
    ## hplot = height of plot 
    ## wplot = width of plot
    ## name = name of plot (with complete path)

    source("/USERS/lte/Prog_com/Lib_R/lib_stats.R")
    t0 <- ISOdatetime(year=1970,month=1,day=1,hour=0,min=0,sec=0,tz="GMT")    

    NtabT <- length(tabT)
    nrow  <- dim(DSD_matrix)[1]
    ncol  <- dim(DSD_matrix)[2]
    NlowD <- length(lowD)
    NupD  <- length(upD)
    
    if(NlowD!=ncol){stop()}
    if(NupD!=ncol){stop()}
    if(NtabT!=nrow){stop()}
    if(min_class<1){stop()}
    if(max_class>ncol){stop()}
    if(min_class>=max_class){stop()}
    if(nxlab<=0){stop()}
    if(minz>=maxz){stop()}

    tabT   <- as.numeric(tabT)
    dx     <- mode(diff(tabT))
    meanD  <- (lowD+upD)/2
    tbegin <- tabT[1]
    tend   <- tabT[NtabT]
    minx   <- tbegin-dx/2
    maxx   <- tend+dx/2+(tend+dx-tbegin)/8
    miny   <- lowD[min_class]
    maxy   <- upD[max_class]
    yend   <- maxy+(maxy-miny)/9
    xlim   <- c(minx,maxx)
    ylim   <- c(miny,yend)

    mai    <- c(1.1,1.1,0.2,0.2)
    cl     <- 1.6
    ca     <- 1.2
    xlab   <- "Time [GMT]"
    ylab   <- "drop diameter [mm]"

    tab_col <- c("darkblue","blue","cyan","yellow","orange","red","darkred")
    color_ramp <- colorRamp(tab_col,bias=1,space="rgb",interpolate="spline")
    color_ramp2 <- colorRamp(c("white","darkred"),bias=1,space="rgb",interpolate="spline")
    dC <- 0.01
    color_palette <- seq(dC,1.0,dC)
    nC  <- length(color_palette)
    maxN <- max(DSD,na.rm=TRUE)
    if(log==TRUE){
	minz2 <- log10(minz)
	maxz2 <- log10(maxz)
    }

    postscript(name,height=hplot,width=wplot,horizontal=FALSE,onefile=FALSE,bg="white")
    par(mai=mai)
    plot(0,0,"n",xaxt="n",xlim=xlim,ylim=ylim,xlab=xlab,ylab=ylab,cex.lab=cl,cex.axis=ca)
    for(i in 1:NtabT){
	x <- tabT[i]
	if(is.na(x)){next}
	tabN <- DSD_matrix[i,]
	for(j in (min_class:max_class)){
	    n  <- tabN[j]
	    if(is.na(n)){next}
	    if(log==TRUE){
		if(n==0){next}
		n <- log10(n)
	    }
	    y0 <- lowD[j]
	    y1 <- upD[j]
	    if(n<=minz2){mycol <- color_ramp2(0)}
	    if(n>=maxz2){mycol <- color_ramp2(1)}
	    if(n>minz2 && n<maxz2){mycol <- color_ramp((n-minz2)/(maxz2-minz2))}
	    x0 <- x-dx/2
	    x1 <- x+dx/2
	    rect(x0,y0,x1,y1,col=rgb(mycol/255),border=rgb(mycol/255))
	}
    }

    for(i in 1:nC){
	mycol <- color_ramp(color_palette[i])
	x0 <- tend+dx/2+(maxx-tend-dx/2)/4
	x1 <- tend+dx/2+2*(maxx-tend-dx/2)/4
	y0 <- miny+(color_palette[i]-dC)*(maxy-miny)
	y1 <- miny+color_palette[i]*(maxy-miny)
	rect(x0,y0,x1,y1,col=rgb(mycol/255),border=rgb(mycol/255))
    }

    for(i in 1:nlegend){
	y <- miny+(i-1)*(maxy-miny)/(nlegend-1)
	z <- (maxz-minz)*i/nlegend
	z <- 50*round(z/50)
	x <- tend+dx/2+2.25*(maxx-tend-dx/2)/4
	text(x,y,z,adj=0,cex=1.05)
    }

    x <- (tend+dx/2+(maxx-tend-dx/2)/4+maxx)/2
    y <- yend
    text(x,y,expression(paste(N[t]," [",m^-3,"]")),cex=1.20,adj=0.5)

    seqT <- seq(t0+tabT[1],t0+tabT[NtabT],15*60)
    axis.POSIXct(1,at=seqT,format="%H:%M",cex.axis=ca)

    dev.off()
}
