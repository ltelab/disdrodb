## Make 3D grid ##

make_3Dgrid <- function(seqx,seqy,seqz){
    Nx <- length(seqx)
    Ny <- length(seqy)
    Nz <- length(seqz)
    grid <- matrix(NA,nrow=Nx*Ny*Nz,ncol=3)
    itr  <- 0
    for(i in 1:Nx){
	x <- seqx[i]
	for(j in 1:Ny){
	    y <- seqy[j]
	    for(k in 1:Nz){
		z <- seqz[k]
		itr <- itr+1
		grid[itr,] <- c(x,y,z)
	    }
	}
    }
    return(grid)
}


########################################### Levelplot, 3rd version ############################################

plot_map3 <- function(M,value_range,xlim,ylim,pixel_size,rounding,rescale=FALSE){

	## Inputs :
	## M           = The map matrix (x ; y ; value)
	## value_range = tabular containing the minimum and the maximum value for the levelplot
	## xlim        = tabular containing the minimum and the maximum value on the x-axis 
	## ylim        = tabular containing the minimum and the maximum value on the y-axis
	## pixel_size  = the size of the pixels (x,y) for the plot.
	## rounding    = the precision of the legend annotation
	## rescale     = logical, if rescale==TRUE the original pixels are rescaled and merged to form greater pixels.
	##               This is of interest if the field contains a huge amount of pixels 
	## 		 The new dimension of the pixels is given by the parameter pixel_size.

	## Remarks : (1) The desired color_ramp must be created by the user before the call to plot_map3() 
	## 	     (2) The plot must be generated before the call to plot_map3().
	## 	     (3) Additional annotations or points can be drawn after the call to plot_map3() 

	dimM  <- dim(M)[1]
	xrange <- c(min(M[,1]),max(M[,1]))		# 
	yrange <- c(min(M[,2]),max(M[,2]))
	span  <- value_range[2]-value_range[1]
	xspan <- xlim[2]-xlim[1]
	yspan <- ylim[2]-ylim[1]
	
	Seqx  <- seq(xrange[1],xrange[2],pixel_size[1])
	Seqy  <- seq(yrange[1],yrange[2],pixel_size[2])
	nX    <- length(Seqx)
	nY    <- length(Seqy)
	nXY   <- nX*nY
	xy    <- expand.grid(Seqx,Seqy)
	xy    <- as.matrix(xy)
	
	## Rescaling the pixels
	## Creation of a new map matrix called new_M (x ; y ; number of pixels ; sum ; mean)
	if(rescale==TRUE){

		new_M <- matrix(rep(0,3*nXY),nrow=nXY,ncol=3)
		new_M <- cbind(xy,new_M)
		
		Mapped_X <- value_mapping(M[,1],Seqx)
		Mapped_Y <- value_mapping(M[,2],Seqy)
		
		for(itr in 1:dimM){
			z <- M[itr,3]
			if(is.na(z)==FALSE){
				x <- Mapped_X[itr]
				y <- Mapped_Y[itr]
				idx <- sum(c(Seqx<=x))
				idy <- sum(c(Seqy<=y))
				id  <- (idy-1)*nX + idx
				new_M[id,3] <- new_M[id,3]+1
				new_M[id,4] <- new_M[id,4]+z
			}
		}
		for(itr in 1:nXY){
			n   <- new_M[itr,3]
			if(n==0){new_M[itr,5] <- NA}
			else{new_M[itr,5] <- new_M[itr,4]/n}
		}
	}
	else{
		new_M <- matrix(rep(0,3*dimM),nrow=dimM,ncol=3)
		new_M <- cbind(M[,1:2],new_M)
		new_M[,5] <- M[,3]
	}
	
	## plotting the pixels
	nnM <- dim(new_M)[1]
	for(itr in 1:nnM){
		z <- new_M[itr,5]
		if(is.na(z)==FALSE){
			if(z>=value_range[1]){
				xinf <- new_M[itr,1]-pixel_size[1]
				yinf <- new_M[itr,2]-pixel_size[2]
				xsup <- new_M[itr,1]
				ysup <- new_M[itr,2]
				if(xinf>=xlim[1] && xsup<=xlim[2] && yinf>=ylim[1] && ysup<=ylim[2]){
					if(z>=value_range[2]){mycol <- color_ramp(1)}
					else{mycol <- color_ramp((z-value_range[1])/span)}
					rect(xinf,yinf,xsup,ysup,col=rgb(mycol/255),border=rgb(mycol/255))
				}
			}
		}
	}

	## plotting the colour bar
	color_palette <- seq(0,1,0.01)
	nC <- length(color_palette)
	for(itr in 1:(nC-1)){
		xinf  <- xlim[2]+xspan/15
		xsup  <- xlim[2]+xspan/10
		yinf  <- ylim[1]+color_palette[itr]*yspan
		ysup  <- ylim[1]+color_palette[itr+1]*yspan
		mycol <- color_ramp(color_palette[itr])
		rect(xinf,yinf,xsup,ysup,col=rgb(mycol/255),border=rgb(mycol/255))
	}

	## plotting the legend
	tab_x <- rep(xlim[2]+xspan/10+xspan/15,5)
	tab_y <- seq(ylim[1],ylim[2],yspan*0.25)
	tab_z <- seq(value_range[1],value_range[2],span*0.25)
	for(itr in 1:5){
		round_z <- rounding*round(tab_z[itr]/rounding,digits=0)
		text(tab_x[itr],tab_y[itr],round_z,cex=1.85)
	}
	
}


########################################### Levelplot, 2nd version ############################################


plot_map2 <- function(M,Seqx,Seqy,dxy,extrema,xlim,ylim,color_ramp,color_ramp2,rounding,minmin){

	## Warning : This version is out of date and may not work anymore.
	## Use the latest plot_map version for better results.

        color_ramp3 <- colorRamp(c("grey","black"),bias=1,space="rgb",interpolate="spline")

	range <- max-min
	nSeqx <- length(Seqx)
 	nSeqy <- length(Seqy)
	nxy   <- nSeqx*nSeqy
	dx <- dxy[1]
	dy <- dxy[2]
	min <- extrema[1]
	max <- extrema[2]
	xinf <- xlim[1]
	xsup <- xlim[2]
	yinf <- ylim[1]
	ysup <- xlim[2]

	for(i in 1:nxy){
		z <- M[i,3]
		if(z==minmin){mycol<-color_ramp3(0)}
		if(z<min && z!=minmin){mycol<-color_ramp2(0)}
	  	if(z>=max && z!=minmin){mycol<-color_ramp(1)}
		if(z>min && z<max && z!=minmin){mycol<-color_ramp((z-min)/range)}
		rect(M[i,1]-dx,M[i,2]-dy,M[i,1],M[i,2],col=rgb(mycol/255),border=rgb(mycol/255))
	}

	color_palette <- seq(0,1,0.01)
	nC <- length(color_palette)

	for(i in 1:nC){
		ydown  <- i*(M[nxy,2]-yinf)/(nC-1)+yinf-(M[nxy,2]-yinf)/(nC-1)
		ytop   <- (i+1)*(M[nxy,2]-yinf)/(nC-1)+yinf-(M[nxy,2]-yinf)/(nC-1)
		xleft  <- M[nxy,1]+(M[nxy,1]-xinf)/20
		xright <- M[nxy,1]+(M[nxy,1]-xinf)/12
		mycol  <- color_ramp(color_palette[i])
		rect(xleft,ydown,xright,ytop,col=rgb(mycol/255),border=rgb(mycol/255))
	}

	xtext <- M[nxy,1]+(M[nxy,1]-xinf)/8
	minV <- round(min/rounding)*rounding
	V14  <- round((min+(max-min)/4)/rounding)*rounding
	V12  <- round((min+(max-min)/2)/rounding)*rounding
	V34  <- round((min+3*(max-min)/4)/rounding)*rounding
	maxV <- round(max/rounding)*rounding
	text(xtext,yinf,minV)
	text(xtext,yinf+(M[nxy,2]-yinf)/4,V14)
	text(xtext,yinf+(M[nxy,2]-yinf)/2,V12)
	text(xtext,yinf+3*(M[nxy,2]-yinf)/4,V34)
	text(xtext,M[nxy,2],maxV)
} 

########################################### Levelplot, 1st version ############################################

plot_map <- function(M,Seqx,Seqy,dxy,extrema,xlim,ylim,color_ramp,color_ramp2,rounding){


	## Warning : This version is out of date and may not work anymore.
	## Use the latest plot_map version for better results.

	nSeqx <- length(Seqx)
 	nSeqy <- length(Seqy)
	nxy   <- nSeqx*nSeqy
	dx <- dxy[1]
	dy <- dxy[2]
	min <- extrema[1]
	max <- extrema[2]
	range <- max-min
	xinf <- xlim[1]
	xsup <- xlim[2]
	yinf <- ylim[1]
	ysup <- xlim[2]

	for(i in 1:nxy){
		x <- M[i,1]
		y <- M[i,2]
		if(x>=500 && x<=600 && y>=2400 && y<=2500){ 
			z <- M[i,3]
			if(is.na(z)==FALSE){
				if(z<min){mycol<-color_ramp2(0)}
				if(z>=max){mycol<-color_ramp(1)}
				if(z>=min && z<max){mycol<-color_ramp((z-min)/range)}
				rect(M[i,1]-dx,M[i,2]-dy,M[i,1],M[i,2],col=rgb(mycol/255),border=rgb(mycol/255))
			}
		}
	}

	color_palette <- seq(0,1,0.01)
	nC <- length(color_palette)

	for(i in 1:nC){
		ydown  <- i*(M[nxy,2]-yinf)/(nC-1)+yinf-(M[nxy,2]-yinf)/(nC-1)
		ytop   <- (i+1)*(M[nxy,2]-yinf)/(nC-1)+yinf-(M[nxy,2]-yinf)/(nC-1)
		xleft  <- M[nxy,1]+(M[nxy,1]-xinf)/10
		xright <- M[nxy,1]+(M[nxy,1]-xinf)/8
		mycol  <- color_ramp(color_palette[i])
		rect(xleft,ydown,xright,ytop,col=rgb(mycol/255),border=rgb(mycol/255))
	}

	xtext <- M[nxy,1]+(M[nxy,1]-xinf)/7.5
	minV <- round(min/rounding)*rounding
	V14  <- round((min+(max-min)/4)/rounding)*rounding
	V12  <- round((min+(max-min)/2)/rounding)*rounding
	V34  <- round((min+3*(max-min)/4)/rounding)*rounding
	maxV <- round(max/rounding)*rounding
	text(xtext,yinf,minV,cex=2.0)
	text(xtext,yinf+(M[nxy,2]-yinf)/4,V14,cex=2.0)
	text(xtext,yinf+(M[nxy,2]-yinf)/2,V12,cex=2.0)
	text(xtext,yinf+3*(M[nxy,2]-yinf)/4,V34,cex=2.0)
	text(xtext,M[nxy,2],maxV,cex=2.0)
} 


################################# Histogram function ####################################

my_histogram <- function(Y,min,max,bins){

   ## Computes a histogram of Y

   ## Inputs :
   ##   Y    = sample vector
   ##   min  = minimum value on x-axis
   ##   max  = maximum value on x-axis
   ##   bins = number of bins between min and max.

   ## Outputs :
   ##   H    = a histogram matrix
   ##          1st column : the ending value of the bin
   ##          2nd column : the frequency associated to this bin

   ## Remarks : min must be less or equal to min(Y).
   ##           max must be greater or equal to max(Y)

   N    <- length(Y)-sum(c(is.na(Y)))
   Y    <- sort(Y)
   minY <- min(Y,na.rm=TRUE)
   maxY <- max(Y,na.rm=TRUE)

   if(min>minY){min <- minY}
   if(max<maxY){max <- maxY}

   dx <- (max-min)/bins 
   B <- seq(min+dx,max,dx)
   NB <- rep(0,bins)

   bin_number <- 1
   for(i in 1:N){
      if(is.na(Y[i])){}
      else{
         while(Y[i]>B[bin_number] && bin_number<bins){
            bin_number <- bin_number+1
         }
         NB[bin_number] <- NB[bin_number]+1
      }
   }
   H <- matrix(B,ncol=1)
   H <- cbind(H,NB/N)
   return(H)
   
}



############################################ 3D plot #####################################


map3D <- function(tabx,taby,tabz,azimuth,dx,min,max,bar=FALSE,np=100){

	## 3D Plot
	## Inputs:
	## tabx  = x-coordinates
	## taby  = y-coordinates
	## tabz  = z-coordinates (or intensity of the process)
	## azimuth = azimuth angle (in degrees) 
	## dx = size of pixels
	## min = z-value that must be plotted in darkblue
	## max = z-value that must be plotted in darkred
	## np = number of pixels for vertical bar

	## Basic verifications
	N <- length(tabx)
	if(length(taby)!=N || length(tabz)!=N){stop("x,y,z must have same length")}

	alpha <- 2*pi*(90-azimuth)/360

	## sorting by decreasing y-coordinate (for perspective)
	S <- sort(taby,decreasing=TRUE,index.return=TRUE)
	taby <- S$x
	idx  <- S$ix
	tabx <- tabx[idx]
	tabz <- tabz[idx]

	## Computing projections on (x,y) plane
	Coordx <- tabx + taby*cos(alpha)
	Coordy <- tabz + taby*sin(alpha)

	## Drawing axis
	points(c(0,0),c(0,max(Coordy)),"l")
	points(c(0,max(Coordx)),c(0,0),"l")
	points(c(0,max(Coordx)),c(0,max(Coordx)*tan(alpha)),"l")

	## Defining Color ramp
	my_ramp <- colorRamp(c("darkblue","blue","cyan","yellow","orange","red","darkred"),interpolate="spline")

	## Computing range of z-coordinate (intensity)
	range <- max-min

	## Plotting intensity bars for each pixel
	for(i in 1:N){
		x <- Coordx[i]
		y <- Coordy[i]
		if(bar==TRUE){
			## Computing ground projection
			ground <- taby[i]*sin(alpha)
			## Defining the intensity bar
			seqy <- seq(ground,y,(y-ground)/np)
			dY   <- seqy[2]-seqy[1]
			seqcol <- seq(0,(tabz[i]-min)/range,(tabz[i]-min)/(np*range))
			for(j in 1:length(seqy)){
				if(!is.na(seqcol[j])){
					col    <- my_ramp(seqcol[j])
					red    <- col[1]/255
					green  <- col[2]/255
					blue   <- col[3]/255
					ydown  <- seqy[j]-dY
					ytop   <- seqy[j]
					rect(x-dx,ydown,x,ytop,col=rgb(red,green,blue),border=rgb(red,green,blue))
				}
			}
		}
		if(bar==FALSE){
			z <- tabz[i]
			if(!is.na(z)){
				if(z>max){col <- my_ramp(1)}
				if(z<min){col <- my_ramp(0)}
				if(z>=min && z<=max){col <- my_ramp((z-min)/range)}
				red    <- col[1]/255
				green  <- col[2]/255
				blue   <- col[3]/255
				rect(x-dx,y-dx,x,y,col=rgb(red,green,blue),border=rgb(red,green,blue))
			}
		}
	}
}

################################################### Plotting an Adjacence matrix #######################################

plot_Adj <- function(tabX,tabY,Adj,color=1){
	ltA <- dim(Adj)[1]
	for(i in 1:ltA){
		points(tabX[Adj[i,1:2]],tabY[Adj[i,1:2]],"l",lwd=2,col=color)
	}
}



