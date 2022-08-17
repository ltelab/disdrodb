## This script can be used to plot DSD spectra recorded by a Parsivel disdrometer:
## Last Modifications : M.Schleiss, 19 September 2008

## The following packages must be loaded

source("/USERS/lte/Prog_com/Lib_R/lib_Graphics.R")
source("/USERS/lte/Prog_com/Lib_R/lib_Gamma.R")

## The following parameters must be defined by the user:
## maxD = the maximum drop diameter to be represented on the graph (in mm)
## min  = the minimum number of drops to be represented. Smaller values will be plotted in white
## max  = the maximum number of drops to be represented. Larger values will be plotted in darkred
## rounding defines the precision with which the values in the legend are computed (rounding=1 approximate to unity)

maxD <- 5
min  <- 2
max  <- 240
rounding <- 5

t_begin <- as.POSIXct("1998-09-11 8:33:00") 
t_end   <- as.POSIXct("1998-09-11 10:51:00") 
x_begin <- as.POSIXct("1998-09-11 8:30:00") 
x_lab1  <- as.POSIXct("1998-09-11 9:00:00") 
x_lab2  <- as.POSIXct("1998-09-11 9:30:00") 
x_lab3  <- as.POSIXct("1998-09-11 10:00:00")
x_lab4  <- as.POSIXct("1998-09-11 10:30:00")
x_end   <- as.POSIXct("1998-09-11 11:00:00") 
seqxlab <- c(x_begin,x_lab1,x_lab2,x_lab3,x_lab4,x_end)
seqT    <- seq(t_begin,t_end,20)

## The DSD values must be stored in a matrix called DSD_matrix:
## Put on the rows : the observations
## Put on the columns : the number of drops per diameter classes (32 classes for Parsivel = 32 columns in the matrix)


######################################################################################################################################

## Definition of the color palette
color_ramp  <- colorRamp(c("darkblue","blue","cyan","yellow","orange","red","darkred"),bias=1,space="rgb",interpolate="spline")
color_palette <- seq(0,1,0.01)
nC  <- length(color_palette)
nDSD <- dim(DSD_matrix)[1]

name <- "/USERS/mschleis/DSD_spectra.eps"

postscript(name,height=6,width=7.5,horizontal=FALSE,onefile=FALSE,bg="transparent")

## Basic plot
xrange <- diff(range(as.numeric(seqxlab)))
xlim <- c(as.numeric(x_begin),as.numeric(x_end)+xrange/7.5)
plot(0,0,"n",xaxt="n",xlim=xlim,ylim=c(0,maxD+maxD/10),xaxp=c(0,nDSD,10),yaxp=c(0,maxD,10),main="",xlab="time UT",ylab="diameter [mm]",bty="n",cex.lab=1.5,cex.axis=1.5,cex.main=1.5)
for(i in 1:nDSD){
	tab_DSD <- DSD_matrix[i,]
	for(j in 1:length(low_D)){
		z <- tab_DSD[j]
		if(z>0 && mean_D[j]<=maxD){
			if(z<=min){mycol<-color_ramp(0)}
			if(z>=max){mycol<-color_ramp(1)}
			if(z>min && z<max){mycol<-color_ramp((z-min)/(max-min))}
			xleft  <- as.numeric(seqT[i])-20
			xright <- as.numeric(seqT[i])
			ydown  <- low_D[j]
			ytop   <- up_D[j]
			rect(xleft,ydown,xright,ytop,col=rgb(mycol/255),border=rgb(mycol/255))
		}
	}
}

## Plotting x-axis
axis.POSIXct(1,at=seqxlab,format="%H:%M",cex.axis=1.6)

## Plotting the color bar on the right
Seqy <- seq(maxD/nC,maxD,maxD/nC)
for(i in 1:nC){
	ydown  <- Seqy[i]-maxD/nC
	ytop   <- Seqy[i]
	xleft  <- as.numeric(x_end) + xrange/20
	xright <- as.numeric(x_end) + xrange/12
	mycol  <- color_ramp(color_palette[i])
	rect(xleft,ydown,xright,ytop,col=rgb(mycol/255),border=rgb(mycol/255))
}

## Plotting the legend on the right of the color bar
xtext <- as.numeric(x_end) + xrange/7.5
minV <- round(min/rounding)*rounding
V14  <- round((min+(max-min)/4)/rounding)*rounding
V12  <- round((min+(max-min)/2)/rounding)*rounding
V34  <- round((min+3*(max-min)/4)/rounding)*rounding
maxV <- round(max/rounding)*rounding
text(xtext,0,minV,cex=1.5)
text(xtext,maxD/4,V14,cex=1.5)
text(xtext,maxD/2,V12,cex=1.5)
text(xtext,3*maxD/4,V34,cex=1.5)
text(xtext,maxD,maxV,cex=1.5)

dev.off()
