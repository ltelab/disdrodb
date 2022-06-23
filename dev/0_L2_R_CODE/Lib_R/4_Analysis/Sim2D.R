############################### READ ME #################################

## This code computes reflectivity,rain rate and other parameters out of 2D DSD Simulations
## The Simulation itself must have been done previously
## Some real radar images from the HIRE98 experiment are used for comparison

## Source : A.Berne, 2007-2008
##          M.Schleiss, 2008


################################ LIBRARY #################################

library(gstat)
source("/USERS/lte/Prog_Com/lib_R/lib_DSD.R")
source("/USERS/lte/Prog_Com/lib_R/Beard_Model.R")
source("/USERS/lte/Prog_Com/lib_R/lib_Gamma.R")
source("/USERS/lte/Prog_com/Lib_R/lib_Graphics.R")

################################## MAIN ##################################

initial_time <- Sys.time()

## Initialization :
## Global parameters :
##   tp      = the temperature (in °C)
##   mwv     = the mean wind velocity (in m/s)
##   dr      = the spatial resolution (in m)
##   dt      = the temporal resolution (in m)
##   wlength = the wavelength (in cm)
##   ref_ind = the refractive index of liquid water
##   K_w     = the dielectric constant associated with ref_ind
##   nx      = the grid size

## DSD Parameters :
##   D_min   = the minimum drop size (in mm)
##   D_max   = the maximum drop size (in mm)
##   d_D     = the discretization of the drop size (in mm)
##   tab_D   = the tabular with all the possible drop sizes
##   N_D     = the size of tab_D

## Radar related parameters :
##   tab_E   = the extinction cross section (in cm^2)
##   tab_TS  = the scattering cross section (in cm^2)
##   tab_BS  = the backscattering cross section (in cm^2)
##   tab_R   = the rain rate (in mm/h)
##   tab_Z   = the reflectivity (in mm6/m3)

## Other parameters :
##   nang    = the number of angles for numerical integration in Mie Theory
##   radar_nx = the resolution of the radar (in pixels)
##   radar_dr = the spatial resolution of the radar (in m)
##   sim_nx   = the resolution of the simulated field (in pixels)
##   sim_dr   = the spatial resolution of the simulated field (in m)

## for(time in seq(50,50,5)){
time <- 30

file1 <- sprintf("/home/mschleis/Mediterranean Intense/Radar Data/radar_data_hire_1998090710%i_05min_processed.txt",time)
file2  <- "/home/mschleis/Mediterranean Intense/Fields/Sim_Lambda_Resolution250m_128x128_Sim02.txt"
file3  <- "/home/mschleis/Mediterranean Intense/Fields/Sim_Nt_Resolution250m_128x128_Sim02.txt"

wlength <- 10
tp <- 20
mwv <- 12.5
dt  <- 20
dr  <- mwv*dt
ref_ind <- refractive_index_water(tp,wlength)
K_w <- (ref_ind^2-1)/(ref_ind^2+2)
nang <- 20

radar_nx <- 32
radar_dr <- 1000
sim_nx   <- 128
sim_dr   <- 250

## Discretization of the drop size (in mm)
D_min <- 0.01
D_max <- 5.00
d_D   <- 0.01
tab_D <- seq(D_min,D_max,d_D)
n_D   <- length(tab_D)

## Lookup table for terminal drop velocity
tab_V <- c()
for(i in 1:n_D){
  D <- tab_D[i]
  tab_V[i] <- drop_velocity(D)
}

## Lookup table for Mie coefficients
tab_E  <- rep(0,n_D)
tab_TS <- rep(0,n_D)
tab_BS <- rep(0,n_D)
for(i in 1:n_D){
  D <- tab_D[i]
  x <- pi*D/(wlength*10)
  Mie <- mie_scat(x,ref_ind,nang)
  ## Mie is a vector of 3 components
  ## Mie[1] = scattering efficiency
  ## Mie[2] = extinction efficiency
  ## Mie[3] = backscattering efficiency
  ## In order to have the cross section in cm^2 one must multiply by pi*r^2/100
  tab_TS[i] <- Mie[1]*(pi*D^2*0.01/4)
  tab_E[i]  <- Mie[2]*(pi*D^2*0.01/4)
  tab_BS[i] <- Mie[3]*(pi*D^2*0.01/4)
}

## Reading the radar data (in dBZ)
radar_data <- read.table(file1,skip=4) 
radar_data <- unlist(radar_data)

## Radar grid size = radar_nx (in pixels) 
## Radar resolution = radar_dr (in m)
xy <- expand.grid((1:radar_nx)*radar_dr,(1:radar_nx)*radar_dr)
xy <- as.matrix(xy)

## Plotting the radar map
# radar_map <- cbind(xy/1000,radar_data)
# color_ramp <- colorRamp(c("darkblue","blue","cyan","yellow","orange","red","darkred"),bias=1,space="rgb",interpolate="spline")
# color_ramp2 <- colorRamp(c("white","black"))
# color_palette <- seq(0,1,0.01)
# nC   <- length(color_palette)
# 
# M <- radar_map
# Seq         <- seq(1,32,1)
# Seqx        <- Seq
# Seqy        <- Seq
# dx          <- 1
# dy          <- 1
# nSeqx       <- length(Seqx)
# nSeqy       <- length(Seqy)
# nxy         <- nSeqx*nSeqy
# nSeq        <- length(Seq)
# xinf <- M[1,1]-dx
# xsup <- M[nxy,1]+(M[nxy,1]-xinf)/9.5
# yinf <- M[1,2]-dx
# ysup <- M[nxy,2]+(M[nxy,2]-yinf)/15
# xlim <- c(xinf,xsup)
# ylim <- c(yinf,ysup)
# pdf(sprintf("/home/mschleis/Mediterranean Intense/Radar_map_%i.pdf",time),h=6,w=6)
# min <- 20
# max <- 60
# par(mai=c(1,1,0.25,0.4))
# plot(M[,1],M[,2],"n",xlab="Distance [km]",ylab="Distance [km]",xlim=xlim,ylim=ylim,cex.lab=1.1,bty="n",xaxp=c(0,32,8),yaxp=c(0,32,8))
# plot_map2(M,Seqx,Seqy,c(dx,dy),c(min,max),xlim,ylim,color_ramp,color_ramp2,5,-99)
# text(M[nxy,1]+(M[nxy,1]-xinf)/13,M[nxy,2]+(M[nxy,2]-yinf)/15,"dBZ")
# text((xsup-xinf)/2,ysup,"Observed Radar Reflectivity",cex=1.4)
# dev.off()


## The radar data is filtered
## On can choose to cut/drop bad and/or missing values 

radar_data_matrix <- filter_data(radar_data,radar_nx,radar_dr,21,100,1)
N_radar <- c()
radar_data <- radar_data_matrix[,3]
N_radar <- length(radar_data)

## The histogram of the radar picture is computed
## title <- "radar reflectivity"
## xlab  <- "reflectivity [dBZ]"
## pdf(sprintf("H%i.pdf",time))
## radar_hist <- my_histogram(radar_data,min(radar_data),max(radar_data),10,title,xlab)
## dev.off()
## }

## The radar map is plotted
## pdf(sprintf("a%i.pdf",time))
## map(radar_data_matrix,radar_nx,radar_dr,0,60,"Radar reflectivity map","32km","32km")
## dev.off()

## The variogram of the radar picture is computed
## As there are missing values we cannot use our simple variogram function
## The irregular_grid_variogram() function is called to do the job.

## title <- "Variogram of Reflectivity [dBZ]"
## xlab  <- "distance [m]"
## pdf(sprintf("Vario%i.pdf",time))
## radar_variogram <- irregular_grid_variogram(radar_data_matrix,radar_dr,radar_nx*radar_dr,title,xlab,20000,70)
## dev.off()

## Reading the simulated Nt and Lambda fields from a file
## Simulated grid size = sim_nx (in pixels)
## Simulation resolution : sim_dr (in m)
tab_Nt <- read.table(file3,skip=11)
tab_L  <- read.table(file2,skip=11)
tab_Nt <- unlist(tab_Nt)
tab_L  <- unlist(tab_L)
N_tab_Nt <- length(tab_Nt)
N_tab_L  <- length(tab_L)

m_ln_Nt <- mean(tab_Nt)
m_ln_L  <- mean(tab_L)
s_ln_Nt <- sd(tab_Nt)
s_ln_L  <- sd(tab_L)


## The radar data resolution and the simulated data resolution must match !
## This job is done by the function "change_resolution(...)"
sim_grid <- expand.grid((1:sim_nx)*sim_dr,(1:sim_nx)*sim_dr)
sim_grid <- as.matrix(sim_grid)
xy       <- expand.grid((1:radar_nx)*radar_dr,(1:radar_nx)*radar_dr)
xy       <- as.matrix(xy)

Nt_matrix <- cbind(sim_grid/1000,tab_Nt)
L_matrix  <- cbind(sim_grid/1000,tab_L)

## We plot the log(Lambda) field with 250m resolution
# M <- L_matrix
# Seq         <- seq(0.25,32,0.25)
# Seqx        <- Seq
# Seqy        <- Seq
# dx          <- 0.25
# dy          <- 0.25
# nSeqx       <- length(Seqx)
# nSeqy       <- length(Seqy)
# nxy         <- nSeqx*nSeqy
# nSeq        <- length(Seq)
# xinf <- M[1,1]-dx
# xsup <- M[nxy,1]+(M[nxy,1]-xinf)/9.5
# yinf <- M[1,2]-dx
# ysup <- M[nxy,2]+(M[nxy,2]-yinf)/15
# xlim <- c(xinf,xsup)
# ylim <- c(yinf,ysup)
# pdf(sprintf("/home/mschleis/Mediterranean Intense/Simulated_Lambda_250m.pdf",time),h=6,w=6)
# min <- 1
# max <- 5
# par(mai=c(1,1,0.25,0.4))
# plot(M[,1],M[,2],"n",xlab="Distance [km]",ylab="Distance [km]",xlim=xlim,ylim=ylim,cex.lab=1.1,bty="n",xaxp=c(0,32,8),yaxp=c(0,32,8))
# plot_map2(M,Seqx,Seqy,c(dx,dy),c(min,max),xlim,ylim,color_ramp,color_ramp2,0.5,-99)
# text(M[nxy,1]+(M[nxy,1]-xinf)/13,M[nxy,2]+(M[nxy,2]-yinf)/15,expression(mm^-1))
# text((xsup-xinf)/2,ysup,expression(paste("Simulated ",Lambda)),cex=1.4)
# dev.off()


## We plot the log(Nt) field with 250m resolution
# M <- Nt_matrix
# pdf(sprintf("/home/mschleis/Mediterranean Intense/Simulated_Nt_250m.pdf",time),h=6,w=6)
# min <- 4
# max <- 10
# par(mai=c(1,1,0.25,0.4))
# plot(M[,1],M[,2],"n",xlab="Distance [km]",ylab="Distance [km]",xlim=xlim,ylim=ylim,cex.lab=1.1,bty="n",xaxp=c(0,32,8),yaxp=c(0,32,8))
# plot_map2(M,Seqx,Seqy,c(dx,dy),c(min,max),xlim,ylim,color_ramp,color_ramp2,1,-99)
# text((xsup-xinf)/2,ysup,expression(paste("Simulated ",ln(N[t]))),cex=1.4)
# dev.off()


## We compute the rain intensity
tab_R <- rep(0,N_tab_Nt)
for(i in 1:N_tab_Nt){
   tab_R[i] <- Rain_int(exp(tab_Nt[i]),exp(tab_L[i]),tab_D,tab_V)
}
Seq <- seq(0.25,32,0.25)
xy <- expand.grid(Seq,Seq)
xy <- as.matrix(xy)
Sim_R_map <- cbind(xy,tab_R)

## We plot the rain intensity
# M <- Sim_R_map
# M[,1:2] <- M[,1:2]
# ## M[,3] <- round(M[,3])
# Seq         <- seq(0.25,32,0.25)
# Seqx        <- Seq
# Seqy        <- Seq
# dx          <- 0.25
# dy          <- 0.25
# nSeqx       <- length(Seqx)
# nSeqy       <- length(Seqy)
# nxy         <- nSeqx*nSeqy
# nSeq        <- length(Seq)
# xinf <- M[1,1]-dx
# xsup <- M[nxy,1]+(M[nxy,1]-xinf)/8.5
# yinf <- M[1,2]-dx
# ysup <- M[nxy,2]+(M[nxy,2]-yinf)/15
# xlim <- c(xinf,xsup)
# ylim <- c(yinf,ysup)
# pdf(sprintf("/home/mschleis/Mediterranean Intense/Simulated_R_250m.pdf",time),h=6,w=6)
# min <- 0
# max <- 100
# par(mai=c(1,1,0.25,0.4))
# plot(M[,1],M[,2],"n",xlab="Distance [km]",ylab="Distance [km]",xlim=xlim,ylim=ylim,cex.lab=1.1,bty="n",xaxp=c(0,32,8),yaxp=c(0,32,8))
# plot_map2(M,Seqx,Seqy,c(dx,dy),c(min,max),xlim,ylim,color_ramp,color_ramp2,5,-99)
# text(M[nxy,1]+(M[nxy,1]-xinf)/13,M[nxy,2]+(M[nxy,2]-yinf)/15,"mm/h")
# text((xsup-xinf)/2,ysup,"Simulated Rain Rate",cex=1.4)
# dev.off()

## The radar reflectivity is calculated at resolution 250m
# tab_Z  <- rep(0,N_tab_Nt)
# for(i in 1:N_tab_Nt){
#   tab_Z[i]  <- Radar_ref(exp(tab_Nt[i]),exp(tab_L[i]),tab_D,tab_BS,wlength,K_w)
# }
# N_tab_Z <- length(tab_Z)
# 
# ## We convert the reflectivity Z in dBZ by taking the log of base 10
# tab_Z  <- 10*log(tab_Z)/log(10)
# Z_data_matrix  <- cbind(xy,tab_Z)
# 
# 
# ## We plot the simulated reflectivity field
# M <- Z_data_matrix
# M[,1:2] <- M[,1:2]
# ## M[,3] <- round(M[,3])
# pdf(sprintf("/home/mschleis/Mediterranean Intense/Simulated_Z_250m.pdf",time),h=6,w=6)
# min <- 20
# max <- 60
# par(mai=c(1,1,0.25,0.4))
# plot(M[,1],M[,2],"n",xlab="Distance [km]",ylab="Distance [km]",xlim=xlim,ylim=ylim,cex.lab=1.1,bty="n",xaxp=c(0,32,8),yaxp=c(0,32,8))
# plot_map2(M,Seqx,Seqy,c(dx,dy),c(min,max),xlim,ylim,color_ramp,color_ramp2,5,-99)
# text(M[nxy,1]+(M[nxy,1]-xinf)/13,M[nxy,2]+(M[nxy,2]-yinf)/15,"dBZ")
# text((xsup-xinf)/2,ysup,"Simulated Reflectivity",cex=1.4)
# dev.off()




Nt_matrix_new <- change_resolution(Nt_matrix,4)
L_matrix_new  <- change_resolution(L_matrix,4)

rm(Nt_matrix)
rm(L_matrix)

Nt_matrix <- Nt_matrix_new
L_matrix  <- L_matrix_new

Nt_matrix[,3] <- exp(Nt_matrix[,3])
L_matrix[,3]  <- exp(L_matrix[,3])

tab_Nt <- c()
tab_L  <- c()

tab_Nt    <- c(Nt_matrix[,3])
tab_L     <- c(L_matrix[,3])
N_tab_Nt  <- length(tab_Nt)
N_tab_L   <- length(tab_L)


## We plot the Lambda field with 1km resolution
# M <- L_matrix
# Seq         <- seq(1,32,1)
# Seqx        <- Seq
# Seqy        <- Seq
# dx          <- 1
# dy          <- 1
# nSeqx       <- length(Seqx)
# nSeqy       <- length(Seqy)
# nxy         <- nSeqx*nSeqy
# nSeq        <- length(Seq)
# xinf <- M[1,1]-dx
# xsup <- M[nxy,1]+(M[nxy,1]-xinf)/9.5
# yinf <- M[1,2]-dx
# ysup <- M[nxy,2]+(M[nxy,2]-yinf)/15
# xlim <- c(xinf,xsup)
# ylim <- c(yinf,ysup)
# pdf(sprintf("/home/mschleis/Mediterranean Intense/Simulated_Lambda_1km.pdf",time),h=6,w=6)
# min <- 1
# max <- 5
# par(mai=c(1,1,0.25,0.4))
# plot(M[,1],M[,2],"n",xlab="Distance [km]",ylab="Distance [km]",xlim=xlim,ylim=ylim,cex.lab=1.1,bty="n",xaxp=c(0,32,8),yaxp=c(0,32,8))
# plot_map2(M,Seqx,Seqy,c(dx,dy),c(min,max),xlim,ylim,color_ramp,color_ramp2,0.5,-99)
# text(M[nxy,1]+(M[nxy,1]-xinf)/13,M[nxy,2]+(M[nxy,2]-yinf)/15,expression(mm^-1))
# text((xsup-xinf)/2,ysup,expression(paste("Simulated ",Lambda)),cex=1.4)
# dev.off()

## We plot the log(Nt) field with 1km resolution
# M <- Nt_matrix
# M[,3] <- log(M[,3])
# pdf(sprintf("/home/mschleis/Mediterranean Intense/Simulated_Nt_1km.pdf",time),h=6,w=6)
# min <- 4
# max <- 10
# par(mai=c(1,1,0.25,0.4))
# plot(M[,1],M[,2],"n",xlab="Distance [km]",ylab="Distance [km]",xlim=xlim,ylim=ylim,cex.lab=1.1,bty="n",xaxp=c(0,32,8),yaxp=c(0,32,8))
# plot_map2(M,Seqx,Seqy,c(dx,dy),c(min,max),xlim,ylim,color_ramp,color_ramp2,1,-99)
# text((xsup-xinf)/2,ysup,expression(paste("Simulated ",ln(N[t]))),cex=1.4)
# dev.off()



## The radar reflectivity is calculated using the DSD
## tab_Z  : the reflectivity calculated with the usual formula
## tab_Z2 : the reflectivity calculated using the Rayleigh approximation
tab_Z  <- c()
tab_Z2 <- c()
for(i in 1:N_tab_L){
  tab_Z  <- c(tab_Z,Radar_ref(tab_Nt[i],tab_L[i],tab_D,tab_BS,wlength,K_w))
  tab_Z2 <- c(tab_Z2,Radar_ref_ray(tab_Nt[i],tab_L[i],tab_D))
}
N_tab_Z <- length(tab_Z)
N_tab_Z2  <- length(tab_Z2)

## We convert the reflectivity Z in dBZ by taking the log of base 10
tab_Z  <- 10*log(tab_Z)/log(10)
tab_Z2 <- 10*log(tab_Z2)/log(10)

Z_data_matrix  <- cbind(xy,tab_Z)
Z2_data_matrix <- cbind(xy,tab_Z2)


## We plot the simulated reflectivity field
# M <- Z_data_matrix
# M[,1:2] <- M[,1:2]/1000
# M[,3] <- round(M[,3])
# pdf(sprintf("/home/mschleis/Mediterranean Intense/Simulated_Z_1km.pdf",time),h=6,w=6)
# min <- 20
# max <- 60
# par(mai=c(1,1,0.25,0.4))
# plot(M[,1],M[,2],"n",xlab="Distance [km]",ylab="Distance [km]",xlim=xlim,ylim=ylim,cex.lab=1.1,bty="n",xaxp=c(0,32,8),yaxp=c(0,32,8))
# plot_map2(M,Seqx,Seqy,c(dx,dy),c(min,max),xlim,ylim,color_ramp,color_ramp2,5,-99)
# text(M[nxy,1]+(M[nxy,1]-xinf)/13,M[nxy,2]+(M[nxy,2]-yinf)/15,"dBZ")
# text((xsup-xinf)/2,ysup,"Simulated Reflectivity",cex=1.4)
# dev.off()


## We compute the rain intensity
Seq <- seq(1,32,1)
xy <- expand.grid(Seq,Seq)
xy <- as.matrix(xy)
tab_R <- c()
for(i in 1:N_tab_Z){
   tab_R[i] <- Rain_int(tab_Nt[i],tab_L[i],tab_D,tab_V)
}
Sim_R_map <- cbind(xy,tab_R)

## We plot the rain intensity
M <- Sim_R_map
M[,1:2] <- M[,1:2]
M[,3] <- round(M[,3])
Seq         <- seq(1,32,1)
Seqx        <- Seq
Seqy        <- Seq
dx          <- 1
dy          <- 1
nSeqx       <- length(Seqx)
nSeqy       <- length(Seqy)
nxy         <- nSeqx*nSeqy
nSeq        <- length(Seq)
xinf <- M[1,1]-dx
xsup <- M[nxy,1]+(M[nxy,1]-xinf)/8.5
yinf <- M[1,2]-dx
ysup <- M[nxy,2]+(M[nxy,2]-yinf)/15
xlim <- c(xinf,xsup)
ylim <- c(yinf,ysup)
pdf(sprintf("/home/mschleis/Mediterranean Intense/Simulated_R_1km.pdf",time),h=6,w=6)
min <- 0
max <- 100
par(mai=c(1,1,0.25,0.4))
plot(M[,1],M[,2],"n",xlab="Distance [km]",ylab="Distance [km]",xlim=xlim,ylim=ylim,cex.lab=1.1,bty="n",xaxp=c(0,32,8),yaxp=c(0,32,8))
plot_map2(M,Seqx,Seqy,c(dx,dy),c(min,max),xlim,ylim,color_ramp,color_ramp2,5,-99)
text(M[nxy,1]+(M[nxy,1]-xinf)/13,M[nxy,2]+(M[nxy,2]-yinf)/15,"mm/h")
text((xsup-xinf)/2,ysup,"Simulated Rain Rate",cex=1.4)
dev.off()








pdf(sprintf("/home/mschleis/Mediterranean Intense/Simulated_R_1km.pdf",time),h=6,w=6)
min <- 0
max <- 100
par(mai=c(1,1,0.25,0.4))
plot(M[,1],M[,2],"n",xlab="Distance [km]",ylab="Distance [km]",xlim=xlim,ylim=ylim,cex.lab=1.1,bty="n",xaxp=c(0,32,8),yaxp=c(0,32,8))
plot_map2(M,Seqx,Seqy,c(dx,dy),c(min,max),xlim,ylim,color_ramp,color_ramp2,5,-99)
text(M[nxy,1]+(M[nxy,1]-xinf)/13,M[nxy,2]+(M[nxy,2]-yinf)/15,"mm/h")
text((xsup-xinf)/2,ysup,"Simulated Rain Rate",cex=1.4)
dev.off()




## The histogram of the simulated field is computed
# title <- "simulated radar reflectivity"
# xlab  <- "reflectivity [dBZ]"
# pdf("SimH.pdf")
# myH <- my_histogram(tab_Z,min(tab_Z),max(tab_Z),10,title,xlab)
# dev.off()


## We compute the variogram of the reflectivity field

## title <- "Variogram of Simulated Reflectivity [dBZ]"
## xlab  <- "distance [m]"
## pdf("SimVario.pdf")
## radar_variogram <- irregular_grid_variogram(Z_data_matrix,radar_dr,radar_nx*radar_dr,title,xlab,20000,70)
## dev.off()


