### ALL FONCTIONS MADE BY FLORIAN PANTILLON FOR 2DVD AND PARSIVEL ANALYSIS (2009) ###
# File made by Yann Chavaillaz, February 10th, 2010.

path_data_parsivel <- "/USERS/lte/Yann/ANALYSE_2DVD_PARSIVEL/"
path_data_2DVD <- "/USERS/lte/Yann/ANALYSE_2DVD_PARSIVEL/"
path_data_raingauge <- "/USERS/lte/Yann/ANALYSE_2DVD_PARSIVEL/"

path_scripts_libR <- "/USERS/lte/Prog_com/Lib_R/"
#path_scripts_parsivel <- "/USERS/pantillo/Parsivel/scripts/"
#path_scripts_2DVD <- "/USERS/pantillo/2DVD/scripts/"
#path_scripts_raingauge <- "/USERS/pantillo/RainGauge/scripts/"

source(paste(path_scripts_libR,"new_lib_DSD.R",sep=""))
source(paste(path_scripts_libR,"lib_Gamma.R",sep=""))



### Parsivel size and velocity classes -------------------------------------------

size_average <- c(0.062, 0.187, 0.312, 0.437, 0.562, 0.687, 0.812, 0.937,
                  1.062, 1.187, 1.375, 1.625, 1.875, 2.125, 2.375, 2.750,
                  3.250, 3.750, 4.250, 4.750, 5.500, 6.500, 7.500, 8.500,
                  9.500,11.000,13.000,15.000,17.000,19.000,21.500,24.500)

size_width <- c(rep(0.125,10),rep(0.25,5),rep(0.5,5),rep(1.,5),rep(2.,5),rep(3.,2))

classes_size <- c(0, size_average+size_width/2)

velocity_average <- c(0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75,
                      0.85, 0.95, 1.10, 1.30, 1.50, 1.70, 1.90, 2.20,
                      2.60, 3.00, 3.40, 3.80, 4.40, 5.20, 6.00, 6.80,
                      7.60, 8.80,10.40,12.00,13.60,15.20,17.60,20.80)

velocity_width <- c(rep(0.1,10),rep(0.2,5),rep(0.4,5),rep(0.8,5),rep(1.6,5),
                    rep(3.2,2))

classes_velocity <- c(0, velocity_average+velocity_width/2)

# color palettes

jet.colors <- colorRampPalette(c("#00007F", "blue", "#007FFF", "cyan", "#7FFF7F", 
                                 "yellow", "#FF7F00", "red", "#7F0000"))

sign.colors <- colorRampPalette(c("blue","grey","red"))

### -------------------------------------------------------------------------------

### computes Parsivel axis ratio after Yuter et al. 2006 --------------------------
### size is computed equivalent diameter and NOT measured maximum diameter
### i.e. inverse relation must be used by Parsivel to compute equivalent diameter

axis_ratio_parsivel <- function(size) {

if (size<=1) {
  axis_ratio_parsivel <- 1
} else if (size>1 && size<5) {
  axis_ratio_parsivel <- 1 + (3-3*size)/52
} else if (size>=5) {
  axis_ratio_parsivel <- 1/1.3
} else {
  axis_ratio_parsivel <- NaN
}

}

### -------------------------------------------------------------------------------

### function compute_gamma --------------------------------------------------------
### 
### fits PSD=log10(n(D)[#/m^3/mm]) for Parsivel size and velocity classes
### with gamma distribution using lib_Gamma.R
### 
### optional argument: filter=logical vector in order to fit only PSD[filter,]
### e.g. filter=nb_particles>threshold
### 
### returns fit parameters, Cumulative Density Function and fitted PSD

compute_gamma <- function(PSD,filter=TRUE) {

n_data <- dim(PSD)[1]

print(paste('Fitting gamma distributions to',n_data,'PSDs'))

gamma_param <- array(NaN,c(n_data,3))
gamma_Nt <- array(NaN,c(n_data,1))
gamma_CDF <- array(NaN,c(n_data,33))
gamma_PSD <- array(NaN,c(n_data,32))

for (i in 1:n_data) {
  if (filter[i]) {
    gamma_param[i,] <- Newton_MLE_Gamma(10^PSD[i,]*size_width,size_average)
    gamma_Nt[i] <- sum(10^PSD[i,]*size_width) # total particle concentration
    gamma_CDF[i,] <- pgamma(size_classes,gamma_param[i,1],gamma_param[i,2])
    gamma_PSD[i,] <- log10(diff(gamma_CDF[i,])*gamma_Nt[i]/size_width)
  }
}

compute_gamma <- data.frame(parameters=I(gamma_param),Nt=gamma_Nt,
                            CDF=I(gamma_CDF),PSD=I(gamma_PSD))

}

### ---------------------------------------------------------------------------------

### function compute_PSD ------------------------------------------------------------
### 
### computes PSD=log10(n(D)[#/m^3/mm]) from Parsivel particle distribution (RawData)
### using area=54cm^2, time=20s and standard size and velocity classes

compute_PSD <- function(distribution,n_interval=1) {

area <- 54e-4 # [m^2]
time_interval <- 20*n_interval # [s]

tmp <- numeric(32)
for (i in 1:32) tmp[i] <- sum(distribution[i,]/velocity_average)/
                          size_width[i]/area/time_interval

compute_PSD <- log10(tmp)

}

### -------------------------------------------------------------------------------

### function global_values --------------------------------------------------------
### 
### computes total particle distribution, average PSD=log10(n(D)[#/m^3/mm])
### and average mean_velocity=v(D)[m/s]
### 
### optional argument: filter=logical vector e.g. filter=(time>ti & time<tf)

global_values <- function(distribution,PSD,mean_velocity,filter=TRUE) {

global_distribution <- colSums(distribution[filter,,])
global_PSD <- log10(colMeans(10^PSD[filter,]))
global_mean_velocity <- colMeans(mean_velocity[filter,],na.rm=TRUE)

global_values <- data.frame(distribution=I(global_distribution),
                         PSD=global_PSD,mean_velocity=global_mean_velocity)

}

### ------------------------------------------------------------------------------

### function read_2DVD -----------------------------------------------------------
###
### reads 2DVD data from ASCII file 'filename'
### returns data.frame containing standard information on each particle

# written by Florian Pantillon, 2009-04-16
# Last modification: ...

read_2DVD <- function(filename) {

# read data

print(paste("Reading 2DVD data file",filename))

header <- scan(file=filename,skip=1,nlines=1,what='list',quiet=TRUE)

data <- read.table(file=filename,skip=3,header=FALSE)

names(data) <- header

# extract date and precipitation type from filename

l <- nchar(filename)
date <- substr(filename,l-14,l-10) # year and Julian day
ext <- substr(filename,l-6,l-4) # 'hyd' or 'sno'

# convert strings to POSIX

tmp <- strptime(paste(date,data$TIMESTAMP),format="%y%j %H:%M:%S",tz="GMT")
data$TIMESTAMP <- as.POSIXct(tmp)

# perform operations depending on precipitation type

## A compléter!
if (ext=='hyd') {

} else if (ext=='sno') {

} else {

}

# return data.frame

read_2DVD <- data

}

### ---------------------------------------------------------------------------------------

### function read_MCH ---------------------------------------------------------------------
###
### reads MeteoSwiss rain gauge data of CLIMAP server from txt file 'filename'
### optional arguments: 'time_start' and 'time_end' to select time interval in UTC
### returns data.frame containing filtered and formatted read data

read_MCH <- function(filename,time_start='2000-01-01',time_end='2020-01-01') {

print(paste("Reading MeteoSwiss rain gauge data file",filename))

# convert strings to POSIX

time_start <- as.POSIXct(time_start,tz='GMT')
time_end <- as.POSIXct(time_end,tz='GMT')

# read data

data <- read.table(file=filename,skip=9,header=FALSE)

# convert time columns to POSIX

timestamp <- ISOdatetime(data[,2],data[,3],data[,4],data[,5],data[,6],
             sec=0,tz='GMT')

# select time interval

filter_time <- timestamp>=time_start & timestamp<=time_end
data <- data[filter_time,]
timestamp <- timestamp[filter_time]

# compute intensity

intensity <- data[,7]*60/10 # [mm/10min] to [mm/h]

# compute accumulation

accumulation <- diffinv(data[2:dim(data)[1],7]) # integrate 10 minutes values

# return data.frame

read_MCH <- data.frame(timestamp,accumulation,intensity)

}

### ---------------------------------------------------------------------------------------

### function read_parsivel ----------------------------------------------------------------
###
### reads Parsivel data from ASCII file 'filename'
### optional arguments: 'time_start' and 'time_end' to select time interval in UTC
###                     'format_data' can be set to FALSE to save time and memory
### returns data.frame containing filtered and formatted read data

read_parsivel <- function(filename,time_start='2000-01-01',time_end='2020-01-01',
                          format_data=TRUE) {

# convert strings to POSIX

time_start <- as.POSIXct(time_start,tz='GMT')
time_end <- as.POSIXct(time_end,tz='GMT')

# read data file

print(paste("Reading Parsivel data file",filename))

header <- scan(file=filename,skip=1,nlines=1,sep=",",what='list',quiet=TRUE)

data <- read.table(file=filename,sep=",",skip=4,header=FALSE)

names(data) <- header

# filter data

print(paste("Selecting data without errors between",time_start,"and",time_end))

data$TIMESTAMP <- as.POSIXct(data$TIMESTAMP,tz="GMT") # convert string to POSIX

no_error <- c(TRUE,diff(data$CommErrorCount)==0) # select data without errors

filter_time <- data$TIMESTAMP>=time_start & 
               data$TIMESTAMP<=time_end # select time interval

data <- data[no_error & filter_time, ] # keep selected data only

# format data

n_rows <- dim(data)[1]

if (n_rows>0 && format_data) { # selected data not void and format_data==TRUE

  print("Formatting data")

  # coerce data to integer to save memory

  data$RECORD <- as.integer(data$RECORD)
  data$Code4680 <- as.integer(data$Code4680)
  data$Code4677 <- as.integer(data$Code4677)
  data$Visibility <- as.integer(data$Visibility)
  data$LaserAmplitude <- as.integer(data$LaserAmplitude)
  data$NumberOfParticles <- as.integer(data$NumberOfParticles)
  data$Temperature <- as.integer(data$Temperature)
  data$Status <- as.integer(data$Status)
  data$CommErrorCount <- as.integer(data$CommErrorCount)

  # if field exists, change name to Transmission

  if (length(data$Error)>0) {
    data$Transmission <- data$Error
    data$Error <- NULL
  }

  # extract value from string

  tmp <- as.character(data$Transmission)
  data$Transmission <- as.numeric(substr(tmp,6,nchar(tmp)-2))

  # extract vector from string

  new_FieldN <- array(-Inf,c(n_rows,32))
  for (i in 1:n_rows) {
    tmp <- strsplit(as.character(data$FieldN[[i]]),split=',')
    new_FieldN[i,] <- as.numeric(tmp[[1]])
  }
  new_FieldN[new_FieldN == -9.999] <- -Inf # replace -9.999 by log10(0)=-Inf
  data$FieldN <- new_FieldN

  # extract vector from string

  new_Fieldv <- array(NaN,c(n_rows,32))
  for (i in 1:n_rows) {
    tmp <- strsplit(as.character(data$Fieldv[[i]]),split=',')
    new_Fieldv[i,] <- as.numeric(tmp[[1]])
  }
  new_Fieldv[new_Fieldv == 0.000] <- NaN # replace 0.000 by mean(VOID)=NaN
  data$Fieldv <- new_Fieldv

  # extract matrix from string and change field name from RowData to RawData

  RawData <- array(as.integer(0),c(n_rows,32,32))
  for (i in 1:n_rows) {
    tmp <- strsplit(as.character(data$RowData[[i]]),split=',')
    RawData[i,,] <- as.integer(tmp[[1]])
  }
  data$RowData <- NULL
  data$RawData <- RawData

}

# return data.frame

read_parsivel <- data

}

### ---------------------------------------------------------------------------------

### function read_SLF ---------------------------------------------------------------
###
### reads SLF data of GSN server from csv file 'filename'
### optional arguments: 'time_start' and 'time_end' to select time interval in UTC
### returns data.frame containing filtered and formatted read data

read_SLF <- function(filename,time_start='2000-01-01',time_end='2020-01-01') {

print(paste("Reading SLF data file",filename))

# convert strings to POSIX

time_start <- as.POSIXct(time_start,tz='GMT')
time_end <- as.POSIXct(time_end,tz='GMT')

# read file header

header <- scan(file=filename,skip=2,nlines=1,sep=",",what='list',quiet=TRUE)
tmp <- header[1]
header[1] <- substr(tmp,2,nchar(tmp)) # remove character '#' which begins first field

# read data

data <- read.table(file=filename,sep=",",skip=3,header=FALSE,na.strings='null')

names(data) <- header

# reverse order

n_data <- dim(data)[1]
data[1:n_data,] <- data[n_data:1,]

# convert time string to POSIX

tmp <- strptime(data$timed,format="%Y-%m-%dT%H:%M:%S",tz="GMT")
data$timestamp <- as.POSIXct(tmp)-3600
data$timed <- NULL

# select time interval

filter_time <- data$timestamp>=time_start & data$timestamp<=time_end 
data <- data[filter_time,]

# return data.frame

read_SLF <- data

}

### -----------------------------------------------------------------------------------

### function rectify_accumulation -----------------------------------------------------
###
### sets first accumulation value to 0 and rectifies accumulation resettings
### use this function before comparing accumulation from different instruments
### 
### optional argument: resetting threshold (for Parsivel: threshold=300[mm])
### 
### returns rectified accumulation

rectify_accumulation <- function(accumulation,threshold=0) {

n_acc <- length(accumulation)

new_acc <- accumulation - accumulation[1] # remove level

for (i in 2:n_acc) if (is.na(new_acc[i])) new_acc[i] <- new_acc[i-1] # remove NAs

d_acc <- c(0,diff(new_acc))

for (i in 2:n_acc) {
  if (d_acc[i]<0) { # rectify
    if (threshold>0) { # known threshold
      new_acc[i:n_acc] <- new_acc[i:n_acc] + threshold
    } else { # unknown threshold
      new_acc[i:n_acc] <- new_acc[i:n_acc] - d_acc[i]
    }
  }
}

rectify_accumulation <- new_acc

}

### -----------------------------------------------------------------------------------

### function resample_accumulation ----------------------------------------------------
###
### resample accumulation according to timestamp 'time_sampling'
### use this function before comparing accumulation and intensity
### from different instruments
### accumulation must first be rectified with rectify_accumulation
### 
### returns timestamp, accumulation and intensity computed from accumulation

resample_accumulation <- function(time,accumulation,time_sampling) {

print(paste("Resampling precipitation accumulation between",min(time_sampling),
            "and",max(time_sampling)))

# compute length of data

n_data <- length(time)
n_sampling <- length(time_sampling)

# initialise arrays

new_accumulation <- numeric(n_sampling)
new_intensity <- numeric(n_sampling)

# convert POSIXct time to numeric for much faster calculation

time <- as.numeric(time)
new_time <- time_sampling # for output
time_sampling <- as.numeric(time_sampling)

# initialize max i so that time[i] <= time_sampling[1]

i <- 1
while (time[i+1]<=time_sampling[1] & i<n_data) i <- i+1

# start resampling

for (j in 2:n_sampling) {

  # find max i so that time[i] <= time_sampling[j]

  i0 <- i
  while (time[i+1]<=time_sampling[j] & i<n_data) i <- i+1

  if (i>i0) { # at least one measurement in time interval

    new_accumulation[j] <- accumulation[i]

  } else { # no measurement in time interval
    new_accumulation[j] <- new_accumulation[j-1]
  }

}

# compute new intensity

d_time <- diff(time_sampling) # compute time increments
new_intensity <- c(0,diff(new_accumulation)/d_time)*3600

# return timestamp [POSIXct], accumulation [mm], intensity [mm/h]

resample_accumulation <- data.frame(timestamp=new_time,accumulation=new_accumulation,
           intensity=new_intensity)

}

### ---------------------------------------------------------------------------------------

### function resample_parsivel ------------------------------------------------------------
###
### resample most useful Parsivel fields according to timestamp 'time_sampling'
### accumulation must first be rectified with rectify_accumulation
###
### returns resampled fields and intensity computed from accumulation

resample_parsivel <- function(time,accumulation,nb_particles,
                              distribution,PSD,mean_velocity,time_sampling) {

print(paste("Resampling Parsivel data between",min(time_sampling),
            "and",max(time_sampling)))

# compute length of data

n_data <- length(time)
n_sampling <- length(time_sampling)

# initialise arrays

new_accumulation <- numeric(n_sampling)
new_intensity <- numeric(n_sampling)
new_nb_particles <- numeric(n_sampling)
new_distribution <- array(0,c(n_sampling,32,32))
new_PSD <- array(-Inf,c(n_sampling,32))
new_mean_velocity <- array(NaN,c(n_sampling,32))

# convert POSIXct time to numeric for much faster calculation

time <- as.numeric(time)
new_time <- time_sampling # for output
time_sampling <- as.numeric(time_sampling)

# initialize max i so that time[i] <= time_sampling[1]

i <- 1
while (time[i+1]<=time_sampling[1] & i<n_data) i <- i+1

# start resampling

for (j in 2:n_sampling) {

  # find max i so that time[i] <= time_sampling[j]

  i0 <- i
  while (time[i+1]<=time_sampling[j] & i<n_data) i <- i+1

  if (i>i0+1) { # more than one measurement in time interval

    new_accumulation[j] <- accumulation[i]
    new_nb_particles[j] <- sum(nb_particles[(i0+1):i])

    new_distribution[j,,] <- colSums(distribution[(i0+1):i,,])
    new_PSD[j,] <- log10(colMeans(10^PSD[(i0+1):i,]))
    new_mean_velocity[j,] <- colMeans(mean_velocity[(i0+1):i,],na.rm=TRUE)

  } else if (i==i0+1) { # exactly one measurement in time interval

    new_accumulation[j] <- accumulation[i]
    new_nb_particles[j] <- nb_particles[i]

    new_distribution[j,,] <- distribution[i,,]
    new_PSD[j,] <- PSD[i,]
    new_mean_velocity[j,] <- mean_velocity[i,]

  } else { # no measurement in time interval

    new_accumulation[j] <- new_accumulation[j-1]

  }

}

# compute new intensity

d_time <- diff(time_sampling) # compute time increments
new_intensity <- c(0,diff(new_accumulation)/d_time)*3600

# return resampled variables
# with I() preserving matrices to be converted to individual columns

output <- data.frame(timestamp=new_time,accumulation=new_accumulation,
                     intensity=new_intensity,nb_particles=new_nb_particles,
                     PSD=I(new_PSD),mean_velocity=I(new_mean_velocity))

output$distribution <- new_distribution # preserve array structure

resample_parsivel <- output

}

### ----------------------------------------------------------------------------------

### function sample_2DVD -------------------------------------------------------------
###
### samples 2DVD data of individual particles
### contained in data.frame 'data_2DVD' with format as returned by read_2DVD.R
### after timestamp given by 'time_sampling'
### and in classes given by 'classes_size' and classes_velocity'
### 
### returns data.frame containing sampled precipitation data

# written by Florian Pantillon, 2009-04-16
# Last modification: ...

sample_2DVD <- function(data_2DVD,time_sampling,size_classes,velocity_classes) {

print(paste("Sampling 2DVD data between",min(time_sampling),
            "and",max(time_sampling)))

# compute length of data

n_2DVD <- dim(data_2DVD)[1]
n_sampling <- length(time_sampling)
n_size <- length(classes_size)-1
n_velocity <- length(classes_velocity)-1

# initialise arrays

accumulation <- numeric(n_sampling)
intensity <- numeric(n_sampling)
nb_particles <- numeric(n_sampling)
distribution <- array(0,c(n_sampling,n_size,n_velocity))
PSD <- array(-Inf,c(n_sampling,n_size))
mean_velocity <- array(NaN,c(n_sampling,n_size))

# convert POSIXct time to numeric for much faster calculation

time_2DVD <- as.numeric(data_2DVD$TIMESTAMP)
timestamp <- time_sampling # for output
time_sampling <- as.numeric(timestamp)

# compute time increments and size widths

d_time <- diff(time_sampling)
d_size <- diff(classes_size)

# initialize i so that time_2DVD[i] > time_sampling[1]

i <- 1
while (time_2DVD[i]<time_sampling[1] & i<n_2DVD) {
  i <- i+1
}

# start sampling

for (j in 2:n_sampling) {

  # find i so that time_2DVD[i] > time_sampling[j]

  i0 <- i
  while (time_2DVD[i]<time_sampling[j] & i<n_2DVD) i <- i+1

  if (i>i0) { # at least one particle in time interval

    # select data between time_sampling[j-1] and time_sampling[j]

    i1 <- i-1
    sel_vol <- data_2DVD$VOL[i0:i1]
    sel_area <- data_2DVD$AREA[i0:i1]
    sel_diam <- data_2DVD$DIAM[i0:i1]
    sel_vel <- data_2DVD$VEL[i0:i1]

    # compute prec. accumulation [mm] and intensity [mm/h], and number of particles

    add <- sum(sel_vol/sel_area)
    accumulation[j] <- accumulation[j-1]+add
    intensity[j] <- add/d_time[j-1]*3600
    nb_particles[j] <- i1+1-i0

    # find velocity class of each particle

    filter_velocity <- array(FALSE,c(i1+1-i0,n_velocity))
    for (l in 1:n_velocity) {
      filter_velocity[,l] <- sel_vel>=classes_velocity[l] &
                             sel_vel<classes_velocity[l+1]
    }

    # compute distribution, PSD and mean_velocity

    for (k in 1:n_size) {
      filter_size <- sel_diam>=classes_size[k] & sel_diam<classes_size[k+1]
      distribution[j,k,] <- colSums(filter_size & filter_velocity)
      PSD[j,k] <- log10(sum(1/sel_vel[filter_size]/sel_area[filter_size])/
                  d_size[k]/d_time[j-1]*1.e6) # 1 [mm^2] = 1.e-6 [m^2]
      mean_velocity[j,k] <- mean(sel_vel[filter_size])
    }

  } else { # no particle in time interval
    accumulation[j] <- accumulation[j-1]
  }

}

# return time [POSIXct], accumulation [mm], intensity [mm/h], nb of particles,
# distribution [n_size x n_diameter], PSD [n_size], mean_velocity [n_size]
# with I() preserving matrices to be converted to individual columns

output <- data.frame(timestamp,accumulation,intensity,nb_particles,
                     I(PSD),I(mean_velocity))

output$distribution <- distribution # preserve array structure

sample_2DVD <- output

}

### ------------------------------------------------------------------------------

### plots measured and fitted PSDs vs both time and diameter ---------------------

# Dmax modified by Yann Chavaillaz (now -> 11, before -> 5), 2010-02-10.
show_gamma_PSD <- function(timestamp,PSD_observations,PSD_gamma,Dmax=11) {

#dev.new() modified by Y. C., 2010-02-10.
postscript(paste("/USERS/lte/Yann/ANALYSE_2DVD_PARSIVEL/Observation_PSD",video_filename,'.eps'),horizontal=FALSE,onefile=FALSE,bg="white",width=8,height=6.5)
image(as.numeric(timestamp),classes_size,PSD_observations,zlim=c(0,5),ylim=c(0,Dmax),
      axes=FALSE,xlab='Time',ylab='Diameter [mm]',col=jet.colors(50),
      main=paste('Observations: log10 ( n(D) [#/m^3/mm] )', new_time[i_max]))
axis.POSIXct(1,x=timestamp,format='%Y-%m-%d\n%H:%M %Z')
axis(2)
box()
abline(h=classes_size[3])
dev.off()

#dev.new() modified by Y. C., 2010-02-10.
postscript(paste("/USERS/lte/Yann/ANALYSE_2DVD_PARSIVEL/Gamma_PSD",video_filename,'.eps'),horizontal=FALSE,onefile=FALSE,bg="white",width=8,height=6.5)
image(as.numeric(timestamp),classes_size,PSD_gamma,zlim=c(0,5),ylim=c(0,Dmax),
      axes=FALSE,xlab='Time',ylab='Diameter [mm]',col=jet.colors(50),
      main=paste('Gamma: log10 ( n(D) [#/m^3/mm] )', new_time[i_max]))
axis.POSIXct(1,x=timestamp,format='%Y-%m-%d\n%H:%M %Z')
axis(2)
box()
abline(h=classes_size[3])
dev.off()

#dev.new() modified by Y. C., 2010-02-10.
postscript(paste("/USERS/lte/Yann/ANALYSE_2DVD_PARSIVEL/Gamma_Observation_PSD",video_filename,'.eps'),horizontal=FALSE,onefile=FALSE,bg="white",width=8,height=6.5)
image(as.numeric(timestamp),classes_size,PSD_gamma-PSD_observations,zlim=c(-1,1),
      ylim=c(0,Dmax),axes=FALSE,xlab='Time',ylab='Diameter [mm]',col=sign.colors(50),
      main=paste('Gamma - Observations: log10 ( n(D) [#/m^3/mm] )', new_time[i_max]))
axis.POSIXct(1,x=timestamp,format='%Y-%m-%d\n%H:%M %Z')
axis(2)
box()
abline(h=classes_size[3])
dev.off()

}

### ------------------------------------------------------------------------------

### plots Weissfluhjoch meteorological parameters---------------------------------

show_meteo_WFJ <- function(timestamp,wind_speed_average,wind_speed_max,
                           wind_direction,temperature,humidity) {

dev.new()
plot(timestamp,wind_speed_average,type='l',xlab='Time',
     ylab='Wind Speed [m/s]',ylim=c(0,max(wind_speed_max)),axes=FALSE)
lines(timestamp,wind_speed_max,col='blue')
legend('topleft',legend=c('30 Minutes Average','30 Minutes Maximum'),
       col=c('black','blue'),lty=1)
axis.POSIXct(1,x=timestamp,format='%Y-%m-%d\n%H:%M %Z')
axis(2)
box()

dev.new()
plot(timestamp,wind_direction,type='l',xlab='Time',
     ylab='Wind Direction [°]',ylim=c(0,360),axes=FALSE)
axis.POSIXct(1,x=timestamp,format='%Y-%m-%d\n%H:%M %Z')
axis(2,at=c(0,90,180,270,360))
box()

dev.new()
plot(timestamp,temperature,type='l',xlab='Time',
     ylab='Air Temperature [°C]',axes=FALSE)
axis.POSIXct(1,x=timestamp,format='%Y-%m-%d\n%H:%M %Z')
abline(h=0,col='red')
axis(2)
box()

dev.new()
plot(timestamp,humidity,type='l',xlab='Time',
     ylab='Relative Humidity [%]',axes=FALSE)
axis.POSIXct(1,x=timestamp,format='%Y-%m-%d\n%H:%M %Z')
abline(h=100,col='red')
axis(2)
box()

}

### ---------------------------------------------------------------------------