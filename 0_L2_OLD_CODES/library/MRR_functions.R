# MRR_functions.R
#
# Functions for reading and plotting micro-rain-radar data.
#
# Author: Tim Raupach <tim.raupach@epfl.ch>

library(ncdf4)
library(data.table)
library(ggplot2)
source("library/timeseries_functions.R")
source("library/DSD_functions.R")

retrieveMRRVariables = function(nc, variables, recordIdx=NULL) {
  # Retrieve variables from an MRR netCdf file and package them
  # in a data.table.
  #
  # Args:
  #   nc: The open nc file.
  #   variables: Which variables to retrieve.
  #   recordIdx: Indexes of records to read (default: all records).
  # 
  # Returns: data.table with variable information.
  
    times = as.POSIXct(ncvar_get(nc, "Time"), tz="UTC", 
        format="%y%m%d%H%M%S")
    
    if(is.null(recordIdx)) {
        recordIdx = seq(1, length(times))
    }
    
    ## Subset the list of times.
    times = times[recordIdx]
    
    ## Get list of heights above the instrument [m].
    heights = ncvar_get(nc, "H")
    
    ## Get missing data value.
    missingVal = ncatt_get(nc, 0, "MissingData")$value
    
    ## Get each requested variable by height.
    allvars = NULL
    for(varName in variables) {
        ## Variable will return as matrix with dimensions [height, time].
        var = ncvar_get(nc, varName)
        dims = length(dim(var))
        
        if(dims == 3) {
            ## 3 dimensional variables are of form [bin, height, time].
      
            ## Subset to requested times.
            var = var[, , recordIdx]
            
            ## Transform to a (T * H) x B matrix, where each row is a
            ## time and height combo and each column is a bin.
            m = matrix(var, nrow=length(recordIdx)*length(heights), 
                ncol=dim(var)[1], byrow=T)

            m = data.frame(m)
            nms = paste(varName, "_", seq(1:dim(var)[1]), sep="")
            names(m) = nms

            ## Make a data.table out of it.
            var = data.table(m)

            ## Replace all missing values with NAs.
            for(n in seq_len(ncol(var))) 
                set(var, which(var[[n]] == missingVal), n, NA)
            
            ## Special instructions for DSDs.
            if(varName == "Nnn") {
                ## For DSD concentrations, replace missing values and negatives with zeros.
                for(n in seq_len(ncol(var)))
                    set(var, which(is.na(var[[n]]) | var[[n]] < 0), n, 0)
                
                ## For the DSD, apply a factor of 0.001 to change the units to mm^-1 m^-3.
                var = var * 0.001 
            } 
            
            ## Make a data.table containing time, height, and measurement.
            d = data.table(POSIXtime=rep(times, each=length(heights)),
                height=rep(heights, length(times)),
                var, key=c("POSIXtime", "height"))
        } else if (dims == 2) {
            ## Subset to requested times.
            var = var[, recordIdx]
            
            ## Make a data.table containing time, height, and measurement.
            d = data.table(POSIXtime=rep(times, each=length(heights)), 
                height=rep(heights, length(times)),
                var=as.vector(var), key=c("POSIXtime", "height"))
            
            ## Replace NA values with real NAs.
            d[var == missingVal, var := NA]
            setnames(d, "var", varName)      
        } else {
            stop(paste("Unexpected one-dimensional variable", varName))
        }
        
        ## Merge into the data.table to return.
        if(is.null(allvars)) {
            allvars = d
        } else {
            allvars = merge(d, allvars, by=c("POSIXtime", "height"), all=TRUE)
        }
    } 
    
    return(allvars)
}

readMRRClosestTimes = function(dir, times, variables, maxAllowedTimeDiff=30) {
  # Read MRR data for the closest times to a given set of times.
  #
  # Args:
  #   dir: Input directory.
  #   times: POSIXct UTC times for requested times.
  #   variables: List of variables to read.
  #   maxAllowedTimeDiff: The maximum allowed time difference between 
  #                       requested and closest time [s] (default: 30 s).
  #  
  # Returns: data.table of data per time, with requestedTime as requested time 
  # and MRRtime as the corresponding instrument time. Note that no data
  # are returned for times that are not matched in the MRR data.

  times = data.table(t=times)
  
  # Get one file per required day.
  requiredDates = strftime(unique(trunc(times$t, "day")), "%Y%m%d")
  fileList = NULL
  for(date in requiredDates) {
    dayFile = list.files(dir, recursive=T, pattern=date, full.names=T)
    if(length(dayFile) > 0)
      fileList = rbind(fileList, dayFile)
  }
  
  # Read each file in turn.
  allData = NULL
  for(file in fileList) {
    print(file)
    nc = nc_open(file)
    
    # Get the measurement times [UTC].
    MRRtimes = as.POSIXct(ncvar_get(nc, "Time"), tz="UTC", 
                          format="%y%m%d%H%M%S")
    
    dayTimes = times
    dayTimes[, minTimeIdx := 
            which.min(abs(as.numeric(difftime(as.POSIXct(t, tz="UTC", 
                                                         origin="1970-1-1"), 
                                              MRRtimes)))), by=t]
    dayTimes = dayTimes[, list(requestedTime=t, MRRtimestamp=MRRtimes[minTimeIdx])]
    dayTimes[, timeDiff := as.numeric(difftime(requestedTime, MRRtimestamp))]
    dayTimes = dayTimes[abs(timeDiff) <= maxAllowedTimeDiff]
        
    # Get the times for which we have records.
    getTimes = which(MRRtimes %in% dayTimes$MRRtimestamp)
    
    # Get the variables for these times.
    allvars = retrieveMRRVariables(nc=nc, variables=variables, 
                                   recordIdx=getTimes)
    
    # Make POSIXtime the requested time and MRR time the instrument timestamp.
    setnames(allvars, "POSIXtime", "MRRtime")
    allvars[, requestedTime := dayTimes[MRRtimestamp == MRRtime, 
                                    requestedTime], by=MRRtime]
    allData = rbind(allData, allvars)
    nc_close(nc)
  }
  
  return(allData)
}

readMRRVariables = function(dir, start, end, variables, 
                            resample=FALSE, res="5 min", 
                            resStart=NULL) {
  # Read MRR variables from NetCDF files, for a given time period. For 
  # variables in which there is more than one value per height,
  # the values are split into separate columns. Eg for the DSD (Nnn),
  # there will be columns for Nnn_1 up to Nnn_64.
  #
  # For speed this function is able to resample as it reads in each 
  # day's data.
  # 
  # Args:
  #   dir: Input directory.
  #   start, end: POSIXct UTC times for start and end of time period.
  #   variables: List of variables to read.
  # 
  # Returns: data.table containing each variable by POSIXtime, height.
  
  # Get one file per required day.
  requiredDates = strftime(trunc(seq(start, end, by="1 day"), "day"), "%Y%m%d")
  fileList = NULL
  for(date in requiredDates) {
    dayFile = list.files(dir, recursive=T, pattern=date, full.names=T)
    if(length(dayFile) > 0)
      fileList = rbind(fileList, dayFile)
  }
    
  # Read each file in turn.
  allData = NULL
  for(file in fileList) {
    print(file)
    nc = nc_open(file)
    
    # Get the measurement times [UTC].
    times = as.POSIXct(ncvar_get(nc, "Time"), tz="UTC", 
                       format="%y%m%d%H%M%S")
    
    # Which records correspond to the times we want?
    recordIdx = which(times >= start & times < end & !duplicated(times))
            
    # Get the variables for these times.
    allvars = retrieveMRRVariables(nc=nc, variables=variables, 
                                   recordIdx=recordIdx)
    
    # Resample if required.
    if(resample) {
      allvars = resampleMRRData(d=allvars, res=res, start=resStart)
    }
    
    allData = rbind(allData, allvars)
    nc_close(nc)
  }
  
  return(allData)
}
 
plotMRRTimeseries = function(d, var, varName, varUnit, textSize=16, 
                             zlims = d[, range(.SD, na.rm=T), .SDcols=var],
                             ncolours=200, velocity=FALSE) {
  # Define a set of colours to use.
  col.Rbar = c("darkblue","blue3","blue1","dodgerblue","deepskyblue","cyan")
  col.Rbar = c(col.Rbar,"yellow","gold","orange","red1","red3","darkred")
  if(velocity) {
    col.Rbar = c("blue3", "white", "red3")
  }
  col.Rbar = colorRampPalette(col.Rbar)
  col.Rbar = col.Rbar(ncolours)
  
  scaleName = 
    parse(text=paste(varName, "~group('[',", varUnit, ",']')", sep=""))
  
  scale = scale_fill_gradientn(colours=col.Rbar, name=scaleName,
                               na.value="white", limits=zlims)
  lineScale = scale_colour_gradientn(colours=col.Rbar, name=scaleName,
                                     na.value="white", limits=zlims)
  
  plot = ggplot(d, aes(x=as.numeric(factor(POSIXtime)), y=height)) +
    geom_raster(aes_string(fill=var, colour=var)) +
    theme_bw(textSize) + scale + lineScale +
    labs(x=paste("Timestep after", d[, strftime(min(POSIXtime))]), 
         y="Height [m]", title=paste("MRR", varName)) +
    scale_y_discrete(breaks=d[, seq(0, max(height), by=200)])
  return(plot)
}

resampleMRRData = function(d, res, start, logVariables=c("Z", "z")) {
  # Resample MRR data to a new time resolution.
  # 
  # Args:
  #   d: MRR data as a data.table.
  #   res: New time resolution, eg "5 min".
  #   start: POSIXct UTC start time.
  #   logVariables: Which variables are expected to be in logarithmic units?
  # 
  # Returns: data.table resampled, all columns other than POSIXtime and
  # height averaged over the new time resolution.
  #
  # Note: drop concentrations will be averaged for each height, without
  # checking whether the diameter classes are the same for each record.
  # But, for the MRR the diameter classes are defined by height so this
  # should not be a problem.
  
  resSec = convertTimeStringsToUnit(res, "sec")
  newTimes = seq(start-resSec, d[, max(POSIXtime)]+(2*resSec), by=res)
  d[, ts := cut(POSIXtime+resSec, newTimes, right=TRUE)]
  stopifnot(!any(d[, is.na(ts)]))

  cols = names(d)
  cols = cols[which(cols != "POSIXtime" & cols != "height" & cols != "ts")]
  
  # Convert columns in log scale to linear scale.
  n = names(d)
  for(c in n[which(n %in% logVariables)]) {
    d[[c]] = 10^(d[[c]]/10)
  }

  resampled = d[, lapply(.SD, mean, na.rm=TRUE), .SDcols=cols, by="ts,height"]
  setnames(resampled, "ts", "POSIXtime")
  
  # Convert back to log scale.
  n = names(resampled)
  for(c in n[which(n %in% logVariables)]) {
    resampled[[c]] = 10*log10(resampled[[c]])
  }
  
  return(resampled)
}

readMRRDataAtTimeRes = function(dir, start, end, variables, res, resample=TRUE) {
  # Read MRR data at a certain time resolution.
  #
  # Args:
  #   dir: Input directory.
  #   start, end: POSIXct UTC times for start and end of time period.
  #   variables: List of variables to read.
  #   res: New time resolution, eg "5 min".
  #   resample: Perform resampling? (Default: TRUE).
  #
s  # Returns: MRR data as a data.table containing POSIXtime, height, and
  # selected variables.
  
  # Read data and resample on the fly.
  d = readMRRVariables(dir, start, end, variables, resample=resample, 
                       res=res, resStart=start)
  if(is.null(d)) {return(NULL)}
  d[, POSIXtime := as.POSIXct(POSIXtime, tz="UTC")]
  return(d)
}

MRRDiameterBins = function(altitude, 
                           heights=seq(100, 3100, by=100),
                           v=seq(0, by=0.18873, length.out=64)) {
  # Calculate the diameter bins for each height above the MRR. The
  # bins depend on the altitude of the instrument.
  #
  # Args:
  #   altitude: The altitude of the MRR instrument [m].
  #   heights: Measurement heights above MRR [m].
  #   v: The MRR velocity bins starting values [m/s].
  # 
  # Returns: a list containing min/max diameters for each height.
  
  # The altitude for each level is the instrument altitude plus the height
  # of each requested height above the instrument.
  altitudes = altitude + heights

  # Calculate diameter bins.
  diameterBins = MRRDiameterByVelocity(altitudes, v)
  
  # Sort into a list with an entry for each height level.
  binsByHeight = list()
  for(h in seq(1, length(heights))) {
    min = as.numeric(diameterBins[h,])
    n = length(min)
    max = c(min[2:n], NA)
    
    bins = cbind(min, max)
    nas = which(is.na(rowMeans(bins)))
    bins[nas,] = c(NA, NA)
    binsByHeight[[h]] = bins
  }
  
  return(binsByHeight)
}
  
MRRDiameterByVelocity = function(z, v) {
  # Return the diameter of a drop falling at a velocity at an altitude.
  # 
  # Args:
  #   z: Vector of m altitudes to find diameters at [m].
  #   v: Vector of n velocities for which to find diameters [m/s].
  #
  # Returns: an m by n data.frame containing a diameter for each altitude/
  # velocity combination.
  
  # Find dv, the height dependent correction for fall velocity, for each 
  # requested altitude.
  dv = 1 + ((3.68 * 1e-5)*z) + ((1.71 * 1e-9)*z^2)
  
  # Find the diameters for each altitude and velocity combination.
  res = NULL
  for(i in seq(1, length(z))) {
    y = 10.3/(9.65 - v/dv[i])
    y = replace(y, y<0, NA)
    D = (1/0.6) * log(y)
    res = rbind(res, data.frame(t(D)))
  }
  
  return(res)
}

MRRXbandReflectivityFromDSD = function(mrr, altitude, 
                                       minDiam=0.246, maxDiam=5.03,
                                       radarFreq=9.4,
                                       temperature=10, radarIncidence=0) {
  # Calculate the horizontal and vertical reflectivities at X-band
  # using the MRR DSD data. (Rayleigh/Tmatrix reflectivity).
  #
  # Args:
  #   mrr: A data.table (or frame) of MRR data, containing columns named 
  #        POSIXtime, height, and Nnn_{bin}.
  #   altitude: The altitude of the MRR instrument.
  #   minDiam: Minimum diameter considered by the MRR [mm] (default: 0.246 mm).
  #   maxDiam: Maximum diameter considered by the MRR [mm] (default: 5.03 mm).
  
  # Make sure mrr is a data.table.
  mrr = data.table(mrr)
  heights = mrr[, unique(height)]
  
  # Get the diameter bin sizes.
  binsByHeight = MRRDiameterBins(altitude=altitude, heights=heights)
  
  # Each height has different bin sizes. Loop through heights.
  allStats = data.table()
  for(i in seq(1, length(heights))) {
    h = heights[i]
    diamClasses = binsByHeight[[i]]
    
    # The MRR only uses classes between diameters of 0.246 mm to 5.03 mm. So
    # we select only these columns.
    startCol = which(diamClasses[,2] >= minDiam)[1]
    endCol = which(diamClasses[,2] >= maxDiam)[1]
    diamClasses = diamClasses[startCol:endCol,]
    stopifnot(!any(is.na(diamClasses)))
    
    # Get DSD classes from the MRR data.
    cols = paste("Nnn_", seq(startCol, endCol), sep="")
    dsds = data.frame(mrr[height == h, c("POSIXtime", cols), with=FALSE])
    times = dsds$POSIXtime
    spectra = as.matrix(dsds[, 2:dim(dsds)[2]])
    
    # Set NaNs and negative numbers in the DSD to zero.
    spectra[which(is.na(spectra))] = 0
    spectra[which(spectra < 0)] = 0
    
    # Find the Rayleigh reflectivity at X-Band for each DSD.
    reflectivities = radarReflectivityFromDSD(DSD=spectra, 
                                              classes=diamClasses,
                                              freq=radarFreq, 
                                              temp=temperature, 
                                              incidence=radarIncidence)
    reflectivities = 10*log10(reflectivities)
    names(reflectivities)
    
    stats = data.table(POSIXtime=times, height=h, reflectivities)
    allStats = rbind(allStats, stats)
  }
  
  return(allStats)
}

MRRReflectivityFromDSD = function(mrr, altitude, minDiam=0.246, maxDiam=5.03) {
  # Calculate the radar reflectivity using the 6th moment of the DSD, as
  # is done by the MRR instrument.
  # 
  # Args:
  #   mrr: A data.table (or frame) of MRR data, containing columns named 
  #        POSIXtime, height, and Nnn_{bin}.
  #   altitude: The altitude of the MRR instrument.
  #   minDiam: Minimum diameter considered by the MRR [mm] (default: 0.246 mm).
  #   maxDiam: Maximum diameter considered by the MRR [mm] (default: 5.03 mm).
  #
  # Returns: A data.table containing Z per POSIXtime and height.
  
  # Make sure mrr is a data.table.
  mrr = data.table(mrr)
  heights = mrr[, unique(height)]
  
  # Get the diameter bin sizes.
  binsByHeight = MRRDiameterBins(altitude=altitude, heights=heights)
  
  # Each height has different bin sizes. Loop through heights.
  allStats = data.table()
  for(i in seq(1, length(heights))) {
    h = heights[i]
    diamClasses = binsByHeight[[i]]
    
    # The MRR only uses classes between diameters of 0.246 mm to 5.03 mm. So
    # we select only these columns.
    startCol = which(diamClasses[,2] >= minDiam)[1]
    endCol = which(diamClasses[,2] >= maxDiam)[1]
    diamClasses = diamClasses[startCol:endCol,]
    stopifnot(!any(is.na(diamClasses)))
    
    # Get DSD classes from the MRR data.
    cols = paste("Nnn_", seq(startCol, endCol), sep="")
    dsds = data.frame(mrr[height == h, c("POSIXtime", cols), with=FALSE])
    times = dsds$POSIXtime
    spectra = as.matrix(dsds[, 2:dim(dsds)[2]])
    
    # Set NaNs and negative numbers in the DSD to zero.
    spectra[which(is.na(spectra))] = 0
    spectra[which(spectra < 0)] = 0
    
    # Find the 6th moment radar reflectivity for each row in the DSD spectra.
    centreDiams = rowMeans(diamClasses)       # Class centre diameters [mm].
    classWidths = apply(diamClasses, 1, diff) # Class widths [mm].
    
    reflectivityForRow = function(x) {
      return(sum(x * centreDiams^6 * classWidths))
    }
    
    # Get reflecitivities in linear units. Unit will be:
    # (mm-1 m-3).mm6.mm = mm6 m-3.
    reflectivities = apply(spectra, 1, reflectivityForRow)
    
    # Convert to log.
    reflectivities = 10*log10(reflectivities)
    
    stats = data.table(POSIXtime=times, height=h, Z=reflectivities)
    allStats = rbind(allStats, stats)
  }
    
  return(allStats)
}

MRRDSDStats = function(mrr, altitude, latitude, minDiam=0.246, maxDiam=5.03, 
                       timeRes=10) {
  # Calculate rain statistics for a set of MRR DSDs.
  # 
  # Args:
  #   mrr: A data.table (or frame) of MRR data, containing columns named 
  #        POSIXtime, height, and Nnn_{bin}.
  #   altitude: The altitude of the MRR instrument.
  #   minDiam: Minimum diameter considered by the MRR [mm] (default: 0.246 mm).
  #   maxDiam: Maximum diameter considered by the MRR [mm] (default: 5.03 mm).
  #   timeRes: Integration time [s] (default: 10 s).
  #
  # Returns: A data.table containing derived rain statistics per 
  # POSIXtime and height.
  
  # Make sure mrr is a data.table.
  mrr = data.table(mrr)
  heights = mrr[, unique(height)]
  
  # Get the diameter bin sizes.
  binsByHeight = MRRDiameterBins(altitude=altitude, heights=heights)
  
  # Each height has different bin sizes. Loop through heights.
  allStats = data.table()
  for(i in seq(1, length(heights))) {
    h = heights[i]
    diamClasses = binsByHeight[[i]]
      
    # The MRR only uses classes between diameters of 0.246 mm to 5.03 mm. So
    # we select only these columns.
    startCol = which(diamClasses[,2] >= minDiam)[1]
    endCol = which(diamClasses[,2] >= maxDiam)[1]
    diamClasses = diamClasses[startCol:endCol,]
    stopifnot(!any(is.na(diamClasses)))
    
    # Get DSD classes from the MRR data.
    cols = paste("Nnn_", seq(startCol, endCol), sep="")
    dsds = data.frame(mrr[height == h, c("POSIXtime", cols), with=FALSE])
    times = dsds$POSIXtime
    spectra = as.matrix(dsds[, 2:dim(dsds)[2]])
    
    # Set NaNs and negative numbers in the DSD to zero.
    spectra[which(is.na(spectra))] = 0
    spectra[which(spectra < 0)] = 0

    spectra = data.frame(spectra)
    names(spectra) = paste("class", seq(1, dim(spectra)[2]), sep="")
    spectra$station = "MRR"
    MRRstation = data.frame(name="MRR", altitude=altitude, lat=latitude)
    stats = DSDRainStats(spectra=spectra, timestepSeconds=timeRes,
                         classes=diamClasses, radarIncidence=90, radar=TRUE,
                         stations=MRRstation)
    
    stats = data.table(POSIXtime=times, height=h, stats)
    allStats = rbind(allStats, stats)
  }
  
  return(allStats)
}

averageMRRDSDByHeight = function(mrr, altitude) {
  # Find the average DSD by height given by MRR data.
  #
  # Args:
  #   mrr: MRR data, containing POSIXtime, height, and Nnn_{bin}.
  #   altitude: Altitude of the instrument.
  #
  # Returns: A data.table with average values for each Nnn column, 
  #          per height.
  
  mrr = data.table(mrr)
  
  # Function to take mean of positive values and to set NaN values to zero.
  meanFunc = function(x) {
    x[which(x < 0)] = 0
    x[which(is.na(x))] = 0
    return(mean(x))
  }
  
  numClasses = length(grep("Nnn", names(mrr)))
  classCols = paste("Nnn_", seq(1, numClasses), sep="")
  
  aveDSDs = mrr[, lapply(.SD, meanFunc), by=height, .SDcols=classCols]
  return(aveDSDs)
}

findAverageMRRDSDAtHeight = function(mrrDir="/ltedata/HYMEX/SOP_2013/MRR_LTHE_Montbrun/ProcessedData/", 
                                     height=500) {
  
  files = list.files(mrrDir, recursive=TRUE, full.name=TRUE)
  allCounts = NULL
  for(file in files) {
    print(file)
    nc = nc_open(file)
    
    heights = ncvar_get(nc, "H")
    height_idx = which.min(abs(heights - height))
    
    Z = ncvar_get(nc, "Z")[height_idx,]
    record_idx = which(Z >= 10)
    if(length(record_idx) == 0) {
      nc_close(nc)
      next
    } 
    
    diams = ncvar_get(nc, "Dnn")[,height_idx,record_idx]
    counts = ncvar_get(nc, "Nnn")[,height_idx,record_idx]
    
    stopifnot(identical(dim(diams), dim(counts)))
    
    diams[which(diams == -99900)] = NA
    counts[which(counts == -99900)] = NA
    
    idx = which(!is.na(diams) & !is.na(counts))
    if(length(idx) > 0) {
      d = data.table(diam=diams[idx], count=counts[idx])
      allCounts = rbind(allCounts, d)
    }
    
    nc_close(nc)
  }
  
  averageDSD = allCounts[, mean(count, na.rm=TRUE), by=diam]
  save(averageDSD, file="~/Desktop/MRR-dsd-avg.Rdata")
  return(averageDSD)
}

readMRR_LAMP = function(dir, start, end) {
    ## Read LAMP-processed MRR files.
    stop("Function untested!")
    
    ## Get one file per required day.
    requiredDates = strftime(trunc(seq(start, end, by="1 day"), "day"), "%Y%m%d")
    fileList = NULL
    for(date in requiredDates) {
        dayFile = list.files(dir, recursive=T, pattern=date, full.names=T)
        if(length(dayFile) > 0)
            fileList = rbind(fileList, dayFile)
    }
    
    ## Read each file in turn.
    res = NULL
    for(file in fileList) {
        print(file)
        nc = nc_open(file)
    
        ## Get the measurement times [UTC].
        year = ncvar_get(nc, "start_year")
        month = ncvar_get(nc, "start_month")
        day = ncvar_get(nc, "start_day")
        hour = ncvar_get(nc, "start_hour")
        mins = ncvar_get(nc, "time_offset")
        
        times = as.POSIXct(paste(year, "-", month, "-", day, " ",
            hour, ":", mins, ":00", sep=""), tz="UTC")
        
        ## Which records correspond to the times we want?
        recordIdx = which(times >= start & times < end & !duplicated(times))
        if(length(recordIdx) == 0) next
        
        ## Altitudes for these times (height above instrument).
        heights = ncvar_get(nc, "data_altitude") - ncvar_get(nc, "altitude")
        
        ## Get the DSD for these times. Order is altitude, size class, time.
        dsds = ncvar_get(nc, "drop_spectra")
        
        dsdNames = paste("Nnn_", seq(1, dim(dsds)[2]), sep="")
        for(altitude in seq(1, length(heights))) {
            dsdsForAltitude = data.frame(t(dsds[altitude, ,]))
            names(dsdsForAltitude) = dsdNames
            res = rbind(res, data.table(POSIXtime=times, dsdsForAltitude))
        }
        
        nc_close(nc)
    }
    
    return(res)
}

readMRREventsAtTimeRes = function(dir, events, timeRes,
    variables, stationName, stations) {

    timeResSeconds = convertTimeStringsToUnit(timeRes)
    
    mrr = NULL
    for(e in seq(1, length(events$start))) {
        start = as.numeric(events$start[e])
        start = as.POSIXct(ceil(start / timeResSeconds) * timeResSeconds,
            tz="UTC", origin="1970-1-1")
        end = events$end[e]
        
        mrrData = readMRRDataAtTimeRes(dir=dir,
            start=start, end=end, variables=MRRVariables, res=timeRes)
        
        if(!is.null(mrrData))
            mrr = rbind(mrr, data.table(station=stationName, mrrData))
        rm(list=c("mrrData"))
    }

    ## Get real diameter bins per station/height.
    MRRDiamBins = getMRRDiamBins(mrr, stations)

    ## Define MRR data columns.
    MRRDiamCols = paste("Dnn_", seq(1,64), sep="")
    MRRDSDCols = paste("Nnn_", seq(1,64), sep="")
    MRRWidthCols = paste("DeltaD_", seq(1,64), sep="")

    ## Replace the diameter classes in mrr with those calculated for
    ## the correct station and height. Also, truncate DSD so no
    ## drops above 7 mm are counted.
    mrr[, heightBin := which(mrr[, unique(height)] == height), by=height]
    for(s in names(MRRDiamBins)) {
        for(h in seq(1, length(MRRDiamBins[[s]]))) {
            n = mrr[station == s & heightBin == h, length(station)]
            d = as.numeric(rowMeans(MRRDiamBins[[s]][[h]]))
            w = as.numeric(apply(MRRDiamBins[[s]][[h]], 1, diff))

            ## Assign diameters.
            m = data.table(matrix(rep(d, n), nrow=n, byrow=TRUE))
            mrr[station == s & heightBin == h, (MRRDiamCols) := m]

            ## Assign widths.
            wm = data.table(matrix(rep(w, n), nrow=n, byrow=TRUE))
            mrr[station == s & heightBin == h, (MRRWidthCols) := wm]
            
            l = as.numeric(apply(MRRDiamBins[[s]][[h]], 1, max))
            idx = which(l > 7)
            if(length(idx) > 0)
                mrr[station == s & heightBin == h, (MRRDSDCols[idx]) := 0]
        }
    }

    ## Replace negative and NA counts with zero.
    for(col in MRRDSDCols) {
        mrr[which(mrr[[col]] < 0), col := 0, with=FALSE]
        mrr[which(is.na(mrr[[col]])), col := 0, with=FALSE]
    }
    setkey(mrr, POSIXtime, station, heightBin)
    mrr$station = as.character(mrr$station)
    
    return(mrr)
}

getMRRDiamBins = function(mrr, stations) {
    ## Calculate the diameter bins for all measured heights; the diameter bins for the
    ## MRR data depend on instrument altitude.
    stations = data.table(stations)
    MRRDiamBins = list()
    for(station in mrr[, unique(station)]) {
        MRRDiamBins[[station]] =
            MRRDiameterBins(altitude=stations[name==station, altitude],
                            heights=mrr[, unique(height)])
    }

    return(MRRDiamBins)
}
