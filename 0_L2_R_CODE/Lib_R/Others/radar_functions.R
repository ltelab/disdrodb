# radar_functions.R
#
# Functions for reading and plotting radar data from netCDF files.
#
# Author: Tim Raupach <tim.raupach@epfl.ch>

require(ncdf4)
require(reshape)
require(parallel)
require(plyr)
require(gstat)
require(ggplot2)
require(data.table)
require(gridExtra)
require(geosphere)
require(raster)
require(sp)

#############################################################################

# Proj4 string for metres.
# UTM Zone 32. France is across zones 31, 32, 33.
MetresProjString = paste("+proj=utm +zone=31 +ellps=WGS84",
    "+datum=WGS84 +units=m +no_defs")

# Variables that are assumed to be in log scale.
LogScaleVars = c("ZhCorr", "Zh", "Zv", "ZvCorr", "Zdr", "ZdrCorr")

getRadarDimensionsTable = function(file) {
  # Return a table of dimensions and their sizes from a filename.
  # 
  # Args:
  #   file: The filename to open and get information about.
  #
  # Returns: table ready for display.
  
  nc = readRadarFile(file)  
  res = NULL 

  for(dim in names(nc$dim)) {
    d = nc$dim[[dim]]
    res = rbind(res, data.frame(d$name, d$len))
  }
  
  names(res) = c("Dimension", "Size")
  return(res)
}

getRadarVariablesTable = function(file) {
  # Produce a table of variables and the dimensions they are defined on.
  #
  # Args:
  #  file: An NC filename.
  #
  # Returns: data.frame ready for display.
  
  nc = readRadarFile(file)  
  res = NULL
  
  for(var in names(nc$var)) {
    v = nc$var[[var]]
    # Warning: this assumes the dimensions are in ID order. 
    dims = names(nc$dim[(v$dimids + 1)])
    dimString = paste(dims, collapse=", ")
    res = rbind(res, data.frame(v$name, dimString))
  }
  
  names(res) = c("Variable", "Dimensions")
  return(res)
}

getRadarTimeResolution = function(radarDir, start, end, elevation, 
                                  radarType="PPI",
                                  getFileTimeFunc=getTimeFromNamePPI,  
                                  roundToNearest=30) {
  # Find the average time gap between radar scans, in seconds, 
  # for a certain time period and elevation.
  #
  # Args:
  #   radarDir: The directory in which the radar files are stored in 
  #             directories by year, month, then day.
  #   start, end: Start and end times of the selected period (POSIXct, UTC).
  #   elevation: Elevation to select [degrees].
  #   radarType: The radar scan type to read (default: PPI).
  #   getFileTimeFunc: Function that will take the year, month, day, and a 
  #                    filename and return the scan time.
  #   roundToNearest: Round to nearest x seconds (default: 30).
  #
  # Returns: The mean number of seconds between scans.
  
  dates = seq(as.Date(start), as.Date(end), by="1 day")
  
  years  = unique(sprintf("%02d", as.numeric(strftime(dates, "%Y"))))
  months = unique(sprintf("%02d", as.numeric(strftime(dates, "%m"))))
  days   = unique(sprintf("%02d", as.numeric(strftime(dates, "%d"))))
  
  times = NULL
  for(year in years) {
    for(month in months) {
      for(day in days) {
        dir = paste(radarDir, year, month, day, sep="/")      
        pattern = paste(".*", radarType, ".*", elevation, ".*", sep="")
        fileList = list.files(dir, pattern=pattern)
        times = c(times, strftime(getFileTimeFunc(year, month, day, fileList), 
                                  "%Y-%m-%d %H:%M:%S", tz="UTC"))
      }
    }
  }

  times = as.POSIXct(times, tz="UTC")
  times = as.numeric(times, tz="UTC")
  times = times[order(times)]
  
  sdTimeDiffs = sd(diff(times))
  stopifnot(diff(range(diff(times))) < 5)
  meanSeconds = round(mean(diff(times)))
  return(roundToNearest * (meanSeconds %/% roundToNearest))
}

getRadarValuesTimeseries = function(radarDir, times, variable, elevation, 
                                    maxAllowedTimeDiff=NULL,
                                    radarType="PPI",
                                    pattern=sprintf(".*-%s-%03d.*.nc", 
                                                    radarType, elevation),
                                    getFileTimeFunc=getTimeFromNamePPI) {
  # Get a timeseries of all radar values and their ground coordinates in 
  # lat/long.
  #
  # Args:
  #  radarDir: The directory in which the radar files are stored in directories
  #            by year, month, then day.
  #  times: Series of times to find values for (POSIXct, UTC).
  #  variable: The radar variable to read.
  #  elevation: Radar elevation to consider.
  #  maxAllowedTimeDiff: The maximum allowed time difference [s]. If NULL
  #                      then no maximum is set and the closest possible scan
  #                      is returned. (Default: NULL).
  #  radarType: Radar type ("PPI" or "RHI") (default: "PPI").
  #  pattern: File-matching pattern to use.
  #  getFileTimeFunc: Function to get time from a filename.
  #
  # Returns: a spatial data.frame containing all the radar values across the
  #          requested times, 
  
  allValues = NULL
  
  # Cache the list of files to search through.
  files=list.files(radarDir, pattern=pattern, 
                   full.names=TRUE, 
                   recursive=FALSE)
  fileTimes = getFileTimeFunc(files)
  listVals = list()
  i = 1
  
  # Loop through times.
  for(time in times) {
    time = as.POSIXct(time, tz="UTC", origin="1970-1-1")
   
    closest = readClosestRadarTimestep(radarDir=radarDir, 
                                    time=time, 
                                    radarType=radarType,
                                    elevation=elevation, 
                                    pattern=pattern,
                                    maxAllowedTimeDiff=maxAllowedTimeDiff,
                                    files=files, times=fileTimes)
    
    # No file? Try the next time.
    if(is.null(closest)) next
 
    # If there is more than one matching file, we have to average values,
    # which is not yet implemented. So stop if > 1 file.
    stopifnot(length(closest$nc) == 1)
    
    print(paste("File for", time, "with scan time", closest$scanTime))
    
    ## Get the values and ground coordinates from the file.
    nc = closest$nc[[1]]
    values = radarValuesWithGroundCoords(nc, variable)
    closeRadarFile(nc)
    
    listVals[[i]] = data.table(requestedTime=time, 
                               scanTime=closest$scanTime, 
                               data.frame(values))
    i = i + 1
  }
  
  allValues = rbindlist(listVals, use.names=TRUE)
  return(allValues)
}

radarValuesWithGroundCoords = function(nc, variable) {
  # For a PPI scan, return each radar point with its ground coordinate 
  # in latitude/longitude.
  #
  # Args:
  #   nc: The radar NC file. 
  #   variable: The radar variable(s) to get.
  # 
  # Returns: data.frame with radar data and each point's range (straight line 
  #          from radar, not ground range), azimuth, and ground-level lat/long 
  #          at the centre of the corresponding volume (warning: volumes get 
  #          bigger as the distance from the radar increases)
  
  # First get the points as functions of range and azimuth.
  radarData = radarValuesAsPoints(nc, variable[1])
  if(length(variable) > 1) {
    for(var in variable[2:length(variable)]) {
      radarData = cbind(radarData, radarValuesAsPoints(nc, var)[[var]])
    }
  }
  if(!all(names(radarData)[1:2] == c("azimuth", "range"))) {
    print("To get lat/long values, scan must be PPI.")
    stop()
  }
  names(radarData) = c("azimuth", "range", variable)
  
  # Get centre point in lat/long.
  centreLat = ncatt_get(nc, 0, "Latitude-value")$value  # Radar lat [degN].
  centreLon = ncatt_get(nc, 0, "Longitude-value")$value # Radar lon [degE].
  elevation = ncatt_get(nc, 0, "Elevation-value")$value # Beam elevation [deg].
  
  # Convert centre point into spatial object.
  projCRS = CRS(MetresProjString)
  centrePoint = data.frame(lat=centreLat, lon=centreLon)
  coordinates(centrePoint) = ~lon+lat
  proj4string(centrePoint) = CRS("+proj=longlat +datum=WGS84")
  
  # Project the ranges to horizontal range from the radar [m].
  radarData$groundRange = volumeGroundRange(radarData$range, elevation)
  
  # Find the great-circle destination points from the radar, following
  # bearings and going in a straight line over the earth's surface
  # by the ground range distance.
  coords = destPointEllipsoid(startLat=centrePoint$lat, 
                              startLon=centrePoint$lon,
                              bearing=radarData$azimuth, 
                              dist=radarData$groundRange)
  radarData$lon = coords$lon
  radarData$lat = coords$lat
  
  # Get the height above ground of the volume centres in m.
  radarData$height = volumeHeight(radarData$range, elevation)
  
  # Convert to a spatial object.
  coordinates(radarData) = ~lon+lat
  proj4string(radarData) = CRS("+proj=longlat +datum=WGS84")
  
  return(radarData)
}

radarValuesAsPoints = function(nc, variable) {
  # Extract all unique radar measurements from a given NC file, return as 
  # points indexed by range and angle (either azimuth, for PPI, or elevation,
  # for RHI).
  #
  # Args: 
  #   nc: The netCDF object to extract from.
  #   variable: The variable to extract.
  # 
  # Returns: Each unique value 
  
  # Determine whether this is a PPI or RHI file.
  radarType = "PPI"
  if("Elevation" %in% names(nc$dim)) {
    radarType = "RHI"
  }
  
  # Get variables from NC file.
  azimuths = ncvar_get(nc, "Azimuth")     # Angle from N [degrees].
  elevations = ncvar_get(nc, "Elevation") # Angle from horizontal [degrees].
  ranges   = ncvar_get(nc, "Range")       # Range from radar [m].
  varData  = ncvar_get(nc, variable)      # The data to plot.
  
  # Get attributes.
  NAVal = ncatt_get(nc, 0, "MissingData")$value              # NA value [-].
  rangeRes = ncatt_get(nc, 0, "RangeResolution-value")$value # Range res [m].
  
  # Melt the data so each point has its two dimensional coordinate.
  radarData = melt(varData)
  names(radarData) = c("angle", "range", variable)
  
  # Reset NA values -- note floating point check set to 1e-6.
  radarData[[variable]][which(abs(radarData[[variable]] - NAVal) < 1e-6)] = NA
  
  # Get the real ranges.
  radarData$range = ranges[radarData$range]
  
  # The real angle to use depends on the scan type.
  if(radarType == "PPI") {
    radarData$angle = azimuths[radarData$angle]
    names(radarData) = c("azimuth", "range", variable)
  } else { # RHI
    radarData$angle = elevations[radarData$angle]
    names(radarData) = c("elevation", "range", variable)
  }
  
  return(radarData)
}

closestRadarTimestepFile = function(radarDir, time, elevation, radarType="PPI",
    getFileTimeFunc=getTimeFromNamePPI,
    pattern=sprintf(".*-%s-(%s).*.nc",
        radarType, paste(sprintf("%03d", elevation), collapse="|")),
    timeRes=NULL, scanLength=20,
    files=list.files(radarDir, pattern=pattern, 
        full.names=TRUE, 
        recursive=FALSE),
    times = getFileTimeFunc(files)) {
    ## Find the radar files that fall in a given timestep. 
    ##
    ## Args:
    ##   radarDir: The directory in which all radar files are stored.
    ##   time: The time to look for (end of the time step)
    ##   elevation: Which elevation (or azimuth for RHI) to read? [Degrees].
    ##   radarType: PPI (horizontal) or RHI (vertical).  
    ##   getFileTimeFunc: Function that will take the year, month, day, and a 
    ##                    filename and return the scan time.
    ##   pattern: The filename pattern to match.
    ##   timeRes: Allowed time difference from requested time [s]. (Default NULL, get closest time).
    ##   scanLength: The assumed radar scan length [s] (default: 20 s).
    ##   files: List of files to search through; can be specified to 'cache'
    ##          the process of reading the files from disk.
    ##   times: Time differences from files, can be cached also.
    ##
    ## Returns: A list containing the filename of the closest radar timestep 
    ## to the specified time (file) and the radar timestep for that time (time).
    ## NULL if no possible file is found.
  
  if(length(files) == 0)
    return(NULL)

  ## "times" contains the scan times, which are the END of the radar scan.
  ## "time" is the end of the time period we want to match.
  if(is.null(timeRes)) {
      ## Return the closest time.
      timeDiffs = as.numeric(abs(difftime(time, times, tz="UTC", unit="sec")))
      idx = which.min(timeDiffs)
  } else {
      ## Return all scans for which at least half the scan overlapped the
      ## requested time step.
      stopifnot(scanLength / 2 <= timeRes)
      minTime = time - timeRes + (scanLength/2)
      maxTime = time + (scanLength/2)      
      idx = which(times > minTime & times <= maxTime)
  }
  
  if(length(idx) == 0)
    return(NULL)
  
  filesToOpen = files[idx]  
  return(list(file=filesToOpen, time=times[idx]))
}

readClosestRadarTimestep = function(radarDir, time, elevation, radarType="PPI",
    getFileTimeFunc=getTimeFromNamePPI,
    maxAllowedTimeDiff=NULL,
    pattern=sprintf(".*-%s-(%s).*.nc",
        radarType, paste(sprintf("%03d", elevation), collapse="|")),
    ...) {
  # Read radar data from file, automatically choosing the closest file
  # to a particular timestep. 
  # 
  # Args:
  #   radarDir: The directory in which the radar files are stored in directories
  #             by year, month, then day.
  #   time: The time to look for.
  #   elevation: Which elevation to read? [Degrees].
  #   radarType: PPI (horizontal) or RHI (vertical).
  #   maxAllowedTimeDiff: The maximum allowed time difference [s]. If NULL
  #                       then no maximum is set (default: NULL).
  #   getFileTimeFunc: Function that will take the year, month, day, and a 
  #                    filename and return the scan time.
  #   pattern: The filename pattern to match.
  #   ...: Extra arguments to closestRadarTimestepFile().
  #
  # Returns: A list containing nc (a list of NC objects for the files closest to the 
  #          specified time), scanTime (the time the scan was made), or NULL 
  #          if no suitable file is found.
  
  filesToOpen = closestRadarTimestepFile(radarDir=radarDir, 
      time=time, elevation=elevation,
      radarType=radarType,
      getFileTimeFunc=getFileTimeFunc,
      pattern=pattern,
      timeRes=maxAllowedTimeDiff,
      ...)
  
  if(is.null(filesToOpen))
    return(NULL)
  
  ncs = list()
  for(file in filesToOpen$file) {
    ncs = c(ncs, list(readRadarFile(file)))
  }
  
  return(list(nc=ncs, scanTime=filesToOpen$time))
}

getTimeFromNamePPI = function(file, offset=0) {
  # From a radar file name, return the time of the scan.
  # 
  # Args:
  #   file: The filename.
  # 
  # Returns: UTC POSIXct time of the scan, from the filename.
  
  file = basename(file)
  time = substr(file, 22+offset, 27+offset)  
  hour = substr(time, 1, 2)
  min = substr(time, 3, 4)
  sec = substr(time, 5, 6)
  
  date = substr(file, 13+offset, 20+offset)
  year = substr(date, 1, 4)
  month = substr(date, 5, 6)
  day = substr(date, 7, 8)
  
  daystring = paste(year, month, day, sep="-")
  timestring = paste(hour, min, sec, sep=":")
  ptime = as.POSIXct(paste(daystring, timestring), tz="UTC")
  return(ptime)
}

getTimeFromNameVertical = function(file) {
  # From a radar file name, return the time of the scan.
  # 
  # Args:
  #   year: The year (already known).
  #   month: The month (already known).
  #   day: The day (already known).
  #   file: The filename.
  # 
  # Returns: UTC POSIXct time of the scan, from the filename.
  
  file = basename(file)
  time = substr(file, 24, 29)  
  hour = substr(time, 1, 2)
  min = substr(time, 3, 4)
  sec = substr(time, 5, 6)
  
  date = substr(file, 15, 22)
  year = substr(date, 1, 4)
  month = substr(date, 5, 6)
  day = substr(date, 7, 8)
  
  daystring = paste(year, month, day, sep="-")
  timestring = paste(hour, min, sec, sep=":")
  ptime = as.POSIXct(paste(daystring, timestring), tz="UTC")
  return(ptime)
}

readRadarFile = function(file) {
  # Read a radar observation file from disk.
  # 
  # Args:
  #   file: The file name to read.
  #
  # Returns: A ncdf object read from the file.
  
  ncObj = nc_open(file)
  return(ncObj)
}

closeRadarFile = function(nc) {
  # Close a radar object.
  #
  # Args:
  #   nc: The NC object to close.
  #
  # Returns: void.
  
  nc_close(nc)
}

indicatorVariogramFromRadar = function(radarDir, times, elevation,
                                       radarVar="ZhCorr", 
                                       rainyThreshold=20,
                                       useCressie=FALSE,
                                       model="Sph",
                                       angle=0, tol.hor=90,
                                       aroundPoint=numeric(0),
                                       maxDist=10) {
  # Produce an indicator variogram from a radar record.
  # 
  # Args:
  #  radarDir: The directory in which the radar files are stored in directories
  #            by year, month, then day.
  #  time: The time to match. The closest radar image to that time will be used.
  #  radarVar: Which radar variable to use?
  #  rainyThreshold: What is the lowest value that corresponds to rain?
  #  useCressie: Use Cressie's robust variogram estimator (default: TRUE).
  #  model: The model to fit to the variogram? (Default: spherical).
  #  angle, tol.hor: Angle and horizontal tolerance for variogram fit 
  #                  (default: 0 and 90, ie all points).
  #  aroundPoint: If defined as an SP object, only use radar points around 
  #               this point (default: use all points).
  #  maxDist: Only use radar points within this distance [km] from 
  #           aroundPoint. (default: 30 km).
  #
  # Returns: A indicator variogram and model.
  
  index = 1
  allPoints = NULL
  prevNCFilename = NULL
  
  # Loop through times and read in data.
  for(t in times) {
    time = as.POSIXct(t, origin="1970-1-1", tz="UTC")
    
    closestFile = readClosestRadarTimestep(radarDir, time, elevation)
    nc = closestFile$nc
    if(identical(nc$filename, prevNCFilename)) { 
      closeRadarFile(nc)
      next 
    }
    
    pointVals = radarValuesAsPoints(nc, radarVar)
    pointVals$rainy = rep(0, length(pointVals[[radarVar]]))
    pointVals$rainy[which(pointVals[[radarVar]] > rainyThreshold)] = 1
    
    if(length(aroundPoint) > 0) {
      
      if(!identical(proj4string(pointVals), proj4string(aroundPoint))) {
        aroundPoint = spTransform(aroundPoint, CRS(proj4string(pointVals)))
      }
      
      # Get distances from aroundPoint in km, only keep those points within 
      # range. Note that this is Euclidean distance.
      idx = which(spDists(pointVals, aroundPoint) < maxDist)
      stopifnot(length(idx) > 0)
      pointVals = pointVals[idx,]
    }  
    
    pointVals$index = index
    allPoints = rbind(allPoints, as.data.frame(pointVals))
    index = index + 1
    prevNCFilename = nc$filename
    closeRadarFile(nc)
  }
  
  # Re-spatialise the collected points.
  coordinates(allPoints) = ~x+y
  proj4string(allPoints) = proj4string(pointVals)
  
  # Calculate the variogram.
  vario = variogram(rainy~index, data=allPoints, 
                    dX=0.5, alpha=angle, 
                    tol.hor=tol.hor)
  
  # Fit a model to the variogram.
  fittedModel = fit.variogram(vario, model=vgm(model=model, nugget=0, 
                                               psill=0.05, range=10))
  
  plot(vario, model=fittedModel)
  
  # Collect both together in a list and return.
  res = list(vario=vario, model=fittedModel)
  return(res)
}

RHIGrid = function(res=75, maxRange=40000, maxHeight=10000) {
  # Create a rectangular RHI grid around the radar centre point. Return
  # the range and elevation for each point compared to the radar centre.
  #
  # Args:
  #  res: The grid resolution [m] (default: 75 m).
  #  maxRange: The maximum distance from the radar to find [m] 
  #            (default: 40000 m). 
  #  maxHeight: The maximum height above the radar to find [m]
  #             (default: 10000 m).  
  #
  # Returns: The grid as a data.frame containing x, y, elevation and range.
  #          x is the horizontal range from the radar, y is the vertical height.
  
  n = maxRange %/% res
  x = seq(-n, n) * res
  y = seq(0, maxHeight %/% res) * res
  
  grid = as.data.frame(expand.grid(x, y))
  names(grid) = c("x", "y")
  grid$range = sqrt(grid$x^2 + grid$y^2)
  
  # Get elevation angle from centre, for each point.
  grid$elevation = atan2(grid$y, grid$x) * 180/pi 
  
  return(grid)
}

radarPlaneGrid = function(res=75, maxDist=40000) {
  # Create a square grid in the 2D radar plane, around the radar. Return 
  # the range and azimuth for each point compared to the radar centre. 
  # The radar is assumed to be exactly at the centre of the centre pixel 
  # in the grid.
  #
  # Args:
  #  res: The grid resolution [m] (default: 75 m).
  #  maxDist: The maximum distance from the radar to plot [m] 
  #           (default: 40000 m). This is the distance on the axes.
  #
  # Returns: The grid as a data.frame containing x, y, azimuth and range.  
  
  n = maxDist %/% res
  x = seq(-n, n) * res
  
  grid = as.data.frame(expand.grid(x, x))
  names(grid) = c("x", "y")
  grid$range = sqrt(grid$x^2 + grid$y^2)
  
  # Get angle between x axis (E) and line from origin to point.
  grid$azimuth = atan2(grid$y, grid$x) * 180/pi 
  grid$azimuth = 90 - grid$azimuth # Shift to angle between point and y axis.
  grid$azimuth[which(grid$azimuth < 0)] = 
    360 + grid$azimuth[which(grid$azimuth < 0)]
  
  return(grid)
}

groundToRadarCoords = function(nc, grid, transform=TRUE, cache=FALSE) {
  # Take a set of spatial ground point values, and return the azimuth 
  # and range from the radar of each point. That is, project each point
  # on the ground to the corresponding point on the radar 2D plane,
  # and return those coordinates.
  # 
  # Args:
  #  nc: A NetCDF file for a PPI scan.
  #  grid: The grid of points as an SP object with CRS defined.
  #  transform: Transform the points to lat/lon before lookup?
  #             If FALSE then grid must be in lat/lon already.
  #  cache: Cache results by elevation? (Only indexed by elevation, 
  #         so be careful). (Default: FALSE).
  #
  # Returns: The grid as a data.frame, with azimuth and range added.
  
  # Get elevation angle. If cache is provided, return cached results
  # for this elevation angle.
  elevation = ncatt_get(nc, 0, "Elevation-value")$value # Radar elev [deg].
  if(cache) {
    cacheVar=paste("groundToRadarCoords_cache", elevation, sep="")
    if(exists(cacheVar, inherits=TRUE))
      return(get(cacheVar, inherits=TRUE))
  }
  
  # Get radar information: its location and elevation angle.
  centreLat = ncatt_get(nc, 0, "Latitude-value")$value  # Radar lat [degN].
  centreLon = ncatt_get(nc, 0, "Longitude-value")$value # Radar lon [degE].
  
  # Make a spatial object for the radar location, in lat/long and in metres.
  centreLatLong = data.frame(lat=centreLat, lon=centreLon)
  coordinates(centreLatLong) = ~lon+lat
  proj4string(centreLatLong) = CRS("+proj=longlat +datum=WGS84")
  
  # Convert the grid (a spatial object with CRS) to lat/long coordinates.
  if(transform) {
    gridLatLong = spTransform(grid, CRS("+proj=longlat +datum=WGS84"))
  } else {
    gridLatLong = copy(grid)
  }
  
  # Convert to a data.table.
  grid = data.table(data.frame(grid))
  
  ## Find the ground-level distance between the centre point and each 
  ## grid point; uses the Meeus method that uses an ellipsoid, from the 
  ## geospheres package. By default a WGS84 ellipsoid is used.
  ## TR (13.10.2017) Note this used to use distMeeus, but its documentation
  ## says distGeo is more accurate. On my test it made at most 30 cm
  ## difference.
  grid[, horizRange := distGeo(gridLatLong, centreLatLong)]
  
  # Project the ranges to the plane of the radar, given the scanning elevation.
  grid[, range := radarRangeFromGroundRange(horizRange, elevation)]

  # Add in the volume height.
  grid[, heightAboveRadar := volumeHeight(range, theta=elevation)]
  
  # Find the ground-level great-circle bearing between the radar and each 
  # grid point. If you start at the radar, point to this bearing, then walk
  # in a straight line, you will hit the requested point.
  grid[, azimuth := fastBearing(centreLatLong, gridLatLong)]
  
  # Cache result if required.
  if(cache) {
    assign(cacheVar, copy(grid), inherits=TRUE)
  }
  
  return(grid)
}

distMeeusFast = function(from, to, a=6378137, f=1/298.257223563) {
  ## A faster version of geosphere::distMeeus.
  ## from = multiple points, lat/long.
  ## to = one point, lat/long.
  toRad = pi/180
  from = data.frame(coordinates(from) * toRad)
  to = data.frame(coordinates(to) * toRad)
  names(from) = c("lon", "lat")
  names(to) = c("lon", "lat")
  from = data.table(from)
  to = data.table(to)
  
  res1 = from[, list(F = (lat + to$lat)/2,
                     G = (lat - to$lat)/2,
                     L = (lon - to$lon)/2)]
  res2 = res1[, list(sinG2 = (sin(G))^2,
                     cosG2 = (cos(G))^2,
                     sinF2 = (sin(F))^2,
                     cosF2 = (cos(F))^2,
                     sinL2 = (sin(L))^2,
                     cosL2 = (cos(L))^2)]
  res3 = res2[, list(S = sinG2 * cosL2 + cosF2 * sinL2,
                     C = cosG2 * cosL2 + sinF2 * sinL2)]
  res3[, w := atan(sqrt(S/C))]
  
  res4 = res3[, list(R = sqrt(S * C)/w,
                     D = 2 * w * a)]
  res5 = res4[, list(H1 = (3 * R - 1)/(2 * res3$C),
                     H2 = (3 * R + 1)/(2 * res3$S))]
  
  res3[, dist := 0]
  res3[w != 0, dist := res4$D * (1 + f * res5$H1 * res2$sinF2 * 
                                   res2$cosG2 - f * res5$H2 * 
                                   res2$cosF2 * res2$sinG2)]
  return(res3[, dist])
}

fastBearing = function(point, dests) {
  ## A version of the 'bearing' function from the Geosphere package,
  ## that uses data.table and less checks (does not check for the same points)
  ## to be faster.
  ## 
  ## Points must all be in longitude/latitude.
  ## 
  ## For usage see ?geosphere::bearing.
  
  toRad = pi/180
  point = data.frame(coordinates(point) * toRad)
  dests = data.frame(coordinates(dests) * toRad)
  names(point) = c("lon", "lat")
  names(dests) = c("lon", "lat")
  point = data.table(point)
  dests = data.table(dests)
  
  dests[, dLon := lon - point$lon] 
  dests[, y := sin(dLon)*cos(lat)]
  dests[, x := cos(point$lat)*sin(lat) - sin(point$lat)*cos(lat)*cos(dLon)]
  dests[, azm := ((atan2(y, x) / toRad) + 360) %% 360]
  return(dests[, azm])
}

latLongGrid = function(minLat=44.55508, # HYMEX SOP Ardeche area.
                       maxLat=44.61232, 
                       minLon=4.449667, 
                       maxLon=4.498717,                             
                       xRes=75, yRes=75, buffer=1000) {
  # Make grid of points evently spaced on the ground, at a certain resolution
  # within a lat/long bounding box. 
  # 
  # Args:
  #   minLat: Minimum latitude to plot [degrees N].
  #   maxLat: Maximum latitude to plot [degrees N].
  #   minLon: Minimum longitude to plot [degrees E].
  #   maxLon: Maximum longitude to plot [degrees E].
  #   buffer: Add a buffer around the plot area? [m] (default: 1000).
  #   xRes: X resolution of output image [m] (default: 75).
  #   yRes: Y resolution of output image [m] (default: 75).
  #
  # Returns: the grid as a set of spatial coordinates (an SP object).
  
  # Define bounding box.
  coords = data.frame(name=c("min", "max"),
                      lat=c(minLat, maxLat), 
                      lon=c(minLon, maxLon))
  coordinates(coords) = ~lon+lat
  proj4string(coords) = CRS("+proj=longlat +datum=WGS84")
  
  # Project these points to metres.    
  projString = MetresProjString
  projCRS = CRS(projString)
  metreCoords = spTransform(coords, projCRS)
  
  # Get the grid min and max points.
  minPoint = as.data.frame(subset(metreCoords, name=="min"))
  maxPoint = as.data.frame(subset(metreCoords, name=="max"))
  names(minPoint) = c("name", "x", "y")    
  names(maxPoint) = c("name", "x", "y")    
  
  # Create the grid to plot on.
  gridX = seq(minPoint$x - buffer, maxPoint$x + buffer, by=xRes)
  gridY = seq(minPoint$y - buffer, maxPoint$y + buffer, by=yRes)
  grid = expand.grid(x=gridX, y=gridY)
  
  # Turn the grid into a spatial object.
  gridSp = grid
  coordinates(gridSp) = ~x+y
  proj4string(gridSp) = projCRS
  
  return(gridSp)
}

extractRadarVerticalValues = function(nc, variables, thresholdSNR=5, ...) {
    ## Extract radar variables from vertical doppler profile files. For PPI
    ## or RHI scans, use extractRadarValues(). Currently only deals with 
    ## variables which are returned per range.
    ## 
    ## Args:
    ##   nc: The netCDF object to extract from.
    ##   variable: The variable to extract.
    ##   thresholdSNR: Threshold out values of reflectivity and drop
    ##                 concentrations if SNR < this value [dB]. If NULL,
    ##                 don't threshold.
    ##
    ## Returns:
  
    ## Get ranges from the file.
    ranges = ncvar_get(nc, "Range")
    
    ## Get attributes.
    NAVal = ncatt_get(nc, 0, "MissingData")$value # NA value [-].
    
    ## Get variable per range.
    radarData = data.frame(range=ranges)
    variableNames = NULL
    for(var in variables) {
        vals = ncvar_get(nc, var) 
        vals[which(vals == NAVal)] = NA
        
        ## For DSD concentrations, and reflectivities, 
        ## replace NAs and negatives with zeros.
        if(var == "N_nn")
            vals[which(is.na(vals) | vals < 0)] = 0
        if(var %in% c("Zh", "Zv", "ZhCorr", "ZvCorr"))
            vals[which(is.na(vals))] = -Inf
        vals = data.frame(vals)
        
        ## 2D result means there are multiple values per range.
        if(dim(vals)[2] != 1) {
            names(vals) = paste(var, seq(1, dim(vals)[2]), sep="")
        } else {    
            names(vals) = var
        }
        radarData = cbind(radarData, vals)
        variableNames = c(variableNames, names(vals))
    }

    if(!is.null(thresholdSNR)) {
        radarData = thresholdRadarData(radarData, snrThreshold=thresholdSNR)
        variableNames = c(variableNames, "thresholdedForSNR")
    }
    
    return(list(data=radarData, variables=variableNames))
}

thresholdRadarData = function(x, snrThreshold=5) {
    ## For radar records in which the signal to noise ratio is too low,
    ## set reflectivities and drop concentrations to zero.
    
    x = data.table(x)
    if(!("SNRh" %in% names(x)))
        stop("SNRh must be included for thresholding to take place.")

    for(col in names(x)) {
        if(col %in% c("Zh", "Zv", "ZhCorr", "ZvCorr"))
            x[is.na(SNRh) | SNRh < snrThreshold, (col) := -Inf]
        if(substr(col, 1, 4) == "N_nn")
            x[is.na(SNRh) | SNRh < snrThreshold, (col) := 0]
        if(col %in% c("Zdr", "ZdrCorr", "Kdp"))
            x[is.na(SNRh) | SNRh < snrThreshold, (col) := 0]
    }

    x[, thresholdedForSNR := FALSE]
    x[is.na(SNRh) | SNRh < snrThreshold, thresholdedForSNR := TRUE]
    return(x)
}

splitIntervals = function(intervals, n=1) {
  ## Split a set of intervals into non-overlapping sets.
  ## 
  ## Args: 
  ##   intervals: data.table with min, max for each interval.
  ##   n: Number of the first set.
  ## 
  ## Returns: a data.table with 'set' as the non-overlapping set number.
  
  setkey(intervals, max)
  mi = c(1)
  ma = intervals[1, max]
  
  if(intervals[, length(min)] == 0)
    return(NULL)
  
  if(intervals[, length(min)] == 1)
    return(intervals[, set := n])
  
  for(i in seq(2, length(intervals$min))) {
      if(intervals[i, min] >= ma) { ## Note >= allows min and max values to be
          ## the same; findInterval (used later) will deal with this so this
          ## allows for (slightly) faster execution.
      mi = c(mi, i)
      ma = intervals[i, max]
    }
  }
  
  intervals[mi, set := n]
  return(rbind(intervals[mi,], splitIntervals(intervals[-mi], n=n+1)))
}

extractRadarValues = function(nc, variables, coords, radarType="PPI",
    scaleBeamWidth=1, logScaleVars=LogScaleVars,
    transformFunc=function(x) {return(10^(x/10))},
    backtransFunc=function(x) {return(10*log10(x))},
    thresholdSNR=5, setNAtoZero=TRUE,
    ...) {
    ## Extract radar variables for specific angle and range combinations. Works
    ## for both PPI and RHI scans. For vertical scans, use 
    ## extractRadarVerticalValues().
    ##
    ## This function is used to extract radar values for *point* coordinates.
    ##
    ## Args:
    ##   nc: The netCDF object to extract from.
    ##   variables: The variables to extract.
    ##   coords: Coords to extract - each to have azimuth/range or 
    ##           elevation/range. The range should be the straight-line ranges 
    ##           from the radar centre point to the point requested.
    ##   scaleBeamWidth: Scaling factor to apply to beamwidth (default: 1).
    ##   transform: Transform returned values before taking the mean, and 
    ##              backtransform them after? Use for values in dBZ for example.
    ##              (Default: FALSE). 
    ##   transformFunc: Function to transform variables. (Default: dBZ to linear).
    ##   backtransFunc: Function to undo transform. (Default: linear to dBZ).
    ##
    ## Returns: data.frame with requested values per requested location.
    
    ## Get variables from NC file.
    elevations = ncvar_get(nc, "Elevation") # Angle above horizontal [degrees].
    azimuths   = ncvar_get(nc, "Azimuth")   # Angle clockwise from north [degrees].
    ranges     = ncvar_get(nc, "Range")     # Range from radar [m].
    
    ## Get attributes.
    NAVal     = ncatt_get(nc, 0, "MissingData")$value           # NA value [-].
    beamWidth = ncatt_get(nc, 0, "BW3dB-value")$value           # Beamwidth [deg].
    rangeRes  = ncatt_get(nc, 0, "RangeResolution-value")$value # Range res [m].
    
    ## For a PPI, angles are azimuths. For RHI angles are elevations.
    coords = data.table(coords)
    if(radarType == "PPI") {
        angles = azimuths
        stopifnot("azimuth" %in% names(coords))
        coords[, angle := azimuth]
    } else {
        angles = elevations
        stopifnot("elevation" %in% names(coords))
        coords[, angle := elevation]
    }
    
    ## Get variables from the NC file.
    radarValues = data.table(radRange=rep(ranges, length(angles)), 
        radAngle=rep(angles, each=length(ranges)))
    for(var in variables) {
        vals = ncvar_get(nc, var) 
        vals[which(vals == NAVal)] = NA
        radarValues$var = as.vector(t(vals)) # Vector by matrix row, ie range then angle
        setnames(radarValues, "var", var)
    }
    
    ## Scale the beamwidth if required.
    beamWidth = beamWidth * scaleBeamWidth
    
    ## Select only unique angles and ranges.
    angles = unique(angles)
    ranges = unique(ranges)
    
    ## Determine the angle and range classes in the radar file.
    angleIntervals = data.table(min=angles-beamWidth/2, max=angles+beamWidth/2, mid=angles)
    rangeIntervals = data.table(min=ranges-rangeRes/2, max=ranges+rangeRes/2, mid=ranges)
    
    ## Find non-overlapping sets of angle/range bins.
    angleIntervals = splitIntervals(angleIntervals)
    rangeIntervals = splitIntervals(rangeIntervals)
    
    ## Assign a range and angle 'set' to each radar point.
    setkey(radarValues, radAngle)
    setkey(angleIntervals, mid)
    radarValues[angleIntervals, angleSetNum := set]
    
    setkey(radarValues, radRange)
    setkey(rangeIntervals, mid)
    radarValues[rangeIntervals, rangeSetNum := set]

    stopifnot(radarValues[, !any(is.na(angleSetNum))])
    stopifnot(radarValues[, !any(is.na(rangeSetNum))])
    
    ## For each combination of non-overlapping set, 
    foundValues = NULL
    for(angleSet in angleIntervals[, unique(set)]) {
        ## Determine angle breaks.
        angleBreaks = c(angleIntervals[set == angleSet, min],
            angleIntervals[set == angleSet, max])
        angleBreaks = angleBreaks[order(angleBreaks)]
        
        for(rangeSet in rangeIntervals[, unique(set)]) {
            ## Determine range breaks.
            rangeBreaks = c(rangeIntervals[set == rangeSet, min],
                rangeIntervals[set == rangeSet, max])
            rangeBreaks = rangeBreaks[order(rangeBreaks)]
            
            ## Subset to the current angle and range set.
            rad = radarValues[angleSetNum == angleSet & rangeSetNum == rangeSet]
            
            ## Wipe angle and range bins clean.
            rad[, angleBin := NA]
            rad[, rangeBin := NA]
            coords[, angleBin := NA]
            coords[, rangeBin := NA]
            
            ## Put each radar and coordinate value into angle and range bins.
            rad[, angleBin := findInterval(radAngle, angleBreaks)]
            rad[, rangeBin := findInterval(radRange, rangeBreaks)]
            
            ## Put each requested coordinate into radar angle and range bins.
            coords[, angleBin := findInterval(angle, angleBreaks)]
            coords[, rangeBin := findInterval(range, rangeBreaks)]
            
            ## Note that, because the breaks are in the order c(min, max, min, max, ...)
            ## we only want to accept odd numbered cut values - to avoid selecting
            ## a value in a c(max, min) range.
            rad[angleBin %% 2 == 0, angleBin := NA]
            rad[rangeBin %% 2 == 0, rangeBin := NA]
            coords[angleBin %% 2 == 0, angleBin := NA]
            coords[rangeBin %% 2 == 0, rangeBin := NA]
            
            ## Look up radar values for each coordinate location. In the case of over-
            ## lapping radar bins, there may be more than one radar value per coordinate.
            ## allow.cartesian allows multiple rows per coordinate to be returned.
            rad = rad[!is.na(rangeBin) & !is.na(angleBin)]
            crd = coords[!is.na(rangeBin) & !is.na(angleBin)]
            setkey(rad, angleBin, rangeBin)
            setkey(crd, angleBin, rangeBin)
            f = rad[crd, allow.cartesian=TRUE][, c(names(coords), variables),
                             with=FALSE, allow.cartesian=TRUE] 
            foundValues = rbind(foundValues, f)
        }   
    }
    
    if(length(foundValues$range) == 0)
        return(NULL)

    ## Replace NAs in reflectivities with zeros (-Inf in dBZ).
    if(setNAtoZero) {
      stopifnot(!("var" %in% names(foundValues)))
      for(var in intersect(names(foundValues), c("Zh", "ZhCorr", "Zv", "ZvCorr"))) {
          setnames(foundValues, var, "var")
          foundValues[is.na(var), var := -Inf]
          setnames(foundValues, "var", var)
      }
  
      ## Replace Kdp, Zdr, ZdrCorr NAs with zeros.
      for(var in intersect(names(foundValues), c("Kdp", "Zdr", "ZdrCorr"))) {
          setnames(foundValues, var, "var")
          foundValues[is.na(var), var := 0]
          setnames(foundValues, "var", var)
      }
    }
      
    ## Threshold on SNR if required.
    foundValues[, thresholdedForSNR := NA]
    variables = c(variables, "thresholdedForSNR")
    if(!is.null(thresholdSNR)) {
        foundValues = thresholdRadarData(foundValues, snrThreshold=thresholdSNR)
    }
    
    ## In the case of multiple radar values per coordinate (rare), find the mean.
    foundValues = foundValues[, c(names(coords), variables), with=FALSE]
    newKey = names(coords)[which(names(coords) != "angleBin" & names(coords) != "rangeBin")]
    setkeyv(foundValues, newKey)
    
    if(any(logScaleVars %in% variables)) {
        vars = LogScaleVars[which(LogScaleVars %in% variables)]
        foundValues[, (vars) := as.data.table(transformFunc(foundValues[, vars, with=FALSE]))]
    }

    radarResults = foundValues[, lapply(.SD, mean, na.rm=T), .SDcols=variables, by=newKey]

    if(any(logScaleVars %in% variables)) {
        radarResults[, (vars) := as.data.table(backtransFunc(radarResults[, vars, with=FALSE]))]
    }

    stopifnot(length(coords[,1]) == length(radarResults[,1]))
    return(radarResults)
}

plotRadarRHIGrid = function(grid, variable, unit, azimuth="", scanTime=NULL,
                            ncolours=200, zlims=numeric(0), 
                            textSize=16, showTitle=TRUE, title=numeric(0), 
                            xLab="Range [km]", yLab="Height [km]",
                            scaleName=paste(variable, " [", unit, "]", sep=""),
                            crossHairs=TRUE, lineAtElevation=numeric(0),
                            colourBias=1, velocity=F, ...) {
  # Plot a radar RHI grid. 
  #
  # Args:
  #   grid: The grid, containing x, z, and the variable to plot.
  #   variable: The name of the variable to plot.
  #   unit: The units of the variable.
  #   azimuth: The azimuth (to display in the title).
  #   scanTime: The scan time to display in the title.
  #   ncolours: Number of colours to use in the scale (default: 200).
  #   zlims: Limits for the Z (colour) axis (default: range of data).
  #   textSize: Size of the text (default: 24).
  #   showTitle: Display the title? (Default: TRUE).
  #   xLab, yLab: Labels for x and y axes (default: "range", "height").
  #   scaleName: What name to use for the scale?
  #   crossHairs: Include crosshairs? (Default: TRUE).
  #   lineAtElevation: Draw a line at a certain elevation [deg]? 
  #                    (Default: none).
  #   colourBias: Bias colours (if > 1) to make breaks higher at larger 
  #               end (default: 1).
  #   velocity: If TRUE, plot with a red/blue colour scheme with white 
  #             as zero. Else use a normal colour scheme (default: FALSE).
  # 
  # Returns: A ggplot2 plot.
  
  # Define a set of colours to use.
  col.Rbar = c("darkblue","blue3","blue1","dodgerblue","deepskyblue","cyan")
  col.Rbar = c(col.Rbar,"yellow","gold","orange","red1","red3","darkred")
  if(velocity) {
    col.Rbar = c("blue3", "white", "red3")
  }
  
  col.Rbar = colorRampPalette(col.Rbar, bias=colourBias)
  col.Rbar = col.Rbar(ncolours)
  
  # Define z axis limits.
  if(length(zlims) == 0) {
    zlims = range(grid[[variable]], na.rm=TRUE)
  }
  
  # Define the colour scale.
  scale = scale_fill_gradientn(colours=col.Rbar, name=scaleName,
                               na.value="white", limits=zlims)
  lineScale = scale_colour_gradientn(colours=col.Rbar, name=scaleName,
                                     na.value="white", limits=zlims)
  
  # Plot the grid of points.
  grid$x = round(grid$x, 5)
  grid$y = round(grid$y, 5)
  plot = ggplot(grid, aes(x=x, y=y)) + 
    geom_raster(aes_string(fill=variable, colour=variable)) + 
    scale + lineScale +
    theme(legend.key.height = unit(0.6, "npc"))
  
  # Use km scale for the x and y axes if required.
  xScale = scale_x_continuous(breaks=seq(min(grid$x), max(grid$x), 
                                         length.out=5),
                              labels=round((seq(min(grid$x), max(grid$x), 
                                                length.out=5)) / 1000, 1),
                              expand=c(0,0))
    
  yScale = scale_y_continuous(breaks=seq(min(grid$y), max(grid$y), 
                                         length.out=5),
                              labels=round((seq(min(grid$y), max(grid$y), 
                                         length.out=5)) / 1000, 1),
                              expand=c(0,0))
    
  plot = plot + xScale + yScale + labs(x=xLab, y=yLab)  
  
  # Put in crosshairs if required.
  if(crossHairs) {
    plot = plot + geom_vline(aes(xintercept=0), colour="black", lty=2)
  }
  
  # Draw a line at an elevation if required.
  if(length(lineAtElevation) > 0) {
    plot = plot + geom_abline(intercept=0, 
                              slope=tan(lineAtElevation*pi/180),
                              lty=2)
  }
  
  # Set up and add the title.
  if(!is.null(scanTime)) {
    title = strftime(scanTime, "%Y-%m-%d %H:%M:%S", tz="UTC")
  }
  if(azimuth != "") {
    title = paste(title, ", ", variable, " [", unit, "]", 
                  " Az: ", azimuth, " deg.", sep="")
  }
  if(showTitle) {
    plot = plot + labs(title=title)
  }
  
  plot = plot + theme_bw(textSize) + 
    theme(panel.grid.major=element_blank(), 
          panel.grid.minor=element_blank())
  return(plot)
}

plotRadarPPIGrid = function(grid, variable, unit, elevation="", scanTime=NULL,
                            ncolours=200, zlims=numeric(0), kmScale=TRUE,
                            textSize=16, showTitle=TRUE, 
                            title=numeric(0), xLab="x", yLab="y",
                            additionalPoints=numeric(0),
                            includeAllAdditionalPoints=FALSE,
                            scaleName=paste(variable, " [", unit, "]", sep=""),
                            origin="bottomLeft",
                            crossHairs=FALSE, lineAtAzimuth=numeric(0), 
                            colourBias=1, velocity=FALSE, 
                            additionalPointsSize=4, 
                            addPointsColour="black", ...) {
  # Plot a radar PPI grid. 
  #
  # Args:
  #   grid: The grid, containing x, y, and the variable to plot.
  #   variable: The name of the variable to plot.
  #   unit: The units of the variable.
  #   elevation: The scan elevation (to display in the title).
  #   scanTime: The scan time to display in the title.
  #   ncolours: Number of colours to use in the scale (default: 200).
  #   zlims: Limits for the Z axis (default: range of data).
  #   kmScale: Convert the scale to kms from lower left corner (default: T).
  #   textSize: Size of the text (default: 24).
  #   showTitle: Display the title? (Default: TRUE).
  #   xLab, yLab: Labels for x and y axes (default: x, y, overwridden by 
  #               kmScale).
  #   additionalPoints: Additional points to display on the plot 
  #                     (default: none). Should be an SP object.
  #   includeAllAdditionalPoints: Force the plot to include all additional 
  #                               points, even if the grid doesn't cover 
  #                               their positions? (Default: FALSE).
  #   scaleName: What name to use for the scale?
  #   origin: Only valid when kmScale is TRUE, set to "bottomLeft" or "centre"
  #           to define the origin position. (Default: bottomLeft).
  #   crossHairs: Plot crosshairs at 0,0? (Default: FALSE).
  #   lineAtAzimuth: Plot a line at an azimuth [deg]? (Default: none).
  #   velocity: If TRUE, plot with a red/blue colour scheme with white 
  #             as zero. Else use a normal colour scheme (default: FALSE).
  #   additionalPointsSize: Size for additional point markers (default: 4).
  # 
  # Returns: A ggplot2 plot.
  
  # Define a set of colours to use.
  col.Rbar = c("darkblue","blue3","blue1","dodgerblue","deepskyblue","cyan")
  col.Rbar = c(col.Rbar,"yellow","gold","orange","red1","red3","darkred")
  if(velocity) {
    col.Rbar = c("blue3", "white", "red3")
  }
  
  col.Rbar = colorRampPalette(col.Rbar, bias=colourBias)
  col.Rbar = col.Rbar(ncolours)
  
  # Define z axis limits.
  if(length(zlims) == 0) {
    zlims = range(grid[[variable]], na.rm=TRUE)
  }
  
  # Define the colour scale.
  scale = scale_fill_gradientn(colours=col.Rbar, name=scaleName,
                               na.value="white", limits=zlims)
  lineScale = scale_colour_gradientn(colours=col.Rbar, name=scaleName,
                                     na.value="white", limits=zlims)
  
  # Plot the grid of points.
  grid$x = round(grid$x, 5)
  grid$y = round(grid$y, 5)
  plot = ggplot(grid, aes(x=x, y=y)) + 
    geom_raster(aes_string(fill=variable, colour=variable)) + 
    scale + lineScale
  
  # Use km scale for the x and y axes if required.
  if(kmScale) {
    correctionX = min(grid$x)
    correctionY = min(grid$y)

    if(origin == "centre" | origin == "center") {
      correctionY = 0
      correctionX = 0
    }

    xScale = 
      scale_x_continuous(breaks=seq(min(grid$x), max(grid$x), length.out=5),
                         labels=round((seq(min(grid$x), max(grid$x), 
                                           length.out=5)
                                       - correctionX) / 1000, 1),
                         expand=c(0,0))
    yScale = 
      scale_y_continuous(breaks=seq(min(grid$y), max(grid$y), length.out=5),
                         labels=round((seq(min(grid$y), max(grid$y), 
                                           length.out=5)
                                       - correctionY) / 1000, 1),
                         expand=c(0,0))
    
    plot = plot + xScale + yScale + labs(x="x [km]", y="y [km]")
  } else {
    plot = plot + labs(x=xLab, y=yLab)    
  }

  # Put in crosshairs if required.
  if(crossHairs) {
    plot = plot + geom_hline(aes(yintercept=0), colour="black", lty=2)
    plot = plot + geom_vline(aes(xintercept=0), colour="black", lty=2)
  }

  # Draw a line at an elevation if required.
  if(length(lineAtAzimuth) > 0) {
    plot = plot + geom_abline(intercept=0, 
                              slope=tan((-1*lineAtAzimuth-90)*pi/180),
                              lty=2)
  }
  
  # Set up and add the title.
  if(length(title) == 0) {
    if(!is.null(scanTime)) {
      title = strftime(scanTime, "%Y-%m-%d %H:%M:%S", tz="UTC")
    } else {
      title = ""
    }
    if(elevation != "") {
      title = paste(title, ", ", variable, " [", unit, "] ", 
                    "El: ", elevation, " deg.", sep="")
    }
  }
  if(showTitle) {
    plot = plot + labs(title=title)
  }
  
  # Add additional point markers if required.
  if(length(additionalPoints) > 0) {
    additionalPoints = spTransform(additionalPoints, CRS(MetresProjString))
    additionalPoints = data.frame(coordinates(additionalPoints))
    names(additionalPoints) = c("x", "y")
    
    if(!includeAllAdditionalPoints) {
      idx = which(additionalPoints$x >= min(grid$x) &
                  additionalPoints$x <= max(grid$x) &
                  additionalPoints$y >= min(grid$y) &
                  additionalPoints$y <= max(grid$y))
      additionalPoints = additionalPoints[idx,]      
    }    
    
    if(length(additionalPoints) == 0) break
    plot = plot + geom_point(data=additionalPoints, aes(x=x, y=y), 
                             pch=17, size=additionalPointsSize, 
                             col=addPointsColour)
  }
  
  plot = plot + theme_bw(textSize) + 
    theme(panel.grid.major=element_blank(),
          panel.grid.minor=element_blank())
  return(plot)
}

plotPPIForRegion = function(nc, variable, varUnits,
                            minLat=44.55508, # HYMEX SOP Ardeche area.
                            maxLat=44.61408, 
                            minLon=4.449667, 
                            maxLon=4.54605,                             
                            xRes=75, yRes=75, buffer=1000, ...) {
  # Plot a PPI for a specific geographical region, identified using a bounding
  # box.
  #
  # Args:
  #   nc: The NetCDF file for the PPI to plot.
  #   variable: The variable to plot.
  #   varUnits: The units for the variable.
  #   minLat: Minimum latitude to plot [degrees N].
  #   maxLat: Maximum latitude to plot [degrees N].
  #   minLon: Minimum longitude to plot [degrees E].
  #   maxLon: Maximum longitude to plot [degrees E].
  #   buffer: Add a buffer around the plot area? [m] (default: 1000).
  #   xRes: X resolution of output image [m] (default: 75).
  #   yRes: Y resolution of output image [m] (default: 75).
  #   ...: Optional extra arguments to plotRadarPPIGrid().
  #
  # Returns: A ggplot2 object.
  
  # Get the elevation and scan time information from the file.
  elevation = ncatt_get(nc, 0, "Elevation-value")$value
  meanScanTime = mean(ncvar_get(nc, "Time"))
  meanScanTime = as.POSIXct(meanScanTime, tz="UTC", 
                            origin=as.POSIXct("1970-01-01", tz="UTC"))
  
  grid = latLongGrid(minLat=minLat, maxLat=maxLat, minLon=minLon, 
                     maxLon=maxLon, xRes=xRes, yRes=yRes, buffer=buffer)
  grid = groundToRadarCoords(nc, grid)
  grid = extractRadarValues(nc=nc, variable=variable, coords=grid)
  plot = plotRadarPPIGrid(grid=grid, variable=variable, unit=varUnits, 
                          elevation=elevation, scanTime=meanScanTime, ...)
  
  return(plot)
}

plotPPIandRHI = function(ncPPI, ncRHI, PPIvariables, PPIunits,
                         RHIvariables=PPIvariables, 
                         RHIunits=PPIunits,
                         maxRange=40000, maxHeight=10000, 
                         res=75, zlims=numeric(0),
                         PPILegendPos="none", RHILegendPos="none",
                         plotMargins=unit(c(rep(.1, 3), .2), "lines"),
                         PPIcolourBias=rep(1, length(PPIvariables)),
                         RHIcolourBias=rep(1, length(RHIvariables)),
                         PPIvelocity=rep(FALSE, length(PPIvariables)),
                         RHIvelocity=rep(FALSE, length(RHIvariables)),
                         ...) {
  # Plot centred PPI and RHI around the radar, on one plot, for
  # multiple variables at once (one plot each).
  # 
  # Args:
  #   ncPPI: The NetCDF file for the PPI to plot.
  #   ncRHI: The NetCDF file for the RHI to plot.
  #   PPIvariable: The PPI variables to plot.
  #   PPIunits: The units for each PPI variable.
  #   RHIvariable: The RHI variables to plot (Default: same as PPI).
  #   RHIunits: The units for each RHI variable (Default: same as PPI).
  #   maxDist: The maximum distance from the radar to plot [m] 
  #            (Default: 40000 m).
  #   res: The resolution for both x and y axes [m] (default: 75 m).
  #   zlims: Variable limits to plot (by default uses range in both scans,
  #          but note that only one legend will be displayed so it's better
  #          to specify the zlims so they match).
  #   PPILegendPos: The legend position for the PPI plot (Default: none).
  #   RHILegendPos: The legend position for the PPI plot (Default: none).
  #   plotMargins: Margins for each of the two plots.
  #   PPIcolourBias, RHIcolourBias: The colour bias for PPI and RHI scans, 
  #                                 either one value or a vector of values 
  #                                 with a colour bias per variable
  #                                 (Default: 1 for all).
  #   PPIvelocity, RHIvelocity: Is each PPI/RHI variable a velocity? For 
  #                             the colour scale. (Default: FALSE for all).
  #   ...: Extra arguments to plotCentredPPI() and/or plotCentredRHI().
  #
  # Returns: A ggplot object.
 
  stopifnot(length(PPIvariables) == length(RHIvariables))
  
  # Get elevation and azimuth for PPI and RHI respectively.
  PPIelev = ncatt_get(ncPPI, 0, "Elevation-value")$value
  RHIazim = ncatt_get(ncRHI, 0, "Azimuth-value")$value
  
  print("Making PPI plots...")
  PPIplots = plotCentredPPI(nc=ncPPI, variables=PPIvariables, units=PPIunits,
                            maxDist=maxRange, res=res, zlims=zlims, 
                            lineAtAzimuth=RHIazim,
                            velocity=PPIvelocity,
                            colourBias=PPIcolourBias, ...)
  print("Making RHI plots...")
  RHIplots = plotCentredRHI(nc=ncRHI, variables=RHIvariables, units=RHIunits,
                            maxRange=maxRange, maxHeight=maxHeight, res=res, 
                            zlims=zlims, colourBias=RHIcolourBias,
                            velocity=RHIvelocity,
                            lineAtElevation=PPIelev, ...)
  
  print("Joining plots...")
  
  plotList = list()
  for(i in seq(1, length(PPIvariables))) {
    print(PPIvariables[i])
    
    PPIplot = PPIplots[[i]]
    RHIplot = RHIplots[[i]]
    
    # Fix the coordinate system to keep the PPIplot square.
    PPIplot = PPIplot + coord_fixed()
    
    # Get the legend from the RHI plot (extend its height first).
    RHIplot = RHIplot + theme(legend.key.height = unit(0.2, "npc"))
    tmp <- ggplot_gtable(ggplot_build(RHIplot))
    leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
    legend <- tmp$grobs[[leg]]
    
    # Strip out (or position) the legend for both plots.
    PPIplot = PPIplot + theme(legend.position=PPILegendPos) + 
      theme(plot.margin=plotMargins)
      
    RHIplot = RHIplot + theme(legend.position=RHILegendPos) + 
      theme(plot.margin=plotMargins)
            
    plots = arrangeGrob(RHIplot, PPIplot, nrow=2, ncol=1, 
                        heights=unit.c(unit(.25,"npc"), unit(.75, "npc")))
    
    widths = unit.c(unit(1, "npc") - sum(legend$widths), sum(legend$widths))
    heights = unit.c(unit(1, "npc"))
    
    plot = arrangeGrob(plots, legend, nrow=1, ncol=2, 
                        widths=widths, heights=heights, ...)
    plotList = c(plotList, list(plot))
  }
  
  names(plotList) = PPIvariables
  return(plotList)
}

plotCentredPPI = function(nc, variables, units, maxDist=40000, res=75,
                          zlims=rep(numeric(0), length(variables)),
                          colourBias=1, velocity=rep(FALSE, length(variables)), 
                          ...) {
  # Plot a centred PPI in the radar plane at a specified resolution. 
  #
  # Args:
  #   nc: The NetCDF file for the PPI to plot.
  #   variables: The variables to plot.
  #   units: The units for each variable.
  #   res: The resolution for both x and y axes [m] (default: 75 m).
  #   maxDist: The maximum distance from the radar to plot [m] 
  #            (Default: 40000 m).
  #   zlims: colour scale limits for each variable (default: auto).
  #   colourBias: The colour bias, either a single value or one per variable. 
  #               Numbers greater than 1 will mean colours are spread more
  #               for higher values (default: 1.).
  #   velocity: T/F whether each variable represents a velocity (default: F).
  #   ...: Extra arguments to extractRadarValues() or plotRadarRHIGrid().
  #
  # Returns: A ggplot object.
  
  # Get the elevation and scan time information from the file.
  elevation = ncatt_get(nc, 0, "Elevation-value")$value
  meanScanTime = mean(ncvar_get(nc, "Time"))
  meanScanTime = as.POSIXct(meanScanTime, tz="UTC", 
                            origin=as.POSIXct("1970-01-01", tz="UTC"))
  
  grid = radarPlaneGrid(maxDist=maxDist, res=res)
  grid = extractRadarValues(nc=nc, variables=variables, coords=grid, ...)
  if("ZhCorr" %in% variables) 
      grid = grid[!is.infinite(ZhCorr)]
  
  # Plot each variable.
  if(length(colourBias) == 1) {
    colourBias = rep(colourBias, length(variables))
  }
  plotList = list()
  for(i in seq(1, length(variables))) {
    zl = numeric(0)
    if(length(zlims) > 0) {
      zl = zlims[[i]]
    }
    
    plot = plotRadarPPIGrid(grid=grid, variable=variables[i], unit=units[i], 
                            elevation=elevation, scanTime=meanScanTime,
                            colourBias=colourBias[i], zlims=zl, 
                            velocity=velocity[i], crossHairs=TRUE, 
                            origin="centre", ...)
    plotList = c(plotList, list(plot))
  }
  names(plotList) = variables
  
  return(plotList)
}

plotCentredRHI = function(nc, variables, units, 
                          maxRange=40000, maxHeight=10000, res=75, 
                          zlims=rep(numeric(0), length(variables)), 
                          colourBias=1, velocity=rep(FALSE, length(variables)),
                          ...) {
  # Plot a centred RHI at a specified resolution. 
  #
  # Args:
  #   nc: The NetCDF file for the RHI to plot.
  #   variables: The variables to plot.
  #   units: The units for each variable.
  #   maxRange: The maximum range distance from the radar to plot [m] 
  #             (Default: 40000 m).
  #   maxHeight: The maximum height above the radar to plot [m] 
  #              (Default: 10000 m). 
  #   res: The resolution for both range and height axes [m] (default: 75 m).
  #   zlims: colour scale limits for each variable (default: auto).
  #   colourBias: The colour bias, either a single value or one per variable. 
  #               Numbers greater than 1 will mean colours are spread more
  #               for higher values (default: 1.).
  #   velocity: T/F whether each variable represents a velocity (default: F).
  #   ...: Extra arguments to extractRadarValues() or plotRadarRHIGrid().
  #
  # Returns: A ggplot object.
  
  # Get the elevation and scan time information from the file.
  azimuth = round(ncatt_get(nc, 0, "Azimuth-value")$value, 2)
  meanScanTime = mean(ncvar_get(nc, "Time"))
  meanScanTime = as.POSIXct(meanScanTime, tz="UTC", 
                            origin=as.POSIXct("1970-01-01", tz="UTC"))
  
  grid = RHIGrid(maxRange=maxRange, maxHeight=maxHeight, res=res)
  grid = extractRadarValues(nc=nc, variables=variables, coords=grid,
                            radarType="RHI", ...)
  
  # Plot each variable.
  if(length(colourBias) == 1) {
    colourBias = rep(colourBias, length(variables))
  }
  plotList = list()
  for(i in seq(1, length(variables))) {
    z = numeric(0)
    if(length(zlims) > 0)
      z = zlims[[i]]
        
    plot = plotRadarRHIGrid(grid=grid, variable=variables[i], unit=units[i], 
                            azimuth=azimuth, scanTime=meanScanTime,
                            colourBias=colourBias[i],
                            zlims=z, velocity=velocity[i], ...)
    plotList[[i]] = plot
  }
  names(plotList) = variables
      
  return(plotList)
}

getRadarVerticalValuesForClosestTime = function(radarDir, time, variables, 
    elevation, locations, maxAllowedTimeDiff=NULL, pattern=".*",
    logScaleVars=LogScaleVars, ...) {
  # Get vertical profile radar values for a time. The closest radar file 
  # (in terms of time) will be used to get the values.
  #
  # Args:
  #  radarDir: The directory in which the radar files are stored in directories
  #            by year, month, then day.
  #  time: Time to look for (POSIXct, UTC).
  #  variables: The radar variables to read.
  #  maxAllowedTimeDiff: The maximum allowed time difference [s]. If NULL
  #                      then no maximum is set (default: NULL).
  #  pattern: File name pattern to match.
  #
  # Returns: Data frame 
    
  closestFiles = 
    readClosestRadarTimestep(radarDir=radarDir, time=time, elevation=90,
                             radarType="VERTICAL_DOPPLER", pattern=pattern, 
                             getFileTimeFunc=getTimeFromNameVertical,
                             maxAllowedTimeDiff=maxAllowedTimeDiff, ...)

  if(length(closestFiles$scanTime) == 0)
    return(NULL)
  
  vals = NULL
  for(i in seq(1, length(closestFiles$scanTime))) {
    nc = closestFiles$nc[[i]]
    if(is.null(nc))
      next
    
    radarVals = extractRadarVerticalValues(nc=nc, variables=variables, ...)
    closeRadarFile(nc)
    vals = rbind(vals, radarVals$data)
  }

  # Find the mean per range over the time step.
  vals = averageRadarVals(vals, variables=radarVals$variables,
      logScaleVars=logScaleVars,
      statCols="range")
  
  # Record the scan times.
  vals$scanTimes = paste(closestFiles$scanTime, collapse=", ")
  return(list(data=vals, variables=radarVals$variables))
}

PPIRainMask = function(radarDir, time, elevation, locations,
    scanLength,
    maxAllowedTimeDiff=NULL,
    threshVar="SNRh", threshAmount=5, supergrid=NULL,
    metresCRS = CRS(paste("+proj=utm +zone=31",
        "+ellps=WGS84 +datum=WGS84",
        "+units=m +no_defs")),
    latLonCRS = CRS(paste("+proj=longlat +datum=WGS84",
        "+ellps=WGS84 +towgs84=0,0,0")),
    buffer=0, cacheCoords=FALSE, supergridLatLong=NULL,
    supergridRes=25, ZhThreshold=10, scaleBeamWidth=1, ...) {
    ## Get the radar-derived values for certain locations at a time,
    ## return 1 if rain was seen there and 0 if not.
    ##
    ## Args:
    ##  radarDir: The directory in which the radar files are stored in dirs
    ##            by year, month, then day.
    ##  time: Time to look for (POSIXct, UTC).
    ##  variables: The radar variables to read.
    ##  elevation: The radar elevation to use.
    ##  locations: Locations to get values for (gridded SP object with CRS).
    ##  scanLength: Expected (typical) radar scan length in seconds.
    ##  supergrid: Locations for which to sample radar values; will then be 
    ##             aggregated back to "locations". If null, subsampling of 
    ##             locations and the buffer will be used. 
    ##  maxAllowedTimeDiff: The maximum allowed time difference [s]. If NULL
    ##                      then no maximum is set (default: NULL).
    ##  threshVar: Threshold variable (default: "SNRh").
    ##  threshAmount: Threshold amount (unit of threshVar, default: 5 dB).
    ##  buffer: Include a buffer [m] around points? (Default: 0 m).
    ##          Must be a multiple of the resolution of the grid in locations.
    ##  cacheCoords: Cache coordinate arrays? (Default: FALSE).
    ##  superGridLatLong: Precalculated lat/long version of supergrid (optional).
    ##  ZhThreshold: Threshold for Zh to be rain.
    ##  ...: Extra arguments to extractRadarValues().
    ##
    ## Returns: locations with buffer added, and column "rainy" that can be either
    ##          1 (rain), 0 (no rain) or NA (no measurement).
  
    ## Open up the radar files for the time period.    
    closestFiles = readClosestRadarTimestep(radarDir=radarDir, time=time, 
        radarType="PPI", elevation=elevation,
        maxAllowedTimeDiff=maxAllowedTimeDiff,
        scanLength=scanLength)
  
    ## Return NULL if not found.
    if(length(closestFiles$scanTime) == 0)
      return(NULL)
  
    ## Locations should be a gridded SP object.
    if(!gridded(locations)) {
      gridded(locations) = TRUE
    }
  
    ## Ensure that the buffer is a multiple of the grid resolution.
    for(res in locations@grid@cellsize)
      stopifnot(buffer %% res == 0)
    
    ## Create a very high resolution grid around "locations". Note
    ## "Var1" and "Var2" become "x" and "y" after the resampling.
    if(is.null(supergrid)) {
      supergrid = subSampleGrid(grid=locations, xRes=supergridRes, 
                                yRes=supergridRes, buffer=buffer)
    }
    if(is.null(supergridLatLong)) {
      supergridLatLong = spTransform(supergrid, latLonCRS)
    }
    
    # require(parallel)
    # require(foreach)
    # require(doMC)
    # registerDoMC()
    
    # numCores = detectCores()
    options(warn=2) # Turn warnings into errors.
    
    ## Find the radar values for each high resolution grid point. For each
    ## point, return whether rain was seen there in any of the radar scans.
    vals = data.table(data.frame(supergrid))
    vals[, lon := coordinates(supergridLatLong)[,1]]
    vals[, lat := coordinates(supergridLatLong)[,2]]
    coordNames = copy(names(data.frame(coordinates(supergridLatLong))))
    setkey(vals, "lon", "lat")
    vals[, rainy := NA]
    for(i in seq(1, length(closestFiles$scanTime))) {
        nc = closestFiles$nc[[i]]
        if(is.null(nc))
            next
        
        coords = groundToRadarCoords(nc, supergridLatLong, transform=FALSE, 
                                     cache=cacheCoords)
   
        radarVals = extractRadarValues(nc=nc, variables=c(threshVar, "ZhCorr"),
                                       thresholdSNR=NULL, coords=coords, 
                                       setNAtoZero=TRUE, scaleBeamWidth=scaleBeamWidth)
        
        # ## Set up for parallel execution.
        # pointsPerCore = rep(dim(coords)[1] %/% numCores, numCores)
        # pointsPerCore[numCores] = pointsPerCore[numCores] + (dim(coords)[1] -
        #                  pointsPerCore[numCores]*numCores)
        # start = seq(1, numCores)
        # stopifnot(dim(coords)[1] == sum(pointsPerCore))
        # 
        # ## Function to run in parallel.
        # parallelGet = function(ncFile, i, pointsPerCore, threshVar, coords) {
        #   coreNC = nc_open(ncFile)
        #   start = (i*pointsPerCore[1])+1
        #   end = start + pointsPerCore[i+1]-1
        #   
        #   res = extractRadarValues(nc=coreNC, variables=c(threshVar, "ZhCorr"),
        #                            thresholdSNR=NULL, coords=coords[start:end,], 
        #                            setNAtoZero=TRUE)
        #   nc_close(coreNC)
        #   return(res)
        # }
        # 
        # #tm = proc.time()
        # ncFile=nc$filename
        # radarVals = foreach(i=seq(0, numCores-1), .combine='rbind') %dopar% 
        #   parallelGet(ncFile, i, pointsPerCore, threshVar, coords)
        # #print(proc.time() - tm)
        
        setnames(radarVals, threshVar, "testVar")
        setkeyv(radarVals, coordNames)
                    
        ## Determine whether each location is rainy. Rainy locations have a
        ## testVar (usually SNR) greater than threshold (usually 5 dB) 
        ## (implies not NA). 
        
        ## A note about NA values. In this case, NA values in ZhCorr or SNRh
        ## indicate a DRY region. Note that very occasionally NAs can also 
        ## appear in the middle of rain regions (ie due to rho_hv filtering, 
        ## beam blockage, etc) but this appears extremely rare.
        vals[radarVals[!(testVar <= threshAmount | ZhCorr <= ZhThreshold)], rainy := TRUE] ## Rainy regions.
        vals[vals[radarVals[(testVar <= threshAmount | ZhCorr <= ZhThreshold)]][is.na(rainy)], rainy := FALSE]  ## Below threshold is dry (do not "undo" any rainy pixel).
        vals[vals[radarVals[is.na(ZhCorr) | is.na(testVar)]][is.na(rainy)], rainy := FALSE] ## If the radar *recorded* NAs, take this as clear air and record dry region.
        
        closeRadarFile(nc)
    }

    ## ggplot(data.frame(vals), aes(x=x, y=y)) + geom_tile(aes(fill=rainy, colour=rainy)) + coord_fixed()
    
    ## Resample to the original grid size.
    coordinates(vals) = ~x+y
    proj4string(vals) = metresCRS
    gridded(vals) = TRUE
    vals = aggregateGrid(grid=vals, "rainy", 
                         resX=locations@grid@cellsize[1],
                         resY=locations@grid@cellsize[2])
    
    ## Once resampled, a cell with >= 50% subpixels rainy is counted as rainy.
    vals$rainy = as.numeric(vals$rainy >= 0.5)
    return(vals)
}

subSampleGrid = function(grid, xRes, yRes, buffer=NULL) {
  ## Subsample a grid of points to a higher resolution. The points are 
  ## taken to represent the centre of each pixel. 
  ##
  ## Args:
  ##   grid: The low-resolution grid to resample (SP, gridded).
  ##   resX: The new X resolution.
  ##   resY: The new Y resolution.
  ##   buffer: Add a buffer around the points? Specified in the same
  ##           unit as the resolution of the grid (Default: no buffer, NULL).
  ## 
  ## Returns: Resampled grid, as a gridded SP object.
  
  ## Convert to raster.
  grid = raster(grid)
  
  ## Add the buffer around the region if required.
  if(!is.null(buffer))
    grid = extend(grid, c(buffer / res(grid)[1], buffer / res(grid)[2]))
    
  ## Resample to high resolution.
  res(grid) = c(xRes, yRes) 
  
  ## Convert to regular gridded SP object.
  gridSP = data.frame(coordinates(grid))
  coordinates(gridSP) = names(gridSP)
  proj4string(gridSP) = proj4string(grid)
  gridded(gridSP) = TRUE
  return(gridSP)
}

aggregateGrid = function(grid, variable, resX, resY, fn=mean) {
  ## Do simple resampling of a value on a spatial grid.
  ##
  ## Args:
  ##  grid: The grid to resample. Points are assumed to be centres of pixels.
  ##  variable: The variable to resample. 
  ##  resX: The new X resolution.
  ##  resY: The new Y resolution.
  ##  fn: The function to use for sub-pixels to get the new pixel value 
  ##      (default: mean).
  ##
  ## Returns: The resampled spatial grid.
  ## Note: Only works on one variable at the moment. To do more would require
  ## use of RasterStack (create rasters r1, r2, etc, use stack()).
  
  ## Ensure we keep only the requested variable.
  for(n in names(grid)) {
    if(n != variable)
      grid[[n]] = NULL
  }
 
  ## Turn the grid into a raster object.
  grid = raster(grid)
  stopifnot(names(grid) == variable)
  
  ## Determine numbers of cells to include in each direction.
  factX = resX / res(grid)[1]  
  factY = resY / res(grid)[2]
  stopifnot(factX > 1 & factY > 1)
  
  ## Aggregate cell values.
  grid = aggregate(grid, fact=c(factX, factY), fun=fn, na.rm=TRUE)
  
  ## Convert to regular gridded SP object.
  gridSP = data.frame(coordinates(grid), getValues(grid)) 
  names(gridSP) = c("x", "y", variable)
  coordinates(gridSP) = names(data.frame(coordinates(grid)))
  proj4string(gridSP) = proj4string(grid)
  gridded(gridSP) = TRUE
  return(gridSP)
}

getRadarPPIValuesForClosestTime = function(radarDir, time, variables, 
                                           elevation, locations, 
                                           maxAllowedTimeDiff=NULL, 
                                           logScaleVars=LogScaleVars,
                                           ...) {
  # Get the radar-derived values for certain locations at a time. The
  # closest radar file(s) (in terms of time) will be used to get the
  # values.  If maxAllowedTimeDiff is specified, return average values
  # between time - maxAllowedTimeDiff and time.
  #
  # Args:
  #  radarDir: The directory in which the radar files are stored in directories
  #            by year, month, then day.
  #  time: Time to look for (POSIXct, UTC).
  #  variables: The radar variables to read.
  #  elevation: The radar elevation to use.
  #  locations: Locations to get values for (SP object with CRS).
  #  maxAllowedTimeDiff: The maximum allowed time difference [s]. If NULL
  #                      then no maximum is set and closest time returned
  #                      (default: NULL).
  #  logScaleVars: Which variables to convert to linear scale before averaging?
  #  ...: Extra arguments to extractRadarValues().
  #
  # Returns: Data frame the same as "locations" but with the radar variable 
  #          added as a column.

  closestFiles = readClosestRadarTimestep(radarDir=radarDir, time=time, 
      radarType="PPI", elevation=elevation,
      maxAllowedTimeDiff=maxAllowedTimeDiff, ...)
  
  if(length(closestFiles$scanTime) == 0)
    return(NULL)
  
  vals = NULL
  for(i in seq(1, length(closestFiles$scanTime))) {
    nc = closestFiles$nc[[i]]
    if(is.null(nc))
      next
    
    coords = groundToRadarCoords(nc, locations)
    radarVals = extractRadarValues(nc=nc, variables=variables, 
                                   coords=coords, ...)
    closeRadarFile(nc)
    vals = rbind(vals, radarVals)
  }
  if(is.null(vals))
    return(NULL)
  
  # Find the mean per location. Keep distance/height from radar.
  coordNames = c(names(locations), names(data.frame(coordinates(locations))))
  if("thresholdedForSNR" %in% names(vals))
      variables = c(variables, "thresholdedForSNR")
  vals = averageRadarVals(vals, variables=variables, logScaleVars=logScaleVars, 
      statCols=c(coordNames, "range", "horizRange", "heightAboveRadar"),
      ...)

  # Record the scan times.
  vals$scanTimes = paste(closestFiles$scanTime, collapse=", ")
  return(vals)
}

averageRadarVals = function(vals, variables, logScaleVars=LogScaleVars,
                            statCols=paste("number,name,label,lat,lon,",
                                           "altitude,x_metres,y_metres,",
                                           "horizRange,range,azimuth,angle,",
                                           "inRange", sep=""), ...) {
  # Average variables per station, in linear scale for those that are
  # originally in log scale.
  # 
  # Args:
  #  vals: Data.frame or data.table of values to work on.
  #  variables: Variables to average.
  #  logScaleVars: list of which variables are in log scale and should 
  #                be averaged in linear scale.
  #  statCols: Which cols are unique per station.
  #  
  # Returns: data.table with averaged values.
  
  vals = data.table(vals)
  
  # Convert log-scale variables into linear scale.
  for(var in variables[which(variables %in% logScaleVars)]) {
    vals[[var]] = 10^(vals[[var]]/10)
  }
  
  # Find the average of all variables.
  vals = vals[, lapply(.SD, mean), .SDcols=variables, by=statCols]
  
  # Convert log-scale variables back into log scale.
  for(var in variables[which(variables %in% logScaleVars)]) {
    vals[[var]] = 10*log10(vals[[var]])
  }
  
  return(vals)
}

getRadarVerticalTimeseries = function(radarDir, times, variables, 
                                      maxAllowedTimeDiff=NULL, ...) {
  # Get a timeseries of radar values from vertical profiles, return a 
  # data.table containing variables by vertical range and time.
  # 
  # Args:
  #  radarDir: The directory in which the radar files are stored in directories
  #            by year, month, then day.
  #  times: Series of times to find values for (POSIXct, UTC).
  #  variables: The radar variables to read.
  #  maxAllowedTimeDiff: The maximum allowed time difference [s]. If NULL
  #                      then no maximum is set (default: NULL).
  #  ...: Optional extra arguments to getRadarVerticalValuesForClosestTime().
  #  
  # Returns: A data.table containing requestedTime, scanTime, vertRange, 
  # and variables.
  
  allVals = data.table()
  stopifnot(is.numeric(maxAllowedTimeDiff))

  ## Cache the list of files to search through; for each elevation.
  pattern=".*.nc"
  files = list.files(radarDir, pattern=pattern,
      full.names=TRUE, recursive=FALSE)
  fileTimes = getTimeFromNameVertical(files)
  
  # Loop through times.
  for(time in times) {
    time = as.POSIXct(time, origin="1970-1-1", tz="UTC")
    
    vals = getRadarVerticalValuesForClosestTime(radarDir=radarDir, 
        time=time, variables=variables,
        maxAllowedTimeDiff=maxAllowedTimeDiff,
        files=files, times=fileTimes,
        ...)
    
    # No values? Try the next time step.
    if(is.null(vals$data)) next
    message(time)
    
    vals$data = data.table(vals$data)
    vals$data[, requestedTime := time]

    # Col names are modified if there is more than one measurement per range.
    # Discard rows for which all variables are NA or NaN.
    validRows = which(!is.na(rowSums(vals$data[, vals$variables, with=FALSE], na.rm=TRUE)))
    vals$data = vals$data[validRows,]
      
    if(dim(vals$data)[1] > 0) {
      allVals = rbind(allVals, vals$data, fill=TRUE)
    }
  }
  
  return(allVals)
}

getRadarPPITimeseries = function(radarDir, times, variables, elevations, 
    locations, maxAllowedTimeDiff=NULL, scanLength=20,
    keepColumns=c("name", "horizRange"),
    fileNameOffset=0, ...) {
  # Get a timeseries of radar values, return a data.table containing 
  # variables by radar elevation, time, and location.
  # 
  # Args:
  #  radarDir: The directory in which the radar files are stored in directories
  #            by year, month, then day.
  #  times: Series of times to find values for (POSIXct, UTC).
  #  variables: The radar variables to read.
  #  elevations: Radar elevation(s) to consider.
  #  locations: Locations to get values for (SP object with CRS).
  #  maxAllowedTimeDiff: The maximum allowed time difference [s]. If NULL
  #                      then no maximum is set (default: NULL).
  #  keepColumns: Names of columns in locations object to keep in output, as 
  #               well as variables, times, and coordinate of location.
  #               Default: name (of station), horizRange.
  #  fileNameOffset: offset to pass to getTimeFromNamePPI().
  #  ...: Extra arguments to getRadarPPIValuesForClosestTime().
  #  
  # Returns: A data.table containing requestedTime, scanTime, keepColumns, 
  # and variables.
  
  coordinateNames = attr(coordinates(locations), "dimnames")[[2]]
  keepColumns = c("requestedTime", "scanTimes", keepColumns, 
                  coordinateNames, variables, "elevation")
  
  allVals = data.table()

  # Cache the list of files to search through; for each elevation.
  files = list()
  fileTimes = list()
  for(el in elevations) {
      pattern=".*PPI.*.nc"

      pattern=sprintf(".*-PPI-(%03d).*.nc", el)
      files[[el]] =list.files(radarDir, pattern=pattern, 
          full.names=TRUE, 
          recursive=FALSE)
      fileTimes[[el]] = getTimeFromNamePPI(files[[el]],
                   offset=fileNameOffset)
  }
      
  # Loop through times.
  for(time in times) {
    time = as.POSIXct(time, origin="1970-1-1", tz="UTC")
    print(time)
    
    # Loop through elevations.
    for(el in elevations) {
      vals = 
        getRadarPPIValuesForClosestTime(radarDir=radarDir, 
                                        time=time,
                                        variables=variables,
                                        elevation=el, 
                                        locations=locations, 
                                        maxAllowedTimeDiff=maxAllowedTimeDiff,
                                        files=files[[el]],
                                        times=fileTimes[[el]],
                                        scanLength=scanLength)
      
      ## No values? Try the next time/elevation combination.
      if(is.null(vals)) next
      
      vals = data.table(vals)
      vals[, requestedTime := time]
      vals[, elevation := el]
      
      # Only keep values that are in the selected columns.
      vals = vals[, keepColumns, with=FALSE]  
      
      # Discard rows for which all variables are NA or NaN.
      validRows = which(!is.na(rowSums(vals[, variables, with=FALSE], na.rm=TRUE)))
      vals = vals[validRows,]
      
      if(dim(vals)[1] > 0) {
        allVals = rbind(allVals, vals)
      }
    }
  }

  return(allVals)
}

getRadarRHIProfileForClosestTime = function(radarDir, time, variables, 
                                            distance, maxAllowedTimeDiff=NULL, 
                                            pattern=".*RHI.*",
                                            logScaleVars=LogScaleVars,
                                            ...) {
  # Get a profile of radar values from an RHI, for a certain distance from
  # the radar, at a time. The closest radar RHI file (in terms of time) 
  # will be used to get the values.
  #
  # Args:
  #  radarDir: The directory in which the radar files are stored in directories
  #            by year, month, then day.
  #  time: Time to look for (POSIXct, UTC).
  #  variables: The radar variables to read.
  #  distance: The horizontal distance from the radar to extract.
  #  maxAllowedTimeDiff: The maximum allowed time difference [s]. If NULL
  #                      then no maximum is set (default: NULL).
  #  pattern: The file pattern to search for, use this to select the azimuth.
  #  logScaleVars: Variables that are in log scale and should be averaged in 
  #                linear scale.
  #  ...: Extra arguments to radarVerticalRHIProfile().
  #
  # Returns: Data frame the same as "locations" but with the radar variable 
  #          added as a column.
  
  closestFiles = readClosestRadarTimestep(radarDir=radarDir, time=time, 
                                          radarType="RHI", pattern=pattern,
                                          elevation=NULL,
                                          maxAllowedTimeDiff=maxAllowedTimeDiff)
  
  if(length(closestFiles$scanTime) == 0)
    return(NULL)
    
  vals = NULL
  for(i in seq(1, length(closestFiles$scanTime))) {
    nc = closestFiles$nc[[i]]
    if(is.null(nc))
      next
    
    profileVals = radarVerticalRHIProfile(nc=nc, distance=distance, 
                                          variables=variables, ...)
    closeRadarFile(nc)
    vals = rbind(vals, profileVals)
  }
  
  # Find the mean per station.
  vals = averageRadarVals(vals, variables, logScaleVars=logScaleVars,
                          statCols=c("height,elevation,range,angle,inRange"))
  
  # Record the scan times.
  vals$scanTimes = paste(closestFiles$scanTime, collapse=", ")
  return(vals)
}

getRadarRHIProfileTimeseries = function(radarDir, times, distance, variables, 
                                        maxAllowedTimeDiff=NULL, ...) {
  # Extract a timeseries of vertical profiles from RHI scans at a 
  # certain distance from the radar. 
  #
  # Args:
  #  radarDir: Directory to look in for files.
  #  times: Times to try to find (POSIXct, UTC).
  #  distance: Distance from radar to extract [m].
  #  variables: Radar variables to get.
  #  maxAllowedTimeDiff: Maximum difference between requested time and file 
  #                      time.
  #  ...: Extra arguments to getRadarRHIProfileForClosestTime().
  # 
  # Returns: A set of RHI profile series.
  
  allVals = data.table()
  
  # Loop through times.
  for(time in times) {
    time = as.POSIXct(time, origin="1970-1-1", tz="UTC")
    
    # Get the profile for this time.
    vals = 
      getRadarRHIProfileForClosestTime(radarDir=radarDir, time=time, 
                                       distance=distance, 
                                       variables=variables, 
                                       maxAllowedTimeDiff=maxAllowedTimeDiff, 
                                       ...)
      
    # No values? Try the next time/elevation combination.
    if(is.null(vals)) next
      
    vals = data.table(vals)
    vals[, requestedTime := time]
      
    if(dim(vals)[1] > 0) {
      allVals = rbind(allVals, vals)
    }
  }
  
  return(allVals)
}

radarVerticalRHIProfile = function(nc, distance, variables, 
                                   res=75, maxHeight=10000, ...) {
  # Extract vertical profiles from an RHI scan at a certain
  # distance from the radar. 
  #
  # Args:
  #   nc: The NetCDF object for the RHI scan.
  #   distance: The distance from the radar to extract.
  #   variables: List of variables to extract.
  #   res: Resolution to extract at.
  #   maxHeight: The maximum height above the radar to extract to.
  #   ...: Optional extra arguments to extractRadarValues(), eg scaleBeamWidth.
  #
  # Result: data.frame containing height and extracted variable values.
  
  # Find the range and elevation (above the radar plane) of each 
  # requested point.
  heights = seq(0, maxHeight, by=res)
  
  elevations = atan2(y=heights, x=distance) * 180/pi
  ranges = distance / cos(elevations / (180/pi))
    
  coords = data.frame(height=heights, elevation=elevations, range=ranges)
  vals = extractRadarValues(nc=nc, variables=variables, coords=coords, ...)
  return(vals)
}

plotVerticalProfile = function(vals, variable, unit,
                               textSize=12, title="",
                               ylab="Height [m]",
                               xlab=paste(variable, " [", unit, "]", sep="")) {
  # Plot a vertical profile.
  #
  # Args:
  #   vals: The values, returned by verticalRHIProfile().
  #   variable: The variable to plot.
  #   unit: The units for the variable.
  #   textSize: The font text size for the plot (default: 12).
  #   title: Title for the plot (default: none).
  #   xlab, ylab: Labels for x and y axes (default: height (y) and 
  #               var/unit (x)).
  # 
  # Returns: a ggplot2 object.
  
  plot = ggplot(vals, aes_string(x=variable, y="height")) + geom_path() +
    theme_bw(textSize) + labs(title=title, y=ylab, x=xlab)
  return(plot)
}

# Scrap function to find average DSD at a given height range from MXPOL 
# vertical scans. 
findAverageDSDAtHeight = function(radarDir, height) {
  
  files = list.files(radarDir, full.name=TRUE, pattern=".*profile-2013.*")
  allCounts = NULL
  for(file in files) {
    print(file)
    nc = nc_open(file)
      
    heights = ncvar_get(nc, "Range")
    height_idx = which.min(abs(heights - height))
    
    Z = ncvar_get(nc, "Zh")[height_idx]
    if(Z < 10) {
      closeRadarFile(nc)
      next
    }
    
    diams = ncvar_get(nc, "D_nn")[height_idx,]
    counts = ncvar_get(nc, "N_nn")[height_idx,]
      
    stopifnot(identical(dim(diams), dim(counts)))
    
    diams[which(diams == -99900)] = NA
    counts[which(counts == -99900)] = NA
        
    idx = which(!is.na(diams) & !is.na(counts))
    if(length(idx) > 0) {
        d = data.table(diam=diams, count=counts)
        allCounts = rbind(allCounts, d)
    }
      
    closeRadarFile(nc)
  }
  
  averageDSD = allCounts[!is.na(diam), list(conc=mean(count, na.rm=T)), by=diam]
  return(averageDSD)
}

volumeHeight = function(r, theta, a=6378.1e3, kea=4/3*a) {
  # Get the height of a radar volume above the ground.
  #
  # Taken from page 21 of "Doppler Radar and Weather Observations" by Doviak 
  # and Zrnia, written by Peter, converted by Tim.
  #
  # Args:
  #  r: Volume range from radar.
  #  theta: Radar elevation angle.
  #  a: Approx. Earth radius in metres.
  #  kea: Equivalent Earth radius.
  #
  # Returns: Volume height(s) taking into account atmospheric effects.
  
  res = sqrt(r^2 + kea^2 + 2*r*kea*sin(theta/(180/pi))) - kea;
  return(res)
}

volumeGroundRange = function(r, theta, a=6378.1e3, kea=4/3*a) {
  # Get the ground-range (along the earth's surface) of a volume from 
  # the radar.
  #
  # Taken from page 21 of "Doppler Radar and Weather Observations" by Doviak 
  # and Zrnia, written by Peter, converted by Tim.
  # 
  # Args:
  #  r: Volume range from radar.
  #  theta: Radar elevation angle [deg].
  #  a: Approx. Earth radius in metres.
  #  kea: Equivalent Earth radius.
  #
  # Returns: Volume range(s) (arc lengths).
  
  res = kea * asin(r * cos(theta/(180/pi)) / 
                     (kea + volumeHeight(r, theta, a, kea)))
  return(res)
}

radarRangeFromGroundRange = function(s, theta, a=6378.1e3, kea=4/3*a) {
  # This is a analytical inversion of the volumeGroundRange function.
  # It takes a range along the ground (ie with curvature!), and returns 
  # the radar range along the beam.
  #
  # Args:
  #  s: The ground (horizontal) range from the radar [m].
  #  theta: Radar elevation angle [deg].
  #  a: Approx. Earth radius in metres.
  #  kea: Equivalent Earth radius.
  #
  # Returns: Distance(s) from the radar along the beam.
  
  sol1 = (2 * (sqrt(kea^2 * cos(theta/(180/pi))^2 * sin(s/kea)^2 - 
                   kea^2 * cos(theta/(180/pi))^2 * sin(s/kea)^4) + kea * 
              sin(theta/(180/pi)) * sin(s/kea)^2)) / 
    (cos(2*(theta/(180/pi)))-2*sin(s/kea)^2+1)
  
  sol2 = (2 * (kea * sin(theta/(180/pi)) * sin(s/kea)^2 - 
                 sqrt(kea^2 * cos(theta/(180/pi))^2 * sin(s/kea)^2 - 
                        kea^2 * cos(theta/(180/pi))^2 * sin(s/kea)^4))) /
    (cos(2*(theta/(180/pi)))-2*sin(s/kea)^2+1)

  stopifnot(all(sol1 > 0) & all(sol2 < 0))
  return(sol1)
}

destPointEllipsoid = function(startLat, startLon, bearing, dist, 
                              a=6378137, b=6356752.314245) {
  # Calculate destination point(s), given a start point, bearing, and 
  # distance. Uses an ellipsoid model of the earth for greater accuracy.
  #
  # Based on code and info here:
  # http://www.movable-type.co.uk/scripts/latlong-vincenty.html
  #
  # Args:
  #   startLat, startLon: Lat/Long of starting point (WGS84).  
  #   bearing: Initial bearing to follow [deg].
  #   dist: Distance to travel [m].
  #   a, b: Major and minor semi-axes of Earth ellipsoid [m]. By default WGS84.
  #
  # Returns: data.frame with lat, lon (WGS84), and final bearing, in degrees.
  
  # Calculate f, flattening (a-b)/a.
  f = (a-b)/a
  
  # Convert to radians.
  bearing = bearing   / (180/pi)
  startLat = startLat / (180/pi)
  startLon = startLon / (180/pi)
  
  # Bearing B.
  sinB = sin(bearing);
  cosB = cos(bearing);
  
  tanU1 = (1-f) * tan(startLat)
  cosU1 = 1 / sqrt((1 + tanU1^2))
  sinU1 = tanU1 * cosU1
  sigma1 = atan2(tanU1, cosB)
  sinA = cosU1 * sinB
  cosSq = 1 - sinA^2
  uSq = cosSq * (a^2 - b^2) / (b^2)
  A = 1 + uSq/16384*(4096+uSq*(-768+uSq*(320-175*uSq)))
  B = uSq/1024 * (256+uSq*(-128+uSq*(74-47*uSq)))
  
  sigma = dist / (b*A)
  repeat {
    cos2SigmaM = cos(2*sigma1 + sigma)
    sinSigma = sin(sigma)
    cosSigma = cos(sigma)
    diffSigma = B * sinSigma * 
      (cos2SigmaM + B/4 * (cosSigma * (-1+2*cos2SigmaM^2) - 
                             B/6 * cos2SigmaM * (-3+4*sinSigma^2) * 
                             (-3+4*cos2SigmaM^2)))
    sigmaPrime = sigma
    sigma = dist / (b*A) + diffSigma
    if(all(abs(sigma-sigmaPrime) <= 1e-12)) break
  }
    
  tmp = sinU1 * sinSigma - cosU1 * cosSigma * cosB
  phi2 = atan2(sinU1 * cosSigma + cosU1 * sinSigma * cosB, 
               (1-f) * sqrt(sinA^2 + tmp^2))
  lambda = atan2(sinSigma * sinB, 
                 cosU1 * cosSigma - sinU1 * sinSigma * cosB)
  C = f/16 * cosSq * (4 + f*(4-3*cosSq))
  L = lambda - (1-C) * f * sinA *
    (sigma + C * sinSigma * (cos2SigmaM + C * cosSigma * 
                               (-1 + 2 * cos2SigmaM^2)))
  lambda2 = ((startLon + L + 3*pi) %% (2*pi)) - pi
  finalBearing = atan2(sinSigma, -tmp) 
  finalBearing = (finalBearing + 2*pi) %% (2*pi)

  # Convert back to degrees.
  finalBearing = finalBearing * (180/pi)
  destLat = phi2 * (180/pi)
  destLong = lambda2 * (180/pi)
  
  return(data.frame(lon=destLong, lat=destLat, finalBearing=finalBearing))
}

readRadarTimestep = function(time, radarDir, elevation, timeRes, locations,
    scanTime, variables, metresCRS = CRS(paste("+proj=utm +zone=31",
                             "+ellps=WGS84 +datum=WGS84",
                             "+units=m +no_defs")), ...) {
  # Read a radar timestep; that is to say the average values read by the radar
  # for grid points over a time period.
  #
  # Args:
  #  time: The time to look for (end of the timestep).
  #  radarDir: Radar files directory.
  #  elevation: Elevations to look for.
  #  timeRes: Temporal resolution [s].
  #  scanLength: estimate of time the radar takes to do one scan [s].
  #  locations: SP object with locations to find values for.
  #  metresCRS: CRS for metres (default UTM zone 31).
  #  ...: Optional extra arguments to extractRadarValues and averageRadarVals.
  #
  # Returns: a data.table containing x, y, and average value observed
  # over the time period and grid cell. Note that NaN values are removed from
  # the average (ie not counted as zeros).
  
  grid = data.frame(coordinates(locations))
  coordinates(grid) = coordinates(locations)
  proj4string(grid) = proj4string(locations)
  
  # Check if there is radar data available for this time.
  closestFiles = readClosestRadarTimestep(radarDir=radarDir, time=time, 
      elevation=elevation,
      maxAllowedTimeDiff=timeRes,
      scanLength=scanLength)
  if(is.null(closestFiles)) return(NULL)
 
  # 'closestFiles' contains all the files within the time period and elevations
  # chosen. Find the average value for each grid point.
  radar = NULL
  for(i in seq(1, length(closestFiles$scanTime))) {
    nc = closestFiles$nc[[i]]
    stopifnot(!is.null(nc))
    
    coords = groundToRadarCoords(nc, grid) 
    radarVals = extractRadarValues(nc=nc, variables=variables, coords=coords, ...)
    
    radar = rbind(radar, radarVals)
    closeRadarFile(nc)
  }
  
  coordNames = names(data.frame(coordinates(locations)))
  vals = averageRadarVals(radar, variables=variables, statCols=coordNames)
  
  return(vals)
}

