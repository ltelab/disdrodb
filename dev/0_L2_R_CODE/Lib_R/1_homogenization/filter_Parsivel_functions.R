# filter_Parsivel_functions.R
#
# Functions to perform filtering of raw Parsivel data. 
#
# Author: Tim Raupach <tim.raupach@epfl.ch>

library(Hmisc)
source("third-party/Beard_Model.R")
source("filter_experiments/compare_drop_numbers_functions.R")

calibrateFilter = function(parsivelFile, dvdFile, events, stations,
                           stationsToUseForFilter, filterOutFile=numeric(0),
                           ...) {
  # Plot a comparison of drop numbers by drop diameter using Rdata files.
  #
  # Args:
  #   parsivelFile: The parsivel's Rdata file.
  #   dvdFile: The 2DVD's Rdata file.
  #   stations: The stations to plot for (by default, all).
  #   events: Events to plot for (default: SOP1 events).
  #   stationsToUseForFilter: Which stations are collocated (default: Pradels).
  #   filterOutFile: Optional. Which file to write the filter to? 
  #                  (Default: None).
  #   ...: Extra arguments to plotFilterFromDataFile() or saveFilter().
  #
  # Returns: A list containing: a ggplot object, the event statistics, and 
  # the derived filter, which is based only on a single run using the selected 
  # stationsToUseForFilter.
  
  res = plotFilterFromDataFile(parsivelFile, dvdFile, 
                               stations=stations, events=events, 
                               stationsToUseForFilter=
                               stationsToUseForFilter, ...)
  
  if(length(filterOutFile) > 0) {
    saveFilter(filter=res$filter, 
               dropNumberFilterFile=filterOutFile, ...)
  }
  
  return(res) 
}

terminalVels = function(altitude, lat, diams, seaLevelTemperature,
                        seaLevelRelativeHumidity=0.95, beardCol="Vt1") {
  # Find terminal velocities taking into account the station position, 
  # altitude, and the temperature.
  # 
  #  altitude: Station altitude above sea level [m].
  #  lat: Station latitude [degrees N].
  #  diams: Diameters for which to find the terminal velocity [mm].
  #  seaLevelTemperature: The temperature at sea level [deg. C] (default: 
  #                       15 degrees C).
  #  seaLevelRelativeHumidity: The relative humidity at sea level [-] 
  #                            (default: 0.95).
  #  diamClasses: The diameter class definition (min and max drop sizes
  #               for each class) [in mm] (default: Parsivel size classes).
  #  beardCol: Which column of beard result to use? "Vt1" for Beard 1976; 
  #            Pruppacher & Klett 1978, or "Vt2" for Beard 1977. (Default: 1).
  #
  # Returns: An array of terminal velocities, one for each drop diameter.
  
  # Relative humidity at sea level must be between 0 and 1 (fractional).
  stopifnot(seaLevelRelativeHumidity >= 0 & seaLevelRelativeHumidity <= 1)
  stopifnot(beardCol == "Vt1" | beardCol == "Vt2")
  
  Kelvin = 273.15 # Freezing temp of water [K].
  lapse  = 0.0065 # Atmospheric lapse rate [K/m].
  pa0    = 101325 # Pressure at sea level [Pascal].
  Rd     = 287.04 # Gas constant of dry air [J/(kg*K)].
  Ta0    = Kelvin + seaLevelTemperature # Sea level temp [K].
  
  velocities = NULL
  diams = diams / 1000 # Convert mm to [m]
  
  beardVals = Beard(diams, altitude, Kelvin, lapse, pa0, Rd, 
                    seaLevelRelativeHumidity, Ta0, lat)
  velocities = beardVals[[beardCol]]
  
  return(velocities)
}

terminalVelocitiesByClass = function(altitude, lat, diamClasses=get.classD(),
                                     ...) {
  # Find the terminal velocities for diameter class centre drops, 
  # taking into account the station position, altitude, and the 
  # temperature.
  #
  # Args:
  #  altitude: Station altitude above sea level [m].
  #  lat: Station latitude [degrees N].
  #  diamClasses: The diameter class definition (min and max drop sizes
  #               for each class) [in mm] (default: Parsivel size classes).
  #  ...: Further arguments to terminalVels().
  #
  # Returns: 
  #  An array of terminal velocities, one for the centre of each drop 
  #  diameter class.
  
  diams = rowMeans(diamClasses)
  return(terminalVels(altitude, lat, diams, ...))
}

createNonPhysicalFilterAtStation = function(stationAltitude,
                                            stationLatitude, 
                                            ...) {
  # Create a filter to remove non-physical drops from Parsivel
  # measurements made at a certain station.
  #
  # Args:  
  #  stationAltitude: Station altitude.
  #  stationLatitude: Station latitude.
  #  ...: More arguments to createNonPhysicalFilter().
  # 
  # Returns: The filter as a matrix, or as an array if asVector=T.
  
  terminalVels = terminalVelocitiesByClass(altitude=stationAltitude, 
                                           lat=stationLatitude, ...)
  filter = createNonPhysicalFilter(terminalVels)
  return(filter)
}

filterDrop = function(filter, 
                      velocity, 
                      diameter,
                      velocityClasses=get.classV(),
                      diameterClasses=get.classD()) {
  # Return whether a drop of a given velocity and diameter should be
  # filtered out.
  #
  # Args:
  #  filter: The filter to use. Matrix of 1024.
  #  velocity: The velocity of the drop to test.
  #  diameter: The diameter of the drop to test.
  #
  # Result: TRUE if the drop should be discarded, FALSE if it should be kept.
  
  diamClass = findInterval(diameter, diameterClasses[,1])
  velClass = findInterval(velocity, velocityClasses[,1])

  result = filter[velClass, diamClass]  
  return(result == 0)
}

defaultFilter = function(v, d) {
  # A default filter tolerance range function to use. Based on 2DVD
  # data from SOP 2013, removes ~0.2% of 2DVD drops.
  
  min = v-3
  max = v+4
   
  min[which(d < 2)] = 0
  return(cbind(min, max))
}

createNonPhysicalFilter = function(terminalVels,
                                   maxDiameter=7.5, 
                                   minDiameter=0,
                                   velocityFilterDiamRange=c(0,8),
                                   velocityTolFunc=defaultFilter,
                                   asVector=F,
                                   velocityClasses=get.classV(),
                                   diameterClasses=get.classD()) {
  # Create a Parsivel drop filter that removes drops that are non-physical.
  #
  # Args:
  #  terminalVels: the terminal velocity for drops in each class (for both
  #                class minimum and class maximum diameters). Use 
  #                terminalVelocitiesByClass().
  #  maxDiameter: the maximum (equivolume) drop diameter to accept [mm] 
  #               (default: 6 mm).
  #  minDiameter: the minimum (equivolume) drop diameter to accept [mm] 
  #               (default: 0 mm).
  #  velocityFilterDiamRange: The range of diameters over which to filter 
  #                           drops by their velocities [mm].
  #  velocityTolFunc: A function that takes a terminal velocity and a drop 
  #                   diameter and returns a range around that velocity which 
  #                   should be kept.
  #  asVector: Return as a vector by row? (Default: F, return as matrix).
  #  velocityClasses: velocity class definition (min and max drop velocities
  #                   for each class).
  #  diameterClasses: diameter class definition (min and max equivolume drop 
  #               diameter for each class).
  # 
  # Returns:
  #   A 32x32 matrix containing 1s where drops are allowed to be kept and zeros 
  #   where drops are to be filtered out from raw Parsivel data. Or, if
  #   asVector=T, the matrix converted into a vector by row.
  
  # Filter allows every class combination by default.
  numV = length(velocityClasses[,1])
  numD = length(diameterClasses[,1])
  filter = matrix(nrow = numV, ncol = numD, 1)
  
  # Find out which diameter classes are too large or small; filter them out.
  idx = which(diameterClasses[,2] > maxDiameter | 
              diameterClasses[,2] < minDiameter)
  filter[,idx] = 0
  
  # Now we filter based on the drop velocity, per diameter class.
  centreVels = rowMeans(velocityClasses)
  centreDiams = rowMeans(diameterClasses)
  minD = which(diameterClasses[,1] >= velocityFilterDiamRange[1])[1]
  maxD = which(diameterClasses[,2] >= velocityFilterDiamRange[2])[1]
  
  # Find out which velocity classes are too fast or slow; filter them out.
  # This depends on drop size so loop through diameter classes from min to max.
  for(d in seq(minD, maxD)) {
    
    # Find which classes are not plausible for the dth drop diameter class.
    # Based on the terminal velocity for this class.
    tVel = terminalVels[d]
    tDiam = centreDiams[d]
    range = velocityTolFunc(tVel, tDiam)
    
    if(all(is.na(range)))
      next
    
    minAllowedClass = findInterval(range[1], velocityClasses[,1])
    maxAllowedClass = findInterval(range[2], velocityClasses[,1])
    
    if(minAllowedClass > 1) {
      filter[seq(1, minAllowedClass-1), d] = 0
    }
    if(maxAllowedClass < length(velocityClasses[,1])) {
      filter[seq(maxAllowedClass+1, length(velocityClasses[,1])), d] = 0
    }
  }
  
  if(asVector) {
    filter = as.vector(t(filter))
  }
  
  return(filter)
}

stationFilters = function(stations, 
                          beardCol="Vt1",
                          seaLevelTemperature,
                          seaLevelHumidity=0.95,
                          diamClasses=get.classD(),
                          velClasses=get.classV(),
                          ...
                          ) {
  # Calculate the terminal drop velocities and the non-physical filter for 
  # each station in a list.
  # 
  # Args:
  #   stations: List of stations. Each must have a number, altitude, lat.
  #   seaLevelTemperature: Sea level temperature (used for terminal velocity 
  #                        calculation) [c]. (Default: 15 c).
  #   seaLevelHumidity: Sea level humidity (used for terminal velocity 
  #                     calculation) [-]. (Default: 0.95).
  #   diamClasses: Drop diameter classes - each row is min/max for a class [mm].
  #   velClasses: Drop velocity classes - each row is min/max for a class [m/s].
  #   beardCol: Which beard result to use? 1 for Beard 1976; 2 for Pruppacher &
  #             Klett 1978, or 3 for Beard 1977.
  #   ...: Optional extra arguments to createNonPhysicalFilter().
  #
  # Returns:
  
  velocities = list()
  filters = list()
  
  for(s in seq(1, length(stations$number))) {
    altitude=stations$altitude[s]
    lat=stations$lat[s]
    
    terminalVels = 
      terminalVelocitiesByClass(altitude=altitude, lat=lat, 
                                seaLevelTemperature=seaLevelTemperature, 
                                seaLevelRelativeHumidity=seaLevelHumidity, 
                                diamClasses=diamClasses, 
                                beardCol=beardCol)
    
    # Create a filter for non-physical drops.
    filter = createNonPhysicalFilter(terminalVels=terminalVels, 
                                     velocityClasses=velClasses, 
                                     diameterClasses=diamClasses, 
                                     asVector=TRUE, ...)
    velocities = c(velocities, list(terminalVels))
    filters = c(filters, list(filter)) 
  }
  
  return(list(velocities=velocities, filters=filters))
}

filterRawParsivel = function(rawArray,
                             stationDetails,
                             stationNum,
                             shiftVels=FALSE,
                             ...) {
  # Filter a raw array of 1024 classes.
  # 
  # Args:
  #   rawArray: The raw array to filter, ordered first by drop size then 
  #             by drop velocity; ie the array as provided by Parsivel.
  #   stationDetails: List returned by stationFilters() for the stations in use.
  #   stationNum: Which station number in the stationDetails lists?
  #   shiftVels: Shift velocities to match the beard curve? (Default: FALSE).
  #   ...: Arguments for shiftRawParsivelVelocities().
  #
  # Returns: The raw array, filtered.
    
  # Don't filter arrays that are just zeros!
  if(all(rawArray == 0))
    return(rawArray)
  
  # Find the terminal velocities for this location, and also its
  # non-physical drop filter.
  terminalVels = stationDetails$velocities[[stationNum]]
  filter = stationDetails$filters[[stationNum]]
    
  # Shift velocities if required, before filtered out non-physical drops.
  if(shiftVels) {
    rawArray = shiftRawParsivelVelocities(rawArray=rawArray,
                                          terminalVels=terminalVels,
                                          ...)$arr
  }
  
  # Apply the filter to remove non-physical drops from the array.
  rawArray[which(filter == 0)] = 0
  
  return(rawArray)
} 

shiftRawParsivelVelocities = function(rawArray, terminalVels,
                                      velClasses=get.classV(),
                                      resampleVelClassSize=0.1,
                                      padding=0, plotProgress=FALSE) {
  # Modify a raw Parsivel array so that each column's distribution of
  # drop counts has its mean over the Beard velocity. A parsivel array is
  # a flattened matrix (row-wise) in which columns signify diameter classes 
  # and rows signify velocity classes. This function shifts each diameter 
  # class' distribution of velocities up or down to match the mean to the 
  # expected terminal velocity for drops of that diameter class.
  # 
  # Args:
  #   rawArray: The raw Parsivel array, which is ordered first by diameter, 
  #             then by velocity.
  #   terminalVels: The per-diameter terminal velocities that we want to 
  #                 match. There must be the same number of terminal velocities
  #                 as there are diameter classes in the raw array.
  #   velClasses: Drop velocity classes - each row is min/max for a class [m/s].
  #   resampleVelClassSize: The width of resampled velocity classes 
  #                         (default: 0.1 m/s).
  #   padding: What value to pad with? (Default: 0).
  #   plotProgress: Produce before/after plots? (Default: FALSE).
  #
  # Returns: The modified array.
  
  if(all(rawArray == 0)) {
    return(rawArray)
  }
  
  # Convert the raw array into a matrix in which columns are diameters
  # and rows are velocities.
  mat = matrix(rawArray, ncol=length(terminalVels), byrow=T)
  resultMat = matrix(padding, ncol=length(terminalVels), nrow=nrow(mat))
  
  resampledClassMins = seq(0, max(velClasses), by=resampleVelClassSize)
  velClassWidths = round(apply(get.classV(), 1, diff), 1)
  
  newVelClasses = seq(min(velClasses), max(velClasses), by=resampleVelClassSize)
  newVelClasses = newVelClasses[1:(length(newVelClasses)-1)]
  
  # Loop through columns and shift each distribution of velocities 
  # so that its mean matches the terminal velocity. This requires 
  # resampling the velocity classes.
  diffsByClass=rep(NA, 32)
  for(i in seq(1, 32)) {
    velocities = mat[,i]
    
    if(all(is.na(velocities)))
      next
    
    if(max(velocities, na.rm=T) == 0)
      next
    
    # Resample into evenly spaced classes
    oldsum = sum(velocities, na.rm=T)
    velocities = splitClasses(velocities, velClassWidths, resampleVelClassSize)
    stopifnot(length(velocities) == length(newVelClasses))
    stopifnot(abs(sum(velocities, na.rm=T) - oldsum) < 0.0001)
                      
    meanVelocity = weighted.mean(newVelClasses, w=velocities)
    beardVelocityDiff = terminalVels[i] - meanVelocity
    shiftClasses = round(beardVelocityDiff / resampleVelClassSize, 0)
    diffsByClass[i] = shiftClasses * resampleVelClassSize
    
    if(shiftClasses > length(newVelClasses))
      next
    
    # Shift the velocities to move the median.
    velocities = shiftArray(velocities, shiftClasses)
       
    # Resample the shifted velocities into the non-evently spaced classes.
    velocities = combineClasses(velocities, resampleVelClassSize, 
                                velClassWidths)
    
    resultMat[,i] = velocities
  }
    
  result = as.vector(t(resultMat))
  
  if(plotProgress) {
    print(plotRawArray(rawArray, "Non-physical drops removed", zscale="linear"))
    print(plotRawArray(result, "After filter", zscale="linear"))
  }
  
  NtBeforeCorr = sum(rawArray, na.rm=T)
  NtAfterCorr = sum(result, na.rm=T)
  # stopifnot(abs(sum(rawArray, na.rm=T) - sum(result, na.rm=T)) < 10e-10)
  return(list(arr=result, NtBeforeCorr=NtBeforeCorr, NtAfterCorr=NtAfterCorr,
              shiftByClass=diffsByClass))
}

shiftArray = function(arr, n, padding=0) {
  # Shift all values in an array by 'n' array bins.
  # 
  # Args:
  #   arr: Array to shift values of.
  #   n: Number of entries to shift.. shift right if n > 0, left if n < 0.
  #   padding: What number to pad with (default: NA).
  # 
  # Return: The shifted array, padded.
  if(n == 0)
    return(arr)
  
  newArr = rep(padding, length(arr))
  
  nonZero = which(arr > 0)
  
  if(n > 0) {
    newArr[(n+1):length(newArr)] = arr[1:(length(arr)-n)]    
  } 
  
  if(n < 0) {
    n = n*-1
    newArr[1:(length(newArr)-n)] = arr[(n+1):length(arr)]    
  }
    
  return(newArr)
}

combineClasses = function(arr, currentWidth, newWidths, padding=0) {
  # Resample an array of values per even-width class into new 
  # uneven-width classes.
  # 
  # Args:
  #   arr: The array of values to resample. 
  #   currentWidth: The width of each array entry.
  #   newWidths: The new widths to use. Each new width should be a multiple
  #              of currentWidths. 
  #   padding: What to pad with? (Default: 0).
  #
  # Returns: The resampled array.
  
  newArr = rep(padding, length(newWidths))
  
  index = 1
  
  for(i in seq(1, length(newWidths))) {
    
    width = newWidths[i]
    numToSelect = width / currentWidth
    
    newArr[i] = sum(arr[index:(index+numToSelect-1)], na.rm=T)
    
    index = index + numToSelect
  }
  
  stopifnot(abs(sum(arr, na.rm=T) - sum(newArr, na.rm=T)) < 10e-5)
  return(newArr)
}

splitClasses = function(arr, classWidths, resampleWidth) {
  # Resample an array of values per uneven-width class into new 
  # even-width classes. 
  # 
  # Args:
  #   arr: The length-n array of values to resample, for n classes.
  #   classWidths: The existing class widths (length-n).
  #   resampleWidth: The new class width. Every class existing width must 
  #                  be a multiple of resampleWidth, and no existing classes 
  #                  can be smaller than this new width.
  # 
  # Returns: The resampled array.
  
  stopifnot(length(arr) == length(classWidths))
  stopifnot(!any(classWidths < resampleWidth))
  
  numNewClasses = classWidths / resampleWidth
  stopifnot(classWidths %% resampleWidth == 0)
  stopifnot(numNewClasses[which(numNewClasses > 1)] %% 2 == 0)
  
  centreVals = arr / numNewClasses
    
  result = NULL
  for(i in seq(1, length(arr))) {
    result = c(result, rep(centreVals[i], numNewClasses[i]))
  }

  # Check that counts in the bins have not been disturbed.
  stopifnot(abs(sum(result, na.rm=T) - sum(arr, na.rm=T)) < 10e-10)
  return(result)
}

applyDropConcentrationFilter = function(parsFile, filterDescFile, outFile,
                                        timeResSeconds, stations,
                                        calcRadarVars=FALSE,
                                        expectedlines=2880, ...) {
  # Apply drop concentration filters to Parsivel data.
  #
  # Args:
  #   parsFile: Rdata file of Parsivel data to be filtered.
  #   filterDesc: Rdata file holding a list containing:
  #        1. Filters: a list of of filters. Each filter must be a data.frame 
  #           containing dropSize [mm] (centre of class), dropPerc [mm] 
  #           (perc to keep), class [-] (class number).
  #        2. Breaks: a data.frame with from and to parsivelR values to apply 
  #           each filter to.
  #   outFile: Rdata file to write filtered data to.
  #   timeResSeconds: Number of seconds per timestep in input parsivel data.
  #   dsdCols: Columns of DSDs in input data.
  #   calcRadarVars: Calculate radar variables? (Default?: FALSE).
  #
  # Returns: void.
  
  filterDesc = get(load(filterDescFile))
  parsivelDSDs = get(load(parsFile))
  
  # Apply the filter, recalculate rain statistics, save.
  print("Applying filter to DSDs.")
  parsivelDSDs = applyFilters(filters=filterDesc$filters, 
                              parsivelData=parsivelDSDs, 
                              Rbreaks=filterDesc$breaks, 
                              timeResSeconds=timeResSeconds,
                              stations=stations, ...)
  save(parsivelDSDs, file=outFile)
}
