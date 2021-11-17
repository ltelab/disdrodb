# timeseries_functions.R
# 
# Functions to resample timeseries of data.
#
# Author: Tim Raupach <tim.raupach@epfl.ch>

library(plyr)
library(data.table)

convertTimeStringsToUnit = function(t, unit="secs") {
  # Convert time strings like "30 mins" or "1 hour" into numbers of 
  # seconds/minutes/hours.
  # 
  # Args:
  #  t: List of time strings to convert.
  #  unit: Unit to convert to (see ?difftime for available units) 
  #        (default: "secs").
  # 
  # Returns: numeric vector of seconds/minutes/hours or whatever unit is 
  # selected.
  
  st = as.POSIXct("2000-01-01", tz="UTC")
  res = NULL
  for(ts in t) {
    gaps = seq.POSIXt(st, by=ts, length.out=2)
    d = as.numeric(difftime(gaps[2], gaps[1], unit=unit))
    res = c(res, d)
  }
 
  return(res)
}

resampleTimeseries = function(data, timespan, dataColumnNames,
    fun, timestampColumnName="POSIXtime", quiet=FALSE,
    start=min(data[[timestampColumnName]])) {
  # Resample a data timeseries into different (larger) timesteps.
  # 
  # Args:
  #  data: The timeseries to resample. The time stamps are assumed
  #        to correspond to the END of each integration period.
  #  timespan: The timespan for each timestep in the new timeseries (eg 
  #            "5 min"). See ?cut.POSIXt for possible formats.
  #  dataColumnNames: The names of the data columns to resample.
  #  fun: The function to use to aggregate all steps within the new 
  #       timespan (default: sum).
  #  timestampColumnName: The name of the POSIXct timestamp column 
  #                       (default: "POSIXtime").
  #  start: Start of new timeseries (default: min time).
  #
  # Returns: a new timeseries in which each timestep is aggregated from 
  # substeps in the original timeseries.
  
  fun = match.fun(fun)
  
  # Cut times into intervals. We add the time resolution in seconds
  # to make sure the resampled times are the end of each integration period.
  resSeconds = convertTimeStringsToUnit(timespan, "sec")
  newTimes = seq(start-resSeconds, 
                 max(data[[timestampColumnName]])+(2*resSeconds),
                 by=timespan)
  times = data[[timestampColumnName]]
  times = as.POSIXct(cut.POSIXt(times + resSeconds, breaks=newTimes, 
                                right=TRUE), tz="UTC")
    
  res = NULL
  d = data.frame(time=times)
  
  for(name in dataColumnNames) {
    d = cbind(d, data.frame(data[[name]]))
  }
  names(d) = c("time", dataColumnNames)
      
  # Use some data.table magic to apply the function to each time
  # group for all columns in 'd' apart from 'time'.
  d = data.table(d)
  res = d[, lapply(.SD, fun), by="time"]

  res = data.frame(res, stringsAsFactors=F)
  names(res) = c("POSIXtime", dataColumnNames)
  
  return(res)
}

getCommonTimes = function(one, two, stationOne, stationTwo, cols) {
  # Get common times from two datasets and rbind into one dataset 
  # with station descriptions.
  #
  # Args:
  #   one: First data.frame or data.table containing POSIXtime.
  #   two: Second data.frame or data.table containing POSIXtime.
  #   stationOne: Name of station one.
  #   stationTwo: Name of station two.
  #   cols: Which columns to select (apart from POSIXtime).
  #
  # Returns: data.table containing POSIXtime, station, and cols,
  # for timesteps that both sets contain only.
  
  # Convert to data.table if required.
  one = data.table(one, key="POSIXtime")
  two = data.table(two, key="POSIXtime")
  
  # Get common times.
  one = one[(POSIXtime %in% two[, POSIXtime])]
  two = two[(POSIXtime %in% one[, POSIXtime])]
  
  # Sort by time.
  one = one[order(POSIXtime)]
  two = two[order(POSIXtime)]
  
  # Check times are truly the same.
  stopifnot(identical(one[, POSIXtime], two[, POSIXtime]))
  
  # Select columns and rbind.
  one = one[, c("POSIXtime", cols), with=FALSE]
  two = two[, c("POSIXtime", cols), with=FALSE]
  
  # Add stations.
  one[, station := stationOne]
  two[, station := stationTwo]
  
  # Rbind and return.
  return(rbind(one, two))
}
