# pluvio-functions.R
#
# Functions to read and examine raingauge data.
#
# Author: Tim Raupach <tim.raupach@epfl.ch>

# The default location to find pluvio datasets.
DefaultPluvioDir = "/ltedata/HYMEX/Grenoble/Grenoble_Pluvio_2010_to_2012/dat"
  
pluvioStationsDefinition = function() {
  # The default stations definition.
  # Must return a data.frame containing station number, name, 
  # label (for plotting), latitude (lat), and longitude (lon),
  # and file_pattern.
  # 
  # Files in which pluvio-functions.R is sourced should override 
  # the pluvioStationsDefinition function.
  
  stop("Pluvio stations are undefined!")
}

readPluvioDataset = function(startDate, endDate, stationName,
                             inDir = DefaultPluvioDir,
                             stations = pluvioStationsDefinition()) {
  # Read pluvio data from a file. 
  #
  # Args:
  #  startDate: Start timestamp (POSIXct).
  #  endDate: End timestamp (POSIXct).
  #  stationName: Name of station to read.
  #  inDir: Directory to look in (default: DefaultPluvioDir).
  #  stations: Stations definition (default: pluvioStationsDefinition()).
  #
  # Returns: The pluvio dataset as a data.frame containing timestamp, 
  #          POSIXtime, amount, for the specified period.
  
  stationId = which(stations$name == stationName)
  inFile = list.files(inDir, pattern=stations$file_pattern[stationId], 
                      full.name=T)
  print(inFile)
  stopifnot(length(inFile) == 1)
  
  data = read.table(inFile)
  names(data) = c("startYear", "startMonth", "startDay", 
                  "startHour", "startMinute", "startSecond",
                  "endYear", "endMonth", "endDay",
                  "endHour", "endMinute", "endSecond", "rainAmount")
  
  # Replace -999s with NAs.
  data$rainAmount[which(data$rainAmount == -999)] = NA
  
  dates = paste(data$startYear, "-", data$startMonth, "-", data$startDay,
                " ", data$startHour, ":", data$startMinute, ":", 
                data$startSecond, sep="")
  dates = as.POSIXct(dates, tz="UTC")
  
  start = as.POSIXct(startDate, tz="UTC")
  end = as.POSIXct(endDate, tz="UTC") + (60*60*24)
  
  idx = which(dates >= start & dates < end)
  res = data.frame(timestamp=strftime(dates[idx], format="%Y-%m-%d %H:%M:%S", 
                                      tz="UTC"), 
                   POSIXtime=dates[idx], 
                   amount=data$rainAmount[idx],
                   stringsAsFactors=F)
  return(res)
}

readAllPluvioData = function(startDate, endDate,
                             stations=pluvioStationsDefinition(),
                             inDir=DefaultPluvioDir) {
  # Read in all pluvio data from file.
  #
  # Args:
  #  startDate: Start timestamp.
  #  endDate: End timestamp.
  #  stationName: Name of station to read.
  #  inDir: Directory to look in (default: DefaultPluvioDir).
  #  stations: Stations definition (default: pluvioStationsDefinition()).
  #
  # Returns: The pluvio dataset as a data.frame containing station, 
  #          timestamp, POSIXtime, amount, for the specified period.
  
  pluvData = NULL
  
  for(stationName in stations$name) {
    print(paste("Reading data for pluvio at", stationName))
    stationData = readPluvioDataset(startDate, endDate, stationName, 
                                    inDir, stations)
    pluvData = rbind(pluvData, data.frame(station=stationName, stationData))
  }
  
  return(pluvData)
}

closestPluvio = function(s=stationsDefinition(),
                         p=pluvioStationsDefinition()) {
  # Find the closest Pluvio station to a Parsivel station.
  #
  # Returns a data.frame with station (parsivel), pluvio (gauge), and
  # the distance between the two in metres.
  
  numStations = length(s$number)
  pluvCoords = data.frame(lat=p$lat, long=p$lon)
  closest = NULL  
  
  coordinates(s) = ~lon+lat
  coordinates(p) = ~lon+lat
  proj4string(s) = CRS("+proj=longlat +datum=WGS84")
  proj4string(p) = CRS("+proj=longlat +datum=WGS84")
  
  dists = spDists(s, p, longlat=TRUE)
  
  # row corresponds to s, col to p
  # find min dist for each row
  pluvIds = apply(dists, 1, which.min)
  pluvDists = apply(dists, 1, min)
  results = data.frame(station=s$name, pluvio=p$name[pluvIds],
                       pluvioDistance=pluvDists * 1000)
  
  return(results)
}
