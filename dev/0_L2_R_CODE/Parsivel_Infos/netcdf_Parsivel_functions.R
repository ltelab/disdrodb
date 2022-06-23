# netcdf_Parsivel_functions.R
# 
# Functions to write Parsivel data to NetCDF files.
#
# Author: Tim Raupach <tim.raupach@epfl.ch>

library(ncdf4)
source("library/HYMEX_SOP_plot_functions.R")

createNC = function(outFile, stationID, stationName, stationLat, 
                    stationLon, stationAltitude, observationDate, title,
                    description, inst, cont, timeSteps, 
                    timeRes, diamClasses=32, velClasses=32,
                    classD=get.classD(), classV=get.classV()) {
  # Create a NetCDF file ready to add data to. 
  #
  # Args:
  #  outFile: The file to create.
  #  stationID: The numeric ID of the station.
  #  stationName: The name of the station.
  #  stationLat: The latitude of the station [deg N].
  #  stationLon: The longitude of the station [deg E].
  #  stationAltitude: The altitude of the station [m].
  #  observationDate: The date of the observations (string, UTC).
  #  title, description, inst, cont: Metadata.
  #  timeSteps: number of measurements.
  #  timeRes: Time resolution.
  # 
  # Returns: an NC object.
  # 
  # Notes:
  # The NetCDF files contain the following variables:
  # Time, PrecipCode, ParsivelStatusCode, RawDrops,
  # VolumetricDrops, CorrectedVolumetricDrops
  
  # Create dimensions.
  dim.time = ncdim_def(name="TimeStep", vals=1:timeSteps, units="-")
  dim.diam = ncdim_def(name="DiameterClass", vals=1:diamClasses, units="-")
  dim.vel = ncdim_def(name="VelocityClass", vals=1:velClasses, units="-")
  
  # Create the variables to go into the NetCDF file.
  var.time = ncvar_def(name="Time", units="Seconds since 1970-01-01 00:00:00", 
                       longname="Time at the end of the measurement interval",
                       dim=dim.time, missval=-999, prec="integer")
  
  var.precipCode = ncvar_def(name="PrecipCode", units="-", dim=dim.time,
                             longname="Precipitation code", 
                             missval=-999, prec="integer") 
  
  var.parsivelCode = ncvar_def(name="ParsivelStatusCode", units="-", 
                               dim=dim.time, missval=-999, prec="integer",
                               longname="Parsivel status code")
  
  var.parsivelIntensity = ncvar_def(name="ParsivelIntensity", 
                                    units="Millimeter PerHour", 
                                    dim=dim.time, missval=-999, 
                                    prec="double",
                                    longname=paste("Rain rate provided by",
                                                   "Parsivel instrument."))
  
  var.diamClassCenter = ncvar_def(name="DiameterClassCenter",
                                  units="Millimeter",
                                  dim=dim.diam, missval=-999,
                                  prec="double",
                                  longname=paste("Diameter class centers."))
  
  var.diamClassWidth = ncvar_def(name="DiameterClassWidth",
                                 units="Millimeter",
                                 dim=dim.diam, missval=-999,
                                 prec="double",
                                 longname=paste("Diameter class widths."))
  
  var.velClassCenter = ncvar_def(name="VelocityClassCenter",
                                 units="MetersPerSecond",
                                 dim=dim.vel, missval=-999,
                                 prec="double",
                                 longname=paste("Velocity class centers."))
  
  var.velClassWidth = ncvar_def(name="VelocityClassWidth",
                                units="MetersPerSecond",
                                dim=dim.vel, missval=-999,
                                prec="double",
                                longname=paste("Velocity class widths"))
  
  longCorrectedDescription = paste("Volumetric drop concentrations per",
                                   "equivolume diameter class, with",
                                   "outlier particle observations",
                                   "removed, velocities corrected, and",
                                   "concentrations corrected using a",
                                   "2D-video-disdrometer as a reference,",
                                   "using technique of Raupach and Berne",
                                   "2015 (DOI: 10.5194/amt-8-343-2015).")
  
  var.filtered = ncvar_def(name="CorrectedVolumetricDrops", missval=-999,
                           prec="double", units="PerCubicMeter PerMillimeter", 
                           dim=list(dim.time, dim.diam),
                           longname=longCorrectedDescription)
  
  var.rawdrops = ncvar_def(name="RawDrops", units="-", 
                           dim=list(dim.time, dim.diam, dim.vel), 
                           prec="integer", missval=-999,
                           longname=paste("Raw observed number of drops by",
                                          "velocity and equivolume diameter",
                                          "class."))
  
  var.unfiltered = ncvar_def(name="VolumetricDrops", missval=-999, 
                             prec="double", units="PerCubicMeter PerMillimeter", 
                             dim=list(dim.time, dim.diam),
                             longname=paste("Volumetric drop concentrations per",
                                            "equivolume diameter class."))
  
  vars = list(var.time, var.precipCode, var.parsivelCode, 
              var.parsivelIntensity, var.filtered, var.rawdrops, 
              var.unfiltered, 
              var.diamClassCenter, var.diamClassWidth,
              var.velClassCenter, var.velClassWidth)
  
  # Create the new NetCDF file with the variables.
  nc = nc_create(filename=outFile, vars=vars)
  
  # Add attributes for variables.
  ncatt_put(nc, varid="PrecipCode", attname="flag_values", 
            attval=c(-3, -2, -1, 0, 1))
  ncatt_put(nc, varid="PrecipCode", attname="flag_meanings", 
            attval=paste("ManuallyModified DropsFilteredByParsivel",
                         "NoDropsDetected LiquidPrecipitationOnly",
                         "NonZeroSolidPrecipitation"))
  
  # Add class info.
  ncvar_put(nc, varid="DiameterClassCenter", vals=rowMeans(classD))
  ncvar_put(nc, varid="VelocityClassCenter", vals=rowMeans(classV))
  ncvar_put(nc, varid="DiameterClassWidth", vals=apply(classD, 1, diff))
  ncvar_put(nc, varid="VelocityClassWidth", vals=apply(classV, 1, diff))
  
  # Add global attributes.
  ncatt_put(nc, varid=0, attname="Title", attval=title)
  ncatt_put(nc, varid=0, attname="Description", attval=description)
  ncatt_put(nc, varid=0, attname="Temporal_resolution", 
            attval=paste(timeRes, "seconds."))
  ncatt_put(nc, varid=0, attname="History", 
            attval=paste("Created", Sys.time()))
  ncatt_put(nc, varid=0, attname="Source", 
            attval=paste("Parsivel observations of drop counts per",
                         "velocity/equivolume diameter class, filtered for",
                         "quality control."))
  
  ncatt_put(nc, varid=0, attname="Institution", attval=inst)
  ncatt_put(nc, varid=0, attname="ContactInformation", attval=cont)
  ncatt_put(nc, varid=0, attname="GPSCoordSystem", attval="WGS84")
  ncatt_put(nc, varid=0, attname="Latitude_unit", attval="DegreesNorth")
  ncatt_put(nc, varid=0, attname="Longitude_unit", attval="DegreesEast") 
  ncatt_put(nc, varid=0, attname="Altitude_unit", attval="MetersAboveSeaLevel")
  
  # Add attributes specific to this set of observations.
  ncatt_put(nc, varid=0, attname="StationID", attval=as.integer(stationID))
  ncatt_put(nc, varid=0, attname="StationName", attval=stationName)
  ncatt_put(nc, varid=0, attname="Latitude_value", attval=stationLat)
  ncatt_put(nc, varid=0, attname="Longitude_value", attval=stationLon)
  ncatt_put(nc, varid=0, attname="Altitude_value", attval=stationAltitude)
  ncatt_put(nc, varid=0, attname="ObservationDate", observationDate) # "2012-10-07" ;
  
  return(nc)
}

getRawParsivelDataFromCSV = function(indir, station, date, stations,
                                     expectedlines=2880) {
  # Collate raw Parsivel data into a .Rdata object.
  #
  # Args:
  #   indir = The directory from which to read CSV files.
  #   station = Station number to read.
  #   date = Date (POSIXtime) to read.
  #   stations = Station definition.
  #   expectedlines = Number of lines to expect per file (default: 2880).
  # 
  # Returns: the contents of the CSV raw file as a data.table.
  
  stations = data.table(stations)
  
  parsivelRawCounts = NULL  
  
  file = list.files(indir, full.names=TRUE,
                    pattern=sprintf("%s_.*_%s.dat.gz", station, 
                                    strftime(date, "%Y%m%d")))
  if(length(file) != 1) {
    print(paste("No file for station", station, "on", 
                strftime(date, "%Y%m%d")))
    return(NULL)
  }
  
  stationNum = substr(basename(file), 1, 2)
  station = stations[number == stationNum, name]
  
  d = cbind(station, read.csv(file, header=F))
  names(d) = c("station", "POSIXtime", "parsivelStatus", "precipCode", 
               "parsivelR", paste("C", seq(1, 1024), sep=""))
  d$POSIXtime = as.POSIXct(d$POSIXtime, tz="UTC")
  stopifnot(dim(d) == c(expectedlines, 1029))
  
  d$precipCode = as.integer(d$precipCode)
  d$parsivelStatus = as.integer(d$parsivelStatus)
  
  return(data.table(d))
}

convertToNetCDF = function(rawCSVdir, unfilteredFile, filteredFile, 
                           stations, res, title, description, inst, cont,
                           dsdCols = paste("class", seq(1,32), sep=""),
                           ...) {
  # Convert a series of Parsivel readings to NetCDF files.
  #
  # Args:
  #  rawCSVdir: Raw CSV file directory.
  #  unfilteredFile: Rdata file for raw volumetric measurements.
  #  filteredFile: Rdata file for filtered measurements.
  #  stations: Stations definitions.
  #  res: Time resolution [s].
  #  title: Title to put into the NetCDF file.
  #  description: Description to put into the NetCDF file.
  #  dsdCols: Names of columns with drop counts (default: class1..class32).
  #
  # Returns: void.
  
  # Number of timesteps per day.
  timeStepsPerDay = 60*60*24/res
  
  # Load the data.
  unfiltered = data.table(get(load(unfilteredFile)))
  filtered = data.table(get(load(filteredFile)))
  
  # Get individual days from data.
  dates = strftime(filtered[, unique(trunc.POSIXt(POSIXtime, "days"))], 
                   format="%Y-%m-%d", tz="UTC")
  
  # Loop through dates.
  for(obsDate in dates) {
    start = as.POSIXct(obsDate, tz="UTC")
    end = start + 60*60*24
    
    allTimesUTC = seq(start, end-res, by=res, tz="UTC")
    allTimes = as.integer(allTimesUTC)
    stopifnot(identical(as.POSIXct(allTimes, origin="1970-1-1", tz="UTC"),
                        allTimesUTC))
    print(paste("Writing NetCDF file for", obsDate))
    
    # Loop through stations.
    for(s in seq(1, length(stations$name))) {
      stationID = stations$number[s]
      stationName = stations$name[s]  
      stationLat = as.character(stations$lat[s])
      stationLon = as.character(stations$lon[s])
      stationAltitude = as.character(stations$altitude[s])
      
      timesDT = data.table(POSIXtime=allTimesUTC, station=stationName)
      setkey(timesDT, POSIXtime, station)
      
      print(paste(" ... processing station:", stationName))
      
      # Subset filtered data for this date and station.
      filteredSub = filtered[POSIXtime >= start & POSIXtime < end & 
                             station == stationName]
      if(nrow(filteredSub) == 0) next
      
      # Get raw data for this date.
      raw = getRawParsivelDataFromCSV(rawCSVdir, station=stationID, 
                                      date=start, stations=stations, 
                                      expectedlines=timeStepsPerDay)
      if(is.null(raw))
        next
      stopifnot(all(raw[, POSIXtime >= start & POSIXtime < end & 
                          station == stationName]))
      stopifnot(identical(
      filteredSub[, list(POSIXtime, precipCode, parsivelStatus)],
        raw[, list(POSIXtime, precipCode, parsivelStatus)]))
              
      # Subset non-phys data for this date and station.
      unfilteredSub = unfiltered[POSIXtime >= start & POSIXtime < end &
                                 station == stationName]
      # Make sure everything lines up.
      stopifnot(identical(
        unfilteredSub[, list(POSIXtime, precipCode, parsivelStatus)],
        filteredSub[, list(POSIXtime, precipCode, parsivelStatus)]))
      
      setkey(unfilteredSub, POSIXtime, station)
      unfilteredSub = unfilteredSub[timesDT]
      unfilteredMatrix = as.matrix(unfilteredSub[, dsdCols, with=FALSE])
      
      # Convert non-phys and filtered arrays into timestep x diamclass matrices.
      setkey(filteredSub, POSIXtime, station)
      filteredSub = filteredSub[timesDT]
      filteredMatrix = as.matrix(filteredSub[, dsdCols, with=FALSE])
      
      # Create the NC file.
      outFile = paste(filePrefix, stationID, "_", obsDate, ".nc", sep="")
      nc = createNC(outFile=outFile, stationID=stationID, 
                    stationName=stationName, stationLat=stationLat, 
                    stationLon=stationLon, stationAltitude=stationAltitude, 
                    observationDate=obsDate, title=title, 
                    description=description, 
                    timeSteps=length(unique(raw$POSIXtime)),
                    timeRes=res, inst=inst, cont=cont, ...)
      
      # Write the NC variables.
      ncvar_put(nc, varid="Time", vals=allTimes)
      ncvar_put(nc, varid="PrecipCode", vals=filteredSub[, precipCode]) 
      ncvar_put(nc, varid="ParsivelStatusCode", vals=filteredSub[, parsivelStatus])  
      ncvar_put(nc, varid="ParsivelIntensity", vals=filteredSub[, parsivelR])
      ncvar_put(nc, varid="CorrectedVolumetricDrops", vals=filteredMatrix)
      
      ncvar_put(nc, varid="VolumetricDrops", vals=unfilteredMatrix)
      
      stopifnot(nrow(raw) == length(allTimes))
      rawMat = as.matrix(raw[, dsdRawCols, with=FALSE])
      dim(rawMat) = c(length(allTimes), 32, 32)
      # rawMat = aperm(rawMat, c(1, 3, 2)) ## NOT required because this inverses diameter/velocity classes.
      # Write the raw information as a 2D array for each timestep.
      ncvar_put(nc, varid="RawDrops", vals=rawMat)
                                   
      # Close the NetCDF file.
      nc_close(nc)
      
      # Zip the file.
      system(paste("gzip", outFile)) 
    }
  }
}
