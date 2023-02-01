# Parsivel-functions.R
#
# Functions for reading and analysing information from a network of 
# Parsivel distrometers. This file contains Parsivel specific functions. For 
# functions to analyse information from a network of instruments, see
# network_functions.R.
#
# Author: Tim Raupach <tim.raupach@epfl.ch>

# Includes.
library(plyr)
library(reshape)
library(gstat)
library(RColorBrewer)
source("library/DSD_functions.R")
source("library/variogram_functions.R")
source("library/timeseries_functions.R")
source("library/network_functions.R")
source("third-party/Beard_Model.R")

reprocessAllParsivels = function(rawDir, rawCSVDir,
                                 noFilterDir, npDir, npVelShiftDir,
                                 filterDir, stations,
                                 filters, dataFileBase,
                                 rawTimeResolution,
                                 performFilter=TRUE, 
                                 overwriteRawCSVs=FALSE,
                                 overwriteNoFilter=FALSE, 
                                 overwriteNonPhys=FALSE,
                                 overwriteNonPhysVelShifted=FALSE,
                                 overwriteFiltered=FALSE,
                                 calcRadarVars=FALSE,
                                 rawPatterns=c(".*"),
                                 rawSkipLines=0, 
                                 fileEnd = "30sec.Rdata",
                                 resampledEnd = "5min.Rdata",
                                 resampleTimeRes = "5 min",
                                 ...) {
  # Reprocess all Parsivel files from completely raw to filtered.
  #
  # Args:
  #  rawDir: Input dir containing raw Parsivel files.
  #  rawCSVDir: Raw CSV output directory.
  #  noFilterDir: Unfiltered data output directory.
  #  npDir: Nonphysical filtered output directory.
  #  npVelShiftDir: Nonphysical and velocity filtered output dir.
  #  filterDir: Filtered output directory.
  #  stations: Stations definitions to use. 
  #  filters: The drop concentration filter files - an Rdata file as saved
  #           by filter calibration. See function applyDropConcentrationFilter().
  #  dataFileBase: Output base directory. Will have subdirs created within it.
  #  rawTimeResolution: Raw data time resolution [s].
  #  performFilter: Run the filter? (Default: TRUE). If false still runs
  #                 unfiltered and nonphysical filtered output.
  #  overwriteRawCSVs, overwriteNoFilter, overwriteNonPhys, overwriteFiltered,
  #  overwriteNonPhysVelShifted:                  
  #                  Overwrite raw, unfiltered, nonphysical, nonphysical
  #                  velocity shfited, and filtered files? (Default: FALSE 
  #                  for all).
  #  calcRadarVars: Calculate radar variables? (Default: FALSE because it's 
  #                 slow).
  #  rawPatterns: Patterns to match in raw file dir. (Default: all files.)
  #  rawSkipLines: Lines to skip in raw file for header. (Default: 0).
  #  resampledEnd: Filename endings for resampled Rdata files (Default: 
  #                5min.Rdata).
  #  resampleTimeRes: Time resolution(s) to resample to. 
  #  ...: extra arguments to processParsivelSet().
  #
  # Returns: void.
  
  stopifnot(length(resampledEnd) == length(resampleTimeRes))
  
  # Convert raw files to CSV files.
  updateAllCSVFromRaw(rawDataDir=rawDir, 
                      patterns=rawPatterns,
                      rawCSVOutDir=rawCSVDir, 
                      overwrite=overwriteRawCSVs,
                      skipLines=rawSkipLines)
  
  # Process using no filter at all.
  print("Converting raw CSV to unfiltered.")
  processAllRawCSVFiles(rawCSVDir, noFilterDir, timeRes=rawTimeResolution, 
                        stations=stations, overwrite=overwriteNoFilter, 
                        removeNonPhysDrops=FALSE, ...)
  processParsivelSet(parsivelDataDir=noFilterDir, 
                     outDataFile=paste(dataFileBase, "unfiltered_", fileEnd, sep=""),
                     outResampledDataFile=paste(dataFileBase, "unfiltered_", resampledEnd, sep=""),
                     rawTimeResolution=rawTimeResolution,
                     stations=stations,
                     calcRadarVars=calcRadarVars, 
                     resampleTimeRes=resampleTimeRes, ...)
  
  # Filter the raw CSV files for non-physical particles only.
  print("Converting raw CSV to non-phys filtered.")
  processAllRawCSVFiles(rawCSVDir, npDir, timeRes=rawTimeResolution, 
                        stations=stations, overwrite=overwriteNonPhys, 
                        removeNonPhysDrops=TRUE, ...)
  processParsivelSet(npDir, 
                     paste(dataFileBase, "filtered_nonphys_", fileEnd,sep=""),
                     paste(dataFileBase, "filtered_nonphys_", resampledEnd, sep=""),
                     rawTimeResolution=rawTimeResolution,
                     stations=stations,
                     calcRadarVars=calcRadarVars, 
                     resampleTimeRes=resampleTimeRes, ...)
  
  print("Converting raw CSV to non-phys filtered with velocity shift.")
  processAllRawCSVFiles(rawCSVDir, npVelShiftDir, timeRes=rawTimeResolution, 
                        stations=stations, overwrite=overwriteNonPhysVelShifted, 
                        removeNonPhysDrops=TRUE, shiftVels=TRUE, ...)
  processParsivelSet(npVelShiftDir, 
                     paste(dataFileBase, "filtered_nonphys_velshift_", fileEnd, sep=""),
                     paste(dataFileBase, "filtered_nonphys_velshift_", resampledEnd, sep=""),
                     rawTimeResolution=rawTimeResolution,
                     stations=stations,
                     calcRadarVars=calcRadarVars,
                     resampleTimeRes=resampleTimeRes, ...)
  
  # Filter the raw CSV files for drop count.
  if(performFilter) {
    print("Filtering drop concentrations.")
    
    resampledEnd = c(fileEnd, resampledEnd)
    resampleTimeRes = c(rawTimeResolution, convertTimeStringsToUnit(resampleTimeRes))
    
    ## Apply filter for resampled time resolutions.
    for(i in seq(1, length(resampleTimeRes))) {
      end = resampledEnd[i]
      seconds = resampleTimeRes[i]
      
      nonPhysFile = paste(dataFileBase, "filtered_nonphys_velshift_", end, sep="")
      filterOutFile = paste(dataFileBase, "filtered_CHECK_", end, sep="")
      
      print(paste("Applying filter for", nonPhysFile, "-", seconds, "seconds"))
      applyDropConcentrationFilter(parsFile=nonPhysFile, 
                                   filterDescFile=filters, 
                                   outFile=filterOutFile, 
                                   timeResSeconds=seconds,
                                   stations=stations, ...)
    }
  }
}

processParsivelSet = function(parsivelDataDir, 
                              outDataFile,
                              outResampledDataFile,
                              rawTimeResolution,
                              stations = stationsDefinition(),
                              patternFunc = function(station) { 
                                return(paste("^", sprintf("%02d", station), 
                                             "_.*", sep="")) 
                              },
                              fileSeparator=",", 
                              dsdCols=paste("class", seq(1,32), sep=""),
                              resampleTimeRes="5 min",
                              ensureStartAt=numeric(0),
                              calcRadarVars=FALSE,
                              fillZeros=FALSE,
                              processCodes=FALSE, ...) {
  # Process a parsivel data set and save Rdata files with rain
  # statistics added, at original and 5 minute temporal resolution.
  #
  # Args:
  #   parsivelDataDir: The directory to read ASCII files from.
  #   outDataFile: Output file name for Parsivel data.
  #   outResampledDataFile: Output file name(s) for resampled Parsivel data.
  #   rawTimeResolution: Time resolution of raw data [s].
  #   stations: Stations definition to use.
  #   patternFunc: Function to create a pattern from the station to match
  #                filenames to read.
  #   fileSeparator: Separator in files (default ",").
  #   dsdCols: Names of columns in which DSD is held.
  #   resampleTimeRes: Time resolution(s) to resample to (default: "5 min").
  #   ensureStartAt: Add a line at this time to ensure resampling starts 
  #                  at this time; ie a round hour. (default: none).
  #   calcRadarVars: Calculate radar variables? (Default: FALSE because it's 
  #                  slow).
  #
  # Returns: void.
  
  # Get Parsivel DSD data.
  parsivelDSDs = readAllParsivelData(parsivelDataDir, stations=stations,
                                     patternFunc=patternFunc, 
                                     sep=fileSeparator)
  parsivelDSDs = data.table(parsivelDSDs)
  
  ## Use processCodes if the original Parsivel weather codes need to be 
  ## translated to our simpler version.
  if(processCodes) {
    parsivelDSDs[, nonZero := rowSums(.SD) != 0, .SD=dsdCols]
    parsivelDSDs[, precipCode := parsivelPrecipCode(precipCode, parsivelR, nonZero)]
    parsivelDSDs[, nonZero := NULL]
  }
  
  # Remove lines for which the POSIXtime was NA; but put out a warning.
  if(any(is.na(parsivelDSDs$POSIXtime))) {
    warning("Undefined POSIXtime detected; error in input files.")
    parsivelDSDs = parsivelDSDs[which(!is.na(parsivelDSDs$POSIXtime)),]
  }
  
  # Add zeros if required. This is used when the instrument reports only
  # rainy time steps and omits zeros. Note that this then makes the assumption
  # that a missing time step means zero rain, which may not be true.
  if(fillZeros) {
    roundToSeconds = max(convertTimeStringsToUnit(resampleTimeRes))
    minTime = parsivelDSDs[, as.POSIXct((as.numeric(min(POSIXtime)) %/% roundToSeconds) * roundToSeconds, 
                                        origin="1970-1-1", tz="UTC") + rawTimeResolution]
    maxTime = parsivelDSDs[, as.POSIXct((as.numeric(max(POSIXtime)) %/% roundToSeconds) * roundToSeconds, 
                                        origin="1970-1-1", tz="UTC") + roundToSeconds]
    
    allTimes = parsivelDSDs[, seq(minTime, maxTime, by=rawTimeResolution)]
    zeroLine = parsivelDSDs[1,]
    zeroLine[, station := NULL]
    zeroLine[, precipCode := -1]  ## Dry timestep, even though manually added.
    zeroLine[, POSIXtime := NULL]
    zeroLine[, (dsdCols) := 0]
     
    zeros = NULL
    for(s in parsivelDSDs[, unique(station)]) {
       z = data.table(station=s, POSIXtime=allTimes, zeroLine)
       zeros = rbind(zeros, z)
    }
     
    setkey(zeros, station, POSIXtime)
    setkey(parsivelDSDs, station, POSIXtime)
    parsivelDSDs = rbind(parsivelDSDs, zeros[!parsivelDSDs])
    setkey(parsivelDSDs, station, POSIXtime)
  }
  
  # Add a line to ensure the start time.
  if(length(ensureStartAt) > 0) {
    stopifnot(ensureStartAt < min(parsivelDSDs$POSIXtime))    
    emptyLine = parsivelDSDs[1,]
    emptyLine$POSIXtime = ensureStartAt
    emptyLine[, (dsdCols) := NA]
    emptyLine[, precipCode := NA]
    emptyLine[, parsivelR := NA]
    emptyLine[, parsivelStatus := NA]
    emptyLine[, seq(3,length(emptyLine))] = NA
    for(station in unique(parsivelDSDs$station)) {
      emptyLine$station = station
      parsivelDSDs = rbind(emptyLine, parsivelDSDs)
    }
    setkey(parsivelDSDs, station, POSIXtime)
  }
  
  # Remove solid precipitation (code 1) and timesteps in which the Parsivel 
  # filtered out data (code -2). This leaves -1 for dry, 0 for rainy, -3 for 
  # removed by hand. Note that the code is left intact. Parsivel rain rates 
  # are set to NA for solid particle timesteps.
  parsivelDSDs[precipCode == 1, (dsdCols) := NA]
  parsivelDSDs[precipCode == -2, (dsdCols) := 0]
  parsivelDSDs[precipCode == 1, "parsivelR" := NA]
  
  # Resample and add rain stats. Note that because solid timesteps have
  # already been removed, the resampled values are the average of the
  # rainy (and zero) timesteps only.
  for(i in seq(1, length(resampleTimeRes))) {
    parsivelDSDsResampled = resampleParsivelByDSDs(data=parsivelDSDs, 
                                                   stations=stations,
                                                   timespan=resampleTimeRes[i], 
                                                   dsdCols=dsdCols,
                                                   radar=calcRadarVars, 
                                                   ...)
    save(parsivelDSDsResampled, file=outResampledDataFile[i])
    rm(list="parsivelDSDsResampled")
  }
  
  # Add rain stats (amount, drop concentration etc) for original spectra.
  parsivelDSDs = addRainStats(parsivelDSDs, timestepSeconds=rawTimeResolution,
                              radar=calcRadarVars, stations=stations, ...)
  
  # Save data.
  save(parsivelDSDs, file=outDataFile)
}

updateAllCSVFromRaw = function(rawDataDir, rawCSVOutDir,
                               rawDSDCol=21, RCol=5, statusCol=16,
                               patterns=c(".*"), skipLines=0,
                               zeroLine=paste(rep("000,", 1024), collapse=""),
                               overwrite=FALSE) {
  # Convert all raw Parsivel files in a directory to CSV files that are 
  # more easily readable by R. These output files will contain the date,
  # followed by parsivel status, precipitation code, and the drop counts
  # per diameter/velocity class.
  #
  # Args:
  #   rawDataDir: Directory in which to look for raw Parsivel output files.
  #   rawCSVOutDir: Output directory.
  #   rawDSDCol: Column in raw files containing the DSD (default: 21).
  #   RCol: Column in raw files containing parsivel rain rate (default: 5).
  #   statusCol: Column in raw files containing parsivel status (default: 16).
  #   patterns: Filename patterns to match (default: all).
  #   skipLines: No. of lines to skip at the top of each raw file (default: 0).
  #   zeroLine: Line that indicates all zeros.
  #   overwrite: Overwrite output files? (Default: FALSE).
  #
  # Returns: void.
  
  if(!file.exists(rawCSVOutDir, recursive=TRUE)) {
    dir.create(rawCSVOutDir, recursive=TRUE)
  }
  
  for(pattern in patterns) {
    files = list.files(rawDataDir, full.names=T, pattern=pattern)
    for(file in files) {
      print(file)
      
      outfile = paste(rawCSVOutDir, basename(file), sep="/")
      if(file.exists(paste(outfile, ".gz", sep="")) & !overwrite) {
        next
      }
      
      fileData = read.table(file, sep=",", header=F, 
                            stringsAsFactors=F, skip=skipLines)    
      
      times = as.POSIXct(fileData[,1], tz="UTC")    
      timestamps = as.character(fileData[,1])
      countMat = matrix(NA, nrow=length(timestamps), ncol=1024)
      code = rawParsivelCode(fileData)
      
      R = as.numeric(unlist(fileData[,RCol])) # Parsivel estimated rain rate.
      parsivelStatus = as.numeric(unlist(fileData[,statusCol]))
      
      process = which(!is.na(R))
      for(i in process) {
        cdata = fileData[i, rawDSDCol]
        if(identical(cdata, zeroLine) | cdata == "0" | cdata == "") {
          countMat[i, ] = 0
        } else {
          line = as.numeric(unlist(strsplit(cdata, ",")))
          if(length(line) != 1024) {
            # Sometimes corruption can occur and then strange characters
            # appear and screw up the line lengths. We ignore these lines.
            countMat[i, ] = 0
          } else {
            countMat[i, ] = line
          }
        }
      } 
      
      converted = data.frame(timestamps, parsivelStatus, 
                             code$precipCode, R, countMat)
      write.table(converted, outfile, col.names=F, row.names=F, sep=",")
      system(paste("gzip", paste(rawCSVOutDir, basename(file), sep="/")))
    }
  }
}

readAllParsivelData = function(datadir, 
                               stations=stationsDefinition(), 
                               patternFunc = function(station) 
                                 { return(paste(sprintf("%02d", station), 
                                                "_.*", sep="")) },
                               ...) {
  # Read in all parsivel DSD data from disk and put it into a data.frame.
  # Assumes that parsivel data will be stored in files with name like
  # station<num>.*.txt, with no headers.
  #
  # Args:
  #   datadir: Directory to read from.
  #   stations: Stations to read. Must contain name and number.
  #   patternFunc: A function that takes a station number and returns the 
  #                files to match for that station.
  #   ...: Extra arguments to readAllDSDFiles()
  # 
  # Returns: Data.frame containing all data from the directory, between
  #          the start and end dates.
  
  allDSDs = NULL
  DSDlist = list()
  
  # Build a list of data.frames, one per file.
  for(station in unique(as.numeric(stations$number))) {
    pattern = patternFunc(station)
    dsd = readAllDSDFiles(datadir, filePattern=pattern, header=F)#, ...)
    if(length(dsd) == 0) {
      print(paste("ERROR: no files found for station", station))
    }
    DSDlist = c(DSDlist, list(data.frame(station_name=station_name, dsd)))
  }
  
  # Stack all the data frames together in a fast way.
  allDSDs = rbind.fill(DSDlist)
  allDSDs$station = factor(allDSDs$station,
                           levels=stations$number,
                           labels=stations$name)
  
  return(allDSDs)
}

resampleParsivelByDSDs = function(data, stations, timespan="5 min", 
                                  dsdCols=paste("class", seq(1,32), sep=""), 
                                  ...) {
  # Resample Parsivel data by resampling first the DSD, then recalculating
  # the rain statistics.
  # 
  # Args:
  #  data: Parsivel DSD data to resample.
  #  stations: Station information (at least name, altitude, latitude).
  #  timespan: Time resolution to resample to (default: "5 min").
  #  dsdCols: Which column names are the DSD values?.
  #  ...: Optional extra arguments to addRainStats().
  # 
  # Returns: resampled data. Precip codes have their positive values averaged; 
  # a new column is added for "containedSusp" which indicates whether the 
  # aggregated timesteps contained suspicious timesteps, ie measurements 
  # that the Parsivel filtered out. Statistics are recalculated based on the
  # resampled DSDs.
  
  dsds = resampleParsivelDSDs(data, timespan, dsdCols)
  secs = convertTimeStringsToUnit(timespan, "sec")
  dsds = addRainStats(spectra=dsds, timestepSeconds=secs, stations=stations, 
                      ...)
  return(dsds)
}

resampleParsivelDSDs = function(data, timespan="5 min", dsdCols=paste("class", seq(1,32), sep="")) {
  # Resample DSDs (raw or 32 class) by station.
  #
  # Args:
  #  data: Parsivel DSD data to resample.
  #  timespan: Time resolution to resample to (default: "5 min").
  #  dsdCols: Which columns are the DSD values? (Default: seq(6, 37)).
  # 
  # Returns: resampled data. Precip codes have their positive values averaged; 
  # a new column is added for "containedSusp" which indicates whether the 
  # aggregated timesteps contained suspicious timesteps (ie measurements 
  # that the Parsivel already filtered out), and another new column is added 
  # for "containedNonZeroStatus" to say whether the Parsivel status was non-zero
  # during the aggregated timesteps.
      
  # Just to be sure it's in a data.frame and not a data.table.
  data = data.frame(data)
  
  # The precipCode flag has its positive values averaged; therefore it
  # becomes an indication of the proportion of the timestep that was solid 
  # precip. 
  codes = resampleNetworkData(data, timespan=timespan,
                              dataColumnNames=c("precipCode"), 
                              func=function(x) {
                                x[which(x < 0)] = 0
                                return(mean(x, na.rm=T))
                              })
    
  # containedSusp is a column that indicates whether the timesteps
  # that were resampled into this timestep contained any suspicious events,
  # ie timesteps that were set to zero by the Parsivel's own filter.
  containedSusp = resampleNetworkData(data, timespan=timespan,
                                      dataColumnNames=c("precipCode"), 
                                      func=function(x) {
                                        if(length(which(x == -2)) > 0) {
                                          return(TRUE)
                                        }
                                        return(FALSE)
                                      })
  names(containedSusp) = c("station", "POSIXtime", "containedSusp")
    
  # containedNonZeroStatus is a column that indicates whether the timestamps
  # that were resampled into this one contained any non zero Parsivel statuses.
  containedNonZeroStatus = resampleNetworkData(data, timespan=timespan,
                                               dataColumnNames=c("parsivelStatus"), 
                                               func=function(x) {
                                                 if(length(which(x != 0)) > 0) {
                                                   return(TRUE)
                                                 }
                                                 return(FALSE)
                                               })
  names(containedNonZeroStatus) = c("station", "POSIXtime", "containedNonZeroStatus")
  
  # Find the average Parsivel-derived rain rate.
  parsivelRainRates = resampleNetworkData(data, timespan=timespan,
                                          dataColumnNames=c("parsivelR"))
  
  # Other columns are drop volumetric drop concentrations and get AVERAGED.
  DSDs = resampleNetworkDSDs(data, timespan=timespan, dsdCols=dsdCols)
  
  stopifnot(identical(DSDs$POSIXtime, containedSusp$POSIXtime))
  stopifnot(identical(DSDs$POSIXtime, containedNonZeroStatus$POSIXtime))
  stopifnot(identical(DSDs$POSIXtime, codes$POSIXtime))
  stopifnot(identical(DSDs$POSIXtime, parsivelRainRates$POSIXtime))
  stopifnot(identical(DSDs$station, codes$station))
    
  res = data.frame(POSIXtime=codes$POSIXtime,
                   station=DSDs$station,
                   solidPrecipProp=codes$precipCode,
                   containedSusp=containedSusp$containedSusp,
                   containedNonZeroStatus=
                     containedNonZeroStatus$containedNonZeroStatus,
                   parsivelR=parsivelRainRates$parsivelR,
                   DSDs[,3:length(DSDs[1,])])
  return(res)
}

readParsivelCodesFromRawFile = function(station, date,
                                        rawDir=
                                          paste("/ltedata/HYMEX/SOP_2012/",
                                                "Parsivel/Raw_data/", sep=""), 
                                        pattern=paste(station, "_.*", date, 
                                           ".*", sep="")) {
  # Read from a Parsivel Raw data file the code for each timestep.
  # The results will be:
  # 
  #   -1 for a dry timestep
  #    0 for a timestep which is all liquid rain
  #    1 for a timestep in which solid precip was detected
  #   -2 for a timestep in which drops were recorded but were 
  #      filtered out by the Parsivel
  #
  # Args:
  #   station: Station number to read.
  #   date: Date in format YYYYMMDD.
  #   rawDir: Directory to find raw parsivel files.
  #   pattern: File pattern to match.
  # 
  # Returns:
  #   A data frame containing timestamp as string, POSIXtime and precipCode.
  
  file = list.files(rawDir, pattern=pattern, full.name=T)  
  stopifnot(length(file) == 1)
  
  pdata = read.table(file,sep=",")
      
  return(rawParsivelCode(pdata))
}
  
parsivelPrecipCode = function(code, parsivelRainRate, nonZeroDrops) {
  # Translate a parsivel code into our precip code.
  #
  # Possible codes are:
  #   -2 for a timestep in which drops were recorded but were 
  #      filtered out by the Parsivel
  #   -1 for a dry timestep
  #    0 for a timestep which is all liquid rain
  #    1 for a timestep in which solid precip was detected
  #
  # Args:
  #   code: Vector of parsivel codes (note: these should be Parsivel Code 4680).
  #   parsivelRainRate: Rain rate determined by the Parsivel for each code.
  #   nonZeroDrops: True or False for each code, whether there were any drops 
  #                 recorded. If unavailable set to FALSE.
  # 
  # Returns: our precip code for each code supplied.
  # Updated on 16.12.2016 to deal with both code 4680 and 4677.
  
  # Parsivel codes:
  # 61 - 63 means rain
  # < 61 means some or all drizzle
  # > 63 means some or all solid precip (snow or hail)
  ## Note if code 4677 is given, code 65 means heavy rain.
  
  naIdx = which(is.na(code))
  liquidIdx = which(code > 0 & code <= 65)
  solidIdx = which(code > 65) 
  drizzleIdx = which(parsivelRainRate == 0 & nonZeroDrops)
    
  # LTE codes:
  # NA means no data recorded.
  # -1 -> means no rain recorded (dry timestep).
  # 0  -> means timestep identified as all liquid (0 < Parsivel code <= 65).
  # 1  -> means timestep identified as solid (Parsivel code > 65).
  # -2 -> means timestep had (a few) drops that were filtered out by the 
  #       Parsivel (Parsivel code != 0 but Parsivel R = 0).
    
  lteCode = rep(-1, length(code))
  lteCode[naIdx] = NA
  lteCode[liquidIdx] = 0
  lteCode[solidIdx] = 1
  lteCode[drizzleIdx] = -2
  
  return(lteCode)
}

rawParsivelCode = function(pdata) {
  # From raw parsivel data, determine the precipitation code for each timestep.
  # 
  # Possible codes are:
  #   -2 for a timestep in which drops were recorded but were 
  #      filtered out by the Parsivel
  #   -1 for a dry timestep
  #    0 for a timestep which is all liquid rain
  #    1 for a timestep in which solid precip was detected
  #
  # Args:
  #   pdata: Raw parsivel data read using read.table(file, sep=",").
  # 
  # Returns: data.frame containing timestep as string, POSIXtime and 
  # precipCode.
    
  allZeroString = paste(rep("000,", 1024), collapse='')
  
  code = unlist(pdata[,7])
  nonZeroDrops = as.character(pdata[,21]) != allZeroString
  parsivelRainRate = unlist(pdata[,5])
  
  lteCode = parsivelPrecipCode(code, parsivelRainRate, nonZeroDrops)
  
  ts = as.character(unlist(pdata[,1]))    
  res = data.frame(timestamp=ts, 
                   POSIXtime=as.POSIXct(ts, tz="UTC"),
                   precipCode=lteCode, 
                   stringsAsFactors=F)
  return(res)
}

readAllRawParsivelData = function(datadir, start, end,
                                  stations=stationsDefinition()$number) {
  # Read raw data for all Parsivel stations.
  # 
  # Args:
  #   datadir: The directory to read from.
  #   start, end: The start and end times (POSIXct, UTC).
  #   stations: Stations numbers to read.
  #
  # Returns: data.frame containing all raw data.
  
  result = NULL
  for(s in stations) {
    res = readRawParsivelData(datadir, s, start, end)
    result = rbind(result, res)
  }
  return(result)
}

parsivelCollectionAreas = function(classes=get.classD(),
                                   width=30, length=180) {
  # Calculate the effective sampling area for each Parsivel drop diameter 
  # class.
  # 
  # Note that we used to use L(W-D/2) as per Battaglia et al. But their 
  # equation referred to older versions of the Parsivel in which 
  # edge particles were not properly removed. In newer versions, photo
  # diodes detect edge particles and remove them, meaning we use
  # L(W-D).
  # 
  # Args: 
  #   classes: The Parsivel drop diameter classes.
  #   width: Laser beam width [mm] (default: 180 mm).
  #   length: Laser beam length [mm] (default: 30 mm).
  #
  # Returns: A vector of collection areas in [m^2].
  
  classCentres = rowMeans(classes)
  classAreas = length * (width - (classCentres)) # [mm^2].
  classAreas = classAreas * 1e-6 # Convert to [m^2].
  return(classAreas)
}

readRawParsivelData = function(datadir, stationNum, start, end,
                               timestepsPerDay=2880) {
  # Read in raw parsivel DSD data from disk and put it into a data.frame.
  # Assumes that parsivel data will be stored in files with name like
  # <stationNum>_ascii_YYYMMDD.dat, with no headers.
  #
  # Raw files are ordered by diameter first, then velocity. So the first 32 
  # raw values are the 32 diameter classes, for the first velocity class.
  #
  # Args:
  #  datadir: Directory in which to find data.
  #  stationNum: The station number to read data for.
  #  start: Start time (POSIXct, UTC).
  #  end: End time to read (POSIXct, UTC).
  #  timestepsPerDay: Number of time steps to read per day (default: 2880).
  #
  # Returns:
  #  A data.frame containing timestamp, precipCode, then one column
  #  for each of the 1024 diameter/velocity classes. These rows can be
  #  turned into a 32x32 matrix of drop counts per velocity and 
  #  size class (row is velocity class, column is size class) using
  #  matrix(row[3:1026], ncol=32, nrow=32, byrow=T). Note these DSDs are
  #  raw unfiltered data.

  # Read list of files and subset for required dates.
  files = list.files(datadir, 
                     pattern=paste(stationNum, "_ascii_.*.dat", sep=""), 
                     full.name=T)
  fileDates = as.Date(substr(basename(files), 10, 17), format="%Y%m%d")
  files = files[which(fileDates >= as.Date(start) &
                        fileDates <= as.Date(end))]
  
  zeroMat = matrix(0, ncol=1024, nrow=timestepsPerDay)
  
  # Read in each required file.
  res = NULL
  for(file in files) {
    print(file)
    fileData = read.table(file, sep=",", header=F, stringsAsFactors=F)    
    timestamps = as.character(fileData[,1])
    
    stopifnot(length(timestamps) == timestepsPerDay)
    countMat = matrix(NA, nrow=timestepsPerDay, ncol=1024)
    R = unlist(fileData[,5])
    code = rawParsivelCode(fileData)
    
    # Set all dry timesteps to zero.
    id.dry = which(R < 0.01) 
    countMat[id.dry,] = 0
    code$precipCode[intersect(id.dry, which(code$precipCode == 0))] = -1
        
    # Test rainy timesteps.
    id.wet = which(R >= 0.01) # Use only timesteps with R>0.01 mm.
    
    for(i in id.wet) {
      cdata = as.character(fileData[i,21])
      cdata = strsplit(cdata, ",")
      counts = as.numeric(unlist(cdata))
      
      if(length(counts) != 1024 |
         any(is.na(counts))) {
        next
      }
      
      countMat[i,] = counts
    }
    
    fileRes = data.frame(timestamp = timestamps,
                         precipCode = code$precipCode,
                         countMat, stringsAsFactors=F)
    res = rbind(res, data.frame(fileRes, stringsAsFactors=F))
  }

  # Subset to times required.
  POSIXtime = as.POSIXct(res$timestamp, tz="UTC")
  idx = which(POSIXtime >= start & POSIXtime < end)
  return(res[idx,])
}

eventPercentageSolid = function(data, events=SOPevents()) {
  # From a set of parsivel data, return the percentage of solid
  # precipitation timesteps for each station.
  # 
  # Args:
  #  data: Parsivel DSDs with precipCode OR solidPrecipProp, POSIXtime.
  #  events: List of events with start, end as POSIXtimes.
  #
  # Returns: a data.frame with the percentage of solid precip for each
  # event, by station. Note names in the data.frame will be "fixed" using
  # make.names; so for example spaces are replaced by period symbols.
  
  res = data.frame()
  for(i in seq(1, length(events$start))) {
    start = events$start[i]
    end = events$end[i]
    
    if("solidPrecipProp" %in% names(data)) {
      data$precipCode = data$solidPrecipProp > 0
    }
    
    # Don't count dry timesteps; so we just select codes 0 (liquid) or 
    # 1 (solid).
    idx = which(data$POSIXtime >= start & 
                  data$POSIXtime <= end &
                  (data$precipCode == 0 | data$precipCode == 1))
    rows = data[idx,]
    
    if(length(rows$station) == 0) {
      print(paste("Warning: no data for event", i))
      next
    }
    
    eventResult = ddply(rows, .(station), summarise, 
                        solidPerc=(length(which(precipCode == 1)) / 
                                     length(precipCode) * 100))
    
    melted = melt(eventResult)  
    eventResult = cast(melted, variable ~ station)
    eventResult$variable = NULL
    
    res = join(res, data.frame(eventNum=i, start=start, end=end,
                               eventResult), type="full")
  }
  res$rem = NULL
  
  return(res)
}

plotRawArray = function(arr, timestring,
                        instrument="Parsivel",
                        diamClasses=get.classD(),
                        velClasses=get.classV(),
                        velocityTolFunc=function(x) {return(rep((x * 0.6),2))},
                        velocityTolMinDiam=0,
                        velocityTolMaxDiam=7.5,
                        diverge=integer(0),
                        plotBeardLine=T,
                        plotToleranceRange=F,
                        stationAltitude=278,
                        stationLat=44.58288,
                        scaleName="# drops",
                        numColours=3,
                        drawClasses=F,
                        xlimits=c(0,8),
                        ylimits=c(0,12),
                        zlimits=numeric(0),
                        title=T,
                        textSize=20,
                        legendPos="right",
                        showAxisLabels=T,
                        margins=numeric(0),
                        zscale="log",
                        colours=c("#FFFFFFFF", "#FF0000FF", "#FF2400FF", 
                          "#FF4900FF", "#FF6D00FF", "#FF9200FF", 
                          "#FFB600FF", "#FFDB00FF", "#FFFF00FF"),
                        overlayArray=numeric(0),
                        overlayLegend=FALSE,
                        overlayName="Filter", 
                        overlayLabels=c("Allowed", "Filtered out"),
                        seaLevelTemp=15) {
  # Plot a raw array of 32x32 diameter and velocity classes as returned by 
  # a Parsivel disdrometer. Drop numbers will be on a log scale.
  # 
  # Args:
  #   arr: The array to plot, length 1024, ordered first by diameter then by
  #        velocity classes. Ie the first 32 elements are per size classes for
  #        a single (the first) velocity class.
  #   timestring: A string to display in the plot title.
  # 
  # Returns: A ggplot ready to display.
  
  # x axis will be the centres of the diameter classes.
  x = rep(rowMeans(diamClasses), 32)
  xmins = rep(diamClasses[,1], 32)
  xmaxs = rep(diamClasses[,2], 32)
  xwidths = xmaxs - xmins
  
  # y will be centres of velocity classes.
  y = as.vector(matrix(rep(rowMeans(velClasses), 32), nrow=32, ncol=32, byrow=T))
  ymins = as.vector(matrix(rep(velClasses[,1], 32), nrow=32, ncol=32, byrow=T))
  ymaxs = as.vector(matrix(rep(velClasses[,2], 32), nrow=32, ncol=32, byrow=T))
  yheights = ymaxs - ymins
  
  # z will be the drop count corresponding to these xs and ys.
  toPlot = data.frame(diam=x, vel=y, z=arr)
  toPlot$z[which(toPlot$z == 0)] = NA
  
  if(length(zlimits) == 0) {
    zlimits = range(arr, na.rm=T)
  }
  
  # Choose a colour scheme based on whether the fill values are diverging 
  # or not.
  if(zscale == "log") {
    # Log scale can't handle zeros!
    zlimits[which(zlimits == 0)] = 0.01
    fill = scale_fill_gradientn(na.value="white", colours=colours,
                                name=scaleName, limits=zlimits, trans="log10")
  } else {
    fill = scale_fill_gradientn(na.value="white", colours=colours,
                                name=scaleName, limits=zlimits)
  }
  
  if(length(diverge) != 0) {
    fill = scale_fill_gradient2(name=scaleName, midpoint=diverge, 
                                limits=zlimits, na.value="white")
  }
    
  ylims = range((toPlot$vel+yheights))
  xlims = range((toPlot$diam+xwidths))
  if(length(xlimits) > 0) {
    xlims = xlimits
  }
  if(length(ylimits) > 0) {
    ylims = ylimits
  }
    
  if(plotBeardLine) {
    ## Plot the Beard 1976 relationship of diameter to terminal
    ## velocity on top of the raw plot.
    warning("Using default sea level temperature for velocities.")
    vels = terminalVelocitiesByClass(altitude=stationAltitude, lat=stationLat, seaLevelTemperature=seaLevelTemp)
    beardRelationship = data.frame(diam=rowMeans(diamClasses), 
                                   velocity=vels, row.names=NULL)
    
    beardRelationship$upper = beardRelationship$velocity + 
      velocityTolFunc(beardRelationship$velocity)[1]
    beardRelationship$lower = beardRelationship$velocity - 
      velocityTolFunc(beardRelationship$velocity)[2]
        
    beardRelationship$upper[beardRelationship$upper > max(ylims)] = max(ylims)
    beardRelationship$lower[beardRelationship$lower < min(ylims)] = min(ylims)
        
    beardLine = geom_line(data=beardRelationship, aes(x=diam, y=velocity))
    beardRibbon = geom_ribbon(data=beardRelationship, 
                              aes(x=diam, y=velocity, ymin=lower, ymax=upper), 
                              col="black", fill="black", alpha=0.3, lwd=0.2)
  }
  
  tiles = geom_tile(height=yheights, width=xwidths, aes(fill=z))
  if(drawClasses) {
    tiles = geom_tile(height=yheights, width=xwidths, aes(fill=z), 
                      lwd=0.2, col="grey")
  }
  
  titleString=paste(instrument, "\n", timestring, sep="")
  if(!title) {
    titleString = ""
  }
  
  labels = labs(title=titleString,
                x="Diameter [mm]",
                y="Velocity [m/s]")
  if(!showAxisLabels) {
    labels = labs(title=titleString, x="", y="")    
  }
  
  plot = 
    ggplot(data=toPlot, aes(x=diam, y=vel)) + 
    tiles + 
    theme_bw(textSize) +
    fill +
    labels +
    scale_y_continuous(limits=ylims) +
    scale_x_continuous(limits=xlims) 
        
  if(plotBeardLine) {
    plot = plot + beardLine
  }
  if(plotToleranceRange) {
    plot = plot + beardRibbon
  }
  
  if(length(overlayArray) > 0) {
    toPlotOverlay = toPlot
    toPlotOverlay$z = overlayArray
    toPlotOverlay$alpha = overlayArray
    #toPlotOverlay$z[which(toPlotOverlay$z == 0)] = NA
    overlayTiles = geom_tile(data=toPlotOverlay,
                             height=yheights, width=xwidths, 
                             aes(alpha=factor(alpha)), fill="black")
    plot = plot + overlayTiles
    if(overlayLegend) {
      plot = plot + scale_alpha_discrete(range=c(0, 0.3), name=overlayName, 
                           labels=overlayLabels)
    } else {
      plot = plot + scale_alpha_discrete(range=c(0, 0.3), name=overlayName, 
                                         labels=overlayLabels, guide=FALSE)
    }
                                      
  }
  
  # Position the legend.
  plot = plot + theme(legend.position=legendPos)
      
  # Change the margin if required.
  if(length(margins) > 0) {
    plot = plot + theme(plot.margin=unit(margins, "lines"))
  }

  return(plot)
}

#############################################################################
############## Process a raw Parsivel file and output filtered ##############
#############################################################################

processAllRawCSVFiles = function(indir, outdir, timeRes, 
                                 pattern="*.gz", 
                                 events=numeric(0),
                                 dateFormat="%Y%m%d",
                                 overwrite=FALSE,
                                 ...) {
  # Take raw Parsivel data (length 1024 per timestep) and output filtered
  # data per diameter class only (length 32). Remember that for normalisation
  # it's necessary to specify the time resolution using timeRes.
  #
  # Args:
  #  indir: The directory containing raw files for input.
  #  outdir: The directory in which to write output filtered files.
  #  timeRes: Time resolution of the input data, in seconds.
  #  pattern: File matching pattern (default: "*.gz").
  #  events: If provided, only convert files within date ranges for events 
  #          listed with start, end.
  #  dateFormat: Date format in filenames (default %Y%m%d).
  #  overwrite: Overwrite output files? (Default: FALSE).
  #  ...: Optional arguments to processRawCSVFile().
  #
  # Returns: void.
  
  if(!file.exists(outdir, recursive=TRUE)) {
    dir.create(outdir, recursive=TRUE)
  }
  
  dates = NULL
  if(length(events) > 0) {
    for(e in seq(1, length(events$start))) {
      start = events$start[e]
      end = events$end[e]
      
      eventDates = strftime(seq(as.Date(start, tz="UTC"), 
                                as.Date(end, tz="UTC"), by=1), 
                            format="%Y%m%d")
      dates = rbind(dates, data.frame(date=eventDates,
                                      stringsAsFactors=F))
    }
  }  
  dates = unique(dates$date)
  
  files = list.files(indir, full.name=T, pattern=pattern)
  for(file in files) {
    if(length(dates) != 0) {
      found = FALSE
      
      for(date in dates) {
        if(grepl(date, basename(file)))
          found = TRUE
          next
      }
      
      if(!found) {
        next
      }
    }
    
    outfile = paste(outdir, basename(file), sep="/")
    exists = file.exists(outfile)
    if(substr(outfile, nchar(outfile)-2, nchar(outfile)) == ".gz") {
      outfile = substr(outfile, 1, nchar(outfile)-3)
    }
    print(file)
    if(!exists | overwrite == TRUE) {
      processRawCSVFile(file, outfile, timeRes=timeRes, ...)
    } 
  }    
}

processRawCSVFile = function(infile, outputFile, timeRes,
                             stations=stationsDefinition(),
                             diamClasses=get.classD(),
                             velocityClasses=get.classV(),
                             expectedLines=2880,
                             timeCol=1,
                             statusCol=2,
                             codeCol=3,
                             Rcol=4,
                             dropCols=seq(5,1028),
                             S=parsivelCollectionAreas(diamClasses),
                             removeNonPhysDrops=TRUE,
                             shiftVels=FALSE, ...) {
  # Take raw drop counts from a CSV file, and process them.
  #
  # Args:
  #   infile: Raw Parsivel output in CSV format; must contain 1024 columns
  #           of drop counts ordered first by diameter then by velocity 
  #           (ie the first 32 numbers are for the first velocity class).
  #   outputFile: The output file to write, also CSV.
  #   timeRes: Temporal resolution of the input data [s].
  #   stations: Stations definition.
  #   diamClasses: min and max diameters for each of the diameter classes.
  #   velocityClasses: min and max velocities for each velocity class.
  #   expectedLines: Number of lines to expect (default: 2880).
  #   timeCol: Column containing timestamps (default: 1).
  #   statusCol: Column containing Parsivel status (default: 2).
  #   codeCol: Column containing Parsivel precip code (default: 3).
  #   RCol: Column containing Parsivel-derived rain rate (default: 4).
  #   dropCols: Column containing drop counts (default: 5:1028).
  #   S: Effective collection area per drop diameter class.
  #   removeNonPhysDrops: Remove non-phys drops from raw data? (Default: TRUE).
  #   shiftVels: shift the velocities of drops (if filtered non-phys)? 
  #              (Default: FALSE).
  #   
  # Returns: void.
  
  # Open the input file.
  stationNum = as.numeric(substr(basename(infile), 1, 2))
  if(is.na(stationNum)) {
    print("WARNING: Station not found in filename; expecting one station def.")
    stopifnot(length(stations$name) == 1)
    stationNum = stations$number
  }
  stationNumber = which(stations$number == stationNum)
  if(length(stationNumber) == 0) {
    print(paste("No station definition for station number", stationNum, 
                "- not processing raw file."))
    return()
  }
  stopifnot(length(stationNumber) == 1)  
  
  data = read.table(infile, header=F, sep=",")
  if(length(data[,1]) != expectedLines) {
    print(paste("WARNING: filter expected", expectedLines, "lines, got", 
                length(data[,1])))
  }
  
  # Normalise drop counts by the collection area, per drop diameter class.
  for(i in seq(1, length(S))) {
    # Find the indexes of all drops from diameter class i.         
    idx = seq(dropCols[1]+i-1, by=32, length.out=32)
    data[,idx] = data[,idx] / S[i]
  }
  
  # Convert times to POSIXct and collect together unnormalised counts.
  res = data.frame(station_name=station_nameNum, 
                   POSIXtime=as.POSIXct(data[,timeCol], tz="UTC"),
                   parsivelStatus=data[,statusCol],
                   precipCode=data[,codeCol],
                   parsivelR=data[,Rcol],
                   data[,dropCols])
  names(res) = c("station", "POSIXtime", "parsivelStatus", "precipCode",
                 "parsivelR", paste("C", seq(1,1024), sep=""))
  
  # Velocity information.
  centreVels = rowMeans(velocityClasses)
  velMat = t(matrix(rep(centreVels, 32), nrow=32, ncol=32, byrow=T))
  
  # Diameter class information.
  diameterClassWidths = apply(diamClasses, 1, diff)
  diamWidthsMat = matrix(rep(diameterClassWidths, 32), nrow=32, ncol=32, byrow=T)
  
  # Filter each raw array.
  stationDetails = stationFilters(stations, ...)
  filteredMat = matrix(0, nrow=length(res$POSIXtime), ncol=32)
    
  mat = as.matrix(res[,6:1029])
  sums = rowSums(mat)
    
  # Rows that contain NAs are set to NA
  filteredMat[which(is.na(sums)),] = rep(NA, 32)
  
  for(row in which(sums != 0)) {
    rawDSD = as.numeric(res[row, 6:1029])

    if(!all(rawDSD == 0)) {
        if(removeNonPhysDrops) {
            filtered = filterRawParsivel(rawDSD,
                stationDetails=stationDetails,
                stationNum=stationNumber, 
                shiftVels=shiftVels)
      } else {
        filtered = rawDSD
      }      
      
      mat = matrix(filtered, ncol=32, nrow=32, byrow=T)
      mat = mat / (timeRes * velMat * diamWidthsMat)
      filteredMat[row,] = colSums(mat)
    }
  }
    
  result = data.frame(res$POSIXtime, res$parsivelStatus, 
                      res$precipCode, res$parsivelR, 
                      filteredMat, row.names=NULL)
  write.table(result, file=outputFile, col.names=F, row.names=F, sep=",")
  system(paste("gzip", outputFile))
}

relativeUncertainty = function(dsds, stations,
                               station1="Pradel 1",
                               station2="Pradel 2",
                               variables=c("R", "Nt", "Dm", "Zh"),
                               minRainRate=0.1,
                               resample=FALSE,
                               timeRes="1 min",
                               dsdCols=seq(6,37)) {
  # Find the relative sampling uncertainty using collocated Parsivel
  # gauges.
  #
  # Uses working from paper:
  # Jaffrain, J. and A. Berne, 2011: Experimental quantification of the 
  # sampling uncertainty associated with measurements from Parsivel
  # disdrometers. J. Hydrometeor., 12, doi:10.1175/2010JHM1244.1.
  # 
  # Args:
  #  dsds: Set of parsivel DSDs.
  #  stations: Station information (at least name, altitude, latitude).
  #  station1: Name of the first collocated station (default: "Pradel 1").
  #  station2: Name of the second collocated station (default: "Pradel 2").
  #  variables: Name of the variables to analyse (default: R, Nt, Dm).
  #  minRainRate: Minimum allowed rain rate - must be present at both 
  #               collocated gauges [mm/h] (default: 0.1 mm/h).
  #  resample: Resample to a different time resolution? (Default: FALSE).
  #  timeRes: Time resolution to resample to (default: 1 min).
  #  dsdCols: Columns in which to find the DSD (default: seq(6, 37)).
  #
  # Returns: the relative uncertainty calculated from the collocated gauges.
  #          [Value between 0 and 1], by variable, in a data.frame.
  
  # Subset to get the collocated data only. 
  station1Data = subset(dsds, station==station1)
  station2Data = subset(dsds, station==station2)
  
  tsSeconds = convertTimeStringsToUnit(timeRes, "secs")
  
  # Resample the data to the required temporal resolution.
  if(resample) {
    station1Data = resampleParsivelByDSDs(station1Data, stations, 
                                          timeRes, dsdCols)
    station2Data = resampleParsivelByDSDs(station2Data, stations, 
                                          timeRes, dsdCols)
  }
  
  # Filter out data that had too weak a rain rate.
  idx = which(station1Data$R < minRainRate | station2Data$R < minRainRate)
  if(length(idx) > 0) {
    station1Data = station1Data[-idx,]
    station2Data = station2Data[-idx,]
  }
  
  # Timesteps and lengths of the two resulting data.frames must match.
  stopifnot(all(station1Data$POSIXtime == station2Data$POSIXtime))
  stopifnot(dim(station1Data) == dim(station2Data))
  
  res = NULL
  for(var in variables) {
    station1VarData = station1Data
    station2VarData = station2Data
    
    # Truncate negative values to zero (for radar reflectivity).
    station1VarData[[var]][which(station1VarData[[var]] < 0)] = 0
    station2VarData[[var]][which(station2VarData[[var]] < 0)] = 0
    
    # Filter out lines in which the variable is NA or infinite.
    idx = which(is.na(station1Data[[var]]) | is.na(station2Data[[var]]) | 
                  is.infinite(station1Data[[var]]) | 
                  is.infinite(station2Data[[var]]))
    if(length(idx) > 0) {
      station1VarData = station1VarData[-idx,]
      station2VarData = station2VarData[-idx,]
    }
    
    # Let the variable of interest be m. Assume that the mean of m over the 
    # collocated instruments is representative of the true value.  Then the 
    # normalised difference between the mean at one of the instruments, 
    # compared to the mean over the collocated instruments, is
    # 
    # epsilon = (instrument1 - mean(collocated)) / mean(collocated)
    
    means = (station1VarData[[var]] + station2VarData[[var]]) / 2
    epsilon = (station1VarData[[var]] - means) /  means
    
    # We now have the normalised difference between instrument mean and 
    # collocated mean per timestep.
    
    # Consider the measurement at time t to be the sum of the instrumental 
    # error w_t and the real variable value M_t, such that m_t = M_t + w_t. 
    # Also consider w_t to be identical for both sensors because the sensors 
    # are identical. Because it is Gaussian white noise, w_t should be zero 
    # on average, so the error is characterised by its standard deviation.
    
    # Following Jaffrain and Berne 2011, we have that the relative sampling
    # uncertainty is
    #
    # \sigma^r_w = sigma_w / E(M) \approx \sigma_\epsilon_S_k \sqrt{nk / n - k}
    # 
    # where n is the number of collocated instruments and k is the number of 
    # instruments in a subset for which we are finding the relative error (in our 
    # case n = 2, k = 1). 
    
    sigmaEpsilon = sqrt(var(epsilon, na.rm=T))
    relativeUncertainty = sigmaEpsilon * sqrt(2) # inside sqrt is nk/(n-k), 
                                                 # which for n=2, k=1 is 2.
    res = rbind(res, data.frame(variable=var, 
                                relUncertainty=relativeUncertainty))
  }
  
  return(res)
}

fillRawMissingDays = function(rawDir, outDir, 
                              startDate = as.Date("2013-09-01"),
                              endDate = as.Date("2013-11-26"),
                              dateFun = function(x) { 
                                r = regexpr("[0-9]{8}", x, perl=TRUE)
                                d = regmatches(x, r)
                                return(as.Date(d, format="%Y%m%d")) },
                              timeRes=30, 
                              stationNums=c(10,11,12,13,20,
                                            30,31,32,33)) {
  # Detect which days are missing from a timeseries of Parsivel files,
  # and create files containing NAs to fill them in. 
  #
  # Args: 
  #  rawDir: The raw directory to inspect.
  #  outDir: Directory in which to write NA files.
  #  startDate, endDate: The start and end dates to fill between (as.Date()).
  #  dateFun: Function to take filenames and return dates they represent.
  #  timeRes: Time resolution [s] (Default: 30 s).
  #  stationNums: Station numbers to work on; filenamess are expected 
  #               to start with the station number.
  #
  # Returns: void.
  
  # Set up required dates.
  requiredDates = seq(startDate, endDate, by=1)
  rawNAline = rep(NA, 21)
  
  for(station in stationNums) { 
    # Read existing dates.
    files = list.files(rawDir, full.name=TRUE, 
                       pattern=paste("^", station, sep=""))
    fileDates = dateFun(files)
    
    # Find missing dates.
    missingDates = requiredDates[which(!(requiredDates %in% fileDates))]
   
    if(length(missingDates) > 0) {   
      for(date in missingDates) {
        date = as.Date(date, origin="1970-1-1")
        print(paste("Station ", station, ": writing missing date ", 
                    strftime(date, "%Y%m%d"), sep=""))
        timeSeq = seq(as.POSIXlt(date, tz="UTC"), length.out=(86400 / timeRes),
                      by=timeRes)
        frame = 
          data.frame(strftime(timeSeq, format="%Y-%m-%d %H:%M:%S", tz="UTC"), 
                           t(rawNAline))
        outFile = paste(outDir, "/", station, "_ascii_", 
                        strftime(date, "%Y%m%d"), ".dat", sep="")
        write.table(frame, outFile, sep=",", row.names=F, col.names=F, quote=T)
      }    
    } 
  }
}

readParsivelREstimates = function(rawDir, pattern) {
  # Read Parsivel estimated statistics from raw files.
  # 
  # Args: 
  #  rawDir: The directory to read from.
  #  pattern: The pattern of files to read.
  #
  # Returns: data.frame containing R estimates from raw Parsivel files.
  
  result = NULL
  files = list.files(rawDir, full.names=TRUE, pattern=pattern)
  for(file in files) {
    print(file)
    data = read.csv(file, header=F, as.is=T)
    
    times = as.POSIXct(data[,1], tz="UTC")
    parsivelR = data[,4]
    
    res = data.frame(POSIXtime=times, parsivelR=as.numeric(parsivelR))
    result = rbind(result, res)
  }
  
  return(result)
}
