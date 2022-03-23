# 2DVD-functions.R
#
# Functions to process and data from the 2D Video Disdrometer.
#
# Tim Raupach <tim.raupach@epfl.ch>

source("library/DSD_functions.R")
source("library/timeseries_functions.R")
source("library/filter_Parsivel_functions.R")
require(reshape)
require(data.table)

adjust2DVDDropTimes = function(DVDdropData, adjustTimes) {
  # Adjust the times in a collection of drops by shifting all drops
  # within a certain period by a certain amount in time.
  # 
  # Warning: Only the times for drops within each range will be adjusted! 
  # The adjustment could result in "overlapping" times such that drop 
  # times are no longer in chronological order.
  # 
  # Args:
  #  DVDdropData: Drop data to adjust, with POSIXtime.
  #  adjustTimes: data.frame containing start and end [POSIXct UTC] and 
  #               amount [s] for each adjustment (+ for forward in time) 
  #               to make.
  # 
  # Returns: The adjusted drops.
  
  stopifnot(names(adjustTimes) == c("start", "end", "amount"))
  
  for(i in seq(1, length(adjustTimes$amount))) {
    start = adjustTimes$start[i]
    end = adjustTimes$end[i]
    amount = adjustTimes$amount[i]  
    
    idx = which(DVDdropData$POSIXtime >= start & 
                DVDdropData$POSIXtime <= end)
    adjustedDates = DVDdropData$POSIXtime[idx] + amount
    DVDdropData$POSIXtime[idx] = adjustedDates
  }
  
  return(DVDdropData)
}

reprocess2DVD = function(binaryDir, asciiDir, RdataOutputPrefix, 
                         startTime, converter, altitude, latitude, 
                         seaLevelTemp, convertBinary=TRUE,
                         resampleTimeRes=c("30 sec", "5 min", "30 min"),
                         DVDExtension=".hyd", adjustTimes=numeric(0), ...) {
  # Reprocess 2DVD data from binary files. 
  #
  # Args:
  #   binaryDir: The directory in which to find binary 2DVD files.
  #   asciiDir: The directory to read 2DVD ASCII drop data files from.
  #             Each line should be for an individual drop. 
  #   RdataOutputPrefix: Output Rdata file prefix, will have time res added 
  #                      to name.
  #   startTime: The start time for the resampled sequence (POSIXct, UTC).
  #   convertBinary: Convert binary data? (Default: TRUE).
  #   overwriteAscii: Overwrite ASCII in binary conversion? (Default: TRUE).
  #   resampleTimeRes: Time resolutions to resample to in seconds 
  #                    (default: 30, 300).
  #   DVDExtension: File extension for binary files (default: .hyd).
  #
  # Returns: void.
    
  if(convertBinary) {
    convertAll2DVDtoASCII(inputDir=binaryDir, 
                          DVDExtension=DVDExtension,
                          asciiOutputDir=asciiDir, 
                          converter=converter, 
                          overwrite=overwriteAscii) 
  }
  
  #print("WARNING: USING Rdata FILE INSTEAD OF RAW DATA, line 75 2DVD lib.")
  #asciiDir = paste(RdataOutputPrefix, "_raw_drop_data.Rdata", sep="")
  
  process2DVDSet(asciiDir, RdataOutputPrefix, startTime, altitude, 
                 latitude, resampleTimeRes, adjustTimes=adjustTimes, 
                 seaLevelTemp=seaLevelTemp, ...)
}

process2DVDSet = function(asciiDir, outputPrefix, startTime, altitude, latitude,
                          seaLevelTemp, resampleTimeRes=c("30 sec", "5 min", "30 min"),
                          adjustTimes=numeric(0), ...) {
  # Process a whole set of 2DVD data, get rain statistics using drop 
  # information at various time resolutions, bin to Parsivel drop classes,
  # and save as an Rdata file.
  # 
  # Args:
  #   asciiDir: The directory to read 2DVD ASCII drop data files from.
  #             Each line should be for an individual drop. OR, specify
  #             an RData file containing drop information to read.
  #   outputPrefix: Output Rdata file prefix, will have time res added to name.
  #   altitude, latitude: Instrument altitude [m] and latitude [deg north].
  #   startTime: The start time for the resampled sequence (POSIXct, UTC).
  #   resampleTimeRes: Time resolutions to resample to in seconds 
  #                    (default: 30, 300).
  #   adjustTimes: Obtionally adjust drop times using adjust2DVDDropTimes().
  #                (Default: no adjustment to drop times).
  #   ...: Optional arguments to nonPhysicalFilterDVDDrops.
  # 
  # Returns: void.
  
  # Read in the drop information.
  if(grepl("\\.Rdata$", asciiDir)) {
    name = load(asciiDir)
    DVDdropData = get(name)
    if(name != "DVDdropData") {
      rm(list=name)
    }
    print("All data read in from Rdata file.")
  } else {
    DVDdropData = read2DVDDropData(asciiDir=asciiDir)
    print("All data read in.")
    
    # Adjust drop times if required.
    if(length(adjustTimes) > 0) {
      print("Adjusting drop times...")
      DVDdropData = adjust2DVDDropTimes(DVDdropData, adjustTimes)    
    }
    
    # Save raw drop data.
    save(DVDdropData, file=paste(outputPrefix, "_raw_drop_data.Rdata", sep=""))
  }
    
  # Filter for non-physical drops.
  print("Filtering drops.")
  DVDFilteredDrops = nonPhysicalFilterDVDDrops(drops=DVDdropData, 
                                               altitude=altitude, 
                                               latitude=latitude, 
                                               seaLevelTemp=seaLevelTemp,
                                               ...)
  save(DVDFilteredDrops, file=paste(outputPrefix, "_filtered_drop_data.Rdata", 
                                    sep=""))
  
  # Remove the original drop data to save memory.
  rm(list="DVDdropData")
  
  # Get the last previous round hour to start resampling at.
  hour = trunc.POSIXt(startTime, "day")
  
  dvdStation = data.frame(name="2DVD", altitude=altitude, lat=latitude)
  
  # Resample filtered drops at each requested time resolution and get rain
  # statistics.
  for(res in resampleTimeRes) {
    print(paste("Finding statistics at", res, "resolution."))
    resSeconds = convertTimeStringsToUnit(res, "secs")
    DVDstats = dropwiseDVDRainStats(drops=DVDFilteredDrops$drops, 
                                    timeRes=res, start=hour, 
                                    stations=dvdStation, seaLevelTemp=seaLevelTemp, 
                                    ...)
    save(DVDstats, file=paste(outputPrefix, "_stats_", resSeconds, 
                              "sec.Rdata", sep=""))
    rm(list="DVDstats")
  }
  
  # Resample to Parsivel classes at each time resolution.
  for(res in resampleTimeRes) {
    resSeconds = convertTimeStringsToUnit(res, "secs")
    
    DVDperTimestep = resample2DVDdrops(DVDFilteredDrops$drops, start=startTime,
        timeRes=resSeconds, seaLevelTemp=seaLevelTemp)
    DVDperTimestep$station = "2DVD"
    DVDperTimestep = addRainStats(DVDperTimestep, timestepSeconds=resSeconds, 
                                  radar=FALSE, stations=dvdStation, 
                                  seaLevelTemperature=seaLevelTemp)#, ...)
    save(DVDperTimestep, file=paste(outputPrefix, "_parsivel_classes_", 
                                    resSeconds, "sec.Rdata", sep=""))
    rm(list="DVDperTimestep")
  }
}

convertAll2DVDtoASCII = function(inputDir, asciiOutputDir, 
    converter="~/2DVD/linux-code/hyd2asc/hyd2asc",
    DVDExtension=".hyd",
    overwrite=FALSE,
    fileNames=NULL) {                                 
  # Generate compressed ASCII files for each 2DVD HYD file in a directory.
  # ASCII files contain information about each drop recorded by the 2DVD.
  # Output files will be named 2DVD-YYYY-MM-DD.txt.gz.
  #
  # Args:
  #   inputDir: The input directory containing 2DVD binary files.
  #   asciiOutputDir: The output directory.
  #   converter: Path to the converter C program.
  #   2DVDExtension: The extension of 2DVD files to read (.hyd or .sno).
  #   overwrite: Overwrite output files? (Default: FALSE).
  #   fileNames: Subset of (basenames) files to convert.
  #  
  # Returns: void.
  
  # Check DSD script exists.
  stopifnot(file.exists(converter))
  
  # Check output directory exists, create it if not.
  if(!file.exists(asciiOutputDir)) {
    dir.create(asciiOutputDir, recursive=TRUE)
  }
  
  # Get list of files to process.
  inputFiles = list.files(inputDir, pattern=DVDExtension, full.names=T)

  if(!is.null(fileNames)) {
      inputFiles = inputFiles[which(basename(inputFiles) %in% fileNames)]
  }
  
  for(file in inputFiles) {
    # Print progress.
    print(paste("Processing file", basename(file)))
    
    # Get the date the file corresponds to from the filename.
    # Files are stored with name in format: VYYDDD_1.hyd, where
    # YY is last two digits of year, and DDD is the day of the year.
    bname = basename(file)
    datedef = substr(bname, 2, 6)
    date = as.Date(datedef, format="%y%j")
    datestring = strftime(date, "%Y-%m-%d", tz="UTC")
    
    # Output file is to be 2DVD-date.txt
    outfile = paste(asciiOutputDir, "/2DVD-", datestring, ".txt", sep="")
    
    # Don't overwrite output files.
    if(file.exists(paste(outfile, ".gz", sep=""))) {
      next
    }
    
    # Run the conversion.
    options=""
    cmd = paste(converter, file, outfile, options)
    print(cmd)
    system(cmd, ignore.stdout=T, ignore.stderr=T)
    
    if(file.exists(outfile)) {
      # Compress the output file.
      system(paste("gzip -9 ", outfile), ignore.stdout=T)
    } else {
      print(paste("No drops recorded for", datestring))    
    }
  }
}

read2DVDDropData = function(asciiDir=Default2DVDAsciiDir,
                            pattern="2DVD-.*.txt.gz") {
  # Read all 2DVD drop data from ascii file records.
  #
  # Args:
  #   asciiDir: Directory in which to find files.
  #   pattern: File matching pattern.
  #
  # Returns: data.frame containing POSIXtime, diameter, volume, velocity, 
  # oblateness, area, Agte, Alte, Bgte, Blte, lineheight, type, heightA,
  # heightB, widthA, widthB, oblA, oblB, numLinesA, numLinesB, scaleA, 
  # scaleB.
  
  files = list.files(asciiDir, pattern=pattern, full.name=T)
  stopifnot(length(files) > 0)
  
  print("Reading in 2DVD ascii files.")
    
  alldata = NULL
  for(file in files) {
    print(file)
    data = data.table(read.table(file, header=T, stringsAsFactors=F))
    alldata = rbind(alldata, data)
  }
  
  POSIXtimes = data.frame(POSIXtime=as.POSIXct(alldata$timestamp, tz="UTC"))
  alldata$timestamp = NULL
  res = cbind(POSIXtimes, alldata)
  return(res)
}

nonPhysicalFilterDVDDrops = function(drops,                      
                                     altitude, latitude, seaLevelTemp,
                                     maxDiam = 7.5,
                                     tolFunc = function(v, d) {
                                       min = v-3
                                       max = v+4
                                       min[which(d < 2)] = 0
                                       return(cbind(min, max))
                                       },
                                     ...) {
  # Remove non-physical drops from a list of drops recorded by the 2DVD.
  #
  # Args:
  #   drops: The 2DVD drops.
  #   altitude: Altitude of the 2DVD.
  #   latitude: Latitude of the 2DVD. 
  #   maxDiam: The maximum allowed diameter.
  #   tolFunc: A function that takes a terminal velocity and a drop 
  #            diameter and returns the minimum and maximum allowed 
  #            velocities for drops of that diameter.
  #   ...: Extra arguments to terminalVels().
  #
  # Returns: A list containing "drops" - the drops with non-physical drops
  #          removed, and "percRemoved" - the percentage of drops that were
  #          removed.
    
  drops = data.table(drops)

  # Remove drops that are too large.
  drops = drops[diameter <= maxDiam]
  
  # For each drop, find its theoretical terminal velocity.
  drops[, beardVel := terminalVels(altitude=altitude, 
                                   lat=latitude, diams=diameter, 
                                   seaLevelTemperature=seaLevelTemp, ...)]
  
  # Function that determines if a drop is allowed or not based on the
  # tolerange range function.
  allowedFunc = function(realVel, beardVel, diameter) {
    range = tolFunc(beardVel, diameter)
    return(realVel >= range[,1] & realVel <= range[,2])
  }
  
  # Work out whether each drop is allowed or not.
  drops[, allowed := allowedFunc(velocity, beardVel, diameter)]
  
  # Find percentage of removed drops.
  percRemoved = drops[, length(which(allowed == FALSE))] / 
    drops[, length(allowed)] * 100
    
  # Remove dis-allowed drops.
  drops = drops[allowed == TRUE]
  drops[, allowed := NULL]
  
  return(list(drops=drops, percRemoved=percRemoved))
}

dropwiseDVDRainStats = function(drops, timeRes, seaLevelTemp,
                                start=trunc.POSIXt(min(drops$POSIXtime), "day"),
                                diameterResolution=0.2,
                                maxDiam=8, radar=FALSE, ...) {
  # Determine rain statistics (R, amount, Nt, Dm, LWC, D0) from 2DVD data 
  # at a given time resolution. 
  #
  # Args:
  #  drops: The 2DVD drop data, one line per drop. 
  #  timeRes: The time resolution to use, as a string (default: "30 sec").
  #  start: Start time (start of first integration period). (Default: last 
  #         round minute before first drop).
  #  diameterResolution: Drop diameter resolution of the 2DVD [mm].
  #  maxDiam: The maximum diameter of drops to consider [mm].
  #  radar: Calculate radar variables? (Default: FALSE).
  #  ...: Extra arguments to addRainStats().
  # 
  # Returns: data.frame containing POSIXtime, R, amount, Nt, Dm, LWC,
  #          plus Zh anad Zv if radar variables are requested.
  
  # Convert drops to a data.table.
  drops = data.table(drops, key="POSIXtime")
  drops = drops[POSIXtime >= start]
  drops = drops[diameter <= maxDiam]
  
  # Make a sequence of the times we need and assign each drop a timestep.
  # The timestep assigned will be the end of the integration time.
  timeResSec = convertTimeStringsToUnit(timeRes, "secs")
  timeSeq = seq(start, drops[,max(POSIXtime)]+(2*timeResSec), 
                by=timeResSec, tz="UTC")
  drops[, timestep := cut.POSIXt(POSIXtime + timeResSec, timeSeq)]
  stopifnot(drops[, !any(is.na(timestep))])
  
  # Convert collection area from mm^2 to m^2.
  drops[, aream2 := area / 1e6]
  
  # Find the rain rate by timestep.
  stats = drops[, list(R = (6*pi*1e-4)*sum(diameter^3/(aream2*timeResSec))), 
                by=timestep]
  
  # Add in amount which is based on R and convert timestep to POSIXtime.
  stats[, amount := R / (3600/timeResSec)]  
  stats[, timestep := as.POSIXct(timestep, tz="UTC")]
  
  # To get other stats such as Nt, Dm, and LWC, we need to bin the drops into
  # diameter classes. To use the smallest possible class size we'll use the
  # 2DVDs diameter resolution.
  dseq = seq(0, maxDiam+diameterResolution, by=diameterResolution)
  dropClasses = cbind(dseq[1:(length(dseq)-1)], dseq[2:length(dseq)])
  d = resample2DVDdrops(drops, start=start, 
                        end=drops[, max(POSIXtime)]+timeResSec,
                        timeRes=timeResSec, 
                        diamClasses=dropClasses, performNonPhysFilter=FALSE,
                        seaLevelTemp=seaLevelTemp)
  d = data.table(d)[POSIXtime %in% stats[, timestep]]
  
  d = addRainStats(d, timestepSeconds=timeResSec, classes=dropClasses, 
                   radar=radar, seaLevelTemperature=seaLevelTemp, ...)
  stopifnot(identical(d$POSIXtime, stats$timestep))
  
  stats[, Nt := d$Nt]
  stats[, Dm := d$Dm]
  stats[, LWC := d$LWC]
  
  if(radar) {
    stats[, Zh := d$Zh]
    stats[, Zv := d$Zv]
  }
  
  # Rename "timestep" to "POSIXtime".
  setnames(stats, "timestep", "POSIXtime")  
  
  # Add zeros for timesteps which are not covered. 
  missingTimes = which(!(timeSeq %in% stats$POSIXtime))
  zeros = data.table(POSIXtime = timeSeq[missingTimes], R=0, 
                     amount=0, Nt=0, Dm=0, LWC=0)
  if(radar) {
    zeros[, Zh := NA]
    zeros[, Zv := NA]
  }
  setkey(zeros, POSIXtime)
  setkey(stats, POSIXtime)
  stats = rbind(stats, zeros)
  stopifnot(all(stats[, duplicated(POSIXtime)] == FALSE))
  stopifnot(all(timeSeq %in% stats$POSIXtime))
  
  # Order by time.
  stats = stats[order(POSIXtime)]
  
  return(stats)
}

DVDNonPhysFilter = function(lat=44.579006, altitude=271, seaLevelTemp=15) {
  # Create a Parsivel-style non-physical drop filter for the 2DVD location.
  return(createNonPhysicalFilterAtStation(altitude, lat, seaLevelTemperature=seaLevelTemp))
}

resample2DVDdrops = function(DVDdropData, 
                             start=min(DVDdropData$POSIXtime), 
                             end=max(DVDdropData$POSIXtime),
                             timeRes=30,
                             diamClasses=get.classD(),
                             precipCode=NA,
                             station="2DVD",
                             velClasses=get.classV()[,1],
                             seaLevelTemp=15,
                             nonPhysFilter=DVDNonPhysFilter(seaLevelTemp=seaLevelTemp),
                             performNonPhysFilter=TRUE,
                             diamRight=TRUE) {
  # From data about each individual drop, count the drops per diameter class.
  # 
  # Args:
  #  DVDdropData: data.frame with a line per drop recorded by the 2DVD.
  #  start: Start time (POSIXct, UTC), start of integration time.
  #  end: End time (POSIXct, UTC), end of integration time.
  #  timeRes: Time resolution to output [s].
  #  diamClasses: The diameter classes to output for (min, max for each) [mm].
  #  velClasses: Velocity classes to use for non physical drop filter (mins for 
  #              each class).
  #  precipCode: Attach a certain precip code to each timestep (default: NA).
  #  performNonPhysFilter: Run the non-physical drop filter? (Default: TRUE).
  #  diamRight: should diameter cuts be closed on the right? (Default: TRUE).
  #
  # Returns: data.frame containing station, POSIXtime, precipCode, and a column
  #          of drop counts [m^-3] for each diameter class. Counts are 
  #          normalised by the 2DVD collection area, which varies per drop.
  #          Note that drops are put in classes if they are strictly larger
  #          than the lower bound of the class and <= the upper bound.
  
  # Convert to data.table.
  DVDdropData = data.table(DVDdropData, key="POSIXtime")

  ## Expect diamClasses to be a data.frame.
  diamClasses = data.frame(diamClasses)
  
  # Get subset of drops to work on.
  print("Retrieving subset...")
  drops = DVDdropData[POSIXtime >= start & POSIXtime <= end]
  
  # Convert drop collection areas from mm^2 to m^2.
  print("Normalising drop counts...")
  drops[, aream2 := area / 1e6]
  
  # Normalise each drop by its collection area, velocity, and the time 
  # resolution. Units will be [1 / (m^2 * s * m/s)] = [m^-3].
  drops[, normalisedCount := 1 / (aream2 * timeRes * velocity)]
  
  # Make a sequence of the times we need.
  print(paste("Putting drops in time seq at", timeRes, "s resolution..."))
  timeSeq = seq(start, end+(2*timeRes), by=timeRes, tz="UTC")
  
  # Place each drop into the sequence. (Note we use the default right=FALSE 
  # here, so that drops that arrive in the first second after the time step 
  # are put into the next class. right=FALSE is default for POSIXct but
  # not for ordinary cut()).
  drops[, timestepClass := cut.POSIXt(POSIXtime + timeRes, breaks=timeSeq, label=FALSE)]
    
  # Place each drop into a diameter class.
  diamClassMins = c(diamClasses[,1], max(diamClasses[,2]))
  print(paste("Putting drops in", length(diamClassMins), "diam classes..."))
  drops[, diamClass := cut(diameter, breaks=diamClassMins, labels=FALSE, right=diamRight)]
  
  # Perform the non-physical filter only if required.
  if(performNonPhysFilter) {
    drops[, velClass := cut(velocity, breaks=velClasses, labels=FALSE)]
    filterLookup = function(row) {
      return(nonPhysFilter[row[1], row[2]])
    }
    classes = as.matrix(data.frame(drops$velClass, drops$diamClass))
    drops[, filterResult := apply(classes, 1, filterLookup)]
    numRem = drops[filterResult == 0, length(filterResult)]
      
    if(numRem > 0) {
      print(paste("Removing", numRem / drops[, length(POSIXtime)] * 100,
                  "% drops due to non-physical filter."))
      drops = drops[filterResult != 0]
    }
  }
    
  # Make sure no drops were not classifiable somehow.
  stopifnot(drops[is.na(timestepClass), length(timestepClass)] == 0)
  stopifnot(drops[is.na(diamClass), length(diamClass)] == 0)
  
  # Remove the last timestep (after cut it is not required).
  timeSeq = timeSeq[1:(length(timeSeq)-1)]
  
  # Sum normalised counts for each timestep and diameter class.
  print(paste("Summing drop counts per timestep and diameter class..."))
  sums = data.frame(drops[, list(sum=sum(normalisedCount)), 
                          by="timestepClass,diamClass"])
  
  # Reshape the sums array.
  print("Reshaping output array...")
  sumsMolten = melt(sums, id.vars=c("timestepClass", "diamClass"))
  sums = cast(sumsMolten, timestepClass ~ diamClass, fill=0) 
    
  # Introduce a column of zeros for every class for which drops were not found.
  sumMatrix = matrix(0, ncol=nrow(diamClasses), 
                     nrow=length(sums$timestepClass))
  for(i in seq(1, length(diamClassMins))) {
    if (as.character(i) %in% names(sums)) {
      sumMatrix[, i] = sums[[as.character(i)]]
    } 
  }
  
  # Introduce a row of zeros for each time step for which drops weren't found.
  resultMat = matrix(0, ncol=nrow(diamClasses), nrow=length(timeSeq))
  for(i in seq(1, length(sums$timestepClass))) {
    resultMat[sums$timestepClass[i],] = sumMatrix[i,]
  }
  
  # Divide each diameter class entry by the width of the class, such that 
  # the units go from [m^-3] to [m^-3 mm^-1].
  diameterClassWidths = apply(diamClasses, 1, diff)
  divRow = function(x) { return(x / diameterClassWidths) }
  resultMat = t(apply(resultMat, 1, divRow))
  
  # Set up the result vector. We add timeRes to POSIXtimes so that 
  # the given time is the end of each integration step.
  result = data.frame(station=station, POSIXtime=timeSeq,
                      precipCode=precipCode, resultMat, stringsAsFactors=FALSE)
  
  names(result) = c("station", "POSIXtime", "precipCode",
                    paste("class", seq(1,length(diameterClassWidths)), sep=""))
  return(result)
}

dropsSampledPerTimestep = function(dsdDrops, from, to, timeRes="5 min") {
  # Return the sample size, ie the number of drops, that were recorded by
  # the 2DVD in each time step between start and end times.
  #
  # Args:
  #  dvdDrops: The 2DVD drop data.
  #  from, to: Start and end times (POSIXct, UTC).
  #  timeRes: The time resolution to work at (default: "5 min").
  # 
  # Returns: data.frame containing time step and number of drops recorded.
  
  # TODO: Make this a data.table function.
  dsdDrops = data.frame(dsdDrops)
  
  dsdDrops = dsdDrops[which(dsdDrops$POSIXtime >= from & 
                            dsdDrops$POSIXtime <= to), ]

  # Change dates into POSIXct date format.
  ts = dsdDrops$POSIXtime
  
  # Cut into periods. 
  timeBreaks = seq(from, to+(2*timeRes), by=timeRes)
  periods = cut(ts+timeRes, timeBreaks, tz="UTC")
  
  # Sum up drops for each period.
  d = data.frame(period=periods, drop=1)
  summedDrops = ddply(d, .(period), summarise, sampleSize=sum(drop))
  
  # Make each period time is the end of the integration period.
  summedDrops$period = as.POSIXct(summedDrops$period, tz="UTC")
  
  # Fill in missing timesteps with zeros.
  missingTimesteps = timeBreaks[which(!(timeBreaks %in% summedDrops$period))]
  zeros = data.frame(period=missingTimesteps, sampleSize=0)
  
  res = rbind(zeros, summedDrops)
  res = res[order(res$period),]  
  names(res) = c("POSIXtime", "sampleSize")
  
  stopifnot(all(duplicated(res$POSIXtime) == F))
  return(res)
}
