# Grenoble_Parsivel_functions.R
#
# Functions to deal with reading in Grenoble's Parsivel v2 instruments.
#
# Author: Tim Raupach <tim.raupach@epfl.ch>

source("library/filter_Parsivel_functions.R")
source("library/Parsivel_functions.R")
require(ncdf4)
require(arrayhelpers)
require(data.table)

convertNCToRaw = function(ncDir, ncPattern, codeFile, outDir, station,
                          codeNames = c("time", "record", "parsivelR",
                                        "accumAmount", "code4680",
                                        "code4677", "Z", "vis", "amplitude",
                                        "numParticles", "temp", "heating",
                                        "voltage", "status", "kinetic")) {
  files = list.files(ncDir, pattern=ncPattern, full.names=TRUE)
  codeInfo = read.table(codeFile, sep=",", skip=5, stringsAsFactors=FALSE)
  names(codeInfo) = codeNames

  codeInfo = data.table(codeInfo)
  codeInfo[, time := as.POSIXct(time, tz="UTC")]

  setkey(codeInfo)
  codeInfo = unique(codeInfo)

  codeInfo$status = as.numeric(codeInfo$status)
  codeInfo$parsivelR = as.numeric(codeInfo$parsivelR)
  codeInfo$code4680 = parsivelPrecipCode(as.numeric(codeInfo$code4680),
                                codeInfo$parsivelR,
                                nonZeroDrops=(codeInfo$numParticles > 0))
  stopifnot(range(codeInfo$status, na.rm=T) == c(0, 3))

  printRange = function(d, n) {
    print(paste("Range of", n, "is", paste(range(d[[n]], na.rm=T), collapse=", ")))
  }

  printRange(codeInfo, "status")
  printRange(codeInfo, "code4680")
  printRange(codeInfo, "parsivelR")

  setkey(codeInfo, "time")

  dropCols = paste("C", seq(1,1024), sep="")

  for(file in files) {
    print(file)
    nc = nc_open(file)

    # Get times and convert from beginning of measurement interval to
    # end of interval. The first time appears to be broken, it should
    # be midnight on the day in question.
    times = ncvar_get(nc, "Time")
    times = as.POSIXct(times, origin="1970-1-1", tz="UTC")
    times[1] = as.POSIXlt(as.Date(times[2]), tz="UTC")
    stop("Check time converstion from date. POSIXlt should work.")

    # Get raw drop counts. Dimensions are velocity, diameter, time.
    rawDrops = ncvar_get(nc, "RawDropsPerClass")

    # Rows are diameter, columns are velocity.
    # Raw array should have first 32 elements belonging to first velocity class.

    # Convert to a matrix of timestep by raw array
    f = function(x) {return(as.vector(x))}
    rawCounts = data.frame(t(apply(rawDrops, 3, f)))
    names(rawCounts) = dropCols

    # Columns of result should be:
    # time, rawCounts
    res = data.frame(time=times, rawCounts)
    res = data.table(res)
    setkey(res, "time")
    comb = codeInfo[res]
    stopifnot(dim(comb)[1] == 1440)

    # Output columns should be:
    # 1: time
    # 2: Parsivel status
    # 3: Parsivel precip code
    # 4: Parsivel R
    # 5..1028: raw drop counts
    cols = c("time", "status", "code4680", "parsivelR", dropCols)
    out = comb[, cols, with=FALSE]

    # Tested using
    # max(abs(rowSums(rawCounts) - as.numeric(as.character(comb[, numParticles]))), na.rm=T)
    # Maximum difference was two drops. Maybe filtered by Parsivel? For the
    # most part they match.

    # write output file here.
    outFile = paste(outDir, "/", station, "_ascii_",
                    strftime(times[1], format="%Y%m%d"), ".dat", sep="")
    write.table(out, file=outFile, sep=",", col.names=FALSE, row.names=FALSE)
    system(paste("gzip", outFile))

    nc_close(nc)
  }
}


compareGrenobleToLTEParsivels =
  function(grenParsivelFile="~/Switch/phd/Rdata/HYMEX_Grenoble_filtered.Rdata",
           lteParsivelFile="~/Switch/phd/Rdata/parsivel_liquid_only_data_filtered_SOP1.Rdata",
           outDir="~/Desktop/grenoble_comparison_5min/") {
  grenName = load(grenParsivelFile)
  gren = get(grenName)

  lteName = load(lteParsivelFile)
  lte = get(lteName)
  rm(list=c(grenName, lteName))

  dsds = rbind(lte, gren)
  rm(list=c("lte", "gren"))

  events=SOPevents()
  comparableEvents=seq(1, length(events$start))

  events$start[5] = events$start[5] - 100*60
  events$start[8] = events$start[8] - 120*60
  events$start[10] = events$start[10] - 160*60

  bandStations = c("Mirabel", "Lussas", "Lavilledieu", "Les Blaches",
                   "St-Germain", "Pradel 1", "Pradel 2")
  bandName = "HPicoNet"

  stats = c("R", "Nt", "D0")
  statNames = c("Rain rate", "Total drop concentration",
                "Median-volume drop diameter")
  statUnits = c("mm~h^{-1}", "m^{-3}", "mm")


  for(e in comparableEvents) {
    print(e)
    start = events$start[e]
    end = events$end[e]

    idx = which(dsds$POSIXtime >= start & dsds$POSIXtime <= end)
    d = resampleParsivelDSDs(data=dsds[idx,(1:35)], timespan="5 min")
    stop("addRainStats needs time res and station information.")
    d = addRainStats(d)

    for(s in seq(1, length(stats))) {
      plot = plotStatPerStation(stats=d, stat=stats[s], statName=statNames[s],
                                statUnits=statUnits[s], start=start, end=end,
                                bandStations=bandStations, bandName=bandName,
                                bandColour="blue", bandAlpha=0.35)
      outName = paste(outDir, "/", stats[s], "/", stats[s],
                      "_genoble_comparison_event_", e, ".png", sep="")
      ggsave(plot=plot, file=outName, width=10, height=4)
    }
  }
}

convertAllGrenToRawByDay = function(indir, outdir,
                               stations=grenobleParsivelStations(),
                               stationNums=c(5,8,33,34,51,84), ...) {
  # Use ... to specify arguments to convertGrenToRawCSVS().

  for(s in stationNums) {
    print(paste("Station", s))
    statsFile = list.files(indir, pattern=sprintf("parsivel%0.2d_", s), full.name=T)
    rawFile = list.files(indir, pattern=sprintf("spectre%0.2d_", s), full.name=T)
    stopifnot(length(rawFile) == 1 & length(statsFile) == 1)

    # Convert files to raw CSV in a temp file.
    tmpFile=paste(outdir, "temp.csv", sep="/")
    convertGrenToRawCSVS(rawFile=rawFile, statsFile=statsFile, outFile=tmpFile, ...)

    splitFileIntoDays(file=tmpFile, outdir=outdir, prefix=sprintf("%.2d_ascii", s))
    system(paste("rm", tmpFile))
  }
}

convertGrenToRawCSVS = function(rawFile, statsFile, outFile,
                                skipNum=4, header=FALSE,
                                dsdCols=seq(65,1088),
                                timeCol=1,
                                intensityCol=3,
                                parsPrecipCodeCol=5,
                                parsStatCol=14) {
  # Convert a raw Grenoble CSV file to a raw CSV file; read
  # spectra and Parsivel2 codes from separate files.
  #
  # Args:
  #   rawFile: The file containing the Parsivel2 spectra.
  #   statsFile: The file containing Parsivel2 statistics (R, code, etc).
  #   outFile: Output file to write.
  #   skipNum: Number of rows to skip in reading input files (default: 4).
  #   header: Header in inputs (default: FALSE).
  #   dsdCols: Which columns contain spectra? (Default: seq(65, 1088)).
  #
  # Returns: void.

  # Get times and drop spectra.
  fileRawData = read.table(rawFile, skip=skipNum, header=header,
                           sep=",", as.is=T)
  times = as.POSIXct(fileRawData[,timeCol], tz="UTC")
  dropCounts = fileRawData[,dsdCols]

  dropCounts = as.matrix(dropCounts)
  dropCounts[dropCounts == "NAN"] = 0
  dropCounts[dropCounts == 999] = NA

  # Convert to integer.
  dropCounts = t(apply(dropCounts, 1, as.integer))

  # Get Parsivel2-derived statistics.
  statsRaw = read.table(statsFile, skip=skipNum, header=header,
                        sep=",", as.is=T)
  statsTimes = as.POSIXct(statsRaw[,timeCol], tz="UTC")

  # Subset stats times to times for which we have spectra.
  # Subset spectra to times for which we have stats.
  sharedTimes = intersect(statsTimes, times)
  statsRaw = statsRaw[statsTimes %in% sharedTimes,]
  statsTimes = statsTimes[statsTimes %in% sharedTimes]

  dropCounts = dropCounts[times %in% sharedTimes,]
  times = times[times %in% sharedTimes]

  stopifnot(identical(statsTimes, times))

  # Get the Parsivel rain rates.
  parsivelR = statsRaw[,intensityCol]

  # Get Parsivel2 code and status.
  code = as.numeric(statsRaw[,parsPrecipCodeCol])
  parsivelStatus = as.numeric(statsRaw[,parsStatCol])

  # Translate code to LTE codes.
  containedDrops = rowSums(dropCounts) > 0
  precipCode = parsivelPrecipCode(code, parsivelRainRate=parsivelR,
                                  nonZeroDrops=containedDrops)

  # Return a data.frame of time, status, code, Pars-derived R and spectra.
  df = data.frame(POSIXtime=times, parsivelStatus=parsivelStatus,
                  precipCode=precipCode, parsivelR=parsivelR, dropCounts)
  write.table(df, file=outFile, row.names=F, col.names=F, sep=",")
}

splitFileIntoDays = function(file, outdir, prefix) {
  data = read.csv(file, header=F)

  times = as.POSIXct(data[,1], tz="UTC")

  dates = seq(as.Date(min(times), tz="UTC"),
              as.Date(max(times), tz="UTC"), by=1)
  days = as.Date(data[,1], tz="UTC")

  for(i in seq(1, length(dates))) {
    date = dates[i]
    print(date)

    idx = which(days == date)
    subset = data[idx,]

    outfile = paste(outdir, prefix,
                    "_", strftime(date, format="%Y%m%d.dat"), sep="")
    write.table(subset, file=outfile, col.names=F, row.names=F, sep=",",
                append=TRUE)
  }
}

padWithNAs = function(file, timeRes=60) {
  bname = basename(file)
  l = nchar(bname)
  date = as.Date(substr(bname, l-11, l-4), format="%Y%m%d")
  start = as.POSIXlt(date, tz="UTC")
  end = start + 60*60*24 - timeRes

  times = seq(start, end, by=60, tz="UTC")
  n = length(times)

  data = read.table(file, header=F, sep=",")
  resMat = matrix(NA, nrow=n, ncol=length(data[1,])-1)

  for(row in seq(1, length(data[,1]))) {
    time = as.POSIXct(data[row,1], tz="UTC")
    # print(time)

    r = which(times == time)
    stopifnot(length(r) == 1)
    resMat[r,] = as.numeric(data[row,(2:length(data[1,]))])
  }

  result = data.frame(POSIXtime=as.POSIXlt(times, tz="UTC"), resMat)
  write.table(result, file=file, col.names=F, row.names=F,
              sep=",", append=FALSE)
}

convertGrenobleParsivelToCSV = function(rawFile,
                                        parsivelFile,
                                        outFile,
                                        skipNum=4,
                                        header=FALSE,
                                        cols=seq(65,1088)) {
  # Convert Parsivel data recorded by Grenoble Parsivels into a raw
  # 1024 length CSV file.
  #
  # Args:
  #   rawFile: The file containing the raw data.
  #   parsivelFile: The file containing the precipCodes.
  #   skipNum: The number of lines to skip at the top of the file.
  #   header: Expect a header in the file? (Default: FALSE).
  #   rawCols: Column numbers for the 1024 class numbers.
  #   timestepCol: The column number of the timesteps.
  #   precipCodeCol: The column number of the precip code.
  #   outFile: Output file name.
  #
  # Returns: void.

  fileRawData = read.table(rawFile, skip=skipNum, header=header,
                           sep=",", as.is=T)
  times = as.POSIXct(fileRawData[,1], tz="UTC")
  dropCounts = fileRawData[,cols]
  dropCounts = as.vector(t(dropCounts))
  dropCounts[which(dropCounts == "NAN")] = 0
  dropCounts = as.numeric(dropCounts)
  dropCounts = matrix(dropCounts, ncol=1024, byrow=T)

  codes = read.table(parsivelFile, skip=skipNum, header=header, sep=",",
                     as.is=T)
  idx = which(as.POSIXct(codes[,1], tz="UTC") %in% times)
  codes = codes[idx,]
  idx = which(times %in% as.POSIXct(codes[,1], tz="UTC"))
  dropCounts = dropCounts[idx,]
  times = times[idx]
  stopifnot(as.POSIXct(codes[,1], tz="UTC") == times)
  stopifnot(length(times) == length(codes[,1]))
  stopifnot(length(times) == length(dropCounts[,1]))

  code = codes[,7]
  rainRate = codes[,5]

  nonZero = function(x) {
    return(any(!is.na(x)))
  }

  nonZeroDrops = apply(dropCounts, 1, nonZero)

  lteCodes = parsivelPrecipCode(code, rainRate, nonZeroDrops)
  stopifnot(length(lteCodes) == length(dropCounts[,1]))

  df = data.frame(POSIXtime=times, precipCode=lteCodes, dropCounts)

  write.table(df, file=outFile, row.names=F, col.names=F, sep=",")
}

grenobleParsivelStations = function() {
  # Define LHTE's Parsivel stations for HYMEX campaign in Ardeche.
  # These were stations 5, 8, 33, 34, 51, and 84.
  #
  # Note: stations were Parsivel2 with the exception of Mont-Redon 
  # and Pradel-Vignes, which were Parsivel1 disdrometers.
  #
  # Returns:
  #  A data.frame containing station number, name, label (for plotting),
  #  latitude (lat), longitude (lon), altitude [m], x_metres, y_metres,
  #  projString.

  stations = NULL
  stations = rbind(stations, data.frame(number=5,
                                        name="Villeneuve-de-Berg",
                                        label="Villeneuve-de-Berg",
                                        lat=44.5548, lon=4.4953,
                                        altitude=301))
  stations = rbind(stations, data.frame(number=8,
                                        name="Mont-Redon",
                                        label="Mont-Redon",
                                        lat=44.61409868, lon=4.51480108,
                                        altitude=636))
  stations = rbind(stations, data.frame(number=33,
                                        name="Pradel-Vignes",
                                        label="Pradel-Vignes",
                                        lat=44.58012902, lon=4.49504229,
                                        altitude=256))
  stations = rbind(stations, data.frame(number=34, # Need to check lat/long.
                                        name="Pradel Grainage v2",
                                        label="Pradel Grainage v2",
                                        lat=44.57900, lon=4.501100,
                                        altitude=271))
  stations = rbind(stations, data.frame(number=51,
                                        name="Villeneuve-de-Berg 2",
                                        label="Villeneuve-de-Berg 2",
                                        lat=44.5547, lon=4.4954,
                                        altitude=301))
  stations = rbind(stations, data.frame(number=84,
                                        name="Saint-Etienne-de-Fontbellon",
                                        label="Saint-Etienne-de-Fontbellon",
                                        lat=44.60000158, lon=4.38258853,
                                        altitude=210))
  stations = rbind(stations, data.frame(number=52,
                                        name="Villeneuve-de-Berg 3",
                                        label="Villeneuve-de-Berg 3",
                                        lat=44.5548, lon=4.4955,
                                        altitude=301))
  stations = as.data.frame(stations, as.is=T)
  stations$intTime = 60
  names(stations) = c("number", "name", "label", "lat", "lon", "altitude", "intTime")

  # Project to metres.
  projString = paste("+proj=lcc +lat_1=44.10000000000001",
                     "+lat_0=44.10000000000001 +lon_0=0",
                     "+k_0=0.999877499 +x_0=600000 +y_0=3200000",
                     "+a=6378249.2 +b=6356515",
                     "+towgs84=-168,-60,320,0,0,0,0 +pm=paris",
                     "+units=m +no_defs")
  projCRS = CRS(projString)
  # See http://spatialreference.org/ref/epsg/27573/ for projection details.

  coords = data.frame(lat=stations$lat, lon=stations$lon)
  coordinates(coords) = ~lon+lat
  proj4string(coords) = CRS("+proj=longlat +datum=WGS84")

  metre_coords = spTransform(coords, projCRS)

  stations$x_metres = coordinates(metre_coords)[,1]
  stations$y_metres = coordinates(metre_coords)[,2]
  stations$metres_proj4 = projString
  stations$altitude = as.integer(stations$altitude)
  return(stations)
}

readVilleneuveParsivel2 = function(dir="/ltedata/HYMEX/Grenoble/Grenoble_Parsivel2_Villeneuve_SOP1/",
                                   stations=villeneuveParsivels(),
                                   pattern="*.dat.gz", skipNum=3, header=T,
                                   cols=seq(65,1088)) {
  # Read Villeneuve Parsivel files into a data.frame.
  #
  # Args:
  #  dir: Where to look for the Parsivel files?
  #  pattern: File pattern to match.
  #  stations: Stations definition.
  #  skipNum: Number of lines to skip before reading the header (default: 3).
  #  header: Read the header? (Default: TRUE).
  #
  # Returns: data.frame containing all raw data from files, with
  #          columns for station, POSIXtime, precipCode, and the 1024 drop
  #          counts.

  files = list.files(dir, pattern=pattern, full.name=TRUE)

  results = NULL
  for(file in files) {
    print(paste("Reading", file))

    stationNum = as.numeric(substr(basename(file), 8, 9))
    stationName = as.character(stations$name[which(stations$number ==
                                                     stationNum)])

    fileRawData = read.table(file, skip=skipNum, header=header, sep=",", as.is=T)
    times = as.POSIXct(fileRawData[,1], tz="UTC")
    dropCounts = fileRawData[,cols]

    df = data.frame(station=stationName, POSIXtime=times, precipCode=NA, dropCounts)

    names(df) = c("station", "POSIXtime", "precipCode", paste("C", seq(1,1024)))
    results = rbind(results, df)
  }

  return(results)
}

