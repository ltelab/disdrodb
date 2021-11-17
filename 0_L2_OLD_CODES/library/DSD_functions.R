# DSD-functions.R
#
# Functions to read and manipulate the drop size distributions (DSDs) from
# disdrometer measurements.
#
# Author: Tim Raupach <tim.raupach@epfl.ch>

# Includes.
library(ggplot2)
library(plyr)
source("library/filter_Parsivel_functions.R")
source("library/DSD_radar_functions.R")
source("third-party/radarReflectivity.R")
source("library/axis_ratio_functions.R")

## Settings.
DefaultNumClasses = 32  # The default number of DSD classes to expect.

## Mean and median with na.rm = T.
meanNaRm = function(x) { return(mean(x, na.rm=T)) }
medianNaRm = function(x) { return(median(x, na.rm=T)) }

## Axis ratios (updated to Thurai 2007 on 20.02.2017).
raindrop_axis_ratio = axisRatioThurai2007

readAllDSDFiles = function(inDir, filePattern="*\\.txt", header=F,
                           numClasses = DefaultNumClasses, sep=",") {
  # Read all DSDs in a directory and create a large data.frame containing
  # all the spectra contained in all files in that directory.
  #
  # Args:
  #   inDir: Input directory.
  #   filePattern: Pattern to use to match files (default "*\\.txt").
  #   header: Expect header row in file? (Default: F).
  #   numClasses: number of classes to expect (Default: DefaultNumClasses).
  #   sep: Field separator in the input file (default: ",").
  #
  # Returns: A data.frame containing DSD spectra.

  # Find list of files to read.
  files = list.files(inDir, pattern=filePattern, full.names=T)

  # Sum up the selected DSD records.
  allDSDs = NULL
  DSDlist = list()

  for(file in files) {
    print(file)
    dsd = readDSDFile(file, numClasses, header, sep)
    if((any(is.character(dsd$parsivelR)))) {
      print("found it")
    }
    DSDlist = c(DSDlist, list(dsd))
  }

  # Stack all the data.frames together in a fast way.
  allDSDs = rbind.fill(DSDlist)
  return(allDSDs)
}

readDSDFile = function(file, numClasses=DefaultNumClasses, header=F, sep=",") {
  # Read a DSD file with a timestamp and drop size classes.
  #
  # Args:
  #   file: The file to read.
  #   numClasses: The number of drop size classes to expect (default:
  #               DefaultNumClasses).
  #   header: Expect header row in file? (Default: F).
  #   sep: Separator to expect in the file (default: ",").
  #
  # Returns: data.frame containing DSD spectra, col names POSIXtime,
  # parsivelStatus, precipCode, parsivelR, c1, c2, ..., cN.

  rows = read.table(file, header=header, sep=sep, stringsAsFactors=F)
  stopifnot(length(rows[1,]) == numClasses+4)

  names(rows) = c("POSIXtime", "parsivelStatus", "precipCode", "parsivelR",
                  paste("class", seq(1, numClasses), sep=""))

  #rows$POSIXtime = as.POSIXct(rows$timestamp, tz="UTC")
  times = as.POSIXct(rows$POSIXtime, tz="UTC")
  rows$POSIXtime = times

  return(rows)
}

addRainStats = function(spectra, timestepSeconds, classes=get.classD(),
    dsdColNames=paste("class", seq(1,length(classes[,1])), sep=""),
    seaLevelTemperature, radarFreq=9.4, radarIncidence=0, lapse=0.0065,
    altitude=342, latitude=44.58, radar=TRUE, stations=NULL,
    ratio_function=raindrop_axis_ratio, cantingSD=7, ...) {
    ## Calculate rain bulk variables for each DSD spectra in a data.frame.
    ##
    ##   spectra: data.frame containing DSD spectra (timestamp as POSIXtime, and
    ##            columns per drop class, with values in [m^-3 mm^-1]).
    ##            Columns for drop count classes must be named. A station
    ##            column must exist if stations are provided.
    ##   timestepSeconds: Number of seconds per timestep.
    ##   classes: n x 2 matrix of drop class diameter min/maxes, one per class
    ##            (default: Parsivel classes).
    ##   dsdColNames: The names of columns of the DSD in spectra (default: class1
    ##                to classN where there are N classes).
    ##   radarFreq: Frequency of the radar to use [Ghz] (Default: X-Band 9.4 Ghz).
    ##   seaLevelTemperature: Sea level temp to use for radar calculations [deg. C]
    ##                        (default: 15 deg C).
    ##   altitude: Default altitude to assume [m] (342 m corresponding to average HYMEX height).
    ##   latitude: Default latitude to use (default: 44.58, corresponding to HYMEX).
    ##   radarIndicence: Shooting angle of the radar above horizontal [degrees]
    ##                   (Default: 0 deg).
    ##   lapse: Atmospheric lapse rate for temperature [deg. m-1].
    ##   radar: Include radar stats? (Default: TRUE).
    ##   stations: Data.frame including at least name, altitude, latitude.
    ##             Or NULL to not use station information, in which case
    ##             altitude is assumed to be 0 m and latitude is assumed to be
    ##             45 degrees N.
    ##   cantingSD: The standard dev. of Gaussian distribution of canting
    ##              angles of raindrops, for radar calculations (default: 0).
    ##
    ## Returns: data.table with addition of the following:
    ##   R - the rain rate over the spectra time period. [mm/h].
    ##   amount - the total rainfall amount over the spectra time period [mm].
    ##   Nt - the total concentration of drops [m^-3].
    ##   D0 - the median volume drop diameter [mm].
    ##   Dm - The mass-weighted diameter [mm].
    ##
    ## And optionally:
    ##   Zh - Radar horizontal reflectivity [dBZ].
    ##   Zv - Radar vertical reflectivity [dBZ].

    spectra = data.table(spectra)
    if((is.null(stations) & "station" %in% names(spectra))|
       (!is.null(stations) & !"station" %in% names(spectra))) {
        print(paste("WARNING: Stations exist, but no station information",
                    "provided to addRainStats()."))
    }

    diameterClasses = rowMeans(classes)
    diameterClassWidths = apply(classes, 1, diff)
    numClasses = length(diameterClasses)

    print("DSD stats using columns with names:")
    print(paste(dsdColNames, collapse=", "))
    dsdMat = as.matrix(spectra[, dsdColNames, with=FALSE])
    stopifnot(names(dsdMat)[1] == "class1")

    ## Calculate moments of these DSD spectra.
    print("Moments...")
    momentCols = c("moment_3", "moment_4")
    spectra[, (momentCols) := calculateDSDMoments(.SD, classes, orders=c(3,4)),
            .SDcols=dsdColNames]

    ## If stations are given, assign altitudes, latitudes,
    ## and temperatures to each DSD depending on the station locations. If
    ## not, use defaults.
    if(is.null(stations) | !("station" %in% names(spectra))) {
        warning("No stations supplied, using default (HYMEX) altitude and latitude!!")

        spectra[, alt := altitude]
        spectra[, lat := latitude]
    } else {
        stations = data.table(stations)

        setkey(stations, name)
        setkey(spectra, station)

        spectra[, alt := stations[spectra, altitude]]
        spectra[, lat := stations[spectra, lat]]
    }

    ## Temperature [deg. C] and water density [g mm-3] rely on altitude and latitude.
    spectra[, temp := seaLevelTemperature - lapse*alt]
    spectra[, wDensity := waterDensity(altitude=alt, temperature=temp, latitude=lat)]

    ## Calculate rainrate R, liquid water content LWC, and radar variables,
    ## per groups of altitudes and latitudes.
    print("Rain rate...")
    spectra[, R := DSDRainrate(spectra=.SD[, dsdColNames, with=FALSE],
                    classes=classes, altitude=.SD[, unique(alt)], latitude=.SD[, unique(lat)],
                    seaLevelTemperature=seaLevelTemperature, ...),
            by="alt,lat"]

    print("LWC...")
    spectra[, LWC := liquidWaterContentFromMoments(moments=.SD[, momentCols, with=FALSE],
                      rho_w=.SD[, wDensity]), by="alt,lat"]

    ## Convert rate in mm/h to rain amount in mm.
    print("Rain amount...")
    spectra[, amount := R * timestepSeconds/3600]

    ## Calculate median volume drop diameter D_0 for the spectra. This is
    ## the drop diameter that divides the volume of water contained in the
    ## sample into two equal parts.
    print("Median volume drop diameter...")
    spectra[, D0 := medianVolumeDropDiameter(spectra=as.matrix(.SD), classes=classes),
            .SDcols=dsdColNames]

    ## Total drop concentration per timestep.
    print("Total drop concentration...")
    spectra[, Nt := totalDropConcentration(as.matrix(.SD), classes=classes),
            .SDcols=dsdColNames]

    ## Calculate mass-weighted diameter Dm. This is a more robust statistical
    ## estimator of the DSD and is the ratio of the 4th to the 3rd moments of the
    ## DSD.
    print("Mass-weighted diameter...")
    spectra[, Dm := moment_4/moment_3]

    radarCols = c("Zh", "Zv")
    spectra[, (radarCols) := NULL]
    spectra[, Zh := as.double(NA)]
    spectra[, Zv := as.double(NA)]
    if(radar) {
        ## Calculate the radar reflectivity for each DSD spectra.
        print("Radar reflectivity...")

        spectra[, (radarCols) := 10*log10(radarReflectivityFromDSD(DSD=.SD, classes=classes,
                                  freq=radarFreq, temp=temp, incidence=radarIncidence,
                                  ratio_function=ratio_function, cantingSD=cantingSD)),
                .SDcols=dsdColNames, by=temp]
    }

    return(spectra)
}

medianVolumeDropDiameter = function(spectra, classes=get.classD()){
  # Compute the median volume drop diameter (D0) for DSD spectra. This is
  # the drop diameter that divides the volume of water contained in the
  # sample into two equal parts. Original code by Marc Schleiss.
  #
  # Args:
  #   spectra: Matrix with one row per DSD to calculate, with one column
  #            per diameter class. Values are drop counts per class
  #            [m^-3 mm^-1].
  #   classes: Diameter classes (col for min, col for max) [mm].
  #
  # Returns: D0 for each DSD [mm].

  stopifnot(class(spectra) == "matrix")
  stopifnot(length(spectra[1,]) == length(classes[,1]))

  classWidths = apply(classes, 1, diff)
  classCentreDiams = rowMeans(classes)
  numClasses = length(classCentreDiams)

  n = length(spectra[,1])
  D0 = rep(0, n)

  # Conserve NA values, so that zero DSD means a median drop diameter of zero,
  # and the median drop diameter is NA if there were NAs in the DSD.
  D0[which(is.na(rowMeans(spectra)))] = NA

  sums = rowSums(spectra, na.rm=TRUE)
  nonZeroDSDs = which(sums > 0)

  for(row in nonZeroDSDs){
    # Convert DSD from [m^-3 mm^-1] to [m^-3].
    DSD = spectra[row,] * classWidths

    cs = cumsum(DSD*classCentreDiams^3)
    sum = cs[numClasses]

    lowerHalf = which(cs <= sum/2)
    upperHalf = which(cs >= sum/2)

    if(length(lowerHalf) * length(upperHalf) == 0) {
        ## In the (unusual) case that the first or last bins contain more than half
        ## of the water volume, we can not determine D0 because we would need
        ## the diameter of the class before/after the given classes. Return NA
        ## in these cases.
        D0[row] = NA
        next
    }

    lower = max(lowerHalf)
    upper = min(upperHalf)
    if(lower == upper) {
      D0[row] = classCentreDiams[lower]
      next
    }

    s1 = cs[upper]
    s2 = cs[lower]
    D0[row] = classCentreDiams[lower] +
      (classCentreDiams[upper] - classCentreDiams[lower]) *
      (sum/2-s1)/(s2-s1)
  }

  return(D0)
}

DSDRainrate = function(spectra, classes=get.classD(),
                       altitude=0, latitude=45, ...) {
  # Compute the rain rate (R) derived from the DSD. Terminal velocities
  # for each diameter class are calculated using the algorithm of
  # Beard 1976; Pruppacher & Klett 1978.
  #
  # Args:
  #   spectra: Matrix with one row per DSD to calculate,
  #            with one column per diameter class. Values are
  #            drop counts per class [m^-3 mm^-1].
  #   classes: Diameter classes (col for min, col for max) [mm].
  #   altitude: Altitude to use for velocity model (default: 0 m, sea level) [m].
  #   latitude: Latitude to use for velocity model (default: 45 deg N) [deg N].
  #   ...: Optional extra arguments to terminalVelocitiessByClass().
  #
  # Returns: R for each DSD [mm h^-1].

  stopifnot(length(altitude) == 1)
  stopifnot(length(latitude) == 1)
    
  if(is.null(dim(spectra))) {
    l = length(spectra)
  } else {
    l = dim(spectra)[2]
  }
  stopifnot(l == length(classes[,1]))

  # Get information per class.
  classWidths = apply(classes, 1, diff)
  classCentreDiams = rowMeans(classes)

  # This used to use Marc's raindrop_velocity() function, but has been
  # updated to go for the more flexible Beard model.
  classVelocities = terminalVelocitiesByClass(altitude=altitude,
                                              lat=latitude,
                                              diamClasses=classes, ...)

  ## Much, much faster version for calculating R.
  spectraMat = t(as.matrix(spectra))
  if(is.null(dim(spectra)))
    spectraMat = as.matrix(spectra)

  R = colSums(spectraMat * classWidths *
              classVelocities * classCentreDiams^3) * (6*pi/10^4)

  return(R)
}

totalDropConcentration = function(dsdMat, classes=get.classD()) {
  # Calculate total drop concentration N_t. This is sum of DSD over range
  # of drop diameters. Because we have discrete classes, DSD volumetric
  # numbers are multiplied by the drop class widths.
  #
  # Args:
  #  spectra: matrix, one DSD per row, col per diameter class [m^-3 mm^-1].
  #  classes: vector of drop diameter class (col for min, col for max) [mm].
  #
  # Returns: The total drop concentration Nt for each row [m^-3].

  # Get diameter class widths.
  classWidths = apply(classes, 1, diff)
  stopifnot(dim(dsdMat)[2] == length(classWidths))

  # Multiply each row by the class widths.
  mult = function(x) { return(x*classWidths) }
  dsdMult = t(apply(dsdMat, 1, mult)) # [m^-3 mm^-1 * mm] = [m^-3].

  # Sum rows and return.
  return(rowSums(dsdMult))
}

DSDDailyAccumAmounts = function(spectra) {
  # Find accumulated rainfall amount over time.
  #
  # Args:
  #   spectra: DSD spectra containing at least timestep and amount.
  #
  # Returns: accumulated rainfall by date.

  subset = data.frame(POSIXtime=spectra$POSIXtime,
                      station=spectra$station,
                      amount=spectra$amount)

  # Sum up daily amounts.
  stop("Check as.Date function; should be as.POSIXlt?")
  subset$date = factor(as.Date(subset$POSIXtime))
  sums = ddply(subset, .(date, station), summarise, sum=sum(amount, na.rm=T))

  # Order spectra by time step by station.
  sums = arrange(sums, station, date)

  # Replace NAs with 0 because they don't contribute to the sum.
  sums[is.na(sums)] = 0

  # Do cumulative sum for each station.
  summed = ddply(sums, .(station), summarise,
                 date = date,
                 accumulatedRainAmount = cumsum(sum))

  # Return date in POSIXct form for easier plotting.
  summed$date = as.POSIXct(summed$date, tz="UTC")

  return(summed)
}

calculateDSDMoments = function(spectra, classes, orders = c(3,4)) {
  # From a data.frame containing DSD spectra, calculate sample moments
  # of specified orders. NAs are removed.
  #
  # Args:
  #  spectra: matrix, one DSD per row, cols of diameter classes [m^-3 mm^-1].
  #  classes: min/max drop diameters [mm] for each class.
  #  orders: calculate moments of these orders (default: c(3, 4)).
  #
  # Returns: a data.frame containing sample moments per spectra and order.
  # Units of moments: [m^-3 mm^N] where N is the order of the moment.

  moments = list()

  classDiams = rowMeans(classes)
  classWidths = apply(classes, 1, diff)

  # Should have same number of diameters as classes.
  stopifnot(dim(spectra)[2] == length(classDiams))

  for(order in orders) {
    # Function to calculate moment for a single spectra.
    momentFun = function(x) {

      # The units of x are [m^-3 mm^-1].
      # Units of the moment are therefore:
      #   [m^-3 mm^-1 * mm^N * mm] = [m^-3 mm^N]
      moment = sum(x * (classDiams^order) * classWidths)
      return(moment)
    }

    # Apply moment function to all spectra.
    momentRes = list(apply(spectra, 1, momentFun))
    names(momentRes) = paste("moment_", order, sep="")
    moments = c(moments, momentRes)
  }

  return(data.frame(moments))
}

waterDensity = function(altitude, temperature, latitude) {
  ## Calculate the water density at a given altitude, for a given
  ## air temperature.
  ##
  ## Args:
  ##   altitude: Elevation above sea level [m]
  ##   temperature: Air temperature at sea level [deg. C].
  ##   latitude: Latitude (degrees N).
  ##
  ## Returns: Water density rho_w [g mm^-3].

  ## Constants.
  Kelvin = 273.15 # Freezing temp of water [K].
  lapse  = 0.0065 # Atmospheric lapse rate [K/m].
  pa0    = 101325 # Pressure at sea level [Pascal].
  Rd     = 287.04 # Gas constant of dry air [J/(kg*K)].
  temperature = temperature + Kelvin
  temperatureAtAltitude = Ta(h=altitude, Ta0=temperature, lapse=lapse)

  ## Calculate air pressure.
  airPressure = pa(h=altitude, pa0=pa0, lat=latitude,
                   Ta0=temperature, lapse=lapse, Rd=Rd)

  density = rhow(t=temperatureAtAltitude,
      p=airPressure, Kelvin=Kelvin, pa0=pa0) # [kg m^-3]
  return(density*1e-6) # [g mm^-3]
}

airDensity_sat = function(altitude, temperature, latitude) {
  ## Calculate the air density at a given altitude, for a given
  ## air temperature and latitude.
  ##
  ## Args:
  ##   altitude: Elevation above sea level [m]
  ##   temperature: Air temperature at sea level [deg. C].
  ##   latitude: Latitude (degrees N).
  ##
  ## Returns: Air density rho_a [g mm^-3].

  ## Constants.
  Kelvin = 273.15 # Freezing temp of water [K].
  lapse  = 0.0065 # Atmospheric lapse rate [K/m].
  pa0    = 101325 # Pressure at sea level [Pascal].
  Rd     = 287.04 # Gas constant of dry air [J/(kg*K)].
  temperature = temperature + Kelvin
  temperatureAtAltitude = Ta(h=altitude, Ta0=temperature, lapse=lapse)

  ## Calculate air pressure.
  airPressure = pa(h=altitude, pa0=pa0, lat=latitude,
                   Ta0=temperature, lapse=lapse, Rd=Rd)

  ## Calculate vapour pressure.
  vapourPressure = es(t=temperatureAtAltitude)

  density = rhoa(t=temperatureAtAltitude,
      p=airPressure, e=vapourPressure, Rd=Rd) # [kg m^-3]
  return(density*1e-6) # [g mm^-3]
}

airDensity = function(altitude, temperature, latitude, humidity) {
  ## Calculate the air density at a given altitude, for a given
  ## air temperature and latitude.
  ##
  ## Args:
  ##   altitude: Elevation above sea level [m].
  ##   temperature: Air temperature at sea level [deg. C].
  ##   latitude: Latitude (degrees N).
  ##   humidity: relative humidity (0-1).
  ##
  ## Returns: Air density rho_a [g mm^-3].

  ## Constants.
  Kelvin = 273.15 # Freezing temp of water [K].
  lapse  = 0.0065 # Atmospheric lapse rate [K/m].
  pa0    = 101325 # Pressure at sea level [Pascal].
  Rd     = 287.04 # Gas constant of dry air [J/(kg*K)].
  temperature = temperature + Kelvin
  temperatureAtAltitude = Ta(h=altitude, Ta0=temperature, lapse=lapse)

  ## Calculate air pressure.
  airPressure = pa(h=altitude, pa0=pa0, lat=latitude,
                   Ta0=temperature, lapse=lapse, Rd=Rd)

  ## Calculate vapour pressure.
  vapourPressure = eact(rh=humidity, t=temperatureAtAltitude, p=airPressure)

  density = rhoa(t=temperatureAtAltitude,
      p=airPressure, e=vapourPressure, Rd=Rd) # [kg m^-3]
  return(density*1e-6) # [g mm^-3]
}

liquidWaterContentFromMoments = function(moments, rho_w) {
  # Calculate the liquid water content (LWC) from spectra moment 3.
  #
  # Args:
  #   moments: data.frame containing at least moment_3 [m^-3 mm^3]
  #   rho_w: the density of water to use [g mm^-3]
  #          (If unknown, use Kell 1975 value of 1000 kg / m^3, ie rho_w=0.001).
  #
  # Returns: the LWC value [g m^3] for each set of moments.

  const = (pi * rho_w) / 6       # [g mm^-3].
  res = const * moments$moment_3 # [g mm^-3 m^-3 mm^3] = [g m^-3].
  return(res)
}

meanVolumeDiameterFromMoments = function(moments) {
  # Calculate the volume weighted mean volume diameter D_m for DSD
  # spectra moments. This is the ratio of the fourth to the third moment
  # of the DSD.
  #
  # Args:
  #  moments: data.frame containing at least moment_3 and moment_4.
  #
  # Returns: The D_m value [mm] for each moment row.

  # Units of Dm are [m^3 mm^4 / m^3 mm^3] = [mm].
  return(moments$moment_4 / moments$moment_3)
}

simpleGammaModel = function(N, mu, lambda, diam, diamClassWidths) {
    ## Calculate alpha, the normalisation factor.
    stopifnot(length(diam) > 1)
    ## alpha = 1 / sum(diam^mu * exp(-lambda*diam) * diamClassWidths)
    return(N * diam^mu * exp(-lambda*diam))
}

weightedGammaModel = function(mu, Nw, Dm, D) {
  # Weighted gamma model based on Bringi_2001 Eq. 7.62a, page 410.
  # Uses weighted gamma of Willis 84.
  # Nw, D0, and D should be known, mu is the parameter.

  fMu = 6/(4)^4 * ((4 + mu)^(mu+4)) / (gamma(mu + 4))
  res = Nw * fMu * (D/Dm)^mu * exp(-(4 + mu)*(D/Dm))
  return(res)
}

fitGammaParameters = function(dsdData, Dm, LWC,
    diams=rowMeans(get.classD()),
    startVals=c(1)) {
  stop("Update to include calculated water density rho_w.")
  # Fit a weighted gamma DSD model to DSD spectra. Return the
  # parameters of the model (N for intercept, mu for shape and Lambda).
  # If the fitting fails, return -1 for all parameters.
  #
  # Args:
  #   dsdData: A vector containing the DSD spectra to fit to.
  #   Dm: The Dm for the DSD [mm].
  #   LWC: The liquid water content [g m-3] for the DSD.
  #            [Note that .
  #   diams: Class centre drop diameters [mm].
  #   startVals: Starting values for N, mu, lambda (default c(1,1,1)).
  #
  # Returns: Fitted value of mu.

  # Calculate Nw [mm-1 m-3] from the LWC [g m-3] and D0 [mm].
  # Assumes a water density of 1 g cm-3.
  Nw = ((4)^4 / pi) * (10^3 * LWC) / Dm^4

  # Check vector lengths.
  stopifnot(length(Nw) == 1)
  stopifnot(length(Dm) == 1)
  stopifnot(length(dsdData) == length(diams))

  ## The upper value here, of 167, is based on the highest mu that
  ## the gamma function can handle before returning Inf.
  startVals = list(mu=1)
  dataToFit = data.table(data.frame(var=t(dsdData), diam=diams, Dm=Dm, Nw=Nw))
  res = try(nlsLM(formula="var~weightedGammaModel(mu, Nw, Dm, diam)",
      data=dataToFit, start=startVals, control=nls.lm.control(maxiter=500)),
      silent=TRUE)
  if(class(res) != "nls") {
    return(data.frame(mu=as.numeric(NA), lambda=as.numeric(NA), Nw=Nw))
  }

  res = data.frame(t(coefficients(res)))
  res$lambda = (4 + res$mu) / Dm
  res$Nw = Nw
  return(res)
}

fitZR = function(data, plot=FALSE) {
  ## Fit the Z-R relationship using orthogonal linear regression
  ## that doesn't require a dependent variable. The fit is
  ## performed on log10(R)~log10(Z).
  ##
  ## Args:
  ##   data: The data to work on, must include Zh and R.
  ##   plot: Plot results? (Default: FALSE). If true produces two
  ##         plots (linear and log).
  ##
  ## Returns: a data.frame containing a and b values for Z=aR^b.

  ## Work as data table, and convert Zh to linear units.
  data = data.table(data)

  ## Use orthogonal linear regression (also called total linear regression)
  ## to find the slope and intercept of log10(R)~log10(Z). The use
  ## of orthogonal linear regression means that there is no dependent variable
  ## and residuals are reduced for both variables.
  require(rgr)
  R = data[, R]
  Z = data[, Z] # Note, linear units for Z.
  res = gx.rma(R, Z, x1lab="R", x2lab="Z", log=TRUE)

  ## Retrieve slope and intercept of log-log fit, and convert to a and b in
  ## Z = aR^b relationship. Note:
  ## log(Z) = log(aR^b) -> log(Z) = b*log(R) + log(a).
  intercept = res$a0
  slope = res$a1
  a = 10^intercept
  b = slope

  ## Produce plots.
  if(plot) {
    ## Log plot.
    print(ggplot(data, aes(x=R, y=10^(Zh/10))) +
      geom_point(shape=1, size=2) + theme_bw(textSize) +
      scale_x_continuous(trans="log10") +
      scale_y_continuous(trans="log10") +
      geom_abline(intercept=intercept, slope=slope, colour="red"))

    ## Linear plot.
    synthR = seq(0.1, max(R), by=0.1)
    lineZh = 10*log10(a*synthR^b)
    line = data.frame(R=synthR, Zh=lineZh)
    print(ggplot(data, aes(x=R, y=Zh)) +
      geom_point(shape=1, size=2) + theme_bw(textSize) +
      geom_line(data=line, colour="red"))
  }

  return(data.table(a=a, b=b))
}

fitZR_nonlinear = function(data) {
  # Fit the Z-R relationship (Z = aR^b).
  #
  # Args:
  #  data: Must contain both Z and R.
  #
  # Results: 'a' and 'b' fitted using non-linear least squares.

  # Start with Marshal-Palmer 1955 values.
  start = list(a=200, b=1.6)

  # Work on Zh in linear units.
  data$ZhLin = 10^(data$Zh/10)

  return(data.frame(t(coefficients(nlsLM(formula="ZhLin~a*(R^b)", data,
                              control=nls.lm.control(maxiter=500),
                              start=start)))))
}

fitRZ_nonlinear = function(data) {
  # Fit the R-Z relationship (R = (1/a)^(1/b)*Z^(1/b)).
  #
  # Args:
  #  data: Must contain both Z and R.
  #
  # Results: 'a' and 'b' fitted using non-linear least squares.

  # Start with Marshal-Palmer 1955 values.
  start = list(a=200, b=1.6)

  # Work on Zh in linear units.
  data$ZhLin = 10^(data$Zh/10)

  return(data.frame(t(coefficients(nlsLM(formula="R~(1/a)^(1/b)*ZhLin^(1/b)", data,
                              control=nls.lm.control(maxiter=500),
                              start=start)))))
}

dsdMoment = function(spectra, n, cols=paste("D", seq(1,32), sep=""),
    diams=rowMeans(get.classD()),
    widths=apply(get.classD(), 1, diff)) {
    ## Calculate a specific DSD moment.
    ## Default classes are for Parsivel DSDs.
    ##
    ## Args:
    ##   spectra: DSD spectra containing columns defined in 'cols'.
    ##   n: order of the moment to calculate.
    ##   cols: Which columns give DSD concentrations?
    ##   diams: Diameter [mm] for each DSD class.
    ##   widths: Width [mm] of each DSD class.
    ##
    ## Returns: the nth statistical moment for each DSD.

    d = matrix(rep(diams, nrow(spectra)), nrow=nrow(spectra), byrow=TRUE)
    w = matrix(rep(widths, nrow(spectra)), nrow=nrow(spectra), byrow=TRUE)
    dsds = as.matrix(spectra[, cols, with=FALSE])
    return(rowSums(dsds*(d^n)*w))
}

massWeightedAxisRatio = function(spectra,
    ratioFunc=raindrop_axis_ratio,
    cols=paste("D", seq(1,32), sep=""),
    diams=rowMeans(get.classD()),
    widths=apply(get.classD(), 1, diff)) {
    ## Find the mass-weighted mean drop axis ratio for DSDs.
    ## Default classes are for Parsivel DSDs.
    ##
    ## Args:
    ##   spectra: DSD spectra containing columns defined in 'cols'.
    ##   ratioFunc: The ratio function to use.
    ##   cols: Which columns give DSD concentrations?
    ##   diams: Diameter [mm] for each DSD class.
    ##   widths: Width [mm] of each DSD class.
    ##
    ## Returns: the mass-weighted mean axis ratio for each DSD.

    ratios = ratioFunc(diams)
    if(!all(as.matrix(spectra)[, which(ratios < 0)] == 0, na.rm=TRUE))
        stop("Resetting axis ratios for columns containing drops.")
    ratios[ratios < 0] = 0.6
    
    ratios = matrix(rep(ratios, nrow(spectra)), nrow=nrow(spectra), byrow=TRUE)
    diams = matrix(rep(diams, nrow(spectra)), nrow=nrow(spectra), byrow=TRUE)
    widths = matrix(rep(widths, nrow(spectra)), nrow=nrow(spectra), byrow=TRUE)
    dsds = as.matrix(spectra[, cols, with=FALSE])

    return(rowSums(ratios * diams^3 * dsds * widths) /
        rowSums(diams^3 * dsds * widths))
}

radarReflectivityFromDSD_old = function(DSD, classes, freq, temp, incidence,
    ratio_function=raindrop_axis_ratio, c = 299792.458*1e3, cantingSD=0,
    minAxisRatio=0.4) {
  # Calculate radar reflectivities from the DSD. This is a function
  # originally written by Jacopo Grazioli, but modified so that drop width
  # classes are specified as an argument.
  #
  # Args:
  #  DSD: A matrix with volumic DSD data. Rows are measurements, columns are
  #       the diameter classes. Entries in [m-3 mm-1].
  #  classes: Diameter classes (column for mins, column for maxs) [mm].
  #  freq: Radar frequency [GHz].
  #  temp: Temperature [degrees C].
  #  incidence: Radar incidence angle [degrees] (eg, 90 indicates a vertical
  #             scan).
  #  ratio_function: Raindrop axis ratio function (default:
  #                  raindrop_axis_ratio())
  #  c: Speed of light [m.s^-1].
  #  cantingSD: Standard deviation of gaussian canting angle distribution.
  #
  # Returns: A data.frame containing Zh and Zv [mm^6 m^-3] for each
  #          measurement.
  #
  # NOTE: The canting angle is not taken into account.

  DSD = data.table(DSD)
  stopifnot(dim(DSD)[2] == length(classes[,1]))

  class_size = rowMeans(classes) # Class centres [mm].
  class_spread = apply(classes, 1, diff) # Class widths [mm].

  axis_ratios = ratio_function(class_size)

  ## TMatrix crashes when drop axis ratios are too small.
  ## Reset small axis ratios, as long as they don't affect
  ## bins that contain drops in the DSD.
  if(length(intersect(which(axis_ratios < minAxisRatio), 
                      which(colSums(DSD) > 0)) != 0))
      stop("Error: axis ratio setting for Tmatrix will affect DSD.")
  axis_ratios[which(axis_ratios < minAxisRatio)] = minAxisRatio

  index_water = ref_index_water(temp, freq)
  wavelength = (c/freq*1e-9)*1e3	# Get wavelength from frequency [mm].

  # Get backscattering cross section horizontal and vertical components.
  dummy = back_scat_cross(tabD=class_size, tab_ratio=axis_ratios,
      w=wavelength, m=index_water, theta=(90-incidence), phi=0,
      canting_sd=c(cantingSD, cantingSD))
  BSH = dummy[,1]
  BSV = dummy[,2]
  ## BSH[which(class_size > 6)] = 0
  ## BSV[which(class_size > 6)] = 0

  Kw = (index_water^2-1) / (index_water^2+2)

  # Cz in [cm^4.10^6]. The 10^6 is used to convert cm^4 into mm^4.
  Cz = (1e6*(wavelength*1e-1)^4) / (pi^5*abs(Kw)^2)

  ## Invert the DSDs so there is one DSD per column.
  ## This is a much, much faster way to calculate (replaces loop below).
  DSDmat = as.matrix(DSD)
  Zh = colSums(Cz * (t(DSDmat) * BSH * class_spread), na.rm=FALSE)
  Zv = colSums(Cz * (t(DSDmat) * BSV * class_spread), na.rm=FALSE)

  ## Find Zh and Zv for each DSD.
  ##for(meas in 1:n_meas) {
  ##  Zh[meas] = Cz*sum(BSH*DSD[meas,]*class_spread,na.rm=FALSE)
  ##  Zv[meas] = Cz*sum(BSV*DSD[meas,]*class_spread,na.rm=FALSE)
  ##}

  out = data.frame(Zh=as.numeric(Zh), Zv=as.numeric(Zv))
  return(out)
}

specificAttenuationFromDSD_old = function(DSD, classes, freq, temp, incidence,
    ratio_function=raindrop_axis_ratio, cl = 299792.458*1e3, cantingSD=0,
    minAxisRatio=0.4) {
  # Calculate specific attenuation from the DSD.
  #
  # Args:
  #  DSD: A matrix with volumic DSD data. Rows are measurements, columns are
  #       the diameter classes. Entries in [m-3 mm-1].
  #  classes: Diameter classes (column for mins, column for maxs) [mm].
  #  freq: Radar frequency [GHz].
  #  temp: Temperature [degrees C].
  #  incidence: Radar incidence angle [degrees] (eg, 90 indicates a vertical
  #             scan).
  #  ratio_function: Raindrop axis ratio function (default:
  #                  raindrop_axis_ratio())
  #  c: Speed of light [m.s^-1].
  #  cantingSD: Standard deviation of Gaussian canting angle distribution (default: 0).
  #
  # Returns: A data.frame containing k [d km-1] for each DSD.

  DSD = data.table(DSD)
  stopifnot(dim(DSD)[2] == length(classes[,1]))

  class_size = rowMeans(classes) # Class centres [mm].
  class_spread = apply(classes, 1, diff) # Class widths [mm].

  ## TMatrix crashes when drop axis ratios are too small.
  ## Reset small axis ratios, as long as they don't affect
  ## bins that contain drops in the DSD.
  axis_ratios = ratio_function(class_size)
  if(length(intersect(which(axis_ratios < minAxisRatio), 
                      which(colSums(DSD) > 0)) != 0))
      stop("Error: axis ratio setting for Tmatrix will affect DSD.")
  axis_ratios[which(axis_ratios < minAxisRatio)] = minAxisRatio
  
  index_water = ref_index_water(temp, freq)
  wavelength = (cl/freq*1e-9)*1e3 # Get wavelength from frequency [mm].

  # Get extinction cross section horizontal and vertical components.
  dummy = ext_scat_cross(tabD=class_size, tab_ratio=axis_ratios,
      w=wavelength, m=index_water, theta=(90-incidence), phi=0,
      canting_sd=c(cantingSD, cantingSD))
  EH = dummy[,1]
  EV = dummy[,2]

  ## Here is how to get Eq. 1.133d in Bringi 2001.
  ## To convert EH and EV from cm2 to m2, multiply by 1e-4.
  ## eg 1 cm2 = 0.0001 m2.

  ## N(D).dD [m-3]
  ## 1e-4.EH [m2]
  ## => x = (N(D) 1e-4 EH dD) [m-1]
  ## => 1e3 x [km-1]
  ## => 1e-1 N(D) EH dD [km-1]
  ## => To be in km-1, divide result by 10.
  ## To convert to dB, multiply by 10*log10(exp(1)) = 4.343.

  ## Invert the DSDs so there is one DSD per column.
  DSDmat = as.matrix(DSD)
  kh = 10*log10(exp(1))*(colSums((t(DSDmat) * EH * class_spread), na.rm=FALSE) / 10)
  kv = 10*log10(exp(1))*(colSums((t(DSDmat) * EV * class_spread), na.rm=FALSE) / 10)

  out = data.frame(kh=as.numeric(kh), kv=as.numeric(kv))
  return(out)
}

specificDiffPhaseFromDSD_old = function(DSD, classes, freq, temp, incidence,
    ratio_function=raindrop_axis_ratio,
    cl=299792.458*1e3, cantingSD=0,
    minAxisRatio=0.4) {
  ## Calculate specific differential phase shift (Kdp) from the DSD.
  ## Based on code by Joel Jaffrain, modified by Jacopo Grazioli, then by TR.
  ##
  ## Args:
  ##  DSD: A matrix with volumic DSD data. Rows are measurements, columns are
  ##       the diameter classes. Entries in [m-3 mm-1].
  ##  classes: Diameter classes (column for mins, column for maxs) [mm].
  ##  freq: Radar frequency [GHz].
  ##  temp: Temperature [degrees C].
  ##  incidence: Radar incidence angle [degrees] (eg, 90 indicates a vertical
  ##             scan).
  ##  ratio_function: Raindrop axis ratio function (default:
  ##                  raindrop_axis_ratio())
  ##  cl: Speed of light [m.s^-1].
  ##  cantingSD: Standard deviation of Gaussian canting angle distribution (default: 0).
  ##
  ## Returns: A data.frame of specific differential phase (Kdp) values
  ##          in [deg. km^-1].

  ## Check dimension of DSD matrix:
  DSD = data.table(DSD)
  stopifnot(dim(DSD)[2] == length(classes[,1]))

  class_size = rowMeans(classes) # Class centres [mm].
  class_spread = apply(classes, 1, diff) # Class widths [mm].

  axis_ratios = ratio_function(class_size)

  ## TMatrix crashes when drop axis ratios are too small.
  ## Reset small axis ratios, as long as they don't affect
  ## bins that contain drops in the DSD.
  if(length(intersect(which(axis_ratios < minAxisRatio), 
                      which(colSums(DSD) > 0)) != 0))
      stop("Error: axis ratio setting for Tmatrix will affect DSD.")
  axis_ratios[which(axis_ratios < minAxisRatio)] = minAxisRatio

  index_water = ref_index_water(temp, freq)
  wavelength = (cl/freq*1e-9)*1e3 # Get wavelength from frequency [mm].

  # Get forward scattering amplitudes.
  # The "90" here is for the azimuth - no difference if 90, 0, or 45;
  # in Jacopo's code it was 90 but here I have set it to 0, for
  # consistency with other Tmatrix routines.
  forwardScatteringAmplitudes = fwrd_scat_ampl_comp(tabD=class_size,
      tab_ratio=axis_ratios, w=wavelength, m=index_water,
      theta=(90-incidence), phi=0, canting_sd=c(cantingSD, cantingSD))

  DSDmat = as.matrix(DSD)
  Shh = forwardScatteringAmplitudes[,1]
  Svv = forwardScatteringAmplitudes[,2]
  ShhMinusSvv = Re(Shh - Svv) # mm
  Ck = (180/pi)*wavelength*1e-3	# [°.m]
  Kdp_res = Ck*colSums(t(DSDmat)*ShhMinusSvv*class_spread, na.rm=FALSE)

  return(data.frame(Kdp=Kdp_res))
}
