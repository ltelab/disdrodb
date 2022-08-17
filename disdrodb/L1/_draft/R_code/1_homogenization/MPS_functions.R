## Functions for processing of MPS particle probe data.
## Author: T. Raupach <tim.raupach@epfl.ch>
##
## Note that MPS files supplied by Thurai tend to be in Windows
## format.  To convert from Windows to unix file style, use
##   > sed -e "s/^M//g" old.csv > new.csv
## and replace "#" character in heading.

MPSDSDs = function(dat, timeRes, startTime, altitude, latitude, seaTemp, humidity,
    Rp=50e-6, n=64, x=seq(1,62), wlength=652e-9, binCols=paste("Bin.", x, sep="")) {
    ## Calculate DSDs from MPS data, using equations A1 to A5 in
    ## Thurai JAMC 2017.
    ##
    ## Args:
    ##   dat: data.table containing MPS data, altitude, latitude.
    ##   timeRes: time resolution for output DSDs [s].
    ##   startTime: time from which to start the series [POSIXct, UTC].
    ##   altitude, latitude: Station information, altitude [m] and latitude [deg N].
    ##   seaTemp: sea level temperature to assume.
    ##   humidity: humidity (at altitude) to assume.
    ##   Rp: Probe resolution [m] (default: 50e-6, 50 microns).
    ##   n: Number of diodes (default: 64).
    ##   x: Bin numbers (default: 1-62).
    ##   wlength: Laser wavelength [m] (default 652e-9, 652 nm).
    ##   binCols: Column names for bins in dat.
    ##
    ## Returns: DSD in mm^-1 m^-3, per time step.

    stopifnot(class(dat)[1] == "data.table")
    stopifnot(length(x) == length(binCols))
    stopifnot(all(binCols %in% names(dat)))
    stopifnot(length(altitude) == 1)
    stopifnot(length(latitude) == 1)
    
    ## Calculate the effective array width (EAW) for each bin (Eq. A1).
    EAW = Rp*(n-x-1) # [m]
    
    ## Calculate equivolume diameters for each bin [m], assuming
    ## that drops this small are spheres so their width equals
    ## their diameter. Convert to mm and microns for later.
    Deq = (x-1)*Rp+(Rp/2) # [m]
    DeqMm = Deq * 1e3     # [mm]
    DeqMi = Deq * 1e6     # [microns]
    
    ## Drop radius for each bin.
    R = Deq/2 # [m]

    ## Bin widths.
    dDmm = rep(Rp * 1e3, length(x)) # [mm]

    ## Calculate the depth of field (DoF) (Eq. A2).
    DoF = (6*R^2) / wlength # [m]
    DoF[DoF > 0.2] = 0.2    # Extra requirement (pers. comm. Thurai).

    ## Calculate the effective sampling area (Eq. A4).
    Aeff = EAW * DoF # [m^2]

    ## Calculate the expected terminal velocity for each bin (Eq. A5).
    ## Equation takes D in microns and v is given in [cm/s].
    ## The -0.50 is changed from +0.50, according to IDL code from Thurai
    ## and MPS user guide Eq. 2.13.
    v = -(19.27 - 0.50*DeqMi) - (9.04e-5 * DeqMi^2) + (5.66e-9 * DeqMi^3) # [cm/s]

    ## Correct velocities using method of Beard JAOT 1985.
    mD = 0.375 + 0.025*DeqMm
    rho_0 = airDensity(altitude=0, temperature=seaTemp,
        latitude=dat[, unique(latitude)], humidity=humidity)
    rho = airDensity(altitude=dat[, unique(altitude)], temperature=seaTemp,
        latitude=dat[, unique(latitude)], humidity=humidity)
    rho_ratio = (rho_0 / rho)
    corrfact = rho_ratio^mD
    v = v*corrfact
    
    ## Convert velocities to m/s and deal with negative velocities.
    v = v/100 # [m/s]

    ## Set negative velocities to zero (pers comm Thurai).
    v[v < 0] = 0

    ## Calculate time steps. as.POSIXct handles time format with
    ## HH:MM:SS.ss.  Note that MPS times are not round seconds. A one
    ## minute period will include the first second that overlaps with
    ## it, and will not include the last second that overlaps with it.
    dat = copy(dat)
    dat[, timestamp := as.POSIXct(paste(Date, Time), tz="UTC")]
    dat = dat[timestamp >= startTime]

    ## Assign each recording to a time step. right=TRUE means that
    ## time t exactly is put into class t, since we assume times are
    ## the end of the recording second.
    timeBreaks = seq(startTime, dat[, max(timestamp)]+(2*timeRes), by=timeRes)
    dat[, POSIXtime := cut.POSIXt(timestamp+timeRes, timeBreaks, right=TRUE)]
    stopifnot(!any(dat[, is.na(POSIXtime)]))
    dat[, POSIXtime := as.POSIXct(POSIXtime, tz="UTC")]

    ## Sum drop counts per time step.
    dropsPerTime = dat[, lapply(.SD, sum), .SDcols=binCols, by=POSIXtime]

    ## Apply factors to get DSDs.
    facts = Aeff * v * timeRes * dDmm # [m^2 m s^-1 s mm = m^3 mm]
    dsds = dropsPerTime[, .SD/facts, .SDcols=binCols, by=POSIXtime] # [mm^-1 m^-3]

    return(dsds)
}

combineDSDsWithMPS = function(mps, dsd, mpsClasses, dsdClasses, mpsCols,
    dsdCols, mpsTo, matchCols=c("POSIXtime", "station", "altitude", "latitude", "longitude")) {
    ## Combine MPS measurements with DSDs measured by another
    ## instrument.
    ##
    ## Args:
    ##   mps: The MPS dsds [mm-1 m-3] per class.
    ##   dsd: Other DSD data [mm-1 m-3] per class.
    ##   mpsClasses: Class definitions for MPS (min/max in mm).
    ##   dsdClasses: Class definitions for other DSDs (min/max in mm).
    ##   mpsCols: Column names for MPS DSD classes.
    ##   dsdCols: Column names for other DSD classes.
    ##   mpsTo: Diameter up to which to use MPS data [mm] (class
    ##          centres are used for this cutoff and an error occurs
    ##          if gaps are introduced into the new classes).
    ##   matchCols: Columns that must match between MPS and DSD data.
    ##
    ## Returns: data.table with combined classes, column names "class"
    ##          for DSDs [mm-1 m-3], "diams" for class diameters [mm],
    ##          "width" for widths [mm]. Combinations are made by time
    ##          step.

    stopifnot("POSIXtime" %in% names(mps))
    stopifnot("POSIXtime" %in% names(dsd))
    stopifnot(all(mpsCols %in% names(mps)))
    stopifnot(all(dsdCols %in% names(dsd)))

    ## Get widths and centres of the classes.
    mpsDiams = rowMeans(mpsClasses)
    mpsWidths = apply(mpsClasses, 1, diff)
    dsdDiams = rowMeans(dsdClasses)
    dsdWidths = apply(dsdClasses, 1, diff)
    
    ## Select classes from each set.
    mpsColIdx = which(mpsDiams + mpsWidths/2 <= mpsTo)
    dsdColIdx = which(dsdDiams - dsdWidths/2 >= mpsTo)

    ## Combine to make new diameter classes.
    newDiams = c(mpsDiams[mpsColIdx], dsdDiams[dsdColIdx])
    newWidths = c(mpsWidths[mpsColIdx], dsdWidths[dsdColIdx])

    ## Check there are no gaps introduced into the diameter classes.
    mins=newDiams - newWidths/2
    maxs=newDiams + newWidths/2
    if(any(abs(mins[2:length(mins)] - maxs[1:length(maxs)-1]) > 1e-10)) {
        stop("combineDSDs: New classes contain gaps.")
    }

    ## Select out the data columns.
    mpsSelectedCols = mpsCols[mpsColIdx]
    dsdSelectedCols = dsdCols[dsdColIdx]

    setkeyv(mps, matchCols)
    setkeyv(dsd, matchCols)
    
    selectMPS = mps[dsd, nomatch=0][, c(matchCols, mpsSelectedCols), with=FALSE]
    selectDSD = dsd[mps, nomatch=0][, c(matchCols, dsdSelectedCols), with=FALSE]

    setnames(selectMPS, mpsSelectedCols,
             paste("class", seq(1, length(mpsSelectedCols)), sep=""))
    setnames(selectDSD, dsdSelectedCols,
             paste("class", seq(length(mpsSelectedCols)+1,
                                length(mpsSelectedCols)+
                                length(dsdSelectedCols)), sep=""))
    
    ## Join selected columns together by time.
    setkeyv(selectMPS, matchCols)
    setkeyv(selectDSD, matchCols)
    res = selectMPS[selectDSD]

    ## Add columns to store widths and diameters.
    for(i in seq(1, length(newDiams))) {
        diamCol = paste("diam", i, sep="")
        res[, (diamCol) := newDiams[i]]

        widthCol = paste("width", i, sep="")
        res[, (widthCol) := newWidths[i]]
    }
    
    ## Sometimes the combination removes all drops; remove these zero lines.
    res[, sum := rowSums(.SD), .SDcols=paste("class", seq(1, length(newDiams)), sep="")]
    res = res[sum > 0]
    res[, sum := NULL]
    
    return(res)
}
