## Functions to calculate radar properties from DSDs using Tmatrix.
## 
## Adapted by T. Raupach from code by M. Schleiss, J. Grazioli, with
## advice from D. Wolfensburger and translations of code from the
## Tmatrix python library.
## 
## Updated November/December 2017 by T. Raupach. Changes are:
##
##   - Canting angles are now taken into account and give
##     integrated/deterministic results. The alpha angle (rotation) is
##     uniformly chosen between 0 and 360 degrees, while the beta
##     angle (canting) is chosen according to a normal distribution
##     with zero mean and a given standard deviation. For each alpha
##     angle the possible scattering amounts are averaged, weighted by
##     the normal distribution of beta angles; then the final result
##     is the mean of these averages across all alpha angles.
##   - Much cleaning up of code, functions for reused sections, etc.
##   - A cache can be used (see "cacheDir" argument to functions) which
##     speeds up processing time for repeated calls.
## 
## Notes:
##   - Assumptions are made that the DSDs are for liquid rain; ie
##     canting angles do not depend on drop size.
##   - To use these functions, Marc Schliess' Tmatrix R library is
##     required so that the Tmatrix() function is available.
##
## Main functions:
##   radarReflectivityFromDSD()    -- calculate Zh and Zv.
##   specificDiffPhaseFromDSD()    -- calculate Kdp.
##   specificAtteonuationFromDSD() -- calculate k.

require(gaussquad)
require(data.table)

###################### RADAR REFLECTIVITY (Zh) ######################

radarReflectivityFromDSD = function(DSD, classes, freq, temp, incidence,
    ratio_function=raindrop_axis_ratio, cantingSD=0, minAxisRatio=0.4,
    cacheDir=NULL, cacheID="", KwSquared=0.93) {
    ## Calculate radar reflectivities from DSDs.
    ##
    ## Args:
    ##  DSD: Volumetric DSD data. Rows are measurements, columns are the diameter
    ##       classes. Entries in [m-3 mm-1].
    ##  classes: Diameter classes (col for mins, col for maxs) [mm].
    ##  freq: Radar frequency [GHz].
    ##  temp: Temperature [degrees C].
    ##  incidence: Radar incidence angle [degrees] (90=vertical scan).
    ##  ratio_function: Raindrop axis ratio function
    ##                  (default: raindrop_axis_ratio())
    ##  cantingSD: Standard deviation of gaussian canting angle dist.
    ##  minAxisRatio: Minimum raindrop axis ratio [v/h] to accept
    ##                (default: 0.4).
    ##  cacheDir: Directory in which to cache Tmatrix results
    ##            (default: NULL, no cache).
    ##  cacheID: ID to use for cache, e.g. axis ratio function
    ##           (default: none).
    ##  Kw: The dielectric constant of water to use; 0.93 for MXPOL,
    ##      depends on radar you want to compare with (default:
    ##      0.93). 0.93 is sourced from Smith_JCAM_1984.
    ##
    ## Returns: A data.table containing Zh and Zv [mm^6 m^-3] for each
    ##          measurement.

    DSD = data.table(DSD)
    stopifnot(dim(DSD)[2] == length(classes[,1]))
    
    class_size = as.numeric(rowMeans(classes)) # Class centres [mm].
    class_spread = as.numeric(apply(classes, 1, diff)) # Class widths [mm].

    axis_ratios = ratio_function(class_size)
    axis_ratios = resetAxisRatios(axis_ratios, DSD, minAxisRatio)

    index_water = ref_index_water(temp, freq)
    wavelength = wavelengthFromFreq(freq) # [mm]

    ## Get backscattering cross section horizontal and vertical components.
    backScatter = backScatterCrossSection(tabD=class_size,
        tab_ratio=axis_ratios, w=wavelength, m=index_water,
        theta=(90-incidence), phi=0, cantingSD=cantingSD,
        cacheDir=cacheDir, cacheID=cacheID)

    BSH = backScatter$BSH
    BSV = backScatter$BSV

    ## K is the dielectric factor of water, which is defined as follows:
    ##warning("Change back Kw value")
    ##Kw = (index_water^2-1) / (index_water^2+2)
    ##Cz = (1e6*(wavelength*1e-1)^4) / (pi^5*abs(Kw)^2)
    ## but since index_water depends on the temperature and radar
    ## frequency, this value is set to a constant in the radar
    ## processing code.  Therefore to properly match measured radar
    ## values, we use the same constant. For MXPOL this is 0.93 (also
    ## from Smith_JCAM_1984).

    ## Cz in [cm^4.10^6]. The 10^6 is used to convert cm^4 into mm^4.
    Cz = (1e6*(wavelength*1e-1)^4) / (pi^5*KwSquared)

    DSDmat = as.matrix(DSD)
    widthMat = matrix(rep(class_spread, nrow(DSD)), byrow=TRUE, nrow=nrow(DSD))
    BSHMat = matrix(rep(BSH, nrow(DSD)), byrow=TRUE, nrow=nrow(DSD))
    BSVMat = matrix(rep(BSV, nrow(DSD)), byrow=TRUE, nrow=nrow(DSD))
    Zh = rowSums(Cz * (DSDmat * BSHMat * widthMat), na.rm=FALSE)
    Zv = rowSums(Cz * (DSDmat * BSVMat * widthMat), na.rm=FALSE)
    
    out = data.table(Zh=as.numeric(Zh), Zv=as.numeric(Zv))
    return(out)
}

backScatterCrossSection = function(tabD, tab_ratio, w, m, theta, phi,
    cantingSD=0, cacheDir=NULL, cacheID="") {
    ## Calculate backscattering cross sections [cm2] for a sequence
    ## of drop diameters and axis-ratios.
    ## 
    ## Args:
    ##   tabD: vector of equivolumetric drop diameters [mm]
    ##   tab_ratio: vector of drop axis ratios (vertical/horizontal) [-]
    ##   w: wavelength [mm]
    ##   m: complex refractivity index of water [-]
    ##   theta: zenith of incident beam [°] (90° = horizontal)
    ##   phi: azimuth of incident beam [°] (90° = along the y-axis)
    ##   cantingSD: standard deviation of the raindrop canting angle
    ##              [deg] (default: 0 degrees). The mean is always
    ##              zero.
    ##   cacheDir: If specified, look for precalculated caches in the directory.
    ##   cacheID: special ID for the cache to use (default: none).
    ## 
    ## Returns: list of backscattering cross sections [cm2].
    ##          BSH: backscattering cross sections for Hpol.
    ##          BSV: backscattering cross sections for Vpol.
    ## 
    ## Note: the canting does not depend on particle size.
    ## 
    ## Source: Marc Schleiss, EPFL-LTE, July 2011.
    ## J. Grazioli introduce canting 2014, and glitch fixes.
    ## T. Raupach updated to use a cache for speed and to improve
    ## canting angle implementation, Nov 2017.

    ## Do input tests.
    checkTmatrixInputs(tabD=tabD, tab_ratio=tab_ratio, w=w, m=m, theta=theta,
                       phi=phi, cantingSD=cantingSD)
    
    ## Determine incidence and scattering angle.
    if(phi>180) {phi <- phi-180}
    PHI0  <- phi
    PHI   <- PHI0+180
    THET0 <- theta
    THET  <- 180-theta

    ## Set wavelength, refractive index and reference angles.
    LAM <- w
    MRR <- Re(m)
    MRI <- Im(m)

    ## Load cache.
    results = NULL
    if(!is.null(cacheDir)) {
        results = loadCache(cacheDir=cacheDir, w=w, m=m, theta=theta, phi=phi,
            cantingSD=cantingSD, tabD=tabD, ratios=tab_ratio,
            type="backscatter", cacheID=cacheID)
    }

    ## If no cache, calculate Tmatrix values.
    if(is.null(results)) {
        results = TMatrixBackScatter(
            diams=tabD, ratios=tab_ratio, cantingSD=cantingSD,
            LAM=LAM, MRR=MRR, MRI=MRI, THET0=THET0, THET=THET,
            PHI0=PHI0, PHI=PHI, RAT=1, NP=-1, NDGS=2, DDELT=0.001)
    }

    ## Save cache if required.
    if(!is.null(cacheDir)) {
        saveCache(cache=results, cacheDir=cacheDir, w=w, m=m, theta=theta,
                  phi=phi, cantingSD=cantingSD, type="backscatter",
                  cacheID=cacheID)
    }
    
    ## Check radii and axis ratios are correct.
    stopifnot(identical(results[, axi], tabD/2))
    stopifnot(identical(results[, eps], 1/tab_ratio))
    
    return(list(BSH=results[, hres], BSV=results[, vres]))
}

TMatrixBackScatter = function(diams, ratios, cantingSD,
    LAM, MRR, MRI, THET0, THET, PHI0, PHI, RAT, NP, NDGS, DDELT, ...) {
    ## Calculate backscattering coefficients using Tmatrix.
    ##
    ## Args:
    ##   diams: Diameters to use (not radii, conversion later).
    ##   ratios: Ratio for each diameter (normal < 1, conversion later).
    ##   cantingSD: raindrop canting angle std. dev [deg].
    ##   LAM, MRR, [...], DDELT: Tmatrix() arguments.
    ##   ...: Extra arguments to makeTMatrix().
    ##
    ## Returns: A filled Tmatrix table.
    
    results = makeTMatrixTable(diams=diams, ratios=ratios,
        cantingSD=cantingSD, ...)
    results[, hres := as.numeric(NA)]
    results[, vres := as.numeric(NA)]
    
    for(i in seq(1, nrow(results))) {
        S = Tmatrix(AXI=results[i, axi], LAM=LAM, MRR=MRR, MRI=MRI,
            EPS=results[i, eps], ALPHA=results[i, alpha], BETA=results[i, beta],
            THET0=THET0, THET=THET, PHI0=PHI0, PHI=PHI, RAT=RAT, NP=NP,
            NDGS=NDGS, DDELT=DDELT)
        
        set(results, i, "hres", (4*pi*Mod(S[2,2])^2)/100)
        set(results, i, "vres", (4*pi*Mod(S[1,1])^2)/100)
    }

    results = TmatrixAverage(results, aveCols=c("hres", "vres"))
    setkey(results, axi)
    return(results)
}  

######################### SPECIFIC DIFFERENTIAL PHASE (Kdp) #########################

specificDiffPhaseFromDSD = function(DSD, classes, freq, temp, incidence,
    ratio_function=raindrop_axis_ratio, cantingSD=0, minAxisRatio=0.4,
    cacheDir=NULL, cacheID="") {
    ## Calculate specific differential phase shift (Kdp) from DSDs.
    ##
    ## Args:
    ##  DSD: Volumetric DSD data. Rows are measurements, columns are the diameter
    ##       classes. Entries in [m-3 mm-1].
    ##  classes: Diameter classes (col for mins, col for maxs) [mm].
    ##  freq: Radar frequency [GHz].
    ##  temp: Temperature [degrees C].
    ##  incidence: Radar incidence angle [degrees] (eg, 90 indicates a vertical
    ##             scan).
    ##  ratio_function: Raindrop axis ratio function (default:
    ##                  raindrop_axis_ratio())
    ##  cantingSD: Standard deviation of gaussian canting angle distribution.
    ##  minAxisRatio: Minimum raindrop axis ratio [v/h] to accept
    ##                (default: 0.4).
    ##  cacheDir: Directory in which to cache Tmatrix results
    ##            (default: NULL, no cache).
    ##  cacheID: ID to use for cache, e.g. axis ratio function
    ##           (default: none).
    ##
    ## Returns: Specific differential phase (Kdp) value [deg. km^-1] per DSD.

    ## Check dimension of DSD matrix:
    DSD = data.table(DSD)
    stopifnot(dim(DSD)[2] == length(classes[,1]))
    
    class_size = as.numeric(rowMeans(classes)) # Class centres [mm].
    class_spread = as.numeric(apply(classes, 1, diff)) # Class widths [mm].

    ## Set up axis ratios.
    axis_ratios = ratio_function(class_size)
    axis_ratios = resetAxisRatios(axis_ratios, DSD, minAxisRatio)
    
    index_water = ref_index_water(temp, freq)
    wavelength = wavelengthFromFreq(freq) # mm.

    ## Get forward scattering amplitudes.
    fAmps = forwardScatterAmpComplex(tabD=class_size,
        tab_ratio=axis_ratios, w=wavelength, m=index_water,
        theta=(90-incidence), phi=0, cantingSD=cantingSD,
        cacheDir=cacheDir, cacheID=cacheID)

    DSDmat = as.matrix(DSD)
    widthMat = matrix(rep(class_spread, nrow(DSD)), byrow=TRUE, nrow=nrow(DSD))

    ShhMinusSvv = Re(fAmps$FSAH - fAmps$FSAV) # Real part in mm
    ShhMinusSvvMat = matrix(rep(ShhMinusSvv, nrow(DSD)), byrow=TRUE, nrow=nrow(DSD))

    Ck = (180/pi)*wavelength*1e-3             # [°.m]
    Kdp_res = Ck*rowSums(DSD*ShhMinusSvvMat*widthMat, na.rm=FALSE)
    return(Kdp_res)
}

######################### SPECIFIC ATTENUATION (k) #########################

specificAttenuationFromDSD = function(DSD, classes, freq, temp, incidence,
    ratio_function=raindrop_axis_ratio, cantingSD=0, minAxisRatio=0.4,
    cacheDir=NULL, cacheID="") {
    ## Calculate specific attenuation [dB km-1] from the DSD.  Based
    ## on code by Joel Jaffrain, modified by Jacopo Grazioli, then by
    ## TR.
    ##
    ## Args:
    ##  DSD: Volumetric DSD data. Rows are measurements, columns are
    ##       the diameter classes. Entries in [m-3 mm-1].
    ##  classes: Diameter classes (column for mins, column for maxs) [mm].
    ##  freq: Radar frequency [GHz].
    ##  temp: Temperature [degrees C].
    ##  incidence: Radar incidence angle [degrees] (eg, 90 indicates a vertical
    ##             scan).
    ##  ratio_function: Raindrop axis ratio function (default:
    ##                  raindrop_axis_ratio())
    ##  cantingSD: Standard deviation of gaussian canting angle distribution.
    ##  minAxisRatio: Minimum raindrop axis ratio [v/h] to accept (default: 0.4).
    ##  cacheDir: Directory in which to cache Tmatrix results
    ##            (default: NULL, no cache).
    ##  cacheID: ID to use for cache, e.g. axis ratio function
    ##           (default: none).
    ##
    ## Returns: A data.table containing k [d km-1] for each DSD.
    ## Note: canting angle does not depend on particle size.

    DSD = data.table(DSD)
    stopifnot(dim(DSD)[2] == length(classes[,1]))

    class_size = as.numeric(rowMeans(classes)) # Class centres [mm].
    class_spread = as.numeric(apply(classes, 1, diff)) # Class widths [mm].

    ## Set up axis ratios.
    axis_ratios = ratio_function(class_size)
    axis_ratios = resetAxisRatios(axis_ratios, DSD, minAxisRatio)
  
    index_water = ref_index_water(temp, freq)
    wavelength = wavelengthFromFreq(freq)

    ## Get forward scattering amplitudes.
    fAmps = forwardScatterAmpComplex(tabD=class_size,
        tab_ratio=axis_ratios, w=wavelength, m=index_water,
        theta=(90-incidence), phi=0, cantingSD=cantingSD,
        cacheDir=cacheDir, cacheID=cacheID)

    ## Calculate extinction cross sections [cm2].
    ## To arrive at this expression, use (from Bringi 2001):
    ## - Eq. 1.129b/c:
    ##     (n*sigma_ext)/(2*k_0) = (-2*pi*n)/k_0^2 * Im(FS)
    ##  => sigma_ext = -4*pi/k_0 * Im(FS)
    ## - Page 8, k_0 = 2*pi/wavelength
    ##  => sigma_ext = -2*wavelength*Im(FS)
    ##
    ## Now we have wavelength and Im() in mm, so we have mm^2. To convert to
    ## cm^2, divide by 100. So we have
    ##    EH = (-2*wavelength*Im(fAmps$FSAH))/100
    ## => EH = -wavelength*Im(fAmps$FSAH)/50
    ##
    ## The negative sign is removed because the attenuation is given
    ## as a positive term as shown in the step from Eq. 1.132b to to
    ## 1.133a.
    EH = wavelength*Im(fAmps$FSAH)/50 ## cm^2
    EV = wavelength*Im(fAmps$FSAV)/50
    
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
    
    DSDmat = as.matrix(DSD)
    widthMat = matrix(rep(class_spread, nrow(DSD)), byrow=TRUE, nrow=nrow(DSD))
    EHMat = matrix(rep(EH, nrow(DSD)), byrow=TRUE, nrow=nrow(DSD))
    EVMat = matrix(rep(EV, nrow(DSD)), byrow=TRUE, nrow=nrow(DSD))

    kh = 10*log10(exp(1))*(rowSums((DSDmat * EHMat * widthMat), na.rm=FALSE) / 10)
    kv = 10*log10(exp(1))*(rowSums((DSDmat * EVMat * widthMat), na.rm=FALSE) / 10)

    out = data.frame(kh=as.numeric(kh), kv=as.numeric(kv))
    return(out)
}

forwardScatterAmpComplex = function(tabD, tab_ratio, w, m, theta, phi, 
    cantingSD=0, cacheDir=NULL, cacheID="") {
    ## Computes the forward scattering amplitudes (complex).
    ## 
    ## Args:
    ##   tabD: vector of equivolumetric drop diameters [mm]
    ##   tab_ratio: vector of drop axis ratios (vertical/horizontal) [-]
    ##   w: wavelength [mm]
    ##   m: complex refractivity index of water [-]
    ##   theta: zenith of incident beam [°] (90° = horizontal)
    ##   phi: azimuth of incident beam [°] (90° = along the y-axis)
    ##   cantingSD: standard deviations canting angles [deg].
    ##   cacheDir: If specified, look for precalculated caches in the directory.
    ##   cacheID: Special ID for the cache to use (default: none).
    ## 
    ## Returns: list of forward scattering amplitudes [complex with real
    ##          part in mm] per drop diameter.
    ##          FSCH: complex forward amplitudes for Hpol.
    ##          FSCV: complex forward amplitudes for Vpol
    ## 
    ## Note: the canting angle does not depend on particle size.
    ## 
    ## Source: Marc Schleiss, EPFL-LTE, July 2011
    ## Modified Jacopo Grazioli,EPFL-LTE,January 2012
    ## Modified Jacopo Grazioli, EPFL-LTE October 2014
    ##   parameter tab_beta is removed and it is substituted
    ##   by the standard deviation of a canting angle with mean.
    ## T. Raupach updated to use a cache for speed and improved
    ##   canting angle implementation, Nov 2017.
    
    ## Do input tests.
    checkTmatrixInputs(tabD=tabD, tab_ratio=tab_ratio, w=w, m=m, theta=theta,
                       phi=phi, cantingSD=cantingSD)
    
    ## Determine incidence and scattering angle.
    if(phi>180) {phi <- phi-180}
    PHI0  <- phi
    PHI   <- phi
    THET0 <- theta
    THET  <- theta

    ## Set wavelength, refractive index and reference angles.
    LAM   <- w
    MRR   <- Re(m)
    MRI   <- Im(m)

    ## Load cache.
    results = NULL
    if(!is.null(cacheDir)) {
        results = loadCache(cacheDir=cacheDir, w=w, m=m, theta=theta, phi=phi,
            cantingSD=cantingSD, tabD=tabD, ratios=tab_ratio,
            type="forwardAmplitude", cacheID=cacheID)
    }

    ## If no cache, calculate Tmatrix values.
    if(is.null(results)) {
        results = TMatrixForwardAmp(
            diams=tabD, ratios=tab_ratio, cantingSD=cantingSD,
            LAM=LAM, MRR=MRR, MRI=MRI, THET0=THET0, THET=THET,
            PHI0=PHI0, PHI=PHI, RAT=1, NP=-1, NDGS=2, DDELT=0.001)
    }

    ## Check ordering is correct.
    stopifnot(identical(results[, axi], tabD/2))
    stopifnot(identical(results[, eps], 1/tab_ratio))

    ## Convert real and imaginary parts to complex numbers.
    FSAH = results[, complex(real=hres_re, imaginary=hres_im)]
    FSAV = results[, complex(real=vres_re, imaginary=vres_im)]

    ## Save cache if required.
    if(!is.null(cacheDir)) {
        saveCache(cache=results, cacheDir=cacheDir, w=w, m=m, theta=theta,
                  phi=phi, cantingSD=cantingSD, type="forwardAmplitude",
                  cacheID=cacheID)
    }

    return(list(FSAH=FSAH, FSAV=FSAV))
}

TMatrixForwardAmp = function(diams, ratios, cantingSD,
    LAM, MRR, MRI, THET0, THET, PHI0, PHI, RAT, NP, NDGS, DDELT,
    ...) {
    ## Make a cache of forward-amplitude results for various diameters,
    ## ratio, alpha and beta angle combinations.
    ##
    ## Args:
    ##   diams: Diameters to use (not radii, conversion later!).
    ##   ratios: Ratio for each diameter (normal < 1, conversion later).
    ##   cantingSD: raindrop canting angle std. dev [deg].
    ##   LAM, MRR, etc: Tmatrix() arguments.
    ##   ...: Extra arguments to makeTMatrix().
    ##
    ## Returns: A filled Tmatrix cache, with real and imaginary parts
    ##          stored separately.

    results = makeTMatrixTable(diams=diams, ratios=ratios,
        cantingSD=cantingSD, ...)

    results[, hres_re := as.numeric(NA)]
    results[, hres_im := as.numeric(NA)]
    results[, vres_re := as.numeric(NA)]
    results[, vres_im := as.numeric(NA)]
    
    for(i in seq(1, nrow(results))) {
        S = Tmatrix(AXI=results[i, axi], LAM=LAM, MRR=MRR, MRI=MRI,
            EPS=results[i, eps], ALPHA=results[i, alpha], BETA=results[i, beta],
            THET0=THET0, THET=THET, PHI0=PHI0, PHI=PHI, RAT=RAT, NP=NP,
            NDGS=NDGS, DDELT=DDELT)
        i = as.integer(i)
        
        set(results, i, "hres_re", Re(S[2,2]))
        set(results, i, "hres_im", Im(S[2,2]))
        set(results, i, "vres_re", Re(S[1,1]))
        set(results, i, "vres_im", Im(S[1,1]))
    }

    results = TmatrixAverage(tMat=results,
        aveCols=c("hres_re", "hres_im", "vres_re", "vres_im"))
    setkey(results, axi)
    return(results)
}

######################### UTILITY FUNCTIONS #########################

wavelengthFromFreq = function(freq, cl=299792.458*1e3) {
    ## Calculate radar wavelength from radar frequency.
    ##
    ## Args:
    ##  freq: The frequency for which to calculate wavelength [GHz].
    ##  cl: The speed of light [m s-1].
    ##
    ## Returns: The wavelength [mm].

    return((cl/freq*1e-9)*1e3) 
}

checkTmatrixInputs = function(tabD, tab_ratio, w, m, theta, phi, cantingSD) {
    ## Jacopo's checks on Tmatrix inputs.
    ##
    ## Args: 
    ##   tabD: vector of equivolumetric drop diameters [mm]
    ##   tab_ratio: vector of drop axis ratios (vertical/horizontal) [-]
    ##   w: wavelength [mm]
    ##   m: complex refractivity index of water [-]
    ##   theta: zenith of incident beam [°] (90° = horizontal)
    ##   phi: azimuth of incident beam [°] (90° = along the y-axis)
    ##   cantingSD: standard deviation of raindrop canting angles [deg].
    ##   cacheDir: If specified, look for precalculated caches in the directory.
    ##
    ## Returns: void; stops if error is detected.
    
    if(any(is.na(tabD))) {stop("NA values not allowed in tabD")}
    if(any(is.na(tab_ratio))) {stop("NA values not allowed in tab_ratio")}
    if(is.na(w)) {stop("NA values not allowed for w")}
    if(is.na(m)) {stop("NA value not allowed for m")}
    if(is.na(theta)) {stop("NA value not allowed for theta")}
    if(is.na(phi)) {stop("NA value not allowed for phi")}
    if(w<=0) {stop("wavelength must be strictly positive")}
    if(any(tabD<=0)) {stop("drop diameters must be strictly positive")}
    if(any(tab_ratio<=0)) {stop("axis ratios must be strictly positive")}
    if(cantingSD<0) {stop("canting amplitude must be strictly positive")}
    if(is.na(cantingSD)) {stop("NA values not allowed in cantingSD")}
    if(length(tab_ratio)!=length(tabD)) {stop("tabD and tab_ratio must have same length")}
}

cacheFile = function(w, m, theta, phi, cantingSD, type, cacheID="", dp=5) {
    ## Make a coded cache file name, unique per cache.
    ##
    ## Args:
    ##   w: wavelength [mm]
    ##   m: complex refractivity index of water [-]
    ##   theta: zenith of incident beam [°] (90° = horizontal)
    ##   phi: azimuth of incident beam [°] (90° = along the y-axis)
    ##   cantingSD: standard deviation of the raindrop canting angles [deg].
    ##   type: Cache type (eg. "backscatter").
    ##   cacheID: Special ID for the cache (string, default none).
    ##   dp: Number of decimal places to use in the filename (default: 5).
    ##
    ## Returns: a unique filename for the cache.
    
    cacheFile = paste("Tmatrix_cache", type, cacheID,
        round(w, dp), round(m, dp),
        round(theta, dp), round(phi, dp),
        round(cantingSD, dp), "Rdata", sep=".")
    return(cacheFile)
}

saveCache = function(cache, cacheDir, w, m, theta, phi, cantingSD, type, cacheID, dp=5) {
    ## Save a cache with a unique filename.
    ##
    ## Args:
    ##   cache: The cache to save.
    ##   cacheDir: The directory to save in.
    ##   w: wavelength [mm]
    ##   m: complex refractivity index of water [-]
    ##   theta: zenith of incident beam [°] (90° = horizontal).
    ##   phi: azimuth of incident beam [°] (90° = along the y-axis).
    ##   cantingSD: standard deviation of the raindrop canting angles [deg].
    ##   type: Cache type (eg "backscatter").
    ##   cacheID: special ID to use for this cache (default:none).
    ##   dp: Number of decimal places to use in the filename (default: 4).
    ##
    ## Returns: NULL.
    
    save(cache, file=paste(cacheDir, cacheFile(w=w, m=m, theta=theta,
                    phi=phi, cantingSD=cantingSD, type=type,
                    cacheID=cacheID, dp=dp), sep="/"))
}

loadCache = function(cacheDir, w, m, theta, phi, cantingSD, type,
    tabD, ratios, cacheID="", dp=5) {
    ## Load a cache from a unique filename.
    ##
    ## Args:
    ##   cache: The cache to save.
    ##   cacheDir: The directory to save in.
    ##   w: wavelength [mm]
    ##   m: complex refractivity index of water [-]
    ##   theta: zenith of incident beam [°] (90° = horizontal)
    ##   phi: azimuth of incident beam [°] (90° = along the y-axis)
    ##   cantingSD: standard deviation of the raindrop canting angles [deg].
    ##   type: The type of cache (e.g. "backscatter").
    ##   tabD: The class-center diameters expected in the cache.
    ##   ratios: The drop axis ratios per class expected in the cache.
    ##   cacheID: Special id string for cache filename (default: none).
    ##   dp: Number of decimal places to use in the filename (default: 4).
    ##
    ## Returns: The cache data, or NULL if no cache is found.
    
    filename=paste(cacheDir, cacheFile(w=w, m=m, theta=theta,
        phi=phi, cantingSD=cantingSD, type=type, cacheID=cacheID,
        dp=dp), sep="/")
    
    if(!file.exists(filename))
        return(NULL)
    
    cacheResults = get(load(filename))

    ## If this cache does not correspond, do not use it! 
    if(!identical(cacheResults$axi, tabD/2) ||
       !identical(cacheResults$eps, 1/ratios)) {
        warning(paste("Cache exists but does not correspond",
                      "to requested diameters or axis ratios;",
                      "recalculating tMatrix values."))
        return(NULL)
    }

    return(cacheResults)
}    

makeTMatrixTable = function(diams, ratios, cantingSD, numAlpha=5, numBeta=10) {
    ## Make an empty data.table for Tmatrix results, with input combinations
    ## that need to be evaluated.
    ##
    ## Args:
    ##   diams: The diameters to include.
    ##   ratios: The ratios to include.
    ##   cantingSD: The std. dev. of the canting angle distribution (normal with
    ##              mean zero) [deg].
    ##   numAlpha: Number of rotation angle samples to take (default: 5).
    ##   numBeta: The number of canting angle samples to take (default: 10).
    ##
    ## Returns: an empty Tmatrix data.table.

    ## Alpha angles are rotations around the z axis; then the raindrop
    ## is tilted by beta degrees around the "new" y axis. See
    ## Mischenko 2002, pp 42-43 and Figure 2.2.

    ## Think of beta as canting angle, alpha as rotation angle.
    ## We use mean of various rotation angles with equal weight.
    ## We use mean of various canting angles weighted by Gaussian
    ## distribution with mean zero and standard deviation cantingSD
    ## degrees.

    if(cantingSD == 0) {
        angles = data.table(alpha=0, beta=0, betaWeight=1, aveByAlpha=0)
    } else {
        ## Rotation angles to evaluate.
        alphaAngles = seq(0, 360, length.out=numAlpha+1)[1:numAlpha]
        
        ## Canting angles to evaluate, and their weights for quadrature.
        quad = get_points_and_weights(cantingSD=cantingSD, left=0, right=180, num_points=numBeta)
        betaAngles = quad$points
        betaWeights = quad$weights
        
        ## Combine angles.
        angles = expand.grid(alphaAngles, betaAngles)
        names(angles) = c("alpha", "beta")
        angles = data.table(angles)
        setkey(angles, beta)
        weights = data.table(beta=betaAngles, weight=betaWeights, key="beta")
        angles[, betaWeight := weights[angles, weight]]
    }
    
    tMatTable = NULL
    for(i in seq(1, length(diams))) {
        ## Convert diameters to radii and ratios to 1/ratio.
        tMatTable = rbind(tMatTable,
            data.table(axi=(diams/2)[i], eps=(1/ratios)[i], angles))
    }
    
    return(tMatTable)
} 

resetAxisRatios = function(axis_ratios, DSD, minAxisRatio) {
    ## TMatrix crashes when drop axis ratios are too small.
    ## Reset small axis ratios, as long as they don't affect
    ## bins that contain drops in the DSD.
    ##
    ## Args:
    ##   axis_ratios: Axis ratios per DSD class.
    ##   DSD: DSDs with rows as measurements, cols as diameter classes.
    ##   minAxisRatio: The minimum axis ratio to accept.
    ##
    ## Returns: Axis ratios with those smaller than minAxisRatio
    ##          replaced by minAxisRatio.
    
    if(length(intersect(which(axis_ratios < minAxisRatio), 
                        which(colSums(DSD) > 0)) != 0))
        stop("Error: axis ratio setting for Tmatrix will affect DSD.")
    axis_ratios[which(axis_ratios < minAxisRatio)] = minAxisRatio
    
    return(axis_ratios)
}

TmatrixAverage = function(tMat, aveCols=c("hres", "vres")) {
    ## Take Tmatrix results for different alpha and beta angles and find 
    ## the integrated result quadrature for the canting angles and
    ## ordinary mean for the rotation angles.
    ##
    ## Args:
    ##  tMat: Tmatrix results to average.
    ##  aveCols: The columns to take the averages of (default: hres, vres).
    ##
    ## Returns: The average results per value of "axi" (ie drop radius).
    
    ## Alpha angles are rotations around the z axis.
    ## Beta angles are canting angles for the raindrop.
    ##
    ## The integrated result is the mean of different beta results,
    ## weighted by the Gaussian distribution of beta angles.
    
    stopifnot(all(c("axi", "eps", "alpha", "beta", "betaWeight", aveCols) %in% names(tMat)))
    res = tMat[, gaussianAverage(.SD, colNames=aveCols, byVal="alpha"),
        by=c("axi", "eps")]
    return(res)
}

gaussianAverage = function(dat, colNames, byVal) {
    ## For each value of "byVal", find the mean weighted by Gaussian
    ## distribution; then take the mean of all results.
    ##
    ## Args:
    ##   dat: The data.table to work on.
    ##   colNames: The column names to find averages of.
    ##   byVal: The value to sort by for Gaussian weighted means.
    ##
    ## Returns: data.table with resulting means (one value per
    ## colName).
    ##
    ## For the source of this equation, see
    ## https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature
    ## Note that here the sqrt(2) is replaced by the sum of all weights.
  
    return(dat[, lapply(1/sum(betaWeight)*betaWeight*.SD, sum),
                 .SDcols=colNames, by=byVal][, lapply(.SD, mean),
                                       .SDcols=colNames])
}


############################################################################

discrete_gautschi = function(z, w, n_iter) {
    ## Translated from Tmatrix python library.
    
    ## Note, here I assume z and w are one-dim vectors and use
    ## sum(a*b) to calculate the inner product. Ensure this is true.
    stopifnot(is.vector(z))
    stopifnot(is.vector(w))
    stopifnot(identical(length(z), length(w)))
    
    p = rep(1, length(z))
    p = p / sqrt(sum(p * p))
    p_prev = rep(0, length(z))
    wz = z*w
    a = rep(NA, n_iter)
    b = rep(NA, n_iter)
    
    for(j in seq(1, n_iter)) {
        p_norm = sum((w*p) * p)
        a[j] = sum((wz*p) * p)/p_norm
        if(j == 1) {
            b[j] = 0
        } else {
            b[j] = p_norm/sum((w*p_prev)*p_prev)
        }
        
        p_new = (z-a[j])*p - b[j]*p_prev 
        p_prev = p
        p_prev_norm = p_norm
        p = p_new
    }
    
    return(list(a=a, b=b[-1]))
}

get_points_and_weights = function(cantingSD, left=0, right=180,
    num_points=10, n=4096) {
    ## Quadratude points and weights for a weighting function.
    ##
    ## Translated from Tmatrix library:
    ## https://github.com/jleinonen/pytmatrix/blob/master/ \
    ## pytmatrix/quadrature/quadrature.py
    ## 
    ## Points and weights for approximating the integral 
    ##   I = \int_left^right f(x) w(x) dx
    ## given the weighting function w(x) using the approximation
    ##   I ~ w_i f(x_i)
    ##
    ## In this translation f(x) is hard coded as a normal
    ## distribution.
    ## 
    ## Args:
    ##   cantingSD: The standard deviation of canting angles.
    ##   left: The left boundary of the interval (0 deg).
    ##   right: The left boundary of the interval (180 deg).
    ##   num_points: number of integration points to return.
    ##   n: the number of points to evaluate w_func at.
    ## 
    ## Returns: A tuple (points, weights) where points is a sorted
    ##          array of the points x_i and weights gives the
    ##          corresponding weights w_i.

    ## Define a PDF of a normal distribution with mean 0 and standard
    ## deviation cantingSD. Translated from Tmatrix library:
    ## https://github.com/jleinonen/pytmatrix/blob/master/ \
    ## pytmatrix/orientation.py
    norm_const = 1
    w_func = function(x) {
        return(norm_const*exp(-0.5 * (x/cantingSD)**2) * sin(pi/180 * x))
    }
    
    norm_dev = integrate(w_func, lower=left, upper=right)$value
    norm_const = norm_const / norm_dev 

    ## Check that the PDF integrates to 1. 
    stopifnot(abs(integrate(w_func, lower=left, upper=right)$value - 1) < 1e-10)
    
    dx = (right-left)/n
    z = seq(left+0.5*dx, right-0.5*dx, length.out=n)
    w = dx*w_func(z)  
    
    gautschi = discrete_gautschi(z, w, num_points)
    alpha = gautschi$a
    beta = sqrt(gautschi$b)
    
    J = diag(alpha)
    J[matrix(c(seq(1, length(alpha)), seq(1, length(alpha))+1), ncol=2)[1:(length(alpha)-1),]] = beta
    J[matrix(c(seq(1, length(alpha))+1, seq(1, length(alpha))), ncol=2)[1:(length(alpha)-1),]] = beta
    
    eigh = eigen(J)
    points = eigh$values
    idx = order(points)
    points = points[idx]
    weights = eigh$vectors[1,]^2 * sum(w)
    weights = weights[idx]
    
    ## (points,v) = np.linalg.eigh(J)
    ## ind = points.argsort()
    ## points = points[ind]
    ## weights = v[0,:]**2 * w.sum()
    ## weights = weights[ind]
    
    return(list(points=points, weights=weights))
}
