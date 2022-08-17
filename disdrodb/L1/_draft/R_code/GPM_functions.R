## GPM_functions.R
##
## Functions to simulate GPM-style retrieval of the DSD, from radar
## reflectivities.
##
## Author: Tim Raupach <tim.raupach@epfl.ch>

normalisedReflectivity = function(DmVals, freq, temp,
    D=rowMeans(get.classD()), widths=apply(get.classD(), 1, diff),
    mu=3, ratio_function=raindrop_axis_ratio,
    incidence=90, sigmaB=NULL) {
    ## Calculate Nw normalised reflectivity lookup tables for
    ## mass-weighted mean diameters 'DmVals' [mm] and frequency 'freq'
    ## [GHz]. Based on I_b in Liao et al 2014 (their Equation 9).
    ##
    ## Args:
    ##   DmVals: The Dm values for which to calculate normalised
    ##           reflectivities [mm].
    ##   freq: Radar frequency to consider.
    ##   temp: Air temperature [deg. C].
    ##   D: Drop diameter classes [mm].
    ##   widths: Drop diameter class widths [mm].
    ##   mu: mu (shape) parameter for DSD model [-].
    ##   ratio_function: Function to give axis ratio for drop size.
    ##   incidence: Radar incidence, 90 = vertical.
    ##   sigmaB: Back-scattering cross sections, if cached [cm2].
    ##
    ## Returns: the normalised reflectivity [dB] for each value of Dm.

    # Get wavelength.
    cL = 299792.458*1e3              # Speed of light [m/s].
    wavelength = (cL/freq*1e-9)*1e3  # Wavelength [mm].

    # Get electric factor Kw [-].
    index_water = ref_index_water(temp, freq)
    Kw = (index_water^2-1) / (index_water^2+2)

    ## Back scattering cross sections [cm2].
    if(is.null(sigmaB)) {
        ## Get drop axis ratios [vert/horiz].
        axis_ratios = ratio_function(D)
        axis_ratios[which(axis_ratios < 0)] = 0.6

        sigmaB = back_scat_cross(D, axis_ratios, wavelength,
            index_water, (90-incidence), 0)[,1]
    } else {
        print("Warning: using precompiled sigmaB values.")
    }

    ## For the factor, keep wavelength as mm.
    factor = (wavelength)^4 / (pi^5*abs(Kw)^2) # [mm^4]

    ## f_mu function [-].
    fMu = (6*(4 + mu)^(mu+4))/(4^4*gamma(mu+4))

    ## Note that 1e2*sigmaB is in mm2, widths is in mm,
    ## D is in mm, D/Dm is [-], fMu is [-], so the result
    ## is in mm7, but we assume N_w = 1 mm-1 m-3, so
    ## final result is in mm6 m-3.
    Dm = data.table(Dm=DmVals, mu=mu, fMu=fMu)
    Dm = Dm[, lambda := (4+mu)/Dm]

    func = function(Dm, lambda, mu, fMu) {
        return(10*log10(factor * sum(fMu * (D/Dm)^mu * exp(-lambda*D) *
                                     (1e2*sigmaB) * widths)))
    }

    ## UNITS (in equation order):
    ## factor [mm4]
    ## fMu [-]
    ## D [mm]
    ## Dm [mm]
    ## D/Dm [-]
    ## lambda [mm-1]
    ## lambda*D [-]
    ## 1e2 sigmaB [mm2]
    ## widths [mm]
    ## Thus sum [mm3]
    ## sum * factor [mm7]
    ## sum * factor * Nw = 1 [mm-1 m-3] = mm6 m-3

    lookup = Dm[, list(Dm=Dm, Ib=func(Dm, lambda, mu, fMu)), by=1:length(DmVals)]
    return(lookup)
}

normalisedAttenuation = function(DmVals, freq, temp,
    D=rowMeans(get.classD()), widths=apply(get.classD(), 1, diff),
    mu=3, ratio_function=raindrop_axis_ratio,
    incidence=90, sigmaE=NULL) {
    ## Calculate Nw normalised specific attenuation lookup tables for
    ## mass-weighted mean diameters 'DmVals' [mm] and frequency 'freq'
    ## [GHz]. Based on I_e in Liao et al 2014 (their Equation 10).
    ##
    ## Args:
    ##   DmVals: The Dm values for which to calculate normalised
    ##           reflectivities [mm].
    ##   freq: Radar frequency to consider.
    ##   temp: Air temperature [deg. C].
    ##   D: Drop diameter classes [mm].
    ##   widths: Drop diameter class widths [mm].
    ##   mu: mu (shape) parameter for DSD model [-].
    ##   ratio_function: Function to give axis ratio for drop size.
    ##   incidence: Radar incidence, 90 = vertical.
    ##   sigmaE: Back-scattering extinctions, if cached [cm2].
    ##
    ## Returns: the normalised attenuation for each value of Dm.
    
    # Get wavelength.
    cL = 299792.458*1e3              # Speed of light [m/s].
    wavelength = (cL/freq*1e-9)*1e3  # Wavelength [mm].

    # Get electric factor Kw [-].
    index_water = ref_index_water(temp, freq)

    ## Extinction scattering cross sections [cm2].
    if(is.null(sigmaE)) {
        ## Get drop axis ratios [vert/horiz].
        axis_ratios = ratio_function(D)
        axis_ratios[which(axis_ratios < 0)] = 0.6

        sigmaE = ext_scat_cross(D, axis_ratios, wavelength,
            index_water, (90-incidence), 0)[,1] ## Using horizontal; at 90 degrees
                                                ## will be equal to vertical pol.
    } else {
        print("Warning: using precompiled sigmaE values.")
    }

    ## f_mu function [-].
    fMu = (6*(4 + mu)^(mu+4))/(4^4*gamma(mu+4))
    Dm = data.table(Dm=DmVals, mu=mu, fMu=fMu)
    Dm = Dm[, lambda := (4+mu)/Dm]

    func = function(Dm, lambda, mu, fMu) {
        return(log10(exp(1)) * (sum(fMu * (D/Dm)^mu * exp(-lambda*D) *
                                     (sigmaE) * widths)))
    }

    ## UNITS (in equation order):
    ## fMu [-]
    ## D [mm]
    ## Dm [mm]
    ## D/Dm [-]
    ## lambda [mm-1]
    ## lambda*D [-]
    ## sigmaE [cm2]
    ## widths [mm]
    ## Thus sum is in [cm2 mm]
    ## Sum multiplied by N_w = 1 [mm-1 m-3] gives [m-3 cm2] = 10.[km-1]
    ## So sum/10 is in km-1
    ## Multiply (sim/10) by (10log10(e)) to get dB km-1.
    ## So sum*log10(exp(1)) is in dB km-1.

    lookup = Dm[, list(Dm=Dm, Ie=func(Dm, lambda, mu, fMu)), by=1:length(DmVals)]
    return(lookup)
}

GPMLookupTable = function(DmVals=seq(0.001, 7, by=0.001),
    KuFreq=13.6, KaFreq=35.55, ...) {
    ## Calculate a GPM lookup table that determines Dm from either DFR
    ## or k/Z.
    ##
    ## Args:
    ##   DmVals: The values of Dm to include in the table [mm].
    ##   KuFreq, KaFreq: Frequences of Ku and Ka-band [GHz].
    ##   ...: Arguments to normalisedReflectivity(),
    ##        normalisedAttenuation().
    
    ## Find values of I_b for a range of D_m values.
    dualLookupKa = normalisedReflectivity(DmVals=DmVals, freq=KaFreq, ...)
    dualLookupKu = normalisedReflectivity(DmVals=DmVals, freq=KuFreq, ...)
    
    ## Find values of I_e for a range of D_m values.
    singleLookupKa = normalisedAttenuation(DmVals=DmVals, freq=KaFreq, ...)
    singleLookupKu = normalisedAttenuation(DmVals=DmVals, freq=KuFreq, ...)
    
    ## Create lookup table for all values.
    lookupTable = data.table(Dm=dualLookupKu$Dm,
        Ib_Ku=dualLookupKu$Ib, Ib_Ka=dualLookupKa$Ib,
        Ie_Ku=singleLookupKu$Ie, Ie_Ka=singleLookupKa$Ie)
    
    ## Determine DFR from Ib in Ku and Ka band (subtraction since both are in dB).
    lookupTable = lookupTable[, DFR := Ib_Ku - Ib_Ka]
    
    ## Find Ie / Ib per wavelength. Ie is in decibels km-1, Ib is in dB.
    lookupTable = lookupTable[, Ie_Ib_Ku := Ie_Ku / (10^(Ib_Ku/10))]
    lookupTable = lookupTable[, Ie_Ib_Ka := Ie_Ka / (10^(Ib_Ka/10))]

    return(lookupTable)
}

radarRef = function(dsdMat, D=rowMeans(get.classD()),
        widths=apply(get.classD(), 1, diff)) {
    ## Calculate Z (6th-moment, linear) for a DSD.
    ##
    ## Args:
    ##   dsdMat: DSD to find Z for [m-3 mm-1].
    ##   D: Class-center diameters [mm].
    ##   widths: Class widths [mm].
    ##
    ## Returns: Z for the DSD [m-3 mm^6].

    ## Multiply each row by the class widths.
    mult = function(x) { return(x*widths*D^6) } 
    dsdMult = t(apply(dsdMat, 1, mult)) # [m^-3 (mm^-1 * mm) mm^6] = [m^-3 mm^6].
    return(rowSums(dsdMult))
}

addGPMVars = function(GPM, temp, incidence=90, KuFreq=13.6, KaFreq=35.55,
    dsdCols=paste("class", seq(1,32), sep="")) {
    ## To a set of DSDs, add GPM variables.
    ##
    ## Args:
    ##   GPM: DSDs to add variables to.
    ##   temp: Temperature (deg. C).
    ##   incidence: Beam incidence (90 = vertical).
    ##   KuFrequ, KaFreq: Radar frequences for Ku and Ka bands.
    ##
    ## Returns: the same data set, with 6th moment reflectivity from
    ##          the DSD (simZ [m^-3 mm^6]),
    ##          Ku reflectivity (Z_Ku [dBZ] and Z_Ku_lin [m^-3 mm^6]),
    ##          Ka reflectivity (Z_Ka [dBZ] and Z_Ka_lin [m^-3 mm^6]),
    ##          and DFR [dB] added.
    
    GPM = GPM[, k_Ku := specificAttenuationFromDSD(.SD, classes=get.classD(),
                         temp=temp, freq=KuFreq,
                         incidence=incidence)[,1], .SDcols=dsdCols]
    GPM = GPM[, k_Ka := specificAttenuationFromDSD(.SD, classes=get.classD(),
                         temp=temp, freq=KaFreq,
                         incidence=incidence)[,1], .SDcols=dsdCols]
    GPM = GPM[, simZ := radarRef(.SD), .SDcols=dsdCols]
    
    ## Z_Ku and Z_Ka are given in dBZ.
    GPM = GPM[, list(POSIXtime, simNumber, Z_Ku, Z_Ka,
        k_Ku, k_Ka, simDm=Dm, simR=R, simW=LWC, simNt=Nt, simZ)]
    GPM = GPM[, Z_Ku_lin := 10^(Z_Ku/10)]
    GPM = GPM[, Z_Ka_lin := 10^(Z_Ka/10)]
    GPM = GPM[, DFR := 10*log10(Z_Ku_lin/Z_Ka_lin)]
    
    return(GPM)
}

trimGPM = function(GPM, lookupTable) {
    ## Trim records to those with DFR within range of a lookup table.
    ##
    ## Args:
    ##   GPM: Records with GPM variables added (use addGPMVars()).
    ##   lookupTable: GPM lookup table to use.
    ##
    ## Note: records with Dm < 1.02 are removed; those with Z_Ku < 18
    ## or Z_Ka < 12 are removed (sensitivies of the GPM DPR).
    ## 
    ## Returns: list containing data.table trimmed to only valid records (GPM),
    ##          plus stats on those records removed.
    
    GPM = GPM[!is.na(DFR)]
    
    minDFR = lookupTable[, min(DFR)]
    maxDFR = lookupTable[, max(DFR)]
    minKZ = min(lookupTable$Ie_Ib_Ku)
    maxKZ = max(lookupTable$Ie_Ib_Ku)

    # First remove Dm < 1.02.
    totalRows = nrow(GPM)
    removedDueDm = round(GPM[, length(which(simDm < 1.02))] / totalRows * 100, 1)
    GPM = GPM[simDm >= 1.02]

    # Now remove outside GPM DPR detection limit.
    removedDueDetectionLimit = round(GPM[, length(which(Z_Ku < 18 | Z_Ka < 12))] / 
        totalRows * 100, 1)
    GPM = GPM[Z_Ku > 18 & Z_Ka > 12]
    
    ## ## # Now check single-frequency limits in lookup table.
    ## removedDuekZ = round(GPM[, length(which(k_Ku/Z_Ku_lin < minKZ | k_Ku/Z_Ku_lin > maxKZ))] / 
    ##     totalRows * 100, 1)    
    ## ## GPM = GPM[k_Ku/Z_Ku_lin >= minKZ | k_Ku/Z_Ku_lin <= maxKZ]

    ## ## Now check DFR limits in lookup table.
    ## removedDueDFR = round(GPM[, length(which(DFR < minDFR-.5 | DFR > maxDFR+.5))] / 
    ##     totalRows * 100, 1)

    removedPercGPM = removedDueDm + removedDueDetectionLimit
    
    ## maxDFRDiffBelow = round(GPM[DFR <
    ##     min(lookupTable$DFR), max(abs(DFR - min(lookupTable$DFR)))], 1)
    ## maxDFRDiffAbove = round(GPM[DFR >
    ##     max(lookupTable$DFR), max(abs(DFR - max(lookupTable$DFR)))], 1)
    
    return(list(GPM=GPM,
                stats=list(removedPercGPM=removedPercGPM,
                    #removedDueDFR=removedDueDFR,
                    #removedDuekZ=removedDuekZ,
                    removedDueDm=removedDueDm,
                    removedDueDetectionLimit=removedDueDetectionLimit)))
                    #maxDFRDiffBelow=maxDFRDiffBelow,
                    #maxDFRDiffAbove=maxDFRDiffAbove,
                    #minDFR=minDFR,
                    #maxDFR=maxDFR)))
}

DmForDFR = function(DFR, lookup) {
    ## Return a Dm from a lookup table for each DFR.
    ##
    ## Args:
    ##   DFR: Values of DFR to look up.
    ##   lookup: The lookup table to use.
    ##
    ## Returns: Dm corresponding to each DFR.
    
    DFR = as.matrix(DFR, ncol=1)
    lookupDFRs = matrix(lookup$DFR, ncol=1)
    lookupDms = matrix(lookup$Dm, ncol=1)
    closestIdxs = nn2(data=lookupDFRs, query=DFR, k=1)$nn.idx
    return(lookupDms[closestIdxs])
}
 
DmForKonZ = function(KonZ, lookup=lookupTable, col="Ie_Ib_Ku") {
    ## Return a Dm from a lookup table for each k/Z.
    ##
    ## Args:
    ##   KonZ: Values of k/Z to look up.
    ##   lookup: The lookup table to use.
    ##   cols: Which column in the lookup table corresponds to k/Z?
    ##
    ## Returns: Dm corresponding to each DFR.   
    
    ## K on Z should be K [db km-1] divided by linear Z [mm6 m-3].
    KonZ = as.matrix(KonZ, ncol=1)
    lookupKonZ = as.matrix(lookup[, col, with=FALSE], ncol=1)
    lookupDms = matrix(lookup[, Dm], ncol=1)
    closestIdxs = nn2(data=lookupKonZ, query=KonZ, k=1)$nn.idx
    return(lookupDms[closestIdxs])
}

calcGPMDm = function(GPM, lookupTable, minDFRDm=1.02) {
    ## Use a lookup table to calculate Dm for each value of DFR in
    ## a dataset.
    ##
    ## Args:
    ##   GPM: data.table containing DFR.
    ##   lookupTable: The lookup table to use.
    ##   minDFRDm: For DFR lookup, only look for Dms greater than
    ##             this value [mm]. This is to avoid finding two
    ##             solutions in the DFR lookup.
    ##
    ## Returns: the same data table with lookup Dm added as

    GPM = GPM[, gpmDm_DFR := DmForDFR(DFR, lookup=lookupTable[Dm >= minDFRDm])]
    GPM = GPM[, gpmDm_kZ := DmForKonZ(k_Ku/Z_Ku_lin)]
    return(GPM)
}

normalisedDSD = function(Nw, Dm, fMuVal=fMu, mu=3, D=rowMeans(get.classD())) {
    ## The normalised DSD model used by GPM.
    ##
    ## Args:
    ##  Nw: Scaling factor [mm-1 m-3].
    ##  Dm: Mass-weighted mean drop diameter [mm].
    ##  fMu: Value of fMu (6*(4+mu)^(mu+4))/(4^4*gamma(mu+4)).
    ##  mu: DSD shape parameter [-].
    ##  D: Class center diameters.
    ##
    ## Returns: DSD values for each class diameter.
    
    res = data.frame(t(Nw * fMuVal * (D/Dm)^mu * exp(-(4+mu)*D/Dm)))
    return(res)
}

GPMAddBulkVars = function(GPM,
    altitude=270, latitude=44.56826,
    dsdCols=paste("class", seq(1,32), sep="")) {
    ## Calculate bulk variables as GPM would calculate them, using
    ## Parsivel classes and truncating DSDs to Parsivel drop size
    ## limits.
    ##
    ## Args:
    ##   GPM: data.table containing Nw_DFR, Nw_kZ, gpmDm_DFR, gpmDm_kZ.
    ##   mu: Value(s) of shape parameter (mu) to use [-] (GPM uses 3).
    ##   
    ## Returns: data.frame with Nt, R, Z added for GPM and k/Z methods.
    
    ## Columns for the resulting DSDs.
    dsdCols_DFR = paste(dsdCols, "_DFR", sep="")
    dsdCols_kZ = paste(dsdCols, "_kZ", sep="")
    
    ## Add drop classes as seen by GPM. Inefficient but it works!
    stopifnot("mu" %in% names(GPM))
    stopifnot(!("mu" %in% ls()))
    GPM[, fMu := (6*(4+mu)^(mu+4))/(4^4*gamma(mu+4))]
    GPM[, (dsdCols_DFR) := normalisedDSD(Nw_DFR, gpmDm_DFR, fMu, mu), by="POSIXtime,simNumber"]
    GPM[, (dsdCols_kZ) := normalisedDSD(Nw_kZ, gpmDm_kZ, fMu, mu), by="POSIXtime,simNumber"]

    ## Truncate DSDs as per Parsivel DSDs.
    GPM[, (dsdCols_DFR[1:2]) := 0]
    GPM[, (dsdCols_DFR[23:32]) := 0]
    GPM[, (dsdCols_kZ[1:2]) := 0]
    GPM[, (dsdCols_kZ[23:32]) := 0]

    ## Add in Nt.
    GPM[, gpmNt_DFR := totalDropConcentration(dsdMat=GPM[, dsdCols_DFR, with=FALSE])]
    GPM[, gpmNt_kZ := totalDropConcentration(dsdMat=GPM[, dsdCols_kZ, with=FALSE])]

    ## Rain rate.
    GPM[, gpmR_DFR := DSDRainrate(spectra=.SD, altitude=altitude, latitude=latitude),
        .SDcols=dsdCols_DFR]
    GPM[, gpmR_kZ := DSDRainrate(spectra=GPM[, dsdCols_kZ, with=FALSE], 
                      altitude=altitude, latitude=latitude)]

    ## Linear 6th-moment radar reflectivity.
    GPM = GPM[, gpmZ_DFR := radarRef(.SD), .SDcols=dsdCols_DFR]
    GPM = GPM[, gpmZ_kZ := radarRef(.SD), .SDcols=dsdCols_kZ]
    
    return(GPM)
}

GPMAddNw = function(GPM, D=rowMeans(get.classD()),
    widths=apply(get.classD(), 1, diff), temp=12.8, incidence=90,
    cL=299792.458*1e3, axis_ratios=raindrop_axis_ratio(D),
    waterDensity=1, KuFreq=13.6, KaFreq=35.55, ...) {

    ## Calculate values for Nw [m-3 mm-1], the scaling normalisation
    ## factor used in the normalised DSD model.
    ##
    ## Args:
    ##   GPM: data.table containing DSDs.
    ##   D: drop diameter class centres [mm].
    ##   widths: drop diameter class widths [mm].
    ##   temp: Temperature [deg. C].
    ##   incidence: Beam incidence (90 = vertical).
    ##   cL: Speed of light m/s.
    ##   axis_ratios: Axis ratio for each diameter in D.
    ##   waterDensity: Water density [g cm-3]
    ##   KuFreq, KaFreq: Radar frequencies [GHz].
    ##
    ## Returns: the data.table with Nw added.
    
    ## Precalculate the back-scattering cross-sections for drop sizes.
    ## Back scattering cross sections [cm2] - Ku band.
    axis_ratios[which(axis_ratios < 0)] = 0.6
    index_water = ref_index_water(temp, KuFreq)
    sigmaB_Ku = back_scat_cross(D, axis_ratios, w=(cL/KuFreq*1e-9)*1e3,
        index_water, (90-incidence), 0)[,1]

    ## Calculate I_b (Nw-normalised reflectivity) for each looked-up
    ## Dm value.
    stopifnot("mu" %in% names(GPM))
    GPM = GPM[, Ib_Ku_DFR := normalisedReflectivity(DmVals=gpmDm_DFR,
                              freq=KuFreq, sigmaB=sigmaB_Ku, D=D, widths=widths,
                              mu=mu, ...)$Ib]
    GPM = GPM[, Ib_Ku_kZ := normalisedReflectivity(DmVals=gpmDm_kZ,
                             freq=KuFreq, sigmaB=sigmaB_Ku, D=D, widths=widths,
                             mu=mu, ...)$Ib]

    ## Z = 10log10(Nw) + Ib for each row. So Nw = 10^((Z-Ib)/10)
    ## Z and Ib must be in dBZ.
    GPM = GPM[, Nw_DFR := 10^((Z_Ku - Ib_Ku_DFR)/10)]
    GPM = GPM[, Nw_kZ := 10^((Z_Ku - Ib_Ku_kZ)/10)]
    GPM = GPM[, simNw := (4^4)/(pi*waterDensity) * (10^3*simW/simDm^4)]
   
    return(GPM)
}

processAsGPM = function(data, lookupTable, altitude=275,
    latitude=44.56826, temp=12.8, muValue=3, ...) {
    ## Process a set of DSD data, to produce the way in which it would
    ## appear to the GPM algorithms. Note altitude/temp/latitude for HyMeX
    ## field site.
    ## 
    ## Args:
    ##   data: data.table containing all DSD data.
    ##   lookupTable: The DFR and k/Z to Dm lookup table to use.
    ##   muValue: Value of mu to use (should match lookup table, default = 3).
    ##   ...: Extra arguments (mu, Nw, etc).
    ## 
    ## Returns: data.table with all GPM variables calculated.
    
    ## Add GPM radar variables to the DSDs.
    GPM = addGPMVars(data, temp=temp)
    
    ## Trim the data to the lookup table values.
    trimmed = trimGPM(GPM, lookupTable)
    GPM = trimmed$GPM
    remStats = trimmed$stats
    rm(list="trimmed")
    
    ## Lookup Dm values based on DFR and k/Z.
    GPM = calcGPMDm(GPM, lookupTable)

    ## Add Nw.
    waterDensity = waterDensity(altitude=altitude, temperature=temp,
        latitude=latitude) * 1e3 # Water density [g cm^-3].

    GPM[, mu := muValue] ## Default value of mu for GPM.
    GPM = GPMAddNw(GPM, temp=temp, waterDensity=waterDensity)
    
    ## Add "regular" bulk variables.
    GPM = GPMAddBulkVars(GPM)
    
    return(list(GPM=GPM, remStats=remStats))
}

densityPlotComparison = function(dat, xVal, yVal, xLab, yLab,
    plot1Lab, plot2Lab, unit="mm") {
    ## Make two density plots for two methods. Stats in 'dat' must
    ## have a column named "diff" containing differences, 'relDiff'
    ## containing relative diffs, and "method".
    ##
    ## Args:
    ##  dat: Data to plot.
    ##  xVal: X axis column name.
    ##  yVal: Y axis column name.
    ##  xLab, yLab: Labels for columns.
    ##  plot1Lab, plot2Lab: Labels for plots.
    ##  unit: Unit to show for bias.
    ##
    ## Returns: void, plots are output to screen/file.
    
    stats = copy(dat)
    setnames(stats, xVal, "x")
    setnames(stats, yVal, "y")

    yMax = stats[, max(y)]

    meanErr = stats[, list(lab=paste("Bias:", round(mean(diff), 2),
                               unit, sep="~"), pos=yMax*0.95), by=method]
    medianRelErr = stats[, list(lab=paste("Median~rel.~diff.:",
                                    round(median(relDiff), 2),
                                    "symbol('\045')", sep="~"),
        pos=yMax*0.88), by=method]
    meanRelErr = stats[, list(lab=paste("Mean~rel.~diff.:",
                                  round(mean(relDiff), 2),
                                  "symbol('\045')", sep="~"),
        pos=yMax*0.81), by=method]
    r2 = stats[, list(lab=paste("r^2:", round(cor(x,y)^2, 2), sep="~"),
        pos=yMax*0.74), by=method]

    scatterPlot =
        ggplot(stats, aes(x=x, y=y)) +
        stat_binhex(bins=80) + facet_wrap(~method) +
        scale_fill_gradientn(name="# points", trans="log10",
                             colours=topo.colors(20)) +
        geom_smooth(method="lm") +
        geom_abline(colour="red", intercept=0, slope=1) +
        theme_bw(textSize) +
        labs(x=parse(text=xLab), y=parse(text=yLab)) +
        geom_text(data=meanErr,      aes(x=1, y=pos, label=lab), size=4.5, parse=TRUE, hjust=0) +
        geom_text(data=medianRelErr, aes(x=1, y=pos, label=lab), size=4.5, parse=TRUE, hjust=0) +
        geom_text(data=meanRelErr,   aes(x=1, y=pos, label=lab), size=4.5, parse=TRUE, hjust=0) +
        geom_text(data=r2,           aes(x=1, y=pos, label=lab), size=4.5, parse=TRUE, hjust=0) +
        theme(strip.background=element_rect(colour="white", fill="white"),
              plot.margin=unit(c(0,0,0,0), "cm"))

    #facet_wrap_labeller(scatterPlot, c(parse(text=plot1Lab), parse(text=plot2Lab)))
    return(scatterPlot)
}

makeGPMScatterPlots = function(GPM) {
    ## Plot for Dm.
    GPMscatterData_Dm = rbind(GPM[, list(simDm, gpmDm=gpmDm_DFR, method="DFR")],
        GPM[, list(simDm, gpmDm=gpmDm_kZ, method="kZ")])
    GPMscatterData_Dm = GPMscatterData_Dm[, diff := gpmDm - simDm]
    GPMscatterData_Dm = GPMscatterData_Dm[, relDiff := (gpmDm - simDm) / abs(simDm) * 100]
    
    Dm = densityPlotComparison(GPMscatterData_Dm, xVal="simDm", yVal="gpmDm",
        xLab="DSD-based~D[m]~group('[',mm,']')",
        yLab="GPM-simulated~D[m]~group('[',mm,']')",
        plot1Lab="(a)~D[m]~using~DFR",
        plot2Lab="(b)~D[m]~using~k/Z[l]~at~Ku-band",
        unit="mm")

    ## Plot for Nw.
    GPMscatterData_Nw = rbind(GPM[, list(simNw, gpmNw=Nw_DFR, method="DFR")],
        GPM[, list(simNw, gpmNw=Nw_kZ, method="kZ")])
    GPMscatterData_Nw = GPMscatterData_Nw[, diff := gpmNw - simNw]
    GPMscatterData_Nw = GPMscatterData_Nw[, relDiff := (gpmNw - simNw) / abs(simNw) * 100]
    
    Nw = densityPlotComparison(GPMscatterData_Nw, xVal="simNw", yVal="gpmNw",
        xLab="DSD-based~N[w]~group('[',mm^{-1}~m^{-3},']')",
        yLab="GPM-simulated~N[w]~group('[',mm^{-1}~m^{-3},']')",
        plot1Lab="(a)~N[w]~using~DFR",
        plot2Lab="(b)~N[w]~using~k/Z[l]~at~Ku-band",
        unit="mm^{-1}~m^{-3}")

    ## Plot for R.
    GPMscatterData_R = rbind(GPM[, list(simR, gpmR=gpmR_DFR, method="DFR")],
        GPM[, list(simR, gpmR=gpmR_kZ, method="kZ")])
    GPMscatterData_R = GPMscatterData_R[, diff := gpmR - simR]
    GPMscatterData_R = GPMscatterData_R[, relDiff := (gpmR - simR) / abs(simR) * 100]
    
    R = densityPlotComparison(dat=GPMscatterData_R, xVal="simR", yVal="gpmR",
        xLab="DSD-based~R~group('[',mm~h^{-1},']')",
        yLab="GPM-simulated~R~group('[',mm~h^{-1},']')",
        plot1Lab="(a)~R~from~DFR",
        plot2Lab="(b)~R~from~k/Z[l]~at~Ku-band",
        unit="mm~h^{-1}")

    ## Plot for Nt.
    GPMscatterData_Nt = rbind(GPM[, list(simNt, gpmNt=gpmNt_DFR, method="DFR")],
        GPM[, list(simNt, gpmNt=gpmNt_kZ, method="kZ")])
    GPMscatterData_Nt = GPMscatterData_Nt[, diff := gpmNt - simNt]
    GPMscatterData_Nt = GPMscatterData_Nt[, relDiff := (gpmNt - simNt) / abs(simNt) * 100]
    
    Nt = densityPlotComparison(dat=GPMscatterData_Nt, xVal="simNt", yVal="gpmNt",
        xLab="DSD-based~Nt~group('[',m^{-3},']')",
        yLab="GPM-simulated~Nt~group('[',m^{-3},']')",
        plot1Lab="(a)~N[t]~using~DFR",
        plot2Lab="(b)~N[t]~using~k/Z[l]~at~Ku-band",
        unit="m^{-3}")

    ## Plot for Z.
    GPMscatterData_Z = rbind(GPM[, list(simZ, gpmZ=gpmZ_DFR, method="DFR")],
        GPM[, list(simZ, gpmZ=gpmZ_kZ, method="kZ")])
    GPMscatterData_Z = GPMscatterData_Z[, diff := gpmZ - simZ]
    GPMscatterData_Z = GPMscatterData_Z[, relDiff := (gpmZ - simZ) / abs(simZ) * 100]
    
    Z = densityPlotComparison(dat=GPMscatterData_Z, xVal="simZ", yVal="gpmZ",
        xLab="DSD-based~Z~group('[',mm^6~m^{-3},']')",
        yLab="GPM-simulated~Z~group('[',mm^6~m^{-3},']')",
        plot1Lab="(a)~Z~using~DFR",
        plot2Lab="(b)~Z~using~k/Z[l]~at~Ku-band",
        unit="mm^6~m^{-3}")
    
    return(list(Dm=Dm, Nw=Nw, R=R, Nt=Nt, Z=Z))
}

perc = function(subGrid, areal, numPix) {
    ## Determine percentile of areal value in subgrid values.
    ##
    ## Args:
    ##  subGrid: Non-zero subgrid values to compare to.
    ##  areal: Areal value to compare.
    ##  numPix: The total number of pixels in the subgrid.
    ##
    ## Returns: Percentile of areal value in subgrid distribution,
    ##          counting zeros.
    
    stopifnot(length(subGrid) <= numPix)
    subGrid = c(subGrid, rep(0, numPix - length(subGrid)))
    return(mean(subGrid <= areal, na.rm=T) * 100)
}

percNoZeros = function(subGrid, areal, numPix) {
    ## Determine percentile of areal value in subgrid values,
    ## without filling empty values with zeros.
    ## 
    ## Args:
    ##  subGrid: Non-zero subgrid values to compare to.
    ##  areal: Areal value to compare.
    ##  numPix: The total number of pixels in the subgrid.
    ##
    ## Returns: Percentile of areal value in subgrid distribution,
    ##          not counting zeros.
    
    return(mean(subGrid <= areal, na.rm=T) * 100)
}

compareGPMToSubGrid = function(GPM,
    simBulkPattern="allSimsBulkLarge",
    simBulkDir="/data/DSD scale effects cache/1 min/") {

    ## Pull simulated values in to allSimBulk.
    statsPerSim = NULL
    for(file in list.files(simBulkDir,
                           pattern=simBulkPattern,
                           full.name=TRUE)) {
        allSimBulk = get(name <- load(file))
        if(name != "allSimBulk")
            rm(list=name)
        if(is.null(allSimBulk)) next
        
        setkey(allSimBulk, POSIXtime, simNumber)
        setkey(GPM, POSIXtime, simNumber)
        allSimBulk = allSimBulk[GPM, list(POSIXtime, simNumber, x, y, Dm, 
            LWC, R, Nt, Z, simDm, simR, simNt, simZ, simW, gpmDm_DFR, gpmDm_kZ, 
            Nw_DFR, Nw_kZ, gpmNt_DFR, gpmNt_kZ, gpmR_DFR, gpmR_kZ, 
            gpmZ_DFR, gpmZ_kZ), nomatch=0]

        ## Nw per pixel, as calculated from simulated Dm and LWC.
        largePixDensity = waterDensity(altitude=altitude, temperature=15,
            latitude=latitude) * 1e3 # Water density [g cm^-3].
        allSimBulk = allSimBulk[, Nw := (4^4)/(pi*largePixDensity) * (10^3*LWC/Dm^4)]
        allSimBulk = allSimBulk[, simNw := (4^4)/(pi*largePixDensity) *
            (10^3*simW/simDm^4)]
        
        ## Convert 6th moment DSD to linear units.
        allSimBulk[, Z := 10^(Z/10)]
        
        p = length(largeGrid)
        statsPer = allSimBulk[, list(n=length(Dm),         ## Num pixels in simul. field.
            perc_gpmDm_DFR=percNoZeros(Dm, unique(gpmDm_DFR), p), ## Percentile of GPM's Dm (DFR).
            perc_gpmDm_kZ=percNoZeros(Dm, unique(gpmDm_kZ), p),   ## Percentile of GPM's Dm (k/Z).
            perc_simDm=percNoZeros(Dm, unique(simDm), p),         ## Percentile of mean DSD's Dm.
            perc_gpmNw_DFR=perc(Nw, unique(Nw_DFR), p),    ## Percentile of GPM's Nw (DFR).
            perc_gpmNw_kZ=perc(Nw, unique(Nw_kZ), p),      ## Percentile of GPM's Nw (kZ).
            perc_simNw=perc(Nw, unique(simNw), p),         ## Percentile of mean DSD's Nw.
            perc_gpmR_DFR=perc(R, unique(gpmR_DFR), p),    ## Percentile of GPM's R (DFR).
            perc_gpmR_kZ=perc(R, unique(gpmR_kZ), p),      ## Percentile of GPM's R (Kz).
            perc_simR=perc(R, unique(simR), p),            ## Percentile of mean DSD's R.
            perc_gpmNt_DFR=perc(Nt, unique(gpmNt_DFR), p), ## Percentile of GPM's Nt (DFR).
            perc_gpmNt_kZ=perc(Nt, unique(gpmNt_kZ), p),   ## Percentile of GPM's Nt (Kz).
            perc_simNt=perc(Nt, unique(simNt), p),         ## Percentile of mean DSD's Nt.
            perc_gpmZ_DFR=perc(Z, unique(gpmZ_DFR), p),    ## Percentile of GPM's R (DFR).
            perc_gpmZ_kZ=perc(Z, unique(gpmZ_kZ), p),      ## Percentile of GPM's R (Kz).
            perc_simZ=perc(Z, unique(simZ), p)),           ## Percentile of mean DSD's R.
            by="POSIXtime,simNumber"]
        
        statsPerSim = rbind(statsPerSim, statsPer)
        rm(list=c("statsPer", "allSimBulk"))
    }

    densities = rbind(
        statsPerSim[, list(var="Dm", meth="DFR", value=perc_gpmDm_DFR)],
        statsPerSim[, list(var="Dm", meth="k/Z", value=perc_gpmDm_kZ)],
        statsPerSim[, list(var="Dm", meth="Avg", value=perc_simDm)],
        statsPerSim[, list(var="Nw", meth="DFR", value=perc_gpmNw_DFR)],
        statsPerSim[, list(var="Nw", meth="k/Z", value=perc_gpmNw_kZ)],
        statsPerSim[, list(var="Nw", meth="Avg", value=perc_simNw)],
        statsPerSim[, list(var="R",  meth="DFR", value=perc_gpmR_DFR)],
        statsPerSim[, list(var="R",  meth="k/Z", value=perc_gpmR_kZ)],
        statsPerSim[, list(var="R",  meth="Avg", value=perc_simR)],
        statsPerSim[, list(var="Nt",  meth="DFR", value=perc_gpmNt_DFR)],
        statsPerSim[, list(var="Nt",  meth="k/Z", value=perc_gpmNt_kZ)],
        statsPerSim[, list(var="Nt",  meth="Avg", value=perc_simNt)],
        statsPerSim[, list(var="Z",  meth="DFR", value=perc_gpmZ_DFR)],
        statsPerSim[, list(var="Z",  meth="k/Z", value=perc_gpmZ_kZ)],
        statsPerSim[, list(var="Z",  meth="Avg", value=perc_simZ)])

    return(densities)
}

