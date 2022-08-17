## COSMO_functions.R
##
## Functions to simulate COSMO-style retrieval of the DSD.
##
## Author: Tim Raupach <tim.raupach@epfl.ch>

source("library/GPM_functions.R")

RfromCosmoModel = function(N0, mu, lambda, D, widths, VDcosmo) {
    return(6*pi*10^(-4) * sum(N0*exp(-lambda*D)*D^mu*D^3*VDcosmo*widths))
}

DmFromCosmoModel = function(N0, mu, lambda, D, widths, VDcosmo) {
    return(sum(N0*D^mu*exp(-lambda*D)*D^4*widths)/
           sum(N0*D^mu*exp(-lambda*D)*D^3*widths))
}

NtFromCosmoModel = function(N0, mu, lambda, D, widths, VDcosmo) {
    return(sum(N0*D^mu*exp(-lambda*D)*widths))
}
   
ZFromCosmoModel = function(N0, mu, lambda, D, widths, VDcosmo) {
    return(sum(N0*D^mu*exp(-lambda*D)*D^6*widths))
}

addCOSMOVars = function(cosmo, N0_gamma=1253, mu_gamma=0.5,
    dsdCols=paste("class", seq(1, 32), sep=""),
    waterDensity=1) {
    ## Calculate bulk variables for COSMO.
    ##
    ## Args:
    ##   cosmo: data.table to work with, must contain at
    ##          least R, Dm, LWC, Nt, and DSD drop cols.
    ##   mu_gamma: values of shape parameter mu [-] to use
    ##             with the gamma model.
    ##   N0_gamma: values of intercept parameter N0
    ##             [m-3 mm-1-mu] to use with the gamma model.
    ##   waterDensity: Water density [g cm-3].
    ##
    ## Returns: data.table with simR, simDm, simLWC, simNt, simZ (all
    ##          from DSD), then lambda_gamma, lambda_exp, then
    ##          R, Dm, Nt, Z from COSMO gamma and exponential models.
    
    ## Values for newer gamma model.
    cosmo[, N0_gamma := N0_gamma]
    cosmo[, mu_gamma := mu_gamma]
    
    ## Values for older exponential model.
    cosmo[, N0_exp := 8e3]
    cosmo[, mu_exp := 0]

    ## Add 6th-moment radar reflectivity [m-3 mm6].
    cosmo = cosmo[!is.na(LWC) & !is.na(Dm)]
    cosmo = cosmo[, simZ := radarRef(.SD), .SDcols=dsdCols]
    cosmo = cosmo[, list(POSIXtime, simNumber, simR=R, simDm=Dm,
        simLWC=LWC, simNt=Nt, simZ, mu_gamma, N0_gamma,
        mu_exp, N0_exp)]
    
    ## Find Lambda using the total water content.
    cosmo[, lambda_gamma := (pi*(1e-3)*waterDensity*N0_gamma*gamma(4+mu_gamma)/
                             (6*simLWC))^(1/(4+mu_gamma))]
    cosmo[, lambda_exp := (pi*(1e-3)*waterDensity*N0_exp*gamma(4+mu_exp)/
                           (6*simLWC))^(1/(4+mu_exp))]

    ## Determine R and Dm using the cosmo model.
    ## Use cosmo's estimation of terminal velocity.
    ## Truncate DSD to same classes as measured in simulated fields.
    D = get.classD()[3:23] 
    widths = apply(get.classD(), 1, diff)[3:23]
    VDcosmo = 130*(1e-3*D)^0.5
    cosmo = cosmo[, cosmoR_gamma := RfromCosmoModel(N0_gamma, mu_gamma, lambda_gamma,
                                     D, widths, VDcosmo), by=1:length(simLWC)]
    cosmo = cosmo[, cosmoR_exp := RfromCosmoModel(N0_exp, mu_exp, lambda_exp,
                                   D, widths, VDcosmo), by=1:length(simLWC)]

    cosmo = cosmo[, cosmoDm_gamma:= DmFromCosmoModel(N0_gamma, mu_gamma, lambda_gamma,
                        D, widths, VDcosmo), by=1:length(simLWC)]
    cosmo = cosmo[, cosmoDm_exp:= DmFromCosmoModel(N0_exp, mu_exp, lambda_exp,
                        D, widths, VDcosmo), by=1:length(simLWC)]

    cosmo = cosmo[, cosmoNt_gamma:= NtFromCosmoModel(N0_gamma, mu_gamma, lambda_gamma,
                        D, widths, VDcosmo), by=1:length(simLWC)]
    cosmo = cosmo[, cosmoNt_exp:= NtFromCosmoModel(N0_exp, mu_exp, lambda_exp,
                        D, widths, VDcosmo), by=1:length(simLWC)]

    cosmo = cosmo[, cosmoZ_gamma:= ZFromCosmoModel(N0_gamma, mu_gamma, lambda_gamma,
                        D, widths, VDcosmo), by=1:length(simLWC)]
    cosmo = cosmo[, cosmoZ_exp:= ZFromCosmoModel(N0_exp, mu_exp, lambda_exp,
                        D, widths, VDcosmo), by=1:length(simLWC)]
    
    return(cosmo)
}

makeCOSMOScatterPlots = function(cosmo) {
    ## Make scatter plots for comparisons of COSMO parameters.
    
    ## Nt
    cosmoScatterData_Nt = rbind(cosmo[, list(simNt, cosmoNt=cosmoNt_exp, method="Exp")],
        cosmo[, list(simNt, cosmoNt=cosmoNt_gamma, method="Gamma")])
    cosmoScatterData_Nt = cosmoScatterData_Nt[, diff := cosmoNt - simNt]
    cosmoScatterData_Nt = cosmoScatterData_Nt[, relDiff := (cosmoNt - simNt) / abs(simNt) * 100]
    
    Nt = densityPlotComparison(dat=cosmoScatterData_Nt, xVal="simNt", yVal="cosmoNt",
        xLab="DSD-based~N[t]~group('[',m^{-3},']')",
        yLab="COSMO-simulated~N[t]~group('[',m^{-3},']')",
        plot1Lab="(a)~N[t]~using~exp.~DSD~model",
        plot2Lab="(b)~N[t]~using~'Gamma'~DSD~model")

    ## Dm
    cosmoScatterData_Dm = rbind(cosmo[, list(simDm, cosmoDm=cosmoDm_exp, method="Exp")],
        cosmo[, list(simDm, cosmoDm=cosmoDm_gamma, method="Gamma")])
    cosmoScatterData_Dm = cosmoScatterData_Dm[, diff := cosmoDm - simDm]
    cosmoScatterData_Dm = cosmoScatterData_Dm[, relDiff := (cosmoDm - simDm) / abs(simDm) * 100]

    Dm = densityPlotComparison(dat=cosmoScatterData_Dm, xVal="simDm", yVal="cosmoDm",
        xLab="DSD-based~D[m]~group('[',mm,']')",
        yLab="COSMO-simulated~D[m]~group('[',mm,']')",
        plot1Lab="(a)~D[m]~using~exp.~DSD~model",
        plot2Lab="(b)~D[m]~using~'Gamma'~DSD~model",
        unit="mm")

    ## R
    cosmoScatterData_R = rbind(cosmo[, list(simR, cosmoR=cosmoR_exp, method="Exp")],
        cosmo[, list(simR, cosmoR=cosmoR_gamma, method="Gamma")])
    cosmoScatterData_R = cosmoScatterData_R[, diff := cosmoR - simR]
    cosmoScatterData_R = cosmoScatterData_R[, relDiff := (cosmoR - simR) / abs(simR) * 100]
    
    R = densityPlotComparison(dat=cosmoScatterData_R, xVal="simR", yVal="cosmoR",
        xLab="DSD-based~R~group('[',mm~h^{-1},']')",
        yLab="COSMO-simulated~R~group('[',mm~h^{-1},']')",
        plot1Lab="(a)~R~from~exp.~DSD~model",
        plot2Lab="(b)~R~from~'Gamma'~DSD~model",
        unit="mm~h^{-1}")
    
    ## Z
    cosmoScatterData_Z = rbind(cosmo[, list(simZ, cosmoZ=cosmoZ_exp, method="Exp")],
    cosmo[, list(simZ, cosmoZ=cosmoZ_gamma, method="Gamma")])
    cosmoScatterData_Z = cosmoScatterData_Z[, diff := cosmoZ - simZ]
    cosmoScatterData_Z = cosmoScatterData_Z[, relDiff := (cosmoZ - simZ) / abs(simZ) * 100]
    
    Z = densityPlotComparison(dat=cosmoScatterData_Z, xVal="simZ", yVal="cosmoZ",
        xLab="DSD-based~Z~group('[',mm^6~m^{-3},']')",
        yLab="COSMO-simulated~Z~group('[',mm^6~m^{-3},']')",
        plot1Lab="(a)~Z~using~exp.~DSD~model",
        plot2Lab="(b)~Z~using~'Gamma'~DSD~model",
        unit="mm^6~m^{-3}")

    return(list(Nt=Nt, Dm=Dm, R=R, Z=Z))
}

compareCOSMOToSubGrid = function(cosmo,
    simBulkPattern="allSimsBulkSmall",
    simBulkDir="/data/DSD scale effects cache/1 min/") {
    
    cosmoStatsPerSim = NULL
    for(file in list.files(simBulkDir,
                           pattern=simBulkPattern,
                           full.name=TRUE)) {
        allSimBulkSmall = get(name <- load(file))
        if(name != "allSimBulkSmall")
            rm(list=name)
        if(is.null(allSimBulkSmall)) next
        
        setkey(allSimBulkSmall, POSIXtime, simNumber)
        setkey(cosmo, POSIXtime, simNumber)
        allSimBulkSmall = cosmo[allSimBulkSmall, nomatch=0]
        
        ## Convert 6th moment Z to linear units.
        allSimBulkSmall[, Z := 10^(Z/10)]
        
        p = length(smallGrid)
        cosmoStatsPer = allSimBulkSmall[, list(n=length(Dm),       ## Num pixels in simul. field.
            perc_cosmoDm_exp=percNoZeros(Dm, unique(cosmoDm_exp), p),     ## Percentile of COSMO's Dm (Exp).
            perc_cosmoR_exp=perc(R, unique(cosmoR_exp), p),        ## Percentile of COSMO's R (Exp).
            perc_cosmoNt_exp=perc(Nt, unique(cosmoNt_exp), p),     ## Percentile of COSMO's Nt (Exp).
            perc_cosmoZ_exp=perc(Z, unique(cosmoZ_exp), p),        ## Percentile of COSMO's Z (Exp).
            perc_cosmoDm_gamma=percNoZeros(Dm, unique(cosmoDm_gamma), p), ## Percentile of COSMO's Dm (Gamma).
            perc_cosmoR_gamma=perc(R, unique(cosmoR_gamma), p),    ## Percentile of COSMO's R (Gamma).
            perc_cosmoNt_gamma=perc(Nt, unique(cosmoNt_gamma), p), ## Percentile of COSMO's Nt (Gamma).
            perc_cosmoZ_gamma=perc(Z, unique(cosmoZ_gamma), p),    ## Percentile of COSMO's Z (Gamma).
            perc_simR=perc(R, unique(simR), p),                    ## Percentile of avg R.
            perc_simDm=percNoZeros(Dm, unique(simDm), p),                 ## Percentile of avg Dm.
            perc_simNt=perc(Nt, unique(simNt), p),                 ## Percentile of avg Nt.
            perc_simZ=perc(Z, unique(simZ), p)),                   ## Percentile of avg Z.
            by="POSIXtime,simNumber"]
        
        cosmoStatsPerSim = rbind(cosmoStatsPerSim, cosmoStatsPer)
        rm(list=c("allSimBulkSmall", "cosmoStatsPer"))
    }
    
    cosmoDensities = rbind(
        cosmoStatsPerSim[, list(var="Dm", meth="Exp", value=perc_cosmoDm_exp)], 
        cosmoStatsPerSim[, list(var="R", meth="Exp", value=perc_cosmoR_exp)],
        cosmoStatsPerSim[, list(var="Nt", meth="Exp", value=perc_cosmoNt_exp)], 
        cosmoStatsPerSim[, list(var="Z", meth="Exp", value=perc_cosmoZ_exp)],
        cosmoStatsPerSim[, list(var="Dm", meth="Gamma", value=perc_cosmoDm_gamma)],
        cosmoStatsPerSim[, list(var="R", meth="Gamma", value=perc_cosmoR_gamma)],
        cosmoStatsPerSim[, list(var="Nt", meth="Gamma", value=perc_cosmoNt_gamma)],
        cosmoStatsPerSim[, list(var="Z", meth="Gamma", value=perc_cosmoZ_gamma)],
        cosmoStatsPerSim[, list(var="Dm", meth="Avg", value=perc_simDm)],
        cosmoStatsPerSim[, list(var="R", meth="Avg", value=perc_simR)],
        cosmoStatsPerSim[, list(var="Nt", meth="Avg", value=perc_simNt)],
        cosmoStatsPerSim[, list(var="Z", meth="Avg", value=perc_simZ)])
    
    return(cosmoDensities)
}

