## SCOP-ME_functions.R
##
## Implementation of the SCOP-ME DSD-retrieval algorithm, as published by
## Kalogiros_IEEE_TGRS_2013 and as presented in Anagnotou_JHM_2013.

source("library/axis_ratio_functions.R")

scopmeDSD = function(dat, D, dD, dsdCols, diamCols, widthCols,
    k=c("POSIXtime","station"), kdpNoiseLimit=0) {
    ## Reconstruct the DSD using the SCOP-ME method.
    ##
    ## Args:
    ##   dat: The data.table containing radar data (Zh, Zv, Kdp) to use.
    ##   D: Centres of drop size classes to reconstruct.
    ##   dD: Width of drop size classes to reconstruct.
    ##   dsdCols, diamCols, widthCols: Names for reconstructed
    ##                                 columns for concentrations,
    ##                                 diameters, widths.
    ##   kdpNoiseLimit: The value of Kdp (deg. km-1) under which to
    ##                  assume it is noisy. In this case D_Z is
    ##                  calculated using Eq. 14b instead of Eq. 14a in
    ##                  Kalogiros_IEEE_TGRS_2013 (default 0, do not
    ##                  use).
    ##
    ## Returns: reconstructed DSDs.

    stopifnot(length(D) == length(dsdCols))
    stopifnot(length(D) == length(diamCols))
    stopifnot(length(D) == length(widthCols))
    
    x = copy(dat)
    x = x[, c("Zh", "Zv", "Kdp", k), with=FALSE]

    ## x should contain Zh [dBZ], Zv [dBZ] and Kdp [deg. km-1].
    x[, ZhLin := 10^(Zh/10)]
    x[, ZvLin := 10^(Zv/10)]
    x[, ZdrLin := ZhLin/ZvLin]

    ## The third-degree rational polynomial regression function.
    f = function(D_Z, a_0, a_1, a_2, a_3, b_0, b_1, b_2, b_3) {
        return((a_0*D_Z^0 + a_1*D_Z^1 + a_2*D_Z^2 + a_3*D_Z^3) /
               (b_0*D_Z^0 + b_1*D_Z^1 + b_2*D_Z^2 + b_3*D_Z^3))
    }

    ## Reflectivity-weighted mean diameter (mm). Using Kalogiros eq. 14a.
    x[Kdp >= kdpNoiseLimit,
      D_Z1 := 0.1802 *((ZhLin/Kdp) * ZdrLin^-0.2929 * (1 - ZdrLin^-0.4922))^(1/3)]
    x[Kdp >= kdpNoiseLimit,
      D_Z := D_Z1 * f(D_Z1, a_0=0.9190, a_1=0.1501,  a_2=-0.1722, a_3=0.0511,
                            b_0=1,      b_1=-0.2248, b_2=0.0182,  b_3=0.023)]

    ## Reflectivity-weighted mean diameter (mm). Using Kalogiros eq. 14b for
    ## low or missing Kdp.
    x[Kdp < kdpNoiseLimit, D_Z2 := 2.4780 * (1 - ZdrLin^-0.5089)]
    x[Kdp < kdpNoiseLimit, 
      D_Z := D_Z2 * f(D_Z2, a_0=0.0546, a_1=0.1056, a_2=-0.1587,  a_3=0.0976,
                            b_0=0.0012, b_1=0.0361, b_2=-0.0180, b_3=-0.0084)]
    stopifnot(!any(is.na(x$D_Z)))
    
    x[, D_0 := D_Z * f(D_Z,
                a_0=0.9542, a_1=0.2989, a_2=0.0577, a_3=0.0030,
                b_0=1,      b_1=0.2243, b_2=0.2949, b_3=-0.005)]

    x[, mu := 165 * exp(-2.56*D_0) - 1]
    x[, fmu := 6/(3.67)^4 * ((3.67 + mu)^(mu+4)/gamma(mu+4))]
    
    ## Again Nw is calculated using Kdp, only for "good" values of Kdp (Eq. 16b). When
    ## Kdp is below the noise limit, we use Eq. 16a.
    x[Kdp >= kdpNoiseLimit, Nw := 3610 * (Kdp / (1 - ZdrLin^-0.3893)) * D_0^-4 * f(D_Z,
                                   a_0=1, a_1=-0.6792, a_2=0.2112, a_3=-0.0109,
                                   b_0=1, b_1=-0.6410, b_2=0.1551, b_3=-0.0065)]
    x[Kdp < kdpNoiseLimit, Nw := 1.0174 *
      (Zh / (fmu * gamma(mu + 6 + 1) / (mu + 3.67)^(mu+6+1))) *
      D_0^-7 * f(D_Z,
                 a_0=1, a_1=-0.3487, a_2=-0.0185, a_3=-0.0109,
                 b_0=1, b_1=-0.3689, b_2=-0.0256, b_3=0.0234)]
    x[is.infinite(Nw), Nw := NA]
    
    ## Reconstruct DSDs.
    dsds = x[, c("Nw", "mu", "D_0", "fmu", k), with=FALSE]

    for(class in seq(1, length(D))) {
        diam = D[class]
        width = dD[class]
        dsds[, (dsdCols[class]) := Nw * fmu * (diam/D_0)^mu * exp(-(mu + 3.67)*(diam/D_0))]
        dsds[, (diamCols[class]) := diam]
        dsds[, (widthCols[class]) := width]
    }

    dsds[, containsNA := is.na(rowSums(.SD)), .SDcols=dsdCols]
    return(dsds)
}

testScopme = function(measured, toRecon, D, dD, dsdCols, diamCols, widthCols,
    k=c("station","POSIXtime","alt","lat"), seaLevelTemp=15, ...) {
    ## Test the SCOP-ME function against measured data.
    ##
    ## Args:
    ##   measured: only used to compare against.
    ##   toRecon: contains radar variables to use for the reconstruction.
    ##   D, dD: Centres and widths of classes to reconstruct [mm].
    ##   k: Columns to keep in reconstructed set.
    ##   seaLevelTemp: Assumed sea level temperature for R [deg].
    ##
    ## Returns: comparison statistics.

    ## Reconstruct the DSD.
    toRecon = toRecon[, c("Zh", "Zv", "Kdp", k), with=FALSE]
    reconScopme = scopmeDSD(toRecon, D=D, dD=dD,
        widthCols=widthCols, diamCols=diamCols,
        dsdCols=dsdCols, ...)

    ## Calculate moments.
    if("n" %in% names(measured)) measured[, n := NULL]
    for(n in seq(0, 7)) {
        measured[, (paste("moment_", n, sep="")) :=
                 DSDMoment(.SD, n=n, dsdCols=dsdCols,
                           widthCols=widthCols,
                           diamCols=diamCols)]

        reconScopme[, (paste("moment_", n, sep="")) :=
                 DSDMoment(.SD, n=n, dsdCols=dsdCols,
                           widthCols=widthCols,
                           diamCols=diamCols)]
    }

    ## Calculate R.
    stopifnot(identical(measured$POSIXtime, reconScopme$POSIXtime))
    stopifnot(identical(measured$station, reconScopme$station))
    reconScopme$alt = measured$alt
    reconScopme$lat = measured$lat
    classes = cbind(D - dD/2, D + dD/2)
    reconScopme[, R := DSDRainrate(spectra=.SD, classes=classes,
                  altitude=alt, latitude=lat, seaLevelTemperature=seaLevelTemp),
          by=c("alt","lat"),
          .SDcols=dsdCols]

    ## Calculate Dm.
    reconScopme[, Dm := moment_4 / moment_3]
    measured[, Dm := moment_4 / moment_3]

    ## Run comparison.
    moments = paste("moment_", seq(0, 7), sep="")
    measured = measured[, c(k, moments, "R", "Dm"), with=FALSE]
    reconScopme = reconScopme[, c(k, moments, "R", "Dm"), with=FALSE]
    measuredMoments = melt(measured, id.vars=k)
    reconMoments = melt(reconScopme, id.vars=k)

    ## Join together the two sets to compare.
    setnames(reconMoments, "value", "reconstructed")
    setnames(measuredMoments, "value", "measured")
    setkeyv(reconMoments, c(k, "variable"))
    setkeyv(measuredMoments, c(k, "variable"))

    results = reconMoments[measuredMoments, nomatch=0]
    results = results[!is.na(measured) & !is.na(reconstructed)]
    results = results[!is.infinite(measured) & !is.infinite(reconstructed)]

    momentsPlot = ggplot(results, aes(x=measured, y=reconstructed)) +
        geom_point() + facet_wrap(~variable, scale="free") +
        geom_abline(intercept=0, slope=1, colour="red")

    results[, diff := reconstructed - measured]
    results[, relDiff := diff / measured * 100]

    return(results)
}

testScopmeWithRatios = function(dat_andsager, dat_thurai, dat_brandes, dat_beard,
    D, dD, dsdCols, widthCols, diamCols, ...) {
    ## Test the SCOP-ME method with different axis ratios.
    ##
    ## Args:
    ##   dat_*: Data set of radar variables to use by axis ratio function.
    ##   D, dD: centre and widths of drop size classes [mm].
    ##   dsdCols, widthCols, diamCols: Names of columns to
    ##                                 reconstruct for DSD
    ##                                 concentrations, class
    ##                                 widths and centre diameters
    ##                                 respectively. Must be the
    ##                                 same names as those in "dat".
    ##   ...: Optional extra arguments to testScopme.
    ##
    ## Returns: A list with raw results and a table of
    ##          summary statistics.

    stopifnot(!("n" %in% names(dat_andsager)))
    stopifnot(!("n" %in% names(dat_thurai)))
    stopifnot(!("n" %in% names(dat_brandes)))
    stopifnot(!("n" %in% names(dat_beard)))
    
    ## Test with Thurai axis ratios.
    results_thurai = testScopme(measured=copy(dat_thurai), toRecon=copy(dat_thurai),
        D=D, dD=dD, dsdCols=dsdCols, diamCols=diamCols, widthCols=widthCols,
        ...)

    ## Test with Brandes axis ratios.
    results_brandes = testScopme(measured=copy(dat_brandes), toRecon=copy(dat_brandes),
        D=D, dD=dD, dsdCols=dsdCols, diamCols=diamCols, widthCols=widthCols,
        ...)

    ## Test with Andsager axis ratios.
    results_andsager = testScopme(measured=copy(dat_andsager), toRecon=copy(dat_andsager),
        D=D, dD=dD, dsdCols=dsdCols, diamCols=diamCols, widthCols=widthCols,
        ...)

    ## Test with Beard axis ratios.
    results_beard = testScopme(measured=copy(dat_beard), toRecon=copy(dat_beard),
        D=D, dD=dD, dsdCols=dsdCols, diamCols=diamCols, widthCols=widthCols,
        ...)

    ## Join together all results.
    results = rbindlist(list(data.table(results_thurai, axisfunc="Thurai"),
        data.table(results_brandes, axisfunc="Brandes"),
        data.table(results_andsager, axisfunc="Andsager"),
        data.table(results_beard, axisfunc="Beard")),
        use.names=TRUE)

    testRes =
        results[, list(order=paste("$M_", str_extract(variable, "[0-9]"), "$", sep=""),
                       medRel=median(relDiff, na.rm=TRUE),
                       RBIQR=IQR(relDiff, na.rm=TRUE),
                       r2=cor(reconstructed, measured, use="pairwise.complete")^2,
                       slope=coef(lm(reconstructed~measured))[2]),
                by="axisfunc,variable"]
    testRes[variable == "R", order := "$R$"]
    testRes[variable == "Dm", order := "$D_m$"]
    testRes[, variable := NULL]

    return(list(res=results, display=testRes))
}
