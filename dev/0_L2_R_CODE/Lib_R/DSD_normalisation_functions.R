## DSD_normalisation_functions.R
##
## Functions to perform normalisation of the DSD using the technique of
## Lee et al 2004.
##
## Code by Tim Raupach <tim.raupach@epfl.ch>

require(minpack.lm)
require(foreach)
require(sp)
require(parallel)
require(doParallel)
require(compiler)

normalisedDSDs = function(spectra, D, widths, i, j, keep="POSIXtime") {
  ## Calculate the normalised DSD for each DSD in a set of spectra,
  ## using the double-moment normalisation technique of Lee et al
  ## 2004.
  ##
  ## Finds x and h where
  ##   N(D) = Mi^((j+1)/(j-i)) Mj^((i+1)/(i-j)) h(x)
  ## and
  ##   x = D Mi^(1/(j-i)) Mj^(-1/(j-i)).
  ##
  ## Args:
  ##   spectra: data.table with DSD spectra, one per row.
  ##   D: Diameters of each class.
  ##   widths: Widths of each diameter class.
  ##   i, j: Moment orders to use.
  ##
  ## Returns: a data.table containing the normalised diameters
  ## (x), the widths of normalised diameter classes (xWidths),
  ## the values of the corresponding normalised DSD h(x),
  ## and n, the row number in the original spectra.

  spectra = data.table(spectra)
  D = as.numeric(D)
  widths = as.numeric(widths)
  stopifnot(length(D) == ncol(spectra))
  stopifnot(length(widths) == ncol(spectra))
  stopifnot(!any(duplicated(D[which(!is.na(D))])))
  stopifnot(length(unique(widths)) > 1)

  ## Turn diameters and weights into a matrices the same
  ## dimensions as the spectra.
  D = matrix(rep(as.numeric(D), nrow(spectra)),
             byrow=T, ncol=ncol(spectra))
  widths = matrix(rep(as.numeric(widths),
                      nrow(spectra)), byrow=T, ncol=ncol(spectra))

  ## Calculate moments Mi and Mj.
  cols = copy(names(spectra))
  spectra[, Mi := rowSums(.SD*(D^i)*widths, na.rm=T), .SDcols=cols]
  spectra[, Mj := rowSums(.SD*(D^j)*widths, na.rm=T), .SDcols=cols]

  ## Find the second-moment normalised diameter classes.
  normalisedDiams = data.table(D * spectra[, Mi^(1/(j-i)) * Mj^(-1/(j-i))])
  normalisedWidths = data.table(widths * spectra[, Mi^(1/(j-i)) * Mj^(-1/(j-i))])

  ## Calculate the normalised DSDs. This gives h(x), a function of
  ## normalised diameter.
  normalisedDSDs = spectra[, .SD[, cols, with=FALSE] /
                             (Mi^((j+1)/(j-i)) * Mj^((i+1)/(i-j)))]

  ## Return results.
  res = data.table(hx=normalisedDSDs[, as.numeric(.SD), by=1:nrow(normalisedDSDs)][, V1],
                   D=rep(as.numeric(D[1,]), nrow(normalisedDSDs)),
                   x=normalisedDiams[, as.numeric(.SD), by=1:nrow(normalisedDiams)][, V1],
                   xWidth=normalisedWidths[, as.numeric(.SD), by=1:nrow(normalisedWidths)][, V1],
                   n=rep(1:nrow(spectra), each=ncol(normalisedDiams)))
  return(res)
}

fitNormalisedGamma = function(normDSDs, i, j, startMu=10,
    xClassSize=0.2, weightFactor=1, plotDuring=FALSE,
    startClassesAt=0, ...) {
    ## Fit a Lee et al. generalised gamma model to normalised DSDs.
    ## See generalisedGamma(). This function uses a least squares
    ## approach, and works on values of x between the 1st and 99th
    ## quantiles, since there is often a lack of data points at the
    ## extremes of the range of x.
    ##
    ## Args:
    ##   normDSDs: Output from normalisedDSDs(). Should
    ##             contain second-normalised diameter x [-]
    ##             and normalised DSD value hx [-].
    ##   i, j: Moment orders used for normalisation.
    ##
    ## Returns: data.table containing fitted values for mu and c,
    ##          or NA if not possible, and the residual standard
    ##          error.

    toFit = copy(normDSDs[!is.na(x)])
    xClasses = seq(startClassesAt, max(toFit$x)+xClassSize, by=xClassSize)
    toFit[, xClass := cut(x, xClasses, labels=FALSE)]
    toFit = toFit[, list(hx=median(hx), n=length(hx)), by=xClass]
    toFit[, x := xClasses[xClass]+xClassSize/2]
    toFit[, weight := n^weightFactor]
    toFit$i = i
    toFit$j = j
    toFit = toFit[hx > 0]
    xvals = seq(0, 8, by=0.1)

    resStdErr = NA
    res = NULL
    for(startMu in seq(0, 10, by=2)) {
        testRes = try(nlsLM(formula="log(hx)~log(generalisedGamma(x, mu, c, i, j))",
            data=toFit, control=nls.lm.control(maxiter=100),
            start=list(mu=startMu, c=0.5),
            weights=weight), silent=TRUE)

        if(class(testRes) != "try-error") {
            if(is.na(resStdErr) | summary(testRes)$sigma < resStdErr) {

                ## Make sure that gamma results for a range of values of
                ## x are not NaN.
                gammaRes = generalisedGamma(xvals, mu=coef(testRes)[1],
                    c=coef(testRes)[2], i=i, j=j)
                if(!any(is.na(gammaRes))) {
                    resStdErr = summary(testRes)$sigma
                    res = testRes
                }
            }
        }
    }

    if(is.null(res)) {
        stop(paste("Fit failed with i =", i, "and j =", j))
    }

    if(plotDuring) {
        plot = ggplot(toFit, aes(x=x, y=hx)) +
               geom_point(aes(colour=n)) +
               scale_y_continuous(trans="log10")
        xVals = seq(0.1, max(toFit$x), by=0.1)
        line = data.table(x=xVals, hx=generalisedGamma(x=xVals, mu=coef(res)[1],
                                       c=coef(res)[2], i=i, j=j))
        print(plot + geom_line(data=line))
    }

    return(data.table(t(coef(res)), resStdErr=resStdErr))
}

meanNormalisedDSD = function(dat, numInc=100) {
  ## Assign a class to each value of x and hx.
  dat = dat[!is.na(x) & !is.na(hx)]
  incX = diff(range(dat$x) / numInc)
  incHx = diff(range(dat$hx) / numInc)
  xClasses = seq(0, max(dat$x)+incX, by=incX)
  hxClasses = seq(0, max(dat$hx)+incHx, by=incHx)
  dat[, xClass := cut(x, xClasses, labels=FALSE, right=FALSE)]
  dat[, hxClass := cut(hx, hxClasses, labels=FALSE, right=FALSE)]
  stopifnot(!any(is.na(dat$xClass)))
  stopifnot(!any(is.na(dat$hxClass)))

  ## Count the number of points in each class.
  counts = data.table(count(data.frame(dat), vars=c("xClass","hxClass")))
  counts[, x := xClasses[xClass]]
  counts[, hx := hxClasses[hxClass]]

  ## Find the weighted mean DSD (for each x class, weighted by the number
  ## in each hx class).
  meanDSD = counts[, list(hx=weighted.mean(hx, w=freq)), by=x]
  return(meanDSD)
}

callNormalisedDSD = function(x, dsdCols, diamCols, widthCols, i, j, ...) {
  ## Helper function to perform the DSD normalisation, for use when
  ## the DSD classes are variable.
  ##
  ## Args:
  ##   x: input data.table.
  ##   dsdCols: Column names for DSD concentrations [mm-1 m-3] in x.
  ##   diamCols: Column names for class diameters [mm] in x.
  ##   widthCols: Column names for class widths [mm] in x.
  ##   i, j: Moment orders to use.
  ##   ...: Extra arguments to normalisedDSDs.
  ##
  ## Returns: normalised DSDs.

  stopifnot(length(dsdCols) == length(diamCols))
  stopifnot(length(dsdCols) == length(widthCols))

  ## Loop through sets of diameters/widths.
  diamsAndWidths = unique(x[, c(diamCols, widthCols), with=FALSE])
  x = copy(x)

  if(nrow(diamsAndWidths) > 1) {
    res = NULL

    v = copy(x)
    setkeyv(v, c(diamCols, widthCols))
    setkeyv(diamsAndWidths, c(diamCols, widthCols))

    for(n in seq(1, nrow(diamsAndWidths))) {
      line = diamsAndWidths[n]
      res = rbind(res, v[line,
                         callNormalisedDSD(.SD, dsdCols, diamCols,
                                           widthCols, i, j, ...)])
    }
    return(res)
  }

  res = normalisedDSDs(spectra=x[, dsdCols, with=FALSE],
                       D=unique(x[, diamCols, with=FALSE]),
                       widths=unique(x[, widthCols, with=FALSE]),
                       i=i, j=j, ...)

  x[, n := 1:nrow(x)]
  setkey(x, n)
  setkey(res, n)

  times = res[x, POSIXtime]
  res[x, POSIXtime := times]
  stopifnot(res[x, all(POSIXtime == i.POSIXtime)])
  res[, n := NULL]
  return(res)
}

findNormalisedDSD = function(dsds, dsdCols, diamCols, widthCols, i, j, by=NULL, ...) {
    ## Find the normalised DSD model for a set of data.
    ##
    ## Args:
    ##   dsds: input data.table containing DSDs and class definitions.
    ##   dsdCols: Column names for DSD concentrations [mm-1 m-3] in x.
    ##   diamCols: Column names for class diameters [mm] in x.
    ##   widthCols: Column names for class widths [mm] in x.
    ##   i, j: Moment orders to use.
    ##   by: Optionally, group by values of these columns.
    ##   ...: Optional parameters to fitNormalisedGamma().
    ##
    ## Returns: a list containing normDSDs, the normalised DSDs per "by"
    ##          variable, and gammaParams, a data.table containing the
    ##          fitted mu and c values for the fitted generalised Gamma
    ##          model. Note that the model is fitted per "by" grouping,
    ##          then the median mu and c are found across all fits.

    ## Find each normalised DSD.
    normDSDs = dsds[, callNormalisedDSD(.SD, dsdCols=dsdCols,
        diamCols=diamCols, widthCols=widthCols, i=i, j=j), by=by]

    ## Find Gamma model parameters for all normalised DSDs.
    gammaParams = normDSDs[, fitNormalisedGamma(normDSDs=.SD, i=i, j=j, ...)]
    stopifnot(!any(is.na(gammaParams)))

    return(list(normDSDs=normDSDs, gammaParams=gammaParams))
}

testModel = function(normDSDs, dsds, mu, c, i, j,
                     dsdCols, diamCols, widthCols, by=NULL,
                     compareMoments=seq(0, 7), testBulk=TRUE,
                     testModel=TRUE, testMoments=TRUE, ...) {
  ## Test how well a generalised gamma model captures data.
  ##
  ## Args:
  ##  normDSDs: Empirical normalised DSDs.
  ##  dsds: The DSDs on which the normalised DSDs are based.
  ##  mu, c: Parameters of the gamma model to test.
  ##  i, j: Moment orders to use.
  ##  dsdCols: Column names for DSD concentrations [mm-1 m-3] in x.
  ##  diamCols: Column names for class diameters [mm] in x.
  ##  widthCols: Column names for class widths [mm] in x.
  ##  by: Optional, also calculate stats by group?
  ##  testBulk: Compare bulk params?
  ##  testModel: Compare model?
  ##  testMoments: Compare moments?
  ##
  ## Returns: Differences between model and empirical
  ## normalised DSDs and bulk variables.

  ## Reconstruct the normalised DSD from the Gamma model.
  normDSDs = normDSDs[, modelHx := generalisedGamma(x, mu=mu, c=c, i=i, j=j)]
  normDSDs = normDSDs[!is.na(x)]
  stopifnot(!any(is.na(normDSDs$modelHx)))

  modelFit = NULL
  if(testModel) {
    ## 1. Look at differences between normalised measured DSDs and
    ## the normalised DSD model.
    normDSDs = normDSDs[!is.na(hx) & !is.na(modelHx)]
    normDSDs = normDSDs[, diff := modelHx - hx]
    normDSDs = normDSDs[, relDiff := as.numeric(NA)]
    normDSDs = normDSDs[hx > 0, relDiff := diff / hx * 100]

    ## Difference stats over the whole set.
    modelFit = normDSDs[, list(
      bias = mean(diff),
      medRelBias = median(relDiff, na.rm=TRUE),
      q10RelBias = quantile(relDiff, probs=0.10, na.rm=TRUE),
      q25RelBias = quantile(relDiff, probs=0.25, na.rm=TRUE),
      q75RelBias = quantile(relDiff, probs=0.75, na.rm=TRUE),
      q90RelBias = quantile(relDiff, probs=0.90, na.rm=TRUE),
      rmse = sqrt(mean(diff^2)),
      r2 = cor(hx, modelHx)^2)]
  }

  ## 2. Look at differences between reconstructed DSD moments and
  ## measured DSD moments.
  if("n" %in% names(dsds))
    stop("dsds must not contain a column named 'n'")

  ## Calculate the ith and jth moments of each DSD.
  dsds[, ithMoment := DSDMoment(dsd=.SD, dsdCols=dsdCols,
                                diamCols=diamCols, widthCols=widthCols, n=i)]
  dsds[, jthMoment := DSDMoment(dsd=.SD, dsdCols=dsdCols,
                                diamCols=diamCols, widthCols=widthCols, n=j)]

  ## Reconstruct DSDs based on moments (validation) and fitted
  ## gamma parameters (training).
  momentsOnly = dsds[, c("ithMoment", "jthMoment", diamCols), with=FALSE]
  recon = reconstructDSDs(dat=momentsOnly,
                          i=i, j=j, c=c, mu=mu, diamCols=diamCols)
  reconDSDCols = paste("reconClass", seq(1, length(diamCols)), sep="")
  dsds[, (reconDSDCols) := recon[, reconDSDCols, with=FALSE]]

  momentDiffs = NULL
  moments = NULL
  if(testMoments) {
    ## Differences between DSD moments.
    for(n in compareMoments) {
      momentName = paste("moment_", n, sep="")

      momentRes = data.table(order=n,
                             measured=dsds[, momentName, with=FALSE][[momentName]],
                             reconstructed=DSDMoment(dsds, n=n, dsdCols=reconDSDCols,
                                 widthCols=widthCols, diamCols=diamCols))

      if(!is.null(by))
        momentRes = cbind(momentRes, dsds[, by, with=FALSE])
      moments = rbind(moments, momentRes)
    }
    moments[, diff := reconstructed - measured]
    moments[, relDiff := diff / measured * 100]

    ## Difference stats over the whole set.
    momentDiffs = moments[, list(
      bias = mean(diff),
      medRelBias = median(relDiff),
      q10RelBias = quantile(relDiff, probs=0.10),
      q25RelBias = quantile(relDiff, probs=0.25),
      q75RelBias = quantile(relDiff, probs=0.75),
      q90RelBias = quantile(relDiff, probs=0.90),
      rmse = sqrt(mean(diff^2)),
      r2 = cor(measured, reconstructed)^2), by=order]
  }

  bulkDiffs = NULL
  bulk = NULL
  if(testBulk) {
    ## Calculate R and Dm.
    stopifnot(!is.null(dsds$altitude) & !is.null(dsds$latitude))

    ## To get R we have to process in batches of diameter/width combinations,
    ## as well as in batches of altitude and latitude.
    diamColNames = paste(c(diamCols, widthCols, "altitude", "latitude"), collapse=",")

    measuredR = dsds[, list(R=getR(.SD, dsdCols=dsdCols,
                                   diamCols=diamCols, widthCols=widthCols)),
                     by=diamColNames, .SDcols=names(dsds)]$R
    reconstructedR = dsds[, list(R=getR(.SD, dsdCols=reconDSDCols,
                                        diamCols=diamCols, widthCols=widthCols)),
                          by=diamColNames, .SDcols=names(dsds)]$R
    RDiffs = data.table(var="R", measured=measuredR,
                        reconstructed=reconstructedR)
    if(!is.null(by)) {
      byR = dsds[, by, with=FALSE, by=diamColNames]
      RDiffs = cbind(RDiffs, byR)
    }

    DmDiffs = data.table(var="Dm",
                         measured=dsds[, DSDMoment(.SD, n=4, dsdCols=dsdCols,
                                                   diamCols=diamCols, widthCols=widthCols) /
                                         DSDMoment(.SD, n=3, dsdCols=dsdCols,
                                                   diamCols=diamCols, widthCols=widthCols)],
                         reconstructed=dsds[, DSDMoment(.SD, n=4, dsdCols=reconDSDCols,
                                                        diamCols=diamCols, widthCols=widthCols) /
                                              DSDMoment(.SD, n=3, dsdCols=reconDSDCols,
                                                        diamCols=diamCols, widthCols=widthCols)])
    if(!is.null(by))
      DmDiffs = cbind(DmDiffs, dsds[, by, with=FALSE])

    bulk = rbind(RDiffs, DmDiffs)
    bulk[, diff := reconstructed - measured]
    bulk[, relDiff := diff / measured * 100]

    ## Difference stats over the whole set.
    bulkDiffs = bulk[, list(
      bias = mean(diff),
      medRelBias = median(diff / measured * 100),
      q10RelBias = quantile(relDiff, probs=0.10),
      q25RelBias = quantile(relDiff, probs=0.25),
      q75RelBias = quantile(relDiff, probs=0.75),
      q90RelBias = quantile(relDiff, probs=0.90),
      rmse = sqrt(mean(diff^2)),
      r2 = cor(measured, reconstructed)^2), by=var]
  }

  return(list(modelFit=modelFit,
              bulkDiffs=bulkDiffs,
              momentDiffs=momentDiffs,
              normDSDs=normDSDs,
              momentComp=moments, bulkComp=bulk))
}

testModelAgainstDSDs = function(dsds, mu, c, i, j,
    dsdCols, diamCols, widthCols, seaLevelTemp, by=NULL,
    compareMoments=seq(0, 7), ...) {
    ## Test how well a generalised gamma model captures data.
    ##
    ## Args:
    ##  dsds: The DSDs on which the normalised DSDs are based.
    ##  mu, c: Parameters of the gamma model to test.
    ##  i, j: Moment orders to use.
    ##  dsdCols: Column names for DSD concentrations [mm-1 m-3] in x.
    ##  diamCols: Column names for class diameters [mm] in x.
    ##  widthCols: Column names for class widths [mm] in x.
    ##  by: Optional, also calculate stats by group?
    ##  testBulk: Compare bulk params?
    ##  testModel: Compare model?
    ##  testMoments: Compare moments?
    ##
    ## Returns: Differences between model and empirical
    ## normalised DSDs and bulk variables.
    
    ## measured DSD moments.
    if("n" %in% names(dsds))
        stop("dsds must not contain a column named 'n'")
    
    ## Calculate the ith and jth moments of each DSD.
    dsds[, ithMoment := DSDMoment(dsd=.SD, dsdCols=dsdCols,
                         diamCols=diamCols, widthCols=widthCols, n=i)]
    dsds[, jthMoment := DSDMoment(dsd=.SD, dsdCols=dsdCols,
                         diamCols=diamCols, widthCols=widthCols, n=j)]
    
    ## Reconstruct DSDs based on moments (validation) and fitted
    ## gamma parameters (training).
    momentsOnly = dsds[, c("ithMoment", "jthMoment", diamCols), with=FALSE]
    recon = reconstructDSDs(dat=momentsOnly,
        i=i, j=j, c=c, mu=mu, diamCols=diamCols)
    reconDSDCols = paste("reconClass", seq(1, length(diamCols)), sep="")
    dsds[, (reconDSDCols) := recon[, reconDSDCols, with=FALSE]]
    
    moments = NULL
    for(n in compareMoments) {
        momentName = paste("moment_", n, sep="")
        
        momentRes = data.table(order=n,
            measured=dsds[, momentName, with=FALSE][[momentName]],
            reconstructed=DSDMoment(dsds, n=n, dsdCols=reconDSDCols,
                widthCols=widthCols, diamCols=diamCols))
        
        if(!is.null(by))
            momentRes = cbind(momentRes, dsds[, by, with=FALSE])
        moments = rbind(moments, momentRes)
    }
    moments[, diff := reconstructed - measured]
    moments[, relDiff := diff / measured * 100]
    
    ## Difference stats over the whole set.
    momentDiffs = moments[, list(
        bias = mean(diff),
        medRelBias = median(relDiff),
        q10RelBias = quantile(relDiff, probs=0.10),
        q25RelBias = quantile(relDiff, probs=0.25),
        q75RelBias = quantile(relDiff, probs=0.75),
        q90RelBias = quantile(relDiff, probs=0.90),
        rmse = sqrt(mean(diff^2)),
        r2 = cor(measured, reconstructed)^2), by=order]
    
    ## Calculate R and Dm.
    stopifnot(!is.null(dsds$altitude) & !is.null(dsds$latitude))
    
    ## To get R we have to process in batches of diameter/width combinations,
    ## as well as in batches of altitude and latitude.
    diamColNames = paste(c(diamCols, widthCols, "altitude", "latitude"), collapse=",")
    
    measuredR = dsds[, list(R=getR(.SD, dsdCols=dsdCols,
                                diamCols=diamCols, widthCols=widthCols,
                                seaLevelTemp=seaLevelTemp)),
        by=diamColNames, .SDcols=names(dsds)]$R
    reconstructedR = dsds[, list(R=getR(.SD, dsdCols=reconDSDCols,
                                     diamCols=diamCols, widthCols=widthCols,
                                     seaLevelTemp=seaLevelTemp)),
        by=diamColNames, .SDcols=names(dsds)]$R
    RDiffs = data.table(var="R", measured=measuredR,
        reconstructed=reconstructedR)
    if(!is.null(by)) {
        byR = dsds[, by, with=FALSE, by=diamColNames]
        RDiffs = cbind(RDiffs, byR)
    }
    
    DmDiffs = data.table(var="Dm",
        measured=dsds[, DSDMoment(.SD, n=4, dsdCols=dsdCols,
            diamCols=diamCols, widthCols=widthCols) /
            DSDMoment(.SD, n=3, dsdCols=dsdCols,
                      diamCols=diamCols, widthCols=widthCols)],
        reconstructed=dsds[, DSDMoment(.SD, n=4, dsdCols=reconDSDCols,
            diamCols=diamCols, widthCols=widthCols) /
            DSDMoment(.SD, n=3, dsdCols=reconDSDCols,
                      diamCols=diamCols, widthCols=widthCols)])
    if(!is.null(by))
        DmDiffs = cbind(DmDiffs, dsds[, by, with=FALSE])
    
    bulk = rbind(RDiffs, DmDiffs)
    bulk[, diff := reconstructed - measured]
    bulk[, relDiff := diff / measured * 100]
    
    ## Difference stats over the whole set.
    bulkDiffs = bulk[, list(
        bias = mean(diff),
        medRelBias = median(diff / measured * 100),
        q10RelBias = quantile(relDiff, probs=0.10),
        q25RelBias = quantile(relDiff, probs=0.25),
        q75RelBias = quantile(relDiff, probs=0.75),
        q90RelBias = quantile(relDiff, probs=0.90),
        rmse = sqrt(mean(diff^2)),
        r2 = cor(measured, reconstructed)^2), by=var]
    
    return(list(bulkDiffs=bulkDiffs,
                momentDiffs=momentDiffs,
                momentComp=moments,
                bulkComp=bulk))
}

testNormalisedDSDs = function(training, validation, dsdCols,
    diamCols, widthCols, i, j, by=NULL,
    displacement="horizontal", stations=NULL,
    compareMoments=seq(0,7), cores=8, ...) {
    ## Test the error around the fitted normalised DSD.
    ##
    ## Args:
    ##   training: input data.table containing DSDs and class definitions,
    ##             used to fit gamma model.
    ##   validation: data.table containing DSDs, used to match against
    ##               normalised DSDs. Can be the same as the training set!
    ##   dsdCols: Column names for DSD concentrations [mm-1 m-3] in x.
    ##   diamCols: Column names for class diameters [mm] in x.
    ##   widthCols: Column names for class widths [mm] in x.
    ##   i, j: Moment orders to use.
    ##   by: Grouping variable(s) (default: none).
    ##   displacement: Test displacement "horizontal" or "vertical" or NULL?
    ##   compareMoments: moments to compare (default: 1 to 7).
    ##   ...: Extra (optional) arguments to testModel()
    ##
    ##
    ## Returns: modelFit, the statistics on
    ##          the whole set, normDSDs, the normalised DSDs per
    ##          "by" variable, and gammaParams, a data.table containing
    ##          the fitted mu and c values for the fitted generalised
    ##          Gamma model.
    ##
    ## The data set is divided in two, into training and validation sets.
    ## The training set is used to find gamma parameters to test.
    ## The validation set is used to see how well the gamma parameters work.

    ## Make a copy, to avoid changing the original dataset.
    training = copy(training)
    validation = copy(validation)

    ## Find gamma parameters using the training set.
    trainingSet = findNormalisedDSD(dsds=training, dsdCols=dsdCols,
        diamCols=diamCols, widthCols=widthCols, i=i, j=j, by=by, ...)
    if(identical(training, validation)) {
        normDSDs = copy(trainingSet$normDSDs)
    } else {
        ## Find normalised DSDs of the validation set.
        normDSDs = validation[, callNormalisedDSD(.SD, dsdCols=dsdCols,
            diamCols=diamCols, widthCols=widthCols, i=i, j=j), by=by]
    }

    ## Find gamma parameters for each group in the training set.
    gammaParams = trainingSet$gammaParams
    gammaByLocation = trainingSet$normDSDs[, fitNormalisedGamma(.SD, i=i, j=j, ...), by=by]

    ## Make sure all gamma parameters were properly found.
    stopifnot(!any(is.na(gammaByLocation)))

    ## Calculate validation moments.
    for(n in compareMoments) {
        validation = validation[, (paste("moment_", n, sep="")) :=
            DSDMoment(.SD, n=n, dsdCols=dsdCols, widthCols=widthCols,
                      diamCols=diamCols)]
    }

    ## Test with displacement.
    if(is.na(displacement)) {
        displacementResults = NULL
    } else if(displacement == "horizontal") {
        displacementResults = testHorizDisplacement(gammaByLocation=gammaByLocation,
            stations=stations, dsds=validation, normDSDs=normDSDs, dsdCols=dsdCols,
            widthCols=widthCols, diamCols=diamCols, i=i, j=j, cores=cores)
    } else if (displacement == "vertical") {
        displacementResults = testVertDisplacement(gammaByLocation=gammaByLocation,
            dsds=validation, normDSDs=normDSDs, dsdCols=dsdCols,
            widthCols=widthCols, diamCols=diamCols, i=i, j=j, cores=cores)
    } else {
        stop("Invalid displacement, should be 'horizontal' or 'vertical'.")
    }

    ## Perform the test using the training gamma model on the validation set.
    testResults = testModel(normDSDs=normDSDs,
        dsds=validation, mu=gammaParams$mu, c=gammaParams$c, i=i, j=j,
        diamCols=diamCols, dsdCols=dsdCols, widthCols=widthCols, by=by,
        compareMoments=compareMoments, ...)

    return(list(res=testResults, displacement=displacementResults,
                gammaParams=gammaParams,
                gammaParamsByLocation=gammaByLocation))
}

testVertDisplacement = function(gammaByLocation, dsds, normDSDs,
    dsdCols, diamCols, widthCols, i, j, cores=8, minPairs=20, distClassSize=0.1) {
    ## Test the error introduced by vertical displacement (by altitude).
    ##
    ## Args:
    ##   dsds: DSDs on which to test, by station.
    ##   dsdCols: Column names for DSD concentrations [mm-1 m-3] in x.
    ##   diamCols: Column names for class diameters [mm] in x.
    ##   widthCols: Column names for class widths [mm] in x.
    ##   i, j: Moment orders to use.
    ##   cores: Number of parallel cores to use.
    ##   distClassSize: class size for distances [km].
    ##
    ## Returns: distributions of error, per moment, per displacement
    ## distance in the vertical (in km).

    ## Loop through combinations of altitudes.
    cl = makeCluster(cores)
    registerDoParallel(cl)

    altitudes = normDSDs[, unique(altitude)]
    altitudes = altitudes[order(altitudes)]
    altitudes = altitudes[1:length(altitudes)-1]
    res = foreach(from=altitudes, .combine=rbind,
        .export=c("testModel", "generalisedGamma", "DSDMoment",
            "reconstructDSDs")) %dopar% {
                require(data.table)

                model = gammaByLocation[altitude == from]
                print(model)
                ## Compare the modelled DSD (at the "from" station)
                ## to the empirical normalised DSD at all "to" stations.
                testResults = testModel(normDSDs=normDSDs[altitude > from],
                    dsds=dsds[altitude > from], mu=model$mu, c=model$c, i=i, j=j,
                    diamCols=diamCols, dsdCols=dsdCols, widthCols=widthCols,
                    by="altitude", testBulk=FALSE, testModel=FALSE)

                momentRes = testResults$momentComp
                momentRes[, fromAlt := from]
                momentRes[, dist := abs(altitude - from)/1000] ## Work in km.
                momentRes
            }
    stopCluster(cl)

    ## Put the distances into classes.
    distClasses = round(seq(0, max(res$dist)+distClassSize, by=distClassSize), 1)
    res[, distClass := cut(dist, distClasses, right=FALSE, label=FALSE)]
    res[, dist := distClasses[distClass] + distClassSize/2]
    stopifnot(!any(is.na(res$dist)))
    
    ## Find the median relative bias and relative bias IQR for
    ## moments 0 to 7, per distance.
    results = res[,
        list(medAbsRB=median(abs(relDiff)),
             meanAbsRB=mean(abs(relDiff)),
             meanDiff=mean(diff),
             q25RelBias=quantile(abs(relDiff), probs=0.25),
             q75RelBias=quantile(abs(relDiff), probs=0.75),
             r2=cor(measured, reconstructed)^2,
             var=paste("M", order, sep=""),
             n=length(measured),
             stationPairs=length(unique(altitude))),
        by="dist,order"]

    intercepts = results[n >= minPairs, list(intercept=coef(lm("medAbsRB~dist",
                                    data=.SD))[1]), by="var"]
    slopes = results[n >= minPairs, list(slope=coef(lm("medAbsRB~dist",
                                data=.SD))[2]), by="var"]
    slopes_t = results[n >= minPairs, list(pval=summary(lm("medAbsRB~dist", data=.SD))$coefficients[2,4]), by="var"]

    setkey(intercepts, var)
    setkey(slopes, var)
    setkey(slopes_t, var)
    line = intercepts[slopes][slopes_t]
    line[, significantRelationshipFound := pval < 0.01]

    return(list(comp=results, line=line))
}

testHorizDisplacement = function(gammaByLocation, stations, dsds,
    normDSDs, dsdCols, widthCols,
    diamCols, i, j, cores=8,
    crs=CRS("+proj=longlat +datum=WGS84 +ellps=WGS84 +towgs84=0,0,0"),
    distClassSize=1, minPairs=20) {
    ## Test the error introduced by horizontal displacement (by station).
    ##
    ## Args:
    ##   stations: Station information.
    ##   dsds: Recorded DSDs to test against.
    ##   normDSDs: normalised DSDs on which to test, by station.
    ##   dsdCols: Column names for DSD concentrations [mm-1 m-3] in x.
    ##   diamCols: Column names for class diameters [mm] in x.
    ##   widthCols: Column names for class widths [mm] in x.
    ##   i, j: Moment orders to use.
    ##   cores: Processor cores to use in parallel.
    ##
    ##
    ## Returns:

  if(is.null(stations))
    stop("Stations are required for horizontal displacement test.")
  coords = stations
  coordinates(coords) = ~lon+lat
  proj4string(coords) = crs
  dists = spDists(coords, longlat=TRUE)

  cl = makeCluster(cores)
  registerDoParallel(cl)

  ## Loop through combinations of stations.
  res = foreach(from=normDSDs[, unique(station)], .combine=rbind,
      .export=c("testModel", "generalisedGamma", "DSDMoment",
          "reconstructDSDs")) %dopar% {
              require(data.table)
              model = gammaByLocation[station == from]

              ## Compare the modelled DSD (at the "from" station)
              ## to the empirical normalised DSD at the "to" station.
              testResults = testModel(normDSDs=normDSDs[station != from],
                  dsds=dsds[station != from], by="station",
                  mu=model$mu, c=model$c, i=i, j=j,
                  diamCols=diamCols, dsdCols=dsdCols, widthCols=widthCols,
                  testBulk=FALSE, testModel=FALSE)

              momentRes = testResults$momentComp
              momentRes[, fromStation := from]
              momentRes[, stationIdx := which(stations$name == station), by=station]
              fromIdx = which(stations$name == from)
              momentRes[, dist := dists[fromIdx, stationIdx]]
              momentRes[, stationIdx := NULL]
              momentRes
          }
  stopCluster(cl)

  ## Divide distances into classes.
  distClasses = seq(0, max(res$dist)+distClassSize, by=distClassSize)
  res[, distClass := cut(dist, distClasses, right=FALSE, label=FALSE)]
  res[, dist := distClasses[distClass] + distClassSize/2]
  stopifnot(!any(is.na(res$dist)))
  
  ## Find the median relative bias and relative bias IQR for
  ## moments 0 to 7, per variable, and distance.
  results = res[,
      list(medAbsRB=median(abs(relDiff)),
           meanAbsRB=mean(abs(relDiff)),
           meanRef=mean(measured),
           medRef=median(measured),
           meanDiff=mean(diff),
           medDiff=median(diff),
           q25RelBias=quantile(abs(relDiff), probs=0.25),
           q75RelBias=quantile(abs(relDiff), probs=0.75),
           r2=cor(measured, reconstructed)^2,
           var=paste("M", order, sep=""),
           n=length(measured)),
      by="dist,order"]
  results[, order := NULL]

  intercepts = results[n >= minPairs, list(intercept=coef(lm("medAbsRB~dist",
                                  data=.SD))[1]), by="var"]
  slopes = results[n >= minPairs, list(slope=coef(lm("medAbsRB~dist",
                              data=.SD))[2]), by="var"]
  slopes_t = results[n >= minPairs, list(pval=summary(lm("medAbsRB~dist", data=.SD))$coefficients[2,4]), by="var"]

  setkey(intercepts, var)
  setkey(slopes, var)
  setkey(slopes_t, var)
  line = intercepts[slopes][slopes_t]
  line[, significantRelationshipFound := pval < 0.01]

  return(list(comp=results, line=line))
}

## Calculate Kdp.
getKdp = function(x, dsdCols, diamCols, widthCols, temp, elev, ratioFunc,
                  freq=9.4, cantingSD=6) {
  stopifnot(nrow(unique(x[, diamCols, with=FALSE])) == 1)
  stopifnot(nrow(unique(x[, widthCols, with=FALSE]))== 1)
  stopifnot(length(unique(x$altitude)) == 1)
  stopifnot(length(unique(x$latitude)) == 1)

  classes = data.frame(
    min=as.numeric(x[1, diamCols, with=FALSE] - x[1, widthCols, with=FALSE]/2),
    max=as.numeric(x[1, diamCols, with=FALSE] + x[1, widthCols, with=FALSE]/2))
  rows = which(!is.na(rowMeans(classes)))
  classes = classes[rows,]
  spectra = x[, dsdCols[rows], with=FALSE]
  spectra[is.na(spectra)] = 0

  return(specificDiffPhaseFromDSD(DSD=spectra, classes=classes,
                                  freq=freq, temp=temp, incidence=elev,
                                  ratio_function=ratioFunc,
                                  cantingSD=cantingSD))
}

## Calculate rain rates.
getR = function(x, dsdCols, diamCols, widthCols, seaLevelTemp) {
  stopifnot(nrow(unique(x[, diamCols, with=FALSE])) == 1)
  stopifnot(nrow(unique(x[, widthCols, with=FALSE]))== 1)
  stopifnot(length(unique(x[["altitude"]])) == 1)
  stopifnot(length(unique(x$latitude)) == 1)
  stopifnot(length(dsdCols) == length(diamCols))
  stopifnot(length(dsdCols) == length(widthCols))

  classes = data.frame(
    min=as.numeric(x[1, diamCols, with=FALSE] - x[1, widthCols, with=FALSE]/2),
    max=as.numeric(x[1, diamCols, with=FALSE] + x[1, widthCols, with=FALSE]/2))
  rows = which(!is.na(rowMeans(classes)))
  classes = classes[rows,]
  spectra = x[, dsdCols[rows], with=FALSE]
  spectra[is.na(spectra)] = 0

  return(DSDRainrate(spectra=spectra, classes=classes,
                     altitude=x$altitude[1],
                     latitude=x$latitude[1],
                     seaLevelTemperature=seaLevelTemp))
}

## Calculate liquid water content.
getLWC = function(x, dsdCols, diamCols, widthCols, seaLevelTemp) {
  N = x[, dsdCols, with=FALSE]
  diams = x[, diamCols, with=FALSE]
  widths = x[, widthCols, with=FALSE]
  rho_w = waterDensity(x[, altitude], temperature=seaLevelTemp,
      latitude=x[, latitude])

  lwc = rowSums(diams^3*widths*N, na.rm=T)*(pi*rho_w/6)
  return(lwc)
}

## Calculate 6th moment reflectivity Z.
getZ = function(x, dsdCols, diamCols, widthCols) {
  N = x[, dsdCols, with=FALSE]
  diams = x[, diamCols, with=FALSE]
  widths = x[, widthCols, with=FALSE]

  Z = rowSums(diams^6*widths*N, na.rm=T)
  return(Z)
}

## Calculate total drop concentration.
getNt = function(x, dsdCols, diamCols, widthCols) {
  N = x[, dsdCols, with=FALSE]
  diams = x[, diamCols, with=FALSE]
  widths = x[, widthCols, with=FALSE]

  Nt = rowSums(widths*N, na.rm=T)
  return(Nt)
}

testGammaModels = function(dat, dsdCols, diamCols, widthCols, gammaByOrder,
    momentsToTest=seq(0,7)) {
    ## Test Gamma models against a dataset to see how well they recover the
    ## DSDs, bulk stats, and how well the modelled h(x) matches empirical h(x).

    dat = copy(dat)
    bulk = NULL
    moments = NULL

    ## Calculate moments.
    for(n in momentsToTest) {
        dat = dat[, (paste("moment_", n, sep="")) :=
            DSDMoment(.SD, n=n, dsdCols=dsdCols, widthCols=widthCols,
                      diamCols=diamCols)]
    }

    for(mi in momentsToTest) {
        for(mj in momentsToTest) {
            if(mi >= mj) next

            print(paste(mi, mj))

            test = testModelAgainstDSDs(dsds=dat,
                mu=gammaByOrder[i==mi & j==mj, mu],
                c=gammaByOrder[i==mi & j==mj, c], i=mi, j=mj,
                diamCols=diamCols, dsdCols=dsdCols,
                widthCols=widthCols, by=NULL,
                compareMoments=momentsToTest)

            ## Statistics on selected moments.
            momentStats = data.table(test$momentDiffs, i=mi, j=mj)
            bulkStats = data.table(test$bulkDiffs, i=mi, j=mj)
            moments = rbind(moments, momentStats)
            bulk = rbind(bulk, bulkStats)
        }
    }

    return(list(moments=moments, bulk=bulk))
}

testMomentCombinations = function(data, dsdCols, diamCols, widthCols, by=NULL,
    momentsToTest=seq(0,7), displacement=NA, keepExampleMoments=c(3,6),
    stations=NULL, ...) {
    ## Test the error around the fitted normalised DSDs for
    ## different combinations of moments.
    ##
    ## Args:
    ##   data: Data to use.
    ##   dsdCols: Column names for DSD concentrations [mm-1 m-3] in x.
    ##   diamCols: Column names for class diameters [mm] in x.
    ##   widthCols: Column names for class widths [mm] in x.
    ##   by: Grouping variable(s) (default: none).
    ##   momentsToTest: Moment orders to test combinations of.
    ##   localDistance: For comparison of moments of h(x) only,
    ##                  cut-off for "local" distance on which to find
    ##                  slopes.
    ##   ...: Optional extra arguments to testNormalisedDSDs()

    gammaParams = NULL
    moments = NULL
    displacementRes = NULL
    normDSDdisp = NULL
    normDSDslopes = NULL

    bySeason = NULL
    byRainType = NULL
    exampleTestRes = NULL

    for(i in momentsToTest) {
        for(j in momentsToTest) {
            if(i >= j) next

            print(paste(i, j))

            ## Run the test for this moment combination.
            test = testNormalisedDSDs(training=data, validation=data,
                dsdCols=dsdCols, diamCols=diamCols,
                widthCols=widthCols, i=i, j=j, by=by,
                displacement=displacement,
                testBulk=FALSE, testModel=FALSE,
                stations=stations, ...)

            ## Collect gamma parameters.
            gammaParams = rbind(gammaParams,
                test$gammaParams[, list(mu, c, resStdErr, i=i, j=j)])

            ## Statistics on selected moments.
            momentStats = data.table(test$res$momentDiffs, i=i, j=j)
            moments = rbind(moments, momentStats)

            ## Displacement stats.
            displacementRes = rbind(displacementRes,
                data.table(test$displacement$line, i=i, j=j))

            ## Comparison of moments of the normalised DSD by distance.
            normDSDcomp = NULL
            if(displacement == "horizontal") {
                stopifnot(!is.null(stations))
                normDSDcomp = distribComparisonByStation(test$res$normDSDs,
                    stations=stations, i=i, j=j)
            } else if(displacement == "vertical") {
                normDSDcomp = distribComparisonByAltitude(test$res$normDSDs,
                    i=i, j=j)
            }

            if(!is.null(normDSDcomp)) {
                normDSDdisp = rbind(normDSDdisp, data.table(i=i, j=j, normDSDcomp))
            }

            ## Keep a certain set of i,j as example output.
            if(i == keepExampleMoments[1] & j == keepExampleMoments[2]) {
                print("Keeping test results!")
                exampleTestRes = copy(test)
            }
        }
    }

    return(list(moments=moments, gammaParams=gammaParams,
                displacement=displacementRes,
                exampleMomentResults=exampleTestRes,
                exampleMoments=keepExampleMoments,
                normDSDdisp=normDSDdisp))
}

DSDMoment = function(dsd, n, w=1, dsdCols, widthCols, diamCols) {
  ## Calculate a weighted moment of the DSD.
  ##
  ## Args:
  ##   dsd: data.table including DSD, widths, diams.
  ##   n: the moment order to calculate.
  ##   w: the weight to use (default: 1).
  ##   dsdCols, widthCols, diamCols: Column names for DSD, widths, diams.
  ##
  ## Returns: (w*nth_moment) for each DSD.

  N = dsd[, dsdCols, with=FALSE]
  dD = dsd[, widthCols, with=FALSE]
  D = dsd[, diamCols, with=FALSE]

  if("n" %in% names(dsd)) {
    stop("Calculation of moment will be incorrect if data.table contains 'n' column.")
  }

  return(w * rowSums(D^n * N * dD, na.rm=TRUE))
}

reconstructDSDs = function(dat, i, j, c, mu, diamCols) {
  ## Reconstruct DSDs using only two moments and the parameters of a
  ## generalised gamma model.
  ##
  ## Args:
  ##   dat: data.table containing at least ithMoment, jthMoment,
  ##        diam and width columns.
  ##   i, j: Moment orders.
  ##   c, mu: parameters to generalised gamma model.
  ##   diamCols: Column names for drop diameters to calculate.
  ##
  ## Returns: dat, with "reconClass" and "normDiam" columns added.
  ##          reconClass contains reconstructed DSDs [mm-1 m-3].
  ##          normDiam contains second-normalised diameters [mm].

  normDSDCols = paste("reconClass", seq(1, length(diamCols)), sep="")
  normDiamCols = paste("normDiam", seq(1, length(diamCols)), sep="")

  ## Use the generalised gamma model to reconstruct the DSD.
  ## Calculate the second-normalised diameter x for each diameter D.
  dat[, (normDiamCols) := .SD[, diamCols, with=FALSE] *
        .SD[, ithMoment]^(1/(j-i)) *
        .SD[, jthMoment]^(-1/(j-i))]

  ## Calculate the generalised gamma model output for each x.
  dat[, (normDSDCols) := dat[, lapply(.SD, generalisedGamma,
                                      mu=mu, c=c, i=i, j=j),
                             .SDcols=normDiamCols]]

  ## To save memory, remove the normalised diameters.
  dat[, (normDiamCols) := NULL]

  ## Multiply by the moments to obtain the reconstructed DSDs.
  dat[, const := ithMoment^((j+1)/(j-i)) * jthMoment^((i+1)/(i-j))]
  dat[, (normDSDCols) := dat[, normDSDCols, with=FALSE] * dat$const]
  dat[, const := NULL]

  return(dat)
}

lookupHx = function(x, hxLookup) {
    ## Given a lookup table that contains classes of x and hx
    ## values for those classes, return matching values of hx
    ## for given x values.

    ## For x to be in a class it must be > min and <= max for the
    ## class.
    setkey(hxLookup, xClass)
    classes = hxLookup[, xMin]
    classWidth = hxLookup[, unique(round(xMax - xMin, 5))]
    idx = cut(x, classes, right=TRUE, labels=FALSE)
    if(any(is.na(idx))) {
        ## Ensure the missing value is at the large drop side; if so we
        ## can safely say the h(x) value is zero.
        stopifnot(x[which(is.na(idx))] > hxLookup[, min(xMin)])
    }

    stopifnot(all(abs(hxLookup[idx, xClass] - x) < 0.1, na.rm=TRUE))
    res = hxLookup[idx, hx]
    res[which(is.na(res))] = 0
    return(res)
}

generalisedGamma = function(x, mu, c, i, j) {
    ## The generalised gamma function presented in Lee et al 2004,
    ## equation 43. 
    ##
    ## Args:
    ##   x: A second-normalised diameter [-].
    ##   mu: shape parameter.
    ##   c: scaling parameter.
    ##   i, j: Moment orders.
    ##
    ## Returns: h(x), where
    ##
    ## h(x) = c Ti^((j+c*mu)/(i-j)) Tj^((-i-c*mu)/(i-j)) x^(c*mu-1) *
    ##        exp[-(Ti/Tj)^(c/(i-j)) x^c]
    ##
    ## and Ti = gamma(mu + i/c) and Tj = gamma(mu + j/c).
    
    if(c <= 0) return(rep(NA, length(x)))
    
    ## Suppress warnings on the gamma function; otherwise
    ## out-of-range errors are produced. If the value is
    ## out of range, the gamma will return Inf, and
    ## this function will return NaN.
    suppressWarnings(Ti <- gamma(mu + i/c))
    suppressWarnings(Tj <- gamma(mu + j/c))
    
    res = c * Ti^((j+c*mu)/(i-j)) * Tj^((-i-c*mu)/(i-j)) *
        x^(c*mu-1) * exp(-((Ti/Tj)^(c/(i-j)))*x^c)
    
    return(res)
}

truncateDSDs = function(dsds, dsdCols, diamCols, minD, maxD) {
  ## Set DSD concentrations in classes with centres outside
  ## a given range to zero.
  ##
  ## Args:
  ##   dsds: data.table of DSDs and diam columns.
  ##   dsdCols, diamCols: Names of DSD and diameter columns in dsds.
  ##   minD: Minimum allowed class-centre diameter.
  ##   maxD: Maximum allowed class-centre diameter.
  ##
  ## Returns: The data.frame with drop concentrations outside
  ##          the range set to zero, and the diameter classes
  ##          removed.

  concentrations = as.matrix(dsds[, dsdCols, with=FALSE])
  diameters = as.matrix(dsds[, diamCols, with=FALSE])
  resetIdx = which((diameters < minD | diameters > maxD), arr.ind=TRUE)
  concentrations[resetIdx] = 0
  diameters[resetIdx] = NA
  dsds[, (dsdCols) := data.frame(concentrations)]
  dsds[, (diamCols) := data.frame(diameters)]
  return(dsds)
}

DSDfromPPI = function(dat, mu, c, axisRatioFunc, D, dD, momentOrders=seq(0,7)) {
  ## From PPI data, reconstruct the DSD using the double-moment
  ## normalisation technique of Lee et al, JAM 2004, and
  ## relationships between Kdp and Z and DSD moments.
  ##
  ## For details see documents "Double moment normalisation" and
  ## "Moment from polarimetric vars" by Tim Raupach.
  ##
  ## Args:
  ##   dat: PPI data containing Kdp [deg. km-1], Zh [dBZ], and Zdr [dB].
  ##   mu, c: Parameters of generalised gamma model to use for the
  ##          normalised DSD.
  ##   axisRatioFunc: Which drop axis ratio function relationship to use
  ##                  for predicting moment 3? Can be Thurai, Brandes, Beard, or
  ##                  Andsager.
  ##   D: Diameters to reconstruct [mm] (default: 0.4 to 6 mm by 0.1 mm).
  ##   dD: Diameter class widths [mm] (default: all 0.05 mm).
  ##   momentOrders: Moments to calculate from reconstructed DSDs
  ##                 (default: 1-7).
  ##
  ## Returns: data.table with new columns reconstructed DSD
  ## concentrations (reconClass*), and reconstructed moments (moment_*).

  dat = copy(dat)

  ## Updated March 2018.
  
  coefsThurai = data.table(func="Thurai",     C=3.472, rC1=1, rC2=-0.073273, rC3=0.041236, rC4=-0.016342, rC5=0.002186, rC6=-5.8e-05, maxZdr=4.73)
  coefsBrandes = data.table(func="Brandes",   C=3.297, rC1=1, rC2=-0.079489, rC3=0.052234, rC4=-0.023047, rC5=0.004255, rC6=-0.000285, maxZdr=7.99)
  coefsAndsager = data.table(func="Andsager", C=3.242, rC1=1, rC2=-0.091394, rC3=0.072726, rC4=-0.035161, rC5=0.007148, rC6=-0.000534, maxZdr=7.05)
  coefsBeard = data.table(func="Beard",       C=3.24, rC1=1, rC2=-0.087146, rC3=0.053131, rC4=-0.020315, rC5=0.00291, rC6=-0.000122, maxZdr=5.09)
  ratioCoefs = rbindlist(list(coefsThurai, coefsBrandes, coefsAndsager, coefsBeard), use.names=TRUE)

  ## Choose the coefficients to use based on the drop axis ratio function.
  C = ratioCoefs[func == axisRatioFunc, C]
  rC1 = ratioCoefs[func == axisRatioFunc, rC1]
  rC2 = ratioCoefs[func == axisRatioFunc, rC2]
  rC3 = ratioCoefs[func == axisRatioFunc, rC3]
  rC4 = ratioCoefs[func == axisRatioFunc, rC4]
  rC5 = ratioCoefs[func == axisRatioFunc, rC5]
  rC6 = ratioCoefs[func == axisRatioFunc, rC6]
  maxZdr = ratioCoefs[func == axisRatioFunc, maxZdr]
  
  ## Get Zdr in linear units from Zdr.
  dat[, ZdrLin := 10^(Zdr/10)]

  ## Predict the mass-weighted mean drop axis ratio.
  dat[, rm := rC1 + rC2*Zdr + rC3*Zdr^2 + rC4*Zdr^3 + rC5*Zdr^4 + rC6*Zdr^5]
  if(nrow(dat[Zdr > maxZdr]) > 0)
      warning(paste("Warning: maximum Zdr reached on",
                    nrow(dat[Zdr > maxZdr]), "records."))
  dat[Zdr > maxZdr,  rm := 0.8]

  ## Predict moment i (3) from Kdp and Zdr.
  dat = dat[, ithMoment := (338.4 / C) * (Kdp/(1-rm))]

  ## Moment j (6) is predicted from ZhLin (in two regimes separated by ZhThresh_M6).
  ## Threshold for "low" and "high" fits is based on Zh (dBZ).
  ZhThresh_M6 = 28

  ## Parameters for M6 fit:
  M6C_low = 1.00
  M6E_low = 1.01
  M6C_high = 2.68
  M6E_high = 0.86
  
  dat = dat[Zh <= ZhThresh_M6, jthMoment := M6C_low * (10^(Zh/10))^M6E_low]
  dat = dat[Zh > ZhThresh_M6, jthMoment := M6C_high * (10^(Zh/10))^M6E_high]

  ## Add diameters.
  diamCols = paste("D", seq(1, length(D)), sep="")
  diams = data.frame(matrix(rep(D, nrow(dat)), nrow=nrow(dat), byrow=TRUE))
  names(diams) = diamCols
  dat = cbind(dat, diams)

  ## Reconstruct DSDs.
  dat = reconstructDSDs(dat=dat, i=3, j=6, c=c, mu=mu, diamCols=diamCols)

  ## Save memory by removing diameter columns.
  dat[, (diamCols) := NULL]

  ## Add moments.
  if("n" %in% names(dat)) stop("dat must not contain a column named 'n'")
  for(n in momentOrders) {
    dat[, (paste("moment_", n, sep="")) := nthMoment(.SD, n=n, w=1, widths=dD, diams=D),
        .SDcols=paste("reconClass", seq(1, length(D)), sep="")]
  }

  ## Add in diameter and width columns.
  reconDiamCols = paste("diam", seq(1, length(D)), sep="")
  reconWidthCols = paste("width", seq(1, length(D)), sep="")
  n = nrow(dat)
  diamTable = data.table(matrix(rep(D, n), nrow=n, byrow=TRUE))
  widthTable = data.table(matrix(rep(dD, n), nrow=n, byrow=TRUE))
  dat[, (reconDiamCols) := diamTable]
  dat[, (reconWidthCols) := widthTable]

  return(dat)
}

DSDfromPPI_orig = function(dat, mu, c, axisRatioFunc, D, dD, momentOrders=seq(0,7)) {
  ## From PPI data, reconstruct the DSD using the double-moment
  ## normalisation technique of Lee et al, JAM 2004, and
  ## relationships between Kdp and Z and DSD moments.
  ##
  ## Parameters as per Raupach & Berne, AMT 2017.  
  ##
  ## For details see documents "Double moment normalisation" and
  ## "Moment from polarimetric vars" by Tim Raupach.
  ##
  ## Args:
  ##   dat: PPI data containing Kdp [deg. km-1], Zh [dBZ], and Zdr [dB].
  ##   mu, c: Parameters of generalised gamma model to use for the
  ##          normalised DSD.
  ##   axisRatioFunc: Which drop axis ratio function relationship to use
  ##                  for predicting moment 3? Can be Thurai, Brandes, Beard, or
  ##                  Andsager.
  ##   D: Diameters to reconstruct [mm] (default: 0.4 to 6 mm by 0.1 mm).
  ##   dD: Diameter class widths [mm] (default: all 0.05 mm).
  ##   momentOrders: Moments to calculate from reconstructed DSDs
  ##                 (default: 1-7).
  ##
  ## Returns: data.table with new columns reconstructed DSD
  ## concentrations (reconClass*), and reconstructed moments (moment_*).

  dat = copy(dat)

  ## Updated April 2017.
  coefsThurai = data.table(func="Thurai",     C=3.456, rC1=1, rC2=-0.073624, rC3=0.041651, rC4=-0.017042, rC5=0.002498, rC6=-9.3e-05)
  coefsBrandes = data.table(func="Brandes",   C=3.311, rC1=1, rC2=-0.077672, rC3=0.047704, rC4=-0.020042, rC5=0.003505, rC6=-0.00022)
  coefsAndsager = data.table(func="Andsager", C=3.256, rC1=1, rC2=-0.090137, rC3=0.070235, rC4=-0.033933, rC5=0.006913, rC6=-0.000514)
  coefsBeard = data.table(func="Beard",       C=3.217, rC1=1, rC2=-0.087646, rC3=0.053086, rC4=-0.020336, rC5=0.002963, rC6=-0.000129)
  ratioCoefs = rbindlist(list(coefsThurai, coefsBrandes, coefsAndsager, coefsBeard), use.names=TRUE)

  ## Choose the coefficients to use based on the drop axis ratio function.
  C = ratioCoefs[func == axisRatioFunc, C]
  rC1 = ratioCoefs[func == axisRatioFunc, rC1]
  rC2 = ratioCoefs[func == axisRatioFunc, rC2]
  rC3 = ratioCoefs[func == axisRatioFunc, rC3]
  rC4 = ratioCoefs[func == axisRatioFunc, rC4]
  rC5 = ratioCoefs[func == axisRatioFunc, rC5]
  rC6 = ratioCoefs[func == axisRatioFunc, rC6]

  ## Get Zdr in linear units from Zdr.
  dat[, ZdrLin := 10^(Zdr/10)]

  ## Predict the mass-weighted mean drop axis ratio.
  dat[, rm := rC1 + rC2*Zdr + rC3*Zdr^2 + rC4*Zdr^3 + rC5*Zdr^4 + rC6*Zdr^5]
  if(nrow(dat[rm > 1 | rm <= 0]) > 0)
      warning(paste("Warning: axis ratio > 1 or <= 0 on",
                    nrow(dat[rm > 1 | rm <= 0]), "records."))
  dat[rm > 1 | rm <= 0, rm := 0.75]

  ## Predict moment i (3) from Kdp and Zdr.
  dat = dat[, ithMoment := (338.4 / C) * (Kdp/(1-rm))]

  ## Moment j (6) is predicted from ZhLin (in two regimes separated by ZhThresh_M6).
  ## Threshold for "low" and "high" fits is based on Zh (dBZ).
  ZhThresh_M6 = 28
  M6C_low = 1.00
  M6E_low = 1.01

  ## As submitted:
  ## M6C_high = 2.71
  ## M6E_high = 0.86
  M6C_high = 2.67
  M6E_high = 0.86
  dat = dat[Zh <= ZhThresh_M6, jthMoment := M6C_low * (10^(Zh/10))^M6E_low]
  dat = dat[Zh > ZhThresh_M6, jthMoment := M6C_high * (10^(Zh/10))^M6E_high]

  ## Add diameters.
  diamCols = paste("D", seq(1, length(D)), sep="")
  diams = data.frame(matrix(rep(D, nrow(dat)), nrow=nrow(dat), byrow=TRUE))
  names(diams) = diamCols
  dat = cbind(dat, diams)

  ## Reconstruct DSDs.
  dat = reconstructDSDs(dat=dat, i=3, j=6, c=c, mu=mu, diamCols=diamCols)

  ## Save memory by removing diameter columns.
  dat[, (diamCols) := NULL]

  ## Add moments.
  if("n" %in% names(dat)) stop("dat must not contain a column named 'n'")
  for(n in momentOrders) {
    dat[, (paste("moment_", n, sep="")) := nthMoment(.SD, n=n, w=1, widths=dD, diams=D),
        .SDcols=paste("reconClass", seq(1, length(D)), sep="")]
  }

  ## Add in diameter and width columns.
  reconDiamCols = paste("diam", seq(1, length(D)), sep="")
  reconWidthCols = paste("width", seq(1, length(D)), sep="")
  n = nrow(dat)
  diamTable = data.table(matrix(rep(D, n), nrow=n, byrow=TRUE))
  widthTable = data.table(matrix(rep(dD, n), nrow=n, byrow=TRUE))
  dat[, (reconDiamCols) := diamTable]
  dat[, (reconWidthCols) := widthTable]

  return(dat)
}

nthMoment = function(x, n, w, widths, diams) {
  ## A helper function to quickly calculate the nth moment of the DSD.
  ## n is the order, w is a weight.
  ## n can be a list of orders.

  stopifnot(nrow(widths) == 1)
  stopifnot(nrow(diams) == 1)
  stopifnot(length(n) == length(w))
  stopifnot(!("n" %in% names(x)))

  dD = matrix(rep(widths, nrow(x)), byrow=T, ncol=ncol(x))
  D = matrix(rep(diams, nrow(x)), byrow=T, ncol=ncol(x))
  return(w * rowSums(x * dD * D^n))
}

cleanRadarNoise = function(rad) {
    ## Take radar data, and attempt to replace noisy values of Kdp and
    ## Zdr using regressions on Zh. Updated version after that
    ## published, so that one fit is used and fits are split into two
    ## depending on Zh value, and the replaced value is based on a
    ## gradient function.
    ##
    ## Args:
    ##   rad: Radar data containing at least Zh, Kdp, Zdr.
    ##
    ## Returns: modified data set with noisy values of Kdp and Zdr replaced.
    
    ## Avoid changing the original dataset.
    dat = copy(rad)

    params = predictZdrAndKdpParams()
    ZC1 = params$ZC1
    ZE1 = params$ZE1
    ZC1high = params$ZC1high
    ZE1high = params$ZE1high
    ZdrBreak = params$ZdrBreak
    KC1 = params$KC1
    KE1 = params$KE1
    KC1high = params$KC1high
    KE1high = params$KE1high
    KdpBreak = params$KdpBreak
    
    ## Function to return proportion (0 <= p <= 1) of the value x between 
    ## values 'from' and 'to'. 0 for below, 1 for above.
    propFunc = function(x, from, to) {
        p = (x-from) / (to-from)
        p[which(p > 1)] = 1
        p[which(p < 0)] = 0
        return(p)
    }
    
    ## Calculate linear units for Zh.
    dat[, ZhLin := 10^(Zh/10)]
    dat[, ZdrLin := 10^(Zdr/10)]
    
    ## Get expected Zdr linear values, using a regression on Zh in linear units.
    dat[Zh <= ZdrBreak, expectedZdrLin := ZC1*ZhLin^ZE1]
    dat[Zh > ZdrBreak, expectedZdrLin := ZC1high*ZhLin^ZE1high]
    dat[, expectedZdr := 10*log10(expectedZdrLin)]
    
    ## Replace Zdr values when required. We know noisy values are
    ## likely below about Zdr = 0.2 dB (Bringi paper S-band),
    ## which corresponds to:
    zdrZhStart = 10^(11.6/10) ## 1% quantile of Zh with Zdr = 0.2 (linear)
    zdrZhEnd = 10^(29.3/10)   ## 99% quantile of Zh with Zdr = 0.2 (linear)
    dat[, ZhProp_Zdr := propFunc(ZhLin, zdrZhStart, zdrZhEnd)]
    dat[, ZdrLin := expectedZdrLin*(1-ZhProp_Zdr) + ZdrLin*ZhProp_Zdr]
    
    ## Also replace Zdr values that are far too low for the Zh value.
    dat[, ZdrProp := propFunc(Zdr, 0.1*expectedZdr, 0.4*expectedZdr)]
    dat[, ZdrLin := expectedZdrLin*(1-ZdrProp) + ZdrLin*ZdrProp]
    
    ## Calculate log units for Zdr.
    dat[, Zdr := 10*log10(ZdrLin)]
    
    ## Get expected Kdp values, using a regression on Zh and Zdr in linear units.
    dat[Zh <= KdpBreak, expectedKdp := KC1*ZhLin^KE1]
    dat[Zh > KdpBreak, expectedKdp := KC1high*ZhLin^KE1high]

    ## Replace Kdp values when required. We know noisy values are
    ## likely below about Kdp = 0.3 deg km-1 (Bringi paper for S-band), which
    ## coresponds to:
    kdpZhStart = 10^(33.8/10) ## 1% quantile of Zh with Kdp = 0.3.
    kdpZhEnd = 10^(43.6/10)   ## 99% quantile of Zh with Kdp = 0.3.
    dat[, ZhProp_kdp := propFunc(ZhLin, kdpZhStart, kdpZhEnd)]
    dat[, Kdp := expectedKdp*(1-ZhProp_kdp) + Kdp*ZhProp_kdp]

    ## Also replace Kdp values that are far too low for the Zh value.
    dat[, kdpProp := propFunc(Kdp, 0.1*expectedKdp, 0.4*expectedKdp)]
    dat[, Kdp := expectedKdp*(1-kdpProp) + Kdp*kdpProp]
    
    ## Recalculate Zv.
    dat[, Zv := Zh - Zdr]

    ## Clean up.
    dat[, ZhProp_Zdr := NULL]
    dat[, ZhProp_kdp := NULL]
    dat[, ZdrProp := NULL]
    dat[, kdpProp := NULL]
    return(dat)
}

cleanRadarNoise_orig = function(rad, axisFunc="Thurai") {
  ## Take radar data, and attempt to replace noisy values of Kdp and Zdr using
  ## regressions on Zh.
  ##
  ## Args:
  ##   rad: Radar data containing at least Zh, Kdp, Zdr.
  ##
  ## Returns: modified data set with noisy values of Kdp and Zdr replaced.

  ## Avoid changing the original dataset.
  dat = copy(rad)

  ## Regression parameters for Zdr.
  zdrParams_thurai   = data.table(func="Thurai",   ZC1=0.030, ZE1=0.436)
  zdrParams_brandes  = data.table(func="Brandes",  ZC1=0.027, ZE1=0.449)
  zdrParams_andsager = data.table(func="Andsager", ZC1=0.043, ZE1=0.377)
  zdrParams_beard = data.table(func="Beard",       ZC1=0.048, ZE1=0.384)
  zdrParams = rbindlist(list(zdrParams_thurai, zdrParams_brandes,
                             zdrParams_andsager, zdrParams_beard), use.names=TRUE)
  ZC1 = zdrParams[func==axisFunc, ZC1]
  ZE1 = zdrParams[func==axisFunc, ZE1]

  ## Regression parameters for Kdp.
  kdpParams_thurai   = data.table(func="Thurai",   KC1=0.00010, KE1=1.055, KE2=-3.156)
  kdpParams_brandes  = data.table(func="Brandes",  KC1=0.00010, KE1=1.038, KE2=-2.723)
  kdpParams_andsager = data.table(func="Andsager", KC1=0.00017, KE1=0.976, KE2=-3.251)
  kdpParams_beard = data.table(func="Beard",       KC1=0.00017, KE1=1.013, KE2=-3.338)
  kdpParams = rbindlist(list(kdpParams_thurai, kdpParams_brandes,
                             kdpParams_andsager, kdpParams_beard), use.names=TRUE)
  KC1 = kdpParams[func==axisFunc, KC1]
  KE1 = kdpParams[func==axisFunc, KE1]
  KE2 = kdpParams[func==axisFunc, KE2]

  ## Calculate linear units for Zh.
  dat[, ZhLin := 10^(Zh/10)]

  ## Replace noisy Zdr values, using a regression on Zh in linear units.
  dat[, expectedZdr := ZC1*ZhLin^ZE1]
  dat[Zh < 37 | Zdr < 0.2, Zdr := expectedZdr]

  ## Calculate linear units for Zdr.
  dat[, ZdrLin := 10^(Zdr/10)]

  ## Replace noisy Kdp values, using a regression on Zh and Zdr in linear units.
  dat[, expectedKdp := KC1*ZhLin^KE1*ZdrLin^KE2]
  dat[Zh < 37 | Kdp < 0.3, Kdp := expectedKdp]

  ## Recalculate Zv.
  dat[, Zv := Zh - Zdr]

  return(dat)
}

testDSDReconPars = function(dat, axisfuncName, gammaParams, classes, dsdCols, seaTemp=15) {
  ## Test the DSD reconstruction using Parsivel data.
  ##
  ## Args:
  ##   dat: data.table of Parsivel data.
  ##   axisfuncName: Name of drop axis ratio function to use.
  ##   gammaParams: Generalised gamma parameters (mu and c).
  ##   classes: Classes on which to perform comparisons.
  ##   dsdCols: column names corresponding to classes.
  ##
  ## Returns: comparison results and plots in a list.

    ## stop("Set sea level temp!!")
    
  ## Define drop sizes and widths to reconstruct from classes.
  D=rowMeans(classes)
  dD=apply(classes, 1, diff)

  stopifnot(length(dsdCols) == nrow(classes))
  
  datTest = dat[, list(POSIXtime, station, Kdp, Zh, Zdr)]

  ## Reconstruct DSDs from PPI scans.
  datRecon = DSDfromPPI(dat=datTest, axisRatioFunc=axisfuncName,
                        mu=gammaParams$mu, c=gammaParams$c, D=D, dD=dD)

  reconCols = paste("reconClass", seq(1,length(D)), sep="")
  stopifnot(identical(dat$POSIXtime, datRecon$POSIXtime))
  stopifnot(identical(dat$station, datRecon$station))
  datRecon$alt = dat$alt
  datRecon$lat = dat$lat
  datRecon[, R := DSDRainrate(spectra=.SD, classes=classes,
                   altitude=alt, latitude=lat,
                   seaLevelTemperature=seaTemp),
           by=c("alt","lat"), .SDcols=reconCols]

  datMoments = copy(dat)
  for(n in seq(0, 7)) {
      datMoments[, (paste("moment_", n, sep="")) :=
                 nthMoment(.SD, n=n, w=1, widths=dD, diams=D),
                 .SDcols=dsdCols]
  }

  datMoments[, Dm := moment_4 / moment_3]
  datRecon[, Dm := moment_4 / moment_3]

  ## Select columns.
  datReconMoments = datRecon[, list(time=POSIXtime, station,
      moment_0, moment_1, moment_2, moment_3, moment_4, moment_5,
      moment_6, moment_7, R, Dm)]
  datMoments = datMoments[, list(time=POSIXtime, station,
      moment_0, moment_1, moment_2, moment_3, moment_4, moment_5,
      moment_6, moment_7, R, Dm)]

  ## Turn data.tables into long forms.
  datReconMoments = melt(datReconMoments, id.vars=c("time","station"))
  datMoments = melt(datMoments, id.vars=c("time","station"))

  ## Join together the two sets to compare.
  setnames(datReconMoments, "value", "reconstructed")
  setnames(datMoments, "value", "measured")
  setkey(datReconMoments, time, station, variable)
  setkey(datMoments, time, station, variable)

  datResults = datReconMoments[datMoments, nomatch=0]
  datResults = datResults[!is.na(measured) & !is.na(reconstructed)]
  datResults = datResults[!is.infinite(measured) & !is.infinite(reconstructed)]

  momentsPlot = ggplot(datResults, aes(x=measured, y=reconstructed)) +
    geom_point() + facet_wrap(~variable, scale="free") +
    geom_abline(intercept=0, slope=1, colour="red")

  datResults[, diff := reconstructed - measured]
  datResults[, relDiff := diff / abs(measured) * 100]

  densityPlot = ggplot(datResults, aes(x=relDiff)) +
    geom_density(aes(colour=variable)) +
    scale_x_continuous(limits=c(-100, 100))

  return(list(res=datResults, momentsPlot=momentsPlot,
              densityPlot=densityPlot, recon=datRecon))
}

compareToInstrument = function(reconstructed, instrument,
                               by="altitudeClass", instDSDCols, instWidthCols, instDiamCols,
                               reconDSDCols, reconDiamCols, reconWidthCols, seaLevelTemp) {
  ## Compare reconstructed DSDs to measured DSDs.
  ##
  ## Args:
  ##  reconstructed: The reconstructed DSDs to compare.
  ##  instrument: Measured DSDs to compare.
  ##  by: Sort by what variable?
  ##  inst{DSD|Width|Diam}Cols: instrument concentration, width,
  ##                            and diameter class column names.
  ##                            Must be already present.
  ##  recon{DSD|Width|Diam}Cols: reconstructed concentration, width,
  ##                             and diameter class column names.
  ##
  ## reconstructed and instrument must both contain "time" and the
  ## columns in "by".
  ##
  ## Returns: Comparison stats and plots.

  inst = copy(instrument)
  reconDSDs = copy(reconstructed)

  ## Calculate R for PPI-retrieved DSDs.
  setkey(reconDSDs, time)
  setnames(reconDSDs, "lat", "latitude")

  reconDSDs = reconDSDs[, R := getR(.SD, dsdCols=reconDSDCols,
                                diamCols=reconDiamCols,
                                widthCols=reconWidthCols,
                                seaLevelTemp=seaLevelTemp),
                        by=altitude, .SDcols=names(reconDSDs)]

  ## Calculate mean/median DSDs per group.
  instSets = inst[, c("time", by), with=FALSE]
  PPIsets = reconDSDs[, c("time", by), with=FALSE]
  setkeyv(instSets, c("time", by))
  setkeyv(PPIsets, c("time", by))
  sets = instSets[PPIsets, nomatch=0]

  setkeyv(reconDSDs, c("time", by))
  setkeyv(inst, c("time", by))
  PPIDSDsForMean = reconDSDs[sets, nomatch=0]
  instDSDsForMean = inst[sets, nomatch=0]

  ## Calculate moments.
  instMoments = copy(inst)
  stopifnot(!("n" %in% names(instMoments)))
  stopifnot(!("n" %in% names(reconDSDs)))
  for(n in seq(0, 7)) {
    instMoments = instMoments[, (paste("moment_", n, sep="")) :=
                                DSDMoment(.SD, n=n, dsdCols=instDSDCols, widthCols=instWidthCols,
                                          diamCols=instDiamCols)]

    reconDSDs = reconDSDs[, (paste("moment_", n, sep="")) :=
                            DSDMoment(.SD, n=n, dsdCols=reconDSDCols, widthCols=reconWidthCols,
                                      diamCols=reconDiamCols)]
  }

  ## Select columns. If there is no Zh, use Z.
  if(!("Zh" %in% names(instMoments)))
    instMoments[, Zh := Z]

  ## Calculate Dm.
  instMoments[, Dm := moment_4 / moment_3]
  reconDSDs[, Dm := moment_4 / moment_3]

  instMoments = instMoments[, c("time", "Zh",
                                paste("moment_", seq(0,7), sep=""), "R", "Dm", by),
                            with=FALSE]
  reconMoments = reconDSDs[, c("time", "Zh",
                               paste("moment_", seq(0,7), sep=""), "R", "Dm", by),
                           with=FALSE]

  ## Turn data.tables into long forms.
  reconMoments = melt(reconMoments, id.vars=c("time", by))
  instMoments = melt(instMoments, id.vars=c("time", by))

  setnames(instMoments, "value", "Measured")
  setkeyv(instMoments, c("time", by, "variable"))

  setnames(reconMoments, "value", "PPI")
  setkeyv(reconMoments, c("time", by, "variable"))

  ## Join together the two sets to compare.
  recon = reconMoments[instMoments, nomatch=0]

  ## Calculate difference statistics.
  recon[, diff := PPI - Measured]
  recon[, relDiff := diff / Measured * 100]

  scatterplot = ggplot(recon, aes(x=Measured, y=PPI)) + geom_point() +
    geom_abline(intercept=0, slope=1, colour="red") +
    coord_fixed() +
    facet_wrap(~variable, scales="free") +
    scale_x_continuous(trans="log10") +
    scale_y_continuous(trans="log10")

  return(list(comp=recon, plot=scatterplot,
              PPIDSDsForMean=PPIDSDsForMean,
              instDSDsForMean=instDSDsForMean))
}

compareDSDsByDisplacement = function(dat, dsdCols, diamCols, widthCols,
    by=NULL, stations=NULL, displacement="horizontal") {

    dat = copy(dat)
    ## Calculate moments zero to seven.
    for(k in seq(0, 7)) {
        dat[, moment := DSDMoment(.SD, n=k, w=1, dsdCols=dsdCols,
                         widthCols=widthCols, diamCols=diamCols)]
        setnames(dat, "moment", paste("moment_", k, sep=""))
    }

    results = compareDSDsByDisplacement_ind(dat=dat, stations=stations, displacement=displacement)

    resultsBySeason = NULL
    if(length(unique(dat$season)) == 1)
        resultsBySeason[[unique(dat$season)]] = results
    else {
        for(s in unique(dat$season)) {
            resultsBySeason[[s]] = compareDSDsByDisplacement_ind(dat=dat[season == s],
                               stations=stations, displacement=displacement)
        }
    }

    resultsByRainClass = NULL
    if(length(unique(dat$rainClass)) == 1)
        resultsByRainClass[[unique(dat$rainClass)]] = results
    else {
        for(s in unique(dat$rainClass)) {
            resultsByRainClass[[s]] = compareDSDsByDisplacement_ind(dat=dat[rainClass == s],
                                  stations=stations, displacement=displacement)
        }
    }

    return(list(all=results, resultsBySeason=resultsBySeason,
                resultsByRainClass=resultsByRainClass))
}

compareDSDsByDisplacement_ind = function(dat, stations=NULL, displacement="horizontal") {
    if(displacement == "horizontal") {
        stopifnot(!is.null(stations))
        res = dsdComparisonByStation(dat, stations=stations)
    } else {
        stopifnot(displacement == "vertical")
        res = dsdComparisonByAltitude(dat)
    }

    lines = res[, list(slope=coef(lm(medAbsRB~dist))[2]), by=momentOrder]
    return(list(slopes=data.table(lines), results=res))
}

dsdComparisonByStation = function(dsds, stations, distClassSize=1,
    crs=CRS("+proj=longlat +datum=WGS84 +ellps=WGS84 +towgs84=0,0,0")) {
  ## Calculate distances and call dsdComparisonByDistance for stations.
  s = copy(stations)
  coordinates(s) = ~lon+lat
  proj4string(s) = crs
  d = spDists(s, longlat=TRUE)

  ## Distances in KM.
  dists = NULL
  for(f in seq(1, length(s$name))) {
      for(t in seq(1, length(s$name))) {
          if(f == t) next
          dists = rbind(dists, data.table(from=s$name[f],
              to=s$name[t], dist=d[f,t]))
      }
  }

  return(dsdComparisonByDistance(dsds, by="station", dists=dists,
                                 distClassSize=distClassSize))
}

dsdComparisonByAltitude = function(dsds, distClassSize=0.1) {
  ## Calculate distances and call dsdComparisonByDistance for altitudes.

  ## Distances in km.
  dists = NULL
  for(f in unique(dsds$altitude)) {
    for(t in unique(dsds$altitude)) {
      if(f == t) next
      dists = rbind(dists, data.table(from=f, to=t, dist=abs((f-t)/1000)))
    }
  }

  return(dsdComparisonByDistance(dsds, by="altitude", dists=dists,
                                 distClassSize=distClassSize))
}

dsdComparisonByDistance = function(dsds, by, dists, distClassSize=1) {
    ## Compare DSDs by location, by comparing
    ## moments from zero to seven.
    ##
    ## Args:
    ##   dsds: Normalised DSDs containing hx, x, and "by".
    ##   by: The column specifying location.
    ##   dists: Distances between each two values of "by".
    ##   distClassSize: size of classes in which to compare DSDs.
    ##
    ## Returns: comparison of DSDs between locations by moment order.

    dsds = copy(dsds)
    colsToCompare = paste("moment_", seq(0, 7), sep="")
    moments = dsds[, c("POSIXtime", by, colsToCompare), with=FALSE]
    setnames(moments, by, "key")

    moments = melt(moments, id.vars=c("POSIXtime", "key"))
    moments = moments[, momentOrder := as.numeric(str_extract(variable, "[0-9]"))]
    setnames(moments, "value", "moment")

    res = NULL
    setnames(dsds, by, "key")
    for(fromLoc in unique(dsds[, key])) {
        from = moments[key == fromLoc, ]
        to = moments[key != fromLoc]
        setkey(to, key)

        comp = to[, compareMoments(from=from, to=copy(.SD)), by="key"]
        setnames(comp, "key", "to")
        res = rbind(res, data.table(comp, from=fromLoc))
    }

    ## Add distances.
    setkey(dists, from, to)
    setkey(res, from, to)
    res = dists[res]

    ## Divide distances into classes.
    if(!is.na(distClassSize)) {
        ## Round to one decimal place to avoid seq rounding problems.
        distClasses = round(seq(0, max(res$dist)+distClassSize, by=distClassSize), 1)
        res[, distClass := cut(dist, distClasses, label=FALSE, right=FALSE)]
        res[, dist := distClasses[distClass] + distClassSize/2]
        stopifnot(!any(is.na(res$dist)))
    }

    res = res[, list(medAbsRB=median(relDiff),
        RBIQR=IQR(relDiff),
        RBq25=quantile(relDiff, probs=0.25),
        RBq75=quantile(relDiff, probs=0.75),
        n=length(relDiff)), by="dist,momentOrder"]

    return(res)
}

compareMoments = function(from, to) {
  ## Compare normalised DSDs in classes of x, by comparing the
  ## values of h(x) for corresponding classes.

  setkey(from, POSIXtime, momentOrder)
  setkey(to, POSIXtime, momentOrder)
  setnames(to, "moment", "toMoment")

  moments = from[to, nomatch=0]
  moments[, relDiff := abs(toMoment - moment) / moment * 100]
  results = moments[, relDiff, by=momentOrder]

  ## results = moments[, list(medAbsRB=median(relDiff),
  ##     RBIQR=IQR(relDiff),
  ##     RBq25=quantile(relDiff, probs=0.25),
  ##     RBq75=quantile(relDiff, probs=0.75),
  ##     n=length(relDiff)), by=momentOrder]

  return(results)
}

distribComparisonByStation = function(normDSDs, stations, i, j, distClassSize=1,
    crs=CRS("+proj=longlat +datum=WGS84 +ellps=WGS84 +towgs84=0,0,0")) {
    ## Calculate distances and call distribComparisonByDistance for stations.

    s = copy(stations)
    coordinates(s) = ~lon+lat
    proj4string(s) = crs
    d = spDists(s, longlat=TRUE)

    ## Distances in KM.
    dists = NULL
    for(f in seq(1, length(s$name))) {
        for(t in seq(1, length(s$name))) {
            if(f == t) next
            dists = rbind(dists, data.table(from=s$name[f],
                to=s$name[t], dist=d[f,t]))
        }
    }

    return(distribComparisonByDistance(normDSDs, by="station",
                                       dists=dists, i=i, j=j,
                                       distClassSize=distClassSize))
}

distribComparisonByAltitude = function(normDSDs, i, j, distClassSize=0.1) {
  ## Calculate distances and call distribComparisonByDistance for altitudes.

  ## Distances in km.
  dists = NULL
  for(f in unique(normDSDs$altitude)) {
    for(t in unique(normDSDs$altitude)) {
      if(f == t) next
      dists = rbind(dists, data.table(from=f, to=t, dist=abs((f-t)/1000)))
    }
  }

  return(distribComparisonByDistance(normDSDs, by="altitude",
                                     dists=dists, i=i, j=j,
                                     distClassSize=distClassSize))
}

distribComparisonByDistance = function(normDSDs, by, dists, i, j, distClassSize=1) {
    ## Compare normalised DSDs by location, by comparing
    ## moments from zero to seven.
    ##
    ## Args:
    ##   normDSDs: Normalised DSDs containing hx, x, and "by".
    ##   by: The column specifying location.
    ##   dists: Distances between each two values of "by".
    ##   i, j: Moment orders used by normDSDs.
    ##
    ## Returns:

    normDSDs = copy(normDSDs)
    setnames(normDSDs, by, "key")
    normDSDs = normDSDs[hx > 0 & !is.na(x)]

    ## Calculate moments from zero to seven.
    moments = NULL
    for(n in seq(0, 7)) {
        moments = rbind(moments,
            normDSDs[, list(momentOrder=n,
                            moment=sum(hx*x^n*xWidth)), by=c("key","POSIXtime")])
    }

    res = NULL
    for(fromLoc in unique(moments[, key])) {
        from = moments[key == fromLoc]
        to = moments[key != fromLoc]
        setkey(to, key)

        comp = to[, compareMoments(from=from, to=copy(.SD)), by="key"]

        setnames(comp, "key", "to")
        res = rbind(res, data.table(comp, from=fromLoc))
    }

    ## Add distances.
    setkey(dists, from, to)
    setkey(res, from, to)
    res = dists[res]

    if(!is.na(distClassSize)) {
        ## Divide distances into classes.
        ## NOTE that we round to 1 decimal place to avoid rounding error.
        distClasses = round(seq(0, max(res$dist)+distClassSize, by=distClassSize), 1)
        res[, distClass := cut(dist, distClasses, label=FALSE, right=FALSE)]
        res[, dist := distClasses[distClass] + distClassSize/2]
        stopifnot(!any(is.na(res$dist)))
    }

    res = res[, list(medAbsRB=median(relDiff),
        RBIQR=IQR(relDiff),
        RBq25=quantile(relDiff, probs=0.25),
        RBq75=quantile(relDiff, probs=0.75),
        n=length(relDiff)), by="dist,momentOrder"]

    return(res)
}

classNormDSDs = function(dat, i, j, dsdCols, diamCols, widthCols, xClassWidth=0.1) {
    ## Find normalised DSDs and resample them into classes
    ## of x from startPoint to the maximum.
    ##
    ## Args:
    ##   dat: The data.table of DSDs, including POSIXtime,
    ##        station and altitude.
    ##   i, j: Moment combination to use for normalisation.
    ##   xClassWidth: Divide x into classes of this width (default: 0.1).
    ##
    ## Returns: data.table with POSIXtime, station, altitude,
    ##          and classed normalised DSDs.

    ## Find normalised DSDs.
    normDSDs = dat[, callNormalisedDSD(.SD, dsdCols=dsdCols,
        diamCols=diamCols, widthCols=widthCols, i=i, j=j), by="station,altitude"]
    normDSDs = normDSDs[!is.na(x)]

    ## Class normalised DSD by classes of x.
    normDSDs[, minX := x - xWidth / 2]
    normDSDs[, maxX := x + xWidth / 2]
    xclasses = seq(startPoint, normDSDs[, max(maxX)+xClassWidth], by=xClassWidth)

    classedNormDSDs = NULL
    for(k in seq(length(xclasses), 1)) {
        name = paste("Xclass", k, sep="")

        newMin = xclasses[k]
        newMax = newMin + xClassWidth

        ## Select rows to include.
        x = normDSDs[minX < newMax & maxX > newMin]
        if(nrow(x) == 0) {
            x = normDSDs[minX < newMax & maxX > newMin, NA,
                by=c("POSIXtime", "station", "altitude")]
        } else {
            ## Determine overlapping proportion of each bin.
            ## This should be the proportion of the measurement in x that covers
            ## the new bin.
            x[, incWidth := (min(maxX, newMax) - max(minX, newMin)), by=1:length(minX)]
            x = x[, sum(hx * incWidth) / xClassWidth,
                by=c("POSIXtime", "station", "altitude")]
        }

        setnames(x, "V1", name)
        setkeyv(x, c("POSIXtime", "station", "altitude"))
        if(k == length(xclasses)) {
            classedNormDSDs = x
            next
        }
        else {
            classedNormDSDs = merge(x, classedNormDSDs, all=TRUE)
        }
    }

    ## ## Testing code: two zeroth moments should be the same.
    ## cols = paste("Xclass", seq(1, length(xclasses)), sep="")
    ## sum(classedNormDSDs[1, cols, with=FALSE]*xClassWidth, na.rm=T)
    ## normDSDs[station == classedNormDSDs[1, station] &
    ##          POSIXtime == classedNormDSDs[1, POSIXtime],
    ##           sum(hx*xWidth)]

    return(list(normDSDs=classedNormDSDs,
                D=(xclasses + xClassWidth / 2),
                dD=rep(xClassWidth, length(xclasses))))
}

resampleDSDClasses = function(x, D, dD, diamCols, widthCols, dsdCols) {
    ## Resample MRR or MXPOL data into given drop size classes.
    ## diamCols, widthCols, dsdCols give existing columns.
    ## D and dD give classes to resample into.
    ##
    ## Args:
    ##   x: The data.table of DSDs to resample [mm-1 m-3].
    ##   D: class centres to resample into [mm].
    ##   dD: class widths for each class [mm].
    ##   diamCols, widthCols, dsdCols: Column names for diameters,
    ##                                 widths, and concentrations in x.
    ##
    ## Returns: a data.table of resampled DSDs.

    stopifnot(length(diamCols) == length(widthCols))
    stopifnot(length(diamCols) == length(dsdCols))

    diams = x[, diamCols, with=FALSE]
    widths = x[, widthCols, with=FALSE]

    minDiams = as.matrix(diams - widths / 2)
    maxDiams = as.matrix(diams + widths / 2)

    ## Convert DSD concentrations from mm-1 m-3 to m-3.  (This is not
    ## required any more because incWidths does this conversion).
    dsds = x[, dsdCols, with=FALSE]

    minD = D - dD/2
    maxD = D + dD/2

    ## Count up the number of drops per class.
    res = NULL
    for(i in seq(1, length(minD))) {
        idx = which(minDiams < maxD[i] & maxDiams > minD[i], arr.ind=TRUE)

        incWidths = pmin(maxDiams, maxD[i]) - pmax(minDiams, minD[i])
        stopifnot(all(incWidths[idx] > 0))

        N = matrix(NA, nrow=nrow(dsds), ncol=ncol(dsds))
        N[idx] = as.matrix(dsds)[idx] * incWidths[idx]
        res = cbind(res, rowSums(N, na.rm=TRUE))
    }

    ## Convert back from m-3 to mm-1 m-3.
    newWidths = matrix(rep(dD, nrow(res)), ncol=length(dD), byrow=TRUE)
    res = res / newWidths
    res[is.na(res)] = 0

    res = data.frame(res)
    names(res) = paste("class", seq(1, length(D)), sep="")

    return(data.table(res))
}

classNormDSD = function(dat, xClassSize=0.2, startAt=0) {
    dat = copy(dat[!is.na(x)])
    xClasses = seq(startAt, max(dat$x)+xClassSize, by=xClassSize)
    dat[, xClass := cut(x, xClasses, labels=FALSE)]
    dat = dat[, list(median=median(hx), n=length(hx),
        q25=quantile(hx, probs=0.25),
        q75=quantile(hx,probs=0.75)), by=xClass]
    dat[, x := xClasses[xClass]+xClassSize/2]
    return(dat)
}

predictZdrAndKdpParams_Raupach2017 = function(axisFunc="Thurai") {
    ## Return regression parameters for predicting Zdr and Kdp from Zh,
    ## as specified in Raupach AMT 2017.
    ##
    ## Args:
    ##   axisFunc: The raindrop axis ratio function to assume.
    ##
    ## Returns: list containing ZC1, ZE1, KC1, KE1, KE2.

    ## Check for valid axis function definition.
    stopifnot(axisFunc %in% c("Thurai", "Brandes", "Andsager", "Beard"))
    
    ## Regression parameters for Zdr.
    zdrParams_thurai   = data.table(func="Thurai",   ZC1=0.030, ZE1=0.436)
    zdrParams_brandes  = data.table(func="Brandes",  ZC1=0.027, ZE1=0.449)
    zdrParams_andsager = data.table(func="Andsager", ZC1=0.043, ZE1=0.377)
    zdrParams_beard = data.table(func="Beard",       ZC1=0.048, ZE1=0.384)
    zdrParams = rbindlist(list(zdrParams_thurai, zdrParams_brandes,
        zdrParams_andsager, zdrParams_beard), use.names=TRUE)
    ZC1 = zdrParams[func==axisFunc, ZC1]
    ZE1 = zdrParams[func==axisFunc, ZE1]
    
    ## Regression parameters for Kdp.
    kdpParams_thurai   = data.table(func="Thurai",   KC1=0.00010, KE1=1.055, KE2=-3.156)
    kdpParams_brandes  = data.table(func="Brandes",  KC1=0.00010, KE1=1.038, KE2=-2.723)
    kdpParams_andsager = data.table(func="Andsager", KC1=0.00017, KE1=0.976, KE2=-3.251)
    kdpParams_beard = data.table(func="Beard",       KC1=0.00017, KE1=1.013, KE2=-3.338)
    kdpParams = rbindlist(list(kdpParams_thurai, kdpParams_brandes,
        kdpParams_andsager, kdpParams_beard), use.names=TRUE)
    KC1 = kdpParams[func==axisFunc, KC1]
    KE1 = kdpParams[func==axisFunc, KE1]
    KE2 = kdpParams[func==axisFunc, KE2]
    
    params = list(ZC1=ZC1, ZE1=ZE1, KC1=KC1, KE1=KE1, KE2=KE2)
    return(params)
}

predictZdrAndKdpParams = function() {
    ## Return regression parameters for predicting Zdr and Kdp from Zh,
    ## most up to date version.
    ## 
    ## Returns: list containing ZC1, ZE1, ZC1high, ZE1high, ZdrBreak,
    ## KC1, KE1, KE2, KC1, KE1, KE2, kdpBreak.

    ## Regression parameters for Zdr.
    ZC1 = 0.979
    ZE1 = 0.018
    ZC1high = 0.599
    ZE1high = 0.097
    ZdrBreak = 27.01

    ## Regression parameters for Kdp.
    ##KC1 = 0.00013
    ##KE1 = 1.016
    ##KE2 = -2.973

    KC1 = 0.00007
    KE1 = 1.152
    KC1high = 0.00066
    KE1high = 0.701
    KdpBreak = 21.61
    
    params = list(ZC1=ZC1, ZE1=ZE1, ZC1high=ZC1high, ZE1high=ZE1high, ZdrBreak=ZdrBreak,
        KC1=KC1, KE1=KE1, KC1high=KC1high, KE1high=KE1high, KdpBreak=KdpBreak)
    return(params)
}

