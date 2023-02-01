# dsd_interpolation_functions.R
#
# Functions for interpolating and simulating DSDs. Also includes
# functions for calculating and applying the "dry drift", as described
# by Schleiss_JHR_2014 and Schleiss_WRR_2014.
#
# Author: Tim Raupach <tim.raupach@epfl.ch>

require(data.table)
require(minpack.lm)
require(stringr)
require(RANN)
require(car)
require(doParallel)
require(parallel)
require(foreach)
require(pryr)
require(testit)
source("library/radar_functions.R")
source("library/HYMEX_SOP_plot_functions.R")
source("library/variogram_functions.R")

stationComparisons = function(stations, predictions, data, ts,
                              dsdColNames=paste("class", seq(1,32), sep=""),
                              textSize=12, logScale=TRUE,
                              classDiameters=rowMeans(get.classD()),
                              maxDiamToPlot=6.5) {
  # Compare the DSD predicted at grid points closest to stations, with the
  # DSD recorded at those stations.
  #
  # Args:
  #   stations: The stations to compare (use eg stationsDefinition_2012()).
  #   predictions: The predictions - gridded DSDs
  #                (use interpolateDetrendedDSDs())..
  #   data: The original data with DSD per station per time.
  #   ts: The timestep we are comparing (POSIXct, UTC).
  #   dsdCols: Columns for the DSD in data (default class1 to class32).
  #   textSize: Text size for the plots (default: 10).
  #   logScale: Plot the y-axis using a log scale? (Default: TRUE).
  #   classDiameters: Centre diameters of each diameter class (Default:
  #                   Parsivel values).
  #   maxDiamToPlot: Maximum centre diameter to show on the plot [mm]
  #                  (Default: 6.5).
  #
  # Returns: A list containing dsds (DSDs at each point, predicted and
  #          measured), and plots, a list of plots, one per station.

  # Make sure data is a data.table.
  if(class(data)[1] != "data.frame")
    data = data.frame(data)

  # Get coordinates of stations.
  stationCoords = stations
  stationCoords$station = stationCoords$name
  stationCoords$lat = NULL
  stationCoords$lon = NULL
  stationCoords = addStationLocations(stationCoords, stations)

  # Find the closest grid entries to each station.
  closestGridDSDs = data.table(closestGridPoints(predictions, stationCoords))

  # For each station, find the original DSD for this timestep and compare
  # to the gridded version.
  stationPlots = list()
  dsds = list()
  for(s in seq(1, length(stationCoords$name))) {
    station = stationCoords$name[s]

    # Get original DSD.
    idx = which(data$POSIXtime == ts & data$station == station)
    original = data[idx, dsdCols]
    if(length(idx) == 0)
      next

    # Get gridded DSD and distance from station.
    grid = closestGridDSDs[s, dsdColNames, with=FALSE]
    statDist = round(closestGridDSDs$dist[s], 2)

    # Plot a comparison between the station and the closest grid DSD.
    plotData = rbind(data.frame(name="Original",
                                class=seq(1, length(original)),
                                diam=classDiameters,
                                melt(original)),
                     data.frame(name="Grid",
                                class=seq(1, length(grid)),
                                diam=classDiameters,
                                melt(data.frame(grid))))

    stationPlots[[station]] =
      ggplot(plotData, aes(x=diam, y=value, group=name)) +
      geom_line(aes(colour=name), size=.75) + theme_bw(textSize) +
      scale_colour_discrete(name="DSD") +
      scale_x_continuous(limits=c(0, maxDiamToPlot))

    ylabel = parse(text="N(D)~group('[',mm^{-1}~m^{-3},']')")
    if(logScale) {
      stationPlots[[station]] = stationPlots[[station]] +
        scale_y_continuous(trans="log10")
      ylabel = "log10(N(D))"
    }

    stationPlots[[station]] = stationPlots[[station]] +
      labs(x="Equivolume diameter [mm]",
         y=ylabel,
         title=paste(station, " (", statDist, " m)", sep=""))

    # Save DSDs for potential use later.
    dsds[[station]] = list(predicted=grid, measured=original)
  }

  return(list(plots=stationPlots, dsds=dsds))
}

readDataForInterpolation = function(dsdFile, start, end,
                                    stationCol=1, timeCol=2,
                                    dsdCols=seq(7,38)) {
  # Read in Parsivel data for interpolation.
  #
  # Args:
  #   dsdRDataFile: The RData file to read.
  #   start, end: Times to read (inclusive, POSIXct UTC).
  #   stationCol: Which column contains the station? (Default: 1).
  #   timeCol: Which column contains the time? (Default: 2).
  #   dsdCols: Which columns correspond to the DSD? (Default: 5:36)
  #
  # Returns: data.frame of data with columns time, station, DSD. Rows
  #          containing NAs or that are all zero are removed.

  dataName = load(dsdFile)
  parsivel = get(dataName)
  rm(list=dataName)

  # Define where in the data the DSD is, and subset for those columns.
  parsivel = parsivel[, c(timeCol, stationCol, dsdCols)]

  # Subset for event times.
  idx = which(parsivel$POSIXtime >= start & parsivel$POSIXtime <= end)
  data = parsivel[idx,]
  dsdCols=seq(3,34)

  # Remove rows that are all zero or which contain NAs.
  sums = rowSums(data[,dsdCols])
  idx = which(is.na(sums) | sums == 0)
  if(length(idx) > 0) {
    data = data[-idx,]
  }

  return(data)
}

plotComponents = function(orthComponents, textSize=14, legendPos="none") {
  # Plot orthogonal component loadings by original variable.
  #
  # Args:
  #   orthComponents: The components, as returned by orthogonalComponents().
  #   textSize: Font size (default: 14).
  #   legendPos: Position for the legend (default: "none", no legend).
  #
  # Returns: A ggplot object.

  compMelted = melt(orthComponents$rotation)
  names(compMelted) = c("Variable", "Component", "Loading")
  compMelted$Class = rep(seq(1, length(orthComponents$variables)),
                         length(orthComponents$rotation[1,]))
  plot = ggplot(compMelted, aes(x=Variable, y=Loading, group=Component)) +
    geom_line(aes(colour=Component), size=0.75) + theme_bw(textSize) +
    scale_x_discrete(labels=orthComponents$variables) +
    theme(axis.text.x = element_text(angle=90, hjust=1)) +
    theme(legend.position=legendPos)

  return(plot)
}

addRainStatsToSpatialDSDs = function(dsds, stations, timestepSeconds, radar=TRUE, ...) {
  # Add rain statistics to a spatial set of DSDs.
  #
  # Args:
  #   dsds: The grid of DSDs. One DSD per point.
  #   stations: Station information (at least name, altitude, latitude).
  #   timestepSeconds: Seconds per timestep.
  #   radar: Calculate radar reflectivities?
  #   ...: Extra arguments to DSDRainStats(), eg class diameters.
  #
  # Note, altitude and latitude of points are not taken into account.
  #
  # Returns: A spatial grid with rain statistics added to each point.

  gridStats = DSDRainStats(spectra=data.frame(dsds),
                           stations=stations,
                           timestepSeconds=timestepSeconds,
                           radar=radar, ...)
  coords = coordinates(dsds)
  proj4 = proj4string(dsds)
  dsds = cbind(dsds, gridStats)
  coordinates(dsds) = coords
  proj4string(dsds) = proj4

  return(dsds)
}

plotGriddedRainVars = function(gridDSDs, variables=c("R", "Nt", "Dm", "Zh"),
    units=c("mm/h", "m-3", "mm", "dBZ"),
    labels=variables,
    kmScale=TRUE, textSize=16,
    knownData=numeric(0), knownPointSize=2,
    zlims=rep(numeric(0), length(variables))) {
  # Plotted gridded rain variables.
  #
  # Args:
  #   gridDSDs: Spatial object with rain variables included for each point.
  #   variables: The variables to plot.
  #   units: Unit for each variable.
  #   labels: Variable labels, will be parsed (default: variables).
  #   kmScale: Plot with scale in KMs from bottom left? (Default: TRUE).
  #   textSize: The text size to use in the plots (default: 10).
  #   knownData: Points used for kriging (default: none specified).
  #   knownPointSize: The size for the known points (default: 2).
  #
  # Returns: A list of plots named by variable.

  statPlots = list()
  for(s in seq(1, length(variables))) {
    var = variables[s]
    zlim = numeric(0)
    if(length(zlims) > 0) {
      zlim = zlims[[s]]
    }
    statPlots[[var]] = plotGrid(gridDSDs, variable=var, varUnit=units[s],
                 varName=labels[s],
                 zlims=zlim, title=var, knownData=knownData,
                 kmScale=kmScale, textSize=textSize,
                 knownPointSize=knownPointSize)

  }
  return(statPlots)
}

closestGridPoints = function(gridData, points) {
  # Return the closest grid values to a list of spatial points.
  #
  # Args:
  #   gridData: Gridded data, with coordinates.
  #   points: Points to retrieve.
  #
  # Returns: A data.frame with gridData entries closest to points,
  #          one line per point, with a column "dist" added which contains
  #          the Euclidean distance from the grid centre to the point.

  # Find the Euclidean distances between points.
  dists = spDists(gridData, points, longlat=FALSE)

  # Loop through columns in the dists matrix, these correspond to each point.
  rows = NULL
  for(p in seq(1, length(dists[1,]))) {
    closestIdx = which.min(dists[,p])
    row = data.frame(gridData[closestIdx,], dist=dists[closestIdx,p])
    rows = rbind(rows, row)
  }

  return(rows)
}

logTransformCols = function(data, variables, n=0) {
  # Log transform variables in a data.frame. x becomes log(x+n).
  #
  # Args:
  #   data: The data.table to log transform.
  #   variables: The variables (columns) to log transform.
  #   n: Shift data by 'n' before log transform (default: 0, no shift).
  #
  # Returns: The data.table with all values replaced by their log.

  res = copy(data)

  # No NA values are allowed.
  stopifnot(!any(is.na(res[, variables, with=FALSE])))

  # If no shift, then set zeros to NA.
  if(n == 0) {
      setZeros = function(x) { x[which(x==0)] = NA; return(x) }
      res = res[, (variables) := lapply(.SD, setZeros), .SDcols=variables]
  }

  res = res[, (variables) := lapply(.SD+n, log), .SDcols=variables]
  return(res)
}

backTransformCols = function(data, variables, rm.tiny=TRUE, tiny.thresh=0.002,
    n=0) {
  # Back-transform variables in a data.frame. log(x+1) becomes x.
  #
  # Args:
  #   data: The data.table to log transform.
  #   variables: The variables (columns) to log transform.
  #   rm.tiny: Set negative values and very small values to zero before
  #            backtransforming? (Default: TRUE).
  #   tiny.thresh: The threshold for "tiny" values (Default: 1e-3).
  #   n: Shift data after back-transform by -n (default: 0, no shift).
  #
  # Returns: The data.table with all values replaced by their log.

  e = function(x) {
      bt = exp(x)-n
      if(rm.tiny) bt[which(bt < tiny.thresh)] = 0

      # If no shift, NAs are interpreded as zeros.
      if(n == 0) bt[which(is.na(bt))] = 0

      return(bt)
  }
  res = copy(data)
  res[, (variables) := lapply(.SD, e), .SDcols=variables]
  return(res)
}

## logTransform = function(data) {
##   # Log transform all values in a matrix or data frame. Undone by
##   # backTransform().
##   #
##   # Args:
##   #   data: Numeric data to be transformed.
##   #
##   # Returns: Transformed data.

##   return(log(data+1))
## }

## backTransform = function(data) {
##   # Back transform all values in a matrix or data frame. Undoes what
##   # logTransform() does.
##   #
##   # Args:
##   #   data: Numeric data to be back transformed.
##   #
##   # Returns: Backtransformed data.

##   # Backtransform the log-transformation.
##   return(exp(data)-1)
## }

reconstructData = function(orthComponents, data, columnNames) {
  # Given values from a set of orthogonal components, reconstruct
  # the original values that would form those component values.
  #
  # Args:
  #   orthComponents: Description of the orthogonal component set, as
  #                   provided by orthogonalComponents().
  #   data: A matrix containing component data to reconstruct, with
  #         a row per data point and a named column per component.
  #   columnNames: Column names for the reconstructed data.
  #
  # Returns: Data reconstructed from component values.

  stopifnot(names(data) == orthComponents$componentNames)

  # Reconstruct using the PCA rotation matrix.
  transformed = data %*% t(orthComponents$rotation)

  # Rescale and recenter.
  if(orthComponents$scale[1] != FALSE) {
    transformed=scale(transformed, scale=1/orthComponents$scale, center=FALSE)
  }
  if(orthComponents$center[1] != FALSE) {
    transformed=scale(transformed, center=-orthComponents$center, scale=FALSE)
  }

  transformed = data.frame(transformed)
  names(transformed) = columnNames
  return(transformed)
}

orthogonalComponents = function(data, variables, keepVars,
                                tolerance=0, logTransform=FALSE, scale=TRUE,
                                centre=TRUE, logShift=1) {
  # Use PCA to find orthogonal components over a set of variables.
  #
  # Args:
  #  data: The data.table over which to perform PCA.
  #  variables: Which variables (columns) in data should be considered?
  #  keepVars: Which variables should be kept together with the components?
  #  tolerance: The PCA tolerance to use. Components that have a standard
  #             deviation greater than (standard deviation
  #             of the most important component)*(tolerance) will be returned. At 0,
  #             all components are returned and no information is lost. At
  #             higher values, some information will be lost.
  #  logTransform: Log transform the data before finding components?
  #                (Default: TRUE).
  #  scale: Use scaling for PCA? (Default: TRUE).
  #  centre: Should values be shifted to be zero-centred? (Default: TRUE).
  #  logShift: Shift values before/after log transform (default: 1).
  #
  # Returns: A list containing components (the components plus keepVars),
  #          componentNames (the component names), variables (the names of
  #          the variables rotated), rotation (the PCA rotation matrix),
  #          center (the PCA centering amounts), scale (the PCA scaling
  #          amounts), sdev (the standard deviation of each component)
  #          and summary (the PCA summary).

  # Log transform if required.
  if(logTransform) {
    data = logTransformCols(data, variables, n=logShift)
  }

  # Use PCA to find orthogonal components.
  pca = prcomp(data[, variables, with=FALSE], center=centre,
               scale.=scale, tol=tolerance)

  # The names of the components returned by PCA.
  componentNames = names(pca$x[1,])

  # Put together the timestep, station, and rotated matrix.
  components = data.frame(data[, keepVars, with=FALSE], pca$x)
  names(components) = c(keepVars, componentNames)

  return(list(components=components, componentNames=componentNames,
              variables=variables, rotation=pca$rotation, center=pca$center,
              scale=pca$scale, sdev=pca$sdev, summary=summary(pca),
              princomp=pca))
}

fittedVariograms = function(data, variables, distBoundaries,
    useCressie=FALSE,
    nuggets=numeric(0),
    realisationVarName="realisation",
    modelType=c("Sph"),
    textSize=11,
    checkForWarnings=TRUE) {
  # Find a sample variogram and a fitted model for each variable in a
  # spatial data.frame.
  #
  # Args:
  #   data: Spatial data (use prepareDataForGeostats() to prepare them).
  #         Must be an SP object.
  #   variables: Variables for which to find variograms, assumed to be
  #              independent.
  #   distBoundaries: Distance boundaries to use.
  #   useCressie: Use Cressie's robust estimator? (Default: FALSE).
  #   nuggets: Nuggets to use named by variable, or none to fit nuggets.
  #            (Default: none, fit nuggets from zero).
  #   realisationVarName: Column that is unique per realisation in the data.
  #   modelType: What type of model to fit? (Default: Sph). Will be tried in
  #              order; if these models can not be fitted a Lin model will
  #              be tried, and the function returns NULL if this linear fit
  #              fails.
  #   textSize: Font size for plots (Default: 11).
  #
  # Returns: A list containing a list of variograms, a list of models,
  #          and a list of variogram plots, each named by variable. Also,
  #          useCressie to indicate whether Cressie's robust estimator was
  #          used.

  varios = list()
  models = list()
  plots = list()

  for(var in variables) {
    print(var)

    # Estimate range as a third of the maximum distance given, and sill
    # as data variance.
    range = max(distBoundaries)/2
    sill = stats::var(data[[var]])

    varios[[var]] =
      variogram(as.formula(paste(var, realisationVarName, sep="~")),
                data=data, dX=0.5, boundaries=distBoundaries,
                cressie=useCressie)

    # Fit nugget or not?
    if(length(nuggets) == 0) {
      nugget = 0
      fit.sills=c(T, T)
    } else {
      nugget = nuggets[[var]]
      fit.sills=c(F, T)
    }

    # Define model and fit it to the variogram.
    for(type in modelType) {
      model = vgm(model=type, nugget=nugget, psill=sill, range=range)
      warn = has_warning(models[[var]] <- fit.variogram(varios[[var]], model, fit.sills=fit.sills))

      if(checkForWarnings) {
          if(warn == FALSE)
              break

          ## If there is a warning and the model is not singular, try again.
          if(!attr(models[[var]], "singular")) {
              if(type == "Lin") {
                  warning(paste("Accepting non-converged linear model as last resort for", var))
                  break
              }
              models[[var]] = NULL
              next
          }
      }
      
      if(attr(models[[var]], "singular")) {
          ## Fit had a singular fit. This means that (from ?fit.variogram:)
          ## ------------------------------------------------------------------
          ## "On singular model fits: If your variogram turns out to be a flat,
          ##  horizontal or sloping line, then fitting a three parameter model
          ##  such as the exponential or spherical with nugget is a bit heavy:
          ##  there's an infinite number of possible combinations of sill and
          ##  range (both very large) to fit to a sloping line. In this case,
          ##  the returned, singular model may still be useful: just try and
          ##  plot it. Gstat converges when the parameter values stabilize, and
          ##  this may not be the case. Another case of singular model fits
          ##  happens when a model that reaches the sill (such as the spherical)
          ##  is fit with a nugget, and the range parameter starts, or converges
          ##  to a value smaller than the distance of the second sample
          ##  variogram estimate. In this case, again, an infinite number of
          ##  possibilities occur essentially for fitting a line through a
          ##  single (first sample variogram) point. In both cases, fixing one
          ##  or more of the variogram model parameters may help you out."
          ## -------------------------------------------------------------------
          ##
          ## A sloping line is no problem for us. Test for the second
          ## condition and if it is true, force the range to be equal
          ## to the second distance in the observed variogram; the
          ## variogram will represent noise after this distance.
          if(models[[var]]$range[2] <= varios[[var]][["dist"]][2]) {
              model = vgm(model=type, nugget=nugget, psill=sill, range=varios[[var]][["dist"]][2])
              models[[var]] = fit.variogram(varios[[var]], model, fit.sills=fit.sills,
                        fit.range=FALSE)
          }
          break
      }
    } 
    
    if(is.null(models[[var]]))
        stop("Could not fit variogram! Try more model types?")
        
    plots[[var]] = ggplotVariogram(varios[[var]], var, model=models[[var]],
                                   varUnits="log", varSymbol=var,
                                   distUnits="m", textSize=textSize, title=var)
  }

  return(list(variograms=varios, models=models, plots=plots,
              useCressie=useCressie))
}

gridAroundPoints = function(points, buffer=1000, xRes=100, yRes=100) {
  # Make a grid of points around the bounding box of a set of spatial points.
  #
  # Args:
  #   points: Points around which to build the grid, an SP object.
  #   buffer: Space to add around all sizes of bounding box [same unit as
  #           points] (Default: 1000).
  #   xRes, yRes: X and Y resolutions [same unit as points] (Default: 100).
  #
  # Returns: SP object of points.

  xRange = range(coordinates(points)[,1])
  yRange = range(coordinates(points)[,2])
  xVals = seq(min(xRange)-buffer, max(xRange)+buffer+xRes, by=xRes)
  yVals = seq(min(yRange)-buffer, max(yRange)+buffer+xRes, by=yRes)
  gridpoints = expand.grid(xVals, yVals)
  coordinates(gridpoints) = ~Var1+Var2
  proj4string(gridpoints) = proj4string(points)

  return(gridpoints)
}

predictAtPoints = function(models, variables, knownData, points, nsim=0,
                           debug=0, nmax=Inf, maxdist=Inf) {
  # Using ordinary kriging to predict the value of each variable at unknown
  # points, given data at known points.
  #
  # Args:
  #   models: Fitted variogram models named by variable.
  #   variables: Variables to find values for.
  #   knownData: A spatial object with variable values at various coordinates.
  #   points: Points at which to predict the variable values.
  #   nsim: Number of simulations to produce. If non-zero, simulation will be
  #         performed instead of interpolation (Default: 0).
  #   debug: Debug level for krige() (default: 0).
  #   maxdist: Maximum distance for kriging/simulation (default: Inf).
  #   nmax: Maximum number of nearest simulated values to use (default: Inf).
  #
  # Returns: A list of predictions named by variable.

  if(nsim != 0) {
      ## For simulation, ensure we only simulate at points that are
      ## not the same as known data points. Bump coincident points
      ## by a tiny margin.

      ## Which grid point is exactly the same as a known data point?
      dists = spDists(knownData, points)
      minDists = apply(dists, 2, min)
      coincidentPoints = which(minDists < 1e-10)

      ## Bump the lat/long for the coincident points.
      projString = proj4string(points)
      coordNames = names(data.frame(coordinates(points)))
      points = data.table(data.frame(points))[, coordNames, with=FALSE]
      points[coincidentPoints, ] = points[coincidentPoints, ] + 1e-9
      coordinates(points) = coordNames
      proj4string(points) = projString

      dists = spDists(knownData, points)
      minDists = apply(dists, 2, min)
      coincidentPoints = which(minDists < 1e-10)
      stopifnot(length(coincidentPoints) == 0)
  }

  predictions = list()

  ## Process each variable in parallel.
  require(doParallel)

  options(warn=2) # Turn warnings into errors.

  list = ls()
  includeVars = c("knownData", "points", "models", "debug",
                  "nsim", "maxdist", "nmax")
  excludeVars = list[which(!(list %in% includeVars))]
  
  #tm = proc.time()
  predictions = foreach(variable=variables, .combine=c, .inorder=TRUE, 
                        .verbose=(debug!=0),
                        .noexport=excludeVars,
                        .packages=c("gstat")) %dopar%
    krige(as.formula(paste(variable, "~1", sep="")),
          knownData, newdata=points,
          model=models[[variable]], debug.level=debug,
          nsim=nsim, maxdist=maxdist, nmax=nmax)
  #print(proc.time() - tm)
  names(predictions) = variables

  ## Set warnings back to normal warnings.
  options(warn=1)

  return(predictions)
}

plotPredictions = function(predictions, variables, knownData, ...) {
  # Plot the values and estimation variances for each in a set of predictions.
  #
  # Args:
  #   predictions: The predictions from predictAtPoints().
  #   variables: The variables for which to plot.
  #   knownData: Known data points used to make the predictions.
  #   ...: Optional arguments to plotInterpolatedField().
  #
  # Returns: A list of two lists, one of predicted values named by variable,
  #          the other of estimation variances named by variable.

  predictionPlots = list()
  estVarPlots = list()
  for(variable in variables) {
    p = plotInterpolatedField(predictions=predictions[[variable]],
                              knownData=knownData, title=variable,
                              statName=variable, ...)
    predictionPlots[[variable]] = p$prediction
    estVarPlots[[variable]] = p$estimationVariance
  }

  return(list(predictionPlots=predictionPlots, estVarPlots=estVarPlots))
}

collectPredictions = function(predictions, variables, predictionName="var1.pred") {
  # Collect together all predictions out of a list of predictions and
  # estimation variances.
  #
  # Args:
  #   predictions: Predictions, output from predictAtPoints().
  #   variables: Variables to get predictions for.
  #
  # Returns: A matrix containing a column per variable and a row per point.

  predictedVals = list()
  for(variable in variables) {
    predictedVals[[variable]] = as.numeric(predictions[[variable]][[predictionName]])
  }
  predictedVals = do.call("data.frame", predictedVals)
  predictedVals = as.matrix(predictedVals)

  return(predictedVals)
}

interpolationQuantileRange = function(predictions, gridDSDs,
    orthComponents, dryDriftModels, timeRes,
    variables=c(dsdColumnNames, "R", "Dm"),
    dsdColumnNames=paste("class", seq(1,32), sep=""),
    logShift=1, radar=FALSE, samples=1000) {
    ## Find the IQR of interpolation estimations by assuming each estimated component
    ## is from a Gaussian distribution, and probabilistically drawing samples from this
    ## distribution. 
    ##
    ## Args:
    ##
    ##   predictions: Predictions, from predictAtPoints(). Predicted components and errors.
    ##   gridDSDs: Predicted DSDs. Must match predictions and contain dry drift.
    ##   orthComponents: The orthogonal components object produced by
    ##                   orthogonalComponents().
    ##   dryDriftModels: Models for the dry drift, found using dryDriftByRadar().
    ##                   Should contain models and functions.
    ##   timeRes: Time resolution of the data.
    ##   variables: Variables for which to find probability ranges.
    ##   dsdColumnNames: Column names of DSD classes.
    ##   logShift: Log back-transformation shift.
    ##   radar: Calculate radar values? (Default: FALSE).
    ##   samples: Number of samples to draw from each distribution (default: 1000).
    ##
    ## Returns: A spatial object, with quantile ranges for each point and variable.

    ## gridDSDs is ordered by y first
    ## prediction coordinates are ordered by x first
    points = data.table(data.frame(coordinates(predictions[[1]])))
    coordNames = names(points)

    ## Use estimations as means.
    predictedMeans = collectPredictions(predictions,
        variables=orthComponents$componentNames,
        predictionName="var1.pred")
    names(predictedMeans) = orthComponents$componentNames
    predictedMeans = data.table(predictedMeans)
    predictedMeans = as.matrix(predictedMeans)

    ## Get standard deviations for estimations.
    predictedSDs = collectPredictions(predictions,
        variables=orthComponents$componentNames,
        predictionName="var1.var")
    names(predictedSDs) = orthComponents$componentNames
    predictedSDs = data.table(predictedSDs)
    predictedSDs = as.matrix(predictedSDs)

    ## For each coordinate, draw 'samples' points from the distribution
    ## defined by the mean and standard deviation.
    predictedComps = matrix(NA, nrow=samples*nrow(predictedMeans), ncol=ncol(predictedMeans))
    for(row in seq(1, nrow(predictedMeans))) {
        start = ((row-1)*samples)+1
        end = start+samples-1

        for(col in seq(1, ncol(predictedMeans))) {
            predictedComps[(start:end), col] = rnorm(samples, mean=predictedMeans[row, col],
                            sd=sqrt(predictedSDs[row, col]))
        }
    }

    ## Transform each predicted component set into a DSD.
    predictedDSDs = reconstructData(orthComponents=orthComponents,
        data=predictedComps,
        columnNames=orthComponents$variables)

    ## Pad with zeros for DSD classes not covered.
    dsds = data.frame(matrix(0, ncol=length(dsdColumnNames),
        nrow=nrow(predictedDSDs)))
    names(dsds) = dsdColumnNames
    dsds[,names(predictedDSDs)] = predictedDSDs
    dsds = data.table(dsds)

    ## Assign coordinates. Repeat each coordinate 'samples' times.
    dsds[, x := rep(points[[coordNames[1]]], each=samples)]
    dsds[, y := rep(points[[coordNames[2]]], each=samples)]
    setkey(dsds, x, y)

    ## Set dry distances.
    gridTable = data.table(data.frame(gridDSDs))[, c(coordNames, "dryDist", "edgeDist"), with=FALSE]
    setkeyv(gridTable, coordNames)
    dsds[, dryDist := gridTable[dsds, dryDist]]
    dsds[, edgeDist := gridTable[dsds, edgeDist]]

    ## Add dry drift.
    dsds = addDryDrift(data=dsds, models=dryDriftModels$models,
        functions=dryDriftModels$functions)

    ## Set to NA dsdCols for coordinates that do not have a dry drift.
    dsds = dsds[is.na(dryDist), (dsdCols) := NA, with=FALSE]

    ## Back-transform the log-transform.
    dsds = backTransformCols(data=dsds, variables=dsdColumnNames, n=logShift)

    ## Add bulk variables.
    dsds = data.table(addRainStats(dsds, timestepSeconds=timeRes, radar=radar))

    ## Calculate quantile ranges for the requested variables, per location.
    ranges = dsds[, lapply(.SD, IQR, na.rm=TRUE), .SDcols=variables, by=coordNames]

    ## Return a spatial object.
    coordinates(ranges) = ~x+y
    proj4string(ranges) = proj4string(predictions[[1]])
    return(ranges)
}

interpolateDSDForTime = function(data, time, dsdCols, stations, varios, pca,
    radarDir, dryDriftModels, timeRes,
    radarElevation, grid, at=NULL,
    logShift=1, modComponents=identity,
    addRainStats=TRUE, nsim=0, debug=0,
    maxdist=Inf, nmax=Inf,
    ...) {
  # Interpolate the DSD for a given time, using detrended data, and returning
  # the reconstructed DSDs, either at specified points or in a grid around
  # the known points.
  #
  # Args:
  #  data: Observed data; already detrended using subtractDryDrift(),
  #  time: POSIXct time for which to interpolate.
  #  dsdCols: The column names of the DSD columns.
  #  stations: Station information.
  #  varios: Variograms for each PCA component, returned from
  #          fittedVariograms().
  #  pca: Orthogonal components, as returned from orthogonalComponents().
  #  radarDir: Directory in which all radar files are stored.
  #  dryDriftModels: Models for the dry drift, found using dryDriftByRadar().
  #                  Should contain models and functions.
  #  timeRes: Time resolution of the data.
  #  radarElevation: Radar elevation to search for.
  #  grid: The grid of points on which to map radar values for the dry
  #        distance (note not used if dryPixels are specified!).
  #  at: Points to interpolate at. Or, NULL to make a grid around the
  #      known points (default: NULL).
  #  logShift: Amount to shift values back by after backtransforming log
  #            (default: 1).
  #  modComponents: Modify components (ie backtransform them) before doing
  #                 anything else with them, using this function.
  #                 Default: identity().
  #  addRainStats: Add rain statistics to interpolated DSDs? (Default: TRUE).
  #  nsim: Number of simulations to produce. If non-zero, simulation will be
  #        performed instead of interpolation (Default: 0).
  #  debug: Debug level for krige() (default: 0).
  #  maxdist: Maximum distance for kriging/simulation (default: Inf).
  #  nmax: Maximum number of nearest simulated values to use (default: Inf).
  #  ...: Extra arguments to dryDistRadarForTime(); eg if dryPixels are
  #       specified it will use cached dry pixels.
  #
  # Returns: A list containing three spatial objects;
  #            . gridDSDs, the gridded DSDs, fully back-transformed
  #            . gridComponents, the gridded PCA components
  #            . knownData, the detrended input points.

  # The known data are the station data for the timestep. Should
  # already be prepared for geostatistics.
  knownData = data[which(data$POSIXtime == time),]
  if(length(knownData$POSIXtime) == 0) return(NULL)

  # Get the smallest observed concentration; interpolation results
  # will not be allowed to be smaller than this and not zero.
  knownDataTable = addDryDrift(data=data.table(data.frame(knownData)),
      models=dryDriftModels$models, functions=dryDriftModels$functions)
  minNonZero = function(x) {
    idx = which(x != 0)
    if(length(idx) > 0) return(min(x[idx]))
    return(NA)
  }
  smallestConcentration = min(as.numeric(knownDataTable[, lapply(.SD, minNonZero),
      .SDcols=dsdCols]), na.rm=T)

  # Construct a grid to krige over, around the known points.
  if(!is.null(at)) {
      points = at
  } else {
      points = gridAroundPoints(knownData)
  }

  # Predict each DSD at the grid points.
  predictions = interpolateDetrendedDSD(orthComponents=pca,
      knownData=knownData, points=points, models=varios$models,
      modComponents=modComponents, nsim=nsim, debug=debug,
      maxdist=maxdist, nmax=nmax)

  ## Find the maximum range of any model.
  maxRange = 0
  for(class in names(dryDriftModels$models)) {
    func = dryDriftModels$functions[[class]]
    model = dryDriftModels$models[[class]]
    range = data.frame(t(coefficients(model)))$range
    nugget = data.frame(t(coefficients(model)))$nugget
    if(func == "gaussianModel")
      range = 1.73*range + nugget # Gaussian models use a pseudorange.

    if(range > maxRange)
      maxRange = range
  }

  # Find dry distances for each interpolated point using radar data.
  dryDists = dryDistRadarForTime(pts=points, grid=grid, time=time,
      timeRes=timeRes, radarDir=radarDir,
      radarElevation=radarElevation, maxDist=maxRange, ...)

  # Add the dry drift back to interpolated DSDs (keep the DSDs as a
  # spatial object).
  gridDSDs = data.table(data.frame(predictions$predictedDSDs))

  n = names(data.frame(coordinates(predictions$predictedDSDs), row.names=NULL))
  setnames(gridDSDs, n[1], "x")
  setnames(gridDSDs, n[2], "y")

  dryDists = data.table(dryDists)
  setkey(dryDists, x, y)
  dryDists = unique(dryDists, by=c("x","y"))
  setkey(gridDSDs, x, y)
  setnames(dryDists, "dryDist", "RdryDist")
  setnames(dryDists, "edgeDist", "RedgeDist")
  gridDSDs[dryDists, dryDist := RdryDist]
  gridDSDs[dryDists, edgeDist := RedgeDist]

  gridDSDs = addDryDrift(data=gridDSDs, models=dryDriftModels$models,
                         functions=dryDriftModels$functions)

  ## Set to NA dsdCols for coordinates that do not have a dry drift.
  gridDSDs = gridDSDs[is.na(dryDist), (dsdCols) := NA]

  ## Backtransform.
  gridDSDs = backTransformCols(gridDSDs, dsdCols,
      rm.tiny=TRUE, tiny.thresh=smallestConcentration, n=logShift)

  setnames(gridDSDs, "x", n[1])
  setnames(gridDSDs, "y", n[2])
  coordinates(gridDSDs) = names(data.frame(coordinates(predictions$predictedDSDs), row.names=NULL))
  proj4string(gridDSDs) = proj4string(predictions$predictedDSDs)

  # Get rain variables from interpolated DSDs. Stations are set to NULL
  # here because we assume we don't know the altitude for the grid points.
  if(addRainStats) {
      gridDSDs = addRainStatsToSpatialDSDs(gridDSDs,
          timestepSeconds=timeRes, stations=NULL)
  }

  return(list(gridDSDs=gridDSDs, knownData=knownData,
              gridComponents=predictions$predictedComponents))
}

interpolateDetrendedDSD = function(orthComponents, knownData, points, models,
    dsdColumnNames=paste("class", seq(1,32), sep=""),
    modComponents=identity, nsim=0, debug=0, maxdist=Inf,
    nmax=Inf) {
  # Interpolate the detrended DSD from detrended DSD observations, to
  # other specified points.
  #
  # Args:
  #   orthComponents: Output from orthogonalComponents() for the data set.
  #   knownData: DSD observations at various points, as a spatial object.
  #   points: Spatial object; points to predict at.
  #   models: Variogram model for each orthogonal component (use
  #           fittedVariograms()).
  #   dsdColumnNames: Names of DSD columns in knownData.
  #   modComponents: Modify predicted components (ie backtransform them)
  #                  before doing anything else with them, using this function.
  #                  Default: identity().
  #   nsim: Number of simulations to produce. If non-zero, simulation will be
  #         performed instead of interpolation (Default: 0).
  #   debug: Debug level for krige() (default: 0).
  #   maxdist: Maximum distance for kriging/simulation (default: Inf).
  #   nmax: Maximum number of nearest simulated values to use (default: Inf).
  #
  # Returns: A list with predictedDSDs (a predicted DSD at each requested
  #          point) and predictedComponents (predicted component and estimation
  #          variance at each grid point).

  # Get the PCA components that correspond to knownData. These are
  # assumed to be already transformed if required, modComponents is the
  # back-transformation function.
  knownDataTable = data.table(data.frame(knownData))
  orthComponents$components = data.table(orthComponents$components)
  setkey(orthComponents$components, "POSIXtime", "station")
  setkey(knownDataTable, "POSIXtime", "station")
  observedComponents = orthComponents$components[knownDataTable]
  stopifnot(identical(observedComponents[, list(POSIXtime, station)],
                      knownDataTable[, list(POSIXtime, station)]))

  # Add spatial information to the list of observed components.
  coordCols = names(as.data.frame(coordinates(knownData)))
  coordinates(observedComponents) = knownDataTable[, coordCols, with=FALSE]
  proj4string(observedComponents) = proj4string(knownData)

  # Predict components at other points, based on observations.
  compPredictions = predictAtPoints(models=models,
      variables=orthComponents$componentNames,
      knownData=observedComponents, points=points,
      nsim=nsim, debug=debug, maxdist=maxdist, nmax=nmax)
  
  for(n in seq(1, max(1, nsim))) {
      if(nsim > 0) {
          predictionName = paste("sim", n, sep="")
      } else {
          predictionName = "var1.pred"
      }

      ## Backtransform the components into DSD drop counts.
      predictedComps = collectPredictions(compPredictions,
          variables=orthComponents$componentNames,
          predictionName=predictionName)
      names(predictedComps) = orthComponents$componentNames
      predictedComps = data.table(predictedComps)
      predictedComps = modComponents(predictedComps)
      predictedComps = as.matrix(predictedComps)
      predictedDSDs = reconstructData(orthComponents=orthComponents,
          data=predictedComps,
          columnNames=orthComponents$variables)

      # Pad with zeros for columns that were not included in the components.
      dsds = data.frame(matrix(0, ncol=length(dsdColumnNames),
          nrow=length(data.frame(points)[,1])))
      names(dsds) = dsdColumnNames
      dsds[,names(predictedDSDs)] = predictedDSDs

      # Assign coordinates to the predicted DSDs.
      coordinates(dsds) = coordinates(points)
      proj4string(dsds) = proj4string(points)

      if(nsim > 0)
          dsds$simNumber = n

      if(n == 1)
          allDSDs = dsds
      else
          allDSDs = rbind(allDSDs, dsds)
  }

  return(list(predictedDSDs=allDSDs, predictedComponents=compPredictions))
}

leaveOneOutTesting = function(measuredData, detrendedData, stations,
    timeRes, dsdVarios, orthComponents, radarDir, dryDriftModels, dryPixels,
    radarElevation=4,
    bulkVariables=c("Nt", "R", "Dm", "Zh"),
    varUnits=c("m^{-3}", "mm~h^{-1}", "mm", "dBZ"),
    textSize=10,
    dsdColNames=paste("class", seq(1,32), sep=""),
    classDiams=get.classD(),
    metresCRS=CRS(paste("+proj=utm +zone=31 +ellps=WGS84 +datum=WGS84",
        "+units=m +no_defs")),
    latLongCRS=CRS("+proj=longlat +datum=WGS84"),
    modComponents=identity, logShift=1, ...) {
  # Perform leave-one-out testing of interpolations using DSD interpolation
  # and bulk-variable interpolation. Each station is left out in turn.
  #
  # Args:
  #   measuredData: DSDs per station, time, with variables.
  #         All POSIXtime values in data will be tested. This data is used as
  #         the ground truth data, against which the methods are tested.
  #   detrendedData: Detrended versions of the DSD, to use with the DSD
  #                  interpolation technique. Spatial object.
  #   stations: Station information (at least name, altitude, latitude).
  #   timeRes: Seconds per timestep.
  #   dsdVarios: Variograms for DSD components.
  #   orthComponents: DSD component specification.
  #   radarDir: Directory in which to find radar NC files.
  #   dryDriftModels: Models for the dry drift, found using dryDriftByRadar().
  #   timeRes: Time resolution for the data.
  #   radarElevation: Radar elevation [deg] to look for (default: 4 degrees).
  #   bulkVariables: Bulk variable names to interpolate and test.
  #   varUnits: Units for each variable.
  #   textSize: Text size for plots.
  #   dsdColNames: Names of the columns for the DSD in data.
  #   classDiams: Classes to use for moment calculation (default: Parsivel classes).
  #   metresCRS: The CRS for metres (Default: UTM).
  #   latLonCRS: CRS for lat/long (Default: WGS84).
  #   modComponents: Modify components (ie backtransform them) before doing
  #                  anything else with them, using this function.
  #                  Default: identity().
  #
  # Returns: A list containing statistics on the differences, timeseries plots
  #          of differences, and bar charts of statistics on differences, plus
  #          the timesteps in which the DSD interpolation performed worst, by
  #          variable, and perClassStats, containing performance stats for each
  #          DSD class.
  #
  # Note: Radar reflectivities Zh and Zv are converted to linear units for
  # comparison.

  ## Convert data into a data.table for speed.
  dt = data.table(measuredData)

  allPredictedDSDs = NULL
  allDiffs = NULL
  allRelDiffs = NULL
  allRelToQuant = NULL

  ## Loop through each station in the data, leaving it out.
  for(s in dt[, unique(station)]) {
      ## Get coordinates for the test station.
      testStation = stations[which(stations$name == s),]
      coordinates(testStation) = ~lon+lat
      proj4string(testStation) = latLonCRS
      testStation = spTransform(testStation, metresCRS)

      ## Get the real values at the left-out station.
      measured = copy(dt[station == testStation$name,])
      if(measured[, length(POSIXtime)] == 0) {
          print("No measured data available at station/times specified.")
          return(NULL)
      }

      DSDResults = NULL

      ## Loop through all times for which the leave-one-point had a measurement.
      for(t in measured[, POSIXtime]) {
          t = as.POSIXct(t, tz="UTC", origin="1970-1-1")
          print(paste(s, t))

          ## If there is no detrended value for the left-out station and
          ## time, there will not be a dry drift, so don't bother comparing it.
          n = which(detrendedData$POSIXtime == t &
              detrendedData$station == testStation$name)
          if(length(n) == 0) {
              measured = measured[POSIXtime != t]
              next
          }
          stopifnot(length(n) == 1)

          ## Separate known data to use for interpolation from the left out data.
          measuredIdx = which(measuredData$POSIXtime == t &
              measuredData$station != testStation$name)
          detrendedIdx = which(detrendedData$POSIXtime == t &
              detrendedData$station != testStation$name)
          if(length(measuredIdx) == 0 | length(detrendedIdx) == 0) {
              print("Not enough data to compare.")
              measured = measured[POSIXtime != t]
              next
          }
          knownData = detrendedData[detrendedIdx,]

          ## Get the smallest observed concentration; interpolation results
          ## will not be allowed to be smaller than this and not zero.
          knownDataTable = addDryDrift(data=data.table(data.frame(knownData)),
              models=dryDriftModels$models, functions=dryDriftModels$functions)
          minNonZero = function(x) {
              idx = which(x != 0)
              if(length(idx) > 0) return(min(x[idx]))
              return(NA)
          }
          smallestConcentration = min(as.numeric(knownDataTable[, lapply(.SD, minNonZero),
              .SDcols=dsdCols]), na.rm=T)

          ## Predict each DSD at the station point.
          predictions = interpolateDetrendedDSD(orthComponents=orthComponents,
              knownData=knownData, points=testStation, models=dsdVarios$models,
              modComponents=modComponents, nsim=0)

          ## Find dry distances for the test point point using radar data.
          ## Use the already-given dry distance from detrendedData
          dryDist = detrendedData[which(detrendedData$station == testStation$name &
              detrendedData$POSIXtime == t), ]$dryDist
          edgeDist = detrendedData[which(detrendedData$station == testStation$name &
              detrendedData$POSIXtime == t), ]$edgeDist

          ## Add the dry drift back to interpolated DSDs (keep the DSDs as a
          ## spatial object).
          gridDSDs = data.table(data.frame(predictions$predictedDSDs))
          gridDSDs$dryDist = dryDist
          gridDSDs$edgeDist = edgeDist

          gridDSDs = addDryDrift(data=gridDSDs, models=dryDriftModels$models,
              functions=dryDriftModels$functions)

          ## Set to NA dsdCols for coordinates that do not have a dry drift.
          gridDSDs = gridDSDs[is.na(dryDist), (dsdCols) := NA, with=FALSE]
          gridDSDs = backTransformCols(data=gridDSDs, variables=dsdCols,
              rm.tiny=TRUE, tiny.thresh=smallestConcentration, n=logShift)

          ## Save the results.
          DSDResults = rbind(DSDResults, data.table(POSIXtime=t,
              station=s, data.frame(gridDSDs)))
      }

      ## Do we have results?
      if(length(DSDResults$POSIXtime) == 0) next

      ## Add rain statistics using station altitudes.
      DSDResults = data.table(addRainStats(DSDResults,
          timestepSeconds=timeRes, stations=stations))

      ## Calculate differences between DSD interpolation results and
      ## measured values. For both DSD columns, and bulk variables.
      stopifnot(identical(DSDResults$POSIXtime, measured$POSIXtime))
      stopifnot(identical(unique(DSDResults$station),
                          as.character(unique(measured$station))))

      ## Differences are interpolated - measured.
      diffs = DSDResults[, c(dsdColNames, bulkVariables), with=FALSE] -
          measured[, c(dsdColNames, bulkVariables), with=FALSE]

      ## Relative differences count only measured data != 0.
      measuredNonZero = data.frame(measured)
      measuredNonZero[measuredNonZero == 0] = NA
      measuredNonZero = data.table(measuredNonZero)
      relDiffs = diffs / abs(measuredNonZero[, c(dsdColNames, bulkVariables),
          with=FALSE]) * 100

      ## Differences as percentage of 10th to 90th percentile range.
      quantRange = function(x) { return(abs(diff(quantile(x, probs=c(0.1, 0.9), na.rm=T)))) }
      quantMeasuredRow = as.numeric(measured[, lapply(.SD, quantRange),
          .SDcols=c(dsdColNames, bulkVariables)])
      quantMeasuredRow[which(quantMeasuredRow == 0)] = NA
      quantMeasured = NULL
      for(i in seq(1, dim(diffs)[1]))
          quantMeasured = rbind(quantMeasured, quantMeasuredRow)
      relToQuant = diffs / quantMeasured * 100

      ## Assign station.
      diffs$leftOutStation = s
      relDiffs$leftOutStation = s

      diffs = cbind(DSDResults[, POSIXtime], diffs)
      relDiffs = cbind(DSDResults[, POSIXtime], relDiffs)

      allDiffs = rbind(allDiffs, diffs)
      allRelDiffs = rbind(allRelDiffs, relDiffs)
      allPredictedDSDs = rbind(allPredictedDSDs, DSDResults)
      allRelToQuant = rbind(allRelToQuant, relToQuant)
  }

  return(list(diffs=allDiffs, relDiffs=allRelDiffs, relToQuant=allRelToQuant,
              predictedDSDs=allPredictedDSDs))
}

plotDSDComparison = function(dsd1, dsd2, dsd1Name, dsd2Name, title,
                             textSize=10, logScale=TRUE) {
  # Plot a comparison between two DSDs.
  #
  # Args:
  #   dsd1: The first DSD.
  #   dsd2: The second DSD.
  #   dsd1Name: Name for the first.
  #   dsd2Name: Name for the second.
  #   title: Plot title.
  #   textSize: Font size (default: 10).
  #   logScale: Use a log-scale for the drop count axis? (Default: TRUE).
  #
  # Returns: A ggplot2 object.

  # DSDs must be the same length.
  stopifnot(length(dsd1) == length(dsd2))

  # Plot a comparison.
  plotData = rbind(data.frame(name=dsd1Name, class=seq(1, length(dsd1)),
                              value=as.numeric(dsd1)),
                   data.frame(name=dsd2Name, class=seq(1, length(dsd2)),
                              value=as.numeric(dsd2)))

  plot = ggplot(plotData, aes(x=class, y=value, group=name)) +
    geom_line(aes(colour=name), size=.75) + theme_bw(textSize) +
    scale_colour_discrete(name="DSD")

  ylabel = parse(text="N(D)~group('[',mm^{-1}~m^{-3},']')")
  if(logScale) {
      ylabel = "log(N(D))"
      plot = plot + scale_y_continuous(trans="log")
  }

  plot = plot + labs(x="Diameter class", y=ylabel, title=title)

  return(plot)
}

simpleScatterPlot = function(vector1, vector2, name1, name2,
                             title, textSize=10, logScale=FALSE,
                             includeOneToOneLine=FALSE) {
  # Produce a simple scatter plot to compare two vectors.
  #
  # Args:
  #   vector1: The first set of values to compare.
  #   vector2: The second set of values to compare.
  #   name1: The name for the first.
  #   name2: The name for the second.
  #   title: Plot title.
  #   textSize: Font size (default 10).
  #   logScale: Use log scales? (Default: FALSE).
  #   includeOneToOneLine: Include the 1:1 lines?
  #
  # Returns: ggplot scatterplot with and fitted line of best fit.

  stopifnot(length(vector1) == length(vector2))

  d = data.frame(var1=vector1, var2=vector2)

  plot = ggplot(d, aes(x=var1, y=var2)) +
    geom_point(size=0.75) + theme_bw(textSize) +
    labs(x=name1, y=name2, title=title)

  if(logScale) {
    plot = plot + scale_x_continuous(trans="log")  +
      scale_y_continuous(trans="log")
  }

  if(includeOneToOneLine) {
    plot = plot + geom_abline(slope=1, intercept=0)
  }

  # Line of best fit.
  plot = plot + geom_smooth(method="lm", size=0.75, colour="blue")
  return(plot)
}

############################ Dry drift functions ###########################

dryDriftByRadar = function(data, grid, variables, timeRes, radarDir,
    radarElevation, stations, displayDistClassWidth=1,
    textSize=12, logShift=1, logTransform=TRUE, plot=TRUE, ...) {
  # Calculate the dry drift by time in a dataset.
  #
  # Args:
  #  data: The data to calculate the dry drift of; must contain station,
  #        POSIXtime, and the columns on which to calculate the drift.
  #        Must be only rainy timesteps.
  #  grid: Radar data will be mapped to this grid (an SP object) before
  #        dry distances are found.
  #  variables: Which variables (in data) to calculate the dry drift for?
  #  timeRes: The time resolution [s].
  #  radarDir: Directory for radar files.
  #  radarElevation: Elevation to use (can be multiple).
  #  stations: Stations definitions.
  #  displayClassWidth: width of display class [km].
  #  textSize: Size of text used in plots.
  #  logShift: Shift data values by this amount before logtransform?
  #            (Default: 1).
  #  logTransform: Perform the log transform? (Default: TRUE).
  #
  # Returns: A list containing fitted, the fitted dry drift models,
  #          quantilePlots (plots) and logData (the data, log transformed).

  # Log transform the variables.
  if(logTransform) {
    data = logTransformCols(data, variables, n=logShift)
  }

  # Add the dry distances to the data.table.
  data = dryDistByRadarDistance(dataTable=data,
                                grid=grid,
                                timeRes=timeRes,
                                radarDir=radarDir,
                                radarElevation=radarElevation,
                                stations=stations, ...)

  # If no dry distances were found, that means there were no radar
  # files available; return NULL.
  if(!("dryDist" %in% names(data)))
      return(NULL)

  if(plot) {
      quantilePlots = plotQuantilesByDistance(data=data, dryDistVar="dryDist",
          distClassWidth=displayDistClassWidth,
          variables=variables, distUnit="km",
          textSize=textSize)
  } else {
      quantilePlots = NULL
  }

  fitted = fitAllDryDriftModels(data=data, variables=variables,
                                dryDistVar="dryDist", distUnits="km",
                                distClassWidth=displayDistClassWidth,
                                textSize=textSize, ...)

  ## Determine the largest range in the models; NA dry distances that have
  ## edge distances closer to this can be set to have a dry distance equal
  ## to the largest model range.
  maxDist=0
  for(model in fitted$models) {
    range = data.frame(t(coefficients(model)))$range
    if(range > maxDist)
      maxDist = range
  }
  data = data[is.na(dryDist) & edgeDist > maxDist, dryDist := maxDist]

  return(list(fitted=fitted, quantilePlots=quantilePlots, logData=data))
}

dryDriftByTime = function(data, variables, dryTimes,
                          displayDistClassWidth=10, textSize=12,
                          logShift=1) {
  # Calculate the dry drift by time in a dataset.
  #
  # Args:
  #  data: The data to calculate the dry drift of; must contain station,
  #        POSIXtime, and the columns on which to calculate the drift.
  #        Must be only rainy timesteps.
  #  variables: Which variables (in data) to calculate the dry drift for?
  #  dryTimes: The times around the data for which there was no rain
  #            recorded. Must contain POSIXtime, station.
  #  classWidth: With of distance classes used in plots [mins] (Default: 10).
  #  textSize: Size of text used in plots.

  # Log transform the variables.
  data = logTransformCols(data, variables, n=logShift)

  # Find the dry distances by time; merge them into the dataset.
  dists = dryDistancesByTime(data, dryTimes)
  data = merge(data, dists, by=c("station","POSIXtime"))

  quantilePlots = plotQuantilesByDistance(data=data, dryDistVar="dryDist",
                                          distClassWidth=displayDistClassWidth,
                                          variables=variables, distUnit="mins",
                                          textSize=textSize)

  fitted = fitAllDryDriftModels(data=data, variables=variables,
                                dryDistVar="dryDist", distUnits="mins",
                                distClassWidth=displayDistClassWidth,
                                textSize=textSize)

  return(list(fitted=fitted, quantilePlots=quantilePlots, logData=data))
}

subtractDryDrift = function(data, models, functions,
                            distVar="dryDist", edgeVar="edgeDist") {
  # Take log-transformed data, and some fitted dry drift models,
  # and detrend the data by subtracting the dry drift.
  #
  # Args:
  #  dataTable: The data.table to work on. Must already be log-transformed!
  #  models: A list of named models, one for each variable to work on.
  #          Calculate using fitAllDryDriftModels().
  #  functions: The function for each fitted model. From fitAllDryDriftModels().
  #  distVar: The variable in 'data' that corresponds to the distance
  #           from a dry region. Calculate using dryDistByRadarDistance().
  #  edgeVar: The variable in 'data' that corresponds to the distance
  #           from the edge of the occurance map or an NA pixel. Calculate
  #           using dryDistByRadarDistance().
  #
  # Returns: The data.table duplicated but with each variable detrended.

  # Subtract the dry drifts.
  allDrifts = calculateDriftsFromModel(data=data, models=models,
                                       functions=functions, distVar=distVar,
                                       edgeVar=edgeVar)
  subtracted = copy(data)
  variables = names(models)
  subtracted[, (variables) := data[, variables, with=FALSE] - allDrifts]
  return(subtracted)
}

calculateDriftsFromModel = function(data, models, functions,
                                    distVar, edgeVar) {
  # For each point in some data, calculate its dry drift using fitted
  # dry drift models.
  #
  # Args:
  #  dataTable: The data.table to work on.
  #  models: A list of named models, one for each variable to work on.
  #          Calculate using fitAllDryDriftModels().
  #  functions: The model function to use for each model.
  #  distVar: The variable in 'data' that corresponds to the distance
  #           from a dry region.
  #  edgeVar: The variable in 'data' that corresponds to the distance
  #           from the edge of the occurance map or an NA pixel.
  #
  # Returns: A data.table containing the drift for each variable, for
  #          each point.

  variables = names(models)
  edgeDists = data[[edgeVar]]
  dryDists = data[[distVar]]

  # Find the drifts for each variable.
  allDrifts = data.table()
  for(variable in variables) {
    model = models[[variable]]
    func = get(functions[[variable]])
    coeffs = data.table(t(coefficients(model)))
    range = coeffs$range

    drifts = data.table(d=func(dryDists, range=coeffs$range,
                               sill=coeffs$sill,
                               nugget=coeffs$nugget))

    ## A drift of NA means no drift was defined; and we cannot estimate at
    ## that particular point. These values will remain NA.
    setnames(drifts, "d", variable)
    if(dim(allDrifts)[1] == 0)
      allDrifts = drifts
    else
      allDrifts = cbind(allDrifts, drifts)
  }

  return(allDrifts)
}

addDryDrift = function(data, models, functions,
                       distVar="dryDist", edgeVar="edgeDist") {
  # Take log-transformed data, and some fitted dry drift models,
  # and re-trend the data by adding the dry drift.
  #
  # Args:
  #  dataTable: The data.table to work on.
  #  models: A list of named models, one for each variable to work on.
  #          Calculate using fitAllDryDriftModels().
  #  functions: The function for each fitted model. From fitAllDryDriftModels().
  #  distVar: The variable in 'data' that corresponds to the distance
  #           from a dry region. Calculate using dryDistByRadarDistance().
  #  edgeVar: The variable in 'data' that corresponds to the distance
  #           from the edge of the occurance map or an NA pixel. Calculate
  #           using dryDistByRadarDistance().
  #
  # Returns: The data.table duplicated but with each variable re-trended.

  # Get the try drifts, and add them.
  allDrifts = calculateDriftsFromModel(data=data, models=models,
                                       functions=functions, distVar=distVar,
                                       edgeVar=edgeVar)
  variables = names(models)
  added = copy(data)
  added[, (variables) := data[, variables, with=FALSE] + allDrifts]
  return(added)
}


dryDistancesByTime = function(data, dryTimes) {
  # For each record, find the distance in time to a dry period.
  #
  # Args:
  #  data: The records for which the dry distance is to be found,
  #        containing at least station and POSIXtime. Every POSIXtime
  #        here is assumed to be for a rainy time.
  #  dryTimes: A data.table containing station and POSIXtime, where
  #            POSIXtime gives the times for which the station recorded
  #            no rain.
  #
  # Returns: The original data.table "data" with a column called
  #          "dryDistance" added, containing the distance in minutes to
  #          the closest dry time.

  dryTimes = dryTimes[, list(station_name=station_name, dryTime=as.integer(POSIXtime))]
  queryTimes = data[, list(station_name=station_name, rainTime=as.integer(POSIXtime))]

  setkey(dryTimes, "station", "dryTime")
  setkey(queryTimes, "station", "rainTime")

  distances = dryTimes[J(queryTimes), list(dryDist=abs(dryTime-rainTime)), roll="nearest"]
  setnames(distances, "dryTime", "POSIXtime")
  distances[, POSIXtime := as.POSIXct(POSIXtime, origin="1970-1-1", tz="UTC")]

  # Convert from seconds to minutes.
  distances[, dryDist := dryDist / 60]

  return(distances)
}

distToEdgeOrNA = function(pts, grid, colNames=c("x", "y", "rainy")) {
    ## Find the distance from a point to the edge of a grid or a NA pixel
    ##
    ## Args:
    ##   pts: Points to find distances for.
    ##   grid: Grid of points to define edges.
    ##   colNames: Column names for x, y, and the variable.
    ##
    ## Both 'pts' and 'grid' should be data.table objects
    ## with columns for x, y coordinates.
    ##
    ## Returns: the distance of each point to the edge of the
    ## grid, in the same unit as the points are in.

    grid = data.table(data.frame(grid)[, colNames])
    grid = data.frame(grid)
    names(grid) = c("x", "y", "val")
    grid = data.table(grid)
    xRes = abs(diff(unique(grid$x))[1])
    yRes = abs(diff(unique(grid$y))[1])

    minX = grid[, min(x)]
    maxX = grid[, max(x)]
    minY = grid[, min(y)]
    maxY = grid[, max(y)]

    edgePoints = data.frame(bindAll(
      grid[x == minX, list(x=x-xRes, y=y)], # Left edge.
      grid[x == maxX, list(x=x+xRes, y=y)], # Right edge.
      grid[y == maxY, list(x=x, y=y+yRes)], # Top edge.
      grid[y == minY, list(x=x, y=y-yRes)], # Bottom edge.
      grid[is.na(val), list(x=x, y=y)]))      # NA values.

    dists = nn2(edgePoints, pts, k=1)$nn.dists

    return(dists)
}

dryDistRadarForTime = function(pts, time, grid,
                               timeRes, radarDir, radarElevation,
                               metresCRS=CRS(paste("+proj=utm +zone=31",
                                                   "+ellps=WGS84 +datum=WGS84",
                                                   "+units=m +no_defs")),
                               gridBuffer=5000, dryPixels=NULL, warn=FALSE,
                               maxDist=NULL, allowEdgePoints=FALSE, ...) {
  # For a set of spatial points, find the closest dry region
  # in radar images, and return the distance between the point location
  # and the dry region.
  #
  # Args:
  #  pts: A spatial object with coordinates to find distances for.
  #  time: POSIXtime (UTC) for requested radar scan.
  #  grid: Project radar points onto this grid before finding dry distance.
  #        Note: ignored if dryPixels are used as a cache.
  #  timeRes: Input data time resolution [s].
  #  radarDir: Directory in which to find all radar files
  #            (type assumed to be "PPI").
  #  radarElevation: Radar elevation to read.
  #  radarPattern: Pattern of radar files to match (default: PPI for elevation).
  #  getTimeFromNameFunc: Function to get a time from a radar filename
  #                       (default: for PPI).
  #  radarThresh: Threshold for radar variable to use as radar wet/dry point
  #  radarVar: Radar variable to test for wet/dry test.
  #  gridBuffer: The buffer to add around the original grid, to look for
  #              radar values in.
  #  dryPixels: If specified, use precalculated dry pixels to find the
  #             distances. Must contain POSIXtime, x and y in metresCRS, and
  #             boolean "rainy" for each point.
  #  warn: Produce warnings if dry distances can not be calculated for some
  #        points? (Default: FALSE).
  #  maxDist: If specified, this is the maximum 'range' of any dry drift model;
  #           and is thus the distance after which the dry drift does not
  #           increase.
  #  allowEdgePoints: allow points that are closer to an edge than the nearest
  #                   dry pixel to be included? (default: FALSE).
  #  ...: Extra arguments to PPIRainMap().
  #
  # Returns: The data.table containing x, y coordinates, and the dry distance
  #          to each point, and distance to edge of occurance map for each
  #          point. If distance to edge is less than distance to dry pixel,
  #          dry distance is set to NA.


  # Get a list of dry pixels for this time.
  if(is.null(dryPixels)) {
    occuranceMap = PPIRainMask(radarDir=radarDir, time=time,
                               elevation=radarElevation,
                               locations=grid, maxAllowedTimeDiff=timeRes,
                               buffer=gridBuffer, ...)
    if(is.null(occuranceMap)) return(NULL) # No radar map.
    occurancePix = data.table(data.frame(occuranceMap))[, list(x, y, rainy)]
    dryPix = data.table(data.frame(occuranceMap))[rainy == FALSE, list(x, y)]
  } else {
      if(!(identical(key(dryPixels), c("POSIXtime", "rainy"))))
          setkey(dryPixels, POSIXtime, rainy)

      lookup = data.table(POSIXtime=time, rainy=0)
      setkey(lookup, POSIXtime, rainy)
      dryPix = dryPixels[lookup, nomatch=0]

      occurancePix = dryPixels[(POSIXtime == time), list(x, y, rainy)]
      if(dim(occurancePix)[1] == 0) return(NULL) # No occurance map.
  }

  # Transform query points to metres.
  if(!as.character(proj4string(pts)) == as.character(metresCRS))
    pts = spTransform(pts, metresCRS)
  queryPts = data.frame(coordinates(pts))
  names(queryPts) = c("x", "y")

  # Get the distance of each point from the edge of the occurance map.
  coordinates(occurancePix) = ~x+y
  proj4string(occurancePix) = metresCRS
  occurancePts = data.frame(occurancePix)
  toEdge = distToEdgeOrNA(pts=queryPts, grid=occurancePts) / 1000

  if(dim(dryPix)[1] == 0) {
    ## In the case that no dry pixels are found, return NA for the dry
    ## distance, and the edge distances. These are the maximum known
    ## distance for which the region is rainy.
    minDists = rep(as.numeric(NA), length(toEdge))
  } else {
    ## Transform dry pixels and radar coordinates to metres.
    coordinates(dryPix) = ~x+y
    proj4string(dryPix) = metresCRS
    radarPts = coordinates(dryPix)

    ## Find the closest points and get the distance to that point in km.
    minDists = nn2(radarPts, queryPts, k=1)$nn.dists / 1000

    ## We can not calculate an accurate distance for points that are
    ## closer to a dry region than the edge of the occurance map.
    if(any(toEdge < minDists) & !allowEdgePoints) {
      idx = which(toEdge < minDists)
      minDists[idx] = NA

      ## However of those points, those that are further from the edge
      ## than the maximum distance (if specified) can be considered
      ## to be that maximum distance from a dry region.
      if(!is.null(maxDist)) {
        fixed = intersect(which(toEdge > maxDist), idx)
        minDists[fixed] = maxDist
      }
    }
  }

  stopifnot(length(toEdge) == length(minDists))
  return(data.table(data.frame(queryPts, dryDist=minDists, edgeDist=toEdge)))
}

dryDistByRadarDistance = function(dataTable, grid,
                                  timeRes, radarDir,
                                  radarElevation, stations,
                                  latLongCRS=CRS(paste("+proj=longlat",
                                                       "+datum=WGS84")),
                                  metresCRS=CRS(paste("+proj=utm +zone=31",
                                                      "+ellps=WGS84",
                                                      "+datum=WGS84",
                                                      "+units=m +no_defs")),
                                  ...) {
  # For all records in instrument data, find the closest dry region
  # in radar images, and return the distance between the instrument location
  # and the dry region.
  #
  # Args:
  #  dataTable: The instrument data as a data.table with at least
  #             station, POSIXtime.
  #  grid: Grid onto which the radar values should be mapped - should
  #        be the same grid as interpolation will be performed on.
  #  timeRes: Input data time resolution [s].
  #  radarDir: Directory in which to find all radar files
  #            (type assumed to be "PPI").
  #  radarElevation: Radar elevation to read.
  #  stations: Stations definition.
  #  latLongCRS: CRS for latitude/longitude.
  #  metresCRS: CRS for UTM.
  #  ...: Extra arguments to dryDistRadarForTime().
  #
  # Returns: The original data.table with a new column "dryDist",
  #          containing the distance in km to the nearest dry pixel in a
  #          corresponding radar map, or NA if no radar snapshot was near.

  # Get all unique times.
  data = copy(dataTable)
  times = data[, unique(POSIXtime)]

  # Key data by station and time.
  setkey(data, "station", "POSIXtime")

  # Make a spatial object with station locations. Convert to metres in UTM.
  coordinates(stations) = ~lon+lat
  proj4string(stations) = latLongCRS
  stations = spTransform(stations, metresCRS)

  # Loop through all data times, looking for ones for which there
  # is a close enough radar scan.
  for(time in times) {
    time = as.POSIXct(time, tz="UTC", origin="1970-1-1")
    dists = dryDistRadarForTime(pts=stations, grid=grid, time=time,
                                timeRes=timeRes, radarDir=radarDir,
                                radarElevation=radarElevation,
                                ...)
    if(is.null(dists)) next
    if(all(is.na(dists$dryDist))) next

    dists = dists[, station := stations$name]
    dists = dists[, POSIXtime := time]
    setnames(dists, "dryDist", "RdryDist")
    setnames(dists, "edgeDist", "RedgeDist")
    setkey(dists, "station", "POSIXtime")

    # Add the minimum distances to the data table.
    data[dists, dryDist := RdryDist]
    data[dists, edgeDist := RedgeDist]
  }

  return(data)
}

axisLabelsByClass = function(variables, log=TRUE) {
  # For variables named "class3" etc, make a pretty axis name
  # for that class, including the diameter range.

  diamClasses = get.classD()

  classNum = as.integer(str_extract(variables, "[0-9]+"))
  #diamRange = paste("[", round(diamClasses[classNum,1], 2), ", ",
  #                  round(diamClasses[classNum,2], 2), ") mm", sep="")

  labels = parse(text=paste("Conc.~class~", classNum,
                         "~group('[',mm^{-1}~m^{-3},']')", sep=""))

  if(log)
    labels = parse(text=paste("Log~conc.~class~", classNum,
                       "~group('[',log,']')", sep=""))

  return(labels)
}

plotQuantilesByDistance = function(data, dryDistVar, distClassWidth,
                                   variables, distUnit,
                                   textSize=14, fillcolour="lightblue",
                                   axisLabels=axisLabelsByClass(variables)) {
  # For a set of variables, plot variable distributions by dry distance.
  #
  # Args:
  #  data: The data to plot.
  #  dryDistVar: The variable in 'data' that contains the dry distance.
  #  distClassWidth: Distance class width [unit of dry distance].
  #  variables: The name of each variable for which to make a plot.
  #  distUnit: The unit of distance.
  #  diamClasses: The equivolume diameter classes (default: Parsivel).
  #  axisLabels: The axis label for each variable (expression).
  #
  # Returns: A list containing the

  # Determine distance breaks to divide into.
  distBreaks = seq(0, data[, max(.SD, na.rm=TRUE),
                           .SDcols=dryDistVar]+distClassWidth,
                   by=distClassWidth)

  # Cut distances by classes.
  data = data[, distClass := lapply(.SD, cut, breaks=distBreaks,
                                    label=FALSE, right=FALSE),
              .SDcols=dryDistVar]
  data = data[!is.na(distClass)]

  # Plot distance class vs. quantiles for each variable.
  i = 1
  plots = list()
  varNames = list()
  meanPlots = list()

  for(class in variables) {
    if(all(data[, class, with=FALSE] == 0)) next
    varNames[[i]] = class

    q10 = data[, list(q10=as.numeric(lapply(.SD, quantile, probs=0.1, na.rm=TRUE))),
               by=distClass, .SDcols=class]
    q25 = data[, list(q25=as.numeric(lapply(.SD, quantile, probs=0.25, na.rm=TRUE))),
               by=distClass, .SDcols=class]
    q50 = data[, list(q50=as.numeric(lapply(.SD, quantile, probs=0.5, na.rm=TRUE))),
               by=distClass, .SDcols=class]
    q75 = data[, list(q75=as.numeric(lapply(.SD, quantile, probs=0.75, na.rm=TRUE))),
               by=distClass, .SDcols=class]
    q90 = data[, list(q90=as.numeric(lapply(.SD, quantile, probs=0.9, na.rm=TRUE))),
               by=distClass, .SDcols=class]
    mean = data[, list(mean=as.numeric(lapply(.SD, mean, na.rm=TRUE))),
                by=distClass, .SDcols=class]

    quantiles = merge(q10, q25, by="distClass")
    quantiles = merge(quantiles, q50, by="distClass")
    quantiles = merge(quantiles, q75, by="distClass")
    quantiles = merge(quantiles, q90, by="distClass")
    quantiles = merge(quantiles, mean, by="distClass")
    quantiles$xVal = (quantiles$distClass*distClassWidth)-(distClassWidth/2)

    plots[[i]] =
      ggplot(quantiles, aes(x=xVal)) +
      geom_errorbar(stat="identity", aes(ymin=q10, ymax=q90)) +
      geom_crossbar(stat="identity",
                    aes(group=distClass, y=q50, ymin=q25, ymax=q75),
                    fill=fillcolour) +
      geom_point(aes(y=mean)) +
      theme_bw(textSize-3) + labs(x=paste("Dry distance [", distUnit, "]",
                                          sep=""),
                                  y=axisLabels[[i]])

    i = i + 1
  }

  names(plots) = varNames
  return(plots)
}

gaussianModel = function(dist, range, sill, nugget=0) {
  # A modified Gaussian model.
  #
  # Args:
  #  dist: The distance(s) for which to find a value.
  #  range: The pseudorange [same unit as distance].
  #  sill: The model sill [same unit as return value].
  #  nugget: A 'nugget' that works on the distance axis. Shifts
  #          all values to the right by this amount; distances
  #          less than the nugget distance have model values zero
  #          [same unit as distance] (Default: 0, no shift).
  #
  # Returns: The value(s) given by the model.

  ret = sill * (1 - exp(-((dist-nugget)^2/range^2)))
  ret[which(dist < nugget)] = 0
  return(ret)
}

sphericalModel = function(dist, range, sill, nugget) {
  # The spherical + nugget model.
  #
  # Args:
  #   dist: The distance(s) for which to find a value.
  #   range: The model range [same unit as distance].
  #   sill: The model partial sill [same unit as return value].
  #   nugget: The model nugget [same unit as return value].
  #
  # Returns: The value(s) given by the model.

  ret = rep(nugget + sill, length(dist))
  ret[which(is.na(dist))] = NA
  idx = which(dist <= range)
  ret[idx] = nugget + (sill*((3*dist)/(2*range) - 0.5*(dist/range)^3))[idx]
  return(ret)
}

fitDryDriftModel = function(data, variable, dryDistVar, distUnits,
                            models=c("sphericalModel", "gaussianModel"),
                            startVals=list(range=max(data[[dryDistVar]],
                                                     na.rm=TRUE)/4,
                                           sill=max(data[[variable]],
                                                    na.rm=TRUE),
                                           nugget=min(data[[variable]],
                                                      na.rm=TRUE)),
                            lowerParams=c(0,0,0),
                            upperParams=c(max(data[[dryDistVar]], na.rm=TRUE)*2,
                                            max(data[[variable]], na.rm=TRUE)*2,
                                            max(data[[variable]])),
                            distClassWidth=1, textSize=14,
                            showAllPoints=FALSE, verbose=FALSE,
                            nlsControl=nls.lm.control(maxiter=500),
                            varLabel=axisLabelsByClass(variable), ...) {
  # Fit a dry distance model to data using non-linear least squares (nls).
  # The model is fitted to ALL data, non-weighted. A corresponding plot
  # is produced showing the fitted model against mean values per
  # distance class (with mean distance per class on the x-axis).
  #
  # Args:
  #  data: The data to fit to.
  #  variable: The variable to which to fit the model.
  #  dryDistVar: The name of the dry distance column in data.
  #  models: A list of models to try fitting.
  #  startVals: Named list with parameter starting values.
  #  textSize: Text size for the plot (default: 14).
  #  lowerParams: Lower bounds for parameters to fit (default: 0 for spherical).
  #  distClassWidth: The width of the desired distance classes for the plot.
  #  showAllPoints: Show all points on the plot? (Default: FALSE, show only
  #                 distance-class means).
  #  verbose: Print (lots of) information during the fit?
  #  nlsControl: Object of class nls.lm.control() to control fitting.
  #  varLabel: Label to use in the plot, for the variable axis.
  #
  # Returns: A list containing the fitted model ready to use with predict
  #          and a plot showing the data (pts) and the fitted model (line),
  #          plus a list showing which model was used for each variable.x
  if(verbose) {
    print("Starting values are:")
    print(startVals)
  }

  # Cut distances into classes.
  distBreaks = seq(0, data[, max(.SD, na.rm=TRUE),
                           .SDcols=dryDistVar]+distClassWidth,
                   by=distClassWidth)

  data = data[, distClass := lapply(.SD, cut, breaks=distBreaks,
                                    label=FALSE, right=FALSE),
              .SDcols=dryDistVar]
  data = data[!is.na(distClass)]
  stopifnot(length(data$distClass) > 0)

  # Find the mean of the variable for each distance class, plus the
  # mean distance per distance class.
  dataToFit = data[, c(variable, dryDistVar, "distClass"), with=FALSE]
  setnames(dataToFit, variable, "var")
  setnames(dataToFit, dryDistVar, "dist")
  classedDataToFit = dataToFit[, list(var=mean(var), dist=mean(dist),
                                      np=length(dist)), by=distClass]

  bestFit = NA
  model = NULL
  for(modelFunc in models) {
    modelFormula=as.formula(paste("var~", modelFunc,
                                  "(dist, range, sill,",
                                  " nugget)", sep=""))
    testModel = try(nlsLM(formula=modelFormula, data=dataToFit,
                          start=startVals, lower=lowerParams,
                          upper=upperParams, control=nlsControl),
                    silent=TRUE)

    if(class(testModel) == "try-error") next
    if(is.na(bestFit) | abs(mean(residuals(testModel))) < bestFit) {
      model = testModel
      func = modelFunc
      bestFit = abs(mean(residuals(testModel)))
    }
  }

  # If no model can be fitted, give up.
  if(is.null(model)) return(NULL)

  predPoints = data.frame(dist=seq(0, max(dataToFit$dist), length.out=100))
  predictions = predict(model, predPoints)
  predictions = data.table(dist=predPoints$dist, var=predictions)

  toPlot = classedDataToFit
  if(showAllPoints) toPlot = dataToFit

  plot = ggplot(toPlot, aes(x=dist, y=var)) +
    geom_point(colour="black") +
    theme_bw(textSize-3) +
    labs(x=paste("Dry distance [", distUnits, "]", sep=""), y=varLabel) +
    geom_line(data=predictions)

  return(list(model=model, plot=plot, modelFunc=func))
}

fitAllDryDriftModels = function(data, variables, dryDistVar, distClassWidth,
                                distUnits, verbose=FALSE, ...) {
  # Fit dry drift models for all requested variables.
  #
  # Args:
  # Args:
  #  data: The data to fit to.
  #  variables: The variables to which to fit models.
  #  dryDistVar: The name of the dry distance column in data.
  #  distClassWidth: The width of the desired distance classes.
  #  ...: Optional arguments to fitDryDriftModel().
  #  verbose: Print information during fitting? (Default: FALSE).
  #
  # Returns: A list of models and a list of plots, both named by variable.

  models = list()
  plots = list()
  functions = list()

  for(v in seq(1, length(variables))) {
    variable = variables[v]
    if(verbose) print(paste("Variable:", variable))

    fitted = fitDryDriftModel(data=data, variable=variable,
                              dryDistVar=dryDistVar,
                              distUnits=distUnits,
                              distClassWidth=distClassWidth,
                              verbose=verbose,
                              ...)

    models[[variable]] = fitted$model
    plots[[variable]] = fitted$plot
    functions[[variable]] = fitted$modelFunc
  }

  return(list(models=models, plots=plots, functions=functions))
}

summariseNLSModels = function(models, modelName="class") {
  # Summarise NLS models. Returns the fitted parameter values and
  # standard error for each model in a list.
  #
  # Args:
  #  models: A named list of models, each returned from nls().
  #  modelName: The column name for the column to describe the
  #             different models (default: class).
  #
  # Returns: A list containing data.tables for the parameters
  # and the errors. Each data table contains columns "modelName"
  # and one for each parameter.

  parameters = data.table()
  errors = data.table()

  for(model in models) {

    s = summary(model)
    coeffs = s$coefficients[,1]
    errs = s$coefficients[,2]

    errors = rbind(errors, t(errs))
    parameters = rbind(parameters, t(coeffs))
  }

  errors = cbind(name=names(models), errors)
  parameters = cbind(name=names(models), parameters)
  setnames(errors, "name", modelName)

  return(list(parameters=parameters, errors=errors))
}

maskGrid = function(grid, time, radarDir, timeRes,
    maskVars, setToValue, radarElevation=4,
    dsdCols = paste("class", seq(1,32), sep=""),
    radarVar="SNRh",
    radarThresh=5) {
  # Take a set of gridded values and mask them using radar data.
  # By default, dry points are simply removed.
  #
  # If "resetVals" is set to true:
  # Rain statistics R, amount, Nt, LWC, D0 are set to 0.
  # Rain statistics Dm, Zh, and Zv are set to NA.
  # DSD columns (if they exist) are set to 0.
  #
  # Args:
  #   grid: The gridded DSDs.
  #   time: The time corresponding to the grid.
  #   radarDir: Directory in which to find radar .NC files.
  #   timeRes: Integration time for the grid.
  #   radarElevation: Radar elevation [deg] to search for (default: 4 degrees).
  #   dsdCols: Names for DSD class columns.
  #   resetVals: Reset values instead of just removing them? (Default: FALSE).
  #
  # Returns: The original grid, points for which dry pixels are removed.

  res = copy(grid)
  resDT = data.table(data.frame(res))

  coordNames = names(data.frame(coordinates(grid)))
  setnames(resDT, coordNames[1], "x")
  setnames(resDT, coordNames[2], "y")
  radarMask = PPIRainMask(radarDir=radarDir, time=time,
                          elevation=radarElevation,
                          locations=grid, maxAllowedTimeDiff=timeRes,
                          threshVar=radarVar,
                          threshAmount=radarThresh)

  radarMask = data.table(data.frame(radarMask))
  setkey(radarMask, "x", "y")
  setkey(resDT, "x", "y")

  masked = resDT[radarMask[rainy == TRUE]]
  masked$rainy = NULL

  if(dim(masked)[1] > 0) {
      coordinates(masked) = ~x+y
      proj4string(masked) = CRS(proj4string(grid))
  }

  return(masked)
}

boxCoxTransform = function(x) {
  # Perform a Box-Cox transformation on a vector.
  #
  # Args:
  #  x: The data values to transform.
  #
  # Returns: a list containing 'data', the transformed data, 'lambda',
  # the box-cox lambda value, and 'min', the minimum of the original data.

  minVal = min(x)
  x = x - minVal + 1 # Make all values positive.
  params = powerTransform(x) # Find lambda.
  transformed = bcPower(x, params$lambda) # Apply lambda.
  return(list(data=transformed, lambda=params$lambda, min=minVal))
}

boxCoxTransformCols = function(x, cols) {
  # Perform Box-Cox transformations on columns in a data.table.
  #
  # Args:
  #  x: The data table to work on.
  #  cols: The names of columns to transform.
  #
  # Returns: a list with 'table', the same data table x, with each
  # named column transformed, plus 'info', a table containing the
  # name, lambda, and minimum for each named variable.

  info = data.table()
  for(c in cols) {
    t = boxCoxTransform(x[[c]])
    x[, c := t$data, with=FALSE]
    info = rbind(info, data.table(col=c, lambda=t$lambda, min=t$min))
  }
  return(list(table=x, info=info))
}

boxCoxBacktransform = function(x, lambda, minVal) {
  # Back-transform a Box-Cox transformation.
  #
  # Box-Cox transformation is log(x) if lambda=0,
  # (x^lambda - 1)/lambda otherwise.
  #
  # Args:
  #   x: The data to back-transform.
  #   lambda: The Box-Cox lambda to use.
  #   minVal: The minimum value of the back-transformed dataset.
  #
  # Returns: a vector of back-transformed values.

  if(lambda == 0) {
    return(exp(x))
  }

  y = (x*lambda + 1)^(1/lambda) + minVal - 1
  return(y)
}

boxCoxBacktransformCols = function(x, cols, infoTable) {
  # Back-transform each named variable in a table of Box-Cox transformed
  # data.
  #
  # Args:
  #   x: The table to work on.
  #   cols: The columns to back-transform.
  #   infoTable: The information table containing column name, lambda, min
  #              for each column in 'cols'.
  #
  # Returns: the table x with columns back-transformed.

  for(c in cols) {
    i = infoTable[col == c]
    t = boxCoxBacktransform(x[[c]], lambda=i$lambda, minVal=i$min)
    x[, c := t, with=FALSE]
  }
  return(x)
}

plotLeaveOneOutResults = function(testRes, vars, labels, unit, xLabel, textSize,
    minR = 0, pointSize=3.5) {

    dsds = testRes$predictedDSDs
    idx = which(dsds$R >= minR)

    dsds = dsds[idx,]
    diffs = testRes$diffs[idx,]
    relDiffs = testRes$relDiffs[idx,]
    relToQuant = testRes$relToQuant[idx,]

    diffsByVar = data.table(melt(data.frame(diffs[, vars, with=FALSE])))
    relDiffsByVar = data.table(melt(data.frame(relDiffs[, vars, with=FALSE])))
    relToQuantByVar = data.table(melt(data.frame(relToQuant[, vars, with=FALSE])))

    for(n in seq(1, length(vars))) {
        diffsByVar[variable == vars[n], lab := labels[n]]
        relDiffsByVar[variable == vars[n], lab := labels[n]]
        relToQuantByVar[variable == vars[n], lab := labels[n]]
    }

    boxDiffs = diffsByVar[, list(mean=mean(value, na.rm=T),
        median=median(value, na.rm=T),
        min=quantile(value, probs=0.10, na.rm=T),
        max=quantile(value, probs=0.90, na.rm=T),
        q25=quantile(value, probs=0.25,  na.rm=T),
        q75=quantile(value, probs=0.75,  na.rm=T)), by=lab]

    boxRelDiffs = relDiffsByVar[!is.na(value), list(mean=mean(value, na.rm=T),
        median=median(value, na.rm=T),
        min=quantile(value, probs=0.10, na.rm=T),
        max=quantile(value, probs=0.90, na.rm=T),
        q25=quantile(value, probs=0.25,  na.rm=T),
        q75=quantile(value, probs=0.75,  na.rm=T)), by=lab]

    boxRelToQuant = relToQuantByVar[!is.na(value), list(mean=mean(value, na.rm=T),
        median=median(value, na.rm=T),
        min=quantile(value, probs=0.10, na.rm=T),
        max=quantile(value, probs=0.90, na.rm=T),
        q25=quantile(value, probs=0.25,  na.rm=T),
        q75=quantile(value, probs=0.75,  na.rm=T)), by=lab]

    env = environment()

    diffPlot = ggplot(boxDiffs, aes(x=factor(lab, levels=labels), y=mean),
        environment=env) +
            geom_errorbar(stat="identity", aes(ymin=min, ymax=max)) +
            geom_crossbar(stat="identity",
                          aes(y=median, ymin=q25, ymax=q75), fill="lightblue") +
            geom_point(aes(y=mean), size=pointSize) +
        theme_bw(textSize) +
        labs(y=parse(text=paste("Error~group('[',", unit, ",']')", sep="")),
             x=xLabel)

    relDiffPlot = ggplot(boxRelDiffs, aes(x=factor(lab, levels=labels), y=mean),
        environment=env) +
            geom_errorbar(stat="identity", aes(ymin=min, ymax=max)) +
            geom_crossbar(stat="identity",
                          aes(y=median, ymin=q25, ymax=q75), fill="lightblue") +
            geom_point(aes(y=mean), size=pointSize) +
        theme_bw(textSize) +
        labs(y="Relative error [%]", x=xLabel)

    relToQuantPlot =
        ggplot(boxRelToQuant, aes(x=factor(lab, levels=labels), y=mean),
               environment=env) +
            geom_errorbar(stat="identity", aes(ymin=min, ymax=max)) +
            geom_crossbar(stat="identity",
                          aes(y=median, ymin=q25, ymax=q75), fill="lightblue") +
            geom_point(aes(y=mean), size=pointSize) +
        theme_bw(textSize) +
        labs(y="Error relative to 10th to 90th\npercentile range [%]", x=xLabel)

    medianRelToQuantPlot = ggplot(boxRelToQuant, aes(x=factor(lab, levels=labels), y=median),
        environment=env) +
        geom_bar(stat="identity", position="identity", fill="lightblue",
                 colour="black", size=.4) +
        theme_bw(textSize) +
        labs(y="Median error relative to \n10th to 90th percentile range [%]", x=xLabel)

    meanRelToQuantPlot = ggplot(boxRelToQuant, aes(x=factor(lab, levels=labels), y=mean),
        environment=env) +
        geom_bar(stat="identity", position="identity", fill="lightblue",
                 colour="black", size=.4) +
        theme_bw(textSize) +
        labs(y="Mean error relative to mean [%]", x=xLabel)

    medianRelDiffPlot = ggplot(boxRelDiffs, aes(x=factor(lab, levels=labels), y=median),
        environment=env) +
        geom_bar(stat="identity", position="identity", fill="lightblue",
                 colour="black", size=.4) +
        theme_bw(textSize) +
        labs(y="Median relative error [%]", x=xLabel)

    meanRelDiffPlot = ggplot(boxRelDiffs, aes(x=factor(lab, levels=labels), y=mean),
        environment=env) +
        geom_bar(stat="identity", position="identity", fill="lightblue",
                 colour="black", size=.4) +
        theme_bw(textSize) +
        labs(y="Mean relative error [%]", x=xLabel)

    return(list(diffPlot=diffPlot, relDiffPlot=relDiffPlot,
                relToQuantPlot=relToQuantPlot,
                medianRelDiffPlot=medianRelDiffPlot,
                meanRelDiffPlot=meanRelDiffPlot,
                meanRelToQuantPlot=meanRelToQuantPlot,
                medianRelToQuantPlot=medianRelToQuantPlot))
}

dynamicBoundariesVariogram = function(data, variables,
    allDistBoundaries, useCressie, nuggets, modelType, minClasses=6,
    requiredPairs=30, by=100) {
    ## Adjust distance boundaries for variogram estimation so that there
    ## are at least 'requiredPairs' pairs per distance class. NULL if not possible.
    ## allDistBoundaries are initial boundaries to divide into larger classes.
    ## maxDiv the maximum number of 'each' boundary to be taken before giving up.
    ## Other arguments as per fittedVariograms().

    distBoundaries = seq(0, max(allDistBoundaries)+by, by=by)

    varios = fittedVariograms(data=data, variables=variables,
                              distBoundaries=distBoundaries, 
                              useCressie=useCressie, nuggets=nuggets, 
                              modelType=modelType, checkForWarnings=FALSE)
    if(is.null(varios))
      return(NULL)

    pairs = varios$variograms[[1]]

    bounds = 0
    sum = 0
    for(i in seq(1, length(pairs$np))) {
      sum = sum + pairs$np[i]

      if(sum > requiredPairs) {
        b = distBoundaries[which(distBoundaries - pairs$dist[i] > 0)[1]]
        bounds = c(bounds, b)
        sum = 0
      }
    }

    if(length(bounds) < minClasses)
      return(NULL)

    varios = fittedVariograms(data=data, variables=variables,
        distBoundaries=bounds, useCressie=useCressie,
        nuggets=nuggets, modelType=modelType)
    if(is.null(varios))
      return(NULL)

    return(varios)
}

prepareEventForInterpolation = function(start, end, allPars, uniquePars,
    timeRes, dropCols, at, stations,
    radarDir, radarElevation,
    metresCRS=CRS("+proj=utm +zone=31 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"),
    latLonCRS=CRS(paste("+proj=longlat +datum=WGS84")),
    pcaTolerance=0, pcaScaling=TRUE,
    pcaCentre=TRUE, modelType=c("Sph"),
    useCressieDSD=TRUE, keepStation="Pradel 1",
    dryPixels=dryPixels, nsim=0, ...) {
    ## Prepare to do interpolations or simulations for a single event.
    ##
    ## Args:
    ##  start: Start time (POSIXtime, UTC).
    ##  end: End time (POSIXtime, UTC).
    ##  allPars: All Parsivel data.
    ##  uniquePars: Parsivel data with collocated stations removed.
    ##  timeRes: Resolution [s].
    ##  radarElevation: Elevation for radar [degrees].
    ##  metresCRS, latLongCRS: Coordinate reference systems.
    ##  pcaTolerance: Tolerance for PCA; see PCA functions.
    ##  pcaScaling, pcaCenter: Scale and centre before PCA?
    ##  modelType: The variogram model type to use. Linear will be used as backup.
    ##  useCressieDSD: Use Cressie variograms?
    ##  dryPixels: Occurance pixels.
    ##  nsim: Number of simulations required.
    ##
    ## Returns: a list containing:
    ##  detrendedGeo: spatial data ready to interpolate on.
    ##  varios: Variograms.
    ##  pca: Principal component analysis information.
    ##  dryDriftModels: Dry drift models.

    ## Subset to event readings.
    pars = allPars[POSIXtime >= start & POSIXtime <= end]
    eventUnique = uniquePars[POSIXtime >= start & POSIXtime <= end]
    dryPixels = dryPixels[POSIXtime >= start & POSIXtime <= end]

    ## Calculate the dry drift for the event. We use all data in order to be able
    ## to calculate nuggets, but we train the dry drift models only on non-collocated
    ## data.
    print("Calculating drifts...")
    drift = dryDriftByRadar(data=eventUnique, grid=at, variables=dropCols,
        timeRes=timeRes, radarDir=radarDir,
        radarElevation=radarElevation, stations=stations,
        plot=FALSE, dryPixels=dryPixels, ...)
    collocDrift = dryDriftByRadar(data=pars, grid=at,
        variables=dropCols, timeRes=timeRes,
        radarDir=radarDir,
        radarElevation=radarElevation,
        stations=stations, plot=FALSE,
        dryPixels=dryPixels, ...)

    if(length(drift$fitted$models) == 0) {
        print("Could not calculate drift functions for event!")
        return(NULL)
    }

    stopifnot(!is.null(drift))
    stopifnot(!is.null(collocDrift))

    ## Subtract the dry drift to get detrended DSDs.
    print("Detrending measurements...")
    detrended = subtractDryDrift(data=collocDrift$logData,
        models=drift$fitted$models,
        functions=drift$fitted$functions)

    ## Keep only those records that have a radar dry distance; same for pars.
    detrended = detrended[!is.na(dryDist)]
    setkey(detrended, "station", "POSIXtime")
    setkey(pars, "station", "POSIXtime")
    pars = pars[detrended, names(pars), with=FALSE]
    stopifnot(!identical(pars, detrended))

    ## Use stations to determine distance boundaries to use for variograms.
    s = copy(stations[which(stations$name %in% eventUnique$station),])
    coordinates(s) = ~lon+lat
    proj4string(s) = latLonCRS
    s = spTransform(s, metresCRS)
    distBoundaries = sort(unique(as.vector(unique(spDists(s)))))
    allDistBoundaries = distBoundaries[which(distBoundaries != 0)]

    ## Classes to work on.
    classes = paste("class", seq(1,32), sep="")

    ## Use PCA to find orthogonal components on the detrended DSDs.
    ## Note we only apply to columns that contain drops.
    print("Calculating principal components...")
    spectra = copy(detrended[, c("POSIXtime", "station", dropCols), with=FALSE])
    spectraColSums = spectra[, lapply(.SD, sum), .SDcols=dropCols]
    spectraCols = names(spectraColSums)[which(spectraColSums != 0)]
    pca = orthogonalComponents(spectra, spectraCols, keepVars=c("POSIXtime","station"),
                               tolerance=pcaTolerance, scale=pcaScaling,
                               centre=pcaCentre, logTransform=FALSE)

    ## Prepare components for geostatistics.
    print("Preparing for geostatistics...")
    prepared = prepareDataForGeostats(pca$components,
                                      nuggetVariables=pca$componentNames,
                                      stations=stations, useCressie=useCressieDSD,
                                      keepStation=keepStation)

    ## Find a variogram per PCA component.
    varios = dynamicBoundariesVariogram(data=prepared$data,
                                        variables=pca$componentNames,
                                        allDistBoundaries=allDistBoundaries,
                                        useCressie=useCressieDSD,
                                        nuggets=prepared$nuggets,
                                        modelType=modelType)
    if(is.null(varios)) return(NULL)

    ## Prepare detrended data for geostatistcs.
    detrendedGeo = prepareDataForGeostats(detrended, stations=stations,
        useCressie=useCressieDSD,
        nuggetVariables=spectraCols,
        keepStation=keepStation)$data

    return(list(detrendedGeo=detrendedGeo, varios=varios,
                pca=pca, dryDriftModels=drift$fitted))
}

interpEvent = function(start, end, allPars, uniquePars,
    timeRes, dropCols, at, stations, radarDir, radarElevation,
    metresCRS=CRS("+proj=utm +zone=31 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"),
    nsim=0, dryPixels=dryPixels, gridded=TRUE, progressFile=NULL,
    dsdCols=paste("class", seq(1,32), sep=""), ...) {
    ## Perform interpolations or simulations for a single event.
    ##
    ## Args:
    ##  start: Start time (POSIXtime, UTC).
    ##  end: End time (POSIXtime, UTC).
    ##  allPars: All Parsivel data.
    ##  uniquePars: Parsivel data with collocated stations removed.
    ##  timeRes: Resolution [s].
    ##  radarElevation: Elevation for radar [degrees].
    ##  times: Times for which to perform interpolation/simulation
    ##         (default: all event times).
    ##  metresCRS: Coordinate reference systems.
    ##  nsim: Number of simulations to produce per time step. 0 = interpolation.
    ##  useCressieDSD: Use Cressie variograms?
    ##  dryPixels: Occurance pixels.
    ##  gridded: Estimate at grid points that are in dryPixels?
    ##  progressFile: If provided, write results out per time
    ##                step. Otherwise, collect all results and return them.
    ##  ...: Extra arguments to interpolateDSDForTime().
    ##
    ## Returns: DSDs per point per simulation/interpolation per timestep.

    prep = prepareEventForInterpolation(start=start, end=end,
        allPars=allPars, uniquePars=uniquePars,
        timeRes=timeRes, dropCols=dropCols,
        stations=stations, radarDir=radarDir,
        radarElevation=radarElevation,
        dryPixels=dryPixels, nsim=nsim, ...)

    detrendedGeo = prep$detrendedGeo
    varios = prep$varios
    pca = prep$pca
    dryDriftModels = prep$dryDriftModels
    atCoords = prep$atCoords
    if(is.null(varios)) return(NULL)

    ## Convert "at" to a data.table for fast coordinate lookups.
    atCoords = data.frame(coordinates(at))
    setnames(atCoords, c("x", "y"))
    atCoords = data.table(atCoords)
    setkey(atCoords, x, y)

    ## Make sure dryPixels key is set properly.
    if(!(identical(key(dryPixels), c("POSIXtime", "rainy"))))
        setkey(dryPixels, POSIXtime, rainy)

    ## Run the interpolation across all time steps.
    results = NULL
    ## counter = 1
    for(ts in unique(detrendedGeo$POSIXtime)) {
        ts = as.POSIXct(ts, origin="1970-1-1", tz="UTC")

        tsFile = paste(progressFile, strftime(ts, tz="UTC"), ".Rdata", sep="")
        if(!is.null(progressFile) & file.exists(tsFile)) {
            print(paste("Not overwriting output for", strftime(ts, tz="UTC")))
            next
        }

        print(ts)

        knownPoints = detrendedGeo[which(detrendedGeo$POSIXtime == ts),]
       
        ## Mask out dry pixels; keep only pixels that dryPixels says are rainy.
        ## Expects a perfect overlap with requested points and dryPixels.
        if(gridded) {
            lookup = data.table(POSIXtime=ts, rainy=1)
            setkey(lookup, POSIXtime, rainy)
            masked = dryPixels[lookup, nomatch=0] ## Rainy pixels.
            masked[, x := round(x, 1)]
            masked[, y := round(y, 0)]
            setkey(masked, x, y)
            atCoords[, x := round(x, 1)]
            atCoords[, y := round(y, 0)]
            setkey(atCoords, x, y)
            masked = masked[atCoords][rainy == TRUE]
            if(dim(masked)[1] == 0) {
                warning(paste("No radar-detected rain in interpolation region for time",
                              strftime(ts, tz="UTC")))
                next
            }
            coordinates(masked) = ~x+y
            proj4string(masked) = metresCRS
        } else {
            masked = at
        }

        grid = interpolateDSDForTime(data=detrendedGeo, time=ts, 
                                     dsdCols=dsdCols, nsim=nsim,
                                     stations=stations, varios=varios, 
                                     pca=pca, radarDir=radarDir,
                                     dryDriftModels=dryDriftModels, 
                                     timeRes=timeRes,
                                     radarElevation=radarElevation, 
                                     addRainStats=FALSE,
                                     grid=at, at=masked, 
                                     dryPixels=dryPixels, ...)

        dsds = data.table(as.data.frame(grid$gridDSDs))
        dsds$POSIXtime = ts
        rm(list="grid")
       
        if(!is.null(progressFile)) {
            save(dsds, file=tsFile)           
            ## if(object_size(results) > 1e9) {
            ##   save(results, file=paste(progressFile, ".", counter, ".Rdata", sep=""))
            ##   counter = counter + 1
            ##   results = NULL
            ##   gc()
            ## }
        } else {
            results = rbind(results, dsds)
        }
    }

    return(list(results=results, dryDriftModels=dryDriftModels, pca=pca, varios=varios))
}
