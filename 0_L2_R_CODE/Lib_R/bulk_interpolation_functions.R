# bulk_interpolation_functions.R
#
# "Ordinary" interpolation functions.
#
# Author: Tim Raupach <tim.raupach@epfl.ch>

require(ggplot2)
require(gstat)
require(scales)
source("library/dsd_interpolation_functions.R")

interpolateField = function(variogram, xRes, yRes, buffer, knownData,
                            idxCentrePoint=numeric(0),
                            knownDataColumnName="var",
                            xRange = range(coordinates(knownData)[,1]),
                            yRange = range(coordinates(knownData)[,2]),
                            transform=identity,
                            backTransform=identity,
                            excludePoints=numeric(0)) {
  # Interpolate a field of data using kriging and a given variogram.
  #
  # Args:
  #  variogram: Variogram to use for interpolation. Use
  #             fittedRealisationVariogram() to calculate a sample
  #             variogram from data if required.
  #  xRes: The x resolution (same units as range).
  #  yRes: The y resolution (same units as range).
  #  buffer: The spatial buffer around the ranges for which to
  #          interpolate (in spatial units). Vector of
  #          c(x, y).
  #  knownData: Known data points. An sp object with coordinates in same
  #             units as all other coords.
  #  idxCentrePoint: The index in knownData of the point to use as the centre
  #                  of the interpolated frame. If not specified then the
  #                  bounding box of knownData is used.
  #  knownDataColumnName: The column in knownData to be used as known data
  #                       values.
  #  xRange: The x-axis range of the field to produce. By default this is the
  #          range of the known points.
  #  yRange: The y-axis range of the field to produce. By default this is the
  #          range of the known points.
  #  transform: Transform the data using this function (default: none).
  #  backTransform: Back-transform the results using this function
  #                 (default: none).
  #
  # Returns: An sp object containing the interpolated values at each
  #  point in the field as val1.pred, and that points estimation variance
  #  as val1.var.

  stopifnot(length(buffer) == 2)

  if(length(idxCentrePoint) > 0) {
    centreX = coordinates(knownData)[idxCentrePoint,1]
    centreY = coordinates(knownData)[idxCentrePoint,2]
    xRange = c(centreX - buffer[1], centreX + buffer[1])
    yRange = c(centreY - buffer[2], centreY + buffer[2])
    buffer = c(0,0)
  }

  # Remove duplicate points.
  stopifnot(identical(remove.duplicates(knownData), knownData))

  # Transform the data if required.
  knownData$var = transform(knownData[[knownDataColumnName]])

  # Make a grid of points to estimate at.
  xVals = seq(min(xRange)-buffer[1], max(xRange)+buffer[1], by=xRes)
  yVals = seq(min(yRange)-buffer[2], max(yRange)+buffer[2], by=yRes)
  points = expand.grid(xVals, yVals)
  coordinates(points) = ~Var1+Var2
  proj4string(points) = proj4string(knownData)

  # Remove NAs in known data.
  if(any(is.na(knownData[[knownDataColumnName]]))) {
    print("Warning: NAs removed from known data points during kriging.")
    idx = which(is.na(knownData[[knownDataColumnName]]))
    knownData = knownData[-idx,]
  }

  # Find the predicted values using ordinary kriging.
  predictions = krige(var~1, knownData, newdata=points, model=variogram)

  # Back-transform if required.
  predictions$var1.pred = backTransform(predictions$var1.pred)

  # Return the predicted values and locations.
  return(predictions)
}

interpolateVariable = function(data, variograms, variable,
                               xRes=50, yRes=50, buffer=1000,
                               transform=log,
                               backTransform=exp,
                               indicatorVariogram=numeric(0),
                               indicatorObsName=numeric(0),
                               idxCentrePoint=numeric(0),
                               removeZeros=TRUE,
                               maskWithRadar=FALSE,
                               radarDir=NULL,
                               radarElevation=4,
                               radarTime=NULL,
                               radarTimeRes=NULL,
                               ...) {
  # Interpolate a field with an optional indicator field.
  #
  # Args:
  #  data: The data as a spatial object.
  #  variograms: Variogram list, one entry per variable. Entry should contain
  #              $variogram and $model.
  #  variable: Which variable to interpolate?
  #  xRes: X resolution [m] (default: 50).
  #  yRes: Y resolution [m] (default: 50).
  #  buffer: Buffer around region with data [m] (default: 1000).
  #          If idxCentrePoint is specified, the buffer because the buffer
  #          around that data point [m].
  #  transform: Transform the data using this function (default: none).
  #  backTransform: Back-transform the results using this function
  #                 (default: none).
  #  indicatorVariogram: Optional variogram (model) for an indicator field.
  #  indicatorObsName: The name of the indicator observations in data.
  #  idxCentrePoint: The index in data of the point to use as the centre of the
  #                  interpolated frame (by default the bounding box of the
  #                  data is used).
  #  removeZeros: Set zeros to NA in data before interpolation? (Default: TRUE).
  #  maskWithRadar: Mask output using a radar file? (Default: FALSE).
  #  radarDir: Directory in which to find radar files.
  #  radarElevation: Elevation of radar scans to look at.
  #  radarTime: The time to look for in the radar set (POSIXct, UTC).
  #  radarTimeRes: Time resolution [s] of the data, to look for scans within
  #                this amount of time from radarTime.
  #
  # Returns: The interpolated field, its estimation variance, and
  #          indicator and its estimation variance if an indicator variogram
  #          and observations are provided.

  if(removeZeros) {
    data[[variable]][which(data[[variable]] == 0)] = NA
  }

  # Krige to find gridded data.
  stopifnot(!is.null(variograms[[variable]]$model))
  interpolation = interpolateField(variogram=variograms[[variable]]$model,
                                   knownData=data,
                                   knownDataColumnName=variable,
                                   xRes=xRes, yRes=yRes,
                                   buffer=c(buffer, buffer),
                                   idxCentrePoint=idxCentrePoint,
                                   transform=transform,
                                   backTransform=backTransform)

  fields = list(interpolation=interpolation)

  # If an indicator field and variogram are supplied...
  if(length(indicatorVariogram) > 0 &
       length(indicatorObsName) > 0) {
    print("Using indicator field.")

    # Make sure the indicator field is binary.
    stopifnot(all(unique(data[[indicatorObsName]]) %in% c(0, 1, NA)))

    # Krige the indicator field, using the indicator variogram, to get a
    # probability field with the right spatial structure.
    indicatorInterp = interpolateField(variogram=indicatorVariogram,
                                       knownDataColumnName=indicatorObsName,
                                       knownData=data,
                                       xRes=xRes, yRes=yRes,
                                       buffer=c(buffer, buffer),
                                       idxCentrePoint=idxCentrePoint,
                                       transform=identity,
                                       backTransform=
                                         function(x) {as.numeric(x > 0.5)})

    fields$indicator = indicatorInterp

    # Apply the indicator field as a mask.
    fields$interpolation$var1.pred =
      fields$interpolation$var1.pred*indicatorInterp$var1.pred
    fields$interpolation$var1.var =
      fields$interpolation$var1.var*indicatorInterp$var1.var
  }

  # If required, mask using a radar field.
  if(maskWithRadar) {
    fields$interpolation = maskGrid(grid=fields$interpolation, time=radarTime,
                                    radarDir=radarDir, timeRes=radarTimeRes,
                                    radarElevation=radarElevation)
  }

  return(fields)
}

interpMaxRanges = function(data, start, end, variable, variograms,
                           indicatorVariogram=numeric(0),
                           indicatorObsName=numeric(0), ...) {
  # Interpolate a field for the timestep with the greatest range in a variable.
  #
  # Args:
  #  data: Data to interpolate from.
  #  start, end: Start and end times (POSIXct, UTC).
  #  variable: Which variable to interpolate?
  #  variograms: List of variograms; one per variable.
  #  indicatorVariogram: Optional indicator variogram to use (default: none).
  #  indicatorObsName: The name of the data column used for indication
  #                    (default: none).
  #  ...: Further argumentes to interpolateVariable()
  #
  # Returns: a list containing interpolation field, the estimation variance,
  # and the timestep for which the interpolation was performed.

  # Find the range in the variable for each timestep.
  subset = as.data.frame(data[which(data$POSIXtime >= start &
                                      data$POSIXtime <= end),])
  subset$timeFactor = as.factor(strftime(subset$POSIXtime,
                                         "%Y-%m-%d %H:%M:%S", tz="UTC"))
  subset$var = subset[[variable]]
  ranges = ddply(subset, .(timeFactor),
                 summarise, min=min(var, na.rm=T), max=max(var, na.rm=T),
                 range=diff(range(var, na.rm=T)))

  # Which timestep has the largest variable range?
  ranges$range[is.infinite(ranges$range)] = NA
  maxRange = which.max(ranges$range)
  maxIdx = which(data$POSIXtime == subset$POSIXtime[maxRange] &
                   !is.na(data[[variable]]))

  # Interpolate for the maximum range.
  if(length(maxIdx) == 0) {
    print("No maximum range; all ranges are NA.")
    return()
  }

  maxTimestep = subset$timeFactor[maxRange]
  interpMax = interpolateVariable(data=data[maxIdx,],
                                  variograms=variograms, variable=variable,
                                  indicatorVariogram=indicatorVariogram,
                                  indicatorObsName=indicatorObsName, ...)
  return(c(interpMax,
           list(maxTimestep=as.POSIXct(maxTimestep, tz="UTC"))))
}

interpolateSeries = function(data, variable, start, end,
                             variograms, statName, outDir,
                             textSize=28, indicatorVariogram=numeric(0),
                             indicatorObsName=indicatorObsName,
                             filePrefix="frame",
                             ...) {
  # Output a series of interpolations as image files ready to be
  # animated.
  #
  # Args:
  #  data: Data to interpolate from.
  #  start, end: Start and end times (POSIXct, UTC).
  #  variable: Which variable to interpolate?
  #  statName: The long name (including unit) of the variable. For plots.
  #  outDir: Output directory. Must already exist.
  #  variograms: List of variograms; one per variable.
  #  indicatorVariogram: Optional indicator variogram to use (default: none).
  #  indicatorObsName: The name of the data column used for indication
  #                    (default: none).
  #  textSize: Plot text size (default: 28).
  #  filePrefix: Output image filenmae prefix (default: "frame").
  #  ...: Further argumentes to interpolateVariable()
  #
  # Returns: void.

  subset = subset(data, POSIXtime >= start & POSIXtime <= end)

  times = unique(subset$POSIXtime)
  i = 1
  for(time in times[order(times)]) {
    idx = which(subset$POSIXtime == time)

    print(paste("Frame", i, "of", length(times)))
    if(all(is.na(subset[[variable]][idx]) | subset[[variable]][idx] < 0.1)) {
      next
    }

    maxTimestep = subset$POSIXtime[idx[1]]
    interpMax = interpolateVariable(data=subset[idx,],
                                    variograms=variograms, variable=variable,
                                    indicatorVariogram=indicatorVariogram,
                                    indicatorObsName=indicatorObsName,
                                    ...)

    stop("Add plotting code")
    if(any(interpMax$prediction) > 0) {
      outFile = paste(outDir, "/", filePrefix, "_", i, ".png", sep="")
      ggsave(filename=outFile, plot=interpMax$plots$finalPrediction)
      i = i + 1
    }
  }
}

interpolateAndPlotBulk = function(variables, transforms, backtransforms, data,
                                  varUnits, start, end, ts, stations,
                                  useZeroDistNugget=TRUE,
                                  models=c("Sph", "Lin"), useCressie=TRUE,
                                  buffer=c(1000, 1000), xRes=100, yRes=100,
                                  knownPointSize=2, textSize=10,
                                  zlims=numeric(0), ...) {
  # Get variograms for an event and plot a single timestep bulk variables
  # in one step.
  #
  # Args:
  #   variables: The variables to interpolate.
  #   transforms: Transformation functions for each variable.
  #   backtransforms: Backtransforms to undo each transform.
  #   data: The data to use, should contain all variables by timestep and
  #         station.
  #   varUnits: The variable units - expressions named by variable.
  #   start, end: Start and end times for variograms (POSIXct, UTC).
  #   ts: The timestep to plot for.
  #   stations: Stations definition.
  #   userZeroDistNugget: Use the zero distance nuggets?
  #   models: Variogram models to try. By default, Sph and Exp.
  #   useCressie: Use Cressie's robust estimator?
  #   buffer: Buffer to use around the known points? (Default: 1000 m).
  #   knownPointSize: Size for known points in the plots.
  #   xRes, yRes: x and y resolutions (default: 100 m).
  #   zlims: List of z limits per graph, named by variable (default: auto).
  #   ...: Extra arguments to plotInterpolatedField().
  #
  # Returns: A data.frame containing varios, interpolated fields, and plots
  #          for the timestep.

  # Get variograms.
  basicVarios = eventVariograms(data=data, start=start, end=end,
                                useZeroDistNugget=useZeroDistNugget,
                                stations=stations, transform=transforms,
                                useCressie=useCressie, variables=variables,
                                models=models, ...)

  # Plot for fields.
  res = interpolateAndPlotUsingVario(variables=variables, transforms=transforms,
      backtransforms=backtransforms, data=data, varUnits=varUnits,
      start=start, end=end, ts=ts, stations=stations, varios=basicVarios,
      useZeroDistNugget=useZeroDistNugget, models=models, useCressie=useCressie,
      buffer=buffer, xRes=xRes, yRes=yRes, knownPointSize=knownPointSize,
      textSize=textSize, zlims=zlims, ...)

  return(list(varios=basicVarios, plots=res$plots, fields=res$fields))
}

interpolateAndPlotUsingVario = function(variables, transforms, backtransforms, data,
                                        varUnits, start, end, ts, stations, varios,
                                        useZeroDistNugget=TRUE,
                                        models=c("Sph"), useCressie=TRUE,
                                        buffer=c(1000, 1000), xRes=100, yRes=100,
                                        knownPointSize=2, textSize=10,
                                        zlims=numeric(0), ...) {
  # Plot single timestep bulk variables in one step using given varigrams.
  #
  # Args:
  #   variables: The variables to interpolate.
  #   transforms: Transformation functions for each variable.
  #   backtransforms: Backtransforms to undo each transform.
  #   data: The data to use, should contain all variables by timestep and
  #         station.
  #   varUnits: The variable units - expressions named by variable.
  #   start, end: Start and end times for variograms (POSIXct, UTC).
  #   ts: The timestep to plot for.
  #   stations: Stations definition.
  #   varios: Variograms to use.
  #   userZeroDistNugget: Use the zero distance nuggets?
  #   models: Variogram models to try. By default, Sph and Exp.
  #   useCressie: Use Cressie's robust estimator?
  #   buffer: Buffer to use around the known points? (Default: 1000 m).
  #   knownPointSize: Size for known points in the plots.
  #   xRes, yRes: x and y resolutions (default: 100 m).
  #   zlims: List of z limits per graph, named by variable (default: auto).
  #   ...: Extra arguments to plotInterpolatedField().
  #
  # Returns: A data.frame containing interpolated fields, and plots for
  #          the timestep.

  basicKnownData = subset(data, POSIXtime == ts)
  basicKnownData = addStationLocations(basicKnownData, stations=stations)
  basicKnownData$realisation = 1
  basicKnownData = replaceZeroDistPoints(basicKnownData,
      variables=variables,
      useCressie=useCressie)$data

  basicInterpPlots = list()
  basicInterps = list()
  for(var in variables) {
    z = zlims
    if(length(z) != 0) {
      z = zlims[[var]]
    }

    basicInterps[[var]] =
      interpolateVariable(data=basicKnownData, variograms=varios,
                          variable=var, xRes=xRes, yRes=yRes,
                          buffer=buffer[1],
                          transform=get(transforms[[var]]),
                          backTransform=get(backtransforms[[var]]), ...)
    basicInterpPlots[[var]] =
      plotInterpolatedField(predictions=basicInterps[[var]]$interpolation,
                            statName=paste(var, "~group('[',", varUnits[[var]],
                                           ",']')", sep=""),
                            knownData=basicKnownData,
                            title=paste(var, "interp"),
                            textSize=textSize, zlims=z,
                            knownPointSize=knownPointSize, ...)$prediction
  }

  return(list(plots=basicInterpPlots, fields=basicInterps))
}
