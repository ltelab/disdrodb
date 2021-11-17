# variogram-functions.R
#
# Functions to do with variograms and their use; including interpolation by
# kriging.
#
# Author: Tim Raupach <tim.raupach@epfl.ch>

library(ggplot2)
library(gstat)
library(scales)
source("library/network_functions.R")

# Marc's colour definition. Used as default interpolation plot colours.
Ncol     <- 200
col.Rbar <- c("darkblue","blue3","blue1","dodgerblue","deepskyblue","cyan")
col.Rbar <- c(col.Rbar,"yellow","gold","orange","red1","red3","darkred")
col.Rbar <- colorRampPalette(col.Rbar)
col.Rbar <- col.Rbar(Ncol)

plotAnisotropyTest = function(data, variableName, variableLongName,
                              realisationVarName, distBoundaries, model="Sph",
                              angles=c(0,45,90), tol.hor=20,
                              variableUnits="", distUnits="m", subtitle="") {
  # Plot fitted models for 0 degrees, 45 degrees, 90 degrees and 135 degrees
  # against each other. If they are about the same then isotropy can be
  # assumed.
  #
  # Args:
  #  data: Spatial data.frame containing all data.
  #  variableName: name of observed variable to calculate variogram for.
  #  variableLongName: long name of the variable (for plot title).
  #  realisationVarName: name of factor within data that is unique per
  #                      realisation.
  #  boundaries: distance boundaries to use for variogram calculation.
  #  model: model to fit to variogram (see ?vgm default: linear).
  #  angles: angles to test for (default: 0, 45, 90).
  #  tol.hor: horizontal tolerance in angle for each angle (default: 20).
  #  variableUnit: unit for the variable, parseable.
  #  distUnit: unit for the distance.
  #  subtitle: a subtitle.
  #
  # Returns:
  #  A ggplot2 plot ready to display.

  # Find the variogram for each angle.
  angleModels = testAnisotropy(data, variableName, realisationVarName,
                          distBoundaries, model=model, angles=angles,
                          tol.hor=tol.hor)

  variograms = NULL
  models = NULL

  for(i in seq(1, length(angleModels))) {
    variograms = c(variograms, list(angleModels[[i]]$variogram))
    models = c(models, list(angleModels[[i]]$model))
  }

  plot = ggplotVariograms(varios=variograms,
                          variogramDescs=paste(angles, "+/-",
                                               tol.hor, "degrees"),
                          varName=variableName,
                          distUnits=distUnits,
                          varUnits=variableUnits,
                          models=models,
                          title=paste("Anisotropy test for ",
                                      variableLongName,
                                      "\n", subtitle, sep=""),
                          scaleTitle="Angle")
}

testAnisotropy = function(data, variableName, realisationVarName,
                          distBoundaries, model="Sph",
                          angles=c(0, 90, 180), tol.hor=45) {
  # Fit models to variograms for different angles (from north) to test for
  # anisotropy. Return just the fitted models.
  #
  #  data = data.frame containing all data.
  #  variableName = name of observed variable to calculate variogram for.
  #  realisationVarName = name of factor within data that is unique per
  #                       realisation.
  #  boundaries = distance boundaries to use for variogram calculation.
  #  model = model to fit to variogram (see ?vgm default: linear).
  #  angles = angles to test, from north (default: 0, 45, 90, 135, 180).
  #  tol.hor = horizontal tolerance in angle (default: 90).
  #
  # Returns:
  #  A list containing models for each tested angle, where model has
  #  an extra field "variogram_angle" added to it.

  res = NULL

  print("Testing variograms at different angles.")
  for(angle in angles) {
    print(paste("Angle:", angle, "degrees."))
    var = fittedRealisationVariogram(data, variableName, realisationVarName,
                                     distBoundaries, model, angle=angle,
                                     tol.hor=tol.hor, useZeroDistNugget=T)
    res = c(res, list(var))
  }

  return(res)
}

scaleNormalisedVariograms = function(variograms, data) {
  # Scale normalised variograms to fit the variances in data.
  #
  # data: Data to use, already subset.
  # variograms: A named list of variograms to scale. Each list entry must
  #             contain "variogram" and "model". The variograms should be
  #             normalised so the sill is at semivariance 1.
  #
  # Returns: A list of scaled variograms.

  for(variable in names(variograms)) {
    print(paste("Scaling normalised variogram for ", variable))

    # Find the variance of the field, to scale by.
    fieldVariance = var(data[[variable]], na.rm=T)

    # Multiply by the variance to scale the variogram.
    variograms[[variable]]$variogram$gamma =
      variograms[[variable]]$variogram$gamma * fieldVariance

    # Adjust the model too; leave the range the same but adjust the sill and
    # nugget by multiplying them by the variance.
    variograms[[variable]]$model$psill =
      variograms[[variable]]$model$psill * fieldVariance
  }

  return(variograms)
}

fittedRealisationVariogram = function(data, variableName,
                                      realisationVarName,
                                      boundaries,
                                      model=c("Sph", "Exp", "Gau", "Lin"),
                                      angle=0, tol.hor=90,
                                      removeZeros=TRUE,
                                      useZeroDistNugget=FALSE,
                                      useNugget=TRUE,
                                      normalise=FALSE,
                                      useCressie=TRUE, ...) {
  # Calculate a sample pooled variogram from data, using a variable to
  # separate different realisations of the same field. Fit a model to it.
  # Normalise it if required (this will normalise by dividing by the variance
  # each timestep).
  #
  # Args:
  #   data: data.frame containing all data to fit variogram to; must contain
  #         coordinates.
  #   variableName: name of observed variable to calculate variogram for.
  #   realisationVarName: name of factor within data that is unique per
  #                       realisation.
  #   boundaries: distance boundaries to use for variogram calculation.
  #   model: model(s) to fit to variogram (see ?vgm). If set to more than one
  #          model then they are all tried and the one with the lowest SSerr
  #          value is returned (default: c("Sph", "Exp", "Gau", "Lin"))
  #   angle: angle to use for variogram (default: 0).
  #   tol.hor: horizontal tolerance in angle (default: 90).
  #   removeZeros: Remove zeros from data before finding the variograms?
  #   useZeroDistNugget: Use the zero-distance empirical nugget? (Default: F).
  #   useNugget: Use a nugget at all? (Default: T, sometimes gives better
  #              model fits if F).
  #   normalise: Normalise the variogram by dividing each time step by its
  #              variance?
  #   useCressie: Use the cressie variogram estimator to better deal with
  #               outliers? (Default: TRUE).
  #
  # Returns: list containing variogram and fitted variogram model, a boolean
  # indicating whether the zero distance nugget was used, and the fitted model
  # name, the field variance, and a boolean indicating whether the
  # variogram has been normalised.

  # Warn if NAs will be removed.
  if(any(is.na(data[[variableName]]))) {
    print(paste("WARNING: Removing NAs for", variableName))
    data = data[-(which(is.na(data[[variableName]]))),]
  }

  if(normalise) {
    # Normalise each timestep by dividing by the square root of its variance.
    d = data.frame(realisation=data[[realisationVarName]],
                   variable=data[[variableName]])
    variances = ddply(d, .(realisation), summarise,
                      factor=sqrt(var(variable, na.rm=T)))
    for(i in seq(1, length(variances$realisation))) {
      r = variances$realisation[i]
      factor = variances$factor[i]

      if(is.na(factor)) {
        next
      }

      if(factor == 0) {
        next
      }

      idx = which(data[[realisationVarName]] == r)
      data[[variableName]][idx] = data[[variableName]][idx] / factor
    }

    # After normalisation, expect the sill to be close to 1.
    sill = 1
  } else {
    sill = var(data[[variableName]], na.rm=T)
  }

  # Always remove zero-distance points, replacing them with their mean value,
  # if they are present.
  nodup = NULL
  diffSum = 0
  numZeroDistPoints = 0

  # Remove zero-distance points for each realisation.
  res = replaceZeroDistPoints(data=data, variables=variableName,
                              realisationVarName=realisationVarName,
                              useCressie=useCressie)
  data = res$data

  if(useZeroDistNugget) {
    # Use empirical zero-distance semivariance as the nugget.
    fit.sills=c(F,T)
    nugget = as.numeric(res$nugget)
    print(paste("Using zero dist nugget:", nugget))
  } else {
    fit.sills=c(T,T)
    nugget = 1
  }

  # The realisationVarName column is a factor giving the different
  # realisations to use, and because dX < 1 each different value of
  # "realisation" is used as a separate realisation for the variogram
  # calculation.
  print("Calculating variogram.")
  vario = variogram(as.formula(paste(variableName, "~", realisationVarName)),
                    data=data, dX=0.5, boundaries=boundaries,
                    alpha=angle, tol.hor=tol.hor, cressie=useCressie)

  # Define the variogram model to fit, and fit it to the variogram.
  # Try each model and return the one with the lowest SSErr value.
  # Singular fits are not returned.
  print("Fitting models.")
  fits = NULL
  errors = NULL
  for(m in model) {
    if(useNugget) {
      print(paste(m, nugget, sill, max(boundaries)))
      vario.model = vgm(model=m,
                        nugget=nugget,
                        psill=sill,
                        range=max(boundaries))
    } else {
      vario.model = vgm(model=m,
                        psill=sill,
                        range=max(boundaries))
    }

    vario.fitted = fit.variogram(vario, vario.model, fit.sills=fit.sills)

    if(!attr(vario.fitted, "singular")) {
      errors = c(errors, attr(vario.fitted, "SSErr"))
      fits = c(fits, list(vario.fitted))
    } else {
      errors = c(errors, NA)
      fits = c(fits, NA)
    }
  }

  print("SSErr values for different models:")
  print(data.frame(model=model, errors))

  if(length(which(!is.na(errors))) == 0) {
    print("No non-singular fits found; trying a linear fit.")

    vario.model = vgm(model="Lin", nugget=nugget, sill=sill)
    vario.fitted = fit.variogram(vario, vario.model, fit.sills=fit.sills)

    if(!attr(vario.fitted, "singular")) {
      return(list(variogram=vario,
                  model=vario.fitted,
                  realnugget=useZeroDistNugget,
                  modelType="Lin"))
    }

    print("Even linear fit was singular, no fitted model will be returned.")
    return(list(variogram=vario,
                model=NA,
                realNugget=useZeroDistNugget,
                modelType=NA))
  }

  minErrIdx = which(errors == min(errors, na.rm=T))[1]
  vario.fitted = fits[[minErrIdx]]

  return(list(variogram=vario,
              model=vario.fitted,
              realNugget=useZeroDistNugget,
              modelType=model[minErrIdx],
              normalised=normalise))
}

ggplotVariogram = function(vario,
                           varName,
                           varSymbol,
                           distUnits,
                           varUnits,
                           model=numeric(0),
                           displayPoints=T,
                           displayModel=T,
                           title=paste("Variogram for", varName),
                           textSize=16,
                           trans="identity") {
  # Make a ggplot of a variogram and (optionally) a fitted variogram model.
  #
  # Args:
  #  vario: The variogram to plot (output of variogram()).
  #  varName: The name of the variable the variogram is for (long name).
  #  varSymbol: Variable's symbol.
  #  distUnits: Units for distance.
  #  varUnits: Units for variable (semi-variance).
  #  model: The variogram model to plot (output of fit.variogram())
  #         (Default: no model).
  #  displayPoints: Display variogram points on the plot? (Default: T).
  #  displayModel: Display variogram model on the plot? (Default: T).
  #  title: Title for plot (default "Variogram for <variable>").
  #
  # Returns: A ggplot2 object.

  if(displayModel & length(model) == 0) {
    stop("No variogram model provided.")
  }
  if(displayPoints == F & displayModel == F) {
    stop("Must display either points or model.")
  }

  points = NULL
  if(displayPoints) {
    points = geom_point(pch=1, cex=3)
  }

  line = NULL
  if(displayModel) {
    linePts = variogramLine(model, maxdist = max(vario$dist))
    line = geom_line(data=linePts, aes(y=gamma, x=dist))
  }

  if(varUnits!="log") {
    unitStr = paste("group('[',", varUnits, ",']')^2",
                    sep="")
  } else {
    unitStr = paste("(log)")
  }

  xlims = c(0, max(vario$dist))
  ylims = c(0, max(vario$gamma))

  if(nchar(varSymbol) > 0) {
    axisString = paste("Semivariance~of", varSymbol, sep="~")
  } else {
    axisString = "Semivariance"
  }
  varPlot = ggplot(vario, aes(x=dist, y=gamma)) +
    points + line +
    theme_bw(textSize) +
    labs(title=title,
         x=paste("Distance [", distUnits, "]", sep=""),
         y=parse(text=paste(axisString, unitStr, sep="~"))) +
    scale_y_continuous(trans=trans, limits=ylims) +
    scale_x_continuous(limits=xlims)

  return(varPlot)
}

ggplotVariograms = function(varios, variogramDescs,
                            varName, distUnits, varUnits,
                            models=NA,
                            displayPoints=T,
                            displayModels=T,
                            title=paste("Variograms~of", varName, sep="~"),
                            scaleTitle="Variograms",
                            legendPos="top",
                            legendArr="horizontal",
                            trans="identity",
                            lineTypes=rep(1, length(variogramDescs)),
                            lineTypeScaleName="",
                            textSize=16) {
  # Make a ggplot of multiple variograms for the same variable,
  # and (optionally) fitted variogram models for each variogram.
  #
  # Args:
  #  vario: List of variograms to plot (output of variogram()).
  #  variogramDescs: Description for each variogram.
  #  varName: The name of the variable the variogram is for.
  #  distUnits: Units for distance.
  #  varUnits: Units for variable (semi-variance).
  #  models: List of variogram models to plot (output of fit.variogram())
  #         (Default: no model).
  #  displayPoints: Display variogram points on the plot? (Default: T).
  #  displayModels: Display fitted models as lines on the plot? (Default: T).
  #  title: Title for plot (default "Variograms for <variable>").
  #  scaleTitle: Title for scale/legend (default: "Variograms").
  #  legendPos: Legend position (theme(legend.position)) (default: top).
  #  legendArr: Legend arrangement (default: horizontal).
  #  trans: Transform for plot y axis? (Default: none).
  #  lineTypes: Vector of unique values per line type corresponding to
  #             each variogram.
  #  lineTypeScaleName: Scale name for line type.
  #  textSize: Text size for the plot (default: 16).
  #
  # Returns:
  #  A ggplot2 object.

  if(displayModels & (length(models) != length(varios))) {
    stop("Variogram models not provided for all variograms.")
  }
  if(displayPoints == F & displayModels == F) {
    stop("Must display either points or models.")
  }

  combVarios = NULL
  combLines = NULL
  for(i in seq(1, length(varios))) {
    varios[[i]]$desc = variogramDescs[i]
    combVarios = rbind(combVarios, varios[[i]])

    if(displayModels & !any(is.na(models[[i]]))) {
      linePts = variogramLine(models[[i]], maxdist = max(varios[[i]]$dist))
      linePts$desc = variogramDescs[i]
      linePts$lineType = lineTypes[i]
      combLines = rbind(combLines, linePts)
    }
  }
  combLines$lineType = factor(combLines$lineType)

  points = NULL
  if(displayPoints) {
    points = geom_point(pch=1, cex=4, group="desc", aes(colour=desc))
  }

  lines = NULL
  if(displayModels) {
    if(length(unique(combLines$lineType)) == 1) {
      lines = geom_line(data=combLines,
                        aes(y=gamma, x=dist, colour=desc),
                        group="desc", size=1)
    } else {
      # Different different line types.
      lines = geom_line(data=combLines,
                        aes(y=gamma, x=dist, colour=desc, lty=lineType),
                        group="desc", size=1)
    }
  }

  unitStr = paste("(log)")
  if(varUnits!="log") {
    unitStr = paste("group('[',", varUnits, ",']')^2",
                    sep="")
  }

  xlims = c(0, max(combVarios$dist))
  ylims = c(0, max(combVarios$gamma))

  axisString=paste("Semivariance~of", varName, sep="~")
  varPlot =
    ggplot(combVarios, aes(x=dist, y=gamma)) +
    points +
    lines +
    theme_bw(textSize) +
    labs(title=parse(text=title),
         x=paste("Distance [", distUnits, "]", sep=""),
         y=parse(text=paste(axisString, unitStr, sep="~"))) +
    scale_colour_discrete(name=scaleTitle) +
    scale_y_continuous(trans=trans, limits=ylims) +
    scale_x_continuous(limits=xlims) +
    scale_linetype_discrete(name=lineTypeScaleName) +
    theme(legend.position=legendPos)

  return(varPlot)
}

convertToProj4 = function(data,
                 proj4="+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs") {
  # Convert dataset coordinates into a different projection.
  #
  # Args:
  #  data: SP object with coordinates to convert.
  #  proj4: Proj4 string for coordinates to convert to (default: WGS84).
  #
  # Returns:
  #  The same object with coordinates in a different system.

  data = spTransform(data, CRS(proj4))
  return(data)
}

plotInterpolatedField = function(predictions, knownData, title=numeric(0),
                                 xLab="x [km]", yLab="y [km]",
                                 statName="R~[mm~h^{-1}]",
                                 convertToLatLong=F,
                                 plotKnownPoints=T, kmScale=T,
                                 colours=c(col.Rbar),
                                 logZ=F, textSize=20, zlims=numeric(0),
                                 includeAllKnownPoints=FALSE,
                                 knownPointSize=6, ...) {
  # Plot an interpolated field.
  #
  # Args:
  #  predictions: Interpolated field (output from interpolateField()), must
  #               contain val1.pred and val1.var and coordinates. NB: Coords
  #               are assumed to be in a regular grid!
  #  knownData: The known data points used to make the predictions.
  #  title: The title to use, or by default no title.
  #  xLab: Label for x-axis (default: Longitude).
  #  yLab: Label for y-axis (default: Latitude).
  #  statName: The statitic's name (default: R [mm/h]).
  #  convertToLatLong: Convert coordinates to lat/long before plotting?
  #                    (Default: T).
  #  plotKnownPoints: Plot known points as dots?
  #  kmScale: Use kilometers from origin as scale? Assumes coordinates
  #           are in m.
  #  logZ: Use a log scale for the Z values?
  #  textSize: Size of text in plots (default: 20).
  #  zlims: Z limits (default: range of Z values).
  #  knownPointSize: The size of markers for the known points (default: 6).
  #
  # Returns:
  #  A list containing the prediction plot ("prediction") and estimation
  #  variance plot ("estimationVariance").

  x = coordinates(predictions)[,1]
  y = coordinates(predictions)[,2]
  xRes = resolution(x)
  yRes = resolution(y)

  mins = data.frame(x=x-(xRes/2), y=y-(yRes/2))
  maxs = data.frame(x=x+(xRes/2), y=y+(yRes/2))
  coordinates(mins) = ~x+y
  coordinates(maxs) = ~x+y
  proj4string(mins) = proj4string(predictions)
  proj4string(maxs) = proj4string(predictions)

  # Convert predicted coordinates into lat and long coordinates?
  if(convertToLatLong) {
    mins = convertToProj4(mins)
    maxs = convertToProj4(maxs)
    predictions = convertToProj4(predictions)
    knownData = convertToProj4(knownData)
  }

  plotTitle = ""
  estTitle = ""
  if(length(title) > 0) {
    plotTitle = title
    estTitle = paste(title, "\nEstimation variance", sep="")
  }

  # Convert to a data frame for easier handling.
  plotData = data.frame(x=coordinates(predictions)[,1],
                        y=coordinates(predictions)[,2],
                        predictedVal=predictions$var1.pred,
                        estimationVariance=predictions$var1.var)
  knownPoints = data.frame(x=coordinates(knownData)[,1],
                           y=coordinates(knownData)[,2])

  theme = theme_bw(textSize)
  estLabels = labs(x=xLab, y=yLab, title=estTitle)
  plotLabels = labs(x=xLab, y=yLab, title=plotTitle)

  if(!includeAllKnownPoints) {
    idx = which(knownPoints$x >= min(plotData$x) &
                knownPoints$x <= max(plotData$x) &
                knownPoints$y >= min(plotData$y) &
                knownPoints$y <= max(plotData$y))
    knownPoints = knownPoints[idx,]
  }

  known_points = geom_point(pch=17, col="black", data=knownPoints,
                            aes(x=x, y=y), size=knownPointSize)

  pixelWidth = coordinates(maxs)[,1] - coordinates(mins)[,1]
  pixelHeight = coordinates(maxs)[,2] - coordinates(mins)[,2]

  if(length(zlims) == 0) {
    zlims = NULL # range(plotData$predictedVal, na.rm=TRUE)
  }

  predictionScale = scale_fill_gradientn(colours=colours, na.value="grey",
                                         name=parse(text=statName), limits=zlims)
  estScale = scale_fill_gradientn(colours=colours, name="Est. var",
                                  na.value="white")

  lineScale = scale_colour_gradientn(colours=colours, na.value="grey",
                                     name=parse(text=statName), limits=zlims,
                                     guide=FALSE)
  estLineScale = scale_colour_gradientn(colours=colours, na.value="white",
                                        name="Est. var")

  if(logZ) {
    stop("Needs proper implementation...")
  }

  # First plot the predicted values.
  predictionPlot = ggplot(data=plotData, aes(x=x, y=y)) +
    geom_tile(aes(fill=predictedVal, colour=predictedVal),
              width=pixelWidth, height=pixelHeight) +
    predictionScale + lineScale + theme + plotLabels

  # Then the estimation variance.
  estVarPlot = ggplot(data=plotData, aes(x=x, y=y)) +
    geom_tile(aes(fill=estimationVariance, colour=estimationVariance),
              width=pixelWidth, height=pixelHeight) +
    theme + estLabels + estLineScale + estScale

  if(kmScale) {
    xScale = scale_x_continuous(breaks=seq(min(x), max(x), length.out=5),
                                labels=round((seq(min(x), max(x), length.out=5)
                                              - min(x)) / 1000, 2))
    yScale = scale_y_continuous(breaks=seq(min(y), max(y), length.out=5),
                                labels=round((seq(min(y), max(y), length.out=5)
                                              - min(y)) / 1000, 2))
    predictionPlot = predictionPlot + xScale + yScale
    estVarPlot = estVarPlot + xScale + yScale
  }

  if(plotKnownPoints) {
    predictionPlot = predictionPlot + known_points
    estVarPlot = estVarPlot + known_points
  }

  return(list(prediction=predictionPlot, estimationVariance=estVarPlot))
}

plotGrid = function(data, variable, varUnit,
                    knownData=numeric(0), plotKnownPoints=TRUE,
                    zlims=numeric(0), colours=c(col.Rbar), textSize=16,
                    xLab="x [km]", yLab="y [km]", title=numeric(0),
                    kmScale=TRUE, knownPointSize=6,
                    varName=variable, xRes=NULL, yRes=NULL) {
  # Plot a gridded set of values.
  #
  # Args: see plotInterpolatedField().
  # Can override xRes and yRes by specifying, otherwise found automatically.
  #
  # Returns: ggplot object.

  x = coordinates(data)[,1]
  y = coordinates(data)[,2]
  if(is.null(xRes))
      xRes = resolution(x)

  if(is.null(yRes))
      yRes = resolution(y)

  mins = data.frame(x=x-(xRes/2), y=y-(yRes/2))
  maxs = data.frame(x=x+(xRes/2), y=y+(yRes/2))
  coordinates(mins) = ~x+y
  coordinates(maxs) = ~x+y
  proj4string(mins) = proj4string(data)
  proj4string(maxs) = proj4string(data)

  pixelWidth = coordinates(maxs)[,1] - coordinates(mins)[,1]
  pixelHeight = coordinates(maxs)[,2] - coordinates(mins)[,2]

  plotData = data.frame(x=coordinates(data)[,1],
                        y=coordinates(data)[,2],
                        val=data[[variable]])

  if(length(knownData) != 0) {
    knownPoints = data.frame(x=coordinates(knownData)[,1],
                            y=coordinates(knownData)[,2])
    known_points = geom_point(pch=17, col="black", data=knownPoints,
                              aes(x=x, y=y), size=knownPointSize)
  }

  if(length(zlims) == 0) {
    zlims = range(plotData$val)
  }

  if(length(title) == 0) {
    plotTitle = NULL
  } else {
    plotTitle = title
  }

  statName = parse(text=paste(varName, "~group('[',", varUnit, ",']')",
                              sep=""))
  valScale = scale_fill_gradientn(colours=colours, na.value="grey",
                                  name=statName, limits=zlims)
  lineScale = scale_colour_gradientn(colours=colours, na.value="grey",
                                     name=statName, limits=zlims)
  plotLabels = labs(x=xLab, y=yLab, title=plotTitle)

  plot = ggplot(data=plotData, aes(x=x, y=y)) +
    geom_tile(aes(fill=val, colour=val), width=pixelWidth, height=pixelHeight) +
    valScale + lineScale + theme_bw(textSize) + plotLabels

  if(kmScale) {
    xScale = scale_x_continuous(breaks=seq(min(x), max(x), length.out=5),
                                labels=round((seq(min(x), max(x), length.out=5)
                                              - min(x)) / 1000, 2), expand=c(0,0))
    yScale = scale_y_continuous(breaks=seq(min(y), max(y), length.out=5),
                                labels=round((seq(min(y), max(y), length.out=5)
                                              - min(y)) / 1000, 2), expand=c(0,0))
    plot = plot + xScale + yScale
  }

  if(length(knownData) > 0 & plotKnownPoints) {
    plot = plot + known_points
  }
  
  plot = plot + theme(panel.grid.major=element_blank(),
                      panel.grid.minor=element_blank())

  return(plot)
}

exampleVariogram = function() {
  # Plot a variogram for illustration purposes.
  #
  # Returns: ggplot object.

  data(meuse)

  coordinates(meuse) = ~x+y
  var = variogram(log(zinc)~1, meuse)
  model = vgm(1, "Sph", 700, 1)
  v.fit = fit.variogram(var, model)

  plot = ggplotVariogram(var, varName="", varUnit="log", varSymbol="",
                  distUnits="m", model=v.fit, title="Example variogram",
                  textSize=20)
  return(plot)
}

#############################################################################

eventTypeVariogram = function(data, type,
                              variables=c("R", "Nt", "D0", "Zh"),
                              varUnits=c("mm~h^{-1}", "m^{-3}", "mm"),
                              varNames=c("rain rate", "total drop concentration",
                                         "median drop diameter"),
                              timeRes=numeric(0),
                              events=SOPevents(),
                              stations=stationsDefinition(),
                              transform=identity,
                              models="Sph",
                              distUnits="m",
                              plotTrans="identity",
                              trimSeconds=0) {
  # Return a variogram for all timesteps which are in events of a certain type,
  # ie convective or stratiform.
  #
  # Args:
  #   data: The data to fit to.
  #   outputPrefix: Prefix for output files.
  #   type: The event type to match.
  #   events: List of events, each must contain start, end and type.
  #   variables: Variables to find variograms for.
  #   varUnits: Units for each variable.
  #   varNames: Long names for each variable.
  #   timeRes: The time resolution to resample to.
  #   stations: Stations definition (default: HYMEX stations).
  #   transform: Transform data before finding variograms (function)?
  #   models: Models to fit (if more than one, the one with
  #           the lowest error will be returned). (Default: Sph).
  #   distUnits: Distance units (default: "m").
  #   plotTrans: Translation for plot y axis (string, name of function)?
  #              (Default: none).
  #   trimSeconds: If > 0, remove this many seconds from start and end of
  #               each event before variogram calculation. (Default: 0).
  #
  # Returns: The variogram object.

  # Find all timesteps which are part of each event of the requested type.
  events = events[which(events$type == type),]
  idx = NULL
  for(e in seq(1, length(events$start))) {
    start = events$start[e] + trimSeconds
    end = events$end[e] - trimSeconds

    idx = c(idx, which(data$POSIXtime >= start & data$POSIXtime <= end))
  }

  subset = data[idx,]
  varios = eventVariograms(subset, min(data$POSIXtime), max(data$POSIXtime),
                           variables, timeRes, stations, transform, models)
  return(varios)
}

allEventVariograms = function(data,
                              variables=c("R", "Nt", "D0", "Zh"),
                              timeRes=numeric(0),
                              events=SOPevents(),
                              stations=stationsDefinition(),
                              transform=rep("identity", length(variables)),
                              models="Sph",
                              trimSeconds=0,
                              indicator=FALSE,
                              ...) {
  # Calculate all variograms for a set of variables and a set of events.
  #
  # Args:
  #   data: The data to fit to.
  #   variables: Variables to find variograms for.
  #   timeRes: The time resolution to resample to.
  #   events: List of events, each must contain start and end.
  #   stations: Stations definition (default: HYMEX stations).
  #   transform: Transform data before finding variograms (function)?
  #   models: Models to fit (if more than one, the one with
  #           the lowest error will be returned). (Default: Sph).
  #   trimSeconds: If > 0, remove this many seconds from start and end of
  #               each event before variogram calculation. (Default: 0).
  #   ...: Further arguments to fittedRealisationVariogram().
  #
  # Returns: A list of length n where there are n events. Each element of
  #          this list is itself a list, with one element (the variogram) per
  #          variable.

  result = list()

  # Find the variogram for each event.
  for(e in seq(1, length(events$start))) {
    start = events$start[e] + trimSeconds
    end = events$end[e] - trimSeconds
    stopifnot(start < end)

    varios = eventVariograms(data, start, end, variables,
                             timeRes, stations, transform,
                             models, ...)

    result = c(result, list(varios))
  }

  return(result)
}

collectVariogramsByVariable = function(variogramsList, variable) {
  # Given a list of variograms by event, collect together all the
  # variograms and models for a particular variable.
  #
  # Args:
  #   variogramsList: A list of variograms like that returned from
  #                   allEventVariograms().
  #   variable: The name of the variable to collect for.
  #   getEvents: Which event numbers to get (Default: all SOP events)?
  #
  # Returns: a list containing all the variograms and all the models for
  #          the variable.

  variograms = list()
  models = list()

  for(variogramSet in variogramsList) {
    variograms = c(variograms, list(variogramSet[[variable]]$variogram))
    models = c(models, list(variogramSet[[variable]]$model))
  }

  return(list(variograms=variograms, models=models))
}

singlePlotWithAllVariograms = function(variogramsList, variable, varName,
                                       varUnit, events=SOPevents(),
                                       eventDescriptors=events$type,
                                       plotEvents=seq(1, length(events$start)),
                                       legendPos="right", ...) {
  # Plot a single plot with all variograms for all events.
  #
  # Args:
  #   variogramsList: A list of variograms like that returned from
  #                   allEventVariograms().
  #   variable: The name of the variable to collect for.
  #   varName: The long variable name (~s instead of spaces).
  #   varUnit: The unit for the variable (log if log scale used).
  #   events: Events corresponding to each variograms list entry.
  #   plotEvents: Which events to plot (default: all).
  #   eventDescriptors: Descriptor for each event. If repeated will group
  #                     those events together using the same colour.
  #   legendPos: Position for the legend (default: right).
  #   ...: Extra arguments to ggplotVariograms().
  #
  # Returns: ggplot object.

  variogramsList = variogramsList[plotEvents]
  events = events[plotEvents,]

  stopifnot(length(events$start) == length(variogramsList))

  varInfo = collectVariogramsByVariable(variogramsList, variable)
  eventDescriptions = paste(strftime(events$start, "%m-%d %H:%M"),
                            " (", round(events$end - events$start, 0),
                            "h)", sep="")

  # Convert the first letter in the descriptors to uppercase.
  eventDescriptors = paste(toupper(substring(eventDescriptors, 1,1)),
                           substring(eventDescriptors, 2), sep="")

  plot = ggplotVariograms(varInfo$variograms,
                          variogramDescs=eventDescriptions,
                          lineTypeScaleName="Event type",
                          lineTypes=eventDescriptors,
                          varName=varName, varUnits=varUnit,
                          models=varInfo$models, legendPos=legendPos,
                          distUnits="m",
                          scaleTitle="Events",
                          ...)

  return(plot)
}

plotAllEventVariograms = function(data, outputPrefix,
                                  variables=c("R", "Nt", "D0", "Zh"),
                                  varUnits=c("mm~h^{-1}", "m^{-3}", "mm"),
                                  varNames=c("rain rate",
                                             "total drop concentration",
                                             "median drop diameter"),
                                  timeRes=numeric(0),
                                  events=SOPevents(),
                                  stations=stationsDefinition(),
                                  transform=identity,
                                  models="Sph",
                                  distUnits="m",
                                  plotTrans="identity",
                                  trimSeconds=0,
                                  ...) {
  # Calculate and plot to file all variograms for a set of variables
  # and a set of events.
  #
  # Args:
  #   data: The data to fit to.
  #   outputPrefix: Prefix for output files.
  #   events: List of events, each must contain start and end.
  #   variables: Variables to find variograms for.
  #   varUnits: Units for each variable.
  #   varNames: Long names for each variable.
  #   timeRes: Time resolution to resample to (default: no resampling).
  #   stations: Stations definition (default: HYMEX stations).
  #   transform: Transform data before finding variograms (function)?
  #              Can be one function per variable. (Default: no transform).
  #   models: Models to fit (if more than one, the one with
  #           the lowest error will be returned). (Default: Sph).
  #   distUnits: Distance units (default: "m").
  #   plotTrans: Translation for plot y axis (string, name of function)?
  #              (Default: none).
  #   trimSeconds: If > 0, remove this many seconds from start and end of
  #               each event before variogram calculation. (Default: 0).
  #   ...: Further arguments to fittedRealisationVariogram().
  #
  # Returns: void.

  # Get all the event variograms.
  allVarios = allEventVariograms(data=data, variables=variables,
                                 timeRes=timeRes, events=events,
                                 stations=stations, transform=transform,
                                 models=models, trimSeconds=trimSeconds)

  for(varioSet in allVarios) {
    # Plot each variable's variogram.
    for(v in seq(1, length(variables))) {
      model = varioSet[[v]]$model
      displayModel = (length(model) > 1)

      title = paste("Variogram for ", varNames[v],
                    "\n", strftime(start, "%Y-%m-%d %H:%M:%S"),
                    " to ", strftime(end,"%Y-%m-%d %H:%M:%S"), sep="")
      if(displayModel) {
        title = paste(title, "\n(Model type: ", varios[[v]]$modelType, ")", sep="")
      }

      plot = ggplotVariogram(varioSet[[v]]$variogram, varName=varNames[v],
                             varSymbol=variables[v], distUnits=distUnits,
                             varUnits=varUnits[v], model=model,
                             trans=plotTrans, displayModel=displayModel,
                             title=title)

      outFile = paste(outputPrefix, "/variogram_", variables[v], "_event_",
                      strftime(start, "%b_%d_%I_%p"),".png", sep="")

      ggsave(plot=plot, filename=outFile)
    }
  }
}

eventVariograms = function(data, start, end, variables,
                           timeRes=numeric(0),
                           stations=stationsDefinition(),
                           transform=rep("identity", length(variables)),
                           models=c("Sph", "Exp", "Gau", "Lin"),
                           trimSeconds=0,
                           ...) {
  # Calculate a variogram for an event. The variogram returned will
  # have a model fitted, depending on which model has the lowest error.
  #
  # Args:
  #   data: The data to fit on.
  #   start, end: Start and end times for the event (POSIXct, UTC).
  #   variable: Which variable to fit on?
  #   stations: Stations definition.
  #   timeRes: Time resolution to resample to before fitting
  #            (default: no resampling).
  #   transform: Transform function for data (default: no transform).
  #   models: Which models to try? (Default: Sph, Gau, Exp, Lin).
  #   trimSeconds: If > 0, remove this many seconds from start and end of
  #                each event before variogram calculation. (Default: 0).
  #   ...: Further arguments to fittedRealisationVariogram().
  #
  # Returns: List of variogram objects, one for each variable. Each will
  # contain variogram, model, the nugget, and modelType.

  varios = NULL
  for(i in seq(1, length(variables))) {
    variable = variables[i]
    transformFunction = get(transform[i])
    print(paste("Finding variagram for", variable, "with transform:",
          transform[i]))
    vario = eventVariogram(data, start, end, variable, timeRes,
                           stations, transformFunction,
                           models, trimSeconds, ...)
    varios = c(varios, list(variogram=vario))
  }

  names(varios) = variables
  return(varios)
}

eventAnisotropyTest = function(data, start, end, variable,
    varName, varUnits, distUnits="m", timeRes="2 min",
    stations=stationsDefinition(), transform=log,
    models=c("Sph", "Exp", "Gau", "Lin"),
    resampleFunc=function(x)
    {return(mean(x, na.rm=T))},
    distBoundaries=c(0.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 11)*1000,
    ...) {
  # Plot the anisotropy test for an event.
  #
  # Args:
  #   data: The data to fit on.
  #   start, end: Start and end times for the event (POSIXct, UTC).
  #   variable: Which variable to fit on?
  #   varName: Long name for variable.
  #   varUnits: Units for variable.
  #   distUnits: Distance units (default: "m").
  #   timeRes: Time resolution to resample to before fitting.
  #   stations: Stations definition.
  #   transform: Transform function for variable before variogram fitting
  #              (default: log).
  #   models: Models to try fitting.
  #   resampleFunc: The function to use to resample the data;
  #                 (default: mean(na.rm=T)).
  #   ...: Further arguments to plotAnisotropyTest().
  #
  # Returns: Plot from plotAnisotropyTest().

  # Resample subset to the requested time resolution.
  print("Warning: Resampling of data should be changed to use DSD resampling.")
  subset = data[which(data$POSIXtime >= start & data$POSIXtime <= end),]
  subset = resampleNetworkData(subset, timespan=timeRes,
                               dataColumnNames=variable,
                               func=resampleFunc)
  subset[[variable]] = transform(subset[[variable]])
  subset[[variable]][which(is.infinite(subset[[variable]]))] = NA

  subset$realisation = as.numeric(factor(subset$POSIXtime,
                                         labels=seq(1,
                                                    length(unique(
                                                      subset$POSIXtime)))))
  subset = addStationLocations(subset, stations)

  # Calculate and return anisotropy test plot.
  plot = plotAnisotropyTest(data=subset, variableName=variable,
                            variableLongName=varName,
                            realisationVarName="realisation",
                            distBoundaries=distBoundaries, model=models,
                            variableUnits=varUnits, distUnits=distUnits,
                            ...)
  return(plot)
}

eventVariogramComparisonPlot = function(event1varios, event2varios, setNames,
                                        variable, varLongName, varUnits,
                                        distUnits="m", textSize=20,
                                        trans="identity", subTitle="") {
  # Plot two events' variograms for the same variable on the same plot to
  # show a comparison.
  #
  # Args:
  #   event1varios: Variogram list for event 1 (use eventVariograms()).
  #   event2varios: Variogram list for event 2.
  #   setNames: Descriptive names for each set.
  #   variable: The variable for which to plot.
  #   varLongName: Long name of the variable.
  #   distUnits: Distance units (default: "m").
  #   textSize: Size for text in plot (default: 16).
  #   trans: Transform for y axis.
  #   subTitle: Optional plot subtitle.
  #
  # Returns: Plot object.

  vario1 = event1varios[[variable]]$variogram
  model1 = event1varios[[variable]]$model
  modelType1 = event1varios[[variable]]$modelType
  vario2 = event2varios[[variable]]$variogram
  model2 = event2varios[[variable]]$model
  modelType2 = event2varios[[variable]]$modelType

  if(modelType2 != modelType1) {
    print("Warning - model types disagree.")
    print(paste("Model 1 type:", modelType1))
    print(paste("Model 2 type:", modelType2))
  }

  title=""#paste("Variogram comparison for", varLongName)#,
  #"\n(Model type: ", modelType1, ")", sep="")
  if(nchar(subTitle) != 0) {
    title = paste(title, "\n", subTitle, sep="")
  }

  plot = ggplotVariograms(varios=list(vario1, vario2),
                          variogramDescs=setNames,
                          varName=variable, distUnits=distUnits,
                          varUnits=varUnits, models=list(model1, model2),
                          title=title, textSize=textSize,
                          trans=trans)
    return(plot)
}

eventVariogram = function(data, start, end, variable,
    timeRes=numeric(0),
    stations=stationsDefinition(),
    transform=identity,
    models=c("Sph", "Exp", "Gau", "Lin"),
    trimSeconds=0,
    removeZeros = TRUE,
    resampleFunc=function(x)
    {return(mean(x, na.rm=T))},
    boundaries=c(0.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 8, 11)*1000,
    ...) {
  # Calculate a variogram for an event. The variogram returned will
  # have a model fitted, depending on which model has the lowest error.
  #
  # Args:
  #   data: The data to fit on.
  #   start, end: Start and end times for the event (POSIXct, UTC).
  #   variable: Which variable to fit on?
  #   timeRes: Time resolution to resample to before fitting (Default: what
  #            the data is in).
  #   stations: Stations definition with locations.
  #   transform: Transform function for variable before variogram fitting
  #              (default: none).
  #   models: Models to try fitting.
  #   trimSeconds: If > 0, remove this many seconds from start and end of
  #               each event before variogram calculation. (Default: 0).
  #   resampleFunc: Function to use to resample the data
  #                 (default: mean(rm.na=T)).
  #   ...: Further arguments to fittedRealisationVariogram().
  #
  # Returns: Variogram object containing variogram, model, the nugget, and
  # modelType. To plot this use ggplotVariogram().

  # Resample subset to the requested time resolution.
  start = start + trimSeconds
  end = end - trimSeconds
  stopifnot(end > start)
  subset = data[which(data$POSIXtime >= start & data$POSIXtime <= end),]

  if(length(timeRes) != 0) {
    print("Note: If resampling Parsivel data, it is better to resample it")
    print("before calling the variogram function, using ")
    print("resampleParsivelByDSD().")
    stopifnot(is.character(timeRes))
    subset = resampleNetworkData(subset, timespan=timeRes,
                                 dataColumnNames=variable,
                                 func=resampleFunc)
  }

  # Remove zeros.
  if(removeZeros) {
    idx = which(subset[[variable]] == 0)
    print(paste("For variable ", variable, "- removing ", length(idx), "zeros."))
    subset[[variable]][idx] = NA
  }

  # Transform data.
  subset[[variable]] = transform(subset[[variable]])
  subset[[variable]][which(is.infinite(subset[[variable]]))] = NA

  subset$realisation = as.numeric(factor(subset$POSIXtime,
                                         labels=seq(1,
                                                    length(unique(
                                                      subset$POSIXtime)))))

  subset = addStationLocations(subset, stations)

  # Calculate and return variogram.
  vario = fittedRealisationVariogram(data=subset,
                                     variableName=variable,
                                     realisationVarName="realisation",
                                     boundaries=boundaries,
                                     model=models, ...)
  return(vario)
}

plotVariogram = function(data, variable, varModel, varUnits, varLongName,
    distBoundaries=c(0.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 11)*1000,
    subtitle="",
    stations=stationsDefinition(),
    useZeroDistNugget=F, ...) {
  # Find a variogram for instrument network data and plot it with a
  # fitted model.
  #
  # Args:
  #   data: The data to use (can be a subset).
  #   variable: The variable to find the variogram for.
  #   varModel: The name of the variogram model(s) to use.
  #   varUnits: Variable units (for plot).
  #   distBoundaires: Distance lag boundaires (by default, found
  #                   automatically from station distances).
  #   subtitle: Plot subtitle.
  #   stations: Station definition (default: stationsDefinition()).
  #   useZeroDistNugget: Calculate the nugget from zero-distance points?
  #                      (Default: F).
  #   ...: Further arguments to fittedRealisationVariogram().
  #
  # Returns: List containing a variogram list and plot.

  # Make a realisation column.
  data = addStationLocations(data)
  data$realisation =
    as.numeric(factor(data$timestamp,
                      labels=seq(1, length(unique(data$timestamp)))))

  if(length(distBoundaries) < 1) {
    # Use all possible distances between stations.
    coords = data.frame(stations$x_metres, stations$y_metres)
    distBoundaries = unique(dist(coords))

    # Ensure distance boundaries are strictly increasing.
    distBoundaries = distBoundaries[order(distBoundaries)]
  }

  var = fittedRealisationVariogram(data, variable, model=varModel,
      realisationVarName="realisation", boundaries=distBoundaries,
      useZeroDistNugget=useZeroDistNugget, ...)

  plotTitle = paste("Variogram of ",
                    varLongName, "\n",
                    subtitle, " [Model:",
                    var$modelType, "]", sep="")

  if(useZeroDistNugget) {
    plotTitle = paste(plotTitle, "\nZero-distance nugget:",
                      round(var$model[which(var$model$model == "Nug"),
                                      "psill"], 2))
  }

  plot = ggplotVariogram(var$variogram, distUnits="m", varUnits=varUnits,
                         varName=variable, model=var$model,
                         title=plotTitle)

  return(list(variogram=var, plot=plot))
}

plotEventVariogramComparisonByVariable = function(event1varios, event2varios,
                                                  eventDescriptions,
                                                  variables, varNames,
                                                  varUnits, textSize) {
  # Plot variogram comparisons for each of a set of variables.
  #
  # Args:
  #  event1varios: Variograms for event 1.
  #  event2varios: Variograms for event 2.
  #  eventDescriptions: A description per event.
  #  variables: Variables to plot for.
  #  varNames: Names of the variables (one for each).
  #  varUnits: Units of the variables (one for each).
  #  textSize: Size of text in plots.
  #
  # Returns: a named list of ggplot objects, one per variable.

  plots = list()

  for(i in seq(1, length(variables))) {

    plot = eventVariogramComparisonPlot(event1varios, event2varios,
                                        eventDescriptions, variables[i],
                                        varNames[i], varUnits[i],
                                        textSize=textSize)
    plots = c(plots, list(plot))
  }

  names(plots) = variables
  return(plots)
}

replaceZeroDistPoints = function(data, variables,
                                 realisationVarName="realisation",
                                 useCressie=FALSE,
                                 useMean=FALSE,
                                 keepStation=NULL) {
  # Find zero-distance points with the same realisation number; replace
  # them with their mean, or data from only one station, and determine the
  # real nugget for each variable. Assumes that the same set of
  # spatial points will be present in every realisation.
  #
  # Args:
  #   data: The data, a spatial object with a realisation variable.
  #   variables: The variables to find nuggets for and replace.
  #   realisationVarName: The variable name for the realisation
  #                       (default: "realisation").
  #   useCressie: If TRUE, use the Cressie robust semivariance instead of
  #               the normal one (default: FALSE).
  #   useMean: If TRUE, replace the collocated points with the mean values.
  #            If FALSE, keep only the first station's data (default: FALSE).
  #   keepStation: If specified, ensure that this station is the one kept out 
  #                of the collocated stations (just a check!).
  #
  # Returns: The same data with zero distance points averaged per realisation,
  #          or replaced by the first station's data, and the nugget for each
  #          variable, as a list containing "data" and "nuggets".
  #
  # NOTE: If there are more than two observations at the same place for the
  # same realisation, only the maximum difference will be used as the
  # difference at zero distance.

  # Turn the data into a data.table, keyed by location and realisation.
  key = c(names(coordinates(data)[1,]), realisationVarName)
  df = data.frame(data)
  dt = data.table(df, key=key)

  # Split the data frame into lines from collocated stations, and lines
  # from the rest (which we don't touch).
  collocLocations = dt[which(duplicated(dt, by=key)),]
  if(dim(collocLocations)[1] == 0)
      return(list(data=data, nuggets=NULL))
  collocated = dt[collocLocations, names(dt), with=FALSE]

  # Find the max difference between collocated measurements, for each
  # collocated point and realisation.
  if(useCressie) {
    # If Cressie is to be used, we sum the square root of the differences.
    diffFun = function(x) { return(sqrt(abs(max(diff(x, na.rm=T))))) }
  } else {
    # If Cressie is not to be used, we sum the squared differences.
    diffFun = function(x) { return(max(diff(x, na.rm=T))^2) }
  }
  diffs = collocated[, lapply(.SD, diffFun), by=key, .SDcols=variables]

  # Sum the squared differences (to get s).
  sums = diffs[, lapply(.SD, sum, na.rm=T), .SDcols=variables]

  # Find the number that went into each sum (to get n).
  numFunc = function(x) { return(length(which(!is.na(x)))) }
  nums = diffs[, lapply(.SD, numFunc), .SDcols=variables]

  if(useCressie) {
    # For Cressie, the nugget is (1/2 * (1/n * s)^4) / (0.457 + (0.494/nums))
    nuggets = data.frame((0.5 * ((1/nums) * sums)^4) / (0.457 + (0.494/nums)))
  } else {
    # The nugget for each variable is the (1/2*n) * s.
    nuggets = data.frame((1 / (2*nums)) * sums)
  }
  names(nuggets) = variables

  # The final result is the non-collocated stations added to the
  # collocated means (or just unique stations kept).
  result = unique(dt, by=key)

  if(useMean) {
    # Replace the values for the collocated locations with the mean values.
    means = collocated[, lapply(.SD, mean, na.rm=T), .SDcols=variables, by=key]
    result[means, variables := means[, variables, with=FALSE], with=FALSE]
  }
 
  # Remove rows that contain NA.
  idx = which(is.na(rowMeans(result[, variables, with=FALSE])))

  # Stop if any lines are now duplicated (by location/realistion combo).
  stopifnot(!any(duplicated(result, by=key)))

  if(!is.null(keepStation)) 
      if(keepStation %in% data$station) 
          stopifnot(keepStation %in% result$station)
  
  # Convert the result back to a data.frame and re-add location info.
  result = data.frame(result)

  if(length(idx) > 0) {
    print("WARNING: removing rows that contain NAs, in replaceZeroDistPoints().")
    result = result[-idx,]
  }
  coordinates(result) = as.formula(paste("~", names(data.frame(coordinates(data)))[1],
                                         "+", names(data.frame(coordinates(data)))[2],
                                         sep=""))
  proj4string(result) = proj4string(data)
  return(list(data=result, nuggets=nuggets))
}
