# network_functions.R
#
# Functions to analyse information from a network of DSD measuring 
# instruments.
#
# Author: Tim Raupach <tim.raupach@epfl.ch>

stationsDefinition = function() {
  # The default stations definition.
  # Must return a data.frame containing station number, name, 
  # label (for plotting), latitude (lat), and longitude (lon).
  # 
  # Files in which network_functions.R is sourced should override 
  # the stationsDefinition() function.
  
  stop("Network stations are undefined!")
}

meanRainRatePerTimestep = function(data) {
  # Return the mean rain rate per timestep across all stations.
  #
  # Args:
  #   data: data.frame containing at least POSIXtime, station, and R.
  #
  # Returns: a data.frame containing timestamp and average rain rate.
  
  subset = data.frame(POSIXtime=data$POSIXtime,
                      station=data$station,
                      R=data$R)
  meltedSubset = melt(subset, id=c("POSIXtime", "station"))
  castSubset = cast(meltedSubset, POSIXtime ~ station)
  numstations = length(unique(subset$station))
  meanRates = rowMeans(castSubset[, 2:(numstations+1)], na.rm=T)
  return(data.frame(POSIXtime=castSubset$POSIXtime, 
                    R_mean=meanRates))
}

calcSpreadStat = function(spectra, statName) {
  # Calculate mean, standard deviation, and coefficient of variation across 
  # stations for each timestep in a DSD spectra.
  # 
  # Args:
  #   spectra: dsd spectra data.frame containing at least columns POSIXtime, 
  #            and stat of choice.
  #   statName: name of the column to calculate stats for.
  #
  # Returns: data.frame containing timesetamp, mean, sd, andcv.
    
  subset = data.frame(timestamp=strftime(spectra$POSIXtime, 
                                         format="%Y-%m-%d %H:%M:%S", tz="UTC"),
                      station=spectra$station, 
                      val=spectra[[statName]])  
  melted = melt(subset)
  castVals = cast(melted, timestamp ~ station)
  mat = as.matrix(castVals)
  means = rowMeans(mat, na.rm=T)
  sds = apply(mat, 1, function(x) { return(sd(x, na.rm=T)) })
  cvs = sds / means
  
  res = data.frame(POSIXtime=as.POSIXct(castVals$timestamp, tz='UTC'),
                   mean=means,
                   sd=sds,
                   cv=cvs)
  
  return(res)
}

calculateSpreadAcrossStations = function(spectra) {
  # Calculate statistics on how much spread there is in values between all
  # stations in the network.
  # 
  # Args:
  #   spectra: dsd spectra data.frame containing at least columns POSIXtime, 
  #            amount, R, Nt, D0.
  # 
  # Return: a data.frame containing columns per timestep.
  #
  # NOTE: these statistics are only calculated for station/timestep combinations
  # in which the rain amount is > 0.
  
  # Calculate standard deviation, mean, CV for each statistic over 
  # each station.
  
  # Only keep timesteps for which there is an amount recorded.
  spectra = spectra[which(spectra$amount > 0),]
  
  R_stat = calcSpreadStat(spectra, "R")           # Rain rates.
  amount_stat = calcSpreadStat(spectra, "amount") # Rain amount.
  Nt_stat = calcSpreadStat(spectra, "Nt")         # Drop concentration.
  D0_stat = calcSpreadStat(spectra, "D0")         # Median-vol diameter.
  
  # Check all timestamps are the same.
  stopifnot(all(R_stat$POSIXtime == amount_stat$POSIXtime) &
              all(R_stat$POSIXtime == Nt_stat$POSIXtime) & 
              all(R_stat$POSIXtime == D0_stat$POSIXtime))
  
  # Put together to return.
  res = data.frame(POSIXtime=R_stat$POSIXtime,
                   R_mean=R_stat$mean,
                   R_sd=R_stat$sd,
                   R_cv=R_stat$cv,
                   amount_mean=amount_stat$mean,
                   amount_sd=amount_stat$sd,
                   amount_cv=amount_stat$cv,
                   Nt_mean=Nt_stat$mean,
                   Nt_sd=Nt_stat$sd,
                   Nt_cv=Nt_stat$cv,
                   D0_mean=D0_stat$mean,
                   D0_sd=D0_stat$sd,
                   D0_cv=D0_stat$cv)
  
  return(res)
}

compareTwoStations = function(stats, stationOneName, stationTwoName,
                              period="30 second", scale="normal",
                              ...) {
  # Compare two stations to each other. 
  #
  # Args:
  #  stats: data.frame containing timestep, station, amount, R, Nt, D0.
  #  stationOneName: name of station one.
  #  stationTwoName: name of station two.
  #  scale: scale to use. Default is "continuous". If "log" then all zero 
  #         values are replaced with NAs and log scale is used. If "sqrt" 
  #         then a square root scale is used.
  #  ...: Plotting parameters to pass on to comparisonStatsAndPlot().
  # 
  # Returns: a list containing comparison results for rain amount, rainrate R,
  #          total drop concentration Nt and median-volume diameter D0. Each
  #          of these will be a list containing rmse, bias, correlation, and a
  #          comparison scatterplot.
  
  # Amount
  amount_results = comparisonStatsAndPlot(stats,   
                                          statName="amount",
                                          stationOneName=stationOneName,
                                          stationTwoName=stationTwoName, 
                                          statLongName=paste(period, "rain amount"),
                                          axisOneLabel=paste(stationOneName, "rain amount [mm]"),
                                          axisTwoLabel=paste(stationOneName, "rain amount [mm]"),
                                          scale=scale, ...)
  
  # Rainrate
  R_results = comparisonStatsAndPlot(stats,   
                                     statName="R",
                                     stationOneName=stationOneName,
                                     stationTwoName=stationTwoName, 
                                     statLongName=paste(period, "rain rate"),
                                     axisOneLabel=bquote(.(stationOneName) ~ 
                                                           "rainrate [mm" ~ h^{-1} * "]"),
                                     axisTwoLabel=bquote(.(stationTwoName) ~ 
                                                           "rainrate [mm" ~ h^{-1} * "]"),
                                     scale=scale, ...)
  
  # Total drop concentration
  Nt_results = comparisonStatsAndPlot(stats,   
                                      statName="Nt",
                                      stationOneName=stationOneName,
                                      stationTwoName=stationTwoName, 
                                      statLongName=paste(period, 
                                                         "drop concentration"),
                                      axisOneLabel=bquote(.(stationOneName) ~ 
                                          "drop concentration" ~ N[t] ~ "[" * 
                                          m^{-3} * "]"),
                                      axisTwoLabel=bquote(.(stationTwoName) ~ 
                                          "drop concentration" ~ N[t] ~ "[" * 
                                          m^{-3} * "]"),
                                      scale=scale, ...)
  
  # median-volume diameter D0
  D0_results = comparisonStatsAndPlot(stats,   
                                      statName="D0",
                                      stationOneName=stationOneName,
                                      stationTwoName=stationTwoName, 
                                      statLongName=paste(period,
                                                    "median-volume diameter"),
                                      axisOneLabel=bquote(.(stationOneName) ~ 
                                            "median-volume diameter D0 [mm]"),
                                      axisTwoLabel=bquote(.(stationTwoName) ~ 
                                            "median-volume diameter D0 [mm]"),
                                      scale=scale, ...)
  
  # Put everything together in a list to return.
  res = list(amount=amount_results, R=R_results, Nt=Nt_results, D0=D0_results)
  return(res)
}

comparisonStatsAndPlot = function(data, statName,
                                  stationOneName, 
                                  stationTwoName,
                                  statLongName, 
                                  axisOneLabel,
                                  axisTwoLabel,
                                  scale="normal",
                                  title=T,
                                  removeZeroPairs=T,
                                  ...) {
  # Calculate comparison statistics and a plot for comparison of two 
  # stations.
  #
  # Args:
  #   data: data.frame with columns station, and statName.
  #   statName: the name of the column to compare.
  #   stationOneName: the name of the first station to compare (this is 
  #                   taken to be the reference station!).
  #   stationTwoName: the name of the second station to compare (this is taken 
  #                   to be the observation station being tested).
  #   statLongName: the long name of the statistic (for plot title).
  #   axisOneLabel: the label of the axis for stationOne in the plot.
  #   axisTwoLabel: the label of the axis for stationTwo in the plot.
  #   scale: scale to use. Default is "continuous". If "log" then all 
  #          zero values are replaced with NAs and log scale is used.
  #          If "sqrt" then a square root scale is used.
  #   title: Include the title?
  #   ...: Extra arguments to stationComparisonPlot.
  #
  # Returns: list with rmse, bias, correlation, relative bias (%) and
  #          a ggplot2 scatterplot.
  
  
  # Subset for stations of interest.
  setOne = data[which(data$station == stationOneName),]
  setTwo = data[which(data$station == stationTwoName),]
    
  # Check timestamps align.
  stopifnot(identical(setOne$POSIXtime, setTwo$POSIXtime))
  
  # Check only one station.
  stopifnot(length(unique(setOne$station)) == 1)
  stopifnot(length(unique(setTwo$station)) == 1)
  
  # Remove pairs of zeros?
  if(removeZeroPairs) {
    idx = which((setOne[[statName]] == 0 | is.na(setOne[[statName]])) & 
                  (setTwo[[statName]] == 0 | is.na(setTwo[[statName]])))
    if(length(idx) > 0) {
      setOne = setOne[-idx,]
      setTwo = setTwo[-idx,]
    }
  }
  
  # Remove from consideration any timesteps which were missed by either sensor.
  nas = which(is.na(setOne[[statName]]) | is.na(setTwo[[statName]]))
  if(length(nas) > 0) {
    setOne = setOne[-nas,]
    setTwo = setTwo[-nas,]
  }
  
  # Make sure there are some measurements to compare.
  stopifnot(length(setOne[,1]) > 0)
  stopifnot(length(setTwo[,1]) > 0)
  
  diff = setTwo[[statName]] - setOne[[statName]]
  stopifnot(!any(is.na(diff)))
  
  # Absolute stats.
  rmse = sqrt(mean(diff^2))
  bias = mean(diff)
  
  # Relative stats.
  rel_bias_perc = median(diff / setOne[[statName]] * 100)
  # rel_diff = diff / mean(setOne[[statName]], na.rm=T) * 100
  # rel_diff = diff / setOne[[statName]] * 100
  # rel_bias_perc = 100 * mean(diff, na.rm=T) / mean(setOne[[statName]], na.rm=T)

  # Mean ratio.
  mean_ratio = mean(setTwo[[statName]]) / 
    mean(setOne[[statName]])

  # Set one is reference!!
  # Slope of regression line.
  vars = data.frame(one=setOne[[statName]], two=setTwo[[statName]])
  linFit = lm(two ~ one, vars)
  regression_slope = linFit$coefficients[[2]]

  # Only plot data on which the statistics are based.
  data = data.table(data)
  plotData =
      rbind(data[POSIXtime %in% setOne$POSIXtime & station == unique(setOne$station)],
            data[POSIXtime %in% setTwo$POSIXtime & station == unique(setTwo$station)])
  
  corr = cor(setOne[[statName]], setTwo[[statName]],
             use="complete.obs")
  plot = stationComparisonPlot(plotData, stationOneName, stationTwoName,
                               statistic=statName, statName=statLongName,
                               corr=corr, rmse=rmse, bias=bias,
                               rel_bias_perc=rel_bias_perc,
                               mean_ratio=mean_ratio,
                               regression_slope=regression_slope,
                               axisOneLabel=axisOneLabel,
                               axisTwoLabel=axisTwoLabel,
                               scale=scale, title=title, ...)
  
  return(list(rmse=rmse, bias=bias, correlation=corr, 
              relBias=rel_bias_perc, mean_ratio=mean_ratio,
              regression_slope=regression_slope, plot=plot))
}

stationComparisonPlot = function(stats, stationOneName, stationTwoName,
                                 statistic, statName, correlation=NA,
                                 bias=NA, rmse=NA, rel_bias_perc=NA,
                                 mean_ratio=NA, regression_slope=NA,
                                 axisOneLabel=paste(stationOneName, statName),
                                 axisTwoLabel=paste(stationTwoName, statName),
                                 scale="normal",
                                 lineScale=1,
                                 textInset=0.1, 
                                 dp=2, title=TRUE,
                                 statDP=3, textSize=14,
                                 statTextSize=3,
                                 square=TRUE) {
  # Plot a comparison of two stations, using a scatter plot, line of best fit,
  # and the 1:1 line.
  # 
  # Args:
  #   stats: statistics to plot.
  #   stationOneName: first station to compare.
  #   stationTwoName: second station to compare.
  #   statistic: statistic to compare.
  #   statName: name of statistic to compare, for plot title.
  #   correlation: correlation between the stats.
  #   bias: bias to display.
  #   rmse: RMSE to display.
  #   rel_bias_perc: Relative bias percentage (of set one) to display.
  #   mean_ratio: mean(setTwo) / mean(setOne).
  #   regression_slope: Slope of regression line.
  #   axisOneLabel: name of statistic for axis label one
  #                  (default: paste(stationOneName, statName)).
  #   axisTwoLabel: name of statistic for axis label one
  #                  (default: paste(stationOneName, statName)).
  #   scale: scale to use. Default is "continuous". If "log" then all 
  #          zero values are replaced with NAs and log scale is used.
  #          If "sqrt" then a square root scale is used.
  #   lineScale: Used to position text in the plot. In units of y axis. 
  #              (default: 1).
  #   textInset: Used to inset text from left within plot (default: 0.2).
  #   dp: Decimal places for statistics (default: 2).
  #   title: Include the title?
  #   statDP: Round numbers to what decimal place before calculating stats?
  #   textSize: Font size for the plot.
  #   statTextSize: ggplot text size for annotations (default: 3).
  #   square: Make the plot square, ie both axes the same length? (default: 
  #           TRUE).
  # 
  # Returns: ggplot2 plot ready to display.
  
  statsOne = stats[which(stats$station == stationOneName),][[statistic]]
  statsTwo = stats[which(stats$station == stationTwoName),][[statistic]]
  
  statsOne = round(statsOne, statDP)
  statsTwo = round(statsTwo, statDP)
  
  corrText = NA
  biasText = NA
  rmseText = NA
  relBiasText = NA
  meanRatioText = NA
  regressionSlopeText = NA
  
  if(!is.na(correlation)) {
    corrText = paste("r^2 == ", round(correlation^2, dp))
  }
  if(!is.na(rmse)) {
    rmseText = paste("RMSE == ", round(rmse, dp))
  }
  if(!is.na(bias)) {
    biasText = paste("Bias == ", round(bias, dp))
  }
  if(!is.na(rel_bias_perc)) {
    relBiasText = paste("Rel.~bias == ", round(rel_bias_perc, 0), 
                        "symbol('\045')", sep="~")
  }
  if(!is.na(mean_ratio)) {
    meanRatioText = paste("Mean~ratio == ", round(mean_ratio, dp))
  }
  if(!is.na(regression_slope)) {
    regressionSlopeText = paste("Reg.~slope == ", round(regression_slope, dp))
  }
  
  # Select scale to use.
  scale_x = scale_x_continuous()
  scale_y = scale_y_continuous()
  scaleExtra = ""
  if(scale == "log") {
    toRemoveOne = which(statsOne <= 0)
    toRemoveTwo = which(statsTwo <= 0)
    if(length(toRemoveOne) != length(toRemoveTwo) |
         !all(toRemoveOne == toRemoveTwo)) {
      print(paste("WARNING: Removal of values <= 0 for log",
                  "scale will mean lost information!!"))
    }
    
    # Remove zeros because log transformation can not handle them.
    statsOne[which(statsOne <= 0)] = NA
    statsTwo[which(statsTwo <= 0)] = NA
    scale_x = scale_x_log10() 
    scale_y = scale_y_log10()
    scaleExtra = " [log scale]"
  } else if(scale == "sqrt") {
    scale_x = scale_x_sqrt()
    scale_y = scale_y_sqrt()
    scaleExtra = " [sqrt scale]"
  }
  
  toPlot = data.frame(one=statsOne, two=statsTwo)
  
  # Where to put the text values in the plot.
  xpos = min(statsOne + textInset, na.rm=T)
  ypos = max(statsTwo - 5*lineScale, na.rm=T)
  
  start = strftime(min(stats$POSIXtime), "%Y-%m-%d", tz="UTC")
  end = strftime(max(stats$POSIXtime), "%Y-%m-%d", tz="UTC")
  
  titleText=paste("Scatterplot of ", statName, scaleExtra,
                  '\n', stationOneName, " vs. ", stationTwoName,
                  "\nHYMEX SOP, ", start, " to ", end, sep="")
  if(!title) {
    titleText=""
  }
  
  plot = ggplot(toPlot, aes(x=one, y=two)) +
    geom_point() + 
    geom_smooth(method=lm) +# , na.action=na.exclude) + 
    scale_x + scale_y +
    geom_abline(intercept=0, slope=1, size=0.5, lty=2, colour="red") +
    labs(title=titleText, x=axisOneLabel, y=axisTwoLabel) +
    annotate("text", label=c(relBiasText, biasText, corrText, rmseText, 
                             meanRatioText, regressionSlopeText), 
             x=xpos, y=seq(ypos, by=lineScale, length.out=6), 
             size=statTextSize, parse=T) +
    theme_bw(textSize) +
    theme(panel.grid.major=element_blank(),
          panel.grid.minor=element_blank(),
          legend.position=c(0.1, 0.8),
          legend.key=element_rect(colour="white"))
  
  return(plot)
}

aggregateSpectraToDailyByStation = function(spectra) {
  # Take normal spectra and aggregate to daily statistics.
  # 
  #  spectra = DSD spectra with station, timestamp, mean drop numbers, 
  #            summed amount, mean rate, mean drop concentration,
  #            and mean median-volume diameter.
  # 
  # Returns: data.frame containing mean columns by date and station.
  
  spectra$timestamp = as.Date(spectra$POSIXtime)  
  narmMean = function(x) {return(mean(x, na.rm=T))}
  res = ddply(spectra, .(station, timestamp), colwise(narmMean))
  
  # Amount should be summed, not averaged.
  if("amount" %in% names(res)) {
    summedAmounts = ddply(spectra, .(station, timestamp), 
                          summarise, amount=sum(amount))    
    stopifnot(all(summedAmounts$timestamp == res$timestamp))
    stopifnot(all(summedAmounts$station == res$station))
    res$amount = summedAmounts$amount
  }
  
  return(res)
}

addStationLocations = function(data, stations=stationsDefinition(),
                               crs=CRS(paste("+proj=utm +zone=31 +ellps=WGS84",
                                             "+datum=WGS84 +units=m +no_defs"))) {
  # Convert a dataset into an SP object that includes locations for 
  # each station.
  # 
  # Args:
  #  data: Data.frame containing a station field.
  #  stations: Stations definition to use, including name, lat, lon.
  #  crs: The CRS to translate the coordinates to.
  #
  # Returns: The data.frame as an SP object.
  
  stopifnot("station" %in% names(data))
  
  # For variogram, assign locations to data.
  locations = data.frame(station = stations$name,
                         lat=stations$lat, 
                         lon=stations$lon)
  subset = merge(data, locations, by="station")
  
  coordinates(subset) = ~lon+lat
  proj4string(subset) = CRS(paste("+proj=longlat +datum=WGS84",
                                  "+ellps=WGS84 +towgs84=0,0,0"))
  
  subset = spTransform(subset, crs)
  
  # Check dimensions to make sure we haven't lost any rows (due to
  # missing station information, for example).
  stopifnot(identical(dim(subset), dim(data)))
  
  return(subset)
}

plotSummaryStatsComparison = function(stats, names, 
                                      variable, yaxisLabel=NULL, 
                                      title=NULL, textSize=18,
                                      plotTrans=identity) {
  # Plot a comparison of summary stats for a variable.
  # 
  # Args:
  #  stats: List of summary stats objects from summaryStatsByStation.
  #  names: Names for each stats object.
  #  variable: Which variable to plot for?
  #  yaxisLabel: Label for y axis.
  #  title: The title for the plot.
  #  textSize: Text size for output? (Default: 14).
  #  plotTrans: Translation function for data (eg log(), default identity()).
  # 
  # Returns: the plot.
  
  s = NULL
  stopifnot(length(names) == length(stats))
  for(i in seq(1, length(names))) {
    s = rbind(s, data.frame(name=names[i], stats[[i]][[variable]]))
  }
  
  s$q25 = plotTrans(s$q25)
  s$q50 = plotTrans(s$q50)
  s$q75 = plotTrans(s$q75)
  s$q05 = plotTrans(s$q05)
  s$q95 = plotTrans(s$q95)
  
  plot = ggplot(s, aes(x=name)) + 
    geom_boxplot(stat="identity", 
                 aes(lower=q25,
                     middle=q50, 
                     upper=q75, 
                     ymax=q95, 
                     ymin=q05, 
                     fill=station)) + 
    scale_fill_discrete(name="Station") +
    theme_bw(textSize) +
    labs(title=title, x=NULL, y=parse(text=yaxisLabel))
  
  return(plot)
}

summaryStatsByStation = function(data, variable, removeZero=T) {
  # Produce summary statistics by station.
  #
  # Args:
  #   data: data.frame containing amount, R, Nt, and D0.
  #   variable: Variable to find summary statistics for.
  #   removeZero: Remove zeros from the variable? (Default: T).
  #
  # Returns: A data.frame containing min, max, mean, sd, sum and 
  # specified quantiles for the specified variable.
  
  if(removeZero) {
    data[[variable]][which(data[[variable]] == 0)] = NA
  }
      
  data$var = data[[variable]]
  
  summaryStats = ddply(data, .(station), summarise,
                       min=min(var, na.rm=T),
                       max=max(var, na.rm=T),
                       mean=mean(var, na.rm=T),
                       sd=sd(var, na.rm=T),
                       sum=sum(var, na.rm=T),
                       q05=quantile(var, na.rm=T, probs=0.05),
                       q25=quantile(var, na.rm=T, probs=0.25),
                       q50=quantile(var, na.rm=T, probs=0.5),
                       q75=quantile(var, na.rm=T, probs=0.75),
                       q95=quantile(var, na.rm=T, probs=0.95))
  
  return(summaryStats)
}

resampleNetworkData = function(data, timespan="5 min",
                               dataColumnNames=c("R", "Nt", "Dm", "LWC", "D0"),
                               func=function(x) {return(mean(x, na.rm=T))}) {
  # Resample network data by station into a different temporal resolution.
  # 
  # Args:
  #   data: A data.frame containing network data to resample, by station.
  #   timespan: The timespan for each timestep in the new timeseries (eg 
  #             "5 min"). See ?cut.POSIXt for possible formats.
  #   dataColumnNames: The columns to resample (default: "R", "Nt", "Dm", 
  #                    "LWC", "D0").
  #   func: The function to use to aggregate the data (default: mean, na.rm=T).
  #
  # Returns: a data.frame with station, timestep, and data columns.
  
  resample = function(x) {
    res = resampleTimeseries(data=x, timespan=timespan, 
                             dataColumnNames=dataColumnNames, 
                             timestampColumnName="POSIXtime",
                             fun=here(func))
    return(res)
  }
  
  resampled = NULL
  
  for(station in unique(data$station)) {
    print(paste("Resampling station:", station))
    stationData = data[which(data$station == station),]
    
    # Remove station name
    stationData$station = NULL
    stationRes = resample(stationData)
    resampled = rbind(resampled, data.frame(station=station, stationRes))
  }
  
  print("Done resampling.")
  
  return(resampled)
}

stationDistanceTable = function(stations=stationsDefinition()) {
  # Produce a table of distances between network stations.
  #
  # Args: 
  #  stations: Station definitions to use.
  #
  # Returns: data.frame of station distances.
  
  d = dist(stations, diag=T)
  d = as.data.frame(as.matrix(d))
  colnames(d) = stations$number  
  rownames(d) = stations$number  
  return(d)
}

resampleNetworkDSDs = function(data, timespan="5 min", 
                               dsdCols=paste("class", seq(1,32), sep="")) {
  # Resample DSDs (raw or 32 class) by station.
  #
  # Args:
  #  data: Network DSD data to resample.
  #  timespan: Time resolution to resample to (default: "5 min").
  #  dsdCols: Which columns are the DSD values? (Default: seq(6, 37)).
  # 
  # Returns: resampled data. 
  
  stopifnot(dsdCols == paste("class", seq(1, 32), sep=""))
  
  # Columns are drop volumetric drop concentrations and get AVERAGED.
  DSDs = resampleNetworkData(data, timespan=timespan, 
                             dataColumnNames=dsdCols,
                             func=function(x) {
                               return(mean(x, na.rm=TRUE))
                             })
  
  res = data.frame(POSIXtime=DSDs$POSIXtime,
                   station=DSDs$station,
                   DSDs[,3:length(DSDs[1,])])
  return(res)
}

prepareDataForGeostats = function(data, 
    nuggetVariables=numeric(0), 
    stations=stationsDefinition(),
    useCressie=FALSE, ...) {
  # Prepare a dataset with station information for geostatistics, 
  # by adding station locations, adding unique realisation per timestep,
  # and finding nuggets from collocated stations.
  #
  # Args:
  #   data: The data.frame to prepare.
  #   nuggetVariables: The variables for which to find nuggets. If not 
  #                    specified, no nuggets will be found and no collocated 
  #                    data will be dealt with. If specified, collocated points
  #                    are averaged per realisation.
  #   stations: The stations to use.
  #   useCressie: Use Cressie's estimator for nuggets (default: FALSE).
  #   
  # Returns: A list with data (the data, prepared), nuggets (a nugget for each
  #          variable at the collocated points, )
  
  # Add station locations to data.
  data = addStationLocations(data, stations)
  nuggets = NULL
  
  # Add a realisation column which is unique per timestep.
  data$realisation = 
    as.numeric(factor(data$POSIXtime,
                      labels=seq(1, length(unique(data$POSIXtime)))))
  
  # Replace collocated station values with mean values per timestep
  # and
  if(length(nuggetVariables) != 0) {
    res = replaceZeroDistPoints(data=data, variables=nuggetVariables, 
                                realisationVarName="realisation",
                                useCressie=useCressie, ...)
    nuggets = res$nuggets
    data = res$data
  }
  
  return(list(data=data, nuggets=nuggets))
}


