# Functions involving uncertainty analysis of collocated instruments.
#
# Working from paper:
# 
# Jaffrain, J. and A. Berne, 2011: Experimental quantification of the sampling
# uncertainty associated with measurements from Parsivel disdrometers.
# J. Hydrometeor., 12, doi:10.1175/2010JHM1244.1.
# 
# Author: Tim Raupach <tim.raupach@epfl.ch>

relativeUncertaintyByTimeResolution = function(parsivelDSDs,
                                               timeRes = c("30 s", "1 min", 
                                                           "2 min", "5 min", 
                                                           "15 min", "30 min", 
                                                           "1 hour"),
                                               baseRes = "30 s",
                                               ...) {
  # Find the parsivel relative uncertainty by time resolution.
  # 
  # Args:
  #  parsivelDSDs: Data to work on.
  #  timeRes: Resolutions to find the uncertainty at.
  #  baseRes: The resolution that the data are already in.
  #  ...: Further arguments to relativeUncertainty().
  #
  # Returns: relative uncertainty by variable and by time resolution.
  
  res = NULL
  
  # A function to apply in parallel.
  for(t in timeRes) {
    print(t)
    resample = TRUE
    if(t == baseRes) {
      resample = FALSE
    }
    u = relativeUncertainty(parsivelDSDs, timeRes=t, resample=resample, ...)
    res = rbind(res, data.frame(timeRes=t, u))
  }
  
  res$resSeconds = convertTimeStringsToUnit(res$timeRes)
  return(res)
}

plotRelativeUncertainty = function(relativeParsivelErrors, textSize=20, ...) {
  # Plot the relative uncertainties calculated from collocated Parsivels, by
  # time resolution and variable.
  # 
  # Args:
  # relativeParsivelErrors: output from relativeUncertaintyByTimeResolution().
  #  parsivelDSDs: Set of parsivel DSDs to use.
  #  textSize: The text size to use (default: 20).
  #  ...: Further arguments to relativeUncertainyByTimeResolution().
  #  
  # Returns: ggplot2 object.
  
  plot = ggplot(relativeParsivelErrors, 
                aes(x=resSeconds, y=relUncertainty, group=variable)) + 
    geom_line(aes(colour=variable), size=1) + 
    scale_x_continuous(trans="log", 
                       breaks=relativeParsivelErrors$resSeconds) +
    theme_bw(textSize) +
    labs(x="Temporal resolution [s]", y="Relative uncertainty",
         title="Relative Parsivel uncertainty by time resolution") +
    scale_colour_discrete(name="Variable")
  
  return(plot)
}
