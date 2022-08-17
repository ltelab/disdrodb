# timedate_utils.R
#
# Timedate utilities.
#
# Author: <tim.raupach@epfl.ch>

subsetToEvents = function(data, events, inverse=FALSE) {
  # Return only data occurring inside event times; or if inverse=TRUE return
  # data occurring outside event times.
  
  data = data.table(data)
  
  keep = NULL
  for(e in seq(1, length(events$start))) {
    start = events$start[e]
    end = events$end[e]
    keep = rbind(keep, data[POSIXtime >= start & POSIXtime <= end,])
  }
  
  if(inverse)
    keep = data[!(data$POSIXtime %in% keep$POSIXtime),]
  
  # Make sure times remain in UTC (rbind destroys tz info).
  keep[, POSIXtime := as.POSIXct(strftime(POSIXtime, tz="UTC"), tz="UTC")]  
  
  # Remove duplicate rows.
  return(unique(keep))
}
