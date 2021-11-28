# IFloodS_functions.R
#
# Functions for data in the IFloodS campaign.
#
# Author: Tim Raupach <tim.raupach@epfl.ch>

require(gmt)
require(ggplot2)
require(stringr)

NASADiamCentres = function() {
    ## Modified centre diameters for Parsivel classes used by processed NASA data.
    
    return(c(0.064,
             0.193,
             0.321,
             0.450,
             0.579,
             0.708,
             0.836,
             0.965,
             1.094,
             1.223,
             1.416,
             1.674,
             1.931,
             2.189,
             2.446,
             2.832,
             3.347,
             3.862,
             4.378,
             4.892,
             5.665,
             6.695,
             7.725,
             8.755,
             9.785,
             11.330,
             13.390,
             15.450,
             17.510,
             19.570,
             22.145,
             25.235))
}

NASADiamWidths = function() {
    return(c(0.129, 0.129, 0.129, 0.129, 0.129, 0.129, 0.129, 0.129,
             0.129, 0.129, 0.257, 0.257, 0.257, 0.257, 0.257, 0.515,
             0.515, 0.515, 0.515, 0.515, 1.030, 1.030, 1.030, 1.030,
             1.030, 2.060, 2.060, 2.060, 2.060, 2.060, 3.090, 3.090))
}

# Stations definition
IFloodSStations = function() {
  # Define iFloods stations.
  # 
  # Returns: 
  #  A data.frame containing station number, name, label (for plotting),
  #  latitude (lat), longitude (lon), altitude [m], x_metres, y_metres, 
  #  projString, 
  
  # Define stations. Altitudes are found using Google elevation API.
  s = rbindlist(list(
    data.frame(name="apu01", lat="42:14:19.54", lon="92:27:49.33", altitude=284, stringsAsFactors=FALSE),
    data.frame(name="apu02", lat="42:10:56.33", lon="92:21:55.55", altitude=293, stringsAsFactors=FALSE),
    data.frame(name="apu03", lat="42:07:33.52", lon="92:16:54.17", altitude=283, stringsAsFactors=FALSE),
    data.frame(name="apu04", lat="42:07:20.75", lon="92:16:50.32", altitude=280, stringsAsFactors=FALSE),
    data.frame(name="apu05", lat="41:59:33.62", lon="92:03:36.74", altitude=286, stringsAsFactors=FALSE),
    data.frame(name="apu06", lat="41:58:41.36", lon="92:04:32.68", altitude=274, stringsAsFactors=FALSE),
    data.frame(name="apu07", lat="41:59:33.39", lon="92:05:28.97", altitude=272, stringsAsFactors=FALSE),
    data.frame(name="apu08", lat="41:59:33.63", lon="92:04:15.06", altitude=282, stringsAsFactors=FALSE),
    data.frame(name="apu09", lat="41:51:41.03", lon="91:53:07.24", altitude=240, stringsAsFactors=FALSE),
    data.frame(name="apu10", lat="41:51:37.75", lon="91:52:25.37", altitude=255, stringsAsFactors=FALSE),
    data.frame(name="apu11", lat="41:50:49.41", lon="91:51:37.04", altitude=259, stringsAsFactors=FALSE),
    data.frame(name="apu12", lat="41:50:50.52", lon="91:50:44.87", altitude=258, stringsAsFactors=FALSE),
    data.frame(name="apu13", lat="41:38:26.23", lon="91:32:30.40", altitude=197, stringsAsFactors=FALSE),
    data.frame(name="apu14", lat="41:38:26.29", lon="91:32:29.86", altitude=197, stringsAsFactors=FALSE)))
  
  s$lat = deg2num(s$lat)    # Degrees N.
  s$lon = -1*deg2num(s$lon) # Translate to degrees E.
  return(s)
}

IFloodSmap = function() {
  ## Plot the locations of the stations in the IFloodS campaign, with
  ## radars assumed to have an effective range of 40 km.

  ## Parsivels in IFloodS.
  parsivels = data.table(IFloodSStations())
  
  ## X-Band radars in IFloodS.
  radars = rbindlist(list(
    data.frame(name="MXPOL2", lat=43.1785011291504, lon=-91.8583984375),
    data.frame(name="MXPOL4", lat=42.9224014282227, lon=-91.4091033935547),
    data.frame(name="MXPOL5", lat=41.8870010375977, lon=-91.7340545654297)))
          
  instruments = rbind(data.table(parsivels, type="Parsivel"),
    data.table(radars, type="Radar"),
    use.names=TRUE, fill=TRUE)
  
  latlon = copy(instruments)
  coordinates(latlon) = ~lon+lat
  proj4string(latlon) = CRS("+proj=longlat +datum=WGS84")
  
  ## Project to km.
  cols = c("x","y")
  instruments[, (cols) := data.table(coordinates(spTransform(latlon, 
      CRS("+proj=utm +zone=15 +ellps=GRS80 +datum=NAD83 +units=m +no_defs"))))]

  circleFun = function(x, y, diameter = 80000, npoints = 100){
    r = diameter / 2
    tt <- seq(0,2*pi,length.out = npoints)
    xx <- x + r * cos(tt)
    yy <- y + r * sin(tt)
    return(data.frame(x = xx, y = yy))
  }
  circles = instruments[type == "Radar", circleFun(x, y), by=name]
  
  plot = ggplot(instruments, aes(x=x, y=y)) + 
    geom_point(aes(colour=name), size=5) +
    geom_path(data=circles, aes(colour=name)) +
    theme_bw(16) +
    scale_colour_discrete(name="Instrument") +
    coord_fixed()
  
  return(plot)
}

updateNASAData = function(dir="/ltedata/IFLOODS_2013/Parsivel_2014/data/dsds/",
                          outFile="~/Dropbox/phd/Rdata/IFloodS/parsivel_ifloods_NASAprocessed_1min.Rdata") {
  ## Read in all processed DSD data from IFloodS as provided by NASA.
  
  pars = NULL
  for(file in list.files(dir, pattern=".*\\.txt", full.name=TRUE))
    pars = rbind(pars, data.table(station=str_extract(file, "apu[0-9][0-9]"), 
                                  read.table(file)))  
    
  pars = data.frame(pars)
  names(pars) = c("station", "year", "DOY", "hour", "minute", paste("class", seq(1,32), sep=""))
  pars = data.table(pars)
  pars[, timestring := paste(year, DOY, hour, minute)]
  pars[, POSIXtime := as.POSIXct(timestring, format="%Y %j %H %M", tz="UTC")]

  ## Remove unrequired columns.
  cols = c("year", "DOY", "hour", "minute", "timestring")
  pars[, (cols) := NULL]
  
  ## Rearrange for display.
  pars = pars[, c("POSIXtime", "station", paste("class", seq(1, 32), sep="")), with=FALSE]

  save(pars, file=outFile)
}
