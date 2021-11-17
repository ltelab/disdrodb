# geo_functions.R. 
# 
# Utility functions for geographic calculations.
#
# Author: Tim Raupach <tim.raupach@epfl.ch>

require(gmt)
require(ggplot2)
require(gstat)
require(gridExtra)

geolocateStations = function(stations) {
  # From a list of stations containing lat, lon, altitude, convert 
  # the coordinates to metres and reproject, assign coordinates, and fix 
  # altitudes.
  # 
  # Args:
  #   stations: The stations to fix.
  #
  # Returns: Fixed station data.frame with coordinates defined.
  
  # Convert degrees/minutes to decimal degrees.
  stations$lat = deg2num(stations$lat)
  stations$lon = deg2num(stations$lon)
  
  # Project to metres.    
  projString = paste("+proj=lcc +lat_1=44.10000000000001",
                     "+lat_0=44.10000000000001 +lon_0=0",
                     "+k_0=0.999877499 +x_0=600000 +y_0=3200000",
                     "+a=6378249.2 +b=6356515",
                     "+towgs84=-168,-60,320,0,0,0,0 +pm=paris",
                     "+units=m +no_defs")
  projCRS = CRS(projString)
  # See http://spatialreference.org/ref/epsg/27573/ for projection details.
  
  coords = data.frame(lat=stations$lat, lon=stations$lon)
  coordinates(coords) = ~lon+lat
  proj4string(coords) = CRS("+proj=longlat +datum=WGS84")
  
  metre_coords = spTransform(coords, projCRS)
  
  stations$x_metres = coordinates(metre_coords)[,1]
  stations$y_metres = coordinates(metre_coords)[,2]
  stations$metres_proj4 = projString
  stations$altitude = as.integer(stations$altitude)
  return(stations)
}
