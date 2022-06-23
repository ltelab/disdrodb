# HYMEX_SOP_plot_functions.R
# HYMEX_SOP_plot_functions.R
#
# Functions to support plotting of HYMEX SOP data.
#
# Author: Tim Raupach <tim.raupach@epfl.ch>
  
stationsDefinition_2014 = function() {
  # Define Parsivel stations for HYMEX campaign in Ardeche in 2014.
  #
  # Returns:
  #  A data.frame containing station number, name, label (for plotting),
  #  latitude (lat), longitude (lon), altitude [m], x_metres, y_metres,
  #  projString,

  stations = NULL
  stations = rbind(stations, c(10, "Mirabel", "Mirabel", "44:36.416", "4:29.923", 496))
  stations = rbind(stations, c(11, "Lussas", "Lussas", "44:36.739", "4:28.238", 289))
  stations = rbind(stations, c(12, "St-Germain", "St-Germain", "44:33.305", "4:26.980", 204))
  stations = rbind(stations, c(13, "Lavilledieu", "Lavilledieu", "44:34.631", "4:27.195", 227))
  stations = rbind(stations, c(30, "Pradel Grainage", "Pradel Grainage", "44:34.740", "4:30.066", 271))
  stations = rbind(stations, c(31, "Les Blaches", "Les Blaches", "44:36.049", "4:28.859", 429))
  stations = rbind(stations, c(32, "Pradel 1", "Pradel", "44:34.973", "4:29.920", 278))
  stations = rbind(stations, c(33, "Pradel 2", "", "44:34.973",  "4:29.920", 278))

  stations = as.data.frame(stations, stringsAsFactors=F)
  stations$intTime = 30
  names(stations) = c("number", "name", "label", "lat", "lon", "altitude", "intTime")

  stations = geolocateStations(stations)
  return(stations)
}

stationsDefinition_2013 = function() {
  # Define Parsivel stations for HYMEX campaign in Ardeche in 2013.
  #
  # Returns:
  #  A data.frame containing station number, name, label (for plotting),
  #  latitude (lat), longitude (lon), altitude [m], x_metres, y_metres,
  #  projString,

  stations = NULL
  stations = rbind(stations, c(10, "Mirabel", "Mirabel", "44:36.416", "4:29.923", 496))
  stations = rbind(stations, c(11, "Lussas", "Lussas", "44:36.739", "4:28.238", 289))
  stations = rbind(stations, c(12, "St-Germain", "St-Germain", "44:33.305", "4:26.980", 204))
  stations = rbind(stations, c(13, "Lavilledieu", "Lavilledieu", "44:34.631", "4:27.195", 227))
  stations = rbind(stations, c(20, "Montbrun", "Montbrun", "44:36.845", "4:32.763", 602))
  stations = rbind(stations, c(30, "Pradel Grainage", "Pradel Grainage", "44:34.740", "4:30.066", 271))
  stations = rbind(stations, c(31, "Les Blaches", "Les Blaches", "44:36.049", "4:28.859", 429))
  stations = rbind(stations, c(32, "Pradel 1", "Pradel", "44:34.973", "4:29.920", 278))
  stations = rbind(stations, c(33, "Pradel 2", "", "44:34.973",  "4:29.920", 278))

  stations = as.data.frame(stations, stringsAsFactors=F)
  stations$intTime = 30
  names(stations) = c("number", "name", "label", "lat", "lon", "altitude", "intTime")

  stations = geolocateStations(stations)
  return(stations)
}

stationsDefinition_2012 = function() {
  # Define Parsivel stations for HYMEX campaign in Ardeche in 2012.
  #
  # Returns:
  #  A data.frame containing station number, name, label (for plotting),
  #  latitude (lat), longitude (lon), altitude [m], x_metres, y_metres,
  #  projString,

  stations = NULL
  stations = rbind(stations, c(10, "Mirabel", "Mirabel", "44:36.416", "4:29.923", 496))
  stations = rbind(stations, c(11, "Lussas", "Lussas", "44:36.739", "4:28.238", 289))
  stations = rbind(stations, c(13, "Lavilledieu", "Lavilledieu", "44:34.631", "4:27.195", 227))
  stations = rbind(stations, c(30, "Les Blaches", "Les Blaches", "44:36.049", "4:28.859", 429))
  stations = rbind(stations, c(31, "St-Germain", "St-Germain", "44:33.305", "4:26.980", 204))
  stations = rbind(stations, c(32, "Pradel 1", "Pradel", "44:34.973", "4:29.920", 278))
  stations = rbind(stations, c(33, "Pradel 2", "", "44:34.973",  "4:29.920", 278))
  # stations = rbind(stations, c(50, "2DVD Le Pradel", "", "",  ""))
  stations = as.data.frame(stations, stringsAsFactors=F)
  names(stations) = c("number", "name", "label", "lat", "lon", "altitude")

  stations = geolocateStations(stations)
  return(stations)
}
 