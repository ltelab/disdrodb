## weather_functions.R
##
## Functions for dealing with weather station data such as
## that of the Vaisala.
##
## Author: Tim Raupach <tim.raupach@epfl.ch>

freezingHeight = function(temp, altitude, lapse=0.0065) {
    ## Find the expected freezing height.
    ##
    ## Args:
    ##  temp: A temperature [deg. C].
    ##  altitude: Altitude for the temperature [m].
    ##  lapse: Atmospheric lapse rate [K m-1].
    ##
    ## Returns: expected freezing height [m] a.s.l.

    return(altitude+(temp/lapse))
}
