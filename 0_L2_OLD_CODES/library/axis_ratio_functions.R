# Axis ratios of raindrops by different algorithms.
#
# Author: Tim Raupach <tim.raupach@epfl.ch>

require(Hmisc)
require(data.table)

axisRatioBeard = function(diams) {
    ## Return the axis ratio with certain equivolume diameters,
    ## using the algorithm proposed by Beard and Chuang 1987 as
    ## shown in Kalogiros IEEE TGRS 2013, Equation 4.
    
    axisRatios = 1.0048 + 5.7e-4 * diams - 2.628e-2 * diams^2 +
        3.682e-3 * diams^3 - 1.677e-4 * diams^4
    return(axisRatios)
}

axisRatioThurai = function(diams) {
  # Return the axis ratio for drops with certain equivolume diameters,
  # using the algorithm proposed by Thurai and Bringi JAOT 2005.
  #
  # Args:
  #   diams: Vector of equivolume diameters to calculate for.
  #
  # Returns: Vector of corresponding axis ratios.
  
  stop(paste("Thurai 2005 axis ratios superceded by Thurai 2007 and",
             "function axisRatioThurai2007."))
  
  axisRatios = 0.9707 + 4.26e-2*diams - 4.29e-2*diams^2 + 
    6.5e-3*diams^3 - 3e-4*diams^4
  
  return(axisRatios)
}

axisRatioThurai2007 = function(diams) {
  ## Return the axis ratio for drops with certain equivolume diameters,
  ## using the algorithm in Thurai JAOT 2007.
  ##
  ## Args: 
  ##   diams: Vector of equivolume diameters to calculate for.
  ##
  ## Returns: Vector of corresponding axis ratios.
  
  axisRatios = rep(NA, length(diams))
  
  ## Drops less than 0.7 mm are assumed spherical:
  axisRatios[which(diams < 0.7)] = 1
  
  ## Drops with 0.7 <= D <= 1.5 use the Beard and Kubesh (1991) lab measurements (Thurai 2007 Eq. 3):
  ratios_small = 1.173 - 0.5165*diams + 0.4698*diams^2 - 0.1317*diams^3 - 8.5e-3*diams^4
  axisRatios[which(diams >= 0.7 & diams <= 1.5)] = ratios_small[which(diams >= 0.7 & diams <= 1.5)]
  
  ## Drops with D > 1.5 use a new derived relationship from Thurai 2007 Eq. 2.
  ratios_large = 1.065 - 6.25e-2*diams - 3.99e-3*diams^2 + 7.66e-4*diams^3 - 4.095e-5*diams^4
  axisRatios[which(diams > 1.5)] = ratios_large[which(diams > 1.5)]

  stopifnot(!any(is.na(axisRatios)))
  return(axisRatios)
}

axisRatioBrandes = function(diams) {
  # Return the axis ratio for drops with certain equivolume diameters,
  # using the algorithm proposed by Brandes JAM 2002.
  #
  # Args:
  #   diams: Vector of equivolume diameters to calculate for.
  #
  # Returns: Vector of corresponding axis ratios.
    
  axisRatios = 0.9951 + 0.02510*diams - 0.03644*diams^2 + 
    0.005030*diams^3 - 0.0002492*diams^4
  
  return(axisRatios)
}

axisRatioAndsager = function(diams) {
  # Return the axis ratio for drops with certain equivolume diameters,
  # using the algorithm proposed by Andsager JAS 1999.
  #
  # Args:
  #   diams: Vector of equivolume diameters to calculate for.
  #
  # Returns: Vector of corresponding axis ratios.
  
  # Convert diams to centimeters.
  diams = diams * 0.1
  
  # Ratios using empirical data, valid from 1.1 mm to 4.4 mm.
  axisRatios = 1.012 - 0.144*diams - 1.03*diams^2
  
  # Ratios using theoretical equilibrium axis ratios, valid outside 
  # the 1.1 mm to 4.4mm range.
  axisRatiosEquil = 1.0048 + 0.0057*diams - 2.628*diams^2 + 
    3.682*diams^3 - 1.677*diams^4
  
  # Use theory outside empirical range.
  axisRatios[which(diams < 0.11)] = axisRatiosEquil[which(diams < 0.11)]
  axisRatios[which(diams > 0.44)] = axisRatiosEquil[which(diams > 0.44)]
  
  return(axisRatios)
}

axisRatioParsivel = function(diams) {
  # Return the axis ratio for drops with certain equivolume diameters,
  # using the algorithm used by Parsivel disdrometers as defined by 
  # Battaglia JAOT 2010.
  #
  # Args:
  #   diams: Vector of equivolume diameters to calculate for.
  #
  # Returns: Vector of corresponding axis ratios.
  
  axisRatios = rep(NA, length(diams))

  # Drops with diameter <= 1 mm have constant ratio of 1, while drops with d
  # diameter >= 5 mm diameter have constant ratio of 0.7.
  axisRatios[which(diams <= 1)] = 1
  axisRatios[which(diams >= 5)] = 0.7
  
  # In between these two, the axis ratio varies linearly.
  idx = which(diams > 1 & diams < 5)
  axisRatios[idx] = 1.075 - 0.075 * diams[idx]

  stopifnot(!any(is.na(axisRatios)))
  return(axisRatios)  
}

plotAxisRatiosByTechnique = function(diams=seq(0, 8, by=0.1),
                                     textSize=10, lineSize=0.5) {
  # Plot the drop axis ratio by different techniques. 
  #
  # Args:
  #   diams: The equivolume diameters for which to find the axis ratios.
  #   textSize: plot font size (default: 10).
  #   lineSize: plot line size (default: 0.5).
  #
  # Returns: ggplot2 object ready to display.

  parsivel = axisRatioParsivel(diams)
  andsager = axisRatioAndsager(diams)
  brandes = axisRatioBrandes(diams)
  thurai05 = axisRatioThurai(diams)
  thurai07 = axisRatioThurai2007(diams)
  beard = axisRatioBeard(diams)

  toPlot = rbind(data.frame(set="Andsager", axis=andsager, diams=diams),
                 data.frame(set="Brandes", axis=brandes, diams=diams),
                 data.frame(set="Parsivel", axis=parsivel, diams=diams),
                 data.frame(set="Thurai 2005", axis=thurai05, diams=diams),
                 data.frame(set="Thurai 2007", axis=thurai07, diams=diams),
                 data.frame(set="Beard", axis=beard, diams=diams))

  plot = ggplot(toPlot, aes(x=diams, y=axis)) + 
    geom_line(aes(colour=set), size=lineSize) +
    scale_colour_discrete(name="Algorithm") + theme_bw(textSize) +
    labs(x="Equivolume diameter [mm]", y="Axis ratio [-]", 
         title="Axis ratios by algorithm")

  return(plot)
}

parsivelShadedArea = function(diams, h=1, technique="Parsivel") {
  # Back-transform equivolume diameters into Parsivel maximum
  # shaded area, in mm^2, following Battaglia JAOT 2010.
  #
  # Args: 
  #   diams: Equivolume diameters to back-transform.
  #   h: The height of the Parsivel laser beam [mm] (Default: 1 mm).
  #   technique: The technique to use for axis ratios for drops. Can be
  #              parsivel, brandes, andsager, or thurai (default: parsivel).
  #
  # Returns: A vector of measured shaded areas [mm^2].
    
  # Following Battaglia JAOT 2010, we assume that a drop is an oblate 
  # spheroid width major semi-axis A and minor semi-axis B. 2*A is the width 
  # we want to retrieve. We have that the equivolume diameter D for a drop 
  # with axis ratio a is D = 2 * a^{1/3} * A, so the width A is 
  # A = D / (2*a^{1/3}).  
  
  if(!(technique %in% c("Parsivel", "Brandes", "Andsager", "Thurai", "Beard")))
    stop("Invalid ratio technique provided to parsivelShadedArea.")
  axisRatios = switch(technique,
                      Parsivel=axisRatioParsivel(diams),
                      Brandes=axisRatioBrandes(diams),
                      Andsager=axisRatioAndsager(diams),
                      Thurai=axisRatioThurai2007(diams),
                      Beard=axisRatioBeard(diams))
  A = (diams / (2 * axisRatios^(1/3))) ## Semi-major axis (width/2).
  B = A * axisRatios                   ## Height/2.
  
  Fmax = rep(NA, length(diams))
  frac = h/(2*B) ## Fraction of beam taken up by height of drop.
  
  # For large drops (taller than height of beam).
  idx = which(B > (h/2)) 
  if(length(idx) > 0) {
    Fmax[idx] = 2 * A[idx] * B[idx] * 
      (asin(frac[idx]) + frac[idx]*sqrt(1 - frac[idx]^2)) 
  }
  
  # For small drops (less high than height of beam).
  idx = which(B <= (h/2))
  if(length(idx) > 0) {
      ## Ellipse area using semi-major and semi-minor axes.
      Fmax[idx] = pi * A[idx] * B[idx] 
  }
  
  return(Fmax)
}

createShadowLookupTable = function(diams=seq(0, 28, by=0.0001),
                                   techniques=c("Parsivel", "Thurai", 
                                                "Brandes", "Andsager", "Beard")) {
  # Create a lookup table of equivolume diameters to Parsivel shaded area
  # using various axis ratio schemes.
  #
  # Args:
  #   diams: What diameters to calculate [mm] (default 0 to 10 by 0.01).
  #   techniques: What techniques to test?
  # 
  # Returns: lookup table containing diam and Fmax by technique.
  
  table = NULL
  for(t in techniques) {
    f = parsivelShadedArea(diams, technique=t)
    table = rbind(table, data.table(technique=t, diam=diams, Fmax=f))
  }
  
  return(table)
}

lookupDiamFromShadow = function(shad, table, technique) {
  # Look up the diameter that provides a shadow amount closest to a given
  # shadow level.
  #
  # Args:
  #   shad: The Parsivel shadow amount to look up [mm^2].
  #   lookupTable: Result from createShadowLookupTable().
  #   technique: Which axis ratio technique to look up?
  #
  # Returns: The diameter value that gives the closest shadowed amount value.
  
  stopifnot(technique %in% table$technique)
  table = table[which(table$technique == technique),]
  minIdx = apply(as.array(shad), 1, 
                 function(x) { return(which.min(abs(table$Fmax - x))) })
  return(table$diam[minIdx])
}

constructClassesFromBreaks = function(breaks, 
                                      modifyBreaks=c(get.classD()[,1], 
                                                     max(get.classD()[,2]))) {
  # Construct min and max diameter classes from a set of diameter breaks.
  #
  # Args:
  #   breaks: Break points in mm.
  #   modifyClasses: old breaks [mm] to use as a basis, only the first n 
  #                  will be overwritten by the new breaks.
  #
  # Returns: data.frame containing min and max diameters for each class.
  
  modifyBreaks[1:length(breaks)] = breaks
  stopifnot(modifyBreaks[length(breaks)+1] > max(breaks))
  breaks = modifyBreaks
  
  len = length(breaks)
  mins = breaks[1:(len-1)]
  maxs = breaks[2:len]  
  
  return(data.frame(mins, maxs))
}

remapParsivelClasses = function(classes, ratioFunc) {
  ## Remap Parsivel equivolume diameter classes into classes that assume
  ## a different raindrop axis ratio.
  ##
  ## Args: 
  ##   classes: The classes to remap, first col min diam, second col max diam.
  ##   ratioFunc: The axis ratio function to use for remapping, must be one of
  ##              "Thurai", "Parsivel", "Andsager", "Brandes".
  ## 
  ## Returns: Classes remapped.
  
  ## Get the shadow sizes for the existing class mins and maxes.
  mins = classes[,1]
  maxs = classes[,2]
  minShadows = parsivelShadedArea(diams=mins, technique="Parsivel")
  maxShadows = parsivelShadedArea(diams=maxs, technique="Parsivel")
  
  ## Look up the diameters corresponding to these shaded areas,
  ## using the new axis ratio function.
  lookup = createShadowLookupTable(techniques=ratioFunc)
  newClasses = data.frame(
    min=lookupDiamFromShadow(shad=minShadows, table=lookup, technique=ratioFunc),
    max=lookupDiamFromShadow(shad=maxShadows, table=lookup, technique=ratioFunc))
  
  return(newClasses)
}


