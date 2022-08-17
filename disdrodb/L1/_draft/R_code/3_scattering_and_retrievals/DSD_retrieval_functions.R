## DSD_retrieval_functions.R
##
## Functions to do with retrieval of the DSD from radar measurements.
## 
## Based on the techniques shown in Seto et. al and Liao et al for GPM.

source("library/DSD_functions.R")
require(RANN)

ZdrDmLookupTable = function(temps=seq(0, 20, by=1), 
                            elevations=c(4,5,6,8,10,12,14,16,20),
                            freq=9.4,
                            DmVals=seq(0.01, 7, by=0.01), ...) {
  ## Calculate a lookup table for combinations of temperature and
  ## incidence elevation, to get Dm from Zdr.
  ## 
  ## Args: 
  ##   temps: Temperatures to calculate for [deg. C].
  ##   elevations: Incidences to calculate for [deg above horizontal].
  ##   freq: Radar frequency [gHZ].
  ##   DmVals: Dm values to include [mm].
  ##   ...: Extra arguments to ZdrFromDm.
  ## 
  ## Returns: data.table with lookup table.
  
  res = NULL
  for(temp in temps) {
    for(el in elevations) {
      tab = ZdrFromDm(DmVals=DmVals, freq=freq, temp=temp, incidence=el, ...)
      res = rbind(res, tab)
    }
  }
  
  return(res)
}

ZdrFromDm = function(DmVals, freq, temp, incidence,
                     D=rowMeans(get.classD()), 
                     widths=apply(get.classD(), 1, diff),
                     mu=3, lambda=mu+4,
                     ratio_function=raindrop_axis_ratio) {
  ## Calculate Zdr lookup tables for mass-weighted mean diameters.
  ## 
  ## Args:
  ##  DmVals: The values of Dm for which to calculate Zdr [mm].
  ##  freq: Radar frequency [gHZ].
  ##  temp: Temperature to calculate for [deg. C].
  ##  D: diameter class centres (default: Parsivel classes).
  ##  widths: widths of diameter classes (default: Parsivel classes).
  ##  mu: DSD model mu [-] (default: 3)
  ##  lambda: DSD model lambda [mm-1] (default: mu+4).
  ##  ratio_function: Drop axis ratio function.
  ## 
  ## Returns: lookup table giving Zdr for different values of Dm.
  
  ## Find wavelength and refractivity index.
  cL = 299792.458*1e3                       # Speed of light [m/s].
  wavelength = (cL/freq*1e-9)*1e3           # Wavelength [mm].
  index_water = ref_index_water(temp, freq) # Refractivity index of water.
  
  ## Get drop axis ratios [vert/horiz].
  axis_ratios = ratio_function(D)
  axis_ratios[which(axis_ratios < 0)] = 0.6
  
  ## Back scattering cross sections [cm2] in horiz and vertical.
  ## Canting angles are assumed to be zero!! 
  sigmaB_H = back_scat_cross(D, axis_ratios, wavelength,
                             index_water, (90-incidence), 0)[,1]
  sigmaB_V = back_scat_cross(D, axis_ratios, wavelength,
                             index_water, (90-incidence), 0)[,2]
  
  ## Note that 1e2*sigmaB is in mm2, widths is in mm,
  ## D is in mm, D/Dm is [-], fMu is [-], so the result
  ## is in mm7, but we assume N_w = 1 mm-1 m-3, so
  ## final result is in mm6 m-3.
  Dm = data.table(Dm=DmVals)
  Dm = Dm[, lambda := (4+mu)/Dm]
  
  func = function(Dm, lambda) {
    return(10*log10(sum((D/Dm)^mu * exp(-lambda*D) * (1e2*sigmaB_H) * widths) /
             sum((D/Dm)^mu * exp(-lambda*D) * (1e2*sigmaB_V) * widths)))
  }
  
  ## UNITS (in equation order):
  ## D [mm]
  ## Dm [mm]
  ## D/Dm [-]
  ## mu [-]
  ## lambda [mm-1] (slope)
  ## lambda*D [-]
  ## 1e2 sigmaB [mm2]
  ## widths [mm]
  ## Thus sum [mm3]
  ## sum [mm7]
  ## sum * factor * Nw = 1 [mm-1 m-3] = mm6 m-3
  
  lookup = Dm[, list(Dm=Dm, temperature=temp, incidence=incidence, 
                     Zdr=func(Dm, lambda)), by=1:length(DmVals)]
  return(lookup[, list(Dm, temperature, incidence, Zdr)])
}

DmForZdr = function(Zdr, temp, elevation, lookup=lookupTable) {
  ## Return the closest-matching Dm for a given Zdr.
  ## 
  ## Args:
  ##   Zdr: The Zdr(s) to lookup [dB].
  ##   temp: The temperature for each Zdr record [deg. C].
  ##   elevation: The radar incidence [deg above horizontal].
  ##   lookup: The lookup table to use (see ZdrDmLookupTable).
  ## 
  ## Returns: Corresponding values of Dm [mm].
  
  table = lookup[temperature == round(temp, 0) & 
                   incidence == elevation]
  if(length(table[, temperature]) == 0)
    return(NULL)
  
  Zdr = as.matrix(Zdr)
  lookupZdr = matrix(table$Zdr, ncol=1)
  lookupDms = matrix(table$Dm, ncol=1)
  
  closestIdxs = nn2(data=lookupZdr, query=Zdr, k=1)$nn.idx
  return(lookupDms[closestIdxs])
}
