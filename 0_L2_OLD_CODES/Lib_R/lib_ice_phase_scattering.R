############################## ICE PAHSE Library ##############################
##############################  by J.Grazioli    ############################## 

## This is the beginning of a compilation of functions that might be helpful for
## ice-phase scattering calculations

#--------------------#-------------------------#------------------#-------------#
######################### Refractive index of pure ice #########################

ref_index_ice <- function(t,f){

  ## From the article of Hufford 1991: "A model for the complex permittivity of
  ## water at frequencies below 1 THz" 

  ## And also, "Ice and water permettivities for millimeter and sub-millimeter 
  ## remote sensing applications" by Jiang, 2004.

  ## shOULD BE VALID FOR TEMPERATURES IN THE RANGE -40 TO 0 °C
 
  ## Inputs : 
  ##   t = the temperature (in °C)
  ##   f = the frequency (in GHz)

  ## Outputs :
  ##   m = m' + im'' the complex refractive index

  ## Source : J. Grazioli (Jan. 2015)

  Theta       <- -1 + 300/(273.15+t)
  alpha       <- (50.4+62*Theta)*10^(-4.)*exp(-22.1*Theta)
  beta        <- 10.^(-4.)*(0.502-0.131*Theta)/(1+Theta)
  beta        <- beta+0.542*10^(-6)*((1+Theta)/(Theta+0.0073))^2

  epsilonprime    <- 3.15
  epsilonsecond   <- alpha/f+beta*f

  Epsilon <- complex(real=epsilonprime,imaginary=epsilonsecond)

  m <- sqrt(Epsilon)
  return(m)

}

	
	
#################################################################################
################ Refractive index of ice crystals ###############################
# As a function of density (from adapted IDL code of Jordi F. V.) ##############

ref_index_ice_crystals <- function(dens,ref_ind_ice){

  ## From a code of Jordi Figueras y Ventura (Meteo Suisse)

  ## It computes the refractive index of a mixture of air and ice
  ## with a given apparent density (the exact definition of apparent density is unclear)

 
  ## Inputs : 
  ##   dens = apparent density in [g mm^-3]
  ##   ref_ind_ice = refractive index of solid ice (complex)

  ## Outputs :
  ##   m = m' + im'' the complex refractive index

  ## Source : J. Grazioli (May. 2015)


  ks           <-      dens/0.000916*(ref_ind_ice^2.-1.)/(ref_ind_ice^2.+2.)	
  mrs          <-      ((1.+2.*ks)/(1.-ks))^(0.5)


  return(mrs)


} 
