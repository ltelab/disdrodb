############################################ BEARD MODEL FOR TERMINAL DROP VELOCITY #############################

## This source code was first written in Turbo Pascal by R.Uijlenhoet
## A.Berne translated it into IDL (Oct.2006)
## M.Schleiss translated it into R (June.2008)

g0 <- function(lat){

	## Computes latitudinal gravitational acceleration at mean sea level 
	## According to Weast & Astle 1980

	## Inputs :  lat = latitude [in degrees]  
	## Outputs : g0  = gravitational acceleration at the given latitude [in m/s2]

	g0 = 9.806229 - 0.025889372*cos(2*lat)
        return(g0)
}

g <- function(h,lat){

	## Computes the gravitational acceleration at height h and latitude lat
	
	## Inputs  : h   = the height [in m]
	##           lat = the latitude [in degrees]
	## Outputs : g   = the gravitational acceleration [in m/s2]

	g = g0(lat) - 2.879513*h/(1e6)
	return(g)
}


es <- function(t){

	## Computes the saturation of vapor pressure over water as a function of the temperature
	## See Brutsaert 1982 ; Pruppacher & Klett 1978 ; Flatau & al. 1992

	## Inputs  : t  = the temperature [in Kelvin]
	## Outputs : es = saturation of vapor pressure [-]

	g    <- rep(0,8)
	g[1] <- -0.29912729e4 
	g[2] <- -0.60170128e4
	g[3] <- +0.1887643854e2
	g[4] <- -0.28354721e-1
	g[5] <- +0.17838301e-4
	g[6] <- -0.84150417e-9
	g[7] <- +0.44412543e-12
	g[8] <- +0.2858487e1
	esat <- g[7]
	for(i in seq(6,3,-1)){
 	   esat <- esat*t + g[i]
	}
	esat <- esat + g[8]*log(t)
	for(i in seq(2,1,-1)){
 	   esat <- esat*t + g[i]
	}
	es <- exp(esat/(t^2))
	return(es)
}

eact <- function(rh,t,p){

	## Computes the actual vapor pressure over water
	## See Brutsaert, 1982

	## Inputs  : rh = relative humidity [-]
	##           t  = temperature [in Kelvin]
	##	     p  = air pressure [in Pascal]
	## Outputs : eact = the actual vapor pressur over water [in Pascal]	

	eact <- rh/(1/es(t)-(1-rh)/p)
	return(eact)
}

Ta <- function(h,Ta0,lapse){

	## Computes the air temperature at height h in a standard atmosphere
	## See Brutsaert, 1982 ; Ulaby & al. 1981

	## Inputs  : h   = height [in m]
	##           Ta0 = std temperature at sea level [in Kelvin]
	##           lapse = std atmospheric lapse rate [in K/m]
	## Outputs : Ta  = air temperature [in Kelvin]
	
	Ta <- Ta0 -lapse*h
	return(Ta) 
}


pa <- function(h,pa0,lat,Ta0,lapse,Rd){

	## Computes the air pressure at height h in a standard atmosphere
	## According to the hypsometric formula of Brutsaert 1982 ; Ulaby & al. 1981

	## Inputs  : h     = height [in m]
	##           pa0   = std atmospheric pressure at sea level [in Pa]
	##           lat   = latitude [in degrees]
	##           Ta0   = std atmospheric temperature at sea level [in Kelvin]
	## 	     lapse = std atmospheric lapse rate [in K/m]
	##           Rd    = gas constant for dry air [in J/(kg*K)]
	## Outputs : pa    = air pressure [in Pascal]

	pa = pa0*exp(-g(h,lat)/(lapse*Rd)*log(1+lapse*h/Ta(h,Ta0,lapse)))
	return(pa)
}


ea <- function(h,Ta0,pa0,RH0,lapse){
	
	## Computes the vapor pressure 
	## According to Yamamoto's exponential relationship, See Brutsaert 1982

	## Inputs  : h     = heigth [in m]
	##           Ta0   = std atmopsheric temperature at sea level [in Kelvin] 
	## 	     pa0   = std atmopsheric pressure at sea level [in Pascal]
	##           RH0   = relative humidity at sea level [-]
	##           lapse = std atmospheric lapse rate [in K/m]
	## Outputs : ea = vapor pressure [in Pascal]

	ea0 <- eact(RH0,Ta0,pa0)
	ea  <- ea0*exp(-(5.8*1e3*lapse/(Ta(h,Ta0,lapse)^2)+5.5/(1e5))*h)
	return(ea)
}

rhoa <- function(t,p,e,Rd){
	
	## Computes the air density according to the eqn of state of moist air
	## See Brutsaert 1982

	## Inputs  : t  = temperature [in Kelvin]
	##           p  = pressure [in Pascal]
	##           e  = vapor pressure [in Pascal]
	##           Rd = gas constant for dry air [in J/(kg*K)]
	## Outputs :

	rhoa = p*(1-0.378*e/p)/(Rd*t)
	return(rhoa)
}


eta <- function(t,Kelvin){

	## Computes the dynamic viscosity of dry air
	## See Beard 1977 ; Prupacher & Klett 1978

	## Inputs  : t = temperature [in Kelvin]
	##           Kelvin = 
	## Outputs : eta = dynamic viscosity of dry air [in kg/ms]

	t1 <- t-Kelvin
	if(t>0){
	   eta <- (1.721+0.00487*t1)/(1e5)
	}
	else{
	   eta <-.(1.718+0.0049*t1-1.2*t1^2/(1e5))/(1e5)
	}
	return(eta)
}

rhop <- function(t){

	## Computes the density of pure ordinary water at std pressure
	## According to Kell for temperatures above freezing and according to Dorsch & Boyd
	## for temperatures below freezing. See Pruppacher & Klett 1978 ; Weats & Astle 1980

	## Inputs  : t = temperature [in Kelvin]
	## Outputs : rhop = density of pure water [in kg/m3]

	c <- rep(0,7)
	if(t>0){
	   c[1] =  +9.9983952e2
    	   c[2] =  +1.6945176e1
    	   c[3] = -7.9870401e-3
    	   c[4] = -4.6170461e-5
           c[5] =  +1.0556302e-7
    	   c[6] = -2.8054253e-10
    	   c[7] =  +1.6879850e-2
	   rho <- c[1]
	   tPower <- 1.0
	   for(i in 2:6){
 	      tPower <- tPower*t
	      rho    <- rho + c[i]*tPower
	   }
	   rhop <- rho/(1+c[7]*t)
	}
	else{
	   c[1] <- 999.84
	   c[2] <- 0.086
	   c[3] <- -0.0108
	   rho  <- c[1]
	   tPower <- 1.0
	   for(i in 2:3){
	      tPower <- tPower*t
	      rho    <- rho + c[i]*tPower
	   }
	   rhop <- rho
	}
	return(rhop)
}

kT <- function(t){

	## Computes the isotherman compressibility of pure ordinary water
	## According to Kell, Weats & Astle 1980

	## Inputs  : t  = temperature [in Kelvin]
	## Outputs : kT = compressibility of water [in megabar]

	c    <- rep(0,7)
	c[1] <- +5.088496e1
	c[2] <- +6.163813e-1
	c[3] <- +1.459187e-3
	c[4] <- +2.008438e-5
	c[5] <- -5.857727e-8
	c[6] <- +4.10411e-10
 	c[7] <- +1.967348e-2
	k <- c[1]
	tPower <- 1.0
	for(i in 2:6){
	   tPower <- tPower*t
	   k <- k+c[i]*tPower
	}
	kT <- k/(1+c[7]*t)
	return(kT)
}


rhow <- function(t,p,Kelvin,pa0){

	## Computes the water density according to Weats & Astle 1980
	
	## Inputs  : t = temperature [in Kelvin]
	##           p = pressure [in Pascal]
	##           Kelvin =
	##           pa0 =
	## Outputs : rhow = water density [in kg/m3]

	t1 <- t-Kelvin
	dP <- pa0-p
	rhow <- rhop(t1)*exp(-1e-11*kT(t1)*dP)
	return(rhow)
}


sigma <- function(t,Kelvin){

	## Computes the surface tension of pure ordinary water against air
	## According to Pruppacher & Klett 1978

	## Inputs  : t = temperature [in Kelvin]
	##           Kelvin =
	## Outputs : sigma = surface tension [in N/m]

	t1 <- t-Kelvin
	sigma <- 0.0761-0.000155*t1
	return(sigma)
}


NDa <- function(d0,t,p,e,g,Kelvin,pa0,Rd){

	## Computes the Davies number for a raindrop, quantifying the product
	## of the drag coefficient and the square of the Reynolds number at
	## terminal velocity.
	## See Beard 1976 ; Pruppacher & Klett 1978

	## Inputs  : d0 = equivalent spherical diameter [in m]
	##           t  = temperature [in Kelvin]
	##           p  = pressure [in Pascal]
	##           e  = vapor pressure [in Pascal]
	##           g  = gravitational acceleration [in m/s2]
	##           Kelvin =
	##           pa0 =
	##           Rd  =
	## Outputs : NDa = Davies number [-]

	drho <- rhow(t,p,Kelvin,pa0)-rhoa(t,p,e,Rd)
	NDa <- 4*rhoa(t,p,e,Rd)*drho*g*d0^3/(3*eta(t,Kelvin)^2)
	return(NDa)
}


Bo <- function(d0,t,p,e,g,Kelvin,pa0,Rd){

	## Computes the modified Bond number for a raindrop, qauntifying the
	## relative strength of the drag and surface tension forces acting on
	## the raindrop at terminal velocity.
	## See Beard 1976 ; Pruppacher & Klett 1978

	## Inputs  : d0 = equivalent spherical diameter [in m]
	##           t  = temperature [in Kelvin]
	##           p  = pressure [in Pa]
	##           e  = vapor pressure [in Pa]
	##           g  = gravitational acceleration [in m/s2]
	##           Kelvin =
	##           pa0 =
	##           Rd  =
	## Outputs : Bo = modified Bond number [-]

	drho <- rhow(t,p,Kelvin,pa0)-rhoa(t,p,e,Rd)
	Bo <- 4*drho*g*d0^2/(3*sigma(t,Kelvin))
	return(Bo)
}


Np <- function(t,p,e,g,Kelvin,pa0,Rd){

	## Computes the physical property number for a raindrop.
	## According to Beard 1976 ; Pruppacher & Klett 1978

	## Inputs  : d0 = equivalent spherical diameter [in m]
	##           t  = temperature [in Kelvin]
	##           p  = pressure [in Pa]
	##           e  = vapor pressure [in Pa]
	##           g  = gravitational acceleration [in m/s2]
	##           Kelvin =
	##           pa0 =
	##           Rd  =
	## Outputs : Np  = physical property number [-]	

	drho <- rhow(t,p,Kelvin,pa0)-rhoa(t,p,e,Rd)
	Np <- exp(3*log(sigma(t,Kelvin)))*rhoa(t,p,e,Rd)^2/(exp(4*log(eta(t,Kelvin)))*drho*g)
	return(Np)
}


NRe <- function(d0,t,p,e,g,Kelvin,pa0,Rd){

	## Computes the Reynolds number for a raindrop, quantifying the relative strength
	## of the convective inertia and linear viscous forces acting on the drop at terminal velocity.
	## See Beard 1976 ; Pruppacher & Klett 1978

	## Inputs  : d0 = equivalent spherical diameter [in m]
	##           t  = temperature [in Kelvin]
	##           p  = pressure [in Pa]
	##           e  = vapor pressure [in Pa]
	##           g  = gravitational acceleration [in m/s2]
	##           Kelvin =
	##           pa0 =
	##           Rd  =
	## Outputs : NRe = Reynolds number [-]

	b <- rep(0,7)
	if(d0<1.07e-3){
	   b[1] <- -0.318657e1
	   b[2] <- +0.992696
	   b[3] <- -0.153193e-2
	   b[4] <- -0.987059e-3
	   b[5] <- -0.578878e-3
	   b[6] <- +0.855176e-4
	   b[7] <- -0.327815e-5
	   X <- log(NDa(d0,t,p,e,g,Kelvin,pa0,Rd))
	   Y <- b[1]
	   XPower <- 1.0
	   for(i in 2:7){
	      XPower <- XPower*X
	      Y <- Y + b[i]*XPower
	   }
	   NRe <- exp(Y)
	}
	else{
	   b[1] <- -0.500015e1
	   b[2] <- +0.523778e1
	   b[3] <- -0.204914e1
	   b[4] <- +0.475294
	   b[5] <- -0.542819e-1
	   b[6] <- +0.238449e-2
	   X <- log(Bo(d0,t,p,e,g,Kelvin,pa0,Rd)) + log(Np(t,p,e,g,Kelvin,pa0,Rd))/6
	   Y <- b[1]
	   XPower <- 1.0
	   for(i in 2:6){
	      XPower <- XPower*X
	      Y <- Y + b[i]*XPower
	   }
	   NRe <- exp(log(Np(t,p,e,g,Kelvin,pa0,Rd))/6 + Y)
	}
	return(NRe)
}

Vt1 <- function(d0,t,p,e,g,Kelvin,pa0,Rd){

	## Computes the terminal fall velocity of a raindrop in still air
	## See Beard 1976 ; Pruppacher & Klett 1978
	
	## Inputs  : d0 = equivalent spherical diameter [in m]
	##           t  = temperature [in Kelvin]
	##           p  = pressure [in Pa]
	##           e  = vapor pressure [in Pa]
	##           g  = gravitational acceleration [in m/s2]
	##           Kelvin = freezing temperature of water [in Kelvin]
	##           pa0 = std atmospheric pressure at sea level [in Pascal]
	##           Rd  = gas constant for dry air [in J/(kg*K)]
	## Outputs : Vt1 = terminal fall velocity [in m/s]

	Vt1 <- eta(t,Kelvin)*NRe(d0,t,p,e,g,Kelvin,pa0,Rd)/(rhoa(t,p,e,Rd)*d0)
	return(Vt1)
}

Vt2 <- function(d0,t,p,Kelvin,pa0,Rd){

	## Computes the terminal fall velocity of a raindrop in still air.
	## See Beard 1977

	## Inputs  : d0 = equivalent spherical diameter [in m]
	##           t  = temperature [in Kelvin]
	##           p  = pressure [in Pa]
	##           e  = vapor pressure [in Pa]
	##           g  = gravitational acceleration [in m/s2]
	##           Kelvin = freezing temperature of water [in Kelvin]
	##           pa0 = std atmospheric pressure at sea level [in Pascal]
	##           Rd  = gas constant for dry air [in J/(kg*K)]
	## Outputs : Vt2 = terminal fall velocity [in m/s]

	c     <- rep(0,10)
	c[1]  <- 0.706037e1
	c[2]  <- 0.174951e1
	c[3]  <- 0.486324e1
	c[4]  <- 0.660631e1
	c[5]  <- 0.484606e1
	c[6]  <- 0.214922e1
	c[7]  <- 0.58714
	c[8]  <- 0.96348e-1
	c[9]  <- 0.869209e-2
	c[10] <- 0.33089e-3
	Lnd0  <- log(d0*100)
	LnV0  <- c[1]
	Lnd0Power <- 1.0
	for(i in 2:10){
	   Lnd0Power <- Lnd0Power*Lnd0
	   LnV0 <- LnV0 + c[i]*Lnd0Power
	}
	EpsS <- eta(Kelvin+20,Kelvin)/eta(t,Kelvin)-1
	EpsC <- sqrt(rhoa(Kelvin+20,101325,0,Rd)/rhoa(t,p,0,Rd)) -1
	Eps  <- 1.104*EpsS + (1.058*EpsC-1.104*EpsS)*log(d0/(4e-5))/5.01
	Vt2  <- 0.01*exp(LnV0)*(1+Eps)
	return(Vt2)
}

Cd1 <- function(d0,t,p,e,g,Kelvin,pa0,Rd){

	## Computes the drag coefficient for a raindrop falling at terminal velocity in still air
	## See Beard 1976 ; Pruppacher & Klett 1978

	## Inputs  : d0 = equivalent spherical diameter [in m]
	##           t  = temperature [in Kelvin]
	##           p  = pressure [in Pa]
	##           e  = vapor pressure [in Pa]
	##           g  = gravitational acceleration [in m/s2]
	##           Kelvin = freezing temperature of water [in Kelvin]
	##           pa0 = std atmospheric pressure at sea level [in Pascal]
	##           Rd  = gas constant for dry air [in J/(kg*K)]
	## Outputs : Cd1 = drag coefficient [-]

	drho <- rhow(t,p,Kelvin,pa0)-rhoa(t,p,e,Rd)
	Cd1  <- 4*drho*g*d0/(3*rhoa(t,p,e,Rd)*Vt1(d0,t,p,e,g,Kelvin,pa0,Rd)^2)
	return(Cd1)
}


Cd2 <- function(d0,t,p,Kelvin,pa0,Rd){

	## Computes the drag coefficient for a raindrop falling at terminal velocity in still air
	## See Beard 1977

	## Inputs  : d0 = equivalent spherical diameter [in m]
	##           t  = temperature [in Kelvin]
	##           p  = pressure [in Pa]
	##           e  = vapor pressure [in Pa]
	##           g  = gravitational acceleration [in m/s2]
	##           Kelvin = freezing temperature of water [in Kelvin]
	##           pa0 = std atmospheric pressure at sea level [in Pa]
	##           Rd  = gas constant of dry air [in J/(kg*K)]
	## Outputs : Cd2 = drag coefficient [-] 

	g <- 9.81
	drho <- rhow(t,p,Kelvin,pa0)-rhoa(t,p,0,Rd)
	Cd2  <- 4*drho*g*d0/(3*rhoa(t,p,0,Rd)*Vt2(d0,t,p,Kelvin,pa0,Rd)^2)
	return(Cd2)
}


init_Beard <- function(Temp,Lat,RH0){

	## Initialization of variables for the computation of terminal fall velocity
	## and drag coefficients for liquid raindrops.

	## Inputs  : Temp = temperature [in degrees]
	##           Lat  = latitude [in radians]
	##           RH0  = relative humidity [-]
	## Outputs : a vector out=(Kelvin,lapse,lat,pa0,Rd,RH0,Ta0)
	##           Kelvin = freezing temperature of water [in Kelvin]
	##           lapse  = std atmospheric lapse rate [in K/m]
	##           pa0    = std pressure at sea level [in Pascal]
	##           Rd     = gas constant of dry air [in J/(kg*K)]
	##           Ta0    = std temperature at sea level [in Kelvin]
	##           lat    = latitude [in degrees]

	Kelvin <- 273.15
	lapse  <- 0.0065
	pa0    <- 101325
	Rd     <- 287.04
	Ta0    <- Kelvin+Temp
	lat    <- Lat*pi/180
	out <- c(Kelvin,lapse,lat,pa0,Rd,RH0,Ta0)
	return(out)
}

Beard <- function(d,h,Kelvin,lapse,pa0,Rd,RH0,Ta0,lat){

	## Computes the terminal fall velocity and drag coefficients for liquid raindrops
	
	## Inputs  : d = equivalent spherical diameter [in m]
	##           h = altitude [in m]
	##           Kelvin = freezing temperature of water [in Kelvin]
	##           lapse  = std atmospheric lapse rate [in K/m]
	##           pa0    = std pressure at sea level [in Pascal]
	##           Rd     = gas constant of dry air [in J/(kg*K)]
	##           Ta0    = std temperature at sea level [in Kelvin]
	##           lat    = latitude [in degrees]
	## Outputs : a vector out=(Vt1,Cd1,Vt2,Cd2)
	##           Vt1 = terminal fall velocities [m/s] according to Beard 1976 ; Pruppacher & Klett 1978
	##           Cd1 = corresponding drag coefficient
	##           Vt2 = terminal fall velocity [m/s] according to Beard 1977
	##           Cd2 = corresponding drag coefficient

	g   <- g(h,lat)
	Ta  <- Ta(h,Ta0,lapse)
	pa  <- pa(h,pa0,lat,Ta0,lapse,Rd)
	ea  <- ea(h,Ta0,pa0,RH0,lapse)
	Vt1 <- Vt1(d,Ta,pa,ea,g,Kelvin,pa0,Rd)
	Vt2 <- Vt2(d,Ta,pa,Kelvin,pa0,Rd)
	Cd1 <- Cd1(d,Ta,pa,ea,g,Kelvin,pa0,Rd)
	Cd2 <- Cd2(d,Ta,pa,Kelvin,pa0,Rd)

	out <- c(Vt1,Cd1,Vt2,Cd2)
	return(out)
}
 
