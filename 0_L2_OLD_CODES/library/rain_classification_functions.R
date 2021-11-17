classifyDSDType = function(data, c1=-1.6, c2=6.3, plot=TRUE) {
    ## Classify DSDs into convective, transition, or stratiform rain type,
    ## using the technique of Thurai_JAOT_2009_2 (and 2010, and tested by 
    ## Thurai_AR_2016).
    ## 
    ## data must contain: 
    ##  - liquid water content LWC [g m^-3].
    ##  - water density wDensity [g mm^-3].
    ##  - mass-weighted mean drop diameter Dm [mm].
    ##  - median volume drop diameter D0 [mm].
    ## 
    ## c1 and c2 are the values used in the separation of convective/
    ## stratiform rain: in Thurai_AR_2016 it is stated that c1 lies in
    ## the range from -1.6 and -1.7, while c2 lies in the range from
    ## 6.3 to 6.4.
    ## 
    ## Returns: data with intercept parameter Nw [mm-1 m-3] added,
    ##          and a convectiveIndex added.

    stopifnot("LWC" %in% names(data))
    stopifnot("D0" %in% names(data))
    stopifnot("wDensity" %in% names(data))
    
    ## Tim of the future: don't panic about the missing 10^3 in this function.
    ## It is because when writing in a paper, we give water density in g cm^-3 
    ## (and 1 g mm^-3 == 1e3 g cm^-3). When the formula is written with rho_w 
    ## in g mm^-3, this conversion is not required.
    data[, Nw := (4^4)/(pi*wDensity) * (LWC/Dm^4)]
    
    ## The convective index shows which side of the "separator" log10(Nw)/D0 
    ## line the point appears on.
    data[, NwSep_log10 := c1*D0 + c2]
    data[, convectiveIndex := log10(Nw) - NwSep_log10]
    
    ## Define classes of rain type.
    data[convectiveIndex > 0.3,  rainClass := "Convective"]
    data[convectiveIndex < -0.3, rainClass := "Stratiform"]
    data[convectiveIndex >= -0.3 & convectiveIndex <= 0.3, rainClass := "Transition"]

    return(data)
}

assignSeason = function(dat) {
    ## Summer = JJA 6, 7, 8
    ## Autumn = SON 9, 10, 11
    ## Winter = DJF 12, 1, 2
    ## Spring = MAM 3, 4, 5

    stopifnot(!("month" %in% names(dat)))
    dat[, month := as.numeric(strftime(POSIXtime, "%m"))]

    summer = c(6,7,8)
    autumn = c(9,10,11)
    winter = c(12,1,2)
    spring = c(3,4,5)
    
    dat[month %in% summer, season := "Summer"]
    dat[month %in% autumn, season := "Autumn"]
    dat[month %in% winter, season := "Winter"]
    dat[month %in% spring, season := "Spring"]
    dat[, month := NULL]
    
    return(dat)
}
