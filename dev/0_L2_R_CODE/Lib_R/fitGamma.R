gammaParams_ML = function(spectra,
    widths=apply(get.classD(), 1, diff),
    diams=rowMeans(get.classD()),
    mu=0.5, lambda=3, ...) {
    ## Maximum likelihood estimation of gamma model parameters, based
    ## on Johnson QJRMS 2014 (DOI:10.1002/qj.2209).
    ##
    ## The function estimates mu and lambda for the model
    ## used in Johnson QJRMS:
    ##
    ##   N(D) = N_T (lambda^(mu+1) / gamma(mu+1)) D^mu exp(-lambda*D).
    ##
    ## N_T is then converted to N_0 [mm-1-mu m-3], such that
    ##
    ##   N(D) = N_0 D^mu exp(-lambda*D).
    ##
    ## Args:
    ##   spectra: The DSD for which to find parameters [mm-1 m-3].
    ##   diams: Class-centre diameters for each DSD bin [mm].
    ##   widths: Class widths for each DSD bin [mm].
    ##   mu: Initial value for shape parameter mu [-].
    ##   lambda: Initial value for slope parameter lambda [mm^-1].
    ##   ...: Extra arguments to option() or cost function
    ##        gammaMLCostFunc().
    ##
    ## Returns: data.table containing mu (> -1) [-].

    spectra = as.numeric(spectra)
    nullresult = data.table(mu=as.numeric(NA), N0=as.numeric(NA),
        lambda=as.numeric(NA))

    ## Minimise the cost function.
    ## Return NA if no convergence is reached.
    res = try(optim(par=c(mu, lambda),
        fn=gammaMLCostFunc, spectra=spectra,
        widths=widths, bins=diams,
        method="Nelder-Mead", ...), silent=TRUE)
    if(class(res) == "try-error") {
        return(nullresult)
    }
    if(res$convergence != 0) {
      return(nullresult)
    }

    ## Get optimal parameters.
    mu = res$par[1]                                    ## [-]
    lambda = res$par[2]                                ## [mm-1]

    ## Estimate tilde_N_T using the total drop concentration.
    tilde_N_T = sum(spectra * widths)                  ## [m^-3].

    ## Convert tilde_N_T to N_T using Johnson's Eqs. 3 and 4.
    p = 1 - sum(lambda^(mu+1)/gamma(mu+1) * diams^mu * ## [mm^(-mu-1) * mm^mu * mm] = [-]
        exp(-lambda*diams) * widths)
    N_T = tilde_N_T/(1-p)                              ## [m-3]

    ## Convert N_T into N0.
    N0 = N_T * lambda^(mu+1)/gamma(mu+1)               ## [m-3 * mm^(-mu-1)]
    return(data.table(mu=mu, lambda=lambda, N0=N0))
}

gammaMLCostFunc = function(pars, spectra,
    lower=0.2495, upper=7, widths=apply(get.classD(), 1, diff),
    bins=rowMeans(get.classD()), truncationIdx=3:23) {
    ## Cost function for the gamma distribution:
    ##   N(D) = N_t (lambda^(mu+1) / gamma(mu+1)) D^mu exp(-lambda*D)
    ## taking into account truncation and binning of drops.
    ##
    ## This function is a modified version of the R code given in
    ## Johnson QJRMS 2014 (DOI:10.1002/qj.2209). This function fits
    ## both mu and lambda.
    ##
    ## Args:
    ##   pars: Vector containing mu [-] and lambda [mm-1] to test.
    ##   spectra: The DSD spectra to fit to [mm-1 m-3].
    ##   lower, upper: Min and max observed diameters [mm].
    ##   widths: Widths of each diameter bin [mm].
    ##   bins: Centres of each diameter bin [mm].
    ##   truncationIdx: Indices inside 'spectra' that contain
    ##                  measurements. (note: should include one
    ##                  extra class that contains zeros, for
    ##                  j+1 in code).
    ##
    ## Returns: Cost for the given parameters. Smaller cost
    ##          equals better fit.

    mu = pars[1]
    lambda = pars[2]

    ## Truncate the spectra, widths and bins, and convert
    ## spectra from mm-1 m-3 to m-3.
    widths = widths[truncationIdx]
    bins = bins[truncationIdx]
    spectra = spectra[truncationIdx] * widths

    if(mu > -1 & lambda > 0) {
        tmp = 0.0
        for (j in 1:length(spectra)) {
            if (spectra[j]>0)
                tmp = tmp - spectra[j] * log( (pgamma(lambda*bins[j+1],mu+1) -
                    pgamma(lambda*bins[j],mu+1))/(pgamma(lambda*upper,mu+1) -
                                                  pgamma(lambda*lower,mu+1)))
        }
        return(tmp)
    } else return(Inf)
}

gammaMLCostFunc_GPM = function(pars, spectra, Dm,
    lower=0.2495, upper=7, widths=apply(get.classD(), 1, diff),
    bins=rowMeans(get.classD()), truncationIdx=3:23) {
    ## Cost function for the gamma distribution:
    ##   N(D) = N_t (lambda^(mu+1) / gamma(mu+1)) D^mu exp(-lambda*D)
    ## taking into account truncation and binning of drops.
    ##
    ## This function is a modified version of the R code given in
    ## Johnson QJRMS 2014 (DOI:10.1002/qj.2209), in which
    ## lambda is determined from mu as is done for the GPM
    ## normalised (Willis) DSD model.
    ##
    ## This function fits only mu.
    ##
    ## Args:
    ##   pars: Vector containing mu [-] to test.
    ##   spectra: The DSD spectra to fit to [mm-1 m-3].
    ##   Dm: The mass-weighted mean drop diameter [mm].
    ##   lower, upper: Min and max observed diameters [mm].
    ##   widths: Widths of each diameter bin [mm].
    ##   bins: Centres of each diameter bin [mm].
    ##   truncationIdx: Indices inside 'spectra' that contain
    ##                  measurements. (note: should include one
    ##                  extra class that contains zeros, for
    ##                  j+1 in code)
    ##
    ## Returns: Cost for the given parameters. Smaller cost
    ##          equals better fit.

    mu = pars[1]
    lambda = (mu + 4)/Dm

    ## Truncate the spectra, widths and bins, and convert
    ## spectra from mm-1 m-3 to m-3.
    widths = widths[truncationIdx]
    bins = bins[truncationIdx]
    spectra = spectra[truncationIdx] * widths

    if(mu > -1 & lambda > 0) {
        tmp = 0.0
        for (j in 1:length(spectra)) {
            if (spectra[j]>0)
                tmp = tmp - spectra[j] * log( (pgamma(lambda*bins[j+1],mu+1) -
                    pgamma(lambda*bins[j],mu+1))/(pgamma(lambda*upper,mu+1) -
                                                  pgamma(lambda*lower,mu+1)))
        }
        return(tmp)
    } else return(Inf)
}

gammaParams_ML_GPM = function(spectra, Dm,
    widths=apply(get.classD(), 1, diff),
    diams=rowMeans(get.classD()),
    mu=0.5, ...) {
    ## Maximum likelihood estimation of gamma model parameters, based
    ## on Johnson QJRMS 2014 (DOI:10.1002/qj.2209), modified to
    ## fit parameters to the GPM normalised DSD model.
    ##
    ## The function estimates mu for a normalised gamma model:
    ##
    ##   N(D) = N_w f(mu) (D/Dm)^mu exp[-(4+mu) D/Dm]
    ##
    ## by fitting mu to the model used in Johnson QJRMS:
    ##
    ##   N(D) = N_T (lambda^(mu+1) / gamma(mu+1)) D^mu exp(-lambda*D)
    ##
    ## But using
    ##
    ##   lambda = (mu+4)/Dm, and therefore
    ##   Nw = N_T (lambda^(mu+1) / gamma(mu+1)) D^mu / f(mu)
    ##
    ## To modify the method for use with the GPM model.
    ##
    ## Args:
    ##   spectra: The DSD for which to find parameters [mm-1 m-3].
    ##   Dm: The mass-weighted mean drop diameter for the spectra [mm].
    ##   widths: Class widths for each DSD bin [mm].
    ##   mu: Initial value for shape parameter mu [-].
    ##   ...: Extra arguments to option() or cost function
    ##        gammaMLCostFunc_GPM().
    ##
    ## Returns: data.table containing mu (> -1) [-].

    spectra = as.numeric(spectra)
    nullresult = data.table(mu=as.numeric(NA))

    ## Minimise the cost function.
    ## Return NA if no convergence is reached.
    res = try(optim(par=c(mu),
        fn=gammaMLCostFunc_GPM, spectra=spectra,
        Dm=Dm, widths=widths, bins=diams,
        method="BFGS", ...), silent=TRUE)
    if(class(res) == "try-error") {
        return(nullresult)
    }
    if(res$convergence != 0) {
      return(nullresult)
    }

    ## Get optimal parameters.
    mu = res$par[1]
    return(data.table(mu=mu))
}
