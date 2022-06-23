## fitGG.stacy = function(pdf, n=2000) {
##     ## Fit a generalised gamma (GG) distribution to a pdf, using the
##     ## method of Stacy 1962 as implemented in the VGAM package.
##     ## Note warnings in ?gengamma.stacy.
##     ##
##     ## Args:
##     ##   pdf: The pdf to fit, must contain at least x (input), dx
##     ##        (class width), and p (probability).
##     ##   n: The number of points to simulate drawing from the pdf.
##     ##
##     ## Returns: parameters c, mu, and lambda for the GG distribution.

##     require(VGAM)
##     stopifnot(c("x", "p", "dx") %in% names(pdf))

##     dat = data.table(X=rep(x=pdf[, x], times=round(n * pdf[, p] * pdf[, dx], 0)))
##     fit = vglm(X ~ 1, gengamma.stacy, data=dat)

##     coefs = exp(coef(fit, matrix=TRUE))
##     c = coefs[2]
##     lambda = 1/coefs[1]
##     mu = coefs[3]

##     return(data.table(c=c, mu=mu, lambda=lambda))
## }

## ################################ TEST CODE ####################################

## fitGG.noufaily = function(pdf, n=2000, lambda=NA, trace=FALSE, maxIter=200) {
##     ## Fit a generalised gamma (GG) distribution to a pdf, using the
##     ## maximum likelihood method of Noufaily_CS_2013, Section 2.3.
##     ##
##     ## Args:
##     ##   pdf: The pdf to fit, must contain at least x (input), dx
##     ##        (class width), and p (probability).
##     ##   n: The number of points to simulate drawing from the pdf.
##     ##   startC, startMu, startLambda: Starting parameter values.
##     ##   lambda: Force the value of lambda? (Default: NA, estimate lambda).
##     ##   trace: Print debugging information? (defalut FALSE)
##     ##   maxIter: Maximum number of iterations allowed (default: 200).
##     ##
##     ## Returns: parameters c, mu, and lambda for the GG distribution.

##     stopifnot(c("x", "p", "dx") %in% names(pdf))

##     ## Simulate n samples drawn from the pdf and log transform the samples.
##     d = data.table(X=rep(x=pdf[, x], times=round(n * pdf[, p] * pdf[, dx], 0)))
##     d[, Y := log(X)]

##     ## This routine finds parameters k, sigma, and mu. Note that in
##     ## this function, variables are defined as in Noufaily_CS_2013 and
##     ## are transformed back to our c, sigma, mu at the end of the
##     ## function!

##     ## mu = log(1/lambda) + beta^(-1) * log(alpha/beta)
##     ## lambda = 1/theta

##     ## Let T be samples from a GG distribution. Let Y be the
##     ## log-transformed samples.
##     meanY = d[, mean(Y)]
##     n = d[, length(Y)]

##     ## (1) Make initial guess for L = L0 > 0. L = (mu -
##     ## mean(Y))/sigma.  mu can be positive or negative, but initial L0
##     ## needs to be positive, so take absolute value of mu for initial
##     ## guess.
##     i = 0

##     ## Initial values of k and sigma.
##     k = rgamma(1, shape=2, scale=1)
##     sigma = rgamma(1, shape=2, scale=1)
##     mu = rnorm(1)
##     prevLL = -Inf

##     while(i < maxIter) {
##         L = (mu-meanY)/sigma

##         ## (2) Increase iteration number.
##         i = i + 1

##         ## (3) Compute k_i using Noufaily Eq. 2.6.
##         if(L < 0 & i == 0) {
##             k = 1/L^2
##         } else {
##             k_func = function(k) { log(k) - digamma(k) - (L/sqrt(k)) }
##             k = optim(par=1/(L^2), fn=k_func, method="Brent",
##                 lower=1/(4*L^2), upper=1/(L^2))$par
##             stopifnot(k > 0)
##         }

##         ## (4) Use updated k to update sigma using Noufaily Eq. 2.5.
##         S1 = (1/n) * sum(d[, Y] * exp(d[, Y]/(sigma*sqrt(k))))
##         S0 = (1/n) * sum(exp(d[, Y]/(sigma*sqrt(k))))
##         maxSigma = sqrt(k) * (max(d[, Y]) - meanY)
##         sigma_func = function(sigma) { abs(S1/S0 - meanY - sigma/sqrt(k)) }
##         sigma = optim(par=sigma, fn=sigma_func, method="Brent",
##             lower=0, upper=sqrt(k)*(d[, max(Y)]-meanY))$par
##         stopifnot(sigma > 0)

##         ## (5) Use updated k and sigma to update mu using Noufaily Eq. 2.4.
##         S0 = (1/n) * sum(exp(d[, Y]/(sigma*sqrt(k))))
##         mu = sigma*sqrt(k)*log(S0)

##         if(k > 171) {
##             gammaK = sqrt(2*pi/k) * (k/exp(1))^k
##         } else {
##             gammaK = gamma(k)
##         }

##         ## (6) Use estimates to update L and calculate value of
##         ## log-likelihood function.
##         logLikelihood = n*
##             (-log(sigma) +
##              (k - 1/2)*log(k) -
##              gammaK +
##              sqrt(k)*((meanY-mu)/sigma) -
##              k*exp(-mu/(sigma*sqrt(k)))*S0)

##         if(trace) {
##             print(paste("k: ", round(k, 2), " mu: ", round(mu, 3), " sigma: ",
##                         round(sigma, 3), " LL: ", round(logLikelihood, 1), sep=""))
##         }

##         ## If likelihood is reduced, use the previous values.
##         if(logLikelihood < prevLL) {
##             k = bestK
##             sigma = bestSigma
##             mu = bestMu
##             break
##         }

##         bestK = k
##         bestSigma = sigma
##         bestMu = mu
##         prevLL = logLikelihood
##     }

##     ## Back-transform paraters to GG parameters in Noufaily 2013.
##     ## (alpha, beta, theta).
##     beta = 1/(sigma*sqrt(k))
##     alpha = beta*k
##     theta = exp(mu - log(alpha/beta)/beta)

##     ## Transform to our own parameters for the GG function:
##     c = beta
##     mu = alpha/beta
##     lambda = 1/theta

##     return(list(c=c, mu=mu, lambda=lambda))
## }

## fitGG.wingo = function(pdf, n=2000, lambda=NA) {
##     ## Fit a generalised gamma (GG) distribution to a pdf, using the
##     ## maximum likelihood method of Hager and Bain 1970, and the root
##     ## isolation method of Wingo 1987, as written in Noufaily_CS_2013,
##     ## Section 3.
##     ##
##     ## Args:
##     ##   pdf: The pdf to fit, must contain at least x (input), dx
##     ##        (class width), and p (probability).
##     ##   n: The number of points to simulate drawing from the pdf.
##     ##
##     ## Returns: parameters c, mu, and lambda for the GG distribution.

##     ## Simulate n samples drawn from the pdf.
##     d = data.table(X=rep(x=pdf[, x], times=round(n * pdf[, p] * pdf[, dx], 0)))
##     n = d[, length(X)]
##     x = d[, X]

##     ## Noufaily Eq. 3.3.
##     theta_func = function(beta, k) {
##         return((sum(x^beta) / (n*k))^(1/beta))
##     }

##     ## Noufaily Eq. 3.2.
##     k_func = function(beta) {
##         return( (-1/beta) * ((1/n) * sum(log(x)) -
##                              (sum((x^beta)*log(x)))/(sum(x^beta)))^(-1))
##     }

##     ## Noufaily Eq. 3.1.
##     H_func = function(beta) {
##         res = NULL
##         for(b in beta)
##             res = c(res,
##                 (-1*digamma(k_func(b)) + (b/n) * sum(log(x)) -
##                  log(sum(x^b)) + log(n*k_func(b))))
##         return(res/(beta^2))
##     }

##     ## Find the root of H(beta).
##     ## The root must be positive.
##     beta = uniroot.all(H_func, lower=0.1, upper=10000, n=50)

##     results = data.table(beta=beta)
##     results[, k := k_func(beta), by=1:nrow(results)]
##     results[, theta := theta_func(beta, k), by=1:nrow(results)]

##     ## Wingo: a -> theta
##     ## Wingo: b -> beta
##     ## Wingo: k -> k

##     results[, LL := n*log(beta/(theta^(beta*k)*gamma(k))) +
##             (beta*k - 1)*sum(log(x)) - theta^(-beta)*sum(x^beta),
##             by=1:nrow(results)]

##     ## Choose the best root.
##     beta = results[which.max(LL), beta]
##     k = results[which.max(LL), k]
##     theta = results[which.max(LL), theta]

##     ## Convert back to our parameter values.
##     c = beta
##     mu = k
##     lambda = 1/theta

##     return(data.table(c=c, mu=mu, lambda=lambda))
## }

## fitGG.gomes = function(pdf, lambda, n=10000) {
##     ## Fit a generalised gamma (GG) distribution to a pdf, using the
##     ## iterative method of Gomes MSC 2008.
##     ##
##     ## Args:
##     ##   pdf: The pdf to fit, must contain at least x (input), dx
##     ##        (class width), and p (probability).
##     ##   lambda: The known parameter lambda.
##     ##   n: The number of points to simulate drawing from the pdf.
##     ##
##     ## Returns: parameters c, mu, and lambda for the GG distribution.

##     ## Draw n values from the distribution with probabilities given by
##     ## the pdf.
##     x = approx(cumsum(pdf[, p])/sum(pdf[, p]),
##         pdf[, x],
##         runif(n))$y

##     x = x[!is.na(x)]
##     n = length(x)

##     testScale = function(x, c) {
##         ## If x follows a generalised gamma distribution, then Y = x^c
##         ## follows an ordinary gamma distribution.
##         Y = x^c

##         ## Estimate the gamma distribution parameters for Y.
##         scale = mean(Y)/var(Y)
##         shape = mean(Y)^2/var(Y)

##         recon = rgamma(n=length(Y), shape=shape, scale=scale)

##         plot(density(Y))
##         plot(density(recon))

##         probs = seq(0, 1, by=0.01)
##         diffs = (quantile(Y, probs=probs) -
##                  quantile(recon, probs=probs))

##         return(abs(sum(diffs^2)))
##     }

##     c = optim(par=1, testScale, method="Brent", lower=0, upper=100, x=x)$par
##     mu = mean(x^c)^2/var(x^c)
##     lambda = (mean(x^c)/var(x^c))^(1/c)

##     return(data.table(c=c, mu=mu, lambda=lambda))
## }

## momentsGGFit = function(pdf_p, pdf_x, pdf_dx, Dpower=1, maxIter=10) {
##     ## Perform a fit of the generalised gamma function to find the
##     ## optimum values for c and mu, using the technique of
##     ## Maur_JAS_2001.
##     ##
##     ## Args:
##     ##   dat: The data to fit to; must contain:
##     ##        hx - output of h(x) normalised DSD function.
##     ##        x - corresponding normalised diameter x input.
##     ##
##     ## Returns: a data table with fitted values for c and mu.

##     ## Check the pdf integrates to 1.
##     stopifnot(abs(sum(pdf_p * pdf_dx) - 1) < 1e-10)

##     ## In Maur 2001 the equations are written for moments 0, 1, 2, and
##     ## 3 of the distribution to fit to. But they also say that to
##     ## capture large moments, it makes sense to fit to P(D^3) which is
##     ## the same as fitting to moments 0, 3, 6, and 9. Here M0, M1, M2,
##     ## and M3 are moments 0, and orders from fitWithMoments, which can
##     ## be chosen depending on the fit required.

##     ## Moments to fit to, other than zero:
##     moment1 = 1*Dpower
##     moment2 = 2*Dpower
##     moment3 = 3*Dpower

##     ## Calculate moments of the normalised DSD distribution.
##     m1 = sum(pdf_p * pdf_x^moment1 * pdf_dx)
##     m2 = sum(pdf_p * pdf_x^moment2 * pdf_dx)
##     m3 = sum(pdf_p * pdf_x^moment3 * pdf_dx)

##     ## The real v and w are based on moments of the distribution, in
##     ## Auf der Maur 2001, Eq. 18 for v and Eq. 19 for w.
##     v = m2 * m1^(-2) - 1
##     w = (m3 * m1^(-3) - 3*v - 1) * v^(-2)

##     ## First estimate for c (Eq 23 in Auf der Maur 2001).
##     c = (3 + v - w)/(1 + v*(w-1))

##     itns = 0
##     w1 = NULL
##     while(itns < maxIter) {
##         ## Solve for Eq. 21 in Maur 2001 to find a value of mu that corresponds with c,
##         ## using guess suggested in Maur 2001 as starting value.
##         muSolve = nlsLM("v ~ gamma(mu + 2/c) * gamma(mu) * gamma(mu + 1/c)^(-2) - 1",
##             data=data.frame(c=c, v=v), start=list(mu=v^(-1) * c^(-2)))
##         mu = as.numeric(coef(muSolve))

##         ## Calculate w based on this guess for mu (Eq. 22 in Maur 2001).
##         w0 = w1 ## Save previous w.
##         w1 = (gamma(mu + 3/c) * gamma(mu)^2 * gamma(mu + 1/c)^(-3) - 3*v - 1) * v^(-2)

##         ## Update the estimate of r (Eq.
##         if(itns == 0) {
##             w0 = w - 0.1
##             c0 = (3 + v - w0)/(1 + v*(w0-1))
##         }

##         ## Gradient.
##         step = (c-c0)/(w1-w0)

##         ## Update c for iteration.
##         c0 = c
##         c = c + (w - w1) * step

##         itns = itns + 1
##         print(paste("iteration", itns, ", c: ", c, ", mu:", mu, sep=""))

##         ## Stop when c hardly changes.
##         if(abs(c - c0) < 1e-4 |
##            abs(w1 - w0) < 1e-4)
##             break
##     }

##     if(itns == maxIter)
##         stop("Maximum iterations reached in momentsGGfit")

##     lambda = (gamma(mu+1/c) / gamma(mu)) / m1

##     ## Back-transform parameters for Dpower.
##     c = c*Dpower
##     lambda = lambda^(2-1/Dpower)*1^(Dpower)

##     ## dat = data.table(x=pdf_x, p=pdf_p,
##     ##     p_rec=gammaProbability(pdf_x, c=c*10, mu=mu, lambda=lambda))
##     ## ggplot(dat, aes(x=x, y=p)) +
##     ##     geom_point() +
##     ##         geom_line(aes(y=p_rec), col="red") +
##     ##             scale_y_log10(limits=c(0.001, max(dat$p)))

##     return(data.table(mu=mu, c=c))
## }

## empiricalPDF = function(dsds, i, j, dsdCols, diamCols, widthCols, xClassSize=0.02) {
##     ## Find normalised DSDs.
##     normDSDs = dsds[, callNormalisedDSD(.SD, dsdCols=dsdCols,
##         widthCols=widthCols, diamCols=diamCols, i=i, j=j), by=by]

##     ## Find values for the pdf of x.
##     dsds[, M0 := DSDMoment(.SD, n=0, dsdCols=dsdCols, widthCols=widthCols, diamCols=diamCols)]
##     dsds[, ithMoment := DSDMoment(.SD, n=i, dsdCols=dsdCols, widthCols=widthCols, diamCols=diamCols)]
##     dsds[, jthMoment := DSDMoment(.SD, n=j, dsdCols=dsdCols, widthCols=widthCols, diamCols=diamCols)]
##     dsds[, N0_prime := ithMoment^((j+1)/(j-i)) * jthMoment^((i+1)/(i-j))]
##     dsds[, Dm_prime := (jthMoment/ithMoment)^(1/(j-i))]
##     dsds[, pdf_fact := N0_prime * (Dm_prime / M0)]

##     setkeyv(dsds, c("POSIXtime", by))
##     setkeyv(normDSDs, c("POSIXtime", by))
##     normDSDs[, pdf := hx * dsds[normDSDs, pdf_fact]]

##     ## Find the median pdf value per class of x.
##     xClasses = seq(0, max(normDSDs$x)+xClassSize, by=xClassSize)
##     normDSDs[, xClass := cut(x, xClasses, labels=FALSE)]

##     pdf = normDSDs[, list(meanP=mean(pdf),
##         medianP=median(pdf)), by=xClass]

##     pdf[, x := xClasses[xClass]+xClassSize/2]
##     pdf[, xClass := NULL]

##     ggplot(normDSDs, aes(x=x, y=hx)) + geom_point() +
##         scale_y_log10() +
##         stat_smooth(method="auto")



##     stat_smooth(data=normDSDs, aes(x=x, y=hx), method="auto")

##     set = normDSDs[hx > 0]
##     foo = gam(hx~s(x, bs="cs"), data=set)
##     plot(set$x, predict(foo))
##     dat = data.table(x=set$x, hx=predict(foo))

##     ggplot(normDSDs, aes(x=x, y=hx)) + geom_point() +
##         scale_y_log10() +
##             geom_line(data=dat, colour="red")

##     ggplot(pdf, aes(x=x, y=meanP)) +
##         geom_line(colour="red") +
##             geom_line(aes(y=medianP), colour="green") +
##         scale_y_log10()

##     pdf[, sum(medianP*0.1)]

##     ## Check
##     that pdfs integrate to 1.
##     stopifnot(abs(normDSDs[, sum(pdf*xWidth), by=c("POSIXtime", by)]$V1 - 1) < 1e-8)

##     samples = normDSDs[, drawFromEmpiricalPDF(x, xWidth, pdf, n=1000), by=c("POSIXtime", by)]
##     setnames(samples, "V1", "sample")

##     res = density(samples$sample, bw=0.02)
##     plot(res$x, res$y, log="y")

##     quantile(samples$sample, probs=(seq(0, 1, length.out=100)))

##     require(fBasics)
##     f = ssdFit(samples[, sample])
##     u = seq(0.1, 1.4, by=0.01)



## }

## drawFromEmpiricalPDF = function(x, w, p, n, plot=FALSE) {
##     ## Draw samples from an "empirical" PDF.
##     ##
##     ## Args:
##     ##   x: values at centre of each class.
##     ##   w: width of each class.
##     ##   p: probability for each class.
##     ##   n: number of samples to draw.
##     ##
##     ## Returns: n samples that follow the distribution; note that
##     ##          interpolation is "constant" so all samples are
##     ##          members of "x" repeated an appropriate number of
##     ##          times.

##     ## Use left ends of classes for constant interpolation.
##     x = x-w/2

##     ## Check x values are in order.
##     stopifnot(all(diff(x) > 0))

##     ## Find the cumulative density function for the pdf.
##     cdf = cumsum(p)/sum(p)

##     ## Generate random numbers between 0 and 1.
##     rands = runif(n)

##     samples = approx(cdf, x, rands, rule=2, method="constant", f=1)$y

##     if(plot) {
##         ## Check graphically that the results are good.
##         hist(samples, probability=TRUE, breaks=c((x-w/2)[1], x+w/2),
##              xlim=c(0,1), ylim=c(0,max(p)))
##         points(x, p, col="red", pch="x")
##     }

##     stopifnot(!any(is.na(samples)))
##     return(samples)
## }
