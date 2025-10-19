import numpy as np
import math as m

def sigma(lSigma: float, radius: float, rMin: float):
    # Calculate new connective field size from latent lSigma by transforming through a CDF scaling between rMin and radius
    return (radius - rMin) * normcdf(lSigma) + rMin

def weight(d: np.ndarray, lSigma: float, radius: float, rMin: float):
    # Gaussian weighting function
    return np.exp((-d**2) / (2 * sigma(lSigma=lSigma, radius=radius, rMin=rMin)**2))

def beta(lBeta: float):
    # Transform latent beta (log scale) into real effect size (exp).
    return np.exp(lBeta)

def normcdf(x: float, mu: float = 0.0, sigma: float = 1.0):
    # Normal cumulative distribution function 
    # x: value to evaluate
    # mu: mean of the distribution (default 0)
    # sigma: standard deviation (default 1)
    # Use the error function erf to compute the CDF
    return (1.0 + m.erf((x - mu) /(sigma* np.sqrt(2.0)))) / 2.0

def normpdf(x: float, mu: float = 0.0, sigma: float = 1.0):
    # Normal probability density function (PDF).
    return (np.exp(-((x - mu) / sigma) ** 2) / 2) / (sigma * np.sqrt(2 * np.pi))

def compute_mcmc_predictions(y: np.ndarray, tSeriesSource: np.ndarray, d: np.ndarray, lSigma: float, rMin: float, radius: float, betaBool: bool, lBeta: float):
    # Computes MCMC predictions for a given target time series y using weighted source time series, a connective field model, and either a fixed beta or an estimated one
    w = weight(d=d, lSigma=lSigma, radius=radius, rMin=rMin) # Compute weights based on distances and connective field size
    w = w / np.sum(w) # Normalize weights so they sum to 1
    w = w.astype(np.float32)
    # Weighted sum of source time series to obtain a predicted time series from the source area 
    predictions = np.dot(tSeriesSource, w)

    # The connective field parameters calcualtion if beta IS NOT fitted in the model
    if not betaBool:
        varBase = np.ones(len(y)) # Constant baseline term (intercept) for regression
        x = np.column_stack((predictions, varBase)) # Design matrix with predictions and baseline
        bHat = np.dot(np.linalg.pinv(x), y) # Least-squares solution for regression coefficients
        E = y - np.dot(x, bHat) # Get the residuals as difference between demeaned data and demeaned prediction
        varE = np.var(E) # Variance of residuals 
        xi = [sigma(lSigma=lSigma, radius=radius, rMin=rMin), bHat[0]] # Store the connective field size and beta
    # The connective field parameters calcualtion if beta IS fitted in the model
    else:  
        pTimeSeries1 = beta(lBeta=lBeta) * predictions # Scale predictions by effect size beta
        pTimeSeries1Demean = pTimeSeries1 - np.mean(pTimeSeries1) # Demean predictions to remove constant offset
        yDemean = y - np.mean(y) # Demean observed signal (remove constant offset)
        E = yDemean - pTimeSeries1Demean # Get the residuals as difference between demeaned data and demeaned prediction
        varE = np.var(E)  # Variance of residuals
        xi = np.array([sigma(lSigma=lSigma, radius=radius, rMin=rMin), beta(lBeta)]).T # Store the connective field size and beta

    # Compute the log-likelihood
    muHat, sigmaHat = np.mean(E), np.std(E) # Estimate mean and standard deviation of residuals
    estimatedLogLikelihood = np.log(normpdf(E, muHat, sigmaHat)) # Log probability of residuals with normal distribution
    logLikelihood = np.sum(estimatedLogLikelihood) # Total log-likelihood of data given parameters
    priorS = normpdf(lSigma, 0, 1) # Prior probability of lSigma with normal distribution

    if not betaBool:
        prior = np.log(priorS) # Log-prior for sigma with normal distribution
        postDist = np.sum(estimatedLogLikelihood) + prior # Posterior distribution
    else:
        priorB = normpdf(lBeta, -2, 5)  # Prior probability of beta with normal distribution
        prior = np.log(priorS) + np.log(priorB)  # Combined log-prior for sigma and beta
        postDist = np.sum(estimatedLogLikelihood) + prior  # Posterior distribution
    # Output with connective field size and beta, variance of the residuals, posterior, prior and log-likelihood
    return xi, varE, postDist, prior, logLikelihood