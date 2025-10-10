import numpy as np
import math as m

'''
Computes MCMC predictions
'''
def compute_mcmc_predictions(y: np.ndarray, tSeriesSource: np.ndarray, d: np.ndarray, lSigma: float, rMin: float, radius: float, betaBool: bool, lBeta: float):
    # Calculate a weight for each source voxel and normalize these weights
    w = weight(d=d, lSigma=lSigma, radius=radius, rMin=rMin)
    w = w/np.sum(w)
    w = w.astype(np.float32)

    # Apply the weight to the tSeriesSource
    predictions = np.dot(tSeriesSource, w)
    # Classical variance explained (same as standard evaluate_fit)
    #ss_total = np.sum(y ** 2) # TEST
    #ss_residual = np.sum((y - predictions) ** 2) # TEST
    #ve_classical = 1 - (ss_residual / ss_total) # TEST
    ##### I have emitted this code since it is not compatible with numba, betaBool can currently not be False #####
    if not betaBool:
        varBase = np.ones(len(y))
        # make design matrix: two columns [predictions, baseline]
        x = np.column_stack((predictions, varBase))   # shape (T, 2)
        # ordinary least squares fit
        bHat = np.dot(np.linalg.pinv(x), y)          # shape (2,)
        E = y - np.dot(x, bHat)                      # residuals
        varE = np.var(E)
        xi = np.array([sigma(lSigma=lSigma, radius=radius, rMin=rMin), bHat[0]])
    
    elif betaBool:
        # Multiply your weighted time series by the effect size beta and demean
        pTimeSeries1 = beta(lBeta=lBeta) * predictions
        pTimeSeries1Demean = pTimeSeries1 - np.mean(pTimeSeries1)
        yDemean = y - np.mean(y)
        #Calculate error and its variance
        E = yDemean - pTimeSeries1Demean
        varE = np.var(E)
        xi = np.array([sigma(lSigma=lSigma, radius=radius, rMin=rMin), beta(lBeta)]).T

    muHat, sigmaHat = np.mean(E), np.std(E)
    estimatedLogLikelihood = np.log(normpdf(E, muHat, sigmaHat))
    logLikelihood = np.sum(estimatedLogLikelihood)
    priorS = normpdf(lSigma, 0, 1)

    if not betaBool:
        prior = np.log(priorS)
        postDist = np.sum(estimatedLogLikelihood) + prior

    elif betaBool:
        priorB = normpdf(lBeta, -2, 5)
        prior = np.log(priorS) + np.log(priorB)
        postDist = np.sum(estimatedLogLikelihood) + prior

    return xi, varE, postDist, prior, logLikelihood
    #return xi, ve_classical, postDist, prior, logLikelihood # TEST

'''
Calculate new connective field size
'''
def sigma(lSigma: float, radius: float, rMin: float):
    return (radius-rMin)*normcdf(lSigma)+rMin

'''
Calculate weight
'''
def weight(d: np.ndarray, lSigma: float, radius: float, rMin: float):
    return np.exp((-d**2)/(2*sigma(lSigma=lSigma, radius=radius, rMin=rMin)**2))

'''
Calculate beta
'''
def beta(lBeta: float):
    beta = np.exp(lBeta)
    return beta

'''
Calculate normal cumulative distribution function (numba compatible)
'''
def normcdf(x: float, mu: float = 0.0, sigma: float = 1.0):
    return (1.0 + m.erf((x - mu) /(sigma* np.sqrt(2.0)))) / 2.0

'''
Calculate probability density function (numba compatible)
'''
def normpdf(x: float, mu: float = 0.0, sigma: float = 1.0):
    return (np.exp(-((x-mu)/(sigma))**2)/2)/(sigma*np.sqrt(2*np.pi))