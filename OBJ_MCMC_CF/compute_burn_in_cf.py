import numpy as np


def compute_burn_in_cf(postDist: np.ndarray, loglikelihood: np.ndarray, priorDist: np.ndarray, posteriorLatent: np.ndarray, posterior: np.ndarray, ve: np.ndarray, burnIn: bool,percBurnIn: float):
    # Throw away the first part of the MCMC chain because those early samples are biased by the random initialization and don’t represent the true target distribution
    # Apply the burn if specified in the MCMC file
    if burnIn:
        # Make a quick test because if the iterations are shorter than the burn-in percentage the burn-in would throw away everything, so we skip it
        if posteriorLatent.shape[1] < percBurnIn:
            burnIn = False
        # Otherwise, apply it and remove the first percetage of iterations 
        else:
            nBurn = int(np.ceil(posteriorLatent.shape[1] * percBurnIn / 100.0)) # Get the burn in percentage
            # Slice arrays from to contain only the values after the burn in 
            postDistB = postDist[nBurn:] # Posterior distribution  
            posteriorLatentB = posteriorLatent[:, nBurn:] # Latent parameters of lSigma and lBeta
            posteriorB = posterior[:, nBurn:] # Main connective field parameters so the connective field coordinates and center index
            loglikelihoodB = loglikelihood[nBurn:] # Log-likelihood values  
            priorDistB = priorDist[nBurn:] # Prior distribution
            veB = 1 - ve[nBurn:] # From the variance of the residuals to the variance explained
            ind = np.argmax(loglikelihoodB) # Pick the iteration with the maximum log-likelihood 
            bestFit = np.array([posteriorB[0, ind], posteriorB[1, ind], posteriorB[2, ind], veB[ind]]) # Connective field size, beta, center index and variance explained

    # If burn in is not applied
    if not burnIn:
        # Keep the arrays to include the entire iterations
        postDistB = postDist # Posterior distribution  
        posteriorLatentB = posteriorLatent # Latent parameters of lSigma and lBeta
        posteriorB = posterior # Main connective field parameters so the connective field coordinates an
        loglikelihoodB = loglikelihood # Log-likelihood values  
        priorDistB = priorDist # Prior distribution
        veB = 1 - ve # From the variance of the residuals to the variance explained
        ind = np.argmax(loglikelihoodB) # Pick the iteration with the maximum log-likelihood 
        bestFit = np.array([posteriorB[0, ind], posteriorB[1, ind], posteriorB[2, ind], veB[ind]]) # Connective field size, beta, center index and variance explained

    # Output 
    return bestFit, postDistB, loglikelihoodB, priorDistB, posteriorLatentB, posteriorB, veB