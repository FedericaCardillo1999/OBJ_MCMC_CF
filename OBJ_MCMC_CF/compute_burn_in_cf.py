import numpy as np

'''
Applies burn-in to the MCMC procedure. A first percentage of the data is omitted since convergence does not take place there yet.
This allows for more consistent results
'''
def compute_burn_in_cf(postDist: np.ndarray, loglikelihood: np.ndarray, priorDist: np.ndarray, posteriorLatent: np.ndarray, posterior: np.ndarray, ve: np.ndarray, burnIn: bool, percBurnIn: float):
    # Apply burn-in if burnIn = True
    if burnIn:
        # When the amount of iterations is too small, burn-in cannot be applied (all iterations would be omitted)
        if posteriorLatent.shape[1] < percBurnIn:
            burnIn = False
        # Omit the first percentage of iterations
        else:
            nBurn = burn(posteriorLatent, percBurnIn)
            postDistB = postDist[nBurn:]
            posteriorLatentB = posteriorLatent[:, nBurn:]
            posteriorB = posterior[:, nBurn:]
            loglikelihoodB = loglikelihood[nBurn:]
            priorDistB = priorDist[nBurn:]
            veB = 1 - ve[nBurn:] # Correct calculation for bayesian
            # veB = ve[nBurn:]    # Correct for standard 
            # Find best fit using maximum loglikelihood estimation
            ind = np.argmax(loglikelihoodB)
            bestFit = np.array([posteriorB[0, ind], posteriorB[1, ind], posteriorB[2, ind], ve[ind]])
    # When no burn-in is applied, keep all iterations
    if not burnIn:
        postDistB = postDist
        posteriorLatentB = posteriorLatent
        posteriorB = posterior
        loglikelihoodB = loglikelihood
        priorDistB = priorDist
        veB = 1 - ve

        # Find best fit using maximum loglikelihood estimation
        ind = np.argmax(loglikelihoodB)
        bestFit = np.array([posteriorB[0, ind], posteriorB[1, ind], posteriorB[2, ind], ve[ind]])

    return bestFit, postDistB, loglikelihoodB, priorDistB, posteriorLatentB, posteriorB, veB

def burn(posteriorLatent: np.ndarray, percBurnIn: float):
    nBurn = posteriorLatent.shape[1]//percBurnIn
    return nBurn