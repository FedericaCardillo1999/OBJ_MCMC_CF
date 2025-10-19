from computeMCMCPredictions import compute_mcmc_predictions
from compute_burn_in_cf import compute_burn_in_cf
import numpy as np
import math as m

def dProp(lStepSize: float, maxStep: float):
    # Compute a "distance proposal" based on a cumulative normal distribution
    # maxStep: maximum scaling factor
    # normcdf(lStepSize) gives a probability (0–1), multiply by maxStep → scale
    # Subtract maxStep/2 to center it, then take absolute value
    return np.abs(maxStep*normcdf(lStepSize)-maxStep/2)

def centerProp(stProposal: float, distances: np.ndarray, centerSourceIndex: int):
    # Pick the index of the distance that is closest to the proposal value
    # distances: 2D array of distances (matrix)
    # centerSourceIndex: which column in the matrix to compare against
    distances = distances[:, centerSourceIndex] # take only that column
    absDistances = np.abs(distances - stProposal)  # difference between each value and the proposal
    minimumIndex = np.where(absDistances == np.min(absDistances)) # indices of the closest values
    centerProposalIndex = np.random.choice(minimumIndex[0], 1) # randomly choose one if there are ties
    return centerProposalIndex[0] # return the index (not array)

def normcdf(x: float, mu: float = 0.0, sigma: float = 1.0):
    # Normal cumulative distribution function (CDF)
    # x: value to evaluate
    # mu: mean of the distribution (default 0)
    # sigma: standard deviation (default 1)
    # Use the error function erf to compute the CDF
    return (1.0 + m.erf((x - mu) /(sigma* np.sqrt(2.0)))) / 2.0

def MCMC_CF_cluster(idxSource: np.ndarray, distances: np.ndarray, tSeriesSource: np.ndarray,tSeriesTarget: np.ndarray):
    # Run the Markov Chain Monte Carlo (MCMC) approach on top of the connective field modeling
    # idxSource are the indices of vertices in the source are
    # distances is a matrix of the vertices in the source are (2D: n_vertices x n_vertices)
    # tSeriesSource is the time series of source vertices (2D: time x n_target)
    # tSeriesTarget is the time series of target vertices (2D: time x n_target)

    # The settings for the MCMC approach 
    n_iter = 17500 # The number of MCMC iterations 
    TR = 1.5 # The repetition time of the acquisition
    rMin = 0.05 # The minimum allowed connective field size 
    radius = 10.5 # The maximum allowed connective field size 
    # Beta is a scaling parameter which fits the amplitude of the time course to find the best connective field parameters 
    # You can choose to use it to find the best connective field parameters with "True", or to not use it with "False"
    betaBool = False 
    # Decide whether to discard the initial part of the MCMC chain asthe early steps are often biased by the random start and do not yet represent the true target distribution
    burnIn = False # "True" to discard and "False" to keep
    percBurnIn = 10 # The percentage of the initial iterations to be discarded if the burn in is applied

    # Initialize the arrays for the main outputs of the MCMC
    accepted = np.zeros(n_iter) # Stores whether each proposed connective field parameter was accepted or rejected
    pAccept = np.zeros(n_iter) # Stores the acceptance probability per each iteration
    ve = np.zeros(n_iter) # Stores the variance explained of the connective field parameter quality 
    postDist = np.zeros(n_iter) # Stores the posterior distribution 
    loglikelihood = np.zeros(n_iter) # Stores the log-likelihood of the data given parameters
    priorDist = np.zeros(n_iter) # Stores the prior distribution value
    posteriorLatent = np.zeros((2, n_iter)) # Stores the latent parameters per iteration: lSigma and lBeta
    posterior = np.zeros((3, n_iter)) # Stores the main parameters: xi0 and xi1 (the coordinates of the connective field center), source_index (the index of the connective field center)
    
    # Define the initial state of the parameters 
    lSigma = 1.0 # Initial size proposed for the connective field         
    proposalWidth = 2 # Initial step size for the proposal next connective fields      
    lBeta = -5.0 # Initial beta value 

    # Pick random source vertex in the source area as the start for the iteration
    centerSourceIndex = np.random.randint(len(idxSource))
    # Calculate the total number of vertices in the target area that we need to find the connective field for
    n_vertices = tSeriesTarget.shape[1]   
    # Initialize the results array for each vertex in the target area which will store
    # the size (or sigma), the beta, the center vertex, and the variance explained for the connective field chosen
    bestFit = np.zeros((n_vertices, 4))

    # Correct the arrays considering the burn-in application
    # When the burn in is applied compute the number of discarded iterations as a percentage 
    if burnIn:
        n_burn = int(np.ceil(n_iter * percBurnIn / 100.0))
        n_post = n_iter - n_burn
    # if the burn in is not applied, consider the full chain of iterations 
    else:
        n_post = n_iter

    # Apply the correct burn-in lenght to the variables
    postDistB = np.zeros((n_vertices, n_post)) # Posterior distribution per target vertex
    logLikelihoodB = np.zeros((n_vertices, n_post)) # Log-likelihood per target vertex
    priorDistB = np.zeros((n_vertices, n_post)) # Prior distribution per target vertex
    posteriorLatentB = np.zeros((n_vertices, 2, n_post)) # lSigma and lBeta per target vertex
    posteriorB = np.zeros((n_vertices, 3, n_post))  # xi0 and xi1 (the coordinates of the connective field center) and source_index (the index of the connective field center) per target vertex
    veB = np.zeros((n_vertices, n_post)) # Variance explained per target vertex


    # Loop over all the vertices in the target area 
    for i in range(n_vertices):   
        # Get the time series of the current target voxel
        y = tSeriesTarget[:, i]
        # QUESTION: 
        # Azzurra originally z scored the target vertices time course while keeping the source vertices time course raw. Why? What is the correct approach?
        # y = (y - np.mean(y)) / (np.std(y) + 1e-12)

        # Start the MCMC iterations 
        for j in range(n_iter):
            # The connective field parameters calcualtion if beta IS NOT fitted in the model
            if not betaBool:
                # Propose a new random value for lSigma from a normal distribution 
                lSigmaProposal = np.random.normal(lSigma, proposalWidth)
                # Use a random step size proposal obtained from a random normal distribution to move the source center
                lStepsizeProposal = np.random.normal(0, 1)
                # Set the maximum distance from the current source center
                maxStep = np.max(distances[:, centerSourceIndex])
                # Propose a new step size limited by the value of maxStep
                stProposal = dProp(lStepSize=lStepsizeProposal, maxStep=maxStep)
                # Propose a new source center index based on the step
                centerProposalIndex = centerProp(stProposal=stProposal,distances=distances, centerSourceIndex=centerSourceIndex)
                # Calculate the distances from the current and proposed centers to all vertices
                distanceCurrent = distances[:, centerSourceIndex]
                distanceProposal = distances[:, centerProposalIndex]
                
                # Compute predictions, priors, likelihoods, and posteriors for the current state
                xiCurrent, veCurrent, postCurrent, priorCurrent, loglikeCurrent = compute_mcmc_predictions(y=y, tSeriesSource=tSeriesSource, d=distanceCurrent, lSigma=lSigma, rMin=rMin, radius=radius, betaBool=betaBool, lBeta=lBeta)
                # Compute predictions, priors, likelihoods, and posteriors for the proposed state
                xiProposal, veProposal, postProposal, priorProposal, loglikeProposal = compute_mcmc_predictions( y=y, tSeriesSource=tSeriesSource, d=distanceProposal, lSigma=lSigmaProposal, rMin=rMin, radius=radius, betaBool=betaBool, lBeta=lBeta)

                # Metropolis-Hastings acceptance probability 
                pAccept[j] = np.exp(postProposal - postCurrent) # Compute acceptance ratio
                accept = normcdf(np.random.normal()) # Generate a random threshold for acceptance
                accepted[j] = (accept < pAccept[j]) # Decide whether to accept the proposal: we accept if the random draw is less than the acceptance probability

                if accepted[j]: 
                    # If the proposal is accepted, update the parameters 
                    centerSourceIndex = centerProposalIndex # Update the "center" of the source to the newly proposed one
                    lSigma = lSigmaProposal # Update the connective field size value
                    ve[j] = veProposal # Save variance explained from the proposed state
                    postDist[j] = postProposal # Save the posterior probability
                    loglikelihood[j] = loglikeProposal # Save the log-likelihood
                    priorDist[j] = priorProposal # Save the prior probability
                    posteriorLatent[:, j] = np.array([lSigma, lBeta]) # Save the latent variables
                    posterior[:, j] = np.array([xiProposal[0], xiProposal[1], idxSource[centerSourceIndex]]) # Save the connective field parameters
                else:
                    # If the proposal is rejected, keep current parameters
                    ve[j] = veCurrent
                    postDist[j] = postCurrent
                    loglikelihood[j] = loglikeCurrent
                    priorDist[j] = priorCurrent
                    posteriorLatent[:, j] = np.array([lSigma, lBeta])
                    posterior[:, j] = np.array([xiCurrent[0], xiCurrent[1], idxSource[centerSourceIndex]])
            
            # The connective field parameters calcualtion if beta IS fitted in the model
            else:
                # Propose a new random value for lBeta from a normal distribution 
                lBetaProposal = np.random.normal(lBeta, proposalWidth) 
                # Propose a new random value for lSigma from a normal distribution 
                lSigmaProposal = np.random.normal(lSigma, proposalWidth)
                # Use a random step size proposal obtained from a random normal distribution to move the source center
                lStepsizeProposal = np.random.normal(0, 1)
                # Set the maximum distance from the current source center
                maxStep = np.max(distances[:, centerSourceIndex])
                # Propose a new step size limited by the value of maxStep
                stProposal = dProp(lStepSize=lStepsizeProposal, maxStep=maxStep)
                # Propose a new source center index based on the step
                centerProposalIndex = centerProp(stProposal=stProposal, distances=distances, centerSourceIndex=centerSourceIndex)
                # Calculate the distances from the current and proposed centers to all vertices
                distanceCurrent = distances[:, centerSourceIndex]
                distanceProposal = distances[:, centerProposalIndex]
                
                # Compute predictions, priors, likelihoods, and posteriors for the current state
                xiCurrent, veCurrent, postCurrent, priorCurrent, loglikeCurrent = compute_mcmc_predictions( y=y, tSeriesSource=tSeriesSource, d=distanceCurrent, lSigma=lSigma, rMin=rMin, radius=radius, betaBool=betaBool, lBeta=lBeta)
                # Compute predictions, priors, likelihoods, and posteriors for the proposed state
                xiProposal, veProposal, postProposal, priorProposal, loglikeProposal = compute_mcmc_predictions( y=y, tSeriesSource=tSeriesSource, d=distanceProposal, lSigma=lSigmaProposal, rMin=rMin, radius=radius, betaBool=betaBool, lBeta=lBetaProposal)

                # Metropolis-Hastings acceptance probability 
                pAccept[j] = np.exp(postProposal - postCurrent) # Compute acceptance ratio
                accept = normcdf(np.random.normal()) # Generate a random threshold for acceptance
                accepted[j] = (accept < pAccept[j]) # Decide whether to accept the proposal: we accept if the random draw is less than the acceptance probability

                if accepted[j]:
                    # If the proposal is accepted, update the parameters 
                    centerSourceIndex = centerProposalIndex # Update the "center" of the source to the newly proposed one
                    lSigma = lSigmaProposal # Update the connective field size value
                    lBeta = lBetaProposal # Save the connective field size beta value
                    ve[j] = veProposal # Save variance explained from the proposed state
                    postDist[j] = postProposal # Save the posterior probability
                    loglikelihood[j] = loglikeProposal # Save the posterior probability
                    priorDist[j] = priorProposal # Save the prior probability
                    posteriorLatent[:, j] = np.array([lSigma, lBeta]) # Save the latent variables
                    posterior[:, j] = np.array([xiProposal[0], xiProposal[1], idxSource[centerSourceIndex]]) # Save the connective field parameters
                else:
                    # If the proposal is rejected, keep current parameters
                    ve[j] = veCurrent
                    postDist[j] = postCurrent
                    loglikelihood[j] = loglikeCurrent
                    priorDist[j] = priorCurrent
                    posteriorLatent[:, j] = np.array([lSigma, lBeta])
                    posterior[:, j] = np.array([xiCurrent[0], xiCurrent[1], idxSource[centerSourceIndex]])
    
        # Apply a the burn in to discard the first part of the MCMC iterations  
        # bf_row is the best fit row containing the parameters after burn in of the connective field size, beta, the center index, and the variance explained 
        bf_row, postB, loglikB, priorB, postLatB, postFullB, veBurn = compute_burn_in_cf(postDist, loglikelihood, priorDist, posteriorLatent, posterior, ve, burnIn, percBurnIn)
        bestFit[i, :] = bf_row   # Save best-fit parameters for target vertex 
        # Store the variables after then burn in application
        posteriorB[i, :, :] = postFullB # Posterior distriubtion 
        postDistB[i, :] = postB 
        logLikelihoodB[i, :] = loglikB # Log-likelihood values
        priorDistB[i, :] = priorB # Prior distribution
        posteriorLatentB[i, :, :] = postLatB # Latent parameters lSigma and lBeta
        veB[i, :] = veBurn # Variance explained 

    # Prepare your output 
    # If the code is run for only one single vertex squeeze the dimensions of the best fit. 
    # e.g. bestFit.shape = (1,4) → becomes (4,) 
    def squeeze_first_axis_if_single(x):
        if hasattr(x, "shape") and x.shape[0] == 1:
            return np.squeeze(x, axis=0)
        return x
    bestFit_out = squeeze_first_axis_if_single(bestFit) # This stores the size, beta, center index, and the variance explained      
    postDistB_out = squeeze_first_axis_if_single(postDistB) # Posterior distribution values      
    logLikelihoodB_out = squeeze_first_axis_if_single(logLikelihoodB) # Log-likelihood values 
    priorDistB_out = squeeze_first_axis_if_single(priorDistB) # Prior distribution values
    posteriorB_out = squeeze_first_axis_if_single(posteriorB) # Posterior parameters: [xi0, xi1, source_index]
    posteriorLatentB_out = squeeze_first_axis_if_single(posteriorLatentB) # Latent parameters lSigma and lBeta
    veB_out = squeeze_first_axis_if_single(veB) # Variance explained                

    # Outputs
    return (bestFit_out,  postDistB_out,  logLikelihoodB_out,  priorDistB_out,  posteriorB_out,  posteriorLatentB_out,  veB_out)