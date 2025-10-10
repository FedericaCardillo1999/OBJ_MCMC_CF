from computeMCMCPredictions import compute_mcmc_predictions
from compute_burn_in_cf import compute_burn_in_cf
import numpy as np
import math as m

def MCMC_CF_cluster_lloyd(idxSource: np.ndarray, distances: np.ndarray, tSeriesSource: np.ndarray, tSeriesTarget: np.ndarray):
    # Load variables and initiate arrays, could be done using a separate function as well
    n_iter = 20
    TR=1.5
    rMin= 0.05
    radius=10.5
    betaBool = True
    col=1
    burnIn=True
    percBurnIn=10
    accepted = np.zeros(n_iter)
    pAccept = np.zeros(n_iter)
    ve = np.zeros(n_iter)
    postDist = np.zeros(n_iter)
    loglikelihood = np.zeros(n_iter)
    priorDist = np.zeros(n_iter)
    posteriorLatent = np.zeros((2, n_iter))
    posterior = np.zeros((3, n_iter))
    lSigma=1.0
    proposalWidth=2.0
    lBeta=-5.0

    # Choose a random starting voxel
    centerSourceIndex = np.random.randint(len(idxSource))
    centerSource = idxSource[centerSourceIndex]
    
    n_voxels = tSeriesTarget.shape[1]
    bestFit = np.zeros((n_voxels, 4))
    
    # Create arrays dependent on if burn-in is applied or not
    if n_iter < percBurnIn:
        postDistB = np.zeros((n_voxels, n_iter))
        logLikelihoodB = np.zeros((n_voxels, n_iter))
        priorDistB = np.zeros((n_voxels, n_iter))
        posteriorLatentB = np.zeros((n_voxels, 2, n_iter))
        posteriorB = np.zeros((n_voxels, 3, n_iter))
        veB = np.zeros((n_voxels, n_iter))
    else:
        n_iterBurn = n_iter - n_iter//percBurnIn
        postDistB = np.zeros((n_voxels, n_iterBurn))
        logLikelihoodB = np.zeros((n_voxels, n_iterBurn))
        priorDistB = np.zeros((n_voxels, n_iterBurn))
        posteriorLatentB = np.zeros((n_voxels, 2, n_iterBurn))
        posteriorB = np.zeros((n_voxels, 3, n_iterBurn))
        veB = np.zeros((n_voxels, n_iterBurn))

    # Loop over all target voxels
    #for i in range(n_voxels):
    for i in range(min(5, n_voxels)):
        # Get time series of chosen voxel and normalize
        print("VoxelL {i}")
        y = tSeriesTarget[:, i]
        y = (y-np.mean(y))/(len(y)-(len(y)-1))*np.std(y)

        # Loop over all MC iterations
        for j in range(n_iter):
            if not betaBool:
                lSigmaProposal = np.random.normal(lSigma, proposalWidth)
                lStepsizeProposal = np.random.normal(0,1)
                maxStep = np.max(distances[:,centerSourceIndex])

                stProposal = dProp(lStepSize=lStepsizeProposal, maxStep=maxStep)
                centerProposalIndex = centerProp(stProposal=stProposal, distances=distances, centerSourceIndex=centerSourceIndex)
                distanceCurrent = distances[:, centerSourceIndex]
                distanceProposal = distances[:, centerProposalIndex]

                xiCurrent, veCurrent, postCurrent, priorCurrent, loglikeCurrent = compute_mcmc_predictions(y=y, tSeriesSource=tSeriesSource, d=distanceCurrent, lSigma=lSigma, rMin=rMin, radius=radius, betaBool=betaBool, lBeta=lBeta)
                xiProposal, veProposal, postProposal, priorProposal, loglikeProposal = compute_mcmc_predictions(y=y, tSeriesSource=tSeriesSource, d=distanceProposal, lSigma=lSigma, rMin=rMin, radius=radius, betaBool=betaBool, lBeta=lBeta)

                pAccept[j] = np.exp(postProposal - postCurrent)
                testValue = np.random.normal()
                accept = normcdf(testValue)
                accepted[j] = (accept < pAccept[j])

                if accepted[j]:
                    centerSourceIndex = centerProposalIndex
                    lSigma = lSigmaProposal
                    ve[j] = veProposal
                    postDist[j] = postProposal
                    loglikelihood[j] = loglikeProposal
                    priorDist[j] = priorProposal

                    posteriorLatent[:, j] = np.array([lSigmaProposal, lBeta])
                    posterior[:, j] = np.array([xiProposal[0], xiProposal[1], centerSourceIndex])
                
                elif not accepted[j]:
                    ve[j] = veCurrent
                    postDist[j] = postCurrent
                    loglikelihood[j] = loglikeCurrent
                    priorDist[j] = priorCurrent

                    posteriorLatent[:, j] = np.array([lSigma, lBeta])
                    posterior[:, j] = np.array([xiCurrent[0], xiCurrent[1], centerSourceIndex])

            elif betaBool:
                lBetaProposal = np.random.normal(lBeta, proposalWidth)
                lSigmaProposal = np.random.normal(lSigma, proposalWidth)
                lStepsizeProposal = np.random.normal(0,1)
                maxStep = np.max(distances[:,centerSourceIndex])

                stProposal = dProp(lStepSize=lStepsizeProposal, maxStep=maxStep)
                centerProposalIndex = centerProp(stProposal=stProposal, distances=distances, centerSourceIndex=centerSourceIndex)
                distanceCurrent = distances[:, centerSourceIndex]
                distanceProposal = distances[:, centerProposalIndex]

                xiCurrent, veCurrent, postCurrent, priorCurrent, loglikeCurrent = compute_mcmc_predictions(y=y, tSeriesSource=tSeriesSource, d=distanceCurrent, lSigma=lSigma, rMin=rMin, radius=radius, betaBool=betaBool, lBeta=lBeta)
                xiProposal, veProposal, postProposal, priorProposal, loglikeProposal = compute_mcmc_predictions(y=y, tSeriesSource=tSeriesSource, d=distanceProposal, lSigma=lSigmaProposal, rMin=rMin, radius=radius, betaBool=betaBool, lBeta=lBetaProposal)

                pAccept[j] = np.exp(postProposal - postCurrent)
                testValue = np.random.normal()
                accept = normcdf(testValue)
                accepted[j] = (accept < pAccept[j])

                if accepted[j]:
                    centerSourceIndex = centerProposalIndex
                    lSigma = lSigmaProposal
                    lBeta = lBetaProposal
                    ve[j] = veProposal
                    postDist[j] = postProposal
                    loglikelihood[j] = loglikeProposal
                    priorDist[j] = priorProposal

                    posteriorLatent[:, j] = np.array([lSigmaProposal, lBetaProposal])
                    posterior[:, j] = np.array([xiProposal[0], xiProposal[1], centerSourceIndex])
                
                elif not accepted[j]:
                    ve[j] = veCurrent
                    postDist[j] = postCurrent
                    loglikelihood[j] = loglikeCurrent
                    priorDist[j] = priorCurrent

                    posteriorLatent[:, j] = np.array([lSigma, lBeta])
                    posterior[:, j] = np.array([xiCurrent[0], xiCurrent[1], centerSourceIndex])
        bestFit[i, :], postDistB[i, :], logLikelihoodB[i, :], priorDistB[i, :], posteriorLatentB[i, :, :], posteriorB[i, :, :], veB[i, :] = compute_burn_in_cf(postDist, loglikelihood, priorDist, posteriorLatent, posterior, ve, burnIn, percBurnIn)

    return bestFit, postDistB, logLikelihoodB, priorDistB, posteriorB, posteriorLatentB

# Random start 
def MCMC_CF_cluster(idxSource: np.ndarray, distances: np.ndarray, tSeriesSource: np.ndarray,tSeriesTarget: np.ndarray):
# Start biased by the CF standard
# def MCMC_CF_cluster(idxSource: np.ndarray, distances: np.ndarray, tSeriesSource: np.ndarray,tSeriesTarget: np.ndarray, start_center: int = None):
    n_iter = 17500
    TR = 1.5
    rMin = 0.05
    radius = 10.5
    betaBool = True
    burnIn = True
    percBurnIn = 10
    accepted = np.zeros(n_iter)
    pAccept = np.zeros(n_iter)
    ve = np.zeros(n_iter)
    postDist = np.zeros(n_iter)
    loglikelihood = np.zeros(n_iter)
    priorDist = np.zeros(n_iter)
    posteriorLatent = np.zeros((2, n_iter))   # [lSigma, lBeta] each iter
    posterior = np.zeros((3, n_iter))         # [xi0, xi1, source_index] each iter (we will store GLOBAL index here)

    # initial state (in local index space)
    lSigma = 1.0
    proposalWidth = 2.0
    lBeta = -5.0

    #if start_center is not None:
    #    # map global vertex index to local index within idxSource
    #    if start_center in idxSource:
    #        centerSourceIndex = np.where(idxSource == start_center)[0][0]
    #        print("Start center biased ")
    #    else:
    #        raise ValueError(f"start_center {start_center} not in idxSource!")
    #else:
        # random initialization (old behavior)
        # centerSourceIndex = np.random.randint(len(idxSource))

    centerSourceIndex = np.random.randint(len(idxSource))  # local index 0..Nsrc-1
    n_voxels = tSeriesTarget.shape[1]
    # bestFit columns: [sigma, beta, source_index, ve]  (we’ll store GLOBAL index in col 2)
    bestFit = np.zeros((n_voxels, 4))

    # burn-in sizes
    if n_iter < percBurnIn:
        postDistB = np.zeros((n_voxels, n_iter))
        logLikelihoodB = np.zeros((n_voxels, n_iter))
        priorDistB = np.zeros((n_voxels, n_iter))
        posteriorLatentB = np.zeros((n_voxels, 2, n_iter))
        posteriorB = np.zeros((n_voxels, 3, n_iter))  # will hold GLOBAL indices in row 2
        veB = np.zeros((n_voxels, n_iter))
    else:
        n_iterBurn = n_iter - n_iter // percBurnIn
        postDistB = np.zeros((n_voxels, n_iterBurn))
        logLikelihoodB = np.zeros((n_voxels, n_iterBurn))
        priorDistB = np.zeros((n_voxels, n_iterBurn))
        posteriorLatentB = np.zeros((n_voxels, 2, n_iterBurn))
        posteriorB = np.zeros((n_voxels, 3, n_iterBurn))  # will hold GLOBAL indices in row 2
        veB = np.zeros((n_voxels, n_iterBurn))

    # ----------------- helper fns -----------------
    def normcdf(x: float, mu: float = 0.0, sigma: float = 1.0):
        return (1.0 + m.erf((x - mu) / (sigma * np.sqrt(2.0)))) / 2.0

    def dProp(lStepSize: float, maxStep: float):
        return np.abs(maxStep * normcdf(lStepSize) - maxStep / 2)

    def centerProp(stProposal: float, distances: np.ndarray, centerSourceIndex: int):
        dcol = distances[:, centerSourceIndex]
        absDistances = np.abs(dcol - stProposal)
        minimumIndex = np.where(absDistances == np.min(absDistances))
        centerProposalIndex = np.random.choice(minimumIndex[0], 1)
        return centerProposalIndex[0]  # local index

    # Main loop over targets
    tried_centers = np.zeros((n_voxels, n_iter), dtype=int)
    for i in range(min(5, n_voxels)):  # only 5 voxels for testing

        # true z-scoring (you previously multiplied by std by mistake)
        y = tSeriesTarget[:, i]
        #y = (y - np.mean(y)) / (np.std(y) + 1e-12)

        # --------- MCMC iterations ---------
        for j in range(n_iter):
            if not betaBool:
                lSigmaProposal = np.random.normal(lSigma, proposalWidth)
                lStepsizeProposal = np.random.normal(0, 1)
                maxStep = np.max(distances[:, centerSourceIndex])

                stProposal = dProp(lStepSize=lStepsizeProposal, maxStep=maxStep)
                centerProposalIndex = centerProp(stProposal=stProposal,
                                                 distances=distances,
                                                 centerSourceIndex=centerSourceIndex)

                distanceCurrent = distances[:, centerSourceIndex]
                distanceProposal = distances[:, centerProposalIndex]

                xiCurrent, veCurrent, postCurrent, priorCurrent, loglikeCurrent = compute_mcmc_predictions(
                    y=y, tSeriesSource=tSeriesSource, d=distanceCurrent, lSigma=lSigma,
                    rMin=rMin, radius=radius, betaBool=betaBool, lBeta=lBeta)
                xiProposal, veProposal, postProposal, priorProposal, loglikeProposal = compute_mcmc_predictions(
                    y=y, tSeriesSource=tSeriesSource, d=distanceProposal, lSigma=lSigmaProposal,
                    rMin=rMin, radius=radius, betaBool=betaBool, lBeta=lBeta)

                pAccept[j] = np.exp(postProposal - postCurrent)
                accept = normcdf(np.random.normal())
                accepted[j] = (accept < pAccept[j])

                if accepted[j]:
                    centerSourceIndex = centerProposalIndex
                    lSigma = lSigmaProposal
                    ve[j] = veProposal
                    postDist[j] = postProposal
                    loglikelihood[j] = loglikeProposal
                    priorDist[j] = priorProposal

                    # STORE GLOBAL SOURCE INDEX HERE
                    posteriorLatent[:, j] = np.array([lSigma, lBeta])
                    posterior[:, j] = np.array([xiProposal[0], xiProposal[1], idxSource[centerSourceIndex]])
                else:
                    ve[j] = veCurrent
                    postDist[j] = postCurrent
                    loglikelihood[j] = loglikeCurrent
                    priorDist[j] = priorCurrent

                    posteriorLatent[:, j] = np.array([lSigma, lBeta])
                    posterior[:, j] = np.array([xiCurrent[0], xiCurrent[1], idxSource[centerSourceIndex]])

            else:
                lBetaProposal = np.random.normal(lBeta, proposalWidth)
                lSigmaProposal = np.random.normal(lSigma, proposalWidth)
                lStepsizeProposal = np.random.normal(0, 1)
                maxStep = np.max(distances[:, centerSourceIndex])

                stProposal = dProp(lStepSize=lStepsizeProposal, maxStep=maxStep)
                centerProposalIndex = centerProp(stProposal=stProposal,
                                                 distances=distances,
                                                 centerSourceIndex=centerSourceIndex)

                distanceCurrent = distances[:, centerSourceIndex]
                distanceProposal = distances[:, centerProposalIndex]

                xiCurrent, veCurrent, postCurrent, priorCurrent, loglikeCurrent = compute_mcmc_predictions(
                    y=y, tSeriesSource=tSeriesSource, d=distanceCurrent, lSigma=lSigma,
                    rMin=rMin, radius=radius, betaBool=betaBool, lBeta=lBeta)
                xiProposal, veProposal, postProposal, priorProposal, loglikeProposal = compute_mcmc_predictions(
                    y=y, tSeriesSource=tSeriesSource, d=distanceProposal, lSigma=lSigmaProposal,
                    rMin=rMin, radius=radius, betaBool=betaBool, lBeta=lBetaProposal)

                pAccept[j] = np.exp(postProposal - postCurrent)
                accept = normcdf(np.random.normal())
                accepted[j] = (accept < pAccept[j])

                if accepted[j]:
                    centerSourceIndex = centerProposalIndex
                    lSigma = lSigmaProposal
                    lBeta = lBetaProposal
                    ve[j] = veProposal
                    postDist[j] = postProposal
                    loglikelihood[j] = loglikeProposal
                    priorDist[j] = priorProposal

                    # STORE GLOBAL SOURCE INDEX
                    posteriorLatent[:, j] = np.array([lSigma, lBeta])
                    posterior[:, j] = np.array([xiProposal[0], xiProposal[1], idxSource[centerSourceIndex]])
                else:
                    ve[j] = veCurrent
                    postDist[j] = postCurrent
                    loglikelihood[j] = loglikeCurrent
                    priorDist[j] = priorCurrent

                    # STORE GLOBAL SOURCE INDEX
                    posteriorLatent[:, j] = np.array([lSigma, lBeta])
                    posterior[:, j] = np.array([xiCurrent[0], xiCurrent[1], idxSource[centerSourceIndex]])

        # burn-in selection
        bf_row, postB, loglikB, priorB, postLatB, postFullB, veBurn = compute_burn_in_cf(
            postDist, loglikelihood, priorDist, posteriorLatent, posterior, ve, burnIn, percBurnIn)

        # bf_row has [sigma, beta, source_index, ve] with source_index CURRENTLY GLOBAL?:
        # If your compute_burn_in_cf returns the same 'posterior' we fed (with global ids),
        # bf_row[2] is already global. If not, enforce mapping here:
        # If it were local: bf_row[2] = idxSource[int(bf_row[2])]

        bestFit[i, :] = bf_row

        # ensure posteriorB (returned) has GLOBAL source indices as well
        # Our postFullB is the post-burn-in posterior with row 2 = already global (because we stored global in 'posterior').
        posteriorB[i, :, :] = postFullB
        postDistB[i, :] = postB
        logLikelihoodB[i, :] = loglikB
        priorDistB[i, :] = priorB
        posteriorLatentB[i, :, :] = postLatB
        veB[i, :] = veBurn

    return bestFit, postDistB, logLikelihoodB, priorDistB, posteriorB, posteriorLatentB

def dProp(lStepSize: float, maxStep: float):
    return np.abs(maxStep*normcdf(lStepSize)-maxStep/2)

def centerProp(stProposal: float, distances: np.ndarray, centerSourceIndex: int):
    distances = distances[:, centerSourceIndex]
    absDistances = np.abs(distances - stProposal)
    minimumIndex = np.where(absDistances == np.min(absDistances))
    centerProposalIndex = np.random.choice(minimumIndex[0], 1)
    return centerProposalIndex[0]

def normcdf(x: float, mu: float = 0.0, sigma: float = 1.0):
    return (1.0 + m.erf((x - mu) /(sigma* np.sqrt(2.0)))) / 2.0