import numpy as np
import math as m
from scipy.special import erf
import pandas as pd
import os


n_iter = 17500 # Number of iterations or how many times the MCMC algorithm will be executed  
TR =1.5 # Repetition Time 
rMin = 0.01 # How small is the allowable minimum radius for the CF modeling
radius = 10.5
betaBool = True # Is the Beta parameter being used?
burnIn = False # Is the burn-in executed?
percBurnIn=0.10 # What is the percentage of burn-in iterations?
accepted = np.zeros(n_iter) # Is the proposed MCMC chain accepted? 
pAccept = np.zeros(n_iter) # Probability of accepting a proposed move 
ve = np.zeros(n_iter) # VE of the iteration 
postDist = np.zeros(n_iter) # Post distribution of the iteration  
loglikelihood = np.zeros(n_iter) # Loglikelihood of the iteration 
priorDist = np.zeros(n_iter) 
posteriorLatent = np.zeros((2, n_iter)) # Latent variables of the posterior distribution (lSigma and lBeta)
posterior = np.zeros((3, n_iter)) # Posterior distribution of the main parameters (xi, ve, centersourceindex)
lSigma=1.0 
proposalWidth=2.0
lBeta=-5.0

class MCMCConnectiveField:
    # More will be added to this.
    def __init__(self, radius=10.5, r_min=0.01, beta_bool=True):
        self.radius = radius
        self.r_min = r_min
        self.beta_bool = beta_bool
    
    # Are these static methods necessary? 
    @staticmethod
    def normcdf(x: float, mu: float = 0.0, sigma: float = 1.0):
        return (1.0 + erf((x - mu) / (sigma * np.sqrt(2.0)))) / 2.0

    @staticmethod
    def normpdf(x: float, mu: float = 0.0, sigma: float = 1.0):
        return (np.exp(-((x - mu) / sigma) ** 2 / 2)) / (sigma * np.sqrt(2 * np.pi))
        # Why your prposed one is different than the previous code one: return (np.exp(-((x-mu)/(sigma)) ** 2) / 2)

    def propose_distance(self, l_step_size: float, max_step: float):
        return np.abs(max_step * self.normcdf(l_step_size) - max_step / 2)

    def propose_center(self, st_proposal: float, distances: np.ndarray, center_index: int):
        distances = distances[:, center_index]
        abs_distances = np.abs(distances - st_proposal)
        min_indices = np.where(abs_distances == np.min(abs_distances))[0]
        return np.random.choice(min_indices)

    def compute_sigma(self, l_sigma: float):
        return (self.radius - self.r_min) * self.normcdf(l_sigma) + self.r_min

    def compute_weight(self, distances: np.ndarray, l_sigma: float):
        sigma_val = self.compute_sigma(l_sigma)
        weights = np.exp(-distances ** 2 / (2 * sigma_val ** 2))
        return weights / np.sum(weights)
        # Why your prposed one is different than the previous code one: return np.exp((-d**2)/(2*sigma(lSigma=lSigma, radius=radius, rMin=rMin)**2))
        # Why here we cannot use the method def calculate_gaussian_weights? 

    def compute_beta(self, l_beta: float): 
        return np.exp(l_beta)

    def compute_mcmc_predictions(self, y: np.ndarray, source_matrix: np.ndarray, distances: np.ndarray, 
                                 l_sigma: float, l_beta: float):

        weights = self.compute_weight(distances, l_sigma).astype(np.float32)
        predictions = np.dot(source_matrix, weights)

        if self.beta_bool:
            p_time_series = self.compute_beta(l_beta) * predictions
            p_time_series_demean = p_time_series - np.mean(p_time_series)
            y_demean = y - np.mean(y) # Isnt this just a z scoring?
            error = y_demean - p_time_series_demean
            var_e = np.var(error)

            xi = np.array([self.compute_sigma(l_sigma), self.compute_beta(l_beta)]).T

            mu_hat, sigma_hat = np.mean(error), np.std(error)
            estimated_log_likelihood = np.log(self.normpdf(error, mu_hat, sigma_hat))
            log_likelihood = np.sum(estimated_log_likelihood)
            
            prior_s = self.normpdf(l_sigma, 0, 1)
            prior_b = self.normpdf(l_beta, -2, 5)
            prior = np.log(prior_s) + np.log(prior_b)
            post_dist = log_likelihood + prior

            return xi, var_e, post_dist, prior, log_likelihood

        return None

    def run_mcmc(self, y, source_matrix, distances_matrix, distance_df, initial_index, n_iter=1000, proposal_width=2.0,
                 l_sigma_init=1.0, l_beta_init=-5.0, vertex_index=None, save_dir=None):

        accepted = np.zeros(n_iter)
        p_accept = np.zeros(n_iter)
        ve_series = np.zeros(n_iter)
        post_dist_series = np.zeros(n_iter)
        loglikelihood_series = np.zeros(n_iter)
        prior_dist_series = np.zeros(n_iter)
        posterior_latent = np.zeros((2, n_iter))
        posterior = np.zeros((3, n_iter))

        center_index = initial_index
        l_sigma = l_sigma_init
        l_beta = l_beta_init

        for j in range(n_iter):
            l_sigma_prop = np.random.normal(l_sigma, proposal_width)
            l_beta_prop = np.random.normal(l_beta, proposal_width)
            l_step_size = np.random.normal(0, 1)
            max_step = np.max(distances_matrix[:, center_index])
            step_distance = self.propose_distance(l_step_size, max_step)
            center_index_prop = self.propose_center(step_distance, distances_matrix, center_index)

            d_current = distances_matrix[:, center_index]
            d_prop = distances_matrix[:, center_index_prop]

            result_current = self.compute_mcmc_predictions(y, source_matrix, d_current, l_sigma, l_beta)
            result_prop = self.compute_mcmc_predictions(y, source_matrix, d_prop, l_sigma_prop, l_beta_prop)

            if result_current is None or result_prop is None:
                continue

            xi_c, ve_c, post_c, prior_c, loglike_c = result_current
            xi_p, ve_p, post_p, prior_p, loglike_p = result_prop

            p_accept[j] = np.exp(post_p - post_c)
            accept = self.normcdf(np.random.normal())
            accepted[j] = (accept < p_accept[j])

            if accepted[j]:
                center_index = center_index_prop
                l_sigma = l_sigma_prop
                l_beta = l_beta_prop
                ve_series[j] = ve_p
                post_dist_series[j] = post_p
                loglikelihood_series[j] = loglike_p
                prior_dist_series[j] = prior_p
                posterior_latent[:, j] = [l_sigma, l_beta]
                posterior[:, j] = [xi_p[0], xi_p[1], center_index]
            else:
                ve_series[j] = ve_c
                post_dist_series[j] = post_c
                loglikelihood_series[j] = loglike_c
                prior_dist_series[j] = prior_c
                posterior_latent[:, j] = [l_sigma, l_beta]
                posterior[:, j] = [xi_c[0], xi_c[1], center_index]

        results = {
            'accepted': accepted,
            'p_accept': p_accept,
            've': ve_series,
            'post_dist': post_dist_series,
            'loglikelihood': loglikelihood_series,
            'prior_dist': prior_dist_series,
            'posterior_latent': posterior_latent,
            'posterior': posterior
        }

        # Save inside the class if requested
        # Save best fit summary (last sample in chain)
        if save_dir is not None and vertex_index is not None:
            # Create DataFrame with all iterations
            trace_df = pd.DataFrame({
                'Iteration': np.arange(posterior.shape[1]),
                'Sigma': posterior[0],
                'Beta': posterior[1],
                'SourceVertexIndex': [int(distance_df.columns[int(i)]) for i in posterior[2]],
                'VarianceExplained': ve_series
            })

            # Save full MCMC trace (one file per vertex)
            trace_path = os.path.join(save_dir, f"mcmc_trace_target_vertex_{vertex_index}.csv")
            trace_df.to_csv(trace_path, index=False)

            # Save best-fit summary (one row per vertex)

            # distances_matrixdistance_matrix = distances_class.geodesic_dists(hemi=hemi, subject=subj, vertices=idxSource, source=source_visual_area, output_dir=distance_matrix_path)
            best_fit = {
                'TargetVertexIndex': vertex_index,
                'BestSigma': posterior[0, -1],
                'BestBeta': posterior[1, -1],
                'BestSourceVertexIndex': int(distance_df.columns[int(posterior[2, -1])]),
                'BestVE': ve_series[-1]
            }
            best_fit_df = pd.DataFrame([best_fit])
            best_fit_path = os.path.join(save_dir, "best_fits.csv")
            best_fit_df.to_csv(best_fit_path, mode='a', index=False, header=not os.path.exists(best_fit_path))
        return results
