# %% [markdown]
# ### **Connective Field Modeling: Object-Oriented Programming Version**
# 
# Connective Field Modeling is a computational technique used to characterize the relationship between neuronal populations across different regions of the brain. It models how sensory inputs, represented in one visual area, are transformed and projected to another visual area.
# 
# ---
# 
# #### **Connective Field Modeling Parameters**
# 
# 1. **<span style="color: black;">Sigma</span>**  
#    <small>- The spread or size of the connective field.</small>  
#    <small>- Represents the spatial extent of influence from the source region.</small>
# 
# 2. **<span style="color: black;">Eccentricity</span>**  
#    <small>- The radial distance of the center of the connective field from the origin of the visual field representation.</small>
# 
# 3. **<span style="color: black;">Polar Angle</span>**  
#    <small>- The angular position of the connective field in visual space.</small>
# 
# 4. **<span style="color: black;">Variance Explained</span>**  
#    <small>- A measure of how well the modeled time series fits the observed data.</small>  
#    <small>- Indicates the quality of the connective field fit for each voxel.</small>
# 
# 5. **<span style="color: black;">Predicted Time Series</span>**  
#    <small>- The estimated BOLD signal for each voxel in the target area.</small>  
#    <small>- Derived from the best-fit connective field model.</small>
# 
# ---
# 
# #### **Process for Obtaining Connective Field Parameters**
# 
# 1. **<span style="color: black;">Define Source and Target Areas</span>**  
#    <small>- Extract vertices or voxels belonging to these areas.</small>  
#    <small>- Use label files or predefined masks to identify regions of interest.</small>
# 
# 2. **<span style="color: black;">Compute Geodesic Distances</span>**  
#    <small>- Compute the true distances on the cortical surface between the vertices in the source area.</small>  
# 
# 3. **<span style="color: black;">Random Initialization</span>**  
#    <small>- Choose an initial random vertex from the source area as a starting point for the connective field center. </small>
#    <small>-Set initial parameters to random or default values.</small>
# 
# 4. **<span style="color: black;">Iterative Optimization</span>**  
#    <small>- For each voxel in the target area define a Gaussian function centered at the current connective filed locatin in the source area. </small>
#    <small>- Predict the BOLD signal for the target voxel by combining the source time series with the spatial weighting function. </small>
#    <small>- Adjust parameters to maximize the fit using a least-squares or gradient-based optimization. </small>
# 
# 5. **<span style="color: black;">Evaluate Model Fit</span>**  
#    <small>- Calculate the variance explained (R²) for the modeled time series compared to the observed time series.</small>  
#    <small>- Keep the parameters that provide the best fit for each voxel.</small> 
# 
import sys
subj=f'sub-{sys.argv[1:][0]}'


# %% [markdown]
# NEXT STEP
# 1. Finer grid search on the sigma values.
# 2. MCMC implementation.  

# %%
# Export the required libraries
import os
import time 
import math as m 
import pandas as pd
import random 
import cortex
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from numba import jit
from scipy.optimize import minimize
import itertools
from vertex import Vertex
from joblib import Parallel, delayed
from CFandPRF import load_prf, filter_prf, PRFModel, source_eccentricity_two

# %%
def surfs(subject: str, hemi:str):
    """
    Load the cortical surface for a given subject and hemisphere.
    Specifies whether the surface is from the left ("lh") or right ("rh") hemisphere.
    Returns the cortical surface object for the specified hemisphere.
    """
    if hemi == "lh":
        surf_data = cortex.db.get_surf(subject, "fiducial")[0]  # Left hemisphere
    elif hemi == "rh":
        surf_data = cortex.db.get_surf(subject, "fiducial")[1]  # Right hemisphere
        
    surface = cortex.polyutils.Surface(*surf_data)
    return surface

# %%
class Distances(Vertex):
    """ 
    The Distances class computes the geodesic distance matrix for a set of vertices,
    saving it as a CSV file for later use, and provides basic inspection of the results.
    """

    def __init__(self, subject, hemi, matrix_dir, csv_path):
        self.subject = subject
        self.hemi = hemi
        self.matrix_dir = matrix_dir
        self.csv_path = csv_path
    
    def geodesic_dists(self, hemi, subject, vertices, source, output_dir):
        """
        Compute geodesic distances between source vertices and save the result to a CSV file.
        """
        # Extract source vertex indices
        source_verts = np.array([v.index for v in vertices])
        
        # Determine the output file path based on hemisphere and source
        output_path = f"{output_dir}/{subject}_distance_{hemi}_{source}.csv"

        # Try loading the distance matrix from a CSV file
        if os.path.exists(output_path):
            try:
                distance_matrix = pd.read_csv(output_path, index_col=0).values
                print(f"Loaded distance matrix with shape: {distance_matrix.shape}")
                return distance_matrix
            except Exception as e:
                print("Computing the geodesic distance matrix...")
        
        # Load the cortical surface for the given hemisphere
        surface = surfs(subject, hemi)
        
        # Initialize the distance matrix
        dists_source = np.zeros((len(source_verts), len(source_verts)), dtype=np.float32)
        
        for i in range(len(source_verts)):
            dists = surface.geodesic_distance(source_verts[i])  
            for j in range(len(source_verts)):
                dists_source[i, j] = dists[source_verts[j]]  
        
        # Convert the distance matrix to a DataFrame for saving as CSV
        distance_df = pd.DataFrame(dists_source, index=source_verts, columns=source_verts)
        distance_df.to_csv(output_path)
        
        # Print shape and first 4 rows and columns for verification
        print(f"Distance matrix saved with shape: {distance_df.shape}")

        # Return the computed distance matrix
        return dists_source

# %%
class TimeCourse:
    """ 
    Loading, processing, and analyzing time course data for single or multiple vertices.
    """

    def __init__(self, time_course_file: str, vertices: list[Vertex], cutoff_volumes: int):
        self.vertices = vertices  # List of Vertex objects
        self.cutoff_volumes = cutoff_volumes
        self.data = np.load(time_course_file)  # Load time course data
        self.tSeries = self.load_time_courses()

    def load_time_courses(self) -> dict:
        duration = self.data.shape[0]
        tSeries = {}
        # Iterates over the self.vertices list, accessing the index of each vertex.
        for vertex in self.vertices:
            index = vertex.index
            # Extracts the time course for each vertex
            time_course = self.data[self.cutoff_volumes:duration, index]
            # Stores the time course in a dictionary using the vertex index as the key.
            tSeries[index] = time_course

        return tSeries

    # def z_score(self) -> dict:
        # PREVIOUS
        # Performs z-scoring (standardization) of the time course data for each vertex.
        #z_scored_data = {}
        # Computes the z-score for each time course
        #for index, time_course in self.tSeries.items():
        #    # Subtracts the mean and divides by the standard deviation.
        #    # z_scored_data[index] = (time_course - np.mean(time_course)) / np.std(time_course) 
        #    z_scored_data[index] = (time_course - np.nanmean(time_course)) / np.nanstd(time_course)
        #return z_scored_data

        # UPDATED

    def z_score(self, method: str = "zscore") -> dict:
        # zscore to standardize to mean=0, std=1
        # demean to subtract mean 
        # none to keep the raw time series 
        processed_data = {}

        for index, time_course in self.tSeries.items():
            if method == "zscore":
                processed = (time_course - np.nanmean(time_course)) / np.nanstd(time_course)
            elif method == "demean":
                processed = time_course - np.nanmean(time_course)
            elif method == "none":
                processed = time_course
            processed_data[index] = processed

        return processed_data
    
    def plot_time_series(self, vertex_index: int, show: bool = True) -> None:
        if vertex_index not in self.tSeries:
            print(f"Vertex {vertex_index} not found in the time series data.")
            return

        time_course = self.tSeries[vertex_index]
        plt.figure(figsize=(10, 5))
        plt.plot(time_course, label=f'Vertex Index: {vertex_index}', color='blue')
        plt.title(f'Time Series for Vertex {vertex_index}')
        plt.xlabel('Time (Volumes) after Cutoff')
        plt.ylabel('BOLD Signal')
        plt.legend()
        plt.grid()
        if show:
            plt.show()
        
    def plot_comparison(self, z_scored_data: dict, vertex_index: int, title_prefix: str, show: bool = True) -> None:
        """
        Plot the original and z-scored time series for a specific vertex.
        """
        original_time_course = self.tSeries[vertex_index]
        z_scored_time_course = z_scored_data[vertex_index]
        plt.figure(figsize=(12, 6))
        plt.plot(z_scored_time_course, label="Z-Scored Time Series", linestyle="--", marker="x", alpha=0.7)
        plt.plot(original_time_course, label="Original Time Series", linestyle="-", marker="o", alpha=0.7)
        plt.title(f"{title_prefix} Vertex {vertex_index} - Before and After Z-Scoring")
        plt.xlabel("Time Points")
        plt.ylabel("BOLD Signal")
        plt.legend()
        plt.grid()
        if show:
            plt.show()

# %%
class ConnectiveField:
    """Connective Field class to calculate sigma, eccentricity, variance explained, polar angle, and predicted time course for a voxel."""

    def __init__(self, center_vertex: Vertex, vertex: Vertex):
        """
        Initialize the ConnectiveField class with a specific vertex.
        """
        self.vertex = vertex  # Use the vertex passed during initialization
        self.center_vertex = center_vertex  # Center of the Gaussian
        self.sigma = None  # Spread of the connective field
        self.eccentricity = None  # Distance from center (eccentricity)
        self.polar_angle = None  # Angle to indicate direction
        self.variance_explained = None  # Fit metric for model evaluation
        self.predicted_time_course = None  # Predicted BOLD signal time series
        self.observed_time_series = None  # Observed time series for the voxel
        self.best_fit = None  # Stores best optimization fit
        self.gaussian_weights = None #### Used only to plot the gaussian on the surface. Can be an alterantive 


    # Select a Vertex in the Target Area
    def select_target_vertex(self, idxTarget: list[Vertex], index: int = None) -> Vertex:
        if index is not None:
            selected_vertex_target = idxTarget[index]
            print(f"Selected Target Vertex by Index: Index = {selected_vertex_target.index}, Coordinates = ({selected_vertex_target.x}, {selected_vertex_target.y}, {selected_vertex_target.z})")
        else:
            selected_vertex_target = random.choice(idxTarget)
            print(f"Randomly Selected Target Vertex: Index = {selected_vertex_target.index}, Coordinates = ({selected_vertex_target.x}, {selected_vertex_target.y}, {selected_vertex_target.z})")
        return selected_vertex_target

    # Select a Vertex in the Source Area
    def select_source_vertex(self, idxSource: list[Vertex], index: int = None) -> Vertex:
        if index is not None:
            selected_vertex_source = idxSource[index]
            print(f"Selected Source Vertex by Index: Index = {selected_vertex_source.index}, Coordinates = ({selected_vertex_source.x}, {selected_vertex_source.y}, {selected_vertex_source.z})")
        else:
            selected_vertex_source = random.choice(idxSource)
            print(f"Randomly Selected Source Vertex: Index = {selected_vertex_source.index}, Coordinates = ({selected_vertex_source.x}, {selected_vertex_source.y}, {selected_vertex_source.z})")
        return selected_vertex_source

    # Define Range of Sizes
    def define_size_range(self, start: float = 1, stop: float = -1.25, num: int = 50) -> list:
        sigma_values = np.logspace(start, stop, num).tolist()
        print(f"Sigma Values for Optimization: {sigma_values}")
        return sigma_values
   
    def plot_time_series(self, save_path: str = None):
        """
        Plot the observed vs. predicted time series.
        If `save_path` is provided, the plot is saved to the specified location and not displayed.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(self.observed_time_series, label=f'Observed Time Series', linestyle='-', marker='o')
        plt.plot(self.predicted_time_course, label=f'Predicted Time Series', linestyle='--', marker='x')
        plt.title('Observed vs Predicted Time Series')
        plt.xlabel('Time Points')
        plt.ylabel('BOLD Signal')
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)  # Save the plot to the specified path
            plt.close() 
        else:
            plt.show()  # Display the plot on the screen

    def calculate_gaussian_weights(self, distances: np.ndarray, sigma_values: list) -> np.ndarray:
        sigma_values = np.array(sigma_values) # (50) just the values of sigma
        weights = np.exp(-distances / (2 * sigma_values ** 2))
        weights = weights / np.sum(weights, axis=0) # Normalized
        return weights  # (1688, 50) source vertex x sigma value 

    def compute_prediction(self, source_time_series: dict, distances: np.ndarray, sigma_values: np.ndarray):
        weights_matrix = self.calculate_gaussian_weights(distances, sigma_values) 
        
        # Extract time series for all source vertices

        # PREVIOUS 
        # filtered_vertices = list(distance_matrix.index)

        # UPDATED
        filtered_vertices = [v for v in distance_matrix.index if v in source_time_series]
        filtered_time_series = [source_time_series[v] for v in filtered_vertices]

        # Stack time series into a matrix (128, 1688) time course x source vertices 
        time_series_matrix = np.stack(filtered_time_series, axis=1)  
    
        # Compute all predictions at once using dot product (128,50) time course x sigma value 
        # predicted time series for a specific sigma value, and each row represents a specific time point
        predicted_time_series_matrix = np.dot(time_series_matrix, weights_matrix) 
        return predicted_time_series_matrix, weights_matrix 

    def evaluate_fit(self, observed: np.ndarray, predicted_matrix: np.ndarray) -> np.ndarray:
        ss_total = np.sum(observed ** 2) 
        ss_residual = np.sum((observed[:, np.newaxis] - predicted_matrix) ** 2, axis=0)
        variance_explained = 1 - (ss_residual / ss_total)
        return variance_explained

    def evaluate_mse(self, observed: np.ndarray, predicted_matrix: np.ndarray) -> np.ndarray:
        """
        Calculate mean squared error between observed and each predicted time series.
        """
        mse = np.mean((observed[:, np.newaxis] - predicted_matrix) ** 2, axis=0)
        return mse

    def optimize_parameters(self, observed: np.ndarray, source_time_series: dict, 
                            distance_matrix: pd.DataFrame, sigma_values: list, source_vertices) -> tuple:

        source_index = self.center_vertex.index

        # Get the distance column as a Series (preserve index labels!)
        row_data_series = distance_matrix.loc[:, source_index]

        # Filter distances to match the keys in source_time_series (i.e., filtered vertices)
        filtered_vertex_indices = list(source_time_series.keys())
        filtered_row_data = row_data_series.loc[filtered_vertex_indices].to_numpy().reshape(-1, 1)

        # Now compute predictions
        predicted_matrix, weights_matrix = self.compute_prediction(source_time_series, filtered_row_data, sigma_values)

        # Continue as before...
        mse_values = self.evaluate_mse(observed, predicted_matrix)
        best_index = np.argmin(mse_values) 
        best_sigma_coarse = sigma_values[best_index]
        best_prediction = predicted_matrix[:, best_index] 
        ve_for_best = self.evaluate_fit(observed, best_prediction[:, np.newaxis])[0]

        self.sigma_coarse = best_sigma_coarse
        self.variance_explained_coarse = ve_for_best
        return best_sigma_coarse, ve_for_best, best_prediction

    def iterative_fit_target(self, target_vertex: Vertex, target_time_series, source_vertices: list[Vertex], 
                            source_time_series: dict, distance_matrix: pd.DataFrame, 
                            sigma_values: list, best_fit_output: str, individual_output_dir: str, plot_dir: str):

        #self.observed_time_series = target_time_course.tSeries[target_vertex.index]
        self.observed_time_series = target_time_series[target_vertex.index]
        results = []  
        best_fit_temp = None
        best_coarse_ve = -np.inf

        # Iterate through all source vertices: only coarse search
        for source_vertex in source_vertices:  
            self.center_vertex = source_vertex
            sigma_coarse, ve_coarse, prediction_coarse = self.optimize_parameters(
                self.observed_time_series, source_time_series, distance_matrix, sigma_values, source_vertices)

            results.append({
                "Target Vertex Index": target_vertex.index,
                "Source Vertex Index": source_vertex.index,
                "Best Sigma Coarse": sigma_coarse,
                "Best Variance Explained Coarse": ve_coarse,
            })

            # Track best fit across all source vertices (coarse)
            if ve_coarse > best_coarse_ve:
                best_coarse_ve = ve_coarse
                best_fit_temp = {
                    "source_vertex": source_vertex,
                    "sigma_coarse": sigma_coarse,
                    "ve_coarse": ve_coarse,
                    "prediction_coarse": prediction_coarse
                }

        # Save all coarse results
        # results_df = pd.DataFrame(results)
        # individual_file = os.path.join(individual_output_dir, f"all_fits_target_vertex_{target_vertex.index}.csv")
        # results_df.to_csv(individual_file, index=False)

        # UPDATE TO AVOID the code to stop when a best fit is not found 
        if best_fit_temp is None: 
            print(f"No valid fit found for target vertext: {target_vertex}") 
            return

        # Finer search now 
        self.center_vertex = best_fit_temp["source_vertex"]
        row_data_series = distance_matrix.loc[:, self.center_vertex.index]
        filtered_indices = list(source_time_series.keys())
        filtered_row_data = row_data_series.loc[filtered_indices].to_numpy().reshape(-1, 1)

        sigma_finer, prediction_finer, ve_finer = self.finer_search_sigma(
            self.observed_time_series, source_time_series, filtered_row_data, best_fit_temp["sigma_coarse"])

        # Store final best fit
        self.sigma = sigma_finer
        self.variance_explained = ve_finer
        self.predicted_time_course = prediction_finer
        self.best_source_index = self.center_vertex.index

        # Save best fit result
        best_fit_df = pd.DataFrame([{
            "Target Vertex Index": target_vertex.index,
            "Source Vertex Index": self.best_source_index,
            "Best Sigma Coarse": best_fit_temp["sigma_coarse"],
            "Best Sigma Finer": sigma_finer,
            "Best Variance Explained Coarse": best_fit_temp["ve_coarse"],
            "Best Variance Explained Finer": ve_finer
        }])

        best_fit_df.to_csv(best_fit_output, mode="a", index=False, header=not os.path.exists(best_fit_output))

        # Plot and save
        # plot_file = os.path.join(plot_dir, f"best_fit_plot_target_vertex_{target_vertex.index}.png")
        # os.makedirs(os.path.dirname(plot_file), exist_ok=True)
        # self.plot_time_series(save_path=plot_file)

    def finer_search_sigma(self, observed: np.array, source_time_series: dict, distances: np.array, initial_sigma: float):   
        sigma_trials = []
        
        def objective(sigma_array): 
            sigma = sigma_array[0]
            sigma_trials.append(sigma)

            weights = self.calculate_gaussian_weights(distances, [sigma]).flatten()
            vertex_indices = list(source_time_series.keys())
            time_series_matrix = np.stack([source_time_series[v_idx] for v_idx in vertex_indices], axis=1)
            predicted = np.dot(time_series_matrix, weights)

            ve = self.evaluate_fit(observed, predicted[:, np.newaxis])[0]
            return -ve 
            
        result = minimize(objective, [initial_sigma], method='Nelder-Mead', bounds=[(0.05, 10.5)])
        best_sigma = result.x[0] 

        weights = self.calculate_gaussian_weights(distances, [best_sigma]).flatten()
        vertex_indices = list(source_time_series.keys())
        time_series_matrix = np.stack([source_time_series[v_idx] for v_idx in vertex_indices], axis=1)
        prediction = np.dot(time_series_matrix, weights)
        variance_explained = self.evaluate_fit(observed, prediction[:, np.newaxis])[0] 
        return best_sigma, prediction, variance_explained


# %% [markdown]
# ### The Main Script

# %%
if __name__ == "__main__":
    MAIN_PATH = '/scratch/hb-EGRET-AAA/projects/EGRET+/derivatives'

    # subj = 'sub-46'
    ses = 'ses-02'
    cutoff_volumes = 8
    hemispheres = ['lh', 'rh']
    source_visual_area = 1
    source_name = 'V1'
    target_visual_areas = [1, 2, 3] # Why 7?
    rois_list = np.array([['V1', 'V2', 'V3'], [1, 2, 3]]) # Why 7?
    load_one = None
    ncores = 65
    filter_v1 = True
    processing_method = "zscore"  # Options: "zscore", "demean", "none"

    # Lists
    atlases = ['manual', 'benson']
    tasks = ['RET', 'RET2', 'RestingState']
    denoising_methods = ['nordic', 'nordic_sm4']

    start_time = time.time()

    for atlas, task, denoising in itertools.product(atlases, tasks, denoising_methods):
        print(f"\nProcessing: Atlas={atlas}, Task={task}, Denoising={denoising}")

        for hemi, target_visual_area in itertools.product(hemispheres, target_visual_areas):
            if atlas == "manual":
                labels_path = f"{MAIN_PATH}/freesurfer/{subj}/label/{hemi}.manualdelin.label"
            else:
                labels_path = f"{MAIN_PATH}/freesurfer/{subj}/label/{hemi}.benson14_varea-0001.label"

            if hemi == 'lh':
                time_series_path = f"{MAIN_PATH}/pRFM/{subj}/{ses}/{denoising}/{subj}_{ses}_task-{task}_hemi-lh_desc-avg_bold_GM.npy"
            else:
                time_series_path = f"{MAIN_PATH}/pRFM/{subj}/{ses}/{denoising}/{subj}_{ses}_task-{task}_hemi-rh_desc-avg_bold_GM.npy"

            # Outputs
            output_dir = f"{MAIN_PATH}/CFM/{subj}/{ses}/{atlas}/{task}/{denoising}/GM" # Needs the name of the visual area actually
            os.makedirs(output_dir, exist_ok=True)
            target_name_idx = np.where(str(target_visual_area) == rois_list[1])
            target_name = rois_list[0][target_name_idx][0]
            distance_matrix_path = f"{output_dir}/Distance_Matrices"
            os.makedirs(distance_matrix_path, exist_ok=True)
            distance_matrix_file = f"{distance_matrix_path}/{subj}_distance_{hemi}_{source_visual_area}.csv"
            output_dir_itertarget = f"{output_dir}/{hemi}/{target_name}-{source_name}"
            os.makedirs(output_dir_itertarget, exist_ok=True)
            best_fit_output = f"{output_dir_itertarget}/best_fits.csv"
            individual_output_dir = f"{output_dir_itertarget}/individual_fits"
            os.makedirs(individual_output_dir, exist_ok=True)

            # 1. Load Vertices
            idxTarget = Vertex.load_vertices(labels_path, target_visual_area, atlas, load_one)
            print(f"Target Area: {idxTarget.shape}")
            idxSource = Vertex.load_vertices(labels_path, source_visual_area, atlas, load_one)
            print(f"Source Area: {idxSource.shape}")

            # 1a. Apply Eccentricity Filtering Based on Atlas
            if atlas == "benson":
                ecc_dict = source_eccentricity_two(subj=subj, hemi=hemi, main_path=MAIN_PATH, atlas=atlas,
                    denoising=denoising, task=task, freesurfer_path=f"{MAIN_PATH}/freesurfer",label_file=labels_path)
                idxTarget = [v for v in idxTarget if v.index in ecc_dict and ecc_dict[v.index] < 10]
                idxSource = [v for v in idxSource if v.index in ecc_dict and ecc_dict[v.index] < 10]
                filtered_idxSource = idxSource  
                filtered_idxTarget = idxTarget  
                print(f"Filtered Source Area: {len(filtered_idxSource)}")
                print(f"Filtered Target Area: {len(filtered_idxTarget)}")
            else:
                if filter_v1:
                    ecc_dict = source_eccentricity_two(subj=subj, hemi=hemi, main_path=MAIN_PATH, atlas=atlas,
                        denoising=denoising, task=task, freesurfer_path=f"{MAIN_PATH}/freesurfer")
                    filtered_idxSource = [v for v in idxSource if v.index in ecc_dict and 0.5 <= ecc_dict[v.index] <= 6]
                    print(f"Filtered Source Area: {len(filtered_idxSource)}")
                    filtered_idxTarget = idxTarget
                else:
                    filtered_idxSource = idxSource
                    filtered_idxTarget = idxTarget
                    print(f"Filtered Source Area: {len(filtered_idxSource)}")
            
            # 2. Calculate Distance Matrix using filtered_idxSource
            distances_class = Distances(subject=subj, hemi=hemi, matrix_dir=distance_matrix_path, csv_path=distance_matrix_file)
            
            if atlas == "benson":
                distances_class.geodesic_dists(hemi=hemi,subject=subj, vertices=filtered_idxSource, source=source_visual_area,output_dir=distance_matrix_path)
                distance_matrix = pd.read_csv(distance_matrix_file, index_col=0)
                print(f"Distance Matrix Benson: {distance_matrix.shape}")
            else: 
                if filter_v1:
                    distances_class.geodesic_dists(hemi=hemi,subject=subj, vertices=filtered_idxSource, source=source_visual_area,output_dir=distance_matrix_path)
                    distance_matrix = pd.read_csv(distance_matrix_file, index_col=0)
                    print(f"Distance Matrix Manual Filtered: {distance_matrix.shape}")
                else: 
                    distances_class.geodesic_dists(hemi=hemi,subject=subj, vertices=idxSource, source=source_visual_area,output_dir=distance_matrix_path)
                    distance_matrix = pd.read_csv(distance_matrix_file, index_col=0)
                    print(f"Distance Matrix Manual Non Filtered: {distance_matrix.shape}")
            
            # distance_matrix = pd.read_csv(distance_matrix_file, index_col=0)
            distance_matrix.index = distance_matrix.index.astype(int)
            distance_matrix.columns = distance_matrix.columns.astype(int)

            # 3. Load Time Series Data 
            target_time_course_obj = TimeCourse(time_course_file=time_series_path, vertices=filtered_idxTarget, cutoff_volumes=cutoff_volumes)
            source_time_course_obj = TimeCourse(time_course_file=time_series_path, vertices=filtered_idxSource, cutoff_volumes=cutoff_volumes)
            z_scored_target = target_time_course_obj.z_score(method=processing_method)
            z_scored_source = source_time_course_obj.z_score(method=processing_method)

            # 4. Define Sigma Range 
            connective_field = ConnectiveField(center_vertex=None, vertex=None)
            sigma_values = connective_field.define_size_range(start=1, stop=-1.25, num=50)

            # 5. Run Iterative Fit 
            Parallel(n_jobs=ncores)(
                delayed(connective_field.iterative_fit_target)(
                    target_vertex=target_vertex,
                    target_time_series=z_scored_target,
                    source_vertices=filtered_idxSource,
                    source_time_series=z_scored_source,
                    distance_matrix=distance_matrix,
                    sigma_values=sigma_values,
                    best_fit_output=best_fit_output,
                    individual_output_dir=individual_output_dir,
                    plot_dir=individual_output_dir)
                for target_vertex in idxTarget)

            print(f"Completed: {atlas}, {task}, {denoising}, {hemi}, {target_name}")

    elapsed_time = (time.time() - start_time) / 60
    print(f"\nAll processing completed in {elapsed_time:.2f} minutes.")


