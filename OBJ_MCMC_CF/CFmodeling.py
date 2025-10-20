from imports import *
from vertex import Vertex

def surfs(subject: str, hemi:str):
    # Check if the hemisphere is "lh" (left hemisphere)
    # Force pycortex to use the specified filestore
    new_filestore = os.path.join(f'{MAIN_PATH}', 'pycortex')
    cortex.database.default_filestore = new_filestore
    cortex.database.db = cortex.database.Database(new_filestore)
    if hasattr(cortex, 'db'):
        cortex.db = cortex.database.db

    if hemi == "lh":
        surf_data = cortex.db.get_surf(subject, "fiducial")[0] # index [0] is the left hemisphere
    elif hemi == "rh":
        surf_data = cortex.db.get_surf(subject, "fiducial")[1] # index [1] is the right hemisphere

    # Convert the raw surface data into a Surface object that is easier to use
    surface = cortex.polyutils.Surface(*surf_data)
    return surface

class Distances(Vertex):
    def __init__(self, subject, hemi, matrix_dir, csv_path):
        self.subject = subject # Store subject ID
        self.hemi = hemi 
        self.matrix_dir = matrix_dir # Directory where distance matrices are saved
        self.csv_path = csv_path # Path to the CSV file where distances will be stored
    
    def geodesic_dists(self, hemi, subject, vertices, source, output_dir):

        # Convert the list of vertex objects into a NumPy array of vertex indices
        source_verts = np.array([v.index for v in vertices])
        
        # Build the name of the CSV file
        output_path = f"{output_dir}/{subject}_distance_{hemi}_{source}.csv"
        
        # OPTIONAL: Load pre-computed matrix if it exists
        if os.path.exists(output_path):
            try:
                distance_matrix = pd.read_csv(output_path, index_col=0).values
                print(f"Loaded distance matrix with shape: {distance_matrix.shape}")
                return distance_matrix
            except Exception as e:
                print("Computing the geodesic distance matrix...")
        
        # Load the cortical surface for the given hemisphere
        surface = surfs(subject, hemi)
        
        # Initialize an empty square matrix for distances: rows and columns correspond to vertices
        dists_source = np.zeros((len(source_verts), len(source_verts)), dtype=np.float32)
        
        # BUG: PREVIOUS COMPUTATION FOR THE DISTANCE MATRIX 
        # for i in range(len(source_verts)):
        #    dists = surface.geodesic_distance(source_verts[i])  
        #    for j in range(len(source_verts)):
        #        dists_source[i, j] = dists[source_verts[j]]  
        
        # Compute pairwise geodesic distances
        # Only compute half (i,j) and mirror it into (j,i) to save time 
        for i in range(len(source_verts)):
            dists = surface.geodesic_distance(source_verts[i]) # Get distances from vertex i to all other vertices
            for j in range(i, len(source_verts)):
                d = dists[source_verts[j]]
                dists_source[i, j] = d
                dists_source[j, i] = d  

        # # Convert to a DataFrame
        distance_df = pd.DataFrame(dists_source, index=source_verts, columns=source_verts)
        distance_df.to_csv(output_path) # Save the DataFrame as a CSV file
        # print(f"Distance matrix saved with shape: {distance_df.shape}") # For debugging  
        return dists_source

class TimeCourse:
    def __init__(self, time_course_file: str, vertices: list[Vertex], cutoff_volumes: int):
        self.vertices = vertices  # List of Vertex objects
        self.cutoff_volumes = cutoff_volumes # # Number of volumes to cut off at the beginning
        self.data = np.load(time_course_file)  # Load the full time course from file
        self.tSeries = self.load_time_courses() # Dictionary {vertex_index: time_series}

    def load_time_courses(self) -> dict:
        duration = self.data.shape[0] # number of time points in the dataset
        tSeries = {}
        # Iterates over the self.vertices list, accessing the index of each vertex.
        for vertex in self.vertices:
            index = vertex.index # Numeric index of this vertex
            time_course = self.data[self.cutoff_volumes:duration, index] # Extracts the time course for each vertex
            tSeries[index] = time_course # Save this time series in the dictionary
            # print(f"Loaded time series for vertex {index} with length {len(time_course)") # For debugging}")
        return tSeries

    def z_score(self, method: str = "zscore") -> dict:
        processed_data = {} # Dictionary to store the processed time series
        
        # Loop through each vertex and its time course
        for index, time_course in self.tSeries.items():
            # Subtract mean and divide by standard deviation
            # This rescales the data so it's centered at 0 with unit variance
            if method == "zscore": 
                processed = (time_course - np.nanmean(time_course)) / np.nanstd(time_course)
            # Subtract only the mean 
            # Signal is centered at 0 but keeps original variance
            elif method == "demean":
                processed = time_course - np.nanmean(time_course)
            elif method == "none": # Keep the raw signal as it is
                processed = time_course
            processed_data[index] = processed  # Save the processed time series in the dictionary (key = vertex index)
        return processed_data
    
    def plot_time_series(self, vertex_index: int, show: bool = True) -> None:
        time_course = self.tSeries[vertex_index] # Extract the time course for this vertex
        plt.figure(figsize=(10, 5))
        plt.plot(time_course, label=f'Vertex Index: {vertex_index}', color='blue')
        plt.title(f'Time Series for Vertex {vertex_index}')
        plt.xlabel('Time Points')
        plt.ylabel('BOLD Signal')
        plt.legend()
        plt.grid()
        if show:
            plt.show()
        
    def plot_comparison(self, z_scored_data: dict, vertex_index: int, title_prefix: str, show: bool = True) -> None:
        original_time_course = self.tSeries[vertex_index] # Get the raw time course for this vertex
        z_scored_time_course = z_scored_data[vertex_index] # Get the processed time course for the same vertex
        plt.figure(figsize=(12, 6))
        plt.plot(z_scored_time_course, label="Preprocessed Time Series", linestyle="--", marker="x", alpha=0.7)
        plt.plot(original_time_course, label="Raw Time Series", linestyle="-", marker="o", alpha=0.7)
        plt.title(f"{title_prefix} Vertex {vertex_index} - Comparison")
        plt.xlabel("Time Points")
        plt.ylabel("BOLD Signal")
        plt.legend()
        plt.grid()
        if show:
            plt.show()

class ConnectiveField:
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

    def select_target_vertex(self, idxTarget: list[Vertex], index: int = None) -> Vertex:
        # Select a Vertex in the Target Area # Mainly for debugging 
        if index is not None:
            selected_vertex_target = idxTarget[index]
            print(f"Selected Target Vertex by Index: Index = {selected_vertex_target.index}, Coordinates = ({selected_vertex_target.x}, {selected_vertex_target.y}, {selected_vertex_target.z})")
        else:
            selected_vertex_target = random.choice(idxTarget)
            print(f"Randomly Selected Target Vertex: Index = {selected_vertex_target.index}, Coordinates = ({selected_vertex_target.x}, {selected_vertex_target.y}, {selected_vertex_target.z})")
        return selected_vertex_target

    def select_source_vertex(self, idxSource: list[Vertex], index: int = None) -> Vertex:
        # Select a Vertex in the Target Area # Mainly for debugging 
        if index is not None:
            selected_vertex_source = idxSource[index]
            print(f"Selected Source Vertex by Index: Index = {selected_vertex_source.index}, Coordinates = ({selected_vertex_source.x}, {selected_vertex_source.y}, {selected_vertex_source.z})")
        else:
            selected_vertex_source = random.choice(idxSource)
            print(f"Randomly Selected Source Vertex: Index = {selected_vertex_source.index}, Coordinates = ({selected_vertex_source.x}, {selected_vertex_source.y}, {selected_vertex_source.z})")
        return selected_vertex_source

    # Define Range of Sizes
    def define_size_range(self, start: float = 1, stop: float = -1.25, num: int = 50) -> list:
        # np.logspace creates values between start and stop spaced evenly on a log scale
        sigma_values = np.logspace(start, stop, num).tolist()
        # print(f"Sigma Values for Optimization: {sigma_values}") # For debugging
        return sigma_values
   
    def plot_time_series(self, save_path: str = None):
        # THIS IS REPEATED CODE FROM TimeCourse CLASS: should be removed.
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
        sigma_values = np.array(sigma_values) # Convert sigma list into NumPy array for broadcasting
        weights = np.exp(-distances**2 / (2 * sigma_values ** 2)) # Compute Gaussian weights with  (n_distances, n_sigmas)
        weights = weights / np.sum(weights, axis=0) # Normalize across distance
        return weights # (distances x sigma values)

    def compute_prediction(self, ts_matrix: np.ndarray, distances: np.ndarray, sigma_values: list[float]):
        # Predict for all sigmas using arrays that are already aligned
        weights_matrix = self.calculate_gaussian_weights(distances, sigma_values) # ts_matrix: shape (T, Nsrc), columns ordered exactly as your distances.
        predicted_time_series_matrix = np.dot(ts_matrix, weights_matrix) # distances: shape (Nsrc, 1), same order as ts_matrix columns.
        return predicted_time_series_matrix, weights_matrix

    def evaluate_fit(self, observed: np.ndarray, predicted_matrix: np.ndarray) -> np.ndarray:
        # Total sum of squares (SS_total) = variance in the observed signal
        ss_total = np.sum(observed ** 2) 
        # Residual sum of squares (SS_residual) = for each sigma, compute the squared error between observed and predicted
        ss_residual = np.sum((observed[:, np.newaxis] - predicted_matrix) ** 2, axis=0)
        variance_explained = 1 - (ss_residual / ss_total)
        return variance_explained

    def evaluate_mse(self, observed: np.ndarray, predicted_matrix: np.ndarray) -> np.ndarray:
        # # Expand observed to shape (n_time_points, 1) so it can broadcast against predicted_matrix (n_time_points, n_sigmas)
        # Square the differences, then take mean across time points (axis=0)
        mse = np.mean((observed[:, np.newaxis] - predicted_matrix) ** 2, axis=0)
        return mse

    def optimize_parameters(self, observed: np.ndarray, source_time_series: dict, distance_matrix: pd.DataFrame, sigma_values: list, source_vertices) -> tuple:
        source_index = self.center_vertex.index
        vertex_indices = list(source_time_series.keys()) # Define one order for sources
        row_data = distance_matrix.loc[vertex_indices, source_index].to_numpy().reshape(-1, 1) # Distances
        ts_matrix = np.stack([source_time_series[v] for v in vertex_indices], axis=1) # Stack time series   
        # Predictions for all sigma 
        predicted_matrix, _ = self.compute_prediction(ts_matrix, row_data, sigma_values)        
        # Choose sgima by MSE 
        # mse_values = self.evaluate_mse(observed, predicted_matrix)
        # best_index = np.argmin(mse_values)
        # Choose sgima by VE 
        # For each candidate sigma, compute how well the model predicts the observed signal
        ve_values = self.evaluate_fit(observed, predicted_matrix) 
        ve_max = float(ve_values.max())

        # Treat near-equal VEs as ties
        mask = np.isclose(ve_values, ve_max, rtol=1e-12, atol=1e-12)
        idxs = np.flatnonzero(mask)
        # If there are ties, we pick the one with the largest sigma value
        sig_arr = np.asarray(sigma_values, dtype=float)
        # best_tie_idx = idxs[np.argmax(sig_arr[idxs])] ####################################################################################################################### TEST
        
        best_tie_idx = idxs[np.argmin(sig_arr[idxs])]
        # Extract the best sigma and corresponding prediction
        best_index = int(best_tie_idx)
        best_sigma_coarse = float(sig_arr[best_index])
        best_prediction = predicted_matrix[:, best_index]
        ve_for_best = float(ve_values[best_index])
        self.sigma_coarse = best_sigma_coarse
        self.variance_explained_coarse = ve_for_best
        return best_sigma_coarse, ve_for_best, best_prediction
    
    def iterative_fit_target(self, target_vertex: Vertex, target_time_series, source_vertices: list[Vertex], source_time_series: dict, distance_matrix: pd.DataFrame, sigma_values: list, best_fit_output: str, mode: str = "bayesian"):
        # Grab observed signal for the target vertex (what we want to predict)
        self.observed_time_series = target_time_series[target_vertex.index]
        
        if mode == "standard": 
            results = []
            best_fit_temp = None
            import numpy as np
            best_coarse_ve = -np.inf
            # Iterate through all source vertices: coarse search
            for source_vertex in source_vertices:
                self.center_vertex = source_vertex
                sigma_coarse, ve_coarse, prediction_coarse = self.optimize_parameters(self.observed_time_series, source_time_series, distance_matrix, sigma_values, source_vertices)
                results.append({"Target Vertex Index": target_vertex.index, "Source Vertex Index": source_vertex.index, "CF Sigma Coarse": sigma_coarse, "Variance Explained Coarse": ve_coarse})
                if ve_coarse > best_coarse_ve:
                    best_coarse_ve = ve_coarse
                    best_fit_temp = {"source_vertex": source_vertex, "sigma_coarse": sigma_coarse, "ve_coarse": ve_coarse, "prediction_coarse": prediction_coarse}

            if best_fit_temp is None:
                print(f"No valid fit found for target vertex: {target_vertex}")
                return

            # Finer search around the best coarse source vertex
            self.center_vertex = best_fit_temp["source_vertex"]
            row_data_series = distance_matrix.loc[:, self.center_vertex.index]          # distances to chosen source
            filtered_indices = list(source_time_series.keys())                          # same order for distances + series
            filtered_row_data = row_data_series.loc[filtered_indices].to_numpy().reshape(-1, 1)

            sigma_finer, prediction_finer, ve_finer = self.finer_search_sigma(self.observed_time_series, source_time_series, filtered_row_data, best_fit_temp["sigma_coarse"])

            # Store final best-fit params
            self.sigma = sigma_finer
            self.variance_explained = ve_finer
            self.predicted_time_course = prediction_finer
            self.best_source_index = self.center_vertex.index

            # Save one-row CSV with best fit
            best_fit_df = pd.DataFrame([{"Target Vertex Index": target_vertex.index, "Source Vertex Index": self.best_source_index, "CF Sigma Coarse": best_fit_temp["sigma_coarse"], "CF Sigma": sigma_finer,"Variance Explained Coarse": best_fit_temp["ve_coarse"],"Variance Explained": ve_finer}])
            best_fit_df.to_csv(best_fit_output, mode="a", index=False, header=not os.path.exists(best_fit_output))
        
        elif mode == "bayesian":
                    # Build dict with only filtered sources
                    source_idx = [v.index for v in source_vertices]  
                    available_sources = set(source_time_series.keys())
                    valid_source_idx = [i for i in source_idx if i in available_sources]
                    import numpy as np 
                    source_matrix = np.stack([source_time_series[i] for i in valid_source_idx], axis=1)
                    target_matrix = self.observed_time_series[:, None]  # observed signal

                    # MCMC cluster
                    (bestFit, ve, posterior, posteriorLatent, postDist, loglikelihood) = MCMC_CF_cluster( idxSource=np.array(source_idx), distances=distance_matrix.values, tSeriesSource=source_matrix, tSeriesTarget=target_matrix)
                    # Save the full iterations for debugging 
                    n_iter = ve.shape[0]
                    rows = []
                    for j in range(n_iter):
                        rows.append({"TargetVertexIndex": int(target_vertex.index),"Iteration": j,"SourceVertexIndex": int(posterior[2, j]), "Sigma": float(posterior[0, j]),"Beta": float(posteriorLatent[1, j]),"Variance Explained": float(ve[j]), "Posterior": float(postDist[j]),"LogLikelihood": float(loglikelihood[j])})
                    df_all = pd.DataFrame(rows)
                    # chain_csv = os.path.join(output_dir_itertarget, f"iterations_{target_vertex.index}.csv")
                    # df_all.to_csv(chain_csv, index=False)
                    # Save results 
                    row = {"Target Vertex Index": int(target_vertex.index),"Source Vertex Index": int(bestFit[2]),"CF Sigma Bayesian": float(bestFit[0]), "Variance Explained": float(bestFit[3]),"Beta Bayesian": float(bestFit[1])}
                    bestfit_csv = os.path.join(output_dir_itertarget, "bestfit_bayesian_T1.csv")
                    pd.DataFrame([row]).to_csv(bestfit_csv, mode="a", index=False, header=not os.path.exists(bestfit_csv))
                    return {"mode": "bayesian","row": row,"all_iters": df_all,"source_matrix": source_matrix,"valid_source_idx": valid_source_idx}

    def finer_search_sigma(self, observed: np.array, source_time_series: dict, distances: np.array, initial_sigma: float):
        # Objective function: negative variance explained
        def neg_ve(sigma):
            # Compute Gaussian weights for this sigma
            weights = self.calculate_gaussian_weights(distances, [sigma]).flatten()
            # Stack all source time series into a 2D matrix: (n_time_points, n_vertices)
            vertex_indices = list(source_time_series.keys())
            time_series_matrix = np.stack([source_time_series[v] for v in vertex_indices], axis=1)
            predicted = np.dot(time_series_matrix, weights) # Weighted sum of source time series to predict the target
            ve = self.evaluate_fit(observed, predicted[:, np.newaxis])[0] # Compute variance explained between observed and predicted
            # Minimizing it is equivalent to maximizing VE
            return -ve

        # Run scalar optimization with bounds: sigma must stay between 0.05 and 10.5
        # "bounded" forces these limits
        # res = minimize_scalar(neg_ve, bounds=(0.05, 10.5), method='bounded')
        # best_sigma = float(res.x)
        # run a bounded scalar search inside this local window
        res = minimize_scalar(neg_ve, bounds=(0.05, 10.5), method='bounded')
        best_sigma = float(res.x)

        # Evaluate the coarse sigma itself and keep whichever is better
        def ve_at(s):
            w = self.calculate_gaussian_weights(distances, [s]).flatten()
            idx = list(source_time_series.keys())
            X = np.stack([source_time_series[v] for v in idx], axis=1)
            pred = np.dot(X, w)
            return float(self.evaluate_fit(observed, pred[:, None])[0])

        ve_coarse = ve_at(initial_sigma)
        ve_fine   = ve_at(best_sigma)
        if ve_coarse >= ve_fine:
            best_sigma = float(initial_sigma)

        # Compute final prediction with the chosen sigma
        weights = self.calculate_gaussian_weights(distances, [best_sigma]).flatten()
        vertex_indices = list(source_time_series.keys())
        time_series_matrix = np.stack([source_time_series[v] for v in vertex_indices], axis=1)
        prediction = np.dot(time_series_matrix, weights)
        variance_explained = self.evaluate_fit(observed, prediction[:, np.newaxis])[0] # Compute variance explained for this prediction
        return best_sigma, prediction, variance_explained

# Main Script 
if __name__ == "__main__":
    for atlas, task, denoising in itertools.product(atlases, tasks, denoising_methods):
        
        # STEP 1: extract the vertices and the corresponding time courses from the preprocessed fMRI data
        current_runs = runs if (task == "RestingState" and has_multiple_runs) else [None] # Decide which runs to iterate:
        for run in current_runs:
            for hemi, target_visual_area in itertools.product(hemispheres, target_visual_areas):
                # Pick which label file we need 
                label_suffix = "manualdelin" if atlas == "manual" else "benson14_varea-0001"
                labels_path = f"{MAIN_PATH}/freesurfer/{subj}/label/{hemi}.{label_suffix}.label"
                # Time series path depends only on hemisphere
                time_series_path = f"{MAIN_PATH}/pRFM/{subj}/{ses}/{denoising}/{subj}_{ses}_task-{task}_hemi-{hemi}_desc-avg_bold_GM.npy"
                # Build output dir and include the run directory only when we actually have multiple runs
                base_dir = f"{MAIN_PATH}/CFM/{subj}/{ses}/{atlas}/{task}"
                output_dir = f"{base_dir}/run-{run}/{denoising}/GM" if (task == "RestingState" and has_multiple_runs) else f"{base_dir}/{denoising}/GM"
                
                # STEP 2: load the regions of interest
                target_name, target_areas = None, None
                for name, label in rois_list:
                    if target_visual_area == label:
                        target_name, target_areas = name, label
                        break
                # Wrap in a list so we can iterate 
                areas = [target_areas]
                idxTarget = []
                for area in areas:
                    idxTarget.extend(Vertex.load_vertices(labels_path, area, atlas))
                # print(f"Target Area {target_name}: {len(idxTarget)} vertices") # For debugging

                # Find the source area
                idxSource = Vertex.load_vertices(labels_path, source_visual_area, atlas)
                # print(f"Source Area: {len(idxSource)} vertices") # # For debugging

                # STEP 3: set up the directories 
                os.makedirs(output_dir, exist_ok=True)
                distance_matrix_path = f"{output_dir}/Distance_Matrices"
                os.makedirs(distance_matrix_path, exist_ok=True)
                distance_matrix_file = f"{distance_matrix_path}/{subj}_distance_{hemi}_{source_visual_area}.csv"
                output_dir_itertarget = f"{output_dir}/{hemi}/{target_name}-V{source_visual_area}"
                os.makedirs(output_dir_itertarget, exist_ok=True)
                best_fit_output = f"{output_dir_itertarget}/best_fits.csv"

                # STEP 4: filter the data points per eccentricity values
                filtered_idxTarget = idxTarget  # No modification is needed for the target area
                ecc_dict = {}
                if atlas == "benson":
                    labels_eccen_path = f"{MAIN_PATH}/freesurfer/{subj}/label/{hemi}.benson14_eccen-0001.label" # Path for Benson eccentricity labels
                    # Use only Benson labels up to benson_max
                    ecc_label_dict = benson_label_to_dict(labels_eccen_path, value_name="eccentricity")
                    ecc_dict = {v: e for v, e in ecc_label_dict.items() if e <= benson_max}
                    # print(f"Vertices kept with Benson Label: {len(ecc_dict)}")
                    # Filter out eccentricity values for the source vertices
                    if not filter_source:
                        filtered_idxSource = idxSource
                    else: 
                        filtered_idxSource = [ v for v in idxSource if (e := ecc_dict.get(v.index)) is not None and ((0.5 <= e <= max_eccentricity) if filter_source else (e <= benson_max))]
                        # print(f"Benson source vertices kept: {len(filtered_idxSource)}/{len(idxSource)} ", filter_source={filter_source})")
                        
                elif atlas == "manual":
                    # Load pRF-based eccentricity from pickle
                    ecc_dict = source_eccentricity(subj=subj, hemi=hemi, main_path=MAIN_PATH, atlas=atlas, denoising=denoising, task=task, freesurfer_path=f"{MAIN_PATH}/freesurfer")
                    # print(f"pRF-based eccentricity for {len(ecc_dict)} vertices")
                    # manual_all = {v.index for v in idxTarget}
                    # print("Manual all vertices:", len(manual_all))
                    # print("In pRF pickle:", len(manual_all & set(ecc_dict.keys())))

                    # Filter out eccentricity values for the source vertices
                    if not filter_source:
                        filtered_idxSource = idxSource
                    else: 
                        filtered_idxSource = [v for v in idxSource if (e := ecc_dict.get(v.index)) is not None and (0.5 <= e <= max_eccentricity)]
                        # print(f"Manual source vertices kept: {len(filtered_idxSource)}/{len(idxSource)} ", filter_source={filter_source})")
                
                # STEP 5: calculate the distance matrix 
                distances_class = Distances(subject=subj, hemi=hemi, matrix_dir=distance_matrix_path, csv_path=distance_matrix_file)
                distances_class.geodesic_dists(hemi=hemi, subject=subj, vertices=filtered_idxSource, source=source_visual_area, output_dir=distance_matrix_path)
                distance_matrix = pd.read_csv(distance_matrix_file, index_col=0)
                # print(f"Distance Matrix {atlas}: {distance_matrix.shape}") # For debugging 

                distance_matrix.index = distance_matrix.index.astype(int)
                distance_matrix.columns = distance_matrix.columns.astype(int)

                # STEP 6: extract the preprocessed time series 
                target_time_course_obj = TimeCourse(time_course_file=time_series_path, vertices=filtered_idxTarget, cutoff_volumes=cutoff_volumes)
                source_time_course_obj = TimeCourse(time_course_file=time_series_path, vertices=filtered_idxSource, cutoff_volumes=cutoff_volumes)
                
                # STEP 7: Standard connective field modeling
                connective_field = ConnectiveField(center_vertex=None, vertex=None)
                sigma_values = connective_field.define_size_range(start=1, stop=-1.25, num=50)
                z_scored_target = target_time_course_obj.z_score(method="zscore")
                z_scored_source = source_time_course_obj.z_score(method="zscore")
                Parallel(n_jobs=ncores)(delayed(connective_field.iterative_fit_target)(target_vertex=target_vertex, target_time_series=z_scored_target, source_vertices=filtered_idxSource, source_time_series=z_scored_source, distance_matrix=distance_matrix, sigma_values=sigma_values, best_fit_output=best_fit_output, mode = "standard") for target_vertex in idxTarget)
                
                # STEP 8: Bayesian connective field modeling
                z_scored_target = target_time_course_obj.z_score(method="zscore")
                z_scored_source = source_time_course_obj.z_score(method="zscore")
                results = Parallel(n_jobs=ncores)(delayed(connective_field.iterative_fit_target)(target_vertex=target_vertex, target_time_series=z_scored_target, source_vertices=filtered_idxSource, source_time_series=z_scored_source, distance_matrix=distance_matrix, sigma_values=sigma_values, best_fit_output=best_fit_output, mode = "bayesian") for target_vertex in idxTarget[:10])
                bayes_rows = [r["row"] for r in results if isinstance(r, dict) and r.get("mode") == "bayesian"]
                if bayes_rows:
                    df_bayes = pd.DataFrame(bayes_rows)
                    bayes_csv = os.path.join(output_dir_itertarget, "best_fits_bayesian.csv")
                    df_bayes.to_csv(bayes_csv, index=False)  # single write, safe & fast
        print(f"\nConnective Field Modeling Completed")

    # Post Processing
    #project_dir = Path(__file__).parent
    #subprocess.run([sys.executable, str(project_dir  / "clean_bestfits.py"), subj], check=True, cwd=str(project_dir))
    #subprocess.run([sys.executable, str(project_dir  / "visualfieldmaps.py"), subj], check=True, cwd=str(project_dir))
    #subprocess.run([sys.executable, str(project_dir  / "dataquality.py"), subj], check=True, cwd=str(project_dir))