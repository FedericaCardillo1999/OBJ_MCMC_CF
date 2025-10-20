from imports import *

# pRF model 
class PRFModel:
    def __init__(self, r2, size, ecc, angle):
        self.r2 = r2 # r2: model fit quality
        self.size = size # size: size of the receptive field in degrees of visual angle
        self.ecc = ecc # ecc: eccentricity (distance from the fovea, in degrees)
        self.angle = angle # angle: polar angle (direction in the visual field, in radians or degrees)

def read_benson_label_vertices(label_file):
    # Load the label file into a pandas DataFrame
    # sep=r"\s+" means split by any amount of whitespace
    # skiprows=2 skips the first 2 lines (header info in FreeSurfer label files)
    df = pd.read_csv(label_file, sep=r"\s+", skiprows=2, header=None)
    # Add column names to make it easier to read
    df.columns = ["vertex", "x", "y", "z", "value"]
    return df["vertex"].astype(int).to_numpy()

def load_prf(subj, main_path, atlas, denoising, task):
    task_for_file = "RET" if task == "RestingState" else task

    for ses in ["ses-01", "ses-02", "ses-03"]: # Check multiple possible session folders until we find a matching file
        # Build the path to the pickle file
        filepath = os.path.join(main_path, f"pRFM/{subj}/{ses}/{denoising}/model-{atlas}-nelder-mead-GM_desc-prf_params_{task_for_file}.pkl")
        # If the pickle file exists, load it to extract the parameters
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                pkl = pickle.load(f)
            prf_params = pkl["model"].iterative_search_params # Extract pRF parameters for each vertex row
            prf_voxels = np.where(pkl["rois_mask"] == 1)[0] # Extract the vertex indices that are inside the ROI mask 
            return prf_params, prf_voxels
    # If no file was found in any session print it
    raise FileNotFoundError(f"No PRF {main_path}/pRFM/{subj}/<ses>/{denoising}/")

def filter_prf(prf_params, prf_voxels):
    x = prf_params[:, 0] # Extract x-coordinate of receptive field centers 
    y = prf_params[:, 1] # Extract y-coordinate of receptive field centers
    size = prf_params[:, 2]
    r2 = prf_params[:, 7]
    ecc = np.sqrt(x**2 + y**2) # Compute eccenricity
    angle = np.arctan2(y, x) # Compute polar angle 
    return PRFModel(r2=r2, size=size, ecc=ecc, angle=angle)

def adjusting_verticesandecc(prf_voxels, prf_model, freesurfer_path, subj, hemi):
    lh_curv_path = os.path.join(freesurfer_path, subj, "surf", "lh.curv") # Path to left hemisphere curvature file
    lh_curv = read_morph_data(lh_curv_path) # Read number of vertices in the left hemisphere 
    n_lh = lh_curv.shape[0] # Total number of vertices in left hemisphere
    # For right hemisphere the vertices indices are offset by n_lh
    if hemi == "rh":
        mask = prf_voxels >= n_lh
        adjusted_vertices = prf_voxels[mask] - n_lh
        ecc_values = prf_model.ecc[mask]
        angle_values = prf_model.angle[mask]
    else:
        mask = prf_voxels < n_lh # For left hemisphere: use indices directly (less than n_lh)
        adjusted_vertices = prf_voxels[mask]
        ecc_values = prf_model.ecc[mask]
        angle_values = prf_model.angle[mask]
    return adjusted_vertices, ecc_values, angle_values # Return vertex index and the corresponding eccentricity values

def benson_label_to_dict(label_file, value_name="value"):
    df = pd.read_csv(label_file, sep=r"\s+", skiprows=2, header=None) # Load the FreeSurfer label file as a dataframe
    df.columns = ["vertex", "x", "y", "z", value_name]  # Name the different columns of the label
    vertices = df["vertex"].to_numpy() # Extract the vertex indices 
    values = df[value_name].to_numpy() # Extract the eccentricity values from the file.
    return dict(zip(vertices, values)) # Could be eccentricity or polar angle

def source_eccentricity(subj, hemi, main_path, atlas, denoising, task, freesurfer_path, label_file=None, benson_fallback_ecc_max=None):
    # Load the pRF results for this subject, atlas, denoising method, and task to obtain the source vertices filtered by eccentricity
    prf_params, prf_voxels = load_prf(subj, main_path, atlas, denoising, task) # Load the pRF mapping results
    prf_model = filter_prf(prf_params, prf_voxels) # # Filter the pRF mapping parameters and vertex indices
    vertices, ecc, angle = adjusting_verticesandecc(prf_voxels, prf_model, freesurfer_path, subj, hemi) # Adjust vertices and eccentricities so they match FreeSurfer surface space
    ecc_dict = dict(zip(vertices, ecc)) # Return the dictionary of vertex index and respective eccentricity 
    return ecc_dict