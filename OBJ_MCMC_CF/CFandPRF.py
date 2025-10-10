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

def read_benson_eccentricity(label_file):
    df = pd.read_csv(label_file, sep=r"\s+", skiprows=2, header=None)
    df.columns = ["vertex", "x", "y", "z", "eccentricity"]
    vertices = df["vertex"].astype(int).to_numpy()
    # Extract arrays of vertex indices and eccentricities
    ecc = df["eccentricity"].to_numpy()
    return dict(zip(vertices, ecc))

def load_prf(subj, main_path, atlas, denoising, task):
    task_for_file = "RET" if task == "RestingState" else task

    for ses in ["ses-01", "ses-02", "ses-03"]:
        # Build the path to the pickle file
        filepath = os.path.join(main_path, f"pRFM/{subj}/{ses}/{denoising}/model-{atlas}-nelder-mead-GM_desc-prf_params_{task_for_file}.pkl")
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                pkl = pickle.load(f)
            prf_params = pkl["model"].iterative_search_params # Extract pRF parameters (shape: voxels x parameters)
            prf_voxels = np.where(pkl["rois_mask"] == 1)[0] # Extract voxel indices where the ROI mask equals 1
            return prf_params, prf_voxels

    raise FileNotFoundError(f"No pRF file found here {main_path}/pRFM/{subj}/<ses>/{denoising}/")

def filter_prf(prf_params, prf_voxels):
    x = prf_params[:, 0] # Extract x-coordinate of receptive field centers 
    y = prf_params[:, 1] # Extract y-coordinate of receptive field centers
    size = prf_params[:, 2]
    r2 = prf_params[:, 7]
    ecc = np.sqrt(x**2 + y**2) # Compute eccenricity
    angle = np.arctan2(y, x) # Compute polar angle (
    return PRFModel(r2=r2, size=size, ecc=ecc, angle=angle)

def adjusting_verticesandecc(prf_voxels, prf_model, freesurfer_path, subj, hemi):
    lh_curv_path = os.path.join(freesurfer_path, subj, "surf", "lh.curv") # Path to left hemisphere curvature file
    lh_curv = read_morph_data(lh_curv_path) # Read number of vertices in the left hemisphere 
    n_lh = lh_curv.shape[0]
    # For right hemisphere: voxel indices are offset by n_lh so subtract n_lh to get RH vertex indices
    if hemi == "rh":
        mask = prf_voxels >= n_lh
        adjusted_vertices = prf_voxels[mask] - n_lh
        ecc_values = prf_model.ecc[mask]
    else:
        mask = prf_voxels < n_lh # For left hemisphere: use indices directly (less than n_lh)
        adjusted_vertices = prf_voxels[mask]
        ecc_values = prf_model.ecc[mask]

    return adjusted_vertices, ecc_values

def source_eccentricity(subj, hemi, main_path, atlas, denoising, task, freesurfer_path, label_file=None, benson_fallback_ecc_max=None):
    # Try to load eccentricities from pRF (preferred path)
    prf_loaded = False
    try:
        prf_params, prf_voxels = load_prf(subj, main_path, atlas, denoising, task)
        prf_model = filter_prf(prf_params, prf_voxels)
        vertices, ecc = adjusting_verticesandecc(prf_voxels, prf_model, freesurfer_path, subj, hemi)
        ecc_dict = dict(zip(vertices, ecc))  # {vertex: ecc}
        prf_loaded = True
    except FileNotFoundError as e:
        # If pRF is missing and atlas isn't Benson, that's an error
        if atlas != "benson":
            raise
        ecc_dict = {} # If atlas is Benson try the fallback below

    # Benson fallback: if no pRF, use label-derived eccentricities up to a cutoff
    if atlas == "benson" and not prf_loaded:
        if label_file is None:
            raise ValueError("Benson fallback requires an eccentricity label file")
        label_ecc = read_benson_eccentricity(label_file)
        ecc_dict = {v: e for v, e in label_ecc.items() if e <= benson_fallback_ecc_max}
        return ecc_dict

    # if pRF loaded and a label file is available, compute overlap (no print here)
    if atlas == "benson" and prf_loaded and label_file is not None:
        try:
            benson_vertices = read_benson_label_vertices(label_file)
            matching = [v for v in benson_vertices if v in ecc_dict]
        except FileNotFoundError:
            pass

    return ecc_dict