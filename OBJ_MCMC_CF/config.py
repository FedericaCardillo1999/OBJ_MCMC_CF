from imports import *

project = 'RS-7T'  # Name of your project e.g. "RS-7T", "OVGU", "UMCG"
PROJECT_PATH = f'/Users/federicacardillo/Documents/EGRETAAA/{project}'
MAIN_PATH = f'/Users/federicacardillo/Documents/EGRETAAA/{project}/derivatives' # The main path where are stored the freesurfer and retintotopy results directory 
freesurfer_path = f"{MAIN_PATH}/freesurfer" 
ses = 'ses-01' # The session where the retinotopy results are acquired
cutoff_volumes = 8  # The number of initial volumes to discard of the task
hemispheres = ['rh', 'lh']  # The hemisphere to process e.g. "rh", "lh"
source_visual_area = 1  # The number from the freesurfer's label of the source visual area e.g. for V1 = 1, V2 = 2, V3 = 3
target_visual_areas = [2]  # The number from the freesurfer's label of the target visual area
rois_list = [('V2', 2), ('V3', 3), ('V4', 4)]  # THE ROIs name and the bnumber from freesurfer's label of the target visual area e.g.  ('V1', 1), ('V2', 2), ('V3', 3)
ncores = 12  # Number of CPU cores to use for parallel processing
atlases = ['benson']  # The ROIs delineation to use e.g. "manual", "benson", "bayesian_benson (still under development)"
filter_source = False  # To exclude noisy vertices close to the fovea, it is possible to filter out vertices in the source area that have an eccentricity value below 0.5 based on the pRF mapping results. e.g. "True" to remove those vertices, "False" to use the whole source area
tasks = ['RestingState']  # The tasks to run the modeling on e.g. "RestingState", "RET", "RET2 (monocular for glaucoma and simulated scotoma for the healthy controls)"
denoising_methods = ['nordic_sm4']  # The denoising method used on the data e.g. "nordic", "nordic_sm34"
# In case of Resting State scans some datasets run the scan with multiple runs
# If the runs are in the same conditions, e.g. all eyes closed or all eyes open then has_multiple_runs = False. The code will store the results under the same directory 
# If the runs are in different conditions, e.g. one eyes closed and another one eyes open then has_multiple_runs = True. The code will store the results under the different directories e.g. "/run-1", "/run-2" 
has_multiple_runs = False 
runs = [1, 2]  # If the runs are in different conditions, then specify the numbers of runs collected in your datasets to create the specific subdirectories. 
benson_max = 25.0  # When running the code based on the ROIs delineation of the benson atlas you can specify the maximum eccentricity value of the vertices to be included in the analysis 
# The degrees of eccentricity of the retintopic mapping stimulus (which might vary per project!)
max_eccentricity = 10.0 if project == 'OVGU' else 7.0 if project == 'UMCG' else None
groups = {"HC": [f"sub-{i:02}" for i in range(1, 13)], "POAG": [f"sub-{i:02}" for i in range(14, 36)]} if project == "OVGU" else {"HC": [f"sub-{i:02}" for i in range(1, 20)], "POAG": [f"sub-{i:02}" for i in range(21, 46)]} if project == "UMCG" else {"HC": ["sub-01", "sub-02"], "POAG": []}
all_subjects = [d for d in os.listdir(PROJECT_PATH) if d.startswith("sub-") and os.path.isdir(os.path.join(PROJECT_PATH, d))]
subjects = {sub: {"path": os.path.join(PROJECT_PATH, sub)} for sub in all_subjects}