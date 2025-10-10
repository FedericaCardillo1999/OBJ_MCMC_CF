from imports import *

project = 'UMCG'  # Project name; 'RS-7T', 'EGRET+', 'OVGU'
MAIN_PATH = f'/Users/federicacardillo/Documents/EGRETAAA/{project}/derivatives'
ses = 'ses-02'
cutoff_volumes = 8  # Number of initial volumes to discard 
hemispheres = ['rh', 'lh']  # Hemispheres to process; ['lh'], ['rh'], ['lh', 'rh']
source_visual_area = 1  # Numeric label for the source visual area (1 = V1)
target_visual_areas = [2]  # List of numeric labels for target visual areas
rois_list = [('V2', 2)]  # ROI name and numeric label pairs for reference
load_one = None  # Load only one Target Vertex, mainly used for debugging
ncores = 8  # Number of CPU cores to use for parallel processing
processing_method = "zscore"  # Preprocessing method; 'zscore', 'percent', 'raw'
atlases = ['manual']  # Atlases to use; ['benson', 'manual']
filter_v1 = True  # Filter vertices from the source area V1, currenlty to filter out vertices with an eccentricity lower than 0.5
tasks = ['RET']  # Tasks to process; ['RestingState', 'RET', 'RET2']
denoising_methods = ['nordic']  # Denoising methods to use; 
has_multiple_runs = False
runs = [1, 2]  # Number of runs from the Resting State scan to process, in case one is eye opens and other one is eyes closed
benson_max = 25.0  # Maximum Benson eccentricity value to include when using the Label Benson values
if project == 'OVGU':
    benson_max_stimulus = 10.0  # Maximum Benson eccentricity value to include when using the pRF Benson values
else: 
    benson_max_stimulus = 7.0
filter_benson = False  # False if you want to use the whole Label Benson values, True if you want to use the filtered pRF Benson values
benson_mode = "only_labels" # or labels+pkl, only_labels
if project == 'OVGU':
    groups = {"HC": [f"sub-{i:02}" for i in range(1, 13)],"POAG": [f"sub-{i:02}" for i in range(14, 36)]}
else:
    groups = {"HC": [f"sub-{i:02}" for i in range(1, 20)],"POAG": [f"sub-{i:02}" for i in range(21, 46)]}
