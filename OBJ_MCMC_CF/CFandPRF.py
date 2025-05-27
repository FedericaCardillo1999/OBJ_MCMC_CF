import os
import numpy as np
import pickle
from nibabel.freesurfer import read_morph_data
import pandas as pd 

class PRFModel:
    def __init__(self, r2, size, ecc, angle):
        self.r2 = r2
        self.size = size
        self.ecc = ecc
        self.angle = angle

def pickle_file(filepath):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, 'rb') as file:
        return pickle.load(file)

def load_prf(subj, main_path, atlas, denoising, task):
    if task == 'RestingState':
        task_for_file = 'RET'
    else:
        task_for_file = task

    tried_paths = []
    for ses in ['ses-01', 'ses-02']:
        filepath = os.path.join(
            main_path,
            f'pRFM/{subj}/{ses}/{denoising}/model-{atlas}-nelder-mead-GM_desc-prf_params_{task_for_file}.pkl'
        )
        tried_paths.append(filepath)
        if os.path.exists(filepath):
            with open(filepath, 'rb') as file:
                pkl_data = pickle.load(file)
            prf_params = pkl_data['model'].iterative_search_params
            prf_voxels = np.where(pkl_data['rois_mask'] == 1)[0]
            return prf_params, prf_voxels

    print(f"Tried paths for {subj}:")
    for path in tried_paths:
        print(f" - {path}")
    raise FileNotFoundError(f"pRF .pkl file not found for subject {subj} in any session.")

def filter_prf(prf_params, prf_voxels):
    return PRFModel(
        r2=prf_params[:, 7],
        size=prf_params[:, 2],
        ecc=np.sqrt(prf_params[:, 1]**2 + prf_params[:, 0]**2),
        angle=np.arctan2(prf_params[:, 1], prf_params[:, 0])
    )


def source_eccentricity(subj, hemi, main_path, atlas, denoising, task, freesurfer_path, label_file=None):
    prf_params, prf_voxels = load_prf(subj, main_path, atlas, denoising, task)
    prf_model = filter_prf(prf_params, prf_voxels)

    lh_c = read_morph_data(os.path.join(freesurfer_path, subj, 'surf', 'lh.curv'))
    numel_lh = lh_c.shape[0]

    if hemi == 'rh':
        adjusted_voxels = prf_voxels[prf_voxels >= numel_lh] - numel_lh
        ecc = prf_model.ecc[prf_voxels >= numel_lh]
    else:
        adjusted_voxels = prf_voxels[prf_voxels < numel_lh]
        ecc = prf_model.ecc[prf_voxels < numel_lh]

    ecc_dict = dict(zip(adjusted_voxels, ecc))

    if atlas == "benson":
        if label_file is None:
            raise ValueError("Label file must be provided for Benson atlas.")
        
        df = pd.read_csv(label_file, sep='\s+', skiprows=2, header=None)
        df.columns = ['vertex', 'x', 'y', 'z', 'value']
        benson_vertices = df['vertex'].values

        matching_vertices = [v for v in benson_vertices if v in ecc_dict]
        print(f"Total Benson vertices: {len(benson_vertices)}")
        print(f"Vertices matching pRF data: {len(matching_vertices)}")
        print(f"Example: {[(v, round(ecc_dict[v], 2)) for v in matching_vertices[:5]]}")

    return ecc_dict


def source_eccentricity_benson(subj, hemi, main_path, atlas, denoising, task, freesurfer_path, label_file):
    """
    Load eccentricity from pRF .pkl file, filter to vertices from the label file (Benson),
    and return only those with eccentricity ≤ 10 degrees.
    Returns a dictionary {vertex_index: eccentricity}.
    """
    # Step 1: Load pRF fits from the .pkl
    prf_params, prf_voxels = load_prf(subj, main_path, atlas, denoising, task)
    prf_model = filter_prf(prf_params, prf_voxels)

    lh_c = read_morph_data(os.path.join(freesurfer_path, subj, 'surf', 'lh.curv'))
    numel_lh = lh_c.shape[0]

    # Adjust voxel indices for hemisphere
    if hemi == 'rh':
        adjusted_voxels = prf_voxels[prf_voxels >= numel_lh] - numel_lh
        ecc = prf_model.ecc[prf_voxels >= numel_lh]
    else:
        adjusted_voxels = prf_voxels[prf_voxels < numel_lh]
        ecc = prf_model.ecc[prf_voxels < numel_lh]

    # Build full eccentricity dictionary from .pkl
    ecc_dict = dict(zip(adjusted_voxels, ecc))

    # Step 2: Load label vertices (Benson ROI mask)
    if label_file is None:
        raise ValueError("Label file must be provided for Benson atlas.")

    df = pd.read_csv(label_file, sep='\s+', skiprows=2, header=None)
    df.columns = ['vertex', 'x', 'y', 'z', 'value']
    benson_vertices = df['vertex'].values

    # Step 3: Filter ecc_dict to only vertices in the label, and ecc ≤ 10
    filtered_dict = {v: ecc_dict[v] for v in benson_vertices if v in ecc_dict and ecc_dict[v] <= 10}

    print(f"Total Benson label vertices: {len(benson_vertices)}")
    print(f"Matching vertices with pRF eccentricity ≤ 10: {len(filtered_dict)}")
    if filtered_dict:
        sample = list(filtered_dict.items())[:5]
        print("Example vertex/eccentricity:", [(v, round(e, 2)) for v, e in sample])

    return filtered_dict
