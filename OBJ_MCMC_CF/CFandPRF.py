import os
import numpy as np
import pickle
from nibabel.freesurfer import read_morph_data

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

    filepath = os.path.join(
        main_path,
        f'pRFM/{subj}/ses-02/{denoising}/model-manual-nelder-mead-GM_desc-prf_params_{task_for_file}.pkl')
    pkl_data = pickle_file(filepath)
    prf_params = pkl_data['model'].iterative_search_params
    prf_voxels = np.where(pkl_data['rois_mask'] == 1)[0]
    return prf_params, prf_voxels

def filter_prf(prf_params, prf_voxels):
    return PRFModel(
        r2=prf_params[:, 7],
        size=prf_params[:, 2],
        ecc=np.sqrt(prf_params[:, 1]**2 + prf_params[:, 0]**2),
        angle=np.arctan2(prf_params[:, 1], prf_params[:, 0]))

def source_eccentricity(subj, hemi, main_path, atlas, denoising, task, freesurfer_path):
    prf_params, prf_voxels = load_prf(subj, main_path, atlas, denoising, task)
    prf_model = filter_prf(prf_params, prf_voxels)

    lh_c = read_morph_data(os.path.join(freesurfer_path, subj, 'surf', 'lh.curv'))
    numel_lh = lh_c.shape[0]

    if hemi == 'rh':
        adjusted_voxels = prf_voxels[prf_voxels >= numel_lh] - numel_lh
        ecc = prf_model.ecc[prf_voxels >= numel_lh]
    else:
        adjusted_voxels = prf_voxels
        ecc = prf_model.ecc

    return dict(zip(adjusted_voxels, ecc))