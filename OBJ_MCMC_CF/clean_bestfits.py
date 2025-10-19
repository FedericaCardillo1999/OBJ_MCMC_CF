from imports import *
denoising = denoising_methods[0]
sessions = [ses]
atlases = atlases
 

def pRF_values(subj, hemi, atlas, task, denoising, MAIN_PATH, freesurfer_path, has_multiple_runs):
    # Get the eccentricity and polar angle values from the pRF mapping results or from the Benson label 
    if atlas == "manual": # If we are using the manual delineation of the ROIs
        # Extract Source Eccentricity and Source Polar Angle for a given subject and hemisphere from the pRF model pickle file
        ecc_map, ang_map = pRFparams_frompkl(subj, hemi, MAIN_PATH, atlas, denoising, task, freesurfer_path)
    else: # If we are using the Benson Atlas delineation of the ROIs
        label_dir = os.path.join(freesurfer_path, subj, "label") # Directory where the Benson Atlas labels are stored
        ecc_file = os.path.join(label_dir, f"{hemi}.benson14_eccen-0001.label") # Label with the eccentricity values per vertex
        ang_file = os.path.join(label_dir, f"{hemi}.benson14_angle-0001.label") # Label with the polar angle values per vertex
        ecc_map = benson_label_to_dict(ecc_file, value_name="eccentricity") # Extract the eccentricity values 
        ang_map = benson_label_to_dict(ang_file, value_name="angle") # Extract the polar angle values
    return ecc_map, ang_map

def fix_header(path, expected_column='Target Vertex Index'):
    with open(path, 'r') as f: # Open the CSV file and read all lines into a list
        lines = f.readlines()

    header_idx = None # Initialize the index of the first header row as None
    for idx, line in enumerate(lines): # Loop over each line with its index
        if expected_column in line: # If the expected column name is in this line this becomes the header row
            header_idx = idx
            break  # Go ahead 
    header = lines[header_idx] # Save the first header line

    # If there are any other lines that contain the other repeated headers skip it and build the data rows
    data = [line for i, line in enumerate(lines[header_idx + 1:], start=header_idx + 1) if expected_column not in line]
    csv_content = ''.join([header] + data) # Reconstruct the clean csv with one header and all valid data rows
    return pd.read_csv(io.StringIO(csv_content)) # Use StringIO to treat the string as a file-like object then read it into a pandas DataFrame

def pRFparams_frompkl(subj, hemi, main_path, atlas, denoising, task, freesurfer_path):
    # Extract Source Eccentricity and Source Polar Angle for a given subject and hemisphere from the pRF model pickle file
    try:
        # Load pRF parameters from pickle
        prf_params, prf_voxels = load_prf(subj, main_path, atlas, denoising, task)
        prf_model = filter_prf(prf_params, prf_voxels)
    except FileNotFoundError:
        print(f"NO pRF  for {subj}")
        return {}, {}

    vertices, ecc_values, angle_values = adjusting_verticesandecc( prf_voxels, prf_model, freesurfer_path, subj, hemi)
    ecc_dict = dict(zip(vertices, ecc_values))
    angle_dict = dict(zip(vertices, angle_values))
    return ecc_dict, angle_dict

# Clean the best fit files in case there are some problems with the headers 
for subj in subjects: # Loop over all the subjects
    for atlas in atlases: # Loop over all the atlases 
        for task in tasks: # Loop overl all the tasks 
            # RestingState may have run-1/run-2; if not, use the plain path
            runs_to_use = [str(r) for r in runs] if (task == "RestingState" and has_multiple_runs) else [""]
            for run in runs_to_use: # Loop over all the runs, if specified 
                for hemi in hemispheres: # Loop over all the hemispheres 
                    # Extract the pRF parameters values from the pRF mapping results 
    
                    ecc_map, ang_map = pRF_values(subj, hemi, atlas, task, denoising, MAIN_PATH, freesurfer_path, has_multiple_runs)
                    
                    visual_areas = [roi_name for roi_name, _ in rois_list] # Loops over each tuple in the list of ROIs and collects just the first element the ROI name 
                    for target in visual_areas:  # Skip V1 as target
                        source_target = f"{target}-V1"
                        # Find the connective field model results file
                        candidates = [] # Build possible candidate paths for the file containing the results of the connective field modeling 
                        if task == "RestingState": # Consider only the Resting State task 
                            if project == "UMCG":
                                for ses in sessions: # Loop over all the possible sessions 
                                    if run: # If there are multiple runs and there IS a run folder 
                                        candidates.append(os.path.join(MAIN_PATH, "CFM", subj, ses, atlas, task, f"run-{run}", denoising,"GM", hemi, source_target, "best_fits.csv"))
                                    # # If there are not multiple runs and IS NOT a run folder 
                                    candidates.append(os.path.join(MAIN_PATH, "CFM", subj, ses, atlas, task, denoising,"GM", hemi, source_target, "best_fits.csv" ))
                            else: # For OVGU and RS-7T projects the session needs to be specified in the configuration file 
                                # For OVGU can be either ses-01, ses-02 and ses-03
                                # FOR RS-7T only ses-01
                                candidates.append(os.path.join(MAIN_PATH, "CFM", subj, ses, atlas, task, denoising,"GM", hemi, source_target, "best_fits.csv" ))
                        else: # Consider the Retintotopic mapping task, RET and RET2 
                            if project == "UMCG":
                                candidates.append(os.path.join(MAIN_PATH, "CFM", subj, "ses-02", atlas, task, denoising, "GM", hemi, source_target, "best_fits.csv"))
                            else: # For OVGU and RS-7T projects the session needs to be specified in the configuration file 
                                # For OVGU can be either ses-01, ses-02 and ses-03
                                # FOR RS-7T only ses-01
                                candidates.append(os.path.join(MAIN_PATH, "CFM", subj, ses, atlas, task, denoising,"GM", hemi, source_target, "best_fits.csv" ))
                        best_fit_path = next((p for p in candidates if os.path.exists(p)), None) # Pick the first existing candidate
                        if not best_fit_path: # Print if there is not best fit file for that subject and which path was tried
                            print(f"MISSING BEST_FIT here {best_fit_path}")
                            continue

                        # Due to parallelization, sometimes the header of the best fit is broked
                        # e.g. It starts directly with numerical values and the header in in the second row
                        # e.i. The header is repeated for the first and thrid row 
                        repeated = best_fit['Target Vertex Index'] == 'Target Vertex Index' # Detect whether the header is present in multiple rows
                        if repeated.any(): # If there are any fix the best fit file 
                            best_fit = fix_header(best_fit_path)
                            best_fit.to_csv(best_fit_path, index=False) # And overwrite it with the incorrect one

                        # Map for each connective field center the corresponding polar angle and eccentricity 
                        src_col = 'Source Vertex Index' # Get the vertex of the connective field center
                        best_fit['Source Eccentricity'] = best_fit[src_col].map(ecc_map) # Add the corresponding Eccentricity value 
                        best_fit['Source Polar Angle'] = best_fit[src_col].map(ang_map) # Add the corresponding Polar Angle value
                        output_path = os.path.join(os.path.dirname(best_fit_path), 'best_fits_prf.csv')
                        best_fit.to_csv(output_path, index=False)