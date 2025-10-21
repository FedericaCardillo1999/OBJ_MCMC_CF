from imports import * 
import sys
from pathlib import Path
from config import (MAIN_PATH,ses,hemispheres,atlases,tasks,denoising_methods,source_visual_area,target_visual_areas,rois_list,has_multiple_runs,runs,)

def roi_num_to_name(roi_num):
    """Map numeric ROI (1,2,3,4,...) to the label ('V1','V2',...)."""
    for name, num in rois_list:
        if num == roi_num:
            return name
    return f"V{roi_num}"

def build_area_pairs():
    """
    Build area pairs like 'V2-V1', 'V3-V1' using source_visual_area and target_visual_areas.
    We skip pairs where target == source (e.g., V1-V1).
    """
    src = roi_num_to_name(source_visual_area)
    pairs = []
    for tgt in target_visual_areas:
        tgt_name = roi_num_to_name(tgt)
        if tgt_name != src:
            pairs.append(f"{tgt_name}-{src}")
    return pairs

def drop_repeated_headers(df):
    """Remove rows where any cell equals its column name (repeated header inside file)."""
    if df.empty:
        return df
    bad = pd.Series(False, index=df.index)
    for col in df.columns:
        bad |= df[col].astype(str) == str(col)
    return df[~bad].copy()

def load_bestfits_csv(path):
    """
    Load a best_fits_prf.csv using pandas, drop repeated headers, and coerce key columns to numeric.
    Returns an empty DataFrame if missing or unreadable.
    """
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[Warning] Could not read CSV: {path} ({e})")
        return pd.DataFrame()

    df = drop_repeated_headers(df)

    for col in [
        "Target Vertex Index",
        "Source Eccentricity",
        "Source Polar Angle",
        "Variance Explained",
        "CF Sigma",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep rows with a valid target index
    if "Target Vertex Index" in df.columns:
        df = df[df["Target Vertex Index"].notna()]
    return df

def save_maps(subj, hemi, run_tag, atlas, task, ecc_arr, pol_arr, ve_arr, sigma_arr, results_dir, freesurfer_root):
    """Write morphs to FreeSurfer and copies to results (plus .npy)."""
    run_suffix = f"_{run_tag}" if run_tag else ""

    # FreeSurfer
    write_morph_data(f"{freesurfer_root}/{subj}/surf/{hemi}.ecc_{denoising}{run_suffix}",   ecc_arr)
    write_morph_data(f"{freesurfer_root}/{subj}/surf/{hemi}.pol_{denoising}{run_suffix}",   pol_arr)
    write_morph_data(f"{freesurfer_root}/{subj}/surf/{hemi}.ve_{denoising}{run_suffix}",    ve_arr)
    write_morph_data(f"{freesurfer_root}/{subj}/surf/{hemi}.sigma_{denoising}{run_suffix}", sigma_arr)

    # Results
    base = f"{hemi}.{{kind}}_{atlas}_{task}{(f'_run-{run_tag}' if run_tag else '')}_{denoising}.surf"
    ecc_out   = os.path.join(results_dir, base.format(kind="ecc"))
    pol_out   = os.path.join(results_dir, base.format(kind="pol"))
    ve_out    = os.path.join(results_dir, base.format(kind="ve"))
    sigma_out = os.path.join(results_dir, base.format(kind="sigma"))

    write_morph_data(ecc_out,   ecc_arr);  
    write_morph_data(pol_out,   pol_arr);   
    write_morph_data(ve_out,    ve_arr);  
    write_morph_data(sigma_out, sigma_arr); 

# ---------------------------
# 3) Main
# ---------------------------
if __name__ == "__main__":
    # Subject from CLI (e.g., 'sub-01')
    if len(sys.argv) < 2:
        print("Usage: python visualfieldmaps.py sub-XX")
        sys.exit(1)
    subj = "sub-" + sys.argv[1]


    # Build config-driven items
    denoising = denoising_methods[0] if len(denoising_methods) > 0 else "nordic_sm4"
    area_pairs = build_area_pairs()
    freesurfer_root = f"{MAIN_PATH}/freesurfer"

    # Results directory
    results_dir = os.path.join(MAIN_PATH, "CFM", "results", "visualfieldmaps", subj)
    os.makedirs(results_dir, exist_ok=True)

    # Iterate over config atlases, tasks, hemispheres
    for atlas in atlases:
        for task in tasks:
            # Run handling:
            # - If has_multiple_runs and task is RestingState, use config.runs (e.g., [1,2])
            # - Otherwise a single empty run tag "" (keeps your filenames the same as before)
            run_list = [str(r) for r in runs] if (has_multiple_runs and task == "RestingState") else [""]

            for run_tag in run_list:
                for hemi in hemispheres:
                    # --- Derive the number of vertices from FreeSurfer curvature file ---
                    curv_path = f"{freesurfer_root}/{subj}/surf/{hemi}.curv"
                    curv = read_morph_data(curv_path)
                    n_vertices = curv.shape[0]

                    # Initialize arrays with default value 50.0 (meaning "no data")
                    hemi_ecc   = np.full(n_vertices, 50.0)
                    hemi_pol   = np.full(n_vertices, 50.0)
                    hemi_ve    = np.full(n_vertices, 50.0)
                    hemi_sigma = np.full(n_vertices, 50.0)

                    # Fill arrays from each target area pair (e.g., 'V2-V1', 'V3-V1', ...)
                    for pair in area_pairs:
                        # Path to cleaned CSV:
                        # MAIN_PATH/CFM/<subj>/<ses>/<atlas>/<task>[/run-X]/<denoising>/GM/<hemi>/<pair>/best_fits_prf.csv
                        if task == "RestingState" and has_multiple_runs and run_tag:
                            csv_path = f"{MAIN_PATH}/CFM/{subj}/{ses}/{atlas}/{task}/run-{run_tag}/{denoising}/GM/{hemi}/{pair}/best_fits_prf.csv"
                        else:
                            csv_path = f"{MAIN_PATH}/CFM/{subj}/{ses}/{atlas}/{task}/{denoising}/GM/{hemi}/{pair}/best_fits_prf.csv"
                        print("MAIN_PATH:", MAIN_PATH)
                        print("Example path:", f"{MAIN_PATH}/CFM/sub-02/ses-01/benson/RestingState/nordic_sm4/GM/rh/V2-V1/best_fits_prf.csv")
                        if not os.path.exists(csv_path):
                            print(f"[Missing] {csv_path}")
                            continue

                        df = load_bestfits_csv(csv_path)

                        # Valid indices only
                        tgt_idx = pd.to_numeric(df["Target Vertex Index"], errors="coerce")
                        valid_mask = tgt_idx.notna() & (tgt_idx >= 0) & (tgt_idx < n_vertices)
                        idx = tgt_idx[valid_mask].astype(int).values
                        sub = df.loc[valid_mask].copy()

                        # Fill maps if columns exist
                        if "Source Eccentricity" in sub.columns:
                            hemi_ecc[idx] = pd.to_numeric(sub["Source Eccentricity"], errors="coerce").fillna(50.0).values
                            print(hemi_ecc[idx])
                        if "Source Polar Angle" in sub.columns:
                            hemi_pol[idx] = pd.to_numeric(sub["Source Polar Angle"], errors="coerce").fillna(50.0).values
                        if "Variance Explained" in sub.columns:
                            hemi_ve[idx] = pd.to_numeric(sub["Variance Explained"], errors="coerce").fillna(50.0).values
                        if "CF Sigma" in sub.columns:
                            hemi_sigma[idx] = pd.to_numeric(sub["CF Sigma"], errors="coerce").fillna(50.0).values
                    print("Total vertices:", n_vertices)
                    print("Vertices with data:", len(idx))
                    print("Min/Max values:", hemi_ecc.min(), hemi_ecc.max())
                    # Save the four maps for this hemisphere/run
                    save_maps(subj, hemi, run_tag, atlas, task, hemi_ecc, hemi_pol, hemi_ve, hemi_sigma, results_dir, freesurfer_root)

                    print(f"Saved VF maps for {subj} {atlas} {task} {hemi} run='{run_tag or ''}'")
                    print(f" -> FreeSurfer: {freesurfer_root}/{subj}/surf/")
                    print(f" -> Results:    {results_dir}/")