from imports import * 

def histogram(data, columns, group_col, out_dir, tag):
    # Plot histograms of the connective field size and variance explained per visual area or hemisphere ans save the .png file
    for col in columns: # Loop through each column which are the two variables we are plotting, connective field size and variance explained
        plt.figure(figsize=(8, 5)) # Create a new figure
        sns.histplot(data=data, x=col, hue=group_col, kde=True, element="step")
        # Add plot details: title, axis labels, limits, grid
        plt.title(f"Histogram of the {col} by {group_col.capitalize()}") # Title 
        plt.xlabel(col) # X label title 
        plt.ylabel("Frequency") # Y label title 
        plt.ylim(0, 750) # Y-axis range of the values
        plt.grid(True)
        plt.tight_layout()
        out_path = f"{out_dir}/histogram_{tag}_{col}_{group_col}.png"
        plt.savefig(out_path, dpi=150)
        plt.close() 

def plot_variance_vs_ecc(data, out_dir, tag):
    # Plot the connective field variance explained across the eccentricity values 
    plt.figure(figsize=(10, 6)) # Create a new figure for the plot
    # Scatterplot of variance explained vs source eccentricity
    sns.scatterplot(data, x="Source Eccentricity", y="Best Variance Explained Finer",hue="visual_area", alpha=0.4)
    for area in sorted(data["visual_area"].unique()): # For each visual area, fit and plot a trend line
        subset = data[data["visual_area"] == area] # take only this area
        sns.regplot(data=subset, x="Source Eccentricity", y="Best Variance Explained Finer", scatter=False, label=f"{area} trend") # Run regression
    plt.title("Variance Explained vs Eccentricity") # Figure title 
    plt.xlabel("Source Eccentricity (deg)") # X label title 
    plt.ylabel("Best Variance Explained Finer") # Y label title
    plt.legend(title="Visual Area", fontsize=8) # Legend 
    plt.grid(True) 
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"VEversusEccentricity_{tag}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

def cf_vs_benson_ecc_plots(data, subj, out_dir, tag, ve_threshold=0.0):
    # Compare eccentricity values from the connective field model based on the pRF mapping against the one based on the benson atlas
    base = os.path.join(MAIN_PATH, "freesurfer", subj, "label") # Load the Benson atlas label files containing vertex indices and eccentricity values 
    # Build paths to left- and right-hemisphere label files for this subject
    lh = benson_label_to_dict(os.path.join(base, "lh.benson14_eccen-0001.label"), value_name="eccentricity")
    rh = benson_label_to_dict(os.path.join(base, "rh.benson14_eccen-0001.label"), value_name="eccentricity")

    # Add to the data frame the eccentricity value. based on the benson labels 
    values = []
    for index, row in df.iterrows(): # Loop over the data frame row by row
        idx = row.get("Target Vertex Index") # Get the target vertex index for this row
        hemi = row.get("hemisphere", "lh") # Get the hemisphere of the row  
        if hemi == "lh": # If the row is from the left hemisphere grab the Benson eccentricity for that vertex index from the left hemisphere’s map
            values.append(lh.get(idx, np.nan)) # Otherwise store NaN
        else: # Otherwise, grab it from the right hemisphere’s map.
            values.append(rh.get(idx, np.nan)) # Otherwise store NaN
    data["Benson Eccentricity"] = pd.to_numeric(values, errors="coerce") # Add these values as a new columns in the data frame 
    # Keep the connective fields above the variance explained threshold
    data = data[data["Best Variance Explained Finer"].fillna(-1) >= ve_threshold] # And also replaces the missing values with -1 so they don't pass the filter
    results = []  # This will store summary counts for each area

    # Plots per visual area 
    for area in sorted(data["visual_area"].unique()):
        # Only keep rows where both eccentricities are defined
        area_data = data[data["visual_area"] == area].dropna(subset=["Benson Eccentricity", "Source Eccentricity"])
        # Includ only the vertices where the Benson label eccentricity value is less than or equal to the maximum eccentricity valeus that we estimated in the model
        excluded = area_data[area_data["Benson Eccentricity"] > benson_max]
        included = area_data[area_data["Benson Eccentricity"] <= benson_max]
        # Store counts in results table
        results.append({"visual_area": area, "n_total": len(area_data), f"n_benson_≤{benson_max}": len(included), f"n_benson_>{benson_max}": len(excluded)})

        # Create the scatterplot 
        plt.figure(figsize=(6, 6))
        # Plot only included vertices (included from where?)
        plt.scatter(included["Benson Eccentricity"], included["Source Eccentricity"], s=10, alpha=0.6, label=f"Included (≤{benson_max}): {len(included)}")
        plt.plot([0, benson_max], [0, benson_max], "k--", label="Identity Line") # Plot the identity line
        plt.xlabel("Benson Atlas based Eccentricity (deg)") # X label title 
        plt.ylabel("pRF mapping based Eccentricity (deg)") # Y label title 
        plt.title(f"Connective fields Eccentricity comparison — {area}") # Figure label title 
        plt.axis("square") # Make the x and y axis equalr 
        plt.grid(True)
        plt.legend(fontsize=8)
        plt.tight_layout()
        out_file = os.path.join(out_dir, f"BensonversusPRF_Eccentricity_{tag}_{area}.png")
        plt.savefig(out_file, dpi=150)
        plt.close()

    return pd.DataFrame(results)

# Main script 
if __name__ == "__main__":
    subj = sys.argv[1] # Get the subject number 
    atlas = atlases[0] # Choose the atlas used for the delineation of the ROIs
    task = tasks[0] # Choose the task of the aquisition 
    denoising = denoising_methods[0] # Choose the denoising method applied to the time course  
    base_dir = os.path.join(MAIN_PATH, "CFM", subj, ses, atlas, task, denoising, "GM") # Extract the directory where to find the best fit file
    out_dir = os.path.join(MAIN_PATH, "CFM", "results", "dataquality", subj) # Output directory for the data quality plots
    os.makedirs(out_dir, exist_ok=True) # Make sure these directories are applied 
    roi_dict = dict(rois_list) # Turn the list of ROIs in the config file into a dictionary {1:'V1', 2:'V2'}
    src_name = roi_dict[source_visual_area] # Look up the source area's name
    areas = [f"{roi_dict[t]}-{src_name}" for t in target_visual_areas if t != source_visual_area] # For each target area number, look up its name in the dictionary and skips the source area 

    # Load the best file file 
    best_fits = [] # Collect dataframes
    for hemi in hemispheres: # Loop through each hemisphere
        for area in areas: # Loop through each visual area
            path = os.path.join(base_dir, hemi, area, "best_fits_prf.csv") # Build the file path for this hemisphere and area
            try:
                df = pd.read_csv(path) # Read the CSV into a DataFrame
                df["hemisphere"], df["visual_area"] = hemi, area # Add extra columns to keep track of where the data came from
                best_fits.append(df) # Store the dataframe in the list
            except:  
                print(f"NO FILE: {path}")
    best_fit = pd.concat(best_fits, ignore_index=True) if best_fits else pd.DataFrame()

    columns = ["Best Sigma Finer","Best Variance Explained Finer", "Source Eccentricity", "Target Eccentricity", "Target Vertex Index"] # Columns in the best fit files
    for c in columns: # Loop over each column
        if c in best_fit.columns: # Only convert if the column actually exists in best_fit
            best_fit[c] = pd.to_numeric(best_fit[c], errors="coerce") # Force the column values to be numbers and convert the strings to NaNs

    # Plot the histogram of the variance explained and the connective field size per visual area but also per hemispheres
    histogram(best_fit, ["Best Sigma Finer","Best Variance Explained Finer"], "visual_area", out_dir, f"{subj}_{atlas}_{task}_{denoising}")
    histogram(best_fit, ["Best Sigma Finer","Best Variance Explained Finer"], "hemisphere", out_dir, f"{subj}_{atlas}_{task}_{denoising}")
    # Plot the connective field variance explained across the eccentricity values 
    plot_variance_vs_ecc(best_fit, out_dir, f"{subj}_{atlas}_{task}_{denoising}") 
    # Plot for the connective field eccentricity based on the manual delineation versus the ones based on the benson atlas delineation of the ROIs
    counts_table = cf_vs_benson_ecc_plots(data=best_fit, subj=subj, out_dir=out_dir, tag=f"{subj}_{atlas}_{task}_{denoising}", ve_threshold=0.0)