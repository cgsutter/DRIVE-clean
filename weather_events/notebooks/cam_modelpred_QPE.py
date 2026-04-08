# Code written with assistance of Gemini

# To run: from slurm, see clean_qpe_a100_1.sh. Note that we have clean_qpe_a100_1.sh in order to run odm code too. 
# One off runs from terminal in vs code, run:
# /home/csutter/miniconda3/bin/python /home/csutter/DRIVE-clean/weather_events/notebooks/cam_modelpred_QPE_active.py 
# CRITICAL note ^!!!! To run from miniconda 

# Our main aggregate code in the stats python scripts can't simply add the additional step of pulling the xarray QPE data for every event and timestep bc it takes ~15-20 sec ... w/ 96k rows in the events stats, it would take roughly 22 days! Need to do some data prep to only do this ONCE per every modelpred file, that way we're not repeating the operation for events (which are split by regions) for which many will have overlapping times. 

# This code is essentially data preprocessing to 1) for every model pred file that exists in /home/csutter/DRIVE-clean/operational_runs, download the MRMS QPE file and the MRMS precip flag file which contains p type, 2) save out the subsetted model preds withthis data into a new dir: /home/csutter/DRIVE-clean/operational_runs_wMRMS

# Note that if we run more operational model pred runs, will have to come back to this code and run it again. It is already written to NOT run on files that it's already been ran on, though, so this script can be ran as is at any point. Only the first run takes ~13-18 hours. 

# About MRMS data: 
# https://www.nssl.noaa.gov/projects/mrms/operational/tables.php
# E.g. https://noaa-mrms-pds.s3.amazonaws.com/index.html#CONUS/PrecipFlag_00.00/
# QPE: https://noaa-mrms-pds.s3.amazonaws.com/index.html#CONUS/PrecipFlag_00.00/
# The QPE files that we are using are PrecipRate_00.00, which is the instantaneous rate of precipitation, measured in mm/hour. So it is instantaneous, but what that means is - If the snow/rain falling at this exact second continued for a full hour, it would accumulate X millimeters of water equivalent.
# Precip flag: instantaneous classification - what type of hydrometeor the radar thinks is hitting the ground at that exact minute.
# MRMS files available every 2 minutes. 

# w cpu-per-task of 16 and mem-per-cpu of 4gb, set workers = 12
# w/ parallelization, takes ~ 5 sec / run


import xarray as xr
import requests
import gzip
import os
import pandas as pd
import numpy as np
from glob import glob
import time
import gc
from concurrent.futures import ThreadPoolExecutor

# =====================================================================
# 1. CONFIGURATION & DIRECTORIES
# =====================================================================

# Define input directory (where model predictions currently live)
dir_ofpreds = "/home/csutter/DRIVE-clean/operational_runs/*/data_6_ensembling"
# data_odm_3_ensembling
# data_6_ensembling

# Define output directory (where subsetted/updated predictions will be saved)
newdir = "/home/csutter/DRIVE-clean/operational_runs_wMRMS/data_6_ensembling" # data_6_ensembling or data_odm_3_ensembling

# Define parallelization workers (12 is optimal for 16 cpu / 4gb mem per cpu)
MAX_WORKERS = 12

# =====================================================================
# 2. FILE DISCOVERY & FILTERING
# =====================================================================

alldirs_data_preds = glob(dir_ofpreds) 

allfiles_data = []
allfiles_tosaveto = [] 

# Gather all prediction files
for i in alldirs_data_preds:
    fs = glob(f"{i}/*/*/*/*/*")
    for f in fs:
        allfiles_data.append(f)
        # Create the corresponding output path by replacing the base directory
        newsave = f.replace(i, newdir)
        allfiles_tosaveto.append(newsave)

print(f"Total model files found: {len(allfiles_data)}")

# Filter lists to only include files we haven't processed yet
matched_cnn_file = [] 
matched_cnn_time = []
tosave_modelpred_wQPE = [] 

for ind in range(len(allfiles_data)):
    model_f = allfiles_data[ind]
    
    # Parse the timestamp string (e.g., '20250412_0000') from the file path
    beg = model_f.rfind("/")
    model_time = model_f[beg-13:beg]
    
    # Check if the output file already exists to avoid redundant processing
    filterqpe_exist = os.path.isfile(allfiles_tosaveto[ind])

    if (model_time not in matched_cnn_time) and (not filterqpe_exist):
        matched_cnn_time.append(model_time) 
        matched_cnn_file.append(model_f) 
        tosave_modelpred_wQPE.append(allfiles_tosaveto[ind])

# --- COPY THIS BLOCK ---
total_found = len(allfiles_data)
total_to_process = len(matched_cnn_file)
total_skipped = total_found - total_to_process

print(f"Found {total_found} total files.")
print(f"Skipped {total_skipped} files (duplicates or already processed... could be because there are duplicate model pred csvs (and we only need one), or could be because the version with the MRMS has already been ran and exists in the dir already.")
print(f"Processing {total_to_process} new files.")

# print(matched_cnn_file)
# -----------------------

# =====================================================================
# 3. CORE FUNCTIONS
# =====================================================================

def get_mrms_s3(ts, product_type, worker_index):
    """
    Downloads and subsets MRMS data from S3.
    - ts: pandas Timestamp (UTC)
    - product_type: "PrecipRate" (QPE) or "PrecipFlag" (Precip Type)
    - worker_index: Unique integer to prevent temp file collisions between threads
    """
    date_str = ts.strftime('%Y%m%d')
    time_str = ts.strftime('%H%M') + "00"
    
    # Build S3 URL
    base_url = f"https://noaa-mrms-pds.s3.amazonaws.com/CONUS/{product_type}_00.00"
    file_name = f"MRMS_{product_type}_00.00_{date_str}-{time_str}.grib2.gz"
    full_url = f"{base_url}/{date_str}/{file_name}"
    
    r = requests.get(full_url)

    if r.status_code == 200:
        # Create a thread-safe temporary filename using the worker_index
        temp_grib = f"temp_{product_type}_{ts.strftime('%Y%m%d%H%M%S')}_idx{worker_index}.grib2" 
        
        # Save the decompressed GRIB2 file to disk
        with open(temp_grib, "wb") as f:
            f.write(gzip.decompress(r.content))

        # Open dataset and immediately subset to NY bounding box to save RAM
        ds_full = xr.open_dataset(temp_grib, engine="cfgrib", backend_kwargs={'indexpath': ''})
        ds = ds_full.sel(latitude=slice(47.5, 38.5), longitude=slice(278.0, 291.0)).load()
        
        # Safely close the full dataset to release memory locks
        ds_full.close()
        del ds_full
        
        # Cleanup: Delete the physical .grib2 and associated .idx files
        if os.path.exists(temp_grib):
            os.remove(temp_grib)
            for f in glob(f"{temp_grib}*.idx"):
                os.remove(f)
        
        return ds
    else:
        # Usually 403 or 404 means the file doesn't exist for that specific minute
        return None

def execute_for_datetime(predpath, time_str, saveto, worker_index):
    """
    Reads a single camera prediction file, fetches both MRMS datasets, 
    maps them to the cameras via spatial join, and saves the updated CSV.
    """
    # 1. Load the specific camera predictions
    cam_df = pd.read_csv(predpath)

    # 2. Time Alignment (Round down to nearest even 2-minute mark for MRMS)
    raw_ts = pd.to_datetime(time_str, format="%Y%m%d_%H%M")
    even_min = (raw_ts.minute // 2) * 2
    mrms_ts = raw_ts.replace(minute=even_min, second=0)

    # 3. Fetch BOTH QPE Rate and Precipitation Flag datasets
    qpe_ds = get_mrms_s3(mrms_ts, "PrecipRate", worker_index)
    flag_ds = get_mrms_s3(mrms_ts, "PrecipFlag", worker_index)

    # Proceed only if both datasets successfully downloaded
    if qpe_ds and flag_ds:
        
        # Format longitude to match MRMS 0-360 scale
        cam_df['Lon_360'] = cam_df['Longitude'] % 360
        
        # Convert pandas columns to xarray DataArrays for fast vectorized extraction
        lats = xr.DataArray(cam_df['Latitude'], dims="cams")
        lons = xr.DataArray(cam_df['Lon_360'], dims="cams")
        
        # Dynamically grab the variable names to avoid cfgrib 'unknown' naming issues
        var_qpe = list(qpe_ds.data_vars)[0]
        var_flag = list(flag_ds.data_vars)[0]
        
        # 4. Nearest Neighbor Extraction
        # Extracts the specific pixel values for every camera simultaneously
        qpe_values = qpe_ds[var_qpe].sel(latitude=lats, longitude=lons, method="nearest").values
        flag_values = flag_ds[var_flag].sel(latitude=lats, longitude=lons, method="nearest").values
        
        # 5. Assign extracted values back to the DataFrame
        cam_df['qpe_val'] = qpe_values
        cam_df['qpe_val'] = cam_df['qpe_val'].clip(lower=0) # Clean up missing/negative flags
        cam_df['precip_flag'] = flag_values
        
        # Map the integer to the string classification ---
        flag_dict = {
            -3: "No Coverage",
             0: "No Precip", 
             1: "Warm Rain", 
             3: "Snow", 
             6: "Convective Rain", 
             7: "Hail/Mixed", 
            10: "Cold Rain",
            91: "Tropical Stratiform",
            96: "Tropical Convective"
        }
        
        # Map it, and fill any weird/missing values with "Unknown"
        cam_df['precip_class'] = cam_df['precip_flag'].map(flag_dict).fillna("Unknown")
        
        # 6. Save out the updated DataFrame
        os.makedirs(os.path.dirname(saveto), exist_ok=True)
        cam_df.to_csv(saveto, index=False)

        print(f"Worker {worker_index} successfully processed and saved {time_str}")

        # 7. Aggressive RAM Cleanup (Crucial for 13k parallel loops)
        qpe_ds.close()
        flag_ds.close()
        del qpe_ds, flag_ds
        gc.collect()

# =====================================================================
# 4. PARALLEL EXECUTION
# =====================================================================

print(f"Starting parallel processing with {MAX_WORKERS} workers...")

# Wrapper function to pass the index (i) and the lists into the executor
def thread_wrapper(i):
    try:
        execute_for_datetime(
            predpath=matched_cnn_file[i], 
            time_str=matched_cnn_time[i], 
            saveto=tosave_modelpred_wQPE[i],
            worker_index=i  # Passes the unique index for thread safety
        )
    except Exception as e:
        print(f"Error processing index {i} (File: {matched_cnn_file[i]}): {e}")

start_batch = time.time()

# Execute mapping across the pool
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    executor.map(thread_wrapper, range(len(matched_cnn_file)))

end_batch = time.time()
print("=========================================")
print(f"ALL DONE")
print(f"Total time for {len(matched_cnn_file)} files: {end_batch - start_batch:.2f}s")