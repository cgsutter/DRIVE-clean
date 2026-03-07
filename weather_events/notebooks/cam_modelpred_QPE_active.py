# Code written with assistance of Gemini

# Our main aggregate code in the stats python scripts can't simply add the additional step of pulling the xarray QPE data for every event and timestep bc it takes ~15-20 sec ... w/ 96k rows in the events stats, it would take roughly 22 days!

# Need to do some data prep to only do this ONCE per every modelpred file, that way we're not repeating the operation for events (which are split by regions) for which many will have overlapping times. 

# This code is essentially data preprocessing to 1) for every model pred file that exists in /home/csutter/DRIVE-clean/operational_runs, subset the model pred file to only contain cams that had active QPE (rather than all 2700 live cams) 2) save out the subsetted model preds with "active QPE" into a new dir: /home/csutter/DRIVE-clean/operational_runs_wQPE

# Note that since I don't have code built into the operational model run, if i run more case studies, will need to specifially come here and run this code for the new operational runs

# w cpu-per-task of 16 and mem-per-cpu of 4gb, set workers = 12
# w/ parallelization, takes ~ 3-4s/run

# To run: from slurm, see clean_qpe_a100_1.sh
# One off runs from terminal in vs code, run:
# /home/csutter/miniconda3/bin/python /home/csutter/DRIVE-clean/weather_events/notebooks/cam_modelpred_QPE_active.py

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

# First, grab all datetime model pred files that exist in our entire directory

dir_ofpreds = "/home/csutter/DRIVE-clean/operational_runs/*/data_6_ensembling" #HERE!! 
# new dir to save subsetted QPE model preds 
newdir = "/home/csutter/DRIVE-clean/operational_runs_wQPE/data_6_ensembling" # HERE!! 
# For both of the above, Update with whether running 5-cat model or obs model
# For 5cat model: "/home/csutter/DRIVE-clean/operational_runs/*/data_6_ensembling"
# For obs model: "/home/csutter/DRIVE-clean/operational_runs/*/data_odm_3_ensembling"


alldirs_data_preds = glob(dir_ofpreds) 

allfiles_data = []
allfiles_tosaveto = [] 
for i in alldirs_data_preds:
    fs = glob(f"{i}/*/*/*/*/*")
    for f in fs:
        allfiles_data.append(f)
        # prepare dir full path to save to
        newsave = f.replace(i, newdir)
        allfiles_tosaveto.append(newsave)

print(len(allfiles_data))
print(len(allfiles_tosaveto))


# one list will grabs the preduiction file, the other list just tracks the time for the file (not sure if we'll need but might as well)
matched_cnn_file = [] 
matched_cnn_time = []
tosave_modelpred_wQPE = [] # where to save out the subsetted CNN df after tying in QPE data

for ind in range(0, len(allfiles_data)):
    model_f = allfiles_data[ind]
    # parse just the time corresponding to the model pred
    beg = model_f.rfind("/")
    model_time = model_f[beg-13:beg]
    # # print(model_time)
    # see if that model_time exists in the list of event times
    # also just have to make sure we already didn't log that file (just in case was accidentally ran twice in operational_run, which can also happen just due to reorg in set__aggregate_allruns, so need to just use one csv pred)

    # OPTION 0 [from initial run, don't need to run this version again (unless changing the QPE threshold or something)]
    # COMMENT OUT!
    # if model_time not in matched_cnn_time:
    #     matched_cnn_time.append(model_time) # log the time that matches
    #     matched_cnn_file.append(model_f) # log the pred file that matches
    #     tosave_modelpred_wQPE.append(allfiles_tosaveto[ind])

    # OPTION 2 - for when needing to find which model runs from the main dir of preds have not yet been ran w/ QPE subset. E.g. need to do this after running new operational set
    # Check if the file already exists in the QPE directory. 
    filterqpe_exist = os.path.isfile(allfiles_tosaveto[ind])

    if ((model_time not in matched_cnn_time) & (filterqpe_exist == False)):
        matched_cnn_time.append(model_time) # log the time that matches
        matched_cnn_file.append(model_f) # log the pred file that matches
        tosave_modelpred_wQPE.append(allfiles_tosaveto[ind])


print(len(matched_cnn_file))
print(len(matched_cnn_time))
print(len(np.unique(matched_cnn_time)))

print(matched_cnn_file[0:3])
print(matched_cnn_time[0:3])

# # # FOR TESTING SMALL AMOUNT FOR PARALLELIZING
# # # matched_cnn_file = matched_cnn_file[50:100]
# # # matched_cnn_time = matched_cnn_time[50:100]
# # # tosave_modelpred_wQPE = tosave_modelpred_wQPE[50:100] # where to save out the subsetted CNN df after tying in QPE data



# # # Define function that pulls in QPE data, which we execute in teh loop code below


def get_qpe_s3(ts):
    """
    ts: pandas Timestamp (UTC)
    """
    # 1. Format the S3 URL for NOAA Open Data
    # Path: https://noaa-mrms-pds.s3.amazonaws.com/CONUS/PrecipRate_00.00/YYYYMMDD/MRMS_PrecipRate_00.00_YYYYMMDD-HHMMSS.grib2.gz
    date_str = ts.strftime('%Y%m%d')
    # MRMS files are every 2 mins. Let's force it to 00 minutes for the test.
    time_str = ts.strftime('%H%M') + "00"
    
    base_url = "https://noaa-mrms-pds.s3.amazonaws.com/CONUS/PrecipRate_00.00"
    file_name = f"MRMS_PrecipRate_00.00_{date_str}-{time_str}.grib2.gz"
    full_url = f"{base_url}/{date_str}/{file_name}"
    
    print(f"Attempting S3 Download: {full_url}")
    
    # 2. Download
    r = requests.get(full_url)

    # if r.status_code == 200:
    #     temp_grib = "test_qpe.grib2"
    #     with open(temp_grib, "wb") as f:
    #         # We decompress the .gz before saving
    #         f.write(gzip.decompress(r.content))

    #  2. download (updating to before parallelizing)
    if r.status_code == 200:
        # Use a unique name for each thread based on the timestamp string
        temp_grib = f"temp_{ts.strftime('%Y%m%d%H%M%S')}.grib2" 
        with open(temp_grib, "wb") as f:
            f.write(gzip.decompress(r.content))

        # 3. Open and IMMEDIATELY subset to NY box
        ds_full = xr.open_dataset(temp_grib, engine="cfgrib", backend_kwargs={'indexpath': ''})
        
        # New York roughly: Lat 40-45.5, Lon 280-290
        # Subsetting here shrinks the data in RAM by ~90%
        # ds = ds_full.sel(latitude=slice(46, 39), longitude=slice(279, 291)).load()
        # Expanded Box: Covers NY + NJ, PA, CT, and Southern Canada
        ds = ds_full.sel(latitude=slice(47.5, 38.5), longitude=slice(278.0, 291.0)).load()
        
        # CLOSE the file handle (Critical!)
        ds_full.close()
        del ds_full
        
        # NOW delete the physical files from the hard drive
        if os.path.exists(temp_grib):
            os.remove(temp_grib)
            
            # cfgrib creates index files that look like 'temp_123.grib2.923a8.idx'
            # We can use a wildcard to catch any .idx file associated with this temp file
            import glob
            for f in glob.glob(f"{temp_grib}*.idx"):
                os.remove(f)
        
        return ds

    else:
        print(f"Failed. Status Code: {r.status_code}")
        if r.status_code == 403 or r.status_code == 404:
            print("Check: Ensure the time ends in a multiple of 2 (e.g., 02, 04, 10).")
        return None


# Execute

def execute_for_datetime(predpath = matched_cnn_file[0], time_str = matched_cnn_time[0], saveto = tosave_modelpred_wQPE[0], THRESHOLD = 0.1):

    # for logging time to execute..
    start_time = time.time()

    cam_df = pd.read_csv(predpath)
    # # 1. SETUP PARAMETERS
    # time_str = "20250412_0015"  # Your input format
    # THRESHOLD = 0.1             # mm/hr (The "Active" cutoff)

    # 2. TIME ALIGNMENT (The 2-Minute Even Rule)
    # Convert string to Timestamp
    raw_ts = pd.to_datetime(time_str, format="%Y%m%d_%H%M")

    # MRMS files are every 2 mins (usually even). Round down to be safe.
    even_min = (raw_ts.minute // 2) * 2
    mrms_ts = raw_ts.replace(minute=even_min, second=0)

    print(f"Original Time: {raw_ts}")
    print(f"Targeting MRMS File: {mrms_ts.strftime('%Y%m%d-%H%M00')}")

    # 3. GET DATA (Using your existing function)
    qpe_ds = get_qpe_s3(mrms_ts)

    if qpe_ds:
        # Rename 'unknown' if it exists to make it easier to read
        if 'unknown' in qpe_ds.variables:
            qpe_ds = qpe_ds.rename({'unknown': 'precip_rate'})

        # 4. THE JOIN (Mapping Cameras to Grid)
        # Ensure Longitude is 0-360 for MRMS
        cam_df['Lon_360'] = cam_df['Longitude'] % 360
        
        # Create xarray DataArrays for vectorized "sampling"
        # This grabs the QPE value for ALL cameras at once
        lats = xr.DataArray(cam_df['Latitude'], dims="cams")
        lons = xr.DataArray(cam_df['Lon_360'], dims="cams")
        
        # Extract values using nearest neighbor
        qpe_values = qpe_ds.precip_rate.sel(
            latitude=lats, 
            longitude=lons, 
            method="nearest"
        ).values
        
        # 5. ASSIGN RESULTS
        cam_df['qpe_val'] = qpe_values
        # Clean up -999 (missing) or negative values
        cam_df['qpe_val'] = cam_df['qpe_val'].clip(lower=0)
        
        # Create the Boolean Filter
        cam_df['is_active'] = cam_df['qpe_val'] >= THRESHOLD
        
        # Decided to run QPE analysis for all cams, not to artifically discard them based on threshold. Now we'll have mm/hr for all cams and can set whatever threshold we want, or include all cams and just do something w/ average QPE. Either way, don't need to subset here. 
        # print(f"Processing complete.")
        # print(f"Active Cameras (>{THRESHOLD}mm/hr): {cam_df['is_active'].sum()} / {len(cam_df)}")
        # # subset to QPE active
        # cam_QPE_active = cam_df[cam_df['is_active'] == True ]
        # print(len(cam_QPE_active))

        # save out NEED TO ADD THIS
        print(saveto)
        os.makedirs(os.path.dirname(saveto), exist_ok=True)
        cam_df.to_csv(saveto)

        qpe_ds.close() # Close the xarray dataset
        del qpe_ds     # Remove the variable from the namespace
        gc.collect()   # Force the system to actually free the RAM


        # Log
        end_time = time.time()
        duration = end_time - start_time
        # print(f"Finished in: {duration:.2f} seconds")




# Define how many files to download at once. 
MAX_WORKERS = 12

print(f"Starting parallel processing with {MAX_WORKERS} workers...")

# We use a wrapper to pass the multiple lists into the function
def thread_wrapper(i):
    try:
        execute_for_datetime(
            predpath=matched_cnn_file[i], 
            time_str=matched_cnn_time[i], 
            saveto=tosave_modelpred_wQPE[i]
        )
    except Exception as e:
        print(f"Error processing index {i}: {e}")

# This is the engine that runs the parallelization
start_batch = time.time()

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    # We pass the indices of your matched_cnn_file list
    executor.map(thread_wrapper, range(len(matched_cnn_file)))

end_batch = time.time()
print(f"ALL DONE ")
print(f"Total time for {len(matched_cnn_file)} files: {end_batch - start_batch:.2f}s")

# TO EXECUTE FOR ONE SINGLETON
# execute_for_datetime()