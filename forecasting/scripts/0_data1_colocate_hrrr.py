# Portions of this code were writen with the assistance of AI tools (Gemini)

import pandas as pd
import numpy as np
import os
from datetime import timedelta
from scipy.spatial import cKDTree
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc

# =========================================
# CONFIG: Change this for each manual run!
# =========================================
CURRENT_FH = "48"
# =========================================

# 1. THE LEAN WORKER FUNCTION
# This runs outside the main thread, opening files in parallel
def fetch_weather_for_time(valid_time, obs_indices, original_indices, fh, hrrr_base_dir, columns_to_load):
    valid_time_dt = pd.to_datetime(valid_time)
    init_time_dt = valid_time_dt - timedelta(hours=int(fh))
    
    init_yr = init_time_dt.strftime("%Y")
    init_mo = init_time_dt.strftime("%m")
    init_day = init_time_dt.strftime("%d")
    init_hr = init_time_dt.strftime("%H")
    
    hrrr_filename = f"{init_yr}{init_mo}{init_day}_hrrr.t{init_hr}z_{fh}.parquet"
    hrrr_filepath = os.path.join(hrrr_base_dir, init_yr, init_mo, hrrr_filename)
    
    if os.path.exists(hrrr_filepath):
        try:
            hrrr_data = pd.read_parquet(hrrr_filepath, columns=columns_to_load)
            
            # Using raw arrays to keep memory overhead near zero
            matched_hrrr = hrrr_data.iloc[obs_indices].reset_index(drop=True)
            matched_hrrr.index = original_indices
            matched_hrrr['hrrr_file_path'] = hrrr_filepath
            
            del hrrr_data
            return matched_hrrr
        except Exception as e:
            return None
    return None

# 2. MAIN EXECUTION THREAD
if __name__ == '__main__':
    
    labeled_data_path = "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT5_m0_T0V1.csv"
    output_dir = "/home/csutter/DRIVE-clean/forecasting/coloc_hrrrdata/"
    hrrr_base_dir = "/home/csutter/AI2ES/cleaned/HRRR/"

    weather_vars = ["t2m", "r2", "asnow", "tp", "tcc", "u10", "v10"]
    columns_to_load = weather_vars + ["valid_time"] 

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=== Starting Parallel Run for Forecast Hour: {CURRENT_FH} ===")
    base_df = pd.read_csv(labeled_data_path)
    
    raw_datetime_str = (
        base_df['yr'].astype(str) + '-' + 
        base_df['mo'].astype(str) + '-' + 
        base_df['day'].astype(str) + ' ' + 
        base_df['time'].astype(str)
    )
    base_df['valid_datetime'] = pd.to_datetime(raw_datetime_str).dt.floor('H')

    print("Building spatial map...")
    reference_file = "/home/csutter/AI2ES/cleaned/HRRR/2022/01/20220101_hrrr.t11z_02.parquet"
    ref_hrrr = pd.read_parquet(reference_file, columns=["latitude", "longitude"])
    hrrr_coords = np.column_stack((ref_hrrr['latitude'], ref_hrrr['longitude']))
    tree = cKDTree(hrrr_coords)
    obs_coords = np.column_stack((base_df['Latitude'], base_df['Longitude']))
    distances, base_df['hrrr_idx'] = tree.query(obs_coords)
    
    del ref_hrrr
    gc.collect()
    print("Spatial map built.")

    fh_df = base_df.copy()
    results_list = []
    futures = {}
    
    # 3. OPEN THE PARALLEL WORKERS
    # We use 4 workers to read 4 files at the exact same time
    with ProcessPoolExecutor(max_workers=4) as executor:
        for valid_time, group in fh_df.groupby('valid_datetime'):
            obs_indices = group['hrrr_idx'].values
            original_indices = group.index.values
            
            future = executor.submit(
                fetch_weather_for_time, valid_time, obs_indices, original_indices, CURRENT_FH, hrrr_base_dir, columns_to_load
            )
            futures[future] = valid_time
            
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Matching HRRR for FH {CURRENT_FH}"):
            result = future.result()
            if result is not None:
                results_list.append(result)

    print("\nCombining weather data into the main dataset...")
    if results_list:
        all_new_data = pd.concat(results_list)
        fh_df['hrrr_file_path'] = all_new_data['hrrr_file_path']
        for var in weather_vars:
            fh_df[var] = all_new_data[var]
            
    fh_df['hrrr_file_path'] = fh_df['hrrr_file_path'].fillna("Not Found")
    
    fh_df['uavg'] = np.sqrt(fh_df["u10"] ** 2 + fh_df["v10"] ** 2)
    fh_df = fh_df.drop(columns=["u10", "v10"])
    
    cols_to_drop = [col for col in ['innerPhase', 'outerPhase', 'valid_datetime', 'hrrr_idx'] if col in fh_df.columns]
    fh_df = fh_df.drop(columns=cols_to_drop)
    
    output_filename = f"labeleddata_FH{CURRENT_FH}.csv"
    output_full_path = os.path.join(output_dir, output_filename)
    fh_df.to_csv(output_full_path, index=False)
    
    print(f"Finished FH {CURRENT_FH}! Exiting to safely clear all memory.")