# Portions of this code were writen with the assistance of AI tools (Gemini)

import pandas as pd
import os

# 1. Define paths
# The directory where Script 1 saved the 14 base weather files
colocated_data_dir = "/home/csutter/DRIVE-clean/forecasting/coloc_hrrrdata/"

# The base directory where the new FH folders (FH02, FH03, etc.) will be created
output_base_dir = "/home/csutter/DRIVE-clean/forecasting/data_trackers_withhrrr/" 

FHs = ["02", "03", "04", "05", "06", "09", "12", "15", "18", "24", "30", "36", "42", "48"]

# List of all 30 trackers
trackers_list = [
"/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT5_m0_T0V1.csv",
"/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT4_m3_T5V0.csv",
"/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT5_m4_T4V0.csv",
"/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT3_m4_T2V4.csv",
"/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT5_m3_T3V4.csv",
"/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT4_m0_T0V1.csv",
"/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT2_m0_T0V1.csv",
"/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT5_m2_T2V3.csv",
"/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT0_m3_T4V5.csv",
"/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT0_m4_T5V1.csv",
"/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT1_m3_T5V0.csv",
"/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT0_m2_T3V4.csv",
"/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT4_m2_T2V3.csv",
"/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT3_m1_T1V2.csv",
"/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT1_m2_T4V5.csv",
"/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT2_m1_T3V4.csv",
"/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT0_m1_T2V3.csv",
"/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT4_m4_T3V5.csv",
"/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT2_m4_T1V3.csv",
"/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT2_m2_T4V5.csv",
"/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT1_m4_T0V2.csv",
"/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT3_m3_T5V0.csv",
"/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT3_m0_T0V1.csv",
"/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT5_m1_T1V2.csv",
"/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT1_m1_T3V4.csv",
"/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT2_m3_T5V0.csv",
"/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT4_m1_T1V2.csv",
"/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT1_m0_T2V3.csv",
"/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT3_m2_T4V5.csv",
"/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT0_m0_T1V2.csv",
]

# The single column that uniquely identifies an observation row
merge_key = 'img_orig'

# The new columns we want to bring over from Script 1
columns_to_add = ["hrrr_file_path", "t2m", "r2", "asnow", "tp", "tcc", "uavg"]

# 2. Loop through each Forecast Hour
for fh in FHs:
    print(f"--- Merging weather data for FH {fh} ---")
    
    # Create the specific directory for this Forecast Hour (e.g., .../FH09)
    fh_output_dir = os.path.join(output_base_dir, f"FH{fh}")
    os.makedirs(fh_output_dir, exist_ok=True)
    
    # Load the weather data prepared in Script 1 for this FH
    fh_weather_file = os.path.join(colocated_data_dir, f"labeleddata_FH{fh}.csv")
    fh_weather_df = pd.read_csv(fh_weather_file)
    
    # Keep only the identifying column and the new weather columns
    subset_weather_df = fh_weather_df[[merge_key] + columns_to_add]
    
    # 3. Loop through all 30 trackers
    for tracker_path in trackers_list:
        
        # Load the tracker
        tracker_df = pd.read_csv(tracker_path)
        
        # Extract the exact file name to use for saving (e.g., "nestcv_5cat_twotrain_OT5_m0_T0V1.csv")
        tracker_filename = os.path.basename(tracker_path)
        
        # Merge the data based on img_orig
        # This attaches the new weather columns to the right tracker rows
        merged_df = pd.merge(tracker_df, subset_weather_df, on=merge_key, how='left')
        
        # 4. Save the final file into the specific FH directory
        # The resulting path will look like: /home/.../coloc_hrrrdata/FH09/nestcv_5cat...csv
        output_path = os.path.join(fh_output_dir, tracker_filename)
        merged_df.to_csv(output_path, index=False)

print("Script 2 Complete! The 30 files are saved across the 14 FH directories.")