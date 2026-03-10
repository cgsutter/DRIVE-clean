# Written with the assistance of Gemini

# To run in slurm: clean_events_aggstats.sh

# See notebook for work: /home/csutter/DRIVE-clean/weather_events/notebooks/model_scratchwork.ipynb

import pandas as pd 
import numpy as np 
from glob import glob
import os
import geopandas as gpd

import csv
import json

from ast import literal_eval

from shapely import wkt

######## CONFIG

alldirs_data_preds = glob("/home/csutter/DRIVE-clean/operational_runs_QPEdata/data_odm_3_ensembling") #HERE!! Where model pred data lives
#### A - Typical run, considering all camera preds (no QPE filter)
# For 5cat model: glob("/home/csutter/DRIVE-clean/operational_runs/*/data_6_ensembling") 
# For obs model: glob("/home/csutter/DRIVE-clean/operational_runs/*/data_odm_3_ensembling") 
#### B - Filter to cam model preds only with active QPE. NOTE! To do this version, need to run this code first: /home/csutter/DRIVE-clean/weather_events/notebooks/cam_modelpred_QPE_active.py. This has already been ran for about 13k instances in runs dir, but for any new case inference runs we add, will need to run those too. 
# For 5cat model: glob("/home/csutter/DRIVE-clean/operational_runs_wQPE/data_6_ensembling") 
# For obs model: glob("/home/csutter/DRIVE-clean/operational_runs_wQPE/data_odm_3_ensembling") 

csv_path = "/home/csutter/DRIVE-clean/weather_events/models/stats_events_modelpred/stats_odm_QPEdata.csv" # HERE!!! Where to save out aggregated stats that this script produces. Should change this every time!
#### 5cat model was saved here:
# "/home/csutter/DRIVE-clean/weather_events/models/stats_events_modelpred/stats.csv"
#### ynobs model saved here:
# "/home/csutter/DRIVE-clean/weather_events/models/stats_events_modelpred/stats_odm.csv"
#### 5cat model w/ QPE filter:
# "/home/csutter/DRIVE-clean/weather_events/models/stats_events_modelpred/stats_wQPE.csv"
#### ynobs model saved here:
# "/home/csutter/DRIVE-clean/weather_events/models/stats_events_modelpred/stats_wQPE_odm.csv"

run_oneoff = False # HERE!! 
current_aggstats_tocheck_against = "/home/csutter/DRIVE-clean/weather_events/models/stats_events_modelpred/stats_wQPE.csv"

########### FOR RUN_ONEOFF

# run_oneoff is for situations when this agg stats script didn't grab stats for all situtaions , and need to check which ones didn't run. To fix this (which is in the main code below), we will grab the aggstats that we have already ran to cross check the dates that list and backfill those that are missing from aggstats

# --- MOVE THIS TO THE TOP (with file_lookup) ---
if run_oneoff:
    # Use a set for near-instant lookup speeds
    alreadyran_df = pd.read_csv(current_aggstats_tocheck_against)
    alreadyrandates = set(alreadyran_df["datetime"].astype(str))


########### NORMAL CODE STARTS HERE



# Need to read in multiple datasets: 1) NCEI geodataframe with geometries 2) Events csvs (with logic about every 15 min frequency, etc) and 3)  Model run data 

#### 1 - NCEI data
d_readin = gpd.read_file("/home/csutter/DRIVE-clean/weather_events/data/ncei_events/ncei_ny_events_clean.gpkg")

d_readin.head(4)

# add timedelta duration col back (using duration_sec col)
# note that you have to do this w/ every gdf
d_readin["duration"] = pd.to_timedelta(d_readin["duration_sec"], unit="s")

# may take ~40 seconds

######### LOOP 1: events_ofinterest

events_ofinterest_paths = ["/home/csutter/DRIVE-clean/weather_events/data/ncei_events_ofinterest/blizzard_allyrs_ceilfloor5min_nobuffer_freq5min.csv",
"/home/csutter/DRIVE-clean/weather_events/data/ncei_events_ofinterest/heavyrain_2425_ceilfloor15min_nobuffer_freq15min.csv",
"/home/csutter/DRIVE-clean/weather_events/data/ncei_events_ofinterest/heavysnow_2025_ceilfloor5min_nobuffer_freq5min.csv",
"/home/csutter/DRIVE-clean/weather_events/data/ncei_events_ofinterest/heavysnow_2425_ceilfloor15min_nobuffer_freq15min.csv",
"/home/csutter/DRIVE-clean/weather_events/data/ncei_events_ofinterest/lakeeffect_2025_ceilfloor15min_nobuffer_freq15min.csv",
"/home/csutter/DRIVE-clean/weather_events/data/ncei_events_ofinterest/lakeeffect_2025_ceilfloor30min_nobuffer_freq30min.csv",
"/home/csutter/DRIVE-clean/weather_events/data/ncei_events_ofinterest/lakeeffect_2425_ceilfloor15min_nobuffer_freq15min.csv",
"/home/csutter/DRIVE-clean/weather_events/data/ncei_events_ofinterest/winterstorm_2425_ceilfloor15min_nobuffer_freq15min.csv",
"/home/csutter/DRIVE-clean/weather_events/data/ncei_events_ofinterest/winterweather_2425_ceilfloor15min_nobuffer_freq15min.csv"]



# --- DO THIS ONCE AT THE TOP ---
# grab ALL model pred file paths
allfiles_data = []
for i in alldirs_data_preds:
    fs = glob(f"{i}/*/*/*/*/*")
    allfiles_data.extend(fs)

# Pre-parse the times into a dictionary
# The dict maps '20240217_1430' -> '/full/path/to/file.csv'
# The dict is used to search for the file path given a date that is needed for an event (Loop 3)
file_lookup = {}
for f in allfiles_data:
    beg = f.rfind("/")
    time_key = f[beg-13:beg]
    # log all string times of interest with the corresponding model pred file in dictionary
    file_lookup[time_key] = f

for ev in events_ofinterest_paths:
    #### 2 - Identified events
    # grab just one file for now, but will need to eventually tie in all of them

    events = pd.read_csv(ev)
    # convert this events df to geopandas df
    # Convert the 'geometry' column from strings to actual Shapely objects
    events['geometry'] = events['geometry'].apply(wkt.loads)
    # Cast as a GeoDataFrame and set the CRS
    # (Use the CRS of your original ncei_gdf, usually "EPSG:4269" for NWS data)
    events_gdf = gpd.GeoDataFrame(events, geometry='geometry', crs="EPSG:4269")
    # Match the points to the NCEI CRS
    # This converts the points' coordinates to fit the storm polygons perfectly
    events_gdf = events_gdf.to_crs(d_readin.crs)

    unique_eventids = np.unique(events_gdf["EVENT_ID"])
    print(unique_eventids[0:3])

    ######### LOOP 2: event_id WITHIN the events_ofinterest df

    for ev_id in unique_eventids:#unique_eventids:

        # ev_id = 1152731 # HERE!!

        # Run for ONE event (unique by EVENT_ID)

        # Will need to streamline this for the entire database of events that we have. But for now, to get code working, keep it simple and focus on one EVENT_ID

        ### Grab one event to work with
        events1 = events_gdf[events_gdf["EVENT_ID"]==ev_id] # one row of the df, one event ex.

        print(len(events1))

        ### For blizzards we ran these every 5 min (b/c it was a small number of events). All of these dates are already prepped in that df in the "all_times_format" column

        dur = events1["duration"].item()
        # print(dur)

        # read in the df col (which is a list) correctly as a list rather than str
        events1["all_times_format"] = events1["all_times_format"].apply(literal_eval)
        eventtimes = events1["all_times_format"].item()
        # print(eventtimes)
        # print(len(eventtimes)) 

        # the eventtimes list, made above, will be used to identify model runs that we have corresponding to those times. Need to search full directory of operational_runs for that (next code cell)


        ### Find all model run files that correspond to those dates from that event

        # allfiles_data = []
        # for i in alldirs_data_preds:
        #     fs = glob(f"{i}/*/*/*/*/*")
        #     for f in fs:
        #         allfiles_data.append(f)

        # print("check allfiles_data")
        # print(len(np.unique(allfiles_data)))
        # print(allfiles_data[0:3])


        ### Identify files that specifically align with event occurances
        # one list will grabs the preduiction file, the other list just tracks the time for the file (not sure if we'll need but might as well)
        # matched_cnn_file = [] 
        # matched_cnn_time = []
        # for model_f in allfiles_data:
        #     # parse just the time corresponding to the model pred
        #     beg = model_f.rfind("/")
        #     model_time = model_f[beg-13:beg]
        #     # # print(model_time)
        #     # see if that model_time exists in the list of event times
        #     # also just have to make sure we already didn't log that file (just in case was accidentally ran twice in operational_run, which can also happen just due to reorg in set__aggregate_allruns, so need to just use one csv pred)
        #     if ((model_time in eventtimes) & (model_time not in matched_cnn_time)):
        #         matched_cnn_time.append(model_time) # log the time that matches
        #         matched_cnn_file.append(model_f) # log the pred file that matches

        # # print examples and lengts
        # print(matched_cnn_time[0:4])
        # print(len(matched_cnn_time))

        # print(matched_cnn_file[0:4])
        # print(len(matched_cnn_file))

        # See how this compares to the total duration of times from eventtimes 
        # print(len(eventtimes))


        # --- INSIDE LOOP 2 --- ADJUSTED
        matched_cnn_file = []
        matched_cnn_time = []

        for etime in eventtimes:
            if etime in file_lookup: # check the dict we made
                if etime not in matched_cnn_time: # also just have to make sure we already didn't log that file (just in case was accidentally ran twice in operational_run, which can also happen just due to reorg in set__aggregate_allruns, so need to just use one csv pred)
                    matched_cnn_time.append(etime)
                    matched_cnn_file.append(file_lookup[etime])
        
        # Filter for one-offs if needed
        if run_oneoff:
            cnnfiles_filtered = []
            cnntimes_filtered = []
            for f, t in zip(matched_cnn_file, matched_cnn_time):
                if str(t) not in alreadyrandates:
                    cnnfiles_filtered.append(f)
                    cnntimes_filtered.append(t)
            
            # overwrite w/ just the files after checking against them
            matched_cnn_file = cnnfiles_filtered
            matched_cnn_time = cnntimes_filtered

            # IF NEED TO OVERWRITE COMPLETELY 
            # matched_cnn_file = ["/home/csutter/DRIVE-clean/operational_runs_wQPE/data_6_ensembling/2024/02/17/20240217_1430/finalpreds.csv"]
            # matched_cnn_time = ["20240217_1430"]

            print("after oneoff")
            print(len(matched_cnn_file))
            print(len(matched_cnn_time))


        # --- BEFORE LOOP 3 ---
        id_event = events1["EVENT_ID"].iloc[0]
        id_ep = events1["EPISODE_ID"].iloc[0]
        id_type = events1["EVENT_TYPE"].iloc[0]
        id_loc = events1["CZ_NAME"].iloc[0]
        id_wfo = events1["WFO"].iloc[0]

        # Pre-calculate the transformation logic (the "Projection") once
        # This avoids GeoPandas having to "figure out" the CRS mapping 50 times
        target_crs = events1.crs
                    

        # ########## FOR RUN_ONEOFF

        # if run_oneoff:

        #     print("before oneoff") 
        #     print(len(matched_cnn_file))
        #     print(len(matched_cnn_time))

        #     cnnfiles_oneoff = []
        #     cnntimes_oneoff = []

        #     alreadyran = pd.read_csv(current_aggstats_tocheck_against)
        #     print(type(alreadyran))
        #     alreadyrandates = list(alreadyran["datetime"])
        #     print(type(alreadyrandates))
        #     # print(alreadyrandates)

        #     for x in range(0, len(matched_cnn_file)):
        #         time_x = matched_cnn_time[x]
        #         file_x = matched_cnn_file[x]
        #         if ~(time_x in alreadyrandates):
        #             cnntimes_oneoff.append(time_x)
        #             cnnfiles_oneoff.append(file_x)

        #     # then overwrite matched_cnn_file and matched_cnn_time with the oneoff lists, so that the loops below work regardless of oneoff runs or not
        #     matched_cnn_file = cnnfiles_oneoff
        #     matched_cnn_time = cnntimes_oneoff

        #     matched_cnn_file = ["/home/csutter/DRIVE-clean/operational_runs_wQPE/data_6_ensembling/2024/02/17/20240217_1430/finalpreds.csv"]
        #     matched_cnn_time = ["20240217_1430"]

        #     print("after oneoff")
        #     print(len(matched_cnn_file))
        #     print(len(matched_cnn_time))

        


        # # ############## LOOP 3: MODEL FILES

        for i_modelfile in range(0, len(matched_cnn_file)):

            
            mf = matched_cnn_file[i_modelfile]
            mdate = matched_cnn_time[i_modelfile]
            # Now, using the relevant model files, for which we know there was XYZ event happening, need to correspond the pred observations of interest to where the event was spatially. This is the spatial join. Since we just simplified it to the EVENT_ID level right now, it's just one region, but note that EPISODE_ID is a weather system, wich will be helpful for visualizations later

            # For now, just focus on identifying the cam preds that are IN the weather event

            # For each model pred file, spatial join with the event location (geometry) in the events1 df

            # Start with one model file for example...

            cols_to_use = ["Latitude", "Longitude", "select", "img_name", "select_prob", "qpe_val"] # only keeping the cols we need makes the join much faster

            modelfile = pd.read_csv(mf, usecols=cols_to_use)

            # print(len(modelfile))

            # # print(list(modelfile.columns))

            modelfile_gdf = gpd.GeoDataFrame(
                modelfile, 
                geometry=gpd.points_from_xy(modelfile.Longitude, modelfile.Latitude),
                crs="EPSG:4326" # Start with standard GPS coordinates
            )
            # print(len(modelfile_gdf))


            modelfile_gdf = modelfile_gdf.to_crs(target_crs) # match the CRS to be exactly the system being used in the events dataset. We preloaded this prior to Loop 3 so that the same crs doesn't need to be found for each modelfile in that event

            # Perform the spatial join
            # INNER JOIN for now to limit the result to model preds that were WITHIN the event
            # This says: "For every point, find the row in ncei_gdf that contains it"
            joined_df = gpd.sjoin(
                modelfile_gdf, 
                events1, 
                how="inner",
                predicate="within" # Check if the point is WITHIN the polygon
            )

            # print(len(joined_df))
            # display(joined_df.head(4))


            #### THEN JUST LOG EVERYTHING

            # note that "select" is the model pred col w final model pred across the 5 ensemble members
            # ctsdf = joined_df[["select","img_name"]].groupby(["select"]).count().reset_index()

            # ctsdf = ctsdf.rename(columns = {"img_name":"count"})

            # countdict = ctsdf.set_index("select")["count"].to_dict()

            # display(ctsdf.head(4))
            # print(countdict)

            ### ---- ADJUSTED FROM ABOVE LOGGING CODE

            # 1. Group by the predicted class ('select'), which is the model pred col w final model pred across the 5 ensemble members
            # We count 'img_name' and take the mean of all our probability columns. Note that in step 3, we'll grab just the prob_snow associated with that the snow predictions, etc. 
            agg_results = joined_df.groupby("select").agg({
                "img_name": "count",
                "select_prob": "mean"
            }).rename(columns={"img_name": "count", "select_prob": "avg_predprob"})

            # 2. Create the Counts Dictionary
            countdict = agg_results["count"].to_dict()

            # 3. Create the Probabilities Dictionary
            probdict = {
                pred_class: round(avg, 3) 
                for pred_class, avg in agg_results["avg_predprob"].items()
            }

            # other info needed in addition to the id_event etc created above
            datetime_stamp = mdate 

            # grab the avg QPE - which is ONE stat for this event & datetime (one avg value for this row of agg stat)
            avgqpe = np.mean(joined_df["qpe_val"])

            # Your data
            data_to_log = {
                "id_event": id_event, #id_event,
                "id_ep": id_ep,
                "id_type": id_type,
                "id_loc":id_loc,
                "id_wfo":id_wfo,
                "datetime":datetime_stamp,
                "model_counts": countdict, # This will be stringified
                "model_probs": probdict,
                "qpe_avg":avgqpe
            }

            # Pre-process: Convert the nested dict to a JSON string
            data_to_log["model_counts"] = json.dumps(data_to_log["model_counts"])

            data_to_log["model_probs"] = json.dumps(data_to_log["model_probs"])

            # Check if file exists to determine if we need a header
            file_exists = os.path.isfile(csv_path)

            with open(csv_path, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data_to_log.keys())
                
                # Write header only if the file is being created for the first time
                if not file_exists:
                    writer.writeheader()
                    
                writer.writerow(data_to_log)

            print(f"Logged event {data_to_log['id_event']} to {csv_path}")