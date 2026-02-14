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

    ######### LOOP 2: event_id WITHIN the events_ofinterest df

    for ev_id in unique_eventids:

        # Run for ONE event (unique by EVENT_ID)

        # Will need to streamline this for the entire database of events that we have. But for now, to get code working, keep it simple and focus on one EVENT_ID

        ### Grab one event to work with
        events1 = events_gdf[events_gdf["EVENT_ID"]==ev_id] # one row of the df, one event ex.

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
        # Note: will need to repeat this for ODM since that may be beneficial for snow squall/blizzards

        alldirs_data_preds = glob("/home/csutter/DRIVE-clean/operational_runs/*/data_odm_3_ensembling") #HERE!! Update with whether running 5-cat model or obs model
        # For 5cat model: glob("/home/csutter/DRIVE-clean/operational_runs/*/data_6_ensembling") 
        # For obs model: glob("/home/csutter/DRIVE-clean/operational_runs/*/data_odm_3_ensembling") 


        allfiles_data = []
        for i in alldirs_data_preds:
            fs = glob(f"{i}/*/*/*/*/*")
            for f in fs:
                allfiles_data.append(f)
        # print(allfiles_data[0:3])


        ### Identify files that specifically align with event occurances
        # one list will grabs the preduiction file, the other list just tracks the time for the file (not sure if we'll need but might as well)
        matched_cnn_file = [] 
        matched_cnn_time = []
        for model_f in allfiles_data:
            # parse just the time corresponding to the model pred
            beg = model_f.rfind("/")
            model_time = model_f[beg-13:beg]
            # # print(model_time)
            # see if that model_time exists in the list of event times
            # also just have to make sure we already didn't log that file (just in case was accidentally ran twice in operational_run, which can also happen just due to reorg in set__aggregate_allruns, so need to just use one csv pred)
            if ((model_time in eventtimes) & (model_time not in matched_cnn_time)):
                matched_cnn_time.append(model_time) # log the time that matches
                matched_cnn_file.append(model_f) # log the pred file that matches

        # # print examples and lengts
        # print(matched_cnn_time[0:4])
        # print(len(matched_cnn_time))

        # print(matched_cnn_file[0:4])
        # print(len(matched_cnn_file))

        # See how this compares to the total duration of times from eventtimes 
        # print(len(eventtimes))


        ############## LOOP 3: MODEL FILES

        for i_modelfile in range(0, len(matched_cnn_file)):
            
            mf = matched_cnn_file[i_modelfile]
            mdate = matched_cnn_time[i_modelfile]
            # Now, using the relevant model files, for which we know there was XYZ event happening, need to correspond the pred observations of interest to where the event was spatially. This is the spatial join. Since we just simplified it to the EVENT_ID level right now, it's just one region, but note that EPISODE_ID is a weather system, wich will be helpful for visualizations later

            # For now, just focus on identifying the cam preds that are IN the weather event

            # For each model pred file, spatial join with the event location (geometry) in the events1 df

            # Start with one model file for example...

            modelfile = pd.read_csv(mf)

            # print(len(modelfile))

            # # print(list(modelfile.columns))

            modelfile_gdf = gpd.GeoDataFrame(
                modelfile, 
                geometry=gpd.points_from_xy(modelfile.Longitude, modelfile.Latitude),
                crs="EPSG:4326" # Start with standard GPS coordinates
            )
            # print(len(modelfile_gdf))

            modelfile_gdf = modelfile_gdf.to_crs(events1.crs) # match the CRS to be exactly the system being used in the events dataset
            # print(len(modelfile_gdf))

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
            ctsdf = joined_df[["select","img_name"]].groupby(["select"]).count().reset_index()

            ctsdf = ctsdf.rename(columns = {"img_name":"count"})

            countdict = ctsdf.set_index("select")["count"].to_dict()

            # display(ctsdf.head(4))
            # print(countdict)

            # other info

            id_event = events1["EVENT_ID"].item()
            id_ep = events1["EPISODE_ID"].item()
            id_type = events1["EVENT_TYPE"].item()
            id_loc = events1["CZ_NAME"].item()
            id_wfo = events1["WFO"].item()
            datetime_stamp = mdate 


            # print(id_event,id_ep,id_type)


            # Your data
            data_to_log = {
                "id_event": id_event, #id_event,
                "id_ep": id_ep,
                "id_type": id_type,
                "id_loc":id_loc,
                "id_wfo":id_wfo,
                "datetime":datetime_stamp,
                "model_counts": countdict # This will be stringified
            }

            csv_path = "/home/csutter/DRIVE-clean/weather_events/models/stats_events_modelpred/stats_odm.csv" # HERE!!! Where to save out stats, I like to change this every time since still playing around w/ how to deal w everything

            # original w events and 5cat model was saved here:
            # "/home/csutter/DRIVE-clean/weather_events/models/stats_events_modelpred/stats.csv"

            # ynobs model saved here:
            # "/home/csutter/DRIVE-clean/weather_events/models/stats_events_modelpred/stats_odm.csv"

            # Pre-process: Convert the nested dict to a JSON string
            data_to_log["model_counts"] = json.dumps(data_to_log["model_counts"])

            # Check if file exists to determine if we need a header
            file_exists = os.path.isfile(csv_path)

            with open(csv_path, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data_to_log.keys())
                
                # Write header only if the file is being created for the first time
                if not file_exists:
                    writer.writeheader()
                    
                writer.writerow(data_to_log)

            print(f"Logged event {data_to_log['id_event']} to {csv_path}")