# Written with the assistance of Gemini

# To run in slurm: clean_evBuffer_aggstats.sh

import pandas as pd 
import numpy as np 
from glob import glob
import os
import geopandas as gpd

import csv
import json

from ast import literal_eval
from shapely import wkt

######### CONFIG

alldirs_data_preds = glob("/home/csutter/DRIVE-clean/operational_runs_QPEdata/data_odm_3_ensembling") #HERE!! 
# data_6_ensembling
# data_odm_3_ensembling

csv_path = "/home/csutter/DRIVE-clean/weather_events/models/stats_events_modelpred/stats_buffer_odm.csv"# HERE!!!

##########
#### 1 - NCEI data
d_readin = gpd.read_file("/home/csutter/DRIVE-clean/weather_events/data/ncei_events/ncei_ny_events_clean.gpkg")

# add timedelta duration col back (using duration_sec col)
d_readin["duration"] = pd.to_timedelta(d_readin["duration_sec"], unit="s")

#### 2 - Cam lat and lons 
cams = pd.read_csv("/home/csutter/DRIVE/site_analysis/_reference/511NY_API_GetCameras_response.csv")

cams = cams[((cams["Disabled"]==False)&(cams["Blocked"]==False))]

cams_gdf = gpd.GeoDataFrame(
    cams, 
    geometry=gpd.points_from_xy(cams.Longitude, cams.Latitude),
    crs="EPSG:4326" # Start with standard GPS coordinates
)

cams_gdf = cams_gdf.to_crs(d_readin.crs) # match the CRS to be exactly the system being used in the events dataset

# Perform the spatial join
events_camloc = gpd.sjoin(
    cams_gdf, 
    d_readin[["EVENT_ID","geometry"]], 
    how="inner",
    predicate="within" # Check if the point is WITHIN the polygon
)


#### Connect to events of interest (inner join on EVENT ID)
events_ofinterest_paths = [
    "/home/csutter/DRIVE-clean/weather_events/data/ncei_events_ofinterest/blizzard_allyrs_ceilfloor5min_nobuffer_freq5min.csv",
    "/home/csutter/DRIVE-clean/weather_events/data/ncei_events_ofinterest/heavyrain_2425_ceilfloor15min_nobuffer_freq15min.csv",
    "/home/csutter/DRIVE-clean/weather_events/data/ncei_events_ofinterest/heavysnow_2025_ceilfloor5min_nobuffer_freq5min.csv",
    "/home/csutter/DRIVE-clean/weather_events/data/ncei_events_ofinterest/heavysnow_2425_ceilfloor15min_nobuffer_freq15min.csv",
    "/home/csutter/DRIVE-clean/weather_events/data/ncei_events_ofinterest/lakeeffect_2025_ceilfloor15min_nobuffer_freq15min.csv",
    "/home/csutter/DRIVE-clean/weather_events/data/ncei_events_ofinterest/lakeeffect_2025_ceilfloor30min_nobuffer_freq30min.csv",
    "/home/csutter/DRIVE-clean/weather_events/data/ncei_events_ofinterest/lakeeffect_2425_ceilfloor15min_nobuffer_freq15min.csv",
    "/home/csutter/DRIVE-clean/weather_events/data/ncei_events_ofinterest/winterstorm_2425_ceilfloor15min_nobuffer_freq15min.csv",
    "/home/csutter/DRIVE-clean/weather_events/data/ncei_events_ofinterest/winterweather_2425_ceilfloor15min_nobuffer_freq15min.csv"
]

gdfs_list = []

for ev in events_ofinterest_paths:
    events = pd.read_csv(ev)
    events['geometry'] = events['geometry'].apply(wkt.loads)
    events_gdf = gpd.GeoDataFrame(events, geometry='geometry', crs="EPSG:4269")
    events_gdf = events_gdf.to_crs(d_readin.crs)
    gdfs_list.append(events_gdf)

eventsofint = pd.concat(gdfs_list, ignore_index=True)

# Inner join w the gdf made in the beginning which connects all events to cams
eventsofint_camloc = eventsofint.merge(events_camloc[['Latitude', 'Longitude', 'ID', 'Name','DirectionOfTravel', 'RoadwayName', 'Url', 'VideoUrl', 'Disabled','Blocked','EVENT_ID']], how="inner", on="EVENT_ID")

locs = eventsofint_camloc[['geometry','CZ_FIPS', 'CZ_NAME', 'WFO','CZ_FIPS_FORMAT', 'ZONE', 'FIPS', 'FIPS_FORMAT','Latitude', 'Longitude', 'ID', 'Name','DirectionOfTravel', 'RoadwayName', 'Url', 'VideoUrl', 'Disabled','Blocked']]
locsunique = locs.drop_duplicates()
locsunique['ID'] = locsunique['ID'].str.replace('-', '_')

locsunique_bare = locsunique[["geometry","CZ_NAME","WFO", "ID"]]

##### Buffer Tracker
buffertracker = pd.read_csv("/home/csutter/DRIVE-clean/weather_events/data/nonevents_ofinterest/buffertimes_for_nceievents.csv")
alltimes = sorted(np.unique(buffertracker["buffer_datetime"]))

# --- EFFICIENCY UPGRADE 1: Dictionary Lookup ---
allfiles_data = []
for i in alldirs_data_preds:
    fs = glob(f"{i}/*/*/*/*/*")
    allfiles_data.extend(fs) # Faster than appending in a loop

file_lookup = {}
for f in allfiles_data:
    beg = f.rfind("/")
    time_key = f[beg-13:beg]
    file_lookup[time_key] = f

matched_cnn_file = [] 
matched_cnn_time = []
for etime in alltimes:
    # Use dictionary for instant lookup instead of looping through all files
    if str(etime) in file_lookup:
        if str(etime) not in matched_cnn_time:
            matched_cnn_time.append(str(etime))
            matched_cnn_file.append(file_lookup[str(etime)])

# --- EFFICIENCY UPGRADE 2: usecols for fast reading ---
# Added the prob and qpe columns here!
cols_to_use = ["site", "select", "img_name", "select_prob", "qpe_val"] 

# Loop 1 - model files
for i in range(len(matched_cnn_file)):
    mf = matched_cnn_file[i]
    t = matched_cnn_time[i]

    # Load only necessary columns
    d = pd.read_csv(mf, usecols=cols_to_use)

    d_allinfo = d.merge(locsunique_bare, how="inner", left_on="site", right_on="ID")

    # Loop 2 -- by event region (CZ_NAME)
    for l in np.unique(d_allinfo["CZ_NAME"]):
        ld1 = d_allinfo[d_allinfo["CZ_NAME"]==l] 

        # Loop 3 -- by WFO
        for w in np.unique(ld1["WFO"]):
            
            # Filter by WFO to get the exact subset we care about
            ld2 = ld1[ld1["WFO"]==w] 

            # Just for tracking purposes, which may make other analyses easier in the future - we won't need to spatially do the gdf join between NCEI events and the lat lons from the modelfile of int
            siteslist = list(ld2["site"])

            # --- CALCULATE COUNTS AND PROBABILITIES ---
            # Group by 'select' to get both counts and mean probabilities efficiently
            agg_results = ld2.groupby("select").agg({
                "img_name": "count",
                "select_prob": "mean",
                "qpe_val": "mean",
            }).rename(columns={"img_name": "count", "select_prob": "avg_predprob", "qpe_val": "avg_qpe"})

            countdict = agg_results["count"].to_dict()
            
            probdict = {
                pred_class: round(avg, 3) 
                for pred_class, avg in agg_results["avg_predprob"].items()
            }

            avgprob_overall = np.mean(ld2["select_prob"])

            qpedict = {
                pred_class: round(avg, 3) 
                for pred_class, avg in agg_results["avg_qpe"].items()
            }

            avgqpe_overall = round(np.mean(ld2["qpe_val"]), 4)

            id_event = t
            id_ep = t
            id_type = "buffer"
            id_loc = l 
            id_wfo = w 
            datetime_stamp = t 

            # --- UPDATE LOGGING DICTIONARY ---
            data_to_log = {
                "id_event": id_event,
                "id_ep": id_ep,
                "id_type": id_type,
                "id_loc": id_loc,
                "id_wfo": id_wfo,
                "datetime": datetime_stamp,
                "modelfile": mf,
                "sites": siteslist,
                "model_counts": json.dumps(countdict),
                "model_probs": json.dumps(probdict), 
                "prob_avg": avgprob_overall,
                "model_qpe":qpedict,
                "qpe_avg": avgqpe_overall
            }

            file_exists = os.path.isfile(csv_path)

            with open(csv_path, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data_to_log.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(data_to_log)

            print(f"Logged buffer {data_to_log['id_loc']} / {data_to_log['id_wfo']} at {t} to {csv_path}")

