import pandas as pd 
import numpy as np 
from glob import glob
import os
import geopandas as gpd

import csv
import json

from ast import literal_eval

from shapely import wkt



#### 1 - NCEI data
d_readin = gpd.read_file("/home/csutter/DRIVE-clean/weather_events/data/ncei_events/ncei_ny_events_clean.gpkg")

d_readin.head(4)

# add timedelta duration col back (using duration_sec col)
# note that you have to do this w/ every gdf
d_readin["duration"] = pd.to_timedelta(d_readin["duration_sec"], unit="s")

# may take ~40 seconds

# print(len(d_readin))

# print(type(d_readin))

#### 2 - Cam lat and lons (not sure we need, just load data for now)

cams = pd.read_csv("/home/csutter/DRIVE/site_analysis/_reference/511NY_API_GetCameras_response.csv")

cams = cams[((cams["Disabled"]==False)&(cams["Blocked"]==False))]
# print(len(cams))


cams_gdf = gpd.GeoDataFrame(
    cams, 
    geometry=gpd.points_from_xy(cams.Longitude, cams.Latitude),
    crs="EPSG:4326" # Start with standard GPS coordinates
)
# # print(len(cams_gdf))

cams_gdf = cams_gdf.to_crs(d_readin.crs) # match the CRS to be exactly the system being used in the events dataset
# # print(len(cams_gdf))

# Perform the spatial join
events_camloc = gpd.sjoin(
    cams_gdf, 
    d_readin[["EVENT_ID","geometry"]], 
    how="inner",
    predicate="within" # Check if the point is WITHIN the polygon
)

cams_gdf.head(3)

# print(len(d_readin[["EVENT_ID","geometry"]]))
# print(len(cams))
# print(len(events_camloc))

# Splits an event (which was usually one row) into multiple rows, one per each camera, so they larger df size makes sense

# note that doing the join this way (inner) ensures that we're only keeping events and geometries that have cams in them. To get this only for events that we ran (rather than the full d_readin df using right now), see below. Just repeat the steps with joining the cams df only on the events of interest!!



#### Connect to eventsof interest (inner join on EVENT ID) to get just the locations / times we care about

events_ofinterest_paths = ["/home/csutter/DRIVE-clean/weather_events/data/ncei_events_ofinterest/blizzard_allyrs_ceilfloor5min_nobuffer_freq5min.csv",
"/home/csutter/DRIVE-clean/weather_events/data/ncei_events_ofinterest/heavyrain_2425_ceilfloor15min_nobuffer_freq15min.csv",
"/home/csutter/DRIVE-clean/weather_events/data/ncei_events_ofinterest/heavysnow_2025_ceilfloor5min_nobuffer_freq5min.csv",
"/home/csutter/DRIVE-clean/weather_events/data/ncei_events_ofinterest/heavysnow_2425_ceilfloor15min_nobuffer_freq15min.csv",
"/home/csutter/DRIVE-clean/weather_events/data/ncei_events_ofinterest/lakeeffect_2025_ceilfloor15min_nobuffer_freq15min.csv",
"/home/csutter/DRIVE-clean/weather_events/data/ncei_events_ofinterest/lakeeffect_2025_ceilfloor30min_nobuffer_freq30min.csv",
"/home/csutter/DRIVE-clean/weather_events/data/ncei_events_ofinterest/lakeeffect_2425_ceilfloor15min_nobuffer_freq15min.csv",
"/home/csutter/DRIVE-clean/weather_events/data/ncei_events_ofinterest/winterstorm_2425_ceilfloor15min_nobuffer_freq15min.csv",
"/home/csutter/DRIVE-clean/weather_events/data/ncei_events_ofinterest/winterweather_2425_ceilfloor15min_nobuffer_freq15min.csv"]

gdfs_list = []

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
    gdfs_list.append(events_gdf)

eventsofint = pd.concat(gdfs_list,ignore_index=True)
# print(type(eventsofint))
# print(eventsofint.columns)
# print(len(eventsofint))


# Inner join w the gdf made in the beginning which connects all events to cams

eventsofint_camloc = eventsofint.merge(events_camloc[['Latitude', 'Longitude', 'ID', 'Name','DirectionOfTravel', 'RoadwayName', 'Url', 'VideoUrl', 'Disabled','Blocked','EVENT_ID']], how = "inner", on = "EVENT_ID")

# For collecting (polygons) and cams within them, make a df just of the location related cols, including those from the geometry (geom, wfo, etc), as well as those from the cameras (lat, Lon, and ID)

# Use the df from eventsofint_camloc bc these are the locations we will care to compare (like for non-events and buffer events, as in the other sections below)

locs = eventsofint_camloc[['geometry','CZ_FIPS', 'CZ_NAME', 'WFO','CZ_FIPS_FORMAT', 'ZONE', 'FIPS', 'FIPS_FORMAT','Latitude', 'Longitude', 'ID', 'Name','DirectionOfTravel', 'RoadwayName', 'Url', 'VideoUrl', 'Disabled','Blocked']]

# print(len(locs))
locsunique = locs.drop_duplicates()

# print(len(locsunique))

# Note that this is JUST unique locations - loosely, the structure is geometry | cam-level

locsunique.head(3)
locsunique['ID'] = locsunique['ID'].str.replace('-', '_')



locsunique_bare = locsunique[["geometry","CZ_NAME","WFO", "ID"]] # "Latitude", "Longitude" is already in model pred dfs, dont need it here

locsunique_bare.head(3)


##### For connecting to buffer runs (which don't have geometries with them!) 
# - But we will eventually want to "connect" event geometries so that we can compare model preds in regions that have events vs when those same regions don't have events
# - Uses the df built from events work in cells above



inf_sets_ran = ["/home/csutter/DRIVE-clean/operational_runs/set45_buffermisc"]

dirs_w_preds = [f"{i}/data_odm_3_ensembling/*/*/*/*/*" for i in inf_sets_ran] #HERE!!! 
# If looking for 5cat model or ODM model, update accordingly
# 5-cat model: f"{i}/data_6_ensembling/*/*/*/*/*"
# ODM model: f"{i}/data_odm_3_ensembling/*/*/*/*/*"

# # print(dirs_w_preds)

pred_datetimes = []
pred_path = []
for dr in dirs_w_preds:
    # # print(dr) # just for checking counts per dir
    # dr_files = [] # just for checking counts per dir
    listpredfiles = glob(dr)
    for fl in listpredfiles:
        # # print(fl)
        i = fl.rfind("/")
        filedate = fl[i-13:i]
        pred_datetimes.append(filedate)
        # dr_files.append(filedate) # just for checking counts per dir

        ### Must also grab the tracker path which contains the predictions and the image path (should have all the data in it we need)
        pred_path.append(fl)
    # # print(len(dr_files)) # just for checking counts per dir


# print(len(pred_datetimes))
# print(len(pred_path))


# print(pred_datetimes[0:4])
# print(pred_path[0:4])


# Get stats for nonevents...

# First -- take the model pred df and Connect to locations (wfo, geom, etc.. things in ncei events) that have events associated with them, bc we will ultimately want to compare how locations with events compare to locations with nonevents
# We ran the model preds statewide, but when we merge with locations-with-events, this will shorten the amount of cameras in a model pred file
# Already have the location df with geoms and cam lat/lon
# Connect the model pred df , by site, with the "ID" in that location df

# Read in all model pred dfs, which have cams in them, and tie in the corresponding region for each cam

# Note the looping order here for nonevents is different than for events, given the different datastructure, have to start with model pred files. And don't need to "find" the right model pred files that match events, bc starting w model pred files.

# Loop 1 - model files
for i in range(0, len(pred_path)):


    d = pd.read_csv(pred_path[i])
    t = pred_datetimes[i]# for tracking results
    # # print(len(d))
    # note that im not making the modelpred df into a gdf right now, dont think we need the geom for anything rn at least for stats runs
    d_allinfo = d.merge(locsunique_bare, how = "inner", left_on = "site", right_on = "ID")
    # # print(len(d_allinfo))

    # Loop 2 -- by event region, which for us will be CZ_NAME 
    for l in np.unique(d_allinfo["CZ_NAME"]):
        # subset 
        ld1 = d_allinfo[d_allinfo["CZ_NAME"]==l].reset_index()

        # I think one CZ is split between two WFOs, account for that small detail with one more loop
        for w in np.unique(ld1["WFO"]):

            ld2 = ld1[ld1["CZ_NAME"]==l].reset_index()


            # JUST LOG EVERYTHING NOW THAT IT'S BY REGION / TIME (note: for now, just set episode and event equal to the datetime. Will work out those kinks of how to aggregate them later to make the events/episodes easier to compare to the real events)
            ctsdf = ld2[["select","img_name"]].groupby(["select"]).count().reset_index()

            ctsdf = ctsdf.rename(columns = {"img_name":"count"})

            countdict = ctsdf.set_index("select")["count"].to_dict()

            # # print(countdict)

            # other info

            id_event = t
            id_ep = t
            id_type ="nonevent"
            id_loc = l # loop 2
            id_wfo = w # loop 3
            datetime_stamp = t 


            # # print(id_event,id_ep,id_type)


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

            csv_path = "/home/csutter/DRIVE-clean/weather_events/models/stats_events_modelpred/stats_buffer_odm.csv"# HERE!!! Where to save out stats, I like to change this every time since still playing around w/ how to deal w everything

            # original w events and 5cat model was saved here:
            # "/home/csutter/DRIVE-clean/weather_events/models/stats_events_modelpred/stats_buffer.csv"

            # ynobs model saved here:
            # "/home/csutter/DRIVE-clean/weather_events/models/stats_events_modelpred/stats_buffer_odm.csv"


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





