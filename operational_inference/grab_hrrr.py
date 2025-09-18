import multiprocessing

import config
import pandas as pd

total_cores = multiprocessing.cpu_count()
# print(total_cores)
cores_to_use = max(1, int(total_cores * 0.75))
import ast
import os

import geopy.distance
import numpy as np
from joblib import Parallel, delayed


def step3_fn(dirsave):

    print("Starting step 3, grabbing relevant HRRR files for observations")
    # grab path where csv from step 1 is saved out
    # comment out one or the other...

    step1file = f"{dirsave}/step1_imgfiles.csv"

    # Add camera latitude and longitude to df
    step1data = pd.read_csv(step1file)

    # cam lat and lon reference file
    camlocs = pd.read_csv(
        "/home/csutter/DRIVE/site_analysis/_reference/ny511sites_ID_latlon.csv"
    )

    # merge to get
    imgcamdata = step1data.merge(
        camlocs[["site", "Latitude", "Longitude"]], how="inner", on="site"
    )

    # imgcamdata.head(3)
    print(len(imgcamdata))
    # print(imgcamdata[0:8])

    # CODE FROM /home/csutter/DRIVE/inference/hrrr_colocate.py
    # copied and pasted in notebook to test functions and timing of them

    def format_df_for_mapping_hrrr(modelpreds, fcsthr_num=2):
        """
        Inputdf: a dataframe (e.g. tracker) with image lat and lons. For inference mode, this will be the most recent file of images and their already-ran CNN predictions
        """

        # for testing, read in this csv: f"/home/csutter/DRIVE/predictions_model_inference/allcams_givenhr_{spec}.csv"

        # modelpreds = modelpreds[0:20] # FOR TESTING ONLY!
        # print(modelpreds[1:4])
        print(len(modelpreds))

        yrlist = []
        molist = []
        daylist = []
        datelist = []
        timelist = []
        for ind in range(0, len(modelpreds)):
            i = modelpreds["img_model"][ind]
            # if confighrrr.for_inference == True:
            #     sitestart = i.find("data/") + 5  # find first occurance
            #     siteinter = i[sitestart:]
            #     datebegin = (
            #         siteinter.rfind("_") + 1
            #     )  # find first occurance, note rfind finds last
            #     dateend = siteinter.rfind(".")  # find last occurance
            #     datename = siteinter[datebegin:dateend]
            # else:
            begindate = i.rfind("_") + 1
            enddate = i.rfind("-")
            date1 = i[begindate:enddate]
            datename = date1.replace("-", "")
            yrlist.append(datename[0:4])
            molist.append(datename[4:6])
            daylist.append(datename[6:8])
            datelist.append(datename)
            timeisend = i.find(".jpg")
            timeis = i[timeisend - 8 : timeisend]
            timelist.append(timeis)

        # print("before here???")
        # print(yrlist[1:5])
        # print(molist[1:5])
        # print(daylist[1:5])
        # print(datelist[1:5])
        # print(timelist[1:5])

        modelpreds["yr"] = yrlist
        modelpreds["mo"] = molist
        modelpreds["day"] = daylist
        modelpreds["date"] = datelist
        modelpreds["time"] = timelist

        ### add more date columns to model pred df to easily map to the right hrrr files

        modelpreds["ymd"] = modelpreds["date"]
        # modelpreds["ymd"] = modelpreds.date.str.replace("-", "")

        # UNCOMMENT IF WANT TO TROUBLE SHOOT
        # print("few examples of dates and times of img observations")
        # print(modelpreds["ymd"][1:5])
        # print(modelpreds.time[1:5])
        # print(modelpreds.date[1:5])
        modelpreds["date_and_time_str"] = modelpreds.time + " " + modelpreds.date
        # print(modelpreds["date_and_time_str"][1:5])
        # print(modelpreds.dtypes)

        modelpreds["date_and_time_dt"] = pd.to_datetime(modelpreds["date_and_time_str"])

        modelpreds["date_and_time_dt_round"] = modelpreds["date_and_time_dt"].dt.round(
            "H"
        )

        # init time + forecast hour = valid time
        modelpreds["date_and_time_dt_init"] = modelpreds[
            "date_and_time_dt_round"
        ] - pd.Timedelta(hours=fcsthr_num)

        # if want to parse them out separately 10/4
        modelpreds["init_yyyy"] = (
            modelpreds["date_and_time_dt_init"].dt.year.apply(str).str.zfill(4)
        )
        modelpreds["init_month"] = (
            modelpreds["date_and_time_dt_init"].dt.month.apply(str).str.zfill(2)
        )
        modelpreds["init_day"] = (
            modelpreds["date_and_time_dt_init"].dt.day.apply(str).str.zfill(2)
        )
        modelpreds["init_hour"] = (
            modelpreds["date_and_time_dt_init"].dt.hour.apply(str).str.zfill(2)
        )

        # modelpreds["date_and_time_dt_init_opt2"] = modelpreds[
        #     "date_and_time_dt_round"
        # ] - pd.Timedelta(hours=2)

        # modelpreds["date_and_time_dt_init_opt3"] = modelpreds[
        #     "date_and_time_dt_round"
        # ] - pd.Timedelta(hours=3)

        modelpreds["init_hr"] = modelpreds["date_and_time_dt_init"].dt.strftime("%H")
        # modelpreds["init_hr_opt2"] = modelpreds["date_and_time_dt_init_opt2"].dt.strftime("%H")
        # modelpreds["init_hr_opt3"] = modelpreds["date_and_time_dt_init_opt3"].dt.strftime("%H")

        # now we can just find the hrrr files that have init time matching the init time column above, and are also for forecast hour = 1 (note that this is the closest to "neartime"... later see if we have hrrr 0z data). The init time col I added above is assuming we're using fcst hour 1. If forceast hour 0, would just use the rounded column.

        # modelpreds.head(6)
        modelpreds["yyyy"] = modelpreds.yr.apply(str)

        # filename_without fcst hour (will pull 3 different init times)
        # pre_filename = (
        #     f"/home/csutter/AI2ES/cleaned/HRRR/"
        #     + modelpreds.yr.apply(str)
        #     + "/"
        #     + modelpreds.mo.apply(str).str.zfill(2)
        #     + "/"
        #     + modelpreds.ymd.apply(str)
        #     + "_hrrr.t"
        # )

        # # updated 10/4 to use the directory naming convention of the hrrr file for two hours prior! Not the image date bc it failed on 00Hour images bc was pulling from top of hour
        pre_filename = (
            f"/home/csutter/AI2ES/cleaned/HRRR/"
            + modelpreds.init_yyyy
            + "/"
            + modelpreds.init_month
            + "/"
            + modelpreds.init_yyyy
            + modelpreds.init_month
            + modelpreds.init_day
            + "_hrrr.t"
        )

        #

        fcsthr_str_2dig = str(fcsthr_num).zfill(2)

        # create a column that has the corresponding hrrr file name
        modelpreds["hrrr_file"] = (
            pre_filename
            + modelpreds.init_hr.apply(str)
            + f"z_{fcsthr_str_2dig}.parquet"
        )
        # # add backup file fcst hr 2
        # modelpreds["hrrr_file_opt2"] = (
        #     pre_filename
        #     + modelpreds.init_hr_opt2.apply(str)
        #     + "z_02.parquet"
        # )
        # # add backup file fcst hr 3
        # modelpreds["hrrr_file_opt3"] = (
        #     pre_filename
        #     + modelpreds.init_hr_opt3.apply(str)
        #     + "z_03.parquet"
        # )

        # note that zfill used to make months 2 digits with leading 0 with zfill

        # UNCOMMENT TO TROUBLESHOOT
        # print(modelpreds["hrrr_file"][1:5])

        return modelpreds

    def colocate_add_hrrr_data(row):

        # write mindist as row function to apply on cam pred df
        # def mindist(row):
        # print(row)

        siteinput = row["site"]
        lat = row["Latitude"]
        lon = row["Longitude"]
        # if ((os.path.isfile(row["hrrr_file"]))|(os.path.isfile(row["hrrr_file_opt2"]))|(os.path.isfile(row["hrrr_file_opt3"]))):
        if os.path.isfile(row["hrrr_file"]):
            hrrrfile = row["hrrr_file"]
            # print("using 1 fcst hour")
            # elif os.path.isfile(row["hrrr_file_opt2"]):
            #     hrrrfile = row["hrrr_file_opt2"]
            #     print("using 2 fcst hour")
            # elif os.path.isfile(row["hrrr_file_opt3"]):
            #     hrrrfile = row["hrrr_file_opt3"]
            #     print("using 3 fcst hour")

            # read in hrrr file of interest for that element
            hrrrdf = pd.read_parquet(hrrrfile)
            hrrrdf = hrrrdf.reset_index()
            hrrrdf_vars = hrrrdf[
                [
                    "time",
                    "valid_time",
                    "latitude",
                    "longitude",
                    "t2m",
                    "pt",
                    "sh2",
                    "d2m",
                    "r2",
                    "u10",
                    "v10",
                    "si10",
                    "asnow",
                    "tp",
                    "orog",
                    "cape",
                    "mslma",
                    "dswrf",
                    "dlwrf",
                    "tcc",
                    "gh",
                    "dpt",
                    "atmosphere",
                    "isobaricInhPa",
                ]
            ]  # will subset this for each location below
            # display(hrrrdf_vars.head(3))
            # note that usually each time snapshot of cam data (every 5 mins) will be from the same hrrr file since they are at the top of the hour, but for any in between times (e.g. if one image snapshot is 3:29 for one image and 3:31 for another image they would technically be pulling from)
            # subset the df containing the hrrr data that should already be read in
            # first subsetting by .1 deg above below cam location to make the min dis query list smaller (haversine calc)
            dfsubset_halfdegree = hrrrdf_vars[
                (
                    (hrrrdf_vars["latitude"] < lat + 0.1)
                    & (hrrrdf_vars["latitude"] > lat - 0.1)
                    & (hrrrdf_vars["longitude"] < lon + 0.1)
                    & (hrrrdf_vars["longitude"] > lon - 0.1)
                )
            ]
            comparelist = [
                [lat_2, lon_2]
                for lat_2, lon_2 in zip(
                    dfsubset_halfdegree["latitude"], dfsubset_halfdegree["longitude"]
                )
            ]
            # print(f"comparing {len(comparelist)} number of lat lons rather than {len(dfinput)}")
            dist_mes = (
                {}
            )  # make a dictionary which, for the input x lat/lon, will add the distances to each mesonet station
            for y in range(0, len(comparelist)):
                coords_1 = (lat, lon)  # x is/represents the input (so cams), the row
                coords_2 = (comparelist[y][0], comparelist[y][1])  # hrrr coordinates
                dist = geopy.distance.geodesic(coords_1, coords_2).km
                dist_mes[
                    str(comparelist[y])
                ] = dist  # great circle distance calc but for spheroids, add to dist_mes dictionary w lat lon string as key and dist as value
            # print(dist_mes)
            if len(dist_mes) == 0:
                # UNCOMMENT IF WANT TO SEE SITES W/O LAT LONS
                # print(siteinput, lat, lon, dist_mes)
                # print(
                #     "issue with input lat and lon above. Check it. Setting values for dictionary to 0 for now, to be corrected later."
                # )
                min_site = "[0, 0]"
                min_dist = 1000
                # print(min_site)
                # print(type(min_site))
                fcstdata_list = [999]
            else:
                # print("working!")
                min_site = min(
                    dist_mes, key=dist_mes.get
                )  # min() of a dict grabs the min value, and here we're pulling the key to that min value: https://stackoverflow.com/questions/3282823/get-the-key-corresponding-to-the-minimum-value-within-a-dictionary
                # print(min_site)
                # print(type(min_site))
                min_dist = dist_mes[min_site]
                # print(min_site)
                # print(min_dist)

                # grab hrrr data for that hrrr grid point (closest location just identified)
                hrrrlatlon = ast.literal_eval(min_site)
                fcstdata = hrrrdf_vars[
                    (
                        (hrrrdf_vars["latitude"] == hrrrlatlon[0])
                        & (hrrrdf_vars["longitude"] == hrrrlatlon[1])
                    )
                ].reset_index()
                # print(fcstdata.columns)
                fcstdata_list = fcstdata.iloc[0].tolist()
                # display(fcstdata_list)
        else:
            # print("no relevant hrrr file!")
            # just use blank data that can be identified
            min_site = "[0, 0]"
            min_dist = 1000
            fcstdata_list = [999]

        # print("through selection of hrrr file")

        return min_site, min_dist, fcstdata_list

    def add_hrrrdata_todf(inputdf):

        print(len(inputdf))

        print("starting parallelization to map to nearest data")
        hrrrdatalists = Parallel(n_jobs=4)(
            delayed(colocate_add_hrrr_data)(row) for index, row in inputdf.iterrows()
        )  # (n_jobs=cores_to_use

        # try without parallel
        # hrrrdatalists = inputdf.apply(colocate_add_hrrr_data, axis = 1)

        # add the HRRR data to the main observation df, parsing out listed info from colocated data
        inputdf["HRRR_latlon"] = [i[0] for i in hrrrdatalists]
        inputdf["HRRR_distkm"] = [i[1] for i in hrrrdatalists]
        inputdf["HRRR_data"] = [i[2] for i in hrrrdatalists]

        # clean up df and save out
        # 1. remove observations that lack a camera lat lon and therefore cant be plotted (for UI) and also dont have corresponding HRRR data
        # remove_idx = []
        # for i in range(len(inputdf)):
        #     if inputdf["HRRR_data"][i] == [999]:
        #         remove_idx.append(i)
        print("total number of obs")
        print(len(inputdf))
        # print("removing this many images:")
        # print(len(remove_idx))

        # inputdf = inputdf.drop(remove_idx)

        # 2. Split the HRRR_data column, which has a list of 24 hrrr vars, into 24 separate columns
        df3 = inputdf["HRRR_data"].apply(pd.Series)
        # Optionally, rename the new columns
        df3.columns = [
            "id_old",
            "time",
            "valid_time",
            "latitude",
            "longitude",
            "t2m",
            "pt",
            "sh2",
            "d2m",
            "r2",
            "u10",
            "v10",
            "si10",
            "asnow",
            "tp",
            "orog",
            "cape",
            "mslma",
            "dswrf",
            "dlwrf",
            "tcc",
            "gh",
            "dpt",
            "atmosphere",
            "isobaricInhPa",
        ]

        # Concatenate the original DataFrame with the new columns
        df_obs = pd.concat(
            [inputdf.drop(columns=["HRRR_data"]), df3.drop(columns=["id_old"])], axis=1
        )

        return df_obs

    # EVAL 1: create df with file paths attached
    df2 = format_df_for_mapping_hrrr(imgcamdata)

    # added 9/25
    # Make more efficient by reading in the hrrr data file only once rather than reading it in for each observation (note that most of the observations will be from the same HRRR file, but if there is a time difference (e.g. if running for 10:30, some images may be 10:29 --> rounded to 10 and others may be 10:33 --> rounded to 11), there could be multiple HRRR files needed, and this accounts for that)

    unique_hrrrfiles = np.unique(df2["hrrr_file"])

    list_of_dfs = []
    for hf in unique_hrrrfiles:
        print(f"starting for {hf}")
        # subset to the instances that have hrrr file hf
        df_subset_byfile = df2[df2["hrrr_file"] == hf]
        print("subsetted")
        # proceed with mapping/grabbing hrrr data
        df_subset_withhrrr = add_hrrrdata_todf(df_subset_byfile)
        print("length of returning df")
        print(len(df_subset_withhrrr))
        print("columns are below!")
        print(df_subset_withhrrr.columns)
        list_of_dfs.append(df_subset_withhrrr)
        print("appended to list")

    finaldf_step3 = pd.concat(list_of_dfs)

    # dont save hrrr data where any of the 8 vars are nans
    finaldf_step3_clean = finaldf_step3[
        ~finaldf_step3[["t2m", "r2", "u10", "v10", "asnow", "tp", "orog", "tcc"]]
        .isna()
        .any(axis=1)
    ]

    print(f"{dirsave}/step3_hrrrdata.csv")
    # Save out as step3
    finaldf_step3_clean.to_csv(f"{dirsave}/step3_hrrrdata.csv")

    # MOVED ABOVE IN FN 925
    # hrrrdatalists = Parallel(n_jobs=cores_to_use)(
    #     delayed(colocate_add_hrrr_data)(row) for index, row in df2.iterrows()
    # )

    # # add the HRRR data to the main observation df, parsing out listed info from colocated data
    # df2["HRRR_latlon"] = [i[0] for i in hrrrdatalists]
    # df2["HRRR_distkm"] = [i[1] for i in hrrrdatalists]
    # df2["HRRR_data"] = [i[2] for i in hrrrdatalists]

    # # clean up df and save out
    # # 1. remove observations that lack a camera lat lon and therefore cant be plotted (for UI) and also dont have corresponding HRRR data
    # remove_idx = []
    # for i in range(len(df2)):
    #     if df2["HRRR_data"][i] == [999]:
    #         remove_idx.append(i)
    # print("total number of obs")
    # print(len(df2))
    # print("removing this many images:")
    # print(len(remove_idx))

    # # print(len(df2))
    # df2 = df2.drop(remove_idx)
    # # print(len(df2))

    # # 2. Split the HRRR_data column, which has a list of 24 hrrr vars, into 24 separate columns
    # df3 = df2["HRRR_data"].apply(pd.Series)
    # # Optionally, rename the new columns
    # df3.columns = [
    #     "id_old",
    #     "time",
    #     "valid_time",
    #     "latitude",
    #     "longitude",
    #     "t2m",
    #     "pt",
    #     "sh2",
    #     "d2m",
    #     "r2",
    #     "u10",
    #     "v10",
    #     "si10",
    #     "asnow",
    #     "tp",
    #     "orog",
    #     "cape",
    #     "mslma",
    #     "dswrf",
    #     "dlwrf",
    #     "tcc",
    #     "gh",
    #     "dpt",
    #     "atmosphere",
    #     "isobaricInhPa",
    # ]

    # # Concatenate the original DataFrame with the new columns
    # df_obs = pd.concat(
    #     [df2.drop(columns=["HRRR_data"]), df3.drop(columns=["id_old"])], axis=1
    # )

    # EVAL 2:


# def step3_fn():
#     print("hello new step 3")
