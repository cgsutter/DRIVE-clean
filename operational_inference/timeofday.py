from glob import glob
import xarray as xr
import numpy as np
import pandas as pd

import datetime
from astral.sun import sun, sunset, sunrise, dawn, dusk
# from astral import sunset
from astral import LocationInfo


def sunevents(date, latd, longt, site):
    """ Input date as pandas timestamp (which has date and hr:min:sec attached, but in the function the date is just pulled out and used for sun events)
    Input latd and longt as floats
    Input site as string for tracking when there is no dawn, dusk, etc
    Return time of dawn, sunrise, sunset, and dusk for that date and location
    """
    year = date.year
    month = date.month
    day = date.day
    loc = LocationInfo(latitude=latd, longitude=longt)
    dictevents = {}
    try:
        dawntime = dawn(
            loc.observer,
            date=datetime.date(year, month, day),
        )  
        dawntime = dawntime.replace(tzinfo=None) # make timezone naive
        dawntime = pd.to_datetime(dawntime) # convert to pandas datetime timestamp format
        dictevents[dawntime]="dawn"
    except: 
        print(f"no dawn (UTC) on date {year}{month}{day} at {site}")
    try:
        sunrisetime = sunrise(
            loc.observer,
            date=datetime.date(year, month, day),
        )  
        sunrisetime = sunrisetime.replace(tzinfo=None) # make timezone naive
        sunrisetime = pd.to_datetime(sunrisetime) # convert to pandas datetime timestamp format
        dictevents[sunrisetime]="sunrise"
    except:
        # sunrisetime = float("nan")  # will later drop
        print(f"no sunrise (UTC)  on date {year}{month}{day} at {site}")
    try:
        sunsettime = sunset(
            loc.observer,
            date=datetime.date(year, month, day),
        )  
        sunsettime = sunsettime.replace(tzinfo=None) # make timezone naive
        sunsettime = pd.to_datetime(sunsettime) # convert to pandas datetime timestamp format
        dictevents[sunsettime]="sunset"
    except:
        # sunsettime = float("nan")  # will later drop
        print(f"no sunset (UTC)  on date {year}{month}{day} at {site}")
    try:
        dusktime = dusk(
            loc.observer,
            date=datetime.date(year, month, day),
        )  
        dusktime = dusktime.replace(tzinfo=None) # make timezone naive
        dusktime = pd.to_datetime(dusktime) # convert to pandas datetime timestamp format
        dictevents[dusktime]="dusk"
        # print(dusktime)
    except:
        # dusktime = float("nan")  # will later drop
        print(f"no dusk (UTC) on date {year}{month}{day} at {site}")
    # print(type(dusktime))
    
    return dictevents



def identify_timeofday(timestamp_nysm, dictinput):
    """
    Input one timestampe (each 5 min entry of nysm data)
    Input dictionary of that date and 2 surrounding date sun events
    Return the time of day that the 5 min timestamp belongs to 
    """
    list_times = list(dictinput.keys()) # list keys (sun event times) from dict
    list_times.append(timestamp_nysm) # append time of interest (row df)
    list_times_sorted = sorted(list_times) # sort from least to greatest
    placement = list_times_sorted.index(timestamp_nysm) # find the index of the timestamp of interest in the sorted list
    # print(placement)
    if placement == 0: # if the nysm timestamp is first element of list, there is no sun event before it, so look to the next sun event for categorizing
        nexttime = list_times_sorted[placement+1]
        nextsunevent = dictinput[nexttime]
        if nextsunevent == "dawn":
            timeofday = "night"
        elif nextsunevent == "sunrise":
            timeofday = "dawn"
        elif nextsunevent == "sunset":
            timeofday = "day"
        elif nextsunevent == "dusk":
            timeofday = "dusk"
        else:
            timeofday = float("nan")
    else: 
        # print("entered else")
        prevtime = list_times_sorted[placement-1]
        prevsunevent = dictinput[prevtime ]
        # print(prevsunevent)
        if prevsunevent == "dawn":
            timeofday = "dawn"
        elif prevsunevent == "sunrise":
            timeofday = "day"
        elif prevsunevent == "sunset":
            timeofday = "dusk"
        elif prevsunevent == "dusk":
            timeofday = "night"
        else:
            timeofday = float("nan")

    
    return  timeofday
# Examples to test functions above ^

timeinput1 = pd.to_datetime("2022-05-06 05:05:05")
timeinput1
print(timeinput1)

# timeinput1 = date_series[1]
# print(timeinput1)

k = sunevents(timeinput1, 42.418171, -79.3666, "fred")

print(k)

l= identify_timeofday(timeinput1, k)

print(l)
# print(o)


d = pd.read_csv("/home/csutter/DRIVE-clean/operational_inference/data_1_images/2025/09/23/20250923_2345/step1_imgfiles.csv")

print(d[0:3])
print(d.columns)

def eventtime(row):
    t = row["img_orig"][-23:][:-4]
    return t

d["time"] = d.apply(eventtime, axis =1 )

print(d[0:3])
# def sunevent_row(row):
#     k = sunevents(row[""], 42.418171, -79.3666, "fred")
