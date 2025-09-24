import glob
import os
from datetime import datetime, timedelta

# import config
import numpy as np
import pandas as pd

def time_current():

    # grab the current date so we know which dir to look in
    current_datetime = datetime.now()
    print(current_datetime)
    print(type(current_datetime))

    y = current_datetime.strftime("%Y")
    m = current_datetime.strftime("%m")
    d = current_datetime.strftime("%d")
    currentdate = current_datetime.strftime("%Y%m%d")
    currenthr = current_datetime.strftime("%H")
    currenthr_int = int(currenthr)
    currentmin = current_datetime.strftime("%M")
    currentmin_int = int(currentmin)

    dirstructure = f"{y}/{m}/{d}/{currentdate}_{currenthr}{currentmin}"
    print(dirstructure)

    print(currentdate, currenthr, currentmin)

    return y,m,d,currentdate,currenthr,currenthr_int, currentmin,currentmin_int,dirstructure

def time_given(timestampstring = "20240919_2355"):

    # grab the current date so we know which dir to look in
    current_datetime = datetime.now()
    print(current_datetime)
    print(type(current_datetime))

    y = timestampstring[0:4]
    m = timestampstring[4:6]
    d = timestampstring[6:8]
    print(y,m,d)
    currentdate = timestampstring[0:8]

    currenthr = timestampstring[9:11]
    print(currenthr)
    currenthr_int = int(currenthr)
    currentmin = timestampstring[11:13]
    currentmin_int = int(currentmin)

    print(currentdate, currenthr, currentmin)

    dirstructure = f"{y}/{m}/{d}/{currentdate}_{currenthr}{currentmin}"
    print(dirstructure)
    
    return y,m,d,currentdate,currenthr,currenthr_int, currentmin,currentmin_int,dirstructure

# def dir_structures(parentdir_forimgtracker, parentdir_forhrrrtracker):
#     imgdir = f"{parentdir_forimgtracker}/{y}/{m}/{d}/{currentdate}_{currenthr}{currentmin}"

#     hrrrdir = f"{parentdir_forhrrrtracker}/{y}/{m}/{d}/{currentdate}_{currenthr}{currentmin}"

#     os.makedirs(imgdir, exist_ok = True)
#     os.makedirs(hrrrdir, exist_ok = True)

#     return imgdir, hrrrdir