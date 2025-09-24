import grab_time
import grab_imgs
import grab_hrrr
import inference_calibration
import inference_cnn
import inference_downstream
import inference_ensemble
import os
from datetime import datetime

# inference_run_tracker = (
#     "/home/csutter/DRIVE-clean/operational_inference/data_1_images/tracker_m0.csv"
# )

START = datetime.now()



model_nums = ["m0","m1","m2","m3","m4"] #


## Prep time and date information -- run A or B
## come back and double check as not all of these may have been needed
# A: for using current time (for live/operational run)
# y,m,d,ymd,hr,hr_int,min,min_int,dirstructure = grab_time.time_current()

# B: for providing a given time (for case studies/past dates). Should be in UTC (+4EDT, +5EST)
y,m,d,ymd,hr,hr_int,min,min_int,dirstructure  = grab_time.time_given( timestampstring = "20250914_1230")

## Grab image data 
imgdir = f"/home/csutter/DRIVE-clean/operational_inference/data_1_images/{dirstructure}"
imgcsv = "step1_imgfiles.csv"
imgcsv_save = f"/home/csutter/DRIVE-clean/operational_inference/data_1_images/{dirstructure}/step1_imgfiles.csv"

grab_imgs.step1_fn(
    rundate=ymd,
    runhour=hr_int,  # this is an int. only need hour as int, dont need minute as int which is why we just have minute as string below
    saveimgcsv = imgcsv_save,
    y=y,
    m=m,
    d=d,
    hour_str=hr,
    min_str=min,
)

IMG = datetime.now()

## Grab HRRR data
hrrrcsv_save = f"/home/csutter/DRIVE-clean/operational_inference/data_1b_hrrr/{dirstructure}/step1b_hrrr.csv"

grab_hrrr.runall(fhnum_input = 2, imgdatacsv = imgcsv_save, hrrrdatacsv_tosave = hrrrcsv_save)

HRRR = datetime.now()


# ## Step A
inference_cnn.cnn_run(inference_run_tracker = imgcsv_save, model_nums = model_nums, yyyymmdd = dirstructure)

CNN = datetime.now()

# ## Step B
inference_calibration.calib_run(model_nums = model_nums, yyyymmdd = dirstructure, classif_model = "CNN")

## Step C
inference_downstream.downstream_run(model_nums = model_nums, yyyymmdd = dirstructure, hrrrdatapath = hrrrcsv_save)

# ## Step D
inference_calibration.calib_run(model_nums = model_nums, yyyymmdd = dirstructure, classif_model = "downstream")

# ## Step E
inference_ensemble.ensemble_run(model_nums = model_nums, yyyymmdd = dirstructure)

FINISH = datetime.now()

print(f"TIME CHECK DONE START: {START}")
print(f"TIME CHECK DONE IMG: {IMG}")
print(f"TIME CHECK DONE HRRR: {HRRR}")
print(f"TIME CHECK DONE CNN: {CNN}")
print(f"TIME CHECK DONE EVERYTHING ELSE: {FINISH}")
