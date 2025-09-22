import inference_calibration
import inference_cnn
import inference_downstream
import inference_ensemble

inference_run_tracker = (
    "/home/csutter/DRIVE-clean/operational_inference/data_1_images/example_small.csv"
)

# Where the hrrr data lives
hrrr_data_csv = "/home/csutter/DRIVE/weather_img_concatmodels/cnn_hrrr_fcsthr2/nestedcv_imgname_hrrrdata_fcsthr2.csv"

model_nums = ["m0","m1","m2","m3","m4"] #
yyyymmdd = "20250919"

# Step A
# inference_cnn.cnn_run(inference_run_tracker = inference_run_tracker, model_nums = model_nums, yyyymmdd = yyyymmdd)

# Step B
# inference_calibration.calib_run(model_nums = model_nums, yyyymmdd = yyyymmdd, classif_model = "CNN")

# Step C
inference_downstream.downstream_run(model_nums = model_nums, yyyymmdd = yyyymmdd, hrrrdatapath = hrrr_data_csv)