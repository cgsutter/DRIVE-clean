import inference_calibration
import inference_cnn
# import inference_downstream
import inference_ensemble

inference_run_tracker = (
    "/home/csutter/DRIVE-clean/operational_inference/data_1_images/example_small.csv"
)

model_nums = ["m0","m1","m2","m3","m4"] #
yyyymmdd = "20250919"

# inference_cnn.cnn_run(inference_run_tracker = inference_run_tracker, model_nums = model_nums, yyyymmdd = yyyymmdd)

# model_nums = ["m0","m1","m2","m3","m4"] #
# yyyymmdd = "20250919"
# classif_model = "downstream"  # "CNN" or "downstream" or "fcstOnly" # HERE!!

# classif_model = "downstream"  # "CNN" or "downstream" or "fcstOnly" # HERE!!

inference_calibration.calib_run(model_nums = model_nums, yyyymmdd = yyyymmdd, classif_model = "CNN")
