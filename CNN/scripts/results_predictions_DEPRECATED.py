# 6/28 - probably need to deprecate this

# Portions of this code were writen with the assistance of AI tools (e.g., ChatGPT).

import os

# UPDATE TO BETH IS WHEN IN __MAIN
# from src  import config as config
import _config as config
import helper_fns_adhoc
import model_build

# tensor flow for data load and mobilenet preprocessing
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

# import trackers_to_run as trackers_to_run
from tensorflow import keras
from tensorflow.keras.applications.mobilenet import preprocess_input


def evaluate(
    modeldir,
    dataset,
    imgnames,
    trackerinput,
    saveflag=False,
    saveto="",
    run_arch=config.arch_set,
    run_trle=config.transfer_learning,
    run_ast=config.ast,
    run_dr=config.dr_set,
    run_l2=config.l2_set,
):
    """For a given run, evaluate each of the 30 CNNs on the full dataset; which is the same for all 30 models. Each model differed in terms of the data (folds) that was used for training and validation, but evaluation should be run on the full dataset (all folds), which is the same. Thus, to save memory and data loading time, load the full dataset just once, and then evaluate that same dataset on each of the 30 models, rather than loading the same data 30 times.

    Args:
        modeldir (str): path to one model
        dataset (tf dataset): data that is alrealy prepared for evaluation on model
        imgnames (list): list of image names, which is used for connecting preds to the full tracker that has all info by observation
        trackerinput (str): path to original tracker
        saveflag (bool, optional): whether to save prediction csvs out (usually only set to True for one-off models, don't want to save 30x csvs for each BL and hyperparam model). Defaults to False.
        saveto (str, optional): path to save csv to (if saveflag is True). Defaults to "".

    Returns:
        Pandas dataframe: model predictions for each observation
        String: tracker name
    """

    # print(modeldir)

    # 3. Explicitly delete the model object if it exists from a previous iteration
    # This ensures no old model graph/weights linger.
    if "model" in locals() and model is not None:
        del model
        model = None  # Make sure the reference is truly cleared

    print(f"Recreate model architecture")

    model = model_build.model_baseline(
        # one_off = config.one_off,
        # hyp_run = config.hyp_run,
        evid=config.evid,
        num_classes=config.cat_num,
        input_shape=(config.imheight, config.imwidth, 3),
        arch=run_arch,
        transfer_learning=run_trle,
        ast=run_ast,
        dropout_rate=run_dr,
        l2weight=run_l2,
        activation_layer_def=config.activation_layer_def,
        activation_output_def=config.activation_output_def,
    )

    print(f"Load model weights {modeldir}")

    # Find the path to the latest checkpoint file within that directory
    # tf.train.latest_checkpoint() will return the base name of the checkpoint (e.g., "best_weights")
    # prefixed with the directory path.
    latest_checkpoint_path = tf.train.latest_checkpoint(modeldir)

    #  Load the weights into the recreated model
    model.load_weights(latest_checkpoint_path)
    print(f"Weights loaded successfully from: {latest_checkpoint_path}")

    print("PRINT before predict")
    print(dataset.element_spec)
    dataset_for_prediction = dataset.map(lambda x, y: x)
    print(dataset_for_prediction.element_spec)

    # print("parse out inputs only")
    # dataset_inputs = dataset.map(lambda x, y: x)
    # print(dataset_inputs.element_spec)
    # for i, x in enumerate(dataset_inputs):
    #     print(f"Batch {i}: shape = {x.shape}")
    # Try converting the dataset to a NumPy array (if possible) and predicting on that
    # x_all = tf.concat([x for x, y in dataset], axis=0)
    # print(type(x_all))
    # X_list = []
    # for x_batch, _ in dataset:
    #     X_list.append(x_batch)

    # X_all = tf.concat(X_list, axis=0)

    # print(type(X_all))

    # X_all = tf.concat([x.numpy() for x, _ in dataset_inputs], axis=0)

    # # --- Crucial Check: Is the dataset actually empty? ---
    # is_dataset_empty = True
    # try:
    #     # Attempt to get the first element from the dataset.
    #     # .take(1) creates a new dataset containing at most 1 element.
    #     # next(iter(...)) tries to get that element.
    #     first_element = next(iter(dataset_for_prediction.take(1)))

    #     # If we got here, the dataset is not entirely empty.
    #     is_dataset_empty = False
    #     print(f"DEBUG: First batch shape: {first_element.shape}")

    #     # Also check if the first batch itself has a batch size of 0.
    #     # This might still cause issues downstream in some models.
    #     if first_element.shape[0] == 0:
    #         print("WARNING: First batch has a batch size of 0. This might be problematic.")
    #         # Depending on your model's robustness, you might still want to skip here.
    #         # is_dataset_empty = True # Uncomment if zero-size first batch is a critical failure point

    # except tf.errors.OutOfRangeError:
    #     # This error is raised if the iterator runs out of elements immediately (i.e., dataset is empty).
    #     is_dataset_empty = True
    #     print(f"WARNING: Dataset is empty for iteration {i} (tf.errors.OutOfRangeError)!")
    # except Exception as e:
    #     # Catch any other unexpected errors during dataset peeking, treat as problematic.
    #     print(f"ERROR: Problem peeking into dataset for iteration {i}: {e}")
    #     is_dataset_empty = True # Treat as empty/problematic if an error occurs

    # if is_dataset_empty:
    #     print(f"SKIPPING model.predict for iteration {i} because dataset is empty or problematic.")
    #     # Use 'continue' to skip to the next iteration of your main loop
    #     # or 'break' if you want to stop the entire loop.
    #     # continue

    print(
        f"DEBUG: Starting detailed batch inspection for iteration in loop"
    )  # 'i' from your main loop

    # Use model.predict_on_batch for detailed debugging of individual batches
    # (This will build up predictions, but primarily for debugging)
    all_predictions_from_manual_batches = []

    # IMPORTANT: Ensure your dataset is repeatable if you plan to iterate over it
    # for this check AND then again for model.predict(). If not repeatable,
    # the second call to model.predict() will fail because the dataset is exhausted.
    # If your dataset is not naturally repeatable (e.g., from_generator without .repeat()),
    # you might need to recreate 'dataset_for_prediction' after this inspection block.
    # Alternatively, if this check passes, you can just use all_predictions_from_manual_batches.

    try:
        for batch_idx, batch_data in enumerate(dataset_for_prediction):
            print(
                f"  Inspecting batch {batch_idx} (Iteration). Shape: {batch_data.shape}, Dtype: {batch_data.dtype}"
            )

            # Check for problematic batch shapes:
            if not tf.is_tensor(batch_data):
                print(
                    f"    ERROR: Batch {batch_idx} is not a tensor! Type: {type(batch_data)}"
                )
                raise ValueError("Batch data is not a tensor (Non-Tensor Batch).")

            if batch_data.shape.rank == 0:  # Scalar batch
                print(
                    f"    ERROR: Batch {batch_idx} is a scalar! Shape: {batch_data.shape}, Value: {batch_data.numpy()}"
                )
                raise ValueError("Batch data is a scalar (Scalar Batch).")

            if batch_data.shape[0] == 0:  # Empty batch
                print(
                    f"    WARNING: Batch {batch_idx} has a batch size of 0. This might be problematic for model.predict."
                )
                # This is a very strong candidate for causing issues downstream.
                # You might want to filter these out in your dataset pipeline if they are not expected.
                # raise ValueError("Batch has zero size (Empty Batch).") # Uncomment to fail immediately on empty batch

            # Check for NaN/Inf (critical for numerical stability)
            if tf.reduce_any(tf.math.is_nan(batch_data)):
                print(f"    ERROR: Batch {batch_idx} contains NaN values! (Iteration)")
                raise ValueError("Batch contains NaN values.")
            if tf.reduce_any(tf.math.is_inf(batch_data)):
                print(f"    ERROR: Batch {batch_idx} contains Inf values! (Iteration)")
                raise ValueError("Batch contains Inf values.")

            # Try predicting on a single batch (this is the most direct test)
            try:
                batch_preds = model.predict_on_batch(batch_data)
                all_predictions_from_manual_batches.append(batch_preds)
            except Exception as batch_e:
                print(
                    f"    ERROR: model.predict_on_batch failed for batch {batch_idx} (Iteration). Error: {batch_e}"
                )
                # This is the crucial point! If predict_on_batch fails, the issue is with this specific batch's content.
                # You can save this problematic batch for later inspection:
                # tf.saved_model.save(batch_data, f"problem_batch_i{i}_b{batch_idx}") # For TF2.x
                raise  # Re-raise to stop the loop and debug this specific batch

        print(
            f"DEBUG: All batches successfully processed via predict_on_batch for iteration."
        )

        # If you reached here, all batches passed the individual checks and predict_on_batch.
        # If the original error still happens *after* this exhaustive check, and you *didn't*
        # recreate the dataset, then it means the issue is *still* in model.predict()'s
        # length inference when operating on the full, original dataset object.

        # IMPORTANT: If dataset_for_prediction is not repeatable, you need to either:
        # 1. Recreate it here before calling model.predict() again.
        #    dataset_for_prediction = dataset.map(lambda x, y: x)
        # 2. Or, use the results from your manual iteration:
        #    p2 = tf.concat(all_predictions_from_manual_batches, axis=0) # If your model output is concatenatable

        # For now, let's assume it's repeatable or you're recreating it.
        p2 = model.predict(
            dataset_for_prediction
        )  # This is where the original error happens

    except Exception as full_dataset_iter_e:
        print(
            f"ERROR: An exception occurred during full dataset iteration (Iteration): {full_dataset_iter_e}"
        )
        print(
            "This indicates an issue with a specific batch within the dataset, or a dataset pipeline issue not caught by individual predict_on_batch calls."
        )
        raise  # Re-raise to get the full traceback at the exact failure point.

    p2 = model.predict(dataset_for_prediction)

    print("PRINT after predict")
    c2 = np.argmax(p2, axis=1)

    print("Complete with evaluate() in results_predictions.py")

    # note: this is not set up right now for evid, which requires loading of the custom loss function to deserialize (and that requires class weights which are unique to each of the 30 datasets, come back to this..

    dict_catKey_indValue, dict_indKey_catValue = helper_fns_adhoc.cat_str_ind_dictmap()

    predicted_classname = [dict_indKey_catValue[i] for i in c2]

    df_results = pd.DataFrame(
        p2,
        columns=[
            f"prob_{dict_indKey_catValue[i]}" for i in range(len(dict_indKey_catValue))
        ],
    )

    df_results["model_pred"] = predicted_classname
    df_results["img_name"] = imgnames

    print(df_results[0:10])
    print(df_results.columns)
    print(df_results["img_name"][0])

    # # connect to the unique tracker (unique for each of the 30 datasets)
    df_all = pd.read_csv(trackerinput)
    print(df_all.columns)
    df_final = df_all.merge(df_results, how="inner", on="img_name")

    print(len(df_final))

    tracker_ident = helper_fns_adhoc.tracker_differentiator(trackerpath=trackerinput)

    df_final["tracker"] = tracker_ident

    t_name = os.path.basename(trackerinput)[:-4]

    if saveflag:
        # predssaveto = f"{config.preds_path}/{t_name}_{rundetails}.csv"
        df_final.to_csv(saveto)  # saving the preds df to csv for one-off runs
        print("saved to")
        print(saveto)

    return df_final, t_name
