#### DEPRECATED -- dont need this file. Preds are run in CNN code. 

# # pandas
# # config
# import os

# # NEED TO UPDATE AT SOME POINT TO PUT IN __MAIN
# # from src  import config as config
# import config as config

# # tensor flow for data load and mobilenet preprocessing
# import cv2
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# import trackers_to_run as trackers_to_run
# from tensorflow import keras
# from tensorflow.keras.applications.mobilenet import preprocess_input

# # tensorflow model stuff
# # input tracker, modelname
# # read in data

# descname = "ynobs_A"  # global variable # HERE!!

# run_all_observations = True  # for the full 22k dataset # HERE!!
# run_ynobs_resultsonly = False  # for just the ynobs dataset # HERE!!


# def run_results_allobservations_for_tracker(df_all):
#     # tracker = "/home/csutter/DRIVE/dot/model_trackpaths/fold6_nested_OT0_m0_T1V2.csv"

#     # print(f"on {tracker}")

#     bathsizedef = 32
#     TARGET_SIZE = (224, 224)  # Adjust based on model needs
#     BATCH_SIZE = 32  # Adjust as needed
#     SHUFFLE_BUFFER_SIZE = 1000  # Helps randomize training order

#     # nestcv_OT0_m0_T1V2

#     # # make string descriptions of hyperparameters for file naming
#     # l2use_desc = str(config.l2_weight).replace(".", "_")
#     # dropoutuse_desc = str(config.dropout_rate).replace(".", "_")
#     # batchuse_desc = str(config.batch_size_def).replace(".", "_")
#     # lrinituse_desc = str(config.lr_init).replace(".", "_")
#     # lrdecruse_desc = str(config.lr_decayrate).replace(".", "_")

#     # configdetails = f"arch_{config.arch}_l2{l2use_desc}_dr{dropoutuse_desc}_b{batchuse_desc}_lr{lrinituse_desc}_lrdecr{lrdecruse_desc}_e{config.epoch_def}_es{config.earlystop_patience}_emin{config.min_epochs_before_es}"

#     # b = tracker.rfind("/")
#     # runname = tracker[b + 1 : -4]

#     # modeldir = "/home/csutter/DRIVE/dot/model_json/fold6_nested_OT0_m0_T1V2_arch_mobilenet_l21e-05_dr0_4_b32_lr0_01_lrdecr0_99_e2_es10_emin30"

#     modeldir_0 = f"/home/csutter/DRIVE/dot/model_json/{descname}_arch_mobilenet_spl0_tune_l21e-05_dr0_4_b32_lr0_01_lrdecr0_99_e75_es10_emin30"
#     modeldir_1 = f"/home/csutter/DRIVE/dot/model_json/{descname}_arch_mobilenet_spl1_tune_l21e-05_dr0_4_b32_lr0_01_lrdecr0_99_e75_es10_emin30"

#     # print(modelpath)

#     # df_all = pd.read_csv(f"{tracker}")
#     # df = pd.read_csv(f"{tracker}")
#     # df_all = df.head(50) # EVAL ON ALL DATA! Dont need the phases

#     print(df_all.columns)

#     # print(df[0:15])
#     # just us a small subset 20 to test

#     # if config.kfoldnested == True:
#     #     df_all = df[df[f"innerPhase"] == "innerTrain"]
#     #     df_all = df_all.head(20)
#     #     df_val = df[df[f"innerPhase"] == "innerVal"]

#     # else:
#     #     print("not set up for other types of data")

#     # COPIED AND PASTED FROM BUILD_COMPILE
#     print(f"NUMBER OF OBSERVATOINS:: {len(df_all)}")
#     # print(f"NUMBER OF VAL:: {len(df_val)}")
#     numims_train = len(df_all)  # var used to calc what weights should be
#     df_all_size = len(df_all)
#     train_labels = df_all["img_cat"]  # just for grabbing class weights

#     train_images = list(df_all["img_orig"])  # 3/12
#     train_labels = list(df_all["img_cat"])
#     # train_labels = df_all["img_cat"]#  Replace with numeric or one-hot encoded labels

#     def read_imgs_as_np_array(listims, listlabels):
#         """
#         Read in images as pixels and skip those that are severely corrupted.
#         Note the checking for severe corruption really isn't needed for training bc we have a preprocessing step that removes all the poorly corrupted images BUT keeping this code because it will be helpful for inference.
#         """
#         brokenimgs = []
#         images_pixel = []
#         labels_imgs = []
#         for im_i in range(0, len(listims)):  # len(listims) range(8800, 8900)
#             if im_i % 1000 == 0:
#                 print(im_i)

#             try:
#                 image_array = cv2.imread(listims[im_i], cv2.IMREAD_UNCHANGED)
#                 # print(image_array)
#                 # Convert BGR (OpenCV default) to RGB
#                 image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

#                 # Crop 20% from the top)
#                 h, w, _ = image_array.shape
#                 crop_height = int(0.2 * h)
#                 image_array = image_array[crop_height:, :, :]

#                 image_array = cv2.resize(image_array, TARGET_SIZE)  # Resize

#                 # Apply MobileNet preprocessing
#                 image_array = preprocess_input(image_array)

#                 images_pixel.append(image_array)
#                 labels_imgs.append(listlabels[im_i])

#             except:
#                 brokenimgs.append(listims[im_i])
#         # for broken in brokenimgs:
#         #     print(f"total number of corrupted images {len(broken)}")
#         #     print(f"corrupted image that can't be fixed: {broken}")
#         print(f"number of broken imgs is {len(brokenimgs)}")
#         return images_pixel, labels_imgs

#     print("before running")
#     print(len(train_images))
#     print(len(train_labels))
#     imgs_train, cats_train = read_imgs_as_np_array(train_images, train_labels)
#     print("here print!!")
#     print(len(train_images))
#     print(len(imgs_train))

#     # COMMENT OUT 2 END

#     print("place2")
#     print(type(imgs_train))
#     # print(imgs_train[0:5])
#     # print(imgs_train[0][0])

#     cs = ["obs", "nonobs"]  # HERE! Hardcoded only for the ynobs model
#     cats_alphabetical = sorted(cs)

#     # make a list of values 0 through 5 for 6-cats. Will be dynamic if cat length changes

#     print("place3")
#     cat_inds = [i for i in range(0, len(cats_alphabetical))]

#     inputdictmap = dict(zip(cats_alphabetical, cat_inds))

#     label_encoding_train = [inputdictmap[catname] for catname in cats_train]

#     # to subset
#     # label_encoding_train = [inputdictmap[catname] for catname in train_labels[0:SUBSETNUM]]
#     # label_encoding_val = [inputdictmap[catname] for catname in val_labels[0:SUBSETNUM]]

#     numcats = 2  # HERE! Hardcoded only for the ynobs model

#     train_labels_one_hot = np.eye(numcats)[label_encoding_train]

#     print("place4")
#     # Ensure dtype is float32 for ims and int for labels
#     images_fortfd_train = np.array(imgs_train, dtype=np.float32)
#     labels_fortfd_train = np.array(train_labels_one_hot, dtype=np.int32)

#     print(
#         "part5 data prepped before making it to tensor slices, see if it slows down here and if so then it's just the tf dataset creation that is taking time"
#     )
#     dataset_train = tf.data.Dataset.from_tensor_slices(
#         (images_fortfd_train, labels_fortfd_train)
#     )

#     print("part6")

#     print(type(dataset_train))
#     print("PART6VAL")

#     # Count the number of elements (images) in the dataset
#     num_images = sum(1 for _ in dataset_train)
#     print(f"Number of images in the dataset: {num_images}")
#     # Create a TensorFlow dataset from the NumPy array
#     # dataset = tf.data.Dataset.from_tensor_slices(image_array)

#     # Print dataset structure
#     print(dataset_train)

#     # REMOVE SHUFFLING LINE!!!
#     dataset_train = dataset_train.batch(BATCH_SIZE)  # Batch the data
#     dataset_train = dataset_train.prefetch(tf.data.AUTOTUNE)  # Optimize pipeline

#     print("part7")

#     # read in model and evaluate data

#     def my_loss_function(y_true, y_pred):
#         print("y_pred dtype in loss:", y_pred.dtype)  # Check this!
#         print("y_true dtype in loss:", y_true.dtype)  # Check this!
#         print("y_true shape in loss:", y_true.shape)  # Check this!
#         print("y_pred shape in loss:", y_pred.shape)  # Check this!
#         loss = tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred)
#         return loss

#     modelload0 = keras.models.load_model(
#         modeldir_0, custom_objects={"my_loss_function": my_loss_function}
#     )

#     modelload1 = keras.models.load_model(
#         modeldir_1, custom_objects={"my_loss_function": my_loss_function}
#     )

#     # apply model on all data (entire tracker) to get predictions

#     p0 = modelload0.predict(dataset_train)
#     # class_pred_ind
#     c0 = np.argmax(p0, axis=1)

#     p1 = modelload1.predict(dataset_train)
#     # class_pred_ind
#     c1 = np.argmax(p1, axis=1)

#     classdict = {}
#     for i in range(0, len(cats_alphabetical)):
#         # append key value pair to classdict
#         # dynamic for if changing the classes being used
#         classdict[i] = cats_alphabetical[i]

#     # previously was hard coded as
#     # classdict = {
#     #     0: "dry",
#     #     1: "obs",
#     #     2: "poor_viz",
#     #     3: "snow",
#     #     4: "snow_severe",
#     #     5: "wet",
#     # }

#     print(len(c0))

#     print(c0[0:5])

#     predicted_classname_0 = [classdict[i] for i in c0]
#     predicted_classname_1 = [classdict[i] for i in c1]

#     print(len(predicted_classname_0))

#     print(predicted_classname_0[0:5])

#     # just a df of results will concat these COLS to the end of the tracker df
#     df_results_0 = pd.DataFrame(
#         p0, columns=[f"prob_{classdict[i]}" for i in range(len(classdict))]
#     )
#     df_results_1 = pd.DataFrame(
#         p1, columns=[f"prob_{classdict[i]}" for i in range(len(classdict))]
#     )

#     print("check here to make sure all good before concat")
#     print(len(df_all))
#     print(len(df_results_0))
#     df_results_0 = df_results_0.reset_index()
#     df_results_1 = df_results_1.reset_index()

#     df_final_0 = pd.concat([df_all, df_results_0], axis=1)
#     df_final_0["model_pred"] = predicted_classname_0

#     df_final_1 = pd.concat([df_all, df_results_1], axis=1)
#     df_final_1["model_pred"] = predicted_classname_1

#     print(df_final_0[0:10])

#     df_final_0.to_csv(
#         f"/home/csutter/DRIVE/dot/model_trackpaths_results/{descname}_split0.csv"
#     )

#     df_final_1.to_csv(
#         f"/home/csutter/DRIVE/dot/model_trackpaths_results/{descname}_split1.csv"
#     )


# # for adding phase col which will be helpful for then training calibration model
# def rowfn_phase_split0(row):
#     if row["foldnum_ynobs"] == 0:
#         phaseset = "val"
#     else:
#         phaseset = "train"
#     return phaseset


# def rowfn_phase_split1(row):
#     if row["foldnum_ynobs"] == 1:
#         phaseset = "val"
#     else:
#         phaseset = "train"
#     return phaseset


# def run_results_ynobs_only(df_all, splitnumstr):
#     """
#     df_all dataframe needs to be prepared already
#     splitnumstr is string "0" or "1"
#     """
#     # tracker = "/home/csutter/DRIVE/dot/model_trackpaths/fold6_nested_OT0_m0_T1V2.csv"

#     # print(f"on {tracker}")

#     bathsizedef = 32
#     TARGET_SIZE = (224, 224)  # Adjust based on model needs
#     BATCH_SIZE = 32  # Adjust as needed
#     SHUFFLE_BUFFER_SIZE = 1000  # Helps randomize training order

#     modeldir = f"/home/csutter/DRIVE/dot/model_json/{descname}_arch_mobilenet_spl{splitnumstr}_tune_l21e-05_dr0_4_b32_lr0_01_lrdecr0_99_e75_es10_emin30"

#     # print(modelpath)

#     # df_all = pd.read_csv(f"{tracker}")
#     # df = pd.read_csv(f"{tracker}")
#     # df_all = df.head(50) # EVAL ON ALL DATA! Dont need the phases

#     print(df_all.columns)
#     print(df_all.dtypes)
#     print(df_all["foldnum_ynobs"][0:5])
#     if splitnumstr == "0":
#         ph = df_all.apply(rowfn_phase_split0, axis=1)
#     elif splitnumstr == "1":
#         ph = df_all.apply(rowfn_phase_split1, axis=1)
#     else:
#         print("issue w split num")

#     df_all["phase"] = ph
#     print(df_all.columns)

#     # print(df[0:15])
#     # just us a small subset 20 to test

#     # if config.kfoldnested == True:
#     #     df_all = df[df[f"innerPhase"] == "innerTrain"]
#     #     df_all = df_all.head(20)
#     #     df_val = df[df[f"innerPhase"] == "innerVal"]

#     # else:
#     #     print("not set up for other types of data")

#     # COPIED AND PASTED FROM BUILD_COMPILE
#     print(f"NUMBER OF OBSERVATOINS:: {len(df_all)}")
#     # print(f"NUMBER OF VAL:: {len(df_val)}")
#     numims_train = len(df_all)  # var used to calc what weights should be
#     df_all_size = len(df_all)
#     train_labels = df_all["img_cat"]  # just for grabbing class weights

#     train_images = list(df_all["img_orig"])  # 3/12
#     train_labels = list(df_all["img_cat"])
#     # train_labels = df_all["img_cat"]#  Replace with numeric or one-hot encoded labels

#     def read_imgs_as_np_array(listims, listlabels):
#         """
#         Read in images as pixels and skip those that are severely corrupted.
#         Note the checking for severe corruption really isn't needed for training bc we have a preprocessing step that removes all the poorly corrupted images BUT keeping this code because it will be helpful for inference.
#         """
#         brokenimgs = []
#         images_pixel = []
#         labels_imgs = []
#         for im_i in range(0, len(listims)):  # len(listims) range(8800, 8900)
#             if im_i % 1000 == 0:
#                 print(im_i)

#             try:
#                 image_array = cv2.imread(listims[im_i], cv2.IMREAD_UNCHANGED)
#                 # print(image_array)
#                 # Convert BGR (OpenCV default) to RGB
#                 image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

#                 # Crop 20% from the top)
#                 h, w, _ = image_array.shape
#                 crop_height = int(0.2 * h)
#                 image_array = image_array[crop_height:, :, :]

#                 image_array = cv2.resize(image_array, TARGET_SIZE)  # Resize

#                 # Apply MobileNet preprocessing
#                 image_array = preprocess_input(image_array)

#                 images_pixel.append(image_array)
#                 labels_imgs.append(listlabels[im_i])

#             except:
#                 brokenimgs.append(listims[im_i])
#         # for broken in brokenimgs:
#         #     print(f"total number of corrupted images {len(broken)}")
#         #     print(f"corrupted image that can't be fixed: {broken}")
#         print(f"number of broken imgs is {len(brokenimgs)}")
#         return images_pixel, labels_imgs

#     print("before running")
#     print(len(train_images))
#     print(len(train_labels))
#     imgs_train, cats_train = read_imgs_as_np_array(train_images, train_labels)
#     print("here print!!")
#     print(len(train_images))
#     print(len(imgs_train))

#     # COMMENT OUT 2 END

#     print("place2")
#     print(type(imgs_train))
#     # print(imgs_train[0:5])
#     # print(imgs_train[0][0])

#     cs = ["obs", "nonobs"]  # HERE! Hardcoded only for the ynobs model
#     cats_alphabetical = sorted(cs)

#     # make a list of values 0 through 5 for 6-cats. Will be dynamic if cat length changes

#     print("place3")
#     cat_inds = [i for i in range(0, len(cats_alphabetical))]

#     inputdictmap = dict(zip(cats_alphabetical, cat_inds))

#     label_encoding_train = [inputdictmap[catname] for catname in cats_train]

#     # to subset
#     # label_encoding_train = [inputdictmap[catname] for catname in train_labels[0:SUBSETNUM]]
#     # label_encoding_val = [inputdictmap[catname] for catname in val_labels[0:SUBSETNUM]]

#     numcats = 2  # HERE! Hardcoded only for the ynobs model

#     train_labels_one_hot = np.eye(numcats)[label_encoding_train]

#     print("place4")
#     # Ensure dtype is float32 for ims and int for labels
#     images_fortfd_train = np.array(imgs_train, dtype=np.float32)
#     labels_fortfd_train = np.array(train_labels_one_hot, dtype=np.int32)

#     print(
#         "part5 data prepped before making it to tensor slices, see if it slows down here and if so then it's just the tf dataset creation that is taking time"
#     )
#     dataset_train = tf.data.Dataset.from_tensor_slices(
#         (images_fortfd_train, labels_fortfd_train)
#     )

#     print("part6")

#     print(type(dataset_train))
#     print("PART6VAL")

#     # Count the number of elements (images) in the dataset
#     num_images = sum(1 for _ in dataset_train)
#     print(f"Number of images in the dataset: {num_images}")
#     # Create a TensorFlow dataset from the NumPy array
#     # dataset = tf.data.Dataset.from_tensor_slices(image_array)

#     # Print dataset structure
#     print(dataset_train)

#     # REMOVE SHUFFLING LINE!!!
#     dataset_train = dataset_train.batch(BATCH_SIZE)  # Batch the data
#     dataset_train = dataset_train.prefetch(tf.data.AUTOTUNE)  # Optimize pipeline

#     print("part7")

#     # read in model and evaluate data

#     def my_loss_function(y_true, y_pred):
#         print("y_pred dtype in loss:", y_pred.dtype)  # Check this!
#         print("y_true dtype in loss:", y_true.dtype)  # Check this!
#         print("y_true shape in loss:", y_true.shape)  # Check this!
#         print("y_pred shape in loss:", y_pred.shape)  # Check this!
#         loss = tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred)
#         return loss

#     modelload = keras.models.load_model(
#         modeldir, custom_objects={"my_loss_function": my_loss_function}
#     )

#     # apply model on all data (entire tracker) to get predictions

#     p0 = modelload.predict(dataset_train)
#     # class_pred_ind
#     c0 = np.argmax(p0, axis=1)

#     classdict = {}
#     for i in range(0, len(cats_alphabetical)):
#         # append key value pair to classdict
#         # dynamic for if changing the classes being used
#         classdict[i] = cats_alphabetical[i]

#     # previously was hard coded as
#     # classdict = {
#     #     0: "dry",
#     #     1: "obs",
#     #     2: "poor_viz",
#     #     3: "snow",
#     #     4: "snow_severe",
#     #     5: "wet",
#     # }

#     print(len(c0))

#     print(c0[0:5])

#     predicted_classname_0 = [classdict[i] for i in c0]

#     print(len(predicted_classname_0))

#     print(predicted_classname_0[0:5])

#     # just a df of results will concat these COLS to the end of the tracker df
#     df_results_0 = pd.DataFrame(
#         p0, columns=[f"prob_{classdict[i]}" for i in range(len(classdict))]
#     )
#     print("check here to make sure all good before concat")
#     print(len(df_all))
#     print(len(df_results_0))
#     df_results_0 = df_results_0.reset_index()

#     df_final_0 = pd.concat([df_all, df_results_0], axis=1)
#     df_final_0["model_pred"] = predicted_classname_0

#     df_final_0.to_csv(
#         f"/home/csutter/DRIVE/dot/model_trackpaths_results/{descname}_split{splitnumstr}_results.csv"
#     )  # note _results_ here to indicate its the small ynobs df only


# #### IF WANT TO RUN FOR BOTH SPLITS OF YNOBS AND ALL DATA INCLUDED 5CAT

# # combine the 5cat (w/ relabeled) and ynobs datasets

# if run_all_observations:
#     # ynobs dataset
#     ynobs = pd.read_csv(f"/home/csutter/DRIVE/dot/model_trackpaths/{descname}.csv")
#     print("num nonobs")
#     print(len(ynobs[ynobs["img_cat"] == "nonobs"]))  # to help w double checking amounts
#     ynobs = ynobs.rename(columns={"foldhalf": "foldnum_ynobs"})
#     print("cols 1")
#     print(ynobs.columns)
#     print(np.unique(ynobs["foldnum_nested"]))
#     print(np.unique(ynobs["foldnum_ynobs"]))

#     # nest5cat dataset
#     cat5 = pd.read_csv("/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat.csv")
#     cat5 = cat5.rename(
#         columns={"foldnum": "foldnum_nested"}
#     )  # will replace these classes with nonobs for model evaluation
#     cat5 = cat5.drop(columns=["img_cat"])
#     cat5["img_cat"] = "nonobs"
#     print("cols 2")
#     print(cat5.columns)
#     print(np.unique(cat5["foldnum_nested"]))

#     # make a small df which will be used to mapping foldnum_nest to foldnum from ynobs df
#     mapfolds = ynobs[["foldnum_ynobs", "foldnum_nested"]]
#     mapfolds = mapfolds.drop_duplicates().reset_index(drop=True)
#     # mapfolds = mapfolds.dropna()
#     # mapfolds = mapfolds.rename(columns = {"foldnum_ynobs":"foldnum_ynobs_all"})
#     print(mapfolds)

#     cat5 = cat5.merge(mapfolds, how="left", on="foldnum_nested")
#     print(len(ynobs))
#     print(len(cat5))
#     print(len(cat5))

#     print("cols3")
#     print(cat5.columns)
#     print("cats 1")
#     print(np.unique(ynobs["img_cat"]))
#     # cat5 = cat5.drop(columns = ["foldnum_ynobs"])
#     # cat5 = cat5.rename(columns = {"foldnum_ynobs_all":"foldnum_ynobs"})
#     # cat5 = cat5.drop(columns = ["_merge"])

#     # add in the OBS examples from ynobs that are not in cat5, then concat them
#     print("check here")
#     ynobs = ynobs.loc[:, ~ynobs.columns.str.contains("^Unnamed|^$", regex=True)]
#     ynobs_obs = ynobs[
#         ynobs["img_cat"] == "obs"
#     ]  # original way took the obs (ignored nonobs) from the ynobs dataset and concatted w the 5cat dataset which has the nonobs + other 5 cats, so all examples. Updated way below will merge the full ynobs df (not ynobs_obs) so we can grab which of the nonobs were used in training the ynobs model vs not

#     cat5 = cat5.loc[:, ~cat5.columns.str.contains("^Unnamed|^$", regex=True)]

#     print("cols here 12")
#     print(ynobs_obs.columns)
#     print(np.unique(ynobs_obs["img_cat"]))
#     print(cat5.columns)
#     print(np.unique(cat5["img_cat"]))

#     # original way concatted the two ynobs_obs and cat 5, but updated way we will merge with ynobs and cat5
#     # mc = pd.concat([ynobs_obs, cat5])
#     # print(ynobs.columns)
#     # print("trhoug here??")

#     # add in the observations that were used in both the ynobs training and are 5cat
#     mc1 = ynobs.merge(cat5[["img_name"]], how="inner", on="img_name")
#     mc1["in_ynobs"] = "included"
#     print("included")
#     print(len(mc1))

#     # add in observations that are 5 cat but were not in ynobs training
#     mc2 = cat5.merge(ynobs[["img_name"]], on="img_name", how="left", indicator=True)
#     mc2 = mc2[mc2["_merge"] == "left_only"].drop(columns=["_merge"])
#     mc2["in_ynobs"] = "excluded"
#     print("excluded")
#     print(len(mc2))

#     # add in obs examples (used in ynobs training, never in 5cat)
#     ynobs_obs["in_ynobs"] = "included"

#     mc = pd.concat([mc1, mc2, ynobs_obs])

#     print("total")
#     print(len(mc))
#     print(mc.columns)
#     print(mc[0:5])
#     print("here updated way!!")

#     print(np.unique(cat5["foldnum_ynobs"]))
#     print(np.unique(ynobs["foldnum_ynobs"]))
#     print(np.unique(mc["foldnum_ynobs"]))

#     print(np.unique(cat5["foldnum_nested"]))
#     print(np.unique(ynobs["foldnum_nested"]))
#     print(np.unique(mc["foldnum_nested"]))

#     print(mc.columns)
#     print(np.unique(mc["img_cat"]))

#     mc = mc.reset_index()

#     run_results_allobservations_for_tracker(mc)


# ### IF WANT TO RUN JUST THE YNOBS DATASET (FOR MODEL PERFORMANCE RESULTS)

# if run_ynobs_resultsonly:

#     print("running ynobs results only not all observations")

#     # combine the 5cat (w/ relabeled) and ynobs datasets

#     # ynobs dataset
#     ynobs = pd.read_csv(f"/home/csutter/DRIVE/dot/model_trackpaths/{descname}.csv")
#     print("num nonobs")
#     print(len(ynobs[ynobs["img_cat"] == "nonobs"]))  # to help w double checking amounts
#     ynobs = ynobs.rename(columns={"foldhalf": "foldnum_ynobs"})
#     print("cols 1")
#     print(ynobs.columns)
#     print(np.unique(ynobs["foldnum_nested"]))
#     print(np.unique(ynobs["foldnum_ynobs"]))

#     # add in the OBS examples from ynobs that are not in cat5, then concat them
#     print("check here")
#     mc = ynobs.loc[:, ~ynobs.columns.str.contains("^Unnamed|^$", regex=True)]

#     print("total")
#     print(len(mc))
#     print(mc.columns)
#     print(mc[0:5])
#     print("here updated way!!")

#     print(np.unique(ynobs["foldnum_ynobs"]))
#     print(np.unique(mc["foldnum_ynobs"]))

#     print(np.unique(ynobs["foldnum_nested"]))
#     print(np.unique(mc["foldnum_nested"]))

#     print(mc.columns)
#     print(np.unique(mc["img_cat"]))

#     mc = mc.reset_index()

#     run_results_ynobs_only(mc, splitnumstr="0")
#     run_results_ynobs_only(mc, splitnumstr="1")
