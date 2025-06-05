# pandas
# config
import os

# UPDATE TO BETH IS WHEN IN __MAIN
# from src  import config as config
import config as config

# tensor flow for data load and mobilenet preprocessing
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import trackers_to_run as trackers_to_run
from tensorflow import keras
from tensorflow.keras.applications.mobilenet import preprocess_input

# tensorflow model stuff
# input tracker, modelname
# read in data


def run_results_allobservations_for_tracker(tracker):
    # tracker = "/home/csutter/DRIVE/dot/model_trackpaths/fold6_nested_OT0_m0_T1V2.csv"

    print(f"on {tracker}")

    bathsizedef = 32
    TARGET_SIZE = (224, 224)  # Adjust based on model needs
    BATCH_SIZE = 32  # Adjust as needed
    SHUFFLE_BUFFER_SIZE = 1000  # Helps randomize training order

    # nestcv_OT0_m0_T1V2

    # make string descriptions of hyperparameters for file naming
    l2use_desc = str(config.l2_weight).replace(".", "_")
    dropoutuse_desc = str(config.dropout_rate).replace(".", "_")
    batchuse_desc = str(config.batch_size_def).replace(".", "_")
    lrinituse_desc = str(config.lr_init).replace(".", "_")
    lrdecruse_desc = str(config.lr_decayrate).replace(".", "_")

    configdetails = f"arch_{config.arch}_l2{l2use_desc}_dr{dropoutuse_desc}_b{batchuse_desc}_lr{lrinituse_desc}_lrdecr{lrdecruse_desc}_e{config.epoch_def}_es{config.earlystop_patience}_emin{config.min_epochs_before_es}"

    b = tracker.rfind("/")
    runname = tracker[b + 1 : -4]

    # modeldir = "/home/csutter/DRIVE/dot/model_json/fold6_nested_OT0_m0_T1V2_arch_mobilenet_l21e-05_dr0_4_b32_lr0_01_lrdecr0_99_e2_es10_emin30"

    modeldir = f"/home/csutter/DRIVE/dot/model_json/{runname}_{configdetails}"

    # print(modelpath)

    print(modeldir)

    print("modeldir")
    df_all = pd.read_csv(f"{tracker}")
    # df = pd.read_csv(f"{tracker}")
    # df_all = df.head(50) # EVAL ON ALL DATA! Dont need the phases

    print(df_all.columns)

    # print(df[0:15])
    # just us a small subset 20 to test

    # if config.kfoldnested == True:
    #     df_all = df[df[f"innerPhase"] == "innerTrain"]
    #     df_all = df_all.head(20)
    #     df_val = df[df[f"innerPhase"] == "innerVal"]

    # else:
    #     print("not set up for other types of data")

    # COPIED AND PASTED FROM BUILD_COMPILE
    print(f"NUMBER OF OBSERVATOINS:: {len(df_all)}")
    # print(f"NUMBER OF VAL:: {len(df_val)}")
    numims_train = len(df_all)  # var used to calc what weights should be
    df_all_size = len(df_all)
    train_labels = df_all["img_cat"]  # just for grabbing class weights

    train_images = list(df_all["img_orig"])  # 3/12
    train_labels = list(df_all["img_cat"])
    # train_labels = df_all["img_cat"]#  Replace with numeric or one-hot encoded labels

    def read_imgs_as_np_array(listims, listlabels):
        """
        Read in images as pixels and skip those that are severely corrupted.
        Note the checking for severe corruption really isn't needed for training bc we have a preprocessing step that removes all the poorly corrupted images BUT keeping this code because it will be helpful for inference.
        """
        brokenimgs = []
        images_pixel = []
        labels_imgs = []
        for im_i in range(0, len(listims)):  # len(listims) range(8800, 8900)
            if im_i % 1000 == 0:
                print(im_i)

            try:
                image_array = cv2.imread(listims[im_i], cv2.IMREAD_UNCHANGED)
                # print(image_array)
                # Convert BGR (OpenCV default) to RGB
                image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

                # Crop 20% from the top)
                h, w, _ = image_array.shape
                crop_height = int(0.2 * h)
                image_array = image_array[crop_height:, :, :]

                image_array = cv2.resize(image_array, TARGET_SIZE)  # Resize

                # Apply MobileNet preprocessing
                image_array = preprocess_input(image_array)

                images_pixel.append(image_array)
                labels_imgs.append(listlabels[im_i])

            except:
                brokenimgs.append(listims[im_i])
        # for broken in brokenimgs:
        #     print(f"total number of corrupted images {len(broken)}")
        #     print(f"corrupted image that can't be fixed: {broken}")
        print(f"number of broken imgs is {len(brokenimgs)}")
        return images_pixel, labels_imgs

    print("before running")
    print(len(train_images))
    print(len(train_labels))
    imgs_train, cats_train = read_imgs_as_np_array(train_images, train_labels)
    print("check")
    print(len(train_images))
    print(len(imgs_train))

    # COMMENT OUT 2 END

    print("place2")
    print(type(imgs_train))
    # print(imgs_train[0:5])
    # print(imgs_train[0][0])

    # previously was hard coded
    cs = (
        config.category_dirs
    )  # ["wet", "dry", "snow", "snow_severe", "obs", "poor_viz"]

    cats_alphabetical = sorted(cs)

    # make a list of values 0 through 5 for 6-cats. Will be dynamic if cat length changes

    print("place3")
    print(np.unique(cats_train))
    cat_inds = [i for i in range(0, len(cats_alphabetical))]

    inputdictmap = dict(zip(cats_alphabetical, cat_inds))

    label_encoding_train = [inputdictmap[catname] for catname in cats_train]

    # to subset
    # label_encoding_train = [inputdictmap[catname] for catname in train_labels[0:SUBSETNUM]]
    # label_encoding_val = [inputdictmap[catname] for catname in val_labels[0:SUBSETNUM]]

    # previously was hardcoded as 6
    numcats = config.cat_outputs
    print("numcats")
    print(numcats)

    train_labels_one_hot = np.eye(numcats)[label_encoding_train]

    print("place4")
    # Ensure dtype is float32 for ims and int for labels
    images_fortfd_train = np.array(imgs_train, dtype=np.float32)
    labels_fortfd_train = np.array(train_labels_one_hot, dtype=np.int32)

    print(
        "part5 data prepped before making it to tensor slices, see if it slows down here and if so then it's just the tf dataset creation that is taking time"
    )
    dataset_train = tf.data.Dataset.from_tensor_slices(
        (images_fortfd_train, labels_fortfd_train)
    )

    print("part6")

    print(type(dataset_train))
    print("PART6VAL")

    # Count the number of elements (images) in the dataset
    num_images = sum(1 for _ in dataset_train)
    print(f"Number of images in the dataset: {num_images}")
    # Create a TensorFlow dataset from the NumPy array
    # dataset = tf.data.Dataset.from_tensor_slices(image_array)

    # Print dataset structure
    print(dataset_train)

    # REMOVE SHUFFLING LINE!!!
    dataset_train = dataset_train.batch(BATCH_SIZE)  # Batch the data
    dataset_train = dataset_train.prefetch(tf.data.AUTOTUNE)  # Optimize pipeline

    print("part7")

    # read in model and evaluate data

    def my_loss_function(y_true, y_pred):
        print("y_pred dtype in loss:", y_pred.dtype)  # Check this!
        print("y_true dtype in loss:", y_true.dtype)  # Check this!
        print("y_true shape in loss:", y_true.shape)  # Check this!
        print("y_pred shape in loss:", y_pred.shape)  # Check this!
        loss = tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred)
        return loss

    model2 = keras.models.load_model(
        modeldir, custom_objects={"my_loss_function": my_loss_function}
    )

    # apply model on all data (entire tracker) to get predictions

    p2 = model2.predict(dataset_train)
    # class_pred_ind
    c2 = np.argmax(p2, axis=1)

    classdict = {}
    print("cats alph")
    print(np.unique(cats_alphabetical))
    for i in range(0, len(cats_alphabetical)):
        # append key value pair to classdict
        # dynamic for if changing the classes being used
        classdict[i] = cats_alphabetical[i]

    # previously was hard coded as
    # classdict = {
    #     0: "dry",
    #     1: "obs",
    #     2: "poor_viz",
    #     3: "snow",
    #     4: "snow_severe",
    #     5: "wet",
    # }

    print(len(c2))

    print(c2[0:5])

    predicted_classname = [classdict[i] for i in c2]

    print(len(predicted_classname))

    print(predicted_classname[0:5])

    # just a df of results will concat these COLS to the end of the tracker df

    print("major check here")
    print(np.unique(p2[0]))
    print(len(p2[0]))
    columns = [f"prob_{classdict[i]}" for i in range(len(classdict))]
    print(np.unique(columns))

    df_results = pd.DataFrame(
        p2, columns=[f"prob_{classdict[i]}" for i in range(len(classdict))]
    )

    df_final = pd.concat([df_all, df_results], axis=1)

    df_final["model_pred"] = predicted_classname

    print(df_final[0:10])

    df_final.to_csv(
        f"/home/csutter/DRIVE/dot/model_trackpaths_results/halvedShuffle/{runname}_{configdetails}.csv"  # UPDATE HERE!!! For subdir or wherever to save these results. And also be sure to update the trackers_to_run.py accordingly
    )

    # variations ^:
    # twotrain_evensplit
    # onetrain_fourfolds

    # main original way w 3 vs 1 folds for CNN train (innerTrain) vs RF train (innerTest)
    # df_final.to_csv(
    #     f"/home/csutter/DRIVE/dot/model_trackpaths_results/{runname}_{configdetails}.csv"
    # )

    # Results step 1: save out predictions for each tracker for its model

    ### IGNORE BELOW! Dont do ensembling at the CNN level. Why? Bc for the weather concat model, it's trained on CNN's val. But the val is different for each of the models that would go into an ensemble. Therefore need to do 30x weather models
    # Results step 2: do 5-model ensemble -- do this by grabbing the ONE tracker, grabbing the corresponding predictions of the relevant models (there are 5 of them done in step 1) and then do an ensemble. Vote count, as well as average probabilties. Can play with also doing some combination of avg probabilities and then if there is a large std (which can make the probability way less reliable) then do vote avg and also consider probabi
    # Note!! Need to consdier if want to do ensembling BEFORE doing concat model or not... for example, we could do it a couple ways:
    # way 1: decide on ONE cnn-predictions, i.e. do the ensembling now. Get probabilties to feed into concat model
    # way 2: do concat model

    # # average and get final
    # j = (p1 + p2 + p3 + p4 + p5) / 5
    # class_pred_ind = np.argmax(j, axis=1)

    # count_folds_pred_max = []
    # count_distinct_preds = []
    # stats_dry = []
    # stats_obs = []
    # stats_poor_viz = []
    # stats_snow = []
    # stats_snow_severe = []
    # stats_wet = []

    # for i in range(0, len(j)):
    #     # the max index is
    #     maxind_is = class_pred_ind[i]
    #     preds_eachfold = [c1[i], c2[i], c3[i], c4[i], c5[i]]
    #     countoutof5 = preds_eachfold.count(maxind_is)
    #     count_folds_pred_max.append(countoutof5)
    #     # get the total number of unique preds
    #     count_distinct_preds.append(len(np.unique(preds_eachfold)))
    #     # get stats by cat (for each obs) across folds
    #     # dry_ls = [p1[0],p2[0],p3[0],p4[0],p5[0]]
    #     # stats_dry.append([np.std(dry_ls),np.mean(dry_ls)])
    #     # obs_ls = [p1[1],p2[1],p3[1],p4[1],p5[1]]
    #     # stats_obs.append([np.std(obs_ls),np.mean(obs_ls)])
    #     # poor_viz_ls = [p1[2],p2[2],p3[2],p4[2],p5[2]]
    #     # stats_poor_viz.append([np.std(poor_viz_ls),np.mean(poor_viz_ls)])
    #     # snow_ls = [p1[3],p2[3],p3[3],p4[3],p5[3]]
    #     # stats_snow.append([np.std(snow_ls),np.mean(snow_ls)])
    #     # snow_severe_ls = [p1[4],p2[4],p3[4],p4[4],p5[4]]
    #     # stats_snow_severe.append([np.std(snow_severe_ls),np.mean(snow_severe_ls)])
    #     # wet_ls = [p1[5],p2[5],p3[5],p4[5],p5[5]]
    #     dry_ls = [p1[i][0], p2[i][0], p3[i][0], p4[i][0], p5[i][0]]
    #     stats_dry.append(np.std(dry_ls))
    #     # stats_dry.append([np.std(dry_ls),np.mean(dry_ls)])
    #     obs_ls = [p1[i][1], p2[i][1], p3[i][1], p4[i][1], p5[i][1]]
    #     stats_obs.append(np.std(obs_ls))
    #     # stats_obs.append([np.std(obs_ls),np.mean(obs_ls)])
    #     poor_viz_ls = [p1[i][2], p2[i][2], p3[i][2], p4[i][2], p5[i][2]]
    #     stats_poor_viz.append(np.std(poor_viz_ls))
    #     # stats_poor_viz.append([np.std(poor_viz_ls),np.mean(poor_viz_ls)])
    #     snow_ls = [p1[i][3], p2[i][3], p3[i][3], p4[i][3], p5[i][3]]
    #     stats_snow.append(np.std(snow_ls))
    #     # stats_snow.append([np.std(snow_ls),np.mean(snow_ls)])
    #     snow_severe_ls = [p1[i][4], p2[i][4], p3[i][4], p4[i][4], p5[i][4]]
    #     stats_snow_severe.append(np.std(snow_severe_ls))
    #     # stats_snow_severe.append([np.std(snow_severe_ls),np.mean(snow_severe_ls)])
    #     wet_ls = [p1[i][5], p2[i][5], p3[i][5], p4[i][5], p5[i][5]]
    #     stats_wet.append(np.std(wet_ls))
    #     # stats_wet.append([np.std(wet_ls),np.mean(wet_ls)])

    # # save out a version of the tracker w all predictions, to new dir, here:


# run_results_allobservations_for_tracker("/home/csutter/DRIVE/dot/model_trackpaths/nestcv_OT0_m0_T1V2.csv")


#### IF WANT TO RUN FOR ALL CNNS THAT HAVE BEEN TRAINED IN TRACKER FILE
#### Comment this out unless running this script, bc o/w when reading in this script into otehr steps of the work, the import line will run this entire script!

tracker_files = trackers_to_run.trackers_list
print("number of tracker files that should be ran")
print(len(tracker_files))

for t in tracker_files:
    print(t)
    run_results_allobservations_for_tracker(t)
