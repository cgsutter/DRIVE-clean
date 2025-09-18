# to do: combine the file saving step rather than case study vs current, make more efficient the time it takes to grab images? clean up code so that all of this is one function and it's being called? or maybe dont need to bc all of these will be in a cron job and we can run each script independently

import glob
import os
from datetime import datetime, timedelta

import config
import numpy as np
import pandas as pd

# print(config.parentdir)
# print(config.flag_now_event)


def testfn():
    print("hii!")


def step1_fn(rundate, runhour, dirsave, y, m, d, hour_str, min_str):

    print("Starting step 1, grabbing relevant image file paths")
    # runhour = config.run_hour

    # # first organize in the right subdir if it's a ciwro case study
    # if config.ciwro_run:
    #     subdir = f"{config.parentdir}/ciwro/{config.ciwro_run_identifier}"
    # else:
    #     subdir = config.parentdir

    # grab all site sub dir names
    sites = os.listdir("/home/csutter/cron/data")
    if ".ipynb_checkpoints" in sites:
        sites.remove(".ipynb_checkpoints")
    print(len(sites))

    ### COMMENTING OUT THE IF RUNNOW AND ELSE! WILL NEED TO ADD THIS BACK IN, WORKING ON ADDING BACK IN 916
    # if runnow == True:
    #     print("running current conditions relevant right now")

    #     # grab the current date so we know which dir to look in
    #     current_datetime = datetime.now()
    #     print(current_datetime)
    #     # print(current_datetime.strftime('%H%M%S'))
    #     # print(current_datetime.strftime('%M'))
    #     y = current_datetime.strftime('%Y')
    #     m = current_datetime.strftime('%m')
    #     d = current_datetime.strftime('%d')
    #     today = current_datetime.strftime('%Y%m%d')
    #     currenthr = current_datetime.strftime('%H')
    #     currentmin = current_datetime.strftime('%M')
    #     # also get yesterday's date. Bc when we pull the most recent image, if it happens to be midnight and there are no images yet, will have to look to yesterdays date to get the most recent one
    #     yesterday = current_datetime - timedelta(days=1)
    #     yesterday = yesterday.strftime('%Y%m%d')
    #     # print(today)
    #     # print(yesterday)

    #     dirsave = f"{subdir}/current/{y}/{m}/{d}/{today}_{currenthr}_{currentmin}"
    #     print(dirsave)
    #     # parentdir = f"/home/csutter/DRIVE/dot/inference_pipeline/case_studies/{daytopull}_{hourtopull}"
    #     if not os.path.exists(dirsave):
    #         os.makedirs(dirsave)
    #     print(f"created directory")

    #     # grab the most recent image file from each site
    #     # do this by listing all images from today and yesterday, ordering, and taking the first one (which would be the most recent)
    #     current_batch_of_images = []
    #     for i, s in enumerate(sites):
    #         # glob lists full file path rather than os.listdir which is just the file name, and would need to separately append the full path after listing files
    #         imgs_today = glob.glob(f"/home/csutter/cron/data/{s}/{today}/*")
    #         imgs_ydy = glob.glob(f"/home/csutter/cron/data/{s}/{yesterday}/*")
    #         imgs = imgs_today + imgs_ydy
    #         if i%100 == 0:
    #             print(f"done through site {i}")
    #             # print(f"number image timestamps considered, yesterday and today: {len(imgs)}")

    #         # quickest way is to sort reverse alphabetically take first item. Note: this is faster than using a package that finds modification time (see lag cron archive analysis above for an example doing that way)
    #         imgs.sort(reverse=True)
    #         current_batch_of_images.append(imgs[0])

    #     print(f"found {len(current_batch_of_images)} images")
    #     # print(current_batch_of_images[300:310])

    #     # make as df and save as csv to directory
    #     step1_imgfiles = pd.DataFrame({"img_model": current_batch_of_images, "site":sites})

    #     print(len(step1_imgfiles))

    #     # save out
    #     step1_imgfiles.to_csv(f"{dirsave}/step1_imgfiles.csv")

    #     print(f"saved out list of images to {dirsave}/step1_imgfiles.csv")

    # else: # FROM HERE WOULD NEED TO INDENT

    # print(f"running case study for {config.run_date} {config.run_hour}Z min {config.run_min}")

    print(f"running code for {rundate} {hour_str}{min_str}Z")

    # y = config.run_date[0:4]
    # m = config.run_date[4:6]
    # d = config.run_date[6:8]
    # hour_str = str(config.run_hour).zfill(2)
    # min_str = str(config.run_min).zfill(2)

    # # grab string subset that we want the image filenames to have, considering the edge cases between hours and dates
    # if config.run_hour == 0:
    #     datehour =

    # strftime

    # dirsave = f"{subdir}/case_studies/{y}/{m}/{d}/{config.run_date}_{hour_str}_{min_str}"
    # print(dirsave)
    # parentdir = f"/home/csutter/DRIVE/dot/inference_pipeline/case_studies/{daytopull}_{hourtopull}"
    # if not os.path.exists(dirsave):
    #     os.makedirs(dirsave)
    # print(f"created directory")

    # target datetime for differencing
    target_datetime = datetime.strptime(
        f"{y}-{m}-{d}-{hour_str}:{min_str}:00", "%Y-%m-%d-%H:%M:%S"
    )

    print(target_datetime)

    # for grabbing img file names
    today = rundate  # config.run_date
    yesterday = target_datetime - timedelta(days=1)
    yesterday = yesterday.strftime("%Y%m%d")
    tomorrow = target_datetime + timedelta(days=1)
    tomorrow = tomorrow.strftime("%Y%m%d")

    # grab the most recent image file from each site
    # do this by listing all images within 10 mins of the case study date|hour|min and then taking the closest time. We dont want to just take the *closest* time because since we're doing a case study, we want it to be accurate to the specific time given.
    # Note, may want to adjust the "current RSC" way to align w this too. For example if the most recent image is from 2 hours ago, probably don't want to use that.

    # # grab the time range, acocunting for cases near top of hour
    # # lower bound of range
    # if config.run_min >=5: # easy case
    #     min_lowerbound = config.run_min-5
    #     hour_lowerbound = config.run_hour
    # else:
    #     min_lowerbound = config.run_min-5+60
    #     hour_lowerbound = config.run_hour - 1
    # # upper bound of minute
    # if config.run_min <55: #easy case
    #     min_upperbound = config.run_min+5
    #     hour_upperbound = config.run_hour
    # else:
    #     min_upperbound = config.run_min+5-60
    #     hour_upperbound = config.run_hour + 1

    # create strings which will be used with wildcard characters to shorten the list of images being considered for each site.
    # in addition to the current date and hour, will also consider the previous hour and the next hour
    currenthour_string_imgname = f"{y}-{m}-{d}-{hour_str}:"
    # grab the previous and next hour, acocunting for cases near top of hour
    # lower bound of range
    # if hour of interest is 0, then consider previous day
    if runhour == 0:
        day_of_prevhour = yesterday
        lowerbound_string_imgname = (
            f"{yesterday[0:4]}-{yesterday[4:6]}-{yesterday[6:8]}-23:"
        )
    else:
        hour_minus1 = str(runhour - 1).zfill(2)
        day_of_prevhour = rundate  # same as today/date of interest
        lowerbound_string_imgname = f"{y}-{m}-{d}-{hour_minus1}:"
    # upper bound of range
    # if hour of interest is 23, then consider next day
    if runhour == 23:
        day_of_nexthour = tomorrow
        upperbound_string_imgname = (
            f"{tomorrow[0:4]}-{tomorrow[4:6]}-{tomorrow[6:8]}-00:"
        )
    else:
        hour_plus1 = str(runhour + 1).zfill(2)
        day_of_nexthour = rundate  # same as today/date of interest
        upperbound_string_imgname = f"{y}-{m}-{d}-{hour_plus1}:"

    # note for current RSC, there will be no consideration of images from the next hour bc it's the current time
    # for case study we want closer so just do the same way with time diffs and then ALSO check that its less than 10 min, o/w, dont use it! can do the same with "current" if we wish
    print("hour of interest")
    print(currenthour_string_imgname)
    print("next hour")
    print(upperbound_string_imgname)
    print("previous hour")
    print(lowerbound_string_imgname)

    print(
        f"{config.max_time_diff_imgs} minutes: Maximum difference between time of interest and most recent image considered is {config.max_time_diff_imgs} minutes"
    )
    max_min_diff = timedelta(minutes=config.max_time_diff_imgs)
    # print(type(max_min_diff))
    # print(max_min_diff)

    current_batch_of_images = []
    sites_with_images = []
    print("number of sites considered")
    print(len(sites))
    for i, s in enumerate(
        sites
    ):  # this is where you can limit the number of observations to test code out through all steps, just do site[10:20] here instead of all 2371 sites
        # print(s)
        # glob lists full file path rather than os.listdir which is just the file name, and would need to separately append the full path after listing files

        # glob takes a lot of time, especially when doing it 3 times
        # so subsetting so (to specific dates and hours) not necessaruly helpful. But did that to limit the amount of datetime diffs needed in the loop, thinking that was the time sink
        # the loop for diffs does take too long if you dont subset the glob list at all (i.e. cant just grab ALL images for that site, do need to subset some). But overall, teh globbing is what takes a while
        # the solution is likely to come back and reorganize all of the cron directory in site/y/m/d , although will that even help bc we'll still have 3 globs to do

        # old way (but need to grab tomorrow too)
        # imgs_today = glob.glob(f"/home/csutter/cron/data/{s}/{today}/*")
        # imgs_ydy = glob.glob(f"/home/csutter/cron/data/{s}/{yesterday}/*")
        # # imgs = imgs_today
        # imgs = imgs_today + imgs_ydy

        # print("check issue w running current")
        # print(f"/home/csutter/cron/data/{s}/{today}/*{currenthour_string_imgname}*")
        # print(f"/home/csutter/cron/data/{s}/{day_of_prevhour}/*{lowerbound_string_imgname}*")
        # print(f"/home/csutter/cron/data/{s}/{day_of_nexthour}/*{upperbound_string_imgname}*")
        # new way 1
        imgs_current_hour = glob.glob(
            f"/home/csutter/cron/data/{s}/{today}/*{currenthour_string_imgname}*"
        )
        # print("current hour dir of images")
        # print(len(imgs_current_hour))
        # print("example of one dir looking for")

        # print(len(imgs_current_hour))
        imgs_prev_hour = glob.glob(
            f"/home/csutter/cron/data/{s}/{day_of_prevhour}/*{lowerbound_string_imgname}*"
        )
        # print(len(imgs_prev_hour))
        imgs_next_hour = glob.glob(
            f"/home/csutter/cron/data/{s}/{day_of_nexthour}/*{upperbound_string_imgname}*"
        )
        # print(len(imgs_next_hour))
        imgs = imgs_current_hour + imgs_prev_hour + imgs_next_hour
        # print(s)
        # print(len(imgs))
        # /home/csutter/cron/data/NYSDOT_4861013/20240916
        # imgs = imgs_current_hour
        # print(imgs[9:12])

        # print(len(imgs))

        # # new way 2 (took the longest)
        # # maybe try globbing ALL in the directory and then grabbing all that contain the three dates: today, ydy, tomorrow.
        # # print(s)
        # imgs_current_hour = glob.glob(f"/home/csutter/cron/data/{s}/*/*")
        # filtered_list = [item for item in imgs_current_hour if any(sub in item for sub in [config.run_date, day_of_prevhour, day_of_nexthour])]

        # grab datetimes of all the img files being considered
        # the date and time in image name will always be positioned here bc img filename ends in .jpg (-4) and then working backwards it is 2024-01-09-22:10:00.jpg (-23 character spots)
        imgs_datetime_str = [
            im[len(im) - 23 : -4] for im in imgs
        ]  # grab the string date and time from file name
        # print(imgs_datetime_str)
        # convert to datetime format so that we can difference it with the datetime of interest
        imgs_datetime_dt = [
            datetime.strptime(dt_str, "%Y-%m-%d-%H:%M:%S")
            for dt_str in imgs_datetime_str
        ]

        # print("number of considered images:")
        # print(len(imgs_datetime_dt))
        # print("target datetime")
        # print(target_datetime)

        time_diffs = []
        for k in imgs_datetime_dt:
            diff = abs(k - target_datetime)
            time_diffs.append(diff)

        # print("time diffs")
        # print(len(time_diffs))
        # IMPORTANT! if find 0 usable images (i.e. the code in next line breaks) it's bc there are no images in the archive for the given data. Check k8 and archive database
        min_diff = min(time_diffs)

        # print(min_diff)
        # print(type(min_diff))
        # print("minimum time difference is found:")
        # print(min_diff)

        # here is where to add the maximum difference allowed! for example, for case study, maybe we dont want an image from an hour ago EVEN IF it IS the most recent image for whatever reason. To do this, just add an if statement: if min_diff < 10 min (check formatting there) then append. o/w dont append that image
        if min_diff <= max_min_diff:
            # print("min diff okay!")
            img_index_match = time_diffs.index(min_diff)
            img_grab = imgs[img_index_match]
            current_batch_of_images.append(img_grab)
            sites_with_images.append(s)
        # else:
        #     print("min diff not okay")

        # COMMENT OUT FOR MOST RUNS BUT UNCOMMENT TO SEE PROGRESS
        # if i % 100 == 0:
        #     print(f"done through site {i}")

    print(f"found {len(current_batch_of_images)} images")
    # print(current_batch_of_images[300:310])

    # make as df and save as csv to directory
    step1_imgfiles = pd.DataFrame(
        {"img_model": current_batch_of_images, "site": sites_with_images}
    )

    print(len(step1_imgfiles))

    print(f"{dirsave}/step1_imgfiles.csv")

    # save out
    step1_imgfiles.to_csv(f"{dirsave}/step1_imgfiles.csv")
