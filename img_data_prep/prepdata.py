# Code written with the help of Gemini

import os
import cv2
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count

# Prevent OpenCV from competing with Python's multiprocessing
cv2.setNumThreads(0) 

def process_single_image(args):
    """
    Bare-bones worker function. 
    Takes a tuple of (input_path, output_path). 
    Does the math and saves the file. NO logging, NO folder checking.
    """
    input_img_path, output_webp_path = args
    try:
        # Read image
        img = cv2.imread(input_img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return False
            
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            return False

        # Crop top 20%
        crop_height = int(0.2 * h)
        img_cropped = img[crop_height:, :, :]

        # Resize to 224x224
        img_resized = cv2.resize(img_cropped, (224, 224))

        # Save directly to .webp
        success = cv2.imwrite(output_webp_path, img_resized)
        return success

    except Exception:
        return False


def batch_convert_images(list_of_img_paths, base_output_dir):
    """
    Pre-calculates paths, builds directories instantly, and launches parallel workers.
    """
    print("Step 1: Parsing paths and mapping new directory structure...")
    
    tasks = []
    unique_dirs = set()
    
    # 1. Prepare all the input/output paths in memory first
    for img_path in list_of_img_paths:
        p = Path(img_path)
        filename_no_ext = p.stem
        camera_id = p.parent.parent.name  # Extracts SITE / camera_id
        
        # Extract date from filename (last 19 chars)
        date_str = filename_no_ext[-19:]
        year = date_str[0:4]
        month = date_str[5:7]
        day = date_str[8:10]
        yyyymmdd = f"{year}{month}{day}" # Needed for your requested structure
        
        # Build requested output structure: .../SITE/YYYY/MM/DD/YYYYMMDD/image.webp
        out_dir = Path(base_output_dir) / camera_id / year / month / day / yyyymmdd
        out_file = out_dir / f"{filename_no_ext}.webp"
        
        # Add to sets/lists
        unique_dirs.add(out_dir)
        tasks.append((str(p), str(out_file)))
        
    print(f"Step 2: Creating {len(unique_dirs)} unique folders...")
    # 2. Create all directories at once (WAY faster than checking per-image)
    for d in unique_dirs:
        d.mkdir(parents=True, exist_ok=True)
        
    print(f"Step 3: Launching parallel processing for {len(tasks)} images...")
    # 3. Process the images in parallel across all available CPU cores
    num_cores = cpu_count()
    print(f"--> Utilizing {num_cores} CPU cores.")
    
    # Use Pool to distribute the tasks
    success_count = 0
    with Pool(processes=num_cores) as pool:
        # imap_unordered is much faster than standard map for large datasets
        for i, result in enumerate(pool.imap_unordered(process_single_image, tasks, chunksize=100)):
            if result:
                success_count += 1
            
            # Print a progress update every 10,000 images so you know it's not frozen
            if (i + 1) % 10000 == 0:
                print(f"Processed {i + 1} / {len(tasks)} images...")

    print("========================================")
    print("DONE!")
    print(f"Successfully converted {success_count} out of {len(tasks)} images.")
    print("========================================")


# --- Execution Example ---
# Load your dataframe
# listims = ["/home/csutter/cron/data/NYSDOT_174008/20220112/QEW_E_Welland_Canal__Unknown__NYSDOT_174008_2022-01-12-00:09:05.jpg", 
# "/home/csutter/cron/data/Skyline_6499/20220112/I_90_at_Interchange_57_(Hamburg)__Eastbound__Skyline_6499_2022-01-12-00:15:32.jpg"]
# base_out = "/home/csutter/cron/data_convert_dissertation_images/data_images"

batch_convert_images(list_of_img_paths = listims, base_output_dir = base_out)