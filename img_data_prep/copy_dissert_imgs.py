# Written with help of Gemini
# Code keeps full quality copies of images used in dissertation. Approx ~20 million including inference / verification chapters. Not a "hard" copy, but another link to the image bytes, but it's effectively the same outcome as making a copy. More details below. 

"""
Organizes full-quality images used in dissertation analyses (~20M images).

Streams paths directly from operational run CSVs and creates filesystem hard links 
(os.link) into the standardized date/site directory hierarchy. 

Note on storage:
Uses hard links instead of byte copies to avoid duplicating ~10+ TB of storage. 
Each image points directly to existing disk inodes (0 extra disk space used). 
Deleting the original source directory later will safely leave these files fully intact.
"""

import os
import csv
from glob import iglob
from concurrent.futures import ProcessPoolExecutor
from itertools import islice


# --- CONFIGURATION FOR SMALL TEST OF IMAGES [ ONE DIR ] ---
# # 1. Point to just ONE specific run folder
# BASE_SEARCH_DIR = "/home/csutter/DRIVE-clean/operational_runs/set58_buffer_blizzard2"

# # 2. Point to a dedicated test directory (so it doesn't mix with production data)
# BASE_OUTPUT_DIR = "/home/csutter/cron/data_dissertation_images_originalCopies/smallTest4"

# # 3. Lower worker count for a small run
# NUM_WORKERS = 2
# CHUNK_SIZE = 100

# # # --- CONFIGURATION ---
BASE_SEARCH_DIR = "/home/csutter/DRIVE-clean/operational_runs"
BASE_OUTPUT_DIR = "/home/csutter/cron/data_dissertation_images_originalCopies/data_images"

NUM_WORKERS = 16        # Optimal concurrency for an NFS export
CHUNK_SIZE = 5000       # Batch size sent to worker processes


def parse_and_map_path(img_path: str, base_out: str):
    """
    Parses original file path and calculates target folder and filename.
    Matches structure: .../SITE/YYYYMMDD/<filename>.jpg -> .../SITE/YYYY/MM/DD/YYYYMMDD/<filename>.ext
    """
    img_path = os.path.normpath(img_path)
    parent_dir, filename = os.path.split(img_path)
    site_dir, yyyymmdd = os.path.split(parent_dir)
    camera_id = os.path.basename(site_dir)

    name_no_ext, _ = os.path.splitext(filename)

    # Date extraction from the last 19 characters (YYYY-MM-DD-HH:MM:SS)
    date_str = name_no_ext[-19:]
    year = date_str[0:4]
    month = date_str[5:7]
    day = date_str[8:10]

    target_dir = os.path.join(base_out, camera_id, year, month, day, yyyymmdd)
    target_file = os.path.join(target_dir, filename)

    return img_path, target_dir, target_file


def process_batch(batch_records):
    """
    Worker task: creates parent directories and executes the hard link.
    """
    known_dirs = set()
    success = 0
    failed = 0

    for src, out_dir, dst in batch_records:
        try:
            if out_dir not in known_dirs:
                os.makedirs(out_dir, exist_ok=True)
                known_dirs.add(out_dir)

            try:
                # HARD LINK: Points new path directly to existing inode
                os.link(src, dst)
            except FileExistsError:
                pass  # Avoids errors if resumed or duplicate entries exist

            success += 1
        except Exception:
            failed += 1

    return success, failed


def stream_image_paths_from_csvs(root_dir: str):
    """
    Finds all step1_imgfiles.csv files and yields paths line by line.
    Never stores 20 million paths in memory simultaneously.
    """
    # iglob yields paths one by one rather than building a giant list
    pattern = os.path.join(root_dir, "**", "step1_imgfiles.csv")
    print(f"Scanning for CSV files matching: {pattern}")

    csv_count = 0
    for csv_path in iglob(pattern, recursive=True):
        csv_count += 1
        try:
            with open(csv_path, mode="r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if not header:
                    continue

                try:
                    col_idx = header.index("img_orig")
                except ValueError:
                    # Skip files missing the 'img_orig' column
                    continue

                for row in reader:
                    if len(row) > col_idx:
                        path_val = row[col_idx].strip()
                        if path_val:
                            yield path_val
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")

    print(f"Found and streamed from {csv_count:,} CSV files.")


def chunked_iterable(iterable, chunk_size=5000):
    """Packs stream items into lists of `chunk_size`."""
    it = iter(iterable)
    while True:
        chunk = list(islice(it, chunk_size))
        if not chunk:
            break
        yield chunk


def run_pipeline():
    total_processed = 0
    total_success = 0
    total_failed = 0

    print("Launching streaming hard-link pipeline...")
    print(f"Workers: {NUM_WORKERS} | Chunk Size: {CHUNK_SIZE}")

    path_stream = stream_image_paths_from_csvs(BASE_SEARCH_DIR)

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        in_flight = {}

        for raw_chunk in chunked_iterable(path_stream, CHUNK_SIZE):
            batch = [parse_and_map_path(p, BASE_OUTPUT_DIR) for p in raw_chunk]
            future = executor.submit(process_batch, batch)
            in_flight[future] = len(raw_chunk)

            # Limit backlog of unfinished tasks to bound memory
            while len(in_flight) >= NUM_WORKERS * 3:
                done = [f for f in in_flight if f.done()]
                if not done:
                    import time
                    time.sleep(0.02)
                    continue

                for df in done:
                    s, f = df.result()
                    total_success += s
                    total_failed += f
                    total_processed += in_flight.pop(df)

                if total_processed % 50000 < CHUNK_SIZE:
                    print(f"Processed: {total_processed:,} | Linked: {total_success:,} | Errors: {total_failed:,}")

        # Finish remaining batches
        for future in in_flight:
            s, f = future.result()
            total_success += s
            total_failed += f
            total_processed += in_flight[future]

    print("\n" + "=" * 40)
    print("FINISHED")
    print(f"Total Processed: {total_processed:,}")
    print(f"Successfully Linked: {total_success:,}")
    print(f"Failed: {total_failed:,}")
    print("=" * 40)


# Keep this 2-line guard so background workers don't self-replicate
if __name__ == "__main__":
    run_pipeline()