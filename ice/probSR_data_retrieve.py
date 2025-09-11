# Data provided by Heather Reeves
# NSSL - NOAA National Severe Storms Laboratory

import requests
import xarray as xr
import pandas as pd


### example of opening NYSM data which is saved as .nc
# open_files = ["your_file.nc"]
# # "/home/csutter/NYSM/netcdf/proc/2024/02/20240201.nc"
# print(type(open_files))
# print(open_files[0])
# print(f" open files {open_files}")
# # fil = open_files[1]
# # print("listed files to open")
# # print(f"files to open are {open_files}")
# df = xr.open_mfdataset(open_files, parallel=True).to_dataframe().reset_index()
# print(df[0:4])


import requests

fileinstance = "20230101-010000"
url = f"https://data.nssl.noaa.gov/thredds/fileServer/WRDD/TAT/Data/ALBANY/probsrt/00.00/{fileinstance}.netcdf"
filename = f"{fileinstance}"
# "20230101-000000.netcdf"
filepath_netcdf = f"/home/csutter/DRIVE-clean/ice/data_probSR/{filename}.netcdf"
filepath_parquet = f"/home/csutter/DRIVE-clean/ice/data_probSR/{filename}.parquet"

### STEP 1: Download the data from NOAA NSSL and save it locally as .netcdf
# this is one file, will need to loop through to download them all

# try:
#     response = requests.get(url, stream=True)
#     response.raise_for_status()  # This will raise an exception if the request fails

#     with open(filepath, "wb") as f:
#         for chunk in response.iter_content(chunk_size=8192):
#             f.write(chunk)

#     print(f"File '{filepath}' downloaded successfully.")

# except requests.exceptions.RequestException as e:
#     print(f"An error occurred: {e}")

### STEP 2: Load the local data to view information/make sure data is there
ds = xr.open_dataset(filepath_netcdf)

# print basic info
print(ds)
print("\nVariables:", list(ds.data_vars))
print("\nDimensions:", ds.dims)
print("\nCoordinates:", ds.coords)

### STEP 3: Save out as parquet file
### Convert to DF to do analysis on
# convert to DF and flatten data to save as parquet file
# df_all = ds.to_dataframe().reset_index()


# df_all.to_parquet(filepath_parquet)

# print(f"Saved as {filepath_parquet}")

### STEP 4: Remove the .netcdf files, which are redundant now that we have the .parquet files saved... need do add this

### STEP 5: Make sure data is there. Read in one parquet file and do some basic stats

probsr_df = pd.read_parquet(filepath_parquet)
probsr_df = probsr_df.reset_index()
print("sample of prob_sr dataframe")
print(probsr_df[0:4])
print(probsr_df.columns)

### STEP 3B -- need to add this in: Preprocessing to add in other information into the dataset, like lat and lon, and times (based on the file name). O/w the dataset as is just has Lat and Lons (unusable in current form, see Heather's code), and don't have any date cols, which may be helpful even if slightly repetitive w dates being in the name of the file
