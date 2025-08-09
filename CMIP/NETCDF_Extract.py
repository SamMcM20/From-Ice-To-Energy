import xarray as xr
# 1. Define File Paths and Region of Interest 

# Use a raw string (r"...") to avoid errors with file paths
netcdf_file_path = r"C:\Users\samam\OneDrive\MSC Diss\Data\CMIP\CMIP2_45\pr_Amon_NorESM2-MM_ssp245_r1i1p1f1_gn_20260116-21001216.nc"
output_excel_path = r"C:\Users\samam\OneDrive\MSC Diss\Data\nordland_rainfall.xlsx"

# Define the latitude and longitude bounding box for Nordland county, Norway
# You can adjust these coordinates for more precision if needed.
nordland_bbox = {
    'lat': slice(65.2, 67.5),
    'lon': slice(12, 17.3)
}


#  2. Open Dataset and Extract Data 

# Open the NetCDF file
with xr.open_dataset(netcdf_file_path) as ds:
    print("--- Original Dataset Information ---")
    print(ds) # This helps verify variable names (e.g., 'pr', 'lat', 'lon')

    # Select the data for the Nordland region using the bounding box
    nordland_data = ds.sel(nordland_bbox)

    # Calculate the average precipitation across the spatial dimensions (lat, lon)
    regional_mean_precip = nordland_data['pr'].mean(dim=['lat', 'lon'])


# 3. Convert to Pandas DataFrame and Save to Excel 

# Convert the resulting xarray DataArray to a pandas DataFrame
df = regional_mean_precip.to_dataframe()

# Save the DataFrame to an Excel file
# The 'time' coordinate will automatically become the index of the spreadsheet
df.to_excel(output_excel_path)


print("\n--- Process Complete ---")
print(f"Data for Nordland has been saved to: {output_excel_path}")
print("\nPreview of the first 5 rows of data:")
print(df.head())