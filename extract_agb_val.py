import pandas as pd
import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import Point
from rasterio.mask import mask
import zarr
import gc
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Paths
shapefile_path = 's3://ctrees-input-data/tree_height/Mexico/Mexico/Data/NFI/Daso_A_Cgl_C3_V7_24052022.csv'
raster_paths = {
    "mex_100m_2015": "/Users/stephaniegeorge/Documents/ctrees/Projects/Mexico_agb_validation/100m/mex_2020_agb_100m_cog.tif",
    "mex_100m_2020": "/Users/stephaniegeorge/Documents/ctrees/Projects/Mexico_agb_validation/100m/mex_2020_agb_100m_cog.tif"
}
output_dir = '/Users/stephaniegeorge/Documents/ctrees/Projects/Mexico_agb_validation/100m/'

def create_buffer(row, radius=56.42):
    """Creates a circular buffer around the given coordinates."""
    point = Point(row['X_C3'], row['Y_C3'])
    return point.buffer(radius)

def extract_raster_values(gdf, raster_path, map_name):
    """Extracts mean raster values for the 1 ha buffer locations."""
    raster = rasterio.open(raster_path)
    stats = []

    for _, row in gdf.iterrows():
        geometry = row['geometry']
        try:
            out_image, _ = mask(raster, [geometry], crop=True)
            out_image = out_image[out_image > 0]  # Filter out no-data values
            if out_image.size > 0:  # Ensure there's data to calculate mean
                mean_value = np.mean(out_image) / 10  # Divide by 10
            else:
                mean_value = np.nan  # Assign NaN if no valid data
        except ValueError:
            mean_value = np.nan  # Handle any unexpected errors
        stats.append(mean_value)
    
    return stats

def save_to_zarr(data, map_name, output_dir):
    """Saves the data to a Zarr file."""
    zarr_file = f"{output_dir}/{map_name}_values.zarr"
    store = zarr.DirectoryStore(zarr_file)
    root = zarr.group(store=store)
    root.create_dataset('mean_values', data=data, chunks=(1000,), dtype='f4')

def perform_regression_and_plot(gdf, map_name, is_first_plot):
    """Performs regression and plots observed vs. predicted values."""
    if 'Daso_A_Biomasa_aerea_Ton/HA' not in gdf.columns or map_name not in gdf.columns:
        print(f"Skipping regression plot for {map_name} as required columns are missing.")
        return
    
    plt.figure(figsize=(10, 6))
    X = gdf[['Daso_A_Biomasa_aerea_Ton/HA']].values
    y = gdf[map_name].values

    # Remove NaNs
    valid = ~np.isnan(X).flatten() & ~np.isnan(y)
    X = X[valid]
    y = y[valid]
    
    if len(X) == 0:
        print(f"No valid data for regression plot for {map_name}.")
        return
    
    reg = LinearRegression().fit(X, y)
    predictions = reg.predict(X)
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)

    plt.scatter(X, y, color='lightblue', label='Data Points')
    plt.plot(X, predictions, color='orange', linewidth=2, label='Fit Line')
    plt.plot(X, X, color='black', linestyle='--', label='1:1 Line')
    
    title_str = f'{map_name}: Observed vs. Predicted'
    if is_first_plot:
        plt.title(f'{title_str}\n(R²={r2:.2f}, RMSE={np.sqrt(mse):.2f})')
        plt.legend()
    else:
        plt.title(title_str)
    
    plt.xlabel('Observed (Daso_A_Biomasa_aerea_Ton/HA)')
    plt.ylabel('Predicted')
    plt.gca().set_xlim([0, plt.gca().get_xlim()[1]])  # Ensure x-axis is not cut off
    plt.gca().set_ylim([0, plt.gca().get_ylim()[1]])  # Ensure y-axis is not cut off
    
    # Add metrics inside the plot margins
    plt.text(0.05, 0.95, f'R²={r2:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    plt.text(0.05, 0.90, f'RMSE={np.sqrt(mse):.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    
    plt.tight_layout(pad=2.0)  # Adjust space to avoid overlap
    plt.savefig(f'{output_dir}/{map_name}_regression_plot.png')
    plt.close()
    print(f"Regression plot for {map_name} saved.")

# Read the shapefile using pandas
gdf = pd.read_csv(shapefile_path, encoding='ISO-8859-1')

# Remove rows where 'Daso_A_Biomasa_aerea_Ton/HA' is NA
gdf = gdf.dropna(subset=['Daso_A_Biomasa_aerea_Ton/HA'])

# Filter rows where 'Daso_A_Biomasa_aerea_Ton/HA' is greater than 0 (no sampling)
filtered_gdf = gdf[gdf['Daso_A_Biomasa_aerea_Ton/HA'] > 0]

# Check if filtered_gdf is empty
if filtered_gdf.empty:
    raise ValueError("Filtered GeoDataFrame is empty. Please check your data filtering criteria.")

# Ensure that 'filtered_gdf' is a GeoDataFrame
if 'geometry' not in filtered_gdf.columns:
    filtered_gdf['geometry'] = filtered_gdf.apply(lambda row: Point(row['X_C3'], row['Y_C3']), axis=1)
filtered_gdf = gpd.GeoDataFrame(filtered_gdf, geometry='geometry', crs="EPSG:4326")

# Extract raster values for each map and save them as Zarr datasets
for map_name, raster_path in raster_paths.items():
    print(f"Processing {map_name}...")
    values = extract_raster_values(filtered_gdf, raster_path, map_name)
    save_to_zarr(values, map_name, output_dir)
    gc.collect()  # Free up memory

# Process each map and save the Zarr datasets
for map_name in raster_paths.keys():
    zarr_file_path = f"{output_dir}/{map_name}_values.zarr"
    try:
        store = zarr.DirectoryStore(zarr_file_path)
        root = zarr.group(store=store)
        values = root['mean_values'][:]
        filtered_gdf[map_name] = values
    except KeyError as e:
        print(f"KeyError: {e} while accessing Zarr data for {map_name}")
    gc.collect()  # Free up memory

# Perform regression and save the plots
for i, map_name in enumerate(raster_paths.keys()):
    perform_regression_and_plot(filtered_gdf, map_name, is_first_plot=(i == 0))

# Save the final dataset (optional: shapefile or CSV)
buffered_csv_path = f'{output_dir}/extracted_val_data.csv'
filtered_gdf.drop(columns='geometry').to_csv(buffered_csv_path, index=False)
print(f"Final CSV saved to {buffered_csv_path}")