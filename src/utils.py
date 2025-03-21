import numpy as np
from sklearn.cluster import KMeans

# Custom modules
import sys

sys.path.append("../")
import Datasets

import geopandas as gpd
import rasterio as rio
from rasterio.features import rasterize
from rasterio.transform import from_bounds

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors

def custom_colormap():
    """
    Returns a custom colormap for visualizing Gaussian Process outputs.
    """
    custom_colors = [
        "royalblue",
        "mistyrose",
        "salmon",
        "red",
        "darkred",
        "tan",
        "lawngreen",
        "darkgreen",
        "yellowgreen",
        "goldenrod",
        "lemonchiffon",
        "yellow",
        "violet",
        "lightblue",
        "cyan"
    ]
    print('ed')
    return colors.ListedColormap(custom_colors)



def standardize_data(x_train, x_val, y_train, y_val, scaler="minmax"):
    # Get the mean and standard deviation of the training data
    # and standardize the training and validation data using those stats
    # train_mean = x_train.mean(axis=0)
    # train_std = x_train.std(axis=0)

    # max/min standardization
    train_scale = np.nanmax(x_train, axis=0) - np.nanmin(x_train, axis=0)
    train_shift = np.nanmin(x_train, axis=0)

    # Add in a small number 1e-8 to prevent divide by zero errors
    x_train = (x_train - train_shift) / (train_scale + 1e-8)
    x_val = (x_val - train_shift) / (train_scale + 1e-8)

    # Standardize the labels
    if scaler == "minmax":
        shift = y_train.min()
        scale = y_train.max() - y_train.min()
    else:
        shift = y_train.mean()
        scale = y_train.std()

    y_train = (y_train - shift) / scale
    y_val = (y_val - shift) / scale

    return x_train, x_val, y_train, y_val, shift, scale, train_shift, train_scale


def  generate_data(data, idx, ndvi_ls, albedo_ls, window_size, use_coords):
    """This function generates the data based on the given length scale and window size"""

    dist_matrix = np.sqrt(
        np.arange(-window_size, window_size + 1)[np.newaxis, :] ** 2
        + np.arange(-window_size, window_size + 1)[:, np.newaxis] ** 2
    )

    nlcd_w = np.ones([window_size * 2 + 1, window_size * 2 + 1])
    nlcd_w /= nlcd_w.sum()
    ndvi_w = create_weight_matrix(dist_matrix, window_size, ndvi_ls)
    albedo_w = create_weight_matrix(dist_matrix, window_size, albedo_ls)

    X = []
    y = []

    for i in idx:
        lat, lon, nlcd, ndvi, albedo, temp = data[i]

        ndvi_p = ndvi[window_size, window_size]
        ndvi_s = (ndvi * ndvi_w).sum()

        albedo_p = albedo[window_size, window_size]
        albedo_s = (albedo * albedo_w).sum()

        # We aren't calculating a point measurement for this.
        nlcd_s = (nlcd * nlcd_w[:, :, np.newaxis]).sum(axis=(0, 1))
        if use_coords:
            row = np.concatenate(
                [[lat, lon], nlcd_s, [ndvi_p, ndvi_s, albedo_p, albedo_s]]
            )
        else:
            row = np.concatenate([nlcd_s, [ndvi_p, ndvi_s, albedo_p, albedo_s]])
        X.append(row)
        y.append(temp)

    return np.array(X), np.array(y)


def create_weight_matrix(dist_matrix, window_size, length_scale):
    """This function generates the weight matrix based on the given length scale."""
    weight_matrix = np.exp(-dist_matrix / length_scale)
    weight_matrix[window_size, window_size] = 0
    weight_matrix /= weight_matrix.sum()
    return weight_matrix


def create_folds(data, k_folds):
    # Cluster the coordinates
    km = KMeans(n_clusters=k_folds, random_state=42)
    km.fit(data)

    # Return the labels
    return km.labels_


def load_dataset(data_dir, city):
    """This function merely loads the Durham dataset"""
    dataset = Datasets.load(city, data_dir)
    return dataset

def rasterize_temp(shapefile_path):
    # Define the raster grid resolution and extent
    gdf = gpd.read_file(shapefile_path)
    resolution = 30  # Cell size in projection units (e.g., meters)
    xmin, ymin, xmax, ymax = gdf.total_bounds  # Extent of the shapefile
    width = int((xmax - xmin) / resolution)  # Number of columns
    height = int((ymax - ymin) / resolution)  # Number of rows
    transform = from_bounds(xmin, ymin, xmax, ymax, width, height)

    # Prepare geometries and temperature values for rasterization
    shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf['temp_f']))

    # Rasterize the shapefile
    raster = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,  # Fill value for empty cells
        dtype="float32"  # Data type of the raster
    )

    # Save the rasterized data to a GeoTIFF file
    output_tiff_path = "./temp.tif"
    with rio.open(
        output_tiff_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,  # Single band
        dtype="float32",
        crs=gdf.crs,  # Use the CRS from the shapefile
        transform=transform,
    ) as dst:
        dst.write(raster, 1)  # Write the raster data to band 1

    print(f"Raster saved to {output_tiff_path}")

