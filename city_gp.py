import argparse
import logging
import os

import geopandas as gpd
import numpy as np
import torch
import gpytorch

from src.models import ExactGPModel



def main(args):
    # Initialize a logger
    logging.basicConfig(
        filename=os.path.join(args.output_dir, f"{args.city}_output.log"),
        filemode="w",
        level=args.log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info(args)

    # Load the dataset
    # if os.path.exists(args.data_dir):
    #     dataset = load_dataset(args.data_dir, args.window_size, args.city)
    # else:
    #     os.makedirs(args.data_dir)

    shapefile_path = f"{os.path.join(args.data_dir,args.city)}/pm_trav.shp"
    gdf = gpd.read_file(shapefile_path)

    # Convert from EPSG: 4326 to EPSG: 3857
    gdf = gdf.to_crs(epsg=3857)

    gdf['x'] = gdf.geometry.x
    gdf['y'] = gdf.geometry.y

    coordinates = gdf[['x', 'y']].to_numpy()
    
    try:
        temp_column = get_temperature_column(gdf)
        logging.info(f"Temperature column found: {temp_column}")
        X, y = coordinates, gdf[temp_column].to_numpy()

    except ValueError as e:
        print(e)
    
    # Now, we need to standardize the data for the final fit
    x_shift = X.min(axis=0)
    x_scale = X.max(axis=0) - X.min(axis=0)

    # Add in a small number 1e-8 to prevent divide by zero errors
    x_train = (X - x_shift) / (x_scale + 1e-16)

    # Standardize the labels
    y_mean = y.mean()
    y_std = y.std()

    y_train = (y - y_mean) / (y_std)

    # TRAINING using GPyTorch
    # Convert the arrays to tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    logging.info(f"Training data shape: {x_train.shape}, {y_train.shape}")
    
    fixed_noise = torch.full_like(y_train, args.fixed_noise)
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=fixed_noise)
    model = ExactGPModel(x_train, y_train, likelihood)

    # Send to GPU
    x_train = x_train.cuda()
    y_train = y_train.cuda()
    model = model.cuda()
    likelihood = likelihood.cuda()

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Fit the model
    logging.info(f"Starting training")
    for i in range(50):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(x_train)
        # Calc loss and backprop gradients
        loss = -mll(output, y_train)
        loss.backward()
        logging.info(f'Iter {i+1}/50 - Loss: {loss.item(): .3f}   {model.covar_module.base_kernel.lengthscale.item(): .3f}')
        optimizer.step()
    logging.info(f"Finished training")

    #TODO: maybe distinguish between morning and evening later

    # Generate visualizations
    return

def get_temperature_column(gdf):
    # List of potential column names for temperature
    possible_columns = ["temp", "temp_f", "t_f"] # Update this list as needed

    # Check if any of these columns exist in the GeoDataFrame
    for col in possible_columns:
        if col in gdf.columns:
            return col

    # Raise an error if none are found
    raise ValueError("No temperature column found in the GeoDataFrame.")

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Train an Exact Gaussian process on a city-level dataset")
    parser.add_argument(
        "data_dir", 
        type=str, 
        default="data",
        help="The directory where the data is stored"
        )
    
    parser.add_argument(
        "city", 
        type=str, 
        help="The city where data is being collected from"
        )
    
    parser.add_argument(
         "fixed_noise",
         type=float,
         help="The fixed assumption of noise from the observations during training"
        )
    
    # Create the output arguments
    parser.add_argument("--output_dir", type=str, help="Path to the output directory")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")

    args = parser.parse_args()
    print(args)
    main(args)