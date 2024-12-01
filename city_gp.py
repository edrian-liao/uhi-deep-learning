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

    # Load and preprocess GeoDataFrame
    shapefile_path = f"{os.path.join(args.data_dir, args.city)}/pm_trav.shp"
    gdf = gpd.read_file(shapefile_path)

    # Convert CRS
    gdf = gdf.to_crs(epsg=3857)
    gdf['x'] = gdf.geometry.x
    gdf['y'] = gdf.geometry.y

    # Identify the temperature column
    temp_column = get_temperature_column(gdf)

    # Prepare data
    X = gdf[['x', 'y']].to_numpy()
    y = gdf[temp_column].to_numpy()

    logging.info(f"Loaded {len(X)} data points")

    # Sample random points
    if len(X) > args.num_random_points:
        random_indices = np.random.choice(len(X), size=args.num_random_points, replace=False)
        X = X[random_indices]
        y = y[random_indices]

    # Standardize the data
    x_shift = X.min(axis=0)
    x_scale = X.max(axis=0) - x_shift
    X = (X - x_shift) / (x_scale + 1e-8)

    y_mean = y.mean()
    y_std = y.std()
    y = (y - y_mean) / y_std

    # Convert to PyTorch tensors
    x_train = torch.tensor(X, dtype=torch.float32)
    y_train = torch.tensor(y, dtype=torch.float32)

    # Move data to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train = x_train.to(device)
    y_train = y_train.to(device)

    # Set up the Gaussian Process model
    fixed_noise = torch.full_like(y_train, args.fixed_noise)
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=fixed_noise)
    model = ExactGPModel(x_train, y_train, likelihood)

    likelihood = likelihood.to(device)
    model = model.to(device)

    # Train the model
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    logging.info(f"Starting training")
    min_loss = float("inf")
    learned_ls = 0
    for epoch in range(50):
        optimizer.zero_grad()
        output = model(x_train)
        loss = -mll(output, y_train)
        if loss.item() < min_loss:
            min_loss = loss.item()
            learned_ls = model.covar_module.base_kernel.lengthscale.item()
            logging.info(f"New best loss: {min_loss:.3f}")
        loss.backward()
        optimizer.step()

        lengthscale = model.covar_module.base_kernel.lengthscale.item()
        logging.info(f"Epoch {epoch+1}/50 - Loss: {loss.item():.3f} - Lengthscale: {lengthscale:.3f}")
        print(f"Epoch {epoch+1}/50 - Loss: {loss.item():.3f} - Lengthscale: {lengthscale:.3f}")
    
    logging.info(f"Finished training")

    # Convert the learned lengthscale back to the original scale
    learned_ls = learned_ls * x_scale.mean() # Assume isotropic kernel
    logging.info(f"Learned lengthscale: {learned_ls:.3f}")
    return


def get_temperature_column(gdf):
    # List of potential column names for temperature
    possible_columns = ["temp", "temp_f", "t_f", "T", "T_F"]  # Update this list as needed

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
    parser.add_argument(
        "--num_random_points",
        type=int,
        default=20000,
        help="The number of random points to sample from the dataset"
    )
    parser.add_argument("--output_dir", type=str, help="Path to the output directory")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")

    args = parser.parse_args()
    main(args)