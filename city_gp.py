import argparse
import logging
import os
import geopandas as gpd
import pandas as pd
import numpy as np
import torch
import gpytorch
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from src.models import ExactGPModel


def main(args):
    # Initialize a logger
    logging.basicConfig(
        filename=os.path.join(args.output_dir, f"logs/{args.city}_output.log"),
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

    # # Standardize the data
    x_shift = X.min(axis=0)
    x_scale = X.max(axis=0) - x_shift
    X = (X - x_shift) / (x_scale + 1e-8)

    y_mean = y.mean()
    y_std = y.std()
    y = (y - y_mean) / y_std

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Initialize GroupKFold
    kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)
    fold_r2_scores = []
    learned_length_scales = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_tensor)):
        # Split data into training and validation sets
        x_train, x_val = X_tensor[train_idx], X_tensor[val_idx]
        y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]

        # Set up the Gaussian Process model
        fixed_noise = torch.full_like(y_train, args.fixed_noise)
        # likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=fixed_noise)
        model = ExactGPModel(x_train, y_train, likelihood)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        likelihood = likelihood.to(device)
        x_train, y_train = x_train.to(device), y_train.to(device)
        x_val, y_val = x_val.to(device), y_val.to(device)

        # Train the model
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        min_loss = float("inf")
        learned_length_scale = 0
        best_model_path = os.path.join(args.output_dir, f"models/{args.city}/best_model_fold_{fold + 1}.pth")
        for epoch in range(70):
            optimizer.zero_grad()
            output = model(x_train)
            loss = -mll(output, y_train)

            # Save the model if the current loss is the lowest
            if loss.item() < min_loss:
                min_loss = loss.item()
                torch.save(model.state_dict(), best_model_path)
                learned_length_scale = model.covar_module.base_kernel.lengthscale.item()
                logging.info(f"New best model saved for fold {fold + 1}")

            loss.backward()
            optimizer.step()

            logging.info(f"Fold {fold + 1}, Epoch {epoch + 1}/70 - Loss: {loss.item():.3f}")

        learned_length_scales.append(learned_length_scale)
        logging.info(f"Finished training for fold {fold + 1} with best loss: {min_loss:.3f} and learned length scale: {learned_length_scale:.3f}")

        # Load the best model for validation
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        likelihood.eval()

        # Evaluate on validation set
        with torch.no_grad():
            val_output = model(x_val)
            y_val_pred = val_output.mean.cpu().numpy()
            y_val_true = y_val.cpu().numpy()

            # Calculate R^2 metric
            r2 = r2_score(y_val_true, y_val_pred)
            fold_r2_scores.append(r2)

        logging.info(f"Validation R^2 for fold {fold + 1}: {r2:.3f}")

    # Compute average \( R^2 \) across all folds
    avg_r2 = np.mean(fold_r2_scores)
    avg_length_scale = np.mean(learned_length_scales)
    avg_length_scale_scaled = avg_length_scale * x_scale
    logging.info(f"Average validation R^2 across all folds: {avg_r2:.3f}")
    logging.info(f"Average learned length scale across latitude: {avg_length_scale_scaled[0]:.3f}")
    logging.info(f"Average learned length scale across longitude: {avg_length_scale_scaled[1]:.3f}")

    # Update the CSV file
    results_csv = os.path.join(args.output_dir, "results.csv")
    results = {
        "city": args.city,
        "avg_r2": avg_r2,
        "avg_length_scale_lat": avg_length_scale_scaled[0],
        "avg_length_scale_lon": avg_length_scale_scaled[1],
    }

    # Check if the file exists
    if os.path.exists(results_csv):
        results_df = pd.read_csv(results_csv)
        if args.city in results_df["city"].values:
            results_df.loc[results_df["city"] == args.city, ["avg_r2", "avg_length_scale_lat", "avg_length_scale_lon"]] = [
                avg_r2,
                avg_length_scale_scaled[0],
                avg_length_scale_scaled[1],
            ]
        else:
            results_df = pd.concat([results_df, pd.DataFrame([results])], ignore_index=True)
    else:
        results_df = pd.DataFrame([results])

    # Save the updated DataFrame
    results_df.to_csv(results_csv, index=False)

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
        default="data/shapefiles",
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
    parser.add_argument(
        "--k_folds",
        type=int,
        default=5,
        help="The number of folds to use for cross-validation"
    )
    parser.add_argument("--output_dir", type=str, help="Path to the output directory")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")

    args = parser.parse_args()
    main(args)
