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
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath("../src"))

# Now import the models
from src.models import NonStationaryGPModel, ExactGPModel

# --- Argparse Setup ---
parser = argparse.ArgumentParser(description="Train GPs and generate temperature predictions.")
parser.add_argument("--city", type=str, required=True, help="City name corresponding to shapefile directory")
parser.add_argument("--resolution", type=int, default=500, help="Grid resolution in meters")
parser.add_argument("--num_points", type=int, default=9, help="Number of RBF centers (non-stationary)")

args = parser.parse_args()

city = args.city
resolution = args.resolution
num_points = args.num_points

# --- Logging Setup ---
# Define log file name
log_filename = os.path.join("logs", f"{city}_generate_gp.log")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename, mode='w'),
        logging.StreamHandler()
    ]
)

logging.info(f"Logging initialized for city: {city}")

def get_temperature_column(gdf):
    # List of potential column names for temperature
    possible_columns = ["temp", "temp_f", "t_f", "T", "T_F"]  # Update this list as needed

    # Check if any of these columns exist in the GeoDataFrame
    for col in possible_columns:
        if col in gdf.columns:
            return col

    # Raise an error if none are found
    raise ValueError("No temperature column found in the GeoDataFrame.")

def train_model(X, y, model_str='ExactGP', num_points=16, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_scores = []

    best_val_score = -float('inf')  # Initialize for max R²
    best_model_weights = None
    best_likelihood_weights = None
    best_fold_model = None
    best_fold_likelihood = None

    for fold, (train_index, val_index) in enumerate(kf.split(X)):
        logging.info(f"\nFold {fold+1}/{n_splits}")

        # Move data to device
        train_x = torch.tensor(X[train_index], dtype=torch.float32).to(device)
        train_y = torch.tensor(y[train_index], dtype=torch.float32).to(device)
        val_x = torch.tensor(X[val_index], dtype=torch.float32).to(device)
        val_y = torch.tensor(y[val_index], dtype=torch.float32).to(device)

        # Initialize likelihood & model
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        # likelihood = gpytorch.likelihoods.StudentTLikelihood()
        if model_str == 'ExactGP':
            model = ExactGPModel(train_x, train_y, likelihood).to(device)
            EPOCHS = 40
        elif model_str == 'NonStationaryGP':
            model = NonStationaryGPModel(train_x, train_y, likelihood, num_points=num_points).to(device)
            EPOCHS = 30

        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        best_train_loss = float('inf')
        best_model_state = None

        for i in range(EPOCHS):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

            logging.info(f"Epoch {i+1}/{EPOCHS} - Loss: {loss.item():.4f}")
            if loss.item() < best_train_loss:
                best_train_loss = loss.item()
                best_model_state = model.state_dict()

        # Evaluate on validation set
        model.load_state_dict(best_model_state)
        model.eval()
        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            val_pred = likelihood(model(val_x)).mean
            val_r2 = r2_score(val_y.cpu().numpy(), val_pred.cpu().numpy())
            logging.info(f"Validation R² (Fold {fold+1}): {val_r2:.4f}")
            fold_scores.append(val_r2)

            # Track the best-performing model
            if val_r2 > best_val_score:
                best_val_score = val_r2
                best_model_weights = best_model_state
                best_likelihood_weights = likelihood.state_dict()
                best_fold_model = model.__class__  # Save class to reinitialize later
                best_fold_likelihood = likelihood.__class__

    logging.info(f"\nAverage R² across folds: {np.mean(fold_scores):.4f}")
    logging.info(f"Best R² across folds: {best_val_score:.4f}")

    # Re-initialize and load best model and likelihood
    final_likelihood = best_fold_likelihood().to(device)
    if model_str == 'ExactGP':
        final_model = best_fold_model(torch.tensor(X, dtype=torch.float32).to(device),
                                  torch.tensor(y, dtype=torch.float32).to(device),
                                  final_likelihood).to(device)
    elif model_str == 'NonStationaryGP':
        final_model = best_fold_model(torch.tensor(X, dtype=torch.float32).to(device),
                                  torch.tensor(y, dtype=torch.float32).to(device),
                                  final_likelihood, num_points=num_points).to(device)
    final_model.load_state_dict(best_model_weights)
    final_likelihood.load_state_dict(best_likelihood_weights)

    return final_model

def generate_predictions(model, likelihood, points, num_samples=4, y_mean=0.0, y_std=1.0):
    model = model.to("cpu")
    likelihood = likelihood.to("cpu")
    points = points.to("cpu")

    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictive_dist = likelihood(model(points))
        mean_pred = predictive_dist.mean.cpu().numpy() * y_std + y_mean
        sample_preds = predictive_dist.rsample(torch.Size([num_samples])).cpu().numpy() * y_std + y_mean

    return mean_pred, sample_preds

# Read preprocessed CSV
csv_path = f"data/csv/{city}_processed.csv"
df = pd.read_csv(csv_path)

# Ensure correct format
required_cols = {"x", "y", "temperature"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"CSV file must contain columns: {required_cols}")

# Get coordinates and temperature
X = df[['x', 'y']].to_numpy()
y = df['temperature'].astype(float).to_numpy()

# Compute bounding box from the coordinates
xmin, ymin = X.min(axis=0)
xmax, ymax = X.max(axis=0)
bounds = (xmin, ymin, xmax, ymax)

# Now, we need to standardize the data for the final fit
x_shift = X.min(axis=0)
x_scale = X.max(axis=0) - X.min(axis=0)

# Add in a small number 1e-8 to prevent divide by zero errors
x_norm = (X - x_shift) / (x_scale + 1e-16)

# Standardize the labels
y_mean = y.mean()
y_std = y.std()

y_norm = (y - y_mean) / (y_std)

random_indices = np.random.choice(len(X), size=20000, replace=False)
x_tensor, y_tensor = torch.tensor(x_norm[random_indices]), torch.tensor(y_norm[random_indices])

# Set device: use CUDA if available, fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Train the GP model
exact_gp_model = train_model(x_tensor, y_tensor, 'ExactGP', n_splits=4)
logging.info('\n')
ns_gp_model = train_model(x_tensor, y_tensor,'NonStationaryGP', num_points=num_points, n_splits=4)

# Adjust bounding box based on city-specific overrides
if city == "Durham":
    xmin, ymin, xmax, ymax = bounds[0], 4.291e6, -8.773e6, bounds[3]
elif city == "Raleigh":
    xmin, ymin, xmax, ymax = -8.764e6, bounds[1], bounds[2], 4.285e6
elif city == "Baltimore_A":
    xmin, ymin, xmax, ymax = -8.54e6, 4.75e6, bounds[2], bounds[3]
elif city == "Baltimore_B":
    xmin, ymin, xmax, ymax = bounds[0], bounds[1], -8.56e6, 4.722e6
elif city == "NYC":
    xmin, ymin, xmax, ymax = bounds[0], 4.976e6, -8.224e6, bounds[3]
else:
    xmin, ymin, xmax, ymax = bounds


# Generate coordinate grid
x_coords = np.arange(xmin, xmax + resolution, resolution)
x_coords_norm = (x_coords - x_shift[0]) / (x_scale[0] + 1e-16)
y_coords = np.arange(ymin, ymax + resolution, resolution)
y_coords_norm = (y_coords - x_shift[1]) / (x_scale[1] + 1e-16)
X, Y = np.meshgrid(x_coords_norm, y_coords_norm)

# Combine into (N, 2) array of 2D points
points = np.vstack([X.ravel(), Y.ravel()]).T

logging.info(f"Generated {points.shape[0]} points with {int(resolution)}m resolution")

points = torch.tensor(points, dtype=torch.float32)
likelihood = gpytorch.likelihoods.GaussianLikelihood()

# Generate predictions for sampled points
mean_pred_ex, sample_preds_ex = generate_predictions(exact_gp_model, likelihood, points, num_samples=3, y_std=y_std, y_mean=y_mean)
mean_pred_ns, sample_preds_ns = generate_predictions(ns_gp_model, likelihood, points, num_samples=3, y_std=y_std, y_mean=y_mean)

# Necessary transformations to plot the results
ex_gp_sample0 = np.flipud(mean_pred_ex.reshape(len(y_coords), len(x_coords)))
ex_gp_samples = []
for i in range(sample_preds_ex.shape[0]):
    ex_gp_samples.append(np.flipud(sample_preds_ex[i].reshape(len(y_coords), len(x_coords))))
ns_gp_sample0 = np.flipud(mean_pred_ns.reshape(len(y_coords), len(x_coords)))
ns_gp_samples = []
for i in range(sample_preds_ns.shape[0]):
    ns_gp_samples.append(np.flipud(sample_preds_ns[i].reshape(len(y_coords), len(x_coords))))

# Save to numpy files
np.savez(f"results/gp/{city}_gp_predictions.npz",
         ex_gp_sample0=ex_gp_sample0,
         ex_gp_samples=np.array(ex_gp_samples),
         ns_gp_sample0=ns_gp_sample0,
         ns_gp_samples=np.array(ns_gp_samples),
         x_coords=x_coords,
         y_coords=y_coords)

# Plot the results
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(f'Samples from Stationary GP in {city}', fontsize=18)

# Plot the mean prediction
im0 = ax[0, 0].imshow(ex_gp_sample0, cmap="coolwarm", interpolation='bicubic')
ax[0, 0].set_title('Mean')
ax[0, 0].axis('off')
colorbar = fig.colorbar(im0, ax=ax[0, 0])
colorbar.set_label('Temperature (C)', rotation=270, labelpad=15)

# Plot sample 1
im1 = ax[0, 1].imshow(ex_gp_samples[0], cmap="coolwarm", interpolation='bicubic')
ax[0, 1].set_title('Sample 1')
ax[0, 1].axis('off')
colorbar = fig.colorbar(im1, ax=ax[0, 1])
colorbar.set_label('Temperature (C)', rotation=270, labelpad=15)

# Plot sample 2
im2 = ax[1, 0].imshow(ex_gp_samples[1], cmap="coolwarm", interpolation='bicubic')
ax[1, 0].set_title('Sample 2')
ax[1, 0].axis('off')
colorbar = fig.colorbar(im2, ax=ax[1, 0])
colorbar.set_label('Temperature (C)', rotation=270, labelpad=15)

# Plot sample 3
im3 = ax[1, 1].imshow(ex_gp_samples[2], cmap="coolwarm", interpolation='bicubic')
ax[1, 1].set_title('Sample 3')
ax[1, 1].axis('off')
colorbar = fig.colorbar(im3, ax=ax[1, 1])
colorbar.set_label('Temperature (C)', rotation=270, labelpad=15)

plt.tight_layout()
plt.savefig(f"data/figures/gp/{city}_exact_gp.png", dpi=300)

# Plot the results
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(f'Samples from Non-stationary GP in {city}', fontsize=18)

# Plot the mean prediction
im0 = ax[0, 0].imshow(ns_gp_sample0, cmap="coolwarm", interpolation='bicubic')
ax[0, 0].set_title('Mean')
ax[0, 0].axis('off')
colorbar = fig.colorbar(im0, ax=ax[0, 0])
colorbar.set_label('Temperature (C)', rotation=270, labelpad=15)

# Plot sample 1
im1 = ax[0, 1].imshow(ns_gp_samples[0], cmap="coolwarm", interpolation='bicubic')
ax[0, 1].set_title('Sample 1')
ax[0, 1].axis('off')
colorbar = fig.colorbar(im1, ax=ax[0, 1])
colorbar.set_label('Temperature (C)', rotation=270, labelpad=15)

# Plot sample 2
im2 = ax[1, 0].imshow(ns_gp_samples[1], cmap="coolwarm", interpolation='bicubic')
ax[1, 0].set_title('Sample 2')
ax[1, 0].axis('off')
colorbar = fig.colorbar(im2, ax=ax[1, 0])
colorbar.set_label('Temperature (C)', rotation=270, labelpad=15)

# Plot sample 3
im3 = ax[1, 1].imshow(ns_gp_samples[2], cmap="coolwarm", interpolation='bicubic')
ax[1, 1].set_title('Sample 3')
ax[1, 1].axis('off')
colorbar = fig.colorbar(im3, ax=ax[1, 1])
colorbar.set_label('Temperature (C)', rotation=270, labelpad=15)

plt.tight_layout()
plt.savefig(f"data/figures/gp/{city}_ns_gp.png", dpi=300)
