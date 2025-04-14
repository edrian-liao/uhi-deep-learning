import geopandas as gpd
import pandas as pd
import os

def preprocess_shapefile_to_csv(city, output_dir="preprocessed_csv"):
    """
    Preprocess a city's shapefile into a CSV with x, y, and temperature (in Celsius).
    Handles special spatial filtering for Durham, Baltimore, and NYC.
    """
    input_path = f"data/shapefiles/{city}/pm_trav.shp"
    gdf = gpd.read_file(input_path).to_crs(epsg=3857)

    # Extract coordinates
    gdf["x"] = gdf.geometry.x
    gdf["y"] = gdf.geometry.y

    # Identify temperature column
    temp_col = next((col for col in gdf.columns if col.lower() in ["temp_f", "t_f", "t"]), None)
    if temp_col is None:
        raise ValueError(f"No temperature column found in: {input_path}")

    gdf["temperature"] = gdf[temp_col].astype(float)

    # Convert to Celsius unless already in Celsius (column name is 't')
    if temp_col.lower() != "t":
        gdf["temperature"] = (gdf["temperature"] - 32) * 5.0 / 9.0

    max_temp = gdf["temperature"].max()
    min_temp = gdf["temperature"].min()
    print(f"Processing {city} with temperature range: {min_temp} to {max_temp}")

    os.makedirs(output_dir, exist_ok=True)
    ominx, ominy, omaxx, omaxy = gdf.total_bounds

    # ----- Special case: Durham -----
    if city == "Durham":
        durham_bounds = (ominx, 4.291e6, -8.773e6, omaxy)
        raleigh_bounds = (-8.764e6, ominy, omaxx, 4.285e6)

        mask_durham = (
            (gdf["x"] >= durham_bounds[0]) & (gdf["x"] <= durham_bounds[1]) &
            (gdf["y"] >= durham_bounds[2]) & (gdf["y"] <= durham_bounds[3])
        )
        gdf[mask_durham][["x", "y", "temperature"]].to_csv(
            os.path.join(output_dir, "Durham_processed.csv"), index=False)

        mask_raleigh = (
            (gdf["x"] >= raleigh_bounds[0]) & (gdf["x"] <= raleigh_bounds[1]) &
            (gdf["y"] >= raleigh_bounds[2]) & (gdf["y"] <= raleigh_bounds[3])
        )
        gdf[mask_raleigh][["x", "y", "temperature"]].to_csv(
            os.path.join(output_dir, "Raleigh_processed.csv"), index=False)

        print("Saved Durham and Raleigh subsets")

    # ----- Special case: Baltimore -----
    elif city == "Baltimore":
        bounds_a = (-8.54e6, 4.75e6, omaxx, omaxy)
        bounds_b = (ominx, ominy, -8.56e6, 4.722e6)

        mask_a = (
            (gdf["x"] >= bounds_a[0]) & (gdf["x"] <= bounds_a[2]) &
            (gdf["y"] >= bounds_a[1]) & (gdf["y"] <= bounds_a[3])
        )
        gdf[mask_a][["x", "y", "temperature"]].to_csv(
            os.path.join(output_dir, "Baltimore_A_processed.csv"), index=False)

        mask_b = (
            (gdf["x"] >= bounds_b[0]) & (gdf["x"] <= bounds_b[2]) &
            (gdf["y"] >= bounds_b[1]) & (gdf["y"] <= bounds_b[3])
        )
        gdf[mask_b][["x", "y", "temperature"]].to_csv(
            os.path.join(output_dir, "Baltimore_B_processed.csv"), index=False)

        print("Saved Baltimore A and B subsets")

    # ----- Special case: NYC -----
    elif city == "Nyc":
        nyc_bounds = (ominx, 4.976e6, -8.224e6, omaxy)
        mask_nyc = (
            (gdf["x"] >= nyc_bounds[0]) & (gdf["x"] <= nyc_bounds[2]) &
            (gdf["y"] >= nyc_bounds[1]) & (gdf["y"] <= nyc_bounds[3])
        )
        gdf[mask_nyc][["x", "y", "temperature"]].to_csv(
            os.path.join(output_dir, "NYC_processed.csv"), index=False)

        print("Saved NYC clipped subset")

    # ----- Default case -----
    else:
        output_path = os.path.join(output_dir, f"{city}_processed.csv")
        gdf[["x", "y", "temperature"]].to_csv(output_path, index=False)
        print(f"Saved {city} to {output_path}")


# Example usage
cities = [
    "Albuquerque",
    "Atlanta",
    "Baltimore",
    "Boston",
    "Boulder",
    "Chicago",
    "Detroit",
    "Durham",
    "Houston",
    "Kansas",
    "Las Vegas",
    "Los Angeles",
    "Miami",
    "Nashville",
    "New Orleans",
    "NYC",
    "Oklahoma",
    "San Francisco",
    "Seattle",
    "Washington DC"
]
for city in cities:
    preprocess_shapefile_to_csv(city, "./data/csv")