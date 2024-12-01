from .dataset import Dataset
from .spatial_dataset import SpatialDataset

list_cities = ["Boston", "Durham", "San Francisco"]

def load(name, data_dir):
    if name in list_cities:
        return Dataset(data_dir, name)
    else:
        raise ValueError("Dataset {} not supported".format(name))
