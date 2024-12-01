import os
import numpy as np
import rasterio as rio


class Dataset():
    
    def __init__(self, dataset_path, city):

        with rio.open(os.path.join(dataset_path, f'temp.tif')) as src:
            self.temp = src.read(1)
        
        self.coords = self.get_coords()
    
    def get_coords(self):
        coords = np.array(np.where(self.temp != 0)).T
        return coords
    
    def __getitem__(self, idx):
        i, j = self.coords[idx]
        return self.temp[i, j]
        