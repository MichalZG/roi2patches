import numpy as np
import cv2
import os

class Patch:
    """
    Patch class
    Attributes:
        type (int): flag for patch type
        data (array): 2D or 3D data
    """
    def __init__(self, rel_mask, size, xy, save_dir, fmt):
        """
        Args:
            rel_mask (object): related mask
            size (two element list): x, y -> [28, 28]
            xy (two element list): x, y center of patch
            save_dir (path): root dir to save (without type)
            fmt (string): output extension eg. .jgp, tif etc.
        """
        self.rel_mask = rel_mask
        self.size = size
        self.xy = xy
        self.save_dir = save_dir
        self.fmt = fmt
        self.type = None
        self.data = None
        self.ds = {1: 'background',
                   2: 'cytoplasm',
                   3: 'nuclei',
                   4: 'border'}

    def set_type(self, flag):
        """
        set patch type from self.ds
        """
        self.type = self.ds[flag]


    def create_data(self): 
        """
        create data based on mask and image related to mask
        """
        self.data = self.rel_mask.rel_image.data[
            self.xy[0]-self.size[0]/2.:self.xy[0]+self.size[0]/2.,
            self.xy[1]-self.size[1]/2.:self.xy[1]+self.size[1]/2.]


    def save(self):
        """
        Save patch
        """
        self.patch_name = "_".join(
            (self.rel_mask.rel_image.name,
             str(self.xy[0]), str(self.xy[1]), self.type)) + self.fmt
        cv2.imwrite(os.path.join(
            self.save_dir, self.type, self.patch_name), self.data)
