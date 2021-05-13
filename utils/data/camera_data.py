import numpy as np
import torch
import os
import glob

from utils.dataset_processing import image, grasp
from .grasp_data import GraspDatasetBase

class CameraData(GraspDatasetBase):
    """
    Dataset wrapper for the camera data.
    """
    def __init__(self, file_path, ds_rotate=0,
                 width=640,
                 height=480,
                 output_size=224,
                 include_depth=True,
                 include_rgb=True,
                 **kwargs):
        """
        :param output_size: Image output size in pixels (square)
        :param include_depth: Whether depth image is included
        :param include_rgb: Whether RGB image is included
        """

        super(CameraData, self).__init__(**kwargs)

        self.output_size = output_size
        self.include_depth = include_depth
        self.include_rgb = include_rgb

        self.depth_files = glob.glob(os.path.join(file_path, 'depth_*.npy'))
        self.depth_files.sort()
        self.rgb_files = glob.glob(os.path.join(file_path, 'color_*.png'))
        self.rgb_files.sort()
        self.length = len(self.depth_files)

        if include_depth is False and include_rgb is False:
            raise ValueError('At least one of Depth or RGB must be specified.')

        left = (width - output_size) // 2
        top = (height - output_size) // 2
        right = (width + output_size) // 2
        bottom = (height + output_size) // 2

        self.bottom_right = (bottom, right)
        self.top_left = (top, left)

    @staticmethod
    def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        rect = np.array([[
            [0.0, 10.0],
            [10.0, 10.0],
            [10.0, 0.0],
            [0.0, 0.0]
        ]])

        gtbbs = grasp.GraspRectangles.load_from_array(rect)
        c = self.output_size // 2
        # gtbbs.rotate(rot, (c, c))
        # gtbbs.zoom(zoom, (c, c))
        return gtbbs

    def get_depth(self, idx, rot=0, zoom=1.0, normalise=True):
        arr = np.load(self.depth_files[idx])
        depth_img = image.Image(arr)
        depth_img.crop(bottom_right=self.bottom_right, top_left=self.top_left)
        depth_img.rotate(rot)
        # depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))
        # depth_img.resize((self.output_size, self.output_size))
        # depth_img.img = depth_img.img.transpose((2, 0, 1))
        if normalise:
            depth_img.normalise()
        return np.squeeze(depth_img.img)

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        rgb_img = image.Image.from_file(self.rgb_files[idx])
        rgb_img.crop(bottom_right=self.bottom_right, top_left=self.top_left)
        rgb_img.rotate(rot)
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        
        if normalise:
                rgb_img.normalise()
                rgb_img.img = rgb_img.img.transpose((2, 0, 1))
                
        return rgb_img.img

    def get_data(self, rgb=None, depth=None):
        depth_img = None
        rgb_img = None
        # Load the depth image
        if self.include_depth:
            depth_img = self.get_depth(img=depth)

        # Load the RGB image
        if self.include_rgb:
            rgb_img = self.get_rgb(img=rgb)

        if self.include_depth and self.include_rgb:
            x = self.numpy_to_torch(
                    np.concatenate(
                        (np.expand_dims(depth_img, 0),
                         np.expand_dims(rgb_img, 0)),
                        1
                    )
                )
        elif self.include_depth:
            x = self.numpy_to_torch(depth_img)
        elif self.include_rgb:
            x = self.numpy_to_torch(np.expand_dims(rgb_img, 0))

        return x, depth_img, rgb_img
