"""CustomDataset class for custom dataloader of infant vision project"""

from torch.utils.data import Dataset
import os
import torch
from pathlib import Path
from skimage.io import imread
from skimage import color
import torchvision as tv
import numpy as np


class CustomDataset(Dataset):
    def __init__(self,
                 *,
                 data_path: str = "",
                 data_type: str = "JPEG",
                 mode: str = "train"):
        super().__init__()
        self.data = self.get_image_paths(data_path=data_path,
                                         data_type=data_type)
        print(f"Found {self.__len__()} images")
        self._infant_age = 0
        self.mode = mode
        self.train_transforms = [tv.transforms.Resize((256, 256))]
        self.val_transforms = [tv.transforms.Resize((256, 256))]
        self._transform = None
        if self.mode == "train":
            self.transform = self.train_transforms
        else:
            self.transform = self.val_transforms

    def get_image_paths(self, *, data_path: str = "", data_type: str = "JPEG"):
        if data_path == "":
            for root, dirs, _ in os.walk("."):
                for dir_name in dirs:
                    if dir_name == "data1":
                        data_path = os.path.join(root, dir_name)
                        break
                if data_path != "":
                    break
        data = []
        for root, _, files in os.walk(data_path):
            for file_name in files:
                if file_name.endswith(
                    data_type.upper()
                    ) or file_name.endswith(
                    data_type.lower()
                ):
                    category = root.split(os.path.sep)[-1]
                    data.append((os.path.join(root, file_name), category))
        return data

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, transforms_list=[]):
        self._transform = tv.transforms.Compose(transforms=transforms_list)

    @property
    def infant_age(self):
        return self._infant_age

    @infant_age.setter
    def infant_age(self, infant_age=0):
        self._infant_age = infant_age

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        filename, category = self.data[index]
        img = imread(Path(filename))
        img_shape = img.shape
        if len(img_shape) == 3 and img_shape[2] == 3:  # if image is colored
            if self.infant_age < 3:
                img = np.apply_along_axis(self.rgb_to_infant_rgb, 2, img)
        else:
            img = color.gray2rgb(img)
        img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
        transformer = self.transform
        img = transformer(img)
        return (img, category)

    def rgb_to_infant_rgb(self, rgb):
        """Convert RGB values (0-255) to equivalent RGB for infant eyes"""
        r, g, b = rgb
        hue, saturation, luminance = CustomDataset.rgb_to_hsl(r, g, b)

        match self.infant_age:
            case 0:
                # Check if hue is in the range for blue (around 180° to 270°)
                if 180 <= hue <= 270:
                    # Decrease saturation while maintaining luminance and hue
                    factor = CustomDataset.saturation_decrease_factor(hue, 180, 270)
                    saturation *= factor
                # Check if hue is in the range for yellow (around 30° to 90°)
                elif 30 <= hue <= 90:
                    # Decrease saturation while maintaining luminance and hue
                    factor = CustomDataset.saturation_decrease_factor(hue, 30, 90)
                    saturation *= factor
                # Check if hue is in the range for green (around 90° to 180°)
                elif 90 <= hue <= 180:
                    # Decrease saturation while maintaining luminance and hue
                    factor = CustomDataset.saturation_decrease_factor(hue, 90, 180)
                    saturation = saturation * factor if saturation < 0.5 else saturation
                r, g, b = CustomDataset.hsl_to_rgb(hue, saturation, luminance)
            case 1:
                # Check if hue is in the range for yellow (around 30° to 90°)
                if 30 <= hue <= 90:
                    # Decrease saturation while maintaining luminance and hue
                    factor = CustomDataset.saturation_decrease_factor(hue, 30, 90)
                    saturation *= factor
                # Check if hue is in the range for blue (around 180° to 270°)
                elif 180 <= hue <= 270:
                    # Decrease saturation while maintaining luminance and hue
                    factor = CustomDataset.saturation_decrease_factor(hue, 180, 270)
                    saturation = saturation * factor if saturation < 0.5 else saturation
                # Check if hue is in the range for green (around 90° to 180°)
                elif 90 <= hue <= 180:
                    # Decrease saturation while maintaining luminance and hue
                    factor = CustomDataset.saturation_decrease_factor(hue, 90, 180)
                    saturation = saturation * factor if saturation < 0.5 else saturation
                r, g, b = CustomDataset.hsl_to_rgb(hue, saturation, luminance)
            case 2:
                # Check if hue is in the range for yellow (around 30° to 90°)
                if 30 <= hue <= 90:
                    # Decrease saturation while maintaining luminance and hue
                    factor = CustomDataset.saturation_decrease_factor(hue, 30, 90)
                    saturation = saturation * factor if saturation < 0.5 else saturation
                # Check if hue is in the range for green (around 90° to 180°)
                elif 90 <= hue <= 180:
                    # Decrease saturation while maintaining luminance and hue
                    factor = CustomDataset.saturation_decrease_factor(hue, 90, 180)
                    saturation = saturation * factor if saturation < 0.5 else saturation
                r, g, b = CustomDataset.hsl_to_rgb(hue, saturation, luminance)
            case 3:
                pass
            case _:
                pass
        rgb = np.array([r, g, b], dtype=rgb.dtype)
        return rgb

    @staticmethod
    def rgb_to_hsl(r, g, b):
        """Convert RGB values (0-255) to HSL"""
        r /= 255.0
        g /= 255.0
        b /= 255.0

        max_val = max(r, g, b)
        min_val = min(r, g, b)
        luminance = (max_val + min_val) / 2

        if max_val == min_val:
            hue = saturation = 0  # achromatic
        else:
            d = max_val - min_val
            saturation = (
                d / (2 - max_val - min_val)
                if luminance > 0.5
                else d / (max_val + min_val)
            )
            if max_val == r:
                hue = (g - b) / d + (6 if g < b else 0)
            elif max_val == g:
                hue = (b - r) / d + 2
            else:
                hue = (r - g) / d + 4
            hue /= 6

        return int(hue * 360), saturation, luminance

    @staticmethod
    def hsl_to_rgb(h, s, l):
        """Convert HSL values to RGB (0-255)"""
        c = (1 - abs(2 * l - 1)) * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = l - c / 2

        if 0 <= h < 60:
            r, g, b = c, x, 0
        elif 60 <= h < 120:
            r, g, b = x, c, 0
        elif 120 <= h < 180:
            r, g, b = 0, c, x
        elif 180 <= h < 240:
            r, g, b = 0, x, c
        elif 240 <= h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x

        r = int((r + m) * 255)
        g = int((g + m) * 255)
        b = int((b + m) * 255)

        return r, g, b

    @staticmethod
    def saturation_decrease_factor(hue, low_limit, high_limit):
        mid = (low_limit + high_limit) / 2
        if hue <= mid:
            return (hue - mid) / (low_limit - mid)
        else:
            return (hue - mid) / (high_limit - mid)
