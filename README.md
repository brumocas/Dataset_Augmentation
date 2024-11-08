# Image Augmentation with `ImageAugmentor`

## Overview
The `ImageAugmentor` class provides a simple way to apply multiple image augmentations, including grayscale conversion, Canny edge detection, and Sobel edge detection. This class can be used as part of a dataset augmentation pipeline to generate variations of images, which can improve the robustness of machine learning models by introducing different transformations.

The `augment_dataset` function automates the augmentation of all images in a specified dataset folder, creating augmented copies of each image (and corresponding JSON files, if available) in subdirectories.

## Code Explanation

### `ImageAugmentor` Class

```python
import cv2
import numpy as np
import os
import shutil

class ImageAugmentor:
    def __init__(self, canny_threshold1=100, canny_threshold2=250, sobel_ksize=3):
        self.canny_threshold1 = canny_threshold1
        self.canny_threshold2 = canny_threshold2
        self.sobel_ksize = sobel_ksize
        self.changed_images = 0
```

- `canny_threshold1` and `canny_threshold2` are parameters for the Canny edge detection threshold.
- `sobel_ksize` is the kernel size for Sobel edge detection.
- `changed_images` keeps track of how many images were augmented.

#### Grayscale Conversion
```python
    def convert_to_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```
- This function converts an input image to grayscale using OpenCV.

#### Canny Edge Detection
```python
    def apply_canny(self, image):
        return cv2.Canny(image, self.canny_threshold1, self.canny_threshold2)
```
- This function applies the Canny edge detection algorithm with the given thresholds.

#### Sobel Edge Detection
```python
    def apply_sobel(self, image):
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=self.sobel_ksize)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=self.sobel_ksize)
        sobel_edges = cv2.magnitude(sobel_x, sobel_y)
        return np.uint8(sobel_edges)
```
- This function computes the Sobel edges in both x and y directions and combines them to get the magnitude.

#### Image Augmentation with JSON Support
```python
    def augment_image(self, input_image_path, json_path=None, output_directory=None, output_format="png", augmentations=None):
        image = cv2.imread(input_image_path)
        if image is None:
            raise ValueError(f"Error: Could not load image at {input_image_path}")

        image_name = os.path.splitext(os.path.basename(input_image_path))[0]
        
        if output_directory is None:
            output_directory = os.path.dirname(input_image_path)
        os.makedirs(output_directory, exist_ok=True)

        if augmentations is None:
            augmentations = ["grayscale", "canny", "sobel"]

        output_paths = {}

        if json_path and os.path.exists(json_path):
            for aug in augmentations:
                shutil.copy(json_path, os.path.join(output_directory, f"{image_name}_{aug}.json"))

        if "grayscale" in augmentations:
            gray_image = self.convert_to_grayscale(image)
            gray_image_path = os.path.join(output_directory, f"{image_name}_gray.{output_format}")
            cv2.imwrite(gray_image_path, gray_image)
            output_paths["grayscale"] = gray_image_path
            self.count()

        if "canny" in augmentations:
            canny_edges = self.apply_canny(image)
            canny_image_path = os.path.join(output_directory, f"{image_name}_canny.{output_format}")
            cv2.imwrite(canny_image_path, canny_edges)
            output_paths["canny"] = canny_image_path
            self.count()

        if "sobel" in augmentations:
            sobel_edges = self.apply_sobel(image)
            sobel_image_path = os.path.join(output_directory, f"{image_name}_sobel.{output_format}")
            cv2.imwrite(sobel_image_path, sobel_edges)
            output_paths["sobel"] = sobel_image_path
            self.count()

        return output_paths
```

- **Parameters**:
  - `input_image_path`: Path to the input image.
  - `json_path`: Optional path to a corresponding JSON file to copy along with augmented images.
  - `output_directory`: Directory where augmented images and JSON files are saved.
  - `output_format`: Format for saved images (e.g., `png` or `jpg`).
  - `augmentations`: List specifying which augmentations to apply (`grayscale`, `canny`, and/or `sobel`).

- **Functionality**:
  - Copies the JSON file (if provided) alongside each augmented image.
  - Saves augmented images in the specified `output_directory` with appropriate suffixes.

### `augment_dataset` Function

This function is used to augment a full dataset.

```python
import os
import argparse
from ImageAugmentor import ImageAugmentor

def augment_dataset(dataset_path, augmentations):
    augmentor = ImageAugmentor()

    for subfolder_name in os.listdir(dataset_path):
        subfolder_path = os.path.join(dataset_path, subfolder_name)

        if not os.path.isdir(subfolder_path):
            continue

        aug_folder_path = os.path.join(dataset_path, f"{subfolder_name}_aug")
        os.makedirs(aug_folder_path, exist_ok=True)

        print(f"Processing folder: {subfolder_name}...")

        for file_name in os.listdir(subfolder_path):
            if file_name.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                input_image_path = os.path.join(subfolder_path, file_name)
                json_path = os.path.join(subfolder_path, f"{os.path.splitext(file_name)[0]}.json")
                if not os.path.exists(json_path):
                    json_path = None

                print(f"Augmenting image: {file_name}")
                augmentor.augment_image(input_image_path, json_path=json_path, output_directory=aug_folder_path, augmentations=augmentations)

    print(f"Changed {augmentor.changed_images} images")
    print(f"Dataset augmentation complete. Augmented folders saved with 'aug' suffix in {dataset_path}.")
```

- **Parameters**:
  - `dataset_path`: Directory containing subfolders of images.
  - `augmentations`: List of augmentation types to apply.

- **Functionality**:
  - Iterates through each subfolder, applies the specified augmentations, and saves the results in a new subfolder with an `_aug` suffix.
  - Logs the progress, showing each folder and image being processed.

## Running the Code

### Requirements
- **Libraries**: OpenCV (`cv2`), NumPy (`numpy`), argparse
- **Installation**: Install the required libraries using `pip`
  ```bash
  pip install opencv-python numpy
  ```

### Command-Line Usage

To run the script, use the following command:

```bash
python augment_dataset.py <dataset_path> --augmentations grayscale canny
```

- Replace `<dataset_path>` with the path to your dataset folder.
- Use `--augmentations` to specify the augmentations to apply (e.g., `grayscale`, `canny`, `sobel`). If no augmentation types are specified, all three augmentations will be applied by default.

### Example

```bash
python augment_dataset.py /path/to/dataset --augmentations grayscale sobel
```

This command applies grayscale and Sobel augmentations to all images in the specified dataset folder, saving augmented images in new folders with an `_aug` suffix.

### Output

The augmented images and corresponding JSON files (if available) will be saved in subdirectories with `_aug` suffixes, like `train_aug`, `test_aug`, etc.