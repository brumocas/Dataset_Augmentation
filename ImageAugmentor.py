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

    def convert_to_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def apply_canny(self, image):
        return cv2.Canny(image, self.canny_threshold1, self.canny_threshold2)

    def apply_sobel(self, image):
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=self.sobel_ksize)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=self.sobel_ksize)
        sobel_edges = cv2.magnitude(sobel_x, sobel_y)
        return np.uint8(sobel_edges)
    
    def count(self):
        self.changed_images += 1

    def changed(self):
        return self.changed_images

    def augment_image(self, input_image_path, json_path=None, output_directory=None, output_format="png", augmentations=None):
        """
        Augments an image by creating specified versions (grayscale, Canny, Sobel),
        and saves each augmented image along with a copied JSON file if available.

        Args:
            input_image_path (str): Path to the input image.
            json_path (str): Path to the corresponding JSON file (if available).
            output_directory (str): Directory to save augmented images and JSON files.
            output_format (str): Desired image format for output files, e.g., 'png' or 'jpg'.
            augmentations (list): List of augmentations to apply; options are "grayscale", "canny", "sobel".
        
        Returns:
            dict: Dictionary with keys for each applied augmentation containing paths to saved images.
        """
        image = cv2.imread(input_image_path)
        if image is None:
            raise ValueError(f"Error: Could not load image at {input_image_path}")

        image_name = os.path.splitext(os.path.basename(input_image_path))[0]

        # Use input image's directory if no output directory specified
        if output_directory is None:
            output_directory = os.path.dirname(input_image_path)
        os.makedirs(output_directory, exist_ok=True)

        # Set default to all augmentations if none specified
        if augmentations is None:
            augmentations = ["grayscale", "canny", "sobel"]

        output_paths = {}

        # Copy JSON file with new names for each augmented image if JSON path is given
        if json_path and os.path.exists(json_path):
            for aug in augmentations:
                shutil.copy(json_path, os.path.join(output_directory, f"{image_name}_{aug}.json"))

        # Apply selected augmentations and save images
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
