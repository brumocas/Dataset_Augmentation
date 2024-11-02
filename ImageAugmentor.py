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
        self.changed_images = self.changed_images + 1

    def changed(self):
        return self.changed_images

    def augment_image(self, input_image_path, json_path=None, output_directory=None):
        """
        Augments an image by creating grayscale, Canny, and Sobel versions,
        and saves each augmented image along with a copied JSON file.

        Args:
            input_image_path (str): Path to the input image.
            json_path (str): Path to the corresponding JSON file (if available).
            output_directory (str): Directory to save augmented images and JSON files.
        """
        image = cv2.imread(input_image_path)
        if image is None:
            raise ValueError(f"Error: Could not load image at {input_image_path}")

        image_name = os.path.splitext(os.path.basename(input_image_path))[0]
        
        if output_directory is None:
            output_directory = os.path.dirname(input_image_path)

        os.makedirs(output_directory, exist_ok=True)

        if json_path:
            # Copy JSON file with different names for each augmented image
            shutil.copy(json_path, os.path.join(output_directory, f"{image_name}_gray.json"))
            shutil.copy(json_path, os.path.join(output_directory, f"{image_name}_canny.json"))
            shutil.copy(json_path, os.path.join(output_directory, f"{image_name}_sobel.json"))

        # Save the grayscale image
        gray_image = self.convert_to_grayscale(image)
        cv2.imwrite(os.path.join(output_directory, f"{image_name}_gray.png"), gray_image)

        # Save the Canny edge-detected image
        canny_edges = self.apply_canny(image)
        cv2.imwrite(os.path.join(output_directory, f"{image_name}_canny.png"), canny_edges)

        # Save the Sobel edge-detected image
        sobel_edges = self.apply_sobel(image)
        cv2.imwrite(os.path.join(output_directory, f"{image_name}_sobel.png"), sobel_edges)

        # Increase changed images number
        self.count()


