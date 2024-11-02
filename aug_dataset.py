import os
import argparse
from ImageAugmentor import ImageAugmentor

def augment_dataset(dataset_path):
    """
    Augments images in a dataset with subfolders, saving augmented images 
    and corresponding JSON files in new folders suffixed with 'aug'.
    
    Args:
        dataset_path (str): Path to the dataset directory containing subfolders with images and JSON files.
    """
    augmentor = ImageAugmentor()

    # Go through each subfolder in the dataset path
    for subfolder_name in os.listdir(dataset_path):
        subfolder_path = os.path.join(dataset_path, subfolder_name)
        
        # Check if it is a directory
        if not os.path.isdir(subfolder_path):
            continue
        
        # Create a new folder with "aug" suffix
        aug_folder_path = os.path.join(dataset_path, f"{subfolder_name}aug")
        os.makedirs(aug_folder_path, exist_ok=True)
        
        # Process each image and corresponding JSON file in the subfolder
        for file_name in os.listdir(subfolder_path):
            if file_name.endswith('.png'):
                input_image_path = os.path.join(subfolder_path, file_name)
                
                # Find the corresponding JSON file
                json_path = os.path.join(subfolder_path, f"{os.path.splitext(file_name)[0]}.json")
                if not os.path.exists(json_path):
                    json_path = None  # No JSON file for this image
                
                # Augment the image and save in the new "aug" folder
                augmentor.augment_image(input_image_path, json_path=json_path, output_directory=aug_folder_path)
    
    print(f"Changed {augmentor.changed_images} images")
    print(f"Dataset augmentation complete. Augmented folders saved with 'aug' suffix in {dataset_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment images in a dataset folder.")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset folder containing subfolders with images and JSON files.")
    args = parser.parse_args()

    augment_dataset(args.dataset_path)
