# Imports here
import data_manipulation.augment as augment
import os

# Global variables
non_augmented_images_path = os.path.join(os.path.dirname(__file__), "non-augmented_images")
augmented_images_path = os.path.join(os.path.dirname(__file__), "augmented_images")

def main():
    aug = augment.ImgAugment()

    images = aug.load_images(non_augmented_images_path)
    # aug.example_augment()
    aug.augmentation(images)


if __name__ == "__main__":
    main()