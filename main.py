# Imports here
import data_manipulation.augment as augment
import os

# Global variables
non_augmented_images_path = os.path.join(os.path.dirname(__file__), "non-augmented_images")
augmented_images_path = os.path.join(os.path.dirname(__file__), "augmented_images_test")

def main():
    aug = augment.ImgAugment()

    images = aug.load_images(non_augmented_images_path)
    # aug.example_augment()
    #aug.augmentation(images)
    aug.augment(augmented_images_path, number_of_aug_images_per_img=3, images=images)


if __name__ == "__main__":
    main()