# Imports here
import data_manipulation.augment as augment
import os
import cv2
import argparse

# Global variables
NON_AUGMENTED_IMAGES_PATH = os.path.join(os.path.dirname(__file__), "non-augmented_images")
AUGMENTED_IMAGES_PATH = os.path.join(os.path.dirname(__file__), "augmented_images")


def main():
    parser = argparse.ArgumentParser()

    aug = augment.ImgAugment()

    images = aug.load_images(NON_AUGMENTED_IMAGES_PATH)
    # aug.example_augment()
    #aug.augmentation(images)
    #aug.augmentation_for_cropped_fish(non_augmented_images_path, number_of_aug_images_per_img=3, images=images)

    parser.add_argument(
        "--n_images",
        type=int,
        default=500,
        help="How many images do you want to generate?"
    )
    
    args = parser.parse_args()

    aug.oversample(args.n_images, images, aug, AUGMENTED_IMAGES_PATH)
    # aug.augmentation_for_cropped_fish()
    # seq = aug.get_active_augmentation()
    # images_aug = seq(images=images)
    
    # aug.save_images(images_aug, augmented_images_path)

if __name__ == "__main__":
    main()