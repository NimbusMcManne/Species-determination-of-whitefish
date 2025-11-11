# Imports here
import data_manipulation.augment as augment
import os
import cv2

# Global variables
non_augmented_images_path = os.path.join(os.path.dirname(__file__), "non-augmented_images")
augmented_images_path = os.path.join(os.path.dirname(__file__), "augmented_images")

def main():
    aug = augment.ImgAugment()

    images = aug.load_images(non_augmented_images_path)
    # aug.example_augment()
    #aug.augmentation(images)
    #aug.augmentation_for_cropped_fish(non_augmented_images_path, number_of_aug_images_per_img=3, images=images)
    aug.augmentation_for_cropped_fish()
    seq = aug.get_active_augmentation()
    images_aug = seq(images=images)
    aug.save_images(images_aug, augmented_images_path)

if __name__ == "__main__":
    main()