import imgaug.augmenters as iaa
import os
import cv2


class ImgAugment:

    def __init__(self):
        pass

    def show_random_img(self):
        pass
    
    def load_images(self, path):
        images = []
        for file in os.listdir(path):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                img = cv2.imread(os.path.join(path, file))
                if img is not None:
                    images.append(img)
        return images

    def augment(self):
        pass

    # Example function to augment images
    def example_augment(self):
        images = self.load_images(img_path)

        # Initialize sequency (pipeline) of all augmentations 
        seq = iaa.Sequential([
            iaa.Affine(rotate=40),
            iaa.GaussianBlur(sigma=(0.0, 3.0)),
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
            iaa.Fliplr(0.5)
        ])

        # seq takes a list of (height, width, channels) images and returns a list of augmented images
        images_aug = seq(images=images)

        for img in images_aug:
            cv2.imshow("Augmented Image", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()