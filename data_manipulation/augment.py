import imgaug.augmenters as iaa
import os
import cv2


class ImgAugment:

    def __init__(self):
        self.filenames = []
        self.active_augmentation = None
        self.image_paths = None

    def show_filenames(self):
        return self.filenames

    def show_random_img(self):
        pass

    def get_active_augmentation(self):
        return self.active_augmentation

    def save_images(self, images, path):
        for i, img in enumerate(images):
            print(self.filenames[i])
            cv2.imwrite(os.path.join(path, f"{self.filenames[i]}.jpg"), img)

    def load_images(self, path):
        images = []
        for file in os.listdir(path):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                self.filenames.append(file.split('.')[0])
                img = cv2.imread(os.path.join(path, file))
                if img is not None:
                    images.append(img)

        return images
    
    def load_image_paths(self, path):
        self.image_paths = []
        for file in os.listdir(path):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                self.image_paths.append(os.path.join(path, file))


    def augmentation(self, images):
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        seq = iaa.Sequential(
            [
                iaa.Fliplr(0.3),
                iaa.Flipud(0.1),
                sometimes(iaa.Add((-50, 50))),
                sometimes(iaa.Invert(0.1)),
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)),
                    iaa.AverageBlur(k=(2, 7)), 
                    iaa.MedianBlur(k=(3, 11))
                ]),
                iaa.OneOf([
                    iaa.AdditiveGaussianNoise(scale=(0, 0.2*255)),
                    iaa.AdditiveLaplaceNoise(scale=(0, 0.2*255)),
                    iaa.AdditivePoissonNoise(30)
                ]),
                iaa.SomeOf((0, 3),
                    [
                        iaa.BlendAlpha((0.0, 1.0), iaa.Grayscale(1.0)),
                        iaa.BlendAlphaMask(
                            iaa.InvertMaskGen(0.5, iaa.VerticalLinearGradientMaskGen()),
                            iaa.Clouds()
                        ),
                        iaa.WithBrightnessChannels(iaa.Add((-50, 50))),
                        iaa.WithHueAndSaturation(
                            iaa.WithChannels(0, iaa.Add((0, 50)))
                        ),
                        iaa.Grayscale(alpha=(0.0, 1.0)),
                        iaa.ChangeColorTemperature((1100, 10000)),
                        iaa.GammaContrast((0.5, 2.0), per_channel=True),
                        iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6), per_channel=True),
                        iaa.AllChannelsCLAHE(),
                        iaa.Affine(shear=(-16, 16)),
                        iaa.ElasticTransformation(alpha=(0, 5.0), sigma=0.25)
                    ],
                    random_order=True,
                )
            ],
            random_order=True
        )

        images_aug = seq(images=images)

        for i, img in enumerate(images_aug):
            cv2.imwrite(os.path.join(os.path.dirname(__file__), "..", "augmented_images", f"{self.filenames[i]}.jpg"), img)
            
    def augmentation_for_cropped_fish(self):
        seq = iaa.Sequential([ 
            iaa.Fliplr(0.5),
            iaa.Affine(
                rotate=(-10, 10),                     # small rotations
                scale=(0.9, 1.1),                     # small zoom
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},  # small shifts
            ),
            iaa.SomeOf((2, 4), [
                iaa.Multiply((0.8, 1.2)),             # brightness
                iaa.LinearContrast((0.4, 1.8)),       # contrast variation
                iaa.AddToHueAndSaturation((-8, 8)),   # subtle colour temperature change
                iaa.GaussianBlur(sigma=(0, 1.0)),     # mild blur
                iaa.AdditiveGaussianNoise(scale=(0, 0.02*255)),  # noise
                iaa.JpegCompression(compression=(70, 95)),       # camera compression
                iaa.Dropout((0.0, 0.04)),             # tiny pixel-level dropout
            ], random_order=True)
        ])
        
        self.active_augmentation = seq

    
    def augment(self, augmented_images_path , number_of_aug_images_per_img = 1, images = None):
        
        if self.active_augmentation == None:
            raise ValueError("No active augmentation sequence. Please select one")
        if images is None and self.image_paths is None:
            raise ValueError("No images or image paths provided for augmentation.")
        os.makedirs(augmented_images_path, exist_ok=True)
        if images is None:
            # Augment using image paths
            for path in self.image_paths:
                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                for i in range(number_of_aug_images_per_img):
                    img_aug = self.active_augmentation(image=img)
                    aug_img = cv2.cvtColor(img_aug, cv2.COLOR_RGB2BGR)
                    base = os.path.basename(path)
                    cv2.imwrite(os.path.join(augmented_images_path, f"{base}_aug_{i}.jpg"), aug_img)
        else:
            # Augment using provided images
            for idx, img in enumerate(images):
                for i in range(number_of_aug_images_per_img):
                    img_aug = self.active_augmentation(image=img)
                    base = self.filenames[idx]
                    cv2.imwrite(os.path.join(augmented_images_path, f"{base}_aug_{i}.jpg"), img_aug)
                    
            
        
            
    
    # Example function to augment images
    def example_augment(self):
        img_path = os.path.join(os.path.dirname(__file__), "..", "images")
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