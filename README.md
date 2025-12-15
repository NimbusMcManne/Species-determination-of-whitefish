# Species-determination-of-whitefish
A project that aims to differentiate two whitefish (Coregonus maraena and Coregonus widegreni) and correctly determine the correct one from images.

To run augmentation switch to augment branch, activate virtual environment and run:

``
python -m main --n_images "int"``

where "int" is an integer number that specifies how many augmented images is created. If there are less images than number of augmentations then it re-augments images until "n_images" is met.
