import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random

class ImageAugmentor:
    """
    Creates augmented versions of images using various transformations
    """
    def __init__(self, augmentation_factor=5):
        self.augmentation_factor = augmentation_factor
        self.transformations = [
            'horizontal_flip',
            'vertical_flip',
            'rotate_90',
            'rotate_180',
            'rotate_270',
            'brightness_increase',
            'brightness_decrease',
            'contrast_increase',
            'contrast_decrease',
            'blur'
        ]

    def horizontal_flip(self, image):
        return cv2.flip(image, 1)

    def vertical_flip(self, image):
        return cv2.flip(image, 0)

    def rotate_90(self, image):
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    def rotate_180(self, image):
        return cv2.rotate(image, cv2.ROTATE_180)

    def rotate_270(self, image):
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    def brightness_increase(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, 50)
        v[v > 255] = 255
        final_hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    def brightness_decrease(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.subtract(v, 50)
        v[v < 0] = 0
        final_hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    def contrast_increase(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.add(l, 30)
        l[l > 255] = 255
        final_lab = cv2.merge((l, a, b))
        return cv2.cvtColor(final_lab, cv2.COLOR_LAB2BGR)

    def contrast_decrease(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.subtract(l, 30)
        l[l < 0] = 0
        final_lab = cv2.merge((l, a, b))
        return cv2.cvtColor(final_lab, cv2.COLOR_LAB2BGR)

    def blur(self, image):
        return cv2.GaussianBlur(image, (5, 5), 0)

    def augment_image(self, image_path, output_dir, base_name):
        """
        Apply random transformations to create augmented versions
        """
        image = cv2.imread(str(image_path))
        if image is None:
            return

        original_name = Path(image_path).stem
        ext = Path(image_path).suffix

        # Create augmented versions
        for i in range(self.augmentation_factor):
            # Randomly select transformation
            transform_name = random.choice(self.transformations)
            transformed = getattr(self, transform_name)(image)

            # Save augmented image
            aug_name = f"{original_name}_aug_{i}_{transform_name}{ext}"
            output_path = Path(output_dir) / aug_name
            cv2.imwrite(str(output_path), transformed)

    def augment_directory(self, source_dir, output_dir):
        """
        Augment all images in a directory
        """
        source_path = Path(source_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG']:
            image_files.extend(source_path.glob(ext))

        print(f"Augmenting {len(image_files)} images from {source_dir}...")
        for image_file in tqdm(image_files):
            self.augment_image(image_file, output_dir, image_file.stem)

        print(f"Created {len(image_files) * self.augmentation_factor} augmented images in {output_dir}")

if __name__ == '__main__':
    random.seed(42)

    print("=" * 60)
    print("Creating Augmented Primary Dataset")
    print("=" * 60)

    augmentor = ImageAugmentor(augmentation_factor=5)

    # Augment classification datasets
    print("\nAugmenting classification crop images...")
    augmentor.augment_directory(
        'datasets/primary/classification/crop',
        'datasets/primary_augmented/classification/crop'
    )

    print("\nAugmenting classification weed images...")
    augmentor.augment_directory(
        'datasets/primary/classification/weed',
        'datasets/primary_augmented/classification/weed'
    )

    # Augment detection datasets
    print("\nAugmenting detection images...")
    augmentor.augment_directory(
        'datasets/primary/detection/images',
        'datasets/primary_augmented/detection/images'
    )

    # Copy labels for detection (no augmentation needed for labels)
    if Path('datasets/primary/detection/labels').exists():
        import shutil
        shutil.copytree(
            'datasets/primary/detection/labels',
            'datasets/primary_augmented/detection/labels',
            dirs_exist_ok=True
        )
        print("Copied detection labels to augmented dataset")

    # Augment segmentation datasets
    print("\nAugmenting segmentation images...")
    augmentor.augment_directory(
        'datasets/primary/segmentation/images',
        'datasets/primary_augmented/segmentation/images'
    )

    # Copy labels for segmentation
    if Path('datasets/primary/segmentation/labels').exists():
        shutil.copytree(
            'datasets/primary/segmentation/labels',
            'datasets/primary_augmented/segmentation/labels',
            dirs_exist_ok=True
        )
        print("Copied segmentation labels to augmented dataset")

    print("\n" + "=" * 60)
    print("Augmented Primary Dataset Created!")
    print("=" * 60)
