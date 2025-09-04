
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMG_SIZE = 512

train_aug = A.Compose(
    [
        A.LongestMaxSize(max_size=IMG_SIZE),
        A.PadIfNeeded(
            min_height=IMG_SIZE,
            min_width=IMG_SIZE,
            border_mode=cv2.BORDER_CONSTANT,
            ),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.10,
            rotate_limit=7,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.6
        ),
        A.GridDistortion(
            num_steps=4,
            distort_limit=0.05,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.10
        ),
        A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.5),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.30),
        A.RandomGamma(gamma_limit=(85, 115), p=0.20),
        A.GaussianNoise(var_limit=(5.0, 15.0), p=0.20),
        A.MedianBlur(blur_limit=3, p=0.05),
        A.Sharpen(alpha=(0.05, 0.15), lightness=(0.9, 1.1), p=0.20),
        A.CoarseDropout(
            max_holes=3,
            max_height=int(IMG_SIZE * 0.08),
            max_width=int(IMG_SIZE * 0.08),
            min_holes=1,
            min_height=int(IMG_SIZE * 0.03),
            min_width=int(IMG_SIZE * 0.03),
            fill_value=0,
            p=0.12
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_visibility=0.4)
)

valid_aug = A.Compose(
    [
        A.LongestMaxSize(max_size=IMG_SIZE),
        A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"])
)