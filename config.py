import albumentations as A
from albumentations.pytorch import ToTensorV2

batch_size = 64

train_dir = 'imagenette2-160/train/'
val_dir = 'imagenette2-160/val/'


def get_train_transforms(size):
    TRAIN_TRANSFORMS = A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30),
        A.ImageCompression(),
        A.OneOf([
            A.GaussianBlur(3, p=0.5),
            A.Blur(blur_limit=3),
            A.MotionBlur(blur_limit=3),
            A.NoOp(),
        ]),
        A.OneOf([
            A.IAAAdditiveGaussianNoise(),
            A.IAASharpen(),
            A.CLAHE(),
            A.RandomBrightnessContrast(),
            A.RGBShift(),
            A.RandomGamma(),
            A.HueSaturationValue(),
            A.NoOp(),
        ]),
        A.OneOf([
            A.RandomFog(p=0.5),
            A.RandomSunFlare(src_radius=100, p=0.3),
            A.RandomRain(p=0.5),
            A.RandomSnow(p=0.5),
            A.NoOp()
        ]),
        A.Normalize(),
        ToTensorV2(),

    ])
    return TRAIN_TRANSFORMS


def get_val_transforms(size):
    VALID_TRANSFORMS = A.Compose([
        A.Resize(size, size),
        A.Normalize(),
        ToTensorV2(),

    ])
    return VALID_TRANSFORMS
