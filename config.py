import albumentations as A
from albumentations.pytorch import ToTensor


def get_TRAIN_TRANSFORMS(size):
    TRAIN_TRANSFORMS = A.Compose([
        A.Resize(size[1], size[0]),
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
    ])
    return TRAIN_TRANSFORMS


def get_VALID_TRANSFORMS(size):
    VALID_TRANSFORMS = A.Compose([
        A.Resize(size[1], size[0]),
        A.Normalize(),

    ])
    return VALID_TRANSFORMS
