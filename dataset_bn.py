import cv2
from albumentations.pytorch import ToTensor
from torch.utils.data import Dataset, DataLoader
from config import get_train_transforms, get_val_transforms, batch_size, train_dir, val_dir
from catalyst.utils import (
    create_dataset, create_dataframe, get_dataset_labeling, map_dataframe
)
import collections
import numpy as np


def get_dataset(data_dir):
    dataset = create_dataset(dirs=f"{data_dir}/*")
    df = create_dataframe(dataset, columns=["class", "filepath"])
    tag_to_label = get_dataset_labeling(df, "class")
    df_with_labels = map_dataframe(
        df,
        tag_column="class",
        class_column="label",
        tag2class=tag_to_label,
        verbose=False
    )
    return df_with_labels


class BnDataset(Dataset):
    def __init__(self, data=None, transform=None, ):
        self.data = data
        self.transform = transform

    def __getitem__(self, i):
        image, label = cv2.imread(self.data.iloc[i]['filepath']), self.data.iloc[i]['label']

        sample = self.transform(image=image)
        image = sample['image']

        OUT = {
            'image': image,
            'label': label
            # 'targets': torch.Tensor(mask).(),
        }
        return OUT

    def __len__(self):
        return len(self.data)


def get_loaders(
        train_dir,
        val_dir,
        train_transforms_fn,
        val_transforms_fn,
        batch_size=64,
        num_workers=12, ):
    train_loader = DataLoader(dataset=BnDataset(get_dataset(train_dir), train_transforms_fn), batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              drop_last=False)

    valid_loader = DataLoader(dataset=BnDataset(get_dataset(val_dir), val_transforms_fn), batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              drop_last=False)

    loaders = collections.OrderedDict()
    loaders["train"] = train_loader
    loaders["valid"] = valid_loader

    return loaders
