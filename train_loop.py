import os
import torch
from catalyst.contrib.nn import RAdam, Lookahead
from torch.nn import functional as F
from torch.utils.data import DataLoader
from catalyst import dl
from catalyst.contrib.data.transforms import ToTensor
from catalyst.contrib.datasets import MNIST
from catalyst.utils import metrics, set_global_seed, prepare_cudnn


from catalyst.utils import (
    create_dataset, create_dataframe, get_dataset_labeling, map_dataframe
)
# train_dir='/home/deepkot/Downloads/imagenette2-160/train/n01440764'
# dataset = create_dataset(dirs=f"{train_dir}/*", extension="*.jpg")
# df = create_dataframe(dataset, columns=["class", "filepath"])
#
# tag_to_label = get_dataset_labeling(df, "class")
# class_names = [
#     name for name, id_ in sorted(tag_to_label.items(), key=lambda x: x[1])
# ]
#
# df_with_labels = map_dataframe(
#     df,
#     tag_column="class",
#     class_column="label",
#     tag2class=tag_to_label,
#     verbose=False
# )
# df_with_labels.head()
from config import train_dir, val_dir, batch_size, get_train_transforms, get_val_transforms
from dataset_bn import get_loaders
from model_batchnorm import resnet18


def train_research_bn(model):
    SEED = 69
    set_global_seed(SEED)
    prepare_cudnn(deterministic=True)

    optimizer = Lookahead(RAdam(model.parameters(), lr=0.02))


    train_data_transforms, val_data_transforms = get_train_transforms(224), get_val_transforms(224)

    loaders = get_loaders(
        train_dir=train_dir,
        val_dir=val_dir,
        train_transforms_fn=train_data_transforms,
        val_transforms_fn=val_data_transforms,
        batch_size=batch_size,
    )



    class CustomRunner(dl.Runner):

        def predict_batch(self, batch):
            # model inference step
            return self.model(batch[0].to(self.device).view(batch[0].size(0), -1))

        def _handle_batch(self, batch):
            # model train/valid step
            x, y = batch['image'],batch['label'],
            y_hat = self.model(x)

            loss = F.cross_entropy(y_hat, y)
            accuracy01, accuracy03 = metrics.accuracy(y_hat, y, topk=(1, 3))
            self.state.batch_metrics.update(
                {"loss": loss, "accuracy01": accuracy01, "accuracy03": accuracy03}
            )

            if self.state.is_train_loader:
                loss.backward()
                self.state.optimizer.step()
                self.state.optimizer.zero_grad()

    runner = CustomRunner()
    # model training
    runner.train(
        model=model,
        optimizer=optimizer,
        loaders=loaders,
        logdir="./logs",
        num_epochs=15,
        verbose=True,
        load_best_on_end=True,
    )
    return  runner


if __name__ == '__main__':
    train_research_bn(resnet18(train_bn=False))
