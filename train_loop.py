from catalyst.contrib.nn import RAdam, Lookahead
from torch.nn import functional as F
from catalyst import dl
from catalyst.utils import metrics, set_global_seed, prepare_cudnn
from config import train_dir, val_dir, batch_size, get_train_transforms, get_val_transforms
from dataset_bn import get_loaders
from model_batchnorm import resnet18


def train_research_bn(model, log):
    optimizer = Lookahead(RAdam(model.parameters(), lr=0.02))

    train_data_transforms, val_data_transforms = get_train_transforms(112), get_val_transforms(112)

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
            x, y = batch['image'], batch['label'],
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
    runner.train(
        model=model,
        optimizer=optimizer,
        loaders=loaders,
        logdir=log,
        num_epochs=15,
        verbose=True,
        load_best_on_end=True,
    )
    return runner


if __name__ == '__main__':
    SEED = 69
    set_global_seed(SEED)
    prepare_cudnn(deterministic=True)

    model = resnet18(train_bn=True)
    train_research_bn(model, "resnet")

    model = resnet18(train_bn=False)
    train_research_bn(model, 'resnet_no_bn')

    model = resnet18(train_bn=True)
    for name, p in model.named_parameters():
        if 'bn' not in name:
            p.requires_grad = False
    train_research_bn(model, 'resnet_bn_only')
