
import os
import torch
from catalyst.contrib.nn import RAdam, Lookahead
from torch.nn import functional as F
from torch.utils.data import DataLoader
from catalyst import dl
from catalyst.contrib.data.transforms import ToTensor

from catalyst.contrib.datasets import MNIST
from catalyst.utils import metrics, set_global_seed, prepare_cudnn


def train_research_bn(model):
    SEED = 69
    set_global_seed(SEED)
    prepare_cudnn(deterministic=True)

    optimizer = Lookahead(RAdam(model.parameters(), lr=0.02))

    loaders = {
        "train": DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=ToTensor()), batch_size=32),
        "valid": DataLoader(MNIST(os.getcwd(), train=False, download=True, transform=ToTensor()), batch_size=32),
    }



    class CustomRunner(dl.Runner):

        def predict_batch(self, batch):
            # model inference step
            return self.model(batch[0].to(self.device).view(batch[0].size(0), -1))

        def _handle_batch(self, batch):
            # model train/valid step
            x, y = batch
            y_hat = self.model(x.view(x.size(0), -1))

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
        num_epochs=5,
        verbose=True,
        load_best_on_end=True,
    )
    return  runner
# model inference
# for prediction in runner.predict_loader(loader=loaders["valid"]):
#     assert prediction.detach().cpu().numpy().shape[-1] == 10
# # model tracing
# # traced_model = runner.trace(loader=loaders["valid"])