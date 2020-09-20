"""The model, data, and runnable code for this simple project exploring experimental tools.

The ML aspect to this is very simple. We'll load MNIST, use a standard feed-forward neural network, and do
very standard train-test loops. The part that is more relevant to my purposes are the integration of three
experiment-management tools:
  1. Hydra             (argument/configuration management)
  2. PyTorch Lightning (code organization)
  3. Botorch/Ax        (hyperparameter tuning)

Usage:
  CLI: TODO

  Jupyter Notebook: TODO
"""

import torch
from argparse             import ArgumentParser
from pytorch_lightning    import LightningModule, LightningDataModule, Trainer
from torchvision.datasets import MNIST
from torchvision          import transforms
from torch.utils.data     import random_split, DataLoader
from torch.optim          import Adam


class FCNN(LightningModule):
    def __init__(
        self,
        layer_sizes = (100,),        # How many and what width fully-connected layers should we use?
        act         = torch.nn.ReLU, # What activation should we use?
    ):
        super().__init__()

        # MNIST images are (1, 28, 28) - (channels, width, height)
        layers, in_size = [], 28*28
        for out_size in layer_sizes:
            layers.append(torch.nn.Linear(in_size, out_size))
            layers.append(act())
            in_size = out_size

        self.hidden_stack = torch.nn.Sequential(*layers)
        self.out_layer    = torch.nn.Linear(in_size, 10)
        self.softmax      = torch.nn.Softmax(dim=1)
        self.loss         = torch.nn.CrossEntropyLoss()

    def forward(self, batch):
        batch_size = batch.size()[0]
        return self.out_layer(self.hidden_stack(batch.view(batch_size, -1)))

    def training_step(self, batch, _):
        batch_x, batch_y = batch
        return self.loss(self(batch_x), batch_y)

    def configure_optimizers(self):
        # TODO(mmd): I don't like that lr is needed here, but batch_size is needed in a different place -- can
        # this method be defined in the Data Module instead?
        return Adam(self.parameters(), lr=1e-3)

    # TODO: evaluation step, testing, etc.

class MNISTDataModule(LightningDataModule):
    def __init__(self, data_dir: str = "/crimea/MNIST", batch_size: int = 32):
        super().__init__()
        self.data_dir   = data_dir
        self.batch_size = batch_size
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # self.dims is returned when you call dm.size(). It is a good practice to set on every DataModule.
        self.dims = (1, 28, 28)

    def setup(self, stage=None):
        self.mnist_test = MNIST(self.data_dir, train=False, download=True, transform=self.transforms)
        mnist_full = MNIST(self.data_dir, train=True, download=True, transform=self.transforms)
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

ACTIVATIONS_MAPPING = {
    'ReLU': torch.nn.ReLU,
}
def main(
    layer_sizes = None,
    batch_size  = 32,
    act         = 'ReLU',
    data_dir    = "/crimea/MNIST",
):
    act = ACTIVATIONS_MAPPING[act]
    model = FCNN(layer_sizes=layer_sizes, act=act)
    data = MNISTDataModule(data_dir=data_dir, batch_size=batch_size)
    trainer = Trainer(gpus=1)

    trainer.fit(model, data)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--layer_sizes', type=int, nargs='+', default=[])
    parser.add_argument('--act', type=str, default='ReLU')
    parser.add_argument('--data_dir', type=str, default='/crimea/MNIST')
    parser.add_argument('--batch_size', type=int, default=32)

    args = parser.parse_args()
    main(**vars(args))
