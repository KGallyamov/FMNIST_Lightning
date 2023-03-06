import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import logging

log = logging.getLogger(__name__)


class FashionMNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./", train_val_split: float = 0.1, batch_size: int = 64):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.1307,), (0.3081,))])
        self.batch_size = batch_size
        self.n_workers = 3
        self.train_val_split = train_val_split

    def prepare_data(self):
        # download
        FashionMNIST(self.data_dir, train=True, download=True)
        FashionMNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = FashionMNIST(self.data_dir, train=True, transform=self.transform)
            val_size = int(len(mnist_full) * self.train_val_split)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [len(mnist_full) - val_size, val_size])
        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = FashionMNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.mnist_predict = FashionMNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.n_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.n_workers, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.n_workers, drop_last=True)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size, num_workers=self.n_workers, drop_last=True)
