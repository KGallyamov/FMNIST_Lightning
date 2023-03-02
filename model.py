import torch
from torch import nn
import pytorch_lightning as pl
import logging
import matplotlib.pyplot as plt
from collections import OrderedDict
from hydra.utils import instantiate

log = logging.getLogger(__name__)


class FCNModule(pl.LightningModule):
    def __init__(self, opt_config, p_dropout: float = 0.25):
        super().__init__()
        self.model = nn.Sequential(OrderedDict([('fc1', nn.Linear(784, 392)),
                                                ('relu1', nn.ReLU()),
                                                ('drop1', nn.Dropout(p_dropout)),
                                                ('fc12', nn.Linear(392, 196)),
                                                ('relu2', nn.ReLU()),
                                                ('drop2', nn.Dropout(p_dropout)),
                                                ('fc3', nn.Linear(196, 98)),
                                                ('relu3', nn.ReLU()),
                                                ('drop3', nn.Dropout(p_dropout)),
                                                ('fc4', nn.Linear(98, 49)),
                                                ('relu4', nn.ReLU()),
                                                ('output', nn.Linear(49, 10)),
                                                ('logsoftmax', nn.LogSoftmax(dim=1))]))
        self.loss = nn.NLLLoss()
        self.opt_config = opt_config
        self.losses_history = {'train': [], 'val': []}

    def forward(self, batch):
        x, _ = batch
        x = x.view(x.shape[0], -1)
        prediction = self.model(x)

        return prediction

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.shape[0], -1)
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        losses = sum([output["loss"].item() for output in outputs])
        self.losses_history['train'].append(losses / len(outputs))

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        acc = torch.sum(torch.argmax(y_hat, dim=1) == y) / y.size(0)
        self.log('val_loss', loss)
        return {"val_loss": loss, "batch_acc": acc}

    def validation_epoch_end(self, outputs):
        losses = sum([output["val_loss"].detach().item() for output in outputs])
        self.losses_history['val'].append(losses / len(outputs))
        # Discard last batch accuracy since that batch has fewer items
        accuracies = torch.stack([outputs[i]["batch_acc"] for i in range(len(outputs) - 1)])
        print(' Validation Accuracy %0.6f' % torch.mean(accuracies).item())

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        acc = torch.sum(torch.argmax(y_hat, dim=1) == y) / y.size(0)
        return {"test_loss": loss, "batch_acc": acc}

    def test_epoch_end(self, outputs):
        # Discard the last batch accuracy since that batch has fewer items
        accuracies = torch.stack([outputs[i]["batch_acc"] for i in range(len(outputs) - 1)])
        self.log('TestAcc', torch.mean(accuracies).item())

    def configure_optimizers(self):
        optimizer = instantiate(self.opt_config, params=self.parameters())
        return optimizer

    def save_train_history(self):
        train_losses, val_losses = self.losses_history['train'], self.losses_history['val']
        del val_losses[0]  # Redundant value, appears because of extra run in sanity checking
        plt.plot(list(range(len(train_losses))), self.losses_history['train'], label='train', marker='.')
        plt.plot(list(range(len(val_losses))), self.losses_history['val'], label='val', marker='.')
        plt.legend(loc="best")
        plt.xlabel('epoch')
        plt.ylabel('NLL loss')
        plt.savefig('losses.png', bbox_inches='tight')
