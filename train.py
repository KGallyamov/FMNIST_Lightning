import os
import hydra
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning.callbacks import EarlyStopping

from model import *
from datamodule import *

log = logging.getLogger(__name__)

torch.set_float32_matmul_precision('medium')


@hydra.main(version_base=None, config_path='config', config_name='config')
def my_app(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # datamodule
    data = FashionMNISTDataModule(os.getcwd(), batch_size=cfg.params.batch_size)

    # model
    fcn = FCNModule(cfg.opt)

    # train model
    callback = EarlyStopping(monitor="val_loss", mode="min", patience=4)
    trainer = pl.Trainer(max_epochs=cfg.params.epochs, accelerator='gpu', callbacks=[callback])

    trainer.fit(model=fcn, datamodule=data)
    trainer.test(model=fcn, datamodule=data)
    fcn.save_train_history()


if __name__ == '__main__':
    my_app()
