import pytorch_lightning as pl


class LightningDataWrapper(pl.LightningDataModule):
    def __init__(
            self,
            hugginface_trainer,
            **kwargs
    ):
        super().__init__()
        self.hf_trainer = hugginface_trainer
        self.train_batch_size = hugginface_trainer.args.train_batch_size
        self.eval_batch_size = hugginface_trainer.args.eval_batch_size
        self.prepare_data()

    def prepare_data(self):
        return

    def setup(self, stage=None):
        return

    def train_dataloader(self):
        # Lightning replaces sampler in Trainer
        return self.hf_trainer.get_train_dataloader()

    def val_dataloader(self):
        datasets = self.hf_trainer.eval_dataset.values()
        val_dataloaders = [self.hf_trainer.get_eval_dataloader(dataset) for dataset in datasets]
        return val_dataloaders

    def test_dataloader(self):
        datasets = self.hf_trainer.test_dataset.values()
        test_dataloaders = [self.hf_trainer.get_test_dataloader(dataset) for dataset in datasets]
        return test_dataloaders

