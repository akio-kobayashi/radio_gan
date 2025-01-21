from pytorch_lightning.callbacks import Callback
import os

class SaveEveryNEpochs(Callback):
    def __init__(self, config):
        super().__init__()
        self.save_dir = os.path.join(config['logger']['save_dir'], 'lightning_logs/version_'+
                                     str(config['logger']['version']), 'checkpoints')
        self.every_n_epochs = config['checkpoint']['every_n_epochs']
        os.makedirs(self.save_dir, exist_ok=True)
        self.template = config['checkpoint']['filename']
                                     
    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        step = trainer.global_step
        filename = eval(f'f"""{self.template}"""')
        if (epoch + 1) % self.every_n_epochs == 0:
            save_path = os.path.join(self.save_dir, filename+".ckpt")
            trainer.save_checkpoint(save_path)
            print(f"Checkpoint saved at: {save_path}")
       # print("callback passed")
