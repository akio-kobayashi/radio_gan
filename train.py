import torch
import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
import torch.utils.data as data
from solver import LitGAN
import torch.utils.data as dat
import torch.multiprocessing as mp
from speech_dataset import SpeechDataset
import speech_dataset
from callback import SaveEveryNEpochs
from argparse import ArgumentParser
import yaml
import warnings
warnings.filterwarnings('ignore')

def main(args, config:dict):

    model = LitGAN(config)
    

    # csv_path, spk_path, stat_path:str, n_frames, max_mask_len, shuffle_data=True       
    train_dataset = SpeechDataset(csv_path=config['train_path'], 
                                  stat_path=config['stat_path'],
                                  n_frames=config['num_frames'], 
                                  max_mask_len=config['max_mask_len'], 
                                  shuffle_data=True
    )
    train_loader = data.DataLoader(dataset=train_dataset,
                                   **config['process'],
                                   pin_memory=False,
                                   shuffle=True, 
                                   collate_fn=speech_dataset.data_processing
    )

    valid_dataset = SpeechDataset(csv_path=config['valid_path'], 
                                  stat_path=config['stat_path'],
                                  n_frames=config['num_frames'], 
                                  max_mask_len=0, 
                                  shuffle_data=True
    )
    valid_loader = data.DataLoader(dataset=valid_dataset,
                                   **config['process'],
                                   pin_memory=False,
                                   shuffle=True, 
                                   collate_fn=speech_dataset.data_processing
    )

    save_callback = SaveEveryNEpochs(config)
    callbacks = [
        #pl.callbacks.ModelCheckpoint( **config['checkpoint'])
        save_callback
    ]
    logger = TensorBoardLogger(**config['logger'])

    trainer = pl.Trainer( callbacks=callbacks,
                          logger=logger,
                          #check_val_every_n_epoch=10,
                          **config['trainer'] )
    trainer.fit(model=model, ckpt_path=args.checkpoint, 
                train_dataloaders=train_loader, val_dataloaders=valid_loader)

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--gpus', nargs='*', type=int)
    args=parser.parse_args()

    torch.set_float32_matmul_precision('high')
    with open(args.config, 'r') as yf:
        config = yaml.safe_load(yf)

    if 'config' in config.keys():
        config = config['config']
        
    main(args, config) 
