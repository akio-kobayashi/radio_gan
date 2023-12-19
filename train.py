import torch
import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
import torch.utils.data as data
from lite.solver import LitGAN
import torch.utils.data as dat
from speech_dataset import SpeechDataset
import speech_dataset
from argparse import ArgumentParser
import yaml
import warnings
warnings.filterwarnings('ignore')

def main(args, config:dict):

    model = LitGAN(config)
    
       
    train_dataset = SpeechDataset(nh_path=config['train_normal_hearing'], 
                                  df_path=config['train_deaf'], 
                                  spk_path=config['speakers'], 
                                  stat_path=config['stats']
                                  n_frames=config['num_frames'], 
                                  max_mask_len=config['max_mask_len'], 
                                  shuffle_data=True
    )
    train_loader = data.DataLoader(dataset=train_dataset,
                                   **config['process'],
                                   pin_memory=True,
                                   shuffle=False, 
                                   collate_fn=lambda x: speech_dataset.data_processing
    )
    valid_dataset = SpeechDataset(nh_path=config['valid_normal_hearing'], 
                                  df_path=config['valid_deaf'], 
                                  spk_path=config['speakers'], 
                                  stat_path=config['stats'],
                                  n_frames=config['num_frames'], 
                                  max_mask_len=config['max_mask_len'], 
                                  shuffle_data=True
    )

    valid_loader = data.DataLoader(dataset=valid_dataset,
                                   **config['process'],
                                   pin_memory=True,
                                   shuffle=False, 
                                   collate_fn=lambda x: speech_dataset.data_processing
    )

    callbacks=[]
    #callbacks = [
    #    pl.callbacks.ModelCheckpoint( **config['checkpoint'])
    #]
    logger = TensorBoardLogger(**config['logger'])

    trainer = pl.Trainer( callbacks=callbacks,
                          logger=logger,
                          devices=args.gpus,
                          check_val_every_n_epoch=10,
                          **config['trainer'] )
    trainer.fit(model=model, ckpt_path=args.checkpoint, 
                train_dataloaders=train_loader,
                val_dataloaders=valid_loader)

if __name__ == '__main__':
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