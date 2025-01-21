import torch
import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
import torch.utils.data as data
from solver import LitGAN
import torch.utils.data as dat
import torch.multiprocessing as mp
import torchaudio
#from speech_dataset import SpeechDataset
import utils.split_tensor as S
#import speech_dataset
import bin.compute_features as C
from einops import rearrange
from argparse import ArgumentParser
import yaml
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import glob
import os, sys

def main(args, config:dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

    hifigan, vocoder_train_setup, denoiser = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_hifigan')
    hifigan.to(device)
    denoiser.to(device)

    for path in glob.glob(args.dir+'/**.pt', recursive=True):
        mel = torch.load(path).to(device)
        denoising_strength = 0.005
        audio = hifigan(mel).float()
        audio = denoiser(audio.squeeze(1), denoising_strength)
        audio = audio.squeeze(1) * vocoder_train_setup['max_wav_value']
        outpath=os.path.join(args.output_dir, os.path.basename(path).splitext()[0] + '.wav')
        torchaudio.save(uri=outpath, src=audio.to('cpu'))

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--denoising_strength', type=float, default=0.005)
    parser.add_argument('--output_dir', type=str, default='./')
    args=parser.parse_args()
       
    main(args) 
