import numpy as np
import sys, os, re, gzip, struct
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import pandas as pd
from typing import Tuple, List
from torch import Tensor
import torchaudio
from einops import rearrange
import bin.compute_features as C

class SpeechDataset(torch.utils.data.Dataset):

    def __init__(self, csv_path:str, stat_path:str, n_frames=128, max_mask_len=25, shuffle_data=True) -> None:
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

        self.n_frames = n_frames
        self.max_mask_len = max_mask_len

        self.mean, self.var = C.load_mean_var(stat_path)
        self.mean = rearrange(self.mean, ' b (f c) -> b f c', c=1)
        self.var = rearrange(self.var, ' b (f c) -> b f c', c=1)
        self.df = pd.read_csv(csv_path)
        self.df_clean = self.df[['key', 'clean']]
        self.df_noisy = self.df[['key', 'noisy']]

        self.data_length = max(len(self.df_clean), len(self.df_noisy))

    def __len__(self) -> int:
        return self.data_length

            
    def get_range(self, melspec):
        total_frames = melspec.shape[-1]
        start = np.random.randint(total_frames - self.n_frames + 1)
        end = start + self.n_frames

        mask_size = mask_start = 0
        if self.max_mask_len > 0:
            mask_size = np.random.randint(0, self.max_mask_len)
            assert self.n_frames > mask_size
            mask_start = np.random.randint(0, self.n_frames - mask_size)
            
        return start, end, mask_start, mask_size
    
    def prepare_data(self, data):
        start, end, mask_start, mask_size = self.get_range(data)
        ranged = data[...,start:end]
        mask = np.ones_like(ranged.cpu().numpy())
        mask[..., mask_start:mask_start+mask_size] =0.

        return ranged, torch.from_numpy(mask).to(self.device)

    def __getitem__(self, idx:int):
        row_clean = self.df_clean.iloc[idx]
        row_noisy = self.df_noisy.iloc[idx]

        mel_clean = torch.load(row_clean['clean'])
        mel_clean = (mel_clean.to(self.device) - self.mean)/self.var
        if mel_clean.shape[-1] < self.n_frames:
            mel_clean = torch.cat([mel_clean, torch.zeros(mel_clean.shape[0], mel_clean.shape[-2],
                                                          self.n_frames - mel_clean.shape[-1],
                                                          device=mel_clean.device)], dim=-1)
        mel_clean_data, mask_clean = self.prepare_data(mel_clean)

        mel_noisy = torch.load(row_noisy['melspec'])
        mel_noisy = (mel_noisy.to(self.device) - self.mean)/self.var
        if mel_noisy.shape[-1] < self.n_frames:
            mel_noisy = torch.cat([mel_noisy, torch.zeros(mel_noisy.shape[0], mel_noisy.shape[-2],
                                                          self.n_frames - mel_noisy.shape[-1],
                                                          device=mel_noisy.device)], dim=-1)
        mel_noisy_data, mask_noisy = self.prepare_data(mel_noisy)

        return mel_clean_data, mask_clean, mel_noisy_data, mask_noisy
    
def data_processing(data):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    data_clean, mask_clean = [], []
    data_noisy, mask_noisy = [], []
    
    for _d_clean, _m_clean, _d_noisy, _m_noisy in data:
        data_clean.append(_d_clean)
        mask_clean.append(_m_clean)

        data_noisy.append(_d_noisy)
        mask_noisy.append(_m_noisy)

    data_clean = nn.utils.rnn.pad_sequence(data_clean, batch_first=True)
    mask_clean = nn.utils.rnn.pad_sequence(mask_clean, batch_first=True)
    data_noisy = nn.utils.rnn.pad_sequence(data_noisy, batch_first=True)
    mask_noisy = nn.utils.rnn.pad_sequence(mask_noisy, batch_first=True)
    data_clean = data_clean.squeeze()
    mask_clean = mask_clean.squeeze()
    data_noisy = data_noisy.squeeze()
    mask_noisy = mask_noisy.squeeze()
    if data_clean.dim() < 3:
        data_clean = data_clean.unsqueeze(0)
        mask_clean = mask_clean.unsqueeze(0)
        data_noisy = data_noisy.unsqueeze(0)
        mask_noisy = mask_noisy.unsqueeze(0)

    return data_clean, mask_clean, data_noisy, mask_noisy
