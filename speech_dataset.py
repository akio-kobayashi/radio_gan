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

    def __init__(self, csv_path:str, spk_path:str, 
                 stat_path:str, n_frames=128, max_mask_len=25, shuffle_data=True) -> None:
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

        self.n_frames = n_frames
        self.max_mask_len = max_mask_len

        self.mean, self.var = C.load_mean_var(stat_path)

        self.df = pd.read_csv(csv_path)
        self.df_nh = self.df[self.df['hearing'] == 'NH']
        self.df_df = self.df[self.df['hearing'] == 'DF']

        self.data_length = max(len(self.df_nh), len(self.df_df))
        self.shuffle_data=shuffle_data
        if self.shuffle_data:
            self.shuffle()

        self.spk2id={}
        self.spk2id['<UNK>'] = 0
        with open(spk_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                ary = line.strip().split()
                if ary[0] != "<UNK>" or ary[0] != "<unk>":
                    self.spk2id[ary[0]] = int(ary[1])

    def __len__(self) -> int:
        return self.data_length

    def shuffle(self):
        if self.shuffle_data:
            replace=True if len(self.df_nh) < self.data_length else False       
            self.df_nh = self.df_nh.sample(n=self.data_length, replace=replace, ignore_index=True)
            replace=True if len(self.df_df) < self.data_length else False       
            self.df_df = self.df_df.sample(n=self.data_length, replace=replace, ignore_index=True)

    def get_range(self, melspec):
        total_frames = melspec.shape[-1]
        assert total_frames >= self.n_frames
        start = np.random.randint(total_frames - self.n_frames + 1)
        end = start + self.n_frames

        mask_size = np.random.randint(0, self.max_mask_len)
        assert self.n_frames > mask_size
        mask_start = np.random.randint(0, self.n_frames - mask_size)

        return start, end, mask_start, mask_size
    
    def prepare_data(self, data):
        start, end, mask_start, mask_size = self.get_range(data)
        ranged = data[...,start:end]
        mask = np.ones_like(data)
        mask[..., mask_start:mask_start+mask_size] =0.

        return ranged, torch.from_numpy(mask).to(self.device)

    def __getitem__(self, idx:int):
        row_nh = self.df_nh.iloc[idx]
        row_df = self.df_df.iloc[idx]

        nh_mel, _ = torch.load(row_nh['melspec'])
        nh_mel = (nh_mel.to(self.device) - self.mean)/self.var
        nh_mel_data, nh_mask = self.prepare_data(nh_mel)
        nh_speaker = self.spk2id[row_nh['speaker']]

        df_mel, _ = torch.load(row_df['melspec'])
        df_mel = (df_mel.to(self.device) - self.mean)/self.var
        df_mel_data, df_mask = self.prepare_data(df_mel)
        df_speaker = self.spk2id[row_df['speaker']]

        return nh_mel_data, nh_mask, nh_speaker, df_mel_data, df_mask, df_speaker
    
def data_processing(data):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    nh_data, nh_mask, nh_spk = [], [], []
    df_data, df_mask, df_spk = [], [], []

    for _nh_d, _nh_m, _nh_s, _df_d, _df_m, _df_s in data:
        nh_data.append(_nh_d)
        nh_mask.append(_nh_m)
        nh_spk.append(_nh_s)

        df_data.append(_df_d)
        df_mask.append(_df_m)
        df_spk.append(_df_s)

    nh_spk = torch.from_numpy(np.array(nh_spk)).to(device)
    df_spk = torch.from_numpy(np.array(df_spk)).to(device)

    nh_data = nh_data.squeeze()
    nh_mask = nh_mask.squeeze()
    df_data = df_data.squeeze()
    df_mask = df_mask.squeeze()

    if nh_data.dim() < 3:
        nh_data = nh_data.unsqueeze(0)
        nh_mask = nh_mask.unsqueeze(0)
        df_data = df_data.unsqueeze(0)
        df_mask = df_mask.unsqueeze(0)

    return nh_data, nh_mask, nh_spk, df_data, df_mask, df_spk
