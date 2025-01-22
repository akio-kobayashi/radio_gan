import os, sys
import numpy as np
import torch
import pandas as pd
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import utils.mel_spectrogram as M
from argparse import ArgumentParser
import yaml
import warnings
warnings.filterwarnings('ignore')

'''
    input_csv format:  key, clean, noisy, 
    output_csv format: key, clean, noisy
'''
def compute_features(input_csv, output_csv, output_dir):
    keys, cleans, noisys = [], [], []

    df = pd.read_csv(input_csv)
    for idx, row in df.iterrows():
        clean = M.get_mel_spectrogram(row['clean'])
        clean_path = os.path.join(output_dir, row['key']+'_clean') + '.pt'
        torch.save(clean, clean_path)

        noisy = M.get_mel_spectrogram(row['noisy'])
        noisy_path = os.path.join(output_dir, row['key']+'_noisy') + '.pt'
        torch.save(noisy, noisy_path)
        
        keys.append(row['key'])
        clean.append(clean_path)
        noisy.append(noisy_path)
    
    out_df = pd.DataFrame(index=None)
    out_df['key'], out_df['clean'], out_df['noisy'] = keys, cleans, noisys

    out_df.to_csv(output_csv, index=False)

def compute_mean_var(input_csv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    df = pd.read_csv(input_csv)

    total_frames = 0
    spec_sum=0.
    spec_sum_square=0.
    for idx, row in df.iterrows():
        melspec = torch.load(row['clean']).to(device)
        total_frames += melspec.shape[-1]
        spec_sum += torch.sum(melspec, dim=-1)
        spec_sum_square += torch.sum(torch.square(melspec))
        melspec = torch.load(row['noisy']).to(device)
        total_frames += melspec.shape[-1]
        spec_sum += torch.sum(melspec, dim=-1)
        spec_sum_square += torch.sum(torch.square(melspec))
        
    spec_mean = spec_sum/total_frames 
    spec_var = spec_sum_square/total_frames - torch.square(spec_mean)

    return spec_mean, torch.sqrt(spec_var + 1.e-8)

def save_mean_var(mean, var, path):
    #if isinstance(mean):
    mean = mean.detach().cpu().numpy()
    var = var.detach().cpu().numpy()

    np.savez(path, mean=mean, var=var)

def load_mean_var(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    npz = np.load(path)
    mean = torch.from_numpy(npz['mean']).to(device)
    var = torch.from_numpy(npz['var']).to(device)

    return mean, var

def remove_short_long_features(df, min_frames=200, max_frames=2500):
    for idx, row in df:
        spec = torch.load(row['melspec'])
        if spec.shape[-1] < min_frames:
            df.drop(index=idx, inplace=True)
            continue
        if spec.shape[-1] < max_frames:
            df.drop(index=idx, inplace=True)
            continue

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_csv', type=str, required=True)
    parser.add_argument('--output_csv', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./')
    parser.add_argument('--output_stats', type=str, default='stats.npz')
    args=parser.parse_args()
       
    compute_features(args.input_csv, args.output_csv, args.output_dir)
    mean, var = compute_mean_var(args.output_csv)
    save_mean_var(mean, var, args.output_stats)
    
