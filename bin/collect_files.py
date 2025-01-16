import os, sys
import pandas as pd
import glob
from argparse import ArgumentParser
import yaml
import warnings
warnings.filterwarnings('ignore')

parser = ArgumentParser()
parser.add_argument('--rootdir', type=str, required=True)
parser.add_argument('--output_csv', type=str, required=True)
args=parser.parse_args()

checked={}
speakers=[]
sources=[]
keys=[]
for path in glob.glob(os.path.join(args.rootdir, '**/*.wav'), recursive=True):
    file = os.path.basename(path)
    if file in checked:
        continue
    checked[file] = 1
    speaker = os.path.basename(os.path.dirname(path))
    key = os.path.splitext(file)[0]
    path = os.path.abspath(path)

    keys.append(key)
    sources.append(path)
    speakers.append(speaker)
df = pd.DataFrame(columns=["key", "source", "speaker"])
df['key'], df['source'], df['speaker'] = keys, sources, speakers
df.to_csv(args.output_csv, index=False)
