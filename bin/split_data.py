import pandas as pd
import random
from argparse import ArgumentParser
import warnings
warnings.filterwarnings('ignore')

parser = ArgumentParser()
parser.add_argument('--input_csv', type=str, required=True)
args=parser.parse_args()

# CSVファイルの読み込み
data = pd.read_csv(args.input_csv)

# speakerごとに行をランダムに1つずつ選択
valid_rows = data.groupby("speaker").apply(lambda x: x.sample(1)).reset_index(drop=True)

# valid.csvを構成する行のindexを取得
valid_indices = valid_rows.index

# valid.csv以外の行をtrain.csvとして構成
train_rows = data.drop(index=valid_indices)

# 出力ファイル名
valid_file = "valid.csv"
train_file = "train.csv"

# CSVファイルとして出力
valid_rows.to_csv(valid_file, index=False)
train_rows.to_csv(train_file, index=False)
