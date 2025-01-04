import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
import json

def modify_json(json_path, train_idx, valid_idx):

    with open(json_path, 'r') as f:
        data = json.load(f)
    for t in train_idx:
        data[str(t)]['subset'] = 'train'
    for v in valid_idx:
        data[str(v)]['subset'] = 'test'
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

def read_csv(csv_dir ='/media/Harddisk/ghy/C/MTSVRC/preprocess/mtsvrc.csv', k = 5):
    df = pd.read_csv(csv_dir)
    print(csv_dir)
    # Convert one column to a list, excluding NaN values
    for i in range(k):
        train = df.iloc[:,i*2].dropna().tolist()
        test = df.iloc[:,i*2+1].dropna().tolist()
        modify_json('/media/Harddisk/ghy/C/MTSVRC/preprocess/mtsvrc_title.json',[int(cl)-1 for cl in train], [int(cl)-1 for cl in test])
        # print([int(cl) for cl in test])
        yield i

if __name__== '__main__':
    a = read_csv()
    for _ in a:
        break

