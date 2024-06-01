
from dataset import NalanDataset
import pickle
import sys
import os

filename = 'dataset.bin'

def load_dataset():
    with open(filename, 'rb') as fp:
        ds = pickle.load(fp)
        return ds
    
if __name__ == '__main__':
    if os.path.exists(filename):
        ds = load_dataset()
        print(f'{filename}已存在，训练集大小：{len(ds)}, 样例数据如下：')
        ids,text = ds[5]
        print(ids, text)
        sys.exit(0)

    ds = NalanDataset()
    with open(filename, 'wb') as fp:
        ds.build_train_data()
        pickle.dump(ds, fp)
        print(f'{filename}已生成，训练集大小：{len(ds)}, 样例数据如下：')
        ids,text = ds[5]
        print(ids, text)