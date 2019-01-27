import numpy as np
from torch.utils.data import Dataset

class Name2Category(object):
    def __init__(self, map_file_path):
        with open(map_file_path, 'r') as f:
            map_data = f.readlines()
        self.map = {}
        for item in map_data:
            item = item[:-2]
            print(item)
            split_data = item.split(',')
            category = int(split_data[0])
            for j in range(1, len(split_data)):
                self.map[split_data[j]] = category

    def lookup(self, key):
        if key not in self.map.items():
            raise KeyError("invalid key in Name2Category: {}".format(key))
        return self.map[key]


class NaiveDataset(Dataset):
    def __init__(self):
        self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return data


class AdvancedDataset(Dataset):
    def __init__(self):
        self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return data


if __name__ == "__main__":
    map_file_path = 'divide.csv'
    handle = Name2Category(map_file_path)
    print(handle.lookup('LEWIS'))
