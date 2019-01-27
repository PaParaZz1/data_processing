import numpy as np
import torch
from torch.utils.data import Dataset


def make_one_hot(space_size, data):
    return (np.arange(space_size) == data).astype(np.int32)


class Name2Category(object):
    def __init__(self, map_file_path):
        with open(map_file_path, 'r') as f:
            map_data = f.readlines()
        self.map = {}
        self.space_size = len(map_data)
        for item in map_data:
            item = item[:-1]
            split_data = item.split(',')
            category = int(split_data[0])
            for j in range(1, len(split_data)):
                self.map[split_data[j]] = category

    def lookup(self, key):
        if key not in self.map:
            raise KeyError("invalid key in Name2Category: {}".format(key))
        return self.map[key]


class NaiveDataset(Dataset):
    def __init__(self, data_file_path, map_file_path, total_year_num=8, history_year_num=3):
        self.data = []
        self.label = []
        self.map_handle = Name2Category(map_file_path)
        with open(data_file_path, 'r') as f:
            origin_data = f.readlines()
        assert(len(origin_data) % total_year_num == 0)
        for i in range(0, len(origin_data), total_year_num):
            for j in range(history_year_num, total_year_num):
                data_item = []
                init_idx = i + j - 3
                for k in range(history_year_num):
                    item = origin_data[init_idx + k][:-1]
                    split_term = item.split('_')
                    scalar_data = [int(split_term[3]), int(split_term[4]), int(split_term[5])]
                    scalar_data_np = np.array(scalar_data)
                    one_hot_data_np = make_one_hot(self.map_handle.space_size, self.map_handle.lookup(split_term[-2]))
                    data_item.append(np.concatenate([scalar_data_np, one_hot_data_np], axis=0))
                self.data.append(np.stack(data_item, axis=0))
                # label
                split_term = origin_data[i + j]
                self.label.append(np.array([int(split_term[3])]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]
        data = torch.from_numpy(data).float()
        label = torch.from_numpy(label).float()
        return data, label


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
    data_file_path = 'processed_data.txt'
    #print(make_one_hot(5, 0))
    handle_dataset = NaiveDataset(data_file_path, map_file_path)
    data, label = handle_dataset[0]
    print(data.shape)
    print(label)
