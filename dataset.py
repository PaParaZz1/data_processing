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
                    one_hot_data_np = make_one_hot(self.map_handle.space_size, self.map_handle.lookup(split_term[-2])-1)
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
    def __init__(self, data_file_path, map_file_path, social_economical_path, total_year_num=8, history_year_num=3, so_eco_dim=11, county_num=120):
        super(AdvancedDataset, self).__init__()
        self.data = []
        self.label = []
        self.county_dict = {}
        self.county_count = 0
        self.map_handle = Name2Category(map_file_path)
        self.social_economical = []
        self.weight = np.zeros((total_year_num-1, so_eco_dim)).astype(np.float32)
        
        with open(data_file_path, 'r') as f:
            origin_data = f.readlines()
        with open(social_economical_path, 'r') as f:
            social_economical_data = f.readlines()
        
        for i in range(len(origin_data)):
            split_term = origin_data[i][:-1].split('_')
            county_name = split_term[-2]
            if county_name not in self.county_dict.keys():
                self.county_dict[county_name] = self.county_count
                self.county_count += 1
        print(len(self.county_dict))
        for i in range(len(social_economical_data)):
            split_data = social_economical_data[i][:-1].split(',')
            self.weight[i//so_eco_dim, i%so_eco_dim] = float(split_data[1][:-1]) / 100.
            year_term = []
            for j in range(county_num):
                year_term.append(int(split_data[2+j]))
            self.social_economical.append(year_term)
        self.social_economical = np.array(self.social_economical).astype(np.float32)
        self.social_economical = self.social_economical.transpose(1, 0)
        self.social_economical = self.social_economical.reshape(120, total_year_num-1, so_eco_dim)
        self.social_economical *= self.weight
        #print(self.social_economical.shape)
        #print(self.weight.shape)
        assert(len(origin_data) % total_year_num == 0)
        for i in range(0, len(origin_data), total_year_num):
            for j in range(history_year_num, total_year_num-1):
                data_item = []
                init_idx = i + j - 3
                for k in range(history_year_num):
                    item = origin_data[init_idx + k][:-1]
                    split_term = item.split('_')
                    county_name = split_term[-2]
                    scalar_data = [int(split_term[3]), int(split_term[4]), int(split_term[5])]
                    scalar_data_np = np.array(scalar_data)
                    scalar_data_np = np.concatenate([scalar_data_np, self.social_economical[self.county_dict[county_name], 0]], axis=0)
                    one_hot_data_np = make_one_hot(self.map_handle.space_size, self.map_handle.lookup(split_term[-2])-1)
                    data_item.append(np.concatenate([scalar_data_np, one_hot_data_np], axis=0))
                self.data.append(np.stack(data_item, axis=0))
                # label
                split_term = origin_data[i + j]
                self.label.append(np.array([int(split_term[3])]))
        '''
        self.data = np.array(self.data)
        self.data_min = self.data.min(axis=2).repeat(1, 1, self.data.size()[2])
        self.data_max = self.data.max(axis=2).repeat0(1, 1, self.data.size()[2])
        self.data = (self.data - self.data_min) / (self.data_max - self.data_min)
        '''

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]
        data = torch.from_numpy(data).float()
        label = torch.from_numpy(label).float()
        return data, label


if __name__ == "__main__":
    map_file_path = 'divide.csv'
    data_file_path = 'processed_data.txt'
    #print(make_one_hot(5, 0))
    #handle_dataset = NaiveDataset(data_file_path, map_file_path)
    handle_dataset = AdvancedDataset(data_file_path, map_file_path, social_economical_path='2010-2016.csv')
    output_path = 'gt_part2.txt'
    output_list = []
    for i in range(len(handle_dataset)):
        feature, label = handle_dataset[i]
        string = ""
        for j in range(3):
            '''
            string += 'feature:{}, {}, {}, [{},{},{},{},{}]---label:{}\n'.format(feature[j, 0] ,feature[j, 1],
                      feature[j, 2], feature[j, 3], feature[j, 4], feature[j, 5], feature[j, 6], feature[j, 7], label[0])
            '''
            for k in range(11+8):
                string += str(feature[j, k]) + ','
            string += '---label:{}\n'.format(label[0])
        output_list.append(string)
    with open(output_path, 'w') as f:
        f.writelines(output_list)
