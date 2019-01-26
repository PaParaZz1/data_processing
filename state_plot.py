import copy
import numpy as np
import matplotlib.pyplot as plt


class MetaData(object):
    drug_number = {}

    def __init__(self, origin_term, is_generated=False):
        self.year = origin_term[0]
        self.id = origin_term[5]
        self.drug = origin_term[6]
        self.drug_report = origin_term[7]
        self.total_county = origin_term[8]
        self.total_state = origin_term[9][:-1]
        self.is_generated = is_generated

    def key(self):
        return self.id + str(self.drug_number[self.drug])

    def __str__(self):
        return (self.year + '_' + self.id + '_' + str(self.drug_number[self.drug]) +
                '_' + self.drug_report + '_' + self.total_county + '_' + self.total_state)


def random_plot(key_list, meta_data_list, random_choice_num=10):
    random_choice = np.random.choice(len(key_list), random_choice_num)
    print(random_choice)
    '''
    for index in random_choice:
        drug_list = []
        for item in meta_data_list:
            if item.key() == key_list[index]:
                drug_list.append(item.drug_report)
        print('---begin')
        for x in drug_list:
            print(x)
        print('---end')
        if len(drug_list) == 8:
            y = [float(i) for i in drug_list]
            x = [i for i in range(2010, 2018)]
            print(len(x))
            print(len(y))
            #plt.plot(x, y)
            #plt.show()
    '''
    count_np = np.zeros(8)
    for i in range(len(key_list)):
        count = 0
        for item in meta_data_list:
            if item.key() == key_list[i]:
                count += 1
        count_np[count - 1] += 1
    sum = 0
    for i in range(8):
        sum += count_np[i]
    print(count_np/sum)


def print_list(list_item):
    if not isinstance(list_item, list):
        raise TypeError('invalid type in func print_list: {}'.format(type(list_item)))
    for item in list_item:
        print(item)


def main(origin_path, target_path):
    year_number = 8
    init_year = 2010
    with open(origin_path, 'r') as f:
        origin = f.readlines()
    print('origin data', len(origin))
    data = origin[1:]
    meta_data_list = []
    drug_dict = {}
    key_list = []
    count = 1
    for item in data:
        split_result = item.split(',')
        if split_result[6] != '' and split_result[6] not in drug_dict.keys():
            # print('append term:{}, count:{}'.format(split_result[6], count))
            drug_dict[split_result[6]] = count
            count += 1
        meta_data_item = MetaData(split_result)
        meta_data_list.append(meta_data_item)
    MetaData.drug_number = drug_dict
    print('drug type num:', len(drug_dict))
    for item in meta_data_list:
        key = item.key()
        if key not in key_list:
            key_list.append(key)
    print('key_list: {}'.format(len(key_list)))
    # random_plot(key_list, meta_data_list)

    # imputation
    for i in range(len(key_list)):
        not_processed = [None for x in range(year_number)]
        template = None

        def mete_data_find(condition, only_one=False):
            ret = []
            for item in meta_data_list:
                if condition(item):
                    ret.append(item)
                    if only_one:
                        return ret
            return ret

        key_find = mete_data_find(lambda x: x.key() == key_list[i])
        template = copy.copy(key_find[0])
        for item in key_find:
            not_processed[int(item.year) - init_year] = item
        print('---not processed')
        print_list(not_processed)
        processed = [None for x in range(year_number)]
        for i in range(year_number):
            year_county_find = mete_data_find(lambda x: x.year == str(2010 + i) and x.id == template.id, only_one=True)
            if not_processed[i] is None:
                imputation_term = copy.copy(template)
                imputation_term.year = str(2010 + i)
                if i == 0:
                    if not_processed[1] is None:
                        imputation_term.drug_report = '0'
                    else:
                        imputation_term.drug_report = not_processed[1].drug_report
                elif i == year_number - 1:
                    if not_processed[year_number-2] is None:
                        imputation_term.drug_report = '0'
                    else:
                        imputation_term.drug_report = not_processed[year_number - 2].drug_report
                else:
                    if not_processed[i+1] is None:
                        imputation_term.drug_report = '0'
                    else:
                        imputation_term.drug_report = str((int(not_processed[i - 1].drug_report) +
                                                           int(not_processed[i + 1].drug_report)) / 2)
                imputation_term.total_county = year_county_find[0].total_county
                imputation_term.total_state = year_county_find[0].total_state
                imputation_term.is_generated = True
                processed[i] = imputation_term
            else:
                processed[i] = not_processed[i]
        print('---processed')
        print_list(processed)

        break


if __name__ == "__main__":
    path = './state1.csv'
    target_path = 'KY.txt'
    main(origin_path=path, target_path=target_path)
