import copy
import numpy as np
import matplotlib.pyplot as plt


class MetaData(object):
    drug_number = {}

    def __init__(self, origin_term, is_generated=False):
        self.year = origin_term[0]
        self.county_name = origin_term[2]
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


class GDPData(object):
    def __init__(self, origin_term):
        def is_alpha(c):
            return c >= '0' and c <= '9'

        def extract_digit(string):
            ret = ""
            list_str = list(string)
            for item in list_str:
                if is_alpha(item):
                    ret += item
            return ret

        self.county_name = origin_term[1]
        self.county_name = self.county_name.upper()
        self.person_GDP = origin_term[2]
        self.person_GDP = extract_digit(self.person_GDP)
        self.population = origin_term[5]
        self.population = extract_digit(self.population)

    def __str__(self):
        return self.person_GDP + '_' + self.population

    def total_GDP(self):
        #return int(self.person_GDP) * int(self.population)
        return int(self.person_GDP)


def GDP_weighted(term_list, GDP_list):
    weight_list = []
    drug_list = []
    for item1 in term_list:
        drug_list.append(int(item1.drug_report))
        county_name = item1.county_name
        for item2 in GDP_list:
            # print('item2:{}'.format(item2.county_name))
            if item2.county_name == county_name:
                #weight_list.append(item2.total_GDP())
                #weight_list.append(float(item1.drug_report) / float(item2.population))
                weight_list.append(float(item2.population))
                break
    weight_np = np.array(weight_list).astype(np.float32)
    weight_np /= weight_np.sum()
    print(weight_np)
    drug_np = np.array(drug_list)
    mean = np.mean(drug_np)
    std = np.std(drug_np)
    print(drug_np)
    print(drug_np[drug_np < mean + std].mean())
    return np.dot(weight_np, drug_np)


def drug_weighted(term_list):
    drug_list = [int(x.drug_report) for x in term_list]
    drug_np = np.array(drug_list)
    mean = drug_np.mean()
    std = drug_np.std()
    drug_np_new = drug_np[drug_np > mean + std]


def generate_real_distance(origin_distance_path):
    def lat_lng_to_distance(lat1, lng1, lat2, lng2):
        return 1

    with open(origin_distance_path, 'r') as f:
        data = f.readlines()
    result = {}
    for i in range(len(data)):
        item1 = data[i][:-1]
        item1 = item1.split(',')
        for j in range(i+1, len(data)):
            item2 = data[j][:-1]
            item2 = item2.split(',')
            key1 = item1[0] + item2[0]
            key2 = item2[0] + item1[0]
            distance = lat_lng_to_distance(item1[1], item1[2], item2[1], item2[2])
            result[key1] = distance
            result[key2] = distance
    print('distance dict length: {}'.format(len(result)))
    return result


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


def main(origin_path, target_path, GDP_path, distance_path):
    year_number = 8
    init_year = 2010
    ratio_threshold = 0.2
    generate_real_distance(distance_path)
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

    # GDP
    GDP_list = []
    with open(GDP_path, 'r') as f:
        origin_GDP = f.readlines()
    origin_GDP_data = origin_GDP[1:]
    print('GDP data num', len(origin_GDP_data))
    for i in range(len(origin_GDP_data)):
        origin_term = origin_GDP_data[i].split(',')
        GDP_list.append(GDPData(origin_term))
    # print(GDP_list[0].county_name)
    # print(GDP_list[0].person_GDP)
    # print(GDP_list[0].population)
    # print(GDP_list[0].total_GDP())

    # imputation
    for i in range(len(key_list)):
        not_processed = [None for x in range(year_number)]
        template = None

        def meta_data_find(condition, only_one=False):
            ret = []
            for item in meta_data_list:
                if condition(item):
                    ret.append(item)
                    if only_one:
                        return ret
            return ret

        key_find = meta_data_find(lambda x: x.key() == key_list[i])
        template = copy.copy(key_find[0])
        for item in key_find:
            not_processed[int(item.year) - init_year] = item
        print('---not processed{}'.format(i))
        print_list(not_processed)
        processed = [None for x in range(year_number)]
        for j in range(year_number):
            if not_processed[j] is None:
                year_str = str(2010 + j)
                year_county_find = meta_data_find(lambda x: x.year == year_str and x.id == template.id, only_one=True)
                year_drug_find = meta_data_find(lambda x: x.year == year_str and x.drug == template.drug)
                # other_np = np.array([float(x.drug_report) for x in year_drug_find])
                imputation_term = copy.copy(template)
                imputation_term.year = year_str
                # other_mean = np.mean(other_np)
                # other_std = np.std(other_np)

                if j == 0:
                    if not_processed[1] is None:
                        imputation_term.drug_report = '0'
                    else:
                        ratio = float(not_processed[1].drug_report) / float(not_processed[1].total_county)
                        predict = int(round(ratio * int(year_county_find[0].total_county)))
                        imputation_term.drug_report = str(predict)
                elif j == year_number - 1:
                    if not_processed[year_number-2] is None:
                        imputation_term.drug_report = '0'
                    else:
                        ratio = float(not_processed[year_number - 2].drug_report) / float(not_processed[year_number - 2].total_county)
                        predict = int(round(ratio * int(year_county_find[0].total_county)))
                        imputation_term.drug_report = str(predict)
                else:
                    if not_processed[j-1] is None or not_processed[j+1] is None:
                        imputation_term.drug_report = '0'
                    else:
                        ratio_before = float(not_processed[j-1].drug_report) / float(not_processed[j-1].total_county)
                        ratio_after = float(not_processed[j+1].drug_report) / float(not_processed[j+1].total_county)
                        div = ratio_before / ratio_after
                        if (div > 1 - ratio_threshold and div < 1 + ratio_threshold):
                            stable_flag = True
                        else:
                            stable_flag = False

                        if stable_flag:
                            ratio = (ratio_before + ratio_after) / 2
                            predict = int(round(ratio * int(year_county_find[0].total_county)))
                            imputation_term.drug_report = str(predict)
                        else:
                            estimate_value = (float(not_processed[j-1].drug_report) + float(not_processed[j+1].drug_report)) / 2
                            print('estimate_value', estimate_value)
                            #sample_value = np.random.normal(other_mean, other_std, 1).clip(0, 9999)
                            sample_value = GDP_weighted(year_drug_find, GDP_list)
                            print('sample_value', sample_value)
                            final_value = int(round((estimate_value + sample_value)/2))
                            print('final_value', final_value)
                            imputation_term.drug_report = str(final_value)
                imputation_term.total_county = year_county_find[0].total_county
                imputation_term.total_state = year_county_find[0].total_state
                imputation_term.is_generated = True
                processed[j] = imputation_term
            else:
                processed[j] = not_processed[j]
        print('---processed{}'.format(i))
        print_list(processed)

        if i == 10:
            break


if __name__ == "__main__":
    path = './state1.csv'
    target_path = 'KY.txt'
    GDP_path = 'GDP_output.csv'
    distance_path = 'distance.txt'
    main(origin_path=path, target_path=target_path, GDP_path=GDP_path, distance_path=distance_path)
