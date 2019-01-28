import copy
import math
from functools import cmp_to_key
import numpy as np
import matplotlib.pyplot as plt


class MetaData(object):
    drug_number = {}
    is_generated_str = {True: 'G', False: 'O'}

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
                '_' + self.drug_report + '_' + self.total_county + '_' + self.total_state +
                '_' + self.county_name + '_' + self.is_generated_str[self.is_generated])


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


def generate_unique_conuty_data(meta_data_list, output_file='unique_data.csv'):
    result = []
    seperator = ','
    for item in meta_data_list:
        new_item = item.year + seperator + item.county_name + seperator + item.total_county + seperator + item.total_state + '\n'
        result.append(new_item)
    result = list(set(result))
    with open(output_file, 'w') as f:
        f.writelines(result)

def exponential_weighted_change_rate(meta_data_list, beta=0.8, output_file='unique_data.csv'):
    def exponential_weighted(data_list, beta=0.8):
        total_len = len(data_list)
        prev_s = 0
        cur_s = 0
        for i in range(total_len):
            cur_s = prev_s * (1 - beta) + data_list[i] * beta
            prev_s = cur_s
        return cur_s

    term_list = []
    seperator = ','
    '''
    for item in meta_data_list:
        term = item.year + seperator + item.county_name + seperator + item.total_county + seperator + item.total_state + '\n'
        term_list.append(term)
    term_list = list(set(term_list))
    '''
    temp_path = 'finish_version.csv'
    with open(temp_path, 'r') as f:
        term_list = f.readlines()
    def own_cmp(x, y):
        x = x.split(seperator)
        y = y.split(seperator)
        if x[1] > y[1]:
            return 1
        elif x[1] < y[1]:
            return -1
        else:
            if x[0] > y[0]:
                return 1
            else:
                return -1

    term_list = sorted(term_list, key=cmp_to_key(own_cmp))
    print(len(term_list))
    # calculate
    result = []
    estimate_rate_list = []
    for i in range(0, len(term_list), 8):
        rate = []
        for j in range(8):
            split_data = term_list[i+j][:-1].split(seperator)
            rate.append(float(split_data[-2]) * 1.0 / float(split_data[-1]))
        split_data = term_list[i][:-1].split(seperator)
        d_rate = [(rate[x+1] - rate[x]) / rate[x] for x in range(8 - 1)]
        estimate_rate = exponential_weighted(d_rate, beta=beta)
        estimate_rate_list.append(estimate_rate)
        if split_data[1] == "BALLARD":
            print(rate)
            print(d_rate)
            print(estimate_rate)
        result.append(split_data[1] + seperator + "%.5f" % (estimate_rate) + '\n')

    estimate_rate_np = np.array(estimate_rate_list)
    print(estimate_rate_np.max())
    print(estimate_rate_np.min())

    with open(output_file, 'w') as f:
        f.writelines(result)


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


def generate_real_distance(origin_distance_path):
    def lat_lng_to_distance(lat1, lng1, lat2, lng2):
        lat1, lng1, lat2, lng2 = map(math.radians, [float(lat1), float(lng1), float(lat2), float(lng2)])
        d_lat = lat2 - lat1
        d_lng = lng2 - lng1
        tmp = math.sin(d_lat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lng/2)**2
        distance = 2*math.asin(math.sqrt(tmp)) * 6371
        distance = round(distance, 3)
        return distance

    with open(origin_distance_path, 'r') as f:
        data = f.readlines()
    result = {}
    distance_list = []
    for i in range(len(data)):
        item1 = data[i][:-1]
        item1 = item1.split(',')
        for j in range(i+1, len(data)):
            item2 = data[j][:-1]
            item2 = item2.split(',')
            key1 = item1[0] + '_' + item2[0]
            key2 = item2[0] + '_' + item1[0]
            distance = lat_lng_to_distance(item1[1], item1[2], item2[1], item2[2])
            result[key1] = distance
            result[key2] = distance
            distance_list.append(distance)
            #print(key1, distance)
    distance_np = np.array(distance_list)
    #print(distance_np.max())
    #print(distance_np.min())
    #print(distance_np.mean())
    return result


def naive_weighted(term_list, use_mean_std=True):
    if use_mean_std:
        drug_list = [int(x.drug_report) for x in term_list]
        drug_np = np.array(drug_list)
        mean = drug_np.mean()
        std = drug_np.std()
        temp_list = []
        for x in term_list:
            if int(x.drug_report) < mean + std:
                temp_list.append(x)
        term_list = temp_list
    drug_list = [int(x.drug_report) for x in term_list]
    drug_np = np.array(drug_list)
    mean = drug_np.mean()
    std = drug_np.std()
    return np.random.normal(mean, std, 1).clip(1, mean + std)


def distance_weighted(term_list, target, distance_dict, distance_threshold=999):
    drug_list = []
    distance_list = []
    for item in term_list:
        key = target + '_' + item.county_name
        distance = distance_dict[key]
        if distance < distance_threshold:
            distance_list.append(distance)
            drug_list.append(int(item.drug_report))
    drug_np = np.array(drug_list)
    distance_np = np.array(distance_list)
    distance_np /= distance_np.sum()
    # print(distance_np)
    return np.dot(drug_np, distance_np)


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


def print_list(list_item, output_file=None):
    if not isinstance(list_item, list):
        raise TypeError('invalid type in func print_list: {}'.format(type(list_item)))
    for item in list_item:
        if output_file == None:
            print(item)
        else:
            print(item, file=output_file)


def main(origin_path, target_path, GDP_path, distance_path, output_path):
    year_number = 8
    init_year = 2010
    ratio_threshold = 0.2
    output_file = open(output_path, 'w')
    distance_dict = generate_real_distance(distance_path)
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
    #generate_unique_conuty_data(meta_data_list)
    exponential_weighted_change_rate(meta_data_list)
    return
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
    processed_list = []
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
        # print('---not processed{}'.format(i))
        # print_list(not_processed)
        processed = [None for x in range(year_number)]
        for j in range(year_number):
            if not_processed[j] is None:
                year_str = str(2010 + j)
                year_county_find = meta_data_find(lambda x: x.year == year_str and x.id == template.id, only_one=True)
                try:
                    test = year_county_find[0]
                except IndexError:
                    print('miss: not find:({}, {})'.format(year_str, template.id))
                    year_county_find.append(template)
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
                            # print('estimate_value', estimate_value)
                            # sample_value = np.random.normal(other_mean, other_std, 1).clip(0, 9999)
                            # sample_value = GDP_weighted(year_drug_find, GDP_list)
                            target = template.county_name
                            sample_value = distance_weighted(year_drug_find, target, distance_dict)
                            # print('sample_value', sample_value)
                            final_value = int(round(estimate_value*0.7 + sample_value*0.3))
                            # print('final_value', final_value)
                            imputation_term.drug_report = str(final_value)
                imputation_term.total_county = year_county_find[0].total_county
                imputation_term.total_state = year_county_find[0].total_state
                imputation_term.is_generated = True
                processed[j] = imputation_term
            else:
                processed[j] = not_processed[j]
        print_list(processed, output_file=output_file)
        print('---processed{}'.format(i))
    output_file.close()



if __name__ == "__main__":
    path = './state1.csv'
    target_path = 'KY.txt'
    GDP_path = 'GDP_output.csv'
    distance_path = 'distance.txt'
    output_path = 'processed_data.txt'
    main(origin_path=path, target_path=target_path, GDP_path=GDP_path, distance_path=distance_path, output_path=output_path)
