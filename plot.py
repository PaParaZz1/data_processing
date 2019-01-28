import matplotlib.pyplot as plt
import numpy as np


def main():
    #input_path_list = ['unique_data_5.csv', 'unique_data_7.csv',
    input_path_list = ['unique_data_5.csv',
                       'unique_data_8.csv', 'unique_data_95.csv']
    total_list = []
    for i in range(len(input_path_list)):
        value_list = []
        with open(input_path_list[i], 'r') as f:
            term_list = f.readlines()
        for j in range(len(term_list)):
            data = term_list[j][:-1].split(',')[-1]
            value_list.append(float(data))
        total_list.append(value_list)
    x = [i for i in range(118)]
    plt.plot(x, total_list[0], label='beta=0.5')
    plt.plot(x, total_list[1], label='beta=0.8')
    plt.plot(x, total_list[2], label='beta=0.95')
    #plt.plot(x, total_list[3])
    plt.legend(loc='upper right')
    plt.xlabel('county id')
    plt.ylabel('cumulative increasing rate')
    plt.show()


if __name__ == "__main__":
    main()
