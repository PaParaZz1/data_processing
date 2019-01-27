def deal_GDP():
    path = 'GDP.csv'
    output_path = 'GDP_output.csv'
    with open(path, 'r') as f:
        data = f.readlines()
    result = []
    for i in range(len(data)):
        string = ""
        print(len(data[i]))
        # print(data[i][60])
        # for j in range(len(data[i])):
        j = 0
        while j < len(data[i]):
            if data[i][j] == '\"':
                j += 1
                # print(str(j) + '|' + data[i][j])
                while(data[i][j] != '\"'):
                    if data[i][j] != ',':
                        string += data[i][j]
                    j += 1
                    # print(str(j) + '|' + data[i][j])
                j += 1
            else:
                string += data[i][j]
                j += 1
        result.append(string)
    with open(output_path, 'w') as f:
        f.writelines(result)


def deal_distance():
    name = 'county.txt'
    locate = 'total.txt'
    output_path = 'distance.txt'
    seperator = ','
    with open(name, 'r') as f:
        names = f.readlines()
    with open(locate, 'r') as f:
        locates = f.readlines()
    assert(len(names) == len(locates))
    result = []
    for i in range(len(names)):
        new_name = names[i][4:]
        new_name = new_name[:-9]
        new_name = new_name.upper()
        # print(new_name)

        new_locate = locates[i][:-2]
        # print(new_locate)
        interval = new_locate.find('N')
        lat = new_locate[:interval]
        lng = new_locate[interval+1:-1]

        lat_a = lat.find('a')
        lat_b = lat.find('b')
        lng_a = lng.find('a')
        lng_b = lng.find('b')
        lat_degree = int(lat[0:lat_a])
        lat_minute = int(lat[lat_a+1:lat_b])
        lng_degree = int(lng[0:lng_a])
        lng_minute = int(lng[lng_a+1:lng_b])
        new_lat = "%.3f" % (lat_degree + lat_minute * 1.0 / 60)
        new_lng = "%.3f" % (lng_degree + lng_minute * 1.0 / 60)
        result.append(new_name + seperator + new_lat + seperator + new_lng + '\n')
    with open(output_path, 'w') as f:
        f.writelines(result)


if __name__ == "__main__":
    deal_distance()
