import csv
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import math
import pandas as pd
from glob import glob
import csv
import numpy as np

# glob setting
import re
re_digits = re.compile(r'(\d+)') # 设置切片规则
def embedded_numbers(list):
    pieces = re_digits.split(list)  # 切成数字和非数字
    pieces[1::2] = map(int, pieces[1::2])  # 将数字部分转成整数
    return pieces

many_add = r"C:\Users\Ning\Desktop\Radar\ITRC_Radar_Preprocessing\Daewoong_car\s\*.csv"

many_glob = glob(many_add)

def sort_string(list):
    return sorted(list, key=embedded_numbers)


many_glob = sort_string(many_glob)

name ="Daewoong_car_s_noisy"

color_list_20 = ['black', 'gray', 'lightgray', 'lightcoral', 'red', 'chocolate', 'darkorange', 'yellow', 'greenyellow',
                 'forestgreen',
                 'aquamarine', 'paleturquoise', 'cyan', 'dodgerblue', 'cornflowerblue', 'blueviolet', 'fuchsia',
                 'hotpink', 'lightpink',
                 'crimson'
                 ]


def test_image(array_name, array_name_ori, title_name):
    # try confrim the result
    # confirm plot
    normal_value = np.empty((0, 3), float)
    plt.ion()
    ################################################################################################################################
    # plot setting
    n_inter = 5

    x_total = (list(range(0, 51)))
    x_n = len(x_total)
    x_values = x_total[0:x_n:n_inter]

    n_inter = 5
    y_total = (list(range(-20, 31)))
    y_n = len(y_total)
    y_values = y_total[0:y_n:n_inter]

    # line setting
    plt.grid()

    # the title and label of the plot
    plt.xlabel("DIS/m")
    plt.ylabel("RCS/dB${m^2}$")

    x_major_locator = MultipleLocator(n_inter)
    y_major_locator = MultipleLocator(n_inter)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)

    plt.xlim(0, 50)
    plt.ylim(-20, 30)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # before change
    # Create the zero array
    time = np.array([])
    dis = np.array([])
    rcs = np.array([])

    print("This is the test image plot begin: ")
    for line in array_name_ori:
        print(line)
        time = np.append(time,np.array(int(line[0])))
        rcs = np.append(rcs, np.array(float(line[1])))
        dis = np.append(dis, np.array(float(line[2])))

    # plt.plot(dis, rcs, color="b", linestyle='solid', linewidth=0.3, label="ori", zorder=1)
    # plt.scatter(dis, rcs, color="b", zorder=2, s=0.3, alpha=1, marker='o')

    # after change - noisy

    # the noisy one with ± 10 random
    # even-odd arithmetic
    if (z & 1) == 0:  # 注意这里的 & 1
        print("{0} 是偶数".format(z))
        noisy_mis = 1

    else:
        print("{0} 是奇数".format(z))
        noisy_mis = -1


    noisy = np.random.randint(5, high=10, size=None, dtype='int')
    # Create the zero array
    time = np.array([])
    dis = np.array([])
    rcs = np.array([])

    print("This is the test image plot begin: ")
    for line in array_name:
        noisy_before = line[1]
        noisy_after = noisy_before + (noisy*noisy_mis)

        # 指删除了第一行
        line =np.delete(line,1)
        # 插入数组
        line = np.insert(line, 1, noisy_after)

        print(line)
        time = np.append(time,np.array(int(line[0])))
        rcs = np.append(rcs, np.array(float(line[1])))
        dis = np.append(dis, np.array(float(line[2])))

        normal_value = np.row_stack((normal_value,line))

        # normal_value = np.append(normal_value,line)
        # normal_value[:, 1] = dis

    # plt.plot(dis, rcs, color="r", linestyle='solid', linewidth=0.3, label="noisy", zorder=1)
    # plt.scatter(dis, rcs, color="r", zorder=2, s=0.3, alpha=1, marker='o')

    df2 = pd.DataFrame(normal_value)
    df2.to_csv("{}_{}.csv".format(name, z), index=False, header=False)
    print("Save well about csv")


    plt.title("RCS-DIS-{}_{}".format(name,title_name))

    plt.tight_layout()

    plt.legend()

    print("done")

    # plt.savefig('{}_{}.png'.format(name,title_name), dpi=500)

    # plt.show()
    print("plot down")

    plt.close(0)


# single_add = r"C:\Users\Ning\Desktop\b_0_man.csv"

# open as csv
# f = open(single_add, 'r', encoding='utf-8')
# rdr = csv.reader(f)

# open as array
for z in range(len(many_glob)):
    with open(many_glob[z]) as file_name:
        array = np.loadtxt(file_name, delimiter=",")

    print("which plot is used: \n")
    print(many_glob[z])

    # keep the dis = 3

    # let it just record the number 0.0
    array2 = np.round(array, 1)

    # calculate the non-repetitive number
    #from the dis1 to dis2
    array_dis = array2[:,2]

    array_dis_re, re_num = np.unique(array_dis, return_counts=True)

    # get the average of each Dis with 0.1
    # create a new array to accept [0.0, Dis_non]

    tran_dis_re = np.array([array_dis_re])
    tran_dis_re = tran_dis_re.T
    # add two more array time + rcs dim
    array_new = np.insert(tran_dis_re, 0, values=0, axis=1)
    array_new = np.insert(array_new, 0, values=0, axis=1)


    new_turn = 0

    # get the average point
    for turn_dis in array_dis_re:

        print(turn_dis)

        # due to add time dim, need to change array2 dis to dim = 2


        rcs_loc = np.where(array2[:, 2] == turn_dis)
        for line in rcs_loc:
            ele = 0

            print(line)
            count = 0
            for ele in line:
                # change to dim = 1, calculate the rcs
                count = array2[ele, 1] + count
                print(ele)

            count = count / (len(line))
            count = round(count, 2)
            # change the count to dim = 1
            # add last of the point time inside
            array_new[new_turn, 1] = count
            time_median = np.median(line)
            array_new[new_turn, 0] = array2[int(time_median), 0]
            new_turn = new_turn + 1

    # Now input is 3 dis array

    test_image(array_new, array2, z)
