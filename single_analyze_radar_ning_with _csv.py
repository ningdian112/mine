import csv
import numpy as np
import os
from glob import glob
import time
import datetime
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import math
import pandas as pd
import csv
import numpy as np
from more_itertools import chunked
from PIL import Image

f_add = r"C:\Users\Ning\Desktop\Radar\with_time\Radar_Camera\b\Daewoong_car_b_average_0\Daewoong_car_b_average_0.csv"
b_add = r"C:\Users\Ning\Desktop\Radar\with_time\Radar_Camera\b\Daewoong_car_b_average_0\Daewoong_car_b_average_0_CHANGED_MIN3.csv"




# num_list =  s_target_list.shape()


color_list=["b","g","r","c","m","y","k","violet"]


################################################################################################################################

# f_target_list
# s_target_list
def pandas_function(iterator):
    for df in iterator:
        yield pd.concat(pd.DataFrame(x) for x in df['content'].map(eval))



##create the new file
import os


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")
        print("---  OK  ---")

    else:
        print("---  There is this folder!  ---")


file = r"D:\radar_dataset\220331\pre_divide\f\prepocessing_save"
mkdir(file)  # 调用函数




#it already be changed
# for z in range(len(b_glob)):
def draw(z_add,color,label):

    f = open(z_add, 'r', encoding='utf-8')
    # rdr = csv.reader(z_add)



    print("which plot is used: \n")
    print(z_add)

    rdr = pd.read_csv(f, index_col=False, header=None)





    # confirm plot
    # Create the zero array
    dis = np.array([])
    rcs = np.array([])


    # for line in rdr:
    for line in range(rdr.shape[0]):
        print(line)
        # line = rdr[line]
        rcs_add = rdr[1][line]
        dis_add = rdr[2][line]
        rcs = np.append(rcs, rcs_add)
        dis = np.append(dis, dis_add)



    print("finish")
    # arr = np.append(arr, np.array([4, 5]))
    # plt.plot(dis, rcs, "ro", color=color_list[z], markersize=2.2)
    # # plt.plot(dis, rcs, color=color_list[z], linewidth=1.0)
    # plt.plot(dis, rcs, "ro", color="b", markersize=2.2)
    #




    plt.plot(dis, rcs, color=color, linestyle='solid', linewidth=0.3, label=label, zorder=1)
    plt.scatter(dis, rcs, color=color, zorder=2, s=0.3, alpha=1, marker='o')



    #
    # if z == 3:


    # df2 = pd.DataFrame(normal_value)
    # df2 = pd.DataFrame(df2).sort_values(by=df2.columns[0], ascending=True, axis=0)
    # df2.to_csv("b_{}.csv".format(z), index=False, header=False)
    print("Save well about csv")
    # f.close()
    return f



# for num_list in range(len(f_glob)):
if True:
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


    origin = draw(f_add,'y',"normal")
    print("original_well")
    change = draw(b_add,'r','abnormal')
    print("changed_well")

    plt.title("compare")

    plt.tight_layout()

    plt.legend()

    print("done")

    plt.savefig('save_combine.png', dpi=500)

    plt.show()
    print("plot down")

    # uese the changed csv not preprocessing
    change.close()
    # the csv without preprocessing
    origin.close()

    plt.close(0)





