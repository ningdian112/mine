import os
from glob import glob
import numpy as np
from PIL import Image
from matplotlib import image
from matplotlib import pyplot
from pathlib import Path
import cv2
import pandas as pd
import tensorflow as tf

# about to reduce the time influence input image
# https://stackoverflow.com/questions/57663734/how-to-speed-up-image-loading-in-pillow-python



# make the time series sliding window
###########################################################Window Sliding##################################################################
from matplotlib import pyplot as plt
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
import sklearn.preprocessing as sp


"""
    in this codem, it just has the normal data and with its informs. the input of the dataset is (X,y,Z,w)
    w is the label of the car and side inform
"""


# 不要求列表有序，只要有个列表、有个目标值就行，只能处理1 维的
def find_nearest(array, value):
    array = np.asarray(array)
    if np.abs(array - value) is None:
        print("None of the image file")

    idx = (np.abs(array - value)).argmin()
    return array[idx]



# def _range_label_(range_name=None):
#     """
#     ths is the label name change to the label encoder, in this example, we have 12 types labels.
#     :param range_name:
#     :return:
#     """
#
#     data = ['Daewoong_car_b', 'Daewoong_car_f', 'Daewoong_car_s', 'Hanyeong_car_s','Jihun_car_b','Jihun_car_f' ,
#             'Jihun_car_s','nn1_car_s', 'nn2_car_s', 'Youngjin_car_b','Youngjin_car_f','Youngjin_car_s' ]
#
#     print(data)
#
#     # encoder
#     ohe= sp.OneHotEncoder(handle_unknown = 'ignore')
#     oh_value = ohe.fit_transform(np.array(data).reshape(1, -1))
#
#     print("____after encode\n", )
#     print(oh_value.toarray()[:12])
#     print(ohe.categories_)
#
#
#


# global varieties
data = ['Daewoong_car_b', 'Daewoong_car_f', 'Daewoong_car_s', 'Hanyeong_car_s', 'Jihun_car_b', 'Jihun_car_f',
            'Jihun_car_s', 'nn1_car_s', 'nn2_car_s', 'Youngjin_car_b', 'Youngjin_car_f', 'Youngjin_car_s']

print(data)

# df
a = pd.get_dummies(data)
print(a )
print(a.columns)

# labelencoder
le = sp.LabelEncoder()
le.fit(data)
# print("well")


def _range_label_pd_(range_name=None):

    # tf.compat.v1.disable_eager_execution()
    """
    ths is the label name change to the label encoder, in this example, we have 12 types labels.
    this code used tensorflow one-hot
    :param range_name:
    :return:
    """


    # change this data to the array
    # data_num = len(data)
    # label = np.arange(data_num)
    # # ohe =  tf.compat.v1.sparse_to_dense(label, sparse_values=0,sparse_indices= = 1)
    # ohe = tf.compat.v1.one_hot(indices=label, depth=1, on_value=1.0, off_value=0.0, axis=-1)
    # with tf.compat.v1.Session() as sess:
    #     print(sess.run(ohe))
    #
    # print("prove this is fine")


    # used_label = np.where(data == range_name )
    # used_target =  tf.compat.v1.one_hot(used_label, num_classes)

    label_result = le.transform(range_name)
    print()
    # print(le.inverse_transform([0]))
    # print("well")
    return label_result



def sliding_window_w_img(train, sw_width=100, in_start=10, sw_steps=10, label_type=None, image_add=None, range_label = None):
    """
    ####### with image 使用条件
    必须所得到的csv(TIME,RCS,DIS)与图片在一个文件夹内，图片编号为数字


    该函数实现窗口宽度为、滑动步长为的滑动窗口截取序列数据
    条件给的都是dis的数据，需要切换成具体哪一行
    sw_width 多少行
    out_end dis数据 暂时不使用，窗口到50m截止
    in_start dis数据
    sw_steps 多少行
    需要设定是normal 还是abnormal

    in_start: When the sliding window begins in a data series.
                          In this dataset, when distance = 10.0, input.
    out_end: When the sliding window ends in a data series.
                           In this dataset, when distance = 50.0, End
    sliding width: The width of the sliding window 滑动窗口的窗口宽度.
                          In this dataset, the width range is 100 points.
    sliding steps: The steps of the windows 滑动窗口的滑动步长
                          In this dataset, the step is 10 points.
    range_label: The informs about the car type and the sides.
    """

    # image catch preprocessing.....
    Z = []
    # from there now get the image list
    # 获取根目录路径、子目录路径，根目录和子目录下所有文件名
    img_file_list = glob(image_add + "//*.jpg")
    img_list = os.listdir(image_add)

    os.chdir(image_add)
    file_list = os.listdir('.')
    img_nocsv = []
    # delete csv file
    file_list.pop()

    for file in file_list:
        file2 = file[:-4]
        file3 = eval(file2)

        img_nocsv.append(file3)

    img_array = np.array(img_nocsv)

    # for the csv catching...........
    data = train[:, 1:3]  # 现在加入了time 需要分开看 train[0] = time, train[1] = rcs, train[2] = dis
    X, y, w = [], [], []

    # in_start_d = np.where(train[:, 1] == in_start,["OK"], ['FIND'])

    in_start = 0
    test_ren = (data[:, 1])
    for search in test_ren:
        if search >= 10.0:
            in_start = search
            break

    in_start = np.where(data[:, 1] == in_start)

    in_start = int(in_start[0])
    print(in_start)

    out_end = data.shape[0]

    s_sum = (out_end - in_start) // sw_steps  # 计算滑动次数

    while sw_width + (sw_steps * s_sum) > out_end:  # 滑动次数需要减少到所有数据采集满足窗口宽度的采样数据
        s_sum = s_sum - 1

    new_rows = sw_width + (sw_steps * s_sum)  # 完整窗口包含的行数，丢弃少于窗口宽度的采样数据；
    in_begain = in_start  # 数据第一次开始的point 传送给in_begin

    for _ in range(s_sum):
        in_end = in_begain + sw_width
        # out_end = in_end + out_end
        # out_end = in_end

        # 保证截取样本完整，最大元素索引不超过原序列索引，则截取数据；否则丢弃该样本
        # if out_end < len(data[0]):
        if (sw_width + in_begain) < new_rows:
            # 训练数据以滑动步长截取
            train_seq = data[in_begain:in_end, :2]  # 截取窗口宽度数量的采样点的全部2个特征 dis rcs

            # 此时获取所需图像的array， train_seq的中位值进行查找到合适的time，将其带入原文件查找到image值
            # medium_num = np.median(train_seq,axis=0)

            # 直接得到想要的行
            medium_num = int(in_begain + sw_width / 2)
            medium_time = int(train[medium_num, 0])
            # 此时导入另外一个def，得到照片临近时间点
            medium_time = find_nearest(img_array, medium_time)
            # 通过得到的的时间点返回获取照片
            used_img = image_add + "//" + '{}'.format(medium_time) + ".jpg"
            used_img = cv2.imread(used_img)
            #let it become 224*224
            used_img = cv2.resize(used_img,(224,224))

            # 让得到的数组也单独成为一个img_array Z
            # used_img_4d = np.stack(used_img, axis=0)
            Z.append(used_img)

            # train_seq = train_seq.reshape(1, (len(train_seq)))
            X.append(train_seq)

            # methion: there need [abnormal, normal]
            label_type = np.array(label_type)
            # y = np.append(y, label_type, axis=0)
            if y == []:
                y = label_type
            else:
                y = np.vstack([y,label_type])


            # plus the car inform label

        w.append(_range_label_pd_(range_name=[range_label]))

        in_begain = in_begain + sw_steps

    return np.array(X), np.array(y), np.array(Z), np.array(w)


def find_csv(address):
    used_csv = os.listdir(address)

    used_csv_list = list()
    # in turn to find every csv with their absolute address
    for find_csv_name in used_csv:
        open_file = address + "\\" + find_csv_name + "\*.csv"
        open_file1 = glob(open_file)
        print("find the csv {}".format(find_csv_name))

        # save the absolute address in a new list
        # do not use append[], use the extend
        used_csv_list.extend(open_file1)

    print("find all of the csv from: {}".format(address))


    return  used_csv_list



# train data
# 3 dim with time rcs dis
# abnormal_add = r"C:\Users\Ning\Desktop\C_R_Final\train\abnormal"
# normal_add = r"C:\Users\Ning\Desktop\C_R_Final\train\normal"

# test data
# abnormal_add_test = r"C:\Users\Ning\Desktop\C_R_Final\test\abnormal"
normal_add_test = r"C:\Users\Ning\Desktop\C_R_Final\train\test_autoencoder"


# let it become the csv list
# train data

# normal_add = find_csv(normal_add_test)

# test data

normal_add_test = find_csv(normal_add_test)



# comfirm the label
def label_test(list_name):
    # abnormal is the first label, the normal is the second label
    if 'original' in list_name:
        print("in the list has average")
        print("output original")
        # normal is 0
        return [0,1]
    elif 'noisy' in list_name:
        print("in the list has noisy")
        # abnormal is 1
        return [1,0]




def sort_string(list):
    return sorted(list, key=embedded_numbers)


# glob setting
import re

re_digits = re.compile(r'(\d+)')  # 设置切片规则


def embedded_numbers(list):
    pieces = re_digits.split(list)  # 切成数字和非数字
    pieces[1::2] = map(int, pieces[1::2])  # 将数字部分转成整数
    return pieces


# from there, one address use a def
# just used positive label csv(time, rcs, dis) with image's file
def pro_combine_w_img(address_list):
    pro_glob = address_list

    for z in range(len(pro_glob)):

        print("which plot is used: \n")
        print(pro_glob[z])

        with open(pro_glob[z]) as file_name:
            # from there test the label type
            label_type = label_test(pro_glob[z])
            array = np.loadtxt(file_name, delimiter=",")

            # from there get the location of this csv os.path.basename返回路径最后的文件名 获取路径名：os.path.dirname()
            img_loc = os.path.dirname(pro_glob[z])

            # plus the range inform like a label, actually it has 12 labels
            file_add = os.path.basename(pro_glob[z])

            # 一定要去掉_noisy_0这种尾缀

            str_loca = file_add.find("_org_")
            range_name = file_add[0:str_loca]





            X_train_pro, y_train_pro, Z_train_pro, w_train_pro = sliding_window_w_img(array, label_type=label_type,
                                                                         image_add=img_loc, range_label= range_name)

            if z == 0:
                X_all_save = X_train_pro
                y_all_save = y_train_pro
                Z_all_save = Z_train_pro
                w_all_save = w_train_pro

            else:
                X_all_save = np.vstack((X_all_save, X_train_pro))
                y_all_save = np.vstack((y_all_save, y_train_pro))
                Z_all_save = np.vstack((Z_all_save, Z_train_pro))
                w_all_save = np.vstack((w_all_save, w_train_pro))

    return X_all_save, y_all_save, Z_all_save,w_all_save



# combine all of the train dataset
# now just the positive used this image
# X_train2, y_train2, Z_train2,w_train2 = pro_combine_w_img(normal_add)

print("train dataset preprocessing finished")

# combine all of the test dataset
X_test2, y_test2, Z_test2,w_test2 = pro_combine_w_img(normal_add_test)
print("test dataset preprocessing finished")


# begin to save npz
np.savez("XyZ_224_half_2label_test0516.npz",   X_test = X_test2, y_test =y_test2, Z_test = Z_test2, w_test = w_test2)
print("finished save npz")