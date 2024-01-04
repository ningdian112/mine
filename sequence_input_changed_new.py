import copy
import random
from tensorflow.python.keras.utils.data_utils import Sequence
import numpy as np
import os


class SequenceData(Sequence, ):
    """
    用于拟合数据序列的基对象，例如一个数据集。
    每一个 Sequence 必须实现 __getitem__ 和 __len__ 方法。 如果你想在迭代之间修改你的数据集，你可以实现 on_epoch_end。 __getitem__ 方法应该范围一个完整的批次。
    Sequence 是进行多进程处理的更安全的方法。这种结构保证网络在每个时期每个样本只训练一次，这与生成器不同。

    整体来讲，getitem 得到最后输出的数据集， len是迭代的batch_size



    """

    def __init__(self, path, batch_size=6,gen_choice = "training"):
        """
        :param path: 原始数据集所在位置
        :param batch_size: 每次批量生成，训练的样本大小。the batch size
        :param datas: 读取的数据集，在这里为找到合适的csv, 文件夹在abnormal和normal里检索。
        :param idx: the training index or testing index
        :param L: the dataset length

        """
        self.gen_choice = gen_choice
        self.path = path
        self.batch_size = batch_size
        self.datas_X, self.datas_y, self.datas_Z = self.open_npz(path)
        self.L = len(self.datas_y)  # 跟 __len__ 函数连接
        self.on_epoch_end()

        # print("test")  # 返回元素个数


    # 返回长度，通过len(<你的实例>)调用(调用len(SequenceData)时返回，返回的是每个epoch我们需要读取数据的次数)
    def __len__(self):
        """
        :return: 返回生成器的长度，也就是总共分批生成数据的次数。
        """
        # print('len')
        return int(np.ceil(self.L / float(self.batch_size)))
        # print(self.batch_size)
        # return self.batch_size

    def on_epoch_end(self):  # 전체 데이터 X 갯수의 배열을받고 셔플한다. #한 epoch을 수행한 후에 fit_generator함수 안에서 호출되는 함수입니다.
        self.indexes = np.arange(len(self.datas_y))

        # if self.shuffle:
        #     np.random.shuffle(self.indexes)
        # print("self.indexes")
        # print(self.indexes)

    def data_generation(self, batch_indexs):
        # 预处理操作
        # 在这里 雷达切分数据 会生成abnormal数据 与同时期的image匹配 image 会变成float
        # batch_datas gotten from the __getitem__ 是int32的矩阵
        # from there, need get the address of single of the csv and image.
        # label to_categorical is not possibility

        # important turn num : if this num is 0, output normal, is 1 output abnormal
        # print(self.datas_y)

        data_X_abnormal = np.empty((0, 2), float)

        # batch_indexs 指的是直接输入的数据 比如从[960 961 ... 988 989] 一共30个数据的数列
        # 如果非要用这个 第一个index是数列中的第多少个 比如有batch_size=30 那么batch_index应该从0~29，num显示的是直接的哪一行 比如960行
        # 所以输出应该是按照数列第一个min 数列最后一个max来计算长度 输出

        # for batch_index, num in enumerate(batch_indexs):
            # print("batch_index")
        before_index = batch_indexs[0]
        after_index = batch_indexs[-1]

        data_X = copy.deepcopy(self.datas_X[before_index:after_index])  # the label
        data_y = copy.deepcopy(self.datas_y[before_index:after_index])  # the label
        # print(data_y)
        data_Z = self.datas_Z[before_index:after_index]  # the image with int type
        # print(self.datas_Z[0:150])

        # firstly change the image to float64
        data_Z = data_Z / 255.0

        # firstly yield is normal data :
        X_train = copy.deepcopy(data_X)
        y_train = copy.deepcopy(data_y)
        Z_train = copy.deepcopy(data_Z)

        random_list = random.sample(list(range(self.batch_size)), int((self.batch_size) / 2))
        random_list.sort()
        # print(random_list)

        for abnormal_num in range(len(random_list)):

            # print('successfully inter the normal')

            # create the abnormal data_X  #########################
            if data_X.shape[0] > random_list[abnormal_num]:
                data_X_before = (data_X[random_list[abnormal_num]])
            else:

                continue

            noisy = np.random.randint(5, high=10, size=None, dtype='int')

            if (int(abnormal_num) & 1) == 0:  # 注意这里的 & 1
                # print("{0} 是偶数")
                noisy_mis = 1

            else:
                # print("{0} 是奇数")
                noisy_mis = -1

            for line in data_X_before:
                noisy_before = line[0]
                noisy_after = noisy_before + (noisy * noisy_mis)

                # # don't let the signal out of range[-20,30]
                if noisy_after >= 29.9:
                    noisy_after = 29.9
                elif noisy_after <= -19.9:
                    noisy_after = -19.9
                else:
                    True

                line = np.delete(line, 0)
                line = np.insert(line, 0, noisy_after)

                data_X_abnormal = np.row_stack((data_X_abnormal, line))

            # create the abnormal data_y
            data_y_abnormal = np.array([1, 0])
            # print("there has abnormal")

            # used the abnormal data output, Z is the same

            X_train[random_list[abnormal_num], :] = data_X_abnormal
            y_train[random_list[abnormal_num]] = data_y_abnormal
            # print("changed nosiy lisyt")
            # print(y_train)

            # change the important turn num back to 0
            data_X_abnormal = np.empty((0, 2), float)
            random_list = random.sample(list(range(self.batch_size)), int((self.batch_size) / 2))
            random_list.sort()
            # print(random_list)
            # print('successfully inter the abnormal{}'.format(abnormal_num))



        return np.array(X_train), np.array(Z_train), np.array(y_train)

    # 即通过索引获取a[0],a[1]这种
    def __getitem__(self, idx):
        """

        :param idx: idx is index that runs from 0 to length of sequence
        :return: 该函数返回每次我们需要的经过处理的数据。
        """

        # print(self.indexes)
        idx = self.indexes[idx * self.batch_size: (idx + 1) * self.batch_size]
        # print("idx")
        # print(idx)


        X_train, Z_train, y_train = self.data_generation(idx)
        # print("test with erevry batch")
        # print(X_train)
        # print(Z_train)
        # print(y_train)
        # print('shape')
        # print(X_train.shape)
        # print(Z_train.shape)
        # print(y_train.shape)




        return ({"Radar_input": X_train, "Camera_input": Z_train}, {"Dense_final": y_train})

    def open_npz(self, path):

        # ***********************************************load the npz dataset******************************************
        """

        np.savez("XyZ.npz",X_train = X_train, y_train = y_train, Z_train = Z_train,X_test = X_test, y_test =y_test, Z_test = Z_test )
        X_train >> Radar data with (100,2) >> (RCS,DIS)
        y_train >> label with normal to abnoral = 0 to 1
        Z_train >> Image data with (n,H,W,C) >> (480,640,3)

        """
        data = np.load(path)
        # print("get the npz data")
        # print("the X train data shape is {}".format(data["X_train"].shape))
        # print("the y train label shape is {}".format(data["y_train"].shape))
        # print("the Z train data shape is {}".format(data["Z_train"].shape))

        if self.gen_choice == "training":
            print("used training")
            X_train_all = data['X_train']
            y_train_all = data['y_train']
            Z_train_all = data['Z_train']

        elif self.gen_choice == "validation":
            print("used validation")
            X_train_all = data['X_test']
            y_train_all = data['y_test']
            Z_train_all = data['Z_test']




        return X_train_all, y_train_all, Z_train_all


path = r"C:\Ning_Document\python_pro\ITRC_Radar_Noisy\input_loading\XyZ_224_half_noab.npz"

# it = iter(SequenceData(path=path))
#
# print(next(it))
