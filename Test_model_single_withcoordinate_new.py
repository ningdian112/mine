import torch
import torch.optim as optim
import segmentation_models_pytorch as smp
from torchvision import transforms
from Losses import DiceLoss
import cv2 as cv
import yaml
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.style as mplstyle
mplstyle.use('fast')
#segementation model setting
# model
model_path = r'C:\Ning_Document\DN\testcode\backbone_efficientnet-b3_decch[128, 64, 32]_loss_diceloss_model.pt'

# parameters
batch_size = 1
epochs = 40
learning_rates = []
for i in range(5):
    learning_rates.append(5e-1)
for i in range(15):
    learning_rates.append(1e-1)
for i in range(15):
    learning_rates.append(5e-2)
for i in range(15):
    learning_rates.append(1e-2)
for i in range(15):
    learning_rates.append(5e-3)
for i in range(15):
    learning_rates.append(1e-3)
for i in range(10):
    learning_rates.append(5e-4)
for i in range(10):
    learning_rates.append(1e-4)
lambda1 = lambda epoch: float(learning_rates[epoch])

data_dir = r'C:\Ning_Document\DN\testcode'

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # cuda 사용
print('Using {} device'.format(device))


model = smp.Unet(encoder_name='efficientnet-b3', encoder_depth=3, encoder_weights='imagenet', decoder_channels=[128, 64, 32], in_channels=3, classes=1).to('cuda')
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))




loss_func = DiceLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

loss_arr = []
iou_bg_arr = []
iou_chip_arr = []
precision_arr = []
recall_arr = []
f1_score_arr = []
pixel_accuracy_arr = []
ten_input = []
ten_label = []
ten_output = []
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_class = lambda x: 1.0 * (x > 0.5)

# def of model
# image >> left  image1 >> right
def segementation(str_img,image,image1):
    # 初始化存儲點擊坐標的列表
    save_3d = []
    test_num = 0

    image =  cv2.resize(image, dsize=(640, 360), interpolation=cv2.INTER_CUBIC)  # 0~255
    image1 =  cv2.resize(image1, dsize=(640, 360), interpolation=cv2.INTER_CUBIC)  # 0~255


    data_transforms = transforms.Compose([transforms.ToTensor()])
    input_image = data_transforms(image).unsqueeze(0).to('cuda')  # 0~1.0

    save_3d1 = []
    input_image1 = data_transforms(image1).unsqueeze(0).to('cuda')  # 0~1.0

    with torch.no_grad():
        import time

        # 记录开始时间
        start_time = time.time()
        output = model(input_image)
        output1 = model(input_image1)


        ten_input = fn_tonumpy(input_image)
        ten_output = fn_tonumpy(fn_class(output))
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)

        ten_input1 = fn_tonumpy(input_image1)
        ten_output1 = fn_tonumpy(fn_class(output1))

        # left side
        # output이 그레이스케일 이미지라면 컬러 이미지로 변환
        converted_images = []
        for img in ten_output:
            # this is the chips 2d location


            converted = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            converted_images.append(converted)
        ten_output = np.stack(converted_images)
        ten_output_rgb = ten_output.copy()
        ten_output_rgb[ten_output_rgb[:, :, :, 2] == 1] = [0, 0, 255]
        red_pixels = np.where(np.all(ten_output_rgb == [0, 0, 255], axis=3))


        overlay_output = cv2.addWeighted(ten_input, 0.7, ten_output_rgb, 0.4, 0)



        # right side
        converted_images1 = []
        for img1 in ten_output1:
            # this is the chips 2d location


            converted1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

            converted_images1.append(converted1)
        ten_output1 = np.stack(converted_images1)
        ten_output_rgb1 = ten_output1.copy()
        ten_output_rgb1[ten_output_rgb1[:, :, :, 2] == 1] = [0, 0, 255]
        red_pixels1 = np.where(np.all(ten_output_rgb1 == [0, 0, 255], axis=3))


        overlay_output1 = cv2.addWeighted(ten_input1, 0.7, ten_output_rgb1, 0.4, 0)


################################ visualization

        # left

        red_pixels = np.where(np.all(ten_output_rgb[0] == [0, 0, 255], axis=2))

        # 获取红色像素的x和y坐标
        x_coords = red_pixels[0]
        y_coords = red_pixels[1]

        # 打印红色像素的坐标
        for x, y in zip(x_coords, y_coords):
            test_num = test_num + 1
            print(f"红色像素坐标：({x}, {y} , {test_num})")
            # changed to the xyz

            # get 3d
            matrix_c = twoDto3Dpts([x, y], w_R0, w_T0, cmtx0)
            matrix_c = np.squeeze(matrix_c)
            # print(matrix_c)
            X_cam_str = f'X: {matrix_c[0]:.2f}, Y: {matrix_c[1]:.2f}, Z: {matrix_c[2]:.2f}'
            print(X_cam_str)




        # right


        red_pixels1 = np.where(np.all(ten_output_rgb1[0] == [0, 0, 255], axis=2))

        # 获取红色像素的x和y坐标
        x_coords1 = red_pixels1[0]
        y_coords1 = red_pixels1[1]

        R_W1 = R1 @ w_R0
        T_W1 = R1 @ w_T0 + T1

        # 打印红色像素的坐标
        for x1, y1 in zip(x_coords1, y_coords1):
            print(f"红色像素坐标：({x}, {y})")
            # changed to the xyz

            # get 3d
            matrix_c1 = twoDto3Dpts([x1, y1], R_W1, T_W1, cmtx1)
            matrix_c1 = np.squeeze(matrix_c1)
            # print(matrix_c)
            X_cam_str1 = f'X: {matrix_c1[0]:.2f}, Y: {matrix_c1[1]:.2f}, Z: {matrix_c1[2]:.2f}'
            print(X_cam_str1)

        # red show
        scale = 0.5

        # 记录结束时间
        end_time = time.time()

        # 计算运行时间
        elapsed_time = end_time - start_time

        print(f"代码运行时间：{elapsed_time} 秒")

        img_c = cv2.resize( ten_output_rgb[0], None, fx=scale, fy=scale)  # 为了完整显示，缩小一倍
        img1_c = cv2.resize( ten_output_rgb1[0], None, fx=scale, fy=scale)  # 为了完整显示，缩小一倍


        vtich = np.vstack((img_c,img1_c))

        cv.imshow("merged_img", vtich)


################### left

        # 创建对应的z坐标数组，假设所有红色像素都在z=0的平面上
        z_coords = np.zeros_like(x_coords)

        # 创建一个3D坐标系图
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 绘制红色像素点
        ax.scatter(x_coords, y_coords, z_coords, c='r', marker='o')

        ################### left
        # 创建对应的z坐标数组，假设所有红色像素都在z=0的平面上
        z_coords1 = np.zeros_like(x_coords1)

        # 绘制红色像素点
        ax.scatter(x_coords1, y_coords1, z_coords1, c='b', marker='o')


        # 设置坐标轴标签
        ax.set_xlabel('X轴')
        ax.set_ylabel('Y轴')
        ax.set_zlabel('Z轴')

        # # 显示图形
        plt.show()


        cv2.waitKey(0)

        return overlay_output[0]







# This will contain the calibration set
calibration_settings = {}


# Open and load the calibration_settings.yaml file
def parse_calibration_settings_file(filename):
    global calibration_settings

    if not os.path.exists(filename):
        print('File does not exist:', filename)
        quit()

    print('Using for calibration settings: ', filename)

    with open(filename) as f:
        calibration_settings = yaml.safe_load(f)

    # rudimentray check to make sure correct file was loaded
    if 'camera0' not in calibration_settings.keys():
        print(
            'camera0 key was not found in the settings file. Check if correct calibration_settings.yaml file was passed')
        quit()


# load camera intrinsic parameters to file
def load_camera_intrinsics(camera_name):
    filename = os.path.join('camera_parameters1', camera_name + '_intrinsics.dat')
    camera_matrix = None
    distortion_coefs = None

    with open(filename, 'r') as inf:
        lines = inf.readlines()

    for idx, line in enumerate(lines):
        if line.startswith('intrinsic:'):
            camera_matrix = []
            for row in lines[idx + 1:idx + 4]:  # Assuming camera_matrix is a 3x3 matrix
                elements = [float(val) for val in row.split()]
                camera_matrix.append(elements)

        if line.startswith('distortion:'):
            distortion_coefs = [float(val) for val in lines[idx + 1].split()]

    return np.array(camera_matrix), np.array(distortion_coefs)


# Converts Rotation matrix R and Translation vector T into a homogeneous representation matrix
def _make_homogeneous_rep_matrix(R, t):
    P = np.zeros((4, 4))
    P[:3, :3] = R
    P[:3, 3] = t.reshape(3)
    P[3, 3] = 1

    return P


# Turn camera calibration data into projection matrix
def get_projection_matrix(cmtx, R, T):
    P = cmtx @ _make_homogeneous_rep_matrix(R, T)[:3, :]
    return P




def twoDto3Dpts(point2D, rVec, tVec, cameraMat, height=0):
    """
           Function used to convert given 2D points back to real-world 3D points
           point2D  : An array of 2D points
           rVec     : Rotation vector
           tVec     : Translation vector
           cameraMat: Camera Matrix used in solvePnP
           height   : Height in real-world 3D space
           Return   : output_array: Output array of 3D points

        """
    point3D = []
    point2D = (np.array(point2D, dtype='float32')).reshape(-1, 2)
    numPts = point2D.shape[0]
    point2D_op = np.hstack((point2D, np.ones((numPts, 1))))
    # rMat = cv2.Rodrigues(rVec)[0]
    rMat_inv = np.linalg.inv(rVec)
    kMat_inv = np.linalg.inv(cameraMat)
    for point in range(numPts):
        uvPoint = point2D_op[point, :].reshape(3, 1)
        tempMat = np.matmul(rMat_inv, kMat_inv)
        tempMat1 = np.matmul(tempMat, uvPoint)
        tempMat2 = np.matmul(rMat_inv, tVec)
        s = (height + tempMat2[2]) / tempMat1[2]
        p = tempMat1 * s - tempMat2
        point3D.append(p)

    point3D = (np.array(point3D, dtype='float32')).reshape([-1, 1, 3])
    return point3D


# 初始化存儲點擊坐標的列表
clicked_points = []


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    global clicked_points  # 告訴 Python 在函數內部使用外部的 clicked_points 變數
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "({}, {})".format(x, y)

        # get 3d
        matrix_c = twoDto3Dpts([x, y], w_R0, w_T0, cmtx0)
        matrix_c = np.squeeze(matrix_c)
        # print(matrix_c)
        X_cam_str = f'X: {matrix_c[0]:.2f}, Y: {matrix_c[1]:.2f}, Z: {matrix_c[2]:.2f}'

        clicked_points.append(f"({x}, {y}) - {X_cam_str}")  # 將點擊坐標信息添加到列表

        cv2.circle(param, (x, y), 2, (0, 255, 0), thickness=2)
        cv2.putText(param, xy, (x + 2, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), thickness=1)
        cv2.putText(param, X_cam_str, (x + 2, y + 15), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 255), thickness=1)

        cv2.imshow("frame0", param)


def on_EVENT_LBUTTONDOWN_1(event, x, y, flags, param):
    global clicked_points  # 告訴 Python 在函數內部使用外部的 clicked_points 變數
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "({}, {})".format(x, y)

        # get 3d
        matrix_c = twoDto3Dpts([x, y], w_R, w_T, cmtx1)
        matrix_c = np.squeeze(matrix_c)
        # print(matrix_c)
        X_cam_str = f'X: {matrix_c[0]:.2f}, Y: {matrix_c[1]:.2f}, Z: {matrix_c[2]:.2f}'

        clicked_points.append(f"({x}, {y}) - {X_cam_str}")  # 將點擊坐標信息添加到列表

        cv2.circle(param, (x, y), 2, (0, 255, 0), thickness=2)
        cv2.putText(param, xy, (x + 2, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), thickness=1)
        cv2.putText(param, X_cam_str, (x + 2, y + 15), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 255), thickness=1)

        cv2.imshow("frame0", param)


# After calibrating, we can see shifted coordinate axes in the video feeds directly
def check_calibration(camera0_name, camera0_data, camera1_name, camera1_data, _zshift=50., world_data=None):
    cmtx0 = np.array(camera0_data[0])
    dist0 = np.array(camera0_data[1])
    R0 = np.array(camera0_data[2])
    T0 = np.array(camera0_data[3])
    cmtx1 = np.array(camera1_data[0])
    dist1 = np.array(camera1_data[1])
    R1 = np.array(camera1_data[2])
    T1 = np.array(camera1_data[3])

    w_R0 = world_data[0]
    w_T0 = world_data[1]
    w_R = world_data[2]
    w_T = world_data[3]


    P0 = get_projection_matrix(cmtx0, R0, T0)
    P1 = get_projection_matrix(cmtx1, R1, T1)
    w_P0 = get_projection_matrix(cmtx0, w_R0, w_T0)
    w_P1 = get_projection_matrix(cmtx1, w_R, w_T)


    unitv_points = 5 * np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype='float32').reshape((4, 1, 3))
    # axes colors are RGB format to indicate XYZ axes.
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

    # open the video streams
    # set camera resolutions
    frameWidth = 1280
    frameHeight = 720  # 设置两个摄像头
    cap = cv2.VideoCapture(0)  # 0对应笔记本自带摄像头
    cap.set(3, frameWidth)  # set中，这里的3，下面的4和10是类似于功能号的东西，数字的值没有实际意义
    cap.set(4, frameHeight)
    cap.set(10, 80)  # 设置亮度

    cap1 = cv2.VideoCapture(1)  # 0对应笔记本自带摄像头
    cap1.set(3, frameWidth)  # set中，这里的3，下面的4和10是类似于功能号的东西，数字的值没有实际意义
    cap1.set(4, frameHeight)
    cap1.set(10, 80)  # 设置亮度

    while True:

        # ret0, img = cap.read()
        # ret1, img1 = cap1.read()

        img = cv2.imread(r'C:\Ning_Document\DN\testcode\1_1.jpg')
        img1 = cv2.imread(r'C:\Ning_Document\DN\testcode\2_1.jpg')

        # img = cv2.imread(r'C:\Ning_Document\DN\temugebmethod\test_chips\input_cam1_01.png')
        # img1 = cv2.imread(r'C:\Ning_Document\DN\temugebmethod\test_chips\input_cam2_01.png')


        # project origin points to frame 0
        points, _ = cv.projectPoints(unitv_points, w_R0, w_T0, cmtx0, dist0)
        points = points.reshape((4, 2)).astype(np.int32)
        origin = tuple(points[0])
        for col, _p in zip(colors, points[1:]):
            _p = tuple(_p.astype(np.int32))
            cv.line(img, origin, _p, col, 2)

        # project origin points to frame1
        R_W1 = R1 @ w_R0
        T_W1 = R1 @ w_T0 + T1
        print("R T\n")
        print(R_W1)
        print(T_W1)
        points, _ = cv.projectPoints(unitv_points, R_W1, T_W1, cmtx1, dist1)
        points = points.reshape((4, 2)).astype(np.int32)
        origin = tuple(points[0])
        for col, _p in zip(colors, points[1:]):
            _p = tuple(_p.astype(np.int32))
            cv.line(img1, origin, _p, col, 2)


        scale = 0.5

        img_c = cv2.resize(img, None, fx=scale, fy=scale)  # 为了完整显示，缩小一倍
        img1_c = cv2.resize(img1, None, fx=scale, fy=scale)  # 为了完整显示，缩小一倍

        vtich = np.vstack((img_c, img1_c))

        cv.imshow("merged_img_1", vtich)

        # segmentation
        img = segementation('frame0', img,img1)


        k = cv.waitKey(1)
        if k == 27: break

    cv.destroyAllWindows()


def get_world_space_origin(cmtx, dist, img_path):
    frame = cv.imread(img_path, 1)

    # calibration pattern settings
    rows = calibration_settings['checkerboard_rows']
    columns = calibration_settings['checkerboard_columns']
    world_scaling = calibration_settings['checkerboard_box_size_scale']

    # coordinates of squares in the checkerboard world space
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)

    cv.drawChessboardCorners(frame, (rows, columns), corners, ret)
    cv.putText(frame, "If you don't see detected points, try with a different image", (50, 50), cv.FONT_HERSHEY_COMPLEX,
               1, (0, 0, 255), 1)
    cv.imshow('img', frame)
    cv.waitKey(0)

    ret, rvec, tvec = cv.solvePnP(objp, corners, cmtx, dist)
    R, _ = cv.Rodrigues(rvec)  # rvec is Rotation matrix in Rodrigues vector form

    return R, tvec


def get_cam1_to_world_transforms(cmtx0, dist0, R_W0, T_W0,
                                 cmtx1, dist1, R_01, T_01,
                                 image_path0,
                                 image_path1):
    frame0 = cv.imread(image_path0, 1)
    frame1 = cv.imread(image_path1, 1)

    unitv_points = 5 * np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype='float32').reshape((4, 1, 3))
    # axes colors are RGB format to indicate XYZ axes.
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

    # project origin points to frame 0
    points, _ = cv.projectPoints(unitv_points, R_W0, T_W0, cmtx0, dist0)
    points = points.reshape((4, 2)).astype(np.int32)
    origin = tuple(points[0])
    for col, _p in zip(colors, points[1:]):
        _p = tuple(_p.astype(np.int32))
        cv.line(frame0, origin, _p, col, 2)

    # project origin points to frame1
    R_W1 = R_01 @ R_W0
    T_W1 = R_01 @ T_W0 + T_01
    points, _ = cv.projectPoints(unitv_points, R_W1, T_W1, cmtx1, dist1)
    points = points.reshape((4, 2)).astype(np.int32)
    origin = tuple(points[0])
    for col, _p in zip(colors, points[1:]):
        _p = tuple(_p.astype(np.int32))
        cv.line(frame1, origin, _p, col, 2)

    cv.imshow('frame0', frame0)
    cv.imshow('frame1', frame1)
    while True:
        key1 = cv2.waitKey(1)
        if key1 == ord('p'):
            raise StopIteration
    #                 key1 = cv2.waitKey(1)

    return R_W1, T_W1


# load the r0 t0 r1 t1
def load_extrinsic_calibration_parameters(prefix=''):
    R0, T0, R1, T1 = None, None, None, None

    camera0_rot_trans_filename = os.path.join('camera_parameters1', prefix + 'camera0_rot_trans.dat')
    with open(camera0_rot_trans_filename, 'r') as inf:
        lines = inf.readlines()

    for idx, line in enumerate(lines):
        if line.startswith('R:'):
            R0 = []
            for row in lines[idx + 1:idx + 4]:  # Assuming R0 is a 3x3 matrix
                elements = [float(val) for val in row.split()]
                R0.append(elements)

        # if line.startswith('T:'):
        #     T0 = [[float(val) for val in lines[idx+1].split()]]
        if line.startswith('T:'):
            T0 = []
            for row in lines[idx + 1:idx + 4]:  # Assuming T1 is a 3x1 matrix
                elements = [float(val) for val in row.split()]  # Remove the extra square brackets
                T0.append(elements)

    camera1_rot_trans_filename = os.path.join('camera_parameters1', prefix + 'camera1_rot_trans.dat')
    with open(camera1_rot_trans_filename, 'r') as inf:
        lines = inf.readlines()

    for idx, line in enumerate(lines):
        if line.startswith('R:'):
            R1 = []
            for row in lines[idx + 1:idx + 4]:  # Assuming R1 is a 3x3 matrix
                elements = [float(val) for val in row.split()]
                R1.append(elements)

        if line.startswith('T:'):
            T1 = []
            for row in lines[idx + 1:idx + 4]:  # Assuming T1 is a 3x1 matrix
                elements = [float(val) for val in row.split()]  # Remove the extra square brackets
                T1.append(elements)

    return np.array(R0), np.array(T0), np.array(R1), np.array(T1)


# load the r0 t0 r1 t1
def load_world_extrinsic_calibration_parameters(prefix=''):
    R0, T0, R1, T1 = None, None, None, None

    camera0_rot_trans_filename = os.path.join('camera_parameters1', prefix + 'world_to_camera0_rot_trans.dat')
    with open(camera0_rot_trans_filename, 'r') as inf:
        lines = inf.readlines()

    for idx, line in enumerate(lines):
        if line.startswith('R:'):
            R0 = []
            for row in lines[idx + 1:idx + 4]:  # Assuming R0 is a 3x3 matrix
                elements = [float(val) for val in row.split()]
                R0.append(elements)

        # if line.startswith('T:'):
        #     T0 = [[float(val) for val in lines[idx+1].split()]]
        if line.startswith('T:'):
            T0 = []
            for row in lines[idx + 1:idx + 4]:  # Assuming T1 is a 3x1 matrix
                elements = [float(val) for val in row.split()]  # Remove the extra square brackets
                T0.append(elements)

    camera1_rot_trans_filename = os.path.join('camera_parameters1', prefix + 'world_to_camera1_rot_trans.dat')
    with open(camera1_rot_trans_filename, 'r') as inf:
        lines = inf.readlines()

    for idx, line in enumerate(lines):
        if line.startswith('R:'):
            R1 = []
            for row in lines[idx + 1:idx + 4]:  # Assuming R1 is a 3x3 matrix
                elements = [float(val) for val in row.split()]
                R1.append(elements)

        if line.startswith('T:'):
            T1 = []
            for row in lines[idx + 1:idx + 4]:  # Assuming T1 is a 3x1 matrix
                elements = [float(val) for val in row.split()]  # Remove the extra square brackets
                T1.append(elements)

    return np.array(R0), np.array(T0), np.array(R1), np.array(T1)


if __name__ == '__main__':

    # Open and parse the settings file
    # parse_calibration_settings_file(sys.argv[1])
    # parse_calibration_settings_file(r"C:\Users\wilco_2303\PycharmProjects\DN\temugebmethod\calibration_settings.yaml")
    parse_calibration_settings_file(r"C:\Ning_Document\DN\temugebmethod\calibration_settings.yaml")

    """Step1. Load calibration frames for single cameras"""

    # camera0 intrinsics
    cmtx0, dist0 = load_camera_intrinsics('camera0')  # this will load cmtx and dist to disk

    # camera1 intrinsics
    cmtx1, dist1 = load_camera_intrinsics('camera1')  # this will load cmtx and dist to disk

    """Step2. Load calibration data where camera0 defines the world space origin."""

    R0, T0, R, T = load_extrinsic_calibration_parameters()
    print("to")
    print(len(T0))

    R1 = R;
    T1 = T  # to avoid confusion, camera1 R and T are labeled R1 and T1
    # check your calibration makes sense
    camera0_data = [cmtx0, dist0, R0, T0]
    print(camera0_data)
    camera1_data = [cmtx1, dist1, R1, T1]

    # load world parameters
    w_R0, w_T0, w_R, w_T = load_world_extrinsic_calibration_parameters()
    print("w_to")
    print(len(w_T))
    world_data = [w_R0, w_T0, w_R, w_T]

    check_calibration('camera0', camera0_data, 'camera1', camera1_data, _zshift=60., world_data=world_data)





