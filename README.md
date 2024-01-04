## twocamera_world_coordinates_visualization
这个代码使两个相机在不同角度下，经过棋盘图checkboard二次矫正后的坐标轴参数，在同一个世界坐标轴视觉化的过程。
The code aligns two cameras set at different angles after the measurement version of the second correction of the extrinsic parameters, in the same world coordinate visualization process.

## RadarRCS_generator_npzfile
The radar RCS signal dataset is stored in the npze file, generate it as a suitable input size for the training model.

## RadarRCS_slidingwindow_savenpz
Sliding window is a technique for extracting data from left to right. Our sliding window can be configured with three different parameters; the position of the window, the width of the window and the step of the window. The signal and image fragment are then saved as an npz file.
	
## RadarRCS_twocsv_compare.py
Compare with two csv with different labels.
