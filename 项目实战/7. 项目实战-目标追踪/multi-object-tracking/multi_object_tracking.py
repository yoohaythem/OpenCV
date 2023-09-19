import argparse
import time
import cv2
import numpy as np


# 配置参数
ap = argparse.ArgumentParser()
# -v或--video参数用于指定输入视频文件的路径。
# -t或--tracker参数用于指定所使用的目标跟踪器类型，默认为"kcf"。
ap.add_argument("-v", "--video", type=str, help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="kcf", help="OpenCV object tracker type")
# 解析器会将这些参数保存在args变量中，以供脚本后续使用。
args = vars(ap.parse_args())


# opencv已经实现了的追踪算法
# "csrt": 基于稳定的噪声分析方法的跟踪器 (cv2.TrackerCSRT_create)。
# "kcf": 基于核相关滤波器的跟踪器 (cv2.TrackerKCF_create)。
# "boosting": 基于Boosting算法的跟踪器 (cv2.legacy.TrackerBoosting_create)。
# "mil": 多实例学习的跟踪器 (cv2.TrackerMIL_create)。
# "tld": 基于孪生跟踪框架的跟踪器 (cv2.legacy.TrackerTLD_create)。
# "medianflow": 中值流跟踪器 (cv2.legacy.TrackerMedianFlow_create)。
# "mosse": Minimum Output Sum of Squared Error (MOSSE)跟踪器 (cv2.legacy.TrackerMOSSE_create)。
OPENCV_OBJECT_TRACKERS = {
	"csrt": cv2.TrackerCSRT_create,
	"kcf": cv2.TrackerKCF_create,
	"boosting": cv2.legacy.TrackerBoosting_create,
	"mil": cv2.TrackerMIL_create,
	"tld": cv2.legacy.TrackerTLD_create,
	"medianflow": cv2.legacy.TrackerMedianFlow_create,
	"mosse": cv2.legacy.TrackerMOSSE_create
}

# 创建一个多目标跟踪器，后续通过trackers.add()来新增。
trackers = cv2.legacy.MultiTracker_create()
# 创建一个视频捕捉对象
vs = cv2.VideoCapture(args["video"])

# 视频流
while True:
	# 取当前帧
	frame = vs.read()
	# 第一个参数，也就是frame[0]代表视频是否被成功读取；第二个参数标识读取到的帧画面。
	frame = frame[1]
	# 到头了就结束
	if frame is None:
		break

	# resize每一帧大小，等比例缩放图片至width = 600
	(h, w) = frame.shape[:2]
	width = 600
	r = width / float(w)
	dim = (width, int(h * r))
	# 使用 cv2.INTER_AREA 插值方法进行插值。
	frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

	# 这行代码执行了一次目标跟踪器（trackers）的更新操作。这段代码返回两个值：
	# success: 一个布尔值，表示目标跟踪是否成功。如果成功，通常意味着跟踪器能够找到并更新所有目标的位置。如果失败，可能表示目标已经离开视野或跟踪器丢失了目标。
	# boxes: 一个包含了每个目标的边界框位置的列表。每个边界框通常表示为一个包含四个值的元组 (x, y, w, h)，其中 (x, y) 是边界框的左上角坐标，而 (w, h) 是边界框的宽度和高度。
	# 这个代码片段中，trackers 被用于跟踪视频帧中的目标，然后 success 和 boxes 用于检查跟踪的成功与否以及目标的位置信息。这通常用于实时目标跟踪应用中，以持续更新目标的位置。
	(success, boxes) = trackers.update(frame)

	# 绘制区域
	for box in boxes:
		(x, y, w, h) = [int(v) for v in box]
		# 根据左上角和右下角位置信息，在当前帧上画出矩形框
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

	# 显示当前帧frame
	cv2.imshow("Frame", frame)
	
	# 从键盘输入缓冲区中读取按键事件，并将其存储在变量 key 中。具体操作如下：
	# cv2.waitKey(100)：这个函数等待一个键盘事件，等待时间为 100 毫秒。它会暂停程序的执行，等待用户按下键盘上的按键。如果在指定的时间内（这里是 100 毫秒）没有按键事件发生，函数会返回一个特殊的值 -1，表示等待超时。
	# & 0xFF：这个操作是为了确保 key 的值是一个字节（8 位），因为在某些操作系统上，cv2.waitKey() 可能返回一个大于 255 的整数，表示特殊按键事件。通过进行与运算（&）并将结果截断为一个字节，可以确保 key 始终在 0 到 255 的范围内。
	key = cv2.waitKey(100) & 0xFF

	if key == ord("s"):
		# 按s：选择一个区域
		# 使用 OpenCV 的 selectROI 函数来交互式地选择图像帧中的一个感兴趣区域（ROI，Region of Interest）。具体参数含义如下：
		# 	"Frame"：这是选择框的窗口的标题，通常是显示图像帧的窗口标题。
		# 	frame：这是当前帧的图像，你想要从中选择感兴趣区域的图像帧。
		# 	fromCenter=False：这个参数指定选择框是否应该从中心开始。如果设置为 True，则选择框将从中心点开始，否则从左上角开始。
		# 	showCrosshair=True：这个参数指定是否显示一个十字交叉标记，以帮助用户选择 ROI 区域。
		# 一旦执行这个代码，它会打开一个新窗口，允许用户拖动鼠标来选择感兴趣区域。
		# 当用户完成选择后，选择框的坐标和尺寸将存储在 box 变量中，通常是一个元组，包含 (x, y, w, h)，其中 (x, y) 是选择框的左上角坐标，(w, h) 是选择框的宽度和高度。您可以使用这些坐标和尺寸来定义感兴趣区域，然后在后续处理中使用它。
		box = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)

		# 在OPENCV_OBJECT_TRACKERS字典中，选择创建一个新的追踪器，由参数指定
		tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
		# 使用trackers.add(tracker, frame, box)将跟踪器添加到多目标跟踪器中。其中参数含义如下：
		# 	tracker：要添加的跟踪器对象。
		# 	frame：当前的图像帧，用于初始化跟踪器。
		#	box：感兴趣区域的初始边界框（ROI），通常是一个包含 (x, y, w, h) 的元组，定义了初始的位置和大小。
		# 一旦添加了跟踪器，它就会开始跟踪位于初始边界框中的对象，并在后续帧中更新对象的位置。这样，您就可以实现多目标跟踪。
		trackers.add(tracker, frame, box)

	# 按ESC，退出
	elif key == 27:
		break

# 释放视频捕获对象 vs
vs.release()
# 关闭所有通过 OpenCV 打开的窗口
cv2.destroyAllWindows()
