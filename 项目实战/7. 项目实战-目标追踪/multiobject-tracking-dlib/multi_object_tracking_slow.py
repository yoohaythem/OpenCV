'''
推荐学习顺序：
1. Faster-RCNN
2. SSD
3. YOLO V5
4. Mask-RCNN
'''

# 导入工具包
from utils import FPS
import numpy as np
import argparse
import dlib
import cv2

# 参数
# --prototxt：指定 Caffe 模型的配置文件的路径，即网络结构的描述文件。
# --model：指定训练好的 Caffe 模型的路径，即包含模型权重的二进制文件。
# --video：指定输入视频文件的路径，这是要进行目标检测的视频。
# --output：指定可选的输出视频文件的路径，如果指定了这个参数，检测结果将会被可视化并保存到这个文件中。
# --confidence：指定目标检测的置信度阈值，默认为 0.2。只有当检测到的目标置信度大于此阈值时，才会被保留。
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
ap.add_argument("-v", "--video", required=True, help="path to input video file")
ap.add_argument("-o", "--output", type=str, help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.2, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
"""
参数可以指定为：
--prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt 
--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel 
--video race.mp4
"""

# SSD标签
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
           "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# 读取网络模型
print("[INFO] loading model...")
# 使用 OpenCV 的 cv2.dnn 模块中的 readNetFromCaffe 函数来加载一个深度学习模型。
# 这个模型是以 Caffe 框架的模型文件（prototxt）和相应的预训练权重文件（model）为输入参数。
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# 初始化
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(args["video"])
writer = None  # 视频写入器对象

# 一会要追踪多个目标
trackers = []
labels = []

# 开始计时，记录开始时间。
fps = FPS().start()

while True:
    # 读取一帧
    (grabbed, frame) = vs.read()

    # 是否是最后了
    if frame is None:
        break

    # resize每一帧大小，等比例缩放图片至width = 600，缩小图片减少计算量，提高计算性能。
    (h, w) = frame.shape[:2]
    width = 600
    r = width / float(w)
    dim = (width, int(h * r))
    # 使用 cv2.INTER_AREA 插值方法进行插值。
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    # opencv用的BGR，但是一般的深度学习框架用的是RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 如果要将结果保存，且目前没有视频写入器对象（第一帧之后，这段代码就肯定不会跑了）
    if args["output"] is not None and writer is None:
        # 创建了一个视频编解码器对象，使用 MJPEG 格式进行视频编码。
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        # 创建了一个视频写入器对象，其中参数解释如下：
        #   args["output"] 是输出视频文件的路径，这是命令行参数中指定的。
        #   fourcc 是视频编解码器对象，指定了视频编码格式。
        #   30 是帧率，表示每秒多少帧。
        #   (frame.shape[1], frame.shape[0]) 是输出视频的帧尺寸，它使用输入帧 frame 的宽度和高度。
        #   True 表示视频文件应该以彩色模式编码。
        writer = cv2.VideoWriter(args["output"], fourcc, 30, (frame.shape[1], frame.shape[0]), True)

    # 先检测，再追踪；第一帧检测，剩余帧追踪
    if len(trackers) == 0:
        # 获取blob数据，即处理后的图像
        (h, w) = frame.shape[:2]
        # 创建了一个 blob 对象，它是用于输入神经网络的图像数据表示。
        #   frame: 输入的图像帧，通常是一个三通道的彩色图像。
        #   0.007843: 缩放比例因子，用于归一化图像像素值，将像素值缩放到[-1, 1]的范围内。（(255-127.5)*0.007843=1）
        #   (w, h): 神经网络期望的输入图像尺寸，通常是 (300, 300)。这里的 (w, h) 是图像的宽度和高度。
        #   127.5: 像素均值。在归一化时，将每个像素值减去这个均值，以使图像的均值接近零。 255/2=127.5
        # blob 对象通常用于将图像传递给深度学习模型进行推理。在这种情况下，它是用于输入 Caffe 模型的预处理步骤，以准备图像进行物体检测。
        # blob是一个四维数组，四个维度分别是图像数量、图像的通道数、高度和宽度。
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (w, h), 127.5)

        # 得到检测结果
        # 将之前生成的 blob 对象（图像）传递给神经网络模型 net，以供模型进行后续的推断操作。
        net.setInput(blob)
        # net.forward() 方法触发了模型的前向传播操作，将之前设置的输入数据传递给模型，并计算模型的输出。
        # 一旦前向传播完成，detections 变量将包含模型的预测结果。
        detections = net.forward()
        # detections是一个四维数组，前两维都是1，从第三维开始，分别代表着检测到的物体，以及物体的参数
        # 物体参数的七个参数里，第二个是类别索引，第三个是置信度，4-7代表了相对位置信息(0-1)，按顺序依次为左上x,左上y,右上x,右上y。
        print(detections.shape)  # (1, 1, 100, 7)

        # 遍历得到的检测结果
        for i in np.arange(0, detections.shape[2]):
            # 能检测到多个结果，只保留概率高的
            confidence = detections[0, 0, i, 2]  # 这个参数表示置信度

            # 过滤
            if confidence > args["confidence"]:
                # 从检测列表中提取类标签的索引
                idx = int(detections[0, 0, i, 1])
                # 从检测列表中获取标签
                label = CLASSES[idx]

                # 只保留人的（将这段代码注释，可以看到有一个人被认为是dog）
                if CLASSES[idx] != "person":
                    continue

                # 得到BOX
                # print(detections[0, 0, i, 3:7])  # 位置信息
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])  # 相对位置转换为绝对位置
                (startX, startY, endX, endY) = box.astype("int")

                # 使用dlib来进行目标追踪
                # http://dlib.net/python/index.html#dlib.correlation_tracker
                # 创建一个空的目标追踪器 t。
                t = dlib.correlation_tracker()
                # 矩形区域用于指定需要追踪的目标在当前帧中的位置
                rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
                # 启动目标追踪器，其中 rgb 是当前帧的图像，rect 是指定的目标位置。这个操作会让追踪器开始跟踪指定位置的目标。
                t.start_track(rgb, rect)

                # 保存结果
                labels.append(label)
                trackers.append(t)

                # 绘图，这是最终目标
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    # 如果已经有了框，就可以直接追踪了
    else:
        # 每一个追踪器都要进行更新
        for (t, l) in zip(trackers, labels):
            t.update(rgb)  # 在当前帧上，更新目标位置
            # 以下5行，获取更新后的位置
            pos = t.get_position()
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            # 绘图（矩形框+文字）
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, l, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    # 如果存在视频写入器对象，则把结果保存下来
    if writer is not None:
        writer.write(frame)

    # 显示
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # 退出
    if key == 27:
        break

    # 更新帧计数，用于在时间间隔内记录处理的帧数。
    fps.update()

fps.stop()  # 停止计时，记录结束时间。
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))  # 计算从开始时间到结束时间的总时间，以秒为单位。
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))  # 计算帧率，即在时间间隔内处理的帧数除以总时间。

# 如果存在视频写入器对象，将其释放
if writer is not None:
    writer.release()

cv2.destroyAllWindows()
vs.release()
