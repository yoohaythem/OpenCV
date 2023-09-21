from utils import FPS
import multiprocessing
import numpy as np
import argparse
import dlib
import cv2
# win+R: perfmon ,  Processor --> % User Time  <所有实例>

def start_tracker(box, label, rgb, inputQueue, outputQueue):
    # 使用dlib来进行目标追踪，创建一个空的目标追踪器 t。
    t = dlib.correlation_tracker()
    # 矩形区域用于指定需要追踪的目标在当前帧中的位置
    rect = dlib.rectangle(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    # 启动目标追踪器，其中 rgb 是当前帧的图像，rect 是指定的目标位置。这个操作会让追踪器开始跟踪指定位置的目标。
    t.start_track(rgb, rect)

    # 通过死循环，等待inputQueue里有值进入，然后开始处理逻辑
    while True:
        # 从队列中获取下一帧
        rgb = inputQueue.get()

        # 非空（没有到最后一帧）就开始处理
        if rgb is not None:
            t.update(rgb)  # 更新追踪器

            # 获取最新位置
            pos = t.get_position()
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            # 把结果放到输出队列中，包括标签和位置信息
            outputQueue.put((label, (startX, startY, endX, endY)))


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
ap.add_argument("-v", "--video", required=True, help="path to input video file")
ap.add_argument("-o", "--output", type=str, help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.2, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# 一会要放多个追踪器
inputQueues = []
outputQueues = []

# SSD标签
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
           "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

print("[INFO] starting video stream...")
vs = cv2.VideoCapture(args["video"])
writer = None

fps = FPS().start()

if __name__ == '__main__':

    while True:
        (grabbed, frame) = vs.read()

        if frame is None:
            break

        (h, w) = frame.shape[:2]
        width = 600
        r = width / float(w)
        dim = (width, int(h * r))
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if args["output"] is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 30, (frame.shape[1], frame.shape[0]), True)

        # 首先检测位置
        if len(inputQueues) == 0:
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (w, h), 127.5)
            net.setInput(blob)
            detections = net.forward()
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > args["confidence"]:
                    idx = int(detections[0, 0, i, 1])
                    label = CLASSES[idx]
                    if CLASSES[idx] != "person":
                        continue
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    bb = (startX, startY, endX, endY)

                    # 每一次 for 循环，即对每一个检测到的物体，创建输入队列和输出队列
                    # 以下四行是相较以前新增的代码
                    iq = multiprocessing.Queue()
                    oq = multiprocessing.Queue()
                    inputQueues.append(iq)
                    outputQueues.append(oq)

                    # 多核执行 start_tracker 函数，代替以下代码
                    '''
                    t = dlib.correlation_tracker()
                    rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
                    t.start_track(rgb, rect)
                    labels.append(label)
                    trackers.append(t)
                    # 以及原本else里面的更新操作，在 start_tracker 函数通过死循环替代实现：
                    for (t, l) in zip(trackers, labels):  
                        t.update(rgb)
                        pos = t.get_position()
                        startX = int(pos.left())
                        startY = int(pos.top())
                        endX = int(pos.right())
                        endY = int(pos.bottom())
                    '''
                    # 这里效率提升点在于，将耗时操作放在start_tracker中，不需要等待例如dlib.rectangle，t.update的结束。
                    # 主线程只需要将这些操作交由 multiprocessing，将他们 start 之后就可以不用管了。
                    p = multiprocessing.Process(target=start_tracker, args=(bb, label, rgb, iq, oq))
                    # ---Python教学---：p.daemon = True 这行代码是将一个线程（Thread）设置为守护线程（daemon thread）。
                    #                  在Python中，线程分为两种类型：守护线程和非守护线程。
                    # 守护线程（daemon thread）：当程序退出时，守护线程会被立即销毁，而不管它是否执行完毕。
                    #                         如果所有的非守护线程都执行完毕，程序就会退出，并杀死所有守护线程。
                    # 非守护线程（non-daemon thread）：程序会等待所有非守护线程会执行完毕后才退出。通常，主线程是非守护线程。
                    p.daemon = True  # 这是 multiprocessing 的死循环终止的条件：等待主线程结束（这里唯一的非守护线程）
                    p.start()

                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        else:
            # 除了第一帧之后的每一帧只需要做两件事：
            #   ① 将当前帧画面投入到输入队列中，队列会自己用死循环检测队列并处理
            #   ② 从输出队列中拿到 multiprocessing 处理得到的数据，并绘图

            for iq in inputQueues:  # 多个追踪器处理的都是相同输入，等价于 for (t, l) in zip(trackers, labels)
                iq.put(rgb)  # 这里不包含第一帧，因为第一帧走了 if len(inputQueues) == 0 分支

            for oq in outputQueues:  # 得到更新结果，等价于 for (t, l) in zip(trackers, labels)
                (label, (startX, startY, endX, endY)) = oq.get()
                # 绘图
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        if writer is not None:
            writer.write(frame)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        fps.update()

    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    if writer is not None:
        writer.release()

    cv2.destroyAllWindows()
    vs.release()
