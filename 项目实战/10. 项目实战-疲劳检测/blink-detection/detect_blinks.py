# 导入工具包
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import argparse
import time
import dlib
import cv2

FACIAL_LANDMARKS_68_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])


# 公式来源：http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf
# 用于计算眼睛的纵横比（Eye Aspect Ratio，EAR）。EAR是一种用于检测眼睛是否闭合的指标，通常用于检测眨眼动作。
def eye_aspect_ratio(eye):
    '''
    :param eye: 一个包含眼睛关键点坐标的列表或数组，通常包括眼睛的六个关键点，按照如下顺序：
                左眼光点（左眼角）
                左眼左上点（左上眼睑的顶点）
                左眼右上点（左上眼睑的底部）
                右眼光点（右眼角）
                右眼左上点（右上眼睑的顶点）
                右眼右上点（右上眼睑的底部）
    :return: 计算眼睛的纵横比（EAR）：EAR = (A + B) / (2.0 * C)。
    '''
    # 计算眼睛的两个纵向边界的欧氏距离 A 和 B，其中 A 是左上点到右下点的距离，B 是左下点到右上点的距离。
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # 计算眼睛的横向边界的欧氏距离 C，即左眼左光点到右眼右光点的距离。
    C = dist.euclidean(eye[0], eye[3])
    # 计算眼睛的纵横比（EAR）
    ear = (A + B) / (2.0 * C)
    return ear


# 输入参数
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="", help="path to input video file")
args = vars(ap.parse_args())

# 设置判断参数
EYE_AR_THRESH = 0.3  # 横纵比EAR阈值0.3
EYE_AR_CONSEC_FRAMES = 3  # 连续3帧以上满足条件算一次闭眼

# 初始化计数器
COUNTER = 0
TOTAL = 0

# 检测与定位工具
print("[INFO] loading facial landmark predictor...")
# 检测人脸 + 关键点定位，两步走，所以下面是两个对象
# 创建了一个人脸检测器对象 detector
detector = dlib.get_frontal_face_detector()
# 创建了一个特征点预测器对象 predictor，它用于在检测到的人脸上预测关键特征点的位置。
# args["shape_predictor"] 包含了预训练的特征点预测器的模型文件的路径。
predictor = dlib.shape_predictor(args["shape_predictor"])

# 分别取两个眼睛区域
(lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]

# 读取视频
print("[INFO] starting video stream thread...")
vs = cv2.VideoCapture(args["video"])
time.sleep(1.0)


def shape_to_np(shape, dtype="int"):
    # shape.num_parts 表示关键点的总数量，通常是68个。这里先创建一个68*2大小的空数组备用
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    # 通过循环遍历 shape 对象中的每个关键点，提取其 x 和 y 坐标，并将它们存储在 coords 数组中的相应位置。
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


# 遍历每一帧
while True:
    # 预处理
    frame = vs.read()[1]
    if frame is None:
        break

    (h, w) = frame.shape[:2]
    width = 1200  # 宽度太小可能检测不到人脸
    r = width / float(w)
    dim = (width, int(h * r))
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    # 使用前面创建的人脸检测器 detector 来检测灰度图像 gray 中的人脸，并返回一个包含检测到的人脸位置信息的矩形列表 rects。
    # 0：这个参数通常被称为 upsample，它控制了检测器在执行检测时对输入图像的上采样级别。上采样可以提高检测器的性能，但会增加计算成本。
    # 0 表示不进行上采样，1 表示进行一次上采样，2…… 以此类推。上采样可以帮助检测小尺寸的人脸，但也可能导致较慢的检测速度。
    rects = detector(gray, 0)

    # 遍历每一个检测到的人脸
    for rect in rects:
        # 获取坐标
        # 在图像 gray 上对人脸框 rect 进行关键点定位
        shape = predictor(gray, rect)
        # 将 shape 中的关键点信息转换为NumPy数组
        shape = shape_to_np(shape)

        # 提取左右眼位置信息，计算横纵比
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # 算左右眼平均横纵比
        ear = (leftEAR + rightEAR) / 2.0

        # 计算给定特征点的凸包，并绘制眼睛区域
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        # 和之前不同的是，最后一个参数1表示轮廓线的宽度，这里设置为1像素。
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # 检查是否满足阈值
        if ear < EYE_AR_THRESH:
            COUNTER += 1
        else:
            # 如果连续几帧都是闭眼的，总数算一次
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
            # 重置
            COUNTER = 0

        # 显示
        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(10) & 0xFF

    if key == 27:
        break

vs.release()
cv2.destroyAllWindows()
