# 导入工具包
from collections import OrderedDict
import numpy as np
import argparse
import dlib
import cv2

# https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/
# http://dlib.net/files/

# 参数
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

# 模型中，不同区域人脸关键点的位置
# 与普通的字典（dict）不同，OrderedDict 会记住元素添加的顺序，因此可以按照添加的顺序迭代访问其中的元素。
# 数组索引标号从0开始，所以索引是68点位图对应标号减一！
FACIAL_LANDMARKS_68_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])

FACIAL_LANDMARKS_5_IDXS = OrderedDict([
    ("right_eye", (2, 3)),
    ("left_eye", (0, 1)),
    ("nose", (4))
])


def shape_to_np(shape, dtype="int"):
    # shape.num_parts 表示关键点的总数量，通常是68个。这里先创建一个68*2大小的空数组备用
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    # 通过循环遍历 shape 对象中的每个关键点，提取其 x 和 y 坐标，并将它们存储在 coords 数组中的相应位置。
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
    '''
    :param image: 输入的图像，通常是一张人脸图像。
    :param shape: 包含人脸关键点信息的 Dlib 形状对象。
    :param colors: 可选参数，指定用于绘制关键点和凸包的颜色。如果未指定，将使用默认的颜色列表。
    :param alpha: 可选参数，指定绘制的关键点和凸包的透明度。
    :return: 输入图像和人脸定位信息参数，绘制结果输出图像。
    '''
    # 函数首先创建了两个图像副本，overlay 和 output，这两个副本用于绘制关键点和凸包。
    overlay = image.copy()
    output = image.copy()
    # 默认颜色列表，七种颜色
    if colors is None:
        colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23), (168, 100, 168), (158, 163, 32), (163, 38, 32),
                  (180, 42, 220)]
    # 按照 mouth, right_eyebrow, left_eyebrow, right_eye, left_eye,nose, jaw 的顺序遍历每个人脸特征区域。
    for (i, name) in enumerate(FACIAL_LANDMARKS_68_IDXS.keys()):
        # 得到该人脸特征区域对应的关键点坐标
        (j, k) = FACIAL_LANDMARKS_68_IDXS[name]
        pts = shape[j:k]
        # 果特征区域是下巴（jaw），则使用 cv2.line 绘制线段按顺序连接关键点
        if name == "jaw":
            for l in range(1, len(pts)):  # 按顺序遍历这些点，连起来
                ptA = tuple(pts[l - 1])
                ptB = tuple(pts[l])
                cv2.line(overlay, ptA, ptB, colors[i], 2)
        # 否则使用 cv2.convexHull 计算凸包，并使用 cv2.drawContours 绘制凸包。
        else:
            # 计算给定特征点 pts 的凸包，即形成凸多边形的最小集合，覆盖了所有特征点。
            hull = cv2.convexHull(pts)
            # overlay: 要绘制凸包轮廓的目标图像。
            # [hull]: 凸包的轮廓信息，以列表的形式传递。注意，这里用列表包装 hull，因为该函数需要接受一个轮廓的列表。
            # -1: 要绘制的轮廓的索引。传递 -1 表示绘制所有的轮廓。
            # colors[i]: 指定绘制轮廓的颜色。
            # -1: 这个参数表示填充轮廓内部，而不是只绘制轮廓线条。
            cv2.drawContours(overlay, [hull], -1, colors[i], -1)
    # 将 overlay 图像与原始输入图像按照alpha比例混合在一起，以产生最终的可视化效果。
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    return output


# 检测人脸 + 关键点定位，两步走，所以下面是两个对象
# 创建了一个人脸检测器对象 detector
detector = dlib.get_frontal_face_detector()
# 创建了一个特征点预测器对象 predictor，它用于在检测到的人脸上预测关键特征点的位置。
# args["shape_predictor"] 包含了预训练的特征点预测器的模型文件的路径。
predictor = dlib.shape_predictor(args["shape_predictor"])

# 读取输入数据，预处理
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
width = 500
r = width / float(w)
dim = (width, int(h * r))
image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用前面创建的人脸检测器 detector 来检测灰度图像 gray 中的人脸，并返回一个包含检测到的人脸位置信息的矩形列表 rects。
# 1：表示对图像进行上采样的次数。上采样可以帮助检测小尺寸的人脸。在这里，1 表示不进行额外的上采样。
rects = detector(gray, 1)

# 遍历检测到的人脸区域
for (i, rect) in enumerate(rects):
    # 在图像 gray 上对人脸框 rect 进行关键点定位
    shape = predictor(gray, rect)
    # 将 shape 中的关键点信息转换为NumPy数组
    shape = shape_to_np(shape)

    # 按照 mouth, right_eyebrow, left_eyebrow, right_eye, left_eye,nose, jaw 的顺序遍历每一个部分
    for (name, (i, j)) in FACIAL_LANDMARKS_68_IDXS.items():
        clone = image.copy()
        cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # 在副本图像上绘制人脸的名称

        # 根据识别到的关键点位置画点
        for (x, y) in shape[i:j]:
            cv2.circle(clone, (x, y), 3, (0, 0, 255), -1)

        (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))  # 计算包围关键点的矩形边界框的坐标和尺寸。
        roi = image[y:y + h, x:x + w]  # 从原始图像中提取矩形边界框内的区域，即人脸关键点区域。
        (h, w) = roi.shape[:2]  # 获取裁剪后的人脸区域的高度和宽度。
        width = 250  # 指定目标宽度（缩放后的宽度）。
        r = width / float(w)  # 计算缩放比例，以确保宽度为目标宽度，高度按比例缩放。
        dim = (width, int(h * r))  # 缩放后的宽度和高度。
        roi = cv2.resize(roi, dim, interpolation=cv2.INTER_AREA)  # 将人脸区域缩放到指定的目标宽度和高度

        # 显示每一部分
        cv2.imshow("ROI", roi)
        cv2.imshow("Image", clone)
        cv2.waitKey(0)

    # 展示当前人脸的全部关键点区域（mouth, right_eyebrow, left_eyebrow, right_eye, left_eye,nose, jaw）
    output = visualize_facial_landmarks(image, shape)
    cv2.imshow("Image", output)
    cv2.waitKey(0)
