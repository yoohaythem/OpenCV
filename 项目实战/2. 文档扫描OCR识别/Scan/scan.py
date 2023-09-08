# 1. 导入工具包
import numpy as np
import argparse
import cv2

# 2. 设置参数
# 创建了一个ArgumentParser对象，它将帮助你定义和解析命令行参数。
ap = argparse.ArgumentParser()
# 使用add_argument方法定义了一个命令行参数。具体解释如下：
# -i 和 --image 是参数的名称，其中 -i 是参数的简短形式，--image 是参数的长形式。用户可以选择使用其中一个来指定参数。
# required = True 表示这个参数是必需的，用户必须提供图像的路径。
# help = "Path to the image to be scanned" 是参数的帮助文本，它会在用户请求帮助时显示，以帮助用户理解参数的用途。
ap.add_argument("-i", "--image", required=True, help="Path to the image to be scanned")
# 使用parse_args()方法解析命令行参数，并将解析结果存储在一个字典对象args中。这个字典包含了用户提供的参数及其对应的值。
args = vars(ap.parse_args())


# 接受一个包含四个坐标点的列表 pts，并返回一个按照特定顺序排列的四个坐标点，以便后续使用。这通常用于矫正透视变换中，以确保坐标点按照正确的顺序排列。
def order_points(pts):
    # 创建一个形状为 (4, 2) 的零矩阵，用于存储排序后的四个坐标点。每个坐标点有两个浮点数值，因此数据类型为 float32。
    rect = np.zeros((4, 2), dtype="float32")

    # axis=1 参数指定了沿着第二个维度（即列方向）进行求和操作。这意味着函数将对每个坐标点的横纵坐标分别求和，而不是将整个数组的元素求和。
    # s 是一个包含了四个和值的Numpy数组，其中每个元素对应一个坐标点的横纵坐标之和。
    s = pts.sum(axis=1)

    # 按顺序找到对应坐标0123分别是 左上，右上，右下，左下
    rect[0] = pts[np.argmin(s)]   # 左上
    rect[2] = pts[np.argmax(s)]   # 右下

    # diff 是一个包含了四个差值的Numpy数组，其中每个元素对应一个坐标点的横纵坐标之差。右上角点的横纵坐标之差最小，而左下角点的横纵坐标之差最大
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # 右上
    rect[3] = pts[np.argmax(diff)]   # 左下
    return rect


# 用于执行透视变换（Perspective Transformation）来校正图像中的透视畸变，使得指定的四个坐标点（pts）变成一个矩形。
def four_point_transform(image, pts):
    # 坐标点按照 左上，右上，右下，左下 的顺序排列
    rect = order_points(pts)
    (tl, tr, br, bl) = rect  # t-top, b-bottom, l-left, r-right

    # 计算输入的w和h值
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 变换后对应坐标位置
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # 计算变换矩阵
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # 返回变换后结果
    return warped


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


# 读取输入
image = cv2.imread(args["image"])
# 坐标也会相同变化
ratio = image.shape[0] / 500.0
orig = image.copy()

image = resize(orig, height=500)

# 预处理
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

# 展示预处理结果
print("STEP 1: 边缘检测")
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 轮廓检测
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

# 遍历轮廓
for c in cnts:
    # 计算轮廓近似
    peri = cv2.arcLength(c, True)
    # C表示输入的点集
    # epsilon表示从原始轮廓到近似轮廓的最大距离，它是一个准确度参数
    # True表示封闭的
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # 4个点的时候就拿出来
    if len(approx) == 4:
        screenCnt = approx
        break

# 展示结果
print("STEP 2: 获取轮廓")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 透视变换
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

# 二值处理
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(warped, 100, 255, cv2.THRESH_BINARY)[1]
cv2.imwrite('scan.jpg', ref)
# 展示结果
print("STEP 3: 变换")
cv2.imshow("Original", resize(orig, height=650))
cv2.imshow("Scanned", resize(ref, height=650))
cv2.waitKey(0)
