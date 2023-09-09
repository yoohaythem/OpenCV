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
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))   # 底部两个点之间的几何举例
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))   # 顶部两个点之间的几何举例
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))   # 右部两个点之间的几何举例
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))   # 左部两个点之间的几何举例
    maxHeight = max(int(heightA), int(heightB))

    # 变换后对应坐标位置，指定目标点
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # cv2.getPerspectiveTransform函数，用于进行透视变换的矩阵计算。
    # rect 和 dst 是作为函数参数传递进去的两个参数：
    #     rect 通常是一个包含四个点坐标的列表或数组，这些点表示输入图像中的一个矩形区域的四个角点。
    #     dst 也是一个包含四个点坐标的列表或数组，这些点表示输出图像中的矩形区域的四个角点。
    #     计算出一个透视变换矩阵 M：这个矩阵可以用于将输入图像中的矩形区域进行透视变换，使其变换为输出图像中对应的矩形区域。
    # 这个透视变换矩阵可以用于将输入图像中的某个区域（如一个矩形）校正为正面视图，或者进行图像的透视矫正操作，可以将这个矩阵应用于图像，以实现相应的透视变换效果。
    M = cv2.getPerspectiveTransform(rect, dst)    # rect--原坐标点位置，dst--目标坐标位置


    # 用透视变换矩阵 M 将输入图像 image 进行透视变换，并将结果保存在 warped 变量中。以下是对该行代码的详细解释：
    #     image 是输入的图像，这是需要进行透视变换的原始图像。
    #     M 是之前使用 cv2.getPerspectiveTransform 函数计算得到的透视变换矩阵，它描述了如何将输入图像中的内容进行透视变换。
    #     (maxWidth, maxHeight) 是一个元组，表示输出图像的宽度和高度，即变换后的图像应该具有的尺寸。
    # 结果保存在 warped 变量中。变换后的图像将具有指定的宽度和高度 (maxWidth, maxHeight)。
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # 返回变换后结果
    return warped


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]   # 图像实际尺寸
    # 目标宽高均缺失，缺少参数，则直接返回
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)  # 目标高和实际高的比值
        # dim 是一个元组，包含两个值，表示输出图像的宽度和高度。这个元组定义了目标图像的尺寸。
        dim = (int(w * r), height)   # 根据高的比值，等比推算目标宽度
    else:   # 与上面一样
        r = width / float(w)
        dim = (width, int(h * r))
    
    # interpolation 是一个参数，用于指定图像缩放时所使用的插值方法。它可以是以下之一：
    #     cv2.INTER_NEAREST：最近邻插值，使用最接近的像素值。
    #     cv2.INTER_LINEAR：线性插值，使用相邻像素的加权平均值。
    #     cv2.INTER_CUBIC：立方插值，使用相邻像素的三次插值。
    #     cv2.INTER_AREA：区域插值，适合缩小图像。
    # 调整大小后的图像被存储在 resized 变量中。
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


# 3. 读取输入
image = cv2.imread(args["image"])
# 坐标也会相同变化
ratio = image.shape[0] / 500.0   # 原始大小与目标大小之间的比例
orig = image.copy()   # 原图拷贝
# 将原始图像等比例变为高度为500的图像。
image = resize(orig, height=500)

# 4. 预处理
# 将输入图像 image 从彩色（BGR格式）转换为灰度图像，只包括亮度信息，不包括图像信息。
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 对灰度图像 gray 进行高斯模糊处理。
# 高斯模糊是一种用于降噪的滤波技术，它通过对每个像素周围的像素进行加权平均来模糊图像。这里使用了一个5x5的高斯内核，内核大小决定了模糊程度。
# 参数 0 表示从内核的大小自动计算高斯标准差。
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# 对高斯模糊后的灰度图像 gray 进行边缘检测，使用了Canny边缘检测算法。
# 第一个参数是输入图像，这里是经过高斯模糊后的灰度图像 gray。
# 第二个参数 75 是低阈值，用于边缘像素的初步检测。像素值低于低阈值的边缘将被丢弃。
# 第三个参数 200 是高阈值，用于强边缘的检测。像素值高于高阈值的边缘被认为是强边缘。75-200之间的边缘需要与强边缘连接才保存，否则丢弃。
# 函数将检测到的边缘存储在 edged 变量中。
edged = cv2.Canny(gray, 75, 200)

# 展示预处理结果
print("STEP 1: 边缘检测")
cv2.imshow("Image", image)   # 原图
cv2.imshow("Edged", edged)   # 边缘检测后的图像
cv2.waitKey(0)
cv2.destroyAllWindows()

# 5. 轮廓检测
# 使用OpenCV库中的 cv2.findContours 函数，在边缘图像 edged 上查找图像中的轮廓。以下是对该行代码的详细解释：
#     edged.copy() 是作为函数的第一个参数，表示要查找轮廓的输入图像。.copy() 函数用于创建 edged 图像的副本，以确保原始图像不会被修改。
#     cv2.RETR_LIST 是作为函数的第二个参数，表示轮廓检测模式。cv2.RETR_LIST 表示检测所有轮廓，不建立轮廓的等级关系。
#     cv2.CHAIN_APPROX_SIMPLE 是作为函数的第三个参数，表示轮廓的逼近方法。cv2.CHAIN_APPROX_SIMPLE 会压缩水平、垂直和对角方向上的冗余点，从而节省轮廓的内存。
# 函数的返回值是一个包含检测到的轮廓的列表。由于 findContours 返回多个值，这里使用了 [0] 来获取第一个返回值，即轮廓列表。
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

# 通过对轮廓列表 cnts 按照轮廓的面积从大到小排序，并选择前五个最大的轮廓。以下是对该行代码的详细解释：
#     cnts 是要排序的轮廓列表。
#     cv2.contourArea 函数用于计算轮廓的面积， key=cv2.contourArea 使得 sorted 函数按照每个轮廓的面积来排序。
#     reverse=True 表示降序排列，即按照轮廓的面积从大到小排列。
# [:5] 表示选择排序后的前五个轮廓，这意味着只保留前五个最大的轮廓。
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

# 6. 遍历轮廓
for c in cnts:
    # 计算给定轮廓 c 的弧长（周长）。
    # 第二个参数 True 表示轮廓是否封闭。如果设置为 True，则函数会计算封闭轮廓的弧长，即轮廓的周长。如果设置为 False，则函数将计算轮廓的线段长度，不考虑轮廓是否封闭。
    peri = cv2.arcLength(c, True)

    # 使用了OpenCV库中的 cv2.approxPolyDP 函数，对给定的轮廓 c 进行多边形逼近。以下是对该行代码的详细解释：
    #     c 是一个由 cv2.findContours 返回的轮廓对象。
    #     第二个参数表示从原始轮廓到近似轮廓的最大距离，它是一个准确度参数。0.02 是一个对周长的缩放因子，用于控制逼近的精度，具体数值可以根据需要调整。
    #     第三个参数 True 表示逼近的多边形应该是封闭的。这意味着逼近后的多边形应该首尾相连，形成一个封闭的多边形。
    # approx 是返回值，它包含了对输入轮廓进行多边形逼近后得到的多边形的顶点坐标。
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # 当该轮循环中，轮廓进行多边形逼近后得到的多边形的顶点坐标恰好为4个点的时候，break掉循环，把screenCnt拿出来，保证是个四边形
    if len(approx) == 4:
        screenCnt = approx
        break

# 展示结果
print("STEP 2: 获取轮廓")
# 绘制轮廓
# [screenCnt] 是一个包含要绘制的轮廓列表的列表。通常，你可以将要绘制的轮廓存储在一个列表中，然后将该列表作为参数传递给函数。
# 在这里根据上文，[screenCnt] 表示要绘制的轮廓列表仅包含一个轮廓 screenCnt。
# -1 是要绘制的轮廓的索引。如果为负数（如 -1），则表示绘制所有的轮廓。
# (0, 255, 0) 是绘制轮廓的颜色。这里使用的是BGR颜色表示法，(0, 255, 0) 表示绘制轮廓的颜色为绿色。
# 2 是轮廓的线宽，表示轮廓的线条宽度。
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 7. 透视变换
# 将输入图像 orig 中的一个矩形区域校正为一个新的视角。以下是对该行代码的详细解释：
#     orig 是输入的原始图像（彩图），即包含待处理矩形区域的图像。
#     screenCnt 是一个包含矩形区域的四个角点坐标的轮廓，这些坐标被假设为源点坐标。
#     reshape(4, 2) 将 screenCnt 的形状从一维数组变为一个包含四个点的二维数组，每个点由 x 和 y 坐标组成。
#     ratio = image.shape[0] / 500.0   原始大小与目标大小之间的比例，用于将源点坐标缩放到目标图像中的正确尺寸。
# 最终的结果被保存在 warped 变量中，表示经过透视变换后的图像。
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

# 8. 二值处理
# 将经过透视变换后的图像 warped 从彩色（BGR格式）转换为灰度图像。
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)  
# 对灰度图像 warped 进行阈值化处理，将图像二值化（转换为只包含两个值的图像，一般是黑和白）。
#     100 是用作阈值的像素值，所有大于等于阈值的像素将被置为白色（255），而小于阈值的像素将被置为黑色（0）。
#     cv2.THRESH_BINARY 是阈值化的方法，表示大于阈值的像素被设为最大值（白色），小于阈值的像素被设为0（黑色）。
#     [1] 表示从 cv2.threshold 函数的返回值中获取阈值化后的图像。
ref = cv2.threshold(warped, 100, 255, cv2.THRESH_BINARY)[1]
# 保存扫描后的图像，命名为'scan.jpg'
cv2.imwrite('scan.jpg', ref)

# 展示结果
print("STEP 3: 变换")
cv2.imshow("Original", resize(orig, height=650))   # 原始图像
cv2.imshow("Scanned", resize(ref, height=650))    # 经过透视变换后的图像
cv2.waitKey(0)
