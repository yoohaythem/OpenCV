# 导入工具包
import numpy as np
import argparse
import imutils
import cv2

# 创建一个参数解析器对象。
ap = argparse.ArgumentParser()
# 定义了一个命令行参数。在这个示例中，参数名是-i和--image，它们是参数的两种不同的表示方式。
# required=True表示这个参数是必需的，用户必须提供它。help参数用于提供关于这个参数的描述信息。
ap.add_argument("-i", "--image", required=True, help="path to the input image")
# ap.parse_args()执行实际的解析工作，并返回一个包含解析结果的命名空间对象。
# vars()函数将这个命名空间对象转换为一个字典，使您可以轻松地访问各个参数的值。
# 解析命令行参数并将它们存储在args变量中。
args = vars(ap.parse_args())

# 正确答案
ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}


def order_points(pts):
    # 创建一个包含四个坐标点的数组，每个坐标点有两个浮点数值（x 和 y 坐标）。
    rect = np.zeros((4, 2), dtype="float32")

    # 按顺序找到对应坐标0123分别是 左上，右上，右下，左下
    # ---python教学：---
    # pts.sum(axis=1) 是一个NumPy数组操作，它对 pts 中的每一行进行求和，因为 axis=1 表示按行进行求和。
    # 假设 pts 是一个形状为 (n, 2) 的NumPy数组，其中 n 是坐标点的数量，每个坐标点由两个浮点数值（x 和 y 坐标）组成。
    # 当执行 pts.sum(axis=1) 时，它将返回一个包含 n 个元素的一维数组，每个元素是对应行的元素的和。
    #   例如，如果 pts 的内容如下：
    #   pts = np.array([[1.0, 2.0],
    #                   [3.0, 4.0],
    #                   [5.0, 6.0]])
    #   那么 pts.sum(axis=1) 将返回一个包含三个元素的一维数组：
    #   array([3.0, 7.0, 11.0])
    s = pts.sum(axis=1)
    # 找到 s 中最小值对应的索引，即左上角点的索引，将其存储在 rect[0] 中。
    rect[0] = pts[np.argmin(s)]
    # 找到 s 中最大值对应的索引，即右下角点的索引，将其存储在 rect[2] 中。
    rect[2] = pts[np.argmax(s)]

    # 计算每个坐标点的 x 和 y 坐标之差，将结果存储在 diff 中。
    diff = np.diff(pts, axis=1)
    # 找到 diff 中最小值对应的索引，即右上角点的索引，将其存储在 rect[1] 中。
    rect[1] = pts[np.argmin(diff)]
    # 找到 diff 中最大值对应的索引，即左下角点的索引，将其存储在 rect[3] 中。
    rect[3] = pts[np.argmax(diff)]

    # 返回一个包含四个已排序坐标点的数组 rect，这些坐标点分别代表了左上、右上、右下和左下四个点。
    return rect


# 用于执行透视变换（Perspective Transformation）来校正图像中的透视畸变，使得指定的四个坐标点（pts）变成一个矩形。
def four_point_transform(image, pts):
    # 坐标点按照 左上，右上，右下，左下 的顺序排列
    rect = order_points(pts)
    (tl, tr, br, bl) = rect  # t-top, b-bottom, l-left, r-right

    # 计算输入的w和h值
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))  # 底部两个点之间的几何举例
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))  # 顶部两个点之间的几何举例
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))  # 右部两个点之间的几何举例
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))  # 左部两个点之间的几何举例
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
    M = cv2.getPerspectiveTransform(rect, dst)  # rect--原坐标点位置，dst--目标坐标位置

    # 用透视变换矩阵 M 将输入图像 image 进行透视变换，并将结果保存在 warped 变量中。以下是对该行代码的详细解释：
    #     image 是输入的图像，这是需要进行透视变换的原始图像。
    #     M 是之前使用 cv2.getPerspectiveTransform 函数计算得到的透视变换矩阵，它描述了如何将输入图像中的内容进行透视变换。
    #     (maxWidth, maxHeight) 是一个元组，表示输出图像的宽度和高度，即变换后的图像应该具有的尺寸。
    # 结果保存在 warped 变量中。变换后的图像将具有指定的宽度和高度 (maxWidth, maxHeight)。
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # 返回变换后结果
    return warped


# 根据轮廓外接矩形的顶点坐标位置顺序，对轮廓进行排序
def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0

    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # 通过列表推导式，对 cnts 列表中的每个轮廓 c，使用 cv2.boundingRect 函数来获取其边界框（外接矩形）的信息。
    # cv2.boundingRect 函数返回一个包含四个值的元组 (x, y, w, h)，分别表示边界框的左上角坐标 (x, y) 和宽度 w 以及高度 h。
    # 这个操作将生成一个包含所有轮廓边界框信息的列表 boundingBoxes。
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]  # 外接矩形：用一个最小的矩形，把找到的形状包起来x,y,h,w

    # 对轮廓和对应的边界框信息进行排序，并将排序后的结果重新分配给 cnts 和 boundingBoxes。
    # zip(cnts, boundingBoxes)：这一部分将 cnts 和 boundingBoxes 列表中的元素一一配对，每个元素都是一个包含轮廓和边界框信息的元组。
    # sorted(..., key=lambda x: x[1][0], reverse=reverse)：这一部分使用 sorted 函数对这些元组进行排序，排序的依据是每个元组中的第二个元素 x[1] 的第一个值 x[1][0]，即边界框的左上角 x 坐标。
    # ---python教学---：lambda后面的参数代指被排序的每个元素，这里就是 zip(cnts, boundingBoxes)里的每个元素，x[1]即 boundingBoxes里的每个元素
    #                  reverse 参数用于控制排序的升序或降序，如果设置为 True，则是降序（从大到小）。
    #                  最后，zip(*sorted(...)) 将排序后的元组再次分开，分别赋值给 cnts 和 boundingBoxes，从而得到排序后的轮廓列表 cnts 和边界框列表 boundingBoxes。

    # ** 整个过程就是 (a3,a4,a2,a1),(b3,b4,b2,b1) 的两个元祖 ---[zip拉链]--->
    #    ((a3,b3),(a4,b4),(a2,b2),(a1,b1)) 的一个迭代器  ---[根据b排序,即x[1]]--->
    #    [(a1,b1),(a2,b2),(a3,b3),(a4,b4)] 的一个列表  ---[*解压缩]--->
    #    (a1,b1),(a2,b2),(a3,b3),(a4,b4)  的四个元祖  ---[zip拉链]--->
    #    (a1,a2,a3,a4),(b1,b2,b3,b4) 的两个元祖，但此时 (a1,a2,a3,a4)，(b1,b2,b3,b4) 已经排序完毕
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda x: x[1][i], reverse=reverse))

    return cnts, boundingBoxes


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 预处理
# 读取图像
image = cv2.imread(args["image"])
contours_img = image.copy()
# 将彩色图像转换为灰度图像，结果存储在 gray 变量中。
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 对灰度图像进行高斯模糊处理，这有助于去除图像中的噪点。 (5, 5) 是高斯核的大小，而 0 是标准差（控制模糊程度）。
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
cv_show('blurred', blurred)
# 使用Canny边缘检测算法检测图像的边缘，边缘采用白色保存。
# blurred 是输入图像，(75, 200) 是Canny算法的两个阈值参数，用于控制边缘检测的敏感度。边缘图像存储在 edged 变量中。
edged = cv2.Canny(blurred, 75, 200)
cv_show('edged', edged)

# 轮廓检测
# 用于查找输入的二值化图像 edged 中的轮廓。
# 参数 cv2.RETR_EXTERNAL 意味着仅提取最外层（外部）的轮廓，
# cv2.CHAIN_APPROX_SIMPLE 表示对轮廓进行适当的逼近，以减少轮廓点的数量，从而节省内存。
# 返回值中的 cnts 是一个包含轮廓信息的列表。每个轮廓都是一组点的坐标。
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
# 在原始图像 contours_img 上绘制轮廓。具体参数如下：
#   contours_img 是目标绘图的图像。
#   cnts 是轮廓列表。
#   -1 表示绘制所有轮廓。
#   (0, 0, 255) 是绘制轮廓的颜色，这里是红色 (BGR 格式)。
#   3 是绘制线条的宽度。
cv2.drawContours(contours_img, cnts, -1, (0, 0, 255), 3)
cv_show('contours_img', contours_img)
docCnt = None

# 确保检测到了
if len(cnts) > 0:
    # 根据轮廓大小进行排序，按降序排列，即面积最大的轮廓排在列表的前面。
    # ===Python教学===：sorted函数的key参数为一个函数，用于对被排序对象的每个值进行运算，得出结果，并按一定顺序排列。
    #                  常见的函数多是匿名lambda函数，在这里采用了计算轮廓面积的cv2.contourArea函数。
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # 遍历每一个轮廓
    for c in cnts:
        # 计算轮廓 c 的周长（弧长），并将结果存储在 peri 变量中。
        # 函数的第一个参数 c 是输入的轮廓，第二个参数 True 表示轮廓是闭合的（即考虑轮廓的最后一段与第一段相连）
        # 如果轮廓不是闭合的，可以将第二个参数设置为 False。
        peri = cv2.arcLength(c, True)
        # 用于对轮廓 c 进行多边形逼近，以减少多边形的顶点数目。该函数的第一个参数是输入的轮廓。
        # 第二个参数是逼近的精度，这里使用 0.02 * peri，其中 peri 是轮廓的周长，表示逼近的精度为周长的2%。
        # 第三个参数 True 表示多边形是闭合的，如果多边形不是闭合的，可以将第三个参数设置为 False。
        # 逼近后的多边形的顶点坐标存储在 approx 变量中。
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # 当遍历到轮廓为四个顶点的时候，跳出循环。
        if len(approx) == 4:
            docCnt = approx
            break

# 执行透视变换，来校正图像中的透视畸变，使得指定的四个坐标点（pts）变成一个矩形。
warped = four_point_transform(gray, docCnt.reshape(4, 2))
cv_show('warped', warped)
# 使用 cv2.threshold 函数进行二值化处理，采用自动二值确定（OTSU）方法。具体解释如下：
#   warped 是要进行二值化处理的图像。
#   0 是用于计算二值的初始值。
#   255 是二值化后的最大值，即二值化后的前景值。
#   cv2.THRESH_BINARY | cv2.THRESH_OTSU 表示使用二进制二值化方法，并采用OTSU自动确定阈值，适合双峰。
thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cv_show('thresh', thresh)

# 复制二值化后的图像为 thresh_Contours
thresh_Contours = thresh.copy()
# 查找二值图像 thresh 中的轮廓。
#   thresh：这是输入的二值图像，也就是要查找轮廓的图像。
#   cv2.RETR_EXTERNAL：这个参数表示轮廓的检索模式。cv2.RETR_EXTERNAL 指定只检测最外层的轮廓，不检测轮廓内部的嵌套轮廓。
#   cv2.CHAIN_APPROX_SIMPLE：这是轮廓的逼近方法。cv2.CHAIN_APPROX_SIMPLE 会对轮廓进行简化，只存储轮廓的端点坐标，节省内存。这个方法通常用于轮廓检测。
#   cv2.findContours() 函数的返回值包括两个元素，第一个是包含所有轮廓的列表，第二个是轮廓的层次结构信息。在这行代码中，通过 [1] 获取了第二个元素，即轮廓的列表。
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
# 在图像 thresh_Contours 上绘制轮廓线的宽度为3个像素，颜色红色的所有轮廓。
cv2.drawContours(thresh_Contours, cnts, -1, (0, 0, 255), 3)
cv_show('thresh_Contours', thresh_Contours)

questionCnts = []

# 遍历
for c in cnts:
    # 计算圆形轮廓的边界框的坐标
    (x, y, w, h) = cv2.boundingRect(c)
    # 计算边界框的宽高比例 ar
    ar = w / float(h)

    # 边界框宽度和高度都大于等于20像素，并且宽高比例在0.9到1.1之间。如果满足这些条件，将该轮廓添加到 questionCnts 列表中。
    # # 根据实际情况指定标准
    if w >= 20 and h >= 20 and 0.9 <= ar <= 1.1:
        questionCnts.append(c)

# 按照从上到下进行排序
questionCnts = sort_contours(questionCnts, method="top-to-bottom")[0]

correct = 0

# q 是循环索引，用于表示题目的序号。
# i 是题目区域的索引，每次循环处理5个答题框的轮廓。
for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
    # 首先对每一组5个答题框的轮廓进行排序，按照外接矩形定点坐标从左到右排序
    cnts = sort_contours(questionCnts[i:i + 5])[0]
    bubbled = None

    # 遍历每一排答题框的5个轮廓
    for (j, c) in enumerate(cnts):
        # 创建一个和答题卡一样大区域的二进制掩码 mask 用于判断答案区域，开始全为黑色
        mask = np.zeros(thresh.shape, dtype="uint8")
        # [c]：包含一个或多个轮廓的列表。在这里，我们只绘制单个轮廓（每个循环只有一个），所以将其放入列表中。
        # -1：轮廓的索引。在这里，-1表示绘制所有的轮廓。
        # 255：绘制的颜色，这里表示白色。
        # -1：轮廓内部的填充。-1表示轮廓内部会被填充成白色，而不是只绘制轮廓线。
        cv2.drawContours(mask, [c], -1, 255, -1)  # -1表示填充
        cv_show('mask', mask)

        # 通过按位与运算 cv2.bitwise_and 将掩码应用于二值化图像 thresh。
        # 使得当前遍历到的轮廓以外的区域全部变为黑色。
        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        # 计算答案区域中非零像素点的数量，通过计算非零点数量来算是否选择这个答案
        total = cv2.countNonZero(mask)

        # 如果当前没有存储答案，或者当前存储的答案区域中非零像素点的数量没有本轮循环中的高
        # 因为前面做了边缘检测，除了边缘都是黑色；所以铅笔涂了的答案，边缘更多，即非零像素点肯定最高
        if bubbled is None or total > bubbled[0]:
            # total：答案区域中非零像素点的数量
            # j：存储的是第j个答案
            bubbled = (total, j)

    # 对比正确答案
    # 为每个题目的答案选项分配颜色（绿色表示正确，红色表示错误）
    color = (0, 0, 255)
    k = ANSWER_KEY[q]  # 从字典里取出第一轮的正确答案

    # 判断正确
    if k == bubbled[1]:  # 正确答案 == 涂黑的答案
        color = (0, 255, 0)
        correct += 1

    # 绘制已识别的答案轮廓，以突出显示正确答案的区域。
    cv2.drawContours(warped, [cnts[k]], -1, color, 3)

# 计算得分并在图像上显示
score = (correct / 5.0) * 100
print("[INFO] score: {:.2f}%".format(score))
cv2.putText(warped, "{:.2f}%".format(score), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
cv2.imshow("Original", image)
cv2.imshow("Exam", warped)
cv2.waitKey(0)
