# 1. 导入工具包
from imutils import contours
import numpy as np
import argparse
import cv2
import myutils

# 2. 设置参数
# argparse.ArgumentParser()：这是创建一个参数解析器的步骤。创建了一个名为ap的参数解析器对象。
# ap.add_argument("-i", "--image", required=True, help="path to input image")：
# 这行代码添加了一个命令行参数--image，它有一个短选项-i，并且要求必须提供这个参数。help参数用于提供关于参数的描述，以便用户知道如何使用它。
# ap.add_argument("-t", "--template", required=True, help="path to template OCR-A image")：
# 类似于上面的代码，这行代码添加了另一个命令行参数--template，它有一个短选项-t，同样要求必须提供这个参数，并提供了关于参数的描述。
# args = vars(ap.parse_args())：
# 这行代码解析了命令行参数，并将它们存储在一个字典args中。字典的键是参数名称，值是用户在命令行上提供的参数值。
# 当运行这个脚本时，可以使用--image和--template参数来指定输入图像和模板图像的路径，然后在脚本中可以通过args["image"]和args["template"]来获取这些路径。
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-t", "--template", required=True, help="path to template OCR-A image")
args = vars(ap.parse_args())

# 3. 指定信用卡类型
FIRST_NUMBER = {
    "3": "American Express",
    "4": "Visa",
    "5": "MasterCard",
    "6": "Discover Card"
}


# 4. 绘图展示函数
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 5. 读取一个模板图像
img = cv2.imread(args["template"])
cv_show('img', img)

# 6. 模板图像转换灰度图
# cv2.cvtColor 函数用于在OpenCV中进行颜色空间转换，将一个图像从一个颜色空间转换为另一个颜色空间。以下是该函数的参数及其意义：
#   src: 要进行颜色空间转换的输入图像。这是希望转换颜色空间的原始图像。
#   code: 转换颜色空间的标志或代码。这个参数确定了要执行的颜色空间转换操作。可以选择以下常见的代码之一：
#   cv2.COLOR_BGR2GRAY: 将图像从BGR（Blue-Green-Red）颜色空间转换为灰度（单通道）颜色空间。这将将彩色图像转换为灰度图像，其中每个像素只有一个灰度值，表示亮度。
#                       还有其他许多颜色空间转换代码，用于将图像从一种颜色空间转换为另一种，例如RGB到HSV、RGB到Lab等等。
#   dst: 可选参数，表示输出图像。如果提供了这个参数，函数将结果存储在这个图像中。通常，可以将这个参数设置为None，以便函数创建一个新的输出图像。
ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv_show('ref', ref)

# 7. 灰度图转二值图像
# 这行代码使用了 OpenCV 中的 cv2.threshold 函数来进行图像二值化处理。以下是该函数的参数及其意义：
#   src: 输入图像，这是要进行二值化处理的原始图像。
#   thresh: 阈值，用于将图像的像素值分成两个类别。在这个示例中，阈值为10，这意味着小于等于10的像素值将被分到一个类别，而大于10的像素值将被分到另一个类别。
#   maxval: 用于表示高像素值的参数，通常为255。在这个示例中，大于10的像素值将被设置为255，即白色。
#   type_: 二值化的类型。它是一个用于指定二值化方法的参数，可以选择以下几种类型之一：
#       cv2.THRESH_BINARY: 大于阈值的像素值将被设置为maxval，小于等于阈值的像素值将被设置为0。
#       cv2.THRESH_BINARY_INV: 与cv2.THRESH_BINARY相反，大于阈值的像素值将被设置为0，小于等于阈值的像素值将被设置为maxval。
#       cv2.THRESH_TRUNC: 大于阈值的像素值将被截断为阈值，小于等于阈值的像素值保持不变。
#       cv2.THRESH_TOZERO: 大于阈值的像素值保持不变，小于等于阈值的像素值将被设置为0。
#       cv2.THRESH_TOZERO_INV: 与cv2.THRESH_TOZERO相反，大于阈值的像素值将被设置为0，小于等于阈值的像素值保持不变。
# 在这行代码中，使用了 cv2.THRESH_BINARY_INV 类型的二值化，它会将大于10的像素值设置为0（黑色），小于等于10的像素值设置为255（白色）。
# 这通常用于创建一个反相的二值化图像，其中对象部分是白色，背景部分是黑色。这种图像通常用于对象检测和分割任务。
# 函数的返回值包括 ret 和 thresh1，其中 ret 是阈值的值，thresh1 是二值化后的图像，所以通过[1]来获取二值化后的图像。
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]
cv_show('ref', ref)

# 8. 计算轮廓
# cv2.findContours 函数用于查找图像中的轮廓，并返回轮廓的列表以及层次信息。下面是该函数的参数及其意义：
#   image: 输入图像，通常是一个灰度图像（单通道图像）。函数会在这个图像上查找轮廓。
#   mode: 查找轮廓的模式。这个参数决定了如何查找轮廓，可以选择以下几种模式之一：
#       cv2.RETR_EXTERNAL: 只检测最外层的轮廓，不考虑轮廓的嵌套关系。
#       cv2.RETR_LIST: 检测所有轮廓，不建立轮廓之间的父子关系。
#       cv2.RETR_CCOMP: 检测所有轮廓，建立两层的轮廓层次结构。外层轮廓的父轮廓是顶层轮廓，内层轮廓的父轮廓是外层轮廓。
#       cv2.RETR_TREE: 检测所有轮廓，建立完整的轮廓层次结构。
#   method: 近似轮廓的方法。这个参数决定了如何近似轮廓的形状，可以选择以下几种方法之一：
#       cv2.CHAIN_APPROX_NONE: 存储所有的轮廓点，不进行近似。
#       cv2.CHAIN_APPROX_SIMPLE: 仅存储水平、垂直和对角线方向的端点，压缩所有其他点。这个方法通常用于减少轮廓的点数，从而节省内存。
#       cv2.CHAIN_APPROX_TC89_L1 和 cv2.CHAIN_APPROX_TC89_KCOS: 使用Teh-Chin链逼近算法的两种不同变种。
# 函数的返回值包括两个部分：
#   refCnts: 一个包含检测到的轮廓的列表。每个轮廓是一个点的集合，以列表的形式表示。
#   hierarchy: 一个包含轮廓层次信息的可选输出。它描述了轮廓之间的父子关系。这个参数在层次结构模式（例如cv2.RETR_CCOMP和cv2.RETR_TREE）中非常有用，但在其他模式下可以设置为None。
# 通常，cv2.findContours 函数用于图像分析中的对象检测和分割任务，它允许获取图像中的各种对象的轮廓，以便后续处理或可视化。
refCnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(len(refCnts))

# cv2.drawContours 函数用于在图像上绘制轮廓。下面是该函数的参数及其意义：
#   img: 要绘制轮廓的输入图像。这是一个包含轮廓的图像，轮廓通常是以白色线条在黑色背景上表示的。
#   contours: 要绘制的轮廓列表，通常是通过 cv2.findContours 函数找到的轮廓列表。contours 是一个包含多个轮廓的列表，每个轮廓都是一个点的集合。
#   contourIdx: 一个整数值，用于指定要绘制的轮廓的索引。如果设置为负数（通常为-1），则绘制所有轮廓。
#   color: 绘制轮廓的颜色，通常表示为一个BGR颜色元组。例如，(0, 0, 255) 表示红色，(0, 255, 0) 表示绿色，(255, 0, 0) 表示蓝色。
#   thickness: 绘制轮廓线条的宽度（像素数）。设置为正整数值表示轮廓线条的宽度，设置为负数或零表示填充轮廓内部。
# 该函数的主要作用是将指定的轮廓绘制到输入图像上，以便可视化轮廓或执行其他与轮廓相关的操作。通常，在图像处理中，轮廓是用于识别和分割对象的关键元素。
cv2.drawContours(img, refCnts, -1, (0, 0, 255), 3)
cv_show('img', img)

# myutils.py的第一个函数调用，对轮廓的列表从左到右进行排序
refCnts = myutils.sort_contours(refCnts, method="left-to-right")[0]  # 排序，从左到右，从上到下
digits = {}  # ---语法教学：这里是字典的语法糖。在Python里，集合是没有语法糖的，只能用set()

# 9. 生成模板与数字对应关系
# 遍历每一个轮廓
# enumerate 是 Python 内置函数，用于将一个可迭代对象（在这里是 refCnts 列表）的元素与它们的索引一起返回。
# 它返回一个迭代器，该迭代器产生包含索引和值的元组。
for (i, c) in enumerate(refCnts):  # 计算外接矩形并且resize成合适大小
    (x, y, w, h) = cv2.boundingRect(c)
    roi = ref[y:y + h, x:x + w]  # 截取轮廓画面
    roi = cv2.resize(roi, (57, 88))  # 规范化轮廓画面的大小
    digits[i] = roi  # 字典里每一个数字对应每一个模板

# 10. 初始化卷积核
# 使用了OpenCV中的cv2.getStructuringElement函数来创建两种形状的结构元素（structuring element）：
# 一个矩形结构元素（rectKernel）和一个正方形结构元素（sqKernel）。
# 结构元素通常用于形态学图像处理操作，如腐蚀（erosion）和膨胀（dilation）。
# cv2.MORPH_RECT：这是指定结构元素的形状为矩形（rectangle）的标志。
# (9, 3)：这是指定矩形结构元素的大小的元组，其中第一个值（9）表示矩形的宽度，第二个值（3）表示矩形的高度。这将创建一个宽度为9个像素、高度为3个像素的矩形结构元素。
# [[1 1 1 1 1 1 1 1 1], [1 1 1 1 1 1 1 1 1], [1 1 1 1 1 1 1 1 1]]
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
# (5, 5)：这是指定正方形结构元素的大小的元组，其中第一个值（5）表示正方形的宽度，第二个值（5）表示正方形的高度。这将创建一个5x5像素大小的正方形结构元素。
# [[1 1 1 1 1], [1 1 1 1 1], [1 1 1 1 1], [1 1 1 1 1], [1 1 1 1 1]]
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# 11. 读取输入图像，转换灰度图
image = cv2.imread(args["image"])
cv_show('image', image)
image = myutils.resize(image, width=300)
# cv2.cvtColor 函数用于在OpenCV中进行颜色空间转换，将一个图像从一个颜色空间转换为另一个颜色空间。以下是该函数的参数及其意义：
#   src: 要进行颜色空间转换的输入图像。这是希望转换颜色空间的原始图像。
#   code: 转换颜色空间的标志或代码。这个参数确定了要执行的颜色空间转换操作。可以选择以下常见的代码之一：
#   cv2.COLOR_BGR2GRAY: 将图像从BGR（Blue-Green-Red）颜色空间转换为灰度（单通道）颜色空间。这将将彩色图像转换为灰度图像，其中每个像素只有一个灰度值，表示亮度。
#                       还有其他许多颜色空间转换代码，用于将图像从一种颜色空间转换为另一种，例如RGB到HSV、RGB到Lab等等。
#   dst: 可选参数，表示输出图像。如果提供了这个参数，函数将结果存储在这个图像中。通常，可以将这个参数设置为None，以便函数创建一个新的输出图像。
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv_show('gray', gray)

# 12. 礼帽操作，突出更明亮的区域
# OpenCV 中的 cv2.morphologyEx 函数，将顶帽操作（Top Hat）应用于灰度图像 gray，并使用之前定义的 rectKernel 结构元素。以下是这行代码的各个部分的解释：
#   gray: 这是输入的灰度图像，是要对其应用顶帽操作的图像。
#   cv2.MORPH_TOPHAT: 这是形态学操作的标志，表示进行顶帽操作。顶帽操作是形态学图像处理中的一种操作，用于检测图像中的亮对象或结构。
#                     它通常用于从图像中分离出比周围区域亮的小对象或细节。
#   rectKernel: 这是之前定义的矩形结构元素，用于指定顶帽操作的内核形状。矩形结构元素是一个矩形形状的窗口，用于定义形态学操作的操作区域和大小。
# 顶帽操作：原始输入-开运算（先腐蚀后膨胀）。这种操作在图像处理中常用于改善图像的质量、增强细节、以及检测和分割亮对象或结构。
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
cv_show('tophat', tophat)

# 13. 使用 Sobel 滤波器计算 tophat 图像的水平方向梯度（这里没做y方向的）。具体解释如下：
#   tophat 是之前进行顶帽操作得到的图像。
#   ddepth=cv2.CV_32F 指定输出图像的数据类型为 32 位浮点数。
#   dx=1 表示计算水平方向的梯度。
#   dy=0 表示不计算垂直方向的梯度。
#   ksize=-1 表示使用默认的 Sobel 内核大小，通常是 3x3。
gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
# 将梯度图像中的所有像素值转换为绝对值，以确保所有梯度都是正数。
gradX = np.absolute(gradX)  # gradX = cv2.convertScaleAbs(gradX)
# 计算梯度图像中的最小值和最大值。
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
# 对梯度图像进行归一化，将像素值映射到 0 到 255 的范围，以便可视化。归一化操作将确保梯度的最小值映射到 0，最大值映射到 255。
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
# 将归一化后的梯度图像转换为 8 位无符号整数类型，以便显示和存储。
gradX = gradX.astype("uint8")
# 打印 gradX 的形状，即图像的高度和宽度。
print(np.array(gradX).shape)
cv_show('gradX', gradX)

# 14. 通过闭操作（先膨胀，再腐蚀）将数字连在一起
# gradX 是之前计算的梯度图像，rectKernel 是矩形结构元素，用于定义形态学操作的内核。
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
cv_show('gradX', gradX)
# 使用 cv2.threshold 函数进行二值化处理，采用自动二值确定（OTSU）方法。具体解释如下：
#   gradX 是要进行二值化处理的图像。
#   0 是用于计算二值的初始值。
#   255 是二值化后的最大值，即二值化后的前景值。
#   cv2.THRESH_BINARY | cv2.THRESH_OTSU 表示使用二进制二值化方法，并采用OTSU自动确定阈值，适合双峰。
thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv_show('thresh', thresh)
# 再来一个闭操作
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
cv_show('thresh', thresh)

# 15. 计算轮廓（和第8点一样，参考上面）
threshCnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = threshCnts
cur_img = image.copy()
# 在原始图像上，绘制轮廓
cv2.drawContours(cur_img, cnts, -1, (0, 0, 255), 3)
cv_show('img', cur_img)
locs = []

# 16. 遍历轮廓，找到合适大小的轮廓
for (i, c) in enumerate(cnts):  # enumerate参考第9点
    # 计算轮廓外接矩形
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)
    # 选择合适的区域，根据实际任务来，这里的基本都是四个数字一组
    if 2.5 < ar < 4.0:
        if 40 < w < 55 and 10 < h < 20:
            # 符合的留下来
            locs.append((x, y, w, h))

# 17. 将符合的轮廓从左到右排序
locs = sorted(locs, key=lambda x: x[0])  # x[0]：轮廓的左上角 x 坐标
output = []  # 里面由一个个的groupOutput组成

# 18. 遍历每一个轮廓中的数字
for (i, (gX, gY, gW, gH)) in enumerate(locs):
    # 初始化数字groupOutput，里面由一个个数字组成
    groupOutput = []

    # 根据坐标提取每一个组
    group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]  # 留 5 个像素预留值
    cv_show('group', group)
    # 二值化处理，采用OTSU自动确定阈值
    group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv_show('group', group)
    # 计算每一组的四个数字的轮廓（和第8点一样，参考上面）
    digitCnts, hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]

    # 计算每一组中的每一个数值
    for c in digitCnts:
        # 找到当前数值的轮廓，resize成合适的的大小
        (x, y, w, h) = cv2.boundingRect(c)
        roi = group[y:y + h, x:x + w]
        roi = cv2.resize(roi, (57, 88))  # 与模板保持大小一致
        cv_show('roi', roi)

        # 计算匹配得分
        scores = []

        # 在模板中计算每一个得分
        for (digit, digitROI) in digits.items():  # digits[i] = roi   字典里每一个数字对应每一个模板
            # 对于每个数字模板 digitROI，执行以下操作：
            # 使用 cv2.matchTemplate 函数进行模板匹配，将 digitROI 与 roi（感兴趣区域）进行匹配。产生一个匹配结果图像 result。
            #   roi: 这是感兴趣区域（Region of Interest），也就是要在其中寻找数字模板的图像区域。
            #   digitROI: 这是待匹配的数字模板图像，通常是一个小的数字图像，你希望在感兴趣区域中找到它的位置。
            #   cv2.TM_CCOEFF: 这是匹配方法的标志，表示使用相关系数匹配方法（Template Matching with Correlation Coefficient）。
            #                  相关系数匹配方法用于度量模板与图像中不同位置之间的相似度。在这个方法中，匹配得分越高，表示匹配越好。
            result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
            # 使用 cv2.minMaxLoc 函数获取 result 中的最大匹配分数 score。
            # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res) ：返回模板匹配后最小值、最大值的位置
            #   _: 这是一个占位符，用于接收不感兴趣的返回值，cv2.minMaxLoc返回的是一个元组，包含了多个值，但在这里我们只关心匹配得分和其位置。
            #   score: 这是匹配结果图像中的最大匹配得分。它表示在感兴趣区域中找到的与数字模板的最佳匹配程度。匹配得分越高，表示匹配越好。
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)

        # 得到得分最高的数字，np.argmax(scores)可以在列表中返回最大数字
        groupOutput.append(str(np.argmax(scores)))

    # 画出来
    # image, top_left, bottom_right, color, thickness
    cv2.rectangle(image, (gX - 5, gY - 5), (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
    # image: 这是原始的输入图像，即要在其上显示检测到的数字的图像。
    # "".join(groupOutput): 这是要显示的文本字符串。groupOutput 是一个包含检测到的数字的列表，通过 "".join() 方法将列表中的数字合并为一个字符串，以便显示在图像上。
    # (gX, gY - 15): 这是文本的放置位置，表示文本的左上角坐标。(gX, gY - 15) 通常用于将文本放置在检测到的数字区域的上方，以避免文本与数字重叠。gX 和 gY 是检测到数字区域的坐标。
    # cv2.FONT_HERSHEY_SIMPLEX: 这是用于绘制文本的字体类型，具体为 OpenCV 中的字体常量之一，用于指定字体样式。
    # 0.65: 这是文本的字体大小，表示文本的相对大小。0.65 是字体的比例因子，可以根据需要进行调整。
    # (0, 0, 255): 这是文本的颜色，以BGR格式指定。在这里，文本颜色为红色，因为(0, 0, 255) 表示红色。
    # 2: 这是文本的线宽，表示文本的轮廓线宽度。在这里，线宽为2。
    cv2.putText(image, "".join(groupOutput), (gX, gY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    # 得到groupOutput，放入output
    output.extend(groupOutput)

# 19. 打印结果
print("Credit Card Type: {}".format(FIRST_NUMBER[output[0]]))
print("Credit Card #: {}".format("".join(output)))
cv2.imshow("Image", image)
cv2.waitKey(0)
