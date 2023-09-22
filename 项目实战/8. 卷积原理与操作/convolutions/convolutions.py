# 导入工具包
from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2


# 自己定义的一个卷积操作函数
def convolve(image, kernel):
    # 输入图像和核的尺寸
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]

    # 边框的宽度为卷积核宽度大小的一半，卷积核宽度一般要求为奇数（3,5,7...）。
    pad = (kW - 1) // 2
    # 在图像的四个边缘（top, bottom, left, right）添加边框，边框的宽度为卷积核宽度大小的一半，以弥补边界点贡献较少的问题。
    # 边框的类型为cv2.BORDER_REPLICATE，表示边框将复制图像边缘的像素值，以填充边框。
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    # 创建一个和图像大小相同的全零矩阵，作为输出基础框架。
    output = np.zeros((iH, iW), dtype="float32")

    # 卷积操作
    # # 创建一个从 pad 开始, 到 iH + pad - 1 结束，即边缘扩展后的原始图像区域
    for y in np.arange(pad, iH + pad):
        # 创建一个从 pad 开始, 到 iW + pad - 1 结束，即边缘扩展后的原始图像区域
        for x in np.arange(pad, iW + pad):
            # 提取以每一个像素点为中心的卷积区域
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
            # 将卷积区域 roi 与 kernel核进行内积，即对应位置的值相乘。然后，对所有乘积结果求和，得到卷积操作的最终结果 k。
            k = (roi * kernel).sum()
            # 将结果保存在对应原始图像的对应位置（y-pad, x-pad）
            output[y - pad, x - pad] = k

    # 函数将对输入图像中的像素值进行线性映射，in_range指定输入范围的最小和最大像素值（在这个范围外的被认为是最大值255/最小值0）；out_range没有指定，因为图像是浮点数类型，依据函数定义这里是默认使用 0~1 的范围。
    output = rescale_intensity(output, in_range=(0, 255))
    # 将 output 图像的像素值乘以255，并将结果转换为无符号8位整数类型。uint8 类型表示每个像素值占用8位，范围在0到255之间。
    output = (output * 255).astype("uint8")

    return output


# 指定输入图像
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

# 分别构建两个卷积核
smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))  # 卷积核较小   (1.0 / (7 * 7))代表归一化
largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))  # 卷积核较大

# 尝试不同的卷积核
sharpen = np.array((
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]), dtype="int")

laplacian = np.array((
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]), dtype="int")

sobelX = np.array((
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]), dtype="int")

sobelY = np.array((
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]), dtype="int")

# 尝试不同结果
kernelBank = (
    ("small_blur", smallBlur),
    ("large_blur", largeBlur),
    ("sharpen", sharpen),
    ("laplacian", laplacian),
    ("sobel_x", sobelX),
    ("sobel_y", sobelY)
)

# 简单起见，用灰度图实验
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 遍历每一个核
for (kernelName, kernel) in kernelBank:
    print("[INFO] applying {} kernel".format(kernelName))
    # 自定义函数执行图像卷积操作。
    convoleOutput = convolve(gray, kernel)
    # 使用了 OpenCV 的 filter2D 函数，它实现了相同的卷积操作。
    # gray 是输入的灰度图像，-1 表示输出图像的深度与输入图像相同，kernel 是卷积核，定义了卷积操作的方式。
    opencvOutput = cv2.filter2D(gray, -1, kernel)

    # 分别展示结果
    cv2.imshow("original", gray)
    cv2.imshow("{} - convole".format(kernelName), convoleOutput)
    cv2.imshow("{} - opencv".format(kernelName), opencvOutput)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
