# 导入工具包
from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2

def convolve(image, kernel):
	# 输入图像和核的尺寸
	(iH, iW) = image.shape[:2]
	(kH, kW) = kernel.shape[:2]

	# 选择pad，卷积后图像大小不变
	pad = (kW - 1) // 2
	# 重复最后一个元素，top, bottom, left, right
	# 在图像的四个边缘添加边框，边框的宽度由变量pad指定，边框的类型为cv2.BORDER_REPLICATE，表示边框将复制图像边缘的像素值，以填充边框。
	image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
	# 创建一个和图像大小相同的全零矩阵
	output = np.zeros((iH, iW), dtype="float32")

	# 卷积操作
	for y in np.arange(pad, iH + pad):   # 创建一个从pad开始, 到iH + pad - 1结束
		for x in np.arange(pad, iW + pad):   # 创建一个从pad开始, 到iW + pad - 1结束
			# 提取每一个卷积区域
			roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

			# 内积运算
			# roi 表示一个图像区域，kernel 是一个与之进行卷积的滤波器。roi * kernel 执行逐元素的乘法，将 roi 中的每个像素与 kernel 中的对应位置的值相乘。
			# 然后，.sum() 函数对所有乘积结果求和，得到卷积操作的最终结果 k。
			k = (roi * kernel).sum()

			# 保存相应的结果
			output[y - pad, x - pad] = k

	# 将得到的结果放缩到[0, 255]
	# 使用 rescale_intensity 函数来对 output 图像的像素值进行强度重缩放。
	# in_range=(0, 255) 指定了输入图像像素值的范围，这表示输入图像的像素值应该在0到255之间。函数将对输入图像中的像素值进行线性映射，以确保它们在指定的范围内，这有助于提高图像的对比度和可视化效果。
	output = rescale_intensity(output, in_range=(0, 255))
	# 这行代码将 output 图像的像素值乘以255，并将结果转换为无符号8位整数类型 (uint8)。这是将像素值重新缩放到0到255范围后，将它们转换为整数类型的常见做法。uint8 类型表示每个像素值占用8位，范围在0到255之间，适合表示图像像素值。
	output = (output * 255).astype("uint8")

	return output

# 指定输入图像
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

# 分别构建两个卷积核
smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))

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

# 简单起见，用灰度图来玩
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 遍历每一个核
for (kernelName, kernel) in kernelBank:

	print("[INFO] applying {} kernel".format(kernelName))
	# 这里使用了一个名为 convolve 的自定义函数，该函数执行了图像卷积操作。通常情况下，卷积操作用于图像滤波、特征提取等任务。
	# gray 是输入的灰度图像，kernel 是卷积核（滤波器），卷积核定义了卷积操作的方式。卷积核在图像上滑动，计算图像中每个位置的加权和，将结果存储在 convolveOutput 中。
	convoleOutput = convolve(gray, kernel)
	# 使用了 OpenCV 的 filter2D 函数，它实现了相同的卷积操作。gray 是输入的灰度图像，-1 表示输出图像的深度与输入图像相同，kernel 是卷积核，定义了卷积操作的方式。结果存储在 opencvOutput 中。
	opencvOutput = cv2.filter2D(gray, -1, kernel)

	# 分别展示结果
	cv2.imshow("original", gray)
	cv2.imshow("{} - convole".format(kernelName), convoleOutput)
	cv2.imshow("{} - opencv".format(kernelName), opencvOutput)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
