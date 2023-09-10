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
	image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
		cv2.BORDER_REPLICATE)
	output = np.zeros((iH, iW), dtype="float32")

	# 卷积操作
	for y in np.arange(pad, iH + pad):
		for x in np.arange(pad, iW + pad):
			# 提取每一个卷积区域
			roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

			# 内积运算
			k = (roi * kernel).sum()

			# 保存相应的结果
			output[y - pad, x - pad] = k

	# 将得到的结果放缩到[0, 255]
	output = rescale_intensity(output, in_range=(0, 255))
	output = (output * 255).astype("uint8")

	return output

# 指定输入图像
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
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
	convoleOutput = convolve(gray, kernel)
	# -1 表示深度一致
	opencvOutput = cv2.filter2D(gray, -1, kernel)

	# 分别展示结果
	cv2.imshow("original", gray)
	cv2.imshow("{} - convole".format(kernelName), convoleOutput)
	cv2.imshow("{} - opencv".format(kernelName), opencvOutput)
	cv2.waitKey(0)
	cv2.destroyAllWindows()