# https://digi.bib.uni-mannheim.de/tesseract/
# 配置环境变量如 E:\Program Files (x86)\Tesseract-OCR
# tesseract -v 进行测试
# tesseract XXX.png result 得到结果
# pip install pytesseract
# anaconda --> lib --> site-packges --> pytesseract --> pytesseract.py：30行左右，tesseract_cmd 修改为绝对路径即可
# 例：tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
from PIL import Image
import pytesseract  # 默认只支持英文
import cv2
import os

preprocess = 'blur'  # thresh

image = cv2.imread('scan.jpg')  # 读取扫描矫正之后的图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转为灰度图

if preprocess == "thresh":
    # cv2.THRESH_BINARY | cv2.THRESH_OTSU，表示同时应用二进制阈值和OTSU阈值化。
    #  ---python教学---：按位或运算符（|）：只要对应的二个二进位有一个为1时，结果位就为1。当两个二进制数所有1位均不相同，效果等同于普通加法。
    #  enum ThresholdTypes {
    #     THRESH_BINARY = 0,   0(2)
    #     THRESH_BINARY_INV = 1,   1(2)
    #     THRESH_TRUNC = 2,   10(2)
    #     THRESH_TOZERO = 3,    11(2)
    #     THRESH_TOZERO_INV = 4,    100(2)
    #     THRESH_MASK = 7,   111(2)
    #     THRESH_OTSU = 8,   1000(2)
    #     THRESH_TRIANGLE = 16   10000(2)
    #  };
    # cv2.THRESH_BINARY 的枚举值为 0，cv2.THRESH_OTSU 的枚举值为 8，cv2.THRESH_BINARY | cv2.THRESH_OTSU也是8，理论上和单独使用cv2.THRESH_OTSU的效果一样。
    # 阈值化操作将图像中的像素根据阈值分为两类：大于阈值的像素被置为255（白色），小于阈值的像素被置为0（黑色）。
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

if preprocess == "blur":
    # 中值滤波，消除图像噪声，这里使用的滤波器窗口大小是3x3。
    gray = cv2.medianBlur(gray, 3)

# 这行代码用于创建一个文件名字符串，基于当前进程的进程ID（PID）生成一个独一无二的文件名。
# os.getpid() 是Python标准库中的一个函数，用于获取当前进程的进程ID（PID）。进程ID是一个唯一的整数标识符，用于区分不同的运行中的进程。
# {} 是一个占位符，format 方法将在 {} 中插入其函数变量os.getpid()。
filename = "{}.png".format(os.getpid())
gray = cv2.resize(gray, (0, 0), fx=0.4, fy=0.4)  # 缩小一下，要不展示的太大了
gray = cv2.rotate(gray, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)  # 逆时针旋转90度，方便阅读
cv2.imwrite(filename, gray)  # 保存图片

# pytesseract 是一个Python库，它是Google的开源OCR引擎Tesseract的Python封装，用于光学字符识别（OCR）任务，可以从图像中提取文本信息。
# image_to_string 是 pytesseract 库的一个函数，用于将图像中的文本转换为字符串。
# Image.open 函数用于打开图像文件，并返回一个表示图像的Pillow图像对象。
# text 用于存储从图像中提取的文本字符串。
text = pytesseract.image_to_string(Image.open(filename))  # 整个py文件的核心
print(text)
os.remove(filename)  # 删除名为 filename 的文件

cv2.imshow("Image", cv2.resize(image, (0, 0), fx=0.4, fy=0.4))
cv2.imshow("Output", gray)
cv2.waitKey(0)
