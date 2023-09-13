from __future__ import division
import matplotlib.pyplot as plt
import cv2
import os, glob
import numpy as np
from PIL import Image
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from keras.preprocessing import image
from Parking import Parking
import pickle

# 这行代码用于获取当前工作目录（Current Working Directory，缩写为 cwd），它会将当前 Python 脚本所在的文件夹的路径存储在 cwd 变量中。
cwd = os.getcwd()


def img_process(test_images, park):
    # ---Python教学---：map 函数接受两个参数：一个函数（park.select_rgb_white_yellow）和一个可迭代对象（test_images），
    #                 然后将函数应用于可迭代对象中的每个元素，并返回结果的迭代器。
    white_yellow_images = list(map(park.select_rgb_white_yellow, test_images))
    park.show_images(white_yellow_images)

    # 将彩色图像转换为灰度图像。
    gray_images = list(map(park.convert_gray_scale, white_yellow_images))
    park.show_images(gray_images)

    # 用于在灰度图像上检测边缘。默认低阈值50，高阈值200。
    # ---Python教学---：使用了一个匿名函数（lambda 函数），该函数代表着将 park.detect_edges 函数应用于一个可迭代对象中的每个元素，并返回这些作用完的元素。
    #              也就相当于 def xxx(image):
    #                           park.detect_edges(image)
    #                           return image
    #           故 map(lambda image: park.detect_edges(image), gray_images) ==相当于==> map(xxx(image), gray_images)
    # 这里由于detect_edges函数返回cv2.Canny，只是在图像上做了修改，并没有单独的返回值，所以需要用匿名函数将返回值包装为图像
    edge_images = list(map(lambda image: park.detect_edges(image), gray_images))
    park.show_images(edge_images)

    # 将图像中想要的区域保留，其余部分置为黑色。
    roi_images = list(map(park.select_region, edge_images))
    park.show_images(roi_images)

    # 检测图像中的直线线段的列表。
    list_of_lines = list(map(park.hough_lines, roi_images))

    line_images = []  # 存放图片
    # ---Python教学---：同时遍历两个列表 zip([a1,a2,a3],[b1,b2,b3]) ==>  ([a1,b1],[a2,b2],[a3,b3])
    # test_images-->white_yellow_images-->gray_images-->gray_images-->edge_images-->roi_images-->list_of_lines，
    # 所以是test_images和list_of_lines里的元素是一一对应的，可以直接zip
    for image, lines in zip(test_images, list_of_lines):
        line_images.append(park.draw_lines(image, lines))  # 在图上画出筛选后的线段
    park.show_images(line_images)  # 画多张子图，整合成一张整图

    rect_images = []
    rect_coords = []
    for image, lines in zip(test_images, list_of_lines):
        new_image, rects = park.identify_blocks(image, lines)
        rect_images.append(new_image)  # 在图像上表示矩形车位
        rect_coords.append(rects)  # 一个rects存放了一张图上的所有矩形车位的坐标

    park.show_images(rect_images)

    delineated = []
    spot_pos = []
    # 依旧是一一对应，rect_coords里的每个元素存放了一张图上的所有矩形车位的坐标
    for image, rects in zip(test_images, rect_coords):
        # 在图像中绘制停车位标记线和编号，并返回带有标记的图像和停车位字典。
        # 将每个停车位的坐标范围和编号存储在 spot_dict 字典中。
        new_image, spot_dict = park.draw_parking(image, rects)
        delineated.append(new_image)
        spot_pos.append(spot_dict)

    park.show_images(delineated)
    final_spot_dict = spot_pos[1]
    print(len(final_spot_dict))

    with open('spot_dict.pickle', 'wb') as handle:
        # 保存车位坐标字典到名为 "spot_dict.pickle" 的文件中。
        pickle.dump(final_spot_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # 将图像中的每个停车位按照其坐标从原图像中裁剪出来，然后将裁剪后的停车位图像保存到指定文件夹中，以便用于卷积神经网络 (CNN) 的训练数据。
    park.save_images_for_cnn(test_images[0], final_spot_dict)

    return final_spot_dict


# 该函数用于加载预训练的Keras模型。它接受一个参数 weights_path，该参数是指预训练模型的权重文件的路径。
def keras_model(weights_path):
    # 这是Keras中的一个函数，用于从指定路径加载一个已经保存的模型，包括模型的架构和权重。
    # 传递给函数的 weights_path 参数是模型权重文件的路径，该文件包含了模型的学习参数。
    model = load_model(weights_path)
    # 函数返回加载的模型
    return model


def img_test(test_images, final_spot_dict, model, class_dictionary):
    for i in range(len(test_images)):
        # 使用深度学习模型在输入图像上进行目标检测和分类，并在图像上标记出识别出的停车位，同时计算并显示可用停车位和总停车位的数量。
        predicted_images = park.predict_on_image(test_images[i], final_spot_dict, model, class_dictionary)


def video_test(video_name, final_spot_dict, model, class_dictionary):
    name = video_name
    cap = cv2.VideoCapture(name)  # 这一行意义不明
    # 使用深度学习模型在输入视频上进行目标检测和分类，标记出识别出的停车位，同时计算并显示可用停车位和总停车位的数量。
    park.predict_on_video(name, final_spot_dict, model, class_dictionary, ret=True)


if __name__ == '__main__':
    # 使用glob模块的glob函数，它接受一个文件路径模式（通配符）作为参数，返回匹配模式的文件列表。
    # 使用Matplotlib库的imread函数，它用于读取图像文件并将其加载为NumPy数组。path参数是文件的路径。
    test_images = [plt.imread(path) for path in glob.glob('test_images/*.jpg')]  # 列表推导式
    weights_path = 'car1.h5'  # 训练好的模型
    video_name = 'parking_video.mp4'
    class_dictionary = {}
    class_dictionary[0] = 'empty'
    class_dictionary[1] = 'occupied'
    park = Parking()  # 封装了所有需要用到的函数的一个类
    park.show_images(test_images)
    # 用于图像处理和车位检测的操作。它执行了一系列的图像处理步骤，并最终返回一个包含车位信息的字典。
    final_spot_dict = img_process(test_images, park)
    model = keras_model(weights_path)
    img_test(test_images, final_spot_dict, model, class_dictionary)
    video_test(video_name, final_spot_dict, model, class_dictionary)
