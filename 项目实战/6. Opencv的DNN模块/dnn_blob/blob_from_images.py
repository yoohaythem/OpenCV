# 1. 导入工具包
import utils_paths
import numpy as np
import cv2

# 2. 标签文件处理
# 使用换行符 \n 分割字符串，将其拆分成多行文本，并返回一个包含每行文本的列表，存储在名为 rows 的变量中。
rows = open("synset_words.txt").read().strip().split("\n")   # 链式编程
# 使用列表推导式遍历 rows 列表，对每行文本进行处理，在每行文本 r 中查找第一个空格字符的索引位置，并获取该位置之后的子字符串。然后根据逗号分割字符串，取第一个逗号前面的文本。
# 得到的是一个类别的标签集合
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

# 3. Caffe所需配置文件
# 使用 OpenCV 的 cv2.dnn 模块中的 readNetFromCaffe 函数加载一个深度神经网络模型，该模型采用了 Caffe 框架的模型文件格式。它接受两个参数：模型的配置文件和模型的权重文件。
# 	"bvlc_googlenet.prototxt"：这是模型的配置文件的文件名，通常以 ".prototxt" 扩展名结尾。配置文件包含了网络的结构信息，例如层的类型、层的参数等。
# 	"bvlc_googlenet.caffemodel"：这是模型的权重文件的文件名，通常以 ".caffemodel" 扩展名结尾。权重文件包含了神经网络中每个层的权重参数。
# 这行代码的目的是加载名为 "bvlc_googlenet" 的深度神经网络模型，该模型的结构和参数信息分别存储在 "bvlc_googlenet.prototxt" 和 "bvlc_googlenet.caffemodel" 两个文件中。一旦加载成功，后续就可以使用这个模型进行图像的推理和预测任务。这通常用于图像分类、对象检测等计算机视觉任务。
net = cv2.dnn.readNetFromCaffe("bvlc_googlenet.prototxt", "bvlc_googlenet.caffemodel")

# 4. 图像路径
# 这行代码的目的是获取images/目录中的图像文件（根据 utils_paths.py 中定义）的路径，并将这些路径按照字母顺序进行排序。
imagePaths = sorted(list(utils_paths.list_images("images/")))

# 5. 图像数据预处理
image = cv2.imread(imagePaths[0])
resized = cv2.resize(image, (224, 224))
# 使用 OpenCV 中的 cv2.dnn.blobFromImage 函数来创建一个 blob 对象。Blob（Binary Large Object）是深度学习中常用的一种数据表示方式，通常用于输入神经网络模型。
# 下面是代码中各参数的解释：
# 	resized：这是输入图像，通常是经过预处理和调整大小后的图像。
# 	1：这是归一化因子，用于对图像的像素值进行缩放。在这里，像素值不进行缩放，因此设置为 1。
# 	(224, 224)：这是目标 blob 的空间维度，表示模型期望的输入图像尺寸。一般来说，深度学习模型有固定的输入尺寸要求，通常是正方形的。
# 	(104, 117, 123)：这是每个通道的均值。在深度学习中，为了对图像进行预处理，通常需要减去均值。这里的均值是在训练模型时计算得出的，用于将图像从原始颜色空间转换为模型期望的颜色空间。不同的模型和数据集有不同的均值。
# 最终，这行代码生成了一个 blob 对象，该对象包含了经过预处理后的图像数据，可以传递给深度学习模型进行推断或预测。这个 blob 对象通常包括图像的通道数、高度和宽度等信息，以适应模型的输入要求。在深度学习中，这种预处理是非常常见的，以确保输入数据与模型的期望相匹配。
blob = cv2.dnn.blobFromImage(resized, 1, (224, 224), (104, 117, 123))
print("First Blob: {}".format(blob.shape))

# 6. 得到预测结果
# 将之前生成的 blob 对象传递给神经网络模型 net，以供模型进行后续的推断操作。模型将使用这个输入 blob 来执行前向传播，计算输出结果，通常是对输入图像的分类、检测、分割等任务。
net.setInput(blob)
# net.forward() 方法触发了模型的前向传播操作，将之前设置的输入数据传递给模型，并计算模型的输出。
# 一旦前向传播完成，preds 变量将包含模型的预测结果。
preds = net.forward()

# 7. 排序，取分类可能性最大的
# 对模型的预测结果 preds[0] 进行排序，返回排序后的索引数组。这些索引表示类别的顺序，从最不可能的类别到最可能的类别排列。
# [::-1]：反转索引数组，将最可能的类别排在前面。
# [0]：选择排序后的第一个索引，即最可能的类别的索引。
idx = np.argsort(preds[0])[::-1][0]
# classes[idx]：使用选定的索引查找类别列表 classes 中对应的类别名称，这是最终的分类结果。
# preds[0][idx] * 100：获取对应类别的置信度得分，并将其乘以 100，以得到以百分比表示的置信度。
text = "Label: {}, {:.2f}%".format(classes[idx], preds[0][idx] * 100)
cv2.putText(image, text, (5, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# 8. 显示
cv2.imshow("Image", image)
cv2.waitKey(0)

# 9. Batch数据制作
images = []

# 10. 方法一样，数据是一个batch，第一张图片已经处理完了，下面批量处理剩下的图片
for p in imagePaths[1:]:
	image = cv2.imread(p)
	image = cv2.resize(image, (224, 224))
	images.append(image)

# 11. blobFromImages函数，注意有s
# 对剩余图像批量预处理，得到blob
blob = cv2.dnn.blobFromImages(images, 1, (224, 224), (104, 117, 123))
print("Second Blob: {}".format(blob.shape))

# 12. 获取预测结果
# 将上面获取的 blob 对象传递给神经网络模型 net，并触发模型的前向传播操作
net.setInput(blob)
preds = net.forward()
# 对剩余每一张图片都执行和7步骤相同的操作
for (i, p) in enumerate(imagePaths[1:]):
	image = cv2.imread(p)
	idx = np.argsort(preds[i])[::-1][0]
	text = "Label: {}, {:.2f}%".format(classes[idx], preds[i][idx] * 100)
	cv2.putText(image, text, (5, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	cv2.imshow("Image", image)
	cv2.waitKey(0)
