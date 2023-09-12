import matplotlib.pyplot as plt
import cv2
import os, glob
import numpy as np


class Parking:

    def show_images(self, images, cmap=None):
        cols = 2  # 要在图中排列子图像的列数
        rows = (len(images) + 1) // cols  # 计算子图像的行数，确保足够的行数以容纳所有图像。

        plt.figure(figsize=(15, 12))  # 创建一个新的Matplotlib图形，指定图形的大小为15x12英寸。
        for i, image in enumerate(images):  # 枚举图像，自然生成索引，索引i从0开始依次增加
            plt.subplot(rows, cols, i + 1)  # 创建子图，子图的索引从1开始，因此使用 i + 1。
            # 确定要使用的颜色映射（colormap）。如果图像是灰度图像（即通道数为1，第三维蜷缩），则使用 'gray' 颜色映射；否则，使用传递给方法的 cmap 参数。
            cmap = 'gray' if len(image.shape) == 2 else cmap
            plt.imshow(image, cmap=cmap)
            # 下两行代码用于隐藏子图的刻度线。
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout(pad=0, h_pad=0, w_pad=0)  # 调整子图的布局，使它们更紧凑，去除了多余的空白。
        plt.show()

    def cv_show(self, name, img):
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def select_rgb_white_yellow(self, image):
        # 过滤掉背景
        # lower 和 upper：这是两个NumPy数组，分别表示颜色阈值的下限和上限。
        # lower 表示白色的下限颜色（RGB值为 [120, 120, 120]），而 upper 表示白色的上限颜色（RGB值为 [255, 255, 255]）。
        lower = np.uint8([120, 120, 120])
        upper = np.uint8([255, 255, 255])
        # 使用OpenCV的 cv2.inRange 函数，根据指定的下限和上限颜色值，创建一个掩码（mask），
        # 该掩码将输入图像中在指定颜色范围内（lower~upper）的部分设置为255，而不在范围内的部分设置为0。这个掩码用于分离白色和黄色区域。
        # （注意，这里是掩码255，而不是图像255）
        white_mask = cv2.inRange(image, lower, upper)
        self.cv_show('white_mask', white_mask)

        # 这行代码使用掩码 white_mask 对输入图像 image 进行按位与（bitwise and）操作，将只保留在指定颜色范围内的部分，其他部分将变为零。
        # 这样，就可以将想要的区域提取出来，其他区域变为黑色。
        masked = cv2.bitwise_and(image, image, mask=white_mask)
        self.cv_show('masked', masked)
        return masked

    def convert_gray_scale(self, image):
        # 将彩色图像转换为灰度图像。
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def detect_edges(self, image, low_threshold=50, high_threshold=200):
        # 用于在灰度图像上检测边缘。默认低阈值50，高阈值200.
        return cv2.Canny(image, low_threshold, high_threshold)

    def filter_region(self, image, vertices):
        # 过滤图像中指定区域以外的部分，并将区域内的部分保留。
        # 与输入图像具有相同形状的零矩阵，用于创建一个与输入图像相同大小的掩码图像。
        mask = np.zeros_like(image)
        if len(mask.shape) == 2:  # 维度是2，为灰度图
            # 使用OpenCV的 cv2.fillPoly 函数，将指定多边形区域（由 vertices 定义）填充为白色（像素值为255），将其他部分保持为黑色（像素值为0）。
            # 这样就创建了一个掩码图像，其中指定区域是白色，其他区域是黑色。
            cv2.fillPoly(mask, vertices, 255)
            self.cv_show('mask', mask)
        # 使用掩码图像 mask 对输入图像 image 进行按位与（bitwise and）操作，将保留指定区域（白色区域），其他区域变为黑色。
        return cv2.bitwise_and(image, mask)

    def select_region(self, image):
        rows, cols = image.shape[:2]
        # 定义了一个由顶点坐标组成的 vertices 列表，表示要手动选择的区域的轮廓。
        pt_1 = [cols * 0.05, rows * 0.90]
        pt_2 = [cols * 0.05, rows * 0.70]
        pt_3 = [cols * 0.30, rows * 0.55]
        pt_4 = [cols * 0.6, rows * 0.15]
        pt_5 = [cols * 0.90, rows * 0.15]
        pt_6 = [cols * 0.90, rows * 0.90]
        # 这些顶点按顺序连接，形成一个多边形，表示手动选择的区域。传入cv2.fillPoly加以使用
        vertices = np.array([[pt_1, pt_2, pt_3, pt_4, pt_5, pt_6]], dtype=np.int32)
        point_img = image.copy()
        point_img = cv2.cvtColor(point_img, cv2.COLOR_GRAY2RGB)  # 转为灰度图
        for point in vertices[0]:  # vertices[0]：[pt_1, pt_2, pt_3, pt_4, pt_5, pt_6]
            # 用于在图像point_img上，以(point[0], point[1])为圆心，半径为10，颜色为红色（BGR），圆的边框线宽度为4的圆。
            cv2.circle(point_img, (point[0], point[1]), 10, (0, 0, 255), 4)
        self.cv_show('point_img', point_img)

        return self.filter_region(image, vertices)

    def hough_lines(self, image):
        # 该方法用于在输入图像上执行霍夫直线变换（Hough Line Transform）。霍夫直线变换通常用于检测二值图像中的直线。
        # 这行代码调用了OpenCV的 cv2.HoughLinesP 函数，执行霍夫直线变换。P 表示Probabilistic（概率性）霍夫直线变换，它可以检测出图像中的直线线段。
        #   image：输入的图像 image 是边缘检测后的结果。
        #   rho：霍夫空间中的距离分辨率，通常以像素为单位。较小的值表示更精细的距离分辨率。
        #   theta：霍夫空间中的角度分辨率，通常以弧度为单位。较小的值表示更精细的角度分辨率。
        #   threshold：霍夫直线变换的阈值，用于确定检测到的直线线段。只有当一条线段的投票数（累积在霍夫空间中的数量）超过阈值时，才会被检测到。
        #   minLineLength：要检测的直线线段的最小长度。小于此长度的线段将被忽略。
        #   maxLineGap：允许将两个线段之间的最大间隔视为同一条线段的参数。如果两个线段之间的间隔小于等于 maxLineGap，则它们将被合并为一条线段。
        # 返回值是一个包含检测到的直线线段的列表，每个线段用其两个端点的坐标表示。
        return cv2.HoughLinesP(image, rho=0.1, theta=np.pi / 10, threshold=15, minLineLength=9, maxLineGap=4)

    def draw_lines(self, image, lines, color=[255, 0, 0], thickness=2, make_copy=True):
        # 过滤霍夫变换检测到的直线线段列表
        if make_copy:
            image = np.copy(image)
        cleaned = []  # 存放筛选后的线段列表
        # lines是一个包含直线信息的列表，每个直线由四个值表示：(x1, y1) 和 (x2, y2)，表示直线的两个端点的坐标。
        for line in lines:
            for x1, y1, x2, y2 in line:
                # 直线的斜率接近水平 （abs(y2 - y1) <= 1）
                # 直线的长度在一定范围内（25 <= abs(x2 - x1) <= 55）。
                if abs(y2 - y1) <= 1 and 25 <= abs(x2 - x1) <= 55:
                    cleaned.append((x1, y1, x2, y2))
                    # 在图上画出筛选后的线段
                    cv2.line(image, (x1, y1), (x2, y2), color, thickness)
        print(" No lines detected: ", len(cleaned))
        return image

    def identify_blocks(self, image, lines, make_copy=True):
        if make_copy:
            new_image = np.copy(image)
        # Step 1: 过滤部分直线
        cleaned = []  # 存放筛选后的线段列表
        for line in lines:
            for x1, y1, x2, y2 in line:
                # 直线的斜率接近水平 （abs(y2 - y1) <= 1）
                # 直线的长度在一定范围内（25 <= abs(x2 - x1) <= 55）。
                if abs(y2 - y1) <= 1 and 25 <= abs(x2 - x1) <= 55:
                    cleaned.append((x1, y1, x2, y2))

        # Step 2: 对直线按照x1进行排序
        import operator
        # operator.itemgetter(0, 1) 表示按照元素的第一个和第二个值进行排序，也就是先按照 x 坐标（第一个值），然后按照 y 坐标（第二个值）进行升序排序。
        # ---Python教学---： operator.itemgetter(0, 1) 是一个函数对象，它接受一个元素作为参数并返回一个包含该元素指定位置的子元素的元组。
        #                   在这里，itemgetter(0, 1) 用于提取元素的第一个和第二个值，然后 sorted 函数使用这些值进行排序。
        list1 = sorted(cleaned, key=operator.itemgetter(0, 1))

        # Step 3: 找到多个列，每列是一排车
        clusters = {}
        dIndex = 0
        clus_dist = 10

        for i in range(len(list1) - 1):
            # 两条直线在 x 坐标上的距离。
            distance = abs(list1[i + 1][0] - list1[i][0])
            # 如果两条直线的 x 坐标差小于等于 clus_dist，则它们将被分到同一组，每一组内是一些横线。
            if distance <= clus_dist:
                # 如果相邻直线的距离大于 clus_dist，则增加 dIndex，创建一个新的分组。
                if not dIndex in clusters.keys(): clusters[dIndex] = []
                # 如果相邻直线的距离小于等于 clus_dist，则将它们添加到当前分组（dIndex）的列表中。
                clusters[dIndex].append(list1[i])
                clusters[dIndex].append(list1[i + 1])
            else:
                # # 如果相邻直线的距离大于 clus_dist，则增加 dIndex，创建一个新的分组。
                dIndex += 1

        # Step 4: 得到坐标
        rects = {}
        i = 0  # 这里用i不用key,是因为 if len(cleaned) > 5 这个条件会让部分的key被清洗掉
        for key in clusters:  # clusters是个字典，遍历出来的是键值，也就是前面的分组索引
            all_list = clusters[key]
            cleaned = list(set(all_list))  # 使用 set 函数去除重复的直线，将结果存储在 cleaned 变量中。
            if len(cleaned) > 5:
                # 对 cleaned 中的直线按照 y 坐标进行升序排序，以确保直线按照从上到下的顺序排列。
                cleaned = sorted(cleaned, key=lambda tup: tup[1])
                avg_y1 = cleaned[0][1]  # 最上面的横线的y值
                avg_y2 = cleaned[-1][1]  # 最下面的横线的y值
                # 下面6行，计算该列的平均 x1 和 x2 坐标，这表示停车位的宽度。
                avg_x1 = 0
                avg_x2 = 0
                for tup in cleaned:
                    avg_x1 += tup[0]
                    avg_x2 += tup[2]
                avg_x1 = avg_x1 / len(cleaned)
                avg_x2 = avg_x2 / len(cleaned)
                rects[i] = (avg_x1, avg_y1, avg_x2, avg_y2)  # 这四个值定位了一个正矩形
                i += 1
        print("Num Parking Lanes: ", len(rects))

        # Step 5: 把列矩形画出来
        buff = 7
        for key in rects:
            tup_topLeft = (int(rects[key][0] - buff), int(rects[key][1]))  # 矩形左上点
            tup_botRight = (int(rects[key][2] + buff), int(rects[key][3]))  # 矩形右下点
            # 在图像上绘制矩形框，设置矩形的颜色为绿色 (0, 255, 0)，线宽为 3。
            cv2.rectangle(new_image, tup_topLeft, tup_botRight, (0, 255, 0), 3)
        return new_image, rects

    # rects:存放停车位坐标的字典，键为依次递增的数字索引
    # 在图像中绘制停车位标记线和编号，并返回带有标记的图像和停车位字典。
    def draw_parking(self, image, rects, make_copy=True, color=[255, 0, 0], thickness=2, save=True):
        if make_copy:
            new_image = np.copy(image)
        gap = 15.5  # 停车位标记线之间的间隔 gap
        spot_dict = {}  # 创建一个字典 spot_dict，用于存储每个停车位的坐标范围和编号。
        tot_spots = 0

        # 微调，这个就是经验值了，可以跳过不影响代码逻辑
        adj_y1 = {0: 20, 1: -10, 2: 0, 3: -11, 4: 28, 5: 5, 6: -15, 7: -15, 8: -10, 9: -30, 10: 9, 11: -32}
        adj_y2 = {0: 30, 1: 50, 2: 15, 3: 10, 4: -15, 5: 15, 6: 15, 7: -20, 8: 15, 9: 15, 10: 0, 11: 30}
        adj_x1 = {0: -8, 1: -15, 2: -15, 3: -15, 4: -15, 5: -15, 6: -15, 7: -15, 8: -10, 9: -10, 10: -10, 11: 0}
        adj_x2 = {0: 0, 1: 15, 2: 15, 3: 15, 4: 15, 5: 15, 6: 15, 7: 15, 8: 10, 9: 10, 10: 10, 11: 0}

        for key in rects:
            tup = rects[key]
            x1 = int(tup[0] + adj_x1[key])
            x2 = int(tup[2] + adj_x2[key])
            y1 = int(tup[1] + adj_y1[key])
            y2 = int(tup[3] + adj_y2[key])
            # 绘制矩形框，颜色为绿色 (0, 255, 0)，线宽为 2
            cv2.rectangle(new_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            num_splits = int(abs(y2 - y1) // gap)  # 计算停车位中标记线的数量 num_splits。
            for i in range(0, num_splits + 1):
                y = int(y1 + i * gap)
                cv2.line(new_image, (x1, y), (x2, y), color, thickness)  # 绘制水平的停车位标记线
            if 0 < key < len(rects) - 1:
                # 如果停车位不是第一个或最后一个，还绘制垂直的停车位标记线。
                x = int((x1 + x2) / 2)
                cv2.line(new_image, (x, y1), (x, y2), color, thickness)
            # 根据停车位类型（第一个、中间、最后），计算停车位总数 tot_spots。
            if key == 0 or key == (len(rects) - 1):
                tot_spots += num_splits + 1
            else:
                tot_spots += 2 * (num_splits + 1)

            # 将每个停车位的坐标范围和编号存储在 spot_dict 字典中。
            # 如果停车位是第一个或最后一个，只存储水平标记线的信息。
            if key == 0 or key == (len(rects) - 1):
                for i in range(0, num_splits + 1):
                    cur_len = len(spot_dict)
                    y = int(y1 + i * gap)
                    spot_dict[(x1, y, x2, y + gap)] = cur_len + 1
            # 如果是中间的停车位，同时存储水平和垂直标记线的信息。
            else:
                for i in range(0, num_splits + 1):
                    cur_len = len(spot_dict)
                    y = int(y1 + i * gap)
                    x = int((x1 + x2) / 2)
                    spot_dict[(x1, y, x, y + gap)] = cur_len + 1
                    spot_dict[(x, y, x2, y + gap)] = cur_len + 2
        # 打印总停车位数。
        print("total parking spaces: ", tot_spots, cur_len)
        # 如果设置了 save 为 True，将带有停车位标记的图像保存为文件 with_parking.jpg。
        # 返回带有标记的新图像 new_image 和停车位字典 spot_dict。
        if save:
            filename = 'with_parking.jpg'
            cv2.imwrite(filename, new_image)
        return new_image, spot_dict

    # 根据停车位字典中的坐标范围，在图像上绘制停车位的矩形标记，并返回带有标记的图像。
    def assign_spots_map(self, image, spot_dict, make_copy=True, color=[255, 0, 0], thickness=2):
        if make_copy:
            new_image = np.copy(image)
        # 对于停车位字典中的每个停车位（表示为 (x1, y1, x2, y2)），执行以下操作：
        # 使用 cv2.rectangle 绘制矩形框，颜色为红色 (255, 0, 0)，线宽为 thickness。
        for spot in spot_dict.keys():
            (x1, y1, x2, y2) = spot
            cv2.rectangle(new_image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        return new_image

    # 将图像中的每个停车位按照其坐标从原图像中裁剪出来，然后将裁剪后的停车位图像保存到指定文件夹中，以便用于卷积神经网络 (CNN) 的训练数据。
    def save_images_for_cnn(self, image, spot_dict, folder_name='cnn_data'):
        for spot in spot_dict.keys():
            (x1, y1, x2, y2) = spot
            (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
            # 裁剪
            spot_img = image[y1:y2, x1:x2]
            # 将停车位图像放大两倍，以便适应训练数据的需求。
            spot_img = cv2.resize(spot_img, (0, 0), fx=2.0, fy=2.0)
            spot_id = spot_dict[spot]
            # 为每个停车位生成一个文件名，命名规则为 'spot' + 停车位编号 + '.jpg'。
            filename = 'spot' + str(spot_id) + '.jpg'
            print(spot_img.shape, filename, (x1, x2, y1, y2))
            # 使用 cv2.imwrite 函数将裁剪后的停车位图像保存到指定的文件夹 folder_name 中。
            cv2.imwrite(os.path.join(folder_name, filename), spot_img)

    # 使用训练好的深度学习模型对输入的图像进行分类预测。
    def make_prediction(self, image, model, class_dictionary):
        # 将输入的图像 image 像素值缩放到 [0, 1] 范围内，以便与模型的训练数据相匹配。这是通过将图像的每个像素值除以 255 来实现的。
        img = image / 255.

        # 将处理后的图像转换成一个4D张量。这是因为深度学习模型通常接受批量图像作为输入，即使我们只有一张图像也需要将其包装成批量。
        image = np.expand_dims(img, axis=0)

        # 用训练好的模型进行训练
        # 调用模型的 predict 方法，传入处理后的图像作为输入，获取模型的分类预测结果。
        class_predicted = model.predict(image)
        # 从分类预测结果中找到具有最高概率的类别索引 inID。
        inID = np.argmax(class_predicted[0])
        # 然后根据类别字典 class_dictionary 找到对应的类别标签 label。
        label = class_dictionary[inID]
        return label

    # 使用深度学习模型在输入图像上进行目标检测和分类，并在图像上标记出识别出的停车位，同时计算并显示可用停车位和总停车位的数量。
    def predict_on_image(self, image, spot_dict, model, class_dictionary, make_copy=True, color=[0, 255, 0], alpha=0.5):
        if make_copy:
            new_image = np.copy(image)
            overlay = np.copy(image)
        self.cv_show('new_image', new_image)
        cnt_empty = 0
        all_spots = 0
        for spot in spot_dict.keys():  # 对于每个停车位，执行以下操作：
            all_spots += 1
            # 下面三行提取图像区域：根据停车位的坐标从输入图像中提取出停车位的图像区域。
            (x1, y1, x2, y2) = spot
            (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
            spot_img = image[y1:y2, x1:x2]
            # 调整图像大小：将停车位的图像区域调整为固定大小（48x48像素），以便输入到深度学习模型进行分类。
            spot_img = cv2.resize(spot_img, (48, 48))
            # 使用模型进行分类：调用 make_prediction 函数，使用深度学习模型对停车位的图像区域进行分类，获取分类标签。
            label = self.make_prediction(spot_img, model, class_dictionary)
            if label == 'empty':
                # 如果分类标签为 'empty'，表示停车位为空，那么在 overlay 图像上以绿色（BGR）标记该停车位。
                cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)
                cnt_empty += 1

        # 将 overlay 图像叠加到 new_image 上，以显示标记的停车位。
        cv2.addWeighted(overlay, alpha, new_image, 1 - alpha, 0, new_image)

        # 计算可用停车位和总停车位数量：统计分类为 'empty' 的停车位数量和所有停车位的数量，并在图像上显示这两个统计信息。
        cv2.putText(new_image, "Available: %d spots" % cnt_empty, (30, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(new_image, "Total: %d spots" % all_spots, (30, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        save = False

        if save:
            filename = 'with_marking.jpg'
            cv2.imwrite(filename, new_image)
        self.cv_show('new_image', new_image)

        return new_image

    # 使用深度学习模型在输入视频上进行目标检测和分类，标记出识别出的停车位，同时计算并显示可用停车位和总停车位的数量。
    def predict_on_video(self, video_name, final_spot_dict, model, class_dictionary, ret=True):
        # 使用OpenCV的VideoCapture函数打开指定的视频文件。
        cap = cv2.VideoCapture(video_name)
        count = 0
        while ret:
            ret, image = cap.read()  # 通过cap.read()函数读取下一帧的图像。
            count += 1
            if count == 5:  # 使用计数器（count）来控制处理的频率。在这里，每5帧处理一次。也可以写作：if not count%5:
                count = 0

                # 复制图像：创建输入图像的副本 new_image 和一个用于叠加标记的图像副本 overlay。
                new_image = np.copy(image)
                overlay = np.copy(image)
                # 初始化变量：初始化用于统计可用停车位和总停车位数量的变量。
                cnt_empty = 0
                all_spots = 0
                color = [0, 255, 0]
                alpha = 0.5
                # 遍历停车位字典：对于每个停车位，执行以下操作：
                for spot in final_spot_dict.keys():
                    all_spots += 1
                    # 下面三行：提取停车位的图像区域。
                    (x1, y1, x2, y2) = spot
                    (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
                    spot_img = image[y1:y2, x1:x2]
                    # 调整图像大小，以便输入到深度学习模型进行分类。
                    spot_img = cv2.resize(spot_img, (48, 48))
                    # 使用模型进行分类：调用 make_prediction 函数，使用深度学习模型对停车位的图像区域进行分类，获取分类标签。
                    label = self.make_prediction(spot_img, model, class_dictionary)
                    if label == 'empty':
                        # 如果分类标签为 'empty'，表示停车位为空，那么在 overlay 图像上以绿色（BGR）标记该停车位。
                        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)
                        cnt_empty += 1

                # 使用 cv2.addWeighted 函数将 overlay 图像叠加到 new_image 上，以显示标记的停车位。这里是0.5:0.5
                cv2.addWeighted(overlay, alpha, new_image, 1 - alpha, 0, new_image)
                # 在图像上（视频的帧）显示可用停车位和总停车位数量。
                cv2.putText(new_image, "Available: %d spots" % cnt_empty, (30, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(new_image, "Total: %d spots" % all_spots, (30, 125),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # 使用 cv2.imshow 函数显示处理后的图像，并等待按键事件。
                cv2.imshow('frame', new_image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        cv2.destroyAllWindows()
        cap.release()
