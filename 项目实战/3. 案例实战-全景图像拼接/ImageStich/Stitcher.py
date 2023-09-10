import numpy as np
import cv2


class Stitcher:

    # 拼接函数
    def stitch(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False):
        # 获取输入图片
        (imageB, imageA) = images
        # 检测A、B图片的SIFT关键特征点，并计算特征描述子
        # kps 和 features 是函数的返回值，它们分别代表了检测到的关键点和对应的特征描述符。
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        # 匹配两张图片的所有特征点，返回匹配结果
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

        # 如果返回结果为空，没有匹配成功的特征点，退出算法
        # 其实是筛选后的匹配对小于等于4，未知数不够无法求解
        if M is None:
            return None

        # 否则，提取匹配结果
        # H是3x3视角变换矩阵      
        (matches, H, status) = M
        # 这行代码使用OpenCV的 cv2.warpPerspective 函数将图像 imageA 进行透视变换，变换的矩阵为 H，并将变换后的图像存储在 result 中。
        #   imageA：这是要进行透视变换的输入图像，通常是一个包含图像像素数据的NumPy数组。
        #   H：这是透视变换的变换矩阵，它描述了如何将输入图像 imageA 映射到输出图像中。这个矩阵通常是通过图像拼接或配准过程中的计算得到的。
        #   (imageA.shape[1] + imageB.shape[1], imageA.shape[0])：
        #   这是输出图像的大小，表示输出图像的宽度为两幅输入图像宽度之和，高度与输入图像 imageA 相同。这确保了输出图像可以容纳两幅输入图像的拼接结果。
        result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        self.cv_show('result', result)
        # 将图片B传入result图片最左端
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
        self.cv_show('result', result)
        # 检测是否需要显示图片匹配
        if showMatches:
            # 生成匹配图片
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
            # 返回结果
            return (result, vis)

        # 返回匹配结果
        return result

    def cv_show(self, name, img):
        cv2.imshow(name, img)
        # 这行代码会等待用户按键响应，参数 0 表示等待无限长的时间，直到用户按下键盘上的任意键。在等待期间，图像窗口会保持打开状态。
        cv2.waitKey(0)
        # 这行代码用于关闭所有打开的图像窗口。一旦用户按下键盘上的任意键，就会执行这行代码关闭图像窗口。
        cv2.destroyAllWindows()

    def detectAndDescribe(self, image):
        # 将彩色图片转换成灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 建立SIFT生成器，将 SIFT 算法实例化出来
        descriptor = cv2.SIFT_create()
        # 检测SIFT特征点，并计算描述子
        # None：这是一个可选参数，通常用于传递掩码图像。如果不需要使用掩码来限制特定区域的关键点检测，可以将其设置为None。
        # (kps, features)：这是函数的返回值，是一个包含两个元素的元组：
        #   kps：这是一个关键点列表，包含了在输入图像中检测到的关键点的位置信息。每个关键点通常包括坐标（x、y）等信息。
        #   features：这是关键点的特征描述符列表，对应于每个关键点。特征描述符通常是一个数值向量，用于描述关键点周围区域的图像信息。
        (kps, features) = descriptor.detectAndCompute(gray, None)
        # 将结果转换成NumPy数组
        # kp.pt：这是一个关键点对象 kp 的属性，表示该关键点的坐标。关键点坐标以浮点数表示，并且通过 kp.pt 可以获取关键点的 (x, y) 坐标。
        # ---Python教学---：[kp.pt for kp in kps]：这是一个列表推导式，通过遍历 kps 中的每个关键点对象 kp，从中提取关键点的坐标 kp.pt，并将这些坐标组成一个列表。
        kps = np.float32([kp.pt for kp in kps])
        # 返回特征点集，及对应的描述特征
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        # 建立暴力匹配器
        matcher = cv2.BFMatcher()  # 实例化

        # 使用KNN检测来自A、B图的SIFT特征匹配对，k对最佳匹配，K=2
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)

        matches = []
        for m in rawMatches:
            # 当最近距离跟次近距离的比值小于ratio = 0.75 值时，保留此匹配对
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:  # 最好的点是次好的点的 1/ratio 倍的时候保留
                # 存储两个点在featuresA, featuresB中的索引值
                #  m[0].trainIdx 和 m[0].queryIdx 分别表示最近匹配点在第1幅图像（A图）中的索引和在第2幅图像（B图）中的索引。
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # 当筛选后的匹配对大于4时，计算视角变换矩阵
        # 因为是3*3的矩阵，八个未知数，解方程至少需要四个点的值
        if len(matches) > 4:
            # 获取匹配对的点坐标
            # matches 是一个包含了匹配对的索引的列表，每个匹配对包含了两个索引：
            # 第一个索引表示第一幅图像（A图像）中的关键点索引，而第二个索引表示第二幅图像（B图像）中的关键点索引。
            # kps[i] 表示从图像的关键点列表 kps 中提取第 i 个关键点的坐标。
            # pts 是一个包含了第图像中匹配对中关键点坐标的Numpy数组。
            ptsA = np.float32([kpsA[i] for (_, i) in matches])  # 同列表推导式
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # 计算视角变换矩阵
            # 使用OpenCV的 cv2.findHomography 函数来估计两幅图像之间的透视变换矩阵（Homography）。这行代码的各个参数和功能：
            # ptsA：这是第一幅图像中的关键点坐标，是一个包含了一组点坐标的Numpy数组。这些关键点通常是在第一幅图像中检测到的。
            # ptsB：这是第二幅图像中的关键点坐标，也是一个Numpy数组。这些关键点是与 ptsA 中的关键点对应的点，在第二幅图像中检测到的。
            # cv2.RANSAC：这是一个估计方法的选择，表示使用随机抽样一致性（RANSAC）算法来估计透视变换矩阵。RANSAC算法可以在存在离群点（不符合模型的点）的情况下，估计出较为准确的变换矩阵。
            # reprojThresh：这是一个重投影误差的阈值，用于筛选掉不满足误差要求的点。如果某个关键点的重投影误差大于 reprojThresh = 4.0，则该点可能被认为是离群点，不会被用于估计变换矩阵。
            # (H, status)：这是函数的返回值，是一个元组，包含两个元素：
            #   H：透视变换矩阵，它可以将第一幅图像中的关键点映射到第二幅图像中。
            #   status：这是一个与输入关键点数目相同的Numpy数组，用于指示哪些关键点被认为是内点（有效匹配）以及哪些是外点（离群点）。
            #   status 中的每个元素都是一个布尔值，表示对应的关键点是否被认为是内点。如果关键点是内点，对应的元素为True；如果是外点，对应的元素为False。
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

            # 返回结果
            return (matches, H, status)

        # 如果匹配对小于4时，返回None
        return None

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # 初始化可视化图片，将A、B图左右连接到一起
        # 获取图像 imageA 和 imageB 的高度和宽度信息。shape 属性返回一个元组，包含图像的高度和宽度。
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        # 这行代码创建了一个全零的Numpy数组 vis，用于存储拼接后的图像。
        # 数组的形状是 (max(hA, hB), wA + wB, 3)，表示高度取两幅图像高度的最大值，宽度取两幅图像宽度之和，通道数为3（RGB颜色通道）。
        # dtype="uint8" 指定了数组的数据类型为无符号8位整数，用于存储像素值。
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        # 将图像 imageA 复制到 vis 数组的左侧，也就是 (0:hA, 0:wA) 区域。这个区域的高度和宽度与 imageA 相同，因此 imageA 将被粘贴到 vis 的左侧。
        vis[0:hA, 0:wA] = imageA
        # 将图像 imageB 复制到 vis 数组的右侧，也就是 (0:hB, wA:) 区域。这个区域的高度和宽度与 imageB 相同，因此 imageB 将被粘贴到 vis 的右侧。
        vis[0:hB, wA:] = imageB

        # 联合遍历，画出匹配对
        # matches 是一个包含了匹配对的索引的列表，每个匹配对包含了两个索引：
        #   第一个索引表示第一幅图像（A图像）中的关键点索引，而第二个索引表示第二幅图像（B图像）中的关键点索引。
        # status：这是一个与输入关键点数目相同的Numpy数组，用于指示哪些关键点被认为是内点（有效匹配）以及哪些是外点（离群点）。
        #   status 中的每个元素都是一个布尔值，表示对应的关键点是否被认为是内点。如果关键点是内点，对应的元素为True；如果是外点，对应的元素为False。
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # 当点对匹配成功时，画到可视化图上
            if s == 1:  # 如果关键点是内点，对应的元素为True的情况
                # 画出匹配对
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))  # A的关键点
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))  # 复制后B的关键点
                # 通过这行代码，可以在图像 vis 上绘制一条从 ptA 到 ptB 的绿色线段，线段的粗细为1个像素。
                # 这是该函数的关键核心
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # 返回可视化结果
        return vis
