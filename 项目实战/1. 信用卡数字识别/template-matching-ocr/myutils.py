import cv2


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


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized
