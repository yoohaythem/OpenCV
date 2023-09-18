import os


image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_images(basePath, contains=None):
    # 返回有效的文件集合
    return list_files(basePath, validExts=image_types, contains=contains)

# basePath：要遍历的基目录路径。
# validExts：一个可选的文件扩展名列表，用于筛选特定类型的文件。如果不提供或将其设置为 None，则不会限制文件类型。
# contains：一个可选的字符串，用于检查文件名是否包含指定的子字符串。如果提供，只有文件名中包含这个子字符串的文件才会被返回。
def list_files(basePath, validExts=None, contains=None):
    # 遍历目录结构
    # rootDir：字符串，表示当前正在遍历的目录的路径;
    # dirNames：列表，包含当前目录中的子目录名称;
    # filenames：列表，包含当前目录中的文件名称。
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # 遍历当前目录中的文件
        for filename in filenames:
            # 如果提供了contains字符串，并且文件名中没有包含该字符串，则忽略。
            if contains is not None and filename.find(contains) == -1:
                continue

            # 用来获取当前文件的扩展名（文件后缀）
            # filename.rfind(".") 会从文件名的右边开始查找，找到文件名中最后一个点的位置，也就是扩展名的起始位置。
            # filename[filename.rfind("."):].lower() 使用切片操作从文件名中提取扩展名部分，然后将其转换为小写字母。
            ext = filename[filename.rfind("."):].lower()

            # 如果不筛选文件后缀类型，或者以validExts中的某种类型结尾
            if validExts is None or ext.endswith(validExts):
                # 将 rootDir（目录路径）和 filename（文件名）合并成一个完整的文件路径。
                # 这是因为在不同的操作系统中，文件路径的分隔符可能不同（例如，在Windows中是反斜杠 \，在Linux和Unix中是正斜杠 /），使用 os.path.join 可以确保生成的路径是符合当前操作系统的规范的。
                imagePath = os.path.join(rootDir, filename)
                # 将构建好的文件路径 imagePath 通过 yield 返回。
                # 这意味着这个函数是一个生成器函数，每次调用它时会生成一个文件路径，并在下一次调用时继续生成下一个文件路径。这个生成器可用于遍历指定目录下的所有符合条件的文件。
                # 简单的理解，yield有些类似于给函数打上断点，每次调用函数的行为就像是debug到下一个断点，然后停止。return则是直接run到函数结束。
                yield imagePath

'''
yield 和 return 都用于从函数中返回值，但它们之间有重要的区别：

1. 返回值类型：
return：用于返回一个值，并终止函数的执行。一旦函数中执行到 return 语句，函数就会立即结束，不再执行后续代码。
yield：用于定义生成器函数。生成器函数返回一个生成器对象，它可以用于生成一个序列的值。执行生成器函数时，函数的状态会被保存，以便下次继续执行。每次调用生成器的 next() 方法或迭代生成器时，函数会从上次暂停的地方继续执行，直到遇到下一个 yield 语句或函数结束。

2. 用途：
return 适用于普通函数，用于返回最终的计算结果。
yield 用于生成器函数，通常用于生成一个序列的值，例如遍历大型数据集时逐个产生数据，或实现惰性计算。

3. 执行次数：
return 只执行一次，返回值后函数终止。
yield 可以被多次执行，每次执行会从上次 yield 暂停的地方继续执行，直到生成器中没有更多的值可生成，然后会引发 StopIteration 异常。
'''

