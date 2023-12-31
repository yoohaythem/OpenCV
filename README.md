##  OpenCV 视频学习笔记完全注释

1. **视频讲解** https://www.bilibili.com/video/BV1PV411774y?from=search&seid=128144269248922245&spm_id_from=333.337.0.0

2. **关于opencv版权** OpenCV 3.4.2之后因专利版权问题移除了SIFT/SURF的相关库，因此在使用较新版本的cv库时会报错，这也就是视频课程一直采用opencv-python 3.4.1.15的原因。但是在2020年3月17日之后（opencv-python 4.3.0以后）一代传奇算法SIFT专利到期了，因此只需更新cv版本即可免费使用。所以本套代码采用了当前最新的opencv-python 4.8.0.76进行修订和注释。

3. **课件资料文件夹** 下资料主要来自于 https://github.com/AccumulateMore/OpenCV ，但对其中内容基于最新的 **opencv-python 4.8.0.76** 版本做出了修订，也增加了部分代码注释、图像输出。例如：26_光流估计.ipynb 等。

4. **项目实战文件夹** 下，一共有10个实战项目，在学习每个项目时，需要以项目文件下的英文子文件夹为工程目录打开。每个项目都进行了细致的评注，并对部分输出结果进行了保存。这不论是对于 **Python高级用法** 的学习、对于 **OpenCV代码的巩固** 、还是对 **整个项目流程** 的理解，都具有很好的学习效果。项目采用了基于opencv-python最新的 **4.8.0.76** 版本实践，代码也基于原版做出了相应的修改。想要原版代码可以在视频讲解下获取。

5. **opencv函数集合.txt** 是对课程用到的cv2相关函数的分模块汇总，在项目学习的过程中如果遇到什么忘记的函数，使用notepad进行搜索定位，可以快速回顾。

