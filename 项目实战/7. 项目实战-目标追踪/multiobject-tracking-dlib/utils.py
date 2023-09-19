import datetime


# 这个类是一个用于测量帧率（Frames Per Second，FPS）的辅助类。它用于测量处理图像或视频所花费的时间，并计算出每秒处理的帧数，以评估性能。
#   start(): 开始计时，记录开始时间。
#   stop(): 停止计时，记录结束时间。
#   update(): 更新帧计数，用于在时间间隔内记录处理的帧数。
#   elapsed(): 计算从开始时间到结束时间的总时间，以秒为单位。
#   fps(): 计算帧率，即在时间间隔内处理的帧数除以总时间。
class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0

    # 开始计时，记录开始时间。
    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self

    # 停止计时，记录结束时间。
    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()

    # 更新帧计数，用于在时间间隔内记录处理的帧数。
    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1

    # 计算从开始时间到结束时间的总时间，以秒为单位。
    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()

    # 计算帧率，即在时间间隔内处理的帧数除以总时间。
    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()
