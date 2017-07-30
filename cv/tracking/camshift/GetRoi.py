# -*- coding:utf-8 -*-


import cv2
import imutils


ix, iy = -1, -1
w, h = -1, -1
count = 0

cv2.CAP_PROP_FRAME_COUNT
def GetFrame(video):
    """
    GetFrame 做视频帧读取，同时转换视频帧为HSV图
    :param video: 视频文件路径
    :return:
    """
    frames_hsv = []
    frames_gray = []
    frames_color = []
    camera = cv2.VideoCapture(video)  # 读取视频
    count = 1
    while True:
        res, frame = camera.read()
        if not res:
            print(count)
            break
        frame = imutils.resize(frame, width=500)
        frames_color.append(frame)
        count = count + 1
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)    # 转HSV图
        hsv[hsv[:, :, 1] > 150] = 0   # 重点小技巧，结合了图片的S通道特征，出去和中间小球颜色特征相似的跑道，从而是中间小球跟踪的关键步骤
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转GRAY图

        frames_hsv.append(hsv)
        frames_gray.append(gray)
    print('done_get_frame')
    camera.release()
    return frames_hsv, frames_gray, frames_color

def Get_ix_iy_w_h(event, x, y, flags, param):
    global ix, iy, w, h, count
    if event == cv2.EVENT_LBUTTONDOWN:
        if count == 0:
            ix, iy = x, y
            count = count + 1
        else:
            w = x - ix
            h = y - iy
            count = 0

def getRoi(frame):
    """
    getRoi 获取目标的位置，大小，使用方法是运行该程序，然后在目标的左上角和右下角各点击一下，然后键盘输入q，即在输出信息中获取target的位置信息
    :param frame: 输入一帧HSV图
    :return:
    """
    global ix, iy, w, h
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', Get_ix_iy_w_h)
    while True:
        cv2.imshow('image', frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
    return iy, ix, h, w  # iy才是真正的ix，ix是iy，同时有h为实际的高，w为实际的宽

if __name__ == '__main__':
    frames, frames_gray, frames_color = GetFrame(r'G:\WindowsDownload\OpenCV\opencv\sources\samples\python\tracking_test\Ball_1.avi')
    getRoi(frames[0])
    print(frames[0].shape)

    print('target:', iy, ix, w, h)
    cv2.rectangle(frames[1], (ix, iy), (ix + w, iy + h), (255, 0, 0), 2)
    cv2.imshow('tt', frames[1])

    cv2.imshow('ttt', frames[1][iy:iy+h, ix:ix+w])
    cv2.waitKey(0)
