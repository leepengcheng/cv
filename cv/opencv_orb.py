#coding:utf-8
"OpenCV ORB特征提取与匹配"
import cv2
import numpy as np


def filter_matches_1(kpts1, kpts2, matches, ratio = 0.75):
    "方法1：当最近的2个点的距离比小于0.75时，认为匹配有效"
    mkpts1, mkpts2,good_matches = [], [],[]
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkpts1.append( kpts1[m.queryIdx] )
            mkpts2.append( kpts2[m.trainIdx] )
            good_matches.append(m)
    p1 = np.float32([kp.pt for kp in mkpts1])
    p2 = np.float32([kp.pt for kp in mkpts2])
    kp_pairs = zip(mkpts1, mkpts2)
    return p1, p2, kp_pairs,good_matches

def filter_matches_2(matches,threhold=30):
    "方法2：当匹配点距离小于最小距离的2倍或threhold时，认为匹配有效"
    dists=[x.distance for x in matches] #匹配点的距离
    min_dist,max_dist=min(dists),max(dists) #匹配点的最小/最大值
    good_matches=[]
    for m in matches:
        if m.distance<=max((2*min_dist,threhold)):
            good_matches.append(m)
    return min_dist,max_dist,good_matches


cv2.ocl.setUseOpenCL(False) #关闭OPENCL
NFEATURES=500  #抽取的特征数

img1=cv2.imread(r"F:\slambook-master\ch7\1.png")
img2=cv2.imread(r"F:\slambook-master\ch7\2.png")
detector=cv2.ORB_create(NFEATURES)  
kpts1, desc1 = detector.detectAndCompute(img1, None)
kpts2, desc2 = detector.detectAndCompute(img2, None)
img1_kps=cv2.drawKeypoints(img1,kpts1, None, flags=2,color=(255,0,0)) #绘制特征点,flag控制圆圈大小
img2_kps=cv2.drawKeypoints(img2,kpts2, None, flags=2,color=(255,0,0)) #绘制特征点,flag控制圆圈大小

matcher=cv2.DescriptorMatcher_create("BruteForce-Hamming") #汉明距离
############匹配方法1:KNN#################
match_1=matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2) # K近邻匹配最近的2个点
p1, p2, kp_pairs,good_matches_1= filter_matches_1(kpts1, kpts2, match_1)
#############匹配方法2:自定义################
match_2=matcher.match(desc1,desc2,None) #匹配最近的点
min_dist,max_dist,good_matches_2=filter_matches_2(match_2)

img_goodmatch_1=cv2.drawMatches(img1,kpts1,img2,kpts2,good_matches_1,None)   #绘制正确的匹配点
img_goodmatch_2=cv2.drawMatches(img1,kpts1,img2,kpts2,good_matches_2,None)   #绘制正确的匹配点

cv2.imshow('img-match-1', img_goodmatch_1)
cv2.imshow('img-match-2', img_goodmatch_2)

cv2.waitKey()
cv2.destroyAllWindows()
