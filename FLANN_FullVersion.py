import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import math
MIN_MATCH_COUNT = 10

def getMatchNum(matches,ratio):
    matchesMask=[[0,0] for i in range(len(matches))]
    matchNum=0
    for i,(m,n) in enumerate(matches):
        if m.distance<ratio*n.distance: #将距离比率小于ratio的匹配点删选出来
            matchesMask[i]=[1,0]
            matchNum+=1
    return (matchNum,matchesMask)

path='C:/Users/xxxxxx/Desktop/database/'
queryPath=path + 'query/' #图库路径
samplePath=path+'given/book.jpg' #样本图片
comparisonImageList=[] 

sift = cv2.xfeatures2d.SIFT_create()
FLANN_INDEX_KDTREE=0
indexParams=dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
searchParams=dict(checks=50)
flann=cv2.FlannBasedMatcher(indexParams,searchParams)

sampleImage=cv2.imread(samplePath,0)
kp1, des1 = sift.detectAndCompute(sampleImage, None) 
for parent,dirnames,filenames in os.walk(queryPath):
    for p in filenames:
        p=queryPath+p
        queryImage=cv2.imread(p,0)
        kp2, des2 = sift.detectAndCompute(queryImage, None)
        print('000')
        matches=flann.knnMatch(des1,des2,k=2) 
        print('111')
        (matchNum,matchesMask)=getMatchNum(matches,0.9) 
        matchRatio=matchNum*100/len(matches)
        drawParams=dict(matchColor=(0,255,0),
                singlePointColor=None,
                matchesMask=matchesMask,
                flags=2)

        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
            h, w = sampleImage.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
        queryImage= cv2.polylines(queryImage, [np.int32(dst)], True, (255, 0, 0), 3, cv2.LINE_AA)
        comparisonImage=cv2.drawMatchesKnn(sampleImage,kp1,queryImage,kp2,matches,None,**drawParams)
        comparisonImageList.append((comparisonImage,matchRatio)) #记录下结果

comparisonImageList.sort(key=lambda x:x[1],reverse=True) #按照匹配度排序
count=len(comparisonImageList)

#绘图显示
figure,ax=plt.subplots()
index = 0
image,ratio=comparisonImageList[index]
ax.imshow(image)
plt.show()