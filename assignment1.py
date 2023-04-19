import cv2
from matplotlib import pyplot as plt

# 読み込み
a1 = cv2.imread('1.png')
a2 = cv2.imread('2.png')

# 表示
cv2.imshow('a1', a1)
cv2.imshow('a2', a2)
cv2.waitKey(0)  # 何かキーを押すまで待機

# 縮小、拡大
b1 = cv2.resize(a1, dsize=(0, 0), fx=0.5, fy=0.5)  # 0.5倍
b2 = cv2.resize(a2, dsize=(0, 0), fx=1.5, fy=1.5)  # 1.5倍
cv2.imwrite('b1.png', b1)  # 保存
cv2.imwrite('b2.png', b2)

# 回転
c1 = cv2.rotate(a1, cv2.ROTATE_90_CLOCKWISE)  # 90度回転
height, width = a2.shape[:2]  # 画像サイズ取得
mat_c2 = cv2.getRotationMatrix2D((width/2, height/2), 45, 1.0)  # 回転行列取得, 45度回転
c2 = cv2.warpAffine(a2, mat_c2, (width, height))  # 回転
cv2.imwrite('c1.png', c1)  # 保存
cv2.imwrite('c2.png', c2)

# 二値化
d0 = cv2.cvtColor(a1, cv2.COLOR_BGR2GRAY)
d1_ret, d1_thresh = cv2.threshold(d0, 127, 255, cv2.THRESH_BINARY)
cv2.imwrite('d1.png', d1_thresh)

# 差分
e = cv2.absdiff(a1, a2)
cv2.imwrite('e.png', e)
# f = cv2.absdiff(a2, a1)
# cv2.imwrite('f.png', f)  # 同じ画像になる

# 特徴量
# グレースケール変換
gray1 = cv2.cvtColor(a1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(a2, cv2.COLOR_BGR2GRAY)
# SIFTで特徴量検出
sift = cv2.xfeatures2d.SIFT_create()  # SIFT特徴量検出器
kp1, des1 = sift.detectAndCompute(a1, None)  # 特徴量検出と特徴量ベクトル計算
kp2, des2 = sift.detectAndCompute(a2, None)
# 特徴量の描画
f1 = cv2.drawKeypoints(a1, kp1, None)
f2 = cv2.drawKeypoints(a2, kp2, None)
cv2.imwrite('f1.png', f1)
cv2.imwrite('f2.png', f2)

# ヒストグラム
img = cv2.imread('2.png')
color = ('b','g','r')
for i,col in enumerate(color):  # 色ごとにヒストグラムを計算
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    # プロット
    plt.plot(histr,color = col)  
    plt.xlim([0,256])
    # 軸ラベル
    plt.xlabel('Intensity')
    plt.ylabel('Number of pixels')

plt.savefig('my_plot.png')  # 保存

