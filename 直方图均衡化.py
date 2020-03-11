import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread(r'D:\OpenCV\Course\Source\test1-3.jpg',0)

def img_show(Title,img):
    cv2.namedWindow(Title,cv2.WINDOW_AUTOSIZE)
    cv2.imshow(Title,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Hist_Normalization(img,Gray_Level):
    hist = np.histogram(img,Gray_Level,[0,Gray_Level], normed= True) #得到归一化直方图
    Pixel_Statistic,Gray_Distration = hist #得到像素概率统计和灰度级分布情况
    Gray_Distra_Normalizaton = Gray_Distration[:-1]/Gray_Distration[:-1].max()#灰度级归一化
    Pixel_cumsum = Pixel_Statistic.cumsum()   #累计分布函数

    #此处应为理就近近似计算论距离 但方便处理直接倍乘最大灰度并整数近似
    Pixel_approxi = Gray_Level*Pixel_cumsum #逆归一化
    Pixel_approxi = Pixel_approxi.astype(int) #化为整型，形成新的灰度分布
    
    #对新灰度级进行去零处理，方便后续映射
    Flag = 0    #设立循环标志
    for i in range(0,Gray_Level):   
        if Pixel_approxi[i] == 0 and Flag == 0:
            for j in range(i,Gray_Level): #将指针摆到新灰度级第一个不为零灰度处
                if Pixel_approxi[j] != 0:   #找到不为零灰度并跳出一次循环
                    i = 0      #准备进行二次循环，目的为赋值
                    Pointer = j     #寄存指针
                    Flag = 1   #已循环一次
                    break
                else:j = j+1
        if Pixel_approxi[i] == 0 and Flag == 1 and i< Pointer:
            Pixel_approxi[i] = Pixel_approxi[Pointer]
        if i >= Pointer:    #不处理非零灰度
            break
    #以灰度级为桥梁作映射 形成新图片
    LogicArray = [] #用于新灰度级指派的逻辑列表
    Flag = 0
    for i in range(Gray_Level):   #以灰度级为索引遍历
        LogicArray.append((img == i))   #寻找图像中灰度级为i的点的位置
    for i in range(Gray_Level):
        if i ==255:
            Pixel_approxi[i] = 255
        img[LogicArray [i]] = Pixel_approxi[i] #将灰度级为i的点重新赋值为新像素
    
    Flag = 0
    #单循环代替双循环
    # i = 0
    # while i < Gray_Level:   #以灰度级为索引遍历
    #     if Flag == 0:
    #         LogicArray.append((img == i))   #寻找图像中灰度级为i的点的位置
    #         if i == Gray_Level-1 :      #一次循环结束
    #             Flag = 1
    #             i = 0
    #     if Flag == 1:
    #         img[LogicArray [i]] = Pixel_approxi[i] #将灰度级为i的点重新赋值为新像素
    #     i = i+1   
    return img


img_copy = img.copy()
img_new_1 = Hist_Normalization(img_copy,256)
#img_new_1 = cv2.equalizeHist(img)
img_new = np.hstack((img,img_new_1))
# plt.subplot(221),plt.imshow(img,cmap='gray')
# plt.subplot(222),plt.imshow(img_new_1, cmap='gray')
plt.subplot(121),plt.hist(img.flatten(),256,[0,256],color = 'r')
plt.title('Intput Image Histogarm')
plt.subplot(122),plt.hist(img_new_1.flatten(),256,[0,256],color = 'r')
plt.title('output Image Histogarm')
plt.show()   
img_show('PIc',img_new)

