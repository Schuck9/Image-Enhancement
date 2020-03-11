import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread(r'D:\OpenCV\Course\Source\test3.jpg',0)

def img_show(Title,img):
    cv2.namedWindow(Title,cv2.WINDOW_AUTOSIZE)
    cv2.imshow(Title,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Kernel_Constructor(Ksize):
    return np.ones(Ksize)

def MedianBlur(img,Kernel_Size):
    M,N = img.shape #图片的大小
    Ksize = np.int(Kernel_Size/2) #掩模的一半，用于确定掩模操作的边界
    for i in range(Ksize,M-Ksize):
        for j in range(Ksize,N-Ksize):
            temp = img[i-Ksize:i+Ksize+1,j-Ksize:j+Ksize+1] #抽离滤波区域
            #print(temp)
            temp = temp.flatten()   #变成一维数组
            temp.sort() #排序
            temp_mid = np.int(temp.size/2)
            img.itemset(i,j,temp.item(temp_mid)) #取中间值
            # print(temp)
            # print("第 %d 个滤波区域mid值为:%d\n" % (i+j,temp.item(temp_mid)))
            # print("\n")  
    return img

img1 = img.copy()
img_new = MedianBlur(img1,5)
img_N = np.hstack((img,img_new))
img_show('Pic',img_N)

# B = np.random.randint(0,10,(20,20))
# print(B)
# print('\n')
# C = MedianBlur(B,3)
# print(C)

#看图像像素直方图
# plt.hist(img.flatten(),256,[0,256],color = 'r')
# plt.show()    
# img_new = cv2.medianBlur(img,3)
# plt.hist(img_new.flatten(),256,[0,256],color = 'r')
# plt.show()   
# # img_new_1 = cv2.GaussianBlur(img,(3,3),0)
# # img_new_2 = cv2.GaussianBlur(img,(3,3),5)
# img_new = np.hstack((img,img_new))
# img_show('pic',img_new)