
import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread(r'D:\OpenCV\Course\Source\test2.jpg',0)

def img_show(Title,img):
    cv2.namedWindow(Title,cv2.WINDOW_AUTOSIZE)
    cv2.imshow(Title,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Magnitude_Spectrum(img_fft):    #幅度谱的幅度计算公式
    return 20*np.log(np.abs(img_fft))

def filter_constructer(filter_size,filter_type):
    if filter_size == 'defalse':  #默认滤波器大小
        filter_size = (3,3)
    Rows, Cols = filter_size   #获得滤波器的长宽
    M, N = np.int(Rows/2),np.int(Cols/2)#获取滤波器中心坐标
    if filter_type == 'defalse':
        Filter = np.ones(filter_size) #取小区域
    if filter_type == 'Means':  #均值滤波器
        temp = np.ones(filter_size)
        Filter = temp/np.sum(temp)
    if filter_type == 'Linear_Means':   #平滑线性滤波器
        temp = np.ones(filter_size)
        temp.itemset((M,N),4)
        temp.itemset((M-1,N),2)
        temp.itemset((M+1,N),2)
        temp.itemset((M,N-1),2)
        temp.itemset((M,N+1),2)
        Filter = temp/np.sum(temp)
    if filter_type == 'Laplace_4':  #拉普拉斯4近邻滤波器
        temp = np.zeros(filter_size)
        temp.itemset((M,N),5)
        temp.itemset((M-1,N),-1)
        temp.itemset((M+1,N),-1)
        temp.itemset((M,N-1),-1)
        temp.itemset((M,N+1),-1)
        Filter = temp
    if filter_type == 'Laplace_8':  #拉普拉斯8近邻滤波器
        temp = (-1)*np.ones(filter_size)
        temp.itemset((M,N),9)
        Filter = temp

    return Filter

#img_show('pic',img)

# #卷积/矩阵掩模
# filter_size = (3,3)
# Filter = filter_constructer(filter_size,'Laplace_4')    #拉普拉斯算子锐化
# dst = cv2.filter2D(img,-1,Filter)   #卷积运算


img_fft = np.fft.fft2(img)  #二维傅里叶变换
img_fft_shift = np.fft.fftshift(img_fft)    #中心变换
img_mg = Magnitude_Spectrum(img_fft_shift)  #幅度谱

plt.subplot(221),plt.imshow(img, cmap='gray') #子图一: 原图
plt.title('Input Image'), plt.xticks([]), plt.yticks([]) 
plt.subplot(222),plt.imshow(img_mg, cmap='gray')   #子图二：幅度谱(频谱)
plt.title('Magnitude Spectrum 1'), plt.xticks([]), plt.yticks([]) 
#plt.show()


Rows ,Cols = img_fft.shape #获得图片尺寸
M,N = np.int(Rows/2),np.int(Cols/2)   #获取中心坐标
Core_Size = 80
#Core = M-Core_Size :M+Core_Size, N-Core_Size:N+Core_Size #构造方形掩模

#高通滤波
img_fft_shift[M-Core_Size :M+Core_Size, N-Core_Size:N+Core_Size] = 0 #进行掩模滤波，高通

# #低通滤波
# img_LPF = np.zeros(img.shape)
# img_LPF[M-Core_Size :M+Core_Size, N-Core_Size:N+Core_Size] = 1
# img_fft_shift = img_fft_shift*img_LPF


img_mg_new = Magnitude_Spectrum(img_fft_shift)
img_fft_new = np.fft.ifftshift(img_fft_shift) #逆中心变换
img_new = np.abs(np.fft.ifft2(img_fft_new)) #逆二维傅里叶变换
plt.subplot(223),plt.imshow(img_new, cmap='gray') #子图一: 原图
plt.title('output Image'), plt.xticks([]), plt.yticks([]) 
plt.subplot(224),plt.imshow(img_mg_new, cmap='gray')   #子图二：幅度谱(频谱)
plt.title('Magnitude Spectrum 2'), plt.xticks([]), plt.yticks([]) 
plt.show()






