## part1

计算Forward energy 能量函数，计算横向能量函数，并阈值化，横向能量太大不要去除（会引入锯齿）

计算显著图（FT图像显著性算法）

显著图归一化，作为每个能量的系数

引入边缘检测器（Canny edge detector），直线检测器（Hough line detector ），人脸检测器（face detector ）

加入到之前的能量图中，权重：人脸>直线>边缘>能量

去除每一条seam时，不是直接去除，而是与周围的像素进行平均

在上述操作完成后的图上进行SC与Scaling结合的方法去除

每去除一个seam，就使用scaling快速到目标大小，然后用图像距离度量函数来计算距离，取最小的距离为解。

距离度量函数采用Optimized Image Resizing Using Seam Carving and Scaling中所定义的

#### FT 图像显著性算法

FT image saliency algorithm

高斯滤波 去除噪声

转化为LAB颜色空间

LAB颜色空间：

Lab是由一个亮度通道（channel）和两个颜色通道组成的。在Lab颜色空间中，每个颜色用L、a、b三个数字表示，各个分量的含义是这样的： 

- **L\***代表**亮度** 
- **a\***代表**从绿色到红色**的分量 
- **b\***代表**从蓝色到黄色**的分量

Lab是基于**人对颜色的感觉**来设计的，更具体地说，它是**感知均匀**（**perceptual uniform**）的。意思是，如果数字（即前面提到的L、a、b这三个数）变化的幅度一样，那么它给人带来视觉上的变化幅度也差不多。

对转换后的图像**imglab** 的L,A,B三个通道的图像分别取均值得到lm,am,bm。 

对LAB图像和均值图像求欧式距离，并三个通道相加

最后归一到[0,255]得到显著图

#### Canny边缘检测

1、高斯平滑

2、计算梯度幅度和方向

3、对梯度幅值进行非极大值抑制

寻找像素点局部最大值，将非极大值点所对应的灰度值置为0(消除伪边缘)

在每一点上，领域中心 x 与沿着其对应的梯度方向的两个像素相比，若中心像素为最大值，则保留，否则中心置0，这样可以抑制非极大值，保留局部梯度最大的点，以得到细化的边缘。

4、用双阈值算法检测和连接边缘

将小于低阈值的点抛弃，赋0；

将大于高阈值的点立即标记（这些点为确定边缘点），赋1或255；（导致边缘不闭合）

将小于高阈值，大于低阈值的点使用8连通区域确定（即：只有与TH像素连接时才会被接受，成为边缘点，赋 1或255）

#### Hough直线检测

x-y空间变换成k-b空间

x-y空间中的两点所代表的直线为k-b空间中的一个点，而x-y空间中的点对应于k-b空间中的直线

如果x-y图像空间中有很多点在k-b空间中相交于一点，那么这个交点就是我们要检测的直线

基于Canny边缘检测

#### 人脸检测

直接用网上已经学习好的模型



权重：人脸>直线>边缘>能量图/显著图