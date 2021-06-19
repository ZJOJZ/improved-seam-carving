from PIL.Image import NORMAL
from seam_carving import SeamCarver
from improveSC import improveSC
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image

def image_resize_without_mask(filename_input, filename_output, new_height, new_width):
    obj = SeamCarver(filename_input, new_height, new_width)
    obj.save_result(filename_output)

def image_resize_without_mask1(filename_input, filename_output, new_height, new_width):
    obj = improveSC(filename_input, new_height, new_width)
    obj.save_result(filename_output)

def image_resize_with_mask(filename_input, filename_output, new_height, new_width, filename_mask):
    obj = SeamCarver(filename_input, new_height, new_width, protect_mask=filename_mask)
    obj.save_result(filename_output)


def object_removal(filename_input, filename_output, filename_mask):
    obj = SeamCarver(filename_input, 0, 0, object_mask=filename_mask)
    obj.save_result(filename_output)



if __name__ == '__main__':
    """
    Put image in in/images folder and protect or object mask in in/masks folder
    Ouput image will be saved to out/images folder with filename_output
    """
    # print("1")
    folder_in = 'in'    #输入文件夹
    folder_out = 'out'  #输出文件夹

    filename_input = 'image1.jpg'   #输入文件名
    filename_output = 'image1_result.png'   #输出文件名
    filename_output1 = 'image1_result1.png'   #输出文件名
    
    filename_mask = 'mask.jpg'
    new_height = 368    #设置新图像的高度
    new_width = 500     #设置新图像的宽度

    input_image = os.path.join(folder_in, filename_input)
    input_mask = os.path.join(folder_in, filename_mask)
    output_image = os.path.join(folder_out, filename_output)
    output_image1 = os.path.join(folder_out, filename_output1)
    in_image = cv2.imread(input_image).astype(np.double)
    '''  
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(in_image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    plt.imshow(gray, cmap='gray')
    plt.show()
    Face = np.zeros(gray.shape)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        in_image = cv2.rectangle(in_image, (x, y), (x+w, y+h), 255, 2)
   '''
    
    '''
    img = cv2.cvtColor(in_image.astype(np.uint8), cv2.COLOR_BGR2RGB)
    img = cv2.GaussianBlur(img,(5,5), 0)
    plt.imshow(img)
    plt.show()
    gray_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_mean = np.mean(gray_lab[:,:,0])
    a_mean = np.mean(gray_lab[:,:,1])
    b_mean = np.mean(gray_lab[:,:,2])
    lab = np.square(gray_lab- np.array([l_mean, a_mean, b_mean]))
    lab = np.sum(lab,axis=2)
    lab = lab/np.max(lab)
    #cv2.normalize(lab,lab,1.0,0.0,cv2.NORM_MINMAX)
    print(lab)
    
    plt.imshow(lab, cmap='gray')
    plt.show()
    #(mean , stddv) = cv2.meanStdDev(in_image)
    #print(mean)
    #print(stddv[0][0])
    
    lenna = cv2.GaussianBlur(in_image, (15, 15), 0)
    canny = cv2.Canny(lenna.astype(np.uint8), 100, 200)
    plt.imshow(canny, cmap='gray')
    plt.show()
    Lines = cv2.HoughLinesP(canny,1,np.pi/180,35,minLineLength=10,maxLineGap=5)
    #print(Lines)
    #print(canny.shape)
    Line = np.zeros(canny.shape)
    
    for o in Lines:
        #print(o)
        cv2.line(Line, (o[0][0], o[0][1]), (o[0][2], o[0][3]), 255, 2)

    plt.imshow(Line, cmap='gray')
    plt.show()
    '''
    image_resize_without_mask(input_image, output_image, new_height, new_width)
    image_resize_without_mask1(input_image, output_image1, new_height, new_width)
    
    #image_resize_with_mask(input_image, output_image, new_height, new_width, input_mask)
    #object_removal(input_image, output_image, input_mask)








