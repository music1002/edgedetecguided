import cv2
import numpy as np
import matplotlib.pyplot as plt

def convert_colour_to_gray(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale



def canny_edge(or_img):
    img=convert_colour_to_gray(or_img)
    edges= cv2.Canny(img,100,200)
    cv2.imshow(" ",edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    cv2.waitKey(1)
    return edges


   
def simple_thresh(inp,thresh,maxval):
    img=convert_colour_to_gray(or_img)
    ret,thresh = cv2.threshold(img,thresh,maxval,cv2.THRESH_BINARY)
    return ret,thresh

def adaptive_thresh(inp):
    img=convert_colour_to_gray(inp)
    img = cv2.medianBlur(img,5)
    th = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
    cv2.imshow(" ",th)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    cv2.waitKey(1)
    return th

def color_the_edge(image):
    _, mask = simple_thresh(image,140,255)
    img=convert_colour_to_gray(image)
    im_thresh_gray = cv2.bitwise_and(img, mask)
    mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # 3 channel mask
    im_thresh_color = cv2.bitwise_and(image, mask3)
    cv2.imshow(" ",im_thresh_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    cv2.waitKey(1)
    
    



or_img= cv2.imread(r"/Users/shreyashrivastava/Desktop/Butterfly.png",cv2.IMREAD_COLOR)
im_gray = cv2.cvtColor(or_img, cv2.COLOR_BGR2GRAY)
#def canny_edge(image):
#canny_edge(or_img)
adaptive_thresh(or_img)






















































'''def plot_both(or_img):
    edge=canny_edge(or_img)
    plt.subplot(121),plt.imshow(or_img,cmap="gray")
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edge,cmap="gray")
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
