import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
 

def conv_helper(fragment, kernel):
    
    f_row, f_col = fragment.shape
    k_row, k_col = kernel.shape 
    result = 0.0

    for row in range(f_row):
        for col in range(f_col):
            result += fragment[row,col] *  kernel[row,col]

    return result


def convolution(image, kernel):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image_row, image_col = image.shape #asigna alto y ancho de la imagen 
    kernel_row, kernel_col = kernel.shape #asigna alto y ancho del filtro
   
    output = np.zeros(image.shape)
   
    for row in range(image_row):
        for col in range(image_col):
                output[row, col] = conv_helper(
                                    image[row:row + kernel_row, 
                                    col:col + kernel_col],kernel)
             
    plt.imshow(output, cmap='gray')
    plt.title("Imagen output usando {}X{} Kernel".format(kernel_row, kernel_col))
    plt.show()
 
    return output


filtro = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])

convolution(image, filtro)

