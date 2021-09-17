
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
 

def convolution(image, kernel, average=False, verbose=False):

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
    if verbose:
        plt.imshow(image, cmap='gray')
        plt.title("Imagen sin color")
        plt.show()
 
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape
 
    output = np.zeros(image.shape)
 
    #Obtiene el padding
    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)
 
    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
 
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image
 
    if verbose:
        plt.imshow(padded_image, cmap='gray')
        plt.title("Imagen con padding")
        plt.show()
 
    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            if average:
                output[row, col] /= kernel.shape[0] * kernel.shape[1]
 
    if verbose:
        plt.imshow(output, cmap='gray')
        plt.title("Imagen output usando {}X{} Kernel".format(kernel_row, kernel_col))
        plt.show()
 
    return output


filtro = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])

convolution(image, filtro, verbose = True)

