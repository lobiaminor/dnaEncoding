import os
import sr_decoder
import sr_encoder
import pywt
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt


def main():
    # Read image
    img_src = "lena.jpg"
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, img_src)
    image = img.imread(img_src)

    x_axis, y_axis = image.shape
    # http://pywavelets.readthedocs.io/en/latest/ref/2d-dwt-and-idwt.html?highlight=haar#d-multilevel-decomposition-using-wavedec2
    new_image = np.zeros_like(image)
    new_image[:] = image
    for i in range(0, 1):
        midx = int(x_axis/(np.power(2,i)))
        midy = int(y_axis/(np.power(2,i)))

        LL = np.zeros((midx, midy)).astype(int)
        LL[:] = new_image[0:midx, 0:midy].astype(int)

        print(LL)

        new_image[0:midx, 0:midy] =  iwt2(image)
        new_image = new_image.astype(np.uint64)
        print(type(new_image[0,0]))
        print(new_image)
    # Show the image
    plt.imshow(new_image)
    plt.gray()
    plt.show()


def iwt(array):
    output = np.zeros_like(array)
    nx, ny = array.shape
    x = nx // 2
    for j in range(ny):
        output[0:x,j] = (array[0::2,j] + array[1::2,j])//2
        output[x:nx,j] = array[0::2,j] - array[1::2,j]
    return output

def iiwt(array):
    output = np.zeros_like(array)
    nx, ny = array.shape
    x = nx // 2
    for j in range(ny):
        output[0::2,j] = array[0:x,j] + (array[x:nx,j] + 1)//2
        output[1::2,j] = output[0::2,j] - array[x:nx,j]
    return output

def iwt2(array):
    return iwt(iwt(array.astype(int)).T).T

def iiwt2(array):
    return iiwt(iiwt(array.astype(int).T).T)


if __name__ == "__main__":
    main()


