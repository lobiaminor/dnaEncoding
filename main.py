import os
import sr_decoder
import sr_encoder
import pywt
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
import wavelets as wv


def main():
    # Read image
    img_src = "lena.jpg"
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, img_src)
    image = img.imread(filename)

    # http://pywavelets.readthedocs.io/en/latest/ref/2d-dwt-and-idwt.html?highlight=haar#d-multilevel-decomposition-using-wavedec2
    new_image = wv.iwtn(image, 3)

    # Show the image
    plt.imshow(wv.iiwtn(new_image, 3))
    plt.gray()
    plt.show()


if __name__ == "__main__":
    main()

# By rows
# a.flatten()
# By cols
# a.T.flatten()
# To reconstruct
# np.reshape(a, (x,y)) -> with a -1 on the dimension we dont want to enforce
# TODO: Scan through the img and encode it